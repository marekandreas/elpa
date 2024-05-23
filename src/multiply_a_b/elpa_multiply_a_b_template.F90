!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), formerly known as
!      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
!    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!      Informatik,
!    - Technische Universität München, Lehrstuhl für Informatik mit
!      Schwerpunkt Wissenschaftliches Rechnen ,
!    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!      and
!    - IBM Deutschland GmbH
!
!    This particular source code file contains additions, changes and
!    enhancements authored by Intel Corporation which is not part of
!    the ELPA consortium.
!
!    More information can be found here:
!    http://elpa.mpcdf.mpg.de/
!
!    ELPA is free software: you can redistribute it and/or modify
!    it under the terms of the version 3 of the license of the
!    GNU Lesser General Public License as published by the Free
!    Software Foundation.
!
!    ELPA is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public License
!    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
!
!    ELPA reflects a substantial effort on the part of the original
!    ELPA consortium, and we ask you to respect the spirit of the
!    license that we chose: i.e., please contribute any changes you
!    may have back to the original ELPA library distribution, and keep
!    any derivatives of ELPA under the same license that we chose for
!    the original distribution, the GNU Lesser General Public License.
!
!
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
! Author: A. Marek, MPCDF

#include "../general/sanity.F90"
#include "../general/error_checking.inc"

! PETERDEBUG
!#define OLD_GENERIC_GRID
!#define SQUARE_GRID
!#define NONSQUARE_GRID

#undef USE_CCL_MULTIPLY
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
#define USE_CCL_MULTIPLY
#endif

  use elpa1_compute
  use elpa_mpi
  use precision
  use elpa_abstract_impl
  use, intrinsic :: iso_c_binding
  use elpa_gpu
  use mod_check_for_gpu
  use elpa_blas_interfaces
  use ELPA_utilities, only : local_index, greatest_common_divisor, check_deallocate_f, check_dealloc_gpu_f, &
                             check_host_dealloc_gpu_f, check_alloc_gpu_f, check_host_alloc_gpu_f, &
                             check_host_unregister_gpu_f, check_memcpy_gpu_f, check_allocate_f, &
                             check_host_register_gpu_f, check_alloc, error_unit
  use mod_query_gpu_usage
#ifdef WITH_GPU_STREAMS
  use elpa_gpu_util
#endif
#ifdef WITH_NVIDIA_GPU_VERSION
  use cuda_functions ! for NVTX labels
#endif
#if defined(USE_CCL_MULTIPLY)
  use elpa_ccl_gpu
#endif
  use multiply_a_b_gpu
  implicit none

#include "../../src/general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout)   :: obj

  character*1                                  :: uplo_a, uplo_c, trans_a, trans_b

  integer(kind=ik), intent(in)                 :: ldb, ldbCols, ldc, ldcCols
  integer(kind=ik)                             :: na, ncb
#ifndef DEVICE_POINTER
!forget device pointer case for the moment implement later
#ifdef USE_ASSUMED_SIZE
  MATH_DATATYPE(kind=rck)                      :: a(obj%local_nrows,*), b(ldb,*), c(ldc,*)
#else
  MATH_DATATYPE(kind=rck)                      :: a(obj%local_nrows,obj%local_ncols), b(ldb,ldbCols), c(ldc,ldcCols)
#endif
#else /* DEVICE_POINTER */
  ! dummy variables
  MATH_DATATYPE(kind=rck), allocatable         :: a(:,:), b(:,:), c(:,:)
  type(c_ptr)                                  :: aDev, bDev, cDev
#endif /* DEVICE_POINTER */
  MATH_DATATYPE(kind=rck), allocatable         :: at_full(:,:), bt_full(:,:) ! PETERDEBUG: needed for TT case

  integer(kind=ik)                             :: my_prow, my_pcol, np_rows, np_cols, myid
  integer(kind=ik)                             :: my_pdir, np_dirs ! PETERDEBUG_NEW
  integer(kind=MPI_KIND)                       :: my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI
  integer(kind=MPI_KIND)                       :: mpierr, myidMPI
  integer(kind=ik)                             :: l_cols, l_rows, l_rows_np
  integer(kind=ik)                             :: l_rows_max, l_cols_max, l_rows_min, l_cols_min, nstor_block, nstor_block_cut ! PETERDEBUG
  integer(kind=ik)                             :: m, n, k, k_a, k_b ! PETERDEBUG_NEW
  integer(kind=ik)                             :: np, nb, nblk_mult, lrs, lre, lcs, lce
  integer(kind=ik)                             :: np_row_curr, n_blocks_loc_x, n_blocks_loc_y, I_block_gl, J_block_gl, &
                                                  i_block_loc, j_block_loc ! PETERDEBUG
  integer(kind=ik)                             :: gcol_min, gcol, goff
  integer(kind=ik)                             :: nstor, nr_done, noff, np_bc, n_aux_bc, nvals
  integer(kind=ik)                             :: np_br, noff_n ! PETERDEBUG
  integer(kind=ik)                             :: np_t ! PETERDEBUG_NEW -> instead of np_bc?
  integer(kind=ik), allocatable                :: lrs_save(:), lre_save(:)

  logical                                      :: a_lower, a_upper, c_lower, c_upper
  logical                                      :: a_transposed, b_transposed
  MATH_DATATYPE(kind=rck)                      :: beta
  MATH_DATATYPE(kind=rck), pointer, contiguous :: aux_mat(:,:), tmp1(:,:)
  MATH_DATATYPE(kind=rck), pointer, contiguous :: aux_a_full(:,:), aux_b_full(:,:), tmp1_full(:,:), tmp2_full(:,:) ! PETERDEBUG
  MATH_DATATYPE(kind=rck), allocatable         :: aux_bc(:), tmp2(:,:)
  logical                                      :: wantDebug
  integer(kind=ik)                             :: istat, debug
  character(200)                               :: errorMessage
  character(20)                                :: gpuString
  logical                                      :: success, successGPU, successGPU2
  logical                                      :: useGPU
  logical                                      :: isSquareGrid
  integer(kind=c_int)                          :: numGPU, blocking
  integer(kind=ik)                             :: mpi_comm_rows, mpi_comm_cols, mpi_comm_all
  integer(kind=ik)                             :: mpi_comm_dirs ! PETERDEBUG_NEW
  integer(kind=ik)                             :: nblk, matrixRows, matrixCols, error
  integer(kind=c_intptr_t)                     :: aux_bc_dev, aux_mat_dev, tmp1_dev, tmp2_dev
  integer(kind=c_intptr_t)                     :: aux_a_full_dev, aux_b_full_dev, tmp1_full_dev, tmp2_full_dev ! PETERDEBUG_NEW
!#ifndef DEVICE_POINTER
  integer(kind=c_intptr_t)                     :: a_dev
  integer(kind=c_intptr_t)                     :: b_dev
  integer(kind=c_intptr_t)                     :: c_dev
!#endif
  type(c_ptr)                                  :: aux_host
  integer(kind=c_intptr_t)                     :: num
  integer(kind=c_intptr_t)                     :: aux_off, b_off
  integer(kind=c_intptr_t), parameter          :: size_of_datatype = size_of_&
                                                            &PRECISION&
                                                            &_&
                                                            &MATH_DATATYPE

  integer(kind=c_intptr_t)                     :: gpuHandle, my_stream
  integer(kind=c_int)                          :: gpu_hermitian_multiply

  logical                                      :: useCCL
#if defined(USE_CCL_MULTIPLY)
  integer(kind=c_intptr_t)                     :: ccl_comm_rows, ccl_comm_cols
  integer(kind=c_int)                          :: cclDataType
  integer(kind=ik)                             :: k_datatype
#endif

#ifdef DEVICE_POINTER
  MATH_DATATYPE(kind=rck), allocatable         :: a_tmp(:,:), c_tmp(:,:)
#endif
  integer(kind=c_intptr_t)                     :: aux_dev
  integer(kind=c_int)                          :: gpu
  integer(kind=c_int)                          :: gpu_multiply_a_b

#ifdef WITH_NVTX
  call nvtxRangePush("multiply")
#endif

  success = .true.
  useGPU = .false.

  a_transposed = .false.
  b_transposed = .false.
  if (trans_a=='t' .or. trans_a=='T' .or. trans_a=='c' .or. trans_a=='C') a_transposed = .true.
  if (trans_b=='t' .or. trans_b=='T' .or. trans_b=='c' .or. trans_b=='C') b_transposed = .true.
  print *, "a_transposed = ", a_transposed ! PETERDEBUG
  print *, "b_transposed = ", b_transposed

  call obj%get("debug", debug, error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "ELPA_MULTIPLY_AB: Problem getting option for debug settings. Aborting..."
    success = .false.
    return
  endif
  if (debug == 1) then
    wantDebug = .true.
  else
    wantDebug = .false.
  endif


#if !defined(DEVICE_POINTER)

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
  if (.not.(query_gpu_usage(obj, "ELPA_MULITPLY_AB", useGPU))) then
    print *,"ELPA_MULITPLY_AB: Problem querrying settings for GPU Aborting..."
    stop 1
  endif
#endif

  ! check whether the above setting should be overriden
  if (obj%is_set("gpu_hermitian_multiply") == 1) then
    call obj%get("gpu_hermitian_multiply", gpu_hermitian_multiply, error)
    if (error .ne. ELPA_OK) then
      print *,"Problem getting option for gpu_hermitian_mutltiply. Aborting..."
      stop 1
    endif
    if (useGPU .and. gpu_hermitian_multiply .eq. 0) then
      useGPU = .false.
    else if (.not.(useGPU) .and. gpu_hermitian_multiply .eq. 1) then
      useGPU = .true.
    else
    endif
  else
    ! no override by user
    ! keep seeting as found before
  endif

#else /* DEVICE_POINTER */

  useGPU = .true.

  a_dev = transfer(aDev, a_dev)
  b_dev = transfer(bDev, b_dev)
  c_dev = transfer(cDev, c_dev)

#endif /* DEVICE_POINTER */

  if(useGPU) then
    gpuString = "_gpu"
  else
    gpuString = ""
  endif

  call obj%timer%start("elpa_multiply_&
  &MATH_DATATYPE&
  &_&
  &PRECISION&
  &"//gpuString)

  na          = obj%na
  nblk        = obj%nblk
  matrixRows  = obj%local_nrows
  matrixCols  = obj%local_ncols

  mpi_comm_all    = obj%mpi_setup%mpi_comm_parent
  mpi_comm_cols   = obj%mpi_setup%mpi_comm_cols
  mpi_comm_rows   = obj%mpi_setup%mpi_comm_rows

  myid    = obj%mpi_setup%myRank_comm_parent
  my_prow = obj%mpi_setup%myRank_comm_rows
  my_pcol = obj%mpi_setup%myRank_comm_cols

  np_rows = obj%mpi_setup%nRanks_comm_rows
  np_cols = obj%mpi_setup%nRanks_comm_cols

  l_rows = local_index(na,  my_prow, np_rows, nblk, -1) ! Local rows of a and b
  l_cols = local_index(ncb, my_pcol, np_cols, nblk, -1) ! Local cols of b

  ! Block factor for matrix multiplications, must be a multiple of nblk

  if (obj%is_set("blocking_in_multiply") == 1) then
    call obj%get("blocking_in_multiply", blocking, error)
    if (error .ne. ELPA_OK) then
      write(error_unit,*) "ELPA_MULTIPLY: Problem in getting keyword 'blocking_in_multiply'. Aborting..."
      stop 1
    endif
    nblk_mult = (blocking/nblk+1) * nblk ! PETERDEBUG: how autotuning works here? d_blocking in one step should be ~nblk
  else ! is_set
    if (useGPU) then
      if (na/np_rows <= 256) then
        nblk_mult = (63/nblk+1)*nblk
      else
        nblk_mult = (351/nblk+1)*nblk
      endif
    else ! useGPU
      if (na/np_rows <= 256) then
        nblk_mult = (31/nblk+1)*nblk
      else
        nblk_mult = (63/nblk+1)*nblk
      endif
    endif ! useGPU
  endif ! is_set

  if (useGPU) then
    call obj%timer%start("check_for_gpu")
    if (check_for_gpu(obj, myid, numGPU)) then
      ! set the neccessary parameters
      call set_gpu_parameters()
    else
      print *,"GPUs are requested but not detected! Aborting..."
      success = .false.
      return
    endif
    call obj%timer%stop("check_for_gpu")
    
#if defined(USE_CCL_MULTIPLY)
    useCCL = .true.

    ccl_comm_rows = obj%gpu_setup%ccl_comm_rows
    ccl_comm_cols = obj%gpu_setup%ccl_comm_cols

#if   REALCASE == 1 && defined(DOUBLE_PRECISION)
    cclDataType = cclDouble
    k_datatype = 1
#elif REALCASE == 1 && defined(SINGLE_PRECISION)
    cclDataType = cclFloat
    k_datatype = 1
#elif COMPLEXCASE == 1 && defined(DOUBLE_PRECISION)
    cclDataType = cclDouble
    k_datatype = 2
#elif COMPLEXCASE == 1 && defined(SINGLE_PRECISION)
    cclDataType = cclFloat
    k_datatype = 2
#endif
#endif /* defined(USE_CCL_MULTIPLY) */

#if !defined(DEVICE_POINTER)
    num = ldc*ldcCols*size_of_datatype
    successGPU = gpu_malloc(c_dev, num)
    check_alloc_gpu("elpa_multiply: c_dev", successGPU)
    ! no copy from c to c_dev needed since c will be overwritten anyway
#endif

#if !defined(DEVICE_POINTER)
    ! copy b to b_dev
    num = ldb*ldbCols*size_of_datatype
    successGPU = gpu_malloc(b_dev, num)
    check_alloc_gpu("elpa_multiply: b_dev", successGPU)

#if !defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) && !defined(WITH_SYCL_GPU_VERSION)
    successGPU = gpu_host_register(int(loc(b),kind=c_intptr_t),num,&
                  gpuHostRegisterDefault)
#endif    

    check_host_register_gpu("elpa_multiply: b", successGPU)
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    call gpu_memcpy_async_and_stream_synchronize &
    ("elpa_multiply: b to b_dev", b_dev, 0_c_intptr_t, &
                                       b(1:ldb,1:ldbCols), &
                                       1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
    successGPU = gpu_memcpy(b_dev,int(loc(b),kind=c_intptr_t),num,&
                  gpuMemcpyHostToDevice)
    check_memcpy_gpu("elpa_multiply: b to b_dev", successGPU)
#endif

#else /* DEVICE_POINTER */

#endif /* DEVICE_POINTER */

    num = l_rows*nblk_mult*size_of_datatype
#if !defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) && !defined(WITH_SYCL_GPU_VERSION)
    successGPU = gpu_malloc_host(aux_host, num) ! aux_host is needed, because pinning host memory can be done only for 1D arrays
    check_host_alloc_gpu("elpa_multiply: aux_host", successGPU)
    call c_f_pointer(aux_host, aux_mat, (/l_rows,nblk_mult/))
#else
    allocate(aux_mat(l_rows, nblk_mult), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_multiply: aux_mat", istat, errorMessage)
#endif

    successGPU = gpu_malloc(aux_mat_dev, num)
    check_alloc_gpu("elpa_multiply: aux_mat_dev", successGPU)

    num = nblk_mult*l_cols*size_of_datatype
    successGPU = gpu_malloc(tmp1_dev, num)
    check_alloc_gpu("elpa_multiply: tmp1_dev", successGPU)

    num = nblk_mult*l_cols*size_of_datatype
    successGPU = gpu_malloc(tmp2_dev, num)
    check_alloc_gpu("elpa_multiply: tmp2_dev", successGPU)

  else ! useGPU
    allocate(aux_mat(l_rows,nblk_mult), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_multiply: aux_mat", istat, errorMessage)
  endif ! useGPU

  allocate(aux_bc(l_rows*nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_multiply: aux_bc", istat, errorMessage)

  allocate(lrs_save(nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_multiply: lrs_save", istat, errorMessage)

  allocate(lre_save(nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_multiply: lre_save", istat, errorMessage)

  a_lower = .false.
  a_upper = .false.
  c_lower = .false.
  c_upper = .false.

  if (uplo_a=='u' .or. uplo_a=='U') a_upper = .true.
  if (uplo_a=='l' .or. uplo_a=='L') a_lower = .true.
  if (uplo_c=='u' .or. uplo_c=='U') c_upper = .true.
  if (uplo_c=='l' .or. uplo_c=='L') c_lower = .true.

  isSquareGrid = .true.
  if (np_rows/=np_cols) isSquareGrid = .false. ! PETERDEBUG switch off old codepath
  if (a_upper .or. a_lower .or. c_upper .or. c_lower) isSquareGrid = .false.
  print *, "isSquareGrid = ", isSquareGrid ! PETERDEBUG
  print *, "a_upper = ", a_upper
  print *, "a_lower = ", a_lower
  print *, "c_upper = ", c_upper
  print *, "c_lower = ", c_lower
  print *, "np_rows = ", np_rows
  print *, "np_cols = ", np_cols

  if (useGPU) then

#if !defined(DEVICE_POINTER)
    num = obj%local_nrows*obj%local_ncols*size_of_datatype
    successGPU = gpu_malloc(a_dev, num)
    check_alloc_gpu("elpa_multiply: a_dev", successGPU)
#endif

    num = l_rows*nblk*size_of_datatype
    successGPU = gpu_malloc(aux_bc_dev, num)
    check_alloc_gpu("elpa_multiply: aux_bc_dev", successGPU)

    num = obj%local_nrows*obj%local_ncols*size_of_datatype
#if !defined(DEVICE_POINTER)

#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    call gpu_memcpy_async_and_stream_synchronize &
    ("elpa_multiply: a to a_dev", a_dev, 0_c_intptr_t, &
                                       a(1:obj%local_nrows,1:obj%local_ncols), &
                                       1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
    successGPU = gpu_memcpy(a_dev, int(loc(a),kind=c_intptr_t), &
                  num, gpuMemcpyHostToDevice)
    check_memcpy_gpu("elpa_multiply: a to a_dev", successGPU)
#endif
#endif /* DEVICE_POINTER */
  endif !useGPU

! _________________________________________________________________________________________________________________________________

  !if (.not. isSquareGrid .or. useGPU) then ! PETERDEBUG: delete useGPU, upon porting square grid case to GPU
  if (.not. isSquareGrid) then
    print *, "Old codepath" ! PETERDEBUG

  ! main loop: build up the result matrix by processor rows
    do np = 0, np_rows-1 ! PETERDEBUG: np -> np_row_curr or np_row_c (for matrix C)

#ifdef WITH_NVTX
      call nvtxRangePush("do np = 0, np_rows-1")
#endif

      ! In this turn, procs of row np assemble the result

      l_rows_np = local_index(na, np, np_rows, nblk, -1) ! local rows on receiving processors

      nr_done = 0 ! Number of rows done
      nstor = 0   ! Number of columns stored in aux_mat

      aux_mat = 0
      if (useGPU) then
        num = l_rows*nblk_mult*size_of_datatype
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        successGPU = gpu_memset_async(aux_mat_dev, 0, num, my_stream)
        check_memcpy_gpu("multiply: aux_mat_dev", successGPU)
#else
        successGPU = gpu_memset(aux_mat_dev, 0, num)
        check_memcpy_gpu("multiply: aux_mat_dev", successGPU)
#endif
      endif ! useGPU

      ! Loop over the blocks on row np; nb is the 0-based local index of the block
      do nb = 0, (l_rows_np-1)/nblk

#ifdef WITH_NVTX
        call nvtxRangePush("do nb = 0, (l_rows_np-1)/nblk")
#endif

        goff  = nb*np_rows + np ! offset in the global grid of blocks
        ! rename: goff -> block_offset_gl or I_block_gl; global coordinate of the given block in the grid of blocks ! PETERDEBUG

        ! Get the processor row (if trans_a='N') or column (if trans_a='T' or 'H') which owns this block
        ! and the offset in blocks within this row/column.
        ! The corresponding block row/column in A is then broadcast to all for multiplication with B

        np_bc = MOD(goff, np_cols) ! np, that posesses the given column of blocks; trans_a='T'; "bc"=block column; rename: np_bc -> np_col_b / np_col_curr
        np_br = MOD(goff, np_rows) ! np, that posesses the given row of blocks   ; trans_a='N'; "br"=block row ! PETERDEBUG
        
        noff = goff/np_cols   ! offset in the local grid of blocks
        noff_n = goff/np_rows ! PETERDEBUG
        
        ! Gather up the complete column/row of blocks of A (for T/N case) on the owner in contigous memory of aux_bc array
        n_aux_bc = 0
        ! PETERDEBUG: this loop below is essentially a one-liner if not for upper/lower cases: aux_bc_2D(1:l_rows,1:l_cols) = a(1:l_rows,1:l_cols)
        do n = 1, min(nblk, l_rows_np-nb*nblk) ! Loop over local columns/rows (for T/N) to be broadcast

          gcol = goff*nblk + n ! global column corresponding to n, needed only for a_lower and a_upper cases

          if (nstor==0 .and. n==1) gcol_min = gcol

          lrs = 1       ! 1st (start) local row number for broadcast
          lre = l_rows  ! last (end)  local row number for broadcast
          if (a_lower) lrs = local_index(gcol, my_prow, np_rows, nblk, +1)
          if (a_upper) lre = local_index(gcol, my_prow, np_rows, nblk, -1)

          if (lrs <= lre) then
            nvals = lre-lrs+1
            if (useGPU) then
              if (my_pcol == np_bc) call gpu_copy_PRECISION_a_aux_bc(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, &
                                                                     nblk, n, l_rows, obj%local_nrows, obj%local_ncols, my_stream)
            else ! useGPU
              if (my_pcol == np_bc) aux_bc(n_aux_bc+1:n_aux_bc+nvals) = a(lrs:lre,noff*nblk+n)
            endif ! useGPU

            n_aux_bc = n_aux_bc + nvals
          endif ! (lrs <= lre)

          lrs_save(n) = lrs
          lre_save(n) = lre

        enddo ! n = 1, min(nblk, l_rows_np-nb*nblk)

#ifdef WITH_MPI
        ! copy data to host for bcast, if needed
        if (useGPU .and. .not. useCCL) then
          num = l_rows*nblk*size_of_datatype
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
               ("elpa_multiply: aux_bc_dev -> aux_bc", aux_bc_dev, 0_c_intptr_t, aux_bc(1:l_rows*nblk), &
                1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
          successGPU = gpu_memcpy(int(loc(aux_bc),kind=c_intptr_t), aux_bc_dev, num, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_multiply: aux_bc_dev -> aux_bc", successGPU)
#endif
        endif ! useGPU  .and. .not. useCCL

        ! Broadcast block column
        if (useCCL) then
#ifdef USE_CCL_MULTIPLY
#ifdef WITH_NVTX
          call nvtxRangePush("ccl_bcast aux_bc_dev")
#endif      
          call obj%timer%start("ccl_bcast")

          my_stream = obj%gpu_setup%my_stream
          ccl_comm_cols = obj%gpu_setup%ccl_comm_cols

          ! PETERDEBUG We can send the whole aux_bc_2D(1:l_rows,1:l_cols) = a(1:l_rows,1:l_cols) straight away ??? ! We don't need to copy it to aux_bc_dev at all!
          successGPU = ccl_bcast(aux_bc_dev, aux_bc_dev, int(k_datatype*n_aux_bc,kind=c_size_t), cclDatatype, &
                                int(np_bc,kind=c_int), ccl_comm_cols, my_stream)

          if (.not. successGPU) then
            print *,"Error in ccl_bcast"
            stop 1
          endif

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_cholesky: ccl_bcast", successGPU)

          call obj%timer%stop("ccl_bcast")
#ifdef WITH_NVTX
          call nvtxRangePop() ! ccl_bcast aux_bc_dev
#endif
#endif /* USE_CCL_MULTIPLY */
        else ! useCCL
          call obj%timer%start("mpi_communication")

          call MPI_Bcast(aux_bc, int(n_aux_bc,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                        int(np_bc,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)

          call obj%timer%stop("mpi_communication")
        endif ! useCCL

        ! copy data back to device, if needed
        if (useGPU .and. .not. useCCL) then
          num = l_rows*nblk*size_of_datatype
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_multiply: aux_bc -> aux_bc_dev", aux_bc_dev, 0_c_intptr_t, aux_bc(1:l_rows*nblk), &
                1, num, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
          successGPU = gpu_memcpy(aux_bc_dev, int(loc(aux_bc),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_multiply: aux_bc -> aux_bc_dev", successGPU)
#endif
        endif ! useGPU .and. .not. useCCL
#endif /* WITH_MPI */


        ! Copy what we got in aux_mat
        if (useGPU) then
          n_aux_bc = 0
          my_stream = obj%gpu_setup%my_stream
          do n = 1, min(nblk, l_rows_np-nb*nblk)
            nstor = nstor+1
            lrs = lrs_save(n)
            lre = lre_save(n)
            if (lrs <= lre) then
              nvals = lre-lrs+1
              call gpu_copy_PRECISION_aux_bc_aux_mat(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, &
                                                    nvals, l_rows, nblk, nblk_mult, my_stream)

              n_aux_bc = n_aux_bc + nvals
            endif
          enddo
        else ! useGPU
          n_aux_bc = 0
          do n = 1, min(nblk, l_rows_np-nb*nblk)
            nstor = nstor+1
            lrs = lrs_save(n)
            lre = lre_save(n)
            if (lrs<=lre) then
              nvals = lre-lrs+1
              aux_mat(lrs:lre,nstor) = aux_bc(n_aux_bc+1:n_aux_bc+nvals)
              n_aux_bc = n_aux_bc + nvals
            endif
          enddo
        endif ! useGPU

        ! If we got nblk_mult columns in aux_mat or this is the last block
        ! do the matrix multiplication

        if (nstor==nblk_mult .or. nb*nblk+nblk >= l_rows_np) then

          lrs = 1       ! 1st local row number for multiply
          lre = l_rows  ! last local row number for multiply
          if (a_lower) lrs = local_index(gcol_min, my_prow, np_rows, nblk, +1)
          if (a_upper) lre = local_index(gcol, my_prow, np_rows, nblk, -1)

          lcs = 1       ! 1st local col number for multiply
          lce = l_cols  ! last local col number for multiply
          if (c_upper) lcs = local_index(gcol_min, my_pcol, np_cols, nblk, +1)
          if (c_lower) lce = MIN(local_index(gcol, my_pcol, np_cols, nblk, -1),l_cols)

          if (lcs <= lce) then
            if (.not. useCCL) then
              ! introduce 1-based indexing
              allocate(tmp1(nstor,1:lce-lcs+1), tmp2(nstor,1:lce-lcs+1), stat=istat, errmsg=errorMessage)
              call check_alloc("elpa_multiply_&
                              &MATH_DATATYPE ", "tmp1", istat, errorMessage)
            endif

            if (lrs <= lre) then
              if (useGPU) then
                aux_off = (lrs-1)*size_of_datatype
                b_off = ((lcs-1)*ldb+lrs-1)*size_of_datatype

#ifdef WITH_NVTX
                call nvtxRangePush("gpublas")
#endif
                call obj%timer%start("gpublas")
                gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
                ! tmp1_dev = aux_mat_dev^{T/N} * b_dev
                call gpublas_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', nstor, lce-lcs+1, lre-lrs+1, ONE, &
                                            aux_mat_dev+aux_off, l_rows, &
                                            b_dev+b_off, ldb, ZERO, &
                                            tmp1_dev, nstor, gpuHandle)
                if (wantDebug) successGPU = gpu_DeviceSynchronize()
                call obj%timer%stop("gpublas")
#ifdef WITH_NVTX
                call nvtxRangePop() ! gpublas
#endif
                num = nstor*(lce-lcs+1)*size_of_datatype ! PETERDEBUG: just a hanging line, delete it
              else ! useGPU
                call obj%timer%start("blas")
                ! tmp1 = aux_mat^{T/N} * b
                call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', int(nstor,kind=BLAS_KIND), &
                                  int(lce-lcs+1,kind=BLAS_KIND), int(lre-lrs+1,kind=BLAS_KIND), ONE, &
                                  aux_mat(lrs:lre,1:nstor), int(lre-lrs+1,kind=BLAS_KIND), &
                                  b(lrs,lcs), int(ldb,kind=BLAS_KIND), ZERO, &
                                  tmp1, int(nstor,kind=BLAS_KIND))
                call obj%timer%stop("blas")
              endif ! useGPU
            else ! (lrs <= lre)
              if (useGPU) then
                num = nstor*(lce-lcs+1)*size_of_datatype
#ifdef WITH_GPU_STREAMS
                my_stream = obj%gpu_setup%my_stream
                successGPU = gpu_memset_async(tmp1_dev, 0, num, my_stream)
                check_memcpy_gpu("multiply: tmp1_dev", successGPU)
#else
                successGPU = gpu_memset(tmp1_dev, 0, num)
                check_memcpy_gpu("multiply: tmp1_dev", successGPU)
#endif
              else ! useGPU 
                tmp1 = 0
              endif ! useGPU
            endif ! (lrs <= lre)

            ! Sum up the results and send to processor row np

#ifdef WITH_MPI
            ! copy data to host, if needed
            if (useGPU .and. .not. useCCL) then
              num = nstor*(lce-lcs+1)*size_of_datatype
#ifdef WITH_GPU_STREAMS
              call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_multiply: tmp1_dev to tmp1", tmp1_dev, 0_c_intptr_t, &
                                                  !tmp1(1:nblk_mult,1:l_cols), &
                                                  tmp1(1:nstor,1:lce-lcs+1), &
                                                  1, 1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
              successGPU = gpu_memcpy(int(loc(tmp1),kind=c_intptr_t), &
                              tmp1_dev, num, gpuMemcpyDeviceToHost)
              check_memcpy_gpu("elpa_multiply: tmp1_dev to tmp1", successGPU)
#endif
            endif ! useGPU .and. .not. useCCL

            ! MPI/ccl Reduce
            if (useCCL) then
#ifdef USE_CCL_MULTIPLY
#ifdef WITH_NVTX
              call nvtxRangePush("ccl_reduce tmp1_dev")
#endif
              call obj%timer%start("ccl_reduce")
              my_stream = obj%gpu_setup%my_stream
              ccl_comm_rows = obj%gpu_setup%ccl_comm_rows

              successGPU = ccl_reduce(tmp1_dev, tmp2_dev, int(k_datatype*nstor*(lce-lcs+1),kind=c_size_t), cclDataType, &
                                      cclSum, int(np,kind=c_int), ccl_comm_rows, my_stream)

              if (.not. successGPU) then
                print *,"Error in ccl_reduce"
                stop 1
              endif

              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("elpa_cholesky: ccl_reduce", successGPU)

              call obj%timer%stop("ccl_reduce")
#ifdef WITH_NVTX
              call nvtxRangePop() ! ccl_reduce tmp1_dev
#endif
#endif /* USE_CCL_MULTIPLY */
            else ! useCCL
              call obj%timer%start("mpi_communication")
              call mpi_reduce(tmp1, tmp2, int(nstor*(lce-lcs+1),kind=MPI_KIND),  MPI_MATH_DATATYPE_PRECISION, &
                            MPI_SUM, int(np,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
              call obj%timer%stop("mpi_communication")
            endif ! useCCL

            ! copy data back to device, if needed
            if (useGPU .and. .not. useCCL) then
              num = nstor*(lce-lcs+1)*size_of_datatype
#ifdef WITH_GPU_STREAMS
              call gpu_memcpy_async_and_stream_synchronize &
                  ("elpa_multiply: tmp2 to tmp2_dev", tmp2_dev, 0_c_intptr_t, &
                                                  !tmp2(1:nblk_mult,1:l_cols), &
                                                  tmp2(1:nstor,1:lce-lcs+1), &
                                                  1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
              successGPU = gpu_memcpy(tmp2_dev, int(loc(tmp2),kind=c_intptr_t), &
                                      num, gpuMemcpyHostToDevice)
              check_memcpy_gpu("elpa_multiply: tmp2 to tmp2_dev", successGPU)
#endif
            endif ! useGPU .and. .not. useCCL
#else /* WITH_MPI */

            if (useGPU)
              num = nstor*(lce-lcs+1)*size_of_datatype
              successGPU = gpu_memcpy(tmp2_dev, tmp1_dev, num, gpuMemcpyDeviceToDevice)
              check_memcpy_gpu("elpa_multiply: tmp2 to tmp2_dev", successGPU)
            endif
#endif /* WITH_MPI */


            if (useGPU) then
              if (my_prow==np) call gpu_copy_PRECISION_tmp2_c(tmp2_dev, c_dev, nr_done, nstor, &
                                                              lcs, lce, ldc, ldcCols, my_stream)
            else ! useGPU
#ifdef WITH_MPI
              ! Put the result into C
              if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp2(1:nstor,1:lce-lcs+1)
#else /* WITH_MPI */
              ! Put the result into C
              if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp1(1:nstor,1:lce-lcs+1)
              !tmp2(:,:) = 0.
#endif /* WITH_MPI */
            endif ! useGPU

            if (.not. useCCL) then
                deallocate(tmp1, tmp2, stat=istat, errmsg=errorMessage)
                call check_alloc("elpa_multiply_&
                  &MATH_DATATYPE ", "tmp1", istat, errorMessage)
            endif
          endif ! (lcs <= lce)

          nr_done = nr_done+nstor
          nstor=0
          if (useGPU) then
            num = l_rows*nblk_mult*size_of_datatype
#ifdef WITH_GPU_STREAMS
            my_stream = obj%gpu_setup%my_stream
            successGPU = gpu_memset_async(aux_mat_dev, 0, num, my_stream)
            check_memcpy_gpu("multiply: aux_mat_dev", successGPU)
#else
            successGPU = gpu_memset(aux_mat_dev, 0, num)
            check_memcpy_gpu("multiply: aux_mat_dev", successGPU)
#endif
          else ! useGPU
            aux_mat(:,:) = 0
          endif ! useGPU
        endif ! (nstor==nblk_mult .or. nb*nblk+nblk >= l_rows_np)
      
#ifdef WITH_NVTX
        call nvtxRangePop() ! do nb = 0, (l_rows_np-1)/nblk
#endif
      enddo ! nb = 0, (l_rows_np-1)/nblk

#ifdef WITH_NVTX
      call nvtxRangePop() ! do np = 0, np_rows-1
#endif
    enddo ! main loop: np = 0, np_rows-1

!_______________________________________________

    if (useGPU) then
      ! copy result c_dev back to CPU
      num = ldc*ldcCols
#ifdef WITH_GPU_STREAMS
      check_stream_synchronize_gpu("elpa_multiply: c_dev -> c", successGPU)
      call gpu_memcpy_async_and_stream_synchronize &
          ("elpa_multiply: c_dev to c", c_dev, 0_c_intptr_t, c(1:ldc,1:ldcCols), &
            1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
      successGPU = gpu_memcpy(int(loc(c),kind=c_intptr_t), c_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("elpa_multiply: c_dev -> c", successGPU)
#endif
    endif ! useGPU
  endif ! .not. isSquareGrid

!______________________________________________________________________________________________

  !if (isSquareGrid .and. .not. useGPU) then ! PETERDEBUG: delete useGPU, upon porting square grid case to GPU
  if (isSquareGrid) then
    ! if (useGPU) then
    !   print *, "elpa_multiply NEW: isSquareGrid and useGPU is not imlemented yet" ! PETERDEBUG
    !   stop 1
    ! endif

    print *, "elpa_multiply NEW: start" ! PETERDEBUG

    ! l_rows_max = l_rows
    call mpi_allreduce(l_rows, l_rows_max, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
    call mpi_allreduce(l_cols, l_cols_max, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
    call mpi_allreduce(l_rows, l_rows_min, 1_MPI_KIND, MPI_INTEGER, MPI_MIN, int(mpi_comm_all,kind=MPI_KIND), mpierr)
    call mpi_allreduce(l_cols, l_cols_min, 1_MPI_KIND, MPI_INTEGER, MPI_MIN, int(mpi_comm_all,kind=MPI_KIND), mpierr)
    
    !nblk_mult = l_rows_max ! = l_cols_max
    nblk_mult = greatest_common_divisor(l_rows_max, l_cols_max)

    allocate(aux_a_full(nblk_mult, nblk_mult), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_multiply: aux_a_full", istat, errorMessage)

    allocate(aux_b_full(nblk_mult, nblk_mult), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_multiply: aux_b_full", istat, errorMessage)
    
    allocate(tmp1_full(nblk_mult, nblk_mult), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_multiply: tmp1_full", istat, errorMessage)

    allocate(tmp2_full(nblk_mult, nblk_mult), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_multiply: tmp2_full", istat, errorMessage)

    ! PETERDEBUG: is it possible to use the original GPU memory, without copying?
    if (useGPU) then
      successGPU = gpu_malloc(aux_a_full_dev, nblk_mult*nblk_mult*size_of_datatype)
      check_alloc_gpu("elpa_multiply: aux_a_full_dev", successGPU)

      successGPU = gpu_malloc(aux_b_full_dev, nblk_mult*nblk_mult*size_of_datatype)
      check_alloc_gpu("elpa_multiply: aux_b_full_dev", successGPU)

      successGPU = gpu_malloc(tmp1_full_dev, nblk_mult*nblk_mult*size_of_datatype)
      check_alloc_gpu("elpa_multiply: tmp1_full_dev", successGPU)
    endif

!_______________________________________________

    if (.not. a_transposed .and. b_transposed .or. &
        a_transposed .and. .not. b_transposed ) then
      print *, "elpa_multiply NEW: SQUARE_GRID start: ( a_transposed XOR b_transposed)" ! PETERDEBUG
      
      ! dir = row/col for TN/NT
      if (a_transposed) then
        my_pdir = my_prow
        np_dirs = np_rows
        mpi_comm_dirs = mpi_comm_rows
      else if (b_transposed) then
        my_pdir = my_pcol
        np_dirs = np_cols
        mpi_comm_dirs = mpi_comm_cols
      endif

      ! main loop: build up the result matrix by processor rows/cols for TN/NT
      do np = 0, np_dirs-1
#ifdef WITH_NVTX
        call nvtxRangePush("do np = 0, np_dirs-1")
#endif
        print *, "np = ", np ! PETERDEBUG

        ! In this turn, procs of row/col np assemble the result for TN/NT case
        
        np_t=np ! np, that posesses the given "col of a"/"row of b" for TN/NT

        if (a_transposed) then
          if (np_t == my_pcol) aux_a_full(1:l_rows,1:l_cols) = a(1:l_rows,1:l_cols)
          aux_b_full(1:l_rows,1:l_cols) = b(1:l_rows,1:l_cols)
        else if (b_transposed) then
          if (np_t == my_prow) aux_b_full(1:l_rows,1:l_cols) = b(1:l_rows,1:l_cols)
          aux_a_full(1:l_rows,1:l_cols) = a(1:l_rows,1:l_cols) ! aux_a_full -> aux_ab_nontransposed_full
        endif
        
        ! aux_a_full/aux_b_full -> aux_ab_trans (aux_ab) ! auxillary buffer for a matrix that is     to be transposed: a/b in TN/NT case
        ! aux_b_full/aux_a_full -> aux_ab_nontr (aux_ba) ! auxillary buffer for a matrix that is not to be transposed: b/a in TN/NT case

        ! Broadcast processor column
        call obj%timer%start("mpi_communication")
#ifdef WITH_NVTX
        call nvtxRangePush("MPI_Bcast(aux_a/b_full)")
#endif
        if (a_transposed) then
          call MPI_Bcast(aux_a_full, int(nblk_mult*nblk_mult,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                        int(np_t,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
        else if (b_transposed) then
          call MPI_Bcast(aux_b_full, int(nblk_mult*nblk_mult,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                        int(np_t,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
        endif

#ifdef WITH_NVTX
        call nvtxRangePop() ! MPI_Bcast(aux_a/b_full)
#endif
        call obj%timer%stop("mpi_communication")

        call obj%timer%start("blas")

        call PRECISION_GEMM(trans_a, trans_b, &
                            int(nblk_mult, kind=BLAS_KIND), &
                            int(nblk_mult, kind=BLAS_KIND), &
                            int(nblk_mult, kind=BLAS_KIND), ONE, &
                            aux_a_full, int(nblk_mult, kind=BLAS_KIND), &
                            aux_b_full, int(nblk_mult, kind=BLAS_KIND), ZERO, &
                            tmp1_full , int(nblk_mult, kind=BLAS_KIND))

        call obj%timer%stop("blas")

        call obj%timer%start("mpi_communication")
        if (my_pdir==np) then
          call mpi_reduce(MPI_IN_PLACE, tmp1_full, int(nblk_mult*nblk_mult,kind=MPI_KIND),  MPI_MATH_DATATYPE_PRECISION, &
                          MPI_SUM, int(np,kind=MPI_KIND), int(mpi_comm_dirs,kind=MPI_KIND), mpierr)
        else
          call mpi_reduce(tmp1_full   , tmp1_full, int(nblk_mult*nblk_mult,kind=MPI_KIND),  MPI_MATH_DATATYPE_PRECISION, &
                          MPI_SUM, int(np,kind=MPI_KIND), int(mpi_comm_dirs,kind=MPI_KIND), mpierr)
        endif
        call obj%timer%stop("mpi_communication")

        ! Put the result into C
        if (my_pdir==np) c(1:l_rows,1:l_cols) = tmp1_full(1:l_rows,1:l_cols)

#ifdef WITH_NVTX
        call nvtxRangePop() ! do np = 0, np_dirs-1
#endif
      enddo ! np = 0, np_dirs-1
    endif ! (.not. a_transposed .and. b_transposed)

!_______________________________________________

    if ((.not. a_transposed) .and. (.not. b_transposed)) then
      print *, "elpa_multiply NEW: SQUARE_GRID start: (.not. a_transposed) .and. (.not. b_transposed)" ! PETERDEBUG

      ! main loop: iterate through np, which are process rows for matrix A and process cols for matrix B
      do np = 0, np_rows-1 ! np_rows=np_cols
#ifdef WITH_NVTX
        call nvtxRangePush("np = 0, np_rows-1")
#endif
        print *, "np = ", np ! PETERDEBUG

        ! In this turn, procs of row np assemble the result
        
        np_bc=np ! np, that posesses the given column of a

        if (np_bc == my_pcol) then
          if (useGPU) then
            call gpu_copy_and_set_zeros_aux_full(PRECISION_CHAR, a_dev, aux_a_full_dev, &
                                                 l_rows, l_cols, nblk_mult, debug, my_stream)
            ! my_stream = obj%gpu_setup%my_stream
            ! successGPU = gpu_memset_async(aux_a_full_dev, 0, nblk_mult*nblk_mult*size_of_datatype, my_stream)
            ! successGPU = gpu_memcpy(aux_a_full_dev, a_dev, l_rows*l_cols*size_of_datatype, gpuMemcpyDeviceToDevice)
          else ! useGPU
            aux_a_full(1:l_rows,1:l_cols) = a(1:l_rows,1:l_cols)
            if (l_rows<nblk_mult) aux_a_full(l_rows+1:nblk_mult,1:l_cols) = 0
            if (l_cols<nblk_mult) aux_a_full(1:l_rows,l_cols+1:nblk_mult) = 0
            if (l_rows<nblk_mult .and. l_cols<nblk_mult) aux_a_full(l_rows+1:nblk_mult,l_cols+1:nblk_mult) = 0
          endif ! useGPU
        endif
        if (np    == my_prow) then
          if (useGPU) then
            call gpu_copy_and_set_zeros_aux_full(PRECISION_CHAR, b_dev, aux_b_full_dev, &
                                                 l_rows, l_cols, nblk_mult, debug, my_stream)
            ! my_stream = obj%gpu_setup%my_stream
            ! successGPU = gpu_memset_async(aux_b_full_dev, 0, nblk_mult*nblk_mult*size_of_datatype, my_stream)
            ! successGPU = gpu_memcpy(aux_b_full_dev, b_dev, l_rows*l_cols*size_of_datatype, gpuMemcpyDeviceToDevice)
          else ! useGPU
            aux_b_full(1:l_rows,1:l_cols) = b(1:l_rows,1:l_cols)
            if (l_rows<nblk_mult) aux_b_full(l_rows+1:nblk_mult,1:l_cols) = 0
            if (l_cols<nblk_mult) aux_b_full(1:l_rows,l_cols+1:nblk_mult) = 0
            if (l_rows<nblk_mult .and. l_cols<nblk_mult) aux_b_full(l_rows+1:nblk_mult,l_cols+1:nblk_mult) = 0
          endif ! useGPU
        endif

        ! copy data to host for bcast, if needed
        if (useGPU .and. .not. useCCL) then
          num = nblk_mult*nblk_mult
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_multiply: aux_a_full_dev -> aux_a_full", aux_a_full_dev, 0_c_intptr_t, aux_a_full(1:nblk_mult,1:nblk_mult), &
                1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .false., .false.)
          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_multiply: aux_b_full_dev -> aux_b_full", aux_b_full_dev, 0_c_intptr_t, aux_b_full(1:nblk_mult,1:nblk_mult), &
                1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
          successGPU = gpu_memcpy(int(loc(aux_a_full),kind=c_intptr_t), aux_a_full_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_multiply: aux_a_full_dev -> aux_a_full", successGPU)

          successGPU = gpu_memcpy(int(loc(aux_b_full),kind=c_intptr_t), aux_b_full_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_multiply: aux_b_full_dev -> aux_b_full", successGPU)
#endif  
        endif ! (useGPU .and. .not. useCCL)

        ! Broadcast processor column
        if (useCCL) then
#ifdef USE_CCL_MULTIPLY
#ifdef WITH_NVTX
          call nvtxRangePush("ccl_bcast aux_a_full_dev, aux_b_full_dev")
#endif
          call obj%timer%start("ccl_bcast")

          my_stream = obj%gpu_setup%my_stream
          ccl_comm_cols = obj%gpu_setup%ccl_comm_cols

          successGPU  = ccl_bcast(aux_a_full_dev, aux_a_full_dev, int(k_datatype*nblk_mult*nblk_mult,kind=c_size_t), cclDatatype, &
                                  int(np_bc,kind=c_int), ccl_comm_cols, my_stream)

          successGPU2 = ccl_bcast(aux_b_full_dev, aux_b_full_dev, int(k_datatype*nblk_mult*nblk_mult,kind=c_size_t), cclDatatype, &
                                  int(np   ,kind=c_int), ccl_comm_rows, my_stream)
          
          if (.not. (successGPU .and. successGPU2)) then
            print *,"Error in ccl_bcast"
            stop 1
          endif

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_multiply: ccl_bcast", successGPU)

          call obj%timer%stop("ccl_bcast")
#ifdef WITH_NVTX
          call nvtxRangePop() ! ccl_bcast aux_a_full_dev, aux_b_full_dev
#endif
#endif /* USE_CCL_MULTIPLY */
        else ! useCCL
          call obj%timer%start("mpi_communication")
#ifdef WITH_NVTX
          call nvtxRangePush("MPI_Bcast: aux_a_full, aux_b_full")
#endif
          call MPI_Bcast(aux_a_full, int(nblk_mult*nblk_mult,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                        int(np_bc,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
          call MPI_Bcast(aux_b_full, int(nblk_mult*nblk_mult,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                        int(np   ,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
#ifdef WITH_NVTX
          call nvtxRangePop() ! MPI_Bcast: aux_a_full, aux_b_full
#endif
          call obj%timer%stop("mpi_communication")
        endif ! useCCL


        ! copy data back to device, if needed
        if (useGPU .and. .not. useCCL) then
          num = nblk_mult*nblk_mult
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_multiply: aux_a_full -> aux_a_full_dev", aux_a_full_dev, 0_c_intptr_t, aux_a_full(1:nblk_mult,1:nblk_mult), &
                1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
          call gpu_memcpy_async_and_stream_synchronize &
                ("elpa_multiply: aux_b_full -> aux_b_full_dev", aux_b_full_dev, 0_c_intptr_t, aux_b_full(1:nblk_mult,1:nblk_mult), &
                1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
          successGPU = gpu_memcpy(aux_a_full_dev, int(loc(aux_a_full),kind=c_intptr_t), num*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_multiply: aux_a_full -> aux_a_full_dev", successGPU)
          
          successGPU = gpu_memcpy(aux_b_full_dev, int(loc(aux_b_full),kind=c_intptr_t), num*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_multiply: aux_b_full -> aux_b_full_dev", successGPU)
#endif
        endif ! (useGPU .and. .not. useCCL)
        
        beta = ZERO
        if (np>0) beta = ONE
        if (useGPU) then
          gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
#ifdef WITH_NVTX
          call nvtxRangePush("gpublas")
#endif
          call obj%timer%start("gpublas")
          call gpublas_PRECISION_GEMM('N', 'N', &
                            nblk_mult, nblk_mult, nblk_mult, ONE, &
                            aux_a_full_dev, nblk_mult, &
                            aux_b_full_dev, nblk_mult, beta, &
                            tmp1_full_dev , nblk_mult, gpuHandle)
          if (wantDebug) successGPU = gpu_DeviceSynchronize()
          call obj%timer%stop("gpublas")
#ifdef WITH_NVTX
          call nvtxRangePop() ! gpublas
#endif
        else ! useGPU
          call obj%timer%start("blas")
          call PRECISION_GEMM('N', 'N', &
                              int(nblk_mult, kind=BLAS_KIND), &
                              int(nblk_mult, kind=BLAS_KIND), &
                              int(nblk_mult, kind=BLAS_KIND), ONE, &
                              aux_a_full, int(nblk_mult,kind=BLAS_KIND), &
                              aux_b_full, int(nblk_mult,kind=BLAS_KIND), beta, &
                              tmp1_full , int(nblk_mult,kind=BLAS_KIND))
          call obj%timer%stop("blas")
        endif ! useGPU

#ifdef WITH_NVTX
        call nvtxRangePop() ! do np_row_curr = 0, np_rows-1
#endif
      enddo ! np = 0, np_rows-1
      
      ! Put the result into C
      if (useGPU) then
        num = nblk_mult*nblk_mult
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        call gpu_memcpy_async_and_stream_synchronize &
            ("elpa_multiply: tmp1_full_dev -> tmp1_full", tmp1_full_dev, 0_c_intptr_t, tmp1_full(1:nblk_mult,1:nblk_mult), &
              1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
        successGPU = gpu_memcpy(int(loc(tmp1_full),kind=c_intptr_t), tmp1_full_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("elpa_multiply: tmp1_full_dev -> tmp1_full", successGPU)
#endif
      endif ! useGPU

      c(1:l_rows,1:l_cols) = tmp1_full(1:l_rows,1:l_cols)

    endif ! (a_transposed .and. b_transposed)
!_______________________________________________

    if ((a_transposed) .and. (b_transposed)) then

      allocate(at_full(l_rows_max, l_cols_max), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_multiply: at_max", istat, errorMessage)
  
      allocate(bt_full(l_rows_max, l_cols_max), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_multiply: bt_full", istat, errorMessage)
  
      !call mpi_allreduce(nblk_mult, na_max, 1_MPI_KIND, MPI_INTEGER, MPI_SUM, int(mpi_comm_all,kind=MPI_KIND), mpierr)

      print *, "elpa_multiply NEW: SQUARE_GRID start: (a_transposed) .and. (b_transposed)" ! PETERDEBUG

      ! main loop: iterate through np, which are process rows for matrix A and process cols for matrix B
      do np = 0, np_rows-1 ! np_rows=np_cols
#ifdef WITH_NVTX
        call nvtxRangePush("np = 0, np_rows-1")
#endif
        print *, "np = ", np ! PETERDEBUG

        ! In this turn, procs of row np assemble the result
        
        np_bc=np ! np, that posesses the given column of a
        
        if (np_bc == my_prow) then
          aux_a_full(1:l_rows,1:l_cols) = a(1:l_rows,1:l_cols)
          if (l_rows<nblk_mult) aux_a_full(l_rows+1:nblk_mult,1:l_cols) = 0
          if (l_cols<nblk_mult) aux_a_full(1:l_rows,l_cols+1:nblk_mult) = 0
          if (l_rows<nblk_mult .and. l_cols<nblk_mult) aux_a_full(l_rows+1:nblk_mult,l_cols+1:nblk_mult) = 0
        endif

        ! PETERDEBUG: approach Bcast+elpa_transpose_vectors can be optimized: it uses two Bcasts instead of one
        ! alternative approach: write a function that transposes one block-column by send-recv
        call MPI_Bcast(aux_a_full, int(nblk_mult*nblk_mult,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
               int(np_bc,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)

        ! a -> at_full: transpose row #np of a
        ! elpa_transpose_vectors: There must be an identical copy of vmat_s in every communicator comm_s
        ! it performs bcast after transpose; it doesn't perform transpose inside a block
        call elpa_transpose_vectors_&
              &MATH_DATATYPE&
              &_&
              &PRECISION&
              (obj, aux_a_full, l_rows_max, mpi_comm_cols, at_full, l_rows_max, mpi_comm_rows, &
               1, nblk_mult*np_rows, l_cols_max, nblk_mult, 1, .true., success) ! PETERDEBUG: 1 (third from last) -> max_threads
        
        ! b -> bt_full: transpose column #np of b
        if (np    == my_pcol) then
          aux_b_full(1:l_rows,1:l_cols) = b(1:l_rows,1:l_cols)
          if (l_rows<nblk_mult) aux_b_full(l_rows+1:nblk_mult,1:l_cols) = 0
          if (l_cols<nblk_mult) aux_b_full(1:l_rows,l_cols+1:nblk_mult) = 0
          if (l_rows<nblk_mult .and. l_cols<nblk_mult) aux_b_full(l_rows+1:nblk_mult,l_cols+1:nblk_mult) = 0
        endif
        call MPI_Bcast(aux_b_full, int(nblk_mult*nblk_mult,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                int(np   ,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)

        call elpa_transpose_vectors_&
               &MATH_DATATYPE&
               &_&
               &PRECISION&
               (obj, aux_b_full, l_rows_max, mpi_comm_rows, bt_full, l_rows_max, mpi_comm_cols, &
                1, nblk_mult*np_rows, l_cols_max, nblk_mult, 1, .false., success) ! PETERDEBUG: allow nb != na

        call obj%timer%start("blas")
        
        beta = ZERO
        if (np>0) beta = ONE
        call PRECISION_GEMM('T', 'T', &
                            int(nblk_mult, kind=BLAS_KIND), &
                            int(nblk_mult, kind=BLAS_KIND), &
                            int(nblk_mult, kind=BLAS_KIND), ONE, &
                            at_full, int(nblk_mult,kind=BLAS_KIND), &
                            bt_full, int(nblk_mult,kind=BLAS_KIND), beta, &
                            tmp1_full , int(nblk_mult,kind=BLAS_KIND))

        call obj%timer%stop("blas")

#ifdef WITH_NVTX
        call nvtxRangePop() ! do np_row_curr = 0, np_rows-1
#endif
      enddo ! np = 0, np_rows-1
      
      ! Put the result into C
      c(1:l_rows,1:l_cols) = tmp1_full(1:l_rows,1:l_cols)
      
      deallocate(at_full, bt_full, stat=istat, errmsg=errorMessage)
      call check_alloc("elpa_multiply", "at_full, bt_full", istat, errorMessage)

    endif ! (a_transposed .and. b_transposed)
!_______________________________________________

    deallocate(aux_a_full, aux_b_full, tmp1_full, tmp2_full, stat=istat, errmsg=errorMessage)
    call check_alloc("elpa_multiply", "aux_a_full, tmp1_full, tmp2_full", istat, errorMessage)

    if (useGPU) then
      successGPU = gpu_free(aux_a_full_dev)
      check_dealloc_gpu("elpa_multiply: aux_a_full_dev", successGPU)

      successGPU = gpu_free(aux_b_full_dev)
      check_dealloc_gpu("elpa_multiply: aux_b_full_dev", successGPU)

      successGPU = gpu_free(tmp1_full_dev)
      check_dealloc_gpu("elpa_multiply: tmp1_full_dev", successGPU)
    endif

  endif ! isSquareGrid

!______________________________________________________________________________________________

!#define NONSQUARE_GRID
#if defined(NONSQUARE_GRID) /* PETEDEBUG */
  print *, "elpa_multiply NEW: NONSQUARE_GRID start"

  ! l_rows_max = l_rows
  call mpi_allreduce(l_rows, l_rows_max, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
  call mpi_allreduce(l_cols, l_cols_max, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
  nstor_block = greatest_common_divisor(l_rows_max, l_cols_max)

  allocate(aux_a_full(l_rows_max, nstor_block), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_multiply: aux_a_full", istat, errorMessage)

  allocate(aux_b_full(l_rows_max, l_cols_max), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_multiply: aux_b_full", istat, errorMessage)
  
  allocate(tmp1_full(nstor_block, l_cols_max), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_multiply: tmp1_full", istat, errorMessage)

  allocate(tmp2_full(nstor_block, l_cols_max), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_multiply: tmp2_full", istat, errorMessage)

  ! main loop: build up the result matrix by processor rows
  do np = 0, np_rows-1 ! PETERDEBUG: np -> np_row_curr or np_row_c (for matrix C)
#ifdef WITH_NVTX
    call nvtxRangePush("do np_row_curr = 0, np_rows-1")
#endif
    print *, "np = ", np ! PETERDEBUG

    ! In this turn, procs of row np assemble the result
    
    !np_bc=np ! np, that posesses the given column of a ! PETERDEBUG: cleanup this line
    aux_a_full = 0 ! PETERDEBUG
    aux_b_full = 0 ! PETERDEBUG

    n_blocks_loc_x = l_rows_max/nstor_block
    n_blocks_loc_y = l_cols_max/nstor_block
    do i_block_loc = 0, n_blocks_loc_x-1
      I_block_gl = n_blocks_loc_x*np +  i_block_loc
      J_block_gl = I_block_gl

      np_bc = J_block_gl/n_blocks_loc_y
      j_block_loc = J_block_gl - n_blocks_loc_y*np_bc

      if (my_pcol == np_bc) then
        nstor_block_cut = min(nstor_block, l_cols)
        aux_a_full(1:l_rows, 1:nstor_block_cut) = a(1:l_rows,j_block_loc*nstor_block_cut+1:j_block_loc*nstor_block+nstor_block_cut) ! nstor_block -> min(nstor_block, l_cols), otherwise mem leak
      endif
      aux_b_full(1:l_rows,1:l_cols) = b(1:l_rows,1:l_cols)

      ! Broadcast processor column
      call obj%timer%start("mpi_communication")
#ifdef WITH_NVTX
      call nvtxRangePush("MPI_Bcast(aux_a_full)")
#endif

      call MPI_Bcast(aux_a_full, int(l_rows_max*nstor_block,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                     int(np_bc,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)

#ifdef WITH_NVTX
      call nvtxRangePop() ! MPI_Bcast(aux_a_full)
#endif
      call obj%timer%stop("mpi_communication")

      call obj%timer%start("blas")
    
      call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', &
                        int(nstor_block,kind=BLAS_KIND), int(l_cols_max,kind=BLAS_KIND), int(l_rows_max,kind=BLAS_KIND), ONE, &
                        aux_a_full(1:l_rows_max,1:nstor_block), int(l_rows_max,kind=BLAS_KIND), &
                        aux_b_full(1:l_rows_max,1:l_cols_max) , int(l_rows_max,kind=BLAS_KIND), ZERO, &
                        tmp1_full, int(nstor_block,kind=BLAS_KIND))   

      call obj%timer%stop("blas")

      call obj%timer%start("mpi_communication")
      call mpi_reduce(tmp1_full, tmp2_full, int(nstor_block*l_cols_max,kind=MPI_KIND),  MPI_MATH_DATATYPE_PRECISION, &
                      MPI_SUM, int(np,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
      call obj%timer%stop("mpi_communication")

      ! Put the result into C
      if (my_prow==np) c(i_block_loc*nstor_block+1:i_block_loc*nstor_block+nstor,1:l_cols) = tmp2_full(1:nstor_block,1:l_cols)
    enddo ! i_block_loc = 0, n_blocks_loc_x-1
#ifdef WITH_NVTX
    call nvtxRangePop() ! do np = 0, np_rows-1
#endif
  enddo ! np = 0, np_rows-1

  deallocate(aux_a_full, tmp1_full, tmp2_full, stat=istat, errmsg=errorMessage)
  call check_alloc("elpa_multiply", "aux_a_full, tmp1_full, tmp2_full", istat, errorMessage)

#endif /* NONSQUARE_GRID PETEDEBUG */

!______________________________________________________________________________________________

  if (useGPU) then
#if !defined(DEVICE_POINTER)
    successGPU = gpu_free(b_dev)
    check_dealloc_gpu("elpa_multiply_a_b: b_dev", successGPU)
#if !defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) && !defined(WITH_SYCL_GPU_VERSION)
    successGPU = gpu_host_unregister(int(loc(b),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_multiply_a_b: b", successGPU)
#endif

    successGPU = gpu_free(c_dev)
    check_dealloc_gpu("elpa_multiply_a_b: c_dev", successGPU)

#else /* DEVICE_POINTER */
    !successGPU = gpu_free(b_dev)
    !check_dealloc_gpu("elpa_multiply_a_b: b_dev", successGPU)

    !num = ldc*ldcCols*size_of_datatype
    !successGPU = gpu_memcpy(cDev, c_dev, num,&
    !              gpuMemcpyDeviceToDevice)
    !check_memcpy_gpu("elpa_multiply: c_dev -> cDev", successGPU)

    !successGPU = gpu_free(c_dev)
    !check_dealloc_gpu("elpa_multiply_a_b: c_dev", successGPU)

#ifdef WITH_GPU_STREAMS
    successGPU = gpu_host_unregister(int(loc(a_tmp),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_multiply: a_tmp", successGPU)
#endif
    deallocate(a_tmp, stat=istat, errmsg=errorMessage)
    check_deallocate("elpa_multiply: a_tmp", istat, errorMessage)

    num = ldc*ldcCols*size_of_datatype
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_multiply: c_tmp to c", successGPU)

    successGPU = gpu_memcpy_async(cDev,int(loc(c_tmp),kind=c_intptr_t),num,&
                  gpuMemcpyHostToDevice, my_stream)
    check_memcpy_gpu("elpa_multiply: c_tmp -> c", successGPU)

    my_stream = obj%gpu_setup%my_stream
    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_multiply: c_tmp -> c", successGPU)
    ! synchronize streamsPerThread; maybe not neccessary
    successGPU = gpu_stream_synchronize()
    check_stream_synchronize_gpu("elpa_multiply: c_tmp -> c", successGPU)
#else
    successGPU = gpu_memcpy(cDev,int(loc(c_tmp),kind=c_intptr_t),num,&
                  gpuMemcpyHostToDevice)
    check_memcpy_gpu("elpa_multiply: c_tmp -> c", successGPU)
#endif
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_host_unregister(int(loc(c_tmp),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_multiply_a_b: c_tmp", successGPU)
#endif

    deallocate(c_tmp, stat=istat, errmsg=errorMessage)
    check_deallocate("elpa_multiply: c_tmp", istat, errorMessage)

#endif /* DEVICE_POINTER */
#if !defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) && !defined(WITH_SYCL_GPU_VERSION)
    nullify(aux_mat)
    !nullify(tmp1)

    successGPU = gpu_free_host(aux_host)
    check_host_dealloc_gpu("elpa_multiply_a_b: aux_host", successGPU)
#else
    deallocate(aux_mat, stat=istat, errmsg=errorMessage)
    check_deallocate("elpa_multiply: aux_mat", istat, errorMessage)

    !deallocate(tmp1, stat=istat, errmsg=errorMessage)
    !check_deallocate("elpa_multiply: tmp1", istat, errorMessage)
#endif

    successGPU = gpu_free(aux_mat_dev)
    check_dealloc_gpu("elpa_multiply_a_b: aux_mat_dev", successGPU)

    successGPU = gpu_free(tmp1_dev)
    check_dealloc_gpu("elpa_multiply_a_b: tmp1_dev", successGPU)

    successGPU = gpu_free(tmp2_dev)
    check_dealloc_gpu("elpa_multiply_a_b: tmp2_dev", successGPU)

    successGPU = gpu_free(aux_bc_dev)
    check_dealloc_gpu("elpa_multiply_a_b: aux_bc_dev", successGPU)

#if !defined(DEVICE_POINTER)
    successGPU = gpu_free(a_dev)
    check_dealloc_gpu("elpa_multiply: a_dev", successGPU)
#else
    !successGPU = gpu_free(a_dev)
    !check_dealloc_gpu("elpa_multiply: a_dev", successGPU)
#endif

  else ! useGPU
    deallocate(aux_mat, stat=istat, errmsg=errorMessage)
    check_deallocate("elpa_multiply: aux_mat", istat, errorMessage)
  endif ! useGPU

  deallocate(aux_bc, lrs_save, lre_save, stat=istat, errmsg=errorMessage)
  check_deallocate("elpa_multiply: aux_bc, lrs_save, lre_save", istat, errorMessage)

  call obj%timer%stop("elpa_multiply_&
  &MATH_DATATYPE&
  &_&
  &PRECISION&
  &"//gpuString)

#ifdef WITH_NVTX
  call nvtxRangePop() ! multiply
#endif
