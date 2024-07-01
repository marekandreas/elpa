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
! Author: P. Karpov, MPCDF

#include "../general/sanity.F90"
#include "../general/error_checking.inc"

! PETERDEBUG
!#define OLD_GENERIC_GRID
!#define SQUARE_GRID
!#define NONSQUARE_GRID

#undef USE_CCL_PXGEMM
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
#define USE_CCL_PXGEMM
#endif

  use elpa1_compute
  use elpa_mpi
  use precision
  use elpa_abstract_impl
  use, intrinsic :: iso_c_binding
  use elpa_gpu
  use mod_check_for_gpu
  use elpa_blas_interfaces
  use ELPA_utilities, only : local_index, greatest_common_divisor, least_common_multiple, &
                             check_deallocate_f, check_dealloc_gpu_f, &
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
#if defined(USE_CCL_PXGEMM)
  use elpa_ccl_gpu
#endif
  use multiply_a_b_gpu
  implicit none

#include "../../src/general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout)   :: obj

  character*1                                  :: uplo_a, uplo_c, trans_a, trans_b
  character(1, c_char)                         :: trans_a_cchar, trans_b_cchar
  integer(kind=ik), intent(in)                 :: ldb, ldbCols, ldc, ldcCols
  integer(kind=ik)                             :: lda, ldaCols
  integer(kind=ik)                             :: na, ncb
#ifndef DEVICE_POINTER
!forget device pointer case for the moment implement later <-- PETERDEBUG - take care of this
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
  integer(kind=ik)                             :: my_pdir, my_pdir_t, np_dirs, np_dirs_t ! PETERDEBUG_NEW
  integer(kind=ik)                             :: np_rows_fine, np_cols_fine, np_dirs_fine, np_fine, np_t_fine, np_bc_fine, &
                                                  np_1_fine, np_1t_fine, np_ab_fine, np_ab_t_fine, dnp_ab, dnp_ab_t ! PETERDEBUG_NEW
  integer(kind=ik)                             :: LCM, nblk_mult_rows, nblk_mult_cols, i_block_loc_fine, j_block_loc_fine ! PETERDEBUG_NEW
  integer(kind=ik)                             :: nblk_mult_rows_max, nblk_mult_cols_max, tail_loc, nblk_rows_min, nblk_cols_min ! PETERDEBUG_NEW
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
  MATH_DATATYPE(kind=rck), pointer, contiguous :: aux_a_full(:,:), aux_b_full(:,:), tmp1_full(:,:) ! PETERDEBUG
  MATH_DATATYPE(kind=rck), allocatable         :: aux_bc(:), tmp2(:,:)
  logical                                      :: wantDebug
  integer(kind=ik)                             :: istat, debug
  character(200)                               :: errorMessage
  character(20)                                :: gpuString
  logical                                      :: success, successGPU, successGPU2
  logical                                      :: useGPU
  logical                                      :: isSquareGrid
  integer(kind=c_int)                          :: numGPU, blocking, SM_count
  integer(kind=ik)                             :: mpi_comm_rows, mpi_comm_cols, mpi_comm_all
  integer(kind=ik)                             :: mpi_comm_dirs ! PETERDEBUG_NEW
  integer(kind=ik)                             :: nblk, error
  integer(kind=c_intptr_t)                     :: aux_bc_dev, aux_mat_dev, tmp1_dev, tmp2_dev
  integer(kind=c_intptr_t)                     :: aux_a_full_dev, aux_b_full_dev, tmp1_full_dev ! PETERDEBUG_NEW
!#ifndef DEVICE_POINTER
  integer(kind=c_intptr_t)                     :: a_dev
  integer(kind=c_intptr_t)                     :: b_dev
  integer(kind=c_intptr_t)                     :: c_dev
!#endif
  type(c_ptr)                                  :: aux_host
  integer(kind=c_intptr_t)                     :: num, num_a, num_b
  integer(kind=c_intptr_t)                     :: aux_off, b_off
  integer(kind=c_intptr_t), parameter          :: size_of_datatype = size_of_&
                                                            &PRECISION&
                                                            &_&
                                                            &MATH_DATATYPE

  integer(kind=c_intptr_t)                     :: gpuHandle, my_stream
  integer(kind=c_int)                          :: gpu_hermitian_multiply

  logical                                      :: useCCL
#if defined(USE_CCL_PXGEMM)
  integer(kind=c_intptr_t)                     :: ccl_comm_rows, ccl_comm_cols, ccl_comm_dirs
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
  call nvtxRangePush("pxgemm")
#endif

  success = .true.
  useGPU = .false.

  a_transposed = .false.
  b_transposed = .false.
  trans_a_cchar = 'N'
  trans_b_cchar = 'N'
  if (trans_a=='t' .or. trans_a=='T' .or. trans_a=='c' .or. trans_a=='C') then
    a_transposed = .true.
    trans_a_cchar = BLAS_TRANS_OR_CONJ
  endif
  if (trans_b=='t' .or. trans_b=='T' .or. trans_b=='c' .or. trans_b=='C') then
    b_transposed = .true.
    trans_b_cchar = BLAS_TRANS_OR_CONJ
  endif
  print *, "a_transposed = ", a_transposed ! PETERDEBUG
  print *, "b_transposed = ", b_transposed
  print *, "trans_a = ", trans_a ! PETERDEBUG
  print *, "trans_b = ", trans_b

  call obj%get("debug", debug, error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "ELPA_PXGEMM_AB: Problem getting option for debug settings. Aborting..."
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
  if (obj%is_set("gpu_hermitian_multiply") == 1) then ! PETERDEBUG: add gpu_pxgemm_multiply keyword
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

  call obj%timer%start("elpa_pxgemm_&
  &MATH_DATATYPE&
  &_&
  &PRECISION&
  &"//gpuString)

  na      = obj%na
  nblk    = obj%nblk
  lda     = obj%local_nrows
  ldaCols = obj%local_ncols

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
      write(error_unit,*) "ELPA_PXGEMM: Problem in getting keyword 'blocking_in_multiply'. Aborting..."
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

  useCCL = .false.
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

    SM_count = obj%gpu_setup%gpuSMcount
    
#if defined(USE_CCL_PXGEMM)
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
#endif /* defined(USE_CCL_PXGEMM) */

#if !defined(DEVICE_POINTER)
    num = ldc*ldcCols*size_of_datatype
    successGPU = gpu_malloc(c_dev, num)
    check_alloc_gpu("elpa_pxgemm: c_dev", successGPU)
    ! no copy from c to c_dev needed since c will be overwritten anyway
#endif

#if !defined(DEVICE_POINTER)
    ! copy b to b_dev
    num = ldb*ldbCols*size_of_datatype
    successGPU = gpu_malloc(b_dev, num)
    check_alloc_gpu("elpa_pxgemm: b_dev", successGPU)

#if !defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) && !defined(WITH_SYCL_GPU_VERSION)
    successGPU = gpu_host_register(int(loc(b),kind=c_intptr_t),num,&
                  gpuHostRegisterDefault)
#endif    

    check_host_register_gpu("elpa_pxgemm: b", successGPU)
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    call gpu_memcpy_async_and_stream_synchronize &
    ("elpa_pxgemm: b to b_dev", b_dev, 0_c_intptr_t, &
                                       b(1:ldb,1:ldbCols), &
                                       1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
    successGPU = gpu_memcpy(b_dev,int(loc(b),kind=c_intptr_t),num,&
                  gpuMemcpyHostToDevice)
    check_memcpy_gpu("elpa_pxgemm: b to b_dev", successGPU)
#endif

#else /* DEVICE_POINTER */

#endif /* DEVICE_POINTER */

    num = l_rows*nblk_mult*size_of_datatype
#if !defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) && !defined(WITH_SYCL_GPU_VERSION)
    successGPU = gpu_malloc_host(aux_host, num) ! aux_host is needed, because pinning host memory can be done only for 1D arrays
    check_host_alloc_gpu("elpa_pxgemm: aux_host", successGPU)
    call c_f_pointer(aux_host, aux_mat, (/l_rows,nblk_mult/))
#else
    allocate(aux_mat(l_rows, nblk_mult), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_pxgemm: aux_mat", istat, errorMessage)
#endif

    successGPU = gpu_malloc(aux_mat_dev, num)
    check_alloc_gpu("elpa_pxgemm: aux_mat_dev", successGPU)

    num = nblk_mult*l_cols*size_of_datatype
    successGPU = gpu_malloc(tmp1_dev, num)
    check_alloc_gpu("elpa_pxgemm: tmp1_dev", successGPU)

    num = nblk_mult*l_cols*size_of_datatype
    successGPU = gpu_malloc(tmp2_dev, num)
    check_alloc_gpu("elpa_pxgemm: tmp2_dev", successGPU)

  else ! useGPU
    allocate(aux_mat(l_rows,nblk_mult), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_pxgemm: aux_mat", istat, errorMessage)
  endif ! useGPU

  allocate(aux_bc(l_rows*nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_pxgemm: aux_bc", istat, errorMessage)

  allocate(lrs_save(nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_pxgemm: lrs_save", istat, errorMessage)

  allocate(lre_save(nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_pxgemm: lre_save", istat, errorMessage)

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
    num = lda*ldaCols*size_of_datatype
    successGPU = gpu_malloc(a_dev, num)
    check_alloc_gpu("elpa_pxgemm: a_dev", successGPU)
#endif

    num = l_rows*nblk*size_of_datatype
    successGPU = gpu_malloc(aux_bc_dev, num)
    check_alloc_gpu("elpa_pxgemm: aux_bc_dev", successGPU)

    num = lda*ldaCols*size_of_datatype
#if !defined(DEVICE_POINTER)
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    call gpu_memcpy_async_and_stream_synchronize &
    ("elpa_pxgemm: a to a_dev", a_dev, 0_c_intptr_t, &
                                       a(1:lda,1:ldaCols), &
                                       1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
    successGPU = gpu_memcpy(a_dev, int(loc(a),kind=c_intptr_t), &
                  num, gpuMemcpyHostToDevice)
    check_memcpy_gpu("elpa_pxgemm: a to a_dev", successGPU)
#endif
#endif /* DEVICE_POINTER */
  endif !useGPU

!______________________________________________________________________________________________

  ! if (useGPU) then
  !   print *, "elpa_pxgemm NEW: isSquareGrid and useGPU is not imlemented yet" ! PETERDEBUG
  !   stop 1
  ! endif

  print *, "elpa_pxgemm NEW: start" ! PETERDEBUG

  ! l_rows_max = l_rows
  call mpi_allreduce(l_rows, l_rows_max, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
  call mpi_allreduce(l_cols, l_cols_max, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
  call mpi_allreduce(l_rows, l_rows_min, 1_MPI_KIND, MPI_INTEGER, MPI_MIN, int(mpi_comm_all,kind=MPI_KIND), mpierr)
  call mpi_allreduce(l_cols, l_cols_min, 1_MPI_KIND, MPI_INTEGER, MPI_MIN, int(mpi_comm_all,kind=MPI_KIND), mpierr)
  
  !nblk_mult = l_rows_max ! = l_cols_max
  nblk_mult = greatest_common_divisor(l_rows_max, l_cols_max)



  if (isSquareGrid) then

    allocate(aux_a_full(l_rows_max, nblk_mult), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_pxgemm: aux_a_full", istat, errorMessage)
  
    allocate(aux_b_full(nblk_mult, l_cols_max), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_pxgemm: aux_b_full", istat, errorMessage)
    
    allocate(tmp1_full(l_rows_max, l_cols_max), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_pxgemm: tmp1_full", istat, errorMessage)

    ! PETERDEBUG: is it possible to use the original GPU memory, without copying?
    if (useGPU) then
      successGPU = gpu_malloc(aux_a_full_dev, nblk_mult*nblk_mult*size_of_datatype)
      check_alloc_gpu("elpa_pxgemm: aux_a_full_dev", successGPU)

      successGPU = gpu_malloc(aux_b_full_dev, nblk_mult*nblk_mult*size_of_datatype)
      check_alloc_gpu("elpa_pxgemm: aux_b_full_dev", successGPU)

      successGPU = gpu_malloc(tmp1_full_dev, nblk_mult*nblk_mult*size_of_datatype)
      check_alloc_gpu("elpa_pxgemm: tmp1_full_dev", successGPU)
    endif
!_______________________________________________

    if (.not. a_transposed .and. b_transposed .or. &
        a_transposed .and. .not. b_transposed ) then
      print *, "elpa_pxgemm NEW: SQUARE_GRID start: ( a_transposed XOR b_transposed)" ! PETERDEBUG
      
      ! dir = row/col for TN/NT
      if (a_transposed) then
        my_pdir = my_prow
        np_dirs = np_rows
        mpi_comm_dirs = mpi_comm_rows
#if defined(USE_CCL_PXGEMM)
        ccl_comm_dirs = obj%gpu_setup%ccl_comm_rows
#endif
      else if (b_transposed) then
        my_pdir = my_pcol
        np_dirs = np_cols
        mpi_comm_dirs = mpi_comm_cols
#if defined(USE_CCL_PXGEMM)
        ccl_comm_dirs = obj%gpu_setup%ccl_comm_cols
#endif
      endif

      call obj%timer%start("main_loop_square_grid_tn_nt")

      ! main loop: build up the result matrix by processor rows/cols for TN/NT
      do np = 0, np_dirs-1
#ifdef WITH_NVTX
        call nvtxRangePush("do np = 0, np_dirs-1")
#endif
        print *, "np = ", np ! PETERDEBUG

        ! In this turn, procs of row/col np assemble the result for TN/NT case
        
        np_t=np ! np, that posesses the given "col of a"/"row of b" for TN/NT

        if (a_transposed) then
          if (useGPU) then
            if (np_t == my_pcol) call gpu_copy_aux_full(PRECISION_CHAR, aux_a_full_dev, a_dev, l_rows, l_cols, &
                                                        l_rows_max, lda, debug, my_stream)
            call gpu_copy_aux_full(PRECISION_CHAR, aux_b_full_dev, b_dev, l_rows, l_cols, nblk_mult, ldb, debug, my_stream)
          else ! useGPU
            if (np_t == my_pcol) aux_a_full(1:l_rows,1:l_cols) = a(1:l_rows,1:l_cols)
            aux_b_full(1:l_rows,1:l_cols) = b(1:l_rows,1:l_cols)
          endif ! useGPU
        else if (b_transposed) then
          if (useGPU) then
            if (np_t == my_prow) call gpu_copy_aux_full(PRECISION_CHAR, aux_b_full_dev, b_dev, l_rows, l_cols, &
                                                        nblk_mult, ldb, debug, my_stream)
            call gpu_copy_aux_full(PRECISION_CHAR, aux_a_full_dev, a_dev, l_rows, l_cols, l_rows_max, lda, debug, my_stream)
          
          else
            if (np_t == my_prow) aux_b_full(1:l_rows,1:l_cols) = b(1:l_rows,1:l_cols)
            aux_a_full(1:l_rows,1:l_cols) = a(1:l_rows,1:l_cols) ! aux_a_full -> aux_ab_nontransposed_full
          endif ! useGPU
        endif

        ! aux_a_full/aux_b_full -> aux_ab_trans (aux_ab) ! auxillary buffer for a matrix that is     to be transposed: a/b in TN/NT case
        ! aux_b_full/aux_a_full -> aux_ab_nontr (aux_ba) ! auxillary buffer for a matrix that is not to be transposed: b/a in TN/NT case

        ! copy data to host for bcast, if needed
        if (useGPU .and. .not. useCCL) then
          num = nblk_mult*nblk_mult
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          if (a_transposed) then
            call gpu_memcpy_async_and_stream_synchronize &
                ("elpa_pxgemm: aux_a_full_dev -> aux_a_full", aux_a_full_dev, 0_c_intptr_t, aux_a_full(1:nblk_mult,1:nblk_mult), &
                  1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .false., .false.)
          else if (b_transposed) then
            call gpu_memcpy_async_and_stream_synchronize &
                ("elpa_pxgemm: aux_b_full_dev -> aux_b_full", aux_b_full_dev, 0_c_intptr_t, aux_b_full(1:nblk_mult,1:nblk_mult), &
                  1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
          endif
#else
          if (a_transposed) then
            successGPU = gpu_memcpy(int(loc(aux_a_full),kind=c_intptr_t), aux_a_full_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
            check_memcpy_gpu("elpa_pxgemm: aux_a_full_dev -> aux_a_full", successGPU)
          else if (b_transposed) then
            successGPU = gpu_memcpy(int(loc(aux_b_full),kind=c_intptr_t), aux_b_full_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
            check_memcpy_gpu("elpa_pxgemm: aux_b_full_dev -> aux_b_full", successGPU)
          endif
#endif  
        endif ! (useGPU .and. .not. useCCL)


        ! Broadcast processor column
        if (useCCL) then
#ifdef USE_CCL_PXGEMM
#ifdef WITH_NVTX
          call nvtxRangePush("ccl_bcast aux_a/b_full")
#endif
          call obj%timer%start("ccl_bcast")

          my_stream = obj%gpu_setup%my_stream

          if (a_transposed) then
            ccl_comm_cols = obj%gpu_setup%ccl_comm_cols
            successGPU = ccl_bcast(aux_a_full_dev, aux_a_full_dev, int(k_datatype*nblk_mult*nblk_mult,kind=c_size_t), &
                                   cclDatatype, int(np_t,kind=c_int), ccl_comm_cols, my_stream)
          else if (b_transposed) then
            ccl_comm_rows = obj%gpu_setup%ccl_comm_rows
            successGPU = ccl_bcast(aux_b_full_dev, aux_b_full_dev, int(k_datatype*nblk_mult*nblk_mult,kind=c_size_t), &
                                   cclDatatype, int(np_t,kind=c_int), ccl_comm_rows, my_stream)
          endif

          if (.not. successGPU) then
            print *,"Error in ccl_bcast"
            stop 1
          endif

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_pxgemm: ccl_bcast", successGPU)

          call obj%timer%stop("ccl_bcast")
#ifdef WITH_NVTX
          call nvtxRangePop() ! ccl_bcast aux_a/b_full
#endif
#endif /* USE_CCL_PXGEMM */
        else ! useCCL
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
        endif ! useCCL


        ! copy data back to device, if needed
        if (useGPU .and. .not. useCCL) then
          num = nblk_mult*nblk_mult
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          if (a_transposed) then
            call gpu_memcpy_async_and_stream_synchronize &
                  ("elpa_pxgemm: aux_a_full -> aux_a_full_dev", aux_a_full_dev, 0_c_intptr_t, aux_a_full(1:nblk_mult,1:nblk_mult), &
                  1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
          else if (b_transposed) then
            call gpu_memcpy_async_and_stream_synchronize &
                  ("elpa_pxgemm: aux_b_full -> aux_b_full_dev", aux_b_full_dev, 0_c_intptr_t, aux_b_full(1:nblk_mult,1:nblk_mult), &
                  1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
          endif
#else
          if (a_transposed) then
            successGPU = gpu_memcpy(aux_a_full_dev, int(loc(aux_a_full),kind=c_intptr_t), num*size_of_datatype, gpuMemcpyHostToDevice)
            check_memcpy_gpu("elpa_pxgemm: aux_a_full -> aux_a_full_dev", successGPU)
          else if (b_transposed) then
            successGPU = gpu_memcpy(aux_b_full_dev, int(loc(aux_b_full),kind=c_intptr_t), num*size_of_datatype, gpuMemcpyHostToDevice)
            check_memcpy_gpu("elpa_pxgemm: aux_b_full -> aux_b_full_dev", successGPU)
          endif
#endif
        endif ! (useGPU .and. .not. useCCL)


        if (useGPU) then
          gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
#ifdef WITH_NVTX
          call nvtxRangePush("gpublas")
#endif
          call obj%timer%start("gpublas")
          call gpublas_PRECISION_GEMM(trans_a_cchar, trans_b_cchar, &
                            nblk_mult, nblk_mult, nblk_mult, ONE, &
                            aux_a_full_dev, nblk_mult, &
                            aux_b_full_dev, nblk_mult, ZERO, &
                            tmp1_full_dev , nblk_mult, gpuHandle)
          if (wantDebug) successGPU = gpu_DeviceSynchronize()
          call obj%timer%stop("gpublas")
#ifdef WITH_NVTX
          call nvtxRangePop() ! gpublas
#endif
        else ! useGPU
          call obj%timer%start("blas")
          call PRECISION_GEMM(trans_a, trans_b, &
                              int(nblk_mult, kind=BLAS_KIND), &
                              int(nblk_mult, kind=BLAS_KIND), &
                              int(nblk_mult, kind=BLAS_KIND), ONE, &
                              aux_a_full, int(nblk_mult, kind=BLAS_KIND), &
                              aux_b_full, int(nblk_mult, kind=BLAS_KIND), ZERO, &
                              tmp1_full , int(nblk_mult, kind=BLAS_KIND))
          call obj%timer%stop("blas")
        endif ! useGPU


        ! copy data to host for mpi_reduce, if needed
        if (useGPU .and. .not. useCCL) then
          num = nblk_mult*nblk_mult
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_pxgemm: tmp1_full_dev -> tmp1_full", tmp1_full_dev, 0_c_intptr_t, tmp1_full(1:nblk_mult,1:nblk_mult), &
                1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .false., .false.)
#else
          successGPU = gpu_memcpy(int(loc(tmp1_full),kind=c_intptr_t), tmp1_full_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_pxgemm: tmp1_full_dev -> tmp1_full", successGPU)
#endif  
        endif ! (useGPU .and. .not. useCCL)


        if (useCCL) then
#ifdef USE_CCL_PXGEMM
#ifdef WITH_NVTX
          call nvtxRangePush("ccl_reduce tmp1_full_dev")
#endif
          call obj%timer%start("ccl_reduce")

          my_stream = obj%gpu_setup%my_stream

          successGPU  = ccl_reduce(tmp1_full_dev, tmp1_full_dev, int(k_datatype*nblk_mult*nblk_mult,kind=c_size_t), cclDatatype, &
                                  cclSum, int(np, kind=c_int), ccl_comm_dirs, my_stream)
          
          if (.not. successGPU) then
            print *,"Error in ccl_reduce"
            stop 1
          endif

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_pxgemm: ccl_bcast", successGPU)

          call obj%timer%stop("ccl_bcast")
#ifdef WITH_NVTX
          call nvtxRangePop() ! ccl_bcast aux_a_full_dev, aux_b_full_dev
#endif
#endif /* USE_CCL_PXGEMM */        
        else ! useCCL
          call obj%timer%start("mpi_communication")
          if (my_pdir==np) then
            call mpi_reduce(MPI_IN_PLACE, tmp1_full, int(nblk_mult*nblk_mult,kind=MPI_KIND),  MPI_MATH_DATATYPE_PRECISION, &
                            MPI_SUM, int(np,kind=MPI_KIND), int(mpi_comm_dirs,kind=MPI_KIND), mpierr)
          else
            call mpi_reduce(tmp1_full   , tmp1_full, int(nblk_mult*nblk_mult,kind=MPI_KIND),  MPI_MATH_DATATYPE_PRECISION, &
                            MPI_SUM, int(np,kind=MPI_KIND), int(mpi_comm_dirs,kind=MPI_KIND), mpierr)
          endif
          call obj%timer%stop("mpi_communication")
        endif ! useCCL

        ! Put the result into C
        ! PETERDEBUG: we put the result to c_dev (ifdef DEVICE_POINTER) or to c (if not DEVICE_POINTER)
        if (my_pdir==np) then
          if (useCCL) then
            call gpu_copy_aux_full(PRECISION_CHAR, c_dev, tmp1_full_dev, l_rows, l_cols, ldc, nblk_mult, debug, my_stream)
          else ! useCCL
            c(1:l_rows,1:l_cols) = tmp1_full(1:l_rows,1:l_cols)
          endif ! useCCL
        endif

#ifdef WITH_NVTX
        call nvtxRangePop() ! do np = 0, np_dirs-1
#endif
      enddo ! np = 0, np_dirs-1
      call obj%timer%stop("main_loop_square_grid_tn_nt")

#if !defined(DEVICE_POINTER)
      if (useCCL) then
        num = ldc*ldcCols
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        call gpu_memcpy_async_and_stream_synchronize &
            ("elpa_pxgemm: c_dev -> c", c_dev, 0_c_intptr_t, c(1:ldc,1:ldcCols), &
              1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
        successGPU = gpu_memcpy(int(loc(c),kind=c_intptr_t), c_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("elpa_pxgemm: c_dev -> c", successGPU)
#endif
      endif ! useCCL
#endif /* !defined(DEVICE_POINTER) */

    endif ! (.not. a_transposed .and. b_transposed)

!_______________________________________________

    if ((.not. a_transposed) .and. (.not. b_transposed)) then
      print *, "elpa_pxgemm NEW: SQUARE_GRID start: (.not. a_transposed) .and. (.not. b_transposed)" ! PETERDEBUG

      call obj%timer%start("main_loop_square_grid_nn")
      
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
              ("elpa_pxgemm: aux_a_full_dev -> aux_a_full", aux_a_full_dev, 0_c_intptr_t, aux_a_full(1:nblk_mult,1:nblk_mult), &
                1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .false., .false.)
          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_pxgemm: aux_b_full_dev -> aux_b_full", aux_b_full_dev, 0_c_intptr_t, aux_b_full(1:nblk_mult,1:nblk_mult), &
                1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
          successGPU = gpu_memcpy(int(loc(aux_a_full),kind=c_intptr_t), aux_a_full_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_pxgemm: aux_a_full_dev -> aux_a_full", successGPU)

          successGPU = gpu_memcpy(int(loc(aux_b_full),kind=c_intptr_t), aux_b_full_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_pxgemm: aux_b_full_dev -> aux_b_full", successGPU)
#endif  
        endif ! (useGPU .and. .not. useCCL)

        ! Broadcast processor column
        if (useCCL) then
#ifdef USE_CCL_PXGEMM
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
          check_stream_synchronize_gpu("elpa_pxgemm: ccl_bcast", successGPU)

          call obj%timer%stop("ccl_bcast")
#ifdef WITH_NVTX
          call nvtxRangePop() ! ccl_bcast aux_a_full_dev, aux_b_full_dev
#endif
#endif /* USE_CCL_PXGEMM */
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
              ("elpa_pxgemm: aux_a_full -> aux_a_full_dev", aux_a_full_dev, 0_c_intptr_t, aux_a_full(1:nblk_mult,1:nblk_mult), &
                1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
          call gpu_memcpy_async_and_stream_synchronize &
                ("elpa_pxgemm: aux_b_full -> aux_b_full_dev", aux_b_full_dev, 0_c_intptr_t, aux_b_full(1:nblk_mult,1:nblk_mult), &
                1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
          successGPU = gpu_memcpy(aux_a_full_dev, int(loc(aux_a_full),kind=c_intptr_t), num*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_pxgemm: aux_a_full -> aux_a_full_dev", successGPU)
          
          successGPU = gpu_memcpy(aux_b_full_dev, int(loc(aux_b_full),kind=c_intptr_t), num*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_pxgemm: aux_b_full -> aux_b_full_dev", successGPU)
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

      call obj%timer%stop("main_loop_square_grid_nn")
      
      ! Put the result into C
      if (useGPU) then
        num = nblk_mult*nblk_mult
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        call gpu_memcpy_async_and_stream_synchronize &
            ("elpa_pxgemm: tmp1_full_dev -> tmp1_full", tmp1_full_dev, 0_c_intptr_t, tmp1_full(1:nblk_mult,1:nblk_mult), &
              1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
        successGPU = gpu_memcpy(int(loc(tmp1_full),kind=c_intptr_t), tmp1_full_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("elpa_pxgemm: tmp1_full_dev -> tmp1_full", successGPU)
#endif
      endif ! useGPU

      c(1:l_rows,1:l_cols) = tmp1_full(1:l_rows,1:l_cols)

    endif ! (a_transposed .and. b_transposed)
!_______________________________________________

    if ((a_transposed) .and. (b_transposed)) then

      allocate(at_full(l_rows_max, l_cols_max), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_pxgemm: at_max", istat, errorMessage)
  
      allocate(bt_full(l_rows_max, l_cols_max), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_pxgemm: bt_full", istat, errorMessage)
  
      !call mpi_allreduce(nblk_mult, na_max, 1_MPI_KIND, MPI_INTEGER, MPI_SUM, int(mpi_comm_all,kind=MPI_KIND), mpierr)

      print *, "elpa_pxgemm NEW: SQUARE_GRID start: (a_transposed) .and. (b_transposed)" ! PETERDEBUG

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
      call check_alloc("elpa_pxgemm", "at_full, bt_full", istat, errorMessage)

    endif ! (a_transposed .and. b_transposed)
  endif ! isSquareGrid

! _________________________________________________________________________________________________________________________________

  if (.not. isSquareGrid) then

    LCM = least_common_multiple(np_rows, np_cols)*nblk
    nblk_mult_rows_max = (l_rows_max*np_rows+LCM-1)/LCM*nblk
    nblk_mult_cols_max = (l_cols_max*np_cols+LCM-1)/LCM*nblk
    nblk_mult = max(nblk_mult_rows_max, nblk_mult_cols_max)
    print *, "LCM = ", LCM ! PETERDEBUG
    print *, "l_rows_max", l_rows_max ! PETERDEBUG
    print *, "l_cols_max", l_cols_max ! PETERDEBUG
    print *, "l_rows", l_rows ! PETERDEBUG
    print *, "l_cols", l_cols ! PETERDEBUG
    print *, "np_rows", np_rows ! PETERDEBUG
    print *, "np_cols", np_cols ! PETERDEBUG
    print *, "nblk_mult = ", nblk_mult ! PETERDEBUG

    np_rows_fine = least_common_multiple(np_rows, np_cols)
    np_cols_fine = np_rows_fine

! ___________________________________________________________________

    if ((.not. a_transposed) .and. (.not. b_transposed)) then
      
      allocate(aux_a_full(l_rows, nblk_mult), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_pxgemm: aux_a_full", istat, errorMessage)
    
      allocate(aux_b_full(nblk_mult, l_cols), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_pxgemm: aux_b_full", istat, errorMessage)
      
      allocate(tmp1_full(l_rows, l_cols), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_pxgemm: tmp1_full", istat, errorMessage)
  
      if (useGPU) then
        successGPU = gpu_malloc(aux_a_full_dev, l_rows*nblk_mult*size_of_datatype)
        check_alloc_gpu("elpa_pxgemm: aux_a_full_dev", successGPU)
  
        successGPU = gpu_malloc(aux_b_full_dev, nblk_mult*l_cols*size_of_datatype)
        check_alloc_gpu("elpa_pxgemm: aux_b_full_dev", successGPU)
  
        successGPU = gpu_malloc(tmp1_full_dev, l_rows*l_cols*size_of_datatype)
        check_alloc_gpu("elpa_pxgemm: tmp1_full_dev", successGPU)
      endif

      call obj%timer%start("main_loop_nn")
      print *, "elpa_pxgemm NEW: NON-SQUARE_GRID start: (.not. a_transposed) .and. (.not. b_transposed)" ! PETERDEBUG

      ! main loop: iterate through np_fine, which are "virtual" process rows for matrix A and process cols for matrix B
      do np_fine = 0, np_rows_fine-1 ! np_rows_fine=np_cols_fine
#ifdef WITH_NVTX
        call nvtxRangePush("np_fine = 0, np_rows_fine-1")
#endif
        print *, "np_fine = ", np_fine ! PETERDEBUG

        ! In this turn, procs of row np_fine assemble the result
        
        np_bc_fine=np_fine ! np_fine, that posesses the given column of a

        np    = mod(np_fine, np_rows)
        np_bc = mod(np_bc_fine, np_cols)

        nblk_mult_cols = l_cols/(LCM/np_cols)*nblk
        tail_loc = mod(l_cols, LCM/np_cols)/nblk ! number of local nblk-blocks in the last (incomplete) LCM block
        if (np_bc_fine/np_cols > tail_loc) then
          nblk_mult_cols = nblk_mult_cols + 0
        else if (np_bc_fine/np_cols < tail_loc) then
          nblk_mult_cols = nblk_mult_cols + nblk
        else ! np_bc_fine/np_cols == tail_loc
          nblk_mult_cols = nblk_mult_cols + mod(l_cols, nblk)
        endif

        nblk_mult_rows = l_rows/(LCM/np_rows)*nblk
        tail_loc = mod(l_rows, LCM/np_rows)/nblk ! number of local nblk-blocks in the last (incomplete) LCM block
        if (np_fine/np_rows > tail_loc) then
          nblk_mult_rows = nblk_mult_rows + 0
        else if (np_fine/np_rows < tail_loc) then
          nblk_mult_rows = nblk_mult_rows + nblk
        else ! np_fine/np_rows == tail_loc
          nblk_mult_rows = nblk_mult_rows + mod(l_rows, nblk)
        endif

        if (mod(np_bc_fine,np_cols) == my_pcol) then
          if (useGPU) then
            call gpu_copy_and_set_zeros_aux_a_full(PRECISION_CHAR, a_dev, aux_a_full_dev, l_rows, l_cols, &
                                                   nblk_mult_cols, nblk, np_bc_fine, np_cols_fine, np_cols, debug, my_stream)
          else ! useGPU
            do j_block_loc_fine = 0, nblk_mult_cols/nblk-1
              j_block_loc = (np_bc_fine + j_block_loc_fine*np_cols_fine)/np_cols
              aux_a_full(1:l_rows, 1+j_block_loc_fine*nblk: nblk+j_block_loc_fine*nblk) = &
                       a(1:l_rows, 1+j_block_loc*nblk     : nblk+j_block_loc*nblk)
            enddo ! j_block_loc_fine
            if (mod(nblk_mult_cols,nblk) /= 0) then ! last incomplete nblk-block
              j_block_loc = (np_bc_fine + j_block_loc_fine*np_cols_fine)/np_cols
              aux_a_full(1:l_rows, 1+j_block_loc_fine*nblk: mod(nblk_mult_cols,nblk)+j_block_loc_fine*nblk) = &
                       a(1:l_rows, 1+j_block_loc*nblk     : mod(nblk_mult_cols,nblk)+j_block_loc*nblk)
            endif
          endif ! useGPU
        endif
        if (mod(np_fine,np_rows) == my_prow) then
          if (useGPU) then
            call gpu_copy_and_set_zeros_aux_b_full(PRECISION_CHAR, b_dev, aux_b_full_dev, l_rows, l_cols, nblk_mult, &
                                                   nblk_mult_rows, nblk, np_fine, np_rows_fine, np_rows, SM_count, debug, my_stream)
          else ! useGPU
            do i_block_loc_fine = 0, nblk_mult_rows/nblk-1
              i_block_loc = (np_fine + i_block_loc_fine*np_rows_fine)/np_rows
              aux_b_full(1+i_block_loc_fine*nblk: nblk+i_block_loc_fine*nblk, 1:l_cols) = &
                       b(1+i_block_loc*nblk     : nblk+i_block_loc*nblk     , 1:l_cols)
            enddo ! i_block_loc_fine
            if (mod(nblk_mult_rows,nblk) /= 0) then ! last incomplete nblk-block
              i_block_loc = (np_fine + i_block_loc_fine*np_rows_fine)/np_rows
              aux_b_full(1+i_block_loc_fine*nblk: mod(nblk_mult_rows,nblk)+i_block_loc_fine*nblk, 1:l_cols) = &
                       b(1+i_block_loc*nblk     : mod(nblk_mult_rows,nblk)+i_block_loc*nblk     , 1:l_cols)
            endif
          endif ! useGPU
        endif

        ! copy data to host for bcast, if needed
        num_a = l_rows*nblk_mult
        num_b = nblk_mult*l_cols
        if (useGPU .and. .not. useCCL) then
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_pxgemm: aux_a_full_dev -> aux_a_full", aux_a_full_dev, 0_c_intptr_t, aux_a_full(1:l_rows,1:nblk_mult), &
                1, 1, num_a*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .false., .false.)
          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_pxgemm: aux_b_full_dev -> aux_b_full", aux_b_full_dev, 0_c_intptr_t, aux_b_full(1:nblk_mult,1:l_cols), &
                1, 1, num_b*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
          successGPU = gpu_memcpy(int(loc(aux_a_full),kind=c_intptr_t), aux_a_full_dev, num_a*size_of_datatype, & 
                                  gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_pxgemm: aux_a_full_dev -> aux_a_full", successGPU)

          successGPU = gpu_memcpy(int(loc(aux_b_full),kind=c_intptr_t), aux_b_full_dev, num_b*size_of_datatype, & 
                                  gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_pxgemm: aux_b_full_dev -> aux_b_full", successGPU)
#endif  
        endif ! (useGPU .and. .not. useCCL)

        ! Broadcast processor column
        if (useCCL) then
#ifdef USE_CCL_PXGEMM
#ifdef WITH_NVTX
          call nvtxRangePush("ccl_bcast aux_a_full_dev, aux_b_full_dev")
#endif
          call obj%timer%start("ccl_bcast")

          my_stream = obj%gpu_setup%my_stream
          ccl_comm_cols = obj%gpu_setup%ccl_comm_cols

          successGPU  = ccl_bcast(aux_a_full_dev, aux_a_full_dev, int(k_datatype*l_rows*nblk_mult,kind=c_size_t), cclDatatype, &
                                  int(np_bc,kind=c_int), ccl_comm_cols, my_stream)

          successGPU2 = ccl_bcast(aux_b_full_dev, aux_b_full_dev, int(k_datatype*nblk_mult*l_cols,kind=c_size_t), cclDatatype, &
                                  int(np   ,kind=c_int), ccl_comm_rows, my_stream)
          
          if (.not. (successGPU .and. successGPU2)) then
            print *,"Error in ccl_bcast"
            stop 1
          endif

          call MPI_Bcast(nblk_mult_rows, 1_MPI_KIND, MPI_INTEGER, &
                         int(np   ,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_pxgemm: ccl_bcast", successGPU)

          call obj%timer%stop("ccl_bcast")
#ifdef WITH_NVTX
          call nvtxRangePop() ! ccl_bcast aux_a_full_dev, aux_b_full_dev
#endif
#endif /* USE_CCL_PXGEMM */
        else ! useCCL
          call obj%timer%start("mpi_communication")
#ifdef WITH_NVTX
          call nvtxRangePush("MPI_Bcast: aux_a_full, aux_b_full")
#endif
          call MPI_Bcast(aux_a_full, int(l_rows*nblk_mult,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                        int(np_bc,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)

          call MPI_Bcast(aux_b_full, int(nblk_mult*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                        int(np   ,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
          call MPI_Bcast(nblk_mult_rows, 1_MPI_KIND, MPI_INTEGER, &
                        int(np   ,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
#ifdef WITH_NVTX
          call nvtxRangePop() ! MPI_Bcast: aux_a_full, aux_b_full
#endif
          call obj%timer%stop("mpi_communication")
        endif ! useCCL


        ! copy data back to device, if needed
        if (useGPU .and. .not. useCCL) then
          num_a = l_rows*nblk_mult
          num_b = nblk_mult*l_cols
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_pxgemm: aux_a_full -> aux_a_full_dev", aux_a_full_dev, 0_c_intptr_t, aux_a_full(1:l_rows,1:nblk_mult), &
                1, 1, num_a*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_pxgemm: aux_b_full -> aux_b_full_dev", aux_b_full_dev, 0_c_intptr_t, aux_b_full(1:nblk_mult,1:l_cols), &
                1, 1, num_b*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
          successGPU = gpu_memcpy(aux_a_full_dev,int(loc(aux_a_full),kind=c_intptr_t), num_a*size_of_datatype, &
                                  gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_pxgemm: aux_a_full -> aux_a_full_dev", successGPU)
          
          successGPU = gpu_memcpy(aux_b_full_dev,int(loc(aux_b_full),kind=c_intptr_t), num_b*size_of_datatype, &
                                  gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_pxgemm: aux_b_full -> aux_b_full_dev", successGPU)
#endif
        endif ! (useGPU .and. .not. useCCL)
        
        beta = ZERO
        if (np_fine>0) beta = ONE
        if (useGPU) then
          gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
#ifdef WITH_NVTX
          call nvtxRangePush("gpublas")
#endif
          call obj%timer%start("gpublas")
          call gpublas_PRECISION_GEMM('N', 'N', &
                            l_rows, l_cols, nblk_mult_rows, ONE, &
                            aux_a_full_dev, l_rows, &
                            aux_b_full_dev, nblk_mult, beta, &
                            tmp1_full_dev , l_rows, gpuHandle)
          if (wantDebug) successGPU = gpu_DeviceSynchronize()
          call obj%timer%stop("gpublas")
#ifdef WITH_NVTX
          call nvtxRangePop() ! gpublas
#endif
        else ! useGPU
          call obj%timer%start("blas")
          call PRECISION_GEMM('N', 'N', &
                              int(l_rows, kind=BLAS_KIND), &
                              int(l_cols, kind=BLAS_KIND), &
                              int(nblk_mult_rows, kind=BLAS_KIND), ONE, &
                              aux_a_full, int(l_rows,kind=BLAS_KIND), &
                              aux_b_full, int(nblk_mult,kind=BLAS_KIND), beta, &
                              tmp1_full , int(l_rows,kind=BLAS_KIND))
          call obj%timer%stop("blas")
        endif ! useGPU

#ifdef WITH_NVTX
        call nvtxRangePop() ! do np_row_curr = 0, np_rows-1
#endif
      enddo ! np_fine = 0, np_rows_fine-1
      
      call obj%timer%stop("main_loop_nn")

      ! Put the result into C
      if (useGPU) then
        num = l_rows*l_cols
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        call gpu_memcpy_async_and_stream_synchronize &
            ("elpa_pxgemm: tmp1_full_dev -> tmp1_full", tmp1_full_dev, 0_c_intptr_t, tmp1_full(1:l_rows,1:l_cols), &
              1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
        successGPU = gpu_memcpy(int(loc(tmp1_full),kind=c_intptr_t), tmp1_full_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("elpa_pxgemm: tmp1_full_dev -> tmp1_full", successGPU)
#endif
      endif ! useGPU

      c(1:l_rows,1:l_cols) = tmp1_full(1:l_rows,1:l_cols)

    endif ! (a_transposed .and. b_transposed)

! ___________________________________________________________________
    
    if (.not. a_transposed .and.       b_transposed .or. &
              a_transposed .and. .not. b_transposed ) then
      print *, "elpa_pxgemm NEW: NON-SQUARE_GRID start: ( a_transposed XOR b_transposed)" ! PETERDEBUG
      
      ! dir = row/col for TN/NT
      if (a_transposed) then
        my_pdir = my_prow
        my_pdir_t = my_pcol
        np_dirs = np_rows
        np_dirs_t = np_cols
        mpi_comm_dirs = mpi_comm_rows
      else if (b_transposed) then
        my_pdir = my_pcol
        my_pdir_t = my_prow
        np_dirs = np_cols
        np_dirs_t = np_rows
        mpi_comm_dirs = mpi_comm_cols
      endif

      np_dirs_fine = least_common_multiple(np_rows, np_cols)

      nblk_mult_rows = nblk_mult
      nblk_mult_cols = nblk_mult

      nblk_mult_cols_max = (na+LCM-1)/LCM * nblk

      allocate(aux_a_full(nblk_mult, nblk_mult), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_pxgemm: aux_a_full", istat, errorMessage)
    
      allocate(aux_b_full(nblk_mult, nblk_mult_cols_max*(np_dirs_fine/np_dirs_t)), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_pxgemm: aux_b_full", istat, errorMessage)
      
      allocate(tmp1_full(nblk_mult, nblk_mult_cols_max*(np_dirs_fine/np_dirs_t)), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_pxgemm: tmp1_full", istat, errorMessage)

      call obj%timer%start("main_loop_tn_nt")

      ! main loop: build up the result matrix C by the (virtual) process rows/cols for TN/NT
      do np_fine = 0, np_dirs_fine-1
#ifdef WITH_NVTX
        call nvtxRangePush("do np_fine")
#endif
        print *, "np_fine = ", np_fine ! PETERDEBUG

        ! In this turn, procs of row/col np assemble the result for TN/NT case
        
        np_t_fine=np_fine ! np, that posesses the given "col of a"/"row of b" for TN/NT

        np   = mod(np_fine, np_rows)
        np_t = mod(np_t_fine, np_cols)

        print *, "my_pcol=", my_pcol, "nblk_mult_cols_max=" , nblk_mult_cols_max

        !do np_ab_t_fine = 0, np_dirs_fine-1
        do dnp_ab = 0, np_dirs_fine/np_dirs-1

          ! PETERDEBUG: optimize this!
          !aux_a_full = 0
          !aux_b_full = 0
          !tmp1_full  = 0

          !np_ab_fine = my_pdir*(np_dirs-1)  + dnp_ab
          np_ab_fine = dnp_ab*np_dirs + my_pdir
          if (a_transposed) then
            if (useGPU) then
              ! call gpu_copy_and_set_zeros_aux_a_full(PRECISION_CHAR, a_dev, aux_a_full_dev, nblk_mult, l_cols, &
              !                                        nblk_mult_cols_max, nblk, np_t_fine, np_cols_fine, np_cols, debug, my_stream)
            else ! useGPU
              !if (mod(np_t_fine,np_cols) == my_pcol .and. mod(np_ab_fine,np_rows) == my_prow) then
              if (mod(np_t_fine,np_cols) == my_pcol) then
                do j_block_loc_fine = 0, nblk_mult_cols_max/nblk-1
                  j_block_loc = (np_t_fine + j_block_loc_fine*np_cols_fine)/np_cols
                  !print *, "nblk+j_block_loc*nblk=", nblk+j_block_loc*nblk
                  
                  do i_block_loc_fine = 0, nblk_mult_rows/nblk-1
                    i_block_loc = (np_ab_fine + i_block_loc_fine*np_rows_fine)/np_rows
                    
                    ! PETERDEBUG: negative nblk_cols_min can cause the problems !!! (here and below in b-part)
                    nblk_cols_min = min(nblk, l_cols - j_block_loc*nblk)
                    nblk_rows_min = min(nblk, l_rows - i_block_loc*nblk)

                    ! PETERDEBUG111
                    aux_a_full(1+i_block_loc_fine*nblk : nblk_rows_min+i_block_loc_fine*nblk, &
                               1+j_block_loc_fine*nblk : nblk_cols_min+j_block_loc_fine*nblk) = &
                             a(1+i_block_loc*nblk      : nblk_rows_min+i_block_loc*nblk, &
                               1+j_block_loc*nblk      : nblk_cols_min+j_block_loc*nblk)
                    
                    ! if (nblk_rows_min<nblk .and. nblk_rows_min>0) then
                    !   aux_a_full(nblk_rows_min+1+i_block_loc_fine*nblk : nblk+i_block_loc_fine*nblk, &
                    !              1+j_block_loc_fine*nblk : nblk_cols_min+j_block_loc_fine*nblk) = 0
                    ! endif

                    ! if (nblk_cols_min<nblk .and. nblk_cols_min>0) then
                    !   aux_a_full(1+i_block_loc_fine*nblk : nblk_rows_min+i_block_loc_fine*nblk, &
                    !              nblk_cols_min+1+j_block_loc_fine*nblk : nblk+j_block_loc_fine*nblk) = 0
                    ! endif

                  enddo ! i_block_loc_fine
                enddo ! j_block_loc_fine
              endif ! (mod(np_t_fine,np_cols) == my_pcol .and. mod(np_ab_fine,np_rows) == my_prow)
            endif ! useGPU

            ! aux_b_full(1:nblk_mult,1:l_cols) = b(1:nblk_mult,1:l_cols)
            if (useGPU) then
              ! call gpu_copy_and_set_zeros_aux_b_full(PRECISION_CHAR, b_dev, aux_b_full_dev, nblk_mult, l_cols, nblk_mult, &
              !                                        nblk_mult_rows, nblk, np_fine, np_rows_fine, np_rows, SM_count, debug, my_stream)
            else ! useGPU

              do dnp_ab_t = 0, np_dirs_fine/np_dirs_t-1
                np_ab_t_fine = dnp_ab_t*np_dirs_t + my_pdir_t

                do j_block_loc_fine = 0, nblk_mult_cols_max/nblk-1
                  j_block_loc = (np_ab_t_fine + j_block_loc_fine*np_cols_fine)/np_cols
                  
                  do i_block_loc_fine = 0, nblk_mult_rows/nblk-1
                    i_block_loc = (np_ab_fine + i_block_loc_fine*np_rows_fine)/np_rows

                    nblk_cols_min = min(nblk, l_cols - j_block_loc*nblk)
                    nblk_rows_min = min(nblk, l_rows - i_block_loc*nblk)

                    aux_b_full(1+i_block_loc_fine*nblk : nblk_rows_min+i_block_loc_fine*nblk, &
                                1            +j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_cols_max : &
                                nblk_cols_min+j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_cols_max) = &
                              b(1+i_block_loc*nblk      :nblk_rows_min+i_block_loc*nblk, &
                                1+j_block_loc*nblk      :nblk_cols_min+j_block_loc*nblk)

                    if (nblk_rows_min<nblk) then
                      if (nblk_rows_min>0) then
                        aux_b_full(nblk_rows_min+1+i_block_loc_fine*nblk : nblk+i_block_loc_fine*nblk, &
                                    1            +j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_cols_max : &
                                    nblk_cols_min+j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_cols_max) = 0
                      else
                        aux_b_full(1+i_block_loc_fine*nblk : nblk+i_block_loc_fine*nblk, &
                                    1   +j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_cols_max : &
                                    nblk+j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_cols_max) = 0
                      endif
                    endif

                    if (nblk_cols_min<nblk) then
                      if (nblk_cols_min>0) then
                        aux_b_full(1+i_block_loc_fine*nblk : nblk_rows_min+i_block_loc_fine*nblk, &
                                    nblk_cols_min+1+j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_cols_max : &
                                    nblk           +j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_cols_max) = 0
                      else
                        aux_b_full(1+i_block_loc_fine*nblk : nblk+i_block_loc_fine*nblk, &
                                    1   +j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_cols_max : &
                                    nblk+j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_cols_max) = 0
                      endif
                    endif

                  enddo ! i_block_loc_fine
                enddo ! j_block_loc_fine
              enddo ! np_ab_t_fine
              
            endif ! useGPU
          else if (b_transposed) then
            print *, "to be done..."
            ! if (np_t == my_prow) aux_b_full(1:nblk_mult,1:l_cols) = b(1:nblk_mult,1:l_cols)
            ! aux_a_full(1:nblk_mult,1:l_cols) = a(1:nblk_mult,1:l_cols) ! aux_a_full -> aux_ab_nontransposed_full
          endif
        
          ! aux_a_full/aux_b_full -> aux_ab_trans (aux_ab) ! auxillary buffer for a matrix that is     to be transposed: a/b in TN/NT case
          ! aux_b_full/aux_a_full -> aux_ab_nontr (aux_ba) ! auxillary buffer for a matrix that is not to be transposed: b/a in TN/NT case
          !enddo ! dnp_ab -> later

          ! Broadcast processor column
          call obj%timer%start("mpi_communication")
#ifdef WITH_NVTX
      call nvtxRangePush("MPI_Bcast(aux_a/b_full)")
#endif
          if (a_transposed) then
            call MPI_Bcast(aux_a_full, int(nblk_mult*nblk_mult,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                          int(np_t,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
          else if (b_transposed) then
            ! call MPI_Bcast(aux_b_full, int(nblk_mult*nblk_mult,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
            !               int(np_t,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
          endif

#ifdef WITH_NVTX
      call nvtxRangePop() ! MPI_Bcast(aux_a/b_full)
#endif
          call obj%timer%stop("mpi_communication")

          call obj%timer%start("blas")

          do dnp_ab_t = 0, np_dirs_fine/np_dirs_t-1

            call PRECISION_GEMM(trans_a, trans_b, &
                                int(nblk_mult, kind=BLAS_KIND), &
                                int(nblk_mult, kind=BLAS_KIND), &
                                int(nblk_mult, kind=BLAS_KIND), ONE, &
                                aux_a_full(1:nblk_mult,1:nblk_mult), int(nblk_mult, kind=BLAS_KIND), &
                                aux_b_full(1:nblk_mult,(1+dnp_ab_t*nblk_mult_cols_max):(nblk_mult+dnp_ab_t*nblk_mult_cols_max)),&
                                int(nblk_mult, kind=BLAS_KIND), ZERO, &
                                tmp1_full (1:nblk_mult,(1+dnp_ab_t*nblk_mult_cols_max):(nblk_mult+dnp_ab_t*nblk_mult_cols_max)),&
                                int(nblk_mult, kind=BLAS_KIND))

          enddo ! dnp_ab_t

          call obj%timer%stop("blas")

          call obj%timer%start("mpi_communication")
          if (my_pdir==np) then
            call mpi_reduce(MPI_IN_PLACE, tmp1_full, int(nblk_mult*nblk_mult_cols_max*(np_dirs_fine/np_dirs_t),kind=MPI_KIND), & 
                            MPI_MATH_DATATYPE_PRECISION, MPI_SUM, int(np,kind=MPI_KIND), int(mpi_comm_dirs,kind=MPI_KIND), mpierr)
          else
            call mpi_reduce(tmp1_full   , tmp1_full, int(nblk_mult*nblk_mult_cols_max*(np_dirs_fine/np_dirs_t),kind=MPI_KIND), &
                            MPI_MATH_DATATYPE_PRECISION, MPI_SUM, int(np,kind=MPI_KIND), int(mpi_comm_dirs,kind=MPI_KIND), mpierr)
          endif
          call obj%timer%stop("mpi_communication")

          ! Put the result into C
          beta = ONE
          if (dnp_ab == 0) beta = ZERO

          if (a_transposed) then
            if (my_pdir==np) then
              do dnp_ab_t = 0, np_dirs_fine/np_dirs_t-1
                np_ab_t_fine = dnp_ab_t*np_dirs_t + my_pdir_t

                do j_block_loc_fine = 0, nblk_mult_cols_max/nblk-1
                  j_block_loc = (np_ab_t_fine + j_block_loc_fine*np_cols_fine)/np_cols
                  
                  do i_block_loc_fine = 0, nblk_mult_rows/nblk-1
                    i_block_loc = (np_fine + i_block_loc_fine*np_rows_fine)/np_rows

                    nblk_cols_min = min(nblk, l_cols - j_block_loc*nblk)
                    nblk_rows_min = min(nblk, l_rows - i_block_loc*nblk)

                    c     (1+i_block_loc*nblk:nblk_rows_min+i_block_loc*nblk, &
                           1+j_block_loc*nblk:nblk_cols_min+j_block_loc*nblk) = &
                    beta*c(1+i_block_loc*nblk:nblk_rows_min+i_block_loc*nblk, &
                           1+j_block_loc*nblk:nblk_cols_min+j_block_loc*nblk) + &
                    tmp1_full(1+i_block_loc_fine*nblk :nblk+i_block_loc_fine*nblk, &
                              1   +j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_cols_max : &
                              nblk+j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_cols_max)
                  enddo ! i_block_loc_fine
                enddo ! j_block_loc_fine
              enddo ! dnp_ab_t = 0, np_dirs_fine/np_dirs_t-1
            endif ! (my_pdir==np)
          endif ! (a_transposed)

        enddo ! dnp_ab = 0, np_dirs_fine/np_dirs-1
        
#ifdef WITH_NVTX
        call nvtxRangePop() ! do np = 0, np_dirs-1
#endif
      enddo ! np = 0, np_dirs-1
      call obj%timer%stop("main_loop_tn_nt")

    endif ! (.not. a_transposed .and. b_transposed)

! ___________________________________________________________________  
  
  endif ! (.not. isSquareGrid)
  
!______________________________________________________________________________________________

    ! PETERDEBUG
    deallocate(aux_a_full, stat=istat, errmsg=errorMessage)
    deallocate(aux_b_full, stat=istat, errmsg=errorMessage)
    deallocate(tmp1_full,  stat=istat, errmsg=errorMessage)
    
    !deallocate(aux_a_full, aux_b_full, tmp1_full, stat=istat, errmsg=errorMessage)
    !call check_alloc("elpa_pxgemm", "aux_a_full, tmp1_full", istat, errorMessage)

    if (useGPU) then
      successGPU = gpu_free(aux_a_full_dev)
      check_dealloc_gpu("elpa_pxgemm: aux_a_full_dev", successGPU)

      successGPU = gpu_free(aux_b_full_dev)
      check_dealloc_gpu("elpa_pxgemm: aux_b_full_dev", successGPU)

      successGPU = gpu_free(tmp1_full_dev)
      check_dealloc_gpu("elpa_pxgemm: tmp1_full_dev", successGPU)
    endif

!______________________________________________________________________________________________

  if (useGPU) then
#if !defined(DEVICE_POINTER)
    successGPU = gpu_free(b_dev)
    check_dealloc_gpu("elpa_pxgemm_a_b: b_dev", successGPU)
#if !defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) && !defined(WITH_SYCL_GPU_VERSION)
    successGPU = gpu_host_unregister(int(loc(b),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_pxgemm_a_b: b", successGPU)
#endif

    successGPU = gpu_free(c_dev)
    check_dealloc_gpu("elpa_pxgemm_a_b: c_dev", successGPU)

#else /* DEVICE_POINTER */
    !successGPU = gpu_free(b_dev)
    !check_dealloc_gpu("elpa_pxgemm_a_b: b_dev", successGPU)

    !num = ldc*ldcCols*size_of_datatype
    !successGPU = gpu_memcpy(cDev, c_dev, num,&
    !              gpuMemcpyDeviceToDevice)
    !check_memcpy_gpu("elpa_pxgemm: c_dev -> cDev", successGPU)

    !successGPU = gpu_free(c_dev)
    !check_dealloc_gpu("elpa_pxgemm_a_b: c_dev", successGPU)

#ifdef WITH_GPU_STREAMS
    successGPU = gpu_host_unregister(int(loc(a_tmp),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_pxgemm: a_tmp", successGPU)
#endif
    deallocate(a_tmp, stat=istat, errmsg=errorMessage)
    check_deallocate("elpa_pxgemm: a_tmp", istat, errorMessage)

    num = ldc*ldcCols*size_of_datatype
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_pxgemm: c_tmp to c", successGPU)

    successGPU = gpu_memcpy_async(cDev,int(loc(c_tmp),kind=c_intptr_t),num,&
                  gpuMemcpyHostToDevice, my_stream)
    check_memcpy_gpu("elpa_pxgemm: c_tmp -> c", successGPU)

    my_stream = obj%gpu_setup%my_stream
    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_pxgemm: c_tmp -> c", successGPU)
    ! synchronize streamsPerThread; maybe not neccessary
    successGPU = gpu_stream_synchronize()
    check_stream_synchronize_gpu("elpa_pxgemm: c_tmp -> c", successGPU)
#else
    successGPU = gpu_memcpy(cDev,int(loc(c_tmp),kind=c_intptr_t),num,&
                  gpuMemcpyHostToDevice)
    check_memcpy_gpu("elpa_pxgemm: c_tmp -> c", successGPU)
#endif
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_host_unregister(int(loc(c_tmp),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_pxgemm_a_b: c_tmp", successGPU)
#endif

    deallocate(c_tmp, stat=istat, errmsg=errorMessage)
    check_deallocate("elpa_pxgemm: c_tmp", istat, errorMessage)

#endif /* DEVICE_POINTER */
#if !defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) && !defined(WITH_SYCL_GPU_VERSION)
    nullify(aux_mat)
    !nullify(tmp1)

    successGPU = gpu_free_host(aux_host)
    check_host_dealloc_gpu("elpa_pxgemm_a_b: aux_host", successGPU)
#else
    deallocate(aux_mat, stat=istat, errmsg=errorMessage)
    check_deallocate("elpa_pxgemm: aux_mat", istat, errorMessage)

    !deallocate(tmp1, stat=istat, errmsg=errorMessage)
    !check_deallocate("elpa_pxgemm: tmp1", istat, errorMessage)
#endif

    successGPU = gpu_free(aux_mat_dev)
    check_dealloc_gpu("elpa_pxgemm_a_b: aux_mat_dev", successGPU)

    successGPU = gpu_free(tmp1_dev)
    check_dealloc_gpu("elpa_pxgemm_a_b: tmp1_dev", successGPU)

    successGPU = gpu_free(tmp2_dev)
    check_dealloc_gpu("elpa_pxgemm_a_b: tmp2_dev", successGPU)

    successGPU = gpu_free(aux_bc_dev)
    check_dealloc_gpu("elpa_pxgemm_a_b: aux_bc_dev", successGPU)

#if !defined(DEVICE_POINTER)
    successGPU = gpu_free(a_dev)
    check_dealloc_gpu("elpa_pxgemm: a_dev", successGPU)
#else
    !successGPU = gpu_free(a_dev)
    !check_dealloc_gpu("elpa_pxgemm: a_dev", successGPU)
#endif

  else ! useGPU
    deallocate(aux_mat, stat=istat, errmsg=errorMessage)
    check_deallocate("elpa_pxgemm: aux_mat", istat, errorMessage)
  endif ! useGPU

  deallocate(aux_bc, lrs_save, lre_save, stat=istat, errmsg=errorMessage)
  check_deallocate("elpa_pxgemm: aux_bc, lrs_save, lre_save", istat, errorMessage)

  call obj%timer%stop("elpa_pxgemm_&
  &MATH_DATATYPE&
  &_&
  &PRECISION&
  &"//gpuString)

#ifdef WITH_NVTX
  call nvtxRangePop() ! pxgemm
#endif
