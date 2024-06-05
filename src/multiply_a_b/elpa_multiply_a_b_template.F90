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

#undef USE_CCL_MULTIPLY
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
#define USE_CCL_MULTIPLY
#endif



#ifdef USE_CCL_MULTIPLY
#define MORE_GPU
#endif
#ifndef USE_CCL_MULTIPLY
#define MORE_GPU
#endif
#ifdef DEVICE_POINTER
#define MORE_GPU
#endif

  use elpa1_compute
  use elpa_mpi
  use precision
  use elpa_abstract_impl
  use, intrinsic :: iso_c_binding
  use elpa_gpu
  use mod_check_for_gpu
  use elpa_blas_interfaces
  use ELPA_utilities, only : local_index, check_deallocate_f, check_dealloc_gpu_f, &
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

  character*1                                  :: uplo_a, uplo_c, trans_a

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

  integer(kind=ik)                             :: my_prow, my_pcol, np_rows, np_cols, myid
  integer(kind=MPI_KIND)                       :: my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI
  integer(kind=MPI_KIND)                       :: mpierr, myidMPI
  integer(kind=ik)                             :: l_cols, l_rows, l_rows_np
  integer(kind=ik)                             :: np, n, nb, nblk_mult, lrs, lre, lcs, lce
  integer(kind=ik)                             :: gcol_min, gcol, goff
  integer(kind=ik)                             :: nstor, nr_done, noff, np_bc, n_aux_bc, nvals
  integer(kind=ik)                             :: np_br, noff_n ! PETERDEBUG
  integer(kind=ik), allocatable                :: lrs_save(:), lre_save(:)

  logical                                      :: a_lower, a_upper, c_lower, c_upper
  MATH_DATATYPE(kind=rck), pointer, contiguous :: aux_mat(:,:), tmp1(:,:)
  MATH_DATATYPE(kind=rck), allocatable         :: aux_bc(:), tmp2(:,:)
  integer(kind=ik)                             :: istat
  character(200)                               :: errorMessage
  character(20)                                :: gpuString
  logical                                      :: success
  logical                                      :: successGPU
  logical                                      :: useGPU
  integer(kind=c_int)                          :: numGPU, blocking
  integer(kind=ik)                             :: mpi_comm_rows, mpi_comm_cols, mpi_comm_all
  integer(kind=ik)                             :: nblk, matrixRows, matrixCols, error
  integer(kind=c_intptr_t)                     :: aux_mat_dev, tmp1_dev
!#ifndef DEVICE_POINTER
  integer(kind=c_intptr_t)                     :: b_dev
#ifdef MORE_GPU
  integer(kind=c_intptr_t)                     :: a_dev
#endif
#ifdef MORE_GPU
  integer(kind=c_intptr_t)                     :: c_dev
#endif
!#endif
#ifdef MORE_GPU
  integer(kind=c_intptr_t)                     :: tmp2_dev, aux_bc_dev
#endif
  type(c_ptr)                                  :: aux_host, tmp1_host
  integer(kind=c_intptr_t)                     :: num
  integer(kind=c_intptr_t)                     :: aux_off, b_off
  integer(kind=c_intptr_t), parameter          :: size_of_datatype = size_of_&
                                                            &PRECISION&
                                                            &_&
                                                            &MATH_DATATYPE

  integer(kind=c_intptr_t)                     :: gpuHandle, my_stream
  integer(kind=c_int)                          :: gpu_hermitian_multiply

#if defined(USE_CCL_MULTIPLY)
  integer(kind=c_intptr_t)                      :: ccl_comm_rows, ccl_comm_cols
#endif
#ifdef DEVICE_POINTER
  MATH_DATATYPE(kind=rck), allocatable         :: a_tmp(:,:), c_tmp(:,:)
#endif
  integer(kind=c_intptr_t)                     :: aux_dev
  integer(kind=c_int)                          :: gpu
  integer(kind=c_int)                          :: gpu_multiply_a_b
  success = .true.
  useGPU = .false.

  trans_a='N' ! PETERDEBUG: propagate it up, accept as an option

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
#endif /* DEVICE_POINTER */

  ! assumption, when DEVICE_POINTER -> MORE_GPU
  !             when DEVICE_POINTER -> useGPU = .true.

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

#ifdef MORE_GPU  
  c_dev = transfer(cDev, c_dev)
  a_dev = transfer(aDev, a_dev)
#else /* MORE_GPU */
  allocate(a_tmp(obj%local_nrows,obj%local_ncols), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_mult_at_b: a_tmp", istat, errorMessage)

  num = obj%local_nrows*obj%local_ncols*size_of_datatype
#ifdef WITH_GPU_STREAMS
  successGPU = gpu_host_register(int(loc(a_tmp),kind=c_intptr_t), &
                  obj%local_nrows*obj%local_ncols * size_of_datatype,&
                  gpuHostRegisterDefault)
  check_host_register_gpu("elpa_mult_at_b: a_tmp", successGPU)

  my_stream = obj%gpu_setup%my_stream
  successGPU = gpu_stream_synchronize(my_stream)
  check_stream_synchronize_gpu("elpa_mult_at_b: a_dev to a_tmp", successGPU)

  successGPU = gpu_memcpy_async(int(loc(a_tmp),kind=c_intptr_t), aDev, num,&
                gpuMemcpyDeviceToHost, my_stream)
  check_memcpy_gpu("elpa_mult_at_b: a_dev -> a_tmp", successGPU)

  my_stream = obj%gpu_setup%my_stream
  successGPU = gpu_stream_synchronize(my_stream)
  check_stream_synchronize_gpu("elpa_mult_at_b: a_dev -> a_tmp", successGPU)
  ! synchronize streamsPerThread; maybe not neccessary
  successGPU = gpu_stream_synchronize()
  check_stream_synchronize_gpu("elpa_mult_at_b: a_dev -> a_tmp", successGPU)
#else
  successGPU = gpu_memcpy(int(loc(a_tmp),kind=c_intptr_t), aDev, num,&
                  gpuMemcpyDeviceToHost)
  check_memcpy_gpu("elpa_mult_at_b: a_dev -> a_tmp", successGPU)
#endif

  allocate(c_tmp(ldc,ldcCols), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_mult_at_b: c_tmp", istat, errorMessage)

#ifdef WITH_GPU_STREAMS
  successGPU = gpu_host_register(int(loc(c_tmp),kind=c_intptr_t),&
               ldc*ldcCols*size_of_datatype, &
                gpuHostRegisterDefault)
#endif

#endif /* MORE_GPU  */


  b_dev = transfer(bDev, b_dev)

#endif /* DEVICE_POINTER */

  if(useGPU) then
    gpuString = "_gpu"
  else
    gpuString = ""
  endif

  call obj%timer%start("elpa_mult_at_b_&
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
      write(error_unit,*) "ELPA_HERMITIAN_MULTIPLY: Problem in getting keyword 'blocking_in_multiply'. Aborting..."
      stop 1
    endif
    nblk_mult = (blocking/nblk+1) * nblk
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

#ifdef MORE_GPU
#if !defined(DEVICE_POINTER)
    num = ldc*ldcCols*size_of_datatype
    successGPU = gpu_malloc(c_dev, num)
    check_alloc_gpu("elpa_mult_at_b: c_dev", successGPU)
    ! no copy from c to c_dev needed since c will be overwritten anyway
#endif
#endif /* MORE_GPU */

#if !defined(DEVICE_POINTER)
    ! copy b to b_dev
    num = ldb*ldbCols*size_of_datatype
    successGPU = gpu_malloc(b_dev, num)
    check_alloc_gpu("elpa_mult_at_b: b_dev", successGPU)

#if !defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) && !defined(WITH_SYCL_GPU_VERSION)
    successGPU = gpu_host_register(int(loc(b),kind=c_intptr_t),num,&
                  gpuHostRegisterDefault)
#endif    

    check_host_register_gpu("elpa_mult_at_b: b", successGPU)
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    call gpu_memcpy_async_and_stream_synchronize &
    ("elpa_mult_at_b: b to b_dev", b_dev, 0_c_intptr_t, &
                                       b(1:ldb,1:ldbCols), &
                                       1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
    successGPU = gpu_memcpy(b_dev,int(loc(b),kind=c_intptr_t),num,&
                  gpuMemcpyHostToDevice)
    check_memcpy_gpu("elpa_mult_at_b: b to b_dev", successGPU)
#endif

#else /* DEVICE_POINTER */

#endif /* DEVICE_POINTER */

    num = l_rows*nblk_mult*size_of_datatype
#if !defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) && !defined(WITH_SYCL_GPU_VERSION)
    successGPU = gpu_malloc_host(aux_host, num)
    check_host_alloc_gpu("elpa_mult_at_b: aux_host", successGPU)
    call c_f_pointer(aux_host, aux_mat, (/l_rows,nblk_mult/))
#else
    allocate(aux_mat(l_rows, nblk_mult), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_mult_at_b: aux_mat", istat, errorMessage)
#endif

    successGPU = gpu_malloc(aux_mat_dev, num)
    check_alloc_gpu("elpa_mult_at_b: aux_mat_dev", successGPU)

    num = nblk_mult*l_cols*size_of_datatype
    successGPU = gpu_malloc(tmp1_dev, num)
    check_alloc_gpu("elpa_mult_at_b: tmp1_dev", successGPU)

!#ifdef MORE_GPU
!    allocate(tmp2(nblk_mult,l_cols), stat=istat, errmsg=errorMessage)
!    check_allocate("elpa_mult_at_b: tmp2", istat, errorMessage)
!#endif

#ifdef MORE_GPU
    num = nblk_mult*l_cols*size_of_datatype
    successGPU = gpu_malloc(tmp2_dev, num)
    check_alloc_gpu("elpa_mult_at_b: tmp2_dev", successGPU)
#endif
  else ! useGPU
    allocate(aux_mat(l_rows,nblk_mult), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_mult_at_b: aux_mat", istat, errorMessage)
  endif ! useGPU

  allocate(aux_bc(l_rows*nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_mult_at_b: aux_bc", istat, errorMessage)

  allocate(lrs_save(nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_mult_at_b: lrs_save", istat, errorMessage)

  allocate(lre_save(nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_mult_at_b: lre_save", istat, errorMessage)

  a_lower = .false.
  a_upper = .false.
  c_lower = .false.
  c_upper = .false.

  if (uplo_a=='u' .or. uplo_a=='U') a_upper = .true.
  if (uplo_a=='l' .or. uplo_a=='L') a_lower = .true.
  if (uplo_c=='u' .or. uplo_c=='U') c_upper = .true.
  if (uplo_c=='l' .or. uplo_c=='L') c_lower = .true.

#ifdef MORE_GPU
  if (useGPU) then

#if !defined(DEVICE_POINTER)
    num = obj%local_nrows*obj%local_ncols*size_of_datatype
    successGPU = gpu_malloc(a_dev, num)
    check_alloc_gpu("elpa_mult_at_b: a_dev", successGPU)
#endif

    num = l_rows*nblk*size_of_datatype
    successGPU = gpu_malloc(aux_bc_dev, num)
    check_alloc_gpu("elpa_mult_at_b: aux_bc_dev", successGPU)

    num = obj%local_nrows*obj%local_ncols*size_of_datatype
#if !defined(DEVICE_POINTER)

#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    call gpu_memcpy_async_and_stream_synchronize &
    ("elpa_mult_at_b: a to a_dev", a_dev, 0_c_intptr_t, &
                                       a(1:obj%local_nrows,1:obj%local_ncols), &
                                       1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
    successGPU = gpu_memcpy(a_dev, int(loc(a),kind=c_intptr_t), &
                  num, gpuMemcpyHostToDevice)
    check_memcpy_gpu("elpa_mult_at_b: a to a_dev", successGPU)
#endif
#endif /* DEVICE_POINTER */
  endif !useGPU
#endif /* MORE_GPU */

  ! main loop: build up the result matrix by processor rows
  do np = 0, np_rows-1

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
      check_memcpy_gpu("hermitian_multiply: aux_mat_dev", successGPU)
#else
      successGPU = gpu_memset(aux_mat_dev, 0, num)
      check_memcpy_gpu("hermitian_multiply: aux_mat_dev", successGPU)
#endif
    endif ! useGPU

    ! Loop over the blocks on row np
    do nb = 0, (l_rows_np-1)/nblk

#ifdef WITH_NVTX
      call nvtxRangePush("do nb = 0, (l_rows_np-1)/nblk")
#endif

      goff  = nb*np_rows + np ! Global offset in blocks corresponding to nb
      ! rename: goff -> block_offset_gl

      ! Get the processor row (if trans_a='N') or column (if trans_a='T' or 'H') which owns this block
      ! and the offset in blocks within this row/column.
      ! The corresponding block row/column in A is then broadcast to all for multiplication with B

      np_br = MOD(goff,np_rows) ! np, that posesses the given block row   ; trans_a='N'; "br"=block row ! PETERDEBUG
      np_bc = MOD(goff,np_cols) ! np, that posesses the given block column; trans_a='T'; "bc"=block column; rename: np_bc -> np_col_b
      
      noff = goff/np_cols ! noff -> noff_t 
      noff_n = goff/np_rows ! PETERDEBUG
      
      ! Gather up the complete block column of A on the owner in contigous memory of aux_bc array
      n_aux_bc = 0
      do n = 1, min(l_rows_np-nb*nblk, nblk) ! Loop over local rows/columns to be broadcast

        gcol = goff*nblk + n ! global column corresponding to n

        if (nstor==0 .and. n==1) gcol_min = gcol

        lrs = 1       ! 1st local row number for broadcast
        lre = l_rows  ! last local row number for broadcast
        if (a_lower) lrs = local_index(gcol, my_prow, np_rows, nblk, +1)
        if (a_upper) lre = local_index(gcol, my_prow, np_rows, nblk, -1)

        if (lrs <= lre) then
          nvals = lre-lrs+1
#ifdef MORE_GPU
          if (useGPU) then
            if (my_pcol == np_bc) call gpu_copy_PRECISION_a_aux_bc(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, &
                                                                   nblk, n, l_rows, obj%local_nrows, obj%local_ncols, my_stream)
          else ! useGPU
            if (my_pcol == np_bc) aux_bc(n_aux_bc+1:n_aux_bc+nvals) = a(lrs:lre,noff*nblk+n)
          endif ! useGPU
#else /* MORE_GPU */
#ifndef DEVICE_POINTER
          if (my_pcol == np_bc) aux_bc(n_aux_bc+1:n_aux_bc+nvals) = a(lrs:lre,noff*nblk+n)
#else
          if (my_pcol == np_bc) aux_bc(n_aux_bc+1:n_aux_bc+nvals) = a_tmp(lrs:lre,noff*nblk+n)
#endif
#endif /* MORE_GPU */
          n_aux_bc = n_aux_bc + nvals
        endif ! (lrs <= lre)

        lrs_save(n) = lrs
        lre_save(n) = lre

      enddo ! n = 1, min(l_rows_np-nb*nblk, nblk)

#ifdef WITH_MPI
! CCL only with MPI
#ifdef USE_CCL_MULTIPLY
      if (useGPU) then ! useGPU-USE_CCL_MULTIPLY
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        ccl_comm_cols = obj%gpu_setup%ccl_comm_cols
        successGPU = ccl_group_start()

        if (.not.successGPU) then
          print *,"Error in setting up ccl_group_start!"
          stop 1
        endif

        successGPU = ccl_bcast(aux_bc_dev, aux_bc_dev, &

#if REALCASE == 1
                         int(n_aux_bc,kind=c_size_t), &
#endif
#if COMPLEXCASE == 1
                         int(2*n_aux_bc,kind=c_size_t), &
#endif
#if REALCASE == 1
#ifdef DOUBLE_PRECISION
                         cclDouble, &
#endif
#ifdef SINGLE_PRECISION
                         cclFloat, &
#endif
#endif /* REALCASE */
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION
                        cclDouble, &
#endif
#ifdef SINGLE_PRECISION
                        cclFloat, &
#endif
#endif /* COMPLEXCASE */
                        int(np_bc,kind=c_int), ccl_comm_cols, my_stream)

        if (.not.successGPU) then
          print *,"Error in ccl_bcast"
          stop 1
        endif

        successGPU = ccl_group_end()
        if (.not.successGPU) then
          print *,"Error in setting up ccl_group_end!"
          stop 1
        endif
#endif /* WITH_GPU_STREAMS */
      else ! useGPU-USE_CCL_MULTIPLY
#endif /* USE_CCL_MULTIPLY */

#if defined(MORE_GPU) && !defined(USE_CCL_MULTIPLY)
      if (useGPU) then
        ! copy data to host for Bcast
        num = l_rows*nblk*size_of_datatype
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        call gpu_memcpy_async_and_stream_synchronize &
        ("elpa_mult_at_b: aux_bc_dev -> aux_bc", aux_bc_dev, 0_c_intptr_t, &
                                           aux_bc(1:l_rows*nblk), &
                                           1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
        num = l_rows*nblk*size_of_datatype
        successGPU = gpu_memcpy(int(loc(aux_bc),kind=c_intptr_t), aux_bc_dev, num,&
                                gpuMemcpyDeviceToHost)
        check_memcpy_gpu("elpa_mult_at_b: aux_bc_dev -> aux_bc", successGPU)
#endif

      endif ! useGPU
#endif /* defined(MORE_GPU) && !defined(USE_CCL_MULTIPLY) */

      ! Broadcast block column
      call obj%timer%start("mpi_communication")

      call MPI_Bcast(aux_bc, int(n_aux_bc,kind=MPI_KIND),    &
                     MPI_MATH_DATATYPE_PRECISION,  &
                     int(np_bc,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)

      call obj%timer%stop("mpi_communication")

#if defined(MORE_GPU) && !defined(USE_CCL_MULTIPLY)
      if (useGPU) then
        ! copy data back to device
        num = l_rows*nblk*size_of_datatype
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        call gpu_memcpy_async_and_stream_synchronize &
        ("elpa_mult_at_b: aux_bc -> aux_bc_dev", aux_bc_dev, 0_c_intptr_t, &
                                           aux_bc(1:l_rows*nblk), &
                                           1, num, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
        successGPU = gpu_memcpy(aux_bc_dev, int(loc(aux_bc),kind=c_intptr_t), num,&
                                gpuMemcpyHostToDevice)
        check_memcpy_gpu("elpa_mult_at_b: aux_bc -> aux_bc_dev", successGPU)
#endif

      endif ! useGPU
#endif /* defined(MORE_GPU) && !defined(USE_CCL_MULTIPLY) */

#ifdef USE_CCL_MULTIPLY
      endif ! useGPU-USE_CCL_MULTIPLY
#endif /* USE_CCL_MULTIPLY */

#else /* WITH_MPI */

#endif /* WITH_MPI */
      ! Insert what we got in aux_mat


#ifdef MORE_GPU
      if (useGPU) then
        n_aux_bc = 0
        my_stream = obj%gpu_setup%my_stream
        do n = 1, min(l_rows_np-nb*nblk,nblk)
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
#endif /* MORE_GPU */

        n_aux_bc = 0
        do n = 1, min(l_rows_np-nb*nblk,nblk)
          nstor = nstor+1
          lrs = lrs_save(n)
          lre = lre_save(n)
          if (lrs<=lre) then
            nvals = lre-lrs+1
            aux_mat(lrs:lre,nstor) = aux_bc(n_aux_bc+1:n_aux_bc+nvals)
            n_aux_bc = n_aux_bc + nvals
          endif
        enddo

#ifdef MORE_GPU
      endif ! useGPU
#endif /* MORE_GPU */

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
#if defined(MORE_GPU) && defined(USE_CCL_MULTIPLY)
          if (.not.useGPU) then
#endif
              ! introduce 1-based indexing
              allocate(tmp1(nstor,1:lce-lcs+1), tmp2(nstor,1:lce-lcs+1), stat=istat, errmsg=errorMessage)
              call check_alloc("elpa_mult_at_b_&
              &MATH_DATATYPE ", "tmp1", istat, errorMessage)
#if defined(MORE_GPU) && defined(USE_CCL_MULTIPLY)
          endif
#endif

          if (lrs <= lre) then
            if (useGPU) then
#ifndef MORE_GPU
              num = l_rows*nblk_mult*size_of_datatype
#ifdef WITH_GPU_STREAMS
              my_stream = obj%gpu_setup%my_stream
              call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_mult_at_b: aux_mat to aux_mat_dev", aux_mat_dev, 0_c_intptr_t, &
                                                 aux_mat(1:l_rows,1:nblk_mult), &
                                                 1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
              successGPU = gpu_memcpy(aux_mat_dev, int(loc(aux_mat),kind=c_intptr_t), &
                            num, gpuMemcpyHostToDevice)
              check_memcpy_gpu("elpa_mult_at_b: aux_mat to aux_mat_dev", successGPU)
#endif

#endif /* MORE_GPU */
              aux_off = (lrs-1)*size_of_datatype
              b_off = ((lcs-1)*ldb+lrs-1)*size_of_datatype

              call obj%timer%start("gpublas")
              gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
              call gpublas_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', nstor, lce-lcs+1, &
                   lre-lrs+1, ONE, aux_mat_dev+aux_off, l_rows, b_dev+b_off, ldb, ZERO, &
                   tmp1_dev, nstor, gpuHandle)
              call obj%timer%stop("gpublas")

#ifndef MORE_GPU
              num = nstor*(lce-lcs+1)*size_of_datatype
#ifdef WITH_GPU_STREAMS
              my_stream = obj%gpu_setup%my_stream
              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("elpa_mult_at_b: tmp1_dev to tmp1", successGPU)

              successGPU = gpu_memcpy_async(int(loc(tmp1),kind=c_intptr_t), &
                            tmp1_dev, num, gpuMemcpyDeviceToHost, my_stream)
              check_memcpy_gpu("elpa_mult_at_b: tmp1_dev to tmp1", successGPU)

              my_stream = obj%gpu_setup%my_stream
              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("elpa_mult_at_b: tmp1_dev to tmp1", successGPU)
              ! synchronize streamsPerThread; maybe not neccessary
              successGPU = gpu_stream_synchronize()
              check_stream_synchronize_gpu("elpa_mult_at_b: tmp1_dev to tmp1", successGPU)
#else
              successGPU = gpu_memcpy(int(loc(tmp1),kind=c_intptr_t), &
                            tmp1_dev, num, gpuMemcpyDeviceToHost)
              check_memcpy_gpu("elpa_mult_at_b: tmp1_dev to tmp1", successGPU)
#endif
#endif /* MORE_GPU */
              

              num = nstor*(lce-lcs+1)*size_of_datatype
            else ! useGPU
              call obj%timer%start("blas")
              call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', int(nstor,kind=BLAS_KIND), &
                                int(lce-lcs+1,kind=BLAS_KIND), int(lre-lrs+1,kind=BLAS_KIND), &
                                ONE, aux_mat(lrs:lre,1:nstor), int(lre-lrs+1,kind=BLAS_KIND), &
                                b(lrs,lcs), int(ldb,kind=BLAS_KIND), ZERO, tmp1, &
                                int(nstor,kind=BLAS_KIND))
              call obj%timer%stop("blas")
            endif ! useGPU
          else ! (lrs <= lre)
#ifdef MORE_GPU
            if (useGPU) then
              num = nstor*(lce-lcs+1)*size_of_datatype
#ifdef WITH_GPU_STREAMS
              my_stream = obj%gpu_setup%my_stream
              successGPU = gpu_memset_async(tmp1_dev, 0, num, my_stream)
              check_memcpy_gpu("hermitian_multiply: tmp1_dev", successGPU)
#else
              successGPU = gpu_memset(tmp1_dev, 0, num)
              check_memcpy_gpu("hermitian_multiply: tmp1_dev", successGPU)
#endif
            else ! useGPU 
              tmp1 = 0
            endif ! useGPU
#else /* MORE_GPU */
              tmp1 = 0
#endif /* MORE_GPU */
          endif ! (lrs <= lre)

          ! Sum up the results and send to processor row np

          if (useGPU) then
#ifdef WITH_MPI

#ifdef USE_CCL_MULTIPLY
#ifdef WITH_GPU_STREAMS
            my_stream = obj%gpu_setup%my_stream
            ccl_comm_rows = obj%gpu_setup%ccl_comm_rows

            successGPU = ccl_group_start()
            if (.not.successGPU) then
              print *,"Error in setting up ccl_group_start!"
              stop 1
            endif

            successGPU = ccl_reduce(tmp1_dev, tmp2_dev, &

#if REALCASE == 1
                         int(nstor*(lce-lcs+1),kind=c_size_t), &
#endif
#if COMPLEXCASE == 1
                         int(2*nstor*(lce-lcs+1),kind=c_size_t), &
#endif
#if REALCASE == 1
#ifdef DOUBLE_PRECISION
                         cclDouble, &
#endif
#ifdef SINGLE_PRECISION
                         cclFloat, &
#endif
#endif /* REALCASE */
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION
                         cclDouble, &
#endif
#ifdef SINGLE_PRECISION
                         cclFloat, &
#endif
#endif /* COMPLEXCASE */
                         cclSum, int(np,kind=c_int), ccl_comm_rows, my_stream)


            if (.not.successGPU) then
              print *,"Error in ccl_reduce"
              stop 1
            endif

            successGPU = ccl_group_end()
            if (.not.successGPU) then
              print *,"Error in setting up ccl_group_end!"
              stop 1
            endif
#endif /* WITH_GPU_STREAMS */

#endif /* USE_CCL_MULTIPLY */

#if defined(MORE_GPU) && !defined(USE_CCL_MULTIPLY)
            ! copy data to host
            num = nstor*(lce-lcs+1)*size_of_datatype
#ifdef WITH_GPU_STREAMS
            call gpu_memcpy_async_and_stream_synchronize &
            ("elpa_mult_at_b: tmp1_dev to tmp1", tmp1_dev, 0_c_intptr_t, &
                                                 !tmp1(1:nblk_mult,1:l_cols), &
                                                 tmp1(1:nstor,1:lce-lcs+1), &
                                                 1, 1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
            successGPU = gpu_memcpy(int(loc(tmp1),kind=c_intptr_t), &
                            tmp1_dev, num, gpuMemcpyDeviceToHost)
            check_memcpy_gpu("elpa_mult_at_b: tmp1_dev to tmp1", successGPU)
#endif
#endif /* defined(MORE_GPU) && !defined(USE_CCL_MULTIPLY) */

#if !defined(USE_CCL_MULTIPLY)
            ! communication already done before with CCL
            call obj%timer%start("mpi_communication")
            call mpi_reduce(tmp1, tmp2, int(nstor*(lce-lcs+1),kind=MPI_KIND),  MPI_MATH_DATATYPE_PRECISION, &
                          MPI_SUM, int(np,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
            call obj%timer%stop("mpi_communication")
#endif /* !defined(USE_CCL_MULTIPLY) */

#if defined(MORE_GPU) && !defined(USE_CCL_MULTIPLY)
            ! copy data to device
            num = nstor*(lce-lcs+1)*size_of_datatype
#ifdef WITH_GPU_STREAMS
            call gpu_memcpy_async_and_stream_synchronize &
            ("elpa_mult_at_b: tmp2 to tmp2_dev", tmp2_dev, 0_c_intptr_t, &
                                                 !tmp2(1:nblk_mult,1:l_cols), &
                                                 tmp2(1:nstor,1:lce-lcs+1), &
                                                 1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
            successGPU = gpu_memcpy(tmp2_dev, int(loc(tmp2),kind=c_intptr_t), &
                                    num, gpuMemcpyHostToDevice)
            check_memcpy_gpu("elpa_mult_at_b: tmp2 to tmp2_dev", successGPU)
#endif
#endif /* defined(MORE_GPU) && !defined(USE_CCL_MULTIPLY) */

#else /* WITH_MPI */

#ifdef MORE_GPU
            num = nstor*(lce-lcs+1)*size_of_datatype
            successGPU = gpu_memcpy(tmp2_dev, tmp1_dev, &
                                    num, gpuMemcpyDeviceToDevice)
            check_memcpy_gpu("elpa_mult_at_b: tmp2 to tmp2_dev", successGPU)
#else /* MORE_GPU */
            tmp2 = tmp1
#endif /* MORE_GPU */
#endif /* WITH_MPI */

          else ! useGPU
#ifdef WITH_MPI
            call obj%timer%start("mpi_communication")
            call mpi_reduce(tmp1, tmp2, int(nstor*(lce-lcs+1),kind=MPI_KIND),  MPI_MATH_DATATYPE_PRECISION, &
                          MPI_SUM, int(np,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
            call obj%timer%stop("mpi_communication")
#endif
          endif ! endif

          if (useGPU) then
#ifdef MORE_GPU
            if (my_prow==np) call gpu_copy_PRECISION_tmp2_c(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#else /* MORE_GPU */
#ifdef WITH_MPI
            ! Put the result into C
#ifndef DEVICE_POINTER
            if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp2(1:nstor,1:lce-lcs+1)
#else
            if (my_prow==np) c_tmp(nr_done+1:nr_done+nstor,lcs:lce) = tmp2(1:nstor,1:lce-lcs+1)
#endif
#else /* WITH_MPI */
            ! Put the result into C
#ifndef DEVICE_POINTER
            if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp1(1:nstor,1:lce-lcs+1)
#else
            if (my_prow==np) c_tmp(nr_done+1:nr_done+nstor,lcs:lce) = tmp1(1:nstor,1:lce-lcs+1)
#endif
            !tmp2(:,:) = 0.
#endif /* WITH_MPI */
            
#endif /* MORE_GPU */
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
#if defined(MORE_GPU) && defined(USE_CCL_MULTIPLY)
          if (.not.useGPU) then
#endif
              deallocate(tmp1, tmp2, stat=istat, errmsg=errorMessage)
              call check_alloc("elpa_mult_at_b_&
                &MATH_DATATYPE ", "tmp1", istat, errorMessage)
#if defined(MORE_GPU) && defined(USE_CCL_MULTIPLY)
          endif
#endif
        endif ! (lcs <= lce)

        nr_done = nr_done+nstor
        nstor=0
        if (useGPU) then
          num = l_rows*nblk_mult*size_of_datatype
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_memset_async(aux_mat_dev, 0, num, my_stream)
          check_memcpy_gpu("hermitian_multiply: aux_mat_dev", successGPU)
#else
          successGPU = gpu_memset(aux_mat_dev, 0, num)
          check_memcpy_gpu("hermitian_multiply: aux_mat_dev", successGPU)
#endif

#ifndef MORE_GPU
          ! additionally set HOST array to zero
          aux_mat(:,:) = 0
#endif
        else
          aux_mat(:,:) = 0
        endif
      endif ! (nstor==nblk_mult .or. nb*nblk+nblk >= l_rows_np)
    
#ifdef WITH_NVTX
      call nvtxRangePop() ! do nb = 0, (l_rows_np-1)/nblk
#endif
    enddo ! nb = 0, (l_rows_np-1)/nbl

#ifdef WITH_NVTX
    call nvtxRangePop() ! do np = 0, np_rows-1
#endif
  enddo ! main loop: np = 0, np_rows-1

!#ifdef MORE_GPU
!  if (useGPU) then
!    deallocate(tmp2, stat=istat, errmsg=errorMessage)
!    check_deallocate("elpa_mult_at_b: tmp1, tmp2", istat, errorMessage)
!   endif
!#endif

  if (useGPU) then
#if !defined(DEVICE_POINTER)
    successGPU = gpu_free(b_dev)
    check_dealloc_gpu("elpa_multiply_a_b: b_dev", successGPU)
#if !defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) && !defined(WITH_SYCL_GPU_VERSION)
    successGPU = gpu_host_unregister(int(loc(b),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_multiply_a_b: b", successGPU)
#endif

#ifdef MORE_GPU
    ! copy result c_dev back to CPU
    num = ldc*ldcCols*size_of_datatype
#ifdef WITH_GPU_STREAMS
    check_stream_synchronize_gpu("elpa_mult_at_b: c_dev -> c", successGPU)
    call gpu_memcpy_async_and_stream_synchronize &
    ("elpa_mult_at_b: c_dev to c", c_dev, 0_c_intptr_t, &
                                       c(1:ldc,1:ldcCols), &
                                       1, 1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
    successGPU = gpu_memcpy(int(loc(c),kind=c_intptr_t), c_dev, num,&
                  gpuMemcpyDeviceToHost)
    check_memcpy_gpu("elpa_mult_at_b: c_dev -> c", successGPU)

#endif
    successGPU = gpu_free(c_dev)
    check_dealloc_gpu("elpa_multiply_a_b: c_dev", successGPU)

#else /* MORE_GPU */
    ! c on host no copy back needed
#endif /* MORE_GPU */

#else /* DEVICE_POINTER */
    !successGPU = gpu_free(b_dev)
    !check_dealloc_gpu("elpa_multiply_a_b: b_dev", successGPU)

    !num = ldc*ldcCols*size_of_datatype
    !successGPU = gpu_memcpy(cDev, c_dev, num,&
    !              gpuMemcpyDeviceToDevice)
    !check_memcpy_gpu("elpa_mult_at_b: c_dev -> cDev", successGPU)

    !successGPU = gpu_free(c_dev)
    !check_dealloc_gpu("elpa_multiply_a_b: c_dev", successGPU)

#ifndef MORE_GPU
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_host_unregister(int(loc(a_tmp),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_mult_at_b: a_tmp", successGPU)
#endif
    deallocate(a_tmp, stat=istat, errmsg=errorMessage)
    check_deallocate("elpa_mult_at_b: a_tmp", istat, errorMessage)

    num = ldc*ldcCols*size_of_datatype
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_mult_at_b: c_tmp to c", successGPU)

    successGPU = gpu_memcpy_async(cDev,int(loc(c_tmp),kind=c_intptr_t),num,&
                  gpuMemcpyHostToDevice, my_stream)
    check_memcpy_gpu("elpa_mult_at_b: c_tmp -> c", successGPU)

    my_stream = obj%gpu_setup%my_stream
    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_mult_at_b: c_tmp -> c", successGPU)
    ! synchronize streamsPerThread; maybe not neccessary
    successGPU = gpu_stream_synchronize()
    check_stream_synchronize_gpu("elpa_mult_at_b: c_tmp -> c", successGPU)
#else
    successGPU = gpu_memcpy(cDev,int(loc(c_tmp),kind=c_intptr_t),num,&
                  gpuMemcpyHostToDevice)
    check_memcpy_gpu("elpa_mult_at_b: c_tmp -> c", successGPU)
#endif
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_host_unregister(int(loc(c_tmp),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_multiply_a_b: c_tmp", successGPU)
#endif

    deallocate(c_tmp, stat=istat, errmsg=errorMessage)
    check_deallocate("elpa_mult_at_b: c_tmp", istat, errorMessage)
#endif /* MORE_GPU */

#endif /* DEVICE_POINTER */
#if !defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) && !defined(WITH_SYCL_GPU_VERSION)
    nullify(aux_mat)
    !nullify(tmp1)

    successGPU = gpu_free_host(aux_host)
    check_host_dealloc_gpu("elpa_multiply_a_b: aux_host", successGPU)

    !successGPU = gpu_free_host(tmp1_host)
    !check_host_dealloc_gpu("elpa_multiply_a_b: tmp1_host", successGPU)
#else
    deallocate(aux_mat, stat=istat, errmsg=errorMessage)
    check_deallocate("elpa_mult_at_b: aux_mat", istat, errorMessage)

    !deallocate(tmp1, stat=istat, errmsg=errorMessage)
    !check_deallocate("elpa_mult_at_b: tmp1", istat, errorMessage)
#endif

    successGPU = gpu_free(aux_mat_dev)
    check_dealloc_gpu("elpa_multiply_a_b: aux_mat_dev", successGPU)

    successGPU = gpu_free(tmp1_dev)
    check_dealloc_gpu("elpa_multiply_a_b: tmp1_dev", successGPU)

#ifdef MORE_GPU
    successGPU = gpu_free(tmp2_dev)
    check_dealloc_gpu("elpa_multiply_a_b: tmp2_dev", successGPU)

    successGPU = gpu_free(aux_bc_dev)
    check_dealloc_gpu("elpa_multiply_a_b: aux_bc_dev", successGPU)
#endif

#if !defined(DEVICE_POINTER)
#ifdef MORE_GPU
    successGPU = gpu_free(a_dev)
    check_dealloc_gpu("elpa_mult_at_b: a_dev", successGPU)
#endif
#else
    !successGPU = gpu_free(a_dev)
    !check_dealloc_gpu("elpa_mult_at_b: a_dev", successGPU)
#endif
  else ! useGPU
    deallocate(aux_mat, stat=istat, errmsg=errorMessage)
    check_deallocate("elpa_mult_at_b: aux_mat", istat, errorMessage)
  endif ! useGPU

  deallocate(aux_bc, lrs_save, lre_save, stat=istat, errmsg=errorMessage)
  check_deallocate("elpa_mult_at_b: aux_bc, lrs_save, lre_save", istat, errorMessage)

  call obj%timer%stop("elpa_mult_at_b_&
  &MATH_DATATYPE&
  &_&
  &PRECISION&
  &"//gpuString)

