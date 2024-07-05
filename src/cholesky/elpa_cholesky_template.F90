!    This file is part of ELPA.
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

#include "../general/sanity.F90"
#include "../general/error_checking.inc"

#undef USE_CCL_CHOLESKY
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
#define USE_CCL_CHOLESKY
#endif

  use elpa1_compute
  use elpa_utilities
  use elpa_mpi
  use precision
  use elpa_abstract_impl
  use elpa_omp
  use elpa_blas_interfaces
  use elpa_gpu
  use mod_check_for_gpu
  use invert_trm_gpu !, only : gpu_copy_&
                     !        &PRECISION&
                     !        &_tmp1_tmp2, &
                     !        gpu_copy_&
                     !        &PRECISION&
                     !        &_a_tmp1
  use cholesky_gpu
  use mod_query_gpu_usage
#ifdef WITH_GPU_STREAMS
  use elpa_gpu_util
#endif
#if defined(USE_CCL_CHOLESKY)
  use elpa_ccl_gpu
#endif
#if defined(WITH_NVIDIA_GPU_VERSION) && defined(WITH_NVTX)
  use cuda_functions ! for NVTX labels
#endif
  implicit none
#include "../general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik)                           :: na, matrixRows, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
#ifndef DEVICE_POINTER
#ifdef USE_ASSUMED_SIZE
  MATH_DATATYPE(kind=rck)                    :: a(obj%local_nrows,*)
#else
  MATH_DATATYPE(kind=rck)                    :: a(obj%local_nrows,obj%local_ncols)
#endif
#else /* DEVICE_POINTER */
  type(c_ptr)                                :: aDev
!#if !defined(WITH_NVIDIA_CUSOLVER) && !defined(WITH_AMD_ROCSOLVER)
  MATH_DATATYPE(kind=rck), allocatable       :: a_tmp(:,:)
!#endif
#endif /* DEVICE_POINTER */
  integer(kind=ik)                           :: my_prow, my_pcol, np_rows, np_cols, myid
  integer(kind=MPI_KIND)                     :: mpierr, my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI, myidMPI
  integer(kind=ik)                           :: l_col1, l_row1, l_colx, l_rowx
  integer(kind=ik)                           :: n, nc, i, info
  integer(kind=BLAS_KIND)                    :: infoBLAS
  integer(kind=ik)                           :: lcs, lce, lrs, lre
  integer(kind=ik)                           :: tile_size, l_rows_tile, l_cols_tile

  MATH_DATATYPE(kind=rck), allocatable       :: tmp1(:), tmp2(:,:), tmatr(:,:), tmatc(:,:)
  logical                                    :: wantDebug
  logical                                    :: success
  integer(kind=ik)                           :: istat, debug, error
  character(200)                             :: errorMessage
  integer(kind=ik)                           :: nrThreads, limitThreads
  character(20)                              :: gpuString
  logical                                    :: successGPU
  logical                                    :: useGPU
  integer(kind=c_int)                        :: numGPU
  integer(kind=c_intptr_t)                   :: num
  integer(kind=c_intptr_t)                   :: tmp1_dev, tmatc_dev, tmatr_dev, a_dev, tmp2_dev
  integer(kind=c_intptr_t)                   :: info_dev, info_abs_accumulated_dev
  type(c_ptr)                                :: tmp1_mpi_dev
  MATH_DATATYPE(kind=rck), pointer           :: tmp1_mpi_fortran_ptr(:,:)
  integer(kind=c_intptr_t)                   :: a_off, tmatc_off, tmatr_off
  type(c_ptr)                                :: tmatc_mpi_dev
  MATH_DATATYPE(kind=rck), pointer           :: tmatc_mpi_fortran_ptr(:,:)

  integer(kind=c_intptr_t), parameter        :: size_of_datatype = size_of_&
                                                            &PRECISION&
                                                            &_&
                                                            &MATH_DATATYPE

  integer(kind=c_intptr_t)                   :: gpublasHandle, gpusolverHandle, my_stream, offset
  integer(kind=c_int)                        :: gpu_cholesky
  integer(kind=ik)                           :: blocking

  logical                                    :: useCCL
#if defined(USE_CCL_CHOLESKY)
  integer(kind=c_intptr_t)                   :: ccl_comm_rows, ccl_comm_cols
  integer(kind=c_int)                        :: cclDataType
  integer(kind=ik)                           :: k_datatype
  integer(kind=ik)                           :: nvs, nvr, nvc, lcm_s_t, nblks_tot, nblks_comm, nblks_skip
  logical                                    :: isSquareGridGPU = .false.
  logical                                    :: isSkewsymmetric = .false.
  integer(kind=c_intptr_t)                   :: aux_transpose_dev
#endif

#if defined(WITH_NVIDIA_CUSOLVER)
  integer(kind=c_intptr_t)                   :: gpusolver_buffer_dev
  integer(kind=c_intptr_t)                   :: gpusolver_buffer_host
  integer(kind=c_size_t)                     :: workspaceInBytesOnDevice, workspaceInBytesOnHost
#endif

#ifdef WITH_NVTX
  call nvtxRangePush("cholesky")
#endif

  success = .true.
  useGPU = .false.
  useCCL = .false.

#if !defined(DEVICE_POINTER)

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
  if (.not.(query_gpu_usage(obj, "ELPA_CHOLESKY", useGPU))) then
    print *,"ELPA_CHOLESKY: Problem querrying settings for GPU Aborting..."
    stop 1
  endif
#endif

  ! check whether the above setting should be overriden
  if (obj%is_set("gpu_cholesky") == 1) then
    call obj%get("gpu_cholesky", gpu_cholesky, error)
    if (error .ne. ELPA_OK) then
      print *,"Problem getting option for gpu_cholesky. Aborting..."
      stop 1
    endif
    if (useGPU .and. gpu_cholesky .eq. 0) then
      useGPU = .false.
    else if (.not.(useGPU) .and. gpu_cholesky .eq. 1) then
      useGPU = .true.
    else 
    endif
  else 
    ! no override by user 
  endif
#else /* DEVICE_POINTER */
  useGPU = .true.
#endif /* DEVICE_POINTER */

  if (.not.(useGPU)) then
#ifdef DEVICE_POINTER
    print *,"You used the interface for device pointers for elpa_cholesky but did not specify GPU usage!. Aborting..."
    stop 1
#endif
  endif

  if(useGPU) then
    gpuString = "_gpu"
  else
    gpuString = ""
  endif
  
  if (useGPU) then
    call obj%timer%start("check_for_gpu")
    if (check_for_gpu(obj, myid, numGPU)) then
      ! set the neccessary parameters
      call set_gpu_parameters()
    else
      print *,"ELPA_CHOLESKY: GPUs are requested but not detected! Aborting..."
      success = .false.
      return
    endif
    call obj%timer%stop("check_for_gpu")
  else ! useGPU
  endif ! useGPU

#if defined(USE_CCL_CHOLESKY)
  if (useGPU) then
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
  endif ! useGPU
#endif /* defined(USE_CCL_CHOLESKY) */
  
  call obj%timer%start("elpa_cholesky_&
  &MATH_DATATYPE&
  &_&
  &PRECISION&
  &"//gpuString)

#ifdef WITH_OPENMP_TRADITIONAL
  ! store the number of OpenMP threads used in the calling function
  ! restore this at the end of ELPA 2
  omp_threads_caller = omp_get_max_threads()

  ! check the number of threads that ELPA should use internally
#if defined(THREADING_SUPPORT_CHECK) && defined(ALLOW_THREAD_LIMITING) && !defined(HAVE_SUFFICIENT_MPI_THREADING_SUPPORT)
  call obj%get("limit_openmp_threads",limitThreads,error)
  if (limitThreads .eq. 0) then
#endif
     if (obj%is_set("omp_threads") == 1) then
       ! user set omp_threads, honour this
       call obj%get("omp_threads", nrThreads, error)
       if (error .ne. ELPA_OK) then
         print *,"cannot get option for omp_threads. Aborting..."
         stop 1
       endif
       call omp_set_num_threads(nrThreads)
     else
       ! use the max threads
       call obj%set("omp_threads",omp_threads_caller, error)
       if (error .ne. ELPA_OK) then
         print *,"cannot set option for omp_threads. Aborting..."
         stop 1
       endif
       nrThreads = omp_threads_caller
       call omp_set_num_threads(omp_threads_caller)
     endif
#if defined(THREADING_SUPPORT_CHECK) && defined(ALLOW_THREAD_LIMITING) && !defined(HAVE_SUFFICIENT_MPI_THREADING_SUPPORT)
  else
    nrThreads = 1
    call omp_set_num_threads(nrThreads)
  endif
#endif

#else
  nrThreads=1
#endif

  na         = obj%na
  matrixRows = obj%local_nrows
  matrixCols = obj%local_ncols
  nblk       = obj%nblk

  call obj%get("debug",debug,error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "ELPA_CHOLESKY: Problem getting option for debug settings. Aborting..."
    success = .false.
    return
  endif
  if (debug == 1) then
    wantDebug = .true.
  else
    wantDebug = .false.
  endif

  mpi_comm_all    = obj%mpi_setup%mpi_comm_parent
  mpi_comm_cols   = obj%mpi_setup%mpi_comm_cols
  mpi_comm_rows   = obj%mpi_setup%mpi_comm_rows
    
  myid    = obj%mpi_setup%myRank_comm_parent
  my_prow = obj%mpi_setup%myRank_comm_rows
  my_pcol = obj%mpi_setup%myRank_comm_cols
        
  np_rows = obj%mpi_setup%nRanks_comm_rows
  np_cols = obj%mpi_setup%nRanks_comm_cols

  success = .true.

  ! Matrix is split into tiles; work is done only for tiles on the diagonal or above

  call obj%timer%start("prepare")

  if (obj%is_set("blocking_in_cholesky") == 1) then
    call obj%get("blocking_in_cholesky", blocking, error)
    if (error .ne. ELPA_OK) then
      write(error_unit,*) "ELPA_CHOLESKY: Problem in getting keyword 'blocking_in_cholesky'. Aborting..."
      stop 1
    endif
  else
    if (useGPU) then
      blocking = 1024
    else
      blocking = 128
    endif
    call obj%set("blocking_in_cholesky", blocking, error)
    if (error .ne. ELPA_OK) then
      write(error_unit,*) "ELPA_CHOLESKY: Problem in setting keyword 'blocking_in_cholesky'. Aborting..."
      stop 1
    endif
  endif 


  tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size
  tile_size = ((blocking*max(np_rows,np_cols)-1)/tile_size+1)*tile_size ! make local tiles at least 128 wide

  l_rows_tile = tile_size/np_rows ! local rows of a tile
  l_cols_tile = tile_size/np_cols ! local cols of a tile

  if (useGPU) then
#if defined(WITH_NVIDIA_CUSOLVER) || defined(WITH_AMD_ROCSOLVER)    
    successGPU = gpu_malloc(info_dev, 1*sizeof(c_int))
    check_alloc_gpu("elpa_cholesky: info_dev", successGPU)

    successGPU = gpu_malloc(info_abs_accumulated_dev, 1*sizeof(c_int))
    check_alloc_gpu("elpa_cholesky: info_abs_accumulated_dev", successGPU)
    
    successGPU = gpu_memset(info_abs_accumulated_dev, 0, 1*sizeof(c_int))
    check_memcpy_gpu("elpa_cholesky: memset info_abs_accumulated_dev", successGPU)
#endif

    successGPU = gpu_malloc(tmp1_dev, nblk*(nblk+1)/2*size_of_datatype)
    check_alloc_gpu("elpa_cholesky: tmp1_dev", successGPU)

    successGPU = gpu_malloc(tmp2_dev, nblk*nblk*size_of_datatype)
    check_alloc_gpu("elpa_cholesky: tmp2_dev", successGPU)

    successGPU = gpu_malloc(tmatc_dev, matrixCols*nblk*size_of_datatype)
    check_alloc_gpu("elpa_cholesky: tmatc_dev", successGPU)

    successGPU = gpu_malloc(tmatr_dev, matrixRows*nblk*size_of_datatype)
    check_alloc_gpu("elpa_cholesky: tmatr_dev", successGPU)

#ifndef DEVICE_POINTER
    successGPU = gpu_malloc(a_dev, matrixRows*matrixCols*size_of_datatype)
    check_alloc_gpu("elpa_cholesky: a_dev", successGPU)
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_host_register(int(loc(a),kind=c_intptr_t), &
                    matrixRows*matrixCols * size_of_datatype,&
                    gpuHostRegisterDefault)
    check_host_register_gpu("elpa_cholesky: a", successGPU)
#endif
#else /* DEVICE_POINTER */
    a_dev = transfer(aDev, a_dev)
!#if !defined(WITH_NVIDIA_CUSOLVER) && !defined(WITH_AMD_ROCSOLVER)
    allocate(a_tmp(obj%local_nrows,obj%local_ncols), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_cholesky: a_tmp", istat, errorMessage)
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_host_register(int(loc(a_tmp),kind=c_intptr_t), &
                    matrixRows*matrixCols * size_of_datatype,&
                    gpuHostRegisterDefault)
    check_host_register_gpu("elpa_cholesky: a_tmp", successGPU)
#endif /* WITH_GPU_STREAMS */
!#endif /* !defined(WITH_NVIDIA_CUSOLVER) && !defined(WITH_AMD_ROCSOLVER) */
#endif /* DEVICE_POINTER */
  endif ! useGPU

#ifdef USE_CCL_CHOLESKY
    ! for gpu_transpose_vectors on non-square grids
    if (np_rows==np_cols .and. (.not. isSkewsymmetric)) then
      ! isSquareGridGPU = .true. ! TODO_23_11 - switched off for now, add test for arbitrary grid mapping and switch back on
    endif

    if (.not. isSquareGridGPU) then
      lcm_s_t   = least_common_multiple(np_rows,np_cols)
      nvs = 1 ! global index where to start in tmatc_dev/tmatr_dev; min value of n is 1 for correct gpu_malloc
      nvr  = na ! global length of tmatc_dev/tmatr_dev
      nvc = nblk ! number of columns in tmatc_dev/tmatr_dev
      nblks_tot = (nvr+nblk-1)/nblk ! number of blocks corresponding to nvr
      ! Get the number of blocks to be skipped at the beginning
      ! This must be a multiple of lcm_s_t (else it is getting complicated),
      ! thus some elements before nvs will be accessed/set.
      nblks_skip = ((nvs-1)/(nblk*lcm_s_t))*lcm_s_t
    
      successGPU = gpu_malloc(aux_transpose_dev, ((nblks_tot-nblks_skip+lcm_s_t-1)/lcm_s_t) * nblk * nvc * size_of_datatype)
      check_alloc_gpu("tridiag: aux_transpose_dev", successGPU)
    endif ! .not. isSquareGridGPU
#endif /* USE_CCL_CHOLESKY */

#ifdef WITH_NVTX
  call nvtxRangePush("allocate tmp1, tmp2, tmatr, tmatc")
#endif

  allocate(tmp1(nblk*nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_cholesky: tmp1", istat, errorMessage)

  allocate(tmp2(nblk,nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_cholesky: tmp2", istat, errorMessage)
  
  allocate(tmatr(matrixRows,nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_cholesky: tmatr", istat, errorMessage)

  allocate(tmatc(matrixCols,nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_cholesky: tmatc", istat, errorMessage)

#ifdef WITH_NVTX
  call nvtxRangePop() ! allocate tmp1, tmp2, tmatr, tmatc
#endif


#ifdef WITH_GPU_STREAMS
  if (useGPU) then
    successGPU = gpu_host_register(int(loc(tmp1),kind=c_intptr_t), &
                    nblk*(nblk+1)/2 * size_of_datatype, gpuHostRegisterDefault)
    check_host_register_gpu("elpa_cholesky: tmp1", successGPU)

    successGPU = gpu_host_register(int(loc(tmp2),kind=c_intptr_t), &
                    nblk*nblk * size_of_datatype, gpuHostRegisterDefault)
    check_host_register_gpu("elpa_cholesky: tmp2", successGPU)

    successGPU = gpu_host_register(int(loc(tmatr),kind=c_intptr_t), &
                      matrixRows*nblk * size_of_datatype, gpuHostRegisterDefault)
    check_host_register_gpu("elpa_cholesky: tmatr", successGPU)

    successGPU = gpu_host_register(int(loc(tmatc),kind=c_intptr_t), &
                      matrixCols*nblk * size_of_datatype, gpuHostRegisterDefault)
    check_host_register_gpu("elpa_cholesky: tmatc", successGPU)
  endif
#endif



#ifndef DEVICE_POINTER
  if (useGPU) then
    num = matrixRows*matrixCols
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    call gpu_memcpy_async_and_stream_synchronize &
        ("elpa_cholesky: a to a_dev", a_dev, 0_c_intptr_t, a(1:obj%local_nrows,1:obj%local_ncols), &
          1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
    successGPU = gpu_memcpy(a_dev, int(loc(a(1,1)),kind=c_intptr_t), &
                            num*size_of_datatype, gpuMemcpyHostToDevice)
    check_memcpy_gpu("elpa_cholesky 1: memcpy a-> a_dev", successGPU)
#endif
  endif
#endif /* DEVICE_POINTER */

#if defined(WITH_NVIDIA_CUSOLVER)
  if (useGPU) then
    gpusolverHandle = obj%gpu_setup%gpusolverHandleArray(0)
    call gpusolver_Xpotrf_bufferSize(gpusolverHandle, 'U', nblk, PRECISION_CHAR, a_dev, matrixRows, &
                                    workspaceInBytesOnDevice, workspaceInBytesOnHost)

    successGPU = gpu_malloc(gpusolver_buffer_dev, workspaceInBytesOnDevice)
    check_alloc_gpu("elpa_cholesky: gpusolver_buffer_dev", successGPU)

    if (workspaceInBytesOnHost > 0) then
      successGPU = gpu_malloc_host(gpusolver_buffer_host, workspaceInBytesOnHost)
      check_host_alloc_gpu("elpa_cholesky: gpusolver_buffer_host", successGPU)
    endif
  endif
#endif

  call obj%timer%stop("prepare")
  call obj%timer%start("loop1")
  do n = 1, na, nblk

#ifdef WITH_NVTX
    call nvtxRangePush("do n = 1, na, nblk")
#endif

    ! Calculate first local row and column of the still remaining matrix
    ! on the local processor

    l_row1 = local_index(n, my_prow, np_rows, nblk, +1)
    l_col1 = local_index(n, my_pcol, np_cols, nblk, +1)

    l_rowx = local_index(n+nblk, my_prow, np_rows, nblk, +1)
    l_colx = local_index(n+nblk, my_pcol, np_cols, nblk, +1)

    if (n+nblk > na) then

      ! This is the last step, just do a Cholesky-Factorization
      ! of the remaining block

      if (useGPU) then
        if (my_prow==prow(n, nblk, np_rows) .and. my_pcol==pcol(n, nblk, np_cols)) then
#if defined(WITH_NVIDIA_CUSOLVER) || defined(WITH_AMD_ROCSOLVER)
          call obj%timer%start("gpusolver")
          gpusolverHandle = obj%gpu_setup%gpusolverHandleArray(0)
          a_off = (l_row1-1 + (l_col1-1)*matrixRows) * size_of_datatype
#ifdef WITH_NVTX
          call nvtxRangePush("gpusolver_POTRF last")
#endif

#if defined(WITH_AMD_ROCSOLVER)
          call gpusolver_PRECISION_POTRF('U', na-n+1, a_dev+a_off, matrixRows, info_dev, gpusolverHandle)
          my_stream = obj%gpu_setup%my_stream
          if (wantDebug) call gpu_check_device_info(info_dev, my_stream)
          call gpu_accumulate_device_info(info_abs_accumulated_dev, info_dev, my_stream)
#endif
#if defined(WITH_NVIDIA_CUSOLVER)
          call gpusolver_Xpotrf(gpusolverHandle, 'U', na-n+1, PRECISION_CHAR, a_dev+a_off, matrixRows, &
                                gpusolver_buffer_dev , workspaceInBytesOnDevice, &
                                gpusolver_buffer_host, workspaceInBytesOnHost, info_dev)
          my_stream = obj%gpu_setup%my_stream
          !if (wantDebug) call gpu_check_device_info(info_dev, my_stream)
          call gpu_accumulate_device_info(info_abs_accumulated_dev, info_dev, my_stream)
#endif

#ifdef WITH_NVTX
          call nvtxRangePop() ! gpusolver_POTRF last
#endif
          call obj%timer%stop("gpusolver")
#else /* defined(WITH_NVIDIA_CUSOLVER) || defined(WITH_AMD_ROCSOLVER) */

          call obj%timer%start("lapack")
          num = matrixRows*matrixCols
#ifndef DEVICE_POINTER
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_cholesky: a_dev to a", a_dev, 0_c_intptr_t, &
                a(1:obj%local_nrows,1:obj%local_ncols), &
                1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
          successGPU = gpu_memcpy(int(loc(a(1,1)),kind=c_intptr_t), a_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a", successGPU)
#endif /* WITH_GPU_STREAMS */

          call PRECISION_POTRF('U', int(na-n+1,kind=BLAS_KIND), a(l_row1,l_col1), &
                             int(matrixRows,kind=BLAS_KIND), infoBLAS )
          info = int(infoBLAS,kind=ik)

          num = matrixRows*matrixCols
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream

          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_cholesky: a -> a_dev", a_dev, 0_c_intptr_t, a(1:obj%local_nrows,1:obj%local_ncols), &
                1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
          successGPU = gpu_memcpy(a_dev, int(loc(a(1,1)),kind=c_intptr_t), num*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a", successGPU)
#endif /* WITH_GPU_STREAMS */

#else /* DEVICE_POINTER */

          num = matrixRows*matrixCols
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_cholesky: a_dev to a_tmp", a_dev, 0_c_intptr_t, a_tmp(1:obj%local_nrows,1:obj%local_ncols), &
                1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
          successGPU = gpu_memcpy(int(loc(a_tmp(1,1)),kind=c_intptr_t), a_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)
#endif /* WITH_GPU_STREAMS */

          call PRECISION_POTRF('U', int(na-n+1,kind=BLAS_KIND), a_tmp(l_row1,l_col1), &
                                int(matrixRows,kind=BLAS_KIND), infoBLAS )
          info = int(infoBLAS,kind=ik)

          num = matrixRows*matrixCols
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_cholesky: a_tmp -> a_dev", a_dev, 0_c_intptr_t, a_tmp(1:obj%local_nrows,1:obj%local_ncols), &
                1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
          successGPU = gpu_memcpy(a_dev, int(loc(a_tmp(1,1)),kind=c_intptr_t), num*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)
#endif /* WITH_GPU_STREAMS */

#endif /* DEVICE_POINTER */
          call obj%timer%stop("lapack")

          if (info/=0) then
            if (wantDebug) write(error_unit,*) "elpa_cholesky_&
            &MATH_DATATYPE&

#if REALCASE == 1
            &: Error in dpotrf: ",info
#endif
#if COMPLEXCASE == 1
            &: Error in zpotrf: ",info
#endif
            success = .false. ! "
            return
          endif ! info
#endif /* defined(WITH_NVIDIA_CUSOLVER) || defined(WITH_AMD_ROCSOLVER) */
        endif ! (my_prow==prow(n, nblk, np_rows) .and. my_pcol==pcol(n, nblk, np_cols))

      else ! useGPU
        if (my_prow==prow(n, nblk, np_rows) .and. my_pcol==pcol(n, nblk, np_cols)) then
          call obj%timer%start("blas")

#ifndef DEVICE_POINTER
          call PRECISION_POTRF('U', int(na-n+1,kind=BLAS_KIND), a(l_row1,l_col1), &
                             int(matrixRows,kind=BLAS_KIND), infoBLAS )
#endif
          info = int(infoBLAS,kind=ik)
          call obj%timer%stop("blas")

          if (info/=0) then
            if (wantDebug) write(error_unit,*) "elpa_cholesky_&
            &MATH_DATATYPE&

#if REALCASE == 1
            &: Error in dpotrf: ",info
#endif
#if COMPLEXCASE == 1
            &: Error in zpotrf: ",info
#endif
            success = .false. ! "
            return
          endif

        endif
      endif ! useGPU
      
#ifdef WITH_NVTX
      call nvtxRangePop() ! do n = 1, na, nblk
#endif      
      exit ! Loop
    endif ! (n+nblk > na) 

    ! This is not the last step

    if (my_prow==prow(n, nblk, np_rows)) then

      if (my_pcol==pcol(n, nblk, np_cols)) then

        if (useGPU) then
#if defined(WITH_NVIDIA_CUSOLVER) || defined(WITH_AMD_ROCSOLVER)
          call obj%timer%start("gpusolver")
          gpusolverHandle = obj%gpu_setup%gpusolverHandleArray(0)
          a_off = (l_row1-1 + (l_col1-1)*matrixRows) * size_of_datatype
#ifdef WITH_NVTX
          call nvtxRangePush("gpusolver_POTRF")
#endif

#if defined(WITH_AMD_ROCSOLVER)
          call gpusolver_PRECISION_POTRF('U', nblk, a_dev+a_off, matrixRows, info_dev, gpusolverHandle)
          my_stream = obj%gpu_setup%my_stream
          if (wantDebug) call gpu_check_device_info(info_dev, my_stream)
          call gpu_accumulate_device_info(info_abs_accumulated_dev, info_dev, my_stream)
#endif
#if defined(WITH_NVIDIA_CUSOLVER)         
          call gpusolver_Xpotrf(gpusolverHandle, 'U', nblk, PRECISION_CHAR, a_dev+a_off, matrixRows, &
                                gpusolver_buffer_dev , workspaceInBytesOnDevice, &
                                gpusolver_buffer_host, workspaceInBytesOnHost, info_dev)
          my_stream = obj%gpu_setup%my_stream
          !if (wantDebug) call gpu_check_device_info(info_dev, my_stream)
          call gpu_accumulate_device_info(info_abs_accumulated_dev, info_dev, my_stream)
#endif

#ifdef WITH_NVTX
          call nvtxRangePop() ! gpusolver_POTRF
#endif

          call obj%timer%stop("gpusolver")
#else /* defined(WITH_NVIDIA_CUSOLVER) || defined(WITH_AMD_ROCSOLVER) */
          call obj%timer%start("lapack")

          num = matrixRows*matrixCols
#ifndef DEVICE_POINTER

#ifdef WITH_NVTX
          call nvtxRangePush("memcpy D-H a_dev->a")
#endif
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_cholesky: a_dev -> a", a_dev, 0_c_intptr_t, a(1:obj%local_nrows,1:obj%local_ncols), &
                1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
          successGPU = gpu_memcpy(int(loc(a(1,1)),kind=c_intptr_t), a_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a", successGPU)
#endif /* WITH_GPU_STREAMS */
#ifdef WITH_NVTX
          call nvtxRangePop() ! memcpy D-H a_dev->a
#endif


#ifdef WITH_NVTX
          call nvtxRangePush("POTRF")
#endif
          call PRECISION_POTRF('U', int(nblk,kind=BLAS_KIND), a(l_row1,l_col1), &
                               int(matrixRows,kind=BLAS_KIND) , infoBLAS )
          info = int(infoBLAS,kind=ik)
#ifdef WITH_NVTX
          call nvtxRangePop() !  POTRF
#endif

          num = matrixRows*matrixCols
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
          ("elpa_cholesky: a -> a_dev", a_dev, 0_c_intptr_t, &
                                          a(1:obj%local_nrows,1:obj%local_ncols), &
                                          1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
          successGPU = gpu_memcpy(a_dev, int(loc(a(1,1)),kind=c_intptr_t), num* size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a", successGPU)
#endif /* WITH_GPU_STREAMS */

#else /* DEVICE_POINTER */

          num = matrixRows*matrixCols
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_cholesky: a_dev -> a_tmp", a_dev, 0_c_intptr_t, a_tmp(1:obj%local_nrows,1:obj%local_ncols), &
                1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
          successGPU = gpu_memcpy(int(loc(a_tmp(1,1)),kind=c_intptr_t), a_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)
#endif /* WITH_GPU_STREAMS */

          call PRECISION_POTRF('U', int(nblk,kind=BLAS_KIND), a_tmp(l_row1,l_col1), &
                               int(matrixRows,kind=BLAS_KIND) , infoBLAS )
          info = int(infoBLAS,kind=ik)

          num = matrixRows*matrixCols
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_cholesky: a_tmp -> a_dev", a_dev, 0_c_intptr_t, a_tmp(1:obj%local_nrows,1:obj%local_ncols), &
                1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
          successGPU = gpu_memcpy(a_dev, int(loc(a_tmp(1,1)),kind=c_intptr_t), num*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)
#endif /* WITH_GPU_STREAMS */

#endif /* DEVICE_POINTER */
          call obj%timer%stop("lapack")

          if (info/=0) then
            if (wantDebug) write(error_unit,*) "elpa_cholesky_&
            &MATH_DATATYPE&

#if REALCASE == 1
            &: Error in dpotrf: ",info
#endif
#if COMPLEXCASE == 1
            &: Error in zpotrf: ",info
#endif
            success = .false. ! "
            return
          endif ! info
#endif /* defined(WITH_NVIDIA_CUSOLVER) || defined(WITH_AMD_ROCSOLVER) */
        else ! useGPU
          ! The process owning the upper left remaining block does the
          ! Cholesky-Factorization of this block
          call obj%timer%start("blas")

#ifndef DEVICE_POINTER
          call PRECISION_POTRF('U', int(nblk,kind=BLAS_KIND), a(l_row1,l_col1), &
                               int(matrixRows,kind=BLAS_KIND) , infoBLAS )
#endif
          info = int(infoBLAS,kind=ik)
          call obj%timer%stop("blas")

          if (info/=0) then
            if (wantDebug) write(error_unit,*) "elpa_cholesky_&
            &MATH_DATATYPE&

#if REALCASE == 1
            &: Error in dpotrf 2: ",info
#endif
#if COMPLEXCASE == 1
            &: Error in zpotrf 2: ",info

#endif
            success = .false. ! "
            return
          endif ! info/=0
        endif ! useGPU
 
        if (useGPU) then
          my_stream = obj%gpu_setup%my_stream
          call gpu_copy_PRECISION_a_tmp1 (a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nblk, my_stream)
        else ! useGPU
          nc = 0
          do i=1,nblk
#ifndef DEVICE_POINTER
            tmp1(nc+1:nc+i) = a(l_row1:l_row1+i-1,l_col1+i-1)
#endif
            nc = nc+i
          enddo
        endif ! useGPU
      endif ! (my_pcol==pcol(n, nblk, np_cols))

#ifdef WITH_MPI
      if (useGPU .and. .not. useCCL) then
        num = nblk*(nblk+1)/2
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_cholesky: tmp1_dev -> tmp1", tmp1_dev, 0_c_intptr_t, tmp1(1:num), &
                1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
        successGPU = gpu_memcpy(int(loc(tmp1),kind=c_intptr_t), tmp1_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("elpa_cholesky: tmp1_dev to tmp1", successGPU)
#endif /* WITH_GPU_STREAMS */
      endif ! (useGPU .and. .not. useCCL)

      if (useCCL) then
#ifdef USE_CCL_CHOLESKY
#ifdef WITH_NVTX
        call nvtxRangePush("ccl_bcast tmp1_dev")
#endif         
        call obj%timer%start("ccl_bcast")
        my_stream = obj%gpu_setup%my_stream
        ccl_comm_cols = obj%gpu_setup%ccl_comm_cols
       
        successGPU = ccl_bcast(tmp1_dev, tmp1_dev, int(k_datatype*nblk*(nblk+1)/2,kind=c_size_t), cclDataType, &
                               int(pcol(n, nblk, np_cols),kind=c_int), ccl_comm_cols, my_stream)

        if (.not.successGPU) then
          print *,"Error in ccl_bcast"
          stop 1
        endif

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("elpa_cholesky: ccl_bcast", successGPU)
        call obj%timer%stop("ccl_bcast")
#ifdef WITH_NVTX
        call nvtxRangePop() ! ccl_bcast tmp1_dev"
#endif 
#endif /* USE_CCL_CHOLESKY */

      else ! useCCL
        call obj%timer%start("mpi_communication")

        call MPI_Bcast(tmp1, int(nblk*(nblk+1)/2,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                      int(pcol(n, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)

        call obj%timer%stop("mpi_communication")
      endif ! useCCL

      if (useGPU .and. .not. useCCL) then
        num = nblk*(nblk+1)/2
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        call gpu_memcpy_async_and_stream_synchronize &
            ("elpa_cholesky: tmp1 -> tmp1_dev", tmp1_dev, 0_c_intptr_t, tmp1(1:num), &
              1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
        successGPU = gpu_memcpy(tmp1_dev, int(loc(tmp1),kind=c_intptr_t), num*size_of_datatype, gpuMemcpyHostToDevice)
        check_memcpy_gpu("elpa_cholesky: tmp1 to tmp1_dev", successGPU)
#endif
      endif ! (useGPU .and. .not. useCCL)
#endif /* WITH_MPI */

      if (useGPU) then
        my_stream = obj%gpu_setup%my_stream
        call gpu_copy_PRECISION_tmp1_tmp2 (tmp1_dev, tmp2_dev, nblk, nblk, my_stream)
      else ! useGPU
        nc = 0
        do i=1,nblk
          tmp2(1:i,i) = tmp1(nc+1:nc+i)
          nc = nc+i
        enddo
      endif ! useGPU

      if (useGPU) then
        call obj%timer%start("gpublas")
        gpublasHandle = obj%gpu_setup%gpublasHandleArray(0)
        if (matrixCols-l_colx+1 > 0) then
          a_off = (l_row1-1 + (l_colx-1)*matrixRows) * size_of_datatype
#ifdef WITH_NVTX
          call nvtxRangePush("gpublas trsm")
#endif          
          call gpublas_PRECISION_TRSM('L', 'U', BLAS_TRANS_OR_CONJ, 'N', nblk, matrixCols-l_colx+1, ONE, &
                            tmp2_dev, nblk, a_dev+a_off, matrixRows, gpublasHandle)
          if (wantDebug .and. gpu_vendor() /= SYCL_GPU) successGPU = gpu_DeviceSynchronize()
#ifdef WITH_NVTX
          call nvtxRangePop() ! gpublas trsm
#endif                             
        endif
        call obj%timer%stop("gpublas")

      else ! useGPU

        call obj%timer%start("lapack")
#ifndef DEVICE_POINTER
        if (matrixCols-l_colx+1>0) then
          call PRECISION_TRSM('L', 'U', BLAS_TRANS_OR_CONJ, 'N', int(nblk,kind=BLAS_KIND),  &
                            int(matrixCols-l_colx+1,kind=BLAS_KIND), ONE, tmp2, &
                            int(ubound(tmp2,dim=1),kind=BLAS_KIND), a(l_row1,l_colx), int(matrixRows,kind=BLAS_KIND) )
        endif
#endif
        call obj%timer%stop("lapack")
      endif ! useGPU
    endif ! (my_prow==prow(n, nblk, np_rows))


    if (useGPU) then
      if (my_prow==prow(n, nblk, np_rows)) then
        ! if matrixCols-l_colx+1 == 0 kernel launch with 0 blocks => raises error
        call obj%timer%start("a_tmatc_kernel")
        if (matrixCols-l_colx+1>0) then
           my_stream = obj%gpu_setup%my_stream
           call gpu_copy_PRECISION_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, matrixCols, l_colx, l_row1, my_stream)
        endif
        call obj%timer%stop("a_tmatc_kernel")
      endif
    else ! useGPU
      do i=1,nblk
#ifndef DEVICE_POINTER
#if REALCASE == 1
        if (my_prow==prow(n, nblk, np_rows)) tmatc(l_colx:matrixCols,i) = a(l_row1+i-1,l_colx:matrixCols)
#endif
#if COMPLEXCASE == 1
        if (my_prow==prow(n, nblk, np_rows)) tmatc(l_colx:matrixCols,i) = conjg(a(l_row1+i-1,l_colx:matrixCols))
#endif
#endif /* DEVICE_POINTER */
      enddo
    endif ! useGPU

!#ifdef WITH_CUDA_AWARE_MPI_2
!      tmatc_mpi_dev = transfer(tmatc_dev, tmatc_mpi_dev)
!      ! and associate a fortran pointer
!      call c_f_pointer(tmatc_mpi_dev, tmatc_mpi_fortran_ptr, [matrixCols,nblk])
!      
!      if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
!      successGPU = gpu_devicesynchronize()
!      check_memcpy_gpu("cholesky: device_synchronize", successGPU)
!      if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
!
!      do i=1,nblk
!        call obj%timer%start("mpi_cuda_communication")
!        if (matrixCols-l_colx+1>0) &
!        call MPI_Bcast(tmatc_mpi_fortran_ptr(l_colx,i), int(matrixCols-l_colx+1,kind=MPI_KIND), &
!                       MPI_MATH_DATATYPE_PRECISION, &
!                       int(prow(n, nblk, np_rows),kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
!
!        call obj%timer%stop("mpi_cuda_communication")
!      enddo
!#endif /* WITH_CUDA_AWARE_MPI_2 */

#ifdef WITH_MPI
    if (matrixCols-l_colx+1 > 0) then
      if (useGPU .and. .not. useCCL) then
        num = matrixCols*nblk
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        call gpu_memcpy_async_and_stream_synchronize &
            ("elpa_cholesky: tmatc_dev -> tmatc", tmatc_dev, 0_c_intptr_t, tmatc(1:matrixCols,1:nblk), &
              1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
        successGPU = gpu_memcpy(int(loc(tmatc),kind=c_intptr_t), tmatc_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("elpa_cholesky: tmatc_dev to tmatc", successGPU)
#endif
      endif ! (useGPU .and. .not. useCCL)

      if (useCCL) then
#ifdef USE_CCL_CHOLESKY
#ifdef WITH_NVTX
        call nvtxRangePush("do i=1,nblk ccl_bcast(tmatc_dev+offset_i)")
#endif 
        call obj%timer%start("ccl_bcast")
        my_stream = obj%gpu_setup%my_stream
        ccl_comm_rows = obj%gpu_setup%ccl_comm_rows
        successGPU = ccl_group_start()
        if (.not.successGPU) then
          print *,"Error in setting up ccl_group_start!"
          stop 1
        endif

        do i=1,nblk
          offset = ((l_colx-1) + (i-1) * matrixCols ) * size_of_datatype

          successGPU = ccl_bcast(tmatc_dev+offset, tmatc_dev+offset, &
                       int(k_datatype*(matrixCols-l_colx+1),kind=c_size_t), cclDataType, &
                       int(prow(n, nblk, np_rows),kind=c_int), ccl_comm_rows, my_stream)

          if (.not. successGPU) then
            print *,"Error in ccl_bcast"
            stop 1
          endif
        enddo

        successGPU = ccl_group_end()
        if (.not.successGPU) then
          print *,"Error in setting up ccl_group_end!"
          stop 1
        endif

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("elpa_cholesky: ccl_bcast", successGPU)

        call obj%timer%stop("ccl_bcast")
#ifdef WITH_NVTX
        call nvtxRangePop() !  do i=1,nblk ccl_bcast(tmatc_dev+offset_i)
#endif 
#endif /* USE_CCL_CHOLESKY */
      else ! useCCL

#ifdef WITH_NVTX
        call nvtxRangePush("MPI_Bcast tmatc")
#endif
        call obj%timer%start("mpi_communication")
        ! do i=1,nblk
        !   call MPI_Bcast(tmatc(l_colx,i), int(matrixCols-l_colx+1,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
        !                  int(prow(n, nblk, np_rows),kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
        ! enddo
        call MPI_Bcast(tmatc(1,1), int(matrixCols*nblk,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                       int(prow(n, nblk, np_rows),kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
        call obj%timer%stop("mpi_communication")
#ifdef WITH_NVTX
        call nvtxRangePop() ! MPI_Bcast tmatc
#endif
      endif ! useCCL

      if (useGPU .and. .not. useCCL) then
        num = matrixCols*nblk
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        call gpu_memcpy_async_and_stream_synchronize &
            ("elpa_cholesky: tmatc -> tmatc_dev", tmatc_dev, 0_c_intptr_t, tmatc(1:matrixCols,1:nblk), &
              1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
        successGPU = gpu_memcpy(tmatc_dev, int(loc(tmatc),kind=c_intptr_t), num*size_of_datatype, gpuMemcpyHostToDevice)
        check_memcpy_gpu("elpa_cholesky: tmatc to tmatc_dev", successGPU)
#endif
      endif ! (useGPU .and. .not. useCCL)
    endif ! (matrixCols-l_colx+1 > 0)
#endif /* WITH_MPI */


#if !defined(WITH_MPI)
    if (useGPU  .and. .not. useCCL) then
      num = matrixCols*nblk
#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
      call gpu_memcpy_async_and_stream_synchronize &
          ("elpa_cholesky: tmatc_dev -> tmatc", tmatc_dev, 0_c_intptr_t, tmatc(1:matrixCols,1:nblk), &
            1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
      successGPU = gpu_memcpy(int(loc(tmatc),kind=c_intptr_t), tmatc_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("elpa_cholesky: tmatc_dev to tmatc", successGPU)
#endif
    endif ! (useGPU  .and. .not. useCCL)
#endif /* !defined(WITH_MPI) */


    if (useGPU .and. .not. useCCL) then
      num = matrixRows*nblk
#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
      call gpu_memcpy_async_and_stream_synchronize &
          ("elpa_cholesky: tmatr_dev -> tmatr", tmatr_dev, 0_c_intptr_t, tmatr(1:matrixRows,1:nblk), &
            1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
      successGPU = gpu_memcpy(int(loc(tmatr),kind=c_intptr_t), tmatr_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("elpa_cholesky: tmatr_dev to tmatr", successGPU)
#endif
    endif ! (useGPU .and. .not. useCCL)


    if (useCCL) then
#ifdef WITH_NVTX
      call nvtxRangePush("elpa_gpu_ccl_transpose_vectors tmatc_dev->tmatr_dev")
#endif
#if defined(USE_CCL_CHOLESKY)
      call elpa_gpu_ccl_transpose_vectors_&
          &MATH_DATATYPE&
          &_&
          &PRECISION &
                (obj, tmatc_dev, ubound(tmatc,dim=1), ccl_comm_cols, mpi_comm_cols, &
                      tmatr_dev, ubound(tmatr,dim=1), ccl_comm_rows, mpi_comm_rows, &
                n, na, nblk, nblk, nrThreads, .false., my_pcol, my_prow, np_cols, np_rows, &
                aux_transpose_dev, isSkewsymmetric, isSquareGridGPU, wantDebug, my_stream, success)
#endif /* USE_CCL_CHOLESKY */

#ifdef WITH_NVTX
      call nvtxRangePop() !  elpa_gpu_ccl_transpose_vectors tmatc_dev->tmatr_dev
#endif   
    else ! useCCL
#ifdef WITH_NVTX
      call nvtxRangePush("elpa_transpose_vectors")
#endif
      call elpa_transpose_vectors_&
      &MATH_DATATYPE&
      &_&
      &PRECISION &
      (obj, tmatc, ubound(tmatc,dim=1), mpi_comm_cols, &
      tmatr, ubound(tmatr,dim=1), mpi_comm_rows, &
      n, na, nblk, nblk, nrThreads, .false., success)
#ifdef WITH_NVTX
      call nvtxRangePop() ! elpa_transpose_vectors
#endif
    endif ! useCCL

    if (useGPU .and. .not. useCCL) then
      num = matrixRows*nblk
#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
      call gpu_memcpy_async_and_stream_synchronize &
          ("elpa_cholesky: tmatr -> tmatr_dev", tmatr_dev, 0_c_intptr_t, tmatr(1:matrixRows,1:nblk), &
            1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
      successGPU = gpu_memcpy(tmatr_dev, int(loc(tmatr),kind=c_intptr_t), num*size_of_datatype, gpuMemcpyHostToDevice)
      check_memcpy_gpu("elpa_cholesky: tmat to tmatr_dev", successGPU)
#endif
    endif ! (useGPU .and. .not. useCCL)

    if (useGPU) then
      do i=0,(na-1)/tile_size
        lcs = max(l_colx,i*l_cols_tile+1)
        lce = min(matrixCols,(i+1)*l_cols_tile)
        lrs = l_rowx
        lre = min(matrixRows,(i+1)*l_rows_tile)
        if (lce  < lcs .or. lre < lrs) cycle
        call obj%timer%start("gpublas")
        gpublasHandle = obj%gpu_setup%gpublasHandleArray(0)
        tmatr_off = (lrs-1 + (1-1)*matrixRows) * size_of_datatype
        tmatc_off = (lcs-1 + (1-1)*matrixCols) * size_of_datatype
        a_off = (lrs-1 + (lcs-1)*matrixRows) * size_of_datatype
#ifdef WITH_NVTX
        call nvtxRangePush("gpublas gemm")
#endif
        call gpublas_PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, lre-lrs+1, lce-lcs+1, nblk, &
                            -ONE, tmatr_dev+tmatr_off, matrixRows, tmatc_dev+tmatc_off, matrixCols, ONE, &
                            a_dev+a_off, matrixRows, gpublasHandle)
        if (wantDebug .and. gpu_vendor() /= SYCL_GPU) successGPU = gpu_DeviceSynchronize()
#ifdef WITH_NVTX
        call nvtxRangePop() ! gpublas gemm
#endif
        call obj%timer%stop("gpublas")
      enddo
    else !useGPU
      do i=0,(na-1)/tile_size
        lcs = max(l_colx,i*l_cols_tile+1)
        lce = min(matrixCols,(i+1)*l_cols_tile)
        lrs = l_rowx
        lre = min(matrixRows,(i+1)*l_rows_tile)
        if (lce<lcs .or. lre<lrs) cycle
        call obj%timer%start("blas")
#ifndef DEVICE_POINTER
        call PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, int(lre-lrs+1,kind=BLAS_KIND), int(lce-lcs+1,kind=BLAS_KIND), &
                            int(nblk,kind=BLAS_KIND), -ONE,  &
                            tmatr(lrs,1), int(ubound(tmatr,dim=1),kind=BLAS_KIND), tmatc(lcs,1), &
                            int(ubound(tmatc,dim=1),kind=BLAS_KIND), &
                            ONE, a(lrs,lcs), int(matrixRows,kind=BLAS_KIND))
#endif
        call obj%timer%stop("blas")
      enddo
    endif ! useGPU

#ifdef WITH_NVTX
    call nvtxRangePop() ! do n = 1, na, nblk
#endif

  enddo ! n = 1, na, nblk

  call obj%timer%stop("loop1")
  
  call obj%timer%start("deallocate")
  if (useGPU) then

#if defined(WITH_NVIDIA_CUSOLVER) || defined(WITH_AMD_ROCSOLVER)
    my_stream = obj%gpu_setup%my_stream
    call gpu_check_device_info(info_abs_accumulated_dev, my_stream)
    
    successGPU = gpu_free(info_dev)
    check_dealloc_gpu("elpa_cholesky: info_dev", successGPU)

    successGPU = gpu_free(info_abs_accumulated_dev)
    check_dealloc_gpu("elpa_cholesky: info_abs_accumulated_dev", successGPU)
#endif

#if defined(WITH_NVIDIA_CUSOLVER)
    successGPU = gpu_free(gpusolver_buffer_dev)
    check_dealloc_gpu("elpa_cholesky: gpusolver_buffer_dev", successGPU)

    if (workspaceInBytesOnHost > 0) then
      successGPU = gpu_free_host(gpusolver_buffer_host)
      check_host_dealloc_gpu("elpa_cholesky: gpusolver_buffer_host", successGPU)
    endif
#endif

    successGPU = gpu_free(tmp1_dev)
    check_dealloc_gpu("elpa_cholesky: tmp1_dev", successGPU)

    successGPU = gpu_free(tmp2_dev)
    check_dealloc_gpu("elpa_cholesky: tmp1_dev", successGPU)

    successGPU = gpu_free(tmatc_dev)
    check_dealloc_gpu("elpa_cholesky: tmatc_dev", successGPU)

    successGPU = gpu_free(tmatr_dev)
    check_dealloc_gpu("elpa_cholesky: tmatr_dev", successGPU)

#ifdef USE_CCL_CHOLESKY
    if (.not. isSquareGridGPU) then
      successGPU = gpu_free(aux_transpose_dev)
      check_dealloc_gpu("tridiag: aux_transpose_dev", successGPU)
    endif
#endif
  endif

#ifdef WITH_GPU_STREAMS
  if (useGPU) then
    successGPU = gpu_host_unregister(int(loc(tmp1),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_cholesky: tmp1", successGPU)

    successGPU = gpu_host_unregister(int(loc(tmp2),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_cholesky: tmp2", successGPU)

    successGPU = gpu_host_unregister(int(loc(tmatc),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_cholesky: tmatc", successGPU)

    successGPU = gpu_host_unregister(int(loc(tmatr),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_cholesky: tmatr", successGPU)
  endif
#endif

  deallocate(tmp1, tmp2, tmatr, tmatc, stat=istat, errmsg=errorMessage)
  check_deallocate("elpa_cholesky: tmp1, tmp2, tmatr, tmatc", istat, errorMessage)

  call obj%timer%stop("deallocate")


  call obj%timer%start("set_lower_triangle_to_0")
  ! Set the lower triangle to 0, it contains garbage (form the above matrix multiplications)
  if (useGPU) then
    call gpu_set_a_lower_to_zero(PRECISION_CHAR, a_dev, na, matrixRows, my_pcol, np_cols, my_prow, np_rows, nblk, debug, my_stream)
  else ! useGPU
    do i=1,na
      if (my_pcol==pcol(i, nblk, np_cols)) then
        ! column i is on local processor
        l_col1 = local_index(i  , my_pcol, np_cols, nblk, +1) ! local column number
        l_row1 = local_index(i+1, my_prow, np_rows, nblk, +1) ! first row below diagonal
#ifndef DEVICE_POINTER
        a(l_row1:matrixRows,l_col1) = 0
#endif
      endif ! (my_pcol==pcol(i, nblk, np_cols))
    enddo
  endif ! useGPU
  call obj%timer%stop("set_lower_triangle_to_0")

  call obj%timer%start("memcpy")
#ifndef DEVICE_POINTER
  if (useGPU) then
    num = matrixRows*matrixCols
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    call gpu_memcpy_async_and_stream_synchronize &
        ("elpa_cholesky: a_dev -> a", a_dev, 0_c_intptr_t, a(1:obj%local_nrows,1:obj%local_ncols), &
          1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
    successGPU = gpu_memcpy(int(loc(a(1,1)),kind=c_intptr_t), a_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
    check_memcpy_gpu("elpa_cholesky: memcpy 2 a-> a_dev", successGPU)
#endif
  endif
#else /* DEVICE_POINTER */
#endif /* DEVICE_POINTER */
    call obj%timer%stop("memcpy")

  call obj%timer%start("cleanup")
#ifndef DEVICE_POINTER
  if (useGPU) then
    successGPU = gpu_free(a_dev)
    check_dealloc_gpu("elpa_cholesky: a_dev", successGPU)

#ifdef WITH_GPU_STREAMS
    successGPU = gpu_host_unregister(int(loc(a),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_cholesky: a", successGPU)
#endif

  endif
#else /* DEVICE_POINTER */
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_host_unregister(int(loc(a_tmp),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_cholesky: a_tmp", successGPU)
#endif

    deallocate(a_tmp, stat=istat, errmsg=errorMessage)
    check_deallocate("elpa_cholesky: a_tmp", istat, errorMessage)
#endif /* DEVICE_POINTER */

  ! restore original OpenMP settings
#ifdef WITH_OPENMP_TRADITIONAL
  ! store the number of OpenMP threads used in the calling function
  ! restore this at the end of ELPA 2
  call omp_set_num_threads(omp_threads_caller)
#endif
  call obj%timer%stop("cleanup")
  call obj%timer%stop("elpa_cholesky_&
  &MATH_DATATYPE&
  &_&
  &PRECISION&
  &"//gpuString)

#ifdef WITH_NVTX
    call nvtxRangePop() ! cholesky
#endif
