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
  type(c_ptr)                                :: a
  MATH_DATATYPE(kind=rck), allocatable       :: a_tmp(:,:)
#endif /* DEVICE_POINTER */
  integer(kind=ik)                           :: my_prow, my_pcol, np_rows, np_cols, myid
  integer(kind=MPI_KIND)                     :: mpierr, my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI, myidMPI
  integer(kind=ik)                           :: l_cols, l_rows, l_col1, l_row1, l_colx, l_rowx
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
  integer(kind=c_int)                        :: gpu, numGPU
  integer(kind=c_intptr_t)                   :: num
  integer(kind=c_intptr_t)                   :: tmp1_dev, tmatc_dev, tmatr_dev, a_dev, tmp2_dev
  type(c_ptr)                                :: tmp1_mpi_dev
  MATH_DATATYPE(kind=rck), pointer           :: tmp1_mpi_fortran_ptr(:,:)
  integer(kind=c_intptr_t)                   :: a_off, tmatc_off, tmatr_off
  type(c_ptr)                                :: tmatc_mpi_dev
  MATH_DATATYPE(kind=rck), pointer           :: tmatc_mpi_fortran_ptr(:,:)
  integer(kind=c_int)                        :: gpu_cholesky

  integer(kind=c_intptr_t), parameter        :: size_of_datatype = size_of_&
                                                            &PRECISION&
                                                            &_&
                                                            &MATH_DATATYPE

  success = .true.

  gpu_cholesky = 0
  ! GPU settings
  if (gpu_vendor() == NVIDIA_GPU) then
    call obj%get("gpu",gpu,error)
    if (error .ne. ELPA_OK) then
      write(error_unit,*) "ELPA_CHOLESKY: Problem getting option for GPU. Aborting..."
      success = .false.
      return
    endif
    if (gpu .eq. 1) then
      write(error_unit,*) "You still use the deprecated option 'gpu', consider switching to 'nvidia-gpu'. Will set the new &
              & keyword 'nvidia-gpu'"
      call obj%set("nvidia-gpu",gpu,error)
      if (error .ne. ELPA_OK) then
        write(error_unit,*) "ELPA_CHOLESKY: Problem setting option for NVIDIA GPU. Aborting..."
        success = .false.
        return
      endif
    endif

    call obj%get("nvidia-gpu",gpu,error)
    if (error .ne. ELPA_OK) then
      write(error_unit,*) "ELPA_CHOLESKY: Problem getting option for NVIDIA GPU. Aborting..."
      success = .false.
      return
    endif

  else if (gpu_vendor() == AMD_GPU) then
    call obj%get("amd-gpu",gpu,error)
    if (error .ne. ELPA_OK) then
      write(error_unit,*) "ELPA_CHOLESKY: Problem getting option for AMD GPU. Aborting..."
      success = .false.
      return
    endif
  else
    gpu = 0
  endif

  call obj%get("gpu_cholesky",gpu_cholesky, error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "ELPA_CHOLESKY: Problem getting option for gpu_cholesky. Aborting..."
    success = .false.
    return
  endif

  if (gpu_cholesky .eq. 1) then
    useGPU = (gpu == 1)
  else
    useGPU = .false.
  endif

  if (.not.(useGPU)) then
#ifdef DEVICE_POINTER
    write(error_unit,*) "You used the interface for device pointers for elpa_cholesky but did not specify GPU usage!. Aborting..."
    success = .false.
    return
#endif
  endif

  if(useGPU) then
    gpuString = "_gpu"
  else
    gpuString = ""
  endif

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
    call obj%get("omp_threads",nrThreads,error)
    call omp_set_num_threads(nrThreads)
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
  nblk       = obj%nblk
  matrixCols = obj%local_ncols

  call obj%get("mpi_comm_parent", mpi_comm_all, error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "ELPA_CHOLESKY: Error getting option for mpi_comm_all. Aborting..."
    success = .false.
    return
  endif
  call obj%get("mpi_comm_rows",mpi_comm_rows,error )
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "ELPA_CHOLESKY: Problem getting option for mpi_comm_rows. Aborting..."
    success = .false.
    return
  endif
  call obj%get("mpi_comm_cols",mpi_comm_cols,error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "ELPA_CHOLESKY: Problem getting option for mpi_comm_cols. Aborting..."
    success = .false.
    return
  endif

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

  call obj%timer%start("mpi_communication")
  call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND), my_prowMPI, mpierr)
  call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)
  call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND), my_pcolMPI, mpierr)
  call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND), np_colsMPI, mpierr)
  call mpi_comm_rank(int(mpi_comm_all,kind=MPI_KIND), myidMPI, mpierr)


  my_prow = int(my_prowMPI, kind=c_int)
  np_rows = int(np_rowsMPI, kind=c_int)
  my_pcol = int(my_pcolMPI, kind=c_int)
  np_cols = int(np_colsMPI, kind=c_int)
  myid    = int(myidMPI,kind=c_int)
  call obj%timer%stop("mpi_communication")
  success = .true.

  ! Matrix is split into tiles; work is done only for tiles on the diagonal or above

  tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size
  tile_size = ((128*max(np_rows,np_cols)-1)/tile_size+1)*tile_size ! make local tiles at least 128 wide

  l_rows_tile = tile_size/np_rows ! local rows of a tile
  l_cols_tile = tile_size/np_cols ! local cols of a tile

  l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a
  l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local cols of a

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


  if (useGPU) then
    successGPU = gpu_malloc(tmp1_dev, nblk*nblk*size_of_datatype)
    check_alloc_gpu("elpa_cholesky: tmp1_dev", successGPU)

#ifdef WITH_GPU_STREAMS
    successGPU = gpu_memset_async(tmp1_dev, 0, nblk*nblk*size_of_datatype, my_stream)
    check_memcpy_gpu("elpa_cholesky: memset tmp1_dev", successGPU)

    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_cholesky: memset", successGPU)
#else
    successGPU = gpu_memset(tmp1_dev, 0, nblk*nblk*size_of_datatype)
    check_memcpy_gpu("elpa_cholesky: memset tmp1_dev", successGPU)
#endif

    successGPU = gpu_malloc(tmp2_dev, nblk*nblk*size_of_datatype)
    check_alloc_gpu("elpa_cholesky: tmp2_dev", successGPU)

#ifdef WITH_GPU_STREAMS
    successGPU = gpu_memset_async(tmp2_dev, 0, nblk*nblk*size_of_datatype, my_stream)
    check_memcpy_gpu("elpa_cholesky: memset tmp2_dev", successGPU)

    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_cholesky: memset", successGPU)
#else
    successGPU = gpu_memset(tmp2_dev, 0, nblk*nblk*size_of_datatype)
    check_memcpy_gpu("elpa_cholesky: memset tmp2_dev", successGPU)
#endif

    successGPU = gpu_malloc(tmatc_dev, l_cols*nblk*size_of_datatype)
    check_alloc_gpu("elpa_cholesky: tmatc_dev", successGPU)

#ifdef WITH_GPU_STREAMS
    successGPU = gpu_memset_async(tmatc_dev, 0, l_cols*nblk*size_of_datatype, my_stream)
    check_memcpy_gpu("elpa_cholesky: memset tmatc_dev", successGPU)

    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_cholesky: memset", successGPU)
#else
    successGPU = gpu_memset(tmatc_dev, 0, l_cols*nblk*size_of_datatype)
    check_memcpy_gpu("elpa_cholesky: memset tmatc_dev", successGPU)
#endif

    successGPU = gpu_malloc(tmatr_dev, l_rows*nblk*size_of_datatype)
    check_alloc_gpu("elpa_cholesky: tmatr_dev", successGPU)

#ifdef WITH_GPU_STREAMS
    successGPU = gpu_memset_async(tmatr_dev, 0, l_rows*nblk*size_of_datatype, my_stream)
    check_memcpy_gpu("elpa_cholesky: memset tmatr_dev", successGPU)

    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_cholesky: memset", successGPU)
#else
    successGPU = gpu_memset(tmatr_dev, 0, l_rows*nblk*size_of_datatype)
    check_memcpy_gpu("elpa_cholesky: memset tmatr_dev", successGPU)
#endif

#ifndef DEVICE_POINTER
    successGPU = gpu_malloc(a_dev, matrixRows*matrixCols*size_of_datatype)
    check_alloc_gpu("elpa_cholesky: a_dev", successGPU)
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_host_register(int(loc(a),kind=c_intptr_t), &
                    matrixRows*matrixCols * size_of_datatype,&
                    gpuHostRegisterDefault)
    check_host_register_gpu("elpa_cholesky: a_tmp", successGPU)
#endif
#else /* DEVICE_POINTER */
    a_dev = transfer(a, a_dev)
    allocate(a_tmp(obj%local_nrows,obj%local_ncols), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_cholesky: a_tmp", istat, errorMessage)
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_host_register(int(loc(a_tmp),kind=c_intptr_t), &
                    matrixRows*matrixCols * size_of_datatype,&
                    gpuHostRegisterDefault)
    check_host_register_gpu("elpa_cholesky: a_tmp", successGPU)
#endif
#endif /* DEVICE_POINTER */
  endif ! useGPU

  allocate(tmp1(nblk*nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_cholesky: tmp1", istat, errorMessage)

  allocate(tmp2(nblk,nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_cholesky: tmp2", istat, errorMessage)

#ifdef WITH_GPU_STREAMS
  if (useGPU) then
    successGPU = gpu_host_register(int(loc(tmp1),kind=c_intptr_t), &
                    nblk*nblk * size_of_datatype,&
                    gpuHostRegisterDefault)
    check_host_register_gpu("elpa_cholesky: tmp1", successGPU)
  endif
#endif


  tmp1 = 0
  tmp2 = 0

  allocate(tmatr(l_rows,nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_cholesky: tmatr", istat, errorMessage)

  allocate(tmatc(l_cols,nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_cholesky: tmatc", istat, errorMessage)

#ifdef WITH_GPU_STREAMS
  if (useGPU) then
    successGPU = gpu_host_register(int(loc(tmatr),kind=c_intptr_t), &
                      l_rows*nblk * size_of_datatype,&
                      gpuHostRegisterDefault)
    check_host_register_gpu("elpa_cholesky: tmatr", successGPU)

    successGPU = gpu_host_register(int(loc(tmatc),kind=c_intptr_t), &
                      l_cols*nblk * size_of_datatype,&
                      gpuHostRegisterDefault)
    check_host_register_gpu("elpa_cholesky: tmatc", successGPU)
  endif
#endif

  tmatr = 0
  tmatc = 0

#ifndef DEVICE_POINTER
  if (useGPU) then
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_cholesky 1: memcpy a-> a_dev", successGPU)

    successGPU = gpu_memcpy_async(a_dev, int(loc(a(1,1)),kind=c_intptr_t), &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyHostToDevice, my_stream)
    check_memcpy_gpu("elpa_cholesky 1: memcpy a-> a_dev", successGPU)

    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_cholesky 1: memcpy a-> a_dev", successGPU)
    ! synchronize threadsPerStream; maybe not neccessary
    successGPU = gpu_stream_synchronize()
    check_stream_synchronize_gpu("elpa_cholesky 1: memcpy a-> a_dev", successGPU)
#else
    successGPU = gpu_memcpy(a_dev, int(loc(a(1,1)),kind=c_intptr_t), &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyHostToDevice)
    check_memcpy_gpu("elpa_cholesky 1: memcpy a-> a_dev", successGPU)
#endif
  endif
#endif

  do n = 1, na, nblk
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
#ifdef WITH_NVIDIA_CUSOLVER
          call obj%timer%start("gpusolver")

          a_off = (l_row1-1 + (l_col1-1)*matrixRows) * size_of_datatype
          call gpusolver_PRECISION_POTRF('U', na-n+1, a_dev+a_off, matrixRows, info)
          if (info .ne. 0) then
            write(error_unit,*) "elpa_cholesky: error in gpusolver_POTRF 1"
            success = .false.
            return
          endif
          call obj%timer%stop("gpusolver")
#else /* WITH_NVIDIA_CUSOLVER */

#ifndef DEVICE_POINTER
          call obj%timer%start("blas")
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_dev-> a", successGPU)

          successGPU = gpu_memcpy_async(int(loc(a(1,1)),kind=c_intptr_t), a_dev,  &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_dev-> a", successGPU)
          ! synchronize threadsPerStream; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_dev-> a", successGPU)
#else
          successGPU = gpu_memcpy(int(loc(a(1,1)),kind=c_intptr_t), a_dev,  &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a", successGPU)
#endif

          call PRECISION_POTRF('U', int(na-n+1,kind=BLAS_KIND), a(l_row1,l_col1), &
                             int(matrixRows,kind=BLAS_KIND), infoBLAS )
          info = int(infoBLAS,kind=ik)
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a -> a_dev", successGPU)

          successGPU = gpu_memcpy_async(a_dev, int(loc(a(1,1)),kind=c_intptr_t), &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("elpa_cholesky: memcpy a -> a_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a -> a_dev", successGPU)
          ! synchronize threadsPerStream; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a -> a_dev", successGPU)
#else
          successGPU = gpu_memcpy(a_dev, int(loc(a(1,1)),kind=c_intptr_t), &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a", successGPU)
#endif
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
            success = .false.
            return
          endif ! info
#else /* DEVICE_POINTER */
          call obj%timer%start("blas")
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)

          successGPU = gpu_memcpy_async(int(loc(a_tmp(1,1)),kind=c_intptr_t), a_dev,  &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)
          ! synchronize threadsPerStream; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)
#else
          successGPU = gpu_memcpy(int(loc(a_tmp(1,1)),kind=c_intptr_t), a_dev,  &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)
#endif

          call PRECISION_POTRF('U', int(na-n+1,kind=BLAS_KIND), a_tmp(l_row1,l_col1), &
                             int(matrixRows,kind=BLAS_KIND), infoBLAS )
          info = int(infoBLAS,kind=ik)
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)

          successGPU = gpu_memcpy_async(a_dev, int(loc(a_tmp(1,1)),kind=c_intptr_t), &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)
          ! synchronize threadsPerStream; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)
#else
          successGPU = gpu_memcpy(a_dev, int(loc(a_tmp(1,1)),kind=c_intptr_t), &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)
#endif
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
            success = .false.
            return
          endif ! info
#endif /* DEVICE_POINTER */
#endif /* WITH_NVIDIA_CUSOLVER */
        endif ! (my_prow==prow(n, nblk, np_rows) .and. my_pcol==pcol(n, nblk, np_cols))
      else ! useGPU
        if (my_prow==prow(n, nblk, np_rows) .and. my_pcol==pcol(n, nblk, np_cols)) then
          call obj%timer%start("blas")

#ifndef DEVICE_POINTER
          call PRECISION_POTRF('U', int(na-n+1,kind=BLAS_KIND), a(l_row1,l_col1), &
                             int(matrixRows,kind=BLAS_KIND), infoBLAS )
#else
          call PRECISION_POTRF('U', int(na-n+1,kind=BLAS_KIND), a_tmp(l_row1,l_col1), &
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
            success = .false.
            return
          endif

        endif
      endif ! useGPU
      exit ! Loop
    endif ! (n+nblk > na) 

    if (my_prow==prow(n, nblk, np_rows)) then

      if (my_pcol==pcol(n, nblk, np_cols)) then

        if (useGPU) then
#ifdef WITH_NVIDIA_CUSOLVER
          call obj%timer%start("gpusolver")

          a_off = (l_row1-1 + (l_col1-1)*matrixRows) * size_of_datatype
          call gpusolver_PRECISION_POTRF('U', nblk, a_dev+a_off, matrixRows, info)
          if (info .ne. 0) then
            write(error_unit,*) "elpa_cholesky: error in gpusolver_POTRF 2"
            success = .false.
            return
          endif
          call obj%timer%stop("gpusolver")
#else /* WITH_NVIDIA_CUSOLVER */
#ifndef DEVICE_POINTER
          call obj%timer%start("blas")
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_dev-> a", successGPU)

          successGPU = gpu_memcpy_async(int(loc(a(1,1)),kind=c_intptr_t), a_dev,  &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_dev-> a", successGPU)
          ! synchronize threadsPerStream; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_dev-> a", successGPU)
#else
          successGPU = gpu_memcpy(int(loc(a(1,1)),kind=c_intptr_t), a_dev,  &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a", successGPU)
#endif

          call PRECISION_POTRF('U', int(nblk,kind=BLAS_KIND), a(l_row1,l_col1), &
                               int(matrixRows,kind=BLAS_KIND) , infoBLAS )
          info = int(infoBLAS,kind=ik)
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a -> a_dev", successGPU)

          successGPU = gpu_memcpy_async(a_dev, int(loc(a(1,1)),kind=c_intptr_t), &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("elpa_cholesky: memcpy a -> a_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a -> a_dev", successGPU)
          ! synchronize threadsPerStream; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a -> a_dev", successGPU)
#else
          successGPU = gpu_memcpy(a_dev, int(loc(a(1,1)),kind=c_intptr_t), &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a", successGPU)
#endif
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
            success = .false.
            return
          endif ! info
#else /* DEVICE_POINTER */
          call obj%timer%start("blas")
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)

          successGPU = gpu_memcpy_async(int(loc(a_tmp(1,1)),kind=c_intptr_t), a_dev,  &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)
          ! synchronize threadsPerStream; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)
#else
          successGPU = gpu_memcpy(int(loc(a_tmp(1,1)),kind=c_intptr_t), a_dev,  &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_cholesky: memcpy a_dev-> a_tmp", successGPU)
#endif

          call PRECISION_POTRF('U', int(nblk,kind=BLAS_KIND), a_tmp(l_row1,l_col1), &
                               int(matrixRows,kind=BLAS_KIND) , infoBLAS )
          info = int(infoBLAS,kind=ik)
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_tmp-> a_dev", successGPU)

          successGPU = gpu_memcpy_async(a_dev, int(loc(a_tmp(1,1)),kind=c_intptr_t), &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("elpa_cholesky: memcpy a_tmp-> a_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_tmp-> a_dev", successGPU)
          ! synchronize threadsPerStream; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("elpa_cholesky: memcpy a_tmp-> a_dev", successGPU)
#else
          successGPU = gpu_memcpy(a_dev, int(loc(a_tmp(1,1)),kind=c_intptr_t), &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_cholesky: memcpy a_tmp-> a_dev", successGPU)
#endif
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
            success = .false.
            return
          endif ! info

#endif /* DEVICE_POINTER */
#endif /* WITH_NVIDIA_CUSOLVER */
        else ! useGPU
          ! The process owning the upper left remaining block does the
          ! Cholesky-Factorization of this block
          call obj%timer%start("blas")

#ifndef DEVICE_POINTER
          call PRECISION_POTRF('U', int(nblk,kind=BLAS_KIND), a(l_row1,l_col1), &
                               int(matrixRows,kind=BLAS_KIND) , infoBLAS )
#else
          call PRECISION_POTRF('U', int(nblk,kind=BLAS_KIND), a_tmp(l_row1,l_col1), &
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
            success = .false.
            return
          endif ! useGPU
        endif
 
        if (useGPU) then
          call gpu_copy_PRECISION_a_tmp1 (a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nblk, my_stream)
        else ! useGPU
          nc = 0
          do i=1,nblk
#ifndef DEVICE_POINTER
            tmp1(nc+1:nc+i) = a(l_row1:l_row1+i-1,l_col1+i-1)
#else
            tmp1(nc+1:nc+i) = a_tmp(l_row1:l_row1+i-1,l_col1+i-1)
#endif
            nc = nc+i
          enddo
        endif ! useGPU
      endif ! (my_pcol==pcol(n, nblk, np_cols))

#ifdef WITH_MPI
!#ifndef WITH_CUDA_AWARE_MPI
      if (useGPU) then
        num = nblk*nblk*size_of_datatype
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("elpa_cholesky: tmp1_dev to tmp1", successGPU)

        successGPU = gpu_memcpy_async(int(loc(tmp1),kind=c_intptr_t), tmp1_dev, num, &
                              gpuMemcpyDeviceToHost, my_stream)
        check_memcpy_gpu("elpa_cholesky: tmp1_dev to tmp1", successGPU)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("elpa_cholesky: tmp1_dev to tmp1", successGPU)
        ! synchronize threadsPerStream; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu("elpa_cholesky: tmp1_dev to tmp1", successGPU)
#else
        successGPU = gpu_memcpy(int(loc(tmp1),kind=c_intptr_t), tmp1_dev, num, &
                              gpuMemcpyDeviceToHost)
        check_memcpy_gpu("elpa_cholesky: tmp1_dev to tmp1", successGPU)
#endif
      endif
!#endif

!#ifndef WITH_CUDA_AWARE_MPI
      call obj%timer%start("mpi_communication")

      call MPI_Bcast(tmp1, int(nblk*(nblk+1)/2,kind=MPI_KIND),      &
#if REALCASE == 1
                    MPI_REAL_PRECISION,         &
#endif
#if COMPLEXCASE == 1
                    MPI_COMPLEX_PRECISION,      &
#endif
                    int(pcol(n, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)

      call obj%timer%stop("mpi_communication")
!#else
!      tmp1_mpi_dev = transfer(tmp1_dev, tmp1_mpi_dev)
!      ! and associate a fortran pointer
!      call c_f_pointer(tmp1_mpi_dev, tmp1_mpi_fortran_ptr, [nblk,nblk])
!      if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
!      successGPU = gpu_devicesynchronize()
!      check_memcpy_gpu("cholesky: device_synchronize", successGPU)
!      if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
!      call obj%timer%start("mpi_cuda_communication")
!
!      call MPI_Bcast(tmp1_mpi_fortran_ptr, int(nblk*(nblk+1)/2,kind=MPI_KIND),      &
!#if REALCASE == 1
!                    MPI_REAL_PRECISION,         &
!#endif
!#if COMPLEXCASE == 1
!                    MPI_COMPLEX_PRECISION,      &
!#endif
!                    int(pcol(n, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
!
!      call obj%timer%stop("mpi_cuda_communication")
!#endif

!#ifndef WITH_CUDA_AWARE_MPI
      if (useGPU) then
        num = nblk*nblk*size_of_datatype
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("elpa_cholesky: tmp1 to tmp1_dev", successGPU)

        successGPU = gpu_memcpy_async(tmp1_dev, int(loc(tmp1),kind=c_intptr_t), num, &
                              gpuMemcpyHostToDevice, my_stream)
        check_memcpy_gpu("elpa_cholesky: tmp1 to tmp1_dev", successGPU)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("elpa_cholesky: tmp1 to tmp1_dev", successGPU)
        ! synchronize threadsPerStream; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu("elpa_cholesky: tmp1 to tmp1_dev", successGPU)
#else
        successGPU = gpu_memcpy(tmp1_dev, int(loc(tmp1),kind=c_intptr_t), num, &
                              gpuMemcpyHostToDevice)
        check_memcpy_gpu("elpa_cholesky: tmp1 to tmp1_dev", successGPU)
#endif
      endif
!#endif

#endif /* WITH_MPI */

      if (useGPU) then
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
        if (l_cols-l_colx+1 > 0) then
          a_off = (l_row1-1 + (l_colx-1)*matrixRows) * size_of_datatype
          call gpublas_PRECISION_TRSM('L', 'U', BLAS_TRANS_OR_CONJ, 'N', nblk, l_cols-l_colx+1, ONE, &
                            tmp2_dev, nblk, a_dev+a_off, matrixRows)
        endif
        call obj%timer%stop("gpublas")

      else ! useGPU

        call obj%timer%start("blas")
#ifndef DEVICE_POINTER
        if (l_cols-l_colx+1>0) &
        call PRECISION_TRSM('L', 'U', BLAS_TRANS_OR_CONJ, 'N', int(nblk,kind=BLAS_KIND),  &
                            int(l_cols-l_colx+1,kind=BLAS_KIND), ONE, tmp2, &
                            int(ubound(tmp2,dim=1),kind=BLAS_KIND), a(l_row1,l_colx), int(matrixRows,kind=BLAS_KIND) )
#else
        if (l_cols-l_colx+1>0) &
        call PRECISION_TRSM('L', 'U', BLAS_TRANS_OR_CONJ, 'N', int(nblk,kind=BLAS_KIND),  &
                            int(l_cols-l_colx+1,kind=BLAS_KIND), ONE, tmp2, &
                            int(ubound(tmp2,dim=1),kind=BLAS_KIND), a_tmp(l_row1,l_colx), int(matrixRows,kind=BLAS_KIND) )
#endif
        call obj%timer%stop("blas")
      endif ! useGPU
    endif ! (my_prow==prow(n, nblk, np_rows))


    if (useGPU) then
      if (my_prow==prow(n, nblk, np_rows)) then
        ! if l_cols-l_colx+1 == 0 kernel launch with 0 blocks => raises error
        if (l_cols-l_colx+1>0) &
           call gpu_copy_PRECISION_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, l_cols, l_colx, l_row1, my_stream)
      endif
    else ! useGPU
      do i=1,nblk
#ifndef DEVICE_POINTER
#if REALCASE == 1
        if (my_prow==prow(n, nblk, np_rows)) tmatc(l_colx:l_cols,i) = a(l_row1+i-1,l_colx:l_cols)
#endif
#if COMPLEXCASE == 1
        if (my_prow==prow(n, nblk, np_rows)) tmatc(l_colx:l_cols,i) = conjg(a(l_row1+i-1,l_colx:l_cols))
#endif
#else
#if REALCASE == 1
        if (my_prow==prow(n, nblk, np_rows)) tmatc(l_colx:l_cols,i) = a_tmp(l_row1+i-1,l_colx:l_cols)
#endif
#if COMPLEXCASE == 1
        if (my_prow==prow(n, nblk, np_rows)) tmatc(l_colx:l_cols,i) = conjg(a_tmp(l_row1+i-1,l_colx:l_cols))
#endif
#endif
      enddo
    endif ! useGPU

#ifdef WITH_MPI
!#ifndef WITH_CUDA_AWARE_MPI
    if (useGPU) then
      if (l_cols-l_colx+1 > 0) then
        num = l_cols*nblk*size_of_datatype
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("elpa_cholesky: tmatc_dev to tmatc", successGPU)

        successGPU = gpu_memcpy_async(int(loc(tmatc),kind=c_intptr_t), tmatc_dev, num, &
                              gpuMemcpyDeviceToHost, my_stream)
        check_memcpy_gpu("elpa_cholesky: tmatc_dev to tmatc", successGPU)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("elpa_cholesky: tmatc_dev to tmatc", successGPU)
        ! synchronize threadsPerStream; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu("elpa_cholesky: tmatc_dev to tmatc", successGPU)
#else
        successGPU = gpu_memcpy(int(loc(tmatc),kind=c_intptr_t), tmatc_dev, num, &
                              gpuMemcpyDeviceToHost)
        check_memcpy_gpu("elpa_cholesky: tmatc_dev to tmatc", successGPU)
#endif
      endif
    endif
!#endif

#endif /* WITH_MPI */

#ifdef WITH_MPI
!#ifndef WITH_CUDA_AWARE_MPI
    do i=1,nblk
      call obj%timer%start("mpi_communication")
      if (l_cols-l_colx+1>0) &
      call MPI_Bcast(tmatc(l_colx,i), int(l_cols-l_colx+1,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                     int(prow(n, nblk, np_rows),kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)

      call obj%timer%stop("mpi_communication")
    enddo
!#else
!    tmatc_mpi_dev = transfer(tmatc_dev, tmatc_mpi_dev)
!    ! and associate a fortran pointer
!    call c_f_pointer(tmatc_mpi_dev, tmatc_mpi_fortran_ptr, [l_cols,nblk])
!    
!    if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
!    successGPU = gpu_devicesynchronize()
!    check_memcpy_gpu("cholesky: device_synchronize", successGPU)
!    if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
!
!    do i=1,nblk
!      call obj%timer%start("mpi_cuda_communication")
!      if (l_cols-l_colx+1>0) &
!      call MPI_Bcast(tmatc_mpi_fortran_ptr(l_colx,i), int(l_cols-l_colx+1,kind=MPI_KIND), &
!                     MPI_MATH_DATATYPE_PRECISION, &
!                     int(prow(n, nblk, np_rows),kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
!
!      call obj%timer%stop("mpi_cuda_communication")
!    enddo
!#endif
#endif /* WITH_MPI */

#ifdef WITH_MPI
!#ifndef WITH_CUDA_AWARE_MPI
    if (useGPU) then
      !if (l_cols-l_colx+1 > 0) then
        num = l_cols*nblk*size_of_datatype
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("elpa_cholesky: tmatc to tmatc_dev", successGPU)

        successGPU = gpu_memcpy_async(tmatc_dev, int(loc(tmatc),kind=c_intptr_t), num, &
                              gpuMemcpyHostToDevice, my_stream)
        check_memcpy_gpu("elpa_cholesky: tmatc to tmatc_dev", successGPU)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("elpa_cholesky: tmatc to tmatc_dev", successGPU)
        ! synchronize threadsPerStream; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu("elpa_cholesky: tmatc to tmatc_dev", successGPU)
#else
        successGPU = gpu_memcpy(tmatc_dev, int(loc(tmatc),kind=c_intptr_t), num, &
                              gpuMemcpyHostToDevice)
        check_memcpy_gpu("elpa_cholesky: tmatc to tmatc_dev", successGPU)
#endif
      !endif
    endif
!#endif
#endif /* WITH_MPI */

    if (useGPU) then
      ! can optimize memcpy here with previous ones

      ! a gpu version of elpa_transpose_vectors is needed

!#if !defined(WITH_MPI) || (defined(WITH_MPI) && defined(WITH_CUDA_AWARE_MPI))
#if !defined(WITH_MPI)
      ! this memcopy is only needed if
      ! - not mpi case
      ! - or mpi and cuda_aware_mpi
      num = l_cols*nblk*size_of_datatype
#ifdef WITH_GPU_STREAMS
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("elpa_cholesky: tmatc_dev to tmatc", successGPU)

      successGPU = gpu_memcpy_async(int(loc(tmatc),kind=c_intptr_t), tmatc_dev, num, &
                              gpuMemcpyDeviceToHost, my_stream)
      check_memcpy_gpu("elpa_cholesky: tmatc_dev to tmatc", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("elpa_cholesky: tmatc_dev to tmatc", successGPU)
      ! synchronize threadsPerStream; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      check_stream_synchronize_gpu("elpa_cholesky: tmatc_dev to tmatc", successGPU)
#else
      successGPU = gpu_memcpy(int(loc(tmatc),kind=c_intptr_t), tmatc_dev, num, &
                              gpuMemcpyDeviceToHost)
      check_memcpy_gpu("elpa_cholesky: tmatc_dev to tmatc", successGPU)
#endif
#endif /* !defined(WITH_MPI) */

      num = l_rows*nblk*size_of_datatype
#ifdef WITH_GPU_STREAMS
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("elpa_cholesky: tmatr_dev to tmatr", successGPU)

      successGPU = gpu_memcpy_async(int(loc(tmatr),kind=c_intptr_t), tmatr_dev, num, &
                              gpuMemcpyDeviceToHost, my_stream)

      check_memcpy_gpu("elpa_cholesky: tmatr_dev to tmatr", successGPU)
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("elpa_cholesky: tmatr_dev to tmatr", successGPU)
      ! synchronize threadsPerStream; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      check_stream_synchronize_gpu("elpa_cholesky: tmatr_dev to tmatr", successGPU)
#else
      successGPU = gpu_memcpy(int(loc(tmatr),kind=c_intptr_t), tmatr_dev, num, &
                              gpuMemcpyDeviceToHost)
      check_memcpy_gpu("elpa_cholesky: tmatr_dev to tmatr", successGPU)
#endif
    endif

    call elpa_transpose_vectors_&
    &MATH_DATATYPE&
    &_&
    &PRECISION &
    (obj, tmatc, ubound(tmatc,dim=1), mpi_comm_cols, &
    tmatr, ubound(tmatr,dim=1), mpi_comm_rows, &
    n, na, nblk, nblk, nrThreads, .false., success)
    if (.not.(success)) then
      write(error_unit,*) "Error in elpa_transpose_vectors. Aborting..."
      return
    endif

    if (useGPU) then
      num = l_rows*nblk*size_of_datatype
#ifdef WITH_GPU_STREAMS
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("elpa_cholesky: tmat to tmatr_dev", successGPU)

      successGPU = gpu_memcpy_async(tmatr_dev, int(loc(tmatr),kind=c_intptr_t), num, &
                              gpuMemcpyHostToDevice, my_stream)
      check_memcpy_gpu("elpa_cholesky: tmat to tmatr_dev", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("elpa_cholesky: tmat to tmatr_dev", successGPU)
      ! synchronize threadsPerStream; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      check_stream_synchronize_gpu("elpa_cholesky: tmat to tmatr_dev", successGPU)
#else
      successGPU = gpu_memcpy(tmatr_dev, int(loc(tmatr),kind=c_intptr_t), num, &
                              gpuMemcpyHostToDevice)
      check_memcpy_gpu("elpa_cholesky: tmat to tmatr_dev", successGPU)
#endif
    endif


    if (useGPU) then
      do i=0,(na-1)/tile_size
        lcs = max(l_colx,i*l_cols_tile+1)
        lce = min(l_cols,(i+1)*l_cols_tile)
        lrs = l_rowx
        lre = min(l_rows,(i+1)*l_rows_tile)
        if (lce  < lcs .or. lre < lrs) cycle
        call obj%timer%start("gpublas")
        tmatr_off = (lrs-1 + (1-1)*l_rows) * size_of_datatype
        tmatc_off = (lcs-1 + (1-1)*l_cols) * size_of_datatype
        a_off = (lrs-1 + (lcs-1)*matrixRows) * size_of_datatype
        call gpublas_PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, lre-lrs+1, lce-lcs+1, nblk, &
                            -ONE, tmatr_dev+tmatr_off, l_rows, tmatc_dev+tmatc_off, l_cols, ONE, &
                            a_dev+a_off, matrixRows)
        call obj%timer%stop("gpublas")
      enddo
    else !useGPU
      do i=0,(na-1)/tile_size
        lcs = max(l_colx,i*l_cols_tile+1)
        lce = min(l_cols,(i+1)*l_cols_tile)
        lrs = l_rowx
        lre = min(l_rows,(i+1)*l_rows_tile)
        if (lce<lcs .or. lre<lrs) cycle
        call obj%timer%start("blas")
#ifndef DEVICE_POINTER
        call PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, int(lre-lrs+1,kind=BLAS_KIND), int(lce-lcs+1,kind=BLAS_KIND), &
                            int(nblk,kind=BLAS_KIND), -ONE,  &
                            tmatr(lrs,1), int(ubound(tmatr,dim=1),kind=BLAS_KIND), tmatc(lcs,1), &
                            int(ubound(tmatc,dim=1),kind=BLAS_KIND), &
                            ONE, a(lrs,lcs), int(matrixRows,kind=BLAS_KIND))
#else
        call PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, int(lre-lrs+1,kind=BLAS_KIND), int(lce-lcs+1,kind=BLAS_KIND), &
                            int(nblk,kind=BLAS_KIND), -ONE,  &
                            tmatr(lrs,1), int(ubound(tmatr,dim=1),kind=BLAS_KIND), tmatc(lcs,1), &
                            int(ubound(tmatc,dim=1),kind=BLAS_KIND), &
                            ONE, a_tmp(lrs,lcs), int(matrixRows,kind=BLAS_KIND))
#endif
        call obj%timer%stop("blas")
      enddo
    endif ! useGPU

  enddo ! n = 1, na, nblk

  if (useGPU) then
    successGPU = gpu_free(tmp1_dev)
    check_dealloc_gpu("elpa_cholesky: tmp1_dev", successGPU)

    successGPU = gpu_free(tmp2_dev)
    check_dealloc_gpu("elpa_cholesky: tmp1_dev", successGPU)

    successGPU = gpu_free(tmatc_dev)
    check_dealloc_gpu("elpa_cholesky: tmatc_dev", successGPU)

    successGPU = gpu_free(tmatr_dev)
    check_dealloc_gpu("elpa_cholesky: tmatr_dev", successGPU)
  endif

#ifdef WITH_GPU_STREAMS
  if (useGPU) then
    successGPU = gpu_host_unregister(int(loc(tmatc),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_cholesky: tmatc", successGPU)

    successGPU = gpu_host_unregister(int(loc(tmatr),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_cholesky: tmatr", successGPU)

    successGPU = gpu_host_unregister(int(loc(tmp1),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_cholesky: tmp1", successGPU)
  endif
#endif

  deallocate(tmp1, tmp2, tmatr, tmatc, stat=istat, errmsg=errorMessage)
  check_deallocate("elpa_cholesky: tmp1, tmp2, tmatr, tmatc", istat, errorMessage)




#ifndef DEVICE_POINTER
  if (useGPU) then
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_cholesky: memcpy 2 a-> a_dev", successGPU)

    successGPU = gpu_memcpy_async(int(loc(a(1,1)),kind=c_intptr_t), a_dev,  &
                     matrixRows*matrixCols* size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
    check_memcpy_gpu("elpa_cholesky: memcpy 2 a-> a_dev", successGPU)

    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_cholesky: memcpy 2 a-> a_dev", successGPU)
    ! synchronize threadsPerStream; maybe not neccessary
    successGPU = gpu_stream_synchronize()
    check_stream_synchronize_gpu("elpa_cholesky: memcpy 2 a-> a_dev", successGPU)
#else
    successGPU = gpu_memcpy(int(loc(a(1,1)),kind=c_intptr_t), a_dev,  &
                     matrixRows*matrixCols* size_of_datatype, gpuMemcpyDeviceToHost)
    check_memcpy_gpu("elpa_cholesky: memcpy 2 a-> a_dev", successGPU)
#endif
  endif
#else /* DEVICE_POINTER */
  if (useGPU) then
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_cholesky: memcpy 2 a -> a_dev", successGPU)

    successGPU = gpu_memcpy_async(int(loc(a_tmp(1,1)),kind=c_intptr_t), a_dev,  &
                     matrixRows*matrixCols* size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
    check_memcpy_gpu("elpa_cholesky: memcpy 2 a-> a_dev", successGPU)

    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_cholesky: memcpy 2 a -> a_dev", successGPU)
    ! synchronize threadsPerStream; maybe not neccessary
    successGPU = gpu_stream_synchronize()
    check_stream_synchronize_gpu("elpa_cholesky: memcpy 2 a -> a_dev", successGPU)
#else
    successGPU = gpu_memcpy(int(loc(a_tmp(1,1)),kind=c_intptr_t), a_dev,  &
                     matrixRows*matrixCols* size_of_datatype, gpuMemcpyDeviceToHost)
    check_memcpy_gpu("elpa_cholesky: memcpy 2 a-> a_dev", successGPU)
#endif
  endif
#endif /* DEVICE_POINTER */
  ! Set the lower triangle to 0, it contains garbage (form the above matrix multiplications)

  !if (useGPU) then
  !else ! useGPU
    do i=1,na
      if (my_pcol==pcol(i, nblk, np_cols)) then
        ! column i is on local processor
        l_col1 = local_index(i  , my_pcol, np_cols, nblk, +1) ! local column number
        l_row1 = local_index(i+1, my_prow, np_rows, nblk, +1) ! first row below diagonal
#ifndef DEVICE_POINTER
        a(l_row1:l_rows,l_col1) = 0
#else
        a_tmp(l_row1:l_rows,l_col1) = 0
#endif
      endif
    enddo
  !endif ! useGPU

#ifndef DEVICE_POINTER
  if (useGPU) then
    ! copy back
    !successGPU = gpu_memcpy(int(loc(a(1,1)),kind=c_intptr_t), a_dev,  &
    !                   matrixRows*matrixCols* size_of_datatype, gpuMemcpyDeviceToHost)
    !check_memcpy_gpu("elpa_cholesky: memcpy a-> d_dev", successGPU)

    successGPU = gpu_free(a_dev)
    check_dealloc_gpu("elpa_cholesky: a_dev", successGPU)

#ifdef WITH_GPU_STREAMS
    successGPU = gpu_host_unregister(int(loc(a),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_cholesky: a", successGPU)
#endif

  endif
#else /* DEVICE_POINTER */

#ifdef WITH_GPU_STREAMS
    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_cholesky: memcpy a_tmp-> a_dev", successGPU)

    successGPU = gpu_memcpy_async(a_dev, int(loc(a_tmp(1,1)),kind=c_intptr_t), &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyHostToDevice, my_stream)
    check_memcpy_gpu("elpa_cholesky: memcpy a_tmp-> a_dev", successGPU)

    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_cholesky: memcpy a_tmp-> a_dev", successGPU)
    ! synchronize threadsPerStream; maybe not neccessary
    successGPU = gpu_stream_synchronize()
    check_stream_synchronize_gpu("elpa_cholesky: memcpy a_tmp-> a_dev", successGPU)
#else
    successGPU = gpu_memcpy(a_dev, int(loc(a_tmp(1,1)),kind=c_intptr_t), &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyHostToDevice)
    check_memcpy_gpu("elpa_cholesky: memcpy a_tmp-> a_dev", successGPU)
#endif

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
  call obj%timer%stop("elpa_cholesky_&
  &MATH_DATATYPE&
  &_&
  &PRECISION&
  &"//gpuString)
