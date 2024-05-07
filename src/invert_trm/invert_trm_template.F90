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
!
!
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".

#undef INVERT_TRM_GPU_SOLVER
#if defined(WITH_NVIDIA_CUSOLVER) || defined(WITH_AMD_ROCSOLVER)
#define INVERT_TRM_GPU_SOLVER
#endif
#if defined(WITH_AMD_ROCSOLVER)
#ifndef WITH_AMD_HIPSOLVER_API
! at the moment {X}trtri not available in hipsolver, only in pure rocsolver
#define INVERT_TRM_GPU_SOLVER
#endif
#endif



#include "../general/sanity.F90"
#include "../general/error_checking.inc"
#include "config-f90.h"
#include "../general/precision_macros.h"

#undef USE_CCL_INVERT
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
#define USE_CCL_INVERT
#endif

  use precision
  use elpa1_compute
  use elpa_utilities
  use elpa_mpi
  use elpa_abstract_impl
  use elpa_gpu
  use mod_check_for_gpu
  use elpa_blas_interfaces
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
  use elpa_ccl_gpu
#endif
#ifdef WITH_GPU_STREAMS
  use elpa_gpu_util
#endif
#if defined(WITH_NVIDIA_GPU_VERSION) && defined(WITH_NVTX)
  use cuda_functions ! for NVTX labels
#endif
  use invert_trm_gpu
  use mod_query_gpu_usage

  implicit none
#include "../general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik)                           :: na, matrixRows, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
  integer(kind=ik)                           :: mpi_comm_all
#ifdef DEVICE_POINTER
  type(c_ptr)                                :: aDev
#if !defined(INVERT_TRM_GPU_SOLVER)
  MATH_DATATYPE(kind=rck), allocatable       :: a_tmp(:,:)
#endif
#else /* DEVICE_POINTER */
#ifdef USE_ASSUMED_SIZE
  MATH_DATATYPE(kind=rck)                    :: a(obj%local_nrows,*)
#else
  MATH_DATATYPE(kind=rck)                    :: a(obj%local_nrows,obj%local_ncols)
#endif
#endif /* DEVICE_POINTER */
  integer :: ii, jj

  integer(kind=ik)                           :: my_prow, my_pcol, np_rows, np_cols, myid
  integer(kind=MPI_KIND)                     :: mpierr, my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI, myidMPI
  integer(kind=ik)                           :: l_cols, l_rows, l_col1, l_row1, l_colx, l_rowx
  integer(kind=ik)                           :: n, nc, i, info, ns, nb
  integer(kind=BLAS_KIND)                    :: infoBLAS
  MATH_DATATYPE(kind=rck), allocatable       :: tmp1(:), tmp2(:,:), tmat1(:,:), tmat2(:,:)
  logical                                    :: wantDebug
  logical                                    :: success
  integer(kind=ik)                           :: istat, debug, error
  character(200)                             :: errorMessage
  character(20)                              :: gpuString
  logical                                    :: successGPU
  logical                                    :: useGPU
  integer(kind=c_int)                        :: numGPU
  integer(kind=c_intptr_t)                   :: tmat1_dev, tmat2_dev, a_dev, tmp1_dev, tmp2_dev
  type(c_ptr)                                :: tmp1_mpi_dev
  MATH_DATATYPE(kind=rck), pointer           :: tmp1_mpi_fortran_ptr(:)
  type(c_ptr)                                :: tmat1_mpi_dev, tmat2_mpi_dev
  MATH_DATATYPE(kind=rck), pointer           :: tmat1_mpi_fortran_ptr(:,:), tmat2_mpi_fortran_ptr(:,:)

  type(c_ptr)                                :: tmp2_mpi_dev, a_mpi_dev
  integer(kind=c_intptr_t)                   :: a_off, tmat2_off, tmp1_off, tmp2_off
  integer(kind=c_intptr_t)                   :: num
  integer(kind=c_intptr_t), parameter        :: size_of_datatype = size_of_&
                                                            &PRECISION&
                                                            &_&
                                                            &MATH_DATATYPE

  integer(kind=c_intptr_t)                   :: gpublasHandle, gpusolverHandle, my_stream
  integer(kind=c_int)                        :: gpu_invert_trm

  logical                                    :: useCCL
#if defined(USE_CCL_INVERT)
  integer(kind=c_intptr_t)                   :: ccl_comm_rows, ccl_comm_cols, offset
  integer(kind=c_int)                        :: cclDataType
  integer(kind=ik)                           :: k_datatype
#endif

#ifdef WITH_NVTX
  call nvtxRangePush("invert_trm")
#endif

  success = .true.
  useGPU = .false.
  useCCL = .false.

#if !defined(DEVICE_POINTER)

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
  if (.not.(query_gpu_usage(obj, "ELPA_INVERT_TRM", useGPU))) then
    print *,"ELPA_INVERT_TRM: Problem querrying settings for GPU Aborting..."
    stop 1
  endif
#endif

  ! check whether the above setting should be overriden
  if (obj%is_set("gpu_invert_trm") == 1) then
    call obj%get("gpu_invert_trm", gpu_invert_trm, error)
    if (error .ne. ELPA_OK) then
      print *,"Problem getting option for gpu_invert_trm. Aborting..."
      stop 1
    endif
    if (useGPU .and. gpu_invert_trm .eq. 0) then
      useGPU = .false.
    else if (.not.(useGPU) .and. gpu_invert_trm .eq. 1) then
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

  if (.not.(useGPU)) then
#ifdef DEVICE_POINTER
    print *,"You used the interface for device pointers for elpa_invert_trm but did not specify GPU usage!. Aborting..."
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
    if (check_for_gpu(obj, myid, numGPU, .TRUE.)) then
       ! set the neccessary parameters       
      call set_gpu_parameters()
    else
      print *,"ELPA_INVERT_TRM: GPUs are requested but not detected! Aborting..."
      success = .false.
      return
    endif
    call obj%timer%stop("check_for_gpu")
  else ! useGPU
  endif ! useGPU

#if defined(USE_CCL_INVERT)
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
#endif /* defined(USE_CCL_INVERT) */

  call obj%timer%start("elpa_invert_trm_&
  &MATH_DATATYPE&
  &_&
  &PRECISION&
  &"//gpuString)

  na         = obj%na
  matrixRows = obj%local_nrows
  nblk       = obj%nblk
  matrixCols = obj%local_ncols

  call obj%get("debug", debug, error)
  if (error .ne. ELPA_OK) then
    print *,"ELPA_INVERT_TRM: Error getting option for debug. Aborting..."
    stop 1
  endif
  if (debug == 1) then
    wantDebug = .true.
  else
    wantDebug = .true.
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

  l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a
  l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local cols of a

  if (useGPU) then
    successGPU = gpu_malloc(tmp1_dev, nblk*(nblk+1)/2*size_of_datatype)
    check_alloc_gpu("elpa_invert_trm: tmp1_dev", successGPU)

    successGPU = gpu_malloc(tmp2_dev, nblk*nblk*size_of_datatype)
    check_alloc_gpu("elpa_invert_trm: tmp2_dev", successGPU)

    successGPU = gpu_malloc(tmat1_dev, l_rows*nblk*size_of_datatype)
    check_alloc_gpu("elpa_invert_trm: tmat1_dev", successGPU)

    successGPU = gpu_malloc(tmat2_dev, nblk*l_cols*size_of_datatype)
    check_alloc_gpu("elpa_invert_trm: tmat2_dev", successGPU)

#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream

    successGPU = gpu_memset_async(tmp2_dev, 0, nblk*nblk*size_of_datatype, my_stream)
    check_memcpy_gpu("elpa_invert_trm: memset tmp2_dev", successGPU)
#else
    successGPU = gpu_memset(tmp2_dev, 0, nblk*nblk*size_of_datatype)
    check_memcpy_gpu("elpa_invert_trm: memset tmp2_dev", successGPU)
#endif

#ifndef DEVICE_POINTER
    successGPU = gpu_malloc(a_dev, matrixRows*matrixCols*size_of_datatype)
    check_alloc_gpu("elpa_invert_trm: a_dev", successGPU)
#ifdef WITH_GPU_STREAMS
    ! successGPU = gpu_host_register(int(loc(a),kind=c_intptr_t), &
    !                 matrixRows*matrixCols * size_of_datatype,&
    !                 gpuHostRegisterDefault)
    ! check_host_register_gpu("elpa_invert_trm: a", successGPU)
#endif
#else /* DEVICE_POINTER */

#ifdef WITH_NVTX
    call nvtxRangePush("transfer(aDev, a_dev)")
#endif
    ! associate with a_dev
    a_dev = transfer(aDev, a_dev)
#ifdef WITH_NVTX
    call nvtxRangePop() ! transfer(aDev, a_dev)")
#endif

#if !defined(INVERT_TRM_GPU_SOLVER)
    ! allocate a_tmp
    allocate(a_tmp(obj%local_nrows,obj%local_ncols), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_invert_trm: a_tmp", istat, errorMessage)
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_host_register(int(loc(a_tmp),kind=c_intptr_t), &
                    matrixRows*matrixCols * size_of_datatype,&
                    gpuHostRegisterDefault)
    check_host_register_gpu("elpa_invert_trm: a_tmp", successGPU)
#endif
#endif /* !defined(INVERT_TRM_GPU_SOLVER) */
#endif /* DEVICE_POINTER */

  endif ! useGPU

#ifdef WITH_NVTX
  call nvtxRangePush("allocate tmp1, tmp2, tmat1, tmat2")
#endif

  allocate(tmp1(nblk*(nblk+1)/2), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_invert_trm: tmp1", istat, errorMessage)

  allocate(tmp2(nblk,nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_invert_trm: tmp2", istat, errorMessage)

  allocate(tmat1(l_rows,nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_invert_trm: tmat1", istat, errorMessage)

  allocate(tmat2(nblk,l_cols), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_invert_trm: tmat2", istat, errorMessage)

#ifdef WITH_NVTX
  call nvtxRangePop() ! allocate tmp1, tmp2, tmat1, tmat2
#endif

  if (.not. useGPU) then
    tmp2 = 0
  endif

#ifdef WITH_GPU_STREAMS
  if (useGPU) then
    successGPU = gpu_host_register(int(loc(tmp1),kind=c_intptr_t), &
                    nblk*(nblk+1)/2 * size_of_datatype, &
                    gpuHostRegisterDefault)
    check_host_register_gpu("elpa_invert_trm: tmp1", successGPU)

    successGPU = gpu_host_register(int(loc(tmat1),kind=c_intptr_t), &
                    l_rows*nblk * size_of_datatype,&
                    gpuHostRegisterDefault)
    check_host_register_gpu("elpa_invert_trm: tmat1", successGPU)

    successGPU = gpu_host_register(int(loc(tmat2),kind=c_intptr_t), &
                    nblk * l_cols * size_of_datatype,&
                    gpuHostRegisterDefault)
    check_host_register_gpu("elpa_invert_trm: tmat1", successGPU)

  endif
#endif


#ifndef DEVICE_POINTER
  if (useGPU) then
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream

    num = matrixRows*matrixCols* size_of_datatype
    call gpu_memcpy_async_and_stream_synchronize &
    ("elpa_invert_trm: a to a_dev", a_dev, 0_c_intptr_t, &
                                    a(1:obj%local_nrows,1:obj%local_ncols), &
                                    1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
    successGPU = gpu_memcpy(a_dev, int(loc(a(1,1)),kind=c_intptr_t),  &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyHostToDevice)
    check_memcpy_gpu("elpa_invert_trm: memcpy a-> a_dev", successGPU)
#endif /* WITH_GPU_STREAMS */
  endif
#endif /* DEVICE_POINTER */


  ns = ((na-1)/nblk)*nblk + 1

  do n = ns,1,-nblk
#ifdef WITH_NVTX
    call nvtxRangePush("do n = ns,1,-nblk")
#endif

    l_row1 = local_index(n, my_prow, np_rows, nblk, +1)
    l_col1 = local_index(n, my_pcol, np_cols, nblk, +1)

    nb = nblk
    if (na-n+1 < nblk) nb = na-n+1

    l_rowx = local_index(n+nb, my_prow, np_rows, nblk, +1)
    l_colx = local_index(n+nb, my_pcol, np_cols, nblk, +1)

    if (my_prow==prow(n, nblk, np_rows)) then

      if (my_pcol==pcol(n, nblk, np_cols)) then
        if (useGPU) then

#if defined(INVERT_TRM_GPU_SOLVER)
#ifdef WITH_NVTX
          call nvtxRangePush("gpusolver_TRTRI")
#endif
          call obj%timer%start("gpusolver")
          gpusolverHandle = obj%gpu_setup%gpusolverHandleArray(0)
          a_off = ((l_row1-1) + (l_col1-1)*matrixRows) * size_of_datatype
          call gpusolver_PRECISION_TRTRI('U', 'N', int(nb,kind=c_int64_t), a_dev+a_off, int(matrixRows,c_int64_t), &
                             info, gpusolverHandle)
          if (info .ne. 0) then
            write(error_unit,*) "elpa_invert_trm: error in gpusolver_TRTRI"
            stop 1
          endif
          call obj%timer%stop("gpusolver")
#ifdef WITH_NVTX
          call nvtxRangePop() ! gpusolver_TRTRI
#endif   
#else /* defined(INVERT_TRM_GPU_SOLVER) */
         
          ! still have to use cpu blas -> a generic GPU implementation would be needed
#ifndef DEVICE_POINTER
          call obj%timer%start("lapack")
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          num = matrixRows*matrixCols* size_of_datatype

          call gpu_memcpy_async_and_stream_synchronize &
          ("elpa_invert_trm: a_dev to a", a_dev, 0_c_intptr_t, &
                                          a(1:obj%local_nrows,1:obj%local_ncols), &
                                          1, 1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
          successGPU = gpu_memcpy(int(loc(a(1,1)),kind=c_intptr_t), a_dev, &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("invert_trm: memcpy a_dev -> a", successGPU)
#endif /* WITH_GPU_STREAMS */

          call PRECISION_TRTRI('U', 'N', int(nb,kind=BLAS_KIND), a(l_row1,l_col1), int(matrixRows,kind=BLAS_KIND), &
                             infoBLAS)
          info = int(infoBLAS,kind=ik)

#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream

          num = matrixRows*matrixCols* size_of_datatype
          call gpu_memcpy_async_and_stream_synchronize &
          ("elpa_invert_trm: a to a_dev", a_dev, 0_c_intptr_t, &
                                    a(1:obj%local_nrows,1:obj%local_ncols), &
                                    1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
          successGPU = gpu_memcpy(a_dev, int(loc(a(1,1)),kind=c_intptr_t),  &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("invert_trm: memcpy a -> a_dev", successGPU)
#endif /* WITH_GPU_STREAMS */
          call obj%timer%stop("lapack")
#else /* DEVICE_POINTER */

#if !defined(INVERT_TRM_GPU_SOLVER)
          call obj%timer%start("lapack")
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream

          num = matrixRows*matrixCols* size_of_datatype

          call gpu_memcpy_async_and_stream_synchronize &
          ("elpa_invert_trm: a_dev to a_tmp", a_dev, 0_c_intptr_t, &
                                          a_tmp(1:obj%local_nrows,1:obj%local_ncols), &
                                          1, 1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
          successGPU = gpu_memcpy(int(loc(a_tmp(1,1)),kind=c_intptr_t), a_dev, &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("invert_trm: memcpy a_dev -> a", successGPU)
#endif /* WITH_GPU_STREAMS */

          call PRECISION_TRTRI('U', 'N', int(nb,kind=BLAS_KIND), a_tmp(l_row1,l_col1), int(matrixRows,kind=BLAS_KIND), &
                             infoBLAS)
          info = int(infoBLAS,kind=ik)

#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream

          num = matrixRows*matrixCols* size_of_datatype
          call gpu_memcpy_async_and_stream_synchronize &
          ("elpa_invert_trm: a_tmp to a_dev", a_dev, 0_c_intptr_t, &
                                    a_tmp(1:obj%local_nrows,1:obj%local_ncols), &
                                    1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
          successGPU = gpu_memcpy(a_dev, int(loc(a_tmp(1,1)),kind=c_intptr_t),  &
                       matrixRows*matrixCols* size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("invert_trm: memcpy a -> a_dev", successGPU)
#endif /* WITH_GPU_STREAMS */
          call obj%timer%stop("lapack")
#endif /* !defined(INVERT_TRM_GPU_SOLVER) */
#endif /* DEVICE_POINTER */
#endif /* defined(INVERT_TRM_GPU_SOLVER) */

        else ! useGPU
          call obj%timer%start("blas")

#ifdef DEVICE_POINTER
!          call PRECISION_TRTRI('U', 'N', int(nb,kind=BLAS_KIND), a_tmp(l_row1,l_col1), int(matrixRows,kind=BLAS_KIND), &
!                             infoBLAS)
#else
          call PRECISION_TRTRI('U', 'N', int(nb,kind=BLAS_KIND), a(l_row1,l_col1), int(matrixRows,kind=BLAS_KIND), &
                             infoBLAS)
#endif
          info = int(infoBLAS,kind=ik)
          call obj%timer%stop("blas")
        endif ! useGPU
        if (info/=0) then
          if (wantDebug) write(error_unit,*) "elpa_invert_trm_&
            &MATH_DATATYPE&

#if REALCASE == 1
          &: Error in DTRTRI"
#endif
#if COMPLEXCASE == 1
          &: Error in ZTRTRI" !"
#endif

          success = .false.
          call obj%timer%stop("elpa_invert_trm_&
          &MATH_DATATYPE&
          &_&
                  &PRECISION&
          &"//gpuString)
          return
        endif

        if (useGPU) then
          my_stream = obj%gpu_setup%my_stream
          call gpu_copy_PRECISION_a_tmp1 (a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb, my_stream)
        else ! useGPU
          nc = 0
          do i=1,nb
#ifndef DEVICE_POINTER
            tmp1(nc+1:nc+i) = a(l_row1:l_row1+i-1,l_col1+i-1)
!#else
!            tmp1(nc+1:nc+i) = a_tmp(l_row1:l_row1+i-1,l_col1+i-1)
#endif
            nc = nc+i
          enddo
        endif ! useGPU
      endif ! my_pcol==pcol(n, nblk, np_cols)

#ifdef WITH_MPI
      if (useGPU .and. .not. useCCL) then
        num = nblk*(nblk+1)/2
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_invert_trm: tmp1_dev to tmp1", tmp1_dev, 0_c_intptr_t, tmp1(1:num), &
                1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
        successGPU = gpu_memcpy(int(loc(tmp1),kind=c_intptr_t), tmp1_dev, num*size_of_datatype, &
                              gpuMemcpyDeviceToHost)
        check_memcpy_gpu("elpa_invert_trm: tmp1_dev to tmp1", successGPU)
#endif /* WITH_GPU_STREAMS */
      endif ! (useGPU .and. .not. useCCL)


! #ifdef WITH_CUDA_AWARE_MPI
!       if (useGPU) then
!         tmp1_mpi_dev = transfer(tmp1_dev, tmp1_mpi_dev) 
!         ! and associate a fortran pointer
!         call c_f_pointer(tmp1_mpi_dev, tmp1_mpi_fortran_ptr, [nblk*nblk])
!         if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
!         successGPU = gpu_devicesynchronize()
!         check_memcpy_gpu("invert_trm: device_synchronize", successGPU)
!         if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")

!         if (wantDebug) call obj%timer%start("cuda_mpi_communication")
!         call MPI_Bcast(tmp1_mpi_fortran_ptr, int(nb*(nb+1)/2,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,       &
!                        int(pcol(n, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
!         if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
!       endif ! useGPU
! #endif /* WITH_CUDA_AWARE_MPI */

      if (useCCL) then
#ifdef USE_CCL_INVERT
        call obj%timer%start("ccl_bcast")
        my_stream = obj%gpu_setup%my_stream
        ccl_comm_cols = obj%gpu_setup%ccl_comm_cols
        
        successGPU = ccl_bcast(tmp1_dev, tmp1_dev, k_datatype*int(nb*(nb+1)/2,kind=c_size_t), &
                               cclDataType, int(pcol(n, nblk, np_cols),kind=c_int), ccl_comm_cols, my_stream)
        if (.not. successGPU) then
          print *,"Error in ccl_bcast"
          stop 1
        endif

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("elpa_invert_trm: ccl_bcast", successGPU)
        call obj%timer%stop("ccl_bcast")
#endif /* USE_CCL_INVERT */

      else ! useCCL
          call obj%timer%start("mpi_communication")
          call MPI_Bcast(tmp1, int(nb*(nb+1)/2,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,       &
                     int(pcol(n, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
          call obj%timer%stop("mpi_communication")
      endif ! useCCL

      if (useGPU .and. .not. useCCL) then  
        num = nblk*(nblk+1)/2
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_invert_trm: tmp1 to tmp1_dev", tmp1_dev, 0_c_intptr_t, tmp1(1:num), &
                1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
        successGPU = gpu_memcpy(tmp1_dev, int(loc(tmp1),kind=c_intptr_t), num*size_of_datatype, &
                              gpuMemcpyHostToDevice)
        check_memcpy_gpu("elpa_invert_trm: tmp1 to tmp1_dev", successGPU)
#endif /* WITH_GPU_STREAMS */
      endif ! (useGPU .and. .not. useCCL)
#endif /* WITH_MPI */
      
      if (useGPU) then
        call gpu_copy_PRECISION_tmp1_tmp2 (tmp1_dev, tmp2_dev, nblk, nb, my_stream)
      else ! useGPU
        nc = 0
        do i=1,nb
          tmp2(1:i,i) = tmp1(nc+1:nc+i)
          nc = nc+i
        enddo
      endif ! useGPU

      if (useGPU) then
        call obj%timer%start("gpublas")
        gpublasHandle = obj%gpu_setup%gpublasHandleArray(0)
        if (l_cols-l_colx+1 > 0) then
          a_off = (l_row1 -1 + (l_colx-1)*matrixRows) * size_of_datatype

#ifdef WITH_NVTX
          call nvtxRangePush("gpublas_TRMM")
#endif
          call gpublas_PRECISION_TRMM('L', 'U', 'N', 'N', nb, l_cols-l_colx+1, ONE, tmp2_dev, &
                                      nblk, a_dev+a_off, matrixRows, gpublasHandle)
#ifdef WITH_NVTX
          call nvtxRangePop()! gpublas_TRMM
#endif
        endif
        call obj%timer%stop("gpublas")

        if (l_colx <= l_cols) then
          my_stream = obj%gpu_setup%my_stream
          call gpu_copy_PRECISION_a_tmat2 (a_dev, tmat2_dev, nblk, matrixRows, l_cols, l_colx, & 
                                       l_row1, nb, my_stream)
        endif

        if (my_pcol==pcol(n, nblk, np_cols)) then
           ! tmp2 has the lower left triangle 0
          call gpu_copy_PRECISION_tmp2_tmat2 (tmp2_dev, tmat2_dev, nblk, l_col1, nb, my_stream) 
        endif
      else ! useGPU
        call obj%timer%start("blas")
#ifndef DEVICE_POINTER
        if (l_cols-l_colx+1>0) then
          call PRECISION_TRMM('L', 'U', 'N', 'N', int(nb,kind=BLAS_KIND), int(l_cols-l_colx+1,kind=BLAS_KIND), ONE, &
                              tmp2, int(ubound(tmp2,dim=1),kind=BLAS_KIND), a(l_row1,l_colx), int(matrixRows,kind=BLAS_KIND))
        endif
        call obj%timer%stop("blas")
        if (l_colx<=l_cols)   tmat2(1:nb,l_colx:l_cols) = a(l_row1:l_row1+nb-1,l_colx:l_cols)
        if (my_pcol==pcol(n, nblk, np_cols)) tmat2(1:nb,l_col1:l_col1+nb-1) = tmp2(1:nb,1:nb) ! tmp2 has the lower left triangle 0
!#else
!        if (l_cols-l_colx+1>0) then
!          call PRECISION_TRMM('L', 'U', 'N', 'N', int(nb,kind=BLAS_KIND), int(l_cols-l_colx+1,kind=BLAS_KIND), ONE, &
!                              tmp2, int(ubound(tmp2,dim=1),kind=BLAS_KIND), a_tmp(l_row1,l_colx), int(matrixRows,kind=BLAS_KIND))
!        endif
!        call obj%timer%stop("blas")
!        if (l_colx<=l_cols)   tmat2(1:nb,l_colx:l_cols) = a_tmp(l_row1:l_row1+nb-1,l_colx:l_cols)
!        if (my_pcol==pcol(n, nblk, np_cols)) tmat2(1:nb,l_col1:l_col1+nb-1) = tmp2(1:nb,1:nb) ! tmp2 has the lower left triangle 0
#endif
      endif ! useGPU

    endif ! (my_prow==prow(n, nblk, np_rows)

    if (l_row1>1) then
      if (my_pcol==pcol(n, nblk, np_cols)) then
        if (useGPU) then
          my_stream = obj%gpu_setup%my_stream
          call gpu_copy_PRECISION_a_tmat1 (a_dev, tmat1_dev, l_rows, matrixRows, nb, l_row1, l_col1, my_stream)
        else
#ifndef DEVICE_POINTER
          tmat1(1:l_row1-1,1:nb) = a(1:l_row1-1,l_col1:l_col1+nb-1)
          a(1:l_row1-1,l_col1:l_col1+nb-1) = 0
!#else
!          tmat1(1:l_row1-1,1:nb) = a_tmp(1:l_row1-1,l_col1:l_col1+nb-1)
!          a_tmp(1:l_row1-1,l_col1:l_col1+nb-1) = 0
#endif
        endif
      endif

#ifdef WITH_MPI
      if (useGPU .and. .not. useCCL) then
        num = l_rows*nblk
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream

        call gpu_memcpy_async_and_stream_synchronize &
          ("elpa_invert_trm: tmat1_dev to tmat1", tmat1_dev, 0_c_intptr_t, &
                                          tmat1(1:l_rows,1:nblk), &
                                          1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
        successGPU = gpu_memcpy(int(loc(tmat1),kind=c_intptr_t), tmat1_dev, num*size_of_datatype, &
                              gpuMemcpyDeviceToHost)
        check_memcpy_gpu("elpa_invert_trm: tmat1_dev to tmat1", successGPU)
#endif /* WITH_GPU_STREAMS */
      endif ! (useGPU .and. .not. useCCL)


! #ifdef WITH_CUDA_AWARE_MPI
!       if (useGPU) then
!         tmat1_mpi_dev = transfer(tmat1_dev, tmat1_mpi_dev)
!         ! and associate a fortran pointer
!         call c_f_pointer(tmat1_mpi_dev, tmat1_mpi_fortran_ptr, [l_rows,nblk])
!         if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
!         successGPU = gpu_devicesynchronize()
!         check_memcpy_gpu("invert_trm: device_synchronize", successGPU)
!         if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
!         call obj%timer%start("mpi_cuda_communication")
!         do i=1,nb
!           call MPI_Bcast(tmat1_mpi_fortran_ptr(1,i), int(l_row1-1,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
!                        int(pcol(n, nblk, np_cols),kind=MPI_KIND), & 
!                        int(mpi_comm_cols,kind=MPI_KIND), mpierr)

!         enddo
!         call obj%timer%stop("mpi_cuda_communication")
!      endif ! useGPU
! #endif

      if (useCCL) then
#ifdef USE_CCL_INVERT
        call obj%timer%start("ccl_bcast")
#ifdef WITH_NVTX
        call nvtxRangePush("ccl_bcast_group tmat1_dev")
#endif
        successGPU = ccl_group_start()
        if (.not. successGPU) then
          print *, "Error in setting up ccl_group_start!"
          stop 1
        endif

        my_stream = obj%gpu_setup%my_stream
        ccl_comm_cols = obj%gpu_setup%ccl_comm_cols
        do i=1,nb
          offset = (1-1 + l_rows*(i-1)) * size_of_datatype

          successGPU = ccl_bcast(tmat1_dev + offset, tmat1_dev + offset, int(k_datatype*(l_row1-1),kind=c_size_t), cclDatatype, &
                                 int(pcol(n, nblk, np_cols),kind=c_int), ccl_comm_cols, my_stream)

          if (.not. successGPU) then
            print *,"Error in ccl_bcast"
            stop 1
          endif
        enddo ! i=1,nb

        successGPU = ccl_group_end()
        if (.not. successGPU) then
          print *, "Error in setting up ccl_group_end!"
          stop 1
        endif

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("elpa_invert_trm: ccl_bcast", successGPU)
#ifdef WITH_NVTX
        call nvtxRangePop() ! ccl_bcast_group tmat1_dev
#endif
        call obj%timer%stop("ccl_bcast")
#endif /* USE_CCL_INVERT */
      else ! useCCL

#ifdef WITH_NVTX
        call nvtxRangePush("MPI_Bcast tmat1")
#endif
        call obj%timer%start("mpi_communication")
        ! do i=1,nb
        !   call MPI_Bcast(tmat1(1,i), int(l_row1-1,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
        !                  int(pcol(n, nblk, np_cols),kind=MPI_KIND), & 
        !                  int(mpi_comm_cols,kind=MPI_KIND), mpierr)
        ! enddo
        call MPI_Bcast(tmat1(1,1), int(l_rows*nblk,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                  int(pcol(n, nblk, np_cols),kind=MPI_KIND), &
                  int(mpi_comm_cols,kind=MPI_KIND), mpierr)
        call obj%timer%stop("mpi_communication")
#ifdef WITH_NVTX
        call nvtxRangePop() ! MPI_Bcast tmat1
#endif
      endif ! useCCL

      if (useGPU .and. .not. useCCL) then
        ! cuda aware MPI here
        num = l_rows*nblk
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream

        call gpu_memcpy_async_and_stream_synchronize &
            ("elpa_invert_trm: tmat1 to tmat1_dev", tmat1_dev, 0_c_intptr_t, tmat1(1:l_rows,1:nblk), &
              1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
        successGPU = gpu_memcpy(tmat1_dev, int(loc(tmat1),kind=c_intptr_t), num*size_of_datatype, &
                              gpuMemcpyHostToDevice)
        check_memcpy_gpu("elpa_invert_trm: tmat1 to tmat1_dev", successGPU)
#endif /* WITH_GPU_STREAMS */
      endif ! (useGPU .and. .not. useCCL)

#endif /* WITH_MPI */
    endif ! (l_row1>1)

! #ifdef WITH_CUDA_AWARE_MPI
!     if (useGPU) then
!       tmat2_mpi_dev = transfer(tmat2_dev, tmat2_mpi_dev)     
!       call c_f_pointer(tmat2_mpi_dev, tmat2_mpi_fortran_ptr, [nblk,l_cols])
      
!       if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
!       successGPU = gpu_devicesynchronize()
!       check_memcpy_gpu("invert_trm: device_synchronize", successGPU)
!       if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
!       call obj%timer%start("mpi_cuda_communication")
!       if (l_cols-l_col1+1 > 0) &
!         call MPI_Bcast(tmat2_mpi_fortran_ptr(1,l_col1), int((l_cols-l_col1+1)*nblk,kind=MPI_KIND), & 
!                        MPI_MATH_DATATYPE_PRECISION, int(prow(n, nblk, np_rows),kind=MPI_KIND), & 
!                        int(mpi_comm_rows,kind=MPI_KIND), mpierr)
!       call obj%timer%stop("mpi_cuda_communication")
!     endif ! useGPU
! #endif /* WITH_CUDA_AWARE_MPI */

#ifdef WITH_MPI
    if (l_cols-l_col1+1 > 0) then

      if (useGPU .and. .not. useCCL) then
        num = nblk*l_cols
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream

        call gpu_memcpy_async_and_stream_synchronize &
            ("elpa_invert_trm: tmat2_dev to tmat2", tmat2_dev, 0_c_intptr_t, tmat2(1:nblk,1:l_cols), &
              1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
        successGPU = gpu_memcpy(int(loc(tmat2),kind=c_intptr_t), tmat2_dev, num*size_of_datatype, &
                              gpuMemcpyDeviceToHost)
        check_memcpy_gpu("elpa_invert_trm: tmat2_dev to tmat2", successGPU)
#endif /* WITH_GPU_STREAMS */
      endif ! (useGPU .and. .not. useCCL)

      if (useCCL) then
#ifdef USE_CCL_INVERT
        call obj%timer%start("ccl_bcast")
        my_stream = obj%gpu_setup%my_stream
        ccl_comm_rows = obj%gpu_setup%ccl_comm_rows
        offset = (1-1 + nblk*(l_col1-1)) * size_of_datatype

        successGPU = ccl_bcast(tmat2_dev+offset, tmat2_dev+offset, int(k_datatype*(l_cols-l_col1+1)*nblk,kind=c_size_t), &
                               cclDataType, int(prow(n, nblk, np_rows),kind=c_int), ccl_comm_rows, my_stream)
        if (.not. successGPU) then
          print *,"Error in ccl_bcast"
          stop 1
        endif

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("elpa_invert_trm: ccl_bcast", successGPU)
        call obj%timer%stop("ccl_bcast")
#endif /* USE_CCL_INVERT */
      else ! useCCL

        call obj%timer%start("mpi_communication")
        call MPI_Bcast(tmat2(1,l_col1), int((l_cols-l_col1+1)*nblk,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                    int(prow(n, nblk, np_rows),kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
        call obj%timer%stop("mpi_communication")

      endif ! useCCL

      if (useGPU .and. .not. useCCL) then
        num = nblk*l_cols
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream

        call gpu_memcpy_async_and_stream_synchronize &
            ("elpa_invert_trm: tmat2 to tmat2_dev", tmat2_dev, 0_c_intptr_t, tmat2(1:nblk,1:l_cols), &
              1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
        successGPU = gpu_memcpy(tmat2_dev, int(loc(tmat2),kind=c_intptr_t), num*size_of_datatype, &
                                gpuMemcpyHostToDevice)
        check_memcpy_gpu("elpa_invert_trm: tmat2 to tmat2_dev", successGPU)
#endif /* WITH_GPU_STREAMS */
      endif ! (useGPU .and. .not. useCCL)

    endif ! (l_cols-l_col1+1 > 0)
#endif /* WITH_MPI */

    if (useGPU) then
      call obj%timer%start("gpublas")

      gpublasHandle = obj%gpu_setup%gpublasHandleArray(0)
      tmat2_off = (1 - 1 + (l_col1-1) * nblk) * size_of_datatype      
      a_off = (1 - 1 + (l_col1-1) * matrixRows) * size_of_datatype
      if (l_row1>1 .and. l_cols-l_col1+1>0) then
        call gpublas_PRECISION_GEMM('N', 'N', l_row1-1, l_cols-l_col1+1, nb, -ONE, &
                                    tmat1_dev, l_rows, tmat2_dev + tmat2_off, &
                                    nblk, ONE, a_dev+a_off, matrixRows, gpublasHandle)
      endif
      call obj%timer%stop("gpublas")

    else ! useGPU
      call obj%timer%start("blas")
#ifndef DEVICE_POINTER
      if (l_row1>1 .and. l_cols-l_col1+1>0) then
        call PRECISION_GEMM('N', 'N', int(l_row1-1,kind=BLAS_KIND), int(l_cols-l_col1+1,kind=BLAS_KIND), &
                            int(nb,kind=BLAS_KIND), -ONE, &
                             tmat1, int(ubound(tmat1,dim=1),kind=BLAS_KIND), tmat2(1,l_col1), &
                             int(ubound(tmat2,dim=1),kind=BLAS_KIND), ONE, &
                              a(1,l_col1), int(matrixRows,kind=BLAS_KIND) )
      endif
!#else
!      if (l_row1>1 .and. l_cols-l_col1+1>0) then
!        call PRECISION_GEMM('N', 'N', int(l_row1-1,kind=BLAS_KIND), int(l_cols-l_col1+1,kind=BLAS_KIND), &
!                            int(nb,kind=BLAS_KIND), -ONE, &
!                             tmat1, int(ubound(tmat1,dim=1),kind=BLAS_KIND), tmat2(1,l_col1), &
!                             int(ubound(tmat2,dim=1),kind=BLAS_KIND), ONE, &
!                              a_tmp(1,l_col1), int(matrixRows,kind=BLAS_KIND) )
!      endif
#endif

      call obj%timer%stop("blas")
    endif ! useGPU
#ifdef WITH_NVTX
    call nvtxRangePop() ! do n = ns,1,-nblk
#endif
  enddo ! n = ns,1,-nblk

#ifndef DEVICE_POINTER
  if (useGPU) then
    ! copy results back
    num = matrixRows*matrixCols
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    call gpu_memcpy_async_and_stream_synchronize &
          ("elpa_invert_trm: a to a_dev", a_dev, 0_c_intptr_t, a(1:matrixRows,1:matrixCols), &
            1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
    successGPU = gpu_memcpy(int(loc(a(1,1)),kind=c_intptr_t), a_dev, &
                            num*size_of_datatype, gpuMemcpyDeviceToHost)
    check_memcpy_gpu("elpa_invert_trm: memcpy a-> d_dev", successGPU)
#endif /* WITH_GPU_STREAMS */
  endif ! useGPU
#endif /* DEVICE_POINTER */

  if (useGPU) then
    successGPU = gpu_free(tmp1_dev)
    check_dealloc_gpu("elpa_invert_trm: tmp1_dev", successGPU)

    successGPU = gpu_free(tmp2_dev)
    check_dealloc_gpu("elpa_invert_trm: tmp2_dev", successGPU)

    successGPU = gpu_free(tmat1_dev)
    check_dealloc_gpu("elpa_invert_trm: tmat1_dev", successGPU)

    successGPU = gpu_free(tmat2_dev)
    check_dealloc_gpu("elpa_invert_trm: tmat2_dev", successGPU)

#ifndef DEVICE_POINTER
    successGPU = gpu_free(a_dev)
    check_dealloc_gpu("elpa_invert_trm: a_dev", successGPU)

#ifdef WITH_GPU_STREAMS
    ! successGPU = gpu_host_unregister(int(loc(a),kind=c_intptr_t))
    ! check_host_unregister_gpu("elpa_invert_trm: a", successGPU)
#endif

#else /* DEVICE_POINTER */

#if !defined(INVERT_TRM_GPU_SOLVER)
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_host_unregister(int(loc(a_tmp),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_invert_trm: a_tmp", successGPU)
#endif

    deallocate(a_tmp, stat=istat, errmsg=errorMessage)
    check_deallocate("elpa_invert_trm: a_tmp", istat, errorMessage)
#endif /* defined(INVERT_TRM_GPU_SOLVER) */
#endif /* DEVICE_POINTER */

  endif ! useGPU

#ifdef WITH_GPU_STREAMS
  if (useGPU) then
    successGPU = gpu_host_unregister(int(loc(tmp1),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_invert_trm: tmp1", successGPU)

    successGPU = gpu_host_unregister(int(loc(tmat1),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_invert_trm: tmat1", successGPU)

    successGPU = gpu_host_unregister(int(loc(tmat2),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_invert_trm: tmat2", successGPU)
  endif
#endif /* WITH_GPU_STREAMS */

  deallocate(tmp1, tmp2, tmat1, tmat2, stat=istat, errmsg=errorMessage)
  check_deallocate("elpa_invert_trm: tmp1, tmp2, tmat1, tmat2", istat, errorMessage)

#ifdef WITH_NVTX
  call nvtxRangePop() ! invert_trm
#endif

  call obj%timer%stop("elpa_invert_trm_&
  &MATH_DATATYPE&
  &_&
  &PRECISION&
  &"//gpuString)