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
!
! Author: A. Marek, MPCDF



#include "../general/sanity.F90"
#include "../general/error_checking.inc"

  use elpa1_compute, solve_tridi_cpu_&
                      &PRECISION&
                      &_private_impl => solve_tridi_cpu_&
                      &PRECISION&
                      &_impl, &
                      solve_tridi_gpu_&
                      &PRECISION&
                      &_private_impl => solve_tridi_gpu_&
                      &PRECISION&
                      &_impl
  use precision
  use elpa_abstract_impl
  use elpa_omp
  use solve_tridi
  use elpa_gpu
  use mod_query_gpu_usage
  use mod_check_for_gpu
  use elpa_utilities, only : check_alloc_gpu_f, check_dealloc_gpu_f, check_memcpy_gpu_f, error_unit
  implicit none
  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik)         :: na, nev, matrixRows, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
  integer(kind=ik)         :: mpi_comm_all, myid
  real(kind=REAL_DATATYPE) :: d(obj%na), e(obj%na)
#ifdef USE_ASSUMED_SIZE
  real(kind=REAL_DATATYPE) :: q(obj%local_nrows,*)
#else
  real(kind=REAL_DATATYPE) :: q(obj%local_nrows, obj%local_ncols)
#endif

  logical                  :: wantDebug
  logical                  :: success

  integer                  :: debug, error
  integer                  :: nrThreads, limitThreads
  
  logical                  :: useGPU, successGPU
  integer(kind=c_int)      :: numGPU
  integer(kind=c_intptr_t) :: d_dev, e_dev, q_dev
  integer(kind=c_intptr_t) :: num
  integer(kind=c_intptr_t) :: gpuHandle, my_stream
  integer(kind=c_intptr_t), parameter :: size_of_datatype_real = size_of_&
                                                                        &PRECISION&
                                                                        &_real

  call obj%timer%start("elpa_solve_tridi_public_&
  &MATH_DATATYPE&
  &_&
  &PRECISION&
  &")
  na         = obj%na
  nev        = obj%nev
  nblk       = obj%nblk
  matrixRows = obj%local_nrows
  matrixCols = obj%local_ncols

  mpi_comm_all    = obj%mpi_setup%mpi_comm_parent
  mpi_comm_cols   = obj%mpi_setup%mpi_comm_cols
  mpi_comm_rows   = obj%mpi_setup%mpi_comm_rows
  myid            = obj%mpi_setup%myRank_comm_parent

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

  call obj%get("debug",debug,error)
  if (error .ne. ELPA_OK) then
    print *,"Problem getting option for debug. Aborting..."
    stop 1
  endif
  if (debug == 1) then
    wantDebug = .true.
  else
    wantDebug = .false.
  endif
  success = .false.

  useGPU     = .false.
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
  ! query keywords gpu, nvidia-gpu, amd-gpu and check for consisteny between actual and legacy GPU setings
  if (.not.(query_gpu_usage(obj, "solve_tridi_impl", useGPU))) then
    write(error_unit,*) "solve_tridi_impl: Problem getting options for GPU. Aborting..."
    success = .false.
    return
  endif
#endif /* defined(WITH_NVIDIA_GPU_VERSION) ... */

  if (useGPU) then
    call obj%timer%start("check_for_gpu")
    if (check_for_gpu(obj, myid, numGPU, wantDebug)) then
      ! set the neccessary parameters
      call set_gpu_parameters()
    else
      write(error_unit,*) "GPUs are requested but not detected! Aborting..."
      success = .false.
      return
    endif
    call obj%timer%stop("check_for_gpu")
  endif
        
      if (useGPU) then
        num = na * size_of_datatype_real
        successGPU = gpu_malloc(d_dev, num)
        check_alloc_gpu("elpa_solve_tridi_impl d_dev", successGPU)

        num = na * size_of_datatype_real
        successGPU = gpu_malloc(e_dev, num)
        check_alloc_gpu("elpa_solve_tridi_impl e_dev", successGPU)

        num = (matrixRows* matrixCols) * size_of_datatype_real
        successGPU = gpu_malloc(q_dev, num)
        check_alloc_gpu("elpa_solve_tridi_impl q_dev", successGPU)
        

        num = na * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        successGPU = gpu_memcpy_async(d_dev, int(loc(d(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
#else
        successGPU = gpu_memcpy(d_dev, int(loc(d(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
#endif
        check_memcpy_gpu("elpa_solve_tridi_impl: d_dev", successGPU)

        num = na * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(e_dev, int(loc(e(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
#else
        successGPU = gpu_memcpy(e_dev, int(loc(e(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
#endif
        check_memcpy_gpu("elpa_solve_tridi_impl: e_dev", successGPU)

        num = (matrixRows* matrixCols) * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(q_dev, int(loc(q(1,1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
        successGPU = gpu_stream_synchronize(my_stream)
#else
        successGPU = gpu_memcpy(q_dev, int(loc(q(1,1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
#endif
        check_memcpy_gpu("elpa_solve_tridi_impl: q_dev", successGPU)

        call solve_tridi_gpu_&
                            &PRECISION&
                            &_private_impl(obj, na, nev, d_dev, e_dev, q_dev, matrixRows, nblk, matrixCols, &
                            mpi_comm_all, mpi_comm_rows, mpi_comm_cols, wantDebug, success, nrThreads)

        num = na * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(int(loc(d(1)),kind=c_intptr_t), d_dev, num, gpuMemcpyDeviceToHost, my_stream)
#else
        successGPU = gpu_memcpy(int(loc(d(1)),kind=c_intptr_t), d_dev, num, gpuMemcpyDeviceToHost)
#endif
        check_memcpy_gpu("elpa_solve_tridi_impl: d_dev", successGPU)

        num = na * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(int(loc(e(1)),kind=c_intptr_t), e_dev, num, gpuMemcpyDeviceToHost, my_stream)
#else
        successGPU = gpu_memcpy(int(loc(e(1)),kind=c_intptr_t), e_dev, num, gpuMemcpyDeviceToHost)
#endif
        check_memcpy_gpu("elpa_solve_tridi_impl: e_dev", successGPU)

        num = (matrixRows* matrixCols) * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(int(loc(q(1,1)),kind=c_intptr_t), q_dev, num, gpuMemcpyDeviceToHost, my_stream)
#else
        successGPU = gpu_memcpy(int(loc(q(1,1)),kind=c_intptr_t), q_dev, num, gpuMemcpyDeviceToHost)
#endif
        check_memcpy_gpu("elpa_solve_tridi_impl: q_dev", successGPU)

        successGPU = gpu_free(q_dev)
        check_dealloc_gpu("solve_tridi_impl q_dev", successGPU)
      else
        call solve_tridi_cpu_&
                            &PRECISION&
                            &_private_impl(obj, na, nev, d, e, q, matrixRows, nblk, matrixCols, &
                            mpi_comm_all, mpi_comm_rows, mpi_comm_cols, wantDebug, success, nrThreads)
      endif

      ! restore original OpenMP settings
#ifdef WITH_OPENMP_TRADITIONAL
      ! store the number of OpenMP threads used in the calling function
      ! restore this at the end of ELPA 2
      call omp_set_num_threads(omp_threads_caller)
#endif


      call obj%timer%stop("elpa_solve_tridi_public_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      &")


#undef REALCASE
#undef COMPLEXCASE
#undef DOUBLE_PRECISION
#undef SINGLE_PRECISION

