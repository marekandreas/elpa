!    Copyright 2014, A. Marek
!
!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
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
! This file was written by A. Marek, MPCDF

#include "config-f90.h"

module mod_check_for_gpu

  contains
    ! TODO: proper cleanup of handles and hanldeArrays
 
    ! check_for_gpu could be called at several places during a run of ELPA
    ! for example in cholesky, invert_trm, multiply and of course the solvers
    ! Thus the following logic is implemented
    ! if use_gpu_id is set -> do according to the user settings
    ! if NOT the first call to check_for_gpu will set the MPI GPU relation and then
    ! _SET_ use_gpu_id such that subsequent calls abide this setting
    function check_for_gpu(obj, myid, numberOfDevices, wantDebug) result(gpuAvailable)
      use cuda_functions
      use hip_functions
      use openmp_offload_functions
      use sycl_functions
      use elpa_gpu, only : gpuDeviceArray, gpublasHandleArray, my_stream
      use precision
      use elpa_mpi
      use elpa_omp
      use elpa_abstract_impl
      implicit none

      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=ik), intent(in)               :: myid
      logical, optional, intent(in)              :: wantDebug
      logical                                    :: success, wantDebugMessage
      integer(kind=ik), intent(out)              :: numberOfDevices
      integer(kind=ik)                           :: deviceNumber, mpierr, maxNumberOfDevices
      logical                                    :: gpuAvailable
      integer(kind=ik)                           :: error, mpi_comm_all, use_gpu_id, min_use_gpu_id
      logical, save                              :: alreadySET=.false.
      integer(kind=ik)                           :: maxThreads, thread
      integer(kind=c_intptr_t)                   :: handle_tmp
      !integer(kind=c_intptr_t)                   :: stream
      !logical                                    :: gpuIsInitialized=.false.
      !character(len=1024)           :: envname
      character(len=8)                           :: fmt 
      character(len=12)                          :: gpu_string
      if (.not.(present(wantDebug))) then
        wantDebugMessage = .false.
      else
        if (wantDebug) then
          wantDebugMessage=.true.
        else
          wantDebugMessage=.false.
        endif
      endif

      gpuAvailable = .false.

      call obj%get("mpi_comm_parent",mpi_comm_all,error)
      if (error .ne. ELPA_OK) then
        print *,"Problem getting option for mpi_comm_parent. Aborting..."
        stop
      endif

      ! needed later for handle creation
#ifdef WITH_OPENMP_TRADITIONAL
      maxThreads=omp_get_max_threads()
#else /* WITH_OPENMP_TRADITIONAL */
      maxThreads=1
#endif /* WITH_OPENMP_TRADITIONAL */

#ifdef WITH_NVIDIA_GPU_VERSION
      if (.not.(allocated(cublasHandleArray))) then
        allocate(cublasHandleArray(0:maxThreads-1))
        allocate(gpublasHandleArray(0:maxThreads-1))
        do thread=0, maxThreads-1
          cublasHandleArray(thread) = -1
          gpublasHandleArray(thread) = -1
        enddo
      endif
#endif
#ifdef WITH_NVIDIA_GPU_VERSION
#ifdef WITH_NVIDIA_CUSOLVER
      if (.not.(allocated(cusolverHandleArray))) then
        allocate(cusolverHandleArray(0:maxThreads-1))
        do thread=0, maxThreads-1
          cusolverHandleArray(thread) = -1
        enddo
      endif
#endif
#endif
#ifdef WITH_AMD_GPU_VERSION
      if (.not.(allocated(rocblasHandleArray))) then
        allocate(rocblasHandleArray(0:maxThreads-1))
        allocate(gpublasHandleArray(0:maxThreads-1))
        do thread=0, maxThreads-1
          rocblasHandleArray(thread) = -1
          gpublasHandleArray(thread) = -1
        enddo
      endif
#endif
#ifdef WITH_AMD_GPU_VERSION
#ifdef WITH_AMD_ROCSOLVER
      if (.not.(allocated(rocsolverHandleArray))) then
        allocate(rocsolverHandleArray(0:maxThreads-1))
        do thread=0, maxThreads-1
          rocsolverHandleArray(thread) = -1
        enddo
      endif
#endif
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (.not.(allocated(openmpOffloadHandleArray))) then
        allocate(openmpOffloadHandleArray(0:maxThreads-1))
        allocate(gpublasHandleArray(0:maxThreads-1))
        do thread=0, maxThreads-1
          openmpOffloadHandleArray(thread) = -1
          gpublasHandleArray(thread) = -1
        enddo
      endif
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      if (.not.(allocated(openmpOffloadsolverHandleArray))) then
        allocate(openmpOffloadsolverHandleArray(0:maxThreads-1))
        do thread=0, maxThreads-1
          openmpOffloadsolverHandleArray(thread) = -1
        enddo
      endif
#endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (.not.(allocated(syclHandleArray))) then
        allocate(syclHandleArray(0:maxThreads-1))
        allocate(gpublasHandleArray(0:maxThreads-1))
        do thread=0, maxThreads-1
          syclHandleArray(thread) = -1
          gpublasHandleArray(thread) = -1
        enddo
      endif
#endif
#ifdef WITH_SYCL_GPU_VERSION
#ifdef WITH_SYCL_SOLVER
      if (.not.(allocated(syclsolverHandleArray))) then
        allocate(syclsolverHandleArray(0:maxThreads-1))
        do thread=0, maxThreads-1
          syclsolverHandleArray(thread) = -1
        enddo
      endif
#endif
#endif

      if (obj%is_set("use_gpu_id") == 1) then ! useGPUid
        call obj%get("use_gpu_id", use_gpu_id, error)
        if (error .ne. ELPA_OK) then
          print *,"check_for_gpu: cannot querry use_gpu_id. Aborting..."
          stop
        endif

        if (use_gpu_id == -99) then
          print *,"Problem you did not set which gpu id this task should use"
        endif
 
        ! check whether gpu ud has been set for each proces
#ifdef WITH_MPI
        call mpi_allreduce(use_gpu_id, min_use_gpu_id, 1, MPI_INTEGER, MPI_MAX, mpi_comm_all, mpierr)

        if (min_use_gpu_id .lt. 0) then
          print *,"Not all tasks have set which GPU id should be used"
          print *,"GPUs will NOT be used!"
          gpuAvailable = .false.
          return
        endif
#endif
        gpuAvailable = .true.

        !if (myid==0) then
        !  if (wantDebugMessage) then
        !    print *
        !    print '(3(a,i0))','Found ', numberOfDevices, ' GPUs'
        !  endif
        !endif

        success = .true.
        if (.not.(alreadySET)) then
          deviceNumber = use_gpu_id
#include "./device_arrays_template.F90"

#include "./handle_creation_template.F90"

        endif ! alreadySET
        alreadySET = .true.
        !gpuIsInitialized = .true.

      else ! useGPUid

        !TODO: have to set this somewhere
        !if (gpuIsInitialized) then
        !  gpuAvailable = .true.
        !  numberOfDevices = -1
        !  if (myid == 0 .and. wantDebugMessage) then
        !    print *, "Skipping GPU init, should have already been initialized "
        !  endif
        !  return
        !else
          if (myid == 0 .and. wantDebugMessage) then
            print *, "Initializing the GPU devices"
          endif
        !endif

        success = .true.
#ifdef WITH_NVIDIA_GPU_VERSION
        ! call getenv("CUDA_PROXY_PIPE_DIRECTORY", envname)
        success = cuda_getdevicecount(numberOfDevices)
#endif
#ifdef WITH_AMD_GPU_VERSION
        success = hip_getdevicecount(numberOfDevices)
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        numberOfDevices = openmp_offload_getdevicecount()
        success = .true.
#endif
#ifdef WITH_SYCL_GPU_VERSION
        numberOfDevices = sycl_getdevicecount()
        success = .true.
#endif
        if (.not.(success)) then
#ifdef WITH_NVIDIA_GPU_VERSION
          print *,"error in cuda_getdevicecount"
#endif
#ifdef WITH_AMD_GPU_VERSION
          print *,"error in hip_getdevicecount"
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
          print *,"error in openmp_offload_getdevicecount"
#endif
#ifdef WITH_SYCL_GPU_VERSION
          print *,"error in sycl_getdevicecount"
#endif
          stop 1
        endif

!#ifdef  WITH_INTEL_GPU_VERSION
!        gpuAvailable = .false.
!        numberOfDevices = -1
!
!        numberOfDevices = 1
!        print *,"Manually setting",numberOfDevices," of GPUs"
!        if (numberOfDevices .ge. 1) then
!          gpuAvailable = .true.
!        endif
!#endif


        ! make sure that all nodes have the same number of GPU's, otherwise
        ! we run into loadbalancing trouble
#ifdef WITH_MPI
        call mpi_allreduce(numberOfDevices, maxNumberOfDevices, 1, MPI_INTEGER, MPI_MAX, mpi_comm_all, mpierr)

        if (maxNumberOfDevices .ne. numberOfDevices) then
          print *,"Different number of GPU devices on MPI tasks!"
          print *,"GPUs will NOT be used!"
          gpuAvailable = .false.
          return
        endif
#endif
        if (numberOfDevices .ne. 0) then
          gpuAvailable = .true.
          ! Usage of GPU is possible since devices have been detected

          if (myid==0) then
            if (wantDebugMessage) then
              print *
              print '(3(a,i0))','Found ', numberOfDevices, ' GPUs'
            endif
          endif

          fmt = '(I5.5)'

          write (gpu_string,fmt) numberOfDevices

          call obj%timer%start("check_gpu_"//gpu_string)
          deviceNumber = mod(myid, numberOfDevices)
          call obj%timer%start("set_device")

          !include device arrays here
#include "./device_arrays_template.F90"

          call obj%timer%stop("set_device")

          call obj%set("use_gpu_id",deviceNumber, error)
          if (error .ne. ELPA_OK) then
            print *,"Cannot set use_gpu_id. Aborting..."
            stop
          endif
 
#include "./handle_creation_template.F90"
          
          alreadySET = .true.
          call obj%timer%stop("check_gpu_"//gpu_string)
        endif ! numberOfDevices .ne. 0
        !gpuIsInitialized = .true.
      endif ! useGPUid
    end function
end module
