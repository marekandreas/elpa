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

    function check_for_gpu(obj, myid, numberOfDevices, wantDebug) result(gpuAvailable)
      use cuda_functions
      use hip_functions
      use precision
      use elpa_mpi
      use elpa_abstract_impl
      implicit none

      class(elpa_abstract_impl_t), intent(inout)                         :: obj
      integer(kind=ik), intent(in)  :: myid
      logical, optional, intent(in) :: wantDebug
      logical                       :: success, wantDebugMessage
      integer(kind=ik), intent(out) :: numberOfDevices
      integer(kind=ik)              :: deviceNumber, mpierr, maxNumberOfDevices
      logical                       :: gpuAvailable
      integer(kind=ik)              :: error, mpi_comm_all, use_gpu_id, min_use_gpu_id
      !character(len=1024)           :: envname

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

      if (obj%is_set("use_gpu_id") == 1) then
        call obj%get("use_gpu_id", use_gpu_id, error)
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

        if (myid==0) then
          if (wantDebugMessage) then
            print *
            print '(3(a,i0))','Found ', numberOfDevices, ' GPUs'
          endif
        endif

        success = .true.
#ifdef WITH_NVIDIA_GPU_VERSION
        success = cuda_setdevice(use_gpu_id)
#endif
#ifdef WITH_AMD_GPU_VERSION
        success = hip_setdevice(use_gpu_id)
#endif
        if (.not.(success)) then
#ifdef WITH_NVIDIA_GPU_VERSION
          print *,"Cannot set CudaDevice"
#endif
#ifdef WITH_AMD_GPU_VERSION
          print *,"Cannot set HIPDevice"
#endif
          stop 1
        endif
        if (wantDebugMessage) then
          print '(3(a,i0))', 'MPI rank ', myid, ' uses GPU #', deviceNumber
        endif
 
        success = .true.        
#ifdef WITH_NVIDIA_GPU_VERSION
        success = cublas_create(cublasHandle)
#endif
#ifdef WITH_AMD_GPU_VERSION
        success = rocblas_create(rocblasHandle)
#endif
        if (.not.(success)) then
#ifdef WITH_NVIDIA_GPU_VERSION
          print *,"Cannot create cublas handle"
#endif
#ifdef WITH_AMD_GPU_VERSION
          print *,"Cannot create rocblas handle"
#endif
          stop 1
        endif

      else

#if defined(WITH_NVIDIA_GPU_VERSION) || !defined(WITH_AMD_GPU_VERSION)
        if (cublasHandle .ne. -1) then
#endif
#ifdef WITH_AMD_GPU_VERSION
        if (rocblasHandle .ne. -1) then
#endif
          gpuAvailable = .true.
          numberOfDevices = -1
          if (myid == 0 .and. wantDebugMessage) then
            print *, "Skipping GPU init, should have already been initialized "
          endif
          return
        else
          if (myid == 0 .and. wantDebugMessage) then
            print *, "Initializing the GPU devices"
          endif
        endif

        success = .true.
#ifdef WITH_NVIDIA_GPU_VERSION
        ! call getenv("CUDA_PROXY_PIPE_DIRECTORY", envname)
        success = cuda_getdevicecount(numberOfDevices)
#endif
#ifdef WITH_AMD_GPU_VERSION
        ! call getenv("CUDA_PROXY_PIPE_DIRECTORY", envname)
        success = hip_getdevicecount(numberOfDevices)
#endif
        if (.not.(success)) then
#ifdef WITH_NVIDIA_GPU_VERSION
          print *,"error in cuda_getdevicecount"
#endif
#ifdef WITH_AMPD_GPU_VERSION
          print *,"error in hip_getdevicecount"
#endif
          stop 1
        endif
#ifdef  WITH_INTEL_GPU_VERSION
      gpuAvailable = .false.
      numberOfDevices = -1

      numberOfDevices = 1
      print *,"Manually setting",numberOfDevices," of GPUs"
      if (numberOfDevices .ge. 1) then
        gpuAvailable = .true.
      endif
#endif
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

          deviceNumber = mod(myid, numberOfDevices)
#ifdef WITH_NIVDIA_GPU_VERSION
          success = cuda_setdevice(deviceNumber)
#endif
#ifdef WITH_AMD_GPU_VERSION
          success = hip_setdevice(deviceNumber)
#endif

          if (.not.(success)) then
#ifdef WITH_NIVDIA_GPU_VERSION
            print *,"Cannot set CudaDevice"
#endif
#ifdef WITH_AMD_GPU_VERSION
            print *,"Cannot set hipDevice"
#endif
            stop 1
          endif
          if (wantDebugMessage) then
            print '(3(a,i0))', 'MPI rank ', myid, ' uses GPU #', deviceNumber
          endif
          
#ifdef WITH_NIVDIA_GPU_VERSION
          success = cublas_create(cublasHandle)
#endif
#ifdef WITH_AMD_GPU_VERSION
          success = rocblas_create(rocblasHandle)
#endif
          if (.not.(success)) then
#ifdef WITH_NIVDIA_GPU_VERSION
            print *,"Cannot create cublas handle"
#endif
#ifdef WITH_AMD_GPU_VERSION
            print *,"Cannot create rocblas handle"
#endif
            stop 1
          endif
          
        endif

      endif  
    end function
end module
