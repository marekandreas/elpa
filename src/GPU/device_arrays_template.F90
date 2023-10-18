#if 0
!    Copyright 2014-2023, A. Marek
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
#endif

#ifdef WITH_NVIDIA_GPU_VERSION
          if (.not.(allocated(OBJECT%gpu_setup%cudaDeviceArray))) then
            allocate(OBJECT%gpu_setup%cudaDeviceArray(0:maxThreads-1))
            allocate(OBJECT%gpu_setup%gpuDeviceArray(0:maxThreads-1))
            success = cuda_setdevice(deviceNumber)
            do thread=0,maxThreads-1
              OBJECT%gpu_setup%cudaDeviceArray(thread) = deviceNumber
              OBJECT%gpu_setup%gpuDeviceArray(thread) = deviceNumber
            enddo
          endif
#endif
#ifdef WITH_AMD_GPU_VERSION
          if (.not.(allocated(OBJECT%gpu_setup%hipDeviceArray))) then
            allocate(OBJECT%gpu_setup%hipDeviceArray(0:maxThreads-1))
            allocate(OBJECT%gpu_setup%gpuDeviceArray(0:maxThreads-1))
            success = hip_setdevice(deviceNumber)
            do thread=0,maxThreads-1
              OBJECT%gpu_setup%hipDeviceArray(thread) = deviceNumber
              OBJECT%gpu_setup%gpuDeviceArray(thread) = deviceNumber
            enddo
          endif
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
          if (.not.(allocated(OBJECT%gpu_setup%openmpOffloadDeviceArray))) then
            allocate(OBJECT%gpu_setup%openmpOffloadDeviceArray(0:maxThreads-1))
            allocate(OBJECT%gpu_setup%gpuDeviceArray(0:maxThreads-1))
            success = openmp_offload_setdevice(deviceNumber)
            do thread=0,maxThreads-1
              OBJECT%gpu_setup%openmpOffloadDeviceArray(thread) = deviceNumber
              OBJECT%gpu_setup%gpuDeviceArray(thread) = deviceNumber
            enddo
          endif
#endif
#ifdef WITH_SYCL_GPU_VERSION
          if (.not.(allocated(OBJECT%gpu_setup%syclDeviceArray))) then
            allocate(OBJECT%gpu_setup%syclDeviceArray(0:maxThreads-1))
            allocate(OBJECT%gpu_setup%gpuDeviceArray(0:maxThreads-1))
            !success = sycl_setdevice(deviceNumber)
            success = sycl_setdevice(0)
            do thread=0,maxThreads-1
              OBJECT%gpu_setup%syclDeviceArray(thread) = deviceNumber
              OBJECT%gpu_setup%gpuDeviceArray(thread) = deviceNumber
            enddo
          endif
#endif
          if (.not.(success)) then
#ifdef WITH_NVIDIA_GPU_VERSION
            print *,"Cannot set CudaDevice"
#endif
#ifdef WITH_AMD_GPU_VERSION
            print *,"Cannot set hipDevice"
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
            print *,"Cannot set openmpOffloadDevice"
#endif
#ifdef WITH_SYCL_GPU_VERSION
            print *,"Cannot set syclDevice"
#endif
            stop 1
          endif
          if (wantDebugMessage) then
            print '(3(a,i0))', 'MPI rank ', myid, ' uses GPU #', deviceNumber
          endif
