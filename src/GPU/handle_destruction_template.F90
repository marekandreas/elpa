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

#ifdef WITH_GPU_STREAMS
          success = .true.
#ifdef WITH_NVIDIA_GPU_VERSION
          success = cuda_stream_destroy(self%gpu_setup%my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
          success = hip_stream_destroy(self%gpu_setup%my_stream)
#endif
          if (.not.(success)) then
#ifdef WITH_NVIDIA_GPU_VERSION
            print *,"Cannot destroy cuda stream handle"
#endif
#ifdef WITH_AMD_GPU_VERSION
            print *,"Cannot destroy hip stream handle"
#endif
            stop 1
          endif
#endif /* WITH_GPU_STREAMS */


          ! handle destruction
          do thread = 0, maxThreads-1
            success = .true.
#ifdef WITH_NVIDIA_GPU_VERSION
            success = cublas_destroy(self%gpu_setup%cublasHandleArray(thread))
#endif
#ifdef WITH_AMD_GPU_VERSION
            success = rocblas_destroy(self%gpu_setup%rocblasHandleArray(thread))
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
            success = openmp_offload_blas_destroy(self%gpu_setup%openmpOffloadHandleArray(thread))
#endif
#ifdef WITH_SYCL_GPU_VERSION
            success = syclblas_destroy(self%gpu_setup%syclHandleArray(thread))
#endif
            if (.not.(success)) then
#ifdef WITH_NVIDIA_GPU_VERSION
              print *,"Cannot destroy cublas handle"
#endif
#ifdef WITH_AMD_GPU_VERSION
              print *,"Cannot destroy rocblas handle"
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
              print *,"Cannot destroy openmpOffloadblas handle"
#endif
#ifdef WITH_SYCL_GPU_VERSION
              print *,"Cannot destroy syclblas handle"
#endif
              stop 1
            endif
          enddo
          

#ifdef WITH_NVIDIA_GPU_VERSION
#ifdef WITH_NVIDIA_CUSOLVER
          do thread=0, maxThreads-1
            success = cusolver_destroy(self%gpu_setup%cusolverHandleArray(thread))
            
            if (.not.(success)) then
              print *,"Cannotdestroy cusolver handle"
              stop 1
            endif 
          enddo
#endif
#endif

#ifdef WITH_AMD_GPU_VERSION
#ifdef WITH_AMD_ROCSOLVER
          !do thread=0, maxThreads-1
            !not needed
            !success = rocsolver_create(handle_tmp)
            !self%gpu_setup%rocsolverHandleArray(thread) = handle_tmp
            !if (.not.(success)) then
            !  print *,"Cannot create rocsolver handle"
            !  stop 1
            !endif
          !enddo
          !self%gpu_setup%rocsolverHandleArray(:) = self%gpu_setup%rocblasHandleArray(:)
#endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
          do thread=0, maxThreads-1
            success = openmp_offload_solver_destroy(self%gpu_setup%openmpOffloadsolverHandleArray(thread))
            
            if (.not.(success)) then
              print *,"Cannotdestroy openmpOffloadsolver handle"
              stop 1
            endif
          enddo
#endif
#endif
#ifdef WITH_SYCL_GPU_VERSION
#ifdef WITH_SYCL_SOLVER
          do thread=0, maxThreads-1
            success = sycl_solver_destroy(self%gpu_setup%syclsolverHandleArray(thread))
            
            if (.not.(success)) then
              print *,"Cannotdestroy syclsolver handle"
              stop 1
            endif 
          enddo
#endif
#endif
