#ifdef WITH_GPU_STREAMS
#ifdef WITH_NVIDIA_GPU_VERSION
          success = cuda_stream_create(my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
          success = hip_stream_create(my_stream)
#endif
          if (.not.(success)) then
#ifdef WITH_NVIDIA_GPU_VERSION
            print *,"Cannot create cuda stream handle"
#endif
#ifdef WITH_AMD_GPU_VERSION
            print *,"Cannot create hip stream handle"
#endif
            stop 1
          endif
#endif /* WITH_GPU_STREAMS */

          ! handle creation
          call obj%timer%start("create_handle")
          do thread = 0, maxThreads-1
#ifdef WITH_NVIDIA_GPU_VERSION
            !print *,"Creating handle for thread:",thread
            success = cublas_create(handle_tmp)
            cublasHandleArray(thread) = handle_tmp
            gpublasHandleArray(thread) = handle_tmp
#endif
#ifdef WITH_AMD_GPU_VERSION
            success = rocblas_create(handle_tmp)
            rocblasHandleArray(thread) = handle_tmp
            gpublasHandleArray(thread) = handle_tmp
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
            handle_tmp = 0
            ! not needed dummy call
            success = openmp_offload_blas_create(handle_tmp)
            openmpOffloadHandleArray(thread) = handle_tmp
            gpublasHandleArray(thread) = handle_tmp
#endif
#ifdef WITH_SYCL_GPU_VERSION
            handle_tmp = 0
            ! not needed dummy call
            success = sycl_blas_create(handle_tmp)
            syclHandleArray(thread) = handle_tmp
            gpublasHandleArray(thread) = handle_tmp
#endif
            if (.not.(success)) then
#ifdef WITH_NVIDIA_GPU_VERSION
              print *,"Cannot create cublas handle"
#endif
#ifdef WITH_AMD_GPU_VERSION
              print *,"Cannot create rocblas handle"
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
              print *,"Cannot create openmpOffloadblas handle"
#endif
#ifdef WITH_SYCL_GPU_VERSION
              print *,"Cannot create syclblas handle"
#endif
              stop 1
            endif
          enddo
          call obj%timer%stop("create_handle")
          call obj%timer%start("create_gpusolver")
#ifdef WITH_NVIDIA_GPU_VERSION
#ifdef WITH_NVIDIA_CUSOLVER
          do thread=0, maxThreads-1
            success = cusolver_create(handle_tmp)
            cusolverHandleArray(thread) = handle_tmp
            if (.not.(success)) then
              print *,"Cannot create cusolver handle"
              stop 1
            endif
          enddo
#endif
#endif
!#ifdef WITH_AMD_GPU_VERSION
!#ifdef WITH_AMD_CUSOLVER
!          do thread=0, maxThreads-1
!            success = cusolver_create(handle_tmp)
!            cusolverHandleArray(thread) = handle_tmp
!            if (.not.(success)) then
!              print *,"Cannot create cusolver handle"
!              stop 1
!            endif
!          enddo
!#endif
!#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
          do thread=0, maxThreads-1
            success = openmp_offload_solver_create(handle_tmp)
            openmpOffloadsolverHandleArray(thread) = handle_tmp
            if (.not.(success)) then
              print *,"Cannot create openmpOffloadsolver handle"
              stop 1
            endif
          enddo
#endif
#endif
#ifdef WITH_SYCL_GPU_VERSION
#ifdef WITH_SYCL_SOLVER
          do thread=0, maxThreads-1
            success = sycl_solver_create(handle_tmp)
            syclsolverHandleArray(thread) = handle_tmp
            if (.not.(success)) then
              print *,"Cannot create syclsolver handle"
              stop 1
            endif
          enddo
#endif
#endif
          call obj%timer%stop("create_gpusolver")

#ifdef WITH_GPU_STREAMS
          ! set stream
          do thread = 0, maxThreads-1
#ifdef WITH_NVIDIA_GPU_VERSION
            success = cublas_set_stream(cublasHandleArray(thread), my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
            success = rocblas_set_stream(rocblasHandleArray(thread), my_stream)
#endif
            if (.not.(success)) then
#ifdef WITH_NVIDIA_GPU_VERSION
              print *,"Cannot create cublas stream handle"
#endif
#ifdef WITH_AMD_GPU_VERSION
              print *,"Cannot create rocblas stream handle"
#endif
              stop 1
            endif
          enddo

#ifdef WITH_NVIDIA_GPU_VERSION
#ifdef WITH_NVIDIA_CUSOLVER
          do thread=0, maxThreads-1
            success = cusolver_set_stream(cusolverHandleArray(thread), my_stream)
            if (.not.(success)) then
              print *,"Cannot create cusolver stream handle"
              stop 1
            endif
          enddo
#endif
#endif


#endif /* WITH_GPU_STREAMS */

