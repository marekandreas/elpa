#ifdef WITH_NVIDIA_GPU_VERSION
          if (.not.(allocated(cudaDeviceArray))) then
            allocate(cudaDeviceArray(0:maxThreads-1))
            allocate(gpuDeviceArray(0:maxThreads-1))
            success = cuda_setdevice(deviceNumber)
            do thread=0,maxThreads-1
              cudaDeviceArray(thread) = deviceNumber
              gpuDeviceArray(thread) = deviceNumber
            enddo
          endif
#endif
#ifdef WITH_AMD_GPU_VERSION
          if (.not.(allocated(hipDeviceArray))) then
            allocate(hipDeviceArray(0:maxThreads-1))
            allocate(gpuDeviceArray(0:maxThreads-1))
            success = hip_setdevice(deviceNumber)
            do thread=0,maxThreads-1
              hipDeviceArray(thread) = deviceNumber
              gpuDeviceArray(thread) = deviceNumber
            enddo
          endif
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
          if (.not.(allocated(openmpOffloadDeviceArray))) then
            allocate(openmpOffloadDeviceArray(0:maxThreads-1))
            allocate(gpuDeviceArray(0:maxThreads-1))
            success = openmp_offload_setdevice(deviceNumber)
            do thread=0,maxThreads-1
              openmpOffloadDeviceArray(thread) = deviceNumber
              gpuDeviceArray(thread) = deviceNumber
            enddo
          endif
#endif
#ifdef WITH_SYCL_GPU_VERSION
          if (.not.(allocated(syclDeviceArray))) then
            allocate(syclDeviceArray(0:maxThreads-1))
            allocate(gpuDeviceArray(0:maxThreads-1))
            !success = sycl_setdevice(deviceNumber)
            success = sycl_setdevice(0)
            do thread=0,maxThreads-1
              syclDeviceArray(thread) = deviceNumber
              gpuDeviceArray(thread) = deviceNumber
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
