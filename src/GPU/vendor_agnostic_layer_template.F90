#if 0
!    Copyright 2021, A. Marek, MPCDF
!
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
#endif


!#ifdef WITH_INTEL_GPU_VERSION
!  use mkl_offload
!#endif
  integer(kind=c_int), parameter :: nvidia_gpu = 1
  integer(kind=c_int), parameter :: amd_gpu = 2
  integer(kind=c_int), parameter :: intel_gpu = 3
  integer(kind=c_int), parameter :: openmp_offload_gpu = 4
  integer(kind=c_int), parameter :: sycl_gpu = 5
  integer(kind=c_int), parameter :: no_gpu = -1

  ! the following variables, as long as they are not stored in the ELPA object
  ! prohibit to run ELPA at the same time on different GPUs of different vendors!
  integer(kind=c_int)            :: use_gpu_vendor
  integer(kind=c_int)            :: gpuHostRegisterDefault    
  integer(kind=c_int)            :: gpuMemcpyHostToDevice    
  integer(kind=c_int)            :: gpuMemcpyDeviceToHost   
  integer(kind=c_int)            :: gpuMemcpyDeviceToDevice
  integer(kind=c_int)            :: gpuHostRegisterMapped
  integer(kind=c_int)            :: gpuHostRegisterPortable

  integer(kind=c_int)            :: gpublasPointerModeHost
  integer(kind=c_int)            :: gpublasPointerModeDevice
  integer(kind=c_int)            :: gpublasDefaultPointerMode

  !! per task information should be stored elsewhere
  !integer(kind=C_intptr_T), allocatable :: gpublasHandleArray(:)
  !integer(kind=c_int), allocatable      :: gpuDeviceArray(:)
  !integer(kind=c_intptr_t)              :: my_stream



  integer(kind=c_intptr_t), parameter :: size_of_double_real    = 8_rk8
#ifdef WANT_SINGLE_PRECISION_REAL
  integer(kind=c_intptr_t), parameter :: size_of_single_real    = 4_rk4
#endif

  integer(kind=c_intptr_t), parameter :: size_of_double_complex = 16_ck8
#ifdef WANT_SINGLE_PRECISION_COMPLEX
  integer(kind=c_intptr_t), parameter :: size_of_single_complex = 8_ck4
#endif

  interface gpu_memcpy
    module procedure gpu_memcpy_intptr
    module procedure gpu_memcpy_cptr
    module procedure gpu_memcpy_mixed_to_device
    module procedure gpu_memcpy_mixed_to_host
  end interface

  interface gpu_memcpy_async
    module procedure gpu_memcpy_async_intptr
    module procedure gpu_memcpy_async_cptr
    module procedure gpu_memcpy_async_mixed_to_device
    module procedure gpu_memcpy_async_mixed_to_host
  end interface

  interface gpu_memcpy2d
    module procedure gpu_memcpy2d_intptr
    module procedure gpu_memcpy2d_cptr
  end interface

  interface gpu_memcpy2d_async
    module procedure gpu_memcpy2d_async_intptr
    module procedure gpu_memcpy2d_async_cptr
  end interface

  interface gpu_malloc
    module procedure gpu_malloc_intptr
    module procedure gpu_malloc_cptr
  end interface

  interface gpu_free
    module procedure gpu_free_intptr
    module procedure gpu_free_cptr
  end interface

  interface gpublas_Dcopy
    module procedure gpublas_Dcopy_intptr
    module procedure gpublas_Dcopy_cptr
  end interface

  interface gpublas_Dtrmm
    module procedure gpublas_Dtrmm_intptr
    module procedure gpublas_Dtrmm_cptr
  end interface
  
  interface gpublas_Dtrsm
    module procedure gpublas_Dtrsm_intptr
    module procedure gpublas_Dtrsm_cptr
  end interface
  
  interface gpublas_Dgemm
    module procedure gpublas_Dgemm_intptr
    module procedure gpublas_Dgemm_cptr
  end interface

  interface gpublas_Dscal
    module procedure gpublas_Dscal_intptr
    module procedure gpublas_Dscal_cptr
  end interface

  interface gpublas_Ddot
    module procedure gpublas_Ddot_intptr
    module procedure gpublas_Ddot_cptr
  end interface

  interface gpublas_Daxpy
    module procedure gpublas_Daxpy_intptr
    module procedure gpublas_Daxpy_cptr
  end interface

  interface gpublas_Scopy
    module procedure gpublas_Scopy_intptr
    module procedure gpublas_Scopy_cptr
  end interface

  interface gpublas_Strmm
    module procedure gpublas_Strmm_intptr
    module procedure gpublas_Strmm_cptr
  end interface
  
  interface gpublas_Strsm
    module procedure gpublas_Strsm_intptr
    module procedure gpublas_Strsm_cptr
  end interface
  
  interface gpublas_Sgemm
    module procedure gpublas_Sgemm_intptr
    module procedure gpublas_Sgemm_cptr
  end interface

  interface gpublas_Sscal
    module procedure gpublas_Sscal_intptr
    module procedure gpublas_Sscal_cptr
  end interface

  interface gpublas_Sdot
    module procedure gpublas_Sdot_intptr
    module procedure gpublas_Sdot_cptr
  end interface

  interface gpublas_Saxpy
    module procedure gpublas_Saxpy_intptr
    module procedure gpublas_Saxpy_cptr
  end interface

  interface gpublas_Zcopy
    module procedure gpublas_Zcopy_intptr
    module procedure gpublas_Zcopy_cptr
  end interface

  interface gpublas_Ztrmm
    module procedure gpublas_Ztrmm_intptr
    module procedure gpublas_Ztrmm_cptr
  end interface
  
  interface gpublas_Ztrsm
    module procedure gpublas_Ztrsm_intptr
    module procedure gpublas_Ztrsm_cptr
  end interface
  
  interface gpublas_Zgemm
    module procedure gpublas_Zgemm_intptr
    module procedure gpublas_Zgemm_cptr
  end interface

  interface gpublas_Zscal
    module procedure gpublas_Zscal_intptr
    module procedure gpublas_Zscal_cptr
  end interface

  interface gpublas_Zdot
    module procedure gpublas_Zdot_intptr
    module procedure gpublas_Zdot_cptr
  end interface

  interface gpublas_Zaxpy
    module procedure gpublas_Zaxpy_intptr
    module procedure gpublas_Zaxpy_cptr
  end interface

  interface gpublas_Ccopy
    module procedure gpublas_Ccopy_intptr
    module procedure gpublas_Ccopy_cptr
  end interface

  interface gpublas_Ctrmm
    module procedure gpublas_Ctrmm_intptr
    module procedure gpublas_Ctrmm_cptr
  end interface
  
  interface gpublas_Ctrsm
    module procedure gpublas_Ctrsm_intptr
    module procedure gpublas_Ctrsm_cptr
  end interface
  
  interface gpublas_Cgemm
    module procedure gpublas_Cgemm_intptr
    module procedure gpublas_Cgemm_cptr
  end interface

  interface gpublas_Cscal
    module procedure gpublas_Cscal_intptr
    module procedure gpublas_Cscal_cptr
  end interface

  interface gpublas_Cdot
    module procedure gpublas_Cdot_intptr
    module procedure gpublas_Cdot_cptr
  end interface

  interface gpublas_Caxpy
    module procedure gpublas_Caxpy_intptr
    module procedure gpublas_Caxpy_cptr
  end interface

  interface gpu_vendor
    module procedure gpu_vendor_internal
    module procedure gpu_vendor_external_tests
  end interface

  contains
    function gpu_vendor_internal() result(vendor)
      implicit none
      integer(kind=c_int) :: vendor
      ! default
      vendor = no_gpu
#ifdef WITH_NVIDIA_GPU_VERSION
      vendor = nvidia_gpu
#endif
#ifdef WITH_AMD_GPU_VERSION
      vendor = amd_gpu
#endif
!#ifdef WITH_INTEL_GPU_VERSION
!      vendor = intel_gpu
!#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      vendor = openmp_offload_gpu
#endif
#ifdef WITH_SYCL_GPU_VERSION
      vendor = sycl_gpu
#endif
      use_gpu_vendor = vendor
      return
    end function

    function gpu_vendor_external_tests(set_vendor) result(vendor)
      implicit none
      integer(kind=c_int)             :: vendor
      integer(kind=c_int), intent(in) :: set_vendor
      ! default
      vendor = no_gpu
      if (set_vendor == nvidia_gpu) then
        vendor = nvidia_gpu
      endif
      if (set_vendor == amd_gpu) then
        vendor = amd_gpu
      endif
      if (vendor == no_gpu) then
        print *,"setting gpu vendor in tests does not work"
        stop
      endif
!#if TEST_INTEL_GPU == 1
!      vendor = intel_gpu
!#endif
      use_gpu_vendor = vendor
      return
    end function

    subroutine set_gpu_parameters
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif
!#ifdef WITH_NVIDIA_NCCL
!      use nccl_functions
!#endif
      use elpa_ccl_gpu
      implicit none

#ifdef WITH_NVIDIA_GPU_VERSION
      if (use_gpu_vendor == nvidia_gpu) then
        cudaMemcpyHostToDevice   = cuda_memcpyHostToDevice()
        gpuMemcpyHostToDevice    = cudaMemcpyHostToDevice
        cudaMemcpyDeviceToHost   = cuda_memcpyDeviceToHost()
        gpuMemcpyDeviceToHost    = cudaMemcpyDeviceToHost
        cudaMemcpyDeviceToDevice = cuda_memcpyDeviceToDevice()
        gpuMemcpyDeviceToDevice  = cudaMemcpyDeviceToDevice
        cudaHostRegisterPortable = cuda_hostRegisterPortable()
        gpuHostRegisterPortable  = cudaHostRegisterPortable
        cudaHostRegisterMapped   = cuda_hostRegisterMapped()
        gpuHostRegisterMapped    = cudaHostRegisterMapped
        cudaHostRegisterDefault  = cuda_hostRegisterDefault()
        gpuHostRegisterDefault   = cudaHostRegisterDefault

        cublasPointerModeDevice  = cublas_PointerModeDevice()
        gpublasPointerModeDevice = cublasPointerModeDevice
        cublasPointerModeHost    = cublas_PointerModeHost()
        gpublasPointerModeHost   = cublasPointerModeHost
      endif
#endif

#ifdef WITH_NVIDIA_NCCL
      if (use_gpu_vendor == nvidia_gpu) then
        ncclSum = nccl_redOp_ncclSum()
        ncclMax = nccl_redOp_ncclMax()
        ncclMin = nccl_redOp_ncclMin()
        ncclAvg = nccl_redOp_ncclAvg()
        ncclProd = nccl_redOp_ncclProd()

        ncclInt = nccl_dataType_ncclInt()
        ncclInt32 = nccl_dataType_ncclInt32()
        ncclInt64 = nccl_dataType_ncclInt64()
        ncclFloat = nccl_dataType_ncclFloat()
        ncclFloat32 = nccl_dataType_ncclFloat32()
        ncclFloat64 = nccl_dataType_ncclFloat64()
        ncclDouble = nccl_dataType_ncclDouble()
      endif
#endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        hipMemcpyHostToDevice   = hip_memcpyHostToDevice()
        gpuMemcpyHostToDevice   = hipMemcpyHostToDevice
        hipMemcpyDeviceToHost   = hip_memcpyDeviceToHost()
        gpuMemcpyDeviceToHost   = hipMemcpyDeviceToHost
        hipMemcpyDeviceToDevice = hip_memcpyDeviceToDevice()
        gpuMemcpyDeviceToDevice = hipMemcpyDeviceToDevice
        hipHostRegisterPortable = hip_hostRegisterPortable()
        gpuHostRegisterPortable = hipHostRegisterPortable
        hipHostRegisterMapped   = hip_hostRegisterMapped()
        gpuHostRegisterMapped   = hipHostRegisterMapped
        hipHostRegisterDefault  = hip_hostRegisterDefault()
        gpuHostRegisterDefault  = hipHostRegisterDefault
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        openmpOffloadMemcpyHostToDevice   = openmp_offload_memcpyHostToDevice()
        gpuMemcpyHostToDevice             = openmpOffloadMemcpyHostToDevice
        openmpOffloadMemcpyDeviceToHost   = openmp_offload_memcpyDeviceToHost()
        gpuMemcpyDeviceToHost             = openmpOffloadMemcpyDeviceToHost
        openmpOffloadMemcpyDeviceToDevice = openmp_offload_memcpyDeviceToDevice()
        gpuMemcpyDeviceToDevice           = openmpOffloadMemcpyDeviceToDevice
        !openmpOffloadHostRegisterPortable = openmp_offload_hostRegisterPortable()
        !gpuHostRegisterPortable           = openmpOffloadHostRegisterPortable
        !openmpOffloadHostRegisterMapped   = openmp_offload_hostRegisterMapped()
        !gpuHostRegisterMapped             = openmpOffloadHostRegisterMapped
        !openmpOffloadHostRegisterDefault  = openmp_offload_hostRegisterDefault()
        !gpuHostRegisterDefault            = openmpOffloadHostRegisterDefault
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        syclMemcpyHostToDevice   = sycl_MemcpyHostToDevice()
        gpuMemcpyHostToDevice    = syclMemcpyHostToDevice
        syclMemcpyDeviceToHost   = sycl_MemcpyDeviceToHost()
        gpuMemcpyDeviceToHost    = syclMemcpyDeviceToHost
        syclMemcpyDeviceToDevice = sycl_MemcpyDeviceToDevice()
        gpuMemcpyDeviceToDevice  = syclMemcpyDeviceToDevice
        !openmpOffloadHostRegisterPortable = openmp_offload_hostRegisterPortable()
        !gpuHostRegisterPortable           = openmpOffloadHostRegisterPortable
        !openmpOffloadHostRegisterMapped   = openmp_offload_hostRegisterMapped()
        !gpuHostRegisterMapped             = openmpOffloadHostRegisterMapped
        !openmpOffloadHostRegisterDefault  = openmp_offload_hostRegisterDefault()
        !gpuHostRegisterDefault            = openmpOffloadHostRegisterDefault
      endif
#endif
    end subroutine

    function gpublas_set_stream(handle, stream) result(success)
      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none

      integer(kind=c_intptr_t), intent(in)  :: handle
      integer(kind=c_intptr_t), intent(in)  :: stream
      logical                               :: success


      success = .true.
#ifdef WITH_NVIDIA_GPU_VERSION
#ifdef WITH_GPU_STREAMS
      if (use_gpu_vendor == nvidia_gpu) then
        success = cublas_set_stream(handle, stream)
      endif
#endif
#endif
#ifdef WITH_AMD_GPU_VERSION
#ifdef WITH_GPU_STREAMS
      if (use_gpu_vendor == amd_gpu) then
        success = rocblas_set_stream(handle, stream)
      endif
#endif
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"gpublasSetStream not implemented for openmp offload"
        stop
      endif
#endif
#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"gpublasSetStream not implemented for sycl"
        stop
      endif
#endif

    end function

    function gpu_stream_synchronize(stream) result(success)
      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none

      integer(kind=c_intptr_t), intent(in), optional  :: stream
      logical                                         :: success

      success = .true.
      if (present(stream)) then
#ifdef WITH_NVIDIA_GPU_VERSION
#ifdef WITH_GPU_STREAMS
        if (use_gpu_vendor == nvidia_gpu) then
          success = cuda_stream_synchronize(stream)
        endif
#endif
#endif
#ifdef WITH_AMD_GPU_VERSION
#ifdef WITH_GPU_STREAMS
        if (use_gpu_vendor == amd_gpu) then
          success = hip_stream_synchronize(stream)
        endif
#endif
#endif
      else
#ifdef WITH_NVIDIA_GPU_VERSION
#ifdef WITH_GPU_STREAMS
        if (use_gpu_vendor == nvidia_gpu) then
          success = cuda_stream_synchronize()
        endif
#endif
#endif
#ifdef WITH_AMD_GPU_VERSION
#ifdef WITH_GPU_STREAMS
        if (use_gpu_vendor == amd_gpu) then
          success = hip_stream_synchronize()
        endif
#endif
#endif
      endif


#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"gpu_stream_syncronize not implemented for openmp offload"
        stop
      endif
#endif
#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"gpu_stream_synchronize not implemented for sycl"
        stop
      endif
#endif

    end function

    function gpu_setdevice(n) result(success)
      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none

      integer(kind=ik), intent(in)  :: n
      logical                       :: success

#ifdef WITH_NVIDIA_GPU_VERSION
      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_setdevice(n)
      endif
#endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_setdevice(n)
      endif
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        success = openmp_offload_setdevice(n)
      endif
#endif
#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        success = sycl_setdevice(n)
      endif
#endif
    end function

    function gpu_devicesynchronize() result(success)
      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
      implicit none
      logical                              :: success

      success = .false.

#ifdef WITH_NVIDIA_GPU_VERSION
      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_devicesynchronize()
      endif
#endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_devicesynchronize()
      endif
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"not yet implemented: device synchronize"
        stop
      endif
#endif
#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: device synchronize"
        stop
      endif
#endif
    end function

    function gpu_malloc_host(array, elements) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions

#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      type(c_ptr)                          :: array
      integer(kind=c_intptr_t), intent(in) :: elements
      logical                              :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_malloc_host(array, elements)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_malloc_host(array, elements)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"not yet implemented: malloc_host"
        stop
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: malloc_host"
        stop
      endif
#endif

    end function

    function gpu_malloc_intptr(array, elements) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif
      implicit none
      integer(kind=C_intptr_T)             :: array
      integer(kind=c_intptr_t), intent(in) :: elements
      logical                              :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_malloc_intptr(array, elements)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_malloc_intptr(array, elements)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        success = openmp_offload_malloc_intptr(array, elements)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        success = sycl_malloc_intptr(array, elements)
      endif
#endif

    end function

    function gpu_malloc_cptr(array, elements) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif
      implicit none
      type(c_ptr)                          :: array
      integer(kind=c_intptr_t), intent(in) :: elements
      logical                              :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_malloc_cptr(array, elements)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_malloc_cptr(array, elements)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        success = openmp_offload_malloc_cptr(array, elements)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        success = sycl_malloc_cptr(array, elements)
      endif
#endif

    end function

    function gpu_host_register(array, elements, flag) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif
      implicit none
      integer(kind=C_intptr_t)              :: array
      integer(kind=c_intptr_t), intent(in)  :: elements
      integer(kind=C_INT), intent(in)       :: flag
      logical :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_host_register(array, elements, flag)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_host_register(array, elements, flag)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"not yet implemented: host_register"
        stop
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: host_register"
        stop
      endif
#endif

    end function
    
    function gpu_memcpy_intptr(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_intptr_t)              :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical                               :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memcpy_intptr(dst, src, size, dir)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_memcpy_intptr(dst, src, size, dir)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        success = openmp_offload_memcpy_intptr(dst, src, size, dir)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        success = sycl_memcpy_intptr(dst, src, size, dir)
      endif
#endif
      return
    
    end function
    
    function gpu_memcpy_cptr(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif
      implicit none
      type(c_ptr)                           :: dst
      type(c_ptr)                           :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical                               :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memcpy_cptr(dst, src, size, dir)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_memcpy_cptr(dst, src, size, dir)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        success = openmp_offload_memcpy_cptr(dst, src, size, dir)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        success = sycl_memcpy_cptr(dst, src, size, dir)
      endif
#endif
      return
    
    end function
    
    function gpu_memcpy_mixed_to_device(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      type(c_ptr)                           :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memcpy_mixed_to_device(dst, src, size, dir)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_memcpy_mixed_to_device(dst, src, size, dir)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        success = openmp_offload_memcpy_mixed_to_device(dst, src, size, dir)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        success = sycl_memcpy_mixed_to_device(dst, src, size, dir)
      endif
#endif
    
    end function
    
    function gpu_memcpy_mixed_to_host(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      type(c_ptr)                           :: src
      integer(kind=C_intptr_t)              :: dst
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memcpy_mixed_to_host(dst, src, size, dir)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_memcpy_mixed_to_host(dst, src, size, dir)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        success = openmp_offload_memcpy_mixed_to_host(dst, src, size, dir)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        success = sycl_memcpy_mixed_to_host(dst, src, size, dir)
      endif
#endif
    
    end function

    function gpu_memcpy_async_intptr(dst, src, size, dir, stream) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_intptr_t)              :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      integer(kind=c_intptr_t), intent(in)  :: stream
      logical                               :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memcpy_async_intptr(dst, src, size, dir, stream)
      endif
  
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_memcpy_async_intptr(dst, src, size, dir, stream)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"MemcpyAsync not implemented for openmp offload"
        stop
        !success = openmp_offload_memcpy_intptr(dst, src, size, dir)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"MemcpyAsync not implemented for sycl"
        stop
        !success = sycl_memcpy_intptr(dst, src, size, dir)
      endif
#endif
      return
    
    end function
    
    function gpu_memcpy_async_cptr(dst, src, size, dir, stream) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif
      implicit none
      type(c_ptr)                           :: dst
      type(c_ptr)                           :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      integer(kind=c_intptr_t), intent(in)  :: stream
      logical                               :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memcpy_async_cptr(dst, src, size, dir, stream)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_memcpy_async_cptr(dst, src, size, dir, stream)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"MemcpyAsync not implemented for openmp offload"
        stop
        !success = openmp_offload_memcpy_cptr(dst, src, size, dir)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"MemcpyAsync not implemented for sycl"
        stop
        !success = sycl_memcpy_cptr(dst, src, size, dir)
      endif
#endif
      return
    
    end function
    
    function gpu_memcpy_async_mixed_to_device(dst, src, size, dir, stream) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      type(c_ptr)                           :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=c_intptr_t), intent(in)  :: stream
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memcpy_async_mixed_to_device(dst, src, size, dir, stream)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_memcpy_async_mixed_to_device(dst, src, size, dir, stream)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"MemcpyAsync not implemented for openmp offload"
        stop
        !success = openmp_offload_memcpy_mixed_to_device(dst, src, size, dir)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"MemcpyAsync not implemented for sycl"
        stop
        !success = sycl_memcpy_mixed_to_device(dst, src, size, dir)
      endif
#endif
    
    end function
    
    function gpu_memcpy_async_mixed_to_host(dst, src, size, dir, stream) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      type(c_ptr)                           :: src
      integer(kind=C_intptr_t)              :: dst
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=c_intptr_t), intent(in)  :: stream
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memcpy_async_mixed_to_host(dst, src, size, dir, stream)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_memcpy_async_mixed_to_host(dst, src, size, dir, stream)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"MemcpyAsync not implemented for openmp offload"
        stop
        !success = openmp_offload_memcpy_mixed_to_host(dst, src, size, dir)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"MemcpyAsync not implemented for sycl"
        !success = sycl_memcpy_mixed_to_host(dst, src, size, dir)
      endif
#endif
    
    end function

    function gpu_memset(a, val, size) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)             :: a
      integer(kind=ik)                     :: val
      integer(kind=c_intptr_t), intent(in) :: size
      integer(kind=C_INT)                  :: istat

      logical :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memset(a, val, size)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_memset(a, val, size)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        success = openmp_offload_memset(a, val, size)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        success = sycl_memset(a, int(val,kind=c_int32_t), size)
      endif
#endif

    end function

    function gpu_memset_async(a, val, size, stream) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)             :: a
      integer(kind=ik)                     :: val
      integer(kind=c_intptr_t), intent(in) :: size
      integer(kind=C_INT)                  :: istat
      integer(kind=c_intptr_t), intent(in) :: stream

      logical :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memset_async(a, val, size, stream)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_memset_async(a, val, size, stream)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        !success = openmp_offload_memset(a, val, size)
        print *,"Openmp Offload memset_async not yet implemented"
        stop
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        !success = sycl_memset(a, int(val,kind=c_int32_t), size)
        print *,"Sycl memset_async not yet implemented"
        stop
      endif
#endif

    end function

    function gpu_free_intptr(a) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)                :: a

      logical :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_free_intptr(a)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_free_intptr(a)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        success = openmp_offload_free_intptr(a)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        success = sycl_free_intptr(a)
      endif
#endif

    end function

    function gpu_free_cptr(a) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      type(c_ptr)              :: a

      logical :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_free_cptr(a)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_free_cptr(a)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        success = openmp_offload_free_cptr(a)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        success = sycl_free_cptr(a)
      endif
#endif

    end function

    function gpu_free_host(a) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      type(c_ptr), value          :: a

      logical :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_free_host(a)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_free_host(a)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"not yet implemented: host_free"
        stop
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: host_free"
        stop
      endif
#endif
    end function

    function gpu_host_unregister(a) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)                :: a

      logical :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_host_unregister(a)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_host_unregister(a)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"not yet implemented: host_unregister"
        stop
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: host_unregister"
        stop
      endif
#endif

    end function

    function gpu_memcpy2d_intptr(dst, dpitch, src, spitch, width, height , dir) result(success)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none

      integer(kind=C_intptr_T)           :: dst
      integer(kind=c_intptr_t), intent(in) :: dpitch
      integer(kind=C_intptr_T)           :: src
      integer(kind=c_intptr_t), intent(in) :: spitch
      integer(kind=c_intptr_t), intent(in) :: width
      integer(kind=c_intptr_t), intent(in) :: height
      integer(kind=C_INT), intent(in)    :: dir
      logical                            :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memcpy2d_intptr(dst, dpitch, src, spitch, width, height , dir)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_memcpy2d_intptr(dst, dpitch, src, spitch, width, height , dir)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"not yet implemented: memcpy2d_intptr"
        stop
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: memcpy2d_intptr"
        stop
      endif
#endif
    end function

    function gpu_memcpy2d_cptr(dst, dpitch, src, spitch, width, height , dir) result(success)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none

      type(c_ptr)         :: dst
      integer(kind=c_intptr_t), intent(in) :: dpitch
      type(c_ptr)           :: src
      integer(kind=c_intptr_t), intent(in) :: spitch
      integer(kind=c_intptr_t), intent(in) :: width
      integer(kind=c_intptr_t), intent(in) :: height
      integer(kind=C_INT), intent(in)    :: dir
      logical                            :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memcpy2d_cptr(dst, dpitch, src, spitch, width, height , dir)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_memcpy2d_cptr(dst, dpitch, src, spitch, width, height , dir)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"not yet implemented: memcpy2d_cptr"
        stop
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: memcpy2d_cptr"
        stop
      endif
#endif
    end function

    function gpu_memcpy2d_async_intptr(dst, dpitch, src, spitch, width, height, dir, stream) result(success)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none

      integer(kind=C_intptr_T)           :: dst
      integer(kind=c_intptr_t), intent(in) :: dpitch
      integer(kind=C_intptr_T)           :: src
      integer(kind=c_intptr_t), intent(in) :: spitch
      integer(kind=c_intptr_t), intent(in) :: width
      integer(kind=c_intptr_t), intent(in) :: height
      integer(kind=C_INT), intent(in)    :: dir
      integer(kind=c_intptr_t), intent(in) :: stream
      logical                            :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memcpy2d_async_intptr(dst, dpitch, src, spitch, width, height, dir, stream)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_memcpy2d_async_intptr(dst, dpitch, src, spitch, width, height, dir, stream)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"not yet implemented: memcpy2d_async_intptr"
        stop
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: memcpy2d_async_intptr"
        stop
      endif
#endif
    end function

    function gpu_memcpy2d_async_cptr(dst, dpitch, src, spitch, width, height, dir, stream) result(success)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none

      type(c_ptr)         :: dst
      integer(kind=c_intptr_t), intent(in) :: dpitch
      type(c_ptr)           :: src
      integer(kind=c_intptr_t), intent(in) :: spitch
      integer(kind=c_intptr_t), intent(in) :: width
      integer(kind=c_intptr_t), intent(in) :: height
      integer(kind=C_INT), intent(in)    :: dir
      integer(kind=c_intptr_t), intent(in) :: stream
      logical                            :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memcpy2d_async_cptr(dst, dpitch, src, spitch, width, height, dir, stream)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_memcpy2d_async_cptr(dst, dpitch, src, spitch, width, height , dir, stream)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"not yet implemented: memcpy2d_async_cptr"
        stop
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: memcpy2d_async-cptr"
        stop
      endif
#endif
    end function

    subroutine gpusolver_Dtrtri(uplo, diag, n, a, lda, info, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_intptr_t)        :: handle

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Dtrtri(uplo, diag, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_Dtrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_Dtrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
!      if (use_gpu_vendor == sycl_gpu) then
!        call mkl_sycl_Dtrtri(uplo, diag, n, a, lda, info, handle)
!      endif
#endif
    end subroutine

    subroutine gpusolver_Dpotrf(uplo, n, a, lda, info, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_intptr_t)        :: handle

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Dpotrf(uplo, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_Dpotrf(uplo, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_Dpotrf(uplo, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
!      if (use_gpu_vendor == sycl_gpu) then
!        call mkl_sycl_Dpotrf(uplo, n, a, lda, info, handle)
!      endif
#endif
    end subroutine

    subroutine gpublas_Dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_DOUBLE)             :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
         if (use_gpu_vendor == openmp_offload_gpu) then
           call mkl_openmp_offload_Dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
         if (use_gpu_vendor == sycl_gpu) then
           call mkl_sycl_Dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif
    end subroutine

    subroutine gpublas_Dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_DOUBLE)             :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_DOUBLE)             :: alpha,beta
      type(c_ptr)                     :: a, b, c
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Dcopy_intptr(n, x, incx, y, incy, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dcopy_intptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Dcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Dcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Dcopy_cptr(n, x, incx, y, incy, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dcopy_cptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Dcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Dcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE)             :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

    end subroutine


    subroutine gpublas_Dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE)             :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE)             :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE)             :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine
    subroutine gpusolver_Strtri(uplo, diag, n, a, lda, info, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_intptr_t)        :: handle

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Strtri(uplo, diag, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_Strtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_Strtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
!      if (use_gpu_vendor == sycl_gpu) then
!        call mkl_sycl_Strtri(uplo, diag, n, a, lda, info, handle)
!      endif
#endif
    end subroutine

    subroutine gpusolver_Spotrf(uplo, n, a, lda, info, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_intptr_t)        :: handle

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Spotrf(uplo, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_Spotrf(uplo, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_Spotrf(uplo, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
!      if (use_gpu_vendor == sycl_gpu) then
!        call mkl_sycl_Spotrf(uplo, n, a, lda, info, handle)
!      endif
#endif
    end subroutine

    subroutine gpublas_Sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_FLOAT)              :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
         if (use_gpu_vendor == openmp_offload_gpu) then
           call mkl_openmp_offload_Sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
         if (use_gpu_vendor == sycl_gpu) then
           call mkl_sycl_Sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif
    end subroutine

    subroutine gpublas_Sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_FLOAT)              :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_FLOAT)              :: alpha,beta
      type(c_ptr)                     :: a, b, c
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Scopy_intptr(n, x, incx, y, incy, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Scopy_intptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Scopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Scopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Scopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Scopy_cptr(n, x, incx, y, incy, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Scopy_cptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Scopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Scopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Scopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT)              :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

    end subroutine


    subroutine gpublas_Strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT)              :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT)              :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT)              :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine
    subroutine gpusolver_Ztrtri(uplo, diag, n, a, lda, info, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_intptr_t)        :: handle

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Ztrtri(uplo, diag, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_Ztrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_Ztrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
!      if (use_gpu_vendor == sycl_gpu) then
!        call mkl_sycl_Ztrtri(uplo, diag, n, a, lda, info, handle)
!      endif
#endif
    end subroutine

    subroutine gpusolver_Zpotrf(uplo, n, a, lda, info, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_intptr_t)        :: handle

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Zpotrf(uplo, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_Zpotrf(uplo, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_Zpotrf(uplo, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
!      if (use_gpu_vendor == sycl_gpu) then
!        call mkl_sycl_Zpotrf(uplo, n, a, lda, info, handle)
!      endif
#endif
    end subroutine

    subroutine gpublas_Zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_double_complex)  :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
         if (use_gpu_vendor == openmp_offload_gpu) then
           call mkl_openmp_offload_Zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
         if (use_gpu_vendor == sycl_gpu) then
           call mkl_sycl_Zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif
    end subroutine

    subroutine gpublas_Zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha,beta
      type(c_ptr)                     :: a, b, c
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Zcopy_intptr(n, x, incx, y, incy, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zcopy_intptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Zcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Zcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Zcopy_cptr(n, x, incx, y, incy, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zcopy_cptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Zcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Zcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

    end subroutine


    subroutine gpublas_Ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine
    subroutine gpusolver_Ctrtri(uplo, diag, n, a, lda, info, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_intptr_t)        :: handle

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Ctrtri(uplo, diag, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_Ctrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_Ctrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
!      if (use_gpu_vendor == sycl_gpu) then
!        call mkl_sycl_Ctrtri(uplo, diag, n, a, lda, info, handle)
!      endif
#endif
    end subroutine

    subroutine gpusolver_Cpotrf(uplo, n, a, lda, info, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_intptr_t)        :: handle

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Cpotrf(uplo, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_Cpotrf(uplo, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_Cpotrf(uplo, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
!      if (use_gpu_vendor == sycl_gpu) then
!        call mkl_sycl_Cpotrf(uplo, n, a, lda, info, handle)
!      endif
#endif
    end subroutine

    subroutine gpublas_Cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_float_complex)  :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
         if (use_gpu_vendor == openmp_offload_gpu) then
           call mkl_openmp_offload_Cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
         if (use_gpu_vendor == sycl_gpu) then
           call mkl_sycl_Cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif
    end subroutine

    subroutine gpublas_Cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX)  :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX)  :: alpha,beta
      type(c_ptr)                     :: a, b, c
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Ccopy_intptr(n, x, incx, y, incy, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ccopy_intptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ccopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ccopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Ccopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Ccopy_cptr(n, x, incx, y, incy, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ccopy_cptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ccopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ccopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Ccopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX)   :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

    end subroutine


    subroutine gpublas_Ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX)   :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX)   :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX)   :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_Ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_getPointerMode(gpublasHandle, mode)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: mode

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_getPointerMode(gpublasHandle, mode)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_getPointerMode(gpublasHandle, mode)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "getPointer mode not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "getPointer mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_setPointerMode(gpublasHandle, mode)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: mode

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_setPointerMode(gpublasHandle, mode)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_setPointerMode(gpublasHandle, mode)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "setPointer mode not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "setPointer mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Ddot_intptr(gpublasHandle, length, x, incx, y, incy, z)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      integer(kind=c_intptr_t)          :: x, y, z

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ddot_intptr(gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ddot_intptr(gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xdot not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xdot mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Ddot_cptr(gpublasHandle, length, x, incx, y, incy, z)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      type(c_ptr)                       :: x, y, z

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ddot_cptr(gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ddot_cptr(gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xdot not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xdot mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Dscal_intptr(gpublasHandle, length, alpha, x, incx)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx
      real(kind=C_DOUBLE)             :: alpha
      integer(kind=c_intptr_t)           :: x

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dscal_intptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dscal_intptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Dscal_cptr(gpublasHandle, length, alpha, x, incx)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx
      real(kind=C_DOUBLE)             :: alpha
      type(c_ptr)                       :: x

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dscal_cptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dscal_cptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine


    subroutine gpublas_Daxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      real(kind=C_DOUBLE)             :: alpha
      integer(kind=c_intptr_t)           :: x, y

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Daxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Daxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Daxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      real(kind=C_DOUBLE)             :: alpha
      type(c_ptr)                       :: x, y

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Daxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Daxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine
    subroutine gpublas_Sdot_intptr(gpublasHandle, length, x, incx, y, incy, z)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      integer(kind=c_intptr_t)          :: x, y, z

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Sdot_intptr(gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Sdot_intptr(gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xdot not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xdot mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Sdot_cptr(gpublasHandle, length, x, incx, y, incy, z)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      type(c_ptr)                       :: x, y, z

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Sdot_cptr(gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Sdot_cptr(gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xdot not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xdot mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Sscal_intptr(gpublasHandle, length, alpha, x, incx)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx
      real(kind=C_FLOAT)              :: alpha
      integer(kind=c_intptr_t)           :: x

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Sscal_intptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Sscal_intptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Sscal_cptr(gpublasHandle, length, alpha, x, incx)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx
      real(kind=C_FLOAT)              :: alpha
      type(c_ptr)                       :: x

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Sscal_cptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Sscal_cptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine


    subroutine gpublas_Saxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      real(kind=C_FLOAT)              :: alpha
      integer(kind=c_intptr_t)           :: x, y

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Saxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Saxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Saxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      real(kind=C_FLOAT)              :: alpha
      type(c_ptr)                       :: x, y

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Saxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Saxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine
    subroutine gpublas_Zdot_intptr(conj, gpublasHandle, length, x, incx, y, incy, z)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: conj
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      integer(kind=c_intptr_t)          :: x, y, z

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zdot_intptr(conj, gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zdot_intptr(conj, gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xdot not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xdot mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Zdot_cptr(conj, gpublasHandle, length, x, incx, y, incy, z)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: conj
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      type(c_ptr)                       :: x, y, z

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zdot_cptr(conj, gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zdot_cptr(conj, gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xdot not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xdot mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Zscal_intptr(gpublasHandle, length, alpha, x, incx)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      integer(kind=c_intptr_t)           :: x

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zscal_intptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zscal_intptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Zscal_cptr(gpublasHandle, length, alpha, x, incx)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      type(c_ptr)                       :: x

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zscal_cptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zscal_cptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine


    subroutine gpublas_Zaxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      integer(kind=c_intptr_t)           :: x, y

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zaxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zaxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Zaxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      type(c_ptr)                       :: x, y

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zaxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zaxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine
    subroutine gpublas_Cdot_intptr(conj, gpublasHandle, length, x, incx, y, incy, z)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: conj
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      integer(kind=c_intptr_t)          :: x, y, z

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Cdot_intptr(conj, gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Cdot_intptr(conj, gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xdot not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xdot mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Cdot_cptr(conj, gpublasHandle, length, x, incx, y, incy, z)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: conj
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      type(c_ptr)                       :: x, y, z

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Cdot_cptr(conj, gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Cdot_cptr(conj, gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xdot not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xdot mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Cscal_intptr(gpublasHandle, length, alpha, x, incx)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx
      complex(kind=C_FLOAT_COMPLEX)  :: alpha
      integer(kind=c_intptr_t)           :: x

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Cscal_intptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Cscal_intptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Cscal_cptr(gpublasHandle, length, alpha, x, incx)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx
      complex(kind=C_FLOAT_COMPLEX)  :: alpha
      type(c_ptr)                       :: x

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Cscal_cptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Cscal_cptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine


    subroutine gpublas_Caxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      complex(kind=C_FLOAT_COMPLEX)  :: alpha
      integer(kind=c_intptr_t)           :: x, y

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Caxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Caxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Caxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      complex(kind=C_FLOAT_COMPLEX)  :: alpha
      type(c_ptr)                       :: x, y

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Caxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Caxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine
