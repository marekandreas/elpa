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
!
! This file is the generated version. Do NOT edit
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

  interface gpu_malloc_host
    module procedure gpu_malloc_host_intptr
    module procedure gpu_malloc_host_cptr
  end interface

  interface gpu_free_host
    module procedure gpu_free_host_intptr
    module procedure gpu_free_host_cptr
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
        stop 1
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
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
      use elpa_ccl_gpu
#endif
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

#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
#ifdef WITH_NVIDIA_NCCL
      if (use_gpu_vendor == nvidia_gpu) then
#endif
#ifdef WITH_AMD_RCCL
      if (use_gpu_vendor == amd_gpu) then
#endif
        cclSum  = ccl_redOp_cclSum()
        cclMax  = ccl_redOp_cclMax()
        cclMin  = ccl_redOp_cclMin()
        cclAvg  = ccl_redOp_cclAvg()
        cclProd = ccl_redOp_cclProd()

        cclInt     = ccl_dataType_cclInt()
        cclInt32   = ccl_dataType_cclInt32()
        cclInt64   = ccl_dataType_cclInt64()
        cclFloat   = ccl_dataType_cclFloat()
        cclFloat32 = ccl_dataType_cclFloat32()
        cclFloat64 = ccl_dataType_cclFloat64()
        cclDouble  = ccl_dataType_cclDouble()
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


    function gpu_get_last_error() result(success)
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

      logical                                         :: success

      success = .true.

#ifdef WITH_NVIDIA_GPU_VERSION
      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_get_last_error()
      endif
#endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_get_last_error()
      endif
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"gpu_get_last_error not implemented for openmp offload"
        stop 1
      endif
#endif
#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"gpu_get_last_error not implemented for sycl"
        stop 1
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
        stop 1
      endif
#endif
#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"gpu_stream_synchronize not implemented for sycl"
        stop 1
      endif
#endif

    end function

    function gpu_getdevicecount(n) result(success)
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

      integer(kind=c_int)           :: n
      logical                       :: success

#ifdef WITH_NVIDIA_GPU_VERSION
      success = cuda_getdevicecount(n)
#endif
#ifdef WITH_AMD_GPU_VERSION
      success = hip_getdevicecount(n)
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      n = openmp_offload_getdevicecount()
      success = .true.
#endif
#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        !obj%gpu_setup%syclCPU=.false.
        !success = sycl_getdevicecount(numberOfDevices)
        success = .true.
        n=0
      endif
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

      integer(kind=c_int), intent(in) :: n
      logical                         :: success

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
        stop 1
      endif
#endif
#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: device synchronize"
        stop 1
      endif
#endif
    end function

    function gpu_malloc_host_intptr(array, elements) result(success)
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
      integer(kind=c_intptr_t)             :: array
      integer(kind=c_intptr_t), intent(in) :: elements
      logical                              :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_malloc_host_intptr(array, elements)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_malloc_host_intptr(array, elements)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"not yet implemented: malloc_host"
        stop 1
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: malloc_host"
        stop 1
      endif
#endif

    end function

    function gpu_malloc_host_cptr(array, elements) result(success)
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
        success = cuda_malloc_host_cptr(array, elements)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_malloc_host_cptr(array, elements)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"not yet implemented: malloc_host"
        stop 1
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: malloc_host"
        stop 1
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
        stop 1
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: host_register"
        stop 1
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
        stop 1
        !success = openmp_offload_memcpy_intptr(dst, src, size, dir)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"MemcpyAsync not implemented for sycl"
        stop 1
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
        stop 1
        !success = openmp_offload_memcpy_cptr(dst, src, size, dir)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"MemcpyAsync not implemented for sycl"
        stop 1
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
        stop 1
        !success = openmp_offload_memcpy_mixed_to_device(dst, src, size, dir)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"MemcpyAsync not implemented for sycl"
        stop 1
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
        stop 1
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
      integer(kind=c_int)                  :: val
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
      integer(kind=c_int)                  :: val
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
        stop 1
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        !success = sycl_memset(a, int(val,kind=c_int32_t), size)
        print *,"Sycl memset_async not yet implemented"
        stop 1
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

    function gpu_free_host_intptr(a) result(success)
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
      integer(kind=c_intptr_t), value          :: a

      logical :: success

      success = .false.

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_free_host_intptr(a)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_free_host_intptr(a)
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

    function gpu_free_host_cptr(a) result(success)
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
        success = cuda_free_host_cptr(a)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_free_host_cptr(a)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"not yet implemented: host_free"
        stop 1
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: host_free"
        stop 1
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
        stop 1
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: host_unregister"
        stop 1
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
        stop 1
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: memcpy2d_intptr"
        stop 1
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
        stop 1
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: memcpy2d_cptr"
        stop 1
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
        stop 1
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: memcpy2d_async_intptr"
        stop 1
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
        stop 1
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"not yet implemented: memcpy2d_async-cptr"
        stop 1
      endif
#endif
    end function


