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


#include "config-f90.h"
module elpa_gpu
  use precision
  use iso_c_binding
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

  interface gpublas_dcopy
    module procedure gpublas_dcopy_intptr
    module procedure gpublas_dcopy_cptr
  end interface

  interface gpublas_scopy
    module procedure gpublas_scopy_intptr
    module procedure gpublas_scopy_cptr
  end interface

  interface gpublas_zcopy
    module procedure gpublas_zcopy_intptr
    module procedure gpublas_zcopy_cptr
  end interface

  interface gpublas_ccopy
    module procedure gpublas_ccopy_intptr
    module procedure gpublas_ccopy_cptr
  end interface

  interface gpublas_dtrmm
    module procedure gpublas_dtrmm_intptr
    module procedure gpublas_dtrmm_cptr
  end interface
  
  interface gpublas_strmm
    module procedure gpublas_strmm_intptr
    module procedure gpublas_strmm_cptr
  end interface

  interface gpublas_ztrmm
    module procedure gpublas_ztrmm_intptr
    module procedure gpublas_ztrmm_cptr
  end interface

  interface gpublas_ctrmm
    module procedure gpublas_ctrmm_intptr
    module procedure gpublas_ctrmm_cptr
  end interface

  interface gpublas_dtrsm
    module procedure gpublas_dtrsm_intptr
    module procedure gpublas_dtrsm_cptr
  end interface
  
  interface gpublas_strsm
    module procedure gpublas_strsm_intptr
    module procedure gpublas_strsm_cptr
  end interface

  interface gpublas_ztrsm
    module procedure gpublas_ztrsm_intptr
    module procedure gpublas_ztrsm_cptr
  end interface

  interface gpublas_ctrsm
    module procedure gpublas_ctrsm_intptr
    module procedure gpublas_ctrsm_cptr
  end interface

  interface gpublas_dgemm
    module procedure gpublas_dgemm_intptr
    module procedure gpublas_dgemm_cptr
  end interface gpublas_dgemm

  interface gpublas_sgemm
    module procedure gpublas_sgemm_intptr
    module procedure gpublas_sgemm_cptr
  end interface gpublas_sgemm

  interface gpublas_zgemm
    module procedure gpublas_zgemm_intptr
    module procedure gpublas_zgemm_cptr
  end interface gpublas_zgemm

  interface gpublas_cgemm
    module procedure gpublas_cgemm_intptr
    module procedure gpublas_cgemm_cptr
  end interface gpublas_cgemm

  contains
    function gpu_vendor() result(vendor)
      use precision
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

    function gpu_malloc(array, elements) result(success)
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
        success = cuda_malloc(array, elements)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_malloc(array, elements)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        success = openmp_offload_malloc(array, elements)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        success = sycl_malloc(array, elements)
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




    function gpu_free(a) result(success)
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
        success = cuda_free(a)
      endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_free(a)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        success = openmp_offload_free(a)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        success = sycl_free(a)
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

    subroutine gpusolver_dtrtri(uplo, diag, n, a, lda, info, handle)
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
        call cusolver_dtrtri(uplo, diag, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_dtrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_dtrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        call mkl_sycl_dtrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif
    end subroutine

    subroutine gpusolver_strtri(uplo, diag, n, a, lda, info, handle)
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
        call cusolver_strtri(uplo, diag, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_strtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_strtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        call mkl_sycl_strtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif
    end subroutine

    subroutine gpusolver_ztrtri(uplo, diag, n, a, lda, info, handle)
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
        call cusolver_ztrtri(uplo, diag, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_ztrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_ztrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        call mkl_sycl_ztrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif
    end subroutine

    subroutine gpusolver_ctrtri(uplo, diag, n, a, lda, info, handle)
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
        call cusolver_ctrtri(uplo, diag, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_ctrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_ctrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        call mkl_sycl_ctrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif
    end subroutine

    subroutine gpusolver_dpotrf(uplo, n, a, lda, info, handle)
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
        call cusolver_dpotrf(uplo, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_dpotrf(uplo, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_dpotrf(uplo, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        call mkl_sycl_dpotrf(uplo, n, a, lda, info, handle)
      endif
#endif
    end subroutine

    subroutine gpusolver_spotrf(uplo, n, a, lda, info, handle)
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
        call cusolver_spotrf(uplo, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_spotrf(uplo, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_spotrf(uplo, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        call mkl_sycl_spotrf(uplo, n, a, lda, info, handle)
      endif
#endif
    end subroutine

    subroutine gpusolver_zpotrf(uplo, n, a, lda, info, handle)
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
        call cusolver_zpotrf(uplo, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_zpotrf(uplo, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_zpotrf(uplo, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        call mkl_sycl_zpotrf(uplo, n, a, lda, info, handle)
      endif
#endif
    end subroutine

    subroutine gpusolver_cpotrf(uplo, n, a, lda, info, handle)
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
        call cusolver_cpotrf(uplo, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_cpotrf(uplo, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_cpotrf(uplo, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        call mkl_sycl_cpotrf(uplo, n, a, lda, info, handle)
      endif
#endif
    end subroutine

    subroutine gpublas_dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
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
          call cublas_dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
         if (use_gpu_vendor == openmp_offload_gpu) then
           call mkl_openmp_offload_dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
         if (use_gpu_vendor == sycl_gpu) then
           call mkl_sycl_dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif
    end subroutine

    subroutine gpublas_sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
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
          call cublas_sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

    end subroutine

    subroutine gpublas_zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
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
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

    end subroutine

    subroutine gpublas_cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
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
      complex(kind=C_FLOAT_COMPLEX)   :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

    end subroutine

    subroutine gpublas_dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
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
          call cublas_dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
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
          call cublas_dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_sycl_dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif
    end subroutine 


    subroutine gpublas_sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
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
          call cublas_sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
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
          call cublas_sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)

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
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)

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
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)

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
      complex(kind=C_FLOAT_COMPLEX)   :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=c_intptr_t)     :: handle


        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)

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
      complex(kind=C_FLOAT_COMPLEX)   :: alpha,beta
      type(c_ptr)                     :: a, b, c
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_dcopy_intptr(n, x, incx, y, incy, handle)

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
          call cublas_dcopy_intptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_dcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_dcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_dcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_dcopy_cptr(n, x, incx, y, incy, handle)

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
          call cublas_dcopy_cptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_dcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_dcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_dcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_scopy_intptr(n, x, incx, y, incy, handle)

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
          call cublas_scopy_intptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_scopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_scopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_scopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_scopy_cptr(n, x, incx, y, incy, handle)

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
          call cublas_scopy_cptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_scopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_scopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_scopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine


    subroutine gpublas_zcopy_intptr(n, x, incx, y, incy, handle)

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
          call cublas_zcopy_intptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_zcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_zcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_zcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_zcopy_cptr(n, x, incx, y, incy, handle)

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
          call cublas_zcopy_cptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_zcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_zcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_zcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

    end subroutine

    subroutine gpublas_ccopy_intptr(n, x, incx, y, incy, handle)

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
          call cublas_ccopy_intptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_ccopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_ccopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_ccopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_ccopy_cptr(n, x, incx, y, incy, handle)

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
          call cublas_ccopy_cptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_ccopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_ccopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_ccopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine


    subroutine gpublas_dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

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
          call cublas_dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

    end subroutine


    subroutine gpublas_dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

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
          call cublas_dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine


    subroutine gpublas_strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

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
          call cublas_strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine


    subroutine gpublas_strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

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
          call cublas_strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine


    subroutine gpublas_ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

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
          call cublas_ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine



    subroutine gpublas_ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

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
          call cublas_ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine


    subroutine gpublas_ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

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
          call cublas_ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine



    subroutine gpublas_ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

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
          call cublas_ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine


    subroutine gpublas_dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

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
          call cublas_dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine


    subroutine gpublas_dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

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
          call cublas_dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine


    subroutine gpublas_strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

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
          call cublas_strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine


    subroutine gpublas_strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

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
          call cublas_strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine


    subroutine gpublas_ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

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
          call cublas_ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine



    subroutine gpublas_ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

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
          call cublas_ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine


    subroutine gpublas_ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

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
          call cublas_ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine



    subroutine gpublas_ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

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
          call cublas_ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call mkl_sycl_ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine


end module

