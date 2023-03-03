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
module test_gpu
  !use precision_for_tests
  use precision_for_tests
  use iso_c_binding
!#if TEST_INTEL_GPU == 1
!  use mkl_offload
!#endif
  integer(kind=c_int), parameter :: nvidia_gpu = 1
  integer(kind=c_int), parameter :: amd_gpu = 2
  integer(kind=c_int), parameter :: intel_gpu = 3
  integer(kind=c_int), parameter :: no_gpu = -1
  integer(kind=c_int)            :: use_gpu_vendor
  integer(kind=c_int)            :: gpuHostRegisterDefault
  integer(kind=c_int)            :: gpuMemcpyHostToDevice
  integer(kind=c_int)            :: gpuMemcpyDeviceToHost
  integer(kind=c_int)            :: gpuMemcpyDeviceToDevice
  integer(kind=c_int)            :: gpuHostRegisterMapped
  integer(kind=c_int)            :: gpuHostRegisterPortable

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
    module procedure gpu_memcpy_mixed
  end interface

  interface gpu_malloc
    module procedure gpu_malloc_intptr
    module procedure gpu_malloc_cptr
  end interface

  interface gpu_free
    module procedure gpu_free_intptr
    module procedure gpu_free_cptr
  end interface

  contains
    function gpu_vendor(set_vendor) result(vendor)
      use precision_for_tests
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
!#if TEST_INTEL_GPU == 1
!      vendor = intel_gpu
!#endif
      use_gpu_vendor = vendor
      return
    end function

    subroutine set_gpu_parameters
#ifdef WITH_NVIDIA_GPU_VERSION
      use test_cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use test_hip_functions
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

    end subroutine

    function gpu_malloc_intptr(array, elements) result(success)
      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use test_cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use test_hip_functions
#endif
      implicit none
      integer(kind=C_intptr_T)             :: array
      integer(kind=c_intptr_t), intent(in) :: elements
      logical                              :: success

      success = .false.

#ifdef WITH_NVIDIA_GPU_VERSION
      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_malloc_intptr(array, elements)
      endif
#endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_malloc_intptr(array, elements)
      endif
#endif

    end function

    function gpu_malloc_cptr(array, elements) result(success)
      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use test_cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use test_hip_functions
#endif
      implicit none
      type(c_ptr)                          :: array
      integer(kind=c_intptr_t), intent(in) :: elements
      logical                              :: success

      success = .false.

#ifdef WITH_NVIDIA_GPU_VERSION
      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_malloc_cptr(array, elements)
      endif
#endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_malloc_cptr(array, elements)
      endif
#endif

    end function

    function gpu_memcpy_intptr(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use test_cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use test_hip_functions
#endif
      implicit none
      integer(kind=C_intptr_t)              :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success

      success = .false.

#ifdef WITH_NVIDIA_GPU_VERSION
      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memcpy_intptr(dst, src, size, dir)
      endif
#endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_memcpy_intptr(dst, src, size, dir)
      endif
#endif

    end function
    
    function gpu_memcpy_cptr(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use test_cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use test_hip_functions
#endif
      implicit none
      type(c_ptr)                           :: dst
      type(c_ptr)                           :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success

      success = .false.

#ifdef WITH_NVIDIA_GPU_VERSION
      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memcpy_cptr(dst, src, size, dir)
      endif
#endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_memcpy_cptr(dst, src, size, dir)
      endif
#endif

    end function
    
    function gpu_memcpy_mixed(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use test_cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use test_hip_functions
#endif
      implicit none
      type(c_ptr)                           :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success

      success = .false.

#ifdef WITH_NVIDIA_GPU_VERSION
      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memcpy_mixed(dst, src, size, dir)
      endif
#endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_memcpy_mixed(dst, src, size, dir)
      endif
#endif

    end function

    function gpu_free_intptr(a) result(success)
      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use test_cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use test_hip_functions
#endif
      implicit none
      integer(kind=c_intptr_t)                :: a

      logical :: success

      success = .false.

#ifdef WITH_NVIDIA_GPU_VERSION
      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_free_intptr(a)
      endif
#endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_free_intptr(a)
      endif
#endif

    end function

    function gpu_free_cptr(a) result(success)
      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use test_cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use test_hip_functions
#endif
      implicit none
      type(c_ptr)                :: a

      logical :: success

      success = .false.

#ifdef WITH_NVIDIA_GPU_VERSION
      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_free_cptr(a)
      endif
#endif

#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        success = hip_free_cptr(a)
      endif
#endif

    end function

end module 
