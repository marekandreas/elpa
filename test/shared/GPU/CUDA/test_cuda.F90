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
module test_cuda_functions
  use, intrinsic :: iso_c_binding
  use precision_for_tests
  implicit none

  public

  integer(kind=ik) :: cudaMemcpyHostToDevice
  integer(kind=ik) :: cudaMemcpyDeviceToHost
  integer(kind=ik) :: cudaMemcpyDeviceToDevice
  integer(kind=ik) :: cudaHostRegisterDefault
  integer(kind=ik) :: cudaHostRegisterPortable
  integer(kind=ik) :: cudaHostRegisterMapped

  ! TODO global variable, has to be changed
  integer(kind=C_intptr_T) :: cublasHandle = -1

  ! functions to set and query the CUDA devices
  interface
    function cuda_setdevice_c(n) result(istat) &
             bind(C, name="cudaSetDeviceFromC")

      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT), value    :: n
      integer(kind=C_INT)           :: istat
    end function cuda_setdevice_c
  end interface

  ! functions to copy CUDA memory
  interface
    function cuda_memcpyDeviceToDevice_c() result(flag) &
            bind(C, name="cudaMemcpyDeviceToDeviceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function cuda_memcpyHostToDevice_c() result(flag) &
             bind(C, name="cudaMemcpyHostToDeviceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function cuda_memcpyDeviceToHost_c() result(flag) &
             bind(C, name="cudaMemcpyDeviceToHostFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function cuda_hostRegisterDefault_c() result(flag) &
             bind(C, name="cudaHostRegisterDefaultFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function cuda_hostRegisterPortable_c() result(flag) &
             bind(C, name="cudaHostRegisterPortableFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function cuda_hostRegisterMapped_c() result(flag) &
             bind(C, name="cudaHostRegisterMappedFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function cuda_memcpy_intptr_c(dst, src, size, dir) result(istat) &
             bind(C, name="cudaMemcpyFromC")

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_intptr_t), value              :: dst
      integer(kind=C_intptr_t), value              :: src
      integer(kind=c_intptr_t), intent(in), value    :: size
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=C_INT)                          :: istat

    end function cuda_memcpy_intptr_c
  end interface

  interface
    function cuda_memcpy_cptr_c(dst, src, size, dir) result(istat) &
             bind(C, name="cudaMemcpyFromC")

      use, intrinsic :: iso_c_binding

      implicit none
      type(c_ptr), value                           :: dst
      type(c_ptr), value                           :: src
      integer(kind=c_intptr_t), intent(in), value  :: size
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=C_INT)                          :: istat

    end function cuda_memcpy_cptr_c
  end interface

  interface
    function cuda_memcpy_mixed_c(dst, src, size, dir) result(istat) &
             bind(C, name="cudaMemcpyFromC")

      use, intrinsic :: iso_c_binding

      implicit none
      type(c_ptr), value                           :: dst
      integer(kind=C_intptr_t), value              :: src
      integer(kind=c_intptr_t), intent(in), value  :: size
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=C_INT)                          :: istat

    end function cuda_memcpy_mixed_c
  end interface

  ! functions to allocate and free CUDA memory

  interface
    function cuda_free_intptr_c(a) result(istat) &
             bind(C, name="cudaFreeFromC")

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_intptr_T), value  :: a
      integer(kind=C_INT)              :: istat

    end function cuda_free_intptr_c
  end interface

  interface
    function cuda_free_cptr_c(a) result(istat) &
             bind(C, name="cudaFreeFromC")

      use, intrinsic :: iso_c_binding

      implicit none
      type(c_ptr), value  :: a
      integer(kind=C_INT)              :: istat

    end function cuda_free_cptr_c
  end interface

  interface cuda_memcpy
    module procedure cuda_memcpy_intptr
    module procedure cuda_memcpy_cptr
    module procedure cuda_memcpy_mixed
  end interface

  interface cuda_free
    module procedure cuda_free_intptr
    module procedure cuda_free_cptr
  end interface

  interface
    function cuda_malloc_intptr_c(a, width_height) result(istat) &
             bind(C, name="cudaMallocFromC")

      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_T)                    :: a
      integer(kind=c_intptr_t), intent(in), value :: width_height
      integer(kind=C_INT)                         :: istat

    end function cuda_malloc_intptr_c
  end interface

  interface
    function cuda_malloc_cptr_c(a, width_height) result(istat) &
             bind(C, name="cudaMallocFromC")

      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                    :: a
      integer(kind=c_intptr_t), intent(in), value :: width_height
      integer(kind=C_INT)                         :: istat

    end function cuda_malloc_cptr_c
  end interface

  !interface cuda_malloc
  !  module procedure cuda_malloc_intptr
  !  module procedure cuda_malloc_cptr
  !end interface

  contains

    function cuda_setdevice(n) result(success)
      use, intrinsic :: iso_c_binding

      implicit none

      integer(kind=ik), intent(in)  :: n
      logical                       :: success
#ifdef WITH_NVIDIA_GPU_VERSION
      success = cuda_setdevice_c(int(n,kind=c_int)) /= 0
#else
      success = .true.
#endif
    end function cuda_setdevice


    function cuda_malloc_intptr(a, width_height) result(success)

     use, intrinsic :: iso_c_binding
     implicit none

     integer(kind=c_intptr_t)                  :: a
     integer(kind=c_intptr_t), intent(in)      :: width_height
     logical                                   :: success
#ifdef WITH_NVIDIA_GPU_VERSION
     success = cuda_malloc_intptr_c(a, width_height) /= 0
#else
     success = .true.
#endif
   end function


    function cuda_malloc_cptr(a, width_height) result(success)

     use, intrinsic :: iso_c_binding
     implicit none

     type(c_ptr)                  :: a
     integer(kind=c_intptr_t), intent(in)      :: width_height
     logical                                   :: success
#ifdef WITH_NVIDIA_GPU_VERSION
     success = cuda_malloc_cptr_c(a, width_height) /= 0
#else
     success = .true.
#endif
   end function

   function cuda_free_intptr(a) result(success)

     use, intrinsic :: iso_c_binding

     implicit none
     integer(kind=C_intptr_T) :: a
     logical                  :: success
#ifdef WITH_NVIDIA_GPU_VERSION
     success = cuda_free_intptr_c(a) /= 0
#else
     success = .true.
#endif
   end function cuda_free_intptr

   function cuda_free_cptr(a) result(success)

     use, intrinsic :: iso_c_binding

     implicit none
     type(c_ptr) :: a
     logical                  :: success
#ifdef WITH_NVIDIA_GPU_VERSION
     success = cuda_free_cptr_c(a) /= 0
#else
     success = .true.
#endif
   end function cuda_free_cptr

 ! functions to memcopy CUDA memory

 function cuda_memcpyDeviceToDevice() result(flag)
   use, intrinsic :: iso_c_binding
   implicit none
   integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_GPU_VERSION
   flag = int(cuda_memcpyDeviceToDevice_c())
#else
   flag = 0
#endif
 end function

 function cuda_memcpyHostToDevice() result(flag)
   use, intrinsic :: iso_c_binding
   use precision_for_tests
   implicit none
   integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_GPU_VERSION
   flag = int(cuda_memcpyHostToDevice_c())
#else
   flag = 0
#endif
 end function

 function cuda_memcpyDeviceToHost() result(flag)
   use, intrinsic :: iso_c_binding
   use precision_for_tests
   implicit none
   integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_GPU_VERSION
   flag = int( cuda_memcpyDeviceToHost_c())
#else
   flag = 0
#endif
 end function

 function cuda_hostRegisterDefault() result(flag)
   use, intrinsic :: iso_c_binding
   use precision_for_tests
   implicit none
   integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_GPU_VERSION
   flag = int(cuda_hostRegisterDefault_c())
#else
   flag = 0
#endif
 end function

 function cuda_hostRegisterPortable() result(flag)
   use, intrinsic :: iso_c_binding
   use precision_for_tests
   implicit none
   integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_GPU_VERSION
   flag = int(cuda_hostRegisterPortable_c())
#else
   flag = 0
#endif
 end function

 function cuda_hostRegisterMapped() result(flag)
   use, intrinsic :: iso_c_binding
   use precision_for_tests
   implicit none
   integer(kind=ik) :: flag
#ifdef WITH_NVIDIA_GPU_VERSION
   flag = int(cuda_hostRegisterMapped_c())
#else
   flag = 0
#endif
 end function

 function cuda_memcpy_intptr(dst, src, size, dir) result(success)

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_intptr_t)              :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success

#ifdef WITH_NVIDIA_GPU_VERSION
        success = cuda_memcpy_intptr_c(dst, src, size, dir) /= 0
#else
        success = .true.
#endif
    end function

 function cuda_memcpy_cptr(dst, src, size, dir) result(success)

      use, intrinsic :: iso_c_binding

      implicit none
      type(c_ptr)                           :: dst
      type(c_ptr)                           :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success

#ifdef WITH_NVIDIA_GPU_VERSION
        success = cuda_memcpy_cptr_c(dst, src, size, dir) /= 0
#else
        !success = .true.
         success = .false.
#endif
    end function

 function cuda_memcpy_mixed(dst, src, size, dir) result(success)

      use, intrinsic :: iso_c_binding

      implicit none
      type(c_ptr)                           :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success

#ifdef WITH_NVIDIA_GPU_VERSION
        success = cuda_memcpy_mixed_c(dst, src, size, dir) /= 0
#else
        success = .true.
#endif
    end function

end module test_cuda_functions
