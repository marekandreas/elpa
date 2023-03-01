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
module test_sycl_functions
  use, intrinsic :: iso_c_binding
  use precision_for_tests
  implicit none

  public

  integer(kind=ik) :: syclMemcpyHostToDevice
  integer(kind=ik) :: syclMemcpyDeviceToHost
  integer(kind=ik) :: syclMemcpyDeviceToDevice
  integer(kind=ik) :: syclHostRegisterDefault
  integer(kind=ik) :: syclHostRegisterPortable
  integer(kind=ik) :: syclHostRegisterMapped

  ! TODO global variable, has to be changed
  integer(kind=C_intptr_T) :: syclHandle = -1

  ! functions to set and query the CUDA devices
  interface
    function sycl_setdevice_c(n) result(istat) &
             bind(C, name="syclSetDeviceFromC")

      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT), value    :: n
      integer(kind=C_INT)           :: istat
    end function
  end interface

  interface
    function sycl_memcpyDeviceToDevice_c() result(flag) &
            bind(C, name="syclMemcpyDeviceToDeviceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function sycl_memcpyHostToDevice_c() result(flag) &
             bind(C, name="syclMemcpyHostToDeviceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function sycl_memcpyDeviceToHost_c() result(flag) &
             bind(C, name="syclMemcpyDeviceToHostFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

!  interface
!    function sycl_hostRegisterDefault_c() result(flag) &
!             bind(C, name="syclHostRegisterDefaultFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface

!  interface
!    function sycl_hostRegisterPortable_c() result(flag) &
!             bind(C, name="syclHostRegisterPortableFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface
!
!  interface
!    function sycl_hostRegisterMapped_c() result(flag) &
!             bind(C, name="syclHostRegisterMappedFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface

  interface
    function sycl_memcpy_intptr_c(dst, src, size, dir) result(istat) &
             bind(C, name="syclMemcpyFromC")

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_intptr_t), value              :: dst
      integer(kind=C_intptr_t), value              :: src
      integer(kind=c_intptr_t), intent(in), value    :: size
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=C_INT)                          :: istat

    end function
  end interface

  interface
    function sycl_memcpy_cptr_c(dst, src, size, dir) result(istat) &
             bind(C, name="syclMemcpyFromC")

      use, intrinsic :: iso_c_binding

      implicit none
      type(c_ptr), value                           :: dst
      type(c_ptr), value                           :: src
      integer(kind=c_intptr_t), intent(in), value  :: size
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=C_INT)                          :: istat

    end function
  end interface

  interface
    function sycl_memcpy_mixed_c(dst, src, size, dir) result(istat) &
             bind(C, name="syclMemcpyFromC")

      use, intrinsic :: iso_c_binding

      implicit none
      type(c_ptr), value                           :: dst
      integer(kind=C_intptr_t), value              :: src
      integer(kind=c_intptr_t), intent(in), value  :: size
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=C_INT)                          :: istat

    end function
  end interface


  interface
    function sycl_free_intptr_c(a) result(istat) &
             bind(C, name="syclFreeFromC")

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_intptr_T), value  :: a
      integer(kind=C_INT)              :: istat

    end function sycl_free_intptr_c
  end interface

  interface
    function sycl_free_cptr_c(a) result(istat) &
             bind(C, name="syclFreeFromC")

      use, intrinsic :: iso_c_binding

      implicit none
      type(c_ptr), value  :: a
      integer(kind=C_INT)              :: istat

    end function
  end interface

  interface sycl_memcpy
    module procedure sycl_memcpy_intptr
    module procedure sycl_memcpy_cptr
    module procedure sycl_memcpy_mixed
  end interface

  interface sycl_free
    module procedure sycl_free_intptr
    module procedure sycl_free_cptr
  end interface

  interface
    function sycl_malloc_intptr_c(a, width_height) result(istat) &
             bind(C, name="syclMallocFromC")

      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_T)                    :: a
      integer(kind=c_intptr_t), intent(in), value :: width_height
      integer(kind=C_INT)                         :: istat

    end function
  end interface

  interface
    function sycl_malloc_cptr_c(a, width_height) result(istat) &
             bind(C, name="syclMallocFromC")

      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                    :: a
      integer(kind=c_intptr_t), intent(in), value :: width_height
      integer(kind=C_INT)                         :: istat

    end function
  end interface


  contains

    function sycl_setdevice(n) result(success)
      use, intrinsic :: iso_c_binding

      implicit none

      integer(kind=ik), intent(in)  :: n
      logical                       :: success
      success = .false.
#ifdef WITH_SYCL_GPU_VERSION
      success = sycl_setdevice_c(int(n,kind=c_int)) /= 0
#else
      success = .true.
#endif
    end function


    function sycl_malloc_intptr(a, width_height) result(success)

     use, intrinsic :: iso_c_binding
     implicit none

     integer(kind=c_intptr_t)                  :: a
     integer(kind=c_intptr_t), intent(in)      :: width_height
     logical                                   :: success
#ifdef WITH_SYCL_GPU_VERSION
     success = sycl_malloc_intptr_c(a, width_height) /= 0
#else
     success = .true.
#endif
   end function


    function sycl_malloc_cptr(a, width_height) result(success)

     use, intrinsic :: iso_c_binding
     implicit none

     type(c_ptr)                  :: a
     integer(kind=c_intptr_t), intent(in)      :: width_height
     logical                                   :: success
#ifdef WITH_SYCL_GPU_VERSION
     success = sycl_malloc_cptr_c(a, width_height) /= 0
#else
     success = .true.
#endif
   end function

   function sycl_free_intptr(a) result(success)

     use, intrinsic :: iso_c_binding

     implicit none
     integer(kind=C_intptr_T) :: a
     logical                  :: success
#ifdef WITH_SYCL_GPU_VERSION
     success = sycl_free_intptr_c(a) /= 0
#else
     success = .true.
#endif
   end function

   function sycl_free_cptr(a) result(success)

     use, intrinsic :: iso_c_binding

     implicit none
     type(c_ptr) :: a
     logical                  :: success
#ifdef WITH_SYCL_GPU_VERSION
     success = sycl_free_cptr_c(a) /= 0
#else
     success = .true.
#endif
   end function

 ! functions to memcopy CUDA memory

 function sycl_memcpyDeviceToDevice() result(flag)
   use, intrinsic :: iso_c_binding
   implicit none
   integer(kind=ik) :: flag
#ifdef WITH_SYCL_GPU_VERSION
   flag = int(sycl_memcpyDeviceToDevice_c())
#else
   flag = 0
#endif
 end function

 function sycl_memcpyHostToDevice() result(flag)
   use, intrinsic :: iso_c_binding
   use precision_for_tests
   implicit none
   integer(kind=ik) :: flag
#ifdef WITH_SYCL_GPU_VERSION
   flag = int(sycl_memcpyHostToDevice_c())
#else
   flag = 0
#endif
 end function

 function sycl_memcpyDeviceToHost() result(flag)
   use, intrinsic :: iso_c_binding
   use precision_for_tests
   implicit none
   integer(kind=ik) :: flag
#ifdef WITH_SYCL_GPU_VERSION
   flag = int( sycl_memcpyDeviceToHost_c())
#else
   flag = 0
#endif
 end function

! function sycl_hostRegisterDefault() result(flag)
!   use, intrinsic :: iso_c_binding
!   use precision_for_tests
!   implicit none
!   integer(kind=ik) :: flag
!#ifdef WITH_SYCL_GPU_VERSION
!   flag = int(sycl_hostRegisterDefault_c())
!#else
!   flag = 0
!#endif
! end function
!
! function sycl_hostRegisterPortable() result(flag)
!   use, intrinsic :: iso_c_binding
!   use precision_for_tests
!   implicit none
!   integer(kind=ik) :: flag
!#ifdef WITH_SYCL_GPU_VERSION
!   flag = int(sycl_hostRegisterPortable_c())
!#else
!   flag = 0
!#endif
! end function
!
! function sycl_hostRegisterMapped() result(flag)
!   use, intrinsic :: iso_c_binding
!   use precision_for_tests
!   implicit none
!   integer(kind=ik) :: flag
!#ifdef WITH_SYCL_GPU_VERSION
!   flag = int(sycl_hostRegisterMapped_c())
!#else
!   flag = 0
!#endif
! end function

 function sycl_memcpy_intptr(dst, src, size, dir) result(success)

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_intptr_t)              :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success

#ifdef WITH_SYCL_GPU_VERSION
        success = sycl_memcpy_intptr_c(dst, src, size, dir) /= 0
#else
        success = .true.
#endif
    end function

 function sycl_memcpy_cptr(dst, src, size, dir) result(success)

      use, intrinsic :: iso_c_binding

      implicit none
      type(c_ptr)                           :: dst
      type(c_ptr)                           :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success

#ifdef WITH_SYCL_GPU_VERSION
        success = sycl_memcpy_cptr_c(dst, src, size, dir) /= 0
#else
        !success = .true.
         success = .false.
#endif
    end function

 function sycl_memcpy_mixed(dst, src, size, dir) result(success)

      use, intrinsic :: iso_c_binding

      implicit none
      type(c_ptr)                           :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success

#ifdef WITH_SYCL_GPU_VERSION
        success = sycl_memcpy_mixed_c(dst, src, size, dir) /= 0
#else
        success = .true.
#endif
    end function

end module test_sycl_functions
