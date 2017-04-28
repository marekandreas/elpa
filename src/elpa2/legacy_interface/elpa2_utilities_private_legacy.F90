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
!
! ELPA2 -- 2-stage solver for ELPA
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
! Author: Andreas Marek, MPCDF


#include "config-f90.h"

module elpa2_utilities_private
  use elpa_utilities
  use precision
  implicit none

  PRIVATE ! By default, all routines contained are private

  ! The following routines are public:

  public :: elpa_get_actual_real_kernel_name, elpa_get_actual_complex_kernel_name

  public :: elpa_get_actual_complex_kernel, elpa_get_actual_real_kernel

  public :: check_allowed_complex_kernels, check_allowed_real_kernels

  !public :: AVAILABLE_COMPLEX_ELPA_KERNELS, AVAILABLE_REAL_ELPA_KERNELS

  public :: print_available_real_kernels, print_available_complex_kernels
  public :: query_available_real_kernels, query_available_complex_kernels

  public :: elpa_number_of_real_kernels, elpa_number_of_complex_kernels
  public :: elpa_real_kernel_is_available, elpa_complex_kernel_is_available
  !public :: elpa_real_kernel_name, elpa_complex_kernel_name

  public :: qr_decomposition_via_environment_variable

!******
  contains
    function elpa_number_of_real_kernels() result(number)
      use elpa2_utilities
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#else
      use timings_dummy
#endif
      implicit none

      integer :: number
      call timer%start("elpa_number_of_real_kernels")

      number = number_of_real_kernels

      call timer%stop("elpa_number_of_real_kernels")
      return

    end function

    function elpa_number_of_complex_kernels() result(number)
      use elpa2_utilities
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#else
      use timings_dummy
#endif
      implicit none

      integer :: number
      call timer%start("elpa_number_of_complex_kernels")

      number = number_of_complex_kernels

      call timer%stop("elpa_number_of_complex_kernels")
      return

    end function

   function elpa_real_kernel_is_available(THIS_ELPA_REAL_KERNEL) result(available)
      use elpa2_utilities
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#else
      use timings_dummy
#endif
      implicit none

      integer, intent(in) :: THIS_ELPA_REAL_KERNEL
      logical             :: available
      call timer%start("elpa_real_kernel_is_available")

     available = .false.

     if (AVAILABLE_REAL_ELPA_KERNELS(THIS_ELPA_REAL_KERNEL) .eq. 1) then
       available = .true.
     endif
      call timer%stop("elpa_real_kernel_is_available")
      return

    end function

   function elpa_complex_kernel_is_available(THIS_ELPA_COMPLEX_KERNEL) result(available)
      use elpa2_utilities
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#else
      use timings_dummy
#endif
      implicit none

      integer, intent(in) :: THIS_ELPA_COMPLEX_KERNEL
      logical             :: available
      call timer%start("elpa_real_kernel_is_available")

     available = .false.

     if (AVAILABLE_COMPLEX_ELPA_KERNELS(THIS_ELPA_COMPLEX_KERNEL) .eq. 1) then
       available = .true.
     endif
      call timer%stop("elpa_real_kernel_is_available")
      return

    end function

!   function elpa_real_kernel_name(THIS_ELPA_REAL_KERNEL) result(name)
!      use elpa2_utilities
!#ifdef HAVE_DETAILED_TIMINGS
!      use timings
!#else
!      use timings_dummy
!#endif
!      implicit none
!
!      integer, intent(in) :: THIS_ELPA_REAL_KERNEL
!      character(35)        :: name
!      call timer%start("elpa_real_kernel_name")
!
!
!     if (AVAILABLE_REAL_ELPA_KERNELS(THIS_ELPA_REAL_KERNEL) .eq. 1) then
!       name = trim(REAL_ELPA_KERNEL_NAMES(THIS_ELPA_REAL_KERNEL))
!     endif
!      call timer%stop("elpa_real_kernel_name")
!      return
!
!    end function
!
!   function elpa_complex_kernel_name(THIS_ELPA_COMPLEX_KERNEL) result(name)
!      use elpa2_utilities
!#ifdef HAVE_DETAILED_TIMINGS
!      use timings
!#else
!      use timings_dummy
!#endif
!      implicit none
!
!      integer, intent(in) :: THIS_ELPA_COMPLEX_KERNEL
!      character(35)       :: name
!      call timer%start("elpa_complex_kernel_name")
!
!
!     if (AVAILABLE_COMPLEX_ELPA_KERNELS(THIS_ELPA_COMPLEX_KERNEL) .eq. 1) then
!       name = trim(COMPLEX_ELPA_KERNEL_NAMES(THIS_ELPA_COMPLEX_KERNEL))
!     endif
!      call timer%stop("elpa_complex_kernel_name")
!      return
!
!    end function

    subroutine print_available_real_kernels
      use elpa2_utilities
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#else
      use timings_dummy
#endif
      use precision
      implicit none

      integer(kind=ik) :: i

      call timer%start("print_available_real_kernels")

      do i=1, number_of_real_kernels
        if (AVAILABLE_REAL_ELPA_KERNELS(i) .eq. 1) then
          write(*,*) REAL_ELPA_KERNEL_NAMES(i)
        endif
      enddo
      write(*,*) " "
      write(*,*) " At the moment the following kernel would be choosen:"
      write(*,*) elpa_get_actual_real_kernel_name()

      call timer%stop("print_available_real_kernels")

    end subroutine print_available_real_kernels

    subroutine query_available_real_kernels
      use elpa2_utilities
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#else
      use timings_dummy
#endif
      implicit none

      integer :: i

      call timer%start("query_available_real_kernels")

      do i=1, number_of_real_kernels
        if (AVAILABLE_REAL_ELPA_KERNELS(i) .eq. 1) then
          write(error_unit,*) REAL_ELPA_KERNEL_NAMES(i)
        endif
      enddo
      write(error_unit,*) " "
      write(error_unit,*) " At the moment the following kernel would be choosen:"
      write(error_unit,*) elpa_get_actual_real_kernel_name()

      call timer%stop("query_available_real_kernels")

    end subroutine query_available_real_kernels

    subroutine print_available_complex_kernels
      use elpa2_utilities
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#else
      use timings_dummy
#endif
      use precision
      implicit none

      integer(kind=ik) :: i
      call timer%start("print_available_complex_kernels")

      do i=1, number_of_complex_kernels
        if (AVAILABLE_COMPLEX_ELPA_KERNELS(i) .eq. 1) then
           write(*,*) COMPLEX_ELPA_KERNEL_NAMES(i)
        endif
      enddo
      write(*,*) " "
      write(*,*) " At the moment the following kernel would be choosen:"
      write(*,*) elpa_get_actual_complex_kernel_name()

      call timer%stop("print_available_complex_kernels")

    end subroutine print_available_complex_kernels

    subroutine query_available_complex_kernels
      use elpa2_utilities
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#else
      use timings_dummy
#endif

      implicit none

      integer :: i
      call timer%start("query_available_complex_kernels")

      do i=1, number_of_complex_kernels
        if (AVAILABLE_COMPLEX_ELPA_KERNELS(i) .eq. 1) then
           write(error_unit,*) COMPLEX_ELPA_KERNEL_NAMES(i)
        endif
      enddo
      write(error_unit,*) " "
      write(error_unit,*) " At the moment the following kernel would be choosen:"
      write(error_unit,*) elpa_get_actual_complex_kernel_name()

      call timer%stop("query_available_complex_kernels")

    end subroutine query_available_complex_kernels

    function elpa_get_actual_real_kernel() result(actual_kernel)
      use elpa2_utilities
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#else
      use timings_dummy
#endif
      use precision
      implicit none

      integer(kind=ik) :: actual_kernel

      call timer%start("elpa_get_actual_real_kernel")

      ! if kernel is not choosen via api
      ! check whether set by environment variable
      actual_kernel = real_kernel_via_environment_variable()

!#ifdef WITH_GPU_VERSION
!      actual_kernel = REAL_ELPA_KERNEL_GPU
!#endif
      if (actual_kernel .eq. 0) then
        ! if not then set default kernel
        actual_kernel = DEFAULT_REAL_ELPA_KERNEL
      endif

!#ifdef WITH_GPU_VERSION
!      if (actual_kernel .ne. REAL_ELPA_KERNEL_GPU) then
!        print *,"if build with GPU you cannot choose another real kernel"
!        stop 1
!      endif
!#endif

      call timer%stop("elpa_get_actual_real_kernel")

    end function elpa_get_actual_real_kernel

    function elpa_get_actual_real_kernel_name() result(actual_kernel_name)
      use elpa2_utilities
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#else
      use timings_dummy
#endif
      use precision
      implicit none

      character(35)    :: actual_kernel_name
      integer(kind=ik) :: actual_kernel

      call timer%start("elpa_get_actual_real_kernel_name")

      actual_kernel = elpa_get_actual_real_kernel()
      actual_kernel_name = REAL_ELPA_KERNEL_NAMES(actual_kernel)

      call timer%stop("elpa_get_actual_real_kernel_name")

    end function elpa_get_actual_real_kernel_name

    function elpa_get_actual_complex_kernel() result(actual_kernel)
      use elpa2_utilities
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#else
      use timings_dummy
#endif
      use precision
      implicit none
      integer(kind=ik) :: actual_kernel

      call timer%start("elpa_get_actual_complex_kernel")


     ! if kernel is not choosen via api
     ! check whether set by environment variable
     actual_kernel = complex_kernel_via_environment_variable()

!#ifdef WITH_GPU_VERSION
!     actual_kernel = COMPLEX_ELPA_KERNEL_GPU
!#endif
     if (actual_kernel .eq. 0) then
       ! if not then set default kernel
       actual_kernel = DEFAULT_COMPLEX_ELPA_KERNEL
     endif

!#ifdef WITH_GPU_VERSION
!      if (actual_kernel .ne. COMPLEX_ELPA_KERNEL_GPU) then
!        print *,"if build with GPU you cannot choose another complex kernel"
!        stop 1
!      endif
!#endif


     call timer%stop("elpa_get_actual_complex_kernel")

   end function elpa_get_actual_complex_kernel

   function elpa_get_actual_complex_kernel_name() result(actual_kernel_name)
     use elpa2_utilities
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#else
     use timings_dummy
#endif
     use precision
     implicit none
     character(35)    :: actual_kernel_name
     integer(kind=ik) :: actual_kernel

     call timer%start("elpa_get_actual_complex_kernel_name")

     actual_kernel = elpa_get_actual_complex_kernel()
     actual_kernel_name = COMPLEX_ELPA_KERNEL_NAMES(actual_kernel)

     call timer%stop("elpa_get_actual_complex_kernel_name")

   end function elpa_get_actual_complex_kernel_name

   function check_allowed_real_kernels(THIS_REAL_ELPA_KERNEL) result(err)
     use elpa2_utilities
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#else
     use timings_dummy
#endif
     use precision
     implicit none
     integer(kind=ik), intent(in) :: THIS_REAL_ELPA_KERNEL
     logical                      :: err

     call timer%start("check_allowed_real_kernels")
     err = .false.

     if (AVAILABLE_REAL_ELPA_KERNELS(THIS_REAL_ELPA_KERNEL) .ne. 1) err=.true.

     call timer%stop("check_allowed_real_kernels")

   end function check_allowed_real_kernels

   function check_allowed_complex_kernels(THIS_COMPLEX_ELPA_KERNEL) result(err)
     use elpa2_utilities
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#else
     use timings_dummy
#endif
     use precision
     implicit none
     integer(kind=ik), intent(in) :: THIS_COMPLEX_ELPA_KERNEL
     logical                      :: err
     call timer%start("check_allowed_complex_kernels")
     err = .false.

     if (AVAILABLE_COMPLEX_ELPA_KERNELS(THIS_COMPLEX_ELPA_KERNEL) .ne. 1) err=.true.

     call timer%stop("check_allowed_complex_kernels")

   end function check_allowed_complex_kernels

   function qr_decomposition_via_environment_variable(useQR) result(isSet)
     use elpa2_utilities
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#else
     use timings_dummy
#endif
     use precision
     implicit none
     logical, intent(out) :: useQR
     logical              :: isSet
     CHARACTER(len=255)   :: ELPA_QR_DECOMPOSITION

     call timer%start("qr_decomposition_via_environment_variable")

     isSet = .false.

#if defined(HAVE_ENVIRONMENT_CHECKING)
     call get_environment_variable("ELPA_QR_DECOMPOSITION",ELPA_QR_DECOMPOSITION)
#endif
     if (trim(ELPA_QR_DECOMPOSITION) .eq. "yes") then
       useQR = .true.
       isSet = .true.
     endif
     if (trim(ELPA_QR_DECOMPOSITION) .eq. "no") then
       useQR = .false.
       isSet = .true.
     endif

     call timer%stop("qr_decomposition_via_environment_variable")

   end function qr_decomposition_via_environment_variable

   function real_kernel_via_environment_variable() result(kernel)
     use elpa2_utilities
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#else
     use timings_dummy
#endif
     use precision
     implicit none
     integer(kind=ik)   :: kernel
     CHARACTER(len=255) :: REAL_KERNEL_ENVIRONMENT
     integer(kind=ik)   :: i

     call timer%start("real_kernel_via_environment_variable")

#if defined(HAVE_ENVIRONMENT_CHECKING)
     call get_environment_variable("REAL_ELPA_KERNEL",REAL_KERNEL_ENVIRONMENT)
#endif
     do i=1,size(REAL_ELPA_KERNEL_NAMES(:))
       !     if (trim(dummy_char) .eq. trim(REAL_ELPA_KERNEL_NAMES(i))) then
       if (trim(REAL_KERNEL_ENVIRONMENT) .eq. trim(REAL_ELPA_KERNEL_NAMES(i))) then
         kernel = i
         exit
       else
         kernel = 0
       endif
     enddo

     call timer%stop("real_kernel_via_environment_variable")

   end function real_kernel_via_environment_variable

   function complex_kernel_via_environment_variable() result(kernel)
     use elpa2_utilities
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#else
     use timings_dummy
#endif
     use precision
     implicit none
     integer :: kernel

     CHARACTER(len=255) :: COMPLEX_KERNEL_ENVIRONMENT
     integer(kind=ik)   :: i

     call timer%start("complex_kernel_via_environment_variable")

#if defined(HAVE_ENVIRONMENT_CHECKING)
     call get_environment_variable("COMPLEX_ELPA_KERNEL",COMPLEX_KERNEL_ENVIRONMENT)
#endif

     do i=1,size(COMPLEX_ELPA_KERNEL_NAMES(:))
       if (trim(COMPLEX_ELPA_KERNEL_NAMES(i)) .eq. trim(COMPLEX_KERNEL_ENVIRONMENT)) then
         kernel = i
         exit
       else
         kernel = 0
       endif
     enddo

     call timer%stop("complex_kernel_via_environment_variable")

   end function
!-------------------------------------------------------------------------------

end module elpa2_utilities_private
