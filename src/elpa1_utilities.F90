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
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".



! ELPA2 -- 2-stage solver for ELPA
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".



#include "config-f90.h"
#include <elpa/elpa_kernel_constants.h>

module ELPA1_utilities
  use ELPA_utilities
  use precision
  implicit none

  PRIVATE ! By default, all routines contained are private

  ! The following routines are public:

  public :: get_actual_real_kernel_name, get_actual_complex_kernel_name

  public :: REAL_ELPA_KERNEL_GENERIC, REAL_ELPA_KERNEL_GPU, DEFAULT_REAL_ELPA_KERNEL

  public :: COMPLEX_ELPA_KERNEL_GENERIC, COMPLEX_ELPA_KERNEL_GPU, DEFAULT_COMPLEX_ELPA_KERNEL

  public :: REAL_ELPA_KERNEL_NAMES, COMPLEX_ELPA_KERNEL_NAMES

  public :: get_actual_complex_kernel, get_actual_real_kernel

  public :: check_allowed_complex_kernels, check_allowed_real_kernels

  public :: AVAILABLE_COMPLEX_ELPA_KERNELS, AVAILABLE_REAL_ELPA_KERNELS

  public :: print_available_real_kernels, print_available_complex_kernels
  public :: query_available_real_kernels, query_available_complex_kernels

  
  integer, parameter :: number_of_real_kernels           = ELPA1_NUMBER_OF_REAL_KERNELS
  integer, parameter :: REAL_ELPA_KERNEL_GENERIC         = ELPA1_REAL_KERNEL_GENERIC
  integer(kind=ik), parameter :: REAL_ELPA_KERNEL_GPU    = ELPA1_REAL_KERNEL_GPU
  
! #ifdef WITH_GPU_VERSION
!   integer(kind=ik), parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_GPU
! #else
  integer(kind=ik), parameter :: DEFAULT_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_GENERIC
! #endif

  character(35), parameter, dimension(number_of_real_kernels) :: &
  REAL_ELPA_KERNEL_NAMES =    (/"REAL_ELPA_KERNEL_GENERIC         ", &
                                "REAL_ELPA_KERNEL_GPU             "/)

  integer, parameter :: number_of_complex_kernels           = ELPA1_NUMBER_OF_COMPLEX_KERNELS
  integer, parameter :: COMPLEX_ELPA_KERNEL_GENERIC         = ELPA1_COMPLEX_KERNEL_GENERIC
  integer(kind=ik), parameter :: COMPLEX_ELPA_KERNEL_GPU    = ELPA1_COMPLEX_KERNEL_GPU

! #ifdef WITH_GPU_VERSION
!   integer(kind=ik), parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_GPU
! #else
   integer(kind=ik), parameter :: DEFAULT_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_GENERIC
! #endif

  character(35), parameter, dimension(number_of_complex_kernels) :: &
  COMPLEX_ELPA_KERNEL_NAMES = (/"COMPLEX_ELPA_KERNEL_GENERIC         ", &
                                "COMPLEX_ELPA_KERNEL_GPU             "/)

  integer(kind=ik), parameter                           ::             &
           AVAILABLE_REAL_ELPA_KERNELS(number_of_real_kernels) =       &
                                      (/                               &
#if WITH_REAL_GENERIC_KERNEL
                                        1                              &
#else
                                        0                              &
#endif
#ifdef WITH_GPU_VERSION
                                                                 ,1    &
#else
                                                                 ,0    &
#endif
                                                       /)

  integer(kind=ik), parameter ::                                          &
           AVAILABLE_COMPLEX_ELPA_KERNELS(number_of_complex_kernels) =    &
                                      (/                                  &
#if WITH_COMPLEX_GENERIC_KERNEL
                                        1                                 &
#else
                                        0                                 &
#endif
#ifdef WITH_GPU_VERSION
                                                             ,1           &
#else
                                                             ,0           &
#endif
                                                               /)

!******
  contains
    subroutine print_available_real_kernels
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      implicit none

      integer(kind=ik) :: i

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("print_available_real_kernels")
#endif

      do i=1, number_of_real_kernels
        if (AVAILABLE_REAL_ELPA_KERNELS(i) .eq. 1) then
          write(*,*) REAL_ELPA_KERNEL_NAMES(i)
        endif
      enddo
      write(*,*) " "
      write(*,*) " At the moment the following kernel would be choosen:"
      write(*,*) get_actual_real_kernel_name()

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("print_available_real_kernels")
#endif

    end subroutine print_available_real_kernels

    subroutine query_available_real_kernels
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      implicit none

      integer :: i

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("query_available_real_kernels")
#endif

      do i=1, number_of_real_kernels
        if (AVAILABLE_REAL_ELPA_KERNELS(i) .eq. 1) then
          write(error_unit,*) REAL_ELPA_KERNEL_NAMES(i)
        endif
      enddo
      write(error_unit,*) " "
      write(error_unit,*) " At the moment the following kernel would be choosen:"
      write(error_unit,*) get_actual_real_kernel_name()

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("query_available_real_kernels")
#endif

    end subroutine query_available_real_kernels

    subroutine print_available_complex_kernels
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      implicit none

      integer(kind=ik) :: i
#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("print_available_complex_kernels")
#endif

      do i=1, number_of_complex_kernels
        if (AVAILABLE_COMPLEX_ELPA_KERNELS(i) .eq. 1) then
           write(*,*) COMPLEX_ELPA_KERNEL_NAMES(i)
        endif
      enddo
      write(*,*) " "
      write(*,*) " At the moment the following kernel would be choosen:"
      write(*,*) get_actual_complex_kernel_name()

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("print_available_complex_kernels")
#endif

    end subroutine print_available_complex_kernels

    subroutine query_available_complex_kernels
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif

      implicit none

      integer :: i
#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("query_available_complex_kernels")
#endif

      do i=1, number_of_complex_kernels
        if (AVAILABLE_COMPLEX_ELPA_KERNELS(i) .eq. 1) then
           write(error_unit,*) COMPLEX_ELPA_KERNEL_NAMES(i)
        endif
      enddo
      write(error_unit,*) " "
      write(error_unit,*) " At the moment the following kernel would be choosen:"
      write(error_unit,*) get_actual_complex_kernel_name()

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("query_available_complex_kernels")
#endif

    end subroutine query_available_complex_kernels

    function get_actual_real_kernel() result(actual_kernel)
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      implicit none

      integer(kind=ik) :: actual_kernel

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("get_actual_real_kernel")
#endif


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
!        stop
!      endif
!#endif

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("get_actual_real_kernel")
#endif

    end function get_actual_real_kernel

    function get_actual_real_kernel_name() result(actual_kernel_name)
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      implicit none

      character(35)    :: actual_kernel_name
      integer(kind=ik) :: actual_kernel

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("get_actual_real_kernel_name")
#endif

      actual_kernel = get_actual_real_kernel()
      actual_kernel_name = REAL_ELPA_KERNEL_NAMES(actual_kernel)

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("get_actual_real_kernel_name")
#endif

    end function get_actual_real_kernel_name

    function get_actual_complex_kernel() result(actual_kernel)
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      use precision
      implicit none
      integer(kind=ik) :: actual_kernel

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("get_actual_complex_kernel")
#endif


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
!        stop
!      endif
!#endif


#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("get_actual_complex_kernel")
#endif

   end function get_actual_complex_kernel

   function get_actual_complex_kernel_name() result(actual_kernel_name)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     use precision
     implicit none
     character(35)    :: actual_kernel_name
     integer(kind=ik) :: actual_kernel

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("get_actual_complex_kernel_name")
#endif

     actual_kernel = get_actual_complex_kernel()
     actual_kernel_name = COMPLEX_ELPA_KERNEL_NAMES(actual_kernel)

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("get_actual_complex_kernel_name")
#endif

   end function get_actual_complex_kernel_name

   function check_allowed_real_kernels(THIS_REAL_ELPA_KERNEL) result(err)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     use precision
     implicit none
     integer(kind=ik), intent(in) :: THIS_REAL_ELPA_KERNEL
     logical                      :: err

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("check_allowed_real_kernels")
#endif
     err = .false.

     if (AVAILABLE_REAL_ELPA_KERNELS(THIS_REAL_ELPA_KERNEL) .ne. 1) err=.true.

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("check_allowed_real_kernels")
#endif

   end function check_allowed_real_kernels

   function check_allowed_complex_kernels(THIS_COMPLEX_ELPA_KERNEL) result(err)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     use precision
     implicit none
     integer(kind=ik), intent(in) :: THIS_COMPLEX_ELPA_KERNEL
     logical                      :: err
#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("check_allowed_complex_kernels")
#endif
     err = .false.

     if (AVAILABLE_COMPLEX_ELPA_KERNELS(THIS_COMPLEX_ELPA_KERNEL) .ne. 1) err=.true.

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("check_allowed_complex_kernels")
#endif

   end function check_allowed_complex_kernels

   function real_kernel_via_environment_variable() result(kernel)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     use precision
     implicit none
     integer(kind=ik)   :: kernel
     CHARACTER(len=255) :: REAL_KERNEL_ENVIRONMENT
     integer(kind=ik)   :: i

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("real_kernel_via_environment_variable")
#endif

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

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("real_kernel_via_environment_variable")
#endif

   end function real_kernel_via_environment_variable

   function complex_kernel_via_environment_variable() result(kernel)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     use precision
     implicit none
     integer :: kernel

     CHARACTER(len=255) :: COMPLEX_KERNEL_ENVIRONMENT
     integer(kind=ik)   :: i

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("complex_kernel_via_environment_variable")
#endif

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

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("complex_kernel_via_environment_variable")
#endif

   end function
!-------------------------------------------------------------------------------

end module ELPA1_utilities
