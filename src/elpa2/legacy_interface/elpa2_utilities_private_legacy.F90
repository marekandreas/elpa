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

  contains

    function elpa_get_actual_real_kernel() result(actual_kernel)
      use elpa_constants
      use precision
      implicit none

      integer(kind=ik) :: actual_kernel

      ! if kernel is not choosen via api
      ! check whether set by environment variable
      actual_kernel = real_kernel_via_environment_variable()

      if (actual_kernel .eq. 0) then
        ! if not then set default kernel
        actual_kernel = ELPA_2STAGE_REAL_DEFAULT
      endif

    end function elpa_get_actual_real_kernel

    function elpa_get_actual_complex_kernel() result(actual_kernel)
      use elpa2_constants
      use precision
      implicit none
      integer(kind=ik) :: actual_kernel

     ! if kernel is not choosen via api
     ! check whether set by environment variable
     actual_kernel = complex_kernel_via_environment_variable()

     if (actual_kernel .eq. 0) then
       ! if not then set default kernel
       actual_kernel = ELPA_2STAGE_COMPLEX_DEFAULT
     endif

   end function elpa_get_actual_complex_kernel


   function qr_decomposition_via_environment_variable(useQR) result(isSet)
     use elpa2_utilities
     use precision
     implicit none
     logical, intent(out) :: useQR
     logical              :: isSet
     character(len=255)   :: ELPA_QR_DECOMPOSITION

     isSet = .false.

#if defined(HAVE_ENVIRONMENT_CHECKING)
     call get_environment_variable("ELPA_QR_DECOMPOSITION",ELPA_QR_DECOMPOSITION)
#else 
     stop "Internal error in elpa2_utilities_private_legacy.F90, this should not happen"
#endif
     if (trim(ELPA_QR_DECOMPOSITION) .eq. "yes") then
       useQR = .true.
       isSet = .true.
     endif
     if (trim(ELPA_QR_DECOMPOSITION) .eq. "no") then
       useQR = .false.
       isSet = .true.
     endif

   end function qr_decomposition_via_environment_variable

   function real_kernel_via_environment_variable(elpa) result(kernel)
     use elpa_constants
     use precision
     implicit none
     type(elpa_t) :: elpa
     integer(kind=ik)   :: kernel
     character(len=255) :: REAL_KERNEL_ENVIRONMENT
     integer(kind=ik)   :: i

#if defined(HAVE_ENVIRONMENT_CHECKING)
     call get_environment_variable("REAL_ELPA_KERNEL",REAL_KERNEL_ENVIRONMENT)
#else
     stop "Internal error in elpa2_utilities_private_legacy.F90, this should not happen"
#endif
     do i=1,size(REAL_ELPA_KERNEL_NAMES(:))
       if (trim(REAL_KERNEL_ENVIRONMENT) .eq. trim(REAL_ELPA_KERNEL_NAMES(i))) then
         kernel = i
         exit
       else
         kernel = 0
       endif
     enddo
   end function real_kernel_via_environment_variable

   function complex_kernel_via_environment_variable() result(kernel)
     use elpa_constants
     use precision
     implicit none
     integer :: kernel

     CHARACTER(len=255) :: COMPLEX_KERNEL_ENVIRONMENT
     integer(kind=ik)   :: i

#if defined(HAVE_ENVIRONMENT_CHECKING)
     call get_environment_variable("COMPLEX_ELPA_KERNEL",COMPLEX_KERNEL_ENVIRONMENT)
#else
     stop "Internal error in elpa2_utilities_private_legacy.F90, this should not happen"
#endif

     do i=1,size(COMPLEX_ELPA_KERNEL_NAMES(:))
       if (trim(COMPLEX_ELPA_KERNEL_NAMES(i)) .eq. trim(COMPLEX_KERNEL_ENVIRONMENT)) then
         kernel = i
         exit
       else
         kernel = 0
       endif
     enddo

   end function
!-------------------------------------------------------------------------------

end module elpa2_utilities_private
