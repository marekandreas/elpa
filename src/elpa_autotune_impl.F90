!
!    Copyright 2017, L. Hüdepohl and A. Marek, MPCDF
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

#include "config-f90.h"

module elpa_autotune_impl
  use elpa_abstract_impl
  use, intrinsic :: iso_c_binding
  implicit none
#ifdef ENABLE_AUTOTUNING
  type, extends(elpa_autotune_t) :: elpa_autotune_impl_t
    class(elpa_abstract_impl_t), pointer :: parent => NULL()
    integer :: current = 0
    real(kind=C_DOUBLE) :: min_val = 0.0_C_DOUBLE
    integer :: min_loc = 0
    integer :: cardinality = 0
    integer :: level = 0
    integer :: domain = 0
    contains
      procedure, public :: print => elpa_autotune_print
      procedure, public :: destroy => elpa_autotune_destroy
  end type

  contains

    !> \brief function to print the autotuning
    !> Parameters
    !> \param   self  class(elpa_autotune_impl_t) the allocated ELPA autotune object
    subroutine elpa_autotune_print(self, error)
      implicit none
      class(elpa_autotune_impl_t), intent(in) :: self
#ifdef USE_FORTRAN2008
      integer, intent(out), optional :: error
#else
      integer, intent(out)           :: error
#endif

      ! nothing to do atm
#ifdef USE_FORTRAN2008
      if (present(error)) error = ELPA_OK
#else
      error = ELPA_OK
#endif
    end subroutine

    !> \brief function to destroy an elpa autotune object
    !> Parameters
    !> \param   self  class(elpa_autotune_impl_t) the allocated ELPA autotune object
    !> \param   error integer, optional error code
    subroutine elpa_autotune_destroy(self, error)
      implicit none
      class(elpa_autotune_impl_t), intent(inout) :: self
#ifdef USE_FORTRAN2008
      integer, optional, intent(out)             :: error
#else
      integer, intent(out)                       :: error
#endif
      
      ! nothing to do atm
#ifdef USE_FORTRAN2008
      if (present(error)) error = ELPA_OK
#else
      error = ELPA_OK
#endif
    end subroutine
#endif
end module
