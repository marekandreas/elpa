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
! ELPA2 -- 2-stage solver for ELPA
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
! This file was written by A. Marek, MPCDF

#include "config-f90.h"

!> \file print_available_elpa2_kernels.F90
!> \par
!> \brief Provide information which ELPA2 kernels are available on this system
!>
!> \details
!> It is possible to configure ELPA2 such, that different compute intensive
!> "ELPA2 kernels" can be choosen at runtime.
!> The service binary print_available_elpa2_kernels will query the library and tell
!> whether ELPA2 has been configured in this way, and if this is the case which kernels can be
!> choosen at runtime.
!> It will furthermore detail whether ELPA has been configured with OpenMP support
!>
!> Synopsis: print_available_elpa2_kernels
!>
!> \author A. Marek (MPCDF)

program print_available_elpa2_kernels
   use precision
   use elpa

   implicit none

   integer(kind=ik) :: i
   class(elpa_t), pointer :: e
   integer :: option

   if (elpa_init(CURRENT_API_VERSION) /= ELPA_OK) then
     print *, "Unsupported ELPA API Version"
     stop 1
   endif

   e => elpa_allocate()

   print *, "This program will give information on the ELPA2 kernels, "
   print *, "which are available with this library and it will give "
   print *, "information if (and how) the kernels can be choosen at "
   print *, "runtime"
   print *
#ifdef WITH_OPENMP
   print *, " ELPA supports threads: yes"
#else
   print *, " ELPA supports threads: no"
#endif
   print *

   print *, "Information on ELPA2 real case: "
   print *, "=============================== "
#ifdef HAVE_ENVIRONMENT_CHECKING
   print *, " choice via environment variable: yes"
   print *, " environment variable name      : ELPA_2STAGE_REAL_KERNEL"
#else
   print *, " choice via environment variable: no"
#endif
   print *
   print *, " Available real kernels are: "
#ifdef HAVE_AVX2
   print *, " AVX kernels are optimized for FMA (AVX2)"
#endif
   print *
   call print_options(e, "real_kernel")
   print *
   print *

   print *, "Information on ELPA2 complex case: "
   print *, "=============================== "
#ifdef HAVE_ENVIRONMENT_CHECKING
   print *, " choice via environment variable: yes"
   print *, " environment variable name      : ELPA_2STAGE_COMPLEX_KERNEL"
#else
   print *,  " choice via environment variable: no"
#endif
   print *
   print *, " Available complex kernels are: "
#ifdef HAVE_AVX2
   print *, " AVX kernels are optimized for FMA (AVX2)"
#endif
   print *
   call print_options(e, "complex_kernel")
   print *
   print *

   call elpa_deallocate(e)

   contains

     subroutine print_options(e, option_name)
       class(elpa_t), intent(inout) :: e
       character(len=*), intent(in) :: option_name
       integer :: i, option

       do i = 0, elpa_option_cardinality(option_name) - 1
         option = elpa_option_enumerate(option_name, i)
         if (e%can_set(option_name, option) == ELPA_OK) then
           print *, "  ", elpa_int_value_to_string(option_name, option)
         endif
       end do
     end subroutine

end program print_available_elpa2_kernels
