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
  use elpa
  use, intrinsic :: iso_c_binding

  implicit none

  integer(kind=c_int) :: i
  class(elpa_t), pointer :: e
  integer :: option, error

  if (elpa_init(CURRENT_API_VERSION) /= ELPA_OK) then
    print *, "Unsupported ELPA API Version"
    stop 1
  endif

  e => elpa_allocate(error)

  print *, "This program will give information on the ELPA2 kernels, "
  print *, "which are available with this library and it will give "
  print *, "information if (and how) the kernels can be choosen at "
  print *, "runtime"
  print *
#ifdef WITH_OPENMP_TRADITIONAL
  print *, " ELPA supports threads: yes"
#else
  print *, " ELPA supports threads: no"
#endif
  print *

  print *, "Information on ELPA2 real case: "
  print *, "=============================== "
#ifdef HAVE_ENVIRONMENT_CHECKING
  print *, " choice via environment variable: yes"
  print *, " environment variable name      : ELPA_DEFAULT_real_kernel"
#else
  print *, " choice via environment variable: no"
#endif
  print *
  print *, " Available real kernels are: "
  print *
  call print_options(e, "real_kernel")
  print *
  print *

  print *, "Information on ELPA2 complex case: "
  print *, "=============================== "
#ifdef HAVE_ENVIRONMENT_CHECKING
  print *, " choice via environment variable: yes"
  print *, " environment variable name      : ELPA_DEFAULT_complex_kernel"
#else
  print *,  " choice via environment variable: no"
#endif
  print *
  print *, " Available complex kernels are: "
  print *
  call print_options(e, "complex_kernel")
  print *
  print *

  call elpa_deallocate(e, error)

  contains

    subroutine print_options(e, KERNEL_KEY)
      class(elpa_t), intent(inout) :: e
      character(len=*), intent(in) :: KERNEL_KEY
      integer                      :: i, kernel,error

      call e%set("solver",ELPA_SOLVER_2STAGE,error)

      do i = 0, elpa_option_cardinality(KERNEL_KEY)
        kernel = elpa_option_enumerate(KERNEL_KEY, i)
        if (elpa_int_value_to_string(KERNEL_KEY, kernel) .eq. "ELPA_2STAGE_COMPLEX_GPU" .or. &
            elpa_int_value_to_string(KERNEL_KEY, kernel) .eq. "ELPA_2STAGE_REAL_GPU") then
          if (e%can_set("gpu",1) == ELPA_OK) then
            call e%set("gpu",1, error)
          endif
        endif 
        if (elpa_int_value_to_string(KERNEL_KEY, kernel) .eq. "ELPA_2STAGE_COMPLEX_NVIDIA_GPU" .or. &
            elpa_int_value_to_string(KERNEL_KEY, kernel) .eq. "ELPA_2STAGE_REAL_NVIDIA_GPU") then
          if (e%can_set("nvidia-gpu",1) == ELPA_OK) then
            call e%set("nvidia-gpu",1, error)
          endif
        endif 
        if (elpa_int_value_to_string(KERNEL_KEY, kernel) .eq. "ELPA_2STAGE_COMPLEX_AMD_GPU" .or. &
            elpa_int_value_to_string(KERNEL_KEY, kernel) .eq. "ELPA_2STAGE_REAL_AMD_GPU") then
          if (e%can_set("amd-gpu",1) == ELPA_OK) then
            call e%set("amd-gpu",1, error)
          endif
        endif 

        if (e%can_set(KERNEL_KEY, kernel) == ELPA_OK) then
          if (elpa_int_value_to_string(KERNEL_KEY, kernel) .eq. "ELPA_2STAGE_COMPLEX_GPU" .or. &
              elpa_int_value_to_string(KERNEL_KEY, kernel) .eq. "ELPA_2STAGE_REAL_GPU" .or. &
              elpa_int_value_to_string(KERNEL_KEY, kernel) .eq. "ELPA_2STAGE_COMPLEX_NVIDIA_GPU" .or. &
              elpa_int_value_to_string(KERNEL_KEY, kernel) .eq. "ELPA_2STAGE_REAL_NVIDIA_GPU" .or. &
              elpa_int_value_to_string(KERNEL_KEY, kernel) .eq. "ELPA_2STAGE_COMPLEX_AMD_GPU" .or. &
              elpa_int_value_to_string(KERNEL_KEY, kernel) .eq. "ELPA_2STAGE_REAL_AMD_GPU" .or. &
              elpa_int_value_to_string(KERNEL_KEY, kernel) .eq. "ELPA_2STAGE_COMPLEX_INTEL_GPU" .or. &
              elpa_int_value_to_string(KERNEL_KEY, kernel) .eq. "ELPA_2STAGE_REAL_INTEL_GPU" ) then
              print *,"  ",elpa_int_value_to_string(KERNEL_KEY, kernel), &
                      "  GPU kernel (might not be usable if no GPUs present on the host)"
          else
            print *, "  ", elpa_int_value_to_string(KERNEL_KEY, kernel)
          endif
        endif
      end do
    end subroutine

end program print_available_elpa2_kernels
