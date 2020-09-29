!    Copyright 2013, A. Marek
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
!
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
! Author: P. Kus, MPCDF

#include "config-f90.h"

module timings_dummy
  implicit none
  
  type, public :: timer_dummy_t
      contains
      procedure, pass :: start => timer_start
      procedure, pass :: stop => timer_stop
      procedure, pass :: enable => timer_enable
      procedure, pass :: free => timer_free
      procedure, pass :: print => timer_print
      procedure, pass :: measure_flops => timer_measure_flops
      procedure, pass :: set_print_options => timer_set_print_options
  end type 

  type(timer_dummy_t) :: timer
  type(timer_dummy_t) :: autotune_timer

  contains

  subroutine timer_print(self, name)
    class(timer_dummy_t), intent(inout), target :: self
    character(len=*), intent(in)  :: name
    
  end subroutine

  subroutine timer_start(self, name, replace)
    class(timer_dummy_t), intent(inout), target :: self
    character(len=*), intent(in)  :: name
    logical, intent(in), optional  :: replace
    
  end subroutine
  
  subroutine timer_stop(self, name)
    class(timer_dummy_t), intent(inout), target :: self
    character(len=*), intent(in), optional :: name
    
  end subroutine

  subroutine timer_enable(self)
    class(timer_dummy_t), intent(inout), target :: self
    
  end subroutine

  subroutine timer_measure_flops(self, enable)
    class(timer_dummy_t), intent(inout), target :: self
    logical                                     :: enable
  end subroutine

  subroutine timer_set_print_options(self, print_allocated_memory, &
        print_virtual_memory, &
        print_max_allocated_memory, &
        print_flop_count, &
        print_flop_rate, &
        print_ldst, &
        print_memory_bandwidth, &
        print_ai, &
        bytes_per_ldst)

    class(timer_dummy_t), intent(inout), target :: self
    logical, intent(in), optional :: &
        print_allocated_memory, &
        print_virtual_memory, &
        print_max_allocated_memory, &
        print_flop_count, &
        print_flop_rate, &
        print_ldst, &
        print_memory_bandwidth, &
        print_ai
    integer, intent(in), optional :: bytes_per_ldst
  end subroutine

  subroutine timer_free(self)
    class(timer_dummy_t), intent(inout), target :: self
    
  end subroutine
end module timings_dummy
