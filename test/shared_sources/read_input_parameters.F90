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
!    http://elpa.rzg.mpg.de/
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
#include "config-f90.h"
module mod_read_input_parameters

  contains

    subroutine read_input_parameters(na, nev, nblk, write_to_file)
      use ELPA_utilities, only : error_unit
      implicit none
      include 'mpif.h'

      integer, intent(out) :: na, nev, nblk
      logical, intent(out) :: write_to_file

      ! Command line arguments
      character(len=128)   :: arg1, arg2, arg3, arg4
      integer :: mpierr

      ! default parameters
      na = 4000
      nev = 1500
      nblk = 16
      write_to_file = .false.

      if (.not. any(COMMAND_ARGUMENT_COUNT() == [0, 3, 4])) then
        write(error_unit, '(a,i0,a)') "Invalid number (", COMMAND_ARGUMENT_COUNT(), ") of command line arguments!"
        write(error_unit, *) "Expected: program [ [matrix_size num_eigenvalues block_size] ""output""]"
        stop 1
      endif

      if (COMMAND_ARGUMENT_COUNT() == 3) then
        call GET_COMMAND_ARGUMENT(1, arg1)
        call GET_COMMAND_ARGUMENT(2, arg2)
        call GET_COMMAND_ARGUMENT(3, arg3)

        read(arg1, *) na
        read(arg2, *) nev
        read(arg3, *) nblk
      endif

      if (COMMAND_ARGUMENT_COUNT() == 4) then
        call GET_COMMAND_ARGUMENT(1, arg1)
        call GET_COMMAND_ARGUMENT(2, arg2)
        call GET_COMMAND_ARGUMENT(3, arg3)
        call GET_COMMAND_ARGUMENT(4, arg4)
        read(arg1, *) na
        read(arg2, *) nev
        read(arg3, *) nblk

        if (arg4 .eq. "output") then
          write_to_file = .true.
        else
          write(error_unit, *) "Invalid value for output flag! Must be ""output"" or omitted"
          stop 1
        endif

      endif

    end subroutine

end module
