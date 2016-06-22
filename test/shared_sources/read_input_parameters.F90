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
#include "config-f90.h"
module mod_read_input_parameters

  contains

    subroutine parse_arguments(command_line_argument, na, nev, nblk, write_to_file, &
                               this_real_kernel, this_complex_kernel, realKernelSet, complexKernelSet)

      use elpa2_utilities
      use precision
      use output_types

      implicit none

      integer(kind=ik)             :: na, nev, nblk
      type(output_t)               :: write_to_file
      integer                      :: this_real_kernel, this_complex_kernel
      logical                      :: realKernelSet, complexKernelSet
      character(len=128)           :: command_line_argument

      integer(kind=ik)             :: kernels

      if (command_line_argument == "--help") then
        print *,"usage: elpa_unified_test_real | elpa_unified_test_complex [--help] [na=number] [nev=number] "
        print *,"                                                        [nblk=number] [--output_eigenvalues]"
        print *,"                                      [--output_eigenvectors] [--real-kernel=name_of_kernel]"
        print *,"                                      [--complex-kernel=name_of_kernel]"
      endif

      if (command_line_argument(1:3) == "na=") then
        read(command_line_argument(4:), *) na
      endif
      if (command_line_argument(1:4) == "nev=") then
        read(command_line_argument(5:), *) nev
      endif
      if (command_line_argument(1:5) == "nblk=") then
        read(command_line_argument(6:), *) nblk
      endif


      if (command_line_argument(1:21)   == "--output_eigenvectors") then
        write_to_file%eigenvectors = .true.
      endif

      if (command_line_argument(1:20)   == "--output_eigenvalues") then
        write_to_file%eigenvalues = .true.
      endif

      if (command_line_argument(1:14) == "--real-kernel=") then
        do kernels = 1, elpa_number_of_real_kernels()
          if (  trim(command_line_argument(15:)) .eq. elpa_real_kernel_name(kernels)) then
            this_real_kernel = kernels
            print *,"Setting ELPA2 real kernel to ",elpa_real_kernel_name(kernels)
            realKernelSet = .true.
          endif
        enddo
      endif

      if (command_line_argument(1:17) == "--complex-kernel=") then
        do kernels = 1, elpa_number_of_complex_kernels()
          if (  trim(command_line_argument(18:)) .eq. elpa_complex_kernel_name(kernels)) then
            this_complex_kernel = kernels
            print *,"Setting ELPA2 complex kernel to ",elpa_complex_kernel_name(kernels)
            realKernelSet = .true.
          endif
        enddo
      endif

    end subroutine

    subroutine read_input_parameters(na, nev, nblk, write_to_file, this_real_kernel, this_complex_kernel, realKernelSet, &
                                     complexKernelSet)
      use ELPA_utilities, only : error_unit
      use precision
      use elpa_mpi
      use elpa2_utilities
      use output_types
      implicit none

      integer(kind=ik), intent(out) :: na, nev, nblk

      type(output_t), intent(out)   :: write_to_file

      ! Command line arguments
      character(len=128)            :: arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8
      integer(kind=ik)              :: mpierr, kernels, this_real_kernel, this_complex_kernel
      logical                       :: realKernelSet, complexKernelSet

      ! default parameters
      na = 4000
      nev = 1500
      nblk = 16
      write_to_file%eigenvectors = .false.
      write_to_file%eigenvalues  = .false.
      this_real_kernel = DEFAULT_REAL_ELPA_KERNEL
      this_complex_kernel = DEFAULT_COMPLEX_ELPA_KERNEL
      realKernelSet = .false.
      complexKernelSet = .false.

      ! test na=1500 nev=50 nblk=16 --help --kernel --output_eigenvectors --output_eigenvalues
      if (COMMAND_ARGUMENT_COUNT() .gt. 8) then
        write(error_unit, '(a,i0,a)') "Invalid number (", COMMAND_ARGUMENT_COUNT(), ") of command line arguments!"
        stop 1
      endif

      if (COMMAND_ARGUMENT_COUNT() .gt. 0) then

        call get_COMMAND_ARGUMENT(1, arg1)

        call parse_arguments(arg1, na, nev, nblk, write_to_file, &
                               this_real_kernel, this_complex_kernel, realKernelSet, complexKernelSet)



        if (COMMAND_ARGUMENT_COUNT() .ge. 2) then
          ! argument 2
          call get_COMMAND_ARGUMENT(2, arg2)

          call parse_arguments(arg2, na, nev, nblk, write_to_file, &
                               this_real_kernel, this_complex_kernel, realKernelSet, complexKernelSet)
        endif

        ! argument 3
        if (COMMAND_ARGUMENT_COUNT() .ge. 3) then

          call get_COMMAND_ARGUMENT(3, arg3)

          call parse_arguments(arg3, na, nev, nblk, write_to_file, &
                               this_real_kernel, this_complex_kernel, realKernelSet, complexKernelSet)
        endif

        ! argument 4
        if (COMMAND_ARGUMENT_COUNT() .ge. 4) then

          call get_COMMAND_ARGUMENT(4, arg4)

          call parse_arguments(arg4, na, nev, nblk, write_to_file, &
                               this_real_kernel, this_complex_kernel, realKernelSet, complexKernelSet)

        endif

        ! argument 5
        if (COMMAND_ARGUMENT_COUNT() .ge. 5) then

          call get_COMMAND_ARGUMENT(5, arg5)

          call parse_arguments(arg5, na, nev, nblk, write_to_file, &
                               this_real_kernel, this_complex_kernel, realKernelSet, complexKernelSet)
        endif

        ! argument 6
        if (COMMAND_ARGUMENT_COUNT() .ge. 6) then

          call get_COMMAND_ARGUMENT(6, arg6)

          call parse_arguments(arg6, na, nev, nblk, write_to_file, &
                               this_real_kernel, this_complex_kernel, realKernelSet, complexKernelSet)
        endif

        ! argument 7
        if (COMMAND_ARGUMENT_COUNT() .ge. 7) then

          call get_COMMAND_ARGUMENT(7, arg7)

          call parse_arguments(arg7, na, nev, nblk, write_to_file, &
                               this_real_kernel, this_complex_kernel, realKernelSet, complexKernelSet)

        endif

        ! argument 8
        if (COMMAND_ARGUMENT_COUNT() .ge. 8) then

          call get_COMMAND_ARGUMENT(8, arg8)

          call parse_arguments(arg8, na, nev, nblk, write_to_file, &
                               this_real_kernel, this_complex_kernel, realKernelSet, complexKernelSet)

        endif
      endif
    end subroutine

end module
