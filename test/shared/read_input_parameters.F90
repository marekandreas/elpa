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

  use output_types

  implicit none

  type input_options_t
    integer        :: datatype
    type(output_t) :: write_to_file
    logical        :: doSolveTridi, do1stage, do2stage, justHelpMessage, &
                      doCholesky, doInvertTrm, doTransposeMultiply
  end type

  interface read_input_parameters
    module procedure read_input_parameters_general
    module procedure read_input_parameters_traditional
  end interface

  contains

    subroutine parse_arguments(obj, command_line_argument, input_options)
      use elpa
      use precision
      use output_types

      implicit none

      type(input_options_t) :: input_options
      character(len=128)    :: command_line_argument
      class(elpa_t), intent(inout) :: obj
      integer :: value, success

      if (command_line_argument == "--help") then
        print *,"usage: elpa_tests [--help] [datatype={real|complex}] [na=number] [nev=number] "
        print *,"                  [nblk=size of block cyclic distribution] [--output_eigenvalues]"
        print *,"                  [--output_eigenvectors] [--real-kernel=name_of_kernel]"
        print *,"                  [--complex-kernel=name_of_kernel] [--use-gpu={0|1}]"
        print *,"                  [--use-qr={0,1}] [--tests={all|solve-tridi|1stage|2stage|cholesky&
            &|invert-triangular|transpose-mulitply}]"
        input_options%justHelpMessage=.true.
        return
      endif


      if (command_line_argument(1:11) == "--datatype=") then
        if (command_line_argument(12:15) == "real") then
          input_options%datatype=1
        else
          if (command_line_argument(12:18) == "complex") then
            input_options%datatype=2
          else
            print *,"datatype unknown! use either --datatype=real or --datatpye=complex"
            stop 1
          endif
        endif
      endif

      if (command_line_argument(1:3) == "na=") then
        read(command_line_argument(4:), *) value
        call obj%set("na", value)
      endif
      if (command_line_argument(1:4) == "nev=") then
        read(command_line_argument(5:), *) value
        call obj%set("nev", value)
      endif
      if (command_line_argument(1:5) == "nblk=") then
        read(command_line_argument(6:), *) value
        call obj%set("nblk", value)
      endif

      if (command_line_argument(1:21)   == "--output_eigenvectors") then
        input_options%write_to_file%eigenvectors = .true.
      endif

      if (command_line_argument(1:20)   == "--output_eigenvalues") then
        input_options%write_to_file%eigenvalues = .true.
      endif

      if (command_line_argument(1:14) == "--real-kernel=") then
        value = elpa_int_string_to_value("real_kernel", trim(command_line_argument(15:)), success)
        if (success /= ELPA_OK) then
          print *, "ERROR: Could not parse value for option --real-kernel"
          stop 1
        endif
        call obj%set("real_kernel", value)
      endif

      if (command_line_argument(1:17) == "--complex-kernel=") then
        value = elpa_int_string_to_value("complex_kernel", trim(command_line_argument(18:)), success)
        if (success /= ELPA_OK) then
          print *, "ERROR: Could not parse value for option --complex-kernel"
          stop 1
        endif
        call obj%set("complex_kernel", value)
      endif

      if (command_line_argument(1:9) == "--use-qr=") then
        read(command_line_argument(10:), *) value
        call obj%set("qr", value)
      endif

      if (command_line_argument(1:10) == "--use-gpu=") then
        read(command_line_argument(11:), *) value
        call obj%set("gpu", value)
      endif

      if (command_line_argument(1:8) == "--tests=") then
        if (command_line_argument(9:11) == "all") then
          input_options%doSolveTridi=.true.
          input_options%do1stage=.true.
          input_options%do2stage=.true.
          input_options%doCholesky=.true.
          input_options%doInvertTrm=.true.
          input_options%doTransposeMultiply=.true.
        else if (command_line_argument(9:19) == "solve-tride") then
          input_options%doSolveTridi=.true.
          input_options%do1stage=.false.
          input_options%do2stage=.false.
          input_options%doCholesky=.false.
          input_options%doInvertTrm=.false.
          input_options%doTransposeMultiply=.false.
        else if (command_line_argument(9:14) == "1stage") then
          input_options%doSolveTridi=.false.
          input_options%do1stage=.true.
          input_options%do2stage=.false.
          input_options%doCholesky=.false.
          input_options%doInvertTrm=.false.
          input_options%doTransposeMultiply=.false.
        else if (command_line_argument(9:14) == "2stage") then
          input_options%doSolveTridi=.false.
          input_options%do1stage=.false.
          input_options%do2stage=.true.
          input_options%doCholesky=.false.
          input_options%doInvertTrm=.false.
          input_options%doTransposeMultiply=.false.
        else if (command_line_argument(9:16) == "cholesky") then
          input_options%doSolveTridi=.false.
          input_options%do1stage=.false.
          input_options%do2stage=.false.
          input_options%doCholesky=.true.
          input_options%doInvertTrm=.false.
          input_options%doTransposeMultiply=.false.
        else if (command_line_argument(9:25) == "invert-triangular") then
          input_options%doSolveTridi=.false.
          input_options%do1stage=.false.
          input_options%do2stage=.false.
          input_options%doCholesky=.false.
          input_options%doInvertTrm=.true.
          input_options%doTransposeMultiply=.false.
        else if (command_line_argument(9:26) == "transpose-multiply") then
          input_options%doSolveTridi=.false.
          input_options%do1stage=.false.
          input_options%do2stage=.false.
          input_options%doCholesky=.false.
          input_options%doInvertTrm=.false.
          input_options%doTransposeMultiply=.true.
        else
           print *,"unknown test specified"
           stop 1
        endif
      endif

    end subroutine

    subroutine read_input_parameters_general(obj, input_options)
      use elpa
      use ELPA_utilities, only : error_unit
      use precision
      use elpa_mpi
      use output_types
      implicit none

      class(elpa_t), intent(inout) :: obj
      type(input_options_t)         :: input_options

      ! Command line arguments
      character(len=128)            :: arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10
      integer(kind=ik)              :: mpierr, kernels

      ! default parameters
      input_options%datatype = 1
      call obj%set("na", 4000)
      call obj%set("nev", 1500)
      call obj%set("nblk", 16)

      input_options%write_to_file%eigenvectors = .false.
      input_options%write_to_file%eigenvalues  = .false.

      input_options%do1Stage = .true.
      input_options%do2Stage = .true.
      input_options%doSolveTridi = .true.
      input_options%doCholesky=.true.
      input_options%doInvertTrm=.true.
      input_options%doTransposeMultiply=.true.
      input_options%justHelpMessage=.false.

      ! test na=1500 nev=50 nblk=16 --help --kernel --output_eigenvectors --output_eigenvalues
      if (COMMAND_ARGUMENT_COUNT() .gt. 8) then
        write(error_unit, '(a,i0,a)') "Invalid number (", COMMAND_ARGUMENT_COUNT(), ") of command line arguments!"
        stop 1
      endif

      if (COMMAND_ARGUMENT_COUNT() .gt. 0) then

        call get_COMMAND_ARGUMENT(1, arg1)

        call parse_arguments(obj, arg1, input_options)



        if (COMMAND_ARGUMENT_COUNT() .ge. 2) then
          ! argument 2
          call get_COMMAND_ARGUMENT(2, arg2)

          call parse_arguments(obj, arg2, input_options)
        endif

        ! argument 3
        if (COMMAND_ARGUMENT_COUNT() .ge. 3) then

          call get_COMMAND_ARGUMENT(3, arg3)

          call parse_arguments(obj, arg3, input_options)
        endif

        ! argument 4
        if (COMMAND_ARGUMENT_COUNT() .ge. 4) then

          call get_COMMAND_ARGUMENT(4, arg4)

          call parse_arguments(obj, arg4, input_options)

        endif

        ! argument 5
        if (COMMAND_ARGUMENT_COUNT() .ge. 5) then

          call get_COMMAND_ARGUMENT(5, arg5)

          call parse_arguments(obj, arg5, input_options)
        endif

        ! argument 6
        if (COMMAND_ARGUMENT_COUNT() .ge. 6) then

          call get_COMMAND_ARGUMENT(6, arg6)

          call parse_arguments(obj, arg6, input_options)
        endif

        ! argument 7
        if (COMMAND_ARGUMENT_COUNT() .ge. 7) then

          call get_COMMAND_ARGUMENT(7, arg7)

          call parse_arguments(obj, arg7, input_options)

        endif

        ! argument 8
        if (COMMAND_ARGUMENT_COUNT() .ge. 8) then

          call get_COMMAND_ARGUMENT(8, arg8)

          call parse_arguments(obj, arg8, input_options)

        endif

        ! argument 9
        if (COMMAND_ARGUMENT_COUNT() .ge. 9) then

          call get_COMMAND_ARGUMENT(9, arg9)

          call parse_arguments(obj, arg8, input_options)

        endif

        ! argument 10
        if (COMMAND_ARGUMENT_COUNT() .ge. 10) then

          call get_COMMAND_ARGUMENT(10, arg10)

          call parse_arguments(obj, arg8, input_options)

        endif

      endif

      if (obj%is_set("qr") == 1 .and. input_options%datatype .eq. 2) then
        print *,"You cannot use QR-decomposition in complex case"
        stop 1
      endif

    end subroutine

    subroutine read_input_parameters_traditional(na, nev, nblk, write_to_file)
      use ELPA_utilities, only : error_unit
      use precision
      use elpa_mpi
      use output_types
      implicit none

      integer(kind=ik), intent(out) :: na, nev, nblk

      type(output_t), intent(out)   :: write_to_file

      ! Command line arguments
      character(len=128)            :: arg1, arg2, arg3, arg4, arg5
      integer(kind=ik)              :: mpierr

      ! default parameters
      na = 4000
      nev = 1500
      nblk = 16
      write_to_file%eigenvectors = .false.
      write_to_file%eigenvalues  = .false.

      if (.not. any(COMMAND_ARGUMENT_COUNT() == [0, 3, 4, 5])) then
        write(error_unit, '(a,i0,a)') "Invalid number (", COMMAND_ARGUMENT_COUNT(), ") of command line arguments!"
        write(error_unit, *) "Expected: program [ [matrix_size num_eigenvalues block_size] &
            ""output_eigenvalues"" ""output_eigenvectors""]"
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

        if (arg4 .eq. "output_eigenvalues") then
          write_to_file%eigenvalues = .true.
        else
          write(error_unit, *) "Invalid value for output flag! Must be ""output_eigenvalues"" or omitted"
          stop 1
        endif

      endif

      if (COMMAND_ARGUMENT_COUNT() == 5) then
        call GET_COMMAND_ARGUMENT(1, arg1)
        call GET_COMMAND_ARGUMENT(2, arg2)
        call GET_COMMAND_ARGUMENT(3, arg3)
        call GET_COMMAND_ARGUMENT(4, arg4)
        call GET_COMMAND_ARGUMENT(5, arg5)
        read(arg1, *) na
        read(arg2, *) nev
        read(arg3, *) nblk

        if (arg4 .eq. "output_eigenvalues") then
          write_to_file%eigenvalues = .true.
        else
          write(error_unit, *) "Invalid value for output flag! Must be ""output_eigenvalues"" or omitted"
          stop 1
        endif

        if (arg5 .eq. "output_eigenvectors") then
          write_to_file%eigenvectors = .true.
        else
          write(error_unit, *) "Invalid value for output flag! Must be ""output_eigenvectors"" or omitted"
          stop 1
        endif

      endif
    end subroutine

end module
