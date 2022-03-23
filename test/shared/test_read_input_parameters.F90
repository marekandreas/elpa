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


#ifdef HAVE_64BIT_INTEGER_MATH_SUPPORT
#define TEST_INT_TYPE integer(kind=c_int64_t)
#define INT_TYPE c_int64_t
#else
#define TEST_INT_TYPE integer(kind=c_int32_t)
#define INT_TYPE c_int32_t
#endif

module test_read_input_parameters
  use elpa, only : ELPA_2STAGE_COMPLEX_DEFAULT, ELPA_2STAGE_REAL_DEFAULT, elpa_int_string_to_value, &
                   elpa_int_value_to_string, ELPA_OK
  use elpa_utilities, only : error_unit
  use iso_c_binding
  use test_util, only : x_ao, x_a
  use test_output_type

  implicit none

  type input_options_t
    TEST_INT_TYPE        :: datatype
    TEST_INT_TYPE        :: na, nev, nblk
    type(output_t) :: write_to_file
    TEST_INT_TYPE        :: this_real_kernel, this_complex_kernel
    logical        :: realKernelIsSet, complexKernelIsSet
    TEST_INT_TYPE        :: useQrIsSet, useGPUIsSet
    logical        :: doSolveTridi, do1stage, do2stage, justHelpMessage, &
                      doCholesky, doInvertTrm, doTransposeMultiply
  end type

  interface read_input_parameters
    module procedure read_input_parameters_general
    module procedure read_input_parameters_traditional
    module procedure read_input_parameters_traditional_noskip
  end interface

  contains

    subroutine parse_arguments(command_line_argument, input_options)
      implicit none

      type(input_options_t)  :: input_options
      character(len=128)     :: command_line_argument
      integer(kind=c_int)    :: elpa_error

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
        read(command_line_argument(4:), *) input_options%na
      endif
      if (command_line_argument(1:4) == "nev=") then
        read(command_line_argument(5:), *) input_options%nev
      endif
      if (command_line_argument(1:5) == "nblk=") then
        read(command_line_argument(6:), *) input_options%nblk
      endif

      if (command_line_argument(1:21)   == "--output_eigenvectors") then
        input_options%write_to_file%eigenvectors = .true.
      endif

      if (command_line_argument(1:20)   == "--output_eigenvalues") then
        input_options%write_to_file%eigenvalues = .true.
      endif

      if (command_line_argument(1:14) == "--real-kernel=") then
        input_options%this_real_kernel = int(elpa_int_string_to_value("real_kernel",     &
                                             command_line_argument(15:), elpa_error), &
                                             kind=INT_TYPE)
        if (elpa_error /= ELPA_OK) then
          print *, "Invalid argument for --real-kernel"
          stop 1
        endif
#ifndef ENABLE_IFX_COMPILER
        print *,"Setting ELPA2 real kernel to ", elpa_int_value_to_string("real_kernel", &
                                                                        int(input_options%this_real_kernel,kind=c_int))
#endif
        input_options%realKernelIsSet = .true.
      endif

      if (command_line_argument(1:17) == "--complex-kernel=") then
        input_options%this_complex_kernel = int(elpa_int_string_to_value("complex_kernel",    &
                                                command_line_argument(18:), elpa_error), kind=INT_TYPE)
        if (elpa_error /= ELPA_OK) then
          print *, "Invalid argument for --complex-kernel"
          stop 1
        endif
#ifndef ENABLE_IFX_COMPILER
        print *,"Setting ELPA2 complex kernel to ", elpa_int_value_to_string("complex_kernel", &
                                                                          int(input_options%this_complex_kernel,kind=c_int))
#endif
        input_options%complexKernelIsSet = .true.
      endif

      if (command_line_argument(1:9) == "--use-qr=") then
        read(command_line_argument(10:), *) input_options%useQrIsSet
      endif

      if (command_line_argument(1:10) == "--use-gpu=") then
        read(command_line_argument(11:), *) input_options%useGPUIsSet
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

    subroutine read_input_parameters_general(input_options)
      use precision_for_tests
      implicit none

      type(input_options_t)         :: input_options

      ! Command line arguments
      character(len=128)            :: arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10

      ! default parameters
      input_options%datatype = 1
      input_options%na = 500
      input_options%nev = 150
      input_options%nblk = 16

      input_options%write_to_file%eigenvectors = .false.
      input_options%write_to_file%eigenvalues  = .false.

      input_options%this_real_kernel = ELPA_2STAGE_REAL_DEFAULT
      input_options%this_complex_kernel = ELPA_2STAGE_COMPLEX_DEFAULT
      input_options%realKernelIsSet = .false.
      input_options%complexKernelIsSet = .false.

      input_options%useQrIsSet = 0

      input_options%useGPUIsSet = 0

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

        call parse_arguments(arg1, input_options)



        if (COMMAND_ARGUMENT_COUNT() .ge. 2) then
          ! argument 2
          call get_COMMAND_ARGUMENT(2, arg2)

          call parse_arguments(arg2, input_options)
        endif

        ! argument 3
        if (COMMAND_ARGUMENT_COUNT() .ge. 3) then

          call get_COMMAND_ARGUMENT(3, arg3)

          call parse_arguments(arg3, input_options)
        endif

        ! argument 4
        if (COMMAND_ARGUMENT_COUNT() .ge. 4) then

          call get_COMMAND_ARGUMENT(4, arg4)

          call parse_arguments(arg4, input_options)

        endif

        ! argument 5
        if (COMMAND_ARGUMENT_COUNT() .ge. 5) then

          call get_COMMAND_ARGUMENT(5, arg5)

          call parse_arguments(arg5, input_options)
        endif

        ! argument 6
        if (COMMAND_ARGUMENT_COUNT() .ge. 6) then

          call get_COMMAND_ARGUMENT(6, arg6)

          call parse_arguments(arg6, input_options)
        endif

        ! argument 7
        if (COMMAND_ARGUMENT_COUNT() .ge. 7) then

          call get_COMMAND_ARGUMENT(7, arg7)

          call parse_arguments(arg7, input_options)

        endif

        ! argument 8
        if (COMMAND_ARGUMENT_COUNT() .ge. 8) then

          call get_COMMAND_ARGUMENT(8, arg8)

          call parse_arguments(arg8, input_options)

        endif

        ! argument 9
        if (COMMAND_ARGUMENT_COUNT() .ge. 9) then

          call get_COMMAND_ARGUMENT(9, arg9)

          call parse_arguments(arg8, input_options)

        endif

        ! argument 10
        if (COMMAND_ARGUMENT_COUNT() .ge. 10) then

          call get_COMMAND_ARGUMENT(10, arg10)

          call parse_arguments(arg8, input_options)

        endif

      endif

      if (input_options%useQrIsSet .eq. 1 .and. input_options%datatype .eq. 2) then
        print *,"You cannot use QR-decomposition in complex case"
        stop 1
      endif

    end subroutine

    subroutine read_input_parameters_traditional_noskip(na, nev, nblk, write_to_file)
      use precision_for_tests
      implicit none

      TEST_INT_TYPE, intent(out) :: na, nev, nblk

      type(output_t), intent(out)   :: write_to_file
      logical                       :: skip_check_correctness

      call read_input_parameters_traditional(na, nev, nblk, write_to_file, skip_check_correctness)
    end subroutine

    subroutine read_input_parameters_traditional(na, nev, nblk, write_to_file, skip_check_correctness)
      use precision_for_tests
      implicit none

      TEST_INT_TYPE, intent(out) :: na, nev, nblk

      type(output_t), intent(out)   :: write_to_file
      logical, intent(out)          :: skip_check_correctness

      ! Command line arguments
      character(len=128)            :: arg1, arg2, arg3, arg4, arg5

      ! default parameters
      na = 5000
      nev = 150
      nblk = 16
      write_to_file%eigenvectors = .false.
      write_to_file%eigenvalues  = .false.
      skip_check_correctness = .false.

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
        elseif (arg4 .eq. "skip_check_correctness") then
          skip_check_correctness = .true.
        else
          write(error_unit, *) &
          "Invalid value for parameter 4.  Must be ""output_eigenvalues"", ""skip_check_correctness"" or omitted"
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
