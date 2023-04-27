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
#undef TEST_INT_TYPE
#undef INT_TYPE
#undef TEST_INT_MPI_TYPE
#undef INT_MPI_TYPE

#ifdef HAVE_64BIT_INTEGER_MATH_SUPPORT
#define TEST_INT_TYPE integer(kind=c_int64_t)
#define INT_TYPE c_int64_t
#else
#define TEST_INT_TYPE integer(kind=c_int32_t)
#define INT_TYPE c_int32_t
#endif
#ifdef HAVE_64BIT_INTEGER_MPI_SUPPORT
#define TEST_INT_MPI_TYPE integer(kind=c_int64_t)
#define INT_MPI_TYPE c_int64_t
#else
#define TEST_INT_MPI_TYPE integer(kind=c_int32_t)
#define INT_MPI_TYPE c_int32_t
#endif

module test_util
  use iso_c_binding
  use precision_for_tests
#ifdef WITH_MPI
#ifdef HAVE_MPI_MODULE
  use mpi
  implicit none
#else
  implicit none
  include 'mpif.h'
#endif
#else
  TEST_INT_MPI_TYPE, parameter :: mpi_comm_world = -1
#endif

  interface is_infinity_or_NaN
    module procedure is_infinity_or_NaN_double
#if defined(WANT_SINGLE_PRECISION_REAL) || defined(WANT_SINGLE_PRECISION_COMPLEX)
    module procedure is_infinity_or_NaN_single
#endif
  end interface

  contains
!>
!> This function translates, if ELPA was build with OpenMP support,
!> the found evel of "thread safetiness" from the internal number
!> of the MPI library into a human understandable value
!>
!> \param level thread-saftiness of the MPI library
!> \return str human understandable value of thread saftiness
  pure function mpi_thread_level_name(level) result(str)
    use, intrinsic :: iso_c_binding
    implicit none
    integer(kind=c_int), intent(in) :: level
    character(len=21)            :: str
#ifdef WITH_MPI
    select case(level)
      case (MPI_THREAD_SINGLE)
        str = "MPI_THREAD_SINGLE"
      case (MPI_THREAD_FUNNELED)
        str = "MPI_THREAD_FUNNELED"
      case (MPI_THREAD_SERIALIZED)
        str = "MPI_THREAD_SERIALIZED"
      case (MPI_THREAD_MULTIPLE)
        str = "MPI_THREAD_MULTIPLE"
      case default
        write(str,'(i0,1x,a)') level, "(Unknown level)"
    end select
#endif
  end function

  function seconds() result(s)
    integer :: ticks, tick_rate
    real(kind=c_double) :: s

    call system_clock(count=ticks, count_rate=tick_rate)
    s = real(ticks, kind=c_double) / tick_rate
  end function

    subroutine x_a(condition, condition_string, file, line)
#ifdef HAVE_ISO_FORTRAN_ENV
      use iso_fortran_env, only : error_unit
#endif
      implicit none
#ifndef HAVE_ISO_FORTRAN_ENV
      integer, parameter :: error_unit = 0
#endif
      logical, intent(in) :: condition
      character(len=*), intent(in) :: condition_string
      character(len=*), intent(in) :: file
      integer, intent(in) :: line

      if (.not. condition) then
        write(error_unit,'(a,i0)') "Assertion `" // condition_string // "` failed at " // file // ":", line
        stop 1
      end if
    end subroutine

    subroutine x_ao(error_code, error_code_string, file, line)
      use elpa
#ifdef HAVE_ISO_FORTRAN_ENV
      use iso_fortran_env, only : error_unit
#endif
      implicit none
#ifndef HAVE_ISO_FORTRAN_ENV
      integer, parameter :: error_unit = 0
#endif
      integer, intent(in) :: error_code
      character(len=*), intent(in) :: error_code_string
      character(len=*), intent(in) :: file
      integer, intent(in) :: line

      if (error_code /= ELPA_OK) then
        write(error_unit,'(a,i0)') "Assertion failed: `" // error_code_string // &
           " is " // elpa_strerr(error_code) // "` at " // file // ":", line
        stop 1
      end if
    end subroutine


    function is_infinity_or_NaN_double(x) result(result_infinity_or_NaN)
      implicit none
      real(kind=c_double) :: x
      logical :: result_infinity_or_NaN
  
      result_infinity_or_NaN = x/=x .or. x>huge(x) .or. x<-huge(x)
    end function

    function is_infinity_or_NaN_single(x) result(result_infinity_or_NaN)
      implicit none
      real(kind=c_float) :: x
      logical :: result_infinity_or_NaN
  
      result_infinity_or_NaN = x/=x .or. x>huge(x) .or. x<-huge(x)
    end function

end module

