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

#define stringify_(x) "x"
#define stringify(x) stringify_(x)
#define assert(x) call x_a(x, stringify(x), __FILE__, __LINE__)

#define assert_elpa_ok(error_code) call x_ao(error_code, stringify(error_code), __FILE__, __LINE__)

! vim: syntax=fortran
