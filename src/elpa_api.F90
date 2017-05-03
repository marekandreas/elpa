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

module elpa_api
  use elpa_constants
  use, intrinsic :: iso_c_binding
  implicit none

  integer, private, parameter :: earliest_api_version = EARLIEST_API_VERSION
  integer, private, parameter :: current_api_version  = CURRENT_API_VERSION
  logical, private :: initDone = .false.

  public :: elpa_t, &
      c_int, &
      c_double, c_double_complex, &
      c_float, c_float_complex

  type, abstract :: elpa_t
    private

    ! \todo: it's ugly that these are public
    integer(kind=c_int), public, pointer :: na => NULL()
    integer(kind=c_int), public, pointer :: nev => NULL()
    integer(kind=c_int), public, pointer :: local_nrows => NULL()
    integer(kind=c_int), public, pointer :: local_ncols => NULL()
    integer(kind=c_int), public, pointer :: nblk => NULL()

    contains
      ! general
      procedure(elpa_setup_i),   deferred, public :: setup
      procedure(elpa_destroy_i), deferred, public :: destroy

      ! key/value store
      generic, public :: set => &
          elpa_set_integer, &
          elpa_set_double
      procedure(elpa_get_integer_i), deferred, public :: get
      procedure(elpa_get_double_i),  deferred, public :: get_double
      procedure(elpa_is_set_i),      deferred, public :: is_set
      procedure(elpa_print_options_i), deferred, public :: print_options

      ! Some parameters can be overridden by environment variables
      procedure(elpa_get_int_i), deferred, private :: get_real_kernel
      procedure(elpa_get_int_i), deferred, private :: get_complex_kernel

      ! actual math routines
      generic, public :: solve => &
          elpa_solve_real_double, &
          elpa_solve_real_single, &
          elpa_solve_complex_double, &
          elpa_solve_complex_single

      generic, public :: hermitian_multiply => &
          elpa_multiply_at_b_double, &
          elpa_multiply_ah_b_double, &
          elpa_multiply_at_b_single, &
          elpa_multiply_ah_b_single

      generic, public :: cholesky => &
          elpa_cholesky_double_real, &
          elpa_cholesky_single_real, &
          elpa_cholesky_double_complex, &
          elpa_cholesky_single_complex

      generic, public :: invert_tridiagonal => &
          elpa_invert_trm_double_real, &
          elpa_invert_trm_single_real, &
          elpa_invert_trm_double_complex, &
          elpa_invert_trm_single_complex

      generic, public :: solve_tridi => &
          elpa_solve_tridi_double_real, &
          elpa_solve_tridi_single_real


      ! privates
      procedure(elpa_set_integer_i), deferred, private :: elpa_set_integer
      procedure(elpa_set_double_i),  deferred, private :: elpa_set_double

      procedure(elpa_solve_real_double_i),    deferred, private :: elpa_solve_real_double
      procedure(elpa_solve_real_single_i),    deferred, private :: elpa_solve_real_single
      procedure(elpa_solve_complex_double_i), deferred, private :: elpa_solve_complex_double
      procedure(elpa_solve_complex_single_i), deferred, private :: elpa_solve_complex_single

      procedure(elpa_multiply_at_b_double_i), deferred, private :: elpa_multiply_at_b_double
      procedure(elpa_multiply_at_b_single_i), deferred, private :: elpa_multiply_at_b_single
      procedure(elpa_multiply_ah_b_double_i), deferred, private :: elpa_multiply_ah_b_double
      procedure(elpa_multiply_ah_b_single_i), deferred, private :: elpa_multiply_ah_b_single

      procedure(elpa_cholesky_double_real_i),    deferred, private :: elpa_cholesky_double_real
      procedure(elpa_cholesky_single_real_i),    deferred, private :: elpa_cholesky_single_real
      procedure(elpa_cholesky_double_complex_i), deferred, private :: elpa_cholesky_double_complex
      procedure(elpa_cholesky_single_complex_i), deferred, private :: elpa_cholesky_single_complex

      procedure(elpa_invert_trm_double_real_i),    deferred, private :: elpa_invert_trm_double_real
      procedure(elpa_invert_trm_single_real_i),    deferred, private :: elpa_invert_trm_single_real
      procedure(elpa_invert_trm_double_complex_i), deferred, private :: elpa_invert_trm_double_complex
      procedure(elpa_invert_trm_single_complex_i), deferred, private :: elpa_invert_trm_single_complex

      procedure(elpa_solve_tridi_double_real_i), deferred, private :: elpa_solve_tridi_double_real
      procedure(elpa_solve_tridi_single_real_i), deferred, private :: elpa_solve_tridi_single_real
  end type elpa_t


  interface
    pure function elpa_strlen_c(ptr) result(size) bind(c, name="strlen")
      use, intrinsic :: iso_c_binding
      type(c_ptr), intent(in), value :: ptr
      integer(kind=c_size_t) :: size
    end function
  end interface


  abstract interface
    function elpa_setup_i(self) result(success)
      import elpa_t
      class(elpa_t), intent(inout) :: self
      integer :: success
    end function
  end interface


  abstract interface
    subroutine elpa_set_integer_i(self, name, value, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
      character(*), intent(in)        :: name
      integer(kind=c_int), intent(in) :: value
      integer, optional               :: success
    end subroutine
  end interface


  abstract interface
    function elpa_get_integer_i(self, name, success) result(value)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                  :: self
      character(*), intent(in)       :: name
      integer(kind=c_int)            :: value
      integer, intent(out), optional :: success
    end function
  end interface


  abstract interface
    function elpa_is_set_i(self, name) result(success)
      import elpa_t
      class(elpa_t)            :: self
      character(*), intent(in) :: name
      integer                  :: success
    end function
  end interface


  abstract interface
    subroutine elpa_print_options_i(self, option_name, unit)
      import elpa_t, c_char
      class(elpa_t), intent(in)                 :: self
      character(kind=c_char, len=*), intent(in) :: option_name
      integer, intent(in), optional             :: unit
    end subroutine
  end interface


  abstract interface
    subroutine elpa_set_double_i(self, name, value, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
      character(*), intent(in)        :: name
      real(kind=c_double), intent(in) :: value
      integer, optional               :: success
    end subroutine
  end interface


  abstract interface
    function elpa_get_double_i(self, name, success) result(value)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                  :: self
      character(*), intent(in)       :: name
      real(kind=c_double)            :: value
      integer, intent(out), optional :: success
    end function
  end interface


  abstract interface
    function elpa_associate_int_i(self, name) result(value)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                  :: self
      character(*), intent(in)       :: name
      integer(kind=c_int), pointer   :: value
    end function
  end interface


  abstract interface
    subroutine elpa_solve_real_double_i(self, a, ev, q, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)       :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_double) :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      real(kind=c_double) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_double) :: ev(self%na)

      integer, optional   :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_solve_real_single_i(self, a, ev, q, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)       :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_float)  :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      real(kind=c_float)  :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_float)  :: ev(self%na)

      integer, optional   :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_solve_complex_double_i(self, a, ev, q, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                  :: self

#ifdef USE_ASSUMED_SIZE
      complex(kind=c_double_complex) :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      complex(kind=c_double_complex) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_double)            :: ev(self%na)

      integer, optional              :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_solve_complex_single_i(self, a, ev, q, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                 :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_float_complex) :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      complex(kind=c_float_complex) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_float)            :: ev(self%na)

      integer, optional             :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_multiply_at_b_double_i (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                          c, ldc, ldcCols, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=c_int), intent(in) :: na, lda, ldaCols, ldb, ldbCols, ldc, ldcCols, ncb
#ifdef USE_ASSUMED_SIZE
      real(kind=c_double)             :: a(lda,*), b(ldb,*), c(ldc,*)
#else
      real(kind=c_double)             :: a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)
#endif
      integer, optional               :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_multiply_at_b_single_i (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                          c, ldc, ldcCols, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=c_int), intent(in) :: na, lda, ldaCols, ldb, ldbCols, ldc, ldcCols, ncb
#ifdef USE_ASSUMED_SIZE
      real(kind=c_float)              :: a(lda,*), b(ldb,*), c(ldc,*)
#else
      real(kind=c_float)              :: a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)
#endif
      integer, optional               :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_multiply_ah_b_double_i (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                          c, ldc, ldcCols, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=c_int), intent(in) :: na, lda, ldaCols, ldb, ldbCols, ldc, ldcCols, ncb
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_double_complex)  :: a(lda,*), b(ldb,*), c(ldc,*)
#else
      complex(kind=c_double_complex)  :: a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)
#endif
      integer, optional               :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_multiply_ah_b_single_i (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                          c, ldc, ldcCols, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=c_int), intent(in) :: na, lda, ldaCols, ldb, ldbCols, ldc, ldcCols, ncb
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_float_complex)   :: a(lda,*), b(ldb,*), c(ldc,*)
#else
      complex(kind=c_float_complex)   :: a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)
#endif
      integer, optional               :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_cholesky_double_real_i (self, a, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_double)             :: a(self%local_nrows,*)
#else
      real(kind=c_double)             :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_cholesky_single_real_i(self, a, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_float)              :: a(self%local_nrows,*)
#else
      real(kind=c_float)              :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_cholesky_double_complex_i (self, a, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_double_complex)  :: a(self%local_nrows,*)
#else
      complex(kind=c_double_complex)  :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_cholesky_single_complex_i (self, a, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_float_complex)   :: a(self%local_nrows,*)
#else
      complex(kind=c_float_complex)   :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_invert_trm_double_real_i (self, a, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_double)             :: a(self%local_nrows,*)
#else
      real(kind=c_double)             :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_invert_trm_single_real_i (self, a, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_float)              :: a(self%local_nrows,*)
#else
      real(kind=c_float)              :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_invert_trm_double_complex_i (self, a, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_double_complex)  :: a(self%local_nrows,*)
#else
      complex(kind=c_double_complex)  :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_invert_trm_single_complex_i (self, a, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_float_complex)   :: a(self%local_nrows,*)
#else
      complex(kind=c_float_complex)   :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_solve_tridi_double_real_i (self, d, e, q, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
      real(kind=c_double)             :: d(self%na), e(self%na)
#ifdef USE_ASSUMED_SIZE
      real(kind=c_double)             :: q(self%local_nrows,*)
#else
      real(kind=c_double)             :: q(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_solve_tridi_single_real_i (self, d, e, q, success)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
      real(kind=c_float)              :: d(self%na), e(self%na)
#ifdef USE_ASSUMED_SIZE
      real(kind=c_float)              :: q(self%local_nrows,*)
#else
      real(kind=c_float)              :: q(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
    end subroutine
  end interface


  abstract interface
    subroutine elpa_destroy_i(self)
      use elpa_generated_fortran_interfaces
      import elpa_t
      class(elpa_t) :: self
    end subroutine
  end interface


  abstract interface
    function elpa_get_int_i(self) result(value)
      use iso_c_binding
      import elpa_t
      class(elpa_t), intent(in) :: self
      integer(kind=C_INT) :: value
    end function
  end interface

  contains

    function elpa_init(api_version) result(error)
      use elpa_utilities, only : error_unit
      integer, intent(in) :: api_version
      integer             :: error

      if (earliest_api_version <= api_version .and. api_version <= current_api_version) then
        initDone = .true.
        error = ELPA_OK
      else
        write(error_unit, "(a,i0,a)") "ELPA: Error API version ", api_version," is not supported by this library"
        error = ELPA_ERROR
      endif
    end function


    function elpa_initialized() result(state)
      logical :: state
      state = initDone
    end function


    subroutine elpa_uninit()
    end subroutine


    function elpa_strerr(elpa_error) result(string)
      use elpa_generated_fortran_interfaces
      integer, intent(in) :: elpa_error
      character(kind=C_CHAR, len=elpa_strlen_c(elpa_strerr_c(elpa_error))), pointer :: string
      call c_f_pointer(elpa_strerr_c(elpa_error), string)
    end function


    function elpa_c_string(ptr) result(string)
      use, intrinsic :: iso_c_binding
      type(c_ptr), intent(in) :: ptr
      character(kind=c_char, len=elpa_strlen_c(ptr)), pointer :: string
      call c_f_pointer(ptr, string)
    end function


    pure function elpa_int_value_to_string_helper(name, value) result(ptr)
      use, intrinsic :: iso_c_binding
      use elpa_generated_fortran_interfaces
      character(kind=c_char, len=*), intent(in) :: name
      integer(kind=c_int), intent(in) :: value
      integer(kind=c_int) :: error
      type(c_ptr) :: ptr

      ptr = elpa_index_int_value_to_string_helper_c(name // C_NULL_CHAR, value)
    end function


    function elpa_int_value_to_string(name, value, error) result(string)
      use elpa_utilities, only : error_unit
      use elpa_generated_fortran_interfaces
      implicit none
      character(kind=c_char, len=*), intent(in) :: name
      integer(kind=c_int), intent(in) :: value
      integer(kind=c_int), intent(out), optional :: error
      character(kind=c_char, len=elpa_strlen_c(elpa_int_value_to_string_helper(name, value))), pointer :: string
      integer(kind=c_int) :: actual_error
      type(c_ptr) :: ptr

      actual_error = elpa_index_int_value_to_string_c(name // C_NULL_CHAR, value, ptr)
      if (c_associated(ptr)) then
        call c_f_pointer(ptr, string)
      else
        nullify(string)
      endif

      if (present(error)) then
        error = actual_error
      else if (actual_error /= ELPA_OK) then
        write(error_unit,'(a,i0,a)') "ELPA: Error converting value '", value, "' to a string for option '" // &
                name // "' and you did not check for errors: " // elpa_strerr(actual_error)
      endif
    end function


    function elpa_int_string_to_value(name, string, error) result(value)
      use elpa_generated_fortran_interfaces
      use elpa_utilities, only : error_unit
      implicit none
      character(kind=c_char, len=*), intent(in) :: name
      character(kind=c_char, len=*), intent(in), target :: string
      integer(kind=c_int), intent(out), optional :: error
      integer(kind=c_int) :: actual_error

      integer(kind=c_int) :: value
      integer(kind=c_int)   :: i
      type(c_ptr) :: repr

      actual_error = elpa_index_int_string_to_value_c(name // C_NULL_CHAR, string // C_NULL_CHAR, value)

      if (present(error)) then
        error = actual_error
      else if (actual_error /= ELPA_OK) then
        write(error_unit,'(a)') "ELPA: Error converting string '" // string // "' to value for option '" // &
                name // "' and you did not check for errors: " // elpa_strerr(actual_error)
      endif
    end function

end module
