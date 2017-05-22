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
!> \brief Fortran module which provides the definition of the ELPA API
module elpa_api
  use elpa_constants
  use, intrinsic :: iso_c_binding
  implicit none

  integer, private, parameter :: earliest_api_version = EARLIEST_API_VERSION !< Definition of the earliest API version supported
                                                                             !< with the current release
  integer, private, parameter :: current_api_version  = CURRENT_API_VERSION  !< Definition of the current API version

  logical, private :: initDone = .false.

  public :: elpa_t, &
      c_int, &
      c_double, c_double_complex, &
      c_float, c_float_complex

  !> \brief Abstract defintion of the elpa_t type
  type, abstract :: elpa_t
    private

    ! these have to be public for proper bounds checking, sadly
    integer(kind=c_int), public, pointer :: na => NULL()
    integer(kind=c_int), public, pointer :: nev => NULL()
    integer(kind=c_int), public, pointer :: local_nrows => NULL()
    integer(kind=c_int), public, pointer :: local_ncols => NULL()
    integer(kind=c_int), public, pointer :: nblk => NULL()

    contains
      !> \brief methods available with the elpa_t type
      ! general
      procedure(elpa_setup_i),   deferred, public :: setup          !< export a setup method
      procedure(elpa_destroy_i), deferred, public :: destroy        !< export a destroy method

      ! key/value store
      generic, public :: set => &                                   !< export a method to set integer/double key/values
          elpa_set_integer, &
          elpa_set_double
      procedure(elpa_get_integer_i), deferred, public :: get        !< get method for integer key/values
      procedure(elpa_get_double_i),  deferred, public :: get_double !< get method for double key/values

      procedure(elpa_is_set_i),  deferred, public :: is_set         !< method to check whether key/value is set
      procedure(elpa_can_set_i), deferred, public :: can_set        !< method to check whether key/value can be set

      ! Timer
      procedure(elpa_get_time_i), deferred, public :: get_time
      procedure(elpa_print_times_i), deferred, public :: print_times

      ! Actual math routines
      generic, public :: solve => &                                 !< method solve for solving the eigenvalue problem
          elpa_solve_real_double, &                                 !< for symmetric real valued / hermitian complex valued
          elpa_solve_real_single, &                                 !< matrices
          elpa_solve_complex_double, &
          elpa_solve_complex_single

      generic, public :: hermitian_multiply => &                    !< method for a "hermitian" multiplication of matrices a and b
          elpa_multiply_at_b_double, &                              !< for real valued matrices:   a**T * b
          elpa_multiply_ah_b_double, &                              !< for complex valued matrices a**H * b
          elpa_multiply_at_b_single, &
          elpa_multiply_ah_b_single

      generic, public :: cholesky => &                              !< method for the cholesky factorisation of matrix a
          elpa_cholesky_double_real, &
          elpa_cholesky_single_real, &
          elpa_cholesky_double_complex, &
          elpa_cholesky_single_complex

      generic, public :: invert_triangular => &                     !< method to invert a upper triangular matrix a
          elpa_invert_trm_double_real, &
          elpa_invert_trm_single_real, &
          elpa_invert_trm_double_complex, &
          elpa_invert_trm_single_complex

      generic, public :: solve_tridi => &                           !< method to solve the eigenvalue problem for a tridiagonal
          elpa_solve_tridi_double_real, &                           !< matrix
          elpa_solve_tridi_single_real


      !> \brief private methods of elpa_t type. NOT accessible for the user
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
    function elpa_setup_i(self) result(error)
      import elpa_t
      class(elpa_t), intent(inout) :: self
      integer :: error
    end function
  end interface


  abstract interface
    subroutine elpa_set_integer_i(self, name, value, error)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
      character(*), intent(in)        :: name
      integer(kind=c_int), intent(in) :: value
      integer, optional               :: error
    end subroutine
  end interface


  abstract interface
    function elpa_get_integer_i(self, name, error) result(value)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                  :: self
      character(*), intent(in)       :: name
      integer(kind=c_int)            :: value
      integer, intent(out), optional :: error
    end function
  end interface


  abstract interface
    function elpa_is_set_i(self, name) result(error)
      import elpa_t
      class(elpa_t)            :: self
      character(*), intent(in) :: name
      integer                  :: error
    end function
  end interface


  abstract interface
    function elpa_can_set_i(self, name, value) result(error)
      import elpa_t, c_int
      class(elpa_t)                   :: self
      character(*), intent(in)        :: name
      integer(kind=c_int), intent(in) :: value
      integer                         :: error
    end function
  end interface


  abstract interface
    subroutine elpa_set_double_i(self, name, value, error)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
      character(*), intent(in)        :: name
      real(kind=c_double), intent(in) :: value
      integer, optional               :: error
    end subroutine
  end interface


  abstract interface
    function elpa_get_double_i(self, name, error) result(value)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                  :: self
      character(*), intent(in)       :: name
      real(kind=c_double)            :: value
      integer, intent(out), optional :: error
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


  ! Timer routines

  abstract interface
    function elpa_get_time_i(self, name1, name2, name3, name4, name5, name6) result(s)
      import elpa_t, c_double
      class(elpa_t), intent(in) :: self
      ! this is clunky, but what can you do..
      character(len=*), intent(in), optional :: name1, name2, name3, name4, name5, name6
      real(kind=c_double) :: s
    end function
  end interface


  abstract interface
    subroutine elpa_print_times_i(self)
      import elpa_t
      class(elpa_t), intent(in) :: self
    end subroutine
  end interface


  ! Actual math routines

  !> \brief abstract defintion of interface to solve double real eigenvalue problem
  !> Parameters
  !> \details
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   a           double real matrix a: defines the problem to solve
  !> \param   ev          double real: on output stores the eigenvalues
  !> \param   q           double real matrix q: on output stores the eigenvalues
  abstract interface
    subroutine elpa_solve_real_double_i(self, a, ev, q, error)
      use iso_c_binding
      import elpa_t
      class(elpa_t)       :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_double) :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      real(kind=c_double) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_double) :: ev(self%na)

      integer, optional   :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface to solve single real eigenvalue problem
  !> Parameters
  !> \details
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   a           single real matrix a: defines the problem to solve
  !> \param   ev          single real: on output stores the eigenvalues
  !> \param   q           single real matrix q: on output stores the eigenvalues
  abstract interface
    subroutine elpa_solve_real_single_i(self, a, ev, q, error)
      use iso_c_binding
      import elpa_t
      class(elpa_t)       :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_float)  :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      real(kind=c_float)  :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_float)  :: ev(self%na)

      integer, optional   :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface to solve double complex eigenvalue problem
  !> Parameters
  !> \details
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   a           double complex matrix a: defines the problem to solve
  !> \param   ev          double real: on output stores the eigenvalues
  !> \param   q           double complex matrix q: on output stores the eigenvalues
  abstract interface
    subroutine elpa_solve_complex_double_i(self, a, ev, q, error)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                  :: self

#ifdef USE_ASSUMED_SIZE
      complex(kind=c_double_complex) :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      complex(kind=c_double_complex) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_double)            :: ev(self%na)

      integer, optional              :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface to solve single complex eigenvalue problem
  !> Parameters
  !> \details
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   a           single complex matrix a: defines the problem to solve
  !> \param   ev          single real: on output stores the eigenvalues
  !> \param   q           single complex matrix q: on output stores the eigenvalues
  abstract interface
    subroutine elpa_solve_complex_single_i(self, a, ev, q, error)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                 :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_float_complex) :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      complex(kind=c_float_complex) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_float)            :: ev(self%na)

      integer, optional             :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface to compute C : = A**T * B
  !> Parameters
  !> \details
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   uplo_a      'U' if A is upper triangular
  !>                      'L' if A is lower triangular
  !>                      anything else if A is a full matrix
  !>                      Please note: This pertains to the original A (as set in the calling program)
  !>                                   whereas the transpose of A is used for calculations
  !>                      If uplo_a is 'U' or 'L', the other triangle is not used at all,
  !>                      i.e. it may contain arbitrary numbers
  !> \param uplo_c        'U' if only the upper diagonal part of C is needed
  !>                      'L' if only the upper diagonal part of C is needed
  !>                      anything else if the full matrix C is needed
  !>                      Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
  !>                                    written to a certain extent, i.e. one shouldn't rely on the content there!
  !> \param na            Number of rows/columns of A, number of rows of B and C
  !> \param ncb           Number of real  of B and C
  !> \param   a           double complex matrix a
  !> \param lda           leading dimension of matrix a
  !> \param ldaCols       columns of matrix a
  !> \param b             double real matrix b
  !> \param ldb           leading dimension of matrix b
  !> \param ldbCols       columns of matrix b
  !> \param c             double real  matrix c
  !> \param ldc           leading dimension of matrix c
  !> \param ldcCols       columns of matrix c
  !> \param   error       integer, optional : error code
  abstract interface
    subroutine elpa_multiply_at_b_double_i (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                          c, ldc, ldcCols, error)
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
      integer, optional               :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface to compute C : = A**T * B
  !> Parameters
  !> \details
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   uplo_a      'U' if A is upper triangular
  !>                      'L' if A is lower triangular
  !>                      anything else if A is a full matrix
  !>                      Please note: This pertains to the original A (as set in the calling program)
  !>                                   whereas the transpose of A is used for calculations
  !>                      If uplo_a is 'U' or 'L', the other triangle is not used at all,
  !>                      i.e. it may contain arbitrary numbers
  !> \param uplo_c        'U' if only the upper diagonal part of C is needed
  !>                      'L' if only the upper diagonal part of C is needed
  !>                      anything else if the full matrix C is needed
  !>                      Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
  !>                                    written to a certain extent, i.e. one shouldn't rely on the content there!
  !> \param na            Number of rows/columns of A, number of rows of B and C
  !> \param ncb           Number of real  of B and C
  !> \param   a           single complex matrix a
  !> \param lda           leading dimension of matrix a
  !> \param ldaCols       columns of matrix a
  !> \param b             single real matrix b
  !> \param ldb           leading dimension of matrix b
  !> \param ldbCols       columns of matrix b
  !> \param c             single real  matrix c
  !> \param ldc           leading dimension of matrix c
  !> \param ldcCols       columns of matrix c
  !> \param   error       integer, optional : error code
  abstract interface
    subroutine elpa_multiply_at_b_single_i (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                          c, ldc, ldcCols, error)
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
      integer, optional               :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface to compute C : = A**H * B
  !> Parameters
  !> \details
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   uplo_a      'U' if A is upper triangular
  !>                      'L' if A is lower triangular
  !>                      anything else if A is a full matrix
  !>                      Please note: This pertains to the original A (as set in the calling program)
  !>                                   whereas the transpose of A is used for calculations
  !>                      If uplo_a is 'U' or 'L', the other triangle is not used at all,
  !>                      i.e. it may contain arbitrary numbers
  !> \param uplo_c        'U' if only the upper diagonal part of C is needed
  !>                      'L' if only the upper diagonal part of C is needed
  !>                      anything else if the full matrix C is needed
  !>                      Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
  !>                                    written to a certain extent, i.e. one shouldn't rely on the content there!
  !> \param na            Number of rows/columns of A, number of rows of B and C
  !> \param ncb           Number of columns  of B and C
  !> \param   a           double complex matrix a
  !> \param lda           leading dimension of matrix a
  !> \param ldaCols       columns of matrix a
  !> \param b             double complex matrix b
  !> \param ldb           leading dimension of matrix b
  !> \param ldbCols       columns of matrix b
  !> \param c             double complex  matrix c
  !> \param ldc           leading dimension of matrix c
  !> \param ldcCols       columns of matrix c
  !> \param   error       integer, optional : error code
  abstract interface
    subroutine elpa_multiply_ah_b_double_i (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                          c, ldc, ldcCols, error)
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
      integer, optional               :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface to compute C : = A**H * B
  !> Parameters
  !> \details
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   uplo_a      'U' if A is upper triangular
  !>                      'L' if A is lower triangular
  !>                      anything else if A is a full matrix
  !>                      Please note: This pertains to the original A (as set in the calling program)
  !>                                   whereas the transpose of A is used for calculations
  !>                      If uplo_a is 'U' or 'L', the other triangle is not used at all,
  !>                      i.e. it may contain arbitrary numbers
  !> \param uplo_c        'U' if only the upper diagonal part of C is needed
  !>                      'L' if only the upper diagonal part of C is needed
  !>                      anything else if the full matrix C is needed
  !>                      Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
  !>                                    written to a certain extent, i.e. one shouldn't rely on the content there!
  !> \param na            Number of rows/columns of A, number of rows of B and C
  !> \param ncb           Number of columns  of B and C
  !> \param   a           single complex matrix a
  !> \param lda           leading dimension of matrix a
  !> \param ldaCols       columns of matrix a
  !> \param b             single complex matrix b
  !> \param ldb           leading dimension of matrix b
  !> \param ldbCols       columns of matrix b
  !> \param c             single complex  matrix c
  !> \param ldc           leading dimension of matrix c
  !> \param ldcCols       columns of matrix c
  !> \param   error       integer, optional : error code
  abstract interface
    subroutine elpa_multiply_ah_b_single_i (self, uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                          c, ldc, ldcCols, error)
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
      integer, optional               :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface to do a cholesky decomposition of a double real matrix
  !> Parameters
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   a           double real matrix: the matrix to be decomposed
  !> \param   error       integer, optional : error code
  abstract interface
    subroutine elpa_cholesky_double_real_i (self, a, error)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_double)             :: a(self%local_nrows,*)
#else
      real(kind=c_double)             :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface to do a cholesky decomposition of a single real matrix
  !> Parameters
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   a           single real matrix: the matrix to be decomposed
  !> \param   error       integer, optional : error code
  abstract interface
    subroutine elpa_cholesky_single_real_i(self, a, error)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_float)              :: a(self%local_nrows,*)
#else
      real(kind=c_float)              :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface to do a cholesky decomposition of a double complex matrix
  !> Parameters
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   a           double complex matrix: the matrix to be decomposed
  !> \param   error       integer, optional : error code
  abstract interface
    subroutine elpa_cholesky_double_complex_i (self, a, error)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_double_complex)  :: a(self%local_nrows,*)
#else
      complex(kind=c_double_complex)  :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface to do a cholesky decomposition of a single complex matrix
  !> Parameters
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   a           single complex matrix: the matrix to be decomposed
  !> \param   error       integer, optional : error code
  abstract interface
    subroutine elpa_cholesky_single_complex_i (self, a, error)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_float_complex)   :: a(self%local_nrows,*)
#else
      complex(kind=c_float_complex)   :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface to invert a triangular double real matrix
  !> Parameters
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   a           double real matrix: the matrix to be inverted
  !> \param   error       integer, optional : error code
  abstract interface
    subroutine elpa_invert_trm_double_real_i (self, a, error)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_double)             :: a(self%local_nrows,*)
#else
      real(kind=c_double)             :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface to invert a triangular single real matrix
  !> Parameters
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   a           single real matrix: the matrix to be inverted
  !> \param   error       integer, optional : error code
  abstract interface
    subroutine elpa_invert_trm_single_real_i (self, a, error)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_float)              :: a(self%local_nrows,*)
#else
      real(kind=c_float)              :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface to invert a triangular double complex matrix
  !> Parameters
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   a           double complex matrix: the matrix to be inverted
  !> \param   error       integer, optional : error code
  abstract interface
    subroutine elpa_invert_trm_double_complex_i (self, a, error)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_double_complex)  :: a(self%local_nrows,*)
#else
      complex(kind=c_double_complex)  :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface to invert a triangular single complex matrix
  !> Parameters
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   a           single complex matrix: the matrix to be inverted
  !> \param   error       integer, optional : error code
  abstract interface
    subroutine elpa_invert_trm_single_complex_i (self, a, error)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_float_complex)   :: a(self%local_nrows,*)
#else
      complex(kind=c_float_complex)   :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface to solve the eigenvalue problem for a real valued tridiangular matrix
  !> Parameters
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   d           double real 1d array: the diagonal elements of a matrix defined in setup
  !> \param   e           double real 1d array: the subdiagonal elements of a matrix defined in setup
  !> \param   q           double real matrix: on output contains the eigenvectors
  !> \param   error       integer, optional : error code
  abstract interface
    subroutine elpa_solve_tridi_double_real_i (self, d, e, q, error)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
      real(kind=c_double)             :: d(self%na), e(self%na)
#ifdef USE_ASSUMED_SIZE
      real(kind=c_double)             :: q(self%local_nrows,*)
#else
      real(kind=c_double)             :: q(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface to solve the eigenvalue problem for a real valued tridiangular matrix
  !> Parameters
  !> \param   self        class(elpa_t), the ELPA object
  !> \param   d           single real 1d array: the diagonal elements of a matrix defined in setup
  !> \param   e           single real 1d array: the subdiagonal elements of a matrix defined in setup
  !> \param   q           single real matrix: on output contains the eigenvectors
  !> \param   error       integer, optional : error code
  abstract interface
    subroutine elpa_solve_tridi_single_real_i (self, d, e, q, error)
      use iso_c_binding
      import elpa_t
      class(elpa_t)                   :: self
      real(kind=c_float)              :: d(self%na), e(self%na)
#ifdef USE_ASSUMED_SIZE
      real(kind=c_float)              :: q(self%local_nrows,*)
#else
      real(kind=c_float)              :: q(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
    end subroutine
  end interface

  !> \brief abstract defintion of interface of subroutine to destroy an ELPA object
  !> Parameters
  !> \param   self        class(elpa_t), the ELPA object
  abstract interface
    subroutine elpa_destroy_i(self)
      import elpa_t
      class(elpa_t) :: self
    end subroutine
  end interface

  !> \brief abstract defintion of interface of function to get an integer option
  !> Parameters
  !> \param   self        class(elpa_t), the ELPA object
  !> \result  value       integer
  abstract interface
    function elpa_get_int_i(self) result(value)
      use iso_c_binding
      import elpa_t
      class(elpa_t), intent(in) :: self
      integer(kind=C_INT) :: value
    end function
  end interface

  contains

    !> \brief function to intialise the ELPA library
    !> Parameters
    !> \param   api_version integer, api_version that ELPA should use
    !> \result  error       integer
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

    !> \brief function to check whether the ELPA library has been correctly initialised
    !> Parameters
    !> \result  state      logical
    function elpa_initialized() result(state)
      integer :: state
      if (initDone) then
        state = ELPA_OK
      else
        state = ELPA_ERROR
      endif
    end function

    !> \brief subroutine to uninit the ELPA library. Does nothing at the moment. Might do sth. later
    subroutine elpa_uninit()
    end subroutine

    !> \brief helper function for error strings: NOT public to the user
    !> Parameters
    !> \param   elpa_error  integer
    !> \result  string      string
    function elpa_strerr(elpa_error) result(string)
      use elpa_generated_fortran_interfaces
      integer, intent(in) :: elpa_error
      character(kind=C_CHAR, len=elpa_strlen_c(elpa_strerr_c(elpa_error))), pointer :: string
      call c_f_pointer(elpa_strerr_c(elpa_error), string)
    end function

    !> \brief helper function for c strings: NOT public to the user
    !> Parameters
    !> \param   ptr         type(c_ptr)
    !> \result  string      string
    function elpa_c_string(ptr) result(string)
      use, intrinsic :: iso_c_binding
      type(c_ptr), intent(in) :: ptr
      character(kind=c_char, len=elpa_strlen_c(ptr)), pointer :: string
      call c_f_pointer(ptr, string)
    end function

    !> \brief function to convert an integer in its string representation: NOT public to the user
    !> Parameters
    !> \param   name        string
    !> \param   value       integer
    !> \param   error       integer, optional
    !> \result  string      string
    function elpa_int_value_to_string(name, value, error) result(string)
      use elpa_utilities, only : error_unit
      use elpa_generated_fortran_interfaces
      implicit none
      character(kind=c_char, len=*), intent(in) :: name
      integer(kind=c_int), intent(in) :: value
      integer(kind=c_int), intent(out), optional :: error
      character(kind=c_char, len=elpa_int_value_to_strlen_c(name // C_NULL_CHAR, value)), pointer :: string
      integer(kind=c_int) :: actual_error
      type(c_ptr) :: ptr

      actual_error = elpa_int_value_to_string_c(name // C_NULL_CHAR, value, ptr)
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

    !> \brief function to convert a string in its integer representation: NOT public to the user
    !> Parameters
    !> \param   name        string
    !> \param   string      string
    !> \param   error       integer, optional
    !> \result  value       integer
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

      actual_error = elpa_int_string_to_value_c(name // C_NULL_CHAR, string // C_NULL_CHAR, value)

      if (present(error)) then
        error = actual_error
      else if (actual_error /= ELPA_OK) then
        write(error_unit,'(a)') "ELPA: Error converting string '" // string // "' to value for option '" // &
                name // "' and you did not check for errors: " // elpa_strerr(actual_error)
      endif
    end function

    !> \brief function to get the cardinality of an option. NOT public to the user
    !> Parameters
    !> \param   option_name string
    !> \result  number      integer
    function elpa_option_cardinality(option_name) result(number)
      use elpa_generated_fortran_interfaces
      character(kind=c_char, len=*), intent(in) :: option_name
      integer                                   :: number
      number = elpa_option_cardinality_c(option_name // C_NULL_CHAR)
    end function

    !> \brief function to enumerate an option. NOT public to the user
    !> Parameters
    !> \param   option_name string
    !> \param   i           integer
    !> \result  option      integer
    function elpa_option_enumerate(option_name, i) result(option)
      use elpa_generated_fortran_interfaces
      character(kind=c_char, len=*), intent(in) :: option_name
      integer, intent(in)                       :: i
      integer                                   :: option
      option = elpa_option_enumerate_c(option_name // C_NULL_CHAR, i)
    end function

end module
