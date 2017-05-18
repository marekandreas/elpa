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

module elpa_impl
  use elpa_api
  use, intrinsic :: iso_c_binding
  implicit none

  private
  public :: elpa_impl_allocate

  type, extends(elpa_t) :: elpa_impl_t
   private
   type(c_ptr)         :: index = C_NULL_PTR

   contains
     ! con-/destructor
     procedure, public :: setup => elpa_setup
     procedure, public :: destroy => elpa_destroy

     ! KV store
     procedure, public :: get => elpa_get_integer
     procedure, public :: get_double => elpa_get_double
     procedure, public :: is_set => elpa_is_set
     procedure, public :: can_set => elpa_can_set

     ! privates:
     procedure, private :: get_real_kernel => elpa_get_real_kernel
     procedure, private :: get_complex_kernel => elpa_get_complex_kernel

     procedure, private :: elpa_set_integer
     procedure, private :: elpa_set_double

     procedure, private :: elpa_solve_real_double
     procedure, private :: elpa_solve_real_single
     procedure, private :: elpa_solve_complex_double
     procedure, private :: elpa_solve_complex_single

     procedure, private :: elpa_multiply_at_b_double
     procedure, private :: elpa_multiply_at_b_single
     procedure, private :: elpa_multiply_ah_b_double
     procedure, private :: elpa_multiply_ah_b_single

     procedure, private :: elpa_cholesky_double_real
     procedure, private :: elpa_cholesky_single_real
     procedure, private :: elpa_cholesky_double_complex
     procedure, private :: elpa_cholesky_single_complex

     procedure, private :: elpa_invert_trm_double_real
     procedure, private :: elpa_invert_trm_single_real
     procedure, private :: elpa_invert_trm_double_complex
     procedure, private :: elpa_invert_trm_single_complex

     procedure, private :: elpa_solve_tridi_double_real
     procedure, private :: elpa_solve_tridi_single_real

     procedure, private :: associate_int => elpa_associate_int

  end type elpa_impl_t

  contains

    function elpa_impl_allocate(error) result(obj)
      use precision
      use elpa_utilities, only : error_unit
      use elpa_generated_fortran_interfaces

      type(elpa_impl_t), pointer   :: obj
      integer, optional            :: error

      allocate(obj)

      ! check whether init has ever been called
      if (.not.(elpa_initialized())) then
        write(error_unit, *) "elpa_allocate(): you must call elpa_init() once before creating instances of ELPA"
        if(present(error)) then
          error = ELPA_ERROR
        endif
        return
      endif

      obj%index = elpa_index_instance_c()

      ! Associate some important integer pointers for convenience
      obj%na => obj%associate_int("na")
      obj%nev => obj%associate_int("nev")
      obj%local_nrows => obj%associate_int("local_nrows")
      obj%local_ncols => obj%associate_int("local_ncols")
      obj%nblk => obj%associate_int("nblk")

      if(present(error)) then
        error = ELPA_OK
      endif
    end function


    function elpa_setup(self) result(error)
      use elpa1_impl, only : elpa_get_communicators_impl
      class(elpa_impl_t), intent(inout) :: self
      integer :: error, error2
      integer :: mpi_comm_rows, mpi_comm_cols, mpierr

      error = ELPA_ERROR

      if (self%is_set("mpi_comm_parent") == 1 .and. &
          self%is_set("process_row") == 1 .and. &
          self%is_set("process_col") == 1) then

        mpierr = elpa_get_communicators_impl(&
                        self%get("mpi_comm_parent"), &
                        self%get("process_row"), &
                        self%get("process_col"), &
                        mpi_comm_rows, &
                        mpi_comm_cols)

        call self%set("mpi_comm_rows", mpi_comm_rows)
        call self%set("mpi_comm_cols", mpi_comm_cols)

        error = ELPA_OK
      endif

      if (self%is_set("mpi_comm_rows") == 1 .and. self%is_set("mpi_comm_cols") == 1) then
        error = ELPA_OK
      endif

    end function

    subroutine elpa_set_integer(self, name, value, error)
      use iso_c_binding
      use elpa_generated_fortran_interfaces
      use elpa_utilities, only : error_unit
      class(elpa_impl_t)              :: self
      character(*), intent(in)        :: name
      integer(kind=c_int), intent(in) :: value
      integer, optional               :: error
      integer                         :: actual_error

      actual_error = elpa_index_set_int_value_c(self%index, name // c_null_char, value, 0)

      if (present(error)) then
        error = actual_error

      else if (actual_error /= ELPA_OK) then
        write(error_unit,'(a,i0,a)') "ELPA: Error setting option '" // name // "' to value ", value, &
                " (got: " // elpa_strerr(actual_error) // ") and you did not check for errors!"
      end if
    end subroutine


    function elpa_get_integer(self, name, error) result(value)
      use iso_c_binding
      use elpa_generated_fortran_interfaces
      use elpa_utilities, only : error_unit
      class(elpa_impl_t)             :: self
      character(*), intent(in)       :: name
      integer(kind=c_int)            :: value
      integer, intent(out), optional :: error
      integer                        :: actual_error

      value = elpa_index_get_int_value_c(self%index, name // c_null_char, actual_error)
      if (present(error)) then
        error = actual_error
      else if (actual_error /= ELPA_OK) then
        write(error_unit,'(a)') "ELPA: Error getting option '" // name // "'" // &
                " (got: " // elpa_strerr(actual_error) // ") and you did not check for errors!"
      end if
    end function


    function elpa_is_set(self, name) result(state)
      use iso_c_binding
      use elpa_generated_fortran_interfaces
      class(elpa_impl_t)       :: self
      character(*), intent(in) :: name
      integer                  :: state

      state = elpa_index_value_is_set_c(self%index, name // c_null_char)
    end function


    function elpa_can_set(self, name, value) result(error)
      use iso_c_binding
      use elpa_generated_fortran_interfaces
      class(elpa_impl_t)       :: self
      character(*), intent(in) :: name
      integer(kind=c_int), intent(in) :: value
      integer                  :: error

      error = elpa_index_int_is_valid_c(self%index, name // c_null_char, value)
    end function


    function elpa_value_to_string(self, option_name, error) result(string)
      use elpa_generated_fortran_interfaces
      class(elpa_impl_t), intent(in) :: self
      character(kind=c_char, len=*), intent(in) :: option_name
      type(c_ptr) :: ptr
      integer, intent(out), optional :: error
      integer :: val, actual_error
      character(kind=c_char, len=elpa_index_int_value_to_strlen_c(self%index, option_name // C_NULL_CHAR)), pointer :: string

      nullify(string)

      val = self%get(option_name, actual_error)
      if (actual_error /= ELPA_OK) then
        if (present(error)) then
          error = actual_error
        endif
        return
      endif

      actual_error = elpa_int_value_to_string_c(option_name // C_NULL_CHAR, val, ptr)
      if (c_associated(ptr)) then
        call c_f_pointer(ptr, string)
      endif

      if (present(error)) then
        error = actual_error
      endif
    end function


    subroutine elpa_set_double(self, name, value, error)
      use iso_c_binding
      use elpa_generated_fortran_interfaces
      use elpa_utilities, only : error_unit
      class(elpa_impl_t)              :: self
      character(*), intent(in)        :: name
      real(kind=c_double), intent(in) :: value
      integer, optional               :: error
      integer                         :: actual_error

      actual_error = elpa_index_set_double_value_c(self%index, name // c_null_char, value, 0)

      if (present(error)) then
        error = actual_error
      else if (actual_error /= ELPA_OK) then
        write(error_unit,'(a,es12.5,a)') "ELPA: Error setting option '" // name // "' to value ", value, &
                " (got: " // elpa_strerr(actual_error) // ") and you did not check for errors!"
      end if
    end subroutine


    function elpa_get_double(self, name, error) result(value)
      use iso_c_binding
      use elpa_generated_fortran_interfaces
      use elpa_utilities, only : error_unit
      class(elpa_impl_t)             :: self
      character(*), intent(in)       :: name
      real(kind=c_double)            :: value
      integer, intent(out), optional :: error
      integer                        :: actual_error

      value = elpa_index_get_double_value_c(self%index, name // c_null_char, actual_error)
      if (present(error)) then
        error = actual_error
      else if (actual_error /= ELPA_OK) then
        write(error_unit,'(a)') "ELPA: Error getting option '" // name // "'" // &
                " (got: " // elpa_strerr(actual_error) // ") and you did not check for errors!"
      end if
    end function


    function elpa_associate_int(self, name) result(value)
      use iso_c_binding
      use elpa_generated_fortran_interfaces
      use elpa_utilities, only : error_unit
      class(elpa_impl_t)             :: self
      character(*), intent(in)       :: name
      integer(kind=c_int), pointer   :: value

      type(c_ptr)                    :: value_p

      value_p = elpa_index_get_int_loc_c(self%index, name // c_null_char)
      if (.not. c_associated(value_p)) then
        write(error_unit, '(a,a,a)') "ELPA: Warning, received NULL pointer for entry '", name, "'"
      endif
      call c_f_pointer(value_p, value)
    end function


    subroutine elpa_solve_real_double(self, a, ev, q, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit
      use precision
      use iso_c_binding
      class(elpa_impl_t)  :: self

#ifdef USE_ASSUMED_SIZE
      real(kind=c_double) :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      real(kind=c_double) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_double) :: ev(self%na)

      integer, optional   :: error
      logical             :: success_l

      logical             :: useGPU, useQR
      integer(kind=c_int) :: kernel

      if (self%get("gpu") .eq. 1) then
        useGPU = .true.
      else
        useGPU = .false.
      endif

      useQR = self%get("qr") == 1

      if (self%get("solver") .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_real_1stage_double_impl(self, self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                           self%local_nrows,  self%nblk, self%local_ncols, &
                                                           self%get("mpi_comm_rows"), self%get("mpi_comm_cols"), &
                                                           self%get("mpi_comm_parent"), useGPU)

      else if (self%get("solver") .eq. ELPA_SOLVER_2STAGE) then
        kernel = self%get_real_kernel()
        success_l = elpa_solve_evp_real_2stage_double_impl(self, self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                           self%local_nrows,  self%nblk, self%local_ncols, &
                                                           self%get("mpi_comm_rows"), self%get("mpi_comm_cols"), &
                                                           self%get("mpi_comm_parent"), useGPU, kernel, useQR)
      else
        print *,"unknown solver"
        stop
      endif

      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in solve() and you did not check for errors!"
      endif
    end subroutine


    subroutine elpa_solve_real_single(self, a, ev, q, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit
      use precision
      use iso_c_binding
      class(elpa_impl_t)  :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_float)  :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      real(kind=c_float)  :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_float)  :: ev(self%na)

      integer, optional   :: error
      logical             :: success_l

      logical             :: useGPU, useQR
      integer(kind=c_int) :: kernel

#ifdef WANT_SINGLE_PRECISION_REAL
      if (self%get("gpu") .eq. 1) then
        useGPU = .true.
      else
        useGPU = .false.
      endif

      useQR = self%get("qr") == 1

      if (self%get("solver") .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_real_1stage_single_impl(self, self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%get("mpi_comm_rows"), self%get("mpi_comm_cols"),         &
                                                          self%get("mpi_comm_parent"), useGPU)

      else if (self%get("solver") .eq. ELPA_SOLVER_2STAGE) then
        kernel = self%get_real_kernel()
        success_l = elpa_solve_evp_real_2stage_single_impl(self, self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%get("mpi_comm_rows"), self%get("mpi_comm_cols"),         &
                                                          self%get("mpi_comm_parent"), useGPU, kernel, useQR)
      else
        print *,"unknown solver"
        stop
      endif

      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in solve() and you did not check for errors!"
      endif
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
    end subroutine


    subroutine elpa_solve_complex_double(self, a, ev, q, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit
      use precision
      use iso_c_binding
      class(elpa_impl_t)             :: self

#ifdef USE_ASSUMED_SIZE
      complex(kind=c_double_complex) :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      complex(kind=c_double_complex) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_double)            :: ev(self%na)

      integer, optional              :: error
      logical                        :: success_l

      logical                        :: useGPU
      integer(kind=c_int) :: kernel

      if (self%get("gpu") .eq. 1) then
        useGPU = .true.
      else
        useGPU = .false.
      endif

      if (self%get("solver") .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_complex_1stage_double_impl(self, self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%get("mpi_comm_rows"), self%get("mpi_comm_cols"),         &
                                                          self%get("mpi_comm_parent"), useGPU)

      else if (self%get("solver") .eq. ELPA_SOLVER_2STAGE) then
        kernel = self%get_complex_kernel()
        success_l = elpa_solve_evp_complex_2stage_double_impl(self, self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%get("mpi_comm_rows"), self%get("mpi_comm_cols"),         &
                                                          self%get("mpi_comm_parent"), useGPU, kernel)
      else
        print *,"unknown solver"
        stop
      endif

      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in solve() and you did not check for errors!"
      endif
    end subroutine


    subroutine elpa_solve_complex_single(self, a, ev, q, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit

      use iso_c_binding
      use precision
      class(elpa_impl_t)            :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=ck4)             :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      complex(kind=ck4)             :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=rk4)                :: ev(self%na)

      integer, optional             :: error
      logical                       :: success_l

      logical                       :: useGPU
      integer(kind=c_int) :: kernel
#ifdef WANT_SINGLE_PRECISION_COMPLEX

      if (self%get("gpu") .eq. 1) then
        useGPU = .true.
      else
        useGPU = .false.
      endif

      if (self%get("solver") .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_complex_1stage_single_impl(self, self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%get("mpi_comm_rows"), self%get("mpi_comm_cols"),         &
                                                          self%get("mpi_comm_parent"), useGPU)

      else if (self%get("solver") .eq. ELPA_SOLVER_2STAGE) then
        kernel = self%get_complex_kernel()
        success_l = elpa_solve_evp_complex_2stage_single_impl(self, self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%get("mpi_comm_rows"), self%get("mpi_comm_cols"),         &
                                                          self%get("mpi_comm_parent"), useGPU, kernel)
      else
        print *,"unknown solver"
        stop
      endif

      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in solve() and you did not check for errors!"
      endif
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
    end subroutine


    subroutine elpa_multiply_at_b_double (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                          c, ldc, ldcCols, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      class(elpa_impl_t)              :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=ik), intent(in)    :: na, lda, ldaCols, ldb, ldbCols, ldc, ldcCols, ncb
#ifdef USE_ASSUMED_SIZE
      real(kind=rk8)                  :: a(lda,*), b(ldb,*), c(ldc,*)
#else
      real(kind=rk8)                  :: a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)
#endif
      integer, optional               :: error
      logical                         :: success_l

      success_l = elpa_mult_at_b_real_double_impl(uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, self%nblk, &
                              self%get("mpi_comm_rows"), self%get("mpi_comm_cols"), c, ldc, ldcCols)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in multiply_a_b() and you did not check for errors!"
      endif
    end subroutine


    subroutine elpa_multiply_at_b_single (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                          c, ldc, ldcCols, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      class(elpa_impl_t)              :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=ik), intent(in)    :: na, lda, ldaCols, ldb, ldbCols, ldc, ldcCols, ncb
#ifdef USE_ASSUMED_SIZE
      real(kind=rk4)                  :: a(lda,*), b(ldb,*), c(ldc,*)
#else
      real(kind=rk4)                  :: a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)
#endif
      integer, optional               :: error
      logical                         :: success_l
#ifdef WANT_SINGLE_PRECISION_REAL
      success_l = elpa_mult_at_b_real_single_impl(uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, self%nblk, &
                              self%get("mpi_comm_rows"), self%get("mpi_comm_cols"), c, ldc, ldcCols)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in multiply_a_b() and you did not check for errors!"
      endif
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
    end subroutine


    subroutine elpa_multiply_ah_b_double (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                          c, ldc, ldcCols, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      class(elpa_impl_t)              :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=ik), intent(in)    :: na, lda, ldaCols, ldb, ldbCols, ldc, ldcCols, ncb
#ifdef USE_ASSUMED_SIZE
      complex(kind=ck8)               :: a(lda,*), b(ldb,*), c(ldc,*)
#else
      complex(kind=ck8)               :: a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)
#endif
      integer, optional               :: error
      logical                         :: success_l

      success_l = elpa_mult_ah_b_complex_double_impl(uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, self%nblk, &
                              self%get("mpi_comm_rows"), self%get("mpi_comm_cols"), c, ldc, ldcCols)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in multiply_a_b() and you did not check for errors!"
      endif
    end subroutine


    subroutine elpa_multiply_ah_b_single (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                          c, ldc, ldcCols, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      class(elpa_impl_t)              :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=ik), intent(in)    :: na, lda, ldaCols, ldb, ldbCols, ldc, ldcCols, ncb
#ifdef USE_ASSUMED_SIZE
      complex(kind=ck4)               :: a(lda,*), b(ldb,*), c(ldc,*)
#else
      complex(kind=ck4)               :: a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)
#endif
      integer, optional               :: error
      logical                         :: success_l

#ifdef WANT_SINGLE_PRECISION_COMPLEX
      success_l = elpa_mult_ah_b_complex_single_impl(uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, self%nblk, &
                              self%get("mpi_comm_rows"), self%get("mpi_comm_cols"), c, ldc, ldcCols)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in multiply_a_b() and you did not check for errors!"
      endif
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
    end subroutine


    subroutine elpa_cholesky_double_real (self, a, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      class(elpa_impl_t)              :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=rk8)                  :: a(self%local_nrows,*)
#else
      real(kind=rk8)                  :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
      logical                         :: success_l
      integer(kind=c_int)             :: error_actual
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",error_actual) .eq. 1) then
        if (error_actual .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif


      success_l = elpa_cholesky_real_double_impl (self%na, a, self%local_nrows, self%nblk, &
                                                 self%local_ncols, self%get("mpi_comm_rows"), self%get("mpi_comm_cols"), &
                                                 wantDebugIntern)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in cholesky() and you did not check for errors!"
      endif
    end subroutine


    subroutine elpa_cholesky_single_real(self, a, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      class(elpa_impl_t)              :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=rk4)                  :: a(self%local_nrows,*)
#else
      real(kind=rk4)                  :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
      logical                         :: success_l
      integer(kind=c_int)             :: error_actual
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",error_actual) .eq. 1) then
        if (error_actual .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif

#if WANT_SINGLE_PRECISION_REAL
      success_l = elpa_cholesky_real_single_impl (self%na, a, self%local_nrows, self%nblk, &
                                                 self%local_ncols, self%get("mpi_comm_rows"), self%get("mpi_comm_cols"), &
                                                 wantDebugIntern)
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in cholesky() and you did not check for errors!"
      endif
    end subroutine


    subroutine elpa_cholesky_double_complex (self, a, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      class(elpa_impl_t)              :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=ck8)               :: a(self%local_nrows,*)
#else
      complex(kind=ck8)               :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
      logical                         :: success_l
      integer(kind=c_int)             :: error_actual
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",error_actual) .eq. 1) then
        if (error_actual .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif

      success_l = elpa_cholesky_complex_double_impl (self%na, a, self%local_nrows, self%nblk, &
                                                 self%local_ncols, self%get("mpi_comm_rows"), self%get("mpi_comm_cols"), &
                                                 wantDebugIntern)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in cholesky() and you did not check for errors!"
      endif
    end subroutine


    subroutine elpa_cholesky_single_complex (self, a, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      class(elpa_impl_t)              :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=ck4)               :: a(self%local_nrows,*)
#else
      complex(kind=ck4)               :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
      logical                         :: success_l
      integer(kind=c_int)             :: error_actual
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",error_actual) .eq. 1) then
        if (error_actual .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif
#if WANT_SINGLE_PRECISION_COMPLEX
      success_l = elpa_cholesky_complex_single_impl (self%na, a, self%local_nrows, self%nblk, &
                                                 self%local_ncols, self%get("mpi_comm_rows"), self%get("mpi_comm_cols"), &
                                                 wantDebugIntern)
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in cholesky() and you did not check for errors!"
      endif
    end subroutine


    subroutine elpa_invert_trm_double_real (self, a, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      class(elpa_impl_t)              :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=rk8)                  :: a(self%local_nrows,*)
#else
      real(kind=rk8)                  :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
      logical                         :: success_l
      integer(kind=c_int)             :: error_actual
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",error_actual) .eq. 1) then
        if (error_actual .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif
      success_l = elpa_invert_trm_real_double_impl (self%na, a, self%local_nrows, self%nblk, &
                                                   self%local_ncols, self%get("mpi_comm_rows"), self%get("mpi_comm_cols"), &
                                                   wantDebugIntern)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in invert_trm() and you did not check for errors!"
      endif
    end subroutine


    subroutine elpa_invert_trm_single_real (self, a, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      class(elpa_impl_t)              :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=rk4)                  :: a(self%local_nrows,*)
#else
      real(kind=rk4)                  :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
      logical                         :: success_l
      integer(kind=c_int)             :: error_actual
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",error_actual) .eq. 1) then
        if (error_actual .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif
#if WANT_SINGLE_PRECISION_REAL
      success_l = elpa_invert_trm_real_single_impl (self%na, a, self%local_nrows, self%nblk, &
                                                   self%local_ncols, self%get("mpi_comm_rows"), self%get("mpi_comm_cols"), &
                                                   wantDebugIntern)
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in invert_trm() and you did not check for errors!"
      endif
    end subroutine


    subroutine elpa_invert_trm_double_complex (self, a, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      class(elpa_impl_t)              :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=ck8)               :: a(self%local_nrows,*)
#else
      complex(kind=ck8)               :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
      logical                         :: success_l
      integer(kind=c_int)             :: error_actual
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",error_actual) .eq. 1) then
        if (error_actual .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif
      success_l = elpa_invert_trm_complex_double_impl (self%na, a, self%local_nrows, self%nblk, &
                                                   self%local_ncols, self%get("mpi_comm_rows"), self%get("mpi_comm_cols"), &
                                                   wantDebugIntern)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in invert_trm() and you did not check for errors!"
      endif
    end subroutine


    subroutine elpa_invert_trm_single_complex (self, a, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      class(elpa_impl_t)              :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=ck4)               :: a(self%local_nrows,*)
#else
      complex(kind=ck4)               :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
      logical                         :: success_l
      integer(kind=c_int)             :: error_actual
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",error_actual) .eq. 1) then
        if (error_actual .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif
#if WANT_SINGLE_PRECISION_COMPLEX
      success_l = elpa_invert_trm_complex_single_impl (self%na, a, self%local_nrows, self%nblk, &
                                                   self%local_ncols, self%get("mpi_comm_rows"), self%get("mpi_comm_cols"), &
                                                   wantDebugIntern)
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in invert_trm() and you did not check for errors!"
      endif
    end subroutine


    !> \todo e should have dimension (na - 1)
    subroutine elpa_solve_tridi_double_real (self, d, e, q, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      class(elpa_impl_t)              :: self
      real(kind=rk8)                  :: d(self%na), e(self%na)
#ifdef USE_ASSUMED_SIZE
      real(kind=rk8)                  :: q(self%local_nrows,*)
#else
      real(kind=rk8)                  :: q(self%local_nrows,self%local_ncols)
#endif

      integer, optional               :: error
      logical                         :: success_l
      integer(kind=c_int)             :: error_actual
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",error_actual) .eq. 1) then
        if (error_actual .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif

      success_l = elpa_solve_tridi_double_impl(self%na, self%nev, d, e, q, self%local_nrows, self%nblk, &
                                              self%local_ncols, self%get("mpi_comm_rows"), self%get("mpi_comm_cols"),&
                                              wantDebugIntern)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in solve_tridi() and you did not check for errors!"
      endif
    end subroutine


    subroutine elpa_solve_tridi_single_real (self, d, e, q, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      class(elpa_impl_t)              :: self
      real(kind=rk4)                  :: d(self%na), e(self%na)
#ifdef USE_ASSUMED_SIZE
      real(kind=rk4)                  :: q(self%local_nrows,*)
#else
      real(kind=rk4)                  :: q(self%local_nrows,self%local_ncols)
#endif

      integer, optional               :: error
      logical                         :: success_l
      integer(kind=c_int)             :: error_actual
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",error_actual) .eq. 1) then
        if (error_actual .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif
#ifdef WANT_SINGLE_PRECISION_REAL
      success_l = elpa_solve_tridi_single_impl(self%na, self%nev, d, e, q, self%local_nrows, self%nblk, &
                                              self%local_ncols, self%get("mpi_comm_rows"), self%get("mpi_comm_cols"),&
                                              wantDebugIntern)
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in solve_tridi() and you did not check for errors!"
      endif
    end subroutine


    subroutine elpa_destroy(self)
      use elpa_generated_fortran_interfaces
      class(elpa_impl_t) :: self
      call elpa_index_free_c(self%index)
    end subroutine


    function elpa_get_real_kernel(self) result(kernel)
      use elpa_utilities, only : error_unit
      class(elpa_impl_t), intent(in) :: self
      integer(kind=c_int) :: kernel, error

      error = ELPA_ERROR
      kernel = self%get("real_kernel")

      ! check consistency between request for GPUs and defined kernel
      if (self%get("gpu") == 1) then
        if (kernel .ne. ELPA_2STAGE_REAL_GPU) then
          write(error_unit,*) "ELPA: Warning, GPU usage has been requested but compute kernel is defined as non-GPU!"
        else if (self%get("nblk") .ne. 128) then
          kernel = ELPA_2STAGE_REAL_GENERIC
        endif
      endif
    end function


    function elpa_get_complex_kernel(self) result(kernel)
      use elpa_utilities, only : error_unit
      class(elpa_impl_t), intent(in) :: self
      integer(kind=C_INT) :: kernel, error

      error = ELPA_ERROR
      kernel = self%get("complex_kernel")

      ! check consistency between request for GPUs and defined kernel
      if (self%get("gpu") == 1) then
        if (kernel .ne. ELPA_2STAGE_COMPLEX_GPU) then
          write(error_unit,*) "ELPA: Warning, GPU usage has been requested but compute kernel is defined as non-GPU!"
        else if (self%get("nblk") .ne. 128) then
          kernel = ELPA_2STAGE_COMPLEX_GENERIC
        endif
      endif

    end function


end module
