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

!> \brief Fortran module which provides the implementation of the API
module elpa_impl
  use elpa_abstract_impl
  use, intrinsic :: iso_c_binding
  implicit none

  private
  public :: elpa_impl_allocate

!> \brief Definition of the extended elpa_impl_t type
  type, extends(elpa_abstract_impl_t) :: elpa_impl_t
   private
   type(c_ptr)         :: index = C_NULL_PTR

   !> \brief methods available with the elpa_impl_t type
   contains
     !> \brief the puplic methods
     ! con-/destructor
     procedure, public :: setup => elpa_setup                   !< a setup method: implemented in elpa_setup
     procedure, public :: destroy => elpa_destroy               !< a destroy method: implemented in elpa_destroy

     ! KV store
     procedure, public :: get => elpa_get_integer               !< a get method for integer key/values: implemented in elpa_get_integer
     procedure, public :: get_double => elpa_get_double         !< a get method for double key/values: implemented in elpa_get_double
     procedure, public :: is_set => elpa_is_set                 !< a method to check whether a key/value pair has been set : implemented
                                                                !< in elpa_is_set
     procedure, public :: can_set => elpa_can_set               !< a method to check whether a key/value pair can be set : implemented
                                                                !< in elpa_can_set


     ! timer
     procedure, public :: get_time => elpa_get_time
     procedure, public :: print_times => elpa_print_times


     !> \brief the private methods

     procedure, private :: elpa_set_integer                     !< private methods to implement the setting of an integer/double key/value pair
     procedure, private :: elpa_set_double

     procedure, private :: elpa_solve_d               !< private methods to implement the solve step for real/complex
                                                                !< double/single matrices
     procedure, private :: elpa_solve_f
     procedure, private :: elpa_solve_dc
     procedure, private :: elpa_solve_fc

     procedure, private :: elpa_hermitian_multiply_d            !< private methods to implement a "hermitian" multiplication of matrices a and b
     procedure, private :: elpa_hermitian_multiply_f            !< for real valued matrices:   a**T * b
     procedure, private :: elpa_hermitian_multiply_dc            !< for complex valued matrices:   a**H * b
     procedure, private :: elpa_hermitian_multiply_fc

     procedure, private :: elpa_cholesky_d            !< private methods to implement the cholesky factorisation of
                                                                !< real/complex double/single matrices
     procedure, private :: elpa_cholesky_f
     procedure, private :: elpa_cholesky_dc
     procedure, private :: elpa_cholesky_fc

     procedure, private :: elpa_invert_trm_d          !< private methods to implement the inversion of a triangular
                                                                !< real/complex double/single matrix
     procedure, private :: elpa_invert_trm_f
     procedure, private :: elpa_invert_trm_dc
     procedure, private :: elpa_invert_trm_fc

     procedure, private :: elpa_solve_tridi_d         !< private methods to implement the solve step for a real valued
     procedure, private :: elpa_solve_tridi_f         !< double/single tridiagonal matrix

     procedure, private :: associate_int => elpa_associate_int  !< private method to set some pointers

  end type elpa_impl_t

  !> \brief the implementation of the private methods
  contains
    !> \brief function to allocate an ELPA object
    !> Parameters
    !> \param   error      integer, optional to get an error code
    !> \result  obj        class(elpa_impl_t) allocated ELPA object
    function elpa_impl_allocate(error) result(obj)
      use precision
      use elpa_utilities, only : error_unit
      use elpa_generated_fortran_interfaces

      type(elpa_impl_t), pointer   :: obj
      integer, optional            :: error

      allocate(obj)

      ! check whether init has ever been called
      if ( elpa_initialized() .ne. ELPA_OK) then
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


    !c> elpa_t elpa_allocate();
    function elpa_impl_allocate_c(error) result(ptr) bind(C, name="elpa_allocate")
      integer(kind=c_int) :: error
      type(c_ptr) :: ptr
      type(elpa_impl_t), pointer :: obj

      obj => elpa_impl_allocate(error)
      ptr = c_loc(obj)
    end function


    !c> void elpa_deallocate(elpa_t handle);
    subroutine elpa_impl_deallocate_c(handle) bind(C, name="elpa_deallocate")
      type(c_ptr), value :: handle
      type(elpa_impl_t), pointer :: self

      call c_f_pointer(handle, self)
      call self%destroy()
      deallocate(self)
    end subroutine


    !> \brief function to setup an ELPA object and to store the MPI communicators internally
    !> Parameters
    !> \param   self       class(elpa_impl_t), the allocated ELPA object
    !> \result  error      integer, the error code
    function elpa_setup(self) result(error)
      use elpa1_impl, only : elpa_get_communicators_impl
      class(elpa_impl_t), intent(inout) :: self
      integer :: error, error2
      integer :: mpi_comm_rows, mpi_comm_cols, mpierr

#ifdef WITH_MPI
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
#else
      error = ELPA_OK
#endif

      if (self%get("timings") == 1) then
        call self%timer%enable()
      endif

    end function


    !c> int elpa_setup(elpa_t handle);
    function elpa_setup_c(handle) result(error) bind(C, name="elpa_setup")
      type(c_ptr), intent(in), value :: handle
      type(elpa_impl_t), pointer :: self
      integer(kind=c_int) :: error

      call c_f_pointer(handle, self)
      error = self%setup()
    end function


    !> \brief subroutine to set an integer key/value pair
    !> Parameters
    !> \param   self       class(elpa_impl_t) the allocated ELPA object
    !> \param   name       string, the key
    !> \param   value      integer, the value to be set
    !> \result  error      integer, the error code
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


    !c> void elpa_set_integer(elpa_t handle, const char *name, int value, int *error);
    subroutine elpa_set_integer_c(handle, name_p, value, error) bind(C, name="elpa_set_integer")
      type(c_ptr), intent(in), value :: handle
      type(elpa_impl_t), pointer :: self
      type(c_ptr), intent(in), value :: name_p
      character(len=elpa_strlen_c(name_p)), pointer :: name
      integer(kind=c_int), intent(in), value :: value
      integer(kind=c_int), optional, intent(in) :: error

      call c_f_pointer(handle, self)
      call c_f_pointer(name_p, name)
      call elpa_set_integer(self, name, value, error)
    end subroutine


    !> \brief function to get an integer key/value pair
    !> Parameters
    !> \param   self       class(elpa_impl_t) the allocated ELPA object
    !> \param   name       string, the key
    !> \param   error      integer, optional, to store an error code
    !> \result  value      integer, the value of the key/vaue pair
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


    !c> int elpa_get_integer(elpa_t handle, const char *name, int *error);
    function elpa_get_integer_c(handle, name_p, error) result(value) bind(C, name="elpa_get_integer")
      type(c_ptr), intent(in), value :: handle
      type(elpa_impl_t), pointer :: self
      type(c_ptr), intent(in), value :: name_p
      character(len=elpa_strlen_c(name_p)), pointer :: name
      integer(kind=c_int)  :: value
      integer(kind=c_int), optional, intent(inout) :: error

      call c_f_pointer(handle, self)
      call c_f_pointer(name_p, name)
      value = elpa_get_integer(self, name, error)
    end function


    !> \brief function to check whether a key/value pair is set
    !> Parameters
    !> \param   self       class(elpa_impl_t) the allocated ELPA object
    !> \param   name       string, the key
    !> \result  state      integer, the state of the key/value pair
    function elpa_is_set(self, name) result(state)
      use iso_c_binding
      use elpa_generated_fortran_interfaces
      class(elpa_impl_t)       :: self
      character(*), intent(in) :: name
      integer                  :: state

      state = elpa_index_value_is_set_c(self%index, name // c_null_char)
    end function

    !> \brief function to check whether a key/value pair can be set
    !> Parameters
    !> \param   self       class(elpa_impl_t) the allocated ELPA object
    !> \param   name       string, the key
    !> \param   value      integer, value
    !> \result  error      integer, error code
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


    !c> void elpa_set_double(elpa_t handle, const char *name, double value, int *error);
    subroutine elpa_set_double_c(handle, name_p, value, error) bind(C, name="elpa_set_double")
      type(c_ptr), intent(in), value :: handle
      type(elpa_impl_t), pointer :: self
      type(c_ptr), intent(in), value :: name_p
      character(len=elpa_strlen_c(name_p)), pointer :: name
      real(kind=c_double), intent(in), value :: value
      integer(kind=c_int), optional, intent(in) :: error

      call c_f_pointer(handle, self)
      call c_f_pointer(name_p, name)
      call elpa_set_double(self, name, value, error)
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

    !c> int elpa_get_double(elpa_t handle, const char *name, int *error);
    function elpa_get_double_c(handle, name_p, error) result(value) bind(C, name="elpa_get_double")
      type(c_ptr), intent(in), value :: handle
      type(elpa_impl_t), pointer :: self
      type(c_ptr), intent(in), value :: name_p
      character(len=elpa_strlen_c(name_p)), pointer :: name
      real(kind=c_double)  :: value
      integer(kind=c_int), optional, intent(inout) :: error

      call c_f_pointer(handle, self)
      call c_f_pointer(name_p, name)
      value = elpa_get_double(self, name, error)
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


    function elpa_get_time(self, name1, name2, name3, name4, name5, name6) result(s)
      class(elpa_impl_t), intent(in) :: self
      ! this is clunky, but what can you do..
      character(len=*), intent(in), optional :: name1, name2, name3, name4, name5, name6
      real(kind=c_double) :: s

      s = self%timer%get(name1, name2, name3, name4, name5, name6)
    end function


    subroutine elpa_print_times(self)
      class(elpa_impl_t), intent(in) :: self
      call self%timer%print()
    end subroutine


    subroutine elpa_solve_d(self, a, ev, q, error)
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
      integer(kind=c_int) :: error_actual
      logical             :: success_l


      if (self%get("solver") .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_real_1stage_double_impl(self, a, ev, q)

      else if (self%get("solver") .eq. ELPA_SOLVER_2STAGE) then
        success_l = elpa_solve_evp_real_2stage_double_impl(self, a, ev, q)
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

    !c> void elpa_solve_d(elpa_t handle, double *a, double *ev, double *q, int *error);
    subroutine elpa_solve_d_c(handle, a_p, ev_p, q_p, error) bind(C, name="elpa_solve_d")
      type(c_ptr), intent(in), value :: handle, a_p, ev_p, q_p
      integer(kind=c_int), optional, intent(in) :: error

      real(kind=c_double), pointer :: a(:, :), q(:, :), ev(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])
      call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])

      call elpa_solve_d(self, a, ev, q, error)
    end subroutine

    subroutine elpa_solve_f(self, a, ev, q, error)
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
      integer(kind=c_int) :: error_actual
      logical             :: success_l

#ifdef WANT_SINGLE_PRECISION_REAL

      if (self%get("solver") .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_real_1stage_single_impl(self, a, ev, q)

      else if (self%get("solver") .eq. ELPA_SOLVER_2STAGE) then
        success_l = elpa_solve_evp_real_2stage_single_impl(self, a, ev, q)
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


    !c> void elpa_solve_f(elpa_t handle, float *a, float *ev, float *q, int *error);
    subroutine elpa_solve_f_c(handle, a_p, ev_p, q_p, error) bind(C, name="elpa_solve_f")
      type(c_ptr), intent(in), value :: handle, a_p, ev_p, q_p
      integer(kind=c_int), optional, intent(in) :: error

      real(kind=c_float), pointer :: a(:, :), q(:, :), ev(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])
      call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])

      call elpa_solve_f(self, a, ev, q, error)
    end subroutine


    subroutine elpa_solve_dc(self, a, ev, q, error)
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
      integer(kind=c_int)            :: error_actual
      logical                        :: success_l

      if (self%get("solver") .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_complex_1stage_double_impl(self, a, ev, q)

      else if (self%get("solver") .eq. ELPA_SOLVER_2STAGE) then
        success_l = elpa_solve_evp_complex_2stage_double_impl(self,  a, ev, q)
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


    !c> void elpa_solve_dc(elpa_t handle, double complex *a, double *ev, double complex *q, int *error);
    subroutine elpa_solve_dc_c(handle, a_p, ev_p, q_p, error) bind(C, name="elpa_solve_dc")
      type(c_ptr), intent(in), value :: handle, a_p, ev_p, q_p
      integer(kind=c_int), optional, intent(in) :: error

      complex(kind=c_double_complex), pointer :: a(:, :), q(:, :)
      real(kind=c_double), pointer :: ev(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])
      call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])

      call elpa_solve_dc(self, a, ev, q, error)
    end subroutine


    subroutine elpa_solve_fc(self, a, ev, q, error)
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
      integer(kind=c_int)           :: error_actual
      logical                       :: success_l

#ifdef WANT_SINGLE_PRECISION_COMPLEX

      if (self%get("solver") .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_complex_1stage_single_impl(self, a, ev, q)

      else if (self%get("solver") .eq. ELPA_SOLVER_2STAGE) then
        success_l = elpa_solve_evp_complex_2stage_single_impl(self,  a, ev, q)
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


    !c> void elpa_solve_fc(elpa_t handle, float complex *a, float *ev, float complex *q, int *error);
    subroutine elpa_solve_fc_c(handle, a_p, ev_p, q_p, error) bind(C, name="elpa_solve_fc")
      type(c_ptr), intent(in), value :: handle, a_p, ev_p, q_p
      integer(kind=c_int), optional, intent(in) :: error

      complex(kind=c_float_complex), pointer :: a(:, :), q(:, :)
      real(kind=c_float), pointer :: ev(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])
      call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])

      call elpa_solve_fc(self, a, ev, q, error)
    end subroutine


    subroutine elpa_hermitian_multiply_d (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
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

      success_l = elpa_mult_at_b_real_double_impl(self, uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                                  c, ldc, ldcCols)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in hermitian_multiply() and you did not check for errors!"
      endif
    end subroutine


    subroutine elpa_hermitian_multiply_f (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
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
      success_l = elpa_mult_at_b_real_single_impl(self, uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                                  c, ldc, ldcCols)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in hermitian_multiply() and you did not check for errors!"
      endif
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
    end subroutine


    subroutine elpa_hermitian_multiply_dc (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
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

      success_l = elpa_mult_ah_b_complex_double_impl(self, uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                                     c, ldc, ldcCols)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in hermitian_multiply() and you did not check for errors!"
      endif
    end subroutine


    subroutine elpa_hermitian_multiply_fc (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
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
      success_l = elpa_mult_ah_b_complex_single_impl(self, uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                                     c, ldc, ldcCols)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in hermitian_multiply() and you did not check for errors!"
      endif
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
    end subroutine


    subroutine elpa_cholesky_d (self, a, error)
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

      success_l = elpa_cholesky_real_double_impl (self, a)
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


    !c> void elpa_cholesky_d(elpa_t handle, double *a, int *error);
    subroutine elpa_choleksy_d_c(handle, a_p, error) bind(C, name="elpa_cholesky_d")
      type(c_ptr), intent(in), value :: handle, a_p
      integer(kind=c_int), optional, intent(in) :: error

      real(kind=c_double), pointer :: a(:, :)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

      call elpa_cholesky_d(self, a, error)
    end subroutine


    subroutine elpa_cholesky_f(self, a, error)
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

#if WANT_SINGLE_PRECISION_REAL
      success_l = elpa_cholesky_real_single_impl (self, a)
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


    !c> void elpa_cholesky_f(elpa_t handle, float *a, int *error);
    subroutine elpa_choleksy_f_c(handle, a_p, error) bind(C, name="elpa_cholesky_f")
      type(c_ptr), intent(in), value :: handle, a_p
      integer(kind=c_int), optional, intent(in) :: error

      real(kind=c_float), pointer :: a(:, :)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

      call elpa_cholesky_f(self, a, error)
    end subroutine


    subroutine elpa_cholesky_dc (self, a, error)
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

      success_l = elpa_cholesky_complex_double_impl (self, a)
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


    !c> void elpa_cholesky_dc(elpa_t handle, double complex *a, int *error);
    subroutine elpa_choleksy_dc_c(handle, a_p, error) bind(C, name="elpa_cholesky_dc")
      type(c_ptr), intent(in), value :: handle, a_p
      integer(kind=c_int), optional, intent(in) :: error

      complex(kind=c_double_complex), pointer :: a(:, :)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

      call elpa_cholesky_dc(self, a, error)
    end subroutine


    subroutine elpa_cholesky_fc (self, a, error)
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

#if WANT_SINGLE_PRECISION_COMPLEX
      success_l = elpa_cholesky_complex_single_impl (self, a)
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


    !c> void elpa_cholesky_fc(elpa_t handle, float complex *a, int *error);
    subroutine elpa_choleksy_fc_c(handle, a_p, error) bind(C, name="elpa_cholesky_fc")
      type(c_ptr), intent(in), value :: handle, a_p
      integer(kind=c_int), optional, intent(in) :: error

      complex(kind=c_float_complex), pointer :: a(:, :)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

      call elpa_cholesky_fc(self, a, error)
    end subroutine


    subroutine elpa_invert_trm_d (self, a, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      class(elpa_impl_t)              :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_double)             :: a(self%local_nrows,*)
#else
      real(kind=c_double)             :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
      logical                         :: success_l
      integer(kind=c_int)             :: error_actual

      success_l = elpa_invert_trm_real_double_impl (self, a)
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


    !c> void elpa_invert_trm_d(elpa_t handle, double *a, int *error);
    subroutine elpa_invert_trm_d_c(handle, a_p, error) bind(C, name="elpa_invert_trm_d")
      type(c_ptr), intent(in), value :: handle, a_p
      integer(kind=c_int), optional, intent(in) :: error

      real(kind=c_double), pointer :: a(:, :)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

      call elpa_invert_trm_d(self, a, error)
    end subroutine


    subroutine elpa_invert_trm_f (self, a, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      class(elpa_impl_t)              :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_float)              :: a(self%local_nrows,*)
#else
      real(kind=c_float)              :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
      logical                         :: success_l
      integer(kind=c_int)             :: error_actual

#if WANT_SINGLE_PRECISION_REAL
      success_l = elpa_invert_trm_real_single_impl (self, a)
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


    !c> void elpa_invert_trm_f(elpa_t handle, float *a, int *error);
    subroutine elpa_invert_trm_f_c(handle, a_p, error) bind(C, name="elpa_invert_trm_f")
      type(c_ptr), intent(in), value :: handle, a_p
      integer(kind=c_int), optional, intent(in) :: error

      real(kind=c_float), pointer :: a(:, :)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

      call elpa_invert_trm_f(self, a, error)
    end subroutine


    subroutine elpa_invert_trm_dc (self, a, error)
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

      success_l = elpa_invert_trm_complex_double_impl (self, a)
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


    !c> void elpa_invert_trm_dc(elpa_t handle, double complex *a, int *error);
    subroutine elpa_invert_trm_dc_c(handle, a_p, error) bind(C, name="elpa_invert_trm_dc")
      type(c_ptr), intent(in), value :: handle, a_p
      integer(kind=c_int), optional, intent(in) :: error

      complex(kind=c_double_complex), pointer :: a(:, :)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

      call elpa_invert_trm_dc(self, a, error)
    end subroutine


    subroutine elpa_invert_trm_fc (self, a, error)
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

#if WANT_SINGLE_PRECISION_COMPLEX
      success_l = elpa_invert_trm_complex_single_impl (self, a)
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


    !c> void elpa_invert_trm_fc(elpa_t handle, float complex *a, int *error);
    subroutine elpa_invert_trm_fc_c(handle, a_p, error) bind(C, name="elpa_invert_trm_fc")
      type(c_ptr), intent(in), value :: handle, a_p
      integer(kind=c_int), optional, intent(in) :: error

      complex(kind=c_float_complex), pointer :: a(:, :)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

      call elpa_invert_trm_fc(self, a, error)
    end subroutine


    !> \todo e should have dimension (na - 1)
    subroutine elpa_solve_tridi_d (self, d, e, q, error)
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

      success_l = elpa_solve_tridi_double_impl(self, d, e, q)
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


    subroutine elpa_solve_tridi_f (self, d, e, q, error)
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

#ifdef WANT_SINGLE_PRECISION_REAL
      success_l = elpa_solve_tridi_single_impl(self, d, e, q)
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


end module
