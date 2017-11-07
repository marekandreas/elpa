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

!> \brief Fortran module which provides the actual implementation of the API. Do not use directly! Use the module "elpa"
module elpa_impl
  use elpa_abstract_impl
  use, intrinsic :: iso_c_binding
  implicit none

  private
  public :: elpa_impl_allocate

!> \brief Definition of the extended elpa_impl_t type
  type, extends(elpa_abstract_impl_t) :: elpa_impl_t
   private
   integer :: communicators_owned

   !> \brief methods available with the elpa_impl_t type
   contains
     !> \brief the puplic methods
     ! con-/destructor
     procedure, public :: setup => elpa_setup                   !< a setup method: implemented in elpa_setup
     procedure, public :: destroy => elpa_destroy               !< a destroy method: implemented in elpa_destroy

     ! KV store
     procedure, public :: is_set => elpa_is_set                 !< a method to check whether a key/value pair has been set : implemented
                                                                !< in elpa_is_set
     procedure, public :: can_set => elpa_can_set               !< a method to check whether a key/value pair can be set : implemented
                                                                !< in elpa_can_set


     ! timer
     procedure, public :: get_time => elpa_get_time
     procedure, public :: print_times => elpa_print_times
     procedure, public :: timer_start => elpa_timer_start
     procedure, public :: timer_stop => elpa_timer_stop


     !> \brief the implemenation methods

     procedure, public :: elpa_eigenvectors_d                  !< public methods to implement the solve step for real/complex
                                                               !< double/single matrices
     procedure, public :: elpa_eigenvectors_f
     procedure, public :: elpa_eigenvectors_dc
     procedure, public :: elpa_eigenvectors_fc

     procedure, public :: elpa_eigenvalues_d                   !< public methods to implement the solve step for real/complex
                                                               !< double/single matrices; only the eigenvalues are computed
     procedure, public :: elpa_eigenvalues_f
     procedure, public :: elpa_eigenvalues_dc
     procedure, public :: elpa_eigenvalues_fc

     procedure, public :: elpa_generalized_eigenvectors_d      !< public methods to implement the solve step for generalized 
                                                               !< eigenproblem and real/complex double/single matrices
     procedure, public :: elpa_generalized_eigenvectors_f
     procedure, public :: elpa_generalized_eigenvectors_dc
     procedure, public :: elpa_generalized_eigenvectors_fc

     procedure, public :: elpa_hermitian_multiply_d            !< public methods to implement a "hermitian" multiplication of matrices a and b
     procedure, public :: elpa_hermitian_multiply_f            !< for real valued matrices:   a**T * b
     procedure, public :: elpa_hermitian_multiply_dc           !< for complex valued matrices:   a**H * b
     procedure, public :: elpa_hermitian_multiply_fc

     procedure, public :: elpa_cholesky_d                      !< public methods to implement the cholesky factorisation of
                                                               !< real/complex double/single matrices
     procedure, public :: elpa_cholesky_f
     procedure, public :: elpa_cholesky_dc
     procedure, public :: elpa_cholesky_fc

     procedure, public :: elpa_invert_trm_d                    !< public methods to implement the inversion of a triangular
                                                               !< real/complex double/single matrix
     procedure, public :: elpa_invert_trm_f
     procedure, public :: elpa_invert_trm_dc
     procedure, public :: elpa_invert_trm_fc

     procedure, public :: elpa_solve_tridiagonal_d             !< public methods to implement the solve step for a real valued
     procedure, public :: elpa_solve_tridiagonal_f             !< double/single tridiagonal matrix

     procedure, public :: associate_int => elpa_associate_int  !< public method to set some pointers

     procedure, private :: elpa_transform_generalized_d
     procedure, private :: elpa_transform_back_generalized_d
     procedure, private :: elpa_transform_generalized_dc
     procedure, private :: elpa_transform_back_generalized_dc
#ifdef WANT_SINGLE_PRECISION_REAL
     procedure, private :: elpa_transform_generalized_f
     procedure, private :: elpa_transform_back_generalized_f
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
     procedure, private :: elpa_transform_generalized_fc
     procedure, private :: elpa_transform_back_generalized_fc
#endif
  end type elpa_impl_t


  !> \brief the implementation of the generic methods
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

    !c> /*! \brief C interface for the implementation of the elpa_allocate method
    !c> *
    !c> *  \param  none
    !c> *  \result elpa_t handle
    !c> */
    !c> elpa_t elpa_allocate();
    function elpa_impl_allocate_c(error) result(ptr) bind(C, name="elpa_allocate")
      integer(kind=c_int) :: error
      type(c_ptr) :: ptr
      type(elpa_impl_t), pointer :: obj

      obj => elpa_impl_allocate(error)
      ptr = c_loc(obj)
    end function

    !c> /*! \brief C interface for the implementation of the elpa_deallocate method
    !c> *
    !c> *  \param  elpa_t  handle of ELPA object to be deallocated
    !c> *  \result void
    !c> */
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
      use elpa_utilities, only : error_unit
#ifdef WITH_MPI
      use elpa_mpi
#endif
      class(elpa_impl_t), intent(inout)   :: self
      integer                             :: error, timings

#ifdef WITH_MPI
      integer                             :: mpi_comm_parent, mpi_comm_rows, mpi_comm_cols, &
                                             mpierr, mpierr2, process_row, process_col, mpi_string_length
      character(len=MPI_MAX_ERROR_STRING) :: mpierr_string
#endif

#ifdef HAVE_DETAILED_TIMINGS
      call self%get("timings",timings)
      if (timings == 1) then
        call self%timer%enable()
      endif
#endif

      error = ELPA_OK

#ifdef WITH_MPI
      ! Create communicators ourselves
      if (self%is_set("mpi_comm_parent") == 1 .and. &
          self%is_set("process_row") == 1 .and. &
          self%is_set("process_col") == 1) then

        call self%get("mpi_comm_parent", mpi_comm_parent)
        call self%get("process_row", process_row)
        call self%get("process_col", process_col)

        ! mpi_comm_rows is used for communicating WITHIN rows, i.e. all processes
        ! having the same column coordinate share one mpi_comm_rows.
        ! So the "color" for splitting is process_col and the "key" is my row coordinate.
        ! Analogous for mpi_comm_cols

        call mpi_comm_split(mpi_comm_parent,process_col,process_row,mpi_comm_rows,mpierr)

        if (mpierr .ne. MPI_SUCCESS) then
          call MPI_ERROR_STRING(mpierr,mpierr_string, mpi_string_length, mpierr2)
          write(error_unit,*) "MPI ERROR occured during mpi_comm_split for row communicator: ", trim(mpierr_string)
          return
        endif

        call mpi_comm_split(mpi_comm_parent,process_row,process_col,mpi_comm_cols, mpierr)
        if (mpierr .ne. MPI_SUCCESS) then
          call MPI_ERROR_STRING(mpierr,mpierr_string, mpi_string_length, mpierr2)
          write(error_unit,*) "MPI ERROR occured during mpi_comm_split for col communicator: ", trim(mpierr_string)
          return
        endif

        call self%set("mpi_comm_rows", mpi_comm_rows)
        call self%set("mpi_comm_cols", mpi_comm_cols)

        ! remember that we created those communicators and we need to free them later
        self%communicators_owned = 1

        error = ELPA_OK
        return
      endif

      ! Externally supplied communicators
      if (self%is_set("mpi_comm_rows") == 1 .and. self%is_set("mpi_comm_cols") == 1) then
        self%communicators_owned = 0
        error = ELPA_OK
        return
      endif

      ! Otherwise parameters are missing
      error = ELPA_ERROR
#endif

    end function

    !c> /*! \brief C interface for the implementation of the elpa_setup method
    !c> *
    !c> *  \param  elpa_t  handle of the ELPA object which describes the problem to
    !c> *                  be set up
    !c> *  \result int     error code, which can be queried with elpa_strerr
    !c> */
    !c> int elpa_setup(elpa_t handle);
    function elpa_setup_c(handle) result(error) bind(C, name="elpa_setup")
      type(c_ptr), intent(in), value :: handle
      type(elpa_impl_t), pointer :: self
      integer(kind=c_int) :: error

      call c_f_pointer(handle, self)
      error = self%setup()
    end function


    !c> /*! \brief C interface for the implementation of the elpa_set_integer method
    !c> *  This method is available to the user as C generic elpa_set method
    !c> *
    !c> *  \param  handle  handle of the ELPA object for which a key/value pair should be set
    !c> *  \param  name    the name of the key
    !c> *  \param  value   the value to be set for the key
    !c> *  \param  error   on return the error code, which can be queried with elpa_strerr()
    !c> *  \result void
    !c> */
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


    !c> /*! \brief C interface for the implementation of the elpa_get_integer method
    !c> *  This method is available to the user as C generic elpa_get method
    !c> *
    !c> *  \param  handle  handle of the ELPA object for which a key/value pair should be queried
    !c> *  \param  name    the name of the key
    !c> *  \param  value   the value to be obtain for the key
    !c> *  \param  error   on return the error code, which can be queried with elpa_strerr()
    !c> *  \result void
    !c> */
    !c> void elpa_get_integer(elpa_t handle, const char *name, int *value, int *error);
    subroutine elpa_get_integer_c(handle, name_p, value, error) bind(C, name="elpa_get_integer")
      type(c_ptr), intent(in), value :: handle
      type(elpa_impl_t), pointer :: self
      type(c_ptr), intent(in), value :: name_p
      character(len=elpa_strlen_c(name_p)), pointer :: name
      integer(kind=c_int)  :: value
      integer(kind=c_int), optional, intent(inout) :: error

      call c_f_pointer(handle, self)
      call c_f_pointer(name_p, name)
      call elpa_get_integer(self, name, value, error)
    end subroutine


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

      call self%get(option_name, val, actual_error)
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


    !c> /*! \brief C interface for the implementation of the elpa_set_double method
    !c> *  This method is available to the user as C generic elpa_set method
    !c> *
    !c> *  \param  handle  handle of the ELPA object for which a key/value pair should be set
    !c> *  \param  name    the name of the key
    !c> *  \param  value   the value to be set for the key
    !c> *  \param  error   on return the error code, which can be queried with elpa_strerr()
    !c> *  \result void
    !c> */
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


    !c> /*! \brief C interface for the implementation of the elpa_get_double method
    !c> *  This method is available to the user as C generic elpa_get method
    !c> *
    !c> *  \param  handle  handle of the ELPA object for which a key/value pair should be queried
    !c> *  \param  name    the name of the key
    !c> *  \param  value   the value to be obtain for the key
    !c> *  \param  error   on return the error code, which can be queried with elpa_strerr()
    !c> *  \result void
    !c> */
    !c> void elpa_get_double(elpa_t handle, const char *name, double *value, int *error);
    subroutine elpa_get_double_c(handle, name_p, value, error) bind(C, name="elpa_get_double")
      type(c_ptr), intent(in), value :: handle
      type(elpa_impl_t), pointer :: self
      type(c_ptr), intent(in), value :: name_p
      character(len=elpa_strlen_c(name_p)), pointer :: name
      real(kind=c_double)  :: value
      integer(kind=c_int), optional, intent(inout) :: error

      call c_f_pointer(handle, self)
      call c_f_pointer(name_p, name)
      call elpa_get_double(self, name, value, error)
    end subroutine


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

#ifdef HAVE_DETAILED_TIMINGS
      s = self%timer%get(name1, name2, name3, name4, name5, name6)
#else
      s = -1.0
#endif
    end function


    subroutine elpa_print_times(self, name1, name2, name3, name4)
      class(elpa_impl_t), intent(in) :: self
      character(len=*), intent(in), optional :: name1, name2, name3, name4
#ifdef HAVE_DETAILED_TIMINGS
      call self%timer%print(name1, name2, name3, name4)
#endif
    end subroutine


    subroutine elpa_timer_start(self, name)
      class(elpa_impl_t), intent(inout) :: self
      character(len=*), intent(in) :: name
#ifdef HAVE_DETAILED_TIMINGS
      call self%timer%start(name)
#endif
    end subroutine


    subroutine elpa_timer_stop(self, name)
      class(elpa_impl_t), intent(inout) :: self
      character(len=*), intent(in) :: name
#ifdef HAVE_DETAILED_TIMINGS
      call self%timer%stop(name)
#endif
    end subroutine


    !>  \brief elpa_eigenvectors_d: class method to solve the eigenvalue problem for double real matrices
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
    !>  blocksize, the number of eigenvectors
    !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param q                                    On output: Eigenvectors of a
    !>                                              Distribution is like in Scalapack.
    !>                                              Must be always dimensioned to the full size (corresponding to (na,na))
    !>                                              even if only a part of the eigenvalues is needed.
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_eigenvectors_d(self, a, ev, q, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit
      use iso_c_binding
      class(elpa_impl_t)  :: self

#ifdef USE_ASSUMED_SIZE
      real(kind=c_double) :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      real(kind=c_double) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_double) :: ev(self%na)

      integer, optional   :: error
      integer(kind=c_int) :: solver
      logical             :: success_l


      call self%get("solver", solver)
      if (solver .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_real_1stage_double_impl(self, a, ev, q)

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
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

    !c> void elpa_eigenvectors_d(elpa_t handle, double *a, double *ev, double *q, int *error);
    subroutine elpa_eigenvectors_d_c(handle, a_p, ev_p, q_p, error) bind(C, name="elpa_eigenvectors_d")
      type(c_ptr), intent(in), value :: handle, a_p, ev_p, q_p
      integer(kind=c_int), optional, intent(in) :: error

      real(kind=c_double), pointer :: a(:, :), q(:, :), ev(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])
      call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])

      call elpa_eigenvectors_d(self, a, ev, q, error)
    end subroutine


    !>  \brief elpa_eigenvectors_f: class method to solve the eigenvalue problem for float real matrices
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
    !>  blocksize, the number of eigenvectors
    !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param q                                    On output: Eigenvectors of a
    !>                                              Distribution is like in Scalapack.
    !>                                              Must be always dimensioned to the full size (corresponding to (na,na))
    !>                                              even if only a part of the eigenvalues is needed.
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_eigenvectors_f(self, a, ev, q, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit
      use iso_c_binding
      class(elpa_impl_t)  :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_float)  :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      real(kind=c_float)  :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_float)  :: ev(self%na)

      integer, optional   :: error
      integer(kind=c_int) :: solver
#ifdef WANT_SINGLE_PRECISION_REAL
      logical             :: success_l

      call self%get("solver",solver)
      if (solver .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_real_1stage_single_impl(self, a, ev, q)

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
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


    !c> void elpa_eigenvectors_f(elpa_t handle, float *a, float *ev, float *q, int *error);
    subroutine elpa_eigenvectors_f_c(handle, a_p, ev_p, q_p, error) bind(C, name="elpa_eigenvectors_f")
      type(c_ptr), intent(in), value :: handle, a_p, ev_p, q_p
      integer(kind=c_int), optional, intent(in) :: error

      real(kind=c_float), pointer :: a(:, :), q(:, :), ev(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])
      call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])

      call elpa_eigenvectors_f(self, a, ev, q, error)
    end subroutine


    !>  \brief elpa_eigenvectors_dc: class method to solve the eigenvalue problem for double complex matrices
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
    !>  blocksize, the number of eigenvectors
    !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param q                                    On output: Eigenvectors of a
    !>                                              Distribution is like in Scalapack.
    !>                                              Must be always dimensioned to the full size (corresponding to (na,na))
    !>                                              even if only a part of the eigenvalues is needed.
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_eigenvectors_dc(self, a, ev, q, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit
      use iso_c_binding
      class(elpa_impl_t)             :: self

#ifdef USE_ASSUMED_SIZE
      complex(kind=c_double_complex) :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      complex(kind=c_double_complex) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_double)            :: ev(self%na)

      integer, optional              :: error
      integer(kind=c_int)            :: solver
      logical                        :: success_l

      call self%get("solver", solver)
      if (solver .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_complex_1stage_double_impl(self, a, ev, q)

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
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


    !c> void elpa_eigenvectors_dc(elpa_t handle, double complex *a, double *ev, double complex *q, int *error);
    subroutine elpa_eigenvectors_dc_c(handle, a_p, ev_p, q_p, error) bind(C, name="elpa_eigenvectors_dc")
      type(c_ptr), intent(in), value :: handle, a_p, ev_p, q_p
      integer(kind=c_int), optional, intent(in) :: error

      complex(kind=c_double_complex), pointer :: a(:, :), q(:, :)
      real(kind=c_double), pointer :: ev(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])
      call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])

      call elpa_eigenvectors_dc(self, a, ev, q, error)
    end subroutine


    !>  \brief elpa_eigenvectors_fc: class method to solve the eigenvalue problem for float complex matrices
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
    !>  blocksize, the number of eigenvectors
    !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param q                                    On output: Eigenvectors of a
    !>                                              Distribution is like in Scalapack.
    !>                                              Must be always dimensioned to the full size (corresponding to (na,na))
    !>                                              even if only a part of the eigenvalues is needed.
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_eigenvectors_fc(self, a, ev, q, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit

      use iso_c_binding
      class(elpa_impl_t)            :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_float_complex) :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      complex(kind=c_float_complex) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_float)            :: ev(self%na)

      integer, optional             :: error
      integer(kind=c_int)           :: solver
#ifdef WANT_SINGLE_PRECISION_COMPLEX
      logical                       :: success_l

      call self%get("solver", solver)
      if (solver .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_complex_1stage_single_impl(self, a, ev, q)

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
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


    !c> void elpa_eigenvectors_fc(elpa_t handle, float complex *a, float *ev, float complex *q, int *error);
    subroutine elpa_eigenvectors_fc_c(handle, a_p, ev_p, q_p, error) bind(C, name="elpa_eigenvectors_fc")
      type(c_ptr), intent(in), value :: handle, a_p, ev_p, q_p
      integer(kind=c_int), optional, intent(in) :: error

      complex(kind=c_float_complex), pointer :: a(:, :), q(:, :)
      real(kind=c_float), pointer :: ev(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])
      call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])

      call elpa_eigenvectors_fc(self, a, ev, q, error)
    end subroutine




    !>  \brief elpa_eigenvalues_d: class method to solve the eigenvalue problem for double real matrices
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
    !>  blocksize, the number of eigenvectors
    !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_eigenvalues_d(self, a, ev, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit
      use iso_c_binding
      class(elpa_impl_t)  :: self

#ifdef USE_ASSUMED_SIZE
      real(kind=c_double) :: a(self%local_nrows, *)
#else
      real(kind=c_double) :: a(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_double) :: ev(self%na)

      integer, optional   :: error
      integer(kind=c_int) :: solver
      logical             :: success_l


      call self%get("solver", solver)
      if (solver .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_real_1stage_double_impl(self, a, ev)

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
        success_l = elpa_solve_evp_real_2stage_double_impl(self, a, ev)
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

    !c> void elpa_eigenvalues_d(elpa_t handle, double *a, double *ev, int *error);
    subroutine elpa_eigenvalues_d_c(handle, a_p, ev_p, error) bind(C, name="elpa_eigenvalues_d")
      type(c_ptr), intent(in), value :: handle, a_p, ev_p
      integer(kind=c_int), optional, intent(in) :: error

      real(kind=c_double), pointer :: a(:, :), ev(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])

      call elpa_eigenvalues_d(self, a, ev, error)
    end subroutine


    !>  \brief elpa_eigenvectors_f: class method to solve the eigenvalue problem for float real matrices
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
    !>  blocksize, the number of eigenvectors
    !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_eigenvalues_f(self, a, ev, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit
      use iso_c_binding
      class(elpa_impl_t)  :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_float)  :: a(self%local_nrows, *)
#else
      real(kind=c_float)  :: a(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_float)  :: ev(self%na)

      integer, optional   :: error
      integer(kind=c_int) :: solver
#ifdef WANT_SINGLE_PRECISION_REAL
      logical             :: success_l

      call self%get("solver",solver)
      if (solver .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_real_1stage_single_impl(self, a, ev)

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
        success_l = elpa_solve_evp_real_2stage_single_impl(self, a, ev)
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


    !c> void elpa_eigenvalues_f(elpa_t handle, float *a, float *ev, int *error);
    subroutine elpa_eigenvalues_f_c(handle, a_p, ev_p,  error) bind(C, name="elpa_eigenvalues_f")
      type(c_ptr), intent(in), value :: handle, a_p, ev_p
      integer(kind=c_int), optional, intent(in) :: error

      real(kind=c_float), pointer :: a(:, :), ev(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])

      call elpa_eigenvalues_f(self, a, ev, error)
    end subroutine


    !>  \brief elpa_eigenvalues_dc: class method to solve the eigenvalue problem for double complex matrices
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
    !>  blocksize, the number of eigenvectors
    !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_eigenvalues_dc(self, a, ev, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit
      use iso_c_binding
      class(elpa_impl_t)             :: self

#ifdef USE_ASSUMED_SIZE
      complex(kind=c_double_complex) :: a(self%local_nrows, *)
#else
      complex(kind=c_double_complex) :: a(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_double)            :: ev(self%na)

      integer, optional              :: error
      integer(kind=c_int)            :: solver
      logical                        :: success_l

      call self%get("solver", solver)
      if (solver .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_complex_1stage_double_impl(self, a, ev)

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
        success_l = elpa_solve_evp_complex_2stage_double_impl(self,  a, ev)
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


    !c> void elpa_eigenvalues_dc(elpa_t handle, double complex *a, double *ev, int *error);
    subroutine elpa_eigenvalues_dc_c(handle, a_p, ev_p, error) bind(C, name="elpa_eigenvalues_dc")
      type(c_ptr), intent(in), value :: handle, a_p, ev_p
      integer(kind=c_int), optional, intent(in) :: error

      complex(kind=c_double_complex), pointer :: a(:, :)
      real(kind=c_double), pointer :: ev(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])

      call elpa_eigenvalues_dc(self, a, ev, error)
    end subroutine


    !>  \brief elpa_eigenvalues_fc: class method to solve the eigenvalue problem for float complex matrices
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
    !>  blocksize, the number of eigenvectors
    !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_eigenvalues_fc(self, a, ev, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit

      use iso_c_binding
      class(elpa_impl_t)            :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_float_complex) :: a(self%local_nrows, *)
#else
      complex(kind=c_float_complex) :: a(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_float)            :: ev(self%na)

      integer, optional             :: error
      integer(kind=c_int)           :: solver
#ifdef WANT_SINGLE_PRECISION_COMPLEX
      logical                       :: success_l

      call self%get("solver", solver)
      if (solver .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_complex_1stage_single_impl(self, a, ev)

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
        success_l = elpa_solve_evp_complex_2stage_single_impl(self,  a, ev)
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


    !c> void elpa_eigenvalues_fc(elpa_t handle, float complex *a, float *ev, int *error);
    subroutine elpa_eigenvalues_fc_c(handle, a_p, ev_p, error) bind(C, name="elpa_eigenvalues_fc")
      type(c_ptr), intent(in), value :: handle, a_p, ev_p
      integer(kind=c_int), optional, intent(in) :: error

      complex(kind=c_float_complex), pointer :: a(:, :)
      real(kind=c_float), pointer :: ev(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])

      call elpa_eigenvalues_fc(self, a, ev, error)
    end subroutine

!********************************************************************************************************
!             GENERALIZED EIGENVECTOR PROBLEM
!********************************************************************************************************

    !>  \brief elpa_generalized_eigenvectors_d: class method to solve the eigenvalue problem for double real matrices
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
    !>  blocksize, the number of eigenvectors
    !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param q                                    On output: Eigenvectors of a
    !>                                              Distribution is like in Scalapack.
    !>                                              Must be always dimensioned to the full size (corresponding to (na,na))
    !>                                              even if only a part of the eigenvalues is needed.
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_generalized_eigenvectors_d(self, a, b, ev, q, sc_desc, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit
      use iso_c_binding
      class(elpa_impl_t)  :: self

#ifdef USE_ASSUMED_SIZE
      real(kind=c_double) :: a(self%local_nrows, *), b(self%local_nrows, *), q(self%local_nrows, *)
#else
      real(kind=c_double) :: a(self%local_nrows, self%local_ncols), b(self%local_nrows, self%local_ncols), &
                             q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_double) :: ev(self%na)
      integer             :: sc_desc(SC_DESC_LEN)

      integer, optional   :: error
      integer             :: error_l
      integer(kind=c_int) :: solver
      logical             :: success_l

      call self%elpa_transform_generalized_d(a, b, sc_desc, error_l)
      if (present(error)) then
          error = error_l
      else if (error_l .ne. ELPA_OK) then
        write(error_unit,'(a)') "ELPA: Error in transform_generalized() and you did not check for errors!"
      endif

      call self%get("solver", solver)
      if (solver .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_real_1stage_double_impl(self, a, ev, q)

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
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

      call self%elpa_transform_back_generalized_d(b, q, sc_desc, error_l)
      if (present(error)) then
          error = error_l
      else if (error_l .ne. ELPA_OK) then
        write(error_unit,'(a)') "ELPA: Error in transform_back_generalized() and you did not check for errors!"
      endif
    end subroutine

    !c> void elpa_generalized_eigenvectors_d(elpa_t handle, double *a, double *ev, double *q, int *error);
    subroutine elpa_generalized_eigenvectors_d_c(handle, a_p, b_p, ev_p, q_p, sc_desc_p, error) &
                                                            bind(C, name="elpa_generalized_eigenvectors_d")
      type(c_ptr), intent(in), value :: handle, a_p, b_p, ev_p, q_p, sc_desc_p
      integer(kind=c_int), optional, intent(in) :: error

      real(kind=c_double), pointer :: a(:, :), b(:, :), q(:, :), ev(:)
      integer(kind=c_int), pointer             :: sc_desc(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(b_p, b, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])
      call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])
      call c_f_pointer(sc_desc_p, sc_desc, [SC_DESC_LEN])

      call elpa_generalized_eigenvectors_d(self, a, b, ev, q, sc_desc, error)
    end subroutine


    !>  \brief elpa_generalized_eigenvectors_f: class method to solve the eigenvalue problem for float real matrices
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
    !>  blocksize, the number of eigenvectors
    !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param q                                    On output: Eigenvectors of a
    !>                                              Distribution is like in Scalapack.
    !>                                              Must be always dimensioned to the full size (corresponding to (na,na))
    !>                                              even if only a part of the eigenvalues is needed.
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_generalized_eigenvectors_f(self, a, b, ev, q, sc_desc, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit
      use iso_c_binding
      class(elpa_impl_t)  :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_float)  :: a(self%local_nrows, *), b(self%local_nrows, *), q(self%local_nrows, *)
#else
      real(kind=c_float)  :: a(self%local_nrows, self%local_ncols), b(self%local_nrows, self%local_ncols), &
                             q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_float)  :: ev(self%na)
      integer             :: sc_desc(SC_DESC_LEN)

      integer, optional   :: error
      integer             :: error_l
      integer(kind=c_int) :: solver
#ifdef WANT_SINGLE_PRECISION_REAL
      logical             :: success_l

      call self%elpa_transform_generalized_f(a, b, sc_desc, error_l)
      if (present(error)) then
          error = error_l
      else if (error_l .ne. ELPA_OK) then
        write(error_unit,'(a)') "ELPA: Error in transform_generalized() and you did not check for errors!"
      endif

      call self%get("solver",solver)
      if (solver .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_real_1stage_single_impl(self, a, ev, q)

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
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

      call self%elpa_transform_back_generalized_f(b, q, sc_desc, error_l)
      if (present(error)) then
          error = error_l
      else if (error_l .ne. ELPA_OK) then
        write(error_unit,'(a)') "ELPA: Error in transform_back_generalized() and you did not check for errors!"
      endif
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
    end subroutine


    !c> void elpa_generalized_eigenvectors_f(elpa_t handle, float *a, float *ev, float *q, int *error);
    subroutine elpa_generalized_eigenvectors_f_c(handle, a_p, b_p, ev_p, q_p, sc_desc_p, error) &
                                                            bind(C, name="elpa_generalized_eigenvectors_f")
      type(c_ptr), intent(in), value :: handle, a_p, b_p, ev_p, q_p, sc_desc_p
      integer(kind=c_int), optional, intent(in) :: error

      real(kind=c_float), pointer :: a(:, :), b(:, :), q(:, :), ev(:)
      integer(kind=c_int), pointer            :: sc_desc(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(b_p, b, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])
      call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])
      call c_f_pointer(sc_desc_p, sc_desc, [SC_DESC_LEN])

      call elpa_generalized_eigenvectors_f(self, a, b, ev, q, sc_desc, error)
    end subroutine


    !>  \brief elpa_generalized_eigenvectors_dc: class method to solve the eigenvalue problem for double complex matrices
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
    !>  blocksize, the number of eigenvectors
    !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param q                                    On output: Eigenvectors of a
    !>                                              Distribution is like in Scalapack.
    !>                                              Must be always dimensioned to the full size (corresponding to (na,na))
    !>                                              even if only a part of the eigenvalues is needed.
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_generalized_eigenvectors_dc(self, a, b, ev, q, sc_desc, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit
      use iso_c_binding
      class(elpa_impl_t)             :: self

#ifdef USE_ASSUMED_SIZE
      complex(kind=c_double_complex) :: a(self%local_nrows, *), b(self%local_nrows, *), q(self%local_nrows, *)
#else
      complex(kind=c_double_complex) :: a(self%local_nrows, self%local_ncols), b(self%local_nrows, self%local_ncols), &
                                        q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_double)            :: ev(self%na)
      integer                        :: sc_desc(SC_DESC_LEN)

      integer, optional              :: error
      integer                        :: error_l
      integer(kind=c_int)            :: solver
      logical                        :: success_l

      call self%elpa_transform_generalized_dc(a, b, sc_desc, error_l)
      if (present(error)) then
          error = error_l
      else if (error_l .ne. ELPA_OK) then
        write(error_unit,'(a)') "ELPA: Error in transform_generalized() and you did not check for errors!"
      endif

      call self%get("solver", solver)
      if (solver .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_complex_1stage_double_impl(self, a, ev, q)

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
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

      call self%elpa_transform_back_generalized_dc(b, q, sc_desc, error_l)
      if (present(error)) then
          error = error_l
      else if (error_l .ne. ELPA_OK) then
        write(error_unit,'(a)') "ELPA: Error in transform_back_generalized() and you did not check for errors!"
      endif
    end subroutine


    !c> void elpa_generalized_eigenvectors_dc(elpa_t handle, double complex *a, double *ev, double complex *q, int *error);
    subroutine elpa_generalized_eigenvectors_dc_c(handle, a_p, b_p, ev_p, q_p, sc_desc_p, error) &
                                                             bind(C, name="elpa_generalized_eigenvectors_dc")
      type(c_ptr), intent(in), value :: handle, a_p, b_p, ev_p, q_p, sc_desc_p
      integer(kind=c_int), optional, intent(in) :: error

      complex(kind=c_double_complex), pointer :: a(:, :), b(:, :), q(:, :)
      real(kind=c_double), pointer :: ev(:)
      integer(kind=c_int), pointer             :: sc_desc(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(b_p, b, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])
      call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])
      call c_f_pointer(sc_desc_p, sc_desc, [SC_DESC_LEN])

      call elpa_generalized_eigenvectors_dc(self, a, b, ev, q, sc_desc, error)
    end subroutine


    !>  \brief elpa_generalized_eigenvectors_fc: class method to solve the eigenvalue problem for float complex matrices
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cyclic distribution
    !>  blocksize, the number of eigenvectors
    !>  to be computed and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
    !>
    !>  \param q                                    On output: Eigenvectors of a
    !>                                              Distribution is like in Scalapack.
    !>                                              Must be always dimensioned to the full size (corresponding to (na,na))
    !>                                              even if only a part of the eigenvalues is needed.
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_generalized_eigenvectors_fc(self, a, b, ev, q, sc_desc, error)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit

      use iso_c_binding
      class(elpa_impl_t)            :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_float_complex) :: a(self%local_nrows, *), b(self%local_nrows, *), q(self%local_nrows, *)
#else
      complex(kind=c_float_complex) :: a(self%local_nrows, self%local_ncols), b(self%local_nrows, self%local_ncols), &
                                       q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_float)            :: ev(self%na)
      integer                       :: sc_desc(SC_DESC_LEN)

      integer, optional             :: error
      integer                       :: error_l
      integer(kind=c_int)           :: solver
#ifdef WANT_SINGLE_PRECISION_COMPLEX
      logical                       :: success_l

      call self%elpa_transform_generalized_fc(a, b, sc_desc, error_l)
      if (present(error)) then
          error = error_l
      else if (error_l .ne. ELPA_OK) then
        write(error_unit,'(a)') "ELPA: Error in transform_generalized() and you did not check for errors!"
      endif

      call self%get("solver", solver)
      if (solver .eq. ELPA_SOLVER_1STAGE) then
        success_l = elpa_solve_evp_complex_1stage_single_impl(self, a, ev, q)

      else if (solver .eq. ELPA_SOLVER_2STAGE) then
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

      call self%elpa_transform_back_generalized_fc(b, q, sc_desc, error_l)
      if (present(error)) then
          error = error_l
      else if (error_l .ne. ELPA_OK) then
        write(error_unit,'(a)') "ELPA: Error in transform_back_generalized() and you did not check for errors!"
      endif
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
    end subroutine


    !c> void elpa_generalized_eigenvectors_fc(elpa_t handle, float complex *a, float *ev, float complex *q, int *error);
    subroutine elpa_generalized_eigenvectors_fc_c(handle, a_p, b_p, ev_p, q_p, sc_desc_p, error) &
                                                             bind(C, name="elpa_generalized_eigenvectors_fc")
      type(c_ptr), intent(in), value :: handle, a_p, b_p, ev_p, q_p, sc_desc_p
      integer(kind=c_int), optional, intent(in) :: error

      complex(kind=c_float_complex), pointer :: a(:, :), b(:, :), q(:, :)
      real(kind=c_float), pointer :: ev(:)
      integer(kind=c_int), pointer            :: sc_desc(:)
      type(elpa_impl_t), pointer  :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])
      call c_f_pointer(b_p, b, [self%local_nrows, self%local_ncols])
      call c_f_pointer(ev_p, ev, [self%na])
      call c_f_pointer(q_p, q, [self%local_nrows, self%local_ncols])
      call c_f_pointer(sc_desc_p, sc_desc, [SC_DESC_LEN])

      call elpa_generalized_eigenvectors_fc(self, a, b, ev, q, sc_desc, error)
    end subroutine




!********************************************************************************************************
!             HERMITIAN MULTIPLY
!********************************************************************************************************

    !> \brief  elpa_hermitian_multiply_d: class method to perform C : = A**T * B for double real matrices
    !>         where   A is a square matrix (self%na,self%na) which is optionally upper or lower triangular
    !>                 B is a (self%na,ncb) matrix
    !>                 C is a (self%na,ncb) matrix where optionally only the upper or lower
    !>                   triangle may be computed
    !>
    !> the MPI commicators and the block-cyclic distribution block size are already known to the type.
    !> Thus the class method "setup" must be called BEFORE this method is used
    !>
    !> \details
    !>
    !> \param  self                 class(elpa_t), the ELPA object
    !> \param  uplo_a               'U' if A is upper triangular
    !>                              'L' if A is lower triangular
    !>                              anything else if A is a full matrix
    !>                              Please note: This pertains to the original A (as set in the calling program)
    !>                                           whereas the transpose of A is used for calculations
    !>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
    !>                              i.e. it may contain arbitrary numbers
    !> \param uplo_c                'U' if only the upper diagonal part of C is needed
    !>                              'L' if only the upper diagonal part of C is needed
    !>                              anything else if the full matrix C is needed
    !>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
    !>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
    !> \param ncb                   Number of columns  of global matrices B and C
    !> \param a                     matrix a
    !> \param local_nrows           number of rows of local (sub) matrix a, set with class method set("local_nrows",value)
    !> \param local_ncols           number of columns of local (sub) matrix a, set with class method set("local_ncols",value)
    !> \param b                     matrix b
    !> \param nrows_b               number of rows of local (sub) matrix b
    !> \param ncols_b               number of columns of local (sub) matrix b
    !> \param c                     matrix c
    !> \param nrows_c               number of rows of local (sub) matrix c
    !> \param ncols_c               number of columns of local (sub) matrix c
    !> \param error                 optional argument, error code which can be queried with elpa_strerr
    subroutine elpa_hermitian_multiply_d (self, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, &
                                          c, nrows_c, ncols_c, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      class(elpa_impl_t)              :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=c_int), intent(in) :: nrows_b, ncols_b, nrows_c, ncols_c, ncb
#ifdef USE_ASSUMED_SIZE
      real(kind=c_double)             :: a(self%local_nrows,*), b(nrows_b,*), c(nrows_c,*)
#else
      real(kind=c_double)             :: a(self%local_nrows,self%local_ncols), b(nrows_b,ncols_b), c(nrows_c,ncols_c)
#endif
      integer, optional               :: error
      logical                         :: success_l

      success_l = elpa_mult_at_b_real_double_impl(self, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, &
                                                  c, nrows_c, ncols_c)
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

    !c> void elpa_hermitian_multiply_d(elpa_t handle, char uplo_a, char uplo_c, int ncb, double *a, double *b, int nrows_b, int ncols_b, double *c, int nrows_c, int ncols_c, int *error);
    subroutine elpa_hermitian_multiply_d_c(handle, uplo_a, uplo_c, ncb, a_p, b, nrows_b, &
                                           ncols_b, c, nrows_c, ncols_c, error)          &
                                           bind(C, name="elpa_hermitian_multiply_d")
      type(c_ptr), intent(in), value            :: handle, a_p
      character(1,C_CHAR), value                :: uplo_a, uplo_c
      integer(kind=c_int), value                :: ncb, nrows_b, ncols_b, nrows_c, ncols_c
      integer(kind=c_int), optional, intent(in) :: error

      real(kind=c_double), pointer              :: a(:, :)
#ifdef USE_ASSUMED_SIZE
      real(kind=c_double)                       :: b(nrows_b,*), c(nrows_c,*)
#else
      real(kind=c_double)                       :: b(nrows_b,ncols_b), c(nrows_c,ncols_c)
#endif
      type(elpa_impl_t), pointer                :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

      call elpa_hermitian_multiply_d(self, uplo_a, uplo_c, ncb, a, b, nrows_b, &
                                     ncols_b, c, nrows_c, ncols_c, error)
    end subroutine

    !> \brief  elpa_hermitian_multiply_f: class method to perform C : = A**T * B for float real matrices
    !>         where   A is a square matrix (self%na,self%na) which is optionally upper or lower triangular
    !>                 B is a (self%na,ncb) matrix
    !>                 C is a (self%na,ncb) matrix where optionally only the upper or lower
    !>                   triangle may be computed
    !>
    !> the MPI commicators and the block-cyclic distribution block size are already known to the type.
    !> Thus the class method "setup" must be called BEFORE this method is used
    !>
    !> \details
    !>
    !> \param  self                 class(elpa_t), the ELPA object
    !> \param  uplo_a               'U' if A is upper triangular
    !>                              'L' if A is lower triangular
    !>                              anything else if A is a full matrix
    !>                              Please note: This pertains to the original A (as set in the calling program)
    !>                                           whereas the transpose of A is used for calculations
    !>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
    !>                              i.e. it may contain arbitrary numbers
    !> \param uplo_c                'U' if only the upper diagonal part of C is needed
    !>                              'L' if only the upper diagonal part of C is needed
    !>                              anything else if the full matrix C is needed
    !>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
    !>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
    !> \param ncb                   Number of columns  of global matrices B and C
    !> \param a                     matrix a
    !> \param self%local_nrows      number of rows of local (sub) matrix a, set with class method set("local_nrows",value)
    !> \param self%local_ncols      number of columns of local (sub) matrix a, set with class method set("local_ncols",value)
    !> \param b                     matrix b
    !> \param nrows_b               number of rows of local (sub) matrix b
    !> \param ncols_b               number of columns of local (sub) matrix b
    !> \param c                     matrix c
    !> \param nrows_c               number of rows of local (sub) matrix c
    !> \param ncols_c               number of columns of local (sub) matrix c
    !> \param error                 optional argument, returns an error code, which can be queried with elpa_strerr
    subroutine elpa_hermitian_multiply_f (self,uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, &
                                          c, nrows_c, ncols_c, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      class(elpa_impl_t)              :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=c_int), intent(in) :: nrows_b, ncols_b, nrows_c, ncols_c, ncb
#ifdef USE_ASSUMED_SIZE
      real(kind=c_float)              :: a(self%local_nrows,*), b(self%local_nrows,*), c(nrows_c,*)
#else
      real(kind=c_float)              :: a(self%local_nrows,self%local_ncols), b(nrows_b,ncols_b), c(nrows_c,ncols_c)
#endif
      integer, optional               :: error
#ifdef WANT_SINGLE_PRECISION_REAL
      logical                         :: success_l

      success_l = elpa_mult_at_b_real_single_impl(self, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, &
                                                  c, nrows_c, ncols_c)
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

    !c> void elpa_hermitian_multiply_f(elpa_t handle, char uplo_a, char uplo_c, int ncb, float *a, float *b, int nrows_b, int ncols_b, float *c, int nrows_c, int ncols_c, int *error);
    subroutine elpa_hermitian_multiply_f_c(handle, uplo_a, uplo_c, ncb, a_p, b, nrows_b, &
                                           ncols_b, c, nrows_c, ncols_c, error)          &
                                           bind(C, name="elpa_hermitian_multiply_f")
      type(c_ptr), intent(in), value            :: handle, a_p
      character(1,C_CHAR), value                :: uplo_a, uplo_c
      integer(kind=c_int), value                :: ncb, nrows_b, ncols_b, nrows_c, ncols_c
      integer(kind=c_int), optional, intent(in) :: error

      real(kind=c_float), pointer               :: a(:, :)
#ifdef USE_ASSUMED_SIZE
      real(kind=c_float)                        :: b(nrows_b,*), c(nrows_c,*)
#else
      real(kind=c_float)                        :: b(nrows_b,ncols_b), c(nrows_c,ncols_c)
#endif
      type(elpa_impl_t), pointer                :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

      call elpa_hermitian_multiply_f(self, uplo_a, uplo_c, ncb, a, b, nrows_b, &
                                     ncols_b, c, nrows_c, ncols_c, error)
    end subroutine
    !> \brief  elpa_hermitian_multiply_dc: class method to perform C : = A**H * B for double complex matrices
    !>         where   A is a square matrix (self%na,self%na) which is optionally upper or lower triangular
    !>                 B is a (self%na,ncb) matrix
    !>                 C is a (self%na,ncb) matrix where optionally only the upper or lower
    !>                   triangle may be computed
    !>
    !> the MPI commicators and the block-cyclic distribution block size are already known to the type.
    !> Thus the class method "setup" must be called BEFORE this method is used
    !>
    !> \details
    !>
    !> \param  self                 class(elpa_t), the ELPA object
    !> \param  uplo_a               'U' if A is upper triangular
    !>                              'L' if A is lower triangular
    !>                              anything else if A is a full matrix
    !>                              Please note: This pertains to the original A (as set in the calling program)
    !>                                           whereas the transpose of A is used for calculations
    !>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
    !>                              i.e. it may contain arbitrary numbers
    !> \param uplo_c                'U' if only the upper diagonal part of C is needed
    !>                              'L' if only the upper diagonal part of C is needed
    !>                              anything else if the full matrix C is needed
    !>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
    !>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
    !> \param ncb                   Number of columns  of global matrices B and C
    !> \param a                     matrix a
    !> \param self%local_nrows      number of rows of local (sub) matrix a, set with class method set("local_nows",value)
    !> \param self%local_ncols      number of columns of local (sub) matrix a, set with class method set("local_ncols",value)
    !> \param b                     matrix b
    !> \param nrows_b               number of rows of local (sub) matrix b
    !> \param ncols_b               number of columns of local (sub) matrix b
    !> \param c                     matrix c
    !> \param nrows_c               number of rows of local (sub) matrix c
    !> \param ncols_c               number of columns of local (sub) matrix c
    !> \param error                 optional argument, returns an error code, which can be queried with elpa_strerr
    subroutine elpa_hermitian_multiply_dc (self,uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, &
                                          c, nrows_c, ncols_c, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      class(elpa_impl_t)              :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=c_int), intent(in) :: nrows_b, ncols_b, nrows_c, ncols_c, ncb
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_double_complex)  :: a(self%local_nrows,*), b(nrows_b,*), c(nrows_c,*)
#else
      complex(kind=c_double_complex)  :: a(self%local_nrows,self%local_ncols), b(nrows_b,ncols_b), c(nrows_c,ncols_c)
#endif
      integer, optional               :: error
      logical                         :: success_l

      success_l = elpa_mult_ah_b_complex_double_impl(self, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, &
                                                     c, nrows_c, ncols_c)
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


    !c> void elpa_hermitian_multiply_dc(elpa_t handle, char uplo_a, char uplo_c, int ncb, double complex *a, double complex *b, int nrows_b, int ncols_b, double complex *c, int nrows_c, int ncols_c, int *error);
    subroutine elpa_hermitian_multiply_dc_c(handle, uplo_a, uplo_c, ncb, a_p, b, nrows_b, &
                                            ncols_b, c, nrows_c, ncols_c, error)          &
                                            bind(C, name="elpa_hermitian_multiply_dc")
      type(c_ptr), intent(in), value            :: handle, a_p
      character(1,C_CHAR), value                :: uplo_a, uplo_c
      integer(kind=c_int), value                :: ncb, nrows_b, ncols_b, nrows_c, ncols_c
      integer(kind=c_int), optional, intent(in) :: error

      complex(kind=c_double_complex), pointer   :: a(:, :)
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_double_complex)            :: b(nrows_b,*), c(nrows_c,*)
#else
      complex(kind=c_double_complex)            :: b(nrows_b,ncols_b), c(nrows_c,ncols_c)
#endif
      type(elpa_impl_t), pointer                :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

      call elpa_hermitian_multiply_dc(self, uplo_a, uplo_c, ncb, a, b, nrows_b, &
                                     ncols_b, c, nrows_c, ncols_c, error)
    end subroutine

    !> \brief  elpa_hermitian_multiply_fc: class method to perform C : = A**H * B for float complex matrices
    !>         where   A is a square matrix (self%na,self%na) which is optionally upper or lower triangular
    !>                 B is a (self%na,ncb) matrix
    !>                 C is a (self%na,ncb) matrix where optionally only the upper or lower
    !>                   triangle may be computed
    !>
    !> the MPI commicators and the block-cyclic distribution block size are already known to the type.
    !> Thus the class method "setup" must be called BEFORE this method is used
    !>
    !> \details
    !>
    !> \param  self                 class(elpa_t), the ELPA object
    !> \param  uplo_a               'U' if A is upper triangular
    !>                              'L' if A is lower triangular
    !>                              anything else if A is a full matrix
    !>                              Please note: This pertains to the original A (as set in the calling program)
    !>                                           whereas the transpose of A is used for calculations
    !>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
    !>                              i.e. it may contain arbitrary numbers
    !> \param uplo_c                'U' if only the upper diagonal part of C is needed
    !>                              'L' if only the upper diagonal part of C is needed
    !>                              anything else if the full matrix C is needed
    !>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
    !>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
    !> \param ncb                   Number of columns  of global matrices B and C
    !> \param a                     matrix a
    !> \param self%local_nrows      number of rows of local (sub) matrix a, set with class method set("local_nrows",value)
    !> \param self%local_ncols      number of columns of local (sub) matrix a, set with class method set("local_ncols",value)
    !> \param b                     matrix b
    !> \param nrows_b               number of rows of local (sub) matrix b
    !> \param ncols_b               number of columns of local (sub) matrix b
    !> \param c                     matrix c
    !> \param nrows_c               number of rows of local (sub) matrix c
    !> \param ncols_c               number of columns of local (sub) matrix c
    !> \param error                 optional argument, returns an error code, which can be queried with elpa_strerr
    subroutine elpa_hermitian_multiply_fc (self,uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, &
                                          c, nrows_c, ncols_c, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      class(elpa_impl_t)              :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=c_int), intent(in) :: nrows_b, ncols_b, nrows_c, ncols_c, ncb
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_float_complex)   :: a(self%local_nrows,*), b(nrows_b,*), c(nrows_c,*)
#else
      complex(kind=c_float_complex)   :: a(self%local_nrows,self%local_ncols), b(nrows_b,ncols_b), c(nrows_c,ncols_c)
#endif
      integer, optional               :: error
#ifdef WANT_SINGLE_PRECISION_COMPLEX
      logical                         :: success_l

      success_l = elpa_mult_ah_b_complex_single_impl(self, uplo_a, uplo_c, ncb, a, b, nrows_b, ncols_b, &
                                                     c, nrows_c, ncols_c)
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


    !c> void elpa_hermitian_multiply_fc(elpa_t handle, char uplo_a, char uplo_c, int ncb, float complex *a, float complex *b, int nrows_b, int ncols_b, float complex *c, int nrows_c, int ncols_c, int *error);
    subroutine elpa_hermitian_multiply_fc_c(handle, uplo_a, uplo_c, ncb, a_p, b, nrows_b, &
                                            ncols_b, c, nrows_c, ncols_c, error)          &
                                            bind(C, name="elpa_hermitian_multiply_fc")
      type(c_ptr), intent(in), value            :: handle, a_p
      character(1,C_CHAR), value                :: uplo_a, uplo_c
      integer(kind=c_int), value                :: ncb, nrows_b, ncols_b, nrows_c, ncols_c
      integer(kind=c_int), optional, intent(in) :: error

      complex(kind=c_float_complex), pointer    :: a(:, :)
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_float_complex)             :: b(nrows_b,*), c(nrows_c,*)
#else
      complex(kind=c_float_complex)             :: b(nrows_b,ncols_b), c(nrows_c,ncols_c)
#endif
      type(elpa_impl_t), pointer                :: self

      call c_f_pointer(handle, self)
      call c_f_pointer(a_p, a, [self%local_nrows, self%local_ncols])

      call elpa_hermitian_multiply_fc(self, uplo_a, uplo_c, ncb, a, b, nrows_b, &
                                      ncols_b, c, nrows_c, ncols_c, error)
    end subroutine


    !>  \brief elpa_choleksy_d: class method to do a cholesky factorization for a double real matrix
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
    !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
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

    !>  \brief elpa_choleksy_f: class method to do a cholesky factorization for a float real matrix
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
    !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
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
#if WANT_SINGLE_PRECISION_REAL
      logical                         :: success_l

      success_l = elpa_cholesky_real_single_impl (self, a)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in cholesky() and you did not check for errors!"
      endif
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
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

    !>  \brief elpa_choleksy_d: class method to do a cholesky factorization for a double complex matrix
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
    !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
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

    !>  \brief elpa_choleksy_fc: class method to do a cholesky factorization for a float complex matrix
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
    !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_cholesky_fc (self, a, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      class(elpa_impl_t)              :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_float_complex)   :: a(self%local_nrows,*)
#else
      complex(kind=c_float_complex)   :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
#if WANT_SINGLE_PRECISION_COMPLEX
      logical                         :: success_l

      success_l = elpa_cholesky_complex_single_impl (self, a)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in cholesky() and you did not check for errors!"
      endif
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
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

    !>  \brief elpa_invert_trm_d: class method to invert a triangular double real matrix
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
    !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
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

    !>  \brief elpa_invert_trm_f: class method to invert a triangular float real matrix
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
    !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
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
#if WANT_SINGLE_PRECISION_REAL
      logical                         :: success_l

      success_l = elpa_invert_trm_real_single_impl (self, a)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in invert_trm() and you did not check for errors!"
      endif
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
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

    !>  \brief elpa_invert_trm_dc: class method to invert a triangular double complex matrix
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
    !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
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

    !>  \brief elpa_invert_trm_fc: class method to invert a triangular float complex matrix
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
    !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
    !>                                              Distribution is like in Scalapack.
    !>                                              The full matrix must be set (not only one half like in scalapack).
    !>                                              Destroyed on exit (upper and lower half).
    !>
    !>  \param error                                integer, optional: returns an error code, which can be queried with elpa_strerr
    subroutine elpa_invert_trm_fc (self, a, error)
      use iso_c_binding
      use elpa1_auxiliary_impl
      class(elpa_impl_t)              :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_float_complex)   :: a(self%local_nrows,*)
#else
      complex(kind=c_float_complex)   :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: error
#if WANT_SINGLE_PRECISION_COMPLEX
      logical                         :: success_l

      success_l = elpa_invert_trm_complex_single_impl (self, a)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in invert_trm() and you did not check for errors!"
      endif
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
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


    !>  \brief elpa_solve_tridiagonal_d: class method to solve the eigenvalue problem for a double real tridiagonal matrix a
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
    !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param d        array d  on input diagonal elements of tridiagonal matrix, on
    !>                           output the eigenvalues in ascending order
    !>  \param e        array e on input subdiagonal elements of matrix, on exit destroyed
    !>  \param q        matrix  on exit : contains the eigenvectors
    !>  \param error    integer, optional: returns an error code, which can be queried with elpa_strerr
    !> \todo e should have dimension (na - 1)
    subroutine elpa_solve_tridiagonal_d (self, d, e, q, error)
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

      success_l = elpa_solve_tridi_double_impl(self, d, e, q)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in solve_tridiagonal() and you did not check for errors!"
      endif
    end subroutine


    !>  \brief elpa_solve_tridiagonal_f: class method to solve the eigenvalue problem for a float real tridiagonal matrix a
    !>
    !>  The dimensions of the matrix a (locally ditributed and global), the block-cylic-distribution
    !>  block size, and the MPI communicators are already known to the object and MUST be set BEFORE
    !>  with the class method "setup"
    !>
    !>  It is possible to change the behaviour of the method by setting tunable parameters with the
    !>  class method "set"
    !>
    !>  Parameters
    !>
    !>  \param d        array d  on input diagonal elements of tridiagonal matrix, on
    !>                           output the eigenvalues in ascending order
    !>  \param e        array e on input subdiagonal elements of matrix, on exit destroyed
    !>  \param q        matrix  on exit : contains the eigenvectors
    !>  \param error    integer, optional: returns an error code, which can be queried with elpa_strerr
    !> \todo e should have dimension (na - 1)
    subroutine elpa_solve_tridiagonal_f (self, d, e, q, error)
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
#ifdef WANT_SINGLE_PRECISION_REAL
      logical                         :: success_l

      success_l = elpa_solve_tridi_single_impl(self, d, e, q)
      if (present(error)) then
        if (success_l) then
          error = ELPA_OK
        else
          error = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in solve_tridiagonal() and you did not check for errors!"
      endif
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      error = ELPA_ERROR
#endif
    end subroutine


    subroutine elpa_destroy(self)
      use elpa_generated_fortran_interfaces
#ifdef WITH_MPI
      integer :: mpi_comm_rows, mpi_comm_cols, mpierr
#endif
      class(elpa_impl_t) :: self

#ifdef WITH_MPI
      if (self%communicators_owned == 1) then
        call self%get("mpi_comm_rows", mpi_comm_rows)
        call self%get("mpi_comm_cols", mpi_comm_cols)

        call mpi_comm_free(mpi_comm_rows, mpierr)
        call mpi_comm_free(mpi_comm_cols, mpierr)
      endif
#endif

      call timer_free(self%timer)
      call elpa_index_free_c(self%index)

    end subroutine

#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "general/precision_macros.h"
#include "elpa_impl_template.F90"
#undef REALCASE
#undef DOUBLE_PRECISION

#ifdef WANT_SINGLE_PRECISION_REAL
#define REALCASE 1
#define SINGLE_PRECISION 1
#include "general/precision_macros.h"
#include "elpa_impl_template.F90"
#undef REALCASE
#undef SINGLE_PRECISION
#endif /* WANT_SINGLE_PRECISION_REAL */

#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#include "general/precision_macros.h"
#include "elpa_impl_template.F90"
#undef DOUBLE_PRECISION
#undef COMPLEXCASE

#ifdef WANT_SINGLE_PRECISION_COMPLEX
#define COMPLEXCASE 1
#define SINGLE_PRECISION
#include "general/precision_macros.h"
#include "elpa_impl_template.F90"
#undef COMPLEXCASE
#undef SINGLE_PRECISION
#endif /* WANT_SINGLE_PRECISION_COMPLEX */

end module
