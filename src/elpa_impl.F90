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
  use precision
  use elpa2_impl
  use elpa1_impl
  use elpa1_auxiliary_impl
  use elpa_mpi
  use elpa_generated_fortran_interfaces
  use elpa_utilities, only : error_unit

  use elpa_abstract_impl
#ifdef ENABLE_AUTOTUNING
  use elpa_autotune_impl
#endif
  use, intrinsic :: iso_c_binding
  use iso_fortran_env
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

     procedure, public :: elpa_generalized_eigenvalues_d      !< public methods to implement the solve step for generalized 
                                                              !< eigenproblem and real/complex double/single matrices
     procedure, public :: elpa_generalized_eigenvalues_f
     procedure, public :: elpa_generalized_eigenvalues_dc
     procedure, public :: elpa_generalized_eigenvalues_fc

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

     procedure, public :: print_all_parameters => elpa_print_all_parameters
#ifdef ENABLE_AUTOTUNING
     procedure, public :: autotune_setup => elpa_autotune_setup
     procedure, public :: autotune_step => elpa_autotune_step
     procedure, public :: autotune_set_best => elpa_autotune_set_best
     procedure, public :: autotune_print_best => elpa_autotune_print_best
     procedure, public :: autotune_print_state => elpa_autotune_print_state
#endif
     procedure, private :: construct_scalapack_descriptor => elpa_construct_scalapack_descriptor
  end type elpa_impl_t

  !> \brief the implementation of the generic methods
  contains


    !> \brief function to allocate an ELPA object
    !> Parameters
    !> \param   error      integer, optional to get an error code
    !> \result  obj        class(elpa_impl_t) allocated ELPA object
    function elpa_impl_allocate(error) result(obj)
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
    !c> elpa_t elpa_allocate(int *error);
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

#ifdef ENABLE_AUTOTUNING
    !c> /*! \brief C interface for the implementation of the elpa_autotune_deallocate method
    !c> *
    !c> *  \param  elpa_autotune_impl_t  handle of ELPA autotune object to be deallocated
    !c> *  \result void
    !c> */
    !c> void elpa_autotune_deallocate(elpa_autotune_t handle);
    subroutine elpa_autotune_impl_deallocate_c( autotune_handle) bind(C, name="elpa_autotune_deallocate")
      type(c_ptr), value                  :: autotune_handle

      type(elpa_autotune_impl_t), pointer :: self

      call c_f_pointer(autotune_handle, self)
      call self%destroy()
      deallocate(self)
    end subroutine
#endif

    ! we want to ensure, that my_prow(col) and np_rows(cols) values are allways accessible trhough
    ! the elpa object, no matter whether the user provides communicators or communicators are created
    ! by elpa. If the walues are present already, they are checked for consistency with the communicators.
    subroutine set_or_check_missing_comm_params(self)
      implicit none
      class(elpa_impl_t), intent(inout)   :: self
      integer                             :: mpi_comm_parent, mpi_comm_rows, mpi_comm_cols, mpierr, error, &
                                             my_prow, my_pcol, my_id, present_my_prow, present_my_pcol, present_my_id, &
                                             np_rows, np_cols, np_total, present_np_rows, present_np_cols, present_np_total, &
                                             is_process_id_zero
      if (.not. (self%is_set("mpi_comm_rows") == 1 .and. self%is_set("mpi_comm_cols") == 1) ) then
        print *,"MPI row and column communicators not set correctly. Aborting..."
        stop
      endif
      call self%get("mpi_comm_rows", mpi_comm_rows, error)
      call self%get("mpi_comm_cols", mpi_comm_cols, error)

      call mpi_comm_size(mpi_comm_rows, np_rows, mpierr)
      if(self%is_set("num_process_rows") == 1) then
        call self%get("num_process_rows", present_np_rows, error)
        if(np_rows .ne. present_np_rows) then
          print *,"MPI row communicator not set correctly. Aborting..."
          stop
        endif
      else
        call self%set("num_process_rows", np_rows, error)
      endif

      call mpi_comm_size(mpi_comm_cols, np_cols, mpierr)
      if(self%is_set("num_process_cols") == 1) then
        call self%get("num_process_cols", present_np_cols, error)
        if(np_cols .ne. present_np_cols) then
          print *,"MPI column communicator not set correctly. Aborting..."
          stop
        endif
      else
        call self%set("num_process_cols", np_cols, error)
      endif

      call mpi_comm_rank(mpi_comm_rows, my_prow, mpierr)
      if(self%is_set("process_row") == 1) then
        call self%get("process_row", present_my_prow, error)
        if(my_prow .ne. present_my_prow) then
          print *,"MPI row communicator not set correctly. Aborting..."
          stop
        endif
      else
        call self%set("process_row", my_prow, error)
      endif

      call mpi_comm_rank(mpi_comm_cols, my_pcol, mpierr)
      if(self%is_set("process_col") == 1) then
        call self%get("process_col", present_my_pcol, error)
        if(my_pcol .ne. present_my_pcol) then
          print *,"MPI column communicator not set correctly. Aborting..."
          stop
        endif
      else
        call self%set("process_col", my_pcol, error)
      endif


      ! sadly, at the moment, the parent mpi communicator is not required to be set, e.g. in legacy tests
      ! we thus cannot obtain process_id
      ! we can, however, determine the number of prcesses and determine, whether the given process has id 0, 
      ! assuming, that that is the wan with row and column ids == 0
      is_process_id_zero = 0
      if (self%is_set("mpi_comm_parent") == 1) then
        call self%get("mpi_comm_parent", mpi_comm_parent, error)

        call mpi_comm_size(mpi_comm_parent, np_total, mpierr)
        if(self%is_set("num_processes") == 1) then
          call self%get("num_processes", present_np_total, error)
          if(np_total .ne. present_np_total) then
            print *,"MPI parent communicator not set correctly. Aborting..."
            stop
          endif
        else
          call self%set("num_processes", np_total, error)
        endif

        if(np_total .ne. np_rows * np_cols) then
          print *,"MPI parent communicator and row/col communicators do not match. Aborting..."
          stop
        endif

        call mpi_comm_rank(mpi_comm_parent, my_id, mpierr)
        if(self%is_set("process_id") == 1) then
          call self%get("process_id", present_my_id, error)
          if(my_id .ne. present_my_id) then
            print *,"MPI parent communicator not set correctly. Aborting..."
            stop
          endif
        else
          call self%set("process_id", my_id, error)
        endif

        if(my_id == 0) &
          is_process_id_zero = 1
      else
        ! we can set number of processes and whether process id is zero, but not the process id.
        ! we assume, that my_pcol == 0 && my_prow == 0  <==> my_id == 0
        call self%set("num_process", np_rows * np_cols, error)
        if((my_prow == 0) .and. (my_pcol == 0)) &
          is_process_id_zero = 1
      endif
        call self%set("is_process_id_zero", is_process_id_zero, error)

    end subroutine

    !> \brief function to setup an ELPA object and to store the MPI communicators internally
    !> Parameters
    !> \param   self       class(elpa_impl_t), the allocated ELPA object
    !> \result  error      integer, the error code
    function elpa_setup(self) result(error)
      class(elpa_impl_t), intent(inout)   :: self
      integer                             :: error, timings

#ifdef WITH_MPI
      integer                             :: mpi_comm_parent, mpi_comm_rows, mpi_comm_cols, &
                                             mpierr, mpierr2, process_row, process_col, mpi_string_length
      character(len=MPI_MAX_ERROR_STRING) :: mpierr_string
#endif

#ifdef HAVE_DETAILED_TIMINGS
      call self%get("timings",timings, error)
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

        call self%get("mpi_comm_parent", mpi_comm_parent, error)
        call self%get("process_row", process_row, error)
        call self%get("process_col", process_col, error)

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

        call self%set("mpi_comm_rows", mpi_comm_rows,error)
        if (error .ne. ELPA_OK) then
          print *,"Problem setting option. Aborting..."
          stop
        endif
        call self%set("mpi_comm_cols", mpi_comm_cols,error)
        if (error .ne. ELPA_OK) then
          print *,"Problem setting option. Aborting..."
          stop
        endif

        call set_or_check_missing_comm_params(self)

        ! remember that we created those communicators and we need to free them later
        self%communicators_owned = 1

        error = ELPA_OK
        return
      endif

      ! Externally supplied communicators
      if (self%is_set("mpi_comm_rows") == 1 .and. self%is_set("mpi_comm_cols") == 1) then
        call set_or_check_missing_comm_params(self)
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

    function elpa_construct_scalapack_descriptor(self, sc_desc, rectangular_for_ev) result(error)
      class(elpa_impl_t), intent(inout)   :: self
      logical, intent(in)                 :: rectangular_for_ev
      integer                             :: error, blacs_ctx
      integer, intent(out)                :: sc_desc(SC_DESC_LEN)

#ifdef WITH_MPI
      if (self%is_set("blacs_context") == 0) then
        print *,"BLACS context has not been set beforehand. Aborting..."
        stop
      endif
      call self%get("blacs_context", blacs_ctx, error)

      sc_desc(1) = 1
      sc_desc(2) = blacs_ctx
      sc_desc(3) = self%na
      if(rectangular_for_ev) then
        sc_desc(4) = self%nev
      else
        sc_desc(4) = self%na
      endif
      sc_desc(5) = self%nblk
      sc_desc(6) = self%nblk
      sc_desc(7) = 0
      sc_desc(8) = 0
      sc_desc(9) = self%local_nrows
#else
      sc_desc = 0
#endif
      error = ELPA_OK
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
      type(c_ptr), intent(in), value                :: handle
      type(elpa_impl_t), pointer                    :: self
      type(c_ptr), intent(in), value                :: name_p
      character(len=elpa_strlen_c(name_p)), pointer :: name
      integer(kind=c_int), intent(in), value        :: value

#ifdef USE_FORTRAN2008
      integer(kind=c_int) , intent(in), optional    :: error
#else
      integer(kind=c_int) , intent(in)              :: error
#endif

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
      type(c_ptr), intent(in), value                :: handle
      type(elpa_impl_t), pointer                    :: self
      type(c_ptr), intent(in), value                :: name_p
      character(len=elpa_strlen_c(name_p)), pointer :: name
      integer(kind=c_int)                           :: value
#ifdef ISE_FORTRAN2008
      integer(kind=c_int), intent(inout), optional  :: error
#else
      integer(kind=c_int), intent(inout)            :: error
#endif
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
      class(elpa_impl_t)       :: self
      character(*), intent(in) :: name
      integer(kind=c_int), intent(in) :: value
      integer                  :: error

      error = elpa_index_int_is_valid_c(self%index, name // c_null_char, value)
    end function


    !> \brief function to convert a value to an human readable string
    !> Parameters
    !> \param   self        class(elpa_impl_t) the allocated ELPA object
    !> \param   option_name string: the name of the options, whose value should be converted
    !> \param   error       integer: errpr code
    !> \result  string      string: the humanreadable string   
    function elpa_value_to_string(self, option_name, error) result(string)
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
      type(c_ptr), intent(in), value                :: handle
      type(elpa_impl_t), pointer                    :: self
      type(c_ptr), intent(in), value                :: name_p
      character(len=elpa_strlen_c(name_p)), pointer :: name
      real(kind=c_double), intent(in), value        :: value
#ifdef USE_FORTRAN2008
      integer(kind=c_int), intent(in), optional     :: error
#else
      integer(kind=c_int), intent(in)               :: error
#endif
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
      type(c_ptr), intent(in), value                :: handle
      type(elpa_impl_t), pointer                    :: self
      type(c_ptr), intent(in), value                :: name_p
      character(len=elpa_strlen_c(name_p)), pointer :: name
      real(kind=c_double)                           :: value
#ifdef USE_FORTRAN2008
      integer(kind=c_int), intent(inout), optional  :: error
#else
      integer(kind=c_int), intent(inout)            :: error
#endif
      call c_f_pointer(handle, self)
      call c_f_pointer(name_p, name)
      call elpa_get_double(self, name, value, error)
    end subroutine
 

    !> \brief function to associate a pointer with an integer value
    !> Parameters
    !> \param   self        class(elpa_impl_t) the allocated ELPA object
    !> \param   name        string: the name of the entry
    !> \result  value       integer, pointer: the value for the entry
    function elpa_associate_int(self, name) result(value)
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


    !> \brief function to querry the timing information at a certain level
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    !> \param   name1 .. name6  string: the string identifier for the timer region.
    !>                                  at the moment 6 nested levels can be queried
    !> \result  s               double: the timer metric for the region. Might be seconds,
    !>                                  or any other supported metric
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


    !> \brief function to print the timing tree below at a certain level
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    !> \param   name1 .. name6  string: the string identifier for the timer region.
    !>                                  at the moment 4 nested levels can be specified
    subroutine elpa_print_times(self, name1, name2, name3, name4)
      class(elpa_impl_t), intent(in) :: self
      character(len=*), intent(in), optional :: name1, name2, name3, name4
#ifdef HAVE_DETAILED_TIMINGS
      call self%timer%print(name1, name2, name3, name4)
#endif
    end subroutine


    !> \brief function to start the timing of a code region
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    !> \param   name            string: a chosen identifier name for the code region
    subroutine elpa_timer_start(self, name)
      class(elpa_impl_t), intent(inout) :: self
      character(len=*), intent(in) :: name
#ifdef HAVE_DETAILED_TIMINGS
      call self%timer%start(name)
#endif
    end subroutine


    !> \brief function to stop the timing of a code region
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    !> \param   name            string: identifier name for the code region to stop
    subroutine elpa_timer_stop(self, name)
      class(elpa_impl_t), intent(inout) :: self
      character(len=*), intent(in) :: name
#ifdef HAVE_DETAILED_TIMINGS
      call self%timer%stop(name)
#endif
    end subroutine


    !> \brief function to destroy an elpa object
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    subroutine elpa_destroy(self)
#ifdef WITH_MPI
      integer :: mpi_comm_rows, mpi_comm_cols, mpierr, error
#endif
      class(elpa_impl_t) :: self

#ifdef WITH_MPI
      if (self%communicators_owned == 1) then
        call self%get("mpi_comm_rows", mpi_comm_rows,error)
        if (error .ne. ELPA_OK) then
           print *,"Problem getting option. Aborting..."
           stop
        endif
        call self%get("mpi_comm_cols", mpi_comm_cols,error)
        if (error .ne. ELPA_OK) then
           print *,"Problem getting option. Aborting..."
           stop
        endif

        call mpi_comm_free(mpi_comm_rows, mpierr)
        call mpi_comm_free(mpi_comm_cols, mpierr)
      endif
#endif

      call timer_free(self%timer)
      call timer_free(self%autotune_timer)
      call elpa_index_free_c(self%index)

    end subroutine

#define REALCASE 1
#define DOUBLE_PRECISION 1
#define INCLUDE_ROUTINES 1
#include "general/precision_macros.h"
#include "elpa_impl_math_template.F90"
#undef REALCASE
#undef DOUBLE_PRECISION
#undef INCLUDE_ROUTINES

#ifdef WANT_SINGLE_PRECISION_REAL
#define INCLUDE_ROUTINES 1
#endif
#define REALCASE 1
#define SINGLE_PRECISION 1
#include "general/precision_macros.h"
#include "elpa_impl_math_template.F90"
#undef REALCASE
#undef SINGLE_PRECISION
#undef INCLUDE_ROUTINES

#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#define INCLUDE_ROUTINES 1
#include "general/precision_macros.h"
#include "elpa_impl_math_template.F90"
#undef DOUBLE_PRECISION
#undef COMPLEXCASE
#undef INCLUDE_ROUTINES

#ifdef WANT_SINGLE_PRECISION_COMPLEX
#define INCLUDE_ROUTINES 1
#endif
#define COMPLEXCASE 1
#define SINGLE_PRECISION
#include "general/precision_macros.h"
#include "elpa_impl_math_template.F90"
#undef COMPLEXCASE
#undef SINGLE_PRECISION
#undef INCLUDE_ROUTINES

#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "general/precision_macros.h"
#include "elpa_impl_generalized_transform_template.F90"
#undef REALCASE
#undef DOUBLE_PRECISION

#ifdef WANT_SINGLE_PRECISION_REAL
#define REALCASE 1
#define SINGLE_PRECISION 1
#include "general/precision_macros.h"
#include "elpa_impl_generalized_transform_template.F90"
#undef REALCASE
#undef SINGLE_PRECISION
#endif

#define COMPLEXCASE 1

#define DOUBLE_PRECISION 1
#include "general/precision_macros.h"
#include "elpa_impl_generalized_transform_template.F90"
#undef DOUBLE_PRECISION
#undef COMPLEXCASE

#ifdef WANT_SINGLE_PRECISION_COMPLEX
#define COMPLEXCASE 1
#define SINGLE_PRECISION
#include "general/precision_macros.h"
#include "elpa_impl_generalized_transform_template.F90"
#undef COMPLEXCASE
#undef SINGLE_PRECISION
#endif

!    function use_cannons_algorithm(self) result(use_cannon, do_print)
!      class(elpa_impl_t), intent(inout), target :: self
!      logical                                   :: use_cannon
!      logical, intent(in)                       :: do_print
!    end function
!
#ifdef ENABLE_AUTOTUNING
    !> \brief function to setup the ELPA autotuning and create the autotune object
    !> Parameters
    !> \param   self            the allocated ELPA object
    !> \param   level           integer: the "thoroughness" of the planed autotuning
    !> \param   domain          integer: the domain (real/complex) which should be tuned
    !> \result  tune_state      the created autotuning object
    function elpa_autotune_setup(self, level, domain, error) result(tune_state)
      class(elpa_impl_t), intent(inout), target :: self
      integer, intent(in)                       :: level, domain
      type(elpa_autotune_impl_t), pointer       :: ts_impl
      class(elpa_autotune_t), pointer           :: tune_state
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional             :: error
#else
      integer(kind=c_int)                       :: error
#endif

#ifdef USE_FORTRAN2008
      if (present(error)) then
        error = ELPA_OK
      endif
#else
      error = ELPA_OK
#endif
      if (elpa_get_api_version() < EARLIEST_AUTOTUNE_VERSION) then
        write(error_unit, "(a,i0,a)") "ELPA: Error API version: Autotuning does not support ", elpa_get_api_version()
#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = ELPA_ERROR
        endif
#else
        error = ELPA_ERROR
#endif
        return
      endif

      allocate(ts_impl)
      ts_impl%parent => self
      ts_impl%level = level
      ts_impl%domain = domain

      ts_impl%current = -1
      ts_impl%min_loc = -1
      ts_impl%cardinality = elpa_index_autotune_cardinality_c(self%index, level, domain)

      tune_state => ts_impl

      call self%autotune_timer%enable()
    end function



    !c> /*! \brief C interface for the implementation of the elpa_autotune_setup method
    !c> *
    !c> *  \param  elpa_t           handle: of the ELPA object which should be tuned
    !c> *  \param  int              level:  "thoroughness" of autotuning
    !c> *  \param  int              domain: real/complex autotuning
    !c> *  \result elpa_autotune_t  handle:  on the autotune object
    !c> */
    !c> elpa_autotune_t elpa_autotune_setup(elpa_t handle, int level, int domain, int *error);
    function elpa_autotune_setup_c(handle ,level, domain, error) result(ptr) bind(C, name="elpa_autotune_setup")
      type(c_ptr), intent(in), value         :: handle
      type(elpa_impl_t), pointer             :: self
      class(elpa_autotune_t), pointer        :: tune_state
      type(elpa_autotune_impl_t), pointer    :: obj        
      integer(kind=c_int), intent(in), value :: level
      integer(kind=c_int), intent(in), value :: domain
      type(c_ptr)                            :: ptr
#ifdef USE_FORTRAN2008
      integer(kind=c_int) , intent(in), optional    :: error
#else
      integer(kind=c_int) , intent(in)              :: error
#endif

      call c_f_pointer(handle, self)

      tune_state => self%autotune_setup(level, domain, error)
      select type(tune_state)
        type is (elpa_autotune_impl_t)
          obj => tune_state
        class default
          print *, "This should not happen"
          stop
      end select                
      ptr = c_loc(obj)

    end function


    !> \brief function to do an autotunig step
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    !> \param   tune_state      class(elpa_autotune_t): the autotuning object
    !> \result  unfinished      logical: describes the state of the autotuning (completed/uncompleted)
    function elpa_autotune_step(self, tune_state) result(unfinished)
      implicit none
      class(elpa_impl_t), intent(inout) :: self
      class(elpa_autotune_t), intent(inout), target :: tune_state
      type(elpa_autotune_impl_t), pointer :: ts_impl
      logical :: unfinished
      integer :: i
      real(kind=C_DOUBLE) :: time_spent

      select type(tune_state)
        type is (elpa_autotune_impl_t)
          ts_impl => tune_state
        class default
          print *, "This should not happen"
      end select

      unfinished = .false.

      if (ts_impl%current >= 0) then
#ifdef HAVE_DETAILED_TIMINGS
        time_spent = self%autotune_timer%get("accumulator")
#else
        print *, "Cannot do autotuning without detailed timings"
#endif
        if (ts_impl%min_loc == -1 .or. (time_spent < ts_impl%min_val)) then
          ts_impl%min_val = time_spent
          ts_impl%min_loc = ts_impl%current
        end if
        call self%autotune_timer%free()
      endif

      do while (ts_impl%current < ts_impl%cardinality - 1)
        ts_impl%current = ts_impl%current + 1
        if (elpa_index_set_autotune_parameters_c(self%index, ts_impl%level, ts_impl%domain, ts_impl%current) == 1) then
          unfinished = .true.
          return
        end if
      end do

    end function



    !c> /*! \brief C interface for the implementation of the elpa_autotune_step method
    !c> *
    !c> *  \param  elpa_t           handle: of the ELPA object which should be tuned
    !c> *  \param  elpa_autotune_t  autotune_handle: the autotuning object
    !c> *  \result int              unfinished:  describes whether autotuning finished (0) or not (1)
    !c> */
    !c> int elpa_autotune_step(elpa_t handle, elpa_autotune_t autotune_handle);
    function elpa_autotune_step_c(handle, autotune_handle) result(unfinished) bind(C, name="elpa_autotune_step")
      type(c_ptr), intent(in), value       :: handle
      type(c_ptr), intent(in), value       :: autotune_handle
      type(elpa_impl_t), pointer           :: self
      type(elpa_autotune_impl_t), pointer  :: tune_state
      logical                              :: unfinished_f
      integer(kind=c_int)                  :: unfinished

      call c_f_pointer(handle, self)
      call c_f_pointer(autotune_handle, tune_state)

      unfinished_f = self%autotune_step(tune_state)
      if (unfinished_f) then
        unfinished = 1
      else
        unfinished = 0
      endif

    end function



    !> \brief function to set the up-to-know best options of the autotuning
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    !> \param   tune_state      class(elpa_autotune_t): the autotuning object
    subroutine elpa_autotune_set_best(self, tune_state)
      implicit none
      class(elpa_impl_t), intent(inout) :: self
      class(elpa_autotune_t), intent(in), target :: tune_state
      type(elpa_autotune_impl_t), pointer :: ts_impl

      select type(tune_state)
        type is (elpa_autotune_impl_t)
          ts_impl => tune_state
        class default
          print *, "This should not happen"
      end select

      if (elpa_index_set_autotune_parameters_c(self%index, ts_impl%level, ts_impl%domain, ts_impl%min_loc) /= 1) then
        stop "This should not happen (in elpa_autotune_set_best())"
      endif
    end subroutine



    !> \brief function to print the up-to-know best options of the autotuning
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    !> \param   tune_state      class(elpa_autotune_t): the autotuning object
    subroutine elpa_autotune_print_best(self, tune_state)
      implicit none
      class(elpa_impl_t), intent(inout) :: self
      class(elpa_autotune_t), intent(in), target :: tune_state
      type(elpa_autotune_impl_t), pointer :: ts_impl

      select type(tune_state)
        type is (elpa_autotune_impl_t)
          ts_impl => tune_state
        class default
          print *, "This should not happen"
      end select

      print *, "The following settings were found to be best:"
      print *, "Best, i = ", ts_impl%min_loc, "best time = ", ts_impl%min_val
      flush(output_unit)
      if (elpa_index_print_autotune_parameters_c(self%index, ts_impl%level, ts_impl%domain) /= 1) then
        stop "This should not happen (in elpa_autotune_print_best())"
      endif
    end subroutine


    !> \brief function to print all the parameters, that have been set
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    subroutine elpa_print_all_parameters(self)
      implicit none
      class(elpa_impl_t), intent(inout) :: self

      !print *, "The following parameters have been set"
      if (elpa_index_print_all_parameters_c(self%index) /= 1) then
        stop "This should not happen (in elpa_print_all_parameters())"
      endif
    end subroutine


    !> \brief function to print the state of the autotuning
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    !> \param   tune_state      class(elpa_autotune_t): the autotuning object
    subroutine elpa_autotune_print_state(self, tune_state)
      implicit none
      class(elpa_impl_t), intent(inout) :: self
      class(elpa_autotune_t), intent(in), target :: tune_state
      type(elpa_autotune_impl_t), pointer :: ts_impl

      select type(tune_state)
        type is (elpa_autotune_impl_t)
          ts_impl => tune_state
        class default
          print *, "This should not happen"
      end select

      !print *, "The following settings were found to be best:"
      if (elpa_index_print_autotune_state_c(self%index, ts_impl%level, ts_impl%domain, ts_impl%min_loc, &
                  ts_impl%min_val, ts_impl%current, ts_impl%cardinality) /= 1) then
        stop "This should not happen (in elpa_autotune_print_state())"
      endif
    end subroutine


    !c> /*! \brief C interface for the implementation of the elpa_autotune_set_best method
    !c> *
    !c> *  \param  elpa_t           handle: of the ELPA object which should be tuned
    !c> *  \param  elpa_autotune_t  autotune_handle: the autotuning object
    !c> *  \result none 
    !c> */
    !c> void elpa_autotune_set_best(elpa_t handle, elpa_autotune_t autotune_handle);
    subroutine elpa_autotune_set_best_c(handle, autotune_handle) bind(C, name="elpa_autotune_set_best")
      type(c_ptr), intent(in), value       :: handle
      type(c_ptr), intent(in), value       :: autotune_handle
      type(elpa_impl_t), pointer           :: self
      type(elpa_autotune_impl_t), pointer  :: tune_state

      call c_f_pointer(handle, self)
      call c_f_pointer(autotune_handle, tune_state)

      call self%autotune_set_best(tune_state)

    end subroutine



    !c> /*! \brief C interface for the implementation of the elpa_autotune_print_best method
    !c> *
    !c> *  \param  elpa_t           handle: of the ELPA object which should be tuned
    !c> *  \param  elpa_autotune_t  autotune_handle: the autotuning object
    !c> *  \result none 
    !c> */
    !c> void elpa_autotune_print_best(elpa_t handle, elpa_autotune_t autotune_handle);
    subroutine elpa_autotune_print_best_c(handle, autotune_handle) bind(C, name="elpa_autotune_print_best")
      type(c_ptr), intent(in), value       :: handle
      type(c_ptr), intent(in), value       :: autotune_handle
      type(elpa_impl_t), pointer           :: self
      type(elpa_autotune_impl_t), pointer  :: tune_state

      call c_f_pointer(handle, self)
      call c_f_pointer(autotune_handle, tune_state)

      call self%autotune_print_best(tune_state)

    end subroutine
#endif


end module
