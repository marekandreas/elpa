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
#ifdef HAVE_LIKWID
  use likwid
#endif

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

   !This object has been created through the legacy api.
   integer :: from_legacy_api

   !> \brief methods available with the elpa_impl_t type
   contains
     !> \brief the puplic methods
     ! con-/destructor
     procedure, public :: setup => elpa_setup                   !< a setup method: implemented in elpa_setup
     procedure, public :: destroy => elpa_destroy               !< a destroy method: implemented in elpa_destroy

     ! KV store
     procedure, public :: is_set => elpa_is_set             !< a method to check whether a key/value pair has been set : implemented
                                                            !< in elpa_is_set
     procedure, public :: can_set => elpa_can_set           !< a method to check whether a key/value pair can be set : implemented
                                                            !< in elpa_can_set

     ! call before setup if created from the legacy api
     ! remove this function completely after the legacy api is dropped
     procedure, public :: creating_from_legacy_api => elpa_creating_from_legacy_api

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

     procedure, public :: elpa_skew_eigenvectors_d             !< public methods to implement the solve step for real skew-symmetric
                                                               !< double/single matrices
     procedure, public :: elpa_skew_eigenvectors_f

     procedure, public :: elpa_skew_eigenvalues_d              !< public methods to implement the solve step for real skew-symmetric
                                                               !< double/single matrices; only the eigenvalues are computed
     procedure, public :: elpa_skew_eigenvalues_f


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

     procedure, public :: elpa_hermitian_multiply_d      !< public methods to implement a "hermitian" multiplication of matrices a and b
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

     procedure, public :: print_settings => elpa_print_settings
     procedure, public :: store_settings => elpa_store_settings
     procedure, public :: load_settings => elpa_load_settings
#ifdef ENABLE_AUTOTUNING
     procedure, public :: autotune_setup => elpa_autotune_setup
     procedure, public :: autotune_step => elpa_autotune_step
     procedure, public :: autotune_set_best => elpa_autotune_set_best
     procedure, public :: autotune_print_best => elpa_autotune_print_best
     procedure, public :: autotune_print_state => elpa_autotune_print_state
     procedure, public :: autotune_save_state => elpa_autotune_save_state
     procedure, public :: autotune_load_state => elpa_autotune_load_state
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
      type(elpa_impl_t), pointer     :: obj
#ifdef USE_FORTRAN2008
      integer, optional, intent(out) :: error
#else
      integer, intent(out)           :: error
#endif
      integer                        :: error2, output_build_config

      allocate(obj, stat=error2)
      if (error2 .ne. 0) then
        write(error_unit, *) "elpa_allocate(): could not allocate object"
      endif

      obj%from_legacy_api = 0

      ! check whether init has ever been called
      if ( elpa_initialized() .ne. ELPA_OK) then
        write(error_unit, *) "elpa_allocate(): you must call elpa_init() once before creating instances of ELPA"
#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = ELPA_ERROR_API_VERSION
        endif
#else
        error = ELPA_ERROR_API_VERSION
#endif
        return
      endif

      obj%index = elpa_index_instance_c()

      ! Associate some important integer pointers for convenience
      obj%na => obj%associate_int("na")
      obj%nev => obj%associate_int("nev")
      obj%local_nrows => obj%associate_int("local_nrows")
      obj%local_ncols => obj%associate_int("local_ncols")
      obj%nblk => obj%associate_int("nblk")

#ifdef USE_FORTRAN2008
      if (present(error)) then
        error = ELPA_OK
      endif
#else
      error = ELPA_OK
#endif
    end function

#ifdef OPTIONAL_C_ERROR_ARGUMENT
    !c_o> #ifdef OPTIONAL_C_ERROR_ARGUMENT
    !c_o> #define elpa_allocate(...) CONC(elpa_allocate, NARGS(__VA_ARGS__))(__VA_ARGS__)
    !c_o> #endif
#endif
    !c> /*! \brief C interface for the implementation of the elpa_allocate method
    !c> *
    !c> *  \param  none
    !c> *  \result elpa_t handle
    !c> */
#ifdef OPTIONAL_C_ERROR_ARGUMENT
    !c_o> #ifdef OPTIONAL_C_ERROR_ARGUMENT
    !c_o> elpa_t elpa_allocate2(int *error);
    !c_o> elpa_t elpa_allocate1();
    !c_o> #endif
    function elpa_impl_allocate_c1() result(ptr) bind(C, name="elpa_allocate1")
      type(c_ptr)                :: ptr
      type(elpa_impl_t), pointer :: obj

      obj => elpa_impl_allocate()
      ptr = c_loc(obj)
    end function

    function elpa_impl_allocate_c2(error) result(ptr) bind(C, name="elpa_allocate2")
      integer(kind=c_int)        :: error
      type(c_ptr)                :: ptr
      type(elpa_impl_t), pointer :: obj

      obj => elpa_impl_allocate(error)
      ptr = c_loc(obj)
    end function
#else
    !c_no> #ifndef OPTIONAL_C_ERROR_ARGUMENT
    !c_no> elpa_t elpa_allocate(int *error);
    !c_no> #endif
    function elpa_impl_allocate_c(error) result(ptr) bind(C, name="elpa_allocate")
      integer(kind=c_int)        :: error
      type(c_ptr)                :: ptr
      type(elpa_impl_t), pointer :: obj

      obj => elpa_impl_allocate(error)
      ptr = c_loc(obj)
    end function
#endif

#ifdef OPTIONAL_C_ERROR_ARGUMENT
    !c_o> #ifdef OPTIONAL_C_ERROR_ARGUMENT
    !c_o> #define NARGS(...) NARGS_(__VA_ARGS__, 5, 4, 3, 2, 1, 0)
    !c_o> #define NARGS_(_5, _4, _3, _2, _1, N, ...) N
    !c_o> #define CONC(A, B) CONC_(A, B)
    !c_o> #define CONC_(A, B) A##B
    !c_o> #define elpa_deallocate(...) CONC(elpa_deallocate, NARGS(__VA_ARGS__))(__VA_ARGS__)
    !c_o> #endif
#endif
    !c> /*! \brief C interface for the implementation of the elpa_deallocate method
    !c> *
    !c> *  \param  elpa_t  handle of ELPA object to be deallocated
    !c> *  \param  int*    error code
    !c> *  \result void
    !c> */
#ifdef OPTIONAL_C_ERROR_ARGUMENT
    !c_o> #ifdef OPTIONAL_C_ERROR_ARGUMENT
    !c_o> void elpa_deallocate2(elpa_t handle, int *error);
    !c_o> void elpa_deallocate1(elpa_t handle);
    !c_o> #endif
    subroutine elpa_impl_deallocate_c2(handle, error) bind(C, name="elpa_deallocate2")
      type(c_ptr), value         :: handle
      type(elpa_impl_t), pointer :: self
      integer(kind=c_int)        :: error

      call c_f_pointer(handle, self)
      call self%destroy(error)
      deallocate(self)
    end subroutine

    subroutine elpa_impl_deallocate_c1(handle) bind(C, name="elpa_deallocate1")
      type(c_ptr), value         :: handle
      type(elpa_impl_t), pointer :: self

      call c_f_pointer(handle, self)
      call self%destroy()
      deallocate(self)
    end subroutine
#else
    !c_no> #ifndef OPTIONAL_C_ERROR_ARGUMENT
    !c_no> void elpa_deallocate(elpa_t handle, int *error);
    !c_no> #endif
    subroutine elpa_impl_deallocate_c(handle, error) bind(C, name="elpa_deallocate")
      type(c_ptr), value         :: handle
      type(elpa_impl_t), pointer :: self
      integer(kind=c_int)        :: error

      call c_f_pointer(handle, self)
      call self%destroy(error)
      deallocate(self)
    end subroutine

#endif

    !> \brief function to load all the parameters, which have been saved to a file
    !> Parameters
    !> \param   self        class(elpa_impl_t) the allocated ELPA object
    !> \param   file_name   string, the name of the file from which to load the parameters
    !> \param   error       integer, optional
    subroutine elpa_load_settings(self, file_name, error)
      implicit none
      class(elpa_impl_t), intent(inout) :: self
      character(*), intent(in)          :: file_name
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(out)    :: error
#else
      integer(kind=c_int), intent(out)              :: error
#endif

#ifdef USE_FORTRAN2008
      if (present(error)) then
        error = ELPA_OK
      endif
#else
      error = ELPA_OK
#endif
      if (elpa_index_load_settings_c(self%index, file_name // c_null_char) /= 1) then
        write(error_unit, *) "This should not happen (in elpa_load_settings())"

#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = ELPA_ERROR_CANNOT_OPEN_FILE
        endif
#else
        error = ELPA_ERROR_CANNOT_OPEN_FILE
#endif
      endif
    end subroutine

    !c> /*! \brief C interface for the implementation of the elpa_load_settings method
    !c> *
    !c> *  \param elpa_t handle
    !c> *  \param  char* filename
    !c> */
    !c> void elpa_load_settings(elpa_t handle, const char *filename, int *error);
    subroutine elpa_load_settings_c(handle, filename_p, error) bind(C, name="elpa_load_settings")
      type(c_ptr), value         :: handle
      type(elpa_impl_t), pointer :: self

      integer(kind=c_int)        :: error
      type(c_ptr), intent(in), value :: filename_p
      character(len=elpa_strlen_c(filename_p)), pointer :: filename

      call c_f_pointer(handle, self)
      call c_f_pointer(filename_p, filename)
      call elpa_load_settings(self, filename, error)

    end subroutine

    !> \brief function to print all the parameters, that have been set
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    !> \param   error           optional, integer
    subroutine elpa_print_settings(self, error)
      implicit none
      class(elpa_impl_t), intent(inout) :: self
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(out)    :: error
#else
      integer(kind=c_int), intent(out)              :: error
#endif

#ifdef USE_FORTRAN2008
      if (present(error)) then
        error = ELPA_OK
      endif
#else
      error = ELPA_OK
#endif
      if (elpa_index_print_settings_c(self%index, c_null_char) /= 1) then
        write(error_unit, *) "This should not happen (in elpa_print_settings())"

#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = ELPA_ERROR_CRITICAL
        endif
#else
        error = ELPA_ERROR_CRITICAL
#endif
      endif
    end subroutine

    !c> /*! \brief C interface for the implementation of the elpa_print_settings method
    !c> *
    !c> *  \param elpa_t handle
    !c> *  \param  char* filename
    !c> */
    !c> void elpa_print_settings(elpa_t handle, int *error);
    subroutine elpa_print_settings_c(handle, error) bind(C, name="elpa_print_settings")
      type(c_ptr), value         :: handle
      type(elpa_impl_t), pointer :: self
 
      integer(kind=c_int)        :: error

      call c_f_pointer(handle, self)
      call elpa_print_settings(self, error)

    end subroutine


    !> \brief function to save all the parameters, that have been set
    !> Parameters
    !> \param   self        class(elpa_impl_t) the allocated ELPA object
    !> \param   file_name   string, the name of the file where to save the parameters
    !> \param   error       integer, optional
    subroutine elpa_store_settings(self, file_name, error)
      implicit none
      class(elpa_impl_t), intent(inout) :: self
      character(*), intent(in)          :: file_name
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(out)    :: error
#else
      integer(kind=c_int), intent(out)              :: error
#endif

#ifdef USE_FORTRAN2008
      if (present(error)) then
        error = ELPA_OK
      endif
#else
      error = ELPA_OK
#endif
      if (elpa_index_print_settings_c(self%index, file_name // c_null_char) /= 1) then
        write(error_unit, *) "This should not happen (in elpa_store_settings())"

#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = ELPA_ERROR_CANNOT_OPEN_FILE
        endif
#else
        error = ELPA_ERROR_CANNOT_OPEN_FILE
#endif
      endif
    end subroutine


    !c> /*! \brief C interface for the implementation of the elpa_store_settings method
    !c> *
    !c> *  \param elpa_t handle
    !c> *  \param  char* filename
    !c> */
    !c> void elpa_store_settings(elpa_t handle, const char *filename, int *error);
    subroutine elpa_store_settings_c(handle, filename_p, error) bind(C, name="elpa_store_settings")
      type(c_ptr), value         :: handle
      type(elpa_impl_t), pointer :: self
      type(c_ptr), intent(in), value :: filename_p
      character(len=elpa_strlen_c(filename_p)), pointer :: filename
      integer(kind=c_int)        :: error

      call c_f_pointer(handle, self)
      call c_f_pointer(filename_p, filename)
      call elpa_store_settings(self, filename, error)

    end subroutine


#ifdef ENABLE_AUTOTUNING
#ifdef OPTIONAL_C_ERROR_ARGUMENT
    !c_o> #ifdef OPTIONAL_C_ERROR_ARGUMENT
    !c_o> #define elpa_autotune_deallocate(...) CONC(elpa_autotune_deallocate, NARGS(__VA_ARGS__))(__VA_ARGS__)
    !c_o> #endif
#endif
    !c> /*! \brief C interface for the implementation of the elpa_autotune_deallocate method
    !c> *
    !c> *  \param  elpa_autotune_impl_t  handle of ELPA autotune object to be deallocated
    !c> *  \result void
    !c> */
#ifdef OPTIONAL_C_ERROR_ARGUMENT
    !c_o> #ifdef OPTIONAL_C_ERROR_ARGUMENT
    !c_o> void elpa_autotune_deallocate2(elpa_autotune_t handle, int *error);
    !c_o> void elpa_autotune_deallocate1(elpa_autotune_t handle);
    !c_o> #endif
    subroutine elpa_autotune_impl_deallocate_c1( autotune_handle) bind(C, name="elpa_autotune_deallocate1")
      type(c_ptr), value                  :: autotune_handle

      type(elpa_autotune_impl_t), pointer :: self
      integer(kind=c_int)                 :: error

      call c_f_pointer(autotune_handle, self)
      call self%destroy(error)
      deallocate(self)
    end subroutine

    subroutine elpa_autotune_impl_deallocate_c2( autotune_handle, error) bind(C, name="elpa_autotune_deallocate2")
      type(c_ptr), value                  :: autotune_handle

      type(elpa_autotune_impl_t), pointer :: self
      integer(kind=c_int)                 :: error
      call c_f_pointer(autotune_handle, self)
      call self%destroy(error)
      deallocate(self)
    end subroutine
#else
    !c_no> #ifndef OPTIONAL_C_ERROR_ARGUMENT
    !c_no> void elpa_autotune_deallocate(elpa_autotune_t handle, int *error);
    !c_no> #endif
    subroutine elpa_autotune_impl_deallocate( autotune_handle, error) bind(C, name="elpa_autotune_deallocate")
      type(c_ptr), value                  :: autotune_handle

      type(elpa_autotune_impl_t), pointer :: self
      integer(kind=c_int)                 :: error
      call c_f_pointer(autotune_handle, self)
      call self%destroy(error)
      deallocate(self)
    end subroutine

#endif
#endif /* ENABLE_AUTOTUNING */

    !> \brief function to setup an ELPA object and to store the MPI communicators internally
    !> Parameters
    !> \param   self       class(elpa_impl_t), the allocated ELPA object
    !> \result  error      integer, the error code
    function elpa_setup(self) result(error)
      class(elpa_impl_t), intent(inout)   :: self
      integer                             :: error, timings, performance, build_config

#ifdef WITH_MPI
      integer                             :: mpi_comm_parent, mpi_comm_rows, mpi_comm_cols, np_rows, np_cols, my_id, &
                                             process_row, process_col, mpi_string_length, &
                                             present_np_rows, present_np_cols, np_total
      integer(kind=MPI_KIND)              :: mpierr, mpierr2, my_idMPI, np_totalMPI, process_rowMPI, process_colMPI
      integer(kind=MPI_KIND)              :: mpi_comm_rowsMPI, mpi_comm_colsMPI, np_rowsMPI, np_colsMPI, &
                                             mpi_string_lengthMPI
      character(len=MPI_MAX_ERROR_STRING) :: mpierr_string
      character(*), parameter             :: MPI_CONSISTENCY_MSG = &
        "Provide mpi_comm_parent and EITHER process_row and process_col OR mpi_comm_rows and mpi_comm_cols. Aborting..."

#endif


#ifdef HAVE_LIKWID
      !initialize likwid
      call likwid_markerInit()
      call likwid_markerThreadInit()
      call likwid_markerStartRegion("TOTAL")
#endif

#ifdef HAVE_DETAILED_TIMINGS
      call self%get("timings",timings, error)
      call self%get("measure_performance",performance, error)
      if (check_elpa_get(error, ELPA_ERROR_SETUP)) return
      if (timings == 1) then
        call self%timer%enable()
        if (performance == 1) then
          call self%timer%measure_flops(.true.)
          call self%timer%set_print_options(print_flop_count=.true.,print_flop_rate=.true.)
        endif
      endif
#endif

      error = ELPA_OK

      ! In most cases, we actually need the parent communicator to be supplied,
      ! ELPA internally requires it when either GPU is enabled or when ELPA2 is
      ! used. It thus seems reasonable that we should ALLWAYS require it. It
      ! should then be accompanied by EITHER process_row and process_col
      ! indices, OR mpi_comm_rows and mpi_comm_cols communicators, but NOT both.
      ! This assumption will significanlty simplify the logic, avoid possible
      ! inconsistencies and is rather natural from the user point of view

#ifdef WITH_MPI
      if (self%is_set("mpi_comm_parent") == 1) then
        call self%get("mpi_comm_parent", mpi_comm_parent, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return

        call mpi_comm_rank(int(mpi_comm_parent,kind=MPI_KIND), my_idMPI, mpierr)
        my_id = int(my_idMPI, kind=c_int)
        call self%set("process_id", my_id, error)
        if (check_elpa_set(error, ELPA_ERROR_SETUP)) return

        call mpi_comm_size(int(mpi_comm_parent,kind=MPI_KIND), np_totalMPI, mpierr)
        np_total = int(np_totalMPI,kind=c_int)
        call self%set("num_processes", np_total, error)
        if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
      else
        if (self%from_legacy_api .ne. 1) then
          write(error_unit,*) MPI_CONSISTENCY_MSG
          error = ELPA_ERROR
          return
        endif
      endif

      ! Create communicators ourselves
      if (self%is_set("process_row") == 1 .and. self%is_set("process_col") == 1) then

        if (self%is_set("mpi_comm_rows") == 1 .or. self%is_set("mpi_comm_cols") == 1) then
          write(error_unit,*) MPI_CONSISTENCY_MSG
          error = ELPA_ERROR
          return
        endif

        call self%get("process_row", process_row, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return

        call self%get("process_col", process_col, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return

        ! mpi_comm_rows is used for communicating WITHIN rows, i.e. all processes
        ! having the same column coordinate share one mpi_comm_rows.
        ! So the "color" for splitting is process_col and the "key" is my row coordinate.
        ! Analogous for mpi_comm_cols

        call mpi_comm_split(int(mpi_comm_parent,kind=MPI_KIND), int(process_col,kind=MPI_KIND), &
                            int(process_row,kind=MPI_KIND), mpi_comm_rowsMPI, mpierr)
        mpi_comm_rows = int(mpi_comm_rowsMPI,kind=c_int)
        if (mpierr .ne. MPI_SUCCESS) then
          call MPI_ERROR_STRING(mpierr, mpierr_string, mpi_string_lengthMPI, mpierr2)
          mpi_string_length = int(mpi_string_lengthMPI, kind=c_int)
          write(error_unit,*) "MPI ERROR occured during mpi_comm_split for row communicator: ", trim(mpierr_string)
          return
        endif

        call mpi_comm_split(int(mpi_comm_parent,kind=MPI_KIND), int(process_row,kind=MPI_KIND), &
                            int(process_col,kind=MPI_KIND), mpi_comm_colsMPI, mpierr)
        mpi_comm_cols = int(mpi_comm_colsMPI,kind=c_int)
        if (mpierr .ne. MPI_SUCCESS) then
          call MPI_ERROR_STRING(mpierr, mpierr_string, mpi_string_lengthMPI, mpierr2)
          mpi_string_length = int(mpi_string_lengthMPI, kind=c_int)
          write(error_unit,*) "MPI ERROR occured during mpi_comm_split for col communicator: ", trim(mpierr_string)
          return
        endif

        call self%set("mpi_comm_rows", mpi_comm_rows,error)
        if (check_elpa_set(error, ELPA_ERROR_SETUP)) return

        call self%set("mpi_comm_cols", mpi_comm_cols,error)
        if (check_elpa_set(error, ELPA_ERROR_SETUP)) return

        ! remember that we created those communicators and we need to free them later
        self%communicators_owned = 1

      ! Externally supplied communicators
      else if ( self%is_set("mpi_comm_rows") == 1 .and.  self%is_set("mpi_comm_cols") == 1) then

        if (self%is_set("process_row") == 1 .or. self%is_set("process_col") == 1) then
          write(error_unit,*) MPI_CONSISTENCY_MSG
          error = ELPA_ERROR
          return
        endif

        call self%get("mpi_comm_rows", mpi_comm_rows,error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return

        call self%get("mpi_comm_cols", mpi_comm_cols,error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return

        process_rowMPI = int(process_row,kind=c_int)
        call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND), process_rowMPI, mpierr)
        process_row = int(process_rowMPI,kind=MPI_KIND)
        call self%set("process_row", process_row, error)
        if (check_elpa_set(error, ELPA_ERROR_SETUP)) return

        process_colMPI = int(process_col,kind=c_int)
        call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND), process_colMPI, mpierr)
        process_col = int(process_colMPI,kind=MPI_KIND)
        call self%set("process_col", process_col, error)
        if (check_elpa_set(error, ELPA_ERROR_SETUP)) return

        ! remember that we DID NOT created those communicators and we WILL NOT free them later
        self%communicators_owned = 0
      else
        ! Otherwise parameters are missing
        write(error_unit,*) MPI_CONSISTENCY_MSG
        error = ELPA_ERROR
        return
      endif

      ! set num_process_rows (and cols), if they are not supplied. Check them
      ! for consistency if they are. Maybe we could instead require, that they
      ! are never supplied?
      call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)
      np_rows = int(np_rowsMPI, kind=c_int)
      if (self%is_set("num_process_rows") == 1) then
        call self%get("num_process_rows", present_np_rows, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return

        if (np_rows .ne. present_np_rows) then
          print *,"MPI row communicator not set correctly. Aborting..."
          stop
        endif
      else
        call self%set("num_process_rows", np_rows, error)
        if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
      endif

      call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND), np_colsMPI, mpierr)
      np_cols = int(np_colsMPI, kind=c_int)
      if (self%is_set("num_process_cols") == 1) then
        call self%get("num_process_cols", present_np_cols, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return

        if (np_cols .ne. present_np_cols) then
          print *,"MPI column communicator not set correctly. Aborting..."
          stop
        endif
      else
        call self%set("num_process_cols", np_cols, error)
        if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
      endif

      if (self%from_legacy_api .ne. 1) then
        if (np_total .ne. np_rows * np_cols) then
          print *,"MPI parent communicator and row/col communicators do not match. Aborting..."
          stop
        endif
      endif

#else
      call self%set("process_row", 0, error)
      if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
      call self%set("process_col", 0, error)
      if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
      call self%set("process_id", 0, error)
      if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
      call self%set("num_process_rows", 1, error)
      if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
      call self%set("num_process_cols", 1, error)
      if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
      call self%set("num_processes", 1, error)
      if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
#endif

#if STORE_BUILD_CONFIG
      call self%get("output_build_config",build_config, error)
      if ( build_config .eq. 1) then
#ifdef WITH_MPI
        if (my_id .eq. 0) then
#endif
          call print_build_config()
#ifdef WITH_MPI
        endif
#endif
      endif
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
      if (check_elpa_get(error, ELPA_ERROR_CRITICAL)) return

      sc_desc(1) = 1
      sc_desc(2) = blacs_ctx
      sc_desc(3) = self%na
      if (rectangular_for_ev) then
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
      integer(kind=c_int) , intent(in)              :: error

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
      integer(kind=c_int), intent(inout)            :: error
 
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
#ifdef USE_FORTRAN2008
      integer, intent(out), optional :: error
#else
      integer, intent(out)           :: error
#endif

      integer :: val, actual_error
      character(kind=c_char, len=elpa_index_int_value_to_strlen_c(self%index, option_name // C_NULL_CHAR)), pointer :: string

      nullify(string)

      call self%get(option_name, val, actual_error)
      if (actual_error /= ELPA_OK) then
#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = actual_error
        endif
#else
          error = actual_error
#endif
        return
      endif

      actual_error = elpa_int_value_to_string_c(option_name // C_NULL_CHAR, val, ptr)
      if (c_associated(ptr)) then
        call c_f_pointer(ptr, string)
      endif

#ifdef USE_FORTRAN2008
      if (present(error)) then
        error = actual_error
      endif
#else
        error = actual_error
#endif
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
      integer(kind=c_int), intent(in)               :: error

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
      integer(kind=c_int), intent(inout)            :: error

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
    !> \param   error           integer, optional error code
    subroutine elpa_destroy(self, error)
#ifdef WITH_MPI
      integer                              :: mpi_comm_rows, mpi_comm_cols, &
                                              mpi_string_length
      integer(kind=MPI_KIND)               :: mpierr, mpierr2, mpi_string_lengthMPI, &
                                              mpi_comm_rowsMPI, mpi_comm_colsMPI
      character(len=MPI_MAX_ERROR_STRING)  :: mpierr_string
#endif
      class(elpa_impl_t)                   :: self
#ifdef USE_FORTRAN2008
      integer, optional, intent(out)       :: error
#else
      integer, intent(out)                 :: error
#endif
      integer                              :: error2

#ifdef USE_FORTRAN2008
      if (present(error)) then
        error = ELPA_OK
      endif
#else
      error = ELPA_OK
#endif

#ifdef HAVE_LIKWID
      call likwid_markerStopRegion("TOTAL")
      call likwid_markerClose()
#endif

#ifdef WITH_MPI
      if (self%communicators_owned == 1) then
        call self%get("mpi_comm_rows", mpi_comm_rows, error2)
        if (error2 .ne. ELPA_OK) then
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = error2
          else
            write(error_unit, *) "Error in elpa_destroy but you do not check the error codes!"
          endif
#else
          error = error2
#endif
          return
        endif ! error happend

        call self%get("mpi_comm_cols", mpi_comm_cols,error2)
        if (error2 .ne. ELPA_OK) then
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = error2
          else
            write(error_unit, *) "Error in elpa_destroy but you do not check the error codes!"
          endif
#else
          error = error2
#endif
          return
        endif ! error happend

        ! this is just for debugging ! do not leave in a relase
        !write(error_unit, '(A,2I13)') "FREE comms", mpi_comm_rows, mpi_comm_cols
        mpi_comm_rowsMPI = int(mpi_comm_rows,kind=MPI_KIND)
        call mpi_comm_free(mpi_comm_rowsMPI, mpierr)
        mpi_comm_rows = int(mpi_comm_rowsMPI,kind=c_int)
        if (mpierr .ne. MPI_SUCCESS) then
          call MPI_ERROR_STRING(mpierr, mpierr_string, mpi_string_lengthMPI, mpierr2)
          mpi_string_length = int(mpi_string_lengthMPI,kind=c_int)
          write(error_unit,*) "MPI ERROR occured during mpi_comm_free for row communicator: ", trim(mpierr_string)
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = ELPA_ERROR_CRITICAL
          endif
#else
          error = ELPA_ERROR_CRITICAL
#endif
          return
        endif ! mpierr happend
        call self%set("mpi_comm_cols", -12345, error2)
        if (error2 .ne. ELPA_OK) then
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = error2
          else
            write(error_unit, *) "Error in elpa_destroy but you do not check the error codes!"
          endif
#else
          error = error2
#endif
          return
        endif ! error happend
        mpi_comm_colsMPI = int(mpi_comm_cols,kind=MPI_KIND)
        call mpi_comm_free(mpi_comm_colsMPI, mpierr)
        mpi_comm_cols = int(mpi_comm_colsMPI, kind=c_int)
        if (mpierr .ne. MPI_SUCCESS) then
          call MPI_ERROR_STRING(mpierr, mpierr_string, mpi_string_lengthMPI, mpierr2)
          mpi_string_length = int(mpi_string_lengthMPI,kind=c_int)
          write(error_unit,*) "MPI ERROR occured during mpi_comm_free for col communicator: ", trim(mpierr_string)
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = ELPA_ERROR_CRITICAL
          endif
#else
          error = ELPA_ERROR_CRITICAL
#endif
          return
        endif ! mpierr happend
        call self%set("mpi_comm_rows", -12345,error2)
        if (error2 .ne. ELPA_OK) then
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = error2
          else
            write(error_unit, *) "Error in elpa_destroy but you do not check the error codes!"
          endif
#else
          error = error2
#endif
          return
        endif ! error happend
      endif
#endif /* WITH_MPI */

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
          error = ELPA_ERROR_AUTOTUNE_API_VERSION
        endif
#else
        error = ELPA_ERROR_AUTOTUNE_API_VERSION
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
      integer(kind=c_int) , intent(in)       :: error

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
    function elpa_autotune_step(self, tune_state, error) result(unfinished)
      implicit none
      class(elpa_impl_t), intent(inout)             :: self
      class(elpa_autotune_t), intent(inout), target :: tune_state
      type(elpa_autotune_impl_t), pointer           :: ts_impl
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(out)    :: error
#else
      integer(kind=c_int),  intent(out)             :: error
#endif
      integer(kind=c_int)                           :: error2, error3
      integer                                       :: mpi_comm_parent, mpi_string_length, np_total
      integer(kind=MPI_KIND)                        :: mpierr, mpierr2, mpi_string_lengthMPI
      logical                                       :: unfinished
      integer                                       :: i
      real(kind=C_DOUBLE)                           :: time_spent, sendbuf(1), recvbuf(1)
#ifdef WITH_MPI
      character(len=MPI_MAX_ERROR_STRING)           :: mpierr_string
#endif


#ifdef USE_FORTRAN2008
      if (present(error)) then
        error = ELPA_OK
      endif
#else
      error = ELPA_OK
#endif
      select type(tune_state)
        type is (elpa_autotune_impl_t)
          ts_impl => tune_state
        class default
          print *, "This should not happen"
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = ELPA_ERROR
          endif
#else
          error = ELPA_ERROR
#endif
      end select

      unfinished = .false.

      if (ts_impl%current >= 0) then
#ifdef HAVE_DETAILED_TIMINGS
        time_spent = self%autotune_timer%get("accumulator")
#else
        print *, "Cannot do autotuning without detailed timings"

        ! TODO check this. Do we really want to return only if error is present? And should it be ELPA_OK?
#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = ELPA_ERROR_CRITICAL
        endif
#else
        error = ELPA_OK
#endif
        return
#endif /* HAVE_DETAILED_TIMINGS */

#ifdef WITH_MPI
        ! find the average time spent .. we need a unique value on all ranks
        call self%get("mpi_comm_parent", mpi_comm_parent, error2)
        call self%get("num_processes", np_total, error3)
        if ((error2 .ne. ELPA_OK) .or. (error3 .ne. ELPA_OK)) then
          print *, "Parent communicator is not set properly. Aborting..."
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = ELPA_ERROR_CRITICAL
          endif
#else
          error = ELPA_ERROR_CRITICAL
#endif
          return
        endif

        sendbuf(1) = time_spent
        call MPI_Allreduce(sendbuf, recvbuf, 1_MPI_KIND, MPI_REAL8, MPI_SUM, int(mpi_comm_parent,kind=MPI_KIND), mpierr)
        if (mpierr .ne. MPI_SUCCESS) then
          call MPI_ERROR_STRING(mpierr, mpierr_string, mpi_string_lengthMPI, mpierr2)
          mpi_string_length = int(mpi_string_lengthMPI,kind=c_int)
          write(error_unit,*) "MPI ERROR occured during elpa_autotune_step: ", trim(mpierr_string)
          return
        endif
        time_spent = recvbuf(1) / np_total
#endif /* WITH_MPI */

        if (ts_impl%min_loc == -1 .or. (time_spent < ts_impl%min_val)) then
          ts_impl%min_val = time_spent
          ts_impl%min_loc = ts_impl%current
        end if
        call self%autotune_timer%free()
      endif ! (ts_impl%current >= 0)

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
    !c> *  \param  error            int *error code
    !c> *  \result int              unfinished:  describes whether autotuning finished (0) or not (1)
    !c> */
    !c> int elpa_autotune_step(elpa_t handle, elpa_autotune_t autotune_handle, int *error);
    function elpa_autotune_step_c(handle, autotune_handle, &
                    error) result(unfinished) bind(C, name="elpa_autotune_step")
      type(c_ptr), intent(in), value       :: handle
      type(c_ptr), intent(in), value       :: autotune_handle
      type(elpa_impl_t), pointer           :: self
      type(elpa_autotune_impl_t), pointer  :: tune_state
      logical                              :: unfinished_f
      integer(kind=c_int)                  :: unfinished
      integer(kind=c_int)                  :: error

      call c_f_pointer(handle, self)
      call c_f_pointer(autotune_handle, tune_state)

      unfinished_f = self%autotune_step(tune_state, error)
      if (unfinished_f) then
        unfinished = 1
      else
        unfinished = 0
      endif

    end function

    !> \brief function to set the up-to-now best options of the autotuning
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    !> \param   tune_state      class(elpa_autotune_t): the autotuning object
    !> \param   error code      optional, integer
    subroutine elpa_autotune_set_best(self, tune_state, error)
      implicit none
      class(elpa_impl_t), intent(inout)          :: self
      class(elpa_autotune_t), intent(in), target :: tune_state
      type(elpa_autotune_impl_t), pointer        :: ts_impl
#ifdef USE_FORTRAN2008
      integer(kind=ik), optional, intent(out)    :: error
#else
      integer(kind=ik), intent(out)              :: error
#endif

#ifdef USE_FORTRAN2008
      if (present(error)) then
        error = ELPA_OK
      endif
#else
      error = ELPA_OK
#endif
      select type(tune_state)
        type is (elpa_autotune_impl_t)
          ts_impl => tune_state
        class default
          write(error_unit, *) "This should not happen! Critical error"
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = ELPA_ERROR_CRITICAL
          endif
#else
          error = ELPA_ERROR_CRITICAL
#endif
      end select

      if (elpa_index_set_autotune_parameters_c(self%index, ts_impl%level, ts_impl%domain, ts_impl%min_loc) /= 1) then
        write(error_unit, *) "This should not happen (in elpa_autotune_set_best())"
#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
        endif
#else
        error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
#endif
      endif
    end subroutine



    !> \brief function to print the up-to-now best options of the autotuning
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    !> \param   tune_state      class(elpa_autotune_t): the autotuning object
    !> \param   error           integer, optional
    subroutine elpa_autotune_print_best(self, tune_state, error)
      implicit none
      class(elpa_impl_t), intent(inout)          :: self
      class(elpa_autotune_t), intent(in), target :: tune_state
      type(elpa_autotune_impl_t), pointer        :: ts_impl
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(out) :: error
#else
      integer(kind=c_int),  intent(out)          :: error
#endif

#ifdef USE_FORTRAN2008
      if (present(error)) then
        error = ELPA_OK
      endif
#else
      error = ELPA_OK
#endif
      select type(tune_state)
        type is (elpa_autotune_impl_t)
          ts_impl => tune_state
        class default
          write(error_unit, *) "This should not happen! Critical error"
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = ELPA_ERROR_CRITICAL
          endif
#else
          error = ELPA_ERROR_CRITICAL
#endif
      end select

      !print *, "The following settings were found to be best:"
      !print *, "Best, i = ", ts_impl%min_loc, "best time = ", ts_impl%min_val
      flush(output_unit)
      if (elpa_index_print_autotune_parameters_c(self%index, ts_impl%level, ts_impl%domain) /= 1) then
        write(error_unit, *) "This should not happen (in elpa_autotune_print_best())"
#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
        endif
#else
        error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
#endif
      endif
    end subroutine



    !> \brief function to print the state of the autotuning
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    !> \param   tune_state      class(elpa_autotune_t): the autotuning object
    !> \param   error           integer, optional
    subroutine elpa_autotune_print_state(self, tune_state, error)
      implicit none
      class(elpa_impl_t), intent(inout)          :: self
      class(elpa_autotune_t), intent(in), target :: tune_state
      type(elpa_autotune_impl_t), pointer        :: ts_impl
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(out) :: error
#else
      integer(kind=c_int), intent(out)           :: error
#endif

#ifdef USE_FORTRAN2008
      if (present(error)) then
        error = ELPA_OK
      endif
#else
      error = ELPA_OK
#endif
      select type(tune_state)
        type is (elpa_autotune_impl_t)
          ts_impl => tune_state
        class default
          write(error_unit, *) "This should not happen! Critical erro"
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = ELPA_ERROR_CRITICAL
          endif
#else
          error = ELPA_ERROR_CRITICAL
#endif
      end select

      if (elpa_index_print_autotune_state_c(self%index, ts_impl%level, ts_impl%domain, ts_impl%min_loc, &
                  ts_impl%min_val, ts_impl%current, ts_impl%cardinality, c_null_char) /= 1) then
        write(error_unit, *) "This should not happen (in elpa_autotune_print_state())"
#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
        endif
#else
        error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
#endif
      endif
    end subroutine


    !c> /*! \brief C interface for the implementation of the elpa_autotune_print_state method
    !c> *
    !c> *  \param  elpa_t           handle: of the ELPA object which should be tuned
    !c> *  \param  elpa_autotune_t  autotune_handle: the autotuning object
    !c> *  \param  error            int *
    !c> *  \result none 
    !c> */
    !c> void elpa_autotune_print_state(elpa_t handle, elpa_autotune_t autotune_handle, int *error);
    subroutine elpa_autotune_print_state_c(handle, autotune_handle, error) bind(C, name="elpa_autotune_print_state")
      type(c_ptr), intent(in), value       :: handle
      type(c_ptr), intent(in), value       :: autotune_handle
      type(elpa_impl_t), pointer           :: self
      type(elpa_autotune_impl_t), pointer  :: tune_state
      integer(kind=c_int)                  :: error

      call c_f_pointer(handle, self)
      call c_f_pointer(autotune_handle, tune_state)

      call self%autotune_print_state(tune_state, error)

    end subroutine



    !> \brief function to save the state of the autotuning
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    !> \param   tune_state      class(elpa_autotune_t): the autotuning object
    !> \param   file_name       string, the name of the file where to save the state
    !> \param   error           integer, optional
    subroutine elpa_autotune_save_state(self, tune_state, file_name, error)
      implicit none
      class(elpa_impl_t), intent(inout)          :: self
      class(elpa_autotune_t), intent(in), target :: tune_state
      type(elpa_autotune_impl_t), pointer        :: ts_impl
      character(*), intent(in)                   :: file_name
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(out) :: error
#else
      integer(kind=c_int), intent(out)           :: error
#endif

#ifdef USE_FORTRAN2008
      if (present(error)) then
        error = ELPA_OK
      endif
#else
      error = ELPA_OK
#endif
      select type(tune_state)
        type is (elpa_autotune_impl_t)
          ts_impl => tune_state
        class default
          write(error_unit, *) "This should not happen! Critical error"
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = ELPA_ERROR_CRITICAL
          endif
#else
          error = ELPA_ERROR_CRITICAL
#endif
      end select

      if (elpa_index_print_autotune_state_c(self%index, ts_impl%level, ts_impl%domain, ts_impl%min_loc, &
                  ts_impl%min_val, ts_impl%current, ts_impl%cardinality, file_name // c_null_char) /= 1) then
        write(error_unit, *) "This should not happen (in elpa_autotune_save_state())"
#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = ELPA_ERROR_CANNOT_OPEN_FILE
        endif
#else
        error = ELPA_ERROR_CANNOT_OPEN_FILE
#endif
      endif
    end subroutine



    !c> /*! \brief C interface for the implementation of the elpa_autotune_save_state method
    !c> *
    !c> *  \param  elpa_t           handle: of the ELPA object which should be tuned
    !c> *  \param  elpa_autotune_t  autotune_handle: the autotuning object
    !c> *  \param  error            int *
    !c> *  \result none 
    !c> */
    !c> void elpa_autotune_save_state(elpa_t handle, elpa_autotune_t autotune_handle, const char *filename, int *error);
    subroutine elpa_autotune_save_state_c(handle, autotune_handle, filename_p, error) bind(C, name="elpa_autotune_save_state")
      type(c_ptr), intent(in), value       :: handle
      type(c_ptr), intent(in), value       :: autotune_handle
      type(elpa_impl_t), pointer           :: self
      type(elpa_autotune_impl_t), pointer  :: tune_state
      type(c_ptr), intent(in), value       :: filename_p
      character(len=elpa_strlen_c(filename_p)), pointer :: filename
      integer(kind=c_int)                  :: error

      call c_f_pointer(handle, self)
      call c_f_pointer(filename_p, filename)
      call c_f_pointer(autotune_handle, tune_state)

      call self%autotune_save_state(tune_state, filename, error)

    end subroutine



    !> \brief function to load the state of the autotuning
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    !> \param   tune_state      class(elpa_autotune_t): the autotuning object
    !> \param   file_name       string, the name of the file from which to load the state
    !> \param   error           integer, optional
    subroutine elpa_autotune_load_state(self, tune_state, file_name, error)
      implicit none
      class(elpa_impl_t), intent(inout)          :: self
      class(elpa_autotune_t), intent(in), target :: tune_state
      type(elpa_autotune_impl_t), pointer        :: ts_impl
      character(*), intent(in)                   :: file_name
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(out) :: error
#else
      integer(kind=c_int), intent(out)           :: error
#endif

#ifdef USE_FORTRAN2008
      if (present(error)) then
        error = ELPA_OK
      endif
#else
      error = ELPA_OK
#endif
      select type(tune_state)
        type is (elpa_autotune_impl_t)
          ts_impl => tune_state
        class default
          write(error_unit, *) "This should not happen! Critical error"
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = ELPA_ERROR_CRITICAL
          endif
#else
          error = ELPA_ERROR_CRITICAL
#endif
      end select


      if (elpa_index_load_autotune_state_c(self%index, ts_impl%level, ts_impl%domain, ts_impl%min_loc, &
                  ts_impl%min_val, ts_impl%current, ts_impl%cardinality, file_name // c_null_char) /= 1) then
         write(error_unit, *) "This should not happen (in elpa_autotune_load_state())"
#ifdef USE_FORTRAN2008
         if (present(error)) then
           error = ELPA_ERROR_CANNOT_OPEN_FILE
         endif
#else
         error = ELPA_ERROR_CANNOT_OPEN_FILE
#endif
      endif
    end subroutine



    !c> /*! \brief C interface for the implementation of the elpa_autotune_load_state method
    !c> *
    !c> *  \param  elpa_t           handle: of the ELPA object which should be tuned
    !c> *  \param  elpa_autotune_t  autotune_handle: the autotuning object
    !c> *  \param  error            int *
    !c> *  \result none 
    !c> */
    !c> void elpa_autotune_load_state(elpa_t handle, elpa_autotune_t autotune_handle, const char *filename, int *error);
    subroutine elpa_autotune_load_state_c(handle, autotune_handle, filename_p, error) bind(C, name="elpa_autotune_load_state")
      type(c_ptr), intent(in), value       :: handle
      type(c_ptr), intent(in), value       :: autotune_handle
      type(elpa_impl_t), pointer           :: self
      type(elpa_autotune_impl_t), pointer  :: tune_state
      type(c_ptr), intent(in), value       :: filename_p
      character(len=elpa_strlen_c(filename_p)), pointer :: filename
      integer(kind=c_int)                  :: error

      call c_f_pointer(handle, self)
      call c_f_pointer(filename_p, filename)
      call c_f_pointer(autotune_handle, tune_state)

      call self%autotune_load_state(tune_state, filename, error)

    end subroutine


    !c> /*! \brief C interface for the implementation of the elpa_autotune_set_best method
    !c> *
    !c> *  \param  elpa_t           handle: of the ELPA object which should be tuned
    !c> *  \param  elpa_autotune_t  autotune_handle: the autotuning object
    !c> *  \param  error            int *
    !c> *  \result none 
    !c> */
    !c> void elpa_autotune_set_best(elpa_t handle, elpa_autotune_t autotune_handle, int *error);
    subroutine elpa_autotune_set_best_c(handle, autotune_handle, error) bind(C, name="elpa_autotune_set_best")
      type(c_ptr), intent(in), value       :: handle
      type(c_ptr), intent(in), value       :: autotune_handle
      type(elpa_impl_t), pointer           :: self
      type(elpa_autotune_impl_t), pointer  :: tune_state
      integer(kind=c_int)                  :: error

      call c_f_pointer(handle, self)
      call c_f_pointer(autotune_handle, tune_state)

      call self%autotune_set_best(tune_state, error)

    end subroutine


    !c> /*! \brief C interface for the implementation of the elpa_autotune_print_best method
    !c> *
    !c> *  \param  elpa_t           handle: of the ELPA object which should be tuned
    !c> *  \param  elpa_autotune_t  autotune_handle: the autotuning object
    !c> *  \param  error            int *
    !c> *  \result none 
    !c> */
    !c> void elpa_autotune_print_best(elpa_t handle, elpa_autotune_t autotune_handle, int *error);
    subroutine elpa_autotune_print_best_c(handle, autotune_handle, error) bind(C, name="elpa_autotune_print_best")
      type(c_ptr), intent(in), value       :: handle
      type(c_ptr), intent(in), value       :: autotune_handle
      type(elpa_impl_t), pointer           :: self
      type(elpa_autotune_impl_t), pointer  :: tune_state
      integer(kind=c_int)                  :: error

      call c_f_pointer(handle, self)
      call c_f_pointer(autotune_handle, tune_state)

      call self%autotune_print_best(tune_state, error)

    end subroutine

#endif /* HAVE_AUTOTUNING */

    function check_elpa(error, str, new_error) result(res)
      integer, intent(inout) :: error
      integer, intent(in)    :: new_error
      character(*)  :: str
      logical :: res
      if (error .ne. ELPA_OK) then
        print *, trim(str)
        res = .true.
        error = new_error
        return
      endif
      res = .false.
    end function

    function check_elpa_get(error, new_error) result(res)
      integer, intent(inout) :: error
      integer, intent(in)    :: new_error
      logical :: res
      res = check_elpa(error, "Problem getting option. Aborting...", new_error)
      return
    end function

    function check_elpa_set(error, new_error) result(res)
      integer, intent(inout) :: error
      integer, intent(in)    :: new_error
      logical :: res
      res = check_elpa(error, "Problem setting option. Aborting...", new_error)
      return
    end function

    subroutine elpa_creating_from_legacy_api(self)
      implicit none
      class(elpa_impl_t), intent(inout)          :: self

      self%from_legacy_api = 1
    end subroutine
end module
