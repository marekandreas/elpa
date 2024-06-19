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
  !use elpa1_auxiliary_impl
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
  use elpa1_auxiliary_impl
  !use elpa_gpu_setup
  use, intrinsic :: iso_c_binding
  use iso_fortran_env
  implicit none

  private
  public :: elpa_impl_allocate

  integer(kind=c_int), private :: autotune_level, autotune_domain
  logical, private, save       :: autotune_substeps_done1stage(0:ELPA_NUMBER_OF_AUTOTUNE_LEVELS-1) = .false.
  logical, private, save       :: autotune_substeps_done2stage(0:ELPA_NUMBER_OF_AUTOTUNE_LEVELS-1) = .false.
  logical, private, save       :: do_autotune_2stage = .false., do_autotune_1stage = .false.
  logical, private, save       :: last_call_2stage = .false., last_call_1stage = .false.
  logical, private, save       :: autotune_api_set = .false.
  logical, private, save       :: new_autotune = .false.
  integer(kind=c_int), private :: consider_solver

!> \brief Definition of the extended elpa_impl_t type
  type, extends(elpa_abstract_impl_t) :: elpa_impl_t
   private
   integer :: communicators_owned

   !This object has been created through the legacy api.
   integer :: from_legacy_api

   !type(elpa_gpu_setup_t), public :: gpu_setup

   !> \brief methods available with the elpa_impl_t type
   contains
     !> \brief the puplic methods
     ! con-/destructor
     procedure, public :: setup => elpa_setup                   !< a setup method: implemented in elpa_setup
     procedure, public :: destroy => elpa_destroy               !< a destroy method: implemented in elpa_destroy

     procedure, public :: setup_gpu => elpa_setup_gpu           !< setup the GPU usage
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

     procedure, public :: elpa_eigenvectors_a_h_a_d                  !< public methods to implement the solve step for real/complex
                                                                               !< double/single matrices
     procedure, public :: elpa_eigenvectors_a_h_a_f
     procedure, public :: elpa_eigenvectors_a_h_a_dc
     procedure, public :: elpa_eigenvectors_a_h_a_fc

     procedure, public :: elpa_eigenvectors_d_ptr_d 
     procedure, public :: elpa_eigenvectors_d_ptr_f
     procedure, public :: elpa_eigenvectors_d_ptr_dc
     procedure, public :: elpa_eigenvectors_d_ptr_fc

     procedure, public :: elpa_eigenvalues_a_h_a_d                   !< public methods to implement the solve step for real/complex
                                                                               !< double/single matrices; only the eigenvalues are computed
     procedure, public :: elpa_eigenvalues_a_h_a_f
     procedure, public :: elpa_eigenvalues_a_h_a_dc
     procedure, public :: elpa_eigenvalues_a_h_a_fc

     procedure, public :: elpa_eigenvalues_d_ptr_d 
     procedure, public :: elpa_eigenvalues_d_ptr_f
     procedure, public :: elpa_eigenvalues_d_ptr_dc
     procedure, public :: elpa_eigenvalues_d_ptr_fc

#ifdef HAVE_SKEWSYMMETRIC
     procedure, public :: elpa_skew_eigenvectors_a_h_a_d             !< public methods to implement the solve step for real skew-symmetric
                                                                               !< double/single matrices
     procedure, public :: elpa_skew_eigenvectors_a_h_a_f

     procedure, public :: elpa_skew_eigenvalues_a_h_a_d              !< public methods to implement the solve step for real skew-symmetric
                                                                               !< double/single matrices; only the eigenvalues are computed
     procedure, public :: elpa_skew_eigenvalues_a_h_a_f

     procedure, public :: elpa_skew_eigenvectors_d_ptr_d 
     procedure, public :: elpa_skew_eigenvectors_d_ptr_f
     procedure, public :: elpa_skew_eigenvalues_d_ptr_d 
     procedure, public :: elpa_skew_eigenvalues_d_ptr_f
#endif

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

     procedure, public :: elpa_hermitian_multiply_a_h_a_d      !< public methods to implement a "hermitian" multiplication of matrices a and b
     procedure, public :: elpa_hermitian_multiply_a_h_a_f            !< for real valued matrices:   a**T * b
     procedure, public :: elpa_hermitian_multiply_a_h_a_dc           !< for complex valued matrices:   a**H * b
     procedure, public :: elpa_hermitian_multiply_a_h_a_fc

     procedure, public :: elpa_hermitian_multiply_d_ptr_d      !< public methods to implement a "hermitian" multiplication of matrices a and b
     procedure, public :: elpa_hermitian_multiply_d_ptr_f            !< for real valued matrices:   a**T * b
     procedure, public :: elpa_hermitian_multiply_d_ptr_dc           !< for complex valued matrices:   a**H * b
     procedure, public :: elpa_hermitian_multiply_d_ptr_fc

     procedure, public :: elpa_cholesky_a_h_a_d      !< public methods to implement the cholesky factorisation of
                                                               !< real/complex double/single matrices
     procedure, public :: elpa_cholesky_a_h_a_f
     procedure, public :: elpa_cholesky_a_h_a_dc
     procedure, public :: elpa_cholesky_a_h_a_fc

     procedure, public :: elpa_cholesky_d_ptr_d       !< public methods to implement the cholesky factorisation of
                                                               !< real/complex double/single matrices
     procedure, public :: elpa_cholesky_d_ptr_f
     procedure, public :: elpa_cholesky_d_ptr_dc
     procedure, public :: elpa_cholesky_d_ptr_fc

     procedure, public :: elpa_invert_trm_a_h_a_d    !< public methods to implement the inversion of a triangular
                                                               !< real/complex double/single matrix
     procedure, public :: elpa_invert_trm_a_h_a_f
     procedure, public :: elpa_invert_trm_a_h_a_dc
     procedure, public :: elpa_invert_trm_a_h_a_fc

     procedure, public :: elpa_invert_trm_d_ptr_d     !< public methods to implement the inversion of a triangular
                                                               !< real/complex double/single matrix
     procedure, public :: elpa_invert_trm_d_ptr_f
     procedure, public :: elpa_invert_trm_d_ptr_dc
     procedure, public :: elpa_invert_trm_d_ptr_fc

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
     procedure, public :: autotune_set_api_version => elpa_autotune_set_api_version
     procedure, public :: autotune_step => elpa_autotune_step
     procedure, public :: autotune_step_worker => elpa_autotune_step_worker
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

    !c> // /src/elpa_impl.F90
    !c> #ifdef __cplusplus
    !c> #define double_complex std::complex<double>
    !c> #define float_complex std::complex<float>
    !c> extern "C" {
    !c> #else
    !c> #define double_complex double complex
    !c> #define float_complex float complex
    !c> #endif
    
#if OPTIONAL_C_ERROR_ARGUMENT == 1
    !c_o> // c_o: /src/elpa_impl.F90 
    !c_o> #if OPTIONAL_C_ERROR_ARGUMENT == 1
    !c_o> #define elpa_allocate(...) CONC(elpa_allocate, NARGS(__VA_ARGS__))(__VA_ARGS__)
    !c_o> #endif
#endif
    !c> /*! \brief C interface for the implementation of the elpa_allocate method
    !c> *
    !c> *  \param  none
    !c> *  \result elpa_t handle
    !c> */
#if OPTIONAL_C_ERROR_ARGUMENT == 1
    !c_o> #if OPTIONAL_C_ERROR_ARGUMENT == 1
    !c_o> #if OPTIONAL_C_ERROR_ARGUMENT == 1
    !c_o> #define NARGS(...) NARGS_(5, ##__VA_ARGS__, 4, 3, 2, 1, 0)
    !c_o> #define NARGS_(_5, _4, _3, _2, _1, N, ...) N
    !c_o> #define CONC(A, B) CONC_(A, B)
    !c_o> #define CONC_(A, B) A##B
    !c_o> elpa_t elpa_allocate1(int *error);
    !c_o> elpa_t elpa_allocate0();
    !c_o> #endif
    function elpa_impl_allocate_c0() result(ptr) bind(C, name="elpa_allocate0")
      type(c_ptr)                :: ptr
      type(elpa_impl_t), pointer :: obj

      obj => elpa_impl_allocate()
      ptr = c_loc(obj)
    end function

    function elpa_impl_allocate_c1(error) result(ptr) bind(C, name="elpa_allocate1")
      integer(kind=c_int)        :: error
      type(c_ptr)                :: ptr
      type(elpa_impl_t), pointer :: obj

      obj => elpa_impl_allocate(error)
      ptr = c_loc(obj)
    end function
#else
    !c_no> // c_no: /src/elpa_impl.F90 
    !c_no> #if OPTIONAL_C_ERROR_ARGUMENT != 1
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

#if OPTIONAL_C_ERROR_ARGUMENT == 1
    !c_o> #define elpa_deallocate(...) CONC(elpa_deallocate, NARGS(__VA_ARGS__))(__VA_ARGS__)
    !c_o> #endif
#endif
    !c> /*! \brief C interface for the implementation of the elpa_deallocate method
    !c> *
    !c> *  \param  elpa_t  handle of ELPA object to be deallocated
    !c> *  \param  int*    error code
    !c> *  \result void
    !c> */
#if OPTIONAL_C_ERROR_ARGUMENT == 1
    !c_o> #if OPTIONAL_C_ERROR_ARGUMENT == 1
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
    !c_no> #if OPTIONAL_C_ERROR_ARGUMENT != 1
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
#if OPTIONAL_C_ERROR_ARGUMENT == 1
    !c_o> #if OPTIONAL_C_ERROR_ARGUMENT == 1
    !c_o> #define elpa_autotune_deallocate(...) CONC(elpa_autotune_deallocate, NARGS(__VA_ARGS__))(__VA_ARGS__)
    !c_o> #endif
#endif
    !c> /*! \brief C interface for the implementation of the elpa_autotune_deallocate method
    !c> *
    !c> *  \param  elpa_autotune_impl_t  handle of ELPA autotune object to be deallocated
    !c> *  \result void
    !c> */
#if OPTIONAL_C_ERROR_ARGUMENT == 1
    !c_o> #if OPTIONAL_C_ERROR_ARGUMENT == 1
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
    !c_no> #if OPTIONAL_C_ERROR_ARGUMENT != 1
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

    !> \brief function to setup the GPU usage
    !> Parameters
    !> \param   self       class(elpa_impl_t), the allocated ELPA object
    !> \result  error      integer, the error code
    function elpa_setup_gpu(self) result(error)
      use precision

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      use mod_query_gpu_usage
      use elpa_gpu, only : gpublasDefaultPointerMode, gpu_getdevicecount, gpublas_get_version
      use elpa_mpi
      use elpa_omp
#endif
#if defined(WITH_NVIDIA_GPU_VERSION)
      use cuda_functions
#endif
#if defined(WITH_AMD_GPU_VERSION)
      use hip_functions
#endif
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION)
      use openmp_offload_functions
#endif
#if defined(WITH_SYCL_GPU_VERSION)
      use sycl_functions
#endif
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
      use elpa_ccl_gpu
#endif

      implicit none
      class(elpa_impl_t), intent(inout)   :: self
      integer(kind=ik)                    :: error
      integer(kind=c_int)                 :: myid
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)

      logical                             :: useGPU
      logical                             :: success
      integer(kind=ik)                    :: numberOfDevices
      integer(kind=ik)                    :: deviceNumber, mpierr, maxNumberOfDevices
      logical                             :: gpuAvailable
      integer(kind=ik)                    ::  mpi_comm_all, use_gpu_id, min_use_gpu_id
      integer(kind=ik)                    :: maxThreads, thread
      integer(kind=c_int)                 :: cublas_version
      integer(kind=c_int)                 :: syclShowOnlyIntelGpus
      integer(kind=ik)                    :: syclShowAllDevices
      integer(kind=c_intptr_t)            :: handle_tmp
      character(len=8)                    :: fmt
      character(len=12)                   :: gpu_string
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
      TYPE(ncclUniqueId)                  :: ncclId
      integer(kind=c_int)                 :: nprocs
      integer(kind=c_intptr_t)            :: ccl_comm_all, ccl_comm_rows, ccl_comm_cols
      integer(kind=ik)                    :: myid_rows, myid_cols, mpi_comm_rows, mpi_comm_cols, nprows, npcols
#endif
#endif
      integer(kind=ik)                    :: attribute, value
      integer(kind=ik)                    :: debug, gpu
      logical                             :: wantDebugMessage
      error = ELPA_ERROR_SETUP

      gpu = 0
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)

#if defined(WITH_NVIDIA_GPU_VERSION)
      call self%get("nvidia-gpu",gpu, error)
      if (check_elpa_get(error, ELPA_ERROR_SETUP)) return
#endif
#if defined(WITH_AMD_GPU_VERSION)
      call self%get("amd-gpu",gpu, error)
      if (check_elpa_get(error, ELPA_ERROR_SETUP)) return
#endif
#if defined(WITH_SYCL_GPU_VERSION)
      call self%get("intel-gpu",gpu, error)
      if (check_elpa_get(error, ELPA_ERROR_SETUP)) return
#endif
      if (gpu .eq. 0) then
        write(error_unit,*) "ELPA_SETUP_GPU: no GPUs used. Leaving..."
      endif
#endif
      if (gpu .eq. 0) then
        error = ELPA_OK
        return
      endif

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    ! check legacy GPU setings
    if (.not.(query_gpu_usage(self, "ELPA_SETUP_GPU", useGPU))) then
      write(error_unit,*) "ELPA_SETUP_GPU: error when querying gpu settings. Aborting..."
      error = ELPA_ERROR_SETUP
      return
    endif
#endif

      if (self%is_set("debug") == 1) then
         call self%get("debug",debug, error)
         print *,"debug ",debug
         if (check_elpa_get(error, ELPA_ERROR_SETUP)) return
         if (debug .eq. 1) then
           wantDebugMessage = .true.
         endif
      endif
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
#undef OBJECT
#define OBJECT self
#undef ADDITIONAL_OBJECT_CODE

#include "./GPU/check_for_gpu_template.F90"
#undef OBJECT
#endif /* defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION) */

      error = ELPA_OK
      return
    end function


    !c> /*! \brief C interface for the implementation of the elpa_setup_gpu method
    !c> *
    !c> *  \param  elpa_t  handle of the ELPA object which describes the problem to
    !c> *                  be set up
    !c> *  \result int     error code, which can be queried with elpa_strerr
    !c> */
    !c> #include <stdint.h>
    !c> int elpa_setup_gpu(elpa_t handle);
    function elpa_setup_gpu_c(handle) result(error) bind(C, name="elpa_setup_gpu")
      implicit none
      type(c_ptr), intent(in), value :: handle
      type(elpa_impl_t), pointer     :: self
      integer(kind=c_int)            :: error

      call c_f_pointer(handle, self)
      error = self%setup_gpu()
    end function



    !> \brief function to setup an ELPA object and to store the MPI communicators internally
    !> Parameters
    !> \param   self       class(elpa_impl_t), the allocated ELPA object
    !> \result  error      integer, the error code
    function elpa_setup(self) result(error)
#ifdef WITH_MPI
      use elpa_scalapack_interfaces
#endif
      class(elpa_impl_t), intent(inout)   :: self
      integer                             :: error, timings, performance, build_config

      integer                             :: np_total, np_rows, np_cols, mpi_comm_parent, mpi_comm_cols, &
                                             mpi_comm_rows, my_id, process_row, process_col
#ifdef WITH_MPI
      integer                             :: mpi_string_length, &
                                             present_np_rows, present_np_cols, np_rows_tmp, np_cols_tmp
      integer(kind=MPI_KIND)              :: mpierr, mpierr2, my_idMPI, np_totalMPI, process_rowMPI, process_colMPI
      integer(kind=MPI_KIND)              :: mpi_comm_rowsMPI, mpi_comm_colsMPI, np_rowsMPI, np_colsMPI, &
                                             mpi_string_lengthMPI, my_pcolMPI, my_prowMPI, providedMPI
      character(len=MPI_MAX_ERROR_STRING) :: mpierr_string
      integer(kind=BLAS_KIND)             :: numroc_resultBLAS
      integer(kind=c_int)                 :: info, na, nblk, na_rows, my_pcol, my_prow, numroc_result
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

      self%mpi_setup%useMPI = .false.
#ifdef WITH_MPI
      self%mpi_setup%useMPI = .true.
#endif
      self%mpi_setup%mpi_comm_parent = -9999
      self%mpi_setup%mpi_comm_cols   = -9999
      self%mpi_setup%mpi_comm_rows   = -9999

      self%mpi_setup%mpi_comm_parentExternal = -9999
      self%mpi_setup%mpi_comm_colsExternal   = -9999
      self%mpi_setup%mpi_comm_rowsExternal   = -9999

      self%mpi_setup%nRanks_comm_parent = -9999
      self%mpi_setup%nRanks_comm_rows   = -9999
      self%mpi_setup%nRanks_comm_cols   = -9999

      self%mpi_setup%nRanksExternal_comm_parent = -9999
      self%mpi_setup%nRanksExternal_comm_rows   = -9999
      self%mpi_setup%nRanksExternal_comm_cols   = -9999

      self%mpi_setup%myRank_comm_parent = -9999
      self%mpi_setup%myRank_comm_rows   = -9999
      self%mpi_setup%myRank_comm_cols   = -9999

      self%mpi_setup%myRankExternal_comm_parent = -9999
      self%mpi_setup%myRankExternal_comm_rows   = -9999
      self%mpi_setup%myRankExternal_comm_cols   = -9999

      self%gpu_setup%gpuAlreadySet=.false.
      self%gpu_setup%gpuIsAssigned=.false.

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

        self%mpi_setup%mpi_comm_parent = mpi_comm_parent

        call mpi_comm_rank(int(mpi_comm_parent,kind=MPI_KIND), my_idMPI, mpierr)
        my_id = int(my_idMPI, kind=c_int)
        call self%set("process_id", my_id, error)
        if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
        self%mpi_setup%myRank_comm_parent = my_id

        call mpi_comm_size(int(mpi_comm_parent,kind=MPI_KIND), np_totalMPI, mpierr)
        np_total = int(np_totalMPI,kind=c_int)
        call self%set("num_processes", np_total, error)
        if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
        self%mpi_setup%nRanks_comm_parent = np_total
      else ! mpi_comm_parent == 1
        if (self%from_legacy_api .ne. 1) then
          write(error_unit,*) MPI_CONSISTENCY_MSG
          error = ELPA_ERROR
          return
        endif
      endif ! mpi_comm_parent == 1

#if defined(WITH_OPENMP_TRADITIONAL) && defined(THREADING_SUPPORT_CHECK) && !defined(HAVE_SUFFICIENT_MPI_THREADING_SUPPORT)
      ! check the threading level supported by the MPI library
      call mpi_query_thread(providedMPI, mpierr)
      if ((providedMPI .ne. MPI_THREAD_SERIALIZED) .and. (providedMPI .ne. MPI_THREAD_MULTIPLE)) then
#if defined(ALLOW_THREAD_LIMITING)
        write(error_unit,*) "WARNING elpa_setup: MPI threading level MPI_THREAD_SERALIZED or MPI_THREAD_MULTIPLE required but &
                            &your implementation does not support this! The number of OpenMP threads within ELPA will be &
                            &limited to 1"
        call self%set("limit_openmp_threads", 1, error)
        if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
#else
        write(error_unit,*) "WARNING elpa_setup: MPI threading level MPI_THREAD_SERALIZED or MPI_THREAD_MULTIPLE required but &
                            &your implementation does not support this! Since you did not build ELPA with &
                            &--enable-allow-thread-limiting, this is severe warning. ELPA will _not_ try to cure this problem and&
                            &the results might be wrong. USE AT YOUR OWN RISK !"
#endif
      endif

#endif

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
          error = ELPA_ERROR_SETUP
          return
        endif

        call mpi_comm_split(int(mpi_comm_parent,kind=MPI_KIND), int(process_row,kind=MPI_KIND), &
                            int(process_col,kind=MPI_KIND), mpi_comm_colsMPI, mpierr)
        mpi_comm_cols = int(mpi_comm_colsMPI,kind=c_int)
        if (mpierr .ne. MPI_SUCCESS) then
          call MPI_ERROR_STRING(mpierr, mpierr_string, mpi_string_lengthMPI, mpierr2)
          mpi_string_length = int(mpi_string_lengthMPI, kind=c_int)
          write(error_unit,*) "MPI ERROR occured during mpi_comm_split for col communicator: ", trim(mpierr_string)
          error = ELPA_ERROR_SETUP
          return
        endif

!        ! get the sizes and return maybe an error
!#ifdef WITH_MPI
!        call mpi_comm_size(mpi_comm_colsMPI, np_colsMPI, mpierr)
!        np_cols_tmp = int(np_colsMPI)
!        if (np_cols_tmp .eq. 1) then
!          write(error_unit,*) "ELPA_SETUP: ERROR you cannot use ELPA with 1 process col "
!          error = ELPA_ERROR_SETUP
!          return
!        endif
!
!        call mpi_comm_size(mpi_comm_rowsMPI, np_rowsMPI, mpierr)
!        np_rows_tmp = int(np_rowsMPI)
!        if (np_rows_tmp .eq. 1) then
!          write(error_unit,*) "ELPA_SETUP: ERROR you cannot use ELPA with 1 process row "
!          error = ELPA_ERROR_SETUP
!          return
!        endif
!#endif

        call self%set("mpi_comm_rows", mpi_comm_rows, error)
        if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
        self%mpi_setup%mpi_comm_rows = mpi_comm_rows


        call self%set("mpi_comm_cols", mpi_comm_cols, error)
        if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
        self%mpi_setup%mpi_comm_cols = mpi_comm_cols

        ! remember that we created those communicators and we need to free them later
        self%communicators_owned = 1

      ! Externally supplied communicators
      else if ( self%is_set("mpi_comm_rows") == 1 .and.  self%is_set("mpi_comm_cols") == 1) then

        if (self%is_set("process_row") == 1 .or. self%is_set("process_col") == 1) then
          write(error_unit,*) MPI_CONSISTENCY_MSG
          error = ELPA_ERROR
          return
        endif

        call self%get("mpi_comm_rows", mpi_comm_rows, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return
        self%mpi_setup%mpi_comm_rows = mpi_comm_rows

        call self%get("mpi_comm_cols", mpi_comm_cols, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return
        self%mpi_setup%mpi_comm_cols = mpi_comm_cols

!        ! get the sizes and return maybe an error
!#ifdef WITH_MPI
!        call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND), np_colsMPI, mpierr)
!        np_cols_tmp = int(np_colsMPI)
!        if (np_cols_tmp .eq. 1) then
!          write(error_unit,*) "ELPA_SETUP: ERROR you cannot use ELPA with 1 process col "
!          error = ELPA_ERROR_SETUP
!          return
!        endif
!
!        call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)
!        np_rows_tmp = int(np_rowsMPI)
!        if (np_rows_tmp .eq. 1) then
!          write(error_unit,*) "ELPA_SETUP: ERROR you cannot use ELPA with 1 process row "
!          error = ELPA_ERROR_SETUP
!          return
!        endif
!#endif
        call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND), process_rowMPI, mpierr)
        process_row = int(process_rowMPI,kind=c_int)
        call self%set("process_row", process_row, error)
        if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
        self%mpi_setup%myRank_comm_rows = process_row

        call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND), process_colMPI, mpierr)
        process_col = int(process_colMPI,kind=c_int)
        call self%set("process_col", process_col, error)
        if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
        self%mpi_setup%myRank_comm_cols = process_col


        ! remember that we DID NOT created those communicators and we WILL NOT free them later
        self%communicators_owned = 0
      else ! (self%is_set("process_row") == 1 .and. self%is_set("process_col") == 1) then
        ! Otherwise parameters are missing
        write(error_unit,*) MPI_CONSISTENCY_MSG
        error = ELPA_ERROR
        return
      endif ! (self%is_set("process_row") == 1 .and. self%is_set("process_col") == 1) then

      ! set num_process_rows (and cols), if they are not supplied. Check them
      ! for consistency if they are. Maybe we could instead require, that they
      ! are never supplied?
      call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)
      np_rows = int(np_rowsMPI, kind=c_int)
      if (self%is_set("num_process_rows") == 1) then
        call self%get("num_process_rows", present_np_rows, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return

        if (np_rows .ne. present_np_rows) then
          write(error_unit, '(a)') "ELPA_SETUP: MPI row communicator not set correctly. Aborting..."
          error = ELPA_ERROR_SETUP
          return   
        endif

!#ifdef WITH_MPI
!        if (np_rows .eq. 1) then
!          write(error_unit,*) "ELPA_SETUP: ERROR you cannot use ELPA with 1 process row "
!          error = ELPA_ERROR_SETUP
!          return
!        endif
!#endif       
      else ! self%is_set("num_process_rows") == 1
        call self%set("num_process_rows", np_rows, error)
        if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
      endif ! self%is_set("num_process_rows") == 1
      self%mpi_setup%nRanks_comm_rows = np_rows

      call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND), np_colsMPI, mpierr)
      np_cols = int(np_colsMPI, kind=c_int)
      if (self%is_set("num_process_cols") == 1) then
        call self%get("num_process_cols", present_np_cols, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return

        if (np_cols .ne. present_np_cols) then
          write(error_unit, '(a)') "ELPA_SETUP: MPI row communicator not set correctly. Aborting..."
          error = ELPA_ERROR_SETUP
          return   
        endif
!#ifdef WITH_MPI
!        if (np_cols .eq. 1) then
!          write(error_unit,*) "ELPA_SETUP: ERROR you cannot use ELPA with 1 process row "
!          error = ELPA_ERROR_SETUP
!          return
!        endif
!#endif       
      else
        call self%set("num_process_cols", np_cols, error)
        if (check_elpa_set(error, ELPA_ERROR_SETUP)) return
      endif
      self%mpi_setup%nRanks_comm_cols = np_cols

      if (self%from_legacy_api .ne. 1) then
        if (np_total .ne. np_rows * np_cols) then
          write(error_unit, '(a)') "ELPA_SETUP: MPI parent communicator and row/col communicators do not match. Aborting..."
          error = ELPA_ERROR_SETUP
          return   
        endif
      endif

      ! check first whether BLACS-GRID, which was setup by the user is reasonable. Too often this is _not_ done by the
      ! user and then there are complaints about "errors" in ELPA but the problem is in a misconfigured setup
      call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND) ,my_prowMPI ,mpierr)
      call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND) ,my_pcolMPI ,mpierr)

      my_prow = int(my_prowMPI, kind=c_int)
      my_pcol = int(my_pcolMPI, kind=c_int)
      self%mpi_setup%myRank_comm_rows = my_prow
      self%mpi_setup%myRank_comm_cols = my_pcol

      call self%get("na", na, error)
      if (check_elpa_set(error, ELPA_ERROR_SETUP)) return

      call self%get("nblk", nblk, error)
      if (check_elpa_set(error, ELPA_ERROR_SETUP)) return

      call self%get("local_nrows", na_rows, error)
      if (check_elpa_set(error, ELPA_ERROR_SETUP)) return

      numroc_resultBLAS = numroc(int(na, kind=BLAS_KIND), int(nblk, kind=BLAS_KIND), int(my_prow, kind=BLAS_KIND), &
                                 0_BLAS_KIND, int(np_rows, kind=BLAS_KIND))
      numroc_result=int(numroc_resultBLAS, kind=c_int)
      info = 0
      if ( na < 0 ) then
        info = -2
      else if ( nblk < 1 ) then
        info = -4
      else if ( np_rows .eq. -1 ) then
        info = -8
      else if ( na_rows < max( 1, numroc_result ) ) then
        info = -9
      endif

      if (info .ne. 0) then
        write(error_unit,*) "ELPA_SETUP ERROR: your provided blacsgrid is not ok!"
        write(error_unit,*) "BLACS_GRIDINFO returned an error! Aborting..."
        error = ELPA_ERROR_SETUP
        return
      endif


#else /* WITH_MPI */
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
#endif /* WITH_MPI */

#ifdef WITH_MPI
      self%myGlobalId = my_id
#else
      self%myGlobalId = 0
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
      ! Check whether all the mpi_setup variables have been set
      if (self%mpi_setup%mpi_comm_parent .eq. -9999) then
        call self%get("mpi_comm_parent", mpi_comm_parent, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return

        self%mpi_setup%mpi_comm_parent = mpi_comm_parent
      endif
      if (self%mpi_setup%mpi_comm_rows .eq. -9999) then
        call self%get("mpi_comm_rows", mpi_comm_rows, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return

        self%mpi_setup%mpi_comm_rows = mpi_comm_rows
      endif
      if (self%mpi_setup%mpi_comm_cols .eq. -9999) then
        call self%get("mpi_comm_rows", mpi_comm_cols, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return

        self%mpi_setup%mpi_comm_rows = mpi_comm_cols
      endif

      if (self%mpi_setup%myRank_comm_parent .eq. -9999) then
        call self%get("process_id", my_id, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return
        self%mpi_setup%myRank_comm_parent = my_id
      endif
      if (self%mpi_setup%myRank_comm_rows .eq. -9999) then
        call self%get("process_row", process_row, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return
        self%mpi_setup%myRank_comm_rows = process_row
      endif
      if (self%mpi_setup%myRank_comm_cols .eq. -9999) then
        call self%get("process_col", process_col, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return
        self%mpi_setup%myRank_comm_cols = process_col
      endif
#ifdef WITH_MPI
      if (self%mpi_setup%nRanks_comm_parent .eq. -9999) then
        call self%get("mpi_comm_parent", mpi_comm_parent, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return
        call mpi_comm_size(int(mpi_comm_parent,kind=MPI_KIND), np_totalMPI, mpierr)
        self%mpi_setup%nRanks_comm_parent = int(np_totalMPI,kind=c_int)
      endif
      if (self%mpi_setup%nRanks_comm_rows .eq. -9999) then
        call self%get("mpi_comm_rows", mpi_comm_rows, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return
        call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)
        self%mpi_setup%nRanks_comm_rows = int(np_rowsMPI,kind=c_int)
      endif
      if (self%mpi_setup%nRanks_comm_cols .eq. -9999) then
        call self%get("mpi_comm_cols", mpi_comm_cols, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return
        call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND), np_colsMPI, mpierr)
        self%mpi_setup%nRanks_comm_cols = int(np_colsMPI,kind=c_int)
      endif
#else /* WITH_MPI */
      if (self%mpi_setup%nRanks_comm_parent .eq. -9999) then
        call self%get("num_processes", np_total, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return
        self%mpi_setup%nRanks_comm_parent = np_total
      endif
      if (self%mpi_setup%nRanks_comm_rows .eq. -9999) then
        call self%get("num_process_rows", np_rows, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return
        self%mpi_setup%nRanks_comm_rows = np_rows
      endif
      if (self%mpi_setup%nRanks_comm_cols .eq. -9999) then
        call self%get("num_process_cols", np_cols, error)
        if (check_elpa_get(error, ELPA_ERROR_SETUP)) return
        self%mpi_setup%nRanks_comm_cols = np_cols
      endif
#endif /* WITH_MPI */


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
        write(error_unit,*) "BLACS context has not been set beforehand. Aborting..."
        error = ELPA_ERROR
        return
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


    !c> /*! \brief C interface for the implementation of the elpa_set_float method
    !c> *  This method is available to the user as C generic elpa_set method
    !c> *
    !c> *  \param  handle  handle of the ELPA object for which a key/value pair should be set
    !c> *  \param  name    the name of the key
    !c> *  \param  value   the value to be set for the key
    !c> *  \param  error   on return the error code, which can be queried with elpa_strerr()
    !c> *  \result void
    !c> */
    !c> void elpa_set_float(elpa_t handle, const char *name, float value, int *error);
    subroutine elpa_set_float_c(handle, name_p, value, error) bind(C, name="elpa_set_float")
      type(c_ptr), intent(in), value                :: handle
      type(elpa_impl_t), pointer                    :: self
      type(c_ptr), intent(in), value                :: name_p
      character(len=elpa_strlen_c(name_p)), pointer :: name
      real(kind=c_float), intent(in), value        :: value
      integer(kind=c_int), intent(in)               :: error

      call c_f_pointer(handle, self)
      call c_f_pointer(name_p, name)
      call elpa_set_float(self, name, value, error)
    end subroutine


    !c> /*! \brief C interface for the implementation of the elpa_get_float method
    !c> *  This method is available to the user as C generic elpa_get method
    !c> *
    !c> *  \param  handle  handle of the ELPA object for which a key/value pair should be queried
    !c> *  \param  name    the name of the key
    !c> *  \param  value   the value to be obtain for the key
    !c> *  \param  error   on return the error code, which can be queried with elpa_strerr()
    !c> *  \result void
    !c> */
    !c> void elpa_get_float(elpa_t handle, const char *name, float *value, int *error);
    subroutine elpa_get_float_c(handle, name_p, value, error) bind(C, name="elpa_get_float")
      type(c_ptr), intent(in), value                :: handle
      type(elpa_impl_t), pointer                    :: self
      type(c_ptr), intent(in), value                :: name_p
      character(len=elpa_strlen_c(name_p)), pointer :: name
      real(kind=c_float)                           :: value
      integer(kind=c_int), intent(inout)            :: error

      call c_f_pointer(handle, self)
      call c_f_pointer(name_p, name)
      call elpa_get_float(self, name, value, error)
    end subroutine


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

    !c> /*! \brief C interface for the implementation of the elpa_print_times method
    !c> *
    !c> *  \param elpa_t handle
    !c> *  \param  char* name1
    !c> */
    !c> void elpa_print_times(elpa_t handle, char* name1);
    subroutine elpa_print_times_c(handle, name1_p) bind(C, name="elpa_print_times")
      type(c_ptr), value         :: handle
      type(elpa_impl_t), pointer :: self
      type(c_ptr), intent(in), value       :: name1_p
      character(len=elpa_strlen_c(name1_p)), pointer :: name1

      call c_f_pointer(handle, self)
      call c_f_pointer(name1_p, name1)

      call elpa_print_times(self, name1)

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

    !c> /*! \brief C interface for the implementation of the elpa_timer_start method
    !c> *
    !c> *  \param elpa_t handle
    !c> *  \param  char* name
    !c> */
    !c> void elpa_timer_start(elpa_t handle, char* name);
    subroutine elpa_timer_start_c(handle, name_p) bind(C, name="elpa_timer_start")
      type(c_ptr), value         :: handle
      type(elpa_impl_t), pointer :: self
      type(c_ptr), intent(in), value       :: name_p
      character(len=elpa_strlen_c(name_p)), pointer :: name

      call c_f_pointer(handle, self)
      call c_f_pointer(name_p, name)

      call elpa_timer_start(self, name)

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

    !c> /*! \brief C interface for the implementation of the elpa_timer_stop method
    !c> *
    !c> *  \param elpa_t handle
    !c> *  \param  char* name
    !c> */
    !c> void elpa_timer_stop(elpa_t handle, char* name);
    subroutine elpa_timer_stop_c(handle, name_p) bind(C, name="elpa_timer_stop")
      type(c_ptr), value         :: handle
      type(elpa_impl_t), pointer :: self
      type(c_ptr), intent(in), value       :: name_p
      character(len=elpa_strlen_c(name_p)), pointer :: name

      call c_f_pointer(handle, self)
      call c_f_pointer(name_p, name)

      call elpa_timer_stop(self, name)

    end subroutine

    !> \brief function to destroy an elpa object
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    !> \param   error           integer, optional error code
    subroutine elpa_destroy(self, error)
      use elpa_gpu_setup
      use elpa_gpu
#ifdef WITH_OPENMP_TRADITIONAL
      use elpa_omp
#endif
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif

      implicit none
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
      integer                              :: error2, istat, maxThreads, thread
      logical                              :: success
      character(200)                       :: errorMessage

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
!        call self%set("mpi_comm_cols", -12345, error2)
!        if (error2 .ne. ELPA_OK) then
!#ifdef USE_FORTRAN2008
!          if (present(error)) then
!            error = error2
!          else
!            write(error_unit, *) "Error in elpa_destroy but you do not check the error codes!"
!          endif
!#else
!          error = error2
!#endif
!          return
!        endif ! error happend
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
!        call self%set("mpi_comm_rows", -12345,error2)
!        if (error2 .ne. ELPA_OK) then
!#ifdef USE_FORTRAN2008
!          if (present(error)) then
!            error = error2
!          else
!            write(error_unit, *) "Error in elpa_destroy but you do not check the error codes!"
!          endif
!#else
!          error = error2
!#endif
!          return
!        endif ! error happend
      endif
#endif /* WITH_MPI */

      ! cleanup GPU allocations
      ! needed for handle destruction
#ifdef WITH_OPENMP_TRADITIONAL
      maxThreads=omp_get_max_threads()
#else /* WITH_OPENMP_TRADITIONAL */
      maxThreads=1
#endif /* WITH_OPENMP_TRADITIONAL */

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      if (allocated(self%gpu_setup%gpublasHandleArray)) then

#include "./GPU/handle_destruction_template.F90"

        deallocate(self%gpu_setup%gpublasHandleArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate gpublasHandleArray: " // errorMessage
        endif 
        deallocate(self%gpu_setup%gpuDeviceArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate gpuDeviceArray: " // errorMessage
        endif 

#ifdef WITH_NVIDIA_GPU_VERSION
        deallocate(self%gpu_setup%cublasHandleArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate cublasHandleArray: " // errorMessage
        endif 
        deallocate(self%gpu_setup%cudaDeviceArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate cudaDeviceArray: " // errorMessage
        endif 
#endif

#ifdef WITH_AMD_GPU_VERSION
        deallocate(self%gpu_setup%rocblasHandleArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate rocblasHandleArray: " // errorMessage
        endif 
        deallocate(self%gpu_setup%hipDeviceArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate hipDeviceArray: " // errorMessage
        endif 
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        deallocate(self%gpu_setup%openmpOffloadHandleArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate openmpOffloadHandleArray: " // errorMessage
        endif 
        deallocate(self%gpu_setup%openmpOffloadDeviceArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate openmpOffloadDeviceArray: " // errorMessage
        endif 
#endif

#ifdef WITH_SYCL_GPU_VERSION
        deallocate(self%gpu_setup%syclHandleArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate syclHandleArray: " // errorMessage
        endif 
        deallocate(self%gpu_setup%syclDeviceArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate syclDeviceArray: " // errorMessage
        endif 
#endif

#if defined(WITH_NVIDIA_GPU_VERSION) && defined(WITH_NVIDIA_CUSOLVER)
        deallocate(self%gpu_setup%gpusolverHandleArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate gpusolverHandleArray: " // errorMessage
        endif 
        deallocate(self%gpu_setup%cusolverHandleArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate cusolverHandleArray: " // errorMessage
        endif 
#endif

#if defined(WITH_AMD_GPU_VERSION) && defined(WITH_AMD_ROCSOLVER)
        deallocate(self%gpu_setup%gpusolverHandleArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate gpusolverHandleArray: " // errorMessage
        endif 
        deallocate(self%gpu_setup%rocsolverHandleArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate rocsolverHandleArray: " // errorMessage
        endif 
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) && defined(WITH_OPENMP_OFFLOAD_SOLVER)
        deallocate(self%gpu_setup%gpusolverHandleArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate gpusolverHandleArray: " // errorMessage
        endif 
        deallocate(self%gpu_setup%openmpOffloadsolverHandleArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate openmpOffloadsolverHandleArray: " // errorMessage
        endif 
#endif

#if defined(WITH_SYCL_GPU_VERSION) && defined(WITH_SYCL_SOLVER)
        deallocate(self%gpu_setup%gpusolverHandleArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate gpusolverHandleArray: " // errorMessage
        endif 
        deallocate(self%gpu_setup%syclsolverHandleArray, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          write(error_unit, "(a,i0,a)") "ELPA: elpa_destroy cannot deallocate openmpOffloadsolverHandleArray: " // errorMessage
        endif 
#endif



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
#include "elpa_impl_math_solvers_template.F90"
#include "elpa_impl_math_generalized_template.F90"
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
#include "elpa_impl_math_solvers_template.F90"
#include "elpa_impl_math_generalized_template.F90"
#undef REALCASE
#undef SINGLE_PRECISION
#undef INCLUDE_ROUTINES

#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#define INCLUDE_ROUTINES 1
#include "general/precision_macros.h"
#include "elpa_impl_math_template.F90"
#include "elpa_impl_math_solvers_template.F90"
#include "elpa_impl_math_generalized_template.F90"
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
#include "elpa_impl_math_solvers_template.F90"
#include "elpa_impl_math_generalized_template.F90"
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
    !> \brief procedure to set the api version used for ELPA autotuning
    !> Parameters
    !> \param   self            the allocated ELPA object
    !> \param   api_version     integer: the api_version
    !> \param   error           integer: error code
    subroutine elpa_autotune_set_api_version(self, api_version, error)
      class(elpa_impl_t), intent(inout), target :: self
      integer, intent(in)                       :: api_version
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional             :: error
#else
      integer(kind=c_int)                       :: error
#endif

      autotune_api_set = .true.

      if (api_version .ge. 20211125) then
        new_autotune = .true.
      else
        new_autotune = .false.
      endif

#ifdef USE_FORTRAN2008
      if (present(error)) then
        error = ELPA_OK
      endif
#else
      error = ELPA_OK
#endif

    end subroutine

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
      integer(kind=c_int)                       :: sublevel, solver
      integer(kind=c_int)                       :: gpu_old, gpu_new

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
      if (new_autotune) then
        ts_impl%new_stepping = 1
      else
        ts_impl%new_stepping = 0
      endif

      ts_impl%parent => self
      ts_impl%level = level
      ts_impl%domain = domain

      ts_impl%current = -1
      ts_impl%min_loc = -1

      if (ts_impl%new_stepping == 1) then
        if (self%is_set("solver") == 1) then
          call self%get("solver", solver, error)
          if (solver == ELPA_SOLVER_2STAGE) then
                  ! why ?
            ts_impl%sublevel_part1stage(level) = ELPA_AUTOTUNE_PART_ELPA1
            ts_impl%sublevel_part2stage(level) = ELPA_AUTOTUNE_PART_ELPA2
          else if (solver == ELPA_SOLVER_1STAGE) then
            ts_impl%sublevel_part1stage(level) = ELPA_AUTOTUNE_PART_ELPA1
            ts_impl%sublevel_part2stage(level) = ELPA_AUTOTUNE_PART_ELPA2
          else
            write(error_unit,*)  "ELPA_AUTOTUNE_SETUP: Unknown solver"
#ifdef USE_FORTRAN2008
            if (present(error)) then
              error = ELPA_ERROR
              return
            else
              return
            endif
#else
            error = ELPA_ERROR
            return
#endif
          endif        
        else ! no solver set, anyways ...
          ts_impl%sublevel_part1stage(level) = ELPA_AUTOTUNE_PART_ELPA1
          ts_impl%sublevel_part2stage(level) = ELPA_AUTOTUNE_PART_ELPA2
        endif

        !ts_impl%cardinality = elpa_index_autotune_cardinality_new_stepping_c(self%index, level, domain, &
        !                                                                     ts_impl%sublevel_part(level))

        !ts_impl%cardinality = elpa_index_autotune_cardinality_new_stepping_c(self%index, level, domain, &
        !                                                                     ts_impl%sublevel_part(level))
      else ! new_stepping
        ts_impl%cardinality = elpa_index_autotune_cardinality_c(self%index, level, domain)
      endif ! new_stepping

      if (ts_impl%new_stepping == 1) then
        autotune_level  = level
        autotune_domain = domain
        ts_impl%sublevel_current1stage(:) = -1
        ts_impl%total_current_1stage = -1
        ts_impl%sublevel_current2stage(:) = -1
        ts_impl%total_current_2stage = -1
        ts_impl%sublevel_cardinality1stage(:) = 0
        ts_impl%sublevel_cardinality2stage(:) = 0
        ts_impl%sublevel_min_val1stage(:) = 0.0_C_DOUBLE
        ts_impl%sublevel_min_val2stage(:) = 0.0_C_DOUBLE
        ts_impl%sublevel_min_loc1stage(:) = -1
        ts_impl%sublevel_min_loc2stage(:) = -1

        ! get cardinality of all sublevels
        do sublevel=1, autotune_level
          if (self%is_set("solver") == 1) then
            call self%get("solver", solver, error)
            if (solver == ELPA_SOLVER_2STAGE) then
              ts_impl%sublevel_part1stage(sublevel) = ELPA_AUTOTUNE_PART_ELPA1
              ts_impl%sublevel_part2stage(sublevel) = ELPA_AUTOTUNE_PART_ELPA2
            else if (solver == ELPA_SOLVER_1STAGE) then
              ts_impl%sublevel_part1stage(sublevel) = ELPA_AUTOTUNE_PART_ELPA1
              ts_impl%sublevel_part2stage(sublevel) = ELPA_AUTOTUNE_PART_ELPA2
            else
              write(error_unit,*)  "ELPA_AUTOTUNE_SETUP: Unknown solver"
#ifdef USE_FORTRAN2008
              if (present(error)) then
                error = ELPA_ERROR
                return
              else
                return
              endif
#else
              error = ELPA_ERROR
              return
#endif
            endif        
          else 
            ts_impl%sublevel_part1stage(sublevel) = ELPA_AUTOTUNE_PART_ELPA1
            ts_impl%sublevel_part2stage(sublevel) = ELPA_AUTOTUNE_PART_ELPA2
          endif

          if (sublevel .eq. 1) then
            ts_impl%sublevel_cardinality1stage(sublevel) = elpa_index_autotune_cardinality_new_stepping_c&
                                                           (self%index, sublevel, ts_impl%domain, &
                                                           ts_impl%sublevel_part1stage(level))

            ts_impl%sublevel_cardinality2stage(sublevel) = elpa_index_autotune_cardinality_new_stepping_c&
                                                           (self%index, sublevel, ts_impl%domain, &
                                                           ts_impl%sublevel_part2stage(level))
          else
            ts_impl%sublevel_cardinality1stage(sublevel) = elpa_index_autotune_cardinality_new_stepping_c&
                                                           (self%index, sublevel, ts_impl%domain, &
                                                           ts_impl%sublevel_part1stage(level)) &
                                                           - sum(ts_impl%sublevel_cardinality1stage(1:sublevel-1))

            ts_impl%sublevel_cardinality2stage(sublevel) = elpa_index_autotune_cardinality_new_stepping_c&
                                                           (self%index, sublevel, ts_impl%domain, &
                                                           ts_impl%sublevel_part2stage(level)) &
                                                           - sum(ts_impl%sublevel_cardinality2stage(1:sublevel-1))
          endif
    
          if (ts_impl%sublevel_cardinality1stage(sublevel) .eq. 0) then
            autotune_substeps_done1stage(sublevel) = .true.
          endif
          if (ts_impl%sublevel_cardinality2stage(sublevel) .eq. 0) then
            autotune_substeps_done2stage(sublevel) = .true.
          endif

        enddo
      endif ! new-stepping

      tune_state => ts_impl

      ! check consistency between "gpu" and "nvidia-gpu"
      if (self%is_set("gpu") == 1) then
        call self%get("gpu", gpu_old, error)
        if (error .ne. ELPA_OK) then
          write(error_unit,*) "ELPA_AUTOTUNE_SETUP: cannot get gpu option. Aborting..."
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = ELPA_ERROR
            return
          else
            return
          endif
#else
          error = ELPA_ERROR
          return
#endif
        endif
        if (self%is_set("nvidia-gpu") == 1) then
         call self%get("nvidia-gpu", gpu_new, error)
         if (error .ne. ELPA_OK) then
           write(error_unit,*) "ELPA_AUTOTUNE_SETUP: cannot get nvidia-gpu option. Aborting..."
#ifdef USE_FORTRAN2008
           if (present(error)) then
             error = ELPA_ERROR
             return
           else
             return
           endif
#else
           error = ELPA_ERROR
           return
#endif
         endif
         if (gpu_old .ne. gpu_new) then
           write(error_unit,*) "ELPA_AUTOTUNE_SETUP: you cannot set gpu =",gpu_old," and nvidia-gpu =",gpu_new," Aborting..."
#ifdef USE_FORTRAN2008
           if (present(error)) then
             error = ELPA_ERROR
             return
           else
             return
           endif
#else
           error = ELPA_ERROR
           return
#endif
         endif
        else ! nvidia-gpu not set
          call self%set("nvidia-gpu", gpu_old, error)
          if (error .ne. ELPA_OK) then
            write(error_unit,*) "ELPA_AUTOTUNE_SETUP: cannot set nvidia-gpu option. Aborting..."
#ifdef USE_FORTRAN2008
           if (present(error)) then
             error = ELPA_ERROR
             return
           else
             return
           endif
#else
           error = ELPA_ERROR
           return
#endif
          endif
        endif ! nvidia-gpu
      else ! gpu not set
        ! nothing to do
        !if (self%is_set("nvidia-gpu") == 1) then
         !call self%get("nvidia-gpu", gpu_new, error)
         !if (error .ne. ELPA_OK) then
         !  print *,"ELPA_AUTOTUNE_SETUP: cannot get nvidia-gpu option. Aborting..."
         !  stop 1
         !endif
         !call self%set("gpu", gpu_new, error)
         !if (error .ne. ELPA_OK) then
         !  print *,"ELPA_AUTOTUNE_SETUP: cannot set gpu option. Aborting..."
         !  stop 1
         !endif
        !else !nvidia not set
        !endif
      endif ! gpu set

      call self%autotune_timer%enable()
    end function

    !c> /*! \brief C interface for the implementation of the elpa_autotune_set_api_version method
    !c> *
    !c> *  \param  elpa_t           handle: of the ELPA object which should be tuned
    !c> *  \param  int              api_version: the version used for autotuning
    !c> */
    !c> void elpa_autotune_set_api_version(elpa_t handle, int api_version, int *error);
    subroutine elpa_autotune_set_api_version_c(handle, api_version, error) bind(C, name="elpa_autotune_set_api_version")
      type(c_ptr), intent(in), value         :: handle
      type(elpa_impl_t), pointer             :: self
      integer(kind=c_int), intent(in), value :: api_version
      integer(kind=c_int) , intent(in)       :: error

      call c_f_pointer(handle, self)

      call self%autotune_set_api_version(api_version, error)

    end subroutine


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
      integer(kind=c_int) , intent(inout)    :: error

      call c_f_pointer(handle, self)

      tune_state => self%autotune_setup(level, domain, error)
      select type(tune_state)
        type is (elpa_autotune_impl_t)
          obj => tune_state
        class default
        write(error_unit,*) "ELPA_AUTOTUNE_SETUP_C ERROR: This should not happen"
        error = ELPA_ERROR_SETUP
        return
      end select
      ptr = c_loc(obj)

    end function

    !> \brief function to do the work an autotunig step
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
      logical                                       :: unfinished
      logical                                       :: unfinished_1stage, unfinished_2stage, compare_solvers
      logical, save                                 :: firstCall = .true.
      integer(kind=c_int)                           :: solver, debug

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
          write(error_unit,*) "ELPA_AUTOTUNE_STEP ERROR: This should not happen"
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = ELPA_ERROR
          endif
#else
          error = ELPA_ERROR
#endif
      end select

      unfinished        = .false.
      call self%get("debug", debug, error)
      if (error .ne. ELPA_OK) then
        write(error_unit,*) "ELPA_AUTOTUNE_STEP: cannot get debug option. Aborting..."
#ifdef USE_FORTRAN2008
        if (present(error)) then
          error = ELPA_ERROR
          return
        else
          return
        endif
#else
        error = ELPA_ERROR_SETUP
        return
#endif
      endif

      if (ts_impl%new_stepping == 1) then
        ! check whether the user fixed the solver. 
        ! if not this routine will add a loop over the solvers, such
        ! that first with fixed SOLVER_1STAGE the tuning is done,
        ! then with SOLVER_2STAGE, and last the best of both tuning results is
        ! chosen
        ! we can only check this at the first call since afterwards we
        ! set the solver :-)
        if (firstCall) then
          if (self%is_set("solver") == 1) then
            call self%get("solver", solver, error)
            if (error .ne. ELPA_OK) then
              write(error_unit,*)  "ELPA_AUTOTUNE_STEP: cannot get solver. Aborting..."
#ifdef USE_FORTRAN2008
              if (present(error)) then
                error = ELPA_ERROR
                return
              else
                return
              endif
#else
              error = ELPA_ERROR_SETUP
              return
#endif
            endif
            if (solver == ELPA_SOLVER_2STAGE) then
              do_autotune_2stage = .true.
              do_autotune_1stage = .false.
              compare_solvers    = .false.
            else if (solver == ELPA_SOLVER_1STAGE) then
              do_autotune_2stage = .false.
              do_autotune_1stage = .true.
              compare_solvers    = .false.
            else
              write(error_unit,*) "ELPA_AUTOTUNE_STEP: Unknown solver"
#ifdef USE_FORTRAN2008
              if (present(error)) then
                error = ELPA_ERROR
                return
              else
                return
              endif
#else
              error = ELPA_ERROR_SETUP
              return
#endif
            endif        
          else 
            do_autotune_2stage = .true.
            do_autotune_1stage = .true.
            compare_solvers    = .true.
          endif
          firstCall = .false.
        endif

        if (do_autotune_1stage) then
          consider_solver = ELPA_SOLVER_1STAGE
          call self%set("solver", ELPA_SOLVER_1STAGE, error)
          if (error .ne. ELPA_OK) then
            write(error_unit,*) "ELPA_AUTOTUNE_STEP: cannot set ELPA_SOLVER_1STAGE for tuning"
            return
          endif
          unfinished_1stage = self%autotune_step_worker(tune_state, ELPA_SOLVER_1STAGE, error)
          if (unfinished_1stage) then
            !if (debug == 1) print *,"Tuning for ELPA_SOLVER_1STAGE NOT DONE"
            unfinished = unfinished_1stage
            return
          else
            !print *,"Tuning for ELPA_SOLVER_1STAGE DONE",do_autotune_2stage
            do_autotune_1stage = .false.
            last_call_1stage = .true.
            if (do_autotune_2stage) then
              if (self%myGlobalId .eq. 0) write(error_unit, "(a)") "Tuning of ELPA 1stage done: Doing one last call for 1stage"
              unfinished = .true.
              return
            else
              unfinished = unfinished_1stage
              ts_impl%best_solver = ELPA_SOLVER_1STAGE
              return
            endif
          endif
        endif

        if (do_autotune_2stage) then
          consider_solver = ELPA_SOLVER_2STAGE
          call self%set("solver", ELPA_SOLVER_2STAGE, error)
          if (error .ne. ELPA_OK) then
            write(error_unit,*) "ELPA_AUTOTUNE_STEP: cannot set ELPA_SOLVER_2STAGE for tuning"
            return
          endif
          unfinished_2stage = self%autotune_step_worker(tune_state, ELPA_SOLVER_2STAGE, error)
          if (unfinished_2stage) then
            !if (debug == 1) print *,"Tuning for ELPA_SOLVER_2STAGE NOT DONE"
            unfinished = unfinished_2stage
            return
          else
            if (self%myGlobalId .eq. 0) write(error_unit, "(a)") "Tuning of ELPA 2stage done"
            !if (debug == 1) print *,"Tuning for ELPA_SOLVER_2STAGE DONE"
            do_autotune_2stage = .false.
            last_call_2stage = .true.
            if (do_autotune_1stage) then
              ! this case should never be possible
              write(error_unit,*) "PANIC in elpa_autotune_step. Aborting!"
              unfinished = .true.
              return
            else
              if (.not.(unfinished_2stage) .and. (compare_solvers)) then
                ! we are done compare which solver was the best
                if (ts_impl%best_val1stage .lt. ts_impl%best_val2stage) then
                  ts_impl%best_solver = ELPA_SOLVER_1STAGE
                else
                  ts_impl%best_solver = ELPA_SOLVER_2STAGE
                endif
              endif
              unfinished = unfinished_2stage
              return
            endif
          endif
        endif
      else ! new_stepping
          unfinished = self%autotune_step_worker(tune_state, ELPA_SOLVER_2STAGE, error)
      endif ! new_stepping
    end function    

    !> \brief function to do the work of an autotunig step
    !> Parameters
    !> \param   self            class(elpa_impl_t) the allocated ELPA object
    !> \param   tune_state      class(elpa_autotune_t): the autotuning object
    !> \result  unfinished      logical: describes the state of the autotuning (completed/uncompleted)
    function elpa_autotune_step_worker(self, tune_state, solver, error) result(unfinished)
      implicit none
      class(elpa_impl_t), intent(inout)             :: self
      class(elpa_autotune_t), intent(inout), target :: tune_state
      type(elpa_autotune_impl_t), pointer           :: ts_impl
#ifdef USE_FORTRAN2008
      integer(kind=c_int), optional, intent(out)    :: error
#else
      integer(kind=c_int),  intent(out)             :: error
#endif
      integer(kind=c_int), intent(in)               :: solver
      integer(kind=c_int)                           :: error2, error3
      integer                                       :: mpi_comm_parent, mpi_string_length, np_total
      integer(kind=MPI_KIND)                        :: mpierr, mpierr2, mpi_string_lengthMPI
      logical                                       :: unfinished
      integer                                       :: i
      real(kind=C_DOUBLE)                           :: time_spent(2), sendbuf(2), recvbuf(2)
#ifdef WITH_MPI
      character(len=MPI_MAX_ERROR_STRING)           :: mpierr_string
#endif
       integer(kind=MPI_KIND)                       :: allreduce_request1
       logical                                      :: useNonBlockingCollectivesAll
       integer(kind=c_int)                          :: sublevel, current, level

       logical                                      :: use1stage, use2stage

       useNonBlockingCollectivesAll = .true.
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
          write(error_unit,*) "ELPA_AUTOTUNE_STEP_WORKER ERROR: This should not happen"
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = ELPA_ERROR
          endif
#else
          error = ELPA_ERROR
#endif
      end select

      unfinished = .false.

      if (ts_impl%new_stepping == 1) then
        if (solver .eq. ELPA_SOLVER_1STAGE) then
          use1stage = .true.
          use2stage = .false.
        endif
        if (solver .eq. ELPA_SOLVER_2STAGE) then
          use1stage = .false.
          use2stage = .true.
        endif
      endif
  
      if (ts_impl%new_stepping == 1) then
        if (use1stage) then
          ! check on which sublevel we should currently work on
          do sublevel = 1, autotune_level
            if (.not.(autotune_substeps_done1stage(sublevel))) then
              exit
            endif
          enddo
          if (sublevel .le. autotune_level) then
            ! do nothing
          else if (sublevel .eq. autotune_level +1) then
            sublevel = autotune_level
          else
            write(error_unit,'(a)') "Problem setting level in elpa_autotune_step"
#ifdef USE_FORTRAN2008
            if (present(error)) then
              error = ELPA_ERROR
              return
            else
              return
            endif
#else
            error = ELPA_ERROR
            return
#endif

           endif
        endif
        if (use2stage) then
          ! check on which sublevel we should currently work on
          do sublevel = 1, autotune_level
            if (.not.(autotune_substeps_done2stage(sublevel))) then
              exit
            endif
          enddo
          if (sublevel .le. autotune_level) then
            ! do nothing
          else if (sublevel .eq. autotune_level +1) then
            sublevel = autotune_level
          else
            write(error_unit,'(a)') "Problem setting level in elpa_autotune_step"
#ifdef USE_FORTRAN2008
            if (present(error)) then
              error = ELPA_ERROR
              return
            else
              return
            endif
#else
            error = ELPA_ERROR
            return
#endif
          endif
        endif
      endif ! new_stepping

      if (ts_impl%new_stepping == 1) then
        if (use1stage) then
          current = ts_impl%sublevel_current1stage(sublevel)
          !ts_impl%total_current_1stage = current
        endif
        if (use2stage) then
          current = ts_impl%sublevel_current2stage(sublevel)
          !ts_impl%total_current_2stage = current
        endif
      else
        current = ts_impl%current
      endif

      ! check here whether this is the last round call if yes update
      ! if yes update and reset current...      ! if yes update and reset current...

      if (current >= 0) then
#ifdef HAVE_DETAILED_TIMINGS
        if (ts_impl%new_stepping == 1) then
          time_spent(2) = self%autotune_timer%get("accumulator")
          select case (sublevel)
            case (ELPA_AUTOTUNE_TRANSPOSE_VECTORS)
              if (solver == ELPA_SOLVER_2STAGE) then
                time_spent(1) = self%autotune_timer%get("accumulator","full_to_band")
              else if (solver == ELPA_SOLVER_1STAGE) then
                time_spent(1) = self%autotune_timer%get("accumulator","full_to_tridi")
              else
                write(error_unit,'(a)') "Unknown solver: Aborting!"
#ifdef USE_FORTRAN2008
                if (present(error)) then
                  error = ELPA_ERROR
                  return
                else
                  return
                endif
#else
                error = ELPA_ERROR
                return
#endif

              endif
            case (ELPA2_AUTOTUNE_FULL_TO_BAND)
              time_spent(1) = self%autotune_timer%get("accumulator","full_to_band")
            case (ELPA2_AUTOTUNE_BAND_TO_TRIDI)
              time_spent(1) = self%autotune_timer%get("accumulator","band_to_tridi")
            case (ELPA_AUTOTUNE_SOLVE)
              time_spent(1) = self%autotune_timer%get("accumulator","solve")
            case (ELPA2_AUTOTUNE_TRIDI_TO_BAND)
              time_spent(1) = self%autotune_timer%get("accumulator","tridi_to_band")
            case (ELPA2_AUTOTUNE_BAND_TO_FULL)
              time_spent(1) = self%autotune_timer%get("accumulator","band_to_full")
            case (ELPA1_AUTOTUNE_FULL_TO_TRIDI)
              time_spent(1) = self%autotune_timer%get("accumulator","full_to_tridi")
            case (ELPA1_AUTOTUNE_TRIDI_TO_FULL)
              time_spent(1) = self%autotune_timer%get("accumulator","tridi_to_full")
            case (ELPA2_AUTOTUNE_KERNEL)
              time_spent(1) = self%autotune_timer%get("accumulator","tridi_to_band")
            case (ELPA_AUTOTUNE_OPENMP)
              time_spent(1) = self%autotune_timer%get("accumulator")
            case (ELPA2_AUTOTUNE_BAND_TO_FULL_BLOCKING)
              time_spent(1) = self%autotune_timer%get("accumulator","band_to_full")
            case (ELPA2_AUTOTUNE_HERMITIAN_MULTIPLY_BLOCKING)
              time_spent(1) = self%autotune_timer%get("accumulator","hermitian_multiply")
            case (ELPA2_AUTOTUNE_CHOLESKY_BLOCKING)
              time_spent(1) = self%autotune_timer%get("accumulator","cholesky")
            case (ELPA1_AUTOTUNE_MAX_STORED_ROWS)
              time_spent(1) = self%autotune_timer%get("accumulator","tridi_to_full")
            case (ELPA2_AUTOTUNE_TRIDI_TO_BAND_STRIPEWIDTH)
              time_spent(1) = self%autotune_timer%get("accumulator","tridi_to_band")
            case default
              time_spent(1) = self%autotune_timer%get("accumulator")
          end select
        else
          time_spent(1) = self%autotune_timer%get("accumulator")
        endif
#else
        write(error_unit,*) "Cannot do autotuning without detailed timings"

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
          write(error_unit,*) "Parent communicator is not set properly. Aborting..."
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = ELPA_ERROR_CRITICAL
          endif
#else
          error = ELPA_ERROR_CRITICAL
#endif
          return
        endif

        sendbuf(1:2) = time_spent(1:2)
        if (useNonBlockingCollectivesAll) then
          call mpi_iallreduce(sendbuf, recvbuf, 2_MPI_KIND, MPI_REAL8, MPI_SUM, int(mpi_comm_parent,kind=MPI_KIND), &
                            allreduce_request1, mpierr)
          call mpi_wait(allreduce_request1, MPI_STATUS_IGNORE, mpierr)
        else
          call mpi_allreduce(sendbuf, recvbuf, 2_MPI_KIND, MPI_REAL8, MPI_SUM, int(mpi_comm_parent,kind=MPI_KIND), &
                             mpierr)
        endif
        if (mpierr .ne. MPI_SUCCESS) then
          call MPI_ERROR_STRING(mpierr, mpierr_string, mpi_string_lengthMPI, mpierr2)
          mpi_string_length = int(mpi_string_lengthMPI,kind=c_int)
          write(error_unit,*) "MPI ERROR occured during elpa_autotune_step: ", trim(mpierr_string)
          return
        endif
        time_spent(1) = recvbuf(1) / np_total
        time_spent(2) = recvbuf(2) / np_total
#endif /* WITH_MPI */

        if (ts_impl%new_stepping == 1) then
          if (use1stage) then
            if (ts_impl%best_solver .lt. 0) ts_impl%best_solver = ELPA_SOLVER_1STAGE
            if (ts_impl%sublevel_min_loc1stage(sublevel) == -1 .or. (time_spent(1) < ts_impl%sublevel_min_val1stage(sublevel))) then
              ts_impl%min_val = time_spent(1)
              ts_impl%min_loc = ts_impl%current
              ts_impl%sublevel_min_val1stage(sublevel) = time_spent(1)
              ts_impl%sublevel_min_loc1stage(sublevel) = ts_impl%sublevel_current1stage(sublevel)
              ts_impl%best_val1stage = time_spent(2)
              !print *,"WORKER best 1stage: ",ts_impl%min_val
            end if
          endif
          if (use2stage) then
            if (ts_impl%best_solver .lt. 0) ts_impl%best_solver = ELPA_SOLVER_2STAGE
            if (ts_impl%sublevel_min_loc2stage(sublevel) == -1 .or. (time_spent(1) < ts_impl%sublevel_min_val2stage(sublevel))) then
              ts_impl%min_val = time_spent(1)
              ts_impl%min_loc = ts_impl%current
              ts_impl%sublevel_min_val2stage(sublevel) = time_spent(1)
              ts_impl%sublevel_min_loc2stage(sublevel) = ts_impl%sublevel_current2stage(sublevel)
              ts_impl%best_val2stage = time_spent(2)
              !print *,"WORKER best 2stage: ",ts_impl%min_val
            end if
          endif
        else ! new stepping
          if (ts_impl%min_loc == -1 .or. (time_spent(1) < ts_impl%min_val)) then
            ts_impl%min_val = time_spent(1)
            ts_impl%min_loc = ts_impl%current
          end if
        endif ! new stepping
        call self%autotune_timer%free()
      endif ! (current >= 0)

      if (ts_impl%new_stepping == 1) then
        if (use1stage) then
          !print *,"step:",sublevel, ts_impl%sublevel_cardinality1stage(sublevel),"for 1stage"
          !check whether we have to switch to new sublevel
          if (ts_impl%sublevel_current1stage(sublevel) .eq. ts_impl%sublevel_cardinality1stage(sublevel)-1 ) then
            !current = -1
            autotune_substeps_done1stage(sublevel) = .true.
            if (sublevel .lt. autotune_level) then
              ! we can go to the next sublevel _with_ cardinality != 0
              do level = sublevel+1, autotune_level
                !print *,"Checking level",level,"for 1stage"
                if (ts_impl%sublevel_cardinality1stage(level) .ne. 0) then
                  exit
                endif
              enddo
              !print *,"choosing level",level,"for 1stage"
              sublevel=level
              !sublevel = sublevel +1
            else if (sublevel .eq. autotune_level) then
              ! we are already at the last level
            else
              write(error_unit,*) "Panic in autotune step.Aborting!"
#ifdef USE_FORTRAN2008
                if (present(error)) then
                  error = ELPA_ERROR
                  return
                else
                  return
                endif
#else
                error = ELPA_ERROR
                return
#endif
            endif
          endif

          if (all(autotune_substeps_done1stage(:))) then
            return
          endif
        endif ! 1stage
        if (use2stage) then
          !print *,"step:",sublevel, ts_impl%sublevel_cardinality2stage(sublevel),"for 2stage"
          !check whether we have to switch to new sublevel
          if (ts_impl%sublevel_current2stage(sublevel) .eq. ts_impl%sublevel_cardinality2stage(sublevel)-1 ) then
            !current = -1
            autotune_substeps_done2stage(sublevel) = .true.
            if (sublevel .lt. autotune_level) then
              ! we can go to the next sublevel _with_ cardinality != 0
              do level = sublevel+1, autotune_level
                !print *,"Checking level",level,"for 2stage"
                if (ts_impl%sublevel_cardinality2stage(level) .ne. 0) then
                  exit
                endif
              enddo
              !print *,"choosing level",level,"for 2stage"
              sublevel=level
              !sublevel = sublevel +1
            else if (sublevel .eq. autotune_level) then
              ! we are already at the last level
            else
              write(error_unit,*) "Panic in autotune step.Aborting!"
#ifdef USE_FORTRAN2008
                if (present(error)) then
                  error = ELPA_ERROR
                  return
                else
                  return
                endif
#else
                error = ELPA_ERROR
                return
#endif
            endif
          endif

          if (all(autotune_substeps_done2stage(:))) then
            return
          endif
        endif ! 1stage
      endif ! new stepping
     
      if (ts_impl%new_stepping == 1) then
        ! check whether solver is set, if yes we can pass t
        if (use1stage) then
          !print *,"Working on sublevel=",sublevel,ts_impl%sublevel_cardinality1stage(sublevel),"for 1stage"
          do while (ts_impl%sublevel_current1stage(sublevel) < ts_impl%sublevel_cardinality1stage(sublevel)-1)
            ts_impl%current = ts_impl%current + 1
            ts_impl%sublevel_current1stage(sublevel) = ts_impl%sublevel_current1stage(sublevel) + 1
            ts_impl%total_current_1stage = ts_impl%total_current_1stage + 1
            !print *,"Updating current1stage", ts_impl%total_current_1stage
            if (elpa_index_set_autotune_parameters_new_stepping_c(self%index, sublevel, ts_impl%domain, &
                ts_impl%sublevel_part1stage(sublevel), &
                ts_impl%sublevel_current1stage(sublevel)) == 1) then
              unfinished = .true.
              return
            end if
          end do

          if (ts_impl%sublevel_current1stage(sublevel) .ne. ts_impl%sublevel_cardinality1stage(sublevel)-1 ) then
             write(error_unit,*) "PANIC in autotune_step 1stage"
             write(error_unit,*) ts_impl%sublevel_current1stage(sublevel),ts_impl%sublevel_cardinality1stage(sublevel) 
#ifdef USE_FORTRAN2008
                if (present(error)) then
                  error = ELPA_ERROR
                  return
                else
                  return
                endif
#else
                error = ELPA_ERROR
                return
#endif
          endif
        endif ! 1stage
        if (use2stage) then
          !print *,"Working on sublevel=",sublevel,ts_impl%sublevel_cardinality2stage(sublevel),"for 2stage"
          do while (ts_impl%sublevel_current2stage(sublevel) < ts_impl%sublevel_cardinality2stage(sublevel)-1)
            ts_impl%current = ts_impl%current + 1
            ts_impl%sublevel_current2stage(sublevel) = ts_impl%sublevel_current2stage(sublevel) + 1
            ts_impl%total_current_2stage = ts_impl%total_current_2stage + 1
            if (elpa_index_set_autotune_parameters_new_stepping_c(self%index, sublevel, ts_impl%domain, &
                ts_impl%sublevel_part2stage(sublevel), &
                ts_impl%sublevel_current2stage(sublevel)) == 1) then
              unfinished = .true.
              return
            end if
          end do

          if (ts_impl%sublevel_current2stage(sublevel) .ne. ts_impl%sublevel_cardinality2stage(sublevel)-1 ) then
             write(error_unit,*) "PANIC in autotune_step 2stage"
             write(error_unit,*) ts_impl%sublevel_current2stage(sublevel),ts_impl%sublevel_cardinality2stage(sublevel) 
#ifdef USE_FORTRAN2008
                if (present(error)) then
                  error = ELPA_ERROR
                  return
                else
                  return
                endif
#else
                error = ELPA_ERROR
                return
#endif
          endif
        endif ! 2stage
      else ! new_stepping
        do while (ts_impl%current < ts_impl%cardinality - 1)
          ts_impl%current = ts_impl%current + 1
          if (elpa_index_set_autotune_parameters_c(self%index, ts_impl%level, ts_impl%domain, ts_impl%current) == 1) then
            unfinished = .true.
            return
          end if
        end do
      endif ! new stepping

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
      use elpa_mpi
      implicit none
      class(elpa_impl_t), intent(inout)          :: self
      class(elpa_autotune_t), intent(in), target :: tune_state
      type(elpa_autotune_impl_t), pointer        :: ts_impl
#ifdef USE_FORTRAN2008
      integer(kind=ik), optional, intent(out)    :: error
#else
      integer(kind=ik), intent(out)              :: error
#endif
      integer(kind=c_int)                        :: sublevel, level, solver
      integer(kind=c_int)                        :: myid, mpi_comm_parent
      integer(kind=MPI_KIND)                     :: myidMPI, mpierr

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

      if (ts_impl%new_stepping == 1) then

#ifdef WITH_MPI
        call self%get("mpi_comm_parent",mpi_comm_parent, error)
        if (error .ne. ELPA_OK) then
          write(error_unit,*) "ELPA_AUTOTUNE_SET_BEST: cannot get mpi_comm_parent"
          return
        endif
        call mpi_comm_rank(int(mpi_comm_parent,kind=MPI_KIND) ,myidMPI ,mpierr)
        myid = int(myidMPI, kind=c_int)
#endif

        ! check on which sublevel we should currently work on
        if (ts_impl%best_solver == ELPA_SOLVER_1STAGE) then
          do sublevel = 1, autotune_level
            if (.not.(autotune_substeps_done1stage(sublevel))) then
              exit
            endif
          enddo
          if (sublevel .le. autotune_level) then
            ! do nothing
          else if (sublevel .eq. autotune_level +1) then
            sublevel = autotune_level
          else
            write(error_unit,'(a)') "Problem setting level in elpa_autotune_step"
#ifdef USE_FORTRAN2008
            if (present(error)) then
              error = ELPA_ERROR
              return
            else
              return
            endif
#else
            error = ELPA_ERROR
            return
#endif
          endif
        endif
        if (ts_impl%best_solver == ELPA_SOLVER_2STAGE) then
          do sublevel = 1, autotune_level
            if (.not.(autotune_substeps_done2stage(sublevel))) then
              exit
            endif
          enddo
          if (sublevel .le. autotune_level) then
            ! do nothing
          else if (sublevel .eq. autotune_level +1) then
            sublevel = autotune_level
          else
            write(error_unit,'(a)') "Problem setting level in elpa_autotune_step"
#ifdef USE_FORTRAN2008
            if (present(error)) then
              error = ELPA_ERROR
              return
            else
              return
            endif
#else
            error = ELPA_ERROR
            return
#endif
          endif
        endif
      endif

      if (ts_impl%new_stepping == 1) then
        ! should be already set!
        !if (self%is_set("solver") == 1) then
        !  call self%get("solver", solver, errddor)
        !  if (solver == ELPA_SOLVER_2STAGE) then
        !    ts_impl%sublevel_part(sublevel) = ELPA_AUTOTUNE_PART_ELPA2
        !  else if (solver == ELPA_SOLVER_1STAGE) then
        !    ts_impl%sublevel_part(sublevel) = ELPA_AUTOTUNE_PART_ELPA1
        !  else
        !    print *,"ELPA_AUTOTUNE_STEP: Unknown solver"
        !    stop 1
        !  endif        
        !else 
        !  ts_impl%sublevel_part(sublevel) = ELPA_AUTOTUNE_PART_ANY
        !endif
        ! loop over sublevels
        if (ts_impl%best_solver .eq. ELPA_SOLVER_1STAGE) then
          if (myid .eq. 0) print *,"ELPA_SOLVER_1STAGE is the best solver: setting tuned values"
          call self%set("solver", ELPA_SOLVER_1STAGE, error)
          if (error .ne. ELPA_OK) then
            write(error_unit,*) "ELPA_AUTOTUNE_SET_BEST: cannot set ELPA_SOLVER_1STAGE for tuning"
            return
          endif
          do level=1, sublevel
            if (ts_impl%sublevel_cardinality1stage(level) .eq. 0) then
              cycle
            endif
            !print *,"level=",level,ts_impl%sublevel_cardinality1stage(level)
            if (elpa_index_set_autotune_parameters_new_stepping_c(self%index, level, ts_impl%domain, &
                    ts_impl%sublevel_part1stage(level), &
                    ts_impl%sublevel_min_loc1stage(level)) /= 1) then
              write(error_unit, *) "This should not happen (in elpa_autotune_set_best())"
#ifdef USE_FORTRAN2008
              if (present(error)) then
                error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
              endif
#else
              error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
#endif
            endif
          enddo
        endif ! 1stage
        if (ts_impl%best_solver .eq. ELPA_SOLVER_2STAGE) then
          if (myid .eq. 0) print *,"ELPA_SOLVER_2STAGE is the best solver: setting tuned values"
          call self%set("solver", ELPA_SOLVER_2STAGE, error)
          if (error .ne. ELPA_OK) then
            write(error_unit,*) "ELPA_AUTOTUNE_SET_BEST: cannot set ELPA_SOLVER_2STAGE for tuning"
            return
          endif
          do level=1, sublevel
            if (ts_impl%sublevel_cardinality2stage(level) .eq. 0) then
              cycle
            endif
            if (elpa_index_set_autotune_parameters_new_stepping_c(self%index, level, ts_impl%domain, &
                    ts_impl%sublevel_part2stage(level), &
                    ts_impl%sublevel_min_loc2stage(level)) /= 1) then
              write(error_unit, *) "This should not happen (in elpa_autotune_set_best())"
#ifdef USE_FORTRAN2008
              if (present(error)) then
                error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
              endif
#else
              error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
#endif
            endif
          enddo
        endif ! 2stage
      else ! new_steppimg
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
      endif ! new stepping

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
      integer(kind=c_int)                        :: sublevel, level, solver

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

      if (ts_impl%new_stepping == 1) then
        if (consider_solver == ELPA_SOLVER_1STAGE) then
          ! check on which sublevel we should currently work on
          do sublevel = 1, autotune_level
            if (.not.(autotune_substeps_done1stage(sublevel))) then
              exit
            endif
          enddo
          if (sublevel .le. autotune_level) then
            ! do nothing
          else if (sublevel .eq. autotune_level +1) then
            sublevel = autotune_level
          else
            write(error_unit,'(a)') "Problem setting level in elpa_autotune_step"
#ifdef USE_FORTRAN2008
            if (present(error)) then
              error = ELPA_ERROR
              return
            else
              return
            endif
#else
            error = ELPA_ERROR
            return
#endif
          endif
        endif ! 1 stage
        if (consider_solver == ELPA_SOLVER_2STAGE) then
          ! check on which sublevel we should currently work on
          do sublevel = 1, autotune_level
            if (.not.(autotune_substeps_done2stage(sublevel))) then
              exit
            endif
          enddo
          if (sublevel .le. autotune_level) then
            ! do nothing
          else if (sublevel .eq. autotune_level +1) then
            sublevel = autotune_level
          else
            write(error_unit,'(a)') "Problem setting level in elpa_autotune_print_best"
#ifdef USE_FORTRAN2008
            if (present(error)) then
              error = ELPA_ERROR
              return
            else
              return
            endif
#else
            error = ELPA_ERROR
            return
#endif
          endif
        endif ! 2 stage
      endif ! new stepping

      flush(output_unit)
      if (ts_impl%new_stepping == 1) then

        ! should already be set
        !if (self%is_set("solver") == 1) then
        !  call self%get("solver", solver, error)
        !  if (solver == ELPA_SOLVER_2STAGE) then
        !    ts_impl%sublevel_part(sublevel) = ELPA_AUTOTUNE_PART_ELPA2
        !  else if (solver == ELPA_SOLVER_1STAGE) then
        !    ts_impl%sublevel_part(sublevel) = ELPA_AUTOTUNE_PART_ELPA1
        !  else
        !    print *,"ELPA_AUTOTUNE_STEP: Unknown solver"
        !    stop 1
        !  endif        
        !else 
        !  ts_impl%sublevel_part(sublevel) = ELPA_AUTOTUNE_PART_ANY
        !endif

        ! loop over sublevels
        if (consider_solver == ELPA_SOLVER_1STAGE) then
          do level=1, sublevel
            if (ts_impl%sublevel_cardinality1stage(level) .eq. 0) then
              cycle
            endif
            if (elpa_index_print_autotune_parameters_new_stepping_c(self%index, level, &
                    ts_impl%domain, ts_impl%sublevel_part1stage(level)) /= 1) then
              write(error_unit, *) "This should not happen (in elpa_autotune_print_best())"
#ifdef USE_FORTRAN2008
              if (present(error)) then
                error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
              endif
#else
              error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
#endif
            endif
          enddo
        endif ! 1stage
        if (consider_solver == ELPA_SOLVER_2STAGE) then
          do level=1, sublevel
            if (ts_impl%sublevel_cardinality2stage(level) .eq. 0) then
              cycle
            endif
            if (elpa_index_print_autotune_parameters_new_stepping_c(self%index, level, &
                    ts_impl%domain, ts_impl%sublevel_part2stage(level)) /= 1) then
              write(error_unit, *) "This should not happen (in elpa_autotune_print_best())"
#ifdef USE_FORTRAN2008
              if (present(error)) then
                error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
              endif
#else
              error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
#endif
            endif
          enddo
        endif ! 2stage
      else ! new stepping
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
      endif ! new stepping
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
      integer(kind=c_int)                        :: level, sublevel, solver

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

      !print *,"print_state: consider_solver",consider_solver
      if (ts_impl%new_stepping == 1) then
        if (consider_solver == ELPA_SOLVER_1STAGE) then
          do sublevel = 1, autotune_level
            if (.not.(autotune_substeps_done1stage(sublevel))) then
              exit
            endif
          enddo
          if (sublevel .le. autotune_level) then
            ! do nothing
          else if (sublevel .eq. autotune_level +1) then
            sublevel = autotune_level
          else
            write(error_unit,'(a)') "Problem setting level in elpa_autotune_print_state"
#ifdef USE_FORTRAN2008
            if (present(error)) then
              error = ELPA_ERROR
              return
            else
              return
            endif
#else
            error = ELPA_ERROR
            return
#endif
          endif

          ! check
          if (sublevel .gt. 1) then
            !if (sum(ts_impl%sublevel_cardinality1stage(0:sublevel-1)) .gt. ts_impl%current) then
            !if (sum(ts_impl%sublevel_cardinality1stage(0:sublevel-1)) .gt. &
            !        ts_impl%sublevel_current1stage(sublevel)) then
            if (sum(ts_impl%sublevel_cardinality1stage(0:sublevel-1)) .gt. &
                    ts_impl%total_current_1stage+1) then
              write(error_unit,*) "something wrong in print state for 1stage 1", &
                      sublevel, autotune_substeps_done1stage(sublevel), &
                      sum(ts_impl%sublevel_cardinality1stage(0:sublevel-1)), &
                      ts_impl%total_current_1stage
                      !ts_impl%sublevel_current1stage(sublevel)
#ifdef USE_FORTRAN2008
              if (present(error)) then
                error = ELPA_ERROR
                return
              else
                return
              endif
#else
              error = ELPA_ERROR
              return
#endif
            endif
          else
            !if (ts_impl%sublevel_cardinality1stage(sublevel) .lt. ts_impl%current) then
            !if (ts_impl%sublevel_cardinality1stage(sublevel) .lt. &
            !        ts_impl%sublevel_current1stage(sublevel)) then
            if (ts_impl%sublevel_cardinality1stage(sublevel) .lt. &
                    ts_impl%total_current_1stage) then
              write(error_unit,*) "something wrong in print state 1stage 2"
#ifdef USE_FORTRAN2008
              if (present(error)) then
                error = ELPA_ERROR
                return
              else
                return
              endif
#else
              error = ELPA_ERROR
              return
#endif
            endif
          endif

          ! should already be done
          !if (self%is_set("solver") == 1) then
          !  call self%get("solver", solver, error)
          !  if (solver == ELPA_SOLVER_2STAGE) then
          !    ts_impl%sublevel_part(sublevel) = ELPA_AUTOTUNE_PART_ELPA2
          !  else if (solver == ELPA_SOLVER_1STAGE) then
          !    ts_impl%sublevel_part(sublevel) = ELPA_AUTOTUNE_PART_ELPA1
          !  else
          !    print *,"ELPA_AUTOTUNE_STEP: Unknown solver"
          !    stop 1
          !  endif        
          !else 
          !  ts_impl%sublevel_part(sublevel) = ELPA_AUTOTUNE_PART_ANY
          !endif
        endif ! 1 stage
        if (consider_solver == ELPA_SOLVER_2STAGE) then
          do sublevel = 1, autotune_level
            if (.not.(autotune_substeps_done2stage(sublevel))) then
              exit
            endif
          enddo
          if (sublevel .le. autotune_level) then
            ! do nothing
          else if (sublevel .eq. autotune_level +1) then
            sublevel = autotune_level
          else
            write(error_unit,*) "Problem setting level in elpa_autotune_print_state"
#ifdef USE_FORTRAN2008
            if (present(error)) then
              error = ELPA_ERROR
              return
            else
              return
            endif
#else
            error = ELPA_ERROR
            return
#endif
          endif

          ! check
          if (sublevel .gt. 1) then
            !if (sum(ts_impl%sublevel_cardinality2stage(0:sublevel-1)) .gt. ts_impl%current) then
            !if (sum(ts_impl%sublevel_cardinality2stage(0:sublevel-1)) .gt. &
            !        ts_impl%sublevel_current2stage(sublevel)) then
            if (sum(ts_impl%sublevel_cardinality2stage(0:sublevel-1)) .gt. &
                    ts_impl%total_current_2stage) then
              write(error_unit,*) "something wrong in print state 2stage 1", &
                      sum(ts_impl%sublevel_cardinality2stage(0:sublevel-1)), &
                      ts_impl%current
#ifdef USE_FORTRAN2008
              if (present(error)) then
                error = ELPA_ERROR
                return
              else
                return
              endif
#else
              error = ELPA_ERROR
              return
#endif
            endif
          else
            !if (ts_impl%sublevel_cardinality2stage(sublevel) .lt. ts_impl%current) then
            !if (ts_impl%sublevel_cardinality2stage(sublevel) .lt. &
            !        ts_impl%sublevel_current2stage(sublevel)) then
            if (ts_impl%sublevel_cardinality2stage(sublevel) .lt. &
                    ts_impl%total_current_2stage) then
              write(error_unit,*) "something wrong in print state 2stage 2", &
                      ts_impl%sublevel_cardinality2stage(sublevel),&
                       ts_impl%current
#ifdef USE_FORTRAN2008
              if (present(error)) then
                error = ELPA_ERROR
                return
              else
                return
              endif
#else
              error = ELPA_ERROR
              return
#endif
            endif
          endif

          ! should already be done
          !if (self%is_set("solver") == 1) then
          !  call self%get("solver", solver, error)
          !  if (solver == ELPA_SOLVER_2STAGE) then
          !    ts_impl%sublevel_part(sublevel) = ELPA_AUTOTUNE_PART_ELPA2
          !  else if (solver == ELPA_SOLVER_1STAGE) then
          !    ts_impl%sublevel_part(sublevel) = ELPA_AUTOTUNE_PART_ELPA1
          !  else
          !    print *,"ELPA_AUTOTUNE_STEP: Unknown solver"
          !    stop 1
          !  endif        
          !else 
          !  ts_impl%sublevel_part(sublevel) = ELPA_AUTOTUNE_PART_ANY
          !endif
        endif ! 2 stage
      endif

      if (ts_impl%new_stepping == 1) then
        if (consider_solver == ELPA_SOLVER_1STAGE) then
          ! loop over sublevels
          do level=1, sublevel
            if (ts_impl%sublevel_cardinality1stage(level) .eq. 0) then
              cycle
            endif
            if (elpa_index_print_autotune_state_new_stepping_c(self%index, level, ts_impl%domain, &
                    ts_impl%sublevel_part1stage(level), &
                    ts_impl%sublevel_min_loc1stage(level), &
                      ts_impl%sublevel_min_val1stage(level), ts_impl%sublevel_current1stage(level), &
                      ts_impl%sublevel_cardinality1stage(level), &
                      ELPA_SOLVER_1STAGE, &
                      c_null_char) /= 1) then
              write(error_unit, *) "This should not happen (in elpa_autotune_print_state())"
#ifdef USE_FORTRAN2008
              if (present(error)) then
                error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
              endif
#else
              error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
#endif
            endif
          enddo
        endif ! 1 stage
        if (consider_solver == ELPA_SOLVER_2STAGE) then
          ! loop over sublevels
          do level=1, sublevel
            if (ts_impl%sublevel_cardinality2stage(level) .eq. 0) then
              cycle
            endif
            if (elpa_index_print_autotune_state_new_stepping_c(self%index, level, ts_impl%domain, &
                    ts_impl%sublevel_part2stage(level), &
                    ts_impl%sublevel_min_loc2stage(level), &
                      ts_impl%sublevel_min_val2stage(level), ts_impl%sublevel_current2stage(level), &
                      ts_impl%sublevel_cardinality2stage(level), &
                      ELPA_SOLVER_2STAGE, &
                      c_null_char) /= 1) then
              write(error_unit, *) "This should not happen (in elpa_autotune_print_state())"
#ifdef USE_FORTRAN2008
              if (present(error)) then
                error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
              endif
#else
              error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
#endif
            endif
          enddo
        endif ! 2 stage
      else ! new stepping
        if (elpa_index_print_autotune_state_c(self%index, ts_impl%level, ts_impl%domain, ts_impl%min_loc, &
                  ts_impl%min_val, ts_impl%current, &
                  ts_impl%cardinality, c_null_char) /= 1) then
          write(error_unit, *) "This should not happen (in elpa_autotune_print_state())"
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
          endif
#else
          error = ELPA_ERROR_AUTOTUNE_OBJECT_CHANGED
#endif
        endif
      endif ! new stepping

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
      integer(kind=c_int)                        :: sublevel, level, solver

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

      if (ts_impl%new_stepping == 1) then
        if (consider_solver == ELPA_SOLVER_1STAGE) then
          do sublevel = 1, autotune_level
            if (.not.(autotune_substeps_done1stage(sublevel))) then
              exit
            endif
          enddo
          if (sublevel .le. autotune_level) then
            ! do nothing
          else if (sublevel .eq. autotune_level +1) then
            sublevel = autotune_level
          else
            write(error_unit,*) "Problem setting level in elpa_autotune_save_state"
#ifdef USE_FORTRAN2008
            if (present(error)) then
              error = ELPA_ERROR
              return
            else
              return
            endif
#else
            error = ELPA_ERROR
            return
#endif
          endif
        endif ! 1 stage
        if (consider_solver == ELPA_SOLVER_2STAGE) then
          do sublevel = 1, autotune_level
            if (.not.(autotune_substeps_done2stage(sublevel))) then
              exit
            endif
          enddo
          if (sublevel .le. autotune_level) then
            ! do nothing
          else if (sublevel .eq. autotune_level +1) then
            sublevel = autotune_level
          else
            write(error_unit,*) "Problem setting level in elpa_autotune_save_state"
#ifdef USE_FORTRAN2008
            if (present(error)) then
              error = ELPA_ERROR
              return
            else
              return
            endif
#else
            error = ELPA_ERROR
            return
#endif
          endif
        endif ! 2 stage
      endif


      if (ts_impl%new_stepping == 1) then

        ! should already be set
        ! 
        !if (self%is_set("solver") == 1) then
        !  call self%get("solver", solver, error)
        !  if (solver == ELPA_SOLVER_2STAGE) then
        !    ts_impl%sublevel_part(sublevel) = ELPA_AUTOTUNE_PART_ELPA2
        !  else if (solver == ELPA_SOLVER_1STAGE) then
        !    ts_impl%sublevel_part(sublevel) = ELPA_AUTOTUNE_PART_ELPA1
        !  else
        !    print *,"ELPA_AUTOTUNE_STEP: Unknown solver"
        !    stop 1
        !  endif        
        !else 
        !  ts_impl%sublevel_part(sublevel) = ELPA_AUTOTUNE_PART_ANY
        !endif


        ! loop over sublevels
        if (consider_solver == ELPA_SOLVER_1STAGE) then
          do level=1, sublevel
            if (ts_impl%sublevel_cardinality1stage(level) .eq. 0) then
              cycle
            endif
            if (elpa_index_print_autotune_state_new_stepping_c(self%index, level, ts_impl%domain, &
                    ts_impl%sublevel_part1stage(level), &
                    ts_impl%sublevel_min_loc1stage(level), &
                      ts_impl%sublevel_min_val1stage(level), ts_impl%sublevel_current1stage(level), &
                      ts_impl%sublevel_cardinality1stage(level), &
                      ELPA_SOLVER_1STAGE, &
                      file_name // c_null_char) /= 1) then
              write(error_unit, *) "This should not happen (in elpa_autotune_save_state())"
#ifdef USE_FORTRAN2008
              if (present(error)) then
                error = ELPA_ERROR_CANNOT_OPEN_FILE
              endif
#else
              error = ELPA_ERROR_CANNOT_OPEN_FILE
#endif
            endif
          enddo
        endif ! 1 stage
        if (consider_solver == ELPA_SOLVER_2STAGE) then
          do level=1, sublevel
            if (ts_impl%sublevel_cardinality2stage(level) .eq. 0) then
              cycle
            endif
            if (elpa_index_print_autotune_state_new_stepping_c(self%index, level, ts_impl%domain, &
                    ts_impl%sublevel_part2stage(level), &
                    ts_impl%sublevel_min_loc2stage(level), &
                      ts_impl%sublevel_min_val2stage(level), ts_impl%sublevel_current2stage(level), &
                      ts_impl%sublevel_cardinality2stage(level), &
                      ELPA_SOLVER_2STAGE, &
                      file_name // c_null_char) /= 1) then
              write(error_unit, *) "This should not happen (in elpa_autotune_save_state())"
#ifdef USE_FORTRAN2008
              if (present(error)) then
                error = ELPA_ERROR_CANNOT_OPEN_FILE
              endif
#else
              error = ELPA_ERROR_CANNOT_OPEN_FILE
#endif
            endif
          enddo
        endif ! 2 stage
      else ! new stepping
        if (elpa_index_print_autotune_state_c(self%index, ts_impl%level, ts_impl%domain, ts_impl%min_loc, &
                  ts_impl%min_val, ts_impl%current, &
                  ts_impl%cardinality, file_name // c_null_char) /= 1) then
          write(error_unit, *) "This should not happen (in elpa_autotune_save_state())"
#ifdef USE_FORTRAN2008
          if (present(error)) then
            error = ELPA_ERROR_CANNOT_OPEN_FILE
          endif
#else
          error = ELPA_ERROR_CANNOT_OPEN_FILE
#endif
        endif
      endif ! new stepping

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
!TODO
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

      if (ts_impl%new_stepping == 1) then
        write(error_unit, *) "elpa_autotune_load_state currently ",&
                 "not implemented for new stepping"
#ifdef USE_FORTRAN2008
           if (present(error)) then
             error = ELPA_ERROR
             return
           endif
#else
           error = ELPA_ERROR
           return
#endif
        !if (elpa_index_load_autotune_state_new_stepping_c(self%index, ts_impl%level, ts_impl%domain, ts_impl%min_loc, &
        !          ts_impl%min_val, ts_impl%current, ts_impl%cardinality, file_name // c_null_char) /= 1) then
        !   write(error_unit, *) "This should not happen (in elpa_autotune_load_state())"
#ifdef USE_FORTRAN2008
        !   if (present(error)) then
        !     error = ELPA_ERROR_CANNOT_OPEN_FILE
        !   endif
#else
        !   error = ELPA_ERROR_CANNOT_OPEN_FILE
#endif
        !endif
      else
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
