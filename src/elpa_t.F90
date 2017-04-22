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
! This file has been rewritten by L. Huedepohl and A. Marek, MPCDF


#include <elpa/elpa_constants.h>
#include "config-f90.h"

module elpa_type
  use, intrinsic :: iso_c_binding

  integer, parameter :: ELPA_SOLVER_1STAGE = ELPA_C_SOLVER_1STAGE
  integer, parameter :: ELPA_SOLVER_2STAGE = ELPA_C_SOLVER_2STAGE

  integer, parameter :: ELPA_2STAGE_REAL_GENERIC           = ELPA_C_2STAGE_REAL_GENERIC
  integer, parameter :: ELPA_2STAGE_REAL_GENERIC_SIMPLE    = ELPA_C_2STAGE_REAL_GENERIC_SIMPLE
  integer, parameter :: ELPA_2STAGE_REAL_BGP               = ELPA_C_2STAGE_REAL_BGP
  integer, parameter :: ELPA_2STAGE_REAL_BGQ               = ELPA_C_2STAGE_REAL_BGQ
  integer, parameter :: ELPA_2STAGE_REAL_SSE               = ELPA_C_2STAGE_REAL_SSE
  integer, parameter :: ELPA_2STAGE_REAL_SSE_BLOCK2        = ELPA_C_2STAGE_REAL_SSE_BLOCK2
  integer, parameter :: ELPA_2STAGE_REAL_SSE_BLOCK4        = ELPA_C_2STAGE_REAL_SSE_BLOCK4
  integer, parameter :: ELPA_2STAGE_REAL_SSE_BLOCK6        = ELPA_C_2STAGE_REAL_SSE_BLOCK6
  integer, parameter :: ELPA_2STAGE_REAL_AVX_BLOCK2        = ELPA_C_2STAGE_REAL_AVX_BLOCK2
  integer, parameter :: ELPA_2STAGE_REAL_AVX_BLOCK4        = ELPA_C_2STAGE_REAL_AVX_BLOCK4
  integer, parameter :: ELPA_2STAGE_REAL_AVX_BLOCK6        = ELPA_C_2STAGE_REAL_AVX_BLOCK6
  integer, parameter :: ELPA_2STAGE_REAL_AVX2_BLOCK2       = ELPA_C_2STAGE_REAL_AVX2_BLOCK2
  integer, parameter :: ELPA_2STAGE_REAL_AVX2_BLOCK4       = ELPA_C_2STAGE_REAL_AVX2_BLOCK4
  integer, parameter :: ELPA_2STAGE_REAL_AVX512_BLOCK2     = ELPA_C_2STAGE_REAL_AVX512_BLOCK2
  integer, parameter :: ELPA_2STAGE_REAL_AVX512_BLOCK4     = ELPA_C_2STAGE_REAL_AVX512_BLOCK4
  integer, parameter :: ELPA_2STAGE_REAL_AVX512_BLOCK6     = ELPA_C_2STAGE_REAL_AVX512_BLOCK6
  integer, parameter :: ELPA_2STAGE_REAL_GPU               = ELPA_C_2STAGE_REAL_GPU
  integer, parameter :: ELPA_2STAGE_REAL_DEFAULT           = ELPA_C_2STAGE_REAL_DEFAULT

  integer, parameter :: ELPA_2STAGE_COMPLEX_GENERIC        = ELPA_C_2STAGE_COMPLEX_GENERIC
  integer, parameter :: ELPA_2STAGE_COMPLEX_GENERIC_SIMPLE = ELPA_C_2STAGE_COMPLEX_GENERIC_SIMPLE
  integer, parameter :: ELPA_2STAGE_COMPLEX_BGP            = ELPA_C_2STAGE_COMPLEX_BGP
  integer, parameter :: ELPA_2STAGE_COMPLEX_BGQ            = ELPA_C_2STAGE_COMPLEX_BGQ
  integer, parameter :: ELPA_2STAGE_COMPLEX_SSE            = ELPA_C_2STAGE_COMPLEX_SSE
  integer, parameter :: ELPA_2STAGE_COMPLEX_SSE_BLOCK1     = ELPA_C_2STAGE_COMPLEX_SSE_BLOCK1
  integer, parameter :: ELPA_2STAGE_COMPLEX_SSE_BLOCK2     = ELPA_C_2STAGE_COMPLEX_SSE_BLOCK2
  integer, parameter :: ELPA_2STAGE_COMPLEX_AVX_BLOCK1     = ELPA_C_2STAGE_COMPLEX_AVX_BLOCK1
  integer, parameter :: ELPA_2STAGE_COMPLEX_AVX_BLOCK2     = ELPA_C_2STAGE_COMPLEX_AVX_BLOCK2
  integer, parameter :: ELPA_2STAGE_COMPLEX_AVX2_BLOCK1    = ELPA_C_2STAGE_COMPLEX_AVX2_BLOCK1
  integer, parameter :: ELPA_2STAGE_COMPLEX_AVX2_BLOCK2    = ELPA_C_2STAGE_COMPLEX_AVX2_BLOCK2
  integer, parameter :: ELPA_2STAGE_COMPLEX_AVX512_BLOCK1  = ELPA_C_2STAGE_COMPLEX_AVX512_BLOCK1
  integer, parameter :: ELPA_2STAGE_COMPLEX_AVX512_BLOCK2  = ELPA_C_2STAGE_COMPLEX_AVX512_BLOCK2
  integer, parameter :: ELPA_2STAGE_COMPLEX_GPU            = ELPA_C_2STAGE_COMPLEX_GPU
  integer, parameter :: ELPA_2STAGE_COMPLEX_DEFAULT        = ELPA_C_2STAGE_COMPLEX_DEFAULT

  integer, parameter :: ELPA_OK    = ELPA_C_OK
  integer, parameter :: ELPA_ERROR = ELPA_C_ERROR

  public :: elpa_init, elpa_initialized, elpa_uninit, elpa_create, elpa_t, c_int, c_double, c_float

  interface elpa_create
    module procedure elpa_create_generic
    module procedure elpa_create_special
  end interface

  type :: elpa_t
   private
   type(c_ptr)         :: options = C_NULL_PTR
   integer             :: mpi_comm_parent = 0
   integer(kind=c_int) :: mpi_comm_rows = 0
   integer(kind=c_int) :: mpi_comm_cols = 0
   integer(kind=c_int) :: na = 0
   integer(kind=c_int) :: nev = 0
   integer(kind=c_int) :: local_nrows = 0
   integer(kind=c_int) :: local_ncols = 0
   integer(kind=c_int) :: nblk = 0
   real(kind=c_double), public  :: time_evp_fwd
   real(kind=c_double), public  :: time_evp_solve
   real(kind=c_double), public  :: time_evp_back

   contains
     generic, public :: set => elpa_set_integer
     generic, public :: get => elpa_get_integer

     procedure, public :: get_communicators => get_communicators

     procedure, public :: set_comm_rows
     procedure, public :: set_comm_cols

     generic, public :: solve => elpa_solve_real_double, &
                                 elpa_solve_real_single, &
                                 elpa_solve_complex_double, &
                                 elpa_solve_complex_single
     generic, public :: multiply_a_b => elpa_multiply_at_b_double, &
                                        elpa_multiply_ah_b_double, &
                                        elpa_multiply_at_b_single, &
                                        elpa_multiply_ah_b_single
     generic, public :: cholesky => elpa_cholesky_double_real, &
                                    elpa_cholesky_single_real, &
                                    elpa_cholesky_double_complex, &
                                    elpa_cholesky_single_complex
     generic, public :: invert_trm => elpa_invert_trm_double_real, &
                                      elpa_invert_trm_single_real, &
                                      elpa_invert_trm_double_complex, &
                                      elpa_invert_trm_single_complex
     generic, public :: solve_tridi => elpa_solve_tridi_double_real, &
                                       elpa_solve_tridi_single_real



     procedure, public :: destroy => elpa_destroy

     ! privates:
     procedure, private :: elpa_set_integer
     procedure, private :: elpa_get_integer

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
  end type elpa_t

  logical :: initDone = .false.

  integer, parameter :: earliest_api_version = 20170403
  integer, parameter :: current_api_version  = 20170403

  contains

    function elpa_init(api_version) result(success)
      use elpa_utilities, only : error_unit
      implicit none
      integer, intent(in) :: api_version
      integer             :: success

      if (earliest_api_version <= api_version .and. api_version <= current_api_version) then
        initDone = .true.
        success = ELPA_OK
      else
        write(error_unit, "(a,i0,a)") "ELPA: Error API version ", api_version," is not supported by this library"
        success = ELPA_ERROR
      endif
    end function


    function elpa_initialized() result(state)
      logical :: state
      state = initDone
    end function


    subroutine elpa_uninit()
    end subroutine


    function elpa_create_generic(na, nev, local_nrows, local_ncols, nblk, mpi_comm_parent, &
                                 process_row, process_col, success) result(obj)
      use precision
      use elpa_mpi
      use elpa_utilities, only : error_unit
      use elpa1_impl, only : elpa_get_communicators_impl
      use elpa_generated_fortran_interfaces
      implicit none

      integer(kind=ik), intent(in) :: na, nev, local_nrows, local_ncols, nblk
      integer, intent(in)          :: mpi_comm_parent, process_row, process_col
      type(elpa_t)                 :: obj
      integer                      :: mpierr

      integer, optional            :: success

      ! check whether init has ever been called
      if (.not.(elpa_initialized())) then
        write(error_unit, *) "elpa_create(): you must call elpa_init() once before creating instances of ELPA"
        if(present(success)) then
          success = ELPA_ERROR
        endif
        return
      endif

      obj%options     = elpa_allocate_options()
      obj%na          = na
      obj%nev         = nev
      obj%local_nrows = local_nrows
      obj%local_ncols = local_ncols
      obj%nblk        = nblk

      obj%mpi_comm_parent = mpi_comm_parent
      mpierr = elpa_get_communicators_impl(mpi_comm_parent, process_row, process_col, obj%mpi_comm_rows, obj%mpi_comm_cols)
      if (mpierr /= MPI_SUCCESS) then
        write(error_unit, *) "elpa_create(): error constructing row and column communicators"
        if(present(success)) then
          success = ELPA_ERROR
        endif
        return
      endif

      if(present(success)) then
        success = ELPA_OK
      endif

    end function

    function elpa_create_special(na, nev, local_nrows, local_ncols, nblk, success) result(obj)
      use precision
      use elpa_mpi
      use elpa_utilities, only : error_unit
      use elpa1_impl, only : elpa_get_communicators_impl
      use elpa_generated_fortran_interfaces
      implicit none

      integer(kind=ik), intent(in) :: na, nev, local_nrows, local_ncols, nblk
      !integer, intent(in)          :: mpi_comm_rows, mpi_comm_cols, process_row, process_col
      type(elpa_t)                 :: obj
      integer                      :: mpierr

      integer                      :: success

      ! check whether init has ever been called
      if (.not.(elpa_initialized())) then
        write(error_unit, *) "elpa_create(): you must call elpa_init() once before creating instances of ELPA"
        success = ELPA_ERROR
        return
      endif

      obj%options     = elpa_allocate_options()
      obj%na          = na
      obj%nev         = nev
      obj%local_nrows = local_nrows
      obj%local_ncols = local_ncols
      obj%nblk        = nblk

      !obj%mpi_comm_rows = mpi_comm_rows
      !obj%mpi_comm_cols = mpi_comm_rows
      success = ELPA_OK

    end function
    subroutine set_comm_rows(self, mpi_comm_rows)
      use iso_c_binding
      implicit none

      integer, intent(in) :: mpi_comm_rows
      class(elpa_t)       :: self
      self%mpi_comm_rows = mpi_comm_rows

    end subroutine

    subroutine set_comm_cols(self, mpi_comm_cols)
      use iso_c_binding
      implicit none

      integer, intent(in) :: mpi_comm_cols
      class(elpa_t)       :: self

      self%mpi_comm_cols = mpi_comm_cols
    end subroutine

    subroutine elpa_set_integer(self, name, value, success)
      use iso_c_binding
      use elpa_generated_fortran_interfaces
      use elpa_utilities, only : error_unit
      implicit none
      class(elpa_t)                   :: self
      character(*), intent(in)        :: name
      integer(kind=c_int), intent(in) :: value
      integer, optional               :: success
      integer                         :: actual_success

      actual_success = elpa_set_int_entry(self%options, name // c_null_char, value)

      if (present(success)) then
        success = actual_success

      else if (actual_success /= ELPA_OK) then
        write(error_unit,'(a,a,i0,a)') "ELPA: Error setting option '", name, "' to value ", value, &
                "and you did not check for errors!"

      end if
    end subroutine


    function elpa_get_integer(self, name, success) result(value)
      use iso_c_binding
      use elpa_generated_fortran_interfaces
      implicit none
      class(elpa_t)                  :: self
      character(*), intent(in)       :: name
      integer(kind=c_int)            :: value
      integer, intent(out), optional :: success
      integer(kind=c_int), target    :: int_success
      type(c_ptr) :: c_success_ptr

      value = elpa_get_int_entry(self%options, name // c_null_char, success)

    end function


    subroutine get_communicators(self, mpi_comm_rows, mpi_comm_cols)
      use iso_c_binding
      implicit none
      class(elpa_t)                    :: self

      integer(kind=c_int), intent(out) :: mpi_comm_rows, mpi_comm_cols
      mpi_comm_rows = self%mpi_comm_rows
      mpi_comm_cols = self%mpi_comm_cols
    end subroutine


    subroutine elpa_solve_real_double(self, a, ev, q, success)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit

      use iso_c_binding
      implicit none
      class(elpa_t)       :: self

#ifdef USE_ASSUMED_SIZE
      real(kind=c_double) :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      real(kind=c_double) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_double) :: ev(self%na)

      real(kind=c_double) :: time_evp_fwd, time_evp_solve, time_evp_back
      integer, optional   :: success
      integer(kind=c_int) :: success_internal
      logical             :: success_l, summary_timings

      logical             :: useGPU, useQR
      integer(kind=c_int) :: THIS_ELPA_KERNEL_API

      if (self%get("summary_timings",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry summary timings"
          stop
        endif

        summary_timings = .true.
      else
        summary_timings = .false.
      endif


      if (self%get("gpu",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry gpu"
          stop
        endif

        useGPU = .true.
      else
        useGPU = .false.
      endif

      if (self%get("qr",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry qr"
          stop
        endif

        useQR = .true.
      else
        useQR = .false.
      endif


      THIS_ELPA_KERNEL_API = self%get("real_kernel",success_internal)
      if (success_internal .ne. ELPA_OK) then
        print *,"Could not querry kernel"
        stop
      endif


      if (self%get("solver",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry solver"
          stop
        endif
        success_l = elpa_solve_evp_real_1stage_double_impl(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%mpi_comm_rows, self%mpi_comm_cols,         &
                                                          self%mpi_comm_parent, useGPU, time_evp_fwd,     &
                                                          time_evp_solve, time_evp_back, summary_timings)

      else if (self%get("solver",success_internal) .eq. 2) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry solver"
          stop
        endif
        success_l = elpa_solve_evp_real_2stage_double_impl(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%mpi_comm_rows, self%mpi_comm_cols,         &
                                                          self%mpi_comm_parent, time_evp_fwd,     &
                                                          time_evp_solve, time_evp_back, summary_timings, useGPU, &
                                                          THIS_ELPA_KERNEL_API, useQR)
      else
        print *,"unknown solver"
        stop
      endif

      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in solve() and you did not check for errors!"
      endif

      if (self%get("summary_timings",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry summary timings"
          stop
        endif

        self%time_evp_fwd = time_evp_fwd
        self%time_evp_solve = time_evp_solve
        self%time_evp_back = time_evp_back
      else

        self%time_evp_fwd = -1.0
        self%time_evp_solve = -1.0
        self%time_evp_back = -1.0
      endif
    end subroutine

    subroutine elpa_solve_real_single(self, a, ev, q, success)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit

      use iso_c_binding
      implicit none
      class(elpa_t)       :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=c_float)  :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      real(kind=c_float)  :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_float)  :: ev(self%na)

      real(kind=c_double) :: time_evp_fwd, time_evp_solve, time_evp_back
      integer, optional   :: success
      integer(kind=c_int) :: success_internal
      logical             :: success_l, summary_timings

      logical             :: useGPU, useQR
      integer(kind=c_int) :: THIS_ELPA_KERNEL_API

#ifdef WANT_SINGLE_PRECISION_REAL
      if (self%get("timings",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry summary timings"
          stop
        endif

        summary_timings = .true.
      else
        summary_timings = .false.
      endif

      if (self%get("gpu",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry gpu"
          stop
        endif

        useGPU = .true.
      else
        useGPU = .false.
      endif

      if (self%get("qr",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry qr"
          stop
        endif

        useQR = .true.
      else
        useQR = .false.
      endif

      THIS_ELPA_KERNEL_API = self%get("real_kernel",success_internal)
      if (success_internal .ne. ELPA_OK) then
        print *,"Could not querry kernel"
        stop
      endif

      if (self%get("solver",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry solver"
          stop
        endif
        success_l = elpa_solve_evp_real_1stage_single_impl(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%mpi_comm_rows, self%mpi_comm_cols,         &
                                                          self%mpi_comm_parent, useGPU, time_evp_fwd,     &
                                                          time_evp_solve, time_evp_back, summary_timings)

      else if (self%get("solver",success_internal) .eq. 2) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry solver"
          stop
        endif
        success_l = elpa_solve_evp_real_2stage_single_impl(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%mpi_comm_rows, self%mpi_comm_cols,         &
                                                          self%mpi_comm_parent, time_evp_fwd,     &
                                                          time_evp_solve, time_evp_back, summary_timings, useGPU, &
                                                          THIS_ELPA_KERNEL_API, useQR)
      else
        print *,"unknown solver"
        stop
      endif

      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in solve() and you did not check for errors!"
      endif


      if (self%get("summary_timings",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry summary timings"
          stop
        endif

        self%time_evp_fwd = time_evp_fwd
        self%time_evp_solve = time_evp_solve
        self%time_evp_back = time_evp_back
      else

        self%time_evp_fwd = -1.0
        self%time_evp_solve = -1.0
        self%time_evp_back = -1.0
      endif
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      success = ELPA_ERROR
#endif

    end subroutine


    subroutine elpa_solve_complex_double(self, a, ev, q, success)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit

      use iso_c_binding
      implicit none
      class(elpa_t)                  :: self

#ifdef USE_ASSUMED_SIZE
      complex(kind=c_double_complex) :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      complex(kind=c_double_complex) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=c_double)            :: ev(self%na)

      real(kind=c_double) :: time_evp_fwd, time_evp_solve, time_evp_back

      integer, optional              :: success
      integer(kind=c_int)            :: success_internal
      logical                        :: success_l, summary_timings

      logical                        :: useGPU
      integer(kind=c_int) :: THIS_ELPA_KERNEL_API
      if (self%get("timings",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry summary timings"
          stop
        endif

        summary_timings = .true.
      else
        summary_timings = .false.
      endif

      if (self%get("gpu",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry gpu"
          stop
        endif

        useGPU = .true.
      else
        useGPU = .false.
      endif

      THIS_ELPA_KERNEL_API = self%get("complex_kernel",success_internal)
      if (success_internal .ne. ELPA_OK) then
        print *,"Could not querry kernel"
        stop
      endif

      if (self%get("solver",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry solver"
          stop
        endif
        success_l = elpa_solve_evp_complex_1stage_double_impl(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%mpi_comm_rows, self%mpi_comm_cols,         &
                                                          self%mpi_comm_parent, useGPU, time_evp_fwd,     &
                                                          time_evp_solve, time_evp_back, summary_timings)

      else if (self%get("solver",success_internal) .eq. 2) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry solver"
          stop
        endif
        success_l = elpa_solve_evp_complex_2stage_double_impl(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%mpi_comm_rows, self%mpi_comm_cols,         &
                                                          self%mpi_comm_parent, time_evp_fwd,     &
                                                          time_evp_solve, time_evp_back, summary_timings, useGPU, &
                                                          THIS_ELPA_KERNEL_API)
      else
        print *,"unknown solver"
        stop
      endif

      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in solve() and you did not check for errors!"
      endif

      if (self%get("summary_timings",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry summary timings"
          stop
        endif

        self%time_evp_fwd = time_evp_fwd
        self%time_evp_solve = time_evp_solve
        self%time_evp_back = time_evp_back
      else

        self%time_evp_fwd = -1.0
        self%time_evp_solve = -1.0
        self%time_evp_back = -1.0
      endif
    end subroutine


    subroutine elpa_solve_complex_single(self, a, ev, q, success)
      use elpa2_impl
      use elpa1_impl
      use elpa_utilities, only : error_unit

      use iso_c_binding
      use precision
      implicit none
      class(elpa_t)                 :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=ck4)             :: a(self%local_nrows, *), q(self%local_nrows, *)
#else
      complex(kind=ck4)             :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
      real(kind=rk4)                :: ev(self%na)

      real(kind=c_double) :: time_evp_fwd, time_evp_solve, time_evp_back
      integer, optional             :: success
      integer(kind=c_int)           :: success_internal
      logical                       :: success_l, summary_timings

      logical                       :: useGPU
      integer(kind=c_int) :: THIS_ELPA_KERNEL_API
#ifdef WANT_SINGLE_PRECISION_COMPLEX

      if (self%get("summary_timings",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry summary timings"
          stop
        endif

        summary_timings = .true.
      else
        summary_timings = .false.
      endif

      if (self%get("gpu",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry gpu"
          stop
        endif

        useGPU = .true.
      else
        useGPU = .false.
      endif

      THIS_ELPA_KERNEL_API = self%get("complex_kernel",success_internal)
      if (success_internal .ne. ELPA_OK) then
        print *,"Could not querry kernel"
        stop
      endif

      if (self%get("solver",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry solver"
          stop
        endif
        success_l = elpa_solve_evp_complex_1stage_single_impl(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%mpi_comm_rows, self%mpi_comm_cols,         &
                                                          self%mpi_comm_parent, useGPU, time_evp_fwd,     &
                                                          time_evp_solve, time_evp_back, summary_timings)

      else if (self%get("solver",success_internal) .eq. 2) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry solver"
          stop
        endif
        success_l = elpa_solve_evp_complex_2stage_single_impl(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%mpi_comm_rows, self%mpi_comm_cols,         &
                                                          self%mpi_comm_parent,  time_evp_fwd,     &
                                                          time_evp_solve, time_evp_back, summary_timings, useGPU, &
                                                          THIS_ELPA_KERNEL_API)
      else
        print *,"unknown solver"
        stop
      endif

      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in solve() and you did not check for errors!"
      endif

      if (self%get("summary_timings",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry summary timings"
          stop
        endif

        self%time_evp_fwd = time_evp_fwd
        self%time_evp_solve = time_evp_solve
        self%time_evp_back = time_evp_back
      else

        self%time_evp_fwd = -1.0
        self%time_evp_solve = -1.0
        self%time_evp_back = -1.0
      endif

#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      success = ELPA_ERROR
#endif


    end subroutine

    subroutine elpa_multiply_at_b_double (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                          c, ldc, ldcCols, success)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      implicit none
      class(elpa_t)                   :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=ik), intent(in)    :: na, lda, ldaCols, ldb, ldbCols, ldc, ldcCols, ncb
#ifdef USE_ASSUMED_SIZE
      real(kind=rk8)                  :: a(lda,*), b(ldb,*), c(ldc,*)
#else
      real(kind=rk8)                  :: a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)
#endif
      integer, optional               :: success
      logical                         :: success_l

      success_l = elpa_mult_at_b_real_double_impl(uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, self%nblk, &
                              self%mpi_comm_rows, self%mpi_comm_cols, c, ldc, ldcCols)
      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in multiply_a_b() and you did not check for errors!"
      endif
    end subroutine

    subroutine elpa_multiply_at_b_single (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                          c, ldc, ldcCols, success)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      implicit none
      class(elpa_t)                   :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=ik), intent(in)    :: na, lda, ldaCols, ldb, ldbCols, ldc, ldcCols, ncb
#ifdef USE_ASSUMED_SIZE
      real(kind=rk4)                  :: a(lda,*), b(ldb,*), c(ldc,*)
#else
      real(kind=rk4)                  :: a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)
#endif
      integer, optional               :: success
      logical                         :: success_l
#ifdef WANT_SINGLE_PRECISION_REAL
      success_l = elpa_mult_at_b_real_single_impl(uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, self%nblk, &
                              self%mpi_comm_rows, self%mpi_comm_cols, c, ldc, ldcCols)
      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in multiply_a_b() and you did not check for errors!"
      endif
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      success = ELPA_ERROR
#endif
    end subroutine

    subroutine elpa_multiply_ah_b_double (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                          c, ldc, ldcCols, success)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      implicit none
      class(elpa_t)                   :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=ik), intent(in)    :: na, lda, ldaCols, ldb, ldbCols, ldc, ldcCols, ncb
#ifdef USE_ASSUMED_SIZE
      complex(kind=ck8)               :: a(lda,*), b(ldb,*), c(ldc,*)
#else
      complex(kind=ck8)               :: a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)
#endif
      integer, optional               :: success
      logical                         :: success_l

      success_l = elpa_mult_ah_b_complex_double_impl(uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, self%nblk, &
                              self%mpi_comm_rows, self%mpi_comm_cols, c, ldc, ldcCols)
      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in multiply_a_b() and you did not check for errors!"
      endif
    end subroutine

    subroutine elpa_multiply_ah_b_single (self,uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, &
                                          c, ldc, ldcCols, success)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      implicit none
      class(elpa_t)                   :: self
      character*1                     :: uplo_a, uplo_c
      integer(kind=ik), intent(in)    :: na, lda, ldaCols, ldb, ldbCols, ldc, ldcCols, ncb
#ifdef USE_ASSUMED_SIZE
      complex(kind=ck4)               :: a(lda,*), b(ldb,*), c(ldc,*)
#else
      complex(kind=ck4)               :: a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)
#endif
      integer, optional               :: success
      logical                         :: success_l

#ifdef WANT_SINGLE_PRECISION_COMPLEX
      success_l = elpa_mult_ah_b_complex_single_impl(uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, self%nblk, &
                              self%mpi_comm_rows, self%mpi_comm_cols, c, ldc, ldcCols)
      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in multiply_a_b() and you did not check for errors!"
      endif
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      success = ELPA_ERROR
#endif
    end subroutine

    subroutine elpa_cholesky_double_real (self, a, success)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      implicit none
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=rk8)                  :: a(self%local_nrows,*)
#else
      real(kind=rk8)                  :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
      logical                         :: success_l
      integer(kind=c_int)             :: success_internal
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif


      success_l = elpa_cholesky_real_double_impl (self%na, a, self%local_nrows, self%nblk, &
                                                 self%local_ncols, self%mpi_comm_rows, self%mpi_comm_cols, &
                                                 wantDebugIntern)
      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in cholesky() and you did not check for errors!"
      endif
    end subroutine

    subroutine elpa_cholesky_single_real(self, a, success)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      implicit none
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=rk4)                  :: a(self%local_nrows,*)
#else
      real(kind=rk4)                  :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
      logical                         :: success_l
      integer(kind=c_int)             :: success_internal
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif

#if WANT_SINGLE_PRECISION_REAL
      success_l = elpa_cholesky_real_single_impl (self%na, a, self%local_nrows, self%nblk, &
                                                 self%local_ncols, self%mpi_comm_rows, self%mpi_comm_cols, &
                                                 wantDebugIntern)
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      success = ELPA_ERROR
#endif
      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in cholesky() and you did not check for errors!"
      endif
    end subroutine

    subroutine elpa_cholesky_double_complex (self, a, success)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      implicit none
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=ck8)               :: a(self%local_nrows,*)
#else
      complex(kind=ck8)               :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
      logical                         :: success_l
      integer(kind=c_int)             :: success_internal
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif

      success_l = elpa_cholesky_complex_double_impl (self%na, a, self%local_nrows, self%nblk, &
                                                 self%local_ncols, self%mpi_comm_rows, self%mpi_comm_cols, &
                                                 wantDebugIntern)
      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in cholesky() and you did not check for errors!"
      endif
    end subroutine

    subroutine elpa_cholesky_single_complex (self, a, success)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      implicit none
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=ck4)               :: a(self%local_nrows,*)
#else
      complex(kind=ck4)               :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
      logical                         :: success_l
      integer(kind=c_int)             :: success_internal
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif
#if WANT_SINGLE_PRECISION_COMPLEX
      success_l = elpa_cholesky_complex_single_impl (self%na, a, self%local_nrows, self%nblk, &
                                                 self%local_ncols, self%mpi_comm_rows, self%mpi_comm_cols, &
                                                 wantDebugIntern)
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      success = ELPA_ERROR
#endif
      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in cholesky() and you did not check for errors!"
      endif
    end subroutine

    subroutine elpa_invert_trm_double_real (self, a, success)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      implicit none
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=rk8)                  :: a(self%local_nrows,*)
#else
      real(kind=rk8)                  :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
      logical                         :: success_l
      integer(kind=c_int)             :: success_internal
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif
      success_l = elpa_invert_trm_real_double_impl (self%na, a, self%local_nrows, self%nblk, &
                                                   self%local_ncols, self%mpi_comm_rows, self%mpi_comm_cols, &
                                                   wantDebugIntern)
      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in invert_trm() and you did not check for errors!"
      endif
    end subroutine

    subroutine elpa_invert_trm_single_real (self, a, success)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      implicit none
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      real(kind=rk4)                  :: a(self%local_nrows,*)
#else
      real(kind=rk4)                  :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
      logical                         :: success_l
      integer(kind=c_int)             :: success_internal
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif
#if WANT_SINGLE_PRECISION_REAL
      success_l = elpa_invert_trm_real_single_impl (self%na, a, self%local_nrows, self%nblk, &
                                                   self%local_ncols, self%mpi_comm_rows, self%mpi_comm_cols, &
                                                   wantDebugIntern)
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      success = ELPA_ERROR
#endif
      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in invert_trm() and you did not check for errors!"
      endif
    end subroutine

    subroutine elpa_invert_trm_double_complex (self, a, success)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      implicit none
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=ck8)               :: a(self%local_nrows,*)
#else
      complex(kind=ck8)               :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
      logical                         :: success_l
      integer(kind=c_int)             :: success_internal
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif
      success_l = elpa_invert_trm_complex_double_impl (self%na, a, self%local_nrows, self%nblk, &
                                                   self%local_ncols, self%mpi_comm_rows, self%mpi_comm_cols, &
                                                   wantDebugIntern)
      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in invert_trm() and you did not check for errors!"
      endif
    end subroutine

    subroutine elpa_invert_trm_single_complex (self, a, success)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      implicit none
      class(elpa_t)                   :: self
#ifdef USE_ASSUMED_SIZE
      complex(kind=ck4)               :: a(self%local_nrows,*)
#else
      complex(kind=ck4)               :: a(self%local_nrows,self%local_ncols)
#endif
      integer, optional               :: success
      logical                         :: success_l
      integer(kind=c_int)             :: success_internal
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif
#if WANT_SINGLE_PRECISION_COMPLEX
      success_l = elpa_invert_trm_complex_single_impl (self%na, a, self%local_nrows, self%nblk, &
                                                   self%local_ncols, self%mpi_comm_rows, self%mpi_comm_cols, &
                                                   wantDebugIntern)
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      success = ELPA_ERROR
#endif
      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in invert_trm() and you did not check for errors!"
      endif
    end subroutine

    subroutine elpa_solve_tridi_double_real (self, d, e, q, success)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      implicit none
      class(elpa_t)                   :: self
      real(kind=rk8)                  :: d(self%na), e(self%na)
#ifdef USE_ASSUMED_SIZE
      real(kind=rk8)                  :: q(self%local_nrows,*)
#else
      real(kind=rk8)                  :: q(self%local_nrows,self%local_ncols)
#endif

      integer, optional               :: success
      logical                         :: success_l
      integer(kind=c_int)             :: success_internal
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif

      success_l = elpa_solve_tridi_double_impl(self%na, self%nev, d, e, q, self%local_nrows, self%nblk, &
                                              self%local_ncols, self%mpi_comm_rows, self%mpi_comm_cols,&
                                              wantDebugIntern)
      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in solve_tridi() and you did not check for errors!"
      endif

    end subroutine

    subroutine elpa_solve_tridi_single_real (self, d, e, q, success)
      use iso_c_binding
      use elpa1_auxiliary_impl
      use precision
      implicit none
      class(elpa_t)                   :: self
      real(kind=rk4)                  :: d(self%na), e(self%na)
#ifdef USE_ASSUMED_SIZE
      real(kind=rk4)                  :: q(self%local_nrows,*)
#else
      real(kind=rk4)                  :: q(self%local_nrows,self%local_ncols)
#endif

      integer, optional               :: success
      logical                         :: success_l
      integer(kind=c_int)             :: success_internal
      logical                         :: wantDebugIntern

      if (self%get("wantDebug",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry wantDebug"
          stop
        endif

        wantDebugIntern = .true.
      else
        wantDebugIntern = .false.
      endif
#ifdef WANT_SINGLE_PRECISION_REAL
      success_l = elpa_solve_tridi_single_impl(self%na, self%nev, d, e, q, self%local_nrows, self%nblk, &
                                              self%local_ncols, self%mpi_comm_rows, self%mpi_comm_cols,&
                                              wantDebugIntern)
#else
      print *,"This installation of the ELPA library has not been build with single-precision support"
      success = ELPA_ERROR
#endif
      if (present(success)) then
        if (success_l) then
          success = ELPA_OK
        else
          success = ELPA_ERROR
        endif
      else if (.not. success_l) then
        write(error_unit,'(a)') "ELPA: Error in solve_tridi() and you did not check for errors!"
      endif

    end subroutine



    subroutine elpa_destroy(self)
      use elpa_generated_fortran_interfaces
      class(elpa_t) :: self
      call elpa_free_index(self%options)
    end subroutine

end module

