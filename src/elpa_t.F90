#include <elpa/elpa_constants.h>

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
  integer, parameter :: ELPA_2STAGE_REAL_AVX2_BLOCK6       = ELPA_C_2STAGE_REAL_AVX2_BLOCK6
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

  type :: elpa_t
   private
   type(c_ptr) :: options = C_NULL_PTR
   integer :: mpi_comm_parent = 0
   integer(kind=c_int) :: mpi_comm_rows = 0
   integer(kind=c_int) :: mpi_comm_cols = 0
   integer(kind=c_int) :: na = 0
   integer(kind=c_int) :: nev = 0
   integer(kind=c_int) :: local_nrows = 0
   integer(kind=c_int) :: local_ncols = 0
   integer(kind=c_int) :: nblk = 0
   contains
     generic, public :: set => elpa_set_integer
     generic, public :: get => elpa_get_integer

     procedure, public :: get_communicators => get_communicators
     generic, public :: solve => elpa_solve_real_double, &
                                 elpa_solve_real_single, &
                                 elpa_solve_complex_double, &
                                 elpa_solve_complex_single


     procedure, public :: destroy => elpa_destroy

     ! privates:
     procedure, private :: elpa_set_integer
     procedure, private :: elpa_get_integer
     procedure, private :: elpa_solve_real_double
     procedure, private :: elpa_solve_real_single
     procedure, private :: elpa_solve_complex_double
     procedure, private :: elpa_solve_complex_single

  end type elpa_t

  logical :: initDone = .false.

  integer, parameter :: earliest_api_version = 20170403
  integer, parameter :: current_api_version  = 20170403

  interface
    function elpa_allocate_options() result(options) bind(C, name="elpa_allocate_options")
      import c_ptr
      type(c_ptr) :: options
    end function
  end interface


  interface
    subroutine elpa_free_options(options) bind(C, name="elpa_free_options")
      import c_ptr
      type(c_ptr), value :: options
    end subroutine
  end interface


  interface
    function get_int_option(options, name, success) result(value) bind(C, name="get_int_option")
      import c_ptr, c_char, c_int
      type(c_ptr), value :: options
      character(kind=c_char), intent(in) :: name(*)
      integer(kind=c_int) :: value
      integer(kind=c_int), optional :: success
    end function
  end interface


  interface
    function set_int_option(options, name, value) result(success) bind(C, name="set_int_option")
      import c_ptr, c_char, c_int
      type(c_ptr), value :: options
      character(kind=c_char), intent(in) :: name(*)
      integer(kind=c_int), intent(in), value :: value
      integer(kind=c_int) :: success
    end function
  end interface


  contains


    function elpa_init(api_version) result(success)
      use elpa_utilities, only : error_unit
      implicit none
      integer, intent(in) :: api_version
      integer :: success

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


    function elpa_create(na, nev, local_nrows, local_ncols, nblk, mpi_comm_parent, process_row, process_col, success) result(obj)
      use precision
      use elpa_mpi
      use elpa_utilities, only : error_unit
      use elpa1, only : elpa_get_communicators
      implicit none

      integer(kind=ik), intent(in) :: na, nev, local_nrows, local_ncols, nblk
      integer, intent(in) :: mpi_comm_parent, process_row, process_col
      type(elpa_t) :: obj
      integer :: mpierr

      integer, optional :: success

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
      mpierr = elpa_get_communicators(mpi_comm_parent, process_row, process_col, obj%mpi_comm_rows, obj%mpi_comm_cols)
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


    subroutine elpa_set_integer(self, name, value, success)
      use iso_c_binding
      use elpa_utilities, only : error_unit
      implicit none
      class(elpa_t)                   :: self
      character(*), intent(in)        :: name
      integer(kind=c_int), intent(in) :: value
      integer, optional               :: success
      integer                         :: actual_success

      actual_success = set_int_option(self%options, name // c_null_char, value)

      if (present(success)) then
        success = actual_success

      else if (actual_success /= ELPA_OK) then
        write(error_unit,'(a,a,i0,a)') "ELPA: Error setting option '", name, "' to value ", value, &
                "and you did not check for errors!"

      end if
    end subroutine


    function elpa_get_integer(self, name, success) result(value)
      use iso_c_binding
      implicit none
      class(elpa_t)                  :: self
      character(*), intent(in)       :: name
      integer(kind=c_int)            :: value
      integer, intent(out), optional :: success
      integer(kind=c_int), target    :: int_success
      type(c_ptr) :: c_success_ptr

      value = get_int_option(self%options, name // c_null_char, success)

    end function


    subroutine get_communicators(self, mpi_comm_rows, mpi_comm_cols)
      use iso_c_binding
      implicit none
      class(elpa_t)                   :: self

      integer(kind=c_int), intent(out) :: mpi_comm_rows, mpi_comm_cols
      mpi_comm_rows = self%mpi_comm_rows
      mpi_comm_cols = self%mpi_comm_cols
    end subroutine


    subroutine elpa_solve_real_double(self, a, ev, q, success)
      use elpa2_new
      use elpa1_new
      use elpa_utilities, only : error_unit

      use iso_c_binding
      implicit none
      class(elpa_t)       :: self

      real(kind=c_double) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols), &
                             ev(self%na)
      integer, optional   :: success
      integer(kind=c_int) :: success_internal
      logical             :: success_l

      logical             :: useGPU

      if (self%get("gpu",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry gpu"
          stop
        endif

        useGPU = .true.
      else
        useGPU = .false.
      endif

      if (self%get("solver",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry solver"
          stop
        endif
        success_l = elpa_solve_evp_real_1stage_double_new(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%mpi_comm_rows, self%mpi_comm_cols,         &
                                                          self%mpi_comm_parent, useGPU)

      else if (self%get("solver",success_internal) .eq. 2) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry solver"
          stop
        endif
        success_l = elpa_solve_evp_real_2stage_double_new(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%mpi_comm_rows, self%mpi_comm_cols,         &
                                                          self%mpi_comm_parent, useGPU)
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

    end subroutine

    subroutine elpa_solve_real_single(self, a, ev, q, success)
      use elpa2_new
      use elpa1_new
      use elpa_utilities, only : error_unit

      use iso_c_binding
      implicit none
      class(elpa_t)       :: self

      real(kind=c_float)  :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols), &
                             ev(self%na)
      integer, optional   :: success
      integer(kind=c_int) :: success_internal
      logical             :: success_l

      logical             :: useGPU

#ifdef WANT_SINGLE_PRECISION_REAL

      if (self%get("gpu",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry gpu"
          stop
        endif

        useGPU = .true.
      else
        useGPU = .false.
      endif

      if (self%get("solver",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry solver"
          stop
        endif
        success_l = elpa_solve_evp_real_1stage_single_new(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%mpi_comm_rows, self%mpi_comm_cols,         &
                                                          self%mpi_comm_parent, useGPU)

      else if (self%get("solver",success_internal) .eq. 2) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry solver"
          stop
        endif
        success_l = elpa_solve_evp_real_2stage_single_new(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%mpi_comm_rows, self%mpi_comm_cols,         &
                                                          self%mpi_comm_parent, useGPU)
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
#else
      success = ELPA_ERROR
#endif

    end subroutine


    subroutine elpa_solve_complex_double(self, a, ev, q, success)
      use elpa2_new
      use elpa1_new
      use elpa_utilities, only : error_unit

      use iso_c_binding
      implicit none
      class(elpa_t)                  :: self

      complex(kind=c_double_complex) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
      real(kind=c_double)            :: ev(self%na)

      integer, optional              :: success
      integer(kind=c_int)            :: success_internal
      logical                        :: success_l

      logical                        :: useGPU

      if (self%get("gpu",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry gpu"
          stop
        endif

        useGPU = .true.
      else
        useGPU = .false.
      endif

      if (self%get("solver",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry solver"
          stop
        endif
        success_l = elpa_solve_evp_complex_1stage_double_new(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%mpi_comm_rows, self%mpi_comm_cols,         &
                                                          self%mpi_comm_parent, useGPU)

      else if (self%get("solver",success_internal) .eq. 2) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry solver"
          stop
        endif
        success_l = elpa_solve_evp_complex_2stage_double_new(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%mpi_comm_rows, self%mpi_comm_cols,         &
                                                          self%mpi_comm_parent, useGPU)
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

    end subroutine


    subroutine elpa_solve_complex_single(self, a, ev, q, success)
      use elpa2_new
      use elpa1_new
      use elpa_utilities, only : error_unit

      use iso_c_binding
      implicit none
      class(elpa_t)                 :: self

      complex(kind=c_float_complex) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
      real(kind=c_float)            :: ev(self%na)

      integer, optional             :: success
      integer(kind=c_int)           :: success_internal
      logical                       :: success_l

      logical                       :: useGPU

#ifdef WANT_SINGLE_PRECISION_COMPLEX
      if (self%get("gpu",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry gpu"
          stop
        endif

        useGPU = .true.
      else
        useGPU = .false.
      endif

      if (self%get("solver",success_internal) .eq. 1) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry solver"
          stop
        endif
        success_l = elpa_solve_evp_complex_1stage_single_new(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%mpi_comm_rows, self%mpi_comm_cols,         &
                                                          self%mpi_comm_parent, useGPU)

      else if (self%get("solver",success_internal) .eq. 2) then
        if (success_internal .ne. ELPA_OK) then
          print *,"Could not querry solver"
          stop
        endif
        success_l = elpa_solve_evp_complex_2stage_single_new(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                          self%local_nrows,  self%nblk, self%local_ncols, &
                                                          self%mpi_comm_rows, self%mpi_comm_cols,         &
                                                          self%mpi_comm_parent, useGPU)
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
#else
      success = ELPA_ERROR
#endif


    end subroutine


    subroutine elpa_destroy(self)
      class(elpa_t) :: self
      call elpa_free_options(self%options)
    end subroutine

end module

