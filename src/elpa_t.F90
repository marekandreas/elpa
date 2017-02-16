module elpa_type
 use iso_c_binding
 use init_elpa

 private

 public :: elpa_init, elpa_initialized, elpa_uninit, elpa_create, elpa_t, C_INT, C_DOUBLE, C_FLOAT

 type :: elpa_t
   private
   integer(kind=c_int) :: mpi_comm_parent, mpi_comm_rows, mpi_comm_cols
   integer(kind=c_int) :: na, nev, local_nrows, local_ncols, nblk
   contains
     generic, public :: set => elpa_set_string, elpa_set_integer
     generic, public :: get => elpa_get_string, elpa_get_integer

     procedure, private :: elpa_set_string, elpa_set_integer
     procedure, private :: elpa_get_string, elpa_get_integer

     procedure, public :: get_communicators => get_communicators
     procedure, public :: solve => elpa_solve_real_double
 end type elpa_t

 contains

   function elpa_create(self, na, nev, local_nrows, local_ncols, nblk, mpi_comm_parent, process_row, process_col) result(success)
      use precision
      use init_elpa
      use elpa_mpi
      use elpa_utilities, only : error_unit
      use elpa2_utilities, only : DEFAULT_REAL_ELPA_KERNEL, DEFAULT_COMPLEX_ELPA_KERNEL
      use elpa1, only : elpa_get_communicators
      implicit none

      integer(kind=ik), intent(in) :: na, nev, local_nrows, local_ncols, nblk
      integer, intent(in) :: mpi_comm_parent, process_row, process_col
      type(elpa_t), intent(out) :: self
      integer :: mpierr

      logical :: success

      success = .true.

      ! check whether init has ever been called
      if (.not.(elpa_initialized())) then
        write(error_unit, *) "elpa_create(): you must call elpa_init() once before creating instances of ELPA"
        success = .false.
        return
      endif

      self%na          = na
      self%nev         = nev
      self%local_nrows = local_nrows
      self%local_ncols = local_ncols
      self%nblk        = nblk

      self%mpi_comm_parent = mpi_comm_parent
      mpierr = elpa_get_communicators(mpi_comm_parent, process_row, process_col, self%mpi_comm_rows, self%mpi_comm_cols)
      if (mpierr /= MPI_SUCCESS) then
        write(error_unit, *) "elpa_create(): error constructing row and column communicators"
        success = .false.
        return
      endif

    end function

    function elpa_set_string(self, keyword, value) result(success)
      use iso_c_binding
      use elpa1, only : elpa_print_times
      implicit none
      class(elpa_t)            :: self
      character(*), intent(in) :: keyword
      character(*), intent(in) :: value
      logical                  :: success

      success = .false.
    end function elpa_set_string

    function elpa_set_integer(self, keyword, value) result(success)
      use iso_c_binding
      use elpa2_utilities, only : check_allowed_real_kernels, check_allowed_complex_kernels
      implicit none
      class(elpa_t)                   :: self
      character(*), intent(in)        :: keyword
      integer(kind=c_int), intent(in) :: value
      logical                         :: success

      success = .false.
    end function elpa_set_integer

    function elpa_get_string(self, keyword, value) result(success)
      use iso_c_binding
      use elpa1, only : elpa_print_times
      implicit none
      class(elpa_t)               :: self
      character(*), intent(in)    :: keyword
      character(*), intent(inout) :: value
      logical                     :: success

      success = .false.
    end function elpa_get_string

    function elpa_get_integer(self, keyword, value) result(success)
      use iso_c_binding
      implicit none
      class(elpa_t)                      :: self
      character(*), intent(in)           :: keyword
      integer(kind=c_int), intent(inout) :: value
      logical                            :: success

      success = .false.
    end function elpa_get_integer

    subroutine get_communicators(self, mpi_comm_rows, mpi_comm_cols)
      use iso_c_binding
      implicit none
      class(elpa_t)                   :: self

      integer(kind=c_int), intent(out) :: mpi_comm_rows, mpi_comm_cols
      mpi_comm_rows = self%mpi_comm_rows
      mpi_comm_cols = self%mpi_comm_cols
    end subroutine

    function elpa_solve_real_double(self, a, ev, q) result(success)
      use elpa

      use iso_c_binding
      implicit none
      class(elpa_t)                   :: self

      real(kind=c_double) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols), &
                             ev(self%na)
      logical :: success

      success = elpa_solve_evp_real_double(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                           self%local_nrows,  self%nblk, self%local_ncols, &
                                           self%mpi_comm_rows, self%mpi_comm_cols,         &
                                           self%mpi_comm_parent)
    end function


end module

