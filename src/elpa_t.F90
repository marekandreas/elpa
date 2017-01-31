module elpa_type
 use iso_c_binding

 private

 public :: elpa_create, elpa_t

 type :: elpa_t
   integer(kind=c_int) :: mpi_comm_rows, mpi_comm_cols, mpi_comm_global
   integer(kind=c_int) :: na, nev, local_nrows, local_ncols, nblk

   integer(kind=c_int) :: real_kernel, complex_kernel

   integer(kind=c_int) :: useQR, useGPU
   character(6)        :: solver
   character(8)        :: timings
   contains
     generic, public :: set_option => elpa_set_option_string, elpa_set_option_integer
     procedure, private :: elpa_set_option_string, elpa_set_option_integer
     generic, public :: get_option => elpa_get_option_string, elpa_get_option_integer
     procedure, private :: elpa_get_option_string, elpa_get_option_integer

     procedure :: get_communicators => get_communicators
     procedure :: solve_real_double => elpa_solve_real_double

 end type elpa_t

 contains



   function elpa_create(na, nev, local_nrows, local_ncols, nblk) result(elpa)
     use precision
     use init_elpa
     use elpa2_utilities, only : DEFAULT_REAL_ELPA_KERNEL, DEFAULT_COMPLEX_ELPA_KERNEL
     implicit none

      integer(kind=ik), intent(in) :: na, nev, local_nrows, local_ncols, nblk
      type(elpa_t) :: elpa

      ! check whether init has ever been called
      if (.not.(initDone))  then
        print *,"ERROR: you must call elpa_init() once before creating instances of ELPA"
        stop
      endif

      elpa%na          = na
      elpa%nev         = nev
      elpa%local_nrows = local_nrows
      elpa%local_ncols = local_ncols
      elpa%nblk        = nblk

      ! some default values
      elpa%solver         = "2stage"
      elpa%real_kernel    = DEFAULT_REAL_ELPA_KERNEL
      elpa%complex_kernel = DEFAULT_COMPLEX_ELPA_KERNEL

      elpa%useQR          = 0
      elpa%useGPU         = 0
      elpa%timings        = "none"

    end function

    function elpa_set_option_string(self, keyword, value) result(success)
      use iso_c_binding
      use elpa1, only : elpa_print_times
      implicit none
      class(elpa_t)            :: self
      character(*), intent(in) :: keyword
      character(*), intent(in) :: value
      integer(kind=c_int)      :: success

      success = 0

      if (trim(keyword) .eq. "solver") then
        if (trim(value) .eq. "1stage") then
          self%solver = "1stage"
          success = 1
        else if (trim(value) .eq. "2stage") then
          self%solver = "2stage"
          success = 1
        else if (trim(value) .eq. "auto") then
          self%solver = "auto "
          success = 1
        else
          print *," not allowed key/value pair: ",trim(keyword),"/",trim(value)
          success = 0
        endif
      else if (trim(keyword) .eq. "timings") then
        if (trim(value) .eq. "balanced") then
          elpa_print_times = .true.
          success = 1
        else if (trim(value) .eq. "detailed") then
          print *,"detailed timings not yet implemented"
          elpa_print_times = .false.
          success = 1
        else if (trim(value) .eq. "none") then
          elpa_print_times = .false.
          success = 1
        else
          print *," not allowed key/value pair: ",trim(keyword),"/",trim(value)
          success = 0
        endif
      else
        print *," not allowed key/value pair: ",trim(keyword),"/",trim(value)
        success = 0
      endif

    end function elpa_set_option_string

    function elpa_set_option_integer(self, keyword, value) result(success)
      use iso_c_binding
      use elpa2_utilities, only : check_allowed_real_kernels, check_allowed_complex_kernels
      implicit none
      class(elpa_t)                   :: self
      character(*), intent(in)        :: keyword
      integer(kind=c_int), intent(in) :: value
      integer(kind=c_int)             :: success

      success = 0

      if (trim(keyword) .eq. "real_kernel") then
        if (.not.(check_allowed_real_kernels(value))) then
          self%real_kernel = value
          success = 1
        else
          print *,"Setting this real_kernel is not possible"
          success = 0
        endif
      else if (trim(keyword) .eq. "complex_kernel" ) then
        if (.not.(check_allowed_complex_kernels(value))) then
          self%complex_kernel = value
          success = 1
        else
          print *,"Setting this complex_kernel is not possible"
          success = 0
        endif
      else if (trim(keyword) .eq. "use_qr") then
        if (value .eq. 1) then
          self%useQr = 1
          success = 1
        else if (value .eq. 0) then
          self%useQr = 0
          success = 1
        else
          print *," not allowed key/value pair: ",trim(keyword),"/",value
          success = 0
        endif
      else if (trim(keyword) .eq. "use_gpu") then
        if (value .eq. 1) then
          self%useGPU = 1
          success = 1
        else if (value .eq. 0) then
          self%useGPU = 0
          success = 1
        else
          print *," not allowed key/value pair: ",trim(keyword),"/",value
          success = 0
        endif
      else
        print *," not allowed key/value pair: ",trim(keyword),"/",value
        success = 0
      endif

    end function elpa_set_option_integer

    function elpa_get_option_string(self, keyword, value) result(success)
      use iso_c_binding
      use elpa1, only : elpa_print_times
      implicit none
      class(elpa_t)               :: self
      character(*), intent(in)    :: keyword
      character(*), intent(inout) :: value
      integer(kind=c_int)         :: success

      success = 0

      if (trim(keyword) .eq. "solver") then
        value = trim(self%solver)
        success = 1
      else if (trim(keyword) .eq. "timings") then
        if (elpa_print_times) then
          value = "balanced"
          success = 1
        else
          ! detailed not yet implemented
          success = 1
        endif
      else
        print *," not allowed key/value pair: ",trim(keyword),"/",trim(value)
        success = 0
      endif

    end function elpa_get_option_string

    function elpa_get_option_integer(self, keyword, value) result(success)
      use iso_c_binding
      implicit none
      class(elpa_t)                      :: self
      character(*), intent(in)           :: keyword
      integer(kind=c_int), intent(inout) :: value
      integer(kind=c_int)                :: success

      success = 0

      if (trim(keyword) .eq. "real_kernel") then
        value = self%real_kernel
        success = 1
      else if (trim(keyword) .eq. "complex_kernel" ) then
        value = self%complex_kernel
        success = 1
      else if (trim(keyword) .eq. "use_qr") then
        value = self%useQr
        success = 1
      else if (trim(keyword) .eq. "use_gpu") then
        value =  self%useGPU
        success = 1
      else
        print *," not allowed key/value pair: ",trim(keyword),"/",value
        success = 0
      endif

    end function elpa_get_option_integer

    function get_communicators(self, mpi_comm_global, my_prow, my_pcol, mpi_comm_rows, mpi_comm_cols) result(mpierr)
      use iso_c_binding
      use elpa_mpi
      use elpa1, only : elpa_get_communicators
      implicit none
      class(elpa_t)                    :: self

      integer(kind=c_int), intent(in)  :: mpi_comm_global, my_prow, my_pcol
      integer(kind=c_int), intent(out) :: mpi_comm_rows, mpi_comm_cols

      integer(kind=c_int)              :: mpierr

      mpierr = elpa_get_communicators(mpi_comm_global, my_prow, my_pcol, mpi_comm_rows, mpi_comm_cols)

      self%mpi_comm_rows   = mpi_comm_rows
      self%mpi_comm_cols   = mpi_comm_cols
      self%mpi_comm_global = mpi_comm_global
    end function

    function elpa_solve_real_double(self, a, ev, q) result(success)
      use elpa

      use iso_c_binding
      implicit none
      class(elpa_t)                    :: self

      real(kind=c_double) :: a(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols), &
                             ev(self%na)
      integer(kind=c_int) :: success

      logical :: successDummy

      successDummy =  elpa_solve_evp_real_double(self%na, self%nev, a, self%local_nrows, ev, q,  &
                                                 self%local_nrows,  self%nblk, self%local_ncols, &
                                                 self%mpi_comm_rows, self%mpi_comm_cols,         &
                                                 self%mpi_comm_global, method=trim(self%solver))

      if (successDummy) then
        success = 1
      else
        success = 0
      endif
    end function


end module

