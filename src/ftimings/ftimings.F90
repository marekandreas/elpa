! Copyright 2014 Lorenz Hüdepohl
!
! This file is part of ftimings.
!
! ftimings is free software: you can redistribute it and/or modify
! it under the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! ftimings is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU Lesser General Public License for more details.
!
! You should have received a copy of the GNU Lesser General Public License
! along with ftimings.  If not, see <http://www.gnu.org/licenses/>.

#ifdef HAVE_CONFIG_H
#include "config-f90.h"
#endif

!> \mainpage Ftimings
!>
!> An almost pure-fortran attempt to play with tree structures, which evolved
!> into the timing library used e.g. by the VERTEX supernova code.
!>
!> All you need to know is contained in the \ref ftimings::timer_t derived type.
module ftimings
  use ftimings_type
  use ftimings_value
  use, intrinsic :: iso_fortran_env, only : error_unit, output_unit
  implicit none
  save

  private

  ! this is mainly needed for Doxygen, they are
  ! by implicitly reachable as type-bound procedures
  ! of timer_t, however Doxygen does not document them
  ! if they are not also public
  public timer_start, timer_stop, timer_free, timer_print, &
         timer_enable, timer_disable, timer_is_enabled, &
         timer_in_entries, timer_get, timer_since, timer_sort, &
         timer_set_print_options, &
         timer_measure_flops, &
         timer_measure_allocated_memory, &
         timer_measure_virtual_memory,   &
         timer_measure_max_allocated_memory,   &
         timer_measure_memory_bandwidth

  character(len=name_length), private, parameter :: own = "(own)"
  character(len=name_length), private, parameter :: below = "(below threshold)"

  !> Type for a timer instance.
  !>
  !> Typical usage:
  !> \code{.f90}
  !>   type(timer_t) :: timer
  !>
  !>   call timer%enable()
  !>
  !>   call timer%start("section")
  !>     ...
  !>   call timer%start("subsection")
  !>     ...
  !>   call timer%stop("subsection")
  !>     ...
  !>   call timer%stop("section")
  !>
  !>   call timer%print()
  !> \endcode
  !>
  !> Every first call to timer%start() at a certain point in the graph
  !> allocates a small amount of memory. If the timer is no longer needed,
  !> all that memory can be freed again with
  !>
  !> \code{.f90}
  !>   call timer%free()
  !> \endcode
  type, public :: timer_t
    logical, private :: active = .false.                         !< If set to .false., most operations return immediately without any action
    logical, private :: record_allocated_memory = .false.        !< IF set to .true., record also the current resident set size
    logical, private :: record_virtual_memory = .false.          !< IF set to .true., record also the virtual memory
    logical, private :: record_max_allocated_memory = .false.    !< IF set to .true., record also the max resident set size ("high water mark")
    logical, private :: record_flop_counts = .false.             !< If set to .true., record also FLOP counts via PAPI calls
    logical, private :: record_memory_bandwidth = .false.        !< If set to .true., record also FLOP counts via PAPI calls

    logical, private :: print_allocated_memory = .false.
    logical, private :: print_max_allocated_memory = .false.
    logical, private :: print_virtual_memory = .false.
    logical, private :: print_flop_count = .false.
    logical, private :: print_flop_rate = .false.
    logical, private :: print_ldst = .false.
    logical, private :: print_memory_bandwidth = .false.
    logical, private :: print_ai = .false.
    integer, private :: bytes_per_ldst = 8

    type(node_t), private, pointer :: root => NULL()             !< Start of graph
    type(node_t), private, pointer :: current_node => NULL()     !< Current position in the graph
    contains
      procedure, pass :: start => timer_start
      procedure, pass :: stop => timer_stop
      procedure, pass :: free => timer_free
      procedure, pass :: print => timer_print
      procedure, pass :: enable => timer_enable
      procedure, pass :: disable => timer_disable
      procedure, pass :: is_enabled => timer_is_enabled
      procedure, pass :: measure_flops => timer_measure_flops
      procedure, pass :: measure_allocated_memory => timer_measure_allocated_memory
      procedure, pass :: measure_virtual_memory => timer_measure_virtual_memory
      procedure, pass :: measure_max_allocated_memory => timer_measure_max_allocated_memory
      procedure, pass :: measure_memory_bandwidth => timer_measure_memory_bandwidth
      procedure, pass :: set_print_options => timer_set_print_options
      procedure, pass :: in_entries => timer_in_entries
      procedure, pass :: get => timer_get
      procedure, pass :: since => timer_since
      procedure, pass :: sort => timer_sort
  end type

  ! Private type node_t, representing a graph node
  !
  type :: node_t
    character(len=name_length) :: name             ! Descriptive name, used when printing the timings
    integer :: count = 0                           ! Number of node_stop calls
    type(value_t) :: value                         ! The actual counter data, see ftimings_values.F90
    logical :: is_running = .false.                ! .true. if still running
    type(node_t), pointer :: firstChild => NULL()
    type(node_t), pointer :: lastChild => NULL()
    type(node_t), pointer :: parent => NULL()
    type(node_t), pointer :: nextSibling => NULL()
    class(timer_t), pointer :: timer
    contains
      procedure, pass :: now => node_now
      procedure, pass :: start => node_start
      procedure, pass :: stop => node_stop
      procedure, pass :: get_value => node_get_value
      procedure, pass :: new_child => node_new_child
      procedure, pass :: get_child => node_get_child
      procedure, pass :: sum_of_children => node_sum_of_children
      procedure, pass :: sum_of_children_with_name => node_sum_of_children_with_name
      procedure, pass :: sum_of_children_below => node_sum_of_children_below
      procedure, pass :: print => node_print
      procedure, pass :: print_graph => node_print_graph
      procedure, pass :: sort_children => node_sort_children
  end type

  interface
    function microseconds_since_epoch() result(us) bind(C, name="ftimings_microseconds_since_epoch")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT64_T) :: us
    end function
  end interface

#ifdef HAVE_LIBPAPI
  interface
    function flop_init() result(ret) bind(C, name="ftimings_flop_init")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT) :: ret
    end function
  end interface

  interface
    function loads_stores_init() result(ret) bind(C, name="ftimings_loads_stores_init")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT) :: ret
    end function
  end interface

  interface
    subroutine papi_counters(flops, ldst) bind(C, name="ftimings_papi_counters")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_LONG_LONG), intent(out) :: flops, ldst
    end subroutine
  end interface
#endif

  interface
    function resident_set_size() result(rsssize) bind(C, name="ftimings_resident_set_size")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_LONG) :: rsssize
    end function
  end interface

  interface
    function virtual_memory() result(virtualmem) bind(C, name="ftimings_virtual_memory")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_LONG) :: virtualmem
    end function
  end interface

  interface
    function max_resident_set_size() result(maxrsssize) bind(C, name="ftimings_highwater_mark")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_LONG) :: maxrsssize
    end function
  end interface

  contains

  !> Activate the timer, without this, most methods are non-ops.
  !>
  subroutine timer_enable(self)
    class(timer_t), intent(inout), target :: self

    self%active = .true.
  end subroutine

  !> Call with enabled = .true. to also record amount of newly allocated memory.
  !> By default, memory usage is not recored. Call with .false. to deactivate again.
  !>
  !> This opens /proc/self/statm, parses it, and closes it agagain and is thus
  !> quite costly, use when appropriate.
  !>
  subroutine timer_measure_allocated_memory(self, enabled)
    class(timer_t), intent(inout) :: self
    logical, intent(in) :: enabled

    self%record_allocated_memory = enabled
  end subroutine

  !> Call with enabled = .true. to also record amount of newly created virtual memory.
  !> By default, memory usage is not recored. Call with .false. to deactivate again.
  !>
  !> This opens /proc/self/statm, parses it, and closes it agagain and is thus
  !> quite costly, use when appropriate.
  !>
  subroutine timer_measure_virtual_memory(self, enabled)
    class(timer_t), intent(inout) :: self
    logical, intent(in) :: enabled

    self%record_virtual_memory = enabled
  end subroutine

  !> Call with enabled = .true. to also record amount of newly increase of max.
  !> resident memory
  !> By default, memory usage is not recored. Call with .false. to deactivate again.
  !>
  !> This opens /proc/self/status, parses it, and closes it agagain and is thus
  !> quite costly, use when appropriate.
  !>
  subroutine timer_measure_max_allocated_memory(self, enabled)
    class(timer_t), intent(inout) :: self
    logical, intent(in) :: enabled

    self%record_max_allocated_memory = enabled
  end subroutine

  !> Call with enabled = .true. to also record the memory bandwidth with PAPI
  !> By default, this is not recorded. Call with .false. to deactivate again.
  !>
  subroutine timer_measure_memory_bandwidth(self, enabled)
    class(timer_t), intent(inout) :: self
    logical, intent(in) :: enabled

    if (enabled) then
#ifdef HAVE_LIBPAPI
      if (loads_stores_init() == 1) then
        self%record_memory_bandwidth = .true.
      else
        write(0,'(a)') "ftimings: Could not initialize PAPI, disabling memory bandwidth counter"
        self%record_memory_bandwidth = .false.
      endif
#else
      write(0,'(a)') "ftimings: not compiled with PAPI support, disabling memory bandwidth counter"
      self%record_memory_bandwidth = .false.
#endif
    else
      ! explicitly set to .false. by caller
      self%record_memory_bandwidth = .false.
    endif
  end subroutine

  !> Call with enabled = .true. to also record FLOP counts via PAPI calls.
  !> By default no FLOPS are recored. Call with .false. to deactivate again.
  !>
  subroutine timer_measure_flops(self, enabled)
    class(timer_t), intent(inout) :: self
    logical, intent(in) :: enabled

    if (enabled) then
#ifdef HAVE_LIBPAPI
      if (flop_init() == 1) then
        self%record_flop_counts = .true.
      else
        write(0,'(a)') "ftimings: Could not initialize PAPI, disabling FLOP counter"
        self%record_flop_counts = .false.
      endif
#else
      write(0,'(a)') "ftimings: not compiled with PAPI support, disabling FLOP counter"
      self%record_flop_counts = .false.
#endif
    else
      ! Explicitly set to .false. by caller
      self%record_flop_counts = .false.
    endif
  end subroutine

  !> Deactivate the timer
  !>
  subroutine timer_disable(self)
    class(timer_t), intent(inout), target :: self
    self%active = .false.
  end subroutine

  !> Return whether the timer is currently running
  !>
  function timer_is_enabled(self) result(is)
    class(timer_t), intent(inout), target :: self
    logical :: is
    is = self%active
  end function

  !> Control what to print on following %print calls
  !>
  !> \param     print_allocated_memory       Amount of newly allocated,
  !>                                         resident memory
  !> \param     print_virtual_memory         Amount of newly created virtual
  !>                                         memory
  !> \param     print_max_allocated_memory   Amount of new increase of max.
  !>                                         resident memory ("high water mark")
  !> \param     print_flop_count             Number of floating point operations
  !> \param     print_flop_rate              Rate of floating point operations per second
  !> \param     print_ldst                   Number of loads+stores
  !> \param     print_memory_bandwidth       Rate of loads+stores per second
  !> \param     print_ai                     Arithmetic intensity, that is number of
  !>                                         floating point operations per
  !>                                         number of load and store
  !>                                         operations (currently untested)
  !> \param     bytes_per_ldst               For calculating the AI, assume this number
  !>                                         of bytes per load or store (default: 8)
  subroutine timer_set_print_options(self, &
	print_allocated_memory, &
	print_virtual_memory, &
	print_max_allocated_memory, &
	print_flop_count, &
	print_flop_rate, &
	print_ldst, &
        print_memory_bandwidth, &
	print_ai, &
        bytes_per_ldst)
    class(timer_t), intent(inout) :: self
    logical, intent(in), optional :: &
        print_allocated_memory, &
        print_virtual_memory, &
        print_max_allocated_memory, &
        print_flop_count, &
        print_flop_rate, &
        print_ldst, &
        print_memory_bandwidth, &
        print_ai
    integer, intent(in), optional :: bytes_per_ldst

    if (present(print_allocated_memory)) then
      self%print_allocated_memory = print_allocated_memory
      if ((.not. self%record_allocated_memory) .and. self%print_allocated_memory) then
         write(0,'(a)') "ftimings: Warning: RSS size recording was disabled, expect zeros!"
      endif
    endif

    if (present(print_virtual_memory)) then
      self%print_virtual_memory = print_virtual_memory
      if ((.not. self%record_virtual_memory) .and. self%print_virtual_memory) then
         write(0,'(a)') "ftimings: Warning: Virtual memory recording was disabled, expect zeros!"
      endif
    endif

    if (present(print_max_allocated_memory)) then
      self%print_max_allocated_memory = print_max_allocated_memory
      if ((.not. self%record_max_allocated_memory) .and. self%print_max_allocated_memory) then
         write(0,'(a)') "ftimings: Warning: HWM recording was disabled, expect zeros!"
      endif
    endif

    if (present(print_flop_count)) then
      self%print_flop_count = print_flop_count
      if ((.not. self%record_flop_counts) .and. self%print_flop_count) then
         write(0,'(a)') "ftimings: Warning: FLOP counter was disabled, expect zeros!"
      endif
    endif

    if (present(print_flop_rate)) then
      self%print_flop_rate = print_flop_rate
      if ((.not. self%record_flop_counts) .and. self%print_flop_rate) then
         write(0,'(a)') "ftimings: Warning: FLOP counter was disabled, expect zeros!"
      endif
    endif

    if (present(print_ldst)) then
      self%print_ldst = print_ldst
      if ((.not. self%record_memory_bandwidth) .and. self%print_ldst) then
         write(0,'(a)') "ftimings: Warning: Load+Store counters were disabled, expect zeros!"
      endif
    endif
    if (present(print_memory_bandwidth)) then
      self%print_memory_bandwidth = print_memory_bandwidth
      if ((.not. self%record_memory_bandwidth) .and. self%print_memory_bandwidth) then
         write(0,'(a)') "ftimings: Warning: Load+Store counters were disabled, expect zeros for memory bandwidth!"
      endif
    endif

    if (present(print_ai)) then
      self%print_ai = print_ai
      if (.not. (self%record_memory_bandwidth .and. self%record_flop_counts)) then
         write(0,'(a)') "ftimings: Warning: Memory bandwidth or FLOP counters were disabled, expect invalid values for AI"
      endif
    endif

    if (present(bytes_per_ldst)) then
      self%bytes_per_ldst = bytes_per_ldst
    endif
  end subroutine

  !> Start a timing section
  !>
  !> \param name        A descriptive name
  !> \param replace     If .true. (default .false.), replace any entries at the
  !>                    current position with the same name. If .false., add the
  !>                    time to a possibly existing entry
  !>
  !> Care must be taken to balance any invocations of %start() and %stop(), e.g.
  !> the following is valid
  !>
  !> \code{.f90}
  !>   call timer%start("A")
  !>   call timer%start("B")
  !>   call timer%stop("B")
  !>   call timer%stop("A")
  !> \endcode
  !>
  !> while the following is not
  !>
  !> \code{.f90}
  !>   call timer%start("A")
  !>   call timer%start("B")
  !>   call timer%stop("A")
  !>   call timer%stop("B")
  !> \endcode
  !>
  subroutine timer_start(self, name, replace)
    class(timer_t), intent(inout), target :: self
    character(len=*), intent(in)  :: name
    logical, intent(in), optional  :: replace
    type(node_t), pointer :: node
    !$ integer :: omp_get_thread_num, omp_get_num_threads, omp_get_level, omp_get_ancestor_thread_num
    !$ integer :: i

    if (.not. self%active) then
      return
    endif

    ! Deal with nested parallelization
    !$ do i = 0, omp_get_level()
    !$   if (omp_get_ancestor_thread_num(i) > 0) then
    !$     return
    !$   endif
    !$ end do

    !$omp master

    if (.not. associated(self%current_node)) then
      ! First call to timer_start()
      allocate(self%root)
      self%root%name = "[Root]"
      self%root%timer => self
      call self%root%start()
      nullify(self%root%firstChild)
      nullify(self%root%lastChild)
      nullify(self%root%parent)
      nullify(self%root%nextSibling)
      self%current_node => self%root
    endif

    if (string_eq(self%current_node%name, name)) then
      !$omp critical
      write(error_unit,*) "Recursion error! Printing tree so far.."
      write(error_unit,*) "Got %start(""" // trim(name) // """), while %start(""" // trim(name) // """) was still active"
      !$ write(*,*) "omp_get_thread_num() = ", omp_get_thread_num()
      !$ write(*,*) "omp_get_num_threads() = ", omp_get_num_threads()
      !$ write(*,*) "omp_get_level() = ", omp_get_level()
      !$ do i = 0, omp_get_level()
      !$   write(*,*) "omp_get_ancestor_thread_num(", i, ") = ", omp_get_ancestor_thread_num(i)
      !$ end do
      call self%root%print_graph(0)
      !$omp end critical
      stop "timer_start() while same timer was active"
    endif
    node => self%current_node%get_child(name)
    if (.not. associated(node)) then
      node => self%current_node%new_child(name)
    else if (present(replace)) then
      if (replace) then
        node%value = null_value
        node%count = 0
        if (associated(node%firstChild)) then
          call deallocate_node(node%firstChild)
          nullify(node%firstChild)
          nullify(node%lastChild)
        endif
      endif
    endif

    call node%start()

    self%current_node => node

    !$omp end master

  end subroutine

  !> End a timing segment, \sa timer_start
  !>
  !> \param name        The exact same name as was used for %start().
  !>                    If not provided, close the currently active region.
  !>                    If given, warns if it does not match the last %start()
  !>                    call on stderr and disables the current timer instance.
  !>
  subroutine timer_stop(self, name)
    class(timer_t), intent(inout), target :: self
    character(len=*), intent(in), optional :: name
    logical :: error
    !$ integer :: omp_get_level, omp_get_ancestor_thread_num
    !$ integer :: i

    if (.not. self%active) then
      return
    endif

    ! Deal with nested parallelization
    !$ do i = 0, omp_get_level()
    !$   if (omp_get_ancestor_thread_num(i) > 0) then
    !$     return
    !$   endif
    !$ end do

    !$omp master
    error = .false.

    if (.not. associated(self%current_node)) then
      write(error_unit,'(a)') "Called timer_stop() without first calling any timer_start(), disabling timings"
      call self%free()
      self%active = .false.
      error = .true.
    else if (present(name)) then
      if (.not. string_eq(self%current_node%name, name)) then
        write(error_unit,'(a)') "Expected %stop(""" // trim(self%current_node%name)  // """),&
                 & but got %stop(""" // trim(name) //  """), disabling timings"
        call self%free()
        self%active = .false.
        error = .true.
      endif
    endif

    if (.not. error) then
      call self%current_node%stop()

      ! climb up to parent
      if (.not. associated(self%current_node%parent)) then
        write(error_unit,'(a)') "Error: No valid parent node found for node '" // trim(self%current_node%name) // "'"
        call self%free()
        self%active = .false.
      endif
      self%current_node => self%current_node%parent

    endif
    !$omp end master

  end subroutine

  !> Deallocate all objects associated with (but not including) self
  !>
  subroutine timer_free(self)
    class(timer_t), intent(inout), target :: self
    if (associated(self%root)) then
      call deallocate_node(self%root)
    endif
    nullify(self%root)
    nullify(self%current_node)
  end subroutine

  !> Print a timing graph
  !>
  !> \param name1       If given, first descend one level to the node with name name1
  !> \param name2       If given, also descend another level to the node with name2 there
  !> \param name3       etc.
  !> \param name4       etc.
  !> \param threshold   If given, subsume any entries with a value of threshold
  !>                    seconds in a single node "(below threshold)"
  !> \param is_sorted   Assume a sorted graph for inserting "(own)" and "(below threshold)"
  !> \param unit        The unit number on which to print, default stdout
  !>
  subroutine timer_print(self, name1, name2, name3, name4, threshold, is_sorted, unit)
    class(timer_t), intent(in), target :: self
    character(len=*), intent(in), optional :: name1, name2, name3, name4
    real(kind=rk), intent(in), optional :: threshold
    logical, intent(in), optional :: is_sorted
    integer, intent(in), optional :: unit

    integer :: unit_act

    type(node_t), pointer :: node
    character(len=64) :: format_spec

    ! I hate fortran's string handling
    character(len=name_length), parameter :: group = "Group"
    character(len=12), parameter :: seconds    = "         [s]"
    character(len=12), parameter :: fract      = "    fraction"
    character(len=12), parameter :: ram        = "  alloc. RAM"
    character(len=12), parameter :: vmem       = "   alloc. VM"
    character(len=12), parameter :: hwm        = "  alloc. HWM"
    character(len=12), parameter :: flop_rate  = "     Mflop/s"
    character(len=12), parameter :: flop_count = "       Mflop"
    character(len=12), parameter :: ldst       = "loads+stores"
    character(len=12), parameter :: bandwidth  = "  mem bandw."
    character(len=12), parameter :: ai         = "arithm. Int."
    character(len=12), parameter :: dash       = "============"

    if (.not. self%active) then
      return
    endif

    if (present(unit)) then
      unit_act = unit
    else
      unit_act = output_unit
    endif

    node => self%root
    if (present(name1)) then
      node => node%get_child(name1)
      if (.not. associated(node)) then
        write(error_unit,'(a)') "Could not descend to """ // trim(name1)  // """"
        return
      endif
    end if
    if (present(name2)) then
      node => node%get_child(name2)
      if (.not. associated(node)) then
        write(error_unit,'(a)') "Could not descend to """ // trim(name2)  // """"
        return
      endif
    end if
    if (present(name3)) then
      node => node%get_child(name3)
      if (.not. associated(node)) then
        write(error_unit,'(a)') "Could not descend to """ // trim(name3)  // """"
        return
      endif
    end if
    if (present(name4)) then
      node => node%get_child(name4)
      if (.not. associated(node)) then
        write(error_unit,'(a)') "Could not descend to """ // trim(name4)  // """"
        return
      endif
    end if

    ! I really do hate it ..
    write(format_spec,'("("" /= "",a",i0,",2x,a12,1x,a12)")') name_length
    write(unit_act, format_spec, advance='no') adjustl(group), seconds, fract

    if (self%print_allocated_memory) then
      write(unit_act,'(1x,a12)',advance='no') ram
    endif

    if (self%print_virtual_memory) then
      write(unit_act,'(1x,a12)',advance='no') vmem
    endif

    if (self%print_max_allocated_memory) then
      write(unit_act,'(1x,a12)',advance='no') hwm
    endif

    if (self%print_flop_count) then
      write(unit_act,'(1x,a12)',advance='no') flop_count
    endif
    if (self%print_flop_rate) then
      write(unit_act,'(1x,a12)',advance='no') flop_rate
    endif
    if (self%print_ldst) then
      write(unit_act,'(1x,a12)',advance='no') ldst
    endif
    if (self%print_memory_bandwidth) then
      write(unit_act,'(1x,a12)',advance='no') bandwidth
    endif
    if (self%print_ai) then
      write(unit_act,'(1x,a12)',advance='no') ai
    endif

    write(unit_act,'(a)') ""

    write(format_spec,'("("" |  "",a",i0,",1x,2(1x,a12))")') name_length
    write(unit_act, format_spec, advance='no') "", dash, dash

    if (self%print_allocated_memory) then
      write(unit_act,'(1x,a12)',advance='no') dash
    endif

    if (self%print_virtual_memory) then
      write(unit_act,'(1x,a12)',advance='no') dash
    endif

    if (self%print_max_allocated_memory) then
      write(unit_act,'(1x,a12)',advance='no') dash
    endif

    if (self%print_flop_count) then
      write(unit_act,'(1x,a12)',advance='no') dash
    endif
    if (self%print_flop_rate) then
      write(unit_act,'(1x,a12)',advance='no') dash
    endif
    if (self%print_ldst) then
      write(unit_act,'(1x,a12)',advance='no') dash
    endif
    if (self%print_memory_bandwidth) then
      write(unit_act,'(1x,a12)',advance='no') dash
    endif
    if (self%print_ai) then
      write(unit_act,'(1x,a12)',advance='no') dash
    endif

    write(unit_act,'(a)') ""

    call node%print_graph(0, threshold, is_sorted, unit=unit)

  end subroutine

  !> Return the sum of all entries with a certain name below
  !> a given node. Specify the name with the last argument, the
  !> path to the starting point with the first few parameters
  !>
  !> \param name1, .., namei-1  The path to the starting node
  !> \param namei               The name of all sub-entries below this
  !>                            node which should be summed together
  !>
  !> For example timer%in_entries("foo", "bar", "parallel") returns
  !> the sum of all entries named "parallel" below the foo->bar node
  !>
  function timer_in_entries(self, name1, name2, name3, name4) result(s)
    use, intrinsic :: iso_fortran_env, only : error_unit
    class(timer_t), intent(in), target :: self
    character(len=*), intent(in) :: name1
    character(len=*), intent(in), optional :: name2, name3, name4
    real(kind=rk) :: s
    type(node_t), pointer :: node ! the starting node
    type(value_t) :: val
    character(len=name_length) :: name ! the name of the sections

    s = 0._rk

    if (.not. self%active) then
      return
    endif

    node => self%root
    name = name1

    if (present(name2)) then
      node => node%get_child(name1)
      if (.not. associated(node)) then
        write(error_unit,'(a)') "Could not descend to """ // trim(name1)  // """"
        return
      endif
      name = name2
    end if
    if (present(name3)) then
      node => node%get_child(name2)
      if (.not. associated(node)) then
        write(error_unit,'(a)') "Could not descend to """ // trim(name2)  // """"
        return
      endif
      name = name3
    end if
    if (present(name4)) then
      node => node%get_child(name3)
      if (.not. associated(node)) then
        write(error_unit,'(a)') "Could not descend to """ // trim(name3)  // """"
        return
      endif
      name = name4
    end if

    val = node%sum_of_children_with_name(name)
    s = real(val%micros, kind=rk) * 1e-6_rk
  end function

  !> Access a specific, already stopped entry of the graph by specifying the
  !> names of the nodes along the graph from the root node
  !>
  !> The result is only meaningfull if the entry was never appended by
  !> additional %start() calls.
  !>
  function timer_get(self, name1, name2, name3, name4, name5, name6) result(s)
    class(timer_t), intent(in), target :: self
    ! this is clunky, but what can you do..
    character(len=*), intent(in), optional :: name1, name2, name3, name4, name5, name6
    real(kind=rk) :: s
    type(node_t), pointer :: node

    s = 0._rk

    if (.not. self%active) then
      return
    endif

    node => self%root
    if (present(name1)) then
      node => node%get_child(name1)
      if (.not. associated(node)) then
        write(error_unit,'(a)') "Could not descend to """ // trim(name1)  // """"
        return
      endif
    end if
    if (present(name2)) then
      node => node%get_child(name2)
      if (.not. associated(node)) then
        write(error_unit,'(a)') "Could not descend to """ // trim(name2)  // """"
        return
      endif
    end if
    if (present(name3)) then
      node => node%get_child(name3)
      if (.not. associated(node)) then
        write(error_unit,'(a)') "Could not descend to """ // trim(name3)  // """"
        return
      endif
    end if
    if (present(name4)) then
      node => node%get_child(name4)
      if (.not. associated(node)) then
        write(error_unit,'(a)') "Could not descend to """ // trim(name4)  // """"
        return
      endif
    end if
    if (present(name5)) then
      node => node%get_child(name5)
      if (.not. associated(node)) then
        write(error_unit,'(a)') "Could not descend to """ // trim(name5)  // """"
        return
      endif
    end if
    if (present(name6)) then
      node => node%get_child(name6)
      if (.not. associated(node)) then
        write(error_unit,'(a)') "Could not descend to """ // trim(name6)  // """"
        return
      endif
    end if
    if (node%is_running) then
      write(error_unit,'(a)') "Timer """ // trim(node%name) // """ not yet stopped"
      return
    endif
    s = real(node%value%micros, kind=rk) * 1e-6_rk
  end function

  !> Access a specific, not yet stopped entry of the graph by specifying the
  !> names of the nodes along the graph from the root node and return the
  !> seconds that have passed since the entry was created.
  !>
  !> The result is only meaningfull if the entry was never appended by
  !> additional %start() calls.
  !>
  function timer_since(self, name1, name2, name3, name4) result(s)
    class(timer_t), intent(in), target :: self
    character(len=*), intent(in), optional :: name1, name2, name3, name4
    real(kind=rk) :: s
    type(value_t) :: val
    type(node_t), pointer :: node

    s = 0._rk

    node => self%root
    if (present(name1)) then
      node => node%get_child(name1)
      if (.not. associated(node)) then
        write(error_unit,'(a)') "Could not descend to """ // trim(name1)  // """"
        return
      endif
    end if
    if (present(name2)) then
      node => node%get_child(name2)
      if (.not. associated(node)) then
        write(error_unit,'(a)') "Could not descend to """ // trim(name2)  // """"
        return
      endif
    end if
    if (present(name3)) then
      node => node%get_child(name3)
      if (.not. associated(node)) then
        write(error_unit,'(a)') "Could not descend to """ // trim(name3)  // """"
        return
      endif
    end if
    if (present(name4)) then
      node => node%get_child(name4)
      if (.not. associated(node)) then
        write(error_unit,'(a)') "Could not descend to """ // trim(name4)  // """"
        return
      endif
    end if
    if (node%is_running .neqv. .true.) then
      write(error_unit,'(a)') "Timer """ // trim(node%name) // """ already stopped"
      return
    endif
    val = node%value + node%now()
    s = real(val%micros, kind=rk) * 1e-6_rk
  end function

  !> Sort the graph on each level.
  !> Warning: This irrevocable destroys the old ordering.
  !>
  subroutine timer_sort(self)
    class(timer_t), intent(inout), target :: self
    type(node_t), pointer :: node

    call sort_nodes(self%root, node)

    node => self%root
    do while (associated(node))
      call node%sort_children()
      node => node%nextSibling
    enddo
  end subroutine



  ! Now methods of node_t:


  ! This is the function that actually returns the current timestamp and all other counters
  function node_now(self) result(val)
    use, intrinsic :: iso_c_binding
    class(node_t), intent(in) :: self
    type(value_t) :: val

    ! current time
    val%micros = microseconds_since_epoch()

    if (self%timer%record_allocated_memory) then
      val%rsssize = resident_set_size()
    endif

    if (self%timer%record_virtual_memory) then
      val%virtualmem = virtual_memory()
    endif

    if (self%timer%record_max_allocated_memory) then
      val%maxrsssize = max_resident_set_size()
    endif

#ifdef HAVE_LIBPAPI
    if (self%timer%record_flop_counts .or. self%timer%record_memory_bandwidth) then
      call papi_counters(val%flop_count, val%ldst)
    endif
#endif
  end function


  subroutine node_start(self)
    class(node_t), intent(inout) :: self

    ! take the time
    self%value = self%value - self%now()
    self%is_running = .true.
  end subroutine

  subroutine node_stop(self)
    class(node_t), intent(inout) :: self

    self%count = self%count + 1

    ! take the time
    self%value = self%value + self%now()
    self%is_running = .false.
  end subroutine

  function node_get_value(self) result(val)
    class(node_t), intent(in) :: self
    type(value_t) :: val
    val = self%value
    if (self%is_running) then
      ! we have not finished, give time up to NOW
      val = val + self%now()
    endif
  end function

  function node_new_child(self, name) result(new)
    class(node_t), intent(inout), target :: self
    character(len=*), intent(in) :: name
    type(node_t), pointer :: new

    if (.not. associated(self%lastChild)) then
      allocate(self%lastChild)
      new => self%lastChild
      self%firstChild => new
    else
      allocate(self%lastChild%nextSibling)
      new => self%lastChild%nextSibling
      self%lastChild => new
    endif

    select type (self)
      type is (node_t)
        new%parent => self
      class default
        stop "node_new_child(): This should not happen"
    end select

    new%name = name
    new%count = 0
    new%timer => self%timer

    nullify(new%firstChild)
    nullify(new%lastChild)
    nullify(new%nextSibling)
  end function


  function string_eq(str1, str2) result(eq)
    character(len=name_length), intent(in) :: str1
    character(len=*), intent(in) :: str2
    logical :: eq
    eq = trim(str1) .eq. str2(1:min(len(trim(str2)), name_length))
  end function

  function node_get_child(self, name) result(child)
    class(node_t), intent(in) :: self
    character(len=*), intent(in) :: name
    type(node_t), pointer :: child

    child => self%firstChild
    do while (associated(child))
      if (string_eq(child%name, name)) then
        return
      endif
      child => child%nextSibling
    enddo
    nullify(child)
  end function

  recursive subroutine deallocate_node(entry)
    type(node_t), intent(inout), pointer :: entry
    type(node_t), pointer :: nextSibling

    if (associated(entry%firstChild)) then
      call deallocate_node(entry%firstChild)
    endif
    nextSibling => entry%nextSibling
    deallocate(entry)
    nullify(entry)
    if (associated(nextSibling)) then
      call deallocate_node(nextSibling)
    endif
  end subroutine

  function node_sum_of_children(self) result(sum_time)
    class(node_t), intent(in) :: self
    type(node_t), pointer :: cur_entry
    type(value_t) :: sum_time

    cur_entry => self%firstChild
    do while (associated(cur_entry))
      sum_time = sum_time + cur_entry%get_value()
      cur_entry => cur_entry%nextSibling
    enddo
  end function

  recursive function node_sum_of_children_with_name(self, name) result(sum_time)
    class(node_t), intent(in) :: self
    character(len=*), intent(in) :: name
    type(node_t), pointer :: cur_entry
    type(value_t) :: sum_time

    cur_entry => self%firstChild
    do while (associated(cur_entry))
      if (string_eq(cur_entry%name, name)) then
        sum_time = sum_time + cur_entry%value
      else
        sum_time = sum_time + cur_entry%sum_of_children_with_name(name)
      endif
      cur_entry => cur_entry%nextSibling
    enddo
  end function

  function node_sum_of_children_below(self, threshold) result(sum_time)
    class(node_t), intent(in) :: self
    real(kind=rk), intent(in), optional :: threshold
    type(node_t), pointer :: cur_entry
    type(value_t) :: sum_time, cur_value

    if (.not. present(threshold)) then
      return
    endif

    cur_entry => self%firstChild

    do while (associated(cur_entry))
      cur_value = cur_entry%get_value()
      if (cur_value%micros * 1e-6_rk < threshold) then
        sum_time = sum_time + cur_value
      endif
      cur_entry => cur_entry%nextSibling
    enddo
  end function

  subroutine insert_into_sorted_list(head, node)
    type(node_t), pointer, intent(inout) :: head
    type(node_t), target, intent(inout)    :: node
    type(node_t), pointer :: cur

    if (node%value%micros >= head%value%micros) then
      node%nextSibling => head
      head => node
      return
    endif

    cur => head
    do while (associated(cur%nextSibling))
      if (cur%value%micros > node%value%micros .and. node%value%micros >= cur%nextSibling%value%micros) then
        node%nextSibling => cur%nextSibling
        cur%nextSibling => node
        return
      endif
      cur => cur%nextSibling
    end do

    ! node has to be appended at the end
    cur%nextSibling => node
    node%nextSibling => NULL()
  end subroutine

  subroutine remove_from_list(head, node)
    type(node_t), pointer, intent(inout) :: head
    type(node_t), pointer, intent(in)    :: node
    type(node_t), pointer :: cur

    if (associated(head,node)) then
      head => head%nextSibling
      return
    endif

    cur => head
    do while (associated(cur%nextSibling))
      if (associated(cur%nextSibling,node)) then
        cur%nextSibling => cur%nextSibling%nextSibling
        return
      endif
      cur => cur%nextSibling
    end do
  end subroutine

  subroutine node_print(self, indent_level, total, unit)
    class(node_t), intent(inout) :: self
    integer, intent(in) :: indent_level
    type(value_t), intent(in) :: total
    type(value_t) :: val
    integer, intent(in) :: unit
    character(len=name_length) :: name, suffix

    if (self%is_running) then
      name = trim(self%name) // " (running)"
    else
      name = self%name
    endif

    if (self%count > 1) then
      write(suffix, '(" (",i0,"x)")') self%count
      name = trim(name) // " " // trim(suffix)
    endif

    if (self%is_running) then
      val = self%value + self%now()
    else
      val = self%value
    endif
    call print_value(val, self%timer, indent_level, name, total, unit)
  end subroutine

  recursive subroutine node_print_graph(self, indent_level, threshold, is_sorted, total, unit)
    use, intrinsic :: iso_fortran_env, only : output_unit
    class(node_t), intent(inout) :: self
    integer, intent(in) :: indent_level
    real(kind=rk), intent(in), optional :: threshold
    logical, intent(in), optional :: is_sorted
    type(value_t), intent(in), optional :: total
    integer, intent(in), optional :: unit

    type(node_t), pointer :: node
    integer :: i
    type(value_t) :: cur_value, node_value, own_value, below_threshold_value, total_act
    type(node_t), pointer :: own_node, threshold_node
    real(kind=rk) :: threshold_act
    logical :: is_sorted_act, print_own, print_threshold
    integer :: unit_act

    nullify(own_node)
    nullify(threshold_node)

    if (present(threshold)) then
      threshold_act = threshold
    else
      threshold_act = 0
    endif

    if (present(is_sorted)) then
      is_sorted_act = is_sorted
    else
      is_sorted_act = .false.
    endif

    cur_value = self%get_value()

    if (present(total)) then
      total_act = total
    else
      total_act = cur_value
    endif

    if (present(unit)) then
      unit_act = unit
    else
      unit_act = output_unit
    endif

    call self%print(indent_level, total_act, unit_act)

    own_value = cur_value - self%sum_of_children()
    below_threshold_value = self%sum_of_children_below(threshold)

    print_own = associated(self%firstChild)
    print_threshold = below_threshold_value%micros > 0

    ! Deal with "(own)" and "(below threshold)" entries
    if (is_sorted_act) then
      ! sort them in
      if (print_own) then
        ! insert an "(own)" node
        allocate(own_node)
        own_node%value = own_value
        own_node%name = own
        own_node%timer => self%timer
        call insert_into_sorted_list(self%firstChild, own_node)
      endif

      if (print_threshold) then
        ! insert a "(below threshold)" node
        allocate(threshold_node)
        threshold_node%value = below_threshold_value
        threshold_node%name = below
        threshold_node%timer => self%timer
        call insert_into_sorted_list(self%firstChild, threshold_node)
      endif

    else
      ! print them first
      if (print_own) then
        call print_value(own_value, self%timer, indent_level + 1, own, cur_value, unit_act)
      endif
      if (print_threshold) then
        call print_value(below_threshold_value, self%timer, indent_level + 1, below, cur_value, unit_act)
      endif
    endif

    ! print children
    node => self%firstChild
    do while (associated(node))
      node_value = node%get_value()
      if (node_value%micros * 1e-6_rk >= threshold_act &
                .or. associated(node, threshold_node) &
                .or. associated(node, own_node)) then
        call node%print_graph(indent_level + 1, threshold, is_sorted, cur_value, unit_act)
      endif
      node => node%nextSibling
    end do

    if (is_sorted_act) then
      ! remove inserted dummy nodes again
      if (print_own) then
        call remove_from_list(self%firstChild, own_node)
        deallocate(own_node)
      endif
      if (print_threshold) then
        call remove_from_list(self%firstChild, threshold_node)
        deallocate(threshold_node)
      endif
    endif

  end subroutine

  ! In-place sort a node_t linked list and return the first and last element,
  subroutine sort_nodes(head, tail)
    type(node_t), pointer, intent(inout) :: head, tail

    type(node_t), pointer :: p, q, e
    type(value_t) :: p_val, q_val
    integer :: insize, nmerges, psize, qsize, i

    if (.not. associated(head)) then
      nullify(tail)
      return
    endif

    insize = 1

    do while (.true.)
      p => head
      nullify(head)
      nullify(tail)
      nmerges = 0

      do while(associated(p))
        nmerges = nmerges + 1
        q => p
        psize = 0
        do i = 1, insize
          psize = psize + 1
          q => q%nextSibling
          if (.not. associated(q)) then
            exit
          endif
        end do

        qsize = insize

        do while (psize > 0 .or. (qsize > 0 .and. associated(q)))
          if (psize == 0) then
            e => q
            q => q%nextSibling
            qsize = qsize - 1

          else if (qsize == 0 .or. (.not. associated(q))) then
            e => p;
            p => p%nextSibling
            psize = psize - 1
          else
            p_val = p%get_value()
            q_val = q%get_value()
            if (p_val%micros >= q_val%micros) then
              e => p
              p => p%nextSibling
              psize = psize - 1

            else
              e => q
              q => q%nextSibling
              qsize = qsize - 1

            end if
          end if

          if (associated(tail)) then
            tail%nextSibling => e
          else
            head => e
          endif
          tail => e

        end do

        p => q

      end do

      nullify(tail%nextSibling)

      if (nmerges <= 1) then
        return
      endif

      insize = insize * 2

    end do
  end subroutine


  recursive subroutine node_sort_children(self)
    class(node_t), intent(inout) :: self
    type(node_t), pointer :: node

    call sort_nodes(self%firstChild, self%lastChild)

    node => self%firstChild
    do while (associated(node))
      call node%sort_children()
      node => node%nextSibling
    enddo
  end subroutine

  subroutine print_value(value, timer, indent_level, label, total, unit)
    type(value_t), intent(in) :: value
    type(timer_t), intent(in) :: timer
    integer, intent(in) :: indent_level
    character(len=name_length), intent(in) :: label
    type(value_t), intent(in) :: total
    integer, intent(in) :: unit

    character(len=64) :: format_spec

    write(format_spec,'("(",i0,"x,""|_ "",a",i0,",2x,f12.6,1x,f12.3)")') indent_level * 2 + 1, name_length
    write(unit,format_spec,advance='no') &
      label, &
      real(value%micros, kind=rk) * 1e-6_rk, &
      real(value%micros, kind=rk) / real(total%micros, kind=rk)

    if (timer%print_allocated_memory) then
      write(unit,'(1x,a12)',advance='no') &
        nice_format(real(value%rsssize, kind=C_DOUBLE))
    endif

    if (timer%print_virtual_memory) then
      write(unit,'(1x,a12)',advance='no') &
        nice_format(real(value%virtualmem, kind=C_DOUBLE))
    endif

    if (timer%print_max_allocated_memory) then
      write(unit,'(1x,a12)',advance='no') &
        nice_format(real(value%maxrsssize, kind=C_DOUBLE))
    endif

    if (timer%print_flop_count) then
      write(unit,'(1x,f12.2)',advance='no') real(value%flop_count, kind=rk) / 1e6_rk
    endif
    if (timer%print_flop_rate) then
      write(unit,'(1x,f12.2)',advance='no') real(value%flop_count, kind=rk) / value%micros
    endif
    if (timer%print_ldst) then
      write(unit,'(1x,a12)',advance='no') nice_format(real(value%ldst, kind=rk))
    endif
    if (timer%print_memory_bandwidth) then
      write(unit,'(1x,a12)',advance='no') nice_format(real(value%ldst*timer%bytes_per_ldst, kind=rk) / (value%micros * 1e-6_rk))
    endif
    if (timer%print_ai) then
      write(unit,'(1x,f12.4)',advance='no') real(value%flop_count, kind=rk) / value%ldst / timer%bytes_per_ldst
    endif

    write(unit,'(a)') ""
  end subroutine

  pure elemental function nice_format(number) result(string)
    real(kind=C_DOUBLE), intent(in) :: number
    character(len=12) :: string
    real(kind=C_DOUBLE), parameter :: &
        kibi = 2.0_C_DOUBLE**10, &
        mebi = 2.0_C_DOUBLE**20, &
        gibi = 2.0_C_DOUBLE**30, &
        tebi = 2.0_C_DOUBLE**40, &
        pebi = 2.0_C_DOUBLE**50

    if (abs(number) >= pebi) then
      write(string,'(es12.2)') number
    else if (abs(number) >= tebi) then
      write(string,'(f9.2,'' Ti'')') number / tebi
    else if (abs(number) >= gibi) then
      write(string,'(f9.2,'' Gi'')') number / gibi
    else if (abs(number) >= mebi) then
      write(string,'(f9.2,'' Mi'')') number / mebi
    else if (abs(number) >= kibi) then
      write(string,'(f9.2,'' ki'')') number / kibi
    else
      write(string,'(f12.2)') number
    endif
  end function


end module
