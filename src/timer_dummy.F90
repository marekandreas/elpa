
#include "config-f90.h"

module timings_dummy
  implicit none
  
  type, public :: timer_dummy_t
      contains
      procedure, pass :: start => timer_start
      procedure, pass :: stop => timer_stop
  end type 

  type(timer_dummy_t) :: timer

  contains

  subroutine timer_start(self, name, replace)
    class(timer_dummy_t), intent(inout), target :: self
    character(len=*), intent(in)  :: name
    logical, intent(in), optional  :: replace
    
  end subroutine
  
  subroutine timer_stop(self, name)
    class(timer_dummy_t), intent(inout), target :: self
    character(len=*), intent(in), optional :: name
    
  end subroutine

end module timings_dummy
