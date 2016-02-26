#include "config-f90.h"

module time_c

  use precision
  use, intrinsic :: iso_c_binding

  interface
    function microseconds_since_epoch() result(ms) bind(C, name="ftimings_microseconds_since_epoch")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT64_T) :: ms
    end function
  end interface

  interface
    function seconds() result(s) bind(C, name="seconds")
      use, intrinsic :: iso_c_binding
      implicit none
      real(kind=C_DOUBLE) :: s
    end function
  end interface

end module time_c
