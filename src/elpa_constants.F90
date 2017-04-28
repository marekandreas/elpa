module elpa_constants
  use, intrinsic :: iso_c_binding, only : C_INT
  implicit none
  public
#include "src/fortran_constants.X90"
end module
