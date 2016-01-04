module precision
  use iso_c_binding, only : C_FLOAT, C_DOUBLE, C_INT32_T

  implicit none
  integer, parameter :: rk  = C_DOUBLE
  integer, parameter :: ck  = C_DOUBLE
  integer, parameter :: ik  = C_INT32_T
end module precision
