#include "config-f90.h"
module precision
  use iso_c_binding, only : C_FLOAT, C_DOUBLE, C_INT32_T, C_INT64_T, C_FLOAT

  implicit none
#ifdef DOUBLE_PRECISION_REAL
  integer, parameter :: rk  = C_DOUBLE
#else
  integer, parameter :: rk  = C_FLOAT
#endif
#ifdef DOUBLE_PRECISION_COMPLEX
  integer, parameter :: ck  = C_DOUBLE
#else
  integer, parameter :: ck  = C_FLOAT
#endif
  integer, parameter :: ik  = C_INT32_T
  integer, parameter :: lik = C_INT64_T
end module precision
