#include "config-f90.h"
module v_add_s
  use precision
  implicit none
  private

  public :: v_add_s_double
#if defined(WANT_SINGLE_PRECISION_REAL) || defined(WANT_SINGLE_PRECISION_COMPLEX)
  public :: v_add_s_single
#endif

  contains

! real double precision first
#define REALCASE
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"
#define _rk _c_double
#include "./v_add_s_template.F90"
#undef REALCASE
#undef DOUBLE_PRECISION
#undef _rk

#ifdef WANT_SINGLE_PRECISION_REAL
! real single precision first
#define REALCASE
#define SINGLE_PRECISION
#include "../general/precision_macros.h"
#define _rk _c_float
#include "./v_add_s_template.F90"
#undef REALCASE
#undef SINGLE_PRECISION
#undef _rk
#endif

end module
