
#include "config-f90.h"
module check_monotony
  use precision
  implicit none
  private

  public :: check_monotony_double
#if defined(WANT_SINGLE_PRECISION_REAL) || defined(WANT_SINGLE_PRECISION_COMPLEX)
  public :: check_monotony_single
#endif

  contains

! real double precision first
#define REALCASE
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"
#include "./check_monotony_template.F90"
#undef REALCASE
#undef DOUBLE_PRECISION

#ifdef WANT_SINGLE_PRECISION_REAL
! real single precision first
#define REALCASE
#define SINGLE_PRECISION
#include "../general/precision_macros.h"
#include "./check_monotony_template.F90"
#undef REALCASE
#undef SINGLE_PRECISION
#endif

end module
