
#include "config-f90.h"
module merge_systems
!DIR$ OPTIMIZE:1
  use precision
  implicit none
  private

  public :: merge_systems_double
#if defined(WANT_SINGLE_PRECISION_REAL) || defined(WANT_SINGLE_PRECISION_COMPLEX)
  public :: merge_systems_single
#endif

  contains

! real double precision first
#define DOUBLE_PRECISION_REAL
#define REALCASE
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"
#include "./merge_systems_template.F90"
#undef DOUBLE_PRECISION_REAL
#undef REALCASE
#undef DOUBLE_PRECISION

#ifdef WANT_SINGLE_PRECISION_REAL
! real single precision first
#define SINGLE_PRECISION_REAL
#define REALCASE
#define SINGLE_PRECISION
#include "../general/precision_macros.h"
#include "./merge_systems_template.F90"
#undef SINGLE_PRECISION_REAL
#undef REALCASE
#undef SINGLE_PRECISION
#endif

end module
