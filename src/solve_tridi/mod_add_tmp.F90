
#include "config-f90.h"
module add_tmp
!DIR$ OPTIMIZE:1
  use precision
  implicit none
  private

  public :: add_tmp_double
#if defined(WANT_SINGLE_PRECISION_REAL) || defined(WANT_SINGLE_PRECISION_COMPLEX)
  public :: add_tmp_single
#endif

  contains

! real double precision first
#define REALCASE
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"
#define _rk _c_double
#include "./add_tmp_template.F90"
#undef REALCASE
#undef DOUBLE_PRECISION
#undef _rk

#ifdef WANT_SINGLE_PRECISION_REAL
! real single precision first
#define REALCASE
#define SINGLE_PRECISION
#include "../general/precision_macros.h"
#define _rk _c_float
#include "./add_tmp_template.F90"
#undef REALCASE
#undef SINGLE_PRECISION
#undef _rk
#endif

end module
