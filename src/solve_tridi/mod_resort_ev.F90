
#include "config-f90.h"
module resort_ev
  use precision
  implicit none
  private

  public :: resort_ev_cpu_double
  public :: resort_ev_gpu_double
#if defined(WANT_SINGLE_PRECISION_REAL) || defined(WANT_SINGLE_PRECISION_COMPLEX)
  public :: resort_ev_cpu_single
  public :: resort_ev_gpu_single
#endif

  contains

! real double precision first
#define DOUBLE_PRECISION_REAL
#define REALCASE
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"
#undef SOLVE_TRIDI_GPU_BUILD
#include "./resort_ev_template.F90"
#define SOLVE_TRIDI_GPU_BUILD
#include "./resort_ev_template.F90"
#undef SOLVE_TRIDI_GPU_BUILD
#undef REALCASE
#undef DOUBLE_PRECISION

#ifdef WANT_SINGLE_PRECISION_REAL
! real single precision first
#define SINGLE_PRECISION_REAL
#define REALCASE
#define SINGLE_PRECISION
#include "../general/precision_macros.h"
#undef SOLVE_TRIDI_GPU_BUILD
#include "./resort_ev_template.F90"
#define SOLVE_TRIDI_GPU_BUILD
#include "./resort_ev_template.F90"
#undef SOLVE_TRIDI_GPU_BUILD
#undef REALCASE
#undef SINGLE_PRECISION
#endif

end module
