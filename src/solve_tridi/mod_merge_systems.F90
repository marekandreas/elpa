#include "config-f90.h"
#include "../general/nvtx_labels.h"

module merge_systems
!DIR$ OPTIMIZE:1
  use precision
  implicit none
  private

  public :: merge_systems_cpu_double
  public :: merge_systems_gpu_double
#if defined(WANT_SINGLE_PRECISION_REAL)
  public :: merge_systems_cpu_single
  public :: merge_systems_gpu_single
#endif

  contains

! real double precision first
#define DOUBLE_PRECISION_REAL
#define REALCASE
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"
#undef SOLVE_TRIDI_GPU_BUILD
#include "./merge_systems_template.F90"
#define SOLVE_TRIDI_GPU_BUILD
#include "./merge_systems_template.F90"
#undef SOLVE_TRIDI_GPU_BUILD

#undef DOUBLE_PRECISION_REAL
#undef REALCASE
#undef DOUBLE_PRECISION

#ifdef WANT_SINGLE_PRECISION_REAL
! real single precision first
#define SINGLE_PRECISION_REAL
#define REALCASE
#define SINGLE_PRECISION
#include "../general/precision_macros.h"
#undef SOLVE_TRIDI_GPU_BUILD
#include "./merge_systems_template.F90"
#define SOLVE_TRIDI_GPU_BUILD
#include "./merge_systems_template.F90"
#undef SOLVE_TRIDI_GPU_BUILD
#undef SINGLE_PRECISION_REAL
#undef REALCASE
#undef SINGLE_PRECISION
#endif

end module
