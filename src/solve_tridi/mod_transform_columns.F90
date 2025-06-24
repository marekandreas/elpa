#include "config-f90.h"

module transform_columns
  use precision
  implicit none
  private

  public :: transform_columns_cpu_double
  public :: transform_columns_gpu_double
#if defined(WANT_SINGLE_PRECISION_REAL) || defined(WANT_SINGLE_PRECISION_COMPLEX)
  public :: transform_columns_cpu_single
  public :: transform_columns_gpu_single
#endif

  contains

! real double precision first
#define REALCASE
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"
#undef SOLVE_TRIDI_GPU_BUILD
#include "./transform_columns_template.F90"
#define SOLVE_TRIDI_GPU_BUILD
#include "./transform_columns_template.F90"
#undef SOLVE_TRIDI_GPU_BUILD
#undef REALCASE
#undef DOUBLE_PRECISION

#ifdef WANT_SINGLE_PRECISION_REAL
! real single precision first
#define REALCASE
#define SINGLE_PRECISION
#include "../general/precision_macros.h"
#undef SOLVE_TRIDI_GPU_BUILD
#include "./transform_columns_template.F90"
#define SOLVE_TRIDI_GPU_BUILD
#include "./transform_columns_template.F90"
#undef SOLVE_TRIDI_GPU_BUILD
#undef REALCASE
#undef SINGLE_PRECISION
#endif

end module