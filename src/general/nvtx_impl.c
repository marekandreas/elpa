/* nvtx_impl.c — Provide externally-visible definitions of NVTX3 functions.
 *
 * NVTX v3 is a header-only library.  By default, `nvtxRangePushA` and friends
 * are declared `static inline`, which means no linkable symbol is emitted.
 *
 * ELPA's Fortran code references these functions via `bind(C, name=...)`,
 * which requires real symbols at link time.  Defining NVTX_EXPORT_API before
 * including the NVTX3 header causes the implementation to be emitted with
 * external (non-static) linkage, providing exactly the symbols Fortran needs.
 *
 * This file must be compiled into the ELPA library when WITH_NVTX is enabled.
 */
#define NVTX_EXPORT_API
#include <nvtx3/nvToolsExt.h>
