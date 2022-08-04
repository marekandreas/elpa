//
//    Copyright 2021, A. Marek
//
//    This file is part of ELPA.
//
//    The ELPA library was originally created by the ELPA consortium,
//    consisting of the following organizations:
//
//    - Max Planck Computing and Data Facility (MPCDF), formerly known as
//      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
//    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
//      Informatik,
//    - Technische Universität München, Lehrstuhl für Informatik mit
//      Schwerpunkt Wissenschaftliches Rechnen ,
//    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
//    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
//      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
//      and
//    - IBM Deutschland GmbH
//
//    This particular source code file contains additions, changes and
//    enhancements authored by Intel Corporation which is not part of
//    the ELPA consortium.
//
//    More information can be found here:
//    http://elpa.mpcdf.mpg.de/
//
//    ELPA is free software: you can redistribute it and/or modify
//    it under the terms of the version 3 of the license of the
//    GNU Lesser General Public License as published by the Free
//    Software Foundation.
//
//    ELPA is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU Lesser General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
//
//    ELPA reflects a substantial effort on the part of the original
//    ELPA consortium, and we ask you to respect the spirit of the
//    license that we chose: i.e., please contribute any changes you
//    may have back to the original ELPA library distribution, and keep
//    any derivatives of ELPA under the same license that we chose for
//    the original distribution, the GNU Lesser General Public License.
//
//
// --------------------------------------------------------------------------------------------------
//
// This file was written by A. Marek, MPCDF
#include "config-f90.h"

#include <stdio.h>
#include <math.h>
#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <stdint.h>
#include <complex.h>
#ifdef WITH_AMD_GPU_VERSION

#ifdef HIPBLAS
#include "hipblas.h"
#else
#include "rocblas.h"
#endif

#include "hip/hip_runtime_api.h"
#endif

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#ifdef DEBUG_HIP
#define debugmessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)
#else
#define debugmessage(x, ...)
#endif

#ifdef WITH_AMD_GPU_VERSION
extern "C" {

  int hipSetDeviceFromC(int n) {

    hipError_t hiperr = hipSetDevice(n);
    if (hiperr != hipSuccess) {
      errormessage("Error in hipSetDevice: %s\n",hipGetErrorString(hiperr));
      return 0;
    }
    return 1;
  }

  int hipMallocFromC(intptr_t *a, size_t width_height) {

    hipError_t hiperr = hipMalloc((void **) a, width_height);
#ifdef DEBUG_HIP
    printf("HIP Malloc,  pointer address: %p, size: %d \n", *a, width_height);
#endif
    if (hiperr != hipSuccess) {
      errormessage("Error in hipMalloc: %s\n",hipGetErrorString(hiperr));
      return 0;
    }
    return 1;
  }

  int hipFreeFromC(intptr_t *a) {
#ifdef DEBUG_HIP
    printf("HIP Free, pointer address: %p \n", a);
#endif
    hipError_t hiperr = hipFree(a);

    if (hiperr != hipSuccess) {
      errormessage("Error in hipFree: %s\n",hipGetErrorString(hiperr));
      return 0;
    }
    return 1;
  }

  int hipMemcpyFromC(intptr_t *dest, intptr_t *src, size_t count, int dir) {

    hipError_t hiperr = hipMemcpy( dest, src, count, (hipMemcpyKind)dir);
    if (hiperr != hipSuccess) {
      errormessage("Error in hipMemcpy: %s\n",hipGetErrorString(hiperr));
      return 0;
    }
    return 1;
  }

  int hipMemcpyDeviceToDeviceFromC(void) {
      int val = (int)hipMemcpyDeviceToDevice;
      return val;
  }
  int hipMemcpyHostToDeviceFromC(void) {
      int val = (int)hipMemcpyHostToDevice;
      return val;
  }
  int hipMemcpyDeviceToHostFromC(void) {
      int val = (int)hipMemcpyDeviceToHost;
      return val;
  }
  int hipHostRegisterDefaultFromC(void) {
      int val = (int)hipHostRegisterDefault;
      return val;
  }
  int hipHostRegisterPortableFromC(void) {
      int val = (int)hipHostRegisterPortable;
      return val;
  }
  int hipHostRegisterMappedFromC(void) {
      int val = (int)hipHostRegisterMapped;
      return val;
  }

}
#endif /* TEST_AMD_GPU == 1 */
