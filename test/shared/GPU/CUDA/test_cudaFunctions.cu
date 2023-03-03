//
//    Copyright 2014, A. Marek
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
#ifdef WITH_NVIDIA_GPU_VERSION
#include <cublas_v2.h>
#endif

#include "./test_cudaFunctions.h"

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#ifdef DEBUG_CUDA
#define debugmessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)
#else
#define debugmessage(x, ...)
#endif

#ifdef WITH_NVIDIA_GPU_VERSION
extern "C"
{
  int cudaSetDeviceFromC(int n) {

    cudaError_t cuerr = cudaSetDevice(n);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaSetDevice(%i): %s\n", n, cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaGetDeviceCountFromC(int *count) {

    cudaError_t cuerr = cudaGetDeviceCount(count);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaGetDeviceCount: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }
  
  int cudaMallocFromC(intptr_t *a, size_t width_height) {

    cudaError_t cuerr = cudaMalloc((void **) a, width_height);
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *a, width_height);
#endif
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaMalloc: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }
  
  int cudaFreeFromC(intptr_t *a) {
#ifdef DEBUG_CUDA
    printf("CUDA Free, pointer address: %p \n", a);
#endif
    cudaError_t cuerr = cudaFree(a);

    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaFree: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaMemcpyFromC(intptr_t *dest, intptr_t *src, size_t count, int dir) {

    cudaError_t cuerr = cudaMemcpy( dest, src, count, (cudaMemcpyKind)dir);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaMemcpy: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaMemcpyDeviceToDeviceFromC(void) {
      int val = cudaMemcpyDeviceToDevice;
      return val;
  }
  int cudaMemcpyHostToDeviceFromC(void) {
      int val = cudaMemcpyHostToDevice;
      return val;
  }
  int cudaMemcpyDeviceToHostFromC(void) {
      int val = cudaMemcpyDeviceToHost;
      return val;
  }
  int cudaHostRegisterDefaultFromC(void) {
      int val = cudaHostRegisterDefault;
      return val;
  }
  int cudaHostRegisterPortableFromC(void) {
      int val = cudaHostRegisterPortable;
      return val;
  }
  int cudaHostRegisterMappedFromC(void) {
      int val = cudaHostRegisterMapped;
      return val;
  }
}
#endif /* TEST_NVIDIA_GPU == 1 */
