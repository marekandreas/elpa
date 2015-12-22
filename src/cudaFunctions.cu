#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <stdint.h>

#include "config-f90.h"

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#ifdef DEBUG_CUDA
#define debugmessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)
#else
#define debugmessage(x, ...)
#endif

#ifdef WITH_GPU_VERSION
extern "C" {
  int cudaSetDeviceFromC(int n) {

    cudaError_t cuerr = cudaSetDevice(n);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaSetDevice: %s\n",cudaGetErrorString(cuerr));
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

  int cudaDeviceSynchronizeFromC() {

    cudaError_t cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaGetDeviceCount: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }


  int cudaMallocFromC(intptr_t *a, size_t width_height) {

    cudaError_t cuerr = cudaMalloc((void **) a, width_height);
#ifdef DEBUG_CUDA
    printf("Malloc pointer address: %p \n", *a);
#endif
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaMalloc: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }
  int cudaFreeFromC(intptr_t *a) {
#ifdef DEBUG_CUDA
    printf("Free pointer address: %p \n", a);
#endif
    cudaError_t cuerr = cudaFree(a);

    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaFree: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaMemsetFromC(intptr_t *a, int value, size_t count) {

    cudaError_t cuerr = cudaMemset( a, value, count);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaMemset: %s\n",cudaGetErrorString(cuerr));
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

  int cudaMemcpy2dFromC(intptr_t *dest, size_t dpitch, intptr_t *src, size_t spitch, size_t width, size_t height, int dir) {

    cudaError_t cuerr = cudaMemcpy2D( dest, dpitch, src, spitch, width, height, (cudaMemcpyKind)dir);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaMemcpy2d: %s\n",cudaGetErrorString(cuerr));
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
  int cudaHostRegisterPortableFromC(void) {
      int val = cudaHostRegisterPortable;
      return val;
  }
  int cudaHostRegisterMappedFromC(void) {
      int val = cudaHostRegisterMapped;
      return val;
  }
}
#endif /* WITH_GPU_VERSION */
