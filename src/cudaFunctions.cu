#include <stdio.h>
#include <math.h>
#include <stdio.h>
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


#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <stdint.h>
#include <complex.h>
#include <cublas.h>

#include "config-f90.h"

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#ifdef DEBUG_CUDA
#define debugmessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)
#else
#define debugmessage(x, ...)
#endif

#ifdef WITH_GPU_VERSION
extern "C" {

  int cudaThreadSynchronizeFromC() {
    cudaError_t cuerr = cudaThreadSynchronize();
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaThreadSynchronize: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }


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
  
  void cublasZgemv_elpa_wrapper (char trans, int m, int n, double complex alpha,
                               const double complex *A, int lda,  const double complex *x, int incx,
                               double complex beta, double complex *y, int incy) {    

    cuDoubleComplex alpha_casted = *((cuDoubleComplex*)(&alpha));
    cuDoubleComplex beta_casted = *((cuDoubleComplex*)(&beta));
    
    const cuDoubleComplex* A_casted = (const cuDoubleComplex*) A;
    const cuDoubleComplex* x_casted = (const cuDoubleComplex*) x;
    cuDoubleComplex* y_casted = (cuDoubleComplex*) y;
    
    cublasZgemv(trans, m, n, alpha_casted, A_casted, lda, x_casted, incx, beta_casted, y_casted, incy);     
  }
  
  void cublasCgemv_elpa_wrapper (char trans, int m, int n, float complex alpha,
                               const float complex *A, int lda,  const float complex *x, int incx,
                               float complex beta, float complex *y, int incy) {    

    cuFloatComplex alpha_casted = *((cuFloatComplex*)(&alpha));
    cuFloatComplex beta_casted = *((cuFloatComplex*)(&beta));
    
    const cuFloatComplex* A_casted = (const cuFloatComplex*) A;
    const cuFloatComplex* x_casted = (const cuFloatComplex*) x;
    cuFloatComplex* y_casted = (cuFloatComplex*) y;
    
    cublasCgemv(trans, m, n, alpha_casted, A_casted, lda, x_casted, incx, beta_casted, y_casted, incy);     
  }
  
  void cublasZgemm_elpa_wrapper (char transa, char transb, int m, int n, int k,
                               double complex alpha, const double complex *A, int lda,
                               const double complex *B, int ldb, double complex beta,
                               double complex *C, int ldc) {
    
    cuDoubleComplex alpha_casted = *((cuDoubleComplex*)(&alpha));
    cuDoubleComplex beta_casted = *((cuDoubleComplex*)(&beta));
    
    const cuDoubleComplex* A_casted = (const cuDoubleComplex*) A;
    const cuDoubleComplex* B_casted = (const cuDoubleComplex*) B;
    cuDoubleComplex* C_casted = (cuDoubleComplex*) C;
    
    cublasZgemm(transa, transb, m, n, k, alpha_casted, A_casted, lda, B_casted, ldb, beta_casted, C_casted, ldc);
  }

  void cublasCgemm_elpa_wrapper (char transa, char transb, int m, int n, int k,
                               float complex alpha, const float complex *A, int lda,
                               const float complex *B, int ldb, float complex beta,
                               float complex *C, int ldc) {
    
    cuFloatComplex alpha_casted = *((cuFloatComplex*)(&alpha));
    cuFloatComplex beta_casted = *((cuFloatComplex*)(&beta));
    
    const cuFloatComplex* A_casted = (const cuFloatComplex*) A;
    const cuFloatComplex* B_casted = (const cuFloatComplex*) B;
    cuFloatComplex* C_casted = (cuFloatComplex*) C;
    
    cublasCgemm(transa, transb, m, n, k, alpha_casted, A_casted, lda, B_casted, ldb, beta_casted, C_casted, ldc);
  }

  void cublasZtrmm_elpa_wrapper (char side, char uplo, char transa, char diag,
                               int m, int n, double complex alpha, const double complex *A,
                               int lda, double complex *B, int ldb){

    cuDoubleComplex alpha_casted = *((cuDoubleComplex*)(&alpha));
    
    const cuDoubleComplex* A_casted = (const cuDoubleComplex*) A;
    cuDoubleComplex* B_casted = (cuDoubleComplex*) B;    
    
    cublasZtrmm(side, uplo, transa, diag, m, n, alpha_casted, A_casted, lda, B_casted, ldb);
  }

  void cublasCtrmm_elpa_wrapper (char side, char uplo, char transa, char diag,
                               int m, int n, float complex alpha, const float complex *A,
                               int lda, float complex *B, int ldb){

    cuFloatComplex alpha_casted = *((cuFloatComplex*)(&alpha));
    
    const cuFloatComplex* A_casted = (const cuFloatComplex*) A;
    cuFloatComplex* B_casted = (cuFloatComplex*) B;    
    
    cublasCtrmm(side, uplo, transa, diag, m, n, alpha_casted, A_casted, lda, B_casted, ldb);
  }

  
}
#endif /* WITH_GPU_VERSION */
