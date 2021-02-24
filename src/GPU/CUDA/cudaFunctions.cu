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

#include <stdio.h>
#include <math.h>
#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <stdint.h>
#include <complex.h>
#include <cublas_v2.h>

#include "config-f90.h"

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#ifdef DEBUG_CUDA
#define debugmessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)
#else
#define debugmessage(x, ...)
#endif

#ifdef WITH_NVIDIA_GPU_VERSION
extern "C" {

  int cublasCreateFromC(intptr_t *cublas_handle) {
//     printf("in c: %p\n", *cublas_handle);
    *cublas_handle = (intptr_t) malloc(sizeof(cublasHandle_t));
//     printf("in c: %p\n", *cublas_handle);
    cublasStatus_t status = cublasCreate((cublasHandle_t*) *cublas_handle);
    if (status == CUBLAS_STATUS_SUCCESS) {
//       printf("all OK\n");
      return 1;
    }
    else if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
      errormessage("Error in cublasCreate: %s\n", "the CUDA Runtime initialization failed");
      return 0;
    }
    else if (status == CUBLAS_STATUS_ALLOC_FAILED) {
      errormessage("Error in cublasCreate: %s\n", "the resources could not be allocated");
      return 0;
    }
    else{
      errormessage("Error in cublasCreate: %s\n", "unknown error");
      return 0;
    }
  }

  int cublasDestroyFromC(intptr_t *cublas_handle) {
    cublasStatus_t status = cublasDestroy(*((cublasHandle_t*) *cublas_handle));
    *cublas_handle = (intptr_t) NULL;
    if (status == CUBLAS_STATUS_SUCCESS) {
//       printf("all OK\n");
      return 1;
    }
    else if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
      errormessage("Error in cublasDestroy: %s\n", "the library has not been initialized");
      return 0;
    }
    else{
      errormessage("Error in cublasDestroy: %s\n", "unknown error");
      return 0;
    }
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
      errormessage("Error in cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
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

  int cudaMallocHostFromC(intptr_t *a, size_t width_height) {

    cudaError_t cuerr = cudaMallocHost((void **) a, width_height);
#ifdef DEBUG_CUDA
    printf("MallocHost pointer address: %p \n", *a);
#endif
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaMallocHost: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaFreeHostFromC(intptr_t *a) {
#ifdef DEBUG_CUDA
    printf("FreeHost pointer address: %p \n", a);
#endif
    cudaError_t cuerr = cudaFreeHost(a);

    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaFreeHost: %s\n",cudaGetErrorString(cuerr));
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

  int cudaHostRegisterFromC(intptr_t *a, int value, int flag) {

    cudaError_t cuerr = cudaHostRegister( a, value, flag);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaHostRegister: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cudaHostUnregisterFromC(intptr_t *a) {

    cudaError_t cuerr = cudaHostUnregister( a);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaHostUnregister: %s\n",cudaGetErrorString(cuerr));
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

  cublasOperation_t operation_new_api(char trans) {
    if (trans == 'N' || trans == 'n') {
      return CUBLAS_OP_N;
    }
    else if (trans == 'T' || trans == 't') {
      return CUBLAS_OP_T;
    }
    else if (trans == 'C' || trans == 'c') {
      return CUBLAS_OP_C;
    }
    else {
      errormessage("Error when transfering %c to cublasOperation_t\n",trans);
      // or abort?
      return CUBLAS_OP_N;
    }
  }


  cublasFillMode_t fill_mode_new_api(char uplo) {
    if (uplo == 'L' || uplo == 'l') {
      return CUBLAS_FILL_MODE_LOWER;
    }
    else if(uplo == 'U' || uplo == 'u') {
      return CUBLAS_FILL_MODE_UPPER;
    }
    else {
      errormessage("Error when transfering %c to cublasFillMode_t\n", uplo);
      // or abort?
      return CUBLAS_FILL_MODE_LOWER;
    }
  }

  cublasSideMode_t side_mode_new_api(char side) {
    if (side == 'L' || side == 'l') {
      return CUBLAS_SIDE_LEFT;
    }
    else if (side == 'R' || side == 'r') {
      return CUBLAS_SIDE_RIGHT;
    }
    else{
      errormessage("Error when transfering %c to cublasSideMode_t\n", side);
      // or abort?
      return CUBLAS_SIDE_LEFT;
    }
  }

  cublasDiagType_t diag_type_new_api(char diag) {
    if (diag == 'N' || diag == 'n') {
      return CUBLAS_DIAG_NON_UNIT;
    }
    else if (diag == 'U' || diag == 'u') {
      return CUBLAS_DIAG_UNIT;
    }
    else {
      errormessage("Error when transfering %c to cublasDiagMode_t\n", diag);
      // or abort?
      return CUBLAS_DIAG_NON_UNIT;
    }
  }


  void cublasDgemv_elpa_wrapper (intptr_t handle, char trans, int m, int n, double alpha,
                               const double *A, int lda,  const double *x, int incx,
                               double beta, double *y, int incy) {

    cublasDgemv(*((cublasHandle_t*)handle), operation_new_api(trans),
                m, n, &alpha, A, lda, x, incx, &beta, y, incy);
  }

  void cublasSgemv_elpa_wrapper (intptr_t handle, char trans, int m, int n, float alpha,
                               const float *A, int lda,  const float *x, int incx,
                               float beta, float *y, int incy) {

    cublasSgemv(*((cublasHandle_t*)handle), operation_new_api(trans),
                m, n, &alpha, A, lda, x, incx, &beta, y, incy);
  }

  void cublasZgemv_elpa_wrapper (intptr_t handle, char trans, int m, int n, double _Complex alpha,
                               const double _Complex *A, int lda,  const double _Complex *x, int incx,
                               double _Complex beta, double _Complex *y, int incy) {

    cuDoubleComplex alpha_casted = *((cuDoubleComplex*)(&alpha));
    cuDoubleComplex beta_casted = *((cuDoubleComplex*)(&beta));

    const cuDoubleComplex* A_casted = (const cuDoubleComplex*) A;
    const cuDoubleComplex* x_casted = (const cuDoubleComplex*) x;
    cuDoubleComplex* y_casted = (cuDoubleComplex*) y;

    cublasZgemv(*((cublasHandle_t*)handle), operation_new_api(trans),
                m, n, &alpha_casted, A_casted, lda, x_casted, incx, &beta_casted, y_casted, incy);
  }

  void cublasCgemv_elpa_wrapper (intptr_t handle, char trans, int m, int n, float _Complex alpha,
                               const float _Complex *A, int lda,  const float _Complex *x, int incx,
                               float _Complex beta, float _Complex *y, int incy) {

    cuFloatComplex alpha_casted = *((cuFloatComplex*)(&alpha));
    cuFloatComplex beta_casted = *((cuFloatComplex*)(&beta));

    const cuFloatComplex* A_casted = (const cuFloatComplex*) A;
    const cuFloatComplex* x_casted = (const cuFloatComplex*) x;
    cuFloatComplex* y_casted = (cuFloatComplex*) y;

    cublasCgemv(*((cublasHandle_t*)handle), operation_new_api(trans),
                m, n, &alpha_casted, A_casted, lda, x_casted, incx, &beta_casted, y_casted, incy);
  }


  void cublasDgemm_elpa_wrapper (intptr_t handle, char transa, char transb, int m, int n, int k,
                               double alpha, const double *A, int lda,
                               const double *B, int ldb, double beta,
                               double *C, int ldc) {

    cublasDgemm(*((cublasHandle_t*)handle), operation_new_api(transa), operation_new_api(transb),
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

  void cublasSgemm_elpa_wrapper (intptr_t handle, char transa, char transb, int m, int n, int k,
                               float alpha, const float *A, int lda,
                               const float *B, int ldb, float beta,
                               float *C, int ldc) {

    cublasSgemm(*((cublasHandle_t*)handle), operation_new_api(transa), operation_new_api(transb),
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

  void cublasZgemm_elpa_wrapper (intptr_t handle, char transa, char transb, int m, int n, int k,
                               double _Complex alpha, const double _Complex *A, int lda,
                               const double _Complex *B, int ldb, double _Complex beta,
                               double _Complex *C, int ldc) {

    cuDoubleComplex alpha_casted = *((cuDoubleComplex*)(&alpha));
    cuDoubleComplex beta_casted = *((cuDoubleComplex*)(&beta));

    const cuDoubleComplex* A_casted = (const cuDoubleComplex*) A;
    const cuDoubleComplex* B_casted = (const cuDoubleComplex*) B;
    cuDoubleComplex* C_casted = (cuDoubleComplex*) C;

    cublasZgemm(*((cublasHandle_t*)handle), operation_new_api(transa), operation_new_api(transb),
                m, n, k, &alpha_casted, A_casted, lda, B_casted, ldb, &beta_casted, C_casted, ldc);
  }

  void cublasCgemm_elpa_wrapper (intptr_t handle, char transa, char transb, int m, int n, int k,
                               float _Complex alpha, const float _Complex *A, int lda,
                               const float _Complex *B, int ldb, float _Complex beta,
                               float _Complex *C, int ldc) {

    cuFloatComplex alpha_casted = *((cuFloatComplex*)(&alpha));
    cuFloatComplex beta_casted = *((cuFloatComplex*)(&beta));

    const cuFloatComplex* A_casted = (const cuFloatComplex*) A;
    const cuFloatComplex* B_casted = (const cuFloatComplex*) B;
    cuFloatComplex* C_casted = (cuFloatComplex*) C;

    cublasCgemm(*((cublasHandle_t*)handle), operation_new_api(transa), operation_new_api(transb),
                m, n, k, &alpha_casted, A_casted, lda, B_casted, ldb, &beta_casted, C_casted, ldc);
  }


  // todo: new CUBLAS API diverged from standard BLAS api for these functions
  // todo: it provides out-of-place (and apparently more efficient) implementation
  // todo: by passing B twice (in place of C as well), we should fall back to in-place algorithm

  void cublasDtrmm_elpa_wrapper (intptr_t handle, char side, char uplo, char transa, char diag,
                               int m, int n, double alpha, const double *A,
                               int lda, double *B, int ldb){

    cublasDtrmm(*((cublasHandle_t*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                diag_type_new_api(diag), m, n, &alpha, A, lda, B, ldb, B, ldb);
  }

  void cublasStrmm_elpa_wrapper (intptr_t handle, char side, char uplo, char transa, char diag,
                               int m, int n, float alpha, const float *A,
                               int lda, float *B, int ldb){

    cublasStrmm(*((cublasHandle_t*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                diag_type_new_api(diag), m, n, &alpha, A, lda, B, ldb, B, ldb);
  }

  void cublasZtrmm_elpa_wrapper (intptr_t handle, char side, char uplo, char transa, char diag,
                               int m, int n, double _Complex alpha, const double _Complex *A,
                               int lda, double _Complex *B, int ldb){

    cuDoubleComplex alpha_casted = *((cuDoubleComplex*)(&alpha));

    const cuDoubleComplex* A_casted = (const cuDoubleComplex*) A;
    cuDoubleComplex* B_casted = (cuDoubleComplex*) B;

    cublasZtrmm(*((cublasHandle_t*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                diag_type_new_api(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb, B_casted, ldb);
  }

  void cublasCtrmm_elpa_wrapper (intptr_t handle, char side, char uplo, char transa, char diag,
                               int m, int n, float _Complex alpha, const float _Complex *A,
                               int lda, float _Complex *B, int ldb){

    cuFloatComplex alpha_casted = *((cuFloatComplex*)(&alpha));

    const cuFloatComplex* A_casted = (const cuFloatComplex*) A;
    cuFloatComplex* B_casted = (cuFloatComplex*) B;

    cublasCtrmm(*((cublasHandle_t*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                diag_type_new_api(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb, B_casted, ldb);
  }


}
#endif /* WITH_NVIDIA_GPU_VERSION */
