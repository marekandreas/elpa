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

#include <stdio.h>
#include <math.h>
#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <stdint.h>
#include <complex.h>
//missing header for rocblas
#include "rocblas.h"
#include "hip/hip_runtime_api.h"
#include "config-f90.h"

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#ifdef DEBUG_HIP
#define debugmessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)
#else
#define debugmessage(x, ...)
#endif

#ifdef WITH_AMD_GPU_VERSION
extern "C" {

  int rocblasCreateFromC(intptr_t *handle) {
//     printf("in c: %p\n", *cublas_handle);
    *handle = (intptr_t) malloc(sizeof(rocblas_handle));
//     printf("in c: %p\n", *cublas_handle);
    rocblas_status status = rocblas_create_handle((rocblas_handle*) *handle);
    if (status == rocblas_status_success) {
//       printf("all OK\n");
      return 1;
    }
    else if (status == rocblas_status_invalid_handle) {
      errormessage("Error in rocblas_create_handle: %s\n", "the rocblas Runtime initialization failed");
      return 0;
    }
    else if (status == rocblas_status_memory_error) {
      errormessage("Error in rocblas_create_handle: %s\n", "the resources could not be allocated");
      return 0;
    }
    else{
      errormessage("Error in rocblas_create_handle: %s\n", "unknown error");
      return 0;
    }
  }

  int rocblasDestroyFromC(intptr_t *handle) {
    rocblas_status status = rocblas_destroy_handle(*((rocblas_handle*) *handle));
    *handle = (intptr_t) NULL;
    if (status == rocblas_status_success) {
//       printf("all OK\n");
      return 1;
    }
    else if (status == rocblas_status_invalid_handle) {
      errormessage("Error in rocblas_destroy_handle: %s\n", "the library has not been initialized");
      return 0;
    }
    else{
      errormessage("Error in rocblas_destroy_handle: %s\n", "unknown error");
      return 0;
    }
  }

  int hipSetDeviceFromC(int n) {

    hipError_t hiperr = hipSetDevice(n);
    if (hiperr != hipSuccess) {
      errormessage("Error in hipSetDevice: %s\n",hipGetErrorString(hiperr));
      return 0;
    }
    return 1;
  }

  int hipGetDeviceCountFromC(int *count) {

    hipError_t hiperr = hipGetDeviceCount(count);
    if (hiperr != hipSuccess) {
      errormessage("Error in hipGetDeviceCount: %s\n",hipGetErrorString(hiperr));
      return 0;
    }
    return 1;
  }

  int hipDeviceSynchronizeFromC() {

    hipError_t hiperr = hipDeviceSynchronize();
    if (hiperr != hipSuccess) {
      errormessage("Error in hipDeviceSynchronize: %s\n",hipGetErrorString(hiperr));
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

  int hipHostMallocFromC(intptr_t *a, size_t width_height) {

    hipError_t hiperr = hipHostMalloc((void **) a, width_height, hipHostMallocMapped);
#ifdef DEBUG_HIP
    printf("MallocHost pointer address: %p \n", *a);
#endif
    if (hiperr != hipSuccess) {
      errormessage("Error in hipHostMalloc: %s\n",hipGetErrorString(hiperr));
      return 0;
    }
    return 1;
  }

  int hipHostFreeFromC(intptr_t *a) {
#ifdef DEBUG_HIP
    printf("FreeHost pointer address: %p \n", a);
#endif
    hipError_t hiperr = hipHostFree(a);

    if (hiperr != hipSuccess) {
      errormessage("Error in hipHostFree: %s\n",hipGetErrorString(hiperr));
      return 0;
    }
    return 1;
  }

  int hipMemsetFromC(intptr_t *a, int value, size_t count) {

    hipError_t hiperr = hipMemset( a, value, count);
    if (hiperr != hipSuccess) {
      errormessage("Error in hipMemset: %s\n",hipGetErrorString(hiperr));
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

  int hipMemcpy2dFromC(intptr_t *dest, size_t dpitch, intptr_t *src, size_t spitch, size_t width, size_t height, int dir) {

    hipError_t hiperr = hipMemcpy2D( dest, dpitch, src, spitch, width, height, (hipMemcpyKind)dir);
    if (hiperr != hipSuccess) {
      errormessage("Error in hipMemcpy2d: %s\n",hipGetErrorString(hiperr));
      return 0;
    }
    return 1;
  }

  int hipHostRegisterFromC(intptr_t *a, int value, int flag) {

    hipError_t hiperr = hipHostRegister( a, value, (unsigned int)flag);
    if (hiperr != hipSuccess) {
      errormessage("Error in hipHostRegister: %s\n",hipGetErrorString(hiperr));
      return 0;
    }
    return 1;
  }

  int hipHostUnregisterFromC(intptr_t *a) {

    hipError_t hiperr = hipHostUnregister( a);
    if (hiperr != hipSuccess) {
      errormessage("Error in hipHostUnregister: %s\n",hipGetErrorString(hiperr));
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

  rocblas_operation hip_operation(char trans) {
    if (trans == 'N' || trans == 'n') {
      return rocblas_operation_none;
    }
    else if (trans == 'T' || trans == 't') {
      return rocblas_operation_transpose;
    }
    else if (trans == 'C' || trans == 'c') {
      return rocblas_operation_conjugate_transpose;
    }
    else {
      errormessage("Error when transfering %c to rocblas_Operation_t\n",trans);
      // or abort?
      return rocblas_operation_none;
    }
  }


  rocblas_fill hip_fill_mode(char uplo) {
    if (uplo == 'L' || uplo == 'l') {
      return rocblas_fill_lower;
    }
    else if(uplo == 'U' || uplo == 'u') {
      return rocblas_fill_upper;
    }
    else {
      errormessage("Error when transfering %c to cublasFillMode_t\n", uplo);
      // or abort?
      return rocblas_fill_lower;
    }
  }

  rocblas_side hip_side_mode(char side) {
    if (side == 'L' || side == 'l') {
      return rocblas_side_left;
    }
    else if (side == 'R' || side == 'r') {
      return rocblas_side_right;
    }
    else{
      errormessage("Error when transfering %c to rocblas_side\n", side);
      // or abort?
      return rocblas_side_left;
    }
  }

  rocblas_diagonal hip_diag_type(char diag) {
    if (diag == 'N' || diag == 'n') {
      return rocblas_diagonal_non_unit;
    }
    else if (diag == 'U' || diag == 'u') {
      return rocblas_diagonal_unit;
    }
    else {
      errormessage("Error when transfering %c to rocblas_diag\n", diag);
      // or abort?
      return rocblas_diagonal_non_unit;
    }
  }


  void rocblas_dgemv_elpa_wrapper (intptr_t handle, char trans, int m, int n, double alpha,
                               const double *A, int lda,  const double *x, int incx,
                               double beta, double *y, int incy) {

    rocblas_dgemv(*((rocblas_handle*)handle), hip_operation(trans),
                m, n, &alpha, A, lda, x, incx, &beta, y, incy);
  }

  void rocblas_sgemv_elpa_wrapper (intptr_t handle, char trans, int m, int n, float alpha,
                               const float *A, int lda,  const float *x, int incx,
                               float beta, float *y, int incy) {

    rocblas_sgemv(*((rocblas_handle*)handle), hip_operation(trans),
                m, n, &alpha, A, lda, x, incx, &beta, y, incy);
  }

  void rocblas_zgemv_elpa_wrapper (intptr_t handle, char trans, int m, int n, double _Complex alpha,
                               const double _Complex *A, int lda,  const double _Complex *x, int incx,
                               double _Complex beta, double _Complex *y, int incy) {

    rocblas_double_complex alpha_casted = *((rocblas_double_complex*)(&alpha));
    rocblas_double_complex beta_casted = *((rocblas_double_complex*)(&beta));

    const rocblas_double_complex* A_casted = (const rocblas_double_complex*) A;
    const rocblas_double_complex* x_casted = (const rocblas_double_complex*) x;
    rocblas_double_complex* y_casted = (rocblas_double_complex*) y;

    rocblas_zgemv(*((rocblas_handle*)handle), hip_operation(trans),
                m, n, &alpha_casted, A_casted, lda, x_casted, incx, &beta_casted, y_casted, incy);
  }

  void rocblas_cgemv_elpa_wrapper (intptr_t handle, char trans, int m, int n, float _Complex alpha,
                               const float _Complex *A, int lda,  const float _Complex *x, int incx,
                               float _Complex beta, float _Complex *y, int incy) {

    rocblas_float_complex alpha_casted = *((rocblas_float_complex*)(&alpha));
    rocblas_float_complex beta_casted = *((rocblas_float_complex*)(&beta));

    const rocblas_float_complex* A_casted = (const rocblas_float_complex*) A;
    const rocblas_float_complex* x_casted = (const rocblas_float_complex*) x;
    rocblas_float_complex* y_casted = (rocblas_float_complex*) y;

    rocblas_cgemv(*((rocblas_handle*)handle), hip_operation(trans),
                m, n, &alpha_casted, A_casted, lda, x_casted, incx, &beta_casted, y_casted, incy);
  }


  void rocblas_dgemm_elpa_wrapper (intptr_t handle, char transa, char transb, int m, int n, int k,
                               double alpha, const double *A, int lda,
                               const double *B, int ldb, double beta,
                               double *C, int ldc) {

    rocblas_dgemm(*((rocblas_handle*)handle), hip_operation(transa), hip_operation(transb),
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

  void rocblas_sgemm_elpa_wrapper (intptr_t handle, char transa, char transb, int m, int n, int k,
                               float alpha, const float *A, int lda,
                               const float *B, int ldb, float beta,
                               float *C, int ldc) {

    rocblas_sgemm(*((rocblas_handle*)handle), hip_operation(transa), hip_operation(transb),
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

  void rocblas_zgemm_elpa_wrapper (intptr_t handle, char transa, char transb, int m, int n, int k,
                               double _Complex alpha, const double _Complex *A, int lda,
                               const double _Complex *B, int ldb, double _Complex beta,
                               double _Complex *C, int ldc) {

    rocblas_double_complex alpha_casted = *((rocblas_double_complex*)(&alpha));
    rocblas_double_complex beta_casted = *((rocblas_double_complex*)(&beta));

    const rocblas_double_complex* A_casted = (const rocblas_double_complex*) A;
    const rocblas_double_complex* B_casted = (const rocblas_double_complex*) B;
    rocblas_double_complex* C_casted = (rocblas_double_complex*) C;

    rocblas_zgemm(*((rocblas_handle*)handle), hip_operation(transa), hip_operation(transb),
                m, n, k, &alpha_casted, A_casted, lda, B_casted, ldb, &beta_casted, C_casted, ldc);
  }

  void rocblas_cgemm_elpa_wrapper (intptr_t handle, char transa, char transb, int m, int n, int k,
                               float _Complex alpha, const float _Complex *A, int lda,
                               const float _Complex *B, int ldb, float _Complex beta,
                               float _Complex *C, int ldc) {

    rocblas_float_complex alpha_casted = *((rocblas_float_complex*)(&alpha));
    rocblas_float_complex beta_casted = *((rocblas_float_complex*)(&beta));

    const rocblas_float_complex* A_casted = (const rocblas_float_complex*) A;
    const rocblas_float_complex* B_casted = (const rocblas_float_complex*) B;
    rocblas_float_complex* C_casted = (rocblas_float_complex*) C;

    rocblas_cgemm(*((rocblas_handle*)handle), hip_operation(transa), hip_operation(transb),
                m, n, k, &alpha_casted, A_casted, lda, B_casted, ldb, &beta_casted, C_casted, ldc);
  }


  // todo: new CUBLAS API diverged from standard BLAS api for these functions
  // todo: it provides out-of-place (and apparently more efficient) implementation
  // todo: by passing B twice (in place of C as well), we should fall back to in-place algorithm

  void rocblas_dtrmm_elpa_wrapper (intptr_t handle, char side, char uplo, char transa, char diag,
                               int m, int n, double alpha, const double *A,
                               int lda, double *B, int ldb){

    rocblas_dtrmm(*((rocblas_handle*)handle), hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha, A, lda, B, ldb);
  }

  void rocblas_strmm_elpa_wrapper (intptr_t handle, char side, char uplo, char transa, char diag,
                               int m, int n, float alpha, const float *A,
                               int lda, float *B, int ldb){
    rocblas_strmm(*((rocblas_handle*)handle), hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha, A, lda, B, ldb);
  }

  void rocblas_ztrmm_elpa_wrapper (intptr_t handle, char side, char uplo, char transa, char diag,
                               int m, int n, double _Complex alpha, const double _Complex *A,
                               int lda, double _Complex *B, int ldb){

    rocblas_double_complex alpha_casted = *((rocblas_double_complex*)(&alpha));

    const rocblas_double_complex* A_casted = (const rocblas_double_complex*) A;
    rocblas_double_complex* B_casted = (rocblas_double_complex*) B;
    rocblas_ztrmm(*((rocblas_handle*)handle), hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb);
  }

  void rocblas_ctrmm_elpa_wrapper (intptr_t handle, char side, char uplo, char transa, char diag,
                               int m, int n, float _Complex alpha, const float _Complex *A,
                               int lda, float _Complex *B, int ldb){

    rocblas_float_complex alpha_casted = *((rocblas_float_complex*)(&alpha));

    const rocblas_float_complex* A_casted = (const rocblas_float_complex*) A;
    rocblas_float_complex* B_casted = (rocblas_float_complex*) B;
    rocblas_ctrmm(*((rocblas_handle*)handle), hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb);
  }


}
#endif /* WITH_AMD_GPU_VERSION */
