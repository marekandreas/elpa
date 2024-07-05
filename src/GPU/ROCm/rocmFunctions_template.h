//
//    Copyright 2021 - 2023, A. Marek
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

// hipStream_t elpa_hip_stm;

extern "C" {
  int hipDeviceGetAttributeFromC(int *value, int attribute) {

    hipDeviceAttribute_t attr;
    switch(attribute) {
      case 0:
        attr = hipDeviceAttributeMaxThreadsPerBlock;
        break;
      case 1:
        attr = hipDeviceAttributeMaxBlockDimX;
        break;
      case 2:
        attr = hipDeviceAttributeMaxBlockDimY;
        break;
      case 3:
        attr = hipDeviceAttributeMaxBlockDimZ;
        break;
      case 4:
        attr = hipDeviceAttributeMaxGridDimX;
        break;
      case 5:
        attr = hipDeviceAttributeMaxGridDimY;
        break;
      case 6:
        attr = hipDeviceAttributeMaxGridDimZ;
        break;
      case 7:
        attr = hipDeviceAttributeWarpSize;
        break;
      case 8:
       //only for ROCm 6.x fix this
        //attr = hipDeviceAttributeMultiProcessorCount;
        break;
    }
    hipError_t status = hipDeviceGetAttribute(value, attr, 0);
    if (status == hipSuccess) {
      return 1;
    }
    else{
      errormessage("Error in hipDeviceGetAttribute: %s\n", "unknown error");
      return 0;
    }
  }

  int rocblasGetVersionFromC(BLAS_handle rocblasHandle, int *version) {
#ifdef HIPBLAS
    errormessage("Error in rocblasGetVersionFromC: %s\n", "HIPBLAS does not support rocblas_get_version_string");
    return 1;
#else
    char *buf;
    size_t len;
    BLAS_status status;

    status = rocblas_get_version_string_size(&len);
    if (status != BLAS_status_success) {
      errormessage("Error in rocblas_get_version_string_size: %s\n", "unknown error");
      return 0;
    }

    status = rocblas_get_version_string(buf, len);
    if (status == BLAS_status_success) {
      int major, minor, patch;

      if (sscanf(buf, "%d.%d.%d", &major, &minor, &patch) == 3) {
        //printf("Major: %d, Minor: %d, Patch: %d\n", major, minor, patch);
        *version = major * 10000 + minor * 100 + patch;
         return 1;
      } else {
        printf("Error parsing version string.\n");
        return 0;
      }
    }
    else{
      errormessage("Error in rocblas_get_version_string: %s\n", "unknown error");
      return 0;
    }
#endif
  }

  int hipGetLastErrorFromC() {
    hipError_t status = hipGetLastError();
    
    if (status == hipSuccess) {
      return 1;
    }
    else{
      printf("Error in executing  hipGetLastErrorFrom: %s\n", hipGetErrorString(status));
      errormessage("Error in hipGetLastError: %s\n", "unknown error");
      return 0;
    }
  }

  int hipStreamCreateFromC(hipStream_t *rocblasStream) {
    //*stream = (intptr_t) malloc(sizeof(hipStream_t));

    if (sizeof(intptr_t) != sizeof(hipStream_t)) {
      printf("Stream sizes do not match \n");
    }

    hipError_t status = hipStreamCreate(rocblasStream);
    if (status == hipSuccess) {
//       printf("all OK\n");
      return 1;
    }
    else{
      errormessage("Error in hipStreamCreate: %s\n", "unknown error");
      return 0;
    }

  }

  int hipStreamDestroyFromC(hipStream_t rocblasStream){
    hipError_t status = hipStreamDestroy(rocblasStream);
    //*stream = (intptr_t) NULL;
    if (status ==hipSuccess) {
//       printf("all OK\n");
      return 1;
    }
    else{
      errormessage("Error in hipStreamDestroy: %s\n", "unknown error");
      return 0;
    }
  }

  int hipStreamSynchronizeExplicitFromC(hipStream_t rocblasStream) {
    hipError_t status = hipStreamSynchronize(rocblasStream);
    if (status == hipSuccess) {
      return 1;
    }
    else{
      errormessage("Error in hipStreamSynchronizeExplicit: %s\n", "unknown error");
      return 0;
    }
  }

  int hipStreamSynchronizeImplicitFromC() {
    hipError_t status = hipStreamSynchronize(hipStreamPerThread);
    if (status == hipSuccess) {
      return 1;
    }
    else{
      errormessage("Error in hipStreamSynchronizeImplicit: %s\n", "unknown error");
      return 0;
    }
  }

  int rocblasSetStreamFromC(BLAS_handle rocblasHandle, hipStream_t rocblasStream) {
    //BLAS_status status = BLAS_set_stream(*((BLAS_handle*)handle), *((hipStream_t*)stream));
    BLAS_status status = BLAS_set_stream(rocblasHandle, rocblasStream);
    if (status == BLAS_status_success ) {
      return 1;
    }
    else if (status == BLAS_status_invalid_handle) {
      errormessage("Error in rocblasSetStream: %s\n", "the HIP Runtime initialization failed");
      return 0;
    }
    else{
      errormessage("Error in rocblasSetStream: %s\n", "unknown error");
      return 0;
    }
  }

  int hipMemcpy2dAsyncFromC(intptr_t *dest, size_t dpitch, intptr_t *src, size_t spitch, size_t width, size_t height, int dir, hipStream_t rocblasStream) {

    hipError_t hiperr = hipMemcpy2DAsync( dest, dpitch, src, spitch, width, height, (hipMemcpyKind)dir, rocblasStream );
    if (hiperr != hipSuccess) {
      errormessage("Error in hipMemcpy2dAsync: %s\n",hipGetErrorString(hiperr));
      return 0;
    }
    return 1;
  }


  int rocblasCreateFromC(BLAS_handle *rocblasHandle) {
//     printf("in c: %p\n", *cublas_handle);
    //*handle = (intptr_t) malloc(sizeof(BLAS_handle));
//     printf("in c: %p\n", *cublas_handle);
    if (sizeof(intptr_t) != sizeof(BLAS_handle)) {
      //errormessage("Error in rocblasCreate: sizes not the same");
        printf("ERROR on sizes\n");
      return 0;
    }

    //BLAS_status status = BLAS_create_handle((BLAS_handle*) *handle);
    BLAS_status status = BLAS_create_handle(rocblasHandle);
    if (status == BLAS_status_success) {
//       printf("all OK\n");
      return 1;
    }
    else if (status == BLAS_status_invalid_handle) {
      errormessage("Error in rocblas_create_handle: %s\n", "the rocblas Runtime initialization failed");
      return 0;
    }
#ifndef HIPBLAS
    else if (status == BLAS_status_memory_error) {
      errormessage("Error in rocblas_create_handle: %s\n", "the resources could not be allocated");
      return 0;
    }
#endif
    else{
      errormessage("Error in rocblas_create_handle: %s\n", "unknown error");
      return 0;
    }
#if 0
    if (hipStreamCreate(&elpa_hip_stm) != hipSuccess) {
        errormessage("failed to create stream, %s, %d\n", __FILE__, __LINE__);
        return 0;
    }

    if (rocblas_set_stream(*(rocblas_handle *)handle, elpa_hip_stm) != rocblas_status_success) {
        errormessage("failed to attach stream to blas handle, %s, %d\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }
#endif
  }

  int rocblasDestroyFromC(BLAS_handle rocblasHandle) {
    //BLAS_status status = BLAS_destroy_handle(*((BLAS_handle*) *handle));
    BLAS_status status = BLAS_destroy_handle(rocblasHandle);
    //*handle = (intptr_t) NULL;
    if (status == BLAS_status_success) {
//       printf("all OK\n");
      return 1;
    }
    else if (status == BLAS_status_invalid_handle) {
      errormessage("Error in rocblas_destroy_handle: %s\n", "the library has not been initialized");
      return 0;
    }
    else{
      errormessage("Error in rocblas_destroy_handle: %s\n", "unknown error");
      return 0;
    }
#if 0
    hipStreamDestroy(elpa_hip_stm);
#endif
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

  int hipMallocHostFromC(intptr_t *a, size_t width_height) {

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

  int hipFreeHostFromC(intptr_t *a) {
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

  int hipMemsetAsyncFromC(intptr_t *a, int value, size_t count, hipStream_t rocblasStream) {

    hipError_t hiperr = hipMemsetAsync( a, value, count, rocblasStream);
    if (hiperr != hipSuccess) {
      errormessage("Error in hipMemsetAsync: %s\n",hipGetErrorString(hiperr));
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

  int hipMemcpyAsyncFromC(intptr_t *dest, intptr_t *src, size_t count, int dir, hipStream_t rocblasStream) {
  
    hipError_t hiperr = hipMemcpyAsync( dest, src, count, (hipMemcpyKind)dir, rocblasStream);
    if (hiperr != hipSuccess) {
      errormessage("Error in hipMemcpyAsync: %s\n",hipGetErrorString(hiperr));
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

  int hipHostRegisterFromC(intptr_t *a, intptr_t value, int flag) {

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

  BLAS_operation hip_operation(char trans) {
    if (trans == 'N' || trans == 'n') {
      return BLAS_operation_none;
    }
    else if (trans == 'T' || trans == 't') {
      return BLAS_operation_transpose;
    }
    else if (trans == 'C' || trans == 'c') {
      return BLAS_operation_conjugate_transpose;
    }
    else {
      errormessage("Error when transfering %c to rocblas_Operation_t\n",trans);
      // or abort?
      return BLAS_operation_none;
    }
  }

  BLAS_fill hip_fill_mode(char uplo) {
    if (uplo == 'L' || uplo == 'l') {
      return BLAS_fill_lower;
    }
    else if(uplo == 'U' || uplo == 'u') {
      return BLAS_fill_upper;
    }
    else {
      errormessage("Error when transfering %c to rocblasFillMode_t\n", uplo);
      // or abort?
      return BLAS_fill_lower;
    }
  }

  BLAS_side hip_side_mode(char side) {
    if (side == 'L' || side == 'l') {
      return BLAS_side_left;
    }
    else if (side == 'R' || side == 'r') {
      return BLAS_side_right;
    }
    else{
      errormessage("Error when transfering %c to rocblas_side\n", side);
      // or abort?
      return BLAS_side_left;
    }
  }

  BLAS_diagonal hip_diag_type(char diag) {
    if (diag == 'N' || diag == 'n') {
      return BLAS_diagonal_non_unit;
    }
    else if (diag == 'U' || diag == 'u') {
      return BLAS_diagonal_unit;
    }
    else {
      errormessage("Error when transfering %c to rocblas_diag\n", diag);
      // or abort?
      return BLAS_diagonal_non_unit;
    }
  }


  void rocblasDgemv_elpa_wrapper (BLAS_handle rocblasHandle, char trans, int m, int n, double alpha,
                               const double *A, int lda,  const double *x, int incx,
                               double beta, double *y, int incy) {

    BLAS_status status = BLAS_dgemv(rocblasHandle, hip_operation(trans),
                m, n, &alpha, A, lda, x, incx, &beta, y, incy);
  }

  void rocblasSgemv_elpa_wrapper (BLAS_handle rocblasHandle, char trans, int m, int n, float alpha,
                               const float *A, int lda,  const float *x, int incx,
                               float beta, float *y, int incy) {

    BLAS_status status = BLAS_sgemv(rocblasHandle, hip_operation(trans),
                m, n, &alpha, A, lda, x, incx, &beta, y, incy);
  }

  void rocblasZgemv_elpa_wrapper (BLAS_handle rocblasHandle, char trans, int m, int n, double _Complex alpha,
                               const double _Complex *A, int lda,  const double _Complex *x, int incx,
                               double _Complex beta, double _Complex *y, int incy) {

    BLAS_double_complex alpha_casted = *((BLAS_double_complex*)(&alpha));
    BLAS_double_complex beta_casted = *((BLAS_double_complex*)(&beta));

#ifndef HIPBLAS
    const BLAS_double_complex* A_casted = (const BLAS_double_complex*) A;
    const BLAS_double_complex* x_casted = (const BLAS_double_complex*) x;
#else
    BLAS_double_complex* A_casted = (BLAS_double_complex*) A;
    BLAS_double_complex* x_casted = (BLAS_double_complex*) x;
#endif
    BLAS_double_complex* y_casted = (BLAS_double_complex*) y;

    BLAS_status status = BLAS_zgemv(rocblasHandle, hip_operation(trans),
                m, n, &alpha_casted, A_casted, lda, x_casted, incx, &beta_casted, y_casted, incy);
  }

  void rocblasCgemv_elpa_wrapper (BLAS_handle rocblasHandle, char trans, int m, int n, float _Complex alpha,
                               const float _Complex *A, int lda,  const float _Complex *x, int incx,
                               float _Complex beta, float _Complex *y, int incy) {

    BLAS_float_complex alpha_casted = *((BLAS_float_complex*)(&alpha));
    BLAS_float_complex beta_casted = *((BLAS_float_complex*)(&beta));

#ifndef HIPBLAS
    const BLAS_float_complex* A_casted = (const BLAS_float_complex*) A;
    const BLAS_float_complex* x_casted = (const BLAS_float_complex*) x;
#else
          BLAS_float_complex* A_casted = (      BLAS_float_complex*) A;
          BLAS_float_complex* x_casted = (      BLAS_float_complex*) x;
#endif
    BLAS_float_complex* y_casted = (BLAS_float_complex*) y;

    BLAS_status status = BLAS_cgemv(rocblasHandle, hip_operation(trans),
                m, n, &alpha_casted, A_casted, lda, x_casted, incx, &beta_casted, y_casted, incy);
  }


  void rocblasDgemm_elpa_wrapper (BLAS_handle rocblasHandle, char transa, char transb, int m, int n, int k,
                               double alpha, const double *A, int lda,
                               const double *B, int ldb, double beta,
                               double *C, int ldc) {

    BLAS_status status = BLAS_dgemm(rocblasHandle, hip_operation(transa), hip_operation(transb),
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

  void rocblasSgemm_elpa_wrapper (BLAS_handle rocblasHandle, char transa, char transb, int m, int n, int k,
                               float alpha, const float *A, int lda,
                               const float *B, int ldb, float beta,
                               float *C, int ldc) {

    BLAS_status status = BLAS_sgemm(rocblasHandle, hip_operation(transa), hip_operation(transb),
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

  void rocblasZgemm_elpa_wrapper (BLAS_handle rocblasHandle, char transa, char transb, int m, int n, int k,
                               double _Complex alpha, const double _Complex *A, int lda,
                               const double _Complex *B, int ldb, double _Complex beta,
                               double _Complex *C, int ldc) {

    BLAS_double_complex alpha_casted = *((BLAS_double_complex*)(&alpha));
    BLAS_double_complex beta_casted = *((BLAS_double_complex*)(&beta));

#ifndef HIPBLAS
    const BLAS_double_complex* A_casted = (const BLAS_double_complex*) A;
    const BLAS_double_complex* B_casted = (const BLAS_double_complex*) B;
#else
          BLAS_double_complex* A_casted = (      BLAS_double_complex*) A;
          BLAS_double_complex* B_casted = (      BLAS_double_complex*) B;
#endif
    BLAS_double_complex* C_casted = (BLAS_double_complex*) C;

    BLAS_status status = BLAS_zgemm(rocblasHandle, hip_operation(transa), hip_operation(transb),
                m, n, k, &alpha_casted, A_casted, lda, B_casted, ldb, &beta_casted, C_casted, ldc);
  }

  void rocblasCgemm_elpa_wrapper (BLAS_handle rocblasHandle, char transa, char transb, int m, int n, int k,
                               float _Complex alpha, const float _Complex *A, int lda,
                               const float _Complex *B, int ldb, float _Complex beta,
                               float _Complex *C, int ldc) {

    BLAS_float_complex alpha_casted = *((BLAS_float_complex*)(&alpha));
    BLAS_float_complex beta_casted = *((BLAS_float_complex*)(&beta));

#ifndef HIPBLAS
    const BLAS_float_complex* A_casted = (const BLAS_float_complex*) A;
    const BLAS_float_complex* B_casted = (const BLAS_float_complex*) B;
#else
          BLAS_float_complex* A_casted = (      BLAS_float_complex*) A;
          BLAS_float_complex* B_casted = (      BLAS_float_complex*) B;
#endif
    BLAS_float_complex* C_casted = (BLAS_float_complex*) C;

    BLAS_status status = BLAS_cgemm(rocblasHandle, hip_operation(transa), hip_operation(transb),
                m, n, k, &alpha_casted, A_casted, lda, B_casted, ldb, &beta_casted, C_casted, ldc);
  }

  void rocblasDgemm_elpa_wrapper_intptr_handle (intptr_t* rocblasHandle, char transa, char transb, int m, int n, int k,
                               double alpha, const double *A, int lda,
                               const double *B, int ldb, double beta,
                               double *C, int ldc) {

    BLAS_status status = BLAS_dgemm((BLAS_handle) *rocblasHandle, hip_operation(transa), hip_operation(transb),
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

  void rocblasSgemm_elpa_wrapper_intptr_handle (intptr_t* rocblasHandle, char transa, char transb, int m, int n, int k,
                               float alpha, const float *A, int lda,
                               const float *B, int ldb, float beta,
                               float *C, int ldc) {

    BLAS_status status = BLAS_sgemm((BLAS_handle) *rocblasHandle, hip_operation(transa), hip_operation(transb),
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

  void rocblasZgemm_elpa_wrapper_intptr_handle (intptr_t* rocblasHandle, char transa, char transb, int m, int n, int k,
                               double _Complex alpha, const double _Complex *A, int lda,
                               const double _Complex *B, int ldb, double _Complex beta,
                               double _Complex *C, int ldc) {

    BLAS_double_complex alpha_casted = *((BLAS_double_complex*)(&alpha));
    BLAS_double_complex beta_casted = *((BLAS_double_complex*)(&beta));

#ifndef HIPBLAS
    const BLAS_double_complex* A_casted = (const BLAS_double_complex*) A;
    const BLAS_double_complex* B_casted = (const BLAS_double_complex*) B;
#else
          BLAS_double_complex* A_casted = (      BLAS_double_complex*) A;
          BLAS_double_complex* B_casted = (      BLAS_double_complex*) B;
#endif
    BLAS_double_complex* C_casted = (BLAS_double_complex*) C;

    BLAS_status status = BLAS_zgemm((BLAS_handle) *rocblasHandle, hip_operation(transa), hip_operation(transb),
                m, n, k, &alpha_casted, A_casted, lda, B_casted, ldb, &beta_casted, C_casted, ldc);
  }

  void rocblasCgemm_elpa_wrapper_intptr_handle (intptr_t* rocblasHandle, char transa, char transb, int m, int n, int k,
                               float _Complex alpha, const float _Complex *A, int lda,
                               const float _Complex *B, int ldb, float _Complex beta,
                               float _Complex *C, int ldc) {

    BLAS_float_complex alpha_casted = *((BLAS_float_complex*)(&alpha));
    BLAS_float_complex beta_casted = *((BLAS_float_complex*)(&beta));

#ifndef HIPBLAS
    const BLAS_float_complex* A_casted = (const BLAS_float_complex*) A;
    const BLAS_float_complex* B_casted = (const BLAS_float_complex*) B;
#else
          BLAS_float_complex* A_casted = (      BLAS_float_complex*) A;
          BLAS_float_complex* B_casted = (      BLAS_float_complex*) B;
#endif
    BLAS_float_complex* C_casted = (BLAS_float_complex*) C;

    BLAS_status status = BLAS_cgemm((BLAS_handle) *rocblasHandle, hip_operation(transa), hip_operation(transb),
                m, n, k, &alpha_casted, A_casted, lda, B_casted, ldb, &beta_casted, C_casted, ldc);
  }


  // todo: new CUBLAS API diverged from standard BLAS api for these functions
  // todo: it provides out-of-place (and apparently more efficient) implementation
  // todo: by passing B twice (in place of C as well), we should fall back to in-place algorithm


  void rocblasDcopy_elpa_wrapper (BLAS_handle rocblasHandle, int n, double *x, int incx, double *y, int incy){

    BLAS_status status = BLAS_dcopy(rocblasHandle, n, x, incx, y, incy);
  }

  void rocblasScopy_elpa_wrapper (BLAS_handle rocblasHandle, int n, float *x, int incx, float *y, int incy){

    BLAS_status status = BLAS_scopy(rocblasHandle, n, x, incx, y, incy);
  }

  void rocblasZcopy_elpa_wrapper (BLAS_handle rocblasHandle, int n, double _Complex *x, int incx, double _Complex *y, int incy){
#ifndef HIPBLAS
    const BLAS_double_complex* X_casted = (const BLAS_double_complex*) x;
#else
          BLAS_double_complex* X_casted = (      BLAS_double_complex*) x;
#endif
          BLAS_double_complex* Y_casted = (BLAS_double_complex*) y;

    BLAS_status status = BLAS_zcopy(rocblasHandle, n, X_casted, incx, Y_casted, incy);
  }

  void rocblasCcopy_elpa_wrapper (BLAS_handle rocblasHandle, int n, float _Complex *x, int incx, float _Complex *y, int incy){
#ifndef HIPBLAS
    const BLAS_float_complex* X_casted = (const BLAS_float_complex*) x;
#else
          BLAS_float_complex* X_casted = (      BLAS_float_complex*) x;
#endif
          BLAS_float_complex* Y_casted = (      BLAS_float_complex*) y;

    BLAS_status status = BLAS_ccopy(rocblasHandle, n, X_casted, incx, Y_casted, incy);
  }


  void rocblasDtrmm_elpa_wrapper (BLAS_handle rocblasHandle, char side, char uplo, char transa, char diag,
                               int m, int n, double alpha, const double *A,
                               int lda, double *B, int ldb){
#ifdef HAVE_ROCBLAS_API_V3
    BLAS_status status = BLAS_dtrmm(rocblasHandle, hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha, A, lda, B, ldb, B, ldb);
#else
    BLAS_status status = BLAS_dtrmm(rocblasHandle, hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha, A, lda, B, ldb);
#endif
  }

  void rocblasStrmm_elpa_wrapper (BLAS_handle rocblasHandle, char side, char uplo, char transa, char diag,
                               int m, int n, float alpha, const float *A,
                               int lda, float *B, int ldb){
#ifdef HAVE_ROCBLAS_API_V3
    BLAS_status status = BLAS_strmm(rocblasHandle, hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha, A, lda, B, ldb, B, ldb);
#else
    BLAS_status status = BLAS_strmm(rocblasHandle, hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha, A, lda, B, ldb);
#endif
  }

  void rocblasZtrmm_elpa_wrapper (BLAS_handle rocblasHandle, char side, char uplo, char transa, char diag,
                               int m, int n, double _Complex alpha, const double _Complex *A,
                               int lda, double _Complex *B, int ldb){

    BLAS_double_complex alpha_casted = *((BLAS_double_complex*)(&alpha));

#ifndef HIPBLAS
    const BLAS_double_complex* A_casted = (const BLAS_double_complex*) A;
#else
          BLAS_double_complex* A_casted = (      BLAS_double_complex*) A;
#endif
    BLAS_double_complex* B_casted = (BLAS_double_complex*) B;
#ifdef HAVE_ROCBLAS_API_V3
    BLAS_status status = BLAS_ztrmm(rocblasHandle, hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb, B_casted, ldb);
#else
    BLAS_status status = BLAS_ztrmm(rocblasHandle, hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb);
#endif
  }

  void rocblasCtrmm_elpa_wrapper (BLAS_handle rocblasHandle, char side, char uplo, char transa, char diag,
                               int m, int n, float _Complex alpha, const float _Complex *A,
                               int lda, float _Complex *B, int ldb){

    BLAS_float_complex alpha_casted = *((BLAS_float_complex*)(&alpha));

#ifndef HIPBLAS
    const BLAS_float_complex* A_casted = (const BLAS_float_complex*) A;
#else
          BLAS_float_complex* A_casted = (      BLAS_float_complex*) A;
#endif
    BLAS_float_complex* B_casted = (BLAS_float_complex*) B;
#ifdef HAVE_ROCBLAS_API_V3
    BLAS_status status = BLAS_ctrmm(rocblasHandle, hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb, B_casted, ldb);
#else
    BLAS_status status = BLAS_ctrmm(rocblasHandle, hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb);
#endif
  }


  void rocblasDtrsm_elpa_wrapper (BLAS_handle rocblasHandle, char side, char uplo, char transa, char diag,
                               int m, int n, double alpha, double *A,
                               int lda, double *B, int ldb){

    BLAS_status status = BLAS_dtrsm(rocblasHandle, hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha, A, lda, B, ldb);
  }

  void rocblasStrsm_elpa_wrapper (BLAS_handle rocblasHandle, char side, char uplo, char transa, char diag,
                               int m, int n, float alpha, float *A,
                               int lda, float *B, int ldb){
    BLAS_status status = BLAS_strsm(rocblasHandle, hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha, A, lda, B, ldb);
  }

  void rocblasZtrsm_elpa_wrapper (BLAS_handle rocblasHandle, char side, char uplo, char transa, char diag,
                               int m, int n, double _Complex alpha, const double _Complex *A,
                               int lda, double _Complex *B, int ldb){

    BLAS_double_complex alpha_casted = *((BLAS_double_complex*)(&alpha));

#ifndef HIPBLAS
    const BLAS_double_complex* A_casted = (const BLAS_double_complex*) A;
#else
          BLAS_double_complex* A_casted = (      BLAS_double_complex*) A;
#endif
    BLAS_double_complex* B_casted = (BLAS_double_complex*) B;
    BLAS_status status = BLAS_ztrsm(rocblasHandle, hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb);
  }

  void rocblasCtrsm_elpa_wrapper (BLAS_handle rocblasHandle, char side, char uplo, char transa, char diag,
                               int m, int n, float _Complex alpha, const float _Complex *A,
                               int lda, float _Complex *B, int ldb){

    BLAS_float_complex alpha_casted = *((BLAS_float_complex*)(&alpha));

#ifndef HIPBLAS
    const BLAS_float_complex* A_casted = (const BLAS_float_complex*) A;
#else
          BLAS_float_complex* A_casted = (      BLAS_float_complex*) A;
#endif
    BLAS_float_complex* B_casted = (BLAS_float_complex*) B;
    BLAS_status status = BLAS_ctrsm(rocblasHandle, hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb);
  }

  // result can be on host or device depending on pointer mode
  void rocblasDdot_elpa_wrapper (BLAS_handle rocblasHandle, int length, const double *X, int incx, const double *Y, int incy, double *result) {

    //BLAS_status BLAS_set_pointer_mode(rocblasHandle, rocblas_pointer_mode_device);
    //if (status != BLAS_status_success ) {
    //  printf("rocblasDdot: error when setting pointer mode\n");
    //}

    BLAS_status status = BLAS_ddot(rocblasHandle, length, X, incx, Y, incy, result);
    if (status != BLAS_status_success) {
       printf("error when calling rocblasDdot\n");
    }
  }

  void rocblasSdot_elpa_wrapper (BLAS_handle rocblasHandle, int length, const float *X, int incx, const float *Y, int incy, float *result) {

    //BLAS_status BLAS_set_pointer_mode(rocblasHandle, rocblas_pointer_mode_device);
    //if (status != BLAS_status_success ) {
    //  printf("rocblasSdot: error when setting pointer mode\n");
    //}

    BLAS_status status = BLAS_sdot(rocblasHandle, length, X, incx, Y, incy, result);
    if (status != BLAS_status_success) {
       printf("error when calling rocblasSdot\n");
    }
  }

  void rocblasZdot_elpa_wrapper (char conju, BLAS_handle rocblasHandle, int length, const double _Complex *X, int incx, const double _Complex *Y, int incy, double _Complex *result) {

    //BLAS_status BLAS_set_pointer_mode(rocblasHandle, rocblas_pointer_mode_device);
    //if (status != BLAS_status_success ) {
    //  printf("rocblasZdot: error when setting pointer mode\n");
    //}

#ifndef HIPBLAS
    const BLAS_double_complex* X_casted = (const BLAS_double_complex*) X;
    const BLAS_double_complex* Y_casted = (const BLAS_double_complex*) Y;
#else
          BLAS_double_complex* X_casted = (      BLAS_double_complex*) X;
          BLAS_double_complex* Y_casted = (      BLAS_double_complex*) Y;
#endif
          BLAS_double_complex* result_casted = (      BLAS_double_complex*) result;

    BLAS_status status;
    if (conju == 'C' || conju == 'c') {
      status = BLAS_zdotc(rocblasHandle, length, X_casted, incx, Y_casted, incy, result_casted);
    }
    if (conju == 'U' || conju == 'u') {
      status = BLAS_zdotu(rocblasHandle, length, X_casted, incx, Y_casted, incy, result_casted);
    }

    if (status != BLAS_status_success) {
       printf("error when calling rocblasZdot\n");
    }
  }

  void rocblasCdot_elpa_wrapper (char conju, BLAS_handle rocblasHandle, int length, const float _Complex *X, int incx, const float _Complex *Y, int incy, float _Complex *result) {

    //BLAS_status BLAS_set_pointer_mode(rocblasHandle, rocblas_pointer_mode_device);
    //if (status != BLAS_status_success ) {
    //  printf("rocblasCdot: error when setting pointer mode\n");
    //}

#ifndef HIPBLAS
    const BLAS_float_complex* X_casted = (const BLAS_float_complex*) X;
    const BLAS_float_complex* Y_casted = (const BLAS_float_complex*) Y;
#else
          BLAS_float_complex* X_casted = (      BLAS_float_complex*) X;
          BLAS_float_complex* Y_casted = (      BLAS_float_complex*) Y;
#endif
          BLAS_float_complex* result_casted = (      BLAS_float_complex*) result;

    BLAS_status status;

    if (conju == 'C' || conju == 'c') {
      status = BLAS_cdotc(rocblasHandle, length, X_casted, incx, Y_casted, incy, result_casted);
    }
    if (conju == 'U' || conju == 'u') {
      status = BLAS_cdotu(rocblasHandle, length, X_casted, incx, Y_casted, incy, result_casted);
    }

    if (status != BLAS_status_success) {
       printf("error when calling rocblasCdot\n");
    }
  }

  void rocblasSetPointerModeFromC(BLAS_handle rocblasHandle, BLAS_pointer_mode mode) {
    BLAS_status status =  BLAS_set_pointer_mode(rocblasHandle, mode);
    if (status != BLAS_status_success) {
       printf("error when calling rocblasSetPointerMode\n");
    }
  }

  void rocblasGetPointerModeFromC(BLAS_handle rocblasHandle, BLAS_pointer_mode *mode) {
    BLAS_status status = BLAS_get_pointer_mode(rocblasHandle, mode);
    if (status != BLAS_status_success) {
       printf("error when calling rocblasGetPointerMode\n");
    }
  }

  int rocblasPointerModeDeviceFromC(void) {
      int val = (int)BLAS_pointer_mode_device;
      return val;
  }

  int rocblasPointerModeHostFromC(void) {
      int val = (int)BLAS_pointer_mode_host;
      return val;
  }


  void rocblasDscal_elpa_wrapper (BLAS_handle rocblasHandle, int n, double alpha, double *x, int incx){

    BLAS_status status = BLAS_dscal(rocblasHandle, n, &alpha, x, incx);
    if (status != BLAS_status_success) {
       printf("error when calling rocblasDscal\n");
    }
  }

  void rocblasSscal_elpa_wrapper (BLAS_handle rocblasHandle, int n, float alpha, float *x, int incx){

    BLAS_status status = BLAS_sscal(rocblasHandle, n, &alpha, x, incx);
    if (status != BLAS_status_success) {
       printf("error when calling rocblasSscal\n");
    }
  }

  void rocblasZscal_elpa_wrapper (BLAS_handle rocblasHandle, int n, double _Complex alpha, double _Complex *X, int incx){

//#ifndef HIPBLAS
//    const BLAS_double_complex* X_casted = (const BLAS_double_complex*) X;
//#else
          BLAS_double_complex* X_casted = (      BLAS_double_complex*) X;
//#endif
    BLAS_double_complex alpha_casted = *((BLAS_double_complex*)(&alpha));

    BLAS_status status = BLAS_zscal(rocblasHandle, n, &alpha_casted, X_casted, incx);
    if (status != BLAS_status_success) {
       printf("error when calling rocblasZscal\n");
    }
  }

  void rocblasCscal_elpa_wrapper (BLAS_handle rocblasHandle, int n, float _Complex alpha, float _Complex *X, int incx){

//#ifndef HIPBLAS
//    const BLAS_float_complex* X_casted = (const BLAS_float_complex*) X;
//#else
          BLAS_float_complex* X_casted = (      BLAS_float_complex*) X;
//#endif
    BLAS_float_complex alpha_casted = *((BLAS_float_complex*)(&alpha));

    BLAS_status status = BLAS_cscal(rocblasHandle, n, &alpha_casted, X_casted, incx);
    if (status != BLAS_status_success) {
       printf("error when calling rocblasCscal\n");
    }
  }

  void rocblasDaxpy_elpa_wrapper (BLAS_handle rocblasHandle, int n, double alpha, double *x, int incx, double *y, int incy){

    BLAS_status status = BLAS_daxpy(rocblasHandle, n, &alpha, x, incx, y, incy);
    if (status != BLAS_status_success) {
       printf("error when calling rocblasDaxpy\n");
    }
  }

  void rocblasSaxpy_elpa_wrapper (BLAS_handle rocblasHandle, int n, float alpha, float *x, int incx, float *y, int incy){

    BLAS_status status = BLAS_saxpy(rocblasHandle, n, &alpha, x, incx, y, incy);
    if (status != BLAS_status_success) {
       printf("error when calling rocblasSaxpy\n");
    }
  }

  void rocblasZaxpy_elpa_wrapper (BLAS_handle rocblasHandle, int n, double _Complex alpha, double _Complex *X, int incx, double _Complex *Y, int incy){

    BLAS_double_complex* X_casted    =   (BLAS_double_complex*) X;
    BLAS_double_complex* Y_casted    =   (BLAS_double_complex*) Y;
    BLAS_double_complex alpha_casted = *((BLAS_double_complex*)(&alpha));

    BLAS_status status = BLAS_zaxpy(rocblasHandle, n, &alpha_casted, X_casted, incx, Y_casted, incy);
    if (status != BLAS_status_success) {
       printf("error when calling rocblasZaxpy\n");
    }
  }

  void rocblasCaxpy_elpa_wrapper (BLAS_handle rocblasHandle, int n, float _Complex alpha, float _Complex *X, int incx, float _Complex *Y, int incy){

    BLAS_float_complex* X_casted    =   (BLAS_float_complex*) X;
    BLAS_float_complex* Y_casted    =   (BLAS_float_complex*) Y;
    BLAS_float_complex alpha_casted = *((BLAS_float_complex*)(&alpha));

    BLAS_status status = BLAS_caxpy(rocblasHandle, n, &alpha_casted, X_casted, incx, Y_casted, incy);
    if (status != BLAS_status_success) {
       printf("error when calling rocblasCaxpy\n");
    }
  }

}