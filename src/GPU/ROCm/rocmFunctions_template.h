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

#include <stdio.h>
#include <math.h>
#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <stdint.h>
#include <complex.h>

#include "config-f90.h"

#ifdef WITH_AMD_ROCSOLVER
#include <rocsolver.h>
#endif


#undef BLAS_status
#undef BLAS_handle
//#undef BLAS_float_complex
#undef BLAS_set_stream
#undef BLAS_status_success
#undef BLAS_status_invalid_handle
#undef BLAS_create_handle
#undef BLAS_destroy_handle
#undef BLAS_double_complex
#undef BLAS_float_complex
#undef BLAS_strsm
#undef BLAS_dtrsm
#undef BLAS_ctrsm
#undef BLAS_ztrsm
#undef BLAS_dtrmm
#undef BLAS_strmm
#undef BLAS_ztrmm
#undef BLAS_ctrmm
#undef BLAS_dcopy
#undef BLAS_scopy
#undef BLAS_zcopy
#undef BLAS_ccopy
#undef BLAS_dgemm
#undef BLAS_sgemm
#undef BLAS_zgemm
#undef BLAS_cgemm
#undef BLAS_dgemv
#undef BLAS_sgemv
#undef BLAS_zgemv
#undef BLAS_cgemv
#undef BLAS_operation
#undef BLAS_operation_none
#undef BLAS_operation_transpose
#undef BLAS_operation_conjugate_transpose
#undef BLAS_operation_none
#undef BLAS_fill
#undef BLAS_fill_lower
#undef BLAS_fill_upper
#undef BLAS_side
#undef BLAS_side_left
#undef BLAS_side_right
#undef BLAS_diagonal
#undef BLAS_diagonal_non_unit
#undef BLAS_diagonal_unit
#undef BLAS_ddot
#undef BLAS_sdot
#undef BLAS_zdot
#undef BLAS_zdotc
#undef BLAS_zdotu
#undef BLAS_cdot
#undef BLAS_cdotu
#undef BLAS_cdotc
#undef BLAS_dscal
#undef BLAS_sscal
#undef BLAS_zscal
#undef BLAS_cscal
#undef BLAS_daxpy
#undef BLAS_saxpy
#undef BLAS_zaxpy
#undef BLAS_caxpy
#undef BLAS_set_pointer_mode
#undef BLAS_get_pointer_mode
#undef BLAS_pointer_mode_host
#undef BLAS_pointer_mode_device
#undef BLAS_pointer_mode

#ifdef HIPBLAS
#define BLAS hipblas
#define BLAS_status hipblasStatus_t
#define BLAS_handle hipblasHandle_t
#define BLAS_set_stream hipblasSetStream
#define BLAS_status_success HIPBLAS_STATUS_SUCCESS
#define BLAS_status_invalid_handle HIPBLAS_STATUS_INVALID_VALUE
#define BLAS_create_handle hipblasCreate
#define BLAS_destroy_handle hipblasDestroy
#define BLAS_double_complex hipblasDoubleComplex
#define BLAS_float_complex hipblasComplex
#define BLAS_ctrsm hipblasCtrsm
#define BLAS_ztrsm hipblasZtrsm
#define BLAS_dtrsm hipblasDtrsm
#define BLAS_strsm hipblasStrsm
#define BLAS_ctrmm hipblasCtrmm
#define BLAS_ztrmm hipblasZtrmm
#define BLAS_dtrmm hipblasDtrmm
#define BLAS_strmm hipblasStrmm
#define BLAS_ccopy hipblasCcopy
#define BLAS_zcopy hipblasZcopy
#define BLAS_dcopy hipblasDcopy
#define BLAS_scopy hipblasScopy
#define BLAS_cgemm hipblasCgemm
#define BLAS_zgemm hipblasZgemm
#define BLAS_dgemm hipblasDgemm
#define BLAS_sgemm hipblasSgemm
#define BLAS_cgemv hipblasCgemv
#define BLAS_zgemv hipblasZgemv
#define BLAS_dgemv hipblasDgemv
#define BLAS_sgemv hipblasSgemv
#define BLAS_operation hipblasOperation_t
#define BLAS_operation_none HIPBLAS_OP_N
#define BLAS_operation_transpose HIPBLAS_OP_T
#define BLAS_operation_conjugate_transpose HIPBLAS_OP_C
#define BLAS_operation_none HIPBLAS_OP_N
#define BLAS_fill hipblasFillMode_t
#define BLAS_fill_lower HIPBLAS_FILL_MODE_LOWER
#define BLAS_fill_upper HIPBLAS_FILL_MODE_UPPER
#define BLAS_side hipblasSideMode_t
#define BLAS_side_left HIPBLAS_SIDE_LEFT
#define BLAS_side_right HIPBLAS_SIDE_RIGHT
#define BLAS_diagonal hipblasDiagType_t
#define BLAS_diagonal_non_unit HIPBLAS_DIAG_NON_UNIT
#define BLAS_diagonal_unit HIPBLAS_DIAG_UNIT
#define BLAS_ddot hipblasDdot
#define BLAS_sdot hipblasSdot
#define BLAS_zdot hipblasZdot
#define BLAS_zdotc hipblasZdotc
#define BLAS_zdotu hipblasZdotu
#define BLAS_cdot hipblasCdot
#define BLAS_cdotc hipblasCdotc
#define BLAS_cdotu hipblasCdotu
#define BLAS_dscal hipblasDscal
#define BLAS_sscal hipblasSscal
#define BLAS_zscal hipblasZscal
#define BLAS_cscal hipblasCscal
#define BLAS_daxpy hipblasDaxpy
#define BLAS_saxpy hipblasSaxpy
#define BLAS_zaxpy hipblasZaxpy
#define BLAS_caxpy hipblasCaxpy
#define BLAS_set_pointer_mode hipblasSetPointerMode
#define BLAS_get_pointer_mode hipblasGetPointerMode
#define BLAS_pointer_mode_host HIPBLAS_POINTER_MODE_HOST
#define BLAS_pointer_mode_device HIPBLAS_POINTER_MODE_DEVICE
#define BLAS_pointer_mode hipblasPointerMode_t
//#define BLAS_float_complex hipblas_float_complex
//#define BLAS_set_stream hipblas_set_stream
#else /* HIPBLAS */
#define BLAS rocblas
#define BLAS_status rocblas_status
#define BLAS_handle rocblas_handle
#define BLAS_set_stream rocblas_set_stream
#define BLAS_status_success rocblas_status_success
#define BLAS_status_invalid_handle rocblas_status_invalid_handle
#define BLAS_status_memory_error rocblas_status_memory_error
#define BLAS_create_handle rocblas_create_handle
#define BLAS_destroy_handle rocblas_destroy_handle
#define BLAS_double_complex rocblas_double_complex
#define BLAS_float_complex rocblas_float_complex
#define BLAS_ctrsm rocblas_ctrsm
#define BLAS_ztrsm rocblas_ztrsm
#define BLAS_dtrsm rocblas_dtrsm
#define BLAS_strsm rocblas_strsm
#define BLAS_ctrmm rocblas_ctrmm
#define BLAS_ztrmm rocblas_ztrmm
#define BLAS_dtrmm rocblas_dtrmm
#define BLAS_strmm rocblas_strmm
#define BLAS_ccopy rocblas_ccopy
#define BLAS_zcopy rocblas_zcopy
#define BLAS_dcopy rocblas_dcopy
#define BLAS_scopy rocblas_scopy
#define BLAS_cgemm rocblas_cgemm
#define BLAS_zgemm rocblas_zgemm
#define BLAS_dgemm rocblas_dgemm
#define BLAS_sgemm rocblas_sgemm
#define BLAS_cgemv rocblas_cgemv
#define BLAS_zgemv rocblas_zgemv
#define BLAS_dgemv rocblas_dgemv
#define BLAS_sgemv rocblas_sgemv
#define BLAS_operation rocblas_operation
#define BLAS_operation_none rocblas_operation_none
#define BLAS_operation_transpose rocblas_operation_transpose
#define BLAS_operation_conjugate_transpose rocblas_operation_conjugate_transpose
#define BLAS_operation_none rocblas_operation_none
#define BLAS_fill rocblas_fill
#define BLAS_fill_lower rocblas_fill_lower
#define BLAS_fill_upper rocblas_fill_upper
#define BLAS_side rocblas_side
#define BLAS_side_left rocblas_side_left
#define BLAS_side_right rocblas_side_right
#define BLAS_diagonal rocblas_diagonal
#define BLAS_diagonal_non_unit rocblas_diagonal_non_unit
#define BLAS_diagonal_unit rocblas_diagonal_unit
#define BLAS_ddot rocblas_ddot
#define BLAS_sdot rocblas_sdot
#define BLAS_zdot rocblas_zdot
#define BLAS_zdotu rocblas_zdotu
#define BLAS_zdotc rocblas_zdotc
#define BLAS_cdot rocblas_cdot
#define BLAS_cdotc rocblas_cdotc
#define BLAS_cdotu rocblas_cdotu
#define BLAS_dscal rocblas_dscal
#define BLAS_sscal rocblas_sscal
#define BLAS_zscal rocblas_zscal
#define BLAS_cscal rocblas_cscal
#define BLAS_daxpy rocblas_daxpy
#define BLAS_saxpy rocblas_saxpy
#define BLAS_zaxpy rocblas_zaxpy
#define BLAS_caxpy rocblas_caxpy
#define BLAS_set_pointer_mode rocblas_set_pointer_mode
#define BLAS_get_pointer_mode rocblas_get_pointer_mode
#define BLAS_pointer_mode_host rocblas_pointer_mode_host
#define BLAS_pointer_mode_device rocblas_pointer_mode_device
#define BLAS_pointer_mode rocblas_pointer_mode
#endif /* HIPBLAS */

#ifdef HIPBLAS
#include "hipblas.h"
#else
#include "rocblas.h"
#endif
#include "hip/hip_runtime_api.h"


#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#ifdef DEBUG_HIP
#define debugmessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)
#else
#define debugmessage(x, ...)
#endif

// hipStream_t elpa_hip_stm;

#ifdef WITH_AMD_GPU_VERSION

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
}


extern "C" {
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

#ifdef WITH_AMD_ROCSOLVER
// not needed for ROCM; rocmsolver users rocblas handle 
//  int rocsolverSetStreamFromC(intptr_t rocsolver_handle, intptr_t stream) {
//    rocsolverStatus_t status = rocsolverDnSetStream(*((cusolverDnHandle_t*)cusolver_handle), *((cudaStream_t*)stream));
//    if (status == CUSOLVER_STATUS_SUCCESS) {
//      return 1;
//    }
//    else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
//      errormessage("Error in cusolverDnSetStream: %s\n", "the CUDA Runtime initialization failed");
//      return 0;
//    }
//    else{
//      errormessage("Error in cusolverDnSetStream: %s\n", "unknown error");
//      return 0;
//    }
//  }
#endif

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


#ifdef WITH_AMD_ROCSOLVER
// not needed for rocm
//  int cusolverCreateFromC(intptr_t *cusolver_handle) {
//    *cusolver_handle = (intptr_t) malloc(sizeof(cusolverDnHandle_t));
//    cusolverStatus_t status = cusolverDnCreate((cusolverDnHandle_t*) *cusolver_handle);
//    if (status == CUSOLVER_STATUS_SUCCESS) {
////       printf("all OK\n");
//      return 1;
//    }
//    else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
//      errormessage("Error in cusolverCreate: %s\n", "the CUDA Runtime initialization failed");
//      return 0;
//    }
//    else if (status == CUSOLVER_STATUS_ALLOC_FAILED) {
//      errormessage("Error in cusolverCreate: %s\n", "the resources could not be allocated");
//      return 0;
//    }
//    else{
//      errormessage("Error in cusolverCreate: %s\n", "unknown error");
//      return 0;
//    }
//  }
//  int cusolverDestroyFromC(intptr_t *cusolver_handle) {
//    cusolverStatus_t status = cusolverDnDestroy(*((cusolverDnHandle_t*) *cusolver_handle));
//    *cusolver_handle = (intptr_t) NULL;
//    if (status == CUSOLVER_STATUS_SUCCESS) {
////       printf("all OK\n");
//      return 1;
//    }
//    else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
//      errormessage("Error in cusolverDestroy: %s\n", "the library has not been initialized");
//      return 0;
//    }
//    else{
//      errormessage("Error in cusolverDestroy: %s\n", "unknown error");
//      return 0;
//    }
//  }
#endif /* WITH_AMD_ROCSOLVER */

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

#ifdef WITH_AMD_ROCSOLVER
  void rocsolverDtrtri_elpa_wrapper (BLAS_handle rocblasHandle, char uplo, char diag, int64_t n, double *A, int64_t lda, int *info) {
    BLAS_status status;

    int info_gpu = 0;

    int *devInfo = NULL;
    hipError_t hiperr = hipMalloc((void**)&devInfo, sizeof(int));
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Dtrtri devInfo: %s\n",hipGetErrorString(hiperr));
    }
#ifdef DEBUG_AMD
    printf("HIP Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif

//    double *d_work = NULL, *h_work=NULL;
//    size_t d_lwork = 0;
//    size_t h_lwork = 0;
//    status = cusolverDnXtrtri_bufferSize(*((cusolverDnHandle_t*)handle), hip_fill_mode(uplo), hip_diag_type(diag), n, CUDA_R_64F, A, lda, &d_lwork, &h_lwork);
//    if (status != CUSOLVER_STATUS_SUCCESS) {
//      errormessage("Error in cusolverDnDtrtri_buffer_size %s \n","aborting");
//    }
//
//    if (h_lwork != 0) {
//      errormessage("Error in cusolver_Dtrtri host work array needed of size=: %d\n",h_lwork);
//    }
//
//    //cuerr = cudaMalloc((void**) &d_work, sizeof(double) * d_lwork);
//    cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork already in bytes
//    if (cuerr != cudaSuccess) {
//      errormessage("Error in cusolver_Dtrtri d_work: %s\n",cudaGetErrorString(cuerr));
//    }
//#ifdef DEBUG_CUDA
//    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
//#endif
    status = rocsolver_dtrtri(rocblasHandle, hip_fill_mode(uplo), hip_diag_type(diag), n, A, lda, devInfo);

    if (status != BLAS_status_success ) {
      errormessage("Error in rocsolver_Dtrtri %s\n",hipGetErrorString(hiperr));
    }

    hiperr = hipMemcpy(&info_gpu, devInfo, sizeof(int), hipMemcpyDeviceToHost);
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Dtrtri info_gpu: %s\n",hipGetErrorString(hiperr));
    }

    *info = info_gpu;
    //hiperr = hipFree(d_work);
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Dtrtri cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    //}

    hiperr = hipFree(devInfo);
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Dtrtri hip_free(devInfo): %s\n",hipGetErrorString(hiperr));
    }
  }

  void rocsolverStrtri_elpa_wrapper (BLAS_handle rocblasHandle, char uplo, char diag, int64_t n, float *A, int64_t lda, int *info) {
    BLAS_status status;

    int info_gpu = 0;

    int *devInfo = NULL;
    hipError_t hiperr = hipMalloc((void**)&devInfo, sizeof(int));
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Dtrtri devInfo: %s\n",hipGetErrorString(hiperr));
    }
#ifdef DEBUG_AMD
    printf("HIP Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif

//    double *d_work = NULL, *h_work=NULL;
//    size_t d_lwork = 0;
//    size_t h_lwork = 0;
//    status = cusolverDnXtrtri_bufferSize(*((cusolverDnHandle_t*)handle), hip_fill_mode(uplo), hip_diag_type(diag), n, CUDA_R_64F, A, lda, &d_lwork, &h_lwork);
//    if (status != CUSOLVER_STATUS_SUCCESS) {
//      errormessage("Error in cusolverDnDtrtri_buffer_size %s \n","aborting");
//    }
//
//    if (h_lwork != 0) {
//      errormessage("Error in cusolver_Dtrtri host work array needed of size=: %d\n",h_lwork);
//    }
//
//    //cuerr = cudaMalloc((void**) &d_work, sizeof(double) * d_lwork);
//    cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork already in bytes
//    if (cuerr != cudaSuccess) {
//      errormessage("Error in cusolver_Dtrtri d_work: %s\n",cudaGetErrorString(cuerr));
//    }
//#ifdef DEBUG_CUDA
//    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
//#endif
    status = rocsolver_strtri(rocblasHandle, hip_fill_mode(uplo), hip_diag_type(diag), n, A, lda, devInfo);

    if (status != BLAS_status_success ) {
      errormessage("Error in rocsolver_Strtri %s\n",hipGetErrorString(hiperr));
    }

    hiperr = hipMemcpy(&info_gpu, devInfo, sizeof(int), hipMemcpyDeviceToHost);
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Strtri info_gpu: %s\n",hipGetErrorString(hiperr));
    }

    *info = info_gpu;
    //hiperr = hipFree(d_work);
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Strtri cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    //}

    hiperr = hipFree(devInfo);
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Strtri hip_free(devInfo): %s\n",hipGetErrorString(hiperr));
    }
  }

  void rocsolverZtrtri_elpa_wrapper (BLAS_handle rocblasHandle, char uplo, char diag, int64_t n, double _Complex *A, int64_t lda, int *info) {
    BLAS_status status;

    int info_gpu = 0;

    BLAS_double_complex* A_casted = (      BLAS_double_complex*) A;

    int *devInfo = NULL;
    hipError_t hiperr = hipMalloc((void**)&devInfo, sizeof(int));
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Ztrtri devInfo: %s\n",hipGetErrorString(hiperr));
    }
#ifdef DEBUG_AMD
    printf("HIP Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif

//    double *d_work = NULL, *h_work=NULL;
//    size_t d_lwork = 0;
//    size_t h_lwork = 0;
//    status = cusolverDnXtrtri_bufferSize(*((cusolverDnHandle_t*)handle), hip_fill_mode(uplo), hip_diag_type(diag), n, CUDA_R_64F, A, lda, &d_lwork, &h_lwork);
//    if (status != CUSOLVER_STATUS_SUCCESS) {
//      errormessage("Error in cusolverDnDtrtri_buffer_size %s \n","aborting");
//    }
//
//    if (h_lwork != 0) {
//      errormessage("Error in cusolver_Dtrtri host work array needed of size=: %d\n",h_lwork);
//    }
//
//    //cuerr = cudaMalloc((void**) &d_work, sizeof(double) * d_lwork);
//    cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork already in bytes
//    if (cuerr != cudaSuccess) {
//      errormessage("Error in cusolver_Dtrtri d_work: %s\n",cudaGetErrorString(cuerr));
//    }
//#ifdef DEBUG_CUDA
//    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
//#endif
    status = rocsolver_ztrtri(rocblasHandle, hip_fill_mode(uplo), hip_diag_type(diag), n, A_casted, lda, devInfo);

    if (status != BLAS_status_success ) {
      errormessage("Error in rocsolver_Ztrtri %s\n",hipGetErrorString(hiperr));
    }

    hiperr = hipMemcpy(&info_gpu, devInfo, sizeof(int), hipMemcpyDeviceToHost);
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Ztrtri info_gpu: %s\n",hipGetErrorString(hiperr));
    }

    *info = info_gpu;
    //hiperr = hipFree(d_work);
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Strtri cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    //}

    hiperr = hipFree(devInfo);
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Ztrtri hip_free(devInfo): %s\n",hipGetErrorString(hiperr));
    }
  }

  void rocsolverCtrtri_elpa_wrapper (BLAS_handle rocblasHandle, char uplo, char diag, int64_t n, float _Complex *A, int64_t lda, int *info) {
    BLAS_status status;

    int info_gpu = 0;

    BLAS_float_complex* A_casted = (      BLAS_float_complex*) A;

    int *devInfo = NULL;
    hipError_t hiperr = hipMalloc((void**)&devInfo, sizeof(int));
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Ctrtri devInfo: %s\n",hipGetErrorString(hiperr));
    }
#ifdef DEBUG_AMD
    printf("HIP Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif

//    double *d_work = NULL, *h_work=NULL;
//    size_t d_lwork = 0;
//    size_t h_lwork = 0;
//    status = cusolverDnXtrtri_bufferSize(*((cusolverDnHandle_t*)handle), hip_fill_mode(uplo), hip_diag_type(diag), n, CUDA_R_64F, A, lda, &d_lwork, &h_lwork);
//    if (status != CUSOLVER_STATUS_SUCCESS) {
//      errormessage("Error in cusolverDnDtrtri_buffer_size %s \n","aborting");
//    }
//
//    if (h_lwork != 0) {
//      errormessage("Error in cusolver_Dtrtri host work array needed of size=: %d\n",h_lwork);
//    }
//
//    //cuerr = cudaMalloc((void**) &d_work, sizeof(double) * d_lwork);
//    cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork already in bytes
//    if (cuerr != cudaSuccess) {
//      errormessage("Error in cusolver_Dtrtri d_work: %s\n",cudaGetErrorString(cuerr));
//    }
//#ifdef DEBUG_CUDA
//    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
//#endif
    status = rocsolver_ctrtri(rocblasHandle, hip_fill_mode(uplo), hip_diag_type(diag), n, A_casted, lda, devInfo);

    if (status != BLAS_status_success ) {
      errormessage("Error in rocsolver_Ctrtri %s\n",hipGetErrorString(hiperr));
    }

    hiperr = hipMemcpy(&info_gpu, devInfo, sizeof(int), hipMemcpyDeviceToHost);
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Ctrtri info_gpu: %s\n",hipGetErrorString(hiperr));
    }

    *info = info_gpu;
    //hiperr = hipFree(d_work);
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Ctrtri cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    //}

    hiperr = hipFree(devInfo);
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Ctrtri hip_free(devInfo): %s\n",hipGetErrorString(hiperr));
    }
  }


  void rocsolverDpotrf_elpa_wrapper (BLAS_handle rocblasHandle, char uplo, int n, double *A, int lda, int *info) {
    BLAS_status  status;

    int info_gpu = 0;

    int *devInfo = NULL;
    hipError_t hiperr = hipMalloc((void**)&devInfo, sizeof(int));
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Dpotrf devInfo: %s\n",hipGetErrorString(hiperr));
    }
#ifdef DEBUG_AMD
    printf("HIP Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif

//    double *d_work = NULL;
//    int d_lwork = 0;
//
//
//    status = cusolverDnDpotrf_bufferSize(*((cusolverDnHandle_t*)handle), hip_fill_mode(uplo),  n, A, lda, &d_lwork);
//    if (status != CUSOLVER_STATUS_SUCCESS) {
//      errormessage("Error in cusolverDnDpotrf_buffer_size %s \n","aborting");
//    }
//
//    cuerr = cudaMalloc((void**) &d_work, sizeof(double) * d_lwork);
//    //cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork already in bytes
//    if (cuerr != cudaSuccess) {
//      errormessage("Error in cusolver_Dpotrf d_work: %s\n",cudaGetErrorString(cuerr));
//    }
//#ifdef DEBUG_CUDA
//    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
//#endif

    status = rocsolver_dpotrf(rocblasHandle, hip_fill_mode(uplo), n, A, lda, devInfo);

    if (status != BLAS_status_success ) {
      errormessage("Error in rocsolver_Dpotrf %s\n",hipGetErrorString(hiperr));
    }

    hiperr = hipMemcpy(&info_gpu, devInfo, sizeof(int), hipMemcpyDeviceToHost);
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Dpotrf info_gpu: %s\n",hipGetErrorString(hiperr));
    }

    *info = info_gpu;
    //cuerr = cudaFree(d_work);
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Dpotrf cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    //}

    hiperr = hipFree(devInfo);
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Dpotrf hip_free(devInfo): %s\n",hipGetErrorString(hiperr));
    }
  }

  void rocsolverSpotrf_elpa_wrapper (BLAS_handle rocblasHandle, char uplo, int n, float *A, int lda, int *info) {
    BLAS_status status;

    int info_gpu = 0;

    int *devInfo = NULL;
    hipError_t hiperr = hipMalloc((void**)&devInfo, sizeof(int));
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Spotrf devInfo: %s\n",hipGetErrorString(hiperr));
    }
#ifdef DEBUG_AMD
    printf("HIP Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif

//    double *d_work = NULL;
//    int d_lwork = 0;
//
//
//    status = cusolverDnDpotrf_bufferSize(*((cusolverDnHandle_t*)handle), hip_fill_mode(uplo),  n, A, lda, &d_lwork);
//    if (status != CUSOLVER_STATUS_SUCCESS) {
//      errormessage("Error in cusolverDnDpotrf_buffer_size %s \n","aborting");
//    }
//
//    cuerr = cudaMalloc((void**) &d_work, sizeof(double) * d_lwork);
//    //cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork already in bytes
//    if (cuerr != cudaSuccess) {
//      errormessage("Error in cusolver_Dpotrf d_work: %s\n",cudaGetErrorString(cuerr));
//    }
//#ifdef DEBUG_CUDA
//    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
//#endif

    status = rocsolver_spotrf(rocblasHandle, hip_fill_mode(uplo), n, A, lda, devInfo);

    if (status != BLAS_status_success ) {
      errormessage("Error in rocsolver_Spotrf %s\n",hipGetErrorString(hiperr));
    }

    hiperr = hipMemcpy(&info_gpu, devInfo, sizeof(int), hipMemcpyDeviceToHost);
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Spotrf info_gpu: %s\n",hipGetErrorString(hiperr));
    }

    *info = info_gpu;
    //cuerr = cudaFree(d_work);
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Dpotrf cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    //}

    hiperr = hipFree(devInfo);
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Spotrf hip_free(devInfo): %s\n",hipGetErrorString(hiperr));
    }
  }

  void rocsolverZpotrf_elpa_wrapper (BLAS_handle rocblasHandle, char uplo, int n, double _Complex *A, int lda, int *info) {
    BLAS_status status;

    int info_gpu = 0;

    int *devInfo = NULL;
    hipError_t hiperr = hipMalloc((void**)&devInfo, sizeof(int));
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Zpotrf devInfo: %s\n",hipGetErrorString(hiperr));
    }
#ifdef DEBUG_AMD
    printf("HIP Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif

    BLAS_double_complex* A_casted = (      BLAS_double_complex*) A;

//    double *d_work = NULL;
//    int d_lwork = 0;
//
//
//    status = cusolverDnDpotrf_bufferSize(*((cusolverDnHandle_t*)handle), hip_fill_mode(uplo),  n, A, lda, &d_lwork);
//    if (status != CUSOLVER_STATUS_SUCCESS) {
//      errormessage("Error in cusolverDnDpotrf_buffer_size %s \n","aborting");
//    }
//
//    cuerr = cudaMalloc((void**) &d_work, sizeof(double) * d_lwork);
//    //cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork already in bytes
//    if (cuerr != cudaSuccess) {
//      errormessage("Error in cusolver_Zpotrf d_work: %s\n",cudaGetErrorString(cuerr));
//    }
//#ifdef DEBUG_CUDA
//    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
//#endif

    status = rocsolver_zpotrf(rocblasHandle, hip_fill_mode(uplo), n, A_casted, lda, devInfo);

    if (status != BLAS_status_success ) {
      errormessage("Error in rocsolver_Zpotrf %s\n",hipGetErrorString(hiperr));
    }

    hiperr = hipMemcpy(&info_gpu, devInfo, sizeof(int), hipMemcpyDeviceToHost);
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Zpotrf info_gpu: %s\n",hipGetErrorString(hiperr));
    }

    *info = info_gpu;
    //cuerr = cudaFree(d_work);
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Zpotrf cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    //}

    hiperr = hipFree(devInfo);
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Zpotrf hip_free(devInfo): %s\n",hipGetErrorString(hiperr));
    }
  }


  void rocsolverCpotrf_elpa_wrapper (BLAS_handle rocblasHandle, char uplo, int n, float _Complex *A, int lda, int *info) {
    BLAS_status status;

    int info_gpu = 0;

    int *devInfo = NULL;
    hipError_t hiperr = hipMalloc((void**)&devInfo, sizeof(int));
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Cpotrf devInfo: %s\n",hipGetErrorString(hiperr));
    }
#ifdef DEBUG_AMD
    printf("HIP Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif

    BLAS_float_complex* A_casted = (      BLAS_float_complex*) A;

//    double *d_work = NULL;
//    int d_lwork = 0;
//
//
//    status = cusolverDnDpotrf_bufferSize(*((cusolverDnHandle_t*)handle), hip_fill_mode(uplo),  n, A, lda, &d_lwork);
//    if (status != CUSOLVER_STATUS_SUCCESS) {
//      errormessage("Error in cusolverDnDpotrf_buffer_size %s \n","aborting");
//    }
//
//    cuerr = cudaMalloc((void**) &d_work, sizeof(double) * d_lwork);
//    //cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork already in bytes
//    if (cuerr != cudaSuccess) {
//      errormessage("Error in cusolver_Zpotrf d_work: %s\n",cudaGetErrorString(cuerr));
//    }
//#ifdef DEBUG_CUDA
//    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
//#endif

    status = rocsolver_cpotrf(rocblasHandle, hip_fill_mode(uplo), n, A_casted, lda, devInfo);

    if (status != BLAS_status_success ) {
      errormessage("Error in rocsolver_Cpotrf %s\n",hipGetErrorString(hiperr));
    }

    hiperr = hipMemcpy(&info_gpu, devInfo, sizeof(int), hipMemcpyDeviceToHost);
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Cpotrf info_gpu: %s\n",hipGetErrorString(hiperr));
    }

    *info = info_gpu;
    //cuerr = cudaFree(d_work);
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Cpotrf cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    //}

    hiperr = hipFree(devInfo);
    if (hiperr != hipSuccess) {
      errormessage("Error in rocsolver_Cpotrf hip_free(devInfo): %s\n",hipGetErrorString(hiperr));
    }
  }

#endif /* WITH_AMD_ROCSOLVER */


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

    BLAS_status status = BLAS_dtrmm(rocblasHandle, hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha, A, lda, B, ldb);
  }

  void rocblasStrmm_elpa_wrapper (BLAS_handle rocblasHandle, char side, char uplo, char transa, char diag,
                               int m, int n, float alpha, const float *A,
                               int lda, float *B, int ldb){
    BLAS_status status = BLAS_strmm(rocblasHandle, hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha, A, lda, B, ldb);
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
    BLAS_status status = BLAS_ztrmm(rocblasHandle, hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb);
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
    BLAS_status status = BLAS_ctrmm(rocblasHandle, hip_side_mode(side), hip_fill_mode(uplo), hip_operation(transa),
                hip_diag_type(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb);
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

#ifndef HIPBLAS
    const BLAS_double_complex* X_casted = (const BLAS_double_complex*) X;
#else
          BLAS_double_complex* X_casted = (      BLAS_double_complex*) X;
#endif
    BLAS_double_complex alpha_casted = *((BLAS_double_complex*)(&alpha));

    BLAS_status status = BLAS_zscal(rocblasHandle, n, &alpha_casted, X_casted, incx);
    if (status != BLAS_status_success) {
       printf("error when calling rocblasZscal\n");
    }
  }

  void rocblasCscal_elpa_wrapper (BLAS_handle rocblasHandle, int n, float _Complex alpha, float _Complex *X, int incx){

#ifndef HIPBLAS
    const BLAS_float_complex* X_casted = (const BLAS_float_complex*) X;
#else
          BLAS_float_complex* X_casted = (      BLAS_float_complex*) X;
#endif
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
#endif /* WITH_AMD_GPU_VERSION */
