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
// This file was written by P. Karpov, MPCDF


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

//_________________________________________________________________________________________________
// rocsolver?trtri

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

//_________________________________________________________________________________________________
// rocsolver?potrf

  void rocsolverDpotrf_elpa_wrapper (BLAS_handle rocblasHandle, char uplo, int n, double *A, int lda, int *info_dev) {
    BLAS_status  status;

//     int info_gpu = 0;

//     int *devInfo = NULL;
//     hipError_t hiperr = hipMalloc((void**)&devInfo, sizeof(int));
//     if (hiperr != hipSuccess) {
//       errormessage("Error in rocsolver_Dpotrf devInfo: %s\n",hipGetErrorString(hiperr));
//     }
// #ifdef DEBUG_AMD
//     printf("HIP Malloc,  pointer address: %p, size: %d \n", &devInfo);
// #endif

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

    status = rocsolver_dpotrf(rocblasHandle, hip_fill_mode(uplo), n, A, lda, info_dev);

    if (status != BLAS_status_success ) {
      errormessage("Error in rocsolver_Dpotrf %s\n",hipGetErrorString(hiperr));
    }

    // hiperr = hipMemcpy(&info_gpu, devInfo, sizeof(int), hipMemcpyDeviceToHost);
    // if (hiperr != hipSuccess) {
    //   errormessage("Error in rocsolver_Dpotrf info_gpu: %s\n",hipGetErrorString(hiperr));
    // }

    // *info = info_gpu;
    //cuerr = cudaFree(d_work);
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Dpotrf cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    //}

    // hiperr = hipFree(devInfo);
    // if (hiperr != hipSuccess) {
    //   errormessage("Error in rocsolver_Dpotrf hip_free(devInfo): %s\n",hipGetErrorString(hiperr));
    // }
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
