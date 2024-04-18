#if 0
//
//    Copyright 2014 - 2023, A. Marek
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
#endif


extern "C" {

  cudaDataType getCudaDataType(char dataType) {
    if      (dataType=='D') return CUDA_R_64F;
    else if (dataType=='S') return CUDA_R_32F;
    else if (dataType=='Z') return CUDA_C_64F;
    else if (dataType=='C') return CUDA_C_32F;
    else {
      errormessage("Error in getCudaDataType: unknown data type, %s\n", "aborting");
      return CUDA_R_64F;
    }
  }

  void cusolverPrintError(cusolverStatus_t status){
    switch (status){
      case CUSOLVER_STATUS_SUCCESS:
          printf("cusolverStatus=CUSOLVER_STATUS_SUCCESS\n");
          break;
      case CUSOLVER_STATUS_NOT_INITIALIZED:
          printf("cusolverStatus=CUSOLVER_STATUS_NOT_INITIALIZED\n");
          break;
      case CUSOLVER_STATUS_ALLOC_FAILED:
          printf("cusolverStatus=CUSOLVER_STATUS_ALLOC_FAILED\n");
          break;
      case CUSOLVER_STATUS_INVALID_VALUE:
          printf("cusolverStatus=CUSOLVER_STATUS_INVALID_VALUE\n");
          break;
      case CUSOLVER_STATUS_ARCH_MISMATCH:
          printf("cusolverStatus=CUSOLVER_STATUS_ARCH_MISMATCH\n");
          break;
      case CUSOLVER_STATUS_MAPPING_ERROR:
          printf("cusolverStatus=CUSOLVER_STATUS_MAPPING_ERROR\n");
          break;
      case CUSOLVER_STATUS_EXECUTION_FAILED:
          printf("cusolverStatus=CUSOLVER_STATUS_EXECUTION_FAILED\n");
          break;
      case CUSOLVER_STATUS_INTERNAL_ERROR:
          printf("cusolverStatus=CUSOLVER_STATUS_INTERNAL_ERROR\n");
          break;
      case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
          printf("cusolverStatus=CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED\n");
          break;
      case CUSOLVER_STATUS_NOT_SUPPORTED:
          printf("cusolverStatus=CUSOLVER_STATUS_NOT_SUPPORTED\n");
          break;
      default:
          printf("Unknown cusolverStatus status: %d\n", status);
    }
  }


  int cusolverSetStreamFromC(cusolverDnHandle_t cusolver_handle, cudaStream_t cudaStream) {
    //cusolverStatus_t status = cusolverDnSetStream(*((cusolverDnHandle_t*)cusolver_handle), *((cudaStream_t*)stream));
    cusolverStatus_t status = cusolverDnSetStream(cusolver_handle, cudaStream);
    if (status == CUSOLVER_STATUS_SUCCESS) {
      return 1;
    }
    else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      errormessage("Error in cusolverDnSetStream: %s\n", "the CUDA Runtime initialization failed");
      return 0;
    }
    else{
      errormessage("Error in cusolverDnSetStream: %s\n", "unknown error");
      return 0;
    }
  }


  int cusolverCreateFromC(cusolverDnHandle_t *cusolver_handle) {
    //*cusolver_handle = (intptr_t) malloc(sizeof(cusolverDnHandle_t));
    //cusolverStatus_t status = cusolverDnCreate((cusolverDnHandle_t*) *cusolver_handle);
    if (sizeof(intptr_t) != sizeof(cusolverDnHandle_t)) {
      printf("cusolver sizes wrong\n");
    }
    cusolverStatus_t status = cusolverDnCreate(cusolver_handle);
    if (status == CUSOLVER_STATUS_SUCCESS) {
//       printf("all OK\n");
      return 1;
    }
    else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      errormessage("Error in cusolverCreate: %s\n", "the CUDA Runtime initialization failed");
      return 0;
    }
    else if (status == CUSOLVER_STATUS_ALLOC_FAILED) {
      errormessage("Error in cusolverCreate: %s\n", "the resources could not be allocated");
      return 0;
    }
    else{
      errormessage("Error in cusolverCreate: %s\n", "unknown error");
      return 0;
    }
  }


  int cusolverDestroyFromC(cusolverDnHandle_t cusolver_handle) {
    //cusolverStatus_t status = cusolverDnDestroy(*((cusolverDnHandle_t*) *cusolver_handle));
    cusolverStatus_t status = cusolverDnDestroy(cusolver_handle);
    if (status == CUSOLVER_STATUS_SUCCESS) {
//       printf("all OK\n");
      //free((void*) *cusolver_handle);
      return 1;
    }
    else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      errormessage("Error in cusolverDestroy: %s\n", "the library has not been initialized");
      return 0;
    }
    else{
      errormessage("Error in cusolverDestroy: %s\n", "unknown error");
      return 0;
    }
  }

//_________________________________________________________________________________________________
// cusolver?trtri

  void cusolverDtrtri_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, char diag, int64_t n, double *A, int64_t lda, int *info) {
    cusolverStatus_t status;

    int info_gpu = 0;

    int *devInfo = NULL; 
    cudaError_t cuerr = cudaMalloc((void**)&devInfo, sizeof(int));
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dtrtri devInfo: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif

    double *d_work = NULL, *h_work=NULL;
    size_t d_lwork = 0;
    size_t h_lwork = 0;
    //status = cusolverDnXtrtri_bufferSize(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_R_64F, A, lda, &d_lwork, &h_lwork);
    status = cusolverDnXtrtri_bufferSize(cudaHandle, fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_R_64F, A, lda, &d_lwork, &h_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnDtrtri_buffer_size %s \n","aborting");
    }

    if (h_lwork != 0) {
      errormessage("Error in cusolver_Dtrtri host work array needed of size=: %d\n",h_lwork);
    }

#if CUSOLVER_VERSION < 11601
    // temporary workaround for cusolverDnXtrtri_bufferSize bug
    // https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html#cusolver-release-12-4
    d_lwork *= 8;

    // the problem is fixed in CUDA 12.4.1 (cuSOLVER 11.6.1.9)
#endif

    //cuerr = cudaMalloc((void**) &d_work, sizeof(double) * d_lwork);
    cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork already in bytes
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dtrtri d_work: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif

    //status = cusolverDnXtrtri(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_R_64F, A, lda, d_work, d_lwork, h_work, h_lwork, devInfo);
    status = cusolverDnXtrtri(cudaHandle, fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_R_64F, A, lda, d_work, d_lwork, h_work, h_lwork, devInfo);

    if (status != CUSOLVER_STATUS_SUCCESS)
      cusolverPrintError(status);

    //cuerr = cudaDeviceSynchronize();
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Dtrtri: cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
    //}

    cuerr = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dtrtri info_gpu: %s\n",cudaGetErrorString(cuerr));
    }

    *info = info_gpu;
    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dtrtri cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }

    cuerr = cudaFree(devInfo);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dtrtri cuda_free(devInfo): %s\n",cudaGetErrorString(cuerr));
    }
  }


  void cusolverStrtri_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, char diag, int64_t n, float *A, int64_t lda, int *info) {
    cusolverStatus_t status;

    int info_gpu = 0;

    int *devInfo = NULL; 
    cudaError_t cuerr = cudaMalloc((void**)&devInfo, sizeof(int));
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Strtri devInfo: %s\n",cudaGetErrorString(cuerr));
    }

    float *d_work = NULL, *h_work=NULL;
    size_t d_lwork = 0;
    size_t h_lwork = 0;

    //status = cusolverDnXtrtri_bufferSize(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_R_32F, A, lda, &d_lwork, &h_lwork);
    status = cusolverDnXtrtri_bufferSize(cudaHandle, fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_R_32F, A, lda, &d_lwork, &h_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnStrtri_buffer_size %s \n","aborting");
    }

    if (h_lwork != 0) {
      errormessage("Error in cusolver_Strtri host work array needed of size=: %d\n",h_lwork);
    }

#if CUSOLVER_VERSION < 11601
    d_lwork *= 4;
#endif

    //cuerr = cudaMalloc((void**) &d_work, sizeof(float) * d_lwork);
    cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork already in bytes
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Strtri d_work: %s\n",cudaGetErrorString(cuerr));
    }

    //status = cusolverDnXtrtri(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_R_32F, A, lda, d_work, d_lwork, h_work, h_lwork, devInfo);
    status = cusolverDnXtrtri(cudaHandle, fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_R_32F, A, lda, d_work, d_lwork, h_work, h_lwork, devInfo);

    if (status != CUSOLVER_STATUS_SUCCESS)
      cusolverPrintError(status);

    //cuerr = cudaDeviceSynchronize();
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Strtri: cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
    //}

    cuerr = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Strtri info_gpu: %s\n",cudaGetErrorString(cuerr));
    }

    *info = info_gpu;
    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Strtri cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }

    cuerr = cudaFree(devInfo);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Strtri cuda_free(devInfo): %s\n",cudaGetErrorString(cuerr));
    }
  }


  void cusolverZtrtri_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, char diag, int64_t n, double _Complex *A, int64_t lda, int *info) {
    cusolverStatus_t status;

    int info_gpu = 0;

    int *devInfo = NULL; 
    cudaError_t cuerr = cudaMalloc((void**)&devInfo, sizeof(int));
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ztrtri devInfo: %s\n",cudaGetErrorString(cuerr));
    }

    //cuDoubleComplex A_casted = *((cuDoubleComplex*)(A));
    double _Complex *d_work = NULL, *h_work=NULL;
    size_t d_lwork = 0;
    size_t h_lwork = 0;

    //status = cusolverDnXtrtri_bufferSize(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_C_64F, A, lda, &d_lwork, &h_lwork);
    status = cusolverDnXtrtri_bufferSize(cudaHandle, fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_C_64F, A, lda, &d_lwork, &h_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnZtrtri_buffer_size %s \n","aborting");
    }

    if (h_lwork != 0) {
      errormessage("Error in cusolver_Ztrtri host work array needed of size=: %d\n",h_lwork);
    }

#if CUSOLVER_VERSION < 11601
    d_lwork *= 16;
#endif

    //cuerr = cudaMalloc((void**) &d_work, sizeof(double _Complex) * d_lwork);
    cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork in bytes
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ztrtri d_work: %s\n",cudaGetErrorString(cuerr));
    }

    //status = cusolverDnXtrtri(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_C_64F, A, lda, d_work, d_lwork, h_work, h_lwork, devInfo);
    status = cusolverDnXtrtri(cudaHandle, fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_C_64F, A, lda, d_work, d_lwork, h_work, h_lwork, devInfo);

    if (status != CUSOLVER_STATUS_SUCCESS)
      cusolverPrintError(status);

    //cuerr = cudaDeviceSynchronize();
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Ztrtri: cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
    //}

    cuerr = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ztrtri info_gpu: %s\n",cudaGetErrorString(cuerr));
    }

    *info = info_gpu;
    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ztrtri cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }

    cuerr = cudaFree(devInfo);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ztrtri cuda_free(devInfo): %s\n",cudaGetErrorString(cuerr));
    }
  }


  void cusolverCtrtri_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, char diag, int64_t n, float _Complex *A, int64_t lda, int *info) {
    cusolverStatus_t status;

    int info_gpu = 0;

    int *devInfo = NULL; 
    cudaError_t cuerr = cudaMalloc((void**)&devInfo, sizeof(int));
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ctrtri devInfo: %s\n",cudaGetErrorString(cuerr));
    }

    //cuFloatComplex A_casted = *((cuFloatComplex*)(A));
    float _Complex *d_work = NULL, *h_work=NULL;
    size_t d_lwork = 0;
    size_t h_lwork = 0;

    //status = cusolverDnXtrtri_bufferSize(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_C_32F, A, lda, &d_lwork, &h_lwork);
    status = cusolverDnXtrtri_bufferSize(cudaHandle, fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_C_32F, A, lda, &d_lwork, &h_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnCtrtri_buffer_size %s \n","aborting");
    }

    if (h_lwork != 0) {
      errormessage("Error in cusolver_Ctrtri host work array needed of size=: %d\n",h_lwork);
    }

#if CUSOLVER_VERSION < 11601
    d_lwork *= 8;
#endif

    //cuerr = cudaMalloc((void**) &d_work, sizeof(float _Complex) * d_lwork);
    cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork already in bytes
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ctrtri d_work: %s\n",cudaGetErrorString(cuerr));
    }

    //status = cusolverDnXtrtri(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_C_32F, A, lda, d_work, d_lwork, h_work, h_lwork, devInfo);
    status = cusolverDnXtrtri(cudaHandle, fill_mode_new_api(uplo), diag_type_new_api(diag), n, CUDA_C_32F, A, lda, d_work, d_lwork, h_work, h_lwork, devInfo);

    if (status != CUSOLVER_STATUS_SUCCESS)
      cusolverPrintError(status);

    //cuerr = cudaDeviceSynchronize();
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Ctrtri: cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
    //}

    cuerr = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ctrtri info_gpu: %s\n",cudaGetErrorString(cuerr));
    }

    *info = info_gpu;
    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ctrtri cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }

    cuerr = cudaFree(devInfo);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Ctrtri cuda_free(devInfo): %s\n",cudaGetErrorString(cuerr));
    }
  }

//_________________________________________________________________________________________________
// cusolver?potrf - deprecated

  void cusolverDpotrf_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, int n, double *A, int lda, int *info_dev) {
    cusolverStatus_t status;
    cudaError_t cuerr;

    double *d_work = NULL;
    int d_lwork = 0;

    //status = cusolverDnDpotrf_bufferSize(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo),  n, A, lda, &d_lwork);
    status = cusolverDnDpotrf_bufferSize(cudaHandle, fill_mode_new_api(uplo),  n, A, lda, &d_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnDpotrf_buffer_size %s \n","aborting");
    }

    cuerr = cudaMalloc((void**) &d_work, sizeof(double) * d_lwork);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dpotrf d_work: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif

    status = cusolverDnDpotrf(cudaHandle, fill_mode_new_api(uplo), n, A, lda, d_work, d_lwork, info_dev);

    if (status != CUSOLVER_STATUS_SUCCESS)
      cusolverPrintError(status);


    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dpotrf cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }
  }

  void cusolverSpotrf_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, int n, float *A, int lda, int *info_dev) {
    cusolverStatus_t status;
    cudaError_t cuerr;

    float *d_work = NULL;
    int d_lwork = 0;

    status = cusolverDnSpotrf_bufferSize(cudaHandle, fill_mode_new_api(uplo),  n, A, lda, &d_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnSpotrf_buffer_size %s \n","aborting");
    }

    cuerr = cudaMalloc((void**) &d_work, sizeof(float) * d_lwork);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Spotrf d_work: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif

    status = cusolverDnSpotrf(cudaHandle, fill_mode_new_api(uplo), n, A, lda, d_work, d_lwork, info_dev);

    if (status != CUSOLVER_STATUS_SUCCESS)
      cusolverPrintError(status);

    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Spotrf cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }
  }

  void cusolverZpotrf_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, int n, double _Complex *A, int lda, int *info_dev) {
    cusolverStatus_t status;
    cudaError_t cuerr;

    cuDoubleComplex *d_work = NULL;
    int d_lwork = 0;
    cuDoubleComplex* A_casted = (cuDoubleComplex*) A;

    status = cusolverDnZpotrf_bufferSize(cudaHandle, fill_mode_new_api(uplo),  n, A_casted, lda, &d_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnZpotrf_buffer_size %s \n","aborting");
    }

    cuerr = cudaMalloc((void**) &d_work, sizeof(cuDoubleComplex) * d_lwork);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Zpotrf d_work: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif

    status = cusolverDnZpotrf(cudaHandle, fill_mode_new_api(uplo), n, A_casted, lda, d_work, d_lwork, info_dev);

    if (status != CUSOLVER_STATUS_SUCCESS)
      cusolverPrintError(status);

    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Zpotrf cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }
  }

  void cusolverCpotrf_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, int n, float _Complex *A, int lda, int *info_dev) {
    cusolverStatus_t status;
    cudaError_t cuerr;

    cuFloatComplex *d_work = NULL;
    int d_lwork = 0;
    cuFloatComplex* A_casted = (cuFloatComplex*) A;

    status = cusolverDnCpotrf_bufferSize(cudaHandle, fill_mode_new_api(uplo),  n, A_casted, lda, &d_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnCpotrf_buffer_size %s \n","aborting");
    }

    cuerr = cudaMalloc((void**) &d_work, sizeof(cuFloatComplex) * d_lwork);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Cpotrf d_work: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif

    status = cusolverDnCpotrf(cudaHandle, fill_mode_new_api(uplo), n, A_casted, lda, d_work, d_lwork, info_dev);

    if (status != CUSOLVER_STATUS_SUCCESS)
      cusolverPrintError(status);

    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Cpotrf cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }
  }


//_________________________________________________________________________________________________
// cusolverXpotrf

// Introduced with CUDA 11.1 (CUDA_VERSION >= 11010)

void cusolverXpotrf_bufferSize_elpa_wrapper(cusolverDnHandle_t cusolverHandle, char uplo, int n, char dataType, intptr_t A, int lda, 
                                            size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost){

    cusolverStatus_t status;
    cudaDataType cuda_data_type =  getCudaDataType(dataType);

    status = cusolverDnXpotrf_bufferSize(cusolverHandle, NULL, fill_mode_new_api(uplo), (int64_t) n, cuda_data_type, (void *) A, (int64_t) lda, 
                                         cuda_data_type, workspaceInBytesOnDevice, workspaceInBytesOnHost);

    if (status != CUSOLVER_STATUS_SUCCESS){
      cusolverPrintError(status);
      errormessage("Error in cusolverDnXpotrf_bufferSize %s \n", "aborting");
    }
  }


void cusolverXpotrf_elpa_wrapper(cusolverDnHandle_t cusolverHandle, char uplo, int n, char dataType, intptr_t A, int lda, 
                                 intptr_t buffer_dev , size_t *workspaceInBytesOnDevice, 
                                 intptr_t buffer_host, size_t *workspaceInBytesOnHost, int *info_dev){

    cusolverStatus_t status;
    cudaDataType cuda_data_type =  getCudaDataType(dataType);

    status = cusolverDnXpotrf(cusolverHandle, NULL, fill_mode_new_api(uplo), (int64_t) n, cuda_data_type, (void *) A, (int64_t) lda, cuda_data_type,
                              (void *) buffer_dev , *workspaceInBytesOnDevice,
                              (void *) buffer_host, *workspaceInBytesOnHost, info_dev);
    
    if (status != CUSOLVER_STATUS_SUCCESS){
      cusolverPrintError(status);
      errormessage("Error in cusolverDnXpotrf %s \n", "aborting");
    }
  }

} // extern "C"