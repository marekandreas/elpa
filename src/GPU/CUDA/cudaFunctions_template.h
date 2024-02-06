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
// This file was written by A. Marek, MPCDF
#endif


#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#ifdef DEBUG_CUDA
#define debugmessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)
#else
#define debugmessage(x, ...)
#endif


#ifdef WITH_NVIDIA_GPU_VERSION
extern "C" {
  int cudaStreamCreateFromC(cudaStream_t *cudaStream) {
    //*stream = (intptr_t) malloc(sizeof(cudaStream_t));

    if (sizeof(intptr_t) != sizeof(cudaStream_t)) {
      printf("Stream sizes do not match \n");
    }

    cudaError_t status = cudaStreamCreate(cudaStream);

    if (status == cudaSuccess) {
//       printf("all OK\n");
      return 1;
    }
    else{
      errormessage("Error in cudaStreamCreate: %s\n", "unknown error");
      return 0;
    }

  }

  int cudaStreamDestroyFromC(cudaStream_t cudaStream){
    cudaError_t status = cudaStreamDestroy(cudaStream);
    if (status == cudaSuccess) {
//       printf("all OK\n");
	 //free((void*) *stream);
      return 1;
    }
    else{
      errormessage("Error in cudaStreamDestroy: %s\n", "unknown error");
      return 0;
    }
  }

  int cudaStreamSynchronizeExplicitFromC(cudaStream_t cudaStream) {
    cudaError_t status = cudaStreamSynchronize(cudaStream);
    if (status == cudaSuccess) {
      return 1;
    }
    else{
      errormessage("Error in cudaStreamSynchronizeExplicit: %s\n", "unknown error");
      return 0;
    }
  }

  int cudaStreamSynchronizeImplicitFromC() {
    cudaError_t status = cudaStreamSynchronize(cudaStreamPerThread);
    if (status == cudaSuccess) {
      return 1;
    }
    else{
      errormessage("Error in cudaStreamSynchronizeImplicit: %s\n", "unknown error");
      return 0;
    }
  }

  int cublasSetStreamFromC(cublasHandle_t cudaHandle, cudaStream_t cudaStream) {
    //cublasStatus_t status = cublasSetStream(*((cublasHandle_t*)handle), *((cudaStream_t*)stream));
    cublasStatus_t status = cublasSetStream(cudaHandle, cudaStream);
    if (status == CUBLAS_STATUS_SUCCESS) {
      return 1;
    }
    else if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
      errormessage("Error in cublasSetStream: %s\n", "the CUDA Runtime initialization failed");
      return 0;
    }
    else{
      errormessage("Error in cublasSetStream: %s\n", "unknown error");
      return 0;
    }
  }


#ifdef WITH_NVIDIA_CUSOLVER
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
#endif

  int cudaMemcpy2dAsyncFromC(intptr_t *dest, size_t dpitch, intptr_t *src, size_t spitch, size_t width, size_t height, int dir, cudaStream_t cudaStream) {
  
    cudaError_t cuerr = cudaMemcpy2DAsync( dest, dpitch, src, spitch, width, height, (cudaMemcpyKind)dir, cudaStream );
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaMemcpy2dAsync: %s\n",cudaGetErrorString(cuerr));
      return 0;
    }
    return 1;
  }

  int cublasCreateFromC(cublasHandle_t *cublas_handle) {
    //*cublas_handle = (intptr_t) malloc(sizeof(cublasHandle_t));
    if (sizeof(intptr_t) != sizeof(cublasHandle_t)) {
      //errormessage("Error in cublasCreate: sizes not the same");
	printf("ERROR on sizes\n");
      return 0;
    }
    cublasStatus_t status = cublasCreate(cublas_handle);
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

  int cublasDestroyFromC(cublasHandle_t cublas_handle) {
    cublasStatus_t status = cublasDestroy(cublas_handle);
    if (status == CUBLAS_STATUS_SUCCESS) {
//	 free((void*) *cublas_handle);
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

#ifdef WITH_NVIDIA_CUSOLVER

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
#endif /* WITH_NVIDIA_CUSOLVER */

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

  int cudaMemsetAsyncFromC(intptr_t *a, int value, size_t count, cudaStream_t cudaStream) {

    cudaError_t cuerr = cudaMemsetAsync( a, value, count, cudaStream);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaMemsetAsync: %s\n",cudaGetErrorString(cuerr));
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

  int cudaMemcpyAsyncFromC(intptr_t *dest, intptr_t *src, size_t count, int dir, cudaStream_t cudaStream) {

    cudaError_t cuerr = cudaMemcpyAsync( dest, src, count, (cudaMemcpyKind)dir, cudaStream);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cudaMemcpyAsync: %s\n",cudaGetErrorString(cuerr));
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

  int cudaHostRegisterFromC(intptr_t *a, intptr_t value, int flag) {

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

#ifdef WITH_NVIDIA_CUSOLVER
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

    if (status == CUSOLVER_STATUS_SUCCESS) {
    } else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      printf("status = CUSOLVER_STATUS_NOT_INITIALIZED\n");
    } else if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
      printf("status = CUSOLVER_STATUS_NOT_SUPPORTED\n");
    } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
      printf("status = CUSOLVER_STATUS_INVALID_VALUE\n");
    } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
      printf("status = CUSOLVER_STATUS_INTERNAL_ERROR\n"); 
    } else {
      printf("status = UNKNOWN\n");
    }

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

    if (status == CUSOLVER_STATUS_SUCCESS) {
    } else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      printf("status = CUSOLVER_STATUS_NOT_INITIALIZED\n");
    } else if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
      printf("status = CUSOLVER_STATUS_NOT_SUPPORTED\n");
    } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
      printf("status = CUSOLVER_STATUS_INVALID_VALUE\n");
    } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
      printf("status = CUSOLVER_STATUS_INTERNAL_ERROR\n"); 
    } else {
      printf("status = UNKNOWN\n");
    }

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

    if (status == CUSOLVER_STATUS_SUCCESS) {
    } else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      printf("status = CUSOLVER_STATUS_NOT_INITIALIZED\n");
    } else if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
      printf("status = CUSOLVER_STATUS_NOT_SUPPORTED\n");
    } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
      printf("status = CUSOLVER_STATUS_INVALID_VALUE\n");
    } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
      printf("status = CUSOLVER_STATUS_INTERNAL_ERROR\n"); 
    } else {
      printf("status = UNKNOWN\n");
    }

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

    if (status == CUSOLVER_STATUS_SUCCESS) {
    } else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      printf("status = CUSOLVER_STATUS_NOT_INITIALIZED\n");
    } else if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
      printf("status = CUSOLVER_STATUS_NOT_SUPPORTED\n");
    } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
      printf("status = CUSOLVER_STATUS_INVALID_VALUE\n");
    } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
      printf("status = CUSOLVER_STATUS_INTERNAL_ERROR\n"); 
    } else {
      printf("status = UNKNOWN\n");
    }

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


  void cusolverDpotrf_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, int n, double *A, int lda, int *info_dev) {
    cusolverStatus_t status;
    cudaError_t cuerr;

//     int info_gpu = 0;

//     int *devInfo = NULL; 
//     cuerr = cudaMalloc((void**)&devInfo, sizeof(int));
//     if (cuerr != cudaSuccess) {
//       errormessage("Error in cusolver_Dpotrf devInfo: %s\n",cudaGetErrorString(cuerr));
//     }
// #ifdef DEBUG_CUDA
//     printf("CUDA Malloc,  pointer address: %p, size: %d \n", &devInfo);
// #endif

    double *d_work = NULL;
    int d_lwork = 0;

    //status = cusolverDnDpotrf_bufferSize(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo),  n, A, lda, &d_lwork);
    status = cusolverDnDpotrf_bufferSize(cudaHandle, fill_mode_new_api(uplo),  n, A, lda, &d_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnDpotrf_buffer_size %s \n","aborting");
    }

    cuerr = cudaMalloc((void**) &d_work, sizeof(double) * d_lwork);
    //cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork already in bytes
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dpotrf d_work: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif

    //status = cusolverDnDpotrf(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), n, A, lda, d_work, d_lwork, devInfo);
    status = cusolverDnDpotrf(cudaHandle, fill_mode_new_api(uplo), n, A, lda, d_work, d_lwork, info_dev);

    if (status == CUSOLVER_STATUS_SUCCESS) {
    } else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      printf("status = CUSOLVER_STATUS_NOT_INITIALIZED\n");
    } else if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
      printf("status = CUSOLVER_STATUS_NOT_SUPPORTED\n");
    } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
      printf("status = CUSOLVER_STATUS_INVALID_VALUE\n");
    } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
      printf("status = CUSOLVER_STATUS_INTERNAL_ERROR\n"); 
    } else {
      printf("status = UNKNOWN\n");
    }

    //cuerr = cudaDeviceSynchronize();
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Dtrtri: cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
    //}

    // cuerr = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    // if (cuerr != cudaSuccess) {
    //   errormessage("Error in cusolver_Dpotrf info_gpu: %s\n",cudaGetErrorString(cuerr));
    // }

    // *info = info_gpu;
    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Dpotrf cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }

    // cuerr = cudaFree(devInfo);
    // if (cuerr != cudaSuccess) {
    //   errormessage("Error in cusolver_Dpotrf cuda_free(devInfo): %s\n",cudaGetErrorString(cuerr));
    // }
  }

  void cusolverSpotrf_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, int n, float *A, int lda, int *info) {
    cusolverStatus_t status;

    int info_gpu = 0;

    int *devInfo = NULL; 
    cudaError_t cuerr = cudaMalloc((void**)&devInfo, sizeof(int));
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Spotrf devInfo: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif

    float *d_work = NULL;
    int d_lwork = 0;

    //status = cusolverDnSpotrf_bufferSize(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo),  n, A, lda, &d_lwork);
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

    //status = cusolverDnSpotrf(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), n, A, lda, d_work, d_lwork, devInfo);
    status = cusolverDnSpotrf(cudaHandle, fill_mode_new_api(uplo), n, A, lda, d_work, d_lwork, devInfo);

    if (status == CUSOLVER_STATUS_SUCCESS) {
    } else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      printf("status = CUSOLVER_STATUS_NOT_INITIALIZED\n");
    } else if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
      printf("status = CUSOLVER_STATUS_NOT_SUPPORTED\n");
    } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
      printf("status = CUSOLVER_STATUS_INVALID_VALUE\n");
    } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
      printf("status = CUSOLVER_STATUS_INTERNAL_ERROR\n"); 
    } else {
      printf("status = UNKNOWN\n");
    }

    //cuerr = cudaDeviceSynchronize();
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Dtrtri: cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
    //}

    cuerr = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Spotrf info_gpu: %s\n",cudaGetErrorString(cuerr));
    }

    *info = info_gpu;
    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Spotrf cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }

    cuerr = cudaFree(devInfo);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Spotrf cuda_free(devInfo): %s\n",cudaGetErrorString(cuerr));
    }
  }

  void cusolverZpotrf_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, int n, double _Complex *A, int lda, int *info) {
    cusolverStatus_t status;

    int info_gpu = 0;

    int *devInfo = NULL; 
    cudaError_t cuerr = cudaMalloc((void**)&devInfo, sizeof(int));
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Zpotrf devInfo: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif

    cuDoubleComplex *d_work = NULL;
    int d_lwork = 0;
    cuDoubleComplex* A_casted = (cuDoubleComplex*) A;

    //status = cusolverDnZpotrf_bufferSize(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo),  n, A_casted, lda, &d_lwork);
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

    //status = cusolverDnZpotrf(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), n, A_casted, lda, d_work, d_lwork, devInfo);
    status = cusolverDnZpotrf(cudaHandle, fill_mode_new_api(uplo), n, A_casted, lda, d_work, d_lwork, devInfo);

    if (status == CUSOLVER_STATUS_SUCCESS) {
    } else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      printf("status = CUSOLVER_STATUS_NOT_INITIALIZED\n");
    } else if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
      printf("status = CUSOLVER_STATUS_NOT_SUPPORTED\n");
    } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
      printf("status = CUSOLVER_STATUS_INVALID_VALUE\n");
    } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
      printf("status = CUSOLVER_STATUS_INTERNAL_ERROR\n"); 
    } else {
      printf("status = UNKNOWN\n");
    }

    //cuerr = cudaDeviceSynchronize();
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Dtrtri: cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
    //}

    cuerr = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Zpotrf info_gpu: %s\n",cudaGetErrorString(cuerr));
    }

    *info = info_gpu;
    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Zpotrf cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }

    cuerr = cudaFree(devInfo);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Zpotrf cuda_free(devInfo): %s\n",cudaGetErrorString(cuerr));
    }
  }

  void cusolverCpotrf_elpa_wrapper (cusolverDnHandle_t cudaHandle, char uplo, int n, float _Complex *A, int lda, int *info) {
    cusolverStatus_t status;

    int info_gpu = 0;

    int *devInfo = NULL; 
    cudaError_t cuerr = cudaMalloc((void**)&devInfo, sizeof(int));
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Cpotrf devInfo: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", &devInfo);
#endif

    cuFloatComplex *d_work = NULL;
    int d_lwork = 0;
    cuFloatComplex* A_casted = (cuFloatComplex*) A;

    //status = cusolverDnCpotrf_bufferSize(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo),  n, A_casted, lda, &d_lwork);
    status = cusolverDnCpotrf_bufferSize(cudaHandle, fill_mode_new_api(uplo),  n, A_casted, lda, &d_lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      errormessage("Error in cusolverDnCpotrf_buffer_size %s \n","aborting");
    }

    cuerr = cudaMalloc((void**) &d_work, sizeof(cuFloatComplex) * d_lwork);
    //cuerr = cudaMalloc((void**) &d_work, d_lwork); // d_lwork is already in bytes
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Cpotrf d_work: %s\n",cudaGetErrorString(cuerr));
    }
#ifdef DEBUG_CUDA
    printf("CUDA Malloc,  pointer address: %p, size: %d \n", *d_work );
#endif

    //status = cusolverDnCpotrf(*((cusolverDnHandle_t*)handle), fill_mode_new_api(uplo), n, A_casted, lda, d_work, d_lwork, devInfo);
    status = cusolverDnCpotrf(cudaHandle, fill_mode_new_api(uplo), n, A_casted, lda, d_work, d_lwork, devInfo);

    if (status == CUSOLVER_STATUS_SUCCESS) {
    } else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
      printf("status = CUSOLVER_STATUS_NOT_INITIALIZED\n");
    } else if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
      printf("status = CUSOLVER_STATUS_NOT_SUPPORTED\n");
    } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
      printf("status = CUSOLVER_STATUS_INVALID_VALUE\n");
    } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
      printf("status = CUSOLVER_STATUS_INTERNAL_ERROR\n"); 
    } else {
      printf("status = UNKNOWN\n");
    }

    //cuerr = cudaDeviceSynchronize();
    //if (cuerr != cudaSuccess) {
    //  errormessage("Error in cusolver_Dtrtri: cudaDeviceSynchronize: %s\n",cudaGetErrorString(cuerr));
    //}

    cuerr = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Cpotrf info_gpu: %s\n",cudaGetErrorString(cuerr));
    }

    *info = info_gpu;
    cuerr = cudaFree(d_work);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Cpotrf cuda_free(d_work): %s\n",cudaGetErrorString(cuerr));
    }

    cuerr = cudaFree(devInfo);
    if (cuerr != cudaSuccess) {
      errormessage("Error in cusolver_Cpotrf cuda_free(devInfo): %s\n",cudaGetErrorString(cuerr));
    }
  }
#endif /* WITH_NVIDIA_CUSOLVER */


  void cublasDgemv_elpa_wrapper (cublasHandle_t cudaHandle, char trans, int m, int n, double alpha,
                               const double *A, int lda,  const double *x, int incx,
                               double beta, double *y, int incy) {

    //cublasStatus_t status = cublasDgemv(*((cublasHandle_t*)handle), operation_new_api(trans),
    cublasStatus_t status = cublasDgemv(cudaHandle, operation_new_api(trans),
                                        m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasDgemv\n");
    }

  }

  void cublasSgemv_elpa_wrapper (cublasHandle_t cudaHandle, char trans, int m, int n, float alpha,
                               const float *A, int lda,  const float *x, int incx,
                               float beta, float *y, int incy) {

    //cublasStatus_t status = cublasSgemv(*((cublasHandle_t*)handle), operation_new_api(trans),
    cublasStatus_t status = cublasSgemv(cudaHandle, operation_new_api(trans),
                m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasSgemv\n");
    }
  }

  void cublasZgemv_elpa_wrapper (cublasHandle_t cudaHandle, char trans, int m, int n, double _Complex alpha,
                               const double _Complex *A, int lda,  const double _Complex *x, int incx,
                               double _Complex beta, double _Complex *y, int incy) {

    cuDoubleComplex alpha_casted = *((cuDoubleComplex*)(&alpha));
    cuDoubleComplex beta_casted = *((cuDoubleComplex*)(&beta));

    const cuDoubleComplex* A_casted = (const cuDoubleComplex*) A;
    const cuDoubleComplex* x_casted = (const cuDoubleComplex*) x;
    cuDoubleComplex* y_casted = (cuDoubleComplex*) y;

    //cublasStatus_t status = cublasZgemv(*((cublasHandle_t*)handle), operation_new_api(trans),
    cublasStatus_t status = cublasZgemv(cudaHandle, operation_new_api(trans),
                m, n, &alpha_casted, A_casted, lda, x_casted, incx, &beta_casted, y_casted, incy);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasZgemv\n");
    }
  }

  void cublasCgemv_elpa_wrapper (cublasHandle_t cudaHandle, char trans, int m, int n, float _Complex alpha,
                               const float _Complex *A, int lda,  const float _Complex *x, int incx,
                               float _Complex beta, float _Complex *y, int incy) {

    cuFloatComplex alpha_casted = *((cuFloatComplex*)(&alpha));
    cuFloatComplex beta_casted = *((cuFloatComplex*)(&beta));

    const cuFloatComplex* A_casted = (const cuFloatComplex*) A;
    const cuFloatComplex* x_casted = (const cuFloatComplex*) x;
    cuFloatComplex* y_casted = (cuFloatComplex*) y;

    //cublasStatus_t status = cublasCgemv(*((cublasHandle_t*)handle), operation_new_api(trans),
    cublasStatus_t status = cublasCgemv(cudaHandle, operation_new_api(trans),
                m, n, &alpha_casted, A_casted, lda, x_casted, incx, &beta_casted, y_casted, incy);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasCgemv\n");
    }
  }


  void cublasDgemm_elpa_wrapper (cublasHandle_t cudaHandle, char transa, char transb, int m, int n, int k,
                               double alpha, const double *A, int lda,
                               const double *B, int ldb, double beta,
                               double *C, int ldc) {

    //cublasStatus_t status = cublasDgemm(*((cublasHandle_t*)handle), operation_new_api(transa), operation_new_api(transb),
    cublasStatus_t status = cublasDgemm(cudaHandle, operation_new_api(transa), operation_new_api(transb),
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasDgemm\n");
    }
  }

  void cublasSgemm_elpa_wrapper (cublasHandle_t cudaHandle, char transa, char transb, int m, int n, int k,
                               float alpha, const float *A, int lda,
                               const float *B, int ldb, float beta,
                               float *C, int ldc) {

    //cublasStatus_t status = cublasSgemm(((cublasHandle_t*)handle), operation_new_api(transa), operation_new_api(transb),
    cublasStatus_t status = cublasSgemm(cudaHandle, operation_new_api(transa), operation_new_api(transb),
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasSgemm\n");
    }
  }

  void cublasZgemm_elpa_wrapper (cublasHandle_t cudaHandle, char transa, char transb, int m, int n, int k,
                               double _Complex alpha, const double _Complex *A, int lda,
                               const double _Complex *B, int ldb, double _Complex beta,
                               double _Complex *C, int ldc) {

    cuDoubleComplex alpha_casted = *((cuDoubleComplex*)(&alpha));
    cuDoubleComplex beta_casted = *((cuDoubleComplex*)(&beta));

    const cuDoubleComplex* A_casted = (const cuDoubleComplex*) A;
    const cuDoubleComplex* B_casted = (const cuDoubleComplex*) B;
    cuDoubleComplex* C_casted = (cuDoubleComplex*) C;

    //cublasStatus_t status = cublasZgemm(*((cublasHandle_t*)handle), operation_new_api(transa), operation_new_api(transb),
    cublasStatus_t status = cublasZgemm(cudaHandle, operation_new_api(transa), operation_new_api(transb),
                m, n, k, &alpha_casted, A_casted, lda, B_casted, ldb, &beta_casted, C_casted, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasZgemm\n");
    }
  }

  void cublasCgemm_elpa_wrapper (cublasHandle_t cudaHandle, char transa, char transb, int m, int n, int k,
                               float _Complex alpha, const float _Complex *A, int lda,
                               const float _Complex *B, int ldb, float _Complex beta,
                               float _Complex *C, int ldc) {

    cuFloatComplex alpha_casted = *((cuFloatComplex*)(&alpha));
    cuFloatComplex beta_casted = *((cuFloatComplex*)(&beta));

    const cuFloatComplex* A_casted = (const cuFloatComplex*) A;
    const cuFloatComplex* B_casted = (const cuFloatComplex*) B;
    cuFloatComplex* C_casted = (cuFloatComplex*) C;

    //cublasStatus_t status =  cublasCgemm(*((cublasHandle_t*)handle), operation_new_api(transa), operation_new_api(transb),
    cublasStatus_t status =  cublasCgemm(cudaHandle, operation_new_api(transa), operation_new_api(transb),
                m, n, k, &alpha_casted, A_casted, lda, B_casted, ldb, &beta_casted, C_casted, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasCgemm\n");
    }
  }


  // todo: new CUBLAS API diverged from standard BLAS api for these functions
  // todo: it provides out-of-place (and apparently more efficient) implementation
  // todo: by passing B twice (in place of C as well), we should fall back to in-place algorithm


  void cublasDcopy_elpa_wrapper (cublasHandle_t cudaHandle, int n, double *x, int incx, double *y, int incy){

    //cublasStatus_t status = cublasDcopy(*((cublasHandle_t*)handle), n, x, incx, y, incy);
    cublasStatus_t status = cublasDcopy(cudaHandle, n, x, incx, y, incy);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasDcopy\n");
    }
  }

  void cublasScopy_elpa_wrapper (cublasHandle_t cudaHandle, int n, float *x, int incx, float *y, int incy){

    //cublasStatus_t status = cublasScopy(*((cublasHandle_t*)handle), n, x, incx, y, incy);
    cublasStatus_t status = cublasScopy(cudaHandle, n, x, incx, y, incy);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasScopy\n");
    }
  }

  void cublasZcopy_elpa_wrapper (cublasHandle_t cudaHandle, int n, double _Complex *x, int incx, double _Complex *y, int incy){
    const cuDoubleComplex* X_casted = (const cuDoubleComplex*) x;
          cuDoubleComplex* Y_casted = (      cuDoubleComplex*) y;

    //cublasStatus_t status = cublasZcopy(*((cublasHandle_t*)handle), n, X_casted, incx, Y_casted, incy);
    cublasStatus_t status = cublasZcopy(cudaHandle, n, X_casted, incx, Y_casted, incy);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasZcopy\n");
    }
  }

  void cublasCcopy_elpa_wrapper (cublasHandle_t cudaHandle, int n, float _Complex *x, int incx, float _Complex *y, int incy){
    const cuFloatComplex* X_casted = (const cuFloatComplex*) x;
          cuFloatComplex* Y_casted = (      cuFloatComplex*) y;

    //cublasStatus_t status = cublasCcopy(handle, n, X_casted, incx, Y_casted, incy);
    cublasStatus_t status = cublasCcopy(cudaHandle, n, X_casted, incx, Y_casted, incy);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasCcopy\n");
    }
  }

  void cublasDtrsm_elpa_wrapper (cublasHandle_t cudaHandle, char side, char uplo, char transa, char diag,
                               int m, int n, double alpha, const double *A,
                               int lda, double *B, int ldb){

    //cublasStatus_t status = cublasDtrsm(*((cublasHandle_t*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
    cublasStatus_t status = cublasDtrsm(cudaHandle, side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                                        diag_type_new_api(diag), m, n, &alpha, A, lda, B, ldb);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasDtrsm\n");
    }
  }

  void cublasStrsm_elpa_wrapper (cublasHandle_t cudaHandle, char side, char uplo, char transa, char diag,
                               int m, int n, float alpha, const float *A,
                               int lda, float *B, int ldb){

    //cublasStatus_t status = cublasStrsm(*((cublasHandle_t*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
    cublasStatus_t status = cublasStrsm(cudaHandle, side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                                        diag_type_new_api(diag), m, n, &alpha, A, lda, B, ldb);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasStrsm\n");
    }
  }

  void cublasZtrsm_elpa_wrapper (cublasHandle_t cudaHandle, char side, char uplo, char transa, char diag,
                               int m, int n, double _Complex alpha, const double _Complex *A,
                               int lda, double _Complex *B, int ldb){

    cuDoubleComplex alpha_casted = *((cuDoubleComplex*)(&alpha));

    const cuDoubleComplex* A_casted = (const cuDoubleComplex*) A;
    cuDoubleComplex* B_casted = (cuDoubleComplex*) B;

    //cublasStatus_t status = cublasZtrsm(*((cublasHandle_t*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
    cublasStatus_t status = cublasZtrsm(cudaHandle, side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                diag_type_new_api(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasZtrsm\n");
    }
  }

  void cublasCtrsm_elpa_wrapper (cublasHandle_t cudaHandle, char side, char uplo, char transa, char diag,
                               int m, int n, float _Complex alpha, const float _Complex *A,
                               int lda, float _Complex *B, int ldb){

    cuFloatComplex alpha_casted = *((cuFloatComplex*)(&alpha));

    const cuFloatComplex* A_casted = (const cuFloatComplex*) A;
    cuFloatComplex* B_casted = (cuFloatComplex*) B;

    //cublasStatus_t status = cublasCtrsm(*((cublasHandle_t*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
    cublasStatus_t status = cublasCtrsm(cudaHandle, side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                diag_type_new_api(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasCtrsm\n");
    }
  }


  void cublasDtrmm_elpa_wrapper (cublasHandle_t cudaHandle, char side, char uplo, char transa, char diag,
                               int m, int n, double alpha, const double *A,
                               int lda, double *B, int ldb){

    //cublasStatus_t status = cublasDtrmm(*((cublasHandle_t*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
    cublasStatus_t status = cublasDtrmm(cudaHandle, side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                diag_type_new_api(diag), m, n, &alpha, A, lda, B, ldb, B, ldb);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasDtrmm\n");
    }
  }

  void cublasStrmm_elpa_wrapper (cublasHandle_t cudaHandle, char side, char uplo, char transa, char diag,
                               int m, int n, float alpha, const float *A,
                               int lda, float *B, int ldb){

    //cublasStatus_t status = cublasStrmm(*((cublasHandle_t*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
    cublasStatus_t status = cublasStrmm(cudaHandle, side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                diag_type_new_api(diag), m, n, &alpha, A, lda, B, ldb, B, ldb);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasStrmm\n");
    }
  }

  void cublasZtrmm_elpa_wrapper (cublasHandle_t cudaHandle, char side, char uplo, char transa, char diag,
                               int m, int n, double _Complex alpha, const double _Complex *A,
                               int lda, double _Complex *B, int ldb){

    cuDoubleComplex alpha_casted = *((cuDoubleComplex*)(&alpha));

    const cuDoubleComplex* A_casted = (const cuDoubleComplex*) A;
    cuDoubleComplex* B_casted = (cuDoubleComplex*) B;

    //cublasStatus_t status = cublasZtrmm(*((cublasHandle_t*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
    cublasStatus_t status = cublasZtrmm(cudaHandle, side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                diag_type_new_api(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb, B_casted, ldb);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasZtrmm\n");
    }
  }

  void cublasCtrmm_elpa_wrapper (cublasHandle_t cudaHandle, char side, char uplo, char transa, char diag,
                               int m, int n, float _Complex alpha, const float _Complex *A,
                               int lda, float _Complex *B, int ldb){

    cuFloatComplex alpha_casted = *((cuFloatComplex*)(&alpha));

    const cuFloatComplex* A_casted = (const cuFloatComplex*) A;
    cuFloatComplex* B_casted = (cuFloatComplex*) B;

    //cublasStatus_t status = cublasCtrmm(*((cublasHandle_t*)handle), side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
    cublasStatus_t status = cublasCtrmm(cudaHandle, side_mode_new_api(side), fill_mode_new_api(uplo), operation_new_api(transa),
                diag_type_new_api(diag), m, n, &alpha_casted, A_casted, lda, B_casted, ldb, B_casted, ldb);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasCtrmm\n");
    }
  }

  // result can be on host or device depending on pointer mode
  void cublasDdot_elpa_wrapper (cublasHandle_t cudaHandle, int length, const double *X, int incx, const double *Y, int incy, double *result) {
    cublasStatus_t status = cublasDdot(cudaHandle, length, X, incx, Y, incy, result);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasDdot\n");
    }
  }

  void cublasSdot_elpa_wrapper (cublasHandle_t cudaHandle, int length, const float *X, int incx, const float *Y, int incy, float *result) {

    //cublasSetPointerMode(cudaHandle, CUBLAS_POINTER_MODE_DEVICE);
    cublasStatus_t status = cublasSdot(cudaHandle, length, X, incx, Y, incy, result);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasSdot\n");
    }
  }

  void cublasZdot_elpa_wrapper (char conju, cublasHandle_t cudaHandle, int length, const double _Complex *X, int incx, const double _Complex *Y, int incy, double _Complex *result) {

    //cublasSetPointerMode(cudaHandle, CUBLAS_POINTER_MODE_DEVICE);
    const cuDoubleComplex* X_casted = (const cuDoubleComplex*) X;
    const cuDoubleComplex* Y_casted = (const cuDoubleComplex*) Y;
          cuDoubleComplex* result_casted = (cuDoubleComplex*) result;
    cublasStatus_t status;
    if (conju == 'C' || conju == 'c') {
      status = cublasZdotc(cudaHandle, length, X_casted, incx, Y_casted, incy, result_casted);
    }
    if (conju == 'U' || conju == 'u') {
	    printf("not using conjugate in dot\n");
      status = cublasZdotu(cudaHandle, length, X_casted, incx, Y_casted, incy, result_casted);
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasZdot\n");
    }
    //cuDoubleComplex* result = (cuDoubleComplex*) result_casted;
  }

  void cublasCdot_elpa_wrapper (char conju, cublasHandle_t cudaHandle, int length, const float _Complex *X, int incx, const float _Complex *Y, int incy, float _Complex *result) {

    //cublasSetPointerMode(cudaHandle, CUBLAS_POINTER_MODE_DEVICE);
    const cuFloatComplex* X_casted = (const cuFloatComplex*) X;
    const cuFloatComplex* Y_casted = (const cuFloatComplex*) Y;
          cuFloatComplex* result_casted = (cuFloatComplex*) result;

    cublasStatus_t status;
    if (conju == 'C' || conju == 'c') {
      status = cublasCdotc(cudaHandle, length, X_casted, incx, Y_casted, incy, result_casted);
    }
    if (conju == 'U' || conju == 'u') {
      status = cublasCdotu(cudaHandle, length, X_casted, incx, Y_casted, incy, result_casted);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasCdot\n");
    }
    printf("Leaving setPointer\n");
  }

  void cublasSetPointerModeFromC(cublasHandle_t cudaHandle, cublasPointerMode_t mode) {
    cublasSetPointerMode(cudaHandle, mode);
  }

  void cublasGetPointerModeFromC(cublasHandle_t cudaHandle, cublasPointerMode_t *mode) {
    //cublasPointerMode_t mode_tmp;
    //cublasGetPointerMode(cudaHandle, &mode_tmp);
    cublasGetPointerMode(cudaHandle, mode);
    //printf("in getpointer mode %d \n",mode_tmp);
    //printf("in getpointer mode %d \n",*mode);
    //*mode = mode_tmp;
  }

  int cublasPointerModeDeviceFromC(void) {
      int val = CUBLAS_POINTER_MODE_DEVICE;
      return val;
  }

  int cublasPointerModeHostFromC(void) {
      int val = CUBLAS_POINTER_MODE_HOST;
      return val;
  }

  void cublasDscal_elpa_wrapper (cublasHandle_t cudaHandle, int n, double alpha, double *x, int incx){

    cublasStatus_t status = cublasDscal(cudaHandle, n, &alpha, x, incx);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasDscal\n");
    }
  }

  void cublasSscal_elpa_wrapper (cublasHandle_t cudaHandle, int n, float alpha, float *x, int incx){

    cublasStatus_t status = cublasSscal(cudaHandle, n, &alpha, x, incx);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasSscal\n");
    }
  }

  void cublasZscal_elpa_wrapper (cublasHandle_t cudaHandle, int n, double _Complex alpha, double _Complex *x, int incx){
    cuDoubleComplex alpha_casted = *((cuDoubleComplex*)(&alpha));
    cuDoubleComplex* X_casted     = (cuDoubleComplex*) x;

    cublasStatus_t status = cublasZscal(cudaHandle, n, &alpha_casted, X_casted, incx);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasZscal\n");
    }
  }

  void cublasCscal_elpa_wrapper (cublasHandle_t cudaHandle, int n, float _Complex alpha, float _Complex *x, int incx){
    cuFloatComplex alpha_casted = *((cuFloatComplex*)(&alpha));
    cuFloatComplex* X_casted     = (cuFloatComplex*) x;

    cublasStatus_t status = cublasCscal(cudaHandle, n, &alpha_casted, X_casted, incx);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasCscal\n");
    }
  }

  void cublasDaxpy_elpa_wrapper (cublasHandle_t cudaHandle, int n, double alpha, double *x, int incx, double *y, int incy){

    cublasStatus_t status = cublasDaxpy(cudaHandle, n, &alpha, x, incx, y, incy);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasDaxpy\n");
    }
  }
  
  void cublasSaxpy_elpa_wrapper (cublasHandle_t cudaHandle, int n, float alpha, float *x, int incx, float *y, int incy){

    cublasStatus_t status = cublasSaxpy(cudaHandle, n, &alpha, x, incx, y, incy);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasSaxpy\n");
    }
  }

  void cublasZaxpy_elpa_wrapper (cublasHandle_t cudaHandle, int n, double _Complex alpha, double _Complex *x, int incx, double _Complex *y, int incy){

    cuDoubleComplex alpha_casted = *((cuDoubleComplex*)(&alpha));
    cuDoubleComplex* X_casted     = (cuDoubleComplex*) x;
    cuDoubleComplex* Y_casted     = (cuDoubleComplex*) y;

    cublasStatus_t status = cublasZaxpy(cudaHandle, n, &alpha_casted, X_casted, incx, Y_casted, incy);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasZaxpy\n");
    }
  }

  void cublasCaxpy_elpa_wrapper (cublasHandle_t cudaHandle, int n, float _Complex alpha, float _Complex *x, int incx, float _Complex *y, int incy){

    cuFloatComplex alpha_casted = *((cuFloatComplex*)(&alpha));
    cuFloatComplex* X_casted     = (cuFloatComplex*) x;
    cuFloatComplex* Y_casted     = (cuFloatComplex*) y;

    cublasStatus_t status = cublasCaxpy(cudaHandle, n, &alpha_casted, X_casted, incx, Y_casted, incy);
    if (status != CUBLAS_STATUS_SUCCESS) {
       printf("error when calling cublasCaxpy\n");
    }
  }

}
#endif /* WITH_NVIDIA_GPU_VERSION */
