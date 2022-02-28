#include <mkl.h>
#include <mkl_omp_offload.h>

#include <omp.h>

#include <complex>
#include <iostream>
#include <cstdint>

int openmpOffloadOmpChosenGpu;
//meaningless but needed for ELPA
int openmpOffloadMemcpyDeviceToHost   = 10;
int openmpOffloadMemcpyHostToDevice   = 100;
int openmpOffloadMemcpyDeviceToDevice = 1000;

extern "C" {
  int openmpOffloadMemcpyDeviceToDeviceFromC(){
    return openmpOffloadMemcpyDeviceToDevice;
  }
  int openmpOffloadMemcpyHostToDeviceFromC(){
    return openmpOffloadMemcpyHostToDevice;
  }
  int openmpOffloadMemcpyDeviceToHostFromC(){
    return openmpOffloadMemcpyDeviceToHost;
  }

  int openmpOffloadSetDeviceFromC(int targetGpuDeviceId) {
    openmpOffloadChosenGpu = targetGpuDeviceId;
    // include here more checks wether this is a reasonable devide id
    return 1;
  }

  int openmpOffloadblasCreateFromC(intptr_t* handle){
    //stub function
    return 1;
  }

  int openmpOffloadsolverCreateFromC(intptr_t* handle){
    //stub function
    return 1;
  }

  int openmpOffloadMallocFromC(intptr_t *a, size_t elems) {
      int device = oneapiOmpChosenGpu;
      *a = (intptr_t) omp_target_alloc(elems, device);
      if (*a) {
          std::cout << "Allocated " << elems << "B starting at address " << *a << std::endl;
          return 1;
      } else {
          std::cout << "Allocation failed!" << std::endl;
          return 0;
      }
  }

  int opemmpOffloadFreeFromC(intptr_t *a) {
      int device = oneapiOmpChosenGpu;
      void * ptr = reinterpret_cast<void *>(*a);
      omp_target_free(ptr, device);
      return 1;
  }

  int openmpOffloadMemcpyFromC(void *dst, void *src, size_t size, int direction) {
      // FIXME: Do not use the default device, but some other device that has been specified.
      // TODO:  Find out how this is done in ELPA.
      int hostDevice = omp_get_initial_device();
      int gpuDevice = omp_get_default_device();
      int dstDevice;
      int srcDevice;
      if (direction == oneapiOmpMemcpyDeviceToDevice) {
          dstDevice = gpuDevice;
          srcDevice = gpuDevice;
      } else if (direction == oneapiOmpMemcpyDeviceToHost) {
          dstDevice = hostDevice;
          srcDevice = gpuDevice;
      } else if (direction == oneapiOmpMemcpyHostToDevice) {
          dstDevice = gpuDevice;
          srcDevice = hostDevice;
      } else {
          std::cerr << "Direction of transfer for memcpy unknown" << std::endl;
          return 0;
      }
      int retVal = omp_target_memcpy(dst, src, size, 0, 0, dstDevice, srcDevice);
      if (!retVal){
	return 0;
      } else {
        std::cout << "Copied " << size << "B successfully from " << reinterpret_cast<intptr_t>(src) << " to " << reinterpret_cast<intptr_t>(dst) << "." << std::endl;
	return 1;
      }
  }

  int openmpOffloadMemsetFromC(void *mem, int val, size_t size) {
    char *mem_bytes = reinterpret_cast<char *>(mem);
    #pragma omp target teams loop is_device_ptr(mem) device(openmpOffloadChosenGpu)
    for (size_t i = 0; i < size; i++) {
      mem_bytes[i] = val;
    }
    return 1;
  }

  void mklOpenmpOffloadDgemmFromC(intptr_t *handle, char cta, char ctb, int* m, int *n, int *k, double *alpha, void *a, int *lda, void *b, int *ldb, double *beta, void *c, int *ldc) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      dgemm(&cta, &ctb, m, n, k, alpha, reinterpret_cast<double *>(a), lda, reinterpret_cast<double *>(b), ldb, beta, reinterpret_cast<double *>(c), ldc);
      std::cout << "DGEMM" << std::endl;
  }  

  void mklOpenmpOffloadSgemmFromC(intptr_t *handle, char cta, char ctb, int* m, int *n, int *k, float *alpha, void *a, int *lda, void *b, int *ldb, float *beta, void *c, int *ldc) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      sgemm(&cta, &ctb, m, n, k, alpha, reinterpret_cast<float *>(a), lda, reinterpret_cast<float *>(b), ldb, beta, reinterpret_cast<float *>(c), ldc);
      std::cout << "SGEMM" << std::endl;
  }

  void mklOpenmpOffloadZgemmFromC(intptr_t *handle, char cta, char ctb, int* m, int *n, int *k, MKL_Complex16 *alpha, void *a, int *lda, void *b, int *ldb, MKL_Complex16 *beta, void *c, int *ldc) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      zgemm(&cta, &ctb, m, n, k, alpha, reinterpret_cast<MKL_Complex16 *>(a), lda, reinterpret_cast<MKL_Complex16 *>(b), ldb, beta, reinterpret_cast<MKL_Complex16 *>(c), ldc);
      std::cout << "ZGEMM" << std::endl;
  }

  void mklOpenmpOffloadCgemmFromC(intptr_t *handle, char cta, char ctb, int* m, int *n, int *k, MKL_Complex8 *alpha, void *a, int *lda, void *b, int *ldb, MKL_Complex8 *beta, void *c, int *ldc) {
      //handle not needed
    #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
    cgemm(&cta, &ctb, m, n, k, alpha, reinterpret_cast<MKL_Complex8 *>(a), lda, reinterpret_cast<MKL_Complex8 *>(b), ldb, beta, reinterpret_cast<MKL_Complex8 *>(c), ldc);
    std::cout << "CGEMM" << std::endl;
  }

  void mklOpenmpOffloadDtrtriFromC(intptr_t *handle, char uplo, char diag, int* n, void *a, int *lda, int *info) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      dtrtri(&uplo, &diag, n, reinterpret_cast<double *>(a), lda, info);
      std::cout << "DTRTRI" << std::endl;
  }  

  void mklOpenmpOffloadStrtriFromC(intptr_t *handle, char uplo, char diag, int* n, void *a, int *lda, int *info) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      strtri(&uplo, &diag, n, reinterpret_cast<float *>(a), lda, info);
      std::cout << "STRTRI" << std::endl;
  }  

  void mklOpenmpOffloadZtrtriFromC(intptr_t *handle, char uplo, char diag, int* n, void *a, int *lda, int *info) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      ztrtri(&uplo, &diag, n, reinterpret_cast<MKL_Complex16 *>(a), lda, info);
      std::cout << "ZTRTRI" << std::endl;
  }  

  void mklOpenmpOffloadCtrtriFromC(intptr_t *handle, char uplo, char diag, int* n, void *a, int *lda, int *info) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      ctrtri(&uplo, &diag, n, reinterpret_cast<MKL_Complex8 *>(a), lda, info);
      std::cout << "CTRTRI" << std::endl;
  }  

  void mklOpenmpOffloadDpotrfFromC(intptr_t *handle, char uplo, int* n, void *a, int *lda, int *info) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      dpotrf(&uplo, n, reinterpret_cast<double *>(a), lda, info);
      std::cout << "DPOTRF" << std::endl;
  }  

  void mklOpenmpOffloadSpotrfFromC(intptr_t *handle, char uplo, int* n, void *a, int *lda, int *info) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      spotrf(&uplo, n, reinterpret_cast<float *>(a), lda, info);
      std::cout << "SPOTRF" << std::endl;
  }  

  void mklOpenmpOffloadZpotrfFromC(intptr_t *handle, char uplo, int* n, void *a, int *lda, int *info) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      zpotrf(&uplo, n, reinterpret_cast<MKL_Complex16 *>(a), lda, info);
      std::cout << "ZPOTRF" << std::endl;
  }  

  void mklOpenmpOffloadCpotrfFromC(intptr_t *handle, char uplo, int* n, void *a, int *lda, int *info) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      cpotrf(&uplo, n, reinterpret_cast<MKL_Complex8 *>(a), lda, info);
      std::cout << "CPOTRF" << std::endl;
  }

  void mklOpenmpOffloadDcopyFromC(intptr_t *handle, int* n, void *x, int *incx, void *y, int *incy) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      dcopy(n, reinterpret_cast<double *>(x), incx, reinterpret_cast<double *>(y), incy);
      std::cout << "DCOPY" << std::endl;
  }  

  void mklOpenmpOffloadScopyFromC(intptr_t *handle, int* n, void *x, int *incx, void *y, int *incy) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      scopy(n, reinterpret_cast<float *>(x), incx, reinterpret_cast<float *>(y), incy);
      std::cout << "SCOPY" << std::endl;
  }  

  void mklOpenmpOffloadZcopyFromC(intptr_t *handle, int* n, void *x, int *incx, void *y, int *incy) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      zcopy(n, reinterpret_cast<MKL_Complex16 *>(x), incx, reinterpret_cast<MKL_Complex16 *>(y), incy);
      std::cout << "ZCOPY" << std::endl;
  }  

  void mklOpenmpOffloadCcopyFromC(intptr_t *handle, int* n, void *x, int *incx, void *y, int *incy) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      ccopy(n, reinterpret_cast<MKL_Complex8 *>(x), incx, reinterpret_cast<MKL_Complex8 *>(y), incy);
      std::cout << "CCOPY" << std::endl;
  }  


  void mklOpenmpOffloadDtrmmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int* m, int *n, double *alpha, void *a, int *lda, void *b, int *ldb) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      dtrmm(&side, &uplo, &trans, &diag, m, n, alpha, reinterpret_cast<double *>(a), lda, reinterpret_cast<double *>(b), ldb);
      std::cout << "DTRMM" << std::endl;
  }  

  void mklOpenmpOffloadStrmmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int* m, int *n, float *alpha, void *a, int *lda, void *b, int *ldb) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      strmm(&side, &uplo, &trans, &diag, m, n, alpha, reinterpret_cast<float *>(a), lda, reinterpret_cast<float *>(b), ldb);
      std::cout << "STRMM" << std::endl;
  }  

  void mklOpenmpOffloadZtrmmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int* m, int *n, MKL_Complex16 *alpha, void *a, int *lda, void *b, int *ldb) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      strmm(&side, &uplo, &trans, &diag, m, n, alpha, reinterpret_cast<MKL_Complex16 *>(a), lda, reinterpret_cast<MKL_Complex16 *>(b), ldb);
      std::cout << "ZTRMM" << std::endl;
  }  

  void mklOpenmpOffloadCtrmmFromC(intptr_t *handle, char side, char uplo, char trans,  char diag, int* m, int *n, MKL_Complex8 *alpha, void *a, int *lda, void *b, int *ldb) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      strmm(&side, &uplo, &trans, &diag, m, n, alpha, reinterpret_cast<MKL_Complex8 *>(a), lda, reinterpret_cast<MKL_Complex8 *>(b), ldb);
      std::cout << "CTRMM" << std::endl;
  }  


  void mklOpenmpOffloadDtrsmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int* m, int *n, double *alpha, void *a, int *lda, void *b, int *ldb) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      dtrsm(&side, &uplo, &trans, &diag, m, n, alpha, reinterpret_cast<double *>(a), lda, reinterpret_cast<double *>(b), ldb);
      std::cout << "DTRSM" << std::endl;
  }  

  void mklOpenmpOffloadStrsmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int* m, int *n, float *alpha, void *a, int *lda, void *b, int *ldb) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      dtrsm(&side, &uplo, &trans, &diag, m, n, alpha, reinterpret_cast<float *>(a), lda, reinterpret_cast<float *>(b), ldb);
      std::cout << "STRSM" << std::endl;
  }  

  void mklOpenmpOffloadZtrsmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int* m, int *n, MKL_Complex16 *alpha, void *a, int *lda, void *b, int *ldb) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      ztrsm(&side, &uplo, &trans, &diag, m, n, alpha, reinterpret_cast<MKL_Complex16 *>(a), lda, reinterpret_cast<float *>(b), ldb);
      std::cout << "ZTRSM" << std::endl;
  }  

  void mklOpenmpOffloadCtrsmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int* m, int *n, MKL_Complex8 *alpha, void *a, int *lda, void *b, int *ldb) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      ctrsm(&side, &uplo, &trans, &diag, m, n, alpha, reinterpret_cast<MKL_Complex8 *>(a), lda, reinterpret_cast<float *>(b), ldb);
      std::cout << "CTRSM" << std::endl;
  }

  void mklOpenmpOffloadDgemvFromC(intptr_t *handle, char cta, int* m, int *n, double *alpha, void *a, int *lda, void *x, int *incx, double *beta, void *y, int *incy) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      dgemv(&cta, m, n, alpha, reinterpret_cast<double *>(a), lda, reinterpret_cast<double *>(x), incx, beta, reinterpret_cast<double *>(y), incy);
      std::cout << "DGEMV" << std::endl;
  }  

  void mklOpenmpOffloadSgemvFromC(intptr_t *handle, char cta, int* m, int *n, float *alpha, void *a, int *lda, void *x, int *incx, float *beta, void *y, int *incy) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      sgemv(&cta, m, n, alpha, reinterpret_cast<float *>(a), lda, reinterpret_cast<float *>(x), incx, beta, reinterpret_cast<float *>(y), incy);
      std::cout << "SGEMV" << std::endl;
  }  

  void mklOpenmpOffloadZgemvFromC(intptr_t *handle, char cta, int* m, int *n, MKL_Complex16 *alpha, void *a, int *lda, void *x, int *incx, MKL_Complex16 *beta, void *y, int *incy) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      zgemv(&cta, m, n, alpha, reinterpret_cast<MKL_Complex16 *>(a), lda, reinterpret_cast<MKL_Complex16 *>(x), incx, beta, reinterpret_cast<MKL_Complex16 *>(y), incy);
      std::cout << "ZGEMV" << std::endl;
  }  

  void mklOpenmpOffloadCgemvFromC(intptr_t *handle, char cta, int* m, int *n, MKL_Complex8 *alpha, void *a, int *lda, void *x, int *incx, MKL_Complex8 *beta, void *y, int *incy) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      zgemv(&cta, m, n, alpha, reinterpret_cast<MKL_Complex8 *>(a), lda, reinterpret_cast<MKL_Complex8 *>(x), incx, beta, reinterpret_cast<MKL_Complex8 *>(y), incy);
      std::cout << "CGEMV" << std::endl;
  }  



}
