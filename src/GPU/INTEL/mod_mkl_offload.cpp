#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

#include "config.h"

#ifdef WITH_INTEL_GPU_VERSION
#include "mkl.h"
#include "mkl_omp_offload.h"
#endif

extern "C" {
void mkl_offload_dgemm(char transa, char transbm int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {

#ifdef WITH_INTEL_GPU_VERSION
  std::cout << "In mkl_offload_dgemm" << std::endl;

  // at a later time the device should be set differently
  int dnum = 0;

  int sizea, sizeb, sizec;
  std::cout << "m=" << m << "lda=" << lda << "ldc=" << ldc << std::endl;
  std::cout << "n=" << n << "ldb=" << ldb << std::endl;
  std::cout << "k=" << k << std::endl;
  std::cout << "alpha=" << alpha << std::endl;
  std::cout << "beta=" << beta << std::endl;


  std::cout << "Transa=",transa << std::endl;
  std::cout << "Transb=",transb << std::endl;

  sizea = lda * k;
  sizeb = ldb * n;
  sizec = ldc * n;


  #pragma omp target data map(to : a [0:sizea], b [0:sizeb]) map(tofrom : c [0:sizec]) device(dnum)
  {
  #pragma omp target variant dispatch device(dnum) use_device_ptr(a, b, c)
  dgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
  }
  std::cout << "leaving mkl_offload_dgemm" << std::endl;
#else
  std::cout << "ERROR: calling mkl_offload_dgemm without build for Intel GPU support!" << std::endl;
  std::cout << "ERROR: You should never see this message" << std::endl;
#endif
}

#ifdef WANT_SINGLE_PRECISION_REAL
void mkl_offload_sgemm(char transa, char transbm int m, int n, int k, float alpha, float *a, int lda, float *b, int ldb, float beta, float *c, int ldc) {

#ifdef WITH_INTEL_GPU_VERSION
  std::cout << "In mkl_offload_sgemm" << std::endl;

  // at a later time the device should be set differently
  int dnum = 0;

  int sizea, sizeb, sizec;
  std::cout << "m=" << m << "lda=" << lda << "ldc=" << ldc << std::endl;
  std::cout << "n=" << n << "ldb=" << ldb << std::endl;
  std::cout << "k=" << k << std::endl;
  std::cout << "alpha=" << alpha << std::endl;
  std::cout << "beta=" << beta << std::endl;

  std::cout << "Transa=",transa << std::endl;
  std::cout << "Transb=",transb << std::endl;

  sizea = lda * k;
  sizeb = ldb * n;
  sizec = ldc * n;


  #pragma omp target data map(to : a [0:sizea], b [0:sizeb]) map(tofrom : c [0:sizec]) device(dnum)
  {
  #pragma omp target variant dispatch device(dnum) use_device_ptr(a, b, c)
  sgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
  }
  std::cout << "leaving mkl_offload_sgemm" << std::endl;
#else
  std::cout << "ERROR: calling mkl_offload_sgemm without build for Intel GPU support!" << std::endl;
  std::cout << "ERROR: You should never see this message" << std::endl;
#endif
}
#endif

void mkl_offload_zgemm(char transa, char transbm int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {

#ifdef WITH_INTEL_GPU_VERSION
  std::cout << "In mkl_offload_zgemm" << std::endl;

  // at a later time the device should be set differently
  int dnum = 0;

  int sizea, sizeb, sizec;
  std::cout << "m=" << m << "lda=" << lda << "ldc=" << ldc << std::endl;
  std::cout << "n=" << n << "ldb=" << ldb << std::endl;
  std::cout << "k=" << k << std::endl;
  std::cout << "alpha=" << alpha << std::endl;
  std::cout << "beta=" << beta << std::endl;

  std::cout << "Transa=",transa << std::endl;
  std::cout << "Transb=",transb << std::endl;

  sizea = lda * k;
  sizeb = ldb * n;
  sizec = ldc * n;


  #pragma omp target data map(to : a [0:sizea], b [0:sizeb]) map(tofrom : c [0:sizec]) device(dnum)
  {
  #pragma omp target variant dispatch device(dnum) use_device_ptr(a, b, c)
  zgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
  }
  std::cout << "leaving mkl_offload_dgemm" << std::endl;
#else
  std::cout << "ERROR: calling mkl_offload_zgemm without build for Intel GPU support!" << std::endl;
  std::cout << "ERROR: You should never see this message" << std::endl;
#endif
}

#ifdef WANT_SINGLE_PRECISION_COMPLEX
void mkl_offload_cgemm(char transa, char transbm int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {

#ifdef WITH_INTEL_GPU_VERSION
  std::cout << "In mkl_offload_cgemm" << std::endl;

  // at a later time the device should be set differently
  int dnum = 0;

  int sizea, sizeb, sizec;
  std::cout << "m=" << m << "lda=" << lda << "ldc=" << ldc << std::endl;
  std::cout << "n=" << n << "ldb=" << ldb << std::endl;
  std::cout << "k=" << k << std::endl;
  std::cout << "alpha=" << alpha << std::endl;
  std::cout << "beta=" << beta << std::endl;

  std::cout << "Transa=",transa << std::endl;
  std::cout << "Transb=",transb << std::endl;

  sizea = lda * k;
  sizeb = ldb * n;
  sizec = ldc * n;


  #pragma omp target data map(to : a [0:sizea], b [0:sizeb]) map(tofrom : c [0:sizec]) device(dnum)
  {
  #pragma omp target variant dispatch device(dnum) use_device_ptr(a, b, c)
  cgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
  }
  std::cout << "leaving mkl_offload_cgemm" << std::endl;
#else
  std::cout << "ERROR: calling mkl_offload_cgemm without build for Intel GPU support!" << std::endl;
  std::cout << "ERROR: You should never see this message" << std::endl;
#endif
}
#endif

}
