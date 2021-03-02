#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
//#include <complex>

#include "config.h"

#ifdef WITH_INTEL_GPU_VERSION
#include "mkl.h"
#include "mkl_omp_offload.h"
#include "mkl_types.h"
//#include <omp.h>
#endif

//#define MKL_Complex16 std::complex<double>
//#define MKL_Complex8 std::complex<float>

extern "C" {
void mkl_offload_dgemm_c(char transa, char transb, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc) {

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


  std::cout << "Transa=" << transa << std::endl;
  std::cout << "Transb=" << transb << std::endl;

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

void mkl_offload_dgemv_c(char trans, int m, int n, double alpha, double *a, int lda, double *x, int incx, double beta, double *y, int incy) {

#ifdef WITH_INTEL_GPU_VERSION
  std::cout << "In mkl_offload_dgemv" << std::endl;

  // at a later time the device should be set differently
  int dnum = 0;

  int sizea, sizex, sizey;
  std::cout << "m=" << m << "lda=" << lda << std::endl;
  std::cout << "n=" << n << std::endl;
  //std::cout << "sizeX=" << sizeX << std::endl;
  //std::cout << "sizeY=" << sizeY << std::endl;
  std::cout << "alpha=" << alpha << std::endl;
  std::cout << "beta=" << beta << std::endl;


  std::cout << "Trans=" << trans << std::endl;

  sizea = lda * n;
  sizex = n;
  sizey = m;


  #pragma omp target data map(to : a [0:sizea], x [0:sizex]) map(tofrom : y [0:sizey]) device(dnum)
  {
  #pragma omp target variant dispatch device(dnum) use_device_ptr(a, x, y)
  dgemv(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
  std::cout << "leaving mkl_offload_dgemv" << std::endl;
#else
  std::cout << "ERROR: calling mkl_offload_dgemv without build for Intel GPU support!" << std::endl;
  std::cout << "ERROR: You should never see this message" << std::endl;
#endif
}

void mkl_offload_dtrmm_c(char side, char uplo, char trans, char diag, int m, int n, double alpha, double *a, int lda, double *b, int ldb) {

#ifdef WITH_INTEL_GPU_VERSION
  std::cout << "In mkl_offload_dtrmm" << std::endl;

  // at a later time the device should be set differently
  int dnum = 0;

  int sizea, sizeb;
  std::cout << "m=" << m << "lda=" << lda << std::endl;
  std::cout << "n=" << n << std::endl;
  //std::cout << "sizeX=" << sizeX << std::endl;
  //std::cout << "sizeY=" << sizeY << std::endl;
  std::cout << "alpha=" << alpha << std::endl;

  if (side == 'L' || side == 'l') {
    std::cout << "Setting a to case L" << std::endl;
    sizea = lda * m;
  }
  if (side == 'R' || side == 'r') {
    std::cout << "Setting a to case R" << std::endl;
    sizea = lda * n;
  }
  sizeb = ldb * n;


  #pragma omp target data map(to : a [0:sizea]) map(tofrom : b [0:sizeb]) device(dnum)
  {
  #pragma omp target variant dispatch device(dnum) use_device_ptr(a, b)
  dtrmm(&side, &uplo, &trans, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
  }
  std::cout << "leaving mkl_offload_dtrmm" << std::endl;
#else
  std::cout << "ERROR: calling mkl_offload_dtrmm without build for Intel GPU support!" << std::endl;
  std::cout << "ERROR: You should never see this message" << std::endl;
#endif
}


#ifdef WANT_SINGLE_PRECISION_REAL
void mkl_offload_sgemm_c(char transa, char transb, int m, int n, int k, float alpha, float *a, int lda, float *b, int ldb, float beta, float *c, int ldc) {

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

  std::cout << "Transa=" << transa << std::endl;
  std::cout << "Transb=" << transb << std::endl;

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


void mkl_offload_sgemv_c(char trans, int m, int n, float alpha, float *a, int lda, float *x, int incx, float beta, float *y, int incy) {

#ifdef WITH_INTEL_GPU_VERSION
  std::cout << "In mkl_offload_sgemv" << std::endl;

  // at a later time the device should be set differently
  int dnum = 0;

  int sizea, sizex, sizey;
  std::cout << "m=" << m << "lda=" << lda << std::endl;
  std::cout << "n=" << n << std::endl;
  //std::cout << "sizeX=" << sizeX << std::endl;
  //std::cout << "sizeY=" << sizeY << std::endl;
  std::cout << "alpha=" << alpha << std::endl;
  std::cout << "beta=" << beta << std::endl;


  std::cout << "Trans=" << trans << std::endl;

  sizea = lda * n;
  sizex = n;
  sizey = m;


  #pragma omp target data map(to : a [0:sizea], x [0:sizex]) map(tofrom : y [0:sizey]) device(dnum)
  {
  #pragma omp target variant dispatch device(dnum) use_device_ptr(a, x, y)
  sgemv(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
  std::cout << "leaving mkl_offload_sgemv" << std::endl;
#else
  std::cout << "ERROR: calling mkl_offload_sgemv without build for Intel GPU support!" << std::endl;
  std::cout << "ERROR: You should never see this message" << std::endl;
#endif
}

void mkl_offload_strmm_c(char side, char uplo, char trans, char diag, int m, int n, float alpha, float *a, int lda, float *b, int ldb) {

#ifdef WITH_INTEL_GPU_VERSION
  std::cout << "In mkl_offload_strmm" << std::endl;

  // at a later time the device should be set differently
  int dnum = 0;

  int sizea, sizeb;
  std::cout << "m=" << m << "lda=" << lda << std::endl;
  std::cout << "n=" << n << std::endl;
  //std::cout << "sizeX=" << sizeX << std::endl;
  //std::cout << "sizeY=" << sizeY << std::endl;
  std::cout << "alpha=" << alpha << std::endl;

  if (side == 'L' || side == 'l') {
    std::cout << "Setting a to case L" << std::endl;
    sizea = lda * m;
  }
  if (side == 'R' || side == 'r') {
    std::cout << "Setting a to case R" << std::endl;
    sizea = lda * n;
  }
  sizeb = ldb * n;


  #pragma omp target data map(to : a [0:sizea]) map(tofrom : b [0:sizeb]) device(dnum)
  {
  #pragma omp target variant dispatch device(dnum) use_device_ptr(a, b)
  strmm(&side, &uplo, &trans, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
  }
  std::cout << "leaving mkl_offload_strmm" << std::endl;
#else
  std::cout << "ERROR: calling mkl_offload_strmm without build for Intel GPU support!" << std::endl;
  std::cout << "ERROR: You should never see this message" << std::endl;
#endif
}

#endif /* WANT_SINGLE_PRECISION_REAL */

void mkl_offload_zgemm_c(char transa, char transb, int m, int n, int k, MKL_Complex16 alpha, MKL_Complex16 *a, int lda, MKL_Complex16 *b, int ldb, MKL_Complex16 beta, MKL_Complex16 *c, int ldc) {

#ifdef WITH_INTEL_GPU_VERSION
  std::cout << "In mkl_offload_zgemm" << std::endl;

  // at a later time the device should be set differently
  int dnum = 0;

  int sizea, sizeb, sizec;
  std::cout << "m=" << m << "lda=" << lda << "ldc=" << ldc << std::endl;
  std::cout << "n=" << n << "ldb=" << ldb << std::endl;
  std::cout << "k=" << k << std::endl;

  std::cout << "Transa=" << transa << std::endl;
  std::cout << "Transb=" << transb << std::endl;

  sizea = lda * k;
  sizeb = ldb * n;
  sizec = ldc * n;


  #pragma omp target data map(to : a [0:sizea], b [0:sizeb]) map(tofrom : c [0:sizec]) device(dnum)
  {
  #pragma omp target variant dispatch device(dnum) use_device_ptr(a, b, c)
  zgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
  }
  std::cout << "leaving mkl_offload_zgemm" << std::endl;
#else
  std::cout << "ERROR: calling mkl_offload_zgemm without build for Intel GPU support!" << std::endl;
  std::cout << "ERROR: You should never see this message" << std::endl;
#endif
}

void mkl_offload_zgemv_c(char trans, int m, int n, MKL_Complex16 alpha, MKL_Complex16 *a, int lda, MKL_Complex16 *x, int incx, MKL_Complex16 beta, MKL_Complex16 *y, int incy) {

#ifdef WITH_INTEL_GPU_VERSION
  std::cout << "In mkl_offload_zgemv" << std::endl;

  // at a later time the device should be set differently
  int dnum = 0;

  int sizea, sizex, sizey;
  std::cout << "m=" << m << "lda=" << lda << std::endl;
  std::cout << "n=" << n << std::endl;
  //std::cout << "sizeX=" << sizeX << std::endl;
  //std::cout << "sizeY=" << sizeY << std::endl;

  std::cout << "Trans=" << trans << std::endl;

  sizea = lda * n;
  sizex = n;
  sizey = m;


  #pragma omp target data map(to : a [0:sizea], x [0:sizex]) map(tofrom : y [0:sizey]) device(dnum)
  {
  #pragma omp target variant dispatch device(dnum) use_device_ptr(a, x, y)
  zgemv(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
  std::cout << "leaving mkl_offload_zgemv" << std::endl;
#else
  std::cout << "ERROR: calling mkl_offload_zgemv without build for Intel GPU support!" << std::endl;
  std::cout << "ERROR: You should never see this message" << std::endl;
#endif
}


void mkl_offload_ztrmm_c(char side, char uplo, char trans, char diag, int m, int n, MKL_Complex16 alpha, MKL_Complex16 *a, int lda, MKL_Complex16 *b, int ldb) {

#ifdef WITH_INTEL_GPU_VERSION
  std::cout << "In mkl_offload_ztrmm" << std::endl;

  // at a later time the device should be set differently
  int dnum = 0;

  int sizea, sizeb;
  std::cout << "m=" << m << "lda=" << lda << std::endl;
  std::cout << "n=" << n << std::endl;
  //std::cout << "sizeX=" << sizeX << std::endl;
  //std::cout << "sizeY=" << sizeY << std::endl;

  if (side == 'L' || side == 'l') {
    std::cout << "Setting a to case L" << std::endl;
    sizea = lda * m;
  }
  if (side == 'R' || side == 'r') {
    std::cout << "Setting a to case R" << std::endl;
    sizea = lda * n;
  }
  sizeb = ldb * n;


  #pragma omp target data map(to : a [0:sizea]) map(tofrom : b [0:sizeb]) device(dnum)
  {
  #pragma omp target variant dispatch device(dnum) use_device_ptr(a, b)
  ztrmm(&side, &uplo, &trans, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
  }
  std::cout << "leaving mkl_offload_ztrmm" << std::endl;
#else
  std::cout << "ERROR: calling mkl_offload_ztrmm without build for Intel GPU support!" << std::endl;
  std::cout << "ERROR: You should never see this message" << std::endl;
#endif
}

#ifdef WANT_SINGLE_PRECISION_COMPLEX
void mkl_offload_cgemm_c(char transa, char transb, int m, int n, int k, MKL_Complex8 alpha, MKL_Complex8 *a, int lda, MKL_Complex8 *b, int ldb, MKL_Complex8 beta, MKL_Complex8 *c, int ldc) {

#ifdef WITH_INTEL_GPU_VERSION
  std::cout << "In mkl_offload_cgemm" << std::endl;

  // at a later time the device should be set differently
  int dnum = 0;

  int sizea, sizeb, sizec;
  std::cout << "m=" << m << "lda=" << lda << "ldc=" << ldc << std::endl;
  std::cout << "n=" << n << "ldb=" << ldb << std::endl;
  std::cout << "k=" << k << std::endl;

  std::cout << "Transa=" << transa << std::endl;
  std::cout << "Transb=" << transb << std::endl;

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

void mkl_offload_cgemv_c(char trans, int m, int n, MKL_Complex8 alpha, MKL_Complex8 *a, int lda, MKL_Complex8 *x, int incx, MKL_Complex8 beta, MKL_Complex8 *y, int incy) {

#ifdef WITH_INTEL_GPU_VERSION
  std::cout << "In mkl_offload_cgemv" << std::endl;

  // at a later time the device should be set differently
  int dnum = 0;

  int sizea, sizex, sizey;
  std::cout << "m=" << m << "lda=" << lda << std::endl;
  std::cout << "n=" << n << std::endl;
  //std::cout << "sizeX=" << sizeX << std::endl;
  //std::cout << "sizeY=" << sizeY << std::endl;


  std::cout << "Trans=" << trans << std::endl;

  sizea = lda * n;
  sizex = n;
  sizey = m;


  #pragma omp target data map(to : a [0:sizea], x [0:sizex]) map(tofrom : y [0:sizey]) device(dnum)
  {
  #pragma omp target variant dispatch device(dnum) use_device_ptr(a, x, y)
  cgemv(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
  std::cout << "leaving mkl_offload_cgemv" << std::endl;
#else
  std::cout << "ERROR: calling mkl_offload_cgemv without build for Intel GPU support!" << std::endl;
  std::cout << "ERROR: You should never see this message" << std::endl;
#endif
}


void mkl_offload_ctrmm_c(char side, char uplo, char trans, char diag, int m, int n, MKL_Complex8 alpha, MKL_Complex8 *a, int lda, MKL_Complex8 *b, int ldb) {

#ifdef WITH_INTEL_GPU_VERSION
  std::cout << "In mkl_offload_ctrmm" << std::endl;

  // at a later time the device should be set differently
  int dnum = 0;

  int sizea, sizeb;
  std::cout << "m=" << m << "lda=" << lda << std::endl;
  std::cout << "n=" << n << std::endl;
  //std::cout << "sizeX=" << sizeX << std::endl;
  //std::cout << "sizeY=" << sizeY << std::endl;

  if (side == 'L' || side == 'l') {
    std::cout << "Setting a to case L" << std::endl;
    sizea = lda * m;
  }
  if (side == 'R' || side == 'r') {
    std::cout << "Setting a to case R" << std::endl;
    sizea = lda * n;
  }
  sizeb = ldb * n;


  #pragma omp target data map(to : a [0:sizea]) map(tofrom : b [0:sizeb]) device(dnum)
  {
  #pragma omp target variant dispatch device(dnum) use_device_ptr(a, b)
  ctrmm(&side, &uplo, &trans, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
  }
  std::cout << "leaving mkl_offload_ctrmm" << std::endl;
#else
  std::cout << "ERROR: calling mkl_offload_ctrmm without build for Intel GPU support!" << std::endl;
  std::cout << "ERROR: You should never see this message" << std::endl;
#endif
}

#endif /*  WANT_SINGLE_PRECISION_COMPLEX */

}
