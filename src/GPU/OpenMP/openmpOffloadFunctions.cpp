/*
//    Copyright 2022, A. Marek
//
//    This file is part of ELPA.
//
//    The ELPA library was originally created by the ELPA consortium,
//    consisting of the following organizations:
//
//    - Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
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
// This file was written by A. Marek, MPCDF (2022)
// it is based on a prototype implementation developed for MPCDF 
// by A. Poeppl, Intel (2022) 
*/
#include "config-f90.h"

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION

#include <mkl.h>
#include <mkl_omp_offload.h>

#include <omp.h>

#include <complex>
#include <iostream>
#include <cstdint>

int openmpOffloadChosenGpu;
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

  int openmpOffloadGetDeviceCountFromC() {
    return omp_get_num_devices();
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

  int openmpOffloadblasDestroyFromC(intptr_t* handle){
    //stub function
    return 1;
  }
  
  int openmpOffloadsolverCreateFromC(intptr_t* handle){
    //stub function
    return 1;
  }

  int openmpOffloadsolverDestroyFromC(intptr_t* handle){
    //stub function
    return 1;
  }
  
  int openmpOffloadMallocFromC(intptr_t *a, size_t elems) {
      int device = openmpOffloadChosenGpu;
      *a = (intptr_t) omp_target_alloc(elems, device);
      if (*a) {
#ifdef OPENMP_OFFLOAD_DEBUG
          std::cout << "Allocated " << elems << "B starting at address " << *a << std::endl;
#endif
          return 1;
      } else {
#ifdef OPENMP_OFFLOAD_DEBUG
          std::cout << "Allocation failed!" << std::endl;
#endif
          return 0;
      }
  }

  int openmpOffloadFreeFromC(intptr_t *a) {
      int device = openmpOffloadChosenGpu;
      void * ptr = reinterpret_cast<void *>(*a);
      omp_target_free(ptr, device);
      return 1;
  }

  int openmpOffloadMemcpyFromC(void *dst, void *src, size_t size, int direction) {
      // FIXME: Do not use the default device, but some other device that has been specified.
      // TODO:  Find out how this is done in ELPA.
      int hostDevice = omp_get_initial_device();
      //int gpuDevice = omp_get_default_device();
      int gpuDevice = openmpOffloadChosenGpu;
      int dstDevice;
      int srcDevice;
#ifdef OPENMP_OFFLOAD_DEBUG
      printf("Direction %d %d\n",direction,openmpOffloadMemcpyHostToDevice);
#endif
      if (direction == openmpOffloadMemcpyDeviceToDevice) {
          dstDevice = gpuDevice;
          srcDevice = gpuDevice;
      } else if (direction == openmpOffloadMemcpyDeviceToHost) {
          dstDevice = hostDevice;
          srcDevice = gpuDevice;
      } else if (direction == openmpOffloadMemcpyHostToDevice) {
          dstDevice = gpuDevice;
          srcDevice = hostDevice;
      } else {
          std::cerr << "Direction of transfer for memcpy unknown" << std::endl;
          return 0;
      }
      int retVal = omp_target_memcpy(dst, src, size, 0, 0, dstDevice, srcDevice);
#ifdef OPENMP_OFFLOAD_DEBUG
      printf("Return val o fmemcpy %d\n",retVal);
#endif
      if (retVal != 0){
	return 0;
      } else {
//#ifdef OPENMP_OFFLOAD_DEBUG
        std::cout << "Copied " << size << "B successfully from " << reinterpret_cast<intptr_t>(src) << " to " << reinterpret_cast<intptr_t>(dst) << "." << std::endl;
//#endif
	return 1;
      }
  }

  int openmpOffloadMemsetFromC(intptr_t *mem, int val, intptr_t size) {
    //std::cout << "Memsetting 0 " << size << "Bytes" << std::endl;
    //std::cout << "Memsetting 1 " << size << "B starting at address " << *mem <<  std::endl;
    char *mem_bytes = reinterpret_cast<char *>(mem);
    char tVal = static_cast<char>(val);
    //std::cout << "Memsetting 2" << size << "B starting at address "  << *mem_bytes << std::endl;
    #pragma omp target teams loop is_device_ptr(mem_bytes) device(openmpOffloadChosenGpu)
    //#pragma omp target device(openmpOffloadChosenGpu) is_device_ptr(mem_bytes) teams distribute parallel for
    for (intptr_t i = 0; i < size; i++) {
      mem_bytes[i] = tVal;
    }
    return 1;
  }

  void mklOpenmpOffloadDgemmFromC(intptr_t *handle, char cta, char ctb, int m, int n, int k, double alpha, void *a, int lda, void *b, int ldb, double beta, void *c, int ldc) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      dgemm(&cta, &ctb, &m, &n, &k, &alpha, reinterpret_cast<double *>(a), &lda, reinterpret_cast<double *>(b), &ldb, &beta, reinterpret_cast<double *>(c), &ldc);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "DGEMM" << std::endl;
#endif
  }  

  void mklOpenmpOffloadSgemmFromC(intptr_t *handle, char cta, char ctb, int m, int n, int k, float alpha, void *a, int lda, void *b, int ldb, float beta, void *c, int ldc) {
      //handle not needed
#ifdef OPENMP_OFFLOAD_DEBUG
	  std::cout << "Calling sgemm" << std::endl;
	  std::cout << "m=" << m << std::endl;
	  std::cout << "n=" << n << std::endl;
	  std::cout << "k=" << k << std::endl;
#endif
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      sgemm(&cta, &ctb, &m, &n, &k, &alpha, reinterpret_cast<float *>(a), &lda, reinterpret_cast<float *>(b), &ldb, &beta, reinterpret_cast<float *>(c), &ldc);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "SGEMM" << std::endl;
#endif
  }

  void mklOpenmpOffloadZgemmFromC(intptr_t *handle, char cta, char ctb, int m, int n, int k, MKL_Complex16 alpha, void *a, int lda, void *b, int ldb, MKL_Complex16 beta, void *c, int ldc) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      zgemm(&cta, &ctb, &m, &n, &k, &alpha, reinterpret_cast<MKL_Complex16 *>(a), &lda, reinterpret_cast<MKL_Complex16 *>(b), &ldb, &beta, reinterpret_cast<MKL_Complex16 *>(c), &ldc);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "ZGEMM" << std::endl;
#endif
  }

  void mklOpenmpOffloadCgemmFromC(intptr_t *handle, char cta, char ctb, int m, int n, int k, MKL_Complex8 alpha, void *a, int lda, void *b, int ldb, MKL_Complex8 beta, void *c, int ldc) {
      //handle not needed
    #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
    cgemm(&cta, &ctb, &m, &n, &k, &alpha, reinterpret_cast<MKL_Complex8 *>(a), &lda, reinterpret_cast<MKL_Complex8 *>(b), &ldb, &beta, reinterpret_cast<MKL_Complex8 *>(c), &ldc);
#ifdef OPENMP_OFFLOAD_DEBUG
    std::cout << "CGEMM" << std::endl;
#endif
  }

  void mklOpenmpOffloadDtrtriFromC(intptr_t *handle, char uplo, char diag, int n, void *a, int lda, int info) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      dtrtri(&uplo, &diag, &n, reinterpret_cast<double *>(a), &lda, &info);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "DTRTRI" << std::endl;
#endif
  }  

  void mklOpenmpOffloadStrtriFromC(intptr_t *handle, char uplo, char diag, int n, void *a, int lda, int info) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      strtri(&uplo, &diag, &n, reinterpret_cast<float *>(a), &lda, &info);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "STRTRI" << std::endl;
#endif
  }  

  void mklOpenmpOffloadZtrtriFromC(intptr_t *handle, char uplo, char diag, int n, void *a, int lda, int info) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      ztrtri(&uplo, &diag, &n, reinterpret_cast<MKL_Complex16 *>(a), &lda, &info);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "ZTRTRI" << std::endl;
#endif
  }  

  void mklOpenmpOffloadCtrtriFromC(intptr_t *handle, char uplo, char diag, int n, void *a, int lda, int info) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      ctrtri(&uplo, &diag, &n, reinterpret_cast<MKL_Complex8 *>(a), &lda, &info);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "CTRTRI" << std::endl;
#endif
  }  

  void mklOpenmpOffloadDpotrfFromC(intptr_t *handle, char uplo, int n, void *a, int lda, int info) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      dpotrf(&uplo, &n, reinterpret_cast<double *>(a), &lda, &info);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "DPOTRF" << std::endl;
#endif
  }  

  void mklOpenmpOffloadSpotrfFromC(intptr_t *handle, char uplo, int n, void *a, int lda, int info) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      spotrf(&uplo, &n, reinterpret_cast<float *>(a), &lda, &info);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "SPOTRF" << std::endl;
#endif
  }  

  void mklOpenmpOffloadZpotrfFromC(intptr_t *handle, char uplo, int n, void *a, int lda, int info) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      zpotrf(&uplo, &n, reinterpret_cast<MKL_Complex16 *>(a), &lda, &info);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "ZPOTRF" << std::endl;
#endif
  }  

  void mklOpenmpOffloadCpotrfFromC(intptr_t *handle, char uplo, int n, void *a, int lda, int info) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      cpotrf(&uplo, &n, reinterpret_cast<MKL_Complex8 *>(a), &lda, &info);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "CPOTRF" << std::endl;
#endif
  }

  void mklOpenmpOffloadDcopyFromC(intptr_t *handle, int n, void *x, int incx, void *y, int incy) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      dcopy(&n, reinterpret_cast<double *>(x), &incx, reinterpret_cast<double *>(y), &incy);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "DCOPY" << std::endl;
#endif
  }  

  void mklOpenmpOffloadScopyFromC(intptr_t *handle, int n, void *x, int incx, void *y, int incy) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      scopy(&n, reinterpret_cast<float *>(x), &incx, reinterpret_cast<float *>(y), &incy);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "SCOPY" << std::endl;
#endif
  }  

  void mklOpenmpOffloadZcopyFromC(intptr_t *handle, int n, void *x, int incx, void *y, int incy) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      zcopy(&n, reinterpret_cast<MKL_Complex16 *>(x), &incx, reinterpret_cast<MKL_Complex16 *>(y), &incy);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "ZCOPY" << std::endl;
#endif
  }  

  void mklOpenmpOffloadCcopyFromC(intptr_t *handle, int n, void *x, int incx, void *y, int incy) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      ccopy(&n, reinterpret_cast<MKL_Complex8 *>(x), &incx, reinterpret_cast<MKL_Complex8 *>(y), &incy);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "CCOPY" << std::endl;
#endif
  }  


  void mklOpenmpOffloadDtrmmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int m, int n, double alpha, void *a, int lda, void *b, int ldb) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      dtrmm(&side, &uplo, &trans, &diag, &m, &n, &alpha, reinterpret_cast<double *>(a), &lda, reinterpret_cast<double *>(b), &ldb);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "DTRMM" << std::endl;
#endif
  }  

  void mklOpenmpOffloadStrmmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int m, int n, float alpha, void *a, int lda, void *b, int ldb) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      strmm(&side, &uplo, &trans, &diag, &m, &n, &alpha, reinterpret_cast<float *>(a), &lda, reinterpret_cast<float *>(b), &ldb);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "STRMM" << std::endl;
#endif
  }  

  void mklOpenmpOffloadZtrmmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int m, int n, MKL_Complex16 alpha, void *a, int lda, void *b, int ldb) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      ztrmm(&side, &uplo, &trans, &diag, &m, &n, &alpha, reinterpret_cast<MKL_Complex16 *>(a), &lda, reinterpret_cast<MKL_Complex16 *>(b), &ldb);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "ZTRMM" << std::endl;
#endif
  }  

  void mklOpenmpOffloadCtrmmFromC(intptr_t *handle, char side, char uplo, char trans,  char diag, int m, int n, MKL_Complex8 alpha, void *a, int lda, void *b, int ldb) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      ctrmm(&side, &uplo, &trans, &diag, &m, &n, &alpha, reinterpret_cast<MKL_Complex8 *>(a), &lda, reinterpret_cast<MKL_Complex8 *>(b), &ldb);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "CTRMM" << std::endl;
#endif
  }  


  void mklOpenmpOffloadDtrsmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int m, int n, double alpha, void *a, int lda, void *b, int ldb) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      dtrsm(&side, &uplo, &trans, &diag, &m, &n, &alpha, reinterpret_cast<double *>(a), &lda, reinterpret_cast<double *>(b), &ldb);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "DTRSM" << std::endl;
#endif
  }  

  void mklOpenmpOffloadStrsmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int m, int n, float alpha, void *a, int lda, void *b, int ldb) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      strsm(&side, &uplo, &trans, &diag, &m, &n, &alpha, reinterpret_cast<float *>(a), &lda, reinterpret_cast<float *>(b), &ldb);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "STRSM" << std::endl;
#endif
  }  

  void mklOpenmpOffloadZtrsmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int m, int n, MKL_Complex16 alpha, void *a, int lda, void *b, int ldb) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      ztrsm(&side, &uplo, &trans, &diag, &m, &n, &alpha, reinterpret_cast<MKL_Complex16 *>(a), &lda, reinterpret_cast<MKL_Complex16 *>(b), &ldb);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "ZTRSM" << std::endl;
#endif
  }  

  void mklOpenmpOffloadCtrsmFromC(intptr_t *handle, char side, char uplo, char trans, char diag, int m, int n, MKL_Complex8 alpha, void *a, int lda, void *b, int ldb) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      ctrsm(&side, &uplo, &trans, &diag, &m, &n, &alpha, reinterpret_cast<MKL_Complex8 *>(a), &lda, reinterpret_cast<MKL_Complex8 *>(b), &ldb);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "CTRSM" << std::endl;
#endif
  }

  void mklOpenmpOffloadDgemvFromC(intptr_t *handle, char cta, int m, int n, double alpha, void *a, int lda, void *x, int incx, double beta, void *y, int incy) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      dgemv(&cta, &m, &n, &alpha, reinterpret_cast<double *>(a), &lda, reinterpret_cast<double *>(x), &incx, &beta, reinterpret_cast<double *>(y), &incy);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "DGEMV" << std::endl;
#endif
  }  

  void mklOpenmpOffloadSgemvFromC(intptr_t *handle, char cta, int m, int n, float alpha, void *a, int lda, void *x, int incx, float beta, void *y, int incy) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      sgemv(&cta, &m, &n, &alpha, reinterpret_cast<float *>(a), &lda, reinterpret_cast<float *>(x), &incx, &beta, reinterpret_cast<float *>(y), &incy);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "SGEMV" << std::endl;
#endif
  }  

  void mklOpenmpOffloadZgemvFromC(intptr_t *handle, char cta, int m, int n, MKL_Complex16 alpha, void *a, int lda, void *x, int incx, MKL_Complex16 beta, void *y, int incy) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      zgemv(&cta, &m, &n, &alpha, reinterpret_cast<MKL_Complex16 *>(a), &lda, reinterpret_cast<MKL_Complex16 *>(x), &incx, &beta, reinterpret_cast<MKL_Complex16 *>(y), &incy);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "ZGEMV" << std::endl;
#endif
  }  

  void mklOpenmpOffloadCgemvFromC(intptr_t *handle, char cta, int m, int n, MKL_Complex8 alpha, void *a, int lda, void *x, int incx, MKL_Complex8 beta, void *y, int incy) {
      //handle not needed
      #pragma omp target variant dispatch device(openmpOffloadChosenGpu)
      cgemv(&cta, &m, &n, &alpha, reinterpret_cast<MKL_Complex8 *>(a), &lda, reinterpret_cast<MKL_Complex8 *>(x), &incx, &beta, reinterpret_cast<MKL_Complex8 *>(y), &incy);
#ifdef OPENMP_OFFLOAD_DEBUG
      std::cout << "CGEMV" << std::endl;
#endif
  }  



}
#endif /* WITH_OPENMP_OFFLOAD_GPU_VERSION */
