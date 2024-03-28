//
//    Copyright 2022, P. Karpov, MPCDF
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

#include <stdio.h>
#include <math.h>

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <complex.h>

#include "config-f90.h"

#include "./gpu_vendor_agnostic_layer.h"

void set_gpu_parameters(int* gpuMemcpyHostToDevice, int* gpuMemcpyDeviceToHost){
#ifdef WITH_NVIDIA_GPU_VERSION
  *gpuMemcpyHostToDevice = cudaMemcpyHostToDeviceFromC();
  *gpuMemcpyDeviceToHost = cudaMemcpyDeviceToHostFromC();
#endif 
#ifdef WITH_AMD_GPU_VERSION
  *gpuMemcpyHostToDevice = hipMemcpyHostToDeviceFromC();
  *gpuMemcpyDeviceToHost = hipMemcpyDeviceToHostFromC();
#endif
#ifdef WITH_SYCL_GPU_VERSION
  *gpuMemcpyHostToDevice = syclMemcpyHostToDeviceFromC();
  *gpuMemcpyDeviceToHost = syclMemcpyDeviceToHostFromC();
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#error "openmp_offload missing"
#endif
}

int gpuGetDeviceCount(int *count){
#ifdef WITH_NVIDIA_GPU_VERSION
  return cudaGetDeviceCountFromC(count);
#endif 
#ifdef WITH_AMD_GPU_VERSION
  return hipGetDeviceCountFromC(count);
#endif
#ifdef WITH_SYCL_GPU_VERSION
  return syclGetDeviceCountFromC(count);
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#error "openmp_offload missing"
#endif
  return -1;
}

int gpuSetDevice(int n){
#ifdef WITH_NVIDIA_GPU_VERSION
  return cudaSetDeviceFromC(n);
#endif   
#ifdef WITH_AMD_GPU_VERSION
  return hipSetDeviceFromC(n);
#endif
#ifdef WITH_SYCL_GPU_VERSION
  return syclSetDeviceFromC(n);
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#error "openmp_offload missing"
#endif
  return -1;
}

int gpuMalloc(intptr_t *a, size_t width_height) {
#ifdef WITH_NVIDIA_GPU_VERSION
  return cudaMallocFromC(a, width_height);
#endif   
#ifdef WITH_AMD_GPU_VERSION
  return hipMallocFromC(a, width_height);
#endif
#ifdef WITH_SYCL_GPU_VERSION
  return syclMallocFromC(a, width_height);
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#error "openmp_offload missing"
#endif
  return -1;
}

int gpuFree(intptr_t *a) {
#ifdef WITH_NVIDIA_GPU_VERSION
  return cudaFreeFromC(a);
#endif   
#ifdef WITH_AMD_GPU_VERSION
  return hipFreeFromC(a);
#endif
#ifdef WITH_SYCL_GPU_VERSION
  return syclFreeFromC(a);
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#error "openmp_offload missing"
#endif
  return -1;
}

int gpuMemcpy(intptr_t *dest, intptr_t *src, size_t count, int dir){
#ifdef WITH_NVIDIA_GPU_VERSION
  return cudaMemcpyFromC(dest, src, count, dir);
#endif  
#ifdef WITH_AMD_GPU_VERSION
  return hipMemcpyFromC(dest, src, count, dir);
#endif
#ifdef WITH_SYCL_GPU_VERSION
  return syclMemcpyFromC(dest, src, count, dir);
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#error "openmp_offload missing"
#endif
  return -1;
}

int gpuDeviceSynchronize(){
#ifdef WITH_NVIDIA_GPU_VERSION
  return cudaDeviceSynchronizeFromC();
#endif  
#ifdef WITH_AMD_GPU_VERSION
  return hipDeviceSynchronizeFromC();
#endif
#ifdef WITH_SYCL_GPU_VERSION
  printf("ELPA warning: gpuDeviceSynchronize() not implemented in SYCL\n");
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
  printf("ELPA warning: gpuDeviceSynchronize() not implemented in OpenMP offload\n");
#endif
  return -1;
}

int gpuMemset(intptr_t *a, int value, size_t count){
#ifdef WITH_NVIDIA_GPU_VERSION
  return cudaMemsetFromC(a, value, count);
#endif
#ifdef WITH_AMD_GPU_VERSION
  return hipMemsetFromC(a, value, count);
#endif
#ifdef WITH_SYCL_GPU_VERSION
  return syclMemsetFromC(a, value, count);
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#error "openmp_offload missing"
#endif
  return -1;
}

void gpublasDgemm(intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                              double alpha, const double *A, int lda,
                              const double *B, int ldb, double beta,
                              double *C, int ldc){
#ifdef WITH_NVIDIA_GPU_VERSION
  cublasDgemm_elpa_wrapper_intptr_handle(gpuHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
#ifdef WITH_AMD_GPU_VERSION
  rocblasDgemm_elpa_wrapper_intptr_handle(gpuHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
#ifdef WITH_SYCL_GPU_VERSION
  syclblasDgemm_elpa_wrapper(gpuHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); // for SYCL, handle is not needed
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
  mklOpenmpOffloadDgemmFromC(gpuHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); // for OpenMP, offload handle is not needed
#endif
}

void gpublasSgemm(intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                              float alpha, const float *A, int lda,
                              const float *B, int ldb, float beta,
                              float *C, int ldc) {
#ifdef WITH_NVIDIA_GPU_VERSION
  cublasSgemm_elpa_wrapper_intptr_handle(gpuHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
#ifdef WITH_AMD_GPU_VERSION
  rocblasSgemm_elpa_wrapper_intptr_handle(gpuHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
#ifdef WITH_SYCL_GPU_VERSION
  syclblasSgemm_elpa_wrapper(gpuHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
  mklOpenmpOffloadSgemmFromC(gpuHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

void gpublasZgemm(intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                              double _Complex alpha, const double _Complex *A, int lda,
                              const double _Complex *B, int ldb, double _Complex beta,
                              double _Complex *C, int ldc) {
#ifdef WITH_NVIDIA_GPU_VERSION
  cublasZgemm_elpa_wrapper_intptr_handle(gpuHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
#ifdef WITH_AMD_GPU_VERSION
  rocblasZgemm_elpa_wrapper_intptr_handle(gpuHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
#ifdef WITH_SYCL_GPU_VERSION
  syclblasZgemm_elpa_wrapper(gpuHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
  mklOpenmpOffloadZgemmFromC(gpuHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

void gpublasCgemm(intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                               float _Complex alpha, const float _Complex *A, int lda,
                               const float _Complex *B, int ldb, float _Complex beta,
                               float _Complex *C, int ldc) {
#ifdef WITH_NVIDIA_GPU_VERSION
  cublasCgemm_elpa_wrapper_intptr_handle(gpuHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
#ifdef WITH_AMD_GPU_VERSION
  rocblasCgemm_elpa_wrapper_intptr_handle(gpuHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
#ifdef WITH_SYCL_GPU_VERSION
  syclblasCgemm_elpa_wrapper(gpuHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
  mklOpenmpOffloadZgemmFromC(gpuHandle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}