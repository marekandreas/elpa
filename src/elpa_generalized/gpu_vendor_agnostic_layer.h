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
#pragma once
#include <stdint.h> // for intptr_t

#ifdef __cplusplus
extern "C" {
#endif

#ifdef WITH_NVIDIA_GPU_VERSION
int cudaMemcpyHostToDeviceFromC();
int cudaMemcpyDeviceToHostFromC();
int cudaGetDeviceCountFromC(int *count);
int cudaSetDeviceFromC(int n);
int cudaMallocFromC(intptr_t *a, size_t width_height);
int cudaFreeFromC(intptr_t *a);
int cudaMemcpyFromC(intptr_t *dest, intptr_t *src, size_t count, int dir);
int cudaDeviceSynchronizeFromC();
int cudaMemsetFromC(intptr_t *a, int value, size_t count);
void cublasDgemm_elpa_wrapper_intptr_handle(intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                              double alpha, const double *A, int lda,
                              const double *B, int ldb, double beta,
                              double *C, int ldc);
void cublasSgemm_elpa_wrapper_intptr_handle(intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                              float alpha, const float *A, int lda,
                              const float *B, int ldb, float beta,
                              float *C, int ldc);                        
void cublasZgemm_elpa_wrapper_intptr_handle(intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                              double _Complex alpha, const double _Complex *A, int lda,
                              const double _Complex *B, int ldb, double _Complex beta,
                              double _Complex *C, int ldc);
void cublasCgemm_elpa_wrapper_intptr_handle(intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                               float _Complex alpha, const float _Complex *A, int lda,
                               const float _Complex *B, int ldb, float _Complex beta,
                               float _Complex *C, int ldc);
#endif
#ifdef WITH_AMD_GPU_VERSION
int hipMemcpyHostToDeviceFromC();
int hipMemcpyDeviceToHostFromC();
int hipGetDeviceCountFromC(int *count);
int hipSetDeviceFromC(int n);
int hipMallocFromC(intptr_t *a, size_t width_height);
int hipFreeFromC(intptr_t *a);
int hipMemcpyFromC (intptr_t *dest, intptr_t *src, size_t count, int dir);
int hipDeviceSynchronizeFromC();
int hipMemsetFromC(intptr_t *a, int value, size_t count);
void rocblasDgemm_elpa_wrapper_intptr_handle(intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                              double alpha, const double *A, int lda,
                              const double *B, int ldb, double beta,
                              double *C, int ldc);
void rocblasSgemm_elpa_wrapper_intptr_handle(intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                              float alpha, const float *A, int lda,
                              const float *B, int ldb, float beta,
                              float *C, int ldc);
void rocblasZgemm_elpa_wrapper_intptr_handle(intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                              double _Complex alpha, const double _Complex *A, int lda,
                              const double _Complex *B, int ldb, double _Complex beta,
                              double _Complex *C, int ldc);
void rocblasCgemm_elpa_wrapper_intptr_handle(intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                               float _Complex alpha, const float _Complex *A, int lda,
                               const float _Complex *B, int ldb, float _Complex beta,
                               float _Complex *C, int ldc);
#endif
#ifdef WITH_SYCL_GPU_VERSION
int syclMemcpyHostToDeviceFromC();
int syclMemcpyDeviceToHostFromC();
int syclGetDeviceCountFromC(int *count);
int syclSetDeviceFromC(int n);
int syclMallocFromC(intptr_t *a, size_t width_height);
int syclFreeFromC(intptr_t *a);
int syclMemcpyFromC(intptr_t *dest, intptr_t *src, size_t count, int dir);
int syclDeviceSynchronizeFromC();
int syclMemsetFromC(intptr_t *a, int value, size_t count);
void syclblasDgemm_elpa_wrapper(intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                              double alpha, const double *A, int lda,
                              const double *B, int ldb, double beta,
                              double *C, int ldc);
void syclblasSgemm_elpa_wrapper(intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                              float alpha, const float *A, int lda,
                              const float *B, int ldb, float beta,
                              float *C, int ldc);
void syclblasZgemm_elpa_wrapper(intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                              double _Complex alpha, const double _Complex *A, int lda,
                              const double _Complex *B, int ldb, double _Complex beta,
                              double _Complex *C, int ldc);
void syclblasCgemm_elpa_wrapper(intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                               float _Complex alpha, const float _Complex *A, int lda,
                               const float _Complex *B, int ldb, float _Complex beta,
                               float _Complex *C, int ldc);
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#error "openmp_offload missing"
#endif


void set_gpu_parameters(int* gpuMemcpyHostToDevice, int* gpuMemcpyDeviceToHost);

int gpuGetDeviceCount(int *count);
int gpuSetDevice(int n); 
   
int gpuMalloc(intptr_t *a, size_t width_height);

int gpuFree(intptr_t *a);
   
int gpuMemcpy(intptr_t *dest, intptr_t *src, size_t count, int dir);

int syclGetCpuCount(int numberOfDevices);

int gpuDeviceSynchronize();

int gpuMemset(intptr_t *a, int value, size_t count);

void gpublasDgemm(intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                              double alpha, const double *A, int lda,
                              const double *B, int ldb, double beta,
                              double *C, int ldc);

void gpublasSgemm (intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                              float alpha, const float *A, int lda,
                              const float *B, int ldb, float beta,
                              float *C, int ldc);

void gpublasZgemm (intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                              double _Complex alpha, const double _Complex *A, int lda,
                              const double _Complex *B, int ldb, double _Complex beta,
                              double _Complex *C, int ldc);

void gpublasCgemm (intptr_t* gpuHandle, char transa, char transb, int m, int n, int k,
                               float _Complex alpha, const float _Complex *A, int lda,
                               const float _Complex *B, int ldb, float _Complex beta,
                               float _Complex *C, int ldc);

#ifdef __cplusplus
}    
#endif