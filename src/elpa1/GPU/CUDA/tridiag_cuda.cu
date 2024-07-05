//    Copyright 2023, P. Karpov
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
//    This file was written by P. Karpov, MPCDF

#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <complex.h>
#include <cuComplex.h>
#include <stdint.h>
#include <stdbool.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include "config-f90.h"

#include "../../../GPU/common_device_functions.h"

#define MAX_THREADS_PER_BLOCK 1024

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#define MIN(x, y) (((x) < (y)) ? (x) : (y))


// Define a helper struct to determine if a type is a pointer
template <typename T>
struct is_pointer { static const bool value = false; };

template <typename T>
struct is_pointer<T*> { static const bool value = true; };

//________________________________________________________________
 
template <typename T, typename T_real>
__global__ void cuda_copy_and_set_zeros (T *v_row_dev, T *a_dev, int l_rows, int l_cols, int matrixRows, int istep,
                                         T *aux1_dev, T *vav_dev, T_real *d_vec_dev, 
                                         int isOurProcessRow, int isOurProcessCol, int isOurProcessCol_prev, int isSkewsymmetric, int useCCL){
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

  if (isOurProcessCol_prev)
    {
    // copy v_row to a_dev
    int i_row = tid;
    while (i_row < l_rows) 
      {
      v_row_dev[i_row] = a_dev[i_row + matrixRows*l_cols];
      i_row += blockDim.x * gridDim.x;
      }
    }

  // set zeros for aux1_dev and vav_dev to be summed with atomicAdd
  if (tid==0)
    {
    aux1_dev[0] = elpaDeviceNumber<T>(0.0);
    if (useCCL) *vav_dev = elpaDeviceNumber<T>(0.0);

    if (isOurProcessRow && isOurProcessCol)
      {
      if (isSkewsymmetric) 
        d_vec_dev[istep-1-1] = 0.0;
      else 
        d_vec_dev[istep-1-1] = elpaDeviceRealPart(a_dev[(l_rows-1) + matrixRows*(l_cols-1)]);
      }
    }
}

template <typename T, typename T_real>
void cuda_copy_and_set_zeros_FromC(T *v_row_dev, T *a_dev, int *l_rows_in, int *l_cols_in, int *matrixRows_in, int *istep_in, 
                                   T *aux1_dev, T *vav_dev, T_real *d_vec_dev, 
                                   int *isOurProcessRow_in,  int *isOurProcessCol_in, int *isOurProcessCol_prev_in, int *isSkewsymmetric_in, int *useCCL_in, int *wantDebug_in, cudaStream_t my_stream){
  int l_rows = *l_rows_in;   
  int l_cols = *l_cols_in;   
  int matrixRows = *matrixRows_in;
  int istep = *istep_in;
  int isOurProcessRow = *isOurProcessRow_in;
  int isOurProcessCol = *isOurProcessCol_in;
  int isOurProcessCol_prev = *isOurProcessCol_prev_in;
  int isSkewsymmetric = *isSkewsymmetric_in;
  int useCCL = *useCCL_in;
  int wantDebug = *wantDebug_in;

  int blocks = std::max((l_rows+MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK, 1);
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1); // TODO_23_11: change to NB?

#ifdef WITH_GPU_STREAMS
  cuda_copy_and_set_zeros<<<blocks,threadsPerBlock,0,my_stream>>>(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, aux1_dev, vav_dev, d_vec_dev, isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL);
#else
  cuda_copy_and_set_zeros<<<blocks,threadsPerBlock>>>            (v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, aux1_dev, vav_dev, d_vec_dev, isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL);
#endif

  if (wantDebug){
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) printf("Error in executing cuda_copy_and_set_zeros: %s\n",cudaGetErrorString(cuerr));
  }
}

extern "C" void cuda_copy_and_set_zeros_double_FromC(double *v_row_dev, double *a_dev, int *l_rows_in, int *l_cols_in, int *matrixRows_in, int *istep_in,
                                                     double *aux1_dev, double *vav_dev, double *d_vec_dev, 
                                                     int *isOurProcessRow_in, int *isOurProcessCol_in, int *isOurProcessCol_prev_in, int *isSkewsymmetric_in, int *useCCL_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_copy_and_set_zeros_FromC(v_row_dev, a_dev, l_rows_in, l_cols_in, matrixRows_in, istep_in, aux1_dev, vav_dev, d_vec_dev, isOurProcessRow_in, isOurProcessCol_in, isOurProcessCol_prev_in, isSkewsymmetric_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void cuda_copy_and_set_zeros_float_FromC(float *v_row_dev, float *a_dev, int *l_rows_in, int *l_cols_in, int *matrixRows_in, int *istep_in,
                                                    float *aux1_dev, float *vav_dev, float *d_vec_dev, 
                                                    int *isOurProcessRow_in, int *isOurProcessCol_in, int *isOurProcessCol_prev_in, int *isSkewsymmetric_in, int *useCCL_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_copy_and_set_zeros_FromC(v_row_dev, a_dev, l_rows_in, l_cols_in, matrixRows_in, istep_in, aux1_dev, vav_dev, d_vec_dev, isOurProcessRow_in, isOurProcessCol_in, isOurProcessCol_prev_in, isSkewsymmetric_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void cuda_copy_and_set_zeros_double_complex_FromC(cuDoubleComplex *v_row_dev, cuDoubleComplex *a_dev, int *l_rows_in, int *l_cols_in, int *matrixRows_in, int *istep_in,
                                                  cuDoubleComplex *aux1_dev, cuDoubleComplex *vav_dev, double *d_vec_dev, 
                                                  int *isOurProcessRow_in, int *isOurProcessCol_in, int *isOurProcessCol_prev_in, int *isSkewsymmetric_in, int *useCCL_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_copy_and_set_zeros_FromC(v_row_dev, a_dev, l_rows_in, l_cols_in, matrixRows_in, istep_in, aux1_dev, vav_dev, d_vec_dev, isOurProcessRow_in, isOurProcessCol_in, isOurProcessCol_prev_in, isSkewsymmetric_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void cuda_copy_and_set_zeros_float_complex_FromC(cuComplex *v_row_dev, cuComplex *a_dev, int *l_rows_in, int *l_cols_in, int *matrixRows_in, int *istep_in,
                                                  cuComplex *aux1_dev, cuComplex *vav_dev, float *d_vec_dev, 
                                                  int *isOurProcessRow_in, int *isOurProcessCol_in, int *isOurProcessCol_prev_in, int *isSkewsymmetric_in, int *useCCL_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_copy_and_set_zeros_FromC(v_row_dev, a_dev, l_rows_in, l_cols_in, matrixRows_in, istep_in, aux1_dev, vav_dev, d_vec_dev, isOurProcessRow_in, isOurProcessCol_in, isOurProcessCol_prev_in, isSkewsymmetric_in, useCCL_in, wantDebug_in, my_stream);
}

//________________________________________________________________
// device syncronization is needed afterwards, e.g. gpu_memcpy
// the kernel used only in NCCL codepath
// result_dev[0] must be initialized to zero before calling the kernel

template <typename T>
__global__ void cuda_dot_product_kernel(int n, T *x_dev, int incx, T *y_dev, int incy, T *result_dev){
  __shared__ T cache[MAX_THREADS_PER_BLOCK]; // extra space of fixed size is reserved for a speedup
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

  // T DeviceZero = elpaDeviceNumber<T>(0.0);
  //if (threadIdx.x==0) result_dev[0] = DeviceZero; // clear old value; it's safe because we perform __syncthreads() before atomicAdd
                                                  // TODO_23_11: actually can be unsafe: merge to the previous kernel

  T temp = elpaDeviceNumber<T>(0.0);
  int i = tid;
  while (i < n) {
    temp = elpaDeviceAdd(temp, elpaDeviceMultiply(elpaDeviceComplexConjugate(x_dev[i*incx]), y_dev[i*incy])); // temp += x_dev[i*incx] * y_dev[i*incy];
    i += blockDim.x * gridDim.x;
  }

  // set the cache values
  cache[threadIdx.x] = temp;
  // synchronize threads in this block
  __syncthreads();

  // for reductions, threadsPerBlock=blockDim.x must be a power of 2
  i = blockDim.x/2;
  while (i > 0) {
    if (threadIdx.x < i) cache[threadIdx.x] = elpaDeviceAdd(cache[threadIdx.x], cache[threadIdx.x + i]);
    __syncthreads();
    i /= 2;
  }

  if (threadIdx.x==0) atomicAdd(&result_dev[0], cache[0]);
  
}

template <typename T>
void cuda_dot_product_FromC(int* n_in, T *x_dev, int *incx_in, T *y_dev, int *incy_in, T *result_dev, int *wantDebug_in, int *SM_count_in, cudaStream_t my_stream){
  int n = *n_in;   
  int incx = *incx_in;
  int incy = *incy_in;
  int wantDebug = *wantDebug_in;
  int SM_count = *SM_count_in;
  //int SM_count=32;
  //cudaDeviceGetAttribute(&SM_count, cudaDevAttrMultiProcessorCount, 0); // TODO_23_11: move this outside, to set_gpu, claim the number only once during GPU setup

  int blocks = SM_count;
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1); // TODO_23_11: use NB here instead? 

#ifdef WITH_GPU_STREAMS
  cuda_dot_product_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(n, x_dev, incx, y_dev, incy, result_dev);
#else
  cuda_dot_product_kernel<<<blocks,threadsPerBlock>>>(n, x_dev, incx, y_dev, incy, result_dev);
#endif
  if (wantDebug){
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess){
      printf("Error in executing cuda_dot_product_kernel: %s\n",cudaGetErrorString(cuerr));
    }
  }
}

extern "C" void cuda_dot_product_double_FromC(int* n_in, double *x_dev, int *incx_in, double *y_dev, int *incy_in, double *result_dev, int *wantDebug_in, int *SM_count_in, cudaStream_t my_stream){
  cuda_dot_product_FromC(n_in, x_dev, incx_in, y_dev, incy_in, result_dev, wantDebug_in, SM_count_in, my_stream);
}

extern "C" void cuda_dot_product_float_FromC (int* n_in, float  *x_dev, int *incx_in, float  *y_dev, int *incy_in, float  *result_dev, int *wantDebug_in, int *SM_count_in, cudaStream_t my_stream){
  cuda_dot_product_FromC(n_in, x_dev, incx_in, y_dev, incy_in, result_dev, wantDebug_in, SM_count_in, my_stream);
}

extern "C" void cuda_dot_product_double_complex_FromC(int* n_in, cuDoubleComplex *x_dev, int *incx_in, cuDoubleComplex *y_dev, int *incy_in, cuDoubleComplex *result_dev, int *wantDebug_in, int *SM_count_in, cudaStream_t my_stream){
  cuda_dot_product_FromC(n_in, x_dev, incx_in, y_dev, incy_in, result_dev, wantDebug_in, SM_count_in, my_stream);
}

extern "C" void cuda_dot_product_float_complex_FromC (int* n_in, cuComplex *x_dev, int *incx_in, cuComplex *y_dev, int *incy_in, cuComplex *result_dev, int *wantDebug_in, int *SM_count_in, cudaStream_t my_stream){
  cuda_dot_product_FromC(n_in, x_dev, incx_in, y_dev, incy_in, result_dev, wantDebug_in, SM_count_in, my_stream);
}

//________________________________________________________________
// aux1_dev[0] must be initialized to zero before calling the kernel

template <typename T>
__global__ void cuda_dot_product_and_assign_kernel(T *v_row_dev, int l_rows, int isOurProcessRow, T *aux1_dev){
  const int threadsPerBlock = MAX_THREADS_PER_BLOCK;
  __shared__ T cache[threadsPerBlock];
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

/*
  if (isOurProcessRow) {
    aux1(1) = dot_product(v_row(1:l_rows-1),v_row(1:l_rows-1)) ! = "q"
    aux1(2) = v_row(l_rows) ! = "a_11" (or rather a_nn)
    }
  else{
    aux1(1) = dot_product(v_row(1:l_rows),v_row(1:l_rows))
    aux1(2) = 0.
  }
*/
  
  //if (threadIdx.x==0) aux1_dev[0] = DeviceZero; // clear old value; it's safe because we perform __syncthreads() before atomicAdd

  T temp = elpaDeviceNumber<T>(0.0);
  int index_global = tid;
  while (index_global < l_rows-1) {
    temp = elpaDeviceAdd(temp, elpaDeviceMultiply(elpaDeviceComplexConjugate(v_row_dev[index_global]), v_row_dev[index_global])); // temp += v_row_dev[index_global]*v_row_dev[index_global];
    index_global += blockDim.x * gridDim.x;
  }

  // set the cache values
  cache[threadIdx.x] = temp;
  // synchronize threads in this block
  __syncthreads();

  // for reductions, threadsPerBlock must be a power of 2
  int i = blockDim.x/2;
  while (i > 0) {
    if (threadIdx.x < i) cache[threadIdx.x] = elpaDeviceAdd(cache[threadIdx.x], cache[threadIdx.x + i]);
    __syncthreads();
    i /= 2;
  }

  if (threadIdx.x==0) atomicAdd(&aux1_dev[0], cache[0]);
  
  if (tid==0)
    {
    if (isOurProcessRow) 
      {
      aux1_dev[1] = v_row_dev[l_rows-1];
      }
    else
      {
      if (l_rows>0) atomicAdd(&aux1_dev[0], elpaDeviceMultiply(elpaDeviceComplexConjugate(v_row_dev[l_rows-1]), v_row_dev[l_rows-1]));
      aux1_dev[1] = elpaDeviceNumber<T>(0.0);
      }
    }
}

template <typename T>
void cuda_dot_product_and_assign_FromC(T *v_row_dev, int *l_rows_in, int *isOurProcessRow_in, T *aux1_dev, int *wantDebug_in, cudaStream_t my_stream){
  int l_rows = *l_rows_in;   
  int isOurProcessRow = *isOurProcessRow_in;
  int wantDebug = *wantDebug_in;

  //int numSMs;
  //cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
  
  //int blocks = (l_rows+1023)/MAX_THREADS_PER_BLOCK;
  int blocks = 32; // TODO_23_11: change blocksPerGrid to number of SM's (108 fo A100) and threadsPerBlock to max threads per block. claim the number only once during GPU setup
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_dot_product_and_assign_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(v_row_dev, l_rows, isOurProcessRow, aux1_dev);
#else
  cuda_dot_product_and_assign_kernel<<<blocks,threadsPerBlock>>>(v_row_dev, l_rows, isOurProcessRow, aux1_dev);
#endif

  if (wantDebug){
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess){
      printf("Error in executing cuda_dot_product_and_assign_kernel: %s\n",cudaGetErrorString(cuerr));
    }
  }

}

extern "C" void cuda_dot_product_and_assign_double_FromC(double *v_row_dev, int *l_rows_in, int *isOurProcessRow_in, double *aux1_dev, int *wantDebug_in, cudaStream_t my_stream){
  cuda_dot_product_and_assign_FromC(v_row_dev, l_rows_in, isOurProcessRow_in, aux1_dev, wantDebug_in, my_stream);
}

extern "C" void cuda_dot_product_and_assign_float_FromC(float *v_row_dev, int *l_rows_in, int *isOurProcessRow_in, float *aux1_dev, int *wantDebug_in, cudaStream_t my_stream){
  cuda_dot_product_and_assign_FromC(v_row_dev, l_rows_in, isOurProcessRow_in, aux1_dev, wantDebug_in, my_stream);
}

extern "C" void cuda_dot_product_and_assign_double_complex_FromC(cuDoubleComplex *v_row_dev, int *l_rows_in, int *isOurProcessRow_in, cuDoubleComplex *aux1_dev, int *wantDebug_in, cudaStream_t my_stream){
  cuda_dot_product_and_assign_FromC(v_row_dev, l_rows_in, isOurProcessRow_in, aux1_dev, wantDebug_in, my_stream);
}

extern "C" void cuda_dot_product_and_assign_float_complex_FromC(cuComplex *v_row_dev, int *l_rows_in, int *isOurProcessRow_in, cuComplex *aux1_dev, int *wantDebug_in, cudaStream_t my_stream){
  cuda_dot_product_and_assign_FromC(v_row_dev, l_rows_in, isOurProcessRow_in, aux1_dev, wantDebug_in, my_stream);
}

//________________________________________________________________

template <typename T, typename T_real, typename T_value_or_pointer>
__global__ void cuda_set_e_vec_scale_set_one_store_v_row_kernel(T_real *e_vec_dev, T *vrl_dev, T *a_dev, T *v_row_dev, T *tau_dev, T_value_or_pointer xf_host_or_dev, 
                                                      int l_rows, int l_cols,  int matrixRows, int istep, int isOurProcessRow, int useCCL){
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

/*
  if (my_prow == prow(istep-1, nblk, np_rows)) then
    if (.not. useCCL) then
#if REALCASE == 1
      e_vec(istep-1) = vrl
#endif
#if COMPLEXCASE == 1
      e_vec(istep-1) = real(vrl,kind=rk)
#endif
    endif ! useCCL
  endif

  call nvtxRangePush("scale v_row *= xf")
  ! Scale v_row and store Householder Vector for back transformation
  v_row(1:l_rows) = v_row(1:l_rows) * xf
  call nvtxRangePop()

  if (my_prow == prow(istep-1, nblk, np_rows)) then
    v_row(l_rows) = 1.
  endif

  ! store Householder Vector for back transformation
  call nvtxRangePush("cpu copy: v_row->a_mat")
  ! update a_mat
  a_mat(1:l_rows,l_cols+1) = v_row(1:l_rows)
  call nvtxRangePop()

  if (.not. useCCL) then
    ! add tau after the end of actuall v_row, to be broadcasted with it
    v_row(l_rows+1) = tau(istep)
  endif
*/

  T xf = convert_to_device(xf_host_or_dev, typename std::conditional<std::is_pointer<T_value_or_pointer>::value, std::true_type, std::false_type>::type());

  if (useCCL && tid==0)
    {
    if (isOurProcessRow) e_vec_dev[istep-1-1] = elpaDeviceRealPart(*vrl_dev);
    v_row_dev[l_rows+1-1] = tau_dev[istep-1];
    }

  int index_global = tid;
  while (index_global < l_rows) {
    v_row_dev[index_global] = elpaDeviceMultiply(v_row_dev[index_global], xf);
    index_global += blockDim.x * gridDim.x;
  }

  if (isOurProcessRow && (index_global - blockDim.x*gridDim.x == l_rows-1)) // last element
    {
    v_row_dev[l_rows-1] = elpaDeviceNumber<T>(1.0);
    }
    
  int i_row = tid;
  while (i_row < l_rows) 
    {
    a_dev[i_row + matrixRows*l_cols] = v_row_dev[i_row];
    i_row += blockDim.x * gridDim.x;
    }

}

template <typename T, typename T_real>
void cuda_set_e_vec_scale_set_one_store_v_row_FromC(T_real *e_vec_dev, T *vrl_dev, T *a_dev, T *v_row_dev, T *tau_dev, T *xf_host_or_dev, 
                                                    int *l_rows_in, int *l_cols_in,  int *matrixRows_in, int *istep_in, int *isOurProcessRow_in, int *useCCL_in, int *wantDebug_in, cudaStream_t my_stream){
  int l_rows = *l_rows_in;   
  int l_cols = *l_cols_in;   
  int matrixRows = *matrixRows_in;
  int istep = *istep_in;
  int isOurProcessRow = *isOurProcessRow_in;
  int useCCL = *useCCL_in;
  int wantDebug = *wantDebug_in;

  int blocks = std::max((l_rows+MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK, 1);
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1); // TODO_23_11: change to NB?

  cudaPointerAttributes attributes;
  cudaError_t error = cudaPointerGetAttributes(&attributes, xf_host_or_dev);

  if (error == cudaSuccess) 
    {
    if (attributes.type == cudaMemoryTypeHost) 
      {
      T xf_host_value = *xf_host_or_dev;
#ifdef WITH_GPU_STREAMS
      cuda_set_e_vec_scale_set_one_store_v_row_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_value,
                                                                                                 l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL);
#else 
      cuda_set_e_vec_scale_set_one_store_v_row_kernel<<<blocks,threadsPerBlock>>>(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_value,
                                                                                    l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL);
#endif
      if (wantDebug)
        {
        cudaError_t cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess) printf("Error in executing cuda_set_e_vec_scale_set_one_store_v_row_kernel: %s\n",cudaGetErrorString(cuerr));
        }
      }
    else if (attributes.type == cudaMemoryTypeDevice) 
      {
#ifdef WITH_GPU_STREAMS
      cuda_set_e_vec_scale_set_one_store_v_row_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev,
                                                                                                 l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL);
#else
      cuda_set_e_vec_scale_set_one_store_v_row_kernel<<<blocks,threadsPerBlock>>>(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev,
                                                                                    l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL);
#endif
      if (wantDebug)
        {
        cudaError_t cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess) printf("Error in executing cuda_set_e_vec_scale_set_one_store_v_row_kernel: %s\n",cudaGetErrorString(cuerr));
        }
      }
    } 
  
  else 
    {
    printf("Error: Pointer is unknown\n");
    }

}

extern "C" void cuda_set_e_vec_scale_set_one_store_v_row_double_FromC(double *e_vec_dev, double *vrl_dev, double *a_dev, double *v_row_dev, double *tau_dev, double *xf_host_or_dev, 
                                                                      int *l_rows_in, int *l_cols_in,  int *matrixRows_in, int *istep_in, int *isOurProcessRow_in, int *useCCL_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_set_e_vec_scale_set_one_store_v_row_FromC(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, l_rows_in, l_cols_in, matrixRows_in, istep_in, isOurProcessRow_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void cuda_set_e_vec_scale_set_one_store_v_row_float_FromC(float *e_vec_dev, float *vrl_dev, float *a_dev, float *v_row_dev, float *tau_dev, float *xf_host_or_dev, 
                                                                     int *l_rows_in, int *l_cols_in,  int *matrixRows_in, int *istep_in, int *isOurProcessRow_in, int *useCCL_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_set_e_vec_scale_set_one_store_v_row_FromC(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, l_rows_in, l_cols_in, matrixRows_in, istep_in, isOurProcessRow_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void cuda_set_e_vec_scale_set_one_store_v_row_double_complex_FromC(double *e_vec_dev, cuDoubleComplex *vrl_dev, cuDoubleComplex *a_dev, cuDoubleComplex *v_row_dev, cuDoubleComplex *tau_dev, cuDoubleComplex *xf_host_or_dev, 
                                                                               int *l_rows_in, int *l_cols_in,  int *matrixRows_in, int *istep_in, int *isOurProcessRow_in, int *useCCL_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_set_e_vec_scale_set_one_store_v_row_FromC(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, l_rows_in, l_cols_in, matrixRows_in, istep_in, isOurProcessRow_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void cuda_set_e_vec_scale_set_one_store_v_row_float_complex_FromC(float *e_vec_dev, cuComplex *vrl_dev, cuComplex *a_dev, cuComplex *v_row_dev, cuComplex *tau_dev, cuComplex *xf_host_or_dev, 
                                                                              int *l_rows_in, int *l_cols_in,  int *matrixRows_in, int *istep_in, int *isOurProcessRow_in, int *useCCL_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_set_e_vec_scale_set_one_store_v_row_FromC(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, l_rows_in, l_cols_in, matrixRows_in, istep_in, isOurProcessRow_in, useCCL_in, wantDebug_in, my_stream);
}

//________________________________________________________________

template <typename T, typename T_value_or_pointer>
__global__ void cuda_store_u_v_in_uv_vu_kernel(T *vu_stored_rows_dev, T *uv_stored_cols_dev, T *v_row_dev, T *u_row_dev,
                T *v_col_dev, T *u_col_dev, T *tau_dev, T *aux_complex_dev,
                T_value_or_pointer vav_host_or_dev, T_value_or_pointer tau_host_or_dev,
                int l_rows, int l_cols, int n_stored_vecs, int max_local_rows, int max_local_cols, int istep, int useCCL){
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

  T conjg_tau = convert_to_device(tau_host_or_dev, typename std::conditional<std::is_pointer<T_value_or_pointer>::value, std::true_type, std::false_type>::type());
  conjg_tau = elpaDeviceComplexConjugate(conjg_tau);
  
  T conjg_tau_v_row_dev, conjg_tau_v_col_dev;

  // recover tau_dev(istep) after broadcasting
  if (useCCL && tid==0)
    {
    tau_dev[istep-1] = v_row_dev[l_rows+1-1];
    }

/*
  // istep
  if (l_rows > 0) then
    ! update vu_stored_rows
    vu_stored_rows(1:l_rows,2*n_stored_vecs+1) = conjg_tau*v_row(1:l_rows)
    vu_stored_rows(1:l_rows,2*n_stored_vecs+2) = 0.5*conjg_tau*vav*v_row(1:l_rows) - u_row(1:l_rows)
  endif
  if (l_cols > 0) then
    ! update uv_stored_cols
    uv_stored_cols(1:l_cols,2*n_stored_vecs+1) = 0.5*conjg_tau*vav*v_col(1:l_cols) - u_col(1:l_cols)
    uv_stored_cols(1:l_cols,2*n_stored_vecs+2) = conjg_tau*v_col(1:l_cols)
  endif
...

// istep = istep-1
// n_stored_vecs = n_stored_vecs+1
#if COMPLEXCASE == 1
          aux(1:2*n_stored_vecs) = conjg(uv_stored_cols(l_cols+1,1:2*n_stored_vecs))
#endif
*/
  T vav =  convert_to_device(vav_host_or_dev, typename std::conditional<std::is_pointer<T_value_or_pointer>::value, std::true_type, std::false_type>::type());

  int i_row = tid;
  while (i_row < l_rows)
    {
    conjg_tau_v_row_dev = elpaDeviceMultiply(conjg_tau, v_row_dev[i_row]);
    vu_stored_rows_dev[i_row + max_local_rows*(2*n_stored_vecs+0)] = conjg_tau_v_row_dev;
    conjg_tau_v_row_dev = elpaDeviceMultiply(conjg_tau_v_row_dev, elpaDeviceNumber<T>(0.5));
    vu_stored_rows_dev[i_row + max_local_rows*(2*n_stored_vecs+1)] =  elpaDeviceSubtract( elpaDeviceMultiply(conjg_tau_v_row_dev, vav) , u_row_dev[i_row] );

    i_row += blockDim.x * gridDim.x;
    }

  int i_col = tid;
  while (i_col < l_cols)
    {
    conjg_tau_v_col_dev = elpaDeviceMultiply(conjg_tau, v_col_dev[i_col]);
    uv_stored_cols_dev[i_col + max_local_cols*(2*n_stored_vecs+1)] = conjg_tau_v_col_dev;
    conjg_tau_v_col_dev = elpaDeviceMultiply(conjg_tau_v_col_dev, elpaDeviceNumber<T>(0.5));
    uv_stored_cols_dev[i_col + max_local_cols*(2*n_stored_vecs+0)] = elpaDeviceSubtract( elpaDeviceMultiply(conjg_tau_v_col_dev, vav) , u_col_dev[i_col] );

    i_col += blockDim.x * gridDim.x;
    }


  if ((std::is_same<T, cuDoubleComplex>::value || std::is_same<T, cuComplex>::value) && l_cols>0)
    {
    int j = tid;
    while (j < 2*(n_stored_vecs+0)) // whole vector aux_complex_dev has to be copied and not two last elements only because l_cols has changed since last istep
      {
      aux_complex_dev[j] = elpaDeviceComplexConjugate(uv_stored_cols_dev[l_cols-1 + max_local_cols*j]);
      j += blockDim.x * gridDim.x;
      }
    
    // two last elements should be treated by the respective threads inorder to avoid sync problems
    i_col -= blockDim.x * gridDim.x;
    if (i_col == l_cols-1) 
      {
      aux_complex_dev[2*n_stored_vecs+0] = elpaDeviceComplexConjugate(uv_stored_cols_dev[l_cols-1 + max_local_cols*(2*n_stored_vecs+0)]);
      aux_complex_dev[2*n_stored_vecs+1] = elpaDeviceComplexConjugate(uv_stored_cols_dev[l_cols-1 + max_local_cols*(2*n_stored_vecs+1)]);
      }
    }
}


template <typename T>
void cuda_store_u_v_in_uv_vu_FromC(T *vu_stored_rows_dev, T *uv_stored_cols_dev, T *v_row_dev, T *u_row_dev,
                T *v_col_dev, T *u_col_dev, T *tau_dev, T *aux_complex_dev, T *vav_host_or_dev, T *tau_host_or_dev,
                int *l_rows_in, int *l_cols_in, int *n_stored_vecs_in, int *max_local_rows_in, int *max_local_cols_in, int *istep_in, int *useCCL_in, int *wantDebug_in, cudaStream_t my_stream){
  int l_rows = *l_rows_in;   
  int l_cols = *l_cols_in;   
  int n_stored_vecs  = *n_stored_vecs_in;
  int max_local_rows = *max_local_rows_in;   
  int max_local_cols = *max_local_cols_in;   
  int istep = *istep_in;   
  int useCCL = *useCCL_in;
  int wantDebug = *wantDebug_in;
  
  int threads = MAX_THREADS_PER_BLOCK/2; // TODO_23_11: the kernel has many local variables, for which we need memory registers. So we use less threads here to save memory. Change to smth else for safety?
  int blocks = std::max({(l_rows+threads-1)/threads, (l_cols+threads-1)/threads, 1});
  
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(threads,1,1);

  cudaPointerAttributes attributes;
  cudaError_t error = cudaPointerGetAttributes(&attributes, vav_host_or_dev);

  if (error == cudaSuccess) 
    {
    if (attributes.type == cudaMemoryTypeHost) 
      {
      T vav_host_value = *vav_host_or_dev;
      T tau_host_value = *tau_host_or_dev;
#ifdef WITH_GPU_STREAMS
      cuda_store_u_v_in_uv_vu_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, 
                                       v_col_dev, u_col_dev, tau_dev, aux_complex_dev,
                                       vav_host_value, tau_host_value, 
                                       l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, useCCL);
#else
      cuda_store_u_v_in_uv_vu_kernel<<<blocks,threadsPerBlock>>>(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, 
                                       v_col_dev, u_col_dev, tau_dev, aux_complex_dev,
                                       vav_host_value, tau_host_value, 
                                       l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, useCCL);
#endif
      if (wantDebug)
        {
        cudaError_t cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess) printf("Error in executing cuda_store_u_v_in_uv_vu_kernel: %s\n",cudaGetErrorString(cuerr));
        }
      } 
    
    else if (attributes.type == cudaMemoryTypeDevice) 
      {
#ifdef WITH_GPU_STREAMS
      cuda_store_u_v_in_uv_vu_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, 
                                       v_col_dev, u_col_dev, tau_dev, aux_complex_dev,
                                       vav_host_or_dev, tau_host_or_dev, 
                                       l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, useCCL);
#else
      cuda_store_u_v_in_uv_vu_kernel<<<blocks,threadsPerBlock>>>(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, 
                                       v_col_dev, u_col_dev, tau_dev, aux_complex_dev,
                                       vav_host_or_dev, tau_host_or_dev, 
                                       l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, useCCL);
#endif

      if (wantDebug)
        {
        cudaError_t cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess) printf("Error in executing cuda_store_u_v_in_uv_vu_kernel: %s\n",cudaGetErrorString(cuerr));
        }
      } 
    } 
  
  else 
    {
    printf("Error: Pointer is unknown\n");
    }

}

extern "C" void cuda_store_u_v_in_uv_vu_double_FromC(double *vu_stored_rows_dev, double *uv_stored_cols_dev, double *v_row_dev, double *u_row_dev,
                                                     double *v_col_dev, double *u_col_dev, double *tau_dev, double *aux_complex_dev,
                                                     double *vav_host_or_dev, double *tau_host_or_dev,
                                                     int *l_rows_in, int *l_cols_in, int *n_stored_vecs_in, int *max_local_rows_in, int *max_local_cols_in, int *istep_in, int *useCCL_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_store_u_v_in_uv_vu_FromC(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, v_col_dev, u_col_dev, tau_dev, aux_complex_dev,
                                vav_host_or_dev, tau_host_or_dev, 
                                l_rows_in, l_cols_in, n_stored_vecs_in, max_local_rows_in, max_local_cols_in, istep_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void cuda_store_u_v_in_uv_vu_float_FromC(float *vu_stored_rows_dev, float *uv_stored_cols_dev, float *v_row_dev, float *u_row_dev,
                                                    float *v_col_dev, float *u_col_dev, float *tau_dev, float *aux_complex_dev,
                                                    float *vav_host_or_dev, float *tau_host_or_dev,
                                                    int *l_rows_in, int *l_cols_in, int *n_stored_vecs_in, int *max_local_rows_in, int *max_local_cols_in, int *istep_in, int *useCCL_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_store_u_v_in_uv_vu_FromC(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, v_col_dev, u_col_dev, tau_dev, aux_complex_dev, vav_host_or_dev, tau_host_or_dev, l_rows_in, l_cols_in, n_stored_vecs_in, max_local_rows_in, max_local_cols_in, istep_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void cuda_store_u_v_in_uv_vu_double_complex_FromC(cuDoubleComplex *vu_stored_rows_dev, cuDoubleComplex *uv_stored_cols_dev, cuDoubleComplex *v_row_dev, cuDoubleComplex *u_row_dev,
                                                             cuDoubleComplex *v_col_dev, cuDoubleComplex *u_col_dev, cuDoubleComplex *tau_dev, cuDoubleComplex *aux_complex_dev,
                                                             cuDoubleComplex *vav_host_or_dev, cuDoubleComplex *tau_host_or_dev,
                                                             int *l_rows_in, int *l_cols_in, int *n_stored_vecs_in, int *max_local_rows_in, int *max_local_cols_in, int *istep_in, int *useCCL_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_store_u_v_in_uv_vu_FromC(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, v_col_dev, u_col_dev, tau_dev, aux_complex_dev, vav_host_or_dev, tau_host_or_dev, l_rows_in, l_cols_in, n_stored_vecs_in, max_local_rows_in, max_local_cols_in, istep_in, useCCL_in, wantDebug_in, my_stream);
}

extern "C" void cuda_store_u_v_in_uv_vu_float_complex_FromC(cuComplex *vu_stored_rows_dev, cuComplex *uv_stored_cols_dev, cuComplex *v_row_dev, cuComplex *u_row_dev,
                                                            cuComplex *v_col_dev, cuComplex *u_col_dev, cuComplex *tau_dev, cuComplex *aux_complex_dev, 
                                                            cuComplex *vav_host_or_dev, cuComplex *tau_host_or_dev,
                                                            int *l_rows_in, int *l_cols_in, int *n_stored_vecs_in, int *max_local_rows_in, int *max_local_cols_in, int *istep_in, int *useCCL_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_store_u_v_in_uv_vu_FromC(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, v_col_dev, u_col_dev, tau_dev, aux_complex_dev, vav_host_or_dev, tau_host_or_dev, l_rows_in, l_cols_in, n_stored_vecs_in, max_local_rows_in, max_local_cols_in, istep_in, useCCL_in, wantDebug_in, my_stream);
}

//________________________________________________________________

template <typename T, typename T_real>
__global__ void cuda_update_matrix_element_add_kernel(T *vu_stored_rows_dev, T *uv_stored_cols_dev, T *a_dev, T_real *d_vec_dev, 
                                                      int l_rows, int l_cols, int matrixRows, int max_local_rows, int max_local_cols, int istep, int n_stored_vecs, int isSkewsymmetric){
  
  const int threadsPerBlock = MAX_THREADS_PER_BLOCK;
  __shared__ T cache[threadsPerBlock];
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

/*
      if (n_stored_vecs > 0) then
        ! update a_mat (only one elememt!)
        dot_prod = dot_product(vu_stored_rows(l_rows,1:2*n_stored_vecs), uv_stored_cols(l_cols,1:2*n_stored_vecs))
        a_mat(l_rows,l_cols) = a_mat(l_rows,l_cols) + dot_prod
      endif
#if REALCASE == 1
      if (isSkewsymmetric) then
        d_vec(istep-1) = 0.0_rk
      else
        d_vec(istep-1) = a_mat(l_rows,l_cols)
      endif
#endif
#if COMPLEXCASE == 1
      d_vec(istep-1) = real(a_mat(l_rows,l_cols),kind=rk)
#endif
*/

  // if (threadIdx.x==0)
  //   { 
  //   if (isSkewsymmetric) 
  //     d_vec_dev[istep-1-1] = 0.0;
  //   else 
  //     d_vec_dev[istep-1-1] = elpaDeviceRealPart(a_dev[(l_rows-1) + matrixRows*(l_cols-1)]); // set initial value // TODO_23_11: move this to the previous kernel for thread safety
  //   }
  if (n_stored_vecs > 0)
    {

    T temp =  elpaDeviceNumber<T>(0.0);
    int index_n = tid;
    while (index_n < 2*n_stored_vecs) 
      {
      temp = elpaDeviceAdd(temp, elpaDeviceMultiply(elpaDeviceComplexConjugate(vu_stored_rows_dev[(l_rows-1)+max_local_rows*index_n]), uv_stored_cols_dev[(l_cols-1)+max_local_cols*index_n]) ); // temp += vu_stored_rows_dev[(l_rows-1)+max_local_rows*index_n] * uv_stored_cols_dev[(l_cols-1)+max_local_cols*index_n]
      index_n += blockDim.x * gridDim.x;
      }

    // set the cache values
    cache[threadIdx.x] = temp;
    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    int i = blockDim.x/2;
    while (i > 0) 
      {
      if (threadIdx.x < i) cache[threadIdx.x] = elpaDeviceAdd(cache[threadIdx.x] , cache[threadIdx.x + i]); // cache[threadIdx.x] += cache[threadIdx.x + i];
      __syncthreads();
      i /= 2;
      }

    if (threadIdx.x==0) 
      {
      atomicAdd(&a_dev[(l_rows-1) + matrixRows*(l_cols-1)], cache[0]);
      if (!isSkewsymmetric) atomicAdd( &d_vec_dev[istep-1-1] , elpaDeviceRealPart(cache[0]) );
      }
    }
}

template <typename T, typename T_real>
void cuda_update_matrix_element_add_FromC(T *vu_stored_rows_dev, T *uv_stored_cols_dev, T *a_dev, T_real *d_vec_dev, 
                                          int *l_rows_in, int *l_cols_in, int *matrixRows_in, int *max_local_rows_in, int *max_local_cols_in, int *istep_in, int *n_stored_vecs_in, 
                                          int* isSkewsymmetric_in, int *wantDebug_in, cudaStream_t my_stream){
  int l_rows = *l_rows_in;   
  int l_cols = *l_cols_in;
  int matrixRows = *matrixRows_in;
  int max_local_rows = *max_local_rows_in;
  int max_local_cols = *max_local_cols_in;
  int istep = *istep_in;   
  int n_stored_vecs = *n_stored_vecs_in; 
  int isSkewsymmetric = *isSkewsymmetric_in;   
  int wantDebug = *wantDebug_in;
  
  int blocks = std::min((2*n_stored_vecs+MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK, 32);
  if (n_stored_vecs==0) blocks=1;
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_update_matrix_element_add_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev,
                                                  l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs,
                                                  isSkewsymmetric);
#else
  cuda_update_matrix_element_add_kernel<<<blocks,threadsPerBlock>>>(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, 
                                                  l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, 
                                                  isSkewsymmetric);
#endif
  if (wantDebug){
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess){
      printf("Error in executing cuda_update_matrix_element_add_kernel: %s\n",cudaGetErrorString(cuerr));
    }
  }
}

extern "C" void cuda_update_matrix_element_add_double_FromC(double *vu_stored_rows_dev, double *uv_stored_cols_dev, double *a_dev, double *d_vec_dev, 
                                                            int *l_rows_in, int *l_cols_in, int *matrixRows_in, int *max_local_rows_in, int *max_local_cols_in, int *istep_in, int *n_stored_vecs_in, 
                                                            int* isSkewsymmetric_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_update_matrix_element_add_FromC(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, l_rows_in, l_cols_in, matrixRows_in, max_local_rows_in, max_local_cols_in, istep_in, n_stored_vecs_in, isSkewsymmetric_in, wantDebug_in, my_stream);
}

extern "C" void cuda_update_matrix_element_add_float_FromC(float *vu_stored_rows_dev, float *uv_stored_cols_dev, float *a_dev, float *d_vec_dev, 
                                                           int *l_rows_in, int *l_cols_in, int *matrixRows_in, int *max_local_rows_in, int *max_local_cols_in, int *istep_in, int *n_stored_vecs_in, 
                                                           int* isSkewsymmetric_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_update_matrix_element_add_FromC(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, l_rows_in, l_cols_in, matrixRows_in, max_local_rows_in, max_local_cols_in, istep_in, n_stored_vecs_in, isSkewsymmetric_in, wantDebug_in, my_stream);
}

extern "C" void cuda_update_matrix_element_add_double_complex_FromC(cuDoubleComplex *vu_stored_rows_dev, cuDoubleComplex *uv_stored_cols_dev, cuDoubleComplex *a_dev, double *d_vec_dev, 
                                                                    int *l_rows_in, int *l_cols_in, int *matrixRows_in, int *max_local_rows_in, int *max_local_cols_in, int *istep_in, int *n_stored_vecs_in, 
                                                                    int* isSkewsymmetric_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_update_matrix_element_add_FromC(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, l_rows_in, l_cols_in, matrixRows_in, max_local_rows_in, max_local_cols_in, istep_in, n_stored_vecs_in, isSkewsymmetric_in, wantDebug_in, my_stream);
}

extern "C" void cuda_update_matrix_element_add_float_complex_FromC(cuComplex *vu_stored_rows_dev, cuComplex *uv_stored_cols_dev, cuComplex *a_dev, float *d_vec_dev, 
                                                                   int *l_rows_in, int *l_cols_in, int *matrixRows_in, int *max_local_rows_in, int *max_local_cols_in, int *istep_in, int *n_stored_vecs_in, 
                                                                   int* isSkewsymmetric_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_update_matrix_element_add_FromC(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, l_rows_in, l_cols_in, matrixRows_in, max_local_rows_in, max_local_cols_in, istep_in, n_stored_vecs_in, isSkewsymmetric_in, wantDebug_in, my_stream);
}

//________________________________________________________________

template <typename T>
__global__ void cuda_update_array_element_kernel(T *array_dev, const int index, T value){

  array_dev[index-1] = value;

}

template <typename T>
void cuda_update_array_element_FromC(T *array_dev, int *index_in, T *value_in, cudaStream_t my_stream){
  int index = *index_in;   
  T value = *value_in;

  dim3 blocks = dim3(1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_update_array_element_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(array_dev, index, value);
#else
  cuda_update_array_element_kernel<<<blocks,threadsPerBlock>>>(array_dev, index, value);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_update_array_element_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

extern "C" void cuda_update_array_element_double_FromC(double *array_dev, int *index_in, double *value_in, cudaStream_t my_stream){
  cuda_update_array_element_FromC(array_dev, index_in, value_in, my_stream);
}

extern "C" void cuda_update_array_element_float_FromC(float *array_dev, int *index_in, float *value_in, cudaStream_t my_stream){
  cuda_update_array_element_FromC(array_dev, index_in, value_in, my_stream);
}

extern "C" void cuda_update_array_element_double_complex_FromC(cuDoubleComplex *array_dev, int *index_in, cuDoubleComplex *value_in, cudaStream_t my_stream){
  cuda_update_array_element_FromC(array_dev, index_in, value_in, my_stream);
}

extern "C" void cuda_update_array_element_float_complex_FromC(cuComplex *array_dev, int *index_in, cuComplex *value_in, cudaStream_t my_stream){
  cuda_update_array_element_FromC(array_dev, index_in, value_in, my_stream);
}

//________________________________________________________________

template <typename T>
__global__ void cuda_hh_transform_kernel(T *alpha_dev, T *xnorm_sq_dev, T *xf_dev, T *tau_dev, int wantDebug_in){

/*
#if complexcase == 1
  alphr = real( alpha, kind=rk )
  alphi = precision_imag( alpha )
#endif

#if realcase == 1
  if ( xnorm_sq==0.0_rk ) then
#endif
#if complexcase == 1
  if ( xnorm_sq==0.0_rk .and. alphi==0.0_rk ) then
#endif

#if realcase == 1
    if ( alpha>=0.0_rk ) then
#endif
#if complexcase == 1
    if ( alphr>=0.0_rk ) then
#endif
      tau = 0.0_rk
    else
      tau = 2.0_rk
      alpha = -alpha
    endif
    xf = 0.0_rk

  else

#if realcase == 1
    beta = sign( sqrt( alpha**2 + xnorm_sq ), alpha )
#endif
#if complexcase == 1
    beta = sign( sqrt( alphr**2 + alphi**2 + xnorm_sq ), alphr )
#endif
    alpha = alpha + beta
    if ( beta<0 ) then
      beta = -beta
      tau  = -alpha / beta
    else
#if realcase == 1
      alpha = xnorm_sq / alpha
#endif
#if complexcase == 1
      alphr = alphi * (alphi/real( alpha , kind=rk))
      alphr = alphr + xnorm_sq/real( alpha, kind=rk )
#endif

#if realcase == 1
      tau = alpha / beta
      alpha = -alpha
#endif
#if complexcase == 1
      tau = precision_cmplx( alphr/beta, -alphi/beta )
      alpha = precision_cmplx( -alphr, alphi )
#endif
    end if
    xf = 1.0_rk/alpha
    alpha = beta
  endif
*/

  auto alpha_r = elpaDeviceRealPart(*alpha_dev);
  auto alpha_i = elpaDeviceImagPart(*alpha_dev);

  if (elpaDeviceRealPart(*xnorm_sq_dev)==0.0 && alpha_i==0.0)
    {
    if (alpha_r >= 0.0) *tau_dev = elpaDeviceNumber<T>(0.0);
    else
      {
      *tau_dev = elpaDeviceNumber<T>(2.0);
      *alpha_dev = elpaDeviceMultiply(*alpha_dev, elpaDeviceNumber<T>(-1.0)); // (*alpha_dev) *= -1
      }
    
    *xf_dev = elpaDeviceNumber<T>(0.0);
    }

  else
    {
    T beta = elpaDeviceNumber<T> (elpaDeviceSign( elpaDeviceSqrt( alpha_r*alpha_r + alpha_i*alpha_i +
                                          elpaDeviceRealPart(*xnorm_sq_dev) ), elpaDeviceRealPart(*alpha_dev) ));

    *alpha_dev = elpaDeviceAdd(*alpha_dev, beta);
    
    if (elpaDeviceRealPart(beta)<0)
      {
      *tau_dev  = elpaDeviceDivide(*alpha_dev, beta);
      beta = elpaDeviceMultiply(beta, elpaDeviceNumber<T>(-1.0)); // beta *= -1
      }
    else
      {
      alpha_r = alpha_i * alpha_i/elpaDeviceRealPart(*alpha_dev);
      alpha_r = alpha_r + elpaDeviceRealPart(*xnorm_sq_dev)/ elpaDeviceRealPart(*alpha_dev);

      *tau_dev = elpaDeviceDivide( elpaDeviceNumberFromRealImag<T>(alpha_r, -alpha_i) , beta );
      *alpha_dev = elpaDeviceNumberFromRealImag<T>(-alpha_r, alpha_i);
      }

    *xf_dev = elpaDeviceDivide(elpaDeviceNumber<T>(1.0), *alpha_dev);
    *alpha_dev = beta;
    }

}

template <typename T>
void cuda_hh_transform_FromC(T *alpha_dev, T *xnorm_sq_dev, T *xf_dev, T *tau_dev, int *index_in, int *wantDebug_in, cudaStream_t my_stream){
  int wantDebug = *wantDebug_in;

  dim3 blocks = dim3(1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

  // trivial single-thread kernel, streams can't be used here
  cuda_hh_transform_kernel<<<blocks,threadsPerBlock>>>(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug);

  if (wantDebug){
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess){
      printf("Error in executing cuda_hh_transform_kernel: %s\n",cudaGetErrorString(cuerr));
    }
  }
}

extern "C" void cuda_hh_transform_double_FromC(double *alpha_dev, double *xnorm_sq_dev, double *xf_dev, double *tau_dev, int *index_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_hh_transform_FromC(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, index_in, wantDebug_in, my_stream);
}

extern "C" void cuda_hh_transform_float_FromC(float *alpha_dev, float *xnorm_sq_dev, float *xf_dev, float *tau_dev, int *index_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_hh_transform_FromC(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, index_in, wantDebug_in, my_stream);
}

extern "C" void cuda_hh_transform_double_complex_FromC(cuDoubleComplex *alpha_dev, cuDoubleComplex *xnorm_sq_dev, cuDoubleComplex *xf_dev, cuDoubleComplex *tau_dev, int *index_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_hh_transform_FromC(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, index_in, wantDebug_in, my_stream);
}

extern "C" void cuda_hh_transform_float_complex_FromC(cuComplex *alpha_dev, cuComplex *xnorm_sq_dev, cuComplex *xf_dev, cuComplex *tau_dev, int *index_in, int *wantDebug_in, cudaStream_t my_stream){
  cuda_hh_transform_FromC(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, index_in, wantDebug_in, my_stream);
}

//________________________________________________________________

template <typename T>
__global__ void cuda_transpose_reduceadd_vectors_copy_block_kernel(T *aux_transpose_dev, T *vmat_st_dev, 
                                              int nvc, int nvr, int n_block, int nblks_skip, int nblks_tot, 
                                              int lcm_s_t, int nblk, int auxstride, int np_st, int ld_st, int direction, int isSkewsymmetric, int isReduceadd){
  int tid_x = threadIdx.x + blockIdx.x*blockDim.x;

/*
  ! direction = 1
  do lc=1,nvc
    do i = nblks_skip+n, nblks_tot-1, lcm_s_t
      k = (i - nblks_skip - n)/lcm_s_t * nblk + (lc - 1) * auxstride
      ns = (i/nps)*nblk ! local start of block i
      nl = min(nvr-i*nblk,nblk) ! length
      aux(k+1:k+nl) = vmat_s(ns+1:ns+nl,lc)
    enddo
  enddo

  ! direction = 2
  do lc=1,nvc
    do i = nblks_skip+n, nblks_tot-1, lcm_s_t
      k = (i - nblks_skip - n)/lcm_s_t * nblk + (lc - 1) * auxstride
      ns = (i/npt)*nblk ! local start of block i
      nl = min(nvr-i*nblk,nblk) ! length
#ifdef SKEW_SYMMETRIC_BUILD
      vmat_t(ns+1:ns+nl,lc) = - aux(k+1:k+nl)
#else
      vmat_t(ns+1:ns+nl,lc) = aux(k+1:k+nl)
#endif
    enddo
  enddo
*/

  T sign = elpaDeviceNumber<T>(1.0);
  if (isSkewsymmetric) sign = elpaDeviceNumber<T>(-1.0);

  int k, ns, nl;
  for (int lc=1; lc <= nvc; lc += 1)
    {
    for (int i = nblks_skip+n_block; i <= nblks_tot-1; i += lcm_s_t)
      {
      k = (i - nblks_skip - n_block)/lcm_s_t * nblk + (lc - 1) * auxstride;
      ns = (i/np_st)*nblk; // local start of block i
      nl = MIN(nvr-i*nblk, nblk); // length
      for (int j=tid_x; j<nl; j+=blockDim.x*gridDim.x) 
        {
        if (direction==1)                 aux_transpose_dev[k+1+j-1]            = vmat_st_dev[ns+1+j-1 + (lc-1)*ld_st];
        if (direction==2 && !isReduceadd) vmat_st_dev[ns+1+j-1 + (lc-1)*ld_st]  = elpaDeviceMultiply(sign, aux_transpose_dev[k+1+j-1]);
        if (direction==2 &&  isReduceadd) vmat_st_dev[ns+1+j-1 + (lc-1)*ld_st]  = elpaDeviceAdd(vmat_st_dev[ns+1+j-1 + (lc-1)*ld_st] , aux_transpose_dev[k+1+j-1]);
        }
      }
    }

}

template <typename T>
void cuda_transpose_reduceadd_vectors_copy_block_FromC(T *aux_transpose_dev, T *vmat_st_dev, 
                                              int *nvc_in, int *nvr_in,  int *n_block_in, int *nblks_skip_in, int *nblks_tot_in, 
                                              int *lcm_s_t_in, int *nblk_in, int *auxstride_in, int *np_st_in, int *ld_st_in, 
                                              int *direction_in, int* isSkewsymmetric_in, int* isReduceadd_in, int* wantDebug_in, int *SM_count_in, cudaStream_t my_stream){
  int nvc = *nvc_in;   
  int nvr = *nvr_in;   
  int n_block = *n_block_in;
  int nblks_skip = *nblks_skip_in;
  int nblks_tot = *nblks_tot_in;
  int lcm_s_t = *lcm_s_t_in;
  int nblk = *nblk_in;
  int auxstride = *auxstride_in;
  int np_st = *np_st_in;
  int ld_st = *ld_st_in;
  int direction = *direction_in;
  int isSkewsymmetric = *isSkewsymmetric_in;
  int isReduceadd = *isReduceadd_in;
  int wantDebug = *wantDebug_in;

  int SM_count = *SM_count_in;
  //int SM_count=32; // TODO_23_11: claim by function call and move outside
  int blocks = SM_count;

  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(nblk,1,1); 

  
#ifdef WITH_GPU_STREAMS
  cuda_transpose_reduceadd_vectors_copy_block_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(aux_transpose_dev, vmat_st_dev, 
                          nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride, np_st, ld_st, direction, isSkewsymmetric, isReduceadd);
#else
  cuda_transpose_reduceadd_vectors_copy_block_kernel<<<blocks,threadsPerBlock>>>(aux_transpose_dev, vmat_st_dev, 
                          nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride, np_st, ld_st, direction, isSkewsymmetric, isReduceadd);
#endif
  if(wantDebug)
    {
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) printf("Error in executing cuda_transpose_reduceadd_vectors_copy_block_kernel: %s\n",cudaGetErrorString(cuerr));
    }
}

extern "C" void cuda_transpose_reduceadd_vectors_copy_block_double_FromC(double *aux_transpose_dev, double *vmat_st_dev, 
                                                                        int *nvc_in, int *nvr_in,  int *n_block_in, int *nblks_skip_in, int *nblks_tot_in, 
                                                                        int *lcm_s_t_in, int *nblk_in, int *auxstride_in, int *np_st_in, int *ld_st_in, 
                                                                        int *direction_in, int* isSkewsymmetric_in, int* isReduceadd_in, int* wantDebug_in, int *SM_count_in, cudaStream_t my_stream){
  cuda_transpose_reduceadd_vectors_copy_block_FromC(aux_transpose_dev, vmat_st_dev, nvc_in, nvr_in, n_block_in, nblks_skip_in, nblks_tot_in, lcm_s_t_in, nblk_in, auxstride_in, np_st_in, ld_st_in, direction_in, isSkewsymmetric_in, isReduceadd_in, wantDebug_in, SM_count_in, my_stream);
}

extern "C" void cuda_transpose_reduceadd_vectors_copy_block_float_FromC(float *aux_transpose_dev, float *vmat_st_dev, 
                                                                       int *nvc_in, int *nvr_in,  int *n_block_in, int *nblks_skip_in, int *nblks_tot_in, 
                                                                       int *lcm_s_t_in, int *nblk_in, int *auxstride_in, int *np_st_in, int *ld_st_in, 
                                                                       int *direction_in, int* isSkewsymmetric_in, int* isReduceadd_in, int* wantDebug_in, int* SM_count_in, cudaStream_t my_stream){
  cuda_transpose_reduceadd_vectors_copy_block_FromC(aux_transpose_dev, vmat_st_dev, nvc_in, nvr_in, n_block_in, nblks_skip_in, nblks_tot_in, lcm_s_t_in, nblk_in, auxstride_in, np_st_in, ld_st_in, direction_in, isSkewsymmetric_in, isReduceadd_in, wantDebug_in, SM_count_in, my_stream);
}

extern "C" void cuda_transpose_reduceadd_vectors_copy_block_double_complex_FromC(cuDoubleComplex *aux_transpose_dev, cuDoubleComplex *vmat_st_dev, 
                                                                        int *nvc_in, int *nvr_in,  int *n_block_in, int *nblks_skip_in, int *nblks_tot_in, 
                                                                        int *lcm_s_t_in, int *nblk_in, int *auxstride_in, int *np_st_in, int *ld_st_in, 
                                                                        int *direction_in, int* isSkewsymmetric_in, int* isReduceadd_in, int* wantDebug_in, int *SM_count_in, cudaStream_t my_stream){
  cuda_transpose_reduceadd_vectors_copy_block_FromC(aux_transpose_dev, vmat_st_dev, nvc_in, nvr_in, n_block_in, nblks_skip_in, nblks_tot_in, lcm_s_t_in, nblk_in, auxstride_in, np_st_in, ld_st_in, direction_in, isSkewsymmetric_in, isReduceadd_in, wantDebug_in, SM_count_in, my_stream);
}

extern "C" void cuda_transpose_reduceadd_vectors_copy_block_float_complex_FromC(cuComplex *aux_transpose_dev, cuComplex *vmat_st_dev, 
                                                                        int *nvc_in, int *nvr_in,  int *n_block_in, int *nblks_skip_in, int *nblks_tot_in, 
                                                                        int *lcm_s_t_in, int *nblk_in, int *auxstride_in, int *np_st_in, int *ld_st_in, 
                                                                        int *direction_in, int* isSkewsymmetric_in, int* isReduceadd_in, int* wantDebug_in, int *SM_count_in, cudaStream_t my_stream){
  cuda_transpose_reduceadd_vectors_copy_block_FromC(aux_transpose_dev, vmat_st_dev, nvc_in, nvr_in, n_block_in, nblks_skip_in, nblks_tot_in, lcm_s_t_in, nblk_in, auxstride_in, np_st_in, ld_st_in, direction_in, isSkewsymmetric_in, isReduceadd_in, wantDebug_in, SM_count_in, my_stream);
}

//________________________________________________________________
