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
#include "config-f90.h"

#define MAX_THREADS_PER_BLOCK 1024

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

template <typename T> 
__device__ T sign(T a, T b) {
    if (b>=0) return fabs(a);
    else return -fabs(a);
}

/*
template <typename T>
void sycl_copy_a_tmat2_kernel(T *a_dev, T *tmat2_dev, const int nblk, const int matrixRows, const int l_colx, const int l_row1, sycl::nd_item<1> it){

  int nb_index = it.get_local_id(0) + 1; // range 1..nb
  int l_col_index = it.get_group(0) + 1; // range 1..l_colx-l_cols-1

  tmat2_dev[nb_index-1 + (l_colx-1 + l_col_index -1) * nblk] = a_dev[l_row1-1 + nb_index-1 + (l_colx-1 + l_col_index -1)  * matrixRows];

}

template <typename T>
void sycl_copy_a_tmat2_FromC(T *a_dev, T *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, intptr_t my_stream){

  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;
  int nb     = *nb_in;

  sycl::range<1> global_range = sycl::range<1>(nb*(l_cols - l_colx + 1));
  sycl::range<1> local_range  = sycl::range<1>(nb);

  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  queue.parallel_for(
      sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> it) {
        sycl_copy_a_tmat2_kernel(a_dev, tmat2_dev, nblk, matrixRows,
                                        l_colx, l_row1, it);
      });
  queue.wait_and_throw();

}

extern "C" void sycl_copy_double_a_tmat2_FromC(double *a_dev, double *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, intptr_t my_stream){
  sycl_copy_a_tmat2_FromC(a_dev, tmat2_dev, nblk_in, matrixRows_in, l_cols_in, l_colx_in, l_row1_in, nb_in, my_stream);
}

extern "C" void sycl_copy_float_a_tmat2_FromC(float *a_dev, float *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, intptr_t my_stream){
  sycl_copy_a_tmat2_FromC(a_dev, tmat2_dev, nblk_in, matrixRows_in, l_cols_in, l_colx_in, l_row1_in, nb_in, my_stream);
}

extern "C" void sycl_copy_double_complex_a_tmat2_FromC(std::complex<double> *a_dev, std::complex<double> *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, intptr_t my_stream){
  sycl_copy_a_tmat2_FromC(a_dev, tmat2_dev, nblk_in, matrixRows_in, l_cols_in, l_colx_in, l_row1_in, nb_in, my_stream);
}

extern "C" void sycl_copy_float_complex_a_tmat2_FromC(std::complex<float> *a_dev, std::complex<float> *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, intptr_t my_stream){
  sycl_copy_a_tmat2_FromC(a_dev, tmat2_dev, nblk_in, matrixRows_in, l_cols_in, l_colx_in, l_row1_in, nb_in, my_stream);
}
*/

//________________________________________________________________
// device syncronization is needed afterwards, e.g. gpu_memcpy

__global__ void cuda_dot_product_double_kernel(int n, double *x_dev, int incx, double *y_dev, int incy, double *result_dev){
  __shared__ double cache[MAX_THREADS_PER_BLOCK]; // extra space of fixed size is reserved for a speedup
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

  if (threadIdx.x==0) result_dev[0] = 0; // clear old value // PETERDEBUG: move this to other kernel for thread safety

  double temp = 0;
  int i = tid;
  while (i < n) {
    temp += x_dev[i*incx] * y_dev[i*incy];
    i += blockDim.x * gridDim.x;
  }

  // set the cache values
  cache[threadIdx.x] = temp;
  // synchronize threads in this block
  __syncthreads();

  // for reductions, threadsPerBlock=blockDim.x must be a power of 2
  i = blockDim.x/2;
  while (i > 0) {
    if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
    __syncthreads();
    i /= 2;
  }

  if (threadIdx.x==0) atomicAdd(&result_dev[0], cache[0]);
  
}

extern "C" void cuda_dot_product_double_FromC(int* n_in, double *x_dev, int *incx_in, double *y_dev, int *incy_in, double *result_dev, bool *wantDebug_in, cudaStream_t my_stream){
  int n = *n_in;   
  int incx = *incx_in;
  int incy = *incy_in;
  bool wantDebug = *wantDebug_in;

  int SM_count=32;
  //cudaDeviceGetAttribute(&SM_count, cudaDevAttrMultiProcessorCount, 0); // PETERDEBUG move this outside, to set_gpu, claim the number only once during GPU setup

  int blocks = SM_count;
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1); // PETERDEBUG: or NB?

#ifdef WITH_GPU_STREAMS
  cuda_dot_product_double_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(n, x_dev, incx, y_dev, incy, result_dev);
#else
  cuda_dot_product_double_kernel<<<blocks,threadsPerBlock>>>(n, x_dev, incx, y_dev, incy, result_dev);
#endif
  if (wantDebug){
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess){
      printf("Error in executing cuda_dot_product_kernel: %s\n",cudaGetErrorString(cuerr));
    }
  }
}

//________________________________________________________________

__global__ void cuda_dot_product_and_assign_double_kernel(double *v_row_dev, int l_rows, int isOurProcessRow, double *aux1_dev){
  const int threadsPerBlock = MAX_THREADS_PER_BLOCK;
  __shared__ double cache[threadsPerBlock];
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
  if (threadIdx.x==0) aux1_dev[0] = 0; // clear old value // PETERDEBUG: move this to other kernel for thread safety

  double temp = 0;
  int index_global = tid;
  while (index_global < l_rows-1) {
    temp += v_row_dev[index_global] * v_row_dev[index_global];
    index_global += blockDim.x * gridDim.x;
  }

  // set the cache values
  cache[threadIdx.x] = temp;
  // synchronize threads in this block
  __syncthreads();

  // for reductions, threadsPerBlock must be a power of 2
  int i = blockDim.x/2;
  while (i > 0) {
    if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
    __syncthreads();
    i /= 2;
  }

  //if (threadIdx.x==0) dot_prod_partial[blockIdx.x] += cache[0];
  if (threadIdx.x==0) atomicAdd(&aux1_dev[0], cache[0]);
  
  if (tid==0)
    {
    if (isOurProcessRow) 
      {
      aux1_dev[1] = v_row_dev[l_rows-1];
      }
    else
      {
      if (l_rows>0) atomicAdd(&aux1_dev[0], v_row_dev[l_rows-1]*v_row_dev[l_rows-1]);
      aux1_dev[1] = 0;
      }
    }
}

extern "C" void cuda_dot_product_and_assign_double_FromC(double *v_row_dev, int *l_rows_in, int *isOurProcessRow_in, double *aux1_dev, bool *wantDebug_in, cudaStream_t my_stream){
  int l_rows = *l_rows_in;   
  int isOurProcessRow = *isOurProcessRow_in;
  bool wantDebug = *wantDebug_in;
  //double dot_prod = *dot_prod_in;
  //double v_row_last = *v_row_last_in;

  //int numSMs;
  //cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
  
  //int blocks = (l_rows+1023)/MAX_THREADS_PER_BLOCK;
  int blocks = 32; // PETERDEBUG: change blocksPerGrid to number of SM's (108 fo A100) and threadsPerBlock to max threads per block. claim the number only once during GPU setup
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);

  
  // zero-copy version
  //double *dot_prod_partial, *dot_prod_partial_dev;
  //cudaSetDeviceFlags( cudaDeviceMapHost ); // move this outside of the function
  //cudaHostAlloc( (void**)&dot_prod_partial, blocks*sizeof(double), cudaHostAllocMapped ); // zero-copy buffer
  //cudaHostGetDevicePointer( &dot_prod_partial_dev, dot_prod_partial, 0 );

  //double *dot_prod_partial_managed;
  //cudaMallocManaged(&dot_prod_dev, blocks*sizeof(double));  

  //double *dot_prod_partial;
  //cudaHostAlloc(&dot_prod_partial, sizeof(dot_prod_partial[0])*blocks, cudaHostAllocDefault);

#ifdef WITH_GPU_STREAMS
  cuda_dot_product_and_assign_double_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(v_row_dev, l_rows, isOurProcessRow, aux1_dev);
#else
  cuda_dot_product_and_assign_double_kernel<<<blocks,threadsPerBlock>>>(v_row_dev, l_rows, isOurProcessRow, aux1_dev);
#endif
  if (wantDebug){
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess){
      printf("Error in executing cuda_dot_product_and_assign_kernel: %s\n",cudaGetErrorString(cuerr));
    }
  }
/*
  double dot_prod=0;
  for (int i=0; i<blocks; i++)
  {
    dot_prod += dot_prod_partial[i];
  }
*/
}

//________________________________________________________________

__global__ void cuda_set_e_vec_scale_set_one_store_v_row_double_kernel(double *e_vec_dev, double *vrl_dev, double *a_dev, double *v_row_dev, double *tau_dev, double *xf_host_or_dev, 
                                                      int l_rows, int l_cols,  int matrixRows, int istep, bool isOurProcessRow, bool useCCL){
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

  if (useCCL && tid==0)
    {
    if (isOurProcessRow) e_vec_dev[istep-1-1] = *vrl_dev;
    v_row_dev[l_rows+1-1] = tau_dev[istep-1];
    }

  int index_global = tid;
  while (index_global < l_rows) {
    v_row_dev[index_global] *= (*xf_host_or_dev);
    index_global += blockDim.x * gridDim.x;
  }

  if (isOurProcessRow && index_global - blockDim.x*gridDim.x == l_rows-1) // last element
    {
    v_row_dev[l_rows-1] = 1.0;
    }
    
  int i_row = tid;
  while (i_row < l_rows) {
    a_dev[i_row + matrixRows*l_cols] = v_row_dev[i_row];
    i_row += blockDim.x * gridDim.x;
  }


}

extern "C" void cuda_set_e_vec_scale_set_one_store_v_row_double_FromC(double *e_vec_dev, double *vrl_dev, double *a_dev, double *v_row_dev, double *tau_dev, double *xf_host_or_dev, 
                                              int *l_rows_in, int *l_cols_in,  int *matrixRows_in, int *istep_in, bool *isOurProcessRow_in, bool *useCCL_in, bool *wantDebug_in, cudaStream_t my_stream){
  int l_rows = *l_rows_in;   
  int l_cols = *l_cols_in;   
  int matrixRows = *matrixRows_in;
  int istep = *istep_in;
  bool isOurProcessRow = *isOurProcessRow_in;
  bool useCCL = *useCCL_in;
  bool wantDebug = *wantDebug_in;

  int blocks = std::max((l_rows+MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK, 1);
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1); // PETERDEBUG change to NB

  
#ifdef WITH_GPU_STREAMS
  cuda_set_e_vec_scale_set_one_store_v_row_double_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev,
                                                                                                 l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL);
#else
  cuda_set_e_vec_scale_set_one_store_v_row_double_kernel<<<blocks,threadsPerBlock>>>(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev,
                                                                                    l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL);
#endif
  if (wantDebug){
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess){
      printf("Error in executing cuda_set_e_vec_scale_set_one_store_v_row_double_kernel: %s\n",cudaGetErrorString(cuerr));
    }
  }
}

//________________________________________________________________

__global__ void cuda_store_u_v_in_uv_vu_double_kernel(double *vu_stored_rows_dev, double *uv_stored_cols_dev, double *v_row_dev, double *u_row_dev,
                double *v_col_dev, double *u_col_dev, double *tau_dev, double* vav_host_or_dev, double *tau_host_or_dev,
                int l_rows, int l_cols, int n_stored_vecs, int max_local_rows, int max_local_cols, int istep, bool useCCL){
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

  double conjg_tau = *tau_host_or_dev; // real (double) case only so far

  // recover tau_dev(istep) after broadcasting
  if (useCCL && tid==0)
    {
    tau_dev[istep-1] = v_row_dev[l_rows+1-1];
    }

/*
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
*/
  double vav = vav_host_or_dev[0];

  int i_row = tid;
  while (i_row < l_rows) {
    vu_stored_rows_dev[i_row + max_local_rows*(2*n_stored_vecs+0)] = conjg_tau*v_row_dev[i_row];
    vu_stored_rows_dev[i_row + max_local_rows*(2*n_stored_vecs+1)] = 0.5*conjg_tau*vav*v_row_dev[i_row]-u_row_dev[i_row];
    i_row += blockDim.x * gridDim.x;
  }

  int i_col = tid;
  while (i_col < l_cols) {
    uv_stored_cols_dev[i_col + max_local_cols*(2*n_stored_vecs+0)] = 0.5*conjg_tau*vav*v_col_dev[i_col]-u_col_dev[i_col];
    uv_stored_cols_dev[i_col + max_local_cols*(2*n_stored_vecs+1)] = conjg_tau*v_col_dev[i_col];
    i_col += blockDim.x * gridDim.x;
  }


}


extern "C" void cuda_store_u_v_in_uv_vu_double_FromC(double *vu_stored_rows_dev, double *uv_stored_cols_dev, double *v_row_dev, double *u_row_dev,
                double *v_col_dev, double *u_col_dev, double *tau_dev, double *vav_host_or_dev, double *tau_host_or_dev,
                int *l_rows_in, int *l_cols_in, int *n_stored_vecs_in, int *max_local_rows_in, int *max_local_cols_in, int *istep_in, bool *useCCL_in, bool *wantDebug_in, cudaStream_t my_stream){
  int l_rows = *l_rows_in;   
  int l_cols = *l_cols_in;   
  int n_stored_vecs  = *n_stored_vecs_in;
  int max_local_rows = *max_local_rows_in;   
  int max_local_cols = *max_local_cols_in;   
  int istep = *istep_in;   
  bool useCCL = *useCCL_in;
  bool wantDebug = *wantDebug_in;
  
  int blocks = std::max({(l_rows+MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK, (l_cols+MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK, 1});
  
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);

  
#ifdef WITH_GPU_STREAMS
  cuda_store_u_v_in_uv_vu_double_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, 
                                       v_col_dev, u_col_dev, tau_dev, vav_host_or_dev, tau_host_or_dev, l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, useCCL);
#else
  cuda_store_u_v_in_uv_vu_double_kernel<<<blocks,threadsPerBlock>>>(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, 
                                       v_col_dev, u_col_dev, tau_dev, vav_host_or_dev, tau_host_or_dev, l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, useCCL);
#endif
  if (wantDebug){
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess){
      printf("Error in executing cuda_store_u_v_in_uv_vu_double_kernel: %s\n",cudaGetErrorString(cuerr));
    }
  }
}

//________________________________________________________________


__global__ void cuda_update_matrix_element_add_double_kernel(double *vu_stored_rows_dev, double *uv_stored_cols_dev, double *a_dev, double *d_vec_dev, 
                                                            int l_rows, int l_cols, int matrixRows, int max_local_rows, int max_local_cols, int istep, int n_stored_vecs, bool isSkewsymmetric){
  
  const int threadsPerBlock = MAX_THREADS_PER_BLOCK;
  __shared__ double cache[threadsPerBlock];
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

  if (threadIdx.x==0)
    { 
    if (isSkewsymmetric) 
      d_vec_dev[istep-1-1] = 0;
    else 
      d_vec_dev[istep-1-1] = a_dev[(l_rows-1) + matrixRows*(l_cols-1)]; // set initial value // PETERDEBUG: move this to other kernel for thread safety
    }
  if (n_stored_vecs > 0)
    {

    double temp = 0;
    int index_n = tid;
    while (index_n < 2*n_stored_vecs) 
      {
      temp += vu_stored_rows_dev[(l_rows-1)+max_local_rows*index_n] * uv_stored_cols_dev[(l_cols-1)+max_local_cols*index_n];
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
      if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
      __syncthreads();
      i /= 2;
      }

    if (threadIdx.x==0) 
      {
      atomicAdd(&a_dev[(l_rows-1) + matrixRows*(l_cols-1)], cache[0]);
      if (!isSkewsymmetric) atomicAdd(&d_vec_dev[istep-1-1], cache[0]);
      }
    }
/*
#endif
#if COMPLEXCASE == 1
      d_vec(istep-1) = real(a_mat(l_rows,l_cols),kind=rk)
#endif
*/
}


extern "C" void cuda_update_matrix_element_add_double_FromC(double *vu_stored_rows_dev, double *uv_stored_cols_dev, double *a_dev, double *d_vec_dev, 
                                                            int *l_rows_in, int *l_cols_in, int *matrixRows_in, int *max_local_rows_in, int *max_local_cols_in, int *istep_in, int *n_stored_vecs_in, 
                                                            bool* isSkewsymmetric_in, bool *wantDebug_in, cudaStream_t my_stream){
  int l_rows = *l_rows_in;   
  int l_cols = *l_cols_in;
  int matrixRows = *matrixRows_in;
  int max_local_rows = *max_local_rows_in;
  int max_local_cols = *max_local_cols_in;
  int istep = *istep_in;   
  int n_stored_vecs = *n_stored_vecs_in; 
  bool isSkewsymmetric = *isSkewsymmetric_in;   
  bool wantDebug = *wantDebug_in;
  
  int blocks = std::min((2*n_stored_vecs+MAX_THREADS_PER_BLOCK-1)/MAX_THREADS_PER_BLOCK, 32);
  if (n_stored_vecs==0) blocks=1;
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_update_matrix_element_add_double_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                                  l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                                  isSkewsymmetric);
#else
  cuda_update_matrix_element_add_double_kernel<<<blocks,threadsPerBlock>>>(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, 
                                                  l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, 
                                                  isSkewsymmetric);
#endif
  if (wantDebug){
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess){
      printf("Error in executing cuda_update_matrix_element_add_double_kernel: %s\n",cudaGetErrorString(cuerr));
    }
  }
}

//________________________________________________________________

__global__ void cuda_update_array_element_double_kernel(double *array_dev, const int index, double value){

  array_dev[index-1] = value;

}

extern "C" void cuda_update_array_element_double_FromC(double *array_dev, int *index_in, double *value_in, cudaStream_t my_stream){
  int index = *index_in;   
  double value = *value_in;

  dim3 blocks = dim3(1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_update_array_element_double_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(array_dev, index, value);
#else
  cuda_update_array_element_double_kernel<<<blocks,threadsPerBlock>>>(array_dev, index, value);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_update_array_element_double_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

//________________________________________________________________

__global__ void cuda_hh_transform_double_kernel(double *alpha_dev, double *xnorm_sq_dev, double *xf_dev, double *tau_dev, bool wantDebug_in){

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


  if (*xnorm_sq_dev==0.0)
    {
    if (*alpha_dev >= 0.0) *tau_dev = 0.0;
    else
      {
      *tau_dev = 2.0;
      *alpha_dev = - (*alpha_dev);
      }
    
    *xf_dev = 0.0;
    }

  else
    {
    double beta = sign( sqrt( (*alpha_dev)*(*alpha_dev) + *xnorm_sq_dev ), *alpha_dev);

    *alpha_dev = *alpha_dev + beta;
    
    if (beta<0)
      {
      beta = -beta;
      *tau_dev  = - (*alpha_dev) / beta;
      }
    else
      {
      *alpha_dev = (*xnorm_sq_dev) / (*alpha_dev);
      *tau_dev = (*alpha_dev) / beta;
      *alpha_dev = - (*alpha_dev);
      }

    *xf_dev = 1.0/(*alpha_dev);
    *alpha_dev = beta;
    }

}

extern "C" void cuda_hh_transform_double_FromC(double *alpha_dev, double *xnorm_sq_dev, double *xf_dev, double *tau_dev, int *index_in, bool *wantDebug_in, cudaStream_t my_stream){
  bool wantDebug = *wantDebug_in;

  dim3 blocks = dim3(1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_hh_transform_double_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug);
#else
  cuda_hh_transform_double_kernel<<<blocks,threadsPerBlock>>>(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug);
#endif

  if (wantDebug){
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess){
      printf("Error in executing cuda_hh_transform_double_kernel: %s\n",cudaGetErrorString(cuerr));
    }
  }
}

//________________________________________________________________

__global__ void cuda_transpose_reduceadd_vectors_copy_block_double_kernel(double *aux_transpose_dev, double *vmat_st_dev, 
                                              int nvc, int nvr, int n_block, int nblks_skip, int nblks_tot, 
                                              int lcm_s_t, int nblk, int auxstride, int np_st, int ld_st, int direction, bool isSkewsymmetric, bool isReduceadd){
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

  int sign = 1;
  if (isSkewsymmetric) sign = -1;

  if (isReduceadd) printf("aux_transpose_dev[0] (before)= %f\n", *aux_transpose_dev); // ! PETERDEBUG: delete after testing

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
        if (direction==2 && !isReduceadd) vmat_st_dev[ns+1+j-1 + (lc-1)*ld_st]  = sign*aux_transpose_dev[k+1+j-1];
        if (direction==2 &&  isReduceadd) vmat_st_dev[ns+1+j-1 + (lc-1)*ld_st]  = vmat_st_dev[ns+1+j-1 + (lc-1)*ld_st] + aux_transpose_dev[k+1+j-1];
        }
      }
    }

  if (isReduceadd) printf("aux_transpose_dev[0] (after) = %f\n", *aux_transpose_dev); // ! PETERDEBUG: delete after testing

}

extern "C" void cuda_transpose_reduceadd_vectors_copy_block_double_FromC(double *aux_transpose_dev, double *vmat_st_dev, 
                                              int *nvc_in, int *nvr_in,  int *n_block_in, int *nblks_skip_in, int *nblks_tot_in, 
                                              int *lcm_s_t_in, int *nblk_in, int *auxstride_in, int *np_st_in, int *ld_st_in, 
                                              int *direction_in, bool* isSkewsymmetric_in, bool* isReduceadd_in, bool* wantDebug_in, cudaStream_t my_stream){
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
  bool isSkewsymmetric = *isSkewsymmetric_in;
  bool isReduceadd = *isReduceadd_in;
  bool wantDebug = *wantDebug_in;

  int SM_count=32; // PETERDEBUG count and move outside
  int blocks = SM_count;

  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(nblk,1,1); 

  
#ifdef WITH_GPU_STREAMS
  cuda_transpose_reduceadd_vectors_copy_block_double_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(aux_transpose_dev, vmat_st_dev, 
                          nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride, np_st, ld_st, direction, isSkewsymmetric, isReduceadd);
#else
  cuda_transpose_reduceadd_vectors_copy_block_double_kernel<<<blocks,threadsPerBlock>>>(aux_transpose_dev, vmat_st_dev, 
                          nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride, np_st, ld_st, direction, isSkewsymmetric, isReduceadd);
#endif
  if(wantDebug)
    {
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) printf("Error in executing cuda_transpose_reduceadd_vectors_copy_block_double_kernel: %s\n",cudaGetErrorString(cuerr));
    }
}

//________________________________________________________________