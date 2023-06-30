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
#include <iostream>
#include <algorithm>
#include "config-f90.h"

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

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

__global__ void cuda_dot_product_and_assign_double_kernel(double *v_row_dev, int l_rows, int isOurProcessRow, double *aux1_dev){
  const int threadsPerBlock = 1024;
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
      atomicAdd(&aux1_dev[0], v_row_dev[l_rows-1]*v_row_dev[l_rows-1]);
      aux1_dev[1] = 0;
      }
    }
}

extern "C" void cuda_dot_product_and_assign_double_FromC(double *v_row_dev, int *l_rows_in, int *isOurProcessRow_in, double *aux1_dev, cudaStream_t my_stream){
  int l_rows = *l_rows_in;   
  int isOurProcessRow = *isOurProcessRow_in;
  //double dot_prod = *dot_prod_in;
  //double v_row_last = *v_row_last_in;

  
  //int blocks = (l_rows+1023)/1024;
  int blocks = 32;
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(1024,1,1);

  
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
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_dot_product_and_assign_kernel: %s\n",cudaGetErrorString(cuerr));
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

__global__ void cuda_scale_set_one_store_v_row_double_kernel(double *a_dev, double *v_row_dev, int l_rows, int l_cols, int matrixRows, int isOurProcessRow, double xf){
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

/*
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
*/

  int index_global = tid;
  while (index_global < l_rows) {
    v_row_dev[index_global] *= xf;
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

extern "C" void cuda_scale_set_one_store_v_row_double_FromC(double *a_dev, double *v_row_dev, int *l_rows_in, int *l_cols_in,  int *matrixRows_in, int *isOurProcessRow_in, double *xf_in, cudaStream_t my_stream){
  int l_rows = *l_rows_in;   
  int l_cols = *l_cols_in;   
  int matrixRows = *matrixRows_in;
  int isOurProcessRow = *isOurProcessRow_in;
  double xf = *xf_in;

  if (l_rows==0) return;
  
  int blocks = std::min((l_rows+1023)/1024, 2147483647);
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(1024,1,1);

  
#ifdef WITH_GPU_STREAMS
  cuda_scale_set_one_store_v_row_double_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(a_dev, v_row_dev, l_rows, l_cols, matrixRows, isOurProcessRow, xf);
#else
  cuda_scale_set_one_store_v_row_double_kernel<<<blocks,threadsPerBlock>>>(a_dev, v_row_dev, l_rows, l_cols, matrixRows, isOurProcessRow, xf);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_scale_set_one_store_v_row_double_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

//________________________________________________________________

__global__ void cuda_store_u_v_in_uv_vu_double_kernel(double *vu_stored_rows_dev, double *uv_stored_cols_dev, 
                double *v_row_dev, double *u_row_dev, double *v_col_dev, double *u_col_dev, double* vav_host_or_dev,
                int l_rows, int l_cols, int n_stored_vecs, int max_local_rows, int max_local_cols, double conjg_tau){
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

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


extern "C" void cuda_store_u_v_in_uv_vu_double_FromC(double *vu_stored_rows_dev, double *uv_stored_cols_dev, 
                double *v_row_dev, double *u_row_dev, double *v_col_dev, double *u_col_dev, double *vav_host_or_dev,
                int *l_rows_in, int *l_cols_in, int *n_stored_vecs_in, int *max_local_rows_in, int *max_local_cols_in, double *conjg_tau_in, cudaStream_t my_stream){
  int l_rows = *l_rows_in;   
  int l_cols = *l_cols_in;   
  int n_stored_vecs  = *n_stored_vecs_in;
  int max_local_rows = *max_local_rows_in;   
  int max_local_cols = *max_local_cols_in;   
  double conjg_tau = *conjg_tau_in;
  //double vav_host = *vav_host_in;

  
  int blocks = std::max((l_rows+1023)/1024, (l_cols+1023)/1024);
  if (blocks==0) return;
  blocks = std::min(blocks, 2147483647);
  
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(1024,1,1);

  
#ifdef WITH_GPU_STREAMS
  cuda_store_u_v_in_uv_vu_double_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(vu_stored_rows_dev, uv_stored_cols_dev,
                                       v_row_dev, u_row_dev, v_col_dev, u_col_dev, vav_host_or_dev, l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, conjg_tau);
#else
  cuda_store_u_v_in_uv_vu_double_kernel<<<blocks,threadsPerBlock>>>(vu_stored_rows_dev, uv_stored_cols_dev,
                                       v_row_dev, u_row_dev, v_col_dev, u_col_dev, vav_host_or_dev, l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, conjg_tau);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_store_u_v_in_uv_vu_double_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

//________________________________________________________________


__global__ void cuda_update_matrix_element_add_double_kernel(double *vu_stored_rows_dev, double *uv_stored_cols_dev, double *a_dev, double *d_vec_dev, 
                                                            int l_rows, int l_cols, int matrixRows, int max_local_rows, int max_local_cols, int istep, int n_stored_vecs, int isSkewsymmetric){
  
  const int threadsPerBlock = 1024;
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
                                                            int* isSkewsymmetric_in, cudaStream_t my_stream){
  int l_rows = *l_rows_in;   
  int l_cols = *l_cols_in;
  int matrixRows = *matrixRows_in;
  int max_local_rows = *max_local_rows_in;
  int max_local_cols = *max_local_cols_in;
  int istep = *istep_in;   
  int n_stored_vecs = *n_stored_vecs_in; 
  int isSkewsymmetric = *isSkewsymmetric_in;   

  
  int blocks = std::min((2*n_stored_vecs+1023)/1024, 32);
  if (n_stored_vecs==0) blocks=1;
  dim3 blocksPerGrid = dim3(blocks,1,1);
  dim3 threadsPerBlock = dim3(1024,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_update_matrix_element_add_double_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                                  l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                                  isSkewsymmetric);
#else
  cuda_update_matrix_element_add_double_kernel<<<blocks,threadsPerBlock>>>(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, 
                                                  l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, 
                                                  isSkewsymmetric);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_update_matrix_element_add_double_kernel: %s\n",cudaGetErrorString(cuerr));
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
