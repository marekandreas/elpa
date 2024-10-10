//    Copyright 2023, A. Marek
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
//    This file was written by A. Marek, MPCDF

// PETERDEBUG: split this file in two .h files and one .cu file
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
#include "config-f90.h"
//#include "../../../GPU/common_device_functions.h"

#define MAX_THREADS_PER_BLOCK 1024

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

// PETERDEBUG: clean up after merge. Uncommnent //#include "../../../GPU/common_device_functions.h" above
// construct a generic double/float/double_complex/float_complex from a double
template <typename T> __forceinline__ __device__ T elpaDeviceNumber(double number);
template <>  __forceinline__ __device__ double elpaDeviceNumber<double>(double number) {return number;}
template <>  __forceinline__ __device__ float  elpaDeviceNumber<float> (double number) {return (float) number;}
template <>  __forceinline__ __device__ cuDoubleComplex elpaDeviceNumber<cuDoubleComplex>(double number) {return make_cuDoubleComplex (number, 0.0);}
template <>  __forceinline__ __device__ cuFloatComplex elpaDeviceNumber<cuFloatComplex> (double number) {return make_cuFloatComplex ((float) number, 0.0);}


__global__ void cuda_copy_double_tmp2_c_kernel(double *tmp2_dev, double *c_dev, const int nr_done, const int nstor, const int lcs, const int lce, const int ldc, const int ldcCols){

  //dim3 blocks = dim3(lce-lcs+1,1,1);
  //dim3 threadsPerBlock = dim3(nr_done+nstor-(nr_done+1)+1,1,1);

  int i_index    = threadIdx.x +1; // range 1..nstor
  int j_index = blockIdx.x + 1; // range 1..lce-lse+1
  //c_dev[nr_done+(i_index-1) + ldc*(lcs-1+j_index-1)] = tmp2_dev[0+(i_index-1)+nstor*(lcs-1+j_index-1)];
  //base 1 index
  c_dev[nr_done+(i_index-1) + ldc*(lcs-1+j_index-1)] = tmp2_dev[0+(i_index-1)+nstor*(j_index-1)];

}

extern "C" void cuda_copy_double_tmp2_c_FromC(double *tmp2_dev, double *c_dev, int *nr_done_in, int *nstor_in, int *lcs_in, int *lce_in, int *ldc_in, int *ldcCols_in, cudaStream_t my_stream) { 
		
  int nr_done = *nr_done_in;   
  int nstor = *nstor_in;
  int lcs = *lcs_in;
  int lce = *lce_in;
  int ldc = *ldc_in;
  int ldcCols = *ldcCols_in;

  dim3 blocks = dim3(lce-lcs+1,1,1);
  dim3 threadsPerBlock = dim3(nr_done+nstor-(nr_done+1)+1,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_copy_double_tmp2_c_kernel<<<blocks,threadsPerBlock, 0, my_stream>>>(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols);
#else
  cuda_copy_double_tmp2_c_kernel<<<blocks,threadsPerBlock>>>(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing copy_double_tmp2_c_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_float_tmp2_c_kernel(float *tmp2_dev, float *c_dev, const int nr_done, const int nstor, const int lcs, const int lce, const int ldc, const int ldcCols){

  //dim3 blocks = dim3(lce-lcs+1,1,1);
  //dim3 threadsPerBlock = dim3(nr_done+nstor-(nr_done+1)+1,1,1);

  int i_index    = threadIdx.x +1; // range 1..nstor
  int j_index = blockIdx.x + 1; // range 1..lce-lse+1
  //c_dev[nr_done+(i_index-1) + ldc*(lcs-1+j_index-1)] = tmp2_dev[0+(i_index-1)+nstor*(lcs-1+j_index-1)];
  c_dev[nr_done+(i_index-1) + ldc*(lcs-1+j_index-1)] = tmp2_dev[0+(i_index-1)+nstor*(j_index-1)];
}

extern "C" void cuda_copy_float_tmp2_c_FromC(float *tmp2_dev, float *c_dev, int *nr_done_in, int *nstor_in, int *lcs_in, int *lce_in, int *ldc_in, int *ldcCols_in, cudaStream_t my_stream) { 
		
  int nr_done = *nr_done_in;   
  int nstor = *nstor_in;
  int lcs = *lcs_in;
  int lce = *lce_in;
  int ldc = *ldc_in;
  int ldcCols = *ldcCols_in;

  dim3 blocks = dim3(lce-lcs+1,1,1);
  dim3 threadsPerBlock = dim3(nr_done+nstor-(nr_done+1)+1,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_copy_float_tmp2_c_kernel<<<blocks,threadsPerBlock, 0, my_stream>>>(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols);
#else
  cuda_copy_float_tmp2_c_kernel<<<blocks,threadsPerBlock>>>(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing copy_float_tmp2_c_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_double_complex_tmp2_c_kernel(cuDoubleComplex *tmp2_dev, cuDoubleComplex *c_dev, const int nr_done, const int nstor, const int lcs, const int lce, const int ldc, const int ldcCols){

  //dim3 blocks = dim3(lce-lcs+1,1,1);
  //dim3 threadsPerBlock = dim3(nr_done+nstor-(nr_done+1)+1,1,1);

  int i_index    = threadIdx.x +1; // range 1..nstor
  int j_index = blockIdx.x + 1; // range 1..lce-lse+1
  //c_dev[nr_done+(i_index-1) + ldc*(lcs-1+j_index-1)] = tmp2_dev[0+(i_index-1)+nstor*(lcs-1+j_index-1)];
  c_dev[nr_done+(i_index-1) + ldc*(lcs-1+j_index-1)] = tmp2_dev[0+(i_index-1)+nstor*(j_index-1)];
}

extern "C" void cuda_copy_double_complex_tmp2_c_FromC(double _Complex *tmp2_dev, double _Complex *c_dev, int *nr_done_in, int *nstor_in, int *lcs_in, int *lce_in, int *ldc_in, int *ldcCols_in, cudaStream_t my_stream) { 
		
  int nr_done = *nr_done_in;   
  int nstor = *nstor_in;
  int lcs = *lcs_in;
  int lce = *lce_in;
  int ldc = *ldc_in;
  int ldcCols = *ldcCols_in;

  dim3 blocks = dim3(lce-lcs+1,1,1);
  dim3 threadsPerBlock = dim3(nr_done+nstor-(nr_done+1)+1,1,1);

  cuDoubleComplex* tmp2_casted = (cuDoubleComplex*) tmp2_dev;
  cuDoubleComplex* c_casted = (cuDoubleComplex*) c_dev;

#ifdef WITH_GPU_STREAMS
  cuda_copy_double_complex_tmp2_c_kernel<<<blocks,threadsPerBlock, 0, my_stream>>>(tmp2_casted, c_casted, nr_done, nstor, lcs, lce, ldc, ldcCols);
#else
  cuda_copy_double_complex_tmp2_c_kernel<<<blocks,threadsPerBlock>>>(tmp2_casted, c_casted, nr_done, nstor, lcs, lce, ldc, ldcCols);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing copy_double_complex_tmp2_c_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_float_complex_tmp2_c_kernel(cuFloatComplex *tmp2_dev, cuFloatComplex *c_dev, const int nr_done, const int nstor, const int lcs, const int lce, const int ldc, const int ldcCols){

  //dim3 blocks = dim3(lce-lcs+1,1,1);
  //dim3 threadsPerBlock = dim3(nr_done+nstor-(nr_done+1)+1,1,1);

  int i_index    = threadIdx.x +1; // range 1..nstor
  int j_index = blockIdx.x + 1; // range 1..lce-lse+1
  //c_dev[nr_done+(i_index-1) + ldc*(lcs-1+j_index-1)] = tmp2_dev[0+(i_index-1)+nstor*(lcs-1+j_index-1)];
  c_dev[nr_done+(i_index-1) + ldc*(lcs-1+j_index-1)] = tmp2_dev[0+(i_index-1)+nstor*(j_index-1)];
}

extern "C" void cuda_copy_float_complex_tmp2_c_FromC(float _Complex *tmp2_dev, float _Complex *c_dev, int *nr_done_in, int *nstor_in, int *lcs_in, int *lce_in, int *ldc_in, int *ldcCols_in, cudaStream_t my_stream) { 
		
  int nr_done = *nr_done_in;   
  int nstor = *nstor_in;
  int lcs = *lcs_in;
  int lce = *lce_in;
  int ldc = *ldc_in;
  int ldcCols = *ldcCols_in;

  dim3 blocks = dim3(lce-lcs+1,1,1);
  dim3 threadsPerBlock = dim3(nr_done+nstor-(nr_done+1)+1,1,1);

  cuFloatComplex* tmp2_casted = (cuFloatComplex*) tmp2_dev;
  cuFloatComplex* c_casted = (cuFloatComplex*) c_dev;

#ifdef WITH_GPU_STREAMS
  cuda_copy_float_complex_tmp2_c_kernel<<<blocks,threadsPerBlock, 0, my_stream>>>(tmp2_casted, c_casted, nr_done, nstor, lcs, lce, ldc, ldcCols);
#else
  cuda_copy_float_complex_tmp2_c_kernel<<<blocks,threadsPerBlock>>>(tmp2_casted, c_casted, nr_done, nstor, lcs, lce, ldc, ldcCols);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing copy_float_complex_tmp2_c_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}



__global__ void cuda_copy_double_a_aux_bc_kernel(double *a_dev, double *aux_bc_dev, const int n_aux_bc, const int nvals, const int lrs, const int lre, const int noff, const int nblk, const int n, const int l_rows, const int lda, const int ldaCols){

  //dim3 blocks = dim3(lre-lrs+1,1,1);
  //dim3 threadsPerBlock = dim3(1,1,1);

  int i_index    = blockIdx.x +1; // range 1..lre-lrs+1
  int j_index = threadIdx.x + 1; // range 1..1
  aux_bc_dev[(n_aux_bc+1-1)+(i_index-1)] = a_dev[(lrs-1)+(i_index-1)+lda*(noff*nblk+n-1)];
}

extern "C" void cuda_copy_double_a_aux_bc_FromC(double *a_dev, double *aux_bc_dev, int *n_aux_bc_in, int *nvals_in, int *lrs_in, int *lre_in, int *noff_in, int *nblk_in, int *n_in, int *l_rows_in, int *lda_in, int *ldaCols_in, cudaStream_t my_stream) { 
		
  int n_aux_bc = *n_aux_bc_in;   
  int nvals = *nvals_in;
  int lrs = *lrs_in;
  int lre = *lre_in;
  int noff = *noff_in;
  int nblk = *nblk_in;
  int n = *n_in;
  int l_rows = *l_rows_in;
  int lda = *lda_in;
  int ldaCols = *ldaCols_in;

  dim3 blocks = dim3(lre-lrs+1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_copy_double_a_aux_bc_kernel<<<blocks,threadsPerBlock, 0, my_stream>>>(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols);
#else
  cuda_copy_double_a_aux_bc_kernel<<<blocks,threadsPerBlock>>>(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing copy_double_a_aux_bc_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}



__global__ void cuda_copy_float_a_aux_bc_kernel(float *a_dev, float *aux_bc_dev, const int n_aux_bc, const int nvals, const int lrs, const int lre, const int noff, const int nblk, const int n, const int l_rows, const int lda, const int ldaCols){

  //dim3 blocks = dim3(lre-lrs+1,1,1);
  //dim3 threadsPerBlock = dim3(1,1,1);

  int i_index    = blockIdx.x +1; // range 1..lre-lrs+1
  int j_index = threadIdx.x + 1; // range 1..1
  aux_bc_dev[(n_aux_bc+1-1)+(i_index-1)] = a_dev[(lrs-1)+(i_index-1)+lda*(noff*nblk+n-1)];
}

extern "C" void cuda_copy_float_a_aux_bc_FromC(float *a_dev, float *aux_bc_dev, int *n_aux_bc_in, int *nvals_in, int *lrs_in, int *lre_in, int *noff_in, int *nblk_in, int *n_in, int *l_rows_in, int *lda_in, int *ldaCols_in, cudaStream_t my_stream) { 
		
  int n_aux_bc = *n_aux_bc_in;   
  int nvals = *nvals_in;
  int lrs = *lrs_in;
  int lre = *lre_in;
  int noff = *noff_in;
  int nblk = *nblk_in;
  int n = *n_in;
  int l_rows = *l_rows_in;
  int lda = *lda_in;
  int ldaCols = *ldaCols_in;

  dim3 blocks = dim3(lre-lrs+1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_copy_float_a_aux_bc_kernel<<<blocks,threadsPerBlock, 0, my_stream>>>(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols);
#else
  cuda_copy_float_a_aux_bc_kernel<<<blocks,threadsPerBlock>>>(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing copy_float_a_aux_bc_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}


__global__ void cuda_copy_double_complex_a_aux_bc_kernel(cuDoubleComplex *a_dev, cuDoubleComplex *aux_bc_dev, const int n_aux_bc, const int nvals, const int lrs, const int lre, const int noff, const int nblk, const int n, const int l_rows, const int lda, const int ldaCols){

  //dim3 blocks = dim3(lre-lrs+1,1,1);
  //dim3 threadsPerBlock = dim3(1,1,1);

  int i_index    = blockIdx.x +1; // range 1..lre-lrs+1
  int j_index = threadIdx.x + 1; // range 1..1
  aux_bc_dev[(n_aux_bc+1-1)+(i_index-1)] = a_dev[(lrs-1)+(i_index-1)+lda*(noff*nblk+n-1)];
}

extern "C" void cuda_copy_double_complex_a_aux_bc_FromC(double _Complex *a_dev, double _Complex *aux_bc_dev, int *n_aux_bc_in, int *nvals_in, int *lrs_in, int *lre_in, int *noff_in, int *nblk_in, int *n_in, int *l_rows_in, int *lda_in, int *ldaCols_in, cudaStream_t my_stream) { 
		
  int n_aux_bc = *n_aux_bc_in;   
  int nvals = *nvals_in;
  int lrs = *lrs_in;
  int lre = *lre_in;
  int noff = *noff_in;
  int nblk = *nblk_in;
  int n = *n_in;
  int l_rows = *l_rows_in;
  int lda = *lda_in;
  int ldaCols = *ldaCols_in;

  dim3 blocks = dim3(lre-lrs+1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

  
  cuDoubleComplex* a_dev_casted = (cuDoubleComplex*) a_dev;
  cuDoubleComplex* aux_bc_dev_casted = (cuDoubleComplex*) aux_bc_dev;

#ifdef WITH_GPU_STREAMS
  cuda_copy_double_complex_a_aux_bc_kernel<<<blocks,threadsPerBlock, 0, my_stream>>>(a_dev_casted, aux_bc_dev_casted, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols);
#else
  cuda_copy_double_complex_a_aux_bc_kernel<<<blocks,threadsPerBlock>>>(a_dev_casted, aux_bc_dev_casted, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing copy_double_complex_a_aux_bc_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}


__global__ void cuda_copy_float_complex_a_aux_bc_kernel(cuFloatComplex *a_dev, cuFloatComplex *aux_bc_dev, const int n_aux_bc, const int nvals, const int lrs, const int lre, const int noff, const int nblk, const int n, const int l_rows, const int lda, const int ldaCols){

  //dim3 blocks = dim3(lre-lrs+1,1,1);
  //dim3 threadsPerBlock = dim3(1,1,1);

  int i_index    = blockIdx.x +1; // range 1..lre-lrs+1
  int j_index = threadIdx.x + 1; // range 1..1
  aux_bc_dev[(n_aux_bc+1-1)+(i_index-1)] = a_dev[(lrs-1)+(i_index-1)+lda*(noff*nblk+n-1)];
}

extern "C" void cuda_copy_float_complex_a_aux_bc_FromC(float _Complex *a_dev, float _Complex *aux_bc_dev, int *n_aux_bc_in, int *nvals_in, int *lrs_in, int *lre_in, int *noff_in, int *nblk_in, int *n_in, int *l_rows_in, int *lda_in, int *ldaCols_in, cudaStream_t my_stream) { 
		
  int n_aux_bc = *n_aux_bc_in;   
  int nvals = *nvals_in;
  int lrs = *lrs_in;
  int lre = *lre_in;
  int noff = *noff_in;
  int nblk = *nblk_in;
  int n = *n_in;
  int l_rows = *l_rows_in;
  int lda = *lda_in;
  int ldaCols = *ldaCols_in;

  dim3 blocks = dim3(lre-lrs+1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

  
  cuFloatComplex* a_dev_casted = (cuFloatComplex*) a_dev;
  cuFloatComplex* aux_bc_dev_casted = (cuFloatComplex*) aux_bc_dev;

#ifdef WITH_GPU_STREAMS
  cuda_copy_float_complex_a_aux_bc_kernel<<<blocks,threadsPerBlock, 0, my_stream>>>(a_dev_casted, aux_bc_dev_casted, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols);
#else
  cuda_copy_float_complex_a_aux_bc_kernel<<<blocks,threadsPerBlock>>>(a_dev_casted, aux_bc_dev_casted, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing copy_float_complex_a_aux_bc_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}


__global__ void cuda_copy_double_aux_bc_aux_mat_kernel(double *aux_bc_dev, double *aux_mat_dev, const int lrs, const int lre, const int nstor, const int n_aux_bc, const int nvals, const int l_rows, const int nblk_mult, const int nblk) {
		
  //dim3 threadsPerBlock = dim3(1,1,1);
  //dim3 blocks = dim3(lre-lrs+1,1,1);
  //dim3 blocks = dim3(1,1,1);

  int i_index    = threadIdx.x +1; // range 1..lre-lrs+1
  int j_index = blockIdx.x + 1; // range 1..lre-lrs+1
  aux_mat_dev[lrs-1+(j_index-1)+l_rows*(nstor-1)] = aux_bc_dev[n_aux_bc+(j_index-1)];

  //aux_mat(lrs:lre,nstor) = aux_bc(n_aux_bc+1:n_aux_bc+nvals)

}


extern "C" void cuda_copy_double_aux_bc_aux_mat_FromC(double *aux_bc_dev, double *aux_mat_dev, int *lrs_in, int *lre_in, int *nstor_in, int *n_aux_bc_in, int *nvals_in, int *l_rows_in, int *nblk_in, int *nblk_mult_in , cudaStream_t my_stream) {
		


  int lrs = *lrs_in;
  int lre = *lre_in;
  int nstor = *nstor_in;
  int n_aux_bc = *n_aux_bc_in;   
  int nvals = *nvals_in;
  int l_rows = *l_rows_in;
  int nblk_mult = *nblk_mult_in;
  int nblk = *nblk_in;
  
  //dim3 blocks = dim3(1,1,1);
  dim3 blocks = dim3(lre-lrs+1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

  //printf("C= lrs=%d, lre=%d, nstor=%d, n_aux_bc=%d, nvals=%d, l_rows=%d, nblk=%d, nblk_mult=%d \n", lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult);
  //printf("nvals=%d lre-lrs+1=%d \n", nvals, lre-lrs+1);

  //printf("Threads per Block %d\n",lre-lrs+1);
#ifdef WITH_GPU_STREAMS
  cuda_copy_double_aux_bc_aux_mat_kernel<<<blocks,threadsPerBlock, 0, my_stream>>>(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult);
#else
  cuda_copy_double_aux_bc_aux_mat_kernel<<<blocks,threadsPerBlock>>>(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing copy_double_aux_bc_aux_mat_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_float_aux_bc_aux_mat_kernel(float *aux_bc_dev, float *aux_mat_dev, const int lrs, const int lre, const int nstor, const int n_aux_bc, const int nvals, const int l_rows, const int nblk, const int nblk_mult) {
		
  dim3 blocks = dim3(lre-lrs+1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

  int i_index    = threadIdx.x +1; // range 1..lre-lrs+1
  int j_index = blockIdx.x + 1; // range 1..1
  aux_mat_dev[lrs-1+(j_index-1)+l_rows*(nstor-1)] = aux_bc_dev[n_aux_bc+(j_index-1)];
}

extern "C" void cuda_copy_float_aux_bc_aux_mat_FromC(float *aux_bc_dev, float *aux_mat_dev, int *lrs_in, int *lre_in, int *nstor_in, int *n_aux_bc_in, int *nvals_in, int *l_rows_in, int *nblk_in, int *nblk_mult_in, cudaStream_t my_stream) {
		


  int lrs = *lrs_in;
  int lre = *lre_in;
  int nstor = *nstor_in;
  int n_aux_bc = *n_aux_bc_in;   
  int nvals = *nvals_in;
  int l_rows = *l_rows_in;
  int nblk_mult = *nblk_mult_in;
  int nblk = *nblk_in;
  
  dim3 blocks = dim3(lre-lrs+1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_copy_float_aux_bc_aux_mat_kernel<<<blocks,threadsPerBlock, 0, my_stream>>>(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult);
#else
  cuda_copy_float_aux_bc_aux_mat_kernel<<<blocks,threadsPerBlock>>>(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing copy_float_aux_bc_aux_mat_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_double_complex_aux_bc_aux_mat_kernel(cuDoubleComplex *aux_bc_dev, cuDoubleComplex *aux_mat_dev, const int lrs, const int lre, const int nstor, const int n_aux_bc, const int nvals, const int l_rows, const int nblk, const int nblk_mult) {
		
  dim3 blocks = dim3(lre-lrs+1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

  int i_index    = threadIdx.x +1; // range 1..lre-lrs+1
  int j_index = blockIdx.x + 1; // range 1..1
  aux_mat_dev[lrs-1+(j_index-1)+l_rows*(nstor-1)] = aux_bc_dev[n_aux_bc+(j_index-1)];
}

extern "C" void cuda_copy_double_complex_aux_bc_aux_mat_FromC(double _Complex *aux_bc_dev, double _Complex *aux_mat_dev, int *lrs_in, int *lre_in, int *nstor_in, int *n_aux_bc_in, int *nvals_in, int *l_rows_in, int *nblk_in, int *nblk_mult_in, cudaStream_t my_stream) {
		


  int lrs = *lrs_in;
  int lre = *lre_in;
  int nstor = *nstor_in;
  int n_aux_bc = *n_aux_bc_in;   
  int nvals = *nvals_in;
  int l_rows = *l_rows_in;
  int nblk_mult = *nblk_mult_in;
  int nblk = *nblk_in;
  
  dim3 blocks = dim3(lre-lrs+1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

  cuDoubleComplex* aux_bc_dev_casted = (cuDoubleComplex*) aux_bc_dev;
  cuDoubleComplex* aux_mat_dev_casted = (cuDoubleComplex*) aux_mat_dev;


#ifdef WITH_GPU_STREAMS
  cuda_copy_double_complex_aux_bc_aux_mat_kernel<<<blocks,threadsPerBlock, 0, my_stream>>>(aux_bc_dev_casted, aux_mat_dev_casted, lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult);
#else
  cuda_copy_double_complex_aux_bc_aux_mat_kernel<<<blocks,threadsPerBlock>>>(aux_bc_dev_casted, aux_mat_dev_casted, lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing copy_double_complex_aux_bc_aux_mat_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}


__global__ void cuda_copy_float_complex_aux_bc_aux_mat_kernel(cuFloatComplex *aux_bc_dev, cuFloatComplex *aux_mat_dev, const int lrs, const int lre, const int nstor, const int n_aux_bc, const int nvals, const int l_rows, const int nblk, const int nblk_mult) {
		
  dim3 blocks = dim3(lre-lrs+1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

  int i_index    = threadIdx.x +1; // range 1..lre-lrs+1
  int j_index = blockIdx.x + 1; // range 1..1
  aux_mat_dev[lrs-1+(j_index-1)+l_rows*(nstor-1)] = aux_bc_dev[n_aux_bc+(j_index-1)];
}

extern "C" void cuda_copy_float_complex_aux_bc_aux_mat_FromC(float _Complex *aux_bc_dev, float _Complex *aux_mat_dev, int *lrs_in, int *lre_in, int *nstor_in, int *n_aux_bc_in, int *nvals_in, int *l_rows_in, int *nblk_in, int *nblk_mult_in, cudaStream_t my_stream) {
		


  int lrs = *lrs_in;
  int lre = *lre_in;
  int nstor = *nstor_in;
  int n_aux_bc = *n_aux_bc_in;   
  int nvals = *nvals_in;
  int l_rows = *l_rows_in;
  int nblk_mult = *nblk_mult_in;
  int nblk = *nblk_in;
  
  dim3 blocks = dim3(lre-lrs+1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

  cuFloatComplex* aux_bc_dev_casted = (cuFloatComplex*) aux_bc_dev;
  cuFloatComplex* aux_mat_dev_casted = (cuFloatComplex*) aux_mat_dev;


#ifdef WITH_GPU_STREAMS
  cuda_copy_float_complex_aux_bc_aux_mat_kernel<<<blocks,threadsPerBlock, 0, my_stream>>>(aux_bc_dev_casted, aux_mat_dev_casted, lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult);
#else
  cuda_copy_float_complex_aux_bc_aux_mat_kernel<<<blocks,threadsPerBlock>>>(aux_bc_dev_casted, aux_mat_dev_casted, lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing copy_float_complex_aux_bc_aux_mat_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

//________________________________________________________________

template <typename T>
__global__ void cuda_copy_aux_full_kernel(T *lhs_dev, T *rhs_dev, int l_rows, int l_cols, int lld_lhs, int lld_rhs) {

  // aux_a_full(1:l_rows,1:l_cols) = a(1:l_rows,1:l_cols)

  int i_loc = threadIdx.x; // 0..l_rows-1
  int j_loc = blockIdx.x ; // 0..l_cowl-1

  for (; j_loc < l_cols; j_loc += gridDim.x) {
    for (; i_loc < l_rows; i_loc += blockDim.x) {
      lhs_dev[i_loc+j_loc*lld_lhs] = rhs_dev[i_loc+j_loc*lld_rhs];
    }
  }
}

template <typename T>
void cuda_copy_aux_full(T *lhs_dev, T *rhs_dev, int *l_rows_in, int *l_cols_in, int *lld_lhs_in, int *lld_rhs_in, int *debug_in, cudaStream_t my_stream){
  int l_rows = *l_rows_in;
  int l_cols = *l_cols_in;
  int lld_lhs = *lld_lhs_in;
  int lld_rhs = *lld_rhs_in;
  int debug = *debug_in;

  dim3 blocks = dim3(l_cols,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_copy_aux_full_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(lhs_dev, rhs_dev, l_rows, l_cols, lld_lhs, lld_rhs);
#else
  cuda_copy_aux_full_kernel<<<blocks,threadsPerBlock>>>            (lhs_dev, rhs_dev, l_rows, l_cols, lld_lhs, lld_rhs);
#endif

  if (debug)
    {
    cudaDeviceSynchronize();
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess){
      printf("Error in executing cuda_copy_aux_full: %s\n",cudaGetErrorString(cuerr));
    }
  }
}

extern "C" void cuda_copy_aux_full_FromC(char dataType, intptr_t lhs_dev, intptr_t rhs_dev,
                                         int *l_rows_in, int *l_cols_in, int *lld_lhs_in, int *lld_rhs_in, int *debug_in, cudaStream_t my_stream){
  if (dataType=='D') cuda_copy_aux_full<double>((double *) lhs_dev, (double *) rhs_dev, l_rows_in, l_cols_in, lld_lhs_in, lld_rhs_in, debug_in, my_stream);
  if (dataType=='S') cuda_copy_aux_full<float> ((float  *) lhs_dev, (float  *) rhs_dev, l_rows_in, l_cols_in, lld_lhs_in, lld_rhs_in, debug_in, my_stream);
  if (dataType=='Z') cuda_copy_aux_full<cuDoubleComplex>((cuDoubleComplex *) lhs_dev, (cuDoubleComplex *) rhs_dev, l_rows_in, l_cols_in, lld_lhs_in, lld_rhs_in, debug_in, my_stream);
  if (dataType=='C') cuda_copy_aux_full<cuFloatComplex> ((cuFloatComplex  *) lhs_dev, (cuFloatComplex  *) rhs_dev, l_rows_in, l_cols_in, lld_lhs_in, lld_rhs_in, debug_in, my_stream);
}

//________________________________________________________________

template <typename T>
__global__ void cuda_copy_and_set_zeros_aux_full_kernel(T *a_dev, T *aux_mat_full_dev, int l_rows, int l_cols, int nblk_mult) {

  // aux_a_full(1:l_rows,1:l_cols) = a(1:l_rows,1:l_cols)
  // if (l_rows<nblk_mult) aux_a_full(l_rows+1:nblk_mult,1:l_cols) = 0
  // if (l_cols<nblk_mult) aux_a_full(1:l_rows,l_cols+1:nblk_mult) = 0
  // if (l_rows<nblk_mult .and. l_cols<nblk_mult) aux_a_full(l_rows+1:nblk_mult,l_cols+1:nblk_mult) = 0

  int i_loc = threadIdx.x; // 0..nblk_mult-1
  int j_loc = blockIdx.x ; // 0..nblk_mult-1

  T Zero = elpaDeviceNumber<T>(0.0);

  for (; j_loc < nblk_mult; j_loc += gridDim.x) {
    for (; i_loc < nblk_mult; i_loc += blockDim.x) {
      if (i_loc < l_rows && j_loc < l_cols) aux_mat_full_dev[i_loc+j_loc*nblk_mult] = a_dev[i_loc+j_loc*l_rows];
      else aux_mat_full_dev[i_loc+j_loc*nblk_mult] = Zero;
    }
  }
}

template <typename T>
void cuda_copy_and_set_zeros_aux_full(T *mat_dev, T *aux_mat_full_dev, int *l_rows_in, int *l_cols_in, int *nblk_mult_in, int *debug_in, cudaStream_t my_stream){
  int l_rows = *l_rows_in;
  int l_cols = *l_cols_in;
  int nblk_mult = *nblk_mult_in;
  int debug = *debug_in;

  dim3 blocks = dim3(nblk_mult,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_copy_and_set_zeros_aux_full_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(mat_dev, aux_mat_full_dev, l_rows, l_cols, nblk_mult);
#else
  cuda_copy_and_set_zeros_aux_full_kernel<<<blocks,threadsPerBlock>>>(mat_dev, aux_mat_full_dev, l_rows, l_cols, nblk_mult);
#endif

  if (debug)
    {
    cudaDeviceSynchronize();
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess){
      printf("Error in executing cuda_copy_and_set_zeros_aux_full: %s\n",cudaGetErrorString(cuerr));
    }
  }
}

extern "C" void cuda_copy_and_set_zeros_aux_full_FromC(char dataType, intptr_t mat_dev, intptr_t aux_mat_full_dev,
                                                       int *l_rows_in, int *l_cols_in, int *nblk_mult_in, int *debug_in, cudaStream_t my_stream){
  if (dataType=='D') cuda_copy_and_set_zeros_aux_full<double>((double *) mat_dev, (double *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_in, debug_in, my_stream);
  if (dataType=='S') cuda_copy_and_set_zeros_aux_full<float> ((float  *) mat_dev, (float  *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_in, debug_in, my_stream);
  if (dataType=='Z') cuda_copy_and_set_zeros_aux_full<cuDoubleComplex>((cuDoubleComplex *) mat_dev, (cuDoubleComplex *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_in, debug_in, my_stream);
  if (dataType=='C') cuda_copy_and_set_zeros_aux_full<cuFloatComplex> ((cuFloatComplex  *) mat_dev, (cuFloatComplex  *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_in, debug_in, my_stream);
}

//________________________________________________________________

// PETERDEBUG:  l_cols is unused, delete it
// also "variable "threadsPerBlock" was declared but never referenced" in this source file
template <typename T>
__global__ void cuda_copy_and_set_zeros_aux_a_full_kernel(T *a_dev, T *aux_a_full_dev, int l_rows, int l_cols, int nblk_mult_cols,
                                                          int nblk, int np_bc_fine, int np_cols_fine, int np_cols) {

  // do j_block_loc_fine = 0, nblk_mult_cols/nblk-1
  //   j_block_loc = (np_bc_fine + j_block_loc_fine*np_cols_fine)/np_cols
  //     aux_a_full(1:l_rows, 1+j_block_loc_fine*nblk: nblk+j_block_loc_fine*nblk) = &
  //              a(1:l_rows, 1+j_block_loc*nblk     : nblk+j_block_loc*nblk)
  //   enddo ! j_block_loc_fine
  //   if (mod(nblk_mult_cols,nblk) /= 0) then ! last incomplete nblk-block
  //     j_block_loc = (np_bc_fine + j_block_loc_fine*np_cols_fine)/np_cols
  //     aux_a_full(1:l_rows, 1+j_block_loc_fine*nblk: mod(nblk_mult_cols,nblk)+j_block_loc_fine*nblk) = &
  //              a(1:l_rows, 1+j_block_loc*nblk     : mod(nblk_mult_cols,nblk)+j_block_loc*nblk)
  //   endif
  // endif ! useGPU

  int i0 = threadIdx.x; // i  = 0..l_rows-1
  int dj0 = blockIdx.x; // dj = 0..nblk-1

  // Loop through full blocks
  int j_block_loc_fine = 0;
  for (; j_block_loc_fine < nblk_mult_cols/nblk; j_block_loc_fine++) 
    {
    int j_block_loc = (np_bc_fine + j_block_loc_fine*np_cols_fine)/np_cols;
    for (int dj = dj0; dj < nblk ; dj += gridDim.x)
      {
      for (int i = i0; i < l_rows; i += blockDim.x)
        {
        aux_a_full_dev[i + (dj+j_block_loc_fine*nblk)*l_rows] = a_dev[i + (dj+j_block_loc*nblk)*l_rows];
        }
      }
    }

  // Handle the last incomplete block if it exists
  if (nblk_mult_cols%nblk != 0) 
    {
    int j_block_loc = (np_bc_fine + j_block_loc_fine*np_cols_fine)/np_cols;
      for (int dj = dj0; dj < nblk_mult_cols%nblk ; dj += gridDim.x) 
        {
        for (int i = i0; i < l_rows; i += blockDim.x)
          {
          aux_a_full_dev[i + (dj+j_block_loc_fine*nblk)*l_rows] = a_dev[i + (dj+j_block_loc*nblk)*l_rows];
          }
        }
    }
}



template <typename T>
void cuda_copy_and_set_zeros_aux_a_full(T *mat_dev, T *aux_mat_full_dev, int *l_rows_in, int *l_cols_in, int *nblk_mult_cols_in, 
                                        int *nblk_in, int *np_bc_fine_in, int *np_cols_fine_in, int *np_cols_in, int *debug_in, cudaStream_t my_stream){
  int l_rows = *l_rows_in;
  int l_cols = *l_cols_in;
  int nblk_mult_cols = *nblk_mult_cols_in;
  int nblk = *nblk_in;
  int np_bc_fine = *np_bc_fine_in;
  int np_cols_fine = *np_cols_fine_in;
  int np_cols = *np_cols_in;
  int debug = *debug_in;

  dim3 blocks = dim3(nblk, 1, 1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK, 1, 1);

#ifdef WITH_GPU_STREAMS
  cuda_copy_and_set_zeros_aux_a_full_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(mat_dev, aux_mat_full_dev, l_rows, l_cols, nblk_mult_cols,
                                                                                    nblk, np_bc_fine, np_cols_fine, np_cols);
#else
  cuda_copy_and_set_zeros_aux_a_full_kernel<<<blocks,threadsPerBlock>>>            (mat_dev, aux_mat_full_dev, l_rows, l_cols, nblk_mult_cols,
                                                                                    nblk, np_bc_fine, np_cols_fine, np_cols);
#endif

  if (debug)
    {
    cudaDeviceSynchronize();
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess){
      printf("Error in executing cuda_copy_and_set_zeros_aux_full: %s\n",cudaGetErrorString(cuerr));
    }
  }
}

extern "C" void cuda_copy_and_set_zeros_aux_a_full_FromC(char dataType, intptr_t mat_dev, intptr_t aux_mat_full_dev,
                                                       int *l_rows_in, int *l_cols_in, int *nblk_mult_cols_in, int *nblk_in, int *np_bc_fine_in, int *np_cols_fine_in, int *np_cols_in, int *debug_in, cudaStream_t my_stream){
  if (dataType=='D') cuda_copy_and_set_zeros_aux_a_full<double>((double *) mat_dev, (double *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_cols_in, nblk_in, np_bc_fine_in, np_cols_fine_in, np_cols_in, debug_in, my_stream);
  if (dataType=='S') cuda_copy_and_set_zeros_aux_a_full<float> ((float  *) mat_dev, (float  *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_cols_in, nblk_in, np_bc_fine_in, np_cols_fine_in, np_cols_in, debug_in, my_stream);
  if (dataType=='Z') cuda_copy_and_set_zeros_aux_a_full<cuDoubleComplex>((cuDoubleComplex *) mat_dev, (cuDoubleComplex *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_cols_in, nblk_in, np_bc_fine_in, np_cols_fine_in, np_cols_in, debug_in, my_stream);
  if (dataType=='C') cuda_copy_and_set_zeros_aux_a_full<cuFloatComplex> ((cuFloatComplex  *) mat_dev, (cuFloatComplex  *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_cols_in, nblk_in, np_bc_fine_in, np_cols_fine_in, np_cols_in, debug_in, my_stream);
}

//________________________________________________________________

template <typename T>
__global__ void cuda_copy_and_set_zeros_aux_b_full_kernel(T *b_dev, T *aux_b_full_dev, int l_rows, int l_cols, int nblk_mult, 
                                                          int nblk_mult_rows, int nblk, int np_fine, int np_rows_fine, int np_rows) {

  // do i_block_loc_fine = 0, nblk_mult_rows/nblk-1
  //   i_block_loc = (np_fine + i_block_loc_fine * np_rows_fine) / np_rows
  //   aux_b_full(1 + i_block_loc_fine * nblk : nblk + i_block_loc_fine * nblk, 1 : l_cols) = &
  //            b(1 + i_block_loc * nblk      : nblk + i_block_loc * nblk     , 1 : l_cols)
  // enddo ! i_block_loc_fine
  // if (mod(nblk_mult_rows, nblk) /= 0) then ! last incomplete nblk-block
  //   i_block_loc = (np_fine + i_block_loc_fine * np_rows_fine) / np_rows
  //   aux_b_full(1 + i_block_loc_fine * nblk : mod(nblk_mult_rows, nblk) + i_block_loc_fine * nblk, 1 : l_cols) = &
  //            b(1 + i_block_loc * nblk : mod(nblk_mult_rows, nblk) + i_block_loc * nblk, 1 : l_cols)
  // endif

  int di0 = threadIdx.x; // di = 0..nblk-1
  int j0  = blockIdx.x ; // j  = 0..l_cols-1

  // Loop through full blocks
  int i_block_loc_fine = 0;
  for (; i_block_loc_fine < nblk_mult_rows/nblk; i_block_loc_fine++) 
    {
    int i_block_loc = (np_fine + i_block_loc_fine*np_rows_fine)/np_rows;
    for (int j = j0; j < l_cols; j += gridDim.x)
      {
      for (int di = di0; di < nblk ; di += blockDim.x)
        {
        aux_b_full_dev[di + i_block_loc_fine*nblk + j*nblk_mult] = b_dev[di + i_block_loc*nblk + j*l_rows];
        }
      }
    }

  // Handle the last incomplete block if it exists
  if (nblk_mult_rows%nblk != 0)
    {
    int i_block_loc = (np_fine + i_block_loc_fine*np_rows_fine)/np_rows;
    for (int j = j0; j < l_cols; j += gridDim.x)
      {
      for (int di = di0; di < nblk_mult_rows%nblk ; di += blockDim.x)
        {
        aux_b_full_dev[di + i_block_loc_fine*nblk + j*nblk_mult] = b_dev[di + i_block_loc*nblk + j*l_rows];
        }
      }
    }
}

template <typename T>
void cuda_copy_and_set_zeros_aux_b_full(T *mat_dev, T *aux_mat_full_dev, int *l_rows_in, int *l_cols_in, int *nblk_mult_in, 
                                        int *nblk_mult_rows_in, int *nblk_in, int *np_fine_in, int *np_rows_fine_in, int *np_rows_in,
                                        int *SM_count_in, int *debug_in, cudaStream_t my_stream){

  int l_rows = *l_rows_in;
  int l_cols = *l_cols_in;
  int nblk_mult = *nblk_mult_in;
  int nblk_mult_rows = *nblk_mult_rows_in;
  int nblk = *nblk_in;
  int np_fine = *np_fine_in;
  int np_rows_fine = *np_rows_fine_in;
  int np_rows = *np_rows_in;
  int SM_count = *SM_count_in;
  int debug = *debug_in;

  dim3 blocks = dim3(SM_count, 1, 1); // PETERDEBUG what happens in the analogous Intel-GPU case? Here and in other analogous places
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK, 1, 1);

  // dim3 blocks = dim3(1,1,1);
  // dim3 threadsPerBlock = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_copy_and_set_zeros_aux_b_full_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(mat_dev, aux_mat_full_dev, l_rows, l_cols, nblk_mult, 
                                                                                    nblk_mult_rows, nblk, np_fine, np_rows_fine, np_rows);
#else
  cuda_copy_and_set_zeros_aux_b_full_kernel<<<blocks,threadsPerBlock>>>            (mat_dev, aux_mat_full_dev, l_rows, l_cols, nblk_mult, 
                                                                                    nblk_mult_rows, nblk, np_fine, np_rows_fine, np_rows);
#endif
  
    if (debug)
      {
      cudaDeviceSynchronize();
      cudaError_t cuerr = cudaGetLastError();
      if (cuerr != cudaSuccess){
        printf("Error in executing cuda_copy_and_set_zeros_aux_b_full: %s\n",cudaGetErrorString(cuerr));
      }
    }
  }

extern "C" void cuda_copy_and_set_zeros_aux_b_full_FromC(char dataType, intptr_t mat_dev, intptr_t aux_mat_full_dev,
                                                       int *l_rows_in, int *l_cols_in, int *nblk_mult_in, 
                                                       int *nblk_mult_rows_in, int *nblk_in, int *np_fine_in, int *np_rows_fine_in, int *np_rows_in,
                                                       int *SM_count_in, int *debug_in, cudaStream_t my_stream){
  if (dataType=='D') cuda_copy_and_set_zeros_aux_b_full<double>((double *) mat_dev, (double *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_in, nblk_mult_rows_in, nblk_in, np_fine_in, np_rows_fine_in, np_rows_in, SM_count_in, debug_in, my_stream);
  if (dataType=='S') cuda_copy_and_set_zeros_aux_b_full<float> ((float  *) mat_dev, (float  *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_in, nblk_mult_rows_in, nblk_in, np_fine_in, np_rows_fine_in, np_rows_in, SM_count_in, debug_in, my_stream);
  if (dataType=='Z') cuda_copy_and_set_zeros_aux_b_full<cuDoubleComplex>((cuDoubleComplex *) mat_dev, (cuDoubleComplex *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_in, nblk_mult_rows_in, nblk_in, np_fine_in, np_rows_fine_in, np_rows_in, SM_count_in, debug_in, my_stream);
  if (dataType=='C') cuda_copy_and_set_zeros_aux_b_full<cuFloatComplex> ((cuFloatComplex  *) mat_dev, (cuFloatComplex  *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_in, nblk_mult_rows_in, nblk_in, np_fine_in, np_rows_fine_in, np_rows_in, SM_count_in, debug_in, my_stream);
}

//________________________________________________________________

template <typename T>
__global__ void cuda_ccl_copy_buf_send_kernel(T *a_dev, T *buf_send_dev, int l_rows, int l_cols, int lld_buf, int nblk,
                                              int i_block_loc_fine_max, int j_block_loc_fine_max, int np_fine, int np_bc_fine, 
                                              int np_rows_fine, int np_cols_fine, int np_rows, int np_cols) {
                                           
  // ! The nested loop is symmetric wrt to i,j, so we use the rigid order of indices for convenience of copying
  // do j_block_loc_fine = 0, j_block_loc_fine_max
  //   j_block_loc = (np_t + j_block_loc_fine*np_cols_fine)/np_cols
  //   nblk_cut_col = min(nblk, l_cols-j_block_loc*nblk)

  //   do i_block_loc_fine = 0, i_block_loc_fine_max
  //     i_block_loc = (np + i_block_loc_fine*np_rows_fine)/np_rows
  //     nblk_cut_row = min(nblk, l_rows-i_block_loc*nblk)

  //     buf_send(1+ i_block_loc_fine*nblk: nblk_cut_row + i_block_loc_fine*nblk,   &
  //               1+ j_block_loc_fine*nblk: nblk_cut_col + j_block_loc_fine*nblk) = &
  //             a(1+ i_block_loc     *nblk: nblk_cut_row + i_block_loc     *nblk,   &
  //               1+ j_block_loc     *nblk: nblk_cut_col + j_block_loc     *nblk)
  //   enddo ! i_block_loc_fine
  // enddo ! j_block_loc_fine

  int di0 = threadIdx.x; // di = 0..nblk_cut_row-1
  int dj0 = blockIdx.x ; // dj = 0..nblk_cut_col-1

  int i_block_loc, j_block_loc, nblk_cut_row, nblk_cut_col;

  int j_block_loc_fine = 0;
  for (; j_block_loc_fine <= j_block_loc_fine_max; j_block_loc_fine++) 
    {
    // printf("j_block_loc_fine = %d\n", j_block_loc_fine); // PETERDEBUG
    j_block_loc = (np_bc_fine + j_block_loc_fine*np_cols_fine)/np_cols;
    nblk_cut_col = min(nblk, l_cols-j_block_loc*nblk);

    int i_block_loc_fine = 0;
    for (; i_block_loc_fine <= i_block_loc_fine_max; i_block_loc_fine++) 
      {
      i_block_loc = (np_fine + i_block_loc_fine*np_rows_fine)/np_rows;
      nblk_cut_row = min(nblk, l_rows-i_block_loc*nblk);
      
      for (int dj = dj0; dj < nblk_cut_col; dj += gridDim.x)
        {
        for (int di = di0; di < nblk_cut_row; di += blockDim.x)
          {
          buf_send_dev[(di+i_block_loc_fine*nblk) + (dj+j_block_loc_fine*nblk)*lld_buf] 
               = a_dev[(di+i_block_loc*     nblk) + (dj+j_block_loc     *nblk)*l_rows];
          }
        }
      }
    }
}

template <typename T>
void cuda_ccl_copy_buf_send(T *a_dev, T *buf_send_dev, int *l_rows_in, int *l_cols_in, int *lld_buf_in, int *nblk_in,
                            int *i_block_loc_fine_max_in, int *j_block_loc_fine_max_in, int *np_fine_in, int *np_bc_fine_in, 
                            int *np_rows_fine_in, int *np_cols_fine_in, int *np_rows_in, int *np_cols_in, int *SM_count_in, int *debug_in, cudaStream_t my_stream){

  int l_rows = *l_rows_in;
  int l_cols = *l_cols_in;
  int lld_buf = *lld_buf_in;
  int nblk = *nblk_in;
  int i_block_loc_fine_max = *i_block_loc_fine_max_in;
  int j_block_loc_fine_max = *j_block_loc_fine_max_in;
  int np_fine = *np_fine_in;
  int np_bc_fine = *np_bc_fine_in;
  int np_rows_fine = *np_rows_fine_in;
  int np_cols_fine = *np_cols_fine_in;
  int np_rows = *np_rows_in;
  int np_cols = *np_cols_in;
  int SM_count = *SM_count_in;
  int debug = *debug_in;

  dim3 blocks = dim3(SM_count, 1, 1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK, 1, 1);

#ifdef WITH_GPU_STREAMS
  cuda_ccl_copy_buf_send_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(a_dev, buf_send_dev, l_rows, l_cols, lld_buf, nblk,
                                                                        i_block_loc_fine_max, j_block_loc_fine_max, np_fine, np_bc_fine, 
                                                                        np_rows_fine, np_cols_fine, np_rows, np_cols);
#else
  cuda_ccl_copy_buf_send_kernel<<<blocks,threadsPerBlock>>>(a_dev, buf_send_dev, l_rows, l_cols, lld_buf, nblk,
                                                            i_block_loc_fine, j_block_loc_fine, np_fine, np_bc_fine, 
                                                            np_rows_fine, np_cols_fine, np_rows, np_cols);
#endif

  if (debug)
    {
    cudaDeviceSynchronize();
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess){
      printf("Error in executing cuda_ccl_copy_buf_send: %s\n",cudaGetErrorString(cuerr));
    }
  }
}

extern "C" void cuda_ccl_copy_buf_send_FromC(char dataType, intptr_t a_dev, intptr_t buf_send_dev, 
                                             int *l_rows_in, int *l_cols_in, int *lld_buf_in, int *nblk_in,
                                             int *i_block_loc_fine_in, int *j_block_loc_fine_in, int *np_fine_in, int *np_bc_fine_in, 
                                             int *np_rows_fine_in, int *np_cols_fine_in, int *np_rows_in, int *np_cols_in, int *SM_count_in, int *debug_in, cudaStream_t my_stream){
  if (dataType=='D') cuda_ccl_copy_buf_send<double>((double *) a_dev, (double *) buf_send_dev, l_rows_in, l_cols_in, lld_buf_in, nblk_in,
                                                    i_block_loc_fine_in, j_block_loc_fine_in, np_fine_in, np_bc_fine_in, 
                                                    np_rows_fine_in, np_cols_fine_in, np_rows_in, np_cols_in, SM_count_in, debug_in, my_stream);
  if (dataType=='S') cuda_ccl_copy_buf_send<float> ((float  *) a_dev, (float  *) buf_send_dev, l_rows_in, l_cols_in, lld_buf_in, nblk_in,
                                                    i_block_loc_fine_in, j_block_loc_fine_in, np_fine_in, np_bc_fine_in, 
                                                    np_rows_fine_in, np_cols_fine_in, np_rows_in, np_cols_in, SM_count_in, debug_in, my_stream);
  if (dataType=='Z') cuda_ccl_copy_buf_send<cuDoubleComplex>((cuDoubleComplex *) a_dev, (cuDoubleComplex *) buf_send_dev, l_rows_in, l_cols_in, lld_buf_in, nblk_in,
                                                    i_block_loc_fine_in, j_block_loc_fine_in, np_fine_in, np_bc_fine_in, 
                                                    np_rows_fine_in, np_cols_fine_in, np_rows_in, np_cols_in, SM_count_in, debug_in, my_stream);
  if (dataType=='C') cuda_ccl_copy_buf_send<cuFloatComplex> ((cuFloatComplex  *) a_dev, (cuFloatComplex  *) buf_send_dev, l_rows_in, l_cols_in, lld_buf_in, nblk_in,
                                                    i_block_loc_fine_in, j_block_loc_fine_in, np_fine_in, np_bc_fine_in,
                                                    np_rows_fine_in, np_cols_fine_in, np_rows_in, np_cols_in, SM_count_in, debug_in, my_stream);
}

//________________________________________________________________

template <typename T>
__global__ void cuda_ccl_copy_buf_recv_kernel(T *at_col_dev, T *buf_recv_dev, int l_rows, int l_cols, int lld_buf, int nblk,
                                              int i_block_loc_fine_max, int j_block_loc_fine_max, int np_fine, int np_bc_fine, 
                                              int np_rows_fine, int np_cols_fine, int np_rows, int np_cols) {

  // do i_block_loc_fine = 0, i_block_loc_fine_max
  //   i_block_loc = (np + i_block_loc_fine*np_rows_fine)/np_rows
  //   nblk_cut_row = min(nblk, l_rows-i_block_loc*nblk)

  //   do j_block_loc_fine = 0, j_block_loc_fine_max
  //     j_block_loc = (np_t + j_block_loc_fine*np_cols_fine)/np_cols
  //     nblk_cut_col = min(nblk, l_cols-j_block_loc*nblk)
      
  //     at(1+ i_block_loc     *nblk: nblk_cut_row + i_block_loc     *nblk,   &
  //        1+ j_block_loc     *nblk: nblk_cut_col + j_block_loc     *nblk) = &
  //     transpose(buf_recv(1+ j_block_loc_fine*nblk: nblk_cut_col + j_block_loc_fine*nblk,   &
  //                        1+ i_block_loc_fine*nblk: nblk_cut_row + i_block_loc_fine*nblk))
  //   enddo ! j_block_loc_fine
  // enddo ! i_block_loc_fine

  int di0 = threadIdx.x; // di = 0..nblk_cut_row-1
  int dj0 = blockIdx.x ; // dj = 0..nblk_cut_col-1

  int i_block_loc, j_block_loc, nblk_cut_row, nblk_cut_col;

  int i_block_loc_fine = 0;
  for (; i_block_loc_fine <= i_block_loc_fine_max; i_block_loc_fine++) 
    {
    i_block_loc = (np_fine + i_block_loc_fine*np_rows_fine)/np_rows;
    nblk_cut_row = min(nblk, l_rows-i_block_loc*nblk);

    int j_block_loc_fine = 0;
    for (; j_block_loc_fine <= j_block_loc_fine_max; j_block_loc_fine++) 
      {
      j_block_loc = (np_bc_fine + j_block_loc_fine*np_cols_fine)/np_cols;
      nblk_cut_col = min(nblk, l_cols-j_block_loc*nblk);
      
      for (int dj = dj0; dj < nblk_cut_col; dj += gridDim.x)
        {
        for (int di = di0; di < nblk_cut_row; di += blockDim.x)
          {
          at_col_dev[(di+i_block_loc*     nblk) + (dj+j_block_loc*     nblk)*l_rows] 
      = buf_recv_dev[(dj+j_block_loc_fine*nblk) + (di+i_block_loc_fine*nblk)*lld_buf];
          }
        }
      }
    }

}

template <typename T>
void cuda_ccl_copy_buf_recv(T *at_col_dev, T *buf_recv_dev, int *l_rows_in, int *l_cols_in, int *lld_buf_in, int *nblk_in,
                            int *i_block_loc_fine_max_in, int *j_block_loc_fine_max_in, int *np_fine_in, int *np_bc_fine_in, 
                            int *np_rows_fine_in, int *np_cols_fine_in, int *np_rows_in, int *np_cols_in, int *SM_count_in, int *debug_in, cudaStream_t my_stream){

  int l_rows = *l_rows_in;
  int l_cols = *l_cols_in;
  int lld_buf = *lld_buf_in;
  int nblk = *nblk_in;
  int i_block_loc_fine_max = *i_block_loc_fine_max_in;
  int j_block_loc_fine_max = *j_block_loc_fine_max_in;
  int np_fine = *np_fine_in;
  int np_bc_fine = *np_bc_fine_in;
  int np_rows_fine = *np_rows_fine_in;
  int np_cols_fine = *np_cols_fine_in;
  int np_rows = *np_rows_in;
  int np_cols = *np_cols_in;
  int SM_count = *SM_count_in;
  int debug = *debug_in;

  dim3 blocks = dim3(SM_count, 1, 1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK, 1, 1);

#ifdef WITH_GPU_STREAMS
  cuda_ccl_copy_buf_recv_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(at_col_dev, buf_recv_dev, l_rows, l_cols, lld_buf, nblk,
                                                                       i_block_loc_fine_max, j_block_loc_fine_max, np_fine, np_bc_fine, 
                                                                       np_rows_fine, np_cols_fine, np_rows, np_cols);
#else
  cuda_ccl_copy_buf_recv_kernel<<<blocks,threadsPerBlock>>>(at_col_dev, buf_recv_dev, l_rows, l_cols, lld_buf, nblk,
                                                           i_block_loc_fine_max, j_block_loc_fine_max, np_fine, np_bc_fine, 
                                                           np_rows_fine, np_cols_fine, np_rows, np_cols);
#endif
  
    if (debug)
      {
      cudaDeviceSynchronize();
      cudaError_t cuerr = cudaGetLastError();
      if (cuerr != cudaSuccess){
        printf("Error in executing cuda_ccl_copy_buf_recv: %s\n",cudaGetErrorString(cuerr));
      }
    }
  }

extern "C" void cuda_ccl_copy_buf_recv_FromC(char dataType, intptr_t at_col_dev, intptr_t buf_recv_dev, 
                                             int *l_rows_in, int *l_cols_in, int *lld_buf_in, int *nblk_in,
                                             int *i_block_loc_fine_max_in, int *j_block_loc_fine_max_in, int *np_fine_in, int *np_bc_fine_in, 
                                             int *np_rows_fine_in, int *np_cols_fine_in, int *np_rows_in, int *np_cols_in, int *SM_count_in, int *debug_in, cudaStream_t my_stream){
  if (dataType=='D') cuda_ccl_copy_buf_recv<double>((double *) at_col_dev, (double *) buf_recv_dev, l_rows_in, l_cols_in, lld_buf_in, nblk_in,
                                                    i_block_loc_fine_max_in, j_block_loc_fine_max_in, np_fine_in, np_bc_fine_in, 
                                                    np_rows_fine_in, np_cols_fine_in, np_rows_in, np_cols_in, SM_count_in, debug_in, my_stream);
  if (dataType=='S') cuda_ccl_copy_buf_recv<float> ((float  *) at_col_dev, (float  *) buf_recv_dev, l_rows_in, l_cols_in, lld_buf_in, nblk_in,
                                                    i_block_loc_fine_max_in, j_block_loc_fine_max_in, np_fine_in, np_bc_fine_in, 
                                                    np_rows_fine_in, np_cols_fine_in, np_rows_in, np_cols_in, SM_count_in, debug_in, my_stream);
  if (dataType=='Z') cuda_ccl_copy_buf_recv<cuDoubleComplex>((cuDoubleComplex *) at_col_dev, (cuDoubleComplex *) buf_recv_dev, l_rows_in, l_cols_in, lld_buf_in, nblk_in,
                                                    i_block_loc_fine_max_in, j_block_loc_fine_max_in, np_fine_in, np_bc_fine_in, 
                                                    np_rows_fine_in, np_cols_fine_in, np_rows_in, np_cols_in, SM_count_in, debug_in, my_stream);
  if (dataType=='C') cuda_ccl_copy_buf_recv<cuFloatComplex> ((cuFloatComplex  *) at_col_dev, (cuFloatComplex  *) buf_recv_dev, l_rows_in, l_cols_in, lld_buf_in, nblk_in,
                                                    i_block_loc_fine_max_in, j_block_loc_fine_max_in, np_fine_in, np_bc_fine_in,
                                                    np_rows_fine_in, np_cols_fine_in, np_rows_in, np_cols_in, SM_count_in, debug_in, my_stream);
}

//_________________________________________________________________________________________________
// non-square grid, TN, NT codepath

template <typename T>
__global__ void cuda_copy_and_set_zeros_aux_ab_full_tn_kernel(T *a_dev, T *b_dev, T *aux_a_full_dev, T *aux_b_full_dev,
                                                              int l_rows, int l_cols, int nblk_mult_max, int nblk_mult, int nblk,
                                                              int np_ab_fine, int np_rows, int my_prow,
                                                              int np_t_fine , int np_cols, int my_pcol,
                                                              int np_dirs_fine){
  // if (mod(np_t_fine,np_cols) == my_pcol) then
  //   do j_block_loc_fine = 0, nblk_mult_max/nblk-1
  //     j_block_loc = (np_t_fine + j_block_loc_fine*np_dirs_fine)/np_cols
      
  //     do i_block_loc_fine = 0, nblk_mult/nblk-1
  //       i_block_loc = (np_ab_fine + i_block_loc_fine*np_dirs_fine)/np_rows
        
  //       nblk_cols_cut = min(nblk, l_cols - j_block_loc*nblk)
  //       nblk_rows_cut = min(nblk, l_rows - i_block_loc*nblk)

  //       if (nblk_rows_cut>0 .and. nblk_cols_cut>0) then
  //         aux_a_full(1+i_block_loc_fine*nblk : nblk_rows_cut+i_block_loc_fine*nblk, &
  //                    1+j_block_loc_fine*nblk : nblk_cols_cut+j_block_loc_fine*nblk) = &
  //                 a (1+i_block_loc*nblk      : nblk_rows_cut+i_block_loc*nblk, &
  //                    1+j_block_loc*nblk      : nblk_cols_cut+j_block_loc*nblk)
  //       endif

  //     enddo ! i_block_loc_fine
  //   enddo ! j_block_loc_fine
  // endif ! (mod(np_t_fine,np_cols) == my_pcol)

  int di0 = threadIdx.x; // 0..nblk
  int dj0 = blockIdx.x ; // 0..nblk

  T Zero = elpaDeviceNumber<T>(0.0);

  int i_block_loc, j_block_loc, i_block_loc_fine, j_block_loc_fine,  nblk_rows_cut, nblk_cols_cut, di, dj;

  if (np_t_fine%np_cols == my_pcol) 
    {
    for (j_block_loc_fine=0; j_block_loc_fine<nblk_mult_max/nblk; j_block_loc_fine++) 
      {
      j_block_loc = (np_t_fine + j_block_loc_fine*np_dirs_fine)/np_cols;
      
      for (i_block_loc_fine=0; i_block_loc_fine<nblk_mult/nblk; i_block_loc_fine++) 
        {
        i_block_loc = (np_ab_fine + i_block_loc_fine*np_dirs_fine)/np_rows;
        
        nblk_cols_cut = min(nblk, l_cols - j_block_loc*nblk);
        nblk_rows_cut = min(nblk, l_rows - i_block_loc*nblk);

        if (nblk_rows_cut>0 && nblk_cols_cut>0) 
          {
          for (dj=dj0; dj<nblk_cols_cut; dj += gridDim.x)
            for (di=di0; di<nblk_rows_cut; di += blockDim.x)
              aux_a_full_dev[di + i_block_loc_fine*nblk + (dj + j_block_loc_fine*nblk)*nblk_mult] 
                     = a_dev[di + i_block_loc*nblk + (dj + j_block_loc*nblk)*l_rows];
          }
        }
      }
    }

  // do dnp_ab_t = 0, np_dirs_fine/np_cols-1
  //   np_ab_t_fine = dnp_ab_t*np_cols + my_pcol

  //   do j_block_loc_fine = 0, nblk_mult_max/nblk-1
  //     j_block_loc = (np_ab_t_fine + j_block_loc_fine*np_dirs_fine)/np_cols
      
  //     do i_block_loc_fine = 0, nblk_mult/nblk-1
  //       i_block_loc = (np_ab_fine + i_block_loc_fine*np_dirs_fine)/np_rows

  //       nblk_rows_cut = min(nblk, l_rows - i_block_loc*nblk)
  //       nblk_cols_cut = min(nblk, l_cols - j_block_loc*nblk)
        
  //       if (nblk_rows_cut>0 .and. nblk_cols_cut>0) then
  //         aux_b_full(1+i_block_loc_fine*nblk : nblk_rows_cut+i_block_loc_fine*nblk, &
  //                   1            +j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max : &
  //                   nblk_cols_cut+j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max) = &
  //                   b(1+i_block_loc*nblk      :nblk_rows_cut+i_block_loc*nblk, &
  //                     1+j_block_loc*nblk      :nblk_cols_cut+j_block_loc*nblk)
  //       endif

  //       ! nullify the unused part of the block in b
  //       if (nblk_rows_cut<nblk) then
  //         if (nblk_rows_cut>0) then
  //           aux_b_full(nblk_rows_cut+1+i_block_loc_fine*nblk : nblk+i_block_loc_fine*nblk, &
  //                       1            +j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max : &
  //                       nblk_cols_cut+j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max) = 0
  //         else ! for negative nblk_rows_cut we nullify the whole block (it's locally absent)
  //           aux_b_full(1+i_block_loc_fine*nblk : nblk+i_block_loc_fine*nblk, &
  //                       1   +j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max : &
  //                       nblk+j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max) = 0
  //         endif
  //       endif

  //       if (nblk_cols_cut<nblk) then
  //         if (nblk_cols_cut>0) then
  //           aux_b_full(1+i_block_loc_fine*nblk : nblk_rows_cut+i_block_loc_fine*nblk, &
  //                       nblk_cols_cut+1+j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max : &
  //                       nblk           +j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max) = 0
  //         else
  //           aux_b_full(1+i_block_loc_fine*nblk : nblk+i_block_loc_fine*nblk, &
  //                       1   +j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max : &
  //                       nblk+j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max) = 0
  //         endif
  //       endif

  //     enddo ! i_block_loc_fine
  //   enddo ! j_block_loc_fine
  // enddo ! np_ab_t_fine

  int dnp_ab_t, np_ab_t_fine;
  for (dnp_ab_t = 0; dnp_ab_t < np_dirs_fine/np_cols; dnp_ab_t++)
    {
    np_ab_t_fine = dnp_ab_t*np_cols + my_pcol;
    for (j_block_loc_fine = 0; j_block_loc_fine < nblk_mult_max/nblk; j_block_loc_fine++)
      {
      j_block_loc = (np_ab_t_fine + j_block_loc_fine*np_dirs_fine)/np_cols;

      for (i_block_loc_fine = 0; i_block_loc_fine < nblk_mult/nblk; i_block_loc_fine++)
        {
        i_block_loc = (np_ab_fine + i_block_loc_fine*np_dirs_fine)/np_rows;

        nblk_rows_cut = min(nblk, l_rows - i_block_loc*nblk);
        nblk_cols_cut = min(nblk, l_cols - j_block_loc*nblk);

        if (nblk_rows_cut>0 && nblk_cols_cut>0)
          {
          for (dj = dj0; dj < nblk_cols_cut; dj += gridDim.x)
            for (di = di0; di < nblk_rows_cut; di += blockDim.x)
              aux_b_full_dev[di + i_block_loc_fine*nblk + (dj + j_block_loc_fine*nblk + dnp_ab_t*nblk_mult_max)*nblk_mult] 
                    = b_dev[di + i_block_loc*nblk + (dj + j_block_loc*nblk)*l_rows];
          }

        // nullify the unused part of the block in b
        if (nblk_rows_cut < nblk)
          {
          if (nblk_rows_cut > 0)
            {
            for (dj = dj0; dj < nblk_cols_cut; dj += gridDim.x)
              for (di = nblk_rows_cut+di0; di < nblk; di += blockDim.x)
                aux_b_full_dev[di + i_block_loc_fine*nblk + (dj + j_block_loc_fine*nblk + dnp_ab_t*nblk_mult_max)*nblk_mult] = Zero;
            }
          else // for negative nblk_rows_cut we nullify the whole block (it's locally absent)
            {
            for (dj = dj0; dj < nblk_cols_cut; dj += gridDim.x)
              for (di = di0; di < nblk; di += blockDim.x)
                aux_b_full_dev[di + i_block_loc_fine*nblk + (dj + j_block_loc_fine*nblk + dnp_ab_t*nblk_mult_max)*nblk_mult] = Zero;
            }
          }

        if (nblk_cols_cut < nblk)
          {
          if (nblk_cols_cut > 0)
            {
            for (dj = nblk_cols_cut+dj0; dj < nblk; dj += gridDim.x)
              for (di = di0; di < nblk_rows_cut; di += blockDim.x)
                aux_b_full_dev[di + i_block_loc_fine*nblk + (dj + j_block_loc_fine*nblk + dnp_ab_t*nblk_mult_max)*nblk_mult] = Zero;
            }
          else 
            {
            for (dj = dj0; dj < nblk; dj += gridDim.x)
              for (di = di0; di < nblk_rows_cut; di += blockDim.x)
                aux_b_full_dev[di + i_block_loc_fine*nblk + (dj + j_block_loc_fine*nblk + dnp_ab_t*nblk_mult_max)*nblk_mult] = Zero;
            }
          }
       }
        
      }
    }
}

template <typename T>
__global__ void cuda_copy_and_set_zeros_aux_ab_full_nt_kernel(T *a_dev, T *b_dev, T *aux_a_full_dev, T *aux_b_full_dev,
                                                              int l_rows, int l_cols, int nblk_mult_max, int nblk_mult, int nblk,
                                                              int np_ab_fine, int np_rows, int my_prow,
                                                              int np_t_fine , int np_cols, int my_pcol,
                                                              int np_dirs_fine){
  int di0 = threadIdx.x;
  int dj0 = blockIdx.x;

  T Zero = elpaDeviceNumber<T>(0.0);

  int i_block_loc, j_block_loc, i_block_loc_fine, j_block_loc_fine, nblk_rows_cut, nblk_cols_cut, di, dj;
  int dnp_ab_t, np_ab_t_fine;

  if (np_t_fine%np_rows == my_prow) 
    {
    for (i_block_loc_fine=0; i_block_loc_fine<nblk_mult_max/nblk; i_block_loc_fine++) 
      {
      i_block_loc = (np_t_fine + i_block_loc_fine*np_dirs_fine)/np_rows;

      for (j_block_loc_fine=0; j_block_loc_fine<nblk_mult/nblk; j_block_loc_fine++) 
        {
        j_block_loc = (np_ab_fine + j_block_loc_fine*np_dirs_fine)/np_cols;

        nblk_rows_cut = min(nblk, l_rows - i_block_loc*nblk);
        nblk_cols_cut = min(nblk, l_cols - j_block_loc*nblk);

        if (nblk_rows_cut> 0 && nblk_cols_cut> 0)
          {
          for (dj=dj0; dj<nblk_cols_cut; dj += gridDim.x)
            for (di=di0; di<nblk_rows_cut; di += blockDim.x)          
              aux_b_full_dev[di + i_block_loc_fine*nblk + (dj + j_block_loc_fine*nblk)*nblk_mult] =
                       b_dev[di + i_block_loc*nblk + (dj + j_block_loc*nblk)*l_rows];
          }

        // Nullify the unused part of the block in b
        if (nblk_rows_cut < nblk)
          {
          if (nblk_rows_cut > 0)
            {
            for (dj = dj0; dj < nblk_cols_cut; dj += gridDim.x)
              for (di = nblk_rows_cut + di0; di < nblk; di += blockDim.x)
                aux_b_full_dev[di + i_block_loc_fine*nblk + (dj + j_block_loc_fine*nblk)*nblk_mult] = Zero;
            }
          else // For negative nblk_rows_cut we nullify the whole block  (it's locally absent)
            {
            for (dj = dj0; dj < nblk_cols_cut; dj += gridDim.x)
              for (di = di0; di < nblk; di += blockDim.x)            
                aux_b_full_dev[di + i_block_loc_fine*nblk + (dj + j_block_loc_fine*nblk)*nblk_mult] = Zero;
            }
          }
        

        if (nblk_cols_cut < nblk)
          {
          if (nblk_cols_cut > 0)
            {
            for (dj = nblk_cols_cut + dj0; dj < nblk; dj += gridDim.x)
              for (di = di0; di < nblk_rows_cut; di += blockDim.x)
                aux_b_full_dev[di + i_block_loc_fine*nblk + (dj + j_block_loc_fine * nblk)*nblk_mult] = Zero;
            }
          else
            {
            for (dj = dj0; dj < nblk; dj += gridDim.x)
              for (di = di0; di < nblk_rows_cut; di += blockDim.x)
                aux_b_full_dev[di + i_block_loc_fine*nblk + (dj + j_block_loc_fine * nblk)*nblk_mult] = Zero;
            }
          }
        }
      }
    }


  for (dnp_ab_t = 0; dnp_ab_t < np_dirs_fine/np_rows; dnp_ab_t++)
    {
    np_ab_t_fine = dnp_ab_t*np_rows + my_prow;

    for (i_block_loc_fine = 0; i_block_loc_fine < nblk_mult_max/nblk; i_block_loc_fine++)
      {
      i_block_loc = (np_ab_t_fine + i_block_loc_fine*np_dirs_fine)/np_rows;

      for (j_block_loc_fine = 0; j_block_loc_fine < nblk_mult/nblk; j_block_loc_fine++)
        {
        j_block_loc = (np_ab_fine + j_block_loc_fine*np_dirs_fine)/np_cols;

        nblk_rows_cut = min(nblk, l_rows - i_block_loc*nblk);
        nblk_cols_cut = min(nblk, l_cols - j_block_loc*nblk);

        if (nblk_rows_cut > 0 && nblk_cols_cut > 0)
          {
          for (dj = dj0; dj < nblk_cols_cut; dj += gridDim.x)
            for (di = di0; di < nblk_rows_cut; di += blockDim.x)
              aux_a_full_dev[di + i_block_loc_fine*nblk + dnp_ab_t*nblk_mult_max + (dj + j_block_loc_fine*nblk)*nblk_mult] =
                       a_dev[di + i_block_loc*nblk + (dj + j_block_loc*nblk)*l_rows];
                    
                
          }
        }
      }
    }
}

template <typename T>
void cuda_copy_and_set_zeros_aux_ab_full_tn_nt(int *a_transoposed_in, T *a_dev, T *b_dev, T *aux_a_full_dev, T *aux_b_full_dev,
                                            int *l_rows_in, int *l_cols_in, int *nblk_mult_max_in, int *nblk_mult_in, int *nblk_in,
                                            int *np_ab_fine_in, int *np_rows_in, int *my_prow_in,
                                            int *np_t_fine_in, int *np_cols_in, int *my_pcol_in,
                                            int *np_dirs_fine_in,int *SM_count_in,
                                            int *debug_in, cudaStream_t my_stream){
    
    int a_transoposed = *a_transoposed_in;
    int l_rows = *l_rows_in;
    int l_cols = *l_cols_in;
    int nblk_mult_max = *nblk_mult_max_in;
    int nblk_mult = *nblk_mult_in;
    int nblk = *nblk_in;
    int np_ab_fine = *np_ab_fine_in;
    int np_rows = *np_rows_in;
    int my_prow = *my_prow_in;
    int np_t_fine = *np_t_fine_in;
    int np_cols = *np_cols_in;
    int my_pcol = *my_pcol_in;
    int np_dirs_fine = *np_dirs_fine_in;
    int SM_count = *SM_count_in;
    int debug = *debug_in;

    dim3 blocksPerGrid(SM_count, 1, 1); 
    dim3 threadsPerBlock(min(nblk, MAX_THREADS_PER_BLOCK/2), 1, 1); // use only half of the max threads due to high register usage

    if (a_transoposed)
      {
#ifdef WITH_GPU_STREAMS
      cuda_copy_and_set_zeros_aux_ab_full_tn_kernel<<<blocksPerGrid, threadsPerBlock, 0, my_stream>>>(
          a_dev, b_dev, aux_a_full_dev, aux_b_full_dev,
          l_rows, l_cols, nblk_mult_max, nblk_mult, nblk,
          np_ab_fine, np_rows, my_prow,
          np_t_fine, np_cols, my_pcol,
          np_dirs_fine);
#else
      cuda_copy_and_set_zeros_aux_ab_full_tn_kernel<<<blocksPerGrid, threadsPerBlock>>>(
          a_dev, b_dev, aux_a_full_dev, aux_b_full_dev,
          l_rows, l_cols, nblk_mult_max, nblk_mult, nblk,
          np_ab_fine, np_rows, my_prow,
          np_t_fine, np_cols, my_pcol,
          np_dirs_fine);
#endif
      }
    else 
      {
#ifdef WITH_GPU_STREAMS
      cuda_copy_and_set_zeros_aux_ab_full_nt_kernel<<<blocksPerGrid, threadsPerBlock, 0, my_stream>>>(
          a_dev, b_dev, aux_a_full_dev, aux_b_full_dev,
          l_rows, l_cols, nblk_mult_max, nblk_mult, nblk,
          np_ab_fine, np_rows, my_prow,
          np_t_fine, np_cols, my_pcol,
          np_dirs_fine);
#else
      cuda_copy_and_set_zeros_aux_ab_full_nt_kernel<<<blocksPerGrid, threadsPerBlock>>>(
          a_dev, b_dev, aux_a_full_dev, aux_b_full_dev,
          l_rows, l_cols, nblk_mult_max, nblk_mult, nblk,
          np_ab_fine, np_rows, my_prow,
          np_t_fine, np_cols, my_pcol,
          np_dirs_fine);
#endif
      }
    if (debug)
    {
        cudaDeviceSynchronize();
        cudaError_t cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess)
        {
            printf("Error in executing cuda_copy_and_set_zeros_aux_ab_full_tn: %s\n", cudaGetErrorString(cuerr));
        }
    }
}

extern "C" void cuda_copy_and_set_zeros_aux_ab_full_tn_nt_FromC(char dataType, int *a_transoposed_in, intptr_t a_dev, intptr_t b_dev, intptr_t aux_a_full_dev, intptr_t aux_b_full_dev,
                                                             int *l_rows_in, int *l_cols_in, int *nblk_mult_max_in, int *nblk_mult_in, int *nblk_in,
                                                             int *np_ab_fine_in, int *np_rows_in, int *my_prow_in,
                                                             int *np_t_fine_in , int *np_cols_in, int *my_pcol_in,
                                                             int *np_dirs_fine_in,
                                                             int *SM_count_in, int *debug_in, cudaStream_t my_stream){
  if (dataType == 'D') cuda_copy_and_set_zeros_aux_ab_full_tn_nt<double>(a_transoposed_in, (double *)a_dev, (double *)b_dev, (double *)aux_a_full_dev, (double *)aux_b_full_dev,
                                                       l_rows_in, l_cols_in, nblk_mult_max_in, nblk_mult_in, nblk_in,
                                                       np_ab_fine_in, np_rows_in, my_prow_in,
                                                       np_t_fine_in , np_cols_in, my_pcol_in,
                                                       np_dirs_fine_in,
                                                       SM_count_in, debug_in, my_stream);
  if (dataType == 'S') cuda_copy_and_set_zeros_aux_ab_full_tn_nt<float>(a_transoposed_in, (float *)a_dev, (float *)b_dev, (float *)aux_a_full_dev, (float *)aux_b_full_dev,
                                                      l_rows_in, l_cols_in, nblk_mult_max_in, nblk_mult_in, nblk_in,
                                                      np_ab_fine_in, np_rows_in, my_prow_in,
                                                      np_t_fine_in , np_cols_in, my_pcol_in,
                                                      np_dirs_fine_in,
                                                      SM_count_in, debug_in, my_stream);
  if (dataType == 'Z') cuda_copy_and_set_zeros_aux_ab_full_tn_nt<cuDoubleComplex>(a_transoposed_in, (cuDoubleComplex *)a_dev, (cuDoubleComplex *)b_dev, (cuDoubleComplex *)aux_a_full_dev, (cuDoubleComplex *)aux_b_full_dev,
                                                                l_rows_in, l_cols_in, nblk_mult_max_in, nblk_mult_in, nblk_in,
                                                                np_ab_fine_in, np_rows_in, my_prow_in,
                                                                np_t_fine_in , np_cols_in, my_pcol_in,
                                                                np_dirs_fine_in,
                                                                SM_count_in, debug_in, my_stream);
  if (dataType == 'C') cuda_copy_and_set_zeros_aux_ab_full_tn_nt<cuFloatComplex>(a_transoposed_in, (cuFloatComplex *)a_dev, (cuFloatComplex *)b_dev, (cuFloatComplex *)aux_a_full_dev, (cuFloatComplex *)aux_b_full_dev,
                                                               l_rows_in, l_cols_in, nblk_mult_max_in, nblk_mult_in, nblk_in,
                                                               np_ab_fine_in, np_rows_in, my_prow_in,
                                                               np_t_fine_in , np_cols_in, my_pcol_in,
                                                               np_dirs_fine_in,
                                                               SM_count_in, debug_in, my_stream);
}

//________________________________________________________________


