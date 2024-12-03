//    Copyright 2021, A. Marek
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

#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <complex.h>
#include "hip/hip_runtime.h"
#include <hip/hip_complex.h>
#include <stdint.h>
#include "config-f90.h"

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

//extern hipStream_t elpa_hip_stm;

#define BLK_X_DIM      32
#define BLK_Y_DIM      4
#define BLK_ALG        1
#define BLK_VERB       0

template <typename T>
__global__ void hip_copy_a_tmat2_kernel(
        T const* __restrict__ a_dev,
        T* __restrict__ tmat2_dev,
        const int nblk,
        const int matrixRows,
        const int l_colx,
        const int l_row1)
{
    int x = blockIdx.x * BLK_X_DIM + threadIdx.x;
    int y = blockIdx.y * BLK_X_DIM + threadIdx.y;
    int j;

    for (j = 0; j < BLK_X_DIM; j += BLK_Y_DIM) {
        tmat2_dev[(l_colx - 1 + y + j) * nblk + x] = a_dev[(l_colx - 1 + y + j) * matrixRows + x + l_row1 - 1];
    }
}

__global__ void hip_copy_double_a_tmat2_kernel(double *a_dev, double *tmat2_dev, const int nblk, const int matrixRows, const int l_colx, const int l_row1){

  int nb_index    = threadIdx.x +1; // range 1..nb
  int l_col_index = blockIdx.x + 1; // range 1..l_colx-l_cols-1

  tmat2_dev[nb_index-1 + (l_colx-1 + l_col_index -1) * nblk] = a_dev[l_row1-1 + nb_index-1 + (l_colx-1 + l_col_index -1)  * matrixRows];

}

extern "C" void hip_copy_double_a_tmat2_FromC(double *a_dev, double *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, hipStream_t my_stream){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;
  int nb     = *nb_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

#if 0
  dim3 blocks = dim3(l_cols-l_colx+1,1,1);
  dim3 threadsPerBlock = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_double_a_tmat2_kernel, blocks, threadsPerBlock, 0, my_stream, a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#else
  hipLaunchKernelGGL(hip_copy_double_a_tmat2_kernel, blocks, threadsPerBlock, 0, 0, a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#endif

#else /* if 0 */
  if ((l_cols-l_colx+1) % BLK_X_DIM == 0 && nb % BLK_X_DIM == 0 && BLK_ALG) {
      dim3 blocks = dim3(nb / BLK_X_DIM, (l_cols-l_colx+1) / BLK_X_DIM);
      dim3 threadsPerBlock = dim3(BLK_X_DIM, BLK_Y_DIM);

#if BLK_VERB
      printf("called elpa_invert_trm_hip tmat2 double ...... hipcc\n");


#endif

#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_a_tmat2_kernel<double>, blocks, threadsPerBlock, 0, my_stream, a_dev,
              tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#else
      hipLaunchKernelGGL(hip_copy_a_tmat2_kernel<double>, blocks, threadsPerBlock, 0, 0, a_dev,
              tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#endif
  }
  else {
      dim3 blocks = dim3(l_cols-l_colx+1,1,1);
      dim3 threadsPerBlock = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_double_a_tmat2_kernel, blocks, threadsPerBlock, 0, my_stream, a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#else
      hipLaunchKernelGGL(hip_copy_double_a_tmat2_kernel, blocks, threadsPerBlock, 0, 0, a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#endif
  }
#endif /* if 0 */
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_double_a_tmat2_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_float_a_tmat2_kernel(float *a_dev, float *tmat2_dev, const int nblk, const int matrixRows, const int l_colx, const int l_row1){

  int nb_index    = threadIdx.x +1; // range 1..nb
  int l_col_index = blockIdx.x + 1; // range 1..l_colx-l_cols-1

  tmat2_dev[nb_index-1 + (l_colx-1 + l_col_index -1) * nblk] = a_dev[l_row1-1 + nb_index-1 + (l_colx-1 + l_col_index -1)  * matrixRows];
}

extern "C" void hip_copy_float_a_tmat2_FromC(float *a_dev, float *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, hipStream_t my_stream){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;
  int nb     = *nb_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

#if 0
  dim3 blocks = dim3(l_cols-l_colx+1,1,1);
  dim3 threadsPerBlock = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_float_a_tmat2_kernel, blocks, threadsPerBlock, 0, my_stream, a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#else
  hipLaunchKernelGGL(hip_copy_float_a_tmat2_kernel, blocks, threadsPerBlock, 0, 0, a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#endif

#else /* if 0 */
  if ((l_cols-l_colx+1) % BLK_X_DIM == 0 && nb % BLK_X_DIM == 0 && BLK_ALG) {
      dim3 blocks = dim3(nb / BLK_X_DIM, (l_cols-l_colx+1) / BLK_X_DIM);
      dim3 threadsPerBlock = dim3(BLK_X_DIM, BLK_Y_DIM);

#if BLK_VERB
      printf("called elpa_invert_trm_hip tmat2 float ...... hipcc\n");
#endif

#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_a_tmat2_kernel<float>, blocks, threadsPerBlock, 0, my_stream, a_dev,
              tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#else
      hipLaunchKernelGGL(hip_copy_a_tmat2_kernel<float>, blocks, threadsPerBlock, 0, 0, a_dev,
              tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#endif
  }
  else {
      dim3 blocks = dim3(l_cols-l_colx+1,1,1);
      dim3 threadsPerBlock = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_float_a_tmat2_kernel, blocks, threadsPerBlock, 0, my_stream, a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#else
      hipLaunchKernelGGL(hip_copy_float_a_tmat2_kernel, blocks, threadsPerBlock, 0, 0, a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#endif
  }
#endif /* if 0 */
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_float_a_tmat2_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_double_complex_a_tmat2_kernel(hipDoubleComplex *a_dev, hipDoubleComplex *tmat2_dev, const int nblk, const int matrixRows, const int l_colx, const int l_row1){

  int nb_index    = threadIdx.x +1; // range 1..nb
  int l_col_index = blockIdx.x + 1; // range 1..l_colx-l_cols-1

  tmat2_dev[nb_index-1 + (l_colx-1 + l_col_index -1) * nblk] = a_dev[l_row1-1 + nb_index-1 + (l_colx-1 + l_col_index -1)  * matrixRows];

}

extern "C" void hip_copy_double_complex_a_tmat2_FromC(double _Complex *a_dev, double _Complex *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, hipStream_t my_stream){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;
  int nb     = *nb_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

  hipDoubleComplex* a_casted = (hipDoubleComplex*) a_dev;
  hipDoubleComplex* tmat2_casted = (hipDoubleComplex*) tmat2_dev;

#if 0
  dim3 blocks = dim3(l_cols-l_colx+1,1,1);
  dim3 threadsPerBlock = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_double_complex_a_tmat2_kernel, blocks, threadsPerBlock, 0, my_stream, a_casted, tmat2_casted, nblk, matrixRows, l_colx, l_row1);
#else
  hipLaunchKernelGGL(hip_copy_double_complex_a_tmat2_kernel, blocks, threadsPerBlock, 0, 0, a_casted, tmat2_casted, nblk, matrixRows, l_colx, l_row1);
#endif
#else /* if 0 */
  if ((l_cols-l_colx+1) % BLK_X_DIM == 0 && nb % BLK_X_DIM == 0 && BLK_ALG) {
      dim3 blocks = dim3(nb / BLK_X_DIM, (l_cols-l_colx+1) / BLK_X_DIM);
      dim3 threadsPerBlock = dim3(BLK_X_DIM, BLK_Y_DIM);

#if BLK_VERB
      printf("called elpa_invert_trm_hip tmat2 hipDoubleComplex ...... hipcc\n");
#endif

#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_a_tmat2_kernel<hipDoubleComplex>, blocks, threadsPerBlock, 0, my_stream, a_casted,
              tmat2_casted, nblk, matrixRows, l_colx, l_row1);
#else
      hipLaunchKernelGGL(hip_copy_a_tmat2_kernel<hipDoubleComplex>, blocks, threadsPerBlock, 0, 0, a_casted,
              tmat2_casted, nblk, matrixRows, l_colx, l_row1);
#endif
  }
  else {
      dim3 blocks = dim3(l_cols-l_colx+1,1,1);
      dim3 threadsPerBlock = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_double_complex_a_tmat2_kernel, blocks, threadsPerBlock, 0, my_stream, a_casted, tmat2_casted, nblk, matrixRows, l_colx, l_row1);
#else
      hipLaunchKernelGGL(hip_copy_double_complex_a_tmat2_kernel, blocks, threadsPerBlock, 0, 0, a_casted, tmat2_casted, nblk, matrixRows, l_colx, l_row1);
#endif
  }
#endif /* if 0 */
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_double_complex_a_tmat2_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_float_complex_a_tmat2_kernel(hipFloatComplex *a_dev, hipFloatComplex *tmat2_dev, const int nblk, const int matrixRows, const int l_colx, const int l_row1){

  int nb_index    = threadIdx.x +1; // range 1..nb
  int l_col_index = blockIdx.x + 1; // range 1..l_colx-l_cols-1

  tmat2_dev[nb_index-1 + (l_colx-1 + l_col_index -1) * nblk] = a_dev[l_row1-1 + nb_index-1 + (l_colx-1 + l_col_index -1)  * matrixRows];

}

extern "C" void hip_copy_float_complex_a_tmat2_FromC(float _Complex *a_dev, float _Complex *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, hipStream_t my_stream){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;
  int nb     = *nb_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

  hipFloatComplex* a_casted = (hipFloatComplex*) a_dev;
  hipFloatComplex* tmat2_casted = (hipFloatComplex*) tmat2_dev;

  dim3 blocks = dim3(l_cols-l_colx+1,1,1);
  dim3 threadsPerBlock = dim3(nb,1,1);

#if 0

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_float_complex_a_tmat2_kernel, blocks, threadsPerBlock, 0, my_stream, a_casted, tmat2_casted, nblk, matrixRows, l_colx, l_row1);
#else
  hipLaunchKernelGGL(hip_copy_float_complex_a_tmat2_kernel, blocks, threadsPerBlock, 0, 0, a_casted, tmat2_casted, nblk, matrixRows, l_colx, l_row1);
#endif
#else /* if 0 */
  if ((l_cols-l_colx+1) % BLK_X_DIM == 0 && nb % BLK_X_DIM == 0 && BLK_ALG) {
      dim3 blocks = dim3(nb / BLK_X_DIM, (l_cols-l_colx+1) / BLK_X_DIM);
      dim3 threadsPerBlock = dim3(BLK_X_DIM, BLK_Y_DIM);

#if BLK_VERB
      printf("called elpa_invert_trm_hip tmat2 hipFloatComplex ...... hipcc\n");
#endif

#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_a_tmat2_kernel<hipFloatComplex>, blocks, threadsPerBlock, 0, my_stream, a_casted,
              tmat2_casted, nblk, matrixRows, l_colx, l_row1);
#else
      hipLaunchKernelGGL(hip_copy_a_tmat2_kernel<hipFloatComplex>, blocks, threadsPerBlock, 0, 0, a_casted,
              tmat2_casted, nblk, matrixRows, l_colx, l_row1);
#endif
  }
  else {
      dim3 blocks = dim3(l_cols-l_colx+1,1,1);
      dim3 threadsPerBlock = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_float_complex_a_tmat2_kernel, blocks, threadsPerBlock, 0, my_stream, a_casted, tmat2_casted, nblk, matrixRows, l_colx, l_row1);
#else
      hipLaunchKernelGGL(hip_copy_float_complex_a_tmat2_kernel, blocks, threadsPerBlock, 0, 0, a_casted, tmat2_casted, nblk, matrixRows, l_colx, l_row1);
#endif
  }
#endif /* if 0 */
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_float_complex_a_tmat2_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_double_tmp2_tmat2_kernel(double *tmp2_dev, double *tmat2_dev, const int nblk, const int l_col1){

  int nb_index    = threadIdx.x +1;  // range 1..nb
  int l_col_index = blockIdx.x + 1;  // range 1..nb

  tmat2_dev[nb_index-1 + (l_col1-1 + l_col_index -1)*nblk] = tmp2_dev[nb_index-1 + (1 -1 + l_col_index-1)  * nblk];

}

extern "C" void hip_copy_double_tmp2_tmat2_FromC(double *tmp2_dev, double *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, hipStream_t my_stream){
  int nblk   = *nblk_in;   
  int l_col1 = *l_col1_in;
  int nb     = *nb_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

  dim3 blocks = dim3(nb,1,1);
  dim3 threadsPerBlock = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_double_tmp2_tmat2_kernel, blocks, threadsPerBlock, 0, my_stream, tmp2_dev, tmat2_dev, nblk, l_col1);
#else
  hipLaunchKernelGGL(hip_copy_double_tmp2_tmat2_kernel, blocks, threadsPerBlock, 0, 0, tmp2_dev, tmat2_dev, nblk, l_col1);
#endif
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_double_tmp2_tmat2_kernel: %s\n",hipGetErrorString(cuerr));
  }
}


__global__ void hip_copy_float_tmp2_tmat2_kernel(float *tmp2_dev, float *tmat2_dev, const int nblk, const int l_col1){

  int nb_index    = threadIdx.x +1;  // range 1..nb
  int l_col_index = blockIdx.x + 1;  // range 1..nb

  tmat2_dev[nb_index-1 + (l_col1-1 + l_col_index -1)*nblk] = tmp2_dev[nb_index-1 + (1 -1 + l_col_index-1)  * nblk];

}

extern "C" void hip_copy_float_tmp2_tmat2_FromC(float *tmp2_dev, float *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, hipStream_t my_stream){
  int nblk   = *nblk_in;   
  int l_col1 = *l_col1_in;
  int nb     = *nb_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

  dim3 blocks = dim3(nb,1,1);
  dim3 threadsPerBlock = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_float_tmp2_tmat2_kernel, blocks, threadsPerBlock, 0, my_stream, tmp2_dev, tmat2_dev, nblk, l_col1);
#else
  hipLaunchKernelGGL(hip_copy_float_tmp2_tmat2_kernel, blocks, threadsPerBlock, 0, 0, tmp2_dev, tmat2_dev, nblk, l_col1);
#endif
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_float_tmp2_tmat2_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_double_complex_tmp2_tmat2_kernel(hipDoubleComplex *tmp2_dev, hipDoubleComplex *tmat2_dev, const int nblk, const int l_col1){

  int nb_index    = threadIdx.x +1;  // range 1..nb
  int l_col_index = blockIdx.x + 1;  // range 1..nb

  tmat2_dev[nb_index-1 + (l_col1-1 + l_col_index -1)*nblk] = tmp2_dev[nb_index-1 + (1 -1 + l_col_index-1)  * nblk];

}

extern "C" void hip_copy_double_complex_tmp2_tmat2_FromC(double _Complex *tmp2_dev, double _Complex *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, hipStream_t my_stream){
  int nblk   = *nblk_in;   
  int l_col1 = *l_col1_in;
  int nb     = *nb_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

  hipDoubleComplex* tmp2_casted = (hipDoubleComplex*) tmp2_dev;
  hipDoubleComplex* tmat2_casted = (hipDoubleComplex*) tmat2_dev;


  dim3 blocks = dim3(nb,1,1);
  dim3 threadsPerBlock = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_double_complex_tmp2_tmat2_kernel, blocks, threadsPerBlock, 0, my_stream, tmp2_casted, tmat2_casted, nblk, l_col1);
#else
  hipLaunchKernelGGL(hip_copy_double_complex_tmp2_tmat2_kernel, blocks, threadsPerBlock, 0, 0, tmp2_casted, tmat2_casted, nblk, l_col1);
#endif
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_double_complex_tmp2_tmat2_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_float_complex_tmp2_tmat2_kernel(hipFloatComplex *tmp2_dev, hipFloatComplex *tmat2_dev, const int nblk, const int l_col1){

  int nb_index    = threadIdx.x +1;  // range 1..nb
  int l_col_index = blockIdx.x + 1;  // range 1..nb

  tmat2_dev[nb_index-1 + (l_col1-1 + l_col_index -1)*nblk] = tmp2_dev[nb_index-1 + (1 -1 + l_col_index-1)  * nblk];

}

extern "C" void hip_copy_float_complex_tmp2_tmat2_FromC(float _Complex *tmp2_dev, float _Complex *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, hipStream_t my_stream){
  int nblk   = *nblk_in;   
  int l_col1 = *l_col1_in;
  int nb     = *nb_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

  hipFloatComplex* tmp2_casted = (hipFloatComplex*) tmp2_dev;
  hipFloatComplex* tmat2_casted = (hipFloatComplex*) tmat2_dev;


  dim3 blocks = dim3(nb,1,1);
  dim3 threadsPerBlock = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_float_complex_tmp2_tmat2_kernel, blocks, threadsPerBlock, 0, my_stream, tmp2_casted, tmat2_casted, nblk, l_col1);
#else
  hipLaunchKernelGGL(hip_copy_float_complex_tmp2_tmat2_kernel, blocks, threadsPerBlock, 0, 0, tmp2_casted, tmat2_casted, nblk, l_col1);
#endif
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_float_complex_tmp2_tmat2_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

template <typename T>
__global__ void hip_copy_a_tmat1_kernel(
        T * __restrict__ a_dev,
        T * __restrict__ tmat1_dev,
        const int l_rows,
        const int matrixRows,
        const int l_col1,
        const int nb,
        const int l_row1)
{
    int x = blockIdx.x * BLK_X_DIM + threadIdx.x;
    int y = blockIdx.y * BLK_X_DIM + threadIdx.y;
    int j;

    for (j = 0; j < BLK_X_DIM; j += BLK_Y_DIM) {
        tmat1_dev[(y + j) * l_rows + x] = a_dev[(l_col1 - 1 + y + j) * matrixRows + x];
    }

    __syncthreads();

    for (j = 0; j < BLK_X_DIM; j += BLK_Y_DIM) {
        a_dev[(l_col1 - 1 + y + j) * matrixRows + x] = 0;
    }
}

__global__ void hip_copy_double_a_tmat1_kernel(double *a_dev, double *tmat1_dev, const int l_rows, const int matrixRows, const int l_col1, const int nb, const int l_row1){

  int nb_index    = threadIdx.x +1;  // range 1..nb
  int l_row1_index = blockIdx.x + 1; // we need l_row1-1 blocks

  tmat1_dev[l_row1_index-1 + (nb_index-1)*l_rows] = a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1 ) * matrixRows];
  a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1)*matrixRows] = 0;

}

extern "C" void hip_copy_double_a_tmat1_FromC(double *a_dev, double *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, hipStream_t my_stream){
  int l_rows = *l_rows_in;   
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

#if 0
  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(l_row1-1,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_double_a_tmat1_kernel, blocks, threadsPerBlock, 0, my_stream, a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#else
  hipLaunchKernelGGL(hip_copy_double_a_tmat1_kernel, blocks, threadsPerBlock, 0, 0, a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#endif

#else /* if 0 */
  if ((l_row1-1) % BLK_X_DIM == 0 && nb % BLK_X_DIM == 0 && BLK_ALG) {
      dim3 threadsPerBlock = dim3(BLK_X_DIM, BLK_Y_DIM);
      dim3 blocks = dim3((l_row1-1) / BLK_X_DIM, nb / BLK_X_DIM);

#if BLK_VERB
      printf("called elpa_invert_trm_hip tmat1 double ...... hipcc\n");
#endif
#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_a_tmat1_kernel<double>, blocks, threadsPerBlock, 0, my_stream, a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#else
      hipLaunchKernelGGL(hip_copy_a_tmat1_kernel<double>, blocks, threadsPerBlock, 0, 0, a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#endif
  }
  else {
      dim3 threadsPerBlock = dim3(nb, 1, 1);
      dim3 blocks = dim3(l_row1-1,1,1);

#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_double_a_tmat1_kernel, blocks, threadsPerBlock, 0, my_stream, a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#else
      hipLaunchKernelGGL(hip_copy_double_a_tmat1_kernel, blocks, threadsPerBlock, 0, 0, a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#endif
  }
#endif /* if 0 */
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_double_a_tmat1_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_float_a_tmat1_kernel(float *a_dev, float *tmat1_dev, const int l_rows, const int matrixRows, const int l_col1, const int nb, const int l_row1){

  int nb_index    = threadIdx.x +1;  // range 1..nb
  int l_row1_index = blockIdx.x + 1; // we need l_row1-1 blocks

  tmat1_dev[l_row1_index-1 + (nb_index-1)*l_rows] = a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1 ) * matrixRows];
  a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1)*matrixRows] = 0;
}

extern "C" void hip_copy_float_a_tmat1_FromC(float *a_dev, float *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, hipStream_t my_stream){
  int l_rows = *l_rows_in;   
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

#if 0
  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(l_row1-1,1,1);
  //dim3 threadsPerBlock = dim3(1, 1, 1);
  //dim3 blocks = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_float_a_tmat1_kernel, blocks, threadsPerBlock, 0, my_stream, a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#else
  hipLaunchKernelGGL(hip_copy_float_a_tmat1_kernel, blocks, threadsPerBlock, 0, 0, a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#endif
#else /* if 0 */
  if ((l_row1-1) % BLK_X_DIM == 0 && nb % BLK_X_DIM == 0 && BLK_ALG) {
      dim3 threadsPerBlock = dim3(BLK_X_DIM, BLK_Y_DIM);
      dim3 blocks = dim3((l_row1-1) / BLK_X_DIM, nb / BLK_X_DIM);

#if BLK_VERB
      printf("called elpa_invert_trm_hip tmat1 float ...... hipcc\n");
#endif
#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_a_tmat1_kernel<float>, blocks, threadsPerBlock, 0, my_stream, a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#else
      hipLaunchKernelGGL(hip_copy_a_tmat1_kernel<float>, blocks, threadsPerBlock, 0, 0, a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#endif
  }
  else {
      dim3 threadsPerBlock = dim3(nb, 1, 1);
      dim3 blocks = dim3(l_row1-1,1,1);
      //dim3 threadsPerBlock = dim3(1, 1, 1);
      //dim3 blocks = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_float_a_tmat1_kernel, blocks, threadsPerBlock, 0, my_stream, a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#else
      hipLaunchKernelGGL(hip_copy_float_a_tmat1_kernel, blocks, threadsPerBlock, 0, 0, a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#endif
  }
#endif /* if 0 */
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_float_a_tmat1_kernel: %s\n",hipGetErrorString(cuerr));
  }
}
template <typename T>
__global__ void hip_copy_a_tmat1_complex_kernel(
        T * __restrict__ a_dev,
        T * __restrict__ tmat1_dev,
        const int l_rows,
        const int matrixRows,
        const int l_col1,
        const int nb,
        const int l_row1)
{
    int x = blockIdx.x * BLK_X_DIM + threadIdx.x;
    int y = blockIdx.y * BLK_X_DIM + threadIdx.y;
    int j;

    for (j = 0; j < BLK_X_DIM; j += BLK_Y_DIM) {
        tmat1_dev[(y + j) * l_rows + x] = a_dev[(l_col1 - 1 + y + j) * matrixRows + x];
    }

    __syncthreads();

    for (j = 0; j < BLK_X_DIM; j += BLK_Y_DIM) {
        a_dev[(l_col1 - 1 + y + j) * matrixRows + x].x = 0;
        a_dev[(l_col1 - 1 + y + j) * matrixRows + x].y = 0;
    }
}

__global__ void hip_copy_double_complex_a_tmat1_kernel(hipDoubleComplex *a_dev, hipDoubleComplex *tmat1_dev, const int l_rows, const int matrixRows, const int l_col1, const int nb, const int l_row1){

  int nb_index    = threadIdx.x +1;  // range 1..nb
  int l_row1_index = blockIdx.x + 1; // we need l_row1-1 blocks

  tmat1_dev[l_row1_index-1 + (nb_index-1)*l_rows] = a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1 ) * matrixRows];
  a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1)*matrixRows].x = 0;
  a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1)*matrixRows].y = 0;
}

extern "C" void hip_copy_double_complex_a_tmat1_FromC(double _Complex *a_dev, double _Complex *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, hipStream_t my_stream){
  int l_rows = *l_rows_in;   
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif


  hipDoubleComplex* a_casted = (hipDoubleComplex*) a_dev;
  hipDoubleComplex* tmat1_casted = (hipDoubleComplex*) tmat1_dev;

#if 0
  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(l_row1-1,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_double_complex_a_tmat1_kernel, blocks, threadsPerBlock, 0, my_stream, a_casted, tmat1_casted, l_rows, matrixRows, l_col1, nb, l_row1);
#else
  hipLaunchKernelGGL(hip_copy_double_complex_a_tmat1_kernel, blocks, threadsPerBlock, 0, 0, a_casted, tmat1_casted, l_rows, matrixRows, l_col1, nb, l_row1);
#endif

#else /* if 0 */
  if ((l_row1-1) % BLK_X_DIM == 0 && nb % BLK_X_DIM == 0 && BLK_ALG) {
      dim3 threadsPerBlock = dim3(BLK_X_DIM, BLK_Y_DIM);
      dim3 blocks = dim3((l_row1-1) / BLK_X_DIM, nb / BLK_X_DIM);

#if BLK_VERB
      printf("called elpa_invert_trm_hip tmat1 double complex ...... hipcc\n");
#endif
#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_a_tmat1_complex_kernel<hipDoubleComplex>, blocks, threadsPerBlock, 0, my_stream, 
              a_casted, tmat1_casted, l_rows, matrixRows, l_col1, nb, l_row1);
#else
      hipLaunchKernelGGL(hip_copy_a_tmat1_complex_kernel<hipDoubleComplex>, blocks, threadsPerBlock, 0, 0, 
              a_casted, tmat1_casted, l_rows, matrixRows, l_col1, nb, l_row1);
#endif
  }
  else {
      dim3 threadsPerBlock = dim3(nb, 1, 1);
      dim3 blocks = dim3(l_row1-1,1,1);

#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_double_complex_a_tmat1_kernel, blocks, threadsPerBlock, 0, my_stream, a_casted, tmat1_casted, l_rows, matrixRows, l_col1, nb, l_row1);
#else
      hipLaunchKernelGGL(hip_copy_double_complex_a_tmat1_kernel, blocks, threadsPerBlock, 0, 0, a_casted, tmat1_casted, l_rows, matrixRows, l_col1, nb, l_row1);
#endif
  }
#endif /* if 0 */

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_double_complex_a_tmat1_kernel: %s\n",hipGetErrorString(cuerr));
  }
}


__global__ void hip_copy_float_complex_a_tmat1_kernel(hipFloatComplex *a_dev, hipFloatComplex *tmat1_dev, const int l_rows, const int matrixRows, const int l_col1, const int nb, const int l_row1){

  int nb_index    = threadIdx.x +1;  // range 1..nb
  int l_row1_index = blockIdx.x + 1; // we need l_row1-1 blocks

  tmat1_dev[l_row1_index-1 + (nb_index-1)*l_rows] = a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1 ) * matrixRows];
  a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1)*matrixRows].x = 0;
  a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1)*matrixRows].y = 0;
}

extern "C" void hip_copy_float_complex_a_tmat1_FromC(float _Complex *a_dev, float _Complex *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, hipStream_t my_stream){
  int l_rows = *l_rows_in;   
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

  hipFloatComplex* a_casted = (hipFloatComplex*) a_dev;
  hipFloatComplex* tmat1_casted = (hipFloatComplex*) tmat1_dev;

#if 0
  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(l_row1-1,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_float_complex_a_tmat1_kernel, blocks, threadsPerBlock, 0, my_stream, a_casted, tmat1_casted, l_rows, matrixRows, l_col1, nb, l_row1);
#else
  hipLaunchKernelGGL(hip_copy_float_complex_a_tmat1_kernel, blocks, threadsPerBlock, 0, 0, a_casted, tmat1_casted, l_rows, matrixRows, l_col1, nb, l_row1);
#endif

#else /* if 0 */
  if ((l_row1-1) % BLK_X_DIM == 0 && nb % BLK_X_DIM == 0 && BLK_ALG) {
      dim3 threadsPerBlock = dim3(BLK_X_DIM, BLK_Y_DIM);
      dim3 blocks = dim3((l_row1-1) / BLK_X_DIM, nb / BLK_X_DIM);

#if BLK_VERB
      printf("called elpa_invert_trm_hip tmat1 float complex ...... hipcc\n");
#endif
#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_a_tmat1_complex_kernel<hipFloatComplex>, blocks, threadsPerBlock, 0, my_stream, 
              a_casted, tmat1_casted, l_rows, matrixRows, l_col1, nb, l_row1);
#else
      hipLaunchKernelGGL(hip_copy_a_tmat1_complex_kernel<hipFloatComplex>, blocks, threadsPerBlock, 0, 0, 
              a_casted, tmat1_casted, l_rows, matrixRows, l_col1, nb, l_row1);
#endif
  }
  else {
      dim3 threadsPerBlock = dim3(nb, 1, 1);
      dim3 blocks = dim3(l_row1-1,1,1);

#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_float_complex_a_tmat1_kernel, blocks, threadsPerBlock, 0, my_stream, a_casted, tmat1_casted, l_rows, matrixRows, l_col1, nb, l_row1);
#else
      hipLaunchKernelGGL(hip_copy_float_complex_a_tmat1_kernel, blocks, threadsPerBlock, 0, 0, a_casted, tmat1_casted, l_rows, matrixRows, l_col1, nb, l_row1);
#endif
  }
#endif /* if 0 */

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_float_complex_a_tmat1_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_double_tmp1_tmp2_kernel(double *tmp1_dev, double *tmp2_dev, const int nblk, const int nb){

  int i_index    = threadIdx.x +1;  // range 1..nb
  int j_index    = blockIdx.x +1;  // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp2_dev[1-1 + j_index-1 + (i_index-1)*nblk] = tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1];
  }
}


extern "C" void hip_copy_double_tmp1_tmp2_FromC(double *tmp1_dev, double *tmp2_dev, int *nblk_in, int *nb_in, hipStream_t my_stream){
  int nblk = *nblk_in;
  int nb = *nb_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_double_tmp1_tmp2_kernel, blocks, threadsPerBlock, 0, my_stream, tmp1_dev, tmp2_dev, nblk, nb);
#else
  hipLaunchKernelGGL(hip_copy_double_tmp1_tmp2_kernel, blocks, threadsPerBlock, 0, 0, tmp1_dev, tmp2_dev, nblk, nb);
#endif
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_double_tmp1_tmp2_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_float_tmp1_tmp2_kernel(float *tmp1_dev, float *tmp2_dev, const int nblk, const int nb){

  int i_index    = threadIdx.x +1;  // range 1..nb
  int j_index    = blockIdx.x +1;  // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp2_dev[1-1 + j_index-1 + (i_index-1)*nblk] = tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1];
  }
}


extern "C" void hip_copy_float_tmp1_tmp2_FromC(float *tmp1_dev, float *tmp2_dev, int *nblk_in, int *nb_in, hipStream_t my_stream){
  int nblk = *nblk_in;
  int nb = *nb_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_float_tmp1_tmp2_kernel, blocks, threadsPerBlock, 0, my_stream, tmp1_dev, tmp2_dev, nblk, nb);
#else
  hipLaunchKernelGGL(hip_copy_float_tmp1_tmp2_kernel, blocks, threadsPerBlock, 0, 0, tmp1_dev, tmp2_dev, nblk, nb);
#endif
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_float_tmp1_tmp2_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_double_complex_tmp1_tmp2_kernel(hipDoubleComplex *tmp1_dev, hipDoubleComplex *tmp2_dev, const int nblk, const int nb){

  int i_index    = threadIdx.x +1;  // range 1..nb
  int j_index    = blockIdx.x +1;  // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp2_dev[1-1 + j_index-1 + (i_index-1)*nblk] = tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1];
  }
}


extern "C" void hip_copy_double_complex_tmp1_tmp2_FromC(double _Complex *tmp1_dev, double _Complex *tmp2_dev, int *nblk_in, int *nb_in, hipStream_t my_stream){
  int nblk = *nblk_in;
  int nb = *nb_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(nb,1,1);

  hipDoubleComplex* tmp1_casted = (hipDoubleComplex*) tmp1_dev;
  hipDoubleComplex* tmp2_casted = (hipDoubleComplex*) tmp2_dev;


#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_double_complex_tmp1_tmp2_kernel, blocks, threadsPerBlock, 0, my_stream, tmp1_casted, tmp2_casted, nblk, nb);
#else
  hipLaunchKernelGGL(hip_copy_double_complex_tmp1_tmp2_kernel, blocks, threadsPerBlock, 0, 0, tmp1_casted, tmp2_casted, nblk, nb);
#endif
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_double_complex_tmp1_tmp2_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_float_complex_tmp1_tmp2_kernel(hipFloatComplex *tmp1_dev, hipFloatComplex *tmp2_dev, const int nblk, const int nb){

  int i_index    = threadIdx.x +1;  // range 1..nb
  int j_index    = blockIdx.x +1;  // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp2_dev[1-1 + j_index-1 + (i_index-1)*nblk] = tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1];
  }
}


extern "C" void hip_copy_float_complex_tmp1_tmp2_FromC(float _Complex *tmp1_dev, float _Complex *tmp2_dev, int *nblk_in, int *nb_in, hipStream_t my_stream){
  int nblk = *nblk_in;
  int nb = *nb_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(nb,1,1);

  hipFloatComplex* tmp1_casted = (hipFloatComplex*) tmp1_dev;
  hipFloatComplex* tmp2_casted = (hipFloatComplex*) tmp2_dev;


#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_float_complex_tmp1_tmp2_kernel, blocks, threadsPerBlock, 0, my_stream, tmp1_casted, tmp2_casted, nblk, nb);
#else
  hipLaunchKernelGGL(hip_copy_float_complex_tmp1_tmp2_kernel, blocks, threadsPerBlock, 0, 0, tmp1_casted, tmp2_casted, nblk, nb);
#endif
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_float_complex_tmp1_tmp2_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_double_a_tmp1_kernel(double *a_dev, double *tmp1_dev, const int l_row1, const int l_col1, const int matrixRows, const int nb){

  int i_index    = threadIdx.x +1;  // range 1..nb
  int j_index    = blockIdx.x +1;  // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1] = a_dev[l_row1-1+j_index-1 + (l_col1-1+i_index-1)*matrixRows];
  }
}

extern "C" void hip_copy_double_a_tmp1_FromC(double *a_dev, double *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, hipStream_t my_stream){
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_double_a_tmp1_kernel, blocks, threadsPerBlock, 0, my_stream, a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb);
#else
  hipLaunchKernelGGL(hip_copy_double_a_tmp1_kernel, blocks, threadsPerBlock, 0, 0, a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb);
#endif
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_double_a_tmp1_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_float_a_tmp1_kernel(float *a_dev, float *tmp1_dev, const int l_row1, const int l_col1, const int matrixRows, const int nb){

  int i_index    = threadIdx.x +1;  // range 1..nb
  int j_index    = blockIdx.x +1;  // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1] = a_dev[l_row1-1+j_index-1 + (l_col1-1+i_index-1)*matrixRows];
  }

}

extern "C" void hip_copy_float_a_tmp1_FromC(float *a_dev, float *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, hipStream_t my_stream){
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_float_a_tmp1_kernel, blocks, threadsPerBlock, 0, my_stream, a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb);
#else
  hipLaunchKernelGGL(hip_copy_float_a_tmp1_kernel, blocks, threadsPerBlock, 0, 0, a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb);
#endif
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_float_a_tmp1_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_double_complex_a_tmp1_kernel(hipDoubleComplex *a_dev, hipDoubleComplex *tmp1_dev, const int l_row1, const int l_col1, const int matrixRows, const int nb){

  int i_index    = threadIdx.x +1;  // range 1..nb
  int j_index    = blockIdx.x +1;  // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1] = a_dev[l_row1-1+j_index-1 + (l_col1-1+i_index-1)*matrixRows];
  }
}

extern "C" void hip_copy_double_complex_a_tmp1_FromC(double _Complex *a_dev, double _Complex *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, hipStream_t my_stream){
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

  hipDoubleComplex* a_casted = (hipDoubleComplex*) a_dev;
  hipDoubleComplex* tmp1_casted = (hipDoubleComplex*) tmp1_dev;


  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_double_complex_a_tmp1_kernel, blocks, threadsPerBlock, 0, my_stream, a_casted, tmp1_casted, l_row1, l_col1, matrixRows, nb);
#else
  hipLaunchKernelGGL(hip_copy_double_complex_a_tmp1_kernel, blocks, threadsPerBlock, 0, 0, a_casted, tmp1_casted, l_row1, l_col1, matrixRows, nb);
#endif
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_double_complex_a_tmp1_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_float_complex_a_tmp1_kernel(hipFloatComplex *a_dev, hipFloatComplex *tmp1_dev, const int l_row1, const int l_col1, const int matrixRows, const int nb){

  int i_index    = threadIdx.x +1;  // range 1..nb
  int j_index    = blockIdx.x +1;  // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1] = a_dev[l_row1-1+j_index-1 + (l_col1-1+i_index-1)*matrixRows];
  }
}

extern "C" void hip_copy_float_complex_a_tmp1_FromC(float _Complex *a_dev, float _Complex *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, hipStream_t my_stream){
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
 
//#ifdef WITH_GPU_STREAMS
//  hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

  hipFloatComplex* a_casted = (hipFloatComplex*) a_dev;
  hipFloatComplex* tmp1_casted = (hipFloatComplex*) tmp1_dev;


  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(nb,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_float_complex_a_tmp1_kernel, blocks, threadsPerBlock, 0, my_stream, a_casted, tmp1_casted, l_row1, l_col1, matrixRows, nb);
#else
  hipLaunchKernelGGL(hip_copy_float_complex_a_tmp1_kernel, blocks, threadsPerBlock, 0, 0, a_casted, tmp1_casted, l_row1, l_col1, matrixRows, nb);
#endif
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_float_complex_a_tmp1_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

