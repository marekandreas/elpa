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
#include <hip/hip_complex.h>
#include "hip/hip_runtime.h"
#include <stdint.h>
#include "config-f90.h"

#ifdef NEW_KERNEL
#define BLK_X_DIM            32
#define BLK_Y_DIM            4
#define TRANS_COALESCED      1
#endif

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

__global__ void hip_check_device_info_kernel(int *info_dev){
  // if (*info_dev != 0){
  //   printf("Error in executing check_device_info_kerne: %d\n", *info_dev);
  // }
  assert(*info_dev == 0);
}


extern "C" void hip_check_device_info_FromC(int *info_dev, hipStream_t my_stream){

  dim3 blocks = dim3(1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_check_device_info_kernel, blocks, threadsPerBlock, 0, my_stream, info_dev);
#else
  hipLaunchKernelGGL(hip_check_device_info_kernel, blocks, threadsPerBlock, 0, 0,  info_dev);
#endif

  hipError_t hiperr = hipGetLastError();
  if (hiperr != hipSuccess){
    printf("Error in executing check_device_info_kernel: %s\n", hipGetErrorString(hiperr));
  }
}


__global__ void hip_accumulate_device_info_kernel(int *info_abs_dev, int *info_new_dev){
  *info_abs_dev += abs(*info_new_dev);
}

extern "C" void hip_accumulate_device_info_FromC(int *info_abs_dev, int *info_new_dev, hipStream_t my_stream){

  dim3 blocks = dim3(1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_accumulate_device_info_kernel, blocks, threadsPerBlock, 0, my_stream, info_abs_dev, info_new_dev);
#else
  hipLaunchKernelGGL(hip_accumulate_device_info_kernel, blocks, threadsPerBlock, 0, 0, info_abs_dev, info_new_dev);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing accumulate_device_info_kernel: %s\n",hipGetErrorString(cuerr));
  }
}


#ifdef NEW_KERNEL
template <typename T>
__global__ void hip_copy_a_tmatc_kernel(
        T const* __restrict__ a_dev,
        T* __restrict__ tmatc_dev,
        const int l_cols,
        const int matrixRows,
        const int l_colx,
        const int l_row1, const int nblk)
{
    __shared__ T tile[BLK_X_DIM][BLK_X_DIM + 1];

    int x = blockIdx.x * BLK_X_DIM + threadIdx.x;
    int y = blockIdx.y * BLK_X_DIM + threadIdx.y;
    int j;

    for (j = 0; j < BLK_X_DIM; j += BLK_Y_DIM) {
        tile[threadIdx.y + j][threadIdx.x] = a_dev[(l_colx - 1 + y + j) * matrixRows + x + l_row1 - 1];
    }

    __syncthreads();

    x = blockIdx.y * BLK_X_DIM + threadIdx.x;
    y = blockIdx.x * BLK_X_DIM + threadIdx.y;

    for (j = 0; j < BLK_X_DIM; j += BLK_Y_DIM) {
        tmatc_dev[(y + j) * l_cols + x + l_colx - 1] = tile[threadIdx.x][threadIdx.y + j];
    }
}
#endif /* NEW_KERNEL */

__global__ void hip_copy_double_a_tmatc_kernel(double *a_dev, double *tmatc_dev, const int l_cols, const int matrixRows, const int l_colx, const int l_row1, const int nblk){

  int ii_index    = threadIdx.x +1; // range 1..nblk
  int jj_index = blockIdx.x + 1; // range 1..l_cols-l_colx+1
  tmatc_dev[l_colx-1+jj_index-1+(ii_index-1)*l_cols] = a_dev[l_row1-1+ii_index-1 + (l_colx-1+jj_index-1)*matrixRows];
}

extern "C" void hip_copy_double_a_tmatc_FromC(double *a_dev, double *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, hipStream_t my_stream){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;
//#ifdef WITH_GPU_STREAMS
//  hipStream_t streamId = *((hipStream_t*)my_stream);
//#endif

#ifdef NEW_KERNEL
  if ((l_cols-l_colx+1) % BLK_X_DIM == 0 && nblk % BLK_X_DIM == 0 && TRANS_COALESCED) {
    dim3 blocks = dim3(nblk / BLK_X_DIM, (l_cols-l_colx+1) / BLK_X_DIM);
    dim3 threadsPerBlock = dim3(BLK_X_DIM, BLK_Y_DIM);

#ifdef WITH_GPU_STREAMS
    hipLaunchKernelGGL(hip_copy_a_tmatc_kernel<double>, blocks, threadsPerBlock, 0, my_stream, a_dev,
              tmatc_dev, l_cols, matrixRows, l_colx, l_row1, nblk);
#else
    hipLaunchKernelGGL(hip_copy_a_tmatc_kernel<double>, blocks, threadsPerBlock, 0, 0, a_dev,
              tmatc_dev, l_cols, matrixRows, l_colx, l_row1, nblk);
#endif
  }
  else {
#endif /* NEW_KERNEL */

  dim3 blocks = dim3(l_cols-l_colx+1,1,1);
  dim3 threadsPerBlock = dim3(nblk,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_double_a_tmatc_kernel, blocks, threadsPerBlock, 0, my_stream, a_dev, tmatc_dev, l_cols, matrixRows, l_colx, l_row1, nblk);
#else
  hipLaunchKernelGGL(hip_copy_double_a_tmatc_kernel, blocks, threadsPerBlock, 0, 0, a_dev, tmatc_dev, l_cols, matrixRows, l_colx, l_row1, nblk);
#endif
#ifdef NEW_KERNEL
  }
#endif /* NEW_KERNEL */

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing copy_double_a_tmatc_kernel: %s\n",hipGetErrorString(cuerr));
    printf("blocks=%d, threadsPerBlock=%d \n", l_cols-l_colx+1, nblk);
  }
}

__global__ void hip_copy_float_a_tmatc_kernel(float *a_dev, float *tmatc_dev, const int l_cols, const int matrixRows, const int l_colx, const int l_row1, const int nblk){

  int ii_index    = threadIdx.x +1; // range 1..nblk
  int jj_index = blockIdx.x + 1; // range 1..l_cols-l_colx+1
  tmatc_dev[l_colx-1+jj_index-1+(ii_index-1)*l_cols] = a_dev[l_row1-1+ii_index-1 + (l_colx-1+jj_index-1)*matrixRows];
}

extern "C" void hip_copy_float_a_tmatc_FromC(float *a_dev, float *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, hipStream_t my_stream){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;

//#ifdef WITH_GPU_STREAMS
//  hipStream_t streamId = *((hipStream_t*)my_stream);
//#endif

#ifdef NEW_KERNEL
  if ((l_cols-l_colx+1) % BLK_X_DIM == 0 && nblk % BLK_X_DIM == 0 && TRANS_COALESCED) {
      dim3 blocks = dim3(nblk / BLK_X_DIM, (l_cols-l_colx+1) / BLK_X_DIM);
      dim3 threadsPerBlock = dim3(BLK_X_DIM, BLK_Y_DIM);

#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_a_tmatc_kernel<float>, blocks, threadsPerBlock, 0, streamID, a_dev,
              tmatc_dev, l_cols, matrixRows, l_colx, l_row1, nblk);
#else
      hipLaunchKernelGGL(hip_copy_a_tmatc_kernel<float>, blocks, threadsPerBlock, 0, 0, a_dev,
              tmatc_dev, l_cols, matrixRows, l_colx, l_row1, nblk);
#endif
  }
  else {
#endif /* NEW_KERNEL */

  dim3 blocks = dim3(l_cols-l_colx+1,1,1);
  dim3 threadsPerBlock = dim3(nblk,1,1);

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_float_a_tmatc_kernel, blocks, threadsPerBlock, 0, my_stream, a_dev, tmatc_dev, l_cols, matrixRows, l_colx, l_row1, nblk);
#else
  hipLaunchKernelGGL(hip_copy_float_a_tmatc_kernel, blocks, threadsPerBlock, 0, 0, a_dev, tmatc_dev, l_cols, matrixRows, l_colx, l_row1, nblk);
#endif
#ifdef NEW_KERNEL
  }
#endif
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing copy_float_a_tmatc_kernel: %s\n",hipGetErrorString(cuerr));
    printf("blocks=%d, threadsPerBlock=%d \n", l_cols-l_colx+1, nblk);
  }
}

__global__ void hip_copy_double_complex_a_tmatc_kernel(hipDoubleComplex *a_dev, hipDoubleComplex *tmatc_dev, const int l_cols, const int matrixRows, const int l_colx, const int l_row1){

  int ii_index    = threadIdx.x +1; // range 1..nblk
  int jj_index = blockIdx.x + 1; // range 1..l_cols-l_colx+1
  tmatc_dev[l_colx-1+jj_index-1+(ii_index-1)*l_cols] = hipConj(a_dev[l_row1-1+ii_index-1 + (l_colx-1+jj_index-1)*matrixRows]);
}

extern "C" void hip_copy_double_complex_a_tmatc_FromC(double _Complex *a_dev, double _Complex *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, hipStream_t my_stream){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;

//#ifdef WITH_GPU_STREAMS
//  hipStream_t streamId = *((hipStream_t*)my_stream);
//#endif

#ifdef NEW_KERNEL
  hipDoubleComplex* a_casted = (hipDoubleComplex*) a_dev;
  hipDoubleComplex* tmatc_casted = (hipDoubleComplex*) tmatc_dev;

  if ((l_cols-l_colx+1) % BLK_X_DIM == 0 && nblk % BLK_X_DIM == 0 && TRANS_COALESCED) {
      dim3 blocks = dim3(nblk / BLK_X_DIM, (l_cols-l_colx+1) / BLK_X_DIM);
      dim3 threadsPerBlock = dim3(BLK_X_DIM, BLK_Y_DIM);

#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_a_tmatc_kernel<hipDoubleComplex>, blocks, threadsPerBlock, 0, steamID, a_casted,
              tmatc_casted, l_cols, matrixRows, l_colx, l_row1, nblk);
#else
      hipLaunchKernelGGL(hip_copy_a_tmatc_kernel<hipDoubleComplex>, blocks, threadsPerBlock, 0, 0, a_casted,
              tmatc_casted, l_cols, matrixRows, l_colx, l_row1, nblk);
#endif
  }
  else {
#endif /* NEW_KERNEL */

  dim3 blocks = dim3(l_cols-l_colx+1,1,1);
  dim3 threadsPerBlock = dim3(nblk,1,1);

  hipDoubleComplex* a_casted = (hipDoubleComplex*) a_dev;
  hipDoubleComplex* tmatc_casted = (hipDoubleComplex*) tmatc_dev;

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_double_complex_a_tmatc_kernel, blocks, threadsPerBlock, 0, my_stream, a_casted, tmatc_casted, l_cols, matrixRows, l_colx, l_row1);
#else
  hipLaunchKernelGGL(hip_copy_double_complex_a_tmatc_kernel, blocks, threadsPerBlock, 0, 0, a_casted, tmatc_casted, l_cols, matrixRows, l_colx, l_row1);
#endif

#ifdef NEW_KERNEL
  }
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing copy_double_complex_a_tmatc_kernel: %s\n",hipGetErrorString(cuerr));
    printf("blocks=%d, threadsPerBlock=%d \n", l_cols-l_colx+1, nblk);
  }
}

__global__ void hip_copy_float_complex_a_tmatc_kernel(hipFloatComplex *a_dev, hipFloatComplex *tmatc_dev, const int l_cols, const int matrixRows, const int l_colx, const int l_row1){

  int ii_index    = threadIdx.x +1; // range 1..nblk
  int jj_index = blockIdx.x + 1; // range 1..l_cols-l_colx+1
  tmatc_dev[l_colx-1+jj_index-1+(ii_index-1)*l_cols] = hipConjf(a_dev[l_row1-1+ii_index-1 + (l_colx-1+jj_index-1)*matrixRows]);
}

extern "C" void hip_copy_float_complex_a_tmatc_FromC(float _Complex *a_dev, float _Complex *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, hipStream_t my_stream){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;

//#ifdef WITH_GPU_STREAMS
//  hipStream_t streamId = *((hipStream_t*)my_stream);
//#endif

#ifdef NEW_KERNEL
  hipFloatComplex* a_casted = (hipFloatComplex*) a_dev;
  hipFloatComplex* tmatc_casted = (hipFloatComplex*) tmatc_dev;

  if ((l_cols-l_colx+1) % BLK_X_DIM == 0 && nblk % BLK_X_DIM == 0 && TRANS_COALESCED) {
      dim3 blocks = dim3(nblk / BLK_X_DIM, (l_cols-l_colx+1) / BLK_X_DIM);
      dim3 threadsPerBlock = dim3(BLK_X_DIM, BLK_Y_DIM);

#ifdef WITH_GPU_STREAMS
      hipLaunchKernelGGL(hip_copy_a_tmatc_kernel<hipComplex>, blocks, threadsPerBlock, 0, my_stream, a_casted,
              tmatc_casted, l_cols, matrixRows, l_colx, l_row1, nblk);
#else
      hipLaunchKernelGGL(hip_copy_a_tmatc_kernel<hipComplex>, blocks, threadsPerBlock, 0, 0, a_casted,
              tmatc_casted, l_cols, matrixRows, l_colx, l_row1, nblk);
#endif
  }
  else {
#endif /* NEW_KERNEL */

  dim3 blocks = dim3(l_cols-l_colx+1,1,1);
  dim3 threadsPerBlock = dim3(nblk,1,1);

  hipFloatComplex* a_casted = (hipFloatComplex*) a_dev;
  hipFloatComplex* tmatc_casted = (hipFloatComplex*) tmatc_dev;

#ifdef WITH_GPU_STREAMS
  hipLaunchKernelGGL(hip_copy_float_complex_a_tmatc_kernel, blocks, threadsPerBlock, 0, my_stream, a_casted, tmatc_casted, l_cols, matrixRows, l_colx, l_row1);
#else
  hipLaunchKernelGGL(hip_copy_float_complex_a_tmatc_kernel, blocks, threadsPerBlock, 0, 0, a_casted, tmatc_casted, l_cols, matrixRows, l_colx, l_row1);
#endif

#ifdef NEW_KERNEL
  }
#endif
  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing copy_float_complex_a_tmatc_kernel: %s\n",hipGetErrorString(cuerr));
    printf("blocks=%d, threadsPerBlock=%d \n", l_cols-l_colx+1, nblk);
  }
}
