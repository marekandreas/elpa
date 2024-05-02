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
#include <cuComplex.h>
#include <stdint.h>
#include <assert.h>
#include "config-f90.h"

#include "../../../GPU/common_device_functions.h"

#define MAX_THREADS_PER_BLOCK 1024

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

//________________________________________________________________

__global__ void cuda_check_device_info_kernel(int *info_dev){
  // if (*info_dev != 0){
  //   printf("Error in executing check_device_info_kerne: %d\n", *info_dev);
  // }
  assert(*info_dev == 0);
}

extern "C" void cuda_check_device_info_FromC(int *info_dev, cudaStream_t my_stream){

  dim3 blocks = dim3(1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_check_device_info_kernel<<<blocks,threadsPerBlock, 0, my_stream>>>(info_dev);
#else
  cuda_check_device_info_kernel<<<blocks,threadsPerBlock>>>(info_dev);
#endif

  cudaDeviceSynchronize();
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing check_device_info_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_accumulate_device_info_kernel(int *info_abs_dev, int *info_new_dev){
  *info_abs_dev += abs(*info_new_dev);
}

extern "C" void cuda_accumulate_device_info_FromC(int *info_abs_dev, int *info_new_dev, cudaStream_t my_stream){

  dim3 blocks = dim3(1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_accumulate_device_info_kernel<<<blocks,threadsPerBlock, 0, my_stream>>>(info_abs_dev, info_new_dev);
#else
  cuda_accumulate_device_info_kernel<<<blocks,threadsPerBlock>>>(info_abs_dev, info_new_dev);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing accumulate_device_info_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

//________________________________________________________________

template <typename T>
__global__ void cuda_copy_a_tmatc_kernel(T *a_dev, T *tmatc_dev, const int l_cols, const int matrixRows, const int l_colx, const int l_row1){

  int ii_index    = threadIdx.x +1; // range 1..nblk
  int jj_index = blockIdx.x + 1; // range 1..l_cols-l_colx+1
  tmatc_dev[l_colx-1+jj_index-1+(ii_index-1)*l_cols] = elpaDeviceComplexConjugate(a_dev[l_row1-1+ii_index-1 + (l_colx-1+jj_index-1)*matrixRows]);
}

template <typename T>
void cuda_copy_a_tmatc_FromC(T *a_dev, T *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, cudaStream_t my_stream){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;

  dim3 blocks = dim3(l_cols-l_colx+1,1,1);
  dim3 threadsPerBlock = dim3(nblk,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_copy_a_tmatc_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(a_dev, tmatc_dev, l_cols, matrixRows, l_colx, l_row1);
#else
  cuda_copy_a_tmatc_kernel<<<blocks,threadsPerBlock>>>(a_dev, tmatc_dev, l_cols, matrixRows, l_colx, l_row1);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing copy_double_complex_a_tmatc_kernel: %s\n",cudaGetErrorString(cuerr));
    printf("blocks=%d, threadsPerBlock=%d \n", l_cols-l_colx+1, nblk);
  }
}

extern "C" void cuda_copy_double_a_tmatc_FromC(double *a_dev, double *tmatc_dev, int *nblk_in, int *matrixRows_in, 
                                               int *l_cols_in, int *l_colx_in, int *l_row1_in, cudaStream_t my_stream){
  cuda_copy_a_tmatc_FromC(a_dev, tmatc_dev, nblk_in, matrixRows_in, l_cols_in, l_colx_in, l_row1_in, my_stream);
}

extern "C" void cuda_copy_float_a_tmatc_FromC(float *a_dev, float *tmatc_dev, int *nblk_in, int *matrixRows_in, 
                                              int *l_cols_in, int *l_colx_in, int *l_row1_in, cudaStream_t my_stream){
  cuda_copy_a_tmatc_FromC(a_dev, tmatc_dev, nblk_in, matrixRows_in, l_cols_in, l_colx_in, l_row1_in, my_stream);
}

extern "C" void cuda_copy_double_complex_a_tmatc_FromC(cuDoubleComplex *a_dev, cuDoubleComplex *tmatc_dev, int *nblk_in, int *matrixRows_in, 
                                                  int *l_cols_in, int *l_colx_in, int *l_row1_in, cudaStream_t my_stream){
  cuda_copy_a_tmatc_FromC(a_dev, tmatc_dev, nblk_in, matrixRows_in, l_cols_in, l_colx_in, l_row1_in, my_stream);
}

extern "C" void cuda_copy_float_complex_a_tmatc_FromC(cuFloatComplex *a_dev, cuFloatComplex *tmatc_dev, int *nblk_in, int *matrixRows_in, 
                                                 int *l_cols_in, int *l_colx_in, int *l_row1_in, cudaStream_t my_stream){
  cuda_copy_a_tmatc_FromC(a_dev, tmatc_dev, nblk_in, matrixRows_in, l_cols_in, l_colx_in, l_row1_in, my_stream);
}

//________________________________________________________________


template <typename T>
__global__ void cuda_set_a_lower_to_zero_kernel (T *a_dev, int na, int matrixRows, int my_pcol, int np_cols, int my_prow, int np_rows, int nblk) {

  int J_gl_0 = blockIdx.x; // 0..nblk-1
  int di_loc_0 = threadIdx.x; // 0..MAX_THREADS_PER_BLOCK-1

  T Zero = elpaDeviceNumber<T>(0.0);

  for (int J_gl = J_gl_0; J_gl < na; J_gl += gridDim.x)
    {
    if (my_pcol == pcol(J_gl, nblk, np_cols))
      {
      // Calculate local column and row indices of the first element below the diagonal (that has to be set to zero)
      int l_col1 = local_index(J_gl  , my_pcol, np_cols, nblk);
      int l_row1 = local_index(J_gl+1, my_prow, np_rows, nblk); // I_gl = J_gl + 1

      // Calculate the offset and number of elements to zero out
      //int offset = l_row1 + matrixRows*l_col1;
      //int num = (matrixRows - l_row1);

      // Set to zero in the GPU memory
      for (int di_loc=di_loc_0; di_loc < (matrixRows-l_row1); di_loc += blockDim.x) a_dev[(l_row1+di_loc) + matrixRows*l_col1] = Zero;
      }
    }
}

template <typename T>
void cuda_set_a_lower_to_zero(T *a_dev, int *na_in, int *matrixRows_in, int *my_pcol_in, int *np_cols_in, int *my_prow_in, int *np_rows_in, int *nblk_in, int *wantDebug_in, cudaStream_t my_stream){
  int na = *na_in;
  int matrixRows = *matrixRows_in;
  int my_pcol = *my_pcol_in;
  int np_cols = *np_cols_in;
  int my_prow = *my_prow_in;
  int np_rows = *np_rows_in;
  int nblk = *nblk_in;
  int wantDebug = *wantDebug_in;

  dim3 blocks = dim3(nblk,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_set_a_lower_to_zero_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(a_dev, na, matrixRows, my_pcol, np_cols, my_prow, np_rows, nblk);
#else
  cuda_set_a_lower_to_zero_kernel<<<blocks,threadsPerBlock>>>(a_dev, na, matrixRows, my_pcol, np_cols, my_prow, np_rows, nblk);
#endif

  if (wantDebug)
    {
    cudaDeviceSynchronize();
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess){
      printf("Error in executing set_a_lower_to_zero_kernel: %s\n",cudaGetErrorString(cuerr));
    }
  }
}

extern "C" void cuda_set_a_lower_to_zero_FromC(char dataType, intptr_t a_dev, int *na_in, int *matrixRows_in, 
                                                      int *my_pcol_in, int *np_cols_in, int *my_prow_in, int *np_rows_in, 
                                                      int *nblk_in, int *wantDebug_in, cudaStream_t my_stream){

  if (dataType=='D') cuda_set_a_lower_to_zero<double>((double *) a_dev, na_in, matrixRows_in, my_pcol_in, np_cols_in, my_prow_in, np_rows_in, nblk_in, wantDebug_in, my_stream);
  if (dataType=='S') cuda_set_a_lower_to_zero<float> ((float *) a_dev, na_in, matrixRows_in, my_pcol_in, np_cols_in, my_prow_in, np_rows_in, nblk_in, wantDebug_in, my_stream);
  if (dataType=='Z') cuda_set_a_lower_to_zero<cuDoubleComplex>((cuDoubleComplex *) a_dev, na_in, matrixRows_in, my_pcol_in, np_cols_in, my_prow_in, np_rows_in, nblk_in, wantDebug_in, my_stream);
  if (dataType=='C') cuda_set_a_lower_to_zero<cuFloatComplex> ((cuFloatComplex *) a_dev, na_in, matrixRows_in, my_pcol_in, np_cols_in, my_prow_in, np_rows_in, nblk_in, wantDebug_in, my_stream);
}