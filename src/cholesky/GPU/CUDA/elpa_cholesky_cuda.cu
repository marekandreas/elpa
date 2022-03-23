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
#include "config-f90.h"

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

__global__ void cuda_copy_double_a_tmatc_kernel(double *a_dev, double *tmatc_dev, const int l_cols, const int matrixRows, const int l_colx, const int l_row1, const int nblk){

  int ii_index    = threadIdx.x +1; // range 1..nblk
  int jj_index = blockIdx.x + 1; // range 1..l_cols-l_colx+1
  tmatc_dev[l_colx-1+jj_index-1+(ii_index-1)*l_cols] = a_dev[l_row1-1+ii_index-1 + (l_colx-1+jj_index-1)*matrixRows];
}

extern "C" void cuda_copy_double_a_tmatc_FromC(double *a_dev, double *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;

  dim3 blocks = dim3(l_cols-l_colx+1,1,1);
  dim3 threadsPerBlock = dim3(nblk,1,1);

  cuda_copy_double_a_tmatc_kernel<<<blocks,threadsPerBlock>>>(a_dev, tmatc_dev, l_cols, matrixRows, l_colx, l_row1, nblk);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing copy_double_a_tmatc_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_float_a_tmatc_kernel(float *a_dev, float *tmatc_dev, const int l_cols, const int matrixRows, const int l_colx, const int l_row1, const int nblk){

  int ii_index    = threadIdx.x +1; // range 1..nblk
  int jj_index = blockIdx.x + 1; // range 1..l_cols-l_colx+1
  tmatc_dev[l_colx-1+jj_index-1+(ii_index-1)*l_cols] = a_dev[l_row1-1+ii_index-1 + (l_colx-1+jj_index-1)*matrixRows];
}

extern "C" void cuda_copy_float_a_tmatc_FromC(float *a_dev, float *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;

  dim3 blocks = dim3(l_cols-l_colx+1,1,1);
  dim3 threadsPerBlock = dim3(nblk,1,1);

  cuda_copy_float_a_tmatc_kernel<<<blocks,threadsPerBlock>>>(a_dev, tmatc_dev, l_cols, matrixRows, l_colx, l_row1, nblk);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing copy_float_a_tmatc_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_double_complex_a_tmatc_kernel(cuDoubleComplex *a_dev, cuDoubleComplex *tmatc_dev, const int l_cols, const int matrixRows, const int l_colx, const int l_row1){

  int ii_index    = threadIdx.x +1; // range 1..nblk
  int jj_index = blockIdx.x + 1; // range 1..l_cols-l_colx+1
  tmatc_dev[l_colx-1+jj_index-1+(ii_index-1)*l_cols] = cuConj(a_dev[l_row1-1+ii_index-1 + (l_colx-1+jj_index-1)*matrixRows]);
}

extern "C" void cuda_copy_double_complex_a_tmatc_FromC(double _Complex *a_dev, double _Complex *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;

  dim3 blocks = dim3(l_cols-l_colx+1,1,1);
  dim3 threadsPerBlock = dim3(nblk,1,1);

  cuDoubleComplex* a_casted = (cuDoubleComplex*) a_dev;
  cuDoubleComplex* tmatc_casted = (cuDoubleComplex*) tmatc_dev;

  cuda_copy_double_complex_a_tmatc_kernel<<<blocks,threadsPerBlock>>>(a_casted, tmatc_casted, l_cols, matrixRows, l_colx, l_row1);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing copy_double_complex_a_tmatc_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_float_complex_a_tmatc_kernel(cuFloatComplex *a_dev, cuFloatComplex *tmatc_dev, const int l_cols, const int matrixRows, const int l_colx, const int l_row1){

  int ii_index    = threadIdx.x +1; // range 1..nblk
  int jj_index = blockIdx.x + 1; // range 1..l_cols-l_colx+1
  tmatc_dev[l_colx-1+jj_index-1+(ii_index-1)*l_cols] = cuConjf(a_dev[l_row1-1+ii_index-1 + (l_colx-1+jj_index-1)*matrixRows]);
}

extern "C" void cuda_copy_float_complex_a_tmatc_FromC(float _Complex *a_dev, float _Complex *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;

  dim3 blocks = dim3(l_cols-l_colx+1,1,1);
  dim3 threadsPerBlock = dim3(nblk,1,1);

  cuFloatComplex* a_casted = (cuFloatComplex*) a_dev;
  cuFloatComplex* tmatc_casted = (cuFloatComplex*) tmatc_dev;

  cuda_copy_float_complex_a_tmatc_kernel<<<blocks,threadsPerBlock>>>(a_casted, tmatc_casted, l_cols, matrixRows, l_colx, l_row1);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing copy_float_complex_a_tmatc_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}
