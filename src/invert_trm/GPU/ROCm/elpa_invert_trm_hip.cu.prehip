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

__global__ void cuda_copy_double_a_tmat2_kernel(double *a_dev, double *tmat2_dev, const int nblk, const int matrixRows, const int l_colx, const int l_row1){

  int nb_index    = threadIdx.x +1; // range 1..nb
  int l_col_index = blockIdx.x + 1; // range 1..l_colx-l_cols-1

  tmat2_dev[nb_index-1 + (l_colx-1 + l_col_index -1) * nblk] = a_dev[l_row1-1 + nb_index-1 + (l_colx-1 + l_col_index -1)  * matrixRows];

}

extern "C" void cuda_copy_double_a_tmat2_FromC(double *a_dev, double *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;
  int nb     = *nb_in;

  dim3 blocks = dim3(l_cols-l_colx+1,1,1);
  dim3 threadsPerBlock = dim3(nb,1,1);

  cuda_copy_double_a_tmat2_kernel<<<blocks,threadsPerBlock>>>(a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_double_a_tmat2_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_float_a_tmat2_kernel(float *a_dev, float *tmat2_dev, const int nblk, const int matrixRows, const int l_colx, const int l_row1){

  int nb_index    = threadIdx.x +1; // range 1..nb
  int l_col_index = blockIdx.x + 1; // range 1..l_colx-l_cols-1

  tmat2_dev[nb_index-1 + (l_colx-1 + l_col_index -1) * nblk] = a_dev[l_row1-1 + nb_index-1 + (l_colx-1 + l_col_index -1)  * matrixRows];
}

extern "C" void cuda_copy_float_a_tmat2_FromC(float *a_dev, float *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;
  int nb     = *nb_in;

  dim3 blocks = dim3(l_cols-l_colx+1,1,1);
  dim3 threadsPerBlock = dim3(nb,1,1);

  cuda_copy_float_a_tmat2_kernel<<<blocks,threadsPerBlock>>>(a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_float_a_tmat2_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_double_complex_a_tmat2_kernel(cuDoubleComplex *a_dev, cuDoubleComplex *tmat2_dev, const int nblk, const int matrixRows, const int l_colx, const int l_row1){

  int nb_index    = threadIdx.x +1; // range 1..nb
  int l_col_index = blockIdx.x + 1; // range 1..l_colx-l_cols-1

  tmat2_dev[nb_index-1 + (l_colx-1 + l_col_index -1) * nblk] = a_dev[l_row1-1 + nb_index-1 + (l_colx-1 + l_col_index -1)  * matrixRows];

}

extern "C" void cuda_copy_double_complex_a_tmat2_FromC(double _Complex *a_dev, double _Complex *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;
  int nb     = *nb_in;

  cuDoubleComplex* a_casted = (cuDoubleComplex*) a_dev;
  cuDoubleComplex* tmat2_casted = (cuDoubleComplex*) tmat2_dev;

  dim3 blocks = dim3(l_cols-l_colx+1,1,1);
  dim3 threadsPerBlock = dim3(nb,1,1);

  cuda_copy_double_complex_a_tmat2_kernel<<<blocks,threadsPerBlock>>>(a_casted, tmat2_casted, nblk, matrixRows, l_colx, l_row1);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_double_complex_a_tmat2_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_float_complex_a_tmat2_kernel(cuFloatComplex *a_dev, cuFloatComplex *tmat2_dev, const int nblk, const int matrixRows, const int l_colx, const int l_row1){

  int nb_index    = threadIdx.x +1; // range 1..nb
  int l_col_index = blockIdx.x + 1; // range 1..l_colx-l_cols-1

  tmat2_dev[nb_index-1 + (l_colx-1 + l_col_index -1) * nblk] = a_dev[l_row1-1 + nb_index-1 + (l_colx-1 + l_col_index -1)  * matrixRows];

}

extern "C" void cuda_copy_float_complex_a_tmat2_FromC(float _Complex *a_dev, float _Complex *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;
  int nb     = *nb_in;

  cuFloatComplex* a_casted = (cuFloatComplex*) a_dev;
  cuFloatComplex* tmat2_casted = (cuFloatComplex*) tmat2_dev;

  dim3 blocks = dim3(l_cols-l_colx+1,1,1);
  dim3 threadsPerBlock = dim3(nb,1,1);

  cuda_copy_float_complex_a_tmat2_kernel<<<blocks,threadsPerBlock>>>(a_casted, tmat2_casted, nblk, matrixRows, l_colx, l_row1);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_float_complex_a_tmat2_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_double_tmp2_tmat2_kernel(double *tmp2_dev, double *tmat2_dev, const int nblk, const int l_col1){

  int nb_index    = threadIdx.x +1;  // range 1..nb
  int l_col_index = blockIdx.x + 1;  // range 1..nb

  tmat2_dev[nb_index-1 + (l_col1-1 + l_col_index -1)*nblk] = tmp2_dev[nb_index-1 + (1 -1 + l_col_index-1)  * nblk];

}

extern "C" void cuda_copy_double_tmp2_tmat2_FromC(double *tmp2_dev, double *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in){
  int nblk   = *nblk_in;   
  int l_col1 = *l_col1_in;
  int nb     = *nb_in;

  dim3 blocks = dim3(nb,1,1);
  dim3 threadsPerBlock = dim3(nb,1,1);

  cuda_copy_double_tmp2_tmat2_kernel<<<blocks,threadsPerBlock>>>(tmp2_dev, tmat2_dev, nblk, l_col1);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_double_tmp2_tmat2_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}


__global__ void cuda_copy_float_tmp2_tmat2_kernel(float *tmp2_dev, float *tmat2_dev, const int nblk, const int l_col1){

  int nb_index    = threadIdx.x +1;  // range 1..nb
  int l_col_index = blockIdx.x + 1;  // range 1..nb

  tmat2_dev[nb_index-1 + (l_col1-1 + l_col_index -1)*nblk] = tmp2_dev[nb_index-1 + (1 -1 + l_col_index-1)  * nblk];

}

extern "C" void cuda_copy_float_tmp2_tmat2_FromC(float *tmp2_dev, float *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in){
  int nblk   = *nblk_in;   
  int l_col1 = *l_col1_in;
  int nb     = *nb_in;

  dim3 blocks = dim3(nb,1,1);
  dim3 threadsPerBlock = dim3(nb,1,1);

  cuda_copy_float_tmp2_tmat2_kernel<<<blocks,threadsPerBlock>>>(tmp2_dev, tmat2_dev, nblk, l_col1);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_float_tmp2_tmat2_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_double_complex_tmp2_tmat2_kernel(cuDoubleComplex *tmp2_dev, cuDoubleComplex *tmat2_dev, const int nblk, const int l_col1){

  int nb_index    = threadIdx.x +1;  // range 1..nb
  int l_col_index = blockIdx.x + 1;  // range 1..nb

  tmat2_dev[nb_index-1 + (l_col1-1 + l_col_index -1)*nblk] = tmp2_dev[nb_index-1 + (1 -1 + l_col_index-1)  * nblk];

}

extern "C" void cuda_copy_double_complex_tmp2_tmat2_FromC(double _Complex *tmp2_dev, double _Complex *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in){
  int nblk   = *nblk_in;   
  int l_col1 = *l_col1_in;
  int nb     = *nb_in;

  cuDoubleComplex* tmp2_casted = (cuDoubleComplex*) tmp2_dev;
  cuDoubleComplex* tmat2_casted = (cuDoubleComplex*) tmat2_dev;


  dim3 blocks = dim3(nb,1,1);
  dim3 threadsPerBlock = dim3(nb,1,1);

  cuda_copy_double_complex_tmp2_tmat2_kernel<<<blocks,threadsPerBlock>>>(tmp2_casted, tmat2_casted, nblk, l_col1);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_double_complex_tmp2_tmat2_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_float_complex_tmp2_tmat2_kernel(cuFloatComplex *tmp2_dev, cuFloatComplex *tmat2_dev, const int nblk, const int l_col1){

  int nb_index    = threadIdx.x +1;  // range 1..nb
  int l_col_index = blockIdx.x + 1;  // range 1..nb

  tmat2_dev[nb_index-1 + (l_col1-1 + l_col_index -1)*nblk] = tmp2_dev[nb_index-1 + (1 -1 + l_col_index-1)  * nblk];

}

extern "C" void cuda_copy_float_complex_tmp2_tmat2_FromC(float _Complex *tmp2_dev, float _Complex *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in){
  int nblk   = *nblk_in;   
  int l_col1 = *l_col1_in;
  int nb     = *nb_in;

  cuFloatComplex* tmp2_casted = (cuFloatComplex*) tmp2_dev;
  cuFloatComplex* tmat2_casted = (cuFloatComplex*) tmat2_dev;


  dim3 blocks = dim3(nb,1,1);
  dim3 threadsPerBlock = dim3(nb,1,1);

  cuda_copy_float_complex_tmp2_tmat2_kernel<<<blocks,threadsPerBlock>>>(tmp2_casted, tmat2_casted, nblk, l_col1);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_float_complex_tmp2_tmat2_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_double_a_tmat1_kernel(double *a_dev, double *tmat1_dev, const int l_rows, const int matrixRows, const int l_col1, const int nb, const int l_row1){

  int nb_index    = threadIdx.x +1;  // range 1..nb
  int l_row1_index = blockIdx.x + 1; // we need l_row1-1 blocks

  tmat1_dev[l_row1_index-1 + (nb_index-1)*l_rows] = a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1 ) * matrixRows];
  a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1)*matrixRows] = 0;

}

extern "C" void cuda_copy_double_a_tmat1_FromC(double *a_dev, double *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in){
  int l_rows = *l_rows_in;   
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;

  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(l_row1-1,1,1);

  cuda_copy_double_a_tmat1_kernel<<<blocks,threadsPerBlock>>>(a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_double_a_tmat1_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_float_a_tmat1_kernel(float *a_dev, float *tmat1_dev, const int l_rows, const int matrixRows, const int l_col1, const int nb, const int l_row1){

  int nb_index    = threadIdx.x +1;  // range 1..nb
  int l_row1_index = blockIdx.x + 1; // we need l_row1-1 blocks

  tmat1_dev[l_row1_index-1 + (nb_index-1)*l_rows] = a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1 ) * matrixRows];
  a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1)*matrixRows] = 0;
}

extern "C" void cuda_copy_float_a_tmat1_FromC(float *a_dev, float *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in){
  int l_rows = *l_rows_in;   
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;

  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(l_row1-1,1,1);
  //dim3 threadsPerBlock = dim3(1, 1, 1);
  //dim3 blocks = dim3(1,1,1);

  cuda_copy_float_a_tmat1_kernel<<<blocks,threadsPerBlock>>>(a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_float_a_tmat1_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_double_complex_a_tmat1_kernel(cuDoubleComplex *a_dev, cuDoubleComplex *tmat1_dev, const int l_rows, const int matrixRows, const int l_col1, const int nb, const int l_row1, cuDoubleComplex *zero_dev){

  int nb_index    = threadIdx.x +1;  // range 1..nb
  int l_row1_index = blockIdx.x + 1; // we need l_row1-1 blocks

  tmat1_dev[l_row1_index-1 + (nb_index-1)*l_rows] = a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1 ) * matrixRows];
  a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1)*matrixRows] = zero_dev[0];
}

extern "C" void cuda_copy_double_complex_a_tmat1_FromC(double _Complex *a_dev, double _Complex *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, double _Complex *ZERO){
  int l_rows = *l_rows_in;   
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;


  cuDoubleComplex* a_casted = (cuDoubleComplex*) a_dev;
  cuDoubleComplex* tmat1_casted = (cuDoubleComplex*) tmat1_dev;
  cuDoubleComplex* zero_casted = (cuDoubleComplex*)ZERO;

  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(l_row1-1,1,1);

  cuda_copy_double_complex_a_tmat1_kernel<<<blocks,threadsPerBlock>>>(a_casted, tmat1_casted, l_rows, matrixRows, l_col1, nb, l_row1, zero_casted);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_double_complex_a_tmat1_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}


__global__ void cuda_copy_float_complex_a_tmat1_kernel(cuFloatComplex *a_dev, cuFloatComplex *tmat1_dev, const int l_rows, const int matrixRows, const int l_col1, const int nb, const int l_row1, cuFloatComplex *zero_dev){

  int nb_index    = threadIdx.x +1;  // range 1..nb
  int l_row1_index = blockIdx.x + 1; // we need l_row1-1 blocks

  tmat1_dev[l_row1_index-1 + (nb_index-1)*l_rows] = a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1 ) * matrixRows];
  a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1)*matrixRows] = zero_dev[0];
}

extern "C" void cuda_copy_float_complex_a_tmat1_FromC(float _Complex *a_dev, float _Complex *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, float _Complex *ZERO){
  int l_rows = *l_rows_in;   
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;

  cuFloatComplex* a_casted = (cuFloatComplex*) a_dev;
  cuFloatComplex* tmat1_casted = (cuFloatComplex*) tmat1_dev;
  cuFloatComplex* zero_casted = (cuFloatComplex*)ZERO;

  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(l_row1-1,1,1);

  cuda_copy_float_complex_a_tmat1_kernel<<<blocks,threadsPerBlock>>>(a_casted, tmat1_casted, l_rows, matrixRows, l_col1, nb, l_row1, zero_casted);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_float_complex_a_tmat1_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_double_tmp1_tmp2_kernel(double *tmp1_dev, double *tmp2_dev, const int nblk, const int nb){

  int i_index    = threadIdx.x +1;  // range 1..nb
  int j_index    = blockIdx.x +1;  // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp2_dev[1-1 + j_index-1 + (i_index-1)*nblk] = tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1];
  }
}


extern "C" void cuda_copy_double_tmp1_tmp2_FromC(double *tmp1_dev, double *tmp2_dev, int *nblk_in, int *nb_in){
  int nblk = *nblk_in;
  int nb = *nb_in;

  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(nb,1,1);

  cuda_copy_double_tmp1_tmp2_kernel<<<blocks,threadsPerBlock>>>(tmp1_dev, tmp2_dev, nblk, nb);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_double_tmp1_tmp2_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_float_tmp1_tmp2_kernel(float *tmp1_dev, float *tmp2_dev, const int nblk, const int nb){

  int i_index    = threadIdx.x +1;  // range 1..nb
  int j_index    = blockIdx.x +1;  // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp2_dev[1-1 + j_index-1 + (i_index-1)*nblk] = tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1];
  }
}


extern "C" void cuda_copy_float_tmp1_tmp2_FromC(float *tmp1_dev, float *tmp2_dev, int *nblk_in, int *nb_in){
  int nblk = *nblk_in;
  int nb = *nb_in;

  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(nb,1,1);

  cuda_copy_float_tmp1_tmp2_kernel<<<blocks,threadsPerBlock>>>(tmp1_dev, tmp2_dev, nblk, nb);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_float_tmp1_tmp2_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_double_complex_tmp1_tmp2_kernel(cuDoubleComplex *tmp1_dev, cuDoubleComplex *tmp2_dev, const int nblk, const int nb){

  int i_index    = threadIdx.x +1;  // range 1..nb
  int j_index    = blockIdx.x +1;  // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp2_dev[1-1 + j_index-1 + (i_index-1)*nblk] = tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1];
  }
}


extern "C" void cuda_copy_double_complex_tmp1_tmp2_FromC(double _Complex *tmp1_dev, double _Complex *tmp2_dev, int *nblk_in, int *nb_in){
  int nblk = *nblk_in;
  int nb = *nb_in;

  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(nb,1,1);

  cuDoubleComplex* tmp1_casted = (cuDoubleComplex*) tmp1_dev;
  cuDoubleComplex* tmp2_casted = (cuDoubleComplex*) tmp2_dev;


  cuda_copy_double_complex_tmp1_tmp2_kernel<<<blocks,threadsPerBlock>>>(tmp1_casted, tmp2_casted, nblk, nb);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_double_complex_tmp1_tmp2_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_float_complex_tmp1_tmp2_kernel(cuFloatComplex *tmp1_dev, cuFloatComplex *tmp2_dev, const int nblk, const int nb){

  int i_index    = threadIdx.x +1;  // range 1..nb
  int j_index    = blockIdx.x +1;  // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp2_dev[1-1 + j_index-1 + (i_index-1)*nblk] = tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1];
  }
}


extern "C" void cuda_copy_float_complex_tmp1_tmp2_FromC(float _Complex *tmp1_dev, float _Complex *tmp2_dev, int *nblk_in, int *nb_in){
  int nblk = *nblk_in;
  int nb = *nb_in;

  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(nb,1,1);

  cuFloatComplex* tmp1_casted = (cuFloatComplex*) tmp1_dev;
  cuFloatComplex* tmp2_casted = (cuFloatComplex*) tmp2_dev;


  cuda_copy_float_complex_tmp1_tmp2_kernel<<<blocks,threadsPerBlock>>>(tmp1_casted, tmp2_casted, nblk, nb);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_float_complex_tmp1_tmp2_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_double_a_tmp1_kernel(double *a_dev, double *tmp1_dev, const int l_row1, const int l_col1, const int matrixRows, const int nb){

  int i_index    = threadIdx.x +1;  // range 1..nb
  int j_index    = blockIdx.x +1;  // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1] = a_dev[l_row1-1+j_index-1 + (l_col1-1+i_index-1)*matrixRows];
  }
}

extern "C" void cuda_copy_double_a_tmp1_FromC(double *a_dev, double *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in){
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;

  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(nb,1,1);

  cuda_copy_double_a_tmp1_kernel<<<blocks,threadsPerBlock>>>(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_double_a_tmp1_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_float_a_tmp1_kernel(float *a_dev, float *tmp1_dev, const int l_row1, const int l_col1, const int matrixRows, const int nb){

  int i_index    = threadIdx.x +1;  // range 1..nb
  int j_index    = blockIdx.x +1;  // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1] = a_dev[l_row1-1+j_index-1 + (l_col1-1+i_index-1)*matrixRows];
  }

}

extern "C" void cuda_copy_float_a_tmp1_FromC(float *a_dev, float *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in){
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;

  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(nb,1,1);

  cuda_copy_float_a_tmp1_kernel<<<blocks,threadsPerBlock>>>(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_float_a_tmp1_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_double_complex_a_tmp1_kernel(cuDoubleComplex *a_dev, cuDoubleComplex *tmp1_dev, const int l_row1, const int l_col1, const int matrixRows, const int nb){

  int i_index    = threadIdx.x +1;  // range 1..nb
  int j_index    = blockIdx.x +1;  // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1] = a_dev[l_row1-1+j_index-1 + (l_col1-1+i_index-1)*matrixRows];
  }
}

extern "C" void cuda_copy_double_complex_a_tmp1_FromC(double _Complex *a_dev, double _Complex *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in){
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;

  cuDoubleComplex* a_casted = (cuDoubleComplex*) a_dev;
  cuDoubleComplex* tmp1_casted = (cuDoubleComplex*) tmp1_dev;


  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(nb,1,1);

  cuda_copy_double_complex_a_tmp1_kernel<<<blocks,threadsPerBlock>>>(a_casted, tmp1_casted, l_row1, l_col1, matrixRows, nb);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_double_complex_a_tmp1_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_copy_float_complex_a_tmp1_kernel(cuFloatComplex *a_dev, cuFloatComplex *tmp1_dev, const int l_row1, const int l_col1, const int matrixRows, const int nb){

  int i_index    = threadIdx.x +1;  // range 1..nb
  int j_index    = blockIdx.x +1;  // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1] = a_dev[l_row1-1+j_index-1 + (l_col1-1+i_index-1)*matrixRows];
  }
}

extern "C" void cuda_copy_float_complex_a_tmp1_FromC(float _Complex *a_dev, float _Complex *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in){
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;

  cuFloatComplex* a_casted = (cuFloatComplex*) a_dev;
  cuFloatComplex* tmp1_casted = (cuFloatComplex*) tmp1_dev;


  dim3 threadsPerBlock = dim3(nb, 1, 1);
  dim3 blocks = dim3(nb,1,1);

  cuda_copy_float_complex_a_tmp1_kernel<<<blocks,threadsPerBlock>>>(a_casted, tmp1_casted, l_row1, l_col1, matrixRows, nb);
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_float_complex_a_tmp1_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

