//    Copyright 2024, A. Marek
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

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

__global__ void hip_copy_real_part_to_q_double_complex_kernel(hipDoubleComplex *q, const double *q_real, const int matrixRows, const int l_rows, const int l_cols_nev) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    //if (row < l_rows && col < l_cols_nev) {
    //    int index = row * l_cols_nev + col;
    //    q[index].x = q_real[index]; 
    //    q[index].y = 0.0; 
    //}

    if (row < l_rows && col < l_cols_nev) {
        int index = row + matrixRows * col;
        q[index].x = q_real[index]; 
        q[index].y = 0.0; 
    }
}

extern "C" void hip_copy_real_part_to_q_double_complex_FromC(double _Complex *q_dev, double *q_real_dev, int *matrixRows_in, int *l_rows_in, int *l_cols_nev_in, hipStream_t  my_stream){
  int l_rows = *l_rows_in;
  int l_cols_nev = *l_cols_nev_in;
  int matrixRows = *matrixRows_in;

  hipDoubleComplex* q_casted = (hipDoubleComplex*) q_dev;

  dim3 threadsPerBlock(32, 32); 
  dim3 blocks((l_rows + threadsPerBlock.x - 1) / threadsPerBlock.x, (l_cols_nev + threadsPerBlock.y - 1) / threadsPerBlock.y);

#ifdef WITH_GPU_STREAMS
  hip_copy_real_part_to_q_double_complex_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_casted, q_real_dev, matrixRows, l_rows, l_cols_nev);
#else
  hip_copy_real_part_to_q_double_complex_kernel<<<blocks, threadsPerBlock>>>(q_casted, q_real_dev, matrixRows, l_rows, l_cols_nev);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_real_part_to_q_double_complex_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_real_part_to_q_float_complex_kernel(hipFloatComplex *q, const float *q_real, const int matrixRows, const int l_rows, const int l_cols_nev) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < l_rows && col < l_cols_nev) {
        int index = row + matrixRows * col;
        q[index].x = q_real[index]; 
        q[index].y = 0.0f; 
    }
}

extern "C" void hip_copy_real_part_to_q_float_complex_FromC(float _Complex *q_dev, float *q_real_dev, int *matrixRows_in, int *l_rows_in, int *l_cols_nev_in, hipStream_t  my_stream){
  int l_rows = *l_rows_in;
  int l_cols_nev = *l_cols_nev_in;
  int matrixRows = *matrixRows_in;

  hipFloatComplex* q_casted = (hipFloatComplex*) q_dev;

  dim3 threadsPerBlock(32, 32); 
  dim3 blocks((l_rows + threadsPerBlock.x - 1) / threadsPerBlock.x, (l_cols_nev + threadsPerBlock.y - 1) / threadsPerBlock.y);

#ifdef WITH_GPU_STREAMS
  hip_copy_real_part_to_q_float_complex_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_casted, q_real_dev, matrixRows, l_rows, l_cols_nev);
#else
  hip_copy_real_part_to_q_float_complex_kernel<<<blocks, threadsPerBlock>>>(q_casted, q_real_dev, matrixRows, l_rows, l_cols_nev);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_real_part_to_q_float_complex_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_zero_skewsymmetric_q_double_kernel(double *q, const int matrixRows, const int matrixCols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < matrixRows && col >= matrixCols && col < 2 * matrixCols) {
        int index = row + (matrixRows) * col;
        q[index] = 0.0;
    }
}

extern "C" void hip_zero_skewsymmetric_q_double_FromC(double *q_dev, int *matrixRows_in, int *matrixCols_in, hipStream_t  my_stream){
  int matrixCols = *matrixCols_in;
  int matrixRows = *matrixRows_in;

  dim3 threadsPerBlock(32, 32);
  dim3 blocks((matrixRows + threadsPerBlock.x - 1) / threadsPerBlock.x, (matrixCols + threadsPerBlock.y - 1) / threadsPerBlock.y);

#ifdef WITH_GPU_STREAMS
  hip_zero_skewsymmetric_q_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, matrixRows, matrixCols);
#else
  hip_zero_skewsymmetric_q_double_kernel<<<blocks, threadsPerBlock>>>(q_dev, matrixRows, matrixCols);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_zero_skewsymmetric_q_double_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_zero_skewsymmetric_q_float_kernel(float *q, const int matrixRows, const int matrixCols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    //if (row < matrixRows && col >= matrixCols && col < 2 * matrixCols) {
    //    int index = row * (2 * matrixCols) + col;
    //    q[index] = 0.0f;
    //}
    if (row < matrixRows && col >= matrixCols && col < 2 * matrixCols) {
        int index = row + (matrixRows) * col;
        q[index] = 0.0f;
    }
}

extern "C" void hip_zero_skewsymmetric_q_float_FromC(float *q_dev, int *matrixRows_in, int *matrixCols_in, hipStream_t  my_stream){
  int matrixCols = *matrixCols_in;
  int matrixRows = *matrixRows_in;

  dim3 threadsPerBlock(32, 32);
  //dim3 blocks((matrixRows + threadsPerBlock.x - 1) / threadsPerBlock.x, (2 * matrixCols + threadsPerBlock.y - 1) / threadsPerBlock.y);
  dim3 blocks((matrixRows + threadsPerBlock.x - 1) / threadsPerBlock.x, (matrixCols + threadsPerBlock.y - 1) / threadsPerBlock.y);

#ifdef WITH_GPU_STREAMS
  hip_zero_skewsymmetric_q_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, matrixRows, matrixCols);
#else
  hip_zero_skewsymmetric_q_float_kernel<<<blocks, threadsPerBlock>>>(q_dev, matrixRows, matrixCols);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_zero_skewsymmetric_q_float_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_skewsymmetric_second_half_q_double_minus_kernel(double *q, const int i, const int matrixRows, const int matrixCols) {


#if 0
             q(i,matrixCols+1:2*matrixCols) = -q(i,1:matrixCols)
             q(i,1:matrixCols) = 0

    i is given from outside
    for (j=matrixCols;j<2*matrixCols;j++){
	    q[(i-1) + matrixRows * j] = q [(i-1) + matrixRows * (j-matrixCols)]

    }
    threadsPerBlock = 1024
	   => threadIdx.x = 0..1023
    blocks = (matrixCols + 1024 - 1) / 1024
          => blocksIdx.x = 0...blocks - 1
    
    col =  (0...blocks-1)*1024 + 0..1023  => col = 0..1023 ; 1024+0..1023 ; 2048+0..1023
#endif

    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (col >= matrixCols && col < 2 * matrixCols) {  // geht nivht
        int    index = (i-1) + matrixRows * (col);
        int indexLow = (i-1) + matrixRows * (col-matrixCols);
        q[index] = -q[indexLow];
        q[indexLow] = 0.0;
    }	

}

__global__ void hip_copy_skewsymmetric_second_half_q_double_plus_kernel(double *q, const int i, const int matrixRows, const int matrixCols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //if (col >= matrixCols && col <= 2 * matrixCols) {
    //    int index = (i-1) * (2 * matrixCols) + col;
    //    q[index] = q[index - matrixCols];
    //    q[index - matrixCols] = 0.0;
    //}	
    //if (col >= matrixCols  && col < 2 * matrixCols) {
    //    int index = (i-1) + matrixRows * (col);
    //    q[index] = q[index - matrixCols];
    //    //q[index - matrixCols] = 0.0;
    //}	
    //if (col < matrixCols) {
    //    int index = (i-1) + matrixRows * (col);    //geht
    //    //q[index] = q[index - matrixCols];
    //    q[index] = 0.0;
    //}	


    //for (col=0; col<matrixCols;col++) {  // geht
    //    int index = (i-1) + matrixRows * (col);
    //    q[index] = 0.0;
    // }

    //for (col=matrixCols; col<2*matrixCols;col++) { //geht nicht
    //    int index = (i-1) + matrixRows * (col);
    //    q[index - matrixCols] = 0.0;
    // }
    //if (col >= matrixCols && col < 2 * matrixCols) {  // geht nivht
    //    int index = (i-1) + matrixRows * (col);
    //    //q[index] = q[index - matrixCols];
    //    q[index - matrixCols] = 0.0;
    //}	
    //for (int col2=matrixCols; col2<2*matrixCols;col2++) {  // geht
    //    int index = (i-1) + matrixRows * (col2-matrixCols);
    //    q[index] = 0.0;
    // }
    if (col >= matrixCols && col < 2 * matrixCols) {  // geht nivht
        int    index = (i-1) + matrixRows * (col);
        int indexLow = (i-1) + matrixRows * (col-matrixCols);
        q[index] = q[indexLow];
        q[indexLow] = 0.0;
    }	
}

extern "C" void hip_copy_skewsymmetric_second_half_q_double_FromC(double *q_dev, int *i_in, int *matrixRows_in, int *matrixCols_in, int *negative_or_positive_in, hipStream_t  my_stream){
  int matrixCols = *matrixCols_in;
  int matrixRows = *matrixRows_in;
  int negative_or_positive = *negative_or_positive_in;
  int i = *i_in;

  dim3 threadsPerBlock(1024);
  dim3 blocks((2*matrixCols + threadsPerBlock.x - 1) / threadsPerBlock.x);
  //dim3 threadsPerBlock(1);
  //dim3 blocks(1);

  if (negative_or_positive == 1) {
#ifdef WITH_GPU_STREAMS
    hip_copy_skewsymmetric_second_half_q_double_plus_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, i, matrixRows, matrixCols);
#else
    hip_copy_skewsymmetric_second_half_q_double_plus_kernel<<<blocks, threadsPerBlock>>>(q_dev, i, matrixRows, matrixCols);
#endif
  } else {
#ifdef WITH_GPU_STREAMS
    hip_copy_skewsymmetric_second_half_q_double_minus_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, i, matrixRows, matrixCols);
#else
    hip_copy_skewsymmetric_second_half_q_double_minus_kernel<<<blocks, threadsPerBlock>>>(q_dev, i, matrixRows, matrixCols);
#endif
  }

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_skewsymmetric_second_half_q_double_plus/minus_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_skewsymmetric_second_half_q_float_minus_kernel(float *q, const int i, const int matrixRows, const int matrixCols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= matrixCols && col < 2 * matrixCols) {  // geht nivht
        int    index = (i-1) + matrixRows * (col);
        int indexLow = (i-1) + matrixRows * (col-matrixCols);
        q[index] = -q[indexLow];
        q[indexLow] = 0.0f;
    }	
}

__global__ void hip_copy_skewsymmetric_second_half_q_float_plus_kernel(float *q, const int i, const int matrixRows, const int matrixCols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= matrixCols && col < 2 * matrixCols) {  // geht nivht
        int    index = (i-1) + matrixRows * (col);
        int indexLow = (i-1) + matrixRows * (col-matrixCols);
        q[index] = q[indexLow];
        q[indexLow] = 0.0f;
    }	
}

extern "C" void hip_copy_skewsymmetric_second_half_q_float_FromC(float *q_dev, int *i_in, int *matrixRows_in, int *matrixCols_in, int *negative_or_positive_in, hipStream_t  my_stream){
  int matrixCols = *matrixCols_in;
  int matrixRows = *matrixRows_in;
  int negative_or_positive = *negative_or_positive_in;
  int i = *i_in;

  dim3 threadsPerBlock(1024);
  dim3 blocks((2*matrixCols + threadsPerBlock.x - 1) / threadsPerBlock.x);

  if (negative_or_positive == 1) {
#ifdef WITH_GPU_STREAMS
    hip_copy_skewsymmetric_second_half_q_float_plus_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, i, matrixRows, matrixCols);
#else
    hip_copy_skewsymmetric_second_half_q_float_plus_kernel<<<blocks, threadsPerBlock>>>(q_dev, i, matrixRows, matrixCols);
#endif
  } else {
#ifdef WITH_GPU_STREAMS
    hip_copy_skewsymmetric_second_half_q_float_minus_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, i, matrixRows, matrixCols);
#else
    hip_copy_skewsymmetric_second_half_q_float_minus_kernel<<<blocks, threadsPerBlock>>>(q_dev, i, matrixRows, matrixCols);
#endif
  }

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_skewsymmetric_second_half_q_float_plus/minus_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_skewsymmetric_first_half_q_double_minus_kernel(double *q, const int i, const int matrixRows, const int matrixCols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < matrixCols) {
    int index = (i-1) + matrixRows * col;
    q[index] = -q[index];
  }
}

extern "C" void hip_copy_skewsymmetric_first_half_q_double_FromC(double *q_dev, int *i_in, int *matrixRows_in, int *matrixCols_in, int *negative_or_positive_in, hipStream_t  my_stream){
  int matrixRows = *matrixRows_in;
  int matrixCols = *matrixCols_in;
  int negative_or_positive = *negative_or_positive_in;
  int i = *i_in;

  dim3 threadsPerBlock(1024);
  dim3 blocks((matrixCols + threadsPerBlock.x - 1) / threadsPerBlock.x);

#ifdef WITH_GPU_STREAMS
    hip_copy_skewsymmetric_first_half_q_double_minus_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, i, matrixRows, matrixCols);
#else
    hip_copy_skewsymmetric_first_half_q_double_minus_kernel<<<blocks, threadsPerBlock>>>(q_dev, i, matrixRows, matrixCols);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_skewsymmetric_first_half_q_double_plus/minus_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_copy_skewsymmetric_first_half_q_float_minus_kernel(float *q, const int i, const int matrixRows, const int matrixCols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < matrixCols) {
    int index = (i-1) + matrixRows * col;
    q[index] = -q[index];
  }
}

extern "C" void hip_copy_skewsymmetric_first_half_q_float_FromC(float *q_dev, int *i_in, int *matrixRows_in, int *matrixCols_in, int *negative_or_positive_in, hipStream_t  my_stream){
  int matrixRows = *matrixRows_in;
  int matrixCols = *matrixCols_in;
  int negative_or_positive = *negative_or_positive_in;
  int i = *i_in;

  dim3 threadsPerBlock(1024);
  dim3 blocks((matrixCols + threadsPerBlock.x - 1) / threadsPerBlock.x);

#ifdef WITH_GPU_STREAMS
    hip_copy_skewsymmetric_first_half_q_float_minus_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, i, matrixRows, matrixCols);
#else
    hip_copy_skewsymmetric_first_half_q_float_minus_kernel<<<blocks, threadsPerBlock>>>(q_dev, i, matrixRows, matrixCols);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_skewsymmetric_first_half_q_float_plus/minus_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_get_skewsymmetric_second_half_q_double_kernel(double *q, double* q_2, const int matrixRows, const int matrixCols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < matrixRows && col < matrixCols) {
    int index  = row + matrixRows * col;
    int index2 = row + matrixRows * (col + matrixCols);
    q_2[index] = q[index2];
  }
}



//         ! copy q_part2(1:matrixRows,1:matrixCols) = q(1:matrixRows, matrixCols+1:2*matrixCols)
//         my_stream = obj%gpu_setup%my_stream
//         call GPU_GET_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION_REAL(q_dev, q_part2_dev, matrixRows, matrixCols, &
//                                                                my_stream)


extern "C" void hip_get_skewsymmetric_second_half_q_double_FromC(double *q_dev, double *q_2_dev, int *matrixRows_in, int *matrixCols_in, hipStream_t  my_stream){
  int matrixRows = *matrixRows_in;
  int matrixCols = *matrixCols_in;

  dim3 threadsPerBlock(32,32);
  dim3 blocks( (matrixRows + threadsPerBlock.x - 1) / threadsPerBlock.x, (matrixCols + threadsPerBlock.y - 1) / threadsPerBlock.y);

#ifdef WITH_GPU_STREAMS
    hip_get_skewsymmetric_second_half_q_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, q_2_dev, matrixRows, matrixCols);
#else
    hip_get_skewsymmetric_second_half_q_double_kernel<<<blocks, threadsPerBlock>>>(q_dev, q_2_dev, matrixRows, matrixCols);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_get_skewsymmetric_second_half_q_double_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_get_skewsymmetric_second_half_q_float_kernel(float *q, float *q_2, const int matrixRows, const int matrixCols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < matrixRows && col < matrixCols) {
    int index  = row + matrixRows * col;
    int index2 = row + matrixRows * (col + matrixCols);
    q_2[index] = q[index2];
  }
}

extern "C" void hip_get_skewsymmetric_second_half_q_float_FromC(float *q_dev, float *q_2_dev, int *matrixRows_in, int *matrixCols_in, hipStream_t  my_stream){
  int matrixRows = *matrixRows_in;
  int matrixCols = *matrixCols_in;

  dim3 threadsPerBlock(32,32);
  dim3 blocks( (matrixRows + threadsPerBlock.x - 1) / threadsPerBlock.x, (matrixCols + threadsPerBlock.y - 1) / threadsPerBlock.y);

#ifdef WITH_GPU_STREAMS
    hip_get_skewsymmetric_second_half_q_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, q_2_dev, matrixRows, matrixCols);
#else
    hip_get_skewsymmetric_second_half_q_float_kernel<<<blocks, threadsPerBlock>>>(q_dev, q_2_dev, matrixRows, matrixCols);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_get_skewsymmetric_second_half_q_float_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_put_skewsymmetric_second_half_q_double_kernel(double *q, double* q_2, const int matrixRows, const int matrixCols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < matrixRows && col < matrixCols) {
    int index  = row + matrixRows * col;
    int index2 = row + matrixRows * (col + matrixCols);
    q[index2] = q_2[index];
  }
}

extern "C" void hip_put_skewsymmetric_second_half_q_double_FromC(double *q_dev, double *q2_dev, int *matrixRows_in, int *matrixCols_in, hipStream_t  my_stream){
  int matrixRows = *matrixRows_in;
  int matrixCols = *matrixCols_in;

  dim3 threadsPerBlock(32,32);
  dim3 blocks( (matrixRows + threadsPerBlock.x - 1) / threadsPerBlock.x, (matrixCols + threadsPerBlock.y - 1) / threadsPerBlock.y);

#ifdef WITH_GPU_STREAMS
    hip_put_skewsymmetric_second_half_q_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, q2_dev, matrixRows, matrixCols);
#else
    hip_put_skewsymmetric_second_half_q_double_kernel<<<blocks, threadsPerBlock>>>(q_dev, q2_dev, matrixRows, matrixCols);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_put_skewsymmetric_second_half_q_double_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

__global__ void hip_put_skewsymmetric_second_half_q_float_kernel(float *q, float* q_2, const int matrixRows, const int matrixCols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < matrixRows && col < matrixCols) {
    int index  = row + matrixRows * col;
    int index2 = row + matrixRows * (col + matrixCols);
    q[index2] = q_2[index];
  }
}

extern "C" void hip_put_skewsymmetric_second_half_q_float_FromC(float *q_dev, float *q2_dev, int *matrixRows_in, int *matrixCols_in, hipStream_t  my_stream){
  int matrixRows = *matrixRows_in;
  int matrixCols = *matrixCols_in;

  dim3 threadsPerBlock(32,32);
  dim3 blocks( (matrixRows + threadsPerBlock.x - 1) / threadsPerBlock.x, (matrixCols + threadsPerBlock.y - 1) / threadsPerBlock.y);

#ifdef WITH_GPU_STREAMS
    hip_put_skewsymmetric_second_half_q_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, q2_dev, matrixRows, matrixCols);
#else
    hip_put_skewsymmetric_second_half_q_float_kernel<<<blocks, threadsPerBlock>>>(q_dev, q2_dev, matrixRows, matrixCols);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_put_skewsymmetric_second_half_q_float_kernel: %s\n",hipGetErrorString(cuerr));
  }
}

