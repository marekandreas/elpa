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
  
#include "config-f90.h"
  
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


__global__ void hip_update_d_double_kernel(int *limits, double *d, double *e, const int ndiv, const int na) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;


    for (int ii=0;ii<ndiv-1;ii++) {
      int n = limits[ii]-1;
      d[n]   = d[n]   - fabs(e[n]);
      d[n+1] = d[n+1] - fabs(e[n]);

    }
}



extern "C" void hip_update_d_double_FromC(int *limits_dev, double *d_dev, double *e_dev, int *ndiv_in, int *na_in, hipStream_t  my_stream){
  int na = *na_in;
  int ndiv = *ndiv_in;

  //dim3 threadsPerBlock(1024);
  //dim3 blocks((na1 + threadsPerBlock.x - 1) / threadsPerBlock.x);

  dim3 threadsPerBlock(1);
  dim3 blocks(1);

#ifdef WITH_GPU_STREAMS
  hip_update_d_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(limits_dev, d_dev, e_dev, ndiv, na);
#else
  hip_update_d_double_kernel<<<blocks, threadsPerBlock>>>              (limits_dev, d_dev, e_dev, ndiv, na);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_update_d_double_kernel: %s\n",hipGetErrorString(cuerr));
  }
}


__global__ void hip_update_d_float_kernel(int *limits, float *d, float *e, const int ndiv, const int na) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;


    for (int ii=0;ii<ndiv-1;ii++) {
      int n = limits[ii]-1;
      d[n]   = d[n]   - fabsf(e[n]);
      d[n+1] = d[n+1] - fabsf(e[n]);

    }
}



extern "C" void hip_update_d_float_FromC(int *limits_dev, float *d_dev, float *e_dev, int *ndiv_in, int *na_in, hipStream_t  my_stream){
  int na = *na_in;
  int ndiv = *ndiv_in;

  //dim3 threadsPerBlock(1024);
  //dim3 blocks((na1 + threadsPerBlock.x - 1) / threadsPerBlock.x);

  dim3 threadsPerBlock(1);
  dim3 blocks(1);

#ifdef WITH_GPU_STREAMS
  hip_update_d_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(limits_dev, d_dev, e_dev, ndiv, na);
#else
  hip_update_d_float_kernel<<<blocks, threadsPerBlock>>>              (limits_dev, d_dev, e_dev, ndiv, na);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_update_d_float_kernel: %s\n",hipGetErrorString(cuerr));
  }
}



__global__ void hip_copy_qmat1_to_qmat2_double_kernel(double *qmat1, double *qmat2, const int max_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 0 && i < max_size) {
      if ( j>=0 && j < max_size) {
        qmat2[i + max_size*j] = qmat1[i + max_size*j];
      }
    }
}



extern "C" void hip_copy_qmat1_to_qmat2_double_FromC(double *qmat1_dev, double *qmat2_dev, int *max_size_in, hipStream_t  my_stream){
  int max_size = *max_size_in;

  dim3 threadsPerBlock(32,32);
  dim3 blocks((max_size + threadsPerBlock.x - 1) / threadsPerBlock.x, (max_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

#ifdef WITH_GPU_STREAMS
  hip_copy_qmat1_to_qmat2_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(qmat1_dev, qmat2_dev, max_size);
#else
  hip_copy_qmat1_to_qmat2_double_kernel<<<blocks, threadsPerBlock>>>              (qmat1_dev, qmat2_dev, max_size);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_qmat1_qmat2_double_kernel: %s\n",hipGetErrorString(cuerr));
  }
}



__global__ void hip_copy_qmat1_to_qmat2_float_kernel(float *qmat1, float *qmat2, const int max_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 0 && i < max_size) {
      if ( j>=0 && j < max_size) {
        qmat2[i + max_size*j] = qmat1[i + max_size*j];
      }
    }
}



extern "C" void hip_copy_qmat1_to_qmat2_float_FromC(float *qmat1_dev, float *qmat2_dev, int *max_size_in, hipStream_t  my_stream){
  int max_size = *max_size_in;

  dim3 threadsPerBlock(32,32);
  dim3 blocks((max_size + threadsPerBlock.x - 1) / threadsPerBlock.x, (max_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

#ifdef WITH_GPU_STREAMS
  hip_copy_qmat1_to_qmat2_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(qmat1_dev, qmat2_dev, max_size);
#else
  hip_copy_qmat1_to_qmat2_float_kernel<<<blocks, threadsPerBlock>>>              (qmat1_dev, qmat2_dev, max_size);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_qmat1_qmat2_float_kernel: %s\n",hipGetErrorString(cuerr));
  }
}




__global__ void hip_copy_d_to_d_tmp_double_kernel(double *d, double *d_tmp, const int na) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 0 && i < na) {
       d_tmp[i] = d[i];
    }
}


extern "C" void hip_copy_d_to_d_tmp_double_FromC(double *d_dev, double *d_tmp_dev, int *na_in, hipStream_t  my_stream){
  int na = *na_in;

  dim3 threadsPerBlock(1024);
  dim3 blocks((na + threadsPerBlock.x - 1) / threadsPerBlock.x);

#ifdef WITH_GPU_STREAMS
  hip_copy_d_to_d_tmp_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(d_dev, d_tmp_dev, na);
#else
  hip_copy_d_to_d_tmp_double_kernel<<<blocks, threadsPerBlock>>>              (d_dev, d_tmp_dev, na);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_d_to_d_tmp_double_kernel: %s\n",hipGetErrorString(cuerr));
  }
}




__global__ void hip_copy_d_to_d_tmp_float_kernel(float *d, float *d_tmp, const int na) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 0 && i < na) {
       d_tmp[i] = d[i];
    }
}


extern "C" void hip_copy_d_to_d_tmp_float_FromC(float *d_dev, float *d_tmp_dev, int *na_in, hipStream_t  my_stream){
  int na = *na_in;

  dim3 threadsPerBlock(1024);
  dim3 blocks((na + threadsPerBlock.x - 1) / threadsPerBlock.x);

#ifdef WITH_GPU_STREAMS
  hip_copy_d_to_d_tmp_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(d_dev, d_tmp_dev, na);
#else
  hip_copy_d_to_d_tmp_float_kernel<<<blocks, threadsPerBlock>>>              (d_dev, d_tmp_dev, na);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_d_to_d_tmp_float_kernel: %s\n",hipGetErrorString(cuerr));
  }
}



__global__ void hip_copy_q_to_q_tmp_double_kernel(double *q, double *q_tmp, const int ldq, const int nlen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 0 && i < ldq) {
      if (j >= 0 && j < nlen) {
         q_tmp[i+ldq*j] = q[i+ldq*j];
      }
    }
}


extern "C" void hip_copy_q_to_q_tmp_double_FromC(double *q_dev, double *q_tmp_dev, int *ldq_in, int *nlen_in, hipStream_t  my_stream){
  int ldq  = *ldq_in;
  int nlen = *nlen_in;

  dim3 threadsPerBlock(32,32);
  dim3 blocks((ldq + threadsPerBlock.x - 1) / threadsPerBlock.x,(nlen + threadsPerBlock.y - 1) / threadsPerBlock.y);

#ifdef WITH_GPU_STREAMS
  hip_copy_q_to_q_tmp_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, q_tmp_dev, ldq, nlen);
#else
  hip_copy_q_to_q_tmp_double_kernel<<<blocks, threadsPerBlock>>>              (q_dev, q_tmp_dev, ldq, nlen);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_q_to_q_tmp_double_kernel: %s\n",hipGetErrorString(cuerr));
  }
}



__global__ void hip_copy_q_to_q_tmp_float_kernel(float *q, float *q_tmp, const int ldq, const int nlen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 0 && i < ldq) {
      if (j >= 0 && j < nlen) {
         q_tmp[i+ldq*j] = q[i+ldq*j];
      }
    }
}


extern "C" void hip_copy_q_to_q_tmp_float_FromC(float *q_dev, float *q_tmp_dev, int *ldq_in, int *nlen_in, hipStream_t  my_stream){
  int ldq  = *ldq_in;
  int nlen = *nlen_in;

  dim3 threadsPerBlock(32,32);
  dim3 blocks((ldq + threadsPerBlock.x - 1) / threadsPerBlock.x,(nlen + threadsPerBlock.y - 1) / threadsPerBlock.y);

#ifdef WITH_GPU_STREAMS
  hip_copy_q_to_q_tmp_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, q_tmp_dev, ldq, nlen);
#else
  hip_copy_q_to_q_tmp_float_kernel<<<blocks, threadsPerBlock>>>              (q_dev, q_tmp_dev, ldq, nlen);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_q_to_q_tmp_float_kernel: %s\n",hipGetErrorString(cuerr));
  }
}



__global__ void hip_copy_q_tmp_to_q_double_kernel(double *q_tmp, double *q, const int ldq, const int nlen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 0 && i < ldq) {
      if (j >= 0 && j < nlen) {
         q[i+ldq*j] = q_tmp[i+ldq*j];
      }
    }
}


extern "C" void hip_copy_q_tmp_to_q_double_FromC(double *q_tmp_dev, double *q_dev, int *ldq_in, int *nlen_in, hipStream_t  my_stream){
  int ldq  = *ldq_in;
  int nlen = *nlen_in;

  dim3 threadsPerBlock(32,32);
  dim3 blocks((ldq + threadsPerBlock.x - 1) / threadsPerBlock.x,(nlen + threadsPerBlock.y - 1) / threadsPerBlock.y);

#ifdef WITH_GPU_STREAMS
  hip_copy_q_tmp_to_q_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_tmp_dev, q_dev, ldq, nlen);
#else
  hip_copy_q_tmp_to_q_double_kernel<<<blocks, threadsPerBlock>>>              (q_tmp_dev, q_dev, ldq, nlen);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_q_tmp_to_q_double_kernel: %s\n",hipGetErrorString(cuerr));
  }
}



__global__ void hip_copy_q_tmp_to_q_float_kernel(float *q_tmp, float *q, const int ldq, const int nlen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 0 && i < ldq) {
      if (j >= 0 && j < nlen) {
         q[i+ldq*j] = q_tmp[i+ldq*j];
      }
    }
}


extern "C" void hip_copy_q_tmp_to_q_float_FromC(float *q_tmp_dev, float *q_dev, int *ldq_in, int *nlen_in, hipStream_t  my_stream){
  int ldq  = *ldq_in;
  int nlen = *nlen_in;

  dim3 threadsPerBlock(32,32);
  dim3 blocks((ldq + threadsPerBlock.x - 1) / threadsPerBlock.x,(nlen + threadsPerBlock.y - 1) / threadsPerBlock.y);

#ifdef WITH_GPU_STREAMS
  hip_copy_q_tmp_to_q_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_tmp_dev, q_dev, ldq, nlen);
#else
  hip_copy_q_tmp_to_q_float_kernel<<<blocks, threadsPerBlock>>>              (q_tmp_dev, q_dev, ldq, nlen);
#endif

  hipError_t cuerr = hipGetLastError();
  if (cuerr != hipSuccess){
    printf("Error in executing hip_copy_q_tmp_to_q_float_kernel: %s\n",hipGetErrorString(cuerr));
  }
}



