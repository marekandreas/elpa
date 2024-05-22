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
#include <cuComplex.h>
#include <stdint.h>
#include "config-f90.h"

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

__global__ void cuda_scale_qmat_double_complex_kernel(cuDoubleComplex *q, cuDoubleComplex *tau, const int ldq, const int l_cols) {
    
    double one = 1.0;
    double zero = 0.0;
    cuDoubleComplex c_one = make_cuDoubleComplex(one, zero);

    //printf("c: tau[1]=%.6f %.6f \n",tau[1].x,tau[1].y);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = (1-1) + ldq * col;

    // q_mat(1,1:l_cols) = q_mat(1,1:l_cols)*(ONE-tau(2))
    if (col < l_cols) {

	    // (a + ib) (c + id) = (ac - bd) + i(ad + bc)
            // a = q.x
	    // b = q.y
	    // c = one-tau.x
	    // d = zero - tau.y
	    // (q.x + i * q.y) * ((one-tau.x) + i * (zero-tau.y)) = (q.x*(one-tau.x) - q.y * (zero-tau.y)) + i * (q.x * (zero-tau.y) + q.y * (one-tau.x)
        //// real part 
        //q[index].x = q[index].x * (one-tau[1].x) - q[index].y * (zero - tau[1].y);
	//// imag part
	//q[index].y = q[index].x * (zero - tau[1].y) + q[index].y * (one - tau[1].x);

        //// real part 
        //q[index].x = q[index].x * (one-tau[1].x) + q[index].y * tau[1].y;
	//// imag part
	//q[index].y = -q[index].x * tau[1].y + q[index].y * (one - tau[1].x);


        //q[index].x = q[index].x * (one - tau[1].x);
        //q[index].y = q[index].y * (zero - tau[1].y);

      q[index] = cuCmul(q[index], cuCsub(c_one, tau[1]));
    }
}

extern "C" void cuda_scale_qmat_double_complex_FromC(int *ldq_in, int *l_cols_in, double _Complex *q_dev, double _Complex *tau_dev, cudaStream_t  my_stream){
  int ldq = *ldq_in;
  int l_cols = *l_cols_in;

  cuDoubleComplex* q_casted = (cuDoubleComplex*) q_dev;
  cuDoubleComplex* tau_casted = (cuDoubleComplex*) tau_dev;

  dim3 threadsPerBlock(1024); 
  dim3 blocks((l_cols + threadsPerBlock.x - 1) / threadsPerBlock.x);

#ifdef WITH_GPU_STREAMS
  cuda_scale_qmat_double_complex_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_casted, tau_casted, ldq, l_cols);
#else
  cuda_scale_qmat_double_complex_kernel<<<blocks, threadsPerBlock>>>(q_casted, tau_casted, ldq, l_cols);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_scale_qmat_double_complex_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

__global__ void cuda_scale_qmat_float_complex_kernel(cuFloatComplex *q, cuFloatComplex *tau, const int ldq, const int l_cols) {
    
    float one = 1.0f;
    float zero = 0.0f;
    cuFloatComplex c_one = make_cuFloatComplex(one, zero);

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = (1-1) + ldq * col;

    // q_mat(1,1:l_cols) = q_mat(1,1:l_cols)*(ONE-tau(2))
    if (col < l_cols) {
      //  q[index].x = q[index].x * (one - tau[1].x);
      //  q[index].y = q[index].y * (zero - tau[1].y);
      q[index] = cuCmulf(q[index], cuCsubf(c_one, tau[1]));
    }

}

extern "C" void cuda_scale_qmat_float_complex_FromC(int *ldq_in, int *l_cols_in, float _Complex *q_dev, float _Complex *tau_dev, cudaStream_t  my_stream){
  int ldq = *ldq_in;
  int l_cols = *l_cols_in;

  cuFloatComplex* q_casted = (cuFloatComplex*) q_dev;
  cuFloatComplex* tau_casted = (cuFloatComplex*) tau_dev;

  dim3 threadsPerBlock(1024); 
  dim3 blocks((l_cols + threadsPerBlock.x - 1) / threadsPerBlock.x);

#ifdef WITH_GPU_STREAMS
  cuda_scale_qmat_float_complex_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_casted, tau_casted, ldq, l_cols);
#else
  cuda_scale_qmat_float_complex_kernel<<<blocks, threadsPerBlock>>>(q_casted, tau_casted, ldq, l_cols);
#endif
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_scale_qmat_float_complex_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

