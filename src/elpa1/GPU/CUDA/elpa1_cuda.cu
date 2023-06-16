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

__global__ void cuda_update_matrix_element_add_double_kernel(double *a_dev, int index, double value, double *d_vec_dev, int istep, int n_stored_vecs, int const isSkewsymmetric){
  if (n_stored_vecs > 0){
    a_dev[index] += value;
    }

    if (isSkewsymmetric) {
      d_vec_dev[istep-1-1] = 0.0;
    }
    else {
      d_vec_dev[istep-1-1] = a_dev[index]; // (l_rows,l_cols)
    }
/*
#endif
#if COMPLEXCASE == 1
      d_vec(istep-1) = real(a_mat(l_rows,l_cols),kind=rk)
#endif
*/
}

extern "C" void cuda_update_matrix_element_add_double_FromC(double *a_dev, int *index_in, double *value_in, 
                          double *d_vec_dev, int *istep_in, int *n_stored_vecs_in, int* isSkewsymmetric_in, cudaStream_t  my_stream){
  int index = *index_in;   
  int istep = *istep_in;   
  int n_stored_vecs = *n_stored_vecs_in; 
  int isSkewsymmetric = *isSkewsymmetric_in;   
  double value = *value_in;

  dim3 blocks = dim3(1,1,1);
  dim3 threadsPerBlock = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  cuda_update_matrix_element_add_double_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(a_dev, index, value, d_vec_dev, istep, n_stored_vecs, isSkewsymmetric);
#else
  cuda_update_matrix_element_add_double_kernel<<<blocks,threadsPerBlock>>>(a_dev, index, value, d_vec_dev, istep, n_stored_vecs, isSkewsymmetric);
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

extern "C" void cuda_update_array_element_double_FromC(double *array_dev, int *index_in, double *value_in, cudaStream_t  my_stream){
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
