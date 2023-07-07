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
#include <stdint.h>
#include "config-f90.h"
#include <complex>

#include <CL/sycl.hpp>
#include "src/GPU/SYCL/syclCommon.hpp"

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

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

//________________________________________________________________

template <typename T>
void sycl_copy_tmp2_tmat2_kernel(T *tmp2_dev, T *tmat2_dev, const int nblk, const int l_col1, sycl::nd_item<1> it){

  int nb_index = it.get_local_id(0) + 1; // range 1..nb
  int l_col_index = it.get_group(0) + 1; // range 1..nb

  tmat2_dev[nb_index-1 + (l_col1-1 + l_col_index -1)*nblk] = tmp2_dev[nb_index-1 + (1 -1 + l_col_index-1)  * nblk];

}

template <typename T>
void sycl_copy_tmp2_tmat2_FromC(T *tmp2_dev, T *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, intptr_t my_stream){

  int nblk   = *nblk_in;   
  int l_col1 = *l_col1_in;
  int nb     = *nb_in;

  sycl::range<1> global_range = sycl::range<1>(nb*nb);
  sycl::range<1> local_range  = sycl::range<1>(nb);

  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  queue.parallel_for(
      sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> it) {
        sycl_copy_tmp2_tmat2_kernel(tmp2_dev, tmat2_dev, nblk, l_col1, it);
      });
  queue.wait_and_throw();

}

extern "C" void sycl_copy_double_tmp2_tmat2_FromC(double *tmp2_dev, double *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, intptr_t my_stream){
  sycl_copy_tmp2_tmat2_FromC(tmp2_dev, tmat2_dev, nblk_in, l_col1_in, nb_in, my_stream);
}

extern "C" void sycl_copy_float_tmp2_tmat2_FromC(float *tmp2_dev, float *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, intptr_t my_stream){
  sycl_copy_tmp2_tmat2_FromC(tmp2_dev, tmat2_dev, nblk_in, l_col1_in, nb_in, my_stream);
}

extern "C" void sycl_copy_double_complex_tmp2_tmat2_FromC(std::complex<double> *tmp2_dev, std::complex<double> *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, intptr_t my_stream){
  sycl_copy_tmp2_tmat2_FromC(tmp2_dev, tmat2_dev, nblk_in, l_col1_in, nb_in, my_stream);
}

extern "C" void sycl_copy_float_complex_tmp2_tmat2_FromC(std::complex<float> *tmp2_dev, std::complex<float> *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, intptr_t my_stream){
  sycl_copy_tmp2_tmat2_FromC(tmp2_dev, tmat2_dev, nblk_in, l_col1_in, nb_in, my_stream);
}

//________________________________________________________________

template <typename T>
void sycl_copy_a_tmat1_kernel(T *a_dev, T *tmat1_dev, const int l_rows, const int matrixRows, const int l_col1, const int nb, const int l_row1, sycl::nd_item<1> it){

  int nb_index = it.get_local_id(0) + 1;  // range 1..nb
  int l_row1_index = it.get_group(0) + 1; // we need l_row1-1 blocks

  tmat1_dev[l_row1_index-1 + (nb_index-1)*l_rows] = a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1 ) * matrixRows];
  a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1)*matrixRows] = 0;

}

template <typename T>
void sycl_copy_a_tmat1_FromC(T *a_dev, T *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, intptr_t my_stream){

  int l_rows = *l_rows_in;   
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;

  sycl::range<1> global_range = sycl::range<1>(nb*(l_row1 - 1));
  sycl::range<1> local_range  = sycl::range<1>(nb);
  
  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  queue.parallel_for(
      sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> it) {
        sycl_copy_a_tmat1_kernel(a_dev, tmat1_dev, l_rows, matrixRows,
                                 l_col1, nb, l_row1, it);
      });
  queue.wait_and_throw();

}


extern "C" void sycl_copy_double_a_tmat1_FromC(double *a_dev, double *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, intptr_t my_stream){
  sycl_copy_a_tmat1_FromC(a_dev, tmat1_dev, l_rows_in, matrixRows_in, nb_in, l_row1_in, l_col1_in, my_stream);
}

extern "C" void sycl_copy_float_a_tmat1_FromC(float *a_dev, float *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, intptr_t my_stream){
  sycl_copy_a_tmat1_FromC(a_dev, tmat1_dev, l_rows_in, matrixRows_in, nb_in, l_row1_in, l_col1_in, my_stream);
}

extern "C" void sycl_copy_double_complex_a_tmat1_FromC(std::complex<double> *a_dev, std::complex<double> *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, intptr_t my_stream){
  sycl_copy_a_tmat1_FromC(a_dev, tmat1_dev, l_rows_in, matrixRows_in, nb_in, l_row1_in, l_col1_in, my_stream);
}

extern "C" void sycl_copy_float_complex_a_tmat1_FromC(std::complex<float> *a_dev, std::complex<float> *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, intptr_t my_stream){
  sycl_copy_a_tmat1_FromC(a_dev, tmat1_dev, l_rows_in, matrixRows_in, nb_in, l_row1_in, l_col1_in, my_stream);
}

//________________________________________________________________

template <typename T>
void sycl_copy_tmp1_tmp2_kernel(T *tmp1_dev, T *tmp2_dev, const int nblk, const int nb, sycl::nd_item<1> it){

  int i_index = it.get_local_id(0) + 1; // range 1..nb
  int j_index = it.get_group(0) + 1;    // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp2_dev[1-1 + j_index-1 + (i_index-1)*nblk] = tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1];
  }
}

template <typename T>
void sycl_copy_tmp1_tmp2_FromC(T *tmp1_dev, T *tmp2_dev, int *nblk_in, int *nb_in, intptr_t my_stream){

  int nblk = *nblk_in;
  int nb = *nb_in;

  sycl::range<1> global_range = sycl::range<1>(nb*nb);
  sycl::range<1> local_range  = sycl::range<1>(nb);
  
  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  queue.parallel_for(
      sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> it) {
        sycl_copy_tmp1_tmp2_kernel(tmp1_dev, tmp2_dev, nblk, nb, it);
      });
  queue.wait_and_throw();

}

extern "C" void sycl_copy_double_tmp1_tmp2_FromC(double *tmp1_dev, double *tmp2_dev, int *nblk_in, int *nb_in, intptr_t my_stream){
  sycl_copy_tmp1_tmp2_FromC(tmp1_dev, tmp2_dev, nblk_in, nb_in, my_stream);
}

extern "C" void sycl_copy_float_tmp1_tmp2_FromC(float *tmp1_dev, float *tmp2_dev, int *nblk_in, int *nb_in, intptr_t my_stream){
  sycl_copy_tmp1_tmp2_FromC(tmp1_dev, tmp2_dev, nblk_in, nb_in, my_stream);
}

extern "C" void sycl_copy_double_complex_tmp1_tmp2_FromC(std::complex<double> *tmp1_dev, std::complex<double> *tmp2_dev, int *nblk_in, int *nb_in, intptr_t my_stream){
  sycl_copy_tmp1_tmp2_FromC(tmp1_dev, tmp2_dev, nblk_in, nb_in, my_stream);
}

extern "C" void sycl_copy_float_complex_tmp1_tmp2_FromC(std::complex<float> *tmp1_dev, std::complex<float> *tmp2_dev, int *nblk_in, int *nb_in, intptr_t my_stream){
  sycl_copy_tmp1_tmp2_FromC(tmp1_dev, tmp2_dev, nblk_in, nb_in, my_stream);
}

//________________________________________________________________

template <typename T>
void sycl_copy_a_tmp1_kernel(T *a_dev, T *tmp1_dev, const int l_row1, const int l_col1, const int matrixRows, const int nb, sycl::nd_item<1> it){

  int i_index = it.get_local_id(0) + 1; // range 1..nb
  int j_index = it.get_group(0) + 1;    // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1] = a_dev[l_row1-1+j_index-1 + (l_col1-1+i_index-1)*matrixRows];
  }
}

template <typename T>
void sycl_copy_a_tmp1_FromC(T *a_dev, T *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, intptr_t my_stream){
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;

  sycl::range<1> global_range = sycl::range<1>(nb*nb);
  sycl::range<1> local_range  = sycl::range<1>(nb);
  
  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  queue.parallel_for(
      sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> it) {
        sycl_copy_a_tmp1_kernel(a_dev, tmp1_dev, l_row1, l_col1,
                                       matrixRows, nb, it);
      });
  queue.wait_and_throw();

}

extern "C" void sycl_copy_double_a_tmp1_FromC(double *a_dev, double *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, intptr_t my_stream){
  sycl_copy_a_tmp1_FromC(a_dev, tmp1_dev, l_row1_in, l_col1_in, matrixRows_in, nb_in, my_stream);
}

extern "C" void sycl_copy_float_a_tmp1_FromC(float *a_dev, float *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, intptr_t my_stream){
  sycl_copy_a_tmp1_FromC(a_dev, tmp1_dev, l_row1_in, l_col1_in, matrixRows_in, nb_in, my_stream);
}

extern "C" void sycl_copy_double_complex_a_tmp1_FromC(std::complex<double> *a_dev, std::complex<double> *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, intptr_t my_stream){
  sycl_copy_a_tmp1_FromC(a_dev, tmp1_dev, l_row1_in, l_col1_in, matrixRows_in, nb_in, my_stream);
}

extern "C" void sycl_copy_float_complex_a_tmp1_FromC(std::complex<float> *a_dev, std::complex<float> *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, intptr_t my_stream){
  sycl_copy_a_tmp1_FromC(a_dev, tmp1_dev, l_row1_in, l_col1_in, matrixRows_in, nb_in, my_stream);
}