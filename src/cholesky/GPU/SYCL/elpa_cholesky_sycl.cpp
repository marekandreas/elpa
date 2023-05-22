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
#include <type_traits>

#include <CL/sycl.hpp>
#include "src/GPU/SYCL/syclCommon.hpp"

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

template <typename T>
void sycl_copy_a_tmatc_kernel(T *a_dev, T *tmatc_dev, const int l_cols, const int matrixRows, const int l_colx, const int l_row1, const int nblk, const sycl::nd_item<1> &it){

  int ii_index = it.get_local_id(0) + 1; // range 1..nblk
  int jj_index = it.get_group(0) + 1;    // range 1..l_cols-l_colx+1
  
  if constexpr (std::is_same<T, std::complex<float>>::value || std::is_same<T, std::complex<double>>::value) {
    tmatc_dev[l_colx-1+jj_index-1+(ii_index-1)*l_cols] = std::conj(a_dev[l_row1-1+ii_index-1 + (l_colx-1+jj_index-1)*matrixRows]);
  } 
  else {
    tmatc_dev[l_colx-1+jj_index-1+(ii_index-1)*l_cols] =           a_dev[l_row1-1+ii_index-1 + (l_colx-1+jj_index-1)*matrixRows];
  }
}

template <typename T>
void sycl_copy_a_tmatc_FromC(T *a_dev, T *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, intptr_t my_stream){

  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;

  sycl::range<1> global_range = sycl::range<1>(nblk*(l_cols - l_colx + 1));
  sycl::range<1> local_range  = sycl::range<1>(nblk);
  
  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  queue.parallel_for(
    sycl::nd_range<1>(global_range, local_range),
    [=](sycl::nd_item<1> it) {
      sycl_copy_a_tmatc_kernel(a_dev, tmatc_dev, l_cols, matrixRows, l_colx, l_row1, nblk, it);
    });
  queue.wait_and_throw();

}

extern "C" void sycl_copy_double_a_tmatc_FromC(double *a_dev, double *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, intptr_t my_stream){
  sycl_copy_a_tmatc_FromC(a_dev, tmatc_dev, nblk_in, matrixRows_in, l_cols_in, l_colx_in, l_row1_in, my_stream);
}

extern "C" void sycl_copy_float_a_tmatc_FromC(float *a_dev, float *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, intptr_t my_stream){
  sycl_copy_a_tmatc_FromC(a_dev, tmatc_dev, nblk_in, matrixRows_in, l_cols_in, l_colx_in, l_row1_in, my_stream);
}

extern "C" void sycl_copy_double_complex_a_tmatc_FromC(std::complex<double> *a_dev, std::complex<double> *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, intptr_t my_stream){
  sycl_copy_a_tmatc_FromC(a_dev, tmatc_dev, nblk_in, matrixRows_in, l_cols_in, l_colx_in, l_row1_in, my_stream);
}

extern "C" void sycl_copy_float_complex_a_tmatc_FromC(std::complex<float> *a_dev, std::complex<float> *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, intptr_t my_stream){
  sycl_copy_a_tmatc_FromC(a_dev, tmatc_dev, nblk_in, matrixRows_in, l_cols_in, l_colx_in, l_row1_in, my_stream);
}
