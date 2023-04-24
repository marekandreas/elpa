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

#include <CL/sycl.hpp>
//#include <dpct/dpct.hpp>
#include "/mpcdf/soft/SLE_15/packages/x86_64/intel_oneapi/2022.3/dpcpp-ct/2022.2.1/include/dpct/dpct.hpp"
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
//#include <complex.h>
#include <stdint.h>
#include "config-f90.h"
#include <complex>

#include "src/GPU/SYCL/syclCommon.hpp"

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

void sycl_copy_double_a_tmatc_kernel(double *a_dev, double *tmatc_dev, const int l_cols, const int matrixRows, const int l_colx, const int l_row1, const int nblk,
                                     const sycl::nd_item<3> &item_ct1){

  int ii_index = item_ct1.get_local_id(2) + 1; // range 1..nblk
  int jj_index = item_ct1.get_group(2) + 1;    // range 1..l_cols-l_colx+1
  tmatc_dev[l_colx-1+jj_index-1+(ii_index-1)*l_cols] = a_dev[l_row1-1+ii_index-1 + (l_colx-1+jj_index-1)*matrixRows];
}

extern "C" void sycl_copy_double_a_tmatc_FromC(double *a_dev, double *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, intptr_t my_stream){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;

  printf("Here: sycl_copy_double_a_tmatc_FromC\n"); // PETERDEBUG

  sycl::range<3> blocks = sycl::range<3>(1, 1, l_cols - l_colx + 1);
  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nblk);
  
  auto device = elpa::gpu::sycl::getDevice();
  auto &queue = elpa::gpu::sycl::getQueue();

  printf( "sycl_copy_double_a_tmatc_FromC: number of available threads: %d \n", device.get_info<sycl::info::device::max_compute_units>() ); // PETERDEBUG

  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */

//  queue.parallel_for(
    dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_double_a_tmatc_kernel(a_dev, tmatc_dev, l_cols, matrixRows,
                                        l_colx, l_row1, nblk, item_ct1);
      });
//  queue.wait_and_throw();


}

extern "C" void sycl_copy_float_a_tmatc_FromC(float *a_dev, float *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, intptr_t my_stream){printf("not implemented\n");}
extern "C" void sycl_copy_double_complex_a_tmatc_FromC(double _Complex *a_dev, double _Complex *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, intptr_t my_stream){printf("not implemented\n");}
extern "C" void sycl_copy_float_complex_a_tmatc_FromC(float _Complex *a_dev, float _Complex *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, intptr_t my_stream){printf("not implemented\n");}

/*
void sycl_copy_float_a_tmatc_kernel(float *a_dev, float *tmatc_dev, const int l_cols, const int matrixRows, const int l_colx, const int l_row1, const int nblk,
                                    const sycl::nd_item<3> &item_ct1){

  int ii_index = item_ct1.get_local_id(2) + 1; // range 1..nblk
  int jj_index = item_ct1.get_group(2) + 1;    // range 1..l_cols-l_colx+1
  tmatc_dev[l_colx-1+jj_index-1+(ii_index-1)*l_cols] = a_dev[l_row1-1+ii_index-1 + (l_colx-1+jj_index-1)*matrixRows];
}

extern "C" void sycl_copy_float_a_tmatc_FromC(float *a_dev, float *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, intptr_t my_stream){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::range<3> blocks = sycl::range<3>(1, 1, l_cols - l_colx + 1);
  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nblk);

#ifdef WITH_GPU_STREAMS
  sycl_copy_float_a_tmatc_kernel<<<blocks,threadsPerBlock,0,streamId>>>(a_dev, tmatc_dev, l_cols, matrixRows, l_colx, l_row1, nblk);
#else

  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_float_a_tmatc_kernel(a_dev, tmatc_dev, l_cols, matrixRows,
                                       l_colx, l_row1, nblk, item_ct1);
      });
#endif

  dpct::err0 cuerr = 0;
}

void sycl_copy_double_complex_a_tmatc_kernel(sycl::double2 *a_dev,
                                             sycl::double2 *tmatc_dev,
                                             const int l_cols,
                                             const int matrixRows,
                                             const int l_colx, const int l_row1,
                                             const sycl::nd_item<3> &item_ct1) {

  int ii_index = item_ct1.get_local_id(2) + 1; // range 1..nblk
  int jj_index = item_ct1.get_group(2) + 1;    // range 1..l_cols-l_colx+1
  tmatc_dev[l_colx - 1 + jj_index - 1 + (ii_index - 1) * l_cols] =
      dpct::conj<double>(a_dev[l_row1 - 1 + ii_index - 1 +
                               (l_colx - 1 + jj_index - 1) * matrixRows]);
}

extern "C" void sycl_copy_double_complex_a_tmatc_FromC(double _Complex *a_dev, double _Complex *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, intptr_t my_stream){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::range<3> blocks = sycl::range<3>(1, 1, l_cols - l_colx + 1);
  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nblk);

  sycl::double2 *a_casted = (sycl::double2 *)a_dev;
  sycl::double2 *tmatc_casted = (sycl::double2 *)tmatc_dev;

#ifdef WITH_GPU_STREAMS
  sycl_copy_double_complex_a_tmatc_kernel<<<blocks,threadsPerBlock,0,streamId>>>(a_casted, tmatc_casted, l_cols, matrixRows, l_colx, l_row1);
#else

  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_double_complex_a_tmatc_kernel(a_casted, tmatc_casted, l_cols,
                                                matrixRows, l_colx, l_row1,
                                                item_ct1);
      });
#endif

  dpct::err0 cuerr = 0;
}

void sycl_copy_float_complex_a_tmatc_kernel(sycl::float2 *a_dev,
                                            sycl::float2 *tmatc_dev,
                                            const int l_cols,
                                            const int matrixRows,
                                            const int l_colx, const int l_row1,
                                            const sycl::nd_item<3> &item_ct1) {

  int ii_index = item_ct1.get_local_id(2) + 1; // range 1..nblk
  int jj_index = item_ct1.get_group(2) + 1;    // range 1..l_cols-l_colx+1
  tmatc_dev[l_colx - 1 + jj_index - 1 + (ii_index - 1) * l_cols] =
      dpct::conj<float>(a_dev[l_row1 - 1 + ii_index - 1 +
                              (l_colx - 1 + jj_index - 1) * matrixRows]);
}

extern "C" void sycl_copy_float_complex_a_tmatc_FromC(float _Complex *a_dev, float _Complex *tmatc_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, intptr_t my_stream){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::range<3> blocks = sycl::range<3>(1, 1, l_cols - l_colx + 1);
  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nblk);

  sycl::float2 *a_casted = (sycl::float2 *)a_dev;
  sycl::float2 *tmatc_casted = (sycl::float2 *)tmatc_dev;

#ifdef WITH_GPU_STREAMS
  sycl_copy_float_complex_a_tmatc_kernel<<<blocks,threadsPerBlock,0,streamId>>>(a_casted, tmatc_casted, l_cols, matrixRows, l_colx, l_row1);
#else

  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_float_complex_a_tmatc_kernel(a_casted, tmatc_casted, l_cols,
                                               matrixRows, l_colx, l_row1,
                                               item_ct1);
      });
#endif

  dpct::err0 cuerr = 0;
}
*/