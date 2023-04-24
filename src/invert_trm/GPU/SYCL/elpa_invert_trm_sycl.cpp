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
//    This file was written by A. Marek, MPCDF

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

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

void sycl_copy_double_a_tmat2_kernel(double *a_dev, double *tmat2_dev, const int nblk, const int matrixRows, const int l_colx, const int l_row1,
                                     sycl::nd_item<3> item_ct1){

  int nb_index = item_ct1.get_local_id(2) + 1; // range 1..nb
  int l_col_index = item_ct1.get_group(2) + 1; // range 1..l_colx-l_cols-1

  tmat2_dev[nb_index-1 + (l_colx-1 + l_col_index -1) * nblk] = a_dev[l_row1-1 + nb_index-1 + (l_colx-1 + l_col_index -1)  * matrixRows];

}

extern "C" void sycl_copy_double_a_tmat2_FromC(double *a_dev, double *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, intptr_t my_stream){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;
  int nb     = *nb_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::range<3> blocks = sycl::range<3>(1, 1, l_cols - l_colx + 1);
  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);

#ifdef WITH_GPU_STREAMS
  sycl_copy_double_a_tmat2_kernel<<<blocks,threadsPerBlock,0,streamId>>>(a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#else
  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_double_a_tmat2_kernel(a_dev, tmat2_dev, nblk, matrixRows,
                                        l_colx, l_row1, item_ct1);
      });
#endif
  /*
  DPCT1010:1: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_float_a_tmat2_kernel(float *a_dev, float *tmat2_dev, const int nblk, const int matrixRows, const int l_colx, const int l_row1,
                                    sycl::nd_item<3> item_ct1){

  int nb_index = item_ct1.get_local_id(2) + 1; // range 1..nb
  int l_col_index = item_ct1.get_group(2) + 1; // range 1..l_colx-l_cols-1

  tmat2_dev[nb_index-1 + (l_colx-1 + l_col_index -1) * nblk] = a_dev[l_row1-1 + nb_index-1 + (l_colx-1 + l_col_index -1)  * matrixRows];
}

extern "C" void sycl_copy_float_a_tmat2_FromC(float *a_dev, float *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, intptr_t my_stream){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;
  int nb     = *nb_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::range<3> blocks = sycl::range<3>(1, 1, l_cols - l_colx + 1);
  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);

#ifdef WITH_GPU_STREAMS
  sycl_copy_float_a_tmat2_kernel<<<blocks,threadsPerBlock,0,streamId>>>(a_dev, tmat2_dev, nblk, matrixRows, l_colx, l_row1);
#else
  /*
  DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_float_a_tmat2_kernel(a_dev, tmat2_dev, nblk, matrixRows,
                                       l_colx, l_row1, item_ct1);
      });
#endif
  /*
  DPCT1010:4: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_double_complex_a_tmat2_kernel(sycl::double2 *a_dev,
                                             sycl::double2 *tmat2_dev,
                                             const int nblk,
                                             const int matrixRows,
                                             const int l_colx, const int l_row1,
                                             sycl::nd_item<3> item_ct1) {

  int nb_index = item_ct1.get_local_id(2) + 1; // range 1..nb
  int l_col_index = item_ct1.get_group(2) + 1; // range 1..l_colx-l_cols-1

  tmat2_dev[nb_index-1 + (l_colx-1 + l_col_index -1) * nblk] = a_dev[l_row1-1 + nb_index-1 + (l_colx-1 + l_col_index -1)  * matrixRows];

}

extern "C" void sycl_copy_double_complex_a_tmat2_FromC(double _Complex *a_dev, double _Complex *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, intptr_t my_stream){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;
  int nb     = *nb_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::double2 *a_casted = (sycl::double2 *)a_dev;
  sycl::double2 *tmat2_casted = (sycl::double2 *)tmat2_dev;

  sycl::range<3> blocks = sycl::range<3>(1, 1, l_cols - l_colx + 1);
  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);

#ifdef WITH_GPU_STREAMS
  sycl_copy_double_complex_a_tmat2_kernel<<<blocks,threadsPerBlock,0,streamId>>>(a_casted, tmat2_casted, nblk, matrixRows, l_colx, l_row1);
#else
  /*
  DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_double_complex_a_tmat2_kernel(
            a_casted, tmat2_casted, nblk, matrixRows, l_colx, l_row1, item_ct1);
      });
#endif
  /*
  DPCT1010:7: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_float_complex_a_tmat2_kernel(sycl::float2 *a_dev,
                                            sycl::float2 *tmat2_dev,
                                            const int nblk,
                                            const int matrixRows,
                                            const int l_colx, const int l_row1,
                                            sycl::nd_item<3> item_ct1) {

  int nb_index = item_ct1.get_local_id(2) + 1; // range 1..nb
  int l_col_index = item_ct1.get_group(2) + 1; // range 1..l_colx-l_cols-1

  tmat2_dev[nb_index-1 + (l_colx-1 + l_col_index -1) * nblk] = a_dev[l_row1-1 + nb_index-1 + (l_colx-1 + l_col_index -1)  * matrixRows];

}

extern "C" void sycl_copy_float_complex_a_tmat2_FromC(float _Complex *a_dev, float _Complex *tmat2_dev, int *nblk_in, int *matrixRows_in, int *l_cols_in, int *l_colx_in, int *l_row1_in, int *nb_in, intptr_t my_stream){
  int nblk = *nblk_in;   
  int matrixRows = *matrixRows_in;
  int l_cols = *l_cols_in;
  int l_colx = *l_colx_in;
  int l_row1 = *l_row1_in;
  int nb     = *nb_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::float2 *a_casted = (sycl::float2 *)a_dev;
  sycl::float2 *tmat2_casted = (sycl::float2 *)tmat2_dev;

  sycl::range<3> blocks = sycl::range<3>(1, 1, l_cols - l_colx + 1);
  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);

#ifdef WITH_GPU_STREAMS
  sycl_copy_float_complex_a_tmat2_kernel<<<blocks,threadsPerBlock,0,streamId>>>(a_casted, tmat2_casted, nblk, matrixRows, l_colx, l_row1);
#else
  /*
  DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_float_complex_a_tmat2_kernel(
            a_casted, tmat2_casted, nblk, matrixRows, l_colx, l_row1, item_ct1);
      });
#endif
  /*
  DPCT1010:10: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_double_tmp2_tmat2_kernel(double *tmp2_dev, double *tmat2_dev, const int nblk, const int l_col1,
                                        sycl::nd_item<3> item_ct1){

  int nb_index = item_ct1.get_local_id(2) + 1; // range 1..nb
  int l_col_index = item_ct1.get_group(2) + 1; // range 1..nb

  tmat2_dev[nb_index-1 + (l_col1-1 + l_col_index -1)*nblk] = tmp2_dev[nb_index-1 + (1 -1 + l_col_index-1)  * nblk];

}

extern "C" void sycl_copy_double_tmp2_tmat2_FromC(double *tmp2_dev, double *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, intptr_t my_stream){
  int nblk   = *nblk_in;   
  int l_col1 = *l_col1_in;
  int nb     = *nb_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::range<3> blocks = sycl::range<3>(1, 1, nb);
  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);

#ifdef WITH_GPU_STREAMS
  sycl_copy_double_tmp2_tmat2_kernel<<<blocks,threadsPerBlock,0,streamId>>>(tmp2_dev, tmat2_dev, nblk, l_col1);
#else
  /*
  DPCT1049:12: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_double_tmp2_tmat2_kernel(tmp2_dev, tmat2_dev, nblk, l_col1,
                                           item_ct1);
      });
#endif
  /*
  DPCT1010:13: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}


void sycl_copy_float_tmp2_tmat2_kernel(float *tmp2_dev, float *tmat2_dev, const int nblk, const int l_col1,
                                       sycl::nd_item<3> item_ct1){

  int nb_index = item_ct1.get_local_id(2) + 1; // range 1..nb
  int l_col_index = item_ct1.get_group(2) + 1; // range 1..nb

  tmat2_dev[nb_index-1 + (l_col1-1 + l_col_index -1)*nblk] = tmp2_dev[nb_index-1 + (1 -1 + l_col_index-1)  * nblk];

}

extern "C" void sycl_copy_float_tmp2_tmat2_FromC(float *tmp2_dev, float *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, intptr_t my_stream){
  int nblk   = *nblk_in;   
  int l_col1 = *l_col1_in;
  int nb     = *nb_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::range<3> blocks = sycl::range<3>(1, 1, nb);
  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);

#ifdef WITH_GPU_STREAMS
  sycl_copy_float_tmp2_tmat2_kernel<<<blocks,threadsPerBlock,0,streamId>>>(tmp2_dev, tmat2_dev, nblk, l_col1);
#else
  /*
  DPCT1049:15: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_float_tmp2_tmat2_kernel(tmp2_dev, tmat2_dev, nblk, l_col1,
                                          item_ct1);
      });
#endif
  /*
  DPCT1010:16: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_double_complex_tmp2_tmat2_kernel(sycl::double2 *tmp2_dev,
                                                sycl::double2 *tmat2_dev,
                                                const int nblk,
                                                const int l_col1,
                                                sycl::nd_item<3> item_ct1) {

  int nb_index = item_ct1.get_local_id(2) + 1; // range 1..nb
  int l_col_index = item_ct1.get_group(2) + 1; // range 1..nb

  tmat2_dev[nb_index-1 + (l_col1-1 + l_col_index -1)*nblk] = tmp2_dev[nb_index-1 + (1 -1 + l_col_index-1)  * nblk];

}

extern "C" void sycl_copy_double_complex_tmp2_tmat2_FromC(double _Complex *tmp2_dev, double _Complex *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, intptr_t my_stream){
  int nblk   = *nblk_in;   
  int l_col1 = *l_col1_in;
  int nb     = *nb_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::double2 *tmp2_casted = (sycl::double2 *)tmp2_dev;
  sycl::double2 *tmat2_casted = (sycl::double2 *)tmat2_dev;

  sycl::range<3> blocks = sycl::range<3>(1, 1, nb);
  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);

#ifdef WITH_GPU_STREAMS
  sycl_copy_double_complex_tmp2_tmat2_kernel<<<blocks,threadsPerBlock,0,streamId>>>(tmp2_casted, tmat2_casted, nblk, l_col1);
#else
  /*
  DPCT1049:18: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_double_complex_tmp2_tmat2_kernel(tmp2_casted, tmat2_casted,
                                                   nblk, l_col1, item_ct1);
      });
#endif
  /*
  DPCT1010:19: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_float_complex_tmp2_tmat2_kernel(sycl::float2 *tmp2_dev,
                                               sycl::float2 *tmat2_dev,
                                               const int nblk, const int l_col1,
                                               sycl::nd_item<3> item_ct1) {

  int nb_index = item_ct1.get_local_id(2) + 1; // range 1..nb
  int l_col_index = item_ct1.get_group(2) + 1; // range 1..nb

  tmat2_dev[nb_index-1 + (l_col1-1 + l_col_index -1)*nblk] = tmp2_dev[nb_index-1 + (1 -1 + l_col_index-1)  * nblk];

}

extern "C" void sycl_copy_float_complex_tmp2_tmat2_FromC(float _Complex *tmp2_dev, float _Complex *tmat2_dev, int *nblk_in, int *l_col1_in, int *nb_in, intptr_t my_stream){
  int nblk   = *nblk_in;   
  int l_col1 = *l_col1_in;
  int nb     = *nb_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::float2 *tmp2_casted = (sycl::float2 *)tmp2_dev;
  sycl::float2 *tmat2_casted = (sycl::float2 *)tmat2_dev;

  sycl::range<3> blocks = sycl::range<3>(1, 1, nb);
  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);

#ifdef WITH_GPU_STREAMS
  sycl_copy_float_complex_tmp2_tmat2_kernel<<<blocks,threadsPerBlock,0,streamId>>>(tmp2_casted, tmat2_casted, nblk, l_col1);
#else
  /*
  DPCT1049:21: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_float_complex_tmp2_tmat2_kernel(tmp2_casted, tmat2_casted,
                                                  nblk, l_col1, item_ct1);
      });
#endif
  /*
  DPCT1010:22: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_double_a_tmat1_kernel(double *a_dev, double *tmat1_dev, const int l_rows, const int matrixRows, const int l_col1, const int nb, const int l_row1,
                                     sycl::nd_item<3> item_ct1){

  int nb_index = item_ct1.get_local_id(2) + 1;  // range 1..nb
  int l_row1_index = item_ct1.get_group(2) + 1; // we need l_row1-1 blocks

  tmat1_dev[l_row1_index-1 + (nb_index-1)*l_rows] = a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1 ) * matrixRows];
  a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1)*matrixRows] = 0;

}

extern "C" void sycl_copy_double_a_tmat1_FromC(double *a_dev, double *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, intptr_t my_stream){
  int l_rows = *l_rows_in;   
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);
  sycl::range<3> blocks = sycl::range<3>(1, 1, l_row1 - 1);

#ifdef WITH_GPU_STREAMS
  sycl_copy_double_a_tmat1_kernel<<<blocks,threadsPerBlock,0,streamId>>>(a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#else
  /*
  DPCT1049:24: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_double_a_tmat1_kernel(a_dev, tmat1_dev, l_rows, matrixRows,
                                        l_col1, nb, l_row1, item_ct1);
      });
#endif
  /*
  DPCT1010:25: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_float_a_tmat1_kernel(float *a_dev, float *tmat1_dev, const int l_rows, const int matrixRows, const int l_col1, const int nb, const int l_row1,
                                    sycl::nd_item<3> item_ct1){

  int nb_index = item_ct1.get_local_id(2) + 1;  // range 1..nb
  int l_row1_index = item_ct1.get_group(2) + 1; // we need l_row1-1 blocks

  tmat1_dev[l_row1_index-1 + (nb_index-1)*l_rows] = a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1 ) * matrixRows];
  a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1)*matrixRows] = 0;
}

extern "C" void sycl_copy_float_a_tmat1_FromC(float *a_dev, float *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, intptr_t my_stream){
  int l_rows = *l_rows_in;   
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);
  sycl::range<3> blocks = sycl::range<3>(1, 1, l_row1 - 1);
  //dim3 threadsPerBlock = dim3(1, 1, 1);
  //dim3 blocks = dim3(1,1,1);

#ifdef WITH_GPU_STREAMS
  sycl_copy_float_a_tmat1_kernel<<<blocks,threadsPerBlock,0,streamId>>>(a_dev, tmat1_dev, l_rows, matrixRows, l_col1, nb, l_row1);
#else
  /*
  DPCT1049:27: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_float_a_tmat1_kernel(a_dev, tmat1_dev, l_rows, matrixRows,
                                       l_col1, nb, l_row1, item_ct1);
      });
#endif
  /*
  DPCT1010:28: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_double_complex_a_tmat1_kernel(
    sycl::double2 *a_dev, sycl::double2 *tmat1_dev, const int l_rows,
    const int matrixRows, const int l_col1, const int nb, const int l_row1,
    sycl::double2 *zero_dev, sycl::nd_item<3> item_ct1) {

  int nb_index = item_ct1.get_local_id(2) + 1;  // range 1..nb
  int l_row1_index = item_ct1.get_group(2) + 1; // we need l_row1-1 blocks

  tmat1_dev[l_row1_index-1 + (nb_index-1)*l_rows] = a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1 ) * matrixRows];
  a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1)*matrixRows] = zero_dev[0];
}

extern "C" void sycl_copy_double_complex_a_tmat1_FromC(double _Complex *a_dev, double _Complex *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, double _Complex *ZERO, intptr_t my_stream){
  int l_rows = *l_rows_in;   
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::double2 *a_casted = (sycl::double2 *)a_dev;
  sycl::double2 *tmat1_casted = (sycl::double2 *)tmat1_dev;
  sycl::double2 *zero_casted = (sycl::double2 *)ZERO;

  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);
  sycl::range<3> blocks = sycl::range<3>(1, 1, l_row1 - 1);

#ifdef WITH_GPU_STREAMS
  sycl_copy_double_complex_a_tmat1_kernel<<<blocks,threadsPerBlock,0,streamId>>>(a_casted, tmat1_casted, l_rows, matrixRows, l_col1, nb, l_row1, zero_casted);
#else
  /*
  DPCT1049:30: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_double_complex_a_tmat1_kernel(a_casted, tmat1_casted, l_rows,
                                                matrixRows, l_col1, nb, l_row1,
                                                zero_casted, item_ct1);
      });
#endif
  /*
  DPCT1010:31: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_float_complex_a_tmat1_kernel(
    sycl::float2 *a_dev, sycl::float2 *tmat1_dev, const int l_rows,
    const int matrixRows, const int l_col1, const int nb, const int l_row1,
    sycl::float2 *zero_dev, sycl::nd_item<3> item_ct1) {

  int nb_index = item_ct1.get_local_id(2) + 1;  // range 1..nb
  int l_row1_index = item_ct1.get_group(2) + 1; // we need l_row1-1 blocks

  tmat1_dev[l_row1_index-1 + (nb_index-1)*l_rows] = a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1 ) * matrixRows];
  a_dev[l_row1_index-1 + (l_col1-1 + nb_index-1)*matrixRows] = zero_dev[0];
}

extern "C" void sycl_copy_float_complex_a_tmat1_FromC(float _Complex *a_dev, float _Complex *tmat1_dev, int *l_rows_in, int *matrixRows_in, int *nb_in, int *l_row1_in, int *l_col1_in, float _Complex *ZERO, intptr_t my_stream){
  int l_rows = *l_rows_in;   
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::float2 *a_casted = (sycl::float2 *)a_dev;
  sycl::float2 *tmat1_casted = (sycl::float2 *)tmat1_dev;
  sycl::float2 *zero_casted = (sycl::float2 *)ZERO;

  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);
  sycl::range<3> blocks = sycl::range<3>(1, 1, l_row1 - 1);

#ifdef WITH_GPU_STREAMS
  sycl_copy_float_complex_a_tmat1_kernel<<<blocks,threadsPerBlock,0,streamId>>>(a_casted, tmat1_casted, l_rows, matrixRows, l_col1, nb, l_row1, zero_casted);
#else
  /*
  DPCT1049:33: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_float_complex_a_tmat1_kernel(a_casted, tmat1_casted, l_rows,
                                               matrixRows, l_col1, nb, l_row1,
                                               zero_casted, item_ct1);
      });
#endif
  /*
  DPCT1010:34: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_double_tmp1_tmp2_kernel(double *tmp1_dev, double *tmp2_dev, const int nblk, const int nb,
                                       sycl::nd_item<3> item_ct1){

  int i_index = item_ct1.get_local_id(2) + 1; // range 1..nb
  int j_index = item_ct1.get_group(2) + 1;    // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp2_dev[1-1 + j_index-1 + (i_index-1)*nblk] = tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1];
  }
}


extern "C" void sycl_copy_double_tmp1_tmp2_FromC(double *tmp1_dev, double *tmp2_dev, int *nblk_in, int *nb_in, intptr_t my_stream){
  int nblk = *nblk_in;
  int nb = *nb_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);
  sycl::range<3> blocks = sycl::range<3>(1, 1, nb);

#ifdef WITH_GPU_STREAMS
  sycl_copy_double_tmp1_tmp2_kernel<<<blocks,threadsPerBlock,0,streamId>>>(tmp1_dev, tmp2_dev, nblk, nb);
#else
  /*
  DPCT1049:36: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_double_tmp1_tmp2_kernel(tmp1_dev, tmp2_dev, nblk, nb,
                                          item_ct1);
      });
#endif
  /*
  DPCT1010:37: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_float_tmp1_tmp2_kernel(float *tmp1_dev, float *tmp2_dev, const int nblk, const int nb,
                                      sycl::nd_item<3> item_ct1){

  int i_index = item_ct1.get_local_id(2) + 1; // range 1..nb
  int j_index = item_ct1.get_group(2) + 1;    // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp2_dev[1-1 + j_index-1 + (i_index-1)*nblk] = tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1];
  }
}


extern "C" void sycl_copy_float_tmp1_tmp2_FromC(float *tmp1_dev, float *tmp2_dev, int *nblk_in, int *nb_in, intptr_t my_stream){
  int nblk = *nblk_in;
  int nb = *nb_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);
  sycl::range<3> blocks = sycl::range<3>(1, 1, nb);

#ifdef WITH_GPU_STREAMS
  sycl_copy_float_tmp1_tmp2_kernel<<<blocks,threadsPerBlock,0,streamId>>>(tmp1_dev, tmp2_dev, nblk, nb);
#else
  /*
  DPCT1049:39: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_float_tmp1_tmp2_kernel(tmp1_dev, tmp2_dev, nblk, nb,
                                         item_ct1);
      });
#endif
  /*
  DPCT1010:40: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_double_complex_tmp1_tmp2_kernel(sycl::double2 *tmp1_dev,
                                               sycl::double2 *tmp2_dev,
                                               const int nblk, const int nb,
                                               sycl::nd_item<3> item_ct1) {

  int i_index = item_ct1.get_local_id(2) + 1; // range 1..nb
  int j_index = item_ct1.get_group(2) + 1;    // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp2_dev[1-1 + j_index-1 + (i_index-1)*nblk] = tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1];
  }
}


extern "C" void sycl_copy_double_complex_tmp1_tmp2_FromC(double _Complex *tmp1_dev, double _Complex *tmp2_dev, int *nblk_in, int *nb_in, intptr_t my_stream){
  int nblk = *nblk_in;
  int nb = *nb_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);
  sycl::range<3> blocks = sycl::range<3>(1, 1, nb);

  sycl::double2 *tmp1_casted = (sycl::double2 *)tmp1_dev;
  sycl::double2 *tmp2_casted = (sycl::double2 *)tmp2_dev;

#ifdef WITH_GPU_STREAMS
  sycl_copy_double_complex_tmp1_tmp2_kernel<<<blocks,threadsPerBlock,0,streamId>>>(tmp1_casted, tmp2_casted, nblk, nb);
#else
  /*
  DPCT1049:42: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_double_complex_tmp1_tmp2_kernel(tmp1_casted, tmp2_casted,
                                                  nblk, nb, item_ct1);
      });
#endif
  /*
  DPCT1010:43: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_float_complex_tmp1_tmp2_kernel(sycl::float2 *tmp1_dev,
                                              sycl::float2 *tmp2_dev,
                                              const int nblk, const int nb,
                                              sycl::nd_item<3> item_ct1) {

  int i_index = item_ct1.get_local_id(2) + 1; // range 1..nb
  int j_index = item_ct1.get_group(2) + 1;    // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp2_dev[1-1 + j_index-1 + (i_index-1)*nblk] = tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1];
  }
}


extern "C" void sycl_copy_float_complex_tmp1_tmp2_FromC(float _Complex *tmp1_dev, float _Complex *tmp2_dev, int *nblk_in, int *nb_in, intptr_t my_stream){
  int nblk = *nblk_in;
  int nb = *nb_in;

#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);
  sycl::range<3> blocks = sycl::range<3>(1, 1, nb);

  sycl::float2 *tmp1_casted = (sycl::float2 *)tmp1_dev;
  sycl::float2 *tmp2_casted = (sycl::float2 *)tmp2_dev;

#ifdef WITH_GPU_STREAMS
  sycl_copy_float_complex_tmp1_tmp2_kernel<<<blocks,threadsPerBlock,0,streamId>>>(tmp1_casted, tmp2_casted, nblk, nb);
#else
  /*
  DPCT1049:45: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_float_complex_tmp1_tmp2_kernel(tmp1_casted, tmp2_casted, nblk,
                                                 nb, item_ct1);
      });
#endif
  /*
  DPCT1010:46: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_double_a_tmp1_kernel(double *a_dev, double *tmp1_dev, const int l_row1, const int l_col1, const int matrixRows, const int nb,
                                    sycl::nd_item<3> item_ct1){

  int i_index = item_ct1.get_local_id(2) + 1; // range 1..nb
  int j_index = item_ct1.get_group(2) + 1;    // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1] = a_dev[l_row1-1+j_index-1 + (l_col1-1+i_index-1)*matrixRows];
  }
}

extern "C" void sycl_copy_double_a_tmp1_FromC(double *a_dev, double *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, intptr_t my_stream){
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);
  sycl::range<3> blocks = sycl::range<3>(1, 1, nb);

#ifdef WITH_GPU_STREAMS
  sycl_copy_double_a_tmp1_kernel<<<blocks,threadsPerBlock,0,streamId>>>(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb);
#else
  /*
  DPCT1049:48: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_double_a_tmp1_kernel(a_dev, tmp1_dev, l_row1, l_col1,
                                       matrixRows, nb, item_ct1);
      });
#endif
  /*
  DPCT1010:49: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_float_a_tmp1_kernel(float *a_dev, float *tmp1_dev, const int l_row1, const int l_col1, const int matrixRows, const int nb,
                                   sycl::nd_item<3> item_ct1){

  int i_index = item_ct1.get_local_id(2) + 1; // range 1..nb
  int j_index = item_ct1.get_group(2) + 1;    // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1] = a_dev[l_row1-1+j_index-1 + (l_col1-1+i_index-1)*matrixRows];
  }

}

extern "C" void sycl_copy_float_a_tmp1_FromC(float *a_dev, float *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, intptr_t my_stream){
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);
  sycl::range<3> blocks = sycl::range<3>(1, 1, nb);

#ifdef WITH_GPU_STREAMS
  sycl_copy_float_a_tmp1_kernel<<<blocks,threadsPerBlock,0,streamId>>>(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb);
#else
  /*
  DPCT1049:51: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_float_a_tmp1_kernel(a_dev, tmp1_dev, l_row1, l_col1,
                                      matrixRows, nb, item_ct1);
      });
#endif
  /*
  DPCT1010:52: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_double_complex_a_tmp1_kernel(sycl::double2 *a_dev,
                                            sycl::double2 *tmp1_dev,
                                            const int l_row1, const int l_col1,
                                            const int matrixRows, const int nb,
                                            sycl::nd_item<3> item_ct1) {

  int i_index = item_ct1.get_local_id(2) + 1; // range 1..nb
  int j_index = item_ct1.get_group(2) + 1;    // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1] = a_dev[l_row1-1+j_index-1 + (l_col1-1+i_index-1)*matrixRows];
  }
}

extern "C" void sycl_copy_double_complex_a_tmp1_FromC(double _Complex *a_dev, double _Complex *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, intptr_t my_stream){
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::double2 *a_casted = (sycl::double2 *)a_dev;
  sycl::double2 *tmp1_casted = (sycl::double2 *)tmp1_dev;

  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);
  sycl::range<3> blocks = sycl::range<3>(1, 1, nb);

#ifdef WITH_GPU_STREAMS
  sycl_copy_double_complex_a_tmp1_kernel<<<blocks,threadsPerBlock,0,streamId>>>(a_casted, tmp1_casted, l_row1, l_col1, matrixRows, nb);
#else
  /*
  DPCT1049:54: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_double_complex_a_tmp1_kernel(
            a_casted, tmp1_casted, l_row1, l_col1, matrixRows, nb, item_ct1);
      });
#endif
  /*
  DPCT1010:55: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

void sycl_copy_float_complex_a_tmp1_kernel(sycl::float2 *a_dev,
                                           sycl::float2 *tmp1_dev,
                                           const int l_row1, const int l_col1,
                                           const int matrixRows, const int nb,
                                           sycl::nd_item<3> item_ct1) {

  int i_index = item_ct1.get_local_id(2) + 1; // range 1..nb
  int j_index = item_ct1.get_group(2) + 1;    // range 1..nb (should be 1..i)

  if (j_index < i_index+1) {
    tmp1_dev[(i_index*(i_index+1)-2*i_index)/2 +1 -1 + j_index-1] = a_dev[l_row1-1+j_index-1 + (l_col1-1+i_index-1)*matrixRows];
  }
}

extern "C" void sycl_copy_float_complex_a_tmp1_FromC(float _Complex *a_dev, float _Complex *tmp1_dev, int *l_row1_in, int *l_col1_in, int *matrixRows_in, int *nb_in, intptr_t my_stream){
  int l_row1 = *l_row1_in;
  int l_col1 = *l_col1_in;
  int matrixRows = *matrixRows_in;
  int nb = *nb_in;
#ifdef WITH_GPU_STREAMS
  syclStream_t streamId = *((syclStream_t*)my_stream);
#endif

  sycl::float2 *a_casted = (sycl::float2 *)a_dev;
  sycl::float2 *tmp1_casted = (sycl::float2 *)tmp1_dev;

  sycl::range<3> threadsPerBlock = sycl::range<3>(1, 1, nb);
  sycl::range<3> blocks = sycl::range<3>(1, 1, nb);

#ifdef WITH_GPU_STREAMS
  sycl_copy_float_complex_a_tmp1_kernel<<<blocks,threadsPerBlock,0,streamId>>>(a_casted, tmp1_casted, l_row1, l_col1, matrixRows, nb);
#else
  /*
  DPCT1049:57: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_float_complex_a_tmp1_kernel(a_casted, tmp1_casted, l_row1,
                                              l_col1, matrixRows, nb, item_ct1);
      });
#endif
  /*
  DPCT1010:58: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  int cuerr = 0;
}

