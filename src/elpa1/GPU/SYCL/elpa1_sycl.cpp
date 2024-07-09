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

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <complex.h>
#include <stdint.h>
#include <complex>

#include "config-f90.h"

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

void sycl_copy_real_part_to_q_double_complex_kernel(
    sycl::double2 *q, const double *q_real, const int matrixRows,
    const int l_rows, const int l_cols_nev, const sycl::nd_item<3> &item_ct1) {
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int col = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
              item_ct1.get_local_id(1);

    //if (row < l_rows && col < l_cols_nev) {
    //    int index = row * l_cols_nev + col;
    //    q[index].x = q_real[index]; 
    //    q[index].y = 0.0; 
    //}

    if (row < l_rows && col < l_cols_nev) {
        int index = row + matrixRows * col;
        q[index].x() = q_real[index];
        q[index].y() = 0.0;
    }
}

extern "C" void sycl_copy_real_part_to_q_double_complex_FromC(
    double _Complex *q_dev, double *q_real_dev, int *matrixRows_in,
    int *l_rows_in, int *l_cols_nev_in, dpct::queue_ptr my_stream) {
  int l_rows = *l_rows_in;
  int l_cols_nev = *l_cols_nev_in;
  int matrixRows = *matrixRows_in;

  sycl::double2 *q_casted = (sycl::double2 *)q_dev;

  sycl::range<3> threadsPerBlock(1, 32, 32);
  sycl::range<3> blocks(
      1, (l_cols_nev + threadsPerBlock[1] - 1) / threadsPerBlock[1],
      (l_rows + threadsPerBlock[2] - 1) / threadsPerBlock[2]);

#ifdef WITH_GPU_STREAMS
  sycl_copy_real_part_to_q_double_complex_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_casted, q_real_dev, matrixRows, l_rows, l_cols_nev);
#else
  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_real_part_to_q_double_complex_kernel(
            q_casted, q_real_dev, matrixRows, l_rows, l_cols_nev, item_ct1);
      });
#endif

  /*
  DPCT1010:14: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 cuerr = 0;
}

void sycl_copy_real_part_to_q_float_complex_kernel(
    sycl::float2 *q, const float *q_real, const int matrixRows,
    const int l_rows, const int l_cols_nev, const sycl::nd_item<3> &item_ct1) {
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int col = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
              item_ct1.get_local_id(1);

    if (row < l_rows && col < l_cols_nev) {
        int index = row + matrixRows * col;
        q[index].x() = q_real[index];
        q[index].y() = 0.0f;
    }
}

extern "C" void sycl_copy_real_part_to_q_float_complex_FromC(
    float _Complex *q_dev, float *q_real_dev, int *matrixRows_in,
    int *l_rows_in, int *l_cols_nev_in, dpct::queue_ptr my_stream) {
  int l_rows = *l_rows_in;
  int l_cols_nev = *l_cols_nev_in;
  int matrixRows = *matrixRows_in;

  sycl::float2 *q_casted = (sycl::float2 *)q_dev;

  sycl::range<3> threadsPerBlock(1, 32, 32);
  sycl::range<3> blocks(
      1, (l_cols_nev + threadsPerBlock[1] - 1) / threadsPerBlock[1],
      (l_rows + threadsPerBlock[2] - 1) / threadsPerBlock[2]);

#ifdef WITH_GPU_STREAMS
  sycl_copy_real_part_to_q_float_complex_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_casted, q_real_dev, matrixRows, l_rows, l_cols_nev);
#else
  /*
  DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_real_part_to_q_float_complex_kernel(
            q_casted, q_real_dev, matrixRows, l_rows, l_cols_nev, item_ct1);
      });
#endif

  /*
  DPCT1010:16: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 cuerr = 0;
}

void sycl_zero_skewsymmetric_q_double_kernel(double *q, const int matrixRows, const int matrixCols,
                                             const sycl::nd_item<3> &item_ct1) {
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int col = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
              item_ct1.get_local_id(1);

    if (row < matrixRows && col >= matrixCols && col < 2 * matrixCols) {
        int index = row + (matrixRows) * col;
        q[index] = 0.0;
    }
}

extern "C" void
sycl_zero_skewsymmetric_q_double_FromC(double *q_dev, int *matrixRows_in,
                                       int *matrixCols_in,
                                       dpct::queue_ptr my_stream) {
  int matrixCols = *matrixCols_in;
  int matrixRows = *matrixRows_in;

  sycl::range<3> threadsPerBlock(1, 32, 32);
  sycl::range<3> blocks(
      1, (matrixCols + threadsPerBlock[1] - 1) / threadsPerBlock[1],
      (matrixRows + threadsPerBlock[2] - 1) / threadsPerBlock[2]);

#ifdef WITH_GPU_STREAMS
  sycl_zero_skewsymmetric_q_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, matrixRows, matrixCols);
#else
  /*
  DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});

    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
        [=](sycl::nd_item<3> item_ct1) {
          sycl_zero_skewsymmetric_q_double_kernel(q_dev, matrixRows, matrixCols,
                                                  item_ct1);
        });
  }
#endif

  /*
  DPCT1010:18: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 cuerr = 0;
}

void sycl_zero_skewsymmetric_q_float_kernel(float *q, const int matrixRows, const int matrixCols,
                                            const sycl::nd_item<3> &item_ct1) {
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int col = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
              item_ct1.get_local_id(1);

    //if (row < matrixRows && col >= matrixCols && col < 2 * matrixCols) {
    //    int index = row * (2 * matrixCols) + col;
    //    q[index] = 0.0f;
    //}
    if (row < matrixRows && col >= matrixCols && col < 2 * matrixCols) {
        int index = row + (matrixRows) * col;
        q[index] = 0.0f;
    }
}

extern "C" void
sycl_zero_skewsymmetric_q_float_FromC(float *q_dev, int *matrixRows_in,
                                      int *matrixCols_in,
                                      dpct::queue_ptr my_stream) {
  int matrixCols = *matrixCols_in;
  int matrixRows = *matrixRows_in;

  sycl::range<3> threadsPerBlock(1, 32, 32);
  //dim3 blocks((matrixRows + threadsPerBlock.x - 1) / threadsPerBlock.x, (2 * matrixCols + threadsPerBlock.y - 1) / threadsPerBlock.y);
  sycl::range<3> blocks(
      1, (matrixCols + threadsPerBlock[1] - 1) / threadsPerBlock[1],
      (matrixRows + threadsPerBlock[2] - 1) / threadsPerBlock[2]);

#ifdef WITH_GPU_STREAMS
  sycl_zero_skewsymmetric_q_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, matrixRows, matrixCols);
#else
  /*
  DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_zero_skewsymmetric_q_float_kernel(q_dev, matrixRows, matrixCols,
                                               item_ct1);
      });
#endif

  /*
  DPCT1010:20: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 cuerr = 0;
}

void sycl_copy_skewsymmetric_second_half_q_double_minus_kernel(double *q, const int i, const int matrixRows, const int matrixCols,
                                                               const sycl::nd_item<3> &item_ct1) {


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

    int col = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);

    if (col >= matrixCols && col < 2 * matrixCols) {  // geht nivht
        int    index = (i-1) + matrixRows * (col);
        int indexLow = (i-1) + matrixRows * (col-matrixCols);
        q[index] = -q[indexLow];
        q[indexLow] = 0.0;
    }	

}

void sycl_copy_skewsymmetric_second_half_q_double_plus_kernel(double *q, const int i, const int matrixRows, const int matrixCols,
                                                              const sycl::nd_item<3> &item_ct1) {
    int col = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
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

extern "C" void sycl_copy_skewsymmetric_second_half_q_double_FromC(
    double *q_dev, int *i_in, int *matrixRows_in, int *matrixCols_in,
    int *negative_or_positive_in, dpct::queue_ptr my_stream) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int matrixCols = *matrixCols_in;
  int matrixRows = *matrixRows_in;
  int negative_or_positive = *negative_or_positive_in;
  int i = *i_in;

  sycl::range<3> threadsPerBlock(1, 1, 1024);
  sycl::range<3> blocks(
      1, 1, (2 * matrixCols + threadsPerBlock[2] - 1) / threadsPerBlock[2]);
  //dim3 threadsPerBlock(1);
  //dim3 blocks(1);

  if (negative_or_positive == 1) {
#ifdef WITH_GPU_STREAMS
    sycl_copy_skewsymmetric_second_half_q_double_plus_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, i, matrixRows, matrixCols);
#else
    /*
    DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp64});

    q_ct1.parallel_for(
        sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
        [=](sycl::nd_item<3> item_ct1) {
          sycl_copy_skewsymmetric_second_half_q_double_plus_kernel(
              q_dev, i, matrixRows, matrixCols, item_ct1);
        });
#endif
  } else {
#ifdef WITH_GPU_STREAMS
    sycl_copy_skewsymmetric_second_half_q_double_minus_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, i, matrixRows, matrixCols);
#else
    /*
    DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp64});

    q_ct1.parallel_for(
        sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
        [=](sycl::nd_item<3> item_ct1) {
          sycl_copy_skewsymmetric_second_half_q_double_minus_kernel(
              q_dev, i, matrixRows, matrixCols, item_ct1);
        });
#endif
  }

  /*
  DPCT1010:22: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 cuerr = 0;
}

void sycl_copy_skewsymmetric_second_half_q_float_minus_kernel(float *q, const int i, const int matrixRows, const int matrixCols,
                                                              const sycl::nd_item<3> &item_ct1) {
    int col = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);

    if (col >= matrixCols && col < 2 * matrixCols) {  // geht nivht
        int    index = (i-1) + matrixRows * (col);
        int indexLow = (i-1) + matrixRows * (col-matrixCols);
        q[index] = -q[indexLow];
        q[indexLow] = 0.0f;
    }	
}

void sycl_copy_skewsymmetric_second_half_q_float_plus_kernel(float *q, const int i, const int matrixRows, const int matrixCols,
                                                             const sycl::nd_item<3> &item_ct1) {
    int col = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);

    if (col >= matrixCols && col < 2 * matrixCols) {  // geht nivht
        int    index = (i-1) + matrixRows * (col);
        int indexLow = (i-1) + matrixRows * (col-matrixCols);
        q[index] = q[indexLow];
        q[indexLow] = 0.0f;
    }	
}

extern "C" void sycl_copy_skewsymmetric_second_half_q_float_FromC(
    float *q_dev, int *i_in, int *matrixRows_in, int *matrixCols_in,
    int *negative_or_positive_in, dpct::queue_ptr my_stream) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int matrixCols = *matrixCols_in;
  int matrixRows = *matrixRows_in;
  int negative_or_positive = *negative_or_positive_in;
  int i = *i_in;

  sycl::range<3> threadsPerBlock(1, 1, 1024);
  sycl::range<3> blocks(
      1, 1, (2 * matrixCols + threadsPerBlock[2] - 1) / threadsPerBlock[2]);

  if (negative_or_positive == 1) {
#ifdef WITH_GPU_STREAMS
    sycl_copy_skewsymmetric_second_half_q_float_plus_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, i, matrixRows, matrixCols);
#else
    /*
    DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q_ct1.parallel_for(
        sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
        [=](sycl::nd_item<3> item_ct1) {
          sycl_copy_skewsymmetric_second_half_q_float_plus_kernel(
              q_dev, i, matrixRows, matrixCols, item_ct1);
        });
#endif
  } else {
#ifdef WITH_GPU_STREAMS
    sycl_copy_skewsymmetric_second_half_q_float_minus_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, i, matrixRows, matrixCols);
#else
    /*
    DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q_ct1.parallel_for(
        sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
        [=](sycl::nd_item<3> item_ct1) {
          sycl_copy_skewsymmetric_second_half_q_float_minus_kernel(
              q_dev, i, matrixRows, matrixCols, item_ct1);
        });
#endif
  }

  /*
  DPCT1010:24: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 cuerr = 0;
}

void sycl_copy_skewsymmetric_first_half_q_double_minus_kernel(double *q, const int i, const int matrixRows, const int matrixCols,
                                                              const sycl::nd_item<3> &item_ct1) {
  int col = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  if (col < matrixCols) {
    int index = (i-1) + matrixRows * col;
    q[index] = -q[index];
  }
}

extern "C" void sycl_copy_skewsymmetric_first_half_q_double_FromC(
    double *q_dev, int *i_in, int *matrixRows_in, int *matrixCols_in,
    int *negative_or_positive_in, dpct::queue_ptr my_stream) {
  int matrixRows = *matrixRows_in;
  int matrixCols = *matrixCols_in;
  int negative_or_positive = *negative_or_positive_in;
  int i = *i_in;

  sycl::range<3> threadsPerBlock(1, 1, 1024);
  sycl::range<3> blocks(
      1, 1, (matrixCols + threadsPerBlock[2] - 1) / threadsPerBlock[2]);

#ifdef WITH_GPU_STREAMS
    sycl_copy_skewsymmetric_first_half_q_double_minus_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, i, matrixRows, matrixCols);
#else
    /*
    DPCT1049:8: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});

    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
        [=](sycl::nd_item<3> item_ct1) {
          sycl_copy_skewsymmetric_first_half_q_double_minus_kernel(
              q_dev, i, matrixRows, matrixCols, item_ct1);
        });
  }
#endif

  /*
  DPCT1010:26: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 cuerr = 0;
}

void sycl_copy_skewsymmetric_first_half_q_float_minus_kernel(float *q, const int i, const int matrixRows, const int matrixCols,
                                                             const sycl::nd_item<3> &item_ct1) {
  int col = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  if (col < matrixCols) {
    int index = (i-1) + matrixRows * col;
    q[index] = -q[index];
  }
}

extern "C" void sycl_copy_skewsymmetric_first_half_q_float_FromC(
    float *q_dev, int *i_in, int *matrixRows_in, int *matrixCols_in,
    int *negative_or_positive_in, dpct::queue_ptr my_stream) {
  int matrixRows = *matrixRows_in;
  int matrixCols = *matrixCols_in;
  int negative_or_positive = *negative_or_positive_in;
  int i = *i_in;

  sycl::range<3> threadsPerBlock(1, 1, 1024);
  sycl::range<3> blocks(
      1, 1, (matrixCols + threadsPerBlock[2] - 1) / threadsPerBlock[2]);

#ifdef WITH_GPU_STREAMS
    sycl_copy_skewsymmetric_first_half_q_float_minus_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, i, matrixRows, matrixCols);
#else
    /*
    DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_copy_skewsymmetric_first_half_q_float_minus_kernel(
            q_dev, i, matrixRows, matrixCols, item_ct1);
      });
#endif

  /*
  DPCT1010:28: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 cuerr = 0;
}

void sycl_get_skewsymmetric_second_half_q_double_kernel(double *q, double* q_2, const int matrixRows, const int matrixCols,
                                                        const sycl::nd_item<3> &item_ct1) {
  int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  int col = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);

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

extern "C" void sycl_get_skewsymmetric_second_half_q_double_FromC(
    double *q_dev, double *q_2_dev, int *matrixRows_in, int *matrixCols_in,
    dpct::queue_ptr my_stream) {
  int matrixRows = *matrixRows_in;
  int matrixCols = *matrixCols_in;

  sycl::range<3> threadsPerBlock(1, 32, 32);
  sycl::range<3> blocks(
      1, (matrixCols + threadsPerBlock[1] - 1) / threadsPerBlock[1],
      (matrixRows + threadsPerBlock[2] - 1) / threadsPerBlock[2]);

#ifdef WITH_GPU_STREAMS
    sycl_get_skewsymmetric_second_half_q_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, q_2_dev, matrixRows, matrixCols);
#else
    /*
    DPCT1049:10: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});

    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
        [=](sycl::nd_item<3> item_ct1) {
          sycl_get_skewsymmetric_second_half_q_double_kernel(
              q_dev, q_2_dev, matrixRows, matrixCols, item_ct1);
        });
  }
#endif

  /*
  DPCT1010:30: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 cuerr = 0;
}

void sycl_get_skewsymmetric_second_half_q_float_kernel(float *q, float *q_2, const int matrixRows, const int matrixCols,
                                                       const sycl::nd_item<3> &item_ct1) {
  int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  int col = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);

  if (row < matrixRows && col < matrixCols) {
    int index  = row + matrixRows * col;
    int index2 = row + matrixRows * (col + matrixCols);
    q_2[index] = q[index2];
  }
}

extern "C" void sycl_get_skewsymmetric_second_half_q_float_FromC(
    float *q_dev, float *q_2_dev, int *matrixRows_in, int *matrixCols_in,
    dpct::queue_ptr my_stream) {
  int matrixRows = *matrixRows_in;
  int matrixCols = *matrixCols_in;

  sycl::range<3> threadsPerBlock(1, 32, 32);
  sycl::range<3> blocks(
      1, (matrixCols + threadsPerBlock[1] - 1) / threadsPerBlock[1],
      (matrixRows + threadsPerBlock[2] - 1) / threadsPerBlock[2]);

#ifdef WITH_GPU_STREAMS
    sycl_get_skewsymmetric_second_half_q_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, q_2_dev, matrixRows, matrixCols);
#else
    /*
    DPCT1049:11: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_get_skewsymmetric_second_half_q_float_kernel(
            q_dev, q_2_dev, matrixRows, matrixCols, item_ct1);
      });
#endif

  /*
  DPCT1010:32: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 cuerr = 0;
}

void sycl_put_skewsymmetric_second_half_q_double_kernel(double *q, double* q_2, const int matrixRows, const int matrixCols,
                                                        const sycl::nd_item<3> &item_ct1) {
  int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  int col = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);

  if (row < matrixRows && col < matrixCols) {
    int index  = row + matrixRows * col;
    int index2 = row + matrixRows * (col + matrixCols);
    q[index2] = q_2[index];
  }
}

extern "C" void sycl_put_skewsymmetric_second_half_q_double_FromC(
    double *q_dev, double *q2_dev, int *matrixRows_in, int *matrixCols_in,
    dpct::queue_ptr my_stream) {
  int matrixRows = *matrixRows_in;
  int matrixCols = *matrixCols_in;

  sycl::range<3> threadsPerBlock(1, 32, 32);
  sycl::range<3> blocks(
      1, (matrixCols + threadsPerBlock[1] - 1) / threadsPerBlock[1],
      (matrixRows + threadsPerBlock[2] - 1) / threadsPerBlock[2]);

#ifdef WITH_GPU_STREAMS
    sycl_put_skewsymmetric_second_half_q_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, q2_dev, matrixRows, matrixCols);
#else
    /*
    DPCT1049:12: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});

    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
        [=](sycl::nd_item<3> item_ct1) {
          sycl_put_skewsymmetric_second_half_q_double_kernel(
              q_dev, q2_dev, matrixRows, matrixCols, item_ct1);
        });
  }
#endif

  /*
  DPCT1010:34: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 cuerr = 0;
}

void sycl_put_skewsymmetric_second_half_q_float_kernel(float *q, float* q_2, const int matrixRows, const int matrixCols,
                                                       const sycl::nd_item<3> &item_ct1) {
  int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  int col = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);

  if (row < matrixRows && col < matrixCols) {
    int index  = row + matrixRows * col;
    int index2 = row + matrixRows * (col + matrixCols);
    q[index2] = q_2[index];
  }
}

extern "C" void sycl_put_skewsymmetric_second_half_q_float_FromC(
    float *q_dev, float *q2_dev, int *matrixRows_in, int *matrixCols_in,
    dpct::queue_ptr my_stream) {
  int matrixRows = *matrixRows_in;
  int matrixCols = *matrixCols_in;

  sycl::range<3> threadsPerBlock(1, 32, 32);
  sycl::range<3> blocks(
      1, (matrixCols + threadsPerBlock[1] - 1) / threadsPerBlock[1],
      (matrixRows + threadsPerBlock[2] - 1) / threadsPerBlock[2]);

#ifdef WITH_GPU_STREAMS
    sycl_put_skewsymmetric_second_half_q_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, q2_dev, matrixRows, matrixCols);
#else
    /*
    DPCT1049:13: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_put_skewsymmetric_second_half_q_float_kernel(
            q_dev, q2_dev, matrixRows, matrixCols, item_ct1);
      });
#endif

  /*
  DPCT1010:36: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 cuerr = 0;
}

