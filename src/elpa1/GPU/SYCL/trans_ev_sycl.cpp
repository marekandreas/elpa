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

void sycl_scale_qmat_double_complex_kernel(sycl::double2 *q, sycl::double2 *tau,
                                           const int ldq, const int l_cols,
                                           const sycl::nd_item<3> &item_ct1) {

    double one = 1.0;
    double zero = 0.0;
    sycl::double2 c_one = sycl::double2(one, zero);

    //printf("c: tau[1]=%.6f %.6f \n",tau[1].x,tau[1].y);
    int col = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
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

      q[index] = dpct::cmul<double>(q[index], c_one - tau[1]);
    }
}

extern "C" void sycl_scale_qmat_double_complex_FromC(
    int *ldq_in, int *l_cols_in, double _Complex *q_dev,
    double _Complex *tau_dev, dpct::queue_ptr my_stream) {
  int ldq = *ldq_in;
  int l_cols = *l_cols_in;

  sycl::double2 *q_casted = (sycl::double2 *)q_dev;
  sycl::double2 *tau_casted = (sycl::double2 *)tau_dev;

  sycl::range<3> threadsPerBlock(1, 1, 1024);
  sycl::range<3> blocks(1, 1,
                        (l_cols + threadsPerBlock[2] - 1) / threadsPerBlock[2]);

#ifdef WITH_GPU_STREAMS
  sycl_scale_qmat_double_complex_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_casted, tau_casted, ldq, l_cols);
#else
  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});

    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
        [=](sycl::nd_item<3> item_ct1) {
          sycl_scale_qmat_double_complex_kernel(q_casted, tau_casted, ldq,
                                                l_cols, item_ct1);
        });
  }
#endif
  /*
  DPCT1010:2: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 cuerr = 0;
}

void sycl_scale_qmat_float_complex_kernel(sycl::float2 *q, sycl::float2 *tau,
                                          const int ldq, const int l_cols,
                                          const sycl::nd_item<3> &item_ct1) {

    float one = 1.0f;
    float zero = 0.0f;
    sycl::float2 c_one = sycl::float2(one, zero);

    int col = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int index = (1-1) + ldq * col;

    // q_mat(1,1:l_cols) = q_mat(1,1:l_cols)*(ONE-tau(2))
    if (col < l_cols) {
      //  q[index].x = q[index].x * (one - tau[1].x);
      //  q[index].y = q[index].y * (zero - tau[1].y);
      q[index] = dpct::cmul<float>(q[index], c_one - tau[1]);
    }

}

extern "C" void sycl_scale_qmat_float_complex_FromC(int *ldq_in, int *l_cols_in,
                                                    float _Complex *q_dev,
                                                    float _Complex *tau_dev,
                                                    dpct::queue_ptr my_stream) {
  int ldq = *ldq_in;
  int l_cols = *l_cols_in;

  sycl::float2 *q_casted = (sycl::float2 *)q_dev;
  sycl::float2 *tau_casted = (sycl::float2 *)tau_dev;

  sycl::range<3> threadsPerBlock(1, 1, 1024);
  sycl::range<3> blocks(1, 1,
                        (l_cols + threadsPerBlock[2] - 1) / threadsPerBlock[2]);

#ifdef WITH_GPU_STREAMS
  sycl_scale_qmat_float_complex_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_casted, tau_casted, ldq, l_cols);
#else
  /*
  DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(blocks * threadsPerBlock, threadsPerBlock),
      [=](sycl::nd_item<3> item_ct1) {
        sycl_scale_qmat_float_complex_kernel(q_casted, tau_casted, ldq, l_cols,
                                             item_ct1);
      });
#endif
  /*
  DPCT1010:4: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 cuerr = 0;
}

