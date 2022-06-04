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
//

#include <CL/sycl.hpp>
#include <stdlib.h>
#include <stdio.h>

template <typename T>
void compute_hh_trafo_sycl_kernel(T restrict *q, T const restrict *hh, T const restrict *hh_tau, int const nev, int const nb, int const ldq, int const ncols) {
  using local_double_buffer = sycl::accessor<T, 1, sycl::access_mode::read_write, sycl::access::target::local>;
  auto device = elpa::gpu::sycl::getDevice();
  auto queue = elpa::gpu:sycl::getQueue();

  queue.submit([&](sycl::handler &h) {
    local_double_buffer q_s(sycl::range(nb+1), h);
    local_double_buffer dotp_s(sycl::range(nb+1), h);
    sycl::range global_range(nev * nb);
    sycl::range local_range(nb);

    cgh.parallel_for({global_range, local_range}, [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(32)]] {
      unsigned int tid = it.get_local_id(0);
      int j = ncols;
      int q_off = it.get_group(0) + (j + tid - 1) * ldq;
      int h_off = tid + (j - 1) * nb;

      q_s[tid] = q[q_off];

      T *dotp_sp = dotp_s.get_pointer();

      for (; j >= 1; j--) {
        if (tid == 0) {
            q_s[tid] = q[q_off];
        }

        T q_v2 = q_s[tid];
        dotp_s[tid] = q_v2 * hh[h_off];

        it.barrier(sycl::access::fence_space::local_space);

        //reduce_real<T, blk>(dotp_s, item_ct1);
        int dotp_res = joint_reduce(it.get_group(), dotp_sp, dotp_sp + s, sycl::plus<>());

        //it.barrier(sycl::access::fence_space::local_space);

        q_v2 -= dotp_res * hh_tau[j - 1] * hh[h_off];
        q_s[tid + 1] = q_v2;

        if ((j == 1) || (tid == it.get_local_range()[0] - 1)) {
            q[q_off] = q_v2;
        }
        it.barrier();

        q_off -= ldq;
        h_off -= nb;
      }
    });
  });
}

extern "C" void launch_compute_hh_trafo_c_cuda_kernel_real_double(double *q, const double *hh, const double *hh_tau, const int nev, const int nb, const int ldq, const int ncols) {
  compute_hh_trafo_c_sycl_kernel(q, hh, hh_tau, nev, nb, ldq, ncols);
}

extern "C" void launch_compute_hh_trafo_c_cuda_kernel_real_single(float *q, const float *hh, const float *hh_tau, const int nev, const int nb, const int ldq, const int ncols) {
  compute_hh_trafo_c_sycl_kernel(q, hh, hh_tau, nev, nb, ldq, ncols);
}
