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
// --------------------------------------------------------------------------------------------------
//
// This file was originally written by NVIDIA, re-written by A. Marek, MPCDF and then ported to SYCL by Alexander Poeppl, Intel Corporation

#include "config-f90.h"
#include "syclCommon.hpp"

#include <CL/sycl.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <complex>
#include <iostream>
#include <cstdint>
#include <vector>
#include <optional>

template<typename T> void launch_my_pack_c_sycl_kernel(const int row_count, const int n_offset, const int max_idx,
                                                        const int stripe_width, const int a_dim2, const int stripe_count,
                                                        const int l_nev, T *a, T *row_group) {
  sycl::queue &queue = elpa::gpu::sycl::getQueue();
  int const maxWgSize = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
  int wgSize = stripe_width > maxWgSize ? maxWgSize : stripe_width;
  sycl::range<2> rLocal = sycl::range<2>(1, wgSize);
  sycl::range<2> rGlobal = sycl::range<2>(stripe_count, row_count * wgSize);

  for (int i_off = 0; i_off < stripe_width / wgSize; i_off++) {
    queue.parallel_for(sycl::nd_range<2>(rGlobal, rLocal), [=](sycl::nd_item<2> it) {
      //my_pack_c_cuda_kernel_real_single(n_offset, max_idx, stripe_width, a_dim2, l_nev, a_dev, row_group_dev, i_off, it);
      int b_id = it.get_group(0);
      int t_id = it.get_local_id(1) + i_off * it.get_local_range().get(1);
      int dst_ind = b_id * stripe_width + t_id;

      if (dst_ind < max_idx) {
        // dimension of dst - lnev, nblk
        // dimension of src - stripe_width, a_dim2, stripe_count
        row_group[dst_ind + (l_nev * it.get_group(1))] = a[t_id + (stripe_width * (n_offset + it.get_group(1))) + (b_id * stripe_width * a_dim2)];
      }
    });
    queue.wait_and_throw();
  }
}


extern "C" void launch_my_pack_c_sycl_kernel_real_single(const int row_count, const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, float *a_dev, float *row_group_dev) {
  launch_my_pack_c_sycl_kernel<float>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev);
}

extern "C" void launch_my_pack_c_sycl_kernel_real_double(const int row_count, const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, double *a_dev, double *row_group_dev) {
  launch_my_pack_c_sycl_kernel<double>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev);
}

extern "C" void launch_my_pack_c_sycl_kernel_complex_single(const int row_count, const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, std::complex<float> *a_dev, std::complex<float> *row_group_dev) {
  launch_my_pack_c_sycl_kernel<std::complex<float>>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev);
}

extern "C" void launch_my_pack_c_sycl_kernel_complex_double(const int row_count, const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, std::complex<double> *a_dev, std::complex<double> *row_group_dev) {
  launch_my_pack_c_sycl_kernel<std::complex<double>>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev);
}

template<typename T> void launch_extract_hh_tau_c_sycl_kernel(T *bcast_buffer, T *hh_tau, int const nbw, int const n, int const is_zero) {
  sycl::queue &queue = elpa::gpu::sycl::getQueue();
  int const maxWgSize = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
  int numWorkItems = (1 + (n-1)/maxWgSize) * maxWgSize;
  sycl::range<1> rGlobal(numWorkItems);
  sycl::range<1> rLocal(maxWgSize);
  T *hh = bcast_buffer;
  queue.parallel_for(sycl::nd_range<1>(rGlobal, rLocal), [=](sycl::nd_item<1> it) {
    //extract_hh_tau_c_cuda_kernel_real_single(bcast_buffer_dev, hh_tau_dev, nbw, n, is_zero, it);
    int h_idx = it.get_global_id(0);
    if (h_idx < n) {
        //dimension of hh - (nbw, max_blk_size)
        //dimension of hh_tau - max_blk_size
        hh_tau[h_idx] = hh[h_idx * nbw];
        // Replace the first element in the HH reflector with 1.0 or 0.0
        if (is_zero == 0) {
            hh[(h_idx * nbw)] = 1.0;
        } else {
            hh[(h_idx * nbw)] = 0.0;
        }
     }
  });
  queue.wait_and_throw();
}

extern "C" void launch_extract_hh_tau_c_sycl_kernel_real_single(float *bcast_buffer_dev, float *hh_tau_dev, const int nbw, const int n, const int is_zero) {
    launch_extract_hh_tau_c_sycl_kernel<float>(bcast_buffer_dev, hh_tau_dev, nbw, n, is_zero);
}

extern "C" void launch_extract_hh_tau_c_sycl_kernel_real_double(double *bcast_buffer_dev, double *hh_tau_dev, const int nbw, const int n, const int is_zero) {
    launch_extract_hh_tau_c_sycl_kernel<double>(bcast_buffer_dev, hh_tau_dev, nbw, n, is_zero);
}

extern "C" void launch_extract_hh_tau_c_sycl_kernel_complex_single(std::complex<float> *bcast_buffer_dev, std::complex<float> *hh_tau_dev, const int nbw, const int n, const int is_zero) {
    launch_extract_hh_tau_c_sycl_kernel<std::complex<float>>(bcast_buffer_dev, hh_tau_dev, nbw, n, is_zero);
}

extern "C" void launch_extract_hh_tau_c_sycl_kernel_complex_double(std::complex<double> *bcast_buffer_dev, std::complex<double> *hh_tau_dev, const int nbw, const int n, const int is_zero) {
    launch_extract_hh_tau_c_sycl_kernel<std::complex<double>>(bcast_buffer_dev, hh_tau_dev, nbw, n, is_zero);
}

template<typename T> void launch_my_unpack_c_sycl_kernel(int const row_count, int const n_offset, int const max_idx,
                                                          int const stripe_width, int const a_dim2, int const stripe_count,
                                                          int const l_nev, T *row_group, T* a) {
  sycl::queue &queue = elpa::gpu::sycl::getQueue();
  int const maxWgSize = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
  int wgSize = stripe_width > maxWgSize ? maxWgSize : stripe_width;
  sycl::range<2> rLocal = sycl::range<2>(1, wgSize);
  sycl::range<2> rGlobal = sycl::range<2>(stripe_count, row_count * wgSize);
  for (int i_off = 0; i_off < stripe_width / wgSize; i_off++) {
    queue.parallel_for(sycl::nd_range<2>(rGlobal, rLocal), [=](sycl::nd_item<2> it) {
        int b_id = it.get_group(0);
        int t_id = it.get_local_id(1) + i_off * it.get_local_range(1);
        int src_ind = b_id * stripe_width + t_id;

        if (src_ind < max_idx) {
          a[(t_id + ((n_offset + it.get_group(1)) * stripe_width) + (b_id * stripe_width * a_dim2))] = row_group[src_ind + it.get_group(1) * l_nev];
        }
    });
  }
  queue.wait_and_throw();
}

extern "C" void launch_my_unpack_c_sycl_kernel_real_single(const int row_count, const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, float *row_group_dev, float *a_dev) {
  launch_my_unpack_c_sycl_kernel<float>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, row_group_dev, a_dev);
}

extern "C" void launch_my_unpack_c_sycl_kernel_real_double(const int row_count, const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, double *row_group_dev, double *a_dev) {
  launch_my_unpack_c_sycl_kernel<double>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, row_group_dev, a_dev);
}

extern "C" void launch_my_unpack_c_sycl_kernel_complex_single(const int row_count, const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, std::complex<float> *row_group_dev, std::complex<float> *a_dev) {
  launch_my_unpack_c_sycl_kernel<std::complex<float>>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, row_group_dev, a_dev);
}

extern "C" void launch_my_unpack_c_sycl_kernel_complex_double(const int row_count, const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, std::complex<double> *row_group_dev, std::complex<double> *a_dev) {
  launch_my_unpack_c_sycl_kernel<std::complex<double>>(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, row_group_dev, a_dev);
}
