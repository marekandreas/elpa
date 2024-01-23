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
//    This file was ported from the NVIDIA version of the component by A. Poeppl, Intel Corporation

#ifndef ELPA_SYCL_IN_PROXY_APP
#include "config-f90.h"
#endif

#include <CL/sycl.hpp>
#include <stdlib.h>
#include <stdio.h>

#include <complex>
//#include <iostream>
#include <cstdint>
//#include <vector>
//#include <optional>
#include <type_traits>

#ifndef ELPA_SYCL_IN_PROXY_APP
#include "src/GPU/SYCL/syclCommon.hpp"
#else
#include "syclCommon.hpp"
#endif

// Detect complex number template arguments
namespace {
    template <class> struct is_complex_number                    : public std::false_type {};
    template <class T> struct is_complex_number<std::complex<T>> : public std::true_type {};
}

template <typename T>
struct extract_float_type;

template <typename X>
struct extract_float_type<std::complex<X>> {
    using type = X;
};


template<typename T, int wg_size, int sg_size, int step>
inline void reduction_step(T *local_mem, sycl::nd_item<1> &it) {
  auto lId = it.get_local_id(0);

  if constexpr (wg_size >= step && sg_size < step) {
    int constexpr half_step = step >> 1;
    if constexpr (step == wg_size) {
      local_mem[lId] += (lId < half_step) ? local_mem[lId + half_step] : 0;
    } else {
      local_mem[lId] += static_cast<T>(lId < half_step) * local_mem[lId + half_step];
    }
    it.barrier(sycl::access::fence_space::local_space);
  }
}

template <typename T, int wg_size, int sg_size>
T parallel_sum_group(sycl::nd_item<1> &it, T *local_mem) {
  it.barrier(sycl::access::fence_space::local_space);
  reduction_step<T, wg_size, sg_size, 1024>(local_mem, it);
  reduction_step<T, wg_size, sg_size,  512>(local_mem, it);
  reduction_step<T, wg_size, sg_size,  256>(local_mem, it);
  reduction_step<T, wg_size, sg_size,  128>(local_mem, it);
  reduction_step<T, wg_size, sg_size,   64>(local_mem, it);
  reduction_step<T, wg_size, sg_size,   32>(local_mem, it);
  reduction_step<T, wg_size, sg_size,   16>(local_mem, it);
  reduction_step<T, wg_size, sg_size,    8>(local_mem, it);
  reduction_step<T, wg_size, sg_size,    4>(local_mem, it);
  reduction_step<T, wg_size, sg_size,    2>(local_mem, it);

  T local_res = local_mem[it.get_local_id(0) & (sg_size - 1)];
  T sg_added_res = sycl::reduce_over_group(it.get_sub_group(), local_res, sycl::plus<>());
  return sycl::select_from_group(it.get_sub_group(), sg_added_res, 0);
}

template<typename T, int wg_size, int sg_size, int step>
inline void reduction_step_complex(T *local_mem, sycl::nd_item<1> &it) {
  auto lId = it.get_local_id(0);
  if constexpr (wg_size >= step && sg_size <= step) {
    local_mem[lId] += static_cast<T>(lId < step) * local_mem[lId + step];
    it.barrier(sycl::access::fence_space::local_space);
  }
}

template <typename T, int sg_size, int step>
inline void sg_reduction_step_complex(T *local_mem, T &accu, sycl::nd_item<1> &it) {
  if constexpr (sg_size >= step) {
    int constexpr half_step = step >> 1;
    auto sg = it.get_sub_group();
    auto sglId = sg.get_local_id();
    accu += static_cast<T>(sglId < step) * sycl::shift_group_left(sg, accu, half_step);
  }
}

template <typename T, int wg_size, int sg_size>
__attribute__((flatten)) std::complex<T> parallel_sum_group_complex(sycl::nd_item<1> &it, std::complex<T> *local_mem) {
  T *local_mem_comps = reinterpret_cast<T *>(local_mem);
  auto lId = it.get_local_id(0);
  it.barrier(sycl::access::fence_space::local_space);
  reduction_step_complex<T, wg_size, sg_size, 1024>(local_mem_comps, it);
  reduction_step_complex<T, wg_size, sg_size,  512>(local_mem_comps, it);
  reduction_step_complex<T, wg_size, sg_size,  256>(local_mem_comps, it);
  reduction_step_complex<T, wg_size, sg_size,  128>(local_mem_comps, it);
  reduction_step_complex<T, wg_size, sg_size,   64>(local_mem_comps, it);
  reduction_step_complex<T, wg_size, sg_size,   32>(local_mem_comps, it);
  reduction_step_complex<T, wg_size, sg_size,   16>(local_mem_comps, it);
  reduction_step_complex<T, wg_size, sg_size,    8>(local_mem_comps, it);
  reduction_step_complex<T, wg_size, sg_size,    4>(local_mem_comps, it);
  reduction_step_complex<T, wg_size, sg_size,    2>(local_mem_comps, it);
  T accu = local_mem_comps[lId & (sg_size - 1)];
  sg_reduction_step_complex<T, sg_size, 1024>(local_mem_comps, accu, it);
  sg_reduction_step_complex<T, sg_size,  512>(local_mem_comps, accu, it);
  sg_reduction_step_complex<T, sg_size,  256>(local_mem_comps, accu, it);
  sg_reduction_step_complex<T, sg_size,  128>(local_mem_comps, accu, it);
  sg_reduction_step_complex<T, sg_size,   64>(local_mem_comps, accu, it);
  sg_reduction_step_complex<T, sg_size,   32>(local_mem_comps, accu, it);
  sg_reduction_step_complex<T, sg_size,   16>(local_mem_comps, accu, it);
  sg_reduction_step_complex<T, sg_size,    8>(local_mem_comps, accu, it);
  sg_reduction_step_complex<T, sg_size,    4>(local_mem_comps, accu, it);
  T real = sycl::select_from_group(it.get_sub_group(), accu, 0);
  T imag = sycl::select_from_group(it.get_sub_group(), accu, 1);
  return std::complex<T>(real, imag);
}

template <typename T, int wg_size, int sg_size, bool is_using_custom_reduction=true>
void compute_hh_trafo_c_sycl_kernel(T *q, T const *hh, T const *hh_tau, int const nev, int const nb, int const ldq, int const ncols) {
  // DPC++ & SYCL 1.2.1 is gradually replaced by SYCL2020. This is to keep ELPA compatible with both old and new versions.
#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER < 20230000
  using local_buffer = sycl::accessor<T, 1, sycl::access_mode::read_write, sycl::access::target::local>;
#else
  using local_buffer = sycl::local_accessor<T>;
#endif
#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER < 20240000
  #define GET_POINTER(x) x.get_pointer()
#else
  #define GET_POINTER(x) x.template get_multi_ptr<sycl::access::decorated::yes>().get()
#endif
  using sf = sycl::access::fence_space;
  auto device = elpa::gpu::sycl::getDevice();
  auto queue = elpa::gpu::sycl::getQueue();

  int constexpr q_reserve_size = wg_size;

  queue.submit([&](sycl::handler &h) {
    sycl::range<1> global_range(nev * nb);
    sycl::range<1> local_range(nb);

    local_buffer q_reserve(sycl::range(q_reserve_size), h);
    local_buffer q_s(sycl::range(nb+1), h);
    local_buffer dotp_s(sycl::range(nb+1), h);

    h.parallel_for(sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> it) /*[[intel::reqd_sub_group_size(32)]]*/ {
      int tid = it.get_local_id(0);
      int local_range = it.get_local_range(0);

      int j = ncols;
      int reserve_counter = q_reserve_size;
      int q_off =     it.get_group(0) + ldq * (j + tid - 1);
      int q_off_res = it.get_group(0) + ldq * (j       - 1);
      int h_off = tid + (j - 1) * nb;

      q_s[tid] = q[q_off];

      for (; j >= 1; j--) {
        if (reserve_counter >= q_reserve_size) {
          if (j - tid >= 1 && tid < q_reserve_size) {
            q_reserve[tid] = q[q_off - 2 * tid * ldq];
          }
          reserve_counter = 0;
        }

        if (tid == 0) {
            q_s[0] = q_reserve[reserve_counter];
        }
        reserve_counter++;

        T q_v2 = q_s[tid];
        T hh_h_off = hh[h_off];
        T q_v2_hh_h_h_off;
        T dotp_res;
        if constexpr (is_complex_number<T>::value) {
          q_v2_hh_h_h_off = q_v2 * std::conj(hh_h_off);
        } else {
          q_v2_hh_h_h_off = q_v2 * hh_h_off;
        }

        if constexpr (is_using_custom_reduction) {
          dotp_s[tid] = q_v2_hh_h_h_off;
          it.barrier(sf::local_space);

          if constexpr (is_complex_number<T>::value) {
            dotp_res = parallel_sum_group_complex<typename extract_float_type<T>::type, wg_size, sg_size>(it, GET_POINTER(dotp_s));
          } else {
            dotp_res = parallel_sum_group<T, wg_size, sg_size>(it, GET_POINTER(dotp_s));
          }
        } else {
          dotp_res = sycl::reduce_over_group(it.get_group(), q_v2_hh_h_h_off, sycl::plus<>());
        }

        q_v2 -= dotp_res * hh_tau[j - 1] * hh_h_off;
        q_s[tid + 1] = q_v2;
        it.barrier(sf::local_space);

        if ((j == 1) || (tid == it.get_local_range()[0] - 1)) {
          q[q_off] = q_v2;
        }

        q_off -= ldq;
        q_off_res -= ldq;
        h_off -= nb;
      }
    });
  });
  queue.wait_and_throw();
}

template <typename T>
void launch_compute_hh_trafo_c_sycl_kernel(T *q, const T *hh, const T *hh_tau, const int nev, const int nb, const int ldq, const int ncols) {
  int const sg_size = 32;
  switch (nb) {
    case 1024: compute_hh_trafo_c_sycl_kernel<T, 1024, sg_size>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
    case 512:  compute_hh_trafo_c_sycl_kernel<T, 512, sg_size>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
    case 256:  compute_hh_trafo_c_sycl_kernel<T, 256, sg_size>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
    case 128:  compute_hh_trafo_c_sycl_kernel<T, 128, sg_size>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
    case 64:   compute_hh_trafo_c_sycl_kernel<T, 64, sg_size>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
    case 32:   compute_hh_trafo_c_sycl_kernel<T, 32, sg_size>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
    case 16:   compute_hh_trafo_c_sycl_kernel<T, 16, sg_size>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
    case 8:    compute_hh_trafo_c_sycl_kernel<T, 8, sg_size>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
    case 4:    compute_hh_trafo_c_sycl_kernel<T, 4, sg_size>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
    case 2:    compute_hh_trafo_c_sycl_kernel<T, 2, sg_size>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
    case 1:    compute_hh_trafo_c_sycl_kernel<T, 1, sg_size>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
    default:   abort();
  }
}

extern "C" void launch_compute_hh_trafo_c_sycl_kernel_real_double(double *q, const double *hh, const double *hh_tau, const int nev, const int nb, const int ldq, const int ncols) {
  launch_compute_hh_trafo_c_sycl_kernel<double>(q, hh, hh_tau, nev, nb, ldq, ncols);
}

extern "C" void launch_compute_hh_trafo_c_sycl_kernel_real_single(float *q, const float *hh, const float *hh_tau, const int nev, const int nb, const int ldq, const int ncols) {
  launch_compute_hh_trafo_c_sycl_kernel<float>(q, hh, hh_tau, nev, nb, ldq, ncols);
}

extern "C" void launch_compute_hh_trafo_c_sycl_kernel_complex_double(std::complex<double> *q, const std::complex<double> *hh, const std::complex<double> *hh_tau, const int nev, const int nb, const int ldq, const int ncols) {
  launch_compute_hh_trafo_c_sycl_kernel<std::complex<double>>(q, hh, hh_tau, nev, nb, ldq, ncols);
}

extern "C" void launch_compute_hh_trafo_c_sycl_kernel_complex_single(std::complex<float> *q, const std::complex<float> *hh, const std::complex<float> *hh_tau, const int nev, const int nb, const int ldq, const int ncols) {
  launch_compute_hh_trafo_c_sycl_kernel<std::complex<float>>(q, hh, hh_tau, nev, nb, ldq, ncols);
}
