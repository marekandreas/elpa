/* This file contains modified/adapted version of the original implementation kindly
 * provided by NVIDIA under the MIT License. The unmodified version can be found
 * in the src at src/shipped_srcs/NVIDIA_A100_kernel/
 *
 * Nov 2021, A. Marek, MPCDF
 *
 */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <cub/cub.cuh>

#define USE_MMA // On Ampere the double precision tensor cores (DMMA) are available

#ifdef USE_MMA
#include "mma_m8n8k4_fp64_sm80.dp.hpp"
#else

template<int bK, int bN>
__device__ inline int shared_memory_offset(int k, int n) {
  // Shared memory layout for non-MMA version.
  return k * bN + n;
}

__device__ inline constexpr int shared_memory_bytes(int bK, int bN) {
  // Shared memory size for the bM by bK matrix. Version for the non-MMA.
  return bN * bK;
}

#endif

/*
Householder transformation

This is based the on the original warp sync version shown above.

(I - tau * hh * hh^T) * q = q - tau * hh * hh^T * q

Name here : Name in paper
q         : X
hh        : v
hh_tau    : tau
nev       : N_C
nb        : nbw (==b)
ncols     : N_R (==n+b-1)
*/
template <typename T, int bM, int bN, int block_y, int block_z>
void compute_hh_trafo_gpu_new(T * __restrict__ q, const T * __restrict__ hh, const T * __restrict__ hh_tau, const int nev, const int nb, const int ldq, const int ncols,
                              sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
  constexpr int bK = bM;

  auto smem = (int *)dpct_local;

  T *hh_s = reinterpret_cast<T *>(smem);
  T *q_s = &hh_s[bM];
  T *hh_tau_s = &q_s[shared_memory_bytes(bK, bN)];
#ifdef USE_MMA
  T *sum_s = &hh_tau_s[1]; // Shared memory buffer if we perform the inner product with DMMA.
#endif

  int j = ncols;

  int bid = item_ct1.get_group(1) * bN; // n-index offset for this block.

  for (int k = item_ct1.get_local_id(0); k < bK; k += block_z) {
    for (int n = item_ct1.get_local_id(1); n < bN; n += block_y) {
      q_s[shared_memory_offset<bK, bN>(k, n)] = (n + bid) < nev ? q[(j + k - 1) * ldq + n + bid] : 0;
    }
  }

  constexpr int thread_m_dim = bM / block_z;
  constexpr int thread_n_dim = bN / block_y;

  T reg[thread_n_dim * thread_m_dim];

  while (j >= 1)
  {
    int hh_idx = item_ct1.get_local_id(0) * item_ct1.get_local_range().get(1) +
                 item_ct1.get_local_id(1);
    if (hh_idx == 0) { *hh_tau_s = hh_tau[j - 1]; }
    while (hh_idx < nb) {
      hh_s[hh_idx] = hh[hh_idx + (j - 1) * nb];
      hh_idx +=
          item_ct1.get_local_range().get(0) * item_ct1.get_local_range().get(1);
    }

    if (j < ncols && item_ct1.get_local_id(0) == 0) {
      for (int n = item_ct1.get_local_id(1); n < bN; n += block_y) {
        q_s[shared_memory_offset<bK, bN>(0, n)] = (n + bid) < nev ? q[(j + 0 - 1) * ldq + n + bid] : 0;
      }
    }

/**
  If we use DMMA to perform the inner product, call the routine here and store results on the buffer.
  If not, for each eigenvector, for each thread we calculate the `sum`.
 */

#ifdef USE_MMA
    /*
    DPCT1065:71: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    sum<bK, bN, block_z * block_y / 32>(hh_s, q_s, sum_s, item_ct1);
    /*
    DPCT1065:72: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#endif

#pragma unroll
    for (int n = 0; n < thread_n_dim; n++) {
      int n_idx = item_ct1.get_local_id(1) + n * block_y;

#ifndef USE_MMA
    T sum = 0;
#pragma unroll 1
    for (int k = 0; k < bK; k++) {
      sum += hh_s[k] * q_s[shared_memory_offset<bK, bN>(k, n_idx)];
    }
#endif

#pragma unroll
      for (int m = 0; m < thread_m_dim; m++) {
        int m_idx = item_ct1.get_local_id(0) + m * block_z;
#ifdef USE_MMA
        reg[m * thread_n_dim + n] = q_s[shared_memory_offset<bK, bN>(m_idx, n_idx)] - *hh_tau_s * hh_s[m_idx] * sum_s[n_idx];
#else
        reg[m * thread_n_dim + n] = q_s[shared_memory_offset<bK, bN>(m_idx, n_idx)] - *hh_tau_s * hh_s[m_idx] * sum;
#endif
        if (j == 1 || m_idx == bM - 1) {
          if (n_idx + bid < nev) { q[(m_idx + j - 1) * ldq + n_idx + bid] = reg[m * thread_n_dim + n]; }
        }
      }
    }

    /*
    DPCT1065:73: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

#pragma unroll
    for (int m = 0; m < thread_m_dim; m++) {
#pragma unroll
      for (int n = 0; n < thread_n_dim; n++) {
        int m_idx = item_ct1.get_local_id(0) + m * block_z;
        int n_idx = item_ct1.get_local_id(1) + n * block_y;
        if (m_idx + 1 < bM) { q_s[shared_memory_offset<bK, bN>(m_idx + 1, n_idx)] = reg[m * thread_n_dim + n]; }
      }
    }

    j -= 1;
  }
}

void set_max_shared_bytes(const void *func)
{
  // Set such that this kernel can use the maximum shared memory available.
  /*
  DPCT1007:74: Migration of this CUDA API is not supported by the Intel(R) DPC++
  Compatibility Tool.
  */
  cudaFuncSetAttribute(func, cudaFuncAttributePreferredSharedMemoryCarveout,
                       (int)cudaSharedmemCarveoutMaxShared);
  int max_shared_bytes;
  cudaDeviceGetAttribute(&max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  /*
  DPCT1007:75: Migration of this CUDA API is not supported by the Intel(R) DPC++
  Compatibility Tool.
  */
  cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       max_shared_bytes);
}

template <int bM, class F>
void launch_NVIDIA_sm80_kernel(F *q, const F *hh, const F *hh_tau, const int nev, const int nb, const int ldq, const int ncols)
{
#ifdef USE_MMA
  // This is set such that shared memory bank conflicts are minimized.
  constexpr int block_y = bM < 64 ? 8 : 4;
  constexpr int block_z = bM < 64 ? 4 : 8;
#else
  constexpr int block_y = 8;
  constexpr int block_z = 4;
#endif
  constexpr int bN = 8;
  auto kernel = compute_hh_trafo_gpu_new<double, bM, bN, block_y, block_z>;
  set_max_shared_bytes((const void *)kernel);
#ifdef USE_MMA
  /*
  DPCT1083:77: The size of local memory in the migrated code may be different
  from the original code. Check that the allocated memory size in the migrated
  code is correct.
  */
  int shared_bytes = (bM + shared_memory_bytes(bM, bN) + bN + 1) * sizeof(F);
#else
  int shared_bytes = (bM + shared_memory_bytes(bM, bN) + 1) * sizeof(F);
#endif
  int grid_y = (nev + bN - 1) / bN;
  /*
  DPCT1049:76: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
    dpct::get_default_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, grid_y, 1) *
                              sycl::range<3>(block_z, block_y, 1),
                          sycl::range<3>(block_z, block_y, 1)),
        [=](sycl::nd_item<3> item_ct1) {
            (q, hh, hh_tau, nev, nb, ldq, ncols);
        });
}

/*
Name here : Name in paper
q         : X
hh        : v
hh_tau    : tau
nev       : N_C
nb        : nbw (==b)
ncols     : N_R (==n+b-1)
*/
extern "C" {
  void launch_compute_hh_trafo_c_cuda_sm80_kernel_real_double(double *q, const double *hh, const double *hh_tau, const int nev, const int nb, const int ldq, const int ncols)
  
  {
  
      switch (nb) {
        case 1024: launch_NVIDIA_sm80_kernel<1024>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
        case  512: launch_NVIDIA_sm80_kernel< 512>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
        case  256: launch_NVIDIA_sm80_kernel< 256>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
        case  128: launch_NVIDIA_sm80_kernel< 128>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
        case   64: launch_NVIDIA_sm80_kernel<  64>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
        case   32: launch_NVIDIA_sm80_kernel<  32>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
        case   16: launch_NVIDIA_sm80_kernel<  16>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
        case    8: launch_NVIDIA_sm80_kernel<   8>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
        case    4: launch_NVIDIA_sm80_kernel<   4>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
        //case    2: launch_new_kernel<   2>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
        //case    1: launch_new_kernel<   1>(q, hh, hh_tau, nev, nb, ldq, ncols); break;
        default: printf("Unsupported nb = %d for new kernel \n", nb);
      }

      /*
      DPCT1010:78: SYCL uses exceptions to report errors and does not use the
      error codes. The call was replaced with 0. You need to rewrite this code.
      */
      int err = 0;
  }

  void launch_compute_hh_trafo_c_cuda_sm80_kernel_real_single(float *q, const float *hh, const float *hh_tau, const int nev, const int nb, const int ldq, const int ncols) {
  double *q_casted, *hh_casted, *hh_tau_casted;

  q_casted = (double*) q;
  hh_casted = (double*) hh;
  hh_tau_casted = (double*) hh_tau;

  launch_compute_hh_trafo_c_cuda_sm80_kernel_real_double(q_casted, hh_casted, hh_tau_casted, nev, nb, ldq, ncols);

  q = (float*) q_casted;

 }
}
