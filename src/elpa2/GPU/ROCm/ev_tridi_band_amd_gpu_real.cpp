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

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <hip/hip_runtime.h>

#include "config-f90.h"

#ifdef WITH_HIPCUB
#include <hipcub/hipcub.hpp>
#endif

#if (CUDART_VERSION > 9000) && !defined(WITH_HIPCUB)
template <typename T, unsigned int blk> __device__ void warp_shfl_reduce_real(volatile T *s_block)
{
    unsigned int tid = threadIdx.x;

    T val;

    if (blk >= 64)
    {
        if (tid < 32)
        {
            s_block[tid] += s_block[tid + 32];
        }
    }

    val = s_block[tid];

    for (int i = 16; i >= 1; i /= 2)
    {
	    // fix this
        val += __shfl_xor_sync(0xffffffff, val, i, 32);
    }

    s_block[tid] = val;
}
#endif /* (CUDART_VERSION > 9000) && !defined(WITH_HIPCUB)*/

#ifndef WITH_HIPCUB
template <typename T, unsigned int blk> __device__ void warp_reduce_real(volatile T *s_block)
{
    unsigned int tid = threadIdx.x;

    if (blk >= 64)
    {
        if (tid < 32)
        {
            s_block[tid] += s_block[tid + 32];
        }
    }

    if (blk >= 32)
    {
        if (tid < 16)
        {
            s_block[tid] += s_block[tid + 16];
        }
    }

    if (blk >= 16)
    {
        if (tid < 8)
        {
            s_block[tid] += s_block[tid + 8];
        }
    }

    if (blk >= 8)
    {
        if (tid < 4)
        {
            s_block[tid] += s_block[tid + 4];
        }
    }

    if (blk >= 4)
    {
        if (tid < 2)
        {
            s_block[tid] += s_block[tid + 2];
        }
    }

    if (blk >= 2)
    {
        if (tid < 1)
        {
            s_block[tid] += s_block[tid + 1];
        }
    }
}

template <typename T, unsigned int blk> __device__ void reduce_real(T *s_block)
{
    unsigned int tid = threadIdx.x;

    if (blk >= 1024)
    {
        if (tid < 512)
        {
            s_block[tid] += s_block[tid + 512];
        }

        __syncthreads();
    }

    if (blk >= 512)
    {
        if (tid < 256)
        {
            s_block[tid] += s_block[tid + 256];
        }

        __syncthreads();
    }

    if (blk >= 256)
    {
        if (tid < 128)
        {
            s_block[tid] += s_block[tid + 128];
        }

        __syncthreads();
    }

    if (blk >= 128)
    {
        if (tid < 64)
        {
            s_block[tid] += s_block[tid + 64];
        }

        __syncthreads();
    }

#if (CUDART_VERSION > 9000)
    if (blk >= 32)
    {
        if (tid < 32)
        {
            warp_shfl_reduce_real<T, blk>(s_block);
        }
    }
    else
    {
        if (tid < 32)
        {
            warp_reduce_real<T, blk>(s_block);
        }
    }
#else
    if (tid < 32)
    {
        warp_reduce_real<T, blk>(s_block);
    }
#endif
}
#endif /* WITH_HIPCUB */


template <typename T, unsigned int blk>
#ifdef WITH_HIPCUB
__global__ void
__launch_bounds__(blk)
#else
__global__ void
#endif
compute_hh_trafo_hip_kernel_real(T * __restrict__ q, const T * __restrict__ hh, const T * __restrict__ hh_tau, const int nb, const int ldq, const int ncols)
{
    __shared__ T q_s[blk + 1];
#ifdef WITH_HIPCUB
    typedef hipcub::BlockReduce<T, blk> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
#else
    __shared__ T dotp_s[blk];
#endif
    T q_v2;
#ifdef WITH_HIPCUB
    T q_v, dt, hv, ht;
    __shared__ T q_vs;
#endif

    int q_off, h_off, j;

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;

    j = ncols;
    q_off = bid + (j + tid - 1) * ldq;
    h_off = tid + (j - 1) * nb;
    q_s[tid] = q[q_off];

    while (j >= 1)
    {
#ifdef WITH_HIPCUB
	ht = hh_tau[j - 1];
        hv = hh[h_off];
#endif
        if (tid == 0)
        {
            q_s[tid] = q[q_off];
        }

        q_v2 = q_s[tid];
#ifdef WITH_HIPCUB
        dt = q_v2 * hv;

        q_v = BlockReduceT(temp_storage).Sum(dt);

	if (tid == 0) q_vs = q_v;
        __syncthreads();

	q_v2 -= q_vs * ht * hv;
#else
        dotp_s[tid] = q_v2 * hh[h_off];

        __syncthreads();

        reduce_real<T, blk>(dotp_s);

        __syncthreads();

        q_v2 -= dotp_s[0] * hh_tau[j - 1] * hh[h_off];
#endif
        q_s[tid + 1] = q_v2;

        if ((j == 1) || (tid == blockDim.x - 1))
        {
            q[q_off] = q_v2;
        }

        __syncthreads();

        q_off -= ldq;
        h_off -= nb;
        j -= 1;
    }
}

extern "C" void launch_compute_hh_trafo_c_hip_kernel_real_double(double *q, const double *hh, const double *hh_tau, const int nev, const int nb, const int ldq, const int ncols,  hipStream_t my_stream)
{
    hipError_t err;
//#ifdef WITH_GPU_STREAMS
//    hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

    switch (nb)
    {
    case 1024:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 1024>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 1024>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 512:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 512>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 512>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 256:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 256>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 256>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 128:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 128>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 128>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 64:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 64>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 64>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 32:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 32>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 32>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 16:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 16>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 16>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 8:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 8>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 8>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 4:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 4>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 4>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 2:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 2>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 2>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 1:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 1>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<double, 1>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    }

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        printf("\n compute_hh_trafo CUDA kernel failed: %s \n",hipGetErrorString(err));
    }
}

extern "C" void launch_compute_hh_trafo_c_hip_kernel_real_single(float *q, const float *hh, const float *hh_tau, const int nev, const int nb, const int ldq, const int ncols, hipStream_t my_stream)
{
    hipError_t err;
//#ifdef WITH_GPU_STREAMS
//    hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

    switch (nb)
    {
    case 1024:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 1024>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 1024>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 512:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 512>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 512>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 256:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 256>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 256>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 128:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 128>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 128>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 64:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 64>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 64>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 32:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 32>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 32>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 16:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 16>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 16>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 8:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 8>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 8>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 4:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 4>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 4>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 2:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 2>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 2>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 1:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 1>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_real<float, 1>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    }

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        printf("\n compute_hh_trafo CUDA kernel failed: %s \n",hipGetErrorString(err));
    }
}
