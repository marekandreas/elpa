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
#include <hip/hip_complex.h>

#include "config-f90.h"
#ifdef WITH_HIPCUB
#include <hipcub/hipcub.hpp>
#endif

#ifndef WITH_HIPCUB
template <typename T, unsigned int blk> __device__ void warp_reduce_complex(volatile T *s_block)
{
    unsigned int tid = threadIdx.x;

    if (blk >= 64)
    {
        if (tid < 32)
        {
            s_block[tid].x += s_block[tid + 32].x;
            s_block[tid].y += s_block[tid + 32].y;
        }
    }

    if (blk >= 32)
    {
        if (tid < 16)
        {
            s_block[tid].x += s_block[tid + 16].x;
            s_block[tid].y += s_block[tid + 16].y;
        }
    }

    if (blk >= 16)
    {
        if (tid < 8)
        {
            s_block[tid].x += s_block[tid + 8].x;
            s_block[tid].y += s_block[tid + 8].y;
        }
    }

    if (blk >= 8)
    {
        if (tid < 4)
        {
            s_block[tid].x += s_block[tid + 4].x;
            s_block[tid].y += s_block[tid + 4].y;
        }
    }

    if (blk >= 4)
    {
        if (tid < 2)
        {
            s_block[tid].x += s_block[tid + 2].x;
            s_block[tid].y += s_block[tid + 2].y;
        }
    }

    if (blk >= 2)
    {
        if (tid < 1)
        {
            s_block[tid].x += s_block[tid + 1].x;
            s_block[tid].y += s_block[tid + 1].y;
        }
    }
}

template <typename T, unsigned int blk> __device__ void reduce_complex(T *s_block)
{
    unsigned int tid = threadIdx.x;

    if (blk >= 1024)
    {
        if (tid < 512)
        {
            s_block[tid].x += s_block[tid + 512].x;
            s_block[tid].y += s_block[tid + 512].y;
        }

        __syncthreads();
    }

    if (blk >= 512)
    {
        if (tid < 256)
        {
            s_block[tid].x += s_block[tid + 256].x;
            s_block[tid].y += s_block[tid + 256].y;
        }

        __syncthreads();
    }

    if (blk >= 256)
    {
        if (tid < 128)
        {
            s_block[tid].x += s_block[tid + 128].x;
            s_block[tid].y += s_block[tid + 128].y;
        }

        __syncthreads();
    }

    if (blk >= 128)
    {
        if (tid < 64)
        {
            s_block[tid].x += s_block[tid + 64].x;
            s_block[tid].y += s_block[tid + 64].y;
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        warp_reduce_complex<T, blk>(s_block);
    }

}
#endif /* WITH_HIPCUB */


template <unsigned int blk>
__global__ void
#ifdef WITH_HIPCUB
__launch_bounds__(blk)
#endif
compute_hh_trafo_hip_kernel_complex_double(hipDoubleComplex * __restrict__ q, const hipDoubleComplex * __restrict__ hh, const hipDoubleComplex * __restrict__ hh_tau, const int nb, const int ldq, const int ncols)
{
    __shared__ hipDoubleComplex q_s[blk + 1];
#ifndef WITH_HIPCUB
    __shared__ hipDoubleComplex dotp_s[blk];
#else
    typedef hipcub::BlockReduce<hipDoubleComplex, blk> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    hipDoubleComplex q_v, dt, hv, ht;
    __shared__ hipDoubleComplex q_vs;
#endif

    hipDoubleComplex q_v2;

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
        dt = hipCmul(q_v2, hipConj(hv));

        q_v = BlockReduceT(temp_storage).Sum(dt);

	if (tid == 0) q_vs = q_v;
        __syncthreads();

	q_v2 = hipCsub(q_v2, hipCmul(hipCmul(q_vs, ht), hv));
#else
        dotp_s[tid] = hipCmul(q_v2, hipConj(hh[h_off]));

        __syncthreads();

        reduce_complex<hipDoubleComplex, blk>(dotp_s);

        __syncthreads();

        q_v2 = hipCsub(q_v2, hipCmul(hipCmul(dotp_s[0], hh_tau[j - 1]), hh[h_off]));
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

extern "C" void launch_compute_hh_trafo_c_hip_kernel_complex_double(hipDoubleComplex *q, const hipDoubleComplex *hh, const hipDoubleComplex *hh_tau, const int nev, const int nb, const int ldq, const int ncols, hipStream_t my_stream)
{
    hipError_t err;
//#ifdef WITH_GPU_STREAMS
//    hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

    switch (nb)
    {
    case 1024:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<1024>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<1024>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 512:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<512>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<512>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 256:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<256>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<256>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 128:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<128>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<128>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 64:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<64>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<64>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 32:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<32>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<32>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 16:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<16>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<16>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 8:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<8>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<8>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 4:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<4>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<4>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 2:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<2>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<2>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 1:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<1>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_double<1>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    }

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        printf("\n compute_hh_trafo CUDA kernel failed: %s \n",hipGetErrorString(err));
    }
}

template <unsigned int blk>
__global__ void 
#ifdef WITH_HIPCUB
__launch_bounds__(blk)
#endif

compute_hh_trafo_hip_kernel_complex_single(hipFloatComplex * __restrict__ q, const hipFloatComplex * __restrict__ hh, const hipFloatComplex * __restrict__ hh_tau, const int nb, const int ldq, const int ncols)
{
    __shared__ hipFloatComplex q_s[blk + 1];
#ifndef WITH_HIPCUB
    __shared__ hipFloatComplex dotp_s[blk];
#else
    typedef hipcub::BlockReduce<hipFloatComplex, blk> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    hipFloatComplex q_v, dt, hv, ht;
    __shared__ hipFloatComplex q_vs;
#endif

    hipFloatComplex q_v2;

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
        dt = hipCmulf(q_v2, hipConjf(hv));

        q_v = BlockReduceT(temp_storage).Sum(dt);

	if (tid == 0) q_vs = q_v;
        __syncthreads();

	q_v2 = hipCsubf(q_v2, hipCmulf(hipCmulf(q_vs, ht), hv));
#else
        dotp_s[tid] = hipCmulf(q_v2, hipConjf(hh[h_off]));

        __syncthreads();

        reduce_complex<hipFloatComplex, blk>(dotp_s);

        __syncthreads();

        q_v2 = hipCsubf(q_v2, hipCmulf(hipCmulf(dotp_s[0], hh_tau[j - 1]), hh[h_off]));
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

extern "C" void launch_compute_hh_trafo_c_hip_kernel_complex_single(hipFloatComplex *q, const hipFloatComplex *hh, const hipFloatComplex *hh_tau, const int nev, const int nb, const int ldq, const int ncols, hipStream_t my_stream)
{
    hipError_t err;
//#ifdef WITH_GPU_STREAMS
//    hipStream_t elpa_hip_stm = *((hipStream_t*)my_stream);
//#endif

    switch (nb)
    {
    case 1024:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<1024>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<1024>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 512:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<512>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<512>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 256:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<256>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<256>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 128:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<128>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<128>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 64:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<64>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<64>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 32:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<32>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<32>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 16:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<16>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<16>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 8:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<8>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<8>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 4:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<4>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<4>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 2:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<2>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<2>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    case 1:
#ifdef WITH_GPU_STREAMS
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<1>), dim3(nev), dim3(nb), 0, my_stream, q, hh, hh_tau, nb, ldq, ncols);
#else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_hh_trafo_hip_kernel_complex_single<1>), dim3(nev), dim3(nb), 0, 0, q, hh, hh_tau, nb, ldq, ncols);
#endif
        break;
    }

    err = hipGetLastError();
    if (err != hipSuccess)
    {
        printf("\n compute_hh_trafo CUDA kernel failed: %s \n",hipGetErrorString(err));
    }
}
