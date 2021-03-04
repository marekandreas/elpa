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
#include <cuda_runtime.h>

#if (CUDART_VERSION >= 9000)
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
        val += __shfl_xor_sync(0xffffffff, val, i, 32);
    }

    s_block[tid] = val;
}
#endif

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

template <typename T, unsigned int blk>
__global__ void compute_hh_trafo_cuda_kernel_real(T * __restrict__ q, const T * __restrict__ hh, const T * __restrict__ hh_tau, const int nb, const int ldq, const int ncols)
{
    __shared__ T q_s[blk + 1];
    __shared__ T dotp_s[blk];

    T q_v2;

    int q_off, h_off, j;

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;

    j = ncols;
    q_off = bid + (j + tid - 1) * ldq;
    h_off = tid + (j - 1) * nb;
    q_s[tid] = q[q_off];

    while (j >= 1)
    {
        if (tid == 0)
        {
            q_s[tid] = q[q_off];
        }

        q_v2 = q_s[tid];
        dotp_s[tid] = q_v2 * hh[h_off];

        __syncthreads();

        reduce_real<T, blk>(dotp_s);

        __syncthreads();

        q_v2 -= dotp_s[0] * hh_tau[j - 1] * hh[h_off];
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

extern "C" void launch_compute_hh_trafo_c_cuda_kernel_real_double(double *q, const double *hh, const double *hh_tau, const int nev, const int nb, const int ldq, const int ncols)
{
    cudaError_t err;

    switch (nb)
    {
    case 1024:
        compute_hh_trafo_cuda_kernel_real<double, 1024><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 512:
        compute_hh_trafo_cuda_kernel_real<double, 512><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 256:
        compute_hh_trafo_cuda_kernel_real<double, 256><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 128:
        compute_hh_trafo_cuda_kernel_real<double, 128><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 64:
        compute_hh_trafo_cuda_kernel_real<double, 64><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 32:
        compute_hh_trafo_cuda_kernel_real<double, 32><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 16:
        compute_hh_trafo_cuda_kernel_real<double, 16><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 8:
        compute_hh_trafo_cuda_kernel_real<double, 8><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 4:
        compute_hh_trafo_cuda_kernel_real<double, 4><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 2:
        compute_hh_trafo_cuda_kernel_real<double, 2><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 1:
        compute_hh_trafo_cuda_kernel_real<double, 1><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("\n compute_hh_trafo CUDA kernel failed: %s \n",cudaGetErrorString(err));
    }
}

extern "C" void launch_compute_hh_trafo_c_cuda_kernel_real_single(float *q, const float *hh, const float *hh_tau, const int nev, const int nb, const int ldq, const int ncols)
{
    cudaError_t err;

    switch (nb)
    {
    case 1024:
        compute_hh_trafo_cuda_kernel_real<float, 1024><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 512:
        compute_hh_trafo_cuda_kernel_real<float, 512><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 256:
        compute_hh_trafo_cuda_kernel_real<float, 256><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 128:
        compute_hh_trafo_cuda_kernel_real<float, 128><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 64:
        compute_hh_trafo_cuda_kernel_real<float, 64><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 32:
        compute_hh_trafo_cuda_kernel_real<float, 32><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 16:
        compute_hh_trafo_cuda_kernel_real<float, 16><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 8:
        compute_hh_trafo_cuda_kernel_real<float, 8><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 4:
        compute_hh_trafo_cuda_kernel_real<float, 4><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 2:
        compute_hh_trafo_cuda_kernel_real<float, 2><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    case 1:
        compute_hh_trafo_cuda_kernel_real<float, 1><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, ncols);
        break;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("\n compute_hh_trafo CUDA kernel failed: %s \n",cudaGetErrorString(err));
    }
}
