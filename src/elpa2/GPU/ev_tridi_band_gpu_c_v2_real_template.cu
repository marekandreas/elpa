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
// This file was originally written by NVIDIA
// and re-written by A. Marek, MPCDF


#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "config-f90.h"

#define BLOCK_CYCLIC_BLOCKSIZE 128
#define GLOBAL_STRIPE_WIDTH 256

// Perform the equivalent of "__shfl_xor" on an 8-byte value
#ifdef DOUBLE_PRECISION_REAL
static __device__ __forceinline__ double shfl_xor_real_double(double r, int mask)
#else
static __device__ __forceinline__ float shfl_xor_real_single(float r, int mask)
#endif
{
    // The following operations do not exist in CUDA 10.1 any more
    // It has been commented out. The code is still compiled, but not used
    // TODO do it properly
    assert(0);

//    int hi = __shfl_xor(__double2hiint(r), mask);
//    int lo = __shfl_xor(__double2loint(r), mask);
//
//    return __hiloint2double(hi, lo);
    return 0.;
}

// Perform the equivalent of "__shfl_down" on an 8-byte value
#ifdef DOUBLE_PRECISION_REAL
static __device__ __forceinline__ double shfl_down_real_double(double r, int offset)
#else
static __device__ __forceinline__ float shfl_down_real_single(float r, int offset)
#endif
{
    // The following operations do not exist in CUDA 10.1 any more
    // It has been commented out. The code is still compiled, but not used
    // TODO do it properly
    assert(0);

//    int hi = __shfl_down(__double2hiint(r), offset);
//    int lo = __shfl_down(__double2loint(r), offset);
//
//    return __hiloint2double(hi, lo);
    return 0.;
}

// Perform a reduction on a warp or the first part of it
template <unsigned int REDUCE_START_OFFSET>
#ifdef DOUBLE_PRECISION_REAL
__device__ __forceinline__ double warp_reduce_real_double(double r)
#else
__device__ __forceinline__ float warp_reduce_real_single(float r)
#endif
{
#pragma unroll
    for (int i = REDUCE_START_OFFSET; i >= 1; i >>= 1)
    {
#ifdef DOUBLE_PRECISION_REAL
        r += shfl_down_real_double(r, i);
#else
        r += shfl_down_real_single(r, i);
#endif
    }

    return r;
}

// Perform 2 reductions, using either 1 or 2 warps
template <unsigned int REDUCE_START_OFFSET, bool HAVE_2_WARPS>
#ifdef DOUBLE_PRECISION_REAL
__device__ __forceinline__ void double_warp_reduce_real_double(double * dotp_s, int w_off)
#else
__device__ __forceinline__ void float_warp_reduce_real_single(float * dotp_s, int w_off)
#endif
{
    int t_idx = threadIdx.x;

    if (HAVE_2_WARPS)
    {
        // In this case, we have 2 warps, each doing 1 reduction
        // attention
        if (t_idx < 64)
        {
#ifdef DOUBLE_PRECISION_REAL
            dotp_s[w_off + t_idx] = warp_reduce_real_double<REDUCE_START_OFFSET>(dotp_s[w_off + t_idx] + dotp_s[w_off + t_idx + 32]);
#else
            dotp_s[w_off + t_idx] = warp_reduce_real_single<REDUCE_START_OFFSET>(dotp_s[w_off + t_idx] + dotp_s[w_off + t_idx + 32]);
#endif
        }
    }
    else
    {
        // In this case we have 1 warp that performs both reductions
        // attention
        if (t_idx < 32)
        {
#ifdef DOUBLE_PRECISION_REAL
            dotp_s[t_idx] = warp_reduce_real_double<REDUCE_START_OFFSET>(dotp_s[t_idx] + dotp_s[t_idx + 32]);
            dotp_s[t_idx + 64] = warp_reduce_real_double<REDUCE_START_OFFSET>(dotp_s[t_idx + 64] + dotp_s[t_idx + 96]);
#else
            dotp_s[t_idx] = warp_reduce_real_single<REDUCE_START_OFFSET>(dotp_s[t_idx] + dotp_s[t_idx + 32]);
            dotp_s[t_idx + 64] = warp_reduce_real_single<REDUCE_START_OFFSET>(dotp_s[t_idx + 64] + dotp_s[t_idx + 96]);
#endif
        }
    }
}

// Reset the entire contents of a shared reduction block; the thread block size must be a power-of-2
#ifdef DOUBLE_PRECISION_REAL
__device__ __forceinline__ void reset_dotp_buffers_real_double(double * const __restrict__ s_block)
#else
__device__ __forceinline__ void reset_dotp_buffers_real_single(float * const __restrict__ s_block)
#endif
{
    // attention
    if (blockDim.x >= 64)
    {
        int t_idx = threadIdx.x;

        if (t_idx < 64)
        {
            s_block[t_idx] = s_block[t_idx + 64] = 0.0;
        }
    }
    else
    {
        int s_chunk = BLOCK_CYCLIC_BLOCKSIZE / blockDim.x;
#ifdef DOUBLE_PRECISION_REAL
        int s_chunk_size = s_chunk * sizeof(double);
#else
        int s_chunk_size = s_chunk * sizeof(float);
#endif
        // Each thread resets an equally-sized, contiguous portion of the buffer
        memset(s_block + threadIdx.x * s_chunk, 0, s_chunk_size);
    }
}

// =========================
// Backtransformation kernel
// =========================

// We use templates here to avoid additional branching based on the actual size of the thread-block
template<unsigned int REDUCE_START_OFFSET, bool HAVE_2_WARPS>
#ifdef DOUBLE_PRECISION_REAL
__global__ void __launch_bounds__( BLOCK_CYCLIC_BLOCKSIZE ) compute_hh_trafo_kernel_real_double(double * const __restrict__ q, const double * const __restrict__ hh, const double * const __restrict__ hh_dot,
    const double * const __restrict__ hh_tau, const int nb, const int ldq, const int off, const int ncols)
#else
__global__ void __launch_bounds__( BLOCK_CYCLIC_BLOCKSIZE ) compute_hh_trafo_kernel_real_single(float * const __restrict__ q, const float * const __restrict__ hh, const float * const __restrict__ hh_dot,
    const float * const __restrict__ hh_tau, const int nb, const int ldq, const int off, const int ncols)
#endif

{
#ifdef DOUBLE_PRECISION_REAL
    __shared__ double dotp_s[BLOCK_CYCLIC_BLOCKSIZE];
    __shared__ double q_s[BLOCK_CYCLIC_BLOCKSIZE+1];
#else
    __shared__ float dotp_s[BLOCK_CYCLIC_BLOCKSIZE];
    __shared__ float q_s[BLOCK_CYCLIC_BLOCKSIZE+1];
#endif

    int b_idx, t_idx, q_off, h_off, w_off, j, t_s, q_delta, hh_delta;
#ifdef DOUBLE_PRECISION_REAL
    double q_v_1, q_v_2, hh_v_1, hh_v_2, tau_1, tau_2, s_1, s_2, dot_p, hh_v_3, my_r1, my_r2;
#else
    float q_v_1, q_v_2, hh_v_1, hh_v_2, tau_1, tau_2, s_1, s_2, dot_p, hh_v_3, my_r1, my_r2;
#endif
    // The block index selects the eigenvector (EV) which the current block is responsible for
    b_idx = blockIdx.x;

    // The thread index selects the position inside the eigenvector selected above
    t_idx = threadIdx.x;

    // The warp offset for the current thread: 0 for the first warp, 32 for the second etc.
    w_off = (t_idx >> 5) << 5;

    // The entire contents of the shared reduction buffers must be reset

#ifdef DOUBLE_PRECISION_REAL
   reset_dotp_buffers_real_double(dotp_s);
#else
    reset_dotp_buffers_real_single(dotp_s);
#endif

    // Compute initial access indices
    j = off + ncols - 1;
    q_off = b_idx + (j + t_idx) * ldq;
    h_off = j * nb + t_idx;
    t_s = t_idx >> 1;
    q_delta = ldq << 1;
    hh_delta = nb << 1;

    // Load the last EV components in the EV cache
    if (t_idx > 0)
    {
        q_s[t_idx + 1] = q[q_off];
    }

    // Ensure the ring buffer and reduction buffers are initialized
    sync_real_threads<HAVE_2_WARPS>();

    while (j >= off + 1)
    {
        // Per-iteration GMem I/O reads are in order to improve cache hit ratio

        // Read the corresponding compotents in the 2 Householder reflectors
        hh_v_1 = __ldg(&hh[h_off]);
        hh_v_2 = __ldg(&hh[h_off - nb]);
        hh_v_3 = (t_idx == 0)? 0.0 : __ldg(&hh[h_off - 1]);

        // Read the pre-computed dot-product of the 2 Householder reflectors
        dot_p = __ldg(&hh_dot[j - 1]);

        // Read the pre-computed values for "Tau" corresponding to the 2 Householder reflectors
        tau_1 = __ldg(&hh_tau[j]);
        tau_2 = __ldg(&hh_tau[j - 1]);

        // Only read the new EV components (the others are already stored in the shared EV cache, q_s)
        if (t_idx == 0)
        {
            q_s[0] = q[q_off - ldq];
            q_s[1] = q[q_off];
        }

        // Fill the shared buffers for the dot products bewtween the EV subset and the Householder reflectors
        q_v_1 = q_s[t_idx + 1];
        q_v_2 = q_s[t_idx];

        my_r1 = q_v_1 * hh_v_1 * tau_1;
        my_r2 = q_v_2 * hh_v_2 * tau_2;

        // After using "shfl_xor", both threads in a pair will hold the same values
#ifdef DOUBLE_PRECISION_REAL
        my_r1 += shfl_xor_real_double(my_r1, 1);
        my_r2 += shfl_xor_real_double(my_r2, 1);
#else
        my_r1 += shfl_xor_real_single(my_r1, 1);
        my_r2 += shfl_xor_real_single(my_r2, 1);
#endif

        // Now both threads in a pair can write to the same reduction buffer address without race-condition issues
        dotp_s[t_s] = my_r1;
	//attention
        dotp_s[t_s + 64] = my_r2;

        // Ensure the reduction buffers are fully populated
        sync_real_threads<HAVE_2_WARPS>();

        // Perform the 2 reductions using only the first warp (we assume the warp size is 32, valid up to CC 3.x)
#ifdef DOUBLE_PRECISION_REAL
        double_warp_reduce_real_double<REDUCE_START_OFFSET, HAVE_2_WARPS>(dotp_s, w_off);
#else
        float_warp_reduce_real_single<REDUCE_START_OFFSET, HAVE_2_WARPS>(dotp_s, w_off);
#endif
        // Ensure every thread will have access to the reduction results
        sync_real_threads<HAVE_2_WARPS>();

        // Each thread collects the reduction results
        s_1 = dotp_s[0];

	// attention
        s_2 = dotp_s[64];

        // Each thread updates its corresponding EV component
        q_v_2 = q_v_2 - hh_v_3 * s_1 - hh_v_2 * s_2 + tau_2 * hh_v_2 * s_1 * dot_p;

        if (t_idx == blockDim.x - 1)
        {
            // The last thread writes the last 2 EV components to the EV matrix
            q[q_off] = q_v_1 - hh_v_1 * s_1;
            q[q_off - ldq] = q_v_2;
        }
        else
        {
            // All other threads update the EV cache for the next iteration
            q_s[t_idx + 2] = q_v_2;
        }

        sync_real_threads<HAVE_2_WARPS>();

        // Update access indices
        q_off -= q_delta;
        h_off -= hh_delta;
        j -= 2;
    }

    // Once the previous loop has finished, we have at most 1 more iteration to perform

    if (j == off - 1)
    {
        // No iterations remain, so the final contents of the EV matrix are updated
        if (t_idx < blockDim.x - 1)
        {
            q[q_off + ldq] = q_v_2;
        }
    }
    else
    {
        // One iteration remains; it must be processed separately
        if (t_idx == 0)
        {
            // Only one more EV element needs to be loaded
            q_s[1] = q[q_off];
        }

        // As before, we first read the EV and Householder components
        q_v_1 = q_s[t_idx + 1];
        hh_v_1 = __ldg(&hh[h_off]);
        tau_1 = __ldg(&hh_tau[j]);

        // We prepare the reduction buffer
        my_r1 = q_v_1 * hh_v_1 * tau_1;
#ifdef DOUBLE_PRECISION_REAL
        my_r1 += shfl_xor_real_double(my_r1, 1);
#else
        my_r1 += shfl_xor_real_single(my_r1, 1);
#endif
        dotp_s[t_s] = my_r1;

        sync_real_threads<HAVE_2_WARPS>();

        // We perform the reduction using the first warp only
	// attention
        if (t_idx < 32)
        {
#ifdef DOUBLE_PRECISION_REAL
            dotp_s[t_idx] = warp_reduce_real_double<REDUCE_START_OFFSET>(dotp_s[t_idx] + dotp_s[t_idx + 32]);
#else
            dotp_s[t_idx] = warp_reduce_real_single<REDUCE_START_OFFSET>(dotp_s[t_idx] + dotp_s[t_idx + 32]);
#endif
        }

        sync_real_threads<HAVE_2_WARPS>();

        // The last EV components are written to the EV matrix
        q[q_off] = q_v_1 - hh_v_1 * dotp_s[0];
    }
}

// This is a host wrapper for calling the appropriate back-transformation kernel, based on the SCALAPACK block size
#ifdef DOUBLE_PRECISION_REAL
 extern "C" void launch_compute_hh_trafo_c_kernel_real_double(double * const q, const double * const hh, const double * const hh_dot,  const double * const hh_tau, const int nev, const int nb, const int ldq, const int off, const int ncols)
#else
 extern "C" void launch_compute_hh_trafo_c_kernel_real_single(float * const q, const float * const hh, const float * const hh_dot,  const float * const hh_tau, const int nev, const int nb, const int ldq, const int off, const int ncols)
#endif
{
    switch (nb)
    {
        // attention
        case 128:
        case 64:
#ifdef DOUBLE_PRECISION_REAL
            compute_hh_trafo_kernel_real_double<16, true><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
#else
            compute_hh_trafo_kernel_real_single<16, true><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
#endif
            break;

        case 32:
#ifdef DOUBLE_PRECISION_REAL
            compute_hh_trafo_kernel_real_double<8, false><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
#else
            compute_hh_trafo_kernel_real_single<8, false><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
#endif
            break;

        case 16:
#ifdef DOUBLE_PRECISION_REAL
            compute_hh_trafo_kernel_real_double<4, false><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
#else
            compute_hh_trafo_kernel_real_single<4, false><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
#endif
            break;

        case 8:
#ifdef DOUBLE_PRECISION_REAL
            compute_hh_trafo_kernel_real_double<2, false><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
#else
            compute_hh_trafo_kernel_real_single<2, false><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
#endif
            break;

        case 4:
#ifdef DOUBLE_PRECISION_REAL
            compute_hh_trafo_kernel_real_double<1, false><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
#else
            compute_hh_trafo_kernel_real_single<1, false><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
#endif
            break;

        case 2:
        case 1:
#ifdef DOUBLE_PRECISION_REAL
            compute_hh_trafo_kernel_real_double<0, false><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
#else
            compute_hh_trafo_kernel_real_single<0, false><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
#endif
            break;

        default:
            printf("Error: please use a power-of-2 SCALAPACK block size which is between 1 and BLOCK_CYCLIC_BLOCKSIZE .\n");
    }
}

