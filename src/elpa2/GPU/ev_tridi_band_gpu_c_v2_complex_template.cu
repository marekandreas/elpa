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
#include <cuComplex.h>
#include "config-f90.h"


#define BLOCK_CYCLIC_BLOCKSIZE 128
#define GLOBAL_STRIPE_WIDTH 256

// ===========================================================================================================
// Important:   due to the use of warp shuffling, the C version of the backtransformation kernel only works on
//              devices with compute capability 3.x; for older devices, please use the Fortran kernel version
// ===========================================================================================================


#if 0
static __device__ __forceinline__ cuDoubleComplex  shfl_xor_complex(cuDoubleComplex r, int mask)
{
    double real = cuCreal(r) ;
    double imag =  cuCimag(r);


    int hr = __shfl_xor(__double2hiint(real), mask);
    int lr = __shfl_xor(__double2loint(real), mask);

    int hi = __shfl_xor(__double2hiint(imag), mask);
    int li = __shfl_xor(__double2loint(imag), mask);



    real =      __hiloint2double(hr, lr);
    imag = __hiloint2double(hi, li);
    return       make_cuDoubleComplex(real, imag);

}
#endif


// Perform the equivalent of "__shfl_down" on an 8-byte value
#ifdef DOUBLE_PRECISION_COMPLEX
static __device__ __forceinline__ double shfl_down_complex_double(double r, int offset)
#else
static __device__ __forceinline__ float shfl_down_complex_single(float r, int offset)
#endif
{
    // The following operations do not exist in CUDA 10.1 any more
    // It has been commented out. The code is still compiled, but not used
    // TODO do it properly

    assert(0);
    //int hi = __shfl_down(__double2hiint(r), offset);
    //int lo = __shfl_down(__double2loint(r), offset);

    //return __hiloint2double(hi, lo);
    return 0.;
}

#ifdef DOUBLE_PRECISION_COMPLEX
__device__ void warp_reduce_1_complex_double( cuDoubleComplex *s_block)
#else
__device__ void warp_reduce_1_complex_single( cuFloatComplex *s_block)
#endif
{
    int t_idx ;
    t_idx = threadIdx.x;
    __syncthreads();
    // attention
#ifdef DOUBLE_PRECISION_COMPLEX
        if (t_idx < 32)
        {

	s_block[t_idx] = cuCadd(cuCadd(s_block[t_idx],s_block[t_idx + 32]) , cuCadd( s_block[t_idx + 64], s_block[t_idx + 96]) );
        if (t_idx < 8)
        {
	s_block[t_idx] = cuCadd(cuCadd(s_block[t_idx],s_block[t_idx + 8] ) , cuCadd( s_block[t_idx + 16] , s_block[t_idx + 24] ) );

        }
        if (t_idx < 4)
        {
        s_block[t_idx] = cuCadd(s_block[t_idx] , s_block[t_idx + 4]) ;
        }
        if (t_idx < 1)
        {
	s_block[t_idx] = cuCadd(cuCadd(s_block[t_idx],s_block[t_idx + 1] ) , cuCadd( s_block[t_idx +2] , s_block[t_idx + 3] ) );
        }
        }
#else
        if (t_idx < 32)
        {

	s_block[t_idx] = cuCaddf(cuCaddf(s_block[t_idx],s_block[t_idx + 32]) , cuCaddf( s_block[t_idx + 64], s_block[t_idx + 96]) );
        if (t_idx < 8)
        {
	s_block[t_idx] = cuCaddf(cuCaddf(s_block[t_idx],s_block[t_idx + 8] ) , cuCaddf( s_block[t_idx + 16] , s_block[t_idx + 24] ) );

        }
        if (t_idx < 4)
        {
        s_block[t_idx] = cuCaddf(s_block[t_idx] , s_block[t_idx + 4]) ;
        }
        if (t_idx < 1)
        {
	s_block[t_idx] = cuCaddf(cuCaddf(s_block[t_idx],s_block[t_idx + 1] ) , cuCaddf( s_block[t_idx +2] , s_block[t_idx + 3] ) );
        }
        }
#endif
}

#ifdef DOUBLE_PRECISION_COMPLEX
__device__ void warp_reduce_2_complex_double( cuDoubleComplex *s_block)
#else
__device__ void warp_reduce_2_complex_single( cuFloatComplex *s_block)
#endif
{
    int t_idx ;
    t_idx = threadIdx.x;
    __syncthreads();
        // attention
#ifdef DOUBLE_PRECISION_COMPLEX
        if(t_idx < 64)
        {
	s_block[t_idx] = cuCadd(cuCadd(s_block[t_idx],s_block[t_idx + 64]) , cuCadd( s_block[t_idx + 128], s_block[t_idx + 192]) );
        if (t_idx < 32)
        {
        s_block[t_idx] = cuCadd(cuCadd(s_block[t_idx],s_block[t_idx + 32]) , cuCadd( s_block[t_idx + 64], s_block[t_idx + 96]) );
        }
        if (t_idx < 8)
        {
        s_block[t_idx] = cuCadd(cuCadd(s_block[t_idx],s_block[t_idx + 8] ) , cuCadd( s_block[t_idx + 16] , s_block[t_idx + 24] ) );

        }
        if (t_idx < 4)
        {
        s_block[t_idx] = cuCadd(s_block[t_idx] , s_block[t_idx + 4]) ;
        }
        if (t_idx < 1)
        {
        s_block[t_idx] = cuCadd(cuCadd(s_block[t_idx],s_block[t_idx + 1] ) , cuCadd( s_block[t_idx +2] , s_block[t_idx + 3] ) );
        }
        }
#else
        if(t_idx < 64)
        {
	s_block[t_idx] = cuCaddf(cuCaddf(s_block[t_idx],s_block[t_idx + 64]) , cuCaddf( s_block[t_idx + 128], s_block[t_idx + 192]) );
        if (t_idx < 32)
        {
        s_block[t_idx] = cuCaddf(cuCaddf(s_block[t_idx],s_block[t_idx + 32]) , cuCaddf( s_block[t_idx + 64], s_block[t_idx + 96]) );
        }
        if (t_idx < 8)
        {
        s_block[t_idx] = cuCaddf(cuCaddf(s_block[t_idx],s_block[t_idx + 8] ) , cuCaddf( s_block[t_idx + 16] , s_block[t_idx + 24] ) );

        }
        if (t_idx < 4)
        {
        s_block[t_idx] = cuCaddf(s_block[t_idx] , s_block[t_idx + 4]) ;
        }
        if (t_idx < 1)
        {
        s_block[t_idx] = cuCaddf(cuCaddf(s_block[t_idx],s_block[t_idx + 1] ) , cuCaddf( s_block[t_idx +2] , s_block[t_idx + 3] ) );
        }
        }

#endif
}

template <unsigned int REDUCE_START_OFFSET>
#ifdef DOUBLE_PRECISION_COMPLEX
__device__ __forceinline__ cuDoubleComplex warp_reduce_complex_double( cuDoubleComplex r)
#else
__device__ __forceinline__ cuFloatComplex warp_reduce_complex_single( cuFloatComplex r)
#endif
{

#ifdef DOUBLE_PRECISION_COMPLEX
     double real = cuCreal(r);
     double imag = cuCimag(r);
#else
     float real = cuCrealf(r);
     float imag = cuCimagf(r);
#endif

#pragma unroll
    for (int i = REDUCE_START_OFFSET; i >= 1; i >>= 1)
    {
#ifdef DOUBLE_PRECISION_COMPLEX
        real += shfl_down_complex_double(real, i);
#else
        real += shfl_down_complex_single(real, i);
#endif
    }
#pragma unroll
    for (int i = REDUCE_START_OFFSET; i >= 1; i >>= 1)
    {
#ifdef DOUBLE_PRECISION_COMPLEX
        imag += shfl_down_complex_double(imag, i);
#else
        imag += shfl_down_complex_single(imag, i);
#endif
    }

#ifdef DOUBLE_PRECISION_COMPLEX
    return make_cuDoubleComplex(real,imag);
#else
    return make_cuFloatComplex(real,imag);
#endif
}

#if 0 /* not used anywhere */
template <unsigned int REDUCE_START_OFFSET, bool HAVE_2_WARPS>
#ifdef DOUBLE_PRECISION_COMPLEX
__device__ __forceinline__ void driver_warp_reduce_complex_double(cuDoubleComplex * dotp_s, int w_off)
#else
__device__ __forceinline__ void driver_warp_reduce_complex_single(cuFloatComplex * dotp_s, int w_off)
#endif
{
    int t_idx = threadIdx.x;

    if (HAVE_2_WARPS)
    {
        // In this case, we have 2 warps, each doing 1 reduction
	//attention
        if (t_idx < 64)
        {
#ifdef DOUBLE_PRECISION_COMPLEX
            dotp_s[w_off + t_idx] = warp_reduce_complex_double<REDUCE_START_OFFSET>(cuCadd(dotp_s[w_off + t_idx] , dotp_s[w_off + t_idx + 32]));
#else
            dotp_s[w_off + t_idx] = warp_reduce_complex_single<REDUCE_START_OFFSET>(cuCaddf(dotp_s[w_off + t_idx] , dotp_s[w_off + t_idx + 32]));
#endif
        }
    }
    else
    {
        // In this case we have 1 warp that performs both reductions
	// attention
        if (t_idx < 32)
        {
#ifdef DOUBLE_PRECISION_COMPLEX
            dotp_s[t_idx] = warp_reduce_complex_double<REDUCE_START_OFFSET>(cuCadd(dotp_s[t_idx] ,  dotp_s[t_idx + 32]));
            dotp_s[t_idx + 64] = warp_reduce_complex_double<REDUCE_START_OFFSET>(cuCadd(dotp_s[t_idx + 64] ,  dotp_s[t_idx + 96]));
#else
            dotp_s[t_idx] = warp_reduce_complex_single<REDUCE_START_OFFSET>(cuCaddf(dotp_s[t_idx] ,  dotp_s[t_idx + 32]));
            dotp_s[t_idx + 64] = warp_reduce_complex_single<REDUCE_START_OFFSET>(cuCaddf(dotp_s[t_idx + 64] ,  dotp_s[t_idx + 96]));
#endif
        }
    }
}
#endif /* not used anywhere */


#ifndef ALREADY_DEFINED_SYNC
// Synchronization wrapper, removing explicit synchronization when the thread-block is at most 32 threads (1 warp) in size
template <bool MUST_SYNC>
__device__ __forceinline__ void sync_real_threads()
{
    if (MUST_SYNC)
    {
        __syncthreads();
    }
}
#define ALREADY_DEFINED_SYNC 1
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
__device__  void reset_dotp_buffers_complex_double( cuDoubleComplex  * const __restrict__ s_block)
#else
__device__  void reset_dotp_buffers_complex_single( cuFloatComplex  * const __restrict__ s_block)
#endif
{
    // attention
    if (blockDim.x >= 64)
    {
        int t_idx = threadIdx.x;

        if (t_idx < 64)
        {
            s_block[t_idx].x = s_block[t_idx + 64].x = 0.0;
	    s_block[t_idx].y = s_block[t_idx + 64].y = 0.0;

        }
    }
    else
    {
        int s_chunk = BLOCK_CYCLIC_BLOCKSIZE / blockDim.x;
#ifdef DOUBLE_PRECISION_COMPLEX
        int s_chunk_size = s_chunk * sizeof(cuDoubleComplex);
#else
        int s_chunk_size = s_chunk * sizeof(cuFloatComplex);
#endif

        // Each thread resets an equally-sized, contiguous portion of the buffer
        memset(&(s_block[ threadIdx.x * s_chunk].x), 0, s_chunk_size);
	memset( & (s_block[ threadIdx.x * s_chunk].y), 0, s_chunk_size);

    }
}
#ifdef DOUBLE_PRECISION_COMPLEX
__device__  void reset_dotp_buffers_2_complex_double( cuDoubleComplex  * const __restrict__ s_block)
#else
__device__  void reset_dotp_buffers_2_complex_single( cuFloatComplex  * const __restrict__ s_block)
#endif
{
    if (blockDim.x >= BLOCK_CYCLIC_BLOCKSIZE)
    {
        int t_idx = threadIdx.x;

        if (t_idx < BLOCK_CYCLIC_BLOCKSIZE)
        {
            s_block[t_idx].x = s_block[t_idx + BLOCK_CYCLIC_BLOCKSIZE].x = 0.0;
            s_block[t_idx].y = s_block[t_idx + BLOCK_CYCLIC_BLOCKSIZE].y = 0.0;

        }
    }
    else
    {
        int s_chunk = GLOBAL_STRIPE_WIDTH / blockDim.x;
#ifdef DOUBLE_PRECISION_COMPLEX
        int s_chunk_size = s_chunk * sizeof(cuDoubleComplex);
#else
        int s_chunk_size = s_chunk * sizeof(cuFloatComplex);
#endif
        // Each thread resets an equally-sized, contiguous portion of the buffer
        memset(&(s_block[ threadIdx.x * s_chunk].x), 0, s_chunk_size);
        memset( & (s_block[ threadIdx.x * s_chunk].y), 0, s_chunk_size);

    }
}


// =========================
// Backtransformation kernel
// =========================
#ifdef DOUBLE_PRECISION_COMPLEX
template<unsigned int REDUCE_START_OFFSET>__global__ void compute_hh_trafo_kernel_2_2_complex_double(cuDoubleComplex * const __restrict__  q, const cuDoubleComplex  * const __restrict__   hh,   const cuDoubleComplex * const __restrict__ hh_tau, const int nb, const int ldq, const int off, const int ncols)
#else
template<unsigned int REDUCE_START_OFFSET>__global__ void compute_hh_trafo_kernel_2_2_complex_single(cuFloatComplex * const __restrict__  q, const cuFloatComplex  * const __restrict__   hh,   const cuFloatComplex * const __restrict__ hh_tau, const int nb, const int ldq, const int off, const int ncols)
#endif
{
#ifdef DOUBLE_PRECISION_COMPLEX
    __shared__ cuDoubleComplex q_s[BLOCK_CYCLIC_BLOCKSIZE];
    __shared__ cuDoubleComplex dotp_s[BLOCK_CYCLIC_BLOCKSIZE];

     cuDoubleComplex q_v2, tau ;
#else
    __shared__ cuFloatComplex q_s[BLOCK_CYCLIC_BLOCKSIZE];
    __shared__ cuFloatComplex dotp_s[BLOCK_CYCLIC_BLOCKSIZE];

     cuFloatComplex q_v2, tau ;
#endif

    int  t_idx,q_off, h_off, j , b_idx;

    // The thread index selects the position inside the eigenvector selected above
    t_idx = threadIdx.x;
    b_idx = blockIdx.x ;

    // Compute intial index
    j = ncols ;
     q_off = b_idx + (j + t_idx) * ldq;
         h_off = j * nb + t_idx;

   if(t_idx>0)
   {    q_s[t_idx] = q[ q_off ];
   }

   while (j>=1)
   {

        if ((j == ncols) || (t_idx ==0))
        {
              q_s[t_idx] = q[q_off ];
        }

        q_v2 = q_s[t_idx];
       tau =  hh_tau[j];

        __syncthreads();

        if(t_idx==0)
        {
                dotp_s[t_idx]= q_v2  ;
        }
       else
        {
#ifdef DOUBLE_PRECISION_COMPLEX
		dotp_s[t_idx]  =  cuCmul(q_v2,cuConj( hh[h_off]));
#else
		dotp_s[t_idx]  =  cuCmulf(q_v2,cuConjf( hh[h_off]));
#endif
        }
#ifdef DOUBLE_PRECISION_COMPLEX
        warp_reduce_1_complex_double( dotp_s);
#else
        warp_reduce_1_complex_single( dotp_s);
#endif

        __syncthreads();
        if(t_idx ==0)
        {
#ifdef DOUBLE_PRECISION_COMPLEX
		q_v2 =  cuCsub(q_v2,cuCmul(dotp_s[0], tau) );
#else
		q_v2 =  cuCsubf(q_v2,cuCmulf(dotp_s[0], tau) );
#endif
        }
        else
        {
#ifdef DOUBLE_PRECISION_COMPLEX
		q_v2 =  cuCsub(q_v2,cuCmul(cuCmul(dotp_s[0], tau),hh[h_off]));
#else
		q_v2 =  cuCsubf(q_v2,cuCmulf(cuCmulf(dotp_s[0], tau),hh[h_off]));
#endif
        }

        if(t_idx < blockDim.x-1)
       {q_s[t_idx+1 ] = q_v2;
        }
       if ((j ==  1) || (t_idx == blockDim.x-1))
       {q[q_off] = q_v2;
        }
       __syncthreads();
       q_off -= ldq;
       h_off -= nb;
	j -=1;
}
}
#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_compute_hh_trafo_c_kernel_complex_double( cuDoubleComplex* q, cuDoubleComplex * hh, cuDoubleComplex * hh_tau, const int nev, const int nb, const int ldq, const int off, const int ncols)
#else
extern "C" void launch_compute_hh_trafo_c_kernel_complex_single( cuFloatComplex* q, cuFloatComplex * hh, cuFloatComplex * hh_tau, const int nev, const int nb, const int ldq, const int off, const int ncols)
#endif
{

#if 0
	cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to compute_ hh_ trafo c kernel: %s, %d\n",cudaGetErrorString(err), err);
        dim3 n_block, n_thread;
	n_block = dim3(nev,1,1);
	n_thread = dim3(nb,1,1);
#endif

    switch (nb)
    {
      // attention
      case  256:
       case 128:
        case 64:
#ifdef DOUBLE_PRECISION_COMPLEX
	     compute_hh_trafo_kernel_2_2_complex_double<16><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, off, ncols);
#else
	     compute_hh_trafo_kernel_2_2_complex_single<16><<<nev, nb>>>(q, hh, hh_tau, nb, ldq, off, ncols);
#endif
            break;

        case 32:
#ifdef DOUBLE_PRECISION_COMPLEX
            compute_hh_trafo_kernel_2_2_complex_double<8><<<nev ,nb>>>(q, hh, hh_tau, nb, ldq, off, ncols);
#else
            compute_hh_trafo_kernel_2_2_complex_single<8><<<nev ,nb>>>(q, hh, hh_tau, nb, ldq, off, ncols);
#endif
            break;

        case 16:
#ifdef DOUBLE_PRECISION_COMPLEX
            compute_hh_trafo_kernel_2_2_complex_double<4><<<nev ,nb>>>(q, hh,  hh_tau, nb, ldq, off, ncols);
#else
            compute_hh_trafo_kernel_2_2_complex_single<4><<<nev ,nb>>>(q, hh,  hh_tau, nb, ldq, off, ncols);
#endif
            break;

        case 8:
#ifdef DOUBLE_PRECISION_COMPLEX
            compute_hh_trafo_kernel_2_2_complex_double<2><<<nev ,nb>>>(q, hh, hh_tau, nb, ldq, off, ncols);
#else
            compute_hh_trafo_kernel_2_2_complex_single<2><<<nev ,nb>>>(q, hh, hh_tau, nb, ldq, off, ncols);
#endif
            break;

        case 4:
#ifdef DOUBLE_PRECISION_COMPLEX
            compute_hh_trafo_kernel_2_2_complex_double<1><<<nev ,nb>>>(q, hh, hh_tau, nb, ldq, off, ncols);
#else
            compute_hh_trafo_kernel_2_2_complex_single<1><<<nev ,nb>>>(q, hh, hh_tau, nb, ldq, off, ncols);
#endif
            break;

        case 2:
        case 1:
#ifdef DOUBLE_PRECISION_COMPLEX
	    compute_hh_trafo_kernel_2_2_complex_double<0><<<nev ,nb>>>(q, hh, hh_tau, nb, ldq, off, ncols);
#else
	    compute_hh_trafo_kernel_2_2_complex_single<0><<<nev ,nb>>>(q, hh, hh_tau, nb, ldq, off, ncols);
#endif
            break;
        default:
            printf("Error: please use a power-of-2 SCALAPACK block size which is between 1 and BLOCK_CYCLIC_BLOCKSIZE.\n");
    }

#if 0
	cudaDeviceSynchronize();
	 err = cudaGetLastError();
        if ( err!= cudaSuccess)
        {
                printf("\n compute hh trafo c kernel failed  %s \n",cudaGetErrorString(err) );
        }
#endif

}


