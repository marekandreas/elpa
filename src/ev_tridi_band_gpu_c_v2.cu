#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cuComplex.h>
#include "config-f90.h"
// ===========================================================================================================
// Important:   due to the use of warp shuffling, the C version of the backtransformation kernel only works on
//              devices with compute capability 3.x; for older devices, please use the Fortran kernel version
// ===========================================================================================================

// Perform the equivalent of "__shfl_xor" on an 8-byte value
#ifdef DOUBLE_PRECISION_COMPLEX
static __device__ __forceinline__ double shfl_xor(double r, int mask)
#else
static __device__ __forceinline__ float shfl_xor(float r, int mask)
#endif
{
#ifdef DOUBLE_PRECISION_COMPLEX
    int hi = __shfl_xor(__double2hiint(r), mask);
    int lo = __shfl_xor(__double2loint(r), mask);

    return __hiloint2double(hi, lo);
#else
    int hi = __shfl_xor(__float2hiint(r), mask);
    int lo = __shfl_xor(__float2loint(r), mask);

    return __hiloint2float(hi, lo);
#endif
}

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
static __device__ __forceinline__ double shfl_down(double r, int offset)
#else
static __device__ __forceinline__ float shfl_down(float r, int offset)
#endif
{
#ifdef DOUBLE_PRECISION_COMPLEX
    int hi = __shfl_down(__double2hiint(r), offset);
    int lo = __shfl_down(__double2loint(r), offset);

    return __hiloint2double(hi, lo);
#else
    int hi = __shfl_down(__float2hiint(r), offset);
    int lo = __shfl_down(__float2loint(r), offset);

    return __hiloint2float(hi, lo);
#endif
}

#ifdef DOUBLE_PRECISION_COMPLEX
__device__ void warp_reduce_complex_1( cuDoubleComplex *s_block)
{
#else
__device__ void warp_reduce_complex_1( cuFloatComplex *s_block)
#endif
    int t_idx ;
    t_idx = threadIdx.x;
    __syncthreads();

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
}

#ifdef DOUBLE_PRECISION_COMPLEX
__device__ void warp_reduce_complex_2( cuDoubleComplex *s_block)
#else
__device__ void warp_reduce_complex_2( cuFloatComplex *s_block)
#endif
{
    int t_idx ;
    t_idx = threadIdx.x;
    __syncthreads();

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
}


// Perform a reduction on a warp or the first part of it
template <unsigned int REDUCE_START_OFFSET>
#ifdef DOUBLE_PRECISION_COMPLEX
__device__ __forceinline__ double warp_reduce(double r)
#else
__device__ __forceinline__ float warp_reduce(float r)
#endif
{
#pragma unroll
    for (int i = REDUCE_START_OFFSET; i >= 1; i >>= 1)
    {
        r += shfl_down(r, i);
    }

    return r;
}
template <unsigned int REDUCE_START_OFFSET>
#ifdef DOUBLE_PRECISION_COMPLEX
__device__ __forceinline__ cuDoubleComplex warp_reduce_c( cuDoubleComplex r)
#else
__device__ __forceinline__ cuFloatComplex warp_reduce_c( cuFloatComplex r)
#endif
{

#ifdef DOUBLE_PRECISION_COMPLEX
     double real = cuCreal(r);
     double imag = cuCimag(r);
#else
     float real = cuCreal(r);
     float imag = cuCimag(r);
#endif

#pragma unroll
    for (int i = REDUCE_START_OFFSET; i >= 1; i >>= 1)
    {
        real += shfl_down(real, i);
    }
#pragma unroll
    for (int i = REDUCE_START_OFFSET; i >= 1; i >>= 1)
    {
        imag += shfl_down(imag, i);
    }

#ifdef DOUBLE_PRECISION_COMPLEX
    return make_cuDoubleComplex(real,imag);
#else
    return make_cuFloatComplex(real,imag);
#endif
}


// Perform 2 reductions, using either 1 or 2 warps
template <unsigned int REDUCE_START_OFFSET, bool HAVE_2_WARPS>
#ifdef DOUBLE_PRECISION_COMPLEX
__device__ __forceinline__ void double_warp_reduce(double * dotp_s, int w_off)
#else
__device__ __forceinline__ void float_warp_reduce(float * dotp_s, int w_off)
#endif
{
    int t_idx = threadIdx.x;

    if (HAVE_2_WARPS)
    {
        // In this case, we have 2 warps, each doing 1 reduction
        if (t_idx < 64)
        {
            dotp_s[w_off + t_idx] = warp_reduce<REDUCE_START_OFFSET>(dotp_s[w_off + t_idx] + dotp_s[w_off + t_idx + 32]);
        }
    }
    else
    {
        // In this case we have 1 warp that performs both reductions
        if (t_idx < 32)
        {
            dotp_s[t_idx] = warp_reduce<REDUCE_START_OFFSET>(dotp_s[t_idx] + dotp_s[t_idx + 32]);
            dotp_s[t_idx + 64] = warp_reduce<REDUCE_START_OFFSET>(dotp_s[t_idx + 64] + dotp_s[t_idx + 96]);
        }
    }
}

template <unsigned int REDUCE_START_OFFSET, bool HAVE_2_WARPS>
#ifdef DOUBLE_PRECISION_COMPLEX
__device__ __forceinline__ void double_warp_reduce_complex(cuDoubleComplex * dotp_s, int w_off)
#else
__device__ __forceinline__ void float_warp_reduce_complex(cuFloatComplex * dotp_s, int w_off)
#endif
{
    int t_idx = threadIdx.x;

    if (HAVE_2_WARPS)
    {
        // In this case, we have 2 warps, each doing 1 reduction
        if (t_idx < 64)
        {
            dotp_s[w_off + t_idx] = warp_reduce_c<REDUCE_START_OFFSET>(cuCadd(dotp_s[w_off + t_idx] , dotp_s[w_off + t_idx + 32]));
        }
    }
    else
    {
        // In this case we have 1 warp that performs both reductions
        if (t_idx < 32)
        {
            dotp_s[t_idx] = warp_reduce_c<REDUCE_START_OFFSET>(cuCadd(dotp_s[t_idx] ,  dotp_s[t_idx + 32]));
            dotp_s[t_idx + 64] = warp_reduce_c<REDUCE_START_OFFSET>(cuCadd(dotp_s[t_idx + 64] ,  dotp_s[t_idx + 96]));
        }
    }
}


// Synchronization wrapper, removing explicit synchronization when the thread-block is at most 32 threads (1 warp) in size
template <bool MUST_SYNC>
__device__ __forceinline__ void sync_threads()
{
    if (MUST_SYNC)
    {
        __syncthreads();
    }
}

// Reset the entire contents of a shared reduction block; the thread block size must be a power-of-2
#ifdef DOUBLE_PRECISION_COMPLEX
__device__ __forceinline__ void reset_dotp_buffers(double * const __restrict__ s_block)
#else
__device__ __forceinline__ void reset_dotp_buffers(float * const __restrict__ s_block)
#endif
{
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
        int s_chunk = 128 / blockDim.x;
#ifdef DOUBLE_PRECISION_COMPLEX
        int s_chunk_size = s_chunk * sizeof(double);
#else
        int s_chunk_size = s_chunk * sizeof(float);
#endif
        // Each thread resets an equally-sized, contiguous portion of the buffer
        memset(s_block + threadIdx.x * s_chunk, 0, s_chunk_size);
    }
}

#ifdef DOUBLE_PRECISION_COMPLEX
__device__  void reset_dotp_buffers_complex( cuDoubleComplex  * const __restrict__ s_block)
#else
__device__  void reset_dotp_buffers_complex( cuFloatComplex  * const __restrict__ s_block)
#endif
{
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
        int s_chunk = 128 / blockDim.x;
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
__device__  void reset_dotp_buffers_complex_2( cuDoubleComplex  * const __restrict__ s_block)
#else
__device__  void reset_dotp_buffers_complex_2( cuFloatComplex  * const __restrict__ s_block)
#endif
{
    if (blockDim.x >= 128)
    {
        int t_idx = threadIdx.x;

        if (t_idx < 128)
        {
            s_block[t_idx].x = s_block[t_idx + 128].x = 0.0;
            s_block[t_idx].y = s_block[t_idx + 128].y = 0.0;

        }
    }
    else
    {
        int s_chunk = 256 / blockDim.x;
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

// We use templates here to avoid additional branching based on the actual size of the thread-block
template<unsigned int REDUCE_START_OFFSET, bool HAVE_2_WARPS>
#ifdef DOUBLE_PRECISION_REAL
__global__ void __launch_bounds__(128) compute_hh_trafo_c_kernel(double * const __restrict__ q, const double * const __restrict__ hh, const double * const __restrict__ hh_dot,
    const double * const __restrict__ hh_tau, const int nb, const int ldq, const int off, const int ncols)
#else
__global__ void __launch_bounds__(128) compute_hh_trafo_c_kernel(float * const __restrict__ q, const float * const __restrict__ hh, const float * const __restrict__ hh_dot,
    const float * const __restrict__ hh_tau, const int nb, const int ldq, const int off, const int ncols)
#endif

{
#ifdef DOUBLE_PRECISION_REAL
    __shared__ double dotp_s[128];
    __shared__ double q_s[129];
#else
    __shared__ float dotp_s[128];
    __shared__ float q_s[129];
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
    reset_dotp_buffers(dotp_s);

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
    sync_threads<HAVE_2_WARPS>();

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
        my_r1 += shfl_xor(my_r1, 1);
        my_r2 += shfl_xor(my_r2, 1);

        // Now both threads in a pair can write to the same reduction buffer address without race-condition issues
        dotp_s[t_s] = my_r1;
        dotp_s[t_s + 64] = my_r2;

        // Ensure the reduction buffers are fully populated
        sync_threads<HAVE_2_WARPS>();

        // Perform the 2 reductions using only the first warp (we assume the warp size is 32, valid up to CC 3.x)
#ifdef DOUBLE_PRECISION_REAL
        double_warp_reduce<REDUCE_START_OFFSET, HAVE_2_WARPS>(dotp_s, w_off);
#else
        float_warp_reduce<REDUCE_START_OFFSET, HAVE_2_WARPS>(dotp_s, w_off);
#endif
        // Ensure every thread will have access to the reduction results
        sync_threads<HAVE_2_WARPS>();

        // Each thread collects the reduction results
        s_1 = dotp_s[0];
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

        sync_threads<HAVE_2_WARPS>();

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
        my_r1 += shfl_xor(my_r1, 1);
        dotp_s[t_s] = my_r1;

        sync_threads<HAVE_2_WARPS>();

        // We perform the reduction using the first warp only
        if (t_idx < 32)
        {
            dotp_s[t_idx] = warp_reduce<REDUCE_START_OFFSET>(dotp_s[t_idx] + dotp_s[t_idx + 32]);
        }

        sync_threads<HAVE_2_WARPS>();

        // The last EV components are written to the EV matrix
        q[q_off] = q_v_1 - hh_v_1 * dotp_s[0];
    }
}

#ifdef DOUBLE_PRECISION_COMPLEX
template<unsigned int REDUCE_START_OFFSET>__global__ void compute_hh_trafo_c_kernel_complex_2_2(cuDoubleComplex * const __restrict__  q, const cuDoubleComplex  * const __restrict__   hh,   const cuDoubleComplex * const __restrict__ hh_tau, const int nb, const int ldq, const int off, const int ncols)
#else
template<unsigned int REDUCE_START_OFFSET>__global__ void compute_hh_trafo_c_kernel_complex_2_2(cuFloatComplex * const __restrict__  q, const cuFloatComplex  * const __restrict__   hh,   const cuFloatComplex * const __restrict__ hh_tau, const int nb, const int ldq, const int off, const int ncols)
#endif
{
#ifdef DOUBLE_PRECISION_COMPLEX
    __shared__ cuDoubleComplex q_s[128];
    __shared__ cuDoubleComplex dotp_s[128];

     cuDoubleComplex q_v2, tau ;
#else
    __shared__ cuFloatComplex q_s[128];
    __shared__ cuFloatComplex dotp_s[128];

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
		dotp_s[t_idx]  =  cuCmul(q_v2,cuConj( hh[h_off]));
        }
        warp_reduce_complex_1( dotp_s);

        __syncthreads();
        if(t_idx ==0)
        {
		q_v2 =  cuCsub(q_v2,cuCmul(dotp_s[0], tau) );
        }
        else
        {
		q_v2 =  cuCsub(q_v2,cuCmul(cuCmul(dotp_s[0], tau),hh[h_off]));
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

// This is a host wrapper for calling the appropriate back-transformation kernel, based on the SCALAPACK block size
#ifdef DOUBLE_PRECISION_REAL
 extern "C" void launch_compute_hh_trafo_c_kernel(double * const q, const double * const hh, const double * const hh_dot,  const double * const hh_tau, const int nev, const int nb, const int ldq, const int off, const int ncols)
#else
 extern "C" void launch_compute_hh_trafo_c_kernel(float * const q, const float * const hh, const float * const hh_dot,  const float * const hh_tau, const int nev, const int nb, const int ldq, const int off, const int ncols)
#endif
{
    switch (nb)
    {
        case 128:
        case 64:
            compute_hh_trafo_c_kernel<16, true><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
            break;

        case 32:
            compute_hh_trafo_c_kernel<8, false><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
            break;

        case 16:
            compute_hh_trafo_c_kernel<4, false><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
            break;

        case 8:
            compute_hh_trafo_c_kernel<2, false><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
            break;

        case 4:
            compute_hh_trafo_c_kernel<1, false><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
            break;

        case 2:
        case 1:
            compute_hh_trafo_c_kernel<0, false><<<nev, nb>>>(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols);
            break;

        default:
            printf("Error: please use a power-of-2 SCALAPACK block size which is between 1 and 128.\n");
    }
}

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_compute_hh_trafo_c_kernel_complex( cuDoubleComplex* q, cuDoubleComplex * hh, cuDoubleComplex * hh_tau, const int nev, const int nb, const int ldq, const int off, const int ncols)
#else
extern "C" void launch_compute_hh_trafo_c_kernel_complex( cuFloatComplex* q, cuFloatComplex * hh, cuFloatComplex * hh_tau, const int nev, const int nb, const int ldq, const int off, const int ncols)
#endif
{

	cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to compute_ hh_ trafo c kernel: %s, %d\n",cudaGetErrorString(err), err);
        dim3 n_block,n_thread;
	n_block = dim3(nev,1,1);
	n_thread = dim3(nb,1,1);

    switch (nb)
    {
      case  256:
       case 128:
        case 64:
	     compute_hh_trafo_c_kernel_complex_2_2<16><<<n_block, n_thread>>>(q, hh, hh_tau, nb, ldq, off, ncols);
            break;

        case 32:
            compute_hh_trafo_c_kernel_complex_2_2<8><<<n_block ,n_thread>>>(q, hh, hh_tau, nb, ldq, off, ncols);
            break;

        case 16:
            compute_hh_trafo_c_kernel_complex_2_2<4><<<n_block ,n_thread>>>(q, hh,  hh_tau, nb, ldq, off, ncols);
            break;

        case 8:
            compute_hh_trafo_c_kernel_complex_2_2<2><<<n_block ,n_thread>>>(q, hh, hh_tau, nb, ldq, off, ncols);
            break;

        case 4:
            compute_hh_trafo_c_kernel_complex_2_2<1><<<n_block ,n_thread>>>(q, hh, hh_tau, nb, ldq, off, ncols);
            break;

        case 2:
        case 1:
	    compute_hh_trafo_c_kernel_complex_2_2<0><<<n_block ,n_thread>>>(q, hh, hh_tau, nb, ldq, off, ncols);
            break;
        default:
            printf("Error: please use a power-of-2 SCALAPACK block size which is between 1 and 128.\n");
    }

	cudaDeviceSynchronize();
	 err = cudaGetLastError();
        if ( err!= cudaSuccess)
        {
                printf("\n compute hh trafo c kernel failed  %s \n",cudaGetErrorString(err) );
        }


}


