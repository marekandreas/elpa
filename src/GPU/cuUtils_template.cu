#if 0
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
#endif


#include "config-f90.h"

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#if COMPLEXCASE == 1
#include <cuComplex.h>
#endif

#define BLOCK_CYCLIC_BLOCKSIZE 128
#define GLOBAL_STRIPE_WIDTH 256
#define WARP_SIZE 32

// Reset a reduction block
// Limitation: the thread-block size must be a divider of the reduction block's size

#if REALCASE == 1

#ifdef DOUBLE_PRECISION_REAL
__device__ void reset_shared_block_c_double ( double * s_block, int b_size)
#else
__device__ void reset_shared_block_c_single ( float * s_block, int b_size)
#endif

#endif

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
__device__ void reset_shared_block_c_complex_double ( cuDoubleComplex * s_block, int b_size)
#else
__device__ void reset_shared_block_c_complex_single ( cuFloatComplex * s_block, int b_size)
#endif
#endif

{
    int i, t_idx, s_chunk ;
    t_idx = threadIdx.x;
    s_chunk = b_size / blockDim.x;
    for(i = ((t_idx - 1) * s_chunk + 1) ; i < (t_idx * s_chunk); i++) {
#if  REALCASE == 1
      s_block[i] = 0.0 ;
#endif
#if COMPLEXCASE == 1
      s_block[i].x = 0.0 ;
      s_block[i].y = 0.0 ;
#endif
    }
    __syncthreads();
}

// Reset 2 reduction blocks without an explicit synchronization at the end
// Limitation: : the thread-block size must be a divider of the reduction block's size
#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
__device__ void reset_shared_block_pair_c_real_double( double *s_block_1, double *s_block_2, int b_size)
#else
__device__ void reset_shared_block_pair_c_real_single( float *s_block_1, float *s_block_2, int b_size)
#endif
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
__device__ void reset_shared_block_pair_c_complex_double( cuDoubleComplex *s_block_1, cuDoubleComplex *s_block_2, int b_size)
#else
__device__ void reset_shared_block_pair_c_complex_single( cuFloatComplex *s_block_1, cuFloatComplex *s_block_2, int b_size)
#endif
#endif
{
    int i, t_idx, s_chunk;

    t_idx = threadIdx.x;
    s_chunk = b_size / blockDim.x;
    for(i = ((t_idx - 1) * s_chunk + 1); i < (t_idx * s_chunk); i++)
    {
#if REALCASE == 1
        s_block_1[i] = 0.0 ;
        s_block_2[i] = 0.0 ;
#endif
#if COMPLEXCASE == 1
        s_block_1[i].x = 0.0 ;
        s_block_2[i].x= 0.0 ;
        s_block_1[i].y = 0.0 ;
        s_block_2[i].y= 0.0 ;
#endif
    }
}
// Reset a reduction block
// Limitation: the thread-block size must be a divider of the reduction block's size

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
__device__ void warp_reduce_c_real_double( double *s_block)
#else
__device__ void warp_reduce_c_real_single( float *s_block)
#endif
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
__device__ void warp_reduce_complex_double( cuDoubleComplex *s_block)
#else
__device__ void warp_reduce_complex_single( cuFloatComplex *s_block)
#endif
#endif
{
    int t_idx ;
    t_idx = threadIdx.x;
    __syncthreads();

#if REALCASE == 1
        // attention
        if (t_idx < 32)
	{
                s_block[t_idx] = s_block[t_idx] + s_block[t_idx + 32] + s_block[t_idx + 64] + s_block[t_idx + 96] ;
        if (t_idx < 8)
                s_block[t_idx] = s_block[t_idx] + s_block[t_idx + 8] + s_block[t_idx + 16] + s_block[t_idx + 24];
        if (t_idx < 4)
                s_block[t_idx] = s_block[t_idx] + s_block[t_idx + 4];
        if (t_idx < 1)
                s_block[t_idx] = s_block[t_idx] + s_block[t_idx + 1] + s_block[t_idx + 2] + s_block[t_idx + 3];
	}
#endif
#if COMPLEXCASE == 1
        // attention
	if (t_idx < 32)
        {
#ifdef DOUBLE_PRECISION_COMPLEX
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
#endif /* COMPLEXCASE == 1 */
}

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
__global__ void my_pack_c_kernel_real_double(const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, double* src, double* dst)
#else
__global__ void my_pack_c_kernel_real_single(const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, float* src, float* dst)
#endif
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void my_pack_c_kernel_complex_double(const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, cuDoubleComplex* src, cuDoubleComplex* dst)
#else
__global__ void my_pack_c_kernel_complex_single(const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, cuFloatComplex* src, cuFloatComplex* dst)
#endif
#endif
{
    int b_id, t_id ;
    int dst_ind ;
    b_id = blockIdx.y;
    t_id = threadIdx.x;

    dst_ind = b_id * stripe_width + t_id;
    if (dst_ind < max_idx)
    {
	// dimension of dst - lnev, nblk
	// dimension of src - stripe_width,a_dim2,stripe_count
#if REALCASE == 1
    	*(dst + dst_ind + (l_nev*blockIdx.x) ) = *(src + t_id + (stripe_width*(n_offset + blockIdx.x)) + ( b_id *stripe_width*a_dim2 ));
#endif
#if COMPLEXCASE == 1
	dst[dst_ind + (l_nev*blockIdx.x)].x = src[t_id + (stripe_width*(n_offset + blockIdx.x)) + ( b_id *stripe_width*a_dim2)].x;
        dst[dst_ind + (l_nev*blockIdx.x)].y = src[t_id + (stripe_width*(n_offset + blockIdx.x)) + ( b_id *stripe_width*a_dim2)].y;
#endif
     }

}

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
__global__ void  my_unpack_c_kernel_real_double( const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, double* src, double* dst)
#else
__global__ void  my_unpack_c_kernel_real_single( const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, float* src, float* dst)
#endif
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void  my_unpack_c_kernel_complex_double( const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, cuDoubleComplex* src, cuDoubleComplex* dst)
#else
__global__ void  my_unpack_c_kernel_complex_single( const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, cuFloatComplex* src, cuFloatComplex* dst)
#endif
#endif
{
    int b_id, t_id ;
    int src_ind;

    b_id = blockIdx.y;
    t_id = threadIdx.x;

    src_ind = b_id * stripe_width + t_id;
    if (src_ind < max_idx)
#if REALCASE == 1
	*(dst + (t_id + ((n_offset + blockIdx.x) * stripe_width) + (b_id * stripe_width * a_dim2 ))) = *(src + src_ind  + (blockIdx.x) *l_nev );
#endif
#if COMPLEXCASE == 1
        dst[ t_id + ((n_offset + blockIdx.x) * stripe_width) + (b_id * stripe_width * a_dim2 )].x = src[ src_ind  + (blockIdx.x) *l_nev ].x;
	dst[ t_id + ((n_offset + blockIdx.x) * stripe_width) + (b_id * stripe_width * a_dim2 )].y = src[ src_ind  + (blockIdx.x) *l_nev ].y;
#endif

}

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
__global__ void extract_hh_tau_c_kernel_real_double(double* hh, double* hh_tau, const int nbw, const int n, int val)
#else
__global__ void extract_hh_tau_c_kernel_real_single(float* hh, float* hh_tau, const int nbw, const int n, int val)
#endif
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void extract_hh_tau_c_kernel_complex_double(cuDoubleComplex* hh, cuDoubleComplex* hh_tau, const int nbw, const int n, int val)
#else
__global__ void extract_hh_tau_c_kernel_complex_single(cuFloatComplex* hh, cuFloatComplex* hh_tau, const int nbw, const int n, int val)
#endif
#endif
{
    int h_idx ;
    h_idx = (blockIdx.x) * blockDim.x + threadIdx.x;

    if (h_idx < n)
    {
	//dimension of hh - (nbw, max_blk_size)
	//dimension of hh_tau - max_blk_size
#if REALCASE == 1
        *(hh_tau + h_idx ) = *(hh +  (h_idx * nbw)) ;
#endif
#if COMPLEXCASE == 1
        hh_tau[h_idx] = hh[h_idx * nbw] ;
#endif
        //  Replace the first element in the HH reflector with 1.0 or 0.0
#if REALCASE == 1
	if( val == 0)
        *(hh + (h_idx * nbw)) = 1.0;
	else
	*(hh + (h_idx * nbw)) = 0.0;
#endif
#if COMPLEXCASE == 1
        if( val == 0)
        {
         hh[(h_idx * nbw)].x = 1.0;
	 hh[h_idx *nbw].y= 0.0;
        }
        else
        {
        hh[(h_idx * nbw)].x = 0.0;
	hh[h_idx*nbw].y =0.0;
        }
#endif
     }
}

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
__global__ void  compute_hh_dotp_c_kernel_real_double(double* hh, double* v_dot, const int nbw, const int n)
{

   __shared__ double hh_s[BLOCK_CYCLIC_BLOCKSIZE] ;
#else
__global__ void  compute_hh_dotp_c_kernel_real_single(float* hh, float* v_dot, const int nbw, const int n)
{

   __shared__ float hh_s[BLOCK_CYCLIC_BLOCKSIZE] ;
#endif
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void  compute_hh_dotp_c_kernel_complex_double(cuDoubleComplex* hh, cuDoubleComplex* v_dot, const int nbw, const int n)
{
   __shared__ cuDoubleComplex hh_s[BLOCK_CYCLIC_BLOCKSIZE] ;

#else
__global__ void  compute_hh_dotp_c_kernel_complex_single(cuFloatComplex* hh, cuFloatComplex* v_dot, const int nbw, const int n)
{
   __shared__ cuFloatComplex hh_s[BLOCK_CYCLIC_BLOCKSIZE] ;
#endif
#endif
    int t_idx, v_idx;

    //  The Vector index (v_idx) identifies the pair of HH reflectors from which the dot product is computed
    v_idx = blockIdx.x  ;

    //  The thread index indicates the position within the two HH reflectors
    t_idx = threadIdx.x ;

//    //  The contents of the shared memory must be fully reset
//     reset_shared_block_c(hh_s, BLOCK_CYCLIC_BLOCKSIZE);

    //  Initialize the contents of the shared buffer (preparing for reduction)
    if (t_idx  > 0)
    {
#if REALCASE == 1
        *(hh_s + t_idx) = *(hh + t_idx + v_idx * nbw ) *  (*(hh + (t_idx - 1) +  (v_idx +1)* nbw)) ;
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
       hh_s[t_idx] = cuCmul(cuConj(hh[t_idx + v_idx * nbw]),   hh[ (t_idx - 1) +  (v_idx +1)* nbw]) ;
#else
       hh_s[t_idx] = cuCmulf(cuConjf(hh[t_idx + v_idx * nbw]),   hh[ (t_idx - 1) +  (v_idx +1)* nbw]) ;
#endif
#endif
    }
    else
    {
#if REALCASE == 1
        *(hh_s + t_idx) = 0.0 ;
#endif
#if COMPLEXCASE == 1
        hh_s[t_idx].x = 0.0 ;
        hh_s[t_idx].y = 0.0;
#endif
    }
     //  Compute the dot product using a fast reduction

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
     warp_reduce_c_real_double(hh_s);
#else
     warp_reduce_c_real_single(hh_s);
#endif
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
     warp_reduce_complex_double(hh_s);
#else
     warp_reduce_complex_single(hh_s);
#endif
     __syncthreads();
#endif

      if(t_idx == 0)
      {
#if REALCASE == 1
      *(v_dot + v_idx) = *(hh_s) ;
#endif
#if COMPLEXCASE == 1
      v_dot[v_idx] = hh_s[0] ;
#endif
      }

}

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
extern "C" void launch_my_pack_c_kernel_real_double(const int row_count, const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, double* a_dev, double* row_group_dev)
#else
extern "C" void launch_my_pack_c_kernel_real_single(const int row_count, const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, float* a_dev, float* row_group_dev)
#endif
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_my_pack_c_kernel_complex_double(const int row_count, const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, cuDoubleComplex* a_dev, cuDoubleComplex* row_group_dev)
#else
extern "C" void launch_my_pack_c_kernel_complex_single(const int row_count, const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, cuFloatComplex* a_dev, cuFloatComplex* row_group_dev)
#endif
#endif
{

	dim3  grid_size;
        grid_size = dim3(row_count, stripe_count, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to mypack kernel: %s, %d\n",cudaGetErrorString(err), err);
#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
	my_pack_c_kernel_real_double<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev);
#else
	my_pack_c_kernel_real_single<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev);
#endif
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
        my_pack_c_kernel_complex_double<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev);
#else
        my_pack_c_kernel_complex_single<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev);
#endif
#endif
	 err = cudaGetLastError();
        if ( err!= cudaSuccess)
        {
                printf("\n my pack_kernel failed  %s \n",cudaGetErrorString(err) );
        }

}
#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
extern "C" void launch_compute_hh_dotp_c_kernel_real_double(double* bcast_buffer_dev, double* hh_dot_dev,const int nbw,const int n)
#else
extern "C" void launch_compute_hh_dotp_c_kernel_real_single(float* bcast_buffer_dev, float* hh_dot_dev,const int nbw,const int n)
#endif
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_compute_hh_dotp_c_kernel_complex_double(cuDoubleComplex* bcast_buffer_dev, cuDoubleComplex* hh_dot_dev,const int nbw,const int n)
#else
extern "C" void launch_compute_hh_dotp_c_kernel_complex_single(cuFloatComplex* bcast_buffer_dev, cuFloatComplex* hh_dot_dev,const int nbw,const int n)
#endif
#endif
{
	cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to compute_hh kernel: %s, %d\n",cudaGetErrorString(err), err);

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
        compute_hh_dotp_c_kernel_real_double<<< n-1, nbw >>>(bcast_buffer_dev, hh_dot_dev, nbw, n);
#else
        compute_hh_dotp_c_kernel_real_single<<< n-1, nbw >>>(bcast_buffer_dev, hh_dot_dev, nbw, n);
#endif
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
        compute_hh_dotp_c_kernel_complex_double<<< n-1, nbw >>>(bcast_buffer_dev, hh_dot_dev, nbw, n);
#else
        compute_hh_dotp_c_kernel_complex_single<<< n-1, nbw >>>(bcast_buffer_dev, hh_dot_dev, nbw, n);
#endif
#endif
	err = cudaGetLastError();
        if ( err!= cudaSuccess)
        {
                printf("\n compute _kernel failed  %s \n",cudaGetErrorString(err) );
        }

}
#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
extern "C" void launch_extract_hh_tau_c_kernel_real_double(double* bcast_buffer_dev, double* hh_tau_dev, const int nbw, const int n , const int is_zero)
#else
extern "C" void launch_extract_hh_tau_c_kernel_real_single(float* bcast_buffer_dev, float* hh_tau_dev, const int nbw, const int n , const int is_zero)
#endif
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_extract_hh_tau_c_kernel_complex_double(cuDoubleComplex* bcast_buffer_dev, cuDoubleComplex* hh_tau_dev, const int nbw, const int n , const int is_zero)
#else
extern "C" void launch_extract_hh_tau_c_kernel_complex_single(cuFloatComplex* bcast_buffer_dev, cuFloatComplex* hh_tau_dev, const int nbw, const int n , const int is_zero)
#endif
#endif
{
	int grid_size;
	grid_size = 1 + (n - 1) / GLOBAL_STRIPE_WIDTH;
	cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to extract kernel: %s, %d\n",cudaGetErrorString(err), err);
#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
	extract_hh_tau_c_kernel_real_double<<<grid_size,GLOBAL_STRIPE_WIDTH>>>(bcast_buffer_dev,hh_tau_dev, nbw, n, is_zero);
#else
	extract_hh_tau_c_kernel_real_single<<<grid_size,GLOBAL_STRIPE_WIDTH>>>(bcast_buffer_dev,hh_tau_dev, nbw, n, is_zero);
#endif
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
        extract_hh_tau_c_kernel_complex_double<<<grid_size,GLOBAL_STRIPE_WIDTH>>>(bcast_buffer_dev,hh_tau_dev, nbw, n, is_zero);
#else
        extract_hh_tau_c_kernel_complex_single<<<grid_size,GLOBAL_STRIPE_WIDTH>>>(bcast_buffer_dev,hh_tau_dev, nbw, n, is_zero);
#endif
#endif
	err = cudaGetLastError();
	if ( err!= cudaSuccess)
       	{
		printf("\n  extract _kernel failed  %s \n",cudaGetErrorString(err) );
        }

}

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
extern "C" void launch_my_unpack_c_kernel_real_double( const int row_count, const int n_offset, const int max_idx, const int stripe_width,const int a_dim2, const int stripe_count, const int l_nev, double* row_group_dev, double* a_dev)
#else
extern "C" void launch_my_unpack_c_kernel_real_single( const int row_count, const int n_offset, const int max_idx, const int stripe_width,const int a_dim2, const int stripe_count, const int l_nev, float* row_group_dev, float* a_dev)
#endif
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_my_unpack_c_kernel_complex_double( const int row_count, const int n_offset, const int max_idx, const int stripe_width,const int a_dim2, const int stripe_count, const int l_nev, cuDoubleComplex* row_group_dev, cuDoubleComplex* a_dev)
#else
extern "C" void launch_my_unpack_c_kernel_complex_single( const int row_count, const int n_offset, const int max_idx, const int stripe_width,const int a_dim2, const int stripe_count, const int l_nev, cuFloatComplex* row_group_dev, cuFloatComplex* a_dev)
#endif
#endif
{
        dim3  grid_size;
	grid_size = dim3(row_count, stripe_count, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to unpack kernel: %s, %d\n",cudaGetErrorString(err), err);
#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
        my_unpack_c_kernel_real_double<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, row_group_dev , a_dev);
#else
        my_unpack_c_kernel_real_single<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, row_group_dev , a_dev);
#endif
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
        my_unpack_c_kernel_complex_double<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, row_group_dev , a_dev);
#else
        my_unpack_c_kernel_complex_single<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, row_group_dev , a_dev);
#endif
#endif
        err = cudaGetLastError();
        if ( err != cudaSuccess)
        {
	    printf("\n  my_unpack_c_kernel failed  %s \n",cudaGetErrorString(err) );
        }
}

#if COMPLEXCASE == 1

#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void compute_kernel_reduce_complex_double( cuDoubleComplex* a_dev, int lda , int n ,int nbw ,  cuDoubleComplex *h1_dev )
#else
__global__ void compute_kernel_reduce_complex_single( cuFloatComplex* a_dev, int lda , int n ,int nbw ,  cuFloatComplex *h1_dev )
#endif
{
    int  t_id ;
    int st_ind;

    t_id = threadIdx.x;

    st_ind = (t_id*(t_id+1))/2;
    if(t_id< n)
    {
	for(int i =0;i<=t_id;i++)
        {
         h1_dev[st_ind + i] = a_dev[t_id *lda + i ] ;
	}
    }
    __syncthreads();
}

#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void compute_kernel_reduce_1_complex_ouble( cuDoubleComplex* a_dev, int lda , int n, cuDoubleComplex *h1_dev )
#else
__global__ void compute_kernel_reduce_1_complex_ingle( cuFloatComplex* a_dev, int lda , int n, cuFloatComplex *h1_dev )
#endif
{
    int  t_id ;
    int st_ind;

    t_id = threadIdx.x;

    st_ind = (t_id*(t_id+1))/2;
    if(t_id< n)
    {
        for(int i =0;i<=t_id;i++)
         {
	  a_dev[t_id *lda + i ] = h1_dev[st_ind + i];
#ifdef DOUBLE_PRECISION_COMPLEX
	  a_dev[ (i-1)*lda + t_id ] = cuConj(a_dev[ t_id *lda + i-1]) ;
#else
	  a_dev[ (i-1)*lda + t_id ] = cuConjf(a_dev[ t_id *lda + i-1]) ;
#endif
	}
    }
    __syncthreads();
}

#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void  dot_product_c_kernel_complex_double( cuDoubleComplex* hs_dev, cuDoubleComplex* hv_new_dev, cuDoubleComplex tau_new_dev, cuDoubleComplex*  x_dev, cuDoubleComplex *h_dev, cuDoubleComplex *hv_dev, int nr)
#else
__global__ void  dot_product_c_kernel_complex_single( cuFloatComplex* hs_dev, cuFloatComplex* hv_new_dev, cuFloatComplex tau_new_dev, cuFloatComplex*  x_dev, cuFloatComplex *h_dev, cuFloatComplex *hv_dev, int nr)
#endif
{
    int t_id ;

#ifdef DOUBLE_PRECISION_COMPLEX
    __shared__ cuDoubleComplex x_dev_temp[128];
    __shared__ cuDoubleComplex x_val;
#else
    __shared__ cuFloatComplex x_dev_temp[128];
    __shared__ cuFloatComplex x_val;
#endif
    //b_id = blockIdx.y;
    t_id = threadIdx.x;

    if(t_id<nr)
#ifdef DOUBLE_PRECISION_COMPLEX
	 x_dev_temp[t_id] = cuCmul( cuConj(hs_dev[t_id]), hv_new_dev[t_id]) ;
#else
	 x_dev_temp[t_id] = cuCmulf( cuConjf(hs_dev[t_id]), hv_new_dev[t_id]) ;
#endif
    __syncthreads();

    if(t_id==0)
    {
        for(int i=1;i<nr;i++)
#ifdef DOUBLE_PRECISION_COMPLEX
	x_dev_temp[t_id] = cuCadd(x_dev_temp[t_id],x_dev_temp[t_id +i]);
#else
	x_dev_temp[t_id] = cuCaddf(x_dev_temp[t_id],x_dev_temp[t_id +i]);
#endif
    }
    __syncthreads();
     if(t_id ==0)
    {
#ifdef DOUBLE_PRECISION_COMPLEX
      x_val =  cuCmul(x_dev_temp[t_id], tau_new_dev);
#else
      x_val =  cuCmulf(x_dev_temp[t_id], tau_new_dev);
#endif
      x_dev[0] = x_val;
    }
	__syncthreads();
}

#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void  dot_product_c_kernel_1_complex_double(   cuDoubleComplex*  ab_dev, cuDoubleComplex *hs_dev,  cuDoubleComplex*  hv_new_dev, cuDoubleComplex*  x_dev, cuDoubleComplex *h_dev, cuDoubleComplex *hv_dev,  int nb, int nr , int ns )
#else
__global__ void  dot_product_c_kernel_1_complex_single(   cuFloatComplex*  ab_dev, cuFloatComplex *hs_dev,  cuFloatComplex*  hv_new_dev, cuFloatComplex*  x_dev, cuFloatComplex *h_dev, cuFloatComplex *hv_dev,  int nb, int nr , int ns )
#endif
{
    int t_id = threadIdx.x;
        int i;

    if((t_id>0 )&& (t_id < nb))
    {
#ifdef DOUBLE_PRECISION_COMPLEX
	h_dev[t_id] = cuCsub(h_dev[t_id], cuCmul(x_dev[0],hv_dev[t_id]));
#else
	h_dev[t_id] = cuCsubf(h_dev[t_id], cuCmulf(x_dev[0],hv_dev[t_id]));
#endif
        for(i=0;i<nr;i++)
	{
#ifdef DOUBLE_PRECISION_COMPLEX
	 ab_dev[ i+nb-t_id + (t_id+ns-1)*2*nb ] = cuCsub(cuCsub(ab_dev[ i+nb-t_id + (t_id+ns-1)*2*nb],cuCmul(hv_new_dev[i],cuConj(h_dev[t_id])) ),cuCmul(hs_dev[i], cuConj(hv_dev[t_id])));
#else
	 ab_dev[ i+nb-t_id + (t_id+ns-1)*2*nb ] = cuCsubf(cuCsubf(ab_dev[ i+nb-t_id + (t_id+ns-1)*2*nb],cuCmulf(hv_new_dev[i],cuConjf(h_dev[t_id])) ),cuCmulf(hs_dev[i], cuConjf(hv_dev[t_id])));
#endif
 	}
    }
   __syncthreads();
}

#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void  double_hh_transform_kernel_complex_double( cuDoubleComplex*  ab_dev, cuDoubleComplex *hs_dev, cuDoubleComplex *hv_dev,  int nb,  int ns )
#else
__global__ void  double_hh_transform_kernel_complex_single( cuFloatComplex*  ab_dev, cuFloatComplex *hs_dev, cuFloatComplex *hv_dev,  int nb,  int ns )
#endif
{
    int t_id = threadIdx.x;
    if((t_id>0 )&& (t_id < nb))
    {
#ifdef DOUBLE_PRECISION_COMPLEX
         ab_dev[ nb-t_id + (t_id+ns-1)*2*nb ] = cuCsub(ab_dev[ nb-t_id + (t_id+ns-1)*2*nb],cuCmul(hs_dev[0], cuConj(hv_dev[t_id])));
#else
         ab_dev[ nb-t_id + (t_id+ns-1)*2*nb ] = cuCsubf(ab_dev[ nb-t_id + (t_id+ns-1)*2*nb],cuCmulf(hs_dev[0], cuConjf(hv_dev[t_id])));
#endif
    }
   __syncthreads();
}

#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void  double_hh_transform_kernel_2_complex_double( cuDoubleComplex*  ab_dev, cuDoubleComplex *hd_dev, cuDoubleComplex *hv_dev,  int nc,  int ns , int nb )
#else
__global__ void  double_hh_transform_kernel_2_complex_single( cuFloatComplex*  ab_dev, cuFloatComplex *hd_dev, cuFloatComplex *hv_dev,  int nc,  int ns , int nb )
#endif
{
    int t_id = threadIdx.x;
    if(t_id < nc)
    {
#ifdef DOUBLE_PRECISION_COMPLEX
         ab_dev[ t_id + (ns-1)*2*nb ] = cuCsub(cuCsub(ab_dev[ t_id + (ns-1)*2*nb],cuCmul(hd_dev[ t_id], cuConj(hv_dev[0]))) , cuCmul(hv_dev[ t_id], cuConj(hd_dev[0])));
#else
         ab_dev[ t_id + (ns-1)*2*nb ] = cuCsubf(cuCsubf(ab_dev[ t_id + (ns-1)*2*nb],cuCmulf(hd_dev[ t_id], cuConjf(hv_dev[0]))) , cuCmulf(hv_dev[ t_id], cuConjf(hd_dev[0])));

#endif
    }
   __syncthreads();
}

#if 0  /* not used anywhere */

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_dot_product_kernel_complex_double( cuDoubleComplex* hs_dev, cuDoubleComplex* hv_new_dev, cuDoubleComplex tau_new_dev, cuDoubleComplex*  x_dev, cuDoubleComplex*  h_dev ,cuDoubleComplex*  hv_dev,int nr )
#else
extern "C" void launch_dot_product_kernel_complex_single( cuFloatComplex* hs_dev, cuFloatComplex* hv_new_dev, cuFloatComplex tau_new_dev, cuFloatComplex*  x_dev, cuFloatComplex*  h_dev ,cuFloatComplex*  hv_dev,int nr )
#endif
{
        dim3  grid_size;
        grid_size = dim3(1,1, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to launch_dot_product_kernel_complex: %s, %d\n",cudaGetErrorString(err), err);
#ifdef DOUBLE_PRECISION_COMPLEX
        dot_product_c_kernel_complex_double<<<grid_size, nr>>>(hs_dev, hv_new_dev, tau_new_dev, x_dev, h_dev, hv_dev, nr );
#else
        dot_product_c_kernel_complex_single<<<grid_size, nr>>>(hs_dev, hv_new_dev, tau_new_dev, x_dev, h_dev, hv_dev, nr );
#endif
        err = cudaGetLastError();
        if ( err != cudaSuccess)
        {
            printf("\n dot product kernel failed  %s \n",cudaGetErrorString(err) );
        }
}
#endif /* not used anywhere */

#if 0 /* not used anywhere */

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_dot_product_kernel_1_complex_double(  cuDoubleComplex*  ab_dev, cuDoubleComplex *hs_dev,  cuDoubleComplex*  hv_new_dev,cuDoubleComplex*  x_dev, cuDoubleComplex*  h_dev ,cuDoubleComplex*  hv_dev, int nb ,int nr , int ns)
#else
extern "C" void launch_dot_product_kernel_1_complex_single(  cuFloatComplex*  ab_dev, cuFloatComplex *hs_dev,  cuFloatComplex*  hv_new_dev,cuFloatComplex*  x_dev, cuFloatComplex*  h_dev ,cuFloatComplex*  hv_dev, int nb ,int nr , int ns)
#endif
{
        dim3  grid_size;
        grid_size = dim3(1,1, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to launch_dot_product_kernel_1_complex: %s, %d\n",cudaGetErrorString(err), err);
#ifdef DOUBLE_PRECISION_COMPLEX
        dot_product_c_kernel_1_complex_double<<<grid_size, nb>>>( ab_dev, hs_dev, hv_new_dev, x_dev, h_dev, hv_dev, nb, nr, ns );
#else
        dot_product_c_kernel_1_complex_single<<<grid_size, nb>>>( ab_dev, hs_dev, hv_new_dev, x_dev, h_dev, hv_dev, nb, nr, ns );
#endif
        err = cudaGetLastError();
        if ( err != cudaSuccess)
        {
            printf("\n dot product kernel failed  %s \n",cudaGetErrorString(err) );
        }
}

#endif /* not used anywhere */

#if 0  /* not used anywhere */

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_dot_product_kernel_2_complex_double(  cuDoubleComplex*  ab_dev, cuDoubleComplex *hs_dev,  cuDoubleComplex*  hv_dev,cuDoubleComplex*  hd_dev, int nb ,int nr , int ne)
#else
extern "C" void launch_dot_product_kernel_2_complex_single(  cuFloatComplex*  ab_dev, cuFloatComplex *hs_dev,  cuFloatComplex*  hv_dev,cuFloatComplex*  hd_dev, int nb ,int nr , int ne)
#endif
{
        dim3  grid_size;
        grid_size = dim3(1,1, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to launch_dot_product_kernel_2_complex: %s, %d\n",cudaGetErrorString(err), err);
        err = cudaGetLastError();
        if ( err != cudaSuccess)
        {
            printf("\n dot product kernel failed  %s \n",cudaGetErrorString(err) );
        }
}
#endif /* not used anywhere */

#if 0 /* not used anywhere */

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_double_hh_transform_1_complex_double( cuDoubleComplex*  ab_dev, cuDoubleComplex *hs_dev,cuDoubleComplex*  hv_dev, int nb , int ns)
#else
extern "C" void launch_double_hh_transform_1_complex_single( cuFloatComplex*  ab_dev, cuFloatComplex *hs_dev,cuFloatComplex*  hv_dev, int nb , int ns)
#endif
{
        dim3  grid_size;
        grid_size = dim3(1,1, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to launch_double_hh_transform_1: %s, %d\n",cudaGetErrorString(err), err);
#ifdef DOUBLE_PRECISION_COMPLEX
        double_hh_transform_kernel_complex_double<<<grid_size, nb>>>( ab_dev, hs_dev, hv_dev, nb,  ns );
#else
        double_hh_transform_kernel_complex_single<<<grid_size, nb>>>( ab_dev, hs_dev, hv_dev, nb,  ns );
#endif
        err = cudaGetLastError();
        if ( err != cudaSuccess)
        {
            printf("\n dot product kernel failed  %s \n",cudaGetErrorString(err) );
        }
}
#endif /* not used anywhere */

#if 0  /* not used anywhere */

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_double_hh_transform_2_complex_double( cuDoubleComplex*  ab_dev, cuDoubleComplex *hd_dev,cuDoubleComplex*  hv_dev, int nc , int ns , int nb )
#else
extern "C" void launch_double_hh_transform_2_complex_single( cuFloatComplex*  ab_dev, cuFloatComplex *hd_dev,cuFloatComplex*  hv_dev, int nc , int ns , int nb )
#endif
{
        dim3  grid_size;
        grid_size = dim3(1,1, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to launch_double_hh_transform_2: %s, %d\n",cudaGetErrorString(err), err);
#ifdef DOUBLE_PRECISION_COMPLEX
        double_hh_transform_kernel_2_complex_double<<<grid_size, nc>>>( ab_dev, hd_dev, hv_dev, nc,  ns, nb );
#else
        double_hh_transform_kernel_2_complex_single<<<grid_size, nc>>>( ab_dev, hd_dev, hv_dev, nc,  ns, nb );
#endif
        err = cudaGetLastError();
        if ( err != cudaSuccess)
        {
            printf("\n dot product kernel failed  %s \n",cudaGetErrorString(err) );
        }
}
#endif /* not used anywhere */

#if 0 /* not used anywhere */
#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_compute_kernel_reduce_complex_double( cuDoubleComplex* a_dev, int lda, int n,int nbw, cuDoubleComplex* h_dev)
#else
extern "C" void launch_compute_kernel_reduce_complex_single( cuFloatComplex* a_dev, int lda, int n,int nbw, cuFloatComplex* h_dev)
#endif
{
        dim3  grid_size;
        grid_size = dim3(1,1, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to launch_compute_kernel_reduce : %s, %d\n",cudaGetErrorString(err), err);
#ifdef DOUBLE_PRECISION_COMPLEX
        compute_kernel_reduce_complex_double<<<grid_size,n>>>(a_dev, lda, n, nbw,h_dev);
#else
        compute_kernel_reduce_complex_single<<<grid_size,n>>>(a_dev, lda, n, nbw,h_dev);
#endif
	cudaDeviceSynchronize();
        err = cudaGetLastError();
        if ( err != cudaSuccess)
        {
            printf("\n dot product kernel failed  %s \n",cudaGetErrorString(err) );
        }
}
#endif /* not used anywhere */

#if 0 /* not used anywhere */

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_compute_kernel_reduce_1_complex_double( cuDoubleComplex* a_dev, int lda, int n , cuDoubleComplex* h_dev)
#else
extern "C" void launch_compute_kernel_reduce_1_complex_single( cuFloatComplex* a_dev, int lda, int n , cuFloatComplex* h_dev)
#endif
{
        dim3  grid_size;
        grid_size = dim3(1,1, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to launch_compute_kernel_reduce_1: %s, %d\n",cudaGetErrorString(err), err);
#ifdef DOUBLE_PRECISION_COMPLEX
        compute_kernel_reduce_1_complex_double<<<grid_size,n>>>(a_dev, lda, n, h_dev);
#else
        compute_kernel_reduce_1_complex_single<<<grid_size,n>>>(a_dev, lda, n, h_dev);
#endif
	cudaDeviceSynchronize();
        err = cudaGetLastError();
        if ( err != cudaSuccess)
        {
            printf("\n dot product kernel failed  %s \n",cudaGetErrorString(err) );
        }

}
#endif  /* not used anywhere */

#endif /* COMPLEXCASE == 1 */

#ifndef MEMCPY_ALREADY_DEFINED
extern "C" int cuda_MemcpyDeviceToDevice(int val)
{
      val = cudaMemcpyDeviceToDevice;
      return val;
}
#define MEMCPY_ALREADY_DEFINED 1
#endif
