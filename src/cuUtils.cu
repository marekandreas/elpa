#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuComplex.h>
#include "config-f90.h"

// Reset a reduction block
// Limitation: the thread-block size must be a divider of the reduction block's size
#ifdef DOUBLE_PRECISION_REAL
__device__ void reset_shared_block_c ( double * s_block, int b_size)
#else
__device__ void reset_shared_block_c ( float * s_block, int b_size)
#endif
{
    int i, t_idx, s_chunk ;
    t_idx = threadIdx.x;
    s_chunk = b_size / blockDim.x;
    for(i = ((t_idx - 1) * s_chunk + 1) ; i < (t_idx * s_chunk); i++)
        s_block[i] = 0.0 ;
    __syncthreads();
}

// Reset 2 reduction blocks without an explicit synchronization at the end
// Limitation: : the thread-block size must be a divider of the reduction block's size
#ifdef DOUBLE_PRECISION_REAL
__device__ void reset_shared_block_pair_c( double *s_block_1, double *s_block_2, int b_size)
#else
__device__ void reset_shared_block_pair_c( float *s_block_1, float *s_block_2, int b_size)
#endif
{
    int i, t_idx, s_chunk;

    t_idx = threadIdx.x;
    s_chunk = b_size / blockDim.x;
    for(i = ((t_idx - 1) * s_chunk + 1); i < (t_idx * s_chunk); i++)
    {    s_block_1[i] = 0.0 ;
        s_block_2[i] = 0.0 ;
    }
}
// Reset a reduction block
// Limitation: the thread-block size must be a divider of the reduction block's size
#ifdef DOUBLE_PRECISION_COMPLEX
__device__ void reset_shared_block_c_complex ( cuDoubleComplex * s_block, int b_size)
#else
__device__ void reset_shared_block_c_complex ( cuFloatComplex * s_block, int b_size)
#endif
{
    int i, t_idx, s_chunk ;
    t_idx = threadIdx.x;
    s_chunk = b_size / blockDim.x;
    for(i = ((t_idx - 1) * s_chunk + 1) ; i < (t_idx * s_chunk); i++)
       { s_block[i].x = 0.0 ;
        s_block[i].y = 0.0 ;}
    __syncthreads();
}

// Reset 2 reduction blocks without an explicit synchronization at the end
// Limitation: : the thread-block size must be a divider of the reduction block's size
#ifdef DOUBLE_PRECISION_COMPLEX
__device__ void reset_shared_block_pair_c_complex( cuDoubleComplex *s_block_1, cuDoubleComplex *s_block_2, int b_size)
#else
__device__ void reset_shared_block_pair_c_complex( cuFloatComplex *s_block_1, cuFloatComplex *s_block_2, int b_size)
#endif
{
    int i, t_idx, s_chunk;

    t_idx = threadIdx.x;
    s_chunk = b_size / blockDim.x;
    for(i = ((t_idx - 1) * s_chunk + 1); i < (t_idx * s_chunk); i++)
    {    s_block_1[i].x = 0.0 ;
        s_block_2[i].x= 0.0 ;
        s_block_1[i].y = 0.0 ;
        s_block_2[i].y= 0.0 ;
    }
}
#ifdef DOUBLE_PRECISION_COMPLEX
__device__ void warp_reduce_complex( cuDoubleComplex *s_block)
#else
__device__ void warp_reduce_complex( cuFloatComplex *s_block)
#endif
{
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
__global__ void my_pack_c_kernel_complex(const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, cuDoubleComplex* src, cuDoubleComplex* dst)
#else
__global__ void my_pack_c_kernel_complex(const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, cuFloatComplex* src, cuFloatComplex* dst)
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
	dst[dst_ind + (l_nev*blockIdx.x)].x = src[t_id + (stripe_width*(n_offset + blockIdx.x)) + ( b_id *stripe_width*a_dim2)].x;
        dst[dst_ind + (l_nev*blockIdx.x)].y = src[t_id + (stripe_width*(n_offset + blockIdx.x)) + ( b_id *stripe_width*a_dim2)].y;
     }

}

#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void  my_unpack_c_kernel_complex( const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, cuDoubleComplex* src, cuDoubleComplex* dst)
#else
__global__ void  my_unpack_c_kernel_complex( const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, cuFloatComplex* src, cuFloatComplex* dst)
#endif
{
    int b_id, t_id ;
    int src_ind;

    b_id = blockIdx.y;
    t_id = threadIdx.x;

    src_ind = b_id * stripe_width + t_id;
    if (src_ind < max_idx)
{	dst[ t_id + ((n_offset + blockIdx.x) * stripe_width) + (b_id * stripe_width * a_dim2 )].x = src[ src_ind  + (blockIdx.x) *l_nev ].x;
	dst[ t_id + ((n_offset + blockIdx.x) * stripe_width) + (b_id * stripe_width * a_dim2 )].y = src[ src_ind  + (blockIdx.x) *l_nev ].y;
}
}

#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void extract_hh_tau_c_kernel_complex(cuDoubleComplex* hh, cuDoubleComplex* hh_tau, const int nbw, const int n, int val)
#else
__global__ void extract_hh_tau_c_kernel_complex(cuFloatComplex* hh, cuFloatComplex* hh_tau, const int nbw, const int n, int val)
#endif
{
    int h_idx ;

    h_idx = (blockIdx.x) * blockDim.x + threadIdx.x;

    if (h_idx < n)
    {
        //dimension of hh - (nbw, max_blk_size)
        //dimension of hh_tau - max_blk_size
        hh_tau[h_idx] = hh[h_idx * nbw] ;
        //  Replace the first element in the HH reflector with 1.0 or 0.0
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
     }
}

#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void  compute_hh_dotp_c_kernel_complex(cuDoubleComplex* hh, cuDoubleComplex* v_dot, const int nbw, const int n)
{
   __shared__ cuDoubleComplex hh_s[128] ;
#else
__global__ void  compute_hh_dotp_c_kernel_complex(cuFloatComplex* hh, cuFloatComplex* v_dot, const int nbw, const int n)
{
   __shared__ cuFloatComplex hh_s[128] ;
#endif

    int t_idx, v_idx;

    //  The vector index (v_idx) identifies the pair of HH reflectors from which the dot product is computed
    v_idx = blockIdx.x  ;

    //  The thread index indicates the position within the two HH reflectors
    t_idx = threadIdx.x ;

    if (t_idx  > 0)
    {

       hh_s[t_idx] = cuCmul(cuConj(hh[t_idx + v_idx * nbw]),   hh[ (t_idx - 1) +  (v_idx +1)* nbw]) ;
    }
    else
    {
        hh_s[t_idx].x = 0.0 ;
        hh_s[t_idx].y = 0.0;
    }

  //  Compute the dot product using a fast reduction
     warp_reduce_complex(hh_s);
     __syncthreads();

      if(t_idx == 0)
       {
	v_dot[v_idx] = hh_s[0] ;
	}

}

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_my_pack_c_kernel_complex(const int row_count, const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, cuDoubleComplex* a_dev, cuDoubleComplex* row_group_dev)
#else
extern "C" void launch_my_pack_c_kernel_complex(const int row_count, const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, cuFloatComplex* a_dev, cuFloatComplex* row_group_dev)
#endif
{

        dim3  grid_size;
        grid_size = dim3(row_count, stripe_count, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to mypack kernel: %s, %d\n",cudaGetErrorString(err), err);
        my_pack_c_kernel_complex<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev);
        err = cudaGetLastError();
        if ( err!= cudaSuccess)
        {
                printf("\n my pack_kernel failed  %s \n",cudaGetErrorString(err) );
        }
}

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_compute_hh_dotp_c_kernel_complex(cuDoubleComplex* bcast_buffer_dev, cuDoubleComplex* hh_dot_dev,const int nbw,const int n)
#else
extern "C" void launch_compute_hh_dotp_c_kernel_complex(cuFloatComplex* bcast_buffer_dev, cuFloatComplex* hh_dot_dev,const int nbw,const int n)
#endif
{
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to compute_hh kernel: %s, %d\n",cudaGetErrorString(err), err);
        compute_hh_dotp_c_kernel_complex<<< n-1, nbw >>>(bcast_buffer_dev, hh_dot_dev, nbw, n);

        err = cudaGetLastError();
        if ( err!= cudaSuccess)
        {
                printf("\n compute _kernel failed  %s \n",cudaGetErrorString(err) );
        }
}

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_extract_hh_tau_c_kernel_complex(cuDoubleComplex* bcast_buffer_dev, cuDoubleComplex* hh_tau_dev, const int nbw, const int n , const int is_zero)
#else
extern "C" void launch_extract_hh_tau_c_kernel_complex(cuFloatComplex* bcast_buffer_dev, cuFloatComplex* hh_tau_dev, const int nbw, const int n , const int is_zero)
#endif
{
        int grid_size;
        grid_size = 1 + (n - 1) / 256;
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to extract kernel: %s, %d\n",cudaGetErrorString(err), err);
        extract_hh_tau_c_kernel_complex<<<grid_size,256>>>(bcast_buffer_dev,hh_tau_dev, nbw, n, is_zero);
        err = cudaGetLastError();
        if ( err!= cudaSuccess)
        {
                printf("\n  extract _kernel failed  %s \n",cudaGetErrorString(err) );
        }

}

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_my_unpack_c_kernel_complex( const int row_count, const int n_offset, const int max_idx, const int stripe_width,const int a_dim2, const int stripe_count, const int l_nev, cuDoubleComplex* row_group_dev, cuDoubleComplex* a_dev)
#else 
extern "C" void launch_my_unpack_c_kernel_complex( const int row_count, const int n_offset, const int max_idx, const int stripe_width,const int a_dim2, const int stripe_count, const int l_nev, cuFloatComplex* row_group_dev, cuFloatComplex* a_dev)
#endif
{

        dim3  grid_size;
        grid_size = dim3(row_count, stripe_count, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to unpack kernel: %s, %d\n",cudaGetErrorString(err), err);
        my_unpack_c_kernel_complex<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, row_group_dev , a_dev);
        err = cudaGetLastError();
        if ( err != cudaSuccess)
        {
            printf("\n  my_unpack_c_kernel failed  %s \n",cudaGetErrorString(err) );
        }
}

#ifdef DOUBLE_PRECISION_REAL
__device__ void warp_reduce_c( double *s_block)
#else
__device__ void warp_reduce_c( float *s_block)
#endif
{
    int t_idx ;
    t_idx = threadIdx.x;
    __syncthreads();

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
}

#ifdef DOUBLE_PRECISION_REAL
__global__ void my_pack_c_kernel(const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, double* src, double* dst)
#else
__global__ void my_pack_c_kernel(const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, float* src, float* dst)
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
    	*(dst + dst_ind + (l_nev*blockIdx.x) ) = *(src + t_id + (stripe_width*(n_offset + blockIdx.x)) + ( b_id *stripe_width*a_dim2 ));
     }

}

#ifdef DOUBLE_PRECISION_REAL
__global__ void  my_unpack_c_kernel( const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, double* src, double* dst)
#else
__global__ void  my_unpack_c_kernel( const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, float* src, float* dst)
#endif
{
    int b_id, t_id ;
    int src_ind;

    b_id = blockIdx.y;
    t_id = threadIdx.x;

    src_ind = b_id * stripe_width + t_id;
    if (src_ind < max_idx)
	*(dst + (t_id + ((n_offset + blockIdx.x) * stripe_width) + (b_id * stripe_width * a_dim2 ))) = *(src + src_ind  + (blockIdx.x) *l_nev );

}
#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void compute_kernel_reduce( cuDoubleComplex* a_dev, int lda , int n ,int nbw ,  cuDoubleComplex *h1_dev )
#else
__global__ void compute_kernel_reduce( cuFloatComplex* a_dev, int lda , int n ,int nbw ,  cuFloatComplex *h1_dev )
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
__global__ void compute_kernel_reduce_1( cuDoubleComplex* a_dev, int lda , int n, cuDoubleComplex *h1_dev )
#else
__global__ void compute_kernel_reduce_1( cuFloatComplex* a_dev, int lda , int n, cuFloatComplex *h1_dev )
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
	  a_dev[ (i-1)*lda + t_id ] = cuConj(a_dev[ t_id *lda + i-1]) ;
	}
    }
    __syncthreads();


}

#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void  dot_product_c_kernel( cuDoubleComplex* hs_dev, cuDoubleComplex* hv_new_dev, cuDoubleComplex tau_new_dev, cuDoubleComplex*  x_dev, cuDoubleComplex *h_dev, cuDoubleComplex *hv_dev, int nr)
#else
__global__ void  dot_product_c_kernel( cuFloatComplex* hs_dev, cuFloatComplex* hv_new_dev, cuFloatComplex tau_new_dev, cuFloatComplex*  x_dev, cuFloatComplex *h_dev, cuFloatComplex *hv_dev, int nr)
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
	 x_dev_temp[t_id] = cuCmul( cuConj(hs_dev[t_id]), hv_new_dev[t_id]) ;
    __syncthreads();

    if(t_id==0)
    {
        for(int i=1;i<nr;i++)
	x_dev_temp[t_id] = cuCadd(x_dev_temp[t_id],x_dev_temp[t_id +i]);
    }
    __syncthreads();
     if(t_id ==0)
    {
      x_val =  cuCmul(x_dev_temp[t_id], tau_new_dev);
      x_dev[0] = x_val;
    }
	__syncthreads();
}

#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void  dot_product_c_kernel_1(   cuDoubleComplex*  ab_dev, cuDoubleComplex *hs_dev,  cuDoubleComplex*  hv_new_dev, cuDoubleComplex*  x_dev, cuDoubleComplex *h_dev, cuDoubleComplex *hv_dev,  int nb, int nr , int ns )
#else
__global__ void  dot_product_c_kernel_1(   cuFloatComplex*  ab_dev, cuFloatComplex *hs_dev,  cuFloatComplex*  hv_new_dev, cuFloatComplex*  x_dev, cuFloatComplex *h_dev, cuFloatComplex *hv_dev,  int nb, int nr , int ns )
#endif
{
    int t_id = threadIdx.x;
        int i;

    if((t_id>0 )&& (t_id < nb))
    {
	h_dev[t_id] = cuCsub(h_dev[t_id], cuCmul(x_dev[0],hv_dev[t_id]));
        for(i=0;i<nr;i++)
	{
	 ab_dev[ i+nb-t_id + (t_id+ns-1)*2*nb ] = cuCsub(cuCsub(ab_dev[ i+nb-t_id + (t_id+ns-1)*2*nb],cuCmul(hv_new_dev[i],cuConj(h_dev[t_id])) ),cuCmul(hs_dev[i], cuConj(hv_dev[t_id])));
 	}
    }

   __syncthreads();

}

#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void  double_hh_transform_kernel( cuDoubleComplex*  ab_dev, cuDoubleComplex *hs_dev, cuDoubleComplex *hv_dev,  int nb,  int ns )
#else
__global__ void  double_hh_transform_kernel( cuFloatComplex*  ab_dev, cuFloatComplex *hs_dev, cuFloatComplex *hv_dev,  int nb,  int ns )
#endif
{
    int t_id = threadIdx.x;
    if((t_id>0 )&& (t_id < nb))
    {
         ab_dev[ nb-t_id + (t_id+ns-1)*2*nb ] = cuCsub(ab_dev[ nb-t_id + (t_id+ns-1)*2*nb],cuCmul(hs_dev[0], cuConj(hv_dev[t_id])));
    }

   __syncthreads();

}

#ifdef DOUBLE_PRECISION_COMPLEX
__global__ void  double_hh_transform_kernel_2( cuDoubleComplex*  ab_dev, cuDoubleComplex *hd_dev, cuDoubleComplex *hv_dev,  int nc,  int ns , int nb )
#else
__global__ void  double_hh_transform_kernel_2( cuFloatComplex*  ab_dev, cuFloatComplex *hd_dev, cuFloatComplex *hv_dev,  int nc,  int ns , int nb )
#endif
{
    int t_id = threadIdx.x;
    if(t_id < nc)
    {

         ab_dev[ t_id + (ns-1)*2*nb ] = cuCsub(cuCsub(ab_dev[ t_id + (ns-1)*2*nb],cuCmul(hd_dev[ t_id], cuConj(hv_dev[0]))) , cuCmul(hv_dev[ t_id], cuConj(hd_dev[0])));

    }

   __syncthreads();

}

#ifdef DOUBLE_PRECISION_REAL
__global__ void extract_hh_tau_c_kernel(double* hh, double* hh_tau, const int nbw, const int n, int val)
#else
__global__ void extract_hh_tau_c_kernel(float* hh, float* hh_tau, const int nbw, const int n, int val)
#endif
{
    int h_idx ;
    h_idx = (blockIdx.x) * blockDim.x + threadIdx.x;

    if (h_idx < n)
    {
	//dimension of hh - (nbw, max_blk_size)
	//dimension of hh_tau - max_blk_size
        *(hh_tau + h_idx ) = *(hh +  (h_idx * nbw)) ;
        //  Replace the first element in the HH reflector with 1.0 or 0.0
	if( val == 0)
        *(hh + (h_idx * nbw)) = 1.0;
	else
	*(hh + (h_idx * nbw)) = 0.0;
     }
}

#ifdef DOUBLE_PRECISION_REAL
__global__ void  compute_hh_dotp_c_kernel(double* hh, double* v_dot, const int nbw, const int n)
{

   __shared__ double hh_s[128] ;
#else
__global__ void  compute_hh_dotp_c_kernel(float* hh, float* v_dot, const int nbw, const int n)
{

   __shared__ float hh_s[128] ;
#endif
    int t_idx, v_idx;

    //  The vector index (v_idx) identifies the pair of HH reflectors from which the dot product is computed
    v_idx = blockIdx.x  ;

    //  The thread index indicates the position within the two HH reflectors
    t_idx = threadIdx.x ;

//    //  The contents of the shared memory must be fully reset
//     reset_shared_block_c(hh_s, 128);

    //  Initialize the contents of the shared buffer (preparing for reduction)
    if (t_idx  > 0)
        *(hh_s + t_idx) = *(hh + t_idx + v_idx * nbw ) *  (*(hh + (t_idx - 1) +  (v_idx +1)* nbw)) ;
    else
        *(hh_s + t_idx) = 0.0 ;

     //  Compute the dot product using a fast reduction
     warp_reduce_c(hh_s);

      if(t_idx == 0)
      *(v_dot + v_idx) = *(hh_s) ;

}

#ifdef DOUBLE_PRECISION_REAL
extern "C" void launch_my_pack_c_kernel(const int row_count, const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, double* a_dev, double* row_group_dev)
#else
extern "C" void launch_my_pack_c_kernel(const int row_count, const int n_offset, const int max_idx, const int stripe_width, const int a_dim2, const int stripe_count, const int l_nev, float* a_dev, float* row_group_dev)
#endif
{

	dim3  grid_size;
        grid_size = dim3(row_count, stripe_count, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to mypack kernel: %s, %d\n",cudaGetErrorString(err), err);

	my_pack_c_kernel<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev);
	 err = cudaGetLastError();
        if ( err!= cudaSuccess)
        {
                printf("\n my pack_kernel failed  %s \n",cudaGetErrorString(err) );
        }

}
#ifdef DOUBLE_PRECISION_REAL
extern "C" void launch_compute_hh_dotp_c_kernel(double* bcast_buffer_dev, double* hh_dot_dev,const int nbw,const int n)
#else
extern "C" void launch_compute_hh_dotp_c_kernel(float* bcast_buffer_dev, float* hh_dot_dev,const int nbw,const int n)
#endif
{
	cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to compute_hh kernel: %s, %d\n",cudaGetErrorString(err), err);
        compute_hh_dotp_c_kernel<<< n-1, nbw >>>(bcast_buffer_dev, hh_dot_dev, nbw, n);
	err = cudaGetLastError();
        if ( err!= cudaSuccess)
        {
                printf("\n compute _kernel failed  %s \n",cudaGetErrorString(err) );
        }

}
#ifdef DOUBLE_PRECISION_REAL
extern "C" void launch_extract_hh_tau_c_kernel(double* bcast_buffer_dev, double* hh_tau_dev, const int nbw, const int n , const int is_zero)
#else
extern "C" void launch_extract_hh_tau_c_kernel(float* bcast_buffer_dev, float* hh_tau_dev, const int nbw, const int n , const int is_zero)
#endif
{
	int grid_size;
	grid_size = 1 + (n - 1) / 256;
	cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to extract kernel: %s, %d\n",cudaGetErrorString(err), err);
	extract_hh_tau_c_kernel<<<grid_size,256>>>(bcast_buffer_dev,hh_tau_dev, nbw, n, is_zero);
	err = cudaGetLastError();
	if ( err!= cudaSuccess)
       	{
		printf("\n  extract _kernel failed  %s \n",cudaGetErrorString(err) );
        }

}

#ifdef DOUBLE_PRECISION_REAL
extern "C" void launch_my_unpack_c_kernel( const int row_count, const int n_offset, const int max_idx, const int stripe_width,const int a_dim2, const int stripe_count, const int l_nev, double* row_group_dev, double* a_dev)
#else
extern "C" void launch_my_unpack_c_kernel( const int row_count, const int n_offset, const int max_idx, const int stripe_width,const int a_dim2, const int stripe_count, const int l_nev, float* row_group_dev, float* a_dev)
#endif
{

        dim3  grid_size;
	grid_size = dim3(row_count, stripe_count, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to unpack kernel: %s, %d\n",cudaGetErrorString(err), err);
        my_unpack_c_kernel<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, row_group_dev , a_dev);
        err = cudaGetLastError();
        if ( err != cudaSuccess)
        {
	    printf("\n  my_unpack_c_kernel failed  %s \n",cudaGetErrorString(err) );
        }
}

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_dot_product_kernel( cuDoubleComplex* hs_dev, cuDoubleComplex* hv_new_dev, cuDoubleComplex tau_new_dev, cuDoubleComplex*  x_dev, cuDoubleComplex*  h_dev ,cuDoubleComplex*  hv_dev,int nr )
#else
extern "C" void launch_dot_product_kernel( cuFloatComplex* hs_dev, cuFloatComplex* hv_new_dev, cuFloatComplex tau_new_dev, cuFloatComplex*  x_dev, cuFloatComplex*  h_dev ,cuFloatComplex*  hv_dev,int nr )
#endif
{

        dim3  grid_size;
        grid_size = dim3(1,1, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to launch_dot_product kernel: %s, %d\n",cudaGetErrorString(err), err);
        dot_product_c_kernel<<<grid_size, nr>>>(hs_dev, hv_new_dev, tau_new_dev, x_dev, h_dev, hv_dev, nr );
        err = cudaGetLastError();
        if ( err != cudaSuccess)
        {
            printf("\n dot product kernel failed  %s \n",cudaGetErrorString(err) );

        }

}

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_dot_product_kernel_1(  cuDoubleComplex*  ab_dev, cuDoubleComplex *hs_dev,  cuDoubleComplex*  hv_new_dev,cuDoubleComplex*  x_dev, cuDoubleComplex*  h_dev ,cuDoubleComplex*  hv_dev, int nb ,int nr , int ns)
#else
extern "C" void launch_dot_product_kernel_1(  cuFloatComplex*  ab_dev, cuFloatComplex *hs_dev,  cuFloatComplex*  hv_new_dev,cuFloatComplex*  x_dev, cuFloatComplex*  h_dev ,cuFloatComplex*  hv_dev, int nb ,int nr , int ns)
#endif
{
        dim3  grid_size;
        grid_size = dim3(1,1, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to launch_dot_product kernel: %s, %d\n",cudaGetErrorString(err), err);
        dot_product_c_kernel_1<<<grid_size, nb>>>( ab_dev, hs_dev, hv_new_dev, x_dev, h_dev, hv_dev, nb, nr, ns );
        err = cudaGetLastError();
        if ( err != cudaSuccess)
        {
            printf("\n dot product kernel failed  %s \n",cudaGetErrorString(err) );

        }

}

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_dot_product_kernel_2(  cuDoubleComplex*  ab_dev, cuDoubleComplex *hs_dev,  cuDoubleComplex*  hv_dev,cuDoubleComplex*  hd_dev, int nb ,int nr , int ne)
#else
extern "C" void launch_dot_product_kernel_2(  cuFloatComplex*  ab_dev, cuFloatComplex *hs_dev,  cuFloatComplex*  hv_dev,cuFloatComplex*  hd_dev, int nb ,int nr , int ne)
#endif
{
        dim3  grid_size;
        grid_size = dim3(1,1, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to launch_dot_product kernel: %s, %d\n",cudaGetErrorString(err), err);
        err = cudaGetLastError();
        if ( err != cudaSuccess)
        {
            printf("\n dot product kernel failed  %s \n",cudaGetErrorString(err) );

        }

}

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_double_hh_transform_1( cuDoubleComplex*  ab_dev, cuDoubleComplex *hs_dev,cuDoubleComplex*  hv_dev, int nb , int ns)
#else
extern "C" void launch_double_hh_transform_1( cuFloatComplex*  ab_dev, cuFloatComplex *hs_dev,cuFloatComplex*  hv_dev, int nb , int ns)
#endif
{
        dim3  grid_size;
        grid_size = dim3(1,1, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to launch_double_hh_transform kernel: %s, %d\n",cudaGetErrorString(err), err);
        double_hh_transform_kernel<<<grid_size, nb>>>( ab_dev, hs_dev, hv_dev, nb,  ns );
        err = cudaGetLastError();
        if ( err != cudaSuccess)
        {
            printf("\n dot product kernel failed  %s \n",cudaGetErrorString(err) );

        }

}

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_double_hh_transform_2( cuDoubleComplex*  ab_dev, cuDoubleComplex *hd_dev,cuDoubleComplex*  hv_dev, int nc , int ns , int nb )
#else
extern "C" void launch_double_hh_transform_2( cuFloatComplex*  ab_dev, cuFloatComplex *hd_dev,cuFloatComplex*  hv_dev, int nc , int ns , int nb )
#endif
{
        dim3  grid_size;
        grid_size = dim3(1,1, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to launch_double_hh_transform kernel: %s, %d\n",cudaGetErrorString(err), err);
        double_hh_transform_kernel_2<<<grid_size, nc>>>( ab_dev, hd_dev, hv_dev, nc,  ns, nb );
        err = cudaGetLastError();
        if ( err != cudaSuccess)
        {
            printf("\n dot product kernel failed  %s \n",cudaGetErrorString(err) );

        }

}

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_compute_kernel_reduce( cuDoubleComplex* a_dev, int lda, int n,int nbw, cuDoubleComplex* h_dev)
#else
extern "C" void launch_compute_kernel_reduce( cuFloatComplex* a_dev, int lda, int n,int nbw, cuFloatComplex* h_dev)
#endif
{

        dim3  grid_size;
        grid_size = dim3(1,1, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to launch_dot_product kernel: %s, %d\n",cudaGetErrorString(err), err);
        compute_kernel_reduce<<<grid_size,n>>>(a_dev, lda, n, nbw,h_dev);
	cudaDeviceSynchronize();
        err = cudaGetLastError();
        if ( err != cudaSuccess)
        {
            printf("\n dot product kernel failed  %s \n",cudaGetErrorString(err) );

        }

}

#ifdef DOUBLE_PRECISION_COMPLEX
extern "C" void launch_compute_kernel_reduce_1( cuDoubleComplex* a_dev, int lda, int n , cuDoubleComplex* h_dev)
#else
extern "C" void launch_compute_kernel_reduce_1( cuFloatComplex* a_dev, int lda, int n , cuFloatComplex* h_dev)
#endif
{

        dim3  grid_size;
        grid_size = dim3(1,1, 1);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) printf("error prior to launch_dot_product kernel: %s, %d\n",cudaGetErrorString(err), err);
        compute_kernel_reduce_1<<<grid_size,n>>>(a_dev, lda, n, h_dev);
	cudaDeviceSynchronize();
        err = cudaGetLastError();
        if ( err != cudaSuccess)
        {
            printf("\n dot product kernel failed  %s \n",cudaGetErrorString(err) );

        }

}


extern "C" int cuda_MemcpyDeviceToDevice(int val)
{
      val = cudaMemcpyDeviceToDevice;
      return val;
}
