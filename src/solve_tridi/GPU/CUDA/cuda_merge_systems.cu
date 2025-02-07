//    Copyright 2024, A. Marek
//
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
//    This file was written by A. Marek, MPCDF

#include "config-f90.h"

#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <alloca.h>
#include <complex.h>
#include <cuComplex.h>
#include <stdint.h>
#include "config-f90.h"

#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)


__global__ void cuda_fill_tmp_arrays_double_kernel(int *idx1, int *p_col, int *coltyp, int *nnzu_val, int *nnzl_val, int *nnzul, double *d1u, double *d1, double *zu, double *z, double *d1l, double *zl, const int na, const int np, const int na1, const int np_rem) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=0 && i < na1) {
      int index = idx1[i] - 1;
      if (p_col[index] == np_rem) {
        if ((coltyp[index] == 1) || (coltyp[index] == 2)) {
	  int nnzu = nnzu_val[(i) + na1 * (np-1)] -1 ;
	  d1u[nnzu] = d1[i];
	   zu[nnzu] =  z[i];
	}
        if ((coltyp[index] == 3) || (coltyp[index] == 2)) {
	  int nnzl = nnzl_val[(i) + na1 * (np-1)]-1;
	  d1l[nnzl] = d1[i];
	  zl[nnzl] =  z[i];
	}
      }
    }

    __syncthreads();
    if (i == 0) {
      nnzul[0]=0;
      nnzul[1]=0;
      int nnzu = 0;
      int nnzl = 0;
      for (int ii=na1-1;ii>=0;ii--) {
        int index = idx1[ii] - 1;
        if (nnzu == 0) {
          if (p_col[index] == np_rem) {
            if ((coltyp[index] == 1) || (coltyp[index] == 2)) {
	      nnzu = nnzu_val[(ii) + na1 * (np-1)] ;
	      nnzul[0] = nnzu;
	    }
	  }
        }
        if (nnzl == 0) {
          if (p_col[index] == np_rem) {
            if ((coltyp[index] == 3) || (coltyp[index] == 2)) {
	      nnzl = nnzl_val[(ii) + na1 * (np-1)];
	      nnzul[1] = nnzl;
	    }
	  }
        }
      }
    }
}

extern "C" void cuda_fill_tmp_arrays_double_FromC(int *idx1_dev, int *p_col_dev, int *coltyp_dev, int *nnzu_val_dev, int *nnzl_val_dev, int *nnzul_dev, double *d1u_dev, double *d1_dev, double *zu_dev, double *z_dev, double *d1l_dev, double *zl_dev, int *na_in, int *np_in, int *na1_in, int *np_rem_in, cudaStream_t  my_stream){
  int na = *na_in;
  int np = *np_in;
  int na1 = *na1_in;
  int np_rem = *np_rem_in;

  dim3 threadsPerBlock(1024);
  dim3 blocks((na1 + threadsPerBlock.x - 1) / threadsPerBlock.x);
  if (blocks.x==0) return;

#ifdef WITH_GPU_STREAMS
  cuda_fill_tmp_arrays_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev, d1u_dev, d1_dev, zu_dev, z_dev, d1l_dev, zl_dev, na, np, na1, np_rem);
#else
  cuda_fill_tmp_arrays_double_kernel<<<blocks, threadsPerBlock>>>              (idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev, d1u_dev, d1_dev, zu_dev, z_dev, d1l_dev, zl_dev, na, np, na1, np_rem);
#endif

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_fill_tmp_arrays_double_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}


__global__ void cuda_fill_tmp_arrays_float_kernel(int *idx1, int *p_col, int *coltyp, int *nnzu_val, int *nnzl_val, int *nnzul, float *d1u, float *d1, float *zu, float *z, float *d1l, float *zl, const int na, const int np, const int na1, const int np_rem) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=0 && i < na1) {
      int index = idx1[i] - 1;
      if (p_col[index] == np_rem) {
        if ((coltyp[index] == 1) || (coltyp[index] == 2)) {
	  int nnzu = nnzu_val[(i) + na1 * (np-1)] -1 ;
	  d1u[nnzu] = d1[i];
	   zu[nnzu] =  z[i];
	}
        if ((coltyp[index] == 3) || (coltyp[index] == 2)) {
	  int nnzl = nnzl_val[(i) + na1 * (np-1)]-1;
	  d1l[nnzl] = d1[i];
	  zl[nnzl] =  z[i];
	}
      }
    }

    __syncthreads();
    if (i == 0) {
      nnzul[0]=0;
      nnzul[1]=0;
      int nnzu = 0;
      int nnzl = 0;
      for (int ii=na1-1;ii>=0;ii--) {
        int index = idx1[ii] - 1;
        if (nnzu == 0) {
          if (p_col[index] == np_rem) {
            if ((coltyp[index] == 1) || (coltyp[index] == 2)) {
	      nnzu = nnzu_val[(ii) + na1 * (np-1)] ;
	      nnzul[0] = nnzu;
	    }
	  }
        }
        if (nnzl == 0) {
          if (p_col[index] == np_rem) {
            if ((coltyp[index] == 3) || (coltyp[index] == 2)) {
	      nnzl = nnzl_val[(ii) + na1 * (np-1)];
	      nnzul[1] = nnzl;
	    }
	  }
        }
      }
    }
}

extern "C" void cuda_fill_tmp_arrays_float_FromC(int *idx1_dev, int *p_col_dev, int *coltyp_dev, int *nnzu_val_dev, int *nnzl_val_dev, int *nnzul_dev, float *d1u_dev, float *d1_dev, float *zu_dev, float *z_dev, float *d1l_dev, float *zl_dev, int *na_in, int *np_in, int *na1_in, int *np_rem_in, cudaStream_t  my_stream){
  int na = *na_in;
  int np = *np_in;
  int na1 = *na1_in;
  int np_rem = *np_rem_in;

  dim3 threadsPerBlock(1024);
  dim3 blocks((na1 + threadsPerBlock.x - 1) / threadsPerBlock.x);
  if (blocks.x==0) return;

#ifdef WITH_GPU_STREAMS
  cuda_fill_tmp_arrays_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev, d1u_dev, d1_dev, zu_dev, z_dev, d1l_dev, zl_dev, na, np, na1, np_rem);
#else
  cuda_fill_tmp_arrays_float_kernel<<<blocks, threadsPerBlock>>>              (idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev, d1u_dev, d1_dev, zu_dev, z_dev, d1l_dev, zl_dev, na, np, na1, np_rem);
#endif

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_fill_tmp_arrays_float_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}



__global__ void cuda_copy_qtmp1_slice_to_q_double_kernel(double *q, double *qtmp1, int *l_col_out, int *p_col_out, int *ndef_c, int *p_col, int *idx2, int *idx, const int l_rqs, const int l_rqe, const int l_rows, const int matrixRows, const int gemm_dim_k, const int my_pcol, const int na1, const int np_rem, const int na) {
    int slice = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i>=0 && i < na) {
      if (slice >=0 && slice < l_rows) {
        int j = idx[i];
        if (j> na1) {
          int index3 = idx2[j-na1-1];
          if (p_col[index3-1] == np_rem) {
            if (p_col_out[i] == my_pcol) {
              if (slice >= 0 && slice < l_rows) {
                int ndef = ndef_c[i];
                int index2 = slice + gemm_dim_k * (ndef-1);
                int l_col = l_col_out[i];
                int index = l_rqs -1 + slice + matrixRows * (l_col-1);
                q[index] = qtmp1[index2];
              }
            }
          }
        }
      }

    }

}

extern "C" void cuda_copy_qtmp1_slice_to_q_double_FromC(double *q_dev, double *qtmp1_dev, int *l_col_out_dev, int *p_col_out_dev, int *ndef_c_dev, int *p_col_dev, int *idx2_dev, int *idx_dev, int *l_rqs_in, int *l_rqe_in, int *l_rows_in, int *matrixRows_in, int *gemm_dim_k_in, int *my_pcol_in, int *na1_in, int *np_rem_in, int *na_in, cudaStream_t  my_stream){
  int l_rqs = *l_rqs_in;
  int l_rqe = *l_rqe_in;
  int l_rows = *l_rows_in;
  int matrixRows = *matrixRows_in;
  int gemm_dim_k = *gemm_dim_k_in;
  int my_pcol = *my_pcol_in;
  int na1 = *na1_in;
  int np_rem = *np_rem_in;
  int na = *na_in;

  dim3 threadsPerBlock(32,32);
  dim3 blocks((l_rows + threadsPerBlock.x - 1) / threadsPerBlock.x,(na + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (blocks.x==0 || blocks.y==0) return;

#ifdef WITH_GPU_STREAMS
  cuda_copy_qtmp1_slice_to_q_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, qtmp1_dev, l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, idx2_dev, idx_dev, l_rqs, l_rqe, l_rows, matrixRows, gemm_dim_k, my_pcol, na1, np_rem, na);
#else
  cuda_copy_qtmp1_slice_to_q_double_kernel<<<blocks, threadsPerBlock>>>              (q_dev, qtmp1_dev, l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, idx2_dev, idx_dev, l_rqs, l_rqe, l_rows, matrixRows, gemm_dim_k, my_pcol, na1, np_rem, na);
#endif

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_qtmp1_slice_to_q_double_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}



__global__ void cuda_copy_qtmp1_slice_to_q_float_kernel(float *q, float *qtmp1, int *l_col_out, int *p_col_out, int *ndef_c, int *p_col, int *idx2, int *idx, const int l_rqs, const int l_rqe, const int l_rows, const int matrixRows, const int gemm_dim_k, const int my_pcol, const int na1, const int np_rem, const int na) {
    int slice = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i>=0 && i < na) {
      if (slice >=0 && slice < l_rows) {
        int j = idx[i];
        if (j> na1) {
          int index3 = idx2[j-na1-1];
          if (p_col[index3-1] == np_rem) {
            if (p_col_out[i] == my_pcol) {
              if (slice >= 0 && slice < l_rows) {
                int ndef = ndef_c[i];
                int index2 = slice + gemm_dim_k * (ndef-1);
                int l_col = l_col_out[i];
                int index = l_rqs -1 + slice + matrixRows * (l_col-1);
                q[index] = qtmp1[index2];
              }
            }
          }
        }
      }

    }

}

extern "C" void cuda_copy_qtmp1_slice_to_q_float_FromC(float *q_dev, float *qtmp1_dev, int *l_col_out_dev, int *p_col_out_dev, int *ndef_c_dev, int *p_col_dev, int *idx2_dev, int *idx_dev, int *l_rqs_in, int *l_rqe_in, int *l_rows_in, int *matrixRows_in, int *gemm_dim_k_in, int *my_pcol_in, int *na1_in, int *np_rem_in, int *na_in, cudaStream_t  my_stream){
  int l_rqs = *l_rqs_in;
  int l_rqe = *l_rqe_in;
  int l_rows = *l_rows_in;
  int matrixRows = *matrixRows_in;
  int gemm_dim_k = *gemm_dim_k_in;
  int my_pcol = *my_pcol_in;
  int na1 = *na1_in;
  int np_rem = *np_rem_in;
  int na = *na_in;

  dim3 threadsPerBlock(32,32);
  dim3 blocks((l_rows + threadsPerBlock.x - 1) / threadsPerBlock.x,(na + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (blocks.x==0 || blocks.y==0) return;

#ifdef WITH_GPU_STREAMS
  cuda_copy_qtmp1_slice_to_q_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, qtmp1_dev, l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, idx2_dev, idx_dev, l_rqs, l_rqe, l_rows, matrixRows, gemm_dim_k, my_pcol, na1, np_rem, na);
#else
  cuda_copy_qtmp1_slice_to_q_float_kernel<<<blocks, threadsPerBlock>>>              (q_dev, qtmp1_dev, l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, idx2_dev, idx_dev, l_rqs, l_rqe, l_rows, matrixRows, gemm_dim_k, my_pcol, na1, np_rem, na);
#endif

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_qtmp1_slice_to_q_float_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}




__global__ void cuda_copy_q_slice_to_qtmp2_double_kernel(double *q, double *qtmp2, int *idxq1, int *l_col_out, const int l_rows, const int l_rqs, const int l_rqe, const int matrixRows, const int matrixCols, const int gemm_dim_k, const int gemm_dim_m, const int ns, const int ncnt, const int indx, const int indx2, const int na) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // 1.._l_rows
    int ii = blockIdx.y * blockDim.y + threadIdx.y + 1; // 1.._l_rows



    if (ii >=1 && ii < ncnt+1) {
      if (j >= 0 && j < l_rows) {
        int idx_2= ii+1+ns-1;
        int idx = idxq1[idx_2-1] ; 
        int k = l_col_out[idx-1]; 

        int index2 = j + l_rqs-1 + matrixRows * (k-1);
        int index  = j + gemm_dim_k * (ii-1);
        qtmp2[index]=q[index2];
      } 
    }
}


extern "C" void cuda_copy_q_slice_to_qtmp2_double_FromC(double *q_dev, double *qtmp2_dev, int *idxq1, int *l_col_out, int *l_rows_in, int *l_rqs_in, int *l_rqe_in, int *matrixRows_in, int *matrixCols_in, int *gemm_dim_k_in, int *gemm_dim_m_in, int *ns_in, int * ncnt_in, int *indx_in, int *indx2_in, int *na_in, cudaStream_t  my_stream){
  int l_rows = *l_rows_in;
  int l_rqs = *l_rqs_in;
  int l_rqe = *l_rqe_in;
  int matrixRows = *matrixRows_in;
  int matrixCols = *matrixCols_in;
  int gemm_dim_k = *gemm_dim_k_in;
  int gemm_dim_m = *gemm_dim_m_in;
  int ns = *ns_in;
  int ncnt = *ncnt_in;
  int indx = *indx_in;
  int indx2 = *indx2_in;
  int na = *na_in;

  dim3 threadsPerBlock(32, 32);
  dim3 blocks((l_rows + threadsPerBlock.x - 1) / threadsPerBlock.x, (ncnt + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (blocks.x==0 || blocks.y==0) return;

#ifdef WITH_GPU_STREAMS
  cuda_copy_q_slice_to_qtmp2_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, qtmp2_dev, idxq1, l_col_out, l_rows, l_rqs, l_rqe, matrixRows, matrixCols, gemm_dim_k, gemm_dim_m, ns, ncnt, indx, indx2, na);
#else
  cuda_copy_q_slice_to_qtmp2_double_kernel<<<blocks, threadsPerBlock>>>              (q_dev, qtmp2_dev, idxq1, l_col_out, l_rows, l_rqs, l_rqe, matrixRows, matrixCols, gemm_dim_k, gemm_dim_m, ns, ncnt, indx, indx2, na);
#endif

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_q_slice_to_qtmp2_double_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}



__global__ void cuda_copy_q_slice_to_qtmp2_float_kernel(float *q, float *qtmp2, int *idxq1, int *l_col_out, const int l_rows, const int l_rqs, const int l_rqe, const int matrixRows, const int matrixCols, const int gemm_dim_k, const int gemm_dim_m, const int ns, const int ncnt, const int indx, const int indx2, const int na) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // 1.._l_rows
    int ii = blockIdx.y * blockDim.y + threadIdx.y + 1; // 1.._l_rows



    if (ii >=1 && ii < ncnt+1) {
      if (j >= 0 && j < l_rows) {
        int idx_2= ii+1+ns-1;
        int idx = idxq1[idx_2-1] ; 
        int k = l_col_out[idx-1]; 

        int index2 = j + l_rqs-1 + matrixRows * (k-1);
        int index  = j + gemm_dim_k * (ii-1);
        qtmp2[index]=q[index2];
      } 
    }
}


extern "C" void cuda_copy_q_slice_to_qtmp2_float_FromC(float *q_dev, float *qtmp2_dev, int *idxq1, int *l_col_out, int *l_rows_in, int *l_rqs_in, int *l_rqe_in, int *matrixRows_in, int *matrixCols_in, int *gemm_dim_k_in, int *gemm_dim_m_in, int *ns_in, int * ncnt_in, int *indx_in, int *indx2_in, int *na_in, cudaStream_t  my_stream){
  int l_rows = *l_rows_in;
  int l_rqs = *l_rqs_in;
  int l_rqe = *l_rqe_in;
  int matrixRows = *matrixRows_in;
  int matrixCols = *matrixCols_in;
  int gemm_dim_k = *gemm_dim_k_in;
  int gemm_dim_m = *gemm_dim_m_in;
  int ns = *ns_in;
  int ncnt = *ncnt_in;
  int indx = *indx_in;
  int indx2 = *indx2_in;
  int na = *na_in;

  dim3 threadsPerBlock(32, 32);
  dim3 blocks((l_rows + threadsPerBlock.x - 1) / threadsPerBlock.x, (ncnt + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (blocks.x==0 || blocks.y==0) return;

#ifdef WITH_GPU_STREAMS
  cuda_copy_q_slice_to_qtmp2_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, qtmp2_dev, idxq1, l_col_out, l_rows, l_rqs, l_rqe, matrixRows, matrixCols, gemm_dim_k, gemm_dim_m, ns, ncnt, indx, indx2, na);
#else
  cuda_copy_q_slice_to_qtmp2_float_kernel<<<blocks, threadsPerBlock>>>              (q_dev, qtmp2_dev, idxq1, l_col_out, l_rows, l_rqs, l_rqe, matrixRows, matrixCols, gemm_dim_k, gemm_dim_m, ns, ncnt, indx, indx2, na);
#endif

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_q_slice_to_qtmp2_float_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

//_________________________________________________________________________________________________

template <typename T>
__global__ void cuda_fill_ev_kernel(T *ev, T *d1u, T *dbase, T *ddiff, T *zu, T *ev_scale, 
                                    int *idxq1, int *idx, const int na, const int gemm_dim_l, const int gemm_dim_m, const int nnzu, const int ns, const int ncnt) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (k >=0 && k< nnzu && k < na) {
      if (i>=1 && i < ncnt+1) {
        if (nnzu >= 1) {
          int idx_2= i+1+ns-1;
          int indx = idxq1[idx_2-1] ;
          int j = idx[indx-1];

          T tmp = d1u[k] - dbase[j-1];
          tmp   = tmp    + ddiff[j-1];
          ev[k + gemm_dim_l*(i-1)] = zu[k] / tmp * ev_scale[j-1];
        }
      }
    }

}

template <typename T>
void cuda_fill_ev(T *ev_dev, T *d1u_dev, T *dbase_dev, T *ddiff_dev, T *zu_dev, T *ev_scale_dev, 
                  int *idxq1_dev, int  *idx_dev, int *na_in, int *gemm_dim_l_in, int *gemm_dim_m_in, int *nnzu_in, int *ns_in, int *ncnt_in, cudaStream_t  my_stream){
  int na = *na_in;
  int gemm_dim_l = *gemm_dim_l_in;
  int gemm_dim_m = *gemm_dim_m_in;
  int nnzu = *nnzu_in;
  int ns = *ns_in;
  int ncnt = *ncnt_in;

  dim3 threadsPerBlock(32,32);
  dim3 blocks((nnzu + threadsPerBlock.x - 1) / threadsPerBlock.x, (ncnt + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (blocks.x==0 || blocks.y==0) return;


  if (nnzu >= 1) {
#ifdef WITH_GPU_STREAMS
  cuda_fill_ev_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev, idxq1_dev, idx_dev, na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt);
#else
  cuda_fill_ev_kernel<<<blocks, threadsPerBlock>>>(ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev, idxq1_dev, idx_dev, na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt);
#endif

  cudaError_t gpuerr = cudaGetLastError();
  if (gpuerr != cudaSuccess){
    printf("Error in executing cuda_fill_ev_kernel: %s\n",cudaGetErrorString(gpuerr));
  }
  }
}

extern "C" void cuda_fill_ev_double_FromC(double *ev_dev, double *d1u_dev, double *dbase_dev, double *ddiff_dev, double *zu_dev, double *ev_scale_dev, 
                                          int *idxq1_dev, int  *idx_dev, int *na_in, int *gemm_dim_l_in, int *gemm_dim_m_in, int *nnzu_in, int *ns_in, int *ncnt_in, cudaStream_t  my_stream){
  cuda_fill_ev (ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev,
                idxq1_dev, idx_dev, na_in, gemm_dim_l_in, gemm_dim_m_in, nnzu_in, ns_in, ncnt_in, my_stream);
}
extern "C" void cuda_fill_ev_float_FromC (float  *ev_dev, float  *d1u_dev, float  *dbase_dev, float  *ddiff_dev, float  *zu_dev, float  *ev_scale_dev, 
                                          int *idxq1_dev, int  *idx_dev, int *na_in, int *gemm_dim_l_in, int *gemm_dim_m_in, int *nnzu_in, int *ns_in, int *ncnt_in, cudaStream_t  my_stream){
  cuda_fill_ev (ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev,
                idxq1_dev, idx_dev, na_in, gemm_dim_l_in, gemm_dim_m_in, nnzu_in, ns_in, ncnt_in, my_stream);
}
//_________________________________________________________________________________________________



__global__ void cuda_copy_qtmp2_slice_to_q_double_kernel(double *q, double *qtmp2, int *idx1q, int *l_col_out, const int l_rqs, const int l_rqe, const int l_rows,  const int ncnt, const int gemm_dim_k, const int matrixRows, const int ns) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j>=0 && j<l_rows) {
      if (i>=0 && i< ncnt) {
        int idx = idx1q[((i+1) + ns) - 1];
        int l_col = l_col_out[idx-1];

        q[j+l_rqs - 1 + matrixRows * (l_col-1)] = qtmp2[j+gemm_dim_k*i];
      }
    }

}

extern "C" void cuda_copy_qtmp2_slice_to_q_double_FromC(double *q_dev, double *qtmp2_dev, int *idx1q_dev, int *l_col_out_dev, int *l_rqs_in, int *l_rqe_in, int *l_rows_in, int *ncnt_in, int *gemm_dim_k_in, int *matrixRows_in, int *ns_in, cudaStream_t  my_stream){

  int l_rqs = *l_rqs_in;
  int l_rqe = *l_rqe_in;
  int l_rows = *l_rows_in;
  int ncnt = *ncnt_in;
  int gemm_dim_k = *gemm_dim_k_in;
  int matrixRows = *matrixRows_in;
  int ns = *ns_in;

  dim3 threadsPerBlock(32,32);
  dim3 blocks((ncnt + threadsPerBlock.x - 1) / threadsPerBlock.x, (l_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (blocks.x==0 || blocks.y==0) return;

#ifdef WITH_GPU_STREAMS
  cuda_copy_qtmp2_slice_to_q_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, qtmp2_dev, idx1q_dev, l_col_out_dev, l_rqs, l_rqe, l_rows, ncnt, gemm_dim_k, matrixRows, ns);
#else
  cuda_copy_qtmp2_slice_to_q_double_kernel<<<blocks, threadsPerBlock>>>(q_dev, qtmp2_dev, idx1q_dev, l_col_out_dev, l_rqs, l_rqe, l_rows, ncnt, gemm_dim_k, matrixRows, ns);
#endif

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_qtmp2_slice_to_q_double_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}



__global__ void cuda_copy_qtmp2_slice_to_q_float_kernel(float *q, float *qtmp2, int *idx1q, int *l_col_out, const int l_rqs, const int l_rqe, const int l_rows,  const int ncnt, const int gemm_dim_k, const int matrixRows, const int ns) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j>=0 && j<l_rows) {
      if (i>=0 && i< ncnt) {
        int idx = idx1q[((i+1) + ns) - 1];
        int l_col = l_col_out[idx-1];

        q[j+l_rqs - 1 + matrixRows * (l_col-1)] = qtmp2[j+gemm_dim_k*i];
      }
    }

}

extern "C" void cuda_copy_qtmp2_slice_to_q_float_FromC(float *q_dev, float *qtmp2_dev, int *idx1q_dev, int *l_col_out_dev, int *l_rqs_in, int *l_rqe_in, int *l_rows_in, int *ncnt_in, int *gemm_dim_k_in, int *matrixRows_in, int *ns_in, cudaStream_t  my_stream){

  int l_rqs = *l_rqs_in;
  int l_rqe = *l_rqe_in;
  int l_rows = *l_rows_in;
  int ncnt = *ncnt_in;
  int gemm_dim_k = *gemm_dim_k_in;
  int matrixRows = *matrixRows_in;
  int ns = *ns_in;

  dim3 threadsPerBlock(32,32);
  dim3 blocks((ncnt + threadsPerBlock.x - 1) / threadsPerBlock.x, (l_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (blocks.x==0 || blocks.y==0) return;

#ifdef WITH_GPU_STREAMS
  cuda_copy_qtmp2_slice_to_q_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, qtmp2_dev, idx1q_dev, l_col_out_dev, l_rqs, l_rqe, l_rows, ncnt, gemm_dim_k, matrixRows, ns);
#else
  cuda_copy_qtmp2_slice_to_q_float_kernel<<<blocks, threadsPerBlock>>>(q_dev, qtmp2_dev, idx1q_dev, l_col_out_dev, l_rqs, l_rqe, l_rows, ncnt, gemm_dim_k, matrixRows, ns);
#endif

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_qtmp2_slice_to_q_float_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}




__global__ void cuda_zero_q_double_kernel(double *q, int *p_col_out, int *l_col_out, const int na, const int my_pcol, const int l_rqs, const int l_rqe, const int matrixRows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i>=0 && i<na) {
      if (j>=0 && j<l_rqe-l_rqs+1) {
        if (p_col_out[i] == my_pcol) {
      	  int index = l_col_out[i] -1;
          q[j+l_rqs - 1 + matrixRows * index] = 0;
      }	      
    }
  }
}

extern "C" void cuda_zero_q_double_FromC(double *q_dev, int *p_col_out_dev, int *l_col_out_dev, int *na_in, int *my_pcol_in, int *l_rqs_in, int *l_rqe_in, int *matrixRows_in, cudaStream_t  my_stream){

  int na = *na_in;
  int my_pcol = *my_pcol_in;
  int l_rqs = *l_rqs_in;
  int l_rqe = *l_rqe_in;
  int matrixRows = *matrixRows_in;

  dim3 threadsPerBlock(32,32);
  dim3 blocks((na + threadsPerBlock.x - 1) / threadsPerBlock.x, ((l_rqe-l_rqs+1) + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (blocks.x==0 || blocks.y==0) return;

#ifdef WITH_GPU_STREAMS
  cuda_zero_q_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, p_col_out_dev, l_col_out_dev, na, my_pcol, l_rqs, l_rqe, matrixRows);
#else
  cuda_zero_q_double_kernel<<<blocks, threadsPerBlock>>>(q_dev, p_col_out_dev, l_col_out_dev, na, my_pcol, l_rqs, l_rqe, matrixRows);
#endif

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_zero_q_double_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}



__global__ void cuda_zero_q_float_kernel(float *q, int *p_col_out, int *l_col_out, const int na, const int my_pcol, const int l_rqs, const int l_rqe, const int matrixRows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i>=0 && i<na) {
      if (j>=0 && j<l_rqe-l_rqs+1) {
        if (p_col_out[i] == my_pcol) {
      	  int index = l_col_out[i] -1;
          q[j+l_rqs - 1 + matrixRows * index] = 0;
      }	      
    }
  }
}

extern "C" void cuda_zero_q_float_FromC(float *q_dev, int *p_col_out_dev, int *l_col_out_dev, int *na_in, int *my_pcol_in, int *l_rqs_in, int *l_rqe_in, int *matrixRows_in, cudaStream_t  my_stream){

  int na = *na_in;
  int my_pcol = *my_pcol_in;
  int l_rqs = *l_rqs_in;
  int l_rqe = *l_rqe_in;
  int matrixRows = *matrixRows_in;

  dim3 threadsPerBlock(32,32);
  dim3 blocks((na + threadsPerBlock.x - 1) / threadsPerBlock.x, ((l_rqe-l_rqs+1) + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (blocks.x==0 || blocks.y==0) return;

#ifdef WITH_GPU_STREAMS
  cuda_zero_q_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(q_dev, p_col_out_dev, l_col_out_dev, na, my_pcol, l_rqs, l_rqe, matrixRows);
#else
  cuda_zero_q_float_kernel<<<blocks, threadsPerBlock>>>(q_dev, p_col_out_dev, l_col_out_dev, na, my_pcol, l_rqs, l_rqe, matrixRows);
#endif

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_zero_q_float_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}





__global__ void cuda_copy_q_slice_to_qtmp1_double_kernel(double *qtmp1, double *q, int *ndef_c,int *l_col, int *idx2, int *p_col, const int na2, const int na, const int my_pcol, const int l_rows, const int l_rqs, const int l_rqe, const int matrixRows, const int gemm_dim_k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // l_rows


    if (j>=0 && j<l_rows) {
      for (int i=1; i<na2+1; i++){
         int l_idx = l_col[idx2[i-1]-1];
         if (p_col[idx2[i-1]-1] == my_pcol) {
	   ndef_c[j] = ndef_c[j]+1;
	   qtmp1[j+gemm_dim_k*(ndef_c[j]-1)] = q[j+l_rqs-1 + matrixRows*(l_idx-1)];      
         }
      }
    }
}

extern "C" void cuda_copy_q_slice_to_qtmp1_double_FromC(double *qtmp1_dev, double *q_dev, int *ndef_c_dev, int *l_col_dev, int *idx2_dev, int *p_col_dev, int *na2_in, int *na_in, int *my_pcol_in, int *l_rows_in, int *l_rqs_in, int *l_rqe_in, int *matrixRows_in, int *gemm_dim_k_in, cudaStream_t  my_stream){

  int na2 = *na2_in;
  int na = *na_in;
  int my_pcol = *my_pcol_in;
  int l_rows = *l_rows_in;
  int l_rqs = *l_rqs_in;
  int l_rqe = *l_rqe_in;
  int matrixRows = *matrixRows_in;
  int gemm_dim_k = *gemm_dim_k_in;

  dim3 threadsPerBlock(1024);
  dim3 blocks((l_rows + threadsPerBlock.x - 1) / threadsPerBlock.x);
  if (blocks.x==0) return;

#ifdef WITH_GPU_STREAMS
  cuda_copy_q_slice_to_qtmp1_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(qtmp1_dev, q_dev, ndef_c_dev, l_col_dev, idx2_dev, p_col_dev, na2, na2, my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k);
#else
  cuda_copy_q_slice_to_qtmp1_double_kernel<<<blocks, threadsPerBlock>>>(qtmp1_dev, q_dev, ndef_c_dev, l_col_dev, idx2_dev, p_col_dev, na2, na, my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k);
#endif

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_q_slice_to_qtmp1_double_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}



__global__ void cuda_copy_q_slice_to_qtmp1_float_kernel(float *qtmp1, float *q, int *ndef_c,int *l_col, int *idx2, int *p_col, const int na2, const int na, const int my_pcol, const int l_rows, const int l_rqs, const int l_rqe, const int matrixRows, const int gemm_dim_k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // l_rows


    if (j>=0 && j<l_rows) {
      for (int i=1; i<na2+1; i++){
         int l_idx = l_col[idx2[i-1]-1];
         if (p_col[idx2[i-1]-1] == my_pcol) {
	   ndef_c[j] = ndef_c[j]+1;
	   qtmp1[j+gemm_dim_k*(ndef_c[j]-1)] = q[j+l_rqs-1 + matrixRows*(l_idx-1)];      
         }
      }
    }
}

extern "C" void cuda_copy_q_slice_to_qtmp1_float_FromC(float *qtmp1_dev, float *q_dev, int *ndef_c_dev, int *l_col_dev, int *idx2_dev, int *p_col_dev, int *na2_in, int *na_in, int *my_pcol_in, int *l_rows_in, int *l_rqs_in, int *l_rqe_in, int *matrixRows_in, int *gemm_dim_k_in, cudaStream_t  my_stream){

  int na2 = *na2_in;
  int na = *na_in;
  int my_pcol = *my_pcol_in;
  int l_rows = *l_rows_in;
  int l_rqs = *l_rqs_in;
  int l_rqe = *l_rqe_in;
  int matrixRows = *matrixRows_in;
  int gemm_dim_k = *gemm_dim_k_in;

  dim3 threadsPerBlock(1024);
  dim3 blocks((l_rows + threadsPerBlock.x - 1) / threadsPerBlock.x);
  if (blocks.x==0) return;

#ifdef WITH_GPU_STREAMS
  cuda_copy_q_slice_to_qtmp1_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(qtmp1_dev, q_dev, ndef_c_dev, l_col_dev, idx2_dev, p_col_dev, na2, na2, my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k);
#else
  cuda_copy_q_slice_to_qtmp1_float_kernel<<<blocks, threadsPerBlock>>>(qtmp1_dev, q_dev, ndef_c_dev, l_col_dev, idx2_dev, p_col_dev, na2, na, my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k);
#endif

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_q_slice_to_qtmp1_float_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}




__global__ void cuda_copy_qtmp1_to_qtmp1_tmp_double_kernel(double *qtmp1, double *qtmp1_tmp, const int gemm_dim_k, const int gemm_dim_l) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;


    if (i>=0 && i<gemm_dim_k) {
      if (j>=0 && j<gemm_dim_l) {
        qtmp1_tmp[i+gemm_dim_k*j] = qtmp1[i+gemm_dim_k*j];
      }    
    }        
           
         
}      
    
extern "C" void cuda_copy_qtmp1_to_qtmp1_tmp_double_FromC(double *qtmp1_dev, double *qtmp1_tmp_dev, int *gemm_dim_k_in, int *gemm_dim_l_in, cudaStream_t  my_stream){
    
  int gemm_dim_k = *gemm_dim_k_in;
  int gemm_dim_l = *gemm_dim_l_in;
    
  dim3 threadsPerBlock(32,32);
  dim3 blocks((gemm_dim_k + threadsPerBlock.x - 1) / threadsPerBlock.x, (gemm_dim_l + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (blocks.x==0 || blocks.y==0) return;
          
#ifdef WITH_GPU_STREAMS
  cuda_copy_qtmp1_to_qtmp1_tmp_double_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(qtmp1_dev, qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l);
#else     
  cuda_copy_qtmp1_to_qtmp1_tmp_double_kernel<<<blocks, threadsPerBlock>>>(qtmp1_dev, qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l);
#endif
      
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_qtmp1_to_qtmp1_tmp_double_kernel: %s\n",cudaGetErrorString(cuerr));
  }       
}



__global__ void cuda_copy_qtmp1_to_qtmp1_tmp_float_kernel(float *qtmp1, float *qtmp1_tmp, const int gemm_dim_k, const int gemm_dim_l) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;


    if (i>=0 && i<gemm_dim_k) {
      if (j>=0 && j<gemm_dim_l) {
        qtmp1_tmp[i+gemm_dim_k*j] = qtmp1[i+gemm_dim_k*j];
      }    
    }        
           
         
}      
    
extern "C" void cuda_copy_qtmp1_to_qtmp1_tmp_float_FromC(float *qtmp1_dev, float *qtmp1_tmp_dev, int *gemm_dim_k_in, int *gemm_dim_l_in, cudaStream_t  my_stream){
    
  int gemm_dim_k = *gemm_dim_k_in;
  int gemm_dim_l = *gemm_dim_l_in;
    
  dim3 threadsPerBlock(32,32);
  dim3 blocks((gemm_dim_k + threadsPerBlock.x - 1) / threadsPerBlock.x, (gemm_dim_l + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (blocks.x==0 || blocks.y==0) return;
          
#ifdef WITH_GPU_STREAMS
  cuda_copy_qtmp1_to_qtmp1_tmp_float_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(qtmp1_dev, qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l);
#else     
  cuda_copy_qtmp1_to_qtmp1_tmp_float_kernel<<<blocks, threadsPerBlock>>>(qtmp1_dev, qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l);
#endif
      
  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_copy_qtmp1_to_qtmp1_tmp_floaet_kernel: %s\n",cudaGetErrorString(cuerr));
  }       
}
            
  


__global__ void cuda_update_ndef_c_kernel(int *ndef_c, int *idx, int *p_col, int *idx2, const int na, const int na1, const int np_rem, const int ndef_start) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x; // na

    //for (int ii=0;ii<na;ii++) {
      if (ii>=0 && ii<na) {
        ndef_c[ii] = ndef_start;
         int jj = idx[ii];
         if (jj > na1) {
           if (p_col[idx2[jj-1-na1]-1] == np_rem) {
             ndef_c[ii] = -1;
           }
         }
       }
    //}

    __syncthreads();

    int counter = 0;
    if (ii == 0) {
      for (int k=0;k<na;k++){
        if (ndef_c[k] == -1) {
          counter = counter + 1;
          ndef_c[k] = ndef_start + counter;
        } else {
          ndef_c[k] = ndef_start;
        }

      }
   }
    
}

extern "C" void cuda_update_ndef_c_FromC(int *ndef_c_dev, int *idx_dev, int *p_col_dev, int *idx2_dev, int *na_in, int *na1_in, int *np_rem_in, int *ndef_in, cudaStream_t  my_stream){

  int na = *na_in;
  int na1 = *na1_in;
  int np_rem = *np_rem_in;
  int ndef = *ndef_in;

  dim3 threadsPerBlock(1024);
  dim3 blocks((na + threadsPerBlock.x - 1) / threadsPerBlock.x);
  if (blocks.x==0) return;

#ifdef WITH_GPU_STREAMS
  cuda_update_ndef_c_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(ndef_c_dev, idx_dev, p_col_dev, idx2_dev, na, na1, np_rem, ndef);
#else
  cuda_update_ndef_c_kernel<<<blocks, threadsPerBlock>>>(ndef_c_dev, idx_dev, p_col_dev, idx2_dev, na, na1, np_rem, ndef);
#endif

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_update_ndef_c_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}



__global__ void cuda_compute_nnzl_nnzu_val_part1_kernel(int *p_col, int *idx1, int *coltyp, int *nnzu_val, int *nnzl_val, const int na, const int na1, const int np_rem, const int npc_n, const int nnzu_start, const int nnzl_start, const int np_in){

    int i  = blockIdx.x * blockDim.x + threadIdx.x;


    int np = np_in - 1;
    //for (int i=0;i<na1;i++) {
      if (i>=0 && i<na1) {
        if (np>=0 && np<npc_n+1) {
          nnzu_val[i + na1*(np)] = 0;
          nnzl_val[i + na1*(np)] = 0;
          if (p_col[idx1[i]-1] == np_rem) {
            if (coltyp[idx1[i]-1] == 1 || coltyp[idx1[i]-1] == 2) {
              nnzu_val[i + na1*(np)] = 1;
            }
            if (coltyp[idx1[i]-1] == 3 || coltyp[idx1[i]-1] == 2) {
              nnzl_val[i + na1*(np)] = 1;
            }
          }
        }
      }
    //}

}

extern "C" void cuda_compute_nnzl_nnzu_val_part1_FromC(int *p_col_dev, int *idx1_dev, int *coltyp_dev, int *nnzu_val_dev, int *nnzl_val_dev, int *na_in, int *na1_in, int *np_rem_in, int *npc_n_in, int *nnzu_start_in, int *nnzl_start_in, int *np_in, cudaStream_t  my_stream){

  int na = *na_in;
  int na1 = *na1_in;
  int np_rem = *np_rem_in;
  int npc_n = *npc_n_in;
  int nnzu_start = *nnzu_start_in;
  int nnzl_start = *nnzl_start_in;
  int np = *np_in;

  dim3 threadsPerBlock(1024);
  dim3 blocks((na1 + threadsPerBlock.x - 1) / threadsPerBlock.x);
  if (blocks.x==0) return;

#ifdef WITH_GPU_STREAMS
  cuda_compute_nnzl_nnzu_val_part1_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(p_col_dev, idx1_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, na, na1, np_rem, npc_n, nnzu_start, nnzl_start, np);
#else
  cuda_compute_nnzl_nnzu_val_part1_kernel<<<blocks, threadsPerBlock>>>(p_col_dev, idx1_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, na, na1, np_rem, npc_n, nnzu_start, nnzl_start, np);
#endif

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_compute_nnzl_nnzu_val_part1_c_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}




__global__ void cuda_compute_nnzl_nnzu_val_part2_kernel(int *nnzu_val, int *nnzl_val, const int na, const int na1, int nnzu_start, const int nnzl_start, const int npc_n){

    int i  = blockIdx.x * blockDim.x + threadIdx.x;

    int counter1 ;
    int counter2 ;
    if (i == 0) {
	 for (int jj=0;jj<npc_n;jj++) {
           counter1 = 0;
           counter2 = 0;
	   for (int ii=0;ii<na1;ii++) {
	     if (nnzu_val[ii+na1*jj] == 1) {
               counter1 = counter1 + 1;
	       nnzu_val[ii+na1*jj] = counter1;
	     } else {
	       nnzu_val[ii+na1*jj] = 0;
	     }
	     if (nnzl_val[ii+na1*jj] == 1) {
               counter2 = counter2 + 1;
	       nnzl_val[ii+na1*jj] = counter2;
	     } else {
	       nnzl_val[ii+na1*jj] = 0;
	     }
	   }
	 }
       }


}


extern "C" void cuda_compute_nnzl_nnzu_val_part2_FromC(int *nnzu_val_dev, int *nnzl_val_dev, int *na_in, int *na1_in, int *nnzu_start_in, int *nnzl_start_in, int *npc_n_in, cudaStream_t  my_stream){

  int na = *na_in;
  int na1 = *na1_in;
  int nnzu_start = *nnzu_start_in;
  int nnzl_start = *nnzl_start_in;
  int npc_n = *npc_n_in;


  dim3 threadsPerBlock(1);
  dim3 blocks(1);

#ifdef WITH_GPU_STREAMS
  cuda_compute_nnzl_nnzu_val_part2_kernel<<<blocks, threadsPerBlock, 0, my_stream>>>(nnzu_val_dev, nnzl_val_dev, na, na1, nnzu_start, nnzl_start, npc_n);
#else
  cuda_compute_nnzl_nnzu_val_part2_kernel<<<blocks, threadsPerBlock>>>(nnzu_val_dev, nnzl_val_dev, na, na1, nnzu_start, nnzl_start, npc_n);
#endif

  cudaError_t cuerr = cudaGetLastError();
  if (cuerr != cudaSuccess){
    printf("Error in executing cuda_compute_nnzl_nnzu_val_part2_c_kernel: %s\n",cudaGetErrorString(cuerr));
  }
}

