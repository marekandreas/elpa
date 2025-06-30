//    Copyright 2025, P. Karpov
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
//      Schwerpunkt Wissenschaftliches Rechnen,
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
//    This file was written by P. Karpov, MPCDF

//________________________________________________________________

__global__ void gpu_update_ndef_c_kernel (int *ndef_c, int *idx, int *p_col, int *idx2, 
                                          const int na, const int na1, const int np_rem, const int ndef_start) {
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

void gpu_update_ndef_c (int *ndef_c_dev, int *idx_dev, int *p_col_dev, int *idx2_dev, 
                        const int na, const int na1, const int np_rem, const int ndef_start, 
                        const int debug, gpuStream_t my_stream) {
  
  dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK);
  dim3 blocks((na + threadsPerBlock.x - 1) / threadsPerBlock.x);
  if (blocks.x==0) return;

#ifdef WITH_GPU_STREAMS
    gpu_update_ndef_c_kernel<<<blocks,threadsPerBlock,0,my_stream>>> (ndef_c_dev, idx_dev, p_col_dev, idx2_dev, 
                                                                      na, na1, np_rem, ndef_start);
#else
    gpu_update_ndef_c_kernel<<<blocks,threadsPerBlock>>>             (ndef_c_dev, idx_dev, p_col_dev, idx2_dev, 
                                                                      na, na1, np_rem, ndef_start);
#endif
    
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess)
      printf("Error in executing gpu_update_ndef_c_kernel: %s\n",gpuGetErrorString(gpuerr));
    }
}

extern "C" void CONCATENATE(ELPA_GPU,  _update_ndef_c_FromC) (intptr_t ndef_c_dev, intptr_t idx_dev, intptr_t p_col_dev, intptr_t idx2_dev, 
                                                              int na, int na1, int np_rem, int ndef_start, 
                                                              int debug, gpuStream_t my_stream) {
  gpu_update_ndef_c((int *) ndef_c_dev, (int *) idx_dev, (int *) p_col_dev, (int *) idx2_dev,
                    na, na1, np_rem, ndef_start, debug, my_stream);
}

//________________________________________________________________

__global__ void gpu_compute_nnzl_nnzu_val_part1_kernel (int *p_col, int *idx1, int *coltyp, int *nnzu_val, int *nnzl_val, 
                                                        const int na, const int na1, const int np_rem, const int npc_n, const int nnzu_start, const int nnzl_start, const int np){
    int i  = blockIdx.x * blockDim.x + threadIdx.x;

    int np_c = np - 1;
    //for (int i=0;i<na1;i++) {
      if (i>=0 && i<na1) {
        if (np_c>=0 && np_c<npc_n+1) {
          nnzu_val[i + na1*(np_c)] = 0;
          nnzl_val[i + na1*(np_c)] = 0;
          if (p_col[idx1[i]-1] == np_rem) {
            if (coltyp[idx1[i]-1] == 1 || coltyp[idx1[i]-1] == 2) {
              nnzu_val[i + na1*(np_c)] = 1;
            }
            if (coltyp[idx1[i]-1] == 3 || coltyp[idx1[i]-1] == 2) {
              nnzl_val[i + na1*(np_c)] = 1;
            }
          }
        }
      }
    //}
}

void gpu_compute_nnzl_nnzu_val_part1 (int *p_col_dev, int *idx1_dev, int *coltyp_dev, int *nnzu_val_dev, int *nnzl_val_dev, 
                                      const int na, const int na1, const int np_rem, const int npc_n, 
                                      const int nnzu_start, const int nnzl_start, const int np,
                                      const int debug, gpuStream_t my_stream) {
  
  dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK);
  dim3 blocks((na1 + threadsPerBlock.x - 1) / threadsPerBlock.x);
  if (blocks.x==0) return;

#ifdef WITH_GPU_STREAMS
    gpu_compute_nnzl_nnzu_val_part1_kernel<<<blocks,threadsPerBlock,0,my_stream>>> (p_col_dev, idx1_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, 
                                                                                    na, na1, np_rem, npc_n, nnzu_start, nnzl_start, np);
#else
    gpu_compute_nnzl_nnzu_val_part1_kernel<<<blocks,threadsPerBlock>>>             (p_col_dev, idx1_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, 
                                                                                    na, na1, np_rem, npc_n, nnzu_start, nnzl_start, np);
#endif
    
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess)
      printf("Error in executing gpu_compute_nnzl_nnzu_val_part1_kernel: %s\n",gpuGetErrorString(gpuerr));
    }
}

extern "C" void CONCATENATE(ELPA_GPU,  _compute_nnzl_nnzu_val_part1_FromC) (intptr_t p_col_dev, intptr_t idx1_dev, intptr_t coltyp_dev, 
                                                                            intptr_t nnzu_val_dev, intptr_t nnzl_val_dev, 
                                                                            int na, int na1, int np_rem, int npc_n, 
                                                                            int nnzu_start, int nnzl_start, int np,
                                                                            int debug, gpuStream_t my_stream) {
  gpu_compute_nnzl_nnzu_val_part1((int *) p_col_dev, (int *) idx1_dev, (int *) coltyp_dev, (int *) nnzu_val_dev, (int *) nnzl_val_dev,
                                  na, na1, np_rem, npc_n, nnzu_start, nnzl_start, np, debug, my_stream);
}

//________________________________________________________________

__global__ void gpu_compute_nnzl_nnzu_val_part2_kernel (int *nnzu_val, int *nnzl_val, 
                                                        const int na, const int na1, int nnzu_start, const int nnzl_start, const int npc_n){
  int i  = blockIdx.x * blockDim.x + threadIdx.x;

  int counter1 ;
  int counter2 ;
  if (i == 0) 
    {
    for (int jj=0; jj<npc_n; jj++)
      {
      counter1 = 0;
      counter2 = 0;
      
      for (int ii=0;ii<na1;ii++)
        {
        if (nnzu_val[ii+na1*jj] == 1)
          {
          counter1 = counter1 + 1;
          nnzu_val[ii+na1*jj] = counter1;
          }
        else 
          {
          nnzu_val[ii+na1*jj] = 0;
          }
        
        if (nnzl_val[ii+na1*jj] == 1)
          {
          counter2 = counter2 + 1;
          nnzl_val[ii+na1*jj] = counter2;
          }
        else 
          {
          nnzl_val[ii+na1*jj] = 0;
          }
        }
      }
    }

}

void gpu_compute_nnzl_nnzu_val_part2 (int *nnzu_val_dev, int *nnzl_val_dev, 
                                      const int na, const int na1, int nnzu_start, const int nnzl_start, const int npc_n,
                                      const int debug, gpuStream_t my_stream) {
  
  dim3 threadsPerBlock(1);
  dim3 blocks(1);

#ifdef WITH_GPU_STREAMS
    gpu_compute_nnzl_nnzu_val_part2_kernel<<<blocks,threadsPerBlock,0,my_stream>>> (nnzu_val_dev, nnzl_val_dev, 
                                                                                    na, na1, nnzu_start, nnzl_start, npc_n);
#else
    gpu_compute_nnzl_nnzu_val_part2_kernel<<<blocks,threadsPerBlock>>>             (nnzu_val_dev, nnzl_val_dev, 
                                                                                    na, na1, nnzu_start, nnzl_start, npc_n);
#endif
    
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess)
      printf("Error in executing gpu_compute_nnzl_nnzu_val_part2_kernel: %s\n",gpuGetErrorString(gpuerr));
    }
}

extern "C" void CONCATENATE(ELPA_GPU,  _compute_nnzl_nnzu_val_part2_FromC) (intptr_t nnzu_val_dev, intptr_t nnzl_val_dev, 
                                                                            int na, int na1, int nnzu_start, int nnzl_start, int npc_n,
                                                                            int debug, gpuStream_t my_stream) {
  gpu_compute_nnzl_nnzu_val_part2((int *) nnzu_val_dev, (int *) nnzl_val_dev,
                                  na, na1, nnzu_start, nnzl_start, npc_n, debug, my_stream);
}

//________________________________________________________________

template <typename T>
__global__ void gpu_copy_qtmp1_slice_to_q_kernel(T *q, T *qtmp1, int *l_col_out, int *p_col_out, int *ndef_c, int *p_col, int *idx2, int *idx, 
                                                 const int l_rqs, const int l_rqe, const int l_rows, const int matrixRows, const int gemm_dim_k, 
                                                 const int my_pcol, const int na1, const int np_rem, const int na){
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

template <typename T>
void gpu_copy_qtmp1_slice_to_q (T *q_dev, T *qtmp1_dev, int *l_col_out_dev, int *p_col_out_dev, int *ndef_c_dev, int *p_col_dev, int *idx2_dev, int *idx_dev, 
                                const int l_rqs, const int l_rqe, const int l_rows, const int matrixRows, const int gemm_dim_k, 
                                const int my_pcol, const int na1, const int np_rem, const int na, const int debug, gpuStream_t my_stream){
  
  dim3 threadsPerBlock(32, 32);
  dim3 blocks((l_rows + threadsPerBlock.x - 1) / threadsPerBlock.x,(na + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (blocks.x==0 || blocks.y==0) return;

#ifdef WITH_GPU_STREAMS
    gpu_copy_qtmp1_slice_to_q_kernel<T><<<blocks,threadsPerBlock,0,my_stream>>>(q_dev, qtmp1_dev, l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, idx2_dev, idx_dev, 
                                                                                l_rqs, l_rqe, l_rows, matrixRows, gemm_dim_k, my_pcol, na1, np_rem, na);
#else
    gpu_copy_qtmp1_slice_to_q_kernel<T><<<blocks,threadsPerBlock>>>            (q_dev, qtmp1_dev, l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, idx2_dev, idx_dev, 
                                                                                l_rqs, l_rqe, l_rows, matrixRows, gemm_dim_k, my_pcol, na1, np_rem, na);
#endif
    
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess)
      printf("Error in executing gpu_copy_qtmp1_slice_to_q_kernel: %s\n",gpuGetErrorString(gpuerr));
    }
}

extern "C" void CONCATENATE(ELPA_GPU,  _copy_qtmp1_slice_to_q_FromC) (char dataType, intptr_t q_dev, intptr_t qtmp1_dev, 
                                                                    intptr_t l_col_out_dev, intptr_t p_col_out_dev, intptr_t ndef_c_dev, 
                                                                    intptr_t p_col_dev, intptr_t idx2_dev, intptr_t idx_dev, 
                                                                    int l_rqs, int l_rqe, int l_rows, int matrixRows, int gemm_dim_k, 
                                                                    int my_pcol, int na1, int np_rem, int na, int debug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_copy_qtmp1_slice_to_q<double>((double *) q_dev, (double *) qtmp1_dev,
                                                            (int *) l_col_out_dev, (int *) p_col_out_dev, (int *) ndef_c_dev, 
                                                            (int *) p_col_dev, (int *) idx2_dev, (int *) idx_dev,
                                                            l_rqs, l_rqe, l_rows, matrixRows, gemm_dim_k, my_pcol, na1, np_rem, na, debug, my_stream);
  else if (dataType=='S') gpu_copy_qtmp1_slice_to_q<float> ((float  *) q_dev, (float  *) qtmp1_dev,
                                                            (int *) l_col_out_dev, (int *) p_col_out_dev, (int *) ndef_c_dev, 
                                                            (int *) p_col_dev, (int *) idx2_dev, (int *) idx_dev,
                                                            l_rqs, l_rqe, l_rows, matrixRows, gemm_dim_k, my_pcol, na1, np_rem, na, debug, my_stream);
  else {
    printf("Error in gpu_copy_qtmp1_slice_to_q: Unsupported data type\n");
  }
}

//________________________________________________________________

template <typename T>
__global__ void gpu_copy_q_slice_to_qtmp2_kernel(T *q, T *qtmp2, int *idxq1, int *l_col_out, 
                                                const int l_rows, const int l_rqs, const int l_rqe, const int matrixRows, const int matrixCols, 
                                                const int gemm_dim_k, const int gemm_dim_m, const int ns, const int ncnt, const int indx, const int indx2, const int na) {
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

template <typename T>
void gpu_copy_q_slice_to_qtmp2 (T *q_dev, T *qtmp2_dev, int *idxq1_dev, int *l_col_out_dev, 
                                const int l_rows, const int l_rqs, const int l_rqe, const int matrixRows, const int matrixCols, 
                                const int gemm_dim_k, const int gemm_dim_m, const int ns, const int ncnt, 
                                const int indx, const int indx2, const int na,
                                const int debug, gpuStream_t my_stream){
  
  dim3 threadsPerBlock(32, 32);
  dim3 blocks((l_rows + threadsPerBlock.x - 1) / threadsPerBlock.x, (ncnt + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (blocks.x==0 || blocks.y==0) return;

#ifdef WITH_GPU_STREAMS
    gpu_copy_q_slice_to_qtmp2_kernel<T><<<blocks,threadsPerBlock,0,my_stream>>>(q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, 
                                                                                l_rows, l_rqs, l_rqe, matrixRows, matrixCols, 
                                                                                gemm_dim_k, gemm_dim_m, ns, ncnt, indx, indx2, na); 
#else
    gpu_copy_q_slice_to_qtmp2_kernel<T><<<blocks,threadsPerBlock>>>            (q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, 
                                                                                l_rows, l_rqs, l_rqe, matrixRows, matrixCols, 
                                                                                gemm_dim_k, gemm_dim_m, ns, ncnt, indx, indx2, na);
#endif
    
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess)
      printf("Error in executing gpu_copy_q_slice_to_qtmp2_kernel: %s\n",gpuGetErrorString(gpuerr));
    }
}

extern "C" void CONCATENATE(ELPA_GPU,  _copy_q_slice_to_qtmp2_FromC) (char dataType, intptr_t q_dev, intptr_t qtmp2_dev,
                                                                      intptr_t idxq1_dev, intptr_t l_col_out_dev, 
                                                                      int l_rows, int l_rqs, int l_rqe, int matrixRows, int matrixCols, 
                                                                      int gemm_dim_k, int gemm_dim_m, int ns, int ncnt, 
                                                                      int indx, int indx2, int na,
                                                                      int debug, gpuStream_t my_stream){

  if      (dataType=='D') gpu_copy_q_slice_to_qtmp2<double>((double *) q_dev, (double *) qtmp2_dev, (int *) idxq1_dev, (int *) l_col_out_dev, 
                                                            l_rows, l_rqs, l_rqe, matrixRows, matrixCols, gemm_dim_k, gemm_dim_m, ns, ncnt, indx, indx2, na,
                                                            debug, my_stream);

  else if (dataType=='S') gpu_copy_q_slice_to_qtmp2<float> ((float  *) q_dev, (float  *) qtmp2_dev, (int *) idxq1_dev, (int *) l_col_out_dev, 
                                                            l_rows, l_rqs, l_rqe, matrixRows, matrixCols, gemm_dim_k, gemm_dim_m, ns, ncnt, indx, indx2, na,
                                                            debug, my_stream);
  else {
    printf("Error in gpu_copy_q_slice_to_qtmp2: Unsupported data type\n");
  }
}

//________________________________________________________________

template <typename T>
__global__ void gpu_copy_qtmp2_slice_to_q_kernel(T *q, T *qtmp2, int *idxq1, int *l_col_out, 
                                                 const int l_rqs, const int l_rqe, const int l_rows,  const int ncnt, const int gemm_dim_k, 
                                                 const int matrixRows, const int ns) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (j>=0 && j<l_rows) {
    if (i>=0 && i< ncnt) {
      int idx = idxq1[((i+1) + ns) - 1];
      int l_col = l_col_out[idx-1];

      q[j+l_rqs - 1 + matrixRows * (l_col-1)] = qtmp2[j+gemm_dim_k*i];
    }
  }
}

template <typename T>
void gpu_copy_qtmp2_slice_to_q (T *q_dev, T *qtmp2_dev, int *idxq1_dev, int *l_col_out_dev, 
                                const int l_rqs, const int l_rqe, const int l_rows,  const int ncnt, const int gemm_dim_k, 
                                const int matrixRows, const int ns,
                                const int debug, gpuStream_t my_stream){
  
  dim3 threadsPerBlock(32, 32);
  dim3 blocks((ncnt + threadsPerBlock.x - 1) / threadsPerBlock.x, (l_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (blocks.x==0 || blocks.y==0) return;

#ifdef WITH_GPU_STREAMS
    gpu_copy_qtmp2_slice_to_q_kernel<T><<<blocks,threadsPerBlock,0,my_stream>>>(q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, 
                                                                                l_rqs, l_rqe, l_rows, ncnt, gemm_dim_k, matrixRows, ns); 
#else
    gpu_copy_qtmp2_slice_to_q_kernel<T><<<blocks,threadsPerBlock>>>            (q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, 
                                                                                l_rqs, l_rqe, l_rows, ncnt, gemm_dim_k, matrixRows, ns);
#endif
    
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess)
      printf("Error in executing gpu_copy_qtmp2_slice_to_q_kernel: %s\n",gpuGetErrorString(gpuerr));
    }
}

extern "C" void CONCATENATE(ELPA_GPU,  _copy_qtmp2_slice_to_q_FromC) (char dataType, intptr_t q_dev, intptr_t qtmp2_dev,
                                                                      intptr_t idxq1_dev, intptr_t l_col_out_dev, 
                                                                      int l_rqs, int l_rqe, int l_rows,  int ncnt, int gemm_dim_k,
                                                                      int matrixRows, int ns,
                                                                      int debug, gpuStream_t my_stream){

  if      (dataType=='D') gpu_copy_qtmp2_slice_to_q<double>((double *) q_dev, (double *) qtmp2_dev, (int *) idxq1_dev, (int *) l_col_out_dev,
                                                             l_rqs, l_rqe, l_rows, ncnt, gemm_dim_k, matrixRows, ns, debug, my_stream);
  else if (dataType=='S') gpu_copy_qtmp2_slice_to_q<float> ((float  *) q_dev, (float  *) qtmp2_dev, (int *) idxq1_dev, (int *) l_col_out_dev, 
                                                             l_rqs, l_rqe, l_rows, ncnt, gemm_dim_k, matrixRows, ns, debug, my_stream);
  else {
    printf("Error in gpu_copy_qtmp2_slice_to_q: Unsupported data type\n");
  }
}

//________________________________________________________________

template <typename T>
__global__ void gpu_fill_ev_kernel (T *ev, T *d1u, T *dbase, T *ddiff, T *zu, T *ev_scale, int *idxq1, int *idx, 
                                    const int na, const int gemm_dim_l, const int gemm_dim_m, const int nnzu, const int ns, const int ncnt) {
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
void gpu_fill_ev (T *ev_dev, T *d1u_dev, T *dbase_dev, T *ddiff_dev, T *zu_dev, T *ev_scale_dev, int *idxq1_dev, int  *idx_dev,
                  int na, int gemm_dim_l, int gemm_dim_m, int nnzu, int ns, int ncnt, int debug, gpuStream_t my_stream){
  
  dim3 threadsPerBlock(32, 32);
  dim3 blocks((nnzu + threadsPerBlock.x - 1) / threadsPerBlock.x, (ncnt + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (blocks.x==0 || blocks.y==0) return;

#ifdef WITH_GPU_STREAMS
    gpu_fill_ev_kernel<T><<<blocks,threadsPerBlock,0,my_stream>>>(ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev, idxq1_dev, idx_dev, 
                                                                  na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt); 
#else
    gpu_fill_ev_kernel<T><<<blocks,threadsPerBlock>>>            (ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev, idxq1_dev, idx_dev, 
                                                                  na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt);
#endif
    
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess)
      printf("Error in executing gpu_fill_ev_kernel: %s\n",gpuGetErrorString(gpuerr));
    }
}

extern "C" void CONCATENATE(ELPA_GPU,  _fill_ev_FromC) (char dataType, intptr_t ev_dev, intptr_t d1u_dev, intptr_t dbase_dev, intptr_t ddiff_dev, 
                                                      intptr_t zu_dev, intptr_t ev_scale_dev, intptr_t idxq1_dev, intptr_t idx_dev,
                                                      int na, int gemm_dim_l, int gemm_dim_m, int nnzu, int ns, int ncnt,
                                                      int debug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_fill_ev<double>((double *) ev_dev, (double *) d1u_dev, (double *) dbase_dev, (double *) ddiff_dev, 
                                              (double *) zu_dev, (double *) ev_scale_dev, (int *) idxq1_dev, (int *) idx_dev,
                                              na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt, debug, my_stream);

  else if (dataType=='S') gpu_fill_ev<float> ((float  *) ev_dev, (float  *) d1u_dev, (float  *) dbase_dev, (float  *) ddiff_dev, 
                                              (float  *) zu_dev, (float  *) ev_scale_dev, (int *) idxq1_dev, (int *) idx_dev,
                                              na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt, debug, my_stream);
  else {
    printf("Error in gpu_fill_ev: Unsupported data type\n");
  }
}

//________________________________________________________________

template <typename T>
__global__ void gpu_fill_tmp_arrays_kernel (T *d1u, T *d1, T *zu, T *z, T *d1l, T *zl,
                                            int *idx1, int *p_col, int *coltyp, int *nnzu_val, int *nnzl_val, int *nnzul, 
                                            const int na, const int np, const int na1, const int np_rem) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i>=0 && i < na1) 
    {
    int index = idx1[i] - 1;
    if (p_col[index] == np_rem)
      {
      if ((coltyp[index] == 1) || (coltyp[index] == 2))
        {
        int nnzu = nnzu_val[(i) + na1 * (np-1)] - 1;
        d1u[nnzu] = d1[i];
        zu[nnzu] =  z[i];
        }
      if ((coltyp[index] == 3) || (coltyp[index] == 2))
        {
        int nnzl = nnzl_val[(i) + na1 * (np-1)]-1;
        d1l[nnzl] = d1[i];
        zl[nnzl] =  z[i];
        }
      }
    }

  __syncthreads();
  if (i == 0) 
    {
    nnzul[0]=0;
    nnzul[1]=0;
    int nnzu = 0;
    int nnzl = 0;
    for (int ii=na1-1;ii>=0;ii--)
      {
      int index = idx1[ii] - 1;
      if (nnzu == 0)
        {
        if (p_col[index] == np_rem)
          {
          if ((coltyp[index] == 1) || (coltyp[index] == 2))
            {
            nnzu = nnzu_val[(ii) + na1 * (np-1)] ;
            nnzul[0] = nnzu;
            }
          }
        }
      if (nnzl == 0)
        {
        if (p_col[index] == np_rem)
          {
          if ((coltyp[index] == 3) || (coltyp[index] == 2))
            {
            nnzl = nnzl_val[(ii) + na1 * (np-1)];
            nnzul[1] = nnzl;
            }
          }
        }
      }
    }
}

template <typename T>
void gpu_fill_tmp_arrays (T *d1u_dev, T *d1_dev, T *zu_dev, T *z_dev, T *d1l_dev, T *zl_dev,
                          int *idx1_dev, int *p_col_dev, int *coltyp_dev, int *nnzu_val_dev, int *nnzl_val_dev, int *nnzul_dev, 
                          const int na, const int np, const int na1, const int np_rem, int debug, gpuStream_t my_stream){
  
  dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK);
  dim3 blocks((na1 + threadsPerBlock.x - 1) / threadsPerBlock.x);
  if (blocks.x==0) return;

#ifdef WITH_GPU_STREAMS
    gpu_fill_tmp_arrays_kernel<T><<<blocks,threadsPerBlock,0,my_stream>>>(d1u_dev, d1_dev, zu_dev, z_dev, d1l_dev, zl_dev, 
                                                                          idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev, 
                                                                          na, np, na1, np_rem); 
#else
    gpu_fill_tmp_arrays_kernel<T><<<blocks,threadsPerBlock>>>            (d1u_dev, d1_dev, zu_dev, z_dev, d1l_dev, zl_dev, 
                                                                          idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev, 
                                                                          na, np, na1, np_rem);
#endif
    
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess)
      printf("Error in executing gpu_fill_tmp_arrays_kernel: %s\n",gpuGetErrorString(gpuerr));
    }
}

extern "C" void CONCATENATE(ELPA_GPU,  _fill_tmp_arrays_FromC) (char dataType, intptr_t d1u_dev, intptr_t d1_dev, intptr_t zu_dev, intptr_t z_dev, 
                                                                intptr_t d1l_dev, intptr_t zl_dev,
                                                                intptr_t idx1_dev, intptr_t p_col_dev, intptr_t coltyp_dev, 
                                                                intptr_t nnzu_val_dev, intptr_t nnzl_val_dev, intptr_t nnzul_dev,
                                                                int na, int np, int na1, int np_rem,
                                                                int debug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_fill_tmp_arrays<double>((double *) d1u_dev, (double *) d1_dev, (double *) zu_dev, (double *) z_dev,  (double *) d1l_dev, (double *) zl_dev,
                                                      (int *) idx1_dev, (int *) p_col_dev, (int *) coltyp_dev, (int *) nnzu_val_dev, (int *) nnzl_val_dev, (int *) nnzul_dev,
                                                      na, np, na1, np_rem, debug, my_stream);
  else if (dataType=='S') gpu_fill_tmp_arrays<float> ((float  *) d1u_dev, (float  *) d1_dev, (float  *) zu_dev, (float  *) z_dev, (float  *) d1l_dev, (float  *) zl_dev,
                                                      (int *) idx1_dev, (int *) p_col_dev, (int *) coltyp_dev, (int *) nnzu_val_dev, (int *) nnzl_val_dev, (int *) nnzul_dev,
                                                      na, np, na1, np_rem, debug, my_stream);
  else {
    printf("Error in gpu_fill_tmp_arrays: Unsupported data type\n");
  }
}

//________________________________________________________________

template <typename T>
__global__ void gpu_zero_q_kernel(T *q, int *p_col_out, int *l_col_out, 
                                  const int na, const int my_pcol, const int l_rqs, const int l_rqe, const int matrixRows) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i>=0 && i<na) {
      if (j>=0 && j<l_rqe-l_rqs+1) {
        if (p_col_out[i] == my_pcol) {
      	  int index = l_col_out[i] - 1;
          q[j+l_rqs - 1 + matrixRows*index] = 0;
      }	      
    }
  }
}

template <typename T>
void gpu_zero_q(T *q_dev, int *p_col_out_dev, int *l_col_out_dev, 
                const int na, const int my_pcol, const int l_rqs, const int l_rqe, const int matrixRows, 
                const int debug, gpuStream_t my_stream) {

  
  dim3 threadsPerBlock(32, 32);
  dim3 blocks((na + threadsPerBlock.x - 1) / threadsPerBlock.x, ((l_rqe-l_rqs+1) + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (blocks.x==0 || blocks.y==0) return;

#ifdef WITH_GPU_STREAMS
    gpu_zero_q_kernel<T><<<blocks,threadsPerBlock,0,my_stream>>>(q_dev, p_col_out_dev, l_col_out_dev, na, my_pcol, l_rqs, l_rqe, matrixRows); 
#else
    gpu_zero_q_kernel<T><<<blocks,threadsPerBlock>>>            (q_dev, p_col_out_dev, l_col_out_dev, na, my_pcol, l_rqs, l_rqe, matrixRows);
#endif
    
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess)
      printf("Error in executing gpu_zero_q_kernel: %s\n",gpuGetErrorString(gpuerr));
    }
}

extern "C" void CONCATENATE(ELPA_GPU,  _zero_q_FromC) (char dataType, intptr_t q_dev, intptr_t p_col_out_dev, intptr_t l_col_out_dev, 
                                                      int na, int my_pcol, int l_rqs, int l_rqe, int matrixRows,
                                                      int debug, gpuStream_t my_stream) {
  if      (dataType=='D') gpu_zero_q<double>((double *) q_dev, (int *) p_col_out_dev, (int *) l_col_out_dev, 
                                              na, my_pcol, l_rqs, l_rqe, matrixRows, debug, my_stream);
  else if (dataType=='S') gpu_zero_q<float> ((float  *) q_dev, (int *) p_col_out_dev, (int *) l_col_out_dev, 
                                              na, my_pcol, l_rqs, l_rqe, matrixRows, debug, my_stream);
  else {
    printf("Error in gpu_zero_q: Unsupported data type\n");
  }
}

//________________________________________________________________

template <typename T>
__global__ void gpu_copy_q_slice_to_qtmp1_kernel (T *qtmp1, T *q, int *ndef_c,int *l_col, int *idx2, int *p_col, 
                                                  const int na2, const int na, const int my_pcol, const int l_rows, const int l_rqs, const int l_rqe, 
                                                  const int matrixRows, const int gemm_dim_k) {
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

template <typename T>
void gpu_copy_q_slice_to_qtmp1(T *qtmp1_dev, T *q_dev, int *ndef_c_dev, int *l_col_out_dev, int *idx2_dev, int *p_col_dev, 
                               const int na2, const int na, const int my_pcol, const int l_rows, const int l_rqs, const int l_rqe, 
                               const int matrixRows, const int gemm_dim_k, const int debug, gpuStream_t my_stream) {
 
  dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK);
  dim3 blocks((l_rows + threadsPerBlock.x - 1) / threadsPerBlock.x);
  if (blocks.x==0) return;

#ifdef WITH_GPU_STREAMS
    gpu_copy_q_slice_to_qtmp1_kernel<T><<<blocks,threadsPerBlock,0,my_stream>>>(qtmp1_dev, q_dev, ndef_c_dev, l_col_out_dev, idx2_dev, p_col_dev, 
                                                                                na2, na2, my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k); 
#else
    gpu_copy_q_slice_to_qtmp1_kernel<T><<<blocks,threadsPerBlock>>>            (qtmp1_dev, q_dev, ndef_c_dev, l_col_out_dev, idx2_dev, p_col_dev, 
                                                                                na2, na2, my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k);
#endif
    
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess)
      printf("Error in executing gpu_copy_q_slice_to_qtmp1_kernel: %s\n",gpuGetErrorString(gpuerr));
    }
}

extern "C" void CONCATENATE(ELPA_GPU,  _copy_q_slice_to_qtmp1_FromC) (char dataType, intptr_t qtmp1_dev, intptr_t q_dev, 
                                                                      intptr_t ndef_c_dev, intptr_t l_col_out_dev, intptr_t idx2_dev, intptr_t p_col_dev, 
                                                                      int na2, int na, int my_pcol, int l_rows, int l_rqs, int l_rqe, 
                                                                      int matrixRows, int gemm_dim_k, int debug, gpuStream_t my_stream) {
  if      (dataType=='D') gpu_copy_q_slice_to_qtmp1<double>((double *) qtmp1_dev, (double *) q_dev, (int *) ndef_c_dev, (int *) l_col_out_dev, (int *) idx2_dev, (int *) p_col_dev,
                                                              na2, na, my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k, debug, my_stream);
  else if (dataType=='S') gpu_copy_q_slice_to_qtmp1<float> ((float  *) qtmp1_dev, (float  *) q_dev, (int *) ndef_c_dev, (int *) l_col_out_dev, (int *) idx2_dev, (int *) p_col_dev,
                                                              na2, na, my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k, debug, my_stream);
  else {
    printf("Error in gpu_copy_q_slice_to_qtmp1: Unsupported data type\n");
  }
}


//________________________________________________________________

template <typename T>
__global__ void gpu_copy_qtmp1_to_qtmp1_tmp_kernel(T* qtmp1, T* qtmp1_tmp, const int gemm_dim_k, const int gemm_dim_l) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i>=0 && i<gemm_dim_k) {
    if (j>=0 && j<gemm_dim_l) {
      qtmp1_tmp[i+gemm_dim_k*j] = qtmp1[i+gemm_dim_k*j];
    }    
  }           
} 

template <typename T>
void gpu_copy_qtmp1_to_qtmp1_tmp (T *qtmp1_dev, T* qtmp1_tmp_dev, const int gemm_dim_k, const int gemm_dim_l, 
                                  const int debug, gpuStream_t my_stream) {
  
  dim3 threadsPerBlock(32, 32);
  dim3 blocks((gemm_dim_k + threadsPerBlock.x - 1) / threadsPerBlock.x, (gemm_dim_l + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (blocks.x==0 || blocks.y==0) return;

#ifdef WITH_GPU_STREAMS
    gpu_copy_qtmp1_to_qtmp1_tmp_kernel<T><<<blocks,threadsPerBlock,0,my_stream>>>(qtmp1_dev, qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l);
#else
    gpu_copy_qtmp1_to_qtmp1_tmp_kernel<T><<<blocks,threadsPerBlock>>>            (qtmp1_dev, qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l);
#endif
    
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess)
      printf("Error in executing gpu_copy_qtmp1_to_qtmp1_tmp_kernel: %s\n",gpuGetErrorString(gpuerr));
    }
}

extern "C" void CONCATENATE(ELPA_GPU,  _copy_qtmp1_to_qtmp1_tmp_FromC) (char dataType, intptr_t qtmp1_dev, intptr_t qtmp1_tmp_dev, 
                                                                        int gemm_dim_k, int gemm_dim_l, int debug, gpuStream_t my_stream) {
  if      (dataType=='D') gpu_copy_qtmp1_to_qtmp1_tmp<double>((double *) qtmp1_dev, (double *) qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l, debug, my_stream);
  else if (dataType=='S') gpu_copy_qtmp1_to_qtmp1_tmp<float> ((float  *) qtmp1_dev, (float  *) qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l, debug, my_stream);
  else {
    printf("Error in gpu_copy_qtmp1_to_qtmp1_tmp: Unsupported data type\n");
  }
}

//________________________________________________________________

template <typename T>
__global__ void gpu_fill_array_kernel(T* array_dev, T* value_dev, int n) {
  int i0 = threadIdx.x + blockIdx.x*blockDim.x;

  for (int i=i0; i<n; i+=blockDim.x*gridDim.x) {
    array_dev[i] = *value_dev;
  }
}

template <typename T>
void gpu_fill_array (T *array_dev, T* value_dev, int n, int SM_count, int debug, gpuStream_t my_stream){
  
  dim3 blocks = dim3(SM_count,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);

#ifdef WITH_GPU_STREAMS
    gpu_fill_array_kernel<T><<<blocks,threadsPerBlock,0,my_stream>>>(array_dev, value_dev, n);
#else
    gpu_fill_array_kernel<T><<<blocks,threadsPerBlock>>>            (array_dev, value_dev, n);
#endif
    
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess)
      printf("Error in executing gpu_fill_array_kernel: %s\n",gpuGetErrorString(gpuerr));
    }
}

extern "C" void CONCATENATE(ELPA_GPU,  _fill_array_FromC) (char dataType, intptr_t array_dev, intptr_t value_dev, int n, int SM_count, int debug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_fill_array<double>((double *) array_dev, (double *) value_dev, n, SM_count, debug, my_stream);
  else if (dataType=='S') gpu_fill_array<float> ((float  *) array_dev, (float  *) value_dev, n, SM_count, debug, my_stream);
  else {
    printf("Error in gpu_fill_array: Unsupported data type\n");
  }
}

//________________________________________________________________

// Generic reduction ("sum") function within one thread block
template <typename T, typename Func>
__device__ T elpa_sum(int n, int tid, int threads_total, T* cache, Func func) {

  T sum = 0;
  for (int j = tid; j < n; j += threads_total) {
    sum += func(j);
  }
    
  cache[tid] = sum;
  __syncthreads();


  for (int stride = threads_total/2; stride > 0; stride /= 2) 
    {
    if (tid < stride) cache[tid] += cache[tid + stride];
    __syncthreads();
    }

  return cache[0];

}


template <typename T>
__forceinline__ __device__ void device_solve_secular_equation(int n, int i_f, T* d1, T* z1, T* delta, T* rho, T* cache,
                                                              int tid, int threads_total) {
  // i_f is the Fortran index (1-indexed); convert to C index:
  int i = i_f - 1;
  //T dshift;
  __shared__ T dshift_sh, a_sh, b_sh, x_sh, y_sh;
  
  __shared__ int break_flag_sh;
  if (tid==0) break_flag_sh=0;
  __syncthreads();

  const int maxIter = 200;
  T eps = (sizeof(T) == sizeof(double)) ? (T)1e-200 : (T)1e-20;

  if(i_f == n) 
    {
    // Special case: last eigenvalue.
    
    if (tid==0)
      {
      dshift_sh = d1[n-1];
      }
    __syncthreads();

    for (int j = tid; j < n; j+=threads_total) 
      {
      delta[j] = d1[j] - dshift_sh;
      }
    
    T sum_zsq = elpa_sum<T>(n, tid, threads_total, cache, [=] __device__ (int j) -> T {
      return z1[j]*z1[j];
    });

    if (tid==0)
      {
      a_sh = 0;
      b_sh = rho[0] * sum_zsq + 1;
      }
    __syncthreads();
    } 
  
  else 
    {
    // Other eigenvalues: lower bound is d1[i] and upper bound is d1[i+1]

    if (tid==0)
      {
      x_sh = 0.5*(d1[i] + d1[i+1]);
      }
    __syncthreads();

    T sum_term = elpa_sum<T>(n, tid, threads_total, cache, [=] __device__ (int j) -> T {
      return z1[j]*z1[j] / (d1[j] - x_sh);
    });

    if (tid==0)
      {
      y_sh = 1.0 + rho[0]*sum_term;
      if (y_sh > 0)
        dshift_sh = d1[i];
      else
        dshift_sh = d1[i+1];
      }
    __syncthreads();

    for (int j = tid; j < n; j += threads_total) 
      {
      delta[j] = d1[j] - dshift_sh;
      }

    __syncthreads(); // so all threads agree on delta and hence a and b

    if (tid==0)
      {
      a_sh = delta[i];
      b_sh = delta[i+1];
      }
    __syncthreads();
  }

  // Bisection
  for (int iter = 0; iter < maxIter; iter++) 
    {
    if (tid==0)
      {
      x_sh = 0.5*(a_sh + b_sh);
      if (x_sh == a_sh || x_sh == b_sh)
          break_flag_sh=1;  // no further subdivision possible
      if (fabs(x_sh) < eps)
          break_flag_sh;  // x is too close to zero (i.e. near a pole)
      }
    __syncthreads(); // so all threads agree on x and break_flag
    if (break_flag_sh) break;
    
    T sum_term = elpa_sum<T>(n, tid, threads_total, cache, [=] __device__ (int j) -> T {
      return z1[j]*z1[j] / (delta[j] - x_sh);
    });

    if (tid==0)
      {
      y_sh = 1.0 + rho[0]*sum_term;
      if (y_sh == 0)
          break_flag_sh=1;  // exact solution found
      else if (y_sh > 0)
          b_sh = x_sh;
      else
          a_sh = x_sh;
      }
    __syncthreads();
    if (break_flag_sh) break;
    }

  // Update delta: delta[j] = delta[j] - x for all j.
  for (int j = tid; j < n; j+=threads_total) 
    {
    delta[j] = delta[j] - x_sh;
    }
}

//________________________________________________________________

template <typename T>
__global__ void gpu_solve_secular_equation_loop_kernel(T *d1_dev, T *z1_dev, T *delta_extended_dev, T *rho_dev,
                                                       T *z_extended_dev, T *dbase_dev, T *ddiff_dev, 
                                                       int my_proc, int na1, int n_procs){
  __shared__ T cache[MAX_THREADS_PER_BLOCK]; 
  int tid = threadIdx.x;

  //int i_loc = threadIdx.x;
  //int j_loc = blockIdx.x ;

  // do i = my_procs+1, na1, n_procs
  //   call solve_secular_equation_&
  //                               &PRECISION&
  //                               &(obj, na1, i, d1, z1, delta, rho, s) 

  //   ! Compute updated z
  //   do j=1,na1
  //     if (i/=j)  z(j) = z(j)*( delta(j) / (d1(j)-d1(i)) )
  //   enddo

  //   z(i) = z(i)*delta(i)
    
  //   ! Store dbase/ddiff

  //   if (i<na1) then
  //     if (abs(delta(i+1)) < abs(delta(i))) then
  //       dbase(i) = d1(i+1)
  //       ddiff(i) = delta(i+1)
  //     else
  //       dbase(i) = d1(i)
  //       ddiff(i) = delta(i)
  //     endif
  //   else
  //     dbase(i) = d1(i)
  //     ddiff(i) = delta(i)
  //   endif
  // enddo

  // T dshift, a, b, x, y;
  // const int maxIter = 200;
  // T eps = (sizeof(T) == sizeof(double)) ? (T)1e-200 : (T)1e-20;

  for (int i=my_proc + n_procs*blockIdx.x; i<na1; i += n_procs*gridDim.x)
    {
    int i_f = i + 1; // i_f is the Fortran index (1-based)

    device_solve_secular_equation(na1, i_f, d1_dev, z1_dev, delta_extended_dev+na1*blockIdx.x, rho_dev, cache, tid, blockDim.x);
    __syncthreads(); // so all threads agree on delta_dev

    // Compute updated z (z_extended_dev)
    T d1_i = d1_dev[i];                     
    int index;                                                 
    for (int j = tid; j < na1; j+=blockDim.x)
      {
      index = j+na1*blockIdx.x;
      if (j != i) z_extended_dev[index] = z_extended_dev[index] * ( delta_extended_dev[index] / (d1_dev[j] - d1_i) );
      else z_extended_dev[index] = z_extended_dev[index] * delta_extended_dev[index];
      }

    // Store dbase/ddiff
    if (tid==0)
      {
      if (i_f < na1) 
        {
        if (fabs(delta_extended_dev[i+1 + na1*blockIdx.x]) < fabs(delta_extended_dev[i + na1*blockIdx.x])) 
          {
          dbase_dev[i] = d1_dev[i+1];
          ddiff_dev[i] = delta_extended_dev[i+1 + na1*blockIdx.x];
          }
        else 
          {
          dbase_dev[i] = d1_dev[i];
          ddiff_dev[i] = delta_extended_dev[i + na1*blockIdx.x];
          }
        } 
      else 
        {
        dbase_dev[i] = d1_dev[i];
        ddiff_dev[i] = delta_extended_dev[i + na1*blockIdx.x];
        }
      }
    }
}


template <typename T>
void gpu_solve_secular_equation_loop (T *d1_dev, T *z1_dev, T *delta_dev, T *rho_dev,
                                      T *z_dev, T *dbase_dev, T *ddiff_dev, 
                                      int my_proc, int na1, int n_procs, int SM_count, int debug, gpuStream_t my_stream){
  
  dim3 blocks = dim3(SM_count,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK/2,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_solve_secular_equation_loop_kernel<<<blocks,threadsPerBlock,0,my_stream>>> (d1_dev, z1_dev, delta_dev, rho_dev,
                                                                                  z_dev, dbase_dev, ddiff_dev, my_proc, na1, n_procs);
#else
  gpu_solve_secular_equation_loop_kernel<<<blocks,threadsPerBlock>>>             (d1_dev, z1_dev, delta_dev, rho_dev,
                                                                                  z_dev, dbase_dev, ddiff_dev, my_proc, na1, n_procs);
#endif

  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_solve_secular_equation_loop: %s\n",gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _solve_secular_equation_loop_FromC) (char dataType, intptr_t d1_dev, intptr_t z1_dev, intptr_t delta_dev, intptr_t rho_dev,
                                                                            intptr_t z_dev, intptr_t dbase_dev, intptr_t ddiff_dev, 
                                                                            int my_proc, int na1, int n_procs, int SM_count, int debug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_solve_secular_equation_loop<double>((double *) d1_dev, (double *) z1_dev, (double *) delta_dev, (double *) rho_dev,
                                                                  (double *) z_dev, (double *) dbase_dev, (double *) ddiff_dev,
                                                                  my_proc, na1, n_procs, SM_count, debug, my_stream);
  else if (dataType=='S') gpu_solve_secular_equation_loop<float> ((float  *) d1_dev, (float  *) z1_dev, (float  *) delta_dev, (float  *) rho_dev,
                                                                  (float  *) z_dev, (float  *) dbase_dev, (float  *) ddiff_dev,
                                                                  my_proc, na1, n_procs, SM_count, debug, my_stream);
  else {
    printf("Error in gpu_solve_secular_equation_loop: Unsupported data type\n");
  }
}

//________________________________________________________________

template <typename T>
__global__ void gpu_local_product_kernel(T *z_dev, T *z_extended_dev, int na1, int SM_count){
  
  int i0 = threadIdx.x;
  //int j0 = blockIdx.x;

  for (int j=0; j<SM_count; j+=1)
    for (int i=i0; i<na1; i+=blockDim.x)
      z_dev[i] = z_dev[i] * z_extended_dev[i + na1*j];
  
}

template <typename T>
void gpu_local_product(T *z_dev, T *z_extended_dev, int na1, int SM_count, int debug, gpuStream_t my_stream){

  dim3 blocks = dim3(1,1,1); // one block, so we don't need atomic_multiply
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_local_product_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(z_dev, z_extended_dev, na1, SM_count);
#else
  gpu_local_product_kernel<<<blocks,threadsPerBlock>>>            (z_dev, z_extended_dev, na1, SM_count);
#endif

  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_local_product: %s\n",gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _local_product_FromC) (char dataType, intptr_t z_dev, intptr_t z_extended_dev, 
                                                                            int na1, int SM_count, int debug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_local_product<double>((double *) z_dev, (double *) z_extended_dev, na1, SM_count, debug, my_stream);
  else if (dataType=='S') gpu_local_product<float> ((float  *) z_dev, (float  *) z_extended_dev, na1, SM_count, debug, my_stream);
  else {
    printf("Error in elpa_local_product: Unsupported data type\n");
  }
}

//________________________________________________________________

template <typename T>
__global__ void gpu_add_tmp_loop_kernel (T *d1_dev, T *dbase_dev, T *ddiff_dev, T *z_dev, T *ev_scale_dev, T *tmp_extended_dev, 
                                         int na1, int my_proc, int n_procs){
  
  // do i = my_proc+1, na1, n_procs ! work distributed over all processors
  //   tmp(1:na1) = d1(1:na1)  - dbase(i)
  //   tmp(1:na1) = tmp(1:na1) + ddiff(i)
  //   tmp(1:na1) = z(1:na1) / tmp(1:na1)
  //   ev_scale(i) = 1.0_rk/sqrt(dot_product(tmp(1:na1),tmp(1:na1)))
  // enddo

  __shared__ T cache[MAX_THREADS_PER_BLOCK];
  int tid = threadIdx.x;

  int index;
  T dbase_or_diff_i;
  for (int i=my_proc + n_procs*blockIdx.x; i<na1; i += n_procs*gridDim.x)
    {
    dbase_or_diff_i = dbase_dev[i];

    for (int j=tid; j<na1; j+=blockDim.x) 
      {
      index = j + na1*blockIdx.x;
      tmp_extended_dev[index] = d1_dev[j] - dbase_or_diff_i;
      }
    
    // separate loop to prevent compiler from optimization
    dbase_or_diff_i = ddiff_dev[i];
    for (int j=tid; j<na1; j+=blockDim.x)
      {
      index = j + na1*blockIdx.x;
      tmp_extended_dev[index] = tmp_extended_dev[index] + dbase_or_diff_i;
      tmp_extended_dev[index] = z_dev[j] / tmp_extended_dev[index];
      }
    
    T dot_product = elpa_sum<T>(na1, tid, blockDim.x, cache, [=] __device__ (int j) -> T {
      return tmp_extended_dev[j+na1*blockIdx.x]*tmp_extended_dev[j+na1*blockIdx.x];
    });
    ev_scale_dev[i] = 1.0/sqrt(dot_product);
    }
  
}

template <typename T>
void gpu_add_tmp_loop(T *d1_dev, T *dbase_dev, T *ddiff_dev, T *z_dev, T *ev_scale_dev, T *tmp_extended_dev, 
                      int na1, int my_proc, int n_procs, int SM_count, int debug, gpuStream_t my_stream){

  dim3 blocks = dim3(SM_count,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);
  
#ifdef WITH_GPU_STREAMS
  gpu_add_tmp_loop_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(d1_dev, dbase_dev, ddiff_dev, z_dev, ev_scale_dev, tmp_extended_dev,
                                                                  na1, my_proc, n_procs);
#else
  gpu_add_tmp_loop_kernel<<<blocks,threadsPerBlock>>>            (d1_dev, dbase_dev, ddiff_dev, z_dev, ev_scale_dev, tmp_extended_dev,
                                                                  na1, my_proc, n_procs);
#endif

  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_add_tmp_loop: %s\n",gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _add_tmp_loop_FromC)(char dataType, intptr_t d1_dev, intptr_t dbase_dev, 
                                                            intptr_t ddiff_dev, intptr_t z_dev, intptr_t ev_scale_dev, intptr_t tmp_extended_dev,  
                                                            int na1, int my_proc, int n_procs, int SM_count, int debug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_add_tmp_loop<double>((double *) d1_dev, (double *) dbase_dev, (double *) ddiff_dev, (double *) z_dev, (double *) ev_scale_dev, (double *) tmp_extended_dev, na1, my_proc, n_procs, SM_count, debug, my_stream);
  else if (dataType=='S') gpu_add_tmp_loop<float> ((float  *) d1_dev, (float  *) dbase_dev, (float  *) ddiff_dev, (float  *) z_dev, (float  *) ev_scale_dev, (float  *) tmp_extended_dev, na1, my_proc, n_procs, SM_count, debug, my_stream);
  else {
    printf("Error in elpa_add_tmp_loop: Unsupported data type\n");
  }
}

//________________________________________________________________

template <typename T>
__global__ void gpu_copy_qtmp1_q_compute_nnzu_nnzl_kernel(T *qtmp1_dev, T *q_dev, int *p_col_dev, int *l_col_dev, int *idx1_dev, int *coltyp_dev, int *nnzul_dev,
                                               int na1, int l_rnm, int l_rqs, int l_rqm, int l_rows, int my_pcol, int ldq_tmp1, int ldq){
  
  // do i = 1, na1
  //   if (p_col(idx1(i))==my_pcol) then
  //     l_idx = l_col(idx1(i))
  //     if (coltyp(idx1(i))==1 .or. coltyp(idx1(i))==2) then
  //       nnzu = nnzu+1
  //       qtmp1(1:l_rnm,nnzu) = q(l_rqs:l_rqm,l_idx)
  //     endif

  //     if (coltyp(idx1(i))==3 .or. coltyp(idx1(i))==2) then
  //       nnzl = nnzl+1
  //       qtmp1(l_rnm+1:l_rows,nnzl) = q(l_rqm+1:l_rqe,l_idx)
  //     endif
  //   endif
  // enddo
  
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

  int nnzu = 0;
  int nnzl = 0;
  for (int i=0; i<na1; i++)
    {
    int idx = idx1_dev[i]-1;
    if (p_col_dev[idx] == my_pcol)
      {
      int l_idx = l_col_dev[idx];
      if (coltyp_dev[idx] == 1 || coltyp_dev[idx] == 2)
        {
        nnzu += 1;

        for (int j=tid; j<l_rnm; j+=blockDim.x*gridDim.x)
          qtmp1_dev[     j + (nnzu-1)*ldq_tmp1] = q_dev[l_rqs-1+j + (l_idx-1)*ldq];
        }
      if (coltyp_dev[idx] == 3 || coltyp_dev[idx] == 2)
        {
        nnzl += 1;

        for (int j=tid; j<l_rows-l_rnm; j+=blockDim.x*gridDim.x)
          qtmp1_dev[l_rnm+j + (nnzl-1)*ldq_tmp1] = q_dev[l_rqm +j + (l_idx-1)*ldq];
        }
      }
    }

  if (tid==0)
    {
    nnzul_dev[0] = nnzu;
    nnzul_dev[1] = nnzl;
    }
}

template <typename T>
void gpu_copy_qtmp1_q_compute_nnzu_nnzl(T *qtmp1_dev, T *q_dev, int *p_col_dev, int *l_col_dev, int *idx1_dev, int *coltyp_dev, int *nnzul_dev,
                                        int na1, int l_rnm, int l_rqs, int l_rqm, int l_rows, int my_pcol, int ldq_tmp1, int ldq, int SM_count,
                                        int debug, gpuStream_t my_stream){

  dim3 blocks = dim3(SM_count,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK,1,1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_qtmp1_q_compute_nnzu_nnzl_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(qtmp1_dev, q_dev, p_col_dev, l_col_dev, idx1_dev, coltyp_dev, nnzul_dev,
                                                                                    na1, l_rnm, l_rqs, l_rqm, l_rows, my_pcol, ldq_tmp1, ldq);
#else
  gpu_copy_qtmp1_q_compute_nnzu_nnzl_kernel<<<blocks,threadsPerBlock>>>            (qtmp1_dev, q_dev, p_col_dev, l_col_dev, idx1_dev, coltyp_dev, nnzul_dev,
                                                                                    na1, l_rnm, l_rqs, l_rqm, l_rows, my_pcol, ldq_tmp1, ldq);
#endif

  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_copy_qtmp1_q_compute_nnzu_nnzl: %s\n",gpuGetErrorString(gpuerr));
    }
  }
}   

extern "C" void CONCATENATE(ELPA_GPU,  _copy_qtmp1_q_compute_nnzu_nnzl_FromC)(char dataType, intptr_t qtmp1_dev, intptr_t q_dev, 
                                                                              intptr_t p_col_dev, intptr_t l_col_dev, intptr_t idx1_dev, intptr_t coltyp_dev, intptr_t nnzul_dev,
                                                                              int na1, int l_rnm, int l_rqs, int l_rqm, int l_rows, int my_pcol, int ldq_tmp1, int ldq, 
                                                                              int SM_count, int debug, gpuStream_t my_stream){

  if      (dataType=='D') gpu_copy_qtmp1_q_compute_nnzu_nnzl<double>((double *) qtmp1_dev, (double *) q_dev, (int *) p_col_dev, (int *) l_col_dev, (int *) idx1_dev, (int *) coltyp_dev, (int *) nnzul_dev, 
                                                                      na1, l_rnm, l_rqs, l_rqm, l_rows, my_pcol, ldq_tmp1, ldq, 
                                                                      SM_count, debug, my_stream);
  else if (dataType=='S') gpu_copy_qtmp1_q_compute_nnzu_nnzl<float> ((float  *) qtmp1_dev, (float  *) q_dev, (int *) p_col_dev, (int *) l_col_dev, (int *) idx1_dev, (int *) coltyp_dev, (int *) nnzul_dev, 
                                                                      na1, l_rnm, l_rqs, l_rqm, l_rows, my_pcol, ldq_tmp1, ldq, 
                                                                      SM_count, debug, my_stream);
  else {
    printf("Error in elpa_copy_qtmp1_q_compute_nnzu_nnzl: Unsupported data type\n");
  }
}

//________________________________________________________________

template <typename T>
__global__ void gpu_fill_z_kernel (T *z_dev, T *q_dev, int *p_col_dev, int *l_col_dev, 
                        int sig_int, int na, int my_pcol, int row_q, int ldq){
  
  // do i = 1, na
  //   if (p_col(i)==my_pcol) z(i) = z(i) + sig*q(l_rqm+1,l_col(i))
  // enddo

  //int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int i0 = blockIdx.x;

  for (int i=i0; i<na; i+=gridDim.x)
    {
    if (p_col_dev[i] == my_pcol) // uncoalesced access here
      {
      z_dev[i] = z_dev[i] + T(sig_int)*q_dev[row_q-1 + (l_col_dev[i]-1)*ldq];
      }
    }
}

template <typename T>
void gpu_fill_z(T *z_dev, T *q_dev, int *p_col_dev, int *l_col_dev, int sig_int, int na, int my_pcol, int row_q, int ldq, 
                int SM_count, int debug, gpuStream_t my_stream){

  dim3 blocks = dim3(SM_count,1,1);
  dim3 threadsPerBlock = dim3(1,1,1); // uncoalesced access, so only one thread per block

#ifdef WITH_GPU_STREAMS
  gpu_fill_z_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(z_dev, q_dev, p_col_dev, l_col_dev, sig_int, na, my_pcol, row_q, ldq);
#else
  gpu_fill_z_kernel<<<blocks,threadsPerBlock>>>            (z_dev, q_dev, p_col_dev, l_col_dev, sig_int, na, my_pcol, row_q, ldq);
#endif
  
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_fill_z: %s\n",gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _fill_z_FromC)(char dataType, intptr_t z_dev, intptr_t q_dev, 
                                                      intptr_t p_col_dev, intptr_t l_col_dev,
                                                      int sig_int, int na, int my_pcol, int row_q, int ldq, 
                                                      int SM_count, int debug, gpuStream_t my_stream){

  if      (dataType=='D') gpu_fill_z<double>((double *) z_dev, (double *) q_dev, (int *) p_col_dev, (int *) l_col_dev, 
                                             sig_int, na, my_pcol, row_q, ldq, 
                                             SM_count, debug, my_stream);
  else if (dataType=='S') gpu_fill_z<float> ((float  *) z_dev, (float  *) q_dev, (int *) p_col_dev, (int *) l_col_dev, 
                                             sig_int, na, my_pcol, row_q, ldq, 
                                             SM_count, debug, my_stream);
  else {
    printf("Error in elpa_fill_z: Unsupported data type\n");
  }
}