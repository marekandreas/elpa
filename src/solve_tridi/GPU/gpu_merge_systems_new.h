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
//    This file was written by P. Karpov, MPCDF

//________________________________________________________________


// PETERDEBUG111: create a separate Fortran function for this
template <typename T>
__global__ void elpa_fill_ones_kernel(T* array, int n) {
  int i0 = threadIdx.x + blockIdx.x*blockDim.x;

  for (int i=i0; i<n; i+=blockDim.x*gridDim.x) {
    array[i] = 1.0;
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
                                                       int my_proc, int na1, int n_procs, int myid){
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
                                      int my_proc, int na1, int n_procs, int myid, int SM_count, int debug, gpuStream_t my_stream){
  
  dim3 blocks = dim3(SM_count,1,1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK/2,1,1);


      // PETERDEBUG111: extract to a separate kernel
#ifdef WITH_GPU_STREAMS
    elpa_fill_ones_kernel<T><<<blocks,threadsPerBlock,0,my_stream>>>(z_dev, na1*SM_count);
#else
    elpa_fill_ones_kernel<T><<<blocks,threadsPerBlock>>>(z_dev, na1*SM_count);
#endif
    
    if (debug) // PETERDEBUG111
      {
      gpuDeviceSynchronize();
      gpuError_t gpuerr = gpuGetLastError();
      if (gpuerr != gpuSuccess)
        printf("Error in executing elpa_fill_ones_kernel: %s\n",gpuGetErrorString(gpuerr));
      }
    }


#ifdef WITH_GPU_STREAMS
  gpu_solve_secular_equation_loop_kernel<<<blocks,threadsPerBlock,0,my_stream>>> (d1_dev, z1_dev, delta_dev, rho_dev,
                                                                                  z_dev, dbase_dev, ddiff_dev, my_proc, na1, n_procs, myid);
#else
  gpu_solve_secular_equation_loop_kernel<<<blocks,threadsPerBlock>>>             (d1_dev, z1_dev, delta_dev, rho_dev,
                                                                                  z_dev, dbase_dev, ddiff_dev, my_proc, na1, n_procs, myid);
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
                                                                            int my_proc, int na1, int n_procs, int myid, int SM_count, int debug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_solve_secular_equation_loop<double>((double *) d1_dev, (double *) z1_dev, (double *) delta_dev, (double *) rho_dev,
                                                                  (double *) z_dev, (double *) dbase_dev, (double *) ddiff_dev,
                                                                  my_proc, na1, n_procs, myid, SM_count, debug, my_stream);
  else if (dataType=='S') gpu_solve_secular_equation_loop<float> ((float  *) d1_dev, (float  *) z1_dev, (float  *) delta_dev, (float  *) rho_dev,
                                                                  (float  *) z_dev, (float  *) dbase_dev, (float  *) ddiff_dev,
                                                                  my_proc, na1, n_procs, myid, SM_count, debug, my_stream);
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