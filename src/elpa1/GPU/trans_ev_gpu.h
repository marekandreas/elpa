//    Copyright 2024, P. Karpov
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

//_________________________________________________________________________________________________

template <typename T>
__global__ void gpu_copy_hvb_a_kernel(T *hvb_dev, T *a_dev, int ld_hvb, int lda, int my_prow, int np_rows,
                                      int my_pcol, int np_cols, int nblk, int ics, int ice) {
  // nb = 0
  // do ic = ics, ice
  //   l_colh = local_index(ic  , my_pcol, np_cols, nblk, -1) ! Column of Householder Vector
  //   l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1) ! # rows of Householder Vector

  //   if (my_pcol == cur_pcol) then
  //     hvb(nb+1:nb+l_rows) = a_mat(1:l_rows,l_colh)
  //     if (my_prow == prow(ic-1, nblk, np_rows)) then
  //       hvb(nb+l_rows) = 1.
  //     endif
  //   endif

  //   nb = nb+l_rows
  // enddo

  int i0   = threadIdx.x; // 0..ld_hvb-1; max(l_rows) = ld_hvb
  int ic_0 = blockIdx.x ;

  T One = elpaDeviceNumber<T>(1.0);

  for (int ic = ic_0 + ics; ic <= ice; ic+=gridDim.x) {
    int l_colh = local_index(ic  , my_pcol, np_cols, nblk, -1); // Column of Householder Vector
    int l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1); // Number of rows of Householder Vector

    // if (my_pcol == cur_pcol) // already true
    for (int i=i0; i < l_rows; i+=blockDim.x) {
      hvb_dev[i + ld_hvb*(ic-ics)] = a_dev[i + (l_colh-1)*lda]; // nb -> ld_hvb*(ic-ics), no compression
    }
    
    if (my_prow == prow(ic-1, nblk, np_rows) && threadIdx.x == (l_rows-1)%blockDim.x) {
      hvb_dev[(l_rows-1) + ld_hvb*(ic-ics)] = One;
    }
    
  }
}

template <typename T>
void gpu_copy_hvb_a(T *hvb_dev, T *a_dev, int ld_hvb, int lda, int my_prow, int np_rows,
                    int my_pcol, int np_cols, int nblk, int ics, int ice, int SM_count, int debug, gpuStream_t my_stream){

  dim3 blocks = dim3(SM_count, 1, 1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK, 1, 1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_hvb_a_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(hvb_dev, a_dev, ld_hvb, lda, my_prow, np_rows, my_pcol, np_cols, nblk, ics, ice);
#else
  gpu_copy_hvb_a_kernel<<<blocks,threadsPerBlock>>>            (hvb_dev, a_dev, ld_hvb, lda, my_prow, np_rows, my_pcol, np_cols, nblk, ics, ice);
#endif

  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_copy_hvb_a: %s\n", gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _copy_hvb_a_FromC) (char dataType, intptr_t hvb_dev, intptr_t a_dev,
                                      int ld_hvb, int lda, int my_prow, int np_rows,
                                      int my_pcol, int np_cols, int nblk, int ics, int ice, 
                                      int SM_count, int debug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_copy_hvb_a<double>((double *) hvb_dev, (double *) a_dev, ld_hvb, lda, my_prow, np_rows, my_pcol, np_cols, nblk, ics, ice, SM_count, debug, my_stream);
  else if (dataType=='S') gpu_copy_hvb_a<float> ((float  *) hvb_dev, (float  *) a_dev, ld_hvb, lda, my_prow, np_rows, my_pcol, np_cols, nblk, ics, ice, SM_count, debug, my_stream);
  else if (dataType=='Z') gpu_copy_hvb_a<gpuDoubleComplex>((gpuDoubleComplex *) hvb_dev, (gpuDoubleComplex *) a_dev, ld_hvb, lda, my_prow, np_rows, my_pcol, np_cols, nblk, ics, ice, SM_count, debug, my_stream);
  else if (dataType=='C') gpu_copy_hvb_a<gpuFloatComplex> ((gpuFloatComplex  *) hvb_dev, (gpuFloatComplex  *) a_dev, ld_hvb, lda, my_prow, np_rows, my_pcol, np_cols, nblk, ics, ice, SM_count, debug, my_stream);
}

//_________________________________________________________________________________________________

template <typename T>
__global__ void gpu_copy_hvm_hvb_kernel(T *hvm_dev, const T *hvb_dev, const T *tau_dev, 
                                        int ld_hvm, int ld_hvb, int my_prow, int np_rows,
                                        int nstor, int nblk, int ics, int ice) {
  // nb = 0
  // NVTX_RANGE_PUSH("loop: copy hvm <- hvb")
  // do ic = ics, ice
  //   l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1) ! # rows of Householder Vector
  //    ! if tau==0, reflector is identity => make this column inactive
  //    if (tau(ic) == ZERO) then
  //      hvm(1:l_rows, nstor+1) = 0 ! PETERDEBUG111: cleanup, it's already zero?
  //    else
  //      hvm(1:l_rows, nstor+1) = hvb(nb+1:nb+l_rows)
  //    endif
  //   nstor = nstor+1
  //   nb = nb+l_rows
  // enddo

  int i0   = threadIdx.x; // 0..ld_hvm-1; max(l_rows) = ld_hvm
  int ic_0 = blockIdx.x ;

  T Zero = elpaDeviceNumber<T>(0.0);
  
  for (int ic = ic_0 + ics; ic <= ice; ic+=gridDim.x) {
    int l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1);
    int shift_hvm = ld_hvm*(ic-ics+nstor);

    if (elpaDeviceEqualBool(tau_dev[ic-1], Zero)) {
      for (int i=i0; i < ld_hvm; i+=blockDim.x) {
        hvm_dev[i + shift_hvm] = Zero; // ! PETERDEBUG111: cleanup, it's already zero?
      }
      continue;
    }

    for (int i=i0; i < l_rows; i+=blockDim.x) {
      hvm_dev[i + shift_hvm] = hvb_dev[i + ld_hvb*(ic-ics)]; // nb -> ld_hvb*(ic-ics), no compression
    }
    
    for (int i=l_rows+i0; i < ld_hvm; i+=blockDim.x) {
      hvm_dev[i + shift_hvm] = Zero; // since we're not compressing, we need to take extra care to clear from previous iterations
    }

  }
}

template <typename T>
void gpu_copy_hvm_hvb(T *hvm_dev, T *hvb_dev, const T *tau_dev,
                      int ld_hvm, int ld_hvb, int my_prow, int np_rows,
                      int nstor, int nblk, int ics, int ice, int SM_count, int debug, gpuStream_t my_stream){

  dim3 blocks = dim3(SM_count, 1, 1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK, 1, 1);

#ifdef WITH_GPU_STREAMS
  gpu_copy_hvm_hvb_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(hvm_dev, hvb_dev, tau_dev, ld_hvm, ld_hvb, my_prow, np_rows, nstor, nblk, ics, ice);
#else
  gpu_copy_hvm_hvb_kernel<<<blocks,threadsPerBlock>>>            (hvm_dev, hvb_dev, tau_dev, ld_hvm, ld_hvb, my_prow, np_rows, nstor, nblk, ics, ice);
#endif
  
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_copy_hvm_hvb: %s\n", gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _copy_hvm_hvb_FromC) (char dataType, intptr_t hvm_dev, intptr_t hvb_dev, intptr_t tau_dev,
                                      int ld_hvm, int ld_hvb, int my_prow, int np_rows,
                                      int nstor, int nblk, int ics, int ice, 
                                      int SM_count, int debug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_copy_hvm_hvb<double>((double *) hvm_dev, (double *) hvb_dev, (double *) tau_dev, ld_hvm, ld_hvb, my_prow, np_rows, nstor, nblk, ics, ice, SM_count, debug, my_stream);
  else if (dataType=='S') gpu_copy_hvm_hvb<float> ((float  *) hvm_dev, (float  *) hvb_dev, (float  *) tau_dev, ld_hvm, ld_hvb, my_prow, np_rows, nstor, nblk, ics, ice, SM_count, debug, my_stream);
  else if (dataType=='Z') gpu_copy_hvm_hvb<gpuDoubleComplex>((gpuDoubleComplex *) hvm_dev, (gpuDoubleComplex *) hvb_dev, (gpuDoubleComplex *) tau_dev, ld_hvm, ld_hvb, my_prow, np_rows, nstor, nblk, ics, ice, SM_count, debug, my_stream);
  else if (dataType=='C') gpu_copy_hvm_hvb<gpuFloatComplex> ((gpuFloatComplex  *) hvm_dev, (gpuFloatComplex  *) hvb_dev, (gpuFloatComplex  *) tau_dev, ld_hvm, ld_hvb, my_prow, np_rows, nstor, nblk, ics, ice, SM_count, debug, my_stream);
}

//_________________________________________________________________________________________________


// PETERDEBUG: cleanup parameter nc (not needed anymore, we use shift_h_dev instead)
template <typename T>
__global__ void gpu_update_tmat_kernel(T *tmat_dev, T *h_dev, T *tau_curr_dev, int max_stored_rows, int nc, int n) {

//    ! update tmat for next iteration
// #if REALCASE == 1
//    tmat(n+1,1:n) = -h(nc+1:nc+n) *tau(ice-nstor+n+1)
// #elif COMPLEXCASE == 1
//    tmat(n+1,1:n) = -conjg(h(nc+1:nc+n)) *tau(ice-nstor+n+1)
// #endif    
//    tmat(n+1,n+1) = tau(ice-nstor+n+1)

  int j0  = blockIdx.x;

  // PETERDEBUG: cleanup
  // for (int j=j0; j<n; j+=gridDim.x) {
  //   tmat_dev[n + j*max_stored_rows] = h_dev[j];
  //   tmat_dev[n + j*max_stored_rows] = elpaDeviceMultiply(elpaDeviceComplexConjugate(h_dev[j]), (*tau_curr_dev));
  //   tmat_dev[n + j*max_stored_rows] = elpaDeviceMultiply(elpaDeviceNumber<T>(-1.0), tmat_dev[n + j*max_stored_rows]);
  // }

  if (blockIdx.x==0) {
    tmat_dev[n + n*max_stored_rows] = *tau_curr_dev;
    //printf("gpu_update_tmat: tau_curr_dev=%f\n", *tau_curr_dev); // PETERDEBUG
  }
  

  // PETERDEBUG: work directly with the non-transposed matrix --> better data access by threads
  // int i0   = threadIdx.x + blockIdx.x*blockDim.x;
  // for (int i=i0; i<n; i+=blockDim.x*gridDim.x) {
  //   tmat_dev[i + n*max_stored_rows] = elpaDeviceMultiply(h_dev[i], (*tau_curr_dev));
  //   tmat_dev[i + n*max_stored_rows] = elpaDeviceMultiply(elpaDeviceNumber<T>(-1.0), tmat_dev[i + n*max_stored_rows]);
  // }
  //
  // if (i0==0) tmat_dev[n + n*max_stored_rows] = *tau_curr_dev; // PETERDEBUG: only one thread does this
}

template <typename T>
void gpu_update_tmat(T *tmat_dev, T *h_dev, T *tau_curr_dev, int max_stored_rows, int nc, int n, int SM_count, int debug, gpuStream_t my_stream){

  // SM_count*MIN_THREADS_PER_BLOCK is the minimal GPU configuration that keeps the GPU busy
  dim3 blocks = dim3(SM_count, 1, 1);
  dim3 threadsPerBlock = dim3(1, 1, 1);
 
#ifdef WITH_GPU_STREAMS
  gpu_update_tmat_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(tmat_dev, h_dev, tau_curr_dev, max_stored_rows, nc, n);
#else
  gpu_update_tmat_kernel<<<blocks,threadsPerBlock>>>            (tmat_dev, h_dev, tau_curr_dev, max_stored_rows, nc, n);
#endif

  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_update_tmat: %s\n", gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _update_tmat_FromC) (char dataType, intptr_t tmat_dev, intptr_t h_dev, intptr_t tau_curr_dev,
                                      int max_stored_rows, int nc, int n, int SM_count, int debug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_update_tmat<double>((double *) tmat_dev, (double *) h_dev, (double *) tau_curr_dev, max_stored_rows, nc, n, SM_count, debug, my_stream);
  else if (dataType=='S') gpu_update_tmat<float> ((float  *) tmat_dev, (float  *) h_dev, (float  *) tau_curr_dev, max_stored_rows, nc, n, SM_count, debug, my_stream);
  else if (dataType=='Z') gpu_update_tmat<gpuDoubleComplex>((gpuDoubleComplex *) tmat_dev, (gpuDoubleComplex *) h_dev, (gpuDoubleComplex *) tau_curr_dev, max_stored_rows, nc, n, SM_count, debug, my_stream);
  else if (dataType=='C') gpu_update_tmat<gpuFloatComplex> ((gpuFloatComplex  *) tmat_dev, (gpuFloatComplex  *) h_dev, (gpuFloatComplex  *) tau_curr_dev, max_stored_rows, nc, n, SM_count, debug, my_stream);
}

//_________________________________________________________________________________________________

template <typename T>
__global__ void gpu_set_tmat_diag_from_tau_kernel(T *tmat_dev, T *tau_dev, int max_stored_rows, int nstor, int tau_offset) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < nstor) { // PETERDEBUG111: grid-stride loop instead?
    T One = elpaDeviceNumber<T>(1.0);
    T Zero = elpaDeviceNumber<T>(0.0);
    
    // PETERDEBUG111: cleanup
    //T mask = elpaDeviceNumber<T>(tau_dev[tau_offset + i] == Zero);
    //tmat_dev[i + i*max_stored_rows] = elpaDeviceMultiply(elpaDeviceSubtract(One, mask), 
    //                                                     elpaDeviceDivide(One, tau_dev[tau_offset + i]));
    
    T tau = tau_dev[i+tau_offset];
    //T is_zero = elpaDeviceEqual(tau, Zero);          // returns 0 or 1 of type T

    // If tau==0 -> tau_safe = 1
    // If tau!=0 -> tau_safe = tau
    
    //T tau_safe = elpaDeviceAdd(tau, elpaDeviceMultiply(is_zero, One));

    //tmat_dev[i + i*max_stored_rows] = elpaDeviceDivide(One, tau_safe);

    tmat_dev[i + i*max_stored_rows] =  elpaDeviceEqualBool(tau, Zero) ? One : elpaDeviceDivide(One, tau);
  }
}

template <typename T>
void gpu_set_tmat_diag_from_tau(T *tmat_dev, T *tau_dev, int max_stored_rows, int nstor, int tau_offset,
                                int SM_count, int debug, gpuStream_t my_stream) {

  int threads = MAX_THREADS_PER_BLOCK;
  int blocks = (nstor + threads - 1) / threads;
  if (blocks <= 0) return;

#ifdef WITH_GPU_STREAMS
  gpu_set_tmat_diag_from_tau_kernel<<<blocks,threads,0,my_stream>>>(tmat_dev, tau_dev, max_stored_rows, nstor, tau_offset);
#else
  gpu_set_tmat_diag_from_tau_kernel<<<blocks,threads>>>            (tmat_dev, tau_dev, max_stored_rows, nstor, tau_offset);
#endif

  if (debug) {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_set_tmat_diag_from_tau: %s\n", gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _set_tmat_diag_from_tau_FromC) (char dataType, intptr_t tmat_dev, intptr_t tau_dev,
                                      int max_stored_rows, int nstor, int tau_offset, int SM_count,
                                      int debug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_set_tmat_diag_from_tau<double>((double *) tmat_dev, (double *) tau_dev, max_stored_rows, nstor, tau_offset, SM_count, debug, my_stream);
  else if (dataType=='S') gpu_set_tmat_diag_from_tau<float> ((float  *) tmat_dev, (float  *) tau_dev, max_stored_rows, nstor, tau_offset, SM_count, debug, my_stream);
  else if (dataType=='Z') gpu_set_tmat_diag_from_tau<gpuDoubleComplex>((gpuDoubleComplex *) tmat_dev, (gpuDoubleComplex *) tau_dev, max_stored_rows, nstor, tau_offset, SM_count, debug, my_stream);
  else if (dataType=='C') gpu_set_tmat_diag_from_tau<gpuFloatComplex> ((gpuFloatComplex  *) tmat_dev, (gpuFloatComplex  *) tau_dev, max_stored_rows, nstor, tau_offset, SM_count, debug, my_stream);
}

//_________________________________________________________________________________________________
// PETERDEBUG: delete result_buffer_dev

template <typename T>
__global__ void gpu_trmv_kernel(T *tmat_dev, T *h_dev, T *result_buffer_dev, T *tau_curr_dev, int max_stored_rows, int n) {
  //__shared__ T cache[MIN_THREADS_PER_BLOCK*MIN_THREADS_PER_BLOCK]; // extra space of fixed size is reserved for a speedup
  __shared__ T cache[MAX_THREADS_PER_BLOCK];

  // h_dev <- tmat_dev^T*h_dev
  // uncoalesced memory access in gpu_update_tmat_kernel, coalesced access here.
  // tmat_dev is lower triangular

  int i0 = threadIdx.x; 
  int j0 = blockIdx.x;

  T Zero = elpaDeviceNumber<T>(0.0);

  // for (int n=0; n<nstor-1; n++)
  //   {
  
    for (int j=j0; j<n; j+=gridDim.x) {
      //if (threadIdx.x==0) result_buffer_dev[j] = Zero;
      if (threadIdx.x==0) tmat_dev[n + j*max_stored_rows] = Zero; // PETERDEBUG: delete, unneeded

      T slice_sum = Zero;
      for (int i=i0 + j; i<n; i+=blockDim.x) { // j because tmat_dev is lower triangular
        slice_sum = elpaDeviceAdd(slice_sum, elpaDeviceMultiply(elpaDeviceComplexConjugate(tmat_dev[i + j*max_stored_rows]), h_dev[i]));
      }

      cache[threadIdx.x] = slice_sum;
      __syncthreads();

      // for reductions, threadsPerBlock=blockDim.x must be a power of 2
      int di = blockDim.x/2;
      while (di > 0) {
        if (threadIdx.x < di) cache[threadIdx.x] = elpaDeviceAdd(cache[threadIdx.x], cache[threadIdx.x + di]);
        __syncthreads();
        di /= 2;
      }

      // if (threadIdx.x==0) {
      //   //atomicAdd(&result_buffer_dev[j], cache[0]); // PETERDEBUG: cleanup
      //   result_buffer_dev[j] = cache[0];
      //   result_buffer_dev[j] = elpaDeviceMultiply(elpaDeviceComplexConjugate(result_buffer_dev[j]), (*tau_curr_dev));
      //   result_buffer_dev[j] = elpaDeviceMultiply(elpaDeviceNumber<T>(-1.0), result_buffer_dev[j]);
      //   printf("gpu_trmv: tau_curr_dev=%f\n", *tau_curr_dev); // PETERDEBUG
      //   }

      if (threadIdx.x==0) {
        //atomicAdd(&tmat_dev[n + j*max_stored_rows], cache[0]);
        int idx = n + j*max_stored_rows;
        tmat_dev[idx] = cache[0];
        tmat_dev[idx] = elpaDeviceMultiply(elpaDeviceComplexConjugate(tmat_dev[idx]), (*tau_curr_dev));
        tmat_dev[idx] = elpaDeviceMultiply(elpaDeviceNumber<T>(-1.0), tmat_dev[idx]);
        }
      }

    if (threadIdx.x==0 && blockIdx.x==0) tmat_dev[n + n*max_stored_rows] = *tau_curr_dev;
    //}
}

template <typename T>
void gpu_trmv(T *tmat_dev, T *h_dev, T *result_buffer_dev, T *tau_curr_dev, int max_stored_rows, int n, int SM_count, int debug, gpuStream_t my_stream){

  dim3 blocks = dim3(SM_count, 1, 1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK, 1, 1);

#ifdef WITH_GPU_STREAMS
  gpu_trmv_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(tmat_dev, h_dev, result_buffer_dev, tau_curr_dev, max_stored_rows, n);
#else
  gpu_trmv_kernel<<<blocks,threadsPerBlock>>>            (tmat_dev, h_dev, result_buffer_dev, tau_curr_dev, max_stored_rows, n);
#endif
  
  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_trmv: %s\n", gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _trmv_FromC) (char dataType, intptr_t tmat_dev, intptr_t h_dev, intptr_t result_buffer_dev, intptr_t tau_curr_dev,
                                      int max_stored_rows, int n, int SM_count, int debug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_trmv<double>((double *) tmat_dev, (double *) h_dev, (double *) result_buffer_dev, (double *) tau_curr_dev, max_stored_rows, n, SM_count, debug, my_stream);
  else if (dataType=='S') gpu_trmv<float> ((float  *) tmat_dev, (float  *) h_dev, (float  *) result_buffer_dev, (float  *) tau_curr_dev, max_stored_rows, n, SM_count, debug, my_stream);
  else if (dataType=='Z') gpu_trmv<gpuDoubleComplex>((gpuDoubleComplex *) tmat_dev, (gpuDoubleComplex *) h_dev, (gpuDoubleComplex *) result_buffer_dev, (gpuDoubleComplex *) tau_curr_dev, max_stored_rows, n, SM_count, debug, my_stream);
  else if (dataType=='C') gpu_trmv<gpuFloatComplex> ((gpuFloatComplex  *) tmat_dev, (gpuFloatComplex  *) h_dev, (gpuFloatComplex  *) result_buffer_dev, (gpuFloatComplex  *) tau_curr_dev, max_stored_rows, n, SM_count, debug, my_stream);
}

template <typename T>
void gpu_trmv_loop(T *tmat_dev, T *h_dev, T *result_buffer_dev, T *tau_curr_dev, int max_stored_rows, int nstor, int ice, int SM_count, int useCCL, int debug, gpuStream_t my_stream){

  dim3 blocks = dim3(SM_count, 1, 1);
  dim3 threadsPerBlock = dim3(MAX_THREADS_PER_BLOCK, 1, 1);

  int size_of_datatype = sizeof(T);
  int shift_h_dev, shift_tau_dev;
  int nc = 0;

  for (int n=1; n <= nstor-1; n++)
    {
    if (useCCL) {
      shift_h_dev = n*max_stored_rows;
    }
    else {
      shift_h_dev = nc;
      nc = nc+n;
    }
    shift_tau_dev = (ice-nstor+n);

#ifdef WITH_GPU_STREAMS
    gpu_trmv_kernel<<<blocks,threadsPerBlock,0,my_stream>>>(tmat_dev, h_dev+shift_h_dev, result_buffer_dev, tau_curr_dev+shift_tau_dev, max_stored_rows, nstor);
#else
    gpu_trmv_kernel<<<blocks,threadsPerBlock>>>            (tmat_dev, h_dev+shift_h_dev, result_buffer_dev, tau_curr_dev+shift_tau_dev, max_stored_rows, nstor);
#endif
    }

  if (debug)
    {
    gpuDeviceSynchronize();
    gpuError_t gpuerr = gpuGetLastError();
    if (gpuerr != gpuSuccess){
      printf("Error in executing gpu_trmv_loop: %s\n", gpuGetErrorString(gpuerr));
    }
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _trmv_loop_FromC) (char dataType, intptr_t tmat_dev, intptr_t h_dev, intptr_t result_buffer_dev, intptr_t tau_curr_dev,
                                      int max_stored_rows, int nstor, int ice, int SM_count, int useCCL, int debug, gpuStream_t my_stream){
  if      (dataType=='D') gpu_trmv_loop<double>((double *) tmat_dev, (double *) h_dev, (double *) result_buffer_dev, (double *) tau_curr_dev, max_stored_rows, nstor, ice, SM_count, useCCL, debug, my_stream);
  else if (dataType=='S') gpu_trmv_loop<float> ((float  *) tmat_dev, (float  *) h_dev, (float  *) result_buffer_dev, (float  *) tau_curr_dev, max_stored_rows, nstor, ice, SM_count, useCCL, debug, my_stream);
  else if (dataType=='Z') gpu_trmv_loop<gpuDoubleComplex>((gpuDoubleComplex *) tmat_dev, (gpuDoubleComplex *) h_dev, (gpuDoubleComplex *) result_buffer_dev, (gpuDoubleComplex *) tau_curr_dev, max_stored_rows, nstor, ice, SM_count, useCCL, debug, my_stream);
  else if (dataType=='C') gpu_trmv_loop<gpuFloatComplex> ((gpuFloatComplex  *) tmat_dev, (gpuFloatComplex  *) h_dev, (gpuFloatComplex  *) result_buffer_dev, (gpuFloatComplex  *) tau_curr_dev, max_stored_rows, nstor, ice, SM_count, useCCL, debug, my_stream);
}
