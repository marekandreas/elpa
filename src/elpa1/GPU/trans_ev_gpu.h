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

#ifdef WITH_NVIDIA_GPU_KERNEL
#include "trans_ev_gpu_cu.h"
#else

#include "../../GPU/SYCL/syclCommon.hpp"
#include <sycl/sycl.hpp>

// using gpuDoubleComplex = std::complex<double>;
// using gpuFloatComplex = std::complex<float>;
// using gpuStream_t = QueueData *;

extern "C" int syclDeviceSynchronizeFromC();

template <typename T>
void gpu_copy_hvb_a_kernel(T *hvb_dev, T *a_dev, int ld_hvb, int lda, int my_prow, int np_rows,
                                      int my_pcol, int np_cols, int nblk, int ics, int ice,
                                      const sycl::nd_item<1> &it) {
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

  int const i0 = it.get_local_id(); // 0..ld_hvb-1; max(l_rows) = ld_hvb
  int const ic_0 = it.get_group(0);

  T constexpr One = static_cast<T>(1.0);

  for (int ic = ic_0 + ics; ic <= ice; ic += it.get_group_range(0)) {
    int l_colh = local_index(ic  , my_pcol, np_cols, nblk, -1); // Column of Householder Vector
    int l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1); // Number of rows of Householder Vector

    // if (my_pcol == cur_pcol) // already true
    for (int i = i0; i < l_rows; i += it.get_local_range(0)) {
      hvb_dev[i + ld_hvb*(ic-ics)] = a_dev[i + (l_colh-1)*lda]; // nb -> ld_hvb*(ic-ics), no compression
    }

    if (my_prow == prow(ic - 1, nblk, np_rows) && i0 == 0) {
      hvb_dev[(l_rows-1) + ld_hvb*(ic-ics)] = One;
    }
  }
}



template <typename T>
void gpu_copy_hvb_a(T *hvb_dev, T *a_dev, int *ld_hvb_in, int *lda_in, int *my_prow_in, int *np_rows_in,
                    int *my_pcol_in, int *np_cols_in, int *nblk_in, int *ics_in, int *ice_in, int *SM_count_in, int *debug_in, QueueData *my_stream){
  int ld_hvb = *ld_hvb_in;
  int lda = *lda_in;
  int my_prow = *my_prow_in;
  int np_rows = *np_rows_in;
  int my_pcol = *my_pcol_in;
  int np_cols = *np_cols_in;
  int nblk = *nblk_in;
  int ics = *ics_in;
  int ice = *ice_in;
  int SM_count = *SM_count_in;
  int debug = *debug_in;

  sycl::queue q = getQueueOrDefault(my_stream);
  sycl::range<1> threadsPerBlock = maxWorkgroupSize<1>(q);
  sycl::range<1> blocks(SM_count);

  q.parallel_for(sycl::nd_range<1>(blocks * threadsPerBlock, threadsPerBlock), [=](sycl::nd_item<1> it) {
        gpu_copy_hvb_a_kernel(hvb_dev, a_dev, ld_hvb, lda, my_prow, np_rows,my_pcol, np_cols, nblk, ics, ice, it);
  });
  if (debug) syclDeviceSynchronizeFromC();
}

extern "C" void CONCATENATE(ELPA_GPU,  _copy_hvb_a_FromC) (char dataType, intptr_t hvb_dev, intptr_t a_dev,
                                      int *ld_hvb_in, int *lda_in, int *my_prow_in, int *np_rows_in,
                                      int *my_pcol_in, int *np_cols_in, int *nblk_in, int *ics_in, int *ice_in, 
                                      int *SM_count_in, int *debug_in, gpuStream_t my_stream){
  if      (dataType=='D') gpu_copy_hvb_a<double>((double *) hvb_dev, (double *) a_dev, ld_hvb_in, lda_in, my_prow_in, np_rows_in, my_pcol_in, np_cols_in, nblk_in, ics_in, ice_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='S') gpu_copy_hvb_a<float> ((float  *) hvb_dev, (float  *) a_dev, ld_hvb_in, lda_in, my_prow_in, np_rows_in, my_pcol_in, np_cols_in, nblk_in, ics_in, ice_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='Z') gpu_copy_hvb_a<gpuDoubleComplex>((gpuDoubleComplex *) hvb_dev, (gpuDoubleComplex *) a_dev, ld_hvb_in, lda_in, my_prow_in, np_rows_in, my_pcol_in, np_cols_in, nblk_in, ics_in, ice_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='C') gpu_copy_hvb_a<gpuFloatComplex> ((gpuFloatComplex  *) hvb_dev, (gpuFloatComplex  *) a_dev, ld_hvb_in, lda_in, my_prow_in, np_rows_in, my_pcol_in, np_cols_in, nblk_in, ics_in, ice_in, SM_count_in, debug_in, my_stream);
}

//_________________________________________________________________________________________________

template <typename T>
void gpu_copy_hvm_hvb_kernel(T *hvm_dev, T *hvb_dev, int ld_hvm, int ld_hvb, int my_prow, int np_rows,
                                        int nstor, int nblk, int ics, int ice,
                                        const sycl::nd_item<1> &it) {
  // nb = 0
  // NVTX_RANGE_PUSH("loop: copy hvm <- hvb")
  // do ic = ics, ice
  //   l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1) ! # rows of Householder Vector
  //   hvm(1:l_rows,nstor+1) = hvb(nb+1:nb+l_rows)
  //   if (useGPU) then
  //     hvm_ubnd = l_rows
  //   endif
  //   nstor = nstor+1
  //   nb = nb+l_rows
  // enddo

  int i0 = it.get_local_id(); // 0..ld_hvm-1; max(l_rows) = ld_hvm
  int ic_0 = it.get_group(0);

  T constexpr Zero = static_cast<T>(0.0);

  for (int ic = ic_0 + ics; ic <= ice; ic += it.get_group_range(0)) {
    int l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1);

    for (int i = i0; i < l_rows; i += it.get_local_range(0)) {
      hvm_dev[i + ld_hvm*(ic-ics+nstor)] = hvb_dev[i + ld_hvb*(ic-ics)]; // nb -> ld_hvb*(ic-ics), no compression
    }

    for (int i = l_rows + i0; i < ld_hvm; i += it.get_local_range(0)) {
      hvm_dev[i + ld_hvm*(ic-ics+nstor)] = Zero; // since we're not compressing, we need to take extra care to clear from previous iterations
    }
  }
}

template <typename T>
void gpu_copy_hvm_hvb(T *hvm_dev, T *hvb_dev, int *ld_hvm_in, int *ld_hvb_in, int *my_prow_in, int *np_rows_in,
                      int *nstor_in, int *nblk_in, int *ics_in, int *ice_in, int *SM_count_in, int *debug_in, gpuStream_t my_stream){
  int ld_hvm = *ld_hvm_in;
  int ld_hvb = *ld_hvb_in;
  int my_prow = *my_prow_in;
  int np_rows = *np_rows_in;
  int nstor = *nstor_in;
  int nblk = *nblk_in;
  int ics = *ics_in;
  int ice = *ice_in;
  int SM_count = *SM_count_in;
  int debug = *debug_in;

  sycl::queue q = getQueueOrDefault(my_stream);
  sycl::range<1> threadsPerBlock = maxWorkgroupSize<1>(q);
  sycl::range<1> blocks(SM_count);

  q.parallel_for(sycl::nd_range<1>(blocks * threadsPerBlock, threadsPerBlock), [=](sycl::nd_item<1> it) {
    gpu_copy_hvm_hvb_kernel(hvm_dev, hvb_dev, ld_hvm, ld_hvb, my_prow, np_rows, nstor, nblk, ics, ice, it);
  });
  
  if (debug) {
    q.wait_and_throw();
    syclDeviceSynchronizeFromC();
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _copy_hvm_hvb_FromC) (char dataType, intptr_t hvm_dev, intptr_t hvb_dev,
                                      int *ld_hvm_in, int *ld_hvb_in, int *my_prow_in, int *np_rows_in,
                                      int *nstor_in, int *nblk_in, int *ics_in, int *ice_in, 
                                      int *SM_count_in, int *debug_in, gpuStream_t my_stream){
  if      (dataType=='D') gpu_copy_hvm_hvb<double>((double *) hvm_dev, (double *) hvb_dev, ld_hvm_in, ld_hvb_in, my_prow_in, np_rows_in, nstor_in, nblk_in, ics_in, ice_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='S') gpu_copy_hvm_hvb<float> ((float  *) hvm_dev, (float  *) hvb_dev, ld_hvm_in, ld_hvb_in, my_prow_in, np_rows_in, nstor_in, nblk_in, ics_in, ice_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='Z') gpu_copy_hvm_hvb<gpuDoubleComplex>((gpuDoubleComplex *) hvm_dev, (gpuDoubleComplex *) hvb_dev, ld_hvm_in, ld_hvb_in, my_prow_in, np_rows_in, nstor_in, nblk_in, ics_in, ice_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='C') gpu_copy_hvm_hvb<gpuFloatComplex> ((gpuFloatComplex  *) hvm_dev, (gpuFloatComplex  *) hvb_dev, ld_hvm_in, ld_hvb_in, my_prow_in, np_rows_in, nstor_in, nblk_in, ics_in, ice_in, SM_count_in, debug_in, my_stream);
}

//_________________________________________________________________________________________________


// PETERDEBUG: Is this okay?
template <typename T>
void gpu_update_tmat_kernel(T *tmat_dev, T *h_dev, T *tau_curr_dev, int max_stored_rows, int nc, int n, const sycl::nd_item<1> &it) {
  int j0 = it.get_group(0);
  if (j0 == 0) {
    tmat_dev[n + n*max_stored_rows] = *tau_curr_dev;
  }
}

template <typename T>
void gpu_update_tmat(T *tmat_dev, T *h_dev, T *tau_curr_dev, int *max_stored_rows_in, int *nc_in, int *n_in, int *SM_count_in, int *debug_in, gpuStream_t my_stream){
  int max_stored_rows = *max_stored_rows_in;
  int nc = *nc_in;
  int n = *n_in;
  int SM_count = *SM_count_in;
  int debug = *debug_in;
  
  sycl::queue q = getQueueOrDefault(my_stream);
  sycl::range<1> threadsPerBlock(1);
  sycl::range<1> blocks(SM_count);

  q.parallel_for(sycl::nd_range<1>(blocks * threadsPerBlock, threadsPerBlock), [=](sycl::nd_item<1> it) {
    gpu_update_tmat_kernel(tmat_dev, h_dev, tau_curr_dev, max_stored_rows, nc, n, it);
  });

  if (debug) {
    q.wait_and_throw();
    syclDeviceSynchronizeFromC();
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _update_tmat_FromC) (char dataType, intptr_t tmat_dev, intptr_t h_dev, intptr_t tau_curr_dev,
                                      int *max_stored_rows_in, int *nc_in, int *n_in, int *SM_count_in, int *debug_in, gpuStream_t my_stream){
  if      (dataType=='D') gpu_update_tmat<double>((double *) tmat_dev, (double *) h_dev, (double *) tau_curr_dev, max_stored_rows_in, nc_in, n_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='S') gpu_update_tmat<float> ((float  *) tmat_dev, (float  *) h_dev, (float  *) tau_curr_dev, max_stored_rows_in, nc_in, n_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='Z') gpu_update_tmat<gpuDoubleComplex>((gpuDoubleComplex *) tmat_dev, (gpuDoubleComplex *) h_dev, (gpuDoubleComplex *) tau_curr_dev, max_stored_rows_in, nc_in, n_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='C') gpu_update_tmat<gpuFloatComplex> ((gpuFloatComplex  *) tmat_dev, (gpuFloatComplex  *) h_dev, (gpuFloatComplex  *) tau_curr_dev, max_stored_rows_in, nc_in, n_in, SM_count_in, debug_in, my_stream);
}

//_________________________________________________________________________________________________
// PETERDEBUG: delete result_buffer_dev

#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER < 20240000
  #define GET_POINTER(x) x.get_pointer()
#else
  #define GET_POINTER(x) x.template get_multi_ptr<sycl::access::decorated::no>().get()
#endif


template <typename T>
void gpu_trmv_kernel(T *tmat_dev, T *h_dev, T *result_buffer_dev, T *tau_curr_dev, int max_stored_rows, int n, const sycl::nd_item<1> &it, T *cache) {
  //__shared__ T cache[MIN_THREADS_PER_BLOCK*MIN_THREADS_PER_BLOCK]; // extra space of fixed size is reserved for a speedup

  // h_dev <- tmat_dev^T*h_dev
  // uncoalesced memory access in gpu_update_tmat_kernel, coalesced access here.
  // tmat_dev is lower triangular

  int i0 = it.get_local_id();
  int j0 = it.get_group(0);

  T constexpr Zero = static_cast<T>(0.0);

  for (int j = j0; j < n; j += it.get_group_range(0)) {
    //if (threadIdx.x==0) result_buffer_dev[j] = Zero;
    if (it.get_local_id() == 0) {
      tmat_dev[n + j * max_stored_rows] = Zero; // PETERDEBUG: delete, unneeded
    } 

    T slice_sum = Zero;
    // j because tmat_dev is lower triangular
    for (int i = i0 + j; i < n; i += it.get_local_range(0)) { 
      slice_sum = elpaDeviceAdd(slice_sum, elpaDeviceMultiply(elpaDeviceComplexConjugate(tmat_dev[i + j*max_stored_rows]), h_dev[i]));
    }
    cache[i0] = slice_sum;
    
    it.barrier();

    // for reductions, threadsPerBlock=blockDim.x must be a power of 2
    int di = it.get_local_range(0) / 2;
    int const lId = it.get_local_id();
    while (di > 0) {
      if (lId < di) {
        cache[lId] = cache[lId] + cache[lId + di];
      }
      it.barrier();
      di /= 2;
    }

    if (it.get_local_id() == 0) {
      //atomicAdd(&tmat_dev[n + j*max_stored_rows], cache[0]);
      int idx = n + j*max_stored_rows;
      tmat_dev[idx] = cache[0];
      tmat_dev[idx] = elpaDeviceMultiply(elpaDeviceComplexConjugate(tmat_dev[idx]), (*tau_curr_dev));
      tmat_dev[idx] = elpaDeviceMultiply(static_cast<T>(-1.0), tmat_dev[idx]);
    }
  }

  if (i0 == 0 && j0 == 0) {
    tmat_dev[n + n * max_stored_rows] = *tau_curr_dev;
  }
}

template <typename T>
void gpu_trmv(T *tmat_dev, T *h_dev, T *result_buffer_dev, T *tau_curr_dev, int *max_stored_rows_in, int *n_in, int *SM_count_in, int *debug_in, gpuStream_t my_stream){
  int max_stored_rows = *max_stored_rows_in;
  int n = *n_in;
  int SM_count = *SM_count_in;
  int debug = *debug_in;

  sycl::queue q = getQueueOrDefault(my_stream);
  sycl::range<1> threadsPerBlock = maxWorkgroupSize<1>(q);
  sycl::range<1> blocks(SM_count);

  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<T, 1> cache_acc_ct1(threadsPerBlock, h);
    h.parallel_for(sycl::nd_range<1>(blocks * threadsPerBlock, threadsPerBlock), [=](sycl::nd_item<1> it) {
      gpu_trmv_kernel(tmat_dev, h_dev, result_buffer_dev, tau_curr_dev, max_stored_rows, n, it, GET_POINTER(cache_acc_ct1));
    });
  });

  
  // PETERDEBUG: cleanup. too much overhead, if called in tight loop?
  if (debug) {
    q.wait_and_throw();
    syclDeviceSynchronizeFromC();
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _trmv_FromC) (char dataType, intptr_t tmat_dev, intptr_t h_dev, intptr_t result_buffer_dev, intptr_t tau_curr_dev,
                                      int *max_stored_rows_in, int *n_in, int *SM_count_in, int *debug_in, gpuStream_t my_stream){
  if      (dataType=='D') gpu_trmv<double>((double *) tmat_dev, (double *) h_dev, (double *) result_buffer_dev, (double *) tau_curr_dev, max_stored_rows_in, n_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='S') gpu_trmv<float> ((float  *) tmat_dev, (float  *) h_dev, (float  *) result_buffer_dev, (float  *) tau_curr_dev, max_stored_rows_in, n_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='Z') gpu_trmv<gpuDoubleComplex>((gpuDoubleComplex *) tmat_dev, (gpuDoubleComplex *) h_dev, (gpuDoubleComplex *) result_buffer_dev, (gpuDoubleComplex *) tau_curr_dev, max_stored_rows_in, n_in, SM_count_in, debug_in, my_stream);
  else if (dataType=='C') gpu_trmv<gpuFloatComplex> ((gpuFloatComplex  *) tmat_dev, (gpuFloatComplex  *) h_dev, (gpuFloatComplex  *) result_buffer_dev, (gpuFloatComplex  *) tau_curr_dev, max_stored_rows_in, n_in, SM_count_in, debug_in, my_stream);
}

template <typename T>
void gpu_trmv_loop(T *tmat_dev, T *h_dev, T *result_buffer_dev, T *tau_curr_dev, int *max_stored_rows_in, int *nstor_in, int *ice_in, int *SM_count_in, int *useCCL_in, int *debug_in, gpuStream_t my_stream){
  int max_stored_rows = *max_stored_rows_in;
  int nstor = *nstor_in;
  int ice = *ice_in;
  int SM_count = *SM_count_in;
  int useCCL = *useCCL_in;
  int debug = *debug_in;

  sycl::queue q = getQueueOrDefault(my_stream);
  sycl::range<1> threadsPerBlock = maxWorkgroupSize<1>(q);
  sycl::range<1> blocks(SM_count);

  int size_of_datatype = sizeof(T);
  int shift_h_dev, shift_tau_dev;
  int nc = 0;

  for (int n=1; n <= nstor-1; n++) {
    if (useCCL) {
      shift_h_dev = n*max_stored_rows;
    } else {
      shift_h_dev = nc;
      nc = nc+n;
    }
    shift_tau_dev = (ice-nstor+n);

    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<T, 1> cache_acc_ct1(threadsPerBlock, h);

      auto h_dev_shift_h_dev_ct1 = h_dev + shift_h_dev;
      auto tau_curr_dev_shift_tau_dev_ct3 = tau_curr_dev + shift_tau_dev;
      h.parallel_for(sycl::nd_range<1>(blocks * threadsPerBlock, threadsPerBlock),[=](sycl::nd_item<1> it) {
        gpu_trmv_kernel(tmat_dev, h_dev_shift_h_dev_ct1, result_buffer_dev, tau_curr_dev_shift_tau_dev_ct3,
                        max_stored_rows, nstor, it, GET_POINTER(cache_acc_ct1));
      });
    });
  }

  if (debug) {
    q.wait_and_throw();
    syclDeviceSynchronizeFromC();
  }
}

extern "C" void CONCATENATE(ELPA_GPU,  _trmv_loop_FromC) (char dataType, intptr_t tmat_dev, intptr_t h_dev, intptr_t result_buffer_dev, intptr_t tau_curr_dev,
                                      int *max_stored_rows_in, int *nstor_in, int *ice_in, int *SM_count_in, int *useCCL_in, int *debug_in, gpuStream_t my_stream){
  if      (dataType=='D') gpu_trmv_loop<double>((double *) tmat_dev, (double *) h_dev, (double *) result_buffer_dev, (double *) tau_curr_dev, max_stored_rows_in, nstor_in, ice_in, SM_count_in, useCCL_in, debug_in, my_stream);
  else if (dataType=='S') gpu_trmv_loop<float> ((float  *) tmat_dev, (float  *) h_dev, (float  *) result_buffer_dev, (float  *) tau_curr_dev, max_stored_rows_in, nstor_in, ice_in, SM_count_in, useCCL_in, debug_in, my_stream);
  else if (dataType=='Z') gpu_trmv_loop<gpuDoubleComplex>((gpuDoubleComplex *) tmat_dev, (gpuDoubleComplex *) h_dev, (gpuDoubleComplex *) result_buffer_dev, (gpuDoubleComplex *) tau_curr_dev, max_stored_rows_in, nstor_in, ice_in, SM_count_in, useCCL_in, debug_in, my_stream);
  else if (dataType=='C') gpu_trmv_loop<gpuFloatComplex> ((gpuFloatComplex  *) tmat_dev, (gpuFloatComplex  *) h_dev, (gpuFloatComplex  *) result_buffer_dev, (gpuFloatComplex  *) tau_curr_dev, max_stored_rows_in, nstor_in, ice_in, SM_count_in, useCCL_in, debug_in, my_stream);
}

#endif // WITH_NVIDIA_GPU_KERNEL