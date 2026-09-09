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

#include "src/GPU/SYCL/syclCommon.hpp"
#include <sycl/sycl.hpp>
#include <math.h>
#include <stdlib.h>
#include <alloca.h>
#include <stdint.h>
#include <complex>

#include "config-f90.h"

#include "src/GPU/common_device_functions.h"
#include "src/GPU/gpu_to_cuda_and_hip_interface.h"

using namespace sycl_be;

extern "C" int syclDeviceSynchronizeFromC();

template <typename T>
void gpu_copy_tmp2_c_kernel(T *tmp2_dev, T *c_dev,
                            const int nr_done, const int nstor, const int lcs, const int lce, const int ldc, const int ldcCols,
                            const sycl::nd_item<1> &it){
  int jdex = it.get_group(0)    + 1; // range 1..lce-lse+1

  for (int idex = it.get_local_id(0) + 1; idex <= nstor; idex += it.get_local_range(0)) {
    //base 1 index
    c_dev[nr_done+(idex-1) + ldc*(lcs-1+jdex-1)] = tmp2_dev[0+(idex-1)+nstor*(jdex-1)];
  }
}

template <typename T>
void gpu_copy_tmp2_c (T *tmp2_dev, T *c_dev,
                      int nr_done, int nstor, int lcs, int lce, int ldc, int ldcCols, int debug, gpuStream_t my_stream){

  sycl::queue q = getQueueOrDefault(my_stream);
  sycl::range<1> blocks(lce-lcs+1);
  sycl::range<1> threadsPerBlock(sycl::min(nstor, MAX_THREADS_PER_BLOCK));

  q.parallel_for(sycl::nd_range<1>(blocks * threadsPerBlock, threadsPerBlock), [=](sycl::nd_item<1> it) {
      gpu_copy_tmp2_c_kernel(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, it);
  });

  if (debug) syclDeviceSynchronizeFromC();
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_tmp2_c_FromC)(char dataType, intptr_t tmp2_dev, intptr_t c_dev,
                                                         int nr_done, int nstor, int lcs, int lce, int ldc, int ldcCols, int debug, gpuStream_t my_stream) {
  if      (dataType=='D') gpu_copy_tmp2_c<double>((double *) tmp2_dev, (double *) c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, debug, my_stream);
  else if (dataType=='S') gpu_copy_tmp2_c<float> ((float  *) tmp2_dev, (float  *) c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, debug, my_stream);
  else if (dataType=='Z') gpu_copy_tmp2_c<gpuDoubleComplex>((gpuDoubleComplex *) tmp2_dev, (gpuDoubleComplex *) c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, debug, my_stream);
  else if (dataType=='C') gpu_copy_tmp2_c<gpuFloatComplex> ((gpuFloatComplex  *) tmp2_dev, (gpuFloatComplex  *) c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, debug, my_stream);
  else printf("Error in gpu_copy_tmp2_c: Unsupported data type\n");
}

//________________________________________________________________

template <typename T>
void gpu_copy_a_aux_bc_loop_kernel (T *a_dev, T *aux_bc_dev, int* lrs_save_dev, int* lre_save_dev, int* n_aux_bc_save_dev,
                                    const int noff, const int nblk, const int lda,
                                    const sycl::nd_item<1> &it){

  int n = it.get_group(0);

  int lrs = lrs_save_dev[n];
  int lre = lre_save_dev[n];
  int n_aux_bc = n_aux_bc_save_dev[n];

  for (int i=it.get_local_id(0); i<(lre-lrs+1); i += it.get_local_range(0))
    aux_bc_dev[(n_aux_bc+1-1)+i] = a_dev[(lrs-1)+i + lda*(noff*nblk+n+1-1)];
}

template <typename T>
void gpu_copy_a_aux_bc_loop(T *a_dev, T *aux_bc_dev, int *lrs_save_dev, int *lre_save_dev, int *n_aux_bc_save_dev,
                            int noff, int nblk, int lda, int n_size, int debug, gpuStream_t my_stream) {
		
  sycl::queue q = getQueueOrDefault(my_stream);
  sycl::range<1> blocks(n_size);
  sycl::range<1> threadsPerBlock(MIN_THREADS_PER_BLOCK);

  q.parallel_for(sycl::nd_range<1>(blocks * threadsPerBlock, threadsPerBlock), [=](sycl::nd_item<1> it) {
      gpu_copy_a_aux_bc_loop_kernel(a_dev, aux_bc_dev, lrs_save_dev, lre_save_dev, n_aux_bc_save_dev,
                                    noff, nblk, lda, it);
  });

  if (debug) syclDeviceSynchronizeFromC();
}

extern "C" void CONCATENATE(ELPA_GPU, _copy_a_aux_bc_loop_FromC) (char dataType, intptr_t a_dev, intptr_t aux_bc_dev, intptr_t lrs_save_dev, intptr_t lre_save_dev, intptr_t n_aux_bc_save_dev,
                                                                  int noff, int nblk, int lda, int n_size, int debug, gpuStream_t my_stream) {
  if      (dataType=='D') gpu_copy_a_aux_bc_loop<double>((double *) a_dev, (double *) aux_bc_dev, (int *) lrs_save_dev, (int *) lre_save_dev, (int *) n_aux_bc_save_dev,
                                                          noff, nblk, lda, n_size, debug, my_stream);
  else if (dataType=='S') gpu_copy_a_aux_bc_loop<float> ((float  *) a_dev, (float  *) aux_bc_dev, (int *) lrs_save_dev, (int *) lre_save_dev, (int *) n_aux_bc_save_dev,
                                                          noff, nblk, lda, n_size, debug, my_stream);
  else if (dataType=='Z') gpu_copy_a_aux_bc_loop<gpuDoubleComplex>((gpuDoubleComplex *) a_dev, (gpuDoubleComplex *) aux_bc_dev, (int *) lrs_save_dev, (int *) lre_save_dev, (int *) n_aux_bc_save_dev,
                                                          noff, nblk, lda, n_size, debug, my_stream);
  else if (dataType=='C') gpu_copy_a_aux_bc_loop<gpuFloatComplex> ((gpuFloatComplex  *) a_dev, (gpuFloatComplex  *) aux_bc_dev, (int *) lrs_save_dev, (int *) lre_save_dev, (int *) n_aux_bc_save_dev,
                                                          noff, nblk, lda, n_size, debug, my_stream);
  else printf("Error in gpu_copy_a_aux_bc: Unsupported data type\n");
}

//________________________________________________________________

template <typename T>
void gpu_copy_aux_bc_aux_mat_loop_kernel (const T* aux_bc_dev, T* aux_mat_dev, int* lrs_save_dev, int* lre_save_dev, int* n_aux_bc_save_dev,
                                          const int nstor0, const int l_rows,
                                          const sycl::nd_item<1> &it) {

  int n = it.get_group(0);
  int nstor = nstor0 + n;
  
  int lrs = lrs_save_dev[n];
  int lre = lre_save_dev[n];
  int n_aux_bc = n_aux_bc_save_dev[n];

  for (int i = it.get_local_id(0); i < (lre-lrs+1); i += it.get_local_range(0)) {
    aux_mat_dev[lrs-1+i + l_rows*(nstor-1)] = aux_bc_dev[n_aux_bc+i];
  }

}

template <typename T>
void gpu_copy_aux_bc_aux_mat_loop(T *aux_bc_dev, T *aux_mat_dev, int* lrs_save_dev, int *lre_save_dev, int *n_aux_bc_save_dev,
                                  int nstor0, int l_rows, 
                                  int n_size, int debug, gpuStream_t my_stream) {
  sycl::queue q = getQueueOrDefault(my_stream);
  sycl::range<1> blocks(n_size);
  sycl::range<1> threadsPerBlock(MIN_THREADS_PER_BLOCK);

  if (n_size<=0) return;

  q.parallel_for(sycl::nd_range<1>(blocks * threadsPerBlock, threadsPerBlock), [=](sycl::nd_item<1> it) {
      gpu_copy_aux_bc_aux_mat_loop_kernel(aux_bc_dev, aux_mat_dev, lrs_save_dev, lre_save_dev, n_aux_bc_save_dev,
                                          nstor0, l_rows, it);
  });

  if (debug) syclDeviceSynchronizeFromC();
}

extern "C" void CONCATENATE(ELPA_GPU,  _copy_aux_bc_aux_mat_loop_FromC)(char dataType, intptr_t aux_bc_dev, intptr_t aux_mat_dev, intptr_t lrs_save_dev, intptr_t lre_save_dev, intptr_t n_aux_bc_save_dev,
                                                                        int nstor0, int l_rows, int n_size, int debug, gpuStream_t my_stream) {
  if      (dataType=='D') gpu_copy_aux_bc_aux_mat_loop<double>((double *) aux_bc_dev, (double *) aux_mat_dev, (int *) lrs_save_dev, (int *) lre_save_dev, (int *) n_aux_bc_save_dev,
                                                                nstor0, l_rows, n_size, debug, my_stream);
  else if (dataType=='S') gpu_copy_aux_bc_aux_mat_loop<float> ((float  *) aux_bc_dev, (float  *) aux_mat_dev, (int *) lrs_save_dev, (int *) lre_save_dev, (int *) n_aux_bc_save_dev,
                                                                nstor0, l_rows, n_size, debug, my_stream);
  else if (dataType=='Z') gpu_copy_aux_bc_aux_mat_loop<gpuDoubleComplex>((gpuDoubleComplex *) aux_bc_dev, (gpuDoubleComplex *) aux_mat_dev, (int *) lrs_save_dev, (int *) lre_save_dev, (int *) n_aux_bc_save_dev,
                                                                nstor0, l_rows, n_size, debug, my_stream);
  else if (dataType=='C') gpu_copy_aux_bc_aux_mat_loop<gpuFloatComplex> ((gpuFloatComplex  *) aux_bc_dev, (gpuFloatComplex  *) aux_mat_dev, (int *) lrs_save_dev, (int *) lre_save_dev, (int *) n_aux_bc_save_dev,
                                                                nstor0, l_rows, n_size, debug, my_stream);
  else printf("Error in gpu_copy_aux_bc_aux_mat_loop: Unsupported data type\n");
}
