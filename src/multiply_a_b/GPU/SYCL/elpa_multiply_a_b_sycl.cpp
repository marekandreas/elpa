//    Copyright 2023, A. Marek
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

// SKELETON VERSION WITHOUT CONTENT, CURRENTLY NOT CALLED

#include <CL/sycl.hpp>

#include <complex>
#include <oneapi/mkl.hpp>

#include <iostream>
#include <cstdint>
#include <vector>
#include <optional>

#include "config-f90.h"


//__global__ void hip_copy_double_tmp2_c_kernel(double *tmp2_dev, double *c_dev, const int nr_done, const int nstor, const int lcs, const int lce, const int ldc, const int ldcCols){
//
//  dim3 blocks = dim3(lce-lcs+1,1,1);
//  dim3 threadsPerBlock = dim3(nr_done+nstor-(nr_done+1)+1,1,1);
//
//  int i_index    = threadIdx.x +1; // range 1..nstor
//  int j_index = blockIdx.x + 1; // range 1..lce-lse+1
//  c_dev[nr_done+(i_index-1) + ldc*(lcs-1+j_index-1)] = tmp2_dev[0+(i_index-1)+nstor*(lcs-1+j_index-1)];
//}

extern "C" void sycl_copy_double_tmp2_c_FromC(double *tmp2_dev, double *c_dev, int *nr_done_in, int *nstor_in, int *lcs_in, int *lce_in, int *ldc_in, int *ldcCols_in) { 
		
  int nr_done = *nr_done_in;   
  int nstor = *nstor_in;
  int lcs = *lcs_in;
  int lce = *lce_in;
  int ldc = *ldc_in;
  int ldcCols = *ldcCols_in;

   std::cout << "This function sycl_copy_double_tmp2_c_FromC should never be called!!" << std::endl;
}

//__global__ void hip_copy_float_tmp2_c_kernel(float *tmp2_dev, float *c_dev, const int nr_done, const int nstor, const int lcs, const int lce, const int ldc, const int ldcCols){
//
//  dim3 blocks = dim3(lce-lcs+1,1,1);
//  dim3 threadsPerBlock = dim3(nr_done+nstor-(nr_done+1)+1,1,1);
//
//  int i_index    = threadIdx.x +1; // range 1..nstor
//  int j_index = blockIdx.x + 1; // range 1..lce-lse+1
//  c_dev[nr_done+(i_index-1) + ldc*(lcs-1+j_index-1)] = tmp2_dev[0+(i_index-1)+nstor*(lcs-1+j_index-1)];
//}
//
extern "C" void sycl_copy_float_tmp2_c_FromC(float *tmp2_dev, float *c_dev, int *nr_done_in, int *nstor_in, int *lcs_in, int *lce_in, int *ldc_in, int *ldcCols_in) { 
		
  int nr_done = *nr_done_in;   
  int nstor = *nstor_in;
  int lcs = *lcs_in;
  int lce = *lce_in;
  int ldc = *ldc_in;
  int ldcCols = *ldcCols_in;

   std::cout << "This function sycl_copy_float_tmp2_c_FromC should never be called!!" << std::endl;
}

//__global__ void hip_copy_double_complex_tmp2_c_kernel(hipDoubleComplex *tmp2_dev, hipDoubleComplex *c_dev, const int nr_done, const int nstor, const int lcs, const int lce, const int ldc, const int ldcCols){
//
//  dim3 blocks = dim3(lce-lcs+1,1,1);
//  dim3 threadsPerBlock = dim3(nr_done+nstor-(nr_done+1)+1,1,1);
//
//  int i_index    = threadIdx.x +1; // range 1..nstor
//  int j_index = blockIdx.x + 1; // range 1..lce-lse+1
//  c_dev[nr_done+(i_index-1) + ldc*(lcs-1+j_index-1)] = tmp2_dev[0+(i_index-1)+nstor*(lcs-1+j_index-1)];
//}

extern "C" void sycl_copy_double_complex_tmp2_c_FromC(double _Complex *tmp2_dev, double _Complex *c_dev, int *nr_done_in, int *nstor_in, int *lcs_in, int *lce_in, int *ldc_in, int *ldcCols_in) { 
		
  int nr_done = *nr_done_in;   
  int nstor = *nstor_in;
  int lcs = *lcs_in;
  int lce = *lce_in;
  int ldc = *ldc_in;
  int ldcCols = *ldcCols_in;

   std::cout << "This function sycl_copy_double_complex_tmp2_c_FromC should never be called!!" << std::endl;
}

//__global__ void hip_copy_float_complex_tmp2_c_kernel(hipFloatComplex *tmp2_dev, hipFloatComplex *c_dev, const int nr_done, const int nstor, const int lcs, const int lce, const int ldc, const int ldcCols){
//
//  dim3 blocks = dim3(lce-lcs+1,1,1);
//  dim3 threadsPerBlock = dim3(nr_done+nstor-(nr_done+1)+1,1,1);
//
//  int i_index    = threadIdx.x +1; // range 1..nstor
//  int j_index = blockIdx.x + 1; // range 1..lce-lse+1
//  c_dev[nr_done+(i_index-1) + ldc*(lcs-1+j_index-1)] = tmp2_dev[0+(i_index-1)+nstor*(lcs-1+j_index-1)];
//}

extern "C" void sycl_copy_float_complex_tmp2_c_FromC(float _Complex *tmp2_dev, float _Complex *c_dev, int *nr_done_in, int *nstor_in, int *lcs_in, int *lce_in, int *ldc_in, int *ldcCols_in) { 
		
  int nr_done = *nr_done_in;   
  int nstor = *nstor_in;
  int lcs = *lcs_in;
  int lce = *lce_in;
  int ldc = *ldc_in;
  int ldcCols = *ldcCols_in;

   std::cout << "This function sycl_copy_float_complex_tmp2_c_FromC should never be called!!" << std::endl;
}



//__global__ void hip_copy_double_a_aux_bc_kernel(double *a_dev, double *aux_bc_dev, const int n_aux_bc, const int nvals, const int lrs, const int lre, const int noff, const int nblk, const int n, const int l_rows, const int lda, const int ldaCols){
//
//  dim3 blocks = dim3(lre-lrs+1,1,1);
//  dim3 threadsPerBlock = dim3(1,1,1);
//
//  int i_index    = blockIdx.x +1; // range 1..lre-lrs+1
//  int j_index = threadIdx.x + 1; // range 1..1
//  aux_bc_dev[(n_aux_bc+1-1)+(i_index-1)] = a_dev[(lrs-1)+(i_index-1)+lda*(noff*nblk+n-1)];
//}

extern "C" void sycl_copy_double_a_aux_bc_FromC(double *a_dev, double *aux_bc_dev, int *n_aux_bc_in, int *nvals_in, int *lrs_in, int *lre_in, int *noff_in, int *nblk_in, int *n_in, int *l_rows_in, int *lda_in, int *ldaCols_in) { 
		
  int n_aux_bc = *n_aux_bc_in;   
  int nvals = *nvals_in;
  int lrs = *lrs_in;
  int lre = *lre_in;
  int noff = *noff_in;
  int nblk = *nblk_in;
  int n = *n_in;
  int l_rows = *l_rows_in;
  int lda = *lda_in;
  int ldaCols = *ldaCols_in;

   std::cout << "This function  sycl_copy_double_a_aux_bc_FromC should never be called!!" << std::endl;
}



//__global__ void hip_copy_float_a_aux_bc_kernel(float *a_dev, float *aux_bc_dev, const int n_aux_bc, const int nvals, const int lrs, const int lre, const int noff, const int nblk, const int n, const int l_rows, const int lda, const int ldaCols){
//
//  dim3 blocks = dim3(lre-lrs+1,1,1);
//  dim3 threadsPerBlock = dim3(1,1,1);
//
//  int i_index    = blockIdx.x +1; // range 1..lre-lrs+1
//  int j_index = threadIdx.x + 1; // range 1..1
//  aux_bc_dev[(n_aux_bc+1-1)+(i_index-1)] = a_dev[(lrs-1)+(i_index-1)+lda*(noff*nblk+n-1)];
//}

extern "C" void sycl_copy_float_a_aux_bc_FromC(float *a_dev, float *aux_bc_dev, int *n_aux_bc_in, int *nvals_in, int *lrs_in, int *lre_in, int *noff_in, int *nblk_in, int *n_in, int *l_rows_in, int *lda_in, int *ldaCols_in) { 
		
  int n_aux_bc = *n_aux_bc_in;   
  int nvals = *nvals_in;
  int lrs = *lrs_in;
  int lre = *lre_in;
  int noff = *noff_in;
  int nblk = *nblk_in;
  int n = *n_in;
  int l_rows = *l_rows_in;
  int lda = *lda_in;
  int ldaCols = *ldaCols_in;

   std::cout << "This function  sycl_copy_float_a_aux_bc_FromC should never be called!!" << std::endl;
}


//__global__ void hip_copy_double_complex_a_aux_bc_kernel(hipDoubleComplex *a_dev, hipDoubleComplex *aux_bc_dev, const int n_aux_bc, const int nvals, const int lrs, const int lre, const int noff, const int nblk, const int n, const int l_rows, const int lda, const int ldaCols){
//
//  dim3 blocks = dim3(lre-lrs+1,1,1);
//  dim3 threadsPerBlock = dim3(1,1,1);
//
//  int i_index    = blockIdx.x +1; // range 1..lre-lrs+1
//  int j_index = threadIdx.x + 1; // range 1..1
//  aux_bc_dev[(n_aux_bc+1-1)+(i_index-1)] = a_dev[(lrs-1)+(i_index-1)+lda*(noff*nblk+n-1)];
//}

extern "C" void sycl_copy_double_complex_a_aux_bc_FromC(double _Complex *a_dev, double _Complex *aux_bc_dev, int *n_aux_bc_in, int *nvals_in, int *lrs_in, int *lre_in, int *noff_in, int *nblk_in, int *n_in, int *l_rows_in, int *lda_in, int *ldaCols_in) { 
		
  int n_aux_bc = *n_aux_bc_in;   
  int nvals = *nvals_in;
  int lrs = *lrs_in;
  int lre = *lre_in;
  int noff = *noff_in;
  int nblk = *nblk_in;
  int n = *n_in;
  int l_rows = *l_rows_in;
  int lda = *lda_in;
  int ldaCols = *ldaCols_in;

   std::cout << "This function  sycl_copy_double_complex_a_aux_bc_FromC should never be called!!" << std::endl;
}


//__global__ void hip_copy_float_complex_a_aux_bc_kernel(hipFloatComplex *a_dev, hipFloatComplex *aux_bc_dev, const int n_aux_bc, const int nvals, const int lrs, const int lre, const int noff, const int nblk, const int n, const int l_rows, const int lda, const int ldaCols){
//
//  dim3 blocks = dim3(lre-lrs+1,1,1);
//  dim3 threadsPerBlock = dim3(1,1,1);
//
//  int i_index    = blockIdx.x +1; // range 1..lre-lrs+1
//  int j_index = threadIdx.x + 1; // range 1..1
//  aux_bc_dev[(n_aux_bc+1-1)+(i_index-1)] = a_dev[(lrs-1)+(i_index-1)+lda*(noff*nblk+n-1)];
//}

extern "C" void sycl_copy_float_complex_a_aux_bc_FromC(float _Complex *a_dev, float _Complex *aux_bc_dev, int *n_aux_bc_in, int *nvals_in, int *lrs_in, int *lre_in, int *noff_in, int *nblk_in, int *n_in, int *l_rows_in, int *lda_in, int *ldaCols_in) { 
		
  int n_aux_bc = *n_aux_bc_in;   
  int nvals = *nvals_in;
  int lrs = *lrs_in;
  int lre = *lre_in;
  int noff = *noff_in;
  int nblk = *nblk_in;
  int n = *n_in;
  int l_rows = *l_rows_in;
  int lda = *lda_in;
  int ldaCols = *ldaCols_in;

   std::cout << "This function  sycl_copy_float_complex_a_aux_bc_FromC should never be called!!" << std::endl;
}


//__global__ void hip_copy_double_aux_bc_aux_mat_kernel(double *aux_bc_dev, double *aux_mat_dev, const int lrs, const int lre, const int nstor, const int n_aux_bc, const int nvals, const int l_rows, const int nblk_mult, const int nblk) {
//		
//  dim3 threadsPerBlock = dim3(1,1,1);
//  dim3 blocks = dim3(lre-lrs+1,1,1);
//  //dim3 blocks = dim3(1,1,1);
//
//  int i_index    = threadIdx.x +1; // range 1..lre-lrs+1
//  int j_index = blockIdx.x + 1; // range 1..1
//  aux_mat_dev[(j_index-1)+l_rows*(nstor-1)] = aux_bc_dev[n_aux_bc+(j_index-1)];
//}
//

extern "C" void sycl_copy_double_aux_bc_aux_mat_FromC(double *aux_bc_dev, double *aux_mat_dev, int *lrs_in, int *lre_in, int *nstor_in, int *n_aux_bc_in, int *nvals_in, int *l_rows_in, int *nblk_in, int *nblk_mult_in) {
		


  int lrs = *lrs_in;
  int lre = *lre_in;
  int nstor = *nstor_in;
  int n_aux_bc = *n_aux_bc_in;   
  int nvals = *nvals_in;
  int l_rows = *l_rows_in;
  int nblk_mult = *nblk_mult_in;
  int nblk = *nblk_in;
  
   std::cout << "This function  sycl_copy_double_aux_bc_aux_mat_FromC should never be called!!" << std::endl;
}

//__global__ void hip_copy_float_aux_bc_aux_mat_kernel(float *aux_bc_dev, float *aux_mat_dev, const int lrs, const int lre, const int nstor, const int n_aux_bc, const int nvals, const int l_rows, const int nblk, const int nblk_mult) {
//		
//  dim3 blocks = dim3(lre-lrs+1,1,1);
//  dim3 threadsPerBlock = dim3(1,1,1);
//
//  int i_index    = threadIdx.x +1; // range 1..lre-lrs+1
//  int j_index = blockIdx.x + 1; // range 1..1
//  aux_mat_dev[(j_index-1)+l_rows*(nstor-1)] = aux_bc_dev[n_aux_bc+(j_index-1)];
//}

extern "C" void sycl_copy_float_aux_bc_aux_mat_FromC(float *aux_bc_dev, float *aux_mat_dev, int *lrs_in, int *lre_in, int *nstor_in, int *n_aux_bc_in, int *nvals_in, int *l_rows_in, int *nblk_in, int *nblk_mult_in) {
		


  int lrs = *lrs_in;
  int lre = *lre_in;
  int nstor = *nstor_in;
  int n_aux_bc = *n_aux_bc_in;   
  int nvals = *nvals_in;
  int l_rows = *l_rows_in;
  int nblk_mult = *nblk_mult_in;
  int nblk = *nblk_in;
  
   std::cout << "This function  sycl_copy_float_aux_bc_aux_mat_FromC should never be called!!" << std::endl;
}

//__global__ void hip_copy_double_complex_aux_bc_aux_mat_kernel(hipDoubleComplex *aux_bc_dev, hipDoubleComplex *aux_mat_dev, const int lrs, const int lre, const int nstor, const int n_aux_bc, const int nvals, const int l_rows, const int nblk, const int nblk_mult) {
//		
//  dim3 blocks = dim3(lre-lrs+1,1,1);
//  dim3 threadsPerBlock = dim3(1,1,1);
//
//  int i_index    = threadIdx.x +1; // range 1..lre-lrs+1
//  int j_index = blockIdx.x + 1; // range 1..1
//  aux_mat_dev[(j_index-1)+l_rows*(nstor-1)] = aux_bc_dev[n_aux_bc+(j_index-1)];
//}

extern "C" void sycl_copy_double_complex_aux_bc_aux_mat_FromC(double _Complex *aux_bc_dev, double _Complex *aux_mat_dev, int *lrs_in, int *lre_in, int *nstor_in, int *n_aux_bc_in, int *nvals_in, int *l_rows_in, int *nblk_in, int *nblk_mult_in) {
		


  int lrs = *lrs_in;
  int lre = *lre_in;
  int nstor = *nstor_in;
  int n_aux_bc = *n_aux_bc_in;   
  int nvals = *nvals_in;
  int l_rows = *l_rows_in;
  int nblk_mult = *nblk_mult_in;
  int nblk = *nblk_in;
  
   std::cout << "This function  sycl_copy_double_complex_aux_bc_aux_mat_FromC should never be called!!" << std::endl;
}


//__global__ void hip_copy_float_complex_aux_bc_aux_mat_kernel(hipFloatComplex *aux_bc_dev, hipFloatComplex *aux_mat_dev, const int lrs, const int lre, const int nstor, const int n_aux_bc, const int nvals, const int l_rows, const int nblk, const int nblk_mult) {
//		
//  dim3 blocks = dim3(lre-lrs+1,1,1);
//  dim3 threadsPerBlock = dim3(1,1,1);
//
//  int i_index    = threadIdx.x +1; // range 1..lre-lrs+1
//  int j_index = blockIdx.x + 1; // range 1..1
//  aux_mat_dev[(j_index-1)+l_rows*(nstor-1)] = aux_bc_dev[n_aux_bc+(j_index-1)];
//}

extern "C" void sycl_copy_float_complex_aux_bc_aux_mat_FromC(float _Complex *aux_bc_dev, float _Complex *aux_mat_dev, int *lrs_in, int *lre_in, int *nstor_in, int *n_aux_bc_in, int *nvals_in, int *l_rows_in, int *nblk_in, int *nblk_mult_in) {
		


  int lrs = *lrs_in;
  int lre = *lre_in;
  int nstor = *nstor_in;
  int n_aux_bc = *n_aux_bc_in;   
  int nvals = *nvals_in;
  int l_rows = *l_rows_in;
  int nblk_mult = *nblk_mult_in;
  int nblk = *nblk_in;
  
   std::cout << "This function  sycl_copy_float_complex_aux_bc_aux_mat_FromC should never be called!!" << std::endl;
}





