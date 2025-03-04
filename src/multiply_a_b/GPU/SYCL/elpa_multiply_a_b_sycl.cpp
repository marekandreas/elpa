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


extern "C" void sycl_copy_double_tmp2_c_FromC(double *tmp2_dev, double *c_dev, int *nr_done_in, int *nstor_in, int *lcs_in, int *lce_in, int *ldc_in, int *ldcCols_in) { 
		
  int nr_done = *nr_done_in;   
  int nstor = *nstor_in;
  int lcs = *lcs_in;
  int lce = *lce_in;
  int ldc = *ldc_in;
  int ldcCols = *ldcCols_in;

   std::cout << "This function sycl_copy_double_tmp2_c_FromC should never be called!!" << std::endl;
}

extern "C" void sycl_copy_float_tmp2_c_FromC(float *tmp2_dev, float *c_dev, int *nr_done_in, int *nstor_in, int *lcs_in, int *lce_in, int *ldc_in, int *ldcCols_in) { 
		
  int nr_done = *nr_done_in;   
  int nstor = *nstor_in;
  int lcs = *lcs_in;
  int lce = *lce_in;
  int ldc = *ldc_in;
  int ldcCols = *ldcCols_in;

   std::cout << "This function sycl_copy_float_tmp2_c_FromC should never be called!!" << std::endl;
}

extern "C" void sycl_copy_double_complex_tmp2_c_FromC(double _Complex *tmp2_dev, double _Complex *c_dev, int *nr_done_in, int *nstor_in, int *lcs_in, int *lce_in, int *ldc_in, int *ldcCols_in) { 
		
  int nr_done = *nr_done_in;   
  int nstor = *nstor_in;
  int lcs = *lcs_in;
  int lce = *lce_in;
  int ldc = *ldc_in;
  int ldcCols = *ldcCols_in;

   std::cout << "This function sycl_copy_double_complex_tmp2_c_FromC should never be called!!" << std::endl;
}

extern "C" void sycl_copy_float_complex_tmp2_c_FromC(float _Complex *tmp2_dev, float _Complex *c_dev, int *nr_done_in, int *nstor_in, int *lcs_in, int *lce_in, int *ldc_in, int *ldcCols_in) { 
		
  int nr_done = *nr_done_in;   
  int nstor = *nstor_in;
  int lcs = *lcs_in;
  int lce = *lce_in;
  int ldc = *ldc_in;
  int ldcCols = *ldcCols_in;

   std::cout << "This function sycl_copy_float_complex_tmp2_c_FromC should never be called!!" << std::endl;
}


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

using gpuStream_t = void*;
////////////////////////////////////////////////////////


extern "C" void sycl_copy_aux_full_FromC (char dataType, intptr_t lhs_dev, intptr_t rhs_dev,
                                         int *l_rows_in, int *l_cols_in, int *lld_lhs_in, int *lld_rhs_in, int *debug_in, gpuStream_t my_stream){
  throw std::runtime_error("This function should never be called!!");
}

//________________________________________________________________



extern "C" void sycl_copy_and_set_zeros_aux_full_FromC (char dataType, intptr_t mat_dev, intptr_t aux_mat_full_dev,
                                                       int *l_rows_in, int *l_cols_in, int *nblk_mult_in, int *debug_in, gpuStream_t my_stream){
                                                        throw std::runtime_error("This function should never be called!!");

}

//________________________________________________________________

extern "C" void sycl_copy_and_set_zeros_aux_a_full_FromC (char dataType, intptr_t mat_dev, intptr_t aux_mat_full_dev,
                                                       int *l_rows_in, int *l_cols_in, int *nblk_mult_cols_in, int *nblk_in, int *np_bc_fine_in, int *np_cols_fine_in, int *np_cols_in, int *debug_in, gpuStream_t my_stream){
                                                        throw std::runtime_error("This function should never be called!!");

}

//________________________________________________________________

extern "C" void sycl_copy_and_set_zeros_aux_b_full_FromC (char dataType, intptr_t mat_dev, intptr_t aux_mat_full_dev,
                                                       int *l_rows_in, int *l_cols_in, int *nblk_mult_in, 
                                                       int *nblk_mult_rows_in, int *nblk_in, int *np_fine_in, int *np_rows_fine_in, int *np_rows_in,
                                                       int *SM_count_in, int *debug_in, gpuStream_t my_stream){
                                                        throw std::runtime_error("This function should never be called!!");

}

//________________________________________________________________



extern "C" void sycl_ccl_copy_buf_send_FromC (char dataType, intptr_t a_dev, intptr_t buf_send_dev, 
                                             int *l_rows_in, int *l_cols_in, int *lld_buf_in, int *nblk_in,
                                             int *i_block_loc_fine_in, int *j_block_loc_fine_in, int *np_fine_in, int *np_bc_fine_in, 
                                             int *np_rows_fine_in, int *np_cols_fine_in, int *np_rows_in, int *np_cols_in, int *SM_count_in, int *debug_in, gpuStream_t my_stream){
                                              throw std::runtime_error("This function should never be called!!");
                                            }

//________________________________________________________________



extern "C" void sycl_ccl_copy_buf_recv_FromC (char dataType, intptr_t at_col_dev, intptr_t buf_recv_dev, 
                                             int *l_rows_in, int *l_cols_in, int *lld_buf_in, int *nblk_in,
                                             int *i_block_loc_fine_max_in, int *j_block_loc_fine_max_in, int *np_fine_in, int *np_bc_fine_in, 
                                             int *np_rows_fine_in, int *np_cols_fine_in, int *np_rows_in, int *np_cols_in, int *SM_count_in, int *debug_in, gpuStream_t my_stream){
                                              throw std::runtime_error("This function should never be called!!");

}

//_________________________________________________________________________________________________
// non-square grid, TN, NT codepath


extern "C" void sycl_copy_and_set_zeros_aux_ab_full_tn_nt_FromC (char dataType, int *a_transoposed_in, intptr_t a_dev, intptr_t b_dev, intptr_t aux_a_full_dev, intptr_t aux_b_full_dev,
                                                             int *l_rows_in, int *l_cols_in, int *nblk_mult_max_in, int *nblk_mult_in, int *nblk_in,
                                                             int *np_ab_fine_in, int *np_rows_in, int *my_prow_in,
                                                             int *np_t_fine_in , int *np_cols_in, int *my_pcol_in,
                                                             int *np_dirs_fine_in,
                                                             int *SM_count_in, int *debug_in, gpuStream_t my_stream){
                                                              throw std::runtime_error("This function should never be called!!");
}

//________________________________________________________________

extern "C" void sycl_update_c_tn_nt_FromC (char dataType,
                                          int *a_transposed_in, 
                                          intptr_t c_dev, intptr_t tmp1_full_dev, int *beta_int_in,
                                          int *l_rows_in, int *l_cols_in, int *nblk_mult_max_in, int *nblk_mult_in, int *nblk_in,
                                          int *np_rows_in, int *np_cols_in, int *np_dirs_fine_in,
                                          int *np_dirs_t_in, int *my_pdir_t_in, int *np_fine_in,
                                          int *SM_count_in, int *debug_in, gpuStream_t my_stream) {
                                            throw std::runtime_error("This function should never be called!!");

}




