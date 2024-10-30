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
#include "../../../GPU/common_device_functions.h"
#include "../../../GPU/gpu_to_cuda_and_hip_interface.h"

#include "../elpa_pxgemm_multiply_gpu.h"

// //________________________________________________________________

// extern "C" void cuda_copy_aux_full_FromC(char dataType, intptr_t lhs_dev, intptr_t rhs_dev,
//                                          int *l_rows_in, int *l_cols_in, int *lld_lhs_in, int *lld_rhs_in, int *debug_in, cudaStream_t my_stream){
//   if (dataType=='D') cuda_copy_aux_full<double>((double *) lhs_dev, (double *) rhs_dev, l_rows_in, l_cols_in, lld_lhs_in, lld_rhs_in, debug_in, my_stream);
//   if (dataType=='S') cuda_copy_aux_full<float> ((float  *) lhs_dev, (float  *) rhs_dev, l_rows_in, l_cols_in, lld_lhs_in, lld_rhs_in, debug_in, my_stream);
//   if (dataType=='Z') cuda_copy_aux_full<cuDoubleComplex>((cuDoubleComplex *) lhs_dev, (cuDoubleComplex *) rhs_dev, l_rows_in, l_cols_in, lld_lhs_in, lld_rhs_in, debug_in, my_stream);
//   if (dataType=='C') cuda_copy_aux_full<cuFloatComplex> ((cuFloatComplex  *) lhs_dev, (cuFloatComplex  *) rhs_dev, l_rows_in, l_cols_in, lld_lhs_in, lld_rhs_in, debug_in, my_stream);
// }

// //________________________________________________________________

// extern "C" void cuda_copy_and_set_zeros_aux_full_FromC(char dataType, intptr_t mat_dev, intptr_t aux_mat_full_dev,
//                                                        int *l_rows_in, int *l_cols_in, int *nblk_mult_in, int *debug_in, cudaStream_t my_stream){
//   if (dataType=='D') cuda_copy_and_set_zeros_aux_full<double>((double *) mat_dev, (double *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_in, debug_in, my_stream);
//   if (dataType=='S') cuda_copy_and_set_zeros_aux_full<float> ((float  *) mat_dev, (float  *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_in, debug_in, my_stream);
//   if (dataType=='Z') cuda_copy_and_set_zeros_aux_full<cuDoubleComplex>((cuDoubleComplex *) mat_dev, (cuDoubleComplex *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_in, debug_in, my_stream);
//   if (dataType=='C') cuda_copy_and_set_zeros_aux_full<cuFloatComplex> ((cuFloatComplex  *) mat_dev, (cuFloatComplex  *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_in, debug_in, my_stream);
// }

// //________________________________________________________________

// extern "C" void cuda_copy_and_set_zeros_aux_a_full_FromC(char dataType, intptr_t mat_dev, intptr_t aux_mat_full_dev,
//                                                        int *l_rows_in, int *l_cols_in, int *nblk_mult_cols_in, int *nblk_in, int *np_bc_fine_in, int *np_cols_fine_in, int *np_cols_in, int *debug_in, cudaStream_t my_stream){
//   if (dataType=='D') cuda_copy_and_set_zeros_aux_a_full<double>((double *) mat_dev, (double *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_cols_in, nblk_in, np_bc_fine_in, np_cols_fine_in, np_cols_in, debug_in, my_stream);
//   if (dataType=='S') cuda_copy_and_set_zeros_aux_a_full<float> ((float  *) mat_dev, (float  *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_cols_in, nblk_in, np_bc_fine_in, np_cols_fine_in, np_cols_in, debug_in, my_stream);
//   if (dataType=='Z') cuda_copy_and_set_zeros_aux_a_full<cuDoubleComplex>((cuDoubleComplex *) mat_dev, (cuDoubleComplex *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_cols_in, nblk_in, np_bc_fine_in, np_cols_fine_in, np_cols_in, debug_in, my_stream);
//   if (dataType=='C') cuda_copy_and_set_zeros_aux_a_full<cuFloatComplex> ((cuFloatComplex  *) mat_dev, (cuFloatComplex  *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_cols_in, nblk_in, np_bc_fine_in, np_cols_fine_in, np_cols_in, debug_in, my_stream);
// }

// //________________________________________________________________

// extern "C" void cuda_copy_and_set_zeros_aux_b_full_FromC(char dataType, intptr_t mat_dev, intptr_t aux_mat_full_dev,
//                                                        int *l_rows_in, int *l_cols_in, int *nblk_mult_in, 
//                                                        int *nblk_mult_rows_in, int *nblk_in, int *np_fine_in, int *np_rows_fine_in, int *np_rows_in,
//                                                        int *SM_count_in, int *debug_in, cudaStream_t my_stream){
//   if (dataType=='D') cuda_copy_and_set_zeros_aux_b_full<double>((double *) mat_dev, (double *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_in, nblk_mult_rows_in, nblk_in, np_fine_in, np_rows_fine_in, np_rows_in, SM_count_in, debug_in, my_stream);
//   if (dataType=='S') cuda_copy_and_set_zeros_aux_b_full<float> ((float  *) mat_dev, (float  *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_in, nblk_mult_rows_in, nblk_in, np_fine_in, np_rows_fine_in, np_rows_in, SM_count_in, debug_in, my_stream);
//   if (dataType=='Z') cuda_copy_and_set_zeros_aux_b_full<cuDoubleComplex>((cuDoubleComplex *) mat_dev, (cuDoubleComplex *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_in, nblk_mult_rows_in, nblk_in, np_fine_in, np_rows_fine_in, np_rows_in, SM_count_in, debug_in, my_stream);
//   if (dataType=='C') cuda_copy_and_set_zeros_aux_b_full<cuFloatComplex> ((cuFloatComplex  *) mat_dev, (cuFloatComplex  *) aux_mat_full_dev, l_rows_in, l_cols_in, nblk_mult_in, nblk_mult_rows_in, nblk_in, np_fine_in, np_rows_fine_in, np_rows_in, SM_count_in, debug_in, my_stream);
// }

// //________________________________________________________________

// extern "C" void cuda_ccl_copy_buf_send_FromC(char dataType, intptr_t a_dev, intptr_t buf_send_dev, 
//                                              int *l_rows_in, int *l_cols_in, int *lld_buf_in, int *nblk_in,
//                                              int *i_block_loc_fine_in, int *j_block_loc_fine_in, int *np_fine_in, int *np_bc_fine_in, 
//                                              int *np_rows_fine_in, int *np_cols_fine_in, int *np_rows_in, int *np_cols_in, int *SM_count_in, int *debug_in, cudaStream_t my_stream){
//   if (dataType=='D') cuda_ccl_copy_buf_send<double>((double *) a_dev, (double *) buf_send_dev, l_rows_in, l_cols_in, lld_buf_in, nblk_in,
//                                                     i_block_loc_fine_in, j_block_loc_fine_in, np_fine_in, np_bc_fine_in, 
//                                                     np_rows_fine_in, np_cols_fine_in, np_rows_in, np_cols_in, SM_count_in, debug_in, my_stream);
//   if (dataType=='S') cuda_ccl_copy_buf_send<float> ((float  *) a_dev, (float  *) buf_send_dev, l_rows_in, l_cols_in, lld_buf_in, nblk_in,
//                                                     i_block_loc_fine_in, j_block_loc_fine_in, np_fine_in, np_bc_fine_in, 
//                                                     np_rows_fine_in, np_cols_fine_in, np_rows_in, np_cols_in, SM_count_in, debug_in, my_stream);
//   if (dataType=='Z') cuda_ccl_copy_buf_send<cuDoubleComplex>((cuDoubleComplex *) a_dev, (cuDoubleComplex *) buf_send_dev, l_rows_in, l_cols_in, lld_buf_in, nblk_in,
//                                                     i_block_loc_fine_in, j_block_loc_fine_in, np_fine_in, np_bc_fine_in, 
//                                                     np_rows_fine_in, np_cols_fine_in, np_rows_in, np_cols_in, SM_count_in, debug_in, my_stream);
//   if (dataType=='C') cuda_ccl_copy_buf_send<cuFloatComplex> ((cuFloatComplex  *) a_dev, (cuFloatComplex  *) buf_send_dev, l_rows_in, l_cols_in, lld_buf_in, nblk_in,
//                                                     i_block_loc_fine_in, j_block_loc_fine_in, np_fine_in, np_bc_fine_in,
//                                                     np_rows_fine_in, np_cols_fine_in, np_rows_in, np_cols_in, SM_count_in, debug_in, my_stream);
// }

// //________________________________________________________________

// extern "C" void cuda_ccl_copy_buf_recv_FromC(char dataType, intptr_t at_col_dev, intptr_t buf_recv_dev, 
//                                              int *l_rows_in, int *l_cols_in, int *lld_buf_in, int *nblk_in,
//                                              int *i_block_loc_fine_max_in, int *j_block_loc_fine_max_in, int *np_fine_in, int *np_bc_fine_in, 
//                                              int *np_rows_fine_in, int *np_cols_fine_in, int *np_rows_in, int *np_cols_in, int *SM_count_in, int *debug_in, cudaStream_t my_stream){
//   if (dataType=='D') cuda_ccl_copy_buf_recv<double>((double *) at_col_dev, (double *) buf_recv_dev, l_rows_in, l_cols_in, lld_buf_in, nblk_in,
//                                                     i_block_loc_fine_max_in, j_block_loc_fine_max_in, np_fine_in, np_bc_fine_in, 
//                                                     np_rows_fine_in, np_cols_fine_in, np_rows_in, np_cols_in, SM_count_in, debug_in, my_stream);
//   if (dataType=='S') cuda_ccl_copy_buf_recv<float> ((float  *) at_col_dev, (float  *) buf_recv_dev, l_rows_in, l_cols_in, lld_buf_in, nblk_in,
//                                                     i_block_loc_fine_max_in, j_block_loc_fine_max_in, np_fine_in, np_bc_fine_in, 
//                                                     np_rows_fine_in, np_cols_fine_in, np_rows_in, np_cols_in, SM_count_in, debug_in, my_stream);
//   if (dataType=='Z') cuda_ccl_copy_buf_recv<cuDoubleComplex>((cuDoubleComplex *) at_col_dev, (cuDoubleComplex *) buf_recv_dev, l_rows_in, l_cols_in, lld_buf_in, nblk_in,
//                                                     i_block_loc_fine_max_in, j_block_loc_fine_max_in, np_fine_in, np_bc_fine_in, 
//                                                     np_rows_fine_in, np_cols_fine_in, np_rows_in, np_cols_in, SM_count_in, debug_in, my_stream);
//   if (dataType=='C') cuda_ccl_copy_buf_recv<cuFloatComplex> ((cuFloatComplex  *) at_col_dev, (cuFloatComplex  *) buf_recv_dev, l_rows_in, l_cols_in, lld_buf_in, nblk_in,
//                                                     i_block_loc_fine_max_in, j_block_loc_fine_max_in, np_fine_in, np_bc_fine_in,
//                                                     np_rows_fine_in, np_cols_fine_in, np_rows_in, np_cols_in, SM_count_in, debug_in, my_stream);
// }

// //_________________________________________________________________________________________________

// extern "C" void cuda_copy_and_set_zeros_aux_ab_full_tn_nt_FromC(char dataType, int *a_transoposed_in, intptr_t a_dev, intptr_t b_dev, intptr_t aux_a_full_dev, intptr_t aux_b_full_dev,
//                                                              int *l_rows_in, int *l_cols_in, int *nblk_mult_max_in, int *nblk_mult_in, int *nblk_in,
//                                                              int *np_ab_fine_in, int *np_rows_in, int *my_prow_in,
//                                                              int *np_t_fine_in , int *np_cols_in, int *my_pcol_in,
//                                                              int *np_dirs_fine_in,
//                                                              int *SM_count_in, int *debug_in, cudaStream_t my_stream){
//   if (dataType == 'D') cuda_copy_and_set_zeros_aux_ab_full_tn_nt<double>(a_transoposed_in, (double *)a_dev, (double *)b_dev, (double *)aux_a_full_dev, (double *)aux_b_full_dev,
//                                                        l_rows_in, l_cols_in, nblk_mult_max_in, nblk_mult_in, nblk_in,
//                                                        np_ab_fine_in, np_rows_in, my_prow_in,
//                                                        np_t_fine_in , np_cols_in, my_pcol_in,
//                                                        np_dirs_fine_in,
//                                                        SM_count_in, debug_in, my_stream);
//   if (dataType == 'S') cuda_copy_and_set_zeros_aux_ab_full_tn_nt<float>(a_transoposed_in, (float *)a_dev, (float *)b_dev, (float *)aux_a_full_dev, (float *)aux_b_full_dev,
//                                                       l_rows_in, l_cols_in, nblk_mult_max_in, nblk_mult_in, nblk_in,
//                                                       np_ab_fine_in, np_rows_in, my_prow_in,
//                                                       np_t_fine_in , np_cols_in, my_pcol_in,
//                                                       np_dirs_fine_in,
//                                                       SM_count_in, debug_in, my_stream);
//   if (dataType == 'Z') cuda_copy_and_set_zeros_aux_ab_full_tn_nt<cuDoubleComplex>(a_transoposed_in, (cuDoubleComplex *)a_dev, (cuDoubleComplex *)b_dev, (cuDoubleComplex *)aux_a_full_dev, (cuDoubleComplex *)aux_b_full_dev,
//                                                                 l_rows_in, l_cols_in, nblk_mult_max_in, nblk_mult_in, nblk_in,
//                                                                 np_ab_fine_in, np_rows_in, my_prow_in,
//                                                                 np_t_fine_in , np_cols_in, my_pcol_in,
//                                                                 np_dirs_fine_in,
//                                                                 SM_count_in, debug_in, my_stream);
//   if (dataType == 'C') cuda_copy_and_set_zeros_aux_ab_full_tn_nt<cuFloatComplex>(a_transoposed_in, (cuFloatComplex *)a_dev, (cuFloatComplex *)b_dev, (cuFloatComplex *)aux_a_full_dev, (cuFloatComplex *)aux_b_full_dev,
//                                                                l_rows_in, l_cols_in, nblk_mult_max_in, nblk_mult_in, nblk_in,
//                                                                np_ab_fine_in, np_rows_in, my_prow_in,
//                                                                np_t_fine_in , np_cols_in, my_pcol_in,
//                                                                np_dirs_fine_in,
//                                                                SM_count_in, debug_in, my_stream);
// }

// //________________________________________________________________

// extern "C" void cuda_update_c_tn_nt_FromC(char dataType,
//                                           int *a_transposed_in, 
//                                           intptr_t c_dev, intptr_t tmp1_full_dev, int *beta_int_in,
//                                           int *l_rows_in, int *l_cols_in, int *nblk_mult_max_in, int *nblk_mult_in, int *nblk_in,
//                                           int *np_rows_in, int *np_cols_in, int *np_dirs_fine_in,
//                                           int *np_dirs_t_in, int *my_pdir_t_in, int *np_fine_in,
//                                           int *SM_count_in, int *debug_in, cudaStream_t my_stream) {

//   if (dataType == 'D') cuda_update_c_tn_nt<double>(a_transposed_in, 
//                                 (double *)c_dev, (double *)tmp1_full_dev, beta_int_in,
//                                 l_rows_in, l_cols_in, nblk_mult_max_in, nblk_mult_in, nblk_in,
//                                 np_rows_in, np_cols_in, np_dirs_fine_in,
//                                 np_dirs_t_in, my_pdir_t_in, np_fine_in,
//                                 SM_count_in, debug_in, my_stream);
//   if (dataType == 'S') cuda_update_c_tn_nt<float>(a_transposed_in, 
//                                 (float *)c_dev, (float *)tmp1_full_dev, beta_int_in,
//                                 l_rows_in, l_cols_in, nblk_mult_max_in, nblk_mult_in, nblk_in,
//                                 np_rows_in, np_cols_in, np_dirs_fine_in,
//                                 np_dirs_t_in, my_pdir_t_in, np_fine_in,
//                                 SM_count_in, debug_in, my_stream);
//   if (dataType == 'Z') cuda_update_c_tn_nt<cuDoubleComplex>(a_transposed_in, 
//                                           (cuDoubleComplex *)c_dev, (cuDoubleComplex *)tmp1_full_dev, beta_int_in,
//                                           l_rows_in, l_cols_in, nblk_mult_max_in, nblk_mult_in, nblk_in,
//                                           np_rows_in, np_cols_in, np_dirs_fine_in,
//                                           np_dirs_t_in, my_pdir_t_in, np_fine_in,
//                                           SM_count_in, debug_in, my_stream);
//   else if (dataType == 'C') cuda_update_c_tn_nt<cuFloatComplex>(a_transposed_in, 
//                                         (cuFloatComplex *)c_dev, (cuFloatComplex *)tmp1_full_dev, beta_int_in,
//                                         l_rows_in, l_cols_in, nblk_mult_max_in, nblk_mult_in, nblk_in,
//                                         np_rows_in, np_cols_in, np_dirs_fine_in,
//                                         np_dirs_t_in, my_pdir_t_in, np_fine_in,
//                                         SM_count_in, debug_in, my_stream);
// }

