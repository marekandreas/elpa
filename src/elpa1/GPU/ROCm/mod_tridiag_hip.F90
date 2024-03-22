#if 0
!    Copyright 2023, P. Karpov, MPCDF
!
!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), formerly known as
!      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
!    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!      Informatik,
!    - Technische Universität München, Lehrstuhl für Informatik mit
!      Schwerpunkt Wissenschaftliches Rechnen ,
!    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!      and
!    - IBM Deutschland GmbH
!
!    More information can be found here:
!    http://elpa.mpcdf.mpg.de/
!
!    ELPA is free software: you can redistribute it and/or modify
!    it under the terms of the version 3 of the license of the
!    GNU Lesser General Public License as published by the Free
!    Software Foundation.
!
!    ELPA is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public License
!    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
!
!    ELPA reflects a substantial effort on the part of the original
!    ELPA consortium, and we ask you to respect the spirit of the
!    license that we chose: i.e., please contribute any changes you
!    may have back to the original ELPA library distribution, and keep
!    any derivatives of ELPA under the same license that we chose for
!    the original distribution, the GNU Lesser General Public License.

!    This file was written by P. Karpov, MPCDF
#endif


#include "config-f90.h"

! This file is auto-generated. Do NOT edit

module tridiag_hip
  use, intrinsic :: iso_c_binding
  use precision

  implicit none

  public

  interface
    subroutine hip_copy_and_set_zeros_double_c(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                                  aux1_dev, vav_dev, d_vec_dev, &
                                                  isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int, &
                                                  isSkewsymmetric_int, useCCL_int, wantDebug_int, &
                                                  my_stream) &
                                                  bind(C, name="hip_copy_and_set_zeros_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: v_row_dev, a_dev, aux1_dev, vav_dev, d_vec_dev
      integer(kind=c_int), intent(in)  :: l_rows, l_cols, matrixRows, istep
      integer(kind=c_int), intent(in)  :: isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int
      integer(kind=c_int), intent(in)  :: isSkewsymmetric_int, useCCL_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_dot_product_double_c(n, x_dev, incx, y_dev, incy, result_dev, wantDebug_int, sm_count, my_stream) &
                                                  bind(C, name="hip_dot_product_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: x_dev, y_dev, result_dev
      integer(kind=c_int), intent(in)  :: n, incx, incy, sm_count
      integer(kind=c_int), intent(in)  :: wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_dot_product_and_assign_double_c(v_row_dev, l_rows, isOurProcessRow_int, aux1_dev, &
                                                  wantDebug_int, my_stream) &
                                                  bind(C, name="hip_dot_product_and_assign_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: v_row_dev, aux1_dev
      integer(kind=c_int), intent(in)  :: l_rows
      integer(kind=c_int), intent(in)  :: isOurProcessRow_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_set_e_vec_scale_set_one_store_v_row_double_c(e_vec_dev, vrl_dev, a_dev, v_row_dev, &
                                     tau_dev, xf_host_or_dev, &
                                     l_rows, l_cols, matrixRows, istep, isOurProcessRow_int, useCCL_int, wantDebug_int, my_stream) &
                                     bind(C, name="hip_set_e_vec_scale_set_one_store_v_row_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev
      integer(kind=c_int), intent(in)  :: l_rows, l_cols, matrixRows, istep
      integer(kind=c_int), intent(in)  :: isOurProcessRow_int, useCCL_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_store_u_v_in_uv_vu_double_c(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                                  v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                                  vav_host_or_dev, tau_istep_host_or_dev, &
                                                  l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                                  useCCL_int, wantDebug_int, my_stream) &
                                                  bind(C, name="hip_store_u_v_in_uv_vu_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev
      integer(kind=c_intptr_t), value  :: v_col_dev, u_col_dev, tau_dev, aux_complex_dev
      integer(kind=c_intptr_t), value  :: vav_host_or_dev, tau_istep_host_or_dev
      integer(kind=c_int), intent(in)  :: l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep
      integer(kind=c_int), intent(in)  :: useCCL_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_update_matrix_element_add_double_c(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                             l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                             isSkewsymmetric_int, wantDebug_int, my_stream) &
                                             bind(C, name="hip_update_matrix_element_add_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev
      integer(kind=c_int), intent(in)  :: l_rows, l_cols, matrixRows, max_local_rows, max_local_cols
      integer(kind=c_int), intent(in)  :: istep, n_stored_vecs
      integer(kind=c_int), intent(in)  :: isSkewsymmetric_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_update_array_element_double_c(array_dev, index, value, my_stream) &
                                                  bind(C, name="hip_update_array_element_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: array_dev
      integer(kind=c_int), intent(in)  :: index
      integer(kind=c_intptr_t)         :: value
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_hh_transform_double_c(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug_int, my_stream) &
                                                  bind(C, name="hip_hh_transform_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: alpha_dev, xnorm_sq_dev, xf_dev, tau_dev
      integer(kind=c_int), intent(in)  :: wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_transpose_reduceadd_vectors_copy_block_double_c(aux_transpose_dev, vmat_st_dev, &
                                                  nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                                  lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                                  isSkewsymmetric_int, isReduceadd_int, wantDebug_int, sm_count, my_stream) &
                                                  bind(C, name="hip_transpose_reduceadd_vectors_copy_block_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_T), value  :: aux_transpose_dev, vmat_st_dev
      integer(kind=c_int), intent(in)  :: nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride
      integer(kind=c_int), intent(in)  :: np_st, ld_st, direction, sm_count
      integer(kind=c_int), intent(in)  :: isSkewsymmetric_int, isReduceadd_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_copy_and_set_zeros_float_c(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                                  aux1_dev, vav_dev, d_vec_dev, &
                                                  isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int, &
                                                  isSkewsymmetric_int, useCCL_int, wantDebug_int, &
                                                  my_stream) &
                                                  bind(C, name="hip_copy_and_set_zeros_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: v_row_dev, a_dev, aux1_dev, vav_dev, d_vec_dev
      integer(kind=c_int), intent(in)  :: l_rows, l_cols, matrixRows, istep
      integer(kind=c_int), intent(in)  :: isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int
      integer(kind=c_int), intent(in)  :: isSkewsymmetric_int, useCCL_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_dot_product_float_c(n, x_dev, incx, y_dev, incy, result_dev, wantDebug_int, sm_count, my_stream) &
                                                  bind(C, name="hip_dot_product_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: x_dev, y_dev, result_dev
      integer(kind=c_int), intent(in)  :: n, incx, incy, sm_count
      integer(kind=c_int), intent(in)  :: wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_dot_product_and_assign_float_c(v_row_dev, l_rows, isOurProcessRow_int, aux1_dev, &
                                                  wantDebug_int, my_stream) &
                                                  bind(C, name="hip_dot_product_and_assign_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: v_row_dev, aux1_dev
      integer(kind=c_int), intent(in)  :: l_rows
      integer(kind=c_int), intent(in)  :: isOurProcessRow_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_set_e_vec_scale_set_one_store_v_row_float_c(e_vec_dev, vrl_dev, a_dev, v_row_dev, &
                                     tau_dev, xf_host_or_dev, &
                                     l_rows, l_cols, matrixRows, istep, isOurProcessRow_int, useCCL_int, wantDebug_int, my_stream) &
                                     bind(C, name="hip_set_e_vec_scale_set_one_store_v_row_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev
      integer(kind=c_int), intent(in)  :: l_rows, l_cols, matrixRows, istep
      integer(kind=c_int), intent(in)  :: isOurProcessRow_int, useCCL_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_store_u_v_in_uv_vu_float_c(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                                  v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                                  vav_host_or_dev, tau_istep_host_or_dev, &
                                                  l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                                  useCCL_int, wantDebug_int, my_stream) &
                                                  bind(C, name="hip_store_u_v_in_uv_vu_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev
      integer(kind=c_intptr_t), value  :: v_col_dev, u_col_dev, tau_dev, aux_complex_dev
      integer(kind=c_intptr_t), value  :: vav_host_or_dev, tau_istep_host_or_dev
      integer(kind=c_int), intent(in)  :: l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep
      integer(kind=c_int), intent(in)  :: useCCL_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_update_matrix_element_add_float_c(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                             l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                             isSkewsymmetric_int, wantDebug_int, my_stream) &
                                             bind(C, name="hip_update_matrix_element_add_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev
      integer(kind=c_int), intent(in)  :: l_rows, l_cols, matrixRows, max_local_rows, max_local_cols
      integer(kind=c_int), intent(in)  :: istep, n_stored_vecs
      integer(kind=c_int), intent(in)  :: isSkewsymmetric_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_update_array_element_float_c(array_dev, index, value, my_stream) &
                                                  bind(C, name="hip_update_array_element_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: array_dev
      integer(kind=c_int), intent(in)  :: index
      integer(kind=c_intptr_t)         :: value
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_hh_transform_float_c(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug_int, my_stream) &
                                                  bind(C, name="hip_hh_transform_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: alpha_dev, xnorm_sq_dev, xf_dev, tau_dev
      integer(kind=c_int), intent(in)  :: wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_transpose_reduceadd_vectors_copy_block_float_c(aux_transpose_dev, vmat_st_dev, &
                                                  nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                                  lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                                  isSkewsymmetric_int, isReduceadd_int, wantDebug_int, sm_count, my_stream) &
                                                  bind(C, name="hip_transpose_reduceadd_vectors_copy_block_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_T), value  :: aux_transpose_dev, vmat_st_dev
      integer(kind=c_int), intent(in)  :: nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride
      integer(kind=c_int), intent(in)  :: np_st, ld_st, direction, sm_count
      integer(kind=c_int), intent(in)  :: isSkewsymmetric_int, isReduceadd_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_copy_and_set_zeros_double_complex_c(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                                  aux1_dev, vav_dev, d_vec_dev, &
                                                  isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int, &
                                                  isSkewsymmetric_int, useCCL_int, wantDebug_int, &
                                                  my_stream) &
                                                  bind(C, name="hip_copy_and_set_zeros_double_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: v_row_dev, a_dev, aux1_dev, vav_dev, d_vec_dev
      integer(kind=c_int), intent(in)  :: l_rows, l_cols, matrixRows, istep
      integer(kind=c_int), intent(in)  :: isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int
      integer(kind=c_int), intent(in)  :: isSkewsymmetric_int, useCCL_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_dot_product_double_complex_c(n, x_dev, incx, y_dev, incy, result_dev, wantDebug_int, sm_count, my_stream) &
                                                  bind(C, name="hip_dot_product_double_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: x_dev, y_dev, result_dev
      integer(kind=c_int), intent(in)  :: n, incx, incy, sm_count
      integer(kind=c_int), intent(in)  :: wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_dot_product_and_assign_double_complex_c(v_row_dev, l_rows, isOurProcessRow_int, aux1_dev, &
                                                  wantDebug_int, my_stream) &
                                                  bind(C, name="hip_dot_product_and_assign_double_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: v_row_dev, aux1_dev
      integer(kind=c_int), intent(in)  :: l_rows
      integer(kind=c_int), intent(in)  :: isOurProcessRow_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_set_e_vec_scale_set_one_store_v_row_double_complex_c(e_vec_dev, vrl_dev, a_dev, v_row_dev, &
                                     tau_dev, xf_host_or_dev, &
                                     l_rows, l_cols, matrixRows, istep, isOurProcessRow_int, useCCL_int, wantDebug_int, my_stream) &
                                     bind(C, name="hip_set_e_vec_scale_set_one_store_v_row_double_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev
      integer(kind=c_int), intent(in)  :: l_rows, l_cols, matrixRows, istep
      integer(kind=c_int), intent(in)  :: isOurProcessRow_int, useCCL_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_store_u_v_in_uv_vu_double_complex_c(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                                  v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                                  vav_host_or_dev, tau_istep_host_or_dev, &
                                                  l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                                  useCCL_int, wantDebug_int, my_stream) &
                                                  bind(C, name="hip_store_u_v_in_uv_vu_double_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev
      integer(kind=c_intptr_t), value  :: v_col_dev, u_col_dev, tau_dev, aux_complex_dev
      integer(kind=c_intptr_t), value  :: vav_host_or_dev, tau_istep_host_or_dev
      integer(kind=c_int), intent(in)  :: l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep
      integer(kind=c_int), intent(in)  :: useCCL_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_update_matrix_element_add_double_complex_c(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                             l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                             isSkewsymmetric_int, wantDebug_int, my_stream) &
                                             bind(C, name="hip_update_matrix_element_add_double_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev
      integer(kind=c_int), intent(in)  :: l_rows, l_cols, matrixRows, max_local_rows, max_local_cols
      integer(kind=c_int), intent(in)  :: istep, n_stored_vecs
      integer(kind=c_int), intent(in)  :: isSkewsymmetric_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_update_array_element_double_complex_c(array_dev, index, value, my_stream) &
                                                  bind(C, name="hip_update_array_element_double_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: array_dev
      integer(kind=c_int), intent(in)  :: index
      integer(kind=c_intptr_t)         :: value
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_hh_transform_double_complex_c(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug_int, my_stream) &
                                                  bind(C, name="hip_hh_transform_double_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: alpha_dev, xnorm_sq_dev, xf_dev, tau_dev
      integer(kind=c_int), intent(in)  :: wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_transpose_reduceadd_vectors_copy_block_double_complex_c(aux_transpose_dev, vmat_st_dev, &
                                                  nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                                  lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                                  isSkewsymmetric_int, isReduceadd_int, wantDebug_int, sm_count, my_stream) &
                                                  bind(C, name="hip_transpose_reduceadd_vectors_copy_block_double_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_T), value  :: aux_transpose_dev, vmat_st_dev
      integer(kind=c_int), intent(in)  :: nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride
      integer(kind=c_int), intent(in)  :: np_st, ld_st, direction, sm_count
      integer(kind=c_int), intent(in)  :: isSkewsymmetric_int, isReduceadd_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_copy_and_set_zeros_float_complex_c(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                                  aux1_dev, vav_dev, d_vec_dev, &
                                                  isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int, &
                                                  isSkewsymmetric_int, useCCL_int, wantDebug_int, &
                                                  my_stream) &
                                                  bind(C, name="hip_copy_and_set_zeros_float_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: v_row_dev, a_dev, aux1_dev, vav_dev, d_vec_dev
      integer(kind=c_int), intent(in)  :: l_rows, l_cols, matrixRows, istep
      integer(kind=c_int), intent(in)  :: isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int
      integer(kind=c_int), intent(in)  :: isSkewsymmetric_int, useCCL_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_dot_product_float_complex_c(n, x_dev, incx, y_dev, incy, result_dev, wantDebug_int, sm_count, my_stream) &
                                                  bind(C, name="hip_dot_product_float_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: x_dev, y_dev, result_dev
      integer(kind=c_int), intent(in)  :: n, incx, incy, sm_count
      integer(kind=c_int), intent(in)  :: wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_dot_product_and_assign_float_complex_c(v_row_dev, l_rows, isOurProcessRow_int, aux1_dev, &
                                                  wantDebug_int, my_stream) &
                                                  bind(C, name="hip_dot_product_and_assign_float_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: v_row_dev, aux1_dev
      integer(kind=c_int), intent(in)  :: l_rows
      integer(kind=c_int), intent(in)  :: isOurProcessRow_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_set_e_vec_scale_set_one_store_v_row_float_complex_c(e_vec_dev, vrl_dev, a_dev, v_row_dev, &
                                     tau_dev, xf_host_or_dev, &
                                     l_rows, l_cols, matrixRows, istep, isOurProcessRow_int, useCCL_int, wantDebug_int, my_stream) &
                                     bind(C, name="hip_set_e_vec_scale_set_one_store_v_row_float_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev
      integer(kind=c_int), intent(in)  :: l_rows, l_cols, matrixRows, istep
      integer(kind=c_int), intent(in)  :: isOurProcessRow_int, useCCL_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_store_u_v_in_uv_vu_float_complex_c(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                                  v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                                  vav_host_or_dev, tau_istep_host_or_dev, &
                                                  l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                                  useCCL_int, wantDebug_int, my_stream) &
                                                  bind(C, name="hip_store_u_v_in_uv_vu_float_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev
      integer(kind=c_intptr_t), value  :: v_col_dev, u_col_dev, tau_dev, aux_complex_dev
      integer(kind=c_intptr_t), value  :: vav_host_or_dev, tau_istep_host_or_dev
      integer(kind=c_int), intent(in)  :: l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep
      integer(kind=c_int), intent(in)  :: useCCL_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_update_matrix_element_add_float_complex_c(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                             l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                             isSkewsymmetric_int, wantDebug_int, my_stream) &
                                             bind(C, name="hip_update_matrix_element_add_float_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev
      integer(kind=c_int), intent(in)  :: l_rows, l_cols, matrixRows, max_local_rows, max_local_cols
      integer(kind=c_int), intent(in)  :: istep, n_stored_vecs
      integer(kind=c_int), intent(in)  :: isSkewsymmetric_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_update_array_element_float_complex_c(array_dev, index, value, my_stream) &
                                                  bind(C, name="hip_update_array_element_float_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: array_dev
      integer(kind=c_int), intent(in)  :: index
      integer(kind=c_intptr_t)         :: value
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_hh_transform_float_complex_c(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug_int, my_stream) &
                                                  bind(C, name="hip_hh_transform_float_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: alpha_dev, xnorm_sq_dev, xf_dev, tau_dev
      integer(kind=c_int), intent(in)  :: wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_transpose_reduceadd_vectors_copy_block_float_complex_c(aux_transpose_dev, vmat_st_dev, &
                                                  nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                                  lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                                  isSkewsymmetric_int, isReduceadd_int, wantDebug_int, sm_count, my_stream) &
                                                  bind(C, name="hip_transpose_reduceadd_vectors_copy_block_float_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_T), value  :: aux_transpose_dev, vmat_st_dev
      integer(kind=c_int), intent(in)  :: nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride
      integer(kind=c_int), intent(in)  :: np_st, ld_st, direction, sm_count
      integer(kind=c_int), intent(in)  :: isSkewsymmetric_int, isReduceadd_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  contains

    subroutine hip_copy_and_set_zeros_double(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep,  &
                                                 aux1_dev, vav_dev, d_vec_dev, &
                                                 isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, &
                                                 isSkewsymmetric, useCCL, wantDebug, my_stream) 
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows, l_cols, matrixRows, istep
      logical, intent(in)             :: isOurProcessRow, isOurProcessCol, isOurProcessCol_prev
      logical, intent(in)             :: isSkewsymmetric, useCCL, wantDebug
      integer(kind=c_intptr_t)        :: v_row_dev, a_dev, aux1_dev, vav_dev, d_vec_dev
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int
      integer(kind=c_int)             :: isSkewsymmetric_int, useCCL_int, wantDebug_int

      isOurProcessRow_int = 0
      isOurProcessCol_int = 0
      isOurProcessCol_prev_int = 0
      isSkewsymmetric_int = 0
      useCCL_int = 0
      wantDebug_int = 0
      if (isOurProcessRow) isOurProcessRow_int = 1
      if (isOurProcessCol) isOurProcessCol_int = 1
      if (isOurProcessCol_prev) isOurProcessCol_prev_int = 1
      if (isSkewsymmetric) isSkewsymmetric_int = 1
      if (useCCL) useCCL_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_and_set_zeros_double_c(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                            aux1_dev, vav_dev, d_vec_dev, &
                                            isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int, &
                                            isSkewsymmetric_int, useCCL_int, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_dot_product_double(n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: n, incx, incy, sm_count
      logical, intent(in)             :: wantDebug
      integer(kind=C_intptr_T)        :: x_dev, y_dev, result_dev
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: wantDebug_int

      wantDebug_int = 0
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_dot_product_double_c(n, x_dev, incx, y_dev, incy, result_dev, wantDebug_int, sm_count, my_stream)
#endif

    end subroutine

    subroutine hip_dot_product_and_assign_double(v_row_dev, l_rows, isOurProcessRow, aux1_dev, &
                                                                     wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows
      logical, intent(in)             :: isOurProcessRow, wantDebug
      integer(kind=c_intptr_t)        :: v_row_dev, aux1_dev
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isOurProcessRow_int, wantDebug_int

      isOurProcessRow_int = 0
      wantDebug_int = 0
      if (isOurProcessRow) isOurProcessRow_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_dot_product_and_assign_double_c(v_row_dev, l_rows, isOurProcessRow_int, aux1_dev, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_set_e_vec_scale_set_one_store_v_row_double(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, &
                                            xf_host_or_dev, l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                            wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows, l_cols, matrixRows, istep
      logical, intent(in)             :: isOurProcessRow, useCCL, wantDebug
      integer(kind=c_intptr_t)        :: e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isOurProcessRow_int, useCCL_int, wantDebug_int

      isOurProcessRow_int = 0
      useCCL_int = 0
      wantDebug_int = 0
      if (isOurProcessRow) isOurProcessRow_int = 1
      if (useCCL) useCCL_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_set_e_vec_scale_set_one_store_v_row_double_c(e_vec_dev, vrl_dev, a_dev, &
                                       v_row_dev, tau_dev, xf_host_or_dev, &
                                       l_rows, l_cols, matrixRows, istep, isOurProcessRow_int, useCCL_int, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_store_u_v_in_uv_vu_double(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                            v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                            vav_host_or_dev, tau_istep_host_or_dev, &
                                            l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                            useCCL, wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep
      integer(kind=c_intptr_t)        :: vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev
      integer(kind=c_intptr_t)        :: v_col_dev, u_col_dev, tau_dev, aux_complex_dev
      integer(kind=c_intptr_t)        :: vav_host_or_dev, tau_istep_host_or_dev
      logical, intent(in)             :: useCCL, wantDebug
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: useCCL_int, wantDebug_int

      useCCL_int = 0
      wantDebug_int = 0
      if (useCCL) useCCL_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_store_u_v_in_uv_vu_double_c(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                       v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                       vav_host_or_dev, tau_istep_host_or_dev, &
                                       l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                       useCCL_int, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_update_matrix_element_add_double(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev,  &
                                            l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                            isSkewsymmetric, wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows, l_cols, matrixRows, max_local_rows, max_local_cols
      integer(kind=c_int), intent(in) :: istep, n_stored_vecs
      integer(kind=c_intptr_t)        :: vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev
      logical, intent(in)             :: isSkewsymmetric, wantDebug
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isSkewsymmetric_int, wantDebug_int

      isSkewsymmetric_int = 0
      wantDebug_int = 0
      if (isSkewsymmetric) isSkewsymmetric_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_update_matrix_element_add_double_c(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev,  &
                                       l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                       isSkewsymmetric_int, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_update_array_element_double(array_dev, index, value, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: index
      integer(kind=c_intptr_t)        :: array_dev
      integer(kind=c_intptr_t)        :: value
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_AMD_GPU_VERSION
      call hip_update_array_element_double_c(array_dev, index, value, my_stream)
#endif

    end subroutine

    subroutine hip_hh_transform_double(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_intptr_t)        :: alpha_dev, xnorm_sq_dev, xf_dev, tau_dev
      logical, intent(in)             :: wantDebug
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: wantDebug_int

      wantDebug_int = 0
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_hh_transform_double_c(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_transpose_reduceadd_vectors_copy_block_double(aux_transpose_dev, vmat_st_dev, &
                                              nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                              lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                              isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride
      integer(kind=c_int), intent(in) :: np_st, ld_st, direction, sm_count
      integer(kind=c_intptr_t)        :: aux_transpose_dev, vmat_st_dev
      logical, intent(in)             :: isSkewsymmetric, isReduceadd, wantDebug
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isSkewsymmetric_int, isReduceadd_int, wantDebug_int

      isSkewsymmetric_int = 0
      isReduceadd_int = 0
      wantDebug_int = 0
      if (isSkewsymmetric) isSkewsymmetric_int = 1
      if (isReduceadd) isReduceadd_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_transpose_reduceadd_vectors_copy_block_double_c(aux_transpose_dev, vmat_st_dev, &
                                          nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                          lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                          isSkewsymmetric_int, isReduceadd_int, wantDebug_int, sm_count, my_stream)
#endif

    end subroutine
    subroutine hip_copy_and_set_zeros_float(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep,  &
                                                 aux1_dev, vav_dev, d_vec_dev, &
                                                 isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, &
                                                 isSkewsymmetric, useCCL, wantDebug, my_stream) 
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows, l_cols, matrixRows, istep
      logical, intent(in)             :: isOurProcessRow, isOurProcessCol, isOurProcessCol_prev
      logical, intent(in)             :: isSkewsymmetric, useCCL, wantDebug
      integer(kind=c_intptr_t)        :: v_row_dev, a_dev, aux1_dev, vav_dev, d_vec_dev
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int
      integer(kind=c_int)             :: isSkewsymmetric_int, useCCL_int, wantDebug_int

      isOurProcessRow_int = 0
      isOurProcessCol_int = 0
      isOurProcessCol_prev_int = 0
      isSkewsymmetric_int = 0
      useCCL_int = 0
      wantDebug_int = 0
      if (isOurProcessRow) isOurProcessRow_int = 1
      if (isOurProcessCol) isOurProcessCol_int = 1
      if (isOurProcessCol_prev) isOurProcessCol_prev_int = 1
      if (isSkewsymmetric) isSkewsymmetric_int = 1
      if (useCCL) useCCL_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_and_set_zeros_float_c(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                            aux1_dev, vav_dev, d_vec_dev, &
                                            isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int, &
                                            isSkewsymmetric_int, useCCL_int, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_dot_product_float(n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: n, incx, incy, sm_count
      logical, intent(in)             :: wantDebug
      integer(kind=C_intptr_T)        :: x_dev, y_dev, result_dev
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: wantDebug_int

      wantDebug_int = 0
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_dot_product_float_c(n, x_dev, incx, y_dev, incy, result_dev, wantDebug_int, sm_count, my_stream)
#endif

    end subroutine

    subroutine hip_dot_product_and_assign_float(v_row_dev, l_rows, isOurProcessRow, aux1_dev, &
                                                                     wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows
      logical, intent(in)             :: isOurProcessRow, wantDebug
      integer(kind=c_intptr_t)        :: v_row_dev, aux1_dev
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isOurProcessRow_int, wantDebug_int

      isOurProcessRow_int = 0
      wantDebug_int = 0
      if (isOurProcessRow) isOurProcessRow_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_dot_product_and_assign_float_c(v_row_dev, l_rows, isOurProcessRow_int, aux1_dev, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_set_e_vec_scale_set_one_store_v_row_float(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, &
                                            xf_host_or_dev, l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                            wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows, l_cols, matrixRows, istep
      logical, intent(in)             :: isOurProcessRow, useCCL, wantDebug
      integer(kind=c_intptr_t)        :: e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isOurProcessRow_int, useCCL_int, wantDebug_int

      isOurProcessRow_int = 0
      useCCL_int = 0
      wantDebug_int = 0
      if (isOurProcessRow) isOurProcessRow_int = 1
      if (useCCL) useCCL_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_set_e_vec_scale_set_one_store_v_row_float_c(e_vec_dev, vrl_dev, a_dev, &
                                       v_row_dev, tau_dev, xf_host_or_dev, &
                                       l_rows, l_cols, matrixRows, istep, isOurProcessRow_int, useCCL_int, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_store_u_v_in_uv_vu_float(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                            v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                            vav_host_or_dev, tau_istep_host_or_dev, &
                                            l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                            useCCL, wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep
      integer(kind=c_intptr_t)        :: vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev
      integer(kind=c_intptr_t)        :: v_col_dev, u_col_dev, tau_dev, aux_complex_dev
      integer(kind=c_intptr_t)        :: vav_host_or_dev, tau_istep_host_or_dev
      logical, intent(in)             :: useCCL, wantDebug
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: useCCL_int, wantDebug_int

      useCCL_int = 0
      wantDebug_int = 0
      if (useCCL) useCCL_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_store_u_v_in_uv_vu_float_c(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                       v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                       vav_host_or_dev, tau_istep_host_or_dev, &
                                       l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                       useCCL_int, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_update_matrix_element_add_float(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev,  &
                                            l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                            isSkewsymmetric, wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows, l_cols, matrixRows, max_local_rows, max_local_cols
      integer(kind=c_int), intent(in) :: istep, n_stored_vecs
      integer(kind=c_intptr_t)        :: vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev
      logical, intent(in)             :: isSkewsymmetric, wantDebug
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isSkewsymmetric_int, wantDebug_int

      isSkewsymmetric_int = 0
      wantDebug_int = 0
      if (isSkewsymmetric) isSkewsymmetric_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_update_matrix_element_add_float_c(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev,  &
                                       l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                       isSkewsymmetric_int, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_update_array_element_float(array_dev, index, value, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: index
      integer(kind=c_intptr_t)        :: array_dev
      integer(kind=c_intptr_t)        :: value
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_AMD_GPU_VERSION
      call hip_update_array_element_float_c(array_dev, index, value, my_stream)
#endif

    end subroutine

    subroutine hip_hh_transform_float(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_intptr_t)        :: alpha_dev, xnorm_sq_dev, xf_dev, tau_dev
      logical, intent(in)             :: wantDebug
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: wantDebug_int

      wantDebug_int = 0
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_hh_transform_float_c(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_transpose_reduceadd_vectors_copy_block_float(aux_transpose_dev, vmat_st_dev, &
                                              nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                              lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                              isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride
      integer(kind=c_int), intent(in) :: np_st, ld_st, direction, sm_count
      integer(kind=c_intptr_t)        :: aux_transpose_dev, vmat_st_dev
      logical, intent(in)             :: isSkewsymmetric, isReduceadd, wantDebug
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isSkewsymmetric_int, isReduceadd_int, wantDebug_int

      isSkewsymmetric_int = 0
      isReduceadd_int = 0
      wantDebug_int = 0
      if (isSkewsymmetric) isSkewsymmetric_int = 1
      if (isReduceadd) isReduceadd_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_transpose_reduceadd_vectors_copy_block_float_c(aux_transpose_dev, vmat_st_dev, &
                                          nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                          lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                          isSkewsymmetric_int, isReduceadd_int, wantDebug_int, sm_count, my_stream)
#endif

    end subroutine
    subroutine hip_copy_and_set_zeros_double_complex(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep,  &
                                                 aux1_dev, vav_dev, d_vec_dev, &
                                                 isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, &
                                                 isSkewsymmetric, useCCL, wantDebug, my_stream) 
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows, l_cols, matrixRows, istep
      logical, intent(in)             :: isOurProcessRow, isOurProcessCol, isOurProcessCol_prev
      logical, intent(in)             :: isSkewsymmetric, useCCL, wantDebug
      integer(kind=c_intptr_t)        :: v_row_dev, a_dev, aux1_dev, vav_dev, d_vec_dev
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int
      integer(kind=c_int)             :: isSkewsymmetric_int, useCCL_int, wantDebug_int

      isOurProcessRow_int = 0
      isOurProcessCol_int = 0
      isOurProcessCol_prev_int = 0
      isSkewsymmetric_int = 0
      useCCL_int = 0
      wantDebug_int = 0
      if (isOurProcessRow) isOurProcessRow_int = 1
      if (isOurProcessCol) isOurProcessCol_int = 1
      if (isOurProcessCol_prev) isOurProcessCol_prev_int = 1
      if (isSkewsymmetric) isSkewsymmetric_int = 1
      if (useCCL) useCCL_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_and_set_zeros_double_complex_c(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                            aux1_dev, vav_dev, d_vec_dev, &
                                            isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int, &
                                            isSkewsymmetric_int, useCCL_int, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_dot_product_double_complex(n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: n, incx, incy, sm_count
      logical, intent(in)             :: wantDebug
      integer(kind=C_intptr_T)        :: x_dev, y_dev, result_dev
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: wantDebug_int

      wantDebug_int = 0
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_dot_product_double_complex_c(n, x_dev, incx, y_dev, incy, result_dev, wantDebug_int, sm_count, my_stream)
#endif

    end subroutine

    subroutine hip_dot_product_and_assign_double_complex(v_row_dev, l_rows, isOurProcessRow, aux1_dev, &
                                                                     wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows
      logical, intent(in)             :: isOurProcessRow, wantDebug
      integer(kind=c_intptr_t)        :: v_row_dev, aux1_dev
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isOurProcessRow_int, wantDebug_int

      isOurProcessRow_int = 0
      wantDebug_int = 0
      if (isOurProcessRow) isOurProcessRow_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_dot_product_and_assign_double_complex_c(v_row_dev, l_rows, isOurProcessRow_int, aux1_dev, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_set_e_vec_scale_set_one_store_v_row_double_complex(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, &
                                            xf_host_or_dev, l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                            wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows, l_cols, matrixRows, istep
      logical, intent(in)             :: isOurProcessRow, useCCL, wantDebug
      integer(kind=c_intptr_t)        :: e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isOurProcessRow_int, useCCL_int, wantDebug_int

      isOurProcessRow_int = 0
      useCCL_int = 0
      wantDebug_int = 0
      if (isOurProcessRow) isOurProcessRow_int = 1
      if (useCCL) useCCL_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_set_e_vec_scale_set_one_store_v_row_double_complex_c(e_vec_dev, vrl_dev, a_dev, &
                                       v_row_dev, tau_dev, xf_host_or_dev, &
                                       l_rows, l_cols, matrixRows, istep, isOurProcessRow_int, useCCL_int, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_store_u_v_in_uv_vu_double_complex(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                            v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                            vav_host_or_dev, tau_istep_host_or_dev, &
                                            l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                            useCCL, wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep
      integer(kind=c_intptr_t)        :: vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev
      integer(kind=c_intptr_t)        :: v_col_dev, u_col_dev, tau_dev, aux_complex_dev
      integer(kind=c_intptr_t)        :: vav_host_or_dev, tau_istep_host_or_dev
      logical, intent(in)             :: useCCL, wantDebug
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: useCCL_int, wantDebug_int

      useCCL_int = 0
      wantDebug_int = 0
      if (useCCL) useCCL_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_store_u_v_in_uv_vu_double_complex_c(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                       v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                       vav_host_or_dev, tau_istep_host_or_dev, &
                                       l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                       useCCL_int, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_update_matrix_element_add_double_complex(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev,  &
                                            l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                            isSkewsymmetric, wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows, l_cols, matrixRows, max_local_rows, max_local_cols
      integer(kind=c_int), intent(in) :: istep, n_stored_vecs
      integer(kind=c_intptr_t)        :: vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev
      logical, intent(in)             :: isSkewsymmetric, wantDebug
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isSkewsymmetric_int, wantDebug_int

      isSkewsymmetric_int = 0
      wantDebug_int = 0
      if (isSkewsymmetric) isSkewsymmetric_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_update_matrix_element_add_double_complex_c(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev,  &
                                       l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                       isSkewsymmetric_int, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_update_array_element_double_complex(array_dev, index, value, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: index
      integer(kind=c_intptr_t)        :: array_dev
      integer(kind=c_intptr_t)        :: value
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_AMD_GPU_VERSION
      call hip_update_array_element_double_complex_c(array_dev, index, value, my_stream)
#endif

    end subroutine

    subroutine hip_hh_transform_double_complex(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_intptr_t)        :: alpha_dev, xnorm_sq_dev, xf_dev, tau_dev
      logical, intent(in)             :: wantDebug
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: wantDebug_int

      wantDebug_int = 0
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_hh_transform_double_complex_c(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_transpose_reduceadd_vectors_copy_block_double_complex(aux_transpose_dev, vmat_st_dev, &
                                              nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                              lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                              isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride
      integer(kind=c_int), intent(in) :: np_st, ld_st, direction, sm_count
      integer(kind=c_intptr_t)        :: aux_transpose_dev, vmat_st_dev
      logical, intent(in)             :: isSkewsymmetric, isReduceadd, wantDebug
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isSkewsymmetric_int, isReduceadd_int, wantDebug_int

      isSkewsymmetric_int = 0
      isReduceadd_int = 0
      wantDebug_int = 0
      if (isSkewsymmetric) isSkewsymmetric_int = 1
      if (isReduceadd) isReduceadd_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_transpose_reduceadd_vectors_copy_block_double_complex_c(aux_transpose_dev, vmat_st_dev, &
                                          nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                          lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                          isSkewsymmetric_int, isReduceadd_int, wantDebug_int, sm_count, my_stream)
#endif

    end subroutine
    subroutine hip_copy_and_set_zeros_float_complex(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep,  &
                                                 aux1_dev, vav_dev, d_vec_dev, &
                                                 isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, &
                                                 isSkewsymmetric, useCCL, wantDebug, my_stream) 
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows, l_cols, matrixRows, istep
      logical, intent(in)             :: isOurProcessRow, isOurProcessCol, isOurProcessCol_prev
      logical, intent(in)             :: isSkewsymmetric, useCCL, wantDebug
      integer(kind=c_intptr_t)        :: v_row_dev, a_dev, aux1_dev, vav_dev, d_vec_dev
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int
      integer(kind=c_int)             :: isSkewsymmetric_int, useCCL_int, wantDebug_int

      isOurProcessRow_int = 0
      isOurProcessCol_int = 0
      isOurProcessCol_prev_int = 0
      isSkewsymmetric_int = 0
      useCCL_int = 0
      wantDebug_int = 0
      if (isOurProcessRow) isOurProcessRow_int = 1
      if (isOurProcessCol) isOurProcessCol_int = 1
      if (isOurProcessCol_prev) isOurProcessCol_prev_int = 1
      if (isSkewsymmetric) isSkewsymmetric_int = 1
      if (useCCL) useCCL_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_and_set_zeros_float_complex_c(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                            aux1_dev, vav_dev, d_vec_dev, &
                                            isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int, &
                                            isSkewsymmetric_int, useCCL_int, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_dot_product_float_complex(n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: n, incx, incy, sm_count
      logical, intent(in)             :: wantDebug
      integer(kind=C_intptr_T)        :: x_dev, y_dev, result_dev
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: wantDebug_int

      wantDebug_int = 0
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_dot_product_float_complex_c(n, x_dev, incx, y_dev, incy, result_dev, wantDebug_int, sm_count, my_stream)
#endif

    end subroutine

    subroutine hip_dot_product_and_assign_float_complex(v_row_dev, l_rows, isOurProcessRow, aux1_dev, &
                                                                     wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows
      logical, intent(in)             :: isOurProcessRow, wantDebug
      integer(kind=c_intptr_t)        :: v_row_dev, aux1_dev
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isOurProcessRow_int, wantDebug_int

      isOurProcessRow_int = 0
      wantDebug_int = 0
      if (isOurProcessRow) isOurProcessRow_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_dot_product_and_assign_float_complex_c(v_row_dev, l_rows, isOurProcessRow_int, aux1_dev, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_set_e_vec_scale_set_one_store_v_row_float_complex(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, &
                                            xf_host_or_dev, l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                            wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows, l_cols, matrixRows, istep
      logical, intent(in)             :: isOurProcessRow, useCCL, wantDebug
      integer(kind=c_intptr_t)        :: e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isOurProcessRow_int, useCCL_int, wantDebug_int

      isOurProcessRow_int = 0
      useCCL_int = 0
      wantDebug_int = 0
      if (isOurProcessRow) isOurProcessRow_int = 1
      if (useCCL) useCCL_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_set_e_vec_scale_set_one_store_v_row_float_complex_c(e_vec_dev, vrl_dev, a_dev, &
                                       v_row_dev, tau_dev, xf_host_or_dev, &
                                       l_rows, l_cols, matrixRows, istep, isOurProcessRow_int, useCCL_int, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_store_u_v_in_uv_vu_float_complex(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                            v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                            vav_host_or_dev, tau_istep_host_or_dev, &
                                            l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                            useCCL, wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep
      integer(kind=c_intptr_t)        :: vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev
      integer(kind=c_intptr_t)        :: v_col_dev, u_col_dev, tau_dev, aux_complex_dev
      integer(kind=c_intptr_t)        :: vav_host_or_dev, tau_istep_host_or_dev
      logical, intent(in)             :: useCCL, wantDebug
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: useCCL_int, wantDebug_int

      useCCL_int = 0
      wantDebug_int = 0
      if (useCCL) useCCL_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_store_u_v_in_uv_vu_float_complex_c(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                       v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                       vav_host_or_dev, tau_istep_host_or_dev, &
                                       l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                       useCCL_int, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_update_matrix_element_add_float_complex(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev,  &
                                            l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                            isSkewsymmetric, wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: l_rows, l_cols, matrixRows, max_local_rows, max_local_cols
      integer(kind=c_int), intent(in) :: istep, n_stored_vecs
      integer(kind=c_intptr_t)        :: vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev
      logical, intent(in)             :: isSkewsymmetric, wantDebug
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isSkewsymmetric_int, wantDebug_int

      isSkewsymmetric_int = 0
      wantDebug_int = 0
      if (isSkewsymmetric) isSkewsymmetric_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_update_matrix_element_add_float_complex_c(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev,  &
                                       l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                       isSkewsymmetric_int, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_update_array_element_float_complex(array_dev, index, value, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: index
      integer(kind=c_intptr_t)        :: array_dev
      integer(kind=c_intptr_t)        :: value
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_AMD_GPU_VERSION
      call hip_update_array_element_float_complex_c(array_dev, index, value, my_stream)
#endif

    end subroutine

    subroutine hip_hh_transform_float_complex(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_intptr_t)        :: alpha_dev, xnorm_sq_dev, xf_dev, tau_dev
      logical, intent(in)             :: wantDebug
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: wantDebug_int

      wantDebug_int = 0
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_hh_transform_float_complex_c(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug_int, my_stream)
#endif

    end subroutine

    subroutine hip_transpose_reduceadd_vectors_copy_block_float_complex(aux_transpose_dev, vmat_st_dev, &
                                              nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                              lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                              isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in) :: nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride
      integer(kind=c_int), intent(in) :: np_st, ld_st, direction, sm_count
      integer(kind=c_intptr_t)        :: aux_transpose_dev, vmat_st_dev
      logical, intent(in)             :: isSkewsymmetric, isReduceadd, wantDebug
      integer(kind=c_intptr_t)        :: my_stream
      integer(kind=c_int)             :: isSkewsymmetric_int, isReduceadd_int, wantDebug_int

      isSkewsymmetric_int = 0
      isReduceadd_int = 0
      wantDebug_int = 0
      if (isSkewsymmetric) isSkewsymmetric_int = 1
      if (isReduceadd) isReduceadd_int = 1
      if (wantDebug) wantDebug_int = 1

#ifdef WITH_AMD_GPU_VERSION
      call hip_transpose_reduceadd_vectors_copy_block_float_complex_c(aux_transpose_dev, vmat_st_dev, &
                                          nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                          lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                          isSkewsymmetric_int, isReduceadd_int, wantDebug_int, sm_count, my_stream)
#endif

    end subroutine
end module
