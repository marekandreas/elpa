#if 0
!    Copyright 2023-2025, P. Karpov, MPCDF
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


module tridiag_gpu
  use, intrinsic :: iso_c_binding
  use precision

  implicit none

  public

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)

  interface
    subroutine gpu_copy_and_set_zeros_c(dataType, v_row_dev, u_col_dev, a_dev, aux1_dev, vav_dev, d_vec_dev, &
                                        l_rows, l_cols, matrixRows, istep, &
                                        isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int, &
                                        isSkewsymmetric_int, useCCL_int, wantDebug_int, SM_count, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_copy_and_set_zeros_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name="hip_copy_and_set_zeros_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_copy_and_set_zeros_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value      :: dataType
      integer(kind=c_intptr_t), value  :: v_row_dev, u_col_dev, a_dev, aux1_dev, vav_dev, d_vec_dev
      integer(kind=c_int), value       :: l_rows, l_cols, matrixRows, istep, SM_count
      integer(kind=c_int), value       :: isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int
      integer(kind=c_int), value       :: isSkewsymmetric_int, useCCL_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_dot_product_c(dataType, n, x_dev, incx, y_dev, incy, result_dev, wantDebug_int, sm_count, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_dot_product_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name="hip_dot_product_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_dot_product_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_intptr_t), value :: x_dev, y_dev, result_dev
      integer(kind=c_int), value      :: n, incx, incy, sm_count
      integer(kind=c_int), value      :: wantDebug_int
      integer(kind=c_intptr_t), value :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_dot_product_and_assign_c(dataType, v_row_dev, l_rows, isOurProcessRow_int, aux1_dev, &
                                            wantDebug_int, SM_count, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_dot_product_and_assign_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name="hip_dot_product_and_assign_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_dot_product_and_assign_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value       :: dataType
      integer(kind=c_intptr_t), value   :: v_row_dev, aux1_dev
      integer(kind=c_int), value        :: l_rows
      integer(kind=c_int), value        :: isOurProcessRow_int, wantDebug_int, SM_count
      integer(kind=c_intptr_t), value   :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_set_e_vec_scale_set_one_store_v_row_c (dataType, e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, &
                                                          l_rows, l_cols, matrixRows, istep, isOurProcessRow_int, useCCL_int, &
                                                          wantDebug_int, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                      bind(C, name="cuda_set_e_vec_scale_set_one_store_v_row_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                      bind(C, name="hip_set_e_vec_scale_set_one_store_v_row_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                      bind(C, name="sycl_set_e_vec_scale_set_one_store_v_row_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value      :: dataType
      integer(kind=c_intptr_t), value  :: e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev
      integer(kind=c_int), value       :: l_rows, l_cols, matrixRows, istep
      integer(kind=c_int), value       :: isOurProcessRow_int, useCCL_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_store_u_v_in_uv_vu_c(dataType, vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                        v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                        vav_host_or_dev, tau_istep_host_or_dev, &
                                        l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                        useCCL_int, wantDebug_int, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                        bind(C, name="cuda_store_u_v_in_uv_vu_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                        bind(C, name="hip_store_u_v_in_uv_vu_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                        bind(C, name="sycl_store_u_v_in_uv_vu_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value      :: dataType
      integer(kind=c_intptr_t), value  :: vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev
      integer(kind=c_intptr_t), value  :: v_col_dev, u_col_dev, tau_dev, aux_complex_dev
      integer(kind=c_intptr_t), value  :: vav_host_or_dev, tau_istep_host_or_dev
      integer(kind=c_int), value       :: l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep
      integer(kind=c_int), value       :: useCCL_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_update_matrix_element_add_c (dataType, vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                                l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                                isSkewsymmetric_int, wantDebug_int, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                              bind(C, name="cuda_update_matrix_element_add_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                              bind(C, name="hip_update_matrix_element_add_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                              bind(C, name="sycl_update_matrix_element_add_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value      :: dataType
      integer(kind=c_intptr_t), value  :: vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev
      integer(kind=c_int), value       :: l_rows, l_cols, matrixRows, max_local_rows, max_local_cols
      integer(kind=c_int), value       :: istep, n_stored_vecs
      integer(kind=c_int), value       :: isSkewsymmetric_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_hh_transform_c(dataType, alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug_int, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                  bind(C, name="cuda_hh_transform_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                  bind(C, name="hip_hh_transform_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                  bind(C, name="sycl_hh_transform_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value      :: dataType
      integer(kind=c_intptr_t), value  :: alpha_dev, xnorm_sq_dev, xf_dev, tau_dev
      integer(kind=c_int), value       :: wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_transpose_reduceadd_vectors_copy_block_c(dataType, aux_transpose_dev, vmat_st_dev, &
                                                  nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                                  lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                                  isSkewsymmetric_int, isReduceadd_int, wantDebug_int, sm_count, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_transpose_reduceadd_vectors_copy_block_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name="hip_transpose_reduceadd_vectors_copy_block_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_transpose_reduceadd_vectors_copy_block_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value      :: dataType
      integer(kind=c_intptr_t), value  :: aux_transpose_dev, vmat_st_dev
      integer(kind=c_int), value       :: nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride
      integer(kind=c_int), value       :: np_st, ld_st, direction, sm_count
      integer(kind=c_int), value       :: isSkewsymmetric_int, isReduceadd_int, wantDebug_int
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

#endif /* defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION) */


  contains


    subroutine gpu_copy_and_set_zeros(dataType, v_row_dev, u_col_dev, a_dev, aux1_dev, vav_dev, d_vec_dev, &
                                      l_rows, l_cols, matrixRows, istep, &
                                      isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, &
                                      isSkewsymmetric, useCCL, wantDebug, SM_count, my_stream) 
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_intptr_t), value :: v_row_dev, u_col_dev, a_dev, aux1_dev, vav_dev, d_vec_dev
      integer(kind=c_intptr_t), value :: my_stream
      integer(kind=c_int), value      :: l_rows, l_cols, matrixRows, istep, SM_count
      logical, intent(in)             :: isOurProcessRow, isOurProcessCol, isOurProcessCol_prev
      logical, intent(in)             :: isSkewsymmetric, useCCL, wantDebug
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

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_copy_and_set_zeros_c(dataType, v_row_dev, u_col_dev, a_dev, aux1_dev, vav_dev, d_vec_dev, &
                                    l_rows, l_cols, matrixRows, istep, &
                                    isOurProcessRow_int, isOurProcessCol_int, isOurProcessCol_prev_int, &
                                    isSkewsymmetric_int, useCCL_int, wantDebug_int, SM_count, my_stream)
#endif
    end subroutine


    subroutine gpu_dot_product(dataType, n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_int),  value     :: n, incx, incy, sm_count
      integer(kind=c_intptr_t), value :: x_dev, y_dev, result_dev
      logical, intent(in)             :: wantDebug
      integer(kind=c_int)             :: wantDebug_int
      integer(kind=c_intptr_t), value :: my_stream

      wantDebug_int = 0
      if (wantDebug) wantDebug_int = 1

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_dot_product_c(dataType, n, x_dev, incx, y_dev, incy, result_dev, wantDebug_int, sm_count, my_stream)
#endif
    end subroutine


    subroutine gpu_dot_product_and_assign(dataType, v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, SM_count, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_int), value      :: l_rows
      logical, intent(in)             :: isOurProcessRow, wantDebug
      integer(kind=c_intptr_t), value :: v_row_dev, aux1_dev
      integer(kind=c_intptr_t), value :: my_stream
      integer(kind=c_int)             :: isOurProcessRow_int, wantDebug_int, SM_count

      isOurProcessRow_int = 0
      wantDebug_int = 0
      if (isOurProcessRow) isOurProcessRow_int = 1
      if (wantDebug) wantDebug_int = 1

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_dot_product_and_assign_c(dataType, v_row_dev, l_rows, isOurProcessRow_int, aux1_dev, &
                                        wantDebug_int, SM_count, my_stream)
#endif
    end subroutine


    subroutine gpu_set_e_vec_scale_set_one_store_v_row (dataType, e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, &
                                                        l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                                        wantDebug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_int), value      :: l_rows, l_cols, matrixRows, istep
      logical, intent(in)             :: isOurProcessRow, useCCL, wantDebug
      integer(kind=c_intptr_t), value :: e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev
      integer(kind=c_intptr_t), value :: my_stream
      integer(kind=c_int)             :: isOurProcessRow_int, useCCL_int, wantDebug_int

      isOurProcessRow_int = 0
      useCCL_int = 0
      wantDebug_int = 0
      if (isOurProcessRow) isOurProcessRow_int = 1
      if (useCCL) useCCL_int = 1
      if (wantDebug) wantDebug_int = 1

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_set_e_vec_scale_set_one_store_v_row_c (dataType, e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, &
                                                      l_rows, l_cols, matrixRows, istep, &
                                                      isOurProcessRow_int, useCCL_int, wantDebug_int, my_stream)
#endif
    end subroutine


    subroutine gpu_store_u_v_in_uv_vu(dataType, vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                      v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                      vav_host_or_dev, tau_istep_host_or_dev, &
                                      l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                      useCCL, wantDebug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_int), value      :: l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep
      integer(kind=c_intptr_t), value :: vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev
      integer(kind=c_intptr_t), value :: v_col_dev, u_col_dev, tau_dev, aux_complex_dev
      integer(kind=c_intptr_t), value :: vav_host_or_dev, tau_istep_host_or_dev
      logical, intent(in)             :: useCCL, wantDebug
      integer(kind=c_intptr_t), value :: my_stream
      integer(kind=c_int)             :: useCCL_int, wantDebug_int

      useCCL_int = 0
      wantDebug_int = 0
      if (useCCL) useCCL_int = 1
      if (wantDebug) wantDebug_int = 1

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_store_u_v_in_uv_vu_c(dataType, vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                    v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                    vav_host_or_dev, tau_istep_host_or_dev, &
                                    l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                    useCCL_int, wantDebug_int, my_stream)
#endif
    end subroutine


    subroutine gpu_update_matrix_element_add (dataType, vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev,  &
                                              l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                              isSkewsymmetric, wantDebug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_int), value      :: l_rows, l_cols, matrixRows, max_local_rows, max_local_cols
      integer(kind=c_int), value      :: istep, n_stored_vecs
      integer(kind=c_intptr_t), value :: vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev
      logical, intent(in)             :: isSkewsymmetric, wantDebug
      integer(kind=c_intptr_t), value :: my_stream
      integer(kind=c_int)             :: isSkewsymmetric_int, wantDebug_int

      isSkewsymmetric_int = 0
      wantDebug_int = 0
      if (isSkewsymmetric) isSkewsymmetric_int = 1
      if (wantDebug) wantDebug_int = 1

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_update_matrix_element_add_c (dataType, vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev,  &
                                            l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                            isSkewsymmetric_int, wantDebug_int, my_stream)
#endif
    end subroutine


    subroutine gpu_hh_transform(dataType, alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_intptr_t), value :: alpha_dev, xnorm_sq_dev, xf_dev, tau_dev
      logical, intent(in)             :: wantDebug
      integer(kind=c_intptr_t), value :: my_stream
      integer(kind=c_int)             :: wantDebug_int

      wantDebug_int = 0
      if (wantDebug) wantDebug_int = 1

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_hh_transform_c(dataType, alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug_int, my_stream)
#endif
    end subroutine


    subroutine gpu_transpose_reduceadd_vectors_copy_block(dataType, aux_transpose_dev, vmat_st_dev, &
                                              nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                              lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                              isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_int), value      :: nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride
      integer(kind=c_int), value      :: np_st, ld_st, direction, sm_count
      integer(kind=c_intptr_t), value :: aux_transpose_dev, vmat_st_dev
      logical, intent(in)             :: isSkewsymmetric, isReduceadd, wantDebug
      integer(kind=c_intptr_t), value :: my_stream
      integer(kind=c_int)             :: isSkewsymmetric_int, isReduceadd_int, wantDebug_int

      isSkewsymmetric_int = 0
      isReduceadd_int = 0
      wantDebug_int = 0
      if (isSkewsymmetric) isSkewsymmetric_int = 1
      if (isReduceadd) isReduceadd_int = 1
      if (wantDebug) wantDebug_int = 1

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_transpose_reduceadd_vectors_copy_block_c(dataType, aux_transpose_dev, vmat_st_dev, &
                                          nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                          lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                          isSkewsymmetric_int, isReduceadd_int, wantDebug_int, sm_count, my_stream)
#endif
    end subroutine

  end module
