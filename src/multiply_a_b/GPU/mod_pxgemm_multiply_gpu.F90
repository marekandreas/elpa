#if 0
!    Copyright 2025, P. Karpov, MPCDF
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
!      Schwerpunkt Wissenschaftliches Rechnen,
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


module pxgemm_multiply_gpu
  use, intrinsic :: iso_c_binding
  use precision
  implicit none

  public

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)

  interface
    subroutine gpu_copy_aux_full_c(dataType, lhs_dev, rhs_dev, l_rows, l_cols, lld_lhs, lld_rhs, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_copy_aux_full_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_copy_aux_full_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_copy_aux_full_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: lhs_dev, rhs_dev
      integer(kind=c_int), value         :: l_rows, l_cols, lld_lhs, lld_rhs, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_copy_and_set_zeros_aux_full_c (dataType, mat_dev, aux_mat_full_dev, &
                                                  l_rows, l_cols, nblk_mult, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_copy_and_set_zeros_aux_full_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_copy_and_set_zeros_aux_full_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_copy_and_set_zeros_aux_full_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: mat_dev, aux_mat_full_dev
      integer(kind=c_int), value         :: l_rows, l_cols, nblk_mult, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_copy_and_set_zeros_aux_a_full_c (dataType, mat_dev, aux_mat_full_dev, l_rows, l_cols, &
                                                    nblk_mult_cols, nblk, np_bc_fine, np_cols_fine, np_cols, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_copy_and_set_zeros_aux_a_full_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_copy_and_set_zeros_aux_a_full_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_copy_and_set_zeros_aux_a_full_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: mat_dev, aux_mat_full_dev
      integer(kind=c_int), value         :: l_rows, l_cols, nblk_mult_cols, nblk, np_bc_fine, np_cols_fine, np_cols, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_copy_and_set_zeros_aux_b_full_c (dataType, mat_dev, aux_mat_full_dev, l_rows, l_cols, nblk_mult, &
                                                    nblk_mult_rows, nblk, np_fine, np_rows_fine, np_rows, &
                                                    SM_count, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_copy_and_set_zeros_aux_b_full_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_copy_and_set_zeros_aux_b_full_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_copy_and_set_zeros_aux_b_full_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: mat_dev, aux_mat_full_dev
      integer(kind=c_int), value         :: l_rows, l_cols, nblk_mult, nblk_mult_rows, nblk, np_fine, np_rows_fine, np_rows, &
                                            SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface 


  interface
    subroutine gpu_ccl_copy_buf_send_c (dataType, a_dev, buf_send_dev, l_rows, l_cols, lld_buf, &
                                        nblk, i_block_loc_fine_max, j_block_loc_fine_max, np_fine, np_bc_fine, &
                                        np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_ccl_copy_buf_send_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_ccl_copy_buf_send_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_ccl_copy_buf_send_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: a_dev, buf_send_dev
      integer(kind=c_int), value         :: l_rows, l_cols, lld_buf, nblk, i_block_loc_fine_max, j_block_loc_fine_max, &
                                            np_fine, np_bc_fine, np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_ccl_copy_buf_recv_c (dataType, at_col_dev, buf_recv_dev, l_rows, l_cols, lld_buf, &
                                        nblk, i_block_loc_fine_max, j_block_loc_fine_max, np_fine, np_bc_fine, &
                                        np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_ccl_copy_buf_recv_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_ccl_copy_buf_recv_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_ccl_copy_buf_recv_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: at_col_dev, buf_recv_dev
      integer(kind=c_int), value         :: l_rows, l_cols, lld_buf, nblk, i_block_loc_fine_max, j_block_loc_fine_max, &
                                            np_fine, np_bc_fine, np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_copy_and_set_zeros_aux_ab_full_tn_nt_c (dataType,  a_transposed, &
                                                        a_dev, b_dev, aux_a_full_dev, aux_b_full_dev, &
                                                        l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                                        np_ab_fine, np_rows, my_prow, &
                                                        np_t_fine , np_cols, my_pcol, &
                                                        np_dirs_fine, SM_count, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_copy_and_set_zeros_aux_ab_full_tn_nt_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_copy_and_set_zeros_aux_ab_full_tn_nt_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_copy_and_set_zeros_aux_ab_full_tn_nt_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: a_dev, b_dev, aux_a_full_dev, aux_b_full_dev
      integer(kind=c_int), value         :: a_transposed, l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                            np_ab_fine, np_rows, my_prow, &
                                            np_t_fine, np_cols, my_pcol, &
                                            np_dirs_fine, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_update_c_tn_nt_c(dataType, a_transposed, &
                                    c_dev, tmp1_full_dev, beta_int, &
                                    l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                    np_rows, np_cols, np_dirs_fine, &
                                    np_dirs_t, my_pdir_t, np_fine, &
                                    SM_count, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_update_c_tn_nt_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_update_c_tn_nt_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_update_c_tn_nt_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: c_dev, tmp1_full_dev
      integer(kind=c_int), value         :: a_transposed, beta_int, l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                            np_rows, np_cols, np_dirs_fine, &
                                            np_dirs_t, my_pdir_t, np_fine, &
                                            SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface

#endif /* defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION) */


  contains


    subroutine gpu_copy_aux_full(dataType, lhs_dev, rhs_dev, l_rows, l_cols, lld_lhs, lld_rhs, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: lhs_dev, rhs_dev
      integer(kind=c_int), value         :: l_rows, l_cols, lld_lhs, lld_rhs, debug
      integer(kind=c_intptr_t), value    :: my_stream
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_copy_aux_full_c(dataType, lhs_dev, rhs_dev, l_rows, l_cols, lld_lhs, lld_rhs, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_copy_and_set_zeros_aux_full (dataType, mat_dev, aux_mat_full_dev, &
                                                l_rows, l_cols, nblk_mult, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: mat_dev, aux_mat_full_dev
      integer(kind=c_int), value         :: l_rows, l_cols, nblk_mult, debug
      integer(kind=c_intptr_t), value    :: my_stream
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_copy_and_set_zeros_aux_full_c (dataType, mat_dev, aux_mat_full_dev, &
                                              l_rows, l_cols, nblk_mult, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_copy_and_set_zeros_aux_a_full (dataType, mat_dev, aux_mat_full_dev, l_rows, l_cols, &
                                                  nblk_mult_cols, nblk, np_bc_fine, np_cols_fine, np_cols, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: mat_dev, aux_mat_full_dev
      integer(kind=c_int), value         :: l_rows, l_cols, nblk_mult_cols, nblk, np_bc_fine, np_cols_fine, np_cols, debug
      integer(kind=c_intptr_t), value    :: my_stream
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_copy_and_set_zeros_aux_a_full_c (dataType, mat_dev, aux_mat_full_dev, l_rows, l_cols, &
                                                nblk_mult_cols, nblk, np_bc_fine, np_cols_fine, np_cols, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_copy_and_set_zeros_aux_b_full (dataType, mat_dev, aux_mat_full_dev, l_rows, l_cols, nblk_mult, &
                                                  nblk_mult_rows, nblk, np_fine, np_rows_fine, np_rows, &
                                                  SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: mat_dev, aux_mat_full_dev
      integer(kind=c_int), value         :: l_rows, l_cols, nblk_mult, nblk_mult_rows, nblk, np_fine, np_rows_fine, np_rows, &
                                            SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_copy_and_set_zeros_aux_b_full_c (dataType, mat_dev, aux_mat_full_dev, l_rows, l_cols, nblk_mult, &
                                                nblk_mult_rows, nblk, np_fine, np_rows_fine, np_rows, &
                                                SM_count, debug, my_stream)
#endif
    end subroutine

    subroutine gpu_ccl_copy_buf_send (dataType, a_dev, buf_send_dev, l_rows, l_cols, lld_buf, &
                                      nblk, i_block_loc_fine_max, j_block_loc_fine_max, np_fine, np_bc_fine, &
                                      np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: a_dev, buf_send_dev
      integer(kind=c_int), value         :: l_rows, l_cols, lld_buf, nblk, i_block_loc_fine_max, j_block_loc_fine_max, &
                                            np_fine, np_bc_fine, np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_ccl_copy_buf_send_c (dataType, a_dev, buf_send_dev, l_rows, l_cols, lld_buf, &
                                    nblk, i_block_loc_fine_max, j_block_loc_fine_max, np_fine, np_bc_fine, &
                                    np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream)
#endif
    end subroutine

    subroutine gpu_ccl_copy_buf_recv (dataType, at_col_dev, buf_recv_dev, l_rows, l_cols, lld_buf, &
                                      nblk, i_block_loc_fine_max, j_block_loc_fine_max, np_fine, np_bc_fine, &
                                      np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: at_col_dev, buf_recv_dev
      integer(kind=c_int), value         :: l_rows, l_cols, lld_buf, nblk, i_block_loc_fine_max, j_block_loc_fine_max, &
                                            np_fine, np_bc_fine, np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_ccl_copy_buf_recv_c (dataType, at_col_dev, buf_recv_dev, l_rows, l_cols, lld_buf, &
                                    nblk, i_block_loc_fine_max, j_block_loc_fine_max, np_fine, np_bc_fine, &
                                    np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_copy_and_set_zeros_aux_ab_full_tn_nt (dataType,  a_transposed, &
                                                        a_dev, b_dev, aux_a_full_dev, aux_b_full_dev, &
                                                        l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                                        np_ab_fine, np_rows, my_prow, &
                                                        np_t_fine , np_cols, my_pcol, &
                                                        np_dirs_fine, SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: a_dev, b_dev, aux_a_full_dev, aux_b_full_dev
      integer(kind=c_int), value         :: a_transposed, l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                            np_ab_fine, np_rows, my_prow, &
                                            np_t_fine, np_cols, my_pcol, &
                                            np_dirs_fine, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_copy_and_set_zeros_aux_ab_full_tn_nt_c(dataType, a_transposed, &
                                                      a_dev, b_dev, aux_a_full_dev, aux_b_full_dev, &
                                                      l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                                      np_ab_fine, np_rows, my_prow, &
                                                      np_t_fine , np_cols, my_pcol, &
                                                      np_dirs_fine, SM_count, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_update_c_tn_nt(dataType, a_transposed, &
                                  c_dev, tmp1_full_dev, beta_int, &
                                  l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                  np_rows, np_cols, np_dirs_fine, &
                                  np_dirs_t, my_pdir_t, np_fine, &
                                  SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: c_dev, tmp1_full_dev
      integer(kind=c_int), value         :: a_transposed, beta_int, l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                            np_rows, np_cols, np_dirs_fine, &
                                            np_dirs_t, my_pdir_t, np_fine, &
                                            SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_update_c_tn_nt_c(dataType, a_transposed, &
                                c_dev, tmp1_full_dev, beta_int, &
                                l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                np_rows, np_cols, np_dirs_fine, &
                                np_dirs_t, my_pdir_t, np_fine, &
                                SM_count, debug, my_stream)
#endif
    end subroutine

end module
