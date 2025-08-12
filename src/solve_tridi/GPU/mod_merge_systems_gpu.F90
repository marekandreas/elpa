#if 0
!    Copyright 2024, A. Marek, MPCDF
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

!    This file was written by A. Marek, MPCDF
#endif


#include "config-f90.h"


module merge_systems_gpu_new
  use, intrinsic :: iso_c_binding
  use precision
  implicit none

  public

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)

  interface
    subroutine gpu_update_ndef_c_c (ndef_c_dev, idx_dev, p_col_dev, idx2_dev, &
                                    na, na1, np_rem, ndef, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_update_ndef_c_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_update_ndef_c_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_update_ndef_c_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value    :: ndef_c_dev, idx_dev, p_col_dev, idx2_dev
      integer(kind=c_int), value         :: na, na1, np_rem, ndef, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_compute_nnzl_nnzu_val_part1_c (p_col_dev, idx1_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, &
                                                  na, na1, np_rem, npc_n, nnzu_start, nnzl_start, np, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_compute_nnzl_nnzu_val_part1_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_compute_nnzl_nnzu_val_part1_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_compute_nnzl_nnzu_val_part1_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value    :: p_col_dev, idx1_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev
      integer(kind=c_int), value         :: na, na1, np_rem, npc_n, nnzu_start, nnzl_start, np, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_compute_nnzl_nnzu_val_part2_c (nnzu_val_dev, nnzl_val_dev, &
                                                  na, na1, nnzu_start, nnzl_start, npc_n, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_compute_nnzl_nnzu_val_part2_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_compute_nnzl_nnzu_val_part2_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_compute_nnzl_nnzu_val_part2_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value    :: nnzu_val_dev, nnzl_val_dev
      integer(kind=c_int), value         :: na, na1, nnzu_start, nnzl_start, npc_n, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_copy_qtmp1_slice_to_q_c (dataType, q_dev, qtmp1_dev, &
                                            l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, idx2_dev, idx_dev, &
                                            l_rqs, l_rqe, l_rows, matrixRows, gemm_dim_k, my_pcol, na1, np_rem, na, &
                                            debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_copy_qtmp1_slice_to_q_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_copy_qtmp1_slice_to_q_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_copy_qtmp1_slice_to_q_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: q_dev, qtmp1_dev
      integer(kind=c_intptr_t), value    :: l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, idx2_dev, idx_dev
      integer(kind=c_int), value         :: l_rqs, l_rqe, l_rows, matrixRows, gemm_dim_k, my_pcol, na1, np_rem, na, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_copy_q_slice_to_qtmp2_c (dataType, q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, &
                                            l_rows, l_rqs, l_rqe, matrixRows, matrixCols, gemm_dim_k,  gemm_dim_m, &
                                            ns, ncnt, indx, indx2, na, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_copy_q_slice_to_qtmp2_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_copy_q_slice_to_qtmp2_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_copy_q_slice_to_qtmp2_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: q_dev, qtmp2_dev
      integer(kind=c_intptr_t), value    :: idxq1_dev, l_col_out_dev
      integer(kind=c_int), value         :: l_rows, l_rqs, l_rqe, matrixRows, matrixCols, gemm_dim_k,  gemm_dim_m, &
                                            ns, ncnt, indx, indx2, na, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_copy_qtmp2_slice_to_q_c (dataType, q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, &
                                            l_rqs, l_rqe, l_rows, ncnt, gemm_dim_k, matrixRows, ns, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_copy_qtmp2_slice_to_q_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_copy_qtmp2_slice_to_q_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_copy_qtmp2_slice_to_q_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: q_dev, qtmp2_dev
      integer(kind=c_intptr_t), value    :: idxq1_dev, l_col_out_dev
      integer(kind=c_int), value         :: l_rqs, l_rqe, l_rows, ncnt, gemm_dim_k, matrixRows, ns, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_fill_ev_c (dataType, ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev, idxq1_dev, idx_dev, &
                              na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_fill_ev_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_fill_ev_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_fill_ev_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev
      integer(kind=c_intptr_t), value    :: idxq1_dev, idx_dev
      integer(kind=c_int), value         :: na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_fill_tmp_arrays_c (dataType, d1u_dev, d1_dev, zu_dev, z_dev, d1l_dev, zl_dev, &
                                      idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev, &
                                      na, np, na1, np_rem, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_fill_tmp_arrays_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_fill_tmp_arrays_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_fill_tmp_arrays_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: d1u_dev, d1_dev, zu_dev, z_dev, d1l_dev, zl_dev
      integer(kind=c_intptr_t), value    :: idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev
      integer(kind=c_int), value         :: na, np, na1, np_rem, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_zero_q_c(dataType, q_dev, p_col_out_dev, l_col_out_dev, &
                            na, my_pcol, l_rqs, l_rqe, matrixRows, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_zero_q_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_zero_q_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_zero_q_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: q_dev
      integer(kind=c_intptr_t), value    :: p_col_out_dev, l_col_out_dev
      integer(kind=c_int), value         :: na, my_pcol, l_rqs, l_rqe, matrixRows, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_copy_q_slice_to_qtmp1_c (dataType, qtmp1_dev, q_dev, ndef_c_dev, l_col_dev, idx2_dev, p_col_dev, &
                                            na2, na, my_pcol, l_rows, l_rqs, l_rqe, &
                                            matrixRows, gemm_dim_k, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_copy_q_slice_to_qtmp1_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_copy_q_slice_to_qtmp1_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_copy_q_slice_to_qtmp1_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: qtmp1_dev, q_dev
      integer(kind=c_intptr_t), value    :: ndef_c_dev, l_col_dev, idx2_dev, p_col_dev
      integer(kind=c_int), value         :: na2, na, my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_copy_qtmp1_to_qtmp1_tmp_c (dataType, qtmp1_dev, qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_copy_qtmp1_to_qtmp1_tmp_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_copy_qtmp1_to_qtmp1_tmp_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_copy_qtmp1_to_qtmp1_tmp_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: qtmp1_dev, qtmp1_tmp_dev
      integer(kind=c_int), value         :: gemm_dim_k, gemm_dim_l, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_fill_array_c (dataType, array_dev, value_dev, n, SM_count, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_fill_array_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_fill_array_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_fill_array_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: array_dev, value_dev
      integer(kind=c_int), value         :: n, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_solve_secular_equation_loop_c (dataType, d1_dev, z1_dev, delta_dev, rho_dev, &
                                            z_dev, dbase_dev, ddiff_dev, my_proc, na1, n_procs, SM_count, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                    bind(C, name="cuda_solve_secular_equation_loop_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                    bind(C, name= "hip_solve_secular_equation_loop_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                    bind(C, name="sycl_solve_secular_equation_loop_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: d1_dev, z1_dev, delta_dev, z_dev, rho_dev, dbase_dev, ddiff_dev
      integer(kind=c_int), value         :: my_proc, na1, n_procs, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_local_product_c (dataType, z_dev, z_extended_dev, na1, SM_count, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                    bind(C, name="cuda_local_product_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                    bind(C, name= "hip_local_product_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                    bind(C, name="sycl_local_product_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: z_dev, z_extended_dev
      integer(kind=c_int), value         :: na1, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_add_tmp_loop_c (dataType, d1_dev, dbase_dev, ddiff_dev, z_dev, ev_scale_dev, tmp_extended_dev, &
                                 na1, my_proc, n_procs, SM_count, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                    bind(C, name="cuda_add_tmp_loop_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                    bind(C, name= "hip_add_tmp_loop_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                    bind(C, name="sycl_add_tmp_loop_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: d1_dev, dbase_dev, ddiff_dev, z_dev, ev_scale_dev, tmp_extended_dev
      integer(kind=c_int), value         :: na1, my_proc, n_procs, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface

  interface
    subroutine gpu_copy_qtmp1_q_compute_nnzu_nnzl_c(dataType, qtmp1_dev, q_dev, &
                                                  p_col_dev, l_col_dev, idx1_dev, coltyp_dev, nnzul_dev, &
                                                  na1, l_rnm, l_rqs, l_rqm, l_rows, my_pcol, ldq_tmp1, ldq, &
                                                  SM_count, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_copy_qtmp1_q_compute_nnzu_nnzl_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_copy_qtmp1_q_compute_nnzu_nnzl_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_copy_qtmp1_q_compute_nnzu_nnzl_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: qtmp1_dev, q_dev, p_col_dev, l_col_dev, idx1_dev, coltyp_dev, nnzul_dev
      integer(kind=c_int), value         :: na1, l_rnm, l_rqs, l_rqm, l_rows, my_pcol, ldq_tmp1, ldq, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
  subroutine gpu_fill_z_c(dataType, z_dev, q_dev, p_col_dev, l_col_dev, &
                          sig_int, na, my_pcol, row_q, ldq, SM_count, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_fill_z_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_fill_z_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_fill_z_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: z_dev, q_dev, p_col_dev, l_col_dev
      integer(kind=c_int), value         :: sig_int, na, my_pcol, row_q, ldq, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface

#endif /* defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION) */


  contains


    subroutine gpu_update_ndef_c (ndef_c_dev, idx_dev, p_col_dev, idx2_dev, &
                                  na, na1, np_rem, ndef, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value    :: ndef_c_dev, idx_dev, p_col_dev, idx2_dev
      integer(kind=c_int), value         :: na, na1, np_rem, ndef, debug
      integer(kind=c_intptr_t), value    :: my_stream
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_update_ndef_c_c (ndef_c_dev, idx_dev, p_col_dev, idx2_dev, &
                                na, na1, np_rem, ndef, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_compute_nnzl_nnzu_val_part1 (p_col_dev, idx1_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, &
                                                na, na1, np_rem, npc_n, nnzu_start, nnzl_start, np, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value    :: p_col_dev, idx1_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev
      integer(kind=c_int), value         :: na, na1, np_rem, npc_n, nnzu_start, nnzl_start, np, debug
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_compute_nnzl_nnzu_val_part1_c (p_col_dev, idx1_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, &
                                              na, na1, np_rem, npc_n, nnzu_start, nnzl_start, np, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_compute_nnzl_nnzu_val_part2 (nnzu_val_dev, nnzl_val_dev, &
                                                  na, na1, nnzu_start, nnzl_start, npc_n, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value    :: nnzu_val_dev, nnzl_val_dev
      integer(kind=c_int), value         :: na, na1, nnzu_start, nnzl_start, npc_n, debug
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_compute_nnzl_nnzu_val_part2_c (nnzu_val_dev, nnzl_val_dev, &
                                              na, na1, nnzu_start, nnzl_start, npc_n, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_copy_qtmp1_slice_to_q (dataType, q_dev, qtmp1_dev, &
                                          l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, idx2_dev, idx_dev, &
                                          l_rqs, l_rqe, l_rows, matrixRows, gemm_dim_k, my_pcol, na1, np_rem, na, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: q_dev, qtmp1_dev
      integer(kind=c_intptr_t), value    :: l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, idx2_dev, idx_dev
      integer(kind=c_int), value         :: l_rqs, l_rqe, l_rows, matrixRows, gemm_dim_k, my_pcol, na1, np_rem, na, debug
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_copy_qtmp1_slice_to_q_c (dataType, q_dev, qtmp1_dev, &
                                        l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, idx2_dev, idx_dev, &
                                        l_rqs, l_rqe, l_rows, matrixRows, gemm_dim_k,  my_pcol, na1, np_rem, na, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_copy_q_slice_to_qtmp2 (dataType, q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, &
                                          l_rows, l_rqs, l_rqe, matrixRows, matrixCols, gemm_dim_k,  gemm_dim_m, &
                                          ns, ncnt, indx, indx2, na, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: q_dev, qtmp2_dev
      integer(kind=c_intptr_t), value    :: idxq1_dev, l_col_out_dev
      integer(kind=c_int), value         :: l_rows, l_rqs, l_rqe, matrixRows, matrixCols, gemm_dim_k,  gemm_dim_m, &
                                            ns, ncnt, indx, indx2, na, debug
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_copy_q_slice_to_qtmp2_c (dataType, q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, &
                                        l_rows, l_rqs, l_rqe, matrixRows, matrixCols, gemm_dim_k,  gemm_dim_m, &
                                        ns, ncnt, indx, indx2, na, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_copy_qtmp2_slice_to_q (dataType, q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, &
                                          l_rqs, l_rqe, l_rows, ncnt, gemm_dim_k, matrixRows, ns, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: q_dev, qtmp2_dev
      integer(kind=c_intptr_t), value    :: idxq1_dev, l_col_out_dev
      integer(kind=c_int), value         :: l_rqs, l_rqe, l_rows, ncnt, gemm_dim_k, matrixRows, ns, debug
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_copy_qtmp2_slice_to_q_c (dataType, q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, &
                                        l_rqs, l_rqe, l_rows, ncnt, gemm_dim_k, matrixRows, ns, debug, my_stream)
#endif    
    end subroutine


    subroutine gpu_fill_ev (dataType, ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev, idxq1_dev, idx_dev, &
                            na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev
      integer(kind=c_intptr_t), value    :: idxq1_dev, idx_dev
      integer(kind=c_int), value         :: na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt, debug
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_fill_ev_c (dataType, ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev, idxq1_dev, idx_dev, &
                          na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt, debug, my_stream)
#endif
    end subroutine

    
    subroutine gpu_fill_tmp_arrays (dataType, d1u_dev, d1_dev, zu_dev, z_dev, d1l_dev, zl_dev, &
                                    idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev, &
                                    na, np, na1, np_rem, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: d1u_dev, d1_dev, zu_dev, z_dev, d1l_dev, zl_dev
      integer(kind=c_intptr_t), value    :: idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev
      integer(kind=c_int), value         :: na, np, na1, np_rem, debug
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_fill_tmp_arrays_c (dataType, d1u_dev, d1_dev, zu_dev, z_dev, d1l_dev, zl_dev, &
                                  idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev, &
                                  na, np, na1, np_rem, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_zero_q(dataType, q_dev, p_col_out_dev, l_col_out_dev, &
                          na, my_pcol, l_rqs, l_rqe, matrixRows, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: q_dev
      integer(kind=c_intptr_t), value    :: p_col_out_dev, l_col_out_dev
      integer(kind=c_int), value         :: na, my_pcol, l_rqs, l_rqe, matrixRows, debug
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_zero_q_c(dataType, q_dev, p_col_out_dev, l_col_out_dev, &
                        na, my_pcol, l_rqs, l_rqe, matrixRows, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_copy_q_slice_to_qtmp1 (dataType, qtmp1_dev, q_dev, ndef_c_dev, l_col_dev, idx2_dev, p_col_dev, &
                                          na2, na, my_pcol, l_rows, l_rqs, l_rqe, &
                                          matrixRows, gemm_dim_k, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: qtmp1_dev, q_dev
      integer(kind=c_intptr_t), value    :: ndef_c_dev, l_col_dev, idx2_dev, p_col_dev
      integer(kind=c_int), value         :: na2, na, my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k, debug
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_copy_q_slice_to_qtmp1_c (dataType, qtmp1_dev, q_dev, ndef_c_dev, l_col_dev, idx2_dev, p_col_dev, &
                                        na2, na, my_pcol, l_rows, l_rqs, l_rqe, &
                                        matrixRows, gemm_dim_k, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_copy_qtmp1_to_qtmp1_tmp (dataType, qtmp1_dev, qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: qtmp1_dev, qtmp1_tmp_dev
      integer(kind=c_int), value         :: gemm_dim_k, gemm_dim_l, debug
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_copy_qtmp1_to_qtmp1_tmp_c (dataType, qtmp1_dev, qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_fill_array (dataType, array_dev, value_dev, n, SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: array_dev, value_dev
      integer(kind=c_int), value         :: n, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_fill_array_c (dataType, array_dev, value_dev, n, SM_count, debug, my_stream)
#endif
    end subroutine
    
    
    subroutine gpu_solve_secular_equation_loop (dataType, d1_dev, z1_dev, delta_dev, rho_dev, &
                                            z_dev, dbase_dev, ddiff_dev,  my_proc, na1, n_procs, SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: d1_dev, z1_dev, delta_dev, z_dev, rho_dev, dbase_dev, ddiff_dev
      integer(kind=c_int), value         :: my_proc, na1, n_procs, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_solve_secular_equation_loop_c (dataType, d1_dev, z1_dev, delta_dev, rho_dev, &
                                              z_dev, dbase_dev, ddiff_dev, my_proc, na1, n_procs, SM_count, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_local_product (dataType, z_dev, z_extended_dev, na1, SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: z_dev, z_extended_dev
      integer(kind=c_int), value         :: na1, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_local_product_c (dataType, z_dev, z_extended_dev, na1, SM_count, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_add_tmp_loop (dataType, d1_dev, dbase_dev, ddiff_dev, z_dev, ev_scale_dev, tmp_extended_dev, &
                                 na1, my_proc, n_procs, SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: d1_dev, dbase_dev, ddiff_dev, z_dev, ev_scale_dev, tmp_extended_dev
      integer(kind=c_int), value         :: na1, my_proc, n_procs, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_add_tmp_loop_c (dataType, d1_dev, dbase_dev, ddiff_dev, z_dev, ev_scale_dev, tmp_extended_dev, &
                               na1, my_proc, n_procs, SM_count, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_copy_qtmp1_q_compute_nnzu_nnzl (dataType, qtmp1_dev, q_dev, &
                                                  p_col_dev, l_col_dev, idx1_dev, coltyp_dev, nnzul_dev, &
                                                  na1, l_rnm, l_rqs, l_rqm, l_rows, my_pcol, ldq_tmp1, ldq, &
                                                  SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: qtmp1_dev, q_dev, p_col_dev, l_col_dev, idx1_dev, coltyp_dev, nnzul_dev
      integer(kind=c_int), value         :: na1, l_rnm, l_rqs, l_rqm, l_rows, my_pcol, ldq_tmp1, ldq, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_copy_qtmp1_q_compute_nnzu_nnzl_c(dataType, qtmp1_dev, q_dev, &
                                                p_col_dev, l_col_dev, idx1_dev, coltyp_dev, nnzul_dev, &
                                                na1, l_rnm, l_rqs, l_rqm, l_rows, my_pcol, ldq_tmp1, ldq, &
                                                SM_count, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_fill_z(dataType, z_dev, q_dev, p_col_dev, l_col_dev, &
                          sig_int, na, my_pcol, row_q, ldq, SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: z_dev, q_dev, p_col_dev, l_col_dev
      integer(kind=c_int), value         :: sig_int, na, my_pcol, row_q, ldq, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_fill_z_c(dataType, z_dev, q_dev, p_col_dev, l_col_dev, &
                        sig_int, na, my_pcol, row_q, ldq, SM_count, debug, my_stream)
#endif
    end subroutine

end module
