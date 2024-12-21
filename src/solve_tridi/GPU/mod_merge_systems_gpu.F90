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

!    This file was written by A.Marek, MPCDF
#endif


#include "config-f90.h"


module merge_systems_gpu
  use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
  use merge_systems_cuda
#endif
#ifdef WITH_AMD_GPU_VERSION
  use merge_systems_hip
#endif
  use precision

  implicit none

  public

  contains
    subroutine gpu_copy_qtmp1_slice_to_q_double(q_dev, qtmp1_dev, l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, &
                                                 idx2_dev, idx_dev, l_rqs, &
                                                 l_rqe, l_rows, matrixRows, &
                                                 gemm_dim_k, my_pcol, na1, np_rem, na, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: l_rqs, l_rqe, l_rows, matrixRows, gemm_dim_k, my_pcol, na1, np_rem, na
      integer(kind=c_intptr_t)           :: q_dev, qtmp1_dev
      integer(kind=c_intptr_t)           :: l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, idx2_dev, idx_dev  
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_qtmp1_slice_to_q_double(q_dev, qtmp1_dev, l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, &
                                                 idx2_dev, idx_dev, l_rqs, l_rqe, l_rows, matrixRows, &
                                                 gemm_dim_k,  my_pcol, na1, np_rem, na, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_qtmp1_slice_to_q_double(q_dev, qtmp1_dev, l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, &
                                                 idx2_dev, idx_dev, l_rqs, l_rqe, l_rows, matrixRows, &
                                                 gemm_dim_k,  my_pcol, na1, np_rem, na, my_stream)
#endif 
  
    end subroutine


    subroutine gpu_copy_qtmp1_slice_to_q_float(q_dev, qtmp1_dev, l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, &
                                                 idx2_dev, idx_dev, l_rqs, &
                                                 l_rqe, l_rows, matrixRows, &
                                                 gemm_dim_k, my_pcol, na1, np_rem, na, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: l_rqs, l_rqe, l_rows, matrixRows, gemm_dim_k, my_pcol, na1, np_rem, na
      integer(kind=c_intptr_t)           :: q_dev, qtmp1_dev
      integer(kind=c_intptr_t)           :: l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, idx2_dev, idx_dev  
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2


#ifdef WANT_SINGLE_PRECISION_REAL
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_qtmp1_slice_to_q_float(q_dev, qtmp1_dev, l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, &
                                                 idx2_dev, idx_dev, l_rqs, l_rqe, l_rows, matrixRows, &
                                                 gemm_dim_k,  my_pcol, na1, np_rem, na, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_qtmp1_slice_to_q_float(q_dev, qtmp1_dev, l_col_out_dev, p_col_out_dev, ndef_c_dev, p_col_dev, &
                                                 idx2_dev, idx_dev, l_rqs, l_rqe, l_rows, matrixRows, &
                                                 gemm_dim_k,  my_pcol, na1, np_rem, na, my_stream)
#endif 
#endif
    end subroutine

    subroutine gpu_copy_q_slice_to_qtmp2_double(q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, l_rows, l_rqs, l_rqe, matrixRows, &
                    matrixCols, gemm_dim_k, gemm_dim_m, ns, ncnt, indx, indx2, na, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k, gemm_dim_m, ns, ncnt, indx, &
                                            indx2, na, &
                                            matrixCols
      integer(kind=c_intptr_t)           :: q_dev, qtmp2_dev
      integer(kind=c_intptr_t)           :: idxq1_dev, l_col_out_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_q_slice_to_qtmp2_double(q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, l_rows, l_rqs, l_rqe, &
                                                       matrixRows, matrixCols, gemm_dim_k, gemm_dim_m, ns, &
                                                       ncnt, indx, indx2, na, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_q_slice_to_qtmp2_double(q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, l_rows, l_rqs, l_rqe, &
                                                       matrixRows, matrixCols, gemm_dim_k, gemm_dim_m, ns, &
                                                       ncnt, indx, indx2, na, my_stream)
#endif 
  
    end subroutine

    subroutine gpu_copy_q_slice_to_qtmp2_float(q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, l_rows, l_rqs, l_rqe, matrixRows, &
                    matrixCols, gemm_dim_k, gemm_dim_m, ns, ncnt, indx, indx2, na, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k, gemm_dim_m, ns, ncnt, indx, &
                                            indx2, na, &
                                            matrixCols
      integer(kind=c_intptr_t)           :: q_dev, qtmp2_dev
      integer(kind=c_intptr_t)           :: idxq1_dev, l_col_out_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WANT_SINGLE_PRECISION_REAL
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_q_slice_to_qtmp2_float(q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, l_rows, l_rqs, l_rqe, &
                                                       matrixRows, matrixCols, gemm_dim_k, gemm_dim_m, ns, &
                                                       ncnt, indx, indx2, na, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_q_slice_to_qtmp2_float(q_dev, qtmp2_dev, idxq1_dev, l_col_out_dev, l_rows, l_rqs, l_rqe, &
                                                       matrixRows, matrixCols, gemm_dim_k, gemm_dim_m, ns, &
                                                       ncnt, indx, indx2, na, my_stream)
#endif 
#endif 
    end subroutine


    subroutine gpu_fill_ev_double(ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev, idxq1_dev, &
                                         idx_dev, &
                                     na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt, my_stream) 
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: na, gemm_dim_l, nnzu, ns, ncnt, gemm_dim_m
      integer(kind=c_intptr_t)           :: ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev
      integer(kind=c_intptr_t)           :: idxq1_dev, idx_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_fill_ev_double(ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev, idxq1_dev, &
                idx_dev, &
                                     na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_fill_ev_double(ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev, idxq1_dev, &
                idx_dev, &
                                     na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt, my_stream)
#endif 
  
    end subroutine

    subroutine gpu_fill_ev_float(ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev, idxq1_dev, &
                                         idx_dev, &
                                     na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt, my_stream) 
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: na, gemm_dim_l, nnzu, ns, ncnt, gemm_dim_m
      integer(kind=c_intptr_t)           :: ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev
      integer(kind=c_intptr_t)           :: idxq1_dev, idx_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WANT_SINGLE_PRECISION_REAL
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_fill_ev_float(ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev, idxq1_dev, &
                idx_dev, &
                                     na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_fill_ev_float(ev_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev, idxq1_dev, &
                idx_dev, &
                                     na, gemm_dim_l, gemm_dim_m, nnzu, ns, ncnt, my_stream)
#endif 
#endif

    end subroutine

    subroutine gpu_copy_qtmp2_slice_to_q_double(q_dev, qtmp2_dev, idx1q_dev, l_col_out_dev, l_rqs, l_rqe, l_rows, ncnt, &   
                                                 gemm_dim_k, matrixRows, ns,  my_stream)
                    
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: l_rqs, l_rqe, l_rows, ncnt, gemm_dim_k, matrixRows, ns
      integer(kind=c_intptr_t)           :: q_dev, qtmp2_dev
      integer(kind=c_intptr_t)           :: idx1q_dev, l_col_out_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2


#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_qtmp2_slice_to_q_double(q_dev, qtmp2_dev, idx1q_dev, l_col_out_dev, l_rqs, l_rqe, l_rows, ncnt, &
                                                 gemm_dim_k, matrixRows, ns, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_qtmp2_slice_to_q_double(q_dev, qtmp2_dev, idx1q_dev, l_col_out_dev, l_rqs, l_rqe, l_rows, ncnt, &
                                                 gemm_dim_k, matrixRows, ns, my_stream)
#endif 
  
    end subroutine

    subroutine gpu_copy_qtmp2_slice_to_q_float(q_dev, qtmp2_dev, idx1q_dev, l_col_out_dev, l_rqs, l_rqe, l_rows, ncnt, &   
                                                 gemm_dim_k, matrixRows, ns,  my_stream)
                    
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: l_rqs, l_rqe, l_rows, ncnt, gemm_dim_k, matrixRows, ns
      integer(kind=c_intptr_t)           :: q_dev, qtmp2_dev
      integer(kind=c_intptr_t)           :: idx1q_dev, l_col_out_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2


#ifdef WANT_SINGLE_PRECISION_REAL
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_qtmp2_slice_to_q_float(q_dev, qtmp2_dev, idx1q_dev, l_col_out_dev, l_rqs, l_rqe, l_rows, ncnt, &
                                                 gemm_dim_k, matrixRows, ns, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_qtmp2_slice_to_q_float(q_dev, qtmp2_dev, idx1q_dev, l_col_out_dev, l_rqs, l_rqe, l_rows, ncnt, &
                                                 gemm_dim_k, matrixRows, ns, my_stream)
#endif 
#endif

    end subroutine


    subroutine gpu_fill_tmp_arrays_double(idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev, d1u_dev, &
                                            d1_dev, &
                                             zu_dev, z_dev, d1l_dev, zl_dev, na, np, na1, np_rem, my_stream)
                    
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: na, np, na1, np_rem
      integer(kind=c_intptr_t)           :: idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev
      integer(kind=c_intptr_t)           :: d1u_dev, d1_dev, zu_dev, z_dev, d1l_dev, zl_dev

      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_fill_tmp_arrays_double(idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev, d1u_dev, d1_dev, &
                                           zu_dev, z_dev, d1l_dev, zl_dev, na, np, na1, np_rem, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_fill_tmp_arrays_double(idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev, d1u_dev, d1_dev, &
                                           zu_dev, z_dev, d1l_dev, zl_dev, na, np, na1, np_rem, my_stream)
#endif 
  
    end subroutine

    subroutine gpu_fill_tmp_arrays_float(idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev, d1u_dev, d1_dev, &
                                             zu_dev, z_dev, d1l_dev, zl_dev, na, np, na1, np_rem, my_stream)
                    
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: na, np, na1, np_rem
      integer(kind=c_intptr_t)           :: idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev
      integer(kind=c_intptr_t)           :: d1u_dev, d1_dev, zu_dev, z_dev, d1l_dev, zl_dev

      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WANT_SINGLE_PRECISION_REAL
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_fill_tmp_arrays_float(idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev, d1u_dev, d1_dev, &
                                           zu_dev, z_dev, d1l_dev, zl_dev, na, np, na1, np_rem, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_fill_tmp_arrays_float(idx1_dev, p_col_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, nnzul_dev, d1u_dev, d1_dev, &
                                           zu_dev, z_dev, d1l_dev, zl_dev, na, np, na1, np_rem, my_stream)
#endif 
#endif  
    end subroutine

    subroutine gpu_zero_q_double(q_dev, p_col_out_dev, l_col_out_dev, na, my_pcol, l_rqs, l_rqe, &
                                                      matrixRows,  my_stream)

                    
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: na, my_pcol, l_rqs, l_rqe, matrixRows
      integer(kind=c_intptr_t)           :: p_col_out_dev, l_col_out_dev
      integer(kind=c_intptr_t)           :: q_dev

      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_zero_q_double(q_dev, p_col_out_dev, l_col_out_dev, na, my_pcol, l_rqs, l_rqe, &
                                                      matrixRows,  my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_zero_q_double(q_dev, p_col_out_dev, l_col_out_dev, na, my_pcol, l_rqs, l_rqe, &
                                                      matrixRows,  my_stream)
#endif 
    end subroutine

    subroutine gpu_zero_q_float(q_dev, p_col_out_dev, l_col_out_dev, na, my_pcol, l_rqs, l_rqe, &
                                                      matrixRows,  my_stream)

                    
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: na, my_pcol, l_rqs, l_rqe, matrixRows
      integer(kind=c_intptr_t)           :: p_col_out_dev, l_col_out_dev
      integer(kind=c_intptr_t)           :: q_dev

      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WANT_SINGLE_PRECISION_REAL
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_zero_q_float(q_dev, p_col_out_dev, l_col_out_dev, na, my_pcol, l_rqs, l_rqe, &
                                                      matrixRows,  my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_zero_q_float(q_dev, p_col_out_dev, l_col_out_dev, na, my_pcol, l_rqs, l_rqe, &
                                                      matrixRows,  my_stream)
#endif 
#endif  
    end subroutine

    subroutine gpu_copy_q_slice_to_qtmp1_double(qtmp1_dev, q_dev, ndef_c_dev, l_col_dev, idx2_dev, p_col_dev, na2, na, &
                                                my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k, my_stream)

                    
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: na2, na, my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k

      integer(kind=c_intptr_t)           :: ndef_c_dev, l_col_dev, idx2_dev, p_col_dev
      integer(kind=c_intptr_t)           :: q_dev, qtmp1_dev

      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_q_slice_to_qtmp1_double(qtmp1_dev, q_dev, ndef_c_dev, l_col_dev, idx2_dev, p_col_dev, na2, na, &
                                                my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_q_slice_to_qtmp1_double(qtmp1_dev, q_dev, ndef_c_dev, l_col_dev, idx2_dev, p_col_dev, na2, na, &
                                                my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k, my_stream)
#endif 
    end subroutine

    subroutine gpu_copy_q_slice_to_qtmp1_float(qtmp1_dev, q_dev, ndef_c_dev, l_col_dev, idx2_dev, p_col_dev, na2, na, &
                                                my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k, &
                                                my_stream)

                    
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: na2, na, my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k
      integer(kind=c_intptr_t)           :: ndef_c_dev, l_col_dev, idx2_dev, p_col_dev
      integer(kind=c_intptr_t)           :: q_dev, qtmp1_dev

      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WANT_SINGLE_PRECISION_REAL
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_q_slice_to_qtmp1_float(qtmp1_dev, q_dev, ndef_c_dev, l_col_dev, idx2_dev, p_col_dev, na2, na, &
                                                my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k, &
                                                my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_q_slice_to_qtmp1_float(qtmp1_dev, q_dev, ndef_c_dev, l_col_dev, idx2_dev, p_col_dev, na2, na, &
                                                my_pcol, l_rows, l_rqs, l_rqe, matrixRows, gemm_dim_k, &
                                                my_stream)
#endif
#endif
    end subroutine



    subroutine gpu_copy_qtmp1_to_qtmp1_tmp_double(qtmp1_dev, qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l, my_stream)

                    
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: gemm_dim_k, gemm_dim_l
      integer(kind=c_intptr_t)           :: qtmp1_dev, qtmp1_tmp_dev

      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_qtmp1_to_qtmp1_tmp_double(qtmp1_dev, qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_qtmp1_to_qtmp1_tmp_double(qtmp1_dev, qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l, my_stream)
#endif
    end subroutine


    subroutine gpu_copy_qtmp1_to_qtmp1_tmp_float(qtmp1_dev, qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l, my_stream)

                    
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: gemm_dim_k, gemm_dim_l
      integer(kind=c_intptr_t)           :: qtmp1_dev, qtmp1_tmp_dev

      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WANT_SINGLE_PRECISION_REAL
#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_qtmp1_to_qtmp1_tmp_float(qtmp1_dev, qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_qtmp1_to_qtmp1_tmp_float(qtmp1_dev, qtmp1_tmp_dev, gemm_dim_k, gemm_dim_l, my_stream)
#endif
#endif
    end subroutine



    subroutine gpu_update_ndef_c(ndef_c_dev, idx_dev, p_col_dev, idx2_dev, na, na1, np_rem, ndef, &
                                                        my_stream)


                    
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: na, na1, np_rem, ndef
      integer(kind=c_intptr_t)           :: ndef_c_dev, idx_dev, p_col_dev, idx2_dev

      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_update_ndef_c(ndef_c_dev, idx_dev, p_col_dev, idx2_dev, na, na1, np_rem, ndef, &
                                                        my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_update_ndef_c(ndef_c_dev, idx_dev, p_col_dev, idx2_dev, na, na1, np_rem, ndef, &
                                                        my_stream2)
#endif
    end subroutine



    subroutine gpu_compute_nnzl_nnzu_val_part1(p_col_dev, idx1_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, na, na1, np_rem, &
                                         npc_n, nnzu_start, nnzl_start, np, my_stream)


                    
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: na, na1, np_rem, npc_n, nnzu_start, nnzl_start, np
      integer(kind=c_intptr_t)           :: p_col_dev, idx1_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev

      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_compute_nnzl_nnzu_val_part1(p_col_dev, idx1_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, na, na1, np_rem, &
                                         npc_n, nnzu_start, nnzl_start, np, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
      call hip_compute_nnzl_nnzu_val_part1(p_col_dev, idx1_dev, coltyp_dev, nnzu_val_dev, nnzl_val_dev, na, na1, np_rem, &
                                         npc_n, nnzu_start, nnzl_start, np, my_stream)
#endif
    end subroutine




    subroutine gpu_compute_nnzl_nnzu_val_part2(nnzu_val_dev, nnzl_val_dev, na, na1, &
                                         nnzu_start, nnzl_start, npc_n, my_stream)


                    
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: na, na1, nnzu_start, nnzl_start, npc_n
      integer(kind=c_intptr_t)           :: nnzu_val_dev, nnzl_val_dev

      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_compute_nnzl_nnzu_val_part2(nnzu_val_dev, nnzl_val_dev, na, na1, &
                                         nnzu_start, nnzl_start, npc_n, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
      call hip_compute_nnzl_nnzu_val_part2(nnzu_val_dev, nnzl_val_dev, na, na1, &
                                         nnzu_start, nnzl_start, npc_n, my_stream)
#endif
    end subroutine
end module

