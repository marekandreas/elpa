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

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION)

  interface
  subroutine gpu_solve_secular_equation_loop_c (dataType, d1_dev, z1_dev, delta_dev, rho_dev, &
                                            z_dev, dbase_dev, ddiff_dev, my_proc, na1, n_procs, myid, SM_count, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                     bind(C, name="cuda_solve_secular_equation_loop_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                     bind(C, name="hip_solve_secular_equation_loop_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: d1_dev, z1_dev, delta_dev, z_dev, rho_dev, dbase_dev, ddiff_dev
      integer(kind=c_int), value         :: my_proc, na1, n_procs, debug, SM_count, myid
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface

  interface
  subroutine gpu_local_product_c (dataType, z_dev, z_extended_dev, na1, SM_count, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                     bind(C, name="cuda_local_product_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                     bind(C, name="hip_local_product_FromC")
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
                                                     bind(C, name="hip_add_tmp_loop_FromC")
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
                                                  bind(C, name="hip_copy_qtmp1_q_compute_nnzu_nnzl_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: qtmp1_dev, q_dev, p_col_dev, l_col_dev, idx1_dev, coltyp_dev, nnzul_dev
      integer(kind=c_int), value         :: na1, l_rnm, l_rqs, l_rqm, l_rows, my_pcol, ldq_tmp1, ldq, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface

#endif /* defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) */


  contains


    subroutine gpu_solve_secular_equation_loop (dataType, d1_dev, z1_dev, delta_dev, rho_dev, &
                                            z_dev, dbase_dev, ddiff_dev,  my_proc, na1, n_procs, myid, SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: d1_dev, z1_dev, delta_dev, z_dev, rho_dev, dbase_dev, ddiff_dev
      integer(kind=c_int), value         :: my_proc, na1, n_procs, debug, SM_count, myid
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION)
      call gpu_solve_secular_equation_loop_c (dataType, d1_dev, z1_dev, delta_dev, rho_dev, &
                                              z_dev, dbase_dev, ddiff_dev, my_proc, na1, n_procs, myid, SM_count, debug, my_stream)
#endif
    end subroutine


    subroutine gpu_local_product (dataType, z_dev, z_extended_dev, na1, SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: z_dev, z_extended_dev
      integer(kind=c_int), value         :: na1, SM_count, debug
      integer(kind=c_intptr_t), value    :: my_stream

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION)
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

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION)
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

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION)
      call gpu_copy_qtmp1_q_compute_nnzu_nnzl_c(dataType, qtmp1_dev, q_dev, &
                                                p_col_dev, l_col_dev, idx1_dev, coltyp_dev, nnzul_dev, &
                                                na1, l_rnm, l_rqs, l_rqm, l_rows, my_pcol, ldq_tmp1, ldq, &
                                                SM_count, debug, my_stream)
#endif
    end subroutine

end module
