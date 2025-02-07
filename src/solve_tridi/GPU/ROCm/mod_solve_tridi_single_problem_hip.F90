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

! This file is auto-generated. Do NOT edit

module solve_single_problem_hip
  use, intrinsic :: iso_c_binding
  use precision

  implicit none

  public

  interface
    subroutine hip_check_monotony_double_c(d_dev, q_dev, qtmp_dev, nlen, ldq, &
                                                             my_stream) &
                                                     bind(C, name="hip_check_monotony_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: d_dev, q_dev, qtmp_dev
      integer(kind=c_int), intent(in)  :: nlen, ldq
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface



  interface
    subroutine hip_construct_tridi_matrix_double_c(q_dev, d_dev, e_dev, nlen, ldq, &
                                            my_stream) &
                                            bind(C, name="hip_construct_tridi_matrix_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: q_dev, d_dev, e_dev
      integer(kind=c_int), intent(in)  :: nlen, ldq
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface



  interface
    subroutine hip_fill_ev_double_c(ev_dev, tmp_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev, idxq1_dev, &
                                           idx_dev, na, gemm_dim_l, gemm_dim_m, nnzu, ns, &
                                           ncnt, my_stream) &
                                           bind(C, name="hip_fill_ev_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: ev_dev, tmp_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev
      type(c_ptr), value               :: idxq1_dev, idx_dev
      integer(kind=c_int), intent(in)  :: na, gemm_dim_l, nnzu, ns, ncnt, gemm_dim_m
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface



#ifdef WANT_SINGLE_PRECISION_REAL
  interface
    subroutine hip_check_monotony_float_c(d_dev, q_dev, qtmp_dev, nlen, ldq, &
                                                             my_stream) &
                                                     bind(C, name="hip_check_monotony_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: d_dev, q_dev, qtmp_dev
      integer(kind=c_int), intent(in)  :: nlen, ldq
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

#endif


#ifdef WANT_SINGLE_PRECISION_REAL
  interface
    subroutine hip_construct_tridi_matrix_float_c(q_dev, d_dev, e_dev, nlen, ldq, &
                                            my_stream) &
                                            bind(C, name="hip_construct_tridi_matrix_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: q_dev, d_dev, e_dev
      integer(kind=c_int), intent(in)  :: nlen, ldq
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

#endif


#ifdef WANT_SINGLE_PRECISION_REAL
  interface
    subroutine hip_fill_ev_float_c(ev_dev, tmp_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev, idxq1_dev, &
                                           idx_dev, na, gemm_dim_l, gemm_dim_m, nnzu, ns, &
                                           ncnt, my_stream) &
                                           bind(C, name="hip_fill_ev_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: ev_dev, tmp_dev, d1u_dev, dbase_dev, ddiff_dev, zu_dev, ev_scale_dev
      type(c_ptr), value               :: idxq1_dev, idx_dev
      integer(kind=c_int), intent(in)  :: na, gemm_dim_l, nnzu, ns, ncnt, gemm_dim_m
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

#endif


  contains
    subroutine hip_check_monotony_double(d_dev, q_dev, qtmp_dev, nlen, ldq, &
                                                              my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in)    :: nlen, ldq
      integer(kind=C_intptr_T)           :: d_dev, q_dev, qtmp_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_AMD_GPU_VERSION
      if (present(my_stream)) then
        call hip_check_monotony_double_c(d_dev, q_dev, qtmp_dev, nlen, ldq, &
                                                  my_stream)
      else
        call hip_check_monotony_double_c(d_dev, q_dev, qtmp_dev, nlen, ldq, &
                                                  my_stream2)
      endif
#endif

    end subroutine


    subroutine hip_construct_tridi_matrix_double(q_dev, d_dev, e_dev, nlen, ldq, &
                                              my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in)    :: nlen, ldq
      integer(kind=C_intptr_T)           :: q_dev, d_dev, e_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_AMD_GPU_VERSION
      if (present(my_stream)) then
        call hip_construct_tridi_matrix_double_c(q_dev, d_dev, e_dev, nlen, ldq, &
                                                  my_stream)
      else
        call hip_construct_tridi_matrix_double_c(q_dev, d_dev, e_dev, nlen, ldq, &
                                                  my_stream2)
      endif
#endif

    end subroutine


#ifdef WANT_SINGLE_PRECISION_REAL
    subroutine hip_check_monotony_float(d_dev, q_dev, qtmp_dev, nlen, ldq, &
                                                              my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in)    :: nlen, ldq
      integer(kind=C_intptr_T)           :: d_dev, q_dev, qtmp_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_AMD_GPU_VERSION
      if (present(my_stream)) then
        call hip_check_monotony_float_c(d_dev, q_dev, qtmp_dev, nlen, ldq, &
                                                  my_stream)
      else
        call hip_check_monotony_float_c(d_dev, q_dev, qtmp_dev, nlen, ldq, &
                                                  my_stream2)
      endif
#endif

    end subroutine
#endif


#ifdef WANT_SINGLE_PRECISION_REAL
    subroutine hip_construct_tridi_matrix_float(q_dev, d_dev, e_dev, nlen, ldq, &
                                              my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in)    :: nlen, ldq
      integer(kind=C_intptr_T)           :: q_dev, d_dev, e_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_AMD_GPU_VERSION
      if (present(my_stream)) then
        call hip_construct_tridi_matrix_float_c(q_dev, d_dev, e_dev, nlen, ldq, &
                                                  my_stream)
      else
        call hip_construct_tridi_matrix_float_c(q_dev, d_dev, e_dev, nlen, ldq, &
                                                  my_stream2)
      endif
#endif

    end subroutine
#endif


end module
