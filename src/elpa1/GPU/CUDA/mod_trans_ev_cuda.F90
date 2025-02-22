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

! This file is auto-generated. Do NOT edit

module trans_ev_cuda
  use, intrinsic :: iso_c_binding
  use precision

  implicit none

  public

  interface
    subroutine cuda_scale_qmat_double_complex_c(ldq, l_cols, q_dev, tau_dev, my_stream) &
                                                  bind(C, name="cuda_scale_qmat_double_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int), intent(in)  :: ldq, l_cols
      integer(kind=c_intptr_t), value  :: q_dev, tau_dev
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_scale_qmat_float_complex_c(ldq, l_cols, q_dev, tau_dev, my_stream) &
                                                  bind(C, name="cuda_scale_qmat_float_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int), intent(in)  :: ldq, l_cols
      integer(kind=c_intptr_t), value  :: q_dev, tau_dev
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_hvb_a_c(dataType, hvb_dev, a_dev, ld_hvb, lda, my_prow, np_rows, my_pcol, np_cols, &
                                 nblk, ics, ice, SM_count, debug, my_stream) &
                                bind(C, name="cuda_copy_hvb_a_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_intptr_t), value :: hvb_dev, a_dev
      integer(kind=c_int), intent(in) :: ld_hvb, lda, my_prow, np_rows, my_pcol, np_cols, nblk, ics, ice, SM_count, debug
      integer(kind=c_intptr_t), value :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_hvm_hvb_c(dataType, hvb_dev, hvm_dev, ld_hvm, ld_hvb, my_prow, np_rows, nstor, nblk, &
                                ics, ice, SM_count, debug, my_stream) &
                                bind(C, name="cuda_copy_hvm_hvb_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_intptr_t), value :: hvb_dev, hvm_dev
      integer(kind=c_int), intent(in) :: ld_hvm, ld_hvb, my_prow, np_rows, nstor, nblk, ics, ice, SM_count, debug
      integer(kind=c_intptr_t), value :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_update_tmat_c(dataType, tmat_dev, h_dev, tau_curr_dev, max_stored_rows, nc, n, &
                                  SM_count, debug, my_stream) &
                                bind(C, name="cuda_update_tmat_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_intptr_t), value :: tmat_dev, h_dev, tau_curr_dev
      integer(kind=c_int), intent(in) :: max_stored_rows, nc, n, SM_count, debug
      integer(kind=c_intptr_t), value :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_trmv_c(dataType, tmat_dev, h_dev, result_buffer_dev, tau_curr_dev, max_stored_rows, n, &
                          SM_count, debug, my_stream) &
                                bind(C, name="cuda_trmv_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_intptr_t), value :: tmat_dev, h_dev, result_buffer_dev, tau_curr_dev
      integer(kind=c_int), intent(in) :: max_stored_rows, n, SM_count, debug
      integer(kind=c_intptr_t), value :: my_stream
    end subroutine
  end interface


  contains


    subroutine cuda_scale_qmat_double_complex(ldq, l_cols, q_dev, tau_dev, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int), intent(in)    :: ldq, l_cols
      integer(kind=c_intptr_t)           :: q_dev, tau_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      if (present(my_stream)) then
        call cuda_scale_qmat_double_complex_c(ldq, l_cols, q_dev, tau_dev, my_stream)
      else
        call cuda_scale_qmat_double_complex_c(ldq, l_cols, q_dev, tau_dev, my_stream2)
      endif
#endif
    end subroutine

    subroutine cuda_scale_qmat_float_complex(ldq, l_cols, q_dev, tau_dev, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int), intent(in)    :: ldq, l_cols
      integer(kind=c_intptr_t)           :: q_dev, tau_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      if (present(my_stream)) then
        call cuda_scale_qmat_float_complex_c(ldq, l_cols, q_dev, tau_dev, my_stream)
      else
        call cuda_scale_qmat_float_complex_c(ldq, l_cols, q_dev, tau_dev, my_stream2)
      endif
#endif
    end subroutine

    subroutine cuda_copy_hvb_a(dataType, hvb_dev, a_dev, ld_hvb, lda, my_prow, np_rows, my_pcol, np_cols, &
                               nblk, ics, ice, SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_intptr_t), value :: hvb_dev, a_dev
      integer(kind=c_int), intent(in) :: ld_hvb, lda, my_prow, np_rows, my_pcol, np_cols, nblk, ics, ice, SM_count, debug
      integer(kind=c_intptr_t), value :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_hvb_a_c(dataType, hvb_dev, a_dev, ld_hvb, lda, my_prow, np_rows, my_pcol, np_cols, &
                             nblk, ics, ice, SM_count, debug, my_stream)
#endif
    end subroutine

    subroutine cuda_copy_hvm_hvb(dataType, hvb_dev, hvm_dev, ld_hvm, ld_hvb, my_prow, np_rows, nstor, nblk, &
                                ics, ice, SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_intptr_t), value :: hvb_dev, hvm_dev
      integer(kind=c_int), intent(in) :: ld_hvm, ld_hvb, my_prow, np_rows, nstor, nblk, ics, ice, SM_count, debug
      integer(kind=c_intptr_t), value :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_hvm_hvb_c(dataType, hvb_dev, hvm_dev, ld_hvm, ld_hvb, my_prow, np_rows, nstor, nblk, &
                            ics, ice, SM_count, debug, my_stream)
#endif
    end subroutine

    subroutine cuda_update_tmat(dataType, tmat_dev, h_dev, tau_curr_dev, max_stored_rows, nc, n, &
                                SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_intptr_t), value :: tmat_dev, h_dev, tau_curr_dev
      integer(kind=c_int), intent(in) :: max_stored_rows, nc, n, SM_count, debug
      integer(kind=c_intptr_t), value :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_update_tmat_c(dataType, tmat_dev, h_dev, tau_curr_dev, max_stored_rows, nc, n, &
                              SM_count, debug, my_stream)
#endif
    end subroutine

    subroutine cuda_trmv(dataType, tmat_dev, h_dev, result_buffer_dev, tau_curr_dev, max_stored_rows, n, &
                         SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_intptr_t), value :: tmat_dev, h_dev, result_buffer_dev, tau_curr_dev
      integer(kind=c_int), intent(in) :: max_stored_rows, n, SM_count, debug
      integer(kind=c_intptr_t), value :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_trmv_c(dataType, tmat_dev, h_dev, result_buffer_dev, tau_curr_dev, max_stored_rows, n, &
                       SM_count, debug, my_stream)
#endif
    end subroutine

end module
