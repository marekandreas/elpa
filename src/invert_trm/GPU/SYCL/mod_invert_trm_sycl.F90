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

module invert_trm_sycl
  use, intrinsic :: iso_c_binding
  use precision

  implicit none

  public

  interface
    subroutine sycl_copy_double_a_tmat2_c(a_dev, tmat2_dev, nblk, matrixRows, l_cols, &
                                                             l_colx, l_row1, nb, my_stream)&
                                                     bind(C, name="sycl_copy_double_a_tmat2_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: a_dev, tmat2_dev
      integer(kind=c_int), intent(in)  :: nblk, matrixRows, l_cols, l_colx, l_row1, nb
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_double_tmp2_tmat2_c(tmp2_dev, tmat2_dev, nblk, l_col1, &
                                                                 nb, my_stream)&
                                                     bind(C, name="sycl_copy_double_tmp2_tmat2_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: tmp2_dev, tmat2_dev
      integer(kind=c_int), intent(in)  :: nblk, l_col1, nb
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_double_a_tmat1_c(a_dev, tmat1_dev, l_rows, matrixRows, nb, l_row1, &
                                                       l_col1, my_stream)&
                                                     bind(C, name="sycl_copy_double_a_tmat1_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: a_dev, tmat1_dev
      integer(kind=c_int), intent(in)  :: l_rows, matrixRows, nb, l_row1, l_col1
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_double_tmp1_tmp2_c(tmp1_dev, tmp2_dev, nblk, nb, &
                                                       my_stream)&
                                                     bind(C, name="sycl_copy_double_tmp1_tmp2_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: tmp1_dev, tmp2_dev
      integer(kind=c_int), intent(in)  :: nblk, nb
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_double_a_tmp1_c(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, &
                                                       nb, my_stream)&
                                                     bind(C, name="sycl_copy_double_a_tmp1_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: a_dev, tmp1_dev
      integer(kind=c_int), intent(in)  :: l_row1, l_col1, matrixRows, nb
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_float_a_tmat2_c(a_dev, tmat2_dev, nblk, matrixRows, l_cols, &
                                                             l_colx, l_row1, nb, my_stream)&
                                                     bind(C, name="sycl_copy_float_a_tmat2_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: a_dev, tmat2_dev
      integer(kind=c_int), intent(in)  :: nblk, matrixRows, l_cols, l_colx, l_row1, nb
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_float_tmp2_tmat2_c(tmp2_dev, tmat2_dev, nblk, l_col1, &
                                                                 nb, my_stream)&
                                                     bind(C, name="sycl_copy_float_tmp2_tmat2_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: tmp2_dev, tmat2_dev
      integer(kind=c_int), intent(in)  :: nblk, l_col1, nb
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_float_a_tmat1_c(a_dev, tmat1_dev, l_rows, matrixRows, nb, l_row1, &
                                                       l_col1, my_stream)&
                                                     bind(C, name="sycl_copy_float_a_tmat1_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: a_dev, tmat1_dev
      integer(kind=c_int), intent(in)  :: l_rows, matrixRows, nb, l_row1, l_col1
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_float_tmp1_tmp2_c(tmp1_dev, tmp2_dev, nblk, nb, &
                                                       my_stream)&
                                                     bind(C, name="sycl_copy_float_tmp1_tmp2_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: tmp1_dev, tmp2_dev
      integer(kind=c_int), intent(in)  :: nblk, nb
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_float_a_tmp1_c(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, &
                                                       nb, my_stream)&
                                                     bind(C, name="sycl_copy_float_a_tmp1_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: a_dev, tmp1_dev
      integer(kind=c_int), intent(in)  :: l_row1, l_col1, matrixRows, nb
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_double_complex_a_tmat2_c(a_dev, tmat2_dev, nblk, matrixRows, l_cols, &
                                                             l_colx, l_row1, nb, my_stream)&
                                                     bind(C, name="sycl_copy_double_complex_a_tmat2_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: a_dev, tmat2_dev
      integer(kind=c_int), intent(in)  :: nblk, matrixRows, l_cols, l_colx, l_row1, nb
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_double_complex_tmp2_tmat2_c(tmp2_dev, tmat2_dev, nblk, l_col1, &
                                                                 nb, my_stream)&
                                                     bind(C, name="sycl_copy_double_complex_tmp2_tmat2_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: tmp2_dev, tmat2_dev
      integer(kind=c_int), intent(in)  :: nblk, l_col1, nb
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_double_complex_a_tmat1_c(a_dev, tmat1_dev, l_rows, matrixRows, nb, l_row1, &
                                                       l_col1, my_stream)&
                                                     bind(C, name="sycl_copy_double_complex_a_tmat1_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: a_dev, tmat1_dev
      integer(kind=c_int), intent(in)  :: l_rows, matrixRows, nb, l_row1, l_col1
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_double_complex_tmp1_tmp2_c(tmp1_dev, tmp2_dev, nblk, nb, &
                                                       my_stream)&
                                                     bind(C, name="sycl_copy_double_complex_tmp1_tmp2_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: tmp1_dev, tmp2_dev
      integer(kind=c_int), intent(in)  :: nblk, nb
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_double_complex_a_tmp1_c(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, &
                                                       nb, my_stream)&
                                                     bind(C, name="sycl_copy_double_complex_a_tmp1_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: a_dev, tmp1_dev
      integer(kind=c_int), intent(in)  :: l_row1, l_col1, matrixRows, nb
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_float_complex_a_tmat2_c(a_dev, tmat2_dev, nblk, matrixRows, l_cols, &
                                                             l_colx, l_row1, nb, my_stream)&
                                                     bind(C, name="sycl_copy_float_complex_a_tmat2_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: a_dev, tmat2_dev
      integer(kind=c_int), intent(in)  :: nblk, matrixRows, l_cols, l_colx, l_row1, nb
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_float_complex_tmp2_tmat2_c(tmp2_dev, tmat2_dev, nblk, l_col1, &
                                                                 nb, my_stream)&
                                                     bind(C, name="sycl_copy_float_complex_tmp2_tmat2_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: tmp2_dev, tmat2_dev
      integer(kind=c_int), intent(in)  :: nblk, l_col1, nb
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_float_complex_a_tmat1_c(a_dev, tmat1_dev, l_rows, matrixRows, nb, l_row1, &
                                                       l_col1, my_stream)&
                                                     bind(C, name="sycl_copy_float_complex_a_tmat1_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: a_dev, tmat1_dev
      integer(kind=c_int), intent(in)  :: l_rows, matrixRows, nb, l_row1, l_col1
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_float_complex_tmp1_tmp2_c(tmp1_dev, tmp2_dev, nblk, nb, &
                                                       my_stream)&
                                                     bind(C, name="sycl_copy_float_complex_tmp1_tmp2_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: tmp1_dev, tmp2_dev
      integer(kind=c_int), intent(in)  :: nblk, nb
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine sycl_copy_float_complex_a_tmp1_c(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, &
                                                       nb, my_stream)&
                                                     bind(C, name="sycl_copy_float_complex_a_tmp1_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: a_dev, tmp1_dev
      integer(kind=c_int), intent(in)  :: l_row1, l_col1, matrixRows, nb
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  contains
    subroutine sycl_copy_double_a_tmat2(a_dev, tmat2_dev, nblk, matrixRows, l_cols, l_colx, &
                                                            l_row1, nb, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nblk, matrixRows, l_cols, l_colx, l_row1, nb
      integer(kind=C_intptr_T)        :: a_dev, tmat2_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_a_tmat2_c(a_dev, tmat2_dev, nblk, matrixRows, l_cols, l_colx, &
                                                          l_row1, nb, my_stream)
#endif

    end subroutine

    subroutine sycl_copy_double_tmp2_tmat2(tmp2_dev, tmat2_dev, nblk, l_col1, &
                                                             nb, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nblk, l_col1, nb
      integer(kind=C_intptr_T)        :: tmp2_dev, tmat2_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_tmp2_tmat2_c(tmp2_dev, tmat2_dev, nblk, l_col1, nb, &
                                                          my_stream)
#endif

    end subroutine

    subroutine sycl_copy_double_a_tmat1(a_dev, tmat1_dev, l_rows, matrixRows, nb, l_row1, l_col1,&
                                                            my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: l_rows, matrixRows, nb, l_row1, l_col1
      integer(kind=C_intptr_T)        :: a_dev, tmat1_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_a_tmat1_c(a_dev, tmat1_dev, l_rows, matrixRows, nb, l_row1, l_col1, &
                                                          my_stream)
#endif

    end subroutine

    subroutine sycl_copy_double_tmp1_tmp2(tmp1_dev, tmp2_dev, nblk, nb,&
                                                            my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nblk, nb
      integer(kind=C_intptr_T)        :: tmp1_dev, tmp2_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_tmp1_tmp2_c(tmp1_dev, tmp2_dev, nblk, nb, &
                                                            my_stream)
#endif

    end subroutine

    subroutine sycl_copy_double_a_tmp1(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb,&
                                                            my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: l_row1, l_col1, matrixRows, nb
      integer(kind=C_intptr_T)        :: a_dev, tmp1_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_a_tmp1_c(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb, &
                                                            my_stream)
#endif

    end subroutine

    subroutine sycl_copy_float_a_tmat2(a_dev, tmat2_dev, nblk, matrixRows, l_cols, l_colx, &
                                                            l_row1, nb, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nblk, matrixRows, l_cols, l_colx, l_row1, nb
      integer(kind=C_intptr_T)        :: a_dev, tmat2_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_a_tmat2_c(a_dev, tmat2_dev, nblk, matrixRows, l_cols, l_colx, &
                                                          l_row1, nb, my_stream)
#endif

    end subroutine

    subroutine sycl_copy_float_tmp2_tmat2(tmp2_dev, tmat2_dev, nblk, l_col1, &
                                                             nb, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nblk, l_col1, nb
      integer(kind=C_intptr_T)        :: tmp2_dev, tmat2_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_tmp2_tmat2_c(tmp2_dev, tmat2_dev, nblk, l_col1, nb, &
                                                          my_stream)
#endif

    end subroutine

    subroutine sycl_copy_float_a_tmat1(a_dev, tmat1_dev, l_rows, matrixRows, nb, l_row1, l_col1,&
                                                            my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: l_rows, matrixRows, nb, l_row1, l_col1
      integer(kind=C_intptr_T)        :: a_dev, tmat1_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_a_tmat1_c(a_dev, tmat1_dev, l_rows, matrixRows, nb, l_row1, l_col1, &
                                                          my_stream)
#endif

    end subroutine

    subroutine sycl_copy_float_tmp1_tmp2(tmp1_dev, tmp2_dev, nblk, nb,&
                                                            my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nblk, nb
      integer(kind=C_intptr_T)        :: tmp1_dev, tmp2_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_tmp1_tmp2_c(tmp1_dev, tmp2_dev, nblk, nb, &
                                                            my_stream)
#endif

    end subroutine

    subroutine sycl_copy_float_a_tmp1(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb,&
                                                            my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: l_row1, l_col1, matrixRows, nb
      integer(kind=C_intptr_T)        :: a_dev, tmp1_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_a_tmp1_c(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb, &
                                                            my_stream)
#endif

    end subroutine

    subroutine sycl_copy_double_complex_a_tmat2(a_dev, tmat2_dev, nblk, matrixRows, l_cols, l_colx, &
                                                            l_row1, nb, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nblk, matrixRows, l_cols, l_colx, l_row1, nb
      integer(kind=C_intptr_T)        :: a_dev, tmat2_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_complex_a_tmat2_c(a_dev, tmat2_dev, nblk, matrixRows, l_cols, l_colx, &
                                                          l_row1, nb, my_stream)
#endif

    end subroutine

    subroutine sycl_copy_double_complex_tmp2_tmat2(tmp2_dev, tmat2_dev, nblk, l_col1, &
                                                             nb, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nblk, l_col1, nb
      integer(kind=C_intptr_T)        :: tmp2_dev, tmat2_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_complex_tmp2_tmat2_c(tmp2_dev, tmat2_dev, nblk, l_col1, nb, &
                                                          my_stream)
#endif

    end subroutine

    subroutine sycl_copy_double_complex_a_tmat1(a_dev, tmat1_dev, l_rows, matrixRows, nb, l_row1, l_col1,&
                                                            my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: l_rows, matrixRows, nb, l_row1, l_col1
      integer(kind=C_intptr_T)        :: a_dev, tmat1_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_complex_a_tmat1_c(a_dev, tmat1_dev, l_rows, matrixRows, nb, l_row1, l_col1, &
                                                          my_stream)
#endif

    end subroutine

    subroutine sycl_copy_double_complex_tmp1_tmp2(tmp1_dev, tmp2_dev, nblk, nb,&
                                                            my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nblk, nb
      integer(kind=C_intptr_T)        :: tmp1_dev, tmp2_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_complex_tmp1_tmp2_c(tmp1_dev, tmp2_dev, nblk, nb, &
                                                            my_stream)
#endif

    end subroutine

    subroutine sycl_copy_double_complex_a_tmp1(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb,&
                                                            my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: l_row1, l_col1, matrixRows, nb
      integer(kind=C_intptr_T)        :: a_dev, tmp1_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_complex_a_tmp1_c(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb, &
                                                            my_stream)
#endif

    end subroutine

    subroutine sycl_copy_float_complex_a_tmat2(a_dev, tmat2_dev, nblk, matrixRows, l_cols, l_colx, &
                                                            l_row1, nb, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nblk, matrixRows, l_cols, l_colx, l_row1, nb
      integer(kind=C_intptr_T)        :: a_dev, tmat2_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_complex_a_tmat2_c(a_dev, tmat2_dev, nblk, matrixRows, l_cols, l_colx, &
                                                          l_row1, nb, my_stream)
#endif

    end subroutine

    subroutine sycl_copy_float_complex_tmp2_tmat2(tmp2_dev, tmat2_dev, nblk, l_col1, &
                                                             nb, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nblk, l_col1, nb
      integer(kind=C_intptr_T)        :: tmp2_dev, tmat2_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_complex_tmp2_tmat2_c(tmp2_dev, tmat2_dev, nblk, l_col1, nb, &
                                                          my_stream)
#endif

    end subroutine

    subroutine sycl_copy_float_complex_a_tmat1(a_dev, tmat1_dev, l_rows, matrixRows, nb, l_row1, l_col1,&
                                                            my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: l_rows, matrixRows, nb, l_row1, l_col1
      integer(kind=C_intptr_T)        :: a_dev, tmat1_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_complex_a_tmat1_c(a_dev, tmat1_dev, l_rows, matrixRows, nb, l_row1, l_col1, &
                                                          my_stream)
#endif

    end subroutine

    subroutine sycl_copy_float_complex_tmp1_tmp2(tmp1_dev, tmp2_dev, nblk, nb,&
                                                            my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nblk, nb
      integer(kind=C_intptr_T)        :: tmp1_dev, tmp2_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_complex_tmp1_tmp2_c(tmp1_dev, tmp2_dev, nblk, nb, &
                                                            my_stream)
#endif

    end subroutine

    subroutine sycl_copy_float_complex_a_tmp1(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb,&
                                                            my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: l_row1, l_col1, matrixRows, nb
      integer(kind=C_intptr_T)        :: a_dev, tmp1_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_complex_a_tmp1_c(a_dev, tmp1_dev, l_row1, l_col1, matrixRows, nb, &
                                                            my_stream)
#endif

    end subroutine

end module
