#if 0
!    Copyright 2023, A. Marek, MPCDF
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
!    This particular source code file contains additions, changes and
!    enhancements authored by Intel Corporation which is not part of
!    the ELPA consortium.
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

module multiply_a_b_cuda
  use, intrinsic :: iso_c_binding
  use precision

  implicit none

  public

  interface
    subroutine cuda_copy_double_tmp2_c_intptr_c(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, &
                                                                     ldc, ldcCols, my_stream)&
                                                     bind(C, name="cuda_copy_double_tmp2_c_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: tmp2_dev, c_dev
      integer(kind=c_int), intent(in)  :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_double_tmp2_c_cptr_c(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, &
                                                                     ldc, ldcCols, my_stream)&
                                                     bind(C, name="cuda_copy_double_tmp2_c_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: tmp2_dev
      type(c_ptr), value               :: c_dev
      integer(kind=c_int), intent(in)  :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_double_a_aux_bc_intptr_c(a_dev, aux_bc_dev, n_aux_bc, &
                                                     nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                                     lda, ldaCols, my_stream)&
                                                     bind(C, name="cuda_copy_double_a_aux_bc_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: a_dev, aux_bc_dev
      integer(kind=c_int), intent(in)  :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_double_a_aux_bc_cptr_c(a_dev, aux_bc_dev, n_aux_bc, &
                                                     nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                                     lda, ldaCols, my_stream)&
                                                     bind(C, name="cuda_copy_double_a_aux_bc_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value               :: a_dev
      integer(kind=C_intptr_T), value  :: aux_bc_dev
      integer(kind=c_int), intent(in)  :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_double_aux_bc_aux_mat_c(aux_bc_dev, aux_mat_dev, lrs, &
                                                          lre, nstor, n_aux_bc, nvals, l_rows,  &
                                                                     nblk, nblk_mult, my_stream)&
                          bind(C, name="cuda_copy_double_aux_bc_aux_mat_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: aux_bc_dev, aux_mat_dev
      integer(kind=c_int), intent(in)  :: lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_float_tmp2_c_intptr_c(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, &
                                                                     ldc, ldcCols, my_stream)&
                                                     bind(C, name="cuda_copy_float_tmp2_c_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: tmp2_dev, c_dev
      integer(kind=c_int), intent(in)  :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_float_tmp2_c_cptr_c(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, &
                                                                     ldc, ldcCols, my_stream)&
                                                     bind(C, name="cuda_copy_float_tmp2_c_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: tmp2_dev
      type(c_ptr), value               :: c_dev
      integer(kind=c_int), intent(in)  :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_float_a_aux_bc_intptr_c(a_dev, aux_bc_dev, n_aux_bc, &
                                                     nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                                     lda, ldaCols, my_stream)&
                                                     bind(C, name="cuda_copy_float_a_aux_bc_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: a_dev, aux_bc_dev
      integer(kind=c_int), intent(in)  :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_float_a_aux_bc_cptr_c(a_dev, aux_bc_dev, n_aux_bc, &
                                                     nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                                     lda, ldaCols, my_stream)&
                                                     bind(C, name="cuda_copy_float_a_aux_bc_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value               :: a_dev
      integer(kind=C_intptr_T), value  :: aux_bc_dev
      integer(kind=c_int), intent(in)  :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_float_aux_bc_aux_mat_c(aux_bc_dev, aux_mat_dev, lrs, &
                                                          lre, nstor, n_aux_bc, nvals, l_rows,  &
                                                                     nblk, nblk_mult, my_stream)&
                          bind(C, name="cuda_copy_float_aux_bc_aux_mat_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: aux_bc_dev, aux_mat_dev
      integer(kind=c_int), intent(in)  :: lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_double_complex_tmp2_c_intptr_c(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, &
                                                                     ldc, ldcCols, my_stream)&
                                                     bind(C, name="cuda_copy_double_complex_tmp2_c_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: tmp2_dev, c_dev
      integer(kind=c_int), intent(in)  :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_double_complex_tmp2_c_cptr_c(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, &
                                                                     ldc, ldcCols, my_stream)&
                                                     bind(C, name="cuda_copy_double_complex_tmp2_c_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: tmp2_dev
      type(c_ptr), value               :: c_dev
      integer(kind=c_int), intent(in)  :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_double_complex_a_aux_bc_intptr_c(a_dev, aux_bc_dev, n_aux_bc, &
                                                     nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                                     lda, ldaCols, my_stream)&
                                                     bind(C, name="cuda_copy_double_complex_a_aux_bc_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: a_dev, aux_bc_dev
      integer(kind=c_int), intent(in)  :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_double_complex_a_aux_bc_cptr_c(a_dev, aux_bc_dev, n_aux_bc, &
                                                     nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                                     lda, ldaCols, my_stream)&
                                                     bind(C, name="cuda_copy_double_complex_a_aux_bc_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value               :: a_dev
      integer(kind=C_intptr_T), value  :: aux_bc_dev
      integer(kind=c_int), intent(in)  :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_double_complex_aux_bc_aux_mat_c(aux_bc_dev, aux_mat_dev, lrs, &
                                                          lre, nstor, n_aux_bc, nvals, l_rows,  &
                                                                     nblk, nblk_mult, my_stream)&
                          bind(C, name="cuda_copy_double_complex_aux_bc_aux_mat_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: aux_bc_dev, aux_mat_dev
      integer(kind=c_int), intent(in)  :: lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_float_complex_tmp2_c_intptr_c(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, &
                                                                     ldc, ldcCols, my_stream)&
                                                     bind(C, name="cuda_copy_float_complex_tmp2_c_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: tmp2_dev, c_dev
      integer(kind=c_int), intent(in)  :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_float_complex_tmp2_c_cptr_c(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, &
                                                                     ldc, ldcCols, my_stream)&
                                                     bind(C, name="cuda_copy_float_complex_tmp2_c_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: tmp2_dev
      type(c_ptr), value               :: c_dev
      integer(kind=c_int), intent(in)  :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_float_complex_a_aux_bc_intptr_c(a_dev, aux_bc_dev, n_aux_bc, &
                                                     nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                                     lda, ldaCols, my_stream)&
                                                     bind(C, name="cuda_copy_float_complex_a_aux_bc_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: a_dev, aux_bc_dev
      integer(kind=c_int), intent(in)  :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_float_complex_a_aux_bc_cptr_c(a_dev, aux_bc_dev, n_aux_bc, &
                                                     nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                                     lda, ldaCols, my_stream)&
                                                     bind(C, name="cuda_copy_float_complex_a_aux_bc_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value               :: a_dev
      integer(kind=C_intptr_T), value  :: aux_bc_dev
      integer(kind=c_int), intent(in)  :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine cuda_copy_float_complex_aux_bc_aux_mat_c(aux_bc_dev, aux_mat_dev, lrs, &
                                                          lre, nstor, n_aux_bc, nvals, l_rows,  &
                                                                     nblk, nblk_mult, my_stream)&
                          bind(C, name="cuda_copy_float_complex_aux_bc_aux_mat_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: aux_bc_dev, aux_mat_dev
      integer(kind=c_int), intent(in)  :: lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  contains

    subroutine cuda_copy_double_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, &
                                                              ldcCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=C_intptr_T)        :: tmp2_dev, c_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_tmp2_c_intptr_c(tmp2_dev, c_dev, nr_done, nstor, lcs, &
                                                                lce, ldc, ldcCols, my_stream)
#endif

    end subroutine

    subroutine cuda_copy_double_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, &
                                                              ldcCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=C_intptr_T)        :: tmp2_dev
      type(c_ptr)                     :: c_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_tmp2_c_cptr_c(tmp2_dev, c_dev, nr_done, nstor, lcs, &
                                                                lce, ldc, ldcCols, my_stream)
#endif

    end subroutine

    subroutine cuda_copy_double_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, &
                                                         nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                              lda, ldaCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=C_intptr_T)        :: a_dev, aux_bc_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_a_aux_bc_intptr_c(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, &
                                                                  noff, nblk, n, l_rows, lda, ldaCols, my_stream)
#endif

    end subroutine

    subroutine cuda_copy_double_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, &
                                                                     lrs, lre, noff, nblk, n, l_rows, &
                                                              lda, ldaCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      type(c_ptr)                     :: a_dev
      integer(kind=C_intptr_T)        :: aux_bc_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_a_aux_bc_cptr_c(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, &
                                                                  noff, nblk, n, l_rows, lda, ldaCols, my_stream)
#endif

    end subroutine

    subroutine cuda_copy_double_aux_bc_aux_mat( aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, &
                                                              l_rows, nblk, nblk_mult, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult
      integer(kind=C_intptr_T)        :: aux_bc_dev, aux_mat_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_aux_bc_aux_mat_c(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, &
                                                                 l_rows, nblk, nblk_mult, my_stream)
#endif

    end subroutine
    subroutine cuda_copy_float_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, &
                                                              ldcCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=C_intptr_T)        :: tmp2_dev, c_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_tmp2_c_intptr_c(tmp2_dev, c_dev, nr_done, nstor, lcs, &
                                                                lce, ldc, ldcCols, my_stream)
#endif

    end subroutine

    subroutine cuda_copy_float_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, &
                                                              ldcCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=C_intptr_T)        :: tmp2_dev
      type(c_ptr)                     :: c_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_tmp2_c_cptr_c(tmp2_dev, c_dev, nr_done, nstor, lcs, &
                                                                lce, ldc, ldcCols, my_stream)
#endif

    end subroutine

    subroutine cuda_copy_float_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, &
                                                         nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                              lda, ldaCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=C_intptr_T)        :: a_dev, aux_bc_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_a_aux_bc_intptr_c(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, &
                                                                  noff, nblk, n, l_rows, lda, ldaCols, my_stream)
#endif

    end subroutine

    subroutine cuda_copy_float_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, &
                                                                     lrs, lre, noff, nblk, n, l_rows, &
                                                              lda, ldaCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      type(c_ptr)                     :: a_dev
      integer(kind=C_intptr_T)        :: aux_bc_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_a_aux_bc_cptr_c(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, &
                                                                  noff, nblk, n, l_rows, lda, ldaCols, my_stream)
#endif

    end subroutine

    subroutine cuda_copy_float_aux_bc_aux_mat( aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, &
                                                              l_rows, nblk, nblk_mult, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult
      integer(kind=C_intptr_T)        :: aux_bc_dev, aux_mat_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_aux_bc_aux_mat_c(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, &
                                                                 l_rows, nblk, nblk_mult, my_stream)
#endif

    end subroutine
    subroutine cuda_copy_double_complex_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, &
                                                              ldcCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=C_intptr_T)        :: tmp2_dev, c_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_complex_tmp2_c_intptr_c(tmp2_dev, c_dev, nr_done, nstor, lcs, &
                                                                lce, ldc, ldcCols, my_stream)
#endif

    end subroutine

    subroutine cuda_copy_double_complex_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, &
                                                              ldcCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=C_intptr_T)        :: tmp2_dev
      type(c_ptr)                     :: c_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_complex_tmp2_c_cptr_c(tmp2_dev, c_dev, nr_done, nstor, lcs, &
                                                                lce, ldc, ldcCols, my_stream)
#endif

    end subroutine

    subroutine cuda_copy_double_complex_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, &
                                                         nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                              lda, ldaCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=C_intptr_T)        :: a_dev, aux_bc_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_complex_a_aux_bc_intptr_c(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, &
                                                                  noff, nblk, n, l_rows, lda, ldaCols, my_stream)
#endif

    end subroutine

    subroutine cuda_copy_double_complex_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, &
                                                                     lrs, lre, noff, nblk, n, l_rows, &
                                                              lda, ldaCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      type(c_ptr)                     :: a_dev
      integer(kind=C_intptr_T)        :: aux_bc_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_complex_a_aux_bc_cptr_c(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, &
                                                                  noff, nblk, n, l_rows, lda, ldaCols, my_stream)
#endif

    end subroutine

    subroutine cuda_copy_double_complex_aux_bc_aux_mat( aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, &
                                                              l_rows, nblk, nblk_mult, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult
      integer(kind=C_intptr_T)        :: aux_bc_dev, aux_mat_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_complex_aux_bc_aux_mat_c(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, &
                                                                 l_rows, nblk, nblk_mult, my_stream)
#endif

    end subroutine
    subroutine cuda_copy_float_complex_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, &
                                                              ldcCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=C_intptr_T)        :: tmp2_dev, c_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_complex_tmp2_c_intptr_c(tmp2_dev, c_dev, nr_done, nstor, lcs, &
                                                                lce, ldc, ldcCols, my_stream)
#endif

    end subroutine

    subroutine cuda_copy_float_complex_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, &
                                                              ldcCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=C_intptr_T)        :: tmp2_dev
      type(c_ptr)                     :: c_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_complex_tmp2_c_cptr_c(tmp2_dev, c_dev, nr_done, nstor, lcs, &
                                                                lce, ldc, ldcCols, my_stream)
#endif

    end subroutine

    subroutine cuda_copy_float_complex_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, &
                                                         nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                              lda, ldaCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=C_intptr_T)        :: a_dev, aux_bc_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_complex_a_aux_bc_intptr_c(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, &
                                                                  noff, nblk, n, l_rows, lda, ldaCols, my_stream)
#endif

    end subroutine

    subroutine cuda_copy_float_complex_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, &
                                                                     lrs, lre, noff, nblk, n, l_rows, &
                                                              lda, ldaCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      type(c_ptr)                     :: a_dev
      integer(kind=C_intptr_T)        :: aux_bc_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_complex_a_aux_bc_cptr_c(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, &
                                                                  noff, nblk, n, l_rows, lda, ldaCols, my_stream)
#endif

    end subroutine

    subroutine cuda_copy_float_complex_aux_bc_aux_mat( aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, &
                                                              l_rows, nblk, nblk_mult, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult
      integer(kind=C_intptr_T)        :: aux_bc_dev, aux_mat_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_complex_aux_bc_aux_mat_c(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, &
                                                                 l_rows, nblk, nblk_mult, my_stream)
#endif

    end subroutine
end module
