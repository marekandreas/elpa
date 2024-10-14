#if 0
!    Copyright 2021, A. Marek, MPCDF
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
module multiply_a_b_gpu
  use, intrinsic :: iso_c_binding
  use precision

#ifdef WITH_NVIDIA_GPU_VERSION
  use multiply_a_b_cuda

#endif
#ifdef WITH_AMD_GPU_VERSION
  use multiply_a_b_hip
#endif
#ifdef WITH_SYCL_GPU_VERSION
  use multiply_a_b_sycl
#endif

  implicit none

  public
  !This file is auto-generated do NOT edit!

  interface gpu_copy_double_a_aux_bc
    module procedure gpu_copy_double_a_aux_bc_intptr
    module procedure gpu_copy_double_a_aux_bc_cptr
  end interface

  interface gpu_copy_double_tmp2_c
    module procedure gpu_copy_double_tmp2_c_intptr
    module procedure gpu_copy_double_tmp2_c_cptr
  end interface
  
  interface gpu_copy_float_a_aux_bc
    module procedure gpu_copy_float_a_aux_bc_intptr
    module procedure gpu_copy_float_a_aux_bc_cptr
  end interface

  interface gpu_copy_float_tmp2_c
    module procedure gpu_copy_float_tmp2_c_intptr
    module procedure gpu_copy_float_tmp2_c_cptr
  end interface
  
  interface gpu_copy_double_complex_a_aux_bc
    module procedure gpu_copy_double_complex_a_aux_bc_intptr
    module procedure gpu_copy_double_complex_a_aux_bc_cptr
  end interface

  interface gpu_copy_double_complex_tmp2_c
    module procedure gpu_copy_double_complex_tmp2_c_intptr
    module procedure gpu_copy_double_complex_tmp2_c_cptr
  end interface
  
  interface gpu_copy_float_complex_a_aux_bc
    module procedure gpu_copy_float_complex_a_aux_bc_intptr
    module procedure gpu_copy_float_complex_a_aux_bc_cptr
  end interface

  interface gpu_copy_float_complex_tmp2_c
    module procedure gpu_copy_float_complex_tmp2_c_intptr
    module procedure gpu_copy_float_complex_tmp2_c_cptr
  end interface

  contains

    subroutine gpu_copy_double_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=C_intptr_T)        :: tmp2_dev, c_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_double_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_double_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=C_intptr_T)        :: tmp2_dev
      type(c_ptr)                     :: c_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_double_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_double_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                        lda, ldaCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=C_intptr_T)        :: a_dev, aux_bc_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_double_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_double_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                        lda, ldaCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      type(c_ptr)                     :: a_dev
      integer(kind=C_intptr_T)        :: aux_bc_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_double_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_double_aux_bc_aux_mat(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, &
                                        l_rows, nblk, nblk_mult, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult
      integer(kind=C_intptr_T)        :: aux_mat_dev, aux_bc_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_aux_bc_aux_mat(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, l_rows, &
                                                      nblk, nblk_mult, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_double_aux_bc_aux_mat(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, l_rows, &
                                                      nblk, nblk_mult, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_aux_bc_aux_mat(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, l_rows, &
                                                      nblk, nblk_mult, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_float_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=C_intptr_T)        :: tmp2_dev, c_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_float_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_float_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=C_intptr_T)        :: tmp2_dev
      type(c_ptr)                     :: c_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_float_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_float_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                        lda, ldaCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=C_intptr_T)        :: a_dev, aux_bc_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_float_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_float_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                        lda, ldaCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      type(c_ptr)                     :: a_dev
      integer(kind=C_intptr_T)        :: aux_bc_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_float_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_float_aux_bc_aux_mat(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, &
                                        l_rows, nblk, nblk_mult, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult
      integer(kind=C_intptr_T)        :: aux_mat_dev, aux_bc_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_aux_bc_aux_mat(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, l_rows, &
                                                      nblk, nblk_mult, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_float_aux_bc_aux_mat(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, l_rows, &
                                                      nblk, nblk_mult, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_aux_bc_aux_mat(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, l_rows, &
                                                      nblk, nblk_mult, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_double_complex_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=C_intptr_T)        :: tmp2_dev, c_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_complex_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_double_complex_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_complex_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_double_complex_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=C_intptr_T)        :: tmp2_dev
      type(c_ptr)                     :: c_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_complex_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_double_complex_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_complex_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_double_complex_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                        lda, ldaCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=C_intptr_T)        :: a_dev, aux_bc_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_complex_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_double_complex_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_complex_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_double_complex_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                        lda, ldaCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      type(c_ptr)                     :: a_dev
      integer(kind=C_intptr_T)        :: aux_bc_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_complex_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_double_complex_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_complex_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_double_complex_aux_bc_aux_mat(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, &
                                        l_rows, nblk, nblk_mult, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult
      integer(kind=C_intptr_T)        :: aux_mat_dev, aux_bc_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_complex_aux_bc_aux_mat(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, l_rows, &
                                                      nblk, nblk_mult, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_double_complex_aux_bc_aux_mat(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, l_rows, &
                                                      nblk, nblk_mult, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_complex_aux_bc_aux_mat(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, l_rows, &
                                                      nblk, nblk_mult, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_float_complex_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=C_intptr_T)        :: tmp2_dev, c_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_complex_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_float_complex_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_complex_tmp2_c_intptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_float_complex_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=C_intptr_T)        :: tmp2_dev
      type(c_ptr)                     :: c_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_complex_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_float_complex_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_complex_tmp2_c_cptr(tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_float_complex_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                        lda, ldaCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=C_intptr_T)        :: a_dev, aux_bc_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_complex_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_float_complex_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_complex_a_aux_bc_intptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_float_complex_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                        lda, ldaCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      type(c_ptr)                     :: a_dev
      integer(kind=C_intptr_T)        :: aux_bc_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_complex_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_float_complex_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_complex_a_aux_bc_cptr(a_dev, aux_bc_dev, n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, &
                                                       lda, ldaCols, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine

    subroutine gpu_copy_float_complex_aux_bc_aux_mat(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, &
                                        l_rows, nblk, nblk_mult, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult
      integer(kind=C_intptr_T)        :: aux_mat_dev, aux_bc_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_complex_aux_bc_aux_mat(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, l_rows, &
                                                      nblk, nblk_mult, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_float_complex_aux_bc_aux_mat(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, l_rows, &
                                                      nblk, nblk_mult, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_complex_aux_bc_aux_mat(aux_bc_dev, aux_mat_dev, lrs, lre, nstor, n_aux_bc, nvals, l_rows, &
                                                      nblk, nblk_mult, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif

    end subroutine


    subroutine gpu_copy_aux_full(dataType, lhs_dev, rhs_dev, l_rows, l_cols, lld_lhs, lld_rhs, debug, my_stream)
                                               
      use, intrinsic :: iso_c_binding

      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_intptr_t)        :: lhs_dev, rhs_dev
      integer(kind=c_int), intent(in) :: l_rows, l_cols, lld_lhs, lld_rhs, debug
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_aux_full(dataType, lhs_dev, rhs_dev, l_rows, l_cols, lld_lhs, lld_rhs, debug, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_aux_full (dataType, lhs_dev, rhs_dev, l_rows, l_cols, lld_lhs, lld_rhs, debug, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_aux_full(dataType, lhs_dev, rhs_dev, l_rows, l_cols, lld_lhs, lld_rhs, debug, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif
    end subroutine


    subroutine gpu_copy_and_set_zeros_aux_full(dataType, mat_dev, aux_mat_full_dev, &
                                               l_rows, l_cols, nblk_mult, debug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_intptr_t)        :: mat_dev, aux_mat_full_dev
      integer(kind=c_int), intent(in) :: l_rows, l_cols, nblk_mult, debug
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_and_set_zeros_aux_full(dataType, mat_dev, aux_mat_full_dev, &
                                            l_rows, l_cols, nblk_mult, debug, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_and_set_zeros_aux_full(dataType, mat_dev, aux_mat_full_dev, &
                                            l_rows, l_cols, nblk_mult, debug, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_and_set_zeros_aux_full(dataType, mat_dev, aux_mat_full_dev, &
                                            l_rows, l_cols, nblk_mult, debug, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif
    end subroutine


    subroutine gpu_copy_and_set_zeros_aux_a_full(dataType, mat_dev, aux_mat_full_dev, l_rows, l_cols, &
                                                 nblk_mult_cols, nblk, np_bc_fine, np_cols_fine, np_cols, debug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_intptr_t)        :: mat_dev, aux_mat_full_dev
      integer(kind=c_int), intent(in) :: l_rows, l_cols, nblk_mult_cols, nblk, np_bc_fine, np_cols_fine, np_cols, debug
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_and_set_zeros_aux_a_full(dataType, mat_dev, aux_mat_full_dev, l_rows, l_cols, &
                                              nblk_mult_cols, nblk, np_bc_fine, np_cols_fine, np_cols, debug, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_and_set_zeros_aux_a_full(dataType, mat_dev, aux_mat_full_dev, l_rows, l_cols, &
                                             nblk_mult_cols, nblk, np_bc_fine, np_cols_fine, np_cols, debug, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_and_set_zeros_aux_a_full(dataType, mat_dev, aux_mat_full_dev, l_rows, l_cols, &
                                              nblk_mult_cols, nblk, np_bc_fine, np_cols_fine, np_cols, debug, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif
    end subroutine


    subroutine gpu_copy_and_set_zeros_aux_b_full(dataType, mat_dev, aux_mat_full_dev, l_rows, l_cols, nblk_mult, &
                                                 nblk_mult_rows, nblk, np_fine, np_rows_fine, np_rows, SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_intptr_t)        :: mat_dev, aux_mat_full_dev
      integer(kind=c_int), intent(in) :: l_rows, l_cols, nblk_mult, nblk_mult_rows, nblk, np_fine, np_rows_fine, np_rows, &
                                         SM_count, debug
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_and_set_zeros_aux_b_full(dataType, mat_dev, aux_mat_full_dev, l_rows, l_cols, nblk_mult, &
                                              nblk_mult_rows, nblk, np_fine, np_rows_fine, np_rows, SM_count, debug, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_and_set_zeros_aux_b_full(dataType, mat_dev, aux_mat_full_dev, l_rows, l_cols, nblk_mult, &
                                             nblk_mult_rows, nblk, np_fine, np_rows_fine, np_rows, SM_count, debug, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_and_set_zeros_aux_b_full(dataType, mat_dev, aux_mat_full_dev, l_rows, l_cols, nblk_mult, &
                                              nblk_mult_rows, nblk, np_fine, np_rows_fine, np_rows, SM_count, debug, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif
    end subroutine


    subroutine gpu_ccl_copy_buf_send(dataType, a_dev, buf_send_dev, l_rows, l_cols, lld_buf, nblk, &
                                     i_block_loc_fine_max, j_block_loc_fine_max, np_fine, np_bc_fine, &
                                     np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_intptr_t)        :: a_dev, buf_send_dev
      integer(kind=c_int), intent(in) :: l_rows, l_cols, lld_buf, nblk, i_block_loc_fine_max, j_block_loc_fine_max, &
                                        np_fine, np_bc_fine, np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_ccl_copy_buf_send(dataType, a_dev, buf_send_dev, l_rows, l_cols, lld_buf, nblk, &
                                  i_block_loc_fine_max, j_block_loc_fine_max, np_fine, np_bc_fine, &
                                  np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_ccl_copy_buf_send(dataType, a_dev, buf_send_dev, l_rows, l_cols, lld_buf, nblk, &
                                 i_block_loc_fine_max, j_block_loc_fine_max, np_fine, np_bc_fine, &
                                 np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_ccl_copy_buf_send(dataType, a_dev, buf_send_dev, l_rows, l_cols, lld_buf, nblk, &
                                  i_block_loc_fine_max, j_block_loc_fine_max, np_fine, np_bc_fine, &
                                  np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif
    end subroutine


    subroutine gpu_ccl_copy_buf_recv(dataType, at_col_dev, buf_recv_dev, l_rows, l_cols, lld_buf, nblk, &
                                     i_block_loc_fine_max, j_block_loc_fine_max, np_fine, np_bc_fine, &
                                     np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_intptr_t)        :: at_col_dev, buf_recv_dev
      integer(kind=c_int), intent(in) :: l_rows, l_cols, lld_buf, nblk, i_block_loc_fine_max, j_block_loc_fine_max, &
                                         np_fine, np_bc_fine, np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_ccl_copy_buf_recv(dataType, at_col_dev, buf_recv_dev, l_rows, l_cols, lld_buf, nblk, &
                                  i_block_loc_fine_max, j_block_loc_fine_max, np_fine, np_bc_fine, &
                                  np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_ccl_copy_buf_recv(dataType, at_col_dev, buf_recv_dev, l_rows, l_cols, lld_buf, nblk, &
                                 i_block_loc_fine_max, j_block_loc_fine_max, np_fine, np_bc_fine, &
                                 np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_ccl_copy_buf_recv(dataType, at_col_dev, buf_recv_dev, l_rows, l_cols, lld_buf, nblk, &
                                  i_block_loc_fine_max, j_block_loc_fine_max, np_fine, np_bc_fine, &
                                  np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print * "NOT implemented yet"
      stop 1
#endif
    end subroutine


    subroutine gpu_copy_and_set_zeros_aux_ab_full_tn_nt(dataType, a_transposed, &
                                                  a_dev, b_dev, aux_a_full_dev, aux_b_full_dev, &
                                                  l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                                  np_ab_fine, np_rows, my_prow, &
                                                  np_t_fine , np_cols, my_pcol, &
                                                  np_dirs_fine, SM_count, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_int), intent(in) :: a_transposed
      integer(kind=c_intptr_t), value :: a_dev, b_dev, aux_a_full_dev, aux_b_full_dev
      integer(kind=c_int), intent(in) :: l_rows, l_cols, nblk_mult_max, nblk_mult, nblk
      integer(kind=c_int), intent(in) :: np_ab_fine, np_rows, my_prow
      integer(kind=c_int), intent(in) :: np_t_fine , np_cols, my_pcol
      integer(kind=c_int), intent(in) :: np_dirs_fine
      integer(kind=c_int), intent(in) :: SM_count, debug
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_and_set_zeros_aux_ab_full_tn_nt_c(dataType, a_transposed, &
                                                    a_dev, b_dev, aux_a_full_dev, aux_b_full_dev, &
                                                    l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                                    np_ab_fine, np_rows, my_prow, &
                                                    np_t_fine , np_cols, my_pcol, &
                                                    np_dirs_fine, SM_count, debug, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_and_set_zeros_aux_ab_full_tn_nt_c(dataType, a_transposed, &
                                                    a_dev, b_dev, aux_a_full_dev, aux_b_full_dev, &
                                                    l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                                    np_ab_fine, np_rows, my_prow, &
                                                    np_t_fine , np_cols, my_pcol, &
                                                    np_dirs_fine, SM_count, debug, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_and_set_zeros_aux_ab_full_tn_nt_c(dataType, a_transposed, &
                                                    a_dev, b_dev, aux_a_full_dev, aux_b_full_dev, &
                                                    l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                                    np_ab_fine, np_rows, my_prow, &
                                                    np_t_fine , np_cols, my_pcol, &
                                                    np_dirs_fine, SM_count, debug, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *, "NOT implemented yet"
      stop 1
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
      character(1, c_char), value     :: dataType
      integer(kind=c_int), intent(in) :: a_transposed
      integer(kind=c_intptr_t), value :: c_dev, tmp1_full_dev
      integer(kind=c_int), intent(in) :: beta_int
      integer(kind=c_int), intent(in) :: l_rows, l_cols, nblk_mult_max, nblk_mult, nblk
      integer(kind=c_int), intent(in) :: np_rows, np_cols, np_dirs_fine
      integer(kind=c_int), intent(in) :: np_dirs_t, my_pdir_t, np_fine
      integer(kind=c_int), intent(in) :: SM_count, debug
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_update_c_tn_nt_c(dataType, a_transposed, &
                                c_dev, tmp1_full_dev, beta_int, &
                                l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                np_rows, np_cols, np_dirs_fine, &
                                np_dirs_t, my_pdir_t, np_fine, &
                                SM_count, debug, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_update_c_tn_nt_c(dataType, a_transposed, &
                                c_dev, tmp1_full_dev, beta_int, &
                                l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                np_rows, np_cols, np_dirs_fine, &
                                np_dirs_t, my_pdir_t, np_fine, &
                                SM_count, debug, my_stream)
#endif

#ifdef WITH_SYCL_GPU_VERSION
      call sycl_update_c_tn_nt_c(dataType, a_transposed, &
                                c_dev, tmp1_full_dev, beta_int, &
                                l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                np_rows, np_cols, np_dirs_fine, &
                                np_dirs_t, my_pdir_t, np_fine, &
                                SM_count, debug, my_stream)
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *, "NOT implemented yet"
      stop 1
#endif
end subroutine

end module
