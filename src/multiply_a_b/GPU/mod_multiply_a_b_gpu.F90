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


module multiply_a_b_gpu
  use, intrinsic :: iso_c_binding
  use precision
  implicit none

  public

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)

  interface
    subroutine gpu_copy_tmp2_c_c (dataType, tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_copy_tmp2_c_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_copy_tmp2_c_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_copy_tmp2_c_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: tmp2_dev, c_dev
      integer(kind=c_int), value         :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_copy_a_aux_bc_c (dataType, a_dev, aux_bc_dev, &
                                    n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_copy_a_aux_bc_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_copy_a_aux_bc_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_copy_a_aux_bc_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: a_dev, aux_bc_dev
      integer(kind=c_int), value         :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


  interface
    subroutine gpu_copy_aux_bc_aux_mat_c (dataType, aux_bc_dev, aux_mat_dev, &
                                          lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_copy_aux_bc_aux_mat_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_copy_aux_bc_aux_mat_FromC")
#elif defined(WITH_SYCL_GPU_VERSION)
                                                  bind(C, name="sycl_copy_aux_bc_aux_mat_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: aux_bc_dev, aux_mat_dev
      integer(kind=c_int), value         :: lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface
  
#endif /* defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION) */


  contains


    subroutine gpu_copy_tmp2_c (dataType, tmp2_dev, c_dev, nr_done, nstor, lcs, lce, &
                                ldc, ldcCols, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: tmp2_dev, c_dev
      integer(kind=c_int), value         :: nr_done, nstor, lcs, lce, ldc, ldcCols
      integer(kind=c_intptr_t), value    :: my_stream
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_copy_tmp2_c_c(dataType, tmp2_dev, c_dev, nr_done, nstor, lcs, lce, ldc, ldcCols, my_stream)
#endif
    end subroutine


    subroutine gpu_copy_a_aux_bc (dataType, a_dev, aux_bc_dev, &
                                  n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: a_dev, aux_bc_dev
      integer(kind=c_int), value         :: n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols
      integer(kind=c_intptr_t), value    :: my_stream
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_copy_a_aux_bc_c (dataType, a_dev, aux_bc_dev, &
                                n_aux_bc, nvals, lrs, lre, noff, nblk, n, l_rows, lda, ldaCols, my_stream)
#endif
    end subroutine


    subroutine gpu_copy_aux_bc_aux_mat (dataType, aux_bc_dev, aux_mat_dev, &
                                        lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: aux_bc_dev, aux_mat_dev
      integer(kind=c_int), value         :: lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult
      integer(kind=c_intptr_t), value    :: my_stream
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      call gpu_copy_aux_bc_aux_mat_c (dataType, aux_bc_dev, aux_mat_dev, &
                                      lrs, lre, nstor, n_aux_bc, nvals, l_rows, nblk, nblk_mult, my_stream)
#endif
    end subroutine


end module
