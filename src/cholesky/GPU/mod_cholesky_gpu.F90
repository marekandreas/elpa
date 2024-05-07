!    Copyright 2021, A. Marek
!
!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
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
!
! This file was written by A. Marek, MPCDF


#include "config-f90.h"
module cholesky_gpu
  use, intrinsic :: iso_c_binding
  use precision

#ifdef WITH_NVIDIA_GPU_VERSION
  use cholesky_cuda
#endif
#ifdef WITH_AMD_GPU_VERSION
  use cholesky_hip
#endif
#ifdef WITH_SYCL_GPU_VERSION
  use cholesky_sycl
#endif

  implicit none

  public
  contains

    subroutine gpu_check_device_info(info_dev, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_intptr_T)        :: info_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_check_device_info(info_dev, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
      call hip_check_device_info(info_dev, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
      print *,"gpu_check_device_info not implemented for sycl"
      stop 1
#endif
    end subroutine


    subroutine gpu_accumulate_device_info(info_abs_dev, info_new_dev, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_intptr_T)        :: info_abs_dev, info_new_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_accumulate_device_info(info_abs_dev, info_new_dev, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
      call hip_accumulate_device_info(info_abs_dev, info_new_dev, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
      print *,"gpu_accumulate_device_info not implemented for sycl"
      stop 1
#endif
    end subroutine


    subroutine gpu_copy_double_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, l_cols, l_colx, l_row1, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nblk, matrixRows, l_cols, l_colx, l_row1
      integer(kind=C_intptr_T)        :: a_dev, tmatc_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, l_cols, l_colx, l_row1, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_double_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, l_cols, l_colx, l_row1, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, l_cols, l_colx, l_row1, my_stream)
#endif
    end subroutine


    subroutine gpu_copy_float_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, l_cols, l_colx, l_row1, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nblk, matrixRows, l_cols, l_colx, l_row1
      integer(kind=C_intptr_T)        :: a_dev, tmatc_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, l_cols, l_colx, l_row1, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_float_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, l_cols, l_colx, l_row1, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, l_cols, l_colx, l_row1, my_stream)
#endif
    end subroutine


    subroutine gpu_copy_double_complex_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, l_cols, l_colx, l_row1, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nblk, matrixRows, l_cols, l_colx, l_row1
      integer(kind=C_intptr_T)        :: a_dev, tmatc_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_double_complex_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, l_cols, l_colx, l_row1, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_double_complex_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, l_cols, l_colx, l_row1, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_double_complex_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, l_cols, l_colx, l_row1, my_stream)
#endif
    end subroutine


    subroutine gpu_copy_float_complex_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, l_cols, l_colx, l_row1, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in) :: nblk, matrixRows, l_cols, l_colx, l_row1
      integer(kind=C_intptr_T)        :: a_dev, tmatc_dev
      integer(kind=C_intptr_T)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_float_complex_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, l_cols, l_colx, l_row1, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_float_complex_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, l_cols, l_colx, l_row1, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_float_complex_a_tmatc(a_dev, tmatc_dev, nblk, matrixRows, l_cols, l_colx, l_row1, my_stream)
#endif
    end subroutine

    
    subroutine gpu_set_a_lower_to_zero(dataType, a_dev, na, matrixRows, my_pcol, np_cols, my_prow, np_rows, &
                                       nblk, debug, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1, c_char), value     :: dataType
      integer(kind=c_int), intent(in) :: na, matrixRows, my_pcol, np_cols, my_prow, np_rows, nblk, debug
      integer(kind=c_intptr_t)        :: a_dev
      integer(kind=c_intptr_t)        :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_set_a_lower_to_zero(dataType, a_dev, na, matrixRows, my_pcol, np_cols, my_prow, np_rows, nblk, debug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
      call hip_set_a_lower_to_zero (dataType, a_dev, na, matrixRows, my_pcol, np_cols, my_prow, np_rows, nblk, debug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
      call sycl_set_a_lower_to_zero(dataType, a_dev, na, matrixRows, my_pcol, np_cols, my_prow, np_rows, nblk, debug, my_stream)
#endif
    end subroutine

end module

