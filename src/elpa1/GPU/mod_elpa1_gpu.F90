!    Copyright 2024, A. Marek
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
module elpa1_gpu
  use, intrinsic :: iso_c_binding
  use precision
#ifdef WITH_NVIDIA_GPU_VERSION
  use elpa1_cuda
#endif
#ifdef WITH_AMD_GPU_VERSION
  use elpa1_hip
#endif
#ifdef WITH_SYCL_GPU_VERSION
  use elpa1_sycl
#endif
  implicit none

  public
  contains

    subroutine gpu_copy_real_part_to_q_double_complex(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(c_int), intent(in)           :: l_cols_nev, l_rows, matrixRows
      integer(kind=C_intptr_T)             :: q_dev, q_real_dev
      integer(kind=c_intptr_t), optional   :: my_stream

      if (present(my_stream)) then
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_real_part_to_q_double_complex(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_real_part_to_q_double_complex(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_scopy_real_part_to_q_complex(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev, my_stream)
#endif
      else
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_real_part_to_q_double_complex(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_real_part_to_q_double_complex(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_scopy_real_part_to_q_complex(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev)
#endif
     endif

    end subroutine


    subroutine gpu_copy_real_part_to_q_float_complex(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(c_int), intent(in)           :: l_cols_nev, l_rows, matrixRows
      integer(kind=C_intptr_T)             :: q_dev, q_real_dev
      integer(kind=c_intptr_t), optional   :: my_stream

      if (present(my_stream)) then
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_real_part_to_q_float_complex(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_real_part_to_q_float_complex(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_scopy_real_part_to_q_float_complex(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev, my_stream)
#endif
      else
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_real_part_to_q_float_complex(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_real_part_to_q_float_complex(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_scopy_real_part_to_q_float_complex(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev)
#endif
      endif

    end subroutine

    subroutine gpu_zero_skewsymmetric_q_double(q_dev, matrixRows, matrixCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(c_int), intent(in)           :: matrixRows, matrixCols
      integer(kind=C_intptr_T)             :: q_dev
      integer(kind=c_intptr_t), optional   :: my_stream


      if (present(my_stream)) then
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_zero_skewsymmetric_q_double(q_dev, matrixRows, matrixCols, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_zero_skewsymmetric_q_double(q_dev, matrixRows, matrixCols, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_zero_skewsymmetric_q_double(q_dev, matrixRows, matrixCols, my_stream)
#endif
      else
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_zero_skewsymmetric_q_double(q_dev, matrixRows, matrixCols)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_zero_skewsymmetric_q_double(q_dev, matrixRows, matrixCols)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_zero_skewsymmetric_q_double(q_dev, matrixRows, matrixCols)
#endif
      endif

    end subroutine

    subroutine gpu_zero_skewsymmetric_q_float(q_dev, matrixRows, matrixCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(c_int), intent(in)           :: matrixRows, matrixCols
      integer(kind=C_intptr_T)             :: q_dev
      integer(kind=c_intptr_t), optional   :: my_stream

      if (present(my_stream)) then
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_zero_skewsymmetric_q_float(q_dev, matrixRows, matrixCols, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_zero_skewsymmetric_q_float(q_dev, matrixRows, matrixCols, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_zero_skewsymmetric_q_float(q_dev, matrixRows, matrixCols, my_stream)
#endif
      else
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_zero_skewsymmetric_q_float(q_dev, matrixRows, matrixCols)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_zero_skewsymmetric_q_float(q_dev, matrixRows, matrixCols)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_zero_skewsymmetric_q_float(q_dev, matrixRows, matrixCols)
#endif
      endif
    end subroutine

    subroutine gpu_copy_skewsymmetric_second_half_q_double(q_dev, i, matrixRows, matrixCols, &
                                                               negative_or_positive, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(c_int), intent(in)           :: i, matrixRows, matrixCols, negative_or_positive
      integer(kind=C_intptr_T)             :: q_dev
      integer(kind=c_intptr_t), optional   :: my_stream

      if (present(my_stream)) then
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_skewsymmetric_second_half_q_double(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_skewsymmetric_second_half_q_double(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_copy_skewsymmetric_second_half_q_double(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive, my_stream)
#endif
      else
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_skewsymmetric_second_half_q_double(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_skewsymmetric_second_half_q_double(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_copy_skewsymmetric_second_half_q_double(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive)
#endif
      endif
    end subroutine

    subroutine gpu_copy_skewsymmetric_second_half_q_float(q_dev, i, matrixRows, matrixCols, &
                                                               negative_or_positive, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(c_int), intent(in)           :: i, matrixRows, matrixCols, negative_or_positive
      integer(kind=C_intptr_T)             :: q_dev
      integer(kind=c_intptr_t), optional   :: my_stream

      if (present(my_stream)) then
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_skewsymmetric_second_half_q_float(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_skewsymmetric_second_half_q_float(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_copy_skewsymmetric_second_half_q_float(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive, my_stream)
#endif
      else
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_skewsymmetric_second_half_q_float(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_skewsymmetric_second_half_q_float(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_copy_skewsymmetric_second_half_q_float(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive)
#endif
      endif
    end subroutine

    subroutine gpu_copy_skewsymmetric_first_half_q_double(q_dev, i, matrixRows, matrixCols, &
                                                               negative_or_positive, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(c_int), intent(in)           :: i, matrixRows, matrixCols, negative_or_positive
      integer(kind=C_intptr_T)             :: q_dev
      integer(kind=c_intptr_t), optional   :: my_stream

      if (present(my_stream)) then
#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_skewsymmetric_first_half_q_double(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_skewsymmetric_first_half_q_double(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_skewsymmetric_first_half_q_double(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive, my_stream)
#endif
      else
#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_skewsymmetric_first_half_q_double(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive)
#endif
#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_skewsymmetric_first_half_q_double(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive)
#endif
#ifdef WITH_SYCL_GPU_VERSION
      call sycl_copy_skewsymmetric_first_half_q_double(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive)
#endif
      endif
    end subroutine

    subroutine gpu_copy_skewsymmetric_first_half_q_float(q_dev, i, matrixRows, matrixCols, &
                                                               negative_or_positive, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(c_int), intent(in)           :: i, matrixRows, matrixCols, negative_or_positive
      integer(kind=C_intptr_T)             :: q_dev
      integer(kind=c_intptr_t), optional   :: my_stream

      if (present(my_stream)) then
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_skewsymmetric_first_half_q_float(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_skewsymmetric_first_half_q_float(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_copy_skewsymmetric_first_half_q_float(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive, my_stream)
#endif
      else
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_copy_skewsymmetric_first_half_q_float(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_copy_skewsymmetric_first_half_q_float(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_copy_skewsymmetric_first_half_q_float(q_dev, i, matrixRows, matrixCols, &
                                                                negative_or_positive)
#endif
      endif

    end subroutine

    subroutine gpu_get_skewsymmetric_second_half_q_double(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                               my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(c_int), intent(in)           :: matrixRows, matrixCols
      integer(kind=C_intptr_T)             :: q_dev, q2nd_dev
      integer(kind=c_intptr_t), optional   :: my_stream

      if (present(my_stream)) then
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_get_skewsymmetric_second_half_q_double(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                                my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_get_skewsymmetric_second_half_q_double(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                                my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_get_skewsymmetric_second_half_q_double(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                                my_stream)
#endif
      else
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_get_skewsymmetric_second_half_q_double(q_dev, q2nd_dev, matrixRows, matrixCols)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_get_skewsymmetric_second_half_q_double(q_dev, q2nd_dev, matrixRows, matrixCols)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_get_skewsymmetric_second_half_q_double(q_dev, q2nd_dev, matrixRows, matrixCols)
#endif
      endif

    end subroutine

    subroutine gpu_get_skewsymmetric_second_half_q_float(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                               my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(c_int), intent(in)           :: matrixRows, matrixCols
      integer(kind=C_intptr_T)             :: q_dev, q2nd_dev
      integer(kind=c_intptr_t), optional   :: my_stream

      if (present(my_stream)) then
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_get_skewsymmetric_second_half_q_float(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                                my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_get_skewsymmetric_second_half_q_float(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                                my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_get_skewsymmetric_second_half_q_float(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                                my_stream)
#endif
      else
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_get_skewsymmetric_second_half_q_float(q_dev, q2nd_dev, matrixRows, matrixCols)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_get_skewsymmetric_second_half_q_float(q_dev, q2nd_dev, matrixRows, matrixCols)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_get_skewsymmetric_second_half_q_float(q_dev, q2nd_dev, matrixRows, matrixCols)
#endif
      endif

    end subroutine

    subroutine gpu_put_skewsymmetric_second_half_q_double(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                               my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(c_int), intent(in)           :: matrixRows, matrixCols
      integer(kind=C_intptr_T)             :: q_dev, q2nd_dev
      integer(kind=c_intptr_t), optional   :: my_stream

      if (present(my_stream)) then
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_put_skewsymmetric_second_half_q_double(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                                my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_put_skewsymmetric_second_half_q_double(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                                my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_put_skewsymmetric_second_half_q_double(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                                my_stream)
#endif
      else
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_put_skewsymmetric_second_half_q_double(q_dev, q2nd_dev, matrixRows, matrixCols)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_put_skewsymmetric_second_half_q_double(q_dev, q2nd_dev, matrixRows, matrixCols)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_put_skewsymmetric_second_half_q_double(q_dev, q2nd_dev, matrixRows, matrixCols)
#endif
      endif

    end subroutine

    subroutine gpu_put_skewsymmetric_second_half_q_float(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                               my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(c_int), intent(in)           :: matrixRows, matrixCols
      integer(kind=C_intptr_T)             :: q_dev, q2nd_dev
      integer(kind=c_intptr_t), optional   :: my_stream

      if (present(my_stream)) then
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_put_skewsymmetric_second_half_q_float(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                                my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_put_skewsymmetric_second_half_q_float(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                                my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_put_skewsymmetric_second_half_q_float(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                                my_stream)
#endif
      else
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_put_skewsymmetric_second_half_q_float(q_dev, q2nd_dev, matrixRows, matrixCols)
#endif
#ifdef WITH_AMD_GPU_VERSION
        call hip_put_skewsymmetric_second_half_q_float(q_dev, q2nd_dev, matrixRows, matrixCols)
#endif
#ifdef WITH_SYCL_GPU_VERSION
        call sycl_put_skewsymmetric_second_half_q_float(q_dev, q2nd_dev, matrixRows, matrixCols)
#endif
      endif

    end subroutine



end module

