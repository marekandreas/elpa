!    Copyright 2023, P. Karpov
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
! This file was written by P. Karpov, MPCDF
#include "config-f90.h"

module tridiag_gpu
  use, intrinsic :: iso_c_binding
  use precision
#ifdef WITH_NVIDIA_GPU_VERSION
  use tridiag_cuda
#endif
#ifdef WITH_AMD_GPU_VERSION
  use tridiag_hip
#endif
#ifdef WITH_SYCL_GPU_VERSION
  use tridiag_sycl
#endif
  implicit none

  public
  contains


  subroutine gpu_copy_and_set_zeros_double(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                           aux1_dev, vav_dev, d_vec_dev, &
                                           isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL, &
                                           wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: l_rows, l_cols, matrixRows, istep
    logical, intent(in)                 :: isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, &
                                           isSkewsymmetric, useCCL, wantDebug
    integer(kind=c_intptr_t)            :: v_row_dev, a_dev, aux1_dev, vav_dev, d_vec_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_and_set_zeros_double(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                        aux1_dev, vav_dev, d_vec_dev, &
                                        isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL, &
                                        wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_copy_and_set_zeros_double (v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                        aux1_dev, vav_dev, d_vec_dev, &
                                        isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL, &
                                        wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_copy_and_set_zeros_double(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                        aux1_dev, vav_dev, d_vec_dev, &
                                        isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL, &
                                        wantDebug, my_stream)
#endif
  end subroutine


  subroutine gpu_copy_and_set_zeros_float(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                           aux1_dev, vav_dev, d_vec_dev, &
                                           isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL, &
                                           wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: l_rows, l_cols, matrixRows, istep
    logical, intent(in)                 :: isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, &
                                           isSkewsymmetric, useCCL, wantDebug
    integer(kind=c_intptr_t)            :: v_row_dev, a_dev, aux1_dev, vav_dev, d_vec_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_and_set_zeros_float(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                        aux1_dev, vav_dev, d_vec_dev, &
                                        isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL, &
                                           wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_copy_and_set_zeros_float (v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                        aux1_dev, vav_dev, d_vec_dev, &
                                        isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL, &
                                           wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_copy_and_set_zeros_float(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                        aux1_dev, vav_dev, d_vec_dev, &
                                        isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL, &
                                           wantDebug, my_stream)
#endif
  end subroutine


  subroutine gpu_copy_and_set_zeros_double_complex(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                           aux1_dev, vav_dev, d_vec_dev, &
                                           isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL, &
                                           wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: l_rows, l_cols, matrixRows, istep
    logical, intent(in)                 :: isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, &
                                           isSkewsymmetric, useCCL, wantDebug
    integer(kind=c_intptr_t)            :: v_row_dev, a_dev, aux1_dev, vav_dev, d_vec_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_and_set_zeros_double_complex(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                        aux1_dev, vav_dev, d_vec_dev, &
                                        isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL, &
                                           wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_copy_and_set_zeros_double_complex (v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                        aux1_dev, vav_dev, d_vec_dev, &
                                        isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL, &
                                           wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_copy_and_set_zeros_double_complex(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                        aux1_dev, vav_dev, d_vec_dev, &
                                        isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL, &
                                           wantDebug, my_stream)
#endif
  end subroutine


  subroutine gpu_copy_and_set_zeros_float_complex(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                           aux1_dev, vav_dev, d_vec_dev, &
                                           isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL, &
                                           wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: l_rows, l_cols, matrixRows, istep
    logical, intent(in)                 :: isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, &
                                           isSkewsymmetric, useCCL, wantDebug
    integer(kind=c_intptr_t)            :: v_row_dev, a_dev, aux1_dev, vav_dev, d_vec_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_copy_and_set_zeros_float_complex(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                        aux1_dev, vav_dev, d_vec_dev, &
                                        isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL, &
                                           wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_copy_and_set_zeros_float_complex (v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                        aux1_dev, vav_dev, d_vec_dev, &
                                        isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL, &
                                           wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_copy_and_set_zeros_float_complex(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                        aux1_dev, vav_dev, d_vec_dev, &
                                        isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, isSkewsymmetric, useCCL, &
                                           wantDebug, my_stream)
#endif
  end subroutine

  !________________________________________________________________

  subroutine gpu_dot_product_double(n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: n, incx, incy, sm_count
    logical, intent(in)                 :: wantDebug
    integer(kind=c_intptr_t)            :: x_dev, y_dev, result_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_dot_product_double(n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_dot_product_double (n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_dot_product_double(n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
#endif
  end subroutine


  subroutine gpu_dot_product_float(n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: n, incx, incy, sm_count
    logical, intent(in)                 :: wantDebug
    integer(kind=c_intptr_t)            :: x_dev, y_dev, result_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_dot_product_float(n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_dot_product_float (n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_dot_product_float(n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
#endif
  end subroutine


  subroutine gpu_dot_product_double_complex(n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: n, incx, incy, sm_count
    logical, intent(in)                 :: wantDebug
    integer(kind=c_intptr_t)            :: x_dev, y_dev, result_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_dot_product_double_complex(n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_dot_product_double_complex (n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_dot_product_double_complex(n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
#endif
  end subroutine


  subroutine gpu_dot_product_float_complex(n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: n, incx, incy, sm_count
    logical, intent(in)                 :: wantDebug
    integer(kind=c_intptr_t)            :: x_dev, y_dev, result_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_dot_product_float_complex(n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_dot_product_float_complex (n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_dot_product_float_complex(n, x_dev, incx, y_dev, incy, result_dev, wantDebug, sm_count, my_stream)
#endif
  end subroutine

  !________________________________________________________________
  
  subroutine gpu_dot_product_and_assign_double(v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: l_rows
    logical, intent(in)                 :: isOurProcessRow, wantDebug
    integer(kind=c_intptr_t)            :: v_row_dev, aux1_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_dot_product_and_assign_double(v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_dot_product_and_assign_double (v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_dot_product_and_assign_double(v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, my_stream)
#endif
  end subroutine


  subroutine gpu_dot_product_and_assign_float(v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: l_rows
    logical, intent(in)                 :: isOurProcessRow, wantDebug
    integer(kind=c_intptr_t)            :: v_row_dev, aux1_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_dot_product_and_assign_float(v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_dot_product_and_assign_float (v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_dot_product_and_assign_float(v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, my_stream)
#endif
  end subroutine


  subroutine gpu_dot_product_and_assign_double_complex(v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: l_rows
    logical, intent(in)                 :: isOurProcessRow, wantDebug
    integer(kind=c_intptr_t)            :: v_row_dev, aux1_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_dot_product_and_assign_double_complex(v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_dot_product_and_assign_double_complex (v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_dot_product_and_assign_double_complex(v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, my_stream)
#endif
  end subroutine


  subroutine gpu_dot_product_and_assign_float_complex(v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: l_rows
    logical, intent(in)                 :: isOurProcessRow, wantDebug
    integer(kind=c_intptr_t)            :: v_row_dev, aux1_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_dot_product_and_assign_float_complex(v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_dot_product_and_assign_float_complex (v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_dot_product_and_assign_float_complex(v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, my_stream)
#endif
  end subroutine

  !________________________________________________________________

  subroutine gpu_set_e_vec_scale_set_one_store_v_row_double(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, & 
                                                            l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                                            wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    logical, intent(in)                 :: isOurProcessRow, useCCL, wantDebug
    integer(kind=c_int), intent(in)     :: l_rows, l_cols, matrixRows, istep
    integer(kind=c_intptr_t)            :: e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_set_e_vec_scale_set_one_store_v_row_double(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, &
                                                         l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                                         wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_set_e_vec_scale_set_one_store_v_row_double (e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, &
                                                         l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                                         wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_set_e_vec_scale_set_one_store_v_row_double(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, &
                                                         l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                                         wantDebug, my_stream)
#endif
  end subroutine


  subroutine gpu_set_e_vec_scale_set_one_store_v_row_float(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, & 
                                                            l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                                            wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    logical, intent(in)                 :: isOurProcessRow, useCCL, wantDebug
    integer(kind=c_int), intent(in)     :: l_rows, l_cols, matrixRows, istep
    integer(kind=c_intptr_t)            :: e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_set_e_vec_scale_set_one_store_v_row_float(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, &
                                                         l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                                         wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_set_e_vec_scale_set_one_store_v_row_float (e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, &
                                                         l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                                         wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_set_e_vec_scale_set_one_store_v_row_float(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, &
                                                         l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                                         wantDebug, my_stream)
#endif
  end subroutine


  subroutine gpu_set_e_vec_scale_set_one_store_v_row_double_complex(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, & 
                                                            l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                                            wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    logical, intent(in)                 :: isOurProcessRow, useCCL, wantDebug
    integer(kind=c_int), intent(in)     :: l_rows, l_cols, matrixRows, istep
    integer(kind=c_intptr_t)            :: e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_set_e_vec_scale_set_one_store_v_row_double_complex(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, &
                                                         l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                                         wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_set_e_vec_scale_set_one_store_v_row_double_complex (e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, &
                                                         l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                                         wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_set_e_vec_scale_set_one_store_v_row_double_complex(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, &
                                                         l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                                         wantDebug, my_stream)
#endif
  end subroutine


  subroutine gpu_set_e_vec_scale_set_one_store_v_row_float_complex(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, & 
                                                            l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                                            wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    logical, intent(in)                 :: isOurProcessRow, useCCL, wantDebug
    integer(kind=c_int), intent(in)     :: l_rows, l_cols, matrixRows, istep
    integer(kind=c_intptr_t)            :: e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_set_e_vec_scale_set_one_store_v_row_float_complex(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, &
                                                         l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                                         wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_set_e_vec_scale_set_one_store_v_row_float_complex (e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, &
                                                         l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                                         wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_set_e_vec_scale_set_one_store_v_row_float_complex(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, &
                                                         l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, &
                                                         wantDebug, my_stream)
#endif
  end subroutine

  !________________________________________________________________

  subroutine gpu_store_u_v_in_uv_vu_double(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, & 
                                           v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                           vav_host_or_dev, tau_istep_host_or_dev, &
                                           l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                           useCCL, wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep
    logical, intent(in)                 :: useCCL, wantDebug
    integer(kind=c_intptr_t)            :: vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                           v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                           vav_host_or_dev, tau_istep_host_or_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_store_u_v_in_uv_vu_double(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, & 
                                        v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                        vav_host_or_dev, tau_istep_host_or_dev, &
                                        l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                        useCCL, wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_store_u_v_in_uv_vu_double (vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, & 
                                        v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                        vav_host_or_dev, tau_istep_host_or_dev, &
                                        l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                        useCCL, wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_store_u_v_in_uv_vu_double(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, & 
                                        v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                        vav_host_or_dev, tau_istep_host_or_dev, &
                                        l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                        useCCL, wantDebug, my_stream)
#endif
  end subroutine


  subroutine gpu_store_u_v_in_uv_vu_float(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, & 
                                           v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                           vav_host_or_dev, tau_istep_host_or_dev, &
                                           l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                           useCCL, wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep
    logical, intent(in)                 :: useCCL, wantDebug
    integer(kind=c_intptr_t)            :: vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                           v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                           vav_host_or_dev, tau_istep_host_or_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_store_u_v_in_uv_vu_float(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, & 
                                        v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                        vav_host_or_dev, tau_istep_host_or_dev, &
                                        l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                        useCCL, wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_store_u_v_in_uv_vu_float (vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, & 
                                        v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                        vav_host_or_dev, tau_istep_host_or_dev, &
                                        l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                        useCCL, wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_store_u_v_in_uv_vu_float(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, & 
                                        v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                        vav_host_or_dev, tau_istep_host_or_dev, &
                                        l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                        useCCL, wantDebug, my_stream)
#endif
  end subroutine


    subroutine gpu_store_u_v_in_uv_vu_double_complex(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, & 
                                           v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                           vav_host_or_dev, tau_istep_host_or_dev, &
                                           l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                           useCCL, wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep
    logical, intent(in)                 :: useCCL, wantDebug
    integer(kind=c_intptr_t)            :: vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                           v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                           vav_host_or_dev, tau_istep_host_or_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_store_u_v_in_uv_vu_double_complex(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, & 
                                        v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                        vav_host_or_dev, tau_istep_host_or_dev, &
                                        l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                        useCCL, wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_store_u_v_in_uv_vu_double_complex (vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, & 
                                        v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                        vav_host_or_dev, tau_istep_host_or_dev, &
                                        l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                        useCCL, wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_store_u_v_in_uv_vu_double_complex(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, & 
                                        v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                        vav_host_or_dev, tau_istep_host_or_dev, &
                                        l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                        useCCL, wantDebug, my_stream)
#endif
  end subroutine


  subroutine gpu_store_u_v_in_uv_vu_float_complex(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, & 
                                           v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                           vav_host_or_dev, tau_istep_host_or_dev, &
                                           l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                           useCCL, wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep
    logical, intent(in)                 :: useCCL, wantDebug
    integer(kind=c_intptr_t)            :: vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                           v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                           vav_host_or_dev, tau_istep_host_or_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_store_u_v_in_uv_vu_float_complex(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, & 
                                        v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                        vav_host_or_dev, tau_istep_host_or_dev, &
                                        l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                        useCCL, wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_store_u_v_in_uv_vu_float_complex (vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, & 
                                        v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                        vav_host_or_dev, tau_istep_host_or_dev, &
                                        l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                        useCCL, wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_store_u_v_in_uv_vu_float_complex(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, & 
                                        v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                        vav_host_or_dev, tau_istep_host_or_dev, &
                                        l_rows, l_cols, n_stored_vecs, max_local_rows, max_local_cols, istep, &
                                        useCCL, wantDebug, my_stream)
#endif
  end subroutine

  !________________________________________________________________
  
  subroutine gpu_update_matrix_element_add_double(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                            l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                            isSkewsymmetric, wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs
    logical, intent(in)                 :: isSkewsymmetric, wantDebug
    integer(kind=c_intptr_t)            :: vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_update_matrix_element_add_double(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                               l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                               isSkewsymmetric, wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_update_matrix_element_add_double (vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                              l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                              isSkewsymmetric, wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_update_matrix_element_add_double(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                               l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                               isSkewsymmetric, wantDebug, my_stream)
#endif
  end subroutine


  subroutine gpu_update_matrix_element_add_float(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                            l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                            isSkewsymmetric, wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs
    logical, intent(in)                 :: isSkewsymmetric, wantDebug
    integer(kind=c_intptr_t)            :: vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_update_matrix_element_add_float(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                               l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                               isSkewsymmetric, wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_update_matrix_element_add_float (vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                              l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                              isSkewsymmetric, wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_update_matrix_element_add_float(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                               l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                               isSkewsymmetric, wantDebug, my_stream)
#endif
  end subroutine 


  subroutine gpu_update_matrix_element_add_double_complex(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                            l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                            isSkewsymmetric, wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs
    logical, intent(in)                 :: isSkewsymmetric, wantDebug
    integer(kind=c_intptr_t)            :: vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_update_matrix_element_add_double_complex(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                               l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                               isSkewsymmetric, wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_update_matrix_element_add_double_complex (vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                              l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                              isSkewsymmetric, wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_update_matrix_element_add_double_complex(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                               l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                               isSkewsymmetric, wantDebug, my_stream)
#endif
  end subroutine


  subroutine gpu_update_matrix_element_add_float_complex(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                            l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                            isSkewsymmetric, wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs
    logical, intent(in)                 :: isSkewsymmetric, wantDebug
    integer(kind=c_intptr_t)            :: vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_update_matrix_element_add_float_complex(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                               l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                               isSkewsymmetric, wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_update_matrix_element_add_float_complex (vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                              l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                              isSkewsymmetric, wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_update_matrix_element_add_float_complex(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                               l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                               isSkewsymmetric, wantDebug, my_stream)
#endif
  end subroutine 

  !________________________________________________________________

  ! Update one element of device array: array_dev[index] = value
  subroutine gpu_update_array_element_double(array_dev, index, value, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: index
    integer(kind=c_intptr_t)            :: value
    integer(kind=c_intptr_t)            :: array_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_update_array_element_double(array_dev, index, value, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_update_array_element_double (array_dev, index, value, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_update_array_element_double(array_dev, index, value, my_stream)
#endif
  end subroutine


  subroutine gpu_update_array_element_float(array_dev, index, value, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: index
    integer(kind=c_intptr_t)            :: value
    integer(kind=c_intptr_t)            :: array_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_update_array_element_float(array_dev, index, value, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_update_array_element_float (array_dev, index, value, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_update_array_element_float(array_dev, index, value, my_stream)
#endif
  end subroutine


  subroutine gpu_update_array_element_double_complex(array_dev, index, value, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: index
    integer(kind=c_intptr_t)            :: value
    integer(kind=c_intptr_t)            :: array_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_update_array_element_double_complex(array_dev, index, value, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_update_array_element_double_complex (array_dev, index, value, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_update_array_element_double_complex(array_dev, index, value, my_stream)
#endif
  end subroutine


  subroutine gpu_update_array_element_float_complex(array_dev, index, value, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: index
    integer(kind=c_intptr_t)            :: value
    integer(kind=c_intptr_t)            :: array_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_update_array_element_float_complex(array_dev, index, value, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_update_array_element_float_complex (array_dev, index, value, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_update_array_element_float_complex(array_dev, index, value, my_stream)
#endif
  end subroutine

  !________________________________________________________________

  subroutine gpu_hh_transform_double(obj, alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use elpa_abstract_impl
    implicit none

    class(elpa_abstract_impl_t), intent(inout)  :: obj
    logical, intent(in)                 :: wantDebug
    integer(kind=c_intptr_t)            :: alpha_dev, xnorm_sq_dev, xf_dev, tau_dev
    integer(kind=c_intptr_t)            :: my_stream

    if (wantDebug) call obj%timer%start("gpu_hh_transform")

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_hh_transform_double(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_hh_transform_double (alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_hh_transform_double(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
#endif
    
    if (wantDebug) call obj%timer%stop("gpu_hh_transform")
  end subroutine


  subroutine gpu_hh_transform_float(obj, alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use elpa_abstract_impl
    implicit none

    class(elpa_abstract_impl_t), intent(inout)  :: obj
    logical, intent(in)                 :: wantDebug
    integer(kind=c_intptr_t)            :: alpha_dev, xnorm_sq_dev, xf_dev, tau_dev
    integer(kind=c_intptr_t)            :: my_stream

    if (wantDebug) call obj%timer%start("gpu_hh_transform")

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_hh_transform_float(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_hh_transform_float (alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_hh_transform_float(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
#endif
    
    if (wantDebug) call obj%timer%stop("gpu_hh_transform")
  end subroutine


  subroutine gpu_hh_transform_double_complex(obj, alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use elpa_abstract_impl
    implicit none

    class(elpa_abstract_impl_t), intent(inout)  :: obj
    logical, intent(in)                 :: wantDebug
    integer(kind=c_intptr_t)            :: alpha_dev, xnorm_sq_dev, xf_dev, tau_dev
    integer(kind=c_intptr_t)            :: my_stream

    if (wantDebug) call obj%timer%start("gpu_hh_transform")

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_hh_transform_double_complex(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_hh_transform_double_complex (alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_hh_transform_double_complex(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
#endif
    
    if (wantDebug) call obj%timer%stop("gpu_hh_transform")
  end subroutine

  subroutine gpu_hh_transform_float_complex(obj, alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
    use, intrinsic :: iso_c_binding
    use elpa_abstract_impl
    implicit none

    class(elpa_abstract_impl_t), intent(inout)  :: obj
    logical, intent(in)                 :: wantDebug
    integer(kind=c_intptr_t)            :: alpha_dev, xnorm_sq_dev, xf_dev, tau_dev
    integer(kind=c_intptr_t)            :: my_stream

    if (wantDebug) call obj%timer%start("gpu_hh_transform")

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_hh_transform_float_complex(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_hh_transform_float_complex (alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_hh_transform_float_complex(alpha_dev, xnorm_sq_dev, xf_dev, tau_dev, wantDebug, my_stream)
#endif
    
    if (wantDebug) call obj%timer%stop("gpu_hh_transform")
  end subroutine

  !________________________________________________________________

  subroutine gpu_transpose_reduceadd_vectors_copy_block_double(aux_transpose_dev, vmat_st_dev, & 
                                              nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                              lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                              isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride, &  
                                           np_st, ld_st, direction, sm_count
    integer(kind=c_intptr_t)            :: aux_transpose_dev, vmat_st_dev
    logical, intent(in)                 :: isSkewsymmetric, isReduceadd, wantDebug
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_transpose_reduceadd_vectors_copy_block_double(aux_transpose_dev, vmat_st_dev, &
                                                  nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                                  lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                                  isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_transpose_reduceadd_vectors_copy_block_double (aux_transpose_dev, vmat_st_dev, &
                                                  nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                                  lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                                  isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_transpose_reduceadd_vectors_copy_block_double(aux_transpose_dev, vmat_st_dev, &
                                                  nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                                  lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                                  isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
#endif
  end subroutine


  subroutine gpu_transpose_reduceadd_vectors_copy_block_float(aux_transpose_dev, vmat_st_dev, & 
                                              nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                              lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                              isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride, &  
                                           np_st, ld_st, direction, sm_count
    integer(kind=c_intptr_t)            :: aux_transpose_dev, vmat_st_dev
    logical, intent(in)                 :: isSkewsymmetric, isReduceadd, wantDebug
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_transpose_reduceadd_vectors_copy_block_float(aux_transpose_dev, vmat_st_dev, &
                                                  nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                                  lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                                  isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_transpose_reduceadd_vectors_copy_block_float (aux_transpose_dev, vmat_st_dev, &
                                                  nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                                  lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                                  isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_transpose_reduceadd_vectors_copy_block_float(aux_transpose_dev, vmat_st_dev, &
                                                  nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                                  lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                                  isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
#endif
  end subroutine


  subroutine gpu_transpose_reduceadd_vectors_copy_block_double_complex(aux_transpose_dev, vmat_st_dev, & 
                                              nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                              lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                              isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride, &  
                                           np_st, ld_st, direction, sm_count
    integer(kind=c_intptr_t)            :: aux_transpose_dev, vmat_st_dev
    logical, intent(in)                 :: isSkewsymmetric, isReduceadd, wantDebug
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_transpose_reduceadd_vectors_copy_block_double_complex(aux_transpose_dev, vmat_st_dev, &
                                                  nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                                  lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                                  isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_transpose_reduceadd_vectors_copy_block_double_complex (aux_transpose_dev, vmat_st_dev, &
                                                  nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                                  lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                                  isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_transpose_reduceadd_vectors_copy_block_double_complex(aux_transpose_dev, vmat_st_dev, &
                                                  nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                                  lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                                  isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
#endif
  end subroutine


  subroutine gpu_transpose_reduceadd_vectors_copy_block_float_complex(aux_transpose_dev, vmat_st_dev, & 
                                              nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                              lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                              isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=c_int), intent(in)     :: nvc, nvr, n_block, nblks_skip, nblks_tot, lcm_s_t, nblk, auxstride, &  
                                           np_st, ld_st, direction, sm_count
    integer(kind=c_intptr_t)            :: aux_transpose_dev, vmat_st_dev
    logical, intent(in)                 :: isSkewsymmetric, isReduceadd, wantDebug
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_transpose_reduceadd_vectors_copy_block_float_complex(aux_transpose_dev, vmat_st_dev, &
                                                  nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                                  lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                                  isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_transpose_reduceadd_vectors_copy_block_float_complex (aux_transpose_dev, vmat_st_dev, &
                                                  nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                                  lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                                  isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_transpose_reduceadd_vectors_copy_block_float_complex(aux_transpose_dev, vmat_st_dev, &
                                                  nvc, nvr, n_block, nblks_skip, nblks_tot, &
                                                  lcm_s_t, nblk, auxstride, np_st, ld_st, direction, &
                                                  isSkewsymmetric, isReduceadd, wantDebug, sm_count, my_stream)
#endif
  end subroutine

end module

