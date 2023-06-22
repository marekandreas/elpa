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

  subroutine gpu_dot_product_and_assign_double(v_row_dev, l_rows, isOurProcessRow, dot_prod, v_row_last, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none

    integer(kind=C_INT), intent(in)     :: l_rows, isOurProcessRow
    real(kind=c_double), intent(out)    :: dot_prod, v_row_last
    integer(kind=C_intptr_T)            :: v_row_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_dot_product_and_assign_double(v_row_dev, l_rows, isOurProcessRow, dot_prod, v_row_last, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_dot_product_and_assign_double(v_row_dev, l_rows, isOurProcessRow, dot_prod, v_row_last, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_dot_product_and_assign_double(v_row_dev, l_rows, isOurProcessRow, dot_prod, v_row_last, my_stream)
#endif

  end subroutine

  subroutine gpu_update_matrix_element_add_double(a_dev, index, value, d_vec_dev, istep, n_stored_vecs, isSkewsymmetric, my_stream)
    use, intrinsic :: iso_c_binding
    use precision
    implicit none
#include "../../general/precision_kinds.F90"

    integer(kind=C_INT), intent(in)     :: index, istep, n_stored_vecs
    logical, intent(in)                 :: isSkewsymmetric
    ! MATH_DATATYPE(kind=rck), intent(in) :: value
    real(kind=c_double), intent(in)     :: value
    integer(kind=C_intptr_T)            :: a_dev, d_vec_dev
    integer(kind=c_intptr_t)            :: my_stream
    integer(kind=C_INT)                 :: isSkewsymmetricInt=0

    if (isSkewsymmetric) isSkewsymmetricInt = 1

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_update_matrix_element_add_double(a_dev, index, value, d_vec_dev, istep, n_stored_vecs, isSkewsymmetricInt, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_update_matrix_element_add_double (a_dev, index, value, d_vec_dev, istep, n_stored_vecs, isSkewsymmetricInt, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_update_matrix_element_add_double(a_dev, index, value, d_vec_dev, istep, n_stored_vecs, isSkewsymmetricInt, my_stream)
#endif

  end subroutine

  subroutine gpu_update_array_element_double(array_dev, index, value, my_stream) !< Update one element of device array: array_dev[index] = value
    use, intrinsic :: iso_c_binding
    use precision
    implicit none
#include "../../general/precision_kinds.F90"

    integer(kind=C_INT), intent(in)     :: index
    real(kind=c_double), intent(in)     :: value
    integer(kind=C_intptr_T)            :: array_dev
    integer(kind=c_intptr_t)            :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
    call cuda_update_array_element_double(array_dev, index, value, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
    call hip_update_array_element_double(array_dev, index, value, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
    call sycl_update_array_element_double(array_dev, index, value, my_stream)
#endif

  end subroutine

end module

