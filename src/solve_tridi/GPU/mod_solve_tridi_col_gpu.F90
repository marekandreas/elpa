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


module tridi_col_gpu
  use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
  use solve_tridi_col_cuda
#endif
#ifdef WITH_AMD_GPU_VERSION
  use solve_tridi_col_hip
#endif
  use precision

  implicit none

  public

  contains


    subroutine gpu_update_d_double(limits_dev, d_dev, e_dev, ndiv, na, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: ndiv, na
      integer(kind=c_intptr_t)           :: d_dev, e_dev
      type(c_ptr)                        :: limits_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_update_d_double(limits_dev, d_dev, e_dev, ndiv, na, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_update_d_double(limits_dev, d_dev, e_dev, ndiv, na, my_stream)
#endif 

    end subroutine


    subroutine gpu_update_d_float(limits_dev, d_dev, e_dev, ndiv, na, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: ndiv, na
      integer(kind=c_intptr_t)           :: d_dev, e_dev
      type(c_ptr)                        :: limits_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WANT_SINGLE_PRECISION_REAL
#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_update_d_float(limits_dev, d_dev, e_dev, ndiv, na, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_update_d_float(limits_dev, d_dev, e_dev, ndiv, na, my_stream)
#endif 
#endif
    end subroutine


    subroutine gpu_copy_qmat1_to_qmat2_double(qmat1_dev, qmat2_dev, max_size, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: max_size
      integer(kind=c_intptr_t)           :: qmat1_dev, qmat2_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_qmat1_to_qmat2_double(qmat1_dev, qmat2_dev, max_size, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_qmat1_to_qmat2_double(qmat1_dev, qmat2_dev, max_size, my_stream)
#endif 

    end subroutine


    subroutine gpu_copy_qmat1_to_qmat2_float(qmat1_dev, qmat2_dev, max_size, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: max_size
      integer(kind=c_intptr_t)           :: qmat1_dev, qmat2_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WANT_SINGLE_PRECISION_REAL
#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_qmat1_to_qmat2_float(qmat1_dev, qmat2_dev, max_size, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_qmat1_to_qmat2_float(qmat1_dev, qmat2_dev, max_size, my_stream)
#endif 
#endif

    end subroutine


    subroutine gpu_copy_d_to_d_tmp_double(d_dev, d_tmp_dev, na,  my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: na
      integer(kind=c_intptr_t)           :: d_dev, d_tmp_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_d_to_d_tmp_double(d_dev, d_tmp_dev, na, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_d_to_d_tmp_double(d_dev, d_tmp_dev, na, my_stream)
#endif 

    end subroutine


    subroutine gpu_copy_d_to_d_tmp_float(d_dev, d_tmp_dev, na,  my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: na
      integer(kind=c_intptr_t)           :: d_dev, d_tmp_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WANT_SINGLE_PRECISION_REAL
#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_copy_d_to_d_tmp_float(d_dev, d_tmp_dev, na, my_stream)
#endif

#ifdef WITH_AMD_GPU_VERSION
      call hip_copy_d_to_d_tmp_float(d_dev, d_tmp_dev, na, my_stream)
#endif 
#endif
    end subroutine

end module
