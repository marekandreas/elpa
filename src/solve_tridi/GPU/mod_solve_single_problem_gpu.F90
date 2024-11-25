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


module solve_single_problem_gpu
  use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
  use solve_single_problem_cuda
#endif
#ifdef WITH_AMD_GPU_VERSION
  use solve_single_problem_hip
#endif
  use precision

  implicit none

  public

  contains
    subroutine gpu_check_monotony_double(d_dev, q_dev, qtmp_dev, nlen, ldq, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: nlen, ldq
      integer(kind=c_intptr_t)           :: d_dev, q_dev, qtmp_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_check_monotony_double(d_dev, q_dev, qtmp_dev, nlen, ldq, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_check_monotony_double(d_dev, q_dev, qtmp_dev, nlen, ldq, my_stream)
#endif 

    end subroutine

    subroutine gpu_check_monotony_float(d_dev, q_dev, qtmp_dev, nlen, ldq, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: nlen, ldq
      integer(kind=c_intptr_t)           :: d_dev, q_dev, qtmp_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WANT_SINGLE_PRECISION_REAL
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_check_monotony_float(d_dev, q_dev, qtmp_dev, nlen, ldq, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_check_monotony_float(d_dev, q_dev, qtmp_dev, nlen, ldq, my_stream)
#endif 
#endif
    end subroutine

    subroutine gpu_construct_tridi_matrix_double(q_dev, d_dev, e_dev, nlen, ldq, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: nlen, ldq
      integer(kind=c_intptr_t)           :: q_dev, d_dev,e_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_construct_tridi_matrix_double(q_dev, d_dev, e_dev, nlen, ldq, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_construct_tridi_matrix_double(q_dev, d_dev, e_dev, nlen, ldq, my_stream)
#endif 

    end subroutine

    subroutine gpu_construct_tridi_matrix_float(q_dev, d_dev, e_dev, nlen, ldq, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: nlen, ldq
      integer(kind=c_intptr_t)           :: q_dev, d_dev,e_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WANT_SINGLE_PRECISION_REAL
#ifdef WITH_NVIDIA_GPU_VERSION
        call cuda_construct_tridi_matrix_float(q_dev, d_dev, e_dev, nlen, ldq, my_stream)
#endif 

#ifdef WITH_AMD_GPU_VERSION
        call hip_construct_tridi_matrix_float(q_dev, d_dev, e_dev, nlen, ldq, my_stream)
#endif 
#endif
    end subroutine
end module
