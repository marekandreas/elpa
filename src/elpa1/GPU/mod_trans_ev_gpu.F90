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
module trans_ev_gpu
  use, intrinsic :: iso_c_binding
  use precision
#ifdef WITH_NVIDIA_GPU_VERSION
  use trans_ev_cuda
#endif
#ifdef WITH_AMD_GPU_VERSION
  use trans_ev_hip
#endif
#ifdef WITH_SYCL_GPU_VERSION
  use trans_ev_sycl
#endif
  implicit none

  public
  contains

    subroutine gpu_scale_qmat_double_complex(ldq, l_cols, q_dev, tau_dev, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(c_int), intent(in) :: ldq, l_cols
      integer(kind=C_intptr_T)   :: q_dev, tau_dev
      integer(kind=c_intptr_t)   :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_scale_qmat_double_complex(ldq, l_cols, q_dev, tau_dev, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
      call hip_scale_qmat_double_complex(ldq, l_cols, q_dev, tau_dev, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
      call sycl_scale_qmat_double_complex(ldq, l_cols, q_dev, tau_dev, my_stream)
#endif

    end subroutine


    subroutine gpu_scale_qmat_float_complex(ldq, l_cols, q_dev, tau_dev, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(c_int), intent(in) :: ldq, l_cols
      integer(kind=C_intptr_T)   :: q_dev, tau_dev
      integer(kind=c_intptr_t)   :: my_stream

#ifdef WITH_NVIDIA_GPU_VERSION
      call cuda_scale_qmat_float_complex(ldq, l_cols, q_dev, tau_dev, my_stream)
#endif
#ifdef WITH_AMD_GPU_VERSION
      call hip_scale_qmat_float_complex(ldq, l_cols, q_dev, tau_dev, my_stream)
#endif
#ifdef WITH_SYCL_GPU_VERSION
      call sycl_scale_qmat_float_complex(ldq, l_cols, q_dev, tau_dev, my_stream)
#endif

    end subroutine

end module

