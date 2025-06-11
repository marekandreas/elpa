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


module distribute_global_column_gpu
  use, intrinsic :: iso_c_binding
  use precision
  implicit none

  public

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION)

  interface
    subroutine gpu_distribute_global_column_c(dataType, g_col_dev, l_col_dev, &
                                              g_col_dim1, g_col_dim2, ldq, matrixCols, &
                                              noff_in, noff, nlen, my_prow, np_rows, nblk, debug, my_stream) &
#if   defined(WITH_NVIDIA_GPU_VERSION)
                                                  bind(C, name="cuda_distribute_global_column_FromC")
#elif defined(WITH_AMD_GPU_VERSION)
                                                  bind(C, name= "hip_distribute_global_column_FromC")
#endif
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: g_col_dev, l_col_dev
      integer(kind=c_int), value         :: g_col_dim1, g_col_dim2, ldq, matrixCols, noff_in, noff, nlen, &
                                            my_prow, np_rows, nblk, debug
      integer(kind=c_intptr_t), value    :: my_stream
    end subroutine
  end interface


#endif /* defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) */


  contains


    subroutine gpu_distribute_global_column(dataType, g_col_dev, l_col_dev, &
                                            g_col_dim1, g_col_dim2, ldq, matrixCols, &
                                            noff_in, noff, nlen, my_prow, np_rows, nblk, debug, my_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1, c_char), value        :: dataType
      integer(kind=c_intptr_t), value    :: g_col_dev, l_col_dev
      integer(kind=c_int), value         :: g_col_dim1, g_col_dim2, ldq, matrixCols, noff_in, noff, nlen, &
                                            my_prow, np_rows, nblk, debug
      integer(kind=c_intptr_t), value    :: my_stream
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION)
      call gpu_distribute_global_column_c(dataType, g_col_dev, l_col_dev, &
                                          g_col_dim1, g_col_dim2, ldq, matrixCols, &
                                          noff_in, noff, nlen, my_prow, np_rows, nblk, debug, my_stream)
#endif
    end subroutine

end module
