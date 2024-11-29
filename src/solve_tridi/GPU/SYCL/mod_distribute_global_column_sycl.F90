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

!    This file was written by A. Marek, MPCDF
#endif


#include "config-f90.h"

! This file is auto-generated. Do NOT edit

module distribute_global_column_sycl
  use, intrinsic :: iso_c_binding
  use precision

  implicit none

  public

  interface
    subroutine sycl_distribute_global_column_double_c(g_col_dev, g_col_dim1, g_col_dim2, l_col_dev, ldq, matrixCols, &
                                                            noff_in, noff, &
                                                             nlen, my_prow, np_rows, nblk, my_stream) &
                                                     bind(C, name="sycl_distribute_global_column_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: g_col_dev, l_col_dev
      integer(kind=c_int), intent(in)  :: noff_in, noff, nlen, my_prow, np_rows, nblk, matrixCols, ldq, g_col_dim1, g_col_dim2
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface



#ifdef WANT_SINGLE_PRECISION_REAL
  interface
    subroutine sycl_distribute_global_column_float_c(g_col_dev, g_col_dim1, g_col_dim2, l_col_dev, ldq, matrixCols, &
                                                            noff_in, noff, &
                                                             nlen, my_prow, np_rows, nblk, my_stream) &
                                                     bind(C, name="sycl_distribute_global_column_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: g_col_dev, l_col_dev
      integer(kind=c_int), intent(in)  :: noff_in, noff, nlen, my_prow, np_rows, nblk, matrixCols, ldq, g_col_dim1, g_col_dim2
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

#endif


  contains
    subroutine sycl_distribute_global_column_double(g_col_dev, g_col_dim1, g_col_dim2, l_col_dev, ldq, matrixCols, &
                                         noff_in, noff, nlen, my_prow, np_rows, nblk, &
                                         my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in)    :: noff_in, noff, nlen, my_prow, np_rows, nblk, matrixCols, ldq, g_col_dim1, g_col_dim2
      integer(kind=C_intptr_T)           :: g_col_dev, l_col_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_SYCL_GPU_VERSION
      if (present(my_stream)) then
        call sycl_distribute_global_column_double_c(g_col_dev, g_col_dim1, g_col_dim2, l_col_dev, ldq, matrixCols, &
                                                  noff_in, noff, nlen, my_prow, np_rows, nblk, my_stream)
      else
        call sycl_distribute_global_column_double_c(g_col_dev, g_col_dim1, g_col_dim2, l_col_dev, ldq, matrixCols, &
                                                  noff_in, noff, nlen, my_prow, np_rows, nblk, my_stream2)
      endif
#endif

    end subroutine


#ifdef WANT_SINGLE_PRECISION_REAL
    subroutine sycl_distribute_global_column_float(g_col_dev, g_col_dim1, g_col_dim2, l_col_dev, ldq, matrixCols, &
                                         noff_in, noff, nlen, my_prow, np_rows, nblk, &
                                         my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in)    :: noff_in, noff, nlen, my_prow, np_rows, nblk, matrixCols, ldq, g_col_dim1, g_col_dim2
      integer(kind=C_intptr_T)           :: g_col_dev, l_col_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_SYCL_GPU_VERSION
      if (present(my_stream)) then
        call sycl_distribute_global_column_float_c(g_col_dev, g_col_dim1, g_col_dim2, l_col_dev, ldq, matrixCols, &
                                                  noff_in, noff, nlen, my_prow, np_rows, nblk, my_stream)
      else
        call sycl_distribute_global_column_float_c(g_col_dev, g_col_dim1, g_col_dim2, l_col_dev, ldq, matrixCols, &
                                                  noff_in, noff, nlen, my_prow, np_rows, nblk, my_stream2)
      endif
#endif

    end subroutine
#endif


end module
