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

! This file is auto-generated. Do NOT edit

module elpa1_hip
  use, intrinsic :: iso_c_binding
  use precision

  implicit none

  public

  interface
    subroutine hip_copy_real_part_to_q_double_complex_c(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev, my_stream) &
                                                  bind(C, name="hip_copy_real_part_to_q_double_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int), intent(in)  :: l_cols_nev, l_rows, matrixRows
      integer(kind=c_intptr_t), value  :: q_dev, q_real_dev
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_copy_real_part_to_q_float_complex_c(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev, my_stream) &
                                                  bind(C, name="hip_copy_real_part_to_q_float_complex_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int), intent(in)  :: l_cols_nev, l_rows, matrixRows
      integer(kind=c_intptr_t), value  :: q_dev, q_real_dev
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_zero_skewsymmetric_q_double_c(q_dev, matrixRows, matrixCols, my_stream) &
                                                  bind(C, name="hip_zero_skewsymmetric_q_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int), intent(in)  :: matrixRows, matrixCols
      integer(kind=c_intptr_t), value  :: q_dev
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_copy_skewsymmetric_second_half_q_double_c(q_dev, i, matrixRows, matrixCols, &
                                                         negative_or_positive, my_stream) &
                                                  bind(C, name="hip_copy_skewsymmetric_second_half_q_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int), intent(in)  :: i, matrixRows, matrixCols, negative_or_positive
      integer(kind=c_intptr_t), value  :: q_dev
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_copy_skewsymmetric_first_half_q_double_c(q_dev, i, matrixRows, matrixCols, &
                                                         negative_or_positive, my_stream) &
                                                  bind(C, name="hip_copy_skewsymmetric_first_half_q_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int), intent(in)  :: i, matrixRows, matrixCols, negative_or_positive
      integer(kind=c_intptr_t), value  :: q_dev
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_get_skewsymmetric_second_half_q_double_c(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                         my_stream) &
                             bind(C, name="hip_get_skewsymmetric_second_half_q_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int), intent(in)  :: matrixRows, matrixCols
      integer(kind=c_intptr_t), value  :: q_dev, q2nd_dev
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_put_skewsymmetric_second_half_q_double_c(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                         my_stream) &
                             bind(C, name="hip_put_skewsymmetric_second_half_q_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int), intent(in)  :: matrixRows, matrixCols
      integer(kind=c_intptr_t), value  :: q_dev, q2nd_dev
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_zero_skewsymmetric_q_float_c(q_dev, matrixRows, matrixCols, my_stream) &
                                                  bind(C, name="hip_zero_skewsymmetric_q_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int), intent(in)  :: matrixRows, matrixCols
      integer(kind=c_intptr_t), value  :: q_dev
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_copy_skewsymmetric_second_half_q_float_c(q_dev, i, matrixRows, matrixCols, &
                                                         negative_or_positive, my_stream) &
                                                  bind(C, name="hip_copy_skewsymmetric_second_half_q_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int), intent(in)  :: i, matrixRows, matrixCols, negative_or_positive
      integer(kind=c_intptr_t), value  :: q_dev
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_copy_skewsymmetric_first_half_q_float_c(q_dev, i, matrixRows, matrixCols, &
                                                         negative_or_positive, my_stream) &
                                                  bind(C, name="hip_copy_skewsymmetric_first_half_q_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int), intent(in)  :: i, matrixRows, matrixCols, negative_or_positive
      integer(kind=c_intptr_t), value  :: q_dev
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_get_skewsymmetric_second_half_q_float_c(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                         my_stream) &
                             bind(C, name="hip_get_skewsymmetric_second_half_q_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int), intent(in)  :: matrixRows, matrixCols
      integer(kind=c_intptr_t), value  :: q_dev, q2nd_dev
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  interface
    subroutine hip_put_skewsymmetric_second_half_q_float_c(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                                         my_stream) &
                             bind(C, name="hip_put_skewsymmetric_second_half_q_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int), intent(in)  :: matrixRows, matrixCols
      integer(kind=c_intptr_t), value  :: q_dev, q2nd_dev
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

  contains

    subroutine hip_copy_real_part_to_q_double_complex(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: l_cols_nev, l_rows, matrixRows
      integer(kind=c_intptr_t)           :: q_dev, q_real_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_AMD_GPU_VERSION
      if (present(my_stream)) then
        call hip_copy_real_part_to_q_double_complex_c(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev, my_stream)
      else
        call hip_copy_real_part_to_q_double_complex_c(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev, my_stream2)
      endif
#endif

    end subroutine

    subroutine hip_copy_real_part_to_q_float_complex(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: l_cols_nev, l_rows, matrixRows
      integer(kind=c_intptr_t)           :: q_dev, q_real_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_AMD_GPU_VERSION
      if (present(my_stream)) then
        call hip_copy_real_part_to_q_float_complex_c(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev, my_stream)
      else
        call hip_copy_real_part_to_q_float_complex_c(q_dev, q_real_dev, matrixRows, l_rows, l_cols_nev, my_stream2)
      endif
#endif

    end subroutine

    subroutine hip_zero_skewsymmetric_q_double(q_dev, matrixRows, matrixCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: matrixRows, matrixCols
      integer(kind=c_intptr_t)           :: q_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_AMD_GPU_VERSION
      if (present(my_stream)) then
        call hip_zero_skewsymmetric_q_double_c(q_dev, matrixRows, matrixCols, my_stream)
      else
        call hip_zero_skewsymmetric_q_double_c(q_dev, matrixRows, matrixCols, my_stream2)
      endif
#endif

    end subroutine

    subroutine hip_copy_skewsymmetric_second_half_q_double(q_dev, i, matrixRows, matrixCols, &
                                 negative_or_positive, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: i, matrixRows, matrixCols, negative_or_positive
      integer(kind=c_intptr_t)           :: q_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_AMD_GPU_VERSION
      if (present(my_stream)) then
        call hip_copy_skewsymmetric_second_half_q_double_c(q_dev, i, matrixRows, matrixCols, &
                                       negative_or_positive, my_stream)
      else
        call hip_copy_skewsymmetric_second_half_q_double_c(q_dev, i, matrixRows, matrixCols, &
                                       negative_or_positive, my_stream2)
      endif
#endif

    end subroutine

    subroutine hip_copy_skewsymmetric_first_half_q_double(q_dev, i, matrixRows, matrixCols, &
                                 negative_or_positive, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: i, matrixRows, matrixCols, negative_or_positive
      integer(kind=c_intptr_t)           :: q_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_AMD_GPU_VERSION
      if (present(my_stream)) then
        call hip_copy_skewsymmetric_first_half_q_double_c(q_dev, i, matrixRows, matrixCols, &
                                       negative_or_positive, my_stream)
      else
        call hip_copy_skewsymmetric_first_half_q_double_c(q_dev, i, matrixRows, matrixCols, &
                                       negative_or_positive, my_stream2)
      endif
#endif

    end subroutine

    subroutine hip_get_skewsymmetric_second_half_q_double(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                 my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: matrixRows, matrixCols
      integer(kind=c_intptr_t)           :: q_dev, q2nd_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_AMD_GPU_VERSION
      if (present(my_stream)) then
        call hip_get_skewsymmetric_second_half_q_double_c(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                       my_stream)
      else
        call hip_get_skewsymmetric_second_half_q_double_c(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                       my_stream2)
      endif
#endif

    end subroutine

    subroutine hip_put_skewsymmetric_second_half_q_double(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                 my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: matrixRows, matrixCols
      integer(kind=c_intptr_t)           :: q_dev, q2nd_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_AMD_GPU_VERSION
      if (present(my_stream)) then
        call hip_put_skewsymmetric_second_half_q_double_c(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                       my_stream)
      else
        call hip_put_skewsymmetric_second_half_q_double_c(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                       my_stream2)
      endif
#endif

    end subroutine

    subroutine hip_zero_skewsymmetric_q_float(q_dev, matrixRows, matrixCols, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: matrixRows, matrixCols
      integer(kind=c_intptr_t)           :: q_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_AMD_GPU_VERSION
      if (present(my_stream)) then
        call hip_zero_skewsymmetric_q_float_c(q_dev, matrixRows, matrixCols, my_stream)
      else
        call hip_zero_skewsymmetric_q_float_c(q_dev, matrixRows, matrixCols, my_stream2)
      endif
#endif

    end subroutine

    subroutine hip_copy_skewsymmetric_second_half_q_float(q_dev, i, matrixRows, matrixCols, &
                                 negative_or_positive, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: i, matrixRows, matrixCols, negative_or_positive
      integer(kind=c_intptr_t)           :: q_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_AMD_GPU_VERSION
      if (present(my_stream)) then
        call hip_copy_skewsymmetric_second_half_q_float_c(q_dev, i, matrixRows, matrixCols, &
                                       negative_or_positive, my_stream)
      else
        call hip_copy_skewsymmetric_second_half_q_float_c(q_dev, i, matrixRows, matrixCols, &
                                       negative_or_positive, my_stream2)
      endif
#endif

    end subroutine

    subroutine hip_copy_skewsymmetric_first_half_q_float(q_dev, i, matrixRows, matrixCols, &
                                 negative_or_positive, my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: i, matrixRows, matrixCols, negative_or_positive
      integer(kind=c_intptr_t)           :: q_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_AMD_GPU_VERSION
      if (present(my_stream)) then
        call hip_copy_skewsymmetric_first_half_q_float_c(q_dev, i, matrixRows, matrixCols, &
                                       negative_or_positive, my_stream)
      else
        call hip_copy_skewsymmetric_first_half_q_float_c(q_dev, i, matrixRows, matrixCols, &
                                       negative_or_positive, my_stream2)
      endif
#endif

    end subroutine

    subroutine hip_get_skewsymmetric_second_half_q_float(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                 my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: matrixRows, matrixCols
      integer(kind=c_intptr_t)           :: q_dev, q2nd_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_AMD_GPU_VERSION
      if (present(my_stream)) then
        call hip_get_skewsymmetric_second_half_q_float_c(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                       my_stream)
      else
        call hip_get_skewsymmetric_second_half_q_float_c(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                       my_stream2)
      endif
#endif

    end subroutine

    subroutine hip_put_skewsymmetric_second_half_q_float(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                 my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=c_int), intent(in)    :: matrixRows, matrixCols
      integer(kind=c_intptr_t)           :: q_dev, q2nd_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_AMD_GPU_VERSION
      if (present(my_stream)) then
        call hip_put_skewsymmetric_second_half_q_float_c(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                       my_stream)
      else
        call hip_put_skewsymmetric_second_half_q_float_c(q_dev, q2nd_dev, matrixRows, matrixCols, &
                                       my_stream2)
      endif
#endif

    end subroutine

end module
