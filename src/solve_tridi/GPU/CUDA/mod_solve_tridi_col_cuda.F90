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

module solve_tridi_col_cuda
  use, intrinsic :: iso_c_binding
  use precision

  implicit none

  public

  interface
    subroutine cuda_update_d_double_c(limits_dev, d_dev, e_dev, ndiv, na, &
                                                          my_stream) &
                                                     bind(C, name="cuda_update_d_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: d_dev, e_dev
      type(c_ptr), value               :: limits_dev
      integer(kind=c_int), intent(in)  :: ndiv, na
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface



  interface
    subroutine cuda_copy_qmat1_to_qmat2_double_c(qmat1_dev, qmat2_dev, max_size, &
                                                          my_stream) &
                                                     bind(C, name="cuda_copy_qmat1_to_qmat2_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: qmat1_dev, qmat2_dev
      integer(kind=c_int), intent(in)  :: max_size
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface



  interface
    subroutine cuda_copy_d_to_d_tmp_double_c(d_dev, d_tmp_dev, na, &
                                                          my_stream) &
                                                     bind(C, name="cuda_copy_d_to_d_tmp_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: d_dev, d_tmp_dev
      integer(kind=c_int), intent(in)  :: na
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface



#ifdef WANT_SINGLE_PRECISION_REAL
  interface
    subroutine cuda_update_d_float_c(limits_dev, d_dev, e_dev, ndiv, na, &
                                                          my_stream) &
                                                     bind(C, name="cuda_update_d_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: d_dev, e_dev
      type(c_ptr), value               :: limits_dev
      integer(kind=c_int), intent(in)  :: ndiv, na
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

#endif


#ifdef WANT_SINGLE_PRECISION_REAL
  interface
    subroutine cuda_copy_qmat1_to_qmat2_float_c(qmat1_dev, qmat2_dev, max_size, &
                                                          my_stream) &
                                                     bind(C, name="cuda_copy_qmat1_to_qmat2_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: qmat1_dev, qmat2_dev
      integer(kind=c_int), intent(in)  :: max_size
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

#endif


#ifdef WANT_SINGLE_PRECISION_REAL
  interface
    subroutine cuda_copy_d_to_d_tmp_float_c(d_dev, d_tmp_dev, na, &
                                                          my_stream) &
                                                     bind(C, name="cuda_copy_d_to_d_tmp_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: d_dev, d_tmp_dev
      integer(kind=c_int), intent(in)  :: na
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface

#endif


  interface
    subroutine cuda_copy_q_to_q_tmp_double_c(q_dev, q_tmp_dev, ldq, nlen, &
                                                          my_stream) &
                              bind(C, name="cuda_copy_q_to_q_tmp_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: q_dev, q_tmp_dev
      integer(kind=c_int), intent(in)  :: ldq, nlen
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface


#ifdef WANT_SINGLE_PRECISION_REAL
  interface
    subroutine cuda_copy_q_to_q_tmp_float_c(q_dev, q_tmp_dev, ldq, nlen, &
                                                          my_stream) &
                              bind(C, name="cuda_copy_q_to_q_tmp_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: q_dev, q_tmp_dev
      integer(kind=c_int), intent(in)  :: ldq, nlen
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface
#endif


  interface
    subroutine cuda_copy_q_tmp_to_q_double_c(q_tmp_dev, q_dev, ldq, nlen, &
                                                          my_stream) &
                              bind(C, name="cuda_copy_q_tmp_to_q_double_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: q_dev, q_tmp_dev
      integer(kind=c_int), intent(in)  :: ldq, nlen
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface


#ifdef WANT_SINGLE_PRECISION_REAL
  interface
    subroutine cuda_copy_q_tmp_to_q_float_c(q_tmp_dev, q_dev, ldq, nlen, &
                                                          my_stream) &
                              bind(C, name="cuda_copy_q_tmp_to_q_float_FromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: q_dev, q_tmp_dev
      integer(kind=c_int), intent(in)  :: ldq, nlen
      integer(kind=c_intptr_t), value  :: my_stream
    end subroutine
  end interface
#endif


  contains
    subroutine cuda_update_d_double(limits_dev, d_dev, e_dev, ndiv, na, &
                                         my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in)    :: ndiv, na
      integer(kind=C_intptr_T)           :: d_dev, e_dev
      type(c_ptr)                        :: limits_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      if (present(my_stream)) then
        call cuda_update_d_double_c(limits_dev, d_dev, e_dev, ndiv, na, my_stream)
      else
        call cuda_update_d_double_c(limits_dev, d_dev, e_dev, ndiv, na, my_stream2)
      endif
#endif

    end subroutine


    subroutine cuda_copy_qmat1_to_qmat2_double(qmat1_dev, qmat2_dev, max_size, &
                                              my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in)    :: max_size
      integer(kind=C_intptr_T)           :: qmat1_dev, qmat2_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      if (present(my_stream)) then
        call cuda_copy_qmat1_to_qmat2_double_c(qmat1_dev, qmat2_dev, max_size, &
                                                  my_stream)
      else
        call cuda_copy_qmat1_to_qmat2_double_c(qmat1_dev, qmat2_dev, max_size, &
                                                  my_stream2)
      endif
#endif

    end subroutine


    subroutine cuda_copy_d_to_d_tmp_double(d_dev, d_tmp_dev, na, &
                                               my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in)    :: na
      integer(kind=C_intptr_T)           :: d_dev, d_tmp_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      if (present(my_stream)) then
        call cuda_copy_d_to_d_tmp_double_c(d_dev, d_tmp_dev, na, my_stream)
      else
        call cuda_copy_d_to_d_tmp_double_c(d_dev, d_tmp_dev, na, my_stream2)
      endif
#endif

    end subroutine


#ifdef WANT_SINGLE_PRECISION_REAL
    subroutine cuda_update_d_float(limits_dev, d_dev, e_dev, ndiv, na, &
                                         my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in)    :: ndiv, na
      integer(kind=C_intptr_T)           :: d_dev, e_dev
      type(c_ptr)                        :: limits_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      if (present(my_stream)) then
        call cuda_update_d_float_c(limits_dev, d_dev, e_dev, ndiv, na, my_stream)
      else
        call cuda_update_d_float_c(limits_dev, d_dev, e_dev, ndiv, na, my_stream2)
      endif
#endif

    end subroutine
#endif


#ifdef WANT_SINGLE_PRECISION_REAL
    subroutine cuda_copy_qmat1_to_qmat2_float(qmat1_dev, qmat2_dev, max_size, &
                                              my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in)    :: max_size
      integer(kind=C_intptr_T)           :: qmat1_dev, qmat2_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      if (present(my_stream)) then
        call cuda_copy_qmat1_to_qmat2_float_c(qmat1_dev, qmat2_dev, max_size, &
                                                  my_stream)
      else
        call cuda_copy_qmat1_to_qmat2_float_c(qmat1_dev, qmat2_dev, max_size, &
                                                  my_stream2)
      endif
#endif

    end subroutine
#endif


#ifdef WANT_SINGLE_PRECISION_REAL
    subroutine cuda_copy_d_to_d_tmp_float(d_dev, d_tmp_dev, na, &
                                               my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in)    :: na
      integer(kind=C_intptr_T)           :: d_dev, d_tmp_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      if (present(my_stream)) then
        call cuda_copy_d_to_d_tmp_float_c(d_dev, d_tmp_dev, na, my_stream)
      else
        call cuda_copy_d_to_d_tmp_float_c(d_dev, d_tmp_dev, na, my_stream2)
      endif
#endif

    end subroutine
#endif


    subroutine cuda_copy_q_to_q_tmp_double(q_dev, q_tmp_dev, ldq, nlen, &
                                              my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in)    :: ldq, nlen
      integer(kind=C_intptr_T)           :: q_dev, q_tmp_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      if (present(my_stream)) then
        call cuda_copy_q_to_q_tmp_double_c(q_dev, q_tmp_dev, ldq, nlen, &
                                                  my_stream)
      else
        call cuda_copy_q_to_q_tmp_double_c(q_dev, q_tmp_dev, ldq, nlen, &
                                                  my_stream2)
      endif
#endif

    end subroutine


#ifdef WANT_SINGLE_PRECISION_REAL
    subroutine cuda_copy_q_to_q_tmp_float(q_dev, q_tmp_dev, ldq, nlen, &
                                              my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in)    :: ldq, nlen
      integer(kind=C_intptr_T)           :: q_dev, q_tmp_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      if (present(my_stream)) then
        call cuda_copy_q_to_q_tmp_float_c(q_dev, q_tmp_dev, ldq, nlen, &
                                                  my_stream)
      else
        call cuda_copy_q_to_q_tmp_float_c(q_dev, q_tmp_dev, ldq, nlen, &
                                                  my_stream2)
      endif
#endif

    end subroutine
#endif



    subroutine cuda_copy_q_tmp_to_q_double(q_tmp_dev, q_dev, ldq, nlen, &
                                              my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in)    :: ldq, nlen
      integer(kind=C_intptr_T)           :: q_dev, q_tmp_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      if (present(my_stream)) then
        call cuda_copy_q_tmp_to_q_double_c(q_tmp_dev, q_dev, ldq, nlen, &
                                                  my_stream)
      else
        call cuda_copy_q_tmp_to_q_double_c(q_tmp_dev, q_dev, ldq, nlen, &
                                                  my_stream2)
      endif
#endif

    end subroutine



#ifdef WANT_SINGLE_PRECISION_REAL
    subroutine cuda_copy_q_tmp_to_q_float(q_tmp_dev, q_dev, ldq, nlen, &
                                              my_stream)
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT), intent(in)    :: ldq, nlen
      integer(kind=C_intptr_T)           :: q_dev, q_tmp_dev
      integer(kind=c_intptr_t), optional :: my_stream
      integer(kind=c_intptr_t)           :: my_stream2

#ifdef WITH_NVIDIA_GPU_VERSION
      if (present(my_stream)) then
        call cuda_copy_q_tmp_to_q_float_c(q_tmp_dev, q_dev, ldq, nlen, &
                                                  my_stream)
      else
        call cuda_copy_q_tmp_to_q_float_c(q_tmp_dev, q_dev, ldq, nlen, &
                                                  my_stream2)
      endif
#endif

    end subroutine
#endif


end module
