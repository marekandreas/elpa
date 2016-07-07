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
!
#include "config-f90.h"
module from_c
  implicit none

  public

  interface
    integer(kind=c_int) function elpa1_real_c_double(na, nev,  a, lda, ev, q, ldq,         &
                                       nblk, matrixCols, mpi_comm_rows, mpi_comm_cols ) &
                                       bind(C, name="call_elpa1_real_solver_from_c_double")

      use iso_c_binding
      implicit none

      integer(kind=c_int), value :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
!#ifdef DOUBLE_PRECISION_REAL
      real(kind=c_double)        :: a(1:lda,1:matrixCOls), ev(1:na), q(1:ldq,1:matrixCols)
!#else
!      real(kind=c_float)         :: a(1:lda,1:matrixCOls), ev(1:na), q(1:ldq,1:matrixCols)
!#endif
    end function elpa1_real_c_double


  end interface

#ifdef WANT_SINGLE_PRECISION_REAL

  interface
    integer(kind=c_int) function elpa1_real_c_single(na, nev,  a, lda, ev, q, ldq,         &
                                       nblk, matrixCols, mpi_comm_rows, mpi_comm_cols ) &
                                       bind(C, name="call_elpa1_real_solver_from_c_single")

      use iso_c_binding
      implicit none

      integer(kind=c_int), value :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
!#ifdef DOUBLE_PRECISION_REAL
!      real(kind=c_double)        :: a(1:lda,1:matrixCOls), ev(1:na), q(1:ldq,1:matrixCols)
!#else
      real(kind=c_float)         :: a(1:lda,1:matrixCOls), ev(1:na), q(1:ldq,1:matrixCols)
!#endif
    end function elpa1_real_c_single

  end interface
#endif

  interface
    integer(kind=c_int) function elpa1_complex_c_double(na, nev,  a, lda, ev, q, ldq,         &
                                       nblk, matrixCols, mpi_comm_rows, mpi_comm_cols ) &
                                       bind(C, name="call_elpa1_complex_solver_from_c_double")

      use iso_c_binding
      implicit none

      integer(kind=c_int), value  :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
!#ifdef DOUBLE_PRECISION_COMPLEX
      real(kind=c_double)         :: ev(1:na)
      complex(kind=c_double)      :: a(1:lda,1:matrixCOls), q(1:ldq,1:matrixCols)
!#else
!      real(kind=c_float)          :: ev(1:na)
!      complex(kind=c_float)      :: a(1:lda,1:matrixCOls), q(1:ldq,1:matrixCols)
!#endif
    end function elpa1_complex_c_double


  end interface

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  interface
    integer(kind=c_int) function elpa1_complex_c_single(na, nev,  a, lda, ev, q, ldq,         &
                                       nblk, matrixCols, mpi_comm_rows, mpi_comm_cols ) &
                                       bind(C, name="call_elpa1_complex_solver_from_c_single")

      use iso_c_binding
      implicit none

      integer(kind=c_int), value  :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
!#ifdef DOUBLE_PRECISION_COMPLEX
!      real(kind=c_double)         :: ev(1:na)
!      complex(kind=c_double)      :: a(1:lda,1:matrixCOls), q(1:ldq,1:matrixCols)
!#else
      real(kind=c_float)          :: ev(1:na)
      complex(kind=c_float)      :: a(1:lda,1:matrixCOls), q(1:ldq,1:matrixCols)
!#endif
    end function elpa1_complex_c_single

  end interface

#endif
  interface
    integer(kind=c_int) function elpa_get_comm_c(mpi_comm_world, my_prow, my_pcol, &
                                                 mpi_comm_rows, mpi_comm_cols)     &
                                                 bind(C, name="call_elpa_get_comm_from_c")
      use iso_c_binding
      implicit none
      integer(kind=c_int), value :: mpi_comm_world, my_prow, my_pcol
      integer(kind=c_int)        :: mpi_comm_rows, mpi_comm_cols

    end function
  end interface

  contains

  function solve_elpa1_real_call_from_c_double(na, nev, a, lda, ev, q, ldq,         &
                      nblk, matrixCOls, mpi_comm_rows, mpi_comm_cols ) &
                      result(success)
    use precision
    use iso_c_binding
    implicit none

    integer(kind=ik) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
    logical          :: success
    integer(kind=ik) :: successC
!#ifdef DOUBLE_PRECISION_REAL
    real(kind=c_double)  :: a(1:lda,1:matrixCols), ev(1:na), q(1:ldq,1:matrixCols)
!#else
!    real(kind=c_float)   :: a(1:lda,1:matrixCols), ev(1:na), q(1:ldq,1:matrixCols)
!#endif
    successC = elpa1_real_c_double(na, nev, a, lda, ev, q, ldq, nblk, &
                            matrixCols, mpi_comm_rows, mpi_comm_cols)

    if (successC .eq. 1) then
      success = .true.
    else
      success = .false.
    endif

  end function

#ifdef WANT_SINGLE_PRECISION_REAL
  function solve_elpa1_real_call_from_c_single(na, nev, a, lda, ev, q, ldq,         &
                      nblk, matrixCOls, mpi_comm_rows, mpi_comm_cols ) &
                      result(success)
    use precision
    use iso_c_binding
    implicit none

    integer(kind=ik) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
    logical          :: success
    integer(kind=ik) :: successC
!#ifdef DOUBLE_PRECISION_REAL
!    real(kind=c_double)  :: a(1:lda,1:matrixCols), ev(1:na), q(1:ldq,1:matrixCols)
!#else
    real(kind=c_float)   :: a(1:lda,1:matrixCols), ev(1:na), q(1:ldq,1:matrixCols)
!#endif
    successC = elpa1_real_c_single(na, nev, a, lda, ev, q, ldq, nblk, &
                            matrixCols, mpi_comm_rows, mpi_comm_cols)

    if (successC .eq. 1) then
      success = .true.
    else
      success = .false.
    endif

  end function
#endif

  function solve_elpa1_complex_call_from_c_double(na, nev, a, lda, ev, q, ldq,         &
                      nblk, matrixCOls, mpi_comm_rows, mpi_comm_cols ) &
                      result(success)

    use precision
    use iso_c_binding
    implicit none

    integer(kind=ik) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
    logical          :: success
    integer(kind=ik) :: successC
!#ifdef DOUBLE_PRECISION_COMPLEX
    real(kind=c_double)    :: ev(1:na)
    complex(kind=c_double) :: a(1:lda,1:matrixCols), q(1:ldq,1:matrixCols)
!#else
!    real(kind=c_float)     :: ev(1:na)
!    complex(kind=c_float)  :: a(1:lda,1:matrixCols), q(1:ldq,1:matrixCols)
!#endif

    successC = elpa1_complex_c_double(na, nev, a, lda, ev, q, ldq, nblk, &
                            matrixCols, mpi_comm_rows, mpi_comm_cols)

    if (successC .eq. 1) then
      success = .true.
    else
      success = .false.
    endif

  end function

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  function solve_elpa1_complex_call_from_c_single(na, nev, a, lda, ev, q, ldq,         &
                      nblk, matrixCOls, mpi_comm_rows, mpi_comm_cols ) &
                      result(success)

    use precision
    use iso_c_binding
    implicit none

    integer(kind=ik) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
    logical          :: success
    integer(kind=ik) :: successC
!#ifdef DOUBLE_PRECISION_COMPLEX
!    real(kind=c_double)    :: ev(1:na)
!    complex(kind=c_double) :: a(1:lda,1:matrixCols), q(1:ldq,1:matrixCols)
!#else
    real(kind=c_float)     :: ev(1:na)
    complex(kind=c_float)  :: a(1:lda,1:matrixCols), q(1:ldq,1:matrixCols)
!#endif

    successC = elpa1_complex_c_single(na, nev, a, lda, ev, q, ldq, nblk, &
                            matrixCols, mpi_comm_rows, mpi_comm_cols)

    if (successC .eq. 1) then
      success = .true.
    else
      success = .false.
    endif

  end function

#endif

  function call_elpa_get_comm_from_c(mpi_comm_world, my_prow, my_pcol, &
                                     mpi_comm_rows, mpi_comm_cols) result(mpierr)

      use precision
      use iso_c_binding
      implicit none

      integer(kind=ik) :: mpierr
      integer(kind=ik) :: mpi_comm_world, my_prow, my_pcol, &
                          mpi_comm_rows, mpi_comm_cols

      mpierr = elpa_get_comm_c(mpi_comm_world, my_prow, my_pcol, &
                                    mpi_comm_rows, mpi_comm_cols)
  end function
end module from_c
