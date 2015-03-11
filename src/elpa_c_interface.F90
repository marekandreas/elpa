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
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaftrn,
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!      and
!    - IBM Deutschland GmbH
!
!
!    More information can be found here:
!    http://elpa.rzg.mpg.de/
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
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".

#include "config-f90.h"

  !c> int elpa_get_communicators(int mpi_comm_world, int my_prow, int my_pcol, int *mpi_comm_rows, int *mpi_comm_cols);
  function get_elpa_row_col_comms_wrapper(mpi_comm_world, my_prow, my_pcol, &
                                          mpi_comm_rows, mpi_comm_cols)     &
                                          result(mpierr) bind(C,name="elpa_get_communicators")
    use, intrinsic :: iso_c_binding
    use elpa1, only : get_elpa_row_col_comms

    integer(kind=c_int)         :: mpierr
    integer(kind=c_int), value  :: mpi_comm_world, my_prow, my_pcol
    integer(kind=c_int)         :: mpi_comm_rows, mpi_comm_cols

    mpierr = get_elpa_row_col_comms(mpi_comm_world, my_prow, my_pcol, &
                                    mpi_comm_rows, mpi_comm_cols)

  end function

  !c> int elpa_solve_evp_real_stage1(int na, int nev, int ncols, double *a, int lda, double *ev, double *q, int ldq, int nblk, int mpi_comm_rows, int mpi_comm_cols);
  function solve_elpa1_evp_real_wrapper(na, nev, ncols, a, lda, ev, q, ldq, nblk, &
                                  mpi_comm_rows, mpi_comm_cols)      &
                                  result(success) bind(C,name="elpa_solve_evp_real_1stage")

    use, intrinsic :: iso_c_binding
    use elpa1, only : solve_evp_real

    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, ncols, lda, ldq, nblk, mpi_comm_cols, mpi_comm_rows
    real(kind=c_double)                    :: a(1:lda,1:ncols), ev(1:na), q(1:ldq,1:ncols)

    logical                                :: successFortran

    successFortran = solve_evp_real(na, nev, a, lda, ev, q, ldq, nblk, mpi_comm_rows, mpi_comm_cols)

    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function

  ! int elpa_solve_evp_complex_stage1(int na, int nev, int ncols double_complex *a, int lda, double *ev, double_complex *q, int ldq, int nblk, int mpi_comm_rows, int mpi_comm_cols);
  function solve_evp_real_wrapper(na, nev, ncols, a, lda, ev, q, ldq, nblk, &
                                  mpi_comm_rows, mpi_comm_cols)      &
                                  result(success) bind(C,name="elpa_solve_evp_complex_1stage")

    use, intrinsic :: iso_c_binding
    use elpa1, only : solve_evp_complex

    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, ncols, lda, ldq, nblk, mpi_comm_cols, mpi_comm_rows
    complex(kind=c_double_complex)         :: a(1:lda,1:ncols), q(1:ldq,1:ncols)
    real(kind=c_double)                    :: ev(1:na)

    logical                                :: successFortran

    successFortran = solve_evp_complex(na, nev, a, lda, ev, q, ldq, nblk, mpi_comm_rows, mpi_comm_cols)

    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function

  !c> int elpa_solve_evp_real_stage2(int na, int nev, int ncols, double *a, int lda, double *ev, double *q, int ldq, int nblk, int mpi_comm_rows, int mpi_comm_cols, int THIS_REAL_ELPA_KERNEL_API, int useQR);
  function solve_elpa2_evp_real_wrapper(na, nev, ncols, a, lda, ev, q, ldq, nblk,    &
                                  mpi_comm_rows, mpi_comm_cols, mpi_comm_all, &
                                  THIS_REAL_ELPA_KERNEL_API, useQR)           &
                                  result(success) bind(C,name="elpa_solve_evp_real_2stage")

    use, intrinsic :: iso_c_binding
    use elpa2, only : solve_evp_real_2stage

    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, ncols, lda, ldq, nblk, mpi_comm_cols, mpi_comm_rows, &
                                              mpi_comm_all
    integer(kind=c_int), value, intent(in) :: THIS_REAL_ELPA_KERNEL_API, useQR
    real(kind=c_double)                    :: a(1:lda,1:ncols), ev(1:na), q(1:ldq,1:ncols)



    logical                                :: successFortran, useQRFortran

    if (useQR .eq. 0) then
      useQRFortran =.false.
    else
      useQRFortran = .true.
    endif

    successFortran = solve_evp_real_2stage(na, nev, a, lda, ev, q, ldq, nblk, mpi_comm_rows, mpi_comm_cols, mpi_comm_all, &
                                           THIS_REAL_ELPA_KERNEL_API, useQRFortran)

    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function

  ! int elpa_solve_evp_complex_stage2(int na, int nev, int ncols, double_complex *a, int lda, double *ev, double_complex *q, int ldq, int nblk, int mpi_comm_rows, int mpi_comm_cols);
  function solve_elpa2_evp_complex_wrapper(na, nev, ncols, a, lda, ev, q, ldq, nblk,    &
                                  mpi_comm_rows, mpi_comm_cols, mpi_comm_all,    &
                                  THIS_COMPLEX_ELPA_KERNEL_API)                  &
                                  result(success) bind(C,name="elpa_solve_evp_complex_2stage")

    use, intrinsic :: iso_c_binding
    use elpa2, only : solve_evp_complex_2stage

    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, ncols, lda, ldq, nblk, mpi_comm_cols, mpi_comm_rows, &
                                              mpi_comm_all
    integer(kind=c_int), value, intent(in) :: THIS_COMPLEX_ELPA_KERNEL_API
    complex(kind=c_double_complex)         :: a(1:lda,1:ncols), q(1:ldq,1:ncols)
    real(kind=c_double)                    :: ev(1:na)
    logical                                :: successFortran

    successFortran = solve_evp_complex_2stage(na, nev, a, lda, ev, q, ldq, nblk, mpi_comm_rows, mpi_comm_cols, &
                                              mpi_comm_all, THIS_COMPLEX_ELPA_KERNEL_API)

    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function

