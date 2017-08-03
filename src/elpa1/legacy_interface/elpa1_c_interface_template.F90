#if 0
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
!    This particular source code file contains additions, changes and
!    enhancements authored by Intel Corporation which is not part of
!    the ELPA consortium.
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
! Author: Andreas Marek, MPCDF
#endif

  function solve_elpa1_evp_&
  &MATH_DATATYPE&
  &_wrapper_&
  &PRECISION&
  & (na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all, &
     useGPU) result(success) bind(C,name="elpa_solve_evp_&
     &MATH_DATATYPE&
     &_1stage_&
     &PRECISION&
     &_precision")

    use, intrinsic :: iso_c_binding
    use elpa1

    implicit none
    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_cols, mpi_comm_rows, mpi_comm_all
    integer(kind=c_int), value, intent(in) :: useGPU
    real(kind=C_DATATYPE_KIND)             :: ev(1:na)

#if REALCASE == 1

#ifdef USE_ASSUMED_SIZE
    real(kind=C_DATATYPE_KIND)             :: a(lda,*), q(ldq,*)
#else
    real(kind=C_DATATYPE_KIND)             :: a(1:lda,1:matrixCols), q(1:ldq,1:matrixCols)
#endif

#endif /* REALCASE */

#if COMPLEXCASE == 1

#ifdef USE_ASSUMED_SIZE
    complex(kind=C_DATATYPE_KIND)          :: a(lda,*), q(ldq,*)
#else
    complex(kind=C_DATATYPE_KIND)          :: a(1:lda,1:matrixCols), q(1:ldq,1:matrixCols)
#endif

#endif /* COMPLEXCASE == 1 */
    logical                                :: successFortran

    successFortran = elpa_solve_evp_&
    &MATH_DATATYPE&
    &_1stage_&
    &PRECISION &
    & (na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all, &
       useGPU == 1)

    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function
