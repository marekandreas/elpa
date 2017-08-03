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
!
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
! Author: A. Marek, MPCDF



#include "../general/sanity.F90"

      use elpa1_compute, solve_tridi_&
                         &PRECISION&
			 &_private_impl => solve_tridi_&
			 &PRECISION&
			 &_impl
      use precision
      use elpa_abstract_impl

      implicit none
      class(elpa_abstract_impl_t), intent(inout) :: obj
      integer(kind=ik)         :: na, nev, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
      real(kind=REAL_DATATYPE) :: d(obj%na), e(obj%na)
#ifdef USE_ASSUMED_SIZE
      real(kind=REAL_DATATYPE) :: q(obj%local_nrows,*)
#else
      real(kind=REAL_DATATYPE) :: q(obj%local_nrows, obj%local_ncols)
#endif

      logical                  :: wantDebug
      logical                  :: success

      integer                  :: debug

      call obj%timer%start("elpa_solve_tridi_public_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      &")
      na         = obj%na
      nev        = obj%nev
      nblk       = obj%nblk
      ldq        = obj%local_nrows
      matrixCols = obj%local_ncols

      call obj%get("mpi_comm_rows", mpi_comm_rows)
      call obj%get("mpi_comm_cols", mpi_comm_cols)

      call obj%get("debug",debug)
      if (debug == 1) then
        wantDebug = .true.
      else
        wantDebug = .false.
      endif
      success = .false.

      call solve_tridi_&
      &PRECISION&
      &_private_impl(obj, na, nev, d, e, q, ldq, nblk, matrixCols, &
               mpi_comm_rows, mpi_comm_cols, wantDebug, success)

      call obj%timer%stop("elpa_solve_tridi_public_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      &")


#undef REALCASE
#undef COMPLEXCASE
#undef DOUBLE_PRECISION
#undef SINGLE_PRECISION

