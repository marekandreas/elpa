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
! Author: A. Marek, MPCDF

function elpa_cholesky_&
 &MATH_DATATYPE&
 &_wrapper_&
 &PRECISION&
 & (na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug) result(success) &
       bind(C,name="elpa_cholesky_&
       &MATH_DATATYPE&
       &_&
       &PRECISION&
       &")

   use, intrinsic :: iso_c_binding
   use elpa1_auxiliary, only : elpa_cholesky_&
       &MATH_DATATYPE&
       &_&
       &PRECISION

   implicit none

   integer(kind=c_int), value :: na, lda, nblk, matrixCols,  mpi_comm_rows, mpi_comm_cols, wantDebug
   integer(kind=c_int)        :: success
#if REALCASE == 1
#ifdef USE_ASSUMED_SIZE
   real(kind=C_DATATYPE_KIND) :: a(lda,*)
#else
   real(kind=C_DATATYPE_KIND) :: a(lda,matrixCols)
#endif
#endif
#if COMPLEXCASE == 1
#ifdef USE_ASSUMED_SIZE
   complex(kind=C_DATATYPE_KIND) :: a(lda,*)
#else
   complex(kind=C_DATATYPE_KIND) :: a(lda,matrixCols)
#endif
#endif


   logical                    :: successFortran, wantDebugFortran

   if (wantDebug .ne. 0) then
     wantDebugFortran = .true.
   else
     wantDebugFortran = .false.
   endif

   successFortran = elpa_cholesky_&
   &MATH_DATATYPE&
   &_&
   &PRECISION&
   & (na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebugFortran)

   if (successFortran) then
     success = 1
   else
     success = 0
   endif

 end function

