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

 function map_global_array_index_to_local_index(iGLobal, jGlobal, iLocal, jLocal , nblk, np_rows, np_cols, my_prow, my_pcol) &
   result(possible)
   use, intrinsic :: iso_c_binding, only : c_int
   implicit none

   integer(kind=c_int)              :: pi, pj, li, lj, xi, xj
   integer(kind=c_int), intent(in)  :: iGlobal, jGlobal, nblk, np_rows, np_cols, my_prow, my_pcol
   integer(kind=c_int), intent(out) :: iLocal, jLocal
   logical                       :: possible

   possible = .true.
   iLocal = 0
   jLocal = 0

   pi = prow(iGlobal, nblk, np_rows)

   if (my_prow .ne. pi) then
     possible = .false.
     return
   endif

   pj = pcol(jGlobal, nblk, np_cols)

   if (my_pcol .ne. pj) then
     possible = .false.
     return
   endif
   li = (iGlobal-1)/(np_rows*nblk) ! block number for rows
   lj = (jGlobal-1)/(np_cols*nblk) ! block number for columns

   xi = mod( (iGlobal-1),nblk)+1   ! offset in block li
   xj = mod( (jGlobal-1),nblk)+1   ! offset in block lj

   iLocal = li * nblk + xi
   jLocal = lj * nblk + xj

 end function

