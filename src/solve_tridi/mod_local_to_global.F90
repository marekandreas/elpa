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
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
! Copyright Andreas Marek, MPCDF
#endif

#include "config-f90.h"
module mod_local_to_global
  use precision
  implicit none
  private

  public :: local_to_global
  contains


     subroutine local_to_global(ldq, matrixCols, n, m, local_index, rowGlobal, colGlobal)
       use precision

       implicit none
       integer(kind=ik), intent(in)  :: ldq, matrixCols, n, m
       integer(kind=ik), intent(out) :: rowGlobal, colGlobal
       integer(kind=ik), intent(in)  :: local_index

       integer(kind=ik)              :: number_of_entries, entries_in_started_col, &
                                        colums_in_sub_matrix, entries_in_sub_matrix
       integer(kind=ik)              :: columns_in_sub_matrix, index_sub_matrix, row_sub_matrix, &
                                        col_sub_matrix

       number_of_entries = (matrixCols-m)*ldq+ldq-n+1

       entries_in_started_col=ldq-n+1
       entries_in_sub_matrix=number_of_entries-entries_in_started_col
       if (mod(entries_in_sub_matrix,ldq) .ne. 0) then
         print *,"submatrix dimensions wrong!"
       endif
       columns_in_sub_matrix=entries_in_sub_matrix/ldq

       if (local_index .gt. entries_in_started_col) then
         index_sub_matrix = local_index - entries_in_started_col
         col_sub_matrix = (index_sub_matrix -1)/ ldq + 1
         row_sub_matrix = mod(index_sub_matrix,ldq)
         if (row_sub_matrix .eq. 0) row_sub_matrix=ldq

         colGlobal = col_sub_matrix + m
         rowGlobal = row_sub_matrix
       else
         colGlobal = m
         rowGlobal = n + local_index-1
       endif


     end subroutine
end module
