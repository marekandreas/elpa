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
! Author: Peter Karpov, MPCDF
#endif

#include "../general/error_checking.inc"


! aux_mat_full: 

! if (a_transposed) then
!   allocate(aux_a_full(nblk_mult, nblk_mult), stat=istat, errmsg=errorMessage)
!   allocate(aux_b_full(nblk_mult, nblk_mult_max*(np_dirs_fine/np_dirs_t)), stat=istat, errmsg=errorMessage)

! else if (b_transposed) then
!   allocate(aux_a_full(nblk_mult_max*(np_dirs_fine/np_dirs_t), nblk_mult), stat=istat, errmsg=errorMessage)
!   allocate(aux_b_full(nblk_mult, nblk_mult), stat=istat, errmsg=errorMessage)

subroutine set_zeros_in_unused_block_part_&
                            &MATH_DATATYPE&
                            &_&
                            &PRECISION&
                            (aux_mat_full, nblk, nblk_rows_cut, nblk_cols_cut, &
                            i_block_loc_fine, j_block_loc_fine, shift_i, shift_j)

  use, intrinsic :: iso_c_binding
  use precision
  implicit none
#include "../../src/general/precision_kinds.F90"

  MATH_DATATYPE(kind=rck), pointer, contiguous :: aux_mat_full(:,:)
  integer(kind=ik)        :: nblk, nblk_rows_cut, nblk_cols_cut
  integer(kind=ik)        :: i_block_loc_fine, j_block_loc_fine
  integer(kind=ik), value :: shift_i, shift_j

  if (nblk_rows_cut<nblk .and. nblk_cols_cut>0) then ! for negative nblk_rows_cut we nullify the whole block (it's locally absent)                      
    aux_mat_full(1+max(nblk_rows_cut,0) + i_block_loc_fine*nblk + shift_i : &
                 nblk                   + i_block_loc_fine*nblk + shift_i,  &
                 1                      + j_block_loc_fine*nblk + shift_j : &
                 nblk_cols_cut          + j_block_loc_fine*nblk + shift_j) = 0
  endif

  if (nblk_cols_cut<nblk .and. nblk_rows_cut>0) then
    aux_mat_full(1                      + i_block_loc_fine*nblk + shift_i : &
                 nblk_rows_cut          + i_block_loc_fine*nblk + shift_i,  &
                 1+max(nblk_cols_cut,0) + j_block_loc_fine*nblk + shift_j : &
                 nblk                   + j_block_loc_fine*nblk + shift_j) = 0
  endif

  if (nblk_rows_cut<nblk .and. nblk_cols_cut<nblk) then
    aux_mat_full(1+max(nblk_rows_cut,0) + i_block_loc_fine*nblk + shift_i: &
                 nblk                   + i_block_loc_fine*nblk + shift_i, &
                 1+max(nblk_cols_cut,0) + j_block_loc_fine*nblk + shift_j : &
                 nblk                   + j_block_loc_fine*nblk + shift_j) = 0
  endif

end subroutine

