subroutine distribute_global_column_&
&PRECISION&
&(obj, g_col, l_col, noff, nlen, my_prow, np_rows, nblk)
  use precision
  use elpa_abstract_impl
  implicit none
#include "../general/precision_kinds.F90"

  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik)             :: noff, nlen, my_prow, np_rows, nblk
  real(kind=rk)     :: g_col(nlen), l_col(*) ! chnage this to proper 2d 1d matching ! remove assumed size ! q(1,noff+i) ! q(ldq,matrixCols)

  integer(kind=ik)  :: nbs, nbe, jb, g_off, l_off, js, je

  nbs = noff/(nblk*np_rows)
  nbe = (noff+nlen-1)/(nblk*np_rows)


  ! kernel

  do jb = nbs, nbe
    g_off = jb*nblk*np_rows + nblk*my_prow
    l_off = jb*nblk

    js = MAX(noff+1-g_off,1)
    je = MIN(noff+nlen-g_off,nblk)

    if (je<js) cycle

    l_col(l_off+js:l_off+je) = g_col(g_off+js-noff:g_off+je-noff)

  enddo
end subroutine distribute_global_column_&
&PRECISION


subroutine distribute_global_column_4_&
&PRECISION&
&(obj, g_col, l_col, g_col_dim1, g_col_dim2, ldq, matrixCols, &
  noff_in, noff, nlen, my_prow, np_rows, nblk)
  use precision
  use mod_local_to_global
  use elpa_abstract_impl
  implicit none
#include "../general/precision_kinds.F90"

  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik), intent(in)               :: g_col_dim1, g_col_dim2
  integer(kind=ik)                           :: g_col_offset1, g_col_offset2
  integer(kind=ik), intent(in)               :: ldq, matrixCols
  integer(kind=ik)                           :: l_col_offset1, l_col_offset2
  integer(kind=ik), intent(in)               :: noff_in

  integer(kind=ik)                           :: noff, nlen, my_prow, np_rows, nblk
  real(kind=rk)                              :: g_col(1:g_col_dim1,1:g_col_dim2), l_col(1:ldq,1:matrixCols)
  integer(kind=ik)                           :: nbs, nbe, jb, g_off, l_off, js, je
  integer(kind=ik)                           :: g_col_global_row, g_col_global_col
  integer(kind=ik)                           :: l_col_global_row, l_col_global_col, ii, i

  nbs = noff/(nblk*np_rows)
  nbe = (noff+nlen-1)/(nblk*np_rows)


  do i=1, nlen

    g_col_offset1 = 1
    g_col_offset2 = i

    l_col_offset1 = 1
    l_col_offset2 = noff_in+i

    do jb = nbs, nbe
      g_off = jb*nblk*np_rows + nblk*my_prow
      l_off = jb*nblk

      js = MAX(noff+1-g_off,1)
      je = MIN(noff+nlen-g_off,nblk)

      if (je<js) cycle
       do ii = js, je
         call local_to_global(g_col_dim1, g_col_dim2, g_col_offset1, g_col_offset2, &
                              g_off-noff+ii, g_col_global_row, g_col_global_col)

         call local_to_global(ldq, matrixCols, l_col_offset1, l_col_offset2, &
                              l_off+ii, l_col_global_row, l_col_global_col)

         l_col(l_col_global_row, l_col_global_col) = g_col(g_col_global_row, g_col_global_col)
      enddo

    enddo
  enddo


end subroutine distribute_global_column_4_&
&PRECISION
