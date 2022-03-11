subroutine distribute_global_column_&
&PRECISION&
&(obj, g_col, l_col, noff, nlen, my_prow, np_rows, nblk)
  use precision
  use elpa_abstract_impl
  implicit none
#include "../general/precision_kinds.F90"

  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik)             :: noff, nlen, my_prow, np_rows, nblk
  real(kind=rk)     :: g_col(nlen), l_col(*) ! chnage this to proper 2d 1d matching ! remove assumed size

  integer(kind=ik)  :: nbs, nbe, jb, g_off, l_off, js, je

  nbs = noff/(nblk*np_rows)
  nbe = (noff+nlen-1)/(nblk*np_rows)

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
