subroutine v_add_s_&
&PRECISION&
&(obj, v,n,s)
  use precision
  use elpa_abstract_impl
  implicit none
#include "../general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik)            :: n
  real(kind=rk)    :: v(n),s

  v(:) = v(:) + s
end subroutine v_add_s_&
    &PRECISION
