subroutine add_tmp_&
&PRECISION&
&(obj, d1, dbase, ddiff, z, ev_scale_value, na1,i)
  use precision
  use v_add_s
  use elpa_abstract_impl
  implicit none
  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik), intent(in) :: na1, i

  real(kind=REAL_DATATYPE), intent(in)    :: d1(:), dbase(:), ddiff(:), z(:)
  real(kind=REAL_DATATYPE), intent(inout) :: ev_scale_value
  real(kind=REAL_DATATYPE)                :: tmp(1:na1)

  ! tmp(1:na1) = z(1:na1) / delta(1:na1,i)  ! original code
  ! tmp(1:na1) = z(1:na1) / (d1(1:na1)-d(i))! bad results

  ! All we want to calculate is tmp = (d1(1:na1)-dbase(i))+ddiff(i)
  ! in exactly this order, but we want to prevent compiler optimization

  tmp(1:na1) = d1(1:na1) -dbase(i)
  call v_add_s_&
  &PRECISION&
  &(obj, tmp(1:na1),na1,ddiff(i))
  tmp(1:na1) = z(1:na1) / tmp(1:na1)
  ev_scale_value = 1.0_rk/sqrt(dot_product(tmp(1:na1),tmp(1:na1)))

end subroutine add_tmp_&
&PRECISION

