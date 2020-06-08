subroutine check_monotony_&
&PRECISION&
&(obj, n,d,text, wantDebug, success)
  ! This is a test routine for checking if the eigenvalues are monotonically increasing.
  ! It is for debug purposes only, an error should never be triggered!
  use precision
  use ELPA_utilities
  use elpa_abstract_impl
  implicit none

  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik)              :: n
  real(kind=REAL_DATATYPE)      :: d(n)
  character*(*)                 :: text

  integer(kind=ik)              :: i
  logical, intent(in)           :: wantDebug
  logical, intent(out)          :: success

  success = .true.
  do i=1,n-1
    if (d(i+1)<d(i)) then
      if (wantDebug) write(error_unit,'(a,a,i8,2g25.17)') 'ELPA1_check_monotony: Monotony error on ',text,i,d(i),d(i+1)
      success = .false.
      return
    endif
  enddo
end subroutine check_monotony_&
        &PRECISION
