module baz
  use bar
  implicit none
  contains
  function two_times_two() result(t)
    integer :: t
    t = 2 * two()
  end function
end module

#ifdef PROGRAM_test_baz
program test_bar
  use baz
  if (two_times_two() /= 4) then
    stop 1
  endif
end program
#endif
