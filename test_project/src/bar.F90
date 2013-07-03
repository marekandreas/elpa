module bar
  implicit none
  contains
  function two() result(t)
    integer :: t
    t = 2
  end function
end module

#ifdef PROGRAM_test_bar
program test_bar
  use bar, only : two
  if (two() /= 2) then
    stop 1
  endif
end program
#endif
