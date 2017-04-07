
module assert
  implicit none
  contains
    subroutine x_assert(condition, condition_string, file, line)
      use elpa_utilities, only : error_unit
      logical, intent(in) :: condition
      character(len=*), intent(in) :: condition_string
      character(len=*), intent(in) :: file
      integer, intent(in) :: line

      if (.not. condition) then
        write(error_unit,'(a,i0)') "Assertion failed:" // condition_string // " at " // file // ":", line
      end if
    end subroutine
end module


