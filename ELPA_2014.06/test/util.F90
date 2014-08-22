module test_util
  implicit none
  private
  public mpi_thread_level_name
  include 'mpif.h'

  contains
!>
!> This function translates, if ELPA was build with OpenMP support,
!> the found evel of "thread safetiness" from the internal number
!> of the MPI library into a human understandable value
!>
!> \param level thread-saftiness of the MPI library
!> \return str human understandable value of thread saftiness
  pure function mpi_thread_level_name(level) result(str)
    integer, intent(in) :: level
    character(len=21) :: str
    select case(level)
      case (MPI_THREAD_SINGLE)
        str = "MPI_THREAD_SINGLE"
      case (MPI_THREAD_FUNNELED)
        str = "MPI_THREAD_FUNNELED"
      case (MPI_THREAD_SERIALIZED)
        str = "MPI_THREAD_SERIALIZED"
      case (MPI_THREAD_MULTIPLE)
        str = "MPI_THREAD_MULTIPLE"
      case default
        write(str,'(i0,1x,a)') level, "(Unknown level)"
    end select
  end function

end module
