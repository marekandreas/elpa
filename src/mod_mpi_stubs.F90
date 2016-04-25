#include "config-f90.h"
module elpa_mpi_stubs
  use precision
  implicit none

  public

  integer(kind=ik), parameter :: MPI_COMM_SELF=1, MPI_COMM_WORLD=1

  contains
    function MPI_WTIME() result(time)
      use iso_c_binding
#ifndef WITH_MPI
      use time_c
#endif
      implicit none

      real(kind=c_double) :: time
#ifndef WITH_MPI
      time = seconds()
#endif
    end function

    subroutine mpi_comm_size(mpi_comm_world, ntasks, mpierr)

      use precision

      implicit none

      integer(kind=ik), intent(in)    :: mpi_comm_world
      integer(kind=ik), intent(inout) :: ntasks
      integer(kind=ik), intent(inout) :: mpierr

      ntasks = 1
      mpierr = 0

      return

    end subroutine mpi_comm_size

    subroutine mpi_comm_rank(mpi_comm_world, myid, mpierr)
      use precision
      implicit none
      integer(kind=ik), intent(in)    :: mpi_comm_world
      integer(kind=ik), intent(inout) :: mpierr
      integer(kind=ik), intent(inout) :: myid

      myid = 0
      mpierr = 0

      return
    end subroutine mpi_comm_rank

    subroutine mpi_comm_split(mpi_communicator, color, key, new_comm, mpierr)
      use precision
      implicit none
      integer(kind=ik), intent(in)    :: mpi_communicator, color, key
      integer(kind=ik), intent(inout) :: new_comm, mpierr

      new_comm = mpi_communicator
      mpierr = 0
      return
    end subroutine mpi_comm_split

end module
