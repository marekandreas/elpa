module init_elpa

  private
  public :: elpa_init, initDone

  logical :: initDone = .false.

  contains

  subroutine elpa_init()

    implicit none

    ! must be done by all task using ELPA !!!

    initDone = .true.

  end subroutine


end module init_elpa
