module init_elpa

  private
  public :: elpa_init, elpa_initialized, elpa_uninit

  logical :: initDone = .false.

  contains

  subroutine elpa_init()
    implicit none
    initDone = .true.
  end subroutine

  function elpa_initialized() result(state)
    logical :: state
    state = initDone
  end function

  subroutine elpa_uninit()
  end subroutine

end module init_elpa
