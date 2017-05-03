! The ELPA public API

module elpa
  use elpa_api
  use elpa_constants

  implicit none
  public

  contains

    function elpa_allocate() result(obj)
      use elpa_impl
      class(elpa_t), pointer :: obj
      obj => elpa_impl_allocate()
    end function


    subroutine elpa_deallocate(obj)
      class(elpa_t), pointer :: obj
      call obj%destroy()
      deallocate(obj)
    end subroutine

end module
