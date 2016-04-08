module aligned_mem
  use, intrinsic :: iso_c_binding

  interface
    function posix_memalign(memptr, alignment, size) result(error) bind(C, name="posix_memalign")
      import c_int, c_size_t, c_ptr
      integer(kind=c_int) :: error
      type(c_ptr), intent(inout) :: memptr
      integer(kind=c_size_t), intent(in), value :: alignment, size
    end function
  end interface

  interface
    subroutine free(ptr) bind(C, name="free")
      import c_ptr
      type(c_ptr), value :: ptr
    end subroutine
  end interface

end module
