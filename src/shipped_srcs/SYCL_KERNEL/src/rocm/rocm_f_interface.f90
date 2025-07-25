module rocm_f_interface

  implicit none

  integer, parameter :: hipMemcpyHostToDevice = 0
  integer, parameter :: hipMemcpyDeviceToHost = 1
  integer, parameter :: hipMemcpyDeviceToDevice = 2

  interface
    function hip_set_device(i_gpu) result(ierr) bind(c, name="hipSetDeviceFromC")
      use iso_c_binding, only: c_int
      implicit none
      integer(c_int), value :: i_gpu
      integer(c_int) :: ierr
    end function
  end interface

  interface
    function hip_get_device_count(n_gpu) result(ierr) bind(c, name="hipGetDeviceCountFromC")
      use iso_c_binding, only: c_int
      implicit none
      integer(c_int) :: n_gpu
      integer(c_int) :: ierr
    end function
  end interface

  interface
    function hip_device_synchronize()result(ierr) bind(c, name="hipDeviceSynchronizeFromC")
      use iso_c_binding, only: c_int
      implicit none
      integer(c_int) :: ierr
    end function
  end interface

  interface
    function hip_memcpy(dst, src, size, dir) result(ierr) bind(c, name="hipMemcpyFromC")
      use iso_c_binding, only: c_intptr_t,c_int
      implicit none
      integer(c_intptr_t), value :: dst
      integer(c_intptr_t), value :: src
      integer(c_intptr_t), value :: size
      integer(c_int), value :: dir
      integer(c_int) :: ierr
    end function
  end interface

  interface
    function hip_free(a) result(ierr) bind(c, name="hipFreeFromC")
      use iso_c_binding, only: c_intptr_t,c_int
      implicit none
      integer(c_intptr_t), value :: a
      integer(c_int) :: ierr
    end function
  end interface

  interface
    function hip_malloc(a,size) result(ierr) bind(c, name="hipMallocFromC")
      use iso_c_binding, only: c_intptr_t,c_int
      implicit none
      integer(c_intptr_t) :: a
      integer(c_intptr_t), value :: size
      integer(c_int) :: ierr
    end function
  end interface

  interface
    subroutine compute_hh_hip_gpu_kernel(q,hh,hh_tau,nev,nb,ldq,ncols) bind(c, name="compute_hh_hip_gpu_real_kernel")
      use iso_c_binding, only: c_intptr_t,c_int
      implicit none
      integer(c_int), value :: nev ! (N_C)
      integer(c_int), value :: nb ! (b==nbw)
      integer(c_int), value :: ldq ! (leading dimension of q)
      integer(c_int), value :: ncols ! (n)
      integer(c_intptr_t), value :: q ! (X)
      integer(c_intptr_t), value :: hh_tau ! (tau)
      integer(c_intptr_t), value :: hh ! (v)
    end subroutine
  end interface

  interface
    subroutine compute_hh_hip_gpu_complex_kernel(q,hh,hh_tau,nev,nb,ldq,ncols) bind(c)
      use iso_c_binding, only: c_intptr_t,c_int
      implicit none
      integer(c_int), value :: nev ! (N_C)
      integer(c_int), value :: nb ! (b==nbw)
      integer(c_int), value :: ldq ! (leading dimension of q)
      integer(c_int), value :: ncols ! (n)
      integer(c_intptr_t), value :: q ! (X)
      integer(c_intptr_t), value :: hh_tau ! (tau)
      integer(c_intptr_t), value :: hh ! (v)
    end subroutine
  end interface

  contains

    subroutine gpu_init()

      implicit none

      integer :: ok
      integer :: n_gpu

      ok = hip_get_device_count(n_gpu)

      if(ok == 0) then
        write(*,"(2X,A)") "Error: hip_get_device_count"

        stop
      end if

      if(n_gpu > 0) then
        ok = hip_set_device(0)

        if(ok == 0) then
          write(*,"(2X,A)") "Error: hip_set_device"

          stop
        end if
      else
        write(*,"(2X,A)") "Error: no GPU"

        stop
      end if

    end subroutine

    subroutine check_hip(ok,msg)

      implicit none

      integer, intent(in) :: ok
      character(*), intent(in) :: msg

      if(ok /= 0) then
        write(*,"(2X,2A)") "hip error: ",msg

        stop
      end if

    end subroutine

end module
