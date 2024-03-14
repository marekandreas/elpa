!    Copyright 2014 - 2023, A. Marek
!
!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), formerly known as
!      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
!    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!      Informatik,
!    - Technische Universität München, Lehrstuhl für Informatik mit
!      Schwerpunkt Wissenschaftliches Rechnen ,
!    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!      and
!    - IBM Deutschland GmbH
!
!
!    More information can be found here:
!    http://elpa.mpcdf.mpg.de/
!
!    ELPA is free software: you can redistribute it and/or modify
!    it under the terms of the version 3 of the license of the
!    GNU Lesser General Public License as published by the Free
!    Software Foundation.
!
!    ELPA is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public License
!    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
!
!    ELPA reflects a substantial effort on the part of the original
!    ELPA consortium, and we ask you to respect the spirit of the
!    license that we chose: i.e., please contribute any changes you
!    may have back to the original ELPA library distribution, and keep
!    any derivatives of ELPA under the same license that we chose for
!    the original distribution, the GNU Lesser General Public License.
!
! Author: Andreas Marek, MPCDF
! This file is the generated version. Do NOT edit


  integer(kind=ik) :: openmpOffloadMemcpyHostToDevice
  integer(kind=ik) :: openmpOffloadMemcpyDeviceToHost
  integer(kind=ik) :: openmpOffloadMemcpyDeviceToDevice
  integer(kind=ik) :: openmpOffloadHostRegisterDefault
  integer(kind=ik) :: openmpOffloadHostRegisterPortable
  integer(kind=ik) :: openmpOffloadHostRegisterMapped

  integer(kind=ik) :: mkl_openmp_offloadPointerModeDevice
  integer(kind=ik) :: mkl_openmp_offloadPointerModeHost


!  interface
!    function openmp_offload_device_get_attributes_c(value, attribute) result(istat) &
!             bind(C, name="openmpOffloadDeviceGetAttributeFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!
!      integer(kind=C_INT), value  :: attribute
!      integer(kind=C_INT)         :: value
!      integer(kind=C_INT)         :: istat
!    end function
!  end interface


!  interface
!    function mkl_openmp_offload_get_version_c(mkl_openmp_offloadHandle, version) result(istat) &
!             bind(C, name="mkl_openmp_offloadGetVersionFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!
!      integer(kind=C_intptr_T), value  :: mkl_openmp_offloadHandle
!      integer(kind=C_INT)              :: version
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface


!  interface
!    function openmp_offload_get_last_error_c() result(istat) &
!             bind(C, name="openmpOffloadGetLastErrorFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int)              :: istat
!    end function
!  end interface

!  ! streams
!
!  interface
!    function openmp_offload_stream_create_c(openmp_offloadStream) result(istat) &
!             bind(C, name="openmpOffloadStreamCreateFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T) :: openmp_offloadStream
!      integer(kind=C_INT)      :: istat
!    end function
!  end interface

!  interface
!    function openmp_offload_stream_destroy_c(openmp_offloadStream) result(istat) &
!             bind(C, name="openmp_offloadStreamDestroyFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value :: openmp_offloadStream
!      integer(kind=C_INT)             :: istat
!    end function
!  end interface

!  interface
!    function openmp_offload_stream_synchronize_explicit_c(openmp_offloadStream) result(istat) &
!             bind(C, name="openmpOffloadStreamSynchronizeExplicitFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!
!      integer(kind=C_intptr_T), value  :: openmp_offloadStream
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface

!  interface
!    function openmp_offload_stream_synchronize_implicit_c() result(istat) &
!             bind(C, name="openmpOffloadStreamSynchronizeImplicitFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface

!  interface
!    function mkl_openmp_offload_set_stream_c(openmp_offloadHandle, openmp_offloadStream) result(istat) &
!             bind(C, name="mklOpenmpOffloadSetStreamFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!
!      integer(kind=C_intptr_T), value  :: openmp_offloadHandle
!      integer(kind=C_intptr_T), value  :: openmp_offloadStream
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface

  interface
    function mkl_openmp_offload_create_c(openmp_offloadHandle) result(istat) &
             bind(C, name="mklOpenmpOffloadCreateFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: openmp_offloadHandle
      integer(kind=C_INT)      :: istat
    end function
  end interface

  interface
    function mkl_openmp_offload_destroy_c(openmp_offloadHandle) result(istat) &
             bind(C, name="mklOpenmpOffloadDestroyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: openmp_offloadHandle
      integer(kind=C_INT)      :: istat
    end function
  end interface

  ! functions to set and query the GPU devices
  interface
    function openmp_offload_setdevice_c(n) result(istat) &
             bind(C, name="openmpOffloadSetDeviceFromC")

      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT), value    :: n
      integer(kind=C_INT)           :: istat
    end function
  end interface

  interface
    function openmp_offload_getdevicecount_c(n) result(istat) &
             bind(C, name="openmpOffloadGetDeviceCountFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT), intent(out)         :: n
      integer(kind=C_INT)                      :: istat
    end function
  end interface

!  interface
!    function openmp_offload_devicesynchronize_c()result(istat) &
!             bind(C,name="openmpOffloadDeviceSynchronizeFromC")
!
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_INT)                       :: istat
!    end function
!  end interface

  ! functions to copy GPU memory
  interface
    function openmp_offload_memcpyDeviceToDevice_c() result(flag) &
             bind(C, name="openmpOffloadMemcpyDeviceToDeviceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function openmp_offload_memcpyHostToDevice_c() result(flag) &
             bind(C, name="openmpOffloadMemcpyHostToDeviceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function openmp_offload_memcpyDeviceToHost_c() result(flag) &
             bind(C, name="openmpOffloadMemcpyDeviceToHostFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

!  interface
!    function openmp_offload_hostRegisterDefault_c() result(flag) &
!             bind(C, name="openmpOffloadHostRegisterDefaultFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface

!  interface
!    function openmp_offload_hostRegisterPortable_c() result(flag) &
!             bind(C, name="openmpOffloadHostRegisterPortableFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface

!  interface
!    function openmp_offload_hostRegisterMapped_c() result(flag) &
!             bind(C, name="openmpOffloadHostRegisterMappedFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface

  interface
    function openmp_offload_memcpy_intptr_c(dst, src, size, dir) result(istat) &
             bind(C, name="openmpOffloadMemcpyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t), value              :: dst
      integer(kind=C_intptr_t), value              :: src
      integer(kind=c_intptr_t), intent(in), value    :: size
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function openmp_offload_memcpy_cptr_c(dst, src, size, dir) result(istat) &
             bind(C, name="openmpOffloadMemcpyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                           :: dst
      type(c_ptr), value                           :: src
      integer(kind=c_intptr_t), intent(in), value  :: size
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function openmp_offload_memcpy_mixed_to_device_c(dst, src, size, dir) result(istat) &
             bind(C, name="openmpOffloadMemcpyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                           :: dst
      integer(kind=c_intptr_t), value              :: src
      integer(kind=c_intptr_t), intent(in), value  :: size
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function openmp_offload_memcpy_mixed_to_host_c(dst, src, size, dir) result(istat) &
             bind(C, name="openmpOffloadMemcpyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                           :: src
      integer(kind=c_intptr_t), value              :: dst
      integer(kind=c_intptr_t), intent(in), value  :: size
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=C_INT)                          :: istat
    end function
  end interface

!  interface
!    function openmp_offload_memcpy_async_intptr_c(dst, src, size, dir, openmp_offloadStream) result(istat) &
!             bind(C, name="openmpOffloadMemcpyAsyncFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t), value              :: dst
!      integer(kind=C_intptr_t), value              :: src
!      integer(kind=c_intptr_t), intent(in), value  :: size
!      integer(kind=C_INT), intent(in), value       :: dir
!      integer(kind=c_intptr_t), value              :: openmp_offloadStream
!      integer(kind=C_INT)                          :: istat
!    end function
!  end interface

!  interface
!    function openmp_offload_memcpy_async_cptr_c(dst, src, size, dir, openmp_offloadStream) result(istat) &
!             bind(C, name="openmpOffloadMemcpyAsyncFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr), value                           :: dst
!      type(c_ptr), value                           :: src
!      integer(kind=c_intptr_t), intent(in), value  :: size
!      integer(kind=C_INT), intent(in), value       :: dir
!      integer(kind=c_intptr_t), value              :: openmp_offloadStream
!      integer(kind=C_INT)                          :: istat
!    end function
!  end interface

!  interface
!    function openmp_offload_memcpy_async_mixed_to_device_c(dst, src, size, dir, openmp_offloadStream) result(istat) &
!             bind(C, name="openmpOffloadMemcpyAsyncFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr), value                           :: dst
!      integer(kind=C_intptr_t), value              :: src
!      integer(kind=c_intptr_t), intent(in), value  :: size
!      integer(kind=C_INT), intent(in), value       :: dir
!      integer(kind=c_intptr_t), value              :: openmp_offloadStream
!      integer(kind=C_INT)                          :: istat
!    end function
!  end interface

!  interface
!    function openmp_offload_memcpy_async_mixed_to_host_c(dst, src, size, dir, openmp_offloadStream) result(istat) &
!             bind(C, name="openmpOffloadMemcpyAsyncFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr), value                           :: src
!      integer(kind=C_intptr_t), value              :: dst
!      integer(kind=c_intptr_t), intent(in), value  :: size
!      integer(kind=C_INT), intent(in), value       :: dir
!      integer(kind=c_intptr_t), value              :: openmp_offloadStream
!      integer(kind=C_INT)                          :: istat
!    end function
!  end interface

!  interface
!    function openmp_offload_memcpy2d_intptr_c(dst, dpitch, src, spitch, width, height , dir) result(istat) &
!             bind(C, name="openmpOffloadMemcpy2dFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value                :: dst
!      integer(kind=c_intptr_t), intent(in), value    :: dpitch
!      integer(kind=C_intptr_T), value                :: src
!      integer(kind=c_intptr_t), intent(in), value    :: spitch
!      integer(kind=c_intptr_t), intent(in), value    :: width
!      integer(kind=c_intptr_t), intent(in), value    :: height
!      integer(kind=C_INT), intent(in), value         :: dir
!      integer(kind=C_INT)                            :: istat
!    end function
!  end interface

!  interface
!    function openmp_offload_memcpy2d_cptr_c(dst, dpitch, src, spitch, width, height , dir) result(istat) &
!             bind(C, name="openmpOffloadMemcpy2dFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr), value                :: dst
!      integer(kind=c_intptr_t), intent(in), value    :: dpitch
!      type(c_ptr), value                :: src
!      integer(kind=c_intptr_t), intent(in), value    :: spitch
!      integer(kind=c_intptr_t), intent(in), value    :: width
!      integer(kind=c_intptr_t), intent(in), value    :: height
!      integer(kind=C_INT), intent(in), value         :: dir
!      integer(kind=C_INT)                            :: istat
!    end function
!  end interface

!  interface
!    function openmp_offload_memcpy2d_async_intptr_c(dst, dpitch, src, spitch, width, height, dir, openmp_offloadStream) result(istat) &
!             bind(C, name="openmpOffloadMemcpy2dAsyncFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value                :: dst
!      integer(kind=c_intptr_t), intent(in), value    :: dpitch
!      integer(kind=C_intptr_T), value                :: src
!      integer(kind=c_intptr_t), intent(in), value    :: spitch
!      integer(kind=c_intptr_t), intent(in), value    :: width
!      integer(kind=c_intptr_t), intent(in), value    :: height
!      integer(kind=C_INT), intent(in), value         :: dir
!      integer(kind=c_intptr_t), value                :: openmp_offloadStream
!      integer(kind=C_INT)                            :: istat
!    end function
!  end interface

!  interface
!    function openmp_offload_memcpy2d_async_cptr_c(dst, dpitch, src, spitch, width, height, dir, openmp_offloadStream) result(istat) &
!             bind(C, name="openmpOffloadMemcpy2dAsyncFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr), value                :: dst
!      integer(kind=c_intptr_t), intent(in), value    :: dpitch
!      type(c_ptr), value                :: src
!      integer(kind=c_intptr_t), intent(in), value    :: spitch
!      integer(kind=c_intptr_t), intent(in), value    :: width
!      integer(kind=c_intptr_t), intent(in), value    :: height
!      integer(kind=C_INT), intent(in), value         :: dir
!      integer(kind=c_intptr_t), value                :: openmp_offloadStream
!      integer(kind=C_INT)                            :: istat
!    end function
!  end interface

!  interface
!    function openmp_offload_host_register_c(a, size, flag) result(istat) &
!             bind(C, name="openmpOffloadHostRegisterFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t), value              :: a
!      integer(kind=c_intptr_t), intent(in), value  :: size
!      integer(kind=C_INT), intent(in), value       :: flag
!      integer(kind=C_INT)                          :: istat
!    end function
!  end interface

!  interface
!    function openmp_offload_host_unregister_c(a) result(istat) &
!             bind(C, name="openmpOffloadHostUnregisterFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t), value              :: a
!      integer(kind=C_INT)                          :: istat
!    end function
!  end interface

  interface openmp_offload_free
    module procedure openmp_offload_free_intptr
    module procedure openmp_offload_free_cptr
  end interface

  interface
    function openmp_offload_free_intptr_c(a) result(istat) &
             bind(C, name="openmpOffloadFreeFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T)  :: a
      integer(kind=C_INT)              :: istat
    end function
  end interface

  interface
    function openmp_offload_free_cptr_c(a) result(istat) &
             bind(C, name="openmpOffloadFreeFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)  :: a
      integer(kind=C_INT)              :: istat
    end function
  end interface

  interface openmp_offload_memcpy
    module procedure openmp_offload_memcpy_intptr
    module procedure openmp_offload_memcpy_cptr
    module procedure openmp_offload_memcpy_mixed_to_device
    module procedure openmp_offload_memcpy_mixed_to_host
  end interface

!  interface openmp_offload_memcpy_async
!    module procedure openmp_offload_memcpy_async_intptr
!    module procedure openmp_offload_memcpy_async_cptr
!    module procedure openmp_offload_memcpy_async_mixed_to_device
!    module procedure openmp_offload_memcpy_async_mixed_to_host
!  end interface

  interface openmp_offload_malloc
    module procedure openmp_offload_malloc_intptr
    module procedure openmp_offload_malloc_cptr
  end interface

  interface
    function openmp_offload_malloc_intptr_c(a, width_height) result(istat) &
             bind(C, name="openmpOffloadMallocFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      ! no value since **pointer
      integer(kind=C_intptr_T)                    :: a
      integer(kind=c_intptr_t), intent(in), value :: width_height
      integer(kind=C_INT)                         :: istat
    end function
  end interface

  interface
    function openmp_offload_malloc_cptr_c(a, width_height) result(istat) &
             bind(C, name="openmpOffloadMallocFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      ! no value since **pointer
      type(c_ptr)                                 :: a
      integer(kind=c_intptr_t), intent(in), value :: width_height
      integer(kind=C_INT)                         :: istat
    end function
  end interface

!  interface openmp_offload_free_host
!    module procedure openmp_offload_free_host_intptr
!    module procedure openmp_offload_free_host_cptr
!  end interface
!  interface
!    function openmp_offload_free_host_intptr_c(a) result(istat) &
!             bind(C, name="openmpOffloadFreeHostFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t), value  :: a
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface

!  interface
!    function openmp_offload_free_host_cptr_c(a) result(istat) &
!             bind(C, name="openmpOffloadFreeHostFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr), value               :: a
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface

!  interface openmp_offload_malloc_host
!    module procedure openmp_offload_malloc_host_intptr
!    module procedure openmp_offload_malloc_host_cptr
!  end interface
!  interface
!    function openmp_offload_malloc_host_intptr_c(a, width_height) result(istat) &
!             bind(C, name="openmpOffloadMallocHostFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t)                    :: a
!      integer(kind=c_intptr_t), intent(in), value :: width_height
!      integer(kind=C_INT)                         :: istat
!    end function
!  end interface

!  interface
!    function openmp_offload_malloc_host_cptr_c(a, width_height) result(istat) &
!             bind(C, name="openmpOffloadMallocHostFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)                    :: a
!      integer(kind=c_intptr_t), intent(in), value :: width_height
!      integer(kind=C_INT)                         :: istat
!    end function
!  end interface

  interface
    function openmp_offload_memset_c(a, val, size) result(istat) &
             bind(C, name="openmpOffloadMemsetFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value            :: a
      integer(kind=C_INT), value                 :: val
      integer(kind=c_intptr_t), intent(in), value  :: size
      integer(kind=C_INT)                        :: istat
    end function
  end interface

!  interface
!    function openmp_offload_memset_async_c(a, val, size, openmp_offloadStream) result(istat) &
!             bind(C, name="openmpOffloadMemsetAsyncFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value            :: a
!      integer(kind=C_INT), value                 :: val
!      integer(kind=c_intptr_t), intent(in), value  :: size
!      integer(kind=C_INT)                        :: istat
!      integer(kind=c_intptr_t), value            :: openmp_offloadStream
!    end function
!  end interface

  interface mkl_openmp_offload_Dgemm
    module procedure mkl_openmp_offload_Dgemm_intptr
    module procedure mkl_openmp_offload_Dgemm_cptr
    module procedure mkl_openmp_offload_Dgemm_intptr_cptr_intptr
  end interface

  interface
    subroutine mkl_openmp_offload_Dgemm_intptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklOpenmpOffloadDgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_DOUBLE) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Dgemm_cptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklOpenmpOffloadDgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_DOUBLE) ,value               :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Dgemm_intptr_cptr_intptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, &
                                                                      alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklOpenmpOffloadDgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m, n, k
      integer(kind=C_INT), intent(in), value  :: lda, ldb, ldc
      real(kind=C_DOUBLE) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, c
      type(c_ptr), value                      :: b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface


  interface mkl_openmp_offload_Dcopy
    module procedure mkl_openmp_offload_Dcopy_intptr
    module procedure mkl_openmp_offload_Dcopy_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_Dcopy_intptr_c(mkl_openmp_offloadHandle, n, x, incx, y, incy) &
                              bind(C,name="mklOpenmpOffloadDcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Dcopy_cptr_c(mkl_openmp_offloadHandle, n, x, incx, y, incy) &
                              bind(C,name="mklOpenmpOffloadDcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface


  interface mkl_openmp_offload_Dtrmm
    module procedure mkl_openmp_offload_Dtrmm_intptr
    module procedure mkl_openmp_offload_Dtrmm_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_Dtrmm_intptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklOpenmpOffloadDtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Dtrmm_cptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklOpenmpOffloadDtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface


  interface mkl_openmp_offload_Dtrsm
    module procedure mkl_openmp_offload_Dtrsm_intptr
    module procedure mkl_openmp_offload_Dtrsm_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_Dtrsm_intptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklOpenmpOffloadDtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Dtrsm_cptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklOpenmpOffloadDtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Dgemv_c(mkl_openmp_offloadHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name="mklOpenmpOffloadDgemv_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      real(kind=C_DOUBLE) , value              :: alpha, beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface mkl_openmp_offload_Sgemm
    module procedure mkl_openmp_offload_Sgemm_intptr
    module procedure mkl_openmp_offload_Sgemm_cptr
    module procedure mkl_openmp_offload_Sgemm_intptr_cptr_intptr
  end interface

  interface
    subroutine mkl_openmp_offload_Sgemm_intptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklOpenmpOffloadSgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_FLOAT) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Sgemm_cptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklOpenmpOffloadSgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_FLOAT) ,value               :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Sgemm_intptr_cptr_intptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, &
                                                                      alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklOpenmpOffloadSgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m, n, k
      integer(kind=C_INT), intent(in), value  :: lda, ldb, ldc
      real(kind=C_FLOAT) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, c
      type(c_ptr), value                      :: b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface


  interface mkl_openmp_offload_Scopy
    module procedure mkl_openmp_offload_Scopy_intptr
    module procedure mkl_openmp_offload_Scopy_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_Scopy_intptr_c(mkl_openmp_offloadHandle, n, x, incx, y, incy) &
                              bind(C,name="mklOpenmpOffloadScopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Scopy_cptr_c(mkl_openmp_offloadHandle, n, x, incx, y, incy) &
                              bind(C,name="mklOpenmpOffloadScopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface


  interface mkl_openmp_offload_Strmm
    module procedure mkl_openmp_offload_Strmm_intptr
    module procedure mkl_openmp_offload_Strmm_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_Strmm_intptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklOpenmpOffloadStrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Strmm_cptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklOpenmpOffloadStrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface


  interface mkl_openmp_offload_Strsm
    module procedure mkl_openmp_offload_Strsm_intptr
    module procedure mkl_openmp_offload_Strsm_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_Strsm_intptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklOpenmpOffloadStrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Strsm_cptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklOpenmpOffloadStrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Sgemv_c(mkl_openmp_offloadHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name="mklOpenmpOffloadSgemv_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      real(kind=C_FLOAT) , value              :: alpha, beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface mkl_openmp_offload_Zgemm
    module procedure mkl_openmp_offload_Zgemm_intptr
    module procedure mkl_openmp_offload_Zgemm_cptr
    module procedure mkl_openmp_offload_Zgemm_intptr_cptr_intptr
  end interface

  interface
    subroutine mkl_openmp_offload_Zgemm_intptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklOpenmpOffloadZgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Zgemm_cptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklOpenmpOffloadZgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Zgemm_intptr_cptr_intptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, &
                                                                      alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklOpenmpOffloadZgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m, n, k
      integer(kind=C_INT), intent(in), value  :: lda, ldb, ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, c
      type(c_ptr), value                      :: b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface


  interface mkl_openmp_offload_Zcopy
    module procedure mkl_openmp_offload_Zcopy_intptr
    module procedure mkl_openmp_offload_Zcopy_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_Zcopy_intptr_c(mkl_openmp_offloadHandle, n, x, incx, y, incy) &
                              bind(C,name="mklOpenmpOffloadZcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Zcopy_cptr_c(mkl_openmp_offloadHandle, n, x, incx, y, incy) &
                              bind(C,name="mklOpenmpOffloadZcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface


  interface mkl_openmp_offload_Ztrmm
    module procedure mkl_openmp_offload_Ztrmm_intptr
    module procedure mkl_openmp_offload_Ztrmm_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_Ztrmm_intptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklOpenmpOffloadZtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Ztrmm_cptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklOpenmpOffloadZtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface


  interface mkl_openmp_offload_Ztrsm
    module procedure mkl_openmp_offload_Ztrsm_intptr
    module procedure mkl_openmp_offload_Ztrsm_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_Ztrsm_intptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklOpenmpOffloadZtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Ztrsm_cptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklOpenmpOffloadZtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Zgemv_c(mkl_openmp_offloadHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name="mklOpenmpOffloadZgemv_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha, beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface mkl_openmp_offload_Cgemm
    module procedure mkl_openmp_offload_Cgemm_intptr
    module procedure mkl_openmp_offload_Cgemm_cptr
    module procedure mkl_openmp_offload_Cgemm_intptr_cptr_intptr
  end interface

  interface
    subroutine mkl_openmp_offload_Cgemm_intptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklOpenmpOffloadCgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Cgemm_cptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklOpenmpOffloadCgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Cgemm_intptr_cptr_intptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, &
                                                                      alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklOpenmpOffloadCgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m, n, k
      integer(kind=C_INT), intent(in), value  :: lda, ldb, ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, c
      type(c_ptr), value                      :: b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface


  interface mkl_openmp_offload_Ccopy
    module procedure mkl_openmp_offload_Ccopy_intptr
    module procedure mkl_openmp_offload_Ccopy_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_Ccopy_intptr_c(mkl_openmp_offloadHandle, n, x, incx, y, incy) &
                              bind(C,name="mklOpenmpOffloadCcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Ccopy_cptr_c(mkl_openmp_offloadHandle, n, x, incx, y, incy) &
                              bind(C,name="mklOpenmpOffloadCcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface


  interface mkl_openmp_offload_Ctrmm
    module procedure mkl_openmp_offload_Ctrmm_intptr
    module procedure mkl_openmp_offload_Ctrmm_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_Ctrmm_intptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklOpenmpOffloadCtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Ctrmm_cptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklOpenmpOffloadCtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface


  interface mkl_openmp_offload_Ctrsm
    module procedure mkl_openmp_offload_Ctrsm_intptr
    module procedure mkl_openmp_offload_Ctrsm_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_Ctrsm_intptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklOpenmpOffloadCtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Ctrsm_cptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklOpenmpOffloadCtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_Cgemv_c(mkl_openmp_offloadHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name="mklOpenmpOffloadCgemv_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha, beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
    end subroutine
  end interface

!#ifdef WITH_NVTX
!  ! NVTX profiling interfaces
!  interface nvtxRangePushA
!    subroutine nvtxRangePushA(name) bind(C, name='nvtxRangePushA')
!      use, intrinsic :: iso_c_binding
!      character(kind=C_CHAR,len=1) :: name(*)
!    end subroutine
!  end interface

!  interface nvtxRangePop
!    subroutine nvtxRangePop() bind(C, name='nvtxRangePop')
!    end subroutine
!  end interface
!#endif

!  interface
!    function mkl_openmp_offload_pointerModeDevice_c() result(flag) &
!               bind(C, name="mkl_openmp_offloadPointerModeDeviceFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface

!  interface
!    function mkl_openmp_offload_pointerModeHost_c() result(flag) &
!               bind(C, name="mkl_openmp_offloadPointerModeHostFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface

!  interface
!    subroutine mkl_openmp_offload_getPointerMode_c(mkl_openmp_offloadHandle, mode) &
!               bind(C, name="mkl_openmp_offloadGetPointerModeFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value   :: mkl_openmp_offloadHandle
!      integer(kind=c_int)               :: mode
!    end subroutine
!  end interface

!  interface
!    subroutine mkl_openmp_offload_setPointerMode_c(mkl_openmp_offloadHandle, mode) &
!               bind(C, name="mkl_openmp_offloadSetPointerModeFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T),value    :: mkl_openmp_offloadHandle
!      integer(kind=c_int), value        :: mode
!    end subroutine
!  end interface


  interface mkl_openmp_offload_Ddot
    module procedure mkl_openmp_offload_Ddot_intptr
    module procedure mkl_openmp_offload_Ddot_cptr
  end interface

!  interface
!    subroutine mkl_openmp_offload_Ddot_intptr_c(mkl_openmp_offloadHandle, length, x, incx, y, incy, result) &
!               bind(C, name="mkl_openmp_offloadDdot_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT), value              :: length, incx, incy
!      integer(kind=C_intptr_T), value         :: x, y, result
!    end subroutine
!  end interface

!  interface
!    subroutine mkl_openmp_offload_Ddot_cptr_c(mkl_openmp_offloadHandle, length, x, incx, y, incy, result) &
!               bind(C, name="mkl_openmp_offloadDdot_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT), value              :: length, incx, incy
!      type(c_ptr), value                      :: x, y, result
!    end subroutine
!  end interface


  interface mkl_openmp_offload_Dscal
    module procedure mkl_openmp_offload_Dscal_intptr
    module procedure mkl_openmp_offload_Dscal_cptr
  end interface

!  interface
!    subroutine mkl_openmp_offload_Dscal_intptr_c(mkl_openmp_offloadHandle, length, alpha, x, incx) &
!               bind(C, name="mkl_openmp_offloadDscal_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT),value               :: length, incx
!      real(kind=C_DOUBLE) ,value                :: alpha
!      integer(kind=C_intptr_T), value         :: x
!    end subroutine
!  end interface

!  interface
!    subroutine mkl_openmp_offload_Dscal_cptr_c(mkl_openmp_offloadHandle, length, alpha, x, incx) &
!               bind(C, name="mkl_openmp_offloadDscal_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT),value               :: length, incx
!      real(kind=C_DOUBLE) ,value                :: alpha
!      type(c_ptr), value                      :: x
!    end subroutine
!  end interface


  interface mkl_openmp_offload_Daxpy
    module procedure mkl_openmp_offload_Daxpy_intptr
    module procedure mkl_openmp_offload_Daxpy_cptr
  end interface

!  interface
!    subroutine mkl_openmp_offload_Daxpy_intptr_c(mkl_openmp_offloadHandle, length, alpha, x, incx, y, incy) &
!               bind(C, name="mkl_openmp_offloadDaxpy_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT),value               :: length, incx, incy
!      real(kind=C_DOUBLE) ,value                :: alpha
!      integer(kind=C_intptr_T), value         :: x, y
!    end subroutine
!  end interface

!  interface
!    subroutine mkl_openmp_offload_Daxpy_cptr_c(mkl_openmp_offloadHandle, length, alpha, x, incx, y, incy) &
!               bind(C, name="mkl_openmp_offloadDaxpy_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT),value               :: length, incx, incy
!      real(kind=C_DOUBLE) ,value                :: alpha
!      type(c_ptr), value                      :: x, y
!    end subroutine
!  end interface

  interface mkl_openmp_offload_Sdot
    module procedure mkl_openmp_offload_Sdot_intptr
    module procedure mkl_openmp_offload_Sdot_cptr
  end interface

!  interface
!    subroutine mkl_openmp_offload_Sdot_intptr_c(mkl_openmp_offloadHandle, length, x, incx, y, incy, result) &
!               bind(C, name="mkl_openmp_offloadSdot_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT), value              :: length, incx, incy
!      integer(kind=C_intptr_T), value         :: x, y, result
!    end subroutine
!  end interface

!  interface
!    subroutine mkl_openmp_offload_Sdot_cptr_c(mkl_openmp_offloadHandle, length, x, incx, y, incy, result) &
!               bind(C, name="mkl_openmp_offloadSdot_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT), value              :: length, incx, incy
!      type(c_ptr), value                      :: x, y, result
!    end subroutine
!  end interface


  interface mkl_openmp_offload_Sscal
    module procedure mkl_openmp_offload_Sscal_intptr
    module procedure mkl_openmp_offload_Sscal_cptr
  end interface

!  interface
!    subroutine mkl_openmp_offload_Sscal_intptr_c(mkl_openmp_offloadHandle, length, alpha, x, incx) &
!               bind(C, name="mkl_openmp_offloadSscal_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT),value               :: length, incx
!      real(kind=C_FLOAT) ,value                :: alpha
!      integer(kind=C_intptr_T), value         :: x
!    end subroutine
!  end interface

!  interface
!    subroutine mkl_openmp_offload_Sscal_cptr_c(mkl_openmp_offloadHandle, length, alpha, x, incx) &
!               bind(C, name="mkl_openmp_offloadSscal_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT),value               :: length, incx
!      real(kind=C_FLOAT) ,value                :: alpha
!      type(c_ptr), value                      :: x
!    end subroutine
!  end interface


  interface mkl_openmp_offload_Saxpy
    module procedure mkl_openmp_offload_Saxpy_intptr
    module procedure mkl_openmp_offload_Saxpy_cptr
  end interface

!  interface
!    subroutine mkl_openmp_offload_Saxpy_intptr_c(mkl_openmp_offloadHandle, length, alpha, x, incx, y, incy) &
!               bind(C, name="mkl_openmp_offloadSaxpy_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT),value               :: length, incx, incy
!      real(kind=C_FLOAT) ,value                :: alpha
!      integer(kind=C_intptr_T), value         :: x, y
!    end subroutine
!  end interface

!  interface
!    subroutine mkl_openmp_offload_Saxpy_cptr_c(mkl_openmp_offloadHandle, length, alpha, x, incx, y, incy) &
!               bind(C, name="mkl_openmp_offloadSaxpy_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT),value               :: length, incx, incy
!      real(kind=C_FLOAT) ,value                :: alpha
!      type(c_ptr), value                      :: x, y
!    end subroutine
!  end interface

  interface mkl_openmp_offload_Zdot
    module procedure mkl_openmp_offload_Zdot_intptr
    module procedure mkl_openmp_offload_Zdot_cptr
  end interface

!  interface
!    subroutine mkl_openmp_offload_Zdot_intptr_c(conj, mkl_openmp_offloadHandle, length, x, incx, y, incy, result) &
!               bind(C, name="mkl_openmp_offloadZdot_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      character(1,C_CHAR),value               :: conj
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT), value              :: length, incx, incy
!      integer(kind=C_intptr_T), value         :: x, y, result
!    end subroutine
!  end interface

!  interface
!    subroutine mkl_openmp_offload_Zdot_cptr_c(conj, mkl_openmp_offloadHandle, length, x, incx, y, incy, result) &
!               bind(C, name="mkl_openmp_offloadZdot_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      character(1,C_CHAR),value               :: conj
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT), value              :: length, incx, incy
!      type(c_ptr), value                      :: x, y, result
!    end subroutine
!  end interface


  interface mkl_openmp_offload_Zscal
    module procedure mkl_openmp_offload_Zscal_intptr
    module procedure mkl_openmp_offload_Zscal_cptr
  end interface

!  interface
!    subroutine mkl_openmp_offload_Zscal_intptr_c(mkl_openmp_offloadHandle, length, alpha, x, incx) &
!               bind(C, name="mkl_openmp_offloadZscal_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT),value               :: length, incx
!      complex(kind=C_DOUBLE_COMPLEX) ,value                :: alpha
!      integer(kind=C_intptr_T), value         :: x
!    end subroutine
!  end interface

!  interface
!    subroutine mkl_openmp_offload_Zscal_cptr_c(mkl_openmp_offloadHandle, length, alpha, x, incx) &
!               bind(C, name="mkl_openmp_offloadZscal_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT),value               :: length, incx
!      complex(kind=C_DOUBLE_COMPLEX) ,value                :: alpha
!      type(c_ptr), value                      :: x
!    end subroutine
!  end interface


  interface mkl_openmp_offload_Zaxpy
    module procedure mkl_openmp_offload_Zaxpy_intptr
    module procedure mkl_openmp_offload_Zaxpy_cptr
  end interface

!  interface
!    subroutine mkl_openmp_offload_Zaxpy_intptr_c(mkl_openmp_offloadHandle, length, alpha, x, incx, y, incy) &
!               bind(C, name="mkl_openmp_offloadZaxpy_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT),value               :: length, incx, incy
!      complex(kind=C_DOUBLE_COMPLEX) ,value                :: alpha
!      integer(kind=C_intptr_T), value         :: x, y
!    end subroutine
!  end interface

!  interface
!    subroutine mkl_openmp_offload_Zaxpy_cptr_c(mkl_openmp_offloadHandle, length, alpha, x, incx, y, incy) &
!               bind(C, name="mkl_openmp_offloadZaxpy_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT),value               :: length, incx, incy
!      complex(kind=C_DOUBLE_COMPLEX) ,value                :: alpha
!      type(c_ptr), value                      :: x, y
!    end subroutine
!  end interface

  interface mkl_openmp_offload_Cdot
    module procedure mkl_openmp_offload_Cdot_intptr
    module procedure mkl_openmp_offload_Cdot_cptr
  end interface

!  interface
!    subroutine mkl_openmp_offload_Cdot_intptr_c(conj, mkl_openmp_offloadHandle, length, x, incx, y, incy, result) &
!               bind(C, name="mkl_openmp_offloadCdot_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      character(1,C_CHAR),value               :: conj
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT), value              :: length, incx, incy
!      integer(kind=C_intptr_T), value         :: x, y, result
!    end subroutine
!  end interface

!  interface
!    subroutine mkl_openmp_offload_Cdot_cptr_c(conj, mkl_openmp_offloadHandle, length, x, incx, y, incy, result) &
!               bind(C, name="mkl_openmp_offloadCdot_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      character(1,C_CHAR),value               :: conj
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT), value              :: length, incx, incy
!      type(c_ptr), value                      :: x, y, result
!    end subroutine
!  end interface


  interface mkl_openmp_offload_Cscal
    module procedure mkl_openmp_offload_Cscal_intptr
    module procedure mkl_openmp_offload_Cscal_cptr
  end interface

!  interface
!    subroutine mkl_openmp_offload_Cscal_intptr_c(mkl_openmp_offloadHandle, length, alpha, x, incx) &
!               bind(C, name="mkl_openmp_offloadCscal_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT),value               :: length, incx
!      complex(kind=C_FLOAT_COMPLEX) ,value                :: alpha
!      integer(kind=C_intptr_T), value         :: x
!    end subroutine
!  end interface

!  interface
!    subroutine mkl_openmp_offload_Cscal_cptr_c(mkl_openmp_offloadHandle, length, alpha, x, incx) &
!               bind(C, name="mkl_openmp_offloadCscal_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT),value               :: length, incx
!      complex(kind=C_FLOAT_COMPLEX) ,value                :: alpha
!      type(c_ptr), value                      :: x
!    end subroutine
!  end interface


  interface mkl_openmp_offload_Caxpy
    module procedure mkl_openmp_offload_Caxpy_intptr
    module procedure mkl_openmp_offload_Caxpy_cptr
  end interface

!  interface
!    subroutine mkl_openmp_offload_Caxpy_intptr_c(mkl_openmp_offloadHandle, length, alpha, x, incx, y, incy) &
!               bind(C, name="mkl_openmp_offloadCaxpy_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT),value               :: length, incx, incy
!      complex(kind=C_FLOAT_COMPLEX) ,value                :: alpha
!      integer(kind=C_intptr_T), value         :: x, y
!    end subroutine
!  end interface

!  interface
!    subroutine mkl_openmp_offload_Caxpy_cptr_c(mkl_openmp_offloadHandle, length, alpha, x, incx, y, incy) &
!               bind(C, name="mkl_openmp_offloadCaxpy_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: mkl_openmp_offloadHandle
!      integer(kind=C_INT),value               :: length, incx, incy
!      complex(kind=C_FLOAT_COMPLEX) ,value                :: alpha
!      type(c_ptr), value                      :: x, y
!    end subroutine
!  end interface

  contains

!    function openmp_offload_device_get_attributes(value, attribute) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_INT)                       :: value, attribute
!      logical                                   :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_device_get_attributes_c(value, attribute) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function mkl_openmp_offload_get_version(mkl_openmp_offloadHandle, version) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: mkl_openmp_offloadHandle
!      integer(kind=C_INT)                       :: version
!      logical                                   :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = mkl_openmp_offload_get_version_c(mkl_openmp_offloadHandle, version) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function openmp_offload_get_last_error() result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      logical                                   :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_get_last_error_c() /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function openmp_offload_stream_create(openmpOffloadStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: openmpOffloadStream
!      logical                                   :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_stream_create_c(openmpOffloadStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function openmp_offload_stream_destroy(openmpOffloadStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: openmpOffloadStream
!      logical                                   :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_stream_destroy_c(openmpOffloadStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function mkl_openmp_offload_set_stream(mkl_openmp_offloadHandle, openmpOffloadStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: mkl_openmp_offloadHandle
!      integer(kind=C_intptr_t)                  :: openmpOffloadStream
!      logical                                   :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = mkl_openmp_offload_set_stream_c(mkl_openmp_offloadHandle, openmpOffloadStream) /= 0
!#else
!      success = .true.
!#endif
!    end function


!    function openmp_offload_stream_synchronize(openmpOffloadStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t), optional        :: openmpOffloadStream
!      logical                                   :: success
!      if (present(openmpOffloadStream)) then
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!        success = openmp_offload_stream_synchronize_explicit_c(openmpOffloadStream) /= 0
!#else
!        success = .true.
!#endif
!      else
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!        success = openmp_offload_stream_synchronize_implicit_c() /= 0
!#else
!        success = .true.
!#endif
!      endif
!    end function

!#ifdef WITH_NVTX
!    ! this wrapper is needed for the string conversion
!    subroutine nvtxRangePush(range_name)
!      implicit none
!      character(len=*), intent(in) :: range_name
!
!      character(kind=C_CHAR,len=1), dimension(len(range_name)+1) :: c_name
!      integer i
!
!      do i = 1, len(range_name)
!        c_name(i) = range_name(i:i)
!      end do
!      c_name(len(range_name)+1) = char(0)
!
!      call nvtxRangePushA(c_name)
!    end subroutine
!#endif

    function mkl_openmp_offload_create(mkl_openmp_offloadHandle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: mkl_openmp_offloadHandle
      logical                                   :: success
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = mkl_openmp_offload_create_c(mkl_openmp_offloadHandle) /= 0
#else
      success = .true.
#endif
    end function

    function mkl_openmp_offload_destroy(mkl_openmp_offloadHandle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)   :: mkl_openmp_offloadHandle
      logical                    :: success
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = mkl_openmp_offload_destroy_c(mkl_openmp_offloadHandle) /= 0
#else
      success = .true.
#endif
    end function

    function openmp_offload_setdevice(n) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik), intent(in)  :: n
      logical                       :: success
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = openmp_offload_setdevice_c(int(n,kind=c_int)) /= 0
#else
      success = .true.
#endif
    end function

    function openmp_offload_getdevicecount(n) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik)     :: n
      integer(kind=c_int)  :: nCasted
      logical              :: success
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = openmp_offload_getdevicecount_c(nCasted) /=0
      n = int(nCasted)
#else
      success = .true.
      n = 0
#endif
    end function

!    function openmp_offload_devicesynchronize()result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      logical :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_devicesynchronize_c() /=0
!#else
!      success = .true.
!#endif
!    end function

    function openmp_offload_malloc_intptr(a, width_height) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t)                  :: a
      integer(kind=c_intptr_t), intent(in)      :: width_height
      logical                                   :: success
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = openmp_offload_malloc_intptr_c(a, width_height) /= 0
#else
      success = .true.
#endif
    end function

    function openmp_offload_malloc_cptr(a, width_height) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                               :: a
      integer(kind=c_intptr_t), intent(in)      :: width_height
      logical                                   :: success
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = openmp_offload_malloc_cptr_c(a, width_height) /= 0
#else
      success = .true.
#endif
    end function

    function openmp_offload_free_intptr(a) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: a
      logical                  :: success
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = openmp_offload_free_intptr_c(a) /= 0
#else
      success = .true.
#endif
    end function

    function openmp_offload_free_cptr(a) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr) :: a
      logical                  :: success
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = openmp_offload_free_cptr_c(a) /= 0
#else
      success = .true.
#endif
    end function

!    function openmp_offload_malloc_host_intptr(a, width_height) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t)                  :: a
!      integer(kind=c_intptr_t), intent(in)      :: width_height
!      logical                                   :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_malloc_host_intptr_c(a, width_height) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function openmp_offload_malloc_host_cptr(a, width_height) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)                               :: a
!      integer(kind=c_intptr_t), intent(in)      :: width_height
!      logical                                   :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_malloc_host_cptr_c(a, width_height) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function openmp_offload_free_host_intptr(a) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t) :: a
!      logical                  :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_free_host_intptr_c(a) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function openmp_offload_free_host_cptr(a) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)                   :: a
!      logical                  :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_free_host_cptr_c(a) /= 0
!#else
!      success = .true.
!#endif
!    end function

    function openmp_offload_memset(a, val, size) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t)                :: a
      integer(kind=ik)                        :: val
      integer(kind=c_intptr_t), intent(in)    :: size
      integer(kind=C_INT)                     :: istat
      logical :: success
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success= openmp_offload_memset_c(a, int(val,kind=c_int), int(size,kind=c_intptr_t)) /=0
#else
      success = .true.
#endif
    end function

!    function openmp_offload_memset_async(a, val, size, openmpOffloadStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t)                :: a
!      integer(kind=ik)                        :: val
!      integer(kind=c_intptr_t), intent(in)    :: size
!      integer(kind=C_INT)                     :: istat
!      integer(kind=c_intptr_t)                :: openmpOffloadStream
!      logical :: success
!
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success= openmp_offload_memset_async_c(a, int(val,kind=c_int), int(size,kind=c_intptr_t), openmpOffloadStream) /=0
!#else
!      success = .true.
!#endif
!    end function

    function openmp_offload_memcpyDeviceToDevice() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      flag = int(openmp_offload_memcpyDeviceToDevice_c())
#else
      flag = 0
#endif
    end function

    function openmp_offload_memcpyHostToDevice() result(flag)
      use, intrinsic :: iso_c_binding
      use precision
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      flag = int(openmp_offload_memcpyHostToDevice_c())
#else
      flag = 0
#endif
    end function

    function openmp_offload_memcpyDeviceToHost() result(flag)
      use, intrinsic :: iso_c_binding
      use precision
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      flag = int( openmp_offload_memcpyDeviceToHost_c())
#else
      flag = 0
#endif
    end function

!    function openmp_offload_hostRegisterDefault() result(flag)
!      use, intrinsic :: iso_c_binding
!      use precision
!      implicit none
!      integer(kind=ik) :: flag
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      flag = int(openmp_offload_hostRegisterDefault_c())
!#else
!      flag = 0
!#endif
!    end function

!    function openmp_offload_hostRegisterPortable() result(flag)
!      use, intrinsic :: iso_c_binding
!      use precision
!      implicit none
!      integer(kind=ik) :: flag
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      flag = int(openmp_offload_hostRegisterPortable_c())
!#else
!      flag = 0
!#endif
!    end function

!    function openmp_offload_hostRegisterMapped() result(flag)
!      use, intrinsic :: iso_c_binding
!      use precision
!      implicit none
!      integer(kind=ik) :: flag
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      flag = int(openmp_offload_hostRegisterMapped_c())
!#else
!      flag = 0
!#endif
!    end function

    function openmp_offload_memcpy_intptr(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)              :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = openmp_offload_memcpy_intptr_c(dst, src, size, dir) /= 0
#else
      success = .true.
#endif
    end function

    function openmp_offload_memcpy_cptr(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                           :: dst
      type(c_ptr)                           :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = openmp_offload_memcpy_cptr_c(dst, src, size, dir) /= 0
#else
      success = .true.
#endif
    end function

    function openmp_offload_memcpy_mixed_to_device(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                           :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = openmp_offload_memcpy_mixed_to_device_c(dst, src, size, dir) /= 0
#else
      success = .true.
#endif
    end function

    function openmp_offload_memcpy_mixed_to_host(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                           :: src
      integer(kind=C_intptr_t)              :: dst
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = openmp_offload_memcpy_mixed_to_host_c(dst, src, size, dir) /= 0
#else
      success = .true.
#endif
    end function

!    function openmp_offload_memcpy_async_intptr(dst, src, size, dir, openmpOffloadStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)              :: dst
!      integer(kind=C_intptr_t)              :: src
!      integer(kind=c_intptr_t), intent(in)  :: size
!      integer(kind=C_INT), intent(in)       :: dir
!      integer(kind=c_intptr_t), intent(in)  :: openmpOffloadStream
!      logical :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_memcpy_async_intptr_c(dst, src, size, dir, openmpOffloadStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function openmp_offload_memcpy_async_cptr(dst, src, size, dir, openmpOffloadStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)                           :: dst
!      type(c_ptr)                           :: src
!      integer(kind=c_intptr_t), intent(in)  :: size
!      integer(kind=C_INT), intent(in)       :: dir
!      integer(kind=c_intptr_t), intent(in)  :: openmpOffloadStream
!      logical :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_memcpy_async_cptr_c(dst, src, size, dir, openmpOffloadStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function openmp_offload_memcpy_async_mixed_to_device(dst, src, size, dir, openmpOffloadStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)                           :: dst
!      integer(kind=C_intptr_t)              :: src
!      integer(kind=c_intptr_t), intent(in)  :: size
!      integer(kind=C_INT), intent(in)       :: dir
!      integer(kind=c_intptr_t), intent(in)  :: openmpOffloadStream
!      logical :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_memcpy_async_mixed_to_device_c(dst, src, size, dir, openmpOffloadStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function openmp_offload_memcpy_async_mixed_to_host(dst, src, size, dir, openmpOffloadStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)                           :: src
!      integer(kind=C_intptr_t)              :: dst
!      integer(kind=c_intptr_t), intent(in)  :: size
!      integer(kind=C_INT), intent(in)       :: dir
!      integer(kind=c_intptr_t), intent(in)  :: openmpOffloadStream
!      logical :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_memcpy_async_mixed_to_host_c(dst, src, size, dir, openmpOffloadStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function openmp_offload_memcpy2d_intptr(dst, dpitch, src, spitch, width, height , dir) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T)           :: dst
!      integer(kind=c_intptr_t), intent(in) :: dpitch
!      integer(kind=C_intptr_T)           :: src
!      integer(kind=c_intptr_t), intent(in) :: spitch
!      integer(kind=c_intptr_t), intent(in) :: width
!      integer(kind=c_intptr_t), intent(in) :: height
!      integer(kind=C_INT), intent(in)    :: dir
!      logical                            :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_memcpy2d_intptr_c(dst, dpitch, src, spitch, width, height , dir) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function openmp_offload_memcpy2d_cptr(dst, dpitch, src, spitch, width, height , dir) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)           :: dst
!      integer(kind=c_intptr_t), intent(in) :: dpitch
!      type(c_ptr)           :: src
!      integer(kind=c_intptr_t), intent(in) :: spitch
!      integer(kind=c_intptr_t), intent(in) :: width
!      integer(kind=c_intptr_t), intent(in) :: height
!      integer(kind=C_INT), intent(in)    :: dir
!      logical                            :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_memcpy2d_cptr_c(dst, dpitch, src, spitch, width, height , dir) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function openmp_offload_memcpy2d_async_intptr(dst, dpitch, src, spitch, width, height, dir, openmpOffloadStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T)           :: dst
!      integer(kind=c_intptr_t), intent(in) :: dpitch
!      integer(kind=C_intptr_T)           :: src
!      integer(kind=c_intptr_t), intent(in) :: spitch
!      integer(kind=c_intptr_t), intent(in) :: width
!      integer(kind=c_intptr_t), intent(in) :: height
!      integer(kind=C_INT), intent(in)    :: dir
!      integer(kind=c_intptr_t), intent(in) :: openmpOffloadStream
!      logical                            :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_memcpy2d_async_intptr_c(dst, dpitch, src, spitch, width, height, dir, openmpOffloadStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function openmp_offload_memcpy2d_async_cptr(dst, dpitch, src, spitch, width, height, dir, openmpOffloadStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)           :: dst
!      integer(kind=c_intptr_t), intent(in) :: dpitch
!      type(c_ptr)           :: src
!      integer(kind=c_intptr_t), intent(in) :: spitch
!      integer(kind=c_intptr_t), intent(in) :: width
!      integer(kind=c_intptr_t), intent(in) :: height
!      integer(kind=C_INT), intent(in)    :: dir
!      integer(kind=c_intptr_t), intent(in) :: openmpOffloadStream
!      logical                            :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_memcpy2d_async_cptr_c(dst, dpitch, src, spitch, width, height, dir, openmpOffloadStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function openmp_offload_host_register(a, size, flag) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)              :: a
!      integer(kind=c_intptr_t), intent(in)  :: size
!      integer(kind=C_INT), intent(in)       :: flag
!      logical :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_host_register_c(a, size, flag) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function openmp_offload_host_unregister(a) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)              :: a
!      logical :: success
!#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!      success = openmp_offload_host_unregister_c(a) /= 0
!#else
!      success = .true.
!#endif
!    end function

    subroutine mkl_openmp_offload_Dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_DOUBLE) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Dgemm_intptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_DOUBLE) ,value               :: alpha, beta
      type(c_ptr)                     :: a, b, c
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Dgemm_cptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Dgemm_intptr_cptr_intptr(cta, ctb, m, n, k, &
                                 alpha, a, lda, b, ldb, beta, c, ldc, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_DOUBLE) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, c
      type(c_ptr)                     :: b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Dgemm_intptr_cptr_intptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, &
                                                 alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Dcopy_intptr(n, x, incx, y, incy, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Dcopy_intptr_c(mkl_openmp_offloadHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Dcopy_cptr(n, x, incx, y, incy, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Dcopy_cptr_c(mkl_openmp_offloadHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Dtrmm_intptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE) ,value               :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Dtrmm_cptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Dtrsm_intptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE) ,value               :: alpha
      type(c_ptr)                    :: a, b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Dtrsm_cptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_DOUBLE) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Dgemv_c(mkl_openmp_offloadHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_FLOAT) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Sgemm_intptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_FLOAT) ,value               :: alpha, beta
      type(c_ptr)                     :: a, b, c
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Sgemm_cptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Sgemm_intptr_cptr_intptr(cta, ctb, m, n, k, &
                                 alpha, a, lda, b, ldb, beta, c, ldc, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_FLOAT) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, c
      type(c_ptr)                     :: b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Sgemm_intptr_cptr_intptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, &
                                                 alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Scopy_intptr(n, x, incx, y, incy, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Scopy_intptr_c(mkl_openmp_offloadHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Scopy_cptr(n, x, incx, y, incy, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Scopy_cptr_c(mkl_openmp_offloadHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Strmm_intptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT) ,value               :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Strmm_cptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Strsm_intptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT) ,value               :: alpha
      type(c_ptr)                    :: a, b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Strsm_cptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_FLOAT) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Sgemv_c(mkl_openmp_offloadHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Zgemm_intptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha, beta
      type(c_ptr)                     :: a, b, c
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Zgemm_cptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Zgemm_intptr_cptr_intptr(cta, ctb, m, n, k, &
                                 alpha, a, lda, b, ldb, beta, c, ldc, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, c
      type(c_ptr)                     :: b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Zgemm_intptr_cptr_intptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, &
                                                 alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Zcopy_intptr(n, x, incx, y, incy, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Zcopy_intptr_c(mkl_openmp_offloadHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Zcopy_cptr(n, x, incx, y, incy, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Zcopy_cptr_c(mkl_openmp_offloadHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Ztrmm_intptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Ztrmm_cptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Ztrsm_intptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      type(c_ptr)                    :: a, b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Ztrsm_cptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Zgemv_c(mkl_openmp_offloadHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Cgemm_intptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha, beta
      type(c_ptr)                     :: a, b, c
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Cgemm_cptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Cgemm_intptr_cptr_intptr(cta, ctb, m, n, k, &
                                 alpha, a, lda, b, ldb, beta, c, ldc, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, c
      type(c_ptr)                     :: b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Cgemm_intptr_cptr_intptr_c(mkl_openmp_offloadHandle, cta, ctb, m, n, k, &
                                                 alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Ccopy_intptr(n, x, incx, y, incy, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Ccopy_intptr_c(mkl_openmp_offloadHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Ccopy_cptr(n, x, incx, y, incy, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Ccopy_cptr_c(mkl_openmp_offloadHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Ctrmm_intptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Ctrmm_cptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Ctrsm_intptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      type(c_ptr)                    :: a, b
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Ctrsm_cptr_c(mkl_openmp_offloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_Cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, mkl_openmp_offloadHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=C_intptr_T)        :: mkl_openmp_offloadHandle
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_Cgemv_c(mkl_openmp_offloadHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    function mkl_openmp_offload_pointerModeDevice() result(flag)
      use, intrinsic :: iso_c_binding
      use precision
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"pointerModeDevice not yet implemented!"
      flag = 1
      stop 1
#else
      flag = 0
#endif
    end function

    function mkl_openmp_offload_pointerModeHost() result(flag)
      use, intrinsic :: iso_c_binding
      use precision
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"pointerModeHost not yet implemented!"
      flag = 1
      stop 1
#else
      flag = 0
#endif
    end function

    subroutine mkl_openmp_offload_getPointerMode(mkl_openmp_offloadHandle, mode)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: mode

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"getPointerMode not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_setPointerMode(mkl_openmp_offloadHandle, mode)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: mode

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"setPointerMode not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Ddot_intptr(mkl_openmp_offloadHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx, incy
      integer(kind=c_intptr_t) :: x, y, result

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}DOT not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Ddot_cptr(mkl_openmp_offloadHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx, incy
      type(c_ptr)              :: x, y, result

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}DOT not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Dscal_intptr(mkl_openmp_offloadHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx
      real(kind=C_DOUBLE) ,value               :: alpha
      integer(kind=c_intptr_t) :: x

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}SCAL not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Dscal_cptr(mkl_openmp_offloadHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx
      real(kind=C_DOUBLE) ,value               :: alpha
      type(c_ptr)              :: x

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}SCAL not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Daxpy_intptr(mkl_openmp_offloadHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx, incy
      real(kind=C_DOUBLE) ,value               :: alpha
      integer(kind=c_intptr_t) :: x, y

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}AXPY not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Daxpy_cptr(mkl_openmp_offloadHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx, incy
      real(kind=C_DOUBLE) ,value               :: alpha
      type(c_ptr)              :: x, y

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}AXPY not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Sdot_intptr(mkl_openmp_offloadHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx, incy
      integer(kind=c_intptr_t) :: x, y, result

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}DOT not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Sdot_cptr(mkl_openmp_offloadHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx, incy
      type(c_ptr)              :: x, y, result

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}DOT not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Sscal_intptr(mkl_openmp_offloadHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx
      real(kind=C_FLOAT) ,value               :: alpha
      integer(kind=c_intptr_t) :: x

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}SCAL not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Sscal_cptr(mkl_openmp_offloadHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx
      real(kind=C_FLOAT) ,value               :: alpha
      type(c_ptr)              :: x

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}SCAL not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Saxpy_intptr(mkl_openmp_offloadHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx, incy
      real(kind=C_FLOAT) ,value               :: alpha
      integer(kind=c_intptr_t) :: x, y

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}AXPY not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Saxpy_cptr(mkl_openmp_offloadHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx, incy
      real(kind=C_FLOAT) ,value               :: alpha
      type(c_ptr)              :: x, y

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}AXPY not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Zdot_intptr(conj, mkl_openmp_offloadHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
       character(1,c_char), value   :: conj
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx, incy
      integer(kind=c_intptr_t) :: x, y, result

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}DOT not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Zdot_cptr(conj, mkl_openmp_offloadHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
       character(1,c_char), value   :: conj
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx, incy
      type(c_ptr)              :: x, y, result

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}DOT not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Zscal_intptr(mkl_openmp_offloadHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      integer(kind=c_intptr_t) :: x

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}SCAL not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Zscal_cptr(mkl_openmp_offloadHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      type(c_ptr)              :: x

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}SCAL not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Zaxpy_intptr(mkl_openmp_offloadHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx, incy
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      integer(kind=c_intptr_t) :: x, y

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}AXPY not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Zaxpy_cptr(mkl_openmp_offloadHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx, incy
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      type(c_ptr)              :: x, y

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}AXPY not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Cdot_intptr(conj, mkl_openmp_offloadHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
       character(1,c_char), value   :: conj
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx, incy
      integer(kind=c_intptr_t) :: x, y, result

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}DOT not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Cdot_cptr(conj, mkl_openmp_offloadHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
       character(1,c_char), value   :: conj
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx, incy
      type(c_ptr)              :: x, y, result

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}DOT not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Cscal_intptr(mkl_openmp_offloadHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      integer(kind=c_intptr_t) :: x

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}SCAL not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Cscal_cptr(mkl_openmp_offloadHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      type(c_ptr)              :: x

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}SCAL not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Caxpy_intptr(mkl_openmp_offloadHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx, incy
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      integer(kind=c_intptr_t) :: x, y

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}AXPY not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine mkl_openmp_offload_Caxpy_cptr(mkl_openmp_offloadHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: mkl_openmp_offloadHandle
      integer(kind=c_int)      :: length, incx, incy
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      type(c_ptr)              :: x, y

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      print *,"{X}AXPY not yet implemented!"
      stop 1
#endif
    end subroutine

