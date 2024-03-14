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


  integer(kind=ik) :: syclMemcpyHostToDevice
  integer(kind=ik) :: syclMemcpyDeviceToHost
  integer(kind=ik) :: syclMemcpyDeviceToDevice
  integer(kind=ik) :: syclHostRegisterDefault
  integer(kind=ik) :: syclHostRegisterPortable
  integer(kind=ik) :: syclHostRegisterMapped

  integer(kind=ik) :: syclblasPointerModeDevice
  integer(kind=ik) :: syclblasPointerModeHost


!  interface
!    function sycl_device_get_attributes_c(value, attribute) result(istat) &
!             bind(C, name="syclDeviceGetAttributeFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!
!      integer(kind=C_INT), value  :: attribute
!      integer(kind=C_INT)         :: value
!      integer(kind=C_INT)         :: istat
!    end function
!  end interface


!  interface
!    function syclblas_get_version_c(syclblasHandle, version) result(istat) &
!             bind(C, name="syclblasGetVersionFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!
!      integer(kind=C_intptr_T), value  :: syclblasHandle
!      integer(kind=C_INT)              :: version
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface


!  interface
!    function sycl_get_last_error_c() result(istat) &
!             bind(C, name="syclGetLastErrorFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int)              :: istat
!    end function
!  end interface

!  ! streams
!
!  interface
!    function sycl_stream_create_c(syclStream) result(istat) &
!             bind(C, name="syclStreamCreateFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T) :: syclStream
!      integer(kind=C_INT)      :: istat
!    end function
!  end interface

!  interface
!    function sycl_stream_destroy_c(syclStream) result(istat) &
!             bind(C, name="syclStreamDestroyFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value :: syclStream
!      integer(kind=C_INT)             :: istat
!    end function
!  end interface

!  interface
!    function sycl_stream_synchronize_explicit_c(syclStream) result(istat) &
!             bind(C, name="syclStreamSynchronizeExplicitFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!
!      integer(kind=C_intptr_T), value  :: syclStream
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface

!  interface
!    function sycl_stream_synchronize_implicit_c() result(istat) &
!             bind(C, name="syclStreamSynchronizeImplicitFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface

!  interface
!    function syclblas_set_stream_c(syclHandle, syclStream) result(istat) &
!             bind(C, name="syclblasSetStreamFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!
!      integer(kind=C_intptr_T), value  :: syclHandle
!      integer(kind=C_intptr_T), value  :: syclStream
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface

  interface
    function syclblas_create_c(syclHandle) result(istat) &
             bind(C, name="syclblasCreateFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: syclHandle
      integer(kind=C_INT)      :: istat
    end function
  end interface

  interface
    function syclblas_destroy_c(syclHandle) result(istat) &
             bind(C, name="syclblasDestroyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: syclHandle
      integer(kind=C_INT)      :: istat
    end function
  end interface

  ! functions to set and query the GPU devices
  interface
    function sycl_setdevice_c(n) result(istat) &
             bind(C, name="syclSetDeviceFromC")

      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT), value    :: n
      integer(kind=C_INT)           :: istat
    end function
  end interface

  interface
    function sycl_getdevicecount_c(n, onlyIntelgpus) result(istat) &
             bind(C, name="syclGetDeviceCountFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT), intent(out)         :: n
      integer(kind=C_INT), intent(in), value :: onlyIntelgpus
      integer(kind=C_INT)                      :: istat
    end function
  end interface


  interface
    function sycl_printdevices_c() result(n) &
             bind(C, name="syclPrintDevicesFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT) :: n
    end function
  end interface

  interface
    function sycl_getcpucount_c(n) result(istat) &
             bind(C, name="syclGetCpuCountFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT), intent(out) :: n
      integer(kind=C_INT)              :: istat
    end function
  end interface

!  interface
!    function sycl_devicesynchronize_c()result(istat) &
!             bind(C,name="syclDeviceSynchronizeFromC")
!
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_INT)                       :: istat
!    end function
!  end interface

  ! functions to copy GPU memory
  interface
    function sycl_memcpyDeviceToDevice_c() result(flag) &
             bind(C, name="syclMemcpyDeviceToDeviceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function sycl_memcpyHostToDevice_c() result(flag) &
             bind(C, name="syclMemcpyHostToDeviceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function sycl_memcpyDeviceToHost_c() result(flag) &
             bind(C, name="syclMemcpyDeviceToHostFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

!  interface
!    function sycl_hostRegisterDefault_c() result(flag) &
!             bind(C, name="syclHostRegisterDefaultFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface

!  interface
!    function sycl_hostRegisterPortable_c() result(flag) &
!             bind(C, name="syclHostRegisterPortableFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface

!  interface
!    function sycl_hostRegisterMapped_c() result(flag) &
!             bind(C, name="syclHostRegisterMappedFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface

  interface
    function sycl_memcpy_intptr_c(dst, src, size, dir) result(istat) &
             bind(C, name="syclMemcpyFromC")
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
    function sycl_memcpy_cptr_c(dst, src, size, dir) result(istat) &
             bind(C, name="syclMemcpyFromC")
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
    function sycl_memcpy_mixed_to_device_c(dst, src, size, dir) result(istat) &
             bind(C, name="syclMemcpyFromC")
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
    function sycl_memcpy_mixed_to_host_c(dst, src, size, dir) result(istat) &
             bind(C, name="syclMemcpyFromC")
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
!    function sycl_memcpy_async_intptr_c(dst, src, size, dir, syclStream) result(istat) &
!             bind(C, name="syclMemcpyAsyncFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t), value              :: dst
!      integer(kind=C_intptr_t), value              :: src
!      integer(kind=c_intptr_t), intent(in), value  :: size
!      integer(kind=C_INT), intent(in), value       :: dir
!      integer(kind=c_intptr_t), value              :: syclStream
!      integer(kind=C_INT)                          :: istat
!    end function
!  end interface

!  interface
!    function sycl_memcpy_async_cptr_c(dst, src, size, dir, syclStream) result(istat) &
!             bind(C, name="syclMemcpyAsyncFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr), value                           :: dst
!      type(c_ptr), value                           :: src
!      integer(kind=c_intptr_t), intent(in), value  :: size
!      integer(kind=C_INT), intent(in), value       :: dir
!      integer(kind=c_intptr_t), value              :: syclStream
!      integer(kind=C_INT)                          :: istat
!    end function
!  end interface

!  interface
!    function sycl_memcpy_async_mixed_to_device_c(dst, src, size, dir, syclStream) result(istat) &
!             bind(C, name="syclMemcpyAsyncFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr), value                           :: dst
!      integer(kind=C_intptr_t), value              :: src
!      integer(kind=c_intptr_t), intent(in), value  :: size
!      integer(kind=C_INT), intent(in), value       :: dir
!      integer(kind=c_intptr_t), value              :: syclStream
!      integer(kind=C_INT)                          :: istat
!    end function
!  end interface

!  interface
!    function sycl_memcpy_async_mixed_to_host_c(dst, src, size, dir, syclStream) result(istat) &
!             bind(C, name="syclMemcpyAsyncFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr), value                           :: src
!      integer(kind=C_intptr_t), value              :: dst
!      integer(kind=c_intptr_t), intent(in), value  :: size
!      integer(kind=C_INT), intent(in), value       :: dir
!      integer(kind=c_intptr_t), value              :: syclStream
!      integer(kind=C_INT)                          :: istat
!    end function
!  end interface

!  interface
!    function sycl_memcpy2d_intptr_c(dst, dpitch, src, spitch, width, height , dir) result(istat) &
!             bind(C, name="syclMemcpy2dFromC")
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
!    function sycl_memcpy2d_cptr_c(dst, dpitch, src, spitch, width, height , dir) result(istat) &
!             bind(C, name="syclMemcpy2dFromC")
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
!    function sycl_memcpy2d_async_intptr_c(dst, dpitch, src, spitch, width, height, dir, syclStream) result(istat) &
!             bind(C, name="syclMemcpy2dAsyncFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value                :: dst
!      integer(kind=c_intptr_t), intent(in), value    :: dpitch
!      integer(kind=C_intptr_T), value                :: src
!      integer(kind=c_intptr_t), intent(in), value    :: spitch
!      integer(kind=c_intptr_t), intent(in), value    :: width
!      integer(kind=c_intptr_t), intent(in), value    :: height
!      integer(kind=C_INT), intent(in), value         :: dir
!      integer(kind=c_intptr_t), value                :: syclStream
!      integer(kind=C_INT)                            :: istat
!    end function
!  end interface

!  interface
!    function sycl_memcpy2d_async_cptr_c(dst, dpitch, src, spitch, width, height, dir, syclStream) result(istat) &
!             bind(C, name="syclMemcpy2dAsyncFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr), value                :: dst
!      integer(kind=c_intptr_t), intent(in), value    :: dpitch
!      type(c_ptr), value                :: src
!      integer(kind=c_intptr_t), intent(in), value    :: spitch
!      integer(kind=c_intptr_t), intent(in), value    :: width
!      integer(kind=c_intptr_t), intent(in), value    :: height
!      integer(kind=C_INT), intent(in), value         :: dir
!      integer(kind=c_intptr_t), value                :: syclStream
!      integer(kind=C_INT)                            :: istat
!    end function
!  end interface

!  interface
!    function sycl_host_register_c(a, size, flag) result(istat) &
!             bind(C, name="syclHostRegisterFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t), value              :: a
!      integer(kind=c_intptr_t), intent(in), value  :: size
!      integer(kind=C_INT), intent(in), value       :: flag
!      integer(kind=C_INT)                          :: istat
!    end function
!  end interface

!  interface
!    function sycl_host_unregister_c(a) result(istat) &
!             bind(C, name="syclHostUnregisterFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t), value              :: a
!      integer(kind=C_INT)                          :: istat
!    end function
!  end interface

  interface sycl_free
    module procedure sycl_free_intptr
    module procedure sycl_free_cptr
  end interface

  interface
    function sycl_free_intptr_c(a) result(istat) &
             bind(C, name="syclFreeFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T)  :: a
      integer(kind=C_INT)              :: istat
    end function
  end interface

  interface
    function sycl_free_cptr_c(a) result(istat) &
             bind(C, name="syclFreeFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)  :: a
      integer(kind=C_INT)              :: istat
    end function
  end interface

  interface sycl_memcpy
    module procedure sycl_memcpy_intptr
    module procedure sycl_memcpy_cptr
    module procedure sycl_memcpy_mixed_to_device
    module procedure sycl_memcpy_mixed_to_host
  end interface

!  interface sycl_memcpy_async
!    module procedure sycl_memcpy_async_intptr
!    module procedure sycl_memcpy_async_cptr
!    module procedure sycl_memcpy_async_mixed_to_device
!    module procedure sycl_memcpy_async_mixed_to_host
!  end interface

  interface sycl_malloc
    module procedure sycl_malloc_intptr
    module procedure sycl_malloc_cptr
  end interface

  interface
    function sycl_malloc_intptr_c(a, width_height) result(istat) &
             bind(C, name="syclMallocFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      ! no value since **pointer
      integer(kind=C_intptr_T)                    :: a
      integer(kind=c_intptr_t), intent(in), value :: width_height
      integer(kind=C_INT)                         :: istat
    end function
  end interface

  interface
    function sycl_malloc_cptr_c(a, width_height) result(istat) &
             bind(C, name="syclMallocFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      ! no value since **pointer
      type(c_ptr)                                 :: a
      integer(kind=c_intptr_t), intent(in), value :: width_height
      integer(kind=C_INT)                         :: istat
    end function
  end interface

!  interface sycl_free_host
!    module procedure sycl_free_host_intptr
!    module procedure sycl_free_host_cptr
!  end interface
!  interface
!    function sycl_free_host_intptr_c(a) result(istat) &
!             bind(C, name="syclFreeHostFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t), value  :: a
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface

!  interface
!    function sycl_free_host_cptr_c(a) result(istat) &
!             bind(C, name="syclFreeHostFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr), value               :: a
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface

!  interface sycl_malloc_host
!    module procedure sycl_malloc_host_intptr
!    module procedure sycl_malloc_host_cptr
!  end interface
!  interface
!    function sycl_malloc_host_intptr_c(a, width_height) result(istat) &
!             bind(C, name="syclMallocHostFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t)                    :: a
!      integer(kind=c_intptr_t), intent(in), value :: width_height
!      integer(kind=C_INT)                         :: istat
!    end function
!  end interface

!  interface
!    function sycl_malloc_host_cptr_c(a, width_height) result(istat) &
!             bind(C, name="syclMallocHostFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)                    :: a
!      integer(kind=c_intptr_t), intent(in), value :: width_height
!      integer(kind=C_INT)                         :: istat
!    end function
!  end interface

  interface
    function sycl_memset_c(a, val, size) result(istat) &
             bind(C, name="syclMemsetFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value            :: a
      integer(kind=C_INT), value                 :: val
      integer(kind=c_intptr_t), intent(in), value  :: size
      integer(kind=C_INT)                        :: istat
    end function
  end interface

!  interface
!    function sycl_memset_async_c(a, val, size, syclStream) result(istat) &
!             bind(C, name="syclMemsetAsyncFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value            :: a
!      integer(kind=C_INT), value                 :: val
!      integer(kind=c_intptr_t), intent(in), value  :: size
!      integer(kind=C_INT)                        :: istat
!      integer(kind=c_intptr_t), value            :: syclStream
!    end function
!  end interface

  interface syclblas_Dgemm
    module procedure syclblas_Dgemm_intptr
    module procedure syclblas_Dgemm_cptr
    module procedure syclblas_Dgemm_intptr_cptr_intptr
  end interface

  interface
    subroutine syclblas_Dgemm_intptr_c(syclblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="syclblasDgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_DOUBLE) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Dgemm_cptr_c(syclblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="syclblasDgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_DOUBLE) ,value               :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Dgemm_intptr_cptr_intptr_c(syclblasHandle, cta, ctb, m, n, k, &
                                                                      alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="syclblasDgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m, n, k
      integer(kind=C_INT), intent(in), value  :: lda, ldb, ldc
      real(kind=C_DOUBLE) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, c
      type(c_ptr), value                      :: b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface


  interface syclblas_Dcopy
    module procedure syclblas_Dcopy_intptr
    module procedure syclblas_Dcopy_cptr
  end interface

  interface
    subroutine syclblas_Dcopy_intptr_c(syclblasHandle, n, x, incx, y, incy) &
                              bind(C,name="syclblasDcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Dcopy_cptr_c(syclblasHandle, n, x, incx, y, incy) &
                              bind(C,name="syclblasDcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface


  interface syclblas_Dtrmm
    module procedure syclblas_Dtrmm_intptr
    module procedure syclblas_Dtrmm_cptr
  end interface

  interface
    subroutine syclblas_Dtrmm_intptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="syclblasDtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Dtrmm_cptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="syclblasDtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface


  interface syclblas_Dtrsm
    module procedure syclblas_Dtrsm_intptr
    module procedure syclblas_Dtrsm_cptr
  end interface

  interface
    subroutine syclblas_Dtrsm_intptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="syclblasDtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Dtrsm_cptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="syclblasDtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Dgemv_c(syclblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name="syclblasDgemv_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      real(kind=C_DOUBLE) , value              :: alpha, beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface syclblas_Sgemm
    module procedure syclblas_Sgemm_intptr
    module procedure syclblas_Sgemm_cptr
    module procedure syclblas_Sgemm_intptr_cptr_intptr
  end interface

  interface
    subroutine syclblas_Sgemm_intptr_c(syclblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="syclblasSgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_FLOAT) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Sgemm_cptr_c(syclblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="syclblasSgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_FLOAT) ,value               :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Sgemm_intptr_cptr_intptr_c(syclblasHandle, cta, ctb, m, n, k, &
                                                                      alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="syclblasSgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m, n, k
      integer(kind=C_INT), intent(in), value  :: lda, ldb, ldc
      real(kind=C_FLOAT) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, c
      type(c_ptr), value                      :: b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface


  interface syclblas_Scopy
    module procedure syclblas_Scopy_intptr
    module procedure syclblas_Scopy_cptr
  end interface

  interface
    subroutine syclblas_Scopy_intptr_c(syclblasHandle, n, x, incx, y, incy) &
                              bind(C,name="syclblasScopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Scopy_cptr_c(syclblasHandle, n, x, incx, y, incy) &
                              bind(C,name="syclblasScopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface


  interface syclblas_Strmm
    module procedure syclblas_Strmm_intptr
    module procedure syclblas_Strmm_cptr
  end interface

  interface
    subroutine syclblas_Strmm_intptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="syclblasStrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Strmm_cptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="syclblasStrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface


  interface syclblas_Strsm
    module procedure syclblas_Strsm_intptr
    module procedure syclblas_Strsm_cptr
  end interface

  interface
    subroutine syclblas_Strsm_intptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="syclblasStrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Strsm_cptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="syclblasStrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Sgemv_c(syclblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name="syclblasSgemv_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      real(kind=C_FLOAT) , value              :: alpha, beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface syclblas_Zgemm
    module procedure syclblas_Zgemm_intptr
    module procedure syclblas_Zgemm_cptr
    module procedure syclblas_Zgemm_intptr_cptr_intptr
  end interface

  interface
    subroutine syclblas_Zgemm_intptr_c(syclblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="syclblasZgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Zgemm_cptr_c(syclblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="syclblasZgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Zgemm_intptr_cptr_intptr_c(syclblasHandle, cta, ctb, m, n, k, &
                                                                      alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="syclblasZgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m, n, k
      integer(kind=C_INT), intent(in), value  :: lda, ldb, ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, c
      type(c_ptr), value                      :: b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface


  interface syclblas_Zcopy
    module procedure syclblas_Zcopy_intptr
    module procedure syclblas_Zcopy_cptr
  end interface

  interface
    subroutine syclblas_Zcopy_intptr_c(syclblasHandle, n, x, incx, y, incy) &
                              bind(C,name="syclblasZcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Zcopy_cptr_c(syclblasHandle, n, x, incx, y, incy) &
                              bind(C,name="syclblasZcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface


  interface syclblas_Ztrmm
    module procedure syclblas_Ztrmm_intptr
    module procedure syclblas_Ztrmm_cptr
  end interface

  interface
    subroutine syclblas_Ztrmm_intptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="syclblasZtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Ztrmm_cptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="syclblasZtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface


  interface syclblas_Ztrsm
    module procedure syclblas_Ztrsm_intptr
    module procedure syclblas_Ztrsm_cptr
  end interface

  interface
    subroutine syclblas_Ztrsm_intptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="syclblasZtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Ztrsm_cptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="syclblasZtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Zgemv_c(syclblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name="syclblasZgemv_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha, beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface syclblas_Cgemm
    module procedure syclblas_Cgemm_intptr
    module procedure syclblas_Cgemm_cptr
    module procedure syclblas_Cgemm_intptr_cptr_intptr
  end interface

  interface
    subroutine syclblas_Cgemm_intptr_c(syclblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="syclblasCgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Cgemm_cptr_c(syclblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="syclblasCgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Cgemm_intptr_cptr_intptr_c(syclblasHandle, cta, ctb, m, n, k, &
                                                                      alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="syclblasCgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m, n, k
      integer(kind=C_INT), intent(in), value  :: lda, ldb, ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, c
      type(c_ptr), value                      :: b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface


  interface syclblas_Ccopy
    module procedure syclblas_Ccopy_intptr
    module procedure syclblas_Ccopy_cptr
  end interface

  interface
    subroutine syclblas_Ccopy_intptr_c(syclblasHandle, n, x, incx, y, incy) &
                              bind(C,name="syclblasCcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Ccopy_cptr_c(syclblasHandle, n, x, incx, y, incy) &
                              bind(C,name="syclblasCcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface


  interface syclblas_Ctrmm
    module procedure syclblas_Ctrmm_intptr
    module procedure syclblas_Ctrmm_cptr
  end interface

  interface
    subroutine syclblas_Ctrmm_intptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="syclblasCtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Ctrmm_cptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="syclblasCtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface


  interface syclblas_Ctrsm
    module procedure syclblas_Ctrsm_intptr
    module procedure syclblas_Ctrsm_cptr
  end interface

  interface
    subroutine syclblas_Ctrsm_intptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="syclblasCtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Ctrsm_cptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="syclblasCtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: syclblasHandle
    end subroutine
  end interface

  interface
    subroutine syclblas_Cgemv_c(syclblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name="syclblasCgemv_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha, beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: syclblasHandle
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
!    function syclblas_pointerModeDevice_c() result(flag) &
!               bind(C, name="syclblasPointerModeDeviceFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface

!  interface
!    function syclblas_pointerModeHost_c() result(flag) &
!               bind(C, name="syclblasPointerModeHostFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface

!  interface
!    subroutine syclblas_getPointerMode_c(syclblasHandle, mode) &
!               bind(C, name="syclblasGetPointerModeFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value   :: syclblasHandle
!      integer(kind=c_int)               :: mode
!    end subroutine
!  end interface

!  interface
!    subroutine syclblas_setPointerMode_c(syclblasHandle, mode) &
!               bind(C, name="syclblasSetPointerModeFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T),value    :: syclblasHandle
!      integer(kind=c_int), value        :: mode
!    end subroutine
!  end interface


  interface syclblas_Ddot
    module procedure syclblas_Ddot_intptr
    module procedure syclblas_Ddot_cptr
  end interface

!  interface
!    subroutine syclblas_Ddot_intptr_c(syclblasHandle, length, x, incx, y, incy, result) &
!               bind(C, name="syclblasDdot_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT), value              :: length, incx, incy
!      integer(kind=C_intptr_T), value         :: x, y, result
!    end subroutine
!  end interface

!  interface
!    subroutine syclblas_Ddot_cptr_c(syclblasHandle, length, x, incx, y, incy, result) &
!               bind(C, name="syclblasDdot_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT), value              :: length, incx, incy
!      type(c_ptr), value                      :: x, y, result
!    end subroutine
!  end interface


  interface syclblas_Dscal
    module procedure syclblas_Dscal_intptr
    module procedure syclblas_Dscal_cptr
  end interface

!  interface
!    subroutine syclblas_Dscal_intptr_c(syclblasHandle, length, alpha, x, incx) &
!               bind(C, name="syclblasDscal_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT),value               :: length, incx
!      real(kind=C_DOUBLE) ,value                :: alpha
!      integer(kind=C_intptr_T), value         :: x
!    end subroutine
!  end interface

!  interface
!    subroutine syclblas_Dscal_cptr_c(syclblasHandle, length, alpha, x, incx) &
!               bind(C, name="syclblasDscal_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT),value               :: length, incx
!      real(kind=C_DOUBLE) ,value                :: alpha
!      type(c_ptr), value                      :: x
!    end subroutine
!  end interface


  interface syclblas_Daxpy
    module procedure syclblas_Daxpy_intptr
    module procedure syclblas_Daxpy_cptr
  end interface

!  interface
!    subroutine syclblas_Daxpy_intptr_c(syclblasHandle, length, alpha, x, incx, y, incy) &
!               bind(C, name="syclblasDaxpy_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT),value               :: length, incx, incy
!      real(kind=C_DOUBLE) ,value                :: alpha
!      integer(kind=C_intptr_T), value         :: x, y
!    end subroutine
!  end interface

!  interface
!    subroutine syclblas_Daxpy_cptr_c(syclblasHandle, length, alpha, x, incx, y, incy) &
!               bind(C, name="syclblasDaxpy_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT),value               :: length, incx, incy
!      real(kind=C_DOUBLE) ,value                :: alpha
!      type(c_ptr), value                      :: x, y
!    end subroutine
!  end interface

  interface syclblas_Sdot
    module procedure syclblas_Sdot_intptr
    module procedure syclblas_Sdot_cptr
  end interface

!  interface
!    subroutine syclblas_Sdot_intptr_c(syclblasHandle, length, x, incx, y, incy, result) &
!               bind(C, name="syclblasSdot_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT), value              :: length, incx, incy
!      integer(kind=C_intptr_T), value         :: x, y, result
!    end subroutine
!  end interface

!  interface
!    subroutine syclblas_Sdot_cptr_c(syclblasHandle, length, x, incx, y, incy, result) &
!               bind(C, name="syclblasSdot_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT), value              :: length, incx, incy
!      type(c_ptr), value                      :: x, y, result
!    end subroutine
!  end interface


  interface syclblas_Sscal
    module procedure syclblas_Sscal_intptr
    module procedure syclblas_Sscal_cptr
  end interface

!  interface
!    subroutine syclblas_Sscal_intptr_c(syclblasHandle, length, alpha, x, incx) &
!               bind(C, name="syclblasSscal_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT),value               :: length, incx
!      real(kind=C_FLOAT) ,value                :: alpha
!      integer(kind=C_intptr_T), value         :: x
!    end subroutine
!  end interface

!  interface
!    subroutine syclblas_Sscal_cptr_c(syclblasHandle, length, alpha, x, incx) &
!               bind(C, name="syclblasSscal_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT),value               :: length, incx
!      real(kind=C_FLOAT) ,value                :: alpha
!      type(c_ptr), value                      :: x
!    end subroutine
!  end interface


  interface syclblas_Saxpy
    module procedure syclblas_Saxpy_intptr
    module procedure syclblas_Saxpy_cptr
  end interface

!  interface
!    subroutine syclblas_Saxpy_intptr_c(syclblasHandle, length, alpha, x, incx, y, incy) &
!               bind(C, name="syclblasSaxpy_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT),value               :: length, incx, incy
!      real(kind=C_FLOAT) ,value                :: alpha
!      integer(kind=C_intptr_T), value         :: x, y
!    end subroutine
!  end interface

!  interface
!    subroutine syclblas_Saxpy_cptr_c(syclblasHandle, length, alpha, x, incx, y, incy) &
!               bind(C, name="syclblasSaxpy_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT),value               :: length, incx, incy
!      real(kind=C_FLOAT) ,value                :: alpha
!      type(c_ptr), value                      :: x, y
!    end subroutine
!  end interface

  interface syclblas_Zdot
    module procedure syclblas_Zdot_intptr
    module procedure syclblas_Zdot_cptr
  end interface

!  interface
!    subroutine syclblas_Zdot_intptr_c(conj, syclblasHandle, length, x, incx, y, incy, result) &
!               bind(C, name="syclblasZdot_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      character(1,C_CHAR),value               :: conj
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT), value              :: length, incx, incy
!      integer(kind=C_intptr_T), value         :: x, y, result
!    end subroutine
!  end interface

!  interface
!    subroutine syclblas_Zdot_cptr_c(conj, syclblasHandle, length, x, incx, y, incy, result) &
!               bind(C, name="syclblasZdot_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      character(1,C_CHAR),value               :: conj
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT), value              :: length, incx, incy
!      type(c_ptr), value                      :: x, y, result
!    end subroutine
!  end interface


  interface syclblas_Zscal
    module procedure syclblas_Zscal_intptr
    module procedure syclblas_Zscal_cptr
  end interface

!  interface
!    subroutine syclblas_Zscal_intptr_c(syclblasHandle, length, alpha, x, incx) &
!               bind(C, name="syclblasZscal_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT),value               :: length, incx
!      complex(kind=C_DOUBLE_COMPLEX) ,value                :: alpha
!      integer(kind=C_intptr_T), value         :: x
!    end subroutine
!  end interface

!  interface
!    subroutine syclblas_Zscal_cptr_c(syclblasHandle, length, alpha, x, incx) &
!               bind(C, name="syclblasZscal_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT),value               :: length, incx
!      complex(kind=C_DOUBLE_COMPLEX) ,value                :: alpha
!      type(c_ptr), value                      :: x
!    end subroutine
!  end interface


  interface syclblas_Zaxpy
    module procedure syclblas_Zaxpy_intptr
    module procedure syclblas_Zaxpy_cptr
  end interface

!  interface
!    subroutine syclblas_Zaxpy_intptr_c(syclblasHandle, length, alpha, x, incx, y, incy) &
!               bind(C, name="syclblasZaxpy_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT),value               :: length, incx, incy
!      complex(kind=C_DOUBLE_COMPLEX) ,value                :: alpha
!      integer(kind=C_intptr_T), value         :: x, y
!    end subroutine
!  end interface

!  interface
!    subroutine syclblas_Zaxpy_cptr_c(syclblasHandle, length, alpha, x, incx, y, incy) &
!               bind(C, name="syclblasZaxpy_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT),value               :: length, incx, incy
!      complex(kind=C_DOUBLE_COMPLEX) ,value                :: alpha
!      type(c_ptr), value                      :: x, y
!    end subroutine
!  end interface

  interface syclblas_Cdot
    module procedure syclblas_Cdot_intptr
    module procedure syclblas_Cdot_cptr
  end interface

!  interface
!    subroutine syclblas_Cdot_intptr_c(conj, syclblasHandle, length, x, incx, y, incy, result) &
!               bind(C, name="syclblasCdot_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      character(1,C_CHAR),value               :: conj
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT), value              :: length, incx, incy
!      integer(kind=C_intptr_T), value         :: x, y, result
!    end subroutine
!  end interface

!  interface
!    subroutine syclblas_Cdot_cptr_c(conj, syclblasHandle, length, x, incx, y, incy, result) &
!               bind(C, name="syclblasCdot_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      character(1,C_CHAR),value               :: conj
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT), value              :: length, incx, incy
!      type(c_ptr), value                      :: x, y, result
!    end subroutine
!  end interface


  interface syclblas_Cscal
    module procedure syclblas_Cscal_intptr
    module procedure syclblas_Cscal_cptr
  end interface

!  interface
!    subroutine syclblas_Cscal_intptr_c(syclblasHandle, length, alpha, x, incx) &
!               bind(C, name="syclblasCscal_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT),value               :: length, incx
!      complex(kind=C_FLOAT_COMPLEX) ,value                :: alpha
!      integer(kind=C_intptr_T), value         :: x
!    end subroutine
!  end interface

!  interface
!    subroutine syclblas_Cscal_cptr_c(syclblasHandle, length, alpha, x, incx) &
!               bind(C, name="syclblasCscal_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT),value               :: length, incx
!      complex(kind=C_FLOAT_COMPLEX) ,value                :: alpha
!      type(c_ptr), value                      :: x
!    end subroutine
!  end interface


  interface syclblas_Caxpy
    module procedure syclblas_Caxpy_intptr
    module procedure syclblas_Caxpy_cptr
  end interface

!  interface
!    subroutine syclblas_Caxpy_intptr_c(syclblasHandle, length, alpha, x, incx, y, incy) &
!               bind(C, name="syclblasCaxpy_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT),value               :: length, incx, incy
!      complex(kind=C_FLOAT_COMPLEX) ,value                :: alpha
!      integer(kind=C_intptr_T), value         :: x, y
!    end subroutine
!  end interface

!  interface
!    subroutine syclblas_Caxpy_cptr_c(syclblasHandle, length, alpha, x, incx, y, incy) &
!               bind(C, name="syclblasCaxpy_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value         :: syclblasHandle
!      integer(kind=C_INT),value               :: length, incx, incy
!      complex(kind=C_FLOAT_COMPLEX) ,value                :: alpha
!      type(c_ptr), value                      :: x, y
!    end subroutine
!  end interface

  contains

!    function sycl_device_get_attributes(value, attribute) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_INT)                       :: value, attribute
!      logical                                   :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_device_get_attributes_c(value, attribute) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function syclblas_get_version(syclblasHandle, version) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: syclblasHandle
!      integer(kind=C_INT)                       :: version
!      logical                                   :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = syclblas_get_version_c(syclblasHandle, version) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function sycl_get_last_error() result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      logical                                   :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_get_last_error_c() /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function sycl_stream_create(syclStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: syclStream
!      logical                                   :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_stream_create_c(syclStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function sycl_stream_destroy(syclStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: syclStream
!      logical                                   :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_stream_destroy_c(syclStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function syclblas_set_stream(syclblasHandle, syclStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: syclblasHandle
!      integer(kind=C_intptr_t)                  :: syclStream
!      logical                                   :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = syclblas_set_stream_c(syclblasHandle, syclStream) /= 0
!#else
!      success = .true.
!#endif
!    end function


!    function sycl_stream_synchronize(syclStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t), optional        :: syclStream
!      logical                                   :: success
!      if (present(syclStream)) then
!#ifdef WITH_SYCL_GPU_VERSION
!        success = sycl_stream_synchronize_explicit_c(syclStream) /= 0
!#else
!        success = .true.
!#endif
!      else
!#ifdef WITH_SYCL_GPU_VERSION
!        success = sycl_stream_synchronize_implicit_c() /= 0
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

    function syclblas_create(syclblasHandle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: syclblasHandle
      logical                                   :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = syclblas_create_c(syclblasHandle) /= 0
#else
      success = .true.
#endif
    end function

    function syclblas_destroy(syclblasHandle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)   :: syclblasHandle
      logical                    :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = syclblas_destroy_c(syclblasHandle) /= 0
#else
      success = .true.
#endif
    end function

    function sycl_setdevice(n) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik), intent(in)  :: n
      logical                       :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = sycl_setdevice_c(int(n,kind=c_int)) /= 0
#else
      success = .true.
#endif
    end function

    function sycl_getdevicecount(n, onlyIntelGpus) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik)     :: n
      integer(kind=c_int)  :: onlyIntelGpus
      integer(kind=c_int)  :: nCasted
      logical              :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = sycl_getdevicecount_c(nCasted, onlyIntelGpus) /=0
      n = int(nCasted)
#else
      success = .true.
      n = 0
#endif
    end function

    subroutine  sycl_printdevices()
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik)           :: n

#ifdef WITH_SYCL_GPU_VERSION
      n = sycl_printdevices_c()
#else
      n = 0
#endif
    end subroutine sycl_printdevices

    function sycl_getcpucount(n) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik)     :: n
      integer(kind=c_int)  :: nCasted
      logical              :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = sycl_getcpucount_c(nCasted) /=0
      n = int(nCasted)
#else
      success = .true.
      n = 0
#endif
    end function

!    function sycl_devicesynchronize()result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      logical :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_devicesynchronize_c() /=0
!#else
!      success = .true.
!#endif
!    end function

    function sycl_malloc_intptr(a, width_height) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t)                  :: a
      integer(kind=c_intptr_t), intent(in)      :: width_height
      logical                                   :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = sycl_malloc_intptr_c(a, width_height) /= 0
#else
      success = .true.
#endif
    end function

    function sycl_malloc_cptr(a, width_height) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                               :: a
      integer(kind=c_intptr_t), intent(in)      :: width_height
      logical                                   :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = sycl_malloc_cptr_c(a, width_height) /= 0
#else
      success = .true.
#endif
    end function

    function sycl_free_intptr(a) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: a
      logical                  :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = sycl_free_intptr_c(a) /= 0
#else
      success = .true.
#endif
    end function

    function sycl_free_cptr(a) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr) :: a
      logical                  :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = sycl_free_cptr_c(a) /= 0
#else
      success = .true.
#endif
    end function

!    function sycl_malloc_host_intptr(a, width_height) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t)                  :: a
!      integer(kind=c_intptr_t), intent(in)      :: width_height
!      logical                                   :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_malloc_host_intptr_c(a, width_height) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function sycl_malloc_host_cptr(a, width_height) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)                               :: a
!      integer(kind=c_intptr_t), intent(in)      :: width_height
!      logical                                   :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_malloc_host_cptr_c(a, width_height) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function sycl_free_host_intptr(a) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t) :: a
!      logical                  :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_free_host_intptr_c(a) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function sycl_free_host_cptr(a) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)                   :: a
!      logical                  :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_free_host_cptr_c(a) /= 0
!#else
!      success = .true.
!#endif
!    end function

    function sycl_memset(a, val, size) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t)                :: a
      integer(kind=ik)                        :: val
      integer(kind=c_intptr_t), intent(in)    :: size
      integer(kind=C_INT)                     :: istat
      logical :: success
#ifdef WITH_SYCL_GPU_VERSION
      success= sycl_memset_c(a, int(val,kind=c_int), int(size,kind=c_intptr_t)) /=0
#else
      success = .true.
#endif
    end function

!    function sycl_memset_async(a, val, size, syclStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t)                :: a
!      integer(kind=ik)                        :: val
!      integer(kind=c_intptr_t), intent(in)    :: size
!      integer(kind=C_INT)                     :: istat
!      integer(kind=c_intptr_t)                :: syclStream
!      logical :: success
!
!#ifdef WITH_SYCL_GPU_VERSION
!      success= sycl_memset_async_c(a, int(val,kind=c_int), int(size,kind=c_intptr_t), syclStream) /=0
!#else
!      success = .true.
!#endif
!    end function

    function sycl_memcpyDeviceToDevice() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_SYCL_GPU_VERSION
      flag = int(sycl_memcpyDeviceToDevice_c())
#else
      flag = 0
#endif
    end function

    function sycl_memcpyHostToDevice() result(flag)
      use, intrinsic :: iso_c_binding
      use precision
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_SYCL_GPU_VERSION
      flag = int(sycl_memcpyHostToDevice_c())
#else
      flag = 0
#endif
    end function

    function sycl_memcpyDeviceToHost() result(flag)
      use, intrinsic :: iso_c_binding
      use precision
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_SYCL_GPU_VERSION
      flag = int( sycl_memcpyDeviceToHost_c())
#else
      flag = 0
#endif
    end function

!    function sycl_hostRegisterDefault() result(flag)
!      use, intrinsic :: iso_c_binding
!      use precision
!      implicit none
!      integer(kind=ik) :: flag
!#ifdef WITH_SYCL_GPU_VERSION
!      flag = int(sycl_hostRegisterDefault_c())
!#else
!      flag = 0
!#endif
!    end function

!    function sycl_hostRegisterPortable() result(flag)
!      use, intrinsic :: iso_c_binding
!      use precision
!      implicit none
!      integer(kind=ik) :: flag
!#ifdef WITH_SYCL_GPU_VERSION
!      flag = int(sycl_hostRegisterPortable_c())
!#else
!      flag = 0
!#endif
!    end function

!    function sycl_hostRegisterMapped() result(flag)
!      use, intrinsic :: iso_c_binding
!      use precision
!      implicit none
!      integer(kind=ik) :: flag
!#ifdef WITH_SYCL_GPU_VERSION
!      flag = int(sycl_hostRegisterMapped_c())
!#else
!      flag = 0
!#endif
!    end function

    function sycl_memcpy_intptr(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)              :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = sycl_memcpy_intptr_c(dst, src, size, dir) /= 0
#else
      success = .true.
#endif
    end function

    function sycl_memcpy_cptr(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                           :: dst
      type(c_ptr)                           :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = sycl_memcpy_cptr_c(dst, src, size, dir) /= 0
#else
      success = .true.
#endif
    end function

    function sycl_memcpy_mixed_to_device(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                           :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = sycl_memcpy_mixed_to_device_c(dst, src, size, dir) /= 0
#else
      success = .true.
#endif
    end function

    function sycl_memcpy_mixed_to_host(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                           :: src
      integer(kind=C_intptr_t)              :: dst
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = sycl_memcpy_mixed_to_host_c(dst, src, size, dir) /= 0
#else
      success = .true.
#endif
    end function

!    function sycl_memcpy_async_intptr(dst, src, size, dir, syclStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)              :: dst
!      integer(kind=C_intptr_t)              :: src
!      integer(kind=c_intptr_t), intent(in)  :: size
!      integer(kind=C_INT), intent(in)       :: dir
!      integer(kind=c_intptr_t), intent(in)  :: syclStream
!      logical :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_memcpy_async_intptr_c(dst, src, size, dir, syclStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function sycl_memcpy_async_cptr(dst, src, size, dir, syclStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)                           :: dst
!      type(c_ptr)                           :: src
!      integer(kind=c_intptr_t), intent(in)  :: size
!      integer(kind=C_INT), intent(in)       :: dir
!      integer(kind=c_intptr_t), intent(in)  :: syclStream
!      logical :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_memcpy_async_cptr_c(dst, src, size, dir, syclStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function sycl_memcpy_async_mixed_to_device(dst, src, size, dir, syclStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)                           :: dst
!      integer(kind=C_intptr_t)              :: src
!      integer(kind=c_intptr_t), intent(in)  :: size
!      integer(kind=C_INT), intent(in)       :: dir
!      integer(kind=c_intptr_t), intent(in)  :: syclStream
!      logical :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_memcpy_async_mixed_to_device_c(dst, src, size, dir, syclStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function sycl_memcpy_async_mixed_to_host(dst, src, size, dir, syclStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)                           :: src
!      integer(kind=C_intptr_t)              :: dst
!      integer(kind=c_intptr_t), intent(in)  :: size
!      integer(kind=C_INT), intent(in)       :: dir
!      integer(kind=c_intptr_t), intent(in)  :: syclStream
!      logical :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_memcpy_async_mixed_to_host_c(dst, src, size, dir, syclStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function sycl_memcpy2d_intptr(dst, dpitch, src, spitch, width, height , dir) result(success)
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
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_memcpy2d_intptr_c(dst, dpitch, src, spitch, width, height , dir) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function sycl_memcpy2d_cptr(dst, dpitch, src, spitch, width, height , dir) result(success)
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
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_memcpy2d_cptr_c(dst, dpitch, src, spitch, width, height , dir) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function sycl_memcpy2d_async_intptr(dst, dpitch, src, spitch, width, height, dir, syclStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T)           :: dst
!      integer(kind=c_intptr_t), intent(in) :: dpitch
!      integer(kind=C_intptr_T)           :: src
!      integer(kind=c_intptr_t), intent(in) :: spitch
!      integer(kind=c_intptr_t), intent(in) :: width
!      integer(kind=c_intptr_t), intent(in) :: height
!      integer(kind=C_INT), intent(in)    :: dir
!      integer(kind=c_intptr_t), intent(in) :: syclStream
!      logical                            :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_memcpy2d_async_intptr_c(dst, dpitch, src, spitch, width, height, dir, syclStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function sycl_memcpy2d_async_cptr(dst, dpitch, src, spitch, width, height, dir, syclStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)           :: dst
!      integer(kind=c_intptr_t), intent(in) :: dpitch
!      type(c_ptr)           :: src
!      integer(kind=c_intptr_t), intent(in) :: spitch
!      integer(kind=c_intptr_t), intent(in) :: width
!      integer(kind=c_intptr_t), intent(in) :: height
!      integer(kind=C_INT), intent(in)    :: dir
!      integer(kind=c_intptr_t), intent(in) :: syclStream
!      logical                            :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_memcpy2d_async_cptr_c(dst, dpitch, src, spitch, width, height, dir, syclStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function sycl_host_register(a, size, flag) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)              :: a
!      integer(kind=c_intptr_t), intent(in)  :: size
!      integer(kind=C_INT), intent(in)       :: flag
!      logical :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_host_register_c(a, size, flag) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function sycl_host_unregister(a) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)              :: a
!      logical :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_host_unregister_c(a) /= 0
!#else
!      success = .true.
!#endif
!    end function

    subroutine syclblas_Dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_DOUBLE) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Dgemm_intptr_c(syclblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine syclblas_Dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_DOUBLE) ,value               :: alpha, beta
      type(c_ptr)                     :: a, b, c
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Dgemm_cptr_c(syclblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine syclblas_Dgemm_intptr_cptr_intptr(cta, ctb, m, n, k, &
                                 alpha, a, lda, b, ldb, beta, c, ldc, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_DOUBLE) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, c
      type(c_ptr)                     :: b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Dgemm_intptr_cptr_intptr_c(syclblasHandle, cta, ctb, m, n, k, &
                                                 alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine syclblas_Dcopy_intptr(n, x, incx, y, incy, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Dcopy_intptr_c(syclblasHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine syclblas_Dcopy_cptr(n, x, incx, y, incy, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Dcopy_cptr_c(syclblasHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine syclblas_Dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Dtrmm_intptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine syclblas_Dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE) ,value               :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Dtrmm_cptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine syclblas_Dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Dtrsm_intptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine syclblas_Dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE) ,value               :: alpha
      type(c_ptr)                    :: a, b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Dtrsm_cptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine syclblas_Dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_DOUBLE) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Dgemv_c(syclblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    subroutine syclblas_Sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_FLOAT) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Sgemm_intptr_c(syclblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine syclblas_Sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_FLOAT) ,value               :: alpha, beta
      type(c_ptr)                     :: a, b, c
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Sgemm_cptr_c(syclblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine syclblas_Sgemm_intptr_cptr_intptr(cta, ctb, m, n, k, &
                                 alpha, a, lda, b, ldb, beta, c, ldc, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_FLOAT) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, c
      type(c_ptr)                     :: b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Sgemm_intptr_cptr_intptr_c(syclblasHandle, cta, ctb, m, n, k, &
                                                 alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine syclblas_Scopy_intptr(n, x, incx, y, incy, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Scopy_intptr_c(syclblasHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine syclblas_Scopy_cptr(n, x, incx, y, incy, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Scopy_cptr_c(syclblasHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine syclblas_Strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Strmm_intptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine syclblas_Strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT) ,value               :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Strmm_cptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine syclblas_Strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Strsm_intptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine syclblas_Strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT) ,value               :: alpha
      type(c_ptr)                    :: a, b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Strsm_cptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine syclblas_Sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_FLOAT) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Sgemv_c(syclblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    subroutine syclblas_Zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Zgemm_intptr_c(syclblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine syclblas_Zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha, beta
      type(c_ptr)                     :: a, b, c
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Zgemm_cptr_c(syclblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine syclblas_Zgemm_intptr_cptr_intptr(cta, ctb, m, n, k, &
                                 alpha, a, lda, b, ldb, beta, c, ldc, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, c
      type(c_ptr)                     :: b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Zgemm_intptr_cptr_intptr_c(syclblasHandle, cta, ctb, m, n, k, &
                                                 alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine syclblas_Zcopy_intptr(n, x, incx, y, incy, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Zcopy_intptr_c(syclblasHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine syclblas_Zcopy_cptr(n, x, incx, y, incy, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Zcopy_cptr_c(syclblasHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine syclblas_Ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Ztrmm_intptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine syclblas_Ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Ztrmm_cptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine syclblas_Ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Ztrsm_intptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine syclblas_Ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      type(c_ptr)                    :: a, b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Ztrsm_cptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine syclblas_Zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Zgemv_c(syclblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    subroutine syclblas_Cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Cgemm_intptr_c(syclblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine syclblas_Cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha, beta
      type(c_ptr)                     :: a, b, c
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Cgemm_cptr_c(syclblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine syclblas_Cgemm_intptr_cptr_intptr(cta, ctb, m, n, k, &
                                 alpha, a, lda, b, ldb, beta, c, ldc, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, c
      type(c_ptr)                     :: b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Cgemm_intptr_cptr_intptr_c(syclblasHandle, cta, ctb, m, n, k, &
                                                 alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine syclblas_Ccopy_intptr(n, x, incx, y, incy, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Ccopy_intptr_c(syclblasHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine syclblas_Ccopy_cptr(n, x, incx, y, incy, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Ccopy_cptr_c(syclblasHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine syclblas_Ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Ctrmm_intptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine syclblas_Ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Ctrmm_cptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine syclblas_Ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Ctrsm_intptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine syclblas_Ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      type(c_ptr)                    :: a, b
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Ctrsm_cptr_c(syclblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine syclblas_Cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, syclblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=C_intptr_T)        :: syclblasHandle
#ifdef WITH_SYCL_GPU_VERSION
      call syclblas_Cgemv_c(syclblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    function syclblas_pointerModeDevice() result(flag)
      use, intrinsic :: iso_c_binding
      use precision
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_SYCL_GPU_VERSION
      print *,"pointerModeDevice not yet implemented!"
      flag = 1
      stop 1
#else
      flag = 0
#endif
    end function

    function syclblas_pointerModeHost() result(flag)
      use, intrinsic :: iso_c_binding
      use precision
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_SYCL_GPU_VERSION
      print *,"pointerModeHost not yet implemented!"
      flag = 1
      stop 1
#else
      flag = 0
#endif
    end function

    subroutine syclblas_getPointerMode(syclblasHandle, mode)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: mode

#ifdef WITH_SYCL_GPU_VERSION
      print *,"getPointerMode not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_setPointerMode(syclblasHandle, mode)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: mode

#ifdef WITH_SYCL_GPU_VERSION
      print *,"setPointerMode not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Ddot_intptr(syclblasHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx, incy
      integer(kind=c_intptr_t) :: x, y, result

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}DOT not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Ddot_cptr(syclblasHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx, incy
      type(c_ptr)              :: x, y, result

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}DOT not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Dscal_intptr(syclblasHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx
      real(kind=C_DOUBLE) ,value               :: alpha
      integer(kind=c_intptr_t) :: x

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}SCAL not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Dscal_cptr(syclblasHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx
      real(kind=C_DOUBLE) ,value               :: alpha
      type(c_ptr)              :: x

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}SCAL not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Daxpy_intptr(syclblasHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx, incy
      real(kind=C_DOUBLE) ,value               :: alpha
      integer(kind=c_intptr_t) :: x, y

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}AXPY not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Daxpy_cptr(syclblasHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx, incy
      real(kind=C_DOUBLE) ,value               :: alpha
      type(c_ptr)              :: x, y

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}AXPY not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Sdot_intptr(syclblasHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx, incy
      integer(kind=c_intptr_t) :: x, y, result

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}DOT not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Sdot_cptr(syclblasHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx, incy
      type(c_ptr)              :: x, y, result

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}DOT not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Sscal_intptr(syclblasHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx
      real(kind=C_FLOAT) ,value               :: alpha
      integer(kind=c_intptr_t) :: x

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}SCAL not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Sscal_cptr(syclblasHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx
      real(kind=C_FLOAT) ,value               :: alpha
      type(c_ptr)              :: x

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}SCAL not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Saxpy_intptr(syclblasHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx, incy
      real(kind=C_FLOAT) ,value               :: alpha
      integer(kind=c_intptr_t) :: x, y

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}AXPY not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Saxpy_cptr(syclblasHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx, incy
      real(kind=C_FLOAT) ,value               :: alpha
      type(c_ptr)              :: x, y

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}AXPY not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Zdot_intptr(conj, syclblasHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
       character(1,c_char), value   :: conj
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx, incy
      integer(kind=c_intptr_t) :: x, y, result

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}DOT not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Zdot_cptr(conj, syclblasHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
       character(1,c_char), value   :: conj
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx, incy
      type(c_ptr)              :: x, y, result

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}DOT not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Zscal_intptr(syclblasHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      integer(kind=c_intptr_t) :: x

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}SCAL not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Zscal_cptr(syclblasHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      type(c_ptr)              :: x

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}SCAL not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Zaxpy_intptr(syclblasHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx, incy
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      integer(kind=c_intptr_t) :: x, y

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}AXPY not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Zaxpy_cptr(syclblasHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx, incy
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      type(c_ptr)              :: x, y

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}AXPY not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Cdot_intptr(conj, syclblasHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
       character(1,c_char), value   :: conj
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx, incy
      integer(kind=c_intptr_t) :: x, y, result

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}DOT not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Cdot_cptr(conj, syclblasHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
       character(1,c_char), value   :: conj
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx, incy
      type(c_ptr)              :: x, y, result

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}DOT not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Cscal_intptr(syclblasHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      integer(kind=c_intptr_t) :: x

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}SCAL not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Cscal_cptr(syclblasHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      type(c_ptr)              :: x

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}SCAL not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Caxpy_intptr(syclblasHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx, incy
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      integer(kind=c_intptr_t) :: x, y

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}AXPY not yet implemented!"
      stop 1
#endif
    end subroutine

    subroutine syclblas_Caxpy_cptr(syclblasHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: syclblasHandle
      integer(kind=c_int)      :: length, incx, incy
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      type(c_ptr)              :: x, y

#ifdef WITH_SYCL_GPU_VERSION
      print *,"{X}AXPY not yet implemented!"
      stop 1
#endif
    end subroutine

