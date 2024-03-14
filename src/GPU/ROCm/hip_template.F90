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


  integer(kind=ik) :: hipMemcpyHostToDevice
  integer(kind=ik) :: hipMemcpyDeviceToHost
  integer(kind=ik) :: hipMemcpyDeviceToDevice
  integer(kind=ik) :: hipHostRegisterDefault
  integer(kind=ik) :: hipHostRegisterPortable
  integer(kind=ik) :: hipHostRegisterMapped

  integer(kind=ik) :: rocblasPointerModeDevice
  integer(kind=ik) :: rocblasPointerModeHost


  interface
    function hip_device_get_attributes_c(value, attribute) result(istat) &
             bind(C, name="hipDeviceGetAttributeFromC")
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_INT), value  :: attribute
      integer(kind=C_INT)         :: value
      integer(kind=C_INT)         :: istat
    end function
  end interface


  interface
    function rocblas_get_version_c(rocblasHandle, version) result(istat) &
             bind(C, name="rocblasGetVersionFromC")
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_T), value  :: rocblasHandle
      integer(kind=C_INT)              :: version
      integer(kind=C_INT)              :: istat
    end function
  end interface


  interface
    function hip_get_last_error_c() result(istat) &
             bind(C, name="hipGetLastErrorFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int)              :: istat
    end function
  end interface

  ! streams

  interface
    function hip_stream_create_c(hipStream) result(istat) &
             bind(C, name="hipStreamCreateFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: hipStream
      integer(kind=C_INT)      :: istat
    end function
  end interface

  interface
    function hip_stream_destroy_c(hipStream) result(istat) &
             bind(C, name="hipStreamDestroyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value :: hipStream
      integer(kind=C_INT)             :: istat
    end function
  end interface

  interface
    function hip_stream_synchronize_explicit_c(hipStream) result(istat) &
             bind(C, name="hipStreamSynchronizeExplicitFromC")
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_T), value  :: hipStream
      integer(kind=C_INT)              :: istat
    end function
  end interface

  interface
    function hip_stream_synchronize_implicit_c() result(istat) &
             bind(C, name="hipStreamSynchronizeImplicitFromC")
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_INT)              :: istat
    end function
  end interface

  interface
    function rocblas_set_stream_c(hipHandle, hipStream) result(istat) &
             bind(C, name="rocblasSetStreamFromC")
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_T), value  :: hipHandle
      integer(kind=C_intptr_T), value  :: hipStream
      integer(kind=C_INT)              :: istat
    end function
  end interface

  interface
    function rocblas_create_c(hipHandle) result(istat) &
             bind(C, name="rocblasCreateFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: hipHandle
      integer(kind=C_INT)      :: istat
    end function
  end interface

  interface
    function rocblas_destroy_c(hipHandle) result(istat) &
             bind(C, name="rocblasDestroyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value :: hipHandle
      integer(kind=C_INT)      :: istat
    end function
  end interface

  ! functions to set and query the GPU devices
  interface
    function hip_setdevice_c(n) result(istat) &
             bind(C, name="hipSetDeviceFromC")

      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT), value    :: n
      integer(kind=C_INT)           :: istat
    end function
  end interface

  interface
    function hip_getdevicecount_c(n) result(istat) &
             bind(C, name="hipGetDeviceCountFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT), intent(out)         :: n
      integer(kind=C_INT)                      :: istat
    end function
  end interface

  interface
    function hip_devicesynchronize_c()result(istat) &
             bind(C,name="hipDeviceSynchronizeFromC")

      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)                       :: istat
    end function
  end interface

  ! functions to copy GPU memory
  interface
    function hip_memcpyDeviceToDevice_c() result(flag) &
             bind(C, name="hipMemcpyDeviceToDeviceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function hip_memcpyHostToDevice_c() result(flag) &
             bind(C, name="hipMemcpyHostToDeviceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function hip_memcpyDeviceToHost_c() result(flag) &
             bind(C, name="hipMemcpyDeviceToHostFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function hip_hostRegisterDefault_c() result(flag) &
             bind(C, name="hipHostRegisterDefaultFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function hip_hostRegisterPortable_c() result(flag) &
             bind(C, name="hipHostRegisterPortableFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function hip_hostRegisterMapped_c() result(flag) &
             bind(C, name="hipHostRegisterMappedFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function hip_memcpy_intptr_c(dst, src, size, dir) result(istat) &
             bind(C, name="hipMemcpyFromC")
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
    function hip_memcpy_cptr_c(dst, src, size, dir) result(istat) &
             bind(C, name="hipMemcpyFromC")
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
    function hip_memcpy_mixed_to_device_c(dst, src, size, dir) result(istat) &
             bind(C, name="hipMemcpyFromC")
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
    function hip_memcpy_mixed_to_host_c(dst, src, size, dir) result(istat) &
             bind(C, name="hipMemcpyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                           :: src
      integer(kind=c_intptr_t), value              :: dst
      integer(kind=c_intptr_t), intent(in), value  :: size
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function hip_memcpy_async_intptr_c(dst, src, size, dir, hipStream) result(istat) &
             bind(C, name="hipMemcpyAsyncFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t), value              :: dst
      integer(kind=C_intptr_t), value              :: src
      integer(kind=c_intptr_t), intent(in), value  :: size
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=c_intptr_t), value              :: hipStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function hip_memcpy_async_cptr_c(dst, src, size, dir, hipStream) result(istat) &
             bind(C, name="hipMemcpyAsyncFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                           :: dst
      type(c_ptr), value                           :: src
      integer(kind=c_intptr_t), intent(in), value  :: size
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=c_intptr_t), value              :: hipStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function hip_memcpy_async_mixed_to_device_c(dst, src, size, dir, hipStream) result(istat) &
             bind(C, name="hipMemcpyAsyncFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                           :: dst
      integer(kind=C_intptr_t), value              :: src
      integer(kind=c_intptr_t), intent(in), value  :: size
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=c_intptr_t), value              :: hipStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function hip_memcpy_async_mixed_to_host_c(dst, src, size, dir, hipStream) result(istat) &
             bind(C, name="hipMemcpyAsyncFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                           :: src
      integer(kind=C_intptr_t), value              :: dst
      integer(kind=c_intptr_t), intent(in), value  :: size
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=c_intptr_t), value              :: hipStream
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function hip_memcpy2d_intptr_c(dst, dpitch, src, spitch, width, height , dir) result(istat) &
             bind(C, name="hipMemcpy2dFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value                :: dst
      integer(kind=c_intptr_t), intent(in), value    :: dpitch
      integer(kind=C_intptr_T), value                :: src
      integer(kind=c_intptr_t), intent(in), value    :: spitch
      integer(kind=c_intptr_t), intent(in), value    :: width
      integer(kind=c_intptr_t), intent(in), value    :: height
      integer(kind=C_INT), intent(in), value         :: dir
      integer(kind=C_INT)                            :: istat
    end function
  end interface

  interface
    function hip_memcpy2d_cptr_c(dst, dpitch, src, spitch, width, height , dir) result(istat) &
             bind(C, name="hipMemcpy2dFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                :: dst
      integer(kind=c_intptr_t), intent(in), value    :: dpitch
      type(c_ptr), value                :: src
      integer(kind=c_intptr_t), intent(in), value    :: spitch
      integer(kind=c_intptr_t), intent(in), value    :: width
      integer(kind=c_intptr_t), intent(in), value    :: height
      integer(kind=C_INT), intent(in), value         :: dir
      integer(kind=C_INT)                            :: istat
    end function
  end interface

  interface
    function hip_memcpy2d_async_intptr_c(dst, dpitch, src, spitch, width, height, dir, hipStream) result(istat) &
             bind(C, name="hipMemcpy2dAsyncFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value                :: dst
      integer(kind=c_intptr_t), intent(in), value    :: dpitch
      integer(kind=C_intptr_T), value                :: src
      integer(kind=c_intptr_t), intent(in), value    :: spitch
      integer(kind=c_intptr_t), intent(in), value    :: width
      integer(kind=c_intptr_t), intent(in), value    :: height
      integer(kind=C_INT), intent(in), value         :: dir
      integer(kind=c_intptr_t), value                :: hipStream
      integer(kind=C_INT)                            :: istat
    end function
  end interface

  interface
    function hip_memcpy2d_async_cptr_c(dst, dpitch, src, spitch, width, height, dir, hipStream) result(istat) &
             bind(C, name="hipMemcpy2dAsyncFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value                :: dst
      integer(kind=c_intptr_t), intent(in), value    :: dpitch
      type(c_ptr), value                :: src
      integer(kind=c_intptr_t), intent(in), value    :: spitch
      integer(kind=c_intptr_t), intent(in), value    :: width
      integer(kind=c_intptr_t), intent(in), value    :: height
      integer(kind=C_INT), intent(in), value         :: dir
      integer(kind=c_intptr_t), value                :: hipStream
      integer(kind=C_INT)                            :: istat
    end function
  end interface

  interface
    function hip_host_register_c(a, size, flag) result(istat) &
             bind(C, name="hipHostRegisterFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t), value              :: a
      integer(kind=c_intptr_t), intent(in), value  :: size
      integer(kind=C_INT), intent(in), value       :: flag
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface
    function hip_host_unregister_c(a) result(istat) &
             bind(C, name="hipHostUnregisterFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t), value              :: a
      integer(kind=C_INT)                          :: istat
    end function
  end interface

  interface hip_free
    module procedure hip_free_intptr
    module procedure hip_free_cptr
  end interface

  interface
    function hip_free_intptr_c(a) result(istat) &
             bind(C, name="hipFreeFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value  :: a
      integer(kind=C_INT)              :: istat
    end function
  end interface

  interface
    function hip_free_cptr_c(a) result(istat) &
             bind(C, name="hipFreeFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value  :: a
      integer(kind=C_INT)              :: istat
    end function
  end interface

  interface hip_memcpy
    module procedure hip_memcpy_intptr
    module procedure hip_memcpy_cptr
    module procedure hip_memcpy_mixed_to_device
    module procedure hip_memcpy_mixed_to_host
  end interface

  interface hip_memcpy_async
    module procedure hip_memcpy_async_intptr
    module procedure hip_memcpy_async_cptr
    module procedure hip_memcpy_async_mixed_to_device
    module procedure hip_memcpy_async_mixed_to_host
  end interface

  interface hip_malloc
    module procedure hip_malloc_intptr
    module procedure hip_malloc_cptr
  end interface

  interface
    function hip_malloc_intptr_c(a, width_height) result(istat) &
             bind(C, name="hipMallocFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      ! no value since **pointer
      integer(kind=C_intptr_T)                    :: a
      integer(kind=c_intptr_t), intent(in), value :: width_height
      integer(kind=C_INT)                         :: istat
    end function
  end interface

  interface
    function hip_malloc_cptr_c(a, width_height) result(istat) &
             bind(C, name="hipMallocFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      ! no value since **pointer
      type(c_ptr)                                 :: a
      integer(kind=c_intptr_t), intent(in), value :: width_height
      integer(kind=C_INT)                         :: istat
    end function
  end interface

  interface hip_free_host
    module procedure hip_free_host_intptr
    module procedure hip_free_host_cptr
  end interface
  interface
    function hip_free_host_intptr_c(a) result(istat) &
             bind(C, name="hipFreeHostFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value  :: a
      integer(kind=C_INT)              :: istat
    end function
  end interface

  interface
    function hip_free_host_cptr_c(a) result(istat) &
             bind(C, name="hipFreeHostFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), value               :: a
      integer(kind=C_INT)              :: istat
    end function
  end interface

  interface hip_malloc_host
    module procedure hip_malloc_host_intptr
    module procedure hip_malloc_host_cptr
  end interface
  interface
    function hip_malloc_host_intptr_c(a, width_height) result(istat) &
             bind(C, name="hipMallocHostFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t)                    :: a
      integer(kind=c_intptr_t), intent(in), value :: width_height
      integer(kind=C_INT)                         :: istat
    end function
  end interface

  interface
    function hip_malloc_host_cptr_c(a, width_height) result(istat) &
             bind(C, name="hipMallocHostFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                    :: a
      integer(kind=c_intptr_t), intent(in), value :: width_height
      integer(kind=C_INT)                         :: istat
    end function
  end interface

  interface
    function hip_memset_c(a, val, size) result(istat) &
             bind(C, name="hipMemsetFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value            :: a
      integer(kind=C_INT), value                 :: val
      integer(kind=c_intptr_t), intent(in), value  :: size
      integer(kind=C_INT)                        :: istat
    end function
  end interface

  interface
    function hip_memset_async_c(a, val, size, hipStream) result(istat) &
             bind(C, name="hipMemsetAsyncFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value            :: a
      integer(kind=C_INT), value                 :: val
      integer(kind=c_intptr_t), intent(in), value  :: size
      integer(kind=C_INT)                        :: istat
      integer(kind=c_intptr_t), value            :: hipStream
    end function
  end interface

  interface rocblas_Dgemm
    module procedure rocblas_Dgemm_intptr
    module procedure rocblas_Dgemm_cptr
    module procedure rocblas_Dgemm_intptr_cptr_intptr
  end interface

  interface
    subroutine rocblas_Dgemm_intptr_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="rocblasDgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_DOUBLE) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Dgemm_cptr_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="rocblasDgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_DOUBLE) ,value               :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Dgemm_intptr_cptr_intptr_c(rocblasHandle, cta, ctb, m, n, k, &
                                                                      alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="rocblasDgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m, n, k
      integer(kind=C_INT), intent(in), value  :: lda, ldb, ldc
      real(kind=C_DOUBLE) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, c
      type(c_ptr), value                      :: b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface


  interface rocblas_Dcopy
    module procedure rocblas_Dcopy_intptr
    module procedure rocblas_Dcopy_cptr
  end interface

  interface
    subroutine rocblas_Dcopy_intptr_c(rocblasHandle, n, x, incx, y, incy) &
                              bind(C,name="rocblasDcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Dcopy_cptr_c(rocblasHandle, n, x, incx, y, incy) &
                              bind(C,name="rocblasDcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface


  interface rocblas_Dtrmm
    module procedure rocblas_Dtrmm_intptr
    module procedure rocblas_Dtrmm_cptr
  end interface

  interface
    subroutine rocblas_Dtrmm_intptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="rocblasDtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Dtrmm_cptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="rocblasDtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface


  interface rocblas_Dtrsm
    module procedure rocblas_Dtrsm_intptr
    module procedure rocblas_Dtrsm_cptr
  end interface

  interface
    subroutine rocblas_Dtrsm_intptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="rocblasDtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Dtrsm_cptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="rocblasDtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Dgemv_c(rocblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name="rocblasDgemv_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      real(kind=C_DOUBLE) , value              :: alpha, beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface rocblas_Sgemm
    module procedure rocblas_Sgemm_intptr
    module procedure rocblas_Sgemm_cptr
    module procedure rocblas_Sgemm_intptr_cptr_intptr
  end interface

  interface
    subroutine rocblas_Sgemm_intptr_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="rocblasSgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_FLOAT) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Sgemm_cptr_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="rocblasSgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_FLOAT) ,value               :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Sgemm_intptr_cptr_intptr_c(rocblasHandle, cta, ctb, m, n, k, &
                                                                      alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="rocblasSgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m, n, k
      integer(kind=C_INT), intent(in), value  :: lda, ldb, ldc
      real(kind=C_FLOAT) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, c
      type(c_ptr), value                      :: b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface


  interface rocblas_Scopy
    module procedure rocblas_Scopy_intptr
    module procedure rocblas_Scopy_cptr
  end interface

  interface
    subroutine rocblas_Scopy_intptr_c(rocblasHandle, n, x, incx, y, incy) &
                              bind(C,name="rocblasScopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Scopy_cptr_c(rocblasHandle, n, x, incx, y, incy) &
                              bind(C,name="rocblasScopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface


  interface rocblas_Strmm
    module procedure rocblas_Strmm_intptr
    module procedure rocblas_Strmm_cptr
  end interface

  interface
    subroutine rocblas_Strmm_intptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="rocblasStrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Strmm_cptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="rocblasStrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface


  interface rocblas_Strsm
    module procedure rocblas_Strsm_intptr
    module procedure rocblas_Strsm_cptr
  end interface

  interface
    subroutine rocblas_Strsm_intptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="rocblasStrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Strsm_cptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="rocblasStrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Sgemv_c(rocblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name="rocblasSgemv_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      real(kind=C_FLOAT) , value              :: alpha, beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface rocblas_Zgemm
    module procedure rocblas_Zgemm_intptr
    module procedure rocblas_Zgemm_cptr
    module procedure rocblas_Zgemm_intptr_cptr_intptr
  end interface

  interface
    subroutine rocblas_Zgemm_intptr_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="rocblasZgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Zgemm_cptr_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="rocblasZgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Zgemm_intptr_cptr_intptr_c(rocblasHandle, cta, ctb, m, n, k, &
                                                                      alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="rocblasZgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m, n, k
      integer(kind=C_INT), intent(in), value  :: lda, ldb, ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, c
      type(c_ptr), value                      :: b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface


  interface rocblas_Zcopy
    module procedure rocblas_Zcopy_intptr
    module procedure rocblas_Zcopy_cptr
  end interface

  interface
    subroutine rocblas_Zcopy_intptr_c(rocblasHandle, n, x, incx, y, incy) &
                              bind(C,name="rocblasZcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Zcopy_cptr_c(rocblasHandle, n, x, incx, y, incy) &
                              bind(C,name="rocblasZcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface


  interface rocblas_Ztrmm
    module procedure rocblas_Ztrmm_intptr
    module procedure rocblas_Ztrmm_cptr
  end interface

  interface
    subroutine rocblas_Ztrmm_intptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="rocblasZtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Ztrmm_cptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="rocblasZtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface


  interface rocblas_Ztrsm
    module procedure rocblas_Ztrsm_intptr
    module procedure rocblas_Ztrsm_cptr
  end interface

  interface
    subroutine rocblas_Ztrsm_intptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="rocblasZtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Ztrsm_cptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="rocblasZtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Zgemv_c(rocblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name="rocblasZgemv_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha, beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface rocblas_Cgemm
    module procedure rocblas_Cgemm_intptr
    module procedure rocblas_Cgemm_cptr
    module procedure rocblas_Cgemm_intptr_cptr_intptr
  end interface

  interface
    subroutine rocblas_Cgemm_intptr_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="rocblasCgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Cgemm_cptr_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="rocblasCgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Cgemm_intptr_cptr_intptr_c(rocblasHandle, cta, ctb, m, n, k, &
                                                                      alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="rocblasCgemm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m, n, k
      integer(kind=C_INT), intent(in), value  :: lda, ldb, ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, c
      type(c_ptr), value                      :: b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface


  interface rocblas_Ccopy
    module procedure rocblas_Ccopy_intptr
    module procedure rocblas_Ccopy_cptr
  end interface

  interface
    subroutine rocblas_Ccopy_intptr_c(rocblasHandle, n, x, incx, y, incy) &
                              bind(C,name="rocblasCcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Ccopy_cptr_c(rocblasHandle, n, x, incx, y, incy) &
                              bind(C,name="rocblasCcopy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface


  interface rocblas_Ctrmm
    module procedure rocblas_Ctrmm_intptr
    module procedure rocblas_Ctrmm_cptr
  end interface

  interface
    subroutine rocblas_Ctrmm_intptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="rocblasCtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Ctrmm_cptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="rocblasCtrmm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface


  interface rocblas_Ctrsm
    module procedure rocblas_Ctrsm_intptr
    module procedure rocblas_Ctrsm_cptr
  end interface

  interface
    subroutine rocblas_Ctrsm_intptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="rocblasCtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Ctrsm_cptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="rocblasCtrsm_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: rocblasHandle
    end subroutine
  end interface

  interface
    subroutine rocblas_Cgemv_c(rocblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name="rocblasCgemv_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha, beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: rocblasHandle
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

  interface
    function rocblas_pointerModeDevice_c() result(flag) &
               bind(C, name="rocblasPointerModeDeviceFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function rocblas_pointerModeHost_c() result(flag) &
               bind(C, name="rocblasPointerModeHostFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    subroutine rocblas_getPointerMode_c(rocblasHandle, mode) &
               bind(C, name="rocblasGetPointerModeFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value   :: rocblasHandle
      integer(kind=c_int)               :: mode
    end subroutine
  end interface

  interface
    subroutine rocblas_setPointerMode_c(rocblasHandle, mode) &
               bind(C, name="rocblasSetPointerModeFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T),value    :: rocblasHandle
      integer(kind=c_int), value        :: mode
    end subroutine
  end interface


  interface rocblas_Ddot
    module procedure rocblas_Ddot_intptr
    module procedure rocblas_Ddot_cptr
  end interface

  interface
    subroutine rocblas_Ddot_intptr_c(rocblasHandle, length, x, incx, y, incy, result) &
               bind(C, name="rocblasDdot_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT), value              :: length, incx, incy
      integer(kind=C_intptr_T), value         :: x, y, result
    end subroutine
  end interface

  interface
    subroutine rocblas_Ddot_cptr_c(rocblasHandle, length, x, incx, y, incy, result) &
               bind(C, name="rocblasDdot_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT), value              :: length, incx, incy
      type(c_ptr), value                      :: x, y, result
    end subroutine
  end interface


  interface rocblas_Dscal
    module procedure rocblas_Dscal_intptr
    module procedure rocblas_Dscal_cptr
  end interface

  interface
    subroutine rocblas_Dscal_intptr_c(rocblasHandle, length, alpha, x, incx) &
               bind(C, name="rocblasDscal_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT),value               :: length, incx
      real(kind=C_DOUBLE) ,value                :: alpha
      integer(kind=C_intptr_T), value         :: x
    end subroutine
  end interface

  interface
    subroutine rocblas_Dscal_cptr_c(rocblasHandle, length, alpha, x, incx) &
               bind(C, name="rocblasDscal_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT),value               :: length, incx
      real(kind=C_DOUBLE) ,value                :: alpha
      type(c_ptr), value                      :: x
    end subroutine
  end interface


  interface rocblas_Daxpy
    module procedure rocblas_Daxpy_intptr
    module procedure rocblas_Daxpy_cptr
  end interface

  interface
    subroutine rocblas_Daxpy_intptr_c(rocblasHandle, length, alpha, x, incx, y, incy) &
               bind(C, name="rocblasDaxpy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT),value               :: length, incx, incy
      real(kind=C_DOUBLE) ,value                :: alpha
      integer(kind=C_intptr_T), value         :: x, y
    end subroutine
  end interface

  interface
    subroutine rocblas_Daxpy_cptr_c(rocblasHandle, length, alpha, x, incx, y, incy) &
               bind(C, name="rocblasDaxpy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT),value               :: length, incx, incy
      real(kind=C_DOUBLE) ,value                :: alpha
      type(c_ptr), value                      :: x, y
    end subroutine
  end interface

  interface rocblas_Sdot
    module procedure rocblas_Sdot_intptr
    module procedure rocblas_Sdot_cptr
  end interface

  interface
    subroutine rocblas_Sdot_intptr_c(rocblasHandle, length, x, incx, y, incy, result) &
               bind(C, name="rocblasSdot_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT), value              :: length, incx, incy
      integer(kind=C_intptr_T), value         :: x, y, result
    end subroutine
  end interface

  interface
    subroutine rocblas_Sdot_cptr_c(rocblasHandle, length, x, incx, y, incy, result) &
               bind(C, name="rocblasSdot_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT), value              :: length, incx, incy
      type(c_ptr), value                      :: x, y, result
    end subroutine
  end interface


  interface rocblas_Sscal
    module procedure rocblas_Sscal_intptr
    module procedure rocblas_Sscal_cptr
  end interface

  interface
    subroutine rocblas_Sscal_intptr_c(rocblasHandle, length, alpha, x, incx) &
               bind(C, name="rocblasSscal_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT),value               :: length, incx
      real(kind=C_FLOAT) ,value                :: alpha
      integer(kind=C_intptr_T), value         :: x
    end subroutine
  end interface

  interface
    subroutine rocblas_Sscal_cptr_c(rocblasHandle, length, alpha, x, incx) &
               bind(C, name="rocblasSscal_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT),value               :: length, incx
      real(kind=C_FLOAT) ,value                :: alpha
      type(c_ptr), value                      :: x
    end subroutine
  end interface


  interface rocblas_Saxpy
    module procedure rocblas_Saxpy_intptr
    module procedure rocblas_Saxpy_cptr
  end interface

  interface
    subroutine rocblas_Saxpy_intptr_c(rocblasHandle, length, alpha, x, incx, y, incy) &
               bind(C, name="rocblasSaxpy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT),value               :: length, incx, incy
      real(kind=C_FLOAT) ,value                :: alpha
      integer(kind=C_intptr_T), value         :: x, y
    end subroutine
  end interface

  interface
    subroutine rocblas_Saxpy_cptr_c(rocblasHandle, length, alpha, x, incx, y, incy) &
               bind(C, name="rocblasSaxpy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT),value               :: length, incx, incy
      real(kind=C_FLOAT) ,value                :: alpha
      type(c_ptr), value                      :: x, y
    end subroutine
  end interface

  interface rocblas_Zdot
    module procedure rocblas_Zdot_intptr
    module procedure rocblas_Zdot_cptr
  end interface

  interface
    subroutine rocblas_Zdot_intptr_c(conj, rocblasHandle, length, x, incx, y, incy, result) &
               bind(C, name="rocblasZdot_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: conj
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT), value              :: length, incx, incy
      integer(kind=C_intptr_T), value         :: x, y, result
    end subroutine
  end interface

  interface
    subroutine rocblas_Zdot_cptr_c(conj, rocblasHandle, length, x, incx, y, incy, result) &
               bind(C, name="rocblasZdot_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: conj
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT), value              :: length, incx, incy
      type(c_ptr), value                      :: x, y, result
    end subroutine
  end interface


  interface rocblas_Zscal
    module procedure rocblas_Zscal_intptr
    module procedure rocblas_Zscal_cptr
  end interface

  interface
    subroutine rocblas_Zscal_intptr_c(rocblasHandle, length, alpha, x, incx) &
               bind(C, name="rocblasZscal_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT),value               :: length, incx
      complex(kind=C_DOUBLE_COMPLEX) ,value                :: alpha
      integer(kind=C_intptr_T), value         :: x
    end subroutine
  end interface

  interface
    subroutine rocblas_Zscal_cptr_c(rocblasHandle, length, alpha, x, incx) &
               bind(C, name="rocblasZscal_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT),value               :: length, incx
      complex(kind=C_DOUBLE_COMPLEX) ,value                :: alpha
      type(c_ptr), value                      :: x
    end subroutine
  end interface


  interface rocblas_Zaxpy
    module procedure rocblas_Zaxpy_intptr
    module procedure rocblas_Zaxpy_cptr
  end interface

  interface
    subroutine rocblas_Zaxpy_intptr_c(rocblasHandle, length, alpha, x, incx, y, incy) &
               bind(C, name="rocblasZaxpy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT),value               :: length, incx, incy
      complex(kind=C_DOUBLE_COMPLEX) ,value                :: alpha
      integer(kind=C_intptr_T), value         :: x, y
    end subroutine
  end interface

  interface
    subroutine rocblas_Zaxpy_cptr_c(rocblasHandle, length, alpha, x, incx, y, incy) &
               bind(C, name="rocblasZaxpy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT),value               :: length, incx, incy
      complex(kind=C_DOUBLE_COMPLEX) ,value                :: alpha
      type(c_ptr), value                      :: x, y
    end subroutine
  end interface

  interface rocblas_Cdot
    module procedure rocblas_Cdot_intptr
    module procedure rocblas_Cdot_cptr
  end interface

  interface
    subroutine rocblas_Cdot_intptr_c(conj, rocblasHandle, length, x, incx, y, incy, result) &
               bind(C, name="rocblasCdot_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: conj
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT), value              :: length, incx, incy
      integer(kind=C_intptr_T), value         :: x, y, result
    end subroutine
  end interface

  interface
    subroutine rocblas_Cdot_cptr_c(conj, rocblasHandle, length, x, incx, y, incy, result) &
               bind(C, name="rocblasCdot_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: conj
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT), value              :: length, incx, incy
      type(c_ptr), value                      :: x, y, result
    end subroutine
  end interface


  interface rocblas_Cscal
    module procedure rocblas_Cscal_intptr
    module procedure rocblas_Cscal_cptr
  end interface

  interface
    subroutine rocblas_Cscal_intptr_c(rocblasHandle, length, alpha, x, incx) &
               bind(C, name="rocblasCscal_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT),value               :: length, incx
      complex(kind=C_FLOAT_COMPLEX) ,value                :: alpha
      integer(kind=C_intptr_T), value         :: x
    end subroutine
  end interface

  interface
    subroutine rocblas_Cscal_cptr_c(rocblasHandle, length, alpha, x, incx) &
               bind(C, name="rocblasCscal_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT),value               :: length, incx
      complex(kind=C_FLOAT_COMPLEX) ,value                :: alpha
      type(c_ptr), value                      :: x
    end subroutine
  end interface


  interface rocblas_Caxpy
    module procedure rocblas_Caxpy_intptr
    module procedure rocblas_Caxpy_cptr
  end interface

  interface
    subroutine rocblas_Caxpy_intptr_c(rocblasHandle, length, alpha, x, incx, y, incy) &
               bind(C, name="rocblasCaxpy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT),value               :: length, incx, incy
      complex(kind=C_FLOAT_COMPLEX) ,value                :: alpha
      integer(kind=C_intptr_T), value         :: x, y
    end subroutine
  end interface

  interface
    subroutine rocblas_Caxpy_cptr_c(rocblasHandle, length, alpha, x, incx, y, incy) &
               bind(C, name="rocblasCaxpy_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value         :: rocblasHandle
      integer(kind=C_INT),value               :: length, incx, incy
      complex(kind=C_FLOAT_COMPLEX) ,value                :: alpha
      type(c_ptr), value                      :: x, y
    end subroutine
  end interface

  contains

    function hip_device_get_attributes(value, attribute) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)                       :: value, attribute
      logical                                   :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_device_get_attributes_c(value, attribute) /= 0
#else
      success = .true.
#endif
    end function

    function rocblas_get_version(rocblasHandle, version) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: rocblasHandle
      integer(kind=C_INT)                       :: version
      logical                                   :: success
#ifdef WITH_AMD_GPU_VERSION
      success = rocblas_get_version_c(rocblasHandle, version) /= 0
#else
      success = .true.
#endif
    end function

    function hip_get_last_error() result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      logical                                   :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_get_last_error_c() /= 0
#else
      success = .true.
#endif
    end function

    function hip_stream_create(hipStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: hipStream
      logical                                   :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_stream_create_c(hipStream) /= 0
#else
      success = .true.
#endif
    end function

    function hip_stream_destroy(hipStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: hipStream
      logical                                   :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_stream_destroy_c(hipStream) /= 0
#else
      success = .true.
#endif
    end function

    function rocblas_set_stream(rocblasHandle, hipStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: rocblasHandle
      integer(kind=C_intptr_t)                  :: hipStream
      logical                                   :: success
#ifdef WITH_AMD_GPU_VERSION
      success = rocblas_set_stream_c(rocblasHandle, hipStream) /= 0
#else
      success = .true.
#endif
    end function


    function hip_stream_synchronize(hipStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t), optional        :: hipStream
      logical                                   :: success
      if (present(hipStream)) then
#ifdef WITH_AMD_GPU_VERSION
        success = hip_stream_synchronize_explicit_c(hipStream) /= 0
#else
        success = .true.
#endif
      else
#ifdef WITH_AMD_GPU_VERSION
        success = hip_stream_synchronize_implicit_c() /= 0
#else
        success = .true.
#endif
      endif
    end function

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

    function rocblas_create(rocblasHandle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: rocblasHandle
      logical                                   :: success
#ifdef WITH_AMD_GPU_VERSION
      success = rocblas_create_c(rocblasHandle) /= 0
#else
      success = .true.
#endif
    end function

    function rocblas_destroy(rocblasHandle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)   :: rocblasHandle
      logical                    :: success
#ifdef WITH_AMD_GPU_VERSION
      success = rocblas_destroy_c(rocblasHandle) /= 0
#else
      success = .true.
#endif
    end function

    function hip_setdevice(n) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik), intent(in)  :: n
      logical                       :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_setdevice_c(int(n,kind=c_int)) /= 0
#else
      success = .true.
#endif
    end function

    function hip_getdevicecount(n) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik)     :: n
      integer(kind=c_int)  :: nCasted
      logical              :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_getdevicecount_c(nCasted) /=0
      n = int(nCasted)
#else
      success = .true.
      n = 0
#endif
    end function

    function hip_devicesynchronize()result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      logical :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_devicesynchronize_c() /=0
#else
      success = .true.
#endif
    end function

    function hip_malloc_intptr(a, width_height) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t)                  :: a
      integer(kind=c_intptr_t), intent(in)      :: width_height
      logical                                   :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_malloc_intptr_c(a, width_height) /= 0
#else
      success = .true.
#endif
    end function

    function hip_malloc_cptr(a, width_height) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                               :: a
      integer(kind=c_intptr_t), intent(in)      :: width_height
      logical                                   :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_malloc_cptr_c(a, width_height) /= 0
#else
      success = .true.
#endif
    end function

    function hip_free_intptr(a) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: a
      logical                  :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_free_intptr_c(a) /= 0
#else
      success = .true.
#endif
    end function

    function hip_free_cptr(a) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr) :: a
      logical                  :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_free_cptr_c(a) /= 0
#else
      success = .true.
#endif
    end function

    function hip_malloc_host_intptr(a, width_height) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t)                  :: a
      integer(kind=c_intptr_t), intent(in)      :: width_height
      logical                                   :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_malloc_host_intptr_c(a, width_height) /= 0
#else
      success = .true.
#endif
    end function

    function hip_malloc_host_cptr(a, width_height) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                               :: a
      integer(kind=c_intptr_t), intent(in)      :: width_height
      logical                                   :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_malloc_host_cptr_c(a, width_height) /= 0
#else
      success = .true.
#endif
    end function

    function hip_free_host_intptr(a) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: a
      logical                  :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_free_host_intptr_c(a) /= 0
#else
      success = .true.
#endif
    end function

    function hip_free_host_cptr(a) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                   :: a
      logical                  :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_free_host_cptr_c(a) /= 0
#else
      success = .true.
#endif
    end function

    function hip_memset(a, val, size) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t)                :: a
      integer(kind=ik)                        :: val
      integer(kind=c_intptr_t), intent(in)    :: size
      integer(kind=C_INT)                     :: istat
      logical :: success
#ifdef WITH_AMD_GPU_VERSION
      success= hip_memset_c(a, int(val,kind=c_int), int(size,kind=c_intptr_t)) /=0
#else
      success = .true.
#endif
    end function

    function hip_memset_async(a, val, size, hipStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t)                :: a
      integer(kind=ik)                        :: val
      integer(kind=c_intptr_t), intent(in)    :: size
      integer(kind=C_INT)                     :: istat
      integer(kind=c_intptr_t)                :: hipStream
      logical :: success

#ifdef WITH_AMD_GPU_VERSION
      success= hip_memset_async_c(a, int(val,kind=c_int), int(size,kind=c_intptr_t), hipStream) /=0
#else
      success = .true.
#endif
    end function

    function hip_memcpyDeviceToDevice() result(flag)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_GPU_VERSION
      flag = int(hip_memcpyDeviceToDevice_c())
#else
      flag = 0
#endif
    end function

    function hip_memcpyHostToDevice() result(flag)
      use, intrinsic :: iso_c_binding
      use precision
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_GPU_VERSION
      flag = int(hip_memcpyHostToDevice_c())
#else
      flag = 0
#endif
    end function

    function hip_memcpyDeviceToHost() result(flag)
      use, intrinsic :: iso_c_binding
      use precision
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_GPU_VERSION
      flag = int( hip_memcpyDeviceToHost_c())
#else
      flag = 0
#endif
    end function

    function hip_hostRegisterDefault() result(flag)
      use, intrinsic :: iso_c_binding
      use precision
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_GPU_VERSION
      flag = int(hip_hostRegisterDefault_c())
#else
      flag = 0
#endif
    end function

    function hip_hostRegisterPortable() result(flag)
      use, intrinsic :: iso_c_binding
      use precision
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_GPU_VERSION
      flag = int(hip_hostRegisterPortable_c())
#else
      flag = 0
#endif
    end function

    function hip_hostRegisterMapped() result(flag)
      use, intrinsic :: iso_c_binding
      use precision
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_GPU_VERSION
      flag = int(hip_hostRegisterMapped_c())
#else
      flag = 0
#endif
    end function

    function hip_memcpy_intptr(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)              :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_memcpy_intptr_c(dst, src, size, dir) /= 0
#else
      success = .true.
#endif
    end function

    function hip_memcpy_cptr(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                           :: dst
      type(c_ptr)                           :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_memcpy_cptr_c(dst, src, size, dir) /= 0
#else
      success = .true.
#endif
    end function

    function hip_memcpy_mixed_to_device(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                           :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_memcpy_mixed_to_device_c(dst, src, size, dir) /= 0
#else
      success = .true.
#endif
    end function

    function hip_memcpy_mixed_to_host(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                           :: src
      integer(kind=C_intptr_t)              :: dst
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_memcpy_mixed_to_host_c(dst, src, size, dir) /= 0
#else
      success = .true.
#endif
    end function

    function hip_memcpy_async_intptr(dst, src, size, dir, hipStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)              :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      integer(kind=c_intptr_t), intent(in)  :: hipStream
      logical :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_memcpy_async_intptr_c(dst, src, size, dir, hipStream) /= 0
#else
      success = .true.
#endif
    end function

    function hip_memcpy_async_cptr(dst, src, size, dir, hipStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                           :: dst
      type(c_ptr)                           :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      integer(kind=c_intptr_t), intent(in)  :: hipStream
      logical :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_memcpy_async_cptr_c(dst, src, size, dir, hipStream) /= 0
#else
      success = .true.
#endif
    end function

    function hip_memcpy_async_mixed_to_device(dst, src, size, dir, hipStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                           :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      integer(kind=c_intptr_t), intent(in)  :: hipStream
      logical :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_memcpy_async_mixed_to_device_c(dst, src, size, dir, hipStream) /= 0
#else
      success = .true.
#endif
    end function

    function hip_memcpy_async_mixed_to_host(dst, src, size, dir, hipStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)                           :: src
      integer(kind=C_intptr_t)              :: dst
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      integer(kind=c_intptr_t), intent(in)  :: hipStream
      logical :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_memcpy_async_mixed_to_host_c(dst, src, size, dir, hipStream) /= 0
#else
      success = .true.
#endif
    end function

    function hip_memcpy2d_intptr(dst, dpitch, src, spitch, width, height , dir) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T)           :: dst
      integer(kind=c_intptr_t), intent(in) :: dpitch
      integer(kind=C_intptr_T)           :: src
      integer(kind=c_intptr_t), intent(in) :: spitch
      integer(kind=c_intptr_t), intent(in) :: width
      integer(kind=c_intptr_t), intent(in) :: height
      integer(kind=C_INT), intent(in)    :: dir
      logical                            :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_memcpy2d_intptr_c(dst, dpitch, src, spitch, width, height , dir) /= 0
#else
      success = .true.
#endif
    end function

    function hip_memcpy2d_cptr(dst, dpitch, src, spitch, width, height , dir) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)           :: dst
      integer(kind=c_intptr_t), intent(in) :: dpitch
      type(c_ptr)           :: src
      integer(kind=c_intptr_t), intent(in) :: spitch
      integer(kind=c_intptr_t), intent(in) :: width
      integer(kind=c_intptr_t), intent(in) :: height
      integer(kind=C_INT), intent(in)    :: dir
      logical                            :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_memcpy2d_cptr_c(dst, dpitch, src, spitch, width, height , dir) /= 0
#else
      success = .true.
#endif
    end function

    function hip_memcpy2d_async_intptr(dst, dpitch, src, spitch, width, height, dir, hipStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T)           :: dst
      integer(kind=c_intptr_t), intent(in) :: dpitch
      integer(kind=C_intptr_T)           :: src
      integer(kind=c_intptr_t), intent(in) :: spitch
      integer(kind=c_intptr_t), intent(in) :: width
      integer(kind=c_intptr_t), intent(in) :: height
      integer(kind=C_INT), intent(in)    :: dir
      integer(kind=c_intptr_t), intent(in) :: hipStream
      logical                            :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_memcpy2d_async_intptr_c(dst, dpitch, src, spitch, width, height, dir, hipStream) /= 0
#else
      success = .true.
#endif
    end function

    function hip_memcpy2d_async_cptr(dst, dpitch, src, spitch, width, height, dir, hipStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr)           :: dst
      integer(kind=c_intptr_t), intent(in) :: dpitch
      type(c_ptr)           :: src
      integer(kind=c_intptr_t), intent(in) :: spitch
      integer(kind=c_intptr_t), intent(in) :: width
      integer(kind=c_intptr_t), intent(in) :: height
      integer(kind=C_INT), intent(in)    :: dir
      integer(kind=c_intptr_t), intent(in) :: hipStream
      logical                            :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_memcpy2d_async_cptr_c(dst, dpitch, src, spitch, width, height, dir, hipStream) /= 0
#else
      success = .true.
#endif
    end function

    function hip_host_register(a, size, flag) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)              :: a
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: flag
      logical :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_host_register_c(a, size, flag) /= 0
#else
      success = .true.
#endif
    end function

    function hip_host_unregister(a) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)              :: a
      logical :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_host_unregister_c(a) /= 0
#else
      success = .true.
#endif
    end function

    subroutine rocblas_Dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_DOUBLE) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Dgemm_intptr_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine rocblas_Dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_DOUBLE) ,value               :: alpha, beta
      type(c_ptr)                     :: a, b, c
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Dgemm_cptr_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine rocblas_Dgemm_intptr_cptr_intptr(cta, ctb, m, n, k, &
                                 alpha, a, lda, b, ldb, beta, c, ldc, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_DOUBLE) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, c
      type(c_ptr)                     :: b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Dgemm_intptr_cptr_intptr_c(rocblasHandle, cta, ctb, m, n, k, &
                                                 alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine rocblas_Dcopy_intptr(n, x, incx, y, incy, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Dcopy_intptr_c(rocblasHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine rocblas_Dcopy_cptr(n, x, incx, y, incy, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Dcopy_cptr_c(rocblasHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine rocblas_Dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Dtrmm_intptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine rocblas_Dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE) ,value               :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Dtrmm_cptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine rocblas_Dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Dtrsm_intptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine rocblas_Dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE) ,value               :: alpha
      type(c_ptr)                    :: a, b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Dtrsm_cptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine rocblas_Dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_DOUBLE) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Dgemv_c(rocblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    subroutine rocblas_Sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_FLOAT) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Sgemm_intptr_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine rocblas_Sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_FLOAT) ,value               :: alpha, beta
      type(c_ptr)                     :: a, b, c
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Sgemm_cptr_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine rocblas_Sgemm_intptr_cptr_intptr(cta, ctb, m, n, k, &
                                 alpha, a, lda, b, ldb, beta, c, ldc, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_FLOAT) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, c
      type(c_ptr)                     :: b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Sgemm_intptr_cptr_intptr_c(rocblasHandle, cta, ctb, m, n, k, &
                                                 alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine rocblas_Scopy_intptr(n, x, incx, y, incy, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Scopy_intptr_c(rocblasHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine rocblas_Scopy_cptr(n, x, incx, y, incy, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Scopy_cptr_c(rocblasHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine rocblas_Strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Strmm_intptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine rocblas_Strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT) ,value               :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Strmm_cptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine rocblas_Strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Strsm_intptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine rocblas_Strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT) ,value               :: alpha
      type(c_ptr)                    :: a, b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Strsm_cptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine rocblas_Sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_FLOAT) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Sgemv_c(rocblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    subroutine rocblas_Zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Zgemm_intptr_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine rocblas_Zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha, beta
      type(c_ptr)                     :: a, b, c
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Zgemm_cptr_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine rocblas_Zgemm_intptr_cptr_intptr(cta, ctb, m, n, k, &
                                 alpha, a, lda, b, ldb, beta, c, ldc, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, c
      type(c_ptr)                     :: b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Zgemm_intptr_cptr_intptr_c(rocblasHandle, cta, ctb, m, n, k, &
                                                 alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine rocblas_Zcopy_intptr(n, x, incx, y, incy, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Zcopy_intptr_c(rocblasHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine rocblas_Zcopy_cptr(n, x, incx, y, incy, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Zcopy_cptr_c(rocblasHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine rocblas_Ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Ztrmm_intptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine rocblas_Ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Ztrmm_cptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine rocblas_Ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Ztrsm_intptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine rocblas_Ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      type(c_ptr)                    :: a, b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Ztrsm_cptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine rocblas_Zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Zgemv_c(rocblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    subroutine rocblas_Cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Cgemm_intptr_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine rocblas_Cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha, beta
      type(c_ptr)                     :: a, b, c
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Cgemm_cptr_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine rocblas_Cgemm_intptr_cptr_intptr(cta, ctb, m, n, k, &
                                 alpha, a, lda, b, ldb, beta, c, ldc, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha, beta
      integer(kind=C_intptr_T)        :: a, c
      type(c_ptr)                     :: b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Cgemm_intptr_cptr_intptr_c(rocblasHandle, cta, ctb, m, n, k, &
                                                 alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine rocblas_Ccopy_intptr(n, x, incx, y, incy, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Ccopy_intptr_c(rocblasHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine rocblas_Ccopy_cptr(n, x, incx, y, incy, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Ccopy_cptr_c(rocblasHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine rocblas_Ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Ctrmm_intptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine rocblas_Ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Ctrmm_cptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine rocblas_Ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Ctrsm_intptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine rocblas_Ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      type(c_ptr)                    :: a, b
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Ctrsm_cptr_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine rocblas_Cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, rocblasHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=C_intptr_T)        :: rocblasHandle
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Cgemv_c(rocblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    function rocblas_pointerModeDevice() result(flag)
      use, intrinsic :: iso_c_binding
      use precision
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_GPU_VERSION
      flag = int(rocblas_pointerModeDevice_c())
#else
      flag = 0
#endif
    end function

    function rocblas_pointerModeHost() result(flag)
      use, intrinsic :: iso_c_binding
      use precision
      implicit none
      integer(kind=ik) :: flag
#ifdef WITH_AMD_GPU_VERSION
      flag = int(rocblas_pointerModeHost_c())
#else
      flag = 0
#endif
    end function

    subroutine rocblas_getPointerMode(rocblasHandle, mode)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: mode

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_getPointerMode_c(rocblasHandle, mode)
#endif
    end subroutine

    subroutine rocblas_setPointerMode(rocblasHandle, mode)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: mode

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_setPointerMode_c(rocblasHandle, mode)
#endif
    end subroutine

    subroutine rocblas_Ddot_intptr(rocblasHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx, incy
      integer(kind=c_intptr_t) :: x, y, result

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Ddot_intptr_c(rocblasHandle, length, x, incx, y, incy, result)
#endif
    end subroutine

    subroutine rocblas_Ddot_cptr(rocblasHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx, incy
      type(c_ptr)              :: x, y, result

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Ddot_cptr_c(rocblasHandle, length, x, incx, y, incy, result)
#endif
    end subroutine

    subroutine rocblas_Dscal_intptr(rocblasHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx
      real(kind=C_DOUBLE) ,value               :: alpha
      integer(kind=c_intptr_t) :: x

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Dscal_intptr_c(rocblasHandle, length, alpha, x, incx)
#endif
    end subroutine

    subroutine rocblas_Dscal_cptr(rocblasHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx
      real(kind=C_DOUBLE) ,value               :: alpha
      type(c_ptr)              :: x

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Dscal_cptr_c(rocblasHandle, length, alpha, x, incx)
#endif
    end subroutine

    subroutine rocblas_Daxpy_intptr(rocblasHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx, incy
      real(kind=C_DOUBLE) ,value               :: alpha
      integer(kind=c_intptr_t) :: x, y

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Daxpy_intptr_c(rocblasHandle, length, alpha, x, incx, y, incy)
#endif
    end subroutine

    subroutine rocblas_Daxpy_cptr(rocblasHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx, incy
      real(kind=C_DOUBLE) ,value               :: alpha
      type(c_ptr)              :: x, y

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Daxpy_cptr_c(rocblasHandle, length, alpha, x, incx, y, incy)
#endif
    end subroutine

    subroutine rocblas_Sdot_intptr(rocblasHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx, incy
      integer(kind=c_intptr_t) :: x, y, result

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Sdot_intptr_c(rocblasHandle, length, x, incx, y, incy, result)
#endif
    end subroutine

    subroutine rocblas_Sdot_cptr(rocblasHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx, incy
      type(c_ptr)              :: x, y, result

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Sdot_cptr_c(rocblasHandle, length, x, incx, y, incy, result)
#endif
    end subroutine

    subroutine rocblas_Sscal_intptr(rocblasHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx
      real(kind=C_FLOAT) ,value               :: alpha
      integer(kind=c_intptr_t) :: x

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Sscal_intptr_c(rocblasHandle, length, alpha, x, incx)
#endif
    end subroutine

    subroutine rocblas_Sscal_cptr(rocblasHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx
      real(kind=C_FLOAT) ,value               :: alpha
      type(c_ptr)              :: x

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Sscal_cptr_c(rocblasHandle, length, alpha, x, incx)
#endif
    end subroutine

    subroutine rocblas_Saxpy_intptr(rocblasHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx, incy
      real(kind=C_FLOAT) ,value               :: alpha
      integer(kind=c_intptr_t) :: x, y

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Saxpy_intptr_c(rocblasHandle, length, alpha, x, incx, y, incy)
#endif
    end subroutine

    subroutine rocblas_Saxpy_cptr(rocblasHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx, incy
      real(kind=C_FLOAT) ,value               :: alpha
      type(c_ptr)              :: x, y

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Saxpy_cptr_c(rocblasHandle, length, alpha, x, incx, y, incy)
#endif
    end subroutine

    subroutine rocblas_Zdot_intptr(conj, rocblasHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
       character(1,c_char), value   :: conj
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx, incy
      integer(kind=c_intptr_t) :: x, y, result

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Zdot_intptr_c(conj,rocblasHandle, length, x, incx, y, incy, result)
#endif
    end subroutine

    subroutine rocblas_Zdot_cptr(conj, rocblasHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
       character(1,c_char), value   :: conj
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx, incy
      type(c_ptr)              :: x, y, result

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Zdot_cptr_c(conj,rocblasHandle, length, x, incx, y, incy, result)
#endif
    end subroutine

    subroutine rocblas_Zscal_intptr(rocblasHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      integer(kind=c_intptr_t) :: x

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Zscal_intptr_c(rocblasHandle, length, alpha, x, incx)
#endif
    end subroutine

    subroutine rocblas_Zscal_cptr(rocblasHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      type(c_ptr)              :: x

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Zscal_cptr_c(rocblasHandle, length, alpha, x, incx)
#endif
    end subroutine

    subroutine rocblas_Zaxpy_intptr(rocblasHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx, incy
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      integer(kind=c_intptr_t) :: x, y

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Zaxpy_intptr_c(rocblasHandle, length, alpha, x, incx, y, incy)
#endif
    end subroutine

    subroutine rocblas_Zaxpy_cptr(rocblasHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx, incy
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      type(c_ptr)              :: x, y

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Zaxpy_cptr_c(rocblasHandle, length, alpha, x, incx, y, incy)
#endif
    end subroutine

    subroutine rocblas_Cdot_intptr(conj, rocblasHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
       character(1,c_char), value   :: conj
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx, incy
      integer(kind=c_intptr_t) :: x, y, result

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Cdot_intptr_c(conj,rocblasHandle, length, x, incx, y, incy, result)
#endif
    end subroutine

    subroutine rocblas_Cdot_cptr(conj, rocblasHandle, length, x, incx, y, incy, result)
      use, intrinsic :: iso_c_binding
      implicit none
       character(1,c_char), value   :: conj
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx, incy
      type(c_ptr)              :: x, y, result

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Cdot_cptr_c(conj,rocblasHandle, length, x, incx, y, incy, result)
#endif
    end subroutine

    subroutine rocblas_Cscal_intptr(rocblasHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      integer(kind=c_intptr_t) :: x

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Cscal_intptr_c(rocblasHandle, length, alpha, x, incx)
#endif
    end subroutine

    subroutine rocblas_Cscal_cptr(rocblasHandle, length, alpha, x, incx)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      type(c_ptr)              :: x

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Cscal_cptr_c(rocblasHandle, length, alpha, x, incx)
#endif
    end subroutine

    subroutine rocblas_Caxpy_intptr(rocblasHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx, incy
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      integer(kind=c_intptr_t) :: x, y

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Caxpy_intptr_c(rocblasHandle, length, alpha, x, incx, y, incy)
#endif
    end subroutine

    subroutine rocblas_Caxpy_cptr(rocblasHandle, length, alpha, x, incx, y, incy)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: rocblasHandle
      integer(kind=c_int)      :: length, incx, incy
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      type(c_ptr)              :: x, y

#ifdef WITH_AMD_GPU_VERSION
      call rocblas_Caxpy_cptr_c(rocblasHandle, length, alpha, x, incx, y, incy)
#endif
    end subroutine

