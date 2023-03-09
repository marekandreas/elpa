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
!
!  interface
!    function sycl_stream_destroy_c(syclStream) result(istat) &
!             bind(C, name="syclStreamDestroyFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value :: syclStream
!      integer(kind=C_INT)             :: istat
!    end function
!  end interface
!
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
!
!  interface
!    function sycl_stream_synchronize_implicit_c() result(istat) &
!             bind(C, name="syclStreamSynchronizeImplicitFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface
!
!  interface
!    function mkl_sycl_set_stream_c(syclHandle, syclStream) result(istat) &
!             bind(C, name="mklSyclSetStreamFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!
!      integer(kind=C_intptr_T), value  :: syclHandle
!      integer(kind=C_intptr_T), value  :: syclStream
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface

!  interface
!    function sycl_solver_set_stream_c(sycl_solverHandle, syclStream) result(istat) &
!             bind(C, name="syclsolverSetStreamFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!
!      integer(kind=C_intptr_T), value  :: sycl_solverHandle
!      integer(kind=C_intptr_T), value  :: syclStream
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface

  ! functions to set and query the GPU devices
  interface
     function sycl_blas_create_c(handle) result(istat) &
              bind(C, name="syclblasCreateFromC")
       use, intrinsic :: iso_c_binding

       implicit none
       integer(kind=C_intptr_T) :: handle
       integer(kind=C_INT)      :: istat
     end function
  end interface

  interface
     function sycl_blas_destroy_c(handle) result(istat) &
              bind(C, name="syclblasDestroyFromC")
       use, intrinsic :: iso_c_binding

       implicit none
       integer(kind=C_intptr_T) :: handle
       integer(kind=C_INT)      :: istat
     end function
  end interface

  interface
    function mkl_sycl_create_c(syclHandle) result(istat) &
             bind(C, name="syclblasCreateFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: syclHandle
      integer(kind=C_INT)      :: istat
    end function
  end interface

  interface
    function mkl_sycl_destroy_c(syclHandle) result(istat) &
             bind(C, name="syclblasDestroyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: syclHandle
      integer(kind=C_INT)      :: istat
    end function
  end interface

  interface
    function sycl_solver_create_c(sycl_solverHandle) result(istat) &
             bind(C, name="syclsolverCreateFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: sycl_solverHandle
      integer(kind=C_INT)      :: istat
    end function
  end interface

  interface
    function sycl_solver_destroy_c(sycl_solverHandle) result(istat) &
             bind(C, name="syclsolverDestroyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: sycl_solverHandle
      integer(kind=C_INT)      :: istat
    end function
  end interface

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
    function sycl_getdevicecount_c(n) result(istat) &
             bind(C, name="syclGetDeviceCountFromC")
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
!
!  interface
!    function sycl_hostRegisterPortable_c() result(flag) &
!             bind(C, name="syclHostRegisterPortableFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface
!
!  interface
!    function sycl_hostRegisterMapped_c() result(flag) &
!             bind(C, name="syclHostRegisterMappedFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface
!
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
!
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
!
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
!
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
!
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
!
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
!
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
!
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
!
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
!
!  interface
!    function sycl_host_unregister_c(a) result(istat) &
!             bind(C, name="syclHostUnregisterFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t), value              :: a
!      integer(kind=C_INT)                          :: istat
!    end function
!  end interface
!
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
      type(c_ptr)                      :: a
      integer(kind=C_INT)              :: istat
    end function
  end interface

  interface sycl_free
    module procedure sycl_free_intptr
    module procedure sycl_free_cptr
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

  interface sycl_malloc
    module procedure sycl_malloc_intptr
    module procedure sycl_malloc_cptr
  end interface

!  interface
!    function sycl_free_host_c(a) result(istat) &
!             bind(C, name="syclFreeHostFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr), value               :: a
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface
!
!  interface
!    function sycl_malloc_host_c(a, width_height) result(istat) &
!             bind(C, name="syclMallocHostFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)                    :: a
!      integer(kind=c_intptr_t), intent(in), value   :: width_height
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

  interface
    subroutine sycl_solver_Dtrtri_c(sycl_solverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="syclsolverDtrtriFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: sycl_solverHandle
    end subroutine
  end interface

  interface
    subroutine sycl_solver_Dpotrf_c(sycl_solverHandle, uplo, n, a, lda, info) &
                              bind(C,name="syclsolverDpotrfFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: sycl_solverHandle
    end subroutine
  end interface

  interface mkl_sycl_Dgemm
    module procedure mkl_sycl_Dgemm_intptr
    module procedure mkl_sycl_Dgemm_cptr
  end interface

  interface
    subroutine mkl_sycl_Dgemm_intptr_c(mkl_syclHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklSyclDgemmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_DOUBLE) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Dgemm_cptr_c(mkl_syclHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklSyclDgemmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_DOUBLE) ,value               :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface mkl_sycl_Dcopy
    module procedure mkl_sycl_Dcopy_intptr
    module procedure mkl_sycl_Dcopy_cptr
  end interface

  interface
    subroutine mkl_sycl_Dcopy_intptr_c(mkl_syclHandle, n, x, incx, y, incy) &
                              bind(C,name="mklSyclDcopyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Dcopy_cptr_c(mkl_syclHandle, n, x, incx, y, incy) &
                              bind(C,name="mklSyclDcopyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface mkl_sycl_Dtrmm
    module procedure mkl_sycl_Dtrmm_intptr
    module procedure mkl_sycl_Dtrmm_cptr
  end interface

  interface
    subroutine mkl_sycl_Dtrmm_intptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklSyclDtrmmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Dtrmm_cptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklSyclDtrmmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface mkl_sycl_Dtrsm
    module procedure mkl_sycl_Dtrsm_intptr
    module procedure mkl_sycl_Dtrsm_cptr
  end interface

  interface
    subroutine mkl_sycl_Dtrsm_intptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklSyclDtrsmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Dtrsm_cptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklSyclDtrsmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Dgemv_c(mkl_syclHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name="mklSyclDgemvFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      real(kind=C_DOUBLE) , value              :: alpha, beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine sycl_solver_Strtri_c(sycl_solverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="syclsolverStrtriFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: sycl_solverHandle
    end subroutine
  end interface

  interface
    subroutine sycl_solver_Spotrf_c(sycl_solverHandle, uplo, n, a, lda, info) &
                              bind(C,name="syclsolverSpotrfFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: sycl_solverHandle
    end subroutine
  end interface

  interface mkl_sycl_Sgemm
    module procedure mkl_sycl_Sgemm_intptr
    module procedure mkl_sycl_Sgemm_cptr
  end interface

  interface
    subroutine mkl_sycl_Sgemm_intptr_c(mkl_syclHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklSyclSgemmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_FLOAT) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Sgemm_cptr_c(mkl_syclHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklSyclSgemmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_FLOAT) ,value               :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface mkl_sycl_Scopy
    module procedure mkl_sycl_Scopy_intptr
    module procedure mkl_sycl_Scopy_cptr
  end interface

  interface
    subroutine mkl_sycl_Scopy_intptr_c(mkl_syclHandle, n, x, incx, y, incy) &
                              bind(C,name="mklSyclScopyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Scopy_cptr_c(mkl_syclHandle, n, x, incx, y, incy) &
                              bind(C,name="mklSyclScopyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface mkl_sycl_Strmm
    module procedure mkl_sycl_Strmm_intptr
    module procedure mkl_sycl_Strmm_cptr
  end interface

  interface
    subroutine mkl_sycl_Strmm_intptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklSyclStrmmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Strmm_cptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklSyclStrmmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface mkl_sycl_Strsm
    module procedure mkl_sycl_Strsm_intptr
    module procedure mkl_sycl_Strsm_cptr
  end interface

  interface
    subroutine mkl_sycl_Strsm_intptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklSyclStrsmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Strsm_cptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklSyclStrsmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Sgemv_c(mkl_syclHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name="mklSyclSgemvFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      real(kind=C_FLOAT) , value              :: alpha, beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine sycl_solver_Ztrtri_c(sycl_solverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="syclsolverZtrtriFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: sycl_solverHandle
    end subroutine
  end interface

  interface
    subroutine sycl_solver_Zpotrf_c(sycl_solverHandle, uplo, n, a, lda, info) &
                              bind(C,name="syclsolverZpotrfFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: sycl_solverHandle
    end subroutine
  end interface

  interface mkl_sycl_Zgemm
    module procedure mkl_sycl_Zgemm_intptr
    module procedure mkl_sycl_Zgemm_cptr
  end interface

  interface
    subroutine mkl_sycl_Zgemm_intptr_c(mkl_syclHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklSyclZgemmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Zgemm_cptr_c(mkl_syclHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklSyclZgemmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface mkl_sycl_Zcopy
    module procedure mkl_sycl_Zcopy_intptr
    module procedure mkl_sycl_Zcopy_cptr
  end interface

  interface
    subroutine mkl_sycl_Zcopy_intptr_c(mkl_syclHandle, n, x, incx, y, incy) &
                              bind(C,name="mklSyclZcopyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Zcopy_cptr_c(mkl_syclHandle, n, x, incx, y, incy) &
                              bind(C,name="mklSyclZcopyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface mkl_sycl_Ztrmm
    module procedure mkl_sycl_Ztrmm_intptr
    module procedure mkl_sycl_Ztrmm_cptr
  end interface

  interface
    subroutine mkl_sycl_Ztrmm_intptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklSyclZtrmmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Ztrmm_cptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklSyclZtrmmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface mkl_sycl_Ztrsm
    module procedure mkl_sycl_Ztrsm_intptr
    module procedure mkl_sycl_Ztrsm_cptr
  end interface

  interface
    subroutine mkl_sycl_Ztrsm_intptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklSyclZtrsmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Ztrsm_cptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklSyclZtrsmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Zgemv_c(mkl_syclHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name="mklSyclZgemvFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      complex(kind=C_DOUBLE_COMPLEX) , value              :: alpha, beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine sycl_solver_Ctrtri_c(sycl_solverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="syclsolverCtrtriFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: sycl_solverHandle
    end subroutine
  end interface

  interface
    subroutine sycl_solver_Cpotrf_c(sycl_solverHandle, uplo, n, a, lda, info) &
                              bind(C,name="syclsolverCpotrfFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: sycl_solverHandle
    end subroutine
  end interface

  interface mkl_sycl_Cgemm
    module procedure mkl_sycl_Cgemm_intptr
    module procedure mkl_sycl_Cgemm_cptr
  end interface

  interface
    subroutine mkl_sycl_Cgemm_intptr_c(mkl_syclHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklSyclCgemmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Cgemm_cptr_c(mkl_syclHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name="mklSyclCgemmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface mkl_sycl_Ccopy
    module procedure mkl_sycl_Ccopy_intptr
    module procedure mkl_sycl_Ccopy_cptr
  end interface

  interface
    subroutine mkl_sycl_Ccopy_intptr_c(mkl_syclHandle, n, x, incx, y, incy) &
                              bind(C,name="mklSyclCcopyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Ccopy_cptr_c(mkl_syclHandle, n, x, incx, y, incy) &
                              bind(C,name="mklSyclCcopyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface mkl_sycl_Ctrmm
    module procedure mkl_sycl_Ctrmm_intptr
    module procedure mkl_sycl_Ctrmm_cptr
  end interface

  interface
    subroutine mkl_sycl_Ctrmm_intptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklSyclCtrmmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Ctrmm_cptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklSyclCtrmmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface mkl_sycl_Ctrsm
    module procedure mkl_sycl_Ctrsm_intptr
    module procedure mkl_sycl_Ctrsm_cptr
  end interface

  interface
    subroutine mkl_sycl_Ctrsm_intptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklSyclCtrsmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Ctrsm_cptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name="mklSyclCtrsmFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
    end subroutine
  end interface

  interface
    subroutine mkl_sycl_Cgemv_c(mkl_syclHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name="mklSyclCgemvFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      complex(kind=C_FLOAT_COMPLEX) , value              :: alpha, beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: mkl_syclHandle
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
!
!  interface nvtxRangePop
!    subroutine nvtxRangePop() bind(C, name='nvtxRangePop')
!    end subroutine
!  end interface
!#endif

  contains

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
!
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
!
!    function mkl_sycl_set_stream(mkl_syclHandle, syclStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: mkl_syclHandle
!      integer(kind=C_intptr_t)                  :: syclStream
!      logical                                   :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = mkl_sycl_set_stream_c(mkl_syclHandle, syclStream) /= 0
!#else
!      success = .true.
!#endif
!    end function
!
!    function sycl_solver_set_stream(sycl_solverHandle, syclStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: sycl_solverHandle
!      integer(kind=C_intptr_t)                  :: syclStream
!      logical                                   :: success
!
!#ifdef WITH_SYCL_SYCL_SOLVER
!      success = sycl_solver_set_stream_c(sycl_solverHandle, syclStream) /= 0
!#else
!      success = .true.
!#endif
!    end function
!
!
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
!
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

    function sycl_blas_create(handle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)                  :: handle
      logical                                   :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = sycl_blas_create_c(handle) /= 0
#else
      success = .true.
#endif
    end function

    function sycl_blas_destroy(handle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_t)                  :: handle
      logical                                   :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = sycl_blas_destroy_c(handle) /= 0
#else
      success = .true.
#endif
    end function

    function mkl_sycl_create(mkl_syclHandle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: mkl_syclHandle
      logical                                   :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = mkl_sycl_create_c(mkl_syclHandle) /= 0
#else
      success = .true.
#endif
    end function

    function mkl_sycl_destroy(mkl_syclHandle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)   :: mkl_syclHandle
      logical                    :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = mkl_sycl_destroy_c(mkl_syclHandle) /= 0
#else
      success = .true.
#endif
    end function

    function sycl_solver_create(sycl_solverHandle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: sycl_solverHandle
      logical                                   :: success
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      success = sycl_solver_create_c(sycl_solverHandle) /= 0
#else
      success = .true.
#endif
    end function

    function sycl_solver_destroy(sycl_solverHandle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: sycl_solverHandle
      logical                                   :: success
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      success = sycl_solver_destroy_c(sycl_solverHandle) /= 0
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

    function sycl_getdevicecount(n) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=ik)     :: n
      integer(kind=c_int)  :: nCasted
      logical              :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = sycl_getdevicecount_c(nCasted) /=0
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
!
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
      type(c_ptr)              :: a
      logical                  :: success
#ifdef WITH_SYCL_GPU_VERSION
      success = sycl_free_cptr_c(a) /= 0
#else
      success = .true.
#endif
    end function


!    function sycl_malloc_host(a, width_height) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)                               :: a
!      integer(kind=c_intptr_t), intent(in)      :: width_height
!      logical                                   :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_malloc_host_c(a, width_height) /= 0
!#else
!      success = .true.
!#endif
!    end function
!
!    function sycl_free_host(a) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      type(c_ptr)                   :: a
!      logical                  :: success
!#ifdef WITH_SYCL_GPU_VERSION
!      success = sycl_free_host_c(a) /= 0
!#else
!      success = .true.
!#endif
!    end function

    function sycl_memset(a, val, size) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t)                :: a
      integer(kind=ik)                        :: val
      integer(kind=c_intptr_t), intent(in)      :: size
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
!
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
!
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
!
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
!
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
!
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
!
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
!
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
!
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
!
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
!
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
!
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

    subroutine sycl_solver_Dtrtri(uplo, diag, n, a, lda, info, sycl_solverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: sycl_solverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call sycl_solver_Dtrtri_c(sycl_solverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    subroutine sycl_solver_Dpotrf(uplo, n, a, lda, info, sycl_solverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: sycl_solverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call sycl_solver_Dpotrf_c(sycl_solverHandle, uplo, n, a, lda, info)
#endif
    end subroutine

    subroutine mkl_sycl_Dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_DOUBLE) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Dgemm_intptr_c(mkl_syclHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_sycl_Dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_DOUBLE) ,value               :: alpha,beta
      type(c_ptr)                     :: a, b, c
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Dgemm_cptr_c(mkl_syclHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_sycl_Dcopy_intptr(n, x, incx, y, incy, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Dcopy_intptr_c(mkl_syclHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_sycl_Dcopy_cptr(n, x, incx, y, incy, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Dcopy_cptr_c(mkl_syclHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_sycl_Dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Dtrmm_intptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_sycl_Dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE) ,value               :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Dtrmm_cptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_sycl_Dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Dtrsm_intptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_sycl_Dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE) ,value               :: alpha
      type(c_ptr)                    :: a, b
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Dtrsm_cptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_sycl_Dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_DOUBLE) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Dgemv_c(mkl_syclHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    subroutine sycl_solver_Strtri(uplo, diag, n, a, lda, info, sycl_solverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: sycl_solverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call sycl_solver_Strtri_c(sycl_solverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    subroutine sycl_solver_Spotrf(uplo, n, a, lda, info, sycl_solverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: sycl_solverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call sycl_solver_Spotrf_c(sycl_solverHandle, uplo, n, a, lda, info)
#endif
    end subroutine

    subroutine mkl_sycl_Sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_FLOAT) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Sgemm_intptr_c(mkl_syclHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_sycl_Sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_FLOAT) ,value               :: alpha,beta
      type(c_ptr)                     :: a, b, c
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Sgemm_cptr_c(mkl_syclHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_sycl_Scopy_intptr(n, x, incx, y, incy, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Scopy_intptr_c(mkl_syclHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_sycl_Scopy_cptr(n, x, incx, y, incy, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Scopy_cptr_c(mkl_syclHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_sycl_Strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Strmm_intptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_sycl_Strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT) ,value               :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Strmm_cptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_sycl_Strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Strsm_intptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_sycl_Strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT) ,value               :: alpha
      type(c_ptr)                    :: a, b
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Strsm_cptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_sycl_Sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_FLOAT) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Sgemv_c(mkl_syclHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    subroutine sycl_solver_Ztrtri(uplo, diag, n, a, lda, info, sycl_solverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: sycl_solverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call sycl_solver_Ztrtri_c(sycl_solverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    subroutine sycl_solver_Zpotrf(uplo, n, a, lda, info, sycl_solverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: sycl_solverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call sycl_solver_Zpotrf_c(sycl_solverHandle, uplo, n, a, lda, info)
#endif
    end subroutine

    subroutine mkl_sycl_Zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Zgemm_intptr_c(mkl_syclHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_sycl_Zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha,beta
      type(c_ptr)                     :: a, b, c
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Zgemm_cptr_c(mkl_syclHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_sycl_Zcopy_intptr(n, x, incx, y, incy, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Zcopy_intptr_c(mkl_syclHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_sycl_Zcopy_cptr(n, x, incx, y, incy, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Zcopy_cptr_c(mkl_syclHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_sycl_Ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Ztrmm_intptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_sycl_Ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Ztrmm_cptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_sycl_Ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Ztrsm_intptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_sycl_Ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha
      type(c_ptr)                    :: a, b
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Ztrsm_cptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_sycl_Zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_DOUBLE_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Zgemv_c(mkl_syclHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    subroutine sycl_solver_Ctrtri(uplo, diag, n, a, lda, info, sycl_solverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: sycl_solverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call sycl_solver_Ctrtri_c(sycl_solverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    subroutine sycl_solver_Cpotrf(uplo, n, a, lda, info, sycl_solverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: sycl_solverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call sycl_solver_Cpotrf_c(sycl_solverHandle, uplo, n, a, lda, info)
#endif
    end subroutine

    subroutine mkl_sycl_Cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Cgemm_intptr_c(mkl_syclHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_sycl_Cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha,beta
      type(c_ptr)                     :: a, b, c
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Cgemm_cptr_c(mkl_syclHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_sycl_Ccopy_intptr(n, x, incx, y, incy, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Ccopy_intptr_c(mkl_syclHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_sycl_Ccopy_cptr(n, x, incx, y, incy, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Ccopy_cptr_c(mkl_syclHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_sycl_Ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Ctrmm_intptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_sycl_Ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Ctrmm_cptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_sycl_Ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Ctrsm_intptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_sycl_Ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha
      type(c_ptr)                    :: a, b
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Ctrsm_cptr_c(mkl_syclHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_sycl_Cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, mkl_syclHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_FLOAT_COMPLEX) ,value               :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=C_intptr_T)        :: mkl_syclHandle
#ifdef WITH_SYCL_GPU_VERSION
      call mkl_sycl_Cgemv_c(mkl_syclHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

