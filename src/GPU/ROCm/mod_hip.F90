!    Copyright 2021, A. Marek
!
!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
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
! This file was written by A. Marek, MPCDF


#include "config-f90.h"
module hip_functions
  use, intrinsic :: iso_c_binding
  use precision
  implicit none

  public

  integer(kind=ik) :: hipMemcpyHostToDevice
  integer(kind=ik) :: hipMemcpyDeviceToHost
  integer(kind=ik) :: hipMemcpyDeviceToDevice
  integer(kind=ik) :: hipHostRegisterDefault
  integer(kind=ik) :: hipHostRegisterPortable
  integer(kind=ik) :: hipHostRegisterMapped

  ! TODO global variable, has to be changed
  integer(kind=C_intptr_T) :: rocblasHandle = -1

!  integer(kind=c_intptr_t), parameter :: size_of_double_real    = 8_rk8
!#ifdef WANT_SINGLE_PRECISION_REAL
!  integer(kind=c_intptr_t), parameter :: size_of_single_real    = 4_rk4
!#endif
!
!  integer(kind=c_intptr_t), parameter :: size_of_double_complex = 16_ck8
!#ifdef WANT_SINGLE_PRECISION_COMPLEX
!  integer(kind=c_intptr_t), parameter :: size_of_single_complex = 8_ck4
!#endif

  ! functions to set and query the CUDA devices
  interface
    function rocblas_create_c(handle) result(istat) &
             bind(C, name="rocblasCreateFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: handle
      integer(kind=C_INT)  :: istat
    end function rocblas_create_c
  end interface

  interface
    function rocblas_destroy_c(handle) result(istat) &
             bind(C, name="rocblasDestroyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: handle
      integer(kind=C_INT)  :: istat
    end function rocblas_destroy_c
  end interface

  interface
    function hip_setdevice_c(n) result(istat) &
             bind(C, name="hipSetDeviceFromC")

      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT), value    :: n
      integer(kind=C_INT)           :: istat
    end function hip_setdevice_c
  end interface

  interface
    function hip_getdevicecount_c(n) result(istat) &
             bind(C, name="hipGetDeviceCountFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT), intent(out) :: n
      integer(kind=C_INT)              :: istat
    end function hip_getdevicecount_c
  end interface

  interface
    function hip_devicesynchronize_c()result(istat) &
             bind(C,name='hipDeviceSynchronizeFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT)                       :: istat

    end function hip_devicesynchronize_c
  end interface


  ! functions to copy CUDA memory
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
    function hip_memcpy_c(dst, src, size, dir) result(istat) &
             bind(C, name="hipMemcpyFromC")

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_intptr_t), value              :: dst
      integer(kind=C_intptr_t), value              :: src
      integer(kind=c_intptr_t), intent(in), value    :: size
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=C_INT)                          :: istat

    end function hip_memcpy_c
  end interface

  interface
    function hip_memcpy2d_c(dst, dpitch, src, spitch, width, height , dir) result(istat) &
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

    end function hip_memcpy2d_c
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

    end function hip_host_register_c
  end interface

  interface
    function hip_host_unregister_c(a) result(istat) &
             bind(C, name="hipHostUnregisterFromC")

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_intptr_t), value              :: a
      integer(kind=C_INT)                          :: istat

    end function hip_host_unregister_c
  end interface

  ! functions to allocate and free CUDA memory

  interface
    function hip_free_c(a) result(istat) &
             bind(C, name="hipFreeFromC")

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_intptr_T), value  :: a
      integer(kind=C_INT)              :: istat

    end function hip_free_c
  end interface

  interface
    function hip_malloc_c(a, width_height) result(istat) &
             bind(C, name="hipMallocFromC")

      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_T)                    :: a
      integer(kind=c_intptr_t), intent(in), value   :: width_height
      integer(kind=C_INT)                         :: istat

    end function hip_malloc_c
  end interface

  interface
    function hip_host_free_c(a) result(istat) &
             bind(C, name="hipHostFreeFromC")

      use, intrinsic :: iso_c_binding

      implicit none
      type(c_ptr), value                    :: a
      integer(kind=C_INT)              :: istat

    end function hip_host_free_c
  end interface

  interface
    function hip_host_malloc_c(a, width_height) result(istat) &
             bind(C, name="hipHostMallocFromC")

      use, intrinsic :: iso_c_binding
      implicit none

      type(c_ptr)                    :: a
      integer(kind=c_intptr_t), intent(in), value   :: width_height
      integer(kind=C_INT)                         :: istat

    end function hip_host_malloc_c
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

    end function hip_memset_c
  end interface

  ! cuBLAS
  interface
    subroutine rocblas_dgemm_c(handle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name='rocblas_dgemm_elpa_wrapper')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_DOUBLE),value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: handle

    end subroutine rocblas_dgemm_c
  end interface

  interface
    subroutine rocblas_sgemm_c(handle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name='rocblas_sgemm_elpa_wrapper')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_FLOAT),value                :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: handle

    end subroutine rocblas_sgemm_c
  end interface

  interface
    subroutine rocblas_dtrmm_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='rocblas_dtrmm_elpa_wrapper')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE), value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: handle

    end subroutine rocblas_dtrmm_c
  end interface

  interface
    subroutine rocblas_strmm_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='rocblas_strmm_elpa_wrapper')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT), value               :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: handle

    end subroutine rocblas_strmm_c
  end interface

  interface
    subroutine rocblas_zgemm_c(handle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc) &
                              bind(C,name='rocblas_zgemm_elpa_wrapper')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: cta, ctb
      integer(kind=C_INT),value              :: m,n,k
      integer(kind=C_INT), intent(in), value :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX),value           :: alpha,beta
      integer(kind=C_intptr_T), value        :: a, b, c
      integer(kind=C_intptr_T), value         :: handle

    end subroutine rocblas_zgemm_c
  end interface

  interface
    subroutine rocblas_cgemm_c(handle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc) &
                              bind(C,name='rocblas_cgemm_elpa_wrapper')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: cta, ctb
      integer(kind=C_INT),value              :: m,n,k
      integer(kind=C_INT), intent(in), value :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX),value            :: alpha,beta
      integer(kind=C_intptr_T), value        :: a, b, c
      integer(kind=C_intptr_T), value         :: handle

    end subroutine rocblas_cgemm_c
  end interface

  interface
    subroutine rocblas_ztrmm_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='rocblas_ztrmm_elpa_wrapper')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: side, uplo, trans, diag
      integer(kind=C_INT),value              :: m,n
      integer(kind=C_INT), intent(in), value :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX), value          :: alpha
      integer(kind=C_intptr_T), value        :: a, b
      integer(kind=C_intptr_T), value         :: handle

    end subroutine rocblas_ztrmm_c
  end interface

  interface
    subroutine rocblas_ctrmm_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='rocblas_ctrmm_elpa_wrapper')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: side, uplo, trans, diag
      integer(kind=C_INT),value              :: m,n
      integer(kind=C_INT), intent(in), value :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX), value           :: alpha
      integer(kind=C_intptr_T), value        :: a, b
      integer(kind=C_intptr_T), value         :: handle

    end subroutine rocblas_ctrmm_c
  end interface

  interface
    subroutine rocblas_dgemv_c(handle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name='rocblas_dgemv_elpa_wrapper')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      real(kind=C_DOUBLE),value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: handle

    end subroutine rocblas_dgemv_c
  end interface

  interface
    subroutine rocblas_sgemv_c(handle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name='rocblas_sgemv_elpa_wrapper')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      real(kind=C_FLOAT),value                :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: handle

    end subroutine rocblas_sgemv_c
  end interface

  interface
    subroutine rocblas_zgemv_c(handle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name='rocblas_zgemv_elpa_wrapper')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      complex(kind=C_DOUBLE_COMPLEX),value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: handle

    end subroutine rocblas_zgemv_c
  end interface

  interface
    subroutine rocblas_cgemv_c(handle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name='rocblas_cgemv_elpa_wrapper')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      complex(kind=C_FLOAT_COMPLEX),value                :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: handle

    end subroutine rocblas_cgemv_c
  end interface

#if 0
#ifdef WITH_NVTX
  ! NVTX profiling interfaces
  interface nvtxRangePushA
    subroutine nvtxRangePushA(name) bind(C, name='nvtxRangePushA')
      use, intrinsic :: iso_c_binding
      character(kind=C_CHAR,len=1) :: name(*)
    end subroutine
  end interface

  interface nvtxRangePop
    subroutine nvtxRangePop() bind(C, name='nvtxRangePop')
    end subroutine
  end interface
#endif
#endif

  contains
#if 0
#ifdef WITH_NVTX
   ! this wrapper is needed for the string conversion
   subroutine nvtxRangePush(range_name)
     implicit none
     character(len=*), intent(in) :: range_name

     character(kind=C_CHAR,len=1), dimension(len(range_name)+1) :: c_name
     integer i

     do i = 1, len(range_name)
       c_name(i) = range_name(i:i)
     end do
     c_name(len(range_name)+1) = char(0)

     call nvtxRangePushA(c_name)
   end subroutine
#endif
#endif
    ! functions to set and query the CUDA devices

   function rocblas_create(handle) result(success)
     use, intrinsic :: iso_c_binding
     implicit none

     integer(kind=C_intptr_t)                  :: handle
     logical                                   :: success
#ifdef WITH_AMD_GPU_VERSION
     success = rocblas_create_c(handle) /= 0
#else
     success = .true.
#endif
   end function

   function rocblas_destroy(handle) result(success)
     use, intrinsic :: iso_c_binding
     implicit none

     integer(kind=C_intptr_t)                  :: handle
     logical                                   :: success
#ifdef WITH_AMD_GPU_VERSION
     success = rocblas_destroy_c(handle) /= 0
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
    end function hip_setdevice

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
    end function hip_getdevicecount

    function hip_devicesynchronize()result(success)

      use, intrinsic :: iso_c_binding

      implicit none
      logical :: success
#ifdef WITH_AMD_GPU_VERSION
      success = hip_devicesynchronize_c() /=0
#else
      success = .true.
#endif
    end function hip_devicesynchronize
    ! functions to allocate and free memory

    function hip_malloc(a, width_height) result(success)

     use, intrinsic :: iso_c_binding
     implicit none

     integer(kind=C_intptr_t)                  :: a
     integer(kind=c_intptr_t), intent(in)        :: width_height
     logical                                   :: success
#ifdef WITH_AMD_GPU_VERSION
     success = hip_malloc_c(a, width_height) /= 0
#else
     success = .true.
#endif
   end function

   function hip_free(a) result(success)

     use, intrinsic :: iso_c_binding

     implicit none
     integer(kind=C_intptr_T) :: a
     logical                  :: success
#ifdef WITH_AMD_GPU_VERSION
     success = hip_free_c(a) /= 0
#else
     success = .true.
#endif
   end function hip_free

    function hip_host_malloc(a, width_height) result(success)

     use, intrinsic :: iso_c_binding
     implicit none

      type(c_ptr)                    :: a
     integer(kind=c_intptr_t), intent(in)        :: width_height
     logical                                   :: success
#ifdef WITH_AMD_GPU_VERSION
     success = hip_host_malloc_c(a, width_height) /= 0
#else
     success = .true.
#endif
   end function

   function hip_host_free(a) result(success)

     use, intrinsic :: iso_c_binding

     implicit none
      type(c_ptr), value                    :: a
     logical                  :: success
#ifdef WITH_AMD_GPU_VERSION
     success = hip_host_free_c(a) /= 0
#else
     success = .true.
#endif
   end function hip_host_free

 function hip_memset(a, val, size) result(success)

   use, intrinsic :: iso_c_binding

   implicit none

   integer(kind=c_intptr_t)                :: a
   integer(kind=ik)                        :: val
   integer(kind=c_intptr_t), intent(in)      :: size
   integer(kind=C_INT)                     :: istat

   logical :: success
#ifdef WITH_AMD_GPU_VERSION
   success= hip_memset_c(a, int(val,kind=c_int), int(size,kind=c_intptr_t)) /=0
#else
   success = .true.
#endif
 end function hip_memset

 ! functions to memcopy CUDA memory

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

 function hip_memcpy(dst, src, size, dir) result(success)

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_intptr_t)              :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)    :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success

#ifdef WITH_AMD_GPU_VERSION
        success = hip_memcpy_c(dst, src, size, dir) /= 0
#else
        success = .true.
#endif
    end function

    function hip_memcpy2d(dst, dpitch, src, spitch, width, height , dir) result(success)

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
      success = hip_memcpy2d_c(dst, dpitch, src, spitch, width, height , dir) /= 0
#else
      success = .true.
#endif
    end function hip_memcpy2d

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

    ! cuBLAS
    subroutine rocblas_dgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_DOUBLE)             :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_dgemm_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine rocblas_dgemm

    subroutine rocblas_sgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_FLOAT)              :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_sgemm_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine rocblas_sgemm

    subroutine rocblas_dtrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE)             :: alpha
      integer(kind=C_intptr_T)        :: a, b
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_dtrmm_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine rocblas_dtrmm

    subroutine rocblas_strmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT)              :: alpha
      integer(kind=C_intptr_T)        :: a, b
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_strmm_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine rocblas_strmm

    subroutine rocblas_zgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX)          :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_zgemm_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)
#endif
    end subroutine rocblas_zgemm

    subroutine rocblas_cgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX)           :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_cgemm_c(rocblasHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)
#endif
    end subroutine rocblas_cgemm

    subroutine rocblas_ztrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX)          :: alpha
      integer(kind=C_intptr_T)        :: a, b
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_ztrmm_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine rocblas_ztrmm

    subroutine rocblas_ctrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX)           :: alpha
      integer(kind=C_intptr_T)        :: a, b
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_ctrmm_c(rocblasHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine rocblas_ctrmm

    subroutine rocblas_dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_DOUBLE)             :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_dgemv_c(rocblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine rocblas_dgemv

    subroutine rocblas_sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_FLOAT)              :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_sgemv_c(rocblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine rocblas_sgemv

    subroutine rocblas_zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_DOUBLE_COMPLEX)             :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_zgemv_c(rocblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine rocblas_zgemv

    subroutine rocblas_cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_FLOAT_COMPLEX)              :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
#ifdef WITH_AMD_GPU_VERSION
      call rocblas_cgemv_c(rocblasHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine rocblas_cgemv


!     subroutine cublas_dsymv(cta, n, alpha, a, lda, x, incx, beta, y, incy)
!       use, intrinsic :: iso_c_binding
!
!       implicit none
!       character(1,C_CHAR),value       :: cta
!       integer(kind=C_INT)             :: n
!       integer(kind=C_INT), intent(in) :: lda,incx,incy
!       real(kind=C_DOUBLE)             :: alpha,beta
!       integer(kind=C_intptr_T)        :: a, x, y
! #ifdef WITH_NVIDIA_GPU_VERSION
!       call cublas_dsymv_c(cta, n, alpha, a, lda, x, incx, beta, y, incy)
! #endif
!     end subroutine cublas_dsymv
!
!     subroutine cublas_ssymv(cta, n, alpha, a, lda, x, incx, beta, y, incy)
!       use, intrinsic :: iso_c_binding
!
!       implicit none
!       character(1,C_CHAR),value       :: cta
!       integer(kind=C_INT)             :: n
!       integer(kind=C_INT), intent(in) :: lda,incx,incy
!       real(kind=C_FLOAT)              :: alpha,beta
!       integer(kind=C_intptr_T)        :: a, x, y
! #ifdef WITH_NVIDIA_GPU_VERSION
!       call cublas_ssymv_c(cta, n, alpha, a, lda, x, incx, beta, y, incy)
! #endif
!     end subroutine cublas_ssymv
!
!     subroutine cublas_zsymv(cta, n, alpha, a, lda, x, incx, beta, y, incy)
!       use, intrinsic :: iso_c_binding
!
!       implicit none
!       character(1,C_CHAR),value       :: cta
!       integer(kind=C_INT)             :: n
!       integer(kind=C_INT), intent(in) :: lda,incx,incy
!       complex(kind=C_DOUBLE_COMPLEX)             :: alpha,beta
!       integer(kind=C_intptr_T)        :: a, x, y
! #ifdef WITH_NVIDIA_GPU_VERSION
! !       call cublas_zsymv_c(cta, n, alpha, a, lda, x, incx, beta, y, incy)
! #endif
!     end subroutine cublas_zsymv
!
!     subroutine cublas_csymv(cta, n, alpha, a, lda, x, incx, beta, y, incy)
!       use, intrinsic :: iso_c_binding
!
!       implicit none
!       character(1,C_CHAR),value       :: cta
!       integer(kind=C_INT)             :: n
!       integer(kind=C_INT), intent(in) :: lda,incx,incy
!       complex(kind=C_FLOAT_COMPLEX)              :: alpha,beta
!       integer(kind=C_intptr_T)        :: a, x, y
! #ifdef WITH_NVIDIA_GPU_VERSION
! !       call cublas_csymv_c(cta, n, alpha, a, lda, x, incx, beta, y, incy)
! #endif
!     end subroutine cublas_csymv


end module hip_functions
