!    Copyright 2014, A. Marek
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
! This file was written by A. Marek, MPCDF (2022)
! it is based on a proto type implementation by A. Poeppl, Intel (2022)


#include "config-f90.h"

module openmp_offload_functions
  use, intrinsic :: iso_c_binding
  use precision
  implicit none

  public

  integer(kind=ik) :: openmpOffloadMemcpyHostToDevice
  integer(kind=ik) :: openmpOffloadMemcpyDeviceToHost
  integer(kind=ik) :: openmpOffloadMemcpyDeviceToDevice
  integer(kind=ik) :: openmpOffloadHostRegisterDefault
  integer(kind=ik) :: openmpOffloadHostRegisterPortable
  integer(kind=ik) :: openmpOffloadHostRegisterMapped

  ! TODO global variable, has to be changed
  integer(kind=C_intptr_T), allocatable :: openmpOffloadHandleArray(:)
  integer(kind=C_intptr_T), allocatable :: openmpOffloadsolverHandleArray(:)
  integer(kind=c_int), allocatable      :: openmpOffloadDeviceArray(:)

  ! functions to set and query the GPU devices
  interface
    function openmp_offload_blas_create_c(handle) result(istat) &
             bind(C, name="openmpOffloadblasCreateFromC")
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_intptr_T) :: handle
      integer(kind=C_INT)      :: istat
    end function
  end interface
!
!  interface
!    function cublas_destroy_c(handle) result(istat) &
!             bind(C, name="cublasDestroyFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T) :: handle
!      integer(kind=C_INT)  :: istat
!    end function cublas_destroy_c
!  end interface
!
  interface
    function openmp_offload_solver_create_c(handle) result(istat) &
             bind(C, name="openmpOffloadsolverCreateFromC")
      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_intptr_T) :: handle
      integer(kind=C_INT)      :: istat
    end function
  end interface

!  interface
!    function cusolver_destroy_c(handle) result(istat) &
!             bind(C, name="cusolverDestroyFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T) :: handle
!      integer(kind=C_INT)  :: istat
!    end function cusolver_destroy_c
!  end interface


  interface
    function openmp_offload_setdevice_c(device_id) result(istat) &
                     bind (C, name="openmpOffloadSetDeviceFromC")
      use, intrinsic :: iso_c_binding

      integer (kind=c_int), intent(in), value :: device_id
      integer(kind=C_INT)                     :: istat
    end function
  end interface

  interface
    function openmp_offload_getdevicecount_c(n) result(istat) &
             bind(C, name="openmpOffloadGetDeviceCountFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_INT), intent(out) :: n
      integer(kind=C_INT)              :: istat
    end function openmp_offload_getdevicecount_c
  end interface

  !interface
  !  function cuda_devicesynchronize_c()result(istat) &
  !           bind(C,name='cudaDeviceSynchronizeFromC')

  !    use, intrinsic :: iso_c_binding

  !    implicit none
  !    integer(kind=C_INT)                       :: istat

  !  end function cuda_devicesynchronize_c
  !end interface


  ! functions to copy memory
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
!    function cuda_hostRegisterDefault_c() result(flag) &
!             bind(C, name="cudaHostRegisterDefaultFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface
!
!  interface
!    function cuda_hostRegisterPortable_c() result(flag) &
!             bind(C, name="cudaHostRegisterPortableFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface
!
!  interface
!    function cuda_hostRegisterMapped_c() result(flag) &
!             bind(C, name="cudaHostRegisterMappedFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_int) :: flag
!    end function
!  end interface

  interface
    function openmp_offload_memcpy_intptr_c(dst, src, elems, direction) &
             result (istat) bind (C, name="openmpOffloadMemcpyFromC")
      use, intrinsic :: iso_c_binding
      integer (kind=c_intptr_t), intent(in), value :: dst
      integer (kind=c_intptr_t), intent(in), value :: src
      integer (kind=c_intptr_t), intent(in), value :: elems
      integer (kind=c_int), intent(in), value      :: direction
      integer (kind=c_int)                         :: istat
    end function
  end interface

  interface
    function openmp_offload_memcpy_cptr_c(dst, src, elems, direction) &
                    result (istat) bind (C, name="openmpOffloadMemcpyFromC")
      use, intrinsic :: iso_c_binding
      type(c_ptr), intent(in), value               :: dst
      type(c_ptr), intent(in), value               :: src
      integer (kind=c_intptr_t), intent(in), value :: elems
      integer (kind=c_int), intent(in), value      :: direction
      integer (kind=c_int)                         :: istat
    end function

  end interface

  interface
    function openmp_offload_memcpy_mixed_to_device_c(dst, src, elems, direction) result (istat) &
                    bind (C, name="openmpOffloadMemcpyFromC")
      use, intrinsic :: iso_c_binding

      type(c_ptr), intent(in), value               :: dst
      integer (kind=c_intptr_t), intent(in), value :: src
      integer (kind=c_intptr_t), intent(in), value :: elems
      integer (kind=c_int), intent(in), value      :: direction
      integer (kind=c_int)                         :: istat
    end function
  end interface

  interface
    function openmp_offload_memcpy_mixed_to_host_c(dst, src, elems, direction) result (istat) &
                    bind (C, name="openmpOffloadMemcpyFromC")
      use, intrinsic :: iso_c_binding

      type(c_ptr), intent(in), value               :: src
      integer (kind=c_intptr_t), intent(in), value :: dst
      integer (kind=c_intptr_t), intent(in), value :: elems
      integer (kind=c_int), intent(in), value      :: direction
      integer (kind=c_int)                         :: istat
    end function


  end interface

!  interface
!    function cuda_memcpy2d_intptr_c(dst, dpitch, src, spitch, width, height , dir) result(istat) &
!             bind(C, name="cudaMemcpy2dFromC")
!
!      use, intrinsic :: iso_c_binding
!
!      implicit none
!
!      integer(kind=C_intptr_T), value                :: dst
!      integer(kind=c_intptr_t), intent(in), value    :: dpitch
!      integer(kind=C_intptr_T), value                :: src
!      integer(kind=c_intptr_t), intent(in), value    :: spitch
!      integer(kind=c_intptr_t), intent(in), value    :: width
!      integer(kind=c_intptr_t), intent(in), value    :: height
!      integer(kind=C_INT), intent(in), value         :: dir
!      integer(kind=C_INT)                            :: istat
!
!    end function cuda_memcpy2d_intptr_c
!  end interface
!
!  interface
!    function cuda_memcpy2d_cptr_c(dst, dpitch, src, spitch, width, height , dir) result(istat) &
!             bind(C, name="cudaMemcpy2dFromC")
!
!      use, intrinsic :: iso_c_binding
!
!      implicit none
!
!      type(c_ptr), value                :: dst
!      integer(kind=c_intptr_t), intent(in), value    :: dpitch
!      type(c_ptr), value                :: src
!      integer(kind=c_intptr_t), intent(in), value    :: spitch
!      integer(kind=c_intptr_t), intent(in), value    :: width
!      integer(kind=c_intptr_t), intent(in), value    :: height
!      integer(kind=C_INT), intent(in), value         :: dir
!      integer(kind=C_INT)                            :: istat
!
!    end function cuda_memcpy2d_cptr_c
!  end interface
!
!  interface
!    function cuda_host_register_c(a, size, flag) result(istat) &
!             bind(C, name="cudaHostRegisterFromC")
!
!      use, intrinsic :: iso_c_binding
!
!      implicit none
!      integer(kind=C_intptr_t), value              :: a
!      integer(kind=c_intptr_t), intent(in), value  :: size
!      integer(kind=C_INT), intent(in), value       :: flag
!      integer(kind=C_INT)                          :: istat
!
!    end function cuda_host_register_c
!  end interface
!
!  interface
!    function cuda_host_unregister_c(a) result(istat) &
!             bind(C, name="cudaHostUnregisterFromC")
!
!      use, intrinsic :: iso_c_binding
!
!      implicit none
!      integer(kind=C_intptr_t), value              :: a
!      integer(kind=C_INT)                          :: istat
!
!    end function cuda_host_unregister_c
!  end interface

  ! functions to allocate and free device memory
  interface
    function openmp_offload_free_c(a) result (istat) &
             bind (C, name="openmpOffloadFreeFromC")
      use, intrinsic :: iso_c_binding

      integer (kind=c_intptr_t), intent(inout) :: a
      integer (kind=c_int)                     :: istat
    end function
  end interface

  interface openmp_offload_memcpy
    module procedure openmp_offload_memcpy_intptr
    module procedure openmp_offload_memcpy_cptr
    module procedure openmp_offload_memcpy_mixed_to_device
    module procedure openmp_offload_memcpy_mixed_to_host
  end interface


  interface
    function openmp_offload_malloc_c(a, elems) result (istat) &
             bind (C, name="openmpOffloadMallocFromC")
      use, intrinsic :: iso_c_binding

      integer (kind=c_intptr_t), intent(inout)     :: a
      integer (kind=c_intptr_t), intent(in), value :: elems
      integer (kind=c_int)                         :: istat
    end function
  end interface

  !interface
  !  function cuda_free_host_c(a) result(istat) &
  !           bind(C, name="cudaFreeHostFromC")

  !    use, intrinsic :: iso_c_binding

  !    implicit none
  !    type(c_ptr), value                    :: a
  !    integer(kind=C_INT)              :: istat

  !  end function cuda_free_host_c
  !end interface

  !interface
  !  function cuda_malloc_host_c(a, width_height) result(istat) &
  !           bind(C, name="cudaMallocHostFromC")

  !    use, intrinsic :: iso_c_binding
  !    implicit none

  !    type(c_ptr)                    :: a
  !    integer(kind=c_intptr_t), intent(in), value   :: width_height
  !    integer(kind=C_INT)                         :: istat

  !  end function cuda_malloc_host_c
  !end interface

  interface
    function openmp_offload_memset_c(array, val, elems) result (istat) &
                    bind (C, name="openmpOffloadMemsetFromC")
      use, intrinsic :: iso_c_binding

      integer (kind=c_intptr_t), intent(in), value :: array
      integer (kind=c_intptr_t), intent(in), value :: elems
      ! changed compared to demo
      integer (kind=c_int), intent(in), value      :: val
      integer (kind=c_int)                         :: istat
    end function
  end interface

  ! mkl lapack openmp offload
  interface
    subroutine mkl_openmp_offload_dtrtri_c(handle, uplo, diag, n, a, lda, info) &
                              bind(C,name='mklOpenmpOffloadDtrtriFromC')
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_strtri_c(handle, uplo, diag, n, a, lda, info) &
                              bind(C,name='mklOpenmpOffloadStrtriFromC')
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_ztrtri_c(handle, uplo, diag, n, a, lda, info) &
                              bind(C,name='mklOpenmpOffloadZtrtriFromC')
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_ctrtri_c(handle, uplo, diag, n, a, lda, info) &
                              bind(C,name='mklOpenmpOffloadCtrtriFromC')
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_dpotrf_c(handle, uplo, n, a, lda, info) &
                              bind(C,name='mklOpenmpOffloadDpotrfFromC')
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_spotrf_c(handle, uplo, n, a, lda, info) &
                              bind(C,name='mklOpenmpOffloadSpotrfFromC')
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_zpotrf_c(handle, uplo, n, a, lda, info) &
                              bind(C,name='mklOpenmpOffloadZpotrfFromC')
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_cpotrf_c(handle, uplo, n, a, lda, info) &
                              bind(C,name='mklOpenmpOffloadCpotrfFromC')
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: handle

    end subroutine
  end interface

  ! mkl blas openmp offload
  interface mkl_opemmp_offload_dgemm
    module procedure mkl_openmp_offload_dgemm_intptr
    module procedure mkl_openmp_offload_dgemm_cptr
  end interface

  interface mkl_openmp_offload_sgemm
    module procedure mkl_openmp_offload_sgemm_intptr
    module procedure mkl_openmp_offload_sgemm_cptr
  end interface

  interface mkl_openmp_offload_zgemm
    module procedure mkl_openmp_offload_zgemm_intptr
    module procedure mkl_openmp_offload_zgemm_cptr
  end interface

  interface mkl_openmp_offload_cgemm
    module procedure mkl_openmp_offload_cgemm_intptr
    module procedure mkl_openmp_offload_cgemm_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_dgemm_c(handle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                           bind (C, name="mklOpenmpOffloadDgemmFromC")
      use, intrinsic :: iso_c_binding

      character (1, c_char), intent(in), value     :: cta, ctb
      integer (kind=c_int), intent(inout)          :: m, n, k
      integer (kind=c_int), intent(in)             :: lda, ldb, ldc
      real (kind=c_double)                         :: alpha, beta
      integer (kind=c_intptr_t), intent(in), value :: a, b, c
      integer(kind=C_intptr_T), value              :: handle
    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_dgemm_cptr_c(handle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name='mklOpenmpOffloadDgemmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_DOUBLE),value               :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: handle

    end subroutine 
  end interface

  interface
    subroutine mkl_openmp_offload_sgemm_intptr_c(handle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name='mklOpenmpOffloadSgemmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_FLOAT),value                :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_sgemm_cptr_c(handle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
                              bind(C,name='mklOpenmpOffloadSgemmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_FLOAT),value                :: alpha,beta
      type(c_ptr), value                      :: a, b, c
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface



  interface mkl_openmp_offload_dcopy
    module procedure mkl_openmp_offload_dcopy_intptr
    module procedure mkl_openmp_offload_dcopy_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_dcopy_intptr_c(handle, n, x, incx, y, incy) &
                              bind(C,name='mklOpenmpOffloadDcopyFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: handle

    end subroutine 
  end interface

  interface
    subroutine mkl_openmp_offload_dcopy_cptr_c(handle, n, x, incx, y, incy) &
                              bind(C,name='openmpOpenmpOffloadDcopyFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface

  interface mkl_openmp_offload_scopy
    module procedure mkl_openmp_offload_scopy_intptr
    module procedure mkl_openmp_offload_scopy_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_scopy_intptr_c(handle, n, x, incx, y, incy) &
                              bind(C,name='mklOpenmpOffloadScopyFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: handle

    end subroutine 
  end interface

  interface
    subroutine mkl_openmp_offload_scopy_cptr_c(handle, n, x, incx, y, incy) &
                              bind(C,name='openmpOpenmpOffloadScopyFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface

  interface mkl_openmp_offload_dtrmm
    module procedure mkl_openmp_offload_dtrmm_intptr
    module procedure mkl_openmp_offload_dtrmm_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_dtrmm_intptr_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='mklOpenmpOffloadDtrmmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE), value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_dtrmm_cptr_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='mklOpenmpOffloadDtrmmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE), value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface


  interface mkl_openmp_offload_strmm
    module procedure mkl_openmp_offload_strmm_intptr
    module procedure mkl_openmp_offload_strmm_cptr
  end interface


  interface
    subroutine mkl_openmp_offload_strmm_intptr_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='mklOpenmpOffloadStrmmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE), value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_strmm_cptr_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='mklOpenmpOffloadStrmmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE), value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface


  interface mkl_openmp_offload_dtrsm
    module procedure mkl_openmp_offload_dtrsm_intptr
    module procedure mkl_openmp_offload_dtrsm_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_dtrsm_intptr_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='mklOpenmpOffloadDtrsm_elpa_wrapper')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE), value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_dtrsm_cptr_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='mklOpenmpOffloadDtrsm_elpa_wrapper')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE), value              :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface


  interface mkl_openmp_offload_strsm
    module procedure mkl_openmp_offload_strsm_intptr
    module procedure mkl_openmp_offload_strsm_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_strsm_intptr_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='mklOpenmpOffloadStrsm_elpa_wrapper')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT), value               :: alpha
      integer(kind=C_intptr_T), value         :: a, b
      integer(kind=C_intptr_T), value         :: handle

    end subroutine 
  end interface

  interface
    subroutine mkl_openmp_offload_strsm_cptr_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='mklOpenmpOffloadStrsm_elpa_wrapper')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT), value               :: alpha
      type(c_ptr), value                      :: a, b
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface


  interface
    subroutine mkl_openmp_offload_zgemm_intptr_c(handle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc) &
                              bind(C,name='mkl_openmp_offload_ZgemmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: cta, ctb
      integer(kind=C_INT),value              :: m,n,k
      integer(kind=C_INT), intent(in), value :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX),value   :: alpha,beta
      integer(kind=C_intptr_T), value        :: a, b, c
      integer(kind=C_intptr_T), value        :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_zgemm_cptr_c(handle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc) &
                              bind(C,name='mkl_openmpOffloadZgemmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: cta, ctb
      integer(kind=C_INT),value              :: m,n,k
      integer(kind=C_INT), intent(in), value :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX),value   :: alpha,beta
      type(c_ptr), value                     :: a, b, c
      integer(kind=C_intptr_T), value        :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_cgemm_intptr_c(handle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc) &
                              bind(C,name='mkl_openmpOffloadCgemmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: cta, ctb
      integer(kind=C_INT),value              :: m,n,k
      integer(kind=C_INT), intent(in), value :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX),value    :: alpha,beta
      integer(kind=C_intptr_T), value        :: a, b, c
      integer(kind=C_intptr_T), value        :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_cgemm_cptr_c(handle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc) &
                              bind(C,name='mkl_openmpOffloadCgemmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: cta, ctb
      integer(kind=C_INT),value              :: m,n,k
      integer(kind=C_INT), intent(in), value :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX),value    :: alpha,beta
      type(c_ptr), value                     :: a, b, c
      integer(kind=C_intptr_T), value        :: handle

    end subroutine
  end interface


  interface mkl_openmp_offload_zcopy
    module procedure mkl_openmp_offload_zcopy_intptr
    module procedure mkl_openmp_offload_zcopy_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_zcopy_intptr_c(handle, n, x, incx, y, incy) &
                              bind(C,name='mklOpenmpOffloadZcopyFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_zcopy_cptr_c(handle, n, x, incx, y, incy) &
                              bind(C,name='mklOpenmpOffloadZcopyFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface


  interface mkl_openmp_offload_ccopy
    module procedure mkl_openmp_offload_ccopy_intptr
    module procedure mkl_openmp_offload_ccopy_cptr
  end interface

  interface
    subroutine mkl_openmmp_offload_ccopy_intptr_c(handle, n, x, incx, y, incy) &
                              bind(C,name='mklOpenmpOffloadCcopyFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      integer(kind=C_intptr_T), value         :: x, y
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_ccopy_cptr_c(handle, n, x, incx, y, incy) &
                              bind(C,name='mklOpenmpOffloadCcopyFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT),value               :: n
      integer(kind=C_INT), intent(in), value  :: incx,incy
      type(c_ptr), value                      :: x, y
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface

  interface mkl_openmp_offload_ztrmm
    module procedure mkl_openmp_offload_ztrmm_intptr
    module procedure mkl_openmp_offload_ztrmm_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_ztrmm_intptr_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='mklOpenmpOffloadZtrmmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: side, uplo, trans, diag
      integer(kind=C_INT),value              :: m,n
      integer(kind=C_INT), intent(in), value :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX), value          :: alpha
      integer(kind=C_intptr_T), value        :: a, b
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_ztrmm_cptr_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='mklOpenmpOffloadZtrmmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: side, uplo, trans, diag
      integer(kind=C_INT),value              :: m,n
      integer(kind=C_INT), intent(in), value :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX), value  :: alpha
      type(c_ptr), value                     :: a, b
      integer(kind=C_intptr_T), value        :: handle

    end subroutine
  end interface

  interface mkl_openmp_offload_ctrmm
    module procedure mkl_openmp_offload_ctrmm_intptr
    module procedure mkl_openmp_offload_ctrmm_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_ctrmm_intptr_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='mklOpenmpOffloadCtrmmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: side, uplo, trans, diag
      integer(kind=C_INT),value              :: m,n
      integer(kind=C_INT), intent(in), value :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX), value   :: alpha
      integer(kind=C_intptr_T), value        :: a, b
      integer(kind=C_intptr_T), value        :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_ctrmm_cptr_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='mklOpenmpOffloadCtrmmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: side, uplo, trans, diag
      integer(kind=C_INT),value              :: m,n
      integer(kind=C_INT), intent(in), value :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX), value   :: alpha
      type(c_ptr), value                     :: a, b
      integer(kind=C_intptr_T), value        :: handle

    end subroutine mkl_openmp_offload_ctrmm_cptr_c
  end interface

  interface mkl_openmp_offload_ztrsm
    module procedure mkl_openmp_offload_ztrsm_intptr
    module procedure mkl_openmp_offload_ztrsm_cptr
  end interface

  interface
    subroutine mkl_openmp_offload_ztrsm_intptr_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='mklOpenmpOffloadZtrsmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: side, uplo, trans, diag
      integer(kind=C_INT),value              :: m,n
      integer(kind=C_INT), intent(in), value :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX), value          :: alpha
      integer(kind=C_intptr_T), value        :: a, b
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface

  interface
    subroutine mkl_openmp_offload_ztrsm_cptr_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='mklOpenmpOffloadZtrsmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: side, uplo, trans, diag
      integer(kind=C_INT),value              :: m,n
      integer(kind=C_INT), intent(in), value :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX), value  :: alpha
      type(c_ptr), value                     :: a, b
      integer(kind=C_intptr_T), value        :: handle

    end subroutine
  end interface

  interface mkl_openmp_offload_ctrsm
    module procedure mkl_openmp_offload_ctrsm_intptr
    module procedure mkl_openmp_offload_ctrsm_cptr
  end interface

 interface
    subroutine mkl_openmp_offload_ctrsm_intptr_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='mklOpenmpOffloadCtrsmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: side, uplo, trans, diag
      integer(kind=C_INT),value              :: m,n
      integer(kind=C_INT), intent(in), value :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX), value   :: alpha
      integer(kind=C_intptr_T), value        :: a, b
      integer(kind=C_intptr_T), value        :: handle

    end subroutine 
  end interface

  interface
    subroutine mkl_openmp_offload_ctrsm_cptr_c(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) &
                              bind(C,name='mklOpenmpOffloadCtrsmFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: side, uplo, trans, diag
      integer(kind=C_INT),value              :: m,n
      integer(kind=C_INT), intent(in), value :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX), value   :: alpha
      type(c_ptr), value                     :: a, b
      integer(kind=C_intptr_T), value        :: handle

    end subroutine
  end interface


  interface
    subroutine mkl_openmp_offload_dgemv_c(handle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name='mklOpenmpOffloadDgemvFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      real(kind=C_DOUBLE),value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: handle

    end subroutine 
  end interface

  interface
    subroutine mkl_openmp_offload_sgemv_c(handle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name='mklOpenmpOffloadSgemvFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      real(kind=C_FLOAT),value                :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: handle

    end subroutine
  end interface

 interface
    subroutine mkl_openmp_offload_zgemv_c(handle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name='mklOpenmpOffloadZgemvFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      complex(kind=C_DOUBLE_COMPLEX),value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: handle

    end subroutine 
  end interface

  interface
    subroutine mkl_openmp_offload_cgemv_c(handle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy) &
                              bind(C,name='mklOpenmpOffloadCgemvFromC')

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: cta
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,incx,incy
      complex(kind=C_FLOAT_COMPLEX),value                :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, x, y
      integer(kind=C_intptr_T), value         :: handle

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

#ifdef WITH_NVTX
!   ! this wrapper is needed for the string conversion
!   subroutine nvtxRangePush(range_name)
!     implicit none
!     character(len=*), intent(in) :: range_name
!
!     character(kind=C_CHAR,len=1), dimension(len(range_name)+1) :: c_name
!     integer i
!
!     do i = 1, len(range_name)
!       c_name(i) = range_name(i:i)
!     end do
!     c_name(len(range_name)+1) = char(0)
!
!     call nvtxRangePushA(c_name)
!   end subroutine
#endif

    ! functions to set and query the GPU devices

   function openmp_offload_blas_create(handle) result(success)
     use, intrinsic :: iso_c_binding
     implicit none

     integer(kind=C_intptr_t)                  :: handle
     logical                                   :: success
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
     success = openmp_offload_blas_create_c(handle) /= 0
#else
     success = .true.
#endif
   end function

!   function cublas_destroy(handle) result(success)
!     use, intrinsic :: iso_c_binding
!     implicit none
!
!     integer(kind=C_intptr_t)                  :: handle
!     logical                                   :: success
!#ifdef WITH_NVIDIA_GPU_VERSION
!     success = cublas_destroy_c(handle) /= 0
!#else
!     success = .true.
!#endif
!   end function

   function openmp_offload_solver_create(handle) result(success)
     use, intrinsic :: iso_c_binding
     implicit none

     integer(kind=C_intptr_t)                  :: handle
     logical                                   :: success
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
     success = openmp_offload_solver_create_c(handle) /= 0
#else
     success = .true.
#endif
   end function

!   function cusolver_destroy(handle) result(success)
!     use, intrinsic :: iso_c_binding
!     implicit none
!
!     integer(kind=C_intptr_t)                  :: handle
!     logical                                   :: success
!#ifdef WITH_NVIDIA_CUSOLVER
!     success = cusolver_destroy_c(handle) /= 0
!#else
!     success = .true.
!#endif
!   end function

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
    end function openmp_offload_getdevicecount

!    function cuda_devicesynchronize()result(success)
!
!      use, intrinsic :: iso_c_binding
!
!      implicit none
!      logical :: success
!#ifdef WITH_NVIDIA_GPU_VERSION
!      success = cuda_devicesynchronize_c() /=0
!#else
!      success = .true.
!#endif
!    end function cuda_devicesynchronize


    ! functions to allocate and free memory

    function openmp_offload_malloc(array, elements) result(success)
      use, intrinsic :: iso_c_binding

      integer (kind=c_intptr_t), intent(inout) :: array
      integer (kind=c_intptr_t), intent(in)    :: elements
      logical                                  :: success

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = openmp_offload_malloc_c(array, elements) /= 0
#else
      success = .true.
#endif
    end function

    function openmp_offload_free(array) result(success)
      use, intrinsic :: iso_c_binding

      integer (kind=c_intptr_t), intent(inout) :: array
      logical                                  :: success

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = openmp_offload_free_c(array) /= 0
#else
      success = .true.
#endif
    end function

!    function cuda_malloc_host(a, width_height) result(success)
!
!     use, intrinsic :: iso_c_binding
!     implicit none
!
!     type(c_ptr)                               :: a
!     integer(kind=c_intptr_t), intent(in)      :: width_height
!     logical                                   :: success
!#ifdef WITH_NVIDIA_GPU_VERSION
!     success = cuda_malloc_host_c(a, width_height) /= 0
!#else
!     success = .true.
!#endif
!   end function
!
!   function cuda_free_host(a) result(success)
!
!     use, intrinsic :: iso_c_binding
!
!     implicit none
!      type(c_ptr), value                    :: a
!     logical                  :: success
!#ifdef WITH_NVIDIA_GPU_VERSION
!     success = cuda_free_host_c(a) /= 0
!#else
!     success = .true.
!#endif
!   end function cuda_free_host


    function openmp_offload_memset(array, val, elems) result(success)
      use, intrinsic :: iso_c_binding

      integer (kind=c_intptr_t), intent(in)        :: array
      integer (kind=c_intptr_t), intent(in)        :: elems
      ! changed compared to demo
      integer (kind=c_int), intent(in), value      :: val
      logical                                      :: success

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = openmp_offload_memset_c(array, val, elems) /= 0
#else
      success = .true.
#endif
    end function


    ! functions to memcopy memory

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
      flag = int(openmp_offload_memcpyDeviceToHost_c())
#else
      flag = 0
#endif
    end function

!   function cuda_hostRegisterDefault() result(flag)
!     use, intrinsic :: iso_c_binding
!     use precision
!     implicit none
!     integer(kind=ik) :: flag
!#ifdef WITH_NVIDIA_GPU_VERSION
!   flag = int(cuda_hostRegisterDefault_c())
!#else
!   flag = 0
!#endif
!   end function
!
!   function cuda_hostRegisterPortable() result(flag)
!     use, intrinsic :: iso_c_binding
!     use precision
!     implicit none
!     integer(kind=ik) :: flag
!#ifdef WITH_NVIDIA_GPU_VERSION
!   flag = int(cuda_hostRegisterPortable_c())
!#else
!   flag = 0
!#endif
!   end function
!
!   function cuda_hostRegisterMapped() result(flag)
!     use, intrinsic :: iso_c_binding
!     use precision
!     implicit none
!     integer(kind=ik) :: flag
!#ifdef WITH_NVIDIA_GPU_VERSION
!   flag = int(cuda_hostRegisterMapped_c())
!#else
!   flag = 0
!#endif
!   end function

    function openmp_offload_memcpy_intptr(dst, src, elems, direction) result(success)
      use, intrinsic :: iso_c_binding

      integer (kind=c_intptr_t), intent(inout) :: dst
      integer (kind=c_intptr_t), intent(inout) :: src
      integer (kind=c_intptr_t), intent(in)    :: elems
      integer (kind=c_int), intent(in)         :: direction
      logical                                  :: success
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = openmp_offload_memcpy_intptr_c(dst, src, elems, direction) /= 0
#else
      success = .true.
#endif
    end function
  
    function openmp_offload_memcpy_cptr(dst, src, elems, direction) result(success)
      use, intrinsic :: iso_c_binding
  
      type(c_ptr), intent(inout)            :: dst
      type(c_ptr), intent(inout)            :: src
      integer (kind=c_intptr_t), intent(in) :: elems
      integer (kind=c_int), intent(in)      :: direction
      logical                               :: success
  
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = openmp_offload_memcpy_cptr_c(dst, src, elems, direction) /= 0
#else
      success = .true.
#endif
    end function

    function openmp_offload_memcpy_mixed_to_device(dst, src, elems, direction) result(success)
      use, intrinsic :: iso_c_binding

      type(c_ptr), intent(inout)               :: dst
      integer (kind=c_intptr_t), intent(inout) :: src
      integer (kind=c_intptr_t), intent(in)    :: elems
      integer (kind=c_int), intent(in)         :: direction
      logical                                  :: success

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = openmp_offload_memcpy_mixed_c(dst, src, elems, direction) /= 0
#else
      success = .true.
#endif
    end function

    function openmp_offload_memcpy_mixed_to_host(dst, src, elems, direction) result(success)
      use, intrinsic :: iso_c_binding

      type(c_ptr), intent(inout)               :: src
      integer (kind=c_intptr_t), intent(inout) :: dst
      integer (kind=c_intptr_t), intent(in)    :: elems
      integer (kind=c_int), intent(in)         :: direction
      logical                                  :: success

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      success = openmp_offload_memcpy_mixed_c(dst, src, elems, direction) /= 0
#else
      success = .true.
#endif
    end function


!    function cuda_memcpy2d_intptr(dst, dpitch, src, spitch, width, height , dir) result(success)
!
!      use, intrinsic :: iso_c_binding
!
!      implicit none
!
!      integer(kind=C_intptr_T)           :: dst
!      integer(kind=c_intptr_t), intent(in) :: dpitch
!      integer(kind=C_intptr_T)           :: src
!      integer(kind=c_intptr_t), intent(in) :: spitch
!      integer(kind=c_intptr_t), intent(in) :: width
!      integer(kind=c_intptr_t), intent(in) :: height
!      integer(kind=C_INT), intent(in)    :: dir
!      logical                            :: success
!#ifdef WITH_NVIDIA_GPU_VERSION
!      success = cuda_memcpy2d_intptr_c(dst, dpitch, src, spitch, width, height , dir) /= 0
!#else
!      success = .true.
!#endif
!    end function cuda_memcpy2d_intptr
!
!    function cuda_memcpy2d_cptr(dst, dpitch, src, spitch, width, height , dir) result(success)
!
!      use, intrinsic :: iso_c_binding
!
!      implicit none
!
!      type(c_ptr)           :: dst
!      integer(kind=c_intptr_t), intent(in) :: dpitch
!      type(c_ptr)           :: src
!      integer(kind=c_intptr_t), intent(in) :: spitch
!      integer(kind=c_intptr_t), intent(in) :: width
!      integer(kind=c_intptr_t), intent(in) :: height
!      integer(kind=C_INT), intent(in)    :: dir
!      logical                            :: success
!#ifdef WITH_NVIDIA_GPU_VERSION
!      success = cuda_memcpy2d_cptr_c(dst, dpitch, src, spitch, width, height , dir) /= 0
!#else
!      success = .true.
!#endif
!    end function cuda_memcpy2d_cptr
!
! function cuda_host_register(a, size, flag) result(success)
!
!      use, intrinsic :: iso_c_binding
!
!      implicit none
!      integer(kind=C_intptr_t)              :: a
!      integer(kind=c_intptr_t), intent(in)  :: size
!      integer(kind=C_INT), intent(in)       :: flag
!      logical :: success
!
!#ifdef WITH_NVIDIA_GPU_VERSION
!        success = cuda_host_register_c(a, size, flag) /= 0
!#else
!        success = .true.
!#endif
!    end function
!
! function cuda_host_unregister(a) result(success)
!
!      use, intrinsic :: iso_c_binding
!
!      implicit none
!      integer(kind=C_intptr_t)              :: a
!      logical :: success
!
!#ifdef WITH_NVIDIA_GPU_VERSION
!        success = cuda_host_unregister_c(a) /= 0
!#else
!        success = .true.
!#endif
!    end function

    subroutine mkl_openmp_offload_dtrtri(uplo, diag, n, a, lda, info, threadID)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadsolverHandle

      if (present(threadID)) then
        openmpOffloadsolverHandle = openmpOffloadsolverHandleArray(threadID)
      else
        openmpOffloadsolverHandle = openmpOffloadsolverHandleArray(0)
      endif      

#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call mkl_openmp_offload_dtrtri_c(openmpOffloadsolverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    subroutine mkl_openmp_offload_strtri(uplo, diag, n, a, lda, info, threadID)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadsolverHandle

      if (present(threadID)) then
        openmpOffloadsolverHandle = openmpOffloadsolverHandleArray(threadID)
      else
        openmpOffloadsolverHandle = openmpOffloadsolverHandleArray(0)
      endif      

#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call mkl_openmp_offload_strtri_c(openmpOffloadsolverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    subroutine mkl_openmp_offload_ztrtri(uplo, diag, n, a, lda, info, threadID)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadsolverHandle

      if (present(threadID)) then
        openmpOffloadsolverHandle = openmpOffloadsolverHandleArray(threadID)
      else
        openmpOffloadsolverHandle = openmpOffloadsolverHandleArray(0)
      endif      

#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call mkl_openmp_offload_ztrtri_c(openmpOffloadsolverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    subroutine mkl_openmp_offload_ctrtri(uplo, diag, n, a, lda, info, threadID)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadsolverHandle

      if (present(threadID)) then
        openmpOffloadsolverHandle = openmpOffloadsolverHandleArray(threadID)
      else
        openmpOffloadsolverHandle = openmpOffloadsolverHandleArray(0)
      endif      

#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call mkl_openmp_offload_ctrtri_c(openmpOffloadsolverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    subroutine mkl_openmp_offload_dpotrf(uplo, n, a, lda, info, threadID)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadsolverHandle

      if (present(threadID)) then
        openmpOffloadsolverHandle = openmpOffloadsolverHandleArray(threadID)
      else
        openmpOffloadsolverHandle = openmpOffloadsolverHandleArray(0)
      endif      

#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call mkl_openmp_offload_dpotrf_c(openmpOffloadsolverHandle, uplo, n, a, lda, info)
#endif
    end subroutine

    subroutine mkl_openmp_offload_spotrf(uplo, n, a, lda, info, threadID)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadsolverHandle

      if (present(threadID)) then
        openmpOffloadsolverHandle = openmpOffloadsolverHandleArray(threadID)
      else
        openmpOffloadsolverHandle = openmpOffloadsolverHandleArray(0)
      endif      

#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call mkl_openmp_offload_spotrf_c(openmpOffloadsolverHandle, uplo, n, a, lda, info)
#endif
    end subroutine

    subroutine mkl_openmp_offload_zpotrf(uplo, n, a, lda, info, threadID)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadsolverHandle

      if (present(threadID)) then
        openmpOffloadsolverHandle = openmpOffloadsolverHandleArray(threadID)
      else
        openmpOffloadsolverHandle = openmpOffloadsolverHandleArray(0)
      endif      

#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call mkl_openmp_offload_zpotrf_c(openmpOffloadsolverHandle, uplo, n, a, lda, info)
#endif
    end subroutine

    subroutine mkl_openmp_offload_cpotrf(uplo, n, a, lda, info, threadID)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadsolverHandle

      if (present(threadID)) then
        openmpOffloadsolverHandle = openmpOffloadsolverHandleArray(threadID)
      else
        openmpOffloadsolverHandle = openmpOffloadsolverHandleArray(0)
      endif      

#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call mkl_openmp_offload_cpotrf_c(openmpOffloadsolverHandle, uplo, n, a, lda, info)
#endif
    end subroutine

    ! mkl
    subroutine mkl_openmp_offload_dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, threadID)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_DOUBLE)             :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_dgemm_intptr_c(openmpOffloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine

    subroutine mkl_openmp_offload_dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, threadID)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_DOUBLE)             :: alpha,beta
      type(c_ptr)                     :: a, b, c
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_dgemm_cptr_c(openmpOffloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine 



    subroutine mkl_openmp_offload_sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, threadID)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_FLOAT)              :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_sgemm_intptr_c(openmpOffloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine 


    subroutine mkl_openmp_offload_sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, threadID)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_FLOAT)              :: alpha,beta
      type(c_ptr)                     :: a, b, c
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_sgemm_cptr_c(openmpOffloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine



    subroutine mkl_openmp_offload_dcopy_intptr(n, x, incx, y, incy, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_dcopy_intptr_c(openmpOffloadHandle, n, x, incx, y, incy)
#endif
    end subroutine 

    subroutine mkl_openmp_offload_dcopy_cptr(n, x, incx, y, incy, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_VERSION
      call mkl_openmp_offload_dcopy_cptr_c(openmpOffloadHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_scopy_intptr(n, x, incx, y, incy, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_VERSION
      call mkl_openmp_offload_scopy_intptr_c(openmpOffloadHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_scopy_cptr(n, x, incx, y, incy, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_scopy_cptr_c(openmpOffloadHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE)             :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_dtrmm_intptr_c(openmpOffloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE)             :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_dtrmm_cptr_c(openmpOffloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine


   subroutine mkl_openmp_offload_strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT)              :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_strmm_intptr_c(openmpOffloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT)              :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_strmm_cptr_c(openmpOffloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE)             :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_dtrsm_intptr_c(openmpOffloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE)             :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_dtrsm_cptr_c(openmpOffloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine


    subroutine mkl_openmp_offload_strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT)              :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_strsm_intptr_c(openmpOffloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT)              :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_strsm_cptr_c(openmpOffloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine


    subroutine mkl_openmp_offload_zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_zgemm_intptr_c(openmpOffloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)
#endif
    end subroutine


    subroutine mkl_openmp_offload_zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha,beta
      type(c_ptr)                     :: a, b, c
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_zgemm_cptr_c(openmpOffloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)
#endif
    end subroutine



    subroutine mkl_openmp_offload_cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX)   :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c  
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_cgemm_intptr_c(openmpOffloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)
#endif
    end subroutine

    subroutine mkl_openmp_offload_cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX)   :: alpha,beta
      type(c_ptr)                     :: a, b, c  
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_cgemm_cptr_c(openmpOffloadHandle, cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)
#endif
    end subroutine



    subroutine mkl_openmp_offload_zcopy_intptr(n, x, incx, y, incy, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_zcopy_intptr_c(openmpOffloadHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_zcopy_cptr(n, x, incx, y, incy, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_zcopy_cptr_c(openmpOffloadHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_ccopy_intptr(n, x, incx, y, incy, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_ccopy_intptr_c(openmpOffloadHandle, n, x, incx, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_ccopy_cptr(n, x, incx, y, incy, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_ccopy_cptr_c(openmpOffloadHandle, n, x, incx, y, incy)
#endif
    end subroutine


    subroutine mkl_openmp_offload_ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_ztrmm_intptr_c(openmpOffloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_ztrmm_cptr_c(openmpOffloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine


    subroutine mkl_openmp_offload_ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX)   :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_ctrmm_intptr_c(openmpOffloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX)   :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_ctrmm_cptr_c(openmpOffloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_ztrsm_intptr_c(openmpOffloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_ztrsm_cptr_c(openmpOffloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine


    subroutine mkl_openmp_offload_ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX)   :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_ctrsm_intptr_c(openmpOffloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine

    subroutine mkl_openmp_offload_ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, threadID)

      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX)   :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_ctrsm_cptr_c(openmpOffloadHandle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine


    subroutine mkl_openmp_offload_dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, threadID)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_DOUBLE)             :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_dgemv_c(openmpOffloadHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, threadID)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_FLOAT)              :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_sgemv_c(openmpOffloadHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, threadID)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_zgemv_c(openmpOffloadHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine

    subroutine mkl_openmp_offload_cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, threadID)
      use, intrinsic :: iso_c_binding

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_FLOAT_COMPLEX)   :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=c_int), optional   :: threadID
      integer(kind=C_intptr_T)        :: openmpOffloadHandle

      if (present(threadID)) then
        openmpOffloadHandle = openmpOffloadHandleArray(threadID)
      else
        openmpOffloadHandle = openmpOffloadHandleArray(0)
      endif      
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      call mkl_openmp_offload_cgemv_c(openmpOffloadHandle, cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
#endif
    end subroutine


!     subroutine mkl_openmp_offload_dsymv(cta, n, alpha, a, lda, x, incx, beta, y, incy)
!       use, intrinsic :: iso_c_binding
!
!       implicit none
!       character(1,C_CHAR),value       :: cta
!       integer(kind=C_INT)             :: n
!       integer(kind=C_INT), intent(in) :: lda,incx,incy
!       real(kind=C_DOUBLE)             :: alpha,beta
!       integer(kind=C_intptr_T)        :: a, x, y
! #ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!       call mkl_openmp_offload_dsymv_c(cta, n, alpha, a, lda, x, incx, beta, y, incy)
! #endif
!     end subroutine mkl_openmp_offload_dsymv
!
!     subroutine mkl_openmp_offload_ssymv(cta, n, alpha, a, lda, x, incx, beta, y, incy)
!       use, intrinsic :: iso_c_binding
!
!       implicit none
!       character(1,C_CHAR),value       :: cta
!       integer(kind=C_INT)             :: n
!       integer(kind=C_INT), intent(in) :: lda,incx,incy
!       real(kind=C_FLOAT)              :: alpha,beta
!       integer(kind=C_intptr_T)        :: a, x, y
! #ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
!       call mkl_openmp_offload_ssymv_c(cta, n, alpha, a, lda, x, incx, beta, y, incy)
! #endif
!     end subroutine mkl_openmp_offload_ssymv
!
!     subroutine mkl_openmp_offload_zsymv(cta, n, alpha, a, lda, x, incx, beta, y, incy)
!       use, intrinsic :: iso_c_binding
!
!       implicit none
!       character(1,C_CHAR),value       :: cta
!       integer(kind=C_INT)             :: n
!       integer(kind=C_INT), intent(in) :: lda,incx,incy
!       complex(kind=C_DOUBLE_COMPLEX)             :: alpha,beta
!       integer(kind=C_intptr_T)        :: a, x, y
! #ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
! !       call mkl_openmp_offload_zsymv_c(cta, n, alpha, a, lda, x, incx, beta, y, incy)
! #endif
!     end subroutine mkl_openmp_offload_zsymv
!
!     subroutine mkl_openmp_offload_csymv(cta, n, alpha, a, lda, x, incx, beta, y, incy)
!       use, intrinsic :: iso_c_binding
!
!       implicit none
!       character(1,C_CHAR),value       :: cta
!       integer(kind=C_INT)             :: n
!       integer(kind=C_INT), intent(in) :: lda,incx,incy
!       complex(kind=C_FLOAT_COMPLEX)              :: alpha,beta
!       integer(kind=C_intptr_T)        :: a, x, y
! #ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
! !       call mkl_openmp_offload_csymv_c(cta, n, alpha, a, lda, x, incx, beta, y, incy)
! #endif
!     end subroutine mkl_openmp_offload_csymv

end module openmp_offload_functions
