#if 0
!    Copyright 2021, A. Marek, MPCDF
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
!    This particular source code file contains additions, changes and
!    enhancements authored by Intel Corporation which is not part of
!    the ELPA consortium.
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
#endif


#include "config-f90.h"
module elpa_gpu
  use precision
  use iso_c_binding
#ifdef WITH_INTEL_GPU_VERSION
  use mkl_offload
#endif
  integer(kind=c_int), parameter :: nvidia_gpu = 1
  integer(kind=c_int), parameter :: amd_gpu = 2
  integer(kind=c_int), parameter :: intel_gpu = 3
  integer(kind=c_int), parameter :: no_gpu = -1
  integer(kind=c_int)            :: use_gpu_vendor
  integer(kind=c_int)            :: gpuHostRegisterDefault    
  integer(kind=c_int)            :: gpuMemcpyHostToDevice    
  integer(kind=c_int)            :: gpuMemcpyDeviceToHost   
  integer(kind=c_int)            :: gpuMemcpyDeviceToDevice
  integer(kind=c_int)            :: gpuHostRegisterMapped
  integer(kind=c_int)            :: gpuHostRegisterPortable

  integer(kind=c_intptr_t), parameter :: size_of_double_real    = 8_rk8
#ifdef WANT_SINGLE_PRECISION_REAL
  integer(kind=c_intptr_t), parameter :: size_of_single_real    = 4_rk4
#endif

  integer(kind=c_intptr_t), parameter :: size_of_double_complex = 16_ck8
#ifdef WANT_SINGLE_PRECISION_COMPLEX
  integer(kind=c_intptr_t), parameter :: size_of_single_complex = 8_ck4
#endif
  contains
    function gpu_vendor() result(vendor)
      use precision
      implicit none
      integer(kind=c_int) :: vendor
      ! default
      vendor = no_gpu
#ifdef WITH_NVIDIA_GPU_VERSION
      vendor = nvidia_gpu
#endif
#ifdef WITH_AMD_GPU_VERSION
      vendor = amd_gpu
#endif
#ifdef WITH_INTEL_GPU_VERSION
      vendor = intel_gpu
#endif
      use_gpu_vendor = vendor
      return
    end function

    subroutine set_gpu_parameters
      use cuda_functions
      use hip_functions
      implicit none

      if (use_gpu_vendor == nvidia_gpu) then
        cudaMemcpyHostToDevice   = cuda_memcpyHostToDevice()
        gpuMemcpyHostToDevice    = cudaMemcpyHostToDevice
        cudaMemcpyDeviceToHost   = cuda_memcpyDeviceToHost()
        gpuMemcpyDeviceToHost    = cudaMemcpyDeviceToHost
        cudaMemcpyDeviceToDevice = cuda_memcpyDeviceToDevice()
        gpuMemcpyDeviceToDevice  = cudaMemcpyDeviceToDevice
        cudaHostRegisterPortable = cuda_hostRegisterPortable()
        gpuHostRegisterPortable  = cudaHostRegisterPortable
        cudaHostRegisterMapped   = cuda_hostRegisterMapped()
        gpuHostRegisterMapped    = cudaHostRegisterMapped
        cudaHostRegisterDefault  = cuda_hostRegisterDefault()
        gpuHostRegisterDefault   = cudaHostRegisterDefault
      endif

      if (use_gpu_vendor == amd_gpu) then
        hipMemcpyHostToDevice   = hip_memcpyHostToDevice()
        gpuMemcpyHostToDevice   = hipMemcpyHostToDevice
        hipMemcpyDeviceToHost   = hip_memcpyDeviceToHost()
        gpuMemcpyDeviceToHost   = hipMemcpyDeviceToHost
        hipMemcpyDeviceToDevice = hip_memcpyDeviceToDevice()
        gpuMemcpyDeviceToDevice = hipMemcpyDeviceToDevice
        hipHostRegisterPortable = hip_hostRegisterPortable()
        gpuHostRegisterPortable = hipHostRegisterPortable
        hipHostRegisterMapped   = hip_hostRegisterMapped()
        gpuHostRegisterMapped   = hipHostRegisterMapped
        hipHostRegisterDefault  = hip_hostRegisterDefault()
        gpuHostRegisterDefault  = hipHostRegisterDefault
      endif

    end subroutine

    function gpu_malloc_host(array, elements) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions
      implicit none
      type(c_ptr)                          :: array
      integer(kind=c_intptr_t), intent(in) :: elements
      logical                              :: success

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_malloc_host(array, elements)
      endif

      if (use_gpu_vendor == amd_gpu) then
        success = hip_host_malloc(array, elements)
      endif

    end function

    function gpu_malloc(array, elements) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions
      implicit none
      integer(kind=C_intptr_T)             :: array
      integer(kind=c_intptr_t), intent(in) :: elements
      logical                              :: success

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_malloc(array, elements)
      endif

      if (use_gpu_vendor == amd_gpu) then
        success = hip_malloc(array, elements)
      endif

    end function

    function gpu_host_register(array, elements, flag) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions
      implicit none
      integer(kind=C_intptr_t)              :: array
      integer(kind=c_intptr_t), intent(in)  :: elements
      integer(kind=C_INT), intent(in)       :: flag
      logical :: success

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_host_register(array, elements, flag)
      endif

      if (use_gpu_vendor == amd_gpu) then
        success = hip_host_register(array, elements, flag)
      endif

    end function
    
    function gpu_memcpy(dst, src, size, dir) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions
      implicit none
      integer(kind=C_intptr_t)              :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=c_intptr_t), intent(in)  :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memcpy(dst, src, size, dir)
      endif

      if (use_gpu_vendor == amd_gpu) then
        success = hip_memcpy(dst, src, size, dir)
      endif
    
    end function


    function gpu_memset(a, val, size) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions
      implicit none
      integer(kind=c_intptr_t)                :: a
      integer(kind=ik)                        :: val
      integer(kind=c_intptr_t), intent(in)      :: size
      integer(kind=C_INT)                     :: istat

      logical :: success

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memset(a, val, size)
      endif

      if (use_gpu_vendor == amd_gpu) then
        success = hip_memset(a, val, size)
      endif

    end function

    function gpu_free(a) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions
      implicit none
      integer(kind=c_intptr_t)                :: a

      logical :: success

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_free(a)
      endif

      if (use_gpu_vendor == amd_gpu) then
        success = hip_free(a)
      endif

    end function

    function gpu_free_host(a) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions
      implicit none
      type(c_ptr), value          :: a

      logical :: success

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_free_host(a)
      endif

      if (use_gpu_vendor == amd_gpu) then
        success = hip_host_free(a)
      endif

    end function

    function gpu_host_unregister(a) result(success)
      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions
      implicit none
      integer(kind=c_intptr_t)                :: a

      logical :: success

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_host_unregister(a)
      endif

      if (use_gpu_vendor == amd_gpu) then
        success = hip_host_unregister(a)
      endif

    end function


    function gpu_memcpy2d(dst, dpitch, src, spitch, width, height , dir) result(success)

      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions

      implicit none

      integer(kind=C_intptr_T)           :: dst
      integer(kind=c_intptr_t), intent(in) :: dpitch
      integer(kind=C_intptr_T)           :: src
      integer(kind=c_intptr_t), intent(in) :: spitch
      integer(kind=c_intptr_t), intent(in) :: width
      integer(kind=c_intptr_t), intent(in) :: height
      integer(kind=C_INT), intent(in)    :: dir
      logical                            :: success

      if (use_gpu_vendor == nvidia_gpu) then
        success = cuda_memcpy2d(dst, dpitch, src, spitch, width, height , dir)
      endif

      if (use_gpu_vendor == amd_gpu) then
        success = hip_memcpy2d(dst, dpitch, src, spitch, width, height , dir)
      endif
    end function

    subroutine gpublas_dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_DOUBLE)             :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y

      if (use_gpu_vendor == nvidia_gpu) then
        call cublas_dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
      endif

      if (use_gpu_vendor == amd_gpu) then
        call rocblas_dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
      endif
    end subroutine

    subroutine gpublas_sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_FLOAT)              :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y

      if (use_gpu_vendor == nvidia_gpu) then
        call cublas_sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
      endif

      if (use_gpu_vendor == amd_gpu) then
        call rocblas_sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
      endif

    end subroutine

    subroutine gpublas_zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_DOUBLE_COMPLEX)             :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y

      if (use_gpu_vendor == nvidia_gpu) then
        call cublas_zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
      endif

      if (use_gpu_vendor == amd_gpu) then
        call rocblas_zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
      endif

    end subroutine

    subroutine gpublas_cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_FLOAT_COMPLEX)              :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y

      if (use_gpu_vendor == nvidia_gpu) then
        call cublas_cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
      endif

      if (use_gpu_vendor == amd_gpu) then
        call rocblas_cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy)
      endif

    end subroutine

    subroutine gpublas_dgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_DOUBLE)             :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c

      if (use_gpu_vendor == nvidia_gpu) then
        call cublas_dgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)
      endif

      if (use_gpu_vendor == amd_gpu) then
        call rocblas_dgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)
      endif

    end subroutine 


    subroutine gpublas_sgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_FLOAT)              :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c

      if (use_gpu_vendor == nvidia_gpu) then
        call cublas_sgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)
      endif

      if (use_gpu_vendor == amd_gpu) then
        call rocblas_sgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)
      endif


    end subroutine

    subroutine gpublas_zgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)

      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_DOUBLE_COMPLEX)          :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c

      if (use_gpu_vendor == nvidia_gpu) then
        call cublas_zgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)
      endif

      if (use_gpu_vendor == amd_gpu) then
        call rocblas_zgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)
      endif

    end subroutine

    subroutine gpublas_cgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)

      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_FLOAT_COMPLEX)           :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c

      if (use_gpu_vendor == nvidia_gpu) then
        call cublas_cgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)
      endif

      if (use_gpu_vendor == amd_gpu) then
        call rocblas_cgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)
      endif
    end subroutine


    subroutine gpublas_dtrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)

      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE)             :: alpha
      integer(kind=C_intptr_T)        :: a, b

      if (use_gpu_vendor == nvidia_gpu) then
        call cublas_dtrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
      endif

      if (use_gpu_vendor == amd_gpu) then
        call rocblas_dtrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
      endif

    end subroutine


    subroutine gpublas_strmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)

      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT)              :: alpha
      integer(kind=C_intptr_T)        :: a, b

      if (use_gpu_vendor == nvidia_gpu) then
        call cublas_strmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
      endif

      if (use_gpu_vendor == amd_gpu) then
        call rocblas_strmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
      endif
    end subroutine



    subroutine gpublas_ztrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)

      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX)          :: alpha
      integer(kind=C_intptr_T)        :: a, b

      if (use_gpu_vendor == nvidia_gpu) then
        call cublas_ztrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
      endif

      if (use_gpu_vendor == amd_gpu) then
        call rocblas_ztrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
      endif
    end subroutine


    subroutine gpublas_ctrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)

      use, intrinsic :: iso_c_binding
      use cuda_functions
      use hip_functions

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX)           :: alpha
      integer(kind=C_intptr_T)        :: a, b

      if (use_gpu_vendor == nvidia_gpu) then
        call cublas_ctrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
      endif

      if (use_gpu_vendor == amd_gpu) then
        call rocblas_ctrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
      endif
    end subroutine


end module 
