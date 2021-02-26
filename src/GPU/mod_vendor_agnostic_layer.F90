#include "config-f90.h"
module elpa_gpu
  use iso_c_binding

  integer(kind=c_int), parameter :: nvidia_gpu = 1
  integer(kind=c_int), parameter :: amd_gpu = 2
  integer(kind=c_int), parameter :: no_gpu = -1
  integer(kind=c_int)            :: use_gpu_vendor
  integer(kind=c_int)            :: gpuHostRegisterDefault    
  integer(kind=c_int)            :: gpuMemcpyHostToDevice    
  integer(kind=c_int)            :: gpuMemcpyDeviceToHost   
  integer(kind=c_int)            :: gpuMemcpyDeviceToDevice
  integer(kind=c_int)            :: gpuHostRegisterMapped
  integer(kind=c_int)            :: gpuHostRegisterPortable

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


end module 
