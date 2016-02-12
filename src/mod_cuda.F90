#include "config-f90.h"
module cuda_functions
  use iso_c_binding
  use precision
  implicit none

  public

  integer(kind=ik) :: cudaMemcpyHostToDevice
  integer(kind=ik) :: cudaMemcpyDeviceToHost
  integer(kind=ik) :: cudaHostRegisterPortable
  integer(kind=ik) :: cudaHostRegisterMapped
  integer(kind=ik) :: cudaMemcpyDeviceToDevice

#ifdef DOUBLE_PRECISION_REAL
  integer(kind=c_size_t), parameter :: size_of_real_datatype    = 8_rk
#else
  integer(kind=c_size_t), parameter :: size_of_real_datatype    = 4_rk
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
  integer(kind=c_size_t), parameter :: size_of_complex_datatype = 16_ck
#else
  integer(kind=c_size_t), parameter :: size_of_complex_datatype = 8_ck
#endif

  ! functions to set and query the CUDA devices

  interface
    function cuda_setdevice_c(n) result(istat) &
             bind(C, name="cudaSetDeviceFromC")

      use iso_c_binding
      implicit none
      integer(kind=C_INT), value    :: n
      integer(kind=C_INT)           :: istat
    end function cuda_setdevice_c
  end interface

  interface
    function cuda_getdevicecount_c(n) result(istat) &
             bind(C, name="cudaGetDeviceCountFromC")
      use iso_c_binding
      implicit none
      integer(kind=C_INT), intent(out) :: n
      integer(kind=C_INT)              :: istat
    end function cuda_getdevicecount_c
  end interface

  interface
    function cuda_devicesynchronize_c()result(istat) &
             bind(C,name='cudaDeviceSynchronizeFromC')

      use iso_c_binding

      implicit none
      integer(kind=C_INT)                       :: istat

    end function cuda_devicesynchronize_c
  end interface


  ! functions to copy CUDA memory
  interface
    function cuda_memcpyDeviceToDevice_c() result(flag) &
             bind(C, name="cudaMemcpyDeviceToDeviceFromC")
      use iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function cuda_memcpyHostToDevice_c() result(flag) &
             bind(C, name="cudaMemcpyHostToDeviceFromC")
      use iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function cuda_memcpyDeviceToHost_c() result(flag) &
             bind(C, name="cudaMemcpyDeviceToHostFromC")
      use iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function cuda_hostRegisterPortable_c() result(flag) &
             bind(C, name="cudaHostRegisterPortableFromC")
      use iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function cuda_hostRegisterMapped_c() result(flag) &
             bind(C, name="cudaHostRegisterMappedFromC")
      use iso_c_binding
      implicit none
      integer(kind=c_int) :: flag
    end function
  end interface

  interface
    function cuda_memcpy_c(dst, src, size, dir) result(istat) &
             bind(C, name="cudaMemcpyFromC")

      use iso_c_binding

      implicit none
      integer(kind=C_intptr_t), value              :: dst
      integer(kind=C_intptr_t), value              :: src
      integer(kind=C_SIZE_T), intent(in), value    :: size
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=C_INT)                          :: istat

    end function cuda_memcpy_c
  end interface

  interface
    function cuda_memcpy2d_c(dst, dpitch, src, spitch, width, height , dir) result(istat) &
             bind(C, name="cudaMemcpy2dFromC")

      use iso_c_binding

      implicit none

      integer(kind=C_intptr_T), value              :: dst
      integer(kind=C_SIZE_T), intent(in), value    :: dpitch
      integer(kind=C_intptr_T), value              :: src
      integer(kind=C_SIZE_T), intent(in), value    :: spitch
      integer(kind=C_SIZE_T), intent(in), value    :: width
      integer(kind=C_SIZE_T), intent(in), value    :: height
      integer(kind=C_INT), intent(in), value       :: dir
      integer(kind=C_INT)                          :: istat

    end function cuda_memcpy2d_c
  end interface

  ! functions to allocate and free CUDA memory

  interface
    function cuda_free_c(a) result(istat) &
             bind(C, name="cudaFreeFromC")

      use iso_c_binding

      implicit none
      integer(kind=C_intptr_T), value  :: a
      integer(kind=C_INT)              :: istat

    end function cuda_free_c
  end interface

  interface
    function cuda_malloc_c(a, width_height) result(istat) &
             bind(C, name="cudaMallocFromC")

      use iso_c_binding
      implicit none

      integer(kind=C_intptr_T)                    :: a
      integer(kind=C_SIZE_T), intent(in), value   :: width_height
      integer(kind=C_INT)                         :: istat

    end function cuda_malloc_c
  end interface

  interface
    function cuda_memset_c(a, val, size) result(istat) &
             bind(C, name="cudaMemsetFromC")

      use iso_c_binding

      implicit none

      integer(kind=C_intptr_T), value            :: a
      integer(kind=C_INT), value                 :: val
      integer(kind=C_SIZE_T), intent(in), value  :: size
      integer(kind=C_INT)                        :: istat

    end function cuda_memset_c
  end interface

  ! cuBLAS
  interface
    subroutine cublas_dgemm_c(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) bind(C,name='cublasDgemm')

      use iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_DOUBLE),value               :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
    end subroutine cublas_dgemm_c
  end interface

  interface
    subroutine cublas_sgemm_c(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) bind(C,name='cublasSgemm')

      use iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: cta, ctb
      integer(kind=C_INT),value               :: m,n,k
      integer(kind=C_INT), intent(in), value  :: lda,ldb,ldc
      real(kind=C_FLOAT),value                :: alpha,beta
      integer(kind=C_intptr_T), value         :: a, b, c
    end subroutine cublas_sgemm_c
  end interface

  interface
    subroutine cublas_dtrmm_c(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) bind(C,name='cublasDtrmm')

      use iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_DOUBLE), value              :: alpha
      integer(kind=C_intptr_T), value         :: a, b
    end subroutine cublas_dtrmm_c
  end interface

  interface
    subroutine cublas_strmm_c(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) bind(C,name='cublasStrmm')

      use iso_c_binding

      implicit none
      character(1,C_CHAR),value               :: side, uplo, trans, diag
      integer(kind=C_INT),value               :: m,n
      integer(kind=C_INT), intent(in), value  :: lda,ldb
      real(kind=C_FLOAT), value               :: alpha
      integer(kind=C_intptr_T), value         :: a, b
    end subroutine cublas_strmm_c
  end interface

  interface
    subroutine cublas_zgemm_c(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc) bind(C,name='cublasZgemm')

      use iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: cta, ctb
      integer(kind=C_INT),value              :: m,n,k
      integer(kind=C_INT), intent(in), value :: lda,ldb,ldc
      complex(kind=C_DOUBLE),value           :: alpha,beta
      integer(kind=C_intptr_T), value        :: a, b, c

    end subroutine cublas_zgemm_c
  end interface

  interface
    subroutine cublas_cgemm_c(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc) bind(C,name='cublasCgemm')

      use iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: cta, ctb
      integer(kind=C_INT),value              :: m,n,k
      integer(kind=C_INT), intent(in), value :: lda,ldb,ldc
      complex(kind=C_FLOAT),value            :: alpha,beta
      integer(kind=C_intptr_T), value        :: a, b, c

    end subroutine cublas_cgemm_c
  end interface

  interface
    subroutine cublas_ztrmm_c(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) bind(C,name='cublasZtrmm')

      use iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: side, uplo, trans, diag
      integer(kind=C_INT),value              :: m,n
      integer(kind=C_INT), intent(in), value :: lda,ldb
      complex(kind=C_DOUBLE), value          :: alpha
      integer(kind=C_intptr_T), value        :: a, b

    end subroutine cublas_ztrmm_c
  end interface

  interface
    subroutine cublas_ctrmm_c(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) bind(C,name='cublasCtrmm')

      use iso_c_binding

      implicit none
      character(1,C_CHAR),value              :: side, uplo, trans, diag
      integer(kind=C_INT),value              :: m,n
      integer(kind=C_INT), intent(in), value :: lda,ldb
      complex(kind=C_FLOAT), value           :: alpha
      integer(kind=C_intptr_T), value        :: a, b

    end subroutine cublas_ctrmm_c
  end interface


  contains

    ! functions to set and query the CUDA devices

    function cuda_setdevice(n) result(success)
      use iso_c_binding

      implicit none

      integer(kind=ik), intent(in)  :: n
      logical                       :: success
#ifdef WITH_GPU_VERSION
      success = cuda_setdevice_c(int(n,kind=c_int)) /= 0
#else
      success = .true.
#endif
    end function cuda_setdevice

    function cuda_getdevicecount(n) result(success)
      use iso_c_binding
      implicit none

      integer(kind=ik)     :: n
      integer(kind=c_int)  :: nCasted
      logical              :: success
#ifdef WITH_GPU_VERSION
      success = cuda_getdevicecount_c(nCasted) /=0
      n = int(nCasted)
#else
      success = .true.
      n = 0
#endif
    end function cuda_getdevicecount

    function cuda_devicesynchronize()result(success)

      use iso_c_binding

      implicit none
      logical :: success
#ifdef WITH_GPU_VERSION
      success = cuda_devicesynchronize_c() /=0
#else
      success = .false.
#endif
    end function cuda_devicesynchronize
    ! functions to allocate and free memory

    function cuda_malloc(a, width_height) result(success)

     use iso_c_binding
     implicit none

     integer(kind=C_intptr_t)                  :: a
     integer(kind=C_SIZE_T), intent(in)        :: width_height
     logical                                   :: success
#ifdef WITH_GPU_VERSION
     success = cuda_malloc_c(a, width_height) /= 0
#else
     success = .false.
#endif
   end function

   function cuda_free(a) result(success)

     use iso_c_binding

     implicit none
     integer(kind=C_intptr_T) :: a
     logical                  :: success
#ifdef WITH_GPU_VERSION
     success = cuda_free_c(a) /= 0
#else
     success = .false.
#endif
   end function cuda_free

 function cuda_memset(a, val, size) result(success)

   use iso_c_binding

   implicit none

   integer(kind=c_intptr_t)                :: a
   integer(kind=ik)                        :: val
   integer(kind=c_size_t), intent(in)      :: size
   integer(kind=C_INT)                     :: istat

   logical :: success
#ifdef WITH_GPU_VERSION
   success= cuda_memset_c(a, int(val,kind=c_int), int(size,kind=c_size_t)) /=0
#else
   success = .false.
#endif
 end function cuda_memset

 ! functions to memcopy CUDA memory

 function cuda_memcpyDeviceToDevice() result(flag)
   use iso_c_binding
   implicit none
   integer(kind=ik) :: flag
#ifdef WITH_GPU_VERSION
   flag = int(cuda_memcpyDeviceToDevice_c())
#else
   flag = 0
#endif
 end function

 function cuda_memcpyHostToDevice() result(flag)
   use iso_c_binding
   use precision
   implicit none
   integer(kind=ik) :: flag
#ifdef WITH_GPU_VERSION
   flag = int(cuda_memcpyHostToDevice_c())
#else
   flag = 0
#endif
 end function

 function cuda_memcpyDeviceToHost() result(flag)
   use iso_c_binding
   use precision
   implicit none
   integer(kind=ik) :: flag
#ifdef WITH_GPU_VERSION
   flag = int( cuda_memcpyDeviceToHost_c())
#else
   flag = 0
#endif
 end function

 function cuda_hostRegisterPortable() result(flag)
   use iso_c_binding
   use precision
   implicit none
   integer(kind=ik) :: flag
#ifdef WITH_GPU_VERSION
   flag = int(cuda_hostRegisterPortable_c())
#else
   flag = 0
#endif
 end function

 function cuda_hostRegisterMapped() result(flag)
   use iso_c_binding
   use precision
   implicit none
   integer(kind=ik) :: flag
#ifdef WITH_GPU_VERSION
   flag = int(cuda_hostRegisterMapped_c())
#else
   flag = 0
#endif
 end function


 function cuda_memcpy(dst, src, size, dir) result(success)

      use iso_c_binding

      implicit none
      integer(kind=C_intptr_t)              :: dst
      integer(kind=C_intptr_t)              :: src
      integer(kind=C_SIZE_T), intent(in)    :: size
      integer(kind=C_INT), intent(in)       :: dir
      logical :: success
#ifdef WITH_GPU_VERSION
        success = cuda_memcpy_c(dst, src, size, dir) /= 0
#else
        success = .false.
#endif
    end function

    function cuda_memcpy2d(dst, dpitch, src, spitch, width, height , dir) result(success)

      use iso_c_binding

      implicit none

      integer(kind=C_intptr_T)           :: dst
      integer(kind=C_SIZE_T), intent(in) :: dpitch
      integer(kind=C_intptr_T)           :: src
      integer(kind=C_SIZE_T), intent(in) :: spitch
      integer(kind=C_SIZE_T), intent(in) :: width
      integer(kind=C_SIZE_T), intent(in) :: height
      integer(kind=C_INT), intent(in)    :: dir
      logical                            :: success
#ifdef WITH_GPU_VERSION
      success = cuda_memcpy2d_c(dst, dpitch, src, spitch, width, height , dir) /= 0
#else
      success = .false.
#endif
    end function cuda_memcpy2d

    ! cuBLAS
    subroutine cublas_dgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
      use iso_c_binding

      implicit none
      character(1,C_CHAR)             :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_DOUBLE)             :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
#ifdef WITH_GPU_VERSION
      call cublas_dgemm_c(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine cublas_dgemm

    subroutine cublas_sgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
      use iso_c_binding

      implicit none
      character(1,C_CHAR)             :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      real(kind=C_FLOAT)              :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
#ifdef WITH_GPU_VERSION
      call cublas_sgemm_c(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
#endif
    end subroutine cublas_sgemm

    subroutine cublas_dtrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)

      use iso_c_binding

      implicit none
      character(1,C_CHAR)             :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE)             :: alpha
      integer(kind=C_intptr_T)        :: a, b
#ifdef WITH_GPU_VERSION
      call cublas_dtrmm_c(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine cublas_dtrmm

    subroutine cublas_strmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)

      use iso_c_binding

      implicit none
      character(1,C_CHAR)             :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT)              :: alpha
      integer(kind=C_intptr_T)        :: a, b
#ifdef WITH_GPU_VERSION
      call cublas_strmm_c(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine cublas_strmm

    subroutine cublas_zgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)

      use iso_c_binding

      implicit none
      character(1,C_CHAR)             :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_DOUBLE)          :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
#ifdef WITH_GPU_VERSION
      call cublas_zgemm_c(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)
#endif
    end subroutine cublas_zgemm

    subroutine cublas_cgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)

      use iso_c_binding

      implicit none
      character(1,C_CHAR)             :: cta, ctb
      integer(kind=C_INT)             :: m,n,k
      integer(kind=C_INT), intent(in) :: lda,ldb,ldc
      complex(kind=C_FLOAT)           :: alpha,beta
      integer(kind=C_intptr_T)        :: a, b, c
#ifdef WITH_GPU_VERSION
      call cublas_cgemm_c(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc)
#endif
    end subroutine cublas_cgemm

    subroutine cublas_ztrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)

      use iso_c_binding

      implicit none
      character(1,C_CHAR)             :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE)          :: alpha
      integer(kind=C_intptr_T)        :: a, b
#ifdef WITH_GPU_VERSION
      call cublas_ztrmm_c(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine cublas_ztrmm

    subroutine cublas_ctrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)

      use iso_c_binding

      implicit none
      character(1,C_CHAR)             :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT)           :: alpha
      integer(kind=C_intptr_T)        :: a, b
#ifdef WITH_GPU_VERSION
      call cublas_ctrmm_c(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
#endif
    end subroutine cublas_ctrmm


end module cuda_functions
