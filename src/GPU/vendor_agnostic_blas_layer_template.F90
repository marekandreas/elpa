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
!
! This file is the generated version. Do NOT edit
#endif


  interface gpublas_Dcopy
    module procedure gpublas_Dcopy_intptr
    module procedure gpublas_Dcopy_cptr
  end interface

  interface gpublas_Dtrmm
    module procedure gpublas_Dtrmm_intptr
    module procedure gpublas_Dtrmm_cptr
  end interface
  
  interface gpublas_Dtrsm
    module procedure gpublas_Dtrsm_intptr
    module procedure gpublas_Dtrsm_cptr
  end interface
  
  interface gpublas_Dgemm
    module procedure gpublas_Dgemm_intptr
    module procedure gpublas_Dgemm_cptr
    module procedure gpublas_Dgemm_intptr_cptr_intptr
  end interface

  interface gpublas_Dscal
    module procedure gpublas_Dscal_intptr
    module procedure gpublas_Dscal_cptr
  end interface

  interface gpublas_Ddot
    module procedure gpublas_Ddot_intptr
    module procedure gpublas_Ddot_cptr
  end interface

  interface gpublas_Daxpy
    module procedure gpublas_Daxpy_intptr
    module procedure gpublas_Daxpy_cptr
  end interface

  interface gpublas_Scopy
    module procedure gpublas_Scopy_intptr
    module procedure gpublas_Scopy_cptr
  end interface

  interface gpublas_Strmm
    module procedure gpublas_Strmm_intptr
    module procedure gpublas_Strmm_cptr
  end interface
  
  interface gpublas_Strsm
    module procedure gpublas_Strsm_intptr
    module procedure gpublas_Strsm_cptr
  end interface
  
  interface gpublas_Sgemm
    module procedure gpublas_Sgemm_intptr
    module procedure gpublas_Sgemm_cptr
    module procedure gpublas_Sgemm_intptr_cptr_intptr
  end interface

  interface gpublas_Sscal
    module procedure gpublas_Sscal_intptr
    module procedure gpublas_Sscal_cptr
  end interface

  interface gpublas_Sdot
    module procedure gpublas_Sdot_intptr
    module procedure gpublas_Sdot_cptr
  end interface

  interface gpublas_Saxpy
    module procedure gpublas_Saxpy_intptr
    module procedure gpublas_Saxpy_cptr
  end interface

  interface gpublas_Zcopy
    module procedure gpublas_Zcopy_intptr
    module procedure gpublas_Zcopy_cptr
  end interface

  interface gpublas_Ztrmm
    module procedure gpublas_Ztrmm_intptr
    module procedure gpublas_Ztrmm_cptr
  end interface
  
  interface gpublas_Ztrsm
    module procedure gpublas_Ztrsm_intptr
    module procedure gpublas_Ztrsm_cptr
  end interface
  
  interface gpublas_Zgemm
    module procedure gpublas_Zgemm_intptr
    module procedure gpublas_Zgemm_cptr
    module procedure gpublas_Zgemm_intptr_cptr_intptr
  end interface

  interface gpublas_Zscal
    module procedure gpublas_Zscal_intptr
    module procedure gpublas_Zscal_cptr
  end interface

  interface gpublas_Zdot
    module procedure gpublas_Zdot_intptr
    module procedure gpublas_Zdot_cptr
  end interface

  interface gpublas_Zaxpy
    module procedure gpublas_Zaxpy_intptr
    module procedure gpublas_Zaxpy_cptr
  end interface

  interface gpublas_Ccopy
    module procedure gpublas_Ccopy_intptr
    module procedure gpublas_Ccopy_cptr
  end interface

  interface gpublas_Ctrmm
    module procedure gpublas_Ctrmm_intptr
    module procedure gpublas_Ctrmm_cptr
  end interface
  
  interface gpublas_Ctrsm
    module procedure gpublas_Ctrsm_intptr
    module procedure gpublas_Ctrsm_cptr
  end interface
  
  interface gpublas_Cgemm
    module procedure gpublas_Cgemm_intptr
    module procedure gpublas_Cgemm_cptr
    module procedure gpublas_Cgemm_intptr_cptr_intptr
  end interface

  interface gpublas_Cscal
    module procedure gpublas_Cscal_intptr
    module procedure gpublas_Cscal_cptr
  end interface

  interface gpublas_Cdot
    module procedure gpublas_Cdot_intptr
    module procedure gpublas_Cdot_cptr
  end interface

  interface gpublas_Caxpy
    module procedure gpublas_Caxpy_intptr
    module procedure gpublas_Caxpy_cptr
  end interface


  contains


      function gpublas_get_version(handle, version) result(success)
      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none

      integer(kind=c_intptr_t), intent(in)  :: handle
      integer(kind=c_int),      intent(out) :: version
      logical                               :: success

      success = .true.
#ifdef WITH_NVIDIA_GPU_VERSION
#ifdef WITH_GPU_STREAMS
      if (use_gpu_vendor == nvidia_gpu) then
        success = cublas_get_version(handle, version)
      endif
#endif
#endif
#ifdef WITH_AMD_GPU_VERSION
#ifdef WITH_GPU_STREAMS
      if (use_gpu_vendor == amd_gpu) then
        print *,"gpublasGetVersion not implemented for amd"
        stop 1
      endif
#endif
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"gpublasGetVersion not implemented for openmp offload"
        stop 1
      endif
#endif
#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"gpublasGetVersion not implemented for sycl"
        stop 1
      endif
#endif
    end function

    function gpublas_set_stream(handle, stream) result(success)
      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none

      integer(kind=c_intptr_t), intent(in)  :: handle
      integer(kind=c_intptr_t), intent(in)  :: stream
      logical                               :: success


      success = .true.
#ifdef WITH_NVIDIA_GPU_VERSION
#ifdef WITH_GPU_STREAMS
      if (use_gpu_vendor == nvidia_gpu) then
        success = cublas_set_stream(handle, stream)
      endif
#endif
#endif
#ifdef WITH_AMD_GPU_VERSION
#ifdef WITH_GPU_STREAMS
      if (use_gpu_vendor == amd_gpu) then
        success = rocblas_set_stream(handle, stream)
      endif
#endif
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        print *,"gpublasSetStream not implemented for openmp offload"
        stop 1
      endif
#endif
#ifdef WITH_SYCL_GPU_VERSION
      if (use_gpu_vendor == sycl_gpu) then
        print *,"gpublasSetStream not implemented for sycl"
        stop 1
      endif
#endif

    end function


    subroutine gpublas_Dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_DOUBLE)             :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
         if (use_gpu_vendor == openmp_offload_gpu) then
           call mkl_openmp_offload_Dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
         if (use_gpu_vendor == sycl_gpu) then
           call syclblas_Dgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif
    end subroutine

    subroutine gpublas_Dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_DOUBLE)             :: alpha, beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Dgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_DOUBLE)             :: alpha, beta
      type(c_ptr)                     :: a, b, c
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Dgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Dgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, lda, &
                                            b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_DOUBLE)             :: alpha, beta
      integer(kind=C_intptr_T)        :: a, c
      type(c_ptr)                     :: b
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, &
                                                           lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, &
                                                            lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Dgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, &
                                                   lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Dgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, &
                                                             lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Dcopy_intptr(n, x, incx, y, incy, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dcopy_intptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Dcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Dcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Dcopy_cptr(n, x, incx, y, incy, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dcopy_cptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Dcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Dcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE)             :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Dtrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

    end subroutine


    subroutine gpublas_Dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE)             :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Dtrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE)             :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Dtrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_DOUBLE)             :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Dtrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine
    subroutine gpublas_Sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      real(kind=C_FLOAT)              :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
         if (use_gpu_vendor == openmp_offload_gpu) then
           call mkl_openmp_offload_Sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
         if (use_gpu_vendor == sycl_gpu) then
           call syclblas_Sgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif
    end subroutine

    subroutine gpublas_Sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_FLOAT)              :: alpha, beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Sgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_FLOAT)              :: alpha, beta
      type(c_ptr)                     :: a, b, c
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Sgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Sgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, lda, &
                                            b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      real(kind=C_FLOAT)              :: alpha, beta
      integer(kind=C_intptr_T)        :: a, c
      type(c_ptr)                     :: b
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Sgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, &
                                                           lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Sgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, &
                                                            lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Sgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, &
                                                   lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Sgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, &
                                                             lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Scopy_intptr(n, x, incx, y, incy, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Scopy_intptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Scopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Scopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Scopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Scopy_cptr(n, x, incx, y, incy, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Scopy_cptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Scopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Scopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Scopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT)              :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Strmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

    end subroutine


    subroutine gpublas_Strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT)              :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Strmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT)              :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Strsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      real(kind=C_FLOAT)              :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Strsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine
    subroutine gpublas_Zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_double_complex)  :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
         if (use_gpu_vendor == openmp_offload_gpu) then
           call mkl_openmp_offload_Zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
         if (use_gpu_vendor == sycl_gpu) then
           call syclblas_Zgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif
    end subroutine

    subroutine gpublas_Zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha, beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Zgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha, beta
      type(c_ptr)                     :: a, b, c
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Zgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Zgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, lda, &
                                            b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha, beta
      integer(kind=C_intptr_T)        :: a, c
      type(c_ptr)                     :: b
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, &
                                                           lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, &
                                                            lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Zgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, &
                                                   lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Zgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, &
                                                             lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Zcopy_intptr(n, x, incx, y, incy, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zcopy_intptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Zcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Zcopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Zcopy_cptr(n, x, incx, y, incy, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zcopy_cptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Zcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Zcopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Ztrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

    end subroutine


    subroutine gpublas_Ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Ztrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Ztrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Ztrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine
    subroutine gpublas_Cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,incx,incy
      complex(kind=C_float_complex)  :: alpha,beta
      integer(kind=C_intptr_T)        :: a, x, y
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
         if (use_gpu_vendor == openmp_offload_gpu) then
           call mkl_openmp_offload_Cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
         if (use_gpu_vendor == sycl_gpu) then
           call syclblas_Cgemv(cta, m, n, alpha, a, lda, x, incx, beta, y, incy, handle)
         endif
#endif
    end subroutine

    subroutine gpublas_Cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_FLOAT_COMPLEX)  :: alpha, beta
      integer(kind=C_intptr_T)        :: a, b, c
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Cgemm_intptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_FLOAT_COMPLEX)  :: alpha, beta
      type(c_ptr)                     :: a, b, c
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Cgemm_cptr(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Cgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, lda, &
                                            b, ldb, beta, c, ldc, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: cta, ctb
      integer(kind=C_INT)             :: m, n, k
      integer(kind=C_INT), intent(in) :: lda, ldb, ldc
      complex(kind=C_FLOAT_COMPLEX)  :: alpha, beta
      integer(kind=C_intptr_T)        :: a, c
      type(c_ptr)                     :: b
      integer(kind=c_intptr_t)        :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Cgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, &
                                                           lda, b, ldb, beta, c, ldc, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Cgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, &
                                                            lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Cgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, &
                                                   lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Cgemm_intptr_cptr_intptr(cta, ctb, m, n, k, alpha, a, &
                                                             lda, b, ldb, beta, c, ldc, handle)
        endif
#endif

    end subroutine 

    subroutine gpublas_Ccopy_intptr(n, x, incx, y, incy, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      integer(kind=C_intptr_T)        :: x, y
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ccopy_intptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ccopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ccopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Ccopy_intptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Ccopy_cptr(n, x, incx, y, incy, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=C_INT)             :: n
      integer(kind=C_INT), intent(in) :: incx, incy
      type(c_ptr)                     :: x, y
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ccopy_cptr(n, x, incx, y, incy, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ccopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ccopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Ccopy_cptr(n, x, incx, y, incy, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX)   :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Ctrmm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

    end subroutine


    subroutine gpublas_Ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX)   :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Ctrmm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX)   :: alpha
      integer(kind=C_intptr_T)        :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Ctrsm_intptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_Ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: side, uplo, trans, diag
      integer(kind=C_INT)             :: m,n
      integer(kind=C_INT), intent(in) :: lda,ldb
      complex(kind=C_FLOAT_COMPLEX)   :: alpha
      type(c_ptr)                     :: a, b
      integer(kind=c_intptr_t)     :: handle

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          call mkl_openmp_offload_Ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          call syclblas_Ctrsm_cptr(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, handle)
        endif
#endif
    end subroutine

    subroutine gpublas_getPointerMode(gpublasHandle, mode)

      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: mode

        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_getPointerMode(gpublasHandle, mode)
        endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_getPointerMode(gpublasHandle, mode)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "getPointer mode not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "getPointer mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_setPointerMode(gpublasHandle, mode)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: mode

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_setPointerMode(gpublasHandle, mode)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_setPointerMode(gpublasHandle, mode)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "setPointer mode not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "setPointer mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Ddot_intptr(gpublasHandle, length, x, incx, y, incy, z)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      integer(kind=c_intptr_t)          :: x, y, z

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ddot_intptr(gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ddot_intptr(gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xdot not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xdot mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Ddot_cptr(gpublasHandle, length, x, incx, y, incy, z)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      type(c_ptr)                       :: x, y, z

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Ddot_cptr(gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Ddot_cptr(gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xdot not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xdot mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Dscal_intptr(gpublasHandle, length, alpha, x, incx)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx
      real(kind=C_DOUBLE)             :: alpha
      integer(kind=c_intptr_t)           :: x

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dscal_intptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dscal_intptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Dscal_cptr(gpublasHandle, length, alpha, x, incx)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx
      real(kind=C_DOUBLE)             :: alpha
      type(c_ptr)                       :: x

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Dscal_cptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Dscal_cptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine


    subroutine gpublas_Daxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      real(kind=C_DOUBLE)             :: alpha
      integer(kind=c_intptr_t)           :: x, y

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Daxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Daxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Daxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      real(kind=C_DOUBLE)             :: alpha
      type(c_ptr)                       :: x, y

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Daxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Daxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine
    subroutine gpublas_Sdot_intptr(gpublasHandle, length, x, incx, y, incy, z)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      integer(kind=c_intptr_t)          :: x, y, z

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Sdot_intptr(gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Sdot_intptr(gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xdot not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xdot mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Sdot_cptr(gpublasHandle, length, x, incx, y, incy, z)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      type(c_ptr)                       :: x, y, z

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Sdot_cptr(gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Sdot_cptr(gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xdot not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xdot mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Sscal_intptr(gpublasHandle, length, alpha, x, incx)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx
      real(kind=C_FLOAT)              :: alpha
      integer(kind=c_intptr_t)           :: x

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Sscal_intptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Sscal_intptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Sscal_cptr(gpublasHandle, length, alpha, x, incx)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx
      real(kind=C_FLOAT)              :: alpha
      type(c_ptr)                       :: x

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Sscal_cptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Sscal_cptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine


    subroutine gpublas_Saxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      real(kind=C_FLOAT)              :: alpha
      integer(kind=c_intptr_t)           :: x, y

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Saxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Saxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Saxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      real(kind=C_FLOAT)              :: alpha
      type(c_ptr)                       :: x, y

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Saxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Saxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine
    subroutine gpublas_Zdot_intptr(conj, gpublasHandle, length, x, incx, y, incy, z)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value         :: conj
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      integer(kind=c_intptr_t)          :: x, y, z

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zdot_intptr(conj, gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zdot_intptr(conj, gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xdot not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xdot mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Zdot_cptr(conj, gpublasHandle, length, x, incx, y, incy, z)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value         :: conj
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      type(c_ptr)                       :: x, y, z

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zdot_cptr(conj, gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zdot_cptr(conj, gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xdot not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xdot mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Zscal_intptr(gpublasHandle, length, alpha, x, incx)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      integer(kind=c_intptr_t)           :: x

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zscal_intptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zscal_intptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Zscal_cptr(gpublasHandle, length, alpha, x, incx)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      type(c_ptr)                       :: x

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zscal_cptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zscal_cptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine


    subroutine gpublas_Zaxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      integer(kind=c_intptr_t)           :: x, y

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zaxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zaxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Zaxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      complex(kind=C_DOUBLE_COMPLEX)  :: alpha
      type(c_ptr)                       :: x, y

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Zaxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Zaxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine
    subroutine gpublas_Cdot_intptr(conj, gpublasHandle, length, x, incx, y, incy, z)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value         :: conj
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      integer(kind=c_intptr_t)          :: x, y, z

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Cdot_intptr(conj, gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Cdot_intptr(conj, gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xdot not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xdot mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Cdot_cptr(conj, gpublasHandle, length, x, incx, y, incy, z)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value         :: conj
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      type(c_ptr)                       :: x, y, z

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Cdot_cptr(conj, gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Cdot_cptr(conj, gpublasHandle, length, x, incx, y, incy, z)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xdot not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xdot mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Cscal_intptr(gpublasHandle, length, alpha, x, incx)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx
      complex(kind=C_FLOAT_COMPLEX)  :: alpha
      integer(kind=c_intptr_t)           :: x

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Cscal_intptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Cscal_intptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Cscal_cptr(gpublasHandle, length, alpha, x, incx)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx
      complex(kind=C_FLOAT_COMPLEX)  :: alpha
      type(c_ptr)                       :: x

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Cscal_cptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Cscal_cptr(gpublasHandle, length, alpha, x, incx)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine


    subroutine gpublas_Caxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      complex(kind=C_FLOAT_COMPLEX)  :: alpha
      integer(kind=c_intptr_t)           :: x, y

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Caxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Caxpy_intptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine

    subroutine gpublas_Caxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)

      use, intrinsic :: iso_c_binding
#ifdef WITH_NVIDIA_GPU_VERSION
      use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)          :: gpublasHandle
      integer(kind=c_int)               :: length, incx, incy
      complex(kind=C_FLOAT_COMPLEX)  :: alpha
      type(c_ptr)                       :: x, y

#ifdef WITH_NVIDIA_GPU_VERSION
        if (use_gpu_vendor == nvidia_gpu) then
          call cublas_Caxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_AMD_GPU_VERSION
        if (use_gpu_vendor == amd_gpu) then
          call rocblas_Caxpy_cptr(gpublasHandle, length, alpha, x, incx, y, incy)
        endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
        if (use_gpu_vendor == openmp_offload_gpu) then
          print *, "Xscal not yet implemented for openmp_offload"
        endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
        if (use_gpu_vendor == sycl_gpu) then
          print *, "Xscal mode not yet implemented for sycl"
        endif
#endif
    end subroutine
