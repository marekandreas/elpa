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
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaftrn,
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!      and
!    - IBM Deutschland GmbH
!
!
!    More information can be found here:
!    http://elpa.rzg.mpg.de/
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

!This is a module contains all CUDA C Calls
! it was provided by NVIDIA with their ELPA GPU port and
! adapted for an ELPA release by A.Marek, RZG

#include "config-f90.h"

module cuda_routines
  implicit none

  public

  integer, parameter :: cudaMemcpyHostToDevice  = 1
  integer, parameter :: cudaMemcpyDeviceToHost  = 2
  integer, parameter :: cudaHostRegisterPortable  = 3
  integer, parameter :: cudaHostRegisterMapped  = 4
  integer, parameter :: cudaMemcpyDeviceToDevice = 5

  interface
    function cuda_setdevice_c(n) result(istat) &
             bind(C, name="cudaSetDeviceFromC")

      use iso_c_binding
      implicit none
      integer(C_INT), value    :: n
      integer(C_INT)           :: istat
    end function cuda_setdevice_c
  end interface

  interface
    function cuda_getdevicecount_c(n) result(istat) &
             bind(C, name="cudaGetDeviceCountFromC")
      use iso_c_binding
      implicit none
      integer(C_INT), intent(out) :: n
      integer(C_INT)              :: istat
    end function cuda_getdevicecount_c
  end interface

!    function cuda_ProfilerStart() result(istat)&
!            bind (C, name="cudaProfilerStart")
!
!      use iso_c_binding
!      implicit none
!      integer(c_int)             :: istat
!    end function cuda_ProfilerStart
!
!    function cuda_ProfilerStop() result(istat)&
!             bind (C, name="cudaProfilerStop")
!
!      use iso_c_binding
!      implicit none
!
!      integer(c_int)             :: istat
!    end function cuda_ProfilerStop


   !*********************Allocate 1D Memory on Device******
   interface
     function cuda_malloc_c(a, width_height) result(istat) &
              bind(C, name="cudaMallocFromC")

       use iso_c_binding
       implicit none

       integer(C_SIZE_T)                    :: a
       integer(C_SIZE_T), intent(in), value :: width_height
       integer(C_INT)                       :: istat

     end function cuda_malloc_c
   end interface

   !******************* Allocate pinned memory***********

!   function cuda_hostalloc(a, size) result(istat) &
!            bind(C, name="cudaHostAlloc")
!
!     use iso_c_binding
!     implicit none
!
!     integer(C_SIZE_T)                    :: a
!     integer(C_SIZE_T), intent(in), value :: size
!     integer(C_INT)                       :: istat
!
!   end function cuda_hostalloc
!
!   function cuda_hostregister(a, size, dir) result(istat) &
!           bind(C, name="cudaHostRegister")
!
!     use iso_c_binding
!
!     implicit none
!     integer(C_SIZE_T)                    :: a
!     integer(C_SIZE_T), intent(in), value :: size
!     integer(C_INT), intent(in),value     :: dir
!     integer(C_INT)                       :: istat
!   end function cuda_hostregister
!
!   !******************* Alloacte 2D Memory on Device*****
!
!   function cuda_malloc_2d(a, width_height) result( istat) &
!            bind(C, name="cudaMalloc2d")
!
!     use iso_c_binding
!
!     implicit none
!     integer(C_SIZE_T)                    :: a
!     integer(C_SIZE_T), intent(in), value :: width_height
!     integer(C_INT)                       :: istat
!
!   end function cuda_malloc_2d
!
!   !******************* Alloacte 2D Memory on Device for coalesed access *****
!
!   function cuda_malloc2d_pitch(a, pitch,width, height) result( istat) &
!            bind(C, name="cudaMallocPitch")
!
!     use iso_c_binding
!
!     implicit none
!     integer(C_SIZE_T)         :: a
!     integer(C_SIZE_T)         :: pitch
!     integer(C_SIZE_T), value  :: width
!     integer(C_SIZE_T), value  :: height
!     integer(C_INT)            :: istat
!
!   end function cuda_malloc2d_pitch
!
!  !******************* Alloacte 3D Memory on Device*****
!
!  function cuda_malloc_3d(a,width_height_depth) result( istat) &
!           bind(C, name="cudaMalloc3d")
!
!    use iso_c_binding
!
!    implicit none
!
!    integer(C_SIZE_T)         :: a
!    integer(C_SIZE_T), value  :: width_height_depth
!    integer(C_INT)            :: istat
!
!  end function cuda_malloc_3d

  !*************Deallocate Device Memory*****************
  interface
    function cuda_free_c(a) result(istat) &
             bind(C, name="cudaFreeFromC")

      use iso_c_binding

      implicit none
      integer(C_SIZE_T), value  :: a
      integer(C_INT)            :: istat

    end function cuda_free_c
  end interface

!  function cuda_freehost(a) result(istat) &
!           bind(C, name="cudaFreeHost")
!
!    use iso_c_binding
!
!    implicit none
!    integer(C_SIZE_T)   :: a
!    integer(C_INT)      :: istat
!  end function cuda_freehost

  !*************Copy Data from device to host / host to device**********************************
  interface
    function cuda_memcpy_c(dst, src, size, dir) result(istat) &
             bind(C, name="cudaMemcpyFromC")

      use iso_c_binding

      implicit none
      integer(C_SIZE_T), value              :: dst
      integer(C_SIZE_T), value              :: src
      integer(C_SIZE_T), intent(in), value  :: size
      integer(C_INT), intent(in), value     :: dir
      integer(C_INT)                        :: istat

    end function cuda_memcpy_c
  end interface

!  function cuda_d2d(val) result(istat)&
!           bind(C, name="cuda_MemcpyDeviceToDevice")
!
!    use iso_c_binding
!
!    implicit none
!    integer(C_INT), value   :: val
!    integer(C_INT)          :: istat
!  end function cuda_d2d

  !******************Copy Data from device to host / host to device for 2D*******
  interface
    function cuda_memcpy2d_c(dst, dpitch, src, spitch, width, height , dir) result(istat) &
             bind(C, name="cudaMemcpy2dFromC")

      use iso_c_binding

      implicit none

      integer(C_SIZE_T), value              :: dst
      integer(C_SIZE_T), intent(in), value  :: dpitch
      integer(C_SIZE_T), value              :: src
      integer(C_SIZE_T), intent(in), value  :: spitch
      integer(C_SIZE_T), intent(in), value  :: width
      integer(C_SIZE_T), intent(in), value  :: height
      integer(C_INT), intent(in), value     :: dir
      integer(C_INT)                        :: istat

    end function cuda_memcpy2d_c
  end interface
  !**************************Copy data to device memory Async*****************

!  function cuda_memcpy2dasync( dst, dpitch, src, spitch, width, height , dir, stream) result(istat) &
!           bind(C, name="cudaMemcpy2DAsync")
!
!    use iso_c_binding
!
!    implicit none
!
!    integer(C_SIZE_T), value                :: dst
!    integer(C_SIZE_T), intent(in), value       :: dpitch
!    integer(C_SIZE_T), value                :: src
!    integer(C_SIZE_T), intent(in), value       :: spitch
!    integer(C_SIZE_T), intent(in), value    :: width
!    integer(C_SIZE_T), intent(in), value    :: height
!    integer(C_INT), intent(in), value       :: dir
!    integer(C_SIZE_T),value                 :: stream
!    integer(C_INT)                          :: istat
!  end function

  !***************Initialise memory***********************************************

  interface 
 function cuda_memset_c(a, val, size) result(istat) &
          bind(C, name="cudaMemsetFromC")

   use iso_c_binding

   implicit none

   integer(C_SIZE_T), value              :: a
   !integer(C_INT)                       :: val
   integer(C_INT)                        :: val
   integer(C_SIZE_T), intent(in), value  :: size
   integer(C_INT)                        :: istat

 end function cuda_memset_c
 end interface
!
!  function c_memset(a, val, size) result(istat) &
!           bind(C, name="memset")
!
!    use iso_c_binding
!
!    implicit none
!    integer(C_SIZE_T)                    :: a
!    integer(C_INT)                       :: val
!    integer(C_SIZE_T), intent(in), value :: size
!    integer(C_INT)                       :: istat
!  end function c_memset
!
!  !***************************** CUDA LIBRARY CALLS***************************!
!  subroutine cublas_dgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)bind(C,name='cublasDgemm')
!
!    use iso_c_binding
!
!    implicit none
!    character(1,C_CHAR),value            :: cta, ctb
!    integer(C_INT),value                 :: m,n,k
!    integer(C_INT), intent(in), value    :: lda,ldb,ldc
!    real(C_DOUBLE),value                 :: alpha,beta
!    integer(C_SIZE_T), value             :: a, b, c
!  end subroutine cublas_dgemm
!
!  subroutine cublas_dtrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) bind(C,name='cublasDtrmm')
!
!    use iso_c_binding
!
!    implicit none
!    character(1,C_CHAR),value            :: side, uplo, trans, diag
!    integer(C_INT),value                 :: m,n
!    integer(C_INT), intent(in), value    :: lda,ldb
!    real(C_DOUBLE), value                :: alpha
!    integer(C_SIZE_T), value             :: a, b
!  end subroutine cublas_dtrmm
!
!  subroutine cublas_zgemm(cta, ctb, m, n, k, alpha, a, lda, b, ldb, beta, c,ldc) bind(C,name='cublasZgemm')
!
!    use iso_c_binding
!
!    implicit none
!    character(1,C_CHAR),value            :: cta, ctb
!    integer(C_INT),value                 :: m,n,k
!    integer(C_INT), intent(in), value    :: lda,ldb,ldc
!    complex(C_DOUBLE),value              :: alpha,beta
!    integer(C_SIZE_T), value             :: a, b, c
!
!  end subroutine cublas_zgemm
!
!  subroutine cublas_zgemv( trans , m, n, alpha, a, lda, b, ldb, beta, c, ldc) bind(C,name='cublasZgemv')
!
!    use iso_c_binding
!
!    implicit none
!    character(1,c_char),value       :: trans
!    integer(c_int),value            :: m,n,lda,ldb,ldc
!    complex(c_double_complex),value :: alpha,beta
!    integer(c_size_t), value        :: a,b,c
!
!  end subroutine cublas_zgemv
!
!  subroutine cublas_zhemv( trans , m, alpha, a, lda, b, ldb, beta, c, ldc)bind(C,name='cublasZhemv')
!
!    use iso_c_binding
!
!    implicit none
!    character(1,c_char),value       :: trans
!    integer(c_int),value            :: m,lda,ldb,ldc
!    complex(c_double_complex),value :: alpha,beta
!    integer(c_size_t), value        :: a,b,c
!
!  end subroutine cublas_zhemv
!
!  subroutine cublas_ztrmm(side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb) bind(C,name='cublasZtrmm')
!
!    use iso_c_binding
!
!    implicit none
!    character(1,C_CHAR),value            :: side, uplo, trans, diag
!    integer(C_INT),value                 :: m,n
!    integer(C_INT), intent(in), value    :: lda,ldb
!    complex(C_DOUBLE), value             :: alpha
!    integer(C_SIZE_T), value             :: a, b
!
!  end subroutine cublas_ztrmm
!
!  subroutine cublas_zherk( uplo, trans, n, k, alpha, a, lda, beta, b, ldb) bind(C,name='cublasZherk')
!
!    use iso_c_binding
!
!    implicit none
!    character(1,C_CHAR),value            :: uplo, trans
!    integer(C_INT),value                 :: n, k
!    integer(C_INT), intent(in), value    :: lda,ldb
!    complex(C_DOUBLE), value             :: alpha, beta
!    integer(c_size_t),value              :: a,b
!
!  end subroutine cublas_zherk
!
!  subroutine cublas_ztrmv( uplo, trans, diag, n, a, lda, b, ldb)bind(C,name='cublasZtrmv')
!
!    use iso_c_binding
!
!    implicit none
!    character(1,C_CHAR),value            :: uplo, trans, diag
!    integer(C_INT),value                 :: n
!    integer(C_INT), intent(in), value    :: lda,ldb
!    integer(C_SIZE_T), value             :: a, b
!
!  end subroutine cublas_ztrmv
!
!
!  subroutine cublas_zher2( uplo, n, alpha, x, incx , y, incy , a, lda)bind(C,name='cublasZher2')
!
!    use iso_c_binding
!
!    implicit none
!    character(1,C_CHAR),value            :: uplo
!    integer(C_INT),value                 :: n
!    integer(C_INT), intent(in), value    :: lda,incx, incy
!    complex(C_DOUBLE_COMPLEX), value     :: alpha
!    integer(c_size_t),value              :: a,x,y
!
!  end subroutine cublas_zher2
!
!  function cuda_devicesynchronize()result(istat) &
!           bind(C,name='cudaDeviceSynchronize')
!
!    use iso_c_binding
!
!    implicit none
!    integer(C_INT)                       :: istat
!
!  end function cuda_devicesynchronize
!
!  function cuda_memcpyAsync( dst, src, d_size, dir,stream ) result(istat) &
!           bind(C, name="cudaMemcpyAsync")
!
!    use iso_c_binding
!
!    implicit none
!    integer(C_SIZE_T), value            :: dst
!    integer(C_SIZE_T), value            :: src
!    integer(C_SIZE_T),intent(in),value  :: d_size
!    integer(C_INT),intent(in),value     :: dir
!    integer(c_size_t),value             :: stream
!    integer(C_INT)                      :: istat
!
!  end function
!
!  function cuda_StreamCreate( pstream)result (istat) &
!           bind(C, name ="cudaStreamCreate")
!
!    use iso_c_binding
!
!    implicit none
!    integer(C_SIZE_T) :: pstream
!    integer(c_int)    :: istat
!  end function
!
!  function cuda_StreamDestroy( pstream)result (istat) &
!           bind(C, name ="cudaStreamDestroy")
!
!    use iso_c_binding
!
!    implicit none
!    integer(C_SIZE_T), value  :: pstream
!    integer(c_int)            :: istat
!
!  end function
!
!  function cuda_streamsynchronize( pstream)result(istat) &
!           bind(C,name='cudaStreamSynchronize')
!
!    use iso_c_binding
!
!    implicit none
!    integer(C_SIZE_T),value  :: pstream
!    integer(C_INT)           :: istat
!
!  end function
!end interface

contains
  function cuda_setdevice(n) result(success)
    use iso_c_binding

    implicit none

    integer, intent(in)  :: n
    logical              :: success

#ifdef WITH_GPU_VERSION
    success = cuda_setdevice_c(int(n,kind=c_int)) /= 0
#else
    success = .true.
#endif
  end function cuda_setdevice

  function cuda_getdevicecount(n) result(success)
    use iso_c_binding
    implicit none

    integer, intent(out) :: n
    logical              :: success

#ifdef WITH_GPU_VERSION
    success = cuda_getdevicecount_c(int(n, kind=c_int)) /=0
#else
    success = .true.
    n     = 0
#endif
  end function cuda_getdevicecount

  function cuda_malloc(a, width_height) result(success)
    use iso_c_binding
    implicit none

    integer(C_SIZE_T)             :: a
    integer(C_SIZE_T), intent(in) :: width_height

#ifdef WTIH_GPU_VERSION
    success = cuda_malloc_c(a,width_height) /= 0
#else
    success == .false.
#endif
  end function

  function cuda_memcpy(dst, src, size, dir) result(success)

    use iso_c_binding

    implicit none
    integer(C_SIZE_T)             :: dst
    integer(C_SIZE_T)             :: src
    integer(C_SIZE_T), intent(in) :: size
    integer(C_INT), intent(in)    :: dir

#ifdef WITH_GPU_VERSION
    success = cuda_memcpy_c(dst, src, size, dir) /=0
#else
    success = .false.
#endif

  end function cuda_memcpy

  function cuda_free(a) result(success)

    use iso_c_binding

    implicit none
    integer(C_SIZE_T) :: a

#ifdef WITH_GPU_VERSION
    success = cuda_free_c(a) /= 0
#else
    success = .false.
#endif

  end function cuda_free

  function cuda_memcpy2d(dst, dpitch, src, spitch, width, height , dir) result(success)

    use iso_c_binding

    implicit none

    integer(C_SIZE_T)             :: dst
    integer(C_SIZE_T), intent(in) :: dpitch
    integer(C_SIZE_T)             :: src
    integer(C_SIZE_T), intent(in) :: spitch
    integer(C_SIZE_T), intent(in) :: width
    integer(C_SIZE_T), intent(in) :: height
    integer(C_INT), intent(in)    :: dir
    integer(C_INT)                :: istat

#ifdef WITH_GPU_VERSION
    success = cuda_memcpy2d_c(dst, dpitch, src, spitch, width, height , dir) /= 0
#else
    success = .false.
#endif

  end function cuda_memcpy2d


end module cuda_routines


