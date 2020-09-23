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

module mkl_offload

  interface
    subroutine mkl_offload_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
    bind(C, name="mkl_offload_dgemm_c")
      use iso_c_binding
      implicit none
      character(1,C_CHAR), value :: transa, transb
      integer(kind=c_int), value :: m, n, k, lda, ldb, ldc
      real(kind=c_double), value :: alpha, beta
      real(kind=c_double)        :: a(lda, k)
      real(kind=c_double)        :: b(ldb, n)
      real(kind=c_double)        :: c(ldc, n)
    end subroutine
  end interface

  interface
    subroutine mkl_offload_dgemv(trans, m, n, alpha, a, lda,  x, incx, beta, y, incy) &
    bind(C, name="mkl_offload_dgemv_c")
      use iso_c_binding
      implicit none
      character(1,C_CHAR), value :: trans
      integer(kind=c_int), value :: m, n, lda, incx, incy
      real(kind=c_double), value :: alpha, beta

      !real(kind=c_double)        :: a(lda,n)
      !real(kind=c_double)        :: x(sizeX)
      !real(kind=c_double)        :: y(sizeY)
      real(kind=c_double)        :: a(lda,*)
      real(kind=c_double)        :: x(*)
      real(kind=c_double)        :: y(*)
    end subroutine
  end interface

  interface
    subroutine mkl_offload_dtrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb) &
    bind(C, name="mkl_offload_dtrmm_c")
      use iso_c_binding
      implicit none
      character(1,C_CHAR), value :: side, uplo, transa, diag
      integer(kind=c_int), value :: m, n, lda, ldb
      real(kind=c_double), value :: alpha

      !real(kind=c_double)        :: a(lda,n)
      !real(kind=c_double)        :: x(sizeX)
      !real(kind=c_double)        :: y(sizeY)
      real(kind=c_double)        :: a(lda,*)
      real(kind=c_double)        :: b(ldb,*)
    end subroutine
  end interface

#ifdef WANT_SINGLE_PRECISION_REAL
  interface
    subroutine mkl_offload_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
    bind(C, name="mkl_offload_sgemm_c")
      use iso_c_binding
      implicit none
      character(1,C_CHAR), value :: transa, transb
      integer(kind=c_int), value :: m, n, k, lda, ldb, ldc
      real(kind=c_float), value  :: alpha, beta
      real(kind=c_float)         :: a(lda, k)
      real(kind=c_float)         :: b(ldb, n)
      real(kind=c_float)         :: c(ldc, n)
    end subroutine
  end interface

  interface
    subroutine mkl_offload_sgemv(trans, m, n, alpha, a, lda,  x, incx, beta, y, incy) &
    bind(C, name="mkl_offload_sgemv_c")
      use iso_c_binding
      implicit none
      character(1,C_CHAR), value :: trans
      integer(kind=c_int), value :: m, n, lda, incx, incy
      real(kind=c_float),  value :: alpha, beta

      real(kind=c_float)         :: a(lda,n)
      real(kind=c_float)         :: x(n)
      real(kind=c_float)         :: y(m)
      !real(kind=c_float)         :: a(lda,*)
      !real(kind=c_float)         :: x(*)
      !real(kind=c_float)         :: y(*)
    end subroutine
  end interface

  interface
    subroutine mkl_offload_strmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb) &
    bind(C, name="mkl_offload_strmm_c")
      use iso_c_binding
      implicit none
      character(1,C_CHAR), value :: side, uplo, transa, diag
      integer(kind=c_int), value :: m, n, lda, ldb
      real(kind=c_float), value  :: alpha

      !real(kind=c_double)        :: a(lda,n)
      !real(kind=c_double)        :: x(sizeX)
      !real(kind=c_double)        :: y(sizeY)
      real(kind=c_float)          :: a(lda,*)
      real(kind=c_float)          :: b(ldb,*)
    end subroutine
  end interface
#endif /*WANT_SINGLE_PRECISION_REAL */

  interface
    subroutine mkl_offload_zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
    bind(C, name="mkl_offload_zgemm_c")
      use iso_c_binding
      implicit none
      character(1,C_CHAR), value            :: transa, transb
      integer(kind=c_int), value            :: m, n, k, lda, ldb, ldc
      complex(kind=c_double_complex), value :: alpha, beta
      complex(kind=c_double_complex)        :: a(lda, k)
      complex(kind=c_double_complex)        :: b(ldb, n)
      complex(kind=c_double_complex)        :: c(ldc, n)
    end subroutine
  end interface

  interface
    subroutine mkl_offload_zgemv(trans, m, n, alpha, a, lda,  x, incx, beta, y, incy) &
    bind(C, name="mkl_offload_zgemv_c")
      use iso_c_binding
      implicit none
      character(1,C_CHAR), value             :: trans
      integer(kind=c_int), value             :: m, n, lda, incx, incy
      complex(kind=c_double_complex),  value :: alpha, beta

      !complex(kind=c_double_complex)         :: a(lda,n)
      !complex(kind=c_double_complex)         :: x(sizeX)
      !complex(kind=c_double_complex)         :: y(sizeY)
      complex(kind=c_double_complex)         :: a(lda,*)
      complex(kind=c_double_complex)         :: x(*)
      complex(kind=c_double_complex)         :: y(*)
    end subroutine
  end interface

  interface
    subroutine mkl_offload_ztrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb) &
    bind(C, name="mkl_offload_ztrmm_c")
      use iso_c_binding
      implicit none
      character(1,C_CHAR), value            :: side, uplo, transa, diag
      integer(kind=c_int), value            :: m, n, lda, ldb
      complex(kind=c_double_complex), value :: alpha

      !real(kind=c_double)        :: a(lda,n)
      !real(kind=c_double)        :: x(sizeX)
      !real(kind=c_double)        :: y(sizeY)
      complex(kind=c_double_complex)        :: a(lda,*)
      complex(kind=c_double_complex)        :: b(ldb,*)
    end subroutine
  end interface

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  interface
    subroutine mkl_offload_cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
    bind(C, name="mkl_offload_cgemm_c")
      use iso_c_binding
      implicit none
      character(1,C_CHAR), value           :: transa, transb
      integer(kind=c_int), value           :: m, n, k, lda, ldb, ldc
      complex(kind=c_float_complex), value :: alpha, beta
      complex(kind=c_float_complex)        :: a(lda, k)
      complex(kind=c_float_complex)        :: b(ldb, n)
      complex(kind=c_float_complex)        :: c(ldc, n)
    end subroutine
  end interface

  interface
    subroutine mkl_offload_cgemv(trans, m, n, alpha, a, lda,  x, incx, beta, y, incy) &
    bind(C, name="mkl_offload_cgemv_c")
      use iso_c_binding
      implicit none
      character(1,C_CHAR), value            :: trans
      integer(kind=c_int), value            :: m, n, lda, incx, incy
      complex(kind=c_float_complex),  value :: alpha, beta

      !complex(kind=c_float_complex)         :: a(lda,n)
      !complex(kind=c_float_complex)         :: x(sizeX)
      !complex(kind=c_float_complex)         :: y(sizeY)
      complex(kind=c_float_complex)         :: a(lda,*)
      complex(kind=c_float_complex)         :: x(*)
      complex(kind=c_float_complex)         :: y(*)
    end subroutine
  end interface

  interface
    subroutine mkl_offload_ctrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb) &
    bind(C, name="mkl_offload_ctrmm_c")
      use iso_c_binding
      implicit none
      character(1,C_CHAR), value            :: side, uplo, transa, diag
      integer(kind=c_int), value            :: m, n, lda, ldb
      complex(kind=c_float_complex), value  :: alpha

      !real(kind=c_double)        :: a(lda,n)
      !real(kind=c_double)        :: x(sizeX)
      !real(kind=c_double)        :: y(sizeY)
      complex(kind=c_float_complex)         :: a(lda,*)
      complex(kind=c_float_complex)         :: b(ldb,*)
    end subroutine
  end interface
#endif /* WANT_SINGLE_PRECISION_COMPLEX */

end module
