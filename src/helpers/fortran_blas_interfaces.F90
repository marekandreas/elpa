!    Copyright 2019, A. Marek
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

  interface
    subroutine dger(M, N, ALPHA, X, INCX, Y, INCY, A, LDA)
    use precision
    implicit none
    integer(kind=BLAS_KIND)       :: M, N, INCX, INCY, LDA
    real(kind=rk8), intent(in)    :: ALPHA, X(*), Y(*)
    real(kind=rk8), intent(inout) :: A(LDA, *)
    end subroutine
  end interface

  interface
    subroutine daxpy(N, DA, DX, INCX, DY, INCY)
    use precision
    implicit none
    integer(kind=BLAS_KIND)       :: N, INCX, INCY
    real(kind=rk8), intent(in)    :: DA, DX(*)
    real(kind=rk8), intent(inout) :: DY(*)
    end subroutine
  end interface

  interface
    subroutine dcopy(N, DX, INCX, DY, INCY)
    use precision
    implicit none
    integer(kind=BLAS_KIND)       :: N, INCX, INCY
    real(kind=rk8), intent(in)    :: DX(*)
    real(kind=rk8), intent(inout) :: DY(*)
    end subroutine
  end interface

  interface
    subroutine dscal(N, DA, DX, INCX)
    use precision
    implicit none
    integer(kind=BLAS_KIND)       :: N, INCX
    real(kind=rk8)                :: DA
    real(kind=rk8), intent(inout) :: DX(*)
    end subroutine

  end interface

  interface
    subroutine dgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
    use PRECISION_MODULE
    implicit none
    character               :: TRANSA, TRANSB
    integer(kind=BLAS_KIND) :: M, N, K, LDA, LDB, LDC
    real(kind=rk8)          :: ALPHA, BETA
    real(kind=rk8)          :: A(LDA, *), B(LDB, *), C(LDC, *)
    end subroutine
  end interface

  interface
    subroutine dtrtri(UPLO, DIAG, N, A, LDA, INFO)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO, DIAG
    integer(kind=BLAS_KIND) :: N, LDA
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    real(kind=rk8)          :: a(lda, *)
    end subroutine
  end interface

  interface
    subroutine dpotrf(UPLO, N, A, LDA, INFO)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: N, LDA
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    real(kind=rk8)          :: a(lda, *)
    end subroutine
  end interface

  interface
    subroutine dtrsm(SIDE, UPLO, TRANSA, DIAG, M,N, ALPHA, A, LDA, B, LDB)
    use PRECISION_MODULE
    implicit none
    character               :: SIDE, UPLO, TRANSA, DIAG
    integer(kind=BLAS_KIND) :: M, N, LDA, LDB
    real(kind=rk8)          :: ALPHA
    real(kind=rk8)          :: a(lda, *), b(ldb, *)
    end subroutine
  end interface

  interface
    subroutine dgemv(TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY)
    use PRECISION_MODULE
    implicit none
    character               :: TRANS
    integer(kind=BLAS_KIND) :: M, N, LDA, INCX, INCY
    real(kind=rk8)          :: ALPHA, BETA
    real(kind=rk8)          :: a(lda, *), x(*), y(*)
    end subroutine
  end interface

  interface
    subroutine dtrmv(UPLO, TRANS, DIAG, N, A, LDA, X, INCX)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO, TRANS, DIAG
    integer(kind=BLAS_KIND) :: N, LDA, INCX
    real(kind=rk8)          :: a(lda, *), x(*)
    end subroutine
  end interface

  interface
    subroutine dtrmm(SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB)
    use PRECISION_MODULE
    implicit none
    character               :: SIDE, UPLO, TRANSA, DIAG
    integer(kind=BLAS_KIND) :: M, N, LDA, LDB
    real(kind=rk8)          :: ALPHA
    real(kind=rk8)          :: a(lda, *), b(ldb, *)
    end subroutine
  end interface

  interface
    subroutine dsyrk(UPLO, TRANS, N, K, ALPHA, A, LDA, BETA, C, LDC)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO, TRANS
    integer(kind=BLAS_KIND) :: N, K, LDA, LDC
    real(kind=rk8)          :: ALPHA, BETA
    real(kind=rk8)          :: a(lda, *), c(ldc, *)
    end subroutine
  end interface

  interface
    subroutine dsymv(UPLO, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: N, LDA, INCX, INCY
    real(kind=rk8)          :: ALPHA, BETA
    real(kind=rk8)          :: a(lda, *), x(*), y(*)
    end subroutine
  end interface

  interface
    subroutine dsymm(SIDE, UPLO, M, N, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
    use PRECISION_MODULE
    implicit none
    character               :: SIDE, UPLO
    integer(kind=BLAS_KIND) :: M, N, LDA, LDB, LDC
    real(kind=rk8)          :: ALPHA, BETA
    real(kind=rk8)          :: a(lda, *), b(ldb, *), c(ldc, *)
    end subroutine
  end interface

  interface
    subroutine dsyr2(UPLO, N, ALPHA, X, INCX, Y, INCY, A, LDA)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: N, INCX, INCY, LDA
    real(kind=rk8)          :: ALPHA
    real(kind=rk8)          :: a(lda, *), x(*), y(*)
    end subroutine
  end interface

  interface
    subroutine dsyr2k(UPLO, TRANS, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO, TRANS
    integer(kind=BLAS_KIND) :: N, K, LDA, LDB, LDC
    real(kind=rk8)          :: ALPHA, BETA
    real(kind=rk8)          :: a(lda, *), b(ldb, *), c(ldc, *)
    end subroutine
  end interface

  interface
    subroutine dgeqrf(M, N, A, LDA, TAU, WORK, LWORK, INFO)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: M, N, LDA, LWORK
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    real(kind=rk8)          :: a(lda, *), TAU(*), WORK(*)
    end subroutine
  end interface

  interface
    subroutine dstedc(COMPZ, N, D, E, Z, LDZ, WORK, LWORK, IWORK, LIWORK, INFO)
    use PRECISION_MODULE
    implicit none
    character               :: COMPZ
    integer(kind=BLAS_KIND) :: N, LDZ, LWORK, IWORK(*), LIWORK
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    real(kind=rk8)          :: D(*), E(*), z(ldz, *), work(*)
    end subroutine
  end interface

  interface
    subroutine dsteqr(COMPZ, N, D, E, Z, LDZ, WORK, INFO)
    use PRECISION_MODULE
    implicit none
    character               :: COMPZ
    integer(kind=BLAS_KIND) :: N, LDZ
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    real(kind=rk8)          :: D(*), E(*), z(ldz, *), work(*)
    end subroutine
  end interface

  interface
    subroutine dlamrg(N1, N2, A, DTRD1, DTRD2, INDEX)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND), intent(in) :: N1, N2, DTRD1, DTRD2
    integer(kind=BLAS_KIND), intent(inout) :: INDEX(*)
    real(kind=rk8), intent(in)          :: A(*)
    end subroutine
  end interface

  interface
    function dlamch(CMACG) result(DMACH)
    use PRECISION_MODULE
    implicit none
    character               :: CMACG
    real(kind=rk8)          :: DMACH
    end function
  end interface

  interface
    function dlapy2(X, Y) result(sqrt_x2_y2)
    use PRECISION_MODULE
    implicit none
    real(kind=rk8)          :: x, y, sqrt_x2_y2

    end function
  end interface

  interface
    subroutine dlaed4(N, I, D, Z, DELTA, RHO, DLAM, INFO)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: N, I
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    real(kind=rk8)          :: D(*), Z(*), DELTA(*), RHO, DLAM
    end subroutine
  end interface

  interface
    subroutine dlaed5(I, D, Z, DELTA, RHO, DLAM)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: I
    real(kind=rk8)          :: D(2), Z(2), DELTA(2), RHO, DLAM
    end subroutine
  end interface

  interface
    function dnrm2(N,X, INCX) result(nrm2)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: N, INCX
    real(kind=rk8)          :: x(*), nrm2

    end function
  end interface

  interface
    subroutine dlaset(UPLO, M, N, ALPHA, BETA, A, LDA)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: M, N, LDA
    real(kind=rk8)          :: ALPHA, BETA, A(lda, *)
    end subroutine
  end interface

  interface
    function dlange(NORM, M, N, A, LDA, WORK) result(norm2)
    use PRECISION_MODULE
    implicit none
    character               :: NORM
    integer(kind=BLAS_KIND) :: M, N, LDA
    real(kind=rk8)          :: A(lda, *)
    real(kind=rk8), intent(inout) :: work(*)
    real(kind=rk8)          :: norm2
    end function
  end interface

!#endif /* DOUBLE_PRECISION_REAL */
  interface
    subroutine sger(M, N, ALPHA, X, INCX, Y, INCY, A, LDA)
    use precision
    implicit none
    integer(kind=BLAS_KIND)       :: M, N, INCX, INCY, LDA
    real(kind=rk4), intent(in)    :: ALPHA, X(*), Y(*)
    real(kind=rk4), intent(inout) :: A(LDA, *)
    end subroutine
  end interface

  interface
    subroutine saxpy(N, DA, DX, INCX, DY, INCY)
    use precision
    implicit none
    integer(kind=BLAS_KIND)       :: N, INCX, INCY
    real(kind=rk4), intent(in)    :: DA, DX(*)
    real(kind=rk4), intent(inout) :: DY(*)
    end subroutine
  end interface

  interface
    subroutine scopy(N, DX, INCX, DY, INCY)
    use precision
    implicit none
    integer(kind=BLAS_KIND)       :: N, INCX, INCY
    real(kind=rk4), intent(in)    :: DX(*)
    real(kind=rk4), intent(inout) :: DY(*)
    end subroutine
  end interface

  interface
    subroutine sscal(N, DA, DX, INCX)
    use precision
    implicit none
    integer(kind=BLAS_KIND)       :: N, INCX
    real(kind=rk4)                :: DA
    real(kind=rk4), intent(inout) :: DX(*)
    end subroutine
  end interface

  interface
    subroutine sgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
    use PRECISION_MODULE
    implicit none
    character               :: TRANSA, TRANSB
    integer(kind=BLAS_KIND) :: M, N, K, LDA, LDB, LDC
    real(kind=rk4)          :: ALPHA, BETA
    real(kind=rk4)          :: A(LDA, *), B(LDB, *), C(LDC, *)
    end subroutine
  end interface

  interface
    subroutine strtri(UPLO, DIAG, N, A, LDA, INFO)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO, DIAG
    integer(kind=BLAS_KIND) :: N, LDA
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    real(kind=rk4)          :: a(lda, *)
    end subroutine
  end interface

  interface
    subroutine spotrf(UPLO, N, A, LDA, INFO)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: N, LDA
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    real(kind=rk4)          :: a(lda, *)
    end subroutine
  end interface

  interface
    subroutine strsm(SIDE, UPLO, TRANSA, DIAG, M,N, ALPHA, A, LDA, B, LDB)
    use PRECISION_MODULE
    implicit none
    character               :: SIDE, UPLO, TRANSA, DIAG
    integer(kind=BLAS_KIND) :: M, N, LDA, LDB
    real(kind=rk4)          :: ALPHA
    real(kind=rk4)          :: a(lda, *), b(ldb, *)
    end subroutine
  end interface

  interface
    subroutine sgemv(TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY)
    use PRECISION_MODULE
    implicit none
    character               :: TRANS
    integer(kind=BLAS_KIND) :: M, N, LDA, INCX, INCY
    real(kind=rk4)          :: ALPHA, BETA
    real(kind=rk4)          :: a(lda, *), x(*), y(*)
    end subroutine
  end interface

  interface
    subroutine strmv(UPLO, TRANS, DIAG, N, A, LDA, X, INCX)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO, TRANS, DIAG
    integer(kind=BLAS_KIND) :: N, LDA, INCX
    real(kind=rk4)          :: a(lda, *), x(*)
    end subroutine
  end interface

  interface
    subroutine strmm(SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB)
    use PRECISION_MODULE
    implicit none
    character               :: SIDE, UPLO, TRANSA, DIAG
    integer(kind=BLAS_KIND) :: M, N, LDA, LDB
    real(kind=rk4)          :: ALPHA
    real(kind=rk4)          :: a(lda, *), b(ldb, *)
    end subroutine
  end interface

  interface
    subroutine ssyrk(UPLO, TRANS, N, K, ALPHA, A, LDA, BETA, C, LDC)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO, TRANS
    integer(kind=BLAS_KIND) :: N, K, LDA, LDC
    real(kind=rk4)          :: ALPHA, BETA
    real(kind=rk4)          :: a(lda, *), c(ldc, *)
    end subroutine
  end interface

  interface
    subroutine ssymv(UPLO, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: N, LDA, INCX, INCY
    real(kind=rk4)          :: ALPHA, BETA
    real(kind=rk4)          :: a(lda, *), x(*), y(*)
    end subroutine
  end interface

  interface
    subroutine ssymm(SIDE, UPLO, M, N, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
    use PRECISION_MODULE
    implicit none
    character               :: SIDE, UPLO
    integer(kind=BLAS_KIND) :: M, N, LDA, LDB, LDC
    real(kind=rk4)          :: ALPHA, BETA
    real(kind=rk4)          :: a(lda, *), b(ldb, *), c(ldc, *)
    end subroutine
  end interface

  interface
    subroutine ssyr2(UPLO, N, ALPHA, X, INCX, Y, INCY, A, LDA)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: N, INCX, INCY, LDA
    real(kind=rk4)          :: ALPHA
    real(kind=rk4)          :: a(lda, *), x(*), y(*)
    end subroutine
  end interface

  interface
    subroutine ssyr2k(UPLO, TRANS, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO, TRANS
    integer(kind=BLAS_KIND) :: N, K, LDA, LDB, LDC
    real(kind=rk4)          :: ALPHA, BETA
    real(kind=rk4)          :: a(lda, *), b(ldb, *), c(ldc, *)
    end subroutine
  end interface

  interface
    subroutine sgeqrf(M, N, A, LDA, TAU, WORK, LWORK, INFO)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: M, N, LDA, LWORK
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    real(kind=rk4)          :: a(lda, *), TAU(*), WORK(*)
    end subroutine
  end interface

  interface
    subroutine sstedc(COMPZ, N, D, E, Z, LDZ, WORK, LWORK, IWORK, LIWORK, INFO)
    use PRECISION_MODULE
    implicit none
    character               :: COMPZ
    integer(kind=BLAS_KIND) :: N, LDZ, LWORK, IWORK(*), LIWORK
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    real(kind=rk4)          :: D(*), E(*), z(ldz, *), work(*)
    end subroutine
  end interface

  interface
    subroutine ssteqr(COMPZ, N, D, E, Z, LDZ, WORK, INFO)
    use PRECISION_MODULE
    implicit none
    character               :: COMPZ
    integer(kind=BLAS_KIND) :: N, LDZ
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    real(kind=rk4)          :: D(*), E(*), z(ldz, *), work(*)
    end subroutine
  end interface

  interface
    subroutine slamrg(N1, N2, A, DTRD1, DTRD2, INDEX)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND), intent(in) :: N1, N2, DTRD1, DTRD2
    integer(kind=BLAS_KIND), intent(inout) :: INDEX(*)
    real(kind=rk4), intent(in)          :: A(*)
    end subroutine
  end interface

  interface
    function slamch(CMACG) result(DMACH)
    use PRECISION_MODULE
    implicit none
    character               :: CMACG
    real(kind=rk4)          :: DMACH
    end function
  end interface

  interface
    function slapy2(X, Y) result(sqrt_x2_y2)
    use PRECISION_MODULE
    implicit none
    real(kind=rk4)          :: x, y, sqrt_x2_y2

    end function
  end interface

  interface
    subroutine slaed4(N, I, D, Z, DELTA, RHO, DLAM, INFO)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: N, I
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    real(kind=rk4)          :: D(*), Z(*), DELTA(*), RHO, DLAM
    end subroutine
  end interface

  interface
    subroutine slaed5(I, D, Z, DELTA, RHO, DLAM)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: I
    real(kind=rk4)          :: D(2), Z(2), DELTA(2), RHO, DLAM
    end subroutine
  end interface


  interface
    function snrm2(N,X, INCX) result(nrm2)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: N, INCX
    real(kind=rk4)          :: x(*), nrm2

    end function
  end interface

  interface
    subroutine slaset(UPLO, M, N, ALPHA, BETA, A, LDA)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: M, N, LDA
    real(kind=rk4)          :: ALPHA, BETA, A(lda, *)
    end subroutine
  end interface

  interface
    function slange(NORM, M, N, A, LDA, WORK) result(norm2)
    use PRECISION_MODULE
    implicit none
    character               :: NORM
    integer(kind=BLAS_KIND) :: M, N, LDA
    real(kind=rk4)          :: A(lda, *)
    real(kind=rk4), intent(inout) :: work(*)
    real(kind=rk4)          :: norm2
    end function
  end interface


!#endif /* SINGLE_PRECSION_REAL */
  interface
   complex*16 function zdotc(N, ZX, INCX, ZY, INCY)
    use precision
    implicit none
    integer(kind=BLAS_KIND)          :: N, INCX, INCY
    complex(kind=ck8), intent(in)    :: ZX(*), ZY(*)
    end function
  end interface

  interface
    subroutine zaxpy(N, DA, DX, INCX, DY, INCY)
    use precision
    implicit none
    integer(kind=BLAS_KIND)          :: N, INCX, INCY
    complex(kind=ck8), intent(in)    :: DA, DX(*)
    complex(kind=ck8), intent(inout) :: DY(*)
    end subroutine
  end interface

  interface
    subroutine zcopy(N, DX, INCX, DY, INCY)
    use precision
    implicit none
    integer(kind=BLAS_KIND)          :: N, INCX, INCY
    complex(kind=ck8), intent(in)    :: DX(*)
    complex(kind=ck8), intent(inout) :: DY(*)
    end subroutine
  end interface


  interface
    subroutine zscal(N, DA, DX, INCX)
    use precision
    implicit none
    integer(kind=BLAS_KIND)          :: N, INCX
    complex(kind=ck8)                :: DA
    complex(kind=ck8), intent(inout) :: DX(*)
    end subroutine
  end interface

  interface
    subroutine zgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
    use PRECISION_MODULE
    implicit none
    character               :: TRANSA, TRANSB
    integer(kind=BLAS_KIND) :: M, N, K, LDA, LDB, LDC
    complex(kind=ck8)       :: ALPHA, BETA
    complex(kind=ck8)       :: A(LDA, *), B(LDB, *), C(LDC, *)
    end subroutine
  end interface

  interface
    subroutine ztrtri(UPLO, DIAG, N, A, LDA, INFO)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO, DIAG
    integer(kind=BLAS_KIND) :: N, LDA
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    complex(kind=ck8)          :: a(lda, *)
    end subroutine
  end interface

  interface
    subroutine zpotrf(UPLO, N, A, LDA, INFO)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: N, LDA
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    complex(kind=ck8)          :: a(lda, *)
    end subroutine
  end interface

  interface
    subroutine ztrsm(SIDE, UPLO, TRANSA, DIAG, M,N, ALPHA, A, LDA, B, LDB)
    use PRECISION_MODULE
    implicit none
    character               :: SIDE, UPLO, TRANSA, DIAG
    integer(kind=BLAS_KIND) :: M, N, LDA, LDB
    complex(kind=ck8)       :: ALPHA
    complex(kind=ck8)       :: a(lda, *), b(ldb, *)
    end subroutine
  end interface

 interface
    subroutine zgemv(TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY)
    use PRECISION_MODULE
    implicit none
    character               :: TRANS
    integer(kind=BLAS_KIND) :: M, N, LDA, INCX, INCY
    complex(kind=ck8)       :: ALPHA, BETA
    complex(kind=ck8)       :: a(lda, *), x(*), y(*)
    end subroutine
  end interface

  interface
    subroutine ztrmv(UPLO, TRANS, DIAG, N, A, LDA, X, INCX)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO, TRANS, DIAG
    integer(kind=BLAS_KIND) :: N, LDA, INCX
    complex(kind=ck8)       :: a(lda, *), x(*)
    end subroutine
  end interface

  interface
    subroutine ztrmm(SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB)
    use PRECISION_MODULE
    implicit none
    character               :: SIDE, UPLO, TRANSA, DIAG
    integer(kind=BLAS_KIND) :: M, N, LDA, LDB
    complex(kind=ck8)       :: ALPHA
    complex(kind=ck8)       :: a(lda, *), b(ldb, *)
    end subroutine
  end interface

  interface
    subroutine zherk(UPLO, TRANS, N, K, ALPHA, A, LDA, BETA, C, LDC)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO, TRANS
    integer(kind=BLAS_KIND) :: N, K, LDA, LDC
    complex(kind=ck8)       :: ALPHA, BETA
    complex(kind=ck8)       :: a(lda, *), c(ldc, *)
    end subroutine
  end interface

  interface
    subroutine zhemv(UPLO, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: N, LDA, INCX, INCY
    complex(kind=ck8)       :: ALPHA, BETA
    complex(kind=ck8)       :: a(lda, *), x(*), y(*)
    end subroutine
  end interface

  interface
    subroutine zsymm(SIDE, UPLO, M, N, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
    use PRECISION_MODULE
    implicit none
    character               :: SIDE, UPLO
    integer(kind=BLAS_KIND) :: M, N, LDA, LDB, LDC
    complex(kind=ck8)       :: ALPHA, BETA
    complex(kind=ck8)       :: a(lda, *), b(ldb, *), c(ldc, *)
    end subroutine
  end interface

  interface
    subroutine zher2(UPLO, N, ALPHA, X, INCX, Y, INCY, A, LDA)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: N, INCX, INCY, LDA
    complex(kind=ck8)       :: ALPHA
    complex(kind=ck8)       :: a(lda, *), x(*), y(*)
    end subroutine
  end interface

  interface
    subroutine zgeqrf(M, N, A, LDA, TAU, WORK, LWORK, INFO)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: M, N, LDA, LWORK
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    complex(kind=ck8)       :: a(lda, *), TAU(*), WORK(*)
    end subroutine
  end interface

#if 0
  ! not used
  interface
    subroutine zstedc(COMPZ, N, D, E, Z, LDZ, WORK, LWORK, RWORK, LRWORK, IWORK, LIWORK, INFO)
    use PRECISION_MODULE
    implicit none
    character               :: COMPZ
    integer(kind=BLAS_KIND) :: N, LDZ, LWORK, LRWORK, IWORK(*), LIWORK
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    real(kind=rk8)          :: D(*), E(*), RWORK(*)
    complex(kind=ck8)       :: z(ldz, *), work(*)
    end subroutine
  end interface
#endif

  interface
    subroutine zlaset(UPLO, M, N, ALPHA, BETA, A, LDA)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: M, N, LDA
    complex(kind=ck8)       :: ALPHA, BETA, A(lda, *)
    end subroutine
  end interface

  interface
    function zlange(NORM, M, N, A, LDA, WORK) result(norm2)
    use PRECISION_MODULE
    implicit none
    character                        :: NORM
    integer(kind=BLAS_KIND)          :: M, N, LDA
    complex(kind=ck8)                :: A(lda, *)
    real(kind=rk8), intent(inout)    :: work(*)
    real(kind=rk8)                   :: norm2
    end function
  end interface



!#endif /* DOUBLE_PRECISION_COMPLEX */
  interface
   complex*8 function cdotc(N, ZX, INCX, ZY, INCY)
    use precision
    implicit none
    integer(kind=BLAS_KIND)          :: N, INCX, INCY
    complex(kind=ck4), intent(in)    :: ZX(*), ZY(*)
    end function
  end interface

  interface
    subroutine caxpy(N, DA, DX, INCX, DY, INCY)
    use precision
    implicit none
    integer(kind=BLAS_KIND)          :: N, INCX, INCY
    complex(kind=ck4), intent(in)    :: DA, DX(*)
    complex(kind=ck4), intent(inout) :: DY(*)
    end subroutine
  end interface

  interface
    subroutine ccopy(N, DX, INCX, DY, INCY)
    use precision
    implicit none
    integer(kind=BLAS_KIND)          :: N, INCX, INCY
    complex(kind=ck4), intent(in)    :: DX(*)
    complex(kind=ck4), intent(inout) :: DY(*)
    end subroutine
  end interface

  interface
    subroutine cscal(N, DA, DX, INCX)
    use precision
    implicit none
    integer(kind=BLAS_KIND)          :: N, INCX
    complex(kind=ck4)                :: DA
    complex(kind=ck4), intent(inout) :: DX(*)
    end subroutine
  end interface

  interface
    subroutine cgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
    use PRECISION_MODULE
    implicit none
    character               :: TRANSA, TRANSB
    integer(kind=BLAS_KIND) :: M, N, K, LDA, LDB, LDC
    complex(kind=ck4)       :: ALPHA, BETA
    complex(kind=ck4)       :: A(LDA, *), B(LDB, *), C(LDC, *)
    end subroutine
  end interface

  interface
    subroutine ctrtri(UPLO, DIAG, N, A, LDA, INFO)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO, DIAG
    integer(kind=BLAS_KIND) :: N, LDA
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    complex(kind=ck4)          :: a(lda, *)
    end subroutine
  end interface

  interface
    subroutine cpotrf(UPLO, N, A, LDA, INFO)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: N, LDA
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    complex(kind=ck4)       :: a(lda, *)
    end subroutine
  end interface

  interface
    subroutine ctrsm(SIDE, UPLO, TRANSA, DIAG, M,N, ALPHA, A, LDA, B, LDB)
    use PRECISION_MODULE
    implicit none
    character               :: SIDE, UPLO, TRANSA, DIAG
    integer(kind=BLAS_KIND) :: M, N, LDA, LDB
    complex(kind=ck4)       :: ALPHA
    complex(kind=ck4)       :: a(lda, *), b(ldb, *)
    end subroutine
  end interface

 interface
    subroutine cgemv(TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY)
    use PRECISION_MODULE
    implicit none
    character               :: TRANS
    integer(kind=BLAS_KIND) :: M, N, LDA, INCX, INCY
    complex(kind=ck4)       :: ALPHA, BETA
    complex(kind=ck4)       :: a(lda, *), x(*), y(*)
    end subroutine
  end interface

  interface
    subroutine ctrmv(UPLO, TRANS, DIAG, N, A, LDA, X, INCX)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO, TRANS, DIAG
    integer(kind=BLAS_KIND) :: N, LDA, INCX
    complex(kind=ck4)       :: a(lda, *), x(*)
    end subroutine
  end interface

  interface
    subroutine ctrmm(SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB)
    use PRECISION_MODULE
    implicit none
    character               :: SIDE, UPLO, TRANSA, DIAG
    integer(kind=BLAS_KIND) :: M, N, LDA, LDB
    complex(kind=ck4)       :: ALPHA
    complex(kind=ck4)       :: a(lda, *), b(ldb, *)
    end subroutine
  end interface

  interface
    subroutine cherk(UPLO, TRANS, N, K, ALPHA, A, LDA, BETA, C, LDC)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO, TRANS
    integer(kind=BLAS_KIND) :: N, K, LDA, LDC
    complex(kind=ck4)       :: ALPHA, BETA
    complex(kind=ck4)       :: a(lda, *), c(ldc, *)
    end subroutine
  end interface

  interface
    subroutine chemv(UPLO, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: N, LDA, INCX, INCY
    complex(kind=ck4)       :: ALPHA, BETA
    complex(kind=ck4)       :: a(lda, *), x(*), y(*)
    end subroutine
  end interface

  interface
    subroutine csymm(SIDE, UPLO, M, N, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
    use PRECISION_MODULE
    implicit none
    character               :: SIDE, UPLO
    integer(kind=BLAS_KIND) :: M, N, LDA, LDB, LDC
    complex(kind=ck4)       :: ALPHA, BETA
    complex(kind=ck4)       :: a(lda, *), b(ldb, *), c(ldc, *)
    end subroutine
  end interface

  interface
    subroutine cher2(UPLO, N, ALPHA, X, INCX, Y, INCY, A, LDA)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: N, INCX, INCY, LDA
    complex(kind=ck4)       :: ALPHA
    complex(kind=ck4)       :: a(lda, *), x(*), y(*)
    end subroutine
  end interface

  interface
    subroutine cgeqrf(M, N, A, LDA, TAU, WORK, LWORK, INFO)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: M, N, LDA, LWORK
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    complex(kind=ck4)       :: a(lda, *), TAU(*), WORK(*)
    end subroutine
  end interface

#if 0
  ! not used
  interface
    subroutine cstedc(COMPZ, N, D, E, Z, LDZ, WORK, LWORK, RWORK, LRWORK, IWORK, LIWORK, INFO)
    use PRECISION_MODULE
    implicit none
    character               :: COMPZ
    integer(kind=BLAS_KIND) :: N, LDZ, LWORK, LRWORK, IWORK(*), LIWORK
    integer(kind=BLAS_KIND), intent(inout) :: INFO
    real(kind=rk4)          :: D(*), E(*), RWORK(*)
    complex(kind=ck4)       :: z(ldz, *), work(*)
    end subroutine
  end interface
#endif


  interface
    subroutine claset(UPLO, M, N, ALPHA, BETA, A, LDA)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: M, N, LDA
    complex(kind=ck4)       :: ALPHA, BETA, A(lda, *)
    end subroutine
  end interface

  interface
    function clange(NORM, M, N, A, LDA, WORK) result(norm2)
    use PRECISION_MODULE
    implicit none
    character                     :: NORM
    integer(kind=BLAS_KIND)       :: M, N, LDA
    complex(kind=ck4)             :: A(lda, *)
    real(kind=rk4), intent(inout) :: work(*)
    real(kind=rk4)                :: norm2
    end function
  end interface


