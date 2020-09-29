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
    subroutine descinit(DESC, M, N, MB, NB, IRSRC, ICSRC, ICTXT, LLD, INFO)
    use precision
    implicit none
    integer(kind=BLAS_KIND) :: DESC(*), M, N, MB, NB, IRSRC, ICSRC, LLD
    integer(kind=BLAS_KIND), intent(inout) :: info
    integer(kind=BLAS_KIND) :: ICTXT
    end subroutine
  end interface

  interface
    subroutine blacs_gridinit(ICONTXT, ORDER, NPROW, NPCOL)
    use precision
    implicit none
    integer(kind=BLAS_KIND) :: ICONTXT
    character(len=1)        :: ORDER
    integer(kind=BLAS_KIND) :: NPROW, NPCOL
    end subroutine
  end interface

  interface
    subroutine blacs_gridexit(ICONTXT)
    use precision
    implicit none
    integer(kind=BLAS_KIND) :: ICONTXT
    end subroutine
  end interface

  interface
    subroutine blacs_gridinfo(ICONTXT, NPROW, NPCOL, MYPROW, MYPCOL)
    use precision
    implicit none
    integer(kind=BLAS_KIND) :: ICONTXT
    integer(kind=BLAS_KIND), intent(inout) :: NPROW, NPCOL, MYPROW, MYPCOL
    end subroutine
  end interface


  interface
    function numroc(N, NB, IPROC, ISRCPROC, NPROCS) result(numr)
    use precision
    implicit none
    integer(kind=BLAS_KIND) :: N, NB, IPROC, ISRCPROC, NPROCS, numr
    end function
  end interface


  interface
    subroutine pdgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, IA, JA, DESCA, B, IB, JB, DESCB, BETA, C, IC, JC, DESCC)
    use PRECISION_MODULE
    implicit none
    character               :: TRANSA, TRANSB
    integer(kind=BLAS_KIND) :: M, N, K, IA, JA, DESCA(*), IB, JB, DESCB(*), IC, JC, DESCC(*)
    real(kind=rk8)          :: ALPHA, BETA
    real(kind=rk8)          :: A(*), B(*), C(*)
    end subroutine
  end interface

  interface
    subroutine pdnrm2(N, norm2, x, ix, jx, descx, incx)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: N, ix, jx, descx(*), incx
    real(kind=rk8)          :: norm2, x(*)
    end subroutine
  end interface

  interface
    subroutine pdlaset(UPLO, M, N, ALPHA, BETA, A, IA, JA, DESCA)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: M, N, IA, JA, DESCA(*)
    real(kind=rk8)          :: ALPHA, BETA
    real(kind=rk8)          :: A(*)
    end subroutine
  end interface

  interface
    subroutine pdtran(M, N, ALPHA, A, IA, JA, DESCA, BETA, C, IC, JC, DESCC)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: M, N, IA, JA, DESCA(*), IC, JC, DESCC(*)
    real(kind=rk8)          :: ALPHA, BETA
    real(kind=rk8)          :: A(*), C(*)
    end subroutine
  end interface

  interface
    function pdlange(NORM, M, N, A, IA, JA, DESCA, WORK) result(norm2)
    use PRECISION_MODULE
    implicit none
    character               :: norm
    integer(kind=BLAS_KIND) :: m, n, ia, ja, desca(*)
    real(kind=rk8)          :: a(*), work(*)
    real(kind=rk8)          :: norm2
    end function
  end interface

  interface
    subroutine pdgenr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt) 
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: m, n, ia, ja, desca(*), ib, jb, descb(*)
    real(kind=rk8)          :: a(*), b(*)
    integer(kind=BLAS_KIND) :: ictxt
    end subroutine
  end interface


  interface
    subroutine psgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, IA, JA, DESCA, B, IB, JB, DESCB, BETA, C, IC, JC, DESCC)
    use PRECISION_MODULE
    implicit none
    character               :: TRANSA, TRANSB
    integer(kind=BLAS_KIND) :: M, N, K, IA, JA, DESCA(*), IB, JB, DESCB(*), IC, JC, DESCC(*)
    real(kind=rk4)          :: ALPHA, BETA
    real(kind=rk4)          :: A(*), B(*), C(*)
    end subroutine
  end interface

  interface
    subroutine psnrm2(N, norm2, x, ix, jx, descx, incx)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: N, ix, jx, descx(*), incx
    real(kind=rk4)          :: norm2, x(*)
    end subroutine
  end interface

  interface
    subroutine pslaset(UPLO, M, N, ALPHA, BETA, A, IA, JA, DESCA)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: M, N, IA, JA, DESCA(*)
    real(kind=rk4)          :: ALPHA, BETA
    real(kind=rk4)          :: A(*)
    end subroutine
  end interface

  interface
    subroutine pstran(M, N, ALPHA, A, IA, JA, DESCA, BETA, C, IC, JC, DESCC)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: M, N, IA, JA, DESCA(*), IC, JC, DESCC(*)
    real(kind=rk4)          :: ALPHA, BETA
    real(kind=rk4)          :: A(*), C(*)
    end subroutine
  end interface

  interface
    function pslange(NORM, M, N, A, IA, JA, DESCA, WORK) result(norm2)
    use PRECISION_MODULE
    implicit none
    character               :: norm
    integer(kind=BLAS_KIND) :: m, n, ia, ja, desca(*)
    real(kind=rk4)          :: a(*), work(*)
    real(kind=rk4)          :: norm2
    end function
  end interface

  interface
    subroutine psgenr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt) 
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: m, n, ia, ja, desca(*), ib, jb, descb(*)
    real(kind=rk8)          :: a(*), b(*)
    integer(kind=BLAS_KIND) :: ictxt
    end subroutine
  end interface

  interface
    subroutine pzgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, IA, JA, DESCA, B, IB, JB, DESCB, BETA, C, IC, JC, DESCC)
    use PRECISION_MODULE
    implicit none
    character               :: TRANSA, TRANSB
    integer(kind=BLAS_KIND) :: M, N, K, IA, JA, DESCA(*), IB, JB, DESCB(*), IC, JC, DESCC(*)
    complex(kind=ck8)       :: ALPHA, BETA
    complex(kind=ck8)       :: A(*), B(*), C(*)
    end subroutine
  end interface

  interface
    subroutine pzdotc(N, DOTC, X, ix, jx, descx, incx, Y, iy, jy, descy, incy)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: N, ix, jx, descx(*), incx, iy, jy, descy(*), incy
    complex(kind=ck8)       :: DOTC
    complex(kind=ck8)       :: X(*), Y(*)
    end subroutine
  end interface

  interface
    subroutine pzlaset(UPLO, M, N, ALPHA, BETA, A, IA, JA, DESCA)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: M, N, IA, JA, DESCA(*)
    complex(kind=ck8)       :: ALPHA, BETA
    complex(kind=ck8)       :: A(*)
    end subroutine
  end interface

  interface
    subroutine pztranc(M, N, ALPHA, A, IA, JA, DESCA, BETA, C, IC, JC, DESCC)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: M, N, IA, JA, DESCA(*), IC, JC, DESCC(*)
    complex(kind=ck8)       :: ALPHA, BETA
    complex(kind=ck8)       :: A(*), C(*)
    end subroutine
  end interface

  interface
    function pzlange(NORM, M, N, A, IA, JA, DESCA, WORK) result(norm2)
    use PRECISION_MODULE
    implicit none
    character               :: norm
    integer(kind=BLAS_KIND) :: m, n, ia, ja, desca(*)
    complex(kind=ck8)       :: a(*)
    real(kind=rk8)          ::work(*)
    real(kind=rk8)          :: norm2
    end function
  end interface

  interface
    subroutine pzgenr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt) 
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: m, n, ia, ja, desca(*), ib, jb, descb(*)
    complex(kind=ck8)       :: a(*), b(*)
    integer(kind=BLAS_KIND) :: ictxt
    end subroutine
  end interface

  interface
    subroutine pcgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, IA, JA, DESCA, B, IB, JB, DESCB, BETA, C, IC, JC, DESCC)
    use PRECISION_MODULE
    implicit none
    character               :: TRANSA, TRANSB
    integer(kind=BLAS_KIND) :: M, N, K, IA, JA, DESCA(*), IB, JB, DESCB(*), IC, JC, DESCC(*)
    complex(kind=ck4)       :: ALPHA, BETA
    complex(kind=ck4)       :: A(*), B(*), C(*)
    end subroutine
  end interface

  interface
    subroutine pcdotc(N, DOTC, X, ix, jx, descx, incx, Y, iy, jy, descy, incy)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: N, ix, jx, descx(*), incx, iy, jy, descy(*), incy
    complex(kind=ck4)       :: DOTC
    complex(kind=ck4)       :: X(*), Y(*)
    end subroutine
  end interface

  interface
    subroutine pclaset(UPLO, M, N, ALPHA, BETA, A, IA, JA, DESCA)
    use PRECISION_MODULE
    implicit none
    character               :: UPLO
    integer(kind=BLAS_KIND) :: M, N, IA, JA, DESCA(*)
    complex(kind=ck4)       :: ALPHA, BETA
    complex(kind=ck4)       :: A(*)
    end subroutine
  end interface

  interface
    subroutine pctranc(M, N, ALPHA, A, IA, JA, DESCA, BETA, C, IC, JC, DESCC)
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: M, N, IA, JA, DESCA(*), IC, JC, DESCC(*)
    complex(kind=ck4)       :: ALPHA, BETA
    complex(kind=ck4)       :: A(*), C(*)
    end subroutine
  end interface

  interface
    function pclange(NORM, M, N, A, IA, JA, DESCA, WORK) result(norm2)
    use PRECISION_MODULE
    implicit none
    character               :: norm
    integer(kind=BLAS_KIND) :: m, n, ia, ja, desca(*)
    complex(kind=ck4)       :: a(*)
    real(kind=rk4)          ::work(*)
    real(kind=rk4)          :: norm2
    end function
  end interface

  interface
    subroutine pcgenr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt) 
    use PRECISION_MODULE
    implicit none
    integer(kind=BLAS_KIND) :: m, n, ia, ja, desca(*), ib, jb, descb(*)
    complex(kind=ck4)       :: a(*), b(*)
    integer(kind=BLAS_KIND) :: ictxt
    end subroutine
  end interface
