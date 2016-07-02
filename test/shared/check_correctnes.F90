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
!
#include "config-f90.h"

module mod_check_correctness


  interface check_correctness
    module procedure check_correctness_complex
    module procedure check_correctness_real
  end interface

  contains

    function check_correctness_complex(na, nev, as, z, ev, sc_desc, myid, tmp1, tmp2) result(status)

      use elpa_mpi
      use precision
      implicit none
      integer(kind=ik)                 :: status
      integer(kind=ik), intent(in)     :: na, nev, myid
      complex(kind=ck), intent(in)     :: as(:,:), z(:,:)
      complex(kind=ck), intent(inout)  :: tmp1(:,:), tmp2(:,:)
      real(kind=rk)                    :: ev(:)
      complex(kind=ck)                 :: xc
      integer(kind=ik)                 :: sc_desc(:), mpierr
      complex(kind=ck), parameter      :: CZERO = (0.d0,0.d0), CONE = (1.d0,0.d0)
      integer(kind=ik)                 :: i
      real(kind=rk)                    :: err, errmax
#ifndef WITH_MPI
      complex(kind=ck)                 :: zdotc
#endif

      status = 0

      ! 1. Residual (maximum of || A*Zi - Zi*EVi ||)
      ! tmp1 =  A * Z
      ! as is original stored matrix, Z are the EVs
#ifdef WITH_MPI
      call pzgemm('N','N',na,nev,na,CONE,as,1,1,sc_desc, &
                  z,1,1,sc_desc,CZERO,tmp1,1,1,sc_desc)
#else
      call zgemm('N','N',na,nev,na,CONE,as,na,z,na,CZERO,tmp1,na)
#endif
      ! tmp2 = Zi*EVi
      tmp2(:,:) = z(:,:)
      do i=1,nev
        xc = ev(i)
#ifdef WITH_MPI
        call pzscal(na,xc,tmp2,1,i,sc_desc,1)
#else
        call zscal(na,xc,tmp2(1,i),1)
!        tmp2(1:na,i) = xc*tmp2(1:na,i)
#endif
      enddo

      !  tmp1 = A*Zi - Zi*EVi
      tmp1(:,:) =  tmp1(:,:) - tmp2(:,:)

      ! Get maximum norm of columns of tmp1
      errmax = 0.0
      do i=1,nev
        xc = 0
#ifdef WITH_MPI
        call pzdotc(na,xc,tmp1,1,i,sc_desc,1,tmp1,1,i,sc_desc,1)
#else
        ! call zdotc(na,xc,1,tmp1,1)
        xc = zdotc(na,tmp1,1,tmp1,1)
#endif
        errmax = max(errmax, sqrt(real(xc,kind=rk)))
      enddo

      ! Get maximum error norm over all processors
      err = errmax
#ifdef WITH_MPI
      call mpi_allreduce(err,errmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
#else
      errmax = err
#endif
      if (myid==0) print *
      if (myid==0) print *,'Error Residual     :',errmax

      if (errmax .gt. 5e-12) then
        status = 1
      endif

      ! 2. Eigenvector orthogonality

      ! tmp1 = Z**T * Z
      tmp1 = 0
#ifdef WITH_MPI
      call pzgemm('C','N',nev,nev,na,CONE,z,1,1,sc_desc, &
                  z,1,1,sc_desc,CZERO,tmp1,1,1,sc_desc)
#else
      call zgemm('C','N',nev,nev,na,CONE,z,na,z,na,CZERO,tmp1,na)
#endif
      ! Initialize tmp2 to unit matrix
      tmp2 = 0
#ifdef WITH_MPI
      call pzlaset('A',nev,nev,CZERO,CONE,tmp2,1,1,sc_desc)
#else
      call zlaset('A',nev,nev,CZERO,CONE,tmp2,na)
#endif
      ! tmp1 = Z**T * Z - Unit Matrix
      tmp1(:,:) =  tmp1(:,:) - tmp2(:,:)

      ! Get maximum error (max abs value in tmp1)
      err = maxval(abs(tmp1))
#ifdef WITH_MPI
      call mpi_allreduce(err,errmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
#else
      errmax = err
#endif
      if (myid==0) print *,'Error Orthogonality:',errmax

      if (errmax .gt. 5e-12) then
        status = 1
      endif
    end function

    function check_correctness_real(na, nev, as, z, ev, sc_desc, myid, tmp1, tmp2) result(status)

      use elpa_mpi
      use precision
      implicit none
      integer(kind=ik)               :: status
      integer(kind=ik), intent(in)   :: na, nev, myid
      real(kind=rk), intent(in)      :: as(:,:), z(:,:)
      real(kind=rk), intent(inout)   :: tmp1(:,:), tmp2(:,:)
      real(kind=rk)                  :: ev(:)
      integer(kind=ik)               :: sc_desc(:), mpierr

      integer(kind=ik)               :: i
      real(kind=rk)                  :: err, errmax
#ifndef WITH_MPI
      real(kind=rk)                  :: dnrm2
#endif

      status = 0

      ! 1. Residual (maximum of || A*Zi - Zi*EVi ||)
      ! tmp1 =  A * Z
#ifdef WITH_MPI
      call pdgemm('N','N',na,nev,na,1.d0,as,1,1,sc_desc, &
                  z,1,1,sc_desc,0.d0,tmp1,1,1,sc_desc)
#else
      call dgemm('N','N',na,nev,na,1.d0,as,na,z,na,0.d0,tmp1,na)
#endif

      ! tmp2 = Zi*EVi
      tmp2(:,:) = z(:,:)
      do i=1,nev
#ifdef WITH_MPI
        call pdscal(na,ev(i),tmp2,1,i,sc_desc,1)
#else
        call dscal(na,ev(i),tmp2(:,i),1)
#endif
      enddo

      !  tmp1 = A*Zi - Zi*EVi
      tmp1(:,:) =  tmp1(:,:) - tmp2(:,:)

      ! Get maximum norm of columns of tmp1
      errmax = 0.0
      do i=1,nev
        err = 0.0
#ifdef WITH_MPI
        call pdnrm2(na,err,tmp1,1,i,sc_desc,1)
#else
!        call dnrm2(na,err,tmp1,1,i,sc_desc,1)
        err = dnrm2(na,tmp1(1,i),1)
#endif
        errmax = max(errmax, err)
      enddo

      ! Get maximum error norm over all processors
      err = errmax

#ifdef WITH_MPI
      call mpi_allreduce(err,errmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
#else
      errmax = err
#endif
      if (myid==0) print *
      if (myid==0) print *,'Error Residual     :',errmax

      if (errmax .gt. 5e-12) then
        status = 1
      endif

      ! 2. Eigenvector orthogonality

      ! tmp1 = Z**T * Z
      tmp1 = 0
#ifdef WITH_MPI
      call pdgemm('T','N',nev,nev,na,1.d0,z,1,1,sc_desc, &
                  z,1,1,sc_desc,0.d0,tmp1,1,1,sc_desc)
#else
      call dgemm('T','N',nev,nev,na,1.d0,z,na, &
                  z,na,0.d0,tmp1,na)
#endif
      ! Initialize tmp2 to unit matrix
      tmp2 = 0
#ifdef WITH_MPI
      call pdlaset('A',nev,nev,0.d0,1.d0,tmp2,1,1,sc_desc)
#else
      call dlaset('A',nev,nev,0.d0,1.d0,tmp2,na)
#endif
      ! tmp1 = Z**T * Z - Unit Matrix
      tmp1(:,:) =  tmp1(:,:) - tmp2(:,:)

      ! Get maximum error (max abs value in tmp1)
      err = maxval(abs(tmp1))
#ifdef WITH_MPI
      call mpi_allreduce(err,errmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
#else
      errmax = err
#endif
      if (myid==0) print *,'Error Orthogonality:',errmax

      if (errmax .gt. 5e-12) then
        status = 1
      endif
    end function

    !c> int check_correctness_real_from_fortran(int na, int nev, int na_rows, int na_cols,
    !c>                                         double *as, double *z, double *ev,
    !c>                                         int sc_desc[9], int myid,
    !c>                                         double *tmp1, double *tmp2);
    function check_correctness_real_wrapper(na, nev, na_rows, na_cols, as, z, ev, sc_desc, myid, tmp1, tmp2) result(status) &
      bind(C,name="check_correctness_real_from_fortran")

      use iso_c_binding

      implicit none

      integer(kind=c_int)         :: status
      integer(kind=c_int), value  :: na, nev, myid, na_rows, na_cols
      real(kind=c_double)         :: as(1:na_rows,1:na_cols), z(1:na_rows,1:na_cols)
      real(kind=c_double)         :: tmp1(1:na_rows,1:na_cols), tmp2(1:na_rows,1:na_cols)
      real(kind=c_double)         :: ev(1:na)
      integer(kind=c_int)         :: sc_desc(1:9)

      status = check_correctness_real(na, nev, as, z, ev, sc_desc, myid, tmp1, tmp2)

    end function
    !c> int check_correctness_complex_from_fortran(int na, int nev, int na_rows, int na_cols,
    !c>                                         complex double *as, complex double *z, double *ev,
    !c>                                         int sc_desc[9], int myid,
    !c>                                         complex double *tmp1, complex double *tmp2);
    function check_correctness_complex_wrapper(na, nev, na_rows, na_cols, as, z, ev, sc_desc, myid, tmp1, tmp2) result(status) &
      bind(C,name="check_correctness_complex_from_fortran")

      use iso_c_binding

      implicit none

      integer(kind=c_int)         :: status
      integer(kind=c_int), value  :: na, nev, myid, na_rows, na_cols
      complex(kind=c_double)      :: as(1:na_rows,1:na_cols), z(1:na_rows,1:na_cols)
      complex(kind=c_double)      :: tmp1(1:na_rows,1:na_cols), tmp2(1:na_rows,1:na_cols)
      real(kind=c_double)         :: ev(1:na)
      integer(kind=c_int)         :: sc_desc(1:9)

      status = check_correctness_complex(na, nev, as, z, ev, sc_desc, myid, tmp1, tmp2)

    end function

end module mod_check_correctness
