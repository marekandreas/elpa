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
! Author: A. Marek, MPCDF

    function check_correctness_evp_numeric_residuals_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    & (na, nev, as, z, ev, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol) result(status)
      implicit none
#include "../../src/general/precision_kinds.F90"
      integer(kind=ik)                 :: status
      integer(kind=ik), intent(in)     :: na, nev, nblk, myid, np_rows, np_cols, my_prow, my_pcol
#if REALCASE == 1
      real(kind=rck), intent(in)     :: as(:,:), z(:,:)
      real(kind=rck)                 :: ev(:)
      real(kind=rck), dimension(size(as,dim=1),size(as,dim=2)) :: tmp1, tmp2
      real(kind=rck)                :: xc

#ifndef WITH_MPI

#ifdef DOUBLE_PRECISION_REAL
      real(kind=rck)                 :: dnrm2
#else
      real(kind=rck)                 :: snrm2
#endif

#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
      complex(kind=rck), intent(in)    :: as(:,:), z(:,:)
      real(kind=rck)                   :: ev(:)
      complex(kind=rck), dimension(size(as,dim=1),size(as,dim=2)) :: tmp1, tmp2
      complex(kind=rck)                :: xc
#ifdef DOUBLE_PRECISION_COMPLEX
#ifndef WITH_MPI
      complex(kind=rck)                :: zdotc, cdotc
#endif

#else /* DOUBLE_PRECISION_COMPLEX */
#ifndef WITH_MPI
      complex(kind=rck)                :: zdotc, cdotc
#endif

#endif /* DOUBLE_PRECISION_COMPLEX */

#endif /* COMPLEXCASE */

      integer(kind=ik)                 :: sc_desc(:)

      integer(kind=ik)                 :: i, rowLocal, colLocal
      real(kind=rck)                   :: err, errmax

      integer :: mpierr

      status = 0

      ! 1. Residual (maximum of || A*Zi - Zi*EVi ||)
      ! tmp1 =  A * Z
      ! as is original stored matrix, Z are the EVs

#ifdef WITH_MPI
      call scal_PRECISION_GEMM('N', 'N', na, nev, na, ONE, as, 1, 1, sc_desc, &
                  z, 1, 1, sc_desc, ZERO, tmp1, 1, 1, sc_desc)
#else /* WITH_MPI */
      call PRECISION_GEMM('N','N',na,nev,na,ONE,as,na,z,na,ZERO,tmp1,na)
#endif /* WITH_MPI */


      ! tmp2 = Zi*EVi
      tmp2(:,:) = z(:,:)
      do i=1,nev
        xc = ev(i)
#if REALCASE == 1
#ifdef WITH_MPI

#ifdef DOUBLE_PRECISION_REAL
        call pdscal(na, xc, tmp2, 1, i, sc_desc, 1)
#else
        call psscal(na, xc, tmp2, 1, i, sc_desc, 1)
#endif

#else /* WITH_MPI */

#ifdef DOUBLE_PRECISION_REAL
        call dscal(na,xc,tmp2(:,i),1)
#else
        call sscal(na,xc,tmp2(:,i),1)
#endif

#endif /* WITH_MPI */
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef WITH_MPI

#ifdef DOUBLE_PRECISION_COMPLEX
        call pzscal(na, xc, tmp2, 1, i, sc_desc, 1)
#else
        call pcscal(na, xc, tmp2, 1, i, sc_desc, 1)
#endif

#else /* WITH_MPI */

#ifdef DOUBLE_PRECISION_COMPLEX
        call zscal(na,xc,tmp2(1,i),1)
#else
        call cscal(na,xc,tmp2(1,i),1)
#endif

#endif /* WITH_MPI */
#endif /* COMPLEXCASE */
      enddo

      !  tmp1 = A*Zi - Zi*EVi
      tmp1(:,:) =  tmp1(:,:) - tmp2(:,:)

      ! Get maximum norm of columns of tmp1
      errmax = 0.0_rk

      do i=1,nev
#if REALCASE == 1
        err = 0.0_rk
#ifdef WITH_MPI
        call scal_PRECISION_NRM2(na, err, tmp1, 1, i, sc_desc, 1)
#else /* WITH_MPI */
        err = PRECISION_NRM2(na,tmp1(1,i),1)
#endif /* WITH_MPI */
        errmax = max(errmax, err)
#endif /* REALCASE */

#if COMPLEXCASE == 1
        xc = 0
#ifdef WITH_MPI
        call scal_PRECISION_DOTC(na, xc, tmp1, 1, i, sc_desc, 1, tmp1, 1, i, sc_desc, 1)
#else /* WITH_MPI */
        xc = PRECISION_DOTC(na,tmp1,1,tmp1,1)
#endif /* WITH_MPI */
        errmax = max(errmax, sqrt(real(xc,kind=REAL_DATATYPE)))
#endif /* COMPLEXCASE */
      enddo

      ! Get maximum error norm over all processors
      err = errmax
#ifdef WITH_MPI
      call mpi_allreduce(err, errmax, 1, MPI_REAL_PRECISION, MPI_MAX, MPI_COMM_WORLD, mpierr)
#else /* WITH_MPI */
      errmax = err
#endif /* WITH_MPI */
      if (myid==0) print *,'Results of numerical residual checks:'
      if (myid==0) print *,'Error Residual     :',errmax
#if REALCASE == 1
      if (nev .ge. 2) then
#ifdef DOUBLE_PRECISION_REAL
        if (errmax .gt. 5e-11_rk8 .or. errmax .eq. 0.0_rk8) then
#else
        if (errmax .gt. 3e-2_rk4 .or. errmax .eq. 0.0_rk4) then
#endif
          status = 1
        endif
      else
#ifdef DOUBLE_PRECISION_REAL
        if (errmax .gt. 5e-11_rk8) then
#else
        if (errmax .gt. 2e-2_rk4) then
#endif
          status = 1
        endif
      endif
#endif
#if COMPLEXCASE == 1
      if (nev .gt. 2) then
#ifdef DOUBLE_PRECISION_COMPLEX
        if (errmax .gt. 5e-11_rk8 .or. errmax .eq. 0.0_rk8) then
#else
        if (errmax .gt. 3e-2_rk4 .or. errmax .eq. 0.0_rk4) then
#endif
          status =1
        endif
      else
#ifdef DOUBLE_PRECISION_COMPLEX
        if (errmax .gt. 5e-11_rk8) then
#else
        if (errmax .gt. 3e-2_rk4) then
#endif
          status = 1
        endif
      endif
#endif

      ! 2. Eigenvector orthogonality

      ! tmp1 = Z**T * Z
      tmp1 = 0
#ifdef WITH_MPI
      call scal_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', nev, nev, na, ONE, z, 1, 1, &
                        sc_desc, z, 1, 1, sc_desc, ZERO, tmp1, 1, 1, sc_desc)
#else /* WITH_MPI */
      call PRECISION_GEMM(BLAS_TRANS_OR_CONJ,'N',nev,nev,na,ONE,z,na,z,na,ZERO,tmp1,na)
#endif /* WITH_MPI */
      !TODO for the C interface, not all information is passed (zeros instead)
      !TODO than this part of the test cannot be done
      !TODO either we will not have this part of test at all, or it will be improved
      if(nblk > 0) then
        ! First check, whether the elements on diagonal are 1 .. "normality" of the vectors
        err = CONST_REAL_0_0
        do i=1, nev
          if (map_global_array_index_to_local_index(i, i, rowLocal, colLocal, nblk, np_rows, np_cols, my_prow, my_pcol)) then
             err = max(err, abs(tmp1(rowLocal,colLocal) - CONST_REAL_1_0))
           endif
        end do
#ifdef WITH_MPI
        call mpi_allreduce(err, errmax, 1, MPI_REAL_PRECISION, MPI_MAX, MPI_COMM_WORLD, mpierr)
#else /* WITH_MPI */
        errmax = err
#endif /* WITH_MPI */
        if (myid==0) print *,'Maximal error in eigenvector lengths:',errmax
      end if

      ! Second, find the maximal error in the whole Z**T * Z matrix (its diference from identity matrix)
      ! Initialize tmp2 to unit matrix
      tmp2 = 0
#ifdef WITH_MPI
      call scal_PRECISION_LASET('A', nev, nev, ZERO, ONE, tmp2, 1, 1, sc_desc)
#else /* WITH_MPI */
      call PRECISION_LASET('A',nev,nev,ZERO,ONE,tmp2,na)
#endif /* WITH_MPI */

      !      ! tmp1 = Z**T * Z - Unit Matrix
      tmp1(:,:) =  tmp1(:,:) - tmp2(:,:)

      ! Get maximum error (max abs value in tmp1)
      err = maxval(abs(tmp1))
#ifdef WITH_MPI
      call mpi_allreduce(err, errmax, 1, MPI_REAL_PRECISION, MPI_MAX, MPI_COMM_WORLD, mpierr)
#else /* WITH_MPI */
      errmax = err
#endif /* WITH_MPI */

      if (myid==0) print *,'Error Orthogonality:',errmax
#if REALCASE == 1
      if (nev .ge. 2) then
#ifdef DOUBLE_PRECISION_REAL
        if (errmax .gt. 5e-12_rk8 .or. errmax .eq. 0.0_rk8) then
#else
        if (errmax .gt. 9e-4_rk4 .or. errmax .eq. 0.0_rk4) then
#endif
          status = 1
        endif
      else
#ifdef DOUBLE_PRECISION_REAL
        if (errmax .gt. 5e-11_rk8) then
#else
        if (errmax .gt. 9e-2_rk4) then
#endif
          status = 1
        endif
      endif
#endif
#if COMPLEXCASE == 1
      if (nev .ge. 2) then
#ifdef DOUBLE_PRECISION_COMPLEX
        if (errmax .gt. 5e-12_rk8 .or. errmax .eq. 0.0_rk8) then
#else
        if (errmax .gt. 9e-4_rk4 .or. errmax .eq. 0.0_rk4) then
#endif
          status = 1
        endif
      else
#ifdef DOUBLE_PRECISION_COMPLEX
        if (errmax .gt. 5e-11_rk8) then
#else
        if (errmax .gt. 9e-3_rk4) then
#endif
          status = 1
        endif
      endif
#endif

    end function


#if REALCASE == 1

#ifdef DOUBLE_PRECISION_REAL
    !c> int check_correctness_evp_numeric_residuals_real_double_f(int na, int nev, int na_rows, int na_cols,
    !c>                                         double *as, double *z, double *ev,
    !c>                                         int sc_desc[9], int myid);
#else
    !c> int check_correctness_evp_numeric_residuals_real_single_f(int na, int nev, int na_rows, int na_cols,
    !c>                                         float *as, float *z, float *ev,
    !c>                                         int sc_desc[9], int myid);
#endif

#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> int check_correctness_evp_numeric_residuals_complex_double_f(int na, int nev, int na_rows, int na_cols,
    !c>                                         complex double *as, complex double *z, double *ev,
    !c>                                         int sc_desc[9], int myid);
#else
    !c> int check_correctness_evp_numeric_residuals_complex_single_f(int na, int nev, int na_rows, int na_cols,
    !c>                                         complex float *as, complex float *z, float *ev,
    !c>                                         int sc_desc[9], int myid);
#endif
#endif /* COMPLEXCASE */

function check_correctness_evp_numeric_residuals_&
&MATH_DATATYPE&
&_&
&PRECISION&
&_f (na, nev, na_rows, na_cols, as, z, ev, sc_desc, myid) result(status) &
      bind(C,name="check_correctness_evp_numeric_residuals_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      &_f")

      use iso_c_binding

      implicit none
#include "../../src/general/precision_kinds.F90"

      integer(kind=c_int)            :: status
      integer(kind=c_int), value     :: na, nev, myid, na_rows, na_cols
      MATH_DATATYPE(kind=rck)     :: as(1:na_rows,1:na_cols), z(1:na_rows,1:na_cols)
      real(kind=rck)    :: ev(1:na)
      integer(kind=c_int)            :: sc_desc(1:9)

      ! TODO: I did not want to add all the variables to the C interface as well
      ! TODO: I think that we should find a better way to pass this information
      ! TODO: to all the functions anyway (get it from sc_desc, pass elpa_t, etc..)
      status = check_correctness_evp_numeric_residuals_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      & (na, nev, as, z, ev, sc_desc, 0, myid, 0, 0, 0, 0)

    end function

    function check_correctness_eigenvalues_toeplitz_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    & (na, diagonalElement, subdiagonalElement, ev, z, myid) result(status)
      use iso_c_binding
      implicit none
#include "../../src/general/precision_kinds.F90"

      integer               :: status, ii, j, myid
      integer, intent(in)   :: na
      real(kind=rck) :: diagonalElement, subdiagonalElement
      real(kind=rck) :: ev_analytic(na), ev(na)
      MATH_DATATYPE(kind=rck) :: z(:,:)

#if defined(DOUBLE_PRECISION_REAL) || defined(DOUBLE_PRECISION_COMPLEX)
      real(kind=rck), parameter   :: pi = 3.141592653589793238462643383279_c_double
#else
      real(kind=rck), parameter   :: pi = 3.1415926535897932_c_float
#endif
      real(kind=rck)              :: tmp, maxerr
      integer                     :: loctmp
      status = 0

     ! analytic solution
     do ii=1, na
#if defined(DOUBLE_PRECISION_REAL) || defined(DOUBLE_PRECISION_COMPLEX)
       ev_analytic(ii) = diagonalElement + 2.0_c_double * &
                         subdiagonalElement *cos( pi*real(ii,kind=c_double)/ &
			 real(na+1,kind=c_double) )
#else
       ev_analytic(ii) = diagonalElement + 2.0_c_float * &
                         subdiagonalElement *cos( pi*real(ii,kind=c_float)/ &
                         real(na+1,kind=c_float) )
#endif
     enddo

     ! sort analytic solution:

     ! this hack is neither elegant, nor optimized: for huge matrixes it might be expensive
     ! a proper sorting algorithmus might be implemented here

     tmp    = minval(ev_analytic)
     loctmp = minloc(ev_analytic, 1)

     ev_analytic(loctmp) = ev_analytic(1)
     ev_analytic(1) = tmp
     do ii=2, na
       tmp = ev_analytic(ii)
       do j= ii, na
         if (ev_analytic(j) .lt. tmp) then
           tmp    = ev_analytic(j)
           loctmp = j
         endif
       enddo
       ev_analytic(loctmp) = ev_analytic(ii)
       ev_analytic(ii) = tmp
     enddo

     ! compute a simple error max of eigenvalues
     maxerr = 0.0
     maxerr = maxval( (ev(:) - ev_analytic(:))/ev_analytic(:) , 1)

#if defined(DOUBLE_PRECISION_REAL) || defined(DOUBLE_PRECISION_COMPLEX)
     if (maxerr .gt. 8.e-13_c_double .or. maxerr .eq. 0.0_c_double) then
#else
     if (maxerr .gt. 8.e-4_c_float .or. maxerr .eq. 0.0_c_float) then
#endif
       status = 1
       if (myid .eq. 0) then
         print *,"Result of Toeplitz matrix test: "
         print *,"Eigenvalues differ from analytic solution: maxerr = ",maxerr
       endif
     endif

    if (status .eq. 0) then
       if (myid .eq. 0) then
         print *,"Result of Toeplitz matrix test: test passed"
         print *,"Eigenvalues differ from analytic solution: maxerr = ",maxerr
       endif
    endif
    end function

    function check_correctness_cholesky_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    & (na, a, as, na_rows, sc_desc, myid) result(status)
      implicit none
#include "../../src/general/precision_kinds.F90"
      integer(kind=ik)                 :: status
      integer(kind=ik), intent(in)     :: na, myid, na_rows
#if REALCASE == 1
      real(kind=rck), intent(in)       :: a(:,:), as(:,:)
      real(kind=rck), dimension(size(as,dim=1),size(as,dim=2)) :: tmp1, tmp2
      real(kind=rck)                   :: norm, normmax

#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_REAL
      real(kind=rck)                   :: pdlange
#else
      real(kind=rck)                   :: pslange
#endif

#else /* WITH_MPI */

#ifdef DOUBLE_PRECISION_REAL
      real(kind=rck)                   :: dlange
#else
      real(kind=rck)                   :: slange
#endif

#endif /* WITH_MPI */

#endif /* REALCASE */

#if COMPLEXCASE == 1
      complex(kind=rck), intent(in)    :: a(:,:), as(:,:)
      complex(kind=rck), dimension(size(as,dim=1),size(as,dim=2)) :: tmp1, tmp2
      real(kind=rck)                :: norm, normmax
#ifdef DOUBLE_PRECISION_COMPLEX
      complex(kind=ck8), parameter   :: CZERO = (0.0_rk8,0.0_rk8), CONE = (1.0_rk8,0.0_rk8)
#else
      complex(kind=ck4), parameter   :: CZERO = (0.0_rk4,0.0_rk4), CONE = (1.0_rk4,0.0_rk8)
#endif

#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_COMPLEX
      real(kind=rck)                   :: pzlange
#else
      real(kind=rck)                   :: pclange
#endif

#else /* WITH_MPI */

#ifdef DOUBLE_PRECISION_COMPLEX
      real(kind=rck)                   :: zlange
#else
      real(kind=rck)                   :: clange
#endif

#endif /* WITH_MPI */
#endif /* COMPLEXCASE */

      integer(kind=ik)                 :: sc_desc(:)

      real(kind=rck)                   :: err, errmax

      integer :: mpierr

      status = 0

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
      tmp1(:,:) = 0.0_rk8
#else
      tmp1(:,:) = 0.0_rk4
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
      tmp1(:,:) = 0.0_ck8
#else
      tmp1(:,:) = 0.0_ck4
#endif
#endif /* COMPLEXCASE */


#if REALCASE == 1
      ! tmp1 = a**T
#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_REAL
      call pdtran(na, na, 1.0_rk8, a, 1, 1, sc_desc, 0.0_rk8, tmp1, 1, 1, sc_desc)
#else

      call pstran(na, na, 1.0_rk4, a, 1, 1, sc_desc, 0.0_rk4, tmp1, 1, 1, sc_desc)
#endif

#else /* WITH_MPI */
      tmp1 = transpose(a)
#endif /* WITH_MPI */

      ! tmp2 = a * a**T
#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_REAL
      call pdgemm("N","N", na, na, na, 1.0_rk8, a, 1, 1, sc_desc, tmp1, 1, 1, &
               sc_desc, 0.0_rk8, tmp2, 1, 1, sc_desc)
#else
      call psgemm("N","N", na, na, na, 1.0_rk4, a, 1, 1, sc_desc, tmp1, 1, 1, &
               sc_desc, 0.0_rk4, tmp2, 1, 1, sc_desc)

#endif

#else /* WITH_MPI */

#ifdef DOUBLE_PRECISION_REAL
      call dgemm("N","N", na, na, na, 1.0_rk8, a, na, tmp1, na, 0.0_rk8, tmp2, na)
#else
      call sgemm("N","N", na, na, na, 1.0_rk4, a, na, tmp1, na, 0.0_rk4, tmp2, na)
#endif
#endif /* WITH_MPI */


#endif /* REALCASE == 1 */

#if COMPLEXCASE == 1
      ! tmp1 = a**H
#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_COMPLEX
      call pztranc(na, na, CONE, a, 1, 1, sc_desc, CZERO, tmp1, 1, 1, sc_desc)
#else

      call pctranc(na, na, CONE, a, 1, 1, sc_desc, CZERO, tmp1, 1, 1, sc_desc)
#endif

#else /* WITH_MPI */
      tmp1 = transpose(conjg(a))
#endif /* WITH_MPI */

      ! tmp2 = a * a**T
#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_COMPLEX
      call pzgemm("N","N", na, na, na, CONE, a, 1, 1, sc_desc, tmp1, 1, 1, &
               sc_desc, CZERO, tmp2, 1, 1, sc_desc)
#else
      call pcgemm("N","N", na, na, na, CONE, a, 1, 1, sc_desc, tmp1, 1, 1, &
               sc_desc, CZERO, tmp2, 1, 1, sc_desc)

#endif

#else /* WITH_MPI */

#ifdef DOUBLE_PRECISION_COMPLEX
      call zgemm("N","N", na, na, na, CONE, a, na, tmp1, na, CZERO, tmp2, na)
#else
      call cgemm("N","N", na, na, na, CONE, a, na, tmp1, na, CZERO, tmp2, na)
#endif
#endif /* WITH_MPI */


#endif /* COMPLEXCASE == 1 */

      ! compare tmp2 with original matrix
      tmp2(:,:) = tmp2(:,:) - as(:,:)

#if REALCASE == 1

#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_REAL

      norm = pdlange("M",na, na, tmp2, 1, 1, sc_desc, tmp1)
#else
      norm = pslange("M",na, na, tmp2, 1, 1, sc_desc, tmp1)
#endif

#else /* WITH_MPI */
#ifdef DOUBLE_PRECISION_REAL
      norm = dlange("M", na, na, tmp2, na_rows, tmp1)
#else
      norm = slange("M", na, na, tmp2, na_rows, tmp1)

#endif
#endif /* WITH_MPI */


#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_REAL
      call mpi_allreduce(norm,normmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
#else
      call mpi_allreduce(norm,normmax,1,MPI_REAL4,MPI_MAX,MPI_COMM_WORLD,mpierr)
#endif
#else /* WITH_MPI */
      normmax = norm
#endif /* WITH_MPI */


#endif /* REALCASE == 1 */

#if COMPLEXCASE == 1

#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_COMPLEX

      norm = pzlange("M",na, na, tmp2, 1, 1, sc_desc, tmp1)
#else
      norm = pclange("M",na, na, tmp2, 1, 1, sc_desc, tmp1)
#endif

#else /* WITH_MPI */
#ifdef DOUBLE_PRECISION_COMPLEX
      norm = zlange("M", na, na, tmp2, na_rows, tmp1)
#else
      norm = clange("M", na, na, tmp2, na_rows, tmp1)

#endif
#endif /* WITH_MPI */


#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_COMPLEX
      call mpi_allreduce(norm,normmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
#else
      call mpi_allreduce(norm,normmax,1,MPI_REAL4,MPI_MAX,MPI_COMM_WORLD,mpierr)
#endif
#else /* WITH_MPI */
      normmax = norm
#endif /* WITH_MPI */


#endif /* COMPLEXCASE == 1 */

      if (myid .eq. 0) then
        print *," Maximum error of result: ", normmax
      endif

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
!      if (normmax .gt. 5e-12_rk8 .or. normmax .eq. 0.0_rk8) then
      if (normmax .gt. 5e-12_rk8) then
        status = 1
      endif
#else
!      if (normmax .gt. 5e-4_rk4 .or. normmax .eq. 0.0_rk4) then
      if (normmax .gt. 5e-4_rk4 ) then
        status = 1
      endif
#endif
#endif

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
!      if (normmax .gt. 5e-11_rk8 .or. normmax .eq. 0.0_rk8) then
      if (normmax .gt. 5e-11_rk8 ) then
        status = 1
      endif
#else
!      if (normmax .gt. 5e-3_rk4 .or. normmax .eq. 0.0_rk4) then
      if (normmax .gt. 5e-3_rk4) then
        status = 1
      endif
#endif
#endif
    end function

    function check_correctness_hermitian_multiply_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    & (na, a, b, c, na_rows, sc_desc, myid) result(status)
      implicit none
#include "../../src/general/precision_kinds.F90"
      integer(kind=ik)                 :: status
      integer(kind=ik), intent(in)     :: na, myid, na_rows
#if REALCASE == 1
      real(kind=rck), intent(in)       :: a(:,:), b(:,:), c(:,:)
      real(kind=rck), dimension(size(a,dim=1),size(a,dim=2)) :: tmp1, tmp2
      real(kind=rck)                   :: norm, normmax

#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_REAL
      real(kind=rck)                   :: pdlange
#else
      real(kind=rck)                   :: pslange
#endif

#else /* WITH_MPI */

#ifdef DOUBLE_PRECISION_REAL
      real(kind=rck)                   :: dlange
#else
      real(kind=rck)                   :: slange
#endif

#endif /* WITH_MPI */

#endif /* REALCASE */

#if COMPLEXCASE == 1
      complex(kind=rck), intent(in)    :: a(:,:), b(:,:), c(:,:)
      complex(kind=rck), dimension(size(a,dim=1),size(a,dim=2)) :: tmp1, tmp2
      real(kind=rck)                :: norm, normmax
#ifdef DOUBLE_PRECISION_COMPLEX
      complex(kind=ck8), parameter   :: CZERO = (0.0_rk8,0.0_rk8), CONE = (1.0_rk8,0.0_rk8)
#else
      complex(kind=ck4), parameter   :: CZERO = (0.0_rk4,0.0_rk4), CONE = (1.0_rk4,0.0_rk4)
#endif

#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_COMPLEX
      real(kind=rck)                   :: pzlange
#else
      real(kind=rck)                   :: pclange
#endif

#else /* WITH_MPI */

#ifdef DOUBLE_PRECISION_COMPLEX
      real(kind=rck)                   :: zlange
#else
      real(kind=rck)                   :: clange
#endif

#endif /* WITH_MPI */
#endif /* COMPLEXCASE */

      integer(kind=ik)                 :: sc_desc(:)

      real(kind=rck)                   :: err, errmax

      integer :: mpierr

      status = 0

#if REALCASE == 1

#ifdef DOUBLE_PRECISION_REAL
      tmp1(:,:) = 0.0_rk8
#else
      tmp1(:,:) = 0.0_rk4
#endif

#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
      tmp1(:,:) = (0.0_c_double, 0.0_c_double)
#else
      tmp1(:,:) = (0.0_c_float, 0.0_c_float)
#endif
#endif /* COMPLEXCASE */


#if REALCASE == 1
      ! tmp1 = a**T
#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_REAL
      call pdtran(na, na, 1.0_rk8, a, 1, 1, sc_desc, 0.0_rk8, tmp1, 1, 1, sc_desc)
#else

      call pstran(na, na, 1.0_rk4, a, 1, 1, sc_desc, 0.0_rk4, tmp1, 1, 1, sc_desc)
#endif

#else /* WITH_MPI */
      tmp1 = transpose(a)
#endif /* WITH_MPI */

   ! tmp2 = tmp1 * b
#ifdef DOUBLE_PRECISION_REAL
#ifdef WITH_MPI
   call pdgemm("N","N", na, na, na, 1.0_rk8, tmp1, 1, 1, sc_desc, b, 1, 1, &
               sc_desc, 0.0_rk8, tmp2, 1, 1, sc_desc)
#else
   call dgemm("N","N", na, na, na, 1.0_rk8, tmp1, na, b, na, 0.0_rk8, tmp2, na)
#endif

#else /* DOUBLE_PRECISION_REAL */
#ifdef WITH_MPI
   call psgemm("N","N", na, na, na, 1.0_rk4, tmp1, 1, 1, sc_desc, b, 1, 1, &
               sc_desc, 0.0_rk4, tmp2, 1, 1, sc_desc)
#else
   call sgemm("N","N", na, na, na, 1.0_rk4, tmp1, na, b, na, 0.0_rk4, tmp2, na)
#endif

#endif /* DOUBLE_PRECISION_REAL */

#endif /* REALCASE == 1 */

#if COMPLEXCASE == 1
      ! tmp1 = a**H
#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_COMPLEX
      call pztranc(na, na, CONE, a, 1, 1, sc_desc, CZERO, tmp1, 1, 1, sc_desc)
#else

      call pctranc(na, na, CONE, a, 1, 1, sc_desc, CZERO, tmp1, 1, 1, sc_desc)
#endif

#else /* WITH_MPI */
      tmp1 = transpose(conjg(a))
#endif /* WITH_MPI */

   ! tmp2 = tmp1 * b
#ifdef DOUBLE_PRECISION_COMPLEX
#ifdef WITH_MPI
   call pzgemm("N","N", na, na, na, CONE, tmp1, 1, 1, sc_desc, b, 1, 1, &
               sc_desc, CZERO, tmp2, 1, 1, sc_desc)
#else
   call zgemm("N","N", na, na, na, CONE, tmp1, na, b, na, CZERO, tmp2, na)
#endif

#else /* DOUBLE_PRECISION_COMPLEX */

#ifdef WITH_MPI
   call pcgemm("N","N", na, na, na, CONE, tmp1, 1, 1, sc_desc, b, 1, 1, &
               sc_desc, CZERO, tmp2, 1, 1, sc_desc)
#else
   call cgemm("N","N", na, na, na, CONE, tmp1, na, b, na, CZERO, tmp2, na)
#endif
#endif /* DOUBLE_PRECISION_COMPLEX */

#endif /* COMPLEXCASE == 1 */

      ! compare tmp2 with c
      tmp2(:,:) = tmp2(:,:) - c(:,:)
#if REALCASE == 1

#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_REAL

      norm = pdlange("M",na, na, tmp2, 1, 1, sc_desc, tmp1)
#else
      norm = pslange("M",na, na, tmp2, 1, 1, sc_desc, tmp1)
#endif

#else /* WITH_MPI */
#ifdef DOUBLE_PRECISION_REAL
      norm = dlange("M", na, na, tmp2, na_rows, tmp1)
#else
      norm = slange("M", na, na, tmp2, na_rows, tmp1)

#endif
#endif /* WITH_MPI */


#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_REAL
      call mpi_allreduce(norm,normmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
#else
      call mpi_allreduce(norm,normmax,1,MPI_REAL4,MPI_MAX,MPI_COMM_WORLD,mpierr)
#endif
#else /* WITH_MPI */
      normmax = norm
#endif /* WITH_MPI */


#endif /* REALCASE == 1 */

#if COMPLEXCASE == 1

#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_COMPLEX
      norm = pzlange("M",na, na, tmp2, 1, 1, sc_desc, tmp1)
#else
      norm = pclange("M",na, na, tmp2, 1, 1, sc_desc, tmp1)
#endif

#else /* WITH_MPI */
#ifdef DOUBLE_PRECISION_COMPLEX
      norm = zlange("M", na, na, tmp2, na_rows, tmp1)
#else
      norm = clange("M", na, na, tmp2, na_rows, tmp1)

#endif
#endif /* WITH_MPI */


#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_COMPLEX
      call mpi_allreduce(norm,normmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
#else
      call mpi_allreduce(norm,normmax,1,MPI_REAL4,MPI_MAX,MPI_COMM_WORLD,mpierr)
#endif
#else /* WITH_MPI */
      normmax = norm
#endif /* WITH_MPI */


#endif /* REALCASE == 1 */

      if (myid .eq. 0) then
        print *," Maximum error of result: ", normmax
      endif

#ifdef DOUBLE_PRECISION_REAL
      if (normmax .gt. 5e-11_rk8 ) then
        status = 1
      endif
#else
      if (normmax .gt. 5e-3_rk4 ) then
        status = 1
      endif
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
      if (normmax .gt. 5e-11_rk8 ) then
        status = 1
      endif
#else
      if (normmax .gt. 5e-3_rk4 ) then
        status = 1
      endif
#endif
    end function

    function check_correctness_eigenvalues_frank_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    & (na, ev, z, myid) result(status)
      use iso_c_binding
      implicit none
#include "../../src/general/precision_kinds.F90"

      integer                   :: status, i, j, myid
      integer, intent(in)       :: na
      real(kind=rck)            :: ev_analytic(na), ev(na)
      MATH_DATATYPE(kind=rck)   :: z(:,:)

#if defined(DOUBLE_PRECISION_REAL) || defined(DOUBLE_PRECISION_COMPLEX)
      real(kind=rck), parameter :: pi = 3.141592653589793238462643383279_c_double
#else
      real(kind=rck), parameter :: pi = 3.1415926535897932_c_float
#endif
      real(kind=rck)            :: tmp, maxerr
      integer                   :: loctmp
      status = 0

     ! analytic solution
     do i = 1, na
       j = na - i
#if defined(DOUBLE_PRECISION_REAL) || defined(DOUBLE_PRECISION_COMPLEX)
       ev_analytic(i) = pi * (2.0_c_double * real(j,kind=c_double) + 1.0_c_double) / &
           (2.0_c_double * real(na,kind=c_double) + 1.0_c_double)
       ev_analytic(i) = 0.5_c_double / (1.0_c_double - cos(ev_analytic(i)))
#else
       ev_analytic(i) = pi * (2.0_c_float * real(j,kind=c_float) + 1.0_c_float) / &
           (2.0_c_float * real(na,kind=c_float) + 1.0_c_float)
       ev_analytic(i) = 0.5_c_float / (1.0_c_float - cos(ev_analytic(i)))
#endif
     enddo

     ! sort analytic solution:

     ! this hack is neither elegant, nor optimized: for huge matrixes it might be expensive
     ! a proper sorting algorithmus might be implemented here

     tmp    = minval(ev_analytic)
     loctmp = minloc(ev_analytic, 1)

     ev_analytic(loctmp) = ev_analytic(1)
     ev_analytic(1) = tmp
     do i=2, na
       tmp = ev_analytic(i)
       do j= i, na
         if (ev_analytic(j) .lt. tmp) then
           tmp    = ev_analytic(j)
           loctmp = j
         endif
       enddo
       ev_analytic(loctmp) = ev_analytic(i)
       ev_analytic(i) = tmp
     enddo

     ! compute a simple error max of eigenvalues
     maxerr = 0.0
     maxerr = maxval( (ev(:) - ev_analytic(:))/ev_analytic(:) , 1)

#if defined(DOUBLE_PRECISION_REAL) || defined(DOUBLE_PRECISION_COMPLEX)
     if (maxerr .gt. 8.e-13_c_double) then
#else
     if (maxerr .gt. 8.e-4_c_float) then
#endif
       status = 1
       if (myid .eq. 0) then
         print *,"Result of Frank matrix test: "
         print *,"Eigenvalues differ from analytic solution: maxerr = ",maxerr
       endif
     endif
    end function

! vim: syntax=fortran
