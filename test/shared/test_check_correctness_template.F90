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
    & (na, nev, as, z, ev, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol, bs) result(status)
 
      use elpa_blas_interfaces
      use elpa_scalapack_interfaces

      implicit none
#include "../../src/general/precision_kinds.F90"
      integer(kind=ik)                 :: status
      integer(kind=ik), intent(in)     :: na, nev, nblk, myid, np_rows, np_cols, my_prow, my_pcol
      MATH_DATATYPE(kind=rck), intent(in)           :: as(:,:), z(:,:)
      MATH_DATATYPE(kind=rck), intent(in), optional :: bs(:,:)
      real(kind=rk)                 :: ev(:)
      MATH_DATATYPE(kind=rck), dimension(size(as,dim=1),size(as,dim=2)) :: tmp1, tmp2
      MATH_DATATYPE(kind=rck)                :: xc

#ifndef WITH_MPI
#if COMPLEXCASE == 1
      complex(kind=rck)                :: zdotc, cdotc
#endif /* COMPLEXCASE */
#endif

      integer(kind=ik)                 :: sc_desc(:)

      integer(kind=ik)                 :: i, rowLocal, colLocal
      real(kind=rck)                   :: err, errmax

      integer :: mpierr

      ! tolerance for the residual test for different math type/precision setups
      real(kind=rk), parameter       :: tol_res_real_double      = 5e-12_rk
      real(kind=rk), parameter       :: tol_res_real_single      = 3e-2_rk
      real(kind=rk), parameter       :: tol_res_complex_double   = 5e-12_rk
      real(kind=rk), parameter       :: tol_res_complex_single   = 3e-2_rk
      real(kind=rk)                  :: tol_res                  = tol_res_&
                                                                          &MATH_DATATYPE&
                                                                          &_&
                                                                          &PRECISION
      ! precision of generalized problem is lower
      real(kind=rk), parameter       :: generalized_penalty = 10.0_rk

      ! tolerance for the orthogonality test for different math type/precision setups
      real(kind=rk), parameter       :: tol_orth_real_double     = 5e-11_rk
      real(kind=rk), parameter       :: tol_orth_real_single     = 9e-2_rk
      real(kind=rk), parameter       :: tol_orth_complex_double  = 5e-11_rk
      real(kind=rk), parameter       :: tol_orth_complex_single  = 9e-3_rk
      real(kind=rk), parameter       :: tol_orth                 = tol_orth_&
                                                                          &MATH_DATATYPE&
                                                                          &_&
                                                                          &PRECISION

      if(present(bs)) then
          tol_res = generalized_penalty * tol_res
      endif
      status = 0

      ! 1. Residual (maximum of || A*Zi - Zi*EVi ||)

      ! tmp1 = Zi*EVi
      tmp1(:,:) = z(:,:)
      do i=1,nev
        xc = ev(i)
#ifdef WITH_MPI
        call p&
            &BLAS_CHAR&
            &scal(na, xc, tmp1, 1, i, sc_desc, 1)
#else /* WITH_MPI */
        call BLAS_CHAR&
            &scal(na,xc,tmp1(:,i),1)
#endif /* WITH_MPI */
      enddo

      ! for generalized EV problem, multiply by bs as well
      ! tmp2 = B * tmp1
      if(present(bs)) then
#ifdef WITH_MPI
      call scal_PRECISION_GEMM('N', 'N', na, nev, na, ONE, bs, 1, 1, sc_desc, &
                  tmp1, 1, 1, sc_desc, ZERO, tmp2, 1, 1, sc_desc)
#else /* WITH_MPI */
      call PRECISION_GEMM('N','N',na,nev,na,ONE,bs,na,tmp1,na,ZERO,tmp2,na)
#endif /* WITH_MPI */
      else
        ! normal eigenvalue problem .. no need to multiply
        tmp2(:,:) = tmp1(:,:)
      end if

      ! tmp1 =  A * Z
      ! as is original stored matrix, Z are the EVs
#ifdef WITH_MPI
      call scal_PRECISION_GEMM('N', 'N', na, nev, na, ONE, as, 1, 1, sc_desc, &
                  z, 1, 1, sc_desc, ZERO, tmp1, 1, 1, sc_desc)
#else /* WITH_MPI */
      call PRECISION_GEMM('N','N',na,nev,na,ONE,as,na,z,na,ZERO,tmp1,na)
#endif /* WITH_MPI */

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
      if (nev .ge. 2) then
        if (errmax .gt. tol_res .or. errmax .eq. 0.0_rk) then
          status = 1
        endif
      else
        if (errmax .gt. tol_res) then
          status = 1
        endif
      endif

      ! 2. Eigenvector orthogonality
      if(present(bs)) then
        !for the generalized EVP, the eigenvectors should be B-orthogonal, not orthogonal
        ! tmp2 = B * Z
        tmp2(:,:) = 0.0_rck
#ifdef WITH_MPI
        call scal_PRECISION_GEMM('N', 'N', na, nev, na, ONE, bs, 1, 1, &
                        sc_desc, z, 1, 1, sc_desc, ZERO, tmp2, 1, 1, sc_desc)
#else /* WITH_MPI */
        call PRECISION_GEMM('N','N', na, nev, na, ONE, bs, na, z, na, ZERO, tmp2, na)
#endif /* WITH_MPI */

      else
        tmp2(:,:) = z(:,:)
      endif
      ! tmp1 = Z**T * tmp2
      ! actually tmp1 = Z**T * Z for standard case and tmp1 = Z**T * B * Z for generalized
      tmp1 = 0
#ifdef WITH_MPI
      call scal_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', nev, nev, na, ONE, z, 1, 1, &
                        sc_desc, tmp2, 1, 1, sc_desc, ZERO, tmp1, 1, 1, sc_desc)
#else /* WITH_MPI */
      call PRECISION_GEMM(BLAS_TRANS_OR_CONJ,'N',nev,nev,na,ONE,z,na,tmp2,na,ZERO,tmp1,na)
#endif /* WITH_MPI */
      ! First check, whether the elements on diagonal are 1 .. "normality" of the vectors
      err = 0.0_rk
      do i=1, nev
        if (map_global_array_index_to_local_index(i, i, rowLocal, colLocal, nblk, np_rows, np_cols, my_prow, my_pcol)) then
           err = max(err, abs(tmp1(rowLocal,colLocal) - 1.0_rk))
         endif
      end do
#ifdef WITH_MPI
      call mpi_allreduce(err, errmax, 1, MPI_REAL_PRECISION, MPI_MAX, MPI_COMM_WORLD, mpierr)
#else /* WITH_MPI */
      errmax = err
#endif /* WITH_MPI */
      if (myid==0) print *,'Maximal error in eigenvector lengths:',errmax

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

      if (nev .ge. 2) then
        if (errmax .gt. tol_orth .or. errmax .eq. 0.0_rk) then
          status = 1
        endif
      else
        if (errmax .gt. tol_orth) then
          status = 1
        endif
      endif
    end function


#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
    !c> int check_correctness_evp_numeric_residuals_real_double_f(int na, int nev, int na_rows, int na_cols,
    !c>                                         double *as, double *z, double *ev, int sc_desc[9],
    !c>                                         int nblk, int myid, int np_rows, int np_cols, int my_prow, int my_pcol);
#else
    !c> int check_correctness_evp_numeric_residuals_real_single_f(int na, int nev, int na_rows, int na_cols,
    !c>                                         float *as, float *z, float *ev, int sc_desc[9],
    !c>                                         int nblk, int myid, int np_rows, int np_cols, int my_prow, int my_pcol);
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> int check_correctness_evp_numeric_residuals_complex_double_f(int na, int nev, int na_rows, int na_cols,
    !c>                                         complex double *as, complex double *z, double *ev, int sc_desc[9],
    !c>                                         int nblk, int myid, int np_rows, int np_cols, int my_prow, int my_pcol);
#else
    !c> int check_correctness_evp_numeric_residuals_complex_single_f(int na, int nev, int na_rows, int na_cols,
    !c>                                         complex float *as, complex float *z, float *ev, int sc_desc[9],
    !c>                                         int nblk, int myid, int np_rows, int np_cols, int my_prow, int my_pcol);
#endif
#endif /* COMPLEXCASE */

function check_correctness_evp_numeric_residuals_&
&MATH_DATATYPE&
&_&
&PRECISION&
&_f (na, nev, na_rows, na_cols, as, z, ev, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol) result(status) &
      bind(C,name="check_correctness_evp_numeric_residuals_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      &_f")

      use iso_c_binding

      implicit none
#include "../../src/general/precision_kinds.F90"

      integer(kind=c_int)            :: status
      integer(kind=c_int), value     :: na, nev, myid, na_rows, na_cols, nblk, np_rows, np_cols, my_prow, my_pcol
      MATH_DATATYPE(kind=rck)     :: as(1:na_rows,1:na_cols), z(1:na_rows,1:na_cols)
      real(kind=rck)    :: ev(1:na)
      integer(kind=c_int)            :: sc_desc(1:9)

      status = check_correctness_evp_numeric_residuals_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      & (na, nev, as, z, ev, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol)

    end function

!---- variant for the generalized eigenproblem
!---- unlike in Fortran, we cannot use optional parameter
!---- we thus define a different function
#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
    !c> int check_correctness_evp_gen_numeric_residuals_real_double_f(int na, int nev, int na_rows, int na_cols,
    !c>                                         double *as, double *z, double *ev, int sc_desc[9],
    !c>                                         int nblk, int myid, int np_rows, int np_cols, int my_prow, int my_pcol,
    !c>                                         double *bs);
#else
    !c> int check_correctness_evp_gen_numeric_residuals_real_single_f(int na, int nev, int na_rows, int na_cols,
    !c>                                         float *as, float *z, float *ev, int sc_desc[9],
    !c>                                         int nblk, int myid, int np_rows, int np_cols, int my_prow, int my_pcol, 
    !c>                                         float *bs);
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> int check_correctness_evp_gen_numeric_residuals_complex_double_f(int na, int nev, int na_rows, int na_cols,
    !c>                                         complex double *as, complex double *z, double *ev, int sc_desc[9],
    !c>                                         int nblk, int myid, int np_rows, int np_cols, int my_prow, int my_pcol,
    !c>                                         complex double *bs);
#else
    !c> int check_correctness_evp_gen_numeric_residuals_complex_single_f(int na, int nev, int na_rows, int na_cols,
    !c>                                         complex float *as, complex float *z, float *ev, int sc_desc[9],
    !c>                                         int nblk, int myid, int np_rows, int np_cols, int my_prow, int my_pcol,
    !c>                                         complex float *bs);
#endif
#endif /* COMPLEXCASE */

function check_correctness_evp_gen_numeric_residuals_&
&MATH_DATATYPE&
&_&
&PRECISION&
&_f (na, nev, na_rows, na_cols, as, z, ev, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol, bs) result(status) &
      bind(C,name="check_correctness_evp_gen_numeric_residuals_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      &_f")

      use iso_c_binding

      implicit none
#include "../../src/general/precision_kinds.F90"

      integer(kind=c_int)            :: status
      integer(kind=c_int), value     :: na, nev, myid, na_rows, na_cols, nblk, np_rows, np_cols, my_prow, my_pcol
      MATH_DATATYPE(kind=rck)     :: as(1:na_rows,1:na_cols), z(1:na_rows,1:na_cols), bs(1:na_rows,1:na_cols)
      real(kind=rck)    :: ev(1:na)
      integer(kind=c_int)            :: sc_desc(1:9)

      status = check_correctness_evp_numeric_residuals_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      & (na, nev, as, z, ev, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol, bs)

    end function

    !-----------------------------------------------------------------------------------------------------------

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
       ev_analytic(ii) = diagonalElement + 2.0_rk * &
                         subdiagonalElement *cos( pi*real(ii,kind=rk)/ &
                         real(na+1,kind=rk) )
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
      use precision
      implicit none
#include "../../src/general/precision_kinds.F90"
      integer(kind=ik)                 :: status
      integer(kind=ik), intent(in)     :: na, myid, na_rows

      MATH_DATATYPE(kind=rck), intent(in)       :: a(:,:), as(:,:)
      MATH_DATATYPE(kind=rck), dimension(size(as,dim=1),size(as,dim=2)) :: tmp1, tmp2
      real(kind=rk)                   :: norm, normmax

#ifdef WITH_MPI
      real(kind=rck)                   :: p&
                                           &BLAS_CHAR&
                                           &lange
#else /* WITH_MPI */
      real(kind=rck)                   :: BLAS_CHAR&
                                          &lange
#endif /* WITH_MPI */

      integer(kind=ik)                 :: sc_desc(:)
      real(kind=rck)                   :: err, errmax
      integer :: mpierr

      status = 0
      tmp1(:,:) = 0.0_rck


#if REALCASE == 1
      ! tmp1 = a**T
#ifdef WITH_MPI
      call p&
          &BLAS_CHAR&
          &tran(na, na, 1.0_rck, a, 1, 1, sc_desc, 0.0_rck, tmp1, 1, 1, sc_desc)
#else /* WITH_MPI */
      tmp1 = transpose(a)
#endif /* WITH_MPI */
#endif /* REALCASE == 1 */

#if COMPLEXCASE == 1
      ! tmp1 = a**H
#ifdef WITH_MPI
      call p&
            &BLAS_CHAR&
            &tranc(na, na, ONE, a, 1, 1, sc_desc, ZERO, tmp1, 1, 1, sc_desc)
#else /* WITH_MPI */
      tmp1 = transpose(conjg(a))
#endif /* WITH_MPI */
#endif /* COMPLEXCASE == 1 */

      ! tmp2 = a**T * a
#ifdef WITH_MPI
      call p&
            &BLAS_CHAR&
            &gemm("N","N", na, na, na, ONE, tmp1, 1, 1, sc_desc, a, 1, 1, &
               sc_desc, ZERO, tmp2, 1, 1, sc_desc)
#else /* WITH_MPI */
      call BLAS_CHAR&
                    &gemm("N","N", na, na, na, ONE, tmp1, na, a, na, ZERO, tmp2, na)
#endif /* WITH_MPI */

      ! compare tmp2 with original matrix
      tmp2(:,:) = tmp2(:,:) - as(:,:)

#ifdef WITH_MPI
      norm = p&
              &BLAS_CHAR&
              &lange("M",na, na, tmp2, 1, 1, sc_desc, tmp1)
#else /* WITH_MPI */
      norm = BLAS_CHAR&
             &lange("M", na, na, tmp2, na_rows, tmp1)
#endif /* WITH_MPI */


#ifdef WITH_MPI
      call mpi_allreduce(norm,normmax,1,MPI_REAL_PRECISION,MPI_MAX,MPI_COMM_WORLD,mpierr)
#else /* WITH_MPI */
      normmax = norm
#endif /* WITH_MPI */

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
      use precision
      implicit none
#include "../../src/general/precision_kinds.F90"
      integer(kind=ik)                 :: status
      integer(kind=ik), intent(in)     :: na, myid, na_rows
      MATH_DATATYPE(kind=rck), intent(in)       :: a(:,:), b(:,:), c(:,:)
      MATH_DATATYPE(kind=rck), dimension(size(a,dim=1),size(a,dim=2)) :: tmp1, tmp2
      real(kind=rck)                   :: norm, normmax

#ifdef WITH_MPI
      real(kind=rck)                   :: p&
                                           &BLAS_CHAR&
                                           &lange
#else /* WITH_MPI */
      real(kind=rck)                   :: BLAS_CHAR&
                                          &lange
#endif /* WITH_MPI */

      integer(kind=ik)                 :: sc_desc(:)
      real(kind=rck)                   :: err, errmax
      integer :: mpierr

      status = 0
      tmp1(:,:) = ZERO

#if REALCASE == 1
      ! tmp1 = a**T
#ifdef WITH_MPI
      call p&
            &BLAS_CHAR&
            &tran(na, na, ONE, a, 1, 1, sc_desc, ZERO, tmp1, 1, 1, sc_desc)
#else /* WITH_MPI */
      tmp1 = transpose(a)
#endif /* WITH_MPI */

#endif /* REALCASE == 1 */

#if COMPLEXCASE == 1
      ! tmp1 = a**H
#ifdef WITH_MPI
      call p&
            &BLAS_CHAR&
            &tranc(na, na, ONE, a, 1, 1, sc_desc, ZERO, tmp1, 1, 1, sc_desc)
#else /* WITH_MPI */
      tmp1 = transpose(conjg(a))
#endif /* WITH_MPI */
#endif /* COMPLEXCASE == 1 */

   ! tmp2 = tmp1 * b
#ifdef WITH_MPI
   call p&
         &BLAS_CHAR&
         &gemm("N","N", na, na, na, ONE, tmp1, 1, 1, sc_desc, b, 1, 1, &
               sc_desc, ZERO, tmp2, 1, 1, sc_desc)
#else
   call BLAS_CHAR&
        &gemm("N","N", na, na, na, ONE, tmp1, na, b, na, ZERO, tmp2, na)
#endif

      ! compare tmp2 with c
      tmp2(:,:) = tmp2(:,:) - c(:,:)

#ifdef WITH_MPI
      norm = p&
              &BLAS_CHAR&
              &lange("M",na, na, tmp2, 1, 1, sc_desc, tmp1)
#else /* WITH_MPI */
      norm = BLAS_CHAR&
             &lange("M", na, na, tmp2, na_rows, tmp1)
#endif /* WITH_MPI */

#ifdef WITH_MPI
      call mpi_allreduce(norm,normmax,1,MPI_REAL_PRECISION,MPI_MAX,MPI_COMM_WORLD,mpierr)
#else /* WITH_MPI */
      normmax = norm
#endif /* WITH_MPI */

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
