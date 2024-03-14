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


#include "config-f90.h"

#ifdef HAVE_64BIT_INTEGER_MATH_SUPPORT
#define TEST_INT_TYPE integer(kind=c_int64_t)
#define INT_TYPE lik
#define TEST_C_INT_TYPE_PTR long int*
#define TEST_C_INT_TYPE long int
#else
#define TEST_INT_TYPE integer(kind=c_int32_t)
#define INT_TYPE ik
#define TEST_C_INT_TYPE_PTR int*
#define TEST_C_INT_TYPE int
#endif

#ifdef HAVE_64BIT_INTEGER_MPI_SUPPORT
#define TEST_INT_MPI_TYPE integer(kind=c_int64_t)
#define INT_MPI_TYPE lik
#define TEST_C_INT_MPI_TYPE_PTR long int*
#define TEST_C_INT_MPI_TYPE long int
#else
#define TEST_INT_MPI_TYPE integer(kind=c_int32_t)
#define INT_MPI_TYPE ik
#define TEST_C_INT_MPI_TYPE_PTR int*
#define TEST_C_INT_MPI_TYPE int
#endif

#if REALCASE == 1
    function check_correctness_evp_numeric_residuals_ss_real_&
    &PRECISION&
    & (na, nev, as, z, ev, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol) result(status)
      use tests_blas_interfaces
      use tests_scalapack_interfaces
      use precision_for_tests
      use iso_c_binding
      use test_util
      implicit none
#include "../../src/general/precision_kinds.F90"
      integer(kind=BLAS_KIND)             :: status, na_cols, na_rows
      integer(kind=BLAS_KIND), intent(in) :: na, nev, nblk, myid, np_rows, np_cols, my_prow, my_pcol
      real(kind=rk), intent(in)           :: as(:,:)
      real(kind=rk)                       :: tmpr
      complex(kind=rck), intent(in)       :: z(:,:)
      real(kind=rk)                       :: ev(:)
      complex(kind=rck), dimension(size(as,dim=1),size(as,dim=2)) :: tmp1, tmp2
      complex(kind=rck)                   :: xc
      
      complex(kind=rck), allocatable      :: as_complex(:,:)

      integer(kind=BLAS_KIND)             :: sc_desc(:)

      integer(kind=BLAS_KIND)             :: i, j, rowLocal, colLocal
      integer(kind=c_int)                 :: row_Local, col_Local
      real(kind=rck)                      :: err, errmax

      integer :: mpierr

      ! tolerance for the residual test for different math type/precision setups
      real(kind=rk), parameter       :: tol_res_real_double      = 9e-10_rk
      real(kind=rk), parameter       :: tol_res_real_single      = 3e-2_rk
      real(kind=rk), parameter       :: tol_res_complex_double   = 9e-10_rk
      real(kind=rk), parameter       :: tol_res_complex_single   = 3e-2_rk
      real(kind=rk)                  :: tol_res                  = tol_res_&
                                                                          &MATH_DATATYPE&
                                                                          &_&
                                                                          &PRECISION
      ! precision of generalized problem is lower
      real(kind=rk), parameter       :: generalized_penalty = 10.0_rk

      ! tolerance for the orthogonality test for different math type/precision setups
!       real(kind=rk), parameter       :: tol_orth_real_double     = 5e-11_rk
      real(kind=rk), parameter       :: tol_orth_real_double     = 9e-10_rk
      real(kind=rk), parameter       :: tol_orth_real_single     = 9e-2_rk
      real(kind=rk), parameter       :: tol_orth_complex_double  = 9e-10_rk
      real(kind=rk), parameter       :: tol_orth_complex_single  = 9e-3_rk
      real(kind=rk), parameter       :: tol_orth                 = tol_orth_&
                                                                          &MATH_DATATYPE&
                                                                          &_&
                                                                          &PRECISION
                                                  
      complex(kind=rck), parameter   :: CZERO = (0.0_rck,0.0_rck), CONE = (1.0_rck,0.0_rck)


      status = 0
      ! Setup complex matrices and eigenvalues
      na_rows = size(as,dim=1)
      na_cols = size(as,dim=2)
      
      allocate(as_complex(na_rows,na_cols))
      do j=1, na_cols
        do i=1,na_rows
#ifdef DOUBLE_PRECISION_REAL
          as_complex(i,j) = dcmplx(as(i,j),0.0_rk)
#else
          as_complex(i,j) = cmplx(as(i,j),0.0_rk)
#endif
       enddo
      enddo
      
      ! 1. Residual (maximum of || A*Zi - Zi*EVi ||)

      ! tmp1 = Zi*EVi
      tmp1(:,:) = z(:,:)
      do i=1,nev
#ifdef DOUBLE_PRECISION_REAL
        xc = dcmplx(0.0_rk,ev(i))
#else
        xc = cmplx(0.0_rk,ev(i))
#endif
#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_REAL
        call pzscal(int(na,kind=BLAS_KIND), xc, tmp1, 1_BLAS_KIND, int(i,kind=BLAS_KIND), sc_desc, 1_BLAS_KIND)
#else
        call pcscal(int(na,kind=BLAS_KIND), xc, tmp1, 1_BLAS_KIND, int(i,kind=BLAS_KIND), sc_desc, 1_BLAS_KIND)
#endif
#else /* WITH_MPI */
#ifdef DOUBLE_PRECISION_REAL
        call zscal(int(na,kind=BLAS_KIND), xc, tmp1(:,i), 1_BLAS_KIND)
#else
        call cscal(int(na,kind=BLAS_KIND), xc, tmp1(:,i), 1_BLAS_KIND)
#endif
#endif /* WITH_MPI */
      enddo

      ! normal eigenvalue problem .. no need to multiply
        tmp2(:,:) = tmp1(:,:)

      ! tmp1 =  A * Z
      ! as is original stored matrix, Z are the EVs
#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_REAL
      call PZGEMM('N', 'N', int(na,kind=BLAS_KIND), int(nev,kind=BLAS_KIND), int(na,kind=BLAS_KIND), &
                  CONE, as_complex, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, &
                  z, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, CZERO, tmp1, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc)
#else
      call PCGEMM('N', 'N', int(na,kind=BLAS_KIND), int(nev,kind=BLAS_KIND), int(na,kind=BLAS_KIND), &
                  CONE, as_complex, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, &
                  z, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, CZERO, tmp1, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc)
#endif
#else /* WITH_MPI */
#ifdef DOUBLE_PRECISION_REAL
      call ZGEMM('N','N',int(na,kind=BLAS_KIND), int(nev,kind=BLAS_KIND), int(na,kind=BLAS_KIND), CONE, &
                 as_complex, int(na,kind=BLAS_KIND), z,int(na,kind=BLAS_KIND), CZERO, tmp1, int(na,kind=BLAS_KIND) )
#else
      call CGEMM('N','N', int(na,kind=BLAS_KIND), int(nev,kind=BLAS_KIND), int(na,kind=BLAS_KIND), CONE, &
                  as_complex, int(na,kind=BLAS_KIND), z, int(na,kind=BLAS_KIND), CZERO, tmp1, int(na,kind=BLAS_KIND) )
#endif
#endif /* WITH_MPI */

      !  tmp1 = A*Zi - Zi*EVi
      tmp1(:,:) =  tmp1(:,:) - tmp2(:,:)
      
      ! Get maximum norm of columns of tmp1
      errmax = 0.0_rk

      do i=1,nev
        xc = (0.0_rk,0.0_rk)
#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_REAL
        call PZDOTC(int(na,kind=BLAS_KIND), xc, tmp1, 1_BLAS_KIND, int(i,kind=BLAS_KIND), sc_desc, &
                    1_BLAS_KIND, tmp1, 1_BLAS_KIND, int(i,kind=BLAS_KIND), sc_desc, 1_BLAS_KIND)
#else
        call PCDOTC(int(na,kind=BLAS_KIND), xc, tmp1, 1_BLAS_KIND, int(i,kind=BLAS_KIND), sc_desc, &
                    1_BLAS_KIND, tmp1, 1_BLAS_KIND, int(i,kind=BLAS_KIND), sc_desc, 1_BLAS_KIND)
#endif
#else /* WITH_MPI */
#ifdef DOUBLE_PRECISION_REAL
        xc = ZDOTC(int(na,kind=BLAS_KIND) ,tmp1, 1_BLAS_KIND, tmp1, 1_BLAS_KIND)
#else
        xc = CDOTC(int(na,kind=BLAS_KIND) ,tmp1, 1_BLAS_KIND, tmp1, 1_BLAS_KIND)
#endif
#endif /* WITH_MPI */
        errmax = max(errmax, sqrt(real(xc,kind=REAL_DATATYPE)))
      enddo

      ! Get maximum error norm over all processors
      err = errmax
#ifdef WITH_MPI
      call mpi_allreduce(err, errmax, 1_MPI_KIND, MPI_REAL_PRECISION, MPI_MAX, int(MPI_COMM_WORLD,kind=MPI_KIND), mpierr)
#else /* WITH_MPI */
      errmax = err
#endif /* WITH_MPI */
      if (myid==0) print *,'%Results of numerical residual checks, using complex arithmetic:'
      if (myid==0) print *,'%Error Residual     :',errmax
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
        tmp2(:,:) = z(:,:)
      tmp1 = 0
#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_REAL
      call PZGEMM('C', 'N', int(nev,kind=BLAS_KIND), int(nev,kind=BLAS_KIND), int(na,kind=BLAS_KIND), &
                  CONE, z, 1_BLAS_KIND, 1_BLAS_KIND, &
                  sc_desc, tmp2, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, CZERO, tmp1, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc)
#else
      call PCGEMM('C', 'N', int(nev,kind=BLAS_KIND), int(nev,kind=BLAS_KIND), int(na,kind=BLAS_KIND), &
                  CONE, z, 1_BLAS_KIND, 1_BLAS_KIND, &
                  sc_desc, tmp2, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, CZERO, tmp1, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc)
#endif

#else /* WITH_MPI */
#ifdef DOUBLE_PRECISION_REAL
      call ZGEMM('C','N', int(nev,kind=BLAS_KIND) , int(nev,kind=BLAS_KIND), int(na,kind=BLAS_KIND),CONE, z, &
                 int(na,kind=BLAS_KIND), tmp2, int(na,kind=BLAS_KIND), CZERO, tmp1, int(na,kind=BLAS_KIND))
#else
      call CGEMM('C','N', int(nev,kind=BLAS_KIND) , int(nev,kind=BLAS_KIND), int(na,kind=BLAS_KIND),CONE, z, &
                 int(na,kind=BLAS_KIND), tmp2, int(na,kind=BLAS_KIND), CZERO, tmp1, int(na,kind=BLAS_KIND))
#endif
#endif /* WITH_MPI */
      ! First check, whether the elements on diagonal are 1 .. "normality" of the vectors
      err = 0.0_rk
      do i=1, nev
        if (map_global_array_index_to_local_index(int(i,kind=c_int), int(i,kind=c_int), row_Local, col_Local, &
                                                  int(nblk,kind=c_int), int(np_rows,kind=c_int), int(np_cols,kind=c_int), &
                                                  int(my_prow,kind=c_int), int(my_pcol,kind=c_int)) ) then
           rowLocal = int(row_Local,kind=INT_TYPE)
           colLocal = int(col_Local,kind=INT_TYPE)
           err = max(err, abs(tmp1(rowLocal,colLocal) - CONE))
         endif
      end do
#ifdef WITH_MPI
      call mpi_allreduce(err, errmax, 1_MPI_KIND, MPI_REAL_PRECISION, MPI_MAX, int(MPI_COMM_WORLD,kind=MPI_KIND), mpierr)
#else /* WITH_MPI */
      errmax = err
#endif /* WITH_MPI */
      if (myid==0) print *,'%Maximal error in eigenvector lengths:',errmax

      ! Second, find the maximal error in the whole Z**T * Z matrix (its diference from identity matrix)
      ! Initialize tmp2 to unit matrix
      tmp2 = 0
#ifdef WITH_MPI
#ifdef DOUBLE_PRECISION_REAL
      call PZLASET('A', int(nev,kind=BLAS_KIND), int(nev,kind=BLAS_KIND), CZERO, CONE, tmp2, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc)
#else
      call PCLASET('A', int(nev,kind=BLAS_KIND), int(nev,kind=BLAS_KIND), CZERO, CONE, tmp2, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc)
#endif
#else /* WITH_MPI */
#ifdef DOUBLE_PRECISION_REAL
      call ZLASET('A',int(nev,kind=BLAS_KIND) ,int(nev,kind=BLAS_KIND) ,CZERO, CONE, tmp2, int(na,kind=BLAS_KIND))
#else
      call CLASET('A',int(nev,kind=BLAS_KIND) ,int(nev,kind=BLAS_KIND) ,CZERO, CONE, tmp2, int(na,kind=BLAS_KIND))
#endif
#endif /* WITH_MPI */

      !      ! tmp1 = Z**T * Z - Unit Matrix
      tmp1(:,:) =  tmp1(:,:) - tmp2(:,:)

      ! Get maximum error (max abs value in tmp1)
      err = maxval(abs(tmp1))
#ifdef WITH_MPI
      call mpi_allreduce(err, errmax, 1_MPI_KIND, MPI_REAL_PRECISION, MPI_MAX, int(MPI_COMM_WORLD,kind=MPI_KIND), mpierr)
#else /* WITH_MPI */
      errmax = err
#endif /* WITH_MPI */
      if (myid==0) print *,'%Error Orthogonality:',errmax
      
      if (is_infinity_or_NaN(errmax)) then
        status = 1
      endif

      if (nev .ge. 2) then
        if (errmax .gt. tol_orth .or. errmax .eq. 0.0_rk) then
          status = 1
        endif
      else
        if (errmax .gt. tol_orth) then
          status = 1
        endif
      endif
      
      deallocate(as_complex)
    end function

#endif /* REALCASE */

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
    !c> TEST_C_INT_TYPE check_correctness_evp_numeric_residuals_ss_real_double_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE nev, TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                         double *as, double_complex *z, double *ev,  TEST_C_INT_TYPE sc_desc[9],
    !c>                                         TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid, TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols, TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#else
    !c> TEST_C_INT_TYPE check_correctness_evp_numeric_residuals_ss_real_single_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE nev, TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                         float *as, float_complex *z, float *ev, TEST_C_INT_TYPE sc_desc[9],
    !c>                                         TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid, TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols, TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#endif
#endif /* REALCASE */

#if REALCASE == 1
function check_correctness_evp_numeric_residuals_ss_real_&
&PRECISION&
&_f (na, nev, na_rows, na_cols, as, z, ev, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol) result(status) &
      bind(C,name="check_correctness_evp_numeric_residuals_ss_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      &_f")

      use precision_for_tests
      use iso_c_binding

      implicit none
#include "./test_precision_kinds.F90"

      TEST_INT_TYPE            :: status
      TEST_INT_TYPE, value     :: na, nev, myid, na_rows, na_cols, nblk, np_rows, np_cols, my_prow, my_pcol
      real(kind=rck)            :: as(1:na_rows,1:na_cols)
      complex(kind=rck)         :: z(1:na_rows,1:na_cols)
      real(kind=rck)            :: ev(1:na)
      TEST_INT_TYPE            :: sc_desc(1:9)

      status = check_correctness_evp_numeric_residuals_ss_real_&
      &PRECISION&
      & (na, nev, as, z, ev, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol)
    end function
#endif /* REALCASE */

function check_correctness_evp_numeric_residuals_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    & (na, nev, as, z, ev, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol, bs) result(status)
 
      use tests_blas_interfaces
      use tests_scalapack_interfaces
      use precision_for_tests
      use test_util
      implicit none
#include "./test_precision_kinds.F90"
      TEST_INT_TYPE                                 :: status
      TEST_INT_TYPE, intent(in)                     :: na, nev, nblk, myid, np_rows, np_cols, my_prow, my_pcol
      MATH_DATATYPE(kind=rck), intent(in)           :: as(:,:), z(:,:)
      MATH_DATATYPE(kind=rck), intent(in), optional :: bs(:,:)
      real(kind=rk)                                 :: ev(:)
      MATH_DATATYPE(kind=rck), dimension(size(as,dim=1),size(as,dim=2)) :: tmp1, tmp2
      MATH_DATATYPE(kind=rck)                       :: xc

      TEST_INT_TYPE                 :: sc_desc(:)

      TEST_INT_TYPE                 :: i, rowLocal, colLocal
      integer(kind=c_int)           :: row_Local, col_Local
      real(kind=rck)                :: err, errmax

      TEST_INT_MPI_TYPE             :: mpierr

! tolerance for the residual test for different math type/precision setups
      real(kind=rk), parameter       :: tol_res_real_double      = 9e-10_rk
      real(kind=rk), parameter       :: tol_res_real_single      = 3e-2_rk
      real(kind=rk), parameter       :: tol_res_complex_double   = 9e-10_rk
      real(kind=rk), parameter       :: tol_res_complex_single   = 3e-2_rk
      real(kind=rk)                  :: tol_res                  = tol_res_&
                                                                          &MATH_DATATYPE&
                                                                          &_&
                                                                          &PRECISION
      ! precision of generalized problem is lower
      real(kind=rk), parameter       :: generalized_penalty = 10.0_rk

      ! tolerance for the orthogonality test for different math type/precision setups
      real(kind=rk), parameter       :: tol_orth_real_double     = 9e-10_rk
      real(kind=rk), parameter       :: tol_orth_real_single     = 9e-2_rk
      real(kind=rk), parameter       :: tol_orth_complex_double  = 9e-10_rk
      real(kind=rk), parameter       :: tol_orth_complex_single  = 9e-3_rk
      real(kind=rk), parameter       :: tol_orth                 = tol_orth_&
                                                                          &MATH_DATATYPE&
                                                                          &_&
                                                                          &PRECISION

      if (present(bs)) then
        tol_res = generalized_penalty * tol_res
      endif
      status = 0

      ! 1. Residual (maximum of || A*Zi - Zi*EVi ||)
     
!       tmp1 = Zi*EVi
      tmp1(:,:) = z(:,:)
      do i=1,nev
        xc = ev(i)
#ifdef WITH_MPI
        call p&
            &BLAS_CHAR&
            &scal(na, xc, tmp1, 1_BLAS_KIND, i, sc_desc, 1_BLAS_KIND)
#else /* WITH_MPI */
        call BLAS_CHAR&
            &scal(na, xc, tmp1(:,i), 1_BLAS_KIND)
#endif /* WITH_MPI */
      enddo

      ! for generalized EV problem, multiply by bs as well
      ! tmp2 = B * tmp1
      if(present(bs)) then
#ifdef WITH_MPI
      call scal_PRECISION_GEMM('N', 'N', na, nev, na, ONE, bs, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, &
                               tmp1, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, ZERO, tmp2, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc)
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
      call scal_PRECISION_GEMM('N', 'N', na, nev, na, ONE, as, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, &
                  z, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, ZERO, tmp1, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc)
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
        call scal_PRECISION_NRM2(na, err, tmp1, 1_BLAS_KIND, i, sc_desc, 1_BLAS_KIND)
#else /* WITH_MPI */
        err = PRECISION_NRM2(na,tmp1(1,i),1_BLAS_KIND)
#endif /* WITH_MPI */
        errmax = max(errmax, err)
#endif /* REALCASE */

#if COMPLEXCASE == 1
        xc = 0
#ifdef WITH_MPI
        call scal_PRECISION_DOTC(na, xc, tmp1, 1_BLAS_KIND, i, sc_desc, &
                                 1_BLAS_KIND, tmp1, 1_BLAS_KIND, i, sc_desc, 1_BLAS_KIND)
#else /* WITH_MPI */
        xc = PRECISION_DOTC(na,tmp1,1_BLAS_KIND,tmp1,1_BLAS_KIND)
#endif /* WITH_MPI */
        errmax = max(errmax, sqrt(real(xc,kind=REAL_DATATYPE)))
#endif /* COMPLEXCASE */
      enddo

      ! Get maximum error norm over all processors
      err = errmax
#ifdef WITH_MPI
      call mpi_allreduce(err, errmax, 1_MPI_KIND, MPI_REAL_PRECISION, MPI_MAX, MPI_COMM_WORLD, mpierr)
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
        call scal_PRECISION_GEMM('N', 'N', na, nev, na, ONE, bs, 1_BLAS_KIND, 1_BLAS_KIND, &
                        sc_desc, z, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, ZERO, tmp2, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc)
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
      call scal_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', nev, nev, na, ONE, z, 1_BLAS_KIND, 1_BLAS_KIND, &
                        sc_desc, tmp2, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, ZERO, &
                        tmp1, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc)
#else /* WITH_MPI */
      call PRECISION_GEMM(BLAS_TRANS_OR_CONJ,'N',nev,nev,na,ONE,z,na,tmp2,na,ZERO,tmp1,na)
#endif /* WITH_MPI */
      ! First check, whether the elements on diagonal are 1 .. "normality" of the vectors
      err = 0.0_rk
      do i=1, nev
        if (map_global_array_index_to_local_index(int(i,kind=c_int), int(i,kind=c_int) , row_Local, col_Local, &
                                                  int(nblk,kind=c_int), int(np_rows,kind=c_int), &
                                                  int(np_cols,kind=c_int), int(my_prow,kind=c_int), &
                                                  int(my_pcol,kind=c_int) )) then
           rowLocal = int(row_Local,kind=INT_TYPE)
           colLocal = int(col_Local,kind=INT_TYPE)
           err = max(err, abs(tmp1(rowLocal,colLocal) - 1.0_rk))
         endif
      end do
#ifdef WITH_MPI
      call mpi_allreduce(err, errmax, 1_MPI_KIND, MPI_REAL_PRECISION, MPI_MAX, MPI_COMM_WORLD, mpierr)
#else /* WITH_MPI */
      errmax = err
#endif /* WITH_MPI */
      if (myid==0) print *,'Maximal error in eigenvector lengths:',errmax

      ! Second, find the maximal error in the whole Z**T * Z matrix (its diference from identity matrix)
      ! Initialize tmp2 to unit matrix
      tmp2 = 0
#ifdef WITH_MPI
      call scal_PRECISION_LASET('A', nev, nev, ZERO, ONE, tmp2, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc)
#else /* WITH_MPI */
      call PRECISION_LASET('A',nev,nev,ZERO,ONE,tmp2,na)
#endif /* WITH_MPI */

      !      ! tmp1 = Z**T * Z - Unit Matrix
      tmp1(:,:) =  tmp1(:,:) - tmp2(:,:)

      ! Get maximum error (max abs value in tmp1)
      err = maxval(abs(tmp1))
#ifdef WITH_MPI
      call mpi_allreduce(err, errmax, 1_MPI_KIND, MPI_REAL_PRECISION, MPI_MAX, MPI_COMM_WORLD, mpierr)
#else /* WITH_MPI */
      errmax = err
#endif /* WITH_MPI */
      if (myid==0) print *,'Error Orthogonality:',errmax

      if (is_infinity_or_NaN(errmax)) then
        status = 1
      endif

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
    !c> TEST_C_INT_TYPE check_correctness_evp_numeric_residuals_real_double_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE nev,
    !c>                                                                       TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                                                       double *as, double *z, double *ev, 
    !c>                                                                       TEST_C_INT_TYPE sc_desc[9],
    !c>                                                                       TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid, 
    !c>                                                                       TEST_C_INT_TYPE np_rows, 
    !c>                                                                       TEST_C_INT_TYPE np_cols, 
    !c>                                                                       TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#else
    !c> TEST_C_INT_TYPE check_correctness_evp_numeric_residuals_real_single_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE nev, 
    !c>                                                                       TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                                                       float *as, float *z, float *ev, 
    !c>                                                                       TEST_C_INT_TYPE sc_desc[9],
    !c>                                                                       TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid, 
    !c>                                                                       TEST_C_INT_TYPE np_rows, 
    !c>                                                                       TEST_C_INT_TYPE np_cols, 
    !c>                                                                       TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> TEST_C_INT_TYPE check_correctness_evp_numeric_residuals_complex_double_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE nev, 
    !c>                                                              TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                                              double_complex *as, double_complex *z, double *ev, 
    !c>                                                              TEST_C_INT_TYPE sc_desc[9],
    !c>                                                              TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid, 
    !c>                                                              TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols, 
    !c>                                                              TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#else
    !c> TEST_C_INT_TYPE check_correctness_evp_numeric_residuals_complex_single_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE nev, 
    !c>                                                                  TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                                                  float_complex *as, float_complex *z, float *ev, 
    !c>                                                                  TEST_C_INT_TYPE sc_desc[9],
    !c>                                                                  TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid, 
    !c>                                                                  TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols, 
    !c>                                                                  TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#endif
#endif /* COMPLEXCASE */

! extra na_rows, na_cols parameters are needed for C-interface
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

      use precision_for_tests
      use iso_c_binding

      implicit none
#include "./test_precision_kinds.F90"

      TEST_INT_TYPE            :: status
      TEST_INT_TYPE, value     :: na, nev, myid, na_rows, na_cols, nblk, np_rows, np_cols, my_prow, my_pcol
      MATH_DATATYPE(kind=rck)  :: as(1:na_rows,1:na_cols), z(1:na_rows,1:na_cols)
      real(kind=rck)           :: ev(1:na)
      TEST_INT_TYPE            :: sc_desc(1:9)

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
    !c> TEST_C_INT_TYPE check_correctness_evp_gen_numeric_residuals_real_double_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE nev, 
    !c>                                                               TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                                               double *as, double *z, double *ev,
    !c>                                                               TEST_C_INT_TYPE sc_desc[9],
    !c>                                                               TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid, 
    !c>                                                               TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols, 
    !c>                                                               TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol,
    !c>                                                               double *bs);
#else
    !c> TEST_C_INT_TYPE check_correctness_evp_gen_numeric_residuals_real_single_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE nev, 
    !c>                                                                           TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                                                           float *as, float *z, float *ev, 
    !c>                                                                           TEST_C_INT_TYPE sc_desc[9],
    !c>                                                                           TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid,
    !c>                                                                           TEST_C_INT_TYPE np_rows, 
    !c>                                                                           TEST_C_INT_TYPE np_cols, 
    !c>                                                                           TEST_C_INT_TYPE my_prow, 
    !c>                                                                           TEST_C_INT_TYPE my_pcol, 
    !c>                                                                           float *bs);
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> TEST_C_INT_TYPE check_correctness_evp_gen_numeric_residuals_complex_double_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE nev,
    !c>                                                                    TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                                                    double_complex *as, double_complex *z, double *ev,
    !c>                                                                    TEST_C_INT_TYPE sc_desc[9],
    !c>                                                                    TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid, 
    !c>                                                                    TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                                                    TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol,
    !c>                                                                    double_complex *bs);
#else
    !c> TEST_C_INT_TYPE check_correctness_evp_gen_numeric_residuals_complex_single_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE nev,
    !c>                                                                    TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                                                    float_complex *as, float_complex *z, float *ev, 
    !c>                                                                    TEST_C_INT_TYPE sc_desc[9],
    !c>                                                                    TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid, 
    !c>                                                                    TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols, 
    !c>                                                                    TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol,
    !c>                                                                    float_complex *bs);
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
      use precision_for_tests
      implicit none
#include "./test_precision_kinds.F90"

      TEST_INT_TYPE            :: status
      TEST_INT_TYPE, value     :: na, nev, myid, na_rows, na_cols, nblk, np_rows, np_cols, my_prow, my_pcol
      MATH_DATATYPE(kind=rck)  :: as(1:na_rows,1:na_cols), z(1:na_rows,1:na_cols), bs(1:na_rows,1:na_cols)
      real(kind=rck)           :: ev(1:na)
      TEST_INT_TYPE            :: sc_desc(1:9)

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
      use precision_for_tests
      implicit none
#include "./test_precision_kinds.F90"

      TEST_INT_TYPE               :: status, ii, j, myid
      TEST_INT_TYPE, intent(in)   :: na
      real(kind=rk) :: diagonalElement, subdiagonalElement
      real(kind=rk) :: ev_analytic(na), ev(na)
      MATH_DATATYPE(kind=rck) :: z(:,:) ! needed only for correct expansion of 4 cases: double/single, real/complex

#if defined(DOUBLE_PRECISION_REAL) || defined(DOUBLE_PRECISION_COMPLEX)
      real(kind=rck), parameter   :: pi = 3.141592653589793238462643383279_c_double
#else
      real(kind=rck), parameter   :: pi = 3.1415926535897932_c_float
#endif
      real(kind=rck)              :: tmp, maxerr
      TEST_INT_TYPE               :: loctmp
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

     if (is_infinity_or_NaN(maxerr)) then
       status = 1
     endif

#if defined(DOUBLE_PRECISION_REAL) || defined(DOUBLE_PRECISION_COMPLEX)
     if (abs(maxerr) .gt. 9.e-10_c_double) then
#else
     if (abs(maxerr) .gt. 8.e-4_c_float) then
#endif
       status = 1
       if (myid .eq. 0) then
         print *,"Result of Toeplitz matrix test: "
         print *,"Eigenvalues differ from analytic solution: maxerr = ",abs(maxerr)
       endif
     endif

    if (status .eq. 0) then
       if (myid .eq. 0) then
         print *,"Result of Toeplitz matrix test: test passed"
         print *,"Eigenvalues differ from analytic solution: maxerr = ",abs(maxerr)
       endif
    endif
    end function


#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
    !c> TEST_C_INT_TYPE check_correctness_eigenvalues_toeplitz_real_double_f(TEST_C_INT_TYPE na, 
    !c>     TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols, 
    !c>     double diagonalElement, double subdiagonalElement,
    !c>     double *ev, 
    !c>     double *z, TEST_C_INT_TYPE myid);
#else
    !c> TEST_C_INT_TYPE check_correctness_eigenvalues_toeplitz_real_single_f(TEST_C_INT_TYPE na, 
    !c>     TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols, 
    !c>     float diagonalElement, float subdiagonalElement,
    !c>     float *ev, 
    !c>     float *z, TEST_C_INT_TYPE myid);
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> TEST_C_INT_TYPE check_correctness_eigenvalues_toeplitz_complex_double_f(TEST_C_INT_TYPE na, 
    !c>     TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols, 
    !c>     double diagonalElement, double subdiagonalElement,
    !c>     double *ev, 
    !c>     double_complex *z, TEST_C_INT_TYPE myid);
#else
    !c> TEST_C_INT_TYPE check_correctness_eigenvalues_toeplitz_complex_single_f(TEST_C_INT_TYPE na, 
    !c>     TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols, 
    !c>     float diagonalElement, float subdiagonalElement,
    !c>     float *ev, 
    !c>     float_complex *z, TEST_C_INT_TYPE myid);
#endif
#endif /* COMPLEXCASE */

! extra na_rows, na_cols parameters are needed for C-interface
   function check_correctness_eigenvalues_toeplitz_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &_f (na, na_rows, na_cols, diagonalElement, subdiagonalElement, ev, z, myid) result(status) &
      bind(C,name="check_correctness_eigenvalues_toeplitz_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      &_f")

      use iso_c_binding
      use precision_for_tests
      implicit none
#include "./test_precision_kinds.F90"

      TEST_INT_TYPE            :: status
      TEST_INT_TYPE, value     :: na, na_rows, na_cols, myid
      real(kind=rk), value     :: diagonalElement, subdiagonalElement
      real(kind=rk)            :: ev(1:na)
      MATH_DATATYPE(kind=rck)  :: z(1:na_rows,1:na_cols) ! needed only for correct expansion of 4 cases: double/single, real/complex
      
      status = check_correctness_eigenvalues_toeplitz_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      & (na, diagonalElement, subdiagonalElement, ev, z, myid)

    end function  
    !-----------------------------------------------------------------------------------------------------------


    function check_correctness_cholesky_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    & (na, a, as, na_rows, sc_desc, myid) result(status)
      use precision_for_tests
      use tests_blas_interfaces
      use tests_scalapack_interfaces
      use test_util
      implicit none
#include "./test_precision_kinds.F90"
      TEST_INT_TYPE                                                     :: status
      TEST_INT_TYPE, intent(in)                                         :: na, myid, na_rows

      MATH_DATATYPE(kind=rck), intent(in)                               :: a(:,:), as(:,:)
      MATH_DATATYPE(kind=rck), dimension(size(as,dim=1),size(as,dim=2)) :: tmp1, tmp2
#if COMPLEXCASE == 1
      ! needed for [z,c]lange from scalapack
      real(kind=rk), dimension(2*size(as,dim=1),size(as,dim=2))         :: tmp1_real
#endif
      real(kind=rk)                                                     :: norm, normmax

      TEST_INT_TYPE                                                     :: sc_desc(:)
      real(kind=rck)                                                    :: err, errmax
      TEST_INT_MPI_TYPE                                                 :: mpierr

      status = 0
      tmp1(:,:) = 0.0_rck
 

#if REALCASE == 1
      ! tmp1 = a**T
#ifdef WITH_MPI
      call p&
          &BLAS_CHAR&
          &tran(na, na, 1.0_rck, a, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, &
                0.0_rck, tmp1, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc)
#else /* WITH_MPI */
      tmp1 = transpose(a)
#endif /* WITH_MPI */
#endif /* REALCASE == 1 */

#if COMPLEXCASE == 1
      ! tmp1 = a**H
#ifdef WITH_MPI
      call p&
            &BLAS_CHAR&
            &tranc(na, na, ONE, a, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, &
                   ZERO, tmp1, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc)
#else /* WITH_MPI */
      tmp1 = transpose(conjg(a))
#endif /* WITH_MPI */
#endif /* COMPLEXCASE == 1 */

      ! tmp2 = a**T * a
#ifdef WITH_MPI
      call p&
            &BLAS_CHAR&
            &gemm("N","N", na, na, na, ONE, tmp1, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, &
                  a, 1_BLAS_KIND, 1_BLAS_KIND, &
                  sc_desc, ZERO, tmp2, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc)
#else /* WITH_MPI */
      call BLAS_CHAR&
                    &gemm("N","N", na, na, na, ONE, tmp1, na, a, na, ZERO, tmp2, na)
#endif /* WITH_MPI */

      ! compare tmp2 with original matrix
      tmp2(:,:) = tmp2(:,:) - as(:,:)

#ifdef WITH_MPI
      norm = p&
              &BLAS_CHAR&
              &lange("M",na, na, tmp2, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, &
#if COMPLEXCASE == 1
              tmp1_real)
#else
              tmp1)
#endif
#else /* WITH_MPI */
      norm = BLAS_CHAR&
             &lange("M", na, na, tmp2, na_rows, &
#if COMPLEXCASE == 1
             tmp1_real)
#else
             tmp1)
#endif
#endif /* WITH_MPI */


#ifdef WITH_MPI
      call mpi_allreduce(norm, normmax, 1_MPI_KIND, MPI_REAL_PRECISION, MPI_MAX, MPI_COMM_WORLD, mpierr)
#else /* WITH_MPI */
      normmax = norm
#endif /* WITH_MPI */

      if (myid .eq. 0) then
        print *," Maximum error of result: ", normmax
      endif

      if (is_infinity_or_NaN(normmax)) then
        status = 1
      endif

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
!      if (normmax .gt. 5e-12_rk8 .or. normmax .eq. 0.0_rk8) then
      if (normmax .gt. 9e-10_rk8) then
        status = 1
      endif
#else
!      if (normmax .gt. 5e-4_rk4 .or. normmax .eq. 0.0_rk4) then
      if (normmax .gt. 9e-2_rk4 ) then
        status = 1
      endif
#endif
#endif

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
!      if (normmax .gt. 5e-11_rk8 .or. normmax .eq. 0.0_rk8) then
      if (normmax .gt. 9e-10_rk8 ) then
        status = 1
      endif
#else
!      if (normmax .gt. 5e-3_rk4 .or. normmax .eq. 0.0_rk4) then
      if (normmax .gt. 9e-2_rk4) then
        status = 1
      endif
#endif
#endif
    end function

   ! cholesky C-interface
   ! additional parameter na_cols is needed on top of Fortran interface
   
#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
    !c> TEST_C_INT_TYPE check_correctness_cholesky_real_double_f(TEST_C_INT_TYPE na, 
    !c>                                                        double *a, double *as,
    !c>                                                        TEST_C_INT_TYPE na_rows,
    !c>                                                        TEST_C_INT_TYPE na_cols,
    !c>                                                        TEST_C_INT_TYPE sc_desc[9],
    !c>                                                        TEST_C_INT_TYPE myid);
#else
    !c> TEST_C_INT_TYPE check_correctness_cholesky_real_single_f(TEST_C_INT_TYPE na, 
    !c>                                                        float  *a, float  *as,
    !c>                                                        TEST_C_INT_TYPE na_rows,
    !c>                                                        TEST_C_INT_TYPE na_cols,
    !c>                                                        TEST_C_INT_TYPE sc_desc[9],
    !c>                                                        TEST_C_INT_TYPE myid);
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> TEST_C_INT_TYPE check_correctness_cholesky_complex_double_f(TEST_C_INT_TYPE na, 
    !c>                                                        double_complex *a, double_complex *as,
    !c>                                                        TEST_C_INT_TYPE na_rows,
    !c>                                                        TEST_C_INT_TYPE na_cols,
    !c>                                                        TEST_C_INT_TYPE sc_desc[9],
    !c>                                                        TEST_C_INT_TYPE myid);
#else
    !c> TEST_C_INT_TYPE check_correctness_cholesky_complex_single_f(TEST_C_INT_TYPE na, 
    !c>                                                        float_complex *a, float_complex *as,
    !c>                                                        TEST_C_INT_TYPE na_rows,    
    !c>                                                        TEST_C_INT_TYPE na_cols,
    !c>                                                        TEST_C_INT_TYPE sc_desc[9],
    !c>                                                        TEST_C_INT_TYPE myid);
#endif
#endif /* COMPLEXCASE */

    function check_correctness_cholesky_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &_f (na, a, as, na_rows, na_cols, sc_desc, myid) result(status) &
      bind(C,name="check_correctness_cholesky_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      &_f")
      use iso_c_binding
      use precision_for_tests
      implicit none
#include "./test_precision_kinds.F90"
      
      TEST_INT_TYPE            :: status
      TEST_INT_TYPE, value     :: na, na_rows, na_cols, myid
      MATH_DATATYPE(kind=rck)  :: a(1:na_rows,1:na_cols), as(1:na_rows,1:na_cols)
      TEST_INT_TYPE            :: sc_desc(1:9)

      status = check_correctness_cholesky_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      & (na, a, as, na_rows, sc_desc, myid)

    end function
    
    !-----------------------------------------------------------------------------------------------------------
    ! transa_NH = 'H' - hermitian multiply  C = A**H * B
    ! transa_NH = 'N' - normal multiply     C = A    * B
	
    function check_correctness_hermitian_multiply_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    & (transa_NH, na, a, b, c, na_rows, sc_desc, myid, nblk, np_rows, np_cols, my_prow, my_pcol, &
       isUpper, isLower) result(status) 
      use precision_for_tests
      use tests_blas_interfaces
      use tests_scalapack_interfaces
      use test_util
      implicit none
#include "./test_precision_kinds.F90"
      logical, intent(in), optional                                   :: isUpper, isLower

      TEST_INT_TYPE, intent(in), optional                             :: nblk, np_rows, np_cols, &
                                                                         my_prow, my_pcol
      TEST_INT_TYPE                                                   :: status
	  character*1                                                     :: transa_NH, transa_NTC
      TEST_INT_TYPE, intent(in)                                       :: na, myid, na_rows
      MATH_DATATYPE(kind=rck)                                         :: a(:,:), c(:,:)
      MATH_DATATYPE(kind=rck), intent(in)                             :: b(:,:)
      MATH_DATATYPE(kind=rck), dimension(size(a,dim=1),size(a,dim=2)) :: tmp1, tmp2
#if COMPLEXCASE == 1
      real(kind=rk), dimension(2*size(a,dim=1),size(a,dim=2))         :: tmp1_real
#endif
      real(kind=rck)                                                  :: norm, normmax


      integer(kind=c_int)                                             :: rowLocal, colLocal
      TEST_INT_TYPE                                                   :: ii, jj
      TEST_INT_TYPE                                                   :: sc_desc(:)
      real(kind=rck)                                                  :: err, errmax
      TEST_INT_MPI_TYPE                                               :: mpierr

      status = 0
      tmp1(:,:) = ZERO

   transa_NTC = "N"
   if (transa_NH .eq. "H") then
#if REALCASE == 1   
      transa_NTC = "T"
#else
      transa_NTC = "C"
#endif
   endif
   
   if (present(isUpper)) then
     if (isUpper) then
       ! a should only be upper triangular

       ! do a dirty hack set lower half to zero
       do ii=1, na
         do jj=1, ii-1
           if (map_global_array_index_to_local_index(int(ii,kind=c_int), int(jj,kind=c_int), rowLocal, &
                                                    colLocal, int(nblk,kind=c_int), int(np_rows,kind=c_int), &
                                                    int(np_cols,kind=c_int), int(my_prow,kind=c_int), &
                                                    int(my_pcol,kind=c_int) ) ) then
             a(int(rowLocal,kind=INT_TYPE),int(colLocal,kind=INT_TYPE)) = 0

             ! and remove garbage in c
             c(int(rowLocal,kind=INT_TYPE),int(colLocal,kind=INT_TYPE)) = 0
           endif
         enddo
       enddo
     endif ! isUpper
   endif ! present (isUpper)

   if (present(isLower)) then
     if (isLower) then
       ! a should only be lower triangular

       ! do a dirty hack set upper half to zero
       do ii=1, na
         do jj=ii+1, na
           if (map_global_array_index_to_local_index(int(ii,kind=c_int), int(jj,kind=c_int), rowLocal, &
                                                    colLocal, int(nblk,kind=c_int), int(np_rows,kind=c_int), &
                                                    int(np_cols,kind=c_int), int(my_prow,kind=c_int), &
                                                    int(my_pcol,kind=c_int) ) ) then
             a(int(rowLocal,kind=INT_TYPE),int(colLocal,kind=INT_TYPE)) = 0


             ! and remove garber in c
             c(int(rowLocal,kind=INT_TYPE),int(colLocal,kind=INT_TYPE)) = 0

           endif
         enddo
       enddo
     endif ! isLower
   endif ! present(isLower)

   ! tmp2 = a * b
#ifdef WITH_MPI
   call p&
         &BLAS_CHAR&
         &gemm(transa_NTC,"N", na, na, na, ONE, a, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, b, 1_BLAS_KIND, 1_BLAS_KIND, &
               sc_desc, ZERO, tmp2, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc)
#else
   call BLAS_CHAR&
        &gemm(transa_NTC,"N", na, na, na, ONE, a, na, b, na, ZERO, tmp2, na)
#endif

   if (present(isUpper)) then
     if (isUpper) then
       ! tmp2 should only be upper triangular

       ! do a dirty hack set lower half to zero
       do ii=1, na
         do jj=1, ii-1
           if (map_global_array_index_to_local_index(int(ii,kind=c_int), int(jj,kind=c_int), rowLocal, &
                                                    colLocal, int(nblk,kind=c_int), int(np_rows,kind=c_int), &
                                                    int(np_cols,kind=c_int), int(my_prow,kind=c_int), &
                                                    int(my_pcol,kind=c_int) ) ) then
             tmp2(int(rowLocal,kind=INT_TYPE),int(colLocal,kind=INT_TYPE)) = 0
           endif
         enddo
       enddo
     endif ! isUpper
   endif ! present (isUpper)

   if (present(isLower)) then
     if (isLower) then
       ! a should only be lower triangular
       ! do a dirty hack set upper half to zero
       do ii=1, na
         do jj=ii+1, na
           if (map_global_array_index_to_local_index(int(ii,kind=c_int), int(jj,kind=c_int), rowLocal, &
                                                    colLocal, int(nblk,kind=c_int), int(np_rows,kind=c_int), &
                                                    int(np_cols,kind=c_int), int(my_prow,kind=c_int), &
                                                    int(my_pcol,kind=c_int) ) ) then
             tmp2(int(rowLocal,kind=INT_TYPE),int(colLocal,kind=INT_TYPE)) = 0
           endif
         enddo
       enddo
     endif ! isLower
   endif ! present(isLower)



   ! compare tmp2 with c
   tmp2(:,:) = tmp2(:,:) - c(:,:)

#ifdef WITH_MPI
      ! dirty hack: the last argument should be a real array, but is not referenced
      ! if mode = "M", thus we get away with a complex argument
      norm = p&
              &BLAS_CHAR&
              &lange("M", na, na, tmp2, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, &
#if COMPLEXCASE == 1              
              tmp1_real)
#else
              tmp1)
#endif
#else /* WITH_MPI */
      ! dirty hack: the last argument should be a real array, but is not referenced
      ! if mode = "M", thus we get away with a complex argument
      norm = BLAS_CHAR&
             &lange("M", na, na, tmp2, na_rows, &
#if COMPLEXCASE == 1              
              tmp1_real)
#else
              tmp1)
#endif
#endif /* WITH_MPI */

#ifdef WITH_MPI
      call mpi_allreduce(norm, normmax, 1_MPI_KIND, MPI_REAL_PRECISION, MPI_MAX, MPI_COMM_WORLD, mpierr)
#else /* WITH_MPI */
      normmax = norm
#endif /* WITH_MPI */

      if (myid .eq. 0) then
        print *," Maximum error of result: ", normmax
      endif

      if (is_infinity_or_NaN(normmax)) then
        status = 1
      endif

#ifdef DOUBLE_PRECISION_REAL
      if (normmax .gt. 9e-10_rk8 ) then
        status = 1
      endif
#else
      if (normmax .gt. 9e-2_rk4 ) then
        status = 1
      endif
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
      if (normmax .gt. 9e-10_rk8 ) then
        status = 1
      endif
#else
      if (normmax .gt. 9e-2_rk4 ) then
        status = 1
      endif
#endif
    end function

   ! hermitian_multiply C-interface
   ! additional parameter na_cols is needed on top of Fortran interface
   
#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
    !c> TEST_C_INT_TYPE check_correctness_hermitian_multiply_real_double_f(char transa_NH, TEST_C_INT_TYPE na, 
    !c>                                                        double *a, double *b, double *c,
    !c>                                                        TEST_C_INT_TYPE na_rows,
    !c>                                                        TEST_C_INT_TYPE na_cols,
    !c>                                                        TEST_C_INT_TYPE sc_desc[9],
    !c>                                                        TEST_C_INT_TYPE myid);
#else
    !c> TEST_C_INT_TYPE check_correctness_hermitian_multiply_real_single_f(char transa_NH, TEST_C_INT_TYPE na, 
    !c>                                                        float *a, float *b, float *c,
    !c>                                                        TEST_C_INT_TYPE na_rows,
    !c>                                                        TEST_C_INT_TYPE na_cols,
    !c>                                                        TEST_C_INT_TYPE sc_desc[9],
    !c>                                                        TEST_C_INT_TYPE myid);
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> TEST_C_INT_TYPE check_correctness_hermitian_multiply_complex_double_f(char transa_NH, TEST_C_INT_TYPE na, 
    !c>                                                        double_complex *a, double_complex *b, double_complex *c,
    !c>                                                        TEST_C_INT_TYPE na_rows,
    !c>                                                        TEST_C_INT_TYPE na_cols,    
    !c>                                                        TEST_C_INT_TYPE sc_desc[9],
    !c>                                                        TEST_C_INT_TYPE myid);
#else
    !c> TEST_C_INT_TYPE check_correctness_hermitian_multiply_complex_single_f(char transa_NH, TEST_C_INT_TYPE na, 
    !c>                                                        float_complex *a, float_complex *b, float_complex *c,
    !c>                                                        TEST_C_INT_TYPE na_rows,
    !c>                                                        TEST_C_INT_TYPE na_cols,
    !c>                                                        TEST_C_INT_TYPE sc_desc[9],
    !c>                                                        TEST_C_INT_TYPE myid);
#endif
#endif /* COMPLEXCASE */

    function check_correctness_hermitian_multiply_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &_f (transa_NH, na, a, b, c, na_rows, na_cols, sc_desc, myid) result(status) &
      bind(C,name="check_correctness_hermitian_multiply_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      &_f")
      use iso_c_binding
      use precision_for_tests
      use test_util
      implicit none
#include "./test_precision_kinds.F90"
      
      TEST_INT_TYPE              :: status
	  character(1,C_CHAR), value :: transa_NH
      TEST_INT_TYPE, value       :: na, na_rows, na_cols, myid
      MATH_DATATYPE(kind=rck)    :: a(1:na_rows,1:na_cols), b(1:na_rows,1:na_cols), c(1:na_rows,1:na_cols)
      TEST_INT_TYPE              :: sc_desc(1:9)

      status = check_correctness_hermitian_multiply_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      & (transa_NH, na, a, b, c, na_rows, sc_desc, myid)

      end function

    !-----------------------------------------------------------------------------------------------------------

    function check_correctness_eigenvalues_frank_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    & (na, ev, z, myid) result(status)
      use iso_c_binding
      use precision_for_tests
      implicit none
#include "./test_precision_kinds.F90"

      TEST_INT_TYPE                   :: status, i, j, myid
      TEST_INT_TYPE, intent(in)       :: na
      real(kind=rck)            :: ev_analytic(na), ev(na)
      MATH_DATATYPE(kind=rck)   :: z(:,:)

#if defined(DOUBLE_PRECISION_REAL) || defined(DOUBLE_PRECISION_COMPLEX)
      real(kind=rck), parameter :: pi = 3.141592653589793238462643383279_c_double
#else
      real(kind=rck), parameter :: pi = 3.1415926535897932_c_float
#endif
      real(kind=rck)            :: tmp, maxerr
      TEST_INT_TYPE                  :: loctmp
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

     if (is_infinity_or_NaN(maxerr)) then
       status = 1
     endif

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
