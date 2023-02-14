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
#define INT_TYPE c_int64_t
#define TEST_C_INT_TYPE_PTR long int*
#define TEST_C_INT_TYPE long int
#else
#define TEST_INT_TYPE integer(kind=c_int32_t)
#define INT_TYPE c_int32_t
#define TEST_C_INT_TYPE_PTR int*
#define TEST_C_INT_TYPE int
#endif
#ifdef HAVE_64BIT_INTEGER_MPI_SUPPORT
#define TEST_INT_MPI_TYPE integer(kind=c_int64_t)
#define INT_MPI_TYPE c_int64_t
#define TEST_C_INT_MPI_TYPE_PTR long int*
#define TEST_C_INT_MPI_TYPE long int
#else
#define TEST_INT_MPI_TYPE integer(kind=c_int32_t)
#define INT_MPI_TYPE c_int32_t
#define TEST_C_INT_MPI_TYPE_PTR int*
#define TEST_C_INT_MPI_TYPE int
#endif


    subroutine prepare_matrix_random_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    & (na, myid, sc_desc, a, z, as, is_skewsymmetric)


      !use test_util
      use tests_scalapack_interfaces

      implicit none
#include "./test_precision_kinds.F90"
      TEST_INT_TYPE, intent(in)                 :: myid, na, sc_desc(:)
      MATH_DATATYPE(kind=rck), intent(inout)    :: z(:,:), a(:,:), as(:,:)

#if COMPLEXCASE == 1
      real(kind=rk)                             :: xr(size(a,dim=1), size(a,dim=2))
#endif /* COMPLEXCASE */

      integer(kind=c_int), allocatable          :: iseed(:)
      integer(kind=c_int)                       ::  n
      integer(kind=c_int), intent(in), optional :: is_skewsymmetric
      logical                                   :: skewsymmetric

      if (present(is_skewsymmetric)) then
        if (is_skewsymmetric .eq. 1) then
          skewsymmetric = .true.
        else
          skewsymmetric = .false.
        endif      
      else
        skewsymmetric = .false.
      endif

      ! for getting a hermitian test matrix A we get a random matrix Z
      ! and calculate A = Z + Z**H
      ! in case of a skewsymmetric matrix A = Z - Z**H

      ! we want different random numbers on every process
      ! (otherwise A might get rank deficient):

      call random_seed(size=n)
      allocate(iseed(n))
      iseed(:) = myid + 1
      call random_seed(put=iseed)
#if REALCASE == 1
      call random_number(z)

      a(:,:) = z(:,:)
#endif /* REALCASE */

#if COMPLEXCASE == 1
      call random_number(xr)

      z(:,:) = xr(:,:)
      call RANDOM_NUMBER(xr)
      z(:,:) = z(:,:) + (0.0_rk,1.0_rk)*xr(:,:)
      a(:,:) = z(:,:)
#endif /* COMPLEXCASE */

      if (myid == 0) then
        print '(a)','| Random matrix block has been set up. (only processor 0 confirms this step)'
      endif

#if REALCASE == 1
#ifdef WITH_MPI
      if (skewsymmetric) then
        call p&
             &BLAS_CHAR&
             &tran(int(na,kind=BLAS_KIND), int(na,kind=BLAS_KIND), -ONE, z, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, &
                   ONE, a, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc) ! A = A + Z**T
      else
        call p&
             &BLAS_CHAR&
             &tran(int(na,kind=BLAS_KIND), int(na,kind=BLAS_KIND), ONE, z, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, &
                   ONE, a, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc) ! A = A + Z**T
      endif
#else /* WITH_MPI */
      if (skewsymmetric) then
        a = a - transpose(z)
      else
        a = a + transpose(z)
      endif
#endif /* WITH_MPI */
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef WITH_MPI
      if (skewsymmetric) then
        call p&
             &BLAS_CHAR&
             &tranc(int(na,kind=BLAS_KIND), int(na,kind=BLAS_KIND), -ONE, z, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, &
                    ONE, a, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc) ! A = A + Z**H
      else
        call p&
             &BLAS_CHAR&
             &tranc(int(na,kind=BLAS_KIND), int(na,kind=BLAS_KIND), ONE, z, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc, &
                    ONE, a, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc) ! A = A + Z**H
      endif
#else /* WITH_MPI */
      if (skewsymmetric) then
        a = a - transpose(conjg(z))
      else
        a = a + transpose(conjg(z))
      endif
#endif /* WITH_MPI */
#endif /* COMPLEXCASE */


      if (myid == 0) then
        print '(a)','| Random matrix block has been symmetrized'
      endif

      ! save original matrix A for later accuracy checks

      as = a

      deallocate(iseed)

    end subroutine

    !c> #ifdef __cplusplus
    !c> extern "C" {
    !c> #endif

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
    !c> void prepare_matrix_random_real_double_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE myid, TEST_C_INT_TYPE na_rows, 
    !c>                                          TEST_C_INT_TYPE na_cols, TEST_C_INT_TYPE sc_desc[9],
    !c>                                          double *a, double *z, double *as, int is_skewsymmetric);
#else
    !c> void prepare_matrix_random_real_single_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE myid, TEST_C_INT_TYPE na_rows, 
    !c>                                          TEST_C_INT_TYPE na_cols, TEST_C_INT_TYPE sc_desc[9],
    !c>                                          float *a, float *z, float *as, int is_skewsymmetric);
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void prepare_matrix_random_complex_double_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE myid, TEST_C_INT_TYPE na_rows, 
    !c>                                             TEST_C_INT_TYPE na_cols, TEST_C_INT_TYPE sc_desc[9],
    !c>                                             double_complex *a, double_complex *z, double_complex *as, int is_skewsymmetric);
#else
    !c> void prepare_matrix_random_complex_single_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE myid, TEST_C_INT_TYPE na_rows, 
    !c>                                             TEST_C_INT_TYPE na_cols, TEST_C_INT_TYPE sc_desc[9],
    !c>                                             float_complex *a, float_complex *z, float_complex *as, int is_skewsymmetric);
#endif
#endif /* COMPLEXCASE */

subroutine prepare_matrix_random_&
&MATH_DATATYPE&
&_wrapper_&
&PRECISION&
& (na, myid, na_rows, na_cols, sc_desc, a, z, as, is_skewsymmetric) &
   bind(C, name="prepare_matrix_random_&
   &MATH_DATATYPE&
   &_&
   &PRECISION&
   &_f")
      use iso_c_binding

      implicit none
#include "./test_precision_kinds.F90"

      TEST_INT_TYPE , value   :: myid, na, na_rows, na_cols
      integer, value          :: is_skewsymmetric
      TEST_INT_TYPE           :: sc_desc(1:9)
      MATH_DATATYPE(kind=rck) :: z(1:na_rows,1:na_cols), a(1:na_rows,1:na_cols),  &
                                 as(1:na_rows,1:na_cols)
      call prepare_matrix_random_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      & (na, myid, sc_desc, a, z, as, is_skewsymmetric=is_skewsymmetric)
    end subroutine

!----------------------------------------------------------------------------------------------------------------

    subroutine prepare_matrix_random_spd_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    & (na, myid, sc_desc, a, z, as, nblk, np_rows, np_cols, my_prow, my_pcol)

      !use test_util
      use precision_for_tests
      implicit none
#include "./test_precision_kinds.F90"
      TEST_INT_TYPE, intent(in)              :: myid, na, sc_desc(:)
      MATH_DATATYPE(kind=rck), intent(inout) :: z(:,:), a(:,:), as(:,:)
      TEST_INT_TYPE, intent(in)              ::  nblk, np_rows, np_cols, my_prow, my_pcol

      TEST_INT_TYPE                          :: ii
      integer(kind=c_int)                    :: rowLocal, colLocal


      call prepare_matrix_random_&
        &MATH_DATATYPE&
        &_&
        &PRECISION&
        & (na, myid, sc_desc, a, z, as)

      ! hermitian diagonaly dominant matrix => positive definite
      do ii=1, na
        if (map_global_array_index_to_local_index(int(ii,kind=c_int), int(ii,kind=c_int), &
                                                  rowLocal, colLocal, &
                                                  int(nblk,kind=c_int), int(np_rows,kind=c_int),      &
                                                  int(np_cols,kind=c_int), int(my_prow,kind=c_int),  &
                                                  int(my_pcol,kind=c_int) )) then
          a(int(rowLocal,kind=INT_TYPE),int(colLocal,kind=INT_TYPE)) = &
                  real(a(int(rowLocal,kind=INT_TYPE), int(colLocal,kind=INT_TYPE))) + na + 1
        end if
      end do

      as = a

   end subroutine

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
    !c> void prepare_matrix_random_spd_real_double_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE myid, TEST_C_INT_TYPE na_rows, 
    !c>                                              TEST_C_INT_TYPE na_cols, TEST_C_INT_TYPE sc_desc[9],
    !c>                                              double *a, double *z, double *as,
    !c>                                              TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols, 
    !c>                                              TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#else
    !c> void prepare_matrix_random_spd_real_single_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE myid, TEST_C_INT_TYPE na_rows, 
    !c>                                              TEST_C_INT_TYPE na_cols, TEST_C_INT_TYPE sc_desc[9],
    !c>                                              float *a, float *z, float *as,
    !c>                                              TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                              TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void prepare_matrix_random_spd_complex_double_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE myid, TEST_C_INT_TYPE na_rows, 
    !c>                                                 TEST_C_INT_TYPE na_cols, TEST_C_INT_TYPE sc_desc[9],
    !c>                                                 double_complex *a, double_complex *z, double_complex *as,
    !c>                                                 TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE np_rows, 
    !c>                                                 TEST_C_INT_TYPE np_cols, TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#else
    !c> void prepare_matrix_random_spd_complex_single_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE myid, TEST_C_INT_TYPE na_rows,
    !c>                                                 TEST_C_INT_TYPE na_cols, TEST_C_INT_TYPE sc_desc[9],
    !c>                                                 float_complex *a, float_complex *z, float_complex *as,
    !c>                                                 TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE np_rows, 
    !c>                                                 TEST_C_INT_TYPE np_cols, TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#endif
#endif /* COMPLEXCASE */

subroutine prepare_matrix_random_spd_&
&MATH_DATATYPE&
&_wrapper_&
&PRECISION&
& (na, myid, na_rows, na_cols, sc_desc, a, z, as, nblk, np_rows, np_cols, my_prow, my_pcol) &
   bind(C, name="prepare_matrix_random_spd_&
   &MATH_DATATYPE&
   &_&
   &PRECISION&
   &_f")
      use iso_c_binding

      implicit none
#include "./test_precision_kinds.F90"

      TEST_INT_TYPE , value   :: myid, na, na_rows, na_cols
      TEST_INT_TYPE           :: sc_desc(1:9)
      MATH_DATATYPE(kind=rck) :: z(1:na_rows,1:na_cols), a(1:na_rows,1:na_cols),  &
                                 as(1:na_rows,1:na_cols)
      TEST_INT_TYPE , value   :: nblk, np_rows, np_cols, my_prow, my_pcol
      call prepare_matrix_random_spd_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      & (na, myid, sc_desc, a, z, as, nblk, np_rows, np_cols, my_prow, my_pcol)
    end subroutine


!----------------------------------------------------------------------------------------------------------------
! a(i,j) = random(0,1) for i<j
!        = i for i=j (important for matrix was well-conditioned)
!        = 0 for i>j

    subroutine prepare_matrix_random_triangular_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    & (na, a, nblk, myid, na_rows, na_cols, np_rows, np_cols, my_prow, my_pcol)

      implicit none
#include "./test_precision_kinds.F90"
      TEST_INT_TYPE, intent(in)                 :: na, nblk, myid, na_rows, na_cols, np_rows, np_cols, my_prow, my_pcol
      MATH_DATATYPE(kind=rck), intent(inout)    :: a(:,:)
    
      TEST_INT_TYPE                             :: l_1, l_2, x_1, x_2, I_glob, J_glob, i_loc, j_loc
#if COMPLEXCASE == 1
      real(kind=rk)                             :: xr(size(a,dim=1), size(a,dim=2))
#endif /* COMPLEXCASE */

      integer(kind=c_int), allocatable          :: iseed(:)
      integer(kind=c_int)                       :: n
      
      
      ! we want different random numbers on every process
      ! (otherwise A might get rank deficient):

      call random_seed(size=n)
      allocate(iseed(n))
      iseed(:) = myid+1
      call random_seed(put=iseed)
      
#if REALCASE == 1
      call random_number(a)
#endif /* REALCASE */

#if COMPLEXCASE == 1
      call random_number(xr)

      a(:,:) = xr(:,:)
      call random_number(xr)
      a(:,:) = a(:,:) + (0.0_rk,1.0_rk)*xr(:,:)
#endif /* COMPLEXCASE */

      if (myid == 0) then
      print '(a)','| Random matrix block has been set up. (only processor 0 confirms this step)'
      endif
        
      ! set lower triangular part of the matrix to zero
      do i_loc=1,na_rows
          ! nblk = "NB"; np_rows = "P_r"; my_prow="p_r"  (quoted-ScaLAPACK userguide notation, p.61-63)
      	  l_1 = (i_loc-1)/nblk ! local coord of the (NBxNB) block among other blocks
	      x_1 = mod(i_loc-1, nblk) + 1 ! local coord within the block
	      I_glob = (l_1*np_rows + my_prow)*nblk + x_1
          
          do j_loc=1,na_cols
              l_2 = (j_loc-1)/nblk 
	          x_2 = mod(j_loc-1, nblk) + 1 
	          J_glob = (l_2*np_cols + my_pcol)*nblk + x_2
              
              if (I_glob == J_glob) then 
                  a(i_loc,j_loc) = I_glob 
              endif
              
              if (I_glob > J_glob) then 
                  a(i_loc,j_loc) = ZERO
              endif
          end do
      end do
      
      if (myid == 0) then
        print '(a)','| Global lower triangular part of the matrix block has been set to zero'
      endif


      deallocate(iseed)

    end subroutine

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
    !c> void prepare_matrix_random_triangular_real_double_f(TEST_C_INT_TYPE na, double *a, TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid, 
    !c>                                       TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                       TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                       TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#else
    !c> void prepare_matrix_random_triangular_real_single_f(TEST_C_INT_TYPE na, float *a, TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid, 
    !c>                                       TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                       TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                       TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void prepare_matrix_random_triangular_complex_double_f(TEST_C_INT_TYPE na, double_complex *a, TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid, 
    !c>                                       TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                       TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                       TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#else
    !c> void prepare_matrix_random_triangular_complex_single_f(TEST_C_INT_TYPE na, float_complex *a, TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid, 
    !c>                                       TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                       TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                       TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#endif
#endif /* COMPLEXCASE */

      
   subroutine prepare_matrix_random_triangular_&
   &MATH_DATATYPE&
   &_wrapper_&
   &PRECISION&
   & (na, a, nblk, myid, na_rows, na_cols, np_rows, np_cols, my_prow, my_pcol) &
   bind(C, name="prepare_matrix_random_triangular_&
   &MATH_DATATYPE&
   &_&
   &PRECISION&
   &_f")
      use iso_c_binding

      implicit none
#include "./test_precision_kinds.F90"
      TEST_INT_TYPE, value    :: na, nblk, myid, na_rows, na_cols, np_rows, np_cols, my_prow, my_pcol
      MATH_DATATYPE(kind=rck) :: a(1:na_rows,1:na_cols)
      
      call prepare_matrix_random_triangular_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      & (na, a, nblk, myid, na_rows, na_cols, np_rows, np_cols, my_prow, my_pcol)
    end subroutine
    
!----------------------------------------------------------------------------------------------------------------

   subroutine prepare_matrix_toeplitz_&
   &MATH_DATATYPE&
   &_&
   &PRECISION&
   & (na, diagonalElement, subdiagonalElement, d, sd, ds, sds, a, as, &
      nblk, np_rows, np_cols, my_prow, my_pcol)
     !use test_util
     use precision_for_tests
     implicit none
#include "./test_precision_kinds.F90"

     TEST_INT_TYPE, intent(in) :: na, nblk, np_rows, np_cols, my_prow, my_pcol
     MATH_DATATYPE(kind=rck)   :: diagonalElement, subdiagonalElement
     MATH_DATATYPE(kind=rck)   :: d(:), sd(:), ds(:), sds(:)
     
     MATH_DATATYPE(kind=rck)   :: a(:,:), as(:,:)

     TEST_INT_TYPE             :: ii
     integer(kind=c_int)       :: rowLocal, colLocal

     d(:) = diagonalElement
     sd(:) = subdiagonalElement
     a(:,:) = ZERO

     ! set up the diagonal and subdiagonals (for general solver test)
     do ii=1, na ! for diagonal elements
       if (map_global_array_index_to_local_index(int(ii,kind=c_int), int(ii,kind=c_int), rowLocal, &
                                                 colLocal, int(nblk,kind=c_int), int(np_rows,kind=c_int), &
                                                 int(np_cols,kind=c_int), int(my_prow,kind=c_int), &
                                                 int(my_pcol,kind=c_int) ) ) then
         a(int(rowLocal,kind=INT_TYPE),int(colLocal,kind=INT_TYPE)) = diagonalElement
       endif
     enddo
     do ii=1, na-1
       if (map_global_array_index_to_local_index(int(ii,kind=c_int), int(ii+1,kind=c_int), rowLocal, &
                                                 colLocal, int(nblk,kind=c_int), int(np_rows,kind=c_int), &
                                                 int(np_cols,kind=c_int), int(my_prow,kind=c_int), &
                                                 int(my_pcol,kind=c_int) ) ) then
         a(int(rowLocal,kind=INT_TYPE),int(colLocal,kind=INT_TYPE)) = subdiagonalElement
       endif
     enddo

     do ii=2, na
       if (map_global_array_index_to_local_index(int(ii,kind=c_int), int(ii-1,kind=c_int), rowLocal, &
                                                 colLocal, int(nblk,kind=c_int), int(np_rows,kind=c_int), &
                                                 int(np_cols,kind=c_int), int(my_prow,kind=c_int), &
                                                 int(my_pcol,kind=c_int) ) ) then
         a(int(rowLocal,kind=INT_TYPE),int(colLocal,kind=INT_TYPE)) = subdiagonalElement
       endif
     enddo

     ds = d
     sds = sd
     as = a
   end subroutine


   subroutine prepare_matrix_toeplitz_mixed_complex&
   &_&
   &MATH_DATATYPE&
   &_&
   &PRECISION&
#if COMPLEXCASE == 1
   & (na, diagonalElement, subdiagonalElement, d, sd, ds, sds, a, as, &
      nblk, np_rows, np_cols, my_prow, my_pcol)
#endif
#if REALCASE == 1
   & (na, diagonalElement, subdiagonalElement, d, sd, ds, sds, &
      nblk, np_rows, np_cols, my_prow, my_pcol)
#endif
     !use test_util
     implicit none

     TEST_INT_TYPE, intent(in)     :: na, nblk, np_rows, np_cols, my_prow, my_pcol
     real(kind=C_DATATYPE_KIND)    :: diagonalElement, subdiagonalElement

     real(kind=C_DATATYPE_KIND)    :: d(:), sd(:), ds(:), sds(:)

#if COMPLEXCASE == 1
     complex(kind=C_DATATYPE_KIND) :: a(:,:), as(:,:)
#endif
#if REALCASE == 1
#endif

     TEST_INT_TYPE                 :: ii
     integer(kind=c_int)           :: rowLocal, colLocal
#if COMPLEXCASE == 1
     d(:) = diagonalElement
     sd(:) = subdiagonalElement

     ! set up the diagonal and subdiagonals (for general solver test)
     do ii=1, na ! for diagonal elements
       if (map_global_array_index_to_local_index(int(ii,kind=c_int), int(ii,kind=c_int), rowLocal, &
                                                 colLocal, int(nblk,kind=c_int),                   &
                                                 int(np_rows,kind=c_int), int(np_cols,kind=c_int),                 &
                                                 int(my_prow,kind=c_int), int(my_pcol,kind=c_int) )) then
         a(int(rowLocal,kind=INT_TYPE),int(colLocal,kind=INT_TYPE)) = diagonalElement
       endif
     enddo
     do ii=1, na-1
       if (map_global_array_index_to_local_index(int(ii,kind=c_int), int(ii+1,kind=c_int), rowLocal, &
                                                 colLocal, int(nblk,kind=c_int),                   &
                                                 int(np_rows,kind=c_int), int(np_cols,kind=c_int),                 &
                                                 int(my_prow,kind=c_int), int(my_pcol,kind=c_int) )) then
         a(int(rowLocal,kind=INT_TYPE),int(colLocal,kind=INT_TYPE)) = subdiagonalElement
       endif
     enddo

     do ii=2, na
       if (map_global_array_index_to_local_index(int(ii,kind=c_int), int(ii-1,kind=c_int), rowLocal, &
                                                 colLocal, int(nblk,kind=c_int),                   &
                                                 int(np_rows,kind=c_int), int(np_cols,kind=c_int),                 &
                                                 int(my_prow,kind=c_int), int(my_pcol,kind=c_int) )) then
         a(int(rowLocal,kind=INT_TYPE),int(colLocal,kind=INT_TYPE)) = subdiagonalElement
       endif
     enddo

     ds = d
     sds = sd
     as = a
#endif
   end subroutine


   
#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
    !c> void prepare_matrix_toeplitz_real_double_f(TEST_C_INT_TYPE na, 
    !c>           double diagonalElement, double subdiagonalElement,
    !c>           double *d, double *sd, double *ds, double *sds,
    !c>           double *a, double *as, TEST_C_INT_TYPE nblk, 
    !c>           TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>           TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>           TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#else
    !c> void prepare_matrix_toeplitz_real_single_f(TEST_C_INT_TYPE na, 
    !c>           float diagonalElement, float subdiagonalElement,
    !c>           float *d, float *sd, float *ds, float *sds,
    !c>           float *a, float *as, TEST_C_INT_TYPE nblk, 
    !c>           TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>           TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>           TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);    
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void prepare_matrix_toeplitz_complex_double_f(TEST_C_INT_TYPE na, 
    !c>           double diagonalElement, double subdiagonalElement,
    !c>           double *d, double *sd, double *ds, double *sds,
    !c>           double_complex *a, double_complex *as, TEST_C_INT_TYPE nblk, 
    !c>           TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>           TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>           TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#else
    !c> void prepare_matrix_toeplitz_complex_single_f(TEST_C_INT_TYPE na, 
    !c>           float diagonalElement, float subdiagonalElement,
    !c>           float *d, float *sd, float *ds, float *sds,
    !c>           float_complex *a, float_complex *as, TEST_C_INT_TYPE nblk, 
    !c>           TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>           TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>           TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#endif
#endif /* COMPLEXCASE */

   ! extra parameters na_rows, na_cols are needed for C-interface
   ! d, sd, ds, sds are real for C -interface
   subroutine prepare_matrix_toeplitz_&
   &MATH_DATATYPE&
   &_wrapper_&
   &PRECISION&
   & (na, diagonalElement, subdiagonalElement, d, sd, ds, sds, a, as, &
      nblk, na_rows, na_cols, np_rows, np_cols, my_prow, my_pcol) &
   bind(C, name="prepare_matrix_toeplitz_&
   &MATH_DATATYPE&
   &_&
   &PRECISION&
   &_f")
      use iso_c_binding

      implicit none
#include "./test_precision_kinds.F90"
      TEST_INT_TYPE, value    :: na, nblk, na_rows, na_cols, np_rows, np_cols, my_prow, my_pcol
      real(kind=rk), value    :: diagonalElement, subdiagonalElement
      real(kind=rk)           :: d(1:na), sd(1:na), ds(1:na), sds(1:na)
      MATH_DATATYPE(kind=rck) :: a(1:na_rows,1:na_cols), as(1:na_rows,1:na_cols)

#if REALCASE == 1
      call prepare_matrix_toeplitz_&
#else
      call prepare_matrix_toeplitz_mixed_complex_&
#endif
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      & (na, diagonalElement, subdiagonalElement, d, sd, ds, sds, a, as, &
      nblk, np_rows, np_cols, my_prow, my_pcol)
    end subroutine
    
!----------------------------------------------------------------------------------------------------------------

!----------------------------------------------------------------------------------------------------------------

   subroutine prepare_matrix_frank_&
   &MATH_DATATYPE&
   &_&
   &PRECISION&
   & (na, a, z, as, nblk, np_rows, np_cols, my_prow, my_pcol)
     !use test_util
     use precision_for_tests
     implicit none

     TEST_INT_TYPE, intent(in)     :: na, nblk, np_rows, np_cols, my_prow, my_pcol

#if REALCASE == 1
     real(kind=C_DATATYPE_KIND)    :: a(:,:), z(:,:), as(:,:)
#endif
#if COMPLEXCASE == 1
     complex(kind=C_DATATYPE_KIND) :: a(:,:), z(:,:), as(:,:)
#endif

     TEST_INT_TYPE                 :: i, j
     integer(kind=c_int)           :: rowLocal, colLocal

     do i = 1, na
       do j = 1, na
         if (map_global_array_index_to_local_index(int(i,kind=c_int), int(j,kind=c_int), rowLocal, &
                                                 colLocal, int(nblk,kind=c_int),                   &
                                                 int(np_rows,kind=c_int), int(np_cols,kind=c_int),                 &
                                                 int(my_prow,kind=c_int), int(my_pcol,kind=c_int) )) then
           if (j .le. i) then
             a(int(rowLocal,kind=INT_TYPE),int(colLocal,kind=INT_TYPE)) = &
                     real((na+1-i), kind=C_DATATYPE_KIND) / real(na, kind=C_DATATYPE_KIND)
           else
             a(int(rowLocal,kind=INT_TYPE),int(colLocal,kind=INT_TYPE)) = &
                     real((na+1-j), kind=C_DATATYPE_KIND) / real(na, kind=C_DATATYPE_KIND)
           endif
         endif
       enddo
     enddo

     z(:,:)  = a(:,:)
     as(:,:) = a(:,:)

   end subroutine

!----------------------------------------------------------------------------------------------------------------

    subroutine prepare_matrix_unit_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    & (na, a, nblk, myid, na_rows, na_cols, np_rows, np_cols, my_prow, my_pcol)
      implicit none
#include "./test_precision_kinds.F90"
      TEST_INT_TYPE, intent(in)                 :: na, nblk, myid, na_rows, na_cols, np_rows, np_cols, my_prow, my_pcol
      MATH_DATATYPE(kind=rck), intent(inout)    :: a(:,:)
    
      TEST_INT_TYPE                             :: l_1, l_2, x_1, x_2, I_glob, J_glob, i_loc, j_loc

      
      a(:,:) = ZERO
	  
      ! set lower triangular part of the matrix to zero
      do i_loc=1,na_rows
          ! nblk = "NB"; np_rows = "P_r"; my_prow="p_r"  (quoted-ScaLAPACK userguide notation, p.61-63)
      	  l_1 = (i_loc-1)/nblk ! local coord of the (NBxNB) block among other blocks
	      x_1 = mod(i_loc-1, nblk) + 1 ! local coord within the block
	      I_glob = (l_1*np_rows + my_prow)*nblk + x_1
          
          do j_loc=1,na_cols
              l_2 = (j_loc-1)/nblk 
	          x_2 = mod(j_loc-1, nblk) + 1 
	          J_glob = (l_2*np_cols + my_pcol)*nblk + x_2
              
			  if (I_glob == J_glob) then 
                  a(i_loc,j_loc) = ONE
              endif
          end do
      end do
      
      if (myid == 0) then
        print '(a)','| Unit matrix was set'
      endif

    end subroutine

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
    !c> void prepare_matrix_unit_real_double_f(TEST_C_INT_TYPE na, double *a, TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid, 
    !c>                                       TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                       TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                       TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#else
    !c> void prepare_matrix_unit_real_single_f(TEST_C_INT_TYPE na, float *a, TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid, 
    !c>                                       TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                       TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                       TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void prepare_matrix_unit_complex_double_f(TEST_C_INT_TYPE na, double_complex *a, TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid, 
    !c>                                       TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                       TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                       TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#else
    !c> void prepare_matrix_unit_complex_single_f(TEST_C_INT_TYPE na, float_complex *a, TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid, 
    !c>                                       TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                       TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                       TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#endif
#endif /* COMPLEXCASE */

      
   subroutine prepare_matrix_unit_&
   &MATH_DATATYPE&
   &_wrapper_&
   &PRECISION&
   & (na, a, nblk, myid, na_rows, na_cols, np_rows, np_cols, my_prow, my_pcol) &
   bind(C, name="prepare_matrix_unit_&
   &MATH_DATATYPE&
   &_&
   &PRECISION&
   &_f")
      use iso_c_binding

      implicit none
#include "./test_precision_kinds.F90"
      TEST_INT_TYPE, value    :: na, nblk, myid, na_rows, na_cols, np_rows, np_cols, my_prow, my_pcol
      MATH_DATATYPE(kind=rck) :: a(1:na_rows,1:na_cols)
      
      call prepare_matrix_unit_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      & (na, a, nblk, myid, na_rows, na_cols, np_rows, np_cols, my_prow, my_pcol)
    end subroutine
    
! vim: syntax=fortran
