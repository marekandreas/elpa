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

module mod_prepare_matrix

  interface prepare_matrix
    module procedure prepare_matrix_complex
    module procedure prepare_matrix_real
  end interface

  contains

    subroutine prepare_matrix_complex(na, myid, sc_desc, iseed, xr, a, z, as)

      use precision
      implicit none

      integer(kind=ik), intent(in)    :: myid, na, sc_desc(:)
      integer(kind=ik), intent(inout) :: iseed(:)
      real(kind=rk), intent(inout)    :: xr(:,:)
      complex(kind=ck), intent(inout) :: z(:,:), a(:,:), as(:,:)

      complex(kind=ck), parameter     :: CZERO = (0.0_rk, 0.0_rk), CONE = (1.0_rk, 0.0_rk)

      ! for getting a hermitian test matrix A we get a random matrix Z
      ! and calculate A = Z + Z**H

      ! we want different random numbers on every process
      ! (otherwise A might get rank deficient):

      iseed(:) = myid
      call RANDOM_SEED(put=iseed)
      call RANDOM_NUMBER(xr)
      z(:,:) = xr(:,:)
      call RANDOM_NUMBER(xr)
      z(:,:) = z(:,:) + (0.0_rk,1.0_rk)*xr(:,:)

      a(:,:) = z(:,:)

      if (myid == 0) then
        print '(a)','| Random matrix block has been set up. (only processor 0 confirms this step)'
      endif
#ifdef WITH_MPI

#ifdef DOUBLE_PRECISION_COMPLEX
      call pztranc(na, na, CONE, z, 1, 1, sc_desc, CONE, a, 1, 1, sc_desc) ! A = A + Z**H
#else
      call pctranc(na, na, CONE, z, 1, 1, sc_desc, CONE, a, 1, 1, sc_desc) ! A = A + Z**H
#endif

#else /* WITH_MPI */
      a = a + transpose(conjg(z))
#endif /* WITH_MPI */

      if (myid == 0) then
        print '(a)','| Random matrix block has been symmetrized'
      endif

      ! save original matrix A for later accuracy checks

      as = a

    end subroutine

    subroutine prepare_matrix_real(na, myid, sc_desc, iseed, a, z, as)

      use precision
      implicit none

      integer(kind=ik), intent(in)     :: myid, na, sc_desc(:)
      integer(kind=ik), intent(inout)  :: iseed(:)
      real(kind=ck), intent(inout)     :: z(:,:), a(:,:), as(:,:)

      ! for getting a hermitian test matrix A we get a random matrix Z
      ! and calculate A = Z + Z**H

      ! we want different random numbers on every process
      ! (otherwise A might get rank deficient):

      iseed(:) = myid
      call RANDOM_SEED(put=iseed)
      call RANDOM_NUMBER(z)

      a(:,:) = z(:,:)

      if (myid == 0) then
        print '(a)','| Random matrix block has been set up. (only processor 0 confirms this step)'
      endif
#ifdef WITH_MPI

#ifdef DOUBLE_PRECISION_REAL
      call pdtran(na, na, 1.0_rk, z, 1, 1, sc_desc, 1.0_rk, a, 1, 1, sc_desc) ! A = A + Z**T
#else
      call pstran(na, na, 1.0_rk, z, 1, 1, sc_desc, 1.0_rk, a, 1, 1, sc_desc) ! A = A + Z**T
#endif

#else /* WITH_MPI */
      a = a + transpose(z)
#endif /* WITH_MPI */

      if (myid == 0) then
        print '(a)','| Random matrix block has been symmetrized'
      endif

      ! save original matrix A for later accuracy checks

      as = a

    end subroutine
#ifdef DOUBLE_PRECISION_REAL
    !c> void prepare_matrix_real_from_fortran_double_precision(int na, int myid, int na_rows, int na_cols,
    !c>                                       int sc_desc[9], int iseed[4096],
    !c>                                       double *a, double *z, double *as);
#else
    !c> void prepare_matrix_real_from_fortran_single_precision(int na, int myid, int na_rows, int na_cols,
    !c>                                       int sc_desc[9], int iseed[4096],
    !c>                                       float *a, float *z, float *as);
#endif
    subroutine prepare_matrix_real_wrapper(na, myid, na_rows, na_cols, sc_desc, iseed, a, z, as) &
#ifdef DOUBLE_PRECISION_REAL
                                          bind(C, name="prepare_matrix_real_from_fortran_double_precision")
#else
                                          bind(C, name="prepare_matrix_real_from_fortran_single_precision")
#endif
      use iso_c_binding

      implicit none

      integer(kind=c_int) , value   :: myid, na, na_rows, na_cols
      integer(kind=c_int)           :: sc_desc(1:9)
      integer(kind=c_int)           :: iseed(1:4096)
#ifdef DOUBLE_PRECISION_REAL
      real(kind=c_double)           :: z(1:na_rows,1:na_cols), a(1:na_rows,1:na_cols),  &
                                       as(1:na_rows,1:na_cols)
#else
      real(kind=c_float)            :: z(1:na_rows,1:na_cols), a(1:na_rows,1:na_cols),  &
                                       as(1:na_rows,1:na_cols)
#endif
      call prepare_matrix_real(na, myid, sc_desc, iseed, a, z, as)
    end subroutine

#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void prepare_matrix_complex_from_fortran_double_precision(int na, int myid, int na_rows, int na_cols,
    !c>                                       int sc_desc[9], int iseed[4096], double *xr,
    !c>                                       complex double *a, complex double *z, complex double *as);
#else
    !c> void prepare_matrix_complex_from_fortran_single_precision(int na, int myid, int na_rows, int na_cols,
    !c>                                       int sc_desc[9], int iseed[4096], float *xr,
    !c>                                       complex *a, complex *z, complex *as);
#endif
    subroutine prepare_matrix_complex_wrapper(na, myid, na_rows, na_cols, sc_desc, iseed, xr, a, z, as) &
#ifdef DOUBLE_PRECISION_COMPLEX
                                          bind(C, name="prepare_matrix_complex_from_fortran_double_precision")
#else
                                          bind(C, name="prepare_matrix_complex_from_fortran_single_precision")
#endif
      use iso_c_binding

      implicit none

      integer(kind=c_int) , value   :: myid, na, na_rows, na_cols
      integer(kind=c_int)           :: sc_desc(1:9)
      integer(kind=c_int)           :: iseed(1:4096)
#ifdef DOUBLE_PRECISION_COMPLEX
      real(kind=c_double)           :: xr(1:na_rows,1:na_cols)
      complex(kind=c_double)        :: z(1:na_rows,1:na_cols), a(1:na_rows,1:na_cols),  &
                                       as(1:na_rows,1:na_cols)
#else
      real(kind=c_float)            :: xr(1:na_rows,1:na_cols)
      complex(kind=c_float)         :: z(1:na_rows,1:na_cols), a(1:na_rows,1:na_cols),  &
                                       as(1:na_rows,1:na_cols)
#endif

      call prepare_matrix_complex(na, myid, sc_desc, iseed, xr, a, z, as)
    end subroutine

end module
