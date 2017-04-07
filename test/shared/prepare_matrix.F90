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

  interface prepare_matrix_double
    module procedure prepare_matrix_complex_double
    module procedure prepare_matrix_real_double
  end interface

  interface prepare_matrix
    module procedure prepare_matrix_complex_double
    module procedure prepare_matrix_real_double
  end interface

#ifdef WANT_SINGLE_PRECISION_REAL
  interface prepare_matrix_single
    module procedure prepare_matrix_real_single
#ifdef WANT_SINGLE_PRECISION_COMPLEX
    module procedure prepare_matrix_complex_single
#endif
   end interface
#endif

  contains
#define DOUBLE_PRECISION_COMPLEX 1

    subroutine prepare_matrix_complex_double(na, myid, sc_desc, a, z, as)

      use precision
      implicit none

      integer(kind=ik), intent(in)    :: myid, na, sc_desc(:)
      complex(kind=ck8), intent(inout) :: z(:,:), a(:,:), as(:,:)

      complex(kind=ck8), parameter     :: CONE = (1.0_rk8, 0.0_rk8)

      real(kind=rk8) :: xr(size(a,dim=1), size(a,dim=2))

      integer, allocatable :: iseed(:)
      integer ::  n

      ! for getting a hermitian test matrix A we get a random matrix Z
      ! and calculate A = Z + Z**H

      ! we want different random numbers on every process
      ! (otherwise A might get rank deficient):

      call random_seed(size=n)
      allocate(iseed(n))
      iseed(:) = myid
      call random_seed(put=iseed)
      call random_number(xr)

      z(:,:) = xr(:,:)
      call RANDOM_NUMBER(xr)
      z(:,:) = z(:,:) + (0.0_rk8,1.0_rk8)*xr(:,:)

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

      deallocate(iseed)

    end subroutine

#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void prepare_matrix_complex_double_f(int na, int myid, int na_rows, int na_cols,
    !c>                                       int sc_desc[9],
    !c>                                       complex double *a, complex double *z, complex double *as);
#else
    !c> void prepare_matrix_complex_single_f(int na, int myid, int na_rows, int na_cols,
    !c>                                       int sc_desc[9],
    !c>                                       complex *a, complex *z, complex *as);
#endif
#ifdef DOUBLE_PRECISION_COMPLEX
    subroutine prepare_matrix_complex_wrapper_double&
#else
    subroutine prepare_matrix_complex_wrapper_single&
#endif
          (na, myid, na_rows, na_cols, sc_desc, a, z, as) &
#ifdef DOUBLE_PRECISION_COMPLEX
                                          bind(C, name="prepare_matrix_complex_double_f")
#else
                                          bind(C, name="prepare_matrix_complex_single_f")
#endif
      use iso_c_binding

      implicit none

      integer(kind=c_int) , value   :: myid, na, na_rows, na_cols
      integer(kind=c_int)           :: sc_desc(1:9)
#ifdef DOUBLE_PRECISION_COMPLEX
      complex(kind=c_double)        :: &
#else
      complex(kind=c_float)         :: &
#endif
                                       z(1:na_rows,1:na_cols), a(1:na_rows,1:na_cols),  &
                                       as(1:na_rows,1:na_cols)

      call prepare_matrix_complex_double(na, myid, sc_desc, a, z, as)
    end subroutine

#ifdef WANT_SINGLE_PRECISION_COMPLEX


#undef DOUBLE_PRECISION_COMPLEX

    subroutine prepare_matrix_complex_single(na, myid, sc_desc, a, z, as)

      use precision
      implicit none

      integer(kind=ik), intent(in)    :: myid, na, sc_desc(:)
      complex(kind=ck4), intent(inout) :: z(:,:), a(:,:), as(:,:)

      complex(kind=ck4), parameter     :: CONE = (1.0_rk4, 0.0_rk4)

      real(kind=rk4) :: xr(size(a,dim=1), size(a,dim=2))
      integer, allocatable :: iseed(:)
      integer :: n

      ! for getting a hermitian test matrix A we get a random matrix Z
      ! and calculate A = Z + Z**H

      ! we want different random numbers on every process
      ! (otherwise A might get rank deficient):

      call random_seed(size=n)
      allocate(iseed(n))
      iseed(:) = myid
      call random_seed(put=iseed)
      call random_number(xr)
      z(:,:) = xr(:,:)
      call random_number(xr)
      z(:,:) = z(:,:) + (0.0_rk4,1.0_rk4)*xr(:,:)

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

      deallocate(iseed)
    end subroutine

#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void prepare_matrix_complex_double_f(int na, int myid, int na_rows, int na_cols,
    !c>                                       int sc_desc[9],
    !c>                                       complex double *a, complex double *z, complex double *as);
#else
    !c> void prepare_matrix_complex_single_f(int na, int myid, int na_rows, int na_cols,
    !c>                                       int sc_desc[9],
    !c>                                       complex *a, complex *z, complex *as);
#endif
    subroutine prepare_matrix_complex_wrapper_single(na, myid, na_rows, na_cols, sc_desc, a, z, as) &
#ifdef DOUBLE_PRECISION_COMPLEX
                                          bind(C, name="prepare_matrix_complex_double_f")
#else
                                          bind(C, name="prepare_matrix_complex_single_f")
#endif
      use iso_c_binding

      implicit none

      integer(kind=c_int) , value   :: myid, na, na_rows, na_cols
      integer(kind=c_int)           :: sc_desc(1:9)
#ifdef DOUBLE_PRECISION_COMPLEX
      complex(kind=c_double)        :: &
#else
      complex(kind=c_float)         :: &
#endif
                                       z(1:na_rows,1:na_cols), a(1:na_rows,1:na_cols),  &
                                       as(1:na_rows,1:na_cols)

      call prepare_matrix_complex_single(na, myid, sc_desc, a, z, as)
    end subroutine

#endif /* WANT_SINGLE_PRECISION_COMPLEX */

#define DOUBLE_PRECISION_REAL 1

    subroutine prepare_matrix_real_double(na, myid, sc_desc, a, z, as)

      use precision
      implicit none

      integer(kind=ik), intent(in)     :: myid, na, sc_desc(:)
      real(kind=rk8), intent(inout)     :: z(:,:), a(:,:), as(:,:)

      integer, allocatable  :: iseed(:)
      integer :: n

      ! for getting a hermitian test matrix A we get a random matrix Z
      ! and calculate A = Z + Z**H

      ! we want different random numbers on every process
      ! (otherwise A might get rank deficient):

      call random_seed(size=n)
      allocate(iseed(n))
      iseed(:) = myid
      call random_seed(put=iseed)
      call random_number(z)

      a(:,:) = z(:,:)

      if (myid == 0) then
        print '(a)','| Random matrix block has been set up. (only processor 0 confirms this step)'
      endif
#ifdef WITH_MPI

#ifdef DOUBLE_PRECISION_REAL
      call pdtran(na, na, 1.0_rk8, z, 1, 1, sc_desc, 1.0_rk8, a, 1, 1, sc_desc) ! A = A + Z**T
#else
      call pstran(na, na, 1.0_rk4, z, 1, 1, sc_desc, 1.0_rk4, a, 1, 1, sc_desc) ! A = A + Z**T
#endif

#else /* WITH_MPI */
      a = a + transpose(z)
#endif /* WITH_MPI */

      if (myid == 0) then
        print '(a)','| Random matrix block has been symmetrized'
      endif

      ! save original matrix A for later accuracy checks

      as = a

      deallocate(iseed)
    end subroutine

#ifdef DOUBLE_PRECISION_REAL
    !c> void prepare_matrix_real_double_f(int na, int myid, int na_rows, int na_cols,
    !c>                                       int sc_desc[9],
    !c>                                       double *a, double *z, double *as);
#else
    !c> void prepare_matrix_real_single_f(int na, int myid, int na_rows, int na_cols,
    !c>                                       int sc_desc[9],
    !c>                                       float *a, float *z, float *as);
#endif
#ifdef DOUBLE_PRECISION_REAL
    subroutine prepare_matrix_real_wrapper_double&
#else
    subroutine prepare_matrix_real_wrapper_single&
#endif
          (na, myid, na_rows, na_cols, sc_desc, a, z, as) &
#ifdef DOUBLE_PRECISION_REAL
                                          bind(C, name="prepare_matrix_real_double_f")
#else
                                          bind(C, name="prepare_matrix_real_single_f")
#endif
      use iso_c_binding

      implicit none

      integer(kind=c_int) , value   :: myid, na, na_rows, na_cols
      integer(kind=c_int)           :: sc_desc(1:9)
#ifdef DOUBLE_PRECISION_REAL
      real(kind=c_double)           :: z(1:na_rows,1:na_cols), a(1:na_rows,1:na_cols),  &
                                       as(1:na_rows,1:na_cols)
#else
      real(kind=c_float)            :: z(1:na_rows,1:na_cols), a(1:na_rows,1:na_cols),  &
                                       as(1:na_rows,1:na_cols)
#endif
      call prepare_matrix_real_double(na, myid, sc_desc, a, z, as)
    end subroutine

#ifdef WANT_SINGLE_PRECISION_REAL

#undef DOUBLE_PRECISION_REAL

    subroutine prepare_matrix_real_single(na, myid, sc_desc, a, z, as)

      use precision
      implicit none

      integer(kind=ik), intent(in)     :: myid, na, sc_desc(:)
      real(kind=rk4), intent(inout)     :: z(:,:), a(:,:), as(:,:)

      integer, allocatable :: iseed(:)
      integer :: n

      ! for getting a hermitian test matrix A we get a random matrix Z
      ! and calculate A = Z + Z**H

      ! we want different random numbers on every process
      ! (otherwise A might get rank deficient):

      call random_seed(size=n)
      allocate(iseed(n))
      iseed(:) = myid
      call random_seed(put=iseed)
      call random_number(z)

      a(:,:) = z(:,:)

      if (myid == 0) then
        print '(a)','| Random matrix block has been set up. (only processor 0 confirms this step)'
      endif
#ifdef WITH_MPI

#ifdef DOUBLE_PRECISION_REAL
      call pdtran(na, na, 1.0_rk8, z, 1, 1, sc_desc, 1.0_rk8, a, 1, 1, sc_desc) ! A = A + Z**T
#else
      call pstran(na, na, 1.0_rk4, z, 1, 1, sc_desc, 1.0_rk4, a, 1, 1, sc_desc) ! A = A + Z**T
#endif

#else /* WITH_MPI */
      a = a + transpose(z)
#endif /* WITH_MPI */

      if (myid == 0) then
        print '(a)','| Random matrix block has been symmetrized'
      endif

      ! save original matrix A for later accuracy checks

      as = a

      deallocate(iseed)
    end subroutine

#ifdef DOUBLE_PRECISION_REAL
    !c> void prepare_matrix_real_double_f(int na, int myid, int na_rows, int na_cols,
    !c>                                       int sc_desc[9],
    !c>                                       double *a, double *z, double *as);
#else
    !c> void prepare_matrix_real_single_f(int na, int myid, int na_rows, int na_cols,
    !c>                                       int sc_desc[9],
    !c>                                       float *a, float *z, float *as);
#endif
#ifdef DOUBLE_PRECISION_REAL
    subroutine prepare_matrix_real_wrapper_double&
#else
    subroutine prepare_matrix_real_wrapper_single&
#endif
          (na, myid, na_rows, na_cols, sc_desc, a, z, as) &
#ifdef DOUBLE_PRECISION_REAL
                                          bind(C, name="prepare_matrix_real_double_f")
#else
                                          bind(C, name="prepare_matrix_real_single_f")
#endif
      use iso_c_binding

      implicit none

      integer(kind=c_int) , value   :: myid, na, na_rows, na_cols
      integer(kind=c_int)           :: sc_desc(1:9)
#ifdef DOUBLE_PRECISION_REAL
      real(kind=c_double)           :: z(1:na_rows,1:na_cols), a(1:na_rows,1:na_cols),  &
                                       as(1:na_rows,1:na_cols)
#else
      real(kind=c_float)            :: z(1:na_rows,1:na_cols), a(1:na_rows,1:na_cols),  &
                                       as(1:na_rows,1:na_cols)
#endif
      call prepare_matrix_real_single(na, myid, sc_desc, a, z, as)
    end subroutine

#endif /* WANT_SINGLE_PRECISION_REAL */


end module
