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
!    This particular source code file contains additions, changes and
!    enhancements authored by Intel Corporation which is not part of
!    the ELPA consortium.
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
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".

#include "config-f90.h"

!> \brief Fortran module which provides helper routines for matrix calculations
module ELPA1_AUXILIARY
  use elpa_utilities

  implicit none

  public :: elpa_mult_at_b_real_double      !< Multiply double-precision real matrices A**T * B
  public :: mult_at_b_real                  !< Old, deprecated interface to multiply double-precision real matrices A**T * B. DO NOT USE

  public :: elpa_mult_ah_b_complex_double   !< Multiply double-precision complex matrices A**H * B
  public :: mult_ah_b_complex               !< Old, deprecated interface to multiply double-precision complex matrices A**H * B. DO NOT USE

  public :: elpa_invert_trm_real_double     !< Invert double-precision real triangular matrix
  public :: invert_trm_real                 !< Old, deprecated interface for inversion of double-precision real triangular matrix. DO NOT USE

  public :: elpa_invert_trm_complex_double  !< Invert double-precision complex triangular matrix
  public :: invert_trm_complex              !< Old, deprecated interface to invert double-precision complex triangular matrix. DO NOT USE

  public :: elpa_cholesky_real_double       !< Cholesky factorization of a double-precision real matrix
  public :: cholesky_real                   !< Old, deprecated name for Cholesky factorization of a double-precision real matrix. DO NOT USE

  public :: elpa_cholesky_complex_double    !< Cholesky factorization of a double-precision complex matrix
  public :: cholesky_complex                !< Old, deprecated interface for a Cholesky factorization of a double-precision complex matrix. DO NOT USE

  public :: elpa_solve_tridi_double         !< Solve tridiagonal eigensystem for a double-precision matrix with divide and conquer method
  public :: solve_tridi                     !< Old, deprecated interface to solve tridiagonal eigensystem for a double-precision matrix with divide and conquer method

#ifdef WANT_SINGLE_PRECISION_REAL
  public :: elpa_cholesky_real_single       !< Cholesky factorization of a single-precision real matrix
  public :: elpa_invert_trm_real_single     !< Invert single-precision real triangular matrix
  public :: elpa_mult_at_b_real_single      !< Multiply single-precision real matrices A**T * B
  public :: elpa_solve_tridi_single         !< Solve tridiagonal eigensystem for a single-precision matrix with divide and conquer method
#endif

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  public :: elpa_cholesky_complex_single    !< Cholesky factorization of a single-precision complex matrix
  public :: elpa_invert_trm_complex_single  !< Invert single-precision complex triangular matrix
  public :: elpa_mult_ah_b_complex_single   !< Multiply single-precision complex matrices A**H * B
#endif

!> \brief  cholesky_real: old, deprecated interface for Cholesky factorization of a double-precision real symmetric matrix
!> \details
!>
!> \param  na                   Order of matrix
!> \param  a(lda,matrixCols)    Distributed matrix which should be factorized.
!>                              Distribution is like in Scalapack.
!>                              Only upper triangle needs to be set.
!>                              On return, the upper triangle contains the Cholesky factor
!>                              and the lower triangle is set to 0.
!> \param  lda                  Leading dimension of a
!> \param                       matrixCols  local columns of matrix a
!> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param wantDebug             logical, more debug information on failure
!> \result succes                logical, reports success or failure
  interface cholesky_real
    module procedure elpa_cholesky_real_double
  end interface

!> \brief  Old, deprecated interface invert_trm_real: Inverts a upper double-precision triangular matrix
!> \details
!> \param  na                   Order of matrix
!> \param  a(lda,matrixCols)    Distributed matrix which should be inverted
!>                              Distribution is like in Scalapack.
!>                              Only upper triangle needs to be set.
!>                              The lower triangle is not referenced.
!> \param  lda                  Leading dimension of a
!> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
!> \param  matrixCols           local columns of matrix a
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param wantDebug             logical, more debug information on failure
!> \param result                logical, reports success or failure

  interface invert_trm_real
    module procedure elpa_invert_trm_real_double
  end interface

!> \brief  old, deprecated interface cholesky_complex: Cholesky factorization of a double-precision complex hermitian matrix
!> \details
!> \param  na                   Order of matrix
!> \param  a(lda,matrixCols)    Distributed matrix which should be factorized.
!>                              Distribution is like in Scalapack.
!>                              Only upper triangle needs to be set.
!>                              On return, the upper triangle contains the Cholesky factor
!>                              and the lower triangle is set to 0.
!> \param  lda                  Leading dimension of a
!> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
!> \param  matrixCols           local columns of matrix a
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param wantDebug             logical, more debug information on failure
!> \result succes               logical, reports success or failure


  interface cholesky_complex
    module procedure elpa_cholesky_real_double
  end interface

!> \brief  old, deprecated interface invert_trm_complex: Inverts a double-precision complex upper triangular matrix
!> \details
!> \param  na                   Order of matrix
!> \param  a(lda,matrixCols)    Distributed matrix which should be inverted
!>                              Distribution is like in Scalapack.
!>                              Only upper triangle needs to be set.
!>                              The lower triangle is not referenced.
!> \param  lda                  Leading dimension of a
!> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
!> \param  matrixCols           local columns of matrix a
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param wantDebug             logical, more debug information on failure
!> \result succes               logical, reports success or failure

  interface invert_trm_complex
    module procedure elpa_invert_trm_complex_double
  end interface

!> \brief  mult_at_b_real: Performs C : = A**T * B for double matrices
!> this is the old, deprecated interface for the newer elpa_mult_at_b_real
!>         where   A is a square matrix (na,na) which is optionally upper or lower triangular
!>                 B is a (na,ncb) matrix
!>                 C is a (na,ncb) matrix where optionally only the upper or lower
!>                   triangle may be computed
!> \details

!> \param  uplo_a               'U' if A is upper triangular
!>                              'L' if A is lower triangular
!>                              anything else if A is a full matrix
!>                              Please note: This pertains to the original A (as set in the calling program)
!>                                           whereas the transpose of A is used for calculations
!>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!>                              i.e. it may contain arbitrary numbers
!> \param uplo_c                'U' if only the upper diagonal part of C is needed
!>                              'L' if only the upper diagonal part of C is needed
!>                              anything else if the full matrix C is needed
!>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!> \param na                    Number of rows/columns of A, number of rows of B and C
!> \param ncb                   Number of columns  of B and C
!> \param a                     matrix a
!> \param lda                   leading dimension of matrix a
!> \param b                     matrix b
!> \param ldb                   leading dimension of matrix b
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param c                     matrix c
!> \param ldc                   leading dimension of matrix c
  interface mult_at_b_real
    module procedure elpa_mult_at_b_real_double
  end interface

!> \brief  Old, deprecated interface mult_ah_b_complex: Performs C : = A**H * B for double-precision matrices
!>         where   A is a square matrix (na,na) which is optionally upper or lower triangular
!>                 B is a (na,ncb) matrix
!>                 C is a (na,ncb) matrix where optionally only the upper or lower
!>                   triangle may be computed
!> \details
!>
!> \param  uplo_a               'U' if A is upper triangular
!>                              'L' if A is lower triangular
!>                              anything else if A is a full matrix
!>                              Please note: This pertains to the original A (as set in the calling program)
!>                                           whereas the transpose of A is used for calculations
!>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!>                              i.e. it may contain arbitrary numbers
!> \param uplo_c                'U' if only the upper diagonal part of C is needed
!>                              'L' if only the upper diagonal part of C is needed
!>                              anything else if the full matrix C is needed
!>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!> \param na                    Number of rows/columns of A, number of rows of B and C
!> \param ncb                   Number of columns  of B and C
!> \param a                     matrix a
!> \param lda                   leading dimension of matrix a
!> \param b                     matrix b
!> \param ldb                   leading dimension of matrix b
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param c                     matrix c
!> \param ldc                   leading dimension of matrix c
  interface mult_ah_b_complex
    module procedure elpa_mult_ah_b_complex_double
  end interface


!> \brief  solve_tridi: Old, deprecated interface to solve a double-precision tridiagonal eigensystem for a double-precision matrix with divide and conquer method
!> \details
!>
!> \param na                    Matrix dimension
!> \param nev                   number of eigenvalues/vectors to be computed
!> \param d                     array d(na) on input diagonal elements of tridiagonal matrix, on
!>                              output the eigenvalues in ascending order
!> \param e                     array e(na) on input subdiagonal elements of matrix, on exit destroyed
!> \param q                     on exit : matrix q(ldq,matrixCols) contains the eigenvectors
!> \param ldq                   leading dimension of matrix q
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param matrixCols            columns of matrix q
!> \param mpi_comm_rows         MPI communicator for rows
!> \param mpi_comm_cols         MPI communicator for columns
!> \param wantDebug             logical, give more debug information if .true.
!> \result success              logical, .true. on success, else .false.
  interface solve_tridi
    module procedure elpa_solve_tridi_double
  end interface

  contains

!> \brief  cholesky_real_double: Cholesky factorization of a double-precision real symmetric matrix
!> \details
!>
!> \param  na                   Order of matrix
!> \param  a(lda,matrixCols)    Distributed matrix which should be factorized.
!>                              Distribution is like in Scalapack.
!>                              Only upper triangle needs to be set.
!>                              On return, the upper triangle contains the Cholesky factor
!>                              and the lower triangle is set to 0.
!> \param  lda                  Leading dimension of a
!> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
!> \param  matrixCols           local columns of matrix a
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param wantDebug             logical, more debug information on failure
!> \param succes                logical, reports success or failure

#define REALCASE 1
#define DOUBLE_PRECISION
#include "precision_macros.h"


   function elpa_cholesky_real_double(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, &
                                            wantDebug) result(success)
#include "elpa_cholesky_template.X90"

    end function elpa_cholesky_real_double

#ifdef WANT_SINGLE_PRECISION_REAL
#define REALCASE 1
#define SINGLE_PRECISION
#include "precision_macros.h"

!> \brief  cholesky_real_single: Cholesky factorization of a single-precision real symmetric matrix
!> \details
!>
!> \param  na                   Order of matrix
!> \param  a(lda,matrixCols)    Distributed matrix which should be factorized.
!>                              Distribution is like in Scalapack.
!>                              Only upper triangle needs to be set.
!>                              On return, the upper triangle contains the Cholesky factor
!>                              and the lower triangle is set to 0.
!> \param  lda                  Leading dimension of a
!> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
!> \param  matrixCols           local columns of matrix a
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param wantDebug             logical, more debug information on failure
!> \param succes                logical, reports success or failure

   function elpa_cholesky_real_single(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, &
                                            wantDebug) result(success)
#include "elpa_cholesky_template.X90"

    end function elpa_cholesky_real_single

#endif /* WANT_SINGLE_PRECSION_REAL */

#define REALCASE 1
#define DOUBLE_PRECISION
#include "precision_macros.h"
!> \brief  elpa_invert_trm_real_double: Inverts a double-precision real upper triangular matrix
!> \details
!> \param  na                   Order of matrix
!> \param  a(lda,matrixCols)    Distributed matrix which should be inverted
!>                              Distribution is like in Scalapack.
!>                              Only upper triangle needs to be set.
!>                              The lower triangle is not referenced.
!> \param  lda                  Leading dimension of a
!> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
!> \param  matrixCols           local columns of matrix a
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param wantDebug             logical, more debug information on failure
!> \result succes               logical, reports success or failure
    function elpa_invert_trm_real_double(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug) result(success)
#include "elpa_invert_trm.X90"
     end function elpa_invert_trm_real_double

#if WANT_SINGLE_PRECISION_REAL
#define REALCASE 1
#define SINGLE_PRECISION
#include "precision_macros.h"

!> \brief  elpa_invert_trm_real_single: Inverts a single-precision real upper triangular matrix
!> \details
!> \param  na                   Order of matrix
!> \param  a(lda,matrixCols)    Distributed matrix which should be inverted
!>                              Distribution is like in Scalapack.
!>                              Only upper triangle needs to be set.
!>                              The lower triangle is not referenced.
!> \param  lda                  Leading dimension of a
!> \param                       matrixCols  local columns of matrix a
!> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param wantDebug             logical, more debug information on failure
!> \result succes               logical, reports success or failure
    function elpa_invert_trm_real_single(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug) result(success)
#include "elpa_invert_trm.X90"
    end function elpa_invert_trm_real_single

#endif /* WANT_SINGLE_PRECISION_REAL */


#define COMPLEXCASE 1
#define DOUBLE_PRECISION
#include "precision_macros.h"

!> \brief  elpa_cholesky_complex_double: Cholesky factorization of a double-precision complex hermitian matrix
!> \details
!> \param  na                   Order of matrix
!> \param  a(lda,matrixCols)    Distributed matrix which should be factorized.
!>                              Distribution is like in Scalapack.
!>                              Only upper triangle needs to be set.
!>                              On return, the upper triangle contains the Cholesky factor
!>                              and the lower triangle is set to 0.
!> \param  lda                  Leading dimension of a
!> \param                       matrixCols  local columns of matrix a
!> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param wantDebug             logical, more debug information on failure
!> \result succes               logical, reports success or failure
    function elpa_cholesky_complex_double(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug) result(success)

#include "elpa_cholesky_template.X90"

    end function elpa_cholesky_complex_double


#ifdef WANT_SINGLE_PRECISION_COMPLEX
#define COMPLEXCASE 1
#define SINGLE_PRECISION
#include "precision_macros.h"

!> \brief  elpa_cholesky_complex_single: Cholesky factorization of a single-precision complex hermitian matrix
!> \details
!> \param  na                   Order of matrix
!> \param  a(lda,matrixCols)    Distributed matrix which should be factorized.
!>                              Distribution is like in Scalapack.
!>                              Only upper triangle needs to be set.
!>                              On return, the upper triangle contains the Cholesky factor
!>                              and the lower triangle is set to 0.
!> \param  lda                  Leading dimension of a
!> \param                       matrixCols  local columns of matrix a
!> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param wantDebug             logical, more debug information on failure
!> \result succes               logical, reports success or failure
    function elpa_cholesky_complex_single(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug) result(success)

#include "elpa_cholesky_template.X90"

    end function elpa_cholesky_complex_single

#endif /* WANT_SINGLE_PRECISION_COMPLEX */

#define COMPLEXCASE 1
#define DOUBLE_PRECISION
#include "precision_macros.h"

!> \brief  elpa_invert_trm_complex_double: Inverts a double-precision complex upper triangular matrix
!> \details
!> \param  na                   Order of matrix
!> \param  a(lda,matrixCols)    Distributed matrix which should be inverted
!>                              Distribution is like in Scalapack.
!>                              Only upper triangle needs to be set.
!>                              The lower triangle is not referenced.
!> \param  lda                  Leading dimension of a
!> \param                       matrixCols  local columns of matrix a
!> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param wantDebug             logical, more debug information on failure
!> \result succes               logical, reports success or failure

     function elpa_invert_trm_complex_double(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug) result(success)
#include "elpa_invert_trm.X90"
    end function elpa_invert_trm_complex_double

#ifdef WANT_SINGLE_PRECISION_COMPLEX
#define COMPLEXCASE 1
#define SINGLE_PRECISION
#include "precision_macros.h"
!> \brief  elpa_invert_trm_complex_single: Inverts a single-precision complex upper triangular matrix
!> \details
!> \param  na                   Order of matrix
!> \param  a(lda,matrixCols)    Distributed matrix which should be inverted
!>                              Distribution is like in Scalapack.
!>                              Only upper triangle needs to be set.
!>                              The lower triangle is not referenced.
!> \param  lda                  Leading dimension of a
!> \param                       matrixCols  local columns of matrix a
!> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param wantDebug             logical, more debug information on failure
!> \result succes               logical, reports success or failure

    function elpa_invert_trm_complex_single(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug) result(success)
#include "elpa_invert_trm.X90"
    end function elpa_invert_trm_complex_single

#endif /* WANT_SINGE_PRECISION_COMPLEX */

#undef DOUBLE_PRECISION_REAL
#undef REAL_DATATYPE
#define DOUBLE_PRECISION_REAL 1
#define REAL_DATATYPE rk8
!> \brief  mult_at_b_real_double: Performs C : = A**T * B
!>         where   A is a square matrix (na,na) which is optionally upper or lower triangular
!>                 B is a (na,ncb) matrix
!>                 C is a (na,ncb) matrix where optionally only the upper or lower
!>                   triangle may be computed
!> \details

!> \param  uplo_a               'U' if A is upper triangular
!>                              'L' if A is lower triangular
!>                              anything else if A is a full matrix
!>                              Please note: This pertains to the original A (as set in the calling program)
!>                                           whereas the transpose of A is used for calculations
!>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!>                              i.e. it may contain arbitrary numbers
!> \param uplo_c                'U' if only the upper diagonal part of C is needed
!>                              'L' if only the upper diagonal part of C is needed
!>                              anything else if the full matrix C is needed
!>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!> \param na                    Number of rows/columns of A, number of rows of B and C
!> \param ncb                   Number of columns  of B and C
!> \param a                     matrix a
!> \param lda                   leading dimension of matrix a
!> \param b                     matrix b
!> \param ldb                   leading dimension of matrix b
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param c                     matrix c
!> \param ldc                   leading dimension of matrix c
!> \result success

    function elpa_mult_at_b_real_double(uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, nblk, &
                              mpi_comm_rows, mpi_comm_cols, c, ldc, ldcCols) result(success)
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#else
      use timings_dummy
#endif
      use elpa1_compute
      use elpa_mpi
      use precision
      implicit none

      character*1                   :: uplo_a, uplo_c

      integer(kind=ik), intent(in)  :: na, lda, ldaCols, ldb, ldbCols, ldc, ldcCols, nblk
      integer(kind=ik)              :: ncb, mpi_comm_rows, mpi_comm_cols
      real(kind=REAL_DATATYPE)                 :: a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)

      integer(kind=ik)              :: my_prow, my_pcol, np_rows, np_cols, mpierr
      integer(kind=ik)              :: l_cols, l_rows, l_rows_np
      integer(kind=ik)              :: np, n, nb, nblk_mult, lrs, lre, lcs, lce
      integer(kind=ik)              :: gcol_min, gcol, goff
      integer(kind=ik)              :: nstor, nr_done, noff, np_bc, n_aux_bc, nvals
      integer(kind=ik), allocatable :: lrs_save(:), lre_save(:)

      logical                       :: a_lower, a_upper, c_lower, c_upper

      real(kind=REAL_DATATYPE), allocatable    :: aux_mat(:,:), aux_bc(:), tmp1(:,:), tmp2(:,:)
      integer(kind=ik)              :: istat
      character(200)                :: errorMessage
      logical                       :: success

#ifdef DOUBLE_PRECISION_REAL
      call timer%start("elpa_mult_at_b_real_double")
#else
      call timer%start("elpa_mult_at_b_real_single")
#endif

      success = .true.

!      if (na .lt. lda) then
!        print *,"na lt lda ",na,lda
!        stop
!      endif
!      if (na .lt. ldb) then
!        print *,"na lt ldb ",na,ldb
!        stop
!      endif
!      if (na .lt. ldc) then
!        print *,"na lt ldc ",na,ldc
!        stop
!      endif
!      if (na .lt. ldaCols) then
!        print *,"na lt ldaCols ",na,ldaCols
!        stop
!      endif
!      if (na .lt. ldbCols) then
!        print *,"na lt ldbCols ",na,ldbCols
!        stop
!      endif
!      if (na .lt. ldcCols) then
!        print *,"na lt ldcCols ",na,ldcCols
!        stop
!      endif
      call timer%start("mpi_communication")
      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
      call timer%stop("mpi_communication")
      l_rows = local_index(na,  my_prow, np_rows, nblk, -1) ! Local rows of a and b
      l_cols = local_index(ncb, my_pcol, np_cols, nblk, -1) ! Local cols of b

      ! Block factor for matrix multiplications, must be a multiple of nblk

      if (na/np_rows<=256) then
         nblk_mult = (31/nblk+1)*nblk
      else
         nblk_mult = (63/nblk+1)*nblk
      endif

      allocate(aux_mat(l_rows,nblk_mult), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_mult_at_b_real: error when allocating aux_mat "//errorMessage
        stop
      endif

      allocate(aux_bc(l_rows*nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_mult_at_b_real: error when allocating aux_bc "//errorMessage
        stop
      endif

      allocate(lrs_save(nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_mult_at_b_real: error when allocating lrs_save "//errorMessage
        stop
      endif

      allocate(lre_save(nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_mult_at_b_real: error when allocating lre_save "//errorMessage
        stop
      endif

      a_lower = .false.
      a_upper = .false.
      c_lower = .false.
      c_upper = .false.

      if (uplo_a=='u' .or. uplo_a=='U') a_upper = .true.
      if (uplo_a=='l' .or. uplo_a=='L') a_lower = .true.
      if (uplo_c=='u' .or. uplo_c=='U') c_upper = .true.
      if (uplo_c=='l' .or. uplo_c=='L') c_lower = .true.

      ! Build up the result matrix by processor rows

      do np = 0, np_rows-1

        ! In this turn, procs of row np assemble the result

        l_rows_np = local_index(na, np, np_rows, nblk, -1) ! local rows on receiving processors

        nr_done = 0 ! Number of rows done
        aux_mat = 0
        nstor = 0   ! Number of columns stored in aux_mat

        ! Loop over the blocks on row np

        do nb=0,(l_rows_np-1)/nblk

          goff  = nb*np_rows + np ! Global offset in blocks corresponding to nb

          ! Get the processor column which owns this block (A is transposed, so we need the column)
          ! and the offset in blocks within this column.
          ! The corresponding block column in A is then broadcast to all for multiplication with B

          np_bc = MOD(goff,np_cols)
          noff = goff/np_cols
          n_aux_bc = 0

          ! Gather up the complete block column of A on the owner

          do n = 1, min(l_rows_np-nb*nblk,nblk) ! Loop over columns to be broadcast

            gcol = goff*nblk + n ! global column corresponding to n
            if (nstor==0 .and. n==1) gcol_min = gcol

            lrs = 1       ! 1st local row number for broadcast
            lre = l_rows  ! last local row number for broadcast
            if (a_lower) lrs = local_index(gcol, my_prow, np_rows, nblk, +1)
            if (a_upper) lre = local_index(gcol, my_prow, np_rows, nblk, -1)

            if (lrs<=lre) then
              nvals = lre-lrs+1
              if (my_pcol == np_bc) aux_bc(n_aux_bc+1:n_aux_bc+nvals) = a(lrs:lre,noff*nblk+n)
              n_aux_bc = n_aux_bc + nvals
            endif

            lrs_save(n) = lrs
            lre_save(n) = lre

          enddo

          ! Broadcast block column
#ifdef WITH_MPI
          call timer%start("mpi_communication")
#ifdef DOUBLE_PRECISION_REAL
          call MPI_Bcast(aux_bc, n_aux_bc, MPI_REAL8, np_bc, mpi_comm_cols, mpierr)
#else
          call MPI_Bcast(aux_bc, n_aux_bc, MPI_REAL4, np_bc, mpi_comm_cols, mpierr)
#endif
          call timer%stop("mpi_communication")
#endif /* WITH_MPI */
          ! Insert what we got in aux_mat

          n_aux_bc = 0
          do n = 1, min(l_rows_np-nb*nblk,nblk)
            nstor = nstor+1
            lrs = lrs_save(n)
            lre = lre_save(n)
            if (lrs<=lre) then
              nvals = lre-lrs+1
              aux_mat(lrs:lre,nstor) = aux_bc(n_aux_bc+1:n_aux_bc+nvals)
              n_aux_bc = n_aux_bc + nvals
            endif
          enddo

          ! If we got nblk_mult columns in aux_mat or this is the last block
          ! do the matrix multiplication

          if (nstor==nblk_mult .or. nb*nblk+nblk >= l_rows_np) then

            lrs = 1       ! 1st local row number for multiply
            lre = l_rows  ! last local row number for multiply
            if (a_lower) lrs = local_index(gcol_min, my_prow, np_rows, nblk, +1)
            if (a_upper) lre = local_index(gcol, my_prow, np_rows, nblk, -1)

            lcs = 1       ! 1st local col number for multiply
            lce = l_cols  ! last local col number for multiply
            if (c_upper) lcs = local_index(gcol_min, my_pcol, np_cols, nblk, +1)
            if (c_lower) lce = MIN(local_index(gcol, my_pcol, np_cols, nblk, -1),l_cols)

            if (lcs<=lce) then
              allocate(tmp1(nstor,lcs:lce),tmp2(nstor,lcs:lce), stat=istat, errmsg=errorMessage)
              if (istat .ne. 0) then
               print *,"elpa_mult_at_b_real: error when allocating tmp1 "//errorMessage
               stop
              endif

              if (lrs<=lre) then
                call timer%start("blas")
#ifdef DOUBLE_PRECISION_REAL
                call dgemm('T', 'N', nstor, lce-lcs+1, lre-lrs+1, 1.0_rk8, aux_mat(lrs,1), ubound(aux_mat,dim=1), &
                             b(lrs,lcs), ldb, 0.0_rk8, tmp1, nstor)
#else
                call sgemm('T', 'N', nstor, lce-lcs+1, lre-lrs+1, 1.0_rk4, aux_mat(lrs,1), ubound(aux_mat,dim=1), &
                             b(lrs,lcs), ldb, 0.0_rk4, tmp1, nstor)
#endif
                call timer%stop("blas")
              else
                tmp1 = 0
              endif

              ! Sum up the results and send to processor row np
#ifdef WITH_MPI
              call timer%start("mpi_communication")
#ifdef DOUBLE_PRECISION_REAL
              call mpi_reduce(tmp1, tmp2, nstor*(lce-lcs+1), MPI_REAL8, MPI_SUM, np, mpi_comm_rows, mpierr)
#else
              call mpi_reduce(tmp1, tmp2, nstor*(lce-lcs+1), MPI_REAL4, MPI_SUM, np, mpi_comm_rows, mpierr)
#endif
              call timer%stop("mpi_communication")
              ! Put the result into C
              if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp2(1:nstor,lcs:lce)

#else /* WITH_MPI */
!              tmp2 = tmp1
              ! Put the result into C
              if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp1(1:nstor,lcs:lce)

#endif /* WITH_MPI */

              deallocate(tmp1,tmp2, stat=istat, errmsg=errorMessage)
              if (istat .ne. 0) then
               print *,"elpa_mult_at_b_real: error when deallocating tmp1 "//errorMessage
               stop
              endif

            endif

            nr_done = nr_done+nstor
            nstor=0
            aux_mat(:,:)=0
          endif
        enddo
      enddo

      deallocate(aux_mat, aux_bc, lrs_save, lre_save, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"elpa_mult_at_b_real: error when deallocating aux_mat "//errorMessage
       stop
      endif

#ifdef DOUBLE_PRECISION_REAL
      call timer%stop("elpa_mult_at_b_real_double")
#else
      call timer%stop("elpa_mult_at_b_real_single")
#endif

    end function elpa_mult_at_b_real_double

#if WANT_SINGLE_PRECISION_REAL
#undef DOUBLE_PRECISION_REAL
#undef REAL_DATATYPE
#define REAL_DATATYPE rk4
!> \brief  elpa_mult_at_b_real_single: Performs C : = A**T * B
!>         where   A is a square matrix (na,na) which is optionally upper or lower triangular
!>                 B is a (na,ncb) matrix
!>                 C is a (na,ncb) matrix where optionally only the upper or lower
!>                   triangle may be computed
!> \details

!> \param  uplo_a               'U' if A is upper triangular
!>                              'L' if A is lower triangular
!>                              anything else if A is a full matrix
!>                              Please note: This pertains to the original A (as set in the calling program)
!>                                           whereas the transpose of A is used for calculations
!>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!>                              i.e. it may contain arbitrary numbers
!> \param uplo_c                'U' if only the upper diagonal part of C is needed
!>                              'L' if only the upper diagonal part of C is needed
!>                              anything else if the full matrix C is needed
!>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!> \param na                    Number of rows/columns of A, number of rows of B and C
!> \param ncb                   Number of columns  of B and C
!> \param a                     matrix a
!> \param lda                   leading dimension of matrix a
!> \param b                     matrix b
!> \param ldb                   leading dimension of matrix b
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param c                     matrix c
!> \param ldc                   leading dimension of matrix c
!> \result success

    function elpa_mult_at_b_real_single(uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, nblk, &
                              mpi_comm_rows, mpi_comm_cols, c, ldc, ldcCols) result(success)
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#else
      use timings_dummy
#endif
      use elpa1_compute
      use elpa_mpi
      use precision
      implicit none

      character*1                   :: uplo_a, uplo_c

      integer(kind=ik), intent(in)  :: na, lda, ldaCols, ldb, ldbCols, ldc, ldcCols, nblk
      integer(kind=ik)              :: ncb, mpi_comm_rows, mpi_comm_cols
      real(kind=REAL_DATATYPE)                 :: a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)

      integer(kind=ik)              :: my_prow, my_pcol, np_rows, np_cols, mpierr
      integer(kind=ik)              :: l_cols, l_rows, l_rows_np
      integer(kind=ik)              :: np, n, nb, nblk_mult, lrs, lre, lcs, lce
      integer(kind=ik)              :: gcol_min, gcol, goff
      integer(kind=ik)              :: nstor, nr_done, noff, np_bc, n_aux_bc, nvals
      integer(kind=ik), allocatable :: lrs_save(:), lre_save(:)

      logical                       :: a_lower, a_upper, c_lower, c_upper

      real(kind=REAL_DATATYPE), allocatable    :: aux_mat(:,:), aux_bc(:), tmp1(:,:), tmp2(:,:)
      integer(kind=ik)              :: istat
      character(200)                :: errorMessage
      logical                       :: success

#ifdef DOUBLE_PRECISION_REAL
      call timer%start("elpa_mult_at_b_real_double")
#else
      call timer%start("elpa_mult_at_b_real_single")
#endif
      success = .true.

!      if (na .lt. lda) then
!        print *,"na lt lda ",na,lda
!        stop
!      endif
!      if (na .lt. ldb) then
!        print *,"na lt ldb ",na,ldb
!        stop
!      endif
!      if (na .lt. ldc) then
!        print *,"na lt ldc ",na,ldc
!        stop
!      endif
!      if (na .lt. ldaCols) then
!        print *,"na lt ldaCols ",na,ldaCols
!        stop
!      endif
!      if (na .lt. ldbCols) then
!        print *,"na lt ldbCols ",na,ldbCols
!        stop
!      endif
!      if (na .lt. ldcCols) then
!        print *,"na lt ldcCols ",na,ldcCols
!        stop
!      endif
      call timer%start("mpi_communication")
      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
      call timer%stop("mpi_communication")
      l_rows = local_index(na,  my_prow, np_rows, nblk, -1) ! Local rows of a and b
      l_cols = local_index(ncb, my_pcol, np_cols, nblk, -1) ! Local cols of b

      ! Block factor for matrix multiplications, must be a multiple of nblk

      if (na/np_rows<=256) then
         nblk_mult = (31/nblk+1)*nblk
      else
         nblk_mult = (63/nblk+1)*nblk
      endif

      allocate(aux_mat(l_rows,nblk_mult), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_mult_at_b_real: error when allocating aux_mat "//errorMessage
        stop
      endif

      allocate(aux_bc(l_rows*nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_mult_at_b_real: error when allocating aux_bc "//errorMessage
        stop
      endif

      allocate(lrs_save(nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_mult_at_b_real: error when allocating lrs_save "//errorMessage
        stop
      endif

      allocate(lre_save(nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_mult_at_b_real: error when allocating lre_save "//errorMessage
        stop
      endif

      a_lower = .false.
      a_upper = .false.
      c_lower = .false.
      c_upper = .false.

      if (uplo_a=='u' .or. uplo_a=='U') a_upper = .true.
      if (uplo_a=='l' .or. uplo_a=='L') a_lower = .true.
      if (uplo_c=='u' .or. uplo_c=='U') c_upper = .true.
      if (uplo_c=='l' .or. uplo_c=='L') c_lower = .true.

      ! Build up the result matrix by processor rows

      do np = 0, np_rows-1

        ! In this turn, procs of row np assemble the result

        l_rows_np = local_index(na, np, np_rows, nblk, -1) ! local rows on receiving processors

        nr_done = 0 ! Number of rows done
        aux_mat = 0
        nstor = 0   ! Number of columns stored in aux_mat

        ! Loop over the blocks on row np

        do nb=0,(l_rows_np-1)/nblk

          goff  = nb*np_rows + np ! Global offset in blocks corresponding to nb

          ! Get the processor column which owns this block (A is transposed, so we need the column)
          ! and the offset in blocks within this column.
          ! The corresponding block column in A is then broadcast to all for multiplication with B

          np_bc = MOD(goff,np_cols)
          noff = goff/np_cols
          n_aux_bc = 0

          ! Gather up the complete block column of A on the owner

          do n = 1, min(l_rows_np-nb*nblk,nblk) ! Loop over columns to be broadcast

            gcol = goff*nblk + n ! global column corresponding to n
            if (nstor==0 .and. n==1) gcol_min = gcol

            lrs = 1       ! 1st local row number for broadcast
            lre = l_rows  ! last local row number for broadcast
            if (a_lower) lrs = local_index(gcol, my_prow, np_rows, nblk, +1)
            if (a_upper) lre = local_index(gcol, my_prow, np_rows, nblk, -1)

            if (lrs<=lre) then
              nvals = lre-lrs+1
              if (my_pcol == np_bc) aux_bc(n_aux_bc+1:n_aux_bc+nvals) = a(lrs:lre,noff*nblk+n)
              n_aux_bc = n_aux_bc + nvals
            endif

            lrs_save(n) = lrs
            lre_save(n) = lre

          enddo

          ! Broadcast block column
#ifdef WITH_MPI
          call timer%start("mpi_communication")
#ifdef DOUBLE_PRECISION_REAL
          call MPI_Bcast(aux_bc, n_aux_bc, MPI_REAL8, np_bc, mpi_comm_cols, mpierr)
#else
          call MPI_Bcast(aux_bc, n_aux_bc, MPI_REAL4, np_bc, mpi_comm_cols, mpierr)
#endif
          call timer%stop("mpi_communication")
#endif /* WITH_MPI */
          ! Insert what we got in aux_mat

          n_aux_bc = 0
          do n = 1, min(l_rows_np-nb*nblk,nblk)
            nstor = nstor+1
            lrs = lrs_save(n)
            lre = lre_save(n)
            if (lrs<=lre) then
              nvals = lre-lrs+1
              aux_mat(lrs:lre,nstor) = aux_bc(n_aux_bc+1:n_aux_bc+nvals)
              n_aux_bc = n_aux_bc + nvals
            endif
          enddo

          ! If we got nblk_mult columns in aux_mat or this is the last block
          ! do the matrix multiplication

          if (nstor==nblk_mult .or. nb*nblk+nblk >= l_rows_np) then

            lrs = 1       ! 1st local row number for multiply
            lre = l_rows  ! last local row number for multiply
            if (a_lower) lrs = local_index(gcol_min, my_prow, np_rows, nblk, +1)
            if (a_upper) lre = local_index(gcol, my_prow, np_rows, nblk, -1)

            lcs = 1       ! 1st local col number for multiply
            lce = l_cols  ! last local col number for multiply
            if (c_upper) lcs = local_index(gcol_min, my_pcol, np_cols, nblk, +1)
            if (c_lower) lce = MIN(local_index(gcol, my_pcol, np_cols, nblk, -1),l_cols)

            if (lcs<=lce) then
              allocate(tmp1(nstor,lcs:lce),tmp2(nstor,lcs:lce), stat=istat, errmsg=errorMessage)
              if (istat .ne. 0) then
               print *,"elpa_mult_at_b_real: error when allocating tmp1 "//errorMessage
               stop
              endif

              if (lrs <= lre) then
                call timer%start("blas")
#ifdef DOUBLE_PRECISION_REAL
                call dgemm('T', 'N', nstor, lce-lcs+1, lre-lrs+1, 1.0_rk8, aux_mat(lrs,1), ubound(aux_mat,dim=1), &
                             b(lrs,lcs), ldb, 0.0_rk8, tmp1, nstor)
#else
                call sgemm('T', 'N', nstor, lce-lcs+1, lre-lrs+1, 1.0_rk4, aux_mat(lrs,1), ubound(aux_mat,dim=1), &
                             b(lrs,lcs), ldb, 0.0_rk4, tmp1, nstor)
#endif
                call timer%stop("blas")
              else
                tmp1 = 0
              endif

              ! Sum up the results and send to processor row np
#ifdef WITH_MPI
              call timer%start("mpi_communication")
#ifdef DOUBLE_PRECISION_REAL
              call mpi_reduce(tmp1, tmp2, nstor*(lce-lcs+1), MPI_REAL8, MPI_SUM, np, mpi_comm_rows, mpierr)
#else
              call mpi_reduce(tmp1, tmp2, nstor*(lce-lcs+1), MPI_REAL4, MPI_SUM, np, mpi_comm_rows, mpierr)
#endif
              call timer%stop("mpi_communication")
              ! Put the result into C
              if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp2(1:nstor,lcs:lce)

#else /* WITH_MPI */
!              tmp2 = tmp1
              ! Put the result into C
              if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp1(1:nstor,lcs:lce)

#endif /* WITH_MPI */

              deallocate(tmp1,tmp2, stat=istat, errmsg=errorMessage)
              if (istat .ne. 0) then
               print *,"elpa_mult_at_b_real: error when deallocating tmp1 "//errorMessage
               stop
              endif

            endif

            nr_done = nr_done+nstor
            nstor=0
            aux_mat(:,:)=0
          endif
        enddo
      enddo

      deallocate(aux_mat, aux_bc, lrs_save, lre_save, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"elpa_mult_at_b_real: error when deallocating aux_mat "//errorMessage
       stop
      endif

#ifdef DOUBLE_PRECISION_REAL
      call timer%stop("elpa_mult_at_b_real_double")
#else
      call timer%stop("elpa_mult_at_b_real_single")
#endif

    end function elpa_mult_at_b_real_single

#endif /* WANT_SINGLE_PRECISION_REAL */


#undef DOUBLE_PRECISION_COMPLEX
#undef COMPLEX_DATATYPE
#define DOUBLE_PRECISION_COMPLEX 1
#define COMPLEX_DATATYPE CK8
!> \brief  elpa_mult_ah_b_complex_double: Performs C : = A**H * B
!>         where   A is a square matrix (na,na) which is optionally upper or lower triangular
!>                 B is a (na,ncb) matrix
!>                 C is a (na,ncb) matrix where optionally only the upper or lower
!>                   triangle may be computed
!> \details
!>
!> \param  uplo_a               'U' if A is upper triangular
!>                              'L' if A is lower triangular
!>                              anything else if A is a full matrix
!>                              Please note: This pertains to the original A (as set in the calling program)
!>                                           whereas the transpose of A is used for calculations
!>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!>                              i.e. it may contain arbitrary numbers
!> \param uplo_c                'U' if only the upper diagonal part of C is needed
!>                              'L' if only the upper diagonal part of C is needed
!>                              anything else if the full matrix C is needed
!>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!> \param na                    Number of rows/columns of A, number of rows of B and C
!> \param ncb                   Number of columns  of B and C
!> \param a                     matrix a
!> \param lda                   leading dimension of matrix a
!> \param ldaCols               columns of matrix a
!> \param b                     matrix b
!> \param ldb                   leading dimension of matrix b
!> \param ldbCols               columns of matrix b
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param c                     matrix c
!> \param ldc                   leading dimension of matrix c
!> \result success

    function elpa_mult_ah_b_complex_double(uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, nblk, &
                                 mpi_comm_rows, mpi_comm_cols, c, ldc, ldcCols) result(success)
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#else
      use timings_dummy
#endif
      use precision
      use elpa1_compute
      use elpa_mpi

      implicit none

      character*1                   :: uplo_a, uplo_c
      integer(kind=ik), intent(in)  :: lda, ldaCols, ldb, ldbCols, ldc, ldcCols
      integer(kind=ik)              :: na, ncb, nblk, mpi_comm_rows, mpi_comm_cols
#ifdef USE_ASSUMED_SIZE
      complex(kind=COMPLEX_DATATYPE)::  a(lda,*), b(ldb,*), c(ldc,*)
#else
      complex(kind=COMPLEX_DATATYPE):: a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)
#endif
      integer(kind=ik)              :: my_prow, my_pcol, np_rows, np_cols, mpierr
      integer(kind=ik)              :: l_cols, l_rows, l_rows_np
      integer(kind=ik)              :: np, n, nb, nblk_mult, lrs, lre, lcs, lce
      integer(kind=ik)              :: gcol_min, gcol, goff
      integer(kind=ik)              :: nstor, nr_done, noff, np_bc, n_aux_bc, nvals
      integer(kind=ik), allocatable :: lrs_save(:), lre_save(:)

      logical                       :: a_lower, a_upper, c_lower, c_upper

      complex(kind=COMPLEX_DATATYPE), allocatable :: aux_mat(:,:), aux_bc(:), tmp1(:,:), tmp2(:,:)
      integer(kind=ik)              :: istat
      character(200)                :: errorMessage
      logical                       :: success

#ifdef DOUBLE_PRECISION_COMPLEX
      call timer%start("elpa_mult_ah_b_complex_double")
#else
      call timer%start("elpa_mult_ah_b_complex_single")
#endif

      success = .true.

!      if (na .lt. lda) then
!        print *,"na lt lda ",na,lda
!        stop
!      endif
!      if (na .lt. ldb) then
!        print *,"na lt ldb ",na,ldb
!        stop
!      endif
!      if (na .lt. ldc) then
!        print *,"na lt ldc ",na,ldc
!        stop
!      endif
!      if (na .lt. ldaCols) then
!        print *,"na lt ldaCols ",na,ldaCols
!        stop
!      endif
!      if (na .lt. ldbCols) then
!        print *,"na lt ldbCols ",na,ldbCols
!        stop
!      endif
!      if (na .lt. ldcCols) then
!        print *,"na lt ldcCols ",na,ldcCols
!        stop
!      endif

      call timer%start("mpi_communication")
      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

      call timer%stop("mpi_communication")
      l_rows = local_index(na,  my_prow, np_rows, nblk, -1) ! Local rows of a and b
      l_cols = local_index(ncb, my_pcol, np_cols, nblk, -1) ! Local cols of b

      ! Block factor for matrix multiplications, must be a multiple of nblk

      if (na/np_rows<=256) then
        nblk_mult = (31/nblk+1)*nblk
      else
        nblk_mult = (63/nblk+1)*nblk
      endif

      allocate(aux_mat(l_rows,nblk_mult), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"elpa_mult_ah_b_complex: error when allocating aux_mat "//errorMessage
       stop
      endif

      allocate(aux_bc(l_rows*nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"elpa_mult_ah_b_complex: error when allocating aux_bc "//errorMessage
       stop
      endif

      allocate(lrs_save(nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"elpa_mult_ah_b_complex: error when allocating lrs_save "//errorMessage
       stop
      endif

      allocate(lre_save(nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"elpa_mult_ah_b_complex: error when allocating lre_save "//errorMessage
       stop
      endif

      a_lower = .false.
      a_upper = .false.
      c_lower = .false.
      c_upper = .false.

      if (uplo_a=='u' .or. uplo_a=='U') a_upper = .true.
      if (uplo_a=='l' .or. uplo_a=='L') a_lower = .true.
      if (uplo_c=='u' .or. uplo_c=='U') c_upper = .true.
      if (uplo_c=='l' .or. uplo_c=='L') c_lower = .true.

      ! Build up the result matrix by processor rows

      do np = 0, np_rows-1

        ! In this turn, procs of row np assemble the result

        l_rows_np = local_index(na, np, np_rows, nblk, -1) ! local rows on receiving processors

        nr_done = 0 ! Number of rows done
        aux_mat = 0
        nstor = 0   ! Number of columns stored in aux_mat

        ! Loop over the blocks on row np

        do nb=0,(l_rows_np-1)/nblk

          goff  = nb*np_rows + np ! Global offset in blocks corresponding to nb

          ! Get the processor column which owns this block (A is transposed, so we need the column)
          ! and the offset in blocks within this column.
          ! The corresponding block column in A is then broadcast to all for multiplication with B

          np_bc = MOD(goff,np_cols)
          noff = goff/np_cols
          n_aux_bc = 0

          ! Gather up the complete block column of A on the owner

          do n = 1, min(l_rows_np-nb*nblk,nblk) ! Loop over columns to be broadcast

            gcol = goff*nblk + n ! global column corresponding to n
            if (nstor==0 .and. n==1) gcol_min = gcol

            lrs = 1       ! 1st local row number for broadcast
            lre = l_rows  ! last local row number for broadcast
            if (a_lower) lrs = local_index(gcol, my_prow, np_rows, nblk, +1)
            if (a_upper) lre = local_index(gcol, my_prow, np_rows, nblk, -1)

            if (lrs <= lre) then
              nvals = lre-lrs+1
              if (my_pcol == np_bc) aux_bc(n_aux_bc+1:n_aux_bc+nvals) = a(lrs:lre,noff*nblk+n)
              n_aux_bc = n_aux_bc + nvals
            endif

            lrs_save(n) = lrs
            lre_save(n) = lre

          enddo

          ! Broadcast block column
#ifdef WITH_MPI
          call timer%start("mpi_communication")
#ifdef DOUBLE_PRECISION_COMPLEX
          call MPI_Bcast(aux_bc, n_aux_bc, MPI_DOUBLE_COMPLEX, np_bc, mpi_comm_cols, mpierr)
#else
          call MPI_Bcast(aux_bc, n_aux_bc, MPI_COMPLEX, np_bc, mpi_comm_cols, mpierr)
#endif
          call timer%stop("mpi_communication")
#endif /* WITH_MPI */
          ! Insert what we got in aux_mat

          n_aux_bc = 0
          do n = 1, min(l_rows_np-nb*nblk,nblk)
            nstor = nstor+1
            lrs = lrs_save(n)
            lre = lre_save(n)
            if (lrs<=lre) then
              nvals = lre-lrs+1
              aux_mat(lrs:lre,nstor) = aux_bc(n_aux_bc+1:n_aux_bc+nvals)
              n_aux_bc = n_aux_bc + nvals
            endif
          enddo

          ! If we got nblk_mult columns in aux_mat or this is the last block
          ! do the matrix multiplication

          if (nstor==nblk_mult .or. nb*nblk+nblk >= l_rows_np) then

            lrs = 1       ! 1st local row number for multiply
            lre = l_rows  ! last local row number for multiply
            if (a_lower) lrs = local_index(gcol_min, my_prow, np_rows, nblk, +1)
            if (a_upper) lre = local_index(gcol, my_prow, np_rows, nblk, -1)

            lcs = 1       ! 1st local col number for multiply
            lce = l_cols  ! last local col number for multiply
            if (c_upper) lcs = local_index(gcol_min, my_pcol, np_cols, nblk, +1)
            if (c_lower) lce = MIN(local_index(gcol, my_pcol, np_cols, nblk, -1),l_cols)

            if (lcs<=lce) then
              allocate(tmp1(nstor,lcs:lce),tmp2(nstor,lcs:lce), stat=istat, errmsg=errorMessage)
              if (istat .ne. 0) then
                print *,"elpa_mult_ah_b_complex: error when allocating tmp1 "//errorMessage
                stop
              endif

              if (lrs <= lre) then
                call timer%start("blas")
#ifdef DOUBLE_PRECISION_COMPLEX
                call zgemm('C', 'N', nstor, lce-lcs+1, lre-lrs+1, (1.0_rk8,0.0_rk8), aux_mat(lrs,1), ubound(aux_mat,dim=1), &
                             b(lrs,lcs), ldb, (0.0_rk8,0.0_rk8), tmp1, nstor)
#else
                call cgemm('C', 'N', nstor, lce-lcs+1, lre-lrs+1, (1.0_rk4,0.0_rk4), aux_mat(lrs,1), ubound(aux_mat,dim=1), &
                             b(lrs,lcs), ldb, (0.0_rk4,0.0_rk4), tmp1, nstor)
#endif
                call timer%stop("blas")
              else
                tmp1 = 0
              endif

               ! Sum up the results and send to processor row np
#ifdef WITH_MPI
               call timer%start("mpi_communication")
#ifdef DOUBLE_PRECISION_COMPLEX
               call mpi_reduce(tmp1, tmp2, nstor*(lce-lcs+1), MPI_DOUBLE_COMPLEX, MPI_SUM, np, mpi_comm_rows, mpierr)

#else
               call mpi_reduce(tmp1, tmp2, nstor*(lce-lcs+1), MPI_COMPLEX, MPI_SUM, np, mpi_comm_rows, mpierr)
#endif
               call timer%stop("mpi_communication")
               ! Put the result into C
               if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp2(1:nstor,lcs:lce)

#else /* WITH_MPI */

!               tmp2 = tmp1

               ! Put the result into C
               if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp1(1:nstor,lcs:lce)

#endif /* WITH_MPI */

!               ! Put the result into C
!               if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp2(1:nstor,lcs:lce)

               deallocate(tmp1,tmp2, stat=istat, errmsg=errorMessage)
               if (istat .ne. 0) then
                 print *,"elpa_mult_ah_b_complex: error when deallocating tmp1 "//errorMessage
                 stop
               endif

            endif

            nr_done = nr_done+nstor
            nstor=0
            aux_mat(:,:)=0
          endif
        enddo
      enddo

      deallocate(aux_mat, aux_bc, lrs_save, lre_save, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_mult_ah_b_complex: error when deallocating aux_mat "//errorMessage
        stop
      endif

#ifdef DOUBLE_PRECISION_COMPLEX
      call timer%stop("elpa_mult_ah_b_complex_double")
#else
      call timer%stop("elpa_mult_ah_b_complex_single")
#endif

    end function elpa_mult_ah_b_complex_double

#ifdef WANT_SINGLE_PRECISION_COMPLEX
#undef DOUBLE_PRECISION_COMPLEX
#undef COMPLEX_DATATYPE
#define COMPLEX_DATATYPE CK4

!> \brief  elpa_mult_ah_b_complex_single: Performs C : = A**H * B
!>         where   A is a square matrix (na,na) which is optionally upper or lower triangular
!>                 B is a (na,ncb) matrix
!>                 C is a (na,ncb) matrix where optionally only the upper or lower
!>                   triangle may be computed
!> \details
!>
!> \param  uplo_a               'U' if A is upper triangular
!>                              'L' if A is lower triangular
!>                              anything else if A is a full matrix
!>                              Please note: This pertains to the original A (as set in the calling program)
!>                                           whereas the transpose of A is used for calculations
!>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
!>                              i.e. it may contain arbitrary numbers
!> \param uplo_c                'U' if only the upper diagonal part of C is needed
!>                              'L' if only the upper diagonal part of C is needed
!>                              anything else if the full matrix C is needed
!>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
!>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
!> \param na                    Number of rows/columns of A, number of rows of B and C
!> \param ncb                   Number of columns  of B and C
!> \param a                     matrix a
!> \param lda                   leading dimension of matrix a
!> \param ldaCols               columns of matrix a
!> \param b                     matrix b
!> \param ldb                   leading dimension of matrix b
!> \param ldbCols               columns of matrix b
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param c                     matrix c
!> \param ldc                   leading dimension of matrix c
!> \result success

    function elpa_mult_ah_b_complex_single(uplo_a, uplo_c, na, ncb, a, lda, ldaCols, b, ldb, ldbCols, nblk, &
                                 mpi_comm_rows, mpi_comm_cols, c, ldc, ldcCols) result(success)
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#else
      use timings_dummy
#endif
      use precision
      use elpa1_compute
      use elpa_mpi

      implicit none

      character*1                   :: uplo_a, uplo_c
      integer(kind=ik), intent(in)  :: lda, ldaCols, ldb, ldbCols, ldc, ldcCols
      integer(kind=ik)              :: na, ncb, nblk, mpi_comm_rows, mpi_comm_cols
#ifdef USE_ASSUMED_SIZE
      complex(kind=COMPLEX_DATATYPE):: a(lda,*), b(ldb,*), c(ldc,*)
#else
      complex(kind=COMPLEX_DATATYPE):: a(lda,ldaCols), b(ldb,ldbCols), c(ldc,ldcCols)
#endif

      integer(kind=ik)              :: my_prow, my_pcol, np_rows, np_cols, mpierr
      integer(kind=ik)              :: l_cols, l_rows, l_rows_np
      integer(kind=ik)              :: np, n, nb, nblk_mult, lrs, lre, lcs, lce
      integer(kind=ik)              :: gcol_min, gcol, goff
      integer(kind=ik)              :: nstor, nr_done, noff, np_bc, n_aux_bc, nvals
      integer(kind=ik), allocatable :: lrs_save(:), lre_save(:)

      logical                       :: a_lower, a_upper, c_lower, c_upper

      complex(kind=COMPLEX_DATATYPE), allocatable :: aux_mat(:,:), aux_bc(:), tmp1(:,:), tmp2(:,:)
      integer(kind=ik)              :: istat
      character(200)                :: errorMessage
      logical                       :: success

#ifdef DOUBLE_PRECISION_COMPLEX
      call timer%start("elpa_mult_ah_b_complex_double")
#else
      call timer%start("elpa_mult_ah_b_complex_single")
#endif

      success = .true.

!      if (na .lt. lda) then
!        print *,"na lt lda ",na,lda
!        stop
!      endif
!      if (na .lt. ldb) then
!        print *,"na lt ldb ",na,ldb
!        stop
!      endif
!      if (na .lt. ldc) then
!        print *,"na lt ldc ",na,ldc
!        stop
!      endif
!      if (na .lt. ldaCols) then
!        print *,"na lt ldaCols ",na,ldaCols
!        stop
!      endif
!      if (na .lt. ldbCols) then
!        print *,"na lt ldbCols ",na,ldbCols
!        stop
!      endif
!      if (na .lt. ldcCols) then
!        print *,"na lt ldcCols ",na,ldcCols
!        stop
!      endif

      call timer%start("mpi_communication")
      call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
      call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
      call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
      call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
      call timer%stop("mpi_communication")
      l_rows = local_index(na,  my_prow, np_rows, nblk, -1) ! Local rows of a and b
      l_cols = local_index(ncb, my_pcol, np_cols, nblk, -1) ! Local cols of b

      ! Block factor for matrix multiplications, must be a multiple of nblk

      if (na/np_rows<=256) then
        nblk_mult = (31/nblk+1)*nblk
      else
        nblk_mult = (63/nblk+1)*nblk
      endif

      allocate(aux_mat(l_rows,nblk_mult), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"elpa_mult_ah_b_complex: error when allocating aux_mat "//errorMessage
       stop
      endif

      allocate(aux_bc(l_rows*nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"elpa_mult_ah_b_complex: error when allocating aux_bc "//errorMessage
       stop
      endif

      allocate(lrs_save(nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"elpa_mult_ah_b_complex: error when allocating lrs_save "//errorMessage
       stop
      endif

      allocate(lre_save(nblk), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
       print *,"elpa_mult_ah_b_complex: error when allocating lre_save "//errorMessage
       stop
      endif

      a_lower = .false.
      a_upper = .false.
      c_lower = .false.
      c_upper = .false.

      if (uplo_a=='u' .or. uplo_a=='U') a_upper = .true.
      if (uplo_a=='l' .or. uplo_a=='L') a_lower = .true.
      if (uplo_c=='u' .or. uplo_c=='U') c_upper = .true.
      if (uplo_c=='l' .or. uplo_c=='L') c_lower = .true.

      ! Build up the result matrix by processor rows

      do np = 0, np_rows-1

        ! In this turn, procs of row np assemble the result

        l_rows_np = local_index(na, np, np_rows, nblk, -1) ! local rows on receiving processors

        nr_done = 0 ! Number of rows done
        aux_mat = 0
        nstor = 0   ! Number of columns stored in aux_mat

        ! Loop over the blocks on row np

        do nb=0,(l_rows_np-1)/nblk

          goff  = nb*np_rows + np ! Global offset in blocks corresponding to nb

          ! Get the processor column which owns this block (A is transposed, so we need the column)
          ! and the offset in blocks within this column.
          ! The corresponding block column in A is then broadcast to all for multiplication with B

          np_bc = MOD(goff,np_cols)
          noff = goff/np_cols
          n_aux_bc = 0

          ! Gather up the complete block column of A on the owner

          do n = 1, min(l_rows_np-nb*nblk,nblk) ! Loop over columns to be broadcast

            gcol = goff*nblk + n ! global column corresponding to n
            if (nstor==0 .and. n==1) gcol_min = gcol

            lrs = 1       ! 1st local row number for broadcast
            lre = l_rows  ! last local row number for broadcast
            if (a_lower) lrs = local_index(gcol, my_prow, np_rows, nblk, +1)
            if (a_upper) lre = local_index(gcol, my_prow, np_rows, nblk, -1)

            if (lrs<=lre) then
              nvals = lre-lrs+1
              if (my_pcol == np_bc) aux_bc(n_aux_bc+1:n_aux_bc+nvals) = a(lrs:lre,noff*nblk+n)
              n_aux_bc = n_aux_bc + nvals
            endif

            lrs_save(n) = lrs
            lre_save(n) = lre

          enddo

          ! Broadcast block column
#ifdef WITH_MPI
          call timer%start("mpi_communication")
#ifdef DOUBLE_PRECISION_COMPLEX
          call MPI_Bcast(aux_bc, n_aux_bc, MPI_DOUBLE_COMPLEX, np_bc, mpi_comm_cols, mpierr)
#else
          call MPI_Bcast(aux_bc, n_aux_bc, MPI_COMPLEX, np_bc, mpi_comm_cols, mpierr)
#endif
          call timer%stop("mpi_communication")
#endif /* WITH_MPI */
          ! Insert what we got in aux_mat

          n_aux_bc = 0
          do n = 1, min(l_rows_np-nb*nblk,nblk)
            nstor = nstor+1
            lrs = lrs_save(n)
            lre = lre_save(n)
            if (lrs<=lre) then
              nvals = lre-lrs+1
              aux_mat(lrs:lre,nstor) = aux_bc(n_aux_bc+1:n_aux_bc+nvals)
              n_aux_bc = n_aux_bc + nvals
            endif
          enddo

          ! If we got nblk_mult columns in aux_mat or this is the last block
          ! do the matrix multiplication

          if (nstor==nblk_mult .or. nb*nblk+nblk >= l_rows_np) then

            lrs = 1       ! 1st local row number for multiply
            lre = l_rows  ! last local row number for multiply
            if (a_lower) lrs = local_index(gcol_min, my_prow, np_rows, nblk, +1)
            if (a_upper) lre = local_index(gcol, my_prow, np_rows, nblk, -1)

            lcs = 1       ! 1st local col number for multiply
            lce = l_cols  ! last local col number for multiply
            if (c_upper) lcs = local_index(gcol_min, my_pcol, np_cols, nblk, +1)
            if (c_lower) lce = MIN(local_index(gcol, my_pcol, np_cols, nblk, -1),l_cols)

            if (lcs<=lce) then
              allocate(tmp1(nstor,lcs:lce),tmp2(nstor,lcs:lce), stat=istat, errmsg=errorMessage)
              if (istat .ne. 0) then
                print *,"elpa_mult_ah_b_complex: error when allocating tmp1 "//errorMessage
                stop
              endif

              if (lrs<=lre) then
                call timer%start("blas")
#ifdef DOUBLE_PRECISION_COMPLEX
                call zgemm('C', 'N', nstor, lce-lcs+1, lre-lrs+1, (1.0_rk8,0.0_rk8), aux_mat(lrs,1), ubound(aux_mat,dim=1), &
                             b(lrs,lcs), ldb, (0.0_rk8,0.0_rk8), tmp1, nstor)
#else
                call cgemm('C', 'N', nstor, lce-lcs+1, lre-lrs+1, (1.0_rk4,0.0_rk4), aux_mat(lrs,1), ubound(aux_mat,dim=1), &
                             b(lrs,lcs), ldb, (0.0_rk4,0.0_rk4), tmp1, nstor)
#endif
                call timer%stop("blas")
              else
                tmp1 = 0
              endif

               ! Sum up the results and send to processor row np
#ifdef WITH_MPI
               call timer%start("mpi_communication")
#ifdef DOUBLE_PRECISION_COMPLEX
               call mpi_reduce(tmp1, tmp2, nstor*(lce-lcs+1), MPI_DOUBLE_COMPLEX, MPI_SUM, np, mpi_comm_rows, mpierr)
#else
               call mpi_reduce(tmp1, tmp2, nstor*(lce-lcs+1), MPI_COMPLEX, MPI_SUM, np, mpi_comm_rows, mpierr)
#endif
               call timer%stop("mpi_communication")
               ! Put the result into C
               if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp2(1:nstor,lcs:lce)

#else /* WITH_MPI */

!               tmp2 = tmp1

               ! Put the result into C
               if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp1(1:nstor,lcs:lce)

#endif /* WITH_MPI */

!               ! Put the result into C
!               if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp2(1:nstor,lcs:lce)

               deallocate(tmp1,tmp2, stat=istat, errmsg=errorMessage)
               if (istat .ne. 0) then
                 print *,"elpa_mult_ah_b_complex: error when deallocating tmp1 "//errorMessage
                 stop
               endif

            endif

            nr_done = nr_done+nstor
            nstor=0
            aux_mat(:,:)=0
          endif
        enddo
      enddo

      deallocate(aux_mat, aux_bc, lrs_save, lre_save, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"elpa_mult_ah_b_complex: error when deallocating aux_mat "//errorMessage
        stop
      endif

#ifdef DOUBLE_PRECISION_COMPLEX
      call timer%stop("elpa_mult_ah_b_complex_double")
#else
      call timer%stop("elpa_mult_ah_b_complex_single")
#endif

    end function elpa_mult_ah_b_complex_single

#endif /* WANT_SINGLE_PRECISION_COMPLEX */


!> \brief  elpa_solve_tridi_double: Solve tridiagonal eigensystem for a double-precision matrix with divide and conquer method
!> \details
!>
!> \param na                    Matrix dimension
!> \param nev                   number of eigenvalues/vectors to be computed
!> \param d                     array d(na) on input diagonal elements of tridiagonal matrix, on
!>                              output the eigenvalues in ascending order
!> \param e                     array e(na) on input subdiagonal elements of matrix, on exit destroyed
!> \param q                     on exit : matrix q(ldq,matrixCols) contains the eigenvectors
!> \param ldq                   leading dimension of matrix q
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param matrixCols            columns of matrix q
!> \param mpi_comm_rows         MPI communicator for rows
!> \param mpi_comm_cols         MPI communicator for columns
!> \param wantDebug             logical, give more debug information if .true.
!> \result success              logical, .true. on success, else .false.

    function elpa_solve_tridi_double(na, nev, d, e, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug) &
          result(success)

      use elpa1_compute, solve_tridi_double_private => solve_tridi_double
      use precision

      implicit none
      integer(kind=ik)       :: na, nev, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
      real(kind=rk8)         :: d(na), e(na)
#ifdef USE_ASSUMED_SIZE
      real(kind=rk8)         :: q(ldq,*)
#else
      real(kind=rk8)         :: q(ldq,matrixCols)
#endif
      logical, intent(in)    :: wantDebug
      logical :: success

      success = .false.

      call solve_tridi_double_private(na, nev, d, e, q, ldq, nblk, matrixCols, &
                                      mpi_comm_rows, mpi_comm_cols, wantDebug, success)

    end function


#ifdef WANT_SINGLE_PRECISION_REAL
!> \brief  elpa_solve_tridi_single: Solve tridiagonal eigensystem for a single-precision matrix with divide and conquer method
!> \details
!>
!> \param na                    Matrix dimension
!> \param nev                   number of eigenvalues/vectors to be computed
!> \param d                     array d(na) on input diagonal elements of tridiagonal matrix, on
!>                              output the eigenvalues in ascending order
!> \param e                     array e(na) on input subdiagonal elements of matrix, on exit destroyed
!> \param q                     on exit : matrix q(ldq,matrixCols) contains the eigenvectors
!> \param ldq                   leading dimension of matrix q
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param matrixCols            columns of matrix q
!> \param mpi_comm_rows         MPI communicator for rows
!> \param mpi_comm_cols         MPI communicator for columns
!> \param wantDebug             logical, give more debug information if .true.
!> \result success              logical, .true. on success, else .false.

    function elpa_solve_tridi_single(na, nev, d, e, q, ldq, nblk, matrixCols, mpi_comm_rows, &
                                     mpi_comm_cols, wantDebug) result(success)

      use elpa1_compute, solve_tridi_single_private => solve_tridi_single
      use precision

      implicit none
      integer(kind=ik)       :: na, nev, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
      real(kind=rk4)         :: d(na), e(na)
#ifdef USE_ASSUMED_SIZE
      real(kind=rk4)         :: q(ldq,*)
#else
      real(kind=rk4)         :: q(ldq,matrixCols)
#endif
      logical, intent(in)    :: wantDebug
      logical :: success

      success = .false.

      call solve_tridi_single_private(na, nev, d, e, q, ldq, nblk, matrixCols, &
                                      mpi_comm_rows, mpi_comm_cols, wantDebug, success)

    end function

#endif /* WANT_SINGLE_PRECISION_REAL */




end module elpa1_auxiliary

