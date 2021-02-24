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
!
! This file has been rewritten by A. Marek, MPCDF
#include "config-f90.h"

!> \brief Fortran module which provides helper routines for matrix calculations
module elpa1_auxiliary_impl
  use elpa_utilities

  implicit none

  public :: elpa_mult_at_b_real_double_impl      !< Multiply double-precision real matrices A**T * B

  public :: elpa_mult_ah_b_complex_double_impl   !< Multiply double-precision complex matrices A**H * B

  public :: elpa_invert_trm_real_double_impl    !< Invert double-precision real triangular matrix

  public :: elpa_invert_trm_complex_double_impl  !< Invert double-precision complex triangular matrix

  public :: elpa_cholesky_real_double_impl       !< Cholesky factorization of a double-precision real matrix

  public :: elpa_cholesky_complex_double_impl    !< Cholesky factorization of a double-precision complex matrix

  public :: elpa_solve_tridi_double_impl         !< Solve tridiagonal eigensystem for a double-precision matrix with divide and conquer method

#ifdef WANT_SINGLE_PRECISION_REAL
  public :: elpa_cholesky_real_single_impl       !< Cholesky factorization of a single-precision real matrix
  public :: elpa_invert_trm_real_single_impl     !< Invert single-precision real triangular matrix
  public :: elpa_mult_at_b_real_single_impl      !< Multiply single-precision real matrices A**T * B
  public :: elpa_solve_tridi_single_impl         !< Solve tridiagonal eigensystem for a single-precision matrix with divide and conquer method
#endif

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  public :: elpa_cholesky_complex_single_impl    !< Cholesky factorization of a single-precision complex matrix
  public :: elpa_invert_trm_complex_single_impl  !< Invert single-precision complex triangular matrix
  public :: elpa_mult_ah_b_complex_single_impl   !< Multiply single-precision complex matrices A**H * B
#endif

  contains

#define REALCASE 1
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"

   function elpa_cholesky_real_double_impl (obj, a) result(success)
#include "elpa_cholesky_template.F90"

    end function elpa_cholesky_real_double_impl
#undef DOUBLE_PRECISION
#undef REALCASE

#ifdef WANT_SINGLE_PRECISION_REAL
#define REALCASE 1
#define SINGLE_PRECISION
#include "../general/precision_macros.h"

   function elpa_cholesky_real_single_impl(obj, a) result(success)
#include "elpa_cholesky_template.F90"

    end function elpa_cholesky_real_single_impl
#undef SINGLE_PRECISION
#undef REALCASE

#endif /* WANT_SINGLE_PRECSION_REAL */

#define REALCASE 1
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"
!> \brief  elpa_invert_trm_real_double: Inverts a double-precision real upper triangular matrix
!> \details
!> \param  obj                    elpa_t object contains:
!> \param     - obj%na            Order of matrix
!> \param     - obj%local_nrows   Leading dimension of a
!> \param     - obj%local_ncols   local columns of matrix a
!> \param     - obj%nblk          blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows MPI communicator for rows
!> \param     - obj%mpi_comm_cols MPI communicator for columns
!> \param     - obj%wantDebug     logical, more debug information on failure
!> \param  a(lda,matrixCols)      Distributed matrix which should be inverted
!>                                Distribution is like in Scalapack.
!>                                Only upper triangle needs to be set.
!>                                The lower triangle is not referenced.
!> \result succes                 logical, reports success or failure
    function elpa_invert_trm_real_double_impl(obj, a) result(success)
#include "elpa_invert_trm.F90"
     end function elpa_invert_trm_real_double_impl
#undef DOUBLE_PRECISION
#undef REALCASE

#ifdef WANT_SINGLE_PRECISION_REAL
#define REALCASE 1
#define SINGLE_PRECISION
#include "../general/precision_macros.h"

!> \brief  elpa_invert_trm_real_single_impl: Inverts a single-precision real upper triangular matrix
!> \details
!> \param  obj                    elpa_t object contains:
!> \param     - obj%na            Order of matrix
!> \param     - obj%local_nrows   Leading dimension of a
!> \param     - obj%local_ncols   local columns of matrix a
!> \param     - obj%nblk          blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows MPI communicator for rows
!> \param     - obj%mpi_comm_cols MPI communicator for columns
!> \param     - obj%wantDebug     logical, more debug information on failure
!> \param  a(lda,matrixCols)      Distributed matrix which should be inverted
!>                                Distribution is like in Scalapack.
!>                                Only upper triangle needs to be set.
!>                                The lower triangle is not referenced.
!> \result succes                 logical, reports success or failure

    function elpa_invert_trm_real_single_impl(obj, a) result(success)
#include "elpa_invert_trm.F90"
    end function elpa_invert_trm_real_single_impl
#undef SINGLE_PRECISION
#undef REALCASE

#endif /* WANT_SINGLE_PRECISION_REAL */


#define COMPLEXCASE 1
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"

!> \brief  elpa_cholesky_complex_double_impl: Cholesky factorization of a double-precision complex hermitian matrix
!> \details
!> \param  obj                    elpa_t object contains:
!> \param     - obj%na            Order of matrix
!> \param     - obj%local_nrows   Leading dimension of a
!> \param     - obj%local_ncols   local columns of matrix a
!> \param     - obj%nblk          blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows MPI communicator for rows
!> \param     - obj%mpi_comm_cols MPI communicator for columns
!> \param     - obj%wantDebug     logical, more debug information on failure
!> \param  a(lda,matrixCols)      Distributed matrix which should be inverted
!>                                Distribution is like in Scalapack.
!>                                Only upper triangle needs to be set.
!>                                The lower triangle is not referenced.
!> \result succes                 logical, reports success or failure
    function elpa_cholesky_complex_double_impl(obj, a) result(success)

#include "elpa_cholesky_template.F90"

    end function elpa_cholesky_complex_double_impl
#undef DOUBLE_PRECISION
#undef COMPLEXCASE

#ifdef WANT_SINGLE_PRECISION_COMPLEX
#define COMPLEXCASE 1
#define SINGLE_PRECISION
#include "../general/precision_macros.h"

!> \brief  elpa_cholesky_complex_single_impl: Cholesky factorization of a single-precision complex hermitian matrix
!> \details
!> \param  obj                    elpa_t object contains:
!> \param     - obj%na            Order of matrix
!> \param     - obj%local_nrows   Leading dimension of a
!> \param     - obj%local_ncols   local columns of matrix a
!> \param     - obj%nblk          blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows MPI communicator for rows
!> \param     - obj%mpi_comm_cols MPI communicator for columns
!> \param     - obj%wantDebug     logical, more debug information on failure
!> \param  a(lda,matrixCols)      Distributed matrix which should be inverted
!>                                Distribution is like in Scalapack.
!>                                Only upper triangle needs to be set.
!>                                The lower triangle is not referenced.
!> \result succes                 logical, reports success or failure
    function elpa_cholesky_complex_single_impl(obj, a) result(success)

#include "elpa_cholesky_template.F90"

    end function elpa_cholesky_complex_single_impl
#undef SINGLE_PRECISION
#undef COMPLEXCASE

#endif /* WANT_SINGLE_PRECISION_COMPLEX */

#define COMPLEXCASE 1
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"

!> \brief  elpa_invert_trm_complex_double_impl: Inverts a double-precision complex upper triangular matrix
!> \details
!> \param  obj                    elpa_t object contains:
!> \param     - obj%na            Order of matrix
!> \param     - obj%local_nrows   Leading dimension of a
!> \param     - obj%local_ncols   local columns of matrix a
!> \param     - obj%nblk          blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows MPI communicator for rows
!> \param     - obj%mpi_comm_cols MPI communicator for columns
!> \param     - obj%wantDebug     logical, more debug information on failure
!> \param  a(lda,matrixCols)      Distributed matrix which should be inverted
!>                                Distribution is like in Scalapack.
!>                                Only upper triangle needs to be set.
!>                                The lower triangle is not referenced.
!> \result succes                 logical, reports success or failure
     function elpa_invert_trm_complex_double_impl(obj, a) result(success)
#include "elpa_invert_trm.F90"
    end function elpa_invert_trm_complex_double_impl
#undef DOUBLE_PRECISION
#undef COMPLEXCASE

#ifdef WANT_SINGLE_PRECISION_COMPLEX
#define COMPLEXCASE 1
#define SINGLE_PRECISION
#include "../general/precision_macros.h"

!> \brief  elpa_invert_trm_complex_single_impl: Inverts a single-precision complex upper triangular matrix
!> \details
!> \param  obj                    elpa_t object contains:
!> \param     - obj%na            Order of matrix
!> \param     - obj%local_nrows   Leading dimension of a
!> \param     - obj%local_ncols   local columns of matrix a
!> \param     - obj%nblk          blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows MPI communicator for rows
!> \param     - obj%mpi_comm_cols MPI communicator for columns
!> \param     - obj%wantDebug     logical, more debug information on failure
!> \param  a(lda,matrixCols)      Distributed matrix which should be inverted
!>                                Distribution is like in Scalapack.
!>                                Only upper triangle needs to be set.
!>                                The lower triangle is not referenced.
!> \result succes                 logical, reports success or failure
    function elpa_invert_trm_complex_single_impl(obj, a) result(success)
#include "elpa_invert_trm.F90"
    end function elpa_invert_trm_complex_single_impl
#undef SINGLE_PRECISION
#undef COMPLEXCASE

#endif /* WANT_SINGE_PRECISION_COMPLEX */

#define REALCASE 1
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"
    function elpa_mult_at_b_real_double_impl(obj, uplo_a, uplo_c, ncb, a, b, ldb, ldbCols, &
                                             c, ldc, ldcCols) result(success)
#include "elpa_multiply_a_b.F90"
    end function elpa_mult_at_b_real_double_impl
#undef DOUBLE_PRECISION
#undef REALCASE

#ifdef WANT_SINGLE_PRECISION_REAL
#define REALCASE 1
#define SINGLE_PRECISION
#include "../general/precision_macros.h"

!> \brief  elpa_mult_at_b_real_single_impl: Performs C : = A**T * B
!>         where   A is a square matrix (obj%na,obj%na) which is optionally upper or lower triangular
!>                 B is a (obj%na,ncb) matrix
!>                 C is a (obj%na,ncb) matrix where optionally only the upper or lower
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
!> \param obj%local_nrows       leading dimension of matrix a, set with class method obj%set("local_nrows",value)
!> \param b                     matrix b
!> \param ldb                   leading dimension of matrix b
!> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
!> \param  mpi_comm_rows        MPI communicator for rows
!> \param  mpi_comm_cols        MPI communicator for columns
!> \param c                     matrix c
!> \param ldc                   leading dimension of matrix c
!> \result success

    function elpa_mult_at_b_real_single_impl(obj, uplo_a, uplo_c, ncb, a, b, ldb, ldbCols, &
                                             c, ldc, ldcCols) result(success)

#include "elpa_multiply_a_b.F90"

    end function elpa_mult_at_b_real_single_impl
#undef SINGLE_PRECISION
#undef REALCASE
#endif /* WANT_SINGLE_PRECISION_REAL */


#define COMPLEXCASE 1
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"

!> \brief  elpa_mult_ah_b_complex_double_impl: Performs C : = A**H * B
!>         where   A is a square matrix (obj%na,obj%na) which is optionally upper or lower triangular
!>                 B is a (obj%na,ncb) matrix
!>                 C is a (obj%na,ncb) matrix where optionally only the upper or lower
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
!> \param obj%local_ncols       leading dimension of matrix a, set with class method obj%set("local_nrows",value)
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

    function elpa_mult_ah_b_complex_double_impl(obj, uplo_a, uplo_c, ncb, a, b, ldb, ldbCols, &
                                                c, ldc, ldcCols) result(success)
#include "elpa_multiply_a_b.F90"

    end function elpa_mult_ah_b_complex_double_impl
#undef DOUBLE_PRECISION
#undef COMPLEXCASE

#ifdef WANT_SINGLE_PRECISION_COMPLEX
#define COMPLEXCASE 1
#define SINGLE_PRECISION
#include "../general/precision_macros.h"

!> \brief  elpa_mult_ah_b_complex_single_impl: Performs C : = A**H * B
!>         where   A is a square matrix (obj%na,obj%na) which is optionally upper or lower triangular
!>                 B is a (obj%na,ncb) matrix
!>                 C is a (obj%na,ncb) matrix where optionally only the upper or lower
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

    function elpa_mult_ah_b_complex_single_impl(obj, uplo_a, uplo_c, ncb, a, b, ldb, ldbCols, &
                                                c, ldc, ldcCols) result(success)

#include "elpa_multiply_a_b.F90"

    end function elpa_mult_ah_b_complex_single_impl
#undef SINGLE_PRECISION
#undef COMPLEXCASE
#endif /* WANT_SINGLE_PRECISION_COMPLEX */

#define REALCASE 1
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"

!> \brief  elpa_solve_tridi_double_impl: Solve tridiagonal eigensystem for a double-precision matrix with divide and conquer method
!> \details
!> \param  obj                    elpa_t object contains:
!> \param     - obj%na            Order of matrix
!> \param     - obj%nev           number of eigenvalues/vectors to be computed
!> \param     - obj%local_nrows   Leading dimension of q
!> \param     - obj%local_ncols   local columns of matrix q
!> \param     - obj%nblk          blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows MPI communicator for rows
!> \param     - obj%mpi_comm_cols MPI communicator for columns
!> \param     - obj%wantDebug     logical, more debug information on failure
!> \param d                       array d(na) on input diagonal elements of tridiagonal matrix, on
!>                                output the eigenvalues in ascending order
!> \param e                       array e(na) on input subdiagonal elements of matrix, on exit destroyed
!> \param q                       on exit : matrix q(ldq,matrixCols) contains the eigenvectors
!> \result succes                 logical, reports success or failure
    function elpa_solve_tridi_double_impl(obj, d, e, q) result(success)

#include "elpa_solve_tridi_impl_public.F90"

    end function
#undef DOUBLE_PRECISION
#undef REALCASE

#ifdef WANT_SINGLE_PRECISION_REAL
#define REALCASE 1
#define SINGLE_PRECISION
#include "../general/precision_macros.h"

!> \brief  elpa_solve_tridi_single_impl: Solve tridiagonal eigensystem for a single-precision matrix with divide and conquer method
!> \details
!> \param  obj                    elpa_t object contains:
!> \param     - obj%na            Order of matrix
!> \param     - obj%nev           number of eigenvalues/vectors to be computed
!> \param     - obj%local_nrows   Leading dimension of q
!> \param     - obj%local_ncols   local columns of matrix q
!> \param     - obj%nblk          blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows MPI communicator for rows
!> \param     - obj%mpi_comm_cols MPI communicator for columns
!> \param     - obj%wantDebug     logical, more debug information on failure
!> \param d                       array d(na) on input diagonal elements of tridiagonal matrix, on
!>                                output the eigenvalues in ascending order
!> \param e                       array e(na) on input subdiagonal elements of matrix, on exit destroyed
!> \param q                       on exit : matrix q(ldq,matrixCols) contains the eigenvectors
!> \result succes                 logical, reports success or failure
    function elpa_solve_tridi_single_impl(obj, d, e, q) result(success)

#include "elpa_solve_tridi_impl_public.F90"

    end function
#undef SINGLE_PRECISION
#undef REALCASE
#endif /* WANT_SINGLE_PRECISION_REAL */




end module elpa1_auxiliary_impl

