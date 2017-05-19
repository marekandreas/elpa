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
! Author: Andreas Marek, MCPDF
#include "config-f90.h"
  !lc> #include <complex.h>

  !lc> /*! \brief C old, deprecated interface, will be deleted. Use "elpa_get_communicators"
  !lc> *
  !lc> * \param mpi_comm_word    MPI global communicator (in)
  !lc> * \param my_prow          Row coordinate of the calling process in the process grid (in)
  !lc> * \param my_pcol          Column coordinate of the calling process in the process grid (in)
  !lc> * \param mpi_comm_rows    Communicator for communicating within rows of processes (out)
  !lc> * \result int             integer error value of mpi_comm_split function
  !lc> */
  !lc> int get_elpa_row_col_comms(int mpi_comm_world, int my_prow, int my_pcol, int *mpi_comm_rows, int *mpi_comm_cols);
  function get_elpa_row_col_comms_wrapper_c_name1(mpi_comm_world, my_prow, my_pcol, &
                                          mpi_comm_rows, mpi_comm_cols)     &
                                          result(mpierr) bind(C,name="get_elpa_row_col_comms")
    use, intrinsic :: iso_c_binding
    use elpa1, only : elpa_get_communicators

    implicit none
    integer(kind=c_int)         :: mpierr
    integer(kind=c_int), value  :: mpi_comm_world, my_prow, my_pcol
    integer(kind=c_int)         :: mpi_comm_rows, mpi_comm_cols

    mpierr = elpa_get_communicators(mpi_comm_world, my_prow, my_pcol, &
                                    mpi_comm_rows, mpi_comm_cols)

  end function
  !lc> #include <complex.h>

  !lc> /*! \brief C old, deprecated interface, will be deleted. Use "elpa_get_communicators"
  !lc> *
  !lc> * \param mpi_comm_word    MPI global communicator (in)
  !lc> * \param my_prow          Row coordinate of the calling process in the process grid (in)
  !lc> * \param my_pcol          Column coordinate of the calling process in the process grid (in)
  !lc> * \param mpi_comm_rows    Communicator for communicating within rows of processes (out)
  !lc> * \result int             integer error value of mpi_comm_split function
  !lc> */
  !lc> int get_elpa_communicators(int mpi_comm_world, int my_prow, int my_pcol, int *mpi_comm_rows, int *mpi_comm_cols);
  function get_elpa_row_col_comms_wrapper_c_name2(mpi_comm_world, my_prow, my_pcol, &
                                          mpi_comm_rows, mpi_comm_cols)     &
                                          result(mpierr) bind(C,name="get_elpa_communicators")
    use, intrinsic :: iso_c_binding
    use elpa1, only : elpa_get_communicators

    implicit none
    integer(kind=c_int)         :: mpierr
    integer(kind=c_int), value  :: mpi_comm_world, my_prow, my_pcol
    integer(kind=c_int)         :: mpi_comm_rows, mpi_comm_cols

    mpierr = elpa_get_communicators(mpi_comm_world, my_prow, my_pcol, &
                                    mpi_comm_rows, mpi_comm_cols)

  end function

  !lc> #include <complex.h>

  !lc> /*! \brief C interface to create ELPA communicators
  !lc> *
  !lc> * \param mpi_comm_word    MPI global communicator (in)
  !lc> * \param my_prow          Row coordinate of the calling process in the process grid (in)
  !lc> * \param my_pcol          Column coordinate of the calling process in the process grid (in)
  !lc> * \param mpi_comm_rows    Communicator for communicating within rows of processes (out)
  !lc> * \result int             integer error value of mpi_comm_split function
  !lc> */
  !lc> int elpa_get_communicators(int mpi_comm_world, int my_prow, int my_pcol, int *mpi_comm_rows, int *mpi_comm_cols);
  function elpa_get_communicators_wrapper_c(mpi_comm_world, my_prow, my_pcol, &
                                          mpi_comm_rows, mpi_comm_cols)     &
                                          result(mpierr) bind(C,name="elpa_get_communicators")
    use, intrinsic :: iso_c_binding
    use elpa1, only : elpa_get_communicators

    implicit none
    integer(kind=c_int)         :: mpierr
    integer(kind=c_int), value  :: mpi_comm_world, my_prow, my_pcol
    integer(kind=c_int)         :: mpi_comm_rows, mpi_comm_cols

    mpierr = elpa_get_communicators(mpi_comm_world, my_prow, my_pcol, &
                                    mpi_comm_rows, mpi_comm_cols)

  end function


  !lc>  /*! \brief C interface to solve the double-precision real eigenvalue problem with 1-stage solver
  !lc>  *
  !lc> *  \param  na                   Order of matrix a
  !lc> *  \param  nev                  Number of eigenvalues needed.
  !lc> *                               The smallest nev eigenvalues/eigenvectors are calculated.
  !lc> *  \param  a                    Distributed matrix for which eigenvalues are to be computed.
  !lc> *                               Distribution is like in Scalapack.
  !lc> *                               The full matrix must be set (not only one half like in scalapack).
  !lc> *  \param lda                   Leading dimension of a
  !lc> *  \param ev(na)                On output: eigenvalues of a, every processor gets the complete set
  !lc> *  \param q                     On output: Eigenvectors of a
  !lc> *                               Distribution is like in Scalapack.
  !lc> *                               Must be always dimensioned to the full size (corresponding to (na,na))
  !lc> *                               even if only a part of the eigenvalues is needed.
  !lc> *  \param ldq                   Leading dimension of q
  !lc> *  \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
  !lc> *  \param matrixCols           distributed number of matrix columns
  !lc> *  \param mpi_comm_rows        MPI-Communicator for rows
  !lc> *  \param mpi_comm_cols        MPI-Communicator for columns
  !lc> *  \param useGPU               use GPU (1=yes, 0=No)
  !lc> *
  !lc> *  \result                     int: 1 if error occured, otherwise 0
  !lc>*/
#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "../../general/precision_macros.h"
  !lc> int elpa_solve_evp_real_1stage_double_precision(int na, int nev, double *a, int lda, double *ev, double *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int useGPU);

#include "./elpa1_c_interface_template.X90"
#undef REALCASE
#undef DOUBLE_PRECISION

#ifdef WANT_SINGLE_PRECISION_REAL

  !lc>  /*! \brief C interface to solve the single-precision real eigenvalue problem with 1-stage solver
  !lc>  *
  !lc> *  \param  na                   Order of matrix a
  !lc> *  \param  nev                  Number of eigenvalues needed.
  !lc> *                               The smallest nev eigenvalues/eigenvectors are calculated.
  !lc> *  \param  a                    Distributed matrix for which eigenvalues are to be computed.
  !lc> *                               Distribution is like in Scalapack.
  !lc> *                               The full matrix must be set (not only one half like in scalapack).
  !lc> *  \param lda                   Leading dimension of a
  !lc> *  \param ev(na)                On output: eigenvalues of a, every processor gets the complete set
  !lc> *  \param q                     On output: Eigenvectors of a
  !lc> *                               Distribution is like in Scalapack.
  !lc> *                               Must be always dimensioned to the full size (corresponding to (na,na))
  !lc> *                               even if only a part of the eigenvalues is needed.
  !lc> *  \param ldq                   Leading dimension of q
  !lc> *  \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
  !lc> *  \param matrixCols           distributed number of matrix columns
  !lc> *  \param mpi_comm_rows        MPI-Communicator for rows
  !lc> *  \param mpi_comm_cols        MPI-Communicator for columns
  !lc> *  \param useGPU               use GPU (1=yes, 0=No)
  !lc> *
  !lc> *  \result                     int: 1 if error occured, otherwise 0
  !lc>*/
#define REALCASE 1
#undef DOUBLE_PRECISION
#define SINGLE_PRECISION 1
#include "../../general/precision_macros.h"

  !lc> int elpa_solve_evp_real_1stage_single_precision(int na, int nev, float *a, int lda, float *ev, float *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int useGPU);

#include "./elpa1_c_interface_template.X90"
#undef SINGLE_PRECISION
#undef REALCASE
#endif /* WANT_SINGLE_PRECISION_REAL */

  !lc> /*! \brief C interface to solve the double-precision complex eigenvalue problem with 1-stage solver
  !lc> *
  !lc> *  \param  na                   Order of matrix a
  !lc> *  \param  nev                  Number of eigenvalues needed.
  !lc> *                               The smallest nev eigenvalues/eigenvectors are calculated.
  !lc> *  \param  a                    Distributed matrix for which eigenvalues are to be computed.
  !lc> *                               Distribution is like in Scalapack.
  !lc> *                               The full matrix must be set (not only one half like in scalapack).
  !lc> *  \param lda                   Leading dimension of a
  !lc> *  \param ev(na)                On output: eigenvalues of a, every processor gets the complete set
  !lc> *  \param q                     On output: Eigenvectors of a
  !lc> *                               Distribution is like in Scalapack.
  !lc> *                               Must be always dimensioned to the full size (corresponding to (na,na))
  !lc> *                               even if only a part of the eigenvalues is needed.
  !lc> *  \param ldq                   Leading dimension of q
  !lc> *  \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
  !lc> *  \param matrixCols           distributed number of matrix columns
  !lc> *  \param mpi_comm_rows        MPI-Communicator for rows
  !lc> *  \param mpi_comm_cols        MPI-Communicator for columns
  !lc> *  \param useGPU               use GPU (1=yes, 0=No)
  !lc> *
  !lc> *  \result                     int: 1 if error occured, otherwise 0
  !lc> */

#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#include "../../general/precision_macros.h"

  !lc> int elpa_solve_evp_complex_1stage_double_precision(int na, int nev, double complex *a, int lda, double *ev, double complex *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int useGPU);

#include "./elpa1_c_interface_template.X90"
#undef COMPLEXCASE
#undef DOUBLE_PRECISION

#ifdef WANT_SINGLE_PRECISION_COMPLEX

  !lc> /*! \brief C interface to solve the single-precision complex eigenvalue problem with 1-stage solver
  !lc> *
  !lc> *  \param  na                   Order of matrix a
  !lc> *  \param  nev                  Number of eigenvalues needed.
  !lc> *                               The smallest nev eigenvalues/eigenvectors are calculated.
  !lc> *  \param  a                    Distributed matrix for which eigenvalues are to be computed.
  !lc> *                               Distribution is like in Scalapack.
  !lc> *                               The full matrix must be set (not only one half like in scalapack).
  !lc> *  \param lda                   Leading dimension of a
  !lc> *  \param ev(na)                On output: eigenvalues of a, every processor gets the complete set
  !lc> *  \param q                     On output: Eigenvectors of a
  !lc> *                               Distribution is like in Scalapack.
  !lc> *                               Must be always dimensioned to the full size (corresponding to (na,na))
  !lc> *                               even if only a part of the eigenvalues is needed.
  !lc> *  \param ldq                   Leading dimension of q
  !lc> *  \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
  !lc> *  \param matrixCols           distributed number of matrix columns
  !lc> *  \param mpi_comm_rows        MPI-Communicator for rows
  !lc> *  \param mpi_comm_cols        MPI-Communicator for columns
  !lc> *  \param useGPU               use GPU (1=yes, 0=No)
  !lc> *
  !lc> *  \result                     int: 1 if error occured, otherwise 0
  !lc> */
#define COMPLEXCASE 1
#undef DOUBLE_PRECISION
#define SINGLE_PRECISION
#include "../../general/precision_macros.h"

  !lc> int elpa_solve_evp_complex_1stage_single_precision(int na, int nev,  complex float *a, int lda, float *ev, complex float *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int useGPU);

#include "./elpa1_c_interface_template.X90"

#undef SINGLE_PRECISION
#undef COMPLEXCASE
#endif /* WANT_SINGLE_PRECISION_COMPLEX */

  !lc> /*
  !lc> \brief  C interface to solve double-precision tridiagonal eigensystem with divide and conquer method
  !lc> \details
  !lc>
  !lc> *\param na                    Matrix dimension
  !lc> *\param nev                   number of eigenvalues/vectors to be computed
  !lc> *\param d                     array d(na) on input diagonal elements of tridiagonal matrix, on
  !lc> *                             output the eigenvalues in ascending order
  !lc> *\param e                     array e(na) on input subdiagonal elements of matrix, on exit destroyed
  !lc> *\param q                     on exit : matrix q(ldq,matrixCols) contains the eigenvectors
  !lc> *\param ldq                   leading dimension of matrix q
  !lc> *\param nblk                  blocksize of cyclic distribution, must be the same in both directions!
  !lc> *\param matrixCols            columns of matrix q
  !lc> *\param mpi_comm_rows         MPI communicator for rows
  !lc> *\param mpi_comm_cols         MPI communicator for columns
  !lc> *\param wantDebug             give more debug information if 1, else 0
  !lc> *\result success              int 1 on success, else 0
  !lc> */
  !lc> int elpa_solve_tridi_double(int na, int nev, double *d, double *e, double *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "../../general/precision_macros.h"
#include "./elpa_solve_tridi_c_interface_template.X90"
#undef DOUBLE_PRECISION
#undef REALCASE

#ifdef WANT_SINGLE_PRECISION_REAL

  !lc> /*
  !lc> \brief  C interface to solve single-precision tridiagonal eigensystem with divide and conquer method
  !lc> \details
  !lc>
  !lc> \param na                    Matrix dimension
  !lc> \param nev                   number of eigenvalues/vectors to be computed
  !lc> \param d                     array d(na) on input diagonal elements of tridiagonal matrix, on
  !lc>                              output the eigenvalues in ascending order
  !lc> \param e                     array e(na) on input subdiagonal elements of matrix, on exit destroyed
  !lc> \param q                     on exit : matrix q(ldq,matrixCols) contains the eigenvectors
  !lc> \param ldq                   leading dimension of matrix q
  !lc> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
  !lc> \param matrixCols            columns of matrix q
  !lc> \param mpi_comm_rows         MPI communicator for rows
  !lc> \param mpi_comm_cols         MPI communicator for columns
  !lc> \param wantDebug             give more debug information if 1, else 0
  !lc> \result success              int 1 on success, else 0
  !lc> */
  !lc> int elpa_solve_tridi_single(int na, int nev, float *d, float *e, float *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
#define REALCASE 1
#define SINGLE_PRECISION 1
#include "../../general/precision_macros.h"
#include "./elpa_solve_tridi_c_interface_template.X90"
#undef SINGLE_PRECISION
#undef REALCASE

#endif /* WANT_SINGLE_PRECISION_REAL */

  !lc> /*
  !lc> \brief  C interface for elpa_mult_at_b_real_double: Performs C : = A**T * B for double-precision matrices
  !lc>         where   A is a square matrix (na,na) which is optionally upper or lower triangular
  !lc>                 B is a (na,ncb) matrix
  !lc>                 C is a (na,ncb) matrix where optionally only the upper or lower
  !lc>                   triangle may be computed
  !lc> \details
  !lc> \param  uplo_a               'U' if A is upper triangular
  !lc>                              'L' if A is lower triangular
  !lc>                              anything else if A is a full matrix
  !lc>                              Please note: This pertains to the original A (as set in the calling program)
  !lc>                                           whereas the transpose of A is used for calculations
  !lc>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
  !lc>                              i.e. it may contain arbitrary numbers
  !lc> \param uplo_c                'U' if only the upper diagonal part of C is needed
  !lc>                              'L' if only the upper diagonal part of C is needed
  !lc>                              anything else if the full matrix C is needed
  !lc>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
  !lc>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
  !lc> \param na                    Number of rows/columns of A, number of rows of B and C
  !lc> \param ncb                   Number of columns  of B and C
  !lc> \param a                     matrix a
  !lc> \param lda                   leading dimension of matrix a
  !lc> \param ldaCols               columns of matrix a
  !lc> \param b                     matrix b
  !lc> \param ldb                   leading dimension of matrix b
  !lc> \param ldbCols               columns of matrix b
  !lc> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
  !lc> \param  mpi_comm_rows        MPI communicator for rows
  !lc> \param  mpi_comm_cols        MPI communicator for columns
  !lc> \param c                     matrix c
  !lc> \param ldc                   leading dimension of matrix c
  !lc> \param ldcCols               columns of matrix c
  !lc> \result success              int report success (1) or failure (0)
  !lc> */

  !lc> int elpa_mult_at_b_real_double(char uplo_a, char uplo_c, int na, int ncb, double *a, int lda, int ldaCols, double *b, int ldb, int ldbCols, int nlbk, int mpi_comm_rows, int mpi_comm_cols, double *c, int ldc, int ldcCols);

#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "../../general/precision_macros.h"
#include "./elpa_mult_at_b_c_interface_template.X90"
#undef DOUBLE_PRECISION
#undef REALCASE

#ifdef WANT_SINGLE_PRECISION_REAL
  !lc> /*
  !lc> \brief  C interface for elpa_mult_at_b_real_single: Performs C : = A**T * B for single-precision matrices
  !lc>         where   A is a square matrix (na,na) which is optionally upper or lower triangular
  !lc>                 B is a (na,ncb) matrix
  !lc>                 C is a (na,ncb) matrix where optionally only the upper or lower
  !lc>                   triangle may be computed
  !lc> \details
  !lc> \param  uplo_a               'U' if A is upper triangular
  !lc>                              'L' if A is lower triangular
  !lc>                              anything else if A is a full matrix
  !lc>                              Please note: This pertains to the original A (as set in the calling program)
  !lc>                                           whereas the transpose of A is used for calculations
  !lc>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
  !lc>                              i.e. it may contain arbitrary numbers
  !lc> \param uplo_c                'U' if only the upper diagonal part of C is needed
  !lc>                              'L' if only the upper diagonal part of C is needed
  !lc>                              anything else if the full matrix C is needed
  !lc>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
  !lc>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
  !lc> \param na                    Number of rows/columns of A, number of rows of B and C
  !lc> \param ncb                   Number of columns  of B and C
  !lc> \param a                     matrix a
  !lc> \param lda                   leading dimension of matrix a
  !lc> \param ldaCols               columns of matrix a
  !lc> \param b                     matrix b
  !lc> \param ldb                   leading dimension of matrix b
  !lc> \param ldbCols               columns of matrix b
  !lc> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
  !lc> \param  mpi_comm_rows        MPI communicator for rows
  !lc> \param  mpi_comm_cols        MPI communicator for columns
  !lc> \param c                     matrix c
  !lc> \param ldc                   leading dimension of matrix c
  !lc> \result success              int report success (1) or failure (0)
  !lc> */

  !lc> int elpa_mult_at_b_real_single(char uplo_a, char uplo_c, int na, int ncb, float *a, int lda, int ldaCols, float *b, int ldb, int ldbCols, int nlbk, int mpi_comm_rows, int mpi_comm_cols, float *c, int ldc, int ldcCols);


#define REALCASE 1
#define SINGLE_PRECISION 1
#include "../../general/precision_macros.h"
#include "./elpa_mult_at_b_c_interface_template.X90"
#undef SINGLE_PRECISION
#undef REALCASE

#endif /* WANT_SINGLE_PRECISION_REAL */

  !lc> /*
  !lc> \brief C interface for elpa_mult_ah_b_complex_double: Performs C : = A**H * B for double-precision matrices
  !lc>         where   A is a square matrix (na,na) which is optionally upper or lower triangular
  !lc>                 B is a (na,ncb) matrix
  !lc>                 C is a (na,ncb) matrix where optionally only the upper or lower
  !lc>                   triangle may be computed
  !lc> \details
  !lc>
  !lc> \param  uplo_a               'U' if A is upper triangular
  !lc>                              'L' if A is lower triangular
  !lc>                              anything else if A is a full matrix
  !lc>                              Please note: This pertains to the original A (as set in the calling program)
  !lc>                                           whereas the transpose of A is used for calculations
  !lc>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
  !lc>                              i.e. it may contain arbitrary numbers
  !lc> \param uplo_c                'U' if only the upper diagonal part of C is needed
  !lc>                              'L' if only the upper diagonal part of C is needed
  !lc>                              anything else if the full matrix C is needed
  !lc>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
  !lc>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
  !lc> \param na                    Number of rows/columns of A, number of rows of B and C
  !lc> \param ncb                   Number of columns  of B and C
  !lc> \param a                     matrix a
  !lc> \param lda                   leading dimension of matrix a
  !lc> \param b                     matrix b
  !lc> \param ldb                   leading dimension of matrix b
  !lc> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
  !lc> \param  mpi_comm_rows        MPI communicator for rows
  !lc> \param  mpi_comm_cols        MPI communicator for columns
  !lc> \param c                     matrix c
  !lc> \param ldc                   leading dimension of matrix c
  !lc> \result success              int reports success (1) or failure (0)
  !lc> */

  !lc> int elpa_mult_ah_b_complex_double(char uplo_a, char uplo_c, int na, int ncb, double complex *a, int lda, int ldaCols, double complex *b, int ldb, int ldbCols, int nblk, int mpi_comm_rows, int mpi_comm_cols, double complex *c, int ldc, int ldcCols);
#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#include "../../general/precision_macros.h"
#include "./elpa_mult_ah_b_c_interface_template.X90"
#undef DOUBLE_PRECISION
#undef COMPLEXCASE


#ifdef WANT_SINGLE_PRECISION_COMPLEX

  !lc> /*
  !lc> \brief C interface for elpa_mult_ah_b_complex_single: Performs C : = A**H * B for single-precision matrices
  !lc>         where   A is a square matrix (na,na) which is optionally upper or lower triangular
  !lc>                 B is a (na,ncb) matrix
  !lc>                 C is a (na,ncb) matrix where optionally only the upper or lower
  !lc>                   triangle may be computed
  !lc> \details
  !lc>
  !lc> \param  uplo_a               'U' if A is upper triangular
  !lc>                              'L' if A is lower triangular
  !lc>                              anything else if A is a full matrix
  !lc>                              Please note: This pertains to the original A (as set in the calling program)
  !lc>                                           whereas the transpose of A is used for calculations
  !lc>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
  !lc>                              i.e. it may contain arbitrary numbers
  !lc> \param uplo_c                'U' if only the upper diagonal part of C is needed
  !lc>                              'L' if only the upper diagonal part of C is needed
  !lc>                              anything else if the full matrix C is needed
  !lc>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
  !lc>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
  !lc> \param na                    Number of rows/columns of A, number of rows of B and C
  !lc> \param ncb                   Number of columns  of B and C
  !lc> \param a                     matrix a
  !lc> \param lda                   leading dimension of matrix a
  !lc> \param b                     matrix b
  !lc> \param ldb                   leading dimension of matrix b
  !lc> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
  !lc> \param  mpi_comm_rows        MPI communicator for rows
  !lc> \param  mpi_comm_cols        MPI communicator for columns
  !lc> \param c                     matrix c
  !lc> \param ldc                   leading dimension of matrix c
  !lc> \result success              int reports success (1) or failure (0)
  !lc> */

  !lc> int elpa_mult_ah_b_complex_single(char uplo_a, char uplo_c, int na, int ncb, complex float *a, int lda, int ldaCols, complex float *b, int ldb, int ldbCols, int nblk, int mpi_comm_rows, int mpi_comm_cols, complex float *c, int ldc, int ldcCols);
#define COMPLEXCASE 1
#define SINGLE_PRECISION 1
#include "../../general/precision_macros.h"
#include "./elpa_mult_ah_b_c_interface_template.X90"
#undef SINGLE_PRECISION
#undef COMPLEXCASE
#endif /* WANT_SINGLE_PRECISION_COMPLEX */

  !lc> /*
  !lc> \brief  C interface to elpa_invert_trm_real_double: Inverts a real double-precision upper triangular matrix
  !lc> \details
  !lc> \param  na                   Order of matrix
  !lc> \param  a(lda,matrixCols)    Distributed matrix which should be inverted
  !lc>                              Distribution is like in Scalapack.
  !lc>                              Only upper triangle is needs to be set.
  !lc>                              The lower triangle is not referenced.
  !lc> \param  lda                  Leading dimension of a
  !lc> \param                       matrixCols  local columns of matrix a
  !lc> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
  !lc> \param  mpi_comm_rows        MPI communicator for rows
  !lc> \param  mpi_comm_cols        MPI communicator for columns
  !lc> \param wantDebug             int more debug information on failure if 1, else 0
  !lc> \result succes               int reports success (1) or failure (0)
  !lc> */

  !lc> int elpa_invert_trm_real_double(int na, double *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "../../general/precision_macros.h"
#include "./elpa_invert_trm_c_interface_template.X90"
#undef DOUBLE_PRECISION
#undef REALCASE

#ifdef WANT_SINGLE_PRECISION_REAL

  !lc> /*
  !lc> \brief  C interface to elpa_invert_trm_real_single: Inverts a real single-precision upper triangular matrix
  !lc> \details
  !lc> \param  na                   Order of matrix
  !lc> \param  a(lda,matrixCols)    Distributed matrix which should be inverted
  !lc>                              Distribution is like in Scalapack.
  !lc>                              Only upper triangle is needs to be set.
  !lc>                              The lower triangle is not referenced.
  !lc> \param  lda                  Leading dimension of a
  !lc> \param                       matrixCols  local columns of matrix a
  !lc> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
  !lc> \param  mpi_comm_rows        MPI communicator for rows
  !lc> \param  mpi_comm_cols        MPI communicator for columns
  !lc> \param wantDebug             int more debug information on failure if 1, else 0
  !lc> \result succes               int reports success (1) or failure (0)
  !lc> */

  !lc> int elpa_invert_trm_real_single(int na, double *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);

#define REALCASE 1
#define SINGLE_PRECISION 1
#include "../../general/precision_macros.h"
#include "./elpa_invert_trm_c_interface_template.X90"
#undef SINGLE_PRECISION
#undef REALCASE

#endif /* WANT_SINGLE_PRECISION_REAL */

 !lc> /*
 !lc> \brief  C interface to elpa_invert_trm_complex_double: Inverts a double-precision complex upper triangular matrix
 !lc> \details
 !lc> \param  na                   Order of matrix
 !lc> \param  a(lda,matrixCols)    Distributed matrix which should be inverted
 !lc>                              Distribution is like in Scalapack.
 !lc>                              Only upper triangle is needs to be set.
 !lc>                              The lower triangle is not referenced.
 !lc> \param  lda                  Leading dimension of a
 !lc> \param                       matrixCols  local columns of matrix a
 !lc> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
 !lc> \param  mpi_comm_rows        MPI communicator for rows
 !lc> \param  mpi_comm_cols        MPI communicator for columns
 !lc> \param wantDebug             int more debug information on failure if 1, else 0
 !lc> \result succes               int reports success (1) or failure (0)
 !lc> */

 !lc> int elpa_invert_trm_complex_double(int na, double complex *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#include "../../general/precision_macros.h"
#include "./elpa_invert_trm_c_interface_template.X90"
#undef DOUBLE_PRECISION
#undef COMPLEXCASE

#ifdef WANT_SINGLE_PRECISION_COMPLEX
 !lc> /*
 !lc> \brief  C interface to elpa_invert_trm_complex_single: Inverts a single-precision complex upper triangular matrix
 !lc> \details
 !lc> \param  na                   Order of matrix
 !lc> \param  a(lda,matrixCols)    Distributed matrix which should be inverted
 !lc>                              Distribution is like in Scalapack.
 !lc>                              Only upper triangle is needs to be set.
 !lc>                              The lower triangle is not referenced.
 !lc> \param  lda                  Leading dimension of a
 !lc> \param                       matrixCols  local columns of matrix a
 !lc> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
 !lc> \param  mpi_comm_rows        MPI communicator for rows
 !lc> \param  mpi_comm_cols        MPI communicator for columns
 !lc> \param wantDebug             int more debug information on failure if 1, else 0
 !lc> \result succes               int reports success (1) or failure (0)
 !lc> */

 !lc> int elpa_invert_trm_complex_single(int na, complex float *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
#define COMPLEXCASE 1
#define SINGLE_PRECISION 1
#include "../../general/precision_macros.h"
#include "./elpa_invert_trm_c_interface_template.X90"
#undef SINGLE_PRECISION
#undef COMPLEXCASE


#endif /* WANT_SINGLE_PRECISION_COMPLEX */

 !lc> /*
 !lc> \brief  elpa_cholesky_real_double: Cholesky factorization of a double-precision real symmetric matrix
 !lc> \details
 !lc>
 !lc> *\param  na                   Order of matrix
 !lc> *\param  a(lda,matrixCols)    Distributed matrix which should be factorized.
 !lc> *                             Distribution is like in Scalapack.
 !lc> *                             Only upper triangle is needs to be set.
 !lc> *                             On return, the upper triangle contains the Cholesky factor
 !lc> *                             and the lower triangle is set to 0.
 !lc> *\param  lda                  Leading dimension of a
 !lc> *\param  matrixCols           local columns of matrix a
 !lc> *\param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
 !lc> *\param  mpi_comm_rows        MPI communicator for rows
 !lc> *\param  mpi_comm_cols        MPI communicator for columns
 !lc> *\param wantDebug             int more debug information on failure if 1, else 0
 !lc> *\result succes               int reports success (1) or failure (0)
 !lc> */

 !lc> int elpa_cholesky_real_double(int na, double *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "../../general/precision_macros.h"
#include "./elpa_cholesky_c_interface_template.X90"
#undef DOUBLE_PRECISION
#undef REALCASE

#ifdef WANT_SINGLE_PRECISION_REAL

 !lc> /*
 !lc> \brief  elpa_cholesky_real_single: Cholesky factorization of a single-precision real symmetric matrix
 !lc> \details
 !lc>
 !lc> \param  na                   Order of matrix
 !lc> \param  a(lda,matrixCols)    Distributed matrix which should be factorized.
 !lc>                              Distribution is like in Scalapack.
 !lc>                              Only upper triangle is needs to be set.
 !lc>                              On return, the upper triangle contains the Cholesky factor
 !lc>                              and the lower triangle is set to 0.
 !lc> \param  lda                  Leading dimension of a
 !lc> \param                       matrixCols  local columns of matrix a
 !lc> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
 !lc> \param  mpi_comm_rows        MPI communicator for rows
 !lc> \param  mpi_comm_cols        MPI communicator for columns
 !lc> \param wantDebug             int more debug information on failure if 1, else 0
 !lc> \result succes               int reports success (1) or failure (0)
 !lc> */

 !lc> int elpa_cholesky_real_single(int na, float *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
#define REALCASE 1
#define SINGLE_PRECISION 1
#include "../../general/precision_macros.h"
#include "./elpa_cholesky_c_interface_template.X90"
#undef SINGLE_PRECISION
#undef REALCASE


#endif /* WANT_SINGLE_PRECISION_REAL */

 !lc> /*
 !lc> \brief  C interface elpa_cholesky_complex_double: Cholesky factorization of a double-precision complex hermitian matrix
 !lc> \details
 !lc> \param  na                   Order of matrix
 !lc> \param  a(lda,matrixCols)    Distributed matrix which should be factorized.
 !lc>                              Distribution is like in Scalapack.
 !lc>                              Only upper triangle is needs to be set.
 !lc>                              On return, the upper triangle contains the Cholesky factor
 !lc>                              and the lower triangle is set to 0.
 !lc> \param  lda                  Leading dimension of a
 !lc> \param                       matrixCols  local columns of matrix a
 !lc> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
 !lc> \param  mpi_comm_rows        MPI communicator for rows
 !lc> \param  mpi_comm_cols        MPI communicator for columns
 !lc> \param wantDebug             int more debug information on failure, if 1, else 0
 !lc> \result succes               int reports success (1) or failure (0)
 !lc> */

 !lc> int elpa_cholesky_complex_double(int na, double complex *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#include "../../general/precision_macros.h"
#include "./elpa_cholesky_c_interface_template.X90"
#undef DOUBLE_PRECISION
#undef COMPLEXCASE

#ifdef WANT_SINGLE_PRECISION_COMPLEX

 !lc> /*
 !lc> \brief  C interface elpa_cholesky_complex_single: Cholesky factorization of a single-precision complex hermitian matrix
 !lc> \details
 !lc> \param  na                   Order of matrix
 !lc> \param  a(lda,matrixCols)    Distributed matrix which should be factorized.
 !lc>                              Distribution is like in Scalapack.
 !lc>                              Only upper triangle is needs to be set.
 !lc>                              On return, the upper triangle contains the Cholesky factor
 !lc>                              and the lower triangle is set to 0.
 !lc> \param  lda                  Leading dimension of a
 !lc> \param                       matrixCols  local columns of matrix a
 !lc> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
 !lc> \param  mpi_comm_rows        MPI communicator for rows
 !lc> \param  mpi_comm_cols        MPI communicator for columns
 !lc> \param wantDebug             int more debug information on failure, if 1, else 0
 !lc> \result succes               int reports success (1) or failure (0)
 !lc> */

 !lc> int elpa_cholesky_complex_single(int na, complex float *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
#define COMPLEXCASE 1
#define SINGLE_PRECISION 1
#include "../../general/precision_macros.h"
#include "./elpa_cholesky_c_interface_template.X90"
#undef SINGLE_PRECISION
#undef COMPLEXCASE
#endif /* WANT_SINGLE_PRECISION_COMPLEX */
