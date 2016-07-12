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
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaftrn,
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
  !c> #include <complex.h>

  !c> /*! \brief C old, deprecated interface to create the MPI communicators for ELPA
  !c> *
  !c> * \param mpi_comm_word    MPI global communicator (in)
  !c> * \param my_prow          Row coordinate of the calling process in the process grid (in)
  !c> * \param my_pcol          Column coordinate of the calling process in the process grid (in)
  !c> * \param mpi_comm_rows    Communicator for communicating within rows of processes (out)
  !c> * \result int             integer error value of mpi_comm_split function
  !c> */
  !c> int elpa_get_communicators(int mpi_comm_world, int my_prow, int my_pcol, int *mpi_comm_rows, int *mpi_comm_cols);
  function get_elpa_row_col_comms_wrapper_c_name1(mpi_comm_world, my_prow, my_pcol, &
                                          mpi_comm_rows, mpi_comm_cols)     &
                                          result(mpierr) bind(C,name="elpa_get_communicators")
    use, intrinsic :: iso_c_binding
    use elpa1, only : get_elpa_row_col_comms

    implicit none
    integer(kind=c_int)         :: mpierr
    integer(kind=c_int), value  :: mpi_comm_world, my_prow, my_pcol
    integer(kind=c_int)         :: mpi_comm_rows, mpi_comm_cols

    mpierr = get_elpa_row_col_comms(mpi_comm_world, my_prow, my_pcol, &
                                    mpi_comm_rows, mpi_comm_cols)

  end function
  !c> #include <complex.h>

  !c> /*! \brief C interface to create the MPI communicators for ELPA
  !c> *
  !c> * \param mpi_comm_word    MPI global communicator (in)
  !c> * \param my_prow          Row coordinate of the calling process in the process grid (in)
  !c> * \param my_pcol          Column coordinate of the calling process in the process grid (in)
  !c> * \param mpi_comm_rows    Communicator for communicating within rows of processes (out)
  !c> * \result int             integer error value of mpi_comm_split function
  !c> */
  !c> int get_elpa_communicators(int mpi_comm_world, int my_prow, int my_pcol, int *mpi_comm_rows, int *mpi_comm_cols);
  function get_elpa_row_col_comms_wrapper_c_name2(mpi_comm_world, my_prow, my_pcol, &
                                          mpi_comm_rows, mpi_comm_cols)     &
                                          result(mpierr) bind(C,name="get_elpa_communicators")
    use, intrinsic :: iso_c_binding
    use elpa1, only : get_elpa_row_col_comms

    implicit none
    integer(kind=c_int)         :: mpierr
    integer(kind=c_int), value  :: mpi_comm_world, my_prow, my_pcol
    integer(kind=c_int)         :: mpi_comm_rows, mpi_comm_cols

    mpierr = get_elpa_row_col_comms(mpi_comm_world, my_prow, my_pcol, &
                                    mpi_comm_rows, mpi_comm_cols)

  end function



  !c>  /*! \brief C interface to solve the real eigenvalue problem with 1-stage solver
  !c>  *
  !c> *  \param  na                   Order of matrix a
  !c> *  \param  nev                  Number of eigenvalues needed.
  !c> *                               The smallest nev eigenvalues/eigenvectors are calculated.
  !c> *  \param  a                    Distributed matrix for which eigenvalues are to be computed.
  !c> *                               Distribution is like in Scalapack.
  !c> *                               The full matrix must be set (not only one half like in scalapack).
  !c> *  \param lda                   Leading dimension of a
  !c> *  \param ev(na)                On output: eigenvalues of a, every processor gets the complete set
  !c> *  \param q                     On output: Eigenvectors of a
  !c> *                               Distribution is like in Scalapack.
  !c> *                               Must be always dimensioned to the full size (corresponding to (na,na))
  !c> *                               even if only a part of the eigenvalues is needed.
  !c> *  \param ldq                   Leading dimension of q
  !c> *  \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
  !c> *  \param matrixCols           distributed number of matrix columns
  !c> *  \param mpi_comm_rows        MPI-Communicator for rows
  !c> *  \param mpi_comm_cols        MPI-Communicator for columns
  !c> *
  !c> *  \result                     int: 1 if error occured, otherwise 0
  !c>*/
  !c> int elpa_solve_evp_real_1stage(int na, int nev, double *a, int lda, double *ev, double *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols);
  function solve_elpa1_evp_real_wrapper(na, nev, a, lda, ev, q, ldq, nblk, &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols)      &
                                  result(success) bind(C,name="elpa_solve_evp_real_1stage")

    use, intrinsic :: iso_c_binding
    use elpa1, only : solve_evp_real

    implicit none
    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_cols, mpi_comm_rows
    real(kind=c_double)                    :: a(1:lda,1:matrixCols), ev(1:na), q(1:ldq,1:matrixCols)

    logical                                :: successFortran

    successFortran = solve_evp_real(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)

    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function


  !c> /*! \brief C interface to solve the complex eigenvalue problem with 1-stage solver
  !c> *
  !c> *  \param  na                   Order of matrix a
  !c> *  \param  nev                  Number of eigenvalues needed.
  !c> *                               The smallest nev eigenvalues/eigenvectors are calculated.
  !c> *  \param  a                    Distributed matrix for which eigenvalues are to be computed.
  !c> *                               Distribution is like in Scalapack.
  !c> *                               The full matrix must be set (not only one half like in scalapack).
  !c> *  \param lda                   Leading dimension of a
  !c> *  \param ev(na)                On output: eigenvalues of a, every processor gets the complete set
  !c> *  \param q                     On output: Eigenvectors of a
  !c> *                               Distribution is like in Scalapack.
  !c> *                               Must be always dimensioned to the full size (corresponding to (na,na))
  !c> *                               even if only a part of the eigenvalues is needed.
  !c> *  \param ldq                   Leading dimension of q
  !c> *  \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
  !c> *  \param matrixCols           distributed number of matrix columns
  !c> *  \param mpi_comm_rows        MPI-Communicator for rows
  !c> *  \param mpi_comm_cols        MPI-Communicator for columns
  !c> *
  !c> *  \result                     int: 1 if error occured, otherwise 0
  !c> */
  !c> int elpa_solve_evp_complex_1stage(int na, int nev, double complex *a, int lda, double *ev, double complex *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols);
  function solve_evp_real_wrapper(na, nev, a, lda, ev, q, ldq, nblk, &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols)      &
                                  result(success) bind(C,name="elpa_solve_evp_complex_1stage")

    use, intrinsic :: iso_c_binding
    use elpa1, only : solve_evp_complex

    implicit none
    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_cols, mpi_comm_rows
    complex(kind=c_double_complex)         :: a(1:lda,1:matrixCols), q(1:ldq,1:matrixCols)
    real(kind=c_double)                    :: ev(1:na)

    logical                                :: successFortran

    successFortran = solve_evp_complex(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)

    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function
  !c> /*! \brief C interface to solve the real eigenvalue problem with 2-stage solver
  !c> *
  !c> *  \param  na                        Order of matrix a
  !c> *  \param  nev                       Number of eigenvalues needed.
  !c> *                                    The smallest nev eigenvalues/eigenvectors are calculated.
  !c> *  \param  a                         Distributed matrix for which eigenvalues are to be computed.
  !c> *                                    Distribution is like in Scalapack.
  !c> *                                    The full matrix must be set (not only one half like in scalapack).
  !c> *  \param lda                        Leading dimension of a
  !c> *  \param ev(na)                     On output: eigenvalues of a, every processor gets the complete set
  !c> *  \param q                          On output: Eigenvectors of a
  !c> *                                    Distribution is like in Scalapack.
  !c> *                                    Must be always dimensioned to the full size (corresponding to (na,na))
  !c> *                                    even if only a part of the eigenvalues is needed.
  !c> *  \param ldq                        Leading dimension of q
  !c> *  \param nblk                       blocksize of cyclic distribution, must be the same in both directions!
  !c> *  \param matrixCols                 distributed number of matrix columns
  !c> *  \param mpi_comm_rows              MPI-Communicator for rows
  !c> *  \param mpi_comm_cols              MPI-Communicator for columns
  !c> *  \param mpi_coll_all               MPI communicator for the total processor set
  !c> *  \param THIS_REAL_ELPA_KERNEL_API  specify used ELPA2 kernel via API
  !c> *  \param use_qr                     use QR decomposition 1 = yes, 0 = no
  !c> *
  !c> *  \result                     int: 1 if error occured, otherwise 0
  !c> */
  !c> int elpa_solve_evp_real_2stage(int na, int nev, double *a, int lda, double *ev, double *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_REAL_ELPA_KERNEL_API, int useQR);
  function solve_elpa2_evp_real_wrapper(na, nev, a, lda, ev, q, ldq, nblk,    &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all, &
                                  THIS_REAL_ELPA_KERNEL_API, useQR)           &
                                  result(success) bind(C,name="elpa_solve_evp_real_2stage")

    use, intrinsic :: iso_c_binding
    use elpa2, only : solve_evp_real_2stage

    implicit none
    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_cols, mpi_comm_rows, &
                                              mpi_comm_all
    integer(kind=c_int), value, intent(in) :: THIS_REAL_ELPA_KERNEL_API, useQR
    real(kind=c_double)                    :: a(1:lda,1:matrixCols), ev(1:na), q(1:ldq,1:matrixCols)



    logical                                :: successFortran, useQRFortran

    if (useQR .eq. 0) then
      useQRFortran =.false.
    else
      useQRFortran = .true.
    endif

    successFortran = solve_evp_real_2stage(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, &
                                           mpi_comm_cols, mpi_comm_all,                                  &
                                           THIS_REAL_ELPA_KERNEL_API, useQRFortran)

    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function


  !c> /*! \brief C interface to solve the complex eigenvalue problem with 2-stage solver
  !c> *
  !c> *  \param  na                        Order of matrix a
  !c> *  \param  nev                       Number of eigenvalues needed.
  !c> *                                    The smallest nev eigenvalues/eigenvectors are calculated.
  !c> *  \param  a                         Distributed matrix for which eigenvalues are to be computed.
  !c> *                                    Distribution is like in Scalapack.
  !c> *                                    The full matrix must be set (not only one half like in scalapack).
  !c> *  \param lda                        Leading dimension of a
  !c> *  \param ev(na)                     On output: eigenvalues of a, every processor gets the complete set
  !c> *  \param q                          On output: Eigenvectors of a
  !c> *                                    Distribution is like in Scalapack.
  !c> *                                    Must be always dimensioned to the full size (corresponding to (na,na))
  !c> *                                    even if only a part of the eigenvalues is needed.
  !c> *  \param ldq                        Leading dimension of q
  !c> *  \param nblk                       blocksize of cyclic distribution, must be the same in both directions!
  !c> *  \param matrixCols                 distributed number of matrix columns
  !c> *  \param mpi_comm_rows              MPI-Communicator for rows
  !c> *  \param mpi_comm_cols              MPI-Communicator for columns
  !c> *  \param mpi_coll_all               MPI communicator for the total processor set
  !c> *  \param THIS_REAL_ELPA_KERNEL_API  specify used ELPA2 kernel via API
  !c> *  \param use_qr                     use QR decomposition 1 = yes, 0 = no
  !c> *
  !c> *  \result                     int: 1 if error occured, otherwise 0
  !c> */
  !c> int elpa_solve_evp_complex_2stage(int na, int nev, double complex *a, int lda, double *ev, double complex *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_COMPLEX_ELPA_KERNEL_API);
  function solve_elpa2_evp_complex_wrapper(na, nev, a, lda, ev, q, ldq, nblk,    &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all,    &
                                  THIS_COMPLEX_ELPA_KERNEL_API)                  &
                                  result(success) bind(C,name="elpa_solve_evp_complex_2stage")

    use, intrinsic :: iso_c_binding
    use elpa2, only : solve_evp_complex_2stage

    implicit none
    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_cols, mpi_comm_rows, &
                                              mpi_comm_all
    integer(kind=c_int), value, intent(in) :: THIS_COMPLEX_ELPA_KERNEL_API
    complex(kind=c_double_complex)         :: a(1:lda,1:matrixCols), q(1:ldq,1:matrixCols)
    real(kind=c_double)                    :: ev(1:na)
    logical                                :: successFortran

    successFortran = solve_evp_complex_2stage(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, &
                                              mpi_comm_all, THIS_COMPLEX_ELPA_KERNEL_API)

    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function

  !c> /*
  !c> \brief  C interface to solve tridiagonal eigensystem with divide and conquer method
  !c> \details
  !c>
  !c> \param na                    Matrix dimension
  !c> \param nev                   number of eigenvalues/vectors to be computed
  !c> \param d                     array d(na) on input diagonal elements of tridiagonal matrix, on
  !c>                              output the eigenvalues in ascending order
  !c> \param e                     array e(na) on input subdiagonal elements of matrix, on exit destroyed
  !c> \param q                     on exit : matrix q(ldq,matrixCols) contains the eigenvectors
  !c> \param ldq                   leading dimension of matrix q
  !c> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
  !c> \param matrixCols            columns of matrix q
  !c> \param mpi_comm_rows         MPI communicator for rows
  !c> \param mpi_comm_cols         MPI communicator for columns
  !c> \param wantDebug             give more debug information if 1, else 0
  !c> \result success              int 1 on success, else 0
  !c> */
  !c> int elpa_solve_tridi(int na, int nev, double *d, double *e, double *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
  function elpa_solve_tridi_wrapper(na, nev, d, e, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug) &
           result(success) bind(C,name="elpa_solve_tridi")

    use, intrinsic :: iso_c_binding
    use elpa1_auxiliary, only : elpa_solve_tridi

    implicit none
    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, ldq, nblk, matrixCols,  mpi_comm_cols, mpi_comm_rows
    integer(kind=c_int), value             :: wantDebug
    real(kind=c_double)                    :: d(1:na), e(1:na), q(1:ldq, 1:matrixCols)
    logical                                :: successFortran, wantDebugFortran

    if (wantDebug .ne. 0) then
      wantDebugFortran = .true.
    else
      wantDebugFortran = .false.
    endif

    successFortran = elpa_solve_tridi(na, nev, d, e, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebugFortran)

    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function

  !c> /*
  !c> \brief  C interface for elpa_mult_at_b_real: Performs C : = A**T * B
  !c>         where   A is a square matrix (na,na) which is optionally upper or lower triangular
  !c>                 B is a (na,ncb) matrix
  !c>                 C is a (na,ncb) matrix where optionally only the upper or lower
  !c>                   triangle may be computed
  !c> \details
  !c> \param  uplo_a               'U' if A is upper triangular
  !c>                              'L' if A is lower triangular
  !c>                              anything else if A is a full matrix
  !c>                              Please note: This pertains to the original A (as set in the calling program)
  !c>                                           whereas the transpose of A is used for calculations
  !c>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
  !c>                              i.e. it may contain arbitrary numbers
  !c> \param uplo_c                'U' if only the upper diagonal part of C is needed
  !c>                              'L' if only the upper diagonal part of C is needed
  !c>                              anything else if the full matrix C is needed
  !c>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
  !c>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
  !c> \param na                    Number of rows/columns of A, number of rows of B and C
  !c> \param ncb                   Number of columns  of B and C
  !c> \param a                     matrix a
  !c> \param lda                   leading dimension of matrix a
  !c> \param b                     matrix b
  !c> \param ldb                   leading dimension of matrix b
  !c> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
  !c> \param  mpi_comm_rows        MPI communicator for rows
  !c> \param  mpi_comm_cols        MPI communicator for columns
  !c> \param c                     matrix c
  !c> \param ldc                   leading dimension of matrix c
  !c> \result success              int report success (1) or failure (0)
  !c> */

  !c> int elpa_mult_at_b_real(char uplo_a, char uplo_c, int na, int ncb, double *a, int lda, double *b, int ldb, int nlbk, int mpi_comm_rows, int mpi_comm_cols, double *c, int ldc);
  function elpa_mult_at_b_real_wrapper(uplo_a, uplo_c, na, ncb, a, lda, b, ldb, nblk, mpi_comm_rows, mpi_comm_cols, c, ldc) &
    bind(C,name="elpa_mult_at_b_real") result(success)
    use, intrinsic :: iso_c_binding
    use elpa1_auxiliary, only : elpa_mult_at_b_real

    implicit none

    character(1,C_CHAR), value  :: uplo_a, uplo_c
    integer(kind=c_int), value  :: na, ncb, lda, ldb, nblk, mpi_comm_rows, mpi_comm_cols, ldc
    integer(kind=c_int)         :: success
    real(kind=c_double)         :: a(lda,*), b(ldb,*), c(ldc,*)
    logical                     :: successFortran

    successFortran = elpa_mult_at_b_real(uplo_a, uplo_c, na, ncb, a, lda, b, ldb, nblk, mpi_comm_rows, mpi_comm_cols, c, ldc)

    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function

  !c> /*
  !c> \brief C interface for elpa_mult_ah_b_complex: Performs C : = A**H * B
  !c>         where   A is a square matrix (na,na) which is optionally upper or lower triangular
  !c>                 B is a (na,ncb) matrix
  !c>                 C is a (na,ncb) matrix where optionally only the upper or lower
  !c>                   triangle may be computed
  !c> \details
  !c>
  !c> \param  uplo_a               'U' if A is upper triangular
  !c>                              'L' if A is lower triangular
  !c>                              anything else if A is a full matrix
  !c>                              Please note: This pertains to the original A (as set in the calling program)
  !c>                                           whereas the transpose of A is used for calculations
  !c>                              If uplo_a is 'U' or 'L', the other triangle is not used at all,
  !c>                              i.e. it may contain arbitrary numbers
  !c> \param uplo_c                'U' if only the upper diagonal part of C is needed
  !c>                              'L' if only the upper diagonal part of C is needed
  !c>                              anything else if the full matrix C is needed
  !c>                              Please note: Even when uplo_c is 'U' or 'L', the other triangle may be
  !c>                                            written to a certain extent, i.e. one shouldn't rely on the content there!
  !c> \param na                    Number of rows/columns of A, number of rows of B and C
  !c> \param ncb                   Number of columns  of B and C
  !c> \param a                     matrix a
  !c> \param lda                   leading dimension of matrix a
  !c> \param b                     matrix b
  !c> \param ldb                   leading dimension of matrix b
  !c> \param nblk                  blocksize of cyclic distribution, must be the same in both directions!
  !c> \param  mpi_comm_rows        MPI communicator for rows
  !c> \param  mpi_comm_cols        MPI communicator for columns
  !c> \param c                     matrix c
  !c> \param ldc                   leading dimension of matrix c
  !c> \result success              int reports success (1) or failure (0)
  !c> */

  !c> int elpa_mult_ah_b_complex(char uplo_a, char uplo_c, int na, int ncb, double complex *a, int lda, double complex *b, int ldb, int nblk, int mpi_comm_rows, int mpi_comm_cols, double complex *c, int ldc);
  function elpa_mult_ah_b_complex_wrapper( uplo_a, uplo_c, na, ncb, a, lda, b, ldb, nblk, mpi_comm_rows, mpi_comm_cols, c, ldc) &
    result(success) bind(C,name="elpa_mult_ah_b_complex")
    use, intrinsic :: iso_c_binding
    use elpa1_auxiliary, only : elpa_mult_ah_b_complex

    implicit none

    character(1,C_CHAR), value     :: uplo_a, uplo_c
    integer(kind=c_int), value     :: na, ncb, lda, ldb, nblk, mpi_comm_rows, mpi_comm_cols, ldc
    integer(kind=c_int)            :: success
    complex(kind=c_double_complex) :: a(lda,*), b(ldb,*), c(ldc,*)
    logical                        :: successFortran

    successFortran = elpa_mult_ah_b_complex(uplo_a, uplo_c, na, ncb, a, lda, b, ldb, nblk, mpi_comm_rows, mpi_comm_cols, c, ldc)

    if (successFortran) then
      success = 1
    else
      success = 0
     endif

  end function

  !c> /*
  !c> \brief  C interface to elpa_invert_trm_real: Inverts a upper triangular matrix
  !c> \details
  !c> \param  na                   Order of matrix
  !c> \param  a(lda,matrixCols)    Distributed matrix which should be inverted
  !c>                              Distribution is like in Scalapack.
  !c>                              Only upper triangle is needs to be set.
  !c>                              The lower triangle is not referenced.
  !c> \param  lda                  Leading dimension of a
  !c> \param                       matrixCols  local columns of matrix a
  !c> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
  !c> \param  mpi_comm_rows        MPI communicator for rows
  !c> \param  mpi_comm_cols        MPI communicator for columns
  !c> \param wantDebug             int more debug information on failure if 1, else 0
  !c> \result succes               int reports success (1) or failure (0)
  !c> */

  !c> int elpa_invert_trm_real(int na, double *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
  function elpa_invert_trm_real_wrapper(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug) &
        result(success) bind(C,name="elpa_invert_trm_real")
   use, intrinsic :: iso_c_binding
   use elpa1_auxiliary, only : elpa_invert_trm_real

   implicit none

   integer(kind=c_int), value  :: na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
   integer(kind=c_int), value  :: wantDebug
   integer(kind=c_int)         :: success
   real(kind=c_double)         :: a(lda,matrixCols)

   logical                     :: wantDebugFortran, successFortran

   if (wantDebug .ne. 0) then
     wantDebugFortran = .true.
   else
     wantDebugFortran = .false.
   endif

   successFortran = elpa_invert_trm_real(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebugFortran)

   if (successFortran) then
     success = 1
   else
     success = 0
   endif

 end function

 !c> /*
 !c> \brief  C interface to elpa_invert_trm_complex: Inverts a complex upper triangular matrix
 !c> \details
 !c> \param  na                   Order of matrix
 !c> \param  a(lda,matrixCols)    Distributed matrix which should be inverted
 !c>                              Distribution is like in Scalapack.
 !c>                              Only upper triangle is needs to be set.
 !c>                              The lower triangle is not referenced.
 !c> \param  lda                  Leading dimension of a
 !c> \param                       matrixCols  local columns of matrix a
 !c> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
 !c> \param  mpi_comm_rows        MPI communicator for rows
 !c> \param  mpi_comm_cols        MPI communicator for columns
 !c> \param wantDebug             int more debug information on failure if 1, else 0
 !c> \result succes               int reports success (1) or failure (0)
 !c> */

 !c> int elpa_invert_trm_complex(int na, double complex *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
 function elpa_invert_trm_complex_wrapper(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug) result(success) &
   bind(C,name="elpa_invert_trm_complex")

   use, intrinsic :: iso_c_binding
   use elpa1_auxiliary, only : elpa_invert_trm_complex

   implicit none

   integer(kind=c_int), value     :: na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
   integer(kind=c_int), value     :: wantDebug
   integer(kind=c_int)            :: success
   complex(kind=c_double_complex) :: a(lda, matrixCols)

   logical                        :: successFortran, wantDebugFortran


   if (wantDebug .ne. 0) then
     wantDebugFortran = .true.
   else
     wantDebugFortran = .false.
   endif

   successFortran = elpa_invert_trm_complex(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebugFortran)

   if (successFortran) then
     success = 1
   else
     success = 0
   endif
 end function

 !c> /*
 !c> \brief  elpa_cholesky_real: Cholesky factorization of a real symmetric matrix
 !c> \details
 !c>
 !c> \param  na                   Order of matrix
 !c> \param  a(lda,matrixCols)    Distributed matrix which should be factorized.
 !c>                              Distribution is like in Scalapack.
 !c>                              Only upper triangle is needs to be set.
 !c>                              On return, the upper triangle contains the Cholesky factor
 !c>                              and the lower triangle is set to 0.
 !c> \param  lda                  Leading dimension of a
 !c> \param                       matrixCols  local columns of matrix a
 !c> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
 !c> \param  mpi_comm_rows        MPI communicator for rows
 !c> \param  mpi_comm_cols        MPI communicator for columns
 !c> \param wantDebug             int more debug information on failure if 1, else 0
 !c> \result succes               int reports success (1) or failure (0)
 !c> */

 !c> int elpa_cholesky_real(int na, double *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
 function elpa_cholesky_real_wrapper(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug) result(success) &
       bind(C,name="elpa_cholesky_real")

   use, intrinsic :: iso_c_binding
   use elpa1_auxiliary, only : elpa_cholesky_real

   implicit none

   integer(kind=c_int), value :: na, lda, nblk, matrixCols,  mpi_comm_rows, mpi_comm_cols, wantDebug
   integer(kind=c_int)        :: success
   real(kind=c_double)        :: a(lda,matrixCols)

   logical                    :: successFortran, wantDebugFortran

   if (wantDebug .ne. 0) then
     wantDebugFortran = .true.
   else
     wantDebugFortran = .false.
   endif

   successFortran = elpa_cholesky_real(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebugFortran)

   if (successFortran) then
     success = 1
   else
     success = 0
   endif

 end function

 !c> /*
 !c> \brief  C interface elpa_cholesky_complex: Cholesky factorization of a complex hermitian matrix
 !c> \details
 !c> \param  na                   Order of matrix
 !c> \param  a(lda,matrixCols)    Distributed matrix which should be factorized.
 !c>                              Distribution is like in Scalapack.
 !c>                              Only upper triangle is needs to be set.
 !c>                              On return, the upper triangle contains the Cholesky factor
 !c>                              and the lower triangle is set to 0.
 !c> \param  lda                  Leading dimension of a
 !c> \param                       matrixCols  local columns of matrix a
 !c> \param  nblk                 blocksize of cyclic distribution, must be the same in both directions!
 !c> \param  mpi_comm_rows        MPI communicator for rows
 !c> \param  mpi_comm_cols        MPI communicator for columns
 !c> \param wantDebug             int more debug information on failure, if 1, else 0
 !c> \result succes               int reports success (1) or failure (0)
 !c> */

 !c> int elpa_cholesky_complex(int na, double complex *a, int lda, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int wantDebug);
 function elpa_cholesky_complex_wrapper(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug) result(success) &
       bind(C,name="elpa_cholesky_complex")
   use, intrinsic :: iso_c_binding
   use elpa1_auxiliary, only : elpa_cholesky_complex

   implicit none

   integer(kind=c_int), value     :: na, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug
   integer(kind=c_int)            :: success

   complex(kind=c_double_complex) :: a(lda,matrixCols)

   logical                        :: wantDebugFortran, successFortran

   if (wantDebug .ne. 0) then
     wantDebugFortran = .true.
   else
     wantDebugFortran = .false.
   endif

   successFortran = elpa_cholesky_complex(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebugFortran)

   if (successFortran) then
     success = 1
   else
     success = 0
   endif

 end function

