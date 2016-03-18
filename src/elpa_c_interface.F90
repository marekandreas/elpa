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
!
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".

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



  !c>  /*! \brief C interface to solve the double-precision real eigenvalue problem with 1-stage solver
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
#define DOUBLE_PRECISION_REAL 1
#ifdef DOUBLE_PRECISION_REAL
  !c> int elpa_solve_evp_real_1stage_double_precision(int na, int nev, double *a, int lda, double *ev, double *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols);
#else
  !c> int elpa_solve_evp_real_1stage_single_precision(int na, int nev, float *a, int lda, float *ev, float *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols);
#endif

#ifdef DOUBLE_PRECISION_REAL
  function solve_elpa1_evp_real_wrapper_double(na, nev, a, lda, ev, q, ldq, nblk, &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols)      &
                                  result(success) bind(C,name="elpa_solve_evp_real_1stage_double_precision")
#else
  function solve_elpa1_evp_real_wrapper_single(na, nev, a, lda, ev, q, ldq, nblk, &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols)      &
                                  result(success) bind(C,name="elpa_solve_evp_real_1stage_single_precision")
#endif

    use, intrinsic :: iso_c_binding
    use elpa1

    implicit none
    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_cols, mpi_comm_rows
#ifdef DOUBLE_PRECISION_REAL
    real(kind=c_double)                    :: a(1:lda,1:matrixCols), ev(1:na), q(1:ldq,1:matrixCols)
#else
    real(kind=c_float)                     :: a(1:lda,1:matrixCols), ev(1:na), q(1:ldq,1:matrixCols)
#endif
    logical                                :: successFortran

#ifdef DOUBLE_PRECISION_REAL
    successFortran = solve_evp_real_1stage_double(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
#else
    successFortran = solve_evp_real_1stage_single(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
#endif
    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function

#ifdef WANT_SINGLE_PRECISION_REAL
#undef DOUBLE_PRECISION_REAL
  !c>  /*! \brief C interface to solve the single-precision real eigenvalue problem with 1-stage solver
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
#ifdef DOUBLE_PRECISION_REAL
  !c> int elpa_solve_evp_real_1stage_double_precision(int na, int nev, double *a, int lda, double *ev, double *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols);
#else
  !c> int elpa_solve_evp_real_1stage_single_precision(int na, int nev, float *a, int lda, float *ev, float *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols);
#endif

#ifdef DOUBLE_PRECISION_REAL
  function solve_elpa1_evp_real_wrapper_double(na, nev, a, lda, ev, q, ldq, nblk, &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols)      &
                                  result(success) bind(C,name="elpa_solve_evp_real_1stage_double_precision")
#else
  function solve_elpa1_evp_real_wrapper_single(na, nev, a, lda, ev, q, ldq, nblk, &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols)      &
                                  result(success) bind(C,name="elpa_solve_evp_real_1stage_single_precision")
#endif
    use, intrinsic :: iso_c_binding
    use elpa1

    implicit none
    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_cols, mpi_comm_rows
#ifdef DOUBLE_PRECISION_REAL
    real(kind=c_double)                    :: a(1:lda,1:matrixCols), ev(1:na), q(1:ldq,1:matrixCols)
#else
    real(kind=c_float)                     :: a(1:lda,1:matrixCols), ev(1:na), q(1:ldq,1:matrixCols)
#endif
    logical                                :: successFortran

#ifdef DOUBLE_PRECISION_REAL
    successFortran = solve_evp_real_1stage_double(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
#else
    successFortran = solve_evp_real_1stage_single(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
#endif
    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function

#endif /* WANT_SINGLE_PRECISION_REAL */



  !c> /*! \brief C interface to solve the double-precision complex eigenvalue problem with 1-stage solver
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
#define DOUBLE_PRECISION_COMPLEX 1
#ifdef DOUBLE_PRECISION_COMPLEX
  !c> int elpa_solve_evp_complex_1stage_double_precision(int na, int nev, double complex *a, int lda, double *ev, double complex *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols);
#else
  !c> int elpa_solve_evp_complex_1stage_single_precision(int na, int nev,  complex *a, int lda, float *ev, complex *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols);
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
  function solve_evp_real_wrapper_double(na, nev, a, lda, ev, q, ldq, nblk, &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols)      &
                                  result(success) bind(C,name="elpa_solve_evp_complex_1stage_double_precision")
#else
  function solve_evp_real_wrapper_single(na, nev, a, lda, ev, q, ldq, nblk, &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols)      &
                                  result(success) bind(C,name="elpa_solve_evp_complex_1stage_single_precision")
#endif
    use, intrinsic :: iso_c_binding
    use elpa1

    implicit none
    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_cols, mpi_comm_rows
#ifdef DOUBLE_PRECISION_COMPLEX
    complex(kind=c_double_complex)         :: a(1:lda,1:matrixCols), q(1:ldq,1:matrixCols)
    real(kind=c_double)                    :: ev(1:na)
#else
    complex(kind=c_float_complex)          :: a(1:lda,1:matrixCols), q(1:ldq,1:matrixCols)
    real(kind=c_float)                     :: ev(1:na)
#endif

    logical                                :: successFortran

#ifdef DOUBLE_PRECISION_COMPLEX
    successFortran = solve_evp_complex_1stage_double(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
#else
    successFortran = solve_evp_complex_1stage_single(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
#endif
    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function

#ifdef WANT_SINGLE_PRECISION_COMPLEX

  !c> /*! \brief C interface to solve the single-precision complex eigenvalue problem with 1-stage solver
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
#undef DOUBLE_PRECISION_COMPLEX
#ifdef DOUBLE_PRECISION_COMPLEX
  !c> int elpa_solve_evp_complex_1stage_double_precision(int na, int nev, double complex *a, int lda, double *ev, double complex *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols);
#else
  !c> int elpa_solve_evp_complex_1stage_single_precision(int na, int nev,  complex *a, int lda, float *ev, complex *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols);
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
  function solve_evp_real_wrapper_double(na, nev, a, lda, ev, q, ldq, nblk, &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols)      &
                                  result(success) bind(C,name="elpa_solve_evp_complex_1stage_double_precision")
#else
  function solve_evp_real_wrapper_single(na, nev, a, lda, ev, q, ldq, nblk, &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols)      &
                                  result(success) bind(C,name="elpa_solve_evp_complex_1stage_single_precision")
#endif
    use, intrinsic :: iso_c_binding
    use elpa1

    implicit none
    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_cols, mpi_comm_rows
#ifdef DOUBLE_PRECISION_COMPLEX
    complex(kind=c_double_complex)         :: a(1:lda,1:matrixCols), q(1:ldq,1:matrixCols)
    real(kind=c_double)                    :: ev(1:na)
#else
    complex(kind=c_float_complex)          :: a(1:lda,1:matrixCols), q(1:ldq,1:matrixCols)
    real(kind=c_float)                     :: ev(1:na)
#endif

    logical                                :: successFortran

#ifdef DOUBLE_PRECISION_COMPLEX
    successFortran = solve_evp_complex_1stage_double(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
#else
    successFortran = solve_evp_complex_1stage_single(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
#endif
    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function

#endif /* WANT_SINGLE_PRECISION_COMPLEX */


  !c> /*! \brief C interface to solve the double-precision real eigenvalue problem with 2-stage solver
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
#define DOUBLE_PRECISION_REAL 1
#ifdef DOUBLE_PRECISION_REAL
  !c> int elpa_solve_evp_real_2stage_double_precision(int na, int nev, double *a, int lda, double *ev, double *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_REAL_ELPA_KERNEL_API, int useQR);
#else
  !c> int elpa_solve_evp_real_2stage_single_precision(int na, int nev, float *a, int lda, float *ev, float *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_REAL_ELPA_KERNEL_API, int useQR);
#endif

#ifdef DOUBLE_PRECISION_REAL
  function solve_elpa2_evp_real_wrapper_double(na, nev, a, lda, ev, q, ldq, nblk,    &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all, &
                                  THIS_REAL_ELPA_KERNEL_API, useQR)           &
                                  result(success) bind(C,name="elpa_solve_evp_real_2stage_double_precision")
#else
  function solve_elpa2_evp_real_wrapper_single(na, nev, a, lda, ev, q, ldq, nblk,    &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all, &
                                  THIS_REAL_ELPA_KERNEL_API, useQR)           &
                                  result(success) bind(C,name="elpa_solve_evp_real_2stage_double_precision")

                                  result(success) bind(C,name="elpa_solve_evp_real_2stage_single_precision")
#endif
    use, intrinsic :: iso_c_binding
    use elpa2

    implicit none
    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_cols, mpi_comm_rows, &
                                              mpi_comm_all
    integer(kind=c_int), value, intent(in) :: THIS_REAL_ELPA_KERNEL_API, useQR
#ifdef DOUBLE_PRECISION_REAL
    real(kind=c_double)                    :: a(1:lda,1:matrixCols), ev(1:na), q(1:ldq,1:matrixCols)
#else
    real(kind=c_float)                     :: a(1:lda,1:matrixCols), ev(1:na), q(1:ldq,1:matrixCols)
#endif

    logical                                :: successFortran, useQRFortran

    if (useQR .eq. 0) then
      useQRFortran =.false.
    else
      useQRFortran = .true.
    endif

#ifdef DOUBLE_PRECISION_REAL
    successFortran = solve_evp_real_2stage_double(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, &
                                           mpi_comm_cols, mpi_comm_all,                                  &
                                           THIS_REAL_ELPA_KERNEL_API, useQRFortran)
#else
    successFortran = solve_evp_real_2stage_single(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, &
                                           mpi_comm_cols, mpi_comm_all,                                  &
                                           THIS_REAL_ELPA_KERNEL_API, useQRFortran)
#endif
    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function

#ifdef WANT_SINGLE_PRECISION_REAL

  !c> /*! \brief C interface to solve the single-precision real eigenvalue problem with 2-stage solver
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
#undef DOUBLE_PRECISION_REAL
#ifdef DOUBLE_PRECISION_REAL
  !c> int elpa_solve_evp_real_2stage_double_precision(int na, int nev, double *a, int lda, double *ev, double *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_REAL_ELPA_KERNEL_API, int useQR);
#else
  !c> int elpa_solve_evp_real_2stage_single_precision(int na, int nev, float *a, int lda, float *ev, float *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_REAL_ELPA_KERNEL_API, int useQR);
#endif

#ifdef DOUBLE_PRECISION_REAL
  function solve_elpa2_evp_real_wrapper_double(na, nev, a, lda, ev, q, ldq, nblk,    &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all, &
                                  THIS_REAL_ELPA_KERNEL_API, useQR)           &
                                  result(success) bind(C,name="elpa_solve_evp_real_2stage_double_precision")
#else
  function solve_elpa2_evp_real_wrapper_single(na, nev, a, lda, ev, q, ldq, nblk,    &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all, &
                                  THIS_REAL_ELPA_KERNEL_API, useQR)           &
                                  result(success) bind(C,name="elpa_solve_evp_real_2stage_single_precision")
#endif
    use, intrinsic :: iso_c_binding
    use elpa2

    implicit none
    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_cols, mpi_comm_rows, &
                                              mpi_comm_all
    integer(kind=c_int), value, intent(in) :: THIS_REAL_ELPA_KERNEL_API, useQR
#ifdef DOUBLE_PRECISION_REAL
    real(kind=c_double)                    :: a(1:lda,1:matrixCols), ev(1:na), q(1:ldq,1:matrixCols)
#else
    real(kind=c_float)                     :: a(1:lda,1:matrixCols), ev(1:na), q(1:ldq,1:matrixCols)
#endif

    logical                                :: successFortran, useQRFortran

    if (useQR .eq. 0) then
      useQRFortran =.false.
    else
      useQRFortran = .true.
    endif

#ifdef DOUBLE_PRECISION_REAL
    successFortran = solve_evp_real_2stage_double(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, &
                                           mpi_comm_cols, mpi_comm_all,                                  &
                                           THIS_REAL_ELPA_KERNEL_API, useQRFortran)
#else
    successFortran = solve_evp_real_2stage_single(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, &
                                           mpi_comm_cols, mpi_comm_all,                                  &
                                           THIS_REAL_ELPA_KERNEL_API, useQRFortran)
#endif
    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function

#endif /* WANT_SINGLE_PRECISION_REAL */

  !c> /*! \brief C interface to solve the double-precision complex eigenvalue problem with 2-stage solver
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
#define DOUBLE_PRECISION_COMPLEX 1

#ifdef DOUBLE_PRECISION_COMPLEX
  !c> int elpa_solve_evp_complex_2stage_double_precision(int na, int nev, double complex *a, int lda, double *ev, double complex *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_COMPLEX_ELPA_KERNEL_API);
#else
  !c> int elpa_solve_evp_complex_2stage_single_precision(int na, int nev, complex *a, int lda, float *ev, complex *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_COMPLEX_ELPA_KERNEL_API);
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
  function solve_elpa2_evp_complex_wrapper_double(na, nev, a, lda, ev, q, ldq, nblk,    &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all,    &
                                  THIS_COMPLEX_ELPA_KERNEL_API)                  &
                                  result(success) bind(C,name="elpa_solve_evp_complex_2stage_double_precision")
#else
  function solve_elpa2_evp_complex_wrapper_single(na, nev, a, lda, ev, q, ldq, nblk,    &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all,    &
                                  THIS_COMPLEX_ELPA_KERNEL_API)                  &
                                  result(success) bind(C,name="elpa_solve_evp_complex_2stage_single_precision")
#endif

    use, intrinsic :: iso_c_binding
    use elpa2

    implicit none
    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_cols, mpi_comm_rows, &
                                              mpi_comm_all
    integer(kind=c_int), value, intent(in) :: THIS_COMPLEX_ELPA_KERNEL_API
#ifdef DOUBLE_PRECISION_COMPLEX
    complex(kind=c_double_complex)         :: a(1:lda,1:matrixCols), q(1:ldq,1:matrixCols)
    real(kind=c_double)                    :: ev(1:na)
#else
    complex(kind=c_float_complex)          :: a(1:lda,1:matrixCols), q(1:ldq,1:matrixCols)
    real(kind=c_float)                     :: ev(1:na)
#endif
    logical                                :: successFortran

#ifdef DOUBLE_PRECISION_COMPLEX
    successFortran = solve_evp_complex_2stage_double(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, &
                                              mpi_comm_all, THIS_COMPLEX_ELPA_KERNEL_API)
#else
    successFortran = solve_evp_complex_2stage_single(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, &
                                              mpi_comm_all, THIS_COMPLEX_ELPA_KERNEL_API)
#endif
    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function

#ifdef WANT_SINGLE_PRECISION_COMPLEX

  !c> /*! \brief C interface to solve the single-precision complex eigenvalue problem with 2-stage solver
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
#undef DOUBLE_PRECISION_COMPLEX

#ifdef DOUBLE_PRECISION_COMPLEX
  !c> int elpa_solve_evp_complex_2stage_double_precision(int na, int nev, double complex *a, int lda, double *ev, double complex *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_COMPLEX_ELPA_KERNEL_API);
#else
  !c> int elpa_solve_evp_complex_2stage_single_precision(int na, int nev, complex *a, int lda, float *ev, complex *q, int ldq, int nblk, int matrixCols, int mpi_comm_rows, int mpi_comm_cols, int mpi_comm_all, int THIS_COMPLEX_ELPA_KERNEL_API);
#endif

#ifdef DOUBLE_PRECISION_COMPLEX
  function solve_elpa2_evp_complex_wrapper_double(na, nev, a, lda, ev, q, ldq, nblk,    &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all,    &
                                  THIS_COMPLEX_ELPA_KERNEL_API)                  &
                                  result(success) bind(C,name="elpa_solve_evp_complex_2stage_double_precision")
#else
  function solve_elpa2_evp_complex_wrapper_single(na, nev, a, lda, ev, q, ldq, nblk,    &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all,    &
                                  THIS_COMPLEX_ELPA_KERNEL_API)                  &
                                  result(success) bind(C,name="elpa_solve_evp_complex_2stage_single_precision")
#endif

    use, intrinsic :: iso_c_binding
    use elpa2

    implicit none
    integer(kind=c_int)                    :: success
    integer(kind=c_int), value, intent(in) :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_cols, mpi_comm_rows, &
                                              mpi_comm_all
    integer(kind=c_int), value, intent(in) :: THIS_COMPLEX_ELPA_KERNEL_API
#ifdef DOUBLE_PRECISION_COMPLEX
    complex(kind=c_double_complex)         :: a(1:lda,1:matrixCols), q(1:ldq,1:matrixCols)
    real(kind=c_double)                    :: ev(1:na)
#else
    complex(kind=c_float_complex)          :: a(1:lda,1:matrixCols), q(1:ldq,1:matrixCols)
    real(kind=c_float)                     :: ev(1:na)
#endif
    logical                                :: successFortran

#ifdef DOUBLE_PRECISION_COMPLEX
    successFortran = solve_evp_complex_2stage_double(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, &
                                              mpi_comm_all, THIS_COMPLEX_ELPA_KERNEL_API)
#else
    successFortran = solve_evp_complex_2stage_single(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, &
                                              mpi_comm_all, THIS_COMPLEX_ELPA_KERNEL_API)
#endif
    if (successFortran) then
      success = 1
    else
      success = 0
    endif

  end function


#endif /* WANT_SINGLE_PRECISION_COMPLEX */

