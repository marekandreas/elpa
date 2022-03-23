!   This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), fomerly known as
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
! ELPA2 -- 2-stage solver for ELPA
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
! Author: Andreas Marek, MPCDF

#include "config-f90.h"
!> \brief Fortran module which provides the routines to use the 2-stage ELPA solver. Implementation only. Should not be used directly
module elpa2_impl
  use elpa_utilities, only : error_unit
#ifdef HAVE_LIKWID
  use likwid
#endif

  implicit none

  private

  public :: elpa_solve_evp_real_2stage_a_h_a_double_impl          !< Driver routine for real double-precision 2-stage eigenvalue problem
  public :: elpa_solve_evp_complex_2stage_a_h_a_double_impl       !< Driver routine for complex double-precision 2-stage eigenvalue problem
#ifdef WANT_SINGLE_PRECISION_REAL
  public :: elpa_solve_evp_real_2stage_a_h_a_single_impl          !< Driver routine for real single-precision 2-stage eigenvalue problem
#endif

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  public :: elpa_solve_evp_complex_2stage_a_h_a_single_impl       !< Driver routine for complex single-precision 2-stage eigenvalue problem
#endif

#ifdef HAVE_SKEWSYMMETRIC
  public :: elpa_solve_skew_evp_real_2stage_a_h_a_double_impl          !< Driver routine for real double-precision 2-stage skew-symmetric eigenvalue problem
#ifdef WANT_SINGLE_PRECISION_REAL
  public :: elpa_solve_skew_evp_real_2stage_a_h_a_single_impl          !< Driver routine for real single-precision 2-stage skew-symmetric eigenvalue problem
#endif
#endif /* HAVE_SKEWSYMMETRIC */

  public :: elpa_solve_evp_real_2stage_d_ptr_double_impl          !< Driver routine for real double-precision 2-stage eigenvalue problem
  public :: elpa_solve_evp_complex_2stage_d_ptr_double_impl       !< Driver routine for complex double-precision 2-stage eigenvalue problem
#ifdef WANT_SINGLE_PRECISION_REAL
  public :: elpa_solve_evp_real_2stage_d_ptr_single_impl          !< Driver routine for real single-precision 2-stage eigenvalue problem
#endif

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  public :: elpa_solve_evp_complex_2stage_d_ptr_single_impl       !< Driver routine for complex single-precision 2-stage eigenvalue problem
#endif

#ifdef HAVE_SKEWSYMMETRIC
  public :: elpa_solve_skew_evp_real_2stage_d_ptr_double_impl          !< Driver routine for real double-precision 2-stage skew-symmetric eigenvalue problem
#ifdef WANT_SINGLE_PRECISION_REAL
  public :: elpa_solve_skew_evp_real_2stage_d_ptr_single_impl          !< Driver routine for real single-precision 2-stage skew-symmetric eigenvalue problem
#endif
#endif /* HAVE_SKEWSYMMETRIC */


  contains

#define REALCASE 1

#define DOUBLE_PRECISION 1
#undef ACTIVATE_SKEW
#undef DEVICE_POINTER
#include "../general/precision_macros.h"
!-------------------------------------------------------------------------------
!>  \brief elpa_solve_evp_real_2stage_host_arrays_double_impl: Fortran function to solve the double-precision real eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param kernel                               specify ELPA2 kernel to use
!>
!>  \param useQR (optional)                     use QR decomposition
!>  \param useGPU (optional)                    decide whether to use GPUs or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
#include "elpa2_template.F90"
#undef REALCASE
#undef DOUBLE_PRECISION

#define REALCASE 1
#define DOUBLE_PRECISION 1
#undef ACTIVATE_SKEW
#define DEVICE_POINTER
#include "../general/precision_macros.h"
!-------------------------------------------------------------------------------
!>  \brief elpa_solve_evp_real_2stage_d_ptr_double_impl: Fortran function to solve the double-precision real eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q                                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param kernel                               specify ELPA2 kernel to use
!>
!>  \param useQR (optional)                     use QR decomposition
!>  \param useGPU (optional)                    decide whether to use GPUs or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
#include "elpa2_template.F90"
#undef DEVICE_POINTER
#undef REALCASE
#undef DOUBLE_PRECISION

#ifdef WANT_SINGLE_PRECISION_REAL
#define REALCASE 1
#define SINGLE_PRECISION 1
#undef ACTIVATE_SKEW
#undef DEVICE_POINTER
#include "../general/precision_macros.h"
!-------------------------------------------------------------------------------
!>  \brief elpa_solve_evp_real_2stage_host_arrays_single_impl: Fortran function to solve the single-precision real eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param kernel                               specify ELPA2 kernel to use
!>
!>  \param useQR (optional)                     use QR decomposition
!>  \param useGPU (optional)                    decide whether GPUs should be used or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
#include "elpa2_template.F90"
#undef REALCASE
#undef SINGLE_PRECISION

#define REALCASE 1
#define SINGLE_PRECISION 1
#undef ACTIVATE_SKEW
#define DEVICE_POINTER
#include "../general/precision_macros.h"
!-------------------------------------------------------------------------------
!>  \brief elpa_solve_evp_real_2stage_d_ptr_single_impl: Fortran function to solve the single-precision real eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q                                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param kernel                               specify ELPA2 kernel to use
!>
!>  \param useQR (optional)                     use QR decomposition
!>  \param useGPU (optional)                    decide whether GPUs should be used or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
#include "elpa2_template.F90"
#undef DEVICE_POINTER
#undef REALCASE
#undef SINGLE_PRECISION

#endif /* WANT_SINGLE_PRECISION_REAL */

#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#undef ACTIVATE_SKEW
#undef DEVICE_POINTER
#include "../general/precision_macros.h"
!>  \brief elpa_solve_evp_complex_2stage_host_arrays_double_impl: Fortran function to solve the double-precision complex eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param kernel                               specify ELPA2 kernel to use
!>  \param useGPU (optional)                    decide whether GPUs should be used or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
#include "elpa2_template.F90"
#undef COMPLEXCASE
#undef DOUBLE_PRECISION

#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#undef ACTIVATE_SKEW
#define DEVICE_POINTER
#include "../general/precision_macros.h"
!>  \brief elpa_solve_evp_complex_2stage_d_ptr_double_impl: Fortran function to solve the double-precision complex eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q                                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param kernel                               specify ELPA2 kernel to use
!>  \param useGPU (optional)                    decide whether GPUs should be used or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
#include "elpa2_template.F90"
#undef DEVICE_POINTER
#undef COMPLEXCASE
#undef DOUBLE_PRECISION

#ifdef WANT_SINGLE_PRECISION_COMPLEX

#define COMPLEXCASE 1
#define SINGLE_PRECISION 1
#undef ACTIVATE_SKEW
#undef DEVICE_POINTER
#include "../general/precision_macros.h"

!>  \brief elpa_solve_evp_complex_2stage_host_arrays_single_impl: Fortran function to solve the single-precision complex eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param kernel                               specify ELPA2 kernel to use
!>  \param useGPU (optional)                    decide whether GPUs should be used or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
#include "elpa2_template.F90"
#undef COMPLEXCASE
#undef SINGLE_PRECISION

#define COMPLEXCASE 1
#define SINGLE_PRECISION 1
#undef ACTIVATE_SKEW
#define DEVICE_POINTER
#include "../general/precision_macros.h"

!>  \brief elpa_solve_evp_complex_2stage_d_ptr_single_impl: Fortran function to solve the single-precision complex eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q                                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param kernel                               specify ELPA2 kernel to use
!>  \param useGPU (optional)                    decide whether GPUs should be used or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
#include "elpa2_template.F90"
#undef DEVICE_POINTER
#undef COMPLEXCASE
#undef SINGLE_PRECISION

#endif /* WANT_SINGLE_PRECISION_COMPLEX */

#ifdef HAVE_SKEWSYMMETRIC
#define REALCASE 1
#define DOUBLE_PRECISION 1
#define ACTIVATE_SKEW
#undef DEVICE_POINTER
#include "../general/precision_macros.h"
!-------------------------------------------------------------------------------
!>  \brief elpa_solve_skew_evp_real_2stage_host_arrays_double_impl: Fortran function to solve the double-precision real skew-symmetric eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param kernel                               specify ELPA2 kernel to use
!>
!>  \param useQR (optional)                     use QR decomposition
!>  \param useGPU (optional)                    decide whether to use GPUs or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
#include "elpa2_template.F90"
#undef REALCASE
#undef DOUBLE_PRECISION
#undef ACTIVATE_SKEW

#define REALCASE 1
#define DOUBLE_PRECISION 1
#define ACTIVATE_SKEW
#define DEVICE_POINTER
#include "../general/precision_macros.h"
!-------------------------------------------------------------------------------
!>  \brief elpa_solve_skew_evp_real_2stage_d_ptr_double_impl: Fortran function to solve the double-precision real skew-symmetric eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q                                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param kernel                               specify ELPA2 kernel to use
!>
!>  \param useQR (optional)                     use QR decomposition
!>  \param useGPU (optional)                    decide whether to use GPUs or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
#include "elpa2_template.F90"
#undef REALCASE
#undef DOUBLE_PRECISION
#undef ACTIVATE_SKEW
#undef DEVICE_POINTER

#ifdef WANT_SINGLE_PRECISION_REAL
#define REALCASE 1
#define SINGLE_PRECISION 1
#define ACTIVATE_SKEW
#undef DEVICE_POINTER
#include "../general/precision_macros.h"
!-------------------------------------------------------------------------------
!>  \brief elpa_solve_skew_evp_real_2stage_host_arrays_single_impl: Fortran function to solve the single-precision real skew-symmetric eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param kernel                               specify ELPA2 kernel to use
!>
!>  \param useQR (optional)                     use QR decomposition
!>  \param useGPU (optional)                    decide whether GPUs should be used or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
#include "elpa2_template.F90"
#undef REALCASE
#undef SINGLE_PRECISION
#undef ACTIVATE_SKEW

#define REALCASE 1
#define SINGLE_PRECISION 1
#define ACTIVATE_SKEW
#define DEVICE_POINTER
#include "../general/precision_macros.h"
!-------------------------------------------------------------------------------
!>  \brief elpa_solve_skew_evp_real_2stage_d_ptr_single_impl: Fortran function to solve the single-precision real skew-symmetric eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a                                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev                                   On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q                                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param kernel                               specify ELPA2 kernel to use
!>
!>  \param useQR (optional)                     use QR decomposition
!>  \param useGPU (optional)                    decide whether GPUs should be used or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
#include "elpa2_template.F90"
#undef REALCASE
#undef SINGLE_PRECISION
#undef ACTIVATE_SKEW
#undef DEVICE_POINTER

#endif /* WANT_SINGLE_PRECISION_REAL */

#endif /* HAVE_SKEWSYMMETRIC */


end module elpa2_impl
