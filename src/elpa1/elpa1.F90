!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
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

!> \mainpage
!> Eigenvalue SoLvers for Petaflop-Applications (ELPA)
!> \par
!> http://elpa.mpcdf.mpg.de
!>
!> \par
!>    The ELPA library was originally created by the ELPA consortium,
!>    consisting of the following organizations:
!>
!>    - Max Planck Computing and Data Facility (MPCDF) formerly known as
!>      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
!>    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!>      Informatik,
!>    - Technische Universität München, Lehrstuhl für Informatik mit
!>      Schwerpunkt Wissenschaftliches Rechnen ,
!>    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
!>    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
!>      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!>      and
!>    - IBM Deutschland GmbH
!>
!>   Some parts and enhancements of ELPA have been contributed and authored
!>   by the Intel Corporation which is not part of the ELPA consortium.
!>
!>   Contributions to the ELPA source have been authored by (in alphabetical order):
!>
!> \author T. Auckenthaler, Volker Blum, A. Heinecke, L. Huedepohl, R. Johanni, Werner Jürgens, and A. Marek


#include "config-f90.h"

!> \brief Fortran module which provides the routines to use the one-stage ELPA solver
module elpa1_impl
  use, intrinsic :: iso_c_binding
  use elpa_utilities
  use elpa1_auxiliary_impl
#ifdef HAVE_LIKWID
  use likwid
#endif

  implicit none

  ! The following routines are public:
  private

  public :: elpa_solve_evp_real_1stage_a_h_a_double_impl    !< Driver routine for real double-precision 1-stage eigenvalue problem

#ifdef WANT_SINGLE_PRECISION_REAL
  public :: elpa_solve_evp_real_1stage_a_h_a_single_impl    !< Driver routine for real single-precision 1-stage eigenvalue problem

#endif
  public :: elpa_solve_evp_complex_1stage_a_h_a_double_impl !< Driver routine for complex 1-stage eigenvalue problem
#ifdef WANT_SINGLE_PRECISION_COMPLEX
  public :: elpa_solve_evp_complex_1stage_a_h_a_single_impl !< Driver routine for complex 1-stage eigenvalue problem
#endif

#ifdef HAVE_SKEWSYMMETRIC
  public :: elpa_solve_skew_evp_real_1stage_a_h_a_double_impl    !< Driver routine for real double-precision 1-stage skew-symmetric eigenvalue problem

#ifdef WANT_SINGLE_PRECISION_REAL
  public :: elpa_solve_skew_evp_real_1stage_a_h_a_single_impl    !< Driver routine for real single-precision 1-stage skew-symmetric eigenvalue problem

#endif
#endif /* HAVE_SKEWSYMMETRIC */

  public :: elpa_solve_evp_real_1stage_d_ptr_double_impl    !< Driver routine for real double-precision 1-stage eigenvalue problem

#ifdef WANT_SINGLE_PRECISION_REAL
  public :: elpa_solve_evp_real_1stage_d_ptr_single_impl    !< Driver routine for real single-precision 1-stage eigenvalue problem

#endif
  public :: elpa_solve_evp_complex_1stage_d_ptr_double_impl !< Driver routine for complex 1-stage eigenvalue problem
#ifdef WANT_SINGLE_PRECISION_COMPLEX
  public :: elpa_solve_evp_complex_1stage_d_ptr_single_impl !< Driver routine for complex 1-stage eigenvalue problem
#endif

#ifdef HAVE_SKEWSYMMETRIC
  public :: elpa_solve_skew_evp_real_1stage_d_ptr_double_impl    !< Driver routine for real double-precision 1-stage skew-symmetric eigenvalue problem

#ifdef WANT_SINGLE_PRECISION_REAL
  public :: elpa_solve_skew_evp_real_1stage_d_ptr_single_impl    !< Driver routine for real single-precision 1-stage skew-symmetric eigenvalue problem

#endif
#endif /* HAVE_SKEWSYMMETRIC */



  ! imported from elpa1_auxilliary

  !public :: elpa_mult_at_b_real_double_impl       !< Multiply double-precision real matrices A**T * B

  !public :: elpa_mult_ah_b_complex_double_impl    !< Multiply double-precision complex matrices A**H * B

  !public :: elpa_invert_trm_real_double_impl      !< Invert double-precision real triangular matrix

  !public :: elpa_invert_trm_complex_double_impl   !< Invert double-precision complex triangular matrix

  !public :: elpa_cholesky_real_double_impl        !< Cholesky factorization of a double-precision real matrix

  !public :: elpa_cholesky_complex_double_impl     !< Cholesky factorization of a double-precision complex matrix

  public :: elpa_solve_tridi_double_impl          !< Solve a double-precision tridiagonal eigensystem with divide and conquer method

#ifdef WANT_SINGLE_PRECISION_REAL
  !public :: elpa_mult_at_b_real_single_impl       !< Multiply single-precision real matrices A**T * B
  !public :: elpa_invert_trm_real_single_impl      !< Invert single-precision real triangular matrix
  !public :: elpa_cholesky_real_single_impl        !< Cholesky factorization of a single-precision real matrix
  public :: elpa_solve_tridi_single_impl          !< Solve a single-precision tridiagonal eigensystem with divide and conquer method
#endif

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  !public :: elpa_mult_ah_b_complex_single_impl    !< Multiply single-precision complex matrices A**H * B
  !public :: elpa_invert_trm_complex_single_impl   !< Invert single-precision complex triangular matrix
  !public :: elpa_cholesky_complex_single_impl     !< Cholesky factorization of a single-precision complex matrix
#endif

contains


!> \brief elpa_solve_evp_real_1stage_host_arrays_double_impl: Fortran function to solve the real double-precision eigenvalue problem with 1-stage solver
!>
!> \details
!> \param  obj                      elpa_t object contains:
!> \param     - obj%na              Order of matrix
!> \param     - obj%nev             number of eigenvalues/vectors to be computed
!>                                  The smallest nev eigenvalues/eigenvectors are calculated.
!> \param     - obj%local_nrows     Leading dimension of a
!> \param     - obj%local_ncols     local columns of matrix q
!> \param     - obj%nblk            blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows   MPI communicator for rows
!> \param     - obj%mpi_comm_cols   MPI communicator for columns
!> \param     - obj%mpi_comm_parent MPI communicator for columns
!> \param     - obj%gpu             use GPU version (1 or 0)
!>
!> \param  a(lda,matrixCols)        Distributed matrix for which eigenvalues are to be computed.
!>                                  Distribution is like in Scalapack.
!>                                  The full matrix must be set (not only one half like in scalapack).
!>                                  Destroyed on exit (upper and lower half).
!>
!>  \param ev(na)                   On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)        On output: Eigenvectors of a
!>                                  Distribution is like in Scalapack.
!>                                  Must be always dimensioned to the full size (corresponding to (na,na))
!>                                  even if only a part of the eigenvalues is needed.
!>
!>
!>  \result                       success
#define REALCASE 1
#define DOUBLE_PRECISION 1
#undef ACTIVATE_SKEW
#undef DEVICE_POINTER
#include "../general/precision_macros.h"
#include "elpa1_template.F90"
#undef REALCASE
#undef DOUBLE_PRECISION


!> \brief elpa_solve_evp_real_1stage_d_ptr_double_impl: Fortran function to solve the real double-precision eigenvalue problem with 1-stage solver
!>
!> \details
!> \param  obj                      elpa_t object contains:
!> \param     - obj%na              Order of matrix
!> \param     - obj%nev             number of eigenvalues/vectors to be computed
!>                                  The smallest nev eigenvalues/eigenvectors are calculated.
!> \param     - obj%local_nrows     Leading dimension of a
!> \param     - obj%local_ncols     local columns of matrix q
!> \param     - obj%nblk            blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows   MPI communicator for rows
!> \param     - obj%mpi_comm_cols   MPI communicator for columns
!> \param     - obj%mpi_comm_parent MPI communicator for columns
!> \param     - obj%gpu             use GPU version (1 or 0)
!>
!> \param  a                        Distributed matrix for which eigenvalues are to be computed.
!>                                  Distribution is like in Scalapack.
!>                                  The full matrix must be set (not only one half like in scalapack).
!>                                  Destroyed on exit (upper and lower half).
!>
!>  \param ev                       On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q                        On output: Eigenvectors of a
!>                                  Distribution is like in Scalapack.
!>                                  Must be always dimensioned to the full size (corresponding to (na,na))
!>                                  even if only a part of the eigenvalues is needed.
!>
!>
!>  \result                       success
#define REALCASE 1
#define DOUBLE_PRECISION 1
#undef ACTIVATE_SKEW
#define DEVICE_POINTER
#include "../general/precision_macros.h"
#include "elpa1_template.F90"
#undef DEVICE_POINTER
#undef REALCASE
#undef DOUBLE_PRECISION


#ifdef WANT_SINGLE_PRECISION_REAL
!> \brief elpa_solve_evp_real_1stage_host_arrays_single_impl: Fortran function to solve the real single-precision eigenvalue problem with 1-stage solver
!> \details
!> \param  obj                      elpa_t object contains:
!> \param     - obj%na              Order of matrix
!> \param     - obj%nev             number of eigenvalues/vectors to be computed
!>                                  The smallest nev eigenvalues/eigenvectors are calculated.
!> \param     - obj%local_nrows     Leading dimension of a
!> \param     - obj%local_ncols     local columns of matrix q
!> \param     - obj%nblk            blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows   MPI communicator for rows
!> \param     - obj%mpi_comm_cols   MPI communicator for columns
!> \param     - obj%mpi_comm_parent MPI communicator for columns
!> \param     - obj%gpu             use GPU version (1 or 0)
!>
!> \param  a(lda,matrixCols)        Distributed matrix for which eigenvalues are to be computed.
!>                                  Distribution is like in Scalapack.
!>                                  The full matrix must be set (not only one half like in scalapack).
!>                                  Destroyed on exit (upper and lower half).
!>
!>  \param ev(na)                   On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)        On output: Eigenvectors of a
!>                                  Distribution is like in Scalapack.
!>                                  Must be always dimensioned to the full size (corresponding to (na,na))
!>                                  even if only a part of the eigenvalues is needed.
!>
!>
!>  \result                       success

#define REALCASE 1
#define SINGLE_PRECISION 1
#undef ACTIVATE_SKEW
#undef DEVICE_POINTER
#include "../general/precision_macros.h"
#include "elpa1_template.F90"
#undef REALCASE
#undef SINGLE_PRECISION

!> \brief elpa_solve_evp_real_1stage_d_ptr_single_impl: Fortran function to solve the real single-precision eigenvalue problem with 1-stage solver
!> \details
!> \param  obj                      elpa_t object contains:
!> \param     - obj%na              Order of matrix
!> \param     - obj%nev             number of eigenvalues/vectors to be computed
!>                                  The smallest nev eigenvalues/eigenvectors are calculated.
!> \param     - obj%local_nrows     Leading dimension of a
!> \param     - obj%local_ncols     local columns of matrix q
!> \param     - obj%nblk            blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows   MPI communicator for rows
!> \param     - obj%mpi_comm_cols   MPI communicator for columns
!> \param     - obj%mpi_comm_parent MPI communicator for columns
!> \param     - obj%gpu             use GPU version (1 or 0)
!>
!> \param  a                        Distributed matrix for which eigenvalues are to be computed.
!>                                  Distribution is like in Scalapack.
!>                                  The full matrix must be set (not only one half like in scalapack).
!>                                  Destroyed on exit (upper and lower half).
!>
!>  \param ev                       On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q                        On output: Eigenvectors of a
!>                                  Distribution is like in Scalapack.
!>                                  Must be always dimensioned to the full size (corresponding to (na,na))
!>                                  even if only a part of the eigenvalues is needed.
!>
!>
!>  \result                       success

#define REALCASE 1
#define SINGLE_PRECISION 1
#undef ACTIVATE_SKEW
#define DEVICE_POINTER
#include "../general/precision_macros.h"
#include "elpa1_template.F90"
#undef DEVICE_POINTER
#undef REALCASE
#undef SINGLE_PRECISION

#endif /* WANT_SINGLE_PRECISION_REAL */

!> \brief elpa_solve_evp_complex_1stage_host_arrays_double_impl: Fortran function to solve the complex double-precision eigenvalue problem with 1-stage solver
!> \details
!> \param  obj                      elpa_t object contains:
!> \param     - obj%na              Order of matrix
!> \param     - obj%nev             number of eigenvalues/vectors to be computed
!>                                  The smallest nev eigenvalues/eigenvectors are calculated.
!> \param     - obj%local_nrows     Leading dimension of a
!> \param     - obj%local_ncols     local columns of matrix q
!> \param     - obj%nblk            blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows   MPI communicator for rows
!> \param     - obj%mpi_comm_cols   MPI communicator for columns
!> \param     - obj%mpi_comm_parent MPI communicator for columns
!> \param     - obj%gpu             use GPU version (1 or 0)
!>
!> \param  a(lda,matrixCols)        Distributed matrix for which eigenvalues are to be computed.
!>                                  Distribution is like in Scalapack.
!>                                  The full matrix must be set (not only one half like in scalapack).
!>                                  Destroyed on exit (upper and lower half).
!>
!>  \param ev(na)                   On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)        On output: Eigenvectors of a
!>                                  Distribution is like in Scalapack.
!>                                  Must be always dimensioned to the full size (corresponding to (na,na))
!>                                  even if only a part of the eigenvalues is needed.
!>
!>
!>  \result                       success
#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#undef ACTIVATE_SKEW
#undef DEVICE_POINTER
#include "../general/precision_macros.h"
#include "elpa1_template.F90"
#undef DOUBLE_PRECISION
#undef COMPLEXCASE

!> \brief elpa_solve_evp_complex_1stage_d_ptr_double_impl: Fortran function to solve the complex double-precision eigenvalue problem with 1-stage solver
!> \details
!> \param  obj                      elpa_t object contains:
!> \param     - obj%na              Order of matrix
!> \param     - obj%nev             number of eigenvalues/vectors to be computed
!>                                  The smallest nev eigenvalues/eigenvectors are calculated.
!> \param     - obj%local_nrows     Leading dimension of a
!> \param     - obj%local_ncols     local columns of matrix q
!> \param     - obj%nblk            blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows   MPI communicator for rows
!> \param     - obj%mpi_comm_cols   MPI communicator for columns
!> \param     - obj%mpi_comm_parent MPI communicator for columns
!> \param     - obj%gpu             use GPU version (1 or 0)
!>
!> \param  a                        Distributed matrix for which eigenvalues are to be computed.
!>                                  Distribution is like in Scalapack.
!>                                  The full matrix must be set (not only one half like in scalapack).
!>                                  Destroyed on exit (upper and lower half).
!>
!>  \param ev                       On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q                        On output: Eigenvectors of a
!>                                  Distribution is like in Scalapack.
!>                                  Must be always dimensioned to the full size (corresponding to (na,na))
!>                                  even if only a part of the eigenvalues is needed.
!>
!>
!>  \result                       success
#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#undef ACTIVATE_SKEW
#define DEVICE_POINTER
#include "../general/precision_macros.h"
#include "elpa1_template.F90"
#undef DEVICE_POINTER
#undef DOUBLE_PRECISION
#undef COMPLEXCASE


#ifdef WANT_SINGLE_PRECISION_COMPLEX

!> \brief elpa_solve_evp_complex_1stage_host_arrays_single_impl: Fortran function to solve the complex single-precision eigenvalue problem with 1-stage solver
!> \details
!> \param  obj                      elpa_t object contains:
!> \param     - obj%na              Order of matrix
!> \param     - obj%nev             number of eigenvalues/vectors to be computed
!>                                  The smallest nev eigenvalues/eigenvectors are calculated.
!> \param     - obj%local_nrows     Leading dimension of a
!> \param     - obj%local_ncols     local columns of matrix q
!> \param     - obj%nblk            blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows   MPI communicator for rows
!> \param     - obj%mpi_comm_cols   MPI communicator for columns
!> \param     - obj%mpi_comm_parent MPI communicator for columns
!> \param     - obj%gpu             use GPU version (1 or 0)
!>
!> \param  a(lda,matrixCols)        Distributed matrix for which eigenvalues are to be computed.
!>                                  Distribution is like in Scalapack.
!>                                  The full matrix must be set (not only one half like in scalapack).
!>                                  Destroyed on exit (upper and lower half).
!>
!>  \param ev(na)                   On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)        On output: Eigenvectors of a
!>                                  Distribution is like in Scalapack.
!>                                  Must be always dimensioned to the full size (corresponding to (na,na))
!>                                  even if only a part of the eigenvalues is needed.
!>
!>
!>  \result                       success

#define COMPLEXCASE 1
#define SINGLE_PRECISION
#undef ACTIVATE_SKEW
#undef DEVICE_POINTER
#include "../general/precision_macros.h"
#include "elpa1_template.F90"
#undef COMPLEXCASE
#undef SINGLE_PRECISION

!> \brief elpa_solve_evp_complex_1stage_d_ptr_single_impl: Fortran function to solve the complex single-precision eigenvalue problem with 1-stage solver
!> \details
!> \param  obj                      elpa_t object contains:
!> \param     - obj%na              Order of matrix
!> \param     - obj%nev             number of eigenvalues/vectors to be computed
!>                                  The smallest nev eigenvalues/eigenvectors are calculated.
!> \param     - obj%local_nrows     Leading dimension of a
!> \param     - obj%local_ncols     local columns of matrix q
!> \param     - obj%nblk            blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows   MPI communicator for rows
!> \param     - obj%mpi_comm_cols   MPI communicator for columns
!> \param     - obj%mpi_comm_parent MPI communicator for columns
!> \param     - obj%gpu             use GPU version (1 or 0)
!>
!> \param  a                        Distributed matrix for which eigenvalues are to be computed.
!>                                  Distribution is like in Scalapack.
!>                                  The full matrix must be set (not only one half like in scalapack).
!>                                  Destroyed on exit (upper and lower half).
!>
!>  \param ev                       On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q                        On output: Eigenvectors of a
!>                                  Distribution is like in Scalapack.
!>                                  Must be always dimensioned to the full size (corresponding to (na,na))
!>                                  even if only a part of the eigenvalues is needed.
!>
!>
!>  \result                       success

#define COMPLEXCASE 1
#define SINGLE_PRECISION
#undef ACTIVATE_SKEW
#define DEVICE_POINTER
#include "../general/precision_macros.h"
#include "elpa1_template.F90"
#undef DEVICE_POINTER
#undef COMPLEXCASE
#undef SINGLE_PRECISION
#endif /* WANT_SINGLE_PRECISION_COMPLEX */


#ifdef HAVE_SKEWSYMMETRIC
!> \brief elpa_solve_skew_evp_real_1stage_host_arrays_double_impl: Fortran function to solve the real double-precision skew-symmetric eigenvalue problem with 1-stage solver
!>
!> \details
!> \param  obj                      elpa_t object contains:
!> \param     - obj%na              Order of matrix
!> \param     - obj%nev             number of eigenvalues/vectors to be computed
!>                                  The smallest nev eigenvalues/eigenvectors are calculated.
!> \param     - obj%local_nrows     Leading dimension of a
!> \param     - obj%local_ncols     local columns of matrix q
!> \param     - obj%nblk            blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows   MPI communicator for rows
!> \param     - obj%mpi_comm_cols   MPI communicator for columns
!> \param     - obj%mpi_comm_parent MPI communicator for columns
!> \param     - obj%gpu             use GPU version (1 or 0)
!>
!> \param  a(lda,matrixCols)        Distributed matrix for which eigenvalues are to be computed.
!>                                  Distribution is like in Scalapack.
!>                                  The full matrix must be set (not only one half like in scalapack).
!>                                  Destroyed on exit (upper and lower half).
!>
!>  \param ev(na)                   On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)        On output: Eigenvectors of a
!>                                  Distribution is like in Scalapack.
!>                                  Must be always dimensioned to the full size (corresponding to (na,na))
!>                                  even if only a part of the eigenvalues is needed.
!>
!>
!>  \result                       success
#define REALCASE 1
#define DOUBLE_PRECISION 1
#define ACTIVATE_SKEW
#undef DEVICE_POINTER
#include "../general/precision_macros.h"
#include "elpa1_template.F90"
#undef ACTIVATE_SKEW
#undef REALCASE
#undef DOUBLE_PRECISION

!> \brief elpa_solve_skew_evp_real_1stage_d_ptr_double_impl: Fortran function to solve the real double-precision skew-symmetric eigenvalue problem with 1-stage solver
!>
!> \details
!> \param  obj                      elpa_t object contains:
!> \param     - obj%na              Order of matrix
!> \param     - obj%nev             number of eigenvalues/vectors to be computed
!>                                  The smallest nev eigenvalues/eigenvectors are calculated.
!> \param     - obj%local_nrows     Leading dimension of a
!> \param     - obj%local_ncols     local columns of matrix q
!> \param     - obj%nblk            blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows   MPI communicator for rows
!> \param     - obj%mpi_comm_cols   MPI communicator for columns
!> \param     - obj%mpi_comm_parent MPI communicator for columns
!> \param     - obj%gpu             use GPU version (1 or 0)
!>
!> \param  a                        Distributed matrix for which eigenvalues are to be computed.
!>                                  Distribution is like in Scalapack.
!>                                  The full matrix must be set (not only one half like in scalapack).
!>                                  Destroyed on exit (upper and lower half).
!>
!>  \param ev                       On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q                        On output: Eigenvectors of a
!>                                  Distribution is like in Scalapack.
!>                                  Must be always dimensioned to the full size (corresponding to (na,na))
!>                                  even if only a part of the eigenvalues is needed.
!>
!>
!>  \result                       success
#define REALCASE 1
#define DOUBLE_PRECISION 1
#define ACTIVATE_SKEW
#define DEVICE_POINTER
#include "../general/precision_macros.h"
#include "elpa1_template.F90"
#undef ACTIVATE_SKEW
#undef DEVICE_POINTER
#undef REALCASE
#undef DOUBLE_PRECISION

#ifdef WANT_SINGLE_PRECISION_REAL
!> \brief elpa_solve_evp_real_1stage_host_arrays_single_impl: Fortran function to solve the real single-precision eigenvalue problem with 1-stage solver
!> \details
!> \param  obj                      elpa_t object contains:
!> \param     - obj%na              Order of matrix
!> \param     - obj%nev             number of eigenvalues/vectors to be computed
!>                                  The smallest nev eigenvalues/eigenvectors are calculated.
!> \param     - obj%local_nrows     Leading dimension of a
!> \param     - obj%local_ncols     local columns of matrix q
!> \param     - obj%nblk            blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows   MPI communicator for rows
!> \param     - obj%mpi_comm_cols   MPI communicator for columns
!> \param     - obj%mpi_comm_parent MPI communicator for columns
!> \param     - obj%gpu             use GPU version (1 or 0)
!>
!> \param  a(lda,matrixCols)        Distributed matrix for which eigenvalues are to be computed.
!>                                  Distribution is like in Scalapack.
!>                                  The full matrix must be set (not only one half like in scalapack).
!>                                  Destroyed on exit (upper and lower half).
!>
!>  \param ev(na)                   On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)        On output: Eigenvectors of a
!>                                  Distribution is like in Scalapack.
!>                                  Must be always dimensioned to the full size (corresponding to (na,na))
!>                                  even if only a part of the eigenvalues is needed.
!>
!>
!>  \result                       success

#define REALCASE 1
#define SINGLE_PRECISION 1
#define ACTIVATE_SKEW
#undef DEVICE_POINTER
#include "../general/precision_macros.h"
#include "elpa1_template.F90"
#undef REALCASE
#undef ACTIVATE_SKEW
#undef SINGLE_PRECISION

!> \brief elpa_solve_evp_real_1stage_d_ptr_single_impl: Fortran function to solve the real single-precision eigenvalue problem with 1-stage solver
!> \details
!> \param  obj                      elpa_t object contains:
!> \param     - obj%na              Order of matrix
!> \param     - obj%nev             number of eigenvalues/vectors to be computed
!>                                  The smallest nev eigenvalues/eigenvectors are calculated.
!> \param     - obj%local_nrows     Leading dimension of a
!> \param     - obj%local_ncols     local columns of matrix q
!> \param     - obj%nblk            blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows   MPI communicator for rows
!> \param     - obj%mpi_comm_cols   MPI communicator for columns
!> \param     - obj%mpi_comm_parent MPI communicator for columns
!> \param     - obj%gpu             use GPU version (1 or 0)
!>
!> \param  a                        Distributed matrix for which eigenvalues are to be computed.
!>                                  Distribution is like in Scalapack.
!>                                  The full matrix must be set (not only one half like in scalapack).
!>                                  Destroyed on exit (upper and lower half).
!>
!>  \param ev                       On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q                        On output: Eigenvectors of a
!>                                  Distribution is like in Scalapack.
!>                                  Must be always dimensioned to the full size (corresponding to (na,na))
!>                                  even if only a part of the eigenvalues is needed.
!>
!>
!>  \result                       success

#define REALCASE 1
#define SINGLE_PRECISION 1
#define ACTIVATE_SKEW
#define DEVICE_POINTER
#include "../general/precision_macros.h"
#include "elpa1_template.F90"
#undef REALCASE
#undef DEVICE_POINTER
#undef ACTIVATE_SKEW
#undef SINGLE_PRECISION
#endif /* WANT_SINGLE_PRECISION_REAL */

#endif /* HAVE_SKEWSYMMETRIC */

end module ELPA1_impl
