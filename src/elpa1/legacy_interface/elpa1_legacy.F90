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
module ELPA1_legacy
  use, intrinsic :: iso_c_binding
  use elpa_utilities
  use elpa1_auxiliary_legacy
  use elpa1_utilities

  implicit none

  ! The following routines are public:
  private

  public :: get_elpa_row_col_comms               !< old, deprecated interface, will be deleted. Use elpa_get_communicators instead
  public :: get_elpa_communicators               !< Sets MPI row/col communicators; OLD and deprecated interface, will be deleted. Use elpa_get_communicators instead
  public :: elpa_get_communicators               !< Sets MPI row/col communicators as needed by ELPA

  public :: solve_evp_real                       !< old, deprecated interface: Driver routine for real double-precision eigenvalue problem DO NOT USE. Will be deleted at some point
  public :: elpa_solve_evp_real_1stage_double    !< Driver routine for real double-precision 1-stage eigenvalue problem

  public :: solve_evp_real_1stage                !< Driver routine for real double-precision eigenvalue problem
  public :: solve_evp_real_1stage_double         !< Driver routine for real double-precision eigenvalue problem
#ifdef WANT_SINGLE_PRECISION_REAL
  public :: solve_evp_real_1stage_single         !< Driver routine for real single-precision eigenvalue problem
  public :: elpa_solve_evp_real_1stage_single    !< Driver routine for real single-precision 1-stage eigenvalue problem

#endif
  public :: solve_evp_complex                    !< old, deprecated interface:  Driver routine for complex double-precision eigenvalue problem DO NOT USE. Will be deleted at some point
  public :: elpa_solve_evp_complex_1stage_double !< Driver routine for complex 1-stage eigenvalue problem
  public :: solve_evp_complex_1stage             !< Driver routine for complex double-precision eigenvalue problem
  public :: solve_evp_complex_1stage_double      !< Driver routine for complex double-precision eigenvalue problem
#ifdef WANT_SINGLE_PRECISION_COMPLEX
  public :: solve_evp_complex_1stage_single      !< Driver routine for complex single-precision eigenvalue problem
  public :: elpa_solve_evp_complex_1stage_single !< Driver routine for complex 1-stage eigenvalue problem
#endif

  ! imported from elpa1_auxilliary

  public :: elpa_mult_at_b_real_double       !< Multiply double-precision real matrices A**T * B
  public :: mult_at_b_real                   !< old, deprecated interface to multiply double-precision real matrices A**T * B  DO NOT USE

  public :: elpa_mult_ah_b_complex_double    !< Multiply double-precision complex matrices A**H * B
  public :: mult_ah_b_complex                !< old, deprecated interface to multiply double-preicion complex matrices A**H * B  DO NOT USE

  public :: elpa_invert_trm_real_double      !< Invert double-precision real triangular matrix
  public :: invert_trm_real                  !< old, deprecated interface to invert double-precision real triangular matrix  DO NOT USE

  public :: elpa_invert_trm_complex_double   !< Invert double-precision complex triangular matrix
  public :: invert_trm_complex               !< old, deprecated interface to invert double-precision complex triangular matrix  DO NOT USE

  public :: elpa_cholesky_real_double        !< Cholesky factorization of a double-precision real matrix
  public :: cholesky_real                    !< old, deprecated interface to do Cholesky factorization of a double-precision real matrix  DO NOT USE

  public :: elpa_cholesky_complex_double     !< Cholesky factorization of a double-precision complex matrix
  public :: cholesky_complex                 !< old, deprecated interface to do Cholesky factorization of a double-precision complex matrix  DO NOT USE

  public :: elpa_solve_tridi_double          !< Solve a double-precision tridiagonal eigensystem with divide and conquer method

#ifdef WANT_SINGLE_PRECISION_REAL
  public :: elpa_mult_at_b_real_single       !< Multiply single-precision real matrices A**T * B
  public :: elpa_invert_trm_real_single      !< Invert single-precision real triangular matrix
  public :: elpa_cholesky_real_single        !< Cholesky factorization of a single-precision real matrix
  public :: elpa_solve_tridi_single          !< Solve a single-precision tridiagonal eigensystem with divide and conquer method
#endif

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  public :: elpa_mult_ah_b_complex_single    !< Multiply single-precision complex matrices A**H * B
  public :: elpa_invert_trm_complex_single   !< Invert single-precision complex triangular matrix
  public :: elpa_cholesky_complex_single     !< Cholesky factorization of a single-precision complex matrix
#endif

  ! Timing results, set by every call to solve_evp_xxx

  real(kind=c_double), public :: time_evp_fwd    !< time for forward transformations (to tridiagonal form)
  real(kind=c_double), public :: time_evp_solve  !< time for solving the tridiagonal system
  real(kind=c_double), public :: time_evp_back   !< time for back transformations of eigenvectors

  logical, public :: elpa_print_times = .false. !< Set elpa_print_times to .true. for explicit timing outputs


!> \brief get_elpa_row_col_comms:  old, deprecated interface, will be deleted. Use "elpa_get_communicators"
!> \details
!> The interface and variable definition is the same as in "elpa_get_communicators"
!> \param  mpi_comm_global   Global communicator for the calculations (in)
!>
!> \param  my_prow           Row coordinate of the calling process in the process grid (in)
!>
!> \param  my_pcol           Column coordinate of the calling process in the process grid (in)
!>
!> \param  mpi_comm_rows     Communicator for communicating within rows of processes (out)
!>
!> \param  mpi_comm_cols     Communicator for communicating within columns of processes (out)
!> \result mpierr            integer error value of mpi_comm_split function
  interface get_elpa_row_col_comms
    module procedure elpa_get_communicators
  end interface

!> \brief elpa_get_communicators:  Fortran interface to set the communicators needed by ELPA
!> \details
!> The interface and variable definition is the same as in "elpa_get_communicators"
!> \param  mpi_comm_global   Global communicator for the calculations (in)
!>
!> \param  my_prow           Row coordinate of the calling process in the process grid (in)
!>
!> \param  my_pcol           Column coordinate of the calling process in the process grid (in)
!>
!> \param  mpi_comm_rows     Communicator for communicating within rows of processes (out)
!>
!> \param  mpi_comm_cols     Communicator for communicating within columns of processes (out)
!> \result mpierr            integer error value of mpi_comm_split function

  interface get_elpa_communicators
    module procedure elpa_get_communicators
  end interface

!> \brief solve_evp_real: old, deprecated Fortran function to solve the real eigenvalue problem with 1-stage solver. Will be deleted at some point. Better use "solve_evp_real_1stage" or "elpa_solve_evp_real"
!>
!> \details
!>  The interface and variable definition is the same as in "elpa_solve_evp_real_1stage_double"
!  Parameters
!
!> \param  na                   Order of matrix a
!>
!> \param  nev                  Number of eigenvalues needed.
!>                              The smallest nev eigenvalues/eigenvectors are calculated.
!>
!> \param  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!>                              Distribution is like in Scalapack.
!>                              The full matrix must be set (not only one half like in scalapack).
!>                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                  Leading dimension of a
!>
!>  \param ev(na)               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)    On output: Eigenvectors of a
!>                              Distribution is like in Scalapack.
!>                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                  Leading dimension of q
!>
!>  \param nblk                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols           distributed number of matrix columns
!>
!>  \param mpi_comm_rows        MPI-Communicator for rows
!>  \param mpi_comm_cols        MPI-Communicator for columns
!>
!>  \result                     success


  interface solve_evp_real
    module procedure elpa_solve_evp_real_1stage_double
  end interface

  interface solve_evp_real_1stage
    module procedure elpa_solve_evp_real_1stage_double
  end interface

!> \brief elpa_solve_evp_real_1stage_double: Fortran function to solve the real eigenvalue problem with 1-stage solver. This is called by "elpa_solve_evp_real"
!>
!  Parameters
!
!> \param  na                   Order of matrix a
!>
!> \param  nev                  Number of eigenvalues needed.
!>                              The smallest nev eigenvalues/eigenvectors are calculated.
!>
!> \param  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!>                              Distribution is like in Scalapack.
!>                              The full matrix must be set (not only one half like in scalapack).
!>                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                  Leading dimension of a
!>
!>  \param ev(na)               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)    On output: Eigenvectors of a
!>                              Distribution is like in Scalapack.
!>                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                  Leading dimension of q
!>
!>  \param nblk                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols           distributed number of matrix columns
!>
!>  \param mpi_comm_rows        MPI-Communicator for rows
!>  \param mpi_comm_cols        MPI-Communicator for columns
!>
!>  \result                     success
  !interface elpa_solve_evp_real_1stage_double
   ! module procedure solve_evp_real_1stage_double
  !end interface

  interface solve_evp_real_1stage_double
    module procedure elpa_solve_evp_real_1stage_double
  end interface


!> \brief solve_evp_complex: old, deprecated Fortran function to solve the complex eigenvalue problem with 1-stage solver. will be deleted at some point. Better use "solve_evp_complex_1stage" or "elpa_solve_evp_complex"
!>
!> \details
!> The interface and variable definition is the same as in "elpa_solve_evp_complex_1stage_double"
!  Parameters
!
!> \param  na                   Order of matrix a
!>
!> \param  nev                  Number of eigenvalues needed.
!>                              The smallest nev eigenvalues/eigenvectors are calculated.
!>
!> \param  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!>                              Distribution is like in Scalapack.
!>                              The full matrix must be set (not only one half like in scalapack).
!>                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                  Leading dimension of a
!>
!>  \param ev(na)               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)    On output: Eigenvectors of a
!>                              Distribution is like in Scalapack.
!>                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                  Leading dimension of q
!>
!>  \param nblk                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols           distributed number of matrix columns
!>
!>  \param mpi_comm_rows        MPI-Communicator for rows
!>  \param mpi_comm_cols        MPI-Communicator for columns
!>
!>  \result                     success
  interface solve_evp_complex
    module procedure elpa_solve_evp_complex_1stage_double
  end interface
!> \brief solve_evp_complex: old, deprecated Fortran function to solve the complex eigenvalue problem with 1-stage solver. will be deleted at some point. Better use "solve_evp_complex_1stage" or "elpa_solve_evp_complex"
!>
!> \details
!> The interface and variable definition is the same as in "elpa_solve_evp_complex_1stage_double"
!  Parameters
!
!> \param  na                   Order of matrix a
!>
!> \param  nev                  Number of eigenvalues needed.
!>                              The smallest nev eigenvalues/eigenvectors are calculated.
!>
!> \param  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!>                              Distribution is like in Scalapack.
!>                              The full matrix must be set (not only one half like in scalapack).
!>                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                  Leading dimension of a
!>
!>  \param ev(na)               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)    On output: Eigenvectors of a
!>                              Distribution is like in Scalapack.
!>                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                  Leading dimension of q
!>
!>  \param nblk                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols           distributed number of matrix columns
!>
!>  \param mpi_comm_rows        MPI-Communicator for rows
!>  \param mpi_comm_cols        MPI-Communicator for columns
!>
!>  \result                     success
  interface solve_evp_complex_1stage
    module procedure elpa_solve_evp_complex_1stage_double
  end interface

!> \brief solve_evp_complex_1stage_double: Fortran function to solve the complex eigenvalue problem with 1-stage solver. This is called by "elpa_solve_evp_complex"
!>
!  Parameters
!
!> \param  na                   Order of matrix a
!>
!> \param  nev                  Number of eigenvalues needed.
!>                              The smallest nev eigenvalues/eigenvectors are calculated.
!>
!> \param  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!>                              Distribution is like in Scalapack.
!>                              The full matrix must be set (not only one half like in scalapack).
!>                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                  Leading dimension of a
!>
!>  \param ev(na)               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)    On output: Eigenvectors of a
!>                              Distribution is like in Scalapack.
!>                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                  Leading dimension of q
!>
!>  \param nblk                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols           distributed number of matrix columns
!>
!>  \param mpi_comm_rows        MPI-Communicator for rows
!>  \param mpi_comm_cols        MPI-Communicator for columns
!>
!>  \result                     success
  interface solve_evp_complex_1stage_double
    module procedure elpa_solve_evp_complex_1stage_double
  end interface

#ifdef WANT_SINGLE_PRECISION_REAL
!> \brief solve_evp_real_1stage_single: Fortran function to solve the real single-precision eigenvalue problem with 1-stage solver
!>
!  Parameters
!
!> \param  na                   Order of matrix a
!>
!> \param  nev                  Number of eigenvalues needed.
!>                              The smallest nev eigenvalues/eigenvectors are calculated.
!>
!> \param  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!>                              Distribution is like in Scalapack.
!>                              The full matrix must be set (not only one half like in scalapack).
!>                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                  Leading dimension of a
!>
!>  \param ev(na)               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)    On output: Eigenvectors of a
!>                              Distribution is like in Scalapack.
!>                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                  Leading dimension of q
!>
!>  \param nblk                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols           distributed number of matrix columns
!>
!>  \param mpi_comm_rows        MPI-Communicator for rows
!>  \param mpi_comm_cols        MPI-Communicator for columns
!>
!>  \result                     success

  interface solve_evp_real_1stage_single
    module procedure elpa_solve_evp_real_1stage_single
  end interface
#endif

#ifdef WANT_SINGLE_PRECISION_COMPLEX
!> \brief solve_evp_complex_1stage_single: Fortran function to solve the complex single-precision eigenvalue problem with 1-stage solver
!>
!  Parameters
!
!> \param  na                   Order of matrix a
!>
!> \param  nev                  Number of eigenvalues needed.
!>                              The smallest nev eigenvalues/eigenvectors are calculated.
!>
!> \param  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!>                              Distribution is like in Scalapack.
!>                              The full matrix must be set (not only one half like in scalapack).
!>                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                  Leading dimension of a
!>
!>  \param ev(na)               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)    On output: Eigenvectors of a
!>                              Distribution is like in Scalapack.
!>                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                  Leading dimension of q
!>
!>  \param nblk                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols           distributed number of matrix columns
!>
!>  \param mpi_comm_rows        MPI-Communicator for rows
!>  \param mpi_comm_cols        MPI-Communicator for columns
!>  \param mpi_comm_all         global MPI communicator
!>  \param useGPU
!>
!>  \result                     success
interface solve_evp_complex_1stage_single
  module procedure elpa_solve_evp_complex_1stage_single
end interface
#endif


contains

!-------------------------------------------------------------------------------

! All ELPA routines need MPI communicators for communicating within
! rows or columns of processes, these are set here.
! mpi_comm_rows/mpi_comm_cols can be free'd with MPI_Comm_free if not used any more.
!
!  Parameters
!
!> \param  mpi_comm_global   Global communicator for the calculations (in)
!>
!> \param  my_prow           Row coordinate of the calling process in the process grid (in)
!>
!> \param  my_pcol           Column coordinate of the calling process in the process grid (in)
!>
!> \param  mpi_comm_rows     Communicator for communicating within rows of processes (out)
!>
!> \param  mpi_comm_cols     Communicator for communicating within columns of processes (out)
!> \result mpierr            integer error value of mpi_comm_split function


function elpa_get_communicators(mpi_comm_global, my_prow, my_pcol, mpi_comm_rows, mpi_comm_cols) result(mpierr)
   ! use precision
   use elpa_mpi
   use iso_c_binding
   implicit none

   integer(kind=c_int), intent(in)  :: mpi_comm_global, my_prow, my_pcol
   integer(kind=c_int), intent(out) :: mpi_comm_rows, mpi_comm_cols

   integer(kind=c_int)              :: mpierr

   ! mpi_comm_rows is used for communicating WITHIN rows, i.e. all processes
   ! having the same column coordinate share one mpi_comm_rows.
   ! So the "color" for splitting is my_pcol and the "key" is my row coordinate.
   ! Analogous for mpi_comm_cols

   call mpi_comm_split(mpi_comm_global,my_pcol,my_prow,mpi_comm_rows,mpierr)
   call mpi_comm_split(mpi_comm_global,my_prow,my_pcol,mpi_comm_cols,mpierr)

end function elpa_get_communicators


!> \brief elpa_solve_evp_real_1stage_double: Fortran function to solve the real double-precision eigenvalue problem with 1-stage solver
!>
!  Parameters
!
!> \param  na                   Order of matrix a
!>
!> \param  nev                  Number of eigenvalues needed.
!>                              The smallest nev eigenvalues/eigenvectors are calculated.
!>
!> \param  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!>                              Distribution is like in Scalapack.
!>                              The full matrix must be set (not only one half like in scalapack).
!>                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                  Leading dimension of a
!>
!>  \param ev(na)               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)    On output: Eigenvectors of a
!>                              Distribution is like in Scalapack.
!>                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                  Leading dimension of q
!>
!>  \param nblk                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols           distributed number of matrix columns
!>
!>  \param mpi_comm_rows        MPI-Communicator for rows
!>  \param mpi_comm_cols        MPI-Communicator for columns
!>  \param mpi_comm_all         global MPI communicator
!>  \param useGPU              use GPU version (.true. or .false.)
!>
!>  \result                     success

#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "../../precision_macros.h"
#include "./elpa1_template_legacy.X90"
#undef REALCASE
#undef DOUBLE_PRECISION

#ifdef WANT_SINGLE_PRECISION_REAL
!> \brief elpa_solve_evp_real_1stage_single: Fortran function to solve the real single-precision eigenvalue problem with 1-stage solver
!>
!  Parameters
!
!> \param  na                   Order of matrix a
!>
!> \param  nev                  Number of eigenvalues needed.
!>                              The smallest nev eigenvalues/eigenvectors are calculated.
!>
!> \param  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!>                              Distribution is like in Scalapack.
!>                              The full matrix must be set (not only one half like in scalapack).
!>                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                  Leading dimension of a
!>
!>  \param ev(na)               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)    On output: Eigenvectors of a
!>                              Distribution is like in Scalapack.
!>                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                  Leading dimension of q
!>
!>  \param nblk                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols           distributed number of matrix columns
!>
!>  \param mpi_comm_rows        MPI-Communicator for rows
!>  \param mpi_comm_cols        MPI-Communicator for columns
!>  \param mpi_comm_all         global MPI commuicator
!>  \param useGPU
!>
!>  \result                     success

#define REALCASE 1
#define SINGLE_PRECISION 1
#include "../../precision_macros.h"
#include "./elpa1_template_legacy.X90"
#undef REALCASE
#undef SINGLE_PRECISION
#endif /* WANT_SINGLE_PRECISION_REAL */

!> \brief elpa_solve_evp_complex_1stage_double: Fortran function to solve the complex double-precision eigenvalue problem with 1-stage solver
!>
!  Parameters
!
!> \param  na                   Order of matrix a
!>
!> \param  nev                  Number of eigenvalues needed.
!>                              The smallest nev eigenvalues/eigenvectors are calculated.
!>
!> \param  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!>                              Distribution is like in Scalapack.
!>                              The full matrix must be set (not only one half like in scalapack).
!>                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                  Leading dimension of a
!>
!>  \param ev(na)               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)    On output: Eigenvectors of a
!>                              Distribution is like in Scalapack.
!>                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                  Leading dimension of q
!>
!>  \param nblk                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols           distributed number of matrix columns
!>
!>  \param mpi_comm_rows        MPI-Communicator for rows
!>  \param mpi_comm_cols        MPI-Communicator for columns
!>  \param mpi_comm_all         global MPI Communicator
!>  \param useGPU              use GPU version (.true. or .false.)
!>
!>  \result                     success
#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#include "../../precision_macros.h"
#include "./elpa1_template_legacy.X90"
#undef DOUBLE_PRECISION
#undef COMPLEXCASE


#ifdef WANT_SINGLE_PRECISION_COMPLEX

!> \brief elpa_solve_evp_complex_1stage_single: Fortran function to solve the complex single-precision eigenvalue problem with 1-stage solver
!>
!  Parameters
!
!> \param  na                   Order of matrix a
!>
!> \param  nev                  Number of eigenvalues needed.
!>                              The smallest nev eigenvalues/eigenvectors are calculated.
!>
!> \param  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!>                              Distribution is like in Scalapack.
!>                              The full matrix must be set (not only one half like in scalapack).
!>                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                  Leading dimension of a
!>
!>  \param ev(na)               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)    On output: Eigenvectors of a
!>                              Distribution is like in Scalapack.
!>                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                  Leading dimension of q
!>
!>  \param nblk                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols           distributed number of matrix columns
!>
!>  \param mpi_comm_rows        MPI-Communicator for rows
!>  \param mpi_comm_cols        MPI-Communicator for columns
!>  \param mpi_comm_all         global MPI communicator
!>  \param useGPU
!>
!>  \result                     success

#define COMPLEXCASE 1
#define SINGLE_PRECISION
#include "../../precision_macros.h"
#include "./elpa1_template_legacy.X90"
#undef COMPLEXCASE
#undef SINGLE_PRECISION
#endif /* WANT_SINGLE_PRECISION_COMPLEX */

end module ELPA1_legacy
