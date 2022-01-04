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
  use elpa_cholesky
  use elpa_invert_trm
  use elpa_multiply_a_b

  implicit none

  contains

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

