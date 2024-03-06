!    Copyright 2021, A. Marek
!
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
! This file was written by A. Marek, MPCDF
#include "config-f90.h"

module elpa_cholesky
  use, intrinsic :: iso_c_binding
  use precision
  implicit none

  public

  public :: elpa_cholesky_a_h_a_real_double_impl       !< Cholesky factorization of a double-precision real matrix
  public :: elpa_cholesky_d_ptr_real_double_impl       !< Cholesky factorization of a double-precision real matrix

  public :: elpa_cholesky_a_h_a_complex_double_impl    !< Cholesky factorization of a double-precision complex matrix
  public :: elpa_cholesky_d_ptr_complex_double_impl    !< Cholesky factorization of a double-precision complex matrix

#ifdef WANT_SINGLE_PRECISION_REAL
  public :: elpa_cholesky_a_h_a_real_single_impl       !< Cholesky factorization of a single-precision real matrix
  public :: elpa_cholesky_d_ptr_real_single_impl       !< Cholesky factorization of a single-precision real matrix

#endif 

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  public :: elpa_cholesky_a_h_a_complex_single_impl    !< Cholesky factorization of a single-precision complex matrix
  public :: elpa_cholesky_d_ptr_complex_single_impl    !< Cholesky factorization of a single-precision complex matrix
#endif

  contains
#define REALCASE 1
#undef DEVICE_POINTER
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"

!> \brief  elpa_cholesky_a_h_a_real_double_impl: Cholesky factorization of a double-precision real symmetric matrix
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
   function elpa_cholesky_a_h_a_real_double_impl (obj, a) result(success)
#include "./elpa_cholesky_template.F90"

    end function elpa_cholesky_a_h_a_real_double_impl
#undef DOUBLE_PRECISION
#undef REALCASE


#define REALCASE 1
#define DEVICE_POINTER
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"

!> \brief  elpa_cholesky_d_ptr_real_double_impl: Cholesky factorization of a double-precision real symmetric matrix
!> \details
!> \param  obj                    elpa_t object contains:
!> \param     - obj%na            Order of matrix
!> \param     - obj%local_nrows   Leading dimension of a
!> \param     - obj%local_ncols   local columns of matrix a
!> \param     - obj%nblk          blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows MPI communicator for rows
!> \param     - obj%mpi_comm_cols MPI communicator for columns
!> \param     - obj%wantDebug     logical, more debug information on failure
!> \param  aDev(lda,matrixCols)   Distributed matrix which should be inverted as type(c_ptr) living on device
!>                                Distribution is like in Scalapack.
!>                                Only upper triangle needs to be set.
!>                                The lower triangle is not referenced.
!> \result succes                 logical, reports success or failure
   function elpa_cholesky_d_ptr_real_double_impl (obj, aDev) result(success)
#include "./elpa_cholesky_template.F90"

    end function elpa_cholesky_d_ptr_real_double_impl
#undef DOUBLE_PRECISION
#undef REALCASE
#undef DEVICE_POINTER


#ifdef WANT_SINGLE_PRECISION_REAL
#undef DEVICE_POINTER
#define REALCASE 1
#define SINGLE_PRECISION
#include "../general/precision_macros.h"


!> \brief  elpa_cholesky_a_h_a_real_single_impl: Cholesky factorization of a double-precision real symmetric matrix
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
   function elpa_cholesky_a_h_a_real_single_impl(obj, a) result(success)
#include "./elpa_cholesky_template.F90"

    end function elpa_cholesky_a_h_a_real_single_impl
#undef SINGLE_PRECISION
#undef REALCASE

#endif /* WANT_SINGLE_PRECSION_REAL */

#ifdef WANT_SINGLE_PRECISION_REAL
#define DEVICE_POINTER
#define REALCASE 1
#define SINGLE_PRECISION
#include "../general/precision_macros.h"


!> \brief  elpa_cholesky_d_ptr_real_single_impl: Cholesky factorization of a double-precision real symmetric matrix
!> \details
!> \param  obj                    elpa_t object contains:
!> \param     - obj%na            Order of matrix
!> \param     - obj%local_nrows   Leading dimension of a
!> \param     - obj%local_ncols   local columns of matrix a
!> \param     - obj%nblk          blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows MPI communicator for rows
!> \param     - obj%mpi_comm_cols MPI communicator for columns
!> \param     - obj%wantDebug     logical, more debug information on failure
!> \param  aDev(lda,matrixCols)   Distributed matrix which should be inverted as type(c_ptr) living on a device
!>                                Distribution is like in Scalapack.
!>                                Only upper triangle needs to be set.
!>                                The lower triangle is not referenced.
!> \result succes                 logical, reports success or failure
   function elpa_cholesky_d_ptr_real_single_impl(obj, aDev) result(success)
#include "./elpa_cholesky_template.F90"

    end function elpa_cholesky_d_ptr_real_single_impl
#undef SINGLE_PRECISION
#undef REALCASE
#undef DEVICE_POINTER

#endif /* WANT_SINGLE_PRECSION_REAL */



#define COMPLEXCASE 1
#undef DEVICE_POINTER
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"

!> \brief  elpa_cholesky_a_h_a_complex_double_impl: Cholesky factorization of a double-precision complex hermitian matrix
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
    function elpa_cholesky_a_h_a_complex_double_impl(obj, a) result(success)

#include "./elpa_cholesky_template.F90"

    end function elpa_cholesky_a_h_a_complex_double_impl
#undef DOUBLE_PRECISION
#undef COMPLEXCASE

#define COMPLEXCASE 1
#define DEVICE_POINTER
#define DOUBLE_PRECISION
#include "../general/precision_macros.h"

!> \brief  elpa_cholesky_d_ptr_complex_double_impl: Cholesky factorization of a double-precision complex hermitian matrix
!> \details
!> \param  obj                    elpa_t object contains:
!> \param     - obj%na            Order of matrix
!> \param     - obj%local_nrows   Leading dimension of a
!> \param     - obj%local_ncols   local columns of matrix a
!> \param     - obj%nblk          blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows MPI communicator for rows
!> \param     - obj%mpi_comm_cols MPI communicator for columns
!> \param     - obj%wantDebug     logical, more debug information on failure
!> \param  aDev(lda,matrixCols)   Distributed matrix which should be inverted as type(c_ptr) living on a device
!>                                Distribution is like in Scalapack.
!>                                Only upper triangle needs to be set.
!>                                The lower triangle is not referenced.
!> \result succes                 logical, reports success or failure
    function elpa_cholesky_d_ptr_complex_double_impl(obj, aDev) result(success)

#include "./elpa_cholesky_template.F90"

    end function elpa_cholesky_d_ptr_complex_double_impl
#undef DOUBLE_PRECISION
#undef COMPLEXCASE
#undef DEVICE_POINTER

#ifdef WANT_SINGLE_PRECISION_COMPLEX
#undef DEVICE_POINTER
#define COMPLEXCASE 1
#define SINGLE_PRECISION
#include "../general/precision_macros.h"

!> \brief  elpa_cholesky_a_h_a_complex_single_impl: Cholesky factorization of a single-precision complex hermitian matrix
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
    function elpa_cholesky_a_h_a_complex_single_impl(obj, a) result(success)

#include "./elpa_cholesky_template.F90"

    end function elpa_cholesky_a_h_a_complex_single_impl
#undef SINGLE_PRECISION
#undef COMPLEXCASE

#endif /* WANT_SINGLE_PRECISION_COMPLEX */

#ifdef WANT_SINGLE_PRECISION_COMPLEX
#define DEVICE_POINTER
#define COMPLEXCASE 1
#define SINGLE_PRECISION
#include "../general/precision_macros.h"

!> \brief  elpa_cholesky_d_ptr_complex_single_impl: Cholesky factorization of a single-precision complex hermitian matrix
!> \details
!> \param  obj                    elpa_t object contains:
!> \param     - obj%na            Order of matrix
!> \param     - obj%local_nrows   Leading dimension of a
!> \param     - obj%local_ncols   local columns of matrix a
!> \param     - obj%nblk          blocksize of cyclic distribution, must be the same in both directions!
!> \param     - obj%mpi_comm_rows MPI communicator for rows
!> \param     - obj%mpi_comm_cols MPI communicator for columns
!> \param     - obj%wantDebug     logical, more debug information on failure
!> \param  aDev(lda,matrixCols)   Distributed matrix which should be inverted, as type(c_ptr) living on device
!>                                Distribution is like in Scalapack.
!>                                Only upper triangle needs to be set.
!>                                The lower triangle is not referenced.
!> \result succes                 logical, reports success or failure
    function elpa_cholesky_d_ptr_complex_single_impl(obj, aDev) result(success)

#include "./elpa_cholesky_template.F90"

    end function elpa_cholesky_d_ptr_complex_single_impl
#undef SINGLE_PRECISION
#undef COMPLEXCASE
#undef DEVICE_POINTER

#endif /* WANT_SINGLE_PRECISION_COMPLEX */


end module
