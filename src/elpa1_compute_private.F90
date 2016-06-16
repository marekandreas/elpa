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

module ELPA1_compute
  use elpa_utilities
#ifdef HAVE_DETAILED_TIMINGS
  use timings
#endif
  use elpa_mpi
  implicit none

  PRIVATE ! set default to private

  public :: tridiag_real_double               ! Transform real symmetric matrix to tridiagonal form
  public :: tridiag_real
  public :: trans_ev_real_double              ! Transform real eigenvectors of a tridiagonal matrix back
  public :: trans_ev_real

  public :: solve_tridi_double

  interface tridiag_real
    module procedure tridiag_real_double
  end interface


  interface trans_ev_real
    module procedure trans_ev_real_double
  end interface

#ifdef WANT_SINGLE_PRECISION_REAL
  public :: tridiag_real_single        ! Transform real single-precision symmetric matrix to tridiagonal form
  public :: trans_ev_real_single       ! Transform real  single-precision eigenvectors of a tridiagonal matrix back
  public :: solve_tridi_single
#endif

  public :: tridiag_complex_double            ! Transform complex hermitian matrix to tridiagonal form
  public :: tridiag_complex
  public :: trans_ev_complex_double           ! Transform eigenvectors of a tridiagonal matrix back
  public :: trans_ev_complex

  interface tridiag_complex
    module procedure tridiag_complex_double
  end interface

  interface trans_ev_complex
    module procedure trans_ev_complex_double
  end interface

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  public :: tridiag_complex_single     ! Transform complex single-precision hermitian matrix to tridiagonal form
  public :: trans_ev_complex_single    ! Transform complex single-precision eigenvectors of a tridiagonal matrix back
#endif

  public :: local_index                ! Get local index of a block cyclic distributed matrix
  public :: least_common_multiple      ! Get least common multiple

  public :: hh_transform_real_double
  public :: hh_transform_real
  public :: elpa_reduce_add_vectors_real_double
  public :: elpa_reduce_add_vectors_real
  public :: elpa_transpose_vectors_real_double
  public :: elpa_transpose_vectors_real

  interface hh_transform_real
    module procedure hh_transform_real_double
  end interface

  interface elpa_reduce_add_vectors_real
    module procedure elpa_reduce_add_vectors_real_double
  end interface

  interface elpa_transpose_vectors_real
    module procedure elpa_transpose_vectors_real_double
  end interface

#ifdef WANT_SINGLE_PRECISION_REAL
  public :: hh_transform_real_single
  public :: elpa_reduce_add_vectors_real_single
  public :: elpa_transpose_vectors_real_single
#endif

  public :: hh_transform_complex_double
  public :: hh_transform_complex
  public :: elpa_reduce_add_vectors_complex_double
  public :: elpa_reduce_add_vectors_complex
  public :: elpa_transpose_vectors_complex_double
  public :: elpa_transpose_vectors_complex

  interface hh_transform_complex
    module procedure hh_transform_complex_double
  end interface

  interface elpa_reduce_add_vectors_complex
    module procedure elpa_reduce_add_vectors_complex_double
  end interface

  interface elpa_transpose_vectors_complex
    module procedure elpa_transpose_vectors_complex_double
  end interface

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  public :: hh_transform_complex_single
  public :: elpa_reduce_add_vectors_complex_single
  public :: elpa_transpose_vectors_complex_single
#endif

  contains

! real double precision first
#define DOUBLE_PRECISION_REAL 1

#define DATATYPE REAL(kind=rk8)
#define BYTESIZE 8
#define REALCASE 1
#include "elpa_transpose_vectors.X90"
#include "elpa_reduce_add_vectors.X90"
#undef DOUBLE_PRECISION_REAL
#undef DATATYPE
#undef BYTESIZE
#undef REALCASE

! single precision
#ifdef WANT_SINGLE_PRECISION_REAL

#undef DOUBLE_PRECISION_REAL
#define DATATYPE REAL(kind=rk4)
#define BYTESIZE 4
#define REALCASE 1
#include "elpa_transpose_vectors.X90"
#include "elpa_reduce_add_vectors.X90"
#undef DATATYPE
#undef BYTESIZE
#undef REALCASE

#endif

! double precision
#define DOUBLE_PRECISION_COMPLEX 1

#define DATATYPE COMPLEX(kind=ck8)
#define BYTESIZE 16
#define COMPLEXCASE 1
#include "elpa_transpose_vectors.X90"
#include "elpa_reduce_add_vectors.X90"
#undef DATATYPE
#undef BYTESIZE
#undef COMPLEXCASE
#undef DOUBLE_PRECISION_COMPLEX

#ifdef WANT_SINGLE_PRECISION_COMPLEX

#undef DOUBLE_PRECISION_COMPLEX
#undef DOUBLE_PRECISION_REAL
#define DATATYPE COMPLEX(kind=ck4)
#define COMPLEXCASE 1
#include "elpa_transpose_vectors.X90"
#include "elpa_reduce_add_vectors.X90"
#undef DATATYPE
#undef BYTESIZE
#undef COMPLEXCASE

#endif /* WANT_SINGLE_PRECISION_COMPLEX */

! real double precision
#define DOUBLE_PRECISION_REAL 1
#define REAL_DATATYPE rk8

#include "elpa1_compute_real_template.X90"

#undef DOUBLE_PRECISION_REAL
#undef REAL_DATATYPE

! real single precision
#if defined(WANT_SINGLE_PRECISION_REAL)

#undef DOUBLE_PRECISION_REAL
#define REAL_DATATYPE rk4

#include "elpa1_compute_real_template.X90"

#undef DOUBLE_PRECISION_REAL
#undef REAL_DATATYPE

#endif /* WANT_SINGLE_PRECISION_REAL */

! complex double precision
#define DOUBLE_PRECISION_COMPLEX 1
#define REAL_DATATYPE rk8
#define COMPLEX_DATATYPE ck8
#include "elpa1_compute_complex_template.X90"

#undef DOUBLE_PRECISION_COMPLEX
#undef REAL_DATATYPE
#undef COMPLEX_DATATYPE


! complex single precision
#if defined(WANT_SINGLE_PRECISION_COMPLEX)

#undef DOUBLE_PRECISION_COMPLEX
#define REAL_DATATYPE rk4
#define COMPLEX_DATATYPE ck4

#include "elpa1_compute_complex_template.X90"

#undef DOUBLE_PRECISION_COMPLEX
#undef COMPLEX_DATATYPE
#undef REAL_DATATYPE

#endif /* WANT_SINGLE_PRECISION_COMPLEX */

end module ELPA1_compute
