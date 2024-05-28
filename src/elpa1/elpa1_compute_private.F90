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

#include "config-f90.h"
!> \brief Fortran module which contains the source of ELPA 1stage
module elpa1_compute
  use elpa_utilities
  use elpa_mpi
  implicit none

  PRIVATE ! set default to private

  public :: tridiag_cpu_real_double               ! Transform real symmetric matrix to tridiagonal form
  public :: tridiag_gpu_real_double               ! Transform real symmetric matrix to tridiagonal form
  !public :: tridiag_real
  public :: trans_ev_cpu_real_double              ! Transform real eigenvectors of a tridiagonal matrix back
  public :: trans_ev_gpu_real_double              ! Transform real eigenvectors of a tridiagonal matrix back

  public :: solve_tridi_cpu_double_impl
  public :: solve_tridi_gpu_double_impl

#ifdef WANT_SINGLE_PRECISION_REAL
  public :: tridiag_cpu_real_single        ! Transform real single-precision symmetric matrix to tridiagonal form
  public :: tridiag_gpu_real_single        ! Transform real single-precision symmetric matrix to tridiagonal form
  public :: trans_ev_cpu_real_single       ! Transform real  single-precision eigenvectors of a tridiagonal matrix back
  public :: trans_ev_gpu_real_single       ! Transform real  single-precision eigenvectors of a tridiagonal matrix back
  !public :: solve_tridi_single
  public :: solve_tridi_cpu_single_impl
  public :: solve_tridi_gpu_single_impl
#endif

  public :: tridiag_cpu_complex_double            ! Transform complex hermitian matrix to tridiagonal form
  public :: tridiag_gpu_complex_double            ! Transform complex hermitian matrix to tridiagonal form
  public :: trans_ev_cpu_complex_double           ! Transform eigenvectors of a tridiagonal matrix back
  public :: trans_ev_gpu_complex_double           ! Transform eigenvectors of a tridiagonal matrix back

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  public :: tridiag_cpu_complex_single     ! Transform complex single-precision hermitian matrix to tridiagonal form
  public :: tridiag_gpu_complex_single     ! Transform complex single-precision hermitian matrix to tridiagonal form
  public :: trans_ev_cpu_complex_single    ! Transform complex single-precision eigenvectors of a tridiagonal matrix back
  public :: trans_ev_gpu_complex_single    ! Transform complex single-precision eigenvectors of a tridiagonal matrix back
#endif

  public :: hh_transform_real_double
  public :: hh_transform_real
  public :: elpa_reduce_add_vectors_real_double
  public :: elpa_reduce_add_vectors_real
  public :: elpa_transpose_vectors_real_double
  public :: elpa_transpose_vectors_ss_real_double
  public :: elpa_transpose_vectors_real
  public :: elpa_transpose_vectors_ss_real

  interface hh_transform_real
    module procedure hh_transform_real_double
  end interface

  interface elpa_reduce_add_vectors_real
    module procedure elpa_reduce_add_vectors_real_double
  end interface

  interface elpa_transpose_vectors_real
    module procedure elpa_transpose_vectors_real_double
  end interface
  
  interface elpa_transpose_vectors_ss_real
    module procedure elpa_transpose_vectors_ss_real_double
  end interface

#ifdef WANT_SINGLE_PRECISION_REAL
  public :: hh_transform_real_single
  public :: elpa_reduce_add_vectors_real_single
  public :: elpa_transpose_vectors_real_single
  public :: elpa_transpose_vectors_ss_real_single
#endif

  public :: hh_transform_complex_double
  public :: hh_transform_complex
  public :: elpa_reduce_add_vectors_complex_double
  public :: elpa_reduce_add_vectors_complex
  public :: elpa_transpose_vectors_complex_double
  public :: elpa_transpose_vectors_ss_complex_double
  public :: elpa_transpose_vectors_complex
  public :: elpa_transpose_vectors_ss_complex

  interface hh_transform_complex
    module procedure hh_transform_complex_double
  end interface

  interface elpa_reduce_add_vectors_complex
    module procedure elpa_reduce_add_vectors_complex_double
  end interface

  interface elpa_transpose_vectors_complex
    module procedure elpa_transpose_vectors_complex_double
  end interface
  
  interface elpa_transpose_vectors_ss_complex
    module procedure elpa_transpose_vectors_ss_complex_double
  end interface

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  public :: hh_transform_complex_single
  public :: elpa_reduce_add_vectors_complex_single
  public :: elpa_transpose_vectors_complex_single
  public :: elpa_transpose_vectors_ss_complex_single
#endif

#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
  public :: elpa_gpu_ccl_transpose_vectors_real_double
  public :: elpa_gpu_ccl_reduce_add_vectors_real_double
  public :: elpa_gpu_ccl_transpose_vectors_complex_double
  public :: elpa_gpu_ccl_reduce_add_vectors_complex_double

#ifdef WANT_SINGLE_PRECISION_REAL
  public :: elpa_gpu_ccl_transpose_vectors_real_single
  public :: elpa_gpu_ccl_reduce_add_vectors_real_single
#endif

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  public :: elpa_gpu_ccl_transpose_vectors_complex_single
  public :: elpa_gpu_ccl_reduce_add_vectors_complex_single
#endif
#endif /* WITH_NVIDIA_NCCL || WITH_AMD_RCCL */

  contains

!________________________________________________________________
! elpa_transpose_vectors_template.F90
! elpa_reduce_add_vectors_template.F90

! real double precision
!#define DOUBLE_PRECISION_REAL 1
#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "../general/precision_macros.h"
#include "elpa_transpose_vectors_template.F90"
#define SKEW_SYMMETRIC_BUILD
#include "elpa_transpose_vectors_template.F90"
#undef SKEW_SYMMETRIC_BUILD
#include "elpa_reduce_add_vectors_template.F90"
#undef DOUBLE_PRECISION
#undef REALCASE

! real single precision
#ifdef WANT_SINGLE_PRECISION_REAL
#define REALCASE 1
#define SINGLE_PRECISION 1
#include "../general/precision_macros.h"
#include "elpa_transpose_vectors_template.F90"
#define SKEW_SYMMETRIC_BUILD
#include "elpa_transpose_vectors_template.F90"
#undef SKEW_SYMMETRIC_BUILD
#include "elpa_reduce_add_vectors_template.F90"
#undef SINGLE_PRECISION
#undef REALCASE
#endif /* WANT_SINGLE_PRECISION_REAL */

! complex double precision
#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#include "../general/precision_macros.h"
#include "elpa_transpose_vectors_template.F90"
#define SKEW_SYMMETRIC_BUILD
#include "elpa_transpose_vectors_template.F90"
#undef SKEW_SYMMETRIC_BUILD
#include "elpa_reduce_add_vectors_template.F90"
#undef COMPLEXCASE
#undef DOUBLE_PRECISION

! complex single precision
#ifdef WANT_SINGLE_PRECISION_COMPLEX
#define COMPLEXCASE 1
#define SINGLE_PRECISION 1
#include "../general/precision_macros.h"
#include "elpa_transpose_vectors_template.F90"
#define SKEW_SYMMETRIC_BUILD
#include "elpa_transpose_vectors_template.F90"
#undef SKEW_SYMMETRIC_BUILD
#include "elpa_reduce_add_vectors_template.F90"
#undef COMPLEXCASE
#undef SINGLE_PRECISION
#endif /* WANT_SINGLE_PRECISION_COMPLEX */

!________________________________________________________________
! ./GPU/elpa_gpu_ccl_transpose_vectors_template.F90
! ./GPU/elpa_gpu_ccl_reduce_add_vectors_template.F90

#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)

! real double precision
!#define DOUBLE_PRECISION_REAL 1
#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "../general/precision_macros.h"
#include "./GPU/elpa_gpu_ccl_transpose_vectors_template.F90"
#include "./GPU/elpa_gpu_ccl_reduce_add_vectors_template.F90"
#undef REALCASE
#undef DOUBLE_PRECISION

! real single precision
#ifdef WANT_SINGLE_PRECISION_REAL
#define REALCASE 1
#define SINGLE_PRECISION 1
#include "../general/precision_macros.h"
#include "./GPU/elpa_gpu_ccl_transpose_vectors_template.F90"
#include "./GPU/elpa_gpu_ccl_reduce_add_vectors_template.F90"
#undef REALCASE
#undef SINGLE_PRECISION
#endif /* WANT_SINGLE_PRECISION_REAL */

! complex double precision
#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#include "../general/precision_macros.h"
#include "./GPU/elpa_gpu_ccl_transpose_vectors_template.F90"
#include "./GPU/elpa_gpu_ccl_reduce_add_vectors_template.F90"
#undef COMPLEXCASE
#undef DOUBLE_PRECISION

! complex single precision
#ifdef WANT_SINGLE_PRECISION_COMPLEX
#define COMPLEXCASE 1
#define SINGLE_PRECISION 1
#include "../general/precision_macros.h"
#include "./GPU/elpa_gpu_ccl_transpose_vectors_template.F90"
#include "./GPU/elpa_gpu_ccl_reduce_add_vectors_template.F90"
#undef COMPLEXCASE
#undef SINGLE_PRECISION
#endif /* WANT_SINGLE_PRECISION_COMPLEX */

#endif /* WITH_NVIDIA_NCCL || WITH_AMD_RCCL */

!________________________________________________________________
! elpa1_compute_template.F90

! real double precision
#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "../general/precision_macros.h"
#include "elpa1_compute_template.F90"
#undef REALCASE
#undef DOUBLE_PRECISION

! real single precision
#if defined(WANT_SINGLE_PRECISION_REAL)
#define REALCASE 1
#define SINGLE_PRECISION 1
#include "../general/precision_macros.h"
#include "elpa1_compute_template.F90"
#undef REALCASE
#undef SINGLE_PRECISION
#endif /* WANT_SINGLE_PRECISION_REAL */

! complex double precision
#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#include "../general/precision_macros.h"
#include "elpa1_compute_template.F90"
#undef COMPLEXCASE
#undef DOUBLE_PRECISION

! complex single precision
#if defined(WANT_SINGLE_PRECISION_COMPLEX)
#define COMPLEXCASE 1
#define SINGLE_PRECISION 1
#include "../general/precision_macros.h"
#include "elpa1_compute_template.F90"
#undef COMPLEXCASE
#undef SINGLE_PRECISION
#endif /* WANT_SINGLE_PRECISION_COMPLEX */

end module elpa1_compute
