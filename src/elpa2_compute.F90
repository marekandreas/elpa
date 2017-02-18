!    This file is part of ELPA.
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
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".



! ELPA2 -- 2-stage solver for ELPA
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".


#include "config-f90.h"

module ELPA2_compute

! Version 1.1.2, 2011-02-21

  use ELPA_utilities
  USE ELPA1_compute
  use elpa1, only : elpa_print_times, time_evp_back, time_evp_fwd, time_evp_solve
  use elpa2_utilities
  use elpa_pdgeqrf
  use precision
  use elpa_mpi
  use aligned_mem

  implicit none

  PRIVATE ! By default, all routines contained are private

  public :: bandred_real_double
  public :: tridiag_band_real_double
  public :: trans_ev_tridi_to_band_real_double
  public :: trans_ev_band_to_full_real_double

#ifdef WANT_SINGLE_PRECISION_REAL
  public :: bandred_real_single
  public :: tridiag_band_real_single
  public :: trans_ev_tridi_to_band_real_single
  public :: trans_ev_band_to_full_real_single
#endif

  public :: bandred_complex_double
  public :: tridiag_band_complex_double
  public :: trans_ev_tridi_to_band_complex_double
  public :: trans_ev_band_to_full_complex_double

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  public :: bandred_complex_single
  public :: tridiag_band_complex_single
  public :: trans_ev_tridi_to_band_complex_single
  public :: trans_ev_band_to_full_complex_single
#endif
  public :: band_band_real_double
!  public :: divide_band

  integer(kind=ik), public :: which_qr_decomposition = 1     ! defines, which QR-decomposition algorithm will be used
                                                    ! 0 for unblocked
                                                    ! 1 for blocked (maxrank: nblk)
  contains

! real double precision first
#define DOUBLE_PRECISION_REAL 1

#define REAL_DATATYPE rk8
#define BYTESIZE 8
#define REALCASE 1
#include "redist_band.X90"
#undef DOUBLE_PRECISION_REAL
#undef REAL_DATATYPE
#undef BYTESIZE
#undef REALCASE

! single precision
#ifdef WANT_SINGLE_PRECISION_REAL

#undef DOUBLE_PRECISION_REAL
#define REAL_DATATYPE rk4
#define BYTESIZE 4
#define REALCASE 1
#include "redist_band.X90"
#undef REAL_DATATYPE
#undef BYTESIZE
#undef REALCASE

#endif

! double precision
#define DOUBLE_PRECISION_COMPLEX 1

#define COMPLEX_DATATYPE ck8
#define BYTESIZE 16
#define COMPLEXCASE 1
#include "redist_band.X90"
#undef COMPLEX_DATATYPE
#undef BYTESIZE
#undef COMPLEXCASE
#undef DOUBLE_PRECISION_COMPLEX

#ifdef WANT_SINGLE_PRECISION_COMPLEX

#undef DOUBLE_PRECISION_COMPLEX
#undef DOUBLE_PRECISION_REAL
#define COMPLEX_DATATYPE ck4
#define COMPLEXCASE 1
#include "redist_band.X90"
#undef COMPLEX_DATATYPE
#undef BYTESIZE
#undef COMPLEXCASE

#endif /* WANT_SINGLE_PRECISION_COMPLEX */



! real double precision
#define DOUBLE_PRECISION_REAL 1
#define REAL_DATATYPE rk8

#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "precision_macros.h"
#include "elpa2_compute_real_template.X90"
#undef REALCASE
#undef DOUBLE_PRECISION

#undef DOUBLE_PRECISION_REAL
#undef REAL_DATATYPE

! real single precision
#if defined(WANT_SINGLE_PRECISION_REAL)

#undef DOUBLE_PRECISION_REAL
#define REAL_DATATYPE rk4

#define REALCASE 1
#define SINGLE_PRECISION 1
#include "precision_macros.h"
#include "elpa2_compute_real_template.X90"
#undef REALCASE
#undef SINGLE_PRECISION

#undef DOUBLE_PRECISION_REAL
#undef REAL_DATATYPE

#endif /* WANT_SINGLE_PRECISION_REAL */

! complex double precision
#define DOUBLE_PRECISION_COMPLEX 1
#define REAL_DATATYPE rk8
#define COMPLEX_DATATYPE ck8

#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#include "precision_macros.h"
#include "elpa2_compute_complex_template.X90"
#undef COMPLEXCASE
#undef DOUBLE_PRECISION

#undef DOUBLE_PRECISION_COMPLEX
#undef REAL_DATATYPE
#undef COMPLEX_DATATYPE


! complex single precision
#if defined(WANT_SINGLE_PRECISION_COMPLEX)

#undef DOUBLE_PRECISION_COMPLEX
#define REAL_DATATYPE rk4
#define COMPLEX_DATATYPE ck4

#define COMPLEXCASE 1
#define SINGLE_PRECISION 1
#include "precision_macros.h"
#include "elpa2_compute_complex_template.X90"
#undef COMPLEXCASE
#undef SINGLE_PRECISION

#undef DOUBLE_PRECISION_COMPLEX
#undef COMPLEX_DATATYPE
#undef REAL_DATATYPE

#endif /* WANT_SINGLE_PRECISION_COMPLEX */

end module ELPA2_compute
