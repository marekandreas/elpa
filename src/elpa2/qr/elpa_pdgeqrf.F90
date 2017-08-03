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

#include "config-f90.h"

module elpa_pdgeqrf

  use elpa_utilities
  use elpa1_compute
  use elpa_pdlarfb
  use qr_utils_mod
  use elpa_qrkernels
  use elpa_mpi
  implicit none

  PRIVATE

  public :: qr_pdgeqrf_2dcomm_double
  public :: qr_pqrparam_init
  public :: qr_pdlarfg2_1dcomm_check_double

#ifdef WANT_SINGLE_PRECISION_REAL
  public :: qr_pdgeqrf_2dcomm_single
  public :: qr_pdlarfg2_1dcomm_check_single
#endif


  contains
  ! real double precision
#define REALCASE 1
#define DOUBLE_PRECISION 1
#undef ALREADY_DEFINED
#include "../../general/precision_macros.h"
#include "elpa_pdgeqrf_template.F90"
#undef REALCASE
#undef DOUBLE_PRECISION
#define ALREADY_DEFINED

#ifdef WANT_SINGLE_PRECISION_REAL
  ! real single precision
#define REALCASE 1
#define ALREADY_DEFINED
#define SINGLE_PRECISION 1
#include "../../general/precision_macros.h"
#include "elpa_pdgeqrf_template.F90"
#undef REALCASE
#undef SINGLE_PRECISION
#endif


end module elpa_pdgeqrf
