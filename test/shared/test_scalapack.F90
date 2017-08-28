! (c) Copyright Pavel Kus, 2017, MPCDF
!
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

#include "../Fortran/assert.h"
#include "config-f90.h"

module test_scalapack
  use test_util

  interface solve_scalapack_all
    module procedure solve_pdsyevd
    module procedure solve_pzheevd
#ifdef WANT_SINGLE_PRECISION_REAL
    module procedure solve_pssyevd
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
    module procedure solve_pcheevd
#endif
  end interface

  interface solve_scalapack_part
    module procedure solve_pdsyevr
    module procedure solve_pzheevr
#ifdef WANT_SINGLE_PRECISION_REAL
    module procedure solve_pssyevr
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
    module procedure solve_pcheevr
#endif
  end interface

contains

#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#include "../../src/general/precision_macros.h"
#include "test_scalapack_template.F90"
#undef DOUBLE_PRECISION
#undef COMPLEXCASE

#ifdef WANT_SINGLE_PRECISION_COMPLEX

#define COMPLEXCASE 1
#define SINGLE_PRECISION 1
#include "../../src/general/precision_macros.h"
#include "test_scalapack_template.F90"
#undef SINGLE_PRECISION
#undef COMPLEXCASE

#endif /* WANT_SINGLE_PRECISION_COMPLEX */

#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "../../src/general/precision_macros.h"
#include "test_scalapack_template.F90"
#undef DOUBLE_PRECISION
#undef REALCASE

#ifdef WANT_SINGLE_PRECISION_REAL

#define REALCASE 1
#define SINGLE_PRECISION 1
#include "../../src/general/precision_macros.h"
#include "test_scalapack_template.F90"
#undef SINGLE_PRECISION
#undef REALCASE

#endif /* WANT_SINGLE_PRECISION_REAL */


end module
