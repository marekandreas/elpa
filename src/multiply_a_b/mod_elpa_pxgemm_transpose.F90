!    Copyright 2024, P. Karpov
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
! This file was written by P. Karpov, MPCDF

#include "config-f90.h"
module elpa_pxgemm_transpose

  public

  contains

#undef USE_CCL_PXGEMM
#define CCL _

#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "../general/precision_macros.h"
#include "elpa_pxgemm_transpose_template.F90"
#undef REALCASE
#undef DOUBLE_PRECISION

! single precision
#ifdef WANT_SINGLE_PRECISION_REAL
#define REALCASE 1
#define SINGLE_PRECISION 1
#include "../general/precision_macros.h"
#include "elpa_pxgemm_transpose_template.F90"
#undef REALCASE
#undef SINGLE_PRECISION
#endif /* WANT_SINGLE_PRECISION_REAL */

! double precision
#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#include "../general/precision_macros.h"
#include "elpa_pxgemm_transpose_template.F90"
#undef COMPLEXCASE
#undef DOUBLE_PRECISION

#ifdef WANT_SINGLE_PRECISION_COMPLEX
#define COMPLEXCASE 1
#define SINGLE_PRECISION 1
#include "../general/precision_macros.h"
#include "elpa_pxgemm_transpose_template.F90"
#undef COMPLEXCASE
#undef SINGLE_PRECISION
#endif /* WANT_SINGLE_PRECISION_COMPLEX */

!___________________________________________________
! CCL-version

#undef USE_CCL_PXGEMM
#undef CCL

#define USE_CCL_PXGEMM
#define CCL _ccl_

#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "../general/precision_macros.h"
#include "elpa_pxgemm_transpose_template.F90"
#undef REALCASE
#undef DOUBLE_PRECISION

! single precision
#ifdef WANT_SINGLE_PRECISION_REAL
#define REALCASE 1
#define SINGLE_PRECISION 1
#include "../general/precision_macros.h"
#include "elpa_pxgemm_transpose_template.F90"
#undef REALCASE
#undef SINGLE_PRECISION
#endif /* WANT_SINGLE_PRECISION_REAL */

! double precision
#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#include "../general/precision_macros.h"
#include "elpa_pxgemm_transpose_template.F90"
#undef COMPLEXCASE
#undef DOUBLE_PRECISION

#ifdef WANT_SINGLE_PRECISION_COMPLEX
#define COMPLEXCASE 1
#define SINGLE_PRECISION 1
#include "../general/precision_macros.h"
#include "elpa_pxgemm_transpose_template.F90"
#undef COMPLEXCASE
#undef SINGLE_PRECISION
#endif /* WANT_SINGLE_PRECISION_COMPLEX */

end module

