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
! Author: A. Marek, MPCDF
#include "config-f90.h"

module test_check_correctness
  use test_util

  interface check_correctness_evp_numeric_residuals
    module procedure check_correctness_evp_numeric_residuals_complex_double
    module procedure check_correctness_evp_numeric_residuals_real_double
#ifdef WANT_SINGLE_PRECISION_REAL
    module procedure check_correctness_evp_numeric_residuals_real_single
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
    module procedure check_correctness_evp_numeric_residuals_complex_single
#endif
  end interface
  
  interface check_correctness_evp_numeric_residuals_ss
!     module procedure check_correctness_evp_numeric_residuals_ss_complex_double
    module procedure check_correctness_evp_numeric_residuals_ss_real_double
#ifdef WANT_SINGLE_PRECISION_REAL
    module procedure check_correctness_evp_numeric_residuals_ss_real_single
#endif
! #ifdef WANT_SINGLE_PRECISION_COMPLEX
!     module procedure check_correctness_evp_numeric_residuals_ss_complex_single
! #endif
  end interface

  interface check_correctness_eigenvalues_toeplitz
    module procedure check_correctness_eigenvalues_toeplitz_complex_double
    module procedure check_correctness_eigenvalues_toeplitz_real_double
#ifdef WANT_SINGLE_PRECISION_REAL
    module procedure check_correctness_eigenvalues_toeplitz_real_single
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
    module procedure check_correctness_eigenvalues_toeplitz_complex_single
#endif
  end interface

  interface check_correctness_eigenvalues_frank
    module procedure check_correctness_eigenvalues_frank_complex_double
    module procedure check_correctness_eigenvalues_frank_real_double
#ifdef WANT_SINGLE_PRECISION_REAL
    module procedure check_correctness_eigenvalues_frank_real_single
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
    module procedure check_correctness_eigenvalues_frank_complex_single
#endif
  end interface

  interface check_correctness_cholesky
    module procedure check_correctness_cholesky_complex_double
    module procedure check_correctness_cholesky_real_double
#ifdef WANT_SINGLE_PRECISION_REAL
    module procedure check_correctness_cholesky_real_single
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
    module procedure check_correctness_cholesky_complex_single
#endif
  end interface

  interface check_correctness_hermitian_multiply
    module procedure check_correctness_hermitian_multiply_complex_double
    module procedure check_correctness_hermitian_multiply_real_double
#ifdef WANT_SINGLE_PRECISION_REAL
    module procedure check_correctness_hermitian_multiply_real_single
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
    module procedure check_correctness_hermitian_multiply_complex_single
#endif
  end interface


  contains

#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#include "../../src/general/precision_macros.h"
#include "test_check_correctness_template.F90"
#undef DOUBLE_PRECISION
#undef COMPLEXCASE

#ifdef WANT_SINGLE_PRECISION_COMPLEX

#define COMPLEXCASE 1
#define SINGLE_PRECISION 1
#include "../../src/general/precision_macros.h"
#include "test_check_correctness_template.F90"
#undef SINGLE_PRECISION
#undef COMPLEXCASE
#endif /* WANT_SINGLE_PRECISION_COMPLEX */

#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "../../src/general/precision_macros.h"
#include "test_check_correctness_template.F90"
#undef DOUBLE_PRECISION
#undef REALCASE

#ifdef WANT_SINGLE_PRECISION_REAL

#define REALCASE 1
#define SINGLE_PRECISION 1
#include "../../src/general/precision_macros.h"
#include "test_check_correctness_template.F90"
#undef SINGLE_PRECISION
#undef REALCASE


#endif /* WANT_SINGLE_PRECISION_REAL */

#include "../../src/general/prow_pcol.F90"
#include "../../src/general/map_global_to_local.F90"

end module
