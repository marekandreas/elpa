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

module test_prepare_matrix

  use precision_for_tests
  interface prepare_matrix_random
    module procedure prepare_matrix_random_complex_double
    module procedure prepare_matrix_random_real_double
#ifdef WANT_SINGLE_PRECISION_REAL
    module procedure prepare_matrix_random_real_single
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
    module procedure prepare_matrix_random_complex_single
#endif
   end interface


  interface prepare_matrix_random_spd
    module procedure prepare_matrix_random_spd_complex_double
    module procedure prepare_matrix_random_spd_real_double
#ifdef WANT_SINGLE_PRECISION_REAL
    module procedure prepare_matrix_random_spd_real_single
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
    module procedure prepare_matrix_random_spd_complex_single
#endif
   end interface

  interface prepare_matrix_random_triangular
    module procedure prepare_matrix_random_triangular_complex_double
    module procedure prepare_matrix_random_triangular_real_double
#ifdef WANT_SINGLE_PRECISION_REAL
    module procedure prepare_matrix_random_triangular_real_single
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
    module procedure prepare_matrix_random_triangular_complex_single
#endif
   end interface

  interface prepare_matrix_toeplitz
    module procedure prepare_matrix_toeplitz_complex_double
    module procedure prepare_matrix_toeplitz_real_double
    module procedure prepare_matrix_toeplitz_mixed_complex_complex_double
#ifdef WANT_SINGLE_PRECISION_REAL
    module procedure prepare_matrix_toeplitz_real_single
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
    module procedure prepare_matrix_toeplitz_complex_single
    module procedure prepare_matrix_toeplitz_mixed_complex_complex_single
#endif
   end interface

  interface prepare_matrix_frank
    module procedure prepare_matrix_frank_complex_double
    module procedure prepare_matrix_frank_real_double
#ifdef WANT_SINGLE_PRECISION_REAL
    module procedure prepare_matrix_frank_real_single
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
    module procedure prepare_matrix_frank_complex_single
#endif
   end interface

  interface prepare_matrix_unit
    module procedure prepare_matrix_unit_complex_double
    module procedure prepare_matrix_unit_real_double
#ifdef WANT_SINGLE_PRECISION_REAL
    module procedure prepare_matrix_unit_real_single
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
    module procedure prepare_matrix_unit_complex_single
#endif
   end interface

   private prows, pcols, map_global_array_index_to_local_index

  contains

#include "../../src/general/prow_pcol.F90"
#include "../../src/general/map_global_to_local.F90"

#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#include "../../src/general/precision_macros.h"
#include "test_prepare_matrix_template.F90"
#undef DOUBLE_PRECISION
#undef COMPLEXCASE


#ifdef WANT_SINGLE_PRECISION_COMPLEX


#define COMPLEXCASE 1
#define SINGLE_PRECISION 1
#include "../../src/general/precision_macros.h"
#include "test_prepare_matrix_template.F90"
#undef SINGLE_PRECISION
#undef COMPLEXCASE
#endif /* WANT_SINGLE_PRECISION_COMPLEX */


#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "../../src/general/precision_macros.h"
#include "test_prepare_matrix_template.F90"
#undef DOUBLE_PRECISION
#undef REALCASE

#ifdef WANT_SINGLE_PRECISION_REAL


#define REALCASE 1
#define SINGLE_PRECISION 1
#include "../../src/general/precision_macros.h"
#include "test_prepare_matrix_template.F90"
#undef SINGLE_PRECISION
#undef REALCASE

#endif /* WANT_SINGLE_PRECISION_REAL */


end module
