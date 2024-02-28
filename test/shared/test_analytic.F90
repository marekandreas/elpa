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

#ifdef HAVE_64BIT_INTEGER_MATH_SUPPORT
#define TEST_INT_TYPE integer(kind=c_int64_t)
#define INT_TYPE c_int64_t
#else
#define TEST_INT_TYPE integer(kind=c_int32_t)
#define INT_TYPE c_int32_t
#endif
#ifdef HAVE_64BIT_INTEGER_MPI_SUPPORT
#define TEST_INT_MPI_TYPE integer(kind=c_int64_t)
#define INT_MPI_TYPE c_int64_t
#else
#define TEST_INT_MPI_TYPE integer(kind=c_int32_t)
#define INT_MPI_TYPE c_int32_t
#endif

module test_analytic

  use test_util
#ifdef HAVE_DETAILED_TIMINGS
  use ftimings
#else
  use timings_dummy
#endif
  use precision_for_tests

  interface prepare_matrix_analytic
    module procedure prepare_matrix_analytic_complex_double
    module procedure prepare_matrix_analytic_real_double
#ifdef WANT_SINGLE_PRECISION_REAL
    module procedure prepare_matrix_analytic_real_single
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
    module procedure prepare_matrix_analytic_complex_single
#endif
  end interface

  interface check_correctness_analytic
    module procedure check_correctness_analytic_complex_double
    module procedure check_correctness_analytic_real_double
#ifdef WANT_SINGLE_PRECISION_REAL
    module procedure check_correctness_analytic_real_single
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
    module procedure check_correctness_analytic_complex_single
#endif
  end interface


  interface print_matrix
    module procedure print_matrix_complex_double
    module procedure print_matrix_real_double
#ifdef WANT_SINGLE_PRECISION_REAL
    module procedure print_matrix_real_single
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
    module procedure print_matrix_complex_single
#endif
  end interface

  TEST_INT_TYPE, parameter, private  :: num_primes = 3
#ifdef BUILD_FUGAKU
  TEST_INT_TYPE, private  :: primes(num_primes)
#else
  TEST_INT_TYPE, parameter, private  :: primes(num_primes) = (/2,3,5/)
#endif

  TEST_INT_TYPE, parameter, private  :: ANALYTIC_MATRIX = 0
  TEST_INT_TYPE, parameter, private  :: ANALYTIC_EIGENVECTORS = 1
  TEST_INT_TYPE, parameter, private  :: ANALYTIC_EIGENVALUES = 2

  private map_global_array_index_to_local_index
  contains

  function decompose(num, decomposition) result(possible)
    implicit none
    TEST_INT_TYPE, intent(in)   :: num
    TEST_INT_TYPE, intent(out)  :: decomposition(num_primes)
    logical                        :: possible
    TEST_INT_TYPE               :: reminder, prime, prime_id

#ifdef BUILD_FUGAKU
    primes(1) = 2
    primes(2) = 3
    primes(3) = 5
#endif
    decomposition = 0
    possible = .true.
    reminder = num
    do prime_id = 1, num_primes
      prime = primes(prime_id)
      do while (MOD(reminder, prime) == 0)
        decomposition(prime_id) = decomposition(prime_id) + 1
        reminder = reminder / prime
      end do
    end do
    if(reminder > 1) then
      possible = .false.
    end if
  end function

  function compose(decomposition) result(num)
    implicit none
    TEST_INT_TYPE, intent(in)   :: decomposition(num_primes)
    TEST_INT_TYPE               :: num, prime_id

    num = 1;
#ifdef BUILD_FUGAKU
    primes(1) = 2
    primes(2) = 3
    primes(3) = 5
#endif
    do prime_id = 1, num_primes
      num = num * primes(prime_id) ** decomposition(prime_id)
    end do
  end function


#include "../../src/general/prow_pcol.F90"
#include "../../src/general/map_global_to_local.F90"


#define COMPLEXCASE 1
#define DOUBLE_PRECISION 1
#include "../../src/general/precision_macros.h"
#include "test_analytic_template.F90"
#undef DOUBLE_PRECISION
#undef COMPLEXCASE

#ifdef WANT_SINGLE_PRECISION_COMPLEX

#define COMPLEXCASE 1
#define SINGLE_PRECISION 1
#include "../../src/general/precision_macros.h"
#include "test_analytic_template.F90"
#undef SINGLE_PRECISION
#undef COMPLEXCASE

#endif /* WANT_SINGLE_PRECISION_COMPLEX */

#define REALCASE 1
#define DOUBLE_PRECISION 1
#include "../../src/general/precision_macros.h"
#include "test_analytic_template.F90"
#undef DOUBLE_PRECISION
#undef REALCASE

#ifdef WANT_SINGLE_PRECISION_REAL

#define REALCASE 1
#define SINGLE_PRECISION 1
#include "../../src/general/precision_macros.h"
#include "test_analytic_template.F90"
#undef SINGLE_PRECISION
#undef REALCASE

#endif /* WANT_SINGLE_PRECISION_REAL */


end module
