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

module test_analytic

  use test_util
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

  integer(kind=ik), parameter, private  :: num_primes = 3
  integer(kind=ik), parameter, private  :: primes(num_primes) = (/2,3,5/)

  integer(kind=ik), parameter, private  :: ANALYTIC_MATRIX = 0
  integer(kind=ik), parameter, private  :: ANALYTIC_EIGENVECTORS = 1
  integer(kind=ik), parameter, private  :: ANALYTIC_EIGENVALUES = 2

  contains

  function decompose(num, decomposition) result(possible)
    implicit none
    integer(kind=ik), intent(in)   :: num
    integer(kind=ik), intent(out)  :: decomposition(num_primes)
    logical                        :: possible
    integer(kind=ik)               :: reminder, prime, prime_id

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
    integer(kind=ik), intent(in)   :: decomposition(num_primes)
    integer(kind=ik)               :: num, prime_id

    num = 1;
    do prime_id = 1, num_primes
      num = num * primes(prime_id) ** decomposition(prime_id)
    end do
  end function

  subroutine print_matrix(myid, na, mat, mat_name)
    implicit none
    integer(kind=ik), intent(in)    :: myid, na
    character(len=*), intent(in)    :: mat_name
    real(kind=rk8)                  :: mat(na, na)
    integer(kind=ik)                :: i
    character(len=20)               :: str

    if(myid .ne. 0) &
      return
    write(*,*) "Matrix: "//trim(mat_name)
    write(str, *) na
    do i = 1, na
      write(*, '('//trim(str)//'f8.3)') mat(i, :)
    end do
    write(*,*)
  end subroutine


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

  subroutine check_matrices(myid, na)
    implicit none
    integer(kind=ik), intent(in)    :: myid, na
    real(kind=rk8)                  :: A(na, na), S(na, na), L(na, na), res(na, na)
    integer(kind=ik)                :: i, j, decomposition(num_primes)

    assert(decompose(na, decomposition))

    do i = 1, na
      do j = 1, na
        A(i,j) = analytic_matrix_real_double(na, i, j)
        S(i,j) = analytic_eigenvectors_real_double(na, i, j)
        L(i,j) = analytic_real_double(na, i, j, ANALYTIC_EIGENVALUES)
      end do
    end do

    res = matmul(A,S) - matmul(S,L)
    assert(maxval(abs(res)) < 1e-8)

    !call print_matrix(myid, na, A, "A")
    !call print_matrix(myid, na, S, "S")
    !call print_matrix(myid, na, L, "L")
    !call print_matrix(myid, na, res , "res")

  end subroutine

  subroutine check_module_sanity(myid)
    implicit none
    integer(kind=ik), intent(in)   :: myid
    integer(kind=ik)               :: decomposition(num_primes)

    if(myid == 0) print *, "Checking test_analytic module sanity.... "
    assert(decompose(1500, decomposition))
    assert(all(decomposition == (/2,1,3/)))
    assert(decompose(6,decomposition))
    assert(all(decomposition == (/1,1,0/)))

    call check_matrices(myid, 2)
    call check_matrices(myid, 3)
    call check_matrices(myid, 5)
    call check_matrices(myid, 6)
    call check_matrices(myid, 10)
    call check_matrices(myid, 25)
    call check_matrices(myid, 150)

    if(myid == 0) print *, "Checking test_analytic module sanity.... DONE"

  end subroutine

end module
