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
    module procedure prepare_matrix_analytic_real_double
  end interface

  interface check_correctness_analytic
    module procedure check_correctness_analytic_real_double
  end interface

  integer(kind=ik), parameter, private  :: num_primes = 3
  integer(kind=ik), parameter, private  :: primes(num_primes) = (/2,3,5/)
  real(kind=rk8), parameter, private :: ZERO = 0.0_rk8, ONE = 1.0_rk8

  contains

#include "../../src/general/prow_pcol.X90"
#include "../../src/general/map_global_to_local.X90"

  subroutine prepare_matrix_analytic_real_double (na, a, nblk, myid, np_rows, &
                            np_cols, my_prow, my_pcol)
    implicit none

    integer(kind=ik), intent(in)    :: na, nblk, myid, np_rows, np_cols, my_prow, my_pcol
    real(kind=rk8), intent(inout)   :: a(:,:)

    integer(kind=ik) :: globI, globJ, locI, locJ, levels(num_primes)

    ! for debug only, do it systematicaly somehow ... unit tests
    ! call check_module_sanity(myid)

    if(.not. decompose(na, levels)) then
      if(myid == 0) then
        print *, "Analytic test can be run only with matrix sizes of the form 2^n * 3^m * 5^o"
        stop 1
      end if
    end if

    do globI = 1, na
      do globJ = 1, na
        if(map_global_array_index_to_local_index(globI, globJ, locI, locJ, &
                 nblk, np_rows, np_cols, my_prow, my_pcol)) then
           a(locI, locJ) = analytic_matrix(na, globI, globJ)
        end if
      end do
    end do

  end subroutine

  function check_correctness_analytic_real_double (na, nev, ev, z, nblk, myid, np_rows, &
                            np_cols, my_prow, my_pcol) result(status)
    implicit none

    integer(kind=ik), intent(in)    :: na, nev, nblk, myid, np_rows, np_cols, my_prow, my_pcol
    integer(kind=ik)                :: status, mpierr
    real(kind=rk8), intent(inout)   :: z(:,:)
    real(kind=rk8), intent(inout)   :: ev(:)

    integer(kind=ik) :: globI, globJ, locI, locJ, levels(num_primes)
    real(kind=rk8)   :: diff, max_z_diff, max_ev_diff, glob_max_z_diff, max_curr_z_diff_minus, max_curr_z_diff_plus
    real(kind=rk8)   :: computed, expected

    if(.not. decompose(na, levels)) then
      print *, "can not decomopse matrix size"
      stop 1
    end if

    max_z_diff = ZERO
    max_ev_diff = ZERO
    do globJ = 1, na
      diff = abs(ev(globJ) - analytic_eigenvalues(na, globJ))
      max_ev_diff = max(diff, max_ev_diff)
    end do

    do globJ = 1, nev
      ! calculated eigenvector can be in opposite direction
      max_curr_z_diff_minus = ZERO
      max_curr_z_diff_plus  = ZERO
      do globI = 1, na
        if(map_global_array_index_to_local_index(globI, globJ, locI, locJ, &
                 nblk, np_rows, np_cols, my_prow, my_pcol)) then
           computed = z(locI, locJ)
           expected = analytic_eigenvectors(na, globI, globJ)
           max_curr_z_diff_minus = max(abs(computed - expected), max_curr_z_diff_minus)
           max_curr_z_diff_plus = max(abs(computed + expected), max_curr_z_diff_plus)
        end if
      end do
      ! we have max difference of one of the eigenvectors, update global
      max_z_diff = max(max_z_diff, min(max_curr_z_diff_minus, max_curr_z_diff_plus))
    end do

#ifdef WITH_MPI
    call mpi_allreduce(max_z_diff, glob_max_z_diff, 1, MPI_REAL8, MPI_MAX, MPI_COMM_WORLD, mpierr)
#else
    glob_max_z_diff = max_z_diff
#endif
    if(myid == 0) print *, 'Maximum error in eigenvalues      :', max_ev_diff
    if(myid == 0) print *, 'Maximum error in eigenvectors     :', glob_max_z_diff
    status = 0
    if (max_ev_diff .gt. 5e-14_rk8 .or. max_ev_diff .eq. ZERO) status = 1
    if (glob_max_z_diff .gt. 6e-11_rk8 .or. glob_max_z_diff .eq. ZERO) status = 1
  end function

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

#define ANALYTIC_MATRIX 0
#define ANALYTIC_EIGENVECTORS 1
#define ANALYTIC_EIGENVALUES 2

  function analytic_matrix(na, i, j) result(element)
    implicit none
    integer(kind=ik), intent(in) :: na, i, j
    real(kind=rk8)               :: element

    element = analytic(na, i, j, ANALYTIC_MATRIX)

  end function

  function analytic_eigenvectors(na, i, j) result(element)
    implicit none
    integer(kind=ik), intent(in) :: na, i, j
    real(kind=rk8)               :: element

    element = analytic(na, i, j, ANALYTIC_EIGENVECTORS)

  end function

  function analytic_eigenvalues(na, i) result(element)
    implicit none
    integer(kind=ik), intent(in) :: na, i
    real(kind=rk8)               :: element

    element = analytic(na, i, i, ANALYTIC_EIGENVALUES)

  end function



  function analytic(na, i, j, what) result(element)
    implicit none
    integer(kind=ik), intent(in)   :: na, i, j, what
    real(kind=rk8)                 :: element, am
    real(kind=rk8)                 :: a, s, c, mat2x2(2,2), mat(5,5)
    integer(kind=ik)               :: levels(num_primes)
    integer(kind=ik)               :: ii, jj, m, prime_id, prime, total_level, level

    real(kind=rk8), parameter      :: largest_ev = 2.0_rk8

    assert(i <= na)
    assert(j <= na)
    assert(i >= 0)
    assert(j >= 0)
    assert(decompose(na, levels))
    ! go to zero-based indexing
    ii = i - 1
    jj = j - 1
    a = exp(log(largest_ev)/(na-1))
    s = 0.5_rk8
    c = sqrt(3.0_rk8)/2.0_rk8
    element = ONE
    total_level = 0
    am = a
    do prime_id = 1,num_primes
      prime = primes(prime_id)
      do  level = 1, levels(prime_id)
        total_level = total_level + 1
        if(what == ANALYTIC_MATRIX) then
          mat2x2 = reshape((/ c*c + am**(prime-1) * s*s, (ONE-am**(prime-1)) * s*c,  &
                           (ONE-am**(prime-1)) * s*c, s*s + am**(prime-1) * c*c  /), &
                                      (/2, 2/))
        else if(what == ANALYTIC_EIGENVECTORS) then
          mat2x2 = reshape((/ c, s,  &
                           -s,  c  /), &
                                (/2, 2/))
        else if(what == ANALYTIC_EIGENVALUES) then
          mat2x2 = reshape((/ ONE, ZERO,  &
                           ZERO, am**(prime-1)  /), &
                                 (/2, 2/))
        else
          assert(.false.)
        end if

        mat = ZERO
        if(prime == 2) then
          mat(1:2, 1:2) = mat2x2
        else if(prime == 3) then
          mat((/1,3/),(/1,3/)) = mat2x2
          if(what == ANALYTIC_EIGENVECTORS) then
            mat(2,2) = ONE
          else
            mat(2,2) = am
          end if
        else if(prime == 5) then
          mat((/1,5/),(/1,5/)) = mat2x2
          if(what == ANALYTIC_EIGENVECTORS) then
            mat(2,2) = ONE
            mat(3,3) = ONE
            mat(4,4) = ONE
          else
            mat(2,2) = am
            mat(3,3) = am**2
            mat(4,4) = am**3
          end if
        else
          assert(.false.)
        end if

  !      write(*,*) "calc value, elem: ", element, ", mat: ", mod(ii,2), mod(jj,2),  mat(mod(ii,2), mod(jj,2)), "am ", am
  !      write(*,*) " matrix mat", mat
        element = element * mat(mod(ii,prime) + 1, mod(jj,prime) + 1)
        ii = ii / prime
        jj = jj / prime

        am = am**prime
      end do
    end do
    !write(*,*) "returning value ", element
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

  subroutine check_matrices(myid, na)
    implicit none
    integer(kind=ik), intent(in)    :: myid, na
    real(kind=rk8)                  :: A(na, na), S(na, na), L(na, na)
    integer(kind=ik)                :: i, j, decomposition(num_primes)

    assert(decompose(na, decomposition))

    do i = 1, na
      do j = 1, na
        A(i,j) = analytic_matrix(na, i, j)
        S(i,j) = analytic_eigenvectors(na, i, j)
        L(i,j) = analytic(na, i, j, ANALYTIC_EIGENVALUES)
      end do
    end do

    call print_matrix(myid, na, A, "A")
    call print_matrix(myid, na, S, "S")
    call print_matrix(myid, na, L, "L")

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

    if(myid == 0) print *, "Checking test_analytic module sanity.... DONE"

  end subroutine

end module
