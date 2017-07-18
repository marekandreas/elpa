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

!#include "assert.h"  ! why complains?
module test_analytic

  use test_util

  interface prepare_matrix_analytic
    module procedure prepare_matrix_analytic_real_double
  end interface

  interface check_correctness_analytic
    module procedure check_correctness_analytic_real_double
  end interface

  contains

  subroutine prepare_matrix_analytic_real_double (na, a, nblk, myid, np_rows, &
                            np_cols, my_prow, my_pcol)
    use elpa_utilities
    implicit none

    integer(kind=ik), intent(in)    :: na, nblk, myid, np_rows, np_cols, my_prow, my_pcol
    real(kind=rk8), intent(inout)   :: a(:,:)

    integer(kind=ik) :: globI, globJ, locI, locJ, levels

    if(.not. decompose(na, levels)) then
      print *, "can not decomopse matrix size"
      stop 1
    end if

    do globI = 1, na
      do globJ = 1, na
        if(map_global_array_index_to_local_index(globI, globJ, locI, locJ, &
                 nblk, np_rows, np_cols, my_prow, my_pcol)) then
           a(locI, locJ) = analytic_matrix(levels, globI, globJ)
        end if
      end do
    end do

  end subroutine

  function check_correctness_analytic_real_double (na, nev, ev, z, nblk, myid, np_rows, &
                            np_cols, my_prow, my_pcol) result(status)
    use elpa_utilities
    implicit none

    integer(kind=ik), intent(in)    :: na, nev, nblk, myid, np_rows, np_cols, my_prow, my_pcol
    integer(kind=ik)                :: status
    real(kind=rk8), intent(inout)   :: z(:,:)
    real(kind=rk8), intent(inout)   :: ev(:)

    integer(kind=ik) :: globI, globJ, locI, locJ, levels
    real(kind=rk8)   :: diff, max_z_diff, max_ev_diff, glob_max_z_diff, max_curr_z_diff_minus, max_curr_z_diff_plus
    real(kind=rk8)   :: computed, expected

    if(.not. decompose(na, levels)) then
      print *, "can not decomopse matrix size"
      stop 1
    end if

    max_z_diff = 0.0_rk8
    max_ev_diff = 0.0_rk8
    do globJ = 1, nev
      diff = abs(ev(globJ) - analytic_eigenvalues(levels, globJ))
      max_ev_diff = max(diff, max_ev_diff)

      ! calculated eigenvector can be in opposite direction
      max_curr_z_diff_minus = 0.0_rk8
      max_curr_z_diff_plus  = 0.0_rk8
      do globI = 1, na
        if(map_global_array_index_to_local_index(globI, globJ, locI, locJ, &
                 nblk, np_rows, np_cols, my_prow, my_pcol)) then
           computed = z(locI, locJ)
           expected = analytic_eigenvectors(levels, globI, globJ)
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
    if(myid == 0) print *, 'Maximal error in eigenvalues      :', max_ev_diff
    if(myid == 0) print *, 'Maximal error in eigenvectors     :', glob_max_z_diff
    status = 0
    if (max_ev_diff .gt. 5e-14_rk8 .or. max_ev_diff .eq. 0.0_rk8) status = 1
    if (glob_max_z_diff .gt. 1e-12_rk8 .or. glob_max_z_diff .eq. 0.0_rk8) status = 1
  end function

  function decompose(num, decomposition) result(possible)
    implicit none
    integer(kind=ik), intent(in)   :: num
    integer(kind=ik), intent(out)  :: decomposition
    logical                        :: possible
    integer(kind=ik)               :: reminder

    decomposition = 0
    possible = .true.
    reminder = num
    do while (reminder > 1)
      if (MOD(reminder, 2) == 0) then
        decomposition = decomposition + 1
        reminder = reminder / 2
      else
        possible = .false.
      end if
    end do
  end function

#define ANALYTIC_MATRIX 0
#define ANALYTIC_EIGENVECTORS 1
#define ANALYTIC_EIGENVALUES 2

  function analytic_matrix(levels, i, j) result(element)
    implicit none
    integer(kind=ik), intent(in) :: levels, i, j
    real(kind=rk8)               :: element

    element = analytic(levels, i, j, ANALYTIC_MATRIX)

  end function

  function analytic_eigenvectors(levels, i, j) result(element)
    implicit none
    integer(kind=ik), intent(in) :: levels, i, j
    real(kind=rk8)               :: element

    element = analytic(levels, i, j, ANALYTIC_EIGENVECTORS)

  end function

  function analytic_eigenvalues(levels, i) result(element)
    implicit none
    integer(kind=ik), intent(in) :: levels, i
    real(kind=rk8)               :: element

    element = analytic(levels, i, i, ANALYTIC_EIGENVALUES)

  end function



  function analytic(n, i, j, what) result(element)
    implicit none
    integer(kind=ik), intent(in)   :: n, i, j, what
    real(kind=rk8)                 :: element, am
    real(kind=rk8)                 :: a, s, c, mat(2,2)
    integer(kind=ik)               :: ii, jj, m

!    assert(i < 2**n)
!    assert(j < 2**n)
!    assert(i >= 0)
!    assert(j >= 0)
    ! go to zero-based indexing
    ii = i - 1
    jj = j - 1
    a = get_a(n)
    s = 0.5_rk8
    c = sqrt(3.0_rk8)/2.0_rk8
    element = 1.0_rk8
    do  m = 1, n
      am = a**(2**(m-1))
      if(what == ANALYTIC_MATRIX) then
        mat = reshape((/ c*c + am * s*s, (1.0_rk8-am) * s*c,  &
                         (1.0_rk8-am) * s*c, s*s + am * c*c  /), &
                                    (/2, 2/))
      else if(what == ANALYTIC_EIGENVECTORS) then
        mat = reshape((/ c, s,  &
                         -s,  c  /), &
                              (/2, 2/))
      else if(what == ANALYTIC_EIGENVALUES) then
        mat = reshape((/ 1.0_rk8, 0.0_rk8,  &
                         0.0_rk8, am  /), &
                               (/2, 2/))
      else
        !assert(0)
      end if
!      write(*,*) "calc value, elem: ", element, ", mat: ", mod(ii,2), mod(jj,2),  mat(mod(ii,2), mod(jj,2)), "am ", am
!      write(*,*) " matrix mat", mat
      element = element * mat(mod(ii,2) + 1, mod(jj,2) + 1)
      ii = ii / 2
      jj = jj / 2
    end do
    !write(*,*) "returning value ", element
  end function

  function get_a(n) result (a)
    implicit none
    integer(kind=ik), intent(in)   :: n
    real(kind=rk8)                 :: a
    real(kind=rk8), parameter      :: largest_ev = 2.0

    a = exp(log(largest_ev)/(2**n-1))
  end function



end module
