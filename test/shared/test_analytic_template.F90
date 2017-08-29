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

  subroutine prepare_matrix_analytic_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &(na, a, nblk, myid, np_rows, np_cols, my_prow, my_pcol)
    implicit none
    integer(kind=ik), intent(in)    :: na, nblk, myid, np_rows, np_cols, my_prow, my_pcol
    MATH_DATATYPE(kind=REAL_DATATYPE), intent(inout)   :: a(:,:)

    integer(kind=ik) :: globI, globJ, locI, locJ, levels(num_primes)

    ! for debug only, do it systematicaly somehow ... unit tests
    call check_module_sanity_&
            &MATH_DATATYPE&
            &_&
            &PRECISION&
            &(myid)

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
           a(locI, locJ) = analytic_matrix_&
              &MATH_DATATYPE&
              &_&
              &PRECISION&
              &(na, globI, globJ)
        end if
      end do
    end do

  end subroutine

  function check_correctness_analytic_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &(na, nev, ev, z, nblk, myid, np_rows, np_cols, my_prow, my_pcol, check_all_evals) result(status)
    implicit none
#include "../../src/general/precision_kinds.F90"
    integer(kind=ik), intent(in)    :: na, nev, nblk, myid, np_rows, np_cols, my_prow, my_pcol
    integer(kind=ik)                :: status, mpierr
    MATH_DATATYPE(kind=rck), intent(inout)   :: z(:,:)
    real(kind=rk), intent(inout)   :: ev(:)
    logical, intent(in)            :: check_all_evals

    integer(kind=ik) :: globI, globJ, locI, locJ, levels(num_primes)
    real(kind=rk)   :: diff, max_z_diff, max_ev_diff, glob_max_z_diff, max_curr_z_diff 
#ifdef DOUBLE_PRECISION
    real(kind=rk), parameter   :: tol_eigenvalues = 5e-14_rk8
    real(kind=rk), parameter   :: tol_eigenvectors = 6e-11_rk8
#endif
#ifdef SINGLE_PRECISION
    ! tolerance needs to be very high due to qr tests
    ! it should be distinguished somehow!
    real(kind=rk), parameter   :: tol_eigenvalues = 7e-6_rk4
    real(kind=rk), parameter   :: tol_eigenvectors = 4e-3_rk4
#endif
    real(kind=rk)             :: computed_ev, expected_ev
    MATH_DATATYPE(kind=rck)   :: computed_z,  expected_z

    MATH_DATATYPE(kind=rck)   :: max_value_for_normalization, computed_z_on_max_position, normalization_quotient
    integer(kind=ik)          :: max_value_idx, rank_with_max, rank_with_max_reduced, num_checked_evals


    if(.not. decompose(na, levels)) then
      print *, "can not decomopse matrix size"
      stop 1
    end if

    if(check_all_evals) then
        num_checked_evals = na
    else
        num_checked_evals = nev
    endif
    !call print_matrix(myid, na, z, "z")
    max_z_diff = 0.0_rk
    max_ev_diff = 0.0_rk
    do globJ = 1, num_checked_evals
      computed_ev = ev(globJ)
      expected_ev = analytic_eigenvalues_real_&
              &PRECISION&
              &(na, globJ)
      diff = abs(computed_ev - expected_ev)
      max_ev_diff = max(diff, max_ev_diff)
    end do

    do globJ = 1, nev
      max_curr_z_diff = 0.0_rk

      ! eigenvectors are unique up to multiplication by scalar (complex in complex case)
      ! to be able to compare them with analytic, we have to normalize them somehow
      ! we will find a value in analytic eigenvector with highest absolut value and enforce
      ! such multiple of computed eigenvector, that the value on corresponding position is the same
      max_value_for_normalization = 0.0_rk
      max_value_idx = -1
      do globI = 1, na
        expected_z = analytic_eigenvectors_&
              &MATH_DATATYPE&
              &_&
              &PRECISION&
              &(na, globI, globJ)
        if(abs(expected_z) > abs(max_value_for_normalization)) then
          max_value_for_normalization = expected_z
          max_value_idx = globI
        end if
      end do

      assert(max_value_idx >= 0)
      if(map_global_array_index_to_local_index(max_value_idx, globJ, locI, locJ, &
               nblk, np_rows, np_cols, my_prow, my_pcol)) then
        rank_with_max = myid
        computed_z_on_max_position = z(locI, locJ)
      else
        rank_with_max = -1
      end if

#ifdef WITH_MPI
      call MPI_Allreduce(rank_with_max, rank_with_max_reduced, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD, mpierr)
      call MPI_Bcast(computed_z_on_max_position, 1, MPI_MATH_DATATYPE_PRECISION, rank_with_max_reduced, MPI_COMM_WORLD, mpierr)
#endif
      !write(*,*) computed_z_on_max_position, max_value_for_normalization
      normalization_quotient = max_value_for_normalization / computed_z_on_max_position
      do globI = 1, na
        if(map_global_array_index_to_local_index(globI, globJ, locI, locJ, &
                 nblk, np_rows, np_cols, my_prow, my_pcol)) then
           computed_z = z(locI, locJ)
           expected_z = analytic_eigenvectors_&
              &MATH_DATATYPE&
              &_&
              &PRECISION&
              &(na, globI, globJ)
           max_curr_z_diff = max(abs(normalization_quotient * computed_z - expected_z), max_curr_z_diff)
        end if
      end do
      ! we have max difference of one of the eigenvectors, update global
      max_z_diff = max(max_z_diff, max_curr_z_diff)
    end do

#ifdef WITH_MPI
    call mpi_allreduce(max_z_diff, glob_max_z_diff, 1, MPI_REAL_PRECISION, MPI_MAX, MPI_COMM_WORLD, mpierr)
#else
    glob_max_z_diff = max_z_diff
#endif
    if(myid == 0) print *, 'Maximum error in eigenvalues      :', max_ev_diff
    if(myid == 0) print *, 'Maximum error in eigenvectors     :', glob_max_z_diff
    status = 0
    if (nev .gt. 2) then
      if (max_ev_diff .gt. tol_eigenvalues .or. max_ev_diff .eq. 0.0_rk) status = 1
      if (glob_max_z_diff .gt. tol_eigenvectors .or. glob_max_z_diff .eq. 0.0_rk) status = 1
    else
      if (max_ev_diff .gt. tol_eigenvalues) status = 1
      if (glob_max_z_diff .gt. tol_eigenvectors) status = 1
    endif
  end function


  function analytic_matrix_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &(na, i, j) result(element)
    implicit none
    integer(kind=ik), intent(in) :: na, i, j
    MATH_DATATYPE(kind=REAL_DATATYPE)     :: element

    element = analytic_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &(na, i, j, ANALYTIC_MATRIX)

  end function

  function analytic_eigenvectors_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &(na, i, j) result(element)
    implicit none
    integer(kind=ik), intent(in) :: na, i, j
    MATH_DATATYPE(kind=REAL_DATATYPE)               :: element

    element = analytic_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &(na, i, j, ANALYTIC_EIGENVECTORS)

  end function

  function analytic_eigenvalues_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &(na, i) result(element)
    implicit none
    integer(kind=ik), intent(in) :: na, i
    real(kind=REAL_DATATYPE)              :: element

    element = analytic_real_&
    &PRECISION&
    &(na, i, i, ANALYTIC_EIGENVALUES)

  end function

  function analytic_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &(na, i, j, what) result(element)
    implicit none
#include "../../src/general/precision_kinds.F90"
    integer(kind=ik), intent(in)   :: na, i, j, what
    MATH_DATATYPE(kind=rck)        :: element, mat2x2(2,2), mat(5,5)
    real(kind=rk)                  :: a, am, amp
    integer(kind=ik)               :: levels(num_primes)
    integer(kind=ik)               :: ii, jj, m, prime_id, prime, total_level, level

    real(kind=rk), parameter      :: s = 0.5_rk
    real(kind=rk), parameter      :: c = 0.86602540378443864679_rk
    real(kind=rk), parameter      :: sq2 = 1.4142135623730950488_rk

    real(kind=rk), parameter      :: largest_ev = 2.0_rk

    assert(i <= na)
    assert(j <= na)
    assert(i >= 0)
    assert(j >= 0)
    assert(decompose(na, levels))
    ! go to zero-based indexing
    ii = i - 1
    jj = j - 1
    if (na .gt. 2) then
      a = exp(log(largest_ev)/(na-1))
    else
      a = exp(log(largest_ev)/(1))
    endif

    element = 1.0_rck
#ifdef COMPLEXCASE
    element = (1.0_rk, 0.0_rk)
#endif
    total_level = 0
    am = a
    do prime_id = 1,num_primes
      prime = primes(prime_id)
      do  level = 1, levels(prime_id)
        amp = am**(prime-1)
        total_level = total_level + 1
        if(what == ANALYTIC_MATRIX) then
#ifdef REALCASE
          mat2x2 = reshape((/ c*c + amp * s*s, (amp - 1.0_rk) * s*c,  &
                           (amp - 1.0_rk) * s*c, s*s + amp * c*c  /), &
                                      (/2, 2/), order=(/2,1/))
#endif
#ifdef COMPLEXCASE
          mat2x2 = reshape((/ 0.5_rck * (amp + 1.0_rck) * (1.0_rk, 0.0_rk),   sq2/4.0_rk * (amp - 1.0_rk) * (1.0_rk, 1.0_rk),   &
                              sq2/4.0_rk * (amp - 1.0_rk) * (1.0_rk, -1.0_rk),  0.5_rck * (amp + 1.0_rck) * (1.0_rk, 0.0_rk) /), &
                                      (/2, 2/), order=(/2,1/))
#endif
        else if(what == ANALYTIC_EIGENVECTORS) then
#ifdef REALCASE
          mat2x2 = reshape((/ c, s,  &
                           -s,  c  /), &
                                (/2, 2/), order=(/2,1/))
#endif
#ifdef COMPLEXCASE
          mat2x2 = reshape((/ -sq2/2.0_rck * (1.0_rk, 0.0_rk),       -sq2/2.0_rck * (1.0_rk, 0.0_rk),  &
                              0.5_rk * (1.0_rk, -1.0_rk),  0.5_rk * (-1.0_rk, 1.0_rk)  /), &
                                (/2, 2/), order=(/2,1/))
#endif
        else if(what == ANALYTIC_EIGENVALUES) then
          mat2x2 = reshape((/ 1.0_rck, 0.0_rck,  &
                           0.0_rck, amp  /), &
                                 (/2, 2/), order=(/2,1/))
        else
          assert(.false.)
        end if

        mat = 0.0_rck
        if(prime == 2) then
          mat(1:2, 1:2) = mat2x2
        else if(prime == 3) then
          mat((/1,3/),(/1,3/)) = mat2x2
          if(what == ANALYTIC_EIGENVECTORS) then
            mat(2,2) = 1.0_rck
          else
            mat(2,2) = am
          end if
        else if(prime == 5) then
          mat((/1,5/),(/1,5/)) = mat2x2
          if(what == ANALYTIC_EIGENVECTORS) then
            mat(2,2) = 1.0_rck
            mat(3,3) = 1.0_rck
            mat(4,4) = 1.0_rck
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


  subroutine print_matrix_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &(myid, na, mat, mat_name)
    implicit none
#include "../../src/general/precision_kinds.F90"
    integer(kind=ik), intent(in)    :: myid, na
    character(len=*), intent(in)    :: mat_name
    MATH_DATATYPE(kind=rck)         :: mat(na, na)
    integer(kind=ik)                :: i,j
    character(len=20)               :: na_str

    if(myid .ne. 0) &
      return
    write(*,*) "Matrix: "//trim(mat_name)
    write(na_str, *) na
    do i = 1, na
#ifdef REALCASE
      write(*, '('//trim(na_str)//'f8.3)') mat(i, :)
#endif
#ifdef COMPLEXCASE
      write(*,'('//trim(na_str)//'(A,f8.3,A,f8.3,A))') ('(', real(mat(i,j)), ',', aimag(mat(i,j)), ')', j=1,na)
#endif
    end do
    write(*,*)
  end subroutine


  subroutine check_matrices_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &(myid, na)
    implicit none
#include "../../src/general/precision_kinds.F90"
    integer(kind=ik), intent(in)    :: myid, na
    MATH_DATATYPE(kind=rck)                  :: A(na, na), S(na, na), L(na, na), res(na, na)
    integer(kind=ik)                :: i, j, decomposition(num_primes)

    assert(decompose(na, decomposition))

    do i = 1, na
      do j = 1, na
        A(i,j) = analytic_matrix_&
              &MATH_DATATYPE&
              &_&
              &PRECISION&
              &(na, i, j)
        S(i,j) = analytic_eigenvectors_&
              &MATH_DATATYPE&
              &_&
              &PRECISION&
              &(na, i, j)
        L(i,j) = analytic_&
              &MATH_DATATYPE&
              &_&
              &PRECISION&
              &(na, i, j, ANALYTIC_EIGENVALUES)
      end do
    end do

    res = matmul(A,S) - matmul(S,L)
#ifdef DOUBLE_PRECISION
    assert(maxval(abs(res)) < 1e-8)
#elif SINGLE_PRECISION
    assert(maxval(abs(res)) < 1e-4)
#else
    assert(.false.)
#endif
    if(.false.) then
    !if(na == 2 .or. na == 5) then
      call print_matrix(myid, na, A, "A")
      call print_matrix(myid, na, S, "S")
      call print_matrix(myid, na, L, "L")

      call print_matrix(myid, na, matmul(A,S), "AS")
      call print_matrix(myid, na, matmul(S,L), "SL")

      call print_matrix(myid, na, res , "res")
    end if

  end subroutine

  subroutine check_module_sanity_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &(myid)
    implicit none
    integer(kind=ik), intent(in)   :: myid
    integer(kind=ik)               :: decomposition(num_primes), i
    integer(kind=ik), parameter    :: check_sizes(7) = (/2, 3, 5, 6, 10, 25, 150/)
    if(myid == 0) print *, "Checking test_analytic module sanity.... "
    assert(decompose(1500, decomposition))
    assert(all(decomposition == (/2,1,3/)))
    assert(decompose(6,decomposition))
    assert(all(decomposition == (/1,1,0/)))

    do i =1, size(check_sizes)
      call check_matrices_&
          &MATH_DATATYPE&
          &_&
          &PRECISION&
          &(myid, check_sizes(i))
    end do

    if(myid == 0) print *, "Checking test_analytic module sanity.... DONE"

  end subroutine
