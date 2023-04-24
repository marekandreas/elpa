!  (c) Copyright Pavel Kus, 2017, MPCDF
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


  subroutine prepare_matrix_analytic_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &(na, a, nblk, myid, np_rows, np_cols, my_prow, my_pcol, print_times)
    use precision_for_tests

    implicit none
    TEST_INT_TYPE, intent(in)                       :: na, nblk, myid, np_rows, np_cols, my_prow, my_pcol
    MATH_DATATYPE(kind=REAL_DATATYPE), intent(inout):: a(:,:)
    logical, optional                               :: print_times
    logical                                         :: print_timer
    TEST_INT_TYPE                                   :: globI, globJ, locI, locJ, pi, pj, levels(num_primes)
    integer(kind=c_int)                             :: loc_I, loc_J, p_i, p_j
#ifdef HAVE_DETAILED_TIMINGS
    type(timer_t)                                   :: timer
#else
    type(timer_dummy_t)                             :: timer
#endif

    call timer%enable()
    call timer%start("prepare_matrix_analytic")

    print_timer = .true.

    if (present(print_times)) then
      print_timer = print_times
    endif

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

    call timer%start("loop")
    do globI = 1, na

      p_i = prow(int(globI,kind=c_int), int(nblk,kind=c_int), int(np_rows,kind=c_int))
      pi = int(p_i,kind=INT_TYPE)
      if (my_prow .ne. pi) cycle

      do globJ = 1, na

        p_j = pcol(int(globJ,kind=c_int), int(nblk,kind=c_int), int(np_cols,kind=c_int))
        pj = int(p_j,kind=INT_TYPE)
        if (my_pcol .ne. pj) cycle

        if(map_global_array_index_to_local_index(int(globI,kind=c_int), int(globJ,kind=c_int), loc_I, loc_J, &
                 int(nblk,kind=c_int), int(np_rows,kind=c_int), int(np_cols,kind=c_int), &
                 int(my_prow,kind=c_int), int(my_pcol,kind=c_int) )) then
           locI = int(loc_i,kind=INT_TYPE)      
           locJ = int(loc_j,kind=INT_TYPE)      
           call timer%start("evaluation")
           a(locI, locJ) = analytic_matrix_&
              &MATH_DATATYPE&
              &_&
              &PRECISION&
              &(na, globI, globJ)
          call timer%stop("evaluation")
        else
          print *, "Warning ... error in preparation loop of the analytic test"
        end if
      end do
    end do
    call timer%stop("loop")

    call timer%stop("prepare_matrix_analytic")
    if(myid == 0 .and. print_timer) then
      call timer%print("prepare_matrix_analytic")
    end if
    call timer%free()
  end subroutine

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
    !c> void prepare_matrix_analytic_real_double_f(TEST_C_INT_TYPE na, 
    !c>                                           double *a,
    !c>                                           TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                           TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid,
    !c>                                           TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                           TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#else
    !c> void prepare_matrix_analytic_real_single_f(TEST_C_INT_TYPE na, 
    !c>                                           float *a,
    !c>                                           TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                           TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid,
    !c>                                           TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                           TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> void prepare_matrix_analytic_complex_double_f(TEST_C_INT_TYPE na, 
    !c>                                           double_complex *a,
    !c>                                           TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                           TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid,
    !c>                                           TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                           TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#else
    !c> void prepare_matrix_analytic_complex_single_f(TEST_C_INT_TYPE na, 
    !c>                                           float_complex *a,
    !c>                                           TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                           TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid,
    !c>                                           TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                           TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol);
#endif
#endif /* COMPLEXCASE */
  subroutine prepare_matrix_analytic_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &_f(na, a, na_rows, na_cols, nblk, myid, np_rows, np_cols, my_prow, my_pcol) &
    bind(C,name="prepare_matrix_analytic_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      &_f")
    use iso_c_binding
    use precision_for_tests

    implicit none
    TEST_INT_TYPE, value              :: na, na_rows, na_cols, nblk, myid, np_rows, np_cols, my_prow, my_pcol
    MATH_DATATYPE(kind=REAL_DATATYPE) ::  a(1:na_rows,1:na_cols)
  
    call prepare_matrix_analytic_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      & (na, a, nblk, myid, np_rows, np_cols, my_prow, my_pcol)
      
  end subroutine  
  
  !-----------------------------------------------------------------------------------------------------------
    
  function check_correctness_analytic_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &(na, nev, ev, z, nblk, myid, np_rows, np_cols, my_prow, my_pcol, check_all_evals, &
      check_eigenvectors, print_times) result(status)
    use precision_for_tests
    use test_util

    implicit none
#include "./test_precision_kinds.F90"
    TEST_INT_TYPE, intent(in)              :: na, nev, nblk, myid, np_rows, &
                                              np_cols, my_prow, my_pcol
    TEST_INT_TYPE                          :: status
    TEST_INT_MPI_TYPE                      :: mpierr
    MATH_DATATYPE(kind=rck), intent(inout) :: z(:,:)
    real(kind=rk), intent(inout)           :: ev(:)
    logical, intent(in)                    :: check_all_evals, check_eigenvectors

    TEST_INT_TYPE                          :: globI, globJ, locI, locJ, &
                                              levels(num_primes)
    integer(kind=c_int)                    :: loc_I, loc_J
    real(kind=rk)                          :: diff, max_z_diff, max_ev_diff, &
                                              glob_max_z_diff, max_curr_z_diff
#ifdef DOUBLE_PRECISION
    real(kind=rk), parameter               :: tol_eigenvalues = 5e-14_rk8
    real(kind=rk), parameter               :: tol_eigenvectors = 6e-10_rk8
#endif
#ifdef SINGLE_PRECISION
    ! tolerance needs to be very high due to qr tests
    ! it should be distinguished somehow!
    real(kind=rk), parameter               :: tol_eigenvalues = 9e-5_rk4
    real(kind=rk), parameter               :: tol_eigenvectors = 9e-2_rk4
#endif
    real(kind=rk)                          :: computed_ev, expected_ev
    MATH_DATATYPE(kind=rck)                :: computed_z,  expected_z

    MATH_DATATYPE(kind=rck)                :: max_value_for_normalization, &
                                              computed_z_on_max_position,  &
                                              normalization_quotient
    MATH_DATATYPE(kind=rck)                :: max_values_array(np_rows * np_cols), &
                                              corresponding_exact_value
    integer(kind=c_int)                    :: max_value_idx, rank_with_max, &
                                              rank_with_max_reduced,        &
                                              num_checked_evals
    integer(kind=c_int)                    :: max_idx_array(np_rows * np_cols), &
                                              rank
    logical, optional                      :: print_times
    logical                                :: print_timer

#ifdef HAVE_DETAILED_TIMINGS
    type(timer_t)    :: timer
#else
    type(timer_dummy_t)    :: timer
#endif

    call timer%enable()
    call timer%start("check_correctness_analytic")


    print_timer = .true.
    if (present(print_times)) then
      print_timer = print_times
    endif

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
    call timer%start("loop_eigenvalues")
    do globJ = 1, num_checked_evals
      computed_ev = ev(globJ)
      call timer%start("evaluation")
      expected_ev = analytic_eigenvalues_real_&
              &PRECISION&
              &(na, globJ)
      call timer%stop("evaluation")
      diff = abs(computed_ev - expected_ev)
      max_ev_diff = max(diff, max_ev_diff)
    end do
    call timer%stop("loop_eigenvalues")

    call timer%start("loop_eigenvectors")
    do globJ = 1, nev
      max_curr_z_diff = 0.0_rk

      ! eigenvectors are unique up to multiplication by scalar (complex in complex case)
      ! to be able to compare them with analytic, we have to normalize them somehow
      ! we will find a value in computed eigenvector with highest absolut value and enforce
      ! such multiple of computed eigenvector, that the value on corresponding position is the same
      ! as an corresponding value in the analytical eigenvector

      ! find the maximal value in the local part of given eigenvector (with index globJ)
      max_value_for_normalization = 0.0_rk
      max_value_idx = -1
      do globI = 1, na
        if(map_global_array_index_to_local_index(int(globI,kind=c_int), int(globJ,kind=c_int), loc_I, loc_J, &
                 int(nblk,kind=c_int), int(np_rows,kind=c_int), int(np_cols,kind=c_int), &
                 int(my_prow,kind=c_int), int(my_pcol,kind=c_int) )) then
          locI = int(loc_I,kind=INT_TYPE)
          locJ = int(loc_J,kind=INT_TYPE)
          computed_z = z(locI, locJ)
          if(abs(computed_z) > abs(max_value_for_normalization)) then
            max_value_for_normalization = computed_z
            max_value_idx = int(globI,kind=c_int)
          end if
        end if
      end do

      ! find the global maximum and its position. From technical reasons (looking for a 
      ! maximum of complex number), it is not so easy to do it nicely. Therefore we 
      ! communicate local maxima to mpi rank 0 and resolve there. If we wanted to do
      ! it without this, it would be tricky.. question of uniquness - two complex numbers
      ! with the same absolut values, but completely different... 
#ifdef WITH_MPI
      call MPI_Gather(max_value_for_normalization, 1_MPI_KIND, MPI_MATH_DATATYPE_PRECISION, &
                      max_values_array, 1_MPI_KIND, MPI_MATH_DATATYPE_PRECISION, 0_MPI_KIND, &
                      int(MPI_COMM_WORLD,kind=MPI_KIND), mpierr)
      call MPI_Gather(max_value_idx, 1_MPI_KIND, MPI_INT, max_idx_array, 1_MPI_KIND, MPI_INT, &
                      0_MPI_KIND, int(MPI_COMM_WORLD,kind=MPI_KIND), mpierr)
      max_value_for_normalization = 0.0_rk
      max_value_idx = -1
      do rank = 1, np_cols * np_rows 
        if(abs(max_values_array(rank)) > abs(max_value_for_normalization)) then
          max_value_for_normalization = max_values_array(rank)
          max_value_idx = max_idx_array(rank)
        end if
      end do
      call MPI_Bcast(max_value_for_normalization, 1_MPI_KIND, MPI_MATH_DATATYPE_PRECISION, &
                     0_MPI_KIND, int(MPI_COMM_WORLD,kind=MPI_KIND), mpierr)
      call MPI_Bcast(max_value_idx, 1_MPI_KIND, MPI_INT, 0_MPI_KIND, &
                     int(MPI_COMM_WORLD,kind=MPI_KIND), mpierr)
#endif
      ! we decided what the maximum computed value is. Calculate expected value on the same 
      if(abs(max_value_for_normalization) < 0.0001_rk) then 
        if(myid == 0) print *, 'Maximal value in eigenvector too small     :', max_value_for_normalization
        status =1
        return
      end if
      call timer%start("evaluation_helper")
      corresponding_exact_value  = analytic_eigenvectors_&
                                       &MATH_DATATYPE&
                                       &_&
                                       &PRECISION&
                                       &(na, int(max_value_idx,kind=INT_TYPE), globJ)
      call timer%stop("evaluation_helper")
      normalization_quotient = corresponding_exact_value / max_value_for_normalization
      ! write(*,*) "normalization q", normalization_quotient

      ! compare computed and expected eigenvector values, but take into account normalization quotient
      do globI = 1, na
        if(map_global_array_index_to_local_index(int(globI,kind=c_int), int(globJ,kind=c_int), loc_I, loc_J, &
                 int(nblk,kind=c_int), int(np_rows,kind=c_int), int(np_cols,kind=c_int), &
                 int(my_prow,kind=c_int), int(my_pcol,kind=c_int) )) then
           locI = int(loc_I,kind=INT_TYPE)
           locJ = int(loc_J,kind=INT_TYPE)
           computed_z = z(locI, locJ)
           call timer%start("evaluation")
           expected_z = analytic_eigenvectors_&
              &MATH_DATATYPE&
              &_&
              &PRECISION&
              &(na, globI, globJ)
           call timer%stop("evaluation")
           max_curr_z_diff = max(abs(normalization_quotient * computed_z - expected_z), max_curr_z_diff)
        end if
      end do
      ! we have max difference of one of the eigenvectors, update global
      max_z_diff = max(max_z_diff, max_curr_z_diff)
    end do !globJ
    call timer%stop("loop_eigenvectors")

#ifdef WITH_MPI
    call mpi_allreduce(max_z_diff, glob_max_z_diff, 1_MPI_KIND, MPI_REAL_PRECISION, MPI_MAX, &
                       int(MPI_COMM_WORLD,kind=MPI_KIND), mpierr)
#else
    glob_max_z_diff = max_z_diff
#endif
    if(myid == 0) print *, 'Maximum error in eigenvalues      :', max_ev_diff
    if (check_eigenvectors) then
      if(myid == 0) print *, 'Maximum error in eigenvectors     :', glob_max_z_diff
    endif

    status = 0

    if (is_infinity_or_NaN(max_ev_diff)) then
      status = 1
    endif

    if (nev .gt. 2) then
      if (max_ev_diff .gt. tol_eigenvalues .or. max_ev_diff .eq. 0.0_rk) status = 1
      if (check_eigenvectors) then
        if (glob_max_z_diff .gt. tol_eigenvectors .or. glob_max_z_diff .eq. 0.0_rk) status = 1
      endif
    else
      if (max_ev_diff .gt. tol_eigenvalues) status = 1
      if (check_eigenvectors) then
        if (glob_max_z_diff .gt. tol_eigenvectors) status = 1
        if (is_infinity_or_NaN(glob_max_z_diff)) then
          status = 1
        endif
      endif
    endif

    call timer%stop("check_correctness_analytic")
    if(myid == 0 .and. print_timer) then
      call timer%print("check_correctness_analytic")
    end if
    call timer%free()
  end function

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
    !c> TEST_C_INT_TYPE check_correctness_analytic_real_double_f(TEST_C_INT_TYPE na, 
    !c>                                                        TEST_C_INT_TYPE nev,
    !c>                                                        double *ev, double *z,
    !c>                                                        TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                                        TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid,
    !c>                                                        TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                                        TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol,
    !c>                                                        TEST_C_INT_TYPE check_all_evals, 
    !c>                                                        TEST_C_INT_TYPE check_eigenvectors);
#else
    !c> TEST_C_INT_TYPE check_correctness_analytic_real_single_f(TEST_C_INT_TYPE na, 
    !c>                                                        TEST_C_INT_TYPE nev,
    !c>                                                        float *ev, float *z,
    !c>                                                        TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                                        TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid,
    !c>                                                        TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                                        TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol,
    !c>                                                        TEST_C_INT_TYPE check_all_evals, 
    !c>                                                        TEST_C_INT_TYPE check_eigenvectors);
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
    !c> TEST_C_INT_TYPE check_correctness_analytic_complex_double_f(TEST_C_INT_TYPE na, 
    !c>                                                        TEST_C_INT_TYPE nev,
    !c>                                                        double *ev, double_complex *z,
    !c>                                                        TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                                        TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid,
    !c>                                                        TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                                        TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol,
    !c>                                                        TEST_C_INT_TYPE check_all_evals, 
    !c>                                                        TEST_C_INT_TYPE check_eigenvectors);
#else
    !c> TEST_C_INT_TYPE check_correctness_analytic_complex_single_f(TEST_C_INT_TYPE na, 
    !c>                                                        TEST_C_INT_TYPE nev,
    !c>                                                        float *ev, float_complex *z,
    !c>                                                        TEST_C_INT_TYPE na_rows, TEST_C_INT_TYPE na_cols,
    !c>                                                        TEST_C_INT_TYPE nblk, TEST_C_INT_TYPE myid,
    !c>                                                        TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                                        TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol,
    !c>                                                        TEST_C_INT_TYPE check_all_evals, 
    !c>                                                        TEST_C_INT_TYPE check_eigenvectors);
#endif
#endif /* COMPLEXCASE */

  function check_correctness_analytic_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &_f (na, nev, ev, z, na_rows, na_cols, nblk, myid, np_rows, np_cols, my_prow, my_pcol, check_all_evals, &
      check_eigenvectors) result(status) &
      bind(C,name="check_correctness_analytic_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      &_f")
    use iso_c_binding
    use precision_for_tests
    
    implicit none
#include "./test_precision_kinds.F90"
    TEST_INT_TYPE, value                   :: na, nev, na_rows, na_cols, nblk, myid, &
                                                       np_rows, np_cols, my_prow, my_pcol
    TEST_INT_TYPE                          :: status
    MATH_DATATYPE(kind=rck)                :: z(1:na_rows,1:na_cols)
    real(kind=rk), intent(inout)           :: ev(1:na)
    TEST_INT_TYPE, value                   :: check_all_evals  , check_eigenvectors
    logical                                :: check_all_evals_f, check_eigenvectors_f
    
    if (check_all_evals == 0) then
        check_all_evals_f = .false.
    else 
        check_all_evals_f = .true.
    end if
    
    if (check_eigenvectors == 0) then
        check_eigenvectors_f = .false.
    else 
        check_eigenvectors_f = .true.
    end if
    
    status = check_correctness_analytic_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    & (na, nev, ev, z, nblk, myid, np_rows, np_cols, my_prow, my_pcol, check_all_evals_f, check_eigenvectors_f)

  end function
    
    !-----------------------------------------------------------------------------------------------------------
    
  function analytic_matrix_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &(na, i, j) result(element)
    use precision_for_tests

    implicit none
    TEST_INT_TYPE, intent(in) :: na, i, j
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
    use precision_for_tests

    implicit none
    TEST_INT_TYPE, intent(in) :: na, i, j
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
    use precision_for_tests

    implicit none
    TEST_INT_TYPE, intent(in) :: na, i
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
    use precision_for_tests

    implicit none
#include "./test_precision_kinds.F90"
    TEST_INT_TYPE, intent(in)     :: na, i, j, what
    MATH_DATATYPE(kind=rck)       :: element, mat2x2(2,2), mat(5,5)
    real(kind=rk)                 :: a, am, amp
    TEST_INT_TYPE                 :: levels(num_primes)
    TEST_INT_TYPE                 :: ii, jj, m, prime_id, prime, total_level, level

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
#ifdef BUILD_FUGAKU
    primes(1) = 2
    primes(2) = 3
    primes(3) = 5
#endif
    do prime_id = 1,num_primes
      prime = primes(prime_id)
      do  level = 1, levels(prime_id)
        amp = am**(prime-1)
        total_level = total_level + 1
        if(what == ANALYTIC_MATRIX) then
#ifdef REALCASE
#ifndef FUGAKU
          mat2x2 = reshape((/ c*c + amp * s*s, (amp - 1.0_rk) * s*c,  &
                           (amp - 1.0_rk) * s*c, s*s + amp * c*c  /), &
                                      (/2, 2/), order=(/2,1/))
#endif
#endif
#ifdef COMPLEXCASE
#ifndef FUGAKU
          mat2x2 = reshape((/ 0.5_rck * (amp + 1.0_rck) * (1.0_rk, 0.0_rk),   sq2/4.0_rk * (amp - 1.0_rk) * (1.0_rk, 1.0_rk),   &
                              sq2/4.0_rk * (amp - 1.0_rk) * (1.0_rk, -1.0_rk),  0.5_rck * (amp + 1.0_rck) * (1.0_rk, 0.0_rk) /), &
                                      (/2, 2/), order=(/2,1/))
! intel 2018 does not reshape correctly (one would have to specify order=(/1,2/)
! until this is resolved, I resorted to the following
          mat2x2(1,2) = sq2/4.0_rk * (amp - 1.0_rk) * (1.0_rk, 1.0_rk)
          mat2x2(2,1) = sq2/4.0_rk * (amp - 1.0_rk) * (1.0_rk, -1.0_rk)
#endif
#endif
        else if(what == ANALYTIC_EIGENVECTORS) then
#ifdef REALCASE
#ifndef FUGAKU
          mat2x2 = reshape((/ c, s,  &
                           -s,  c  /), &
                                (/2, 2/), order=(/2,1/))
! intel 2018 does not reshape correctly (one would have to specify order=(/1,2/)
! until this is resolved, I resorted to the following
          mat2x2(1,2) = s
          mat2x2(2,1) = -s
#endif
#endif
#ifdef COMPLEXCASE
#ifndef FUGAKU
          mat2x2 = reshape((/ -sq2/2.0_rck * (1.0_rk, 0.0_rk),       -sq2/2.0_rck * (1.0_rk, 0.0_rk),  &
                              0.5_rk * (1.0_rk, -1.0_rk),  0.5_rk * (-1.0_rk, 1.0_rk)  /), &
                                (/2, 2/), order=(/2,1/))
! intel 2018 does not reshape correctly (one would have to specify order=(/1,2/)
! until this is resolved, I resorted to the following
          mat2x2(1,2) = -sq2/2.0_rck * (1.0_rk, 0.0_rk)
          mat2x2(2,1) = 0.5_rk * (1.0_rk, -1.0_rk)
#endif
#endif
        else if(what == ANALYTIC_EIGENVALUES) then
#ifndef FUGAKU
          mat2x2 = reshape((/ 1.0_rck, 0.0_rck,  &
                           0.0_rck, amp  /), &
                                 (/2, 2/), order=(/2,1/))
#endif
        else
          assert(.false.)
        end if

        mat = 0.0_rck
        if(prime == 2) then
#ifndef BUILD_FUGAKU
          mat(1:2, 1:2) = mat2x2
#endif
        else if(prime == 3) then
#ifndef BUILD_FUGAKU
          mat((/1,3/),(/1,3/)) = mat2x2
#endif
          if(what == ANALYTIC_EIGENVECTORS) then
            mat(2,2) = 1.0_rck
          else
            mat(2,2) = am
          end if
        else if(prime == 5) then
#ifndef BUILD_FUGAKU
          mat((/1,5/),(/1,5/)) = mat2x2
#endif
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
    use precision_for_tests

    implicit none
#include "./test_precision_kinds.F90"
    TEST_INT_TYPE, intent(in)    :: myid, na
    character(len=*), intent(in) :: mat_name
    MATH_DATATYPE(kind=rck)      :: mat(na, na)
    TEST_INT_TYPE                :: i,j
    character(len=20)            :: na_str

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

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
  !c> void print_matrix_real_double_f (TEST_C_INT_TYPE myid, TEST_C_INT_TYPE na, double *a, const char *mat_name_c);
#else
  !c> void print_matrix_real_single_f (TEST_C_INT_TYPE myid, TEST_C_INT_TYPE na, float  *a, const char *mat_name_c);  
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
  !c> void print_matrix_complex_double_f (TEST_C_INT_TYPE myid, TEST_C_INT_TYPE na, double_complex *a, const char *mat_name_c);
#else
  !c> void print_matrix_complex_single_f (TEST_C_INT_TYPE myid, TEST_C_INT_TYPE na, float_complex  *a, const char *mat_name_c);
#endif
#endif /* COMPLEXCASE */

    
  subroutine print_matrix_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &_f (myid, na, mat, mat_name_c) &
    bind(C,name="print_matrix_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      &_f")
   
    use iso_c_binding
    use precision_for_tests
    use elpa_api 
    
    implicit none
#include "./test_precision_kinds.F90"
    TEST_INT_TYPE, value    :: myid, na
    !character(len=*)  :: mat_name
    !CHARACTER(kind=C_CHAR), value :: mat_name
    MATH_DATATYPE(kind=rck) :: mat(na, na)
    type(c_ptr), intent(in), value                :: mat_name_c
    character(len=elpa_strlen_c(mat_name_c)), pointer :: mat_name_f
      
    call c_f_pointer(mat_name_c, mat_name_f)
      
    call print_matrix_&
      &MATH_DATATYPE&
      &_&
      &PRECISION&
      & (myid, na, mat, mat_name_f)

  end subroutine

  subroutine check_matrices_&
    &MATH_DATATYPE&
    &_&
    &PRECISION&
    &(myid, na)
    use precision_for_tests

    implicit none
#include "./test_precision_kinds.F90"
    TEST_INT_TYPE, intent(in) :: myid, na
    MATH_DATATYPE(kind=rck)   :: A(na, na), S(na, na), L(na, na), res(na, na)
    TEST_INT_TYPE             :: i, j, decomposition(num_primes)

    real(kind=rk)             :: err
#ifdef DOUBLE_PRECISION
    real(kind=rk), parameter  :: TOL =  1e-8
#endif
#ifdef SINGLE_PRECISION
    real(kind=rk), parameter  :: TOL =  1e-4
#endif

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
    err = maxval(abs(res))
    
    if(err > TOL) then
      print *, "WARNING: sanity test in module analytic failed, error is ", err
    end if

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
    use precision_for_tests

    implicit none
    TEST_INT_TYPE, intent(in)   :: myid
    TEST_INT_TYPE               :: decomposition(num_primes), i
#ifndef BUILD_FUGAKU
    TEST_INT_TYPE, parameter    :: check_sizes(7) = (/2, 3, 5, 6, 10, 25, 150/)
#else
    TEST_INT_TYPE    :: check_sizes(7)
#endif
    if(myid == 0) print *, "Checking test_analytic module sanity.... "
#ifndef BUILD_FUGAKU
#ifdef HAVE_64BIT_INTEGER_MATH_SUPPORT
    assert(decompose(1500_lik, decomposition))
#else
    assert(decompose(1500_ik, decomposition))
#endif
    assert(all(decomposition == (/2,1,3/)))
#ifdef HAVE_64BIT_INTEGER_MATH_SUPPORT
    assert(decompose(6_lik,decomposition))
#else
    assert(decompose(6_ik,decomposition))
#endif
    assert(all(decomposition == (/1,1,0/)))

#ifdef BUILD_FUGAKU
    check_sizes(1) = 2
    check_sizes(2) = 3
    check_sizes(3) = 5
    check_sizes(4) = 10
    check_sizes(5) = 25
    check_sizes(6) = 150
#endif
    do i =1, size(check_sizes)
      call check_matrices_&
          &MATH_DATATYPE&
          &_&
          &PRECISION&
          &(myid, check_sizes(i))
    end do

    if(myid == 0) print *, "Checking test_analytic module sanity.... DONE"
#endif
  end subroutine
