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
!
#include "config-f90.h"

! Define one of TEST_REAL or TEST_COMPLEX
! Define one of TEST_SINGLE or TEST_DOUBLE
! Define one of TEST_SOLVER_1STAGE or TEST_SOLVER_2STAGE
! Define TEST_NVIDIA_GPU \in [0, 1]
! Define TEST_INTEL_GPU \in [0, 1]
! Define TEST_INTEL_GPU_OPENMP \in [0, 1]
! Define TEST_INTEL_GPU_SYCL \in [0, 1]
! Define TEST_AMD_GPU \in [0, 1]
! Define either TEST_ALL_KERNELS or a TEST_KERNEL \in [any valid kernel]

#if !(defined(TEST_REAL) ^ defined(TEST_COMPLEX))
error: define exactly one of TEST_REAL or TEST_COMPLEX
#endif

#if !(defined(TEST_SINGLE) ^ defined(TEST_DOUBLE))
error: define exactly one of TEST_SINGLE or TEST_DOUBLE
#endif

#if !(defined(TEST_SOLVER_1STAGE) ^ defined(TEST_SOLVER_2STAGE) ^ defined(TEST_SCALAPACK_ALL) ^ defined(TEST_SCALAPACK_PART))
error: define exactly one of TEST_SOLVER_1STAGE or TEST_SOLVER_2STAGE or TEST_SCALAPACK_ALL or TEST_SCALAPACK_PART
#endif

#ifdef TEST_SOLVER_1STAGE
#ifdef TEST_ALL_KERNELS
error: TEST_ALL_KERNELS cannot be defined for TEST_SOLVER_1STAGE
#endif
#ifdef TEST_KERNEL
error: TEST_KERNEL cannot be defined for TEST_SOLVER_1STAGE
#endif
#endif

#ifdef TEST_SOLVER_2STAGE
#if !(defined(TEST_KERNEL) ^ defined(TEST_ALL_KERNELS))
error: define either TEST_ALL_KERNELS or a valid TEST_KERNEL
#endif
#endif

#ifdef TEST_GENERALIZED_DECOMP_EIGENPROBLEM
#define TEST_GENERALIZED_EIGENPROBLEM
#endif

#ifdef TEST_SINGLE
#  define EV_TYPE real(kind=C_FLOAT)
#  ifdef TEST_REAL
#    define MATRIX_TYPE real(kind=C_FLOAT)
#  else
#    define MATRIX_TYPE complex(kind=C_FLOAT_COMPLEX)
#  endif
#else
#  define EV_TYPE real(kind=C_DOUBLE)
#  ifdef TEST_REAL
#    define MATRIX_TYPE real(kind=C_DOUBLE)
#  else
#    define MATRIX_TYPE complex(kind=C_DOUBLE_COMPLEX)
#  endif
#endif

#ifdef TEST_REAL
#define KERNEL_KEY "real_kernel"
#endif
#ifdef TEST_COMPLEX
#define KERNEL_KEY "complex_kernel"
#endif

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

#define TEST_GPU 0
#if (TEST_NVIDIA_GPU == 1) || (TEST_AMD_GPU == 1) || (TEST_INTEL_GPU == 1) || (TEST_INTEL_GPU_OPENMP == 1) || (TEST_INTEL_GPU_SYCL == 1)
#undef TEST_GPU
#define TEST_GPU 1
#endif

#include "assert.h"

program test
   use elpa
   !use test_util
   use test_setup_mpi
   use test_prepare_matrix
   use test_read_input_parameters
   use test_blacs_infrastructure
   use test_check_correctness
   use test_analytic
#ifdef WITH_SCALAPACK_TESTS
   use test_scalapack
#endif

#ifdef HAVE_REDIRECT
   use test_redirect
#endif
#ifdef WITH_OPENMP_TRADITIONAL
   use omp_lib
#endif
   use precision_for_tests

#if TEST_GPU_DEVICE_POINTER_API == 1 || TEST_GPU_SET_ID == 1
   use test_gpu
   use mod_check_for_gpu
#if TEST_NVIDIA_GPU == 1
   use test_cuda_functions
#endif
#if TEST_AMD_GPU == 1
   use test_hip_functions
#endif
#if TEST_INTEL_GPU_SYCL == 1
   use test_sycl_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
#error "openmp_offload missing"
#endif


#endif /* TEST_GPU_DEVICE_POINTER_API */

   implicit none

   ! matrix dimensions
   TEST_INT_TYPE                       :: na, nev, nblk

   ! mpi
   TEST_INT_TYPE                       :: myid, nprocs
   TEST_INT_MPI_TYPE                   :: myidMPI, nprocsMPI
   TEST_INT_TYPE                       :: na_cols, na_rows  ! local matrix size
   TEST_INT_TYPE                       :: np_cols, np_rows  ! number of MPI processes per column/row
   TEST_INT_TYPE                       :: my_prow, my_pcol  ! local MPI task position (my_prow, my_pcol) in the grid (0..np_cols -1, 0..np_rows -1)
   TEST_INT_MPI_TYPE                   :: mpierr, blacs_ok_mpi

   ! blacs
   TEST_INT_TYPE                       :: my_blacs_ctxt, sc_desc(9), info, blacs_ok

   ! The Matrix
   MATRIX_TYPE, allocatable, target    :: a(:,:)
   MATRIX_TYPE, allocatable            :: as(:,:)
#if defined(TEST_HERMITIAN_MULTIPLY_FULL) || defined(TEST_HERMITIAN_MULTIPLY_UPPER) || defined(TEST_HERMITIAN_MULTIPLY_LOWER)
   MATRIX_TYPE, allocatable, target    :: b(:,:), c(:,:)
   logical                             :: isUpper, isLower
#endif
#if defined(TEST_GENERALIZED_EIGENPROBLEM)
   MATRIX_TYPE, allocatable            :: b(:,:), bs(:,:)
#endif
   ! eigenvectors
   MATRIX_TYPE, allocatable, target    :: z(:,:)
   ! eigenvalues
   EV_TYPE, allocatable, target        :: ev(:)

#if TEST_GPU_DEVICE_POINTER_API == 1
   type(c_ptr)                         :: a_dev, q_dev, ev_dev, b_dev, c_dev ! q_dev -- eigenvectors (analogous to z)
#endif

   logical                             :: check_all_evals, skip_check_correctness

#if defined(TEST_MATRIX_TOEPLITZ) || defined(TEST_MATRIX_FRANK)
   EV_TYPE, allocatable                :: d(:), sd(:), ds(:), sds(:)
   EV_TYPE                             :: diagonalElement, subdiagonalElement
#endif

   TEST_INT_TYPE                       :: status
   integer(kind=c_int)                 :: error_elpa

   type(output_t)                      :: write_to_file
   class(elpa_t), pointer              :: e

#ifdef TEST_ALL_KERNELS
   TEST_INT_TYPE                       :: i
#endif
#ifdef TEST_ALL_LAYOUTS
   TEST_INT_TYPE                       :: i_layout
#ifdef BUILD_FUGAKU
   character(len=1)                    :: layouts(2)
#else
   character(len=1), parameter         :: layouts(2) = [ 'C', 'R' ]
#endif
#endif
   integer(kind=c_int)                 :: kernel
   character(len=1)                    :: layout
   logical                             :: do_test_numeric_residual, do_test_numeric_residual_generalized, &
                                          do_test_analytic_eigenvalues, &
                                          do_test_analytic_eigenvalues_eigenvectors,   &
                                          do_test_frank_eigenvalues,  &
                                          do_test_toeplitz_eigenvalues, do_test_cholesky,   &
                                          do_test_hermitian_multiply
   logical                             :: ignoreError, successGPU
#if TEST_GPU == 1
   TEST_INT_TYPE                       :: numberOfDevices
#endif
#ifdef WITH_OPENMP_TRADITIONAL
   TEST_INT_TYPE                       :: max_threads, threads_caller
#endif
#if TEST_GPU_SET_ID == 1
   TEST_INT_TYPE                       :: gpuID
#endif
#ifdef SPLIT_COMM_MYSELF
   TEST_INT_MPI_TYPE                   :: mpi_comm_rows, mpi_comm_cols, mpi_string_length, mpierr2
   character(len=MPI_MAX_ERROR_STRING) :: mpierr_string
#endif


#if TEST_GPU_DEVICE_POINTER_API == 1
#if TEST_REAL == 1
#if TEST_DOUBLE
   integer(kind=c_intptr_t), parameter :: size_of_datatype      = size_of_double_real
   integer(kind=c_intptr_t), parameter :: size_of_real_datatype = size_of_double_real
#endif
#if TEST_SINGLE
   integer(kind=c_intptr_t), parameter :: size_of_datatype      = size_of_single_real
   integer(kind=c_intptr_t), parameter :: size_of_real_datatype = size_of_single_real
#endif
#endif /* TEST_REAL == 1 */

#if TEST_COMPLEX == 1
#if TEST_DOUBLE
   integer(kind=c_intptr_t), parameter :: size_of_datatype      = size_of_double_complex
   integer(kind=c_intptr_t), parameter :: size_of_real_datatype = size_of_double_real
#endif
#if TEST_SINGLE
   integer(kind=c_intptr_t), parameter :: size_of_datatype      = size_of_single_complex
   integer(kind=c_intptr_t), parameter :: size_of_real_datatype = size_of_single_real
#endif
#endif
#endif /* TEST_GPU_DEVICE_POINTER_API == 1 */



#ifdef TEST_ALL_LAYOUTS
#ifdef BUILD_FUGAKU
   layouts(1) = 'C'
   layouts(2) = 'R'
#endif
#endif

   ignoreError = .false.

   call read_input_parameters_traditional(na, nev, nblk, write_to_file, skip_check_correctness)
   call setup_mpi(myid, nprocs)

#ifdef HAVE_REDIRECT
#ifdef WITH_MPI
     call MPI_BARRIER(MPI_COMM_WORLD, mpierr)
     call redirect_stdout(myid)
#endif
#endif

   check_all_evals = .true.


   do_test_numeric_residual = .false.
   do_test_numeric_residual_generalized = .false.
   do_test_analytic_eigenvalues = .false.
   do_test_analytic_eigenvalues_eigenvectors = .false.
   do_test_frank_eigenvalues = .false.
   do_test_toeplitz_eigenvalues = .false.

   do_test_cholesky = .false.
#if defined(TEST_CHOLESKY)
   do_test_cholesky = .true.
#endif
   do_test_hermitian_multiply = .false.
#if defined(TEST_HERMITIAN_MULTIPLY_FULL) || defined(TEST_HERMITIAN_MULTIPLY_UPPER) || defined(TEST_HERMITIAN_MULTIPLY_LOWER)
   do_test_hermitian_multiply = .true.
#endif

   status = 0
   if (elpa_init(CURRENT_API_VERSION) /= ELPA_OK) then
     print *, "ELPA API version not supported"
     stop 1
   endif

   if (myid == 0) then
     print '((a,i0))', 'Program ' &
     // TEST_CASE
     print *, ""
   endif

#ifdef TEST_ALL_LAYOUTS
   do i_layout = 1, size(layouts)               ! layouts
     layout = layouts(i_layout)
     do np_cols = 1, nprocs                     ! factors
       if (mod(nprocs,np_cols) /= 0 ) then
         cycle
       endif

#else
   layout = 'C'
   do np_cols = NINT(SQRT(REAL(nprocs))),2,-1
      if(mod(nprocs,np_cols) == 0 ) exit
   enddo
#endif

   np_rows = nprocs/np_cols
   assert(nprocs == np_rows * np_cols)

   if (myid == 0) then
     print '((a,i0))', 'Matrix size: ', na
     print '((a,i0))', 'Num eigenvectors: ', nev
     print '((a,i0))', 'Blocksize: ', nblk
#ifdef WITH_MPI
     print '((a,i0))', 'Num MPI proc: ', nprocs
     print '(3(a,i0))','Number of processor rows=',np_rows,', cols=',np_cols,', total=',nprocs
     print '(a)',      'Process layout: ' // layout
#endif
     print *,''
   endif

#ifdef TEST_ALL_KERNELS
   !  discard for the moment until allocates and memcpy are in the correct loop
#if TEST_GPU_DEVICE_POINTER_API == 1
#ifdef WITH_MPI
   call mpi_finalize(mpierr)
#endif
   stop 77
#endif
#endif



#if TEST_QR_DECOMPOSITION == 1

#if (TEST_NVIDIA_GPU == 1) || (TEST_INTEL_GPU == 1) || (TEST_AMD_GPU == 1) || (TEST_INTEL_GPU_OPENMP == 1) || (TEST_INTEL_GPU_SYCL == 1)
#ifdef WITH_MPI
     call mpi_finalize(mpierr)
#endif
     stop 77
#endif /* TEST_NVIDIA_GPU || TEST_INTEL_GPU */
   if (nblk .lt. 64) then
     if (myid .eq. 0) then
       print *,"At the moment QR decomposition need blocksize of at least 64"
     endif
     if ((na .lt. 64) .and. (myid .eq. 0)) then
       print *,"This is why the matrix size must also be at least 64 or only 1 MPI task can be used"
     endif

#ifdef WITH_MPI
     call mpi_finalize(mpierr)
#endif
     stop 77
   endif
#endif /* TEST_QR_DECOMPOSITION */


#ifdef WITH_CUDA_AWARE_MPI
#if TEST_NVIDIA_GPU != 1
#ifdef WITH_MPI
     call mpi_finalize(mpierr)
#endif
     stop 77
#endif

#ifdef TEST_ALL_KERNELS
#ifdef WITH_MPI
     call mpi_finalize(mpierr)
#endif
     stop 77
#endif

#endif /* WITH_CUDA_AWARE_MPI */

   call set_up_blacsgrid(int(mpi_comm_world,kind=BLAS_KIND), np_rows, &
                         np_cols, layout, my_blacs_ctxt, my_prow, &
                         my_pcol)


#if defined(TEST_GENERALIZED_EIGENPROBLEM) && defined(TEST_ALL_LAYOUTS)
! #ifdef WITH_MPI
!      call mpi_finalize(mpierr) ! PETERDEBUG: comment out for testing
! #endif
!      stop 77
#endif

   call set_up_blacs_descriptor(na, nblk, my_prow, my_pcol, &
                                np_rows, np_cols, &
                                na_rows, na_cols, sc_desc, my_blacs_ctxt, info, blacs_ok)
#ifdef WITH_MPI
   blacs_ok_mpi = int(blacs_ok, kind=INT_MPI_TYPE)
   call mpi_allreduce(MPI_IN_PLACE, blacs_ok_mpi, 1_MPI_KIND, MPI_INTEGER, MPI_MIN, int(MPI_COMM_WORLD,kind=MPI_KIND), mpierr)
   blacs_ok = int(blacs_ok_mpi, kind=INT_TYPE)
#endif

   if (blacs_ok .eq. 0) then
#ifdef TEST_ALL_LAYOUTS
     if (myid .eq. 0) then
       print *," Current layout not supported by the blacsgrid. Skipping..."
     endif
     continue
#else
     if (myid .eq. 0) then
       print *," Ecountered critical error when setting up blacs. Aborting..."
     endif
#ifdef WITH_MPI
     call mpi_finalize(mpierr)
#endif
     stop 1
#endif
   endif

   !Allocate the matrices needed for elpa

   allocate(a (na_rows,na_cols))
   allocate(as(na_rows,na_cols))
   allocate(z (na_rows,na_cols))
   allocate(ev(na))

#if defined(TEST_HERMITIAN_MULTIPLY_FULL) || defined(TEST_HERMITIAN_MULTIPLY_UPPER) || defined(TEST_HERMITIAN_MULTIPLY_LOWER)
   allocate(b (na_rows,na_cols))
   allocate(c (na_rows,na_cols))
#endif

#ifdef TEST_GENERALIZED_EIGENPROBLEM
   allocate(b (na_rows,na_cols))
   allocate(bs (na_rows,na_cols))
#endif

#if defined(TEST_MATRIX_TOEPLITZ) || defined(TEST_MATRIX_FRANK)
   allocate(d (na), ds(na))
   allocate(sd (na), sds(na))
#endif

   a(:,:) = 0.0
   z(:,:) = 0.0
   ev(:) = 0.0

#if defined(TEST_MATRIX_RANDOM) && !defined(TEST_SOLVE_TRIDIAGONAL) && !defined(TEST_CHOLESKY) && !defined(TEST_EIGENVALUES)
   ! the random matrix can be used in allmost all tests; but for some no
   ! correctness checks have been implemented; do not allow these
   ! combinations
   ! RANDOM + TEST_SOLVE_TRIDIAGONAL: we need a TOEPLITZ MATRIX
   ! RANDOM + TEST_CHOLESKY: wee need SPD matrix
   ! RANDOM + TEST_EIGENVALUES: no correctness check known

   ! We also have to take care of special case in TEST_EIGENVECTORS
#if !defined(TEST_EIGENVECTORS)
    call prepare_matrix_random(na, myid, sc_desc, a, z, as)
#else /* TEST_EIGENVECTORS */
    if (nev .ge. 1) then
      call prepare_matrix_random(na, myid, sc_desc, a, z, as)
#if !defined(TEST_HERMITIAN_MULTIPLY_FULL) && !defined(TEST_HERMITIAN_MULTIPLY_UPPER) && !defined(TEST_HERMITIAN_MULTIPLY_LOWER)
      do_test_numeric_residual = .true.
#endif
   else
     if (myid .eq. 0) then
       print *,"At the moment with the random matrix you need nev >=1"
     endif
#ifdef WITH_MPI
     call mpi_finalize(mpierr)
#endif
     stop 77
   endif
#endif /* TEST_EIGENVECTORS */
    do_test_analytic_eigenvalues = .false.
    do_test_analytic_eigenvalues_eigenvectors = .false.
    do_test_frank_eigenvalues = .false.
    do_test_toeplitz_eigenvalues = .false.
#endif /* (TEST_MATRIX_RANDOM) */

#if defined(TEST_MATRIX_RANDOM) && defined(TEST_CHOLESKY)
     call prepare_matrix_random_spd(na, myid, sc_desc, a, z, as, &
                 nblk, np_rows, np_cols, my_prow, my_pcol)
    do_test_analytic_eigenvalues = .false.
    do_test_analytic_eigenvalues_eigenvectors = .false.
    do_test_frank_eigenvalues = .false.
    do_test_toeplitz_eigenvalues = .false.
#endif /* TEST_MATRIX_RANDOM and TEST_CHOLESKY */

#if defined(TEST_MATRIX_RANDOM) && defined(TEST_GENERALIZED_EIGENPROBLEM)
   ! call prepare_matrix_random(na, myid, sc_desc, a, z, as)
    call prepare_matrix_random_spd(na, myid, sc_desc, b, z, bs, &
                 nblk, np_rows, np_cols, my_prow, my_pcol)
    do_test_analytic_eigenvalues = .false.
    do_test_analytic_eigenvalues_eigenvectors = .false.
    do_test_frank_eigenvalues = .false.
    do_test_toeplitz_eigenvalues = .false.
    do_test_numeric_residual = .false.
    do_test_numeric_residual_generalized = .true.
#endif /* TEST_MATRIX_RANDOM and TEST_GENERALIZED_EIGENPROBLEM */

#if defined(TEST_MATRIX_RANDOM) && (defined(TEST_SOLVE_TRIDIAGONAL) || defined(TEST_EIGENVALUES))
#error "Random matrix is not allowed in this configuration"
#endif

#if defined(TEST_MATRIX_ANALYTIC)  && !defined(TEST_SOLVE_TRIDIAGONAL) && !defined(TEST_CHOLESKY)
   ! the analytic matrix can be used in allmost all tests; but for some no
   ! correctness checks have been implemented; do not allow these
   ! combinations
   ! ANALYTIC + TEST_SOLVE_TRIDIAGONAL: we need a TOEPLITZ MATRIX
   ! ANALYTIC + TEST_CHOLESKY: no correctness check yet implemented

   call prepare_matrix_analytic(na, a, nblk, myid, np_rows, np_cols, my_prow, my_pcol)
   as(:,:) = a

   do_test_numeric_residual = .false.
   do_test_analytic_eigenvalues_eigenvectors = .false.
#if !defined(TEST_HERMITIAN_MULTIPLY_FULL) && !defined(TEST_HERMITIAN_MULTIPLY_UPPER) && !defined(TEST_HERMITIAN_MULTIPLY_LOWER)
   do_test_analytic_eigenvalues = .true.
#endif
#if defined(TEST_EIGENVECTORS)
   if (nev .ge. 1) then
     do_test_analytic_eigenvalues_eigenvectors = .true.
     do_test_analytic_eigenvalues = .false.
   else
     do_test_analytic_eigenvalues_eigenvectors = .false.
   endif
#endif
   do_test_frank_eigenvalues = .false.
   do_test_toeplitz_eigenvalues = .false.
#endif /* TEST_MATRIX_ANALYTIC */
#if defined(TEST_MATRIX_ANALYTIC) && (defined(TEST_SOLVE_TRIDIAGONAL) || defined(TEST_CHOLESKY))
#error "Analytic matrix is not allowd in this configuration"
#endif

#if defined(TEST_MATRIX_TOEPLITZ)
   ! The Toeplitz matrix works in each test
#ifdef TEST_SINGLE
   diagonalElement = 0.45_c_float
   subdiagonalElement =  0.78_c_float
#else
   diagonalElement = 0.45_c_double
   subdiagonalElement =  0.78_c_double
#endif

! actually we test cholesky for diagonal matrix only
#if defined(TEST_CHOLESKY)
#ifdef TEST_SINGLE
  diagonalElement = (2.546_c_float, 0.0_c_float)
  subdiagonalElement =  (0.0_c_float, 0.0_c_float)
#else
  diagonalElement = (2.546_c_double, 0.0_c_double)
  subdiagonalElement =  (0.0_c_double, 0.0_c_double)
#endif
#endif /* TEST_CHOLESKY */

   ! check first whether to abort
   if (na < 10) then
#ifdef WITH_MPI
     call mpi_finalize(mpierr)
#endif
     stop 77
   endif
   call prepare_matrix_toeplitz(na, diagonalElement, subdiagonalElement, &
                                d, sd, ds, sds, a, as, nblk, np_rows, &
                                np_cols, my_prow, my_pcol)


   do_test_numeric_residual = .false.
#if defined(TEST_EIGENVECTORS)
   if (nev .ge. 1) then
     do_test_numeric_residual = .true.
   else
     do_test_numeric_residual = .false.
   endif
#endif

   do_test_analytic_eigenvalues = .false.
   do_test_analytic_eigenvalues_eigenvectors = .false.
   do_test_frank_eigenvalues = .false.
#if defined(TEST_CHOLESKY)
   do_test_toeplitz_eigenvalues = .false.
#else
   do_test_toeplitz_eigenvalues = .true.
#endif

#endif /* TEST_MATRIX_TOEPLITZ */


#if defined(TEST_MATRIX_FRANK) && !defined(TEST_SOLVE_TRIDIAGONAL) && !defined(TEST_CHOLESKY)
   ! the random matrix can be used in allmost all tests; but for some no
   ! correctness checks have been implemented; do not allow these
   ! combinations
   ! FRANK + TEST_SOLVE_TRIDIAGONAL: we need a TOEPLITZ MATRIX
   ! FRANK + TEST_CHOLESKY: no correctness check yet implemented

   ! We also have to take care of special case in TEST_EIGENVECTORS
#if !defined(TEST_EIGENVECTORS)
    call prepare_matrix_frank(na, a, z, as, nblk, np_rows, np_cols, my_prow, my_pcol)

    do_test_analytic_eigenvalues = .false.
    do_test_analytic_eigenvalues_eigenvectors = .false.
#if !defined(TEST_HERMITIAN_MULTIPLY_FULL) && !defined(TEST_HERMITIAN_MULTIPLY_UPPER) && !defined(TEST_HERMITIAN_MULTIPLY_LOWER)
    do_test_frank_eigenvalues = .true.
#endif
    do_test_toeplitz_eigenvalues = .false.

#else /* TEST_EIGENVECTORS */

    if (nev .ge. 1) then
      call prepare_matrix_frank(na, a, z, as, nblk, np_rows, np_cols, my_prow, my_pcol)

    do_test_analytic_eigenvalues = .false.
    do_test_analytic_eigenvalues_eigenvectors = .false.
#if !defined(TEST_HERMITIAN_MULTIPLY_FULL) && !defined(TEST_HERMITIAN_MULTIPLY_UPPER) && !defined(TEST_HERMITIAN_MULTIPLY_LOWER)
    do_test_frank_eigenvalues = .true.
#endif
    do_test_toeplitz_eigenvalues = .false.
    do_test_numeric_residual = .false.
   else
    do_test_analytic_eigenvalues = .false.
    do_test_analytic_eigenvalues_eigenvectors = .false.
#if !defined(TEST_HERMITIAN_MULTIPLY_FULL) && !defined(TEST_HERMITIAN_MULTIPLY_UPPER) && !defined(TEST_HERMITIAN_MULTIPLY_LOWER)
    do_test_frank_eigenvalues = .true.
#endif
    do_test_toeplitz_eigenvalues = .false.
    do_test_numeric_residual = .false.

   endif

#endif /* TEST_EIGENVECTORS */
#endif /* (TEST_MATRIX_FRANK) */
#if defined(TEST_MATRIX_FRANK) && (defined(TEST_SOLVE_TRIDIAGONAL) || defined(TEST_CHOLESKY))
#error "FRANK matrix is not allowed in this configuration"
#endif


#if defined(TEST_HERMITIAN_MULTIPLY_FULL) || defined(TEST_HERMITIAN_MULTIPLY_UPPER) || defined(TEST_HERMITIAN_MULTIPLY_LOWER)
#ifdef TEST_REAL

#ifdef TEST_DOUBLE
  
   b(:,:) = 2.0_c_double * a(:,:)
   c(:,:) = 0.0_c_double
#else
   b(:,:) = 2.0_c_float * a(:,:)
   c(:,:) = 0.0_c_float
#endif

#endif /* TEST_REAL */

#ifdef TEST_COMPLEX

#ifdef TEST_DOUBLE
   b(:,:) = 2.0_c_double * a(:,:)
   c(:,:) = (1.0_c_double, 1.0_c_double)
#else
   b(:,:) = 2.0_c_float * a(:,:)
   c(:,:) = (1.0_c_float, 1.0_c_float)
#endif

#endif /* TEST_COMPLEX */

#endif /* TEST_HERMITIAN_MULTIPLY_FULL || defined(TEST_HERMITIAN_MULTIPLY_UPPER) || defined(TEST_HERMITIAN_MULTIPLY_UPPER) */

! if the test is used for (repeated) performacne tests, one might want to skip the checking
! of the results, which might be time-consuming and not necessary.
   if(skip_check_correctness) then
     do_test_numeric_residual = .false.
     do_test_numeric_residual_generalized = .false.
     do_test_analytic_eigenvalues = .false.
     do_test_analytic_eigenvalues_eigenvectors = .false.
     do_test_frank_eigenvalues = .false.
     do_test_toeplitz_eigenvalues = .false.
     do_test_cholesky = .false.
   endif


#ifdef WITH_OPENMP_TRADITIONAL
   threads_caller = omp_get_max_threads()
   if (myid == 0) then
     print *,"The calling program uses ",threads_caller," threads"
   endif
#endif

   e => elpa_allocate(error_elpa)
   assert_elpa_ok(error_elpa)

	! Set parameters

   call e%set("na", int(na,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("nev", int(nev,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("local_nrows", int(na_rows,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("local_ncols", int(na_cols,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("nblk", int(nblk,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)

   if (layout .eq. 'C') then
     call e%set("matrix_order",COLUMN_MAJOR_ORDER,error_elpa)
     assert_elpa_ok(error_elpa)
   else
     call e%set("matrix_order",ROW_MAJOR_ORDER,error_elpa)
     assert_elpa_ok(error_elpa)
   endif

#ifdef WITH_MPI
#ifdef SPLIT_COMM_MYSELF
   call mpi_comm_split(MPI_COMM_WORLD, int(my_pcol,kind=MPI_KIND), int(my_prow,kind=MPI_KIND), &
                       mpi_comm_rows, mpierr)
   if (mpierr .ne. MPI_SUCCESS) then
     call MPI_ERROR_STRING(mpierr, mpierr_string, mpi_string_length, mpierr2)
     write(error_unit,*) "MPI ERROR occured during mpi_comm_split for row communicator: ", trim(mpierr_string)
     stop 1
   endif

   call mpi_comm_split(MPI_COMM_WORLD, int(my_prow,kind=MPI_KIND), int(my_pcol,kind=MPI_KIND), &
                       mpi_comm_cols, mpierr)
   if (mpierr .ne. MPI_SUCCESS) then
     call MPI_ERROR_STRING(mpierr,mpierr_string, mpi_string_length, mpierr2)
     write(error_unit,*) "MPI ERROR occured during mpi_comm_split for col communicator: ", trim(mpierr_string)
     stop 1
   endif

   call e%set("mpi_comm_parent", int(MPI_COMM_WORLD,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("mpi_comm_rows", int(mpi_comm_rows,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("mpi_comm_cols", int(mpi_comm_cols,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)

#else /* SPLIT_COMM_MYSELF */
   call e%set("mpi_comm_parent", int(MPI_COMM_WORLD,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("process_row", int(my_prow,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("process_col", int(my_pcol,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("verbose", 1, error_elpa)
   assert_elpa_ok(error_elpa)
#endif /* SPLIT_COMM_MYSELF */
#endif /* WITH_MPI */

#ifdef TEST_GENERALIZED_EIGENPROBLEM
   call e%set("blacs_context", int(my_blacs_ctxt,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
#endif
   call e%set("timings", 1_ik, error_elpa)
   assert_elpa_ok(error_elpa)

   ! Setup
   assert_elpa_ok(e%setup())

   ! Set solver and ELPA2 kernel

#ifdef TEST_SOLVER_1STAGE
   call e%set("solver", ELPA_SOLVER_1STAGE, error_elpa)
   assert_elpa_ok(error_elpa)
#else
   call e%set("solver", ELPA_SOLVER_2STAGE, error_elpa)
   assert_elpa_ok(error_elpa)
#endif

#if TEST_NVIDIA_GPU == 1
   call e%set("nvidia-gpu", TEST_GPU, error_elpa)
   assert_elpa_ok(error_elpa)
#endif

#if TEST_AMD_GPU == 1
   call e%set("amd-gpu", TEST_GPU, error_elpa)
   assert_elpa_ok(error_elpa)
#endif

#if TEST_INTEL_GPU == 1 || TEST_INTEL_GPU_OPENMP == 1  || TEST_INTEL_GPU_SYCL == 1
   call e%set("intel-gpu", TEST_GPU, error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("sycl_show_all_devices", 0, error_elpa)
   assert_elpa_ok(error_elpa)
#endif

#if defined(TEST_NVIDIA_GPU) || defined(TEST_AMD_GPU) || defined(TEST_INTEL_GPU) || defined(TEST_INTEL_GPU_OPENMP) || defined(TEST_INTEL_GPU_SYCL)
   assert_elpa_ok(e%setup_gpu())
#endif

#if (TEST_GPU_SET_ID == 1) && (TEST_INTEL_GPU == 0) && (TEST_INTEL_GPU_OPENMP == 0) && (TEST_INTEL_GPU_SYCL == 0)
   if (gpu_vendor() /= no_gpu) then
      call set_gpu_parameters()
   else
      print *,"Cannot set gpu vendor!"
      stop 1
   endif

   successGPU = gpu_GetDeviceCount(numberOfDevices)
   if (.not.(successGPU)) then
      print *,"Error in gpu_GetDeviceCount. Aborting..."
      stop 1
   endif
   !print *,"numberOfDevices=", numberOfDevices
   gpuID = mod(myid, numberOfDevices)

   call e%set("use_gpu_id", int(gpuID,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
#endif

#if TEST_GPU_DEVICE_POINTER_API == 1
   ! create device pointers for a,q, ev; copy a to device

   if (gpu_vendor() /= no_gpu) then
     call set_gpu_parameters()
   else
      print *,"Cannot set gpu vendor!"
      stop 1
   endif

   ! Set device
   successGPU = .true.
#if TEST_INTEL_GPU_SYCL == 1
   successGPU = sycl_getcpucount(numberOfDevices) ! temporary fix for SYCL on CPU
   if (.not.(successGPU)) then
      print *,"Error in sycl_getcpucount. Aborting..."
      stop 1
    endif
#endif
   successGPU = gpu_setdevice(gpuID)
   if (.not.(successGPU)) then
     print *,"Cannot set GPU device. Aborting..."
     stop 1
   endif

#if defined(TEST_EIGENVECTORS) && defined(TEST_MATRIX_RANDOM)
   ! malloc
   successGPU = gpu_malloc(a_dev, na_rows*na_cols*size_of_datatype)
   if (.not.(successGPU)) then
     print *,"Cannot allocate matrix a on GPU! Aborting..."
     stop 1
   endif
   successGPU = gpu_malloc(q_dev, na_rows*na_cols*size_of_datatype)
   if (.not.(successGPU)) then
     print *,"Cannot allocate matrix q on GPU! Aborting..."
     stop 1
   endif
   successGPU = gpu_malloc(ev_dev, na*size_of_real_datatype)
   if (.not.(successGPU)) then
     print *,"Cannot allocate vector of eigenvalues on GPU! Aborting..."
     stop 1
   endif

   successGPU = gpu_memcpy(a_dev, c_loc(a), na_rows*na_cols*size_of_datatype, &
                           gpuMemcpyHostToDevice)
   if (.not.(successGPU)) then
     print *,"Cannot copy matrix a to GPU! Aborting..."
     stop 1
   endif
#endif

#if defined(TEST_EIGENVALUES)
   ! malloc
   successGPU = gpu_malloc(a_dev, na_rows*na_cols*size_of_datatype)
   if (.not.(successGPU)) then
     print *,"Cannot allocate matrix a on GPU! Aborting..."
     stop 1
   endif

   successGPU = gpu_malloc(ev_dev, na*size_of_real_datatype)
   if (.not.(successGPU)) then
     print *,"Cannot allocate vector of eigenvalues on GPU! Aborting..."
     stop 1
   endif

   successGPU = gpu_memcpy(a_dev, c_loc(a), na_rows*na_cols*size_of_datatype, &
                           gpuMemcpyHostToDevice)
   if (.not.(successGPU)) then
     print *,"Cannot copy matrix a to GPU! Aborting..."
     stop 1
   endif
#endif

#if defined(TEST_CHOLESKY)
#if TEST_NVIDIA_GPU == 1
   call e%set("nvidia-gpu", TEST_GPU, error_elpa)
   assert_elpa_ok(error_elpa)
#endif

#if TEST_AMD_GPU == 1
   call e%set("amd-gpu", TEST_GPU, error_elpa)
   assert_elpa_ok(error_elpa)
#endif

#if TEST_INTEL_GPU == 1 || (TEST_INTEL_GPU_OPENMP == 1) || (TEST_INTEL_GPU_SYCL == 1)
   call e%set("intel-gpu", TEST_GPU, error_elpa)
   assert_elpa_ok(error_elpa)
#endif

   call e%set("gpu_cholesky",1, error_elpa)
   assert_elpa_ok(error_elpa)

   successGPU = gpu_malloc(a_dev, na_rows*na_cols*size_of_datatype)
   if (.not.(successGPU)) then
     print *,"Cannot allocate matrix a on GPU! Aborting..."
     stop 1
   endif

   successGPU = gpu_memcpy(a_dev, c_loc(a), na_rows*na_cols*size_of_datatype, &
                           gpuMemcpyHostToDevice)
   if (.not.(successGPU)) then
     print *,"Cannot copy matrix a to GPU! Aborting..."
     stop 1
   endif
#endif /* TEST_CHOLESKY */

#if defined(TEST_HERMITIAN_MULTIPLY_FULL) || defined(TEST_HERMITIAN_MULTIPLY_UPPER) || defined(TEST_HERMITIAN_MULTIPLY_LOWER)
#if TEST_NVIDIA_GPU == 1
   call e%set("nvidia-gpu", TEST_GPU, error_elpa)
   assert_elpa_ok(error_elpa)
#endif

#if TEST_AMD_GPU == 1
   call e%set("amd-gpu", TEST_GPU, error_elpa)
   assert_elpa_ok(error_elpa)
#endif

#if TEST_INTEL_GPU == 1 || (TEST_INTEL_GPU_OPENMP == 1) || (TEST_INTEL_GPU_SYCL == 1)
   call e%set("intel-gpu", TEST_GPU, error_elpa)
   assert_elpa_ok(error_elpa)
#endif

   call e%set("gpu_hermitian_multiply",1, error_elpa)
   assert_elpa_ok(error_elpa)

   successGPU = gpu_malloc(a_dev, na_rows*na_cols*size_of_datatype)
   if (.not.(successGPU)) then
     print *,"Cannot allocate matrix a on GPU! Aborting..."
     stop 1
   endif
   successGPU = gpu_malloc(b_dev, na_rows*na_cols*size_of_datatype)
   if (.not.(successGPU)) then
     print *,"Cannot allocate matrix b on GPU! Aborting..."
     stop 1
   endif
   successGPU = gpu_malloc(c_dev, na_rows*na_cols*size_of_datatype)
   if (.not.(successGPU)) then
     print *,"Cannot allocate matrix c on GPU! Aborting..."
     stop 1
   endif

   successGPU = gpu_memcpy(a_dev, c_loc(a), na_rows*na_cols*size_of_datatype, &
                           gpuMemcpyHostToDevice)
   if (.not.(successGPU)) then
     print *,"Cannot copy matrix a to GPU! Aborting..."
     stop 1
   endif

   successGPU = gpu_memcpy(b_dev, c_loc(b), na_rows*na_cols*size_of_datatype, &
                           gpuMemcpyHostToDevice)
   if (.not.(successGPU)) then
     print *,"Cannot copy matrix b to GPU! Aborting..."
     stop 1
   endif

   successGPU = gpu_memcpy(c_dev, c_loc(c), na_rows*na_cols*size_of_datatype, &
                           gpuMemcpyHostToDevice)
   if (.not.(successGPU)) then
     print *,"Cannot copy matrix c to GPU! Aborting..."
     stop 1
   endif


#endif /* TEST_HERMITIAN_MULTIPLY_FULL || defined(TEST_HERMITIAN_MULTIPLY_UPPER) || defined(TEST_HERMITIAN_MULTIPLY_LOWER)*/

#endif /* TEST_GPU_DEVICE_POINTER_API */

#if TEST_QR_DECOMPOSITION == 1
   call e%set("qr", 1_ik, error_elpa)
   assert_elpa_ok(error_elpa)
#endif

#ifdef WITH_OPENMP_TRADITIONAL
   max_threads=omp_get_max_threads()
   call e%set("omp_threads", int(max_threads,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
#endif

   if (myid == 0) print *, ""

#ifdef TEST_ALL_KERNELS
   do i = 0, elpa_option_cardinality(KERNEL_KEY)  ! kernels
#if (TEST_NVIDIA_GPU == 0) && (TEST_INTEL_GPU == 0) && (TEST_AMD_GPU == 0) && (TEST_INTEL_GPU_OPENMP == 0) && (TEST_INTEL_GPU_SYCL == 0)
     !if (TEST_GPU .eq. 0) then
       kernel = elpa_option_enumerate(KERNEL_KEY, int(i,kind=c_int))
       if (kernel .eq. ELPA_2STAGE_REAL_NVIDIA_GPU) continue
       if (kernel .eq. ELPA_2STAGE_COMPLEX_NVIDIA_GPU) continue
       if (kernel .eq. ELPA_2STAGE_REAL_AMD_GPU) continue
       if (kernel .eq. ELPA_2STAGE_COMPLEX_AMD_GPU) continue
       if (kernel .eq. ELPA_2STAGE_REAL_INTEL_GPU_SYCL) continue
       if (kernel .eq. ELPA_2STAGE_COMPLEX_INTEL_GPU_SYCL) continue
     !endif
#endif
#endif

#ifdef TEST_KERNEL
     kernel = TEST_KERNEL
#endif

#ifdef TEST_SOLVER_2STAGE

#ifdef TEST_REAL
#if (TEST_NVIDIA_GPU == 1)
#if WITH_NVIDIA_SM80_GPU_KERNEL == 1
     kernel = ELPA_2STAGE_REAL_NVIDIA_SM80_GPU
#else
     kernel = ELPA_2STAGE_REAL_NVIDIA_GPU
#endif
#endif /* TEST_NVIDIA_GPU */

#if (TEST_AMD_GPU == 1)
     kernel = ELPA_2STAGE_REAL_AMD_GPU
#endif

#if (TEST_INTEL_GPU == 1) || (TEST_INTEL_GPU_OPENMP == 1) || (TEST_INTEL_GPU_SYCL == 1)
     kernel = ELPA_2STAGE_REAL_INTEL_GPU_SYCL
#endif
#endif /* TEST_REAL */

#ifdef TEST_COMPLEX
#if (TEST_NVIDIA_GPU == 1)
     kernel = ELPA_2STAGE_COMPLEX_NVIDIA_GPU
#endif
#if (TEST_AMD_GPU == 1)
     kernel = ELPA_2STAGE_COMPLEX_AMD_GPU
#endif
#if (TEST_INTEL_GPU == 1) || (TEST_INTEL_GPU_OPENMP == 1) || (TEST_INTEL_GPU_SYCL == 1)
     kernel = ELPA_2STAGE_COMPLEX_INTEL_GPU_SYCL
#endif
#endif /* TEST_COMPLEX */

     call e%set(KERNEL_KEY, kernel, error_elpa)
#ifdef TEST_KERNEL
     assert_elpa_ok(error_elpa)
#else
     if (error_elpa /= ELPA_OK) then
       cycle
     endif
     ! actually used kernel might be different if forced via environment variables
     call e%get(KERNEL_KEY, kernel, error_elpa)
     assert_elpa_ok(error_elpa)
#endif
     if (myid == 0) then
       print *, "The settings in the test program want to use ",elpa_int_value_to_string(KERNEL_KEY, kernel) // " kernel"
       print *, "(This might be overriden with some environment settings)"

     endif
#endif /* TEST_SOLVER_2STAGE */

#if !defined(TEST_ALL_LAYOUTS)
! print all parameters
     call e%print_settings(error_elpa)
     assert_elpa_ok(error_elpa)
#endif

#ifdef TEST_ALL_KERNELS
     call e%timer_start(elpa_int_value_to_string(KERNEL_KEY, kernel))
#endif

     !_____________________________________________________________________________________________________________________
     ! The actual solve step

#if defined(TEST_EIGENVECTORS)
#if TEST_QR_DECOMPOSITION == 1
     call e%timer_start("e%eigenvectors_qr()")
#else
     call e%timer_start("e%eigenvectors()")
#endif

#ifdef TEST_SCALAPACK_ALL
     call solve_scalapack_all(na, a, sc_desc, ev, z)
#elif TEST_SCALAPACK_PART
     call solve_scalapack_part(na, a, sc_desc, nev, ev, z)
     check_all_evals = .false. ! scalapack does not compute all eigenvectors
#else /* TEST_SCALAPACK_PART */

#ifdef TEST_EXPLICIT_NAME
#if defined(TEST_REAL)
#if defined(TEST_DOUBLE)
#if (TEST_GPU_DEVICE_POINTER_API == 1) && defined(TEST_MATRIX_RANDOM) && defined(TEST_EIGENVECTORS)
     call e%eigenvectors_double(a_dev, ev_dev, q_dev, error_elpa)
     assert_elpa_ok(error_elpa)
#else
     call e%eigenvectors_double(a, ev, z, error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_DOUBLE */
#if defined(TEST_SINGLE)
#if (TEST_GPU_DEVICE_POINTER_API == 1) && defined(TEST_MATRIX_RANDOM) && defined(TEST_EIGENVECTORS)
     call e%eigenvectors_float(a_dev, ev_dev, q_dev, error_elpa)
     assert_elpa_ok(error_elpa)
#else
     call e%eigenvectors_float(a, ev, z, error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_SINGLE */
#endif /* TEST_REAL */

#if defined(TEST_COMPLEX)
#if defined(TEST_DOUBLE)
#if (TEST_GPU_DEVICE_POINTER_API == 1) && defined(TEST_MATRIX_RANDOM) && defined(TEST_EIGENVECTORS)
     call e%eigenvectors_double_complex(a_dev, ev_dev, q_dev, error_elpa)
     assert_elpa_ok(error_elpa)
#else
     call e%eigenvectors_double_complex(a, ev, z, error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_DOUBLE */
#if defined(TEST_SINGLE)
#if (TEST_GPU_DEVICE_POINTER_API == 1) && defined(TEST_MATRIX_RANDOM) && defined(TEST_EIGENVECTORS)
     call e%eigenvectors_float_complex(a_dev, ev_dev, q_dev, error_elpa)
     assert_elpa_ok(error_elpa)
#else
     call e%eigenvectors_float_complex(a, ev, z, error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_SINGLE */
#endif /* TEST_COMPLEX */
#else /* TEST_EXPLICIT_NAME */
     call e%eigenvectors(a, ev, z, error_elpa)
     assert_elpa_ok(error_elpa)
#endif /* TEST_EXPLICIT_NAME */
#endif /* TEST_SCALAPACK_PART */
#if TEST_QR_DECOMPOSITION == 1
     call e%timer_stop("e%eigenvectors_qr()")
#else
     call e%timer_stop("e%eigenvectors()")
#endif
#endif /* TEST_EIGENVECTORS  */

#ifdef TEST_EIGENVALUES
     call e%timer_start("e%eigenvalues()")
#ifdef TEST_EXPLICIT_NAME
#if defined(TEST_REAL)
#if defined(TEST_DOUBLE)
#if (TEST_GPU_DEVICE_POINTER_API == 1)
     call e%eigenvalues_double(a_dev, ev_dev, error_elpa)
     assert_elpa_ok(error_elpa)
#else
     call e%eigenvalues_double(a, ev, error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_DOUBLE */
#if defined(TEST_SINGLE)
#if (TEST_GPU_DEVICE_POINTER_API == 1)
     call e%eigenvalues_float(a_dev, ev_dev, error_elpa)
     assert_elpa_ok(error_elpa)
#else
     call e%eigenvalues_float(a, ev, error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_SINGLE */
#endif /* TEST_REAL */
#if defined(TEST_COMPLEX)
#if defined(TEST_DOUBLE)
#if (TEST_GPU_DEVICE_POINTER_API == 1)
     call e%eigenvalues_double_complex(a_dev, ev_dev, error_elpa)
     assert_elpa_ok(error_elpa)
#else
     call e%eigenvalues_double_complex(a, ev, error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_DOUBLE */
#if defined(TEST_SINGLE)
#if (TEST_GPU_DEVICE_POINTER_API == 1)
     call e%eigenvalues_float_complex(a_dev, ev_dev, error_elpa)
     assert_elpa_ok(error_elpa)
#else
     call e%eigenvalues_float_complex(a, ev, error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_SINGLE */
#endif /* TEST_COMPLEX */
#else /* TEST_EXPLICIT_NAME */
     call e%eigenvalues(a, ev, error_elpa)
     assert_elpa_ok(error_elpa)
#endif /* TEST_EXPLICIT_NAME */
     call e%timer_stop("e%eigenvalues()")
#endif /* TEST_EIGENVALUES */

#if defined(TEST_SOLVE_TRIDIAGONAL)
     call e%timer_start("e%solve_tridiagonal()")
     call e%solve_tridiagonal(d, sd, z, error_elpa)
     assert_elpa_ok(error_elpa)
     call e%timer_stop("e%solve_tridiagonal()")
     ev(:) = d(:)
#endif

#if defined(TEST_CHOLESKY)
     call e%timer_start("e%cholesky()")

#if TEST_GPU_DEVICE_POINTER_API == 1

#if defined(TEST_REAL)
#if defined(TEST_DOUBLE)
     call e%cholesky_double(a_dev, error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#if defined(TEST_SINGLE)
     call e%cholesky_float(a_dev, error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_REAL */
#if defined(TEST_COMPLEX)
#if defined(TEST_DOUBLE)
     call e%cholesky_double_complex(a_dev, error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#if defined(TEST_SINGLE)
     call e%cholesky_float_complex(a_dev, error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_COMPLEX */

#else /* TEST_GPU_DEVICE_POINTER_API */
     call e%cholesky(a, error_elpa)
     assert_elpa_ok(error_elpa)
#endif /* TEST_GPU_DEVICE_POINTER_API */
     call e%timer_stop("e%cholesky()")
#endif /* TEST_CHOLESKY */

#if defined(TEST_HERMITIAN_MULTIPLY_FULL)
     call e%timer_start("e%hermitian_multiply_full()")
#if TEST_GPU_DEVICE_POINTER_API == 1
#if defined(TEST_REAL)
#if defined(TEST_DOUBLE)
     call e%hermitian_multiply_double('F','F', int(na,kind=c_int), a_dev, b_dev, int(na_rows,kind=c_int), &
                               int(na_cols,kind=c_int), c_dev, int(na_rows,kind=c_int),        &
                               int(na_cols,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#if defined(TEST_SINGLE)
     call e%hermitian_multiply_float('F','F', int(na,kind=c_int), a_dev, b_dev, int(na_rows,kind=c_int), &
                               int(na_cols,kind=c_int), c_dev, int(na_rows,kind=c_int),        &
                               int(na_cols,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_REAL */
#if defined(TEST_COMPLEX)
#if defined(TEST_DOUBLE)
     call e%hermitian_multiply_double_complex('F','F', int(na,kind=c_int), a_dev, b_dev, int(na_rows,kind=c_int), &
                               int(na_cols,kind=c_int), c_dev, int(na_rows,kind=c_int),        &
                               int(na_cols,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#if defined(TEST_SINGLE)
     call e%hermitian_multiply_float_complex('F','F', int(na,kind=c_int), a_dev, b_dev, int(na_rows,kind=c_int), &
                               int(na_cols,kind=c_int), c_dev, int(na_rows,kind=c_int),        &
                               int(na_cols,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_COMPLEX */
#else /* TEST_GPU_DEVICE_POINTER_API */
     call e%hermitian_multiply('F','F', int(na,kind=c_int), a, b, int(na_rows,kind=c_int), &
                               int(na_cols,kind=c_int), c, int(na_rows,kind=c_int),        &
                               int(na_cols,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif /* TEST_GPU_DEVICE_POINTER_API */
     call e%timer_stop("e%hermitian_multiply_full()")
#endif /* TEST_HERMITIAN_MULTIPLY_FULL */

#if defined(TEST_HERMITIAN_MULTIPLY_UPPER)
     call e%timer_start("e%hermitian_multiply_upper()")
#if TEST_GPU_DEVICE_POINTER_API == 1
#if defined(TEST_REAL)
#if defined(TEST_DOUBLE)
     call e%hermitian_multiply_double('U','U', int(na,kind=c_int), a_dev, b_dev, int(na_rows,kind=c_int), &
                               int(na_cols,kind=c_int), c_dev, int(na_rows,kind=c_int),        &
                               int(na_cols,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#if defined(TEST_SINGLE)
     call e%hermitian_multiply_float('U','U', int(na,kind=c_int), a_dev, b_dev, int(na_rows,kind=c_int), &
                               int(na_cols,kind=c_int), c_dev, int(na_rows,kind=c_int),        &
                               int(na_cols,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_REAL */
#if defined(TEST_COMPLEX)
#if defined(TEST_DOUBLE)
     call e%hermitian_multiply_double_complex('U','U', int(na,kind=c_int), a_dev, b_dev, int(na_rows,kind=c_int), &
                               int(na_cols,kind=c_int), c_dev, int(na_rows,kind=c_int),        &
                               int(na_cols,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#if defined(TEST_SINGLE)
     call e%hermitian_multiply_float_complex('U','U', int(na,kind=c_int), a_dev, b_dev, int(na_rows,kind=c_int), &
                               int(na_cols,kind=c_int), c_dev, int(na_rows,kind=c_int),        &
                               int(na_cols,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_COMPLEX */
#else /* TEST_GPU_DEVICE_POINTER_API */
     call e%hermitian_multiply('U','U', int(na,kind=c_int), a, b, int(na_rows,kind=c_int), &
                               int(na_cols,kind=c_int), c, int(na_rows,kind=c_int),        &
                               int(na_cols,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif /* TEST_GPU_DEVICE_POINTER_API */
     call e%timer_stop("e%hermitian_multiply_upper()")
#endif /* TEST_HERMITIAN_MULTIPLY_UPPER */

#if defined(TEST_HERMITIAN_MULTIPLY_LOWER)
     call e%timer_start("e%hermitian_multiply_lower()")
#if TEST_GPU_DEVICE_POINTER_API == 1
#if defined(TEST_REAL)
#if defined(TEST_DOUBLE)
     call e%hermitian_multiply_double('L','L', int(na,kind=c_int), a_dev, b_dev, int(na_rows,kind=c_int), &
                               int(na_cols,kind=c_int), c_dev, int(na_rows,kind=c_int),        &
                               int(na_cols,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#if defined(TEST_SINGLE)
     call e%hermitian_multiply_float('L','L', int(na,kind=c_int), a_dev, b_dev, int(na_rows,kind=c_int), &
                               int(na_cols,kind=c_int), c_dev, int(na_rows,kind=c_int),        &
                               int(na_cols,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_REAL */
#if defined(TEST_COMPLEX)
#if defined(TEST_DOUBLE)
     call e%hermitian_multiply_double_complex('L','L', int(na,kind=c_int), a_dev, b_dev, int(na_rows,kind=c_int), &
                               int(na_cols,kind=c_int), c_dev, int(na_rows,kind=c_int),        &
                               int(na_cols,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#if defined(TEST_SINGLE)
     call e%hermitian_multiply_float_complex('L','L', int(na,kind=c_int), a_dev, b_dev, int(na_rows,kind=c_int), &
                               int(na_cols,kind=c_int), c_dev, int(na_rows,kind=c_int),        &
                               int(na_cols,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif
#endif /* TEST_COMPLEX */
#else /* TEST_GPU_DEVICE_POINTER_API */
     call e%hermitian_multiply('L','L', int(na,kind=c_int), a, b, int(na_rows,kind=c_int), &
                               int(na_cols,kind=c_int), c, int(na_rows,kind=c_int),        &
                               int(na_cols,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif /* TEST_GPU_DEVICE_POINTER_API */
     call e%timer_stop("e%hermitian_multiply_lower()")
#endif /* TEST_HERMITIAN_MULTIPLY_LOWER */

#if defined(TEST_GENERALIZED_EIGENPROBLEM)
     call e%timer_start("e%generalized_eigenvectors()")
#if defined(TEST_GENERALIZED_DECOMP_EIGENPROBLEM)
     call e%timer_start("is_already_decomposed=.false.")
#endif
     call e%generalized_eigenvectors(a, b, ev, z, .false., error_elpa)
     assert_elpa_ok(error_elpa)
#if defined(TEST_GENERALIZED_DECOMP_EIGENPROBLEM)
     call e%timer_stop("is_already_decomposed=.false.")
     a = as ! so that the problem can be solved again
     call e%timer_start("is_already_decomposed=.true.")
     call e%generalized_eigenvectors(a, b, ev, z, .true., error_elpa)
     assert_elpa_ok(error_elpa)
     call e%timer_stop("is_already_decomposed=.true.")
#endif /* TEST_GENERALIZED_DECOMP_EIGENPROBLEM */
     call e%timer_stop("e%generalized_eigenvectors()")
#endif /* TEST_GENERALIZED_EIGENPROBLEM */

#ifdef TEST_ALL_KERNELS
     call e%timer_stop(elpa_int_value_to_string(KERNEL_KEY, kernel))
#endif

     if (myid .eq. 0) then
#ifdef TEST_ALL_KERNELS
       call e%print_times(elpa_int_value_to_string(KERNEL_KEY, kernel))
#else /* TEST_ALL_KERNELS */

#if defined(TEST_EIGENVECTORS)
#if TEST_QR_DECOMPOSITION == 1
       call e%print_times("e%eigenvectors_qr()")
#else
       call e%print_times("e%eigenvectors()")
#endif
#endif /* TEST_EIGENVECTORS */
#ifdef TEST_EIGENVALUES
       call e%print_times("e%eigenvalues()")
#endif
#ifdef TEST_SOLVE_TRIDIAGONAL
       call e%print_times("e%solve_tridiagonal()")
#endif
#ifdef TEST_CHOLESKY
       call e%print_times("e%cholesky()")
#endif
#ifdef TEST_HERMITIAN_MULTIPLY_FULL
       call e%print_times("e%hermitian_multiply_full()")
#endif
#ifdef TEST_HERMITIAN_MULTIPLY_UPPER
       call e%print_times("e%hermitian_multiply_upper()")
#endif
#ifdef TEST_HERMITIAN_MULTIPLY_LOWER
       call e%print_times("e%hermitian_multiply_lower()")
#endif
#ifdef TEST_GENERALIZED_EIGENPROBLEM
      call e%print_times("e%generalized_eigenvectors()")
#endif
#endif /* TEST_ALL_KERNELS */
     endif


   !_____________________________________________________________________________________________________________________
   ! TEST_GPU_DEVICE_POINTER_API case: copy for testing from device to host

#if TEST_GPU_DEVICE_POINTER_API == 1

#if defined(TEST_EIGENVECTORS) && defined(TEST_MATRIX_RANDOM)
   ! copy for testing
   successGPU = gpu_memcpy(c_loc(z), q_dev, na_rows*na_cols*size_of_datatype, &
                           gpuMemcpyDeviceToHost)
   if (.not.(successGPU)) then
     print *,"cannot copy matrix of eigenvectors from GPU to host! Aborting..."
     stop 1
   endif

   successGPU = gpu_memcpy(c_loc(ev), ev_dev, na*&
#ifdef TEST_DOUBLE
           size_of_double_real, &
#endif
#ifdef TEST_SINGLE
           size_of_single_real, &
#endif
                           gpuMemcpyDeviceToHost)
   if (.not.(successGPU)) then
     print *,"cannot copy vector of eigenvalues from GPU to host! Aborting..."
     stop 1
   endif

#endif /* defined(TEST_EIGENVECTORS) && defined(TEST_MATRIX_RANDOM) */

#if defined(TEST_EIGENVALUES)
   ! copy for testing
   successGPU = gpu_memcpy(c_loc(ev), ev_dev, na*size_of_real_datatype, gpuMemcpyDeviceToHost)
   if (.not.(successGPU)) then
     print *,"cannot copy vector of eigenvalues from GPU to host! Aborting..."
     stop 1
   endif
#endif /* TEST_EIGENVALUES  */


#if defined(TEST_CHOLESKY)
   successGPU = gpu_memcpy(c_loc(a), a_dev, na_rows*na_cols*size_of_datatype, &
                           gpuMemcpyDeviceToHost)
   if (.not.(successGPU)) then
     print *,"cannot copy matrix of eigenvectors from GPU to host! Aborting..."
     stop 1
   endif
#endif /* TEST_CHOLESKY */

#if defined(TEST_HERMITIAN_MULTIPLY_FULL) || defined(TEST_HERMITIAN_MULTIPLY_UPPER) || defined(TEST_HERMITIAN_MULTIPLY_LOWER)
   successGPU = gpu_memcpy(c_loc(a), a_dev, na_rows*na_cols*size_of_datatype, &
                           gpuMemcpyDeviceToHost)
   if (.not.(successGPU)) then
     print *,"Cannot copy matrix a_dev -> a ! Aborting..."
     stop 1
   endif

   successGPU = gpu_memcpy(c_loc(b), b_dev, na_rows*na_cols*size_of_datatype, &
                           gpuMemcpyDeviceToHost)
   if (.not.(successGPU)) then
     print *,"Cannot copy matrix b_dev -> b ! Aborting..."
     stop 1
   endif

   successGPU = gpu_memcpy(c_loc(c), c_dev, na_rows*na_cols*size_of_datatype, &
                           gpuMemcpyDeviceToHost)
   if (.not.(successGPU)) then
     print *,"Cannot copy matrix c_dev -> c ! Aborting..."
     stop 1
   endif
#endif /* TEST_HERMITIAN_MULTIPLY_FULL || TEST_HERMITIAN_MULTIPLY_UPPER || defined(TEST_HERMITIAN_MULTIPLY_LOWER) */

#endif /* TEST_GPU_DEVICE_POINTER_API */

     !_____________________________________________________________________________________________________________________
     ! Check the results

     if (do_test_analytic_eigenvalues) then
       status = check_correctness_analytic(na, nev, ev, z, nblk, myid, np_rows, np_cols, &
                                           my_prow, my_pcol, check_all_evals, .false.)
       call check_status(status, myid)
     endif

     if (do_test_analytic_eigenvalues_eigenvectors) then
       status = check_correctness_analytic(na, nev, ev, z, nblk, myid, np_rows, np_cols, &
                                           my_prow, my_pcol, check_all_evals, .true.)
       call check_status(status, myid)
     endif

     if(do_test_numeric_residual) then
       status = check_correctness_evp_numeric_residuals(na, nev, as, z, ev, sc_desc, nblk, myid, &
                                                        np_rows,np_cols, my_prow, my_pcol)
       call check_status(status, myid)
     endif

     if (do_test_frank_eigenvalues) then
       status = check_correctness_eigenvalues_frank(na, ev, z, myid)
       call check_status(status, myid)
     endif

     if (do_test_toeplitz_eigenvalues) then
#if defined(TEST_EIGENVALUES) || defined(TEST_SOLVE_TRIDIAGONAL)
       status = check_correctness_eigenvalues_toeplitz(na, diagonalElement, &
                                                       subdiagonalElement, ev, z, myid)
       call check_status(status, myid)
#endif
     endif

     if (do_test_cholesky) then
       status = check_correctness_cholesky(na, a, as, na_rows, sc_desc, myid )
       call check_status(status, myid)
     endif

#ifdef TEST_HERMITIAN_MULTIPLY_FULL
     if (do_test_hermitian_multiply) then
       status = check_correctness_hermitian_multiply("H", na, a, b, c, na_rows, sc_desc, myid )
       call check_status(status, myid)
     endif
#endif

#ifdef TEST_HERMITIAN_MULTIPLY_UPPER
     if (do_test_hermitian_multiply) then
       isUpper = .true.
       isLower = .false.
       status = check_correctness_hermitian_multiply("H", na, a, b, c, na_rows, sc_desc, myid, &
                                                     nblk, np_rows, np_cols, my_prow, my_pcol, &
                                                     isUpper, isLower )
       call check_status(status, myid)
     endif
#endif

#ifdef TEST_HERMITIAN_MULTIPLY_LOWER
     if (do_test_hermitian_multiply) then
       isUpper = .false.
       isLower = .true.
       status = check_correctness_hermitian_multiply("H", na, a, b, c, na_rows, sc_desc, myid, &
                                                     nblk, np_rows, np_cols, my_prow, my_pcol, &
                                                     isUpper, isLower )
       call check_status(status, myid)
     endif
#endif

#ifdef TEST_GENERALIZED_EIGENPROBLEM
     if(do_test_numeric_residual_generalized) then
       status = check_correctness_evp_numeric_residuals(na, nev, as, z, ev, sc_desc, nblk, myid, np_rows, &
                                                        np_cols, my_prow, &
       my_pcol, bs)
       call check_status(status, myid)
     endif
#endif


#ifdef WITH_OPENMP_TRADITIONAL
     if (threads_caller .ne. omp_get_max_threads()) then
       if (myid .eq. 0) then
         print *, " ERROR! the number of OpenMP threads has not been restored correctly"
       endif
       status = 1
     endif
#endif
     if (myid == 0) then
       print *, ""
     endif

#ifdef TEST_ALL_KERNELS
     a(:,:) = as(:,:)
#if defined(TEST_MATRIX_TOEPLITZ) || defined(TEST_MATRIX_FRANK)
     d = ds
     sd = sds
#endif
   end do ! kernels
#endif /* TEST_ALL_KERNELS */

   !_____________________________________________________________________________________________________________________
   ! Deallocate

#if TEST_GPU_DEVICE_POINTER_API == 1

#if defined(TEST_EIGENVECTORS) && defined(TEST_MATRIX_RANDOM)
   ! and deallocate device pointers
   successGPU = gpu_free(a_dev)
   if (.not.(successGPU)) then
     print *,"cannot free memory of a_dev on GPU. Aborting..."
     stop 1
   endif
   successGPU = gpu_free(q_dev)
   if (.not.(successGPU)) then
     print *,"cannot free memory of q_dev on GPU. Aborting..."
     stop 1
   endif
   successGPU = gpu_free(ev_dev)
   if (.not.(successGPU)) then
     print *,"cannot free memory of ev_dev on GPU. Aborting..."
     stop 1
   endif
#endif /* defined(TEST_EIGENVECTORS) && defined(TEST_MATRIX_RANDOM) */

#if defined(TEST_EIGENVALUES)
   successGPU = gpu_free(a_dev)
   if (.not.(successGPU)) then
     print *,"cannot free memory of a_dev on GPU. Aborting..."
     stop 1
   endif

   successGPU = gpu_free(ev_dev)
   if (.not.(successGPU)) then
     print *,"cannot free memory of ev_dev on GPU. Aborting..."
     stop 1
   endif
#endif /* TEST_EIGENVALUES  */

#if defined(TEST_CHOLESKY)
   successGPU = gpu_free(a_dev)
   if (.not.(successGPU)) then
     print *,"cannot free memory of a_dev on GPU. Aborting..."
     stop 1
   endif
#endif /* TEST_CHOLESKY */

#if defined(TEST_HERMITIAN_MULTIPLY_FULL) || defined(TEST_HERMITIAN_MULTIPLY_UPPER)
   successGPU = gpu_free(a_dev)
   if (.not.(successGPU)) then
     print *,"cannot free memory of a_dev on GPU. Aborting..."
     stop 1
   endif

   successGPU = gpu_free(b_dev)
   if (.not.(successGPU)) then
     print *,"cannot free memory of b_dev on GPU. Aborting..."
     stop 1
   endif

   successGPU = gpu_free(c_dev)
   if (.not.(successGPU)) then
     print *,"cannot free memory of c_dev on GPU. Aborting..."
     stop 1
   endif
#endif /* TEST_HERMITIAN_MULTIPLY_FULL */

#endif /* TEST_GPU_DEVICE_POINTER_API */

   call elpa_deallocate(e, error_elpa)
   assert_elpa_ok(error_elpa)

   deallocate(a)
   deallocate(as)
   deallocate(z)
   deallocate(ev)

#if defined(TEST_HERMITIAN_MULTIPLY_FULL) || defined(TEST_HERMITIAN_MULTIPLY_LOWER) || defined(TEST_HERMITIAN_MULTIPLY_UPPER) 
   deallocate(b)
   deallocate(c)
#endif
#if defined(TEST_MATRIX_TOEPLITZ) || defined(TEST_MATRIX_FRANK)
   deallocate(d, ds)
   deallocate(sd, sds)
#endif
#if defined(TEST_GENERALIZED_EIGENPROBLEM)
  deallocate(b, bs)
#endif

#if defined(WITH_MPI) && defined (SPLIT_COMM_MYSELF)
   call mpi_comm_free(mpi_comm_rows, mpierr)
   if (mpierr .ne. MPI_SUCCESS) then
     call MPI_ERROR_STRING(mpierr, mpierr_string, mpi_string_length, mpierr2)
     write(error_unit,*) "MPI ERROR occured during mpi_comm_free for row communicator: ", trim(mpierr_string)
     stop 1
   endif

   call mpi_comm_free(mpi_comm_cols, mpierr)
   if (mpierr .ne. MPI_SUCCESS) then
     call MPI_ERROR_STRING(mpierr, mpierr_string, mpi_string_length, mpierr2)
     write(error_unit,*) "MPI ERROR occured during mpi_comm_free for column communicator: ", trim(mpierr_string)
     stop 1
   endif
#endif /* WITH_MPI && SPLIT_COMM_MYSELF */

#ifdef WITH_MPI
   call blacs_gridexit(my_blacs_ctxt)
#endif

#ifdef TEST_ALL_LAYOUTS
   end do ! factors
   end do ! layouts
#endif

   call elpa_uninit(error_elpa)
   assert_elpa_ok(error_elpa)

#ifdef WITH_MPI
   call mpi_finalize(mpierr)
#endif

   call exit(status)

   contains

     subroutine check_status(status, myid)
       implicit none
       TEST_INT_TYPE, intent(in) :: status, myid
       TEST_INT_MPI_TYPE         :: mpierr
       if (status /= 0) then
         if (myid == 0) print *, "Result incorrect!"
#ifdef WITH_MPI
         call mpi_finalize(mpierr)
#endif
         call exit(status)
       endif
     end subroutine

end program
