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
! Define TEST_GPU \in [0, 1]
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

#ifdef TEST_SINGLE
#  define SINGLE_PRECISION 1
#  undef  DOUBLE_PRECISION
#else
#  define DOUBLE_PRECISION 1
#  undef  SINGLE_PRECISION
#endif

#ifdef TEST_REAL
#  define REALCASE 1
#  undef  COMPLEXCASE
#  define MATH_DATATYPE real
#  define KERNEL_KEY "real_kernel"
#  ifdef TEST_SINGLE
#    define BLAS_CHAR S
#  else
#    define BLAS_CHAR D
#  endif
#else
#  define COMPLEXCASE 1
#  undef  REALCASE
#  define MATH_DATATYPE complex
#  define KERNEL_KEY "complex_kernel"
#  ifdef TEST_SINGLE
#    define BLAS_CHAR C
#  else
#    define BLAS_CHAR Z
#  endif
#endif

#include "assert.h"

program test
   use elpa

   use test_util
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
   implicit none

#include "../../src/general/precision_kinds.F90"

#define EV_TYPE real(kind=rk)
#define MATRIX_TYPE MATH_DATATYPE(kind=rck)

   ! matrix dimensions
   integer                     :: na, nev, nblk

   ! mpi
   integer                     :: myid, nprocs
   integer                     :: na_cols, na_rows  ! local matrix size
   integer                     :: np_cols, np_rows  ! number of MPI processes per column/row
   integer                     :: my_prow, my_pcol  ! local MPI task position (my_prow, my_pcol) in the grid (0..np_cols -1, 0..np_rows -1)
   integer                     :: mpierr

   ! blacs
   integer                     :: my_blacs_ctxt, sc_desc(9), info, nprow, npcol

   ! The Matrix
   MATRIX_TYPE, allocatable    :: a(:,:), as(:,:)
#if defined(TEST_HERMITIAN_MULTIPLY)
   MATRIX_TYPE, allocatable    :: b(:,:), c(:,:)
#endif
#if defined(TEST_GENERALIZED_EIGENPROBLEM)
   MATRIX_TYPE, allocatable    :: b(:,:), bs(:,:)
#endif
   ! eigenvectors
   MATRIX_TYPE, allocatable    :: z(:,:)
   ! eigenvalues
   EV_TYPE, allocatable        :: ev(:), ev_analytic(:)

   logical                     :: check_all_evals

   EV_TYPE, allocatable        :: d(:), sd(:), ds(:), sds(:)
   EV_TYPE                     :: diagonalELement, subdiagonalElement

   integer                     :: error, status

   type(output_t)              :: write_to_file
   class(elpa_t), pointer      :: e
#ifdef TEST_ALL_KERNELS
   integer                     :: i
#endif
#ifdef TEST_ALL_LAYOUTS
   character(len=1), parameter :: layouts(2) = [ 'C', 'R' ]
   integer                     :: i_layout
#endif
   integer                     :: kernel
   character(len=1)            :: layout
#ifdef TEST_COMPLEX
   EV_TYPE                     :: norm, normmax
   MATRIX_TYPE, allocatable    :: tmp1(:,:), tmp2(:,:)
#endif
   call read_input_parameters_traditional(na, nev, nblk, write_to_file)
   call setup_mpi(myid, nprocs)
#ifdef HAVE_REDIRECT
#ifdef WITH_MPI
     call MPI_BARRIER(MPI_COMM_WORLD, mpierr)
     call redirect_stdout(myid)
#endif
#endif

   check_all_evals = .true.

   if (elpa_init(CURRENT_API_VERSION) /= ELPA_OK) then
     print *, "ELPA API version not supported"
     stop 1
   endif

   if (myid == 0) then
     print '((a,i0))', 'Program ' // TEST_CASE
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

#ifdef TEST_QR_DECOMPOSITION

#if TEST_GPU == 1
#ifdef WITH_MPI
     call mpi_finalize(mpierr)
#endif
     stop 77
#endif /* TEST_GPU */
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

   call set_up_blacsgrid(mpi_comm_world, np_rows, np_cols, layout, &
                         my_blacs_ctxt, my_prow, my_pcol)

   call set_up_blacs_descriptor(na, nblk, my_prow, my_pcol, np_rows, np_cols, &
                                na_rows, na_cols, sc_desc, my_blacs_ctxt, info)

   e => elpa_allocate()

   call e%set("na", na, error)
   assert_elpa_ok(error)
   call e%set("nev", nev, error)
   assert_elpa_ok(error)
   call e%set("local_nrows", na_rows, error)
   assert_elpa_ok(error)
   call e%set("local_ncols", na_cols, error)
   assert_elpa_ok(error)
   call e%set("nblk", nblk, error)
   assert_elpa_ok(error)

#ifdef WITH_MPI
   call e%set("mpi_comm_parent", MPI_COMM_WORLD, error)
   assert_elpa_ok(error)
   call e%set("process_row", my_prow, error)
   assert_elpa_ok(error)
   call e%set("process_col", my_pcol, error)
   assert_elpa_ok(error)
#endif

   call e%set("timings",1)
   call e%set("debug", 1)

   assert_elpa_ok(e%setup())

#ifdef TEST_SOLVER_1STAGE
   call e%set("solver", ELPA_SOLVER_1STAGE)
#else
   call e%set("solver", ELPA_SOLVER_2STAGE)
#endif
   assert_elpa_ok(error)

   call e%set("gpu", TEST_GPU, error)
   assert_elpa_ok(error)

#ifdef TEST_QR_DECOMPOSITION
   call e%set("qr", 1, error)
   assert_elpa_ok(error)
#endif

   if (myid == 0) print *, ""

   ! allocate matrices
   allocate(a (na_rows,na_cols))
   allocate(as(na_rows,na_cols))
   allocate(z (na_rows,na_cols))
   allocate(ev(na))

#ifdef TEST_HERMITIAN_MULTIPLY
   allocate(b (na_rows,na_cols))
   allocate(c (na_rows,na_cols))
#endif

#ifdef TEST_GENERALIZED_EIGENPROBLEM
   allocate(b (na_rows,na_cols))
   allocate(bs (na_rows,na_cols))
   ! todo: only temporarily, before we start using random SPD matrix for B
   allocate(d (na), ds(na))
   allocate(sd (na), sds(na))
#endif

#if defined(TEST_EIGENVALUES) || defined(TEST_SOLVE_TRIDIAGONAL) || defined(TEST_EIGENVECTORS) || defined(TEST_QR_DECOMPOSITION) || defined(TEST_CHOLESKY)
   allocate(d (na), ds(na))
   allocate(sd (na), sds(na))
   allocate(ev_analytic(na))
#endif

   ! prepare matrices
   call e%timer_start("prepare_matrices")
   a(:,:) = 0.0
   z(:,:) = 0.0
   ev(:) = 0.0

#if defined(TEST_EIGENVECTORS) || defined(TEST_HERMITIAN_MULTIPLY) || defined(TEST_QR_DECOMPOSITION) || defined(TEST_GENERALIZED_EIGENPROBLEM)
#ifdef TEST_MATRIX_ANALYTIC
   call prepare_matrix_analytic(na, a, nblk, myid, np_rows, np_cols, my_prow, my_pcol)
   as(:,:) = a
#else  /* TEST_MATRIX_ANALYTIC */
   if (nev .ge. 1) then
     call prepare_matrix_random(na, myid, sc_desc, a, z, as)
#if defined(TEST_GENERALIZED_EIGENPROBLEM)
     !call prepare_matrix_random(na, myid, sc_desc, b, z, bs)
     ! TODO create random SPD matrix
     !diagonalElement = (2.546_rk, 0.0_rk)
     diagonalElement = ONE
     subdiagonalElement =  (0.0_rk, 0.0_rk)
     call prepare_matrix_toeplitz(na, diagonalElement, subdiagonalElement, &
                                  d, sd, ds, sds, b, bs, nblk, np_rows, &
                                  np_cols, my_prow, my_pcol)
#endif
   else
     ! zero eigenvectors and not analytic test => toeplitz matrix
     diagonalElement = 0.45_rk
     subdiagonalElement =  0.78_rk
     call prepare_matrix_toeplitz(na, diagonalElement, subdiagonalElement, &
                                  d, sd, ds, sds, a, as, nblk, np_rows, &
                                  np_cols, my_prow, my_pcol)
   endif

#ifdef TEST_HERMITIAN_MULTIPLY
   b(:,:) = 2.0_rk * a(:,:)
   c(:,:) = ONE
#endif /* TEST_HERMITIAN_MULTIPLY */

#endif /* (not) TEST_MATRIX_ANALYTIC */
#endif /* defined(TEST_EIGENVECTORS) || defined(TEST_HERMITIAN_MULTIPLY) || defined(TEST_QR_DECOMPOSITION) */

#if defined(TEST_EIGENVALUES) || defined(TEST_SOLVE_TRIDIAGONAL)
   diagonalElement = 0.45_rk
   subdiagonalElement =  0.78_rk
   call prepare_matrix_toeplitz(na, diagonalElement, subdiagonalElement, &
                                d, sd, ds, sds, a, as, nblk, np_rows, &
                                np_cols, my_prow, my_pcol)
#endif /* EIGENVALUES OR TRIDIAGONAL */

#if defined(TEST_CHOLESKY)
   diagonalElement = 2.0_rk * ONE
   subdiagonalElement = -1.0_rk * ONE
   call prepare_matrix_toeplitz(na, diagonalElement, subdiagonalElement, &
                                d, sd, ds, sds, a, as, nblk, np_rows, &
                                np_cols, my_prow, my_pcol)
#endif /* TEST_CHOLESKY */
   call e%timer_stop("prepare_matrices")
   if (myid == 0) then
     print *, ""
     call e%print_times("prepare_matrices")
     print *, ""
   endif

   ! solve the problem
#ifdef TEST_ALL_KERNELS
   do i = 0, elpa_option_cardinality(KERNEL_KEY)  ! kernels
     kernel = elpa_option_enumerate(KERNEL_KEY, i)
#endif
#ifdef TEST_KERNEL
     kernel = TEST_KERNEL
#endif

#ifdef TEST_SOLVER_2STAGE
     call e%set(KERNEL_KEY, kernel, error)
#ifdef TEST_KERNEL
     assert_elpa_ok(error)
#else
     if (error /= ELPA_OK) then
       cycle
     endif
     ! actually used kernel might be different if forced via environment variables
     call e%get(KERNEL_KEY, kernel)
#endif
     if (myid == 0) then
       print *, elpa_int_value_to_string(KERNEL_KEY, kernel) // " kernel"
     endif
#endif

#ifdef TEST_ALL_KERNELS
     call e%timer_start(elpa_int_value_to_string(KERNEL_KEY, kernel))
#endif

!#if defined(TEST_GENERALIZED_EIGENPROBLEM)
!     call e%timer_start("generalized_eigenproblem")
!     call e%timer_start("e%cholesky()")
!     call e%cholesky(b, error)
!     assert_elpa_ok(error)
!     call e%timer_stop("e%cholesky()")
!     call e%timer_start("e%invert_triangular")
!     call e%invert_triangular(b, error)
!     assert_elpa_ok(error)
!     call e%timer_stop("e%invert_triangular")
!#ifdef WITH_MPI
!     
!#endif
!#endif

     ! The actual solve step
#if defined(TEST_EIGENVECTORS) || defined(TEST_QR_DECOMPOSITION) || defined(TEST_GENERALIZED_EIGENPROBLEM)
     call e%timer_start("e%eigenvectors()")
#ifdef TEST_SCALAPACK_ALL
     call solve_scalapack_all(na, a, sc_desc, ev, z)
#elif TEST_SCALAPACK_PART
     call solve_scalapack_part(na, a, sc_desc, nev, ev, z)
     check_all_evals = .false. ! scalapack does not compute all eigenvectors
#elif TEST_GENERALIZED_EIGENPROBLEM
     call e%generalized_eigenvectors(a, b, ev, z, sc_desc, error)
#else
     call e%eigenvectors(a, ev, z, error)
#endif
     call e%timer_stop("e%eigenvectors()")
#endif /* TEST_EIGENVECTORS || defined(TEST_QR_DECOMPOSITION) */

!#if defined(TEST_GENERALIZED_EIGENPROBLEM)
!     ! todo...
!     call e%timer_stop("generalized_eigenproblem")
!#endif

#ifdef TEST_EIGENVALUES
     call e%timer_start("e%eigenvalues()")
     call e%eigenvalues(a, ev, error)
     call e%timer_stop("e%eigenvalues()")
#endif

#if defined(TEST_SOLVE_TRIDIAGONAL)
     call e%timer_start("e%solve_tridiagonal()")
     call e%solve_tridiagonal(d, sd, z, error)
     call e%timer_stop("e%solve_tridiagonal()")
     ev(:) = d(:)
#endif

#if defined(TEST_CHOLESKY)
     call e%timer_start("e%cholesky()")
     call e%cholesky(a, error)
     call e%timer_stop("e%cholesky()")
#endif

#if defined(TEST_HERMITIAN_MULTIPLY)
     call e%timer_start("e%hermitian_multiply()")
     call e%hermitian_multiply('F','F', na, a, b, na_rows, na_cols, c, na_rows, na_cols, error)
     call e%timer_stop("e%hermitian_multiply()")
#endif

     assert_elpa_ok(error)

#ifdef TEST_ALL_KERNELS
     call e%timer_stop(elpa_int_value_to_string(KERNEL_KEY, kernel))
#endif

     if (myid .eq. 0) then
#ifdef TEST_ALL_KERNELS
       call e%print_times(elpa_int_value_to_string(KERNEL_KEY, kernel))
#else /* TEST_ALL_KERNELS */

#if defined(TEST_EIGENVECTORS) || defined(TEST_QR_DECOMPOSITION) || defined(TEST_GENERALIZED_EIGENPROBLEM)
       call e%print_times("e%eigenvectors()")
#endif
#ifdef TEST_EIGENVALUES
       call e%print_times("e%eigenvalues()")
#endif
#ifdef TEST_SOLVE_TRIDIAGONAL
       call e%print_times("e%solve_tridiagonal()")
#endif
#ifdef TEST_CHOLESKY
       call e%print_times("e%cholesky()")
#endif
#ifdef TEST_HERMITIAN_MULTIPLY
       call e%print_times("e%hermitian_multiply()")
#endif
#endif /* TEST_ALL_KERNELS */
     endif

     ! check the results
     call e%timer_start("check_correctness")
#if defined(TEST_EIGENVECTORS) || defined(TEST_QR_DECOMPOSITION) || defined(TEST_GENERALIZED_EIGENPROBLEM)
#ifdef TEST_MATRIX_ANALYTIC
     status = check_correctness_analytic(na, nev, ev, z, nblk, myid, np_rows, np_cols, my_prow, my_pcol, check_all_evals)
#else
!#elif defined(TEST_MATRIX_FRANK)
!     status = check_correctness_evp_numeric_residuals(na, nev, as, z, ev, sc_desc, nblk, myid, np_rows,np_cols, my_prow, my_pcol)
!#elif defined(TEST_MATRIX_RANDOM)
     if (nev .ge. 1) then
       status = check_correctness_evp_numeric_residuals(na, nev, as, z, ev, sc_desc, nblk, myid, np_rows,np_cols, my_prow, my_pcol)
     else
       ! zero eigenvectors and no analytic test => toeplitz
       status = check_correctness_eigenvalues_toeplitz(na, diagonalElement, &
         subdiagonalElement, ev, z, myid)
     endif
     call check_status(status, myid)
!#else
!#error "MATRIX TYPE"
!#endif
#endif
#endif /* defined(TEST_EIGENVECTORS) || defined(TEST_QR_DECOMPOSITION) */

#if defined(TEST_EIGENVALUES) || defined(TEST_SOLVE_TRIDIAGONAL)
     status = check_correctness_eigenvalues_toeplitz(na, diagonalElement, &
         subdiagonalElement, ev, z, myid)
     call check_status(status, myid)

#ifdef TEST_SOLVE_TRIDIAGONAL
     ! check eigenvectors
     status = check_correctness_evp_numeric_residuals(na, nev, as, z, ev, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol)
     call check_status(status, myid)
#endif
#endif

#if defined(TEST_CHOLESKY)
     status = check_correctness_cholesky(na, a, as, na_rows, sc_desc, myid )
     call check_status(status, myid)
#endif

#if defined(TEST_HERMITIAN_MULTIPLY)
     status = check_correctness_hermitian_multiply(na, a, b, c, na_rows, sc_desc, myid )
     call check_status(status, myid)
#endif
     call e%timer_stop("check_correctness")

     if (myid == 0) then
       print *, ""
     endif

#ifdef TEST_ALL_KERNELS
     a(:,:) = as(:,:)
#if defined(TEST_EIGENVALUES) || defined(TEST_SOLVE_TRIDIAGONAL) || defined(TEST_EIGENVECTORS) || defined(TEST_QR_DECOMPOSITION) || defined(TEST_CHOLESKY)
     d = ds
     sd = sds
#endif
   end do ! kernels
#endif

     if (myid == 0) then
       print *, ""
       call e%print_times("check_correctness")
       print *, ""
     endif

   call elpa_deallocate(e)

   deallocate(a)
   deallocate(as)
   deallocate(z)
   deallocate(ev)

#ifdef TEST_HERMITIAN_MULTIPLY
   deallocate(b)
   deallocate(c)
#endif

#if defined(TEST_EIGENVALUES) || defined(TEST_SOLVE_TRIDIAGONAL) || defined(TEST_EIGENVECTORS) || defined(TEST_QR_DECOMPOSITION) || defined(TEST_CHOLESKY)
   deallocate(d, ds)
   deallocate(sd, sds)
   deallocate(ev_analytic)
#endif

#ifdef TEST_GENERALIZED_EIGENPROBLEM
   deallocate(b)
   deallocate(bs)
   ! todo: only temporarily, before we start using random SPD matrix for B
   deallocate(d, ds)
   deallocate(sd, sds)
#endif

#ifdef TEST_ALL_LAYOUTS
   end do ! factors
   end do ! layouts
#endif

   call elpa_uninit()

#ifdef WITH_MPI
   call blacs_gridexit(my_blacs_ctxt)
   call mpi_finalize(mpierr)
#endif

   call exit(status)

   contains

     subroutine check_status(status, myid)
       implicit none
       integer, intent(in) :: status, myid
       integer :: mpierr
       if (status /= 0) then
         if (myid == 0) print *, "Result incorrect!"
#ifdef WITH_MPI
         call mpi_finalize(mpierr)
#endif
         call exit(status)
       endif
     end subroutine

end program
