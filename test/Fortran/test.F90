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

#if !(defined(TEST_SOLVER_1STAGE) ^ defined(TEST_SOLVER_2STAGE))
error: define exactly one of TEST_SOLVER_1STAGE or TEST_SOLVER_2STAGE
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

   implicit none

   ! matrix dimensions
   integer :: na, nev, nblk

   ! mpi
   integer :: myid, nprocs
   integer :: na_cols, na_rows  ! local matrix size
   integer :: np_cols, np_rows  ! number of MPI processes per column/row
   integer :: my_prow, my_pcol  ! local MPI task position (my_prow, my_pcol) in the grid (0..np_cols -1, 0..np_rows -1)
   integer :: mpierr

   ! blacs
   integer :: my_blacs_ctxt, sc_desc(9), info, nprow, npcol, adjusted_na

   ! The Matrix
   MATRIX_TYPE, allocatable :: a(:,:), as(:,:)
   ! eigenvectors
   MATRIX_TYPE, allocatable :: z(:,:)
   ! eigenvalues
   EV_TYPE, allocatable :: ev(:)
#if defined(__EIGENVALUES) || defined(__SOLVE_TRIDIAGONAL)
   EV_TYPE, allocatable :: d(:), sd(:), ev_analytic(:), ds(:), sds(:)
   EV_TYPE              :: diagonalELement, subdiagonalElement, tmp, maxerr
#ifdef TEST_DOUBLE
   EV_TYPE, parameter   :: pi = 3.141592653589793238462643383279_rk8
#else
   EV_TYPE, parameter   :: pi = 3.1415926535897932_rk4
#endif
   integer              :: loctmp ,rowLocal, colLocal, j,ii
#endif


   integer :: error, status

   type(output_t) :: write_to_file
   class(elpa_t), pointer :: e
#ifdef TEST_ALL_KERNELS
   integer :: i
#endif
   integer :: kernel

#if defined(TEST_COMPLEX) && defined(__SOLVE_TRIDIAGONAL)
   stop 77
#endif
   call read_input_parameters_traditional(na, nev, nblk, write_to_file)

   call setup_mpi(myid, nprocs)

   do np_cols = NINT(SQRT(REAL(nprocs))),2,-1
      if(mod(nprocs,np_cols) == 0 ) exit
   enddo

   np_rows = nprocs/np_cols

#ifdef TEST_MATRIX_ANALYTIC
   adjusted_na = 1
   do while (adjusted_na < na)
     adjusted_na = adjusted_na * 2
   end do
   if (adjusted_na > na) then
     na = adjusted_na
     if(myid == 0) then
       print *, 'At the moment, analytic test works for sizes of matrix of powers of two only. na changed to ', na
     end if
   end if
#endif

   if (myid == 0) then
     print '((a,i0))', 'Matrix size: ', na
     print '((a,i0))', 'Num eigenvectors: ', nev
     print '((a,i0))', 'Blocksize: ', nblk
     print '((a,i0))', 'Num MPI proc: ', nprocs
     print '(3(a,i0))','Number of processor rows=',np_rows,', cols=',np_cols,', total=',nprocs
     print *,''
   endif

   call set_up_blacsgrid(mpi_comm_world, my_blacs_ctxt, np_rows, np_cols, &
                         nprow, npcol, my_prow, my_pcol)

   call set_up_blacs_descriptor(na, nblk, my_prow, my_pcol, np_rows, np_cols, &
                                na_rows, na_cols, sc_desc, my_blacs_ctxt, info)

   allocate(a (na_rows,na_cols))
#ifdef TEST_MATRIX_RANDOM
   allocate(as(na_rows,na_cols))
#endif
   allocate(z (na_rows,na_cols))
   allocate(ev(na))

#if defined(__EIGENVALUES) || defined(__SOLVE_TRIDIAGONAL)
   allocate(d (na), ds(na))
   allocate(sd (na), sds(na))
   allocate(ev_analytic(na))
#endif

   a(:,:) = 0.0
   z(:,:) = 0.0
   ev(:) = 0.0

#ifdef __EIGENVECTORS
#ifdef TEST_MATRIX_ANALYTIC
   call prepare_matrix_analytic(na, a, nblk, myid, np_rows, np_cols, my_prow, my_pcol)
#else
   call prepare_matrix(na, myid, sc_desc, a, z, as)
#endif
#endif

#if defined(__EIGENVALUES) || defined(__SOLVE_TRIDIAGONAL)
   ! set up simple toeplitz matrix
#ifdef TEST_DOUBLE
   diagonalElement = 0.45_rk8
   subdiagonalElement =  0.78_rk8
#else
   diagonalElement = 0.45_rk4
   subdiagonalElement =  0.78_rk4
#endif

   d(:) = diagonalElement
   sd(:) = subdiagonalElement

   ! set up the diagonal and subdiagonals (for general solver test)
   do ii=1, na ! for diagonal elements
     if (map_global_array_index_to_local_index(ii, ii, rowLocal, colLocal, nblk, np_rows, np_cols, my_prow, my_pcol)) then
       a(rowLocal,colLocal) = diagonalElement
     endif
   enddo
   do ii=1, na-1
     if (map_global_array_index_to_local_index(ii, ii+1, rowLocal, colLocal, nblk, np_rows, np_cols, my_prow, my_pcol)) then
       a(rowLocal,colLocal) = subdiagonalElement
     endif
   enddo

   do ii=2, na
     if (map_global_array_index_to_local_index(ii, ii-1, rowLocal, colLocal, nblk, np_rows, np_cols, my_prow, my_pcol)) then
       a(rowLocal,colLocal) = subdiagonalElement
     endif
   enddo
   ds = d
   sds = sd
   as = a
#endif

   if (elpa_init(CURRENT_API_VERSION) /= ELPA_OK) then
     print *, "ELPA API version not supported"
     stop 1
   endif

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

   assert_elpa_ok(e%setup())

#ifdef TEST_SOLVER_1STAGE
   call e%set("solver", ELPA_SOLVER_1STAGE)
#else
   call e%set("solver", ELPA_SOLVER_2STAGE)
#endif
   assert_elpa_ok(error)

   call e%set("gpu", TEST_GPU, error)
   assert_elpa_ok(error)

#ifdef TEST_ALL_KERNELS
   do i = 0, elpa_option_cardinality(KERNEL_KEY)
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
#endif
     if (myid == 0) print *, elpa_int_value_to_string(KERNEL_KEY, kernel), " kernel"

     call e%timer_start(elpa_int_value_to_string(KERNEL_KEY, kernel))
#else /* ALL_KERNELS */

#ifdef __EIGENVECTORS
     call e%timer_start("e%eigenvectors()")
#endif
#ifdef __EIGENVALUES
     call e%timer_start("e%eigenvalues()")
#endif
#ifdef __SOLVE_TRIDIAGONAL
     call e%timer_start("e%solve_tridiagonal()")
#endif
#endif

     ! The actual solve step
#ifdef __EIGENVECTORS
     call e%eigenvectors(a, ev, z, error)
#endif
#ifdef __EIGENVALUES
     call e%eigenvalues(a, ev, error)
#endif
#if defined(__SOLVE_TRIDIAGONAL) && !defined(TEST_COMPLEX)
     call e%solve_tridiagonal(d, sd, z, error)
     ev(:) = d(:)
#endif

     assert_elpa_ok(error)

#ifdef TEST_SOLVER_2STAGE
     call e%timer_stop(elpa_int_value_to_string(KERNEL_KEY, kernel))
#else
#ifdef __EIGENVECTORS
     call e%timer_stop("e%eigenvectors()")
#endif
#ifdef __EIGENVALUES
     call e%timer_stop("e%eigenvalues()")
#endif
#ifdef __SOLVE_TRIDIAGONAL
     call e%timer_stop("e%solve_tridiagonal()")
#endif
#endif

     if (myid .eq. 0) then
#ifdef TEST_SOLVER_2STAGE
       call e%print_times("e%eigenvectors()")
       call e%print_times(elpa_int_value_to_string(KERNEL_KEY, kernel))
#else
#ifdef __EIGENVECTORS
       call e%print_times("e%eigenvectors()")
#endif
#ifdef __EIGENVALUES
       call e%print_times("e%eigenvalues()")
#endif
#ifdef __SOLVE_TRIDIAGONAL
     call e%print_times("e%solve_tridiagonal()")
#endif
#endif
     endif

#ifdef __EIGENVECTORS
#ifdef TEST_MATRIX_ANALYTIC
     status = check_correctness_analytic(na, nev, ev, z, nblk, myid, np_rows, np_cols, my_prow, my_pcol)
#else
     status = check_correctness(na, nev, as, z, ev, sc_desc, myid)
#endif
     if (status /= 0) then
       if (myid == 0) print *, "Result incorrect!"
       call exit(status)
     endif
     if (myid == 0) print *, ""
#endif
#if defined(__EIGENVALUES) || defined(__SOLVE_TRIDIAGONAL)
     status = 0
     ! analytic solution
     do ii=1, na
#ifdef TEST_DOUBLE
       ev_analytic(ii) = diagonalElement + 2.0 * subdiagonalElement *cos( pi*real(ii,kind=rk8)/ real(na+1,kind=rk8) )
#else
       ev_analytic(ii) = diagonalElement + 2.0 * subdiagonalElement *cos( pi*real(ii,kind=rk4)/ real(na+1,kind=rk4) )
#endif
     enddo

     ! sort analytic solution:

     ! this hack is neither elegant, nor optimized: for huge matrixes it might be expensive
     ! a proper sorting algorithmus might be implemented here

     tmp    = minval(ev_analytic)
     loctmp = minloc(ev_analytic, 1)

     ev_analytic(loctmp) = ev_analytic(1)
     ev_analytic(1) = tmp

     do ii=2, na
       tmp = ev_analytic(ii)
       do j= ii, na
         if (ev_analytic(j) .lt. tmp) then
           tmp    = ev_analytic(j)
           loctmp = j
         endif
       enddo
       ev_analytic(loctmp) = ev_analytic(ii)
       ev_analytic(ii) = tmp
     enddo

     ! compute a simple error max of eigenvalues
     maxerr = 0.0
     maxerr = maxval( (ev(:) - ev_analytic(:))/ev_analytic(:) , 1)

#ifdef TEST_DOUBLE
     if (maxerr .gt. 8.e-13) then
#else
     if (maxerr .gt. 8.e-4) then
#endif
       status = 1
       if (myid .eq. 0) then
         print *,"Eigenvalues differ from analytic solution: maxerr = ",maxerr
       endif
     endif

     if (status /= 0) then
       call exit(status)
     endif
#ifdef __SOLVE_TRIDIAGONAL
     ! check eigenvectors
     status = check_correctness(na, nev, as, z, ev, sc_desc, myid)
     if (status /= 0) then
       if (myid == 0) print *, "Result incorrect!"
       call exit(status)
     endif
     if (myid == 0) print *, ""
#endif

#endif

#ifdef TEST_ALL_KERNELS
#ifdef TEST_MATRIX_ANALYTIC
     call prepare_matrix_analytic(na, a, nblk, myid, np_rows, np_cols, my_prow, my_pcol)
#else
     a(:,:) = as(:,:)
#endif
#if defined(__EIGENVALUES) || defined(__SOLVE_TRIDIAGONAL)
     d = ds
     sd = sds
#endif
   end do
#endif

   call elpa_deallocate(e)
   call elpa_uninit()

   deallocate(a)
#ifdef TEST_MATRIX_RANDOM
   deallocate(as)
#endif
   deallocate(z)
   deallocate(ev)

#ifdef __EIGENVALUES
   deallocate(d, ds)
   deallocate(sd, sds)
   deallocate(ev_analytic)
#endif

#ifdef WITH_MPI
   call blacs_gridexit(my_blacs_ctxt)
   call mpi_finalize(mpierr)
#endif

   call exit(status)

!#if defined(__EIGENVALUES) || defined(__SOLVE_TRIDIAGONAL)
!   contains
!
!     !Processor col for global col number
!     pure function pcol(global_col, nblk, np_cols) result(local_col)
!       implicit none
!       integer(kind=c_int), intent(in) :: global_col, nblk, np_cols
!       integer(kind=c_int)             :: local_col
!       local_col = MOD((global_col-1)/nblk,np_cols)
!     end function
!
!     !Processor row for global row number
!     pure function prow(global_row, nblk, np_rows) result(local_row)
!       implicit none
!       integer(kind=c_int), intent(in) :: global_row, nblk, np_rows
!       integer(kind=c_int)             :: local_row
!       local_row = MOD((global_row-1)/nblk,np_rows)
!     end function
!
!     function map_global_array_index_to_local_index(iGLobal, jGlobal, iLocal, jLocal , nblk, np_rows, np_cols, my_prow, my_pcol) &
!       result(possible)
!       implicit none
!
!       integer(kind=c_int)              :: pi, pj, li, lj, xi, xj
!       integer(kind=c_int), intent(in)  :: iGlobal, jGlobal, nblk, np_rows, np_cols, my_prow, my_pcol
!       integer(kind=c_int), intent(out) :: iLocal, jLocal
!       logical                       :: possible
!
!       possible = .true.
!       iLocal = 0
!       jLocal = 0
!
!       pi = prow(iGlobal, nblk, np_rows)
!
!       if (my_prow .ne. pi) then
!         possible = .false.
!         return
!       endif
!
!       pj = pcol(jGlobal, nblk, np_cols)
!
!       if (my_pcol .ne. pj) then
!         possible = .false.
!         return
!       endif
!       li = (iGlobal-1)/(np_rows*nblk) ! block number for rows
!       lj = (jGlobal-1)/(np_cols*nblk) ! block number for columns
!
!       xi = mod( (iGlobal-1),nblk)+1   ! offset in block li
!       xj = mod( (jGlobal-1),nblk)+1   ! offset in block lj
!
!       iLocal = li * nblk + xi
!       jLocal = lj * nblk + xj
!
!     end function
!#endif
end program
