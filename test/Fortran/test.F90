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
   integer :: my_blacs_ctxt, sc_desc(9), info, nprow, npcol 

   ! The Matrix
   MATRIX_TYPE, allocatable :: a(:,:), as(:,:)
   ! eigenvectors
   MATRIX_TYPE, allocatable :: z(:,:)
   ! eigenvalues
   EV_TYPE, allocatable :: ev(:)
#if defined(TEST_EIGENVALUES) || defined(TEST_SOLVE_TRIDIAGONAL)
   EV_TYPE, allocatable :: d(:), sd(:), ev_analytic(:), ds(:), sds(:)
   EV_TYPE              :: diagonalELement, subdiagonalElement
#endif


   integer :: error, status

   type(output_t) :: write_to_file
   class(elpa_t), pointer :: e
#ifdef TEST_ALL_KERNELS
   integer :: i
#endif
#ifdef TEST_ALL_LAYOUTS
   character(len=1), parameter :: layouts(2) = [ 'C', 'R' ]
   integer :: i_layout
#endif
   integer :: kernel
   character(len=1) :: layout

   call read_input_parameters_traditional(na, nev, nblk, write_to_file)
   call setup_mpi(myid, nprocs)

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

   call set_up_blacsgrid(mpi_comm_world, np_rows, np_cols, layout, &
                         my_blacs_ctxt, my_prow, my_pcol)

   call set_up_blacs_descriptor(na, nblk, my_prow, my_pcol, np_rows, np_cols, &
                                na_rows, na_cols, sc_desc, my_blacs_ctxt, info)

   allocate(a (na_rows,na_cols))
   allocate(as(na_rows,na_cols))
   allocate(z (na_rows,na_cols))
   allocate(ev(na))

#if defined(TEST_EIGENVALUES) || defined(TEST_SOLVE_TRIDIAGONAL)
   allocate(d (na), ds(na))
   allocate(sd (na), sds(na))
   allocate(ev_analytic(na))
#endif

   a(:,:) = 0.0
   z(:,:) = 0.0
   ev(:) = 0.0

#ifdef TEST_EIGENVECTORS
#ifdef TEST_MATRIX_ANALYTIC
   call prepare_matrix_analytic(na, a, nblk, myid, np_rows, np_cols, my_prow, my_pcol)
   as(:,:) = a
#else
   call prepare_matrix(na, myid, sc_desc, a, z, as)
#endif
#endif

#if defined(TEST_EIGENVALUES) || defined(TEST_SOLVE_TRIDIAGONAL)

#ifdef TEST_SINGLE
   diagonalElement = 0.45_c_float
   subdiagonalElement =  0.78_c_float
#else
   diagonalElement = 0.45_c_double
   subdiagonalElement =  0.78_c_double
#endif
   call prepare_toeplitz_matrix(na, diagonalElement, subdiagonalElement, &
                                d, sd, ds, sds, a, as, nblk, np_rows, &
                                np_cols, my_prow, my_pcol)
#endif

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

   if (myid == 0) print *, ""

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

     ! The actual solve step
#ifdef TEST_EIGENVECTORS
     call e%timer_start("e%eigenvectors()")
     call e%eigenvectors(a, ev, z, error)
     call e%timer_stop("e%eigenvectors()")
#endif

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

     assert_elpa_ok(error)

#ifdef TEST_ALL_KERNELS
     call e%timer_stop(elpa_int_value_to_string(KERNEL_KEY, kernel))
#endif

     if (myid .eq. 0) then
#ifdef TEST_ALL_KERNELS
       call e%print_times(elpa_int_value_to_string(KERNEL_KEY, kernel))
#else
#ifdef TEST_EIGENVECTORS
       call e%print_times("e%eigenvectors()")
#endif
#ifdef TEST_EIGENVALUES
       call e%print_times("e%eigenvalues()")
#endif
#ifdef TEST_SOLVE_TRIDIAGONAL
       call e%print_times("e%solve_tridiagonal()")
#endif
#endif
     endif

#ifdef TEST_EIGENVECTORS
#ifdef TEST_MATRIX_ANALYTIC
     status = check_correctness_analytic(na, nev, ev, z, nblk, myid, np_rows, np_cols, my_prow, my_pcol)
#else
     status = check_correctness(na, nev, as, z, ev, sc_desc, nblk, myid, np_rows,np_cols, my_prow, my_pcol)
#endif
     call check_status(status, myid)
#endif

#if defined(TEST_EIGENVALUES) || defined(TEST_SOLVE_TRIDIAGONAL)
     status = check_correctness_eigenvalues_toeplitz(na, diagonalElement, &
         subdiagonalElement, ev, z, myid)
     call check_status(status, myid)

#ifdef TEST_SOLVE_TRIDIAGONAL
     ! check eigenvectors
     status = check_correctness(na, nev, as, z, ev, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol)
     call check_status(status, myid)
#endif
#endif

     if (myid == 0) then
       print *, ""
     endif

#ifdef TEST_ALL_KERNELS
     a(:,:) = as(:,:)
#if defined(TEST_EIGENVALUES) || defined(TEST_SOLVE_TRIDIAGONAL)
     d = ds
     sd = sds
#endif
   end do ! kernels
#endif

   call elpa_deallocate(e)

   deallocate(a)
   deallocate(as)
   deallocate(z)
   deallocate(ev)

#if defined(TEST_EIGENVALUES) || defined(TEST_SOLVE_TRIDIAGONAL)
   deallocate(d, ds)
   deallocate(sd, sds)
   deallocate(ev_analytic)
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
