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
#  define AUTOTUNE_DOMAIN ELPA_AUTOTUNE_DOMAIN_REAL
#else
#  define AUTOTUNE_DOMAIN ELPA_AUTOTUNE_DOMAIN_COMPLEX
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
   use iso_fortran_env

#ifdef HAVE_REDIRECT
   use test_redirect
#endif
   implicit none

   ! matrix dimensions
   integer                     :: na, nev, nblk

   ! mpi
   integer                     :: myid, nprocs
   integer                     :: na_cols, na_rows  ! local matrix size
   integer                     :: np_cols, np_rows  ! number of MPI processes per column/row
   integer                     :: my_prow, my_pcol  ! local MPI task position (my_prow, my_pcol) in the grid (0..np_cols -1, 0..np_rows -1)
   integer                     :: mpierr, ierr

   ! blacs
   character(len=1)            :: layout
   integer                     :: my_blacs_ctxt, sc_desc(9), info, nprow, npcol

   ! The Matrix
   MATRIX_TYPE, allocatable    :: a(:,:), as(:,:)
   ! eigenvectors
   MATRIX_TYPE, allocatable    :: z(:,:)
   ! eigenvalues
   EV_TYPE, allocatable        :: ev(:)

   integer                     :: error, status

   type(output_t)              :: write_to_file
   class(elpa_t), pointer      :: e1, e2, e_ptr
   class(elpa_autotune_t), pointer :: tune_state

   integer                     :: iter
   character(len=5)            :: iter_string
   integer                     :: timings, debug, gpu

   call read_input_parameters(na, nev, nblk, write_to_file)
   call setup_mpi(myid, nprocs)
#ifdef HAVE_REDIRECT
#ifdef WITH_MPI
   call MPI_BARRIER(MPI_COMM_WORLD, mpierr)
   call redirect_stdout(myid)
#endif
#endif

   if (elpa_init(CURRENT_API_VERSION) /= ELPA_OK) then
     print *, "ELPA API version not supported"
     stop 1
   endif

   layout = 'C'
   do np_cols = NINT(SQRT(REAL(nprocs))),2,-1
      if(mod(nprocs,np_cols) == 0 ) exit
   enddo
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

   a(:,:) = 0.0
   z(:,:) = 0.0
   ev(:) = 0.0

   call prepare_matrix_analytic(na, a, nblk, myid, np_rows, np_cols, my_prow, my_pcol, print_times=.false.)
   as(:,:) = a(:,:)

   e1 => elpa_allocate(error)
   !assert_elpa_ok(error)

   call set_basic_params(e1, na, nev, na_rows, na_cols, my_prow, my_pcol)

   call e1%set("timings",1, error)
   assert_elpa_ok(error)

   call e1%set("debug",1, error)
   assert_elpa_ok(error)

   call e1%set("gpu", 0, error)
   assert_elpa_ok(error)
   !call e1%set("max_stored_rows", 15, error)

   assert_elpa_ok(e1%setup())

   call e1%store_settings("initial_parameters.txt", error)
   assert_elpa_ok(error)

#ifdef WITH_MPI
     ! barrier after store settings, file created from one MPI rank only, but loaded everywhere
     call MPI_BARRIER(MPI_COMM_WORLD, ierr)
#endif

   ! try to load parameters into another object
   e2 => elpa_allocate(error)
   assert_elpa_ok(error)

   call set_basic_params(e2, na, nev, na_rows, na_cols, my_prow, my_pcol)
   call e2%load_settings("initial_parameters.txt", error)
   assert_elpa_ok(error)

   assert_elpa_ok(e2%setup())

   ! test whether the user setting of e1 are correctly loade to e2
   call e2%get("timings", timings, error)
   assert_elpa_ok(error)
   call e2%get("debug", debug, error)
   assert_elpa_ok(error)
   call e2%get("gpu", gpu, error)
   assert_elpa_ok(error)

   if ((timings .ne. 1) .or. (debug .ne. 1) .or. (gpu .ne. 0)) then
     print *, "Parameters not stored or loaded correctly. Aborting...", timings, debug, gpu
     stop 1
   endif

   if(myid == 0) print *, "parameters of e1"
   call e1%print_settings(error)
   assert_elpa_ok(error)

   if(myid == 0) print *, ""
   if(myid == 0) print *, "parameters of e2"
   call e2%print_settings(error)
   assert_elpa_ok(error)

   e_ptr => e2


   tune_state => e_ptr%autotune_setup(ELPA_AUTOTUNE_FAST, AUTOTUNE_DOMAIN, error)
   assert_elpa_ok(error)


   iter=0
   do while (e_ptr%autotune_step(tune_state, error))
     assert_elpa_ok(error)
 
     iter=iter+1
     write(iter_string,'(I5.5)') iter
     call e_ptr%print_settings(error)
     assert_elpa_ok(error)

     call e_ptr%store_settings("saved_parameters_"//trim(iter_string)//".txt", error)
     assert_elpa_ok(error)

     call e_ptr%timer_start("eigenvectors: iteration "//trim(iter_string))
     call e_ptr%eigenvectors(a, ev, z, error)
     assert_elpa_ok(error)
     call e_ptr%timer_stop("eigenvectors: iteration "//trim(iter_string))

     assert_elpa_ok(error)
     if (myid .eq. 0) then
       print *, ""
       call e_ptr%print_times("eigenvectors: iteration "//trim(iter_string))
     endif
     status = check_correctness_analytic(na, nev, ev, z, nblk, myid, np_rows, np_cols, my_prow, my_pcol, &
                                         .true., .true., print_times=.false.)
     a(:,:) = as(:,:)
     call e_ptr%autotune_print_state(tune_state, error)
     assert_elpa_ok(error)

     call e_ptr%autotune_save_state(tune_state, "saved_state_"//trim(iter_string)//".txt", error)
     assert_elpa_ok(error)
#ifdef WITH_MPI
     ! barrier after save state, file created from one MPI rank only, but loaded everywhere
     call MPI_BARRIER(MPI_COMM_WORLD, ierr)
#endif
     call e_ptr%autotune_load_state(tune_state, "saved_state_"//trim(iter_string)//".txt", error)
     assert_elpa_ok(error)

   end do

   ! set and print the autotuned-settings
   call e_ptr%autotune_set_best(tune_state, error)
   assert_elpa_ok(error)

   if (myid .eq. 0) then
     print *, "The best combination found by the autotuning:"
     flush(output_unit)
     call e_ptr%autotune_print_best(tune_state, error)
     assert_elpa_ok(error)
   endif
   ! de-allocate autotune object
   call elpa_autotune_deallocate(tune_state, error)
   assert_elpa_ok(error)

   if (myid .eq. 0) then
     print *, "Running once more time with the best found setting..."
   endif
   call e_ptr%timer_start("eigenvectors: best setting")
   call e_ptr%eigenvectors(a, ev, z, error)
   assert_elpa_ok(error)

   call e_ptr%timer_stop("eigenvectors: best setting")
   assert_elpa_ok(error)
   if (myid .eq. 0) then
     print *, ""
     call e_ptr%print_times("eigenvectors: best setting")
   endif
   status = check_correctness_analytic(na, nev, ev, z, nblk, myid, np_rows, np_cols, my_prow, my_pcol, &
                                       .true., .true., print_times=.false.)

   call elpa_deallocate(e_ptr, error)
   !assert_elpa_ok(error)

   deallocate(a)
   deallocate(as)
   deallocate(z)
   deallocate(ev)

   call elpa_uninit(error)
   !assert_elpa_ok(error)

#ifdef WITH_MPI
   call blacs_gridexit(my_blacs_ctxt)
   call mpi_finalize(mpierr)
#endif

   call exit(status)

contains
   subroutine set_basic_params(elpa, na, nev, na_rows, na_cols, my_prow, my_pcol)
     implicit none
     class(elpa_t), pointer      :: elpa
     integer, intent(in)         :: na, nev, na_rows, na_cols, my_prow, my_pcol

     call elpa%set("na", na, error)
     assert_elpa_ok(error)
     call elpa%set("nev", nev, error)
     assert_elpa_ok(error)
     call elpa%set("local_nrows", na_rows, error)
     assert_elpa_ok(error)
     call elpa%set("local_ncols", na_cols, error)
     assert_elpa_ok(error)
     call elpa%set("nblk", nblk, error)
     assert_elpa_ok(error)

#ifdef WITH_MPI
     call elpa%set("mpi_comm_parent", MPI_COMM_WORLD, error)
     assert_elpa_ok(error)
     call elpa%set("process_row", my_prow, error)
     assert_elpa_ok(error)
     call elpa%set("process_col", my_pcol, error)
     assert_elpa_ok(error)
#endif
   end subroutine

end program
