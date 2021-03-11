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
! Define TEST_AMD_GPU \in [0, 1]
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
#if (TEST_NVIDIA_GPU == 1) || (TEST_AMD_GPU == 1) || (TEST_INTEL_GPU == 1)
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
   use iso_fortran_env

#ifdef HAVE_REDIRECT
   use test_redirect
#endif
   implicit none

   ! matrix dimensions
   TEST_INT_TYPE                     :: na, nev, nblk

   ! mpi
   TEST_INT_TYPE                     :: myid, nprocs
   TEST_INT_TYPE                     :: na_cols, na_rows  ! local matrix size
   TEST_INT_TYPE                     :: np_cols, np_rows  ! number of MPI processes per column/row
   TEST_INT_TYPE                     :: my_prow, my_pcol  ! local MPI task position (my_prow, my_pcol) in the grid (0..np_cols -1, 0..np_rows -1)
   TEST_INT_TYPE                     :: ierr
   TEST_INT_MPI_TYPE                 :: mpierr
   ! blacs
   character(len=1)                  :: layout
   TEST_INT_TYPE                     :: my_blacs_ctxt, sc_desc(9), info, nprow, npcol

   ! The Matrix
   MATRIX_TYPE, allocatable    :: a(:,:), as(:,:)
   ! eigenvectors
   MATRIX_TYPE, allocatable    :: z(:,:)
   ! eigenvalues
   EV_TYPE, allocatable        :: ev(:)

   TEST_INT_TYPE               :: status
   integer(kind=c_int)         :: error_elpa

   type(output_t)              :: write_to_file
   class(elpa_t), pointer      :: e1, e2, e_ptr
   class(elpa_autotune_t), pointer :: tune_state

   TEST_INT_TYPE                     :: iter
   character(len=5)            :: iter_string
   TEST_INT_TYPE                     :: timings, debug, gpu

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

   call set_up_blacsgrid(int(mpi_comm_world,kind=BLAS_KIND), np_rows, np_cols, layout, &
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

   e1 => elpa_allocate(error_elpa)
   !assert_elpa_ok(error_elpa)

   call set_basic_params(e1, na, nev, na_rows, na_cols, my_prow, my_pcol)

   call e1%set("timings",1, error_elpa)
   assert_elpa_ok(error_elpa)

   call e1%set("debug",1, error_elpa)
   assert_elpa_ok(error_elpa)

#if TEST_NVIDIA_GPU == 1 || (TEST_NVIDIA_GPU == 0) && (TEST_AMD_GPU == 0) && (TEST_INTEL_GPU == 0)
   call e1%set("nvidia-gpu", TEST_GPU, error_elpa)
   assert_elpa_ok(error_elpa)
#endif

#if TEST_AMD_GPU == 1
   call e1%set("amd-gpu", TEST_GPU, error_elpa)
   assert_elpa_ok(error_elpa)
#endif

#if TEST_INTEL_GPU == 1
   call e1%set("intel-gpu", TEST_GPU, error_elpa)
   assert_elpa_ok(error_elpa)
#endif
   !call e1%set("max_stored_rows", 15, error_elpa)

   assert_elpa_ok(e1%setup())

   call e1%store_settings("initial_parameters.txt", error_elpa)
   assert_elpa_ok(error_elpa)

#ifdef WITH_MPI
     ! barrier after store settings, file created from one MPI rank only, but loaded everywhere
     call MPI_BARRIER(MPI_COMM_WORLD, mpierr)
#endif

   ! try to load parameters into another object
   e2 => elpa_allocate(error_elpa)
   assert_elpa_ok(error_elpa)

   call set_basic_params(e2, na, nev, na_rows, na_cols, my_prow, my_pcol)
   call e2%load_settings("initial_parameters.txt", error_elpa)
   assert_elpa_ok(error_elpa)

   assert_elpa_ok(e2%setup())

   ! test whether the user setting of e1 are correctly loade to e2
   call e2%get("timings", int(timings,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e2%get("debug", int(debug,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
#if TEST_NVIDIA_GPU == 1 || (TEST_NVIDIA_GPU == 0) && (TEST_AMD_GPU == 0) && (TEST_INTEL_GPU == 0)
   call e2%get("nvidia-gpu", int(gpu,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
#endif
#if TEST_AMD_GPU == 1
   call e2%get("amd-gpu", int(gpu,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
#endif
#if TEST_INTEL_GPU == 1
   call e2%get("intel-gpu", int(gpu,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
#endif

   if ((timings .ne. 1) .or. (debug .ne. 1) .or. (gpu .ne. 0)) then
     print *, "Parameters not stored or loaded correctly. Aborting...", timings, debug, gpu
     stop 1
   endif

   if(myid == 0) print *, "parameters of e1"
   call e1%print_settings(error_elpa)
   assert_elpa_ok(error_elpa)

   if(myid == 0) print *, ""
   if(myid == 0) print *, "parameters of e2"
   call e2%print_settings(error_elpa)
   assert_elpa_ok(error_elpa)

   e_ptr => e2


   tune_state => e_ptr%autotune_setup(ELPA_AUTOTUNE_FAST, AUTOTUNE_DOMAIN, error_elpa)
   assert_elpa_ok(error_elpa)


   iter=0
   do while (e_ptr%autotune_step(tune_state, error_elpa))
     assert_elpa_ok(error_elpa)
 
     iter=iter+1
     write(iter_string,'(I5.5)') iter
     call e_ptr%print_settings(error_elpa)
     assert_elpa_ok(error_elpa)

     call e_ptr%store_settings("saved_parameters_"//trim(iter_string)//".txt", error_elpa)
     assert_elpa_ok(error_elpa)

     call e_ptr%timer_start("eigenvectors: iteration "//trim(iter_string))
     call e_ptr%eigenvectors(a, ev, z, error_elpa)
     assert_elpa_ok(error_elpa)
     call e_ptr%timer_stop("eigenvectors: iteration "//trim(iter_string))

     assert_elpa_ok(error_elpa)
     if (myid .eq. 0) then
       print *, ""
       call e_ptr%print_times("eigenvectors: iteration "//trim(iter_string))
     endif
     status = check_correctness_analytic(na, nev, ev, z, nblk, myid, np_rows, np_cols, my_prow, my_pcol, &
                                         .true., .true., print_times=.false.)
     a(:,:) = as(:,:)
     call e_ptr%autotune_print_state(tune_state, error_elpa)
     assert_elpa_ok(error_elpa)

     call e_ptr%autotune_save_state(tune_state, "saved_state_"//trim(iter_string)//".txt", error_elpa)
     assert_elpa_ok(error_elpa)
#ifdef WITH_MPI
     ! barrier after save state, file created from one MPI rank only, but loaded everywhere
     call MPI_BARRIER(MPI_COMM_WORLD, mpierr)
#endif
     call e_ptr%autotune_load_state(tune_state, "saved_state_"//trim(iter_string)//".txt", error_elpa)
     assert_elpa_ok(error_elpa)

   end do

   ! set and print the autotuned-settings
   call e_ptr%autotune_set_best(tune_state, error_elpa)
   assert_elpa_ok(error_elpa)

   if (myid .eq. 0) then
     print *, "The best combination found by the autotuning:"
     flush(output_unit)
     call e_ptr%autotune_print_best(tune_state, error_elpa)
     assert_elpa_ok(error_elpa)
   endif
   ! de-allocate autotune object
   call elpa_autotune_deallocate(tune_state, error_elpa)
   assert_elpa_ok(error_elpa)

   if (myid .eq. 0) then
     print *, "Running once more time with the best found setting..."
   endif
   call e_ptr%timer_start("eigenvectors: best setting")
   call e_ptr%eigenvectors(a, ev, z, error_elpa)
   assert_elpa_ok(error_elpa)

   call e_ptr%timer_stop("eigenvectors: best setting")
   assert_elpa_ok(error_elpa)
   if (myid .eq. 0) then
     print *, ""
     call e_ptr%print_times("eigenvectors: best setting")
   endif
   status = check_correctness_analytic(na, nev, ev, z, nblk, myid, np_rows, np_cols, my_prow, my_pcol, &
                                       .true., .true., print_times=.false.)

   call elpa_deallocate(e_ptr, error_elpa)
   !assert_elpa_ok(error_elpa)

   deallocate(a)
   deallocate(as)
   deallocate(z)
   deallocate(ev)

   call elpa_uninit(error_elpa)
   !assert_elpa_ok(error_elpa)

#ifdef WITH_MPI
   call blacs_gridexit(my_blacs_ctxt)
   call mpi_finalize(mpierr)
#endif

   call exit(status)

contains
   subroutine set_basic_params(elpa, na, nev, na_rows, na_cols, my_prow, my_pcol)
     implicit none
     class(elpa_t), pointer      :: elpa
     TEST_INT_TYPE, intent(in)   :: na, nev, na_rows, na_cols, my_prow, my_pcol

     call elpa%set("na", int(na,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
     call elpa%set("nev", int(nev,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
     call elpa%set("local_nrows", int(na_rows,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
     call elpa%set("local_ncols", int(na_cols,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
     call elpa%set("nblk", int(nblk,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)

#ifdef WITH_MPI
     call elpa%set("mpi_comm_parent", int(MPI_COMM_WORLD,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
     call elpa%set("process_row", int(my_prow,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
     call elpa%set("process_col", int(my_pcol,kind=c_int), error_elpa)
     assert_elpa_ok(error_elpa)
#endif
   end subroutine

end program
