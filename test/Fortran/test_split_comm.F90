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
   integer                     :: num_groups, group_size, color, key

   ! mpi
   integer                     :: myid, nprocs
   integer                     :: na_cols, na_rows  ! local matrix size
   integer                     :: np_cols, np_rows  ! number of MPI processes per column/row
   integer                     :: my_prow, my_pcol  ! local MPI task position (my_prow, my_pcol) in the grid (0..np_cols -1, 0..np_rows -1)
   integer                     :: mpierr, ierr
   integer                     :: mpi_sub_comm
   integer                     :: myid_sub, nprocs_sub

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
   class(elpa_t), pointer      :: e

   integer                     :: iter
   character(len=5)            :: iter_string

   status = 0
#ifdef WITH_MPI

   call read_input_parameters(na, nev, nblk, write_to_file)
   !call setup_mpi(myid, nprocs)
   call mpi_init(mpierr)
   call mpi_comm_rank(mpi_comm_world,myid,mpierr)
   call mpi_comm_size(mpi_comm_world,nprocs,mpierr)

   if((mod(nprocs, 4) == 0) .and. (nprocs > 4)) then
     num_groups = 4
   else if(mod(nprocs, 3) == 0) then
     num_groups = 3
   else if(mod(nprocs, 2) == 0) then
     num_groups = 2
   else
     num_groups = 1
   endif

   group_size = nprocs / num_groups

   if(num_groups * group_size .ne. nprocs) then 
     print *, "Something went wrong before splitting the communicator"
     stop 1
   else
     if(myid == 0) then
       print '((a,i0,a,i0))', "The test will split the global communicator into ", num_groups, " groups of size ", group_size
     endif
   endif

   ! each group of processors will have the same color
   color = mod(myid, num_groups)
   ! this will determine the myid in each group
   key = myid/num_groups
   !split the communicator
   call mpi_comm_split(mpi_comm_world, color, key, mpi_sub_comm, mpierr)
   if(mpierr .ne. MPI_SUCCESS) then 
     print *, "communicator splitting not successfull", mpierr
     stop 1
   endif

   call mpi_comm_rank(mpi_sub_comm, myid_sub, mpierr)
   call mpi_comm_size(mpi_sub_comm, nprocs_sub, mpierr)

   !print *, "glob ", myid, nprocs, ", loc ", myid_sub, nprocs_sub, ", color ", color, ", key ", key

   if((mpierr .ne. MPI_SUCCESS) .or. (nprocs_sub .ne. group_size) .or. (myid_sub >= group_size)) then
     print *, "something wrong with the sub communicators"
     stop 1
   endif


#ifdef HAVE_REDIRECT
   call MPI_BARRIER(MPI_COMM_WORLD, mpierr)
   call redirect_stdout(myid)
#endif

   if (elpa_init(CURRENT_API_VERSION) /= ELPA_OK) then
     print *, "ELPA API version not supported"
     stop 1
   endif

   layout = 'C'
   do np_cols = NINT(SQRT(REAL(nprocs_sub))),2,-1
      if(mod(nprocs_sub,np_cols) == 0 ) exit
   enddo
   np_rows = nprocs_sub/np_cols
   assert(nprocs_sub == np_rows * np_cols)
   assert(nprocs == np_rows * np_cols * num_groups)

   if (myid == 0) then
     print '((a,i0))', 'Matrix size: ', na
     print '((a,i0))', 'Num eigenvectors: ', nev
     print '((a,i0))', 'Blocksize: ', nblk
     print '(a)',      'Process layout: ' // layout
     print *,''
   endif
   if (myid_sub == 0) then
     print '(4(a,i0))','GROUP ', color, ': Number of processor rows=',np_rows,', cols=',np_cols,', total=',nprocs_sub
   endif

   ! USING the subcommunicator
   call set_up_blacsgrid(mpi_sub_comm, np_rows, np_cols, layout, &
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

   !call prepare_matrix_analytic(na, a, nblk, myid_sub, np_rows, np_cols, my_prow, my_pcol, print_times=.false.)
   call prepare_matrix_random(na, myid_sub, sc_desc, a, z, as)
   as(:,:) = a(:,:)

   e => elpa_allocate()
   call set_basic_params(e, na, nev, na_rows, na_cols, mpi_sub_comm, my_prow, my_pcol)

   call e%set("timings",1, error)

   call e%set("debug",1)
   call e%set("gpu", 0)
   !call e%set("max_stored_rows", 15, error)

   assert_elpa_ok(e%setup())



!   if(myid == 0) print *, "parameters of e"
!   call e%print_all_parameters()
!   if(myid == 0) print *, ""


   call e%timer_start("eigenvectors")
   call e%eigenvectors(a, ev, z, error)
   call e%timer_stop("eigenvectors")

   assert_elpa_ok(error)

   !status = check_correctness_analytic(na, nev, ev, z, nblk, myid_sub, np_rows, np_cols, my_prow, my_pcol, &
    !                   .true., .true., print_times=.false.)
   status = check_correctness_evp_numeric_residuals(na, nev, as, z, ev, sc_desc, nblk, myid_sub, &
                    np_rows,np_cols, my_prow, my_pcol)
   if (status /= 0) &
     print *, "processor ", myid, ": Result incorrect for processor group ", color

   if (myid .eq. 0) then
     print *, "Showing times of one goup only"
     call e%print_times("eigenvectors")
   endif

   call elpa_deallocate(e)

   deallocate(a)
   deallocate(as)
   deallocate(z)
   deallocate(ev)

   call elpa_uninit()

   call blacs_gridexit(my_blacs_ctxt)
   call mpi_finalize(mpierr)

#endif
   call exit(status)

contains
   subroutine set_basic_params(elpa, na, nev, na_rows, na_cols, communicator, my_prow, my_pcol)
     implicit none
     class(elpa_t), pointer      :: elpa
     integer, intent(in)         :: na, nev, na_rows, na_cols, my_prow, my_pcol, communicator

#ifdef WITH_MPI
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

     call elpa%set("mpi_comm_parent", communicator, error)
     assert_elpa_ok(error)
     call elpa%set("process_row", my_prow, error)
     assert_elpa_ok(error)
     call elpa%set("process_col", my_pcol, error)
     assert_elpa_ok(error)
#endif
   end subroutine

end program