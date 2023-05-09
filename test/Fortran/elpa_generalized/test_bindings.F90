#include "config-f90.h"

#include "../assert.h"

program test_bindings
   use elpa

   use test_util
   use test_setup_mpi
!   use test_prepare_matrix
   use test_read_input_parameters
   use test_blacs_infrastructure
!   use test_check_correctness
!   use test_analytic
!   use test_scalapack


   implicit none

#include "src/elpa_generated_fortran_interfaces.h"

   ! matrix dimensions
   integer                     :: na, nev, nblk

   ! mpi
   integer                     :: myid, nprocs
   integer                     :: na_cols, na_rows  ! local matrix size
   integer                     :: np_cols, np_rows  ! number of MPI processes per column/row
   integer                     :: my_prow, my_pcol  ! local MPI task position (my_prow, my_pcol) in the grid (0..np_cols -1, 0..np_rows -1)
   integer                     :: mpierr, mpi_comm_rows, mpi_comm_cols
   type(output_t)              :: write_to_file

   ! blacs
   integer                     :: my_blacs_ctxt, sc_desc(9), info, nprow, npcol, i, j, blacs_ok
   character(len=1)            :: layout


   ! The Matrix
   real(kind=C_DOUBLE) , allocatable    :: a(:,:), res(:,:)

   logical                     :: skip_check_correctness

   class(elpa_t), pointer      :: e

   integer                     :: error, status

     call read_input_parameters_traditional(na, nev, nblk, write_to_file, skip_check_correctness)
     call setup_mpi(myid, nprocs)
#ifdef WITH_MPI
     call MPI_BARRIER(MPI_COMM_WORLD, mpierr)
     !call redirect_stdout(myid)
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
                                na_rows, na_cols, sc_desc, my_blacs_ctxt, info, blacs_ok)
   if (blacs_ok .eq. 0) then
     if (myid .eq. 0) then
       print *," Ecountered critical error when setting up blacs. Aborting..."
     endif
     call mpi_finalize(mpierr)
     stop 1
   endif

   allocate(a (na_rows,na_cols))
   allocate(res(na_rows,na_cols))   

   e => elpa_allocate(error)
   assert_elpa_ok(error)

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

   call e%get("mpi_comm_rows",mpi_comm_rows, error)
   assert_elpa_ok(error)
   call e%get("mpi_comm_cols",mpi_comm_cols, error)
   assert_elpa_ok(error)

   a(:,:) = 1.0
   res(:,:) = 0.0

   call test_c_bindings(a, na_rows, na_cols, np_rows, np_cols, my_prow, my_pcol, sc_desc, res, mpi_comm_rows, mpi_comm_cols)

   status = 0
   do i = 1, na_rows
     do j = 1, na_cols
       if(a(i,j) .ne. 1.0) then
         write(*,*) i, j, ": wrong value of A: ", a(i,j), ", should be 1"
         status = 1
       endif
       if(res(i,j) .ne. 3.0) then
         write(*,*) i, j, ": wrong value of res: ", res(i,j), ", should be 3"
         status = 1
       endif
     enddo
   enddo

   call check_status(status, myid)

   call elpa_deallocate(e, error)
   assert_elpa_ok(error)

   deallocate(a)
   deallocate(res)
   call elpa_uninit(error)
   assert_elpa_ok(error)

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
