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

#include "assert.h"

program test_interface
   use precision
   use mod_setup_mpi
   use elpa_mpi
   use elpa
   use mod_prepare_matrix
   use mod_read_input_parameters
   use mod_blacs_infrastructure
   use mod_check_correctness
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif

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
   complex(kind=C_DOUBLE_COMPLEX), allocatable :: a(:,:), as(:,:)
   ! eigenvectors
   complex(kind=C_DOUBLE_COMPLEX), allocatable :: z(:,:)
   ! eigenvalues
   real(kind=C_DOUBLE), allocatable :: ev(:)

   integer :: success, status

   integer(kind=c_int) :: solver
   integer(kind=c_int) :: qr

   type(output_t) :: write_to_file
   class(elpa_t), pointer :: e

   call read_input_parameters_traditional(na, nev, nblk, write_to_file)
   call setup_mpi(myid, nprocs)

   status = 0

#ifdef HAVE_DETAILED_TIMINGS

   ! initialise the timing functionality

#ifdef HAVE_LIBPAPI
   call timer%measure_flops(.true.)
#endif

   call timer%measure_allocated_memory(.true.)
   call timer%measure_virtual_memory(.true.)
   call timer%measure_max_allocated_memory(.true.)

   call timer%set_print_options(&
#ifdef HAVE_LIBPAPI
                print_flop_count=.true., &
                print_flop_rate=.true., &
#endif
                print_allocated_memory = .true. , &
                print_virtual_memory=.true., &
                print_max_allocated_memory=.true.)


  call timer%enable()

  call timer%start("program")
#endif

   do np_cols = NINT(SQRT(REAL(nprocs))),2,-1
      if(mod(nprocs,np_cols) == 0 ) exit
   enddo

   np_rows = nprocs/np_cols

   my_prow = mod(myid, np_cols)
   my_pcol = myid / np_cols

   call set_up_blacsgrid(mpi_comm_world, my_blacs_ctxt, np_rows, np_cols, &
                         nprow, npcol, my_prow, my_pcol)

   call set_up_blacs_descriptor(na, nblk, my_prow, my_pcol, np_rows, np_cols, &
                                na_rows, na_cols, sc_desc, my_blacs_ctxt, info)
#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("set up matrix")
#endif

   allocate(a (na_rows,na_cols), as(na_rows,na_cols))
   allocate(z (na_rows,na_cols))
   allocate(ev(na))

   a(:,:) = 0.0
   z(:,:) = 0.0
   ev(:) = 0.0

   call prepare_matrix(na, myid, sc_desc, a, z, as)
#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("set up matrix")
#endif
   if (elpa_init(CURRENT_API_VERSION) /= ELPA_OK) then
     print *, "ELPA API version not supported"
     stop 1
   endif
#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("prepare_elpa")
#endif

   e => elpa_allocate()

   call e%set("na", na, success)
   assert_elpa_ok(success)
   call e%set("nev", nev, success)
   assert_elpa_ok(success)
   call e%set("local_nrows", na_rows, success)
   assert_elpa_ok(success)
   call e%set("local_ncols", na_cols, success)
   assert_elpa_ok(success)
   call e%set("nblk", nblk, success)
   assert_elpa_ok(success)
   call e%set("mpi_comm_parent", MPI_COMM_WORLD, success)
   assert_elpa_ok(success)
   call e%set("process_row", my_prow, success)
   assert_elpa_ok(success)
   call e%set("process_col", my_pcol, success)
   assert_elpa_ok(success)

   assert_elpa_ok(e%setup())

   call e%set("solver", ELPA_SOLVER_1STAGE, success)
   assert_elpa_ok(success)

   call e%set("gpu", 1, success)
   assert_elpa_ok(success)
#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("prepare_elpa")
#endif
#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("solve")
#endif

   call e%solve(a, ev, z, success)
   assert_elpa_ok(success)
#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("solve")
#endif

   call elpa_deallocate(e)

   call elpa_uninit()

   status = check_correctness(na, nev, as, z, ev, sc_desc, myid)

   deallocate(a)
   deallocate(as)
   deallocate(z)
   deallocate(ev)

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("program")
   print *," "
   print *,"Timings program:"
   print *," "
   call timer%print("program")
   print *," "
   print *,"End timings program"
   print *," "
#endif


#ifdef WITH_MPI
   call blacs_gridexit(my_blacs_ctxt)
   call mpi_finalize(mpierr)
#endif

   call EXIT(STATUS)

end program
