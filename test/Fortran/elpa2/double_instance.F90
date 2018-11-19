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

#include "../assert.h"

program test_interface
   use elpa
   use test_util
   use test_setup_mpi
   use test_prepare_matrix
   use test_read_input_parameters
   use test_blacs_infrastructure
   use test_check_correctness
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
   real(kind=C_DOUBLE), allocatable :: a1(:,:), as1(:,:)
   ! eigenvectors
   real(kind=C_DOUBLE), allocatable :: z1(:,:)
   ! eigenvalues
   real(kind=C_DOUBLE), allocatable :: ev1(:)

   ! The Matrix
   complex(kind=C_DOUBLE_COMPLEX), allocatable :: a2(:,:), as2(:,:)
   ! eigenvectors
   complex(kind=C_DOUBLE_COMPLEX), allocatable :: z2(:,:)
   ! eigenvalues
   real(kind=C_DOUBLE), allocatable :: ev2(:)
   integer :: success, status

   integer(kind=c_int) :: solver
   integer(kind=c_int) :: qr

   type(output_t) :: write_to_file
   class(elpa_t), pointer :: e1, e2

   call read_input_parameters(na, nev, nblk, write_to_file)
   call setup_mpi(myid, nprocs)

   status = 0

   do np_cols = NINT(SQRT(REAL(nprocs))),2,-1
      if(mod(nprocs,np_cols) == 0 ) exit
   enddo

   np_rows = nprocs/np_cols

   my_prow = mod(myid, np_cols)
   my_pcol = myid / np_cols

   call set_up_blacsgrid(mpi_comm_world, np_rows, np_cols, 'C', &
                         my_blacs_ctxt, my_prow, my_pcol)

   call set_up_blacs_descriptor(na, nblk, my_prow, my_pcol, np_rows, np_cols, &
                                na_rows, na_cols, sc_desc, my_blacs_ctxt, info)

   allocate(a1 (na_rows,na_cols), as1(na_rows,na_cols))
   allocate(z1 (na_rows,na_cols))
   allocate(ev1(na))

   a1(:,:) = 0.0
   z1(:,:) = 0.0
   ev1(:) = 0.0

   call prepare_matrix_random(na, myid, sc_desc, a1, z1, as1)
   allocate(a2 (na_rows,na_cols), as2(na_rows,na_cols))
   allocate(z2 (na_rows,na_cols))
   allocate(ev2(na))

   a2(:,:) = 0.0
   z2(:,:) = 0.0
   ev2(:) = 0.0

   call prepare_matrix_random(na, myid, sc_desc, a2, z2, as2)

   if (elpa_init(CURRENT_API_VERSION) /= ELPA_OK) then
     print *, "ELPA API version not supported"
     stop 1
   endif

   e1 => elpa_allocate()

   call e1%set("na", na, success)
   assert_elpa_ok(success)
   call e1%set("nev", nev, success)
   assert_elpa_ok(success)
   call e1%set("local_nrows", na_rows, success)
   assert_elpa_ok(success)
   call e1%set("local_ncols", na_cols, success)
   assert_elpa_ok(success)
   call e1%set("nblk", nblk, success)
   assert_elpa_ok(success)
#ifdef WITH_MPI
   call e1%set("mpi_comm_parent", MPI_COMM_WORLD, success)
   assert_elpa_ok(success)
   call e1%set("process_row", my_prow, success)
   assert_elpa_ok(success)
   call e1%set("process_col", my_pcol, success)
   assert_elpa_ok(success)
#endif

   assert(e1%setup() .eq. ELPA_OK)

   call e1%set("solver", ELPA_SOLVER_2STAGE, success)
   assert_elpa_ok(success)

   call e1%set("real_kernel", ELPA_2STAGE_REAL_DEFAULT, success)
   assert_elpa_ok(success)


   e2 => elpa_allocate()

   call e2%set("na", na, success)
   assert_elpa_ok(success)
   call e2%set("nev", nev, success)
   assert_elpa_ok(success)
   call e2%set("local_nrows", na_rows, success)
   assert_elpa_ok(success)
   call e2%set("local_ncols", na_cols, success)
   assert_elpa_ok(success)
   call e2%set("nblk", nblk, success)
   assert_elpa_ok(success)
#ifdef WITH_MPI
   call e2%set("mpi_comm_parent", MPI_COMM_WORLD, success)
   assert_elpa_ok(success)
   call e2%set("process_row", my_prow, success)
   assert_elpa_ok(success)
   call e2%set("process_col", my_pcol, success)
   assert_elpa_ok(success)
#endif
   assert(e2%setup() .eq. ELPA_OK)

   call e2%set("solver", ELPA_SOLVER_1STAGE, success)
   assert_elpa_ok(success)

   call e1%eigenvectors(a1, ev1, z1, success)
   assert_elpa_ok(success)
   call elpa_deallocate(e1)

   call e2%eigenvectors(a2, ev2, z2, success)
   assert_elpa_ok(success)
   call elpa_deallocate(e2)
   call elpa_uninit()

   status = check_correctness_evp_numeric_residuals(na, nev, as1, z1, ev1, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol)

   deallocate(a1)
   deallocate(as1)
   deallocate(z1)
   deallocate(ev1)

   status = check_correctness_evp_numeric_residuals(na, nev, as2, z2, ev2, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol)

   deallocate(a2)
   deallocate(as2)
   deallocate(z2)
   deallocate(ev2)

#ifdef WITH_MPI
   call blacs_gridexit(my_blacs_ctxt)
   call mpi_finalize(mpierr)
#endif
   call EXIT(STATUS)


end program
