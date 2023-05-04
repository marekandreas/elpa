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
#include "../assert.h"

program test_interface
   use elpa

   use precision_for_tests
   !use test_util
   use test_setup_mpi
   use test_prepare_matrix
   use test_read_input_parameters
   use test_blacs_infrastructure
   use test_check_correctness
   implicit none

   ! matrix dimensions
   TEST_INT_TYPE :: na, nev, nblk

   ! mpi
   TEST_INT_TYPE :: myid, nprocs
   TEST_INT_TYPE :: na_cols, na_rows  ! local matrix size
   TEST_INT_TYPE :: np_cols, np_rows  ! number of MPI processes per column/row
   TEST_INT_TYPE :: my_prow, my_pcol  ! local MPI task position (my_prow, my_pcol) in the grid (0..np_cols -1, 0..np_rows -1)
   TEST_INT_MPI_TYPE :: mpierr, blacs_ok_mpi

   ! blacs
   TEST_INT_TYPE :: my_blacs_ctxt, sc_desc(9), info, nprow, npcol, blacs_ok

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
   TEST_INT_TYPE :: status
   integer(kind=c_int) :: error_elpa

   TEST_INT_TYPE :: solver
   TEST_INT_TYPE :: qr

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

#ifdef WITH_CUDA_AWARE_MPI
#ifdef WITH_MPI
     call mpi_finalize(mpierr)
#endif
     stop 77
#endif


   call set_up_blacsgrid(int(mpi_comm_world,kind=BLAS_KIND), np_rows, np_cols, 'C', &
                         my_blacs_ctxt, my_prow, my_pcol)

   call set_up_blacs_descriptor(na, nblk, my_prow, my_pcol, np_rows, np_cols, &
                                na_rows, na_cols, sc_desc, my_blacs_ctxt, info, blacs_ok)
#ifdef WITH_MPI
   blacs_ok_mpi = int(blacs_ok, kind=INT_MPI_TYPE)
   call mpi_allreduce(MPI_IN_PLACE, blacs_ok_mpi, 1_MPI_KIND, MPI_INTEGER, MPI_MIN, int(MPI_COMM_WORLD,kind=MPI_KIND), mpierr)
   blacs_ok = int(blacs_ok_mpi, kind=INT_TYPE)
#endif
   if (blacs_ok .eq. 0) then
     if (myid .eq. 0) then
       print *," Ecountered critical error when setting up blacs. Aborting..."
     endif
#ifdef WITH_MPI
     call mpi_finalize(mpierr)
#endif
     stop 1
   endif

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

   e1 => elpa_allocate(error_elpa)
   assert_elpa_ok(error_elpa)

   call e1%set("na", int(na,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e1%set("nev", int(nev,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e1%set("local_nrows", int(na_rows,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e1%set("local_ncols", int(na_cols,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e1%set("nblk", int(nblk,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
#ifdef WITH_MPI
   call e1%set("mpi_comm_parent", int(MPI_COMM_WORLD,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e1%set("process_row", int(my_prow,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e1%set("process_col", int(my_pcol,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
#endif

   assert(e1%setup() .eq. ELPA_OK)

   call e1%set("solver", ELPA_SOLVER_2STAGE, error_elpa)
   assert_elpa_ok(error_elpa)

   call e1%set("real_kernel", ELPA_2STAGE_REAL_DEFAULT, error_elpa)
   assert_elpa_ok(error_elpa)


   e2 => elpa_allocate(error_elpa)
   assert_elpa_ok(error_elpa)

   call e2%set("na", int(na,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e2%set("nev", int(nev,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e2%set("local_nrows", int(na_rows,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e2%set("local_ncols", int(na_cols,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e2%set("nblk", int(nblk,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
#ifdef WITH_MPI
   call e2%set("mpi_comm_parent", int(MPI_COMM_WORLD,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e2%set("process_row", int(my_prow,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e2%set("process_col", int(my_pcol,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
#endif
   assert(e2%setup() .eq. ELPA_OK)

   call e2%set("solver", ELPA_SOLVER_1STAGE, error_elpa)
   assert_elpa_ok(error_elpa)

   call e1%eigenvectors(a1, ev1, z1, error_elpa)
   assert_elpa_ok(error_elpa)

   call elpa_deallocate(e1, error_elpa)
   assert_elpa_ok(error_elpa)

   call e2%eigenvectors(a2, ev2, z2, error_elpa)
   assert_elpa_ok(error_elpa)

   call elpa_deallocate(e2, error_elpa)
   assert_elpa_ok(error_elpa)

   call elpa_uninit(error_elpa)

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
