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
!>
!> Fortran test programm to demonstrates the use of
!> ELPA 2 real case library.
!> If "HAVE_REDIRECT" was defined at build time
!> the stdout and stderr output of each MPI task
!> can be redirected to files if the environment
!> variable "REDIRECT_ELPA_TEST_OUTPUT" is set
!> to "true".
!>
!> By calling executable [arg1] [arg2] [arg3] [arg4]
!> one can define the size (arg1), the number of
!> Eigenvectors to compute (arg2), and the blocking (arg3).
!> If these values are not set default values (500, 150, 16)
!> are choosen.
!> If these values are set the 4th argument can be
!> "output", which specifies that the EV's are written to
!> an ascii file.
!>
!> The real ELPA 2 kernel is set as the default kernel.
!> However, this can be overriden by setting
!> the environment variable "REAL_ELPA_KERNEL" to an
!> appropiate value.
!>

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

program test_real2_double_banded

!-------------------------------------------------------------------------------
! Standard eigenvalue problem - REAL version
!
! This program demonstrates the use of the ELPA module
! together with standard scalapack routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
!-------------------------------------------------------------------------------
   use elpa

   !use test_util
   use test_read_input_parameters
   use test_check_correctness
   use test_setup_mpi
   use test_blacs_infrastructure
   use test_prepare_matrix
#ifdef HAVE_REDIRECT
   use test_redirect
#endif
   use test_output_type
   implicit none

   !-------------------------------------------------------------------------------
   ! Please set system size parameters below!
   ! na:   System size
   ! nev:  Number of eigenvectors to be calculated
   ! nblk: Blocking factor in block cyclic distribution
   !-------------------------------------------------------------------------------

   TEST_INT_TYPE           :: nblk
   TEST_INT_TYPE           :: na, nev

   TEST_INT_TYPE           :: np_rows, np_cols, na_rows, na_cols

   TEST_INT_TYPE           :: myid, nprocs, my_prow, my_pcol, mpi_comm_rows, mpi_comm_cols
   TEST_INT_TYPE           :: i,  my_blacs_ctxt, sc_desc(9), info, nprow, npcol, blacs_ok
   TEST_INT_MPI_TYPE       :: mpierr, blacs_ok_mpi
   !TEST_INT_TYPE, external :: numroc

   real(kind=rk8), allocatable :: a(:,:), z(:,:), as(:,:), ev(:)

   TEST_INT_TYPE           :: STATUS
#ifdef WITH_OPENMP_TRADITIONAL
   TEST_INT_TYPE           :: omp_get_max_threads,  required_mpi_thread_level, provided_mpi_thread_level
#endif
   integer(kind=c_int)          :: error_elpa
   TEST_INT_TYPE           :: numberOfDevices
   type(output_t)             :: write_to_file
   character(len=8)           :: task_suffix
   TEST_INT_TYPE           :: j
   TEST_INT_TYPE           :: global_row, global_col, local_row, local_col
   TEST_INT_TYPE           :: bandwidth
   class(elpa_t), pointer     :: e
#define DOUBLE_PRECISION_REAL 1


   call read_input_parameters(na, nev, nblk, write_to_file)

   if (nblk .eq. 1) then
     stop 77
   endif

   !-------------------------------------------------------------------------------
   !  MPI Initialization
   call setup_mpi(myid, nprocs)

   STATUS = 0

#ifdef WITH_CUDA_AWARE_MPI
#ifdef WITH_MPI
     call mpi_finalize(mpierr)
#endif
     stop 77
#endif


#define REALCASE

   !-------------------------------------------------------------------------------
   ! Selection of number of processor rows/columns
   ! We try to set up the grid square-like, i.e. start the search for possible
   ! divisors of nprocs with a number next to the square root of nprocs
   ! and decrement it until a divisor is found.

   do np_cols = NINT(SQRT(REAL(nprocs))),2,-1
      if(mod(nprocs,np_cols) == 0 ) exit
   enddo
   ! at the end of the above loop, nprocs is always divisible by np_cols

   np_rows = nprocs/np_cols

   if(myid==0) then
      print *
      print '(a)','Standard eigenvalue problem - REAL version'
      print *
      print '(3(a,i0))','Matrix size=',na,', Number of eigenvectors=',nev,', Block size=',nblk
      print '(3(a,i0))','Number of processor rows=',np_rows,', cols=',np_cols,', total=',nprocs
      print *
   endif

   !-------------------------------------------------------------------------------
   ! Set up BLACS context and MPI communicators
   !
   ! The BLACS context is only necessary for using Scalapack.
   !
   ! For ELPA, the MPI communicators along rows/cols are sufficient,
   ! and the grid setup may be done in an arbitrary way as long as it is
   ! consistent (i.e. 0<=my_prow<np_rows, 0<=my_pcol<np_cols and every
   ! process has a unique (my_prow,my_pcol) pair).

   call set_up_blacsgrid(int(mpi_comm_world,kind=BLAS_KIND), np_rows, np_cols, 'C', &
                         my_blacs_ctxt, my_prow, my_pcol)

   if (myid==0) then
     print '(a)','| Past BLACS_Gridinfo.'
   end if

   call set_up_blacs_descriptor(na ,nblk, my_prow, my_pcol, np_rows, np_cols, &
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

   if (myid==0) then
     print '(a)','| Past scalapack descriptor setup.'
   end if

   !-------------------------------------------------------------------------------
   ! Allocate matrices and set up a test matrix for the eigenvalue problem
   allocate(a (na_rows,na_cols))
   allocate(z (na_rows,na_cols))
   allocate(as(na_rows,na_cols))

   allocate(ev(na))

   call prepare_matrix_random(na, myid, sc_desc, a, z, as)

   ! set values outside of the bandwidth to zero
   bandwidth = nblk

   do local_row = 1, na_rows
     global_row = index_l2g(local_row, nblk, my_prow, np_rows)
     do local_col = 1, na_cols
       global_col = index_l2g(local_col, nblk, my_pcol, np_cols)

       if (ABS(global_row-global_col) > bandwidth) then
         a(local_row, local_col) = 0.0
         as(local_row, local_col) = 0.0
       end if
     end do
   end do

   if (elpa_init(CURRENT_API_VERSION) /= ELPA_OK) then
     print *, "ELPA API version not supported"
     stop 1
   endif
   e => elpa_allocate(error_elpa)
   assert_elpa_ok(error_elpa)

   call e%set("na", int(na,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("nev", int(nev,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("local_nrows", int(na_rows,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("local_ncols", int(na_cols,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("nblk", int(nblk,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
#ifdef WITH_MPI
   call e%set("mpi_comm_parent", int(MPI_COMM_WORLD,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("process_row", int(my_prow,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
   call e%set("process_col", int(my_pcol,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)
#endif

   call e%set("bandwidth", int(bandwidth,kind=c_int), error_elpa)
   assert_elpa_ok(error_elpa)

   assert(e%setup() .eq. ELPA_OK)

   call e%set("solver", ELPA_SOLVER_2STAGE, error_elpa)
   assert_elpa_ok(error_elpa)

   call e%eigenvectors(a, ev, z, error_elpa)
   assert_elpa_ok(error_elpa)

   call elpa_deallocate(e, error_elpa)
   assert_elpa_ok(error_elpa)

   call elpa_uninit(error_elpa)
   assert_elpa_ok(error_elpa)


   !-------------------------------------------------------------------------------
   ! Test correctness of result (using plain scalapack routines)

   
   status = check_correctness_evp_numeric_residuals(na, nev, as, z, ev, sc_desc, nblk, myid, np_rows, np_cols, my_prow, my_pcol)


   deallocate(a)
   deallocate(as)

   deallocate(z)
   deallocate(ev)

#ifdef WITH_MPI
   call blacs_gridexit(my_blacs_ctxt)
   call mpi_finalize(mpierr)
#endif
   call EXIT(STATUS)
end

!-------------------------------------------------------------------------------
