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
!>
!> Fortran test programm to demonstrates the use of
!> ELPA 1 real case library.
!> If "HAVE_REDIRECT" was defined at build time
!> the stdout and stderr output of each MPI task
!> can be redirected to files if the environment
!> variable "REDIRECT_ELPA_TEST_OUTPUT" is set
!> to "true".
!>
!> By calling executable [arg1] [arg2] [arg3] [arg4]
!> one can define the size (arg1), the number of
!> Eigenvectors to compute (arg2), and the blocking (arg3).
!> If these values are not set default values (4000, 1500, 16)
!> are choosen.
!> If these values are set the 4th argument can be
!> "output", which specifies that the EV's are written to
!> an ascii file.
!>
program test_real_example

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

   use iso_c_binding
   use ELPA1
   use elpa_utilities, only : error_unit
#ifdef HAVE_MPI_MODULE
   use mpi
   implicit none
#else
   implicit none
   include 'mpif.h'
#endif

   !-------------------------------------------------------------------------------
   ! Please set system size parameters below!
   ! na:   System size
   ! nev:  Number of eigenvectors to be calculated
   ! nblk: Blocking factor in block cyclic distribution
   !-------------------------------------------------------------------------------
 !  integer, parameter         :: ik = C_INT32_T
 !  integer, parameter         :: rk = C_DOUBLE

   integer           :: nblk
   integer           :: na, nev

   integer           :: np_rows, np_cols, na_rows, na_cols

   integer           :: myid, nprocs, my_prow, my_pcol, mpi_comm_rows, mpi_comm_cols
   integer           :: i, mpierr, my_blacs_ctxt, sc_desc(9), info, nprow, npcol

   integer, external          :: numroc

   real(kind=c_double), allocatable :: a(:,:), z(:,:), ev(:)

   integer           :: iseed(4096) ! Random seed, size should be sufficient for every generator

   integer           :: STATUS
   logical                    :: success
   character(len=8)           :: task_suffix
   integer           :: j

   !-------------------------------------------------------------------------------


   success = .true.

   ! default parameters
   na = 4000
   nev = 1500
   nblk = 16

   call mpi_init(mpierr)
   call mpi_comm_rank(mpi_comm_world,myid,mpierr)
   call mpi_comm_size(mpi_comm_world,nprocs,mpierr)

   do np_cols = NINT(SQRT(REAL(nprocs))),2,-1
     if(mod(nprocs,np_cols) == 0 ) exit
   enddo
   ! at the end of the above loop, nprocs is always divisible by np_cols

   np_rows = nprocs/np_cols

   if (myid==0) then
     print *
     print '(a)','Standard eigenvalue problem - REAL version'
     print *
     print '(3(a,i0))','Matrix size=',na,', Number of eigenvectors=',nev,', Block size=',nblk
     print '(3(a,i0))','Number of processor rows=',np_rows,', cols=',np_cols,', total=',nprocs
     print *
   endif

   ! initialise BLACS
   my_blacs_ctxt = mpi_comm_world
   call BLACS_Gridinit(my_blacs_ctxt, 'C', np_rows, np_cols)
   call BLACS_Gridinfo(my_blacs_ctxt, nprow, npcol, my_prow, my_pcol)

   if (myid==0) then
     print '(a)','| Past BLACS_Gridinfo.'
   end if
   ! determine the neccessary size of the distributed matrices,
   ! we use the scalapack tools routine NUMROC

   na_rows = numroc(na, nblk, my_prow, 0, np_rows)
   na_cols = numroc(na, nblk, my_pcol, 0, np_cols)

   ! All ELPA routines need MPI communicators for communicating within
   ! rows or columns of processes, these are set in get_elpa_communicators.

   mpierr = get_elpa_communicators(mpi_comm_world, my_prow, my_pcol, &
                                   mpi_comm_rows, mpi_comm_cols)

   ! set up the scalapack descriptor for the checks below
   ! For ELPA the following restrictions hold:
   ! - block sizes in both directions must be identical (args 4 a. 5)
   ! - first row and column of the distributed matrix must be on
   !   row/col 0/0 (arg 6 and 7)

   call descinit(sc_desc, na, na, nblk, nblk, 0, 0, my_blacs_ctxt, na_rows, info)

   if (info .ne. 0) then
     write(error_unit,*) 'Error in BLACS descinit! info=',info
     write(error_unit,*) 'Most likely this happend since you want to use'
     write(error_unit,*) 'more MPI tasks than are possible for your'
     write(error_unit,*) 'problem size (matrix size and blocksize)!'
     write(error_unit,*) 'The blacsgrid can not be set up properly'
     write(error_unit,*) 'Try reducing the number of MPI tasks...'
     call MPI_ABORT(mpi_comm_world, 1, mpierr)
   endif

   if (myid==0) then
     print '(a)','| Past scalapack descriptor setup.'
   end if

   allocate(a (na_rows,na_cols))
   allocate(z (na_rows,na_cols))

   allocate(ev(na))

   ! we want different random numbers on every process
   ! (otherwise A might get rank deficient):

   iseed(:) = myid
   call RANDOM_SEED(put=iseed)
   call RANDOM_NUMBER(z)

   a(:,:) = z(:,:)

   if (myid == 0) then
     print '(a)','| Random matrix block has been set up. (only processor 0 confirms this step)'
   endif
   call pdtran(na, na, 1.d0, z, 1, 1, sc_desc, 1.d0, a, 1, 1, sc_desc) ! A = A + Z**T

   !-------------------------------------------------------------------------------
   ! Calculate eigenvalues/eigenvectors

   if (myid==0) then
     print '(a)','| Entering one-step ELPA solver ... '
     print *
   end if

   call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
   success = solve_evp_real_1stage(na, nev, a, na_rows, ev, z, na_rows, nblk, &
                            na_cols, mpi_comm_rows, mpi_comm_cols)

   if (.not.(success)) then
      write(error_unit,*) "solve_evp_real_1stage produced an error! Aborting..."
      call MPI_ABORT(mpi_comm_world, 1, mpierr)
   endif

   if (myid==0) then
     print '(a)','| One-step ELPA solver complete.'
     print *
   end if

   if (myid == 0) print *,'Time tridiag_real     :',time_evp_fwd
   if (myid == 0) print *,'Time solve_tridi      :',time_evp_solve
   if (myid == 0) print *,'Time trans_ev_real    :',time_evp_back
   if (myid == 0) print *,'Total time (sum above):',time_evp_back+time_evp_solve+time_evp_fwd

   call blacs_gridexit(my_blacs_ctxt)
   call mpi_finalize(mpierr)

end

