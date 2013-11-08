!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium, 
!    consisting of the following organizations:
!
!    - Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG), 
!    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!      Informatik,
!    - Technische Universität München, Lehrstuhl für Informatik mit
!      Schwerpunkt Wissenschaftliches Rechnen , 
!    - Fritz-Haber-Institut, Berlin, Abt. Theorie, 
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaftrn, 
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition, 
!      and  
!    - IBM Deutschland GmbH
!
!
!    More information can be found here:
!    http://elpa.rzg.mpg.de/
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
program test_real2

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

   use ELPA1
   use ELPA2

   implicit none
   include 'mpif.h'

   !-------------------------------------------------------------------------------
   ! Please set system size parameters below!
   ! na:   System size
   ! nev:  Number of eigenvectors to be calculated
   ! nblk: Blocking factor in block cyclic distribution
   !-------------------------------------------------------------------------------

   integer, parameter :: na = 4000, nev = 1500, nblk = 16

   !-------------------------------------------------------------------------------
   !  Local Variables

   integer np_rows, np_cols, na_rows, na_cols

   integer myid, nprocs, my_prow, my_pcol, mpi_comm_rows, mpi_comm_cols
   integer i, mpierr, my_blacs_ctxt, sc_desc(9), info, nprow, npcol

   integer, external :: numroc

   real*8 err, errmax
   real*8, allocatable :: a(:,:), z(:,:), tmp1(:,:), tmp2(:,:), as(:,:), ev(:)

   integer :: iseed(4096) ! Random seed, size should be sufficient for every generator

   integer :: STATUS

   !-------------------------------------------------------------------------------
   !  MPI Initialization

   call mpi_init(mpierr)
   call mpi_comm_rank(mpi_comm_world,myid,mpierr)
   call mpi_comm_size(mpi_comm_world,nprocs,mpierr)

   STATUS = 0

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

   my_blacs_ctxt = mpi_comm_world
   call BLACS_Gridinit( my_blacs_ctxt, 'C', np_rows, np_cols )
   call BLACS_Gridinfo( my_blacs_ctxt, nprow, npcol, my_prow, my_pcol )

   if (myid==0) then
     print '(a)','| Past BLACS_Gridinfo.'
   end if

   ! All ELPA routines need MPI communicators for communicating within
   ! rows or columns of processes, these are set in get_elpa_row_col_comms.

   call get_elpa_row_col_comms(mpi_comm_world, my_prow, my_pcol, &
                               mpi_comm_rows, mpi_comm_cols)

   if (myid==0) then
     print '(a)','| Past split communicator setup for rows and columns.'
   end if

   ! Determine the necessary size of the distributed matrices,
   ! we use the Scalapack tools routine NUMROC for that.

   na_rows = numroc(na, nblk, my_prow, 0, np_rows)
   na_cols = numroc(na, nblk, my_pcol, 0, np_cols)

   ! Set up a scalapack descriptor for the checks below.
   ! For ELPA the following restrictions hold:
   ! - block sizes in both directions must be identical (args 4+5)
   ! - first row and column of the distributed matrix must be on row/col 0/0 (args 6+7)

   call descinit( sc_desc, na, na, nblk, nblk, 0, 0, my_blacs_ctxt, na_rows, info )

   if (myid==0) then
     print '(a)','| Past scalapack descriptor setup.'
   end if

   !-------------------------------------------------------------------------------
   ! Allocate matrices and set up a test matrix for the eigenvalue problem

   allocate(a (na_rows,na_cols))
   allocate(z (na_rows,na_cols))
   allocate(as(na_rows,na_cols))

   allocate(ev(na))

   ! For getting a symmetric test matrix A we get a random matrix Z
   ! and calculate A = Z + Z**T

   ! We want different random numbers on every process
   ! (otherways A might get rank deficient):

   iseed(:) = myid
   call RANDOM_SEED(put=iseed)

   call RANDOM_NUMBER(z)

   a(:,:) = z(:,:)

   if (myid==0) then
     print '(a)','| Random matrix block has been set up. (only processor 0 confirms this step)'
   end if

   call pdtran(na, na,  1.d0, z, 1, 1, sc_desc, 1.d0, a, 1, 1, sc_desc) ! A = A + Z**T

   if (myid==0) then
     print '(a)','| Random matrix has been symmetrized.'
   end if

   ! Save original matrix A for later accuracy checks

   as = a

   ! set print flag in elpa1
   elpa_print_times = .true.

   !-------------------------------------------------------------------------------
   ! Calculate eigenvalues/eigenvectors

   if (myid==0) then
     print '(a)','| Entering two-stage ELPA solver ... '
     print *
   end if

   call mpi_barrier(mpi_comm_world, mpierr) ! for correct timings only
   call solve_evp_real_2stage(na, nev, a, na_rows, ev, z, na_rows, nblk, &
                              mpi_comm_rows, mpi_comm_cols, mpi_comm_world)

   if (myid==0) then
     print '(a)','| Two-step ELPA solver complete.'
     print *
   end if

   if(myid == 0) print *,'Time transform to tridi :',time_evp_fwd
   if(myid == 0) print *,'Time solve tridi        :',time_evp_solve
   if(myid == 0) print *,'Time transform back EVs :',time_evp_back

   !-------------------------------------------------------------------------------
   ! Test correctness of result (using plain scalapack routines)

   deallocate(a)
   allocate(tmp1(na_rows,na_cols))

   ! 1. Residual (maximum of || A*Zi - Zi*EVi ||)

   ! tmp1 =  A * Z
   call pdgemm('N','N',na,nev,na,1.d0,as,1,1,sc_desc, &
           z,1,1,sc_desc,0.d0,tmp1,1,1,sc_desc)

   deallocate(as)
   allocate(tmp2(na_rows,na_cols))

   ! tmp2 = Zi*EVi
   tmp2(:,:) = z(:,:)
   do i=1,nev
      call pdscal(na,ev(i),tmp2,1,i,sc_desc,1)
   enddo

   !  tmp1 = A*Zi - Zi*EVi
   tmp1(:,:) =  tmp1(:,:) - tmp2(:,:)

   ! Get maximum norm of columns of tmp1
   errmax = 0
   do i=1,nev
      err = 0
      call pdnrm2(na,err,tmp1,1,i,sc_desc,1)
      errmax = max(errmax, err)
   enddo


   ! Get maximum error norm over all processors
   err = errmax
   call mpi_allreduce(err,errmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
   if(myid==0) print *
   if(myid==0) print *,'Error Residual     :',errmax


   if (errmax .gt. 5e-12) then
      status = 1
   endif

   ! 2. Eigenvector orthogonality

   ! tmp1 = Z**T * Z
   tmp1 = 0
   call pdgemm('T','N',nev,nev,na,1.d0,z,1,1,sc_desc, &
           z,1,1,sc_desc,0.d0,tmp1,1,1,sc_desc)
   ! Initialize tmp2 to unit matrix
   tmp2 = 0
   call pdlaset('A',nev,nev,0.d0,1.d0,tmp2,1,1,sc_desc)

   ! tmp1 = Z**T * Z - Unit Matrix
   tmp1(:,:) =  tmp1(:,:) - tmp2(:,:)

   ! Get maximum error (max abs value in tmp1)
   err = maxval(abs(tmp1))
   call mpi_allreduce(err,errmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
   if(myid==0) print *,'Error Orthogonality:',errmax
   
   if (errmax .gt. 5e-12) then
      status = 1
   endif

   deallocate(z)
   deallocate(tmp1)
   deallocate(tmp2)
   deallocate(ev)
   call blacs_gridexit(my_blacs_ctxt)
   call mpi_finalize(mpierr)
   call EXIT(STATUS)
end

!-------------------------------------------------------------------------------
