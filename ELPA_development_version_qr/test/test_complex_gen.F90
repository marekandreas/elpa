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
program test_complex_gen

!-------------------------------------------------------------------------------
! Generalized eigenvalue problem - COMPLEX version
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
   integer i, n_row, n_col, mpierr, my_blacs_ctxt, sc_desc(9), info, nprow, npcol

   integer, external :: numroc, indxl2g

   real*8 err, errmax
   real*8, allocatable :: ev(:), xr(:,:)
   complex*16 xc
   complex*16, allocatable :: a(:,:), z(:,:), tmp1(:,:), tmp2(:,:), as(:,:)
   complex*16, allocatable :: b(:,:), bs(:,:)

   complex*16, parameter :: CZERO = (0.d0,0.d0), CONE = (1.d0,0.d0)

   integer :: iseed(4096) ! Random seed, size should be sufficient for every generator
   real*8 ttt0, ttt1

   !-------------------------------------------------------------------------------
   !  MPI Initialization

   call mpi_init(mpierr)
   call mpi_comm_rank(mpi_comm_world,myid,mpierr)
   call mpi_comm_size(mpi_comm_world,nprocs,mpierr)

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
      print '(a)','Generalized eigenvalue problem - COMPLEX version'
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
   ! Allocate matrices and set up test matrices for the eigenvalue problem

   allocate(a (na_rows,na_cols))
   allocate(z (na_rows,na_cols))
   allocate(as(na_rows,na_cols))
   allocate(b (na_rows,na_cols))
   allocate(bs(na_rows,na_cols))

   allocate(tmp1(na_rows,na_cols))
   allocate(tmp2(na_rows,na_cols))

   allocate(ev(na))

   ! For getting a hermitian test matrix A we get a random matrix Z
   ! and calculate A = Z + Z**H

   ! We want different random numbers on every process
   ! (otherways A might get rank deficient):

   iseed(:) = myid
   call RANDOM_SEED(put=iseed)

   allocate(xr(na_rows,na_cols))
   call RANDOM_NUMBER(xr)
   z(:,:) = xr(:,:)
   call RANDOM_NUMBER(xr)
   z(:,:) = z(:,:) + (0.d0,1.d0)*xr(:,:)
   deallocate(xr)

   a(:,:) = z(:,:)

   if (myid==0) then
     print '(a)','| Random matrix block has been set up. (only processor 0 confirms this step)'
   end if

   call pztranc(na, na, CONE, z, 1, 1, sc_desc, CONE, a, 1, 1, sc_desc) ! A = A + Z**H

   if (myid==0) then
     print '(a)','| Random matrix has been Hermite-icized.'
   end if

   ! The matrix B in the generalized eigenvalue problem must be symmetric
   ! and positive definite - we use a simple diagonally dominant matrix

   xc = (0.7,0.4)/na ! some random value with abs(xc) < 1/na
   call pzlaset('U', na, na, xc, CONE, b, 1, 1, sc_desc ) ! Upper part
   call pzlaset('L', na, na, conjg(xc), CONE, b, 1, 1, sc_desc ) ! Lower part

   if (myid==0) then
     print '(a)','| Complex Hermitian diagonally dominant overlap matrix has been initialized.'
   end if

   ! Save original matrices A and B for later accuracy checks

   as = a
   bs = b

   !-------------------------------------------------------------------------------
   ! Solve generalized problem
   !
   ! 1. Calculate Cholesky factorization of Matrix B = U**T * U
   !    and invert triangular matrix U
   !
   ! Please note: cholesky_complex/invert_trm_complex are not trimmed for speed.
   ! The only reason having them is that the Scalapack counterpart
   ! PDPOTRF very often fails on higher processor numbers for unknown reasons!

   call cholesky_complex(na, b, na_rows, nblk, mpi_comm_rows, mpi_comm_cols)

   if (myid==0) then
     print '(a)','| Cholesky factorization complete.'
   end if

   call invert_trm_complex(na, b, na_rows, nblk, mpi_comm_rows, mpi_comm_cols)

   if (myid==0) then
     print '(a)','| Cholesky factor inverted.'
   end if

   ttt0 = MPI_Wtime()

   ! 2. Calculate U**-T * A * U**-1

   ! 2a. tmp1 = U**-T * A
   call mult_ah_b_complex('U', 'L', na, na, b, na_rows, a, na_rows, &
                          nblk, mpi_comm_rows, mpi_comm_cols, tmp1, na_rows)

   ! 2b. tmp2 = tmp1**T
   call pztranc(na,na,CONE,tmp1,1,1,sc_desc,CZERO,tmp2,1,1,sc_desc)

   ! 2c. A =  U**-T * tmp2 ( = U**-T * Aorig * U**-1 )
   call mult_ah_b_complex('U', 'U', na, na, b, na_rows, tmp2, na_rows, &
                          nblk, mpi_comm_rows, mpi_comm_cols, a, na_rows)
   ttt1 = MPI_Wtime()

   if (myid==0) then
     print '(a)','| Matrix A transformed from generalized to orthogonal form using Cholesky factors.'
   end if

   if(myid == 0) print *,'Time U**-T*A*U**-1   :',ttt1-ttt0

   ! A is only set in the upper half, solve_evp_real needs a full matrix
   ! Set lower half from upper half

   call pztranc(na,na,CONE,a,1,1,sc_desc,CZERO,tmp1,1,1,sc_desc)

   if (myid==0) then
     print '(a)','| Lower half of A set by pztranc.'
   end if

   do i=1,na_cols
      ! Get global column corresponding to i and number of local rows up to
      ! and including the diagonal, these are unchanged in A
      n_col = indxl2g(i,     nblk, my_pcol, 0, np_cols)
      n_row = numroc (n_col, nblk, my_prow, 0, np_rows)
      a(n_row+1:na_rows,i) = tmp1(n_row+1:na_rows,i)
   enddo

   ! 3. Calculate eigenvalues/eigenvectors of U**-T * A * U**-1
   !    Eigenvectors go to tmp1

   if (myid==0) then
     print '(a)','| Entering one-step ELPA solver ... '
     print *
   end if

   call solve_evp_complex(na, nev, a, na_rows, ev, tmp1, na_rows, nblk, &
                          mpi_comm_rows, mpi_comm_cols)

   if (myid==0) then
     print '(a)','| One-step ELPA solver complete.'
     print *
   end if

   if(myid == 0) print *,'Time tridiag_complex :',time_evp_fwd
   if(myid == 0) print *,'Time solve_tridi     :',time_evp_solve
   if(myid == 0) print *,'Time trans_ev_complex:',time_evp_back

   ! 4. Backtransform eigenvectors: Z = U**-1 * tmp1

   ttt0 = MPI_Wtime()
   ! mult_ah_b_complex needs the transpose of U**-1, thus tmp2 = (U**-1)**T
   call pztranc(na,na,CONE,b,1,1,sc_desc,CZERO,tmp2,1,1,sc_desc)

   call mult_ah_b_complex('L', 'N', na, nev, tmp2, na_rows, tmp1, na_rows, &
                          nblk, mpi_comm_rows, mpi_comm_cols, z, na_rows)
   ttt1 = MPI_Wtime()
   if (myid==0) then
     print '(a)','| Backtransform of eigenvectors to generalized form complete.'
   end if
   if(myid == 0) print *,'Time Back U**-1*Z    :',ttt1-ttt0

   !-------------------------------------------------------------------------------
   ! Test correctness of result (using plain scalapack routines)

   ! 1. Residual (maximum of || A*Zi - B*Zi*EVi ||)

   ! tmp1 =  A * Z
   call pzgemm('N','N',na,nev,na,CONE,as,1,1,sc_desc, &
               z,1,1,sc_desc,CZERO,tmp1,1,1,sc_desc)

   ! tmp2 = B*Zi*EVi
   call pzgemm('N','N',na,nev,na,CONE,bs,1,1,sc_desc, &
               z,1,1,sc_desc,CZERO,tmp2,1,1,sc_desc)
   do i=1,nev
      xc = ev(i)
      call pzscal(na,xc,tmp2,1,i,sc_desc,1)
   enddo

   !  tmp1 = A*Zi - B*Zi*EVi
   tmp1(:,:) =  tmp1(:,:) - tmp2(:,:)

   ! Get maximum norm of columns of tmp1
   errmax = 0
   do i=1,nev
      xc = 0
      call pzdotc(na,xc,tmp1,1,i,sc_desc,1,tmp1,1,i,sc_desc,1)
      errmax = max(errmax, sqrt(real(xc,8)))
   enddo

   ! Get maximum error norm over all processors
   err = errmax
   call mpi_allreduce(err,errmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
   if(myid==0) print *
   if(myid==0) print *,'Error Residual     :',errmax

   ! 2. Eigenvector orthogonality

   ! tmp1 = Z**T * B * Z

   call pzgemm('N','N',na,nev,na,CONE,bs,1,1,sc_desc, &
               z,1,1,sc_desc,CZERO,tmp2,1,1,sc_desc)
   tmp1 = 0
   call pzgemm('C','N',nev,nev,na,CONE,z,1,1,sc_desc, &
               tmp2,1,1,sc_desc,CZERO,tmp1,1,1,sc_desc)

   ! Initialize tmp2 to unit matrix
   tmp2 = 0
   call pzlaset('A',nev,nev,CZERO,CONE,tmp2,1,1,sc_desc)

   ! tmp1 = Z**T * B * Z - Unit Matrix
   tmp1(:,:) =  tmp1(:,:) - tmp2(:,:)

   ! Get maximum error (max abs value in tmp1)
   err = maxval(abs(tmp1))
   call mpi_allreduce(err,errmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
   if(myid==0) print *,'Error Orthogonality:',errmax

   call mpi_finalize(mpierr)
end
