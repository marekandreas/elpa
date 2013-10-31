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
program read_real_gen

!-------------------------------------------------------------------------------
! Generalized eigenvalue problem - REAL version
!
! This program demonstrates the use of the ELPA module
! together with standard scalapack routines
!-------------------------------------------------------------------------------

   use ELPA1

   implicit none
   include 'mpif.h'

   !-------------------------------------------------------------------------------
   ! Please set system size parameters below!
   ! nblk: Blocking factor in block cyclic distribution
   !-------------------------------------------------------------------------------

   integer, parameter :: nblk = 16

   !-------------------------------------------------------------------------------
   !  Local Variables

   integer na, nb, nev

   integer np_rows, np_cols, na_rows, na_cols

   integer myid, nprocs, my_prow, my_pcol, mpi_comm_rows, mpi_comm_cols
   integer i, n_row, n_col, mpierr, my_blacs_ctxt, sc_desc(9), info, nprow, npcol, lenarg

   integer, external :: numroc, indxl2g

   real*8 err, errmax
   real*8, allocatable :: a(:,:), z(:,:), tmp1(:,:), tmp2(:,:), as(:,:), ev(:)
   real*8, allocatable :: b(:,:), bs(:,:)

   character(256) :: filename, fmttype
   real*8 ttt0, ttt1

   !-------------------------------------------------------------------------------
   !  MPI Initialization

   call mpi_init(mpierr)
   call mpi_comm_rank(mpi_comm_world,myid,mpierr)
   call mpi_comm_size(mpi_comm_world,nprocs,mpierr)


   !-------------------------------------------------------------------------------
   ! Get the name of the input files and open input files
   ! Please note:
   ! get_command_argument is a FORTRAN 2003 intrinsic which may not be implemented
   ! for every Fortran compiler!!!

   if (myid == 0) then


      ! 1. Get the format of the matrices.

      call get_command_argument( 1, fmttype, lenarg, info )

      if (  info /= 0  .or.                                              &
            (  trim(fmttype) /= 'bin'  .and.  trim(fmttype) /= 'asc'  )  &
         ) then
         print *, 'Usage:  read_real_gen  format  matrix_file_1  matrix_file_2'
         call mpi_abort(mpi_comm_world,0,mpierr)
      endif

      if ( trim(fmttype) == 'bin' ) then
        fmttype = 'unformatted'
      else  ! 'asc'
        fmttype = 'formatted'
      endif


      ! 2. Get the file name of the first matrix.

      call get_command_argument(2,filename,lenarg,info)

      if(info/=0) then
         print *, 'Usage:  read_real_gen  format  matrix_file_1  matrix_file_2'
         call mpi_abort(mpi_comm_world,0,mpierr)
      endif

      open( 10, file=filename, action='READ', status='OLD', form=trim(fmttype), iostat=info )

      if(info/=0) then
         print *,'Error: Unable to open ',trim(filename)
         call mpi_abort(mpi_comm_world,0,mpierr)
      endif


      ! 3. Get the file name of the second matrix.

      call get_command_argument(3,filename,lenarg,info)

      if(info/=0) then
         print *, 'Usage:  read_real_gen  format  matrix_file_1  matrix_file_2'
         call mpi_abort(mpi_comm_world,0,mpierr)
      endif

      open( 20, file=filename, action='READ', status='OLD', form=trim(fmttype), iostat=info )

      if(info/=0) then
         print *,'Error: Unable to open ',trim(filename)
         call mpi_abort(mpi_comm_world,0,mpierr)
      endif

   endif  ! (myid == 0)


   call mpi_barrier(mpi_comm_world, mpierr) ! Just for safety


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
      print '(a)','Generalized eigenvalue problem - REAL version'
      print *
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

   ! All ELPA routines need MPI communicators for communicating within
   ! rows or columns of processes, these are set in get_elpa_row_col_comms.

   call get_elpa_row_col_comms(mpi_comm_world, my_prow, my_pcol, &
                               mpi_comm_rows, mpi_comm_cols)

   ! Read matrix size
   if(myid==0) then

      if ( trim(fmttype) == 'unformatted' ) then
        read (10  ) na
        read (20  ) nb
      else
        read (10,*) na
        read (20,*) nb
      endif

      if ( na /= nb ) then
         print *, 'Error: Matrix sizes in input differ: ', na, nb
         call mpi_abort(mpi_comm_world,0,mpierr)
      endif

   endif

   call mpi_bcast(na, 1, mpi_integer, 0, mpi_comm_world, mpierr)

   ! Quick check for plausibility
   if(na<=0 .or. na>10000000) then
      if(myid==0) print *,'Illegal value for matrix size: ',na
      call mpi_finalize(mpierr)
      stop
   endif
   if(myid==0) print *,'Matrix size: ',na

   ! Determine the necessary size of the distributed matrices,
   ! we use the Scalapack tools routine NUMROC for that.

   na_rows = numroc(na, nblk, my_prow, 0, np_rows)
   na_cols = numroc(na, nblk, my_pcol, 0, np_cols)

   ! Set up a scalapack descriptor for the checks below.
   ! For ELPA the following restrictions hold:
   ! - block sizes in both directions must be identical (args 4+5)
   ! - first row and column of the distributed matrix must be on row/col 0/0 (args 6+7)

   call descinit( sc_desc, na, na, nblk, nblk, 0, 0, my_blacs_ctxt, na_rows, info )

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

   !-------------------------------------------------------------------------------
   ! Read matrices

   call read_matrix(10, fmttype, na, a, ubound(a,1), nblk, my_prow, my_pcol, np_rows, np_cols)
   call read_matrix(20, fmttype, na, b, ubound(b,1), nblk, my_prow, my_pcol, np_rows, np_cols)
   if(myid==0) close(10)
   if(myid==0) close(20)

   nev = na ! all eigenvaules

   ! Save original matrices A and B for later accuracy checks

   as = a
   bs = b

   !-------------------------------------------------------------------------------
   ! Solve generalized problem
   !
   ! 1. Calculate Cholesky factorization of Matrix B = U**T * U
   !    and invert triangular matrix U
   !
   ! Please note: cholesky_real/invert_trm_real are not trimmed for speed.
   ! The only reason having them is that the Scalapack counterpart
   ! PDPOTRF very often fails on higher processor numbers for unknown reasons!

   call cholesky_real(na, b, na_rows, nblk, mpi_comm_rows, mpi_comm_cols)
   call invert_trm_real(na, b, na_rows, nblk, mpi_comm_rows, mpi_comm_cols)

   ttt0 = MPI_Wtime()

   ! 2. Calculate U**-T * A * U**-1

   ! 2a. tmp1 = U**-T * A
   call mult_at_b_real('U', 'L', na, na, b, na_rows, a, na_rows, &
                       nblk, mpi_comm_rows, mpi_comm_cols, tmp1, na_rows)

   ! 2b. tmp2 = tmp1**T
   call pdtran(na,na,1.d0,tmp1,1,1,sc_desc,0.d0,tmp2,1,1,sc_desc)

   ! 2c. A =  U**-T * tmp2 ( = U**-T * Aorig * U**-1 )
   call mult_at_b_real('U', 'U', na, na, b, na_rows, tmp2, na_rows, &
                       nblk, mpi_comm_rows, mpi_comm_cols, a, na_rows)
   ttt1 = MPI_Wtime()
   if(myid == 0) print *,'Time U**-T*A*U**-1:',ttt1-ttt0

   ! A is only set in the upper half, solve_evp_real needs a full matrix
   ! Set lower half from upper half

   call pdtran(na,na,1.d0,a,1,1,sc_desc,0.d0,tmp1,1,1,sc_desc)

   do i=1,na_cols
      ! Get global column corresponding to i and number of local rows up to
      ! and including the diagonal, these are unchanged in A
      n_col = indxl2g(i,     nblk, my_pcol, 0, np_cols)
      n_row = numroc (n_col, nblk, my_prow, 0, np_rows)
      a(n_row+1:na_rows,i) = tmp1(n_row+1:na_rows,i)
   enddo

   ! 3. Calculate eigenvalues/eigenvectors of U**-T * A * U**-1
   !    Eigenvectors go to tmp1

   call solve_evp_real(na, nev, a, na_rows, ev, tmp1, na_rows, nblk, &
                       mpi_comm_rows, mpi_comm_cols)

   if(myid == 0) print *,'Time tridiag_real :',time_evp_fwd
   if(myid == 0) print *,'Time solve_tridi  :',time_evp_solve
   if(myid == 0) print *,'Time trans_ev_real:',time_evp_back

   ! 4. Backtransform eigenvectors: Z = U**-1 * tmp1

   ttt0 = MPI_Wtime()
   ! mult_at_b_real needs the transpose of U**-1, thus tmp2 = (U**-1)**T
   call pdtran(na,na,1.d0,b,1,1,sc_desc,0.d0,tmp2,1,1,sc_desc)

   call mult_at_b_real('L', 'N', na, nev, tmp2, na_rows, tmp1, na_rows, &
                       nblk, mpi_comm_rows, mpi_comm_cols, z, na_rows)
   ttt1 = MPI_Wtime()
   if(myid == 0) print *,'Time Back U**-1*Z :',ttt1-ttt0

   !-------------------------------------------------------------------------------
   ! Print eigenvalues

   if(myid == 0) then
      do i=1,nev
         print '(i6,g25.15)',i,ev(i)
      enddo
   endif

   !-------------------------------------------------------------------------------
   ! Test correctness of result (using plain scalapack routines)

   ! 1. Residual (maximum of || A*Zi - B*Zi*EVi ||)

   ! tmp1 =  A * Z
   call pdgemm('N','N',na,nev,na,1.d0,as,1,1,sc_desc, &
               z,1,1,sc_desc,0.d0,tmp1,1,1,sc_desc)

   ! tmp2 = B*Zi*EVi
   call pdgemm('N','N',na,nev,na,1.d0,bs,1,1,sc_desc, &
               z,1,1,sc_desc,0.d0,tmp2,1,1,sc_desc)
   do i=1,nev
      call pdscal(na,ev(i),tmp2,1,i,sc_desc,1)
   enddo

   !  tmp1 = A*Zi - B*Zi*EVi
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

   ! 2. Eigenvector orthogonality

   ! tmp1 = Z**T * B * Z

   call pdgemm('N','N',na,nev,na,1.d0,bs,1,1,sc_desc, &
               z,1,1,sc_desc,0.d0,tmp2,1,1,sc_desc)
   tmp1 = 0
   call pdgemm('T','N',nev,nev,na,1.d0,z,1,1,sc_desc, &
               tmp2,1,1,sc_desc,0.d0,tmp1,1,1,sc_desc)

   ! Initialize tmp2 to unit matrix
   tmp2 = 0
   call pdlaset('A',nev,nev,0.d0,1.d0,tmp2,1,1,sc_desc)

   ! tmp1 = Z**T * B * Z - Unit Matrix
   tmp1(:,:) =  tmp1(:,:) - tmp2(:,:)

   ! Get maximum error (max abs value in tmp1)
   err = maxval(abs(tmp1))
   call mpi_allreduce(err,errmax,1,MPI_REAL8,MPI_MAX,MPI_COMM_WORLD,mpierr)
   if(myid==0) print *,'Error Orthogonality:',errmax

   call mpi_finalize(mpierr)


end program read_real_gen







!-------------------------------------------------------------------------------







subroutine read_matrix(iunit, fmttype, na, a, lda, nblk, my_prow, my_pcol, np_rows, np_cols)

   implicit none
   include 'mpif.h'

   integer, intent(in) :: iunit, na, lda, nblk, my_prow, my_pcol, np_rows, np_cols
   character(256), intent(in) :: fmttype
   real*8, intent(out) :: a(lda, *)

   integer i, j, lr, lc, myid, mpierr
   integer, allocatable :: l_row(:), l_col(:)

   real*8, allocatable :: col(:)

   ! allocate and set index arrays

   allocate(l_row(na))
   allocate(l_col(na))

   ! Mapping of global rows/cols to local

   l_row(:) = 0
   l_col(:) = 0

   lr = 0 ! local row counter
   lc = 0 ! local column counter

   do i = 1, na

     if( MOD((i-1)/nblk,np_rows) == my_prow) then
       ! row i is on local processor
       lr = lr+1
       l_row(i) = lr
     endif

     if( MOD((i-1)/nblk,np_cols) == my_pcol) then
       ! column i is on local processor
       lc = lc+1
       l_col(i) = lc
     endif

   enddo

   call mpi_comm_rank(mpi_comm_world,myid,mpierr)
   allocate(col(na))


   do i=1,na

      if (myid==0) then
        if ( trim(fmttype) == 'unformatted' ) then
          read (iunit  ) col(1:i)
        else
          read (iunit,*) col(1:i)
        endif
      endif

      call mpi_bcast(col,i,MPI_REAL8,0,MPI_COMM_WORLD,mpierr)

      if(l_col(i) > 0) then
         do j=1,i
            if(l_row(j)>0) a(l_row(j),l_col(i)) = col(j)
         enddo
      endif

      if(l_row(i) > 0) then
         do j=1,i-1
            if(l_col(j)>0) a(l_row(i),l_col(j)) = col(j)
         enddo
      endif

   enddo


   deallocate(l_row, l_col, col)


end subroutine read_matrix
