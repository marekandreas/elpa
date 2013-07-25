subroutine trans_ev_band_to_full_real(na, nqc, nblk, nbw, a, lda, tmat, q, ldq, mpi_comm_rows, mpi_comm_cols)

!-------------------------------------------------------------------------------
!  trans_ev_band_to_full_real:
!  Transforms the eigenvectors of a band matrix back to the eigenvectors of the original matrix
!
!  Parameters
!
!  na          Order of matrix a, number of rows of matrix q
!
!  nqc         Number of columns of matrix q
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  nbw         semi bandwith
!
!  a(lda,*)    Matrix containing the Householder vectors (i.e. matrix a after bandred_real)
!              Distribution is like in Scalapack.
!
!  lda         Leading dimension of a
!
!  tmat(nbw,nbw,.) Factors returned by bandred_real
!
!  q           On input: Eigenvectors of band matrix
!              On output: Transformed eigenvectors
!              Distribution is like in Scalapack.
!
!  ldq         Leading dimension of q
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!
!-------------------------------------------------------------------------------

   implicit none

   integer na, nqc, lda, ldq, nblk, nbw, mpi_comm_rows, mpi_comm_cols
   real*8 a(lda,*), q(ldq,*), tmat(nbw, nbw, *)

   integer my_prow, my_pcol, np_rows, np_cols, mpierr
   integer max_blocks_row, max_blocks_col, max_local_rows, max_local_cols
   integer l_cols, l_rows, l_colh, n_cols
   integer istep, lc, ncol, nrow, nb, ns

   real*8, allocatable:: tmp1(:), tmp2(:), hvb(:), hvm(:,:)

   integer pcol, prow, i
   pcol(i) = MOD((i-1)/nblk,np_cols) !Processor col for global col number
   prow(i) = MOD((i-1)/nblk,np_rows) !Processor row for global row number


   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)


   max_blocks_row = ((na -1)/nblk)/np_rows + 1  ! Rows of A
   max_blocks_col = ((nqc-1)/nblk)/np_cols + 1  ! Columns of q!

   max_local_rows = max_blocks_row*nblk
   max_local_cols = max_blocks_col*nblk

   allocate(tmp1(max_local_cols*nbw))
   allocate(tmp2(max_local_cols*nbw))
   allocate(hvb(max_local_rows*nbw))
   allocate(hvm(max_local_rows,nbw))

   hvm = 0   ! Must be set to 0 !!!
   hvb = 0   ! Safety only

   l_cols = local_index(nqc, my_pcol, np_cols, nblk, -1) ! Local columns of q

   do istep=1,(na-1)/nbw

      n_cols = MIN(na,(istep+1)*nbw) - istep*nbw ! Number of columns in current step

      ! Broadcast all Householder vectors for current step compressed in hvb

      nb = 0
      ns = 0

      do lc = 1, n_cols
         ncol = istep*nbw + lc ! absolute column number of householder vector
         nrow = ncol - nbw ! absolute number of pivot row

         l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast
         l_colh = local_index(ncol  , my_pcol, np_cols, nblk, -1) ! HV local column number

         if(my_pcol==pcol(ncol)) hvb(nb+1:nb+l_rows) = a(1:l_rows,l_colh)

         nb = nb+l_rows

         if(lc==n_cols .or. mod(ncol,nblk)==0) then
            call MPI_Bcast(hvb(ns+1),nb-ns,MPI_REAL8,pcol(ncol),mpi_comm_cols,mpierr)
            ns = nb
         endif
      enddo

      ! Expand compressed Householder vectors into matrix hvm

      nb = 0
      do lc = 1, n_cols
         nrow = (istep-1)*nbw+lc ! absolute number of pivot row
         l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast

         hvm(1:l_rows,lc) = hvb(nb+1:nb+l_rows)
         if(my_prow==prow(nrow)) hvm(l_rows+1,lc) = 1.

         nb = nb+l_rows
      enddo

      l_rows = local_index(MIN(na,(istep+1)*nbw), my_prow, np_rows, nblk, -1)

      ! Q = Q - V * T**T * V**T * Q

      if(l_rows>0) then
         call dgemm('T','N',n_cols,l_cols,l_rows,1.d0,hvm,ubound(hvm,1), &
                    q,ldq,0.d0,tmp1,n_cols)
      else
         tmp1(1:l_cols*n_cols) = 0
      endif
      call mpi_allreduce(tmp1,tmp2,n_cols*l_cols,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)
      if(l_rows>0) then
         call dtrmm('L','U','T','N',n_cols,l_cols,1.0d0,tmat(1,1,istep),ubound(tmat,1),tmp2,n_cols)
         call dgemm('N','N',l_rows,l_cols,n_cols,-1.d0,hvm,ubound(hvm,1), &
                    tmp2,n_cols,1.d0,q,ldq)
      endif

   enddo

   deallocate(tmp1, tmp2, hvb, hvm)


end subroutine trans_ev_band_to_full_real
