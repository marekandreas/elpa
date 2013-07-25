subroutine bandred_real(na, a, lda, nblk, nbw, mpi_comm_rows, mpi_comm_cols, tmat)

!-------------------------------------------------------------------------------
!  bandred_real: Reduces a distributed symmetric matrix to band form
!
!  Parameters
!
!  na          Order of matrix
!
!  a(lda,*)    Distributed matrix which should be reduced.
!              Distribution is like in Scalapack.
!              Opposed to Scalapack, a(:,:) must be set completely (upper and lower half)
!              a(:,:) is overwritten on exit with the band and the Householder vectors
!              in the upper half.
!
!  lda         Leading dimension of a
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  nbw         semi bandwith of output matrix
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!
!  tmat(nbw,nbw,num_blocks)    where num_blocks = (na-1)/nbw + 1
!              Factors for the Householder vectors (returned), needed for back transformation
!
!-------------------------------------------------------------------------------

   implicit none

   integer na, lda, nblk, nbw, mpi_comm_rows, mpi_comm_cols
   real*8 a(lda,*), tmat(nbw,nbw,*)

   integer my_prow, my_pcol, np_rows, np_cols, mpierr
   integer l_cols, l_rows
   integer i, j, lcs, lce, lre, lc, lr, cur_pcol, n_cols, nrow
   integer istep, ncol, lch, lcx, nlc
   integer tile_size, l_rows_tile, l_cols_tile

   real*8 vnorm2, xf, aux1(nbw), aux2(nbw), vrl, tau, vav(nbw,nbw)

   real*8, allocatable:: tmp(:,:), vr(:), vmr(:,:), umc(:,:)





    !real*8 m1(1:4, 1:4), m2(1:4, 1:4), m3(1:4, 1:4)

    !m1(:, 1) = 1
    !m1(:, 2) = 2
    !m1(:, 3) = 3
    !m1(:, 4) = 4

    !m2 = m1

    !call DGEMM('T','N',lce-lcs+1,n_cols,lre,1.d0,a(1,lcs),ubound(a,1), &
    !                   vmr,ubound(vmr,1),1.d0,umc(lcs,1),ubound(umc,1))

    !print *, 'm1 = '
    










   integer pcol, prow
   pcol(i) = MOD((i-1)/nblk,np_cols) !Processor col for global col number
   prow(i) = MOD((i-1)/nblk,np_rows) !Processor row for global row number


   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   ! Semibandwith nbw must be a multiple of blocksize nblk

   if(mod(nbw,nblk)/=0) then
      if(my_prow==0 .and. my_pcol==0) then
         print *,'ERROR: nbw=',nbw,', nblk=',nblk
         print *,'ELPA2 works only for nbw==n*nblk'
         call mpi_abort(mpi_comm_world,0,mpierr)
      endif
   endif

   ! Matrix is split into tiles; work is done only for tiles on the diagonal or above

   tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size
   tile_size = ((128*max(np_rows,np_cols)-1)/tile_size+1)*tile_size ! make local tiles at least 128 wide

   l_rows_tile = tile_size/np_rows ! local rows of a tile
   l_cols_tile = tile_size/np_cols ! local cols of a tile

   do istep = (na-1)/nbw, 1, -1

      n_cols = MIN(na,(istep+1)*nbw) - istep*nbw ! Number of columns in current step

      ! Number of local columns/rows of remaining matrix
      l_cols = local_index(istep*nbw, my_pcol, np_cols, nblk, -1)
      l_rows = local_index(istep*nbw, my_prow, np_rows, nblk, -1)

      ! Allocate vmr and umc to their exact sizes so that they can be used in bcasts and reduces

      allocate(vmr(max(l_rows,1),2*n_cols))
      allocate(umc(max(l_cols,1),2*n_cols))

      allocate(vr(l_rows+1))

      vmr(1:l_rows,1:n_cols) = 0.
      vr(:) = 0
      tmat(:,:,istep) = 0

      ! Reduce current block to lower triangular form

      do lc = n_cols, 1, -1

         ncol = istep*nbw + lc ! absolute column number of householder vector
         nrow = ncol - nbw ! Absolute number of pivot row

         lr  = local_index(nrow, my_prow, np_rows, nblk, -1) ! current row length
         lch = local_index(ncol, my_pcol, np_cols, nblk, -1) ! HV local column number

         tau = 0

         if(nrow == 1) exit ! Nothing to do

         cur_pcol = pcol(ncol) ! Processor column owning current block

         if(my_pcol==cur_pcol) then

            ! Get vector to be transformed; distribute last element and norm of
            ! remaining elements to all procs in current column

            vr(1:lr) = a(1:lr,lch) ! vector to be transformed

            if(my_prow==prow(nrow)) then
               aux1(1) = dot_product(vr(1:lr-1),vr(1:lr-1))
               aux1(2) = vr(lr)
            else
               aux1(1) = dot_product(vr(1:lr),vr(1:lr))
               aux1(2) = 0.
            endif

            call mpi_allreduce(aux1,aux2,2,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)

            vnorm2 = aux2(1)
            vrl    = aux2(2)

            ! Householder transformation

            call hh_transform_real(vrl, vnorm2, xf, tau)

            ! Scale vr and store Householder vector for back transformation

            vr(1:lr) = vr(1:lr) * xf
            if(my_prow==prow(nrow)) then
               a(1:lr-1,lch) = vr(1:lr-1)
               a(lr,lch) = vrl
               vr(lr) = 1.
            else
               a(1:lr,lch) = vr(1:lr)
            endif

         endif

         ! Broadcast Householder vector and tau along columns

         vr(lr+1) = tau
         call MPI_Bcast(vr,lr+1,MPI_REAL8,cur_pcol,mpi_comm_cols,mpierr)
         vmr(1:lr,lc) = vr(1:lr)
         tau = vr(lr+1)
         tmat(lc,lc,istep) = tau ! Store tau in diagonal of tmat

         ! Transform remaining columns in current block with Householder vector

         ! Local dot product

         aux1 = 0

         nlc = 0 ! number of local columns
         do j=1,lc-1
            lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
            if(lcx>0) then
               nlc = nlc+1
               if(lr>0) aux1(nlc) = dot_product(vr(1:lr),a(1:lr,lcx))
            endif
         enddo

         ! Get global dot products
         if(nlc>0) call mpi_allreduce(aux1,aux2,nlc,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)

         ! Transform

         nlc = 0
         do j=1,lc-1
            lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
            if(lcx>0) then
               nlc = nlc+1
               a(1:lr,lcx) = a(1:lr,lcx) - tau*aux2(nlc)*vr(1:lr)
            endif
         enddo

      enddo

      ! Calculate scalar products of stored Householder vectors.
      ! This can be done in different ways, we use dsyrk

      vav = 0
      if(l_rows>0) &
         call dsyrk('U','T',n_cols,l_rows,1.d0,vmr,ubound(vmr,1),0.d0,vav,ubound(vav,1))
      call symm_matrix_allreduce(n_cols,vav,ubound(vav,1),mpi_comm_rows)

      ! Calculate triangular matrix T for block Householder Transformation

      do lc=n_cols,1,-1
         tau = tmat(lc,lc,istep)
         if(lc<n_cols) then
            call dtrmv('U','T','N',n_cols-lc,tmat(lc+1,lc+1,istep),ubound(tmat,1),vav(lc+1,lc),1)
            tmat(lc,lc+1:n_cols,istep) = -tau * vav(lc+1:n_cols,lc)
         endif
      enddo

      ! Transpose vmr -> vmc (stored in umc, second half)

      call elpa_transpose_vectors  (vmr, ubound(vmr,1), mpi_comm_rows, &
                                    umc(1,n_cols+1), ubound(umc,1), mpi_comm_cols, &
                                    1, istep*nbw, n_cols, nblk)

      ! Calculate umc = A**T * vmr
      ! Note that the distributed A has to be transposed
      ! Opposed to direct tridiagonalization there is no need to use the cache locality
      ! of the tiles, so we can use strips of the matrix

      umc(1:l_cols,1:n_cols) = 0.d0
      vmr(1:l_rows,n_cols+1:2*n_cols) = 0
      if(l_cols>0 .and. l_rows>0) then
         do i=0,(istep*nbw-1)/tile_size

            lcs = i*l_cols_tile+1
            lce = min(l_cols,(i+1)*l_cols_tile)
            if(lce<lcs) cycle

            lre = min(l_rows,(i+1)*l_rows_tile)
            call DGEMM('T','N',lce-lcs+1,n_cols,lre,1.d0,a(1,lcs),ubound(a,1), &
                       vmr,ubound(vmr,1),1.d0,umc(lcs,1),ubound(umc,1))

            if(i==0) cycle
            lre = min(l_rows,i*l_rows_tile)
            call DGEMM('N','N',lre,n_cols,lce-lcs+1,1.d0,a(1,lcs),lda, &
                       umc(lcs,n_cols+1),ubound(umc,1),1.d0,vmr(1,n_cols+1),ubound(vmr,1))
         enddo
      endif

      ! Sum up all ur(:) parts along rows and add them to the uc(:) parts
      ! on the processors containing the diagonal
      ! This is only necessary if ur has been calculated, i.e. if the
      ! global tile size is smaller than the global remaining matrix

      if(tile_size < istep*nbw) then
         call elpa_reduce_add_vectors  (vmr(1,n_cols+1),ubound(vmr,1),mpi_comm_rows, &
                                        umc, ubound(umc,1), mpi_comm_cols, &
                                        istep*nbw, n_cols, nblk)
      endif

      if(l_cols>0) then
         allocate(tmp(l_cols,n_cols))
         call mpi_allreduce(umc,tmp,l_cols*n_cols,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)
         umc(1:l_cols,1:n_cols) = tmp(1:l_cols,1:n_cols)
         deallocate(tmp)
      endif

      ! U = U * Tmat**T

      call dtrmm('Right','Upper','Trans','Nonunit',l_cols,n_cols,1.d0,tmat(1,1,istep),ubound(tmat,1),umc,ubound(umc,1))

      ! VAV = Tmat * V**T * A * V * Tmat**T = (U*Tmat**T)**T * V * Tmat**T

      call dgemm('T','N',n_cols,n_cols,l_cols,1.d0,umc,ubound(umc,1),umc(1,n_cols+1),ubound(umc,1),0.d0,vav,ubound(vav,1))
      call dtrmm('Right','Upper','Trans','Nonunit',n_cols,n_cols,1.d0,tmat(1,1,istep),ubound(tmat,1),vav,ubound(vav,1))

      call symm_matrix_allreduce(n_cols,vav,ubound(vav,1),mpi_comm_cols)

      ! U = U - 0.5 * V * VAV
      call dgemm('N','N',l_cols,n_cols,n_cols,-0.5d0,umc(1,n_cols+1),ubound(umc,1),vav,ubound(vav,1),1.d0,umc,ubound(umc,1))

      ! Transpose umc -> umr (stored in vmr, second half)

       call elpa_transpose_vectors  (umc, ubound(umc,1), mpi_comm_cols, &
                                     vmr(1,n_cols+1), ubound(vmr,1), mpi_comm_rows, &
                                     1, istep*nbw, n_cols, nblk)

      ! A = A - V*U**T - U*V**T

      do i=0,(istep*nbw-1)/tile_size
         lcs = i*l_cols_tile+1
         lce = min(l_cols,(i+1)*l_cols_tile)
         lre = min(l_rows,(i+1)*l_rows_tile)
         if(lce<lcs .or. lre<1) cycle
         call dgemm('N','T',lre,lce-lcs+1,2*n_cols,-1.d0, &
                    vmr,ubound(vmr,1),umc(lcs,1),ubound(umc,1), &
                    1.d0,a(1,lcs),lda)
      enddo

      deallocate(vmr, umc, vr)

   enddo

end subroutine bandred_real

!-------------------------------------------------------------------------------

subroutine symm_matrix_allreduce(n,a,lda,comm)

!-------------------------------------------------------------------------------
!  symm_matrix_allreduce: Does an mpi_allreduce for a symmetric matrix A.
!  On entry, only the upper half of A needs to be set
!  On exit, the complete matrix is set
!-------------------------------------------------------------------------------

   implicit none
   integer n, lda, comm
   real*8 a(lda,*)

   integer i, nc, mpierr
   real*8 h1(n*n), h2(n*n)

   nc = 0
   do i=1,n
      h1(nc+1:nc+i) = a(1:i,i)
      nc = nc+i
   enddo

   call mpi_allreduce(h1,h2,nc,MPI_REAL8,MPI_SUM,comm,mpierr)

   nc = 0
   do i=1,n
      a(1:i,i) = h2(nc+1:nc+i)
      a(i,1:i-1) = a(1:i-1,i)
      nc = nc+i
   enddo

end subroutine symm_matrix_allreduce

