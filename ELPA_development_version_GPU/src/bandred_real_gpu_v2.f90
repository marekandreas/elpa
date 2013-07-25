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

!use mpi
use cublas 
use elpa1
   implicit none

include "mpif.h"

   integer na, lda, nblk, nbw, mpi_comm_rows, mpi_comm_cols
   real*8 a(lda,*), tmat(nbw,nbw,*)

   integer my_prow, my_pcol, np_rows, np_cols, ierr
   integer l_cols, l_rows
   integer i, j, lcs, lce, lre, lc, lr, cur_pcol, n_cols, nrow
   integer istep, ncol, lch, lcx, nlc
   integer tile_size, l_rows_tile, l_cols_tile
   integer work_size

   real*8 eps

   real*8 vnorm2, xf, aux1(nbw), aux2(nbw), vrl, tau, vav(nbw,nbw)

   real*8, allocatable:: tmp(:), vr(:)
   real*8, allocatable, pinned :: vmr(:), umc(:)
   real*8, allocatable:: work(:) ! needed for blocked QR

   real*8, allocatable, device:: a_dev(:,:), vmr_dev(:), umc_dev(:), tmat_dev(:,:)
   real*8, allocatable, device:: vav_dev(:,:)
!   real*8, allocatable, pinned:: a_tmp(:)

   type(C_DEVPTR) :: a_dev_ptr

   integer, external :: numroc
   integer :: na_rows, na_cols
   integer :: istat
   integer :: cur_l_rows, cur_l_cols, vmr_size, umc_size

real*8 times(100), ttt0, ttts, ttt9

   integer pcol, prow
   integer  pid
  
   integer lc_start, lc_end
   integer lr_end

   integer myid, mpierr

   pcol(i) = MOD((i-1)/nblk,np_cols) !Processor col for global col number
   prow(i) = MOD((i-1)/nblk,np_rows) !Processor row for global row number

ttts = mpi_wtime()
times(:) = 0

ttt0 = mpi_wtime()
   call mpi_comm_rank(mpi_comm_rows,my_prow,ierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,ierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,ierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,ierr)

   call mpi_comm_rank(MPI_COMM_WORLD,pid,ierr)

   ! Semibandwith nbw must be a multiple of blocksize nblk


   if(mod(nbw,nblk)/=0) then
      if(my_prow==0 .and. my_pcol==0) then
         print *,'ERROR: nbw=',nbw,', nblk=',nblk
         print *,'ELPA2 works only for nbw==n*nblk'
         call mpi_abort(mpi_comm_world,0,ierr)
      endif
   endif

   na_rows = numroc(na, nblk, my_prow, 0, np_rows)
   na_cols = numroc(na, nblk, my_pcol, 0, np_cols)

   ! Matrix is split into tiles; work is done only for tiles on the diagonal or above

   tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size
   tile_size = ((128*max(np_rows,np_cols)-1)/tile_size+1)*tile_size ! make local tiles at least 128 wide

   l_rows_tile = tile_size/np_rows ! local rows of a tile
   l_cols_tile = tile_size/np_cols ! local cols of a tile


   ! Here we convert the regular host array into a pinned host array
   istat = cudaHostRegister(C_LOC(a(1,1)), 8*lda * na_cols, 0)
   if(istat.ne.0) print*, "HostRegister failed", istat

   allocate(a_dev(lda, na_cols), stat=istat)
   allocate(tmat_dev(nbw, nbw), vav_dev(nbw, nbw))
 
   cur_l_rows = 0
   cur_l_cols = 0


   a_dev(1:lda, 1:na_cols) = a(1:lda, 1:na_cols)

times(1) = times(1) + mpi_wtime()-ttt0

   do istep = (na - 1) / nbw, 1, -1

      n_cols = MIN(na, (istep + 1) * nbw) - istep * nbw ! Number of columns in current step

      ! Number of local columns/rows of remaining matrix
      l_cols = local_index(istep * nbw, my_pcol, np_cols, nblk, -1)
      l_rows = local_index(istep * nbw, my_prow, np_rows, nblk, -1)

ttt0 = mpi_wtime()

        cur_l_rows = max(l_rows, 1)
        cur_l_cols = max(l_cols, 1)
         
        vmr_size = cur_l_rows * 2 * n_cols
        umc_size = cur_l_cols * 2 * n_cols

        ! Allocate vmr and umc only if the new size exceeds their current capacity
        if ((.not. allocated(vr)) .or. (l_rows + 1 .gt. ubound(vr, 1))) then
          if (allocated(vr)) deallocate(vr)
          allocate(vr(l_rows + 1))
        endif

        if ((.not. allocated(vmr)) .or. (vmr_size .gt. ubound(vmr, 1))) then
            if (allocated(vmr)) then
                deallocate(vmr)
                deallocate(vmr_dev)
            endif
            allocate(vmr(vmr_size))
            allocate(vmr_dev(vmr_size))
        endif

        if ((.not. allocated(umc)) .or. (umc_size .gt. ubound(umc, 1))) then
            if (allocated(umc)) then
                deallocate(umc)
                deallocate(umc_dev)
            endif
            allocate(umc(umc_size))
            allocate(umc_dev(umc_size))
        endif

times(2) = times(2) + mpi_wtime()-ttt0
ttt0 = mpi_wtime()

      vmr(1 : cur_l_rows * n_cols) = 0.
      vr(:) = 0
      tmat(:,:,istep) = 0

      umc(1 : umc_size) = 0.

      lc_start = local_index(istep*nbw+1, my_pcol, np_cols, nblk, -1)
      lc_end   = local_index(istep*nbw+n_cols, my_pcol, np_cols, nblk, -1)
      lr_end   = local_index((istep-1)*nbw + n_cols, my_prow, np_rows, nblk, -1)
  
      if(lc_start .le. 0) lc_start = 1

      ! Here we assume that the processor grid and the block grid are aligned
      cur_pcol = pcol(istep*nbw+1)
      call mpi_comm_rank(mpi_comm_world,myid,mpierr)

      if(my_pcol == cur_pcol) then
        istat = cudaMemcpy2D(a(1, lc_start), lda, a_dev(1,lc_start), lda, &
                             lr_end, (lc_end - lc_start+1))
        if(istat .ne. 0) print *, "data transfer error in cudaMemcpy2d",myid
      endif

times(3) = times(3) + mpi_wtime()-ttt0

      do lc = n_cols, 1, -1

ttt0 = mpi_wtime()
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

            call mpi_allreduce(aux1,aux2,2,MPI_REAL8,MPI_SUM,mpi_comm_rows,ierr)

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

times(4) = times(4) + mpi_wtime()-ttt0
ttt0 = mpi_wtime()

         vr(lr+1) = tau

         call MPI_Bcast(vr,lr+1,MPI_REAL8,cur_pcol,mpi_comm_cols,ierr)

         vmr(cur_l_rows * (lc - 1) + 1 : cur_l_rows * (lc - 1) + lr) = vr(1:lr)
         tau = vr(lr+1)
         tmat(lc,lc,istep) = tau ! Store tau in diagonal of tmat

         ! Transform remaining columns in current block with Householder vector

         ! Local dot product

times(5) = times(5) + mpi_wtime()-ttt0
ttt0 = mpi_wtime()

         aux1 = 0

         !  the followin loop should only enter into the dot procudt
         !   for processors in the current col 
         nlc = 0 ! number of local columns
         do j=1,lc-1
            lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
            if(lcx>0) then
               nlc = nlc+1
               if(lr>0) then
         !           a(1:lr, lcx) = a_dev(1:lr, lcx)
                    aux1(nlc) = dot_product(vr(1:lr),a(1:lr,lcx))
               endif
            endif
         enddo

         ! Get global dot products
         if(nlc>0) call mpi_allreduce(aux1,aux2,nlc,MPI_REAL8,MPI_SUM,mpi_comm_rows,ierr)

         ! Transform

         nlc = 0
         do j=1,lc-1
            lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
            if(lcx>0) then
               nlc = nlc+1
               a(1:lr,lcx) = a(1:lr,lcx) - tau*aux2(nlc)*vr(1:lr)
            endif
         enddo

times(6) = times(6) + mpi_wtime()-ttt0

      enddo

ttt0 = mpi_wtime()
      call mpi_comm_rank(mpi_comm_world,myid,mpierr)

! store column tiles back to GPU
      cur_pcol = pcol(istep*nbw+1)
      if(my_pcol == cur_pcol) then
        istat = cudaMemcpy2DAsync(a_dev(1,lc_start), lda, a(1, lc_start), lda,  lr_end, (lc_end - lc_start+1))
        if(istat .ne. 0) print *, "data transfer error cudaMemcpy2dAsync",myid
      endif

times(7) = times(7) + mpi_wtime()-ttt0
ttt0 = mpi_wtime()
    
      ! Calculate scalar products of stored Householder vectors.
      ! This can be done in different ways, we use dsyrk

      vav = 0
      if(l_rows>0) &
         call dsyrk('U','T',n_cols,l_rows,1.d0,vmr,cur_l_rows,0.d0,vav,ubound(vav,1))
      call symm_matrix_allreduce(n_cols,vav,ubound(vav,1),mpi_comm_rows)

      ! Calculate triangular matrix T for block Householder Transformation

      do lc=n_cols,1,-1
         tau = tmat(lc,lc,istep)
         if(lc<n_cols) then
            call dtrmv('U','T','N',n_cols-lc,tmat(lc+1,lc+1,istep),ubound(tmat,1),vav(lc+1,lc),1)
            tmat(lc,lc+1:n_cols,istep) = -tau * vav(lc+1:n_cols,lc)
         endif
      enddo

      tmat_dev(:,:) = tmat(:,:,istep)

      ! Transpose vmr -> vmc (stored in umc, second half)

      call elpa_transpose_vectors  (vmr, cur_l_rows, mpi_comm_rows, &
                                    umc(cur_l_cols * n_cols + 1), cur_l_cols, mpi_comm_cols, &
                                    1, istep*nbw, n_cols, nblk)
times(8) = times(8) + mpi_wtime()-ttt0
ttt0 = mpi_wtime()

      ! Calculate umc = A**T * vmr
      ! Note that the distributed A has to be transposed
      ! Opposed to direct tridiagonalization there is no need to use the cache locality
      ! of the tiles, so we can use strips of the matrix

      umc(1 : l_cols * n_cols) = 0.d0
      vmr(cur_l_rows * n_cols + 1 : cur_l_rows * n_cols * 2) = 0


      if(l_cols>0 .and. l_rows>0) then

         ierr = cudaMemcpy(vmr_dev, vmr, vmr_size)
         ierr = cudaMemcpy(umc_dev, umc, umc_size)

         do i=0,(istep*nbw-1)/tile_size

            lcs = i*l_cols_tile+1
            lce = min(l_cols,(i+1)*l_cols_tile)
            if(lce<lcs) cycle

            lre = min(l_rows,(i+1)*l_rows_tile)

            call cublasDGEMM('T','N',lce-lcs+1,n_cols,lre, &
                            1.d0, a_dev(1,lcs), ubound(a_dev,1), vmr_dev,cur_l_rows, &
                            1.d0, umc_dev(lcs), cur_l_cols)

            if(i==0) cycle
            lre = min(l_rows,i*l_rows_tile)

            call cublasDGEMM('N','N',lre,n_cols,lce-lcs+1,&
                       1.d0, a_dev(1,lcs),lda, umc_dev(cur_l_cols * n_cols + lcs), cur_l_cols, &
                       1.d0, vmr_dev(cur_l_rows * n_cols + 1), cur_l_rows)
         enddo

         ierr = cudaMemcpy(vmr, vmr_dev, vmr_size)
         ierr = cudaMemcpy(umc, umc_dev, umc_size)

      endif

times(9) = times(9) + mpi_wtime()-ttt0
ttt0 = mpi_wtime()

      ! Sum up all ur(:) parts along rows and add them to the uc(:) parts
      ! on the processors containing the diagonal
      ! This is only necessary if ur has been calculated, i.e. if the
      ! global tile size is smaller than the global remaining matrix

      if(tile_size < istep*nbw) then
         call elpa_reduce_add_vectors  (vmr(cur_l_rows * n_cols + 1),cur_l_rows,mpi_comm_rows, &
                                        umc, cur_l_cols, mpi_comm_cols, &
                                        istep*nbw, n_cols, nblk)
      endif

      if(l_cols>0) then
         allocate(tmp(l_cols * n_cols))
         call mpi_allreduce(umc,tmp,l_cols*n_cols,MPI_REAL8,MPI_SUM,mpi_comm_rows,ierr)
         umc(1 : l_cols * n_cols) = tmp(1 : l_cols * n_cols)
         deallocate(tmp)
      endif
times(10) = times(10) + mpi_wtime()-ttt0
ttt0 = mpi_wtime()

      ! U = U * Tmat**T

      ierr = cudaMemcpy(umc_dev, umc, umc_size) 

      call cublasDTRMM('Right','Upper','Trans','Nonunit',l_cols,n_cols, &
                    1.d0, tmat_dev(1,1),ubound(tmat_dev,1),umc_dev,cur_l_cols)

      ! VAV = Tmat * V**T * A * V * Tmat**T = (U*Tmat**T)**T * V * Tmat**T

      call cublasDGEMM('T','N',n_cols,n_cols,l_cols, &
                    1.d0, umc_dev,cur_l_cols,umc_dev(cur_l_cols * n_cols + 1),cur_l_cols, &
                    0.d0, vav_dev,ubound(vav_dev,1))

      call cublasDTRMM('Right','Upper','Trans','Nonunit',n_cols,n_cols, &
                     1.d0, tmat_dev(1,1),ubound(tmat_dev,1), vav_dev, ubound(vav_dev,1))


      vav = vav_dev

      call symm_matrix_allreduce(n_cols,vav,ubound(vav,1),mpi_comm_cols)

      vav_dev = vav


      ! U = U - 0.5 * V * VAV

      call cublasDGEMM('N','N',l_cols,n_cols,n_cols,&
                      -0.5d0, umc_dev(cur_l_cols * n_cols + 1),cur_l_cols, vav_dev,ubound(vav_dev,1),&
                       1.0d0, umc_dev,cur_l_cols)

      ierr = cudaMemcpy(umc, umc_dev, umc_size)

      ! Transpose umc -> umr (stored in vmr, second half)

       call elpa_transpose_vectors  (umc, cur_l_cols, mpi_comm_cols, &
                                     vmr(cur_l_rows * n_cols + 1), cur_l_rows, mpi_comm_rows, &
                                     1, istep*nbw, n_cols, nblk)
       ierr = cudaMemcpy(vmr_dev, vmr, vmr_size)
       ierr = cudaMemcpy(umc_dev, umc, umc_size)

times(11) = times(11) + mpi_wtime()-ttt0
ttt0 = mpi_wtime()
     
      ! A = A - V*U**T - U*V**T

      do i=0,(istep*nbw-1)/tile_size
         lcs = i*l_cols_tile+1
         lce = min(l_cols,(i+1)*l_cols_tile)
         lre = min(l_rows,(i+1)*l_rows_tile)
         if(lce<lcs .or. lre<1) cycle

         call cublasDGEMM('N', 'T', lre, lce-lcs+1, 2*n_cols, -1.d0, &
                    vmr_dev,cur_l_rows,umc_dev(lcs),cur_l_cols, &
                    1.d0,a_dev(1,lcs),lda)
                    
      enddo

times(12) = times(12) + mpi_wtime()-ttt0


   enddo

   a(1:lda, 1:na_cols)   = a_dev(1:lda, 1:na_cols)

   ! Free used memory
   if (allocated(vr)) deallocate(vr)

   if (allocated(vmr)) then
        deallocate(vmr)
        deallocate(vmr_dev)
   endif
   
   if (allocated(umc)) then
        deallocate(umc)
        deallocate(umc_dev)
   endif

ttts = mpi_wtime()-ttts

print '("Times: ",15f7.2)',times(1:12),ttts-sum(times(1:12)),ttts

end subroutine



subroutine symm_matrix_allreduce(n,a,lda,comm)

!-------------------------------------------------------------------------------
!  symm_matrix_allreduce: Does an mpi_allreduce for a symmetric matrix A.
!  On entry, only the upper half of A needs to be set
!  On exit, the complete matrix is set
!-------------------------------------------------------------------------------

   implicit none
   include "mpif.h"
   integer n, lda, comm
   real*8 a(lda,*)

   integer i, nc, ierr
   real*8 h1(n*n), h2(n*n)

   nc = 0
   do i=1,n
      h1(nc+1:nc+i) = a(1:i,i)
      nc = nc+i
   enddo

   call mpi_allreduce(h1,h2,nc,MPI_REAL8,MPI_SUM,comm,ierr)

   nc = 0
   do i=1,n
      a(1:i,i) = h2(nc+1:nc+i)
      a(i,1:i-1) = a(1:i-1,i)
      nc = nc+i
   enddo

end subroutine symm_matrix_allreduce
