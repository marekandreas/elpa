subroutine tridiag_band_real(na, nb, nblk, a, lda, d, e, mpi_comm_rows, mpi_comm_cols, mpi_comm)

!-------------------------------------------------------------------------------
! tridiag_band_real:
! Reduces a real symmetric band matrix to tridiagonal form
!
!  na          Order of matrix a
!
!  nb          Semi bandwith
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  a(lda,*)    Distributed system matrix reduced to banded form in the upper diagonal
!
!  lda         Leading dimension of a
!
!  d(na)       Diagonal of tridiagonal matrix, set only on PE 0 (output)
!
!  e(na)       Subdiagonal of tridiagonal matrix, set only on PE 0 (output)
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!  mpi_comm
!              MPI-Communicator for the total processor set
!-------------------------------------------------------------------------------

   implicit none

   integer, intent(in) ::  na, nb, nblk, lda, mpi_comm_rows, mpi_comm_cols, mpi_comm
   real*8, intent(in)  :: a(lda,*)
   real*8, intent(out) :: d(na), e(na) ! set only on PE 0


   real*8 vnorm2, hv(nb), tau, x, h(nb), ab_s(1+nb), hv_s(nb), hv_new(nb), tau_new, hf
   real*8 hd(nb), hs(nb)

   integer i, j, n, nc, nr, ns, ne, istep, iblk, nblocks_total, nblocks, nt
   integer my_pe, n_pes, mpierr
   integer my_prow, np_rows, my_pcol, np_cols
   integer ireq_ab, ireq_hv
   integer na_s, nx, num_hh_vecs, num_chunks, local_size, max_blk_size, n_off
   integer, allocatable :: ireq_hhr(:), ireq_hhs(:), global_id(:,:), hh_cnt(:), hh_dst(:)
   integer, allocatable :: limits(:), snd_limits(:,:)
   integer, allocatable :: block_limits(:)
   real*8, allocatable :: ab(:,:), hh_gath(:,:,:), hh_send(:,:,:)
   ! dummies for calling redist_band
   complex*16 :: c_a(1,1), c_ab(1,1)


   call mpi_comm_rank(mpi_comm,my_pe,mpierr)
   call mpi_comm_size(mpi_comm,n_pes,mpierr)

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   ! Get global_id mapping 2D procssor coordinates to global id

   allocate(global_id(0:np_rows-1,0:np_cols-1))
   global_id(:,:) = 0
   global_id(my_prow, my_pcol) = my_pe

   call mpi_allreduce(mpi_in_place, global_id, np_rows*np_cols, mpi_integer, mpi_sum, mpi_comm, mpierr)


   ! Total number of blocks in the band:

   nblocks_total = (na-1)/nb + 1

   ! Set work distribution

   allocate(block_limits(0:n_pes))
   call divide_band(nblocks_total, n_pes, block_limits)

   ! nblocks: the number of blocks for my task
   nblocks = block_limits(my_pe+1) - block_limits(my_pe)

   ! allocate the part of the band matrix which is needed by this PE
   ! The size is 1 block larger than needed to avoid extensive shifts
   allocate(ab(2*nb,(nblocks+1)*nb))
   ab = 0 ! needed for lower half, the extra block should also be set to 0 for safety

   ! n_off: Offset of ab within band
   n_off = block_limits(my_pe)*nb

   ! Redistribute band in a to ab
   call redist_band(.true., a, c_a, lda, na, nblk, nb, mpi_comm_rows, mpi_comm_cols, mpi_comm, ab, c_ab)

   ! Calculate the workload for each sweep in the back transformation
   ! and the space requirements to hold the HH vectors

   allocate(limits(0:np_rows))
   call determine_workload(na, nb, np_rows, limits)
   max_blk_size = maxval(limits(1:np_rows) - limits(0:np_rows-1))

   num_hh_vecs = 0
   num_chunks  = 0
   nx = na
   do n = 1, nblocks_total
      call determine_workload(nx, nb, np_rows, limits)
      local_size = limits(my_prow+1) - limits(my_prow)
      ! add to number of householder vectors
      ! please note: for nx==1 the one and only HH vector is 0 and is neither calculated nor send below!
      if(mod(n-1,np_cols) == my_pcol .and. local_size>0 .and. nx>1) then
         num_hh_vecs = num_hh_vecs + local_size
         num_chunks  = num_chunks+1
      endif
      nx = nx - nb
   enddo

   ! Allocate space for HH vectors

   allocate(hh_trans_real(nb,num_hh_vecs))

   ! Allocate and init MPI requests

   allocate(ireq_hhr(num_chunks)) ! Recv requests
   allocate(ireq_hhs(nblocks))    ! Send requests

   num_hh_vecs = 0
   num_chunks  = 0
   nx = na
   nt = 0
   do n = 1, nblocks_total
      call determine_workload(nx, nb, np_rows, limits)
      local_size = limits(my_prow+1) - limits(my_prow)
      if(mod(n-1,np_cols) == my_pcol .and. local_size>0 .and. nx>1) then
         num_chunks  = num_chunks+1
         call mpi_irecv(hh_trans_real(1,num_hh_vecs+1), nb*local_size, mpi_real8, nt, &
                        10+n-block_limits(nt), mpi_comm, ireq_hhr(num_chunks), mpierr)
         num_hh_vecs = num_hh_vecs + local_size
      endif
      nx = nx - nb
      if(n == block_limits(nt+1)) then
         nt = nt + 1
      endif
   enddo

   ireq_hhs(:) = MPI_REQUEST_NULL

   ! Buffers for gathering/sending the HH vectors

   allocate(hh_gath(nb,max_blk_size,nblocks)) ! gathers HH vectors
   allocate(hh_send(nb,max_blk_size,nblocks)) ! send buffer for HH vectors
   hh_gath(:,:,:) = 0
   hh_send(:,:,:) = 0

   ! Some counters

   allocate(hh_cnt(nblocks))
   allocate(hh_dst(nblocks))

   hh_cnt(:) = 1 ! The first transfomation vector is always 0 and not calculated at all
   hh_dst(:) = 0 ! PE number for receive

   ireq_ab = MPI_REQUEST_NULL
   ireq_hv = MPI_REQUEST_NULL

   ! Limits for sending

   allocate(snd_limits(0:np_rows,nblocks))

   do iblk=1,nblocks
      call determine_workload(na-(iblk+block_limits(my_pe)-1)*nb, nb, np_rows, snd_limits(:,iblk))
   enddo

   ! ---------------------------------------------------------------------------
   ! Start of calculations

   na_s = block_limits(my_pe)*nb + 1

   if(my_pe>0 .and. na_s<=na) then
      ! send first column to previous PE
      ! Only the PE owning the diagonal does that (sending 1 element of the subdiagonal block also)
      ab_s(1:nb+1) = ab(1:nb+1,na_s-n_off)
      call mpi_isend(ab_s,nb+1,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
   endif

   do istep=1,na-1

      if(my_pe==0) then
         n = MIN(na-na_s,nb) ! number of rows to be reduced
         hv(:) = 0
         tau = 0
         ! The last step (istep=na-1) is only needed for sending the last HH vectors.
         ! We don't want the sign of the last element flipped (analogous to the other sweeps)
         if(istep < na-1) then
            ! Transform first column of remaining matrix
            vnorm2 = sum(ab(3:n+1,na_s-n_off)**2)
            call hh_transform_real(ab(2,na_s-n_off),vnorm2,hf,tau)
            hv(1) = 1
            hv(2:n) = ab(3:n+1,na_s-n_off)*hf
         endif
         d(istep) = ab(1,na_s-n_off)
         e(istep) = ab(2,na_s-n_off)
         if(istep == na-1) then
            d(na) = ab(1,na_s+1-n_off)
            e(na) = 0
         endif
      else
         if(na>na_s) then
            ! Receive Householder vector from previous task, from PE owning subdiagonal
            call mpi_recv(hv,nb,mpi_real8,my_pe-1,2,mpi_comm,MPI_STATUS_IGNORE,mpierr)
            tau = hv(1)
            hv(1) = 1.
         endif
      endif

      na_s = na_s+1
      if(na_s-n_off > nb) then
         ab(:,1:nblocks*nb) = ab(:,nb+1:(nblocks+1)*nb)
         ab(:,nblocks*nb+1:(nblocks+1)*nb) = 0
         n_off = n_off + nb
      endif

      do iblk=1,nblocks

         ns = na_s + (iblk-1)*nb - n_off ! first column in block
         ne = ns+nb-1                    ! last column in block

         if(ns+n_off>na) exit

         ! Store Householder vector for back transformation

         hh_cnt(iblk) = hh_cnt(iblk) + 1

         hh_gath(1   ,hh_cnt(iblk),iblk) = tau
         hh_gath(2:nb,hh_cnt(iblk),iblk) = hv(2:nb)

         if(hh_cnt(iblk) == snd_limits(hh_dst(iblk)+1,iblk)-snd_limits(hh_dst(iblk),iblk)) then
            ! Wait for last transfer to finish
            call mpi_wait(ireq_hhs(iblk), MPI_STATUS_IGNORE, mpierr)
            ! Copy vectors into send buffer
            hh_send(:,1:hh_cnt(iblk),iblk) = hh_gath(:,1:hh_cnt(iblk),iblk)
            ! Send to destination
            call mpi_isend(hh_send(1,1,iblk), nb*hh_cnt(iblk), mpi_real8, &
                           global_id(hh_dst(iblk),mod(iblk+block_limits(my_pe)-1,np_cols)), &
                           10+iblk, mpi_comm, ireq_hhs(iblk), mpierr)
            ! Reset counter and increase destination row
            hh_cnt(iblk) = 0
            hh_dst(iblk) = hh_dst(iblk)+1
         endif

         ! The following code is structured in a way to keep waiting times for
         ! other PEs at a minimum, especially if there is only one block.
         ! For this reason, it requests the last column as late as possible
         ! and sends the Householder vector and the first column as early
         ! as possible.

         nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
         nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                       ! Note that nr>=0 implies that diagonal block is full (nc==nb)!

         ! Multiply diagonal block and subdiagonal block with Householder vector

         if(iblk==nblocks .and. nc==nb) then

            ! We need the last column from the next PE.
            ! First do the matrix multiplications without last column ...

            ! Diagonal block, the contribution of the last element is added below!
            ab(1,ne) = 0
            call DSYMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,0.d0,hd,1)

            ! Subdiagonal block
            if(nr>0) call DGEMV('N',nr,nb-1,tau,ab(nb+1,ns),2*nb-1,hv,1,0.d0,hs,1)

            ! ... then request last column ...
            call mpi_recv(ab(1,ne),nb+1,mpi_real8,my_pe+1,1,mpi_comm,MPI_STATUS_IGNORE,mpierr)

            ! ... and complete the result
            hs(1:nr) = hs(1:nr) + ab(2:nr+1,ne)*tau*hv(nb)
            hd(nb) = hd(nb) + ab(1,ne)*hv(nb)*tau

         else

            ! Normal matrix multiply
            call DSYMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,0.d0,hd,1)
            if(nr>0) call DGEMV('N',nr,nb,tau,ab(nb+1,ns),2*nb-1,hv,1,0.d0,hs,1)

         endif

         ! Calculate first column of subdiagonal block and calculate new
         ! Householder transformation for this column

         hv_new(:) = 0 ! Needed, last rows must be 0 for nr < nb
         tau_new = 0

         if(nr>0) then

            ! complete (old) Householder transformation for first column

            ab(nb+1:nb+nr,ns) = ab(nb+1:nb+nr,ns) - hs(1:nr) ! Note: hv(1) == 1

            ! calculate new Householder transformation ...
            if(nr>1) then
               vnorm2 = sum(ab(nb+2:nb+nr,ns)**2)
               call hh_transform_real(ab(nb+1,ns),vnorm2,hf,tau_new)
               hv_new(1) = 1.
               hv_new(2:nr) = ab(nb+2:nb+nr,ns)*hf
               ab(nb+2:,ns) = 0
            endif

            ! ... and send it away immediatly if this is the last block

            if(iblk==nblocks) then
               call mpi_wait(ireq_hv,MPI_STATUS_IGNORE,mpierr)
               hv_s(1) = tau_new
               hv_s(2:) = hv_new(2:)
               call mpi_isend(hv_s,nb,mpi_real8,my_pe+1,2,mpi_comm,ireq_hv,mpierr)
            endif

         endif


         ! Transform diagonal block
         x = dot_product(hv(1:nc),hd(1:nc))*tau
         hd(1:nc) = hd(1:nc) - 0.5*x*hv(1:nc)

         if(my_pe>0 .and. iblk==1) then

            ! The first column of the diagonal block has to be send to the previous PE
            ! Calculate first column only ...

            ab(1:nc,ns) = ab(1:nc,ns) - hd(1:nc)*hv(1) - hv(1:nc)*hd(1)

            ! ... send it away ...

            call mpi_wait(ireq_ab,MPI_STATUS_IGNORE,mpierr)
            ab_s(1:nb+1) = ab(1:nb+1,ns)
            call mpi_isend(ab_s,nb+1,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)

            ! ... and calculate remaining columns with rank-2 update
            if(nc>1) call DSYR2('L',nc-1,-1.d0,hd(2),1,hv(2),1,ab(1,ns+1),2*nb-1)
         else
            ! No need to  send, just a rank-2 update
            call DSYR2('L',nc,-1.d0,hd,1,hv,1,ab(1,ns),2*nb-1)
         endif

         ! Do the remaining double Householder transformation on the subdiagonal block cols 2 ... nb

         if(nr>0) then
            if(nr>1) then
               call DGEMV('T',nr,nb-1,tau_new,ab(nb,ns+1),2*nb-1,hv_new,1,0.d0,h(2),1)
               x = dot_product(hs(1:nr),hv_new(1:nr))*tau_new
               h(2:nb) = h(2:nb) - x*hv(2:nb)
               ! Unfortunately the is no BLAS routine like DGER2 for a nonsymmetric rank 2 update
               do i=2,nb
                  ab(2+nb-i:1+nb+nr-i,i+ns-1) = ab(2+nb-i:1+nb+nr-i,i+ns-1) - hv_new(1:nr)*h(i) - hs(1:nr)*hv(i)
               enddo
            else
               ! No double Householder transformation for nr=1, just complete the row
               do i=2,nb
                  ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*hv(i)
               enddo
            endif
         endif

         ! Use new HH vector for the next block
         hv(:) = hv_new(:)
         tau = tau_new

      enddo

   enddo

   ! Finish the last outstanding requests
   call mpi_wait(ireq_ab,MPI_STATUS_IGNORE,mpierr)
   call mpi_wait(ireq_hv,MPI_STATUS_IGNORE,mpierr)

   call mpi_waitall(nblocks, ireq_hhs, MPI_STATUSES_IGNORE, mpierr)
   call mpi_waitall(num_chunks, ireq_hhr, MPI_STATUSES_IGNORE, mpierr)

   call mpi_barrier(mpi_comm,mpierr)

   deallocate(ab)
   deallocate(ireq_hhr, ireq_hhs)
   deallocate(hh_cnt, hh_dst)
   deallocate(hh_gath, hh_send)
   deallocate(limits, snd_limits)
   deallocate(block_limits)
   deallocate(global_id)

 end subroutine tridiag_band_real
