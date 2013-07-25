subroutine trans_ev_tridi_to_band_real(na, nev, nblk, nbw, q, ldq, mpi_comm_rows, mpi_comm_cols)

!-------------------------------------------------------------------------------
!  trans_ev_tridi_to_band_real:
!  Transforms the eigenvectors of a tridiagonal matrix back to the eigenvectors of the band matrix
!
!  Parameters
!
!  na          Order of matrix a, number of rows of matrix q
!
!  nev         Number eigenvectors to compute (= columns of matrix q)
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  nb          semi bandwith
!
!  q           On input: Eigenvectors of tridiagonal matrix
!              On output: Transformed eigenvectors
!              Distribution is like in Scalapack.
!
!  ldq         Leading dimension of q
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns/both
!
!-------------------------------------------------------------------------------

    implicit none

    integer, intent(in) :: na, nev, nblk, nbw, ldq, mpi_comm_rows, mpi_comm_cols
    real*8 q(ldq,*)

    integer np_rows, my_prow, np_cols, my_pcol

    integer i, j, ip, sweep, nbuf, l_nev, a_dim2
    integer current_n, current_local_n, current_n_start, current_n_end
    integer next_n, next_local_n, next_n_start, next_n_end
    integer bottom_msg_length, top_msg_length, next_top_msg_length
    integer stripe_width, last_stripe_width, stripe_count
    integer num_result_blocks, num_result_buffers, num_bufs_recvd
    integer a_off, current_tv_off, max_blk_size
    integer mpierr, src, src_offset, dst, offset, nfact, num_blk
    logical flag

    real*8, allocatable :: a(:,:,:), row(:)
    real*8, allocatable :: top_border_send_buffer(:,:,:), top_border_recv_buffer(:,:,:)
    real*8, allocatable :: bottom_border_send_buffer(:,:,:), bottom_border_recv_buffer(:,:,:)
    real*8, allocatable :: result_buffer(:,:,:)
    real*8, allocatable :: bcast_buffer(:,:)

    integer n_off
    integer, allocatable :: result_send_request(:), result_recv_request(:), limits(:)
    integer, allocatable :: top_send_request(:), bottom_send_request(:)
    integer, allocatable :: top_recv_request(:), bottom_recv_request(:)

    ! MPI send/recv tags, arbitrary

    integer, parameter :: bottom_recv_tag = 111
    integer, parameter :: top_recv_tag    = 222
    integer, parameter :: result_recv_tag = 333

    ! Just for measuring the kernel performance
    real*8 kernel_time
    integer*8 kernel_flops


    kernel_time = 1.d-100
    kernel_flops = 0


    call MPI_Comm_rank(mpi_comm_rows, my_prow, mpierr)
    call MPI_Comm_size(mpi_comm_rows, np_rows, mpierr)
    call MPI_Comm_rank(mpi_comm_cols, my_pcol, mpierr)
    call MPI_Comm_size(mpi_comm_cols, np_cols, mpierr)

    if(mod(nbw,nblk)/=0) then
      if(my_prow==0 .and. my_pcol==0) then
         print *,'ERROR: nbw=',nbw,', nblk=',nblk
         print *,'band backtransform works only for nbw==n*nblk'
         call mpi_abort(mpi_comm_world,0,mpierr)
      endif
    endif

    nfact = nbw / nblk


    ! local number of eigenvectors
    l_nev = local_index(nev, my_pcol, np_cols, nblk, -1)

    if(l_nev==0) then
        stripe_width = 0
        stripe_count = 0
        last_stripe_width = 0
    else
        ! Suggested stripe width is 48 since 48*64 real*8 numbers should fit into
        ! every primary cache
        stripe_width = 48 ! Must be a multiple of 4
        stripe_count = (l_nev-1)/stripe_width + 1
        ! Adapt stripe width so that last one doesn't get too small
        stripe_width = (l_nev-1)/stripe_count + 1
        stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 4 !!!
        last_stripe_width = l_nev - (stripe_count-1)*stripe_width
    endif

    ! Determine the matrix distribution at the beginning

    allocate(limits(0:np_rows))

    call determine_workload(na, nbw, np_rows, limits)

    max_blk_size = maxval(limits(1:np_rows) - limits(0:np_rows-1))

    a_dim2 = max_blk_size + nbw

    allocate(a(stripe_width,a_dim2,stripe_count))
    a(:,:,:) = 0

    allocate(row(l_nev))
    row(:) = 0

    ! Copy q from a block cyclic distribution into a distribution with contiguous rows,
    ! and transpose the matrix using stripes of given stripe_width for cache blocking.

    ! The peculiar way it is done below is due to the fact that the last row should be
    ! ready first since it is the first one to start below

    do ip = np_rows-1, 0, -1
        if(my_prow == ip) then
            ! Receive my rows which have not yet been received
            src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
            do i=limits(ip)+1,limits(ip+1)
                src = mod((i-1)/nblk, np_rows)
                if(src < my_prow) then
                    call MPI_Recv(row, l_nev, MPI_REAL8, src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)
                    call unpack_row(row,i-limits(ip))
                elseif(src==my_prow) then
                    src_offset = src_offset+1
                    row(:) = q(src_offset, 1:l_nev)
                    call unpack_row(row,i-limits(ip))
                endif
            enddo
            ! Send all rows which have not yet been send
            src_offset = 0
            do dst = 0, ip-1
              do i=limits(dst)+1,limits(dst+1)
                if(mod((i-1)/nblk, np_rows) == my_prow) then
                    src_offset = src_offset+1
                    row(:) = q(src_offset, 1:l_nev)
                    call MPI_Send(row, l_nev, MPI_REAL8, dst, 0, mpi_comm_rows, mpierr)
                endif
              enddo
            enddo
        else if(my_prow < ip) then
            ! Send all rows going to PE ip
            src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
            do i=limits(ip)+1,limits(ip+1)
                src = mod((i-1)/nblk, np_rows)
                if(src == my_prow) then
                    src_offset = src_offset+1
                    row(:) = q(src_offset, 1:l_nev)
                    call MPI_Send(row, l_nev, MPI_REAL8, ip, 0, mpi_comm_rows, mpierr)
                endif
            enddo
            ! Receive all rows from PE ip
            do i=limits(my_prow)+1,limits(my_prow+1)
                src = mod((i-1)/nblk, np_rows)
                if(src == ip) then
                    call MPI_Recv(row, l_nev, MPI_REAL8, src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)
                    call unpack_row(row,i-limits(my_prow))
                endif
            enddo
        endif
    enddo


    ! Set up result buffer queue

    num_result_blocks = ((na-1)/nblk + np_rows - my_prow) / np_rows

    num_result_buffers = 4*nfact
    allocate(result_buffer(l_nev,nblk,num_result_buffers))

    allocate(result_send_request(num_result_buffers))
    allocate(result_recv_request(num_result_buffers))
    result_send_request(:) = MPI_REQUEST_NULL
    result_recv_request(:) = MPI_REQUEST_NULL

    ! Queue up buffers

    if(my_prow > 0 .and. l_nev>0) then ! note: row 0 always sends
        do j = 1, min(num_result_buffers, num_result_blocks)
            call MPI_Irecv(result_buffer(1,1,j), l_nev*nblk, MPI_REAL8, 0, result_recv_tag, &
                           mpi_comm_rows, result_recv_request(j), mpierr)
        enddo
    endif

    num_bufs_recvd = 0 ! No buffers received yet

    ! Initialize top/bottom requests

    allocate(top_send_request(stripe_count))
    allocate(top_recv_request(stripe_count))
    allocate(bottom_send_request(stripe_count))
    allocate(bottom_recv_request(stripe_count))

    top_send_request(:) = MPI_REQUEST_NULL
    top_recv_request(:) = MPI_REQUEST_NULL
    bottom_send_request(:) = MPI_REQUEST_NULL
    bottom_recv_request(:) = MPI_REQUEST_NULL

    allocate(top_border_send_buffer(stripe_width, nbw, stripe_count))
    allocate(top_border_recv_buffer(stripe_width, nbw, stripe_count))
    allocate(bottom_border_send_buffer(stripe_width, nbw, stripe_count))
    allocate(bottom_border_recv_buffer(stripe_width, nbw, stripe_count))

    top_border_send_buffer(:,:,:) = 0
    top_border_recv_buffer(:,:,:) = 0
    bottom_border_send_buffer(:,:,:) = 0
    bottom_border_recv_buffer(:,:,:) = 0

    ! Initialize broadcast buffer

    allocate(bcast_buffer(nbw, max_blk_size))
    bcast_buffer = 0

    current_tv_off = 0 ! Offset of next row to be broadcast


    ! ------------------- start of work loop -------------------

    a_off = 0 ! offset in A (to avoid unnecessary shifts)

    top_msg_length = 0
    bottom_msg_length = 0

    do sweep = 0, (na-1)/nbw

        current_n = na - sweep*nbw
        call determine_workload(current_n, nbw, np_rows, limits)
        current_n_start = limits(my_prow)
        current_n_end   = limits(my_prow+1)
        current_local_n = current_n_end - current_n_start

        next_n = max(current_n - nbw, 0)
        call determine_workload(next_n, nbw, np_rows, limits)
        next_n_start = limits(my_prow)
        next_n_end   = limits(my_prow+1)
        next_local_n = next_n_end - next_n_start

        if(next_n_end < next_n) then
            bottom_msg_length = current_n_end - next_n_end
        else
            bottom_msg_length = 0
        endif

        if(next_local_n > 0) then
            next_top_msg_length = current_n_start - next_n_start
        else
            next_top_msg_length = 0
        endif

        if(sweep==0 .and. current_n_end < current_n .and. l_nev > 0) then
            do i = 1, stripe_count
                call MPI_Irecv(bottom_border_recv_buffer(1,1,i), nbw*stripe_width, MPI_REAL8, my_prow+1, bottom_recv_tag, &
                           mpi_comm_rows, bottom_recv_request(i), mpierr)
            enddo
        endif

        if(current_local_n > 1) then
            if(my_pcol == mod(sweep,np_cols)) then
                bcast_buffer(:,1:current_local_n) = hh_trans_real(:,current_tv_off+1:current_tv_off+current_local_n)
                current_tv_off = current_tv_off + current_local_n
            endif
            call mpi_bcast(bcast_buffer, nbw*current_local_n, MPI_REAL8, mod(sweep,np_cols), mpi_comm_cols, mpierr)
        else
            ! for current_local_n == 1 the one and only HH vector is 0 and not stored in hh_trans_real
            bcast_buffer(:,1) = 0
        endif

        if(l_nev == 0) cycle

        if(current_local_n > 0) then

          do i = 1, stripe_count

            !wait_b
            if(current_n_end < current_n) then
                call MPI_Wait(bottom_recv_request(i), MPI_STATUS_IGNORE, mpierr)
                n_off = current_local_n+a_off
                a(:,n_off+1:n_off+nbw,i) = bottom_border_recv_buffer(:,1:nbw,i)
                if(next_n_end < next_n) then
                    call MPI_Irecv(bottom_border_recv_buffer(1,1,i), nbw*stripe_width, MPI_REAL8, my_prow+1, bottom_recv_tag, &
                                   mpi_comm_rows, bottom_recv_request(i), mpierr)
                endif
            endif

            if(current_local_n <= bottom_msg_length + top_msg_length) then

                !wait_t
                if(top_msg_length>0) then
                    call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)
                    a(:,a_off+1:a_off+top_msg_length,i) = top_border_recv_buffer(:,1:top_msg_length,i)
                endif

                !compute
                call compute_hh_trafo(0, current_local_n, i)

                !send_b
                call MPI_Wait(bottom_send_request(i), MPI_STATUS_IGNORE, mpierr)
                if(bottom_msg_length>0) then
                    n_off = current_local_n+nbw-bottom_msg_length+a_off
                    bottom_border_send_buffer(:,1:bottom_msg_length,i) = a(:,n_off+1:n_off+bottom_msg_length,i)
                    call MPI_Isend(bottom_border_send_buffer(1,1,i), bottom_msg_length*stripe_width, MPI_REAL8, my_prow+1, &
                                   top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
                endif

            else

                !compute
                call compute_hh_trafo(current_local_n - bottom_msg_length, bottom_msg_length, i)

                !send_b
                call MPI_Wait(bottom_send_request(i), MPI_STATUS_IGNORE, mpierr)
                if(bottom_msg_length > 0) then
                    n_off = current_local_n+nbw-bottom_msg_length+a_off
                    bottom_border_send_buffer(:,1:bottom_msg_length,i) = a(:,n_off+1:n_off+bottom_msg_length,i)
                    call MPI_Isend(bottom_border_send_buffer(1,1,i), bottom_msg_length*stripe_width, MPI_REAL8, my_prow+1, &
                                   top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
                endif

                !compute
                call compute_hh_trafo(top_msg_length, current_local_n-top_msg_length-bottom_msg_length, i)

                !wait_t
                if(top_msg_length>0) then
                    call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)
                    a(:,a_off+1:a_off+top_msg_length,i) = top_border_recv_buffer(:,1:top_msg_length,i)
                endif

                !compute
                call compute_hh_trafo(0, top_msg_length, i)
            endif

            if(next_top_msg_length > 0) then
                !request top_border data
                call MPI_Irecv(top_border_recv_buffer(1,1,i), next_top_msg_length*stripe_width, MPI_REAL8, my_prow-1, &
                               top_recv_tag, mpi_comm_rows, top_recv_request(i), mpierr)
            endif

            !send_t
            if(my_prow > 0) then
                call MPI_Wait(top_send_request(i), MPI_STATUS_IGNORE, mpierr)
                top_border_send_buffer(:,1:nbw,i) = a(:,a_off+1:a_off+nbw,i)
                call MPI_Isend(top_border_send_buffer(1,1,i), nbw*stripe_width, MPI_REAL8, my_prow-1, bottom_recv_tag, &
                               mpi_comm_rows, top_send_request(i), mpierr)
            endif

            ! Care that there are not too many outstanding top_recv_request's
            if(stripe_count > 1) then
                if(i>1) then
                    call MPI_Wait(top_recv_request(i-1), MPI_STATUS_IGNORE, mpierr)
                else
                    call MPI_Wait(top_recv_request(stripe_count), MPI_STATUS_IGNORE, mpierr)
                endif
            endif

          enddo

          top_msg_length = next_top_msg_length

        else
            ! wait for last top_send_request
          do i = 1, stripe_count
            call MPI_Wait(top_send_request(i), MPI_STATUS_IGNORE, mpierr)
          enddo
        endif

        ! Care about the result

        if(my_prow == 0) then

            ! topmost process sends nbw rows to destination processes

            do j=0,nfact-1

                num_blk = sweep*nfact+j ! global number of destination block, 0 based
                if(num_blk*nblk >= na) exit

                nbuf = mod(num_blk, num_result_buffers) + 1 ! buffer number to get this block

                call MPI_Wait(result_send_request(nbuf), MPI_STATUS_IGNORE, mpierr)

                dst = mod(num_blk, np_rows)

                if(dst == 0) then
                    do i = 1, min(na - num_blk*nblk, nblk)
                        call pack_row(row, j*nblk+i+a_off)
                        q((num_blk/np_rows)*nblk+i,1:l_nev) = row(:)
                    enddo
                else
                    do i = 1, nblk
                        call pack_row(result_buffer(:,i,nbuf),j*nblk+i+a_off)
                    enddo
                    call MPI_Isend(result_buffer(1,1,nbuf), l_nev*nblk, MPI_REAL8, dst, &
                                   result_recv_tag, mpi_comm_rows, result_send_request(nbuf), mpierr)
                endif
            enddo

        else

           ! receive and store final result

            do j = num_bufs_recvd, num_result_blocks-1

                nbuf = mod(j, num_result_buffers) + 1 ! buffer number to get this block

                ! If there is still work to do, just test for the next result request
                ! and leave the loop if it is not ready, otherwise wait for all
                ! outstanding requests

                if(next_local_n > 0) then
                    call MPI_Test(result_recv_request(nbuf), flag, MPI_STATUS_IGNORE, mpierr)
                    if(.not.flag) exit
                else
                    call MPI_Wait(result_recv_request(nbuf), MPI_STATUS_IGNORE, mpierr)
                endif

                ! Fill result buffer into q
                num_blk = j*np_rows + my_prow ! global number of current block, 0 based
                do i = 1, min(na - num_blk*nblk, nblk)
                    q(j*nblk+i, 1:l_nev) = result_buffer(1:l_nev, i, nbuf)
                enddo

                ! Queue result buffer again if there are outstanding blocks left
                if(j+num_result_buffers < num_result_blocks) &
                    call MPI_Irecv(result_buffer(1,1,nbuf), l_nev*nblk, MPI_REAL8, 0, result_recv_tag, &
                                   mpi_comm_rows, result_recv_request(nbuf), mpierr)

            enddo
            num_bufs_recvd = j

        endif

        ! Shift the remaining rows to the front of A (if necessary)

        offset = nbw - top_msg_length
        if(offset<0) then
            print *,'internal error, offset for shifting = ',offset
            call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
        endif
        a_off = a_off + offset
        if(a_off + next_local_n + nbw > a_dim2) then
            do i = 1, stripe_count
                do j = top_msg_length+1, top_msg_length+next_local_n
                   A(:,j,i) = A(:,j+a_off,i)
                enddo
            enddo
            a_off = 0
        endif

    enddo

    ! Just for safety:
    if(ANY(top_send_request    /= MPI_REQUEST_NULL)) print *,'*** ERROR top_send_request ***',my_prow,my_pcol
    if(ANY(bottom_send_request /= MPI_REQUEST_NULL)) print *,'*** ERROR bottom_send_request ***',my_prow,my_pcol
    if(ANY(top_recv_request    /= MPI_REQUEST_NULL)) print *,'*** ERROR top_recv_request ***',my_prow,my_pcol
    if(ANY(bottom_recv_request /= MPI_REQUEST_NULL)) print *,'*** ERROR bottom_recv_request ***',my_prow,my_pcol

    if(my_prow == 0) then
        call MPI_Waitall(num_result_buffers, result_send_request, MPI_STATUSES_IGNORE, mpierr)
    endif

    if(ANY(result_send_request /= MPI_REQUEST_NULL)) print *,'*** ERROR result_send_request ***',my_prow,my_pcol
    if(ANY(result_recv_request /= MPI_REQUEST_NULL)) print *,'*** ERROR result_recv_request ***',my_prow,my_pcol

    if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
        print '(" Kernel time:",f10.3," MFlops: ",f10.3)', kernel_time, kernel_flops/kernel_time*1.d-6

    ! deallocate all working space

    deallocate(a)
    deallocate(row)
    deallocate(limits)
    deallocate(result_send_request)
    deallocate(result_recv_request)
    deallocate(top_border_send_buffer)
    deallocate(top_border_recv_buffer)
    deallocate(bottom_border_send_buffer)
    deallocate(bottom_border_recv_buffer)
    deallocate(result_buffer)
    deallocate(bcast_buffer)
    deallocate(top_send_request)
    deallocate(top_recv_request)
    deallocate(bottom_send_request)
    deallocate(bottom_recv_request)

contains

    subroutine pack_row(row, n)
        real*8 row(:)
        integer n, i, noff, nl

        do i=1,stripe_count
            nl = merge(stripe_width, last_stripe_width, i<stripe_count)
            noff = (i-1)*stripe_width
            row(noff+1:noff+nl) = a(1:nl,n,i)
        enddo

    end subroutine

    subroutine unpack_row(row, n)
        real*8 row(:)
        integer n, i, noff, nl

        do i=1,stripe_count
            nl = merge(stripe_width, last_stripe_width, i<stripe_count)
            noff = (i-1)*stripe_width
            a(1:nl,n,i) = row(noff+1:noff+nl)
        enddo

    end subroutine

    subroutine compute_hh_trafo(off, ncols, istripe)

        integer off, ncols, istripe, j, nl, jj
        real*8 w(nbw,2), ttt

        ttt = mpi_wtime()
        nl = merge(stripe_width, last_stripe_width, istripe<stripe_count)
        do j = ncols, 2, -2
            w(:,1) = bcast_buffer(1:nbw,j+off)
            w(:,2) = bcast_buffer(1:nbw,j+off-1)
            call double_hh_trafo(a(1,j+off+a_off-1,istripe), w, nbw, nl, stripe_width, nbw)
!            print*, "In double"
        enddo
        if(j==1) then
!           print*, " **********  in single ************"
           call single_hh_trafo(a(1,1+off+a_off,istripe),bcast_buffer(1,off+1), nbw, nl, stripe_width)
        endif
        kernel_flops = kernel_flops + 4*int(nl,8)*int(ncols,8)*int(nbw,8)
        kernel_time = kernel_time + mpi_wtime()-ttt

    end subroutine

end subroutine


subroutine single_hh_trafo(q, hh, nb, nq, ldq)

    ! Perform single real Householder transformation.
    ! This routine is not performance critical and thus it is coded here in Fortran

    implicit none
    integer nb, nq, ldq
    real*8 q(ldq, *), hh(*)

    integer i
    real*8 v(nq)

    ! v = q * hh
    v(:) = q(1:nq,1) ! nq - slab height = # of eigenvectors; q(1:nq, 1) -> 1st elem of each eigenvector in the slab
    do i=2,nb
        v(:) = v(:) + q(1:nq,i) * hh(i)
    enddo

    ! v = v * tau
    v(:) = v(:) * hh(1)

    ! q = q - v * hh**T
    q(1:nq,1) = q(1:nq,1) - v(:)
    do i=2,nb
        q(1:nq,i) = q(1:nq,i) - v(:) * hh(i)
    enddo

end subroutine

!-------------------------------------------------------------------------------

subroutine determine_workload(na, nb, nprocs, limits)

    integer, intent(in) :: na, nb, nprocs
    integer, intent(out) :: limits(0:nprocs)

    integer i

    if(na <= 0) then
        limits(:) = 0
        return
    endif

    if(nb*nprocs > na) then
        ! there is not enough work for all
        do i = 0, nprocs
            limits(i) = min(na, i*nb)
        enddo
    else
        do i = 0, nprocs
            limits(i) = (i*na)/nprocs
        enddo
    endif

end subroutine
