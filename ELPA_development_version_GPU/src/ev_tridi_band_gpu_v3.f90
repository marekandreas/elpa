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

    real*8, allocatable :: row(:), row_group(:,:)
    real*8, allocatable :: top_border_send_buffer(:,:,:), top_border_recv_buffer(:,:,:)
    real*8, allocatable :: bottom_border_send_buffer(:,:,:), bottom_border_recv_buffer(:,:,:)
    real*8, allocatable :: result_buffer(:,:,:)
    real*8, allocatable :: bcast_buffer(:,:)

    real*8, allocatable, device :: a_dev(:,:,:)
    real*8, allocatable, device :: bcast_buffer_dev(:,:)
    real*8, allocatable, device :: row_dev(:)
    real*8, allocatable, device :: row_group_dev(:,:)
    real*8, allocatable, device :: hh_dot_dev(:)
    real*8, allocatable, device :: hh_tau_dev(:)
    integer ierr
    integer :: top, chunk, this_chunk
    integer row_group_size, unpack_idx

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

    unpack_idx = 0
    row_group_size = 0
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

    ! Local number of eigenvectors
    ! Important: if l_nev is 0, the current process will fail to allocate device memory and terminate
    !            This happens when the dataset is too small for the number of MPI processes
    l_nev = local_index(nev, my_pcol, np_cols, nblk, -1)

    if(l_nev == 0) then
        stripe_width = 0
        stripe_count = 0
        last_stripe_width = 0
    else
        stripe_width = 256 ! Must be a multiple of 4
        stripe_count = (l_nev - 1) / stripe_width + 1
        last_stripe_width = l_nev - (stripe_count - 1) * stripe_width
    endif

    ! Determine the matrix distribution at the beginning
    allocate(limits(0 : np_rows))

    call determine_workload(na, nbw, np_rows, limits)

    max_blk_size = maxval(limits(1 : np_rows) - limits(0 : np_rows - 1))
    a_dim2 = max_blk_size + nbw

    allocate(a_dev(stripe_width, a_dim2, stripe_count))
    a_dev(:, :, :) = 0

    allocate(row(l_nev))
    row(:) = 0

    allocate(row_dev(l_nev))
    row_dev(:) = 0

    ! "row_group" and "row_group_dev" are needed for GPU optimizations
    allocate(row_group(l_nev, nblk))
    row_group(:, :) = 0

    allocate(row_group_dev(l_nev, nblk))
    row_group_dev(:, :) = 0


    ! Copy q from a block cyclic distribution into a distribution with contiguous rows,
    ! and transpose the matrix using stripes of given stripe_width for cache blocking.

    ! The peculiar way it is done below is due to the fact that the last row should be
    ! ready first since it is the first one to start below

    do ip = np_rows - 1, 0, -1
        if (my_prow == ip) then
            ! Receive my rows which have not yet been received
            src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
            do i=limits(ip)+1,limits(ip+1)
                src = mod((i-1)/nblk, np_rows)
                if(src < my_prow) then
                    ! An unpacking of the current row group may occur before queuing the next row 
                    call unpack_and_prepare_row_group(i - limits(ip), .false.)
                    call MPI_Recv(row_group(:, row_group_size), l_nev, MPI_REAL8, src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)    
                elseif(src==my_prow) then
                    src_offset = src_offset+1
                    ! An unpacking of the current row group may occur before queuing the next row
                    call unpack_and_prepare_row_group(i - limits(ip), .false.)
                    row_group(:, row_group_size) = q(src_offset, 1:l_nev)
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
                    ! An unpacking of the current row group may occur before queuing the next row
                    call unpack_and_prepare_row_group(i - limits(my_prow), .false.)
                    call MPI_Recv(row_group(:, row_group_size), l_nev, MPI_REAL8, src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)
                endif
            enddo
        endif
    enddo

    ! Force an unpacking of all remaining rows that haven't been unpacked yet
    call unpack_and_prepare_row_group(-1, .true.)
    ierr = cudaDeviceSynchronize()

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

    allocate(bcast_buffer_dev(nbw, max_blk_size))
    bcast_buffer_dev = 0

    allocate(hh_dot_dev(max_blk_size - 1))
    hh_dot_dev = 0

    allocate(hh_tau_dev(max_blk_size))
    hh_tau_dev = 0

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
            call mpi_bcast(bcast_buffer, nbw * current_local_n, MPI_REAL8, mod(sweep,np_cols), mpi_comm_cols, mpierr)
            ierr = cudaMemcpy(bcast_buffer_dev(1, 1), bcast_buffer(1, 1), nbw * current_local_n)

            call extract_hh_tau(nbw, current_local_n, .false.)
            call compute_hh_dot_products(nbw, current_local_n)
        else
            ! for current_local_n == 1 the one and only HH vector is 0 and not stored in hh_trans_real
            bcast_buffer(:, 1) = 0
            bcast_buffer_dev(:, 1) = 0
            call extract_hh_tau(nbw, 1, .true.)
        endif

        if(l_nev == 0) cycle

        if(current_local_n > 0) then

          do i = 1, stripe_count

            !wait_b
            if(current_n_end < current_n) then
                call MPI_Wait(bottom_recv_request(i), MPI_STATUS_IGNORE, mpierr)
                n_off = current_local_n+a_off
                ierr = cudaMemcpy(a_dev(1,n_off+1,i), bottom_border_recv_buffer(1,1,i), stripe_width * nbw)

                if(next_n_end < next_n) then
                    call MPI_Irecv(bottom_border_recv_buffer(1,1,i), nbw*stripe_width, MPI_REAL8, my_prow+1, bottom_recv_tag, &
                                   mpi_comm_rows, bottom_recv_request(i), mpierr)
                endif
            endif

            if(current_local_n <= bottom_msg_length + top_msg_length) then

                !wait_t
                if(top_msg_length>0) then
                    call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)
                    ierr = cudaMemcpy(a_dev(:,a_off+1,i),  top_border_recv_buffer(1,1,i), stripe_width * top_msg_length)

                endif

                !compute
                call compute_hh_trafo(0, current_local_n, i)

                !send_b
                call MPI_Wait(bottom_send_request(i), MPI_STATUS_IGNORE, mpierr)
                if(bottom_msg_length>0) then
                    n_off = current_local_n+nbw-bottom_msg_length+a_off
                    ierr = cudaMemcpy(bottom_border_send_buffer(1,1,i), a_dev(1,n_off+1,i), stripe_width * bottom_msg_length)

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
                    ierr = cudaMemcpy(bottom_border_send_buffer(1,1,i), a_dev(1,n_off+1,i), stripe_width * bottom_msg_length)
                  
                    call MPI_Isend(bottom_border_send_buffer(1,1,i), bottom_msg_length*stripe_width, MPI_REAL8, my_prow+1, &
                                   top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
                endif

                !compute
                call compute_hh_trafo(top_msg_length, current_local_n-top_msg_length-bottom_msg_length, i)

                !wait_t
                if(top_msg_length>0) then
                    call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)
                    ierr = cudaMemcpy(a_dev(:,a_off+1,i), top_border_recv_buffer(:,1,i), stripe_width * top_msg_length)
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
                ierr = cudaMemcpy(top_border_send_buffer(:,1,i), a_dev(:,a_off+1,i), stripe_width * nbw)
                call MPI_Isend(top_border_send_buffer(1,1,i), nbw*stripe_width, MPI_REAL8, my_prow-1, bottom_recv_tag, &
                               mpi_comm_rows, top_send_request(i), mpierr)
            endif

            ! Care that there are not too many outstanding top_recv_request's
            if(stripe_count > 1) then
                if(i > 1) then
                    call MPI_Wait(top_recv_request(i - 1), MPI_STATUS_IGNORE, mpierr)
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
                    row_group_size = min(na - num_blk*nblk, nblk)
                    call pack_row_group(row_group(:, :), j * nblk + a_off, row_group_size)

                    do i = 1, row_group_size
                        q((num_blk / np_rows) * nblk + i, 1 : l_nev) = row_group(:, i)
                    enddo
                else
                    call pack_row_group(result_buffer(:, :, nbuf), j * nblk + a_off, nblk)
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
                    if(.not. flag) exit
                else
                    call MPI_Wait(result_recv_request(nbuf), MPI_STATUS_IGNORE, mpierr)
                endif

                ! Fill result buffer into q
                num_blk = j * np_rows + my_prow ! global number of current block, 0 based
                do i = 1, min(na - num_blk*nblk, nblk)
                    q(j * nblk + i, 1 : l_nev) = result_buffer(1 : l_nev, i, nbuf)
                enddo

                ! Queue result buffer again if there are outstanding blocks left
                if(j + num_result_buffers < num_result_blocks) &
                    call MPI_Irecv(result_buffer(1, 1, nbuf), l_nev * nblk, MPI_REAL8, 0, result_recv_tag, &
                                   mpi_comm_rows, result_recv_request(nbuf), mpierr)

            enddo
            num_bufs_recvd = j

        endif

        ! Shift the remaining rows to the front of A (if necessary)

        offset = nbw - top_msg_length
        if(offset < 0) then
            print *,'internal error, offset for shifting = ',offset
            call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
        endif
        a_off = a_off + offset
        if(a_off + next_local_n + nbw > a_dim2) then
            do i = 1, stripe_count

                chunk = min(next_local_n - 1, a_off)
                do j = top_msg_length + 1, top_msg_length + next_local_n, chunk
                   top = min(j + chunk, top_msg_length + next_local_n)
                   this_chunk = top - j + 1
                   ierr = cudaMemcpy(a_dev(1, j, i), a_dev(1, j + a_off,i), &
                             stripe_width * this_chunk)
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
    
    deallocate(a_dev)
    deallocate(bcast_buffer_dev)
    deallocate(hh_dot_dev)
    deallocate(hh_tau_dev)
    deallocate(row_dev)
    deallocate(row_group)
    deallocate(row_group_dev)

contains


    ! Pack a filled row group (i.e. an array of consecutive rows)
    subroutine pack_row_group(rows, n_offset, row_count)
        implicit none
        integer, intent(in) :: n_offset, row_count
        real*8, intent(out) :: rows(:, :)
        integer max_idx
        type(dim3) :: grid_size

        ! Use many blocks for higher GPU occupancy
        grid_size = dim3(row_count, stripe_count, 1)
        max_idx = (stripe_count - 1) * stripe_width + last_stripe_width

        ! Use one kernel call to pack the entire row group
        call my_pack_kernel<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, a_dev, row_group_dev)
    
        ! Issue one single transfer call for all rows (device to host)
        rows(:, 1 : row_count) = row_group_dev(:, 1 : row_count)
    end subroutine


    ! Unpack a filled row group (i.e. an array of consecutive rows)
    subroutine unpack_row_group(rows, n_offset, row_count)        
        implicit none        
        integer, intent(in) :: n_offset, row_count
        real*8, intent(in) :: rows(:, :)
        integer max_idx
        type(dim3) :: grid_size
        integer i

        ! Use many blocks for higher GPU occupancy
        grid_size = dim3(row_count, stripe_count, 1)
        max_idx = (stripe_count - 1) * stripe_width + last_stripe_width

        ! Issue one single transfer call for all rows (host to device)
        row_group_dev(:, 1 : row_count) = rows(:, 1 : row_count)

        ! Use one kernel call to pack the entire row group
        call my_unpack_kernel<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, row_group_dev, a_dev)            
    end subroutine


    ! This subroutine must be called before queuing the next row for unpacking; it ensures that an unpacking of the current row group
    ! occurs when the queue is full or when the next row belongs to another group 
    subroutine unpack_and_prepare_row_group(next_unpack_idx, force)
        implicit none        
        integer, intent(in) :: next_unpack_idx
        logical, intent(in) :: force

        if (row_group_size == 0) then
            ! Nothing to flush, just prepare for the upcoming row
            row_group_size = 1
        else
            if (force .or. (row_group_size == nblk) .or. (unpack_idx + 1 /= next_unpack_idx)) then
                ! A flush and a reset must be performed
                call unpack_row_group(row_group(:, :), unpack_idx - row_group_size, row_group_size)
                row_group_size = 1
            else
                ! Just prepare for the upcoming row
                row_group_size = row_group_size + 1
            endif
        endif
        ! Always update the index for the upcoming row
        unpack_idx = next_unpack_idx    
    end subroutine

    ! Host wrapper for the Householder backtransformation step. Several kernels are available. Performance note:
    ! - "compute_hh_trafo_c_kernel" is the C kernel for the backtransformation (this exhibits best performance)
    ! - "compute_hh_trafo_kernel" is the Fortran equivalent of the C kernel
    ! - "compute_hh_trafo_single_kernel" is the reference Fortran kernel
    subroutine compute_hh_trafo(off, ncols, istripe)
        implicit none
        integer, intent(in) :: off, ncols, istripe
        integer nl
        real*8 ttt

        ! ncols - indicates the number of HH reflectors to apply; at least 1 must be available
        if (ncols < 1) return

        ttt = mpi_wtime()
        nl = merge(stripe_width, last_stripe_width, istripe < stripe_count)

        ! Uncomment the kernel you want to use; comment out the other 2

        call launch_compute_hh_trafo_c_kernel(a_dev(1, a_off + 1, istripe), bcast_buffer_dev(1, 1), hh_dot_dev, hh_tau_dev, nl, nbw, stripe_width, off, ncols)
!        call compute_hh_trafo_kernel<<<nl, nbw>>>(a_dev(1, a_off, istripe), bcast_buffer_dev(1, 1), hh_dot_dev, hh_tau_dev, nbw, stripe_width, off, ncols)
!        call compute_hh_trafo_single_kernel<<<nl, nbw>>>(a_dev(1, off + a_off, istripe), bcast_buffer_dev(1, off + 1), hh_tau_dev(off + 1), nbw, stripe_width, ncols)

        ! Since we only use the default CUDA stream, no explicit device synchronization is necessary here
        kernel_flops = kernel_flops + 4 * int(nl, 8) * int(ncols, 8) * int(nbw, 8)
        kernel_time = kernel_time + mpi_wtime() - ttt

    end subroutine


    ! The host wrapper for computing the dot products between consecutive HH reflectors (see the kernel below)
    subroutine compute_hh_dot_products(nbw, n)
        implicit none
        integer, value :: nbw, n   

        if (n .le. 1) return
        call compute_hh_dotp_kernel<<<n - 1, nbw>>>(bcast_buffer_dev(1, 1), hh_dot_dev, nbw, n)
    end subroutine


    ! The host wrapper for extracting "tau" from the HH reflectors (see the kernel below)
    subroutine extract_hh_tau(nbw, n, is_zero)
        implicit none
        integer, value :: nbw, n
        logical, value :: is_zero
        integer grid_size

        grid_size = 1 + (n - 1) / 256
        call extract_hh_tau_kernel<<<grid_size, 256>>>(bcast_buffer_dev(1, 1), hh_tau_dev, nbw, n, is_zero)
    end subroutine

end subroutine

! -------------------------------------------
! Fortran back-transformation support kernels
! -------------------------------------------

! Reset a reduction block
! Limitation: the thread-block size must be a divider of the reduction block's size
attributes(device) subroutine reset_shared_block(s_block, b_size)
    implicit none
    real*8, intent(out) :: s_block(*)
    integer, intent(in) :: b_size
    integer i, t_idx, s_chunk

    t_idx = threadIdx%x
    s_chunk = b_size / blockDim%x
    do i = (t_idx - 1) * s_chunk + 1, t_idx * s_chunk
        s_block(i) = 0.0
    enddo

    call syncthreads()
end subroutine

! Reset 2 reduction blocks without an explicit synchronization at the end
! Limitation: : the thread-block size must be a divider of the reduction block's size
attributes(device) subroutine reset_shared_block_pair(s_block_1, s_block_2, b_size)
    implicit none
    real*8, intent(out) :: s_block_1(*), s_block_2(*)
    integer, intent(in) :: b_size
    integer i, t_idx, s_chunk

    t_idx = threadIdx%x
    s_chunk = b_size / blockDim%x
    do i = (t_idx - 1) * s_chunk + 1, t_idx * s_chunk
        s_block_1(i) = 0.0
        s_block_2(i) = 0.0
    enddo
end subroutine

! Perform a reduction on an initialized, 128-element shared block
attributes(device) subroutine warp_reduce(s_block)
    implicit none
    real*8 :: s_block(*)
    integer t_idx

    t_idx = threadIdx%x

    call syncthreads()
    if (t_idx .le. 32) then
        s_block(t_idx) = s_block(t_idx) + s_block(t_idx + 32) + s_block(t_idx + 64) + s_block(t_idx + 96)
           
        if (t_idx .le. 8) s_block(t_idx) = s_block(t_idx) + s_block(t_idx + 8) + s_block(t_idx + 16) + s_block(t_idx + 24)
        if (t_idx .le. 4) s_block(t_idx) = s_block(t_idx) + s_block(t_idx + 4)
        if (t_idx .le. 1) s_block(t_idx) = s_block(t_idx) + s_block(t_idx + 1) + s_block(t_idx + 2) + s_block(t_idx + 3)
    endif
end subroutine


! Compute the dot-product between 2 consecutive HH vectors
! Limitation 1: the size of the thread block must be at most 128 and a power-of-2
! Limitation 2: the size of the warp must be equal to 32
attributes(global) subroutine compute_hh_dotp_kernel(hh, v_dot, nb, n)
    implicit none
    real*8, shared, dimension(128) :: hh_s

    integer, value :: nb, n, t_idx, v_idx
    real*8, intent(in)  :: hh(nb, *)
    real*8, intent(out) :: v_dot(*)
    
    ! The vector index (v_idx) identifies the pair of HH reflectors from which the dot product is computed
    v_idx = blockIdx%x

    ! The thread index indicates the position within the two HH reflectors
    t_idx = threadIdx%x

    ! The contents of the shared memory must be fully reset
    call reset_shared_block(hh_s, 128)

    ! Initialize the contents of the shared buffer (preparing for reduction)
    if (t_idx .gt. 1) then
        hh_s(t_idx) = hh(t_idx, v_idx) * hh(t_idx - 1, v_idx + 1)
    else
        hh_s(t_idx) = 0.0
    endif
    
    ! Compute the dot product using a fast reduction
    call warp_reduce(hh_s)  

    if (t_idx .eq. 1) v_dot(v_idx) = hh_s(1)
end subroutine

! Extract "tau" from the HH matrix and replace it with 1.0 or 0.0 (depending on case)
! Having "tau" as the first element in a HH reflector reduces space requirements, but causes undesired branching in the kernels
attributes(global) subroutine extract_hh_tau_kernel(hh, hh_tau, nb, n, is_zero)
    implicit none
    integer, value :: nb, n
    logical, value :: is_zero
    real*8 :: hh(nb, *)
    real*8, intent(out) :: hh_tau(*)
    integer h_idx
    
    ! Select the corresponding HH reflector to change
    h_idx = (blockIdx%x - 1) * blockDim%x + threadIdx%x

    if (h_idx .gt. n) return

    ! Extract "tau" and store it separately
    hh_tau(h_idx) = hh(1, h_idx)

    ! Replace the first element in the HH reflector with 1.0 or 0.0
    if (is_zero) then
        hh(1, h_idx) = 0.0
    else
        hh(1, h_idx) = 1.0
    endif
end subroutine

! -------------------------------------------
! Fortran back-transformation support kernels
! -------------------------------------------

! This is the simplest and slowest available backtransformation kernel 
attributes(global) subroutine compute_hh_trafo_single_kernel(q, hh, hh_tau, nb, ldq, ncols)
    implicit none

    real*8, shared, dimension(128) :: dotp_s
    real*8, shared, dimension(128) :: q_s

    integer, value :: nb, ldq, ncols, b_idx, t_idx
    integer j
    real*8 q(ldq, *)
    real*8, intent(in) :: hh(nb, *), hh_tau(*)
    real*8 q_v, hh_v, tau

    ! The block index selects the eigenvector which the current block is responsible for
    b_idx = blockIdx%x

    ! The thread index selects the position inside the eigenvector selected above
    t_idx = threadIdx%x 

    ! Even if we have fewer than 128 threads, the shared memory blocks must be reset
    call reset_shared_block(dotp_s, 128)

    do j = ncols, 1, -1
        ! Read the local value, either from q or from the ring buffer
        if ((j .eq. ncols) .or. (t_idx .eq. 1)) then
            q_v = q(b_idx, j + t_idx)
        else
            q_v = q_s(t_idx - 1)
        endif
        
        ! Read the corresponding value in the Householder vector
        hh_v = hh(t_idx, j)
        tau = hh_tau(j)

        ! Fill the shared buffer 
        dotp_s(t_idx) = q_v * hh_v

        ! Perform the reduction
        call warp_reduce(dotp_s)
        call syncthreads()
    
        ! Dot-product between current eigenvector chunk and Householder vector + Tau
        q_v = q_v - dotp_s(1) * tau * hh_v    

        ! Each line of "q" (i.e. each individual eigenvector) can be updated independently
        q_s(t_idx) = q_v
        if ((j .eq. 1) .or. (t_idx .eq. blockDim%x)) q(b_idx, j + t_idx) = q_v 
        
        call syncthreads()
    enddo

end subroutine


! This is an improved version of the simple backtransformation kernel; here, we halve the number of iterations and apply
! 2 Householder reflectors per iteration
attributes(global) subroutine compute_hh_trafo_kernel(q, hh, hh_dot, hh_tau, nb, ldq, off, ncols)
    implicit none
    
    real*8, shared, dimension(128) :: dotp_s_1, dotp_s_2
    real*8, shared, dimension(129) :: q_s

    integer, value :: nb, ldq, off, ncols
    integer b_idx, t_idx, q_off, j
    real*8 q(ldq, *)
    real*8, intent(in) :: hh(nb, *), hh_dot(*), hh_tau(*) 
    real*8 q_v_1, q_v_2, hh_v_1, hh_v_2, tau_1, tau_2, s_1, s_2, dot_p, hh_v_3, mask;

    ! The block index selects the eigenvector (EV) which the current block is responsible for
    b_idx = blockIdx%x

    ! The thread index selects the position inside the eigenvector selected above
    t_idx = threadIdx%x

    ! The entire contents of the shared reduction buffers must be reset
    call reset_shared_block_pair(dotp_s_1, dotp_s_2, 128)

    ! Compute initial access indices
    j = off + ncols
    q_off = j + t_idx

    ! Load the last EV components in the EV cache
    if (t_idx .gt. 1) q_s(t_idx + 1) = q(b_idx, q_off)

    ! Ensure the ring buffer and reduction buffers are initialized
    call syncthreads()

    if (t_idx .eq. 1) then 
        mask = 0.0
    else
        mask = 1.0
    endif

    do while (j .ge. off + 2) 
        ! Per-iteration GMem I/O reads are in order to improve cache hit ratio

        ! Read the corresponding compotents in the 2 Householder reflectors
        hh_v_1 = hh(t_idx, j)
        hh_v_2 = hh(t_idx, j - 1)
        hh_v_3 = hh(t_idx - 1, j) * mask

        ! Read the pre-computed dot-product of the 2 Householder reflectors
        dot_p = hh_dot(j - 1)

        ! Read the pre-computed values for "Tau" corresponding to the 2 Householder reflectors
        tau_1 = hh_tau(j)
        tau_2 = hh_tau(j - 1)

        ! Only read the new EV components (the others are already stored in the shared EV cache, q_s)
        if (t_idx .eq. 1) then
            q_s(1) = q(b_idx, q_off - 1)
            q_s(2) = q(b_idx, q_off)
        endif

        ! Fill the shared buffers for the dot products bewtween the EV subset and the Householder reflectors         
        q_v_1 = q_s(t_idx + 1)
        q_v_2 = q_s(t_idx)

        dotp_s_1(t_idx) = q_v_1 * hh_v_1 * tau_1
        dotp_s_2(t_idx) = q_v_2 * hh_v_2 * tau_2

        ! Ensure the reduction buffers are fully populated
        call syncthreads()

        ! Perform the 2 reductions using only the first warp (we assume the warp size is 32, valid up to CC 3.x)
        if (t_idx .le. 32) then
            dotp_s_1(t_idx) = dotp_s_1(t_idx) + dotp_s_1(t_idx + 32) + dotp_s_1(t_idx + 64) + dotp_s_1(t_idx + 96)
            dotp_s_2(t_idx) = dotp_s_2(t_idx) + dotp_s_2(t_idx + 32) + dotp_s_2(t_idx + 64) + dotp_s_2(t_idx + 96)

            if (t_idx .le. 8) then
                dotp_s_1(t_idx) = dotp_s_1(t_idx) + dotp_s_1(t_idx + 8) + dotp_s_1(t_idx + 16) + dotp_s_1(t_idx + 24)
                dotp_s_2(t_idx) = dotp_s_2(t_idx) + dotp_s_2(t_idx + 8) + dotp_s_2(t_idx + 16) + dotp_s_2(t_idx + 24)
            endif

            if (t_idx .le. 2) then
                dotp_s_1(t_idx) = dotp_s_1(t_idx) + dotp_s_1(t_idx + 2) + dotp_s_1(t_idx + 4) + dotp_s_1(t_idx + 6)
                dotp_s_2(t_idx) = dotp_s_2(t_idx) + dotp_s_2(t_idx + 2) + dotp_s_2(t_idx + 4) + dotp_s_2(t_idx + 6)
            endif
        endif
        
        ! Ensure every thread will have access to the reduction results
        call syncthreads()

        ! Each thread collects the reduction results
        s_1 = dotp_s_1(1) + dotp_s_1(2)
        s_2 = dotp_s_2(1) + dotp_s_2(2)

        ! Each thread updates its corresponding EV component
        q_v_2 = q_v_2 - hh_v_3 * s_1 - hh_v_2 * s_2 + tau_2 * hh_v_2 * s_1 * dot_p  

        if (t_idx .eq. blockDim%x) then
            ! The last thread writes the last 2 EV components to the EV matrix
            q(b_idx, q_off) = q_v_1 - hh_v_1 * s_1
            q(b_idx, q_off - 1) = q_v_2
        else
            ! All other threads update the EV cache for the next iteration
            q_s(t_idx + 2) = q_v_2
        endif
        
        call syncthreads()

        ! Update access indices
        q_off = q_off - 2
        j = j - 2
    enddo

    ! Once the previous loop has finished, we have at most 1 more iteration to perform

    if (j .eq. off) then
        ! No iterations remain, so the final contents of the EV matrix are updated
        if (t_idx .lt. blockDim%x) q(b_idx, q_off + 1) = q_v_2
    else
        ! One iteration remains; it must be processed separately
        if (t_idx .eq. 1) then
            ! Only one more EV element needs to be loaded
            q_s(2) = q(b_idx, q_off)
        endif
            
        ! As before, we first read the EV and Householder components
        q_v_1 = q_s(t_idx + 1)
        hh_v_1 = hh(t_idx, j)
        tau_1 = hh_tau(j)

        ! We prepare the reduction buffer
        dotp_s_1(t_idx) = q_v_1 * hh_v_1 * tau_1

        ! Perform the reduction
        call warp_reduce(dotp_s_1)
        call syncthreads()
        
        ! The last EV components are written to the EV matrix
        q(b_idx, q_off) = q_v_1 - hh_v_1 * dotp_s_1(1)
    endif

end subroutine


! ---------------------------------
! Row packing and unpacking kernels
! ---------------------------------

! The row group packing kernel
attributes(global) subroutine my_pack_kernel(n_offset, max_idx, stripe_width, a_dim2, stripe_count, src, dst) 
    implicit none
    integer, value :: n_offset
    integer, value :: max_idx
    integer, value :: stripe_width
    integer, value :: a_dim2
    integer, value :: stripe_count
    real*8, intent(in)  :: src(stripe_width, a_dim2, *)
    real*8, intent(out) :: dst(:, :)
    integer :: sIdx, dIdx, bId, tId
   
    ! blockIdx%x - indicates the row being packed
    ! blockIdx%y - indicates the index within the stripe count
    bId = blockIdx%y 
    tId = threadIdx%x

    dIdx = (bId - 1) * stripe_width + tId
    if (dIdx .le. max_idx) dst(dIdx, blockIdx%x) = src(tId, n_offset + blockIdx%x, bId)
end subroutine

! The row group unpacking kernel
attributes(global) subroutine my_unpack_kernel(n_offset, max_idx, stripe_width, a_dim2, stripe_count, src, dst) 
    implicit none
    integer, value :: n_offset
    integer, value :: max_idx
    integer, value :: stripe_width
    integer, value :: a_dim2
    integer, value :: stripe_count
    real*8, intent(in)  :: src(:, :)
    real*8, intent(out) :: dst(stripe_width, a_dim2, *)
    integer :: sIdx, bId, tId

    ! blockIdx%x - indicates the row being packed
    ! blockIdx%y - indicates the index within the stripe count
    bId = blockIdx%y 
    tId = threadIdx%x

    sIdx = (bId - 1) * stripe_width + tId
    if (sIdx .le. max_idx) dst(tId, n_offset + blockIdx%x, bId) = src(sIdx, blockIdx%x)
end subroutine

! ----------------------
! Workload configuration
! ----------------------

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
