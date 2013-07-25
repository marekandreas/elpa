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

use cublas
    implicit none

    integer, intent(in) :: na, nev, nblk, nbw, ldq, mpi_comm_rows, mpi_comm_cols
    real*8 q(ldq,*)

    integer np_rows, my_prow, np_cols, my_pcol

    integer i, j, ip, sweep, nbuf, l_nev, a_dim2
    integer current_n, current_local_n, current_n_start, current_n_end
    integer next_n, next_local_n, next_n_start, next_n_end
    integer bottom_msg_length, top_msg_length, next_top_msg_length
    integer thread_width, stripe_width, stripe_count, csw
    integer num_result_blocks, num_result_buffers, num_bufs_recvd
    integer a_off, current_tv_off, max_blk_size, b_off, b_len
    integer mpierr, src, src_offset, dst, offset, nfact, num_blk
    integer mpi_status(MPI_STATUS_SIZE)
    logical flag

    real*8, allocatable, device :: a(:,:,:,:)
    real*8, allocatable :: a_host(:,:,:,:)
    real*8, allocatable :: row(:)
    real*8, allocatable :: top_border_send_buffer(:,:), top_border_recv_buffer(:,:)
    real*8, allocatable :: bottom_border_send_buffer(:,:), bottom_border_recv_buffer(:,:)
    real*8, allocatable :: result_buffer(:,:,:)
    real*8, allocatable :: bcast_buffer(:,:)
    real*8, allocatable, device :: bcast_buffer_dev(:,:)
    real*8, allocatable, device :: tau_dev(:)
    !real*8, allocatable :: bcast_buffer_dev(:,:)
    !real*8, allocatable :: tau_dev(:)

    real*8, device :: v_dev(256)

    integer n_off
    integer, allocatable :: result_send_request(:), result_recv_request(:), limits(:)
    integer, allocatable :: top_send_request(:), bottom_send_request(:)
    integer, allocatable :: top_recv_request(:), bottom_recv_request(:)
    integer, allocatable :: mpi_statuses(:,:)

    ! MPI send/recv tags, arbitrary

    integer, parameter :: bottom_recv_tag = 111
    integer, parameter :: top_recv_tag    = 222
    integer, parameter :: result_recv_tag = 333

    integer :: max_threads, my_thread
!$  integer :: omp_get_max_threads

    ! Just for measuring the kernel performance
    real*8 kernel_time
    integer*8 kernel_flops
real*8 ttts
    integer tb_start, tb_end,k 
 
    integer ::my_pe, n_pes

    call MPI_Comm_rank(MPI_COMM_WORLD, my_pe, mpierr)
    call MPI_Comm_size(MPI_COMM_WORLD, n_pes, mpierr)

 

    ttts = mpi_wtime()


    kernel_time = 1.d-100
    kernel_flops = 0

    max_threads = 1
!$  max_threads = omp_get_max_threads()

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
        thread_width = 0
        stripe_width = 0
        stripe_count = 0
    else
        ! Suggested stripe width is 48 since 48*64 real*8 numbers should fit into
        ! every primary cache
        !thread_width = (l_nev-1)/max_threads + 1 ! number of eigenvectors per OMP thread
        !stripe_width = 48 ! Must be a multiple of 4
        !stripe_count = (thread_width-1)/stripe_width + 1
        !! Adapt stripe width so that last one doesn't get too small
        !stripe_width = (thread_width-1)/stripe_count + 1
        !stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 4 !!!

        stripe_width = 32
        thread_width = l_nev
        stripe_count = (thread_width-1)/stripe_width + 1
               
    !    stripe_width = 4 
    !    thread_width = 1
    !    stripe_count = 1
        max_threads  = (l_nev-1)/stripe_width+1
    endif


!    print *, "max_threads = ", max_threads
!    print *, "stripe_width= ", stripe_width
!    print *, "stripe_count= ", stripe_count


    ! Determine the matrix distribution at the beginning

    allocate(limits(0:np_rows))

    call determine_workload(na, nbw, np_rows, limits)

    max_blk_size = maxval(limits(1:np_rows) - limits(0:np_rows-1))

    a_dim2 = max_blk_size + nbw

    allocate(a(stripe_width,a_dim2,stripe_count,max_threads))
    allocate(a_host(stripe_width,a_dim2,stripe_count,max_threads))
    ! a(:,:,:,:) should be set to 0 in a parallel region, not here!

    allocate(row(l_nev))
    row(:) = 0

    ! Copy q from a block cyclic distribution into a distribution with contiguous rows,
    ! and transpose the matrix using stripes of given stripe_width for cache blocking.

    ! The peculiar way it is done below is due to the fact that the last row should be
    ! ready first since it is the first one to start below

    ! Please note about the OMP usage below:
    ! This is not for speed, but because we want the matrix a in the memory and
    ! in the cache of the correct thread (if possible)
!print *,'distri at ',mpi_wtime()-ttts

!$omp parallel do private(my_thread), schedule(static, 1)
    do my_thread = 1, max_threads
        a_host(:,:,:,my_thread) = 0 ! if possible, do first touch allocation!
    enddo

    do ip = np_rows-1, 0, -1
        if(my_prow == ip) then
            ! Receive my rows which have not yet been received
            src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
            do i=limits(ip)+1,limits(ip+1)
                src = mod((i-1)/nblk, np_rows)
                if(src < my_prow) then
                    call MPI_Recv(row, l_nev, MPI_REAL8, src, 0, mpi_comm_rows, mpi_status, mpierr)
!$omp parallel do private(my_thread), schedule(static, 1)
                    do my_thread = 1, max_threads
                        call unpack_row(row,i-limits(ip),my_thread)
                    enddo
                elseif(src==my_prow) then
                    src_offset = src_offset+1
                    row(:) = q(src_offset, 1:l_nev)
!$omp parallel do private(my_thread), schedule(static, 1)
                    do my_thread = 1, max_threads
                        call unpack_row(row,i-limits(ip),my_thread)
                    enddo
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
                    call MPI_Recv(row, l_nev, MPI_REAL8, src, 0, mpi_comm_rows, mpi_status, mpierr)
!$omp parallel do private(my_thread), schedule(static, 1)
                    do my_thread = 1, max_threads
                        call unpack_row(row,i-limits(my_prow),my_thread)
                    enddo
                endif
            enddo
        endif
    enddo
    
!    call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!    do i = 0, n_pes-1
!       if(i .eq.  my_pe) print*, 'a = ', a_host(1:4,:,:,1)
!       call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!    end do



!print *,'copy at ',mpi_wtime()-ttts
    a = a_host
!print *,'copy done at ',mpi_wtime()-ttts


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
!       print *, "posting receive for l_nev*nblk = ", l_nev * nblk
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

!    print *, "Allocating top_border_send_buffer with ", stripe_width*nbw*max_threads, stripe_count
    allocate(top_border_send_buffer(stripe_width*nbw*max_threads, stripe_count))
    allocate(top_border_recv_buffer(stripe_width*nbw*max_threads, stripe_count))
    allocate(bottom_border_send_buffer(stripe_width*nbw*max_threads, stripe_count))
    allocate(bottom_border_recv_buffer(stripe_width*nbw*max_threads, stripe_count))

    top_border_send_buffer(:,:) = 0
    top_border_recv_buffer(:,:) = 0
    bottom_border_send_buffer(:,:) = 0
    bottom_border_recv_buffer(:,:) = 0
!print *,'alloc buf at ',mpi_wtime()-ttts

    ! Initialize broadcast buffer

!    print *, "nbw = ", nbw
!    print *, "Max_blk_size = ", max_blk_size
    allocate(bcast_buffer(nbw, max_blk_size))
    bcast_buffer = 0
    allocate(bcast_buffer_dev(nbw, max_blk_size))
    bcast_buffer_dev = 0
    allocate(tau_dev(max_blk_size))
    tau_dev = 0

    current_tv_off = 0 ! Offset of next row to be broadcast


    ! ------------------- start of work loop -------------------

    a_off = 0 ! offset in A (to avoid unnecessary shifts)

    top_msg_length = 0
    bottom_msg_length = 0
!print *,'starting at ',mpi_wtime()-ttts

    do sweep = 0, (na-1)/nbw
!        print *, " sweep number = ", sweep
!        print *, " my_prow = ", my_prow
!        print *, " l_nev = ", l_nev

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

        ! this branch is not taken
        if(sweep==0 .and. current_n_end < current_n .and. l_nev > 0) then
            do i = 1, stripe_count
                csw = min(stripe_width, thread_width-(i-1)*stripe_width) ! "current_stripe_width"
                b_len = csw*nbw*max_threads
!                print *, "posting irecv for b_len = ", b_len
!                print *, "posting irecv for i = ", i
                call MPI_Irecv(bottom_border_recv_buffer(1,i), b_len, MPI_REAL8, my_prow+1, bottom_recv_tag, &
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
        bcast_buffer_dev(:,1:current_local_n) = bcast_buffer(:,1:current_local_n)
        bcast_buffer_dev(1,:) = 1.
        tau_dev(1:current_local_n) = bcast_buffer(1,1:current_local_n)

!        if(my_pe .eq. 1) print *, "bcast_buffer= ", bcast_buffer(:, 1:current_local_n)

        if(l_nev == 0) cycle

!        print *, " current_local_n prior to branch = ", current_local_n
        if(current_local_n > 0) then

          do i = 1, stripe_count

            ! Get real stripe width for strip i;
            ! The last OpenMP tasks may have an even smaller stripe with,
            ! but we don't care about this, i.e. we send/recv a bit too much in this case.
            ! csw: current_stripe_width
!            a_host = a
!            print *, "a_stripe = ", a_host

            csw = min(stripe_width, thread_width-(i-1)*stripe_width)

            !wait_b
            if(current_n_end < current_n) then
!print *,'Comm 1'
                call MPI_Wait(bottom_recv_request(i), mpi_status, mpierr)
!$omp parallel do private(my_thread, n_off, b_len, b_off), schedule(static, 1)
                do my_thread = 1, max_threads
                    n_off = current_local_n+a_off
                    b_len = csw*nbw
                    b_off = (my_thread-1)*b_len
                    a(1:csw,n_off+1:n_off+nbw,i,my_thread) = &
                      reshape(bottom_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, nbw /))
!+++ debugging .. we do not come across here
!                    a_host = a 
!                    print*, "a 0  = ", a_host(1:csw,n_off+1:n_off+nbw,i,1)
!                    print*, "Bottom_border_recv_buffer= ", bottom_border_recv_buffer(1:csw*nbw,i)
!+++ debugging 
                enddo

                if(next_n_end < next_n) then
                    call MPI_Irecv(bottom_border_recv_buffer(1,i), csw*nbw*max_threads, &
                                   MPI_REAL8, my_prow+1, bottom_recv_tag, &
                                   mpi_comm_rows, bottom_recv_request(i), mpierr)
                endif
            endif

!            print*, "current_local_n = ", current_local_n
!            print*, "bottom_msg_length = ", bottom_msg_length
!            print*, "top_msg_length = ", top_msg_length
            if(current_local_n <= bottom_msg_length + top_msg_length) then

!++++ debug this branch is also not taken
!                print *, "Taking the first branch "
                !wait_t
                if(top_msg_length>0) then
                    call MPI_Wait(top_recv_request(i), mpi_status, mpierr)
                endif

                !compute
!$omp parallel do private(my_thread, n_off, b_len, b_off), schedule(static, 1)
                do my_thread = 1, max_threads
                    if(top_msg_length>0) then
!print *,'Comm 2'
                        b_len = csw*top_msg_length
                        b_off = (my_thread-1)*b_len
                        a(1:csw,a_off+1:a_off+top_msg_length,i,my_thread) = &
                          reshape(top_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, top_msg_length /))
                    endif
                enddo
                    call compute_hh_trafo(0, current_local_n, i)

                !send_b
                call MPI_Wait(bottom_send_request(i), mpi_status, mpierr)
                if(bottom_msg_length>0) then
!print *,'Comm 3'
                    n_off = current_local_n+nbw-bottom_msg_length+a_off
                    b_len = csw*bottom_msg_length*max_threads
                    bottom_border_send_buffer(1:b_len,i) = &
                        reshape(a(1:csw,n_off+1:n_off+bottom_msg_length,i,:), (/ b_len /))
                    call MPI_Isend(bottom_border_send_buffer(1,i), b_len, MPI_REAL8, my_prow+1, &
                                   top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
                endif

            else

!++++ debug: this branch is taken on rank 0
!                    a_host = a 
!                    print*, "a 1 = ", a_host(1:4, :, :, 1)
            

                !compute
!!$omp parallel do private(my_thread, b_len, b_off), schedule(static, 1)
                !do my_thread = 1, max_threads
!                print*, "About to compute trafor 1"
                     call compute_hh_trafo(current_local_n - bottom_msg_length, bottom_msg_length, i)
!                print*, "completed compute trafor 1"
                !enddo
!++++ debug
!                    a_host = a 
!                    print*, "a = ", a_host(1:4, :, :, 1)
!++++ debug

                !send_b
                call MPI_Wait(bottom_send_request(i), mpi_status, mpierr)
                if(bottom_msg_length > 0) then
!print *,'Comm 4'
                    n_off = current_local_n+nbw-bottom_msg_length+a_off
                    b_len = csw*bottom_msg_length*max_threads
!                    bottom_border_send_buffer(1:b_len,i) = &
!                      reshape(a(1:csw,n_off+1:n_off+bottom_msg_length,i,:), (/ b_len /))

                    do j=1, max_threads
                      do k=1, bottom_msg_length
                         !tb_start = csw * (k - 1)  + csw*bottom_msg_length*(j - 1)
                         tb_start = csw * (k - 1)  + csw*bottom_msg_length*(j - 1) + 1
                         tb_end = csw * k   + csw*bottom_msg_length*(j - 1)
                         bottom_border_send_buffer(tb_start:tb_end,i) = a(1:csw, n_off+k, i, j)
                      enddo
                     enddo

                    call MPI_Isend(bottom_border_send_buffer(1,i), b_len, MPI_REAL8, my_prow+1, &
                                   top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
                endif
!++++ debug
!                    a_host = a 
!                    print*, "a = ", a_host
!++++ debug

!                print *, "top_msg_length    = ", top_msg_length
!                print *, "current_local     = ", current_local_n
!                print *, "bottom_msg_length =", bottom_msg_length
              
                !compute
!$omp parallel do private(my_thread), schedule(static, 1)
                !do my_thread = 1, max_threads
!                    print *, "About to compute trafo 2"
                    call compute_hh_trafo(top_msg_length, current_local_n-top_msg_length-bottom_msg_length, i)
!                    print *, "completed compute trafo 2"
                !enddo

!++++ debug
! the following transfer seems to be important
!                    a_host = a 
!                    print*, "a = ", a_host(1:4, :, :, 1)
!++++ debug




                !wait_t
                if(top_msg_length>0) then
                    call MPI_Wait(top_recv_request(i), mpi_status, mpierr)
                endif

                !compute
!$omp parallel do private(my_thread, b_len, b_off), schedule(static, 1)
                do my_thread = 1, max_threads
                    if(top_msg_length>0) then
!print *,'Comm 5'
                        b_len = csw*top_msg_length
                        b_off = (my_thread-1)*b_len
                        a(1:csw,a_off+1:a_off+top_msg_length,i,my_thread) = &
                          reshape(top_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, top_msg_length /))
                    endif
                enddo
!                    print *, "About to compute trafo 3"

                    call compute_hh_trafo(0, top_msg_length, i)
!                    print *, "Completed compute trafo 3"
            endif

            if(next_top_msg_length > 0) then
                !request top_border data
                b_len = csw*next_top_msg_length*max_threads
                call MPI_Irecv(top_border_recv_buffer(1,i), b_len, MPI_REAL8, my_prow-1, &
                               top_recv_tag, mpi_comm_rows, top_recv_request(i), mpierr)
            endif

            !send_t
            if(my_prow > 0) then
!print *,'Comm 6'
                call MPI_Wait(top_send_request(i), mpi_status, mpierr)
                b_len = csw*nbw*max_threads
                !PM HACK: is the following a problem? Was commented out..
                !print *, "max size of top_border_send_buffer",  stripe_width*nbw*max_threads, stripe_count
                !print *, "rearranging top_border_send_buffer to ",blen
                !top_border_send_buffer(1:b_len,i) = reshape(a(1:csw,a_off+1:a_off+nbw,i,:), (/ b_len /))

                do j=1, max_threads
                  do k=1, nbw
                     !tb_start = csw * (k - 1)  + csw*nbw*(j - 1)
                     tb_start = csw * (k - 1)  + csw*nbw*(j - 1) + 1
                     tb_end = csw * k   + csw*nbw*(j - 1)
                     top_border_send_buffer(tb_start:tb_end,i) = a(1:csw, a_off+k, i, j)
                  enddo
                enddo
                call MPI_Isend(top_border_send_buffer(1,i), b_len, MPI_REAL8, &
                               my_prow-1, bottom_recv_tag, &
                               mpi_comm_rows, top_send_request(i), mpierr)
            endif

            ! Care that there are not too many outstanding top_recv_request's
            if(stripe_count > 1) then
                if(i>1) then
                    call MPI_Wait(top_recv_request(i-1), mpi_status, mpierr)
                else
                    call MPI_Wait(top_recv_request(stripe_count), mpi_status, mpierr)
                endif
            endif

          enddo

          top_msg_length = next_top_msg_length

        else
            ! wait for last top_send_request
!          print *, 'Waiting for send request'
          do i = 1, stripe_count
            call MPI_Wait(top_send_request(i), mpi_status, mpierr)
          enddo
        endif

        ! Care about the result
!++++ debugging
!    a_host = a
!    call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!    do i = 0, n_pes-1
!       if(i .eq.  my_pe) print*, 'a = ', a_host(:,:,:,1)
!       call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!    end do
!++++ debugging


!if(.false.)then
        if(my_prow == 0) then

            ! topmost process sends nbw rows to destination processes

            do j=0,nfact-1

                num_blk = sweep*nfact+j ! global number of destination block, 0 based
                if(num_blk*nblk >= na) exit

                nbuf = mod(num_blk, num_result_buffers) + 1 ! buffer number to get this block

                call MPI_Wait(result_send_request(nbuf), mpi_status, mpierr)

                dst = mod(num_blk, np_rows)

!+++ debugging: added the transfer from a here.
!+++            we should see if we can create a pack_row version that operates directly
!+++            on the GPU.
!                a_host = a
!+++++++++++++++++++++++++++++

                if(dst == 0) then
                    do i = 1, min(na - num_blk*nblk, nblk)
!                        call pack_row(row, j*nblk+i+a_off)
                        call pack_row_device(row, j*nblk+i+a_off)
                        q((num_blk/np_rows)*nblk+i,1:l_nev) = row(:)
                    enddo
                else
                    do i = 1, nblk
!                        call pack_row(result_buffer(:,i,nbuf),j*nblk+i+a_off)
                        call pack_row_device(result_buffer(:,i,nbuf),j*nblk+i+a_off)
                    enddo
!                    print *, "send buffer = ", result_buffer(1:l_nev, 1:nblk, nbuf)
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

                !print*, "next_local_n = ", next_local_n

                if(next_local_n > 0) then
                    call MPI_Test(result_recv_request(nbuf), flag, mpi_status, mpierr)
                    !print*, "after testing, at nbuf= ",nbuf 
                    !print*, "after testing, flag is ", flag
                    if(.not.flag) exit
                else
                    !print*, "entering mpi_wait for recv request", nbuf
                    call MPI_Wait(result_recv_request(nbuf), mpi_status, mpierr)
                endif

                ! Fill result buffer into q
                num_blk = j*np_rows + my_prow ! global number of current block, 0 based
                do i = 1, min(na - num_blk*nblk, nblk)
                    q(j*nblk+i, 1:l_nev) = result_buffer(1:l_nev, i, nbuf)
                enddo
                
!                print *, "q buffer after recv= ",  q(1:ldq, 1:l_nev)

                ! Queue result buffer again if there are outstanding blocks left
                if(j+num_result_buffers < num_result_blocks) &
                    call MPI_Irecv(result_buffer(1,1,nbuf), l_nev*nblk, MPI_REAL8, 0, result_recv_tag, &
                                   mpi_comm_rows, result_recv_request(nbuf), mpierr)

            enddo
            num_bufs_recvd = j

        endif
!endif

!++++ debugging
!    a_host = a
!    call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!    do i = 0, n_pes-1
!       if(i .eq.  my_pe) print*, 'a = ', a_host(:,:,:,1)
!       call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!    end do
!++++ debugging

        ! Shift the remaining rows to the front of A (if necessary)

        offset = nbw - top_msg_length
        if(offset<0) then
            print *,'internal error, offset for shifting = ',offset
            call MPI_Abort(MPI_COMM_WORLD, 1, mpierr)
        endif
        a_off = a_off + offset
        if(a_off + next_local_n + nbw > a_dim2) then
!$omp parallel do private(my_thread, i, j), schedule(static, 1)
            do my_thread = 1, max_threads
                do i = 1, stripe_count
                    do j = top_msg_length+1, top_msg_length+next_local_n
!print *,'shift'
!stop
                       !A(:,j,i,my_thread) = A(:,j+a_off,i,my_thread)
                       mpierr = cudamemcpy(A(1,j,i,my_thread), A(1,j+a_off,i,my_thread),stripe_width) 
                    enddo
                enddo
            enddo
            a_off = 0
        endif

    enddo

!print *,'Calc done at:',mpi_wtime()-ttts
!    a_host = a


!    call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!    do i = 0, n_pes-1
!       if(i .eq.  my_pe) print*, 'a = ', a_host(:,:,:,1)
!       call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!    end do



!print *,'copy back done at:',mpi_wtime()-ttts

!++++ Debugging: No clue what this is doing here..  (There's nothing equivalent in the original code)
!++++            Removing for the time being
!    a_off = 0
!    do sweep = 0, (na-1)/nbw
!
!            ! topmost process sends nbw rows to destination processes
!
!            do j=0,nfact-1
!
!                num_blk = sweep*nfact+j ! global number of destination block, 0 based
!                if(num_blk*nblk >= na) exit
!
!                nbuf = mod(num_blk, num_result_buffers) + 1 ! buffer number to get this block
!
!                    do i = 1, min(na - num_blk*nblk, nblk)
!                        call pack_row(row, j*nblk+i+a_off)
!                        q((num_blk/np_rows)*nblk+i,1:l_nev) = row(:)
!                    enddo
!            enddo
!
!        a_off = a_off + nbw
!    enddo
!+++++ END DEBUGGING

!   call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!    do i = 0, n_pes-1
!       if(i .eq.  my_pe) print*, 'q result = ', q(1:ldq, 1:l_nev)
!       call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!    end do

    ! Just for safety:
    if(ANY(top_send_request    /= MPI_REQUEST_NULL)) print *,'*** ERROR top_send_request ***',my_prow,my_pcol
    if(ANY(bottom_send_request /= MPI_REQUEST_NULL)) print *,'*** ERROR bottom_send_request ***',my_prow,my_pcol
    if(ANY(top_recv_request    /= MPI_REQUEST_NULL)) print *,'*** ERROR top_recv_request ***',my_prow,my_pcol
    if(ANY(bottom_recv_request /= MPI_REQUEST_NULL)) print *,'*** ERROR bottom_recv_request ***',my_prow,my_pcol

    if(my_prow == 0) then
        allocate(mpi_statuses(MPI_STATUS_SIZE,num_result_buffers))
        call MPI_Waitall(num_result_buffers, result_send_request, mpi_statuses, mpierr)
        deallocate(mpi_statuses)
    endif

    !print *, "result send request", result_send_request
    !print *, "result recv request", result_recv_request

    if(ANY(result_send_request /= MPI_REQUEST_NULL)) print *,'*** ERROR result_send_request ***',my_prow,my_pcol
    if(ANY(result_recv_request /= MPI_REQUEST_NULL)) print *,'*** ERROR result_recv_request ***',my_prow,my_pcol

    if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
        print '(" Kernel time:",f10.3," MFlops: ",f10.3)', kernel_time, kernel_flops/kernel_time*1.d-6

    ! deallocate all working space

    deallocate(a)
    deallocate(a_host)
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
    deallocate(bcast_buffer_dev)
    deallocate(tau_dev)
    deallocate(top_send_request)
    deallocate(top_recv_request)
    deallocate(bottom_send_request)
    deallocate(bottom_recv_request)

contains

    subroutine pack_row(row, n)
        real*8 row(:)
        integer n, i, noff, nl, nt

        do nt = 1, max_threads
            do i = 1, stripe_count
                noff = (nt-1)*thread_width + (i-1)*stripe_width
                nl   = min(stripe_width, nt*thread_width-noff, l_nev-noff)
                if(nl<=0) exit
                row(noff+1:noff+nl) = a_host(1:nl,n,i,nt)
            enddo
        enddo

    end subroutine

   subroutine pack_row_device(row, n)
        real*8 row(:)
        integer n, i, noff, nl, nt

        do nt = 1, max_threads
            do i = 1, stripe_count
                noff = (nt-1)*thread_width + (i-1)*stripe_width
                nl   = min(stripe_width, nt*thread_width-noff, l_nev-noff)
                if(nl<=0) exit
                row(noff+1:noff+nl) = a(1:nl,n,i,nt)
            enddo
        enddo

    end subroutine


    subroutine unpack_row(row, n, my_thread)

        ! Private variables in OMP regions (my_thread) should better be in the argument list!
        integer, intent(in) :: n, my_thread
        real*8, intent(in)  :: row(:)
        integer i, noff, nl

        do i=1,stripe_count
            noff = (my_thread-1)*thread_width + (i-1)*stripe_width
            nl   = min(stripe_width, my_thread*thread_width-noff, l_nev-noff)
            if(nl<=0) exit
            a_host(1:nl,n,i,my_thread) = row(noff+1:noff+nl)
        enddo

    end subroutine

    subroutine compute_hh_trafo(off, ncols, istripe)

        ! Private variables in OMP regions (my_thread) should better be in the argument list!
        integer, intent(in) :: off, ncols, istripe
        integer i, j, nl, noff, my_thread
        real*8 w(nbw,2), ttt, tau, v(stripe_width), hh(nbw), s
        real*8, allocatable :: sss(:)
        real*8, device, allocatable :: ss(:)

!        print *, "inside compute_hh_trafo, ncols = ", ncols
        if(ncols <= 0) return

        allocate(sss(ncols))
        allocate(ss(ncols))

        ttt = mpi_wtime()
!        if(istripe<stripe_count) then
!          nl = stripe_width
!        else
!          noff = (my_thread-1)*thread_width + (istripe-1)*stripe_width
!          nl = min(my_thread*thread_width-noff, l_nev-noff)
!          if(nl<=0) return
!        endif

         nl = stripe_width

!        do j = ncols, 2, -2
!            w(:,1) = bcast_buffer(1:nbw,j+off)
!            w(:,2) = bcast_buffer(1:nbw,j+off-1)
!            call double_hh_trafo(a(1,j+off+a_off-1,istripe,my_thread), w, nbw, nl, stripe_width, nbw)
!        enddo
!        if(j==1) call single_hh_trafo(a(1,1+off+a_off,istripe,my_thread),bcast_buffer(1,off+1), nbw, nl, stripe_width)

!!!        do j = ncols, 1, -1
!!!          !!!call single_hh_trafo(a(1,j+off+a_off,istripe,my_thread),bcast_buffer(1,off+j), nbw, nl, stripe_width)
!!!
!!!          !hh(1) = 1.d0
!!!          !hh(2:nbw) = bcast_buffer(2:nbw,off+j)
!!!
!!!          call Dgemv('N',nl,nbw,1.d0,a(1,j+off+a_off,istripe,my_thread),ubound(a,1),bcast_buffer_dev(1,off+j),1,0.d0,v,1)
!!!          call Dger(nl,nbw,-tau_dev(off+j),v,1,bcast_buffer_dev(1,off+j),1,a(1,j+off+a_off,istripe,my_thread),ubound(a,1))
!!!          !call cublasDgemv('N',nl,nbw,1.d0,a(1,j+off+a_off,istripe,my_thread),ubound(a,1),bcast_buffer_dev(1,off+j),1,0.d0,v_dev,1)
!!!          !call cublasDger(nl,nbw,-bcast_buffer(1,off+j),v_dev,1,bcast_buffer_dev(1,off+j),1,a(1,j+off+a_off,istripe,my_thread),ubound(a,1))
!!!
!!!        enddo
!         do my_thread = 1, max_threads

  sss = 0
  do j = ncols, 2, -2
   s = bcast_buffer(2,off+j-1)*1
   do i=3,32
      s = s+bcast_buffer(i,off+j-1)*bcast_buffer(i-1,off+j)
   enddo
   sss(j) = s
  enddo
  ss = sss

!        print *, "a_off = ", a_off
!        print *, "istripe= ", istripe
!        print *, "off = ", off
!        print *, "bcast_buffer = ", bcast_buffer
        call hh_kernel<<<max_threads,32>>>(a(1,1+off+a_off,istripe,1),bcast_buffer_dev(1,off+1),tau_dev(off+1),ss,ncols,nbw,ubound(a,2))
        ! call hh_kernel<<<max_threads,4>>>(a(1,1,1,1),bcast_buffer_dev(1,off+1),tau_dev(off+1),ss,ncols,nbw,ubound(a,2))
!         print *, "hh_kernel", cudaGetLastError()
         j = cudaThreadSynchronize()
!         enddo
!        if(my_thread==1) then
            kernel_flops = kernel_flops + 4*int(nl,8)*int(ncols,8)*int(nbw,8)*max_threads
            kernel_time  = kernel_time + mpi_wtime()-ttt
!        endif

    end subroutine

end subroutine

!-------------------------------------------------------------------------------

subroutine single_hh_trafo(q, hh, nb, nq, ldq)

    ! Perform single real Householder transformation.
    ! This routine is not performance critical and thus it is coded here in Fortran

    implicit none
    integer nb, nq, ldq
    real*8 q(ldq, *), hh(*)

    integer i
    real*8 v(nq)

    ! v = q * hh
    v(:) = q(1:nq,1)
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


attributes(global) subroutine hh_kernel(a,hh,tau,ss,ncols,nbw,a_dim2)

  real*8 :: a(32,a_dim2,1,*), hh(nbw,*), tau(*), ss(*)
  integer, value :: ncols, nbw, a_dim2

  integer i, j, jj, n, mythread, myblock, myoff, myt, aoff
  real*8 :: z1, z2, z3, s
  real*8, shared :: ash(32,32+7), zz(32,2), h(32)

  mythread = threadidx%x
  myblock  = blockidx%x
  if(mythread <= 16) then
    myoff = 2*(mythread-1)
    myt = 1
  else
    myoff = 2*(mythread-17)
    myt = 2
  endif

  aoff = 7
  do i = 0, 32
    ash(mythread,aoff+i) = a(mythread,ncols+i-1,1,myblock)
  enddo
  call syncthreads()

  do j = ncols, 2, -2

    s = ss(j)

    jj = j+1-myt
    z1 = 0.
    z2 = 0.
    do i = 1, 32
      z1 = z1 + ash(myoff+1,aoff+i+1-myt)*hh(i,jj)
      z2 = z2 + ash(myoff+2,aoff+i+1-myt)*hh(i,jj)
    enddo

    if(myt==1) then
      z1 = -z1*tau(j)
      z2 = -z2*tau(j)
    endif
    zz(myoff+1,myt) = z1
    zz(myoff+2,myt) = z2
    call syncthreads()

    if(myt==2) then
      zz(myoff+1,2) = -z1*tau(j-1) - zz(myoff+1,1)*tau(j-1)*s
      zz(myoff+2,2) = -z2*tau(j-1) - zz(myoff+2,1)*tau(j-1)*s
    endif
    call syncthreads()

    ash(mythread,aoff) = ash(mythread,aoff) + zz(mythread,2)
    do i = 1, 31
      ash(mythread,aoff+i) = ash(mythread,aoff+i) + zz(mythread,1)*hh(i,j) + zz(mythread,2)*hh(i+1,j-1)
    enddo
    i = 32
    ash(mythread,aoff+i) = ash(mythread,aoff+i) + zz(mythread,1)*hh(i,j)

    a(mythread,j+nbw-1,1,myblock) = ash(mythread,aoff+nbw)
    a(mythread,j+nbw-2,1,myblock) = ash(mythread,aoff+nbw-1)

    if(j>3) then
      aoff = aoff-2
      if(aoff<1) then
        aoff = 7
        do i = 31,1,-1
         ash(mythread,aoff+i+1) = ash(mythread,i)
        enddo
      endif
      ash(mythread,aoff  ) = a(mythread,j-3,1,myblock)
      ash(mythread,aoff+1) = a(mythread,j-2,1,myblock)
    endif

    call syncthreads()
      
  enddo


  do i = 0, nbw-2
    a(mythread,j+i+1,1,myblock) = ash(mythread,aoff+i)
  enddo

  call syncthreads()

!----------------------------------------------------
  if(j==1) then
    z1 = 0.
    do i = 1, 32
      z1 = z1 + a(mythread,i,1,myblock)*hh(i,1)
    enddo
    z1 = z1 * tau(1)
    do i = 1, 32
      a(mythread,i,1,myblock) = a(mythread,i,1,myblock) - z1*hh(i,1)
    enddo
  endif
!----------------------------------------------------

  call syncthreads()

end subroutine
  
