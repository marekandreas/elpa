! after the decomposition: v contains all vectors from the process
! (including the ones calculated by other process columns)

! TODO: cleanup huge number of variables
subroutine tum_pdgeqrf_2dcomm(a,lda,v,ldv,tau,t,ldt,work,lwork,m,n,mb,nb,rowidx,colidx,rev,trans,PQRPARAM,mpicomm_rows,mpicomm_cols,blockheuristic)
    use ELPA1
    use tum_utils
    use mpi
  
    implicit none
 
    ! parameter setup
    INTEGER     gmode_,rank_,eps_
    PARAMETER   (gmode_ = 1,rank_ = 2,eps_=3)

    ! input variables (local)
    integer lda,lwork,ldv,ldt
    double precision a(lda,*),v(ldv,*),tau(*),work(*),t(ldt,*)

    ! input variables (global)
    integer m,n,mb,nb,rowidx,colidx,rev,trans,mpicomm_cols,mpicomm_rows
    integer PQRPARAM(*)
 
    ! output variables (global)
    double precision blockheuristic(*)

    ! input variables derived from PQRPARAM
    integer updatemode,tmerge,size2d

    ! local scalars
    integer mpierr,mpirank_cols,broadcast_size,mpirank_rows
    integer mpirank_cols_qr,mpiprocs_cols
    integer lcols_temp,lcols,icol,lastcol
    integer baseoffset,offset,idx,voffset
    integer update_voffset,update_tauoffset
    integer update_lcols,ivector
    integer work_offset

    double precision dbroadcast_size(1),dtmat_bcast_size(1)
    double precision pdgeqrf_size(1),pdlarft_size(1),pdlarfb_size(1),tmerge_pdlarfb_size(1)
    integer temptau_offset,temptau_size,broadcast_offset,tmat_bcast_size
    integer remaining_cols
    integer total_cols
    integer incremental_update_start,incremental_update_size ! needed for incremental update mode
 
    size2d = PQRPARAM(1)
    updatemode = PQRPARAM(2)
    tmerge = PQRPARAM(3)

    ! filter values
    !if (rev .eq. 1) then
    !    n = min(rowidx,colidx)
    !else
    !    n = min(m-rowidx+1,n-colidx+1)
    !end if

    ! copy value before we are going to filter it
    total_cols = n

    !print *,'going to decompose:',rowidx,colidx,n

    ! derive needed input variables from PQRPARAM

    call mpi_comm_rank(mpicomm_cols,mpirank_cols,mpierr)
    call mpi_comm_rank(mpicomm_rows,mpirank_rows,mpierr)
    call mpi_comm_size(mpicomm_cols,mpiprocs_cols,mpierr)
  
   
    call tum_pdgeqrf_1dcomm(a,lda,v,ldv,tau,t,ldt,pdgeqrf_size(1),-1,m,total_cols,mb,rowidx,rowidx,rev,trans,PQRPARAM(4),mpicomm_rows,blockheuristic)
    call tum_pdgeqrf_pack_unpack(v,ldv,dbroadcast_size(1),-1,m,total_cols,mb,rowidx,rowidx,rev,0,mpicomm_rows)
    call tum_pdgeqrf_pack_unpack_tmatrix(tau,t,ldt,dtmat_bcast_size(1),-1,total_cols,0)
    !call tum_pdlarft_1dcomm(m,mb,total_cols,v,ldv,tau,t,ldt,rowidx,rowidx,rev,mpicomm_rows,pdlarft_size(1),-1)
    pdlarft_size(1) = 0.0d0
    call tum_pdlarfb_1dcomm(m,mb,total_cols,total_cols,a,lda,v,ldv,tau,t,ldt,rowidx,rowidx,rev,mpicomm_rows,pdlarfb_size(1),-1)
    call tum_tmerge_pdlarfb_1dcomm(m,mb,total_cols,total_cols,total_cols,v,ldv,work,t,ldt,a,lda,rowidx,rowidx,rev,updatemode,mpicomm_rows,tmerge_pdlarfb_size(1),-1)


    temptau_offset = 1
    temptau_size = total_cols
    broadcast_offset = temptau_offset + temptau_size 
    broadcast_size = dbroadcast_size(1) + dtmat_bcast_size(1)
    work_offset = broadcast_offset + broadcast_size

    !call MPI_Barrier(mpicomm_cols,mpierr)
    !if (mpirank_cols .eq. 0) then
    !    print *,'work sizes', temptau_size, broadcast_size, pdgeqrf_size
    !end if
    !call MPI_Barrier(mpicomm_cols,mpierr)
    !if (mpirank_cols .eq. 1) then
    !    print *,'work sizes', temptau_size, broadcast_size, pdgeqrf_size
    !end if
    !call MPI_Barrier(mpicomm_cols,mpierr)

    if (lwork .eq. -1) then
        !print *,'broadcast_size',broadcast_size,tmerge_pdlarfb_size(1)
        work(1) = (DBLE(temptau_size) + DBLE(broadcast_size) + max(pdgeqrf_size(1),pdlarft_size(1),pdlarfb_size(1),tmerge_pdlarfb_size(1)))
        return
    end if

    !if (rev .eq. 1) then
        lastcol = colidx-total_cols+1
        !lastcol = total_cols-colidx+1
        voffset = total_cols
    !else
    !    lastcol = total_cols+colidx-1
    !    voffset = 1
    !end if
  
    incremental_update_size = 0
 
    ! clear v buffer: just ensure that there is no junk in the upper triangle
    ! part, otherwise pdlarfb gets some problems
    ! pdlarfl(2) do not have these problems as they are working more on a vector
    ! basis
    v(1:ldv,1:total_cols) = 0.0d0
 
    icol = colidx

    ! TODO: find a valid filter rule, this one prevents multiprocess mode
    ! however a filter rule is needed for rowidx != 1
    !remaining_cols = min(m-rowidx+1,total_cols)
    remaining_cols = total_cols

    !print *,'start decomposition',m,rowidx,colidx
 
    !print *,'start matrix'
    !print *,a(1,1),a(1,2),a(1,3)
    !print *,a(2,1),a(2,2),a(2,3)
    !print *,a(3,1),a(3,2),a(3,3)
    !print *,a(4,1),a(4,2),a(4,3)
    !print *,a(5,1),a(5,2),a(5,3)


    do while (remaining_cols .gt. 0)

        ! determine rank of process column with next qr block
        mpirank_cols_qr = MOD((icol-1)/nb,mpiprocs_cols)

        ! lcols can't be larger than than nb
        ! exception: there is only one process column
 
        ! however, we might not start at the first local column.
        ! therefore assume a matrix of size (1xlcols) starting at (1,icol)
        ! determine the real amount of local columns
        !if (rev .eq. 1) then
            lcols_temp = min(nb,(icol-lastcol+1))

            ! blocking parameter
            lcols_temp = max(min(lcols_temp,size2d),1)

            ! determine size from last decomposition column
            !  to first decomposition column
            call local_size_offset_1d(icol,nb,icol-lcols_temp+1,icol-lcols_temp+1,0, &
                                      mpirank_cols_qr,mpiprocs_cols, &
                                      lcols,baseoffset,offset)
 
            voffset = remaining_cols - lcols + 1
        !else
        !    lcols_temp = min(nb,(lastcol-icol+1))
 
        !    ! blocking parameter
        !    lcols_temp = max(min(lcols_temp,size2d),1)

        !    call local_size_offset_1d(icol+lcols_temp-1,nb,icol,icol,0, &
        !                              mpirank_cols_qr,mpiprocs_cols, &
        !                              lcols,baseoffset,offset)
        !end if

        idx = rowidx - colidx + icol

        if (mpirank_cols .eq. mpirank_cols_qr) then 
            ! qr decomposition part

            tau(offset:offset+lcols-1) = 0.0d0

            call tum_pdgeqrf_1dcomm(a(1,offset),lda,v(1,voffset),ldv,tau(offset),t(voffset,voffset),ldt,work(work_offset),lwork,m,lcols,mb,rowidx,idx,rev,trans,PQRPARAM(4),mpicomm_rows,blockheuristic)
            !print *,'offset voffset',offset,voffset,idx
    
            ! pack broadcast buffer (v + tau)
            call tum_pdgeqrf_pack_unpack(v(1,voffset),ldv,work(broadcast_offset),lwork,m,lcols,mb,rowidx,idx,rev,0,mpicomm_rows)
     
            ! determine broadcast size
            call tum_pdgeqrf_pack_unpack(v(1,voffset),ldv,dbroadcast_size(1),-1,m,lcols,mb,rowidx,idx,rev,0,mpicomm_rows)
            broadcast_size = dbroadcast_size(1)
  
            !if (mpirank_rows .eq. 0) then
            ! pack tmatrix into broadcast buffer and calculate new size
            call tum_pdgeqrf_pack_unpack_tmatrix(tau(offset),t(voffset,voffset),ldt,work(broadcast_offset+broadcast_size),lwork,lcols,0)
            call tum_pdgeqrf_pack_unpack_tmatrix(tau(offset),t(voffset,voffset),ldt,dtmat_bcast_size(1),-1,lcols,0)
            broadcast_size = broadcast_size + dtmat_bcast_size(1)
            !end if
 
            ! initiate broadcast (send part)
            call MPI_Bcast(work(broadcast_offset),broadcast_size,mpi_real8, &
                           mpirank_cols_qr,mpicomm_cols,mpierr)

            ! copy tau parts into temporary tau buffer
            work(temptau_offset+voffset-1:temptau_offset+(voffset-1)+lcols-1) = tau(offset:offset+lcols-1)

            !print *,'generated tau:', tau(offset)
        else
            ! vector exchange part

            ! determine broadcast size
            call tum_pdgeqrf_pack_unpack(v(1,voffset),ldv,dbroadcast_size(1),-1,m,lcols,mb,rowidx,idx,rev,1,mpicomm_rows)
            broadcast_size = dbroadcast_size(1)
 
            call tum_pdgeqrf_pack_unpack_tmatrix(work(temptau_offset+voffset-1),t(voffset,voffset),ldt,dtmat_bcast_size(1),-1,lcols,0)
            tmat_bcast_size = dtmat_bcast_size(1)

            !print *,'broadcast_size (nonqr)',broadcast_size
            !if (mpirank_rows .eq. 0) then
            !    ! we also have the tmatrix in our broadcast buffer
                 broadcast_size = dbroadcast_size(1) + dtmat_bcast_size(1)
            !end if
 
            ! initiate broadcast (recv part)
            call MPI_Bcast(work(broadcast_offset),broadcast_size,mpi_real8, &
                           mpirank_cols_qr,mpicomm_cols,mpierr)
            
            ! last n*n elements in buffer are (still empty) T matrix elements
            ! fetch from first process in each column
 
            ! unpack broadcast buffer (v + tau)
            call tum_pdgeqrf_pack_unpack(v(1,voffset),ldv,work(broadcast_offset),lwork,m,lcols,mb,rowidx,idx,rev,1,mpicomm_rows)
 
            ! now send t matrix to other processes in our process column
            broadcast_size = dbroadcast_size(1)
            tmat_bcast_size = dtmat_bcast_size(1)

            !call MPI_Bcast(work(broadcast_offset+broadcast_size),tmat_bcast_size,mpi_real8, &
            !                    0,mpicomm_rows,mpierr)

            ! t matrix should now be available on all processes => unpack
            call tum_pdgeqrf_pack_unpack_tmatrix(work(temptau_offset+voffset-1),t(voffset,voffset),ldt,work(broadcast_offset+broadcast_size),lwork,lcols,1)
        end if

        ! merge small T matrix to final T matrix
        ! obsolete: we already generated the smaller T matrix part during the column decomposition
        ! generate t matrix
        !call tum_pdlarft_1dcomm(m,mb,n-voffset+1,v(1,voffset),ldv,work(temptau_offset+voffset-1),t(voffset,voffset),ldt,rowidx,idx,rev,mpicomm_rows,work(work_offset),lwork)
        !do ivector=1,lcols
        !    !print *,'writing tau(qr)', voffset, ivector, work(temptau_offset+voffset-1+ivector-1)
        !    t(voffset+ivector-1,voffset+ivector-1) =  work(temptau_offset+voffset-1+ivector-1)
        !end do

        remaining_cols = remaining_cols - lcols
 
        ! apply householder vectors to whole trailing matrix parts (if any)
        ! TODO: toggle behavior using parameters (full/incremental update)

        update_voffset = voffset
        update_tauoffset = icol
        update_lcols = lcols
        incremental_update_size = incremental_update_size + lcols
 
        ! TODO: determine size of nextrank

        !if (rev .eq. 1) then
            icol = icol - lcols
            ! count colums from first column of global block to current index
            call local_size_offset_1d(icol,nb,colidx-n+1,colidx-n+1,0, &
                                      mpirank_cols,mpiprocs_cols, &
                                      lcols,baseoffset,offset)
        !else
        !    icol = icol + lcols
        !    voffset = voffset + lcols
        !    call local_size_offset_1d(lastcol,nb,icol,icol,0, &
        !                              mpirank_cols,mpiprocs_cols, &
        !                              lcols,baseoffset,offset)
        !end if

        if (lcols .gt. 0) then
            !print *,'updating trailing matrix'
            !if (rev .eq. 1) then
                if (updatemode .eq. ichar('I')) then
                    print *,'pdgeqrf_2dcomm: incremental update not yet implemented! rev=1'
                else if (updatemode .eq. ichar('F')) then
                    ! full update no merging
                    call tum_pdlarfb_1dcomm(m,mb,lcols,update_lcols,a(1,offset),lda,v(1,update_voffset),ldv, &
                                            work(temptau_offset+update_voffset-1),t(update_voffset,update_voffset),ldt, &
                                            rowidx,idx,1,mpicomm_rows,work(work_offset),lwork)
                else 
                    ! full update + merging default
                    call tum_tmerge_pdlarfb_1dcomm(m,mb,lcols,n-(update_voffset+update_lcols-1),update_lcols,v(1,update_voffset),ldv, &
                                                   work(temptau_offset+update_voffset-1),t(update_voffset,update_voffset),ldt, &
                                                   a(1,offset),lda,rowidx,idx,1,updatemode,mpicomm_rows,work(work_offset),lwork)
                end if
            !else  ! rev .ne. 1
            !    if (updatemode .eq. ichar('I')) then
            !        print *,'pdgeqrf_2dcomm: incremental update not yet implemented! rev=0'
            !    else if (updatemode .eq. ichar('F')) then
            !        ! full update - no merging
            !        call tum_pdlarfb_1dcomm(m,mb,lcols,update_lcols,a(1,offset),lda,v(1,update_voffset),ldv,&
            !                                work(temptau_offset+update_voffset-1),t(update_voffset,update_voffset),ldt, &
            !                                rowidx,idx,rev,mpicomm_rows,work(work_offset),lwork)
            !    else
            !        ! full update + merging
            !        call tum_tmerge_pdlarfb_1dcomm(m,mb,lcols,update_voffset-1,update_lcols,v,ldv,tau,t,ldt, & 
            !                                      a(1,offset),lda,rowidx,idx,0,updatemode,mpicomm_rows,work(work_offset),lwork)
            !    end if
            !end if
            !print *,'updating trailing matrix done'
        else
            !if (rev .eq. 1) then
                if (updatemode .eq. ichar('I')) then
                    print *,'sole merging of (incremental) T matrix', mpirank_cols, n-(update_voffset+incremental_update_size-1)
                    call tum_tmerge_pdlarfb_1dcomm(m,mb,0,n-(update_voffset+incremental_update_size-1),incremental_update_size,v(1,update_voffset),ldv, &
                                                   work(temptau_offset+update_voffset-1),t(update_voffset,update_voffset),ldt, &
                                                   a,lda,rowidx,idx,1,updatemode,mpicomm_rows,work(work_offset),lwork)

                    ! reset for upcoming incremental updates
                    incremental_update_size = 0
                else if (updatemode .eq. ichar('M')) then
                    ! final merge
                    call tum_tmerge_pdlarfb_1dcomm(m,mb,0,n-(update_voffset+update_lcols-1),update_lcols,v(1,update_voffset),ldv, &
                                                   work(temptau_offset+update_voffset-1),t(update_voffset,update_voffset),ldt, &
                                                   a,lda,rowidx,idx,1,updatemode,mpicomm_rows,work(work_offset),lwork)
                else 
                    ! full updatemode - nothing to update
                end if

                ! reset for upcoming incremental updates
                incremental_update_size = 0
            !else 
            !    if (updatemode .eq. ichar('I')) then
            !        print *,'pdgeqrf_2dcomm: incremental update not yet implemented! rev=0'
            !    else if (updatemode .eq. ichar('M')) then
            !        ! final merging
            !        call tum_tmerge_pdlarfb_1dcomm(m,mb,0,update_voffset-1,update_lcols,v,ldv,tau,t,ldt, & 
            !                                       a,lda,rowidx,idx,0,updatemode,mpicomm_rows,work(work_offset),lwork)
            !    end if
            !end if
        end if
    end do
 
    if ((tmerge .gt. 0) .and. (updatemode .eq. ichar('F'))) then
        ! finally merge all small T parts
        ! TODO: due to distribution specific properties, some 1d properties do
        ! not hold (e.g. smallest parts may come at the beginning instead of the end) => find largest subsets and merge them.
        !call tum_pdlarft_set_merge_1dcomm(m,mb,n,size2d,v,ldv,tau,t,ldt,baseidx,idx,rev,mpicomm_rows,work,lwork)
        call tum_pdlarft_tree_merge_1dcomm(m,mb,n,size2d,tmerge,v,ldv,tau,t,ldt,rowidx,rowidx,rev,mpicomm_rows,work,lwork)
    end if
    !print *,'stop decomposition',rowidx,colidx
 
    !print *,'final matrix'
    !print *,a(1,1),a(1,2),a(1,3)
    !print *,a(2,1),a(2,2),a(2,3)
    !print *,a(3,1),a(3,2),a(3,3)
    !print *,a(4,1),a(4,2),a(4,3)
    !print *,a(5,1),a(5,2),a(5,3)


end subroutine
! after the decomposition: v contains all vectors from the process
! (including the ones calculated by other process columns)

! TODO: cleanup huge number of variables
! TODO: emphasize use of remaining_cols
subroutine tum_pdgeqrf_replica_2dcomm(a,lda,v,ldv,tau,t,ldt,work,lwork,m,n,mb,nb,rowidx,colidx,rev,trans,PQRPARAM,mpicomm_rows,mpicomm_cols,blockheuristic)
    use ELPA1
    use tum_utils
    use mpi
  
    implicit none
 
    ! parameter setup
    INTEGER     gmode_,rank_,eps_
    PARAMETER   (gmode_ = 1,rank_ = 2,eps_=3)

    ! input variables (local)
    integer lda,lwork,ldv,ldt
    double precision a(lda,*),v(ldv,*),tau(*),work(*),t(ldt,*)

    ! input variables (global)
    integer m,n,mb,nb,rowidx,colidx,rev,trans,mpicomm_cols,mpicomm_rows
    integer PQRPARAM(*)
 
    ! output variables (global)
    double precision blockheuristic(*)

    ! input variables derived from PQRPARAM
    integer updatemode,tmerge,size2d

    ! local scalars
    integer mpierr,mpirank_cols,broadcast_size,mpirank_rows
    integer mpirank_cols_qr,mpiprocs_cols
    integer lcols_temp,lcols,icol,lastcol
    integer baseoffset,offset,idx,voffset
    integer update_voffset,update_tauoffset
    integer update_lcols,ivector
    integer work_offset,ldb

    double precision dbroadcast_size(1),dtmat_bcast_size(1)
    double precision pdgeqrf_size(1),pdlarft_size(1),pdlarfb_size(1),tmerge_pdlarfb_size(1),replicate_size(1)
    integer temptau_offset,temptau_size
    integer gatherrecv_offset,gatherrecv_size
    integer remaining_cols,total_cols

    size2d = PQRPARAM(1)
    updatemode = PQRPARAM(2)
    tmerge = PQRPARAM(3)

    ! copy value before we are going to filter it
    total_cols = n

    call mpi_comm_size(mpicomm_cols,mpiprocs_cols,mpierr)

    if (n .gt. mpiprocs_cols*nb) then
        ! replication is not possible in this case
        call tum_pdgeqrf_2dcomm(a,lda,v,ldv,tau,t,ldt,work,lwork,m,n,mb,nb,rowidx,colidx,rev,trans,PQRPARAM,mpicomm_rows,mpicomm_cols,blockheuristic)
        return
    end if
  
    call mpi_comm_rank(mpicomm_cols,mpirank_cols,mpierr)
    call mpi_comm_rank(mpicomm_rows,mpirank_rows,mpierr)
   
    call tum_pdgeqrf_1dcomm(a,lda,v,ldv,tau,t,ldt,pdgeqrf_size(1),-1,m,total_cols,mb,rowidx,rowidx,rev,trans,PQRPARAM(4),mpicomm_rows,blockheuristic)
    call tum_replicate_submatrix_2dcomm(m,n,mb,nb,rowidx,colidx,a,lda,replicate_size(1),-1,mpicomm_rows,mpicomm_cols,rev,work,ldb)

    temptau_offset = 1
    temptau_size = total_cols
    gatherrecv_offset = temptau_offset + temptau_size 
    gatherrecv_size = total_cols * lda
    work_offset = gatherrecv_offset + gatherrecv_size

    if (lwork .eq. -1) then
        work(1) = DBLE(temptau_size + gatherrecv_size) + max(pdgeqrf_size(1),replicate_size(1))
        return
    end if

    v(1:ldv,1:total_cols) = 0.0d0
 
    ! replicate submatrix
    call tum_replicate_submatrix_2dcomm(m,n,mb,nb,rowidx,colidx,a,lda,work(work_offset),lwork,mpicomm_rows,mpicomm_cols,rev,work(gatherrecv_offset),ldb)

    !print *,'original matrix'
    !print *,a(1,1),a(1,2),a(1,3)
    !print *,a(2,1),a(2,2),a(2,3)
    !print *,a(3,1),a(3,2),a(3,3)
    !print *,a(4,1),a(4,2),a(4,3)
    !print *,a(5,1),a(5,2),a(5,3)

    !print *,'replicated matrix'
    !print *,work(gatherrecv_offset),work(gatherrecv_offset+ldb),work(gatherrecv_offset+2*ldb)
    !print *,work(gatherrecv_offset+1),work(gatherrecv_offset+ldb+1),work(gatherrecv_offset+2*ldb+1)
    !print *,work(gatherrecv_offset+2),work(gatherrecv_offset+ldb+2),work(gatherrecv_offset+2*ldb+2)
    !print *,work(gatherrecv_offset+3),work(gatherrecv_offset+ldb+3),work(gatherrecv_offset+2*ldb+3)
    !print *,work(gatherrecv_offset+4),work(gatherrecv_offset+ldb+4),work(gatherrecv_offset+2*ldb+4)

    ! run 1d qr decomposition on replicated matrix

    if (ldb .eq. 0) then
        ldb = lda  ! may become zero due to replication => workaround to avoid dger error messages
    end if

    call tum_pdgeqrf_1dcomm(work(gatherrecv_offset),ldb,v,ldv,work(temptau_offset),t,ldt,work(work_offset),lwork,m,total_cols,mb,rowidx,rowidx,rev,trans,PQRPARAM(4),mpicomm_rows,blockheuristic)
 
    !print *,'final matrix'
    !print *,work(gatherrecv_offset),work(gatherrecv_offset+ldb),work(gatherrecv_offset+2*ldb)
    !print *,work(gatherrecv_offset+1),work(gatherrecv_offset+ldb+1),work(gatherrecv_offset+2*ldb+1)
    !print *,work(gatherrecv_offset+2),work(gatherrecv_offset+ldb+2),work(gatherrecv_offset+2*ldb+2)
    !print *,work(gatherrecv_offset+3),work(gatherrecv_offset+ldb+3),work(gatherrecv_offset+2*ldb+3)
    !print *,work(gatherrecv_offset+4),work(gatherrecv_offset+ldb+4),work(gatherrecv_offset+2*ldb+4)

    ! extract modified replicated parts and new tau values on original process columns
    call tum_merge_submatrix_2dcomm(m,n,mb,nb,rowidx,colidx,work(gatherrecv_offset),ldb,a,lda,mpicomm_rows,mpicomm_cols,rev)
    call tum_merge_subvector_1dcomm(total_cols,nb,colidx,work(temptau_offset),tau,mpicomm_cols,rev)

    !print *,'merged matrix'
    !print *,a(1,1),a(1,2),a(1,3)
    !print *,a(2,1),a(2,2),a(2,3)
    !print *,a(3,1),a(3,2),a(3,3)
    !print *,a(4,1),a(4,2),a(4,3)
    !print *,a(5,1),a(5,2),a(5,3)

end subroutine

subroutine tum_pdgeqrf_1dcomm(a,lda,v,ldv,tau,t,ldt,work,lwork,m,n,mb,baseidx,rowidx,rev,trans,PQRPARAM,mpicomm,blockheuristic)
    use ELPA1
    use mpi
  
    implicit none
 
    ! parameter setup
    INTEGER     gmode_,rank_,eps_
    PARAMETER   (gmode_ = 1,rank_ = 2,eps_=3)

    ! input variables (local)
    integer lda,lwork,ldv,ldt
    double precision a(lda,*),v(ldv,*),tau(*),t(ldt,*),work(*)

    ! input variables (global)
    integer m,n,mb,baseidx,rowidx,rev,trans,mpicomm
    integer PQRPARAM(*)

    ! derived input variables
  
    ! derived further input variables from TUM_PQRPARAM
    integer size1d,updatemode,tmerge

    ! output variables (global)
    double precision blockheuristic(*)
 
    ! local scalars
    integer nr_blocks,remainder,current_block,aoffset,idx,updatesize
    double precision pdgeqr2_size(1),pdlarfb_size(1),tmerge_tree_size(1)

    size1d = max(min(PQRPARAM(1),n),1)
    updatemode = PQRPARAM(2)
    tmerge = PQRPARAM(3)

    if (lwork .eq. -1) then
        call tum_pdgeqr2_1dcomm(a,lda,v,ldv,tau,t,ldt,pdgeqr2_size,-1, & 
                                m,size1d,mb,baseidx,baseidx,rev,trans,PQRPARAM(4),mpicomm,blockheuristic)

        !call tum_tmerge_pdlarfb_1dcomm(m,mb,n,n,size1d,v,ldv,tau,t,ldt, &
        !                               a,lda,baseidx,idx,rev,updatemode,mpicomm,pdlarfb_size,-1)

        ! reserve more space for incremental mode
        call tum_tmerge_pdlarfb_1dcomm(m,mb,n,n,n,v,ldv,tau,t,ldt, &
                                       a,lda,baseidx,baseidx,rev,updatemode,mpicomm,pdlarfb_size,-1)
 
        call tum_pdlarft_tree_merge_1dcomm(m,mb,n,size1d,tmerge,v,ldv,tau,t,ldt,baseidx,baseidx,rev,mpicomm,tmerge_tree_size,-1)

        work(1) = max(pdlarfb_size(1),pdgeqr2_size(1),tmerge_tree_size(1))
        return
    end if

        nr_blocks = n / size1d
        remainder = n - nr_blocks*size1d
  
        current_block = 0
        do while (current_block .lt. nr_blocks)
            idx = rowidx-current_block*size1d
            updatesize = n-(current_block+1)*size1d
            aoffset = 1+updatesize

            call tum_pdgeqr2_1dcomm(a(1,aoffset),lda,v(1,aoffset),ldv,tau(aoffset),t(aoffset,aoffset),ldt,work,lwork, & 
                                    m,size1d,mb,baseidx,idx,1,trans,PQRPARAM(4),mpicomm,blockheuristic)

            if (updatemode .eq. ichar('M')) then
                ! full update + merging
                call tum_tmerge_pdlarfb_1dcomm(m,mb,updatesize,current_block*size1d,size1d, & 
                                               v(1,aoffset),ldv,tau(aoffset),t(aoffset,aoffset),ldt, &
                                               a,lda,baseidx,idx,1,ichar('F'),mpicomm,work,lwork)
            else if (updatemode .eq. ichar('I')) then
                if (updatesize .ge. size1d) then
                    ! incremental update + merging
                    call tum_tmerge_pdlarfb_1dcomm(m,mb,size1d,current_block*size1d,size1d, & 
                                                   v(1,aoffset),ldv,tau(aoffset),t(aoffset,aoffset),ldt, &
                                                   a(1,aoffset-size1d),lda,baseidx,idx,1,updatemode,mpicomm,work,lwork)

                else ! only remainder left
                    ! incremental update + merging
                    call tum_tmerge_pdlarfb_1dcomm(m,mb,remainder,current_block*size1d,size1d, & 
                                                   v(1,aoffset),ldv,tau(aoffset),t(aoffset,aoffset),ldt, &
                                                   a(1,1),lda,baseidx,idx,1,updatemode,mpicomm,work,lwork)
                end if
            else ! full update no merging is default
                ! full update no merging
                call tum_pdlarfb_1dcomm(m,mb,updatesize,size1d,a,lda,v(1,aoffset),ldv, &
                                        tau(aoffset),t(aoffset,aoffset),ldt,baseidx,idx,1,mpicomm,work,lwork)
            end if

            !call tum_tmerge_pdlarfb_1dcomm(m,mb,0,current_block*size1d,size1d, & 
            !                               v(1,aoffset),ldv,tau(aoffset),t(aoffset,aoffset),ldt, &
            !                               a,lda,baseidx,idx,1,'F',mpicomm,work,lwork)

            ! move on to next block
            current_block = current_block+1
        end do

        if (remainder .gt. 0) then
            aoffset = 1
            idx = rowidx-size1d*nr_blocks
            call tum_pdgeqr2_1dcomm(a(1,aoffset),lda,v,ldv,tau,t,ldt,work,lwork, &
                                    m,remainder,mb,baseidx,idx,1,trans,PQRPARAM(4),mpicomm,blockheuristic)

            if ((updatemode .eq. ichar('I')) .or. (updatemode .eq. ichar('M'))) then
                ! final merging
                call tum_tmerge_pdlarfb_1dcomm(m,mb,0,size1d*nr_blocks,remainder, & 
                                               v,ldv,tau,t,ldt, &
                                               a,lda,baseidx,idx,1,updatemode,mpicomm,work,lwork) ! updatemode argument does not matter
            end if
        end if
 
    if ((tmerge .gt. 0) .and. (updatemode .eq. ichar('F'))) then
        ! finally merge all small T parts
        !call tum_pdlarft_set_merge_1dcomm(m,mb,n,size1d,v,ldv,tau,t,ldt,baseidx,idx,rev,mpicomm,work,lwork)
        call tum_pdlarft_tree_merge_1dcomm(m,mb,n,size1d,tmerge,v,ldv,tau,t,ldt,baseidx,idx,rev,mpicomm,work,lwork)
    end if

end subroutine
! local a and tau are assumed to be positioned at the right column from a local
! perspective
! TODO: if local amount of data turns to zero the algorithm might produce wrong
! results (probably due to old buffer contents)
subroutine tum_pdgeqr2_1dcomm(a,lda,v,ldv,tau,t,ldt,work,lwork,m,n,mb,baseidx,rowidx,rev,trans,PQRPARAM,mpicomm,blockheuristic)
    use ELPA1
    use mpi
  
    implicit none
 
    ! parameter setup
    INTEGER     gmode_,rank_,eps_,upmode1_
    PARAMETER   (gmode_ = 1,rank_ = 2,eps_=3, upmode1_=4)

    ! input variables (local)
    integer lda,lwork,ldv,ldt
    double precision a(lda,*),v(ldv,*),tau(*),t(ldt,*),work(*)

    ! input variables (global)
    integer m,n,mb,baseidx,rowidx,rev,trans,mpicomm
    integer PQRPARAM(*)
 
    ! output variables (global)
    double precision blockheuristic(*)
 
    ! derived further input variables from TUM_PQRPARAM
    integer maxrank,hgmode,updatemode

    ! local scalars
    integer icol,incx,idx,topidx,top,ivector,irank
    double precision pdlarfg_size(1),pdlarf_size(1),topbak,temp_size(1),total_size
    double precision pdlarfg2_size(1),pdlarfgk_size(1),pdlarfl2_size(1)
    double precision pdlarft_size(1),pdlarfb_size(1),pdlarft_pdlarfb_size(1),tmerge_pdlarfb_size(1)
    integer mpirank,mpiprocs,mpierr,mpirank_top
    integer rank,lastcol,actualrank,nextrank
    integer update_cols,decomposition_cols
    integer current_column
 
    maxrank = min(PQRPARAM(1),n)
    updatemode = PQRPARAM(2)
    hgmode = PQRPARAM(4)

    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)

    if (trans .eq. 1) then
        incx = lda
    else
        incx = 1
    end if
    
    if (lwork .eq. -1) then
        call tum_pdlarfg_1dcomm(a,incx,tau(1),pdlarfg_size(1),-1,n,rowidx,mb,hgmode,rev,mpicomm)
        call tum_pdlarfl_1dcomm(v,1,baseidx,a,lda,tau(1),pdlarf_size(1),-1,m,n,rowidx,mb,rev,mpicomm)
        call tum_pdlarfg2_1dcomm_ref(a,lda,tau,t,ldt,v,ldv,baseidx,pdlarfg2_size(1),-1,m,n,rowidx,mb,PQRPARAM,rev,mpicomm,actualrank)
        call tum_pdlarfgk_1dcomm(a,lda,tau,t,ldt,v,ldv,baseidx,pdlarfgk_size(1),-1,m,n,rowidx,mb,PQRPARAM,rev,mpicomm,actualrank)
        call tum_pdlarfl2_tmatrix_1dcomm(v,ldv,baseidx,a,lda,t,ldt,pdlarfl2_size(1),-1,m,n,rowidx,mb,rev,mpicomm)
        !call tum_pdlarft_1dcomm(m,mb,n,v,ldv,tau,t,ldt,baseidx,rowidx,1,mpicomm,pdlarft_size(1),-1)
        pdlarft_size(1) = 0.0d0
        call tum_pdlarfb_1dcomm(m,mb,n,n,a,lda,v,ldv,tau,t,ldt,baseidx,rowidx,1,mpicomm,pdlarfb_size(1),-1)
        !call tum_pdlarft_pdlarfb_1dcomm(m,mb,n,0,n,v,ldv,tau,t,ldt,a,lda,baseidx,idx,rev,mpicomm,pdlarft_pdlarfb_size(1),-1)
        pdlarft_pdlarfb_size(1) = 0.0d0
        call tum_tmerge_pdlarfb_1dcomm(m,mb,n,n,n,v,ldv,work,t,ldt,a,lda,rowidx,idx,rev,updatemode,mpicomm,tmerge_pdlarfb_size(1),-1)

        total_size = max(pdlarfg_size(1),pdlarf_size(1),pdlarfg2_size(1),pdlarfgk_size(1),pdlarfl2_size(1),pdlarft_size(1),pdlarfb_size(1),pdlarft_pdlarfb_size(1),tmerge_pdlarfb_size(1))

        work(1) = total_size
        return
    end if

        icol = 1
        lastcol = min(rowidx,n)
        decomposition_cols = lastcol
        update_cols = n
        do while (decomposition_cols .gt. 0) ! local qr block
            icol = lastcol-decomposition_cols+1
            idx = rowidx-icol+1

            ! get possible rank size
            ! limited by number of columns and remaining rows
            rank = min(n-icol+1,maxrank,idx)

            current_column = n-icol+1-rank+1

            if (rank .eq. 1) then
                !print *,'rank 1 called',n-icol+1
                call tum_pdlarfg_1dcomm(a(1,current_column),incx, &
                                        tau(current_column),work,lwork, &
                                        m,idx,mb,hgmode,1,mpicomm)

                v(1:ldv,current_column) = 0.0d0
                call tum_pdlarfg_copy_1dcomm(a(1,current_column),incx, &
                                             v(1,current_column),1, &
                                             m,baseidx,idx,mb,1,mpicomm)

                ! initialize t matrix part
                t(current_column,current_column) = tau(current_column)

                actualrank = 1
                !print *,'actualrank 1'
            else if (rank .eq. 2) then 
                call tum_pdlarfg2_1dcomm_ref(a(1,current_column),lda,tau(current_column), &
                                             t(current_column,current_column),ldt,v(1,current_column),ldv, &
                                            baseidx,work,lwork,m,rank,idx,mb,PQRPARAM,1,mpicomm,actualrank)
            
                 !print '(a,4f)','calc.   t',t(n-icol,n-icol),t(n-icol,n-icol+1),t(n-icol+1,n-icol),t(n-icol+1,n-icol+1)
                !print *,'rank 2:',actualrank,rank
            else 
                call tum_pdlarfgk_1dcomm(a(1,current_column),lda,tau(current_column), &
                                         t(current_column,current_column),ldt,v(1,current_column),ldv, &
                                         baseidx,work,lwork,m,rank,idx,mb,PQRPARAM,1,mpicomm,actualrank)
 
                !print *,'calc.   t',actualrank,rank
                !print '(3f)',t(1,1:3)
                !print '(3f)',t(2,1:3)
                !print '(3f)',t(3,1:3)

                !call tum_pdlarft_1dcomm(m,mb,actualrank,v(1,current_column+(rank-actualrank)),ldv,tau(current_column+(rank-actualrank)), & 
                !                        t(current_column+(rank-actualrank),current_column+(rank-actualrank)),ldt,baseidx,idx,1,mpicomm,work,lwork)
 
                !print *,'working t',actualrank,rank
                !print '(3f)',t(1,1:3)
                !print '(3f)',t(2,1:3)
                !print '(3f)',t(3,1:3)
            end if
  
            blockheuristic(actualrank) = blockheuristic(actualrank) + 1

            ! the blocked decomposition versions already updated their non
            ! decomposed parts using their information after communication
            update_cols = decomposition_cols - rank
            decomposition_cols = decomposition_cols - actualrank
 
            ! needed for incremental update
            nextrank = min(n-(lastcol-decomposition_cols+1)+1,maxrank,rowidx-(lastcol-decomposition_cols+1)+1)

            if (current_column .gt. 1) then
                idx = rowidx-icol+1
 
                if (updatemode .eq. ichar('I')) then
                    ! incremental update + merging
                    call tum_tmerge_pdlarfb_1dcomm(m,mb,nextrank-(rank-actualrank),n-(current_column+rank-1),actualrank,v(1,current_column+(rank-actualrank)),ldv,tau(current_column+(rank-actualrank)), &
                                                   t(current_column+(rank-actualrank),current_column+(rank-actualrank)),ldt, &
                                                   a(1,current_column-nextrank+(rank-actualrank)),lda,baseidx,idx,rev,updatemode,mpicomm,work,lwork)
                else
                    ! full update + merging
                    call tum_tmerge_pdlarfb_1dcomm(m,mb,update_cols,n-(current_column+rank-1),actualrank,v(1,current_column+(rank-actualrank)),ldv,tau(current_column+(rank-actualrank)), &
                                                   t(current_column+(rank-actualrank),current_column+(rank-actualrank)),ldt, &
                                                   a(1,1),lda,baseidx,idx,rev,updatemode,mpicomm,work,lwork)
                end if
            else
                ! TODO: build a merge only version
                call tum_tmerge_pdlarfb_1dcomm(m,mb,0,n-(current_column+rank-1),actualrank,v(1,current_column+(rank-actualrank)),ldv,tau(current_column+(rank-actualrank)), &
                                               t(current_column+(rank-actualrank),current_column+(rank-actualrank)),ldt, &
                                               a,lda,baseidx,idx,rev,updatemode,mpicomm,work,lwork)
            end if

        end do
end subroutine
! incx == 1: column major
! incx != 1: row major

subroutine tum_pdlarfg_1dcomm(x,incx,tau,work,lwork,n,idx,nb,hgmode,rev,mpi_comm)
    use ELPA1
    use tum_utils
    use mpi
    
    implicit none
 
    ! parameter setup
    INTEGER     gmode_,rank_,eps_
    PARAMETER   (gmode_ = 1,rank_ = 2,eps_=3)

    ! input variables (local)
    integer incx,lwork,hgmode
    double precision x(*),work(*)

    ! input variables (global)
    integer mpi_comm,nb,idx,n,rev

    ! output variables (global)
    double precision tau

    ! local scalars
    integer mpierr,mpirank,mpiprocs,mpirank_top
    integer sendsize,recvsize
    integer local_size,local_offset,baseoffset
    integer topidx,top,iproc
    double precision alpha,beta,xnorm,dot,xf

    ! external functions
    double precision ddot,dlapy2,dnrm2
    external ddot,dscal,dlapy2,dnrm2

    ! intrinsic
    intrinsic sign

    !if (rev .eq. 1) then
        if (idx .le. 1) then
            tau = 0.0d0
            !print *,'hg 1: no more data'
            return
        end if
    !else
    !    if (idx .ge. n) then
    !        tau = 0.0d0
    !        return
    !    end if
    !end if

    call MPI_Comm_rank(mpi_comm, mpirank, mpierr)
    call MPI_Comm_size(mpi_comm, mpiprocs, mpierr)

    ! calculate expected work size and store in work(1)
    if (hgmode .eq. ichar('s')) then
        ! allreduce (MPI_SUM)
        sendsize = 2
        recvsize = sendsize
    else if (hgmode .eq. ichar('x')) then
        ! alltoall
        sendsize = mpiprocs*2
        recvsize = sendsize
    else if (hgmode .eq. ichar('g')) then
        ! allgather
        sendsize = 2
        recvsize = mpiprocs*sendsize
    else
        ! no exchange at all (benchmarking)
        sendsize = 2
        recvsize = sendsize
    end if

    if (lwork .eq. -1) then
        work(1) = DBLE(sendsize + recvsize)
        return
    end if
 
    ! Processor id for global index of top element
    mpirank_top = MOD((idx-1)/nb,mpiprocs)
    if (mpirank .eq. mpirank_top) then
        topidx = local_index(idx,mpirank_top,mpiprocs,nb,0)
        top = 1+(topidx-1)*incx
    end if
 
    !if (rev .eq. 1) then
        call local_size_offset_1d(n,nb,idx,idx-1,rev,mpirank,mpiprocs, &
                                  local_size,baseoffset,local_offset)
    !else
    !    call local_size_offset_1d(n,nb,idx,idx+1,rev,mpirank,mpiprocs, &
    !                              local_size,baseoffset,local_offset)
    !end if

    local_offset = local_offset * incx

    ! calculate and exchange information
    if (hgmode .eq. ichar('s')) then
        if (mpirank .eq. mpirank_top) then
            alpha = x(top)
        else 
            alpha = 0.0d0
        end if

        dot = ddot(local_size, &
                   x(local_offset), incx, &
                   x(local_offset), incx)

        !dot =  dot_product(x(local_offset:local_offset+local_size-1), x(local_offset:local_offset+local_size-1))

        work(1) = alpha
        work(2) = dot
        
        call mpi_allreduce(work(1),work(sendsize+1), &
                           sendsize,mpi_real8,mpi_sum, &
                           mpi_comm,mpierr)

        alpha = work(sendsize+1)
        xnorm = sqrt(work(sendsize+2))
    else if (hgmode .eq. ichar('x')) then
        if (mpirank .eq. mpirank_top) then
            alpha = x(top)
        else 
            alpha = 0.0d0
        end if

        xnorm = dnrm2(local_size, x(local_offset), incx)

        do iproc=0,mpiprocs-1
            work(2*iproc+1) = alpha
            work(2*iproc+2) = xnorm
        end do

        call mpi_alltoall(work(1),2,mpi_real8, &
                          work(sendsize+1),2,mpi_real8, &
                          mpi_comm,mpierr)

        ! extract alpha value
        alpha = work(sendsize+1+mpirank_top*2)

        ! copy norm parts of buffer to beginning
        do iproc=0,mpiprocs-1
            work(iproc+1) = work(sendsize+1+2*iproc+1)
        end do

        xnorm = dnrm2(mpiprocs, work(1), 1)
    else if (hgmode .eq. ichar('g')) then
        if (mpirank .eq. mpirank_top) then
            alpha = x(top)
        else 
            alpha = 0.0d0
        end if

        xnorm = dnrm2(local_size, x(local_offset), incx)
        work(1) = alpha
        work(2) = xnorm

        ! allgather
        call mpi_allgather(work(1),sendsize,mpi_real8, &
                          work(sendsize+1),sendsize,mpi_real8, &
                          mpi_comm,mpierr)

        ! extract alpha value
        alpha = work(sendsize+1+mpirank_top*2)
 
        ! copy norm parts of buffer to beginning
        do iproc=0,mpiprocs-1
            work(iproc+1) = work(sendsize+1+2*iproc+1)
        end do

        xnorm = dnrm2(mpiprocs, work(1), 1)
    else
        ! dnrm2
        xnorm = dnrm2(local_size, x(local_offset), incx)

        if (mpirank .eq. mpirank_top) then
            alpha = x(top)
        else 
            alpha = 0.0d0
        end if

        ! no exchange at all (benchmarking)
 
        xnorm = 0.0d0
    end if

    !print *,'ref hg:', idx,xnorm,alpha
    !print *,x(1:n)

    ! calculate householder information
    if (xnorm .eq. 0.0d0) then
        ! H = I

        tau = 0.0d0
    else
        ! General case

        !beta = sign(dlapy2(alpha, xnorm), alpha)
        !tau = (beta+alpha) / beta
        !xf = 1.0d0/(beta+alpha)
        !if (mpirank .eq. mpirank_top) then
        !    x(top) = -beta
        !end if

        call hh_transform_real(alpha,xnorm**2,xf,tau)
        if (mpirank .eq. mpirank_top) then
            x(top) = alpha
        end if

        call dscal(local_size, xf, &
                   x(local_offset), incx)
 
        ! TODO: reimplement norm rescale method of 
        ! original PDLARFG using mpi?

    end if

    ! useful for debugging
    !print *,'hg:mpirank,idx,beta,alpha:',mpirank,idx,beta,alpha,1.0d0/(beta+alpha),tau
    !print *,x(1:n)

end subroutine
! TODO: incx parameter missing
! TODO: make use of k parameter
subroutine tum_pdlarfg2_1dcomm_ref(a,lda,tau,t,ldt,v,ldv,baseidx,work,lwork,m,k,idx,mb,PQRPARAM,rev,mpicomm,actualk)
    implicit none
 
    ! parameter setup
    INTEGER     gmode_,rank_,eps_,upmode1_
    PARAMETER   (gmode_ = 1,rank_ = 2,eps_=3, upmode1_=4)

    ! input variables (local)
    integer lda,lwork,ldv,ldt
    double precision a(lda,*),v(ldv,*),tau(*),work(*),t(ldt,*)

    ! input variables (global)
    integer m,k,idx,baseidx,mb,rev,mpicomm
    integer PQRPARAM(*)
 
    ! output variables (global)
    integer actualk
 
    ! derived input variables from TUM_PQRPARAM
    integer eps

    ! local scalars
    integer irank,ivector
    double precision dseedwork_size(1),dseed_size
    integer seedwork_size,seed_size
    integer seedwork_offset,seed_offset
    logical accurate

    ! external functions
    logical tum_pdlarfg2_1dcomm_check
    external tum_pdlarfg2_1dcomm_check
 
    ! TODO: add fallback routines to work buffer size calculation

    call tum_pdlarfg2_1dcomm_seed(a,lda,dseedwork_size(1),-1,work,m,mb,idx,rev,mpicomm)
    seedwork_size = dseedwork_size(1)
    seed_size = seedwork_size
 
    if (lwork .eq. -1) then
        work(1) = seedwork_size + seed_size
        return
    end if

    seedwork_offset = 1
    seed_offset = seedwork_offset + seedwork_size

    eps = PQRPARAM(3)

    !print *,'hg2:',m,k,baseidx,idx

    ! check for border cases (only a 2x2 matrix left)
    !if (rev .eq. 1) then
        if (idx .le. 1) then
            tau(1:2) = 0.0d0
            t(1:2,1:2) = 0.0d0
            return
        end if
    !else
    !    if (idx .ge. m) then
    !        tau(1:2) = 0.0d0
    !        t(1:2,1:2) = 0.0d0
    !        return
    !    end if
    !end if

    call tum_pdlarfg2_1dcomm_seed(a,lda,work(seedwork_offset),lwork,work(seed_offset),m,mb,idx,rev,mpicomm)

        if (eps .gt. 0) then
            accurate = tum_pdlarfg2_1dcomm_check(work(seed_offset),eps)
        else
            accurate = .true.
        end if

        !print *,'accurate:',accurate

        !call tum_pdlarfg_1dcomm(a(1,1),1, &
        !                        tau(1),work,lwork, &
        !                        m,idx,mb,PQRPARAM,0,mpicomm)

        call tum_pdlarfg2_1dcomm_vector(a(1,2),1,tau(2),work(seed_offset), &
                                        m,mb,idx,0,1,mpicomm)

        call tum_pdlarfg_copy_1dcomm(a(1,2),1, &
                                     v(1,2),1, &
                                     m,baseidx,idx,mb,1,mpicomm)

        !call tum_pdlarfl_1dcomm(v(1,1),1,baseidx,a(1,2),lda,tau(1), &
        !                        work,lwork,m,1,idx,mb,0,mpicomm)

        call tum_pdlarfg2_1dcomm_update(v(1,2),1,baseidx,a(1,1),lda,work(seed_offset),m,idx,mb,rev,mpicomm)

        ! check for 2x2 matrix case => only one householder vector will be
        ! generated
        if (idx .gt. 2) then
            if (accurate .eqv. .true.) then
                call tum_pdlarfg2_1dcomm_vector(a(1,1),1,tau(1),work(seed_offset), &
                                                m,mb,idx-1,1,1,mpicomm)

                call tum_pdlarfg_copy_1dcomm(a(1,1),1, &
                                             v(1,1),1, &
                                             m,baseidx,idx-1,mb,1,mpicomm)

                ! generate fuse element
                call tum_pdlarfg2_1dcomm_finalize_tmatrix(work(seed_offset),tau,t,ldt)

                actualk = 2
            else
                t(1,1) = 0.0d0
                t(1,2) = 0.0d0
                t(2,2) = tau(2)

                actualk = 1
            end if
        else
            t(1,1) = 0.0d0
            t(1,2) = 0.0d0
            t(2,2) = tau(2)

            ! no more vectors to create

            ! TODO: clear space of second vector in v?
            tau(1) = 0.0d0

            actualk = 2

            !print *,'rank2: no more data'
        end if

end subroutine

! TODO: incx parameter missing
subroutine tum_pdlarfg2_1dcomm_seed(a,lda,work,lwork,seed,n,nb,idx,rev,mpicomm)
    use ELPA1
    use tum_utils
    use mpi

    implicit none

    ! input variables (local)
    integer lda,lwork
    double precision a(lda,*),work(*),seed(*)

    ! input variables (global)
    integer n,nb,idx,rev,mpicomm
 
    ! output variables (global)

    ! external functions
    double precision ddot
    external ddot

    ! local scalars
    double precision top11,top21,top12,top22
    double precision dot11,dot12,dot22
    integer mpirank,mpiprocs,mpierr
    integer mpirank_top11,mpirank_top21
    integer top11_offset,top21_offset
    integer baseoffset
    integer local_offset1,local_size1
    integer local_offset2,local_size2

    if (lwork .eq. -1) then
        work(1) = DBLE(8)
        return
    end if
  
    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)

        call local_size_offset_1d(n,nb,idx,idx-1,rev,mpirank,mpiprocs, &
                              local_size1,baseoffset,local_offset1)

        call local_size_offset_1d(n,nb,idx,idx-2,rev,mpirank,mpiprocs, &
                              local_size2,baseoffset,local_offset2)

        mpirank_top11 = MOD((idx-1)/nb,mpiprocs)
        mpirank_top21 = MOD((idx-2)/nb,mpiprocs)

        top11_offset = local_index(idx,mpirank_top11,mpiprocs,nb,0)
        top21_offset = local_index(idx-1,mpirank_top21,mpiprocs,nb,0)

        if (mpirank_top11 .eq. mpirank) then
            top11 = a(top11_offset,2)
            top12 = a(top11_offset,1)
        else
            top11 = 0.0d0
            top12 = 0.0d0
        end if

        if (mpirank_top21 .eq. mpirank) then
            top21 = a(top21_offset,2)
            top22 = a(top21_offset,1)
        else
            top21 = 0.0d0
            top22 = 0.0d0
        end if

        ! calculate 3 dot products
        dot11 = ddot(local_size1,a(local_offset1,2),1,a(local_offset1,2),1)
        dot12 = ddot(local_size1,a(local_offset1,2),1,a(local_offset1,1),1)
        dot22 = ddot(local_size2,a(local_offset2,1),1,a(local_offset2,1),1)

    ! store results in work buffer
    work(1) = top11
    work(2) = dot11
    work(3) = top12
    work(4) = dot12
    work(5) = top21
    work(6) = top22
    work(7) = dot22
    work(8) = 0.0d0 ! fill up buffer

    ! exchange partial results
    call mpi_allreduce(work, seed, 8, mpi_real8, mpi_sum, &
                       mpicomm, mpierr)
end subroutine

logical function tum_pdlarfg2_1dcomm_check(seed,eps)
    implicit none

    ! input variables
    double precision seed(*)
    integer eps

    ! local scalars
    double precision epsd,first,second,first_second,estimate
    logical accurate
    double precision dot11,dot12,dot22
    double precision top11,top12,top21,top22
 
    EPSD = EPS
  
    top11 = seed(1)
    dot11 = seed(2)
    top12 = seed(3)
    dot12 = seed(4)
        
    top21 = seed(5)
    top22 = seed(6)
    dot22 = seed(7)

    ! reconstruct the whole inner products 
    ! (including squares of the top elements)
    first = dot11 + top11*top11
    second = dot22 + top22*top22 + top12*top12
    first_second = dot12 + top11*top12

    estimate = abs((first_second*first_second)/(first*second)) 

    !print *,'estimate:',estimate
    
    ! if accurate the following check holds
    accurate = (estimate .LE. (epsd/(1.0d0+epsd)))
  
    tum_pdlarfg2_1dcomm_check = accurate
end function

! id=0: first vector
! id=1: second vector
subroutine tum_pdlarfg2_1dcomm_vector(x,incx,tau,seed,n,nb,idx,id,rev,mpicomm)
    use ELPA1
    use tum_utils
    use mpi
 
    implicit none

    ! input variables (local)
    integer lda,lwork,incx
    double precision x(*),seed(*),tau

    ! input variables (global)
    integer n,nb,idx,id,rev,mpicomm
 
    ! output variables (global)

    ! external functions
    double precision dlapy2
    external dlapy2,dscal

    ! local scalars
    integer mpirank,mpirank_top,mpiprocs,mpierr
    double precision alpha,dot,beta,xnorm
    integer local_size,baseoffset,local_offset,top,topidx

    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)

    !if (rev .eq. 1) then
        call local_size_offset_1d(n,nb,idx,idx-1,rev,mpirank,mpiprocs, &
                                  local_size,baseoffset,local_offset)
    !else
    !    call local_size_offset_1d(n,nb,idx,idx+1,rev,mpirank,mpiprocs, &
    !                              local_size,baseoffset,local_offset)
    !end if

    local_offset = local_offset * incx

    ! Processor id for global index of top element
    mpirank_top = MOD((idx-1)/nb,mpiprocs)
    if (mpirank .eq. mpirank_top) then
        topidx = local_index(idx,mpirank_top,mpiprocs,nb,0)
        top = 1+(topidx-1)*incx
    end if

    alpha = seed(id*5+1)
    dot = seed(id*5+2)
    
    xnorm = sqrt(dot)

    if (xnorm .eq. 0.0d0) then
        ! H = I

        tau = 0.0d0
    else
        ! General case

        beta = sign(dlapy2(alpha, xnorm), alpha)
        tau = (beta+alpha) / beta

        !print *,'hg2',tau,xnorm,alpha
        
        !call tum_pdscal_1dcomm(1.0d0/(beta+alpha))
        call dscal(local_size, 1.0d0/(beta+alpha), &
                   x(local_offset), incx)
 
        ! TODO: reimplement norm rescale method of 
        ! original PDLARFG using mpi?

        if (mpirank .eq. mpirank_top) then
            x(top) = -beta
        end if

        seed(8) = beta
    end if
end subroutine

subroutine tum_pdlarfg2_1dcomm_update(v,incv,baseidx,a,lda,seed,n,idx,nb,rev,mpicomm)
    use ELPA1
    use tum_utils
    use mpi
 
    implicit none

    ! input variables (local)
    integer incv,lda
    double precision v(*),a(lda,*),seed(*)

    ! input variables (global)
    integer n,baseidx,idx,nb,rev,mpicomm
 
    ! output variables (global)

    ! external functions
    external daxpy

    ! local scalars
    integer mpirank,mpiprocs,mpierr
    integer local_size,local_offset,baseoffset
    double precision z,coeff,beta
    double precision dot11,dot12,dot22
    double precision top11,top12,top21,top22
 
    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)


    ! seed should be updated by previous householder generation
    ! Update inner product of this column and next column vector
    top11 = seed(1)
    dot11 = seed(2)
    top12 = seed(3)
    dot12 = seed(4)
        
    top21 = seed(5)
    top22 = seed(6)
    dot22 = seed(7)
    beta = seed(8)
    
    call local_size_offset_1d(n,nb,baseidx,idx,rev,mpirank,mpiprocs, &
                              local_size,baseoffset,local_offset)
    baseoffset = baseoffset * incv

    z = (dot12 + top11 * top12) / beta + top12

    !print *,'hg2 update:',baseidx,idx,mpirank,local_size

    call daxpy(local_size, -z, v(baseoffset),1, a(local_offset,1),1)
    
    ! prepare a full dot22 for update
    dot22 = dot22 + top22*top22

    ! calculate coefficient
    COEFF = z / (top11 + beta)
  
    ! update inner product of next vector
    dot22 = dot22 - coeff * (2*dot12 - coeff*dot11)
 
    ! update dot12 value to represent update with first vector 
    ! (needed for T matrix)
    dot12 = dot12 - COEFF * dot11 
    
    ! update top element of next vector
    top22 = top22 - coeff * top21
    seed(6) = top22

    ! restore separated dot22 for vector generation
    seed(7) = dot22  - top22*top22 

    !------------------------------------------------------
    ! prepare elements for T matrix
    seed(4) = dot12

    ! prepare dot matrix for fuse element of T matrix
    ! replace top11 value with -beta1
    seed(1) = beta
end subroutine

! run this function after second vector
subroutine tum_pdlarfg2_1dcomm_finalize_tmatrix(seed,tau,t,ldt)
    implicit none

    integer ldt,rev
    double precision seed(*),t(ldt,*),tau(*)
    double precision dot12,beta1,top21,beta2
    double precision tau1,tau2
 
    beta1 = seed(1)
    dot12 = seed(4)
    top21 = seed(5)
    beta2 = seed(8)
 
    !print *,'beta1 beta2',beta1,beta2

    dot12 = dot12 / beta2 + top21
    dot12 = -(dot12 / beta1)

    t(1,1) = tau(1)
    t(1,2) = dot12
    t(2,2) = tau(2)
end subroutine
! TODO: implement incx parameter
subroutine tum_pdlarfgk_1dcomm(a,lda,tau,t,ldt,v,ldv,baseidx,work,lwork,m,k,idx,mb,PQRPARAM,rev,mpicomm,actualk)

    implicit none
 
    ! parameter setup

    ! input variables (local)
    integer lda,lwork,ldv,ldt
    double precision a(lda,*),v(ldv,*),tau(*),work(*),t(ldt,*)

    ! input variables (global)
    integer m,k,idx,baseidx,mb,rev,mpicomm
    integer PQRPARAM(*)
 
    ! output variables (global)
    integer actualk

    ! local scalars
    integer ivector
    double precision pdlarfg_size(1),pdlarf_size(1)
    double precision pdlarfgk_1dcomm_seed_size(1),pdlarfgk_1dcomm_check_size(1)
    double precision pdlarfgk_1dcomm_update_size(1)
    integer seedC_size,seedC_offset
    integer seedD_size,seedD_offset
    integer work_offset

    seedC_size = k*k
    seedC_offset = 1
    seedD_size = k*k
    seedD_offset = seedC_offset + seedC_size
    work_offset = seedD_offset + seedD_size

    if (lwork .eq. -1) then
        call tum_pdlarfg_1dcomm(a,1,tau(1),pdlarfg_size(1),-1,m,baseidx,mb,PQRPARAM(4),rev,mpicomm)
        call tum_pdlarfl_1dcomm(v,1,baseidx,a,lda,tau(1),pdlarf_size(1),-1,m,k,baseidx,mb,rev,mpicomm)
        call tum_pdlarfgk_1dcomm_seed(a,lda,baseidx,pdlarfgk_1dcomm_seed_size(1),-1,work,work,m,k,mb,PQRPARAM,rev,mpicomm)
        call tum_pdlarfgk_1dcomm_check(work,work,k,PQRPARAM,pdlarfgk_1dcomm_check_size(1),-1,actualk,rev)
        call tum_pdlarfgk_1dcomm_update(a,lda,baseidx,pdlarfgk_1dcomm_update_size(1),-1,work,work,k,k,1,work,m,mb,rev,mpicomm)
        work(1) = max(pdlarfg_size(1),pdlarf_size(1),pdlarfgk_1dcomm_seed_size(1),pdlarfgk_1dcomm_check_size(1),pdlarfgk_1dcomm_update_size(1)) + DBLE(seedC_size + seedD_size);
        return
    end if

        call tum_pdlarfgk_1dcomm_seed(a(1,1),lda,idx,work(work_offset),lwork,work(seedC_offset),work(seedD_offset),m,k,mb,PQRPARAM,1,mpicomm)
        call tum_pdlarfgk_1dcomm_check(work(seedC_offset),work(seedD_offset),k,PQRPARAM,work(work_offset),lwork,actualk,1)
 
        !print *,'possible rank:', actualk

        ! override useful for debugging
        !actualk = 1
        !actualk = k
        !actualk= min(actualk,2)
        do ivector=1,actualk
            call tum_pdlarfgk_1dcomm_vector(a(1,k-ivector+1),1,idx,tau(k-ivector+1), &
                                            work(seedC_offset),work(seedD_offset),k, &
                                            ivector,m,mb,rev,mpicomm)

            call tum_pdlarfgk_1dcomm_update(a(1,1),lda,idx,work(work_offset),lwork,work(seedC_offset), &
                                            work(seedD_offset),k,actualk,ivector,tau, & 
                                            m,mb,rev,mpicomm)

            call tum_pdlarfg_copy_1dcomm(a(1,k-ivector+1),1, &
                                         v(1,k-ivector+1),1, &
                                         m,baseidx,idx-ivector+1,mb,1,mpicomm)
        end do

        ! generate final T matrix and convert preliminary tau values into real ones
        call tum_pdlarfgk_1dcomm_generateT(work(seedC_offset),work(seedD_offset),k,actualk,tau,t,ldt,1)

end subroutine

subroutine tum_pdlarfgk_1dcomm_seed(a,lda,baseidx,work,lwork,seedC,seedD,m,k,mb,PQRPARAM,rev,mpicomm)
    use ELPA1
    use tum_utils
    use mpi

    implicit none
 
    ! parameter setup

    ! input variables (local)
    integer lda,lwork
    double precision a(lda,*), work(*)

    ! input variables (global)
    integer m,k,baseidx,mb,rev,mpicomm
    integer PQRPARAM(*)
    double precision seedC(k,*),seedD(k,*)
 
    ! output variables (global)

    ! derived input variables from TUM_PQRPARAM

    ! local scalars
    integer mpierr,mpirank,mpiprocs,mpirank_top
    integer icol,irow,lidx,remsize
    integer remaining_rank

    integer C_size,D_size,sendoffset,recvoffset,sendrecv_size
    integer localoffset,localsize,baseoffset
 
    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)

    C_size = k*k
    D_size = k*k
    sendoffset = 1
    sendrecv_size = C_size+D_size
    recvoffset = sendoffset + sendrecv_size

    if (lwork .eq. -1) then
        work(1) = DBLE(2*sendrecv_size)
        return
    end if
  
    ! clear buffer
    work(sendoffset:sendoffset+sendrecv_size-1)=0.0d0

    ! collect C part
    do icol=1,k
        !irow = 1
        !lidx = baseidx

        remaining_rank = k
        do while (remaining_rank .gt. 0)
            irow = k - remaining_rank + 1
            !if (rev .eq. 1) then
                lidx = baseidx - remaining_rank + 1
            !else
            !    lidx = baseidx + irow - 1
            !end if

            ! determine chunk where the current top element is located
            mpirank_top = MOD((lidx-1)/mb,mpiprocs) 

            ! limit max number of remaining elements of this chunk to the block
            ! distribution parameter
            remsize = min(remaining_rank,mb)

            ! determine the number of needed elements in this chunk 
            call local_size_offset_1d(lidx+remsize-1,mb, &
                                      lidx,lidx,0, &
                                      mpirank_top,mpiprocs, &
                                      localsize,baseoffset,localoffset)

            !print *,'local rank',localsize,localoffset

            if (mpirank .eq. mpirank_top) then
                ! copy elements to buffer
                work(sendoffset+(icol-1)*k+irow-1:sendoffset+(icol-1)*k+irow-1+localsize-1) &
                            = a(localoffset:localoffset+remsize-1,icol)
            end if

            ! jump to next chunk
            remaining_rank = remaining_rank - localsize
        end do
    end do

    ! collect D part
    !if (rev .eq. 1) then
        call local_size_offset_1d(m,mb,baseidx-k,baseidx-k,rev, &
                                  mpirank,mpiprocs, &
                                  localsize,baseoffset,localoffset)
    !else
    !    call local_size_offset_1d(m,mb,baseidx+k,baseidx+k,rev, &
    !                              mpirank,mpiprocs, &
    !                              localsize,baseoffset,localoffset)
    !end if
 
    !print *,'localsize',localsize,localoffset
    if (localsize > 0) then
        call dsyrk("Upper", "Trans", k, localsize, &
                   1.0d0, a(localoffset,1), lda, &
                   0.0d0, work(sendoffset+C_size), k)
    else
        work(sendoffset+C_size:sendoffset+C_size+k*k-1) = 0.0d0
    end if
 
    ! TODO: store symmetric part more efficiently

    ! allreduce operation on results
    !print *,'sendrecv_size', sendrecv_size
    call mpi_allreduce(work(sendoffset),work(recvoffset),sendrecv_size, &
                       mpi_real8,mpi_sum,mpicomm,mpierr)

    ! unpack result from buffer into seedC and seedD
    !print *,'seedC',k
    seedC(1:k,1:k) = 0.0d0
    do icol=1,k
        seedC(1:k,icol) = work(recvoffset+(icol-1)*k:recvoffset+icol*k)
        !print *,seedC(1:k,icol)
    end do
 
    seedD(1:k,1:k) = 0.0d0
    !print *,'seedD',k
    do icol=1,k
        seedD(1:k,icol) = work(recvoffset+C_size+(icol-1)*k:recvoffset+C_size+icol*k)
        !print *,seedD(1:k,icol)
    end do
end subroutine

subroutine tum_pdlarfgk_1dcomm_check(seedC,seedD,k,PQRPARAM,work,lwork,possiblerank,rev)
    use tum_utils

    implicit none

    ! parameter setup

    ! input variables (local)

    ! input variables (global)
    integer k,lwork,rev
    integer PQRPARAM(*)
    double precision seedC(k,*),seedD(k,*),work(k,*)
 
    ! output variables (global)
    integer possiblerank

    ! derived input variables from TUM_PQRPARAM
    integer eps

    ! local scalars
    integer icol,isqr,iprod
    double precision epsd,sum_sqr,sum_products,diff,temp,ortho,ortho_sum
    double precision dreverse_matrix_work(1)

    if (lwork .eq. -1) then
        !if (rev .eq. 1) &
            call reverse_matrix_local(1,k,k,work,k,dreverse_matrix_work,-1)
        work(1,1) = DBLE(k*k) + dreverse_matrix_work(1)
        return
    end if

    eps = PQRPARAM(3)

    if (eps .eq. 0) then 
        possiblerank = k
        return
    end if

    epsd = DBLE(eps)


    ! copy seedD to work
    work(:,1:k) = seedD(:,1:k)

    ! add inner products of seedC to work
    call dsyrk("Upper", "Trans", k, k, &
               1.0d0, seedC(1,1), k, &
               1.0d0, work, k)

    !if (rev .eq. 1) then
        ! TODO: optimize this part!
        call reverse_matrix_local(0,k,k,work(1,1),k,work(1,k+1),lwork-2*k)
        call reverse_matrix_local(1,k,k,work(1,1),k,work(1,k+1),lwork-2*k)
        
        ! transpose matrix
        do icol=1,k
            do isqr=icol+1,k
                work(icol,isqr) = work(isqr,icol)
            end do
        end do
    !end if
 
    !print *,'work buffer'
    !do icol=1,k
    !    print *,work(icol,1:k)
    !end do

    ! work contains now the full inner product of the global (sub-)matrix
    do icol=1,k
        sum_sqr = 0.0d0
        do isqr=1,icol-1
            sum_products = 0.0d0
            do iprod=1,isqr-1
                sum_products = sum_products + work(iprod,isqr)*work(iprod,icol)
            end do

            temp = (work(isqr,icol) - sum_products)/work(isqr,isqr)
            work(isqr,icol) = temp
            sum_sqr = sum_sqr + temp*temp
        end do

        ! calculate diagonal value
        diff = work(icol,icol) - sum_sqr
        if (diff .lt. 0.0d0) then
            ! we definitely have a problem now
            possiblerank = icol-1 ! only decompose to previous column (including)
            return
        end if
        work(icol,icol) = sqrt(diff)

        ! calculate orthogonality
        ortho = 0.0d0
        do isqr=1,icol-1
            ortho_sum = 0.0d0
            do iprod=isqr,icol-1
                temp = work(isqr,iprod)*work(isqr,iprod)
                temp = temp / (work(iprod,iprod)*work(iprod,iprod))
                ortho_sum = ortho_sum + temp
            end do
            ortho = ortho + ortho_sum * (work(isqr,icol)*work(isqr,icol))
        end do
 
        ortho = ortho / diff;

        ! if current estimate is not accurate enough, the following check holds
        if (ortho .gt. epsd) then
            possiblerank = icol-1 ! only decompose to previous column (including)
            return
        end if
    end do

    ! if we get to this point, the accuracy condition holds for the whole block
    possiblerank = k
end subroutine

!sidx: seed idx
!k: max rank used during seed phase
!rank: actual rank (k >= rank)
subroutine tum_pdlarfgk_1dcomm_vector(x,incx,baseidx,tau,seedC,seedD,k,sidx,n,nb,rev,mpicomm)
    use ELPA1
    use tum_utils
    use mpi
 
    implicit none

    ! input variables (local)
    integer lda,lwork,incx
    double precision x(*),tau

    ! input variables (global)
    integer n,nb,baseidx,rev,mpicomm,k,sidx
    double precision seedC(k,*),seedD(k,*)
 
    ! output variables (global)

    ! external functions
    double precision dlapy2,dnrm2
    external dlapy2,dscal,dnrm2

    ! local scalars
    integer mpirank,mpirank_top,mpiprocs,mpierr
    double precision alpha,dot,beta,xnorm
    integer local_size,baseoffset,local_offset,top,topidx
    integer lidx

    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)

    !if (rev .eq. 1) then
        lidx = baseidx-sidx+1
        call local_size_offset_1d(n,nb,baseidx,lidx-1,rev,mpirank,mpiprocs, &
                                  local_size,baseoffset,local_offset)
    !else
    !    lidx = baseidx+sidx-1
    !    call local_size_offset_1d(n,nb,baseidx,lidx+1,rev,mpirank,mpiprocs, &
    !                              local_size,baseoffset,local_offset)
    !end if
 
    local_offset = local_offset * incx

    ! Processor id for global index of top element
    mpirank_top = MOD((lidx-1)/nb,mpiprocs)
    if (mpirank .eq. mpirank_top) then
        topidx = local_index((lidx),mpirank_top,mpiprocs,nb,0)
        top = 1+(topidx-1)*incx
    end if

    !if (rev .eq. 1) then
        alpha = seedC(k-sidx+1,k-sidx+1)
        dot = seedD(k-sidx+1,k-sidx+1)
        ! assemble actual norm from both seed parts
        xnorm = dlapy2(sqrt(dot), dnrm2(k-sidx,seedC(1,k-sidx+1),1))
    !else
    !    alpha = seedC(sidx,sidx)
    !    dot = seedD(sidx,sidx)
    !    ! assemble actual norm from both seed parts
    !    xnorm = dlapy2(sqrt(dot), dnrm2(k-sidx,seedC(sidx+1,sidx),1))
    !end if
    
    !print *,'k hg:', sidx, xnorm, alpha

    if (xnorm .eq. 0.0d0) then
        ! H = I

        ! as indicator that there are no more elements
        ! TODO: is there a better check? or prevent this case in general?
        !if (rev .eq. 1) then
        !    seedC(k-sidx+1,k-sidx+1) = 0.0d0
        !else
        !    seedC(sidx,sidx) = 0.0d0
        !end if
        tau = 0.0d0
    else
        ! General case

        beta = sign(dlapy2(alpha, xnorm), alpha)
        !tau = (beta+alpha) / beta
        ! store a preliminary version of beta in tau
        tau = beta
        
        ! update global part
        call dscal(local_size, 1.0d0/(beta+alpha), &
                   x(local_offset), incx)

        ! do not update local part here due to
        ! dependency of c vector during update process

        ! TODO: reimplement norm rescale method of 
        ! original PDLARFG using mpi?

        if (mpirank .eq. mpirank_top) then
            x(top) = -beta
        end if
    end if
  
    !print *,'k hg:', sidx, alpha, beta

    !print *,'orig C',sidx,local_offset
    !print *,x(local_offset-1:local_offset+(k-sidx)-1)
end subroutine

!k: original max rank used during seed function
!rank: possible rank as from check function
! TODO: if rank is less than k, reduce buffersize in such a way
! that only the required entries for the next pdlarfg steps are
! computed
subroutine tum_pdlarfgk_1dcomm_update(a,lda,baseidx,work,lwork,seedC,seedD,k,rank,sidx,tau,n,nb,rev,mpicomm)
    use ELPA1
    use tum_utils
    use mpi

    implicit none

    ! parameter setup
    INTEGER     gmode_,rank_,eps_,upmode1_
    PARAMETER   (gmode_ = 1,rank_ = 2,eps_=3, upmode1_=4)

    ! input variables (local)
    integer lda,lwork
    double precision a(lda,*),work(*)

    ! input variables (global)
    integer k,rank,sidx,n,baseidx,nb,rev,mpicomm
    double precision beta
 
    ! output variables (global)
    double precision seedC(k,*),seedD(k,*),tau(*)

    ! derived input variables from TUM_PQRPARAM

    ! local scalars
    double precision alpha
    integer coffset,zoffset,yoffset,voffset,buffersize
    integer mpirank,mpierr,mpiprocs,mpirank_top
    integer localsize,baseoffset,localoffset,topidx
    integer lidx,irow

    if (lwork .eq. -1) then
        ! buffer for c,z,y,v
        work(1) = 4*k
        return
    end if

    ! nothing to update anymore
    if (sidx .gt. rank) return

    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)

    ! TODO: verify the cancelation criterions
    !if (rev .eq. 1) then
        lidx = baseidx-sidx
        if (lidx .lt. 1) return
    !else
    !    lidx = baseidx+sidx
    !    if (lidx .gt. n) return
    !end if

    call local_size_offset_1d(n,nb,baseidx,lidx,rev,mpirank,mpiprocs, &
                              localsize,baseoffset,localoffset)

    coffset = 1
    zoffset = coffset + k
    yoffset = zoffset + k
    voffset = yoffset + k
    buffersize = k - sidx

    !print *,'rank k update:',baseidx,sidx

    ! finalize tau values
    !if (rev .eq. 1) then
        alpha = seedC(k-sidx+1,k-sidx+1)
        beta = tau(k-sidx+1)

        tau(k-sidx+1) = (beta+alpha) / beta
    !else
    !    alpha = seedC(sidx,sidx)
    !    beta = tau(sidx)
    !
    !    tau(sidx) = (beta+alpha) / beta
    !end if

    !print *,'k update', beta, alpha, localsize

    ! ---------------------------------------
        ! calculate c vector (extra vector or encode in seedC/seedD?
        work(coffset:coffset+buffersize-1) = seedD(1:buffersize,k-sidx+1)
        call dgemv("Trans", buffersize+1, buffersize, &
               1.0d0,seedC(1,1),k,seedC(1,k-sidx+1),1, &
               1.0d0,work(coffset),1)

        ! calculate z using tau,seedD,seedC and c vector
        work(zoffset:zoffset+buffersize-1) = seedC(k-sidx+1,1:buffersize)
        call daxpy(buffersize, 1.0d0/beta, work(coffset), 1, work(zoffset), 1)
 
        ! update A1(local copy) and generate part of householder vectors for use
        call daxpy(buffersize, -1.0d0, work(zoffset),1,seedC(k-sidx+1,1),k)
        call dscal(buffersize, 1.0d0/(alpha+beta), seedC(1,k-sidx+1),1)
        call dger(buffersize, buffersize, -1.0d0, seedC(1,k-sidx+1),1, work(zoffset), 1, seedC(1,1), k)
        !print *,'k hl: upper c',seedC(k-sidx+1,1:buffersize+1)
 
           ! update A global (householder vector already generated by pdlarfgk)
        ! TODO: check the top process id
        mpirank_top = MOD(lidx/nb,mpiprocs)
        if (mpirank .eq. mpirank_top) then
            ! handle first row seperately
            topidx = local_index(lidx+1,mpirank_top,mpiprocs,nb,0)
            call daxpy(buffersize,-1.0d0,work(zoffset),1,a(topidx,1),lda)
            !print *,'k hl: upper a',a(topidx,1:buffersize)
        end if

        call dger(localsize, buffersize,-1.0d0, &
              a(localoffset,k-sidx+1),1,work(zoffset),1, &
              a(localoffset,1),lda)
        
        !print *,'k hl: dger', localsize, buffersize
        !print *,'k hl: dger 1',a(localoffset:localoffset+localsize-1,k-sidx+1)
        !print *,'k hl: dger 2',a(localoffset:localoffset+localsize-1,k-sidx)
        !print *,'k hl: dger 1',a(localsize,1:buffersize)
        !print *,'k hl: dger 2',a(localoffset,1:buffersize)
 
    !print *,'k hl: coffset', work(coffset:coffset+buffersize-1)
    !print *,'k hl: zoffset', work(zoffset:zoffset+buffersize-1)
 
        ! update D (symmetric) => two buffer vectors of size rank
        ! generate y vector
        work(yoffset:yoffset+buffersize-1) = 0.d0
        call daxpy(buffersize,1.0d0/(alpha+beta),work(zoffset),1,work(yoffset),1)

        ! generate v vector
        work(voffset:voffset+buffersize-1) = seedD(1:buffersize,k-sidx+1)
        call daxpy(buffersize, -0.5d0*seedD(k-sidx+1,k-sidx+1), work(yoffset), 1, work(voffset),1)

        ! symmetric update of D using y and v
        call dsyr2("Upper", buffersize,-1.0d0, &
                   work(yoffset),1,work(voffset),1, &
                   seedD(1,1), k)
  
    ! prepare T matrix inner products
    ! D_k(1:k,k+1:n) = D_(k-1)(1:k,k+1:n) - D_(k-1)(1:k,k) * y'
    ! store coefficient 1.0d0/(alpha+beta) in C diagonal elements
        call dger(k-sidx,sidx,-1.0d0,work(yoffset),1,seedD(k-sidx+1,k-sidx+1),k,seedD(1,k-sidx+1),k)
        seedC(k,k-sidx+1) = 1.0d0/(alpha+beta)

    !print *,'k hl: yoffset',work(yoffset:yoffset+buffersize-1)
    !print *,'k hl: voffset',work(voffset:voffset+buffersize-1)

    !print *,'seedD'
    !print *,seedD(1,1:3)
    !print *,seedD(2,1:3)
    !print *,seedD(3,1:3)

    !print *,'seedC'
    !print *,seedC(1,1:3)
    !print *,seedC(2,1:3)
    !print *,seedC(3,1:3)

end subroutine

subroutine tum_pdlarfgk_1dcomm_generateT(seedC,seedD,k,actualk,tau,t,ldt,rev)
    implicit none

    integer k,actualk,ldt,rev
    double precision seedC(k,*),seedD(k,*),tau(*),t(ldt,*)

    integer irow,icol
    double precision column_coefficient,beta,alpha

        !print *,'reversed on the fly T generation NYI'

        do icol=1,actualk-1
            ! calculate inner product of householder vector parts in seedC
            ! (actually calculating more than necessary, if actualk < k)
            ! => a lot of junk from row 1 to row k-actualk
            call dtrmv('Upper','Trans','Unit',k-icol,seedC(1,1),k,seedC(1,k-icol+1),1)
        
            ! add scaled D parts to current column of C (will become later T rows)
            column_coefficient = seedC(k,k-icol+1)
            do irow=k-actualk+1,k-1
                !seedC(irow,k-icol+1) = ( seedC(irow,k-icol+1)*(-tau(irow)) ) +  ( seedD(irow,k-icol+1) * column_coefficient * seedC(k,irow)*(-tau(irow)) )
                seedC(irow,k-icol+1) = ( seedC(irow,k-icol+1) ) +  ( seedD(irow,k-icol+1) * column_coefficient * seedC(k,irow) )
            end do
        end do
 
        call tum_dlarft_kernel(actualk,tau(k-actualk+1),seedC(k-actualk+1,k-actualk+2),k,t(k-actualk+1,k-actualk+1),ldt,rev)

end subroutine
! IMPORTANT: first column offset is identical to original matrix, this should be
! enough to efficiently perform an allgather operation for the requested columns
subroutine tum_replicate_submatrix_2dcomm(m,n,mb,nb,rowidx,colidx,a,lda,work,lwork,mpicomm_rows,mpicomm_cols,rev,b,ldb)
    use mpi
    use tum_utils

    implicit none

    ! input variables (global)
    integer m,n,mb,nb,rowidx,colidx,lda,lwork,mpicomm_rows,mpicomm_cols,rev
  
    ! input variables (local)
    double precision a(lda,*), work(ldb,*)

    ! output variables (global)
    integer ldb
    double precision b(ldb,*)

    ! local scalars
    integer mpierr,mpirank_cols_block
    integer mpirank_cols,mpiprocs_cols
    integer mpirank_rows,mpiprocs_rows

    integer localsize_rows,baseoffset_rows,offset_rows
    integer localsize_cols,baseoffset_cols,offset_cols

    integer start_column,blocksize,current_column,idx,blocksize_temp
    integer icol,startcolumn,sendsize

    integer, allocatable :: recvsizes(:),recvdisplacements(:)
 
    call mpi_comm_rank(mpicomm_rows,mpirank_rows,mpierr)
    call mpi_comm_size(mpicomm_rows,mpiprocs_rows,mpierr)
  
    call mpi_comm_rank(mpicomm_cols,mpirank_cols,mpierr)
    call mpi_comm_size(mpicomm_cols,mpiprocs_cols,mpierr)

    call local_size_offset_1d(m,mb,rowidx,rowidx,rev,mpirank_rows,mpiprocs_rows, &
                              localsize_rows,baseoffset_rows,offset_rows)
    ldb = localsize_rows

    if (lwork .eq. -1) then
        work(1,1) = DBLE(nb * localsize_rows) ! approximate upper boundary for sendsize
        return
    end if

    if (localsize_rows .eq. 0) return ! no data available for communication in current process row

    allocate(recvsizes(mpiprocs_cols))
    allocate(recvdisplacements(mpiprocs_cols))
 
    !if (rev .eq. 1) then
        startcolumn = colidx-n+1
    !else
    !    startcolumn = colidx
    !end if

    sendsize = 0
    recvsizes = 0
    recvdisplacements = 0

    ! iterate over all (part)nb blocks
    current_column = 1
    do while (current_column .le. n)
        idx = startcolumn+current_column-1

        ! determine rank of process column and size of next block
        mpirank_cols_block = MOD((idx-1)/nb,mpiprocs_cols)
        blocksize_temp = min(nb,n-current_column+1) ! due to blockcyclic distribution there is a maximum of nb columns per block
        call local_size_offset_1d(idx+blocksize_temp-1,nb,startcolumn,idx,0, &
                                  mpirank_cols_block,mpiprocs_cols, &
                                  blocksize,baseoffset_cols,offset_cols)
 
        if (mpirank_cols_block .eq. mpirank_cols) then
            ! this block belongs to me => pack data into sendbuffer
            do icol=1,blocksize
                work(1:localsize_rows,baseoffset_cols+icol-1) = a(offset_rows:offset_rows+localsize_rows-1,offset_cols+icol-1)
            end do

            ! adjust sendsize counter
            sendsize = sendsize + blocksize*ldb
        end if

        recvsizes(mpirank_cols_block+1) = blocksize*ldb
        recvdisplacements(mpirank_cols_block+1) = (current_column-1)*ldb

        current_column = current_column + blocksize

        !print *,'blocksize',mpirank_cols,mpirank_cols_block,offset_cols,baseoffset_cols
    end do

    !print *,'work matrix'
    !do icol=1,localsize_rows
    !    print *,work(icol,1:n)
    !end do

    !do icol=1,n
    !    b(offset_rows:offset_rows+localsize_rows-1,icol) = 0.0d0
    !end do

    !print *,'allgatherv step', sendsize, recvsizes(1), recvdisplacements(1),recvsizes(2),recvdisplacements(2)

    call MPI_Allgatherv(work(1,1), sendsize, mpi_real8, &
                        b(offset_rows,1), recvsizes, recvdisplacements, mpi_real8,  &
                        mpicomm_cols, mpierr)
  
    !print *,'replicated matrix'
    !do icol=1,localsize_rows
    !    print *,b(offset_rows+icol-1,1:n)
    !end do

    deallocate(recvsizes,recvdisplacements)

end subroutine
!direction=0: pack into work buffer
!direction=1: unpack from work buffer
subroutine tum_pdgeqrf_pack_unpack(v,ldv,work,lwork,m,n,mb,baseidx,rowidx,rev,direction,mpicomm)
    use ELPA1
    use tum_utils
    use mpi
  
    implicit none

    ! input variables (local)
    integer ldv,lwork
    double precision v(ldv,*), work(*)

    ! input variables (global)
    integer m,n,mb,baseidx,rowidx,rev,direction,mpicomm
 
    ! output variables (global)
 
    ! local scalars
    integer mpierr,mpirank,mpiprocs
    integer buffersize,idx,icol,irow
    integer local_size,baseoffset,offset,vcol

    ! external functions
 
    call mpi_comm_rank(mpicomm,mpirank,mpierr)
    call mpi_comm_size(mpicomm,mpiprocs,mpierr)
  
    call local_size_offset_1d(m,mb,baseidx,rowidx,rev,mpirank,mpiprocs, &
                                  local_size,baseoffset,offset)

    !print *,'pack/unpack',local_size,baseoffset,offset

    ! rough approximate for buffer size
    if (lwork .eq. -1) then
        buffersize = local_size * n ! vector elements
        work(1) = DBLE(buffersize)
        return
    end if

    if (direction .eq. 0) then
        ! copy v part to buffer (including zeros)
        do icol=1,n
            work(1+local_size*(icol-1):local_size*icol) = v(baseoffset:baseoffset+local_size-1,icol)
        end do
    else
        ! copy v part from buffer (including zeros)
        do icol=1,n
            v(baseoffset:baseoffset+local_size-1,icol) = work(1+local_size*(icol-1):local_size*icol)
        end do
    end if

    return

    ! obsolete part

    !buffersize = 0
    !do icol=1,n
    !    if (rev .eq. 1) then
    !        idx = rowidx-icol+1
    !        vcol=n-icol+1
    !    else
    !        idx = rowidx+icol-1
    !        vcol=icol
    !    end if
        
    !    call local_size_offset_1d(m,mb,baseidx,idx,rev,mpirank,mpiprocs, &
    !                              local_size,baseoffset,offset)
 
    !    !if (lwork .ne. -1) then
    !        if (direction .eq. 0) then
    !            ! pack v column into work buffer
    !            do irow=1,local_size
    !                work(buffersize+irow) = v(irow+baseoffset-1,vcol)
    !            end do
    !            buffersize = buffersize + local_size
    !     
    !            ! pack T matrix into work buffer
    !            work(buffersize+1:buffersize+icol) = t(1:icol,icol)  
    !            buffersize = buffersize + icol
    !
    !            ! obsolete part
    !            ! pack tau value into work buffer
    !            !work(buffersize+1) = tau(vcol) 
    !            !buffersize = buffersize + 1
    !        else
    !            ! unpack v column from work buffer
    !            v(1:ldv,vcol) = 0.0d0
    !            do irow=1,local_size
    !               v(irow+baseoffset-1,vcol) = work(buffersize+irow)
    !            end do
    !            buffersize = buffersize + local_size
    !
    !            ! unpack T matrix from work buffer
    !            t(1:icol,icol) = work(buffersize+1:buffersize+icol)
    !            buffersize = buffersize + icol
    !
    !            ! restore tau values for compatibility reasons
    !            tau(vcol) = t(icol,icol)
    !
    !            ! obsolete part
    !            ! unpack tau value from work buffer
    !            !tau(vcol) = work(buffersize+1)
    !            !buffersize = buffersize + 1
    !        end if
    !    !else
    !    !    ! space for vectors and t matrix
    !    !    buffersize = buffersize + local_size + icol
    !    !
    !    !    ! obsolete part
    !    !    ! space for vectors and tau values
    !    !    ! buffersize = buffersize + local_size + 1
    !    !end if
    !end do
 
    !if (lwork .eq. -1) then
    !    work(1) = DBLE(buffersize)
    !end if
end subroutine
!direction=0: pack into work buffer
!direction=1: unpack from work buffer
subroutine tum_pdgeqrf_pack_unpack_tmatrix(tau,t,ldt,work,lwork,n,direction)
    use ELPA1
    use tum_utils
    use mpi
  
    implicit none

    ! input variables (local)
    integer ldt,lwork
    double precision work(*), t(ldt,*),tau(*)

    ! input variables (global)
    integer n,direction
 
    ! output variables (global)
 
    ! local scalars
    integer icol

    ! external functions
 
    if (lwork .eq. -1) then
        work(1) = DBLE(n*n)
        return
    end if

    if (direction .eq. 0) then
        ! append t matrix to buffer (including zeros)
        do icol=1,n
            work(1+(icol-1)*n:icol*n) = t(1:n,icol)
        end do
    else
        ! append t matrix from buffer (including zeros)
        do icol=1,n
            t(1:n,icol) = work(1+(icol-1)*n:icol*n)
            tau(icol) = t(icol,icol)
        end do
    end if

end subroutine
!merge b into a, b has same offset as a
subroutine tum_merge_submatrix_2dcomm(m,n,mb,nb,rowidx,colidx,b,ldb,a,lda,mpicomm_rows,mpicomm_cols,rev)
    use mpi
    use tum_utils

    implicit none

    ! input variables (global)
    integer m,n,mb,nb,rowidx,colidx,lda,ldb,mpicomm_rows,mpicomm_cols,rev
  
    ! input variables (local)
    double precision a(lda,*), b(ldb,*)

    ! output variables (global)

    ! local scalars
    integer mpierr,mpirank_cols_block
    integer mpirank_cols,mpiprocs_cols
    integer mpirank_rows,mpiprocs_rows

    integer localsize_rows,baseoffset_rows,offset_rows
    integer localsize_cols,baseoffset_cols,offset_cols

    integer blocksize,current_column,idx,blocksize_temp
    integer icol,startcolumn

    call mpi_comm_rank(mpicomm_rows,mpirank_rows,mpierr)
    call mpi_comm_size(mpicomm_rows,mpiprocs_rows,mpierr)
  
    call mpi_comm_rank(mpicomm_cols,mpirank_cols,mpierr)
    call mpi_comm_size(mpicomm_cols,mpiprocs_cols,mpierr)

    call local_size_offset_1d(m,mb,rowidx,rowidx,rev,mpirank_rows,mpiprocs_rows, &
                              localsize_rows,baseoffset_rows,offset_rows)
    !if (rev .eq. 1) then
        startcolumn = colidx-n+1
    !else
    !    startcolumn = colidx
    !end if

    ! iterate over all (part)nb blocks
    current_column = 1
    do while (current_column .le. n)
        idx = startcolumn+current_column-1

        ! determine rank of process column and size of next block
        mpirank_cols_block = MOD((idx-1)/nb,mpiprocs_cols)
        blocksize_temp = min(nb,n-current_column+1) ! due to blockcyclic distribution there is a maximum of nb columns per block
        call local_size_offset_1d(idx+blocksize_temp-1,nb,startcolumn,idx,0, &
                                  mpirank_cols_block,mpiprocs_cols, &
                                  blocksize,baseoffset_cols,offset_cols)
 
        if (mpirank_cols_block .eq. mpirank_cols) then
            ! this block belongs to me => pack data into sendbuffer
            do icol=1,blocksize
                a(offset_rows:offset_rows+localsize_rows-1,offset_cols+icol-1) = b(offset_rows:offset_rows+localsize_rows-1,current_column+icol-1)
            end do
        end if

        current_column = current_column + blocksize

        !print *,'blocksize',blocksize,offset_cols,baseoffset_cols,lda,ldb
    end do

end subroutine
! m is size of subvector
subroutine tum_merge_subvector_1dcomm(m,mb,baseidx,b,a,mpicomm,rev)
    use mpi
    use tum_utils

    implicit none

    ! input variables (global)
    integer m,mb,baseidx,mpicomm,rev
  
    ! input variables (local)
    double precision a(*), b(*)

    ! output variables (global)

    ! local scalars
    integer mpierr,mpirank_block,mpirank,mpiprocs

    integer localsize,baseoffset,offset
    integer icol,startentry,current_entry,idx
    integer blocksize,blocksize_temp

    call mpi_comm_rank(mpicomm,mpirank,mpierr)
    call mpi_comm_size(mpicomm,mpiprocs,mpierr)

    !if (rev .eq. 1) then
        startentry = baseidx-m+1
    !else
    !    startentry = baseidx
    !end if

    ! iterate over all (part)nb blocks
    current_entry = 1
    do while (current_entry .le. m)
        idx = startentry+current_entry-1

        ! determine rank of process column and size of next block
        mpirank_block = MOD((idx-1)/mb,mpiprocs)
        blocksize_temp = min(mb,m-current_entry+1) ! due to blockcyclic distribution there is a maximum of nb columns per block
        call local_size_offset_1d(idx+blocksize_temp-1,mb,startentry,idx,0, &
                                  mpirank_block,mpiprocs, &
                                  blocksize,baseoffset,offset)
 
        if (mpirank_block .eq. mpirank) then
            ! this block belongs to me => pack data into sendbuffer
            a(offset:offset+blocksize-1) = b(current_entry:current_entry+blocksize-1)
        end if

        current_entry = current_entry + blocksize
    end do

end subroutine
! TODO: encode following functionality
!   - Direction? BOTTOM UP or TOP DOWN ("Up", "Down")
!        => influences all related kernels (including DLARFT / DLARFB)
!   - rank-k parameter (k=1,2,...,b)
!        => influences possible update strategies
!        => parameterize the function itself? (FUNCPTR, FUNCARG)
!   - Norm mode? Allreduce, Allgather, AlltoAll, "AllHouse", (ALLNULL = benchmarking local kernels)
!   - subblocking 
!         (maximum block size bounded by data distribution along rows)
!   - blocking method (householder vectors only or compact WY?)
!   - update strategy of trailing parts (incremental, complete) 
!        - difference for subblocks and normal blocks? (UPDATE and UPDATESUB)
!        o "Incremental"
!        o "Full"
!   - final T generation (recursive: subblock wise, block wise, end) (TMERGE)
!        ' (implicitly given by / influences update strategies?)
!        => alternative: during update: iterate over sub t parts
!           => advantage: smaller (cache aware T parts)
!           => disadvantage: more memory write backs 
!                (number of T parts * matrix elements)
!   - partial/sub T generation (TGEN)
!        o add vectors right after creation (Vector)
!        o add set of vectors (Set)
!   - bcast strategy of householder vectors to other process columns 
!        (influences T matrix generation and trailing update 
!         in other process columns)
!        o no broadcast (NONE = benchmarking?, 
!            or not needed due to 1D process grid)
!        o after every housegen (VECTOR)
!        o after every subblk   (SUBBLOCK)
!        o after full local column block decomposition (BLOCK)
!  LOOP Housegen -> BCAST -> GENT/EXTENDT -> LOOP HouseLeft

!subroutine tum_pqrparam_init(PQRPARAM, DIRECTION, RANK, NORMMODE, &
!                             SUBBLK, UPDATE, TGEN, BCAST)

! gmode: control communication pattern of dlarfg
! maxrank: control max number of householder vectors per communication
! eps: error threshold (integer)
! update*: control update pattern in pdgeqr2_1dcomm ('incremental','full','merge')
!               merging = full update with tmatrix merging
! tmerge*: 0: do not merge, 1: incremental merge, >1: recursive merge
!               only matters if update* == full
subroutine tum_pqrparam_init(pqrparam,size2d,update2d,tmerge2d,size1d,update1d,tmerge1d,maxrank,update,eps,hgmode)

    implicit none

    ! input
    CHARACTER   update2d,update1d,update,hgmode
    INTEGER     size2d,size1d,maxrank,eps,tmerge2d,tmerge1d

    ! output
    INTEGER     PQRPARAM(*)

    PQRPARAM(1) = size2d
    PQRPARAM(2) = ichar(update2d)
    PQRPARAM(3) = tmerge2d
    ! TODO: broadcast T yes/no

    PQRPARAM(4) = size1d
    PQRPARAM(5) = ichar(update1d)
    PQRPARAM(6) = tmerge1d

    PQRPARAM(7) = maxrank
    PQRPARAM(8) = ichar(update)
    PQRPARAM(9) = eps
    PQRPARAM(10) = ichar(hgmode)

end subroutine


subroutine tum_pqrparam_print(pqrparam)
    integer PQRPARAM(*)

    print  '(i,a,i,i,a,i,i,a,i,a)', &
        pqrparam(1),char(pqrparam(2)),pqrparam(3), &
        pqrparam(4),char(pqrparam(5)),pqrparam(6), &
        pqrparam(7),char(pqrparam(8)),pqrparam(9),char(pqrparam(10))

end subroutine
subroutine tum_pdlarfg_copy_1dcomm(x,incx,v,incv,n,baseidx,idx,nb,rev,mpicomm)
    use ELPA1
    use tum_utils
    use mpi

    implicit none
 
    ! input variables (local)
    integer incx,incv
    double precision x(*), v(*)

    ! input variables (global)
    integer baseidx,idx,rev,nb,n
    integer mpicomm
 
    ! output variables (global)
 
    ! local scalars
    integer mpierr,mpiprocs
    integer mpirank,mpirank_top
    integer irow,x_offset,x_base
    integer v_offset,local_size


    call MPI_Comm_rank(mpicomm, mpirank, mpierr)
    call MPI_Comm_size(mpicomm, mpiprocs, mpierr)
 
    call local_size_offset_1d(n,nb,baseidx,idx,rev,mpirank,mpiprocs, &
                              local_size,v_offset,x_offset)
    v_offset = v_offset * incv

    !print *,'copy:',mpirank,baseidx,v_offset,x_offset,local_size

    ! copy elements
    do irow=1,local_size
        v((irow-1)*incv+v_offset) = x((irow-1)*incx+x_offset)
    end do

    ! replace top element to build an unitary vector
    mpirank_top = MOD((idx-1)/nb,mpiprocs)
    if (mpirank .eq. mpirank_top) then
        !if (rev .eq. 1) then
            v(local_size*incv) = 1.0d0
        !else
        !    v(v_offset) = 1.0d0
        !end if
    end if

end subroutine
