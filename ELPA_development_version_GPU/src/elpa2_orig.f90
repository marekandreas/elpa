!! ELPA2 -- 2-stage solver for ELPA
! 
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".

module ELPA2

! Version 1.1.2, 2011-02-21

  USE ELPA1
!  USE blockedQR
use cudafor

  implicit none

  PRIVATE ! By default, all routines contained are private

  ! The following routines are public:

  public :: solve_evp_real_2stage
  public :: solve_evp_complex_2stage

  public :: bandred_real
  public :: tridiag_band_real
  public :: trans_ev_tridi_to_band_real
  public :: trans_ev_band_to_full_real

  public :: bandred_complex
  public :: tridiag_band_complex
  public :: trans_ev_tridi_to_band_complex
  public :: trans_ev_band_to_full_complex
  
  public :: band_band_real
  public :: divide_band
  
!-------------------------------------------------------------------------------  

!  integer, public :: which_qr_decomposition = 0     ! defines, which QR-decomposition algorithm will be used
!                                                    ! 0 for unblocked
!                                                    ! 1 for rank-2

!-------------------------------------------------------------------------------

  ! The following array contains the Householder vectors of the
  ! transformation band -> tridiagonal.
  ! It is allocated and set in tridiag_band_real and used in
  ! trans_ev_tridi_to_band_real.
  ! It must be deallocated by the user after trans_ev_tridi_to_band_real!

  real*8, allocatable :: hh_trans_real(:,:)
  complex*16, allocatable :: hh_trans_complex(:,:)

!-------------------------------------------------------------------------------

  include 'mpif.h'


!******
contains

subroutine solve_evp_real_2stage(na, nev, a, lda, ev, q, ldq, nblk, mpi_comm_rows, mpi_comm_cols, mpi_comm_all)

!-------------------------------------------------------------------------------
!  solve_evp_real_2stage: Solves the real eigenvalue problem with a 2 stage approach
!
!  Parameters
!
!  na          Order of matrix a
!
!  nev         Number of eigenvalues needed
!
!  a(lda,*)    Distributed matrix for which eigenvalues are to be computed.
!              Distribution is like in Scalapack.
!              The full matrix must be set (not only one half like in scalapack).
!              Destroyed on exit (upper and lower half).
!
!  lda         Leading dimension of a
!
!  ev(na)      On output: eigenvalues of a, every processor gets the complete set
!
!  q(ldq,*)    On output: Eigenvectors of a
!              Distribution is like in Scalapack.
!              Must be always dimensioned to the full size (corresponding to (na,na))
!              even if only a part of the eigenvalues is needed.
!
!  ldq         Leading dimension of q
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!  mpi_comm_all
!              MPI-Communicator for the total processor set
!
!-------------------------------------------------------------------------------

   implicit none

   integer, intent(in) :: na, nev, lda, ldq, nblk, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
   real*8, intent(inout) :: a(lda,*), ev(na), q(ldq,*)

   integer my_pe, n_pes, my_prow, my_pcol, np_rows, np_cols, mpierr
   integer nbw, num_blocks
   real*8, allocatable :: tmat(:,:,:), e(:)
   real*8 ttt0, ttt1, ttts
   integer myrank
   integer :: i
 
   call mpi_comm_rank(mpi_comm_all,my_pe,mpierr)
   call mpi_comm_size(mpi_comm_all,n_pes,mpierr)

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   ! Choose bandwidth, must be a multiple of nblk, set to a value >= 32

   nbw = (31/nblk+1)*nblk

   num_blocks = (na-1)/nbw + 1

   allocate(tmat(nbw,nbw,num_blocks))

   ! Reduction full -> band

!   print *, "lda = ", lda
!   print *, "na = ", na
!   call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!   do i = 0, n_pes-1 
!     if(i .eq.  my_pe) print*, 'a prior to bandred = ', a(1:lda, 1:na)
!     call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!   end do

   ttt0 = MPI_Wtime()
   ttts = ttt0
!   print *, 'about to bandreduce', my_prow, my_pcol
   call bandred_real(na, a, lda, nblk, nbw, mpi_comm_rows, mpi_comm_cols, tmat)
!    call bandred_real0(na, a, lda, nblk, nbw, mpi_comm_rows, mpi_comm_cols, tmat)
!   print *, "completed bandred real"
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      print *,'Time bandred_real               :',ttt1-ttt0

   ! Reduction band -> tridiagonal

!  call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!  do i = 0, n_pes-1 
!    if(0 .eq.  my_pe) print*, 'a prior to tridiag_band = ', a(1:lda, 1:na)
!    call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!  end do

!   print *, 'allocate', my_prow, my_pcol
   allocate(e(na))

!   print *, 'Starting with triag to band reduction', my_prow, my_pcol
   ttt0 = MPI_Wtime()
   call tridiag_band_real(na, nbw, nblk, a, lda, ev, e, mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      print *,'Time tridiag_band_real          :',ttt1-ttt0

!   call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!   do i = 0, n_pes-1 
!     if(i .eq.  my_pe) print*, 'a = ', a(1:lda, 1:na)
!     call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!   end do
   

!   call mpi_comm_rank(mpi_comm_all, myrank, mpierr)
   call mpi_bcast(ev,na,MPI_DOUBLE,0,mpi_comm_all,mpierr)
   call mpi_bcast(e,na,MPI_DOUBLE,0,mpi_comm_all,mpierr)

!   call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!!   if( my_pe .eq. 0) then
!     print *, "ev = ", ev
!     print *, "e = ", e
!!   endif   
!   call MPI_Barrier(MPI_COMM_WORLD, mpierr)

   ttt1 = MPI_Wtime()
   time_evp_fwd = ttt1-ttts

   ! Solve tridiagonal system

   ttt0 = MPI_Wtime()
!   print *, 'about to solve tridiag'
   call solve_tridi(na, nev, ev, e, q, ldq, nblk, mpi_comm_rows, mpi_comm_cols)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      print *,'Time solve_tridi                :',ttt1-ttt0
   time_evp_solve = ttt1-ttt0
   ttts = ttt1

!   call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!   do i = 0, n_pes-1 
!     if(i .eq.  my_pe) print*, 'q  1 = ', q(1:na, 1:na)
!     call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!   end do

!   call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!   do i = 0, n_pes-1 
!     if(i .eq.  my_pe) print*, 'q  1 sum = ', sum(q(1:ldq, 1:20))
!     call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!   end do

   deallocate(e)

   ! Backtransform stage 1

   ttt0 = MPI_Wtime()
   call trans_ev_tridi_to_band_real(na, nev, nblk, nbw, q, ldq, mpi_comm_rows, mpi_comm_cols)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      print *,'Time trans_ev_tridi_to_band_real:',ttt1-ttt0

!   call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!   do i = 0, n_pes-1 
!     if(i .eq.  my_pe) print*, 'q 2 = ', q(1:na, 1:na)
!     call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!   end do

!   call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!   do i = 0, n_pes-1 
!     if(i .eq.  my_pe) print*, 'q 2 sum = ', sum(q(1:ldq, 1:20))
!     call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!   end do


   ! We can now deallocate the stored householder vectors
   deallocate(hh_trans_real)

   ! Backtransform stage 2

   ttt0 = MPI_Wtime()
   call trans_ev_band_to_full_real(na, nev, nblk, nbw, a, lda, tmat, q, ldq, mpi_comm_rows, mpi_comm_cols)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      print *,'Time trans_ev_band_to_full_real :',ttt1-ttt0
   time_evp_back = ttt1-ttts

!   call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!   do i = 0, n_pes-1 
!     if(i .eq.  my_pe) print*, 'q 3 = ', q(1:na, 1:na)
!     call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!   end do

!   call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!   do i = 0, n_pes-1 
!     if(i .eq.  my_pe) print*, 'q 3 sum = ', sum( q(1:lda, 1:20) )
!     call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!   end do

   deallocate(tmat)

1  format(a,f10.3)

end subroutine solve_evp_real_2stage

!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------

subroutine solve_evp_complex_2stage(na, nev, a, lda, ev, q, ldq, nblk, mpi_comm_rows, mpi_comm_cols, mpi_comm_all)

!-------------------------------------------------------------------------------
!  solve_evp_complex_2stage: Solves the complex eigenvalue problem with a 2 stage approach
!
!  Parameters
!
!  na          Order of matrix a
!
!  nev         Number of eigenvalues needed
!
!  a(lda,*)    Distributed matrix for which eigenvalues are to be computed.
!              Distribution is like in Scalapack.
!              The full matrix must be set (not only one half like in scalapack).
!              Destroyed on exit (upper and lower half).
!
!  lda         Leading dimension of a
!
!  ev(na)      On output: eigenvalues of a, every processor gets the complete set
!
!  q(ldq,*)    On output: Eigenvectors of a
!              Distribution is like in Scalapack.
!              Must be always dimensioned to the full size (corresponding to (na,na))
!              even if only a part of the eigenvalues is needed.
!
!  ldq         Leading dimension of q
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!  mpi_comm_all
!              MPI-Communicator for the total processor set
!
!-------------------------------------------------------------------------------

   implicit none

   integer, intent(in) :: na, nev, lda, ldq, nblk, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
   complex*16, intent(inout) :: a(lda,*), q(ldq,*)
   real*8, intent(inout) :: ev(na)

   integer my_prow, my_pcol, np_rows, np_cols, mpierr
   integer l_cols, l_rows, l_cols_nev, nbw, num_blocks
   complex*16, allocatable :: tmat(:,:,:)
   real*8, allocatable :: q_real(:,:), e(:)
   real*8 ttt0, ttt1, ttts

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   ! Choose bandwidth, must be a multiple of nblk, set to a value >= 32

   nbw = (31/nblk+1)*nblk

   num_blocks = (na-1)/nbw + 1

   allocate(tmat(nbw,nbw,num_blocks))

   ! Reduction full -> band

   ttt0 = MPI_Wtime()
   ttts = ttt0
   call bandred_complex(na, a, lda, nblk, nbw, mpi_comm_rows, mpi_comm_cols, tmat)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      print *,'Time bandred_complex               :',ttt1-ttt0

   ! Reduction band -> tridiagonal

   allocate(e(na))

   ttt0 = MPI_Wtime()
   call tridiag_band_complex(na, nbw, nblk, a, lda, ev, e, mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      print *,'Time tridiag_band_complex          :',ttt1-ttt0

   call mpi_bcast(ev,na,MPI_REAL8,0,mpi_comm_all,mpierr)
   call mpi_bcast(e,na,MPI_REAL8,0,mpi_comm_all,mpierr)

   ttt1 = MPI_Wtime()
   time_evp_fwd = ttt1-ttts

   l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a and q
   l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local columns of q
   l_cols_nev = local_index(nev, my_pcol, np_cols, nblk, -1) ! Local columns corresponding to nev

   allocate(q_real(l_rows,l_cols))

   ! Solve tridiagonal system

   ttt0 = MPI_Wtime()
   call solve_tridi(na, nev, ev, e, q_real, ubound(q_real,1), nblk, mpi_comm_rows, mpi_comm_cols)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times)  &
      print *,'Time solve_tridi                   :',ttt1-ttt0
   time_evp_solve = ttt1-ttt0
   ttts = ttt1

   q(1:l_rows,1:l_cols_nev) = q_real(1:l_rows,1:l_cols_nev)

   deallocate(e, q_real)

   ! Backtransform stage 1

   ttt0 = MPI_Wtime()
   call trans_ev_tridi_to_band_complex(na, nev, nblk, nbw, q, ldq, mpi_comm_rows, mpi_comm_cols)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      print *,'Time trans_ev_tridi_to_band_complex:',ttt1-ttt0

   ! We can now deallocate the stored householder vectors
   deallocate(hh_trans_complex)

   ! Backtransform stage 2

   ttt0 = MPI_Wtime()
   call trans_ev_band_to_full_complex(na, nev, nblk, nbw, a, lda, tmat, q, ldq, mpi_comm_rows, mpi_comm_cols)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      print *,'Time trans_ev_band_to_full_complex :',ttt1-ttt0
   time_evp_back = ttt1-ttts

   deallocate(tmat)

1  format(a,f10.3)

end subroutine solve_evp_complex_2stage

!-------------------------------------------------------------------------------

subroutine bandred_real0(na, a, lda, nblk, nbw, mpi_comm_rows, mpi_comm_cols, tmat)

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
   integer work_size
   
   real*8 eps

   real*8 vnorm2, xf, aux1(nbw), aux2(nbw), vrl, tau, vav(nbw,nbw)

   real*8, allocatable:: tmp(:,:), vr(:), vmr(:,:), umc(:,:)
   real*8, allocatable:: work(:) ! needed for blocked QR
real*8 times(100), ttt0, ttts


   integer pcol, prow
   pcol(i) = MOD((i-1)/nblk,np_cols) !Processor col for global col number
   prow(i) = MOD((i-1)/nblk,np_rows) !Processor row for global row number

ttts = mpi_wtime()
times(:) = 0

ttt0 = mpi_wtime()
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
 
!   if (which_qr_decomposition == 1) then
!          l_rows = local_index(na, my_prow, np_rows, nblk, -1)
!          work_size = max(4*np_rows,2*nbw)
!          work_size = max(l_rows+1,work_size)
!          work_size = max(16, work_size)
!          work_size = max(2*(l_rows+1),work_size)
!          work_size = max(2+4*(nbw+1),work_size)
!          allocate(work(work_size))
!          work = 0
!          eps = 1.0d0
!   endif
times(1) = times(1) + mpi_wtime()-ttt0
!   print *, "About to step", my_pcol, my_pcol

   do istep = (na-1)/nbw, 1, -1

ttt0 = mpi_wtime()
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

      umc(:,:) = 0.

      print*, " a  (after init)= ", a(1:2, 5000)
      print*, "vmr (after init)= ", vmr(1,1)
      print*, "umc (after init)= ", umc(1,1)
      print*, "vav (after init)= ", vav(1,1)

      ! Reduce current block to lower triangular form
!      if (which_qr_decomposition == 1) then
!          call qr_rank2_real(a, lda, vmr, max(l_rows,1), tmat, nbw, istep, n_cols, nblk, mpi_comm_rows, mpi_comm_cols, work, eps)
!      else

!      print *, "About to reduce cols ", my_pcol, my_pcol
      do lc = n_cols, 1, -1

         ncol = istep*nbw + lc ! absolute column number of householder vector
         nrow = ncol - nbw ! Absolute number of pivot row

         lr  = local_index(nrow, my_prow, np_rows, nblk, -1) ! current row length
         lch = local_index(ncol, my_pcol, np_cols, nblk, -1) ! HV local column number
         print *, "a_lch= ", a(1,lch)

         tau = 0

         if(nrow == 1) exit ! Nothing to do

         cur_pcol = pcol(ncol) ! Processor column owning current block

      !   print *, "lr = ", lr
      !   print *, "lch= ", lch

         if(my_pcol==cur_pcol) then

            ! Get vector to be transformed; distribute last element and norm of
            ! remaining elements to all procs in current column

            vr(1:lr) = a(1:lr,lch) ! vector to be transformed
            print *, "vr = ", vr(1)

            if(my_prow==prow(nrow)) then
               aux1(1) = dot_product(vr(1:lr-1),vr(1:lr-1))
               aux1(2) = vr(lr)
            else
               aux1(1) = dot_product(vr(1:lr),vr(1:lr))
               aux1(2) = 0.
            endif

!            print *, "About to allreduce ", my_pcol, my_pcol
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

!            print *, "completed householder  ", my_pcol, my_pcol
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
times(2) = times(2) + mpi_wtime()-ttt0
ttt0 = mpi_wtime()
      print*, " a  (after loop)= ", a(1:2, 5000)
      print*, "vmr (after loop)= ", vmr(1,1)
      print*, "umc (after loop)= ", umc(1,1)
      print*, "vav (after loop)= ", vav(1,1)

!      endif

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
times(3) = times(3) + mpi_wtime()-ttt0
ttt0 = mpi_wtime()

      print*, " a  (after transp)= ", a(1:2, 5000)
      print*, "vmr (after transp)= ", vmr(1,1)
      print*, "umc (after transp)= ", umc(1,1)
      print*, "vav (after transp)= ", vav(1,1)

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
times(4) = times(4) + mpi_wtime()-ttt0
ttt0 = mpi_wtime()

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
times(5) = times(5) + mpi_wtime()-ttt0
ttt0 = mpi_wtime()

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

times(6) = times(6) + mpi_wtime()-ttt0
ttt0 = mpi_wtime()
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
times(7) = times(7) + mpi_wtime()-ttt0

      print*, " a = ", a(1:2, 5000)
      print*, "vmr = ", vmr(1,1)
      print*, "umc = ", umc(1,1)
      print*, "vav = ", vav(1,1)

      deallocate(vmr, umc, vr)

   enddo
 
!   if (which_qr_decomposition == 1) then
!          deallocate(work)
!   endif

ttts = mpi_wtime()-ttts
!print '("Times: ",10f10.3)',times(1:7),ttts-sum(times(1:7)),ttts

end subroutine

!---------------------------------------------------------------------------------------------------
!---------------------------------------------------------------------------------------------------
!---------------------------------------------------------------------------------------------------

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

use cublas


   implicit none
include "mpif.h"

   integer na, lda, nblk, nbw, mpi_comm_rows, mpi_comm_cols
   !real*8 a(lda,na), tmat(nbw,nbw,*)
   real*8 a(lda,*), tmat(nbw,nbw,*)

   integer my_prow, my_pcol, np_rows, np_cols, mpierr
   integer l_cols, l_rows
   integer i, j, lcs, lce, lre, lc, lr, cur_pcol, n_cols, nrow, nnn
   integer istep, ncol, lch, lcx, nlc
   integer tile_size, l_rows_tile, l_cols_tile
   integer work_size
   
   real*8 eps

   real*8 vnorm2, xf, aux1(nbw), aux2(nbw), vrl, tau, vav(nbw,nbw), aux(nbw)

   real*8, allocatable:: tmp(:,:), vr(:), vmr(:,:), umc(:,:)
   real*8, allocatable:: work(:) ! needed for blocked QR

   real*8, allocatable, device:: a_dev(:,:), vr_dev(:), vmr_dev(:,:), umc_dev(:,:), tmat_dev(:,:), vav_dev(:,:), aux_dev(:)
   !real*8, allocatable:: a_dev(:,:), vr_dev(:), vmr_dev(:,:), umc_dev(:,:)
   integer cuda_stream(16)
   type(cublasHandle) :: ch

real*8 times(100), ttt0, ttts
   integer istat

   integer, external :: numroc
   integer:: na_rows, na_cols


   integer pcol, prow
   pcol(i) = MOD((i-1)/nblk,np_cols) !Processor col for global col number
   prow(i) = MOD((i-1)/nblk,np_rows) !Processor row for global row number
!   print*, "blah blah"

   !print *, 'Bandred real'
   !call  MPI_Barrier(MPI_COMM_WORLD, mpierr); 
   !print *, 'survived barrier'
   !call  MPI_Barrier(MPI_COMM_WORLD, mpierr); 
ttts = mpi_wtime()
times(:) = 0

   !print *, "hit barrier afterwards"
   !call  MPI_Barrier(MPI_COMM_WORLD, mpierr); 

   !print *, "about to create streams"
!   do i = 1, 16
!     j = cudaStreamCreate(cuda_stream(i))
!     if(j /= cudaSuccess) STOP 'cudaStreamCreate failed'
!   enddo
!
!   print *, "streams created"
!   call  MPI_Barrier(MPI_COMM_WORLD, mpierr); 

   ch = cublasGetHandle()
  ! print *, "Cublas handle obtained"
   call  MPI_Barrier(MPI_COMM_WORLD, mpierr); 
 

!   call  MPI_Barrier(MPI_COMM_WORLD, mpierr); 

ttt0 = mpi_wtime()
   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)


   na_rows = numroc(na, nblk, my_prow, 0, np_rows)
   na_cols = numroc(na, nblk, my_pcol, 0, np_cols)


   ! Semibandwith nbw must be a multiple of blocksize nblk

!   print*, my_prow, my_pcol, "mod =", mod(nbw, nblk)
!   call flush(6)
!   call MPI_Barrier(MPI_COMM_WORLD, mpierr)
!   print*, my_prow, my_pcol,"nbw = ", nbw, "   blk=", nblk 
!   call flush(6)
!   call MPI_Barrier(MPI_COMM_WORLD, mpierr)
 

   if(mod(nbw,nblk)/=0) then
      if(my_prow==0 .and. my_pcol==0) then
         print *,'ERROR: nbw=',nbw,', nblk=',nblk
         print *,'ELPA2 works only for nbw==n*nblk'
!         call mpi_abort(mpi_comm_world,0,mpierr)
      endif
   endif

!   print *, "survived check:", my_prow, np_rows, my_pcol, np_cols
!   call flush(6)
!   call MPI_Barrier(MPI_COMM_WORLD, mpierr)


   ! Matrix is split into tiles; work is done only for tiles on the diagonal or above

   tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size
   tile_size = ((128*max(np_rows,np_cols)-1)/tile_size+1)*tile_size ! make local tiles at least 128 wide
!   tile_size = ((512*max(np_rows,np_cols)-1)/tile_size+1)*tile_size ! make local tiles at least 128 wide

   l_rows_tile = tile_size/np_rows ! local rows of a tile
   l_cols_tile = tile_size/np_cols ! local cols of a tile
 
!   if (which_qr_decomposition == 1) then
!          l_rows = local_index(na, my_prow, np_rows, nblk, -1)
!          work_size = max(4*np_rows,2*nbw)
!          work_size = max(l_rows+1,work_size)
!          work_size = max(16, work_size)
!          work_size = max(2*(l_rows+1),work_size)
!          work_size = max(2+4*(nbw+1),work_size)
!          allocate(work(work_size))
!          work = 0
!          eps = 1.0d0
!   endif
!    print *, "about to allocate", my_prow, my_pcol, lda, na
    !allocate(a_dev(lda,na), stat=istat)
    allocate(a_dev(lda,na_cols), stat=istat)
   !print *, "a_dev allocated", istat, my_prow, my_pcol, lda, na, lda*na
!    print *, "allocate 1 ", my_prow, my_pcol
  !  a_dev(:,:) = a(:,:)
   !print*, "result =",  cudamemcpy(a_dev, a, lda * na)
    !a_dev(1:10,1) = a(1:10,1)
!   print *, "na = ", na
!   print *, "na_cols = ", na_cols
!   print *, "lda = ", lda
    !a_dev(:,1:1024) = a(:,1:1024)

!*** we don't need to transfer a yet, as it is still being modified on the host
!   a_dev(1:lda, 1:na_cols) = a(1:lda, 1:na_cols)

!    print *, "allocate 2 ", my_prow, my_pcol
    allocate(tmat_dev(nbw,nbw), vav_dev(nbw,nbw), aux_dev(nbw))
!    print *, "allocate 3 ", my_prow, my_pcol
    allocate(vr_dev(na+1))
!    print *, "allocate 4 ", my_prow, my_pcol

      a_dev(1:lda, 1:na_cols) = a(1:lda, 1:na_cols)

times(1) = times(1) + mpi_wtime()-ttt0

   do istep = (na-1)/nbw, 1, -1
      
      n_cols = MIN(na,(istep+1)*nbw) - istep*nbw ! Number of columns in current step

      ! Number of local columns/rows of remaining matrix
      l_cols = local_index(istep*nbw, my_pcol, np_cols, nblk, -1)
      l_rows = local_index(istep*nbw, my_prow, np_rows, nblk, -1)

      a(1:lda, 1:na_cols) = a_dev(1:lda, 1:na_cols)

!      print *, "l_cols = ", l_cols
!      print *, "l_rows = ", l_rows

      ! Allocate vmr and umc to their exact sizes so that they can be used in bcasts and reduces

      allocate(vmr(max(l_rows,1),2*n_cols))
      allocate(umc(max(l_cols,1),2*n_cols))

      allocate(vr(l_rows+1))

      vmr(1:l_rows,1:n_cols) = 0.
      vr(:) = 0
      tmat(:,:,istep) = 0

ttt0 = mpi_wtime()
      ! Reduce current block to lower triangular form
!      if (which_qr_decomposition == 1) then
!          call qr_rank2_real(a, lda, vmr, max(l_rows,1), tmat, nbw, istep, n_cols, nblk, mpi_comm_rows, mpi_comm_cols, work, eps)
!      else
      umc = 0.
      print*, " a  (after init)= ", a(1:2, 5000)
      print*, "vmr (after init)= ", vmr(1,1)
      print*, "umc (after init)= ", umc(1,1)
      print*, "vav (after init)= ", vav(1,1)

      do lc = n_cols, 1, -1

         ncol = istep*nbw + lc ! absolute column number of householder vector
         nrow = ncol - nbw ! Absolute number of pivot row

         lr  = local_index(nrow, my_prow, np_rows, nblk, -1) ! current row length
         lch = local_index(ncol, my_pcol, np_cols, nblk, -1) ! HV local column number
         print *, "lr = ", lr
         print *, "lch = ", lch
         print *, "a_lch= ", a(1,lch)

         tau = 0

         if(nrow == 1) exit ! Nothing to do

         cur_pcol = pcol(ncol) ! Processor column owning current block

!         print *, "lr = ", lr
!         print *, "lch= ", lch
!         print *, "na_cols= ", na_cols

         if(my_pcol==cur_pcol) then
            ! Get vector to be transformed; distribute last element and norm of
            ! remaining elements to all procs in current column

            vr(1:lr) = a(1:lr,lch) ! vector to be transformed
            print *, "vr = ", vr(1)
            
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
!            print *, "vr = ", vr(1)
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

      print*, " a  (after loop)= ", a(1:2, 5000)
      print*, "vmr (after loop)= ", vmr(1,1)
      print*, "umc (after loop)= ", umc(1,1)
      print*, "vav (after loop)= ", vav(1,1)

!      endif

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

!  Up until now nothing has happened on the GPU. As we have messed around with a, we
!  now need to update a_dev again. 

      print*, " a  (after transp)= ", a(1:2, 5000)
      print*, "vmr (after transp)= ", vmr(1,1)
      print*, "umc (after transp)= ", umc(1,1)
      print*, "vav (after transp)= ", vav(1,1)


times(3) = times(3) + mpi_wtime()-ttt0
ttt0 = mpi_wtime()

      a_dev(1:lda, 1:na_cols) = a(1:lda, 1:na_cols)
      tmat_dev(:,:) = tmat(:,:,istep)

      ! Calculate umc = A**T * vmr
      ! Note that the distributed A has to be transposed
      ! Opposed to direct tridiagonalization there is no need to use the cache locality
      ! of the tiles, so we can use strips of the matrix

      umc(1:l_cols,1:n_cols) = 0.d0
      vmr(1:l_rows,n_cols+1:2*n_cols) = 0.0d0

      if(allocated(vmr_dev)) deallocate(vmr_dev)
      if(allocated(umc_dev)) deallocate(umc_dev)
      allocate(vmr_dev(max(l_rows,1),2*n_cols))
      allocate(umc_dev(max(l_cols,1),2*n_cols))

      umc_dev = umc
      vmr_dev = vmr

      if(l_cols>0 .and. l_rows>0) then

         do i=0,(istep*nbw-1)/tile_size

            lcs = i*l_cols_tile+1
            lce = min(l_cols,(i+1)*l_cols_tile)
            if(lce<lcs) cycle

            lre = min(l_rows,(i+1)*l_rows_tile)
            call cublasDGEMM('T','N',lce-lcs+1,n_cols,lre,1.d0,a_dev(1,lcs),ubound(a_dev,1), &
                       vmr_dev,ubound(vmr_dev,1),1.d0,umc_dev(lcs,1),ubound(umc_dev,1))
            mpierr = cublasgeterror()
           if( mpierr .ne. 0) print *, "error dtrmm = ", mpierr

            if(i==0) cycle
            lre = min(l_rows,i*l_rows_tile)
            call cublasDGEMM('N','N',lre,n_cols,lce-lcs+1,1.d0,a_dev(1,lcs),ubound(a_dev,1), &
                       umc_dev(lcs,n_cols+1),ubound(umc_dev,1),1.d0,vmr_dev(1,n_cols+1),ubound(vmr_dev,1))
            mpierr = cublasgeterror()
            if( mpierr .ne. 0) print *, "error dtrmm = ", mpierr
         enddo

      endif

      umc = umc_dev
      vmr = vmr_dev
      !i = cudaThreadSynchronize()
times(4) = times(4) + mpi_wtime()-ttt0
ttt0 = mpi_wtime()
      !      print *, "A**T * vmr  completed"
            call MPI_Barrier(MPI_COMM_WORLD, mpierr)

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
times(5) = times(5) + mpi_wtime()-ttt0
ttt0 = mpi_wtime()

      ! U = U * Tmat**T
      umc_dev = umc
      vmr_dev = vmr
      vav_dev(:,:) = vav
 
!      print *, "Transfer U*Tmat ** T completed"
      !call dtrmm('Right','Upper','Trans','Nonunit',l_cols,n_cols,1.d0,tmat(1,1,istep),ubound(tmat,1),umc,ubound(umc,1))
      call cublasdtrmm('Right','Upper','Trans','Nonunit',l_cols,n_cols,1.d0,tmat_dev,ubound(tmat_dev,1),umc_dev,ubound(umc_dev,1))

      mpierr = cublasgeterror()
      if( mpierr .ne. 0) print *, "error dtrmm = ", mpierr

! out: umc_dev

      ! VAV = Tmat * V**T * A * V * Tmat**T = (U*Tmat**T)**T * V * Tmat**T

      !call dgemm('T','N',n_cols,n_cols,l_cols,1.d0,umc,ubound(umc,1),umc(1,n_cols+1),ubound(umc,1),0.d0,vav,ubound(vav,1))
      !call dtrmm('Right','Upper','Trans','Nonunit',n_cols,n_cols,1.d0,tmat(1,1,istep),ubound(tmat,1),vav,ubound(vav,1))
      call cublasdgemm('T','N',n_cols,n_cols,l_cols,1.d0,umc_dev,ubound(umc_dev,1),umc_dev(1,n_cols+1),ubound(umc_dev,1),&
                       0.d0,vav_dev,ubound(vav_dev,1))
! out: vav_dev

      mpierr = cublasgeterror()
      if( mpierr .ne. 0) print *, "error dgemm = ", mpierr

      call cublasdtrmm('Right','Upper','Trans','Nonunit',n_cols,n_cols,1.d0,tmat_dev,ubound(tmat_dev,1),vav_dev,ubound(vav_dev,1))

      mpierr = cublasgeterror()
      if( mpierr .ne. 0) print *, "error dgemm = ", mpierr

      vav = vav_dev
!      a(1:lda, 1:na_cols) = a_dev(1:lda, 1:na_cols)

      call symm_matrix_allreduce(n_cols,vav,ubound(vav,1),mpi_comm_cols)

      a_dev(1:lda, 1:na_cols) = a(1:lda, 1:na_cols)
      vav_dev = vav

      ! U = U - 0.5 * V * VAV
      call cublasdgemm('N','N',l_cols,n_cols,n_cols,-0.5d0,umc_dev(1,n_cols+1),ubound(umc_dev,1),vav_dev,ubound(vav_dev,1),&
                       1.d0,umc_dev,ubound(umc_dev,1))

      mpierr = cublasgeterror()
      if( mpierr .ne. 0) print *, "error dgemm = ", mpierr
      ! Transpose umc -> umr (stored in vmr, second half)

      umc = umc_dev
      vmr = vmr_dev
      call elpa_transpose_vectors  (umc, ubound(umc,1), mpi_comm_cols, &
                                     vmr(1,n_cols+1), ubound(vmr,1), mpi_comm_rows, &
                                     1, istep*nbw, n_cols, nblk)
      vmr_dev = vmr
      umc_dev = umc

! out: umc_dev = umc
! out: vmr_dev = vmr


      !i = cudaThreadSynchronize()
times(6) = times(6) + mpi_wtime()-ttt0
ttt0 = mpi_wtime()
      ! A = A - V*U**T - U*V**T
!      call cublasdgemm('N','T',l_rows,nbw,2*n_cols,-1.d0, &
!                    vmr_dev(1,1),ubound(vmr_dev,1),umc_dev(l_cols-nbw+1,1),ubound(umc_dev,1), &
!                    1.d0,a_dev(1,l_cols-nbw+1),ubound(a_dev,1))

!      mpierr = cublasgeterror()
!      print *, "error = ", mpierr
      !call cublasdgemm('N','T',l_rows,nbw,2*n_cols,-1.d0, &
      !              vmr_dev,ubound(vmr_dev,1),umc_dev(l_cols-nbw+1,1),ubound(umc_dev,1), &
      !              1.d0,a_dev(1,l_cols-nbw+1),ubound(a_dev,1))
!            print *, "completed first dgemm", my_pcol, my_prow
!            flush(6)
!            call MPI_Barrier(MPI_COMM_WORLD, mpierr)
      ! RJ: Still have to check why the following doesn't work:
      !a(1:MIN(l_rows+nbw,na),l_cols-nbw+1:l_cols) = a_dev(1:MIN(l_rows+nbw,na),l_cols-nbw+1:l_cols)
      !a(:,l_cols-nbw+1:l_cols) = a_dev(:,l_cols-nbw+1:l_cols)
      !mpierr = cudamemcpy(a(:,l_cols-nbw+1:l_cols), c_devloc(a_dev(:,l_cols-nbw+1:l_cols)), l_rows * nbw, cudaMemcpyDeviceToHost) 
      !mpierr = cudamemcpy(a, a_dev, l_cols * nbw, cudaMemcpyDeviceToHost) 
!      mpierr = cudamemcpy(a, a_dev, l_rows * nbw, cudaMemcpyDeviceToHost) 
      !mpierr = cudamemcpy(c_loc(a(1,1)), c_devloc(a_dev(1,1)), 1, cudaMemcpyDeviceToHost) 

      !a(1,1) = a_dev(1,1)
!      print *, "final transfer completed", my_pcol, my_prow
!      call MPI_Barrier(MPI_COMM_WORLD, mpierr)


! in: umc_dev, vmr_dev, a_dev

      do i=(istep*nbw-1)/tile_size,0,-1
         lcs = i*l_cols_tile+1
         lce = min(l_cols-nbw,(i+1)*l_cols_tile)
         lre = min(l_rows,(i+1)*l_rows_tile)
         if(lce<lcs .or. lre<1) cycle
         call cublasdgemm('N','T',lre,lce-lcs+1,2*n_cols,-1.d0, &
                    vmr_dev,ubound(vmr_dev,1),umc_dev(lcs,1),ubound(umc_dev,1), &
                    1.d0,a_dev(1,lcs),ubound(a_dev,1))
      enddo
! out: a_dev

!            print *, "dgemms completed", my_pcol, my_prow
!            call MPI_Barrier(MPI_COMM_WORLD, mpierr)

      vav = vav_dev
      umc = umc_dev
      vmr = vmr_dev
      a(1:lda, 1:na_cols) = a_dev(1:lda, 1:na_cols)
      print*, " a = ", a(1:2, 5000)
      print*, "vmr = ", vmr(1,1) 
      print*, "umc = ", umc(1,1) 
      print*, "vav = ", vav(1,1) 
! a host = a_dev
      deallocate(vmr, umc, vr)
      !i = cudaThreadSynchronize()
times(7) = times(7) + mpi_wtime()-ttt0

   enddo
 
   deallocate(a_dev, vmr_dev, umc_dev, vav_dev, tmat_dev) 

!   if (which_qr_decomposition == 1) then
!          deallocate(work)
!   endif

ttts = mpi_wtime()-ttts
!print '("Times: ",10f10.3)',times(1:7),ttts-sum(times(1:7)),ttts

end subroutine

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

!-------------------------------------------------------------------------------

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

use cublas

   implicit none

   integer na, nqc, lda, ldq, nblk, nbw, mpi_comm_rows, mpi_comm_cols
   real*8 a(lda,*), q(ldq,nqc), tmat(nbw, nbw, *)

   integer my_prow, my_pcol, np_rows, np_cols, mpierr
   integer max_blocks_row, max_blocks_col, max_local_rows, max_local_cols
   integer l_cols, l_rows, l_colh, n_cols
   integer istep, lc, ncol, nrow, nb, ns

   real*8, allocatable:: tmp1(:), tmp2(:), hvb(:)
   real*8, allocatable::  hvm_tmp(:,:)
   real*8, allocatable, device :: hvm_dev(:,:), q_dev(:,:), tmp_dev(:), tmat_dev(:,:)

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

   allocate(hvm_dev(max_local_rows,nbw))
   allocate(tmp_dev(max_local_cols*nbw))
   allocate(tmat_dev(nbw,nbw))
   allocate(q_dev(ldq,nqc))
   allocate(hvm_tmp(max_local_rows,nbw))
   q_dev = q

   hvm_dev = 0   ! Must be set to 0 !!!
   hvb = 0   ! Safety only
   hvm_tmp = 0

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
!         if(my_pcol==pcol(ncol)) print *, "hvb = ", hvb(nb+1:nb+l_rows) 

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

!         hvm_dev(1:l_rows,lc) = hvb(nb+1:nb+l_rows)
!         if(my_prow==prow(nrow)) hvm_dev(l_rows+1,lc) = 1.

         hvm_tmp(1:l_rows,lc) = hvb(nb+1:nb+l_rows)
         if(my_prow==prow(nrow)) hvm_tmp(l_rows+1,lc) = 1.

         nb = nb+l_rows
      enddo
      hvm_dev = hvm_tmp

      l_rows = local_index(MIN(na,(istep+1)*nbw), my_prow, np_rows, nblk, -1)

      ! Q = Q - V * T**T * V**T * Q

      if(l_rows>0) then
!         print *, "Blas call 1:", n_cols, l_cols, l_rows
         call cublasdgemm('T','N',n_cols,l_cols,l_rows,1.d0,hvm_dev,ubound(hvm_dev,1), &
                    q_dev,ldq,0.d0,tmp_dev,n_cols)
      else
         tmp_dev(1:l_cols*n_cols) = 0
      endif
      tmp1 = tmp_dev
! re-introduced this
      call mpi_allreduce(tmp1,tmp2,n_cols*l_cols,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)
      tmp_dev = tmp2
      if(l_rows>0) then
         tmat_dev(:,:) = tmat(:,:,istep)
!         print *, "Blas call 2:", l_rows, l_cols, n_cols
         call cublasdtrmm('L','U','T','N',n_cols,l_cols,1.0d0,tmat_dev,ubound(tmat_dev,1),tmp_dev,n_cols)
         call cublasdgemm('N','N',l_rows,l_cols,n_cols,-1.d0,hvm_dev,ubound(hvm_dev,1), &
                    tmp_dev,n_cols,1.d0,q_dev,ldq)
      endif
     hvm_tmp = hvm_dev
   enddo

   deallocate(tmp1, tmp2, hvb)
   deallocate(hvm_tmp)
   deallocate(hvm_dev,tmp_dev,tmat_dev)
   q = q_dev
   deallocate(q_dev)


end subroutine trans_ev_band_to_full_real

! --------------------------------------------------------------------------------------------------

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
    real*8, allocatable :: hv_t(:,:), tau_t(:)

    integer i, j, n, nc, nr, ns, ne, istep, iblk, nblocks_total, nblocks, nt
    integer my_pe, n_pes, mpierr
    integer my_prow, np_rows, my_pcol, np_cols
    integer ireq_ab, ireq_hv
    integer na_s, nx, num_hh_vecs, num_chunks, local_size, max_blk_size, n_off
    integer max_threads, my_thread, my_block_s, my_block_e, iter
    integer mpi_status(MPI_STATUS_SIZE)
    integer, allocatable :: mpi_statuses(:,:)
    integer, allocatable :: omp_block_limits(:)
    integer, allocatable :: ireq_hhr(:), ireq_hhs(:), global_id(:,:), global_id_tmp(:,:), hh_cnt(:), hh_dst(:)
    integer, allocatable :: limits(:), snd_limits(:,:)
    integer, allocatable :: block_limits(:)
    real*8, allocatable :: ab(:,:), hh_gath(:,:,:), hh_send(:,:,:)
    ! dummies for calling redist_band
    complex*16 :: c_a(1,1), c_ab(1,1)

!$  integer :: omp_get_max_threads


    call mpi_comm_rank(mpi_comm,my_pe,mpierr)
    call mpi_comm_size(mpi_comm,n_pes,mpierr)

    call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
    call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
    call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
    call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

    ! Get global_id mapping 2D procssor coordinates to global id

    allocate(global_id(0:np_rows-1,0:np_cols-1))
    allocate(global_id_tmp(0:np_rows-1,0:np_cols-1))
    global_id(:,:) = 0
    global_id(my_prow, my_pcol) = my_pe

    global_id_tmp(:,:) = global_id(:,:)
    call mpi_allreduce(global_id_tmp, global_id, np_rows*np_cols, mpi_integer, mpi_sum, mpi_comm, mpierr)
    deallocate(global_id_tmp)


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

    ! OpenMP work distribution:

    max_threads = 1
!$ max_threads = omp_get_max_threads()

    ! For OpenMP we need at least 2 blocks for every thread
    max_threads = MIN(max_threads, nblocks/2)
    if(max_threads==0) max_threads = 1

    allocate(omp_block_limits(0:max_threads))

    ! Get the OpenMP block limits
    call divide_band(nblocks, max_threads, omp_block_limits)

    allocate(hv_t(nb,max_threads), tau_t(max_threads))
    hv_t = 0
    tau_t = 0

    ! ---------------------------------------------------------------------------
    ! Start of calculations

    na_s = block_limits(my_pe)*nb + 1

!    print*, "na_s = ", na_s

    if(my_pe>0 .and. na_s<=na) then
      ! send first column to previous PE
      ! Only the PE owning the diagonal does that (sending 1 element of the subdiagonal block also)
      ab_s(1:nb+1) = ab(1:nb+1,na_s-n_off)
      call mpi_isend(ab_s,nb+1,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
    endif

!    print *, "(should be 0): block_limits(my_pe)*nb = ", block_limits(my_pe)*nb
!    if(block_limits(my_pe)*nb .ne. 0) then 
!        print *, "ERROR!!!!! invalid block limits", block_limits(my_pe)*nb
!        print *, "na = ", na
!        print *, "nb = ", nb
!    endif 

! PM HACK: Seems to have  wrong limits
!   do istep=1,na-1-block_limits(my_pe)*nb
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
          call mpi_recv(hv,nb,mpi_real8,my_pe-1,2,mpi_comm,mpi_status,mpierr)
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

      if(max_threads > 1) then

        ! Codepath for OpenMP

        ! Please note that in this case it is absolutely necessary to have at least 2 blocks per thread!
        ! Every thread is one reduction cycle behind its predecessor and thus starts one step later.
        ! This simulates the behaviour of the MPI tasks which also work after each other.
        ! The code would be considerably easier, if the MPI communication would be made within
        ! the parallel region - this is avoided here since this would require 
        ! MPI_Init_thread(MPI_THREAD_MULTIPLE) at the start of the program.

        hv_t(:,1) = hv
        tau_t(1) = tau

        do iter = 1, 2

          ! iter=1 : work on first block
          ! iter=2 : work on remaining blocks
          ! This is done in 2 iterations so that we have a barrier in between:
          ! After the first iteration, it is guaranteed that the last row of the last block
          ! is completed by the next thread.
          ! After the first iteration it is also the place to exchange the last row
          ! with MPI calls

!$omp parallel do private(my_thread, my_block_s, my_block_e, iblk, ns, ne, hv, tau, &
!$omp&                    nc, nr, hs, hd, vnorm2, hf, x, h, i), schedule(static,1), num_threads(max_threads)
          do my_thread = 1, max_threads

            if(iter == 1) then
              my_block_s = omp_block_limits(my_thread-1) + 1
              my_block_e = my_block_s
            else
              my_block_s = omp_block_limits(my_thread-1) + 2
              my_block_e = omp_block_limits(my_thread)
            endif

            do iblk = my_block_s, my_block_e

              ns = na_s + (iblk-1)*nb - n_off - my_thread + 1 ! first column in block
              ne = ns+nb-1                    ! last column in block

              if(istep<my_thread .or. ns+n_off>na) exit

              hv = hv_t(:,my_thread)
              tau = tau_t(my_thread)

              ! Store Householder vector for back transformation

              hh_cnt(iblk) = hh_cnt(iblk) + 1

              hh_gath(1   ,hh_cnt(iblk),iblk) = tau
              hh_gath(2:nb,hh_cnt(iblk),iblk) = hv(2:nb)

              nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
              nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                            ! Note that nr>=0 implies that diagonal block is full (nc==nb)!

              ! Transform diagonal block

              call DSYMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,0.d0,hd,1)

              x = dot_product(hv(1:nc),hd(1:nc))*tau
              hd(1:nc) = hd(1:nc) - 0.5*x*hv(1:nc)

              call DSYR2('L',nc,-1.d0,hd,1,hv,1,ab(1,ns),2*nb-1)

              hv_t(:,my_thread) = 0
              tau_t(my_thread)  = 0

              if(nr<=0) cycle ! No subdiagonal block present any more

              ! Transform subdiagonal block

              call DGEMV('N',nr,nb,tau,ab(nb+1,ns),2*nb-1,hv,1,0.d0,hs,1)

              if(nr>1) then

                ! complete (old) Householder transformation for first column

                ab(nb+1:nb+nr,ns) = ab(nb+1:nb+nr,ns) - hs(1:nr) ! Note: hv(1) == 1

                ! calculate new Householder transformation for first column
                ! (stored in hv_t(:,my_thread) and tau_t(my_thread))

                vnorm2 = sum(ab(nb+2:nb+nr,ns)**2)
                call hh_transform_real(ab(nb+1,ns),vnorm2,hf,tau_t(my_thread))
                hv_t(1   ,my_thread) = 1.
                hv_t(2:nr,my_thread) = ab(nb+2:nb+nr,ns)*hf
                ab(nb+2:,ns) = 0

                ! update subdiagonal block for old and new Householder transformation
                ! This way we can use a nonsymmetric rank 2 update which is (hopefully) faster

                call DGEMV('T',nr,nb-1,tau_t(my_thread),ab(nb,ns+1),2*nb-1,hv_t(1,my_thread),1,0.d0,h(2),1)
                x = dot_product(hs(1:nr),hv_t(1:nr,my_thread))*tau_t(my_thread)
                h(2:nb) = h(2:nb) - x*hv(2:nb)
                ! Unfortunately there is no BLAS routine like DSYR2 for a nonsymmetric rank 2 update ("DGER2")
                do i=2,nb
                  ab(2+nb-i:1+nb+nr-i,i+ns-1) = ab(2+nb-i:1+nb+nr-i,i+ns-1) - hv_t(1:nr,my_thread)*h(i) - hs(1:nr)*hv(i)
                enddo

              else

                ! No new Householder transformation for nr=1, just complete the old one
                ab(nb+1,ns) = ab(nb+1,ns) - hs(1) ! Note: hv(1) == 1
                do i=2,nb
                  ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*hv(i)
                enddo
                ! For safety: there is one remaining dummy transformation (but tau is 0 anyways)
                hv_t(1,my_thread) = 1.

              endif

            enddo

          enddo ! my_thread
!$omp end parallel do

          if (iter==1) then
            ! We are at the end of the first block

            ! Send our first column to previous PE
            if(my_pe>0 .and. na_s <= na) then
              call mpi_wait(ireq_ab,mpi_status,mpierr)
              ab_s(1:nb+1) = ab(1:nb+1,na_s-n_off)
              call mpi_isend(ab_s,nb+1,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
            endif

            ! Request last column from next PE
            ne = na_s + nblocks*nb - (max_threads-1) - 1
            if(istep>=max_threads .and. ne <= na) then
              call mpi_recv(ab(1,ne-n_off),nb+1,mpi_real8,my_pe+1,1,mpi_comm,mpi_status,mpierr)
            endif

          else
            ! We are at the end of all blocks

            ! Send last HH vector and TAU to next PE if it has been calculated above
            ne = na_s + nblocks*nb - (max_threads-1) - 1
            if(istep>=max_threads .and. ne < na) then
              call mpi_wait(ireq_hv,mpi_status,mpierr)
              hv_s(1) = tau_t(max_threads)
              hv_s(2:) = hv_t(2:,max_threads)
              call mpi_isend(hv_s,nb,mpi_real8,my_pe+1,2,mpi_comm,ireq_hv,mpierr)
            endif

            ! "Send" HH vector and TAU to next OpenMP thread
            do my_thread = max_threads, 2, -1
              hv_t(:,my_thread) = hv_t(:,my_thread-1)
              tau_t(my_thread)  = tau_t(my_thread-1)
            enddo

          endif
        enddo ! iter

      else
!        print*, "****** Entering single threaded code path"

        ! Codepath for 1 thread without OpenMP

        ! The following code is structured in a way to keep waiting times for
        ! other PEs at a minimum, especially if there is only one block.
        ! For this reason, it requests the last column as late as possible
        ! and sends the Householder vector and the first column as early
        ! as possible.

        do iblk=1,nblocks

          ns = na_s + (iblk-1)*nb - n_off ! first column in block
          ne = ns+nb-1                    ! last column in block

          if(ns+n_off>na) exit

          ! Store Householder vector for back transformation

          hh_cnt(iblk) = hh_cnt(iblk) + 1

          hh_gath(1   ,hh_cnt(iblk),iblk) = tau
          hh_gath(2:nb,hh_cnt(iblk),iblk) = hv(2:nb)

!++++ debugging: here we missed this following piece form the original code
!PM HACK
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

!++++ debugging


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
            !call mpi_recv(ab(1,ne),nb+1,mpi_real8,my_pe+1,1,mpi_comm,mpi_status,mpierr)
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
              !call mpi_wait(ireq_hv,mpi_status,mpierr)
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

            !call mpi_wait(ireq_ab,mpi_status,mpierr)
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
              ! Unfortunately there is no BLAS routine like DSYR2 for a nonsymmetric rank 2 update ("DGER2")
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

      endif

!+++++ debugging:
!++ PM HACK: The following has been moved inside the code above for single
!            threaded execution
!      do iblk = 1, nblocks
!
!        if(hh_dst(iblk) >= np_rows) exit
!        if(snd_limits(hh_dst(iblk)+1,iblk) == snd_limits(hh_dst(iblk),iblk)) exit
!
!        if(hh_cnt(iblk) == snd_limits(hh_dst(iblk)+1,iblk)-snd_limits(hh_dst(iblk),iblk)) then
!          ! Wait for last transfer to finish
!          call mpi_wait(ireq_hhs(iblk), mpi_status, mpierr)
!          ! Copy vectors into send buffer
!          hh_send(:,1:hh_cnt(iblk),iblk) = hh_gath(:,1:hh_cnt(iblk),iblk)
!          ! Send to destination
!          call mpi_isend(hh_send(1,1,iblk), nb*hh_cnt(iblk), mpi_real8, &
!                         global_id(hh_dst(iblk),mod(iblk+block_limits(my_pe)-1,np_cols)), &
!                         10+iblk, mpi_comm, ireq_hhs(iblk), mpierr)
!          ! Reset counter and increase destination row
!          hh_cnt(iblk) = 0
!          hh_dst(iblk) = hh_dst(iblk)+1
!        endif
!
!      enddo
!+++++ debugging:

    enddo

    ! Finish the last outstanding requests
    call mpi_wait(ireq_ab,mpi_status,mpierr)
    call mpi_wait(ireq_hv,mpi_status,mpierr)

    allocate(mpi_statuses(MPI_STATUS_SIZE,max(nblocks,num_chunks)))
    call mpi_waitall(nblocks, ireq_hhs, mpi_statuses, mpierr)
    call mpi_waitall(num_chunks, ireq_hhr, mpi_statuses, mpierr)
    deallocate(mpi_statuses)

    call mpi_barrier(mpi_comm,mpierr)

    deallocate(ab)
    deallocate(ireq_hhr, ireq_hhs)
    deallocate(hh_cnt, hh_dst)
    deallocate(hh_gath, hh_send)
    deallocate(limits, snd_limits)
    deallocate(block_limits)
    deallocate(global_id)

end subroutine tridiag_band_real

! --------------------------------------------------------------------------------------------------

subroutine tridiag_band_real_test(na, nb, nblk, a, lda, d, e, mpi_comm_rows, mpi_comm_cols, mpi_comm)

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

use cublas
    implicit none

    integer, intent(in) ::  na, nb, nblk, lda, mpi_comm_rows, mpi_comm_cols, mpi_comm
    real*8, intent(in)  :: a(lda,*)
    real*8, intent(out) :: d(na), e(na) ! set only on PE 0


    real*8 vnorm2, hv(nb), tau, x, h(nb), ab_s(1+nb), hv_s(nb), hv_new(nb), tau_new, hf
    real*8 hd(nb), hs(nb), ab_c(nb)
    real*8, allocatable :: hv_t(:,:), tau_t(:)

    real*8, device :: hd_dev(nb), hs_dev(nb), h_dev(nb), hv_dev(nb), hv_t_dev(nb)

    integer i, j, n, nc, nr, ns, ne, istep, iblk, nblocks_total, nblocks, nt
    integer my_pe, n_pes, mpierr
    integer my_prow, np_rows, my_pcol, np_cols
    integer ireq_ab, ireq_hv
    integer na_s, nx, num_hh_vecs, num_chunks, local_size, max_blk_size, n_off
    integer max_threads, my_thread, my_block_s, my_block_e, iter
    integer mpi_status(MPI_STATUS_SIZE)
    integer, allocatable :: mpi_statuses(:,:)
    integer, allocatable :: omp_block_limits(:)
    integer, allocatable :: ireq_hhr(:), ireq_hhs(:), global_id(:,:), global_id_tmp(:,:), hh_cnt(:), hh_dst(:)
    integer, allocatable :: limits(:), snd_limits(:,:)
    integer, allocatable :: block_limits(:)
    real*8, allocatable :: ab(:,:), hh_gath(:,:,:), hh_send(:,:,:)
    ! dummies for calling redist_band
    complex*16 :: c_a(1,1), c_ab(1,1)

    real*8, allocatable, device :: ab_dev(:,:)

!$  integer :: omp_get_max_threads


    call mpi_comm_rank(mpi_comm,my_pe,mpierr)
    call mpi_comm_size(mpi_comm,n_pes,mpierr)

    call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
    call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
    call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
    call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

    ! Get global_id mapping 2D procssor coordinates to global id

    allocate(global_id(0:np_rows-1,0:np_cols-1))
    allocate(global_id_tmp(0:np_rows-1,0:np_cols-1))
    global_id(:,:) = 0
    global_id(my_prow, my_pcol) = my_pe

    global_id_tmp(:,:) = global_id(:,:)
    call mpi_allreduce(global_id_tmp, global_id, np_rows*np_cols, mpi_integer, mpi_sum, mpi_comm, mpierr)
    deallocate(global_id_tmp)


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

    allocate(ab_dev(2*nb,(nblocks+1)*nb))

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

    ! OpenMP work distribution:

    max_threads = 1
!$ max_threads = omp_get_max_threads()

    ! For OpenMP we need at least 2 blocks for every thread
    max_threads = MIN(max_threads, nblocks/2)
    if(max_threads==0) max_threads = 1

max_threads = nblocks/2

    allocate(omp_block_limits(0:max_threads))

    ! Get the OpenMP block limits
    call divide_band(nblocks, max_threads, omp_block_limits)

    allocate(hv_t(nb,max_threads), tau_t(max_threads))
    hv_t = 0
    tau_t = 0

    ! ---------------------------------------------------------------------------
    ! Start of calculations

    na_s = block_limits(my_pe)*nb + 1

    if(my_pe>0 .and. na_s<=na) then
      ! send first column to previous PE
      ! Only the PE owning the diagonal does that (sending 1 element of the subdiagonal block also)
      ab_s(1:nb+1) = ab(1:nb+1,na_s-n_off)
      call mpi_isend(ab_s,nb+1,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
    endif

    ab_dev = ab

    do istep=1,na-1-block_limits(my_pe)*nb

      if(my_pe==0) then
        n = MIN(na-na_s,nb) ! number of rows to be reduced
        hv(:) = 0
        tau = 0
        ! The last step (istep=na-1) is only needed for sending the last HH vectors.
        ! We don't want the sign of the last element flipped (analogous to the other sweeps)
        ab_c(1:nb) = ab_dev(2:nb+1,na_s-n_off)
        if(istep < na-1) then
          ! Transform first column of remaining matrix
          vnorm2 = sum(ab_c(2:n)**2)
          call hh_transform_real(ab_c(1),vnorm2,hf,tau)
          hv(1) = 1
          hv(2:n) = ab_c(2:n)*hf
        endif
        d(istep) = ab_dev(1,na_s-n_off)
        e(istep) = ab_c(1)
        if(istep == na-1) then
          d(na) = ab_dev(1,na_s+1-n_off)
          e(na) = 0
        endif
      else
        if(na>na_s) then
          ! Receive Householder vector from previous task, from PE owning subdiagonal
          call mpi_recv(hv,nb,mpi_real8,my_pe-1,2,mpi_comm,mpi_status,mpierr)
          tau = hv(1)
          hv(1) = 1.
        endif
      endif

      na_s = na_s+1
      if(na_s-n_off > nb) then
        ab(:,1:nblocks*nb) = ab_dev(:,nb+1:(nblocks+1)*nb)
        ab_dev(:,1:nblocks*nb) = ab(:,1:nblocks*nb)
        ab_dev(:,nblocks*nb+1:(nblocks+1)*nb) = 0
        n_off = n_off + nb
      endif

      if(max_threads > 1) then

        ! Codepath for OpenMP

        ! Please note that in this case it is absolutely necessary to have at least 2 blocks per thread!
        ! Every thread is one reduction cycle behind its predecessor and thus starts one step later.
        ! This simulates the behaviour of the MPI tasks which also work after each other.
        ! The code would be considerably easier, if the MPI communication would be made within
        ! the parallel region - this is avoided here since this would require 
        ! MPI_Init_thread(MPI_THREAD_MULTIPLE) at the start of the program.

        hv_t(:,1) = hv
        tau_t(1) = tau

        do iter = 1, 2

          ! iter=1 : work on first block
          ! iter=2 : work on remaining blocks
          ! This is done in 2 iterations so that we have a barrier in between:
          ! After the first iteration, it is guaranteed that the last row of the last block
          ! is completed by the next thread.
          ! After the first iteration it is also the place to exchange the last row
          ! with MPI calls

!$omp parallel do private(my_thread, my_block_s, my_block_e, iblk, ns, ne, hv, tau, &
!$omp&                    nc, nr, hs, hd, vnorm2, hf, x, h, i), schedule(static,1), num_threads(max_threads)
          do my_thread = 1, max_threads

            if(iter == 1) then
              my_block_s = omp_block_limits(my_thread-1) + 1
              my_block_e = my_block_s
            else
              my_block_s = omp_block_limits(my_thread-1) + 2
              my_block_e = omp_block_limits(my_thread)
            endif

            do iblk = my_block_s, my_block_e

              ns = na_s + (iblk-1)*nb - n_off - my_thread + 1 ! first column in block
              ne = ns+nb-1                    ! last column in block

              if(istep<my_thread .or. ns+n_off>na) exit

              hv = hv_t(:,my_thread)
              tau = tau_t(my_thread)

              ! Store Householder vector for back transformation

              hh_cnt(iblk) = hh_cnt(iblk) + 1

              hh_gath(1   ,hh_cnt(iblk),iblk) = tau
              hh_gath(2:nb,hh_cnt(iblk),iblk) = hv(2:nb)

              nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
              nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                            ! Note that nr>=0 implies that diagonal block is full (nc==nb)!

              ! Transform diagonal block

              hv_dev = hv
              call DSYMV('L',nc,tau,ab_dev(1,ns),2*nb-1,hv_dev,1,0.d0,hd_dev,1)
              hd = hd_dev

              x = dot_product(hv(1:nc),hd(1:nc))*tau
              hd(1:nc) = hd(1:nc) - 0.5*x*hv(1:nc)

              hd_dev = hd
              call DSYR2('L',nc,-1.d0,hd_dev,1,hv_dev,1,ab_dev(1,ns),2*nb-1)

              hv_t(:,my_thread) = 0
              tau_t(my_thread)  = 0

              if(nr<=0) cycle ! No subdiagonal block present any more

              ! Transform subdiagonal block

              call DGEMV('N',nr,nb,tau,ab_dev(nb+1,ns),2*nb-1,hv_dev,1,0.d0,hs_dev,1)
              hs = hs_dev

              if(nr>1) then

                ! complete (old) Householder transformation for first column

                ab_c(1:nr) = ab_dev(nb+1:nb+nr,ns)
                ab_c(1:nr) = ab_c(1:nr) - hs(1:nr) ! Note: hv(1) == 1

                ! calculate new Householder transformation for first column
                ! (stored in hv_t(:,my_thread) and tau_t(my_thread))

                vnorm2 = sum(ab_c(2:nr)**2)
                call hh_transform_real(ab_c(1),vnorm2,hf,tau_t(my_thread))
                hv_t(1   ,my_thread) = 1.
                hv_t(2:nr,my_thread) = ab_c(2:nr)*hf
                ab_dev(nb+1,ns) = ab_c(1)
                ab_dev(nb+2:,ns) = 0 ! Necessary ???

                ! update subdiagonal block for old and new Householder transformation
                ! This way we can use a nonsymmetric rank 2 update which is (hopefully) faster

                hv_t_dev = hv_t(:,my_thread)
                call DGEMV('T',nr,nb-1,tau_t(my_thread),ab_dev(nb,ns+1),2*nb-1,hv_t_dev,1,0.d0,h_dev(2),1)
                h = h_dev
                x = dot_product(hs(1:nr),hv_t(1:nr,my_thread))*tau_t(my_thread)
                h(2:nb) = h(2:nb) - x*hv(2:nb)
                ! Unfortunately there is no BLAS routine like DSYR2 for a nonsymmetric rank 2 update ("DGER2")
                !do i=2,nb
                !  ab(2+nb-i:1+nb+nr-i,i+ns-1) = ab(2+nb-i:1+nb+nr-i,i+ns-1) - hv_t(1:nr,my_thread)*h(i) - hs(1:nr)*hv(i)
                !enddo
                h_dev = h
                call cublasDger(nr,nb-1,-1.d0,hv_t_dev,1,h_dev(2),1,ab_dev(nb,ns+1),2*nb-1)
                call cublasDger(nr,nb-1,-1.d0,hs_dev,1,hv_dev(2),1,ab_dev(nb,ns+1),2*nb-1)

              else

                ! No new Householder transformation for nr=1, just complete the old one
                !ab(nb+1,ns) = ab(nb+1,ns) - hs(1) ! Note: hv(1) == 1
                !do i=2,nb
                !  ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*hv(i)
                !enddo
                call cublasDger(1,nb,-1.d0,hs_dev,1,hv_dev,1,ab_dev(nb+1,ns),2*nb-1)
                ! For safety: there is one remaining dummy transformation (but tau is 0 anyways)
                hv_t(1,my_thread) = 1.

              endif

            enddo

          enddo ! my_thread
!$omp end parallel do

          if (iter==1) then
            ! We are at the end of the first block

            ! Send our first column to previous PE
            if(my_pe>0 .and. na_s <= na) then
              call mpi_wait(ireq_ab,mpi_status,mpierr)
              ab_s(1:nb+1) = ab(1:nb+1,na_s-n_off)
              call mpi_isend(ab_s,nb+1,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
            endif

            ! Request last column from next PE
            ne = na_s + nblocks*nb - (max_threads-1) - 1
            if(istep>=max_threads .and. ne <= na) then
              call mpi_recv(ab(1,ne-n_off),nb+1,mpi_real8,my_pe+1,1,mpi_comm,mpi_status,mpierr)
            endif

          else
            ! We are at the end of all blocks

            ! Send last HH vector and TAU to next PE if it has been calculated above
            ne = na_s + nblocks*nb - (max_threads-1) - 1
            if(istep>=max_threads .and. ne < na) then
              call mpi_wait(ireq_hv,mpi_status,mpierr)
              hv_s(1) = tau_t(max_threads)
              hv_s(2:) = hv_t(2:,max_threads)
              call mpi_isend(hv_s,nb,mpi_real8,my_pe+1,2,mpi_comm,ireq_hv,mpierr)
            endif

            ! "Send" HH vector and TAU to next OpenMP thread
            do my_thread = max_threads, 2, -1
              hv_t(:,my_thread) = hv_t(:,my_thread-1)
              tau_t(my_thread)  = tau_t(my_thread-1)
            enddo

          endif
        enddo ! iter

      else

        STOP 'Codepath for 1 thread reached'
!        ! Codepath for 1 thread without OpenMP
!
!        ! The following code is structured in a way to keep waiting times for
!        ! other PEs at a minimum, especially if there is only one block.
!        ! For this reason, it requests the last column as late as possible
!        ! and sends the Householder vector and the first column as early
!        ! as possible.
!
!        do iblk=1,nblocks
!
!          ns = na_s + (iblk-1)*nb - n_off ! first column in block
!          ne = ns+nb-1                    ! last column in block
!
!          if(ns+n_off>na) exit
!
!          ! Store Householder vector for back transformation
!
!          hh_cnt(iblk) = hh_cnt(iblk) + 1
!
!          hh_gath(1   ,hh_cnt(iblk),iblk) = tau
!          hh_gath(2:nb,hh_cnt(iblk),iblk) = hv(2:nb)
!
!          nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
!          nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
!                                        ! Note that nr>=0 implies that diagonal block is full (nc==nb)!
!
!          ! Multiply diagonal block and subdiagonal block with Householder vector
!
!          if(iblk==nblocks .and. nc==nb) then
!
!            ! We need the last column from the next PE.
!            ! First do the matrix multiplications without last column ...
!
!            ! Diagonal block, the contribution of the last element is added below!
!            ab(1,ne) = 0
!            call DSYMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,0.d0,hd,1)
!
!            ! Subdiagonal block
!            if(nr>0) call DGEMV('N',nr,nb-1,tau,ab(nb+1,ns),2*nb-1,hv,1,0.d0,hs,1)
!
!            ! ... then request last column ...
!            call mpi_recv(ab(1,ne),nb+1,mpi_real8,my_pe+1,1,mpi_comm,mpi_status,mpierr)
!
!            ! ... and complete the result
!            hs(1:nr) = hs(1:nr) + ab(2:nr+1,ne)*tau*hv(nb)
!            hd(nb) = hd(nb) + ab(1,ne)*hv(nb)*tau
!
!          else
!
!            ! Normal matrix multiply
!            call DSYMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,0.d0,hd,1)
!            if(nr>0) call DGEMV('N',nr,nb,tau,ab(nb+1,ns),2*nb-1,hv,1,0.d0,hs,1)
!
!          endif
!
!          ! Calculate first column of subdiagonal block and calculate new
!          ! Householder transformation for this column
!
!          hv_new(:) = 0 ! Needed, last rows must be 0 for nr < nb
!          tau_new = 0
!
!          if(nr>0) then
!
!            ! complete (old) Householder transformation for first column
!
!            ab(nb+1:nb+nr,ns) = ab(nb+1:nb+nr,ns) - hs(1:nr) ! Note: hv(1) == 1
!
!            ! calculate new Householder transformation ...
!            if(nr>1) then
!              vnorm2 = sum(ab(nb+2:nb+nr,ns)**2)
!              call hh_transform_real(ab(nb+1,ns),vnorm2,hf,tau_new)
!              hv_new(1) = 1.
!              hv_new(2:nr) = ab(nb+2:nb+nr,ns)*hf
!              ab(nb+2:,ns) = 0
!            endif
!
!            ! ... and send it away immediatly if this is the last block
!
!            if(iblk==nblocks) then
!              call mpi_wait(ireq_hv,mpi_status,mpierr)
!              hv_s(1) = tau_new
!              hv_s(2:) = hv_new(2:)
!              call mpi_isend(hv_s,nb,mpi_real8,my_pe+1,2,mpi_comm,ireq_hv,mpierr)
!            endif
!
!          endif
!
!
!          ! Transform diagonal block
!          x = dot_product(hv(1:nc),hd(1:nc))*tau
!          hd(1:nc) = hd(1:nc) - 0.5*x*hv(1:nc)
!
!          if(my_pe>0 .and. iblk==1) then
!
!            ! The first column of the diagonal block has to be send to the previous PE
!            ! Calculate first column only ...
!
!            ab(1:nc,ns) = ab(1:nc,ns) - hd(1:nc)*hv(1) - hv(1:nc)*hd(1)
!
!            ! ... send it away ...
!
!            call mpi_wait(ireq_ab,mpi_status,mpierr)
!            ab_s(1:nb+1) = ab(1:nb+1,ns)
!            call mpi_isend(ab_s,nb+1,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
!
!            ! ... and calculate remaining columns with rank-2 update
!            if(nc>1) call DSYR2('L',nc-1,-1.d0,hd(2),1,hv(2),1,ab(1,ns+1),2*nb-1)
!          else
!            ! No need to  send, just a rank-2 update
!            call DSYR2('L',nc,-1.d0,hd,1,hv,1,ab(1,ns),2*nb-1)
!          endif
!
!          ! Do the remaining double Householder transformation on the subdiagonal block cols 2 ... nb
!
!          if(nr>0) then
!            if(nr>1) then
!              call DGEMV('T',nr,nb-1,tau_new,ab(nb,ns+1),2*nb-1,hv_new,1,0.d0,h(2),1)
!              x = dot_product(hs(1:nr),hv_new(1:nr))*tau_new
!              h(2:nb) = h(2:nb) - x*hv(2:nb)
!              ! Unfortunately there is no BLAS routine like DSYR2 for a nonsymmetric rank 2 update ("DGER2")
!              do i=2,nb
!                ab(2+nb-i:1+nb+nr-i,i+ns-1) = ab(2+nb-i:1+nb+nr-i,i+ns-1) - hv_new(1:nr)*h(i) - hs(1:nr)*hv(i)
!              enddo
!            else
!              ! No double Householder transformation for nr=1, just complete the row
!              do i=2,nb
!                ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*hv(i)
!              enddo
!            endif
!          endif
!
!          ! Use new HH vector for the next block
!          hv(:) = hv_new(:)
!          tau = tau_new
!
!        enddo

      endif

      do iblk = 1, nblocks

        if(hh_dst(iblk) >= np_rows) exit
        if(snd_limits(hh_dst(iblk)+1,iblk) == snd_limits(hh_dst(iblk),iblk)) exit

        if(hh_cnt(iblk) == snd_limits(hh_dst(iblk)+1,iblk)-snd_limits(hh_dst(iblk),iblk)) then
          ! Wait for last transfer to finish
          call mpi_wait(ireq_hhs(iblk), mpi_status, mpierr)
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

      enddo

    enddo

    ! Finish the last outstanding requests
    call mpi_wait(ireq_ab,mpi_status,mpierr)
    call mpi_wait(ireq_hv,mpi_status,mpierr)

    allocate(mpi_statuses(MPI_STATUS_SIZE,max(nblocks,num_chunks)))
    call mpi_waitall(nblocks, ireq_hhs, mpi_statuses, mpierr)
    call mpi_waitall(num_chunks, ireq_hhr, mpi_statuses, mpierr)
    deallocate(mpi_statuses)

    call mpi_barrier(mpi_comm,mpierr)

    deallocate(ab)
    deallocate(ireq_hhr, ireq_hhs)
    deallocate(hh_cnt, hh_dst)
    deallocate(hh_gath, hh_send)
    deallocate(limits, snd_limits)
    deallocate(block_limits)
    deallocate(global_id)

end subroutine tridiag_band_real_test

! --------------------------------------------------------------------------------------------------

attributes(global) subroutine hh_kernel1(a,hh,tau,ncols,nbw,a_dim2)

  real*8 :: a(48,a_dim2,1,*), hh(nbw,*), tau(*)
  integer, value :: ncols, nbw, a_dim2

  integer i, j, n, mythread, myblock, myoff, myoff_c, myt, aoff
  real*8 :: z1, z2, z3
  real*8, shared :: ash(48,32+7), zz(48,2), h(32)

  mythread = threadidx%x
  myblock  = blockidx%x
  if(mythread <= 16) then
    myoff = 3*(mythread-1)
    myoff_c = 0
    myt = 1
  else
    myoff = 3*(mythread-17)
    myoff_c = 16
    myt = 2
  endif

  aoff = 7

  do i = myoff_c+1, myoff_c+16
    n = i+aoff ! iand(ncols+i-2,31)+1
    ash(myoff+1,n) = a(myoff+1,ncols+i-1,1,myblock)
    ash(myoff+2,n) = a(myoff+2,ncols+i-1,1,myblock)
    ash(myoff+3,n) = a(myoff+3,ncols+i-1,1,myblock)
  enddo
  call syncthreads()

  do j = ncols, 1, -1
    h(mythread) = hh(mythread,j)
    if(j<ncols) then
      if(aoff==0) then
        if(myt==1) then
          do i = 31,1,-1
            ash(myoff+1,i+8) = ash(myoff+1,i)
            ash(myoff+2,i+8) = ash(myoff+2,i)
            ash(myoff+3,i+8) = ash(myoff+3,i)
          enddo
        endif
        aoff = 8
      endif
      aoff = aoff-1
      if(myt==1) then
        n = 1+aoff
        ash(myoff+1,n) = a(myoff+1,j+1-1,1,myblock)
        ash(myoff+2,n) = a(myoff+2,j+1-1,1,myblock)
        ash(myoff+3,n) = a(myoff+3,j+1-1,1,myblock)
      endif
    endif
    call syncthreads()
    z1 = 0.
    z2 = 0.
    z3 = 0.
    do i = myoff_c+1, myoff_c+16, 4
      !z = z + a(mythread,j+i-1,1,myblock)*hh(i,j)
      n = i+aoff
      z1 = z1 + ash(myoff+1,n)*h(i) + ash(myoff+1,n+1)*h(i+1) + ash(myoff+1,n+2)*h(i+2) + ash(myoff+1,n+3)*h(i+3)
      z2 = z2 + ash(myoff+2,n)*h(i) + ash(myoff+2,n+1)*h(i+1) + ash(myoff+2,n+2)*h(i+2) + ash(myoff+2,n+3)*h(i+3)
      z3 = z3 + ash(myoff+3,n)*h(i) + ash(myoff+3,n+1)*h(i+1) + ash(myoff+3,n+2)*h(i+2) + ash(myoff+3,n+3)*h(i+3)
    enddo
    zz(myoff+1,myt) = z1
    zz(myoff+2,myt) = z2
    zz(myoff+3,myt) = z3
    call syncthreads()
    z1 = zz(myoff+1,1) + zz(myoff+1,2)
    z2 = zz(myoff+2,1) + zz(myoff+2,2)
    z3 = zz(myoff+3,1) + zz(myoff+3,2)
    z1 = z1 * tau(j)
    z2 = z2 * tau(j)
    z3 = z3 * tau(j)
    do i = myoff_c+1, myoff_c+16
      !a(mythread,j+i-1,1,myblock) = a(mythread,j+i-1,1,myblock) - z*hh(i,j)
      n = i+aoff
      ash(myoff+1,n) = ash(myoff+1,n) - z1*h(i)
      ash(myoff+2,n) = ash(myoff+2,n) - z2*h(i)
      ash(myoff+3,n) = ash(myoff+3,n) - z3*h(i)
    enddo
    if(myt==2) then
    n = nbw+aoff
    a(myoff+1,j+nbw-1,1,myblock) = ash(myoff+1,n)
    a(myoff+2,j+nbw-1,1,myblock) = ash(myoff+2,n)
    a(myoff+3,j+nbw-1,1,myblock) = ash(myoff+3,n)
    endif
    call syncthreads()
  enddo

  do i = myoff_c+1, myoff_c+16
    n = i+aoff
    a(myoff+1,i,1,myblock) = ash(myoff+1,n)
    a(myoff+2,i,1,myblock) = ash(myoff+2,n)
    a(myoff+3,i,1,myblock) = ash(myoff+3,n)
  enddo

  call syncthreads()

end subroutine

! --------------------------------------------------------------------------------------------------

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
  
! --------------------------------------------------------------------------------------------------

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

!-------------------------------------------------------------------------------

subroutine bandred_complex(na, a, lda, nblk, nbw, mpi_comm_rows, mpi_comm_cols, tmat)

!-------------------------------------------------------------------------------
!  bandred_complex: Reduces a distributed hermitian matrix to band form
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
   complex*16 a(lda,*), tmat(nbw,nbw,*)

   complex*16, parameter :: CZERO = (0.d0,0.d0), CONE = (1.d0,0.d0)

   integer my_prow, my_pcol, np_rows, np_cols, mpierr
   integer l_cols, l_rows
   integer i, j, lcs, lce, lre, lc, lr, cur_pcol, n_cols, nrow
   integer istep, ncol, lch, lcx, nlc
   integer tile_size, l_rows_tile, l_cols_tile

   real*8 vnorm2
   complex*16 xf, aux1(nbw), aux2(nbw), vrl, tau, vav(nbw,nbw)

   complex*16, allocatable:: tmp(:,:), vr(:), vmr(:,:), umc(:,:)

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

            call mpi_allreduce(aux1,aux2,2,MPI_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_rows,mpierr)

            vnorm2 = aux2(1)
            vrl    = aux2(2)

            ! Householder transformation

            call hh_transform_complex(vrl, vnorm2, xf, tau)

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
         call MPI_Bcast(vr,lr+1,MPI_DOUBLE_COMPLEX,cur_pcol,mpi_comm_cols,mpierr)
         vmr(1:lr,lc) = vr(1:lr)
         tau = vr(lr+1)
         tmat(lc,lc,istep) = conjg(tau) ! Store tau in diagonal of tmat

         ! Transform remaining columns in current block with Householder vector

         ! Local dot product

         aux1 = 0

         nlc = 0 ! number of local columns
         do j=1,lc-1
            lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
            if(lcx>0) then
               nlc = nlc+1
               aux1(nlc) = dot_product(vr(1:lr),a(1:lr,lcx))
            endif
         enddo

         ! Get global dot products
         if(nlc>0) call mpi_allreduce(aux1,aux2,nlc,MPI_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_rows,mpierr)

         ! Transform

         nlc = 0
         do j=1,lc-1
            lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
            if(lcx>0) then
               nlc = nlc+1
               a(1:lr,lcx) = a(1:lr,lcx) - conjg(tau)*aux2(nlc)*vr(1:lr)
            endif
         enddo

      enddo

      ! Calculate scalar products of stored Householder vectors.
      ! This can be done in different ways, we use zherk

      vav = 0
      if(l_rows>0) &
         call zherk('U','C',n_cols,l_rows,CONE,vmr,ubound(vmr,1),CZERO,vav,ubound(vav,1))
      call herm_matrix_allreduce(n_cols,vav,ubound(vav,1),mpi_comm_rows)

      ! Calculate triangular matrix T for block Householder Transformation

      do lc=n_cols,1,-1
         tau = tmat(lc,lc,istep)
         if(lc<n_cols) then
            call ztrmv('U','C','N',n_cols-lc,tmat(lc+1,lc+1,istep),ubound(tmat,1),vav(lc+1,lc),1)
            tmat(lc,lc+1:n_cols,istep) = -tau * conjg(vav(lc+1:n_cols,lc))
         endif
      enddo

      ! Transpose vmr -> vmc (stored in umc, second half)

      call elpa_transpose_vectors  (vmr, 2*ubound(vmr,1), mpi_comm_rows, &
                                    umc(1,n_cols+1), 2*ubound(umc,1), mpi_comm_cols, &
                                    1, 2*istep*nbw, n_cols, 2*nblk)

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
            call ZGEMM('C','N',lce-lcs+1,n_cols,lre,CONE,a(1,lcs),ubound(a,1), &
                       vmr,ubound(vmr,1),CONE,umc(lcs,1),ubound(umc,1))

            if(i==0) cycle
            lre = min(l_rows,i*l_rows_tile)
            call ZGEMM('N','N',lre,n_cols,lce-lcs+1,CONE,a(1,lcs),lda, &
                       umc(lcs,n_cols+1),ubound(umc,1),CONE,vmr(1,n_cols+1),ubound(vmr,1))
         enddo
      endif

      ! Sum up all ur(:) parts along rows and add them to the uc(:) parts
      ! on the processors containing the diagonal
      ! This is only necessary if ur has been calculated, i.e. if the
      ! global tile size is smaller than the global remaining matrix

      if(tile_size < istep*nbw) then
         call elpa_reduce_add_vectors  (vmr(1,n_cols+1),2*ubound(vmr,1),mpi_comm_rows, &
                                        umc, 2*ubound(umc,1), mpi_comm_cols, &
                                        2*istep*nbw, n_cols, 2*nblk)
      endif

      if(l_cols>0) then
         allocate(tmp(l_cols,n_cols))
         call mpi_allreduce(umc,tmp,l_cols*n_cols,MPI_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_rows,mpierr)
         umc(1:l_cols,1:n_cols) = tmp(1:l_cols,1:n_cols)
         deallocate(tmp)
      endif

      ! U = U * Tmat**T

      call ztrmm('Right','Upper','C','Nonunit',l_cols,n_cols,CONE,tmat(1,1,istep),ubound(tmat,1),umc,ubound(umc,1))

      ! VAV = Tmat * V**T * A * V * Tmat**T = (U*Tmat**T)**T * V * Tmat**T

      call zgemm('C','N',n_cols,n_cols,l_cols,CONE,umc,ubound(umc,1),umc(1,n_cols+1),ubound(umc,1),CZERO,vav,ubound(vav,1))
      call ztrmm('Right','Upper','C','Nonunit',n_cols,n_cols,CONE,tmat(1,1,istep),ubound(tmat,1),vav,ubound(vav,1))

      call herm_matrix_allreduce(n_cols,vav,ubound(vav,1),mpi_comm_cols)

      ! U = U - 0.5 * V * VAV
      call zgemm('N','N',l_cols,n_cols,n_cols,(-0.5d0,0.d0),umc(1,n_cols+1),ubound(umc,1),vav,ubound(vav,1),CONE,umc,ubound(umc,1))

      ! Transpose umc -> umr (stored in vmr, second half)

       call elpa_transpose_vectors  (umc, 2*ubound(umc,1), mpi_comm_cols, &
                                     vmr(1,n_cols+1), 2*ubound(vmr,1), mpi_comm_rows, &
                                     1, 2*istep*nbw, n_cols, 2*nblk)

      ! A = A - V*U**T - U*V**T

      do i=0,(istep*nbw-1)/tile_size
         lcs = i*l_cols_tile+1
         lce = min(l_cols,(i+1)*l_cols_tile)
         lre = min(l_rows,(i+1)*l_rows_tile)
         if(lce<lcs .or. lre<1) cycle
         call zgemm('N','C',lre,lce-lcs+1,2*n_cols,-CONE, &
                    vmr,ubound(vmr,1),umc(lcs,1),ubound(umc,1), &
                    CONE,a(1,lcs),lda)
      enddo

      deallocate(vmr, umc, vr)

   enddo

end subroutine bandred_complex

!-------------------------------------------------------------------------------

subroutine herm_matrix_allreduce(n,a,lda,comm)

!-------------------------------------------------------------------------------
!  herm_matrix_allreduce: Does an mpi_allreduce for a hermitian matrix A.
!  On entry, only the upper half of A needs to be set
!  On exit, the complete matrix is set
!-------------------------------------------------------------------------------

   implicit none
   integer n, lda, comm
   complex*16 a(lda,*)

   integer i, nc, mpierr
   complex*16 h1(n*n), h2(n*n)

   nc = 0
   do i=1,n
      h1(nc+1:nc+i) = a(1:i,i)
      nc = nc+i
   enddo

   call mpi_allreduce(h1,h2,nc,MPI_DOUBLE_COMPLEX,MPI_SUM,comm,mpierr)

   nc = 0
   do i=1,n
      a(1:i,i) = h2(nc+1:nc+i)
      a(i,1:i-1) = conjg(a(1:i-1,i))
      nc = nc+i
   enddo

end subroutine herm_matrix_allreduce

!-------------------------------------------------------------------------------

subroutine trans_ev_band_to_full_complex(na, nqc, nblk, nbw, a, lda, tmat, q, ldq, mpi_comm_rows, mpi_comm_cols)

!-------------------------------------------------------------------------------
!  trans_ev_band_to_full_complex:
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
!  a(lda,*)    Matrix containing the Householder vectors (i.e. matrix a after bandred_complex)
!              Distribution is like in Scalapack.
!
!  lda         Leading dimension of a
!
!  tmat(nbw,nbw,.) Factors returned by bandred_complex
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
   complex*16 a(lda,*), q(ldq,*), tmat(nbw, nbw, *)

   complex*16, parameter :: CZERO = (0.d0,0.d0), CONE = (1.d0,0.d0)

   integer my_prow, my_pcol, np_rows, np_cols, mpierr
   integer max_blocks_row, max_blocks_col, max_local_rows, max_local_cols
   integer l_cols, l_rows, l_colh, n_cols
   integer istep, lc, ncol, nrow, nb, ns

   complex*16, allocatable:: tmp1(:), tmp2(:), hvb(:), hvm(:,:)

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
            call MPI_Bcast(hvb(ns+1),nb-ns,MPI_DOUBLE_COMPLEX,pcol(ncol),mpi_comm_cols,mpierr)
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
         call zgemm('C','N',n_cols,l_cols,l_rows,CONE,hvm,ubound(hvm,1), &
                    q,ldq,CZERO,tmp1,n_cols)
      else
         tmp1(1:l_cols*n_cols) = 0
      endif
      call mpi_allreduce(tmp1,tmp2,n_cols*l_cols,MPI_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_rows,mpierr)
      if(l_rows>0) then
         call ztrmm('L','U','C','N',n_cols,l_cols,CONE,tmat(1,1,istep),ubound(tmat,1),tmp2,n_cols)
         call zgemm('N','N',l_rows,l_cols,n_cols,-CONE,hvm,ubound(hvm,1), &
                    tmp2,n_cols,CONE,q,ldq)
      endif

   enddo

   deallocate(tmp1, tmp2, hvb, hvm)


end subroutine trans_ev_band_to_full_complex

!---------------------------------------------------------------------------------------------------

subroutine tridiag_band_complex(na, nb, nblk, a, lda, d, e, mpi_comm_rows, mpi_comm_cols, mpi_comm)

!-------------------------------------------------------------------------------
! tridiag_band_complex:
! Reduces a complex hermitian band matrix to tridiagonal form
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
    complex*16, intent(in)  :: a(lda,*)
    real*8, intent(out) :: d(na), e(na) ! set only on PE 0


    real*8 vnorm2
    complex*16 hv(nb), tau, x, h(nb), ab_s(1+nb), hv_s(nb), hv_new(nb), tau_new, hf
    complex*16 hd(nb), hs(nb)
    complex*16, allocatable :: hv_t(:,:), tau_t(:)

    integer i, j, n, nc, nr, ns, ne, istep, iblk, nblocks_total, nblocks, nt
    integer my_pe, n_pes, mpierr
    integer my_prow, np_rows, my_pcol, np_cols
    integer ireq_ab, ireq_hv
    integer na_s, nx, num_hh_vecs, num_chunks, local_size, max_blk_size, n_off
    integer max_threads, my_thread, my_block_s, my_block_e, iter
    integer mpi_status(MPI_STATUS_SIZE)
    integer, allocatable :: mpi_statuses(:,:)
    integer, allocatable :: omp_block_limits(:)
    integer, allocatable :: ireq_hhr(:), ireq_hhs(:), global_id(:,:), global_id_tmp(:,:), hh_cnt(:), hh_dst(:)
    integer, allocatable :: limits(:), snd_limits(:,:)
    integer, allocatable :: block_limits(:)
    complex*16, allocatable :: ab(:,:), hh_gath(:,:,:), hh_send(:,:,:)
    ! dummies for calling redist_band
    real*8 :: r_a(1,1), r_ab(1,1)

!$  integer :: omp_get_max_threads


    call mpi_comm_rank(mpi_comm,my_pe,mpierr)
    call mpi_comm_size(mpi_comm,n_pes,mpierr)

    call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
    call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
    call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
    call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

    ! Get global_id mapping 2D procssor coordinates to global id

    allocate(global_id(0:np_rows-1,0:np_cols-1))
    allocate(global_id_tmp(0:np_rows-1,0:np_cols-1))
    global_id(:,:) = 0
    global_id(my_prow, my_pcol) = my_pe

    global_id_tmp(:,:) = global_id(:,:)
    call mpi_allreduce(global_id_tmp, global_id, np_rows*np_cols, mpi_integer, mpi_sum, mpi_comm, mpierr)
    deallocate(global_id_tmp)


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
    call redist_band(.false., r_a, a, lda, na, nblk, nb, mpi_comm_rows, mpi_comm_cols, mpi_comm, r_ab, ab)

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

    allocate(hh_trans_complex(nb,num_hh_vecs))

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
        call mpi_irecv(hh_trans_complex(1,num_hh_vecs+1), nb*local_size, MPI_COMPLEX16, nt, &
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

    ! OpenMP work distribution:

    max_threads = 1
!$ max_threads = omp_get_max_threads()

    ! For OpenMP we need at least 2 blocks for every thread
    max_threads = MIN(max_threads, nblocks/2)
    if(max_threads==0) max_threads = 1

    allocate(omp_block_limits(0:max_threads))

    ! Get the OpenMP block limits
    call divide_band(nblocks, max_threads, omp_block_limits)

    allocate(hv_t(nb,max_threads), tau_t(max_threads))
    hv_t = 0
    tau_t = 0

    ! ---------------------------------------------------------------------------
    ! Start of calculations

    na_s = block_limits(my_pe)*nb + 1

    if(my_pe>0 .and. na_s<=na) then
      ! send first column to previous PE
      ! Only the PE owning the diagonal does that (sending 1 element of the subdiagonal block also)
      ab_s(1:nb+1) = ab(1:nb+1,na_s-n_off)
      call mpi_isend(ab_s,nb+1,MPI_COMPLEX16,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
    endif

    do istep=1,na-1-block_limits(my_pe)*nb

      if(my_pe==0) then
        n = MIN(na-na_s,nb) ! number of rows to be reduced
        hv(:) = 0
        tau = 0
        ! Transform first column of remaining matrix
        ! Opposed to the real case, the last step (istep=na-1) is needed here for making
        ! the last subdiagonal element a real number
        vnorm2 = sum(dble(ab(3:n+1,na_s-n_off))**2+dimag(ab(3:n+1,na_s-n_off))**2)
        if(n<2) vnorm2 = 0. ! Safety only
        call hh_transform_complex(ab(2,na_s-n_off),vnorm2,hf,tau)

        hv(1) = 1
        hv(2:n) = ab(3:n+1,na_s-n_off)*hf

        d(istep) = ab(1,na_s-n_off)
        e(istep) = ab(2,na_s-n_off)
        if(istep == na-1) then
          d(na) = ab(1,na_s+1-n_off)
          e(na) = 0
        endif
      else
        if(na>na_s) then
          ! Receive Householder vector from previous task, from PE owning subdiagonal
          call mpi_recv(hv,nb,MPI_COMPLEX16,my_pe-1,2,mpi_comm,mpi_status,mpierr)
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

      if(max_threads > 1) then

        ! Codepath for OpenMP

        ! Please note that in this case it is absolutely necessary to have at least 2 blocks per thread!
        ! Every thread is one reduction cycle behind its predecessor and thus starts one step later.
        ! This simulates the behaviour of the MPI tasks which also work after each other.
        ! The code would be considerably easier, if the MPI communication would be made within
        ! the parallel region - this is avoided here since this would require 
        ! MPI_Init_thread(MPI_THREAD_MULTIPLE) at the start of the program.

        hv_t(:,1) = hv
        tau_t(1) = tau

        do iter = 1, 2

          ! iter=1 : work on first block
          ! iter=2 : work on remaining blocks
          ! This is done in 2 iterations so that we have a barrier in between:
          ! After the first iteration, it is guaranteed that the last row of the last block
          ! is completed by the next thread.
          ! After the first iteration it is also the place to exchange the last row
          ! with MPI calls

!$omp parallel do private(my_thread, my_block_s, my_block_e, iblk, ns, ne, hv, tau, &
!$omp&                    nc, nr, hs, hd, vnorm2, hf, x, h, i), schedule(static,1), num_threads(max_threads)
          do my_thread = 1, max_threads

            if(iter == 1) then
              my_block_s = omp_block_limits(my_thread-1) + 1
              my_block_e = my_block_s
            else
              my_block_s = omp_block_limits(my_thread-1) + 2
              my_block_e = omp_block_limits(my_thread)
            endif

            do iblk = my_block_s, my_block_e

              ns = na_s + (iblk-1)*nb - n_off - my_thread + 1 ! first column in block
              ne = ns+nb-1                    ! last column in block

              if(istep<my_thread .or. ns+n_off>na) exit

              hv = hv_t(:,my_thread)
              tau = tau_t(my_thread)

              ! Store Householder vector for back transformation

              hh_cnt(iblk) = hh_cnt(iblk) + 1

              hh_gath(1   ,hh_cnt(iblk),iblk) = tau
              hh_gath(2:nb,hh_cnt(iblk),iblk) = hv(2:nb)

              nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
              nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                            ! Note that nr>=0 implies that diagonal block is full (nc==nb)!

              ! Transform diagonal block

              call ZHEMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,(0.d0,0.d0),hd,1)

              x = dot_product(hv(1:nc),hd(1:nc))*conjg(tau)
              hd(1:nc) = hd(1:nc) - 0.5*x*hv(1:nc)

              call ZHER2('L',nc,(-1.d0,0.d0),hd,1,hv,1,ab(1,ns),2*nb-1)

              hv_t(:,my_thread) = 0
              tau_t(my_thread)  = 0

              if(nr<=0) cycle ! No subdiagonal block present any more

              ! Transform subdiagonal block

              call ZGEMV('N',nr,nb,tau,ab(nb+1,ns),2*nb-1,hv,1,(0.d0,0.d0),hs,1)

              if(nr>1) then

                ! complete (old) Householder transformation for first column

                ab(nb+1:nb+nr,ns) = ab(nb+1:nb+nr,ns) - hs(1:nr) ! Note: hv(1) == 1

                ! calculate new Householder transformation for first column
                ! (stored in hv_t(:,my_thread) and tau_t(my_thread))

                vnorm2 = sum(dble(ab(nb+2:nb+nr,ns))**2+dimag(ab(nb+2:nb+nr,ns))**2)
                call hh_transform_complex(ab(nb+1,ns),vnorm2,hf,tau_t(my_thread))
                hv_t(1   ,my_thread) = 1.
                hv_t(2:nr,my_thread) = ab(nb+2:nb+nr,ns)*hf
                ab(nb+2:,ns) = 0

                ! update subdiagonal block for old and new Householder transformation
                ! This way we can use a nonsymmetric rank 2 update which is (hopefully) faster

                call ZGEMV('C',nr,nb-1,tau_t(my_thread),ab(nb,ns+1),2*nb-1,hv_t(1,my_thread),1,(0.d0,0.d0),h(2),1)
                x = dot_product(hs(1:nr),hv_t(1:nr,my_thread))*tau_t(my_thread)
                h(2:nb) = h(2:nb) - x*hv(2:nb)
                ! Unfortunately there is no BLAS routine like DSYR2 for a nonsymmetric rank 2 update ("DGER2")
                do i=2,nb
                  ab(2+nb-i:1+nb+nr-i,i+ns-1) = ab(2+nb-i:1+nb+nr-i,i+ns-1) - hv_t(1:nr,my_thread)*conjg(h(i)) - hs(1:nr)*conjg(hv(i))
                enddo

              else

                ! No new Householder transformation for nr=1, just complete the old one
                ab(nb+1,ns) = ab(nb+1,ns) - hs(1) ! Note: hv(1) == 1
                do i=2,nb
                  ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*conjg(hv(i))
                enddo
                ! For safety: there is one remaining dummy transformation (but tau is 0 anyways)
                hv_t(1,my_thread) = 1.

              endif

            enddo

          enddo ! my_thread
!$omp end parallel do

          if (iter==1) then
            ! We are at the end of the first block

            ! Send our first column to previous PE
            if(my_pe>0 .and. na_s <= na) then
              call mpi_wait(ireq_ab,mpi_status,mpierr)
              ab_s(1:nb+1) = ab(1:nb+1,na_s-n_off)
              call mpi_isend(ab_s,nb+1,MPI_COMPLEX16,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
            endif

            ! Request last column from next PE
            ne = na_s + nblocks*nb - (max_threads-1) - 1
            if(istep>=max_threads .and. ne <= na) then
              call mpi_recv(ab(1,ne-n_off),nb+1,MPI_COMPLEX16,my_pe+1,1,mpi_comm,mpi_status,mpierr)
            endif

          else
            ! We are at the end of all blocks

            ! Send last HH vector and TAU to next PE if it has been calculated above
            ne = na_s + nblocks*nb - (max_threads-1) - 1
            if(istep>=max_threads .and. ne < na) then
              call mpi_wait(ireq_hv,mpi_status,mpierr)
              hv_s(1) = tau_t(max_threads)
              hv_s(2:) = hv_t(2:,max_threads)
              call mpi_isend(hv_s,nb,MPI_COMPLEX16,my_pe+1,2,mpi_comm,ireq_hv,mpierr)
            endif

            ! "Send" HH vector and TAU to next OpenMP thread
            do my_thread = max_threads, 2, -1
              hv_t(:,my_thread) = hv_t(:,my_thread-1)
              tau_t(my_thread)  = tau_t(my_thread-1)
            enddo

          endif
        enddo ! iter

      else

        ! Codepath for 1 thread without OpenMP

        ! The following code is structured in a way to keep waiting times for
        ! other PEs at a minimum, especially if there is only one block.
        ! For this reason, it requests the last column as late as possible
        ! and sends the Householder vector and the first column as early
        ! as possible.

        do iblk=1,nblocks

          ns = na_s + (iblk-1)*nb - n_off ! first column in block
          ne = ns+nb-1                    ! last column in block

          if(ns+n_off>na) exit

          ! Store Householder vector for back transformation

          hh_cnt(iblk) = hh_cnt(iblk) + 1

          hh_gath(1   ,hh_cnt(iblk),iblk) = tau
          hh_gath(2:nb,hh_cnt(iblk),iblk) = hv(2:nb)

          nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
          nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                        ! Note that nr>=0 implies that diagonal block is full (nc==nb)!

          ! Multiply diagonal block and subdiagonal block with Householder vector

          if(iblk==nblocks .and. nc==nb) then

            ! We need the last column from the next PE.
            ! First do the matrix multiplications without last column ...

            ! Diagonal block, the contribution of the last element is added below!
            ab(1,ne) = 0
            call ZHEMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,(0.d0,0.d0),hd,1)

            ! Subdiagonal block
            if(nr>0) call ZGEMV('N',nr,nb-1,tau,ab(nb+1,ns),2*nb-1,hv,1,(0.d0,0.d0),hs,1)

            ! ... then request last column ...
            call mpi_recv(ab(1,ne),nb+1,MPI_COMPLEX16,my_pe+1,1,mpi_comm,mpi_status,mpierr)

            ! ... and complete the result
            hs(1:nr) = hs(1:nr) + ab(2:nr+1,ne)*tau*hv(nb)
            hd(nb) = hd(nb) + ab(1,ne)*hv(nb)*tau

          else

            ! Normal matrix multiply
            call ZHEMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,(0.d0,0.d0),hd,1)
            if(nr>0) call ZGEMV('N',nr,nb,tau,ab(nb+1,ns),2*nb-1,hv,1,(0.d0,0.d0),hs,1)

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
              vnorm2 = sum(dble(ab(nb+2:nb+nr,ns))**2+dimag(ab(nb+2:nb+nr,ns))**2)
              call hh_transform_complex(ab(nb+1,ns),vnorm2,hf,tau_new)
              hv_new(1) = 1.
              hv_new(2:nr) = ab(nb+2:nb+nr,ns)*hf
              ab(nb+2:,ns) = 0
            endif

            ! ... and send it away immediatly if this is the last block

            if(iblk==nblocks) then
              call mpi_wait(ireq_hv,mpi_status,mpierr)
              hv_s(1) = tau_new
              hv_s(2:) = hv_new(2:)
              call mpi_isend(hv_s,nb,MPI_COMPLEX16,my_pe+1,2,mpi_comm,ireq_hv,mpierr)
            endif

          endif


          ! Transform diagonal block
          x = dot_product(hv(1:nc),hd(1:nc))*conjg(tau)
          hd(1:nc) = hd(1:nc) - 0.5*x*hv(1:nc)

          if(my_pe>0 .and. iblk==1) then

            ! The first column of the diagonal block has to be send to the previous PE
            ! Calculate first column only ...

            ab(1:nc,ns) = ab(1:nc,ns) - hd(1:nc)*conjg(hv(1)) - hv(1:nc)*conjg(hd(1))

            ! ... send it away ...

            call mpi_wait(ireq_ab,mpi_status,mpierr)
            ab_s(1:nb+1) = ab(1:nb+1,ns)
            call mpi_isend(ab_s,nb+1,MPI_COMPLEX16,my_pe-1,1,mpi_comm,ireq_ab,mpierr)

            ! ... and calculate remaining columns with rank-2 update
            if(nc>1) call ZHER2('L',nc-1,(-1.d0,0.d0),hd(2),1,hv(2),1,ab(1,ns+1),2*nb-1)
          else
            ! No need to  send, just a rank-2 update
            call ZHER2('L',nc,(-1.d0,0.d0),hd,1,hv,1,ab(1,ns),2*nb-1)
          endif

          ! Do the remaining double Householder transformation on the subdiagonal block cols 2 ... nb

          if(nr>0) then
            if(nr>1) then
              call ZGEMV('C',nr,nb-1,tau_new,ab(nb,ns+1),2*nb-1,hv_new,1,(0.d0,0.d0),h(2),1)
              x = dot_product(hs(1:nr),hv_new(1:nr))*tau_new
              h(2:nb) = h(2:nb) - x*hv(2:nb)
              ! Unfortunately there is no BLAS routine like DSYR2 for a nonsymmetric rank 2 update ("DGER2")
              do i=2,nb
                ab(2+nb-i:1+nb+nr-i,i+ns-1) = ab(2+nb-i:1+nb+nr-i,i+ns-1) - hv_new(1:nr)*conjg(h(i)) - hs(1:nr)*conjg(hv(i))
              enddo
            else
              ! No double Householder transformation for nr=1, just complete the row
              do i=2,nb
                ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*conjg(hv(i))
              enddo
            endif
          endif

          ! Use new HH vector for the next block
          hv(:) = hv_new(:)
          tau = tau_new

        enddo

      endif

      do iblk = 1, nblocks

        if(hh_dst(iblk) >= np_rows) exit
        if(snd_limits(hh_dst(iblk)+1,iblk) == snd_limits(hh_dst(iblk),iblk)) exit

        if(hh_cnt(iblk) == snd_limits(hh_dst(iblk)+1,iblk)-snd_limits(hh_dst(iblk),iblk)) then
          ! Wait for last transfer to finish
          call mpi_wait(ireq_hhs(iblk), mpi_status, mpierr)
          ! Copy vectors into send buffer
          hh_send(:,1:hh_cnt(iblk),iblk) = hh_gath(:,1:hh_cnt(iblk),iblk)
          ! Send to destination
          call mpi_isend(hh_send(1,1,iblk), nb*hh_cnt(iblk), MPI_COMPLEX16, &
                         global_id(hh_dst(iblk),mod(iblk+block_limits(my_pe)-1,np_cols)), &
                         10+iblk, mpi_comm, ireq_hhs(iblk), mpierr)
          ! Reset counter and increase destination row
          hh_cnt(iblk) = 0
          hh_dst(iblk) = hh_dst(iblk)+1
        endif

      enddo

    enddo

    ! Finish the last outstanding requests
    call mpi_wait(ireq_ab,mpi_status,mpierr)
    call mpi_wait(ireq_hv,mpi_status,mpierr)

    allocate(mpi_statuses(MPI_STATUS_SIZE,max(nblocks,num_chunks)))
    call mpi_waitall(nblocks, ireq_hhs, mpi_statuses, mpierr)
    call mpi_waitall(num_chunks, ireq_hhr, mpi_statuses, mpierr)
    deallocate(mpi_statuses)

    call mpi_barrier(mpi_comm,mpierr)

    deallocate(ab)
    deallocate(ireq_hhr, ireq_hhs)
    deallocate(hh_cnt, hh_dst)
    deallocate(hh_gath, hh_send)
    deallocate(limits, snd_limits)
    deallocate(block_limits)
    deallocate(global_id)

end subroutine tridiag_band_complex

!---------------------------------------------------------------------------------------------------

subroutine trans_ev_tridi_to_band_complex(na, nev, nblk, nbw, q, ldq, mpi_comm_rows, mpi_comm_cols)

!-------------------------------------------------------------------------------
!  trans_ev_tridi_to_band_complex:
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
    complex*16 q(ldq,*)

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

    complex*16, allocatable :: a(:,:,:,:), row(:)
    complex*16, allocatable :: top_border_send_buffer(:,:), top_border_recv_buffer(:,:)
    complex*16, allocatable :: bottom_border_send_buffer(:,:), bottom_border_recv_buffer(:,:)
    complex*16, allocatable :: result_buffer(:,:,:)
    complex*16, allocatable :: bcast_buffer(:,:)

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
        ! Suggested stripe width is 48 - should this be reduced for the complex case ???
        thread_width = (l_nev-1)/max_threads + 1 ! number of eigenvectors per OMP thread
        stripe_width = 48 ! Must be a multiple of 4
        stripe_count = (thread_width-1)/stripe_width + 1
        ! Adapt stripe width so that last one doesn't get too small
        stripe_width = (thread_width-1)/stripe_count + 1
        stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 4 !!!
    endif

    ! Determine the matrix distribution at the beginning

    allocate(limits(0:np_rows))

    call determine_workload(na, nbw, np_rows, limits)

    max_blk_size = maxval(limits(1:np_rows) - limits(0:np_rows-1))

    a_dim2 = max_blk_size + nbw

    allocate(a(stripe_width,a_dim2,stripe_count,max_threads))
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

!$omp parallel do private(my_thread), schedule(static, 1)
    do my_thread = 1, max_threads
        a(:,:,:,my_thread) = 0 ! if possible, do first touch allocation!
    enddo

    do ip = np_rows-1, 0, -1
        if(my_prow == ip) then
            ! Receive my rows which have not yet been received
            src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
            do i=limits(ip)+1,limits(ip+1)
                src = mod((i-1)/nblk, np_rows)
                if(src < my_prow) then
                    call MPI_Recv(row, l_nev, MPI_COMPLEX16, src, 0, mpi_comm_rows, mpi_status, mpierr)
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
                    call MPI_Send(row, l_nev, MPI_COMPLEX16, dst, 0, mpi_comm_rows, mpierr)
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
                    call MPI_Send(row, l_nev, MPI_COMPLEX16, ip, 0, mpi_comm_rows, mpierr)
                endif
            enddo
            ! Receive all rows from PE ip
            do i=limits(my_prow)+1,limits(my_prow+1)
                src = mod((i-1)/nblk, np_rows)
                if(src == ip) then
                    call MPI_Recv(row, l_nev, MPI_COMPLEX16, src, 0, mpi_comm_rows, mpi_status, mpierr)
!$omp parallel do private(my_thread), schedule(static, 1)
                    do my_thread = 1, max_threads
                        call unpack_row(row,i-limits(my_prow),my_thread)
                    enddo
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
            call MPI_Irecv(result_buffer(1,1,j), l_nev*nblk, MPI_COMPLEX16, 0, result_recv_tag, &
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

    allocate(top_border_send_buffer(stripe_width*nbw*max_threads, stripe_count))
    allocate(top_border_recv_buffer(stripe_width*nbw*max_threads, stripe_count))
    allocate(bottom_border_send_buffer(stripe_width*nbw*max_threads, stripe_count))
    allocate(bottom_border_recv_buffer(stripe_width*nbw*max_threads, stripe_count))

    top_border_send_buffer(:,:) = 0
    top_border_recv_buffer(:,:) = 0
    bottom_border_send_buffer(:,:) = 0
    bottom_border_recv_buffer(:,:) = 0

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
                csw = min(stripe_width, thread_width-(i-1)*stripe_width) ! "current_stripe_width"
                b_len = csw*nbw*max_threads
                call MPI_Irecv(bottom_border_recv_buffer(1,i), b_len, MPI_COMPLEX16, my_prow+1, bottom_recv_tag, &
                           mpi_comm_rows, bottom_recv_request(i), mpierr)
            enddo
        endif

        if(current_local_n > 1) then
            if(my_pcol == mod(sweep,np_cols)) then
                bcast_buffer(:,1:current_local_n) = hh_trans_complex(:,current_tv_off+1:current_tv_off+current_local_n)
                current_tv_off = current_tv_off + current_local_n
            endif
            call mpi_bcast(bcast_buffer, nbw*current_local_n, MPI_COMPLEX16, mod(sweep,np_cols), mpi_comm_cols, mpierr)
        else
            ! for current_local_n == 1 the one and only HH vector is 0 and not stored in hh_trans_complex
            bcast_buffer(:,1) = 0
        endif

        if(l_nev == 0) cycle

        if(current_local_n > 0) then

          do i = 1, stripe_count

            ! Get real stripe width for strip i;
            ! The last OpenMP tasks may have an even smaller stripe with,
            ! but we don't care about this, i.e. we send/recv a bit too much in this case.
            ! csw: current_stripe_width

            csw = min(stripe_width, thread_width-(i-1)*stripe_width)

            !wait_b
            if(current_n_end < current_n) then
                call MPI_Wait(bottom_recv_request(i), mpi_status, mpierr)
!$omp parallel do private(my_thread, n_off, b_len, b_off), schedule(static, 1)
                do my_thread = 1, max_threads
                    n_off = current_local_n+a_off
                    b_len = csw*nbw
                    b_off = (my_thread-1)*b_len
                    a(1:csw,n_off+1:n_off+nbw,i,my_thread) = &
                      reshape(bottom_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, nbw /))
                enddo
                if(next_n_end < next_n) then
                    call MPI_Irecv(bottom_border_recv_buffer(1,i), csw*nbw*max_threads, &
                                   MPI_COMPLEX16, my_prow+1, bottom_recv_tag, &
                                   mpi_comm_rows, bottom_recv_request(i), mpierr)
                endif
            endif

            if(current_local_n <= bottom_msg_length + top_msg_length) then

                !wait_t
                if(top_msg_length>0) then
                    call MPI_Wait(top_recv_request(i), mpi_status, mpierr)
                endif

                !compute
!$omp parallel do private(my_thread, n_off, b_len, b_off), schedule(static, 1)
                do my_thread = 1, max_threads
                    if(top_msg_length>0) then
                        b_len = csw*top_msg_length
                        b_off = (my_thread-1)*b_len
                        a(1:csw,a_off+1:a_off+top_msg_length,i,my_thread) = &
                          reshape(top_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, top_msg_length /))
                    endif
                    call compute_hh_trafo(0, current_local_n, i, my_thread)
                enddo

                !send_b
                call MPI_Wait(bottom_send_request(i), mpi_status, mpierr)
                if(bottom_msg_length>0) then
                    n_off = current_local_n+nbw-bottom_msg_length+a_off
                    b_len = csw*bottom_msg_length*max_threads
                    bottom_border_send_buffer(1:b_len,i) = &
                        reshape(a(1:csw,n_off+1:n_off+bottom_msg_length,i,:), (/ b_len /))
                    call MPI_Isend(bottom_border_send_buffer(1,i), b_len, MPI_COMPLEX16, my_prow+1, &
                                   top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
                endif

            else

                !compute
!$omp parallel do private(my_thread, b_len, b_off), schedule(static, 1)
                do my_thread = 1, max_threads
                    call compute_hh_trafo(current_local_n - bottom_msg_length, bottom_msg_length, i, my_thread)
                enddo

                !send_b
                call MPI_Wait(bottom_send_request(i), mpi_status, mpierr)
                if(bottom_msg_length > 0) then
                    n_off = current_local_n+nbw-bottom_msg_length+a_off
                    b_len = csw*bottom_msg_length*max_threads
                    bottom_border_send_buffer(1:b_len,i) = &
                      reshape(a(1:csw,n_off+1:n_off+bottom_msg_length,i,:), (/ b_len /))
                    call MPI_Isend(bottom_border_send_buffer(1,i), b_len, MPI_COMPLEX16, my_prow+1, &
                                   top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
                endif

                !compute
!$omp parallel do private(my_thread), schedule(static, 1)
                do my_thread = 1, max_threads
                    call compute_hh_trafo(top_msg_length, current_local_n-top_msg_length-bottom_msg_length, i, my_thread)
                enddo

                !wait_t
                if(top_msg_length>0) then
                    call MPI_Wait(top_recv_request(i), mpi_status, mpierr)
                endif

                !compute
!$omp parallel do private(my_thread, b_len, b_off), schedule(static, 1)
                do my_thread = 1, max_threads
                    if(top_msg_length>0) then
                        b_len = csw*top_msg_length
                        b_off = (my_thread-1)*b_len
                        a(1:csw,a_off+1:a_off+top_msg_length,i,my_thread) = &
                          reshape(top_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, top_msg_length /))
                    endif
                    call compute_hh_trafo(0, top_msg_length, i, my_thread)
                enddo
            endif

            if(next_top_msg_length > 0) then
                !request top_border data
                b_len = csw*next_top_msg_length*max_threads
                call MPI_Irecv(top_border_recv_buffer(1,i), b_len, MPI_COMPLEX16, my_prow-1, &
                               top_recv_tag, mpi_comm_rows, top_recv_request(i), mpierr)
            endif

            !send_t
            if(my_prow > 0) then
                call MPI_Wait(top_send_request(i), mpi_status, mpierr)
                b_len = csw*nbw*max_threads
                top_border_send_buffer(1:b_len,i) = reshape(a(1:csw,a_off+1:a_off+nbw,i,:), (/ b_len /))
                call MPI_Isend(top_border_send_buffer(1,i), b_len, MPI_COMPLEX16, &
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
          do i = 1, stripe_count
            call MPI_Wait(top_send_request(i), mpi_status, mpierr)
          enddo
        endif

        ! Care about the result

        if(my_prow == 0) then

            ! topmost process sends nbw rows to destination processes

            do j=0,nfact-1

                num_blk = sweep*nfact+j ! global number of destination block, 0 based
                if(num_blk*nblk >= na) exit

                nbuf = mod(num_blk, num_result_buffers) + 1 ! buffer number to get this block

                call MPI_Wait(result_send_request(nbuf), mpi_status, mpierr)

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
                    call MPI_Isend(result_buffer(1,1,nbuf), l_nev*nblk, MPI_COMPLEX16, dst, &
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
                    call MPI_Test(result_recv_request(nbuf), flag, mpi_status, mpierr)
                    if(.not.flag) exit
                else
                    call MPI_Wait(result_recv_request(nbuf), mpi_status, mpierr)
                endif

                ! Fill result buffer into q
                num_blk = j*np_rows + my_prow ! global number of current block, 0 based
                do i = 1, min(na - num_blk*nblk, nblk)
                    q(j*nblk+i, 1:l_nev) = result_buffer(1:l_nev, i, nbuf)
                enddo

                ! Queue result buffer again if there are outstanding blocks left
                if(j+num_result_buffers < num_result_blocks) &
                    call MPI_Irecv(result_buffer(1,1,nbuf), l_nev*nblk, MPI_COMPLEX16, 0, result_recv_tag, &
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
!$omp parallel do private(my_thread, i, j), schedule(static, 1)
            do my_thread = 1, max_threads
                do i = 1, stripe_count
                    do j = top_msg_length+1, top_msg_length+next_local_n
                       A(:,j,i,my_thread) = A(:,j+a_off,i,my_thread)
                    enddo
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
        allocate(mpi_statuses(MPI_STATUS_SIZE,num_result_buffers))
        call MPI_Waitall(num_result_buffers, result_send_request, mpi_statuses, mpierr)
        deallocate(mpi_statuses)
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
        complex*16 row(:)
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
        complex*16, intent(in)  :: row(:)
        integer i, noff, nl

        do i=1,stripe_count
            noff = (my_thread-1)*thread_width + (i-1)*stripe_width
            nl   = min(stripe_width, my_thread*thread_width-noff, l_nev-noff)
            if(nl<=0) exit
            a(1:nl,n,i,my_thread) = row(noff+1:noff+nl)
        enddo

    end subroutine

    subroutine compute_hh_trafo(off, ncols, istripe, my_thread)

        ! Private variables in OMP regions (my_thread) should better be in the argument list!
        integer, intent(in) :: off, ncols, istripe, my_thread
        integer j, nl, noff
        real*8 ttt

        ttt = mpi_wtime()
        if(istripe<stripe_count) then
          nl = stripe_width
        else
          noff = (my_thread-1)*thread_width + (istripe-1)*stripe_width
          nl = min(my_thread*thread_width-noff, l_nev-noff)
          if(nl<=0) return
        endif
        do j = ncols, 1, -1
          call single_hh_trafo_complex(a(1,j+off+a_off,istripe,my_thread),bcast_buffer(1,j+off),nbw,nl,stripe_width)
        enddo
        if(my_thread==1) then
          kernel_flops = kernel_flops + 4*4*int(nl,8)*int(ncols,8)*int(nbw,8)
          kernel_time  = kernel_time + mpi_wtime()-ttt
        endif

    end subroutine

end subroutine

! --------------------------------------------------------------------------------------------------
! redist_band: redistributes band from 2D block cyclic form to 1D band

subroutine redist_band(l_real, r_a, c_a, lda, na, nblk, nbw, mpi_comm_rows, mpi_comm_cols, mpi_comm, r_ab, c_ab)

   logical, intent(in)     :: l_real
   real*8, intent(in)      :: r_a(lda, *)
   complex*16, intent(in)  :: c_a(lda, *)
   integer, intent(in)     :: lda, na, nblk, nbw, mpi_comm_rows, mpi_comm_cols, mpi_comm
   real*8, intent(out)     :: r_ab(:,:)
   complex*16, intent(out) :: c_ab(:,:)

   integer, allocatable :: ncnt_s(:), nstart_s(:), ncnt_r(:), nstart_r(:), global_id(:,:), global_id_tmp(:,:), block_limits(:)
   real*8, allocatable :: r_sbuf(:,:,:), r_rbuf(:,:,:), r_buf(:,:)
   complex*16, allocatable :: c_sbuf(:,:,:), c_rbuf(:,:,:), c_buf(:,:)

   integer i, j, my_pe, n_pes, my_prow, np_rows, my_pcol, np_cols, nfact, np, npr, npc, mpierr, is, js
   integer nblocks_total, il, jl, l_rows, l_cols, n_off

   call mpi_comm_rank(mpi_comm,my_pe,mpierr)
   call mpi_comm_size(mpi_comm,n_pes,mpierr)

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   ! Get global_id mapping 2D procssor coordinates to global id

   allocate(global_id(0:np_rows-1,0:np_cols-1))
   allocate(global_id_tmp(0:np_rows-1,0:np_cols-1))
   global_id(:,:) = 0
   global_id(my_prow, my_pcol) = my_pe

   global_id_tmp(:,:) = global_id(:,:)
   call mpi_allreduce(global_id_tmp, global_id, np_rows*np_cols, mpi_integer, mpi_sum, mpi_comm, mpierr)
   deallocate(global_id_tmp)


   ! Set work distribution

   nblocks_total = (na-1)/nbw + 1

   allocate(block_limits(0:n_pes))
   call divide_band(nblocks_total, n_pes, block_limits)


   allocate(ncnt_s(0:n_pes-1))
   allocate(nstart_s(0:n_pes-1))
   allocate(ncnt_r(0:n_pes-1))
   allocate(nstart_r(0:n_pes-1))


   nfact = nbw/nblk

   ! Count how many blocks go to which PE

   ncnt_s(:) = 0
   np = 0 ! receiver PE number
   do j=0,(na-1)/nblk ! loop over rows of blocks
      if(j/nfact==block_limits(np+1)) np = np+1
      if(mod(j,np_rows) == my_prow) then
         do i=0,nfact
            if(mod(i+j,np_cols) == my_pcol) then
               ncnt_s(np) = ncnt_s(np) + 1
            endif
         enddo
      endif
   enddo

   ! Allocate send buffer

   if(l_real) then
      allocate(r_sbuf(nblk,nblk,sum(ncnt_s)))
      r_sbuf(:,:,:) = 0.
   else
      allocate(c_sbuf(nblk,nblk,sum(ncnt_s)))
      c_sbuf(:,:,:) = 0.
   endif

   ! Determine start offsets in send buffer

   nstart_s(0) = 0
   do i=1,n_pes-1
      nstart_s(i) = nstart_s(i-1) + ncnt_s(i-1)
   enddo

   ! Fill send buffer

   l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a
   l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local columns of a

   np = 0
   do j=0,(na-1)/nblk ! loop over rows of blocks
      if(j/nfact==block_limits(np+1)) np = np+1
      if(mod(j,np_rows) == my_prow) then
         do i=0,nfact
            if(mod(i+j,np_cols) == my_pcol) then
               nstart_s(np) = nstart_s(np) + 1
               js = (j/np_rows)*nblk
               is = ((i+j)/np_cols)*nblk
               jl = MIN(nblk,l_rows-js)
               il = MIN(nblk,l_cols-is)
               if(l_real) then
                  r_sbuf(1:jl,1:il,nstart_s(np)) = r_a(js+1:js+jl,is+1:is+il)
               else
                  c_sbuf(1:jl,1:il,nstart_s(np)) = c_a(js+1:js+jl,is+1:is+il)
               endif
            endif
         enddo
      endif
   enddo

   ! Count how many blocks we get from which PE

   ncnt_r(:) = 0
   do j=block_limits(my_pe)*nfact,min(block_limits(my_pe+1)*nfact-1,(na-1)/nblk)
      npr = mod(j,np_rows)
      do i=0,nfact
         npc = mod(i+j,np_cols)
         np = global_id(npr,npc)
         ncnt_r(np) = ncnt_r(np) + 1
      enddo
   enddo

   ! Allocate receive buffer

   if(l_real) then
      allocate(r_rbuf(nblk,nblk,sum(ncnt_r)))
   else
      allocate(c_rbuf(nblk,nblk,sum(ncnt_r)))
   endif

   ! Set send counts/send offsets, receive counts/receive offsets
   ! now actually in variables, not in blocks

   ncnt_s(:) = ncnt_s(:)*nblk*nblk

   nstart_s(0) = 0
   do i=1,n_pes-1
      nstart_s(i) = nstart_s(i-1) + ncnt_s(i-1)
   enddo

   ncnt_r(:) = ncnt_r(:)*nblk*nblk

   nstart_r(0) = 0
   do i=1,n_pes-1
      nstart_r(i) = nstart_r(i-1) + ncnt_r(i-1)
   enddo

   ! Exchange all data with MPI_Alltoallv

   if(l_real) then
      call MPI_Alltoallv(r_sbuf,ncnt_s,nstart_s,MPI_REAL8,r_rbuf,ncnt_r,nstart_r,MPI_REAL8,mpi_comm,mpierr)
   else
      call MPI_Alltoallv(c_sbuf,ncnt_s,nstart_s,MPI_COMPLEX16,c_rbuf,ncnt_r,nstart_r,MPI_COMPLEX16,mpi_comm,mpierr)
   endif

   ! set band from receive buffer

   ncnt_r(:) = ncnt_r(:)/(nblk*nblk)

   nstart_r(0) = 0
   do i=1,n_pes-1
      nstart_r(i) = nstart_r(i-1) + ncnt_r(i-1)
   enddo

   if(l_real) then
      allocate(r_buf((nfact+1)*nblk,nblk))
   else
      allocate(c_buf((nfact+1)*nblk,nblk))
   endif

   ! n_off: Offset of ab within band
   n_off = block_limits(my_pe)*nbw

   do j=block_limits(my_pe)*nfact,min(block_limits(my_pe+1)*nfact-1,(na-1)/nblk)
      npr = mod(j,np_rows)
      do i=0,nfact
         npc = mod(i+j,np_cols)
         np = global_id(npr,npc)
         nstart_r(np) = nstart_r(np) + 1
         if(l_real) then
            r_buf(i*nblk+1:i*nblk+nblk,:) = transpose(r_rbuf(:,:,nstart_r(np)))
         else
            c_buf(i*nblk+1:i*nblk+nblk,:) = conjg(transpose(c_rbuf(:,:,nstart_r(np))))
         endif
      enddo
      do i=1,MIN(nblk,na-j*nblk)
         if(l_real) then
            r_ab(1:nbw+1,i+j*nblk-n_off) = r_buf(i:i+nbw,i)
         else
            c_ab(1:nbw+1,i+j*nblk-n_off) = c_buf(i:i+nbw,i)
         endif
      enddo
   enddo

   deallocate(ncnt_s, nstart_s)
   deallocate(ncnt_r, nstart_r)
   deallocate(global_id)
   deallocate(block_limits)
   if(l_real) then
      deallocate(r_sbuf, r_rbuf, r_buf)
   else
      deallocate(c_sbuf, c_rbuf, c_buf)
   endif

end subroutine

!---------------------------------------------------------------------------------------------------
! divide_band: sets the work distribution in band
! Proc n works on blocks block_limits(n)+1 .. block_limits(n+1)

subroutine divide_band(nblocks_total, n_pes, block_limits)

   integer, intent(in) :: nblocks_total ! total number of blocks in band
   integer, intent(in) :: n_pes         ! number of PEs for division
   integer, intent(out) :: block_limits(0:n_pes)

   integer :: n, nblocks, nblocks_left

   block_limits(0) = 0
   if(nblocks_total < n_pes) then
      ! Not enough work for all: The first tasks get exactly 1 block
      do n=1,n_pes
         block_limits(n) = min(nblocks_total,n)
      enddo
   else
      ! Enough work for all. If there is no exact loadbalance,
      ! the LAST tasks get more work since they are finishing earlier!
      nblocks = nblocks_total/n_pes
      nblocks_left = nblocks_total - n_pes*nblocks
      do n=1,n_pes
         if(n<=n_pes-nblocks_left) then
            block_limits(n) = block_limits(n-1) + nblocks
         else
            block_limits(n) = block_limits(n-1) + nblocks + 1
         endif
      enddo
   endif

end subroutine

!-------------------------------------------------------------------------------

subroutine band_band_real(na, nb, nb2, ab, ab2, d, e, mpi_comm)

!-------------------------------------------------------------------------------
! band_band_real:
! Reduces a real symmetric banded matrix to a real symmetric matrix with smaller bandwidth. Householder transformations are not stored.
! Matrix size na and original bandwidth nb have to be a multiple of the target bandwidth nb2. (Hint: expand your matrix with zero entries, if this 
! requirement doesn't hold)
!
!  na          Order of matrix
!
!  nb          Semi bandwidth of original matrix
!
!  nb2         Semi bandwidth of target matrix
!
!  ab          Input matrix with bandwidth nb. The leading dimension of the banded matrix has to be 2*nb. The parallel data layout 
!              has to be accordant to divide_band(), i.e. the matrix columns block_limits(n)*nb+1 to min(na, block_limits(n+1)*nb) 
!              are located on rank n.
!
!  ab2         Output matrix with bandwidth nb2. The leading dimension of the banded matrix is 2*nb2. The parallel data layout is
!              accordant to divide_band(), i.e. the matrix columns block_limits(n)*nb2+1 to min(na, block_limits(n+1)*nb2) are located
!              on rank n.
!
!  d(na)       Diagonal of tridiagonal matrix, set only on PE 0, set only if ab2 = 1 (output)
!
!  e(na)       Subdiagonal of tridiagonal matrix, set only on PE 0, set only if ab2 = 1 (output)
!
!  mpi_comm
!              MPI-Communicator for the total processor set
!-------------------------------------------------------------------------------

   implicit none

   integer, intent(in) ::  na, nb, nb2, mpi_comm
   real*8, intent(inout)  :: ab(2*nb,*)
   real*8, intent(inout)  :: ab2(2*nb2,*)
   real*8, intent(out) :: d(na), e(na) ! set only on PE 0

!----------------

   real*8 hv(nb,nb2), w(nb,nb2), w_new(nb,nb2), tau(nb2), hv_new(nb,nb2), tau_new(nb2), ab_s(1+nb,nb2), ab_r(1+nb,nb2), ab_s2(2*nb2,nb2), hv_s(nb,nb2)
   
   real*8 work(nb*nb2), work2(nb2*nb2)
   integer lwork, info
   
   integer istep, i, n, dest
   integer n_off, na_s
   integer my_pe, n_pes, mpierr
   integer nblocks_total, nblocks
   integer nblocks_total2, nblocks2
   integer ireq_ab, ireq_hv
   integer mpi_status(MPI_STATUS_SIZE)
   integer, allocatable :: mpi_statuses(:,:)
   integer, allocatable :: block_limits(:), block_limits2(:), ireq_ab2(:)
   
!----------------

   integer j, nc, nr, ns, ne, iblk
   
   call mpi_comm_rank(mpi_comm,my_pe,mpierr)
   call mpi_comm_size(mpi_comm,n_pes,mpierr)

   ! Total number of blocks in the band:
   nblocks_total = (na-1)/nb + 1
   nblocks_total2 = (na-1)/nb2 + 1
   
   ! Set work distribution
   allocate(block_limits(0:n_pes))
   call divide_band(nblocks_total, n_pes, block_limits)
   
   allocate(block_limits2(0:n_pes))
   call divide_band(nblocks_total2, n_pes, block_limits2)

   ! nblocks: the number of blocks for my task
   nblocks = block_limits(my_pe+1) - block_limits(my_pe)
   nblocks2 = block_limits2(my_pe+1) - block_limits2(my_pe)
   
   allocate(ireq_ab2(1:nblocks2))
   ireq_ab2 = MPI_REQUEST_NULL
   if(nb2>1) then
       do i=0,nblocks2-1
           call mpi_irecv(ab2(1,i*nb2+1),2*nb2*nb2,mpi_real8,0,3,mpi_comm,ireq_ab2(i+1),mpierr)
       enddo
   endif

   ! n_off: Offset of ab within band
   n_off = block_limits(my_pe)*nb
   lwork = nb*nb2
   dest = 0

   ireq_ab = MPI_REQUEST_NULL
   ireq_hv = MPI_REQUEST_NULL
   
   ! ---------------------------------------------------------------------------
   ! Start of calculations

   na_s = block_limits(my_pe)*nb + 1

   if(my_pe>0 .and. na_s<=na) then
      ! send first nb2 columns to previous PE
      ! Only the PE owning the diagonal does that (sending 1 element of the subdiagonal block also)
      do i=1,nb2
      	ab_s(1:nb+1,i) = ab(1:nb+1,na_s-n_off+i-1)
      enddo
      call mpi_isend(ab_s,(nb+1)*nb2,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
   endif
   
   do istep=1,na/nb2
   
      if(my_pe==0) then
      
         n = MIN(na-na_s-nb2+1,nb) ! number of rows to be reduced
         hv(:,:) = 0
         tau(:) = 0
         
         ! The last step (istep=na-1) is only needed for sending the last HH vectors.
         ! We don't want the sign of the last element flipped (analogous to the other sweeps)
         if(istep < na/nb2) then
            
            ! Transform first block column of remaining matrix
            call dgeqrf(n, nb2, ab(1+nb2,na_s-n_off), 2*nb-1, tau, work, lwork, info);
                        
            do i=1,nb2
            	hv(i,i) = 1.0
            	hv(i+1:n,i) = ab(1+nb2+1:1+nb2+n-i,na_s-n_off+i-1)
            	ab(1+nb2+1:2*nb,na_s-n_off+i-1) = 0
            enddo
            
         endif
         
         if(nb2==1) then
            d(istep) = ab(1,na_s-n_off)
	    e(istep) = ab(2,na_s-n_off)
	    if(istep == na) then
	    	e(na) = 0
            endif
         else
            ab_s2 = 0
            ab_s2(:,:) = ab(1:nb2+1,na_s-n_off:na_s-n_off+nb2-1)
            if(block_limits2(dest+1)<istep) then
            	dest = dest+1
            endif
            call mpi_send(ab_s2,2*nb2*nb2,mpi_real8,dest,3,mpi_comm,mpierr)
         endif
         
      else
         if(na>na_s+nb2-1) then
            ! Receive Householder vectors from previous task, from PE owning subdiagonal
            call mpi_recv(hv,nb*nb2,mpi_real8,my_pe-1,2,mpi_comm,mpi_status,mpierr)
            do i=1,nb2
	       	tau(i) = hv(i,i)
	       	hv(i,i) = 1.
            enddo
         endif
      endif
      
      na_s = na_s+nb2
      if(na_s-n_off > nb) then
         ab(:,1:nblocks*nb) = ab(:,nb+1:(nblocks+1)*nb)
         ab(:,nblocks*nb+1:(nblocks+1)*nb) = 0
         n_off = n_off + nb
      endif
      
      do iblk=1,nblocks
         ns = na_s + (iblk-1)*nb - n_off ! first column in block
         ne = ns+nb-nb2                    ! last column in block

         if(ns+n_off>na) exit
         
         nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
         nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                       ! Note that nr>=0 implies that diagonal block is full (nc==nb)!
                                       
         call wy_gen(nc,nb2,w,hv,tau,work,nb)

         if(iblk==nblocks .and. nc==nb) then
             !request last nb2 columns
             call mpi_recv(ab_r,(nb+1)*nb2,mpi_real8,my_pe+1,1,mpi_comm,mpi_status,mpierr)
             do i=1,nb2
	         ab(1:nb+1,ne+i-1) = ab_r(:,i)
             enddo
         endif
         
         hv_new(:,:) = 0 ! Needed, last rows must be 0 for nr < nb
         tau_new(:) = 0
         
         if(nr>0) then
             call wy_right(nr,nb,nb2,ab(nb+1,ns),2*nb-1,w,hv,work,nb)
             
             call dgeqrf(nr,nb2,ab(nb+1,ns),2*nb-1,tau_new,work,lwork,info);
                          
             do i=1,nb2
	     	 hv_new(i,i) = 1.0
	     	 hv_new(i+1:,i) = ab(nb+2:2*nb-i+1,ns+i-1)
	     	 ab(nb+2:,ns+i-1) = 0
	     enddo
	     
	     !send hh-vector
	     if(iblk==nblocks) then
	         call mpi_wait(ireq_hv,mpi_status,mpierr)
	         hv_s = hv_new
	         do i=1,nb2
		     hv_s(i,i) = tau_new(i)
                 enddo
	         call mpi_isend(hv_s,nb*nb2,mpi_real8,my_pe+1,2,mpi_comm,ireq_hv,mpierr)
             endif
             
         endif
         
	 call wy_symm(nc,nb2,ab(1,ns),2*nb-1,w,hv,work,work2,nb)
         
         if(my_pe>0 .and. iblk==1) then
	     !send first nb2 columns to previous PE
	     call mpi_wait(ireq_ab,mpi_status,mpierr)
	     do i=1,nb2
	         ab_s(1:nb+1,i) = ab(1:nb+1,ns+i-1)
	     enddo
	     call mpi_isend(ab_s,(nb+1)*nb2,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
         endif
         
         if(nr>0) then
             call wy_gen(nr,nb2,w_new,hv_new,tau_new,work,nb)
	     call wy_left(nb-nb2,nr,nb2,ab(nb+1-nb2,ns+nb2),2*nb-1,w_new,hv_new,work,nb)
         endif
         
         ! Use new HH vector for the next block
	 hv(:,:) = hv_new(:,:)
         tau = tau_new
         
     enddo

   enddo

   ! Finish the last outstanding requests
   call mpi_wait(ireq_ab,mpi_status,mpierr)
   call mpi_wait(ireq_hv,mpi_status,mpierr)
   allocate(mpi_statuses(MPI_STATUS_SIZE,nblocks2))
   call mpi_waitall(nblocks2,ireq_ab2,mpi_statuses,mpierr)
   deallocate(mpi_statuses)

   call mpi_barrier(mpi_comm,mpierr)

   deallocate(block_limits)
   deallocate(block_limits2)
   deallocate(ireq_ab2)

end subroutine

! --------------------------------------------------------------------------------------------------

subroutine wy_gen(n, nb, W, Y, tau, mem, lda)
    
    integer, intent(in) :: n		!length of householder-vectors
    integer, intent(in) :: nb		!number of householder-vectors
    integer, intent(in) :: lda		!leading dimension of Y and W
    real*8, intent(in) :: Y(lda,nb)	!matrix containing nb householder-vectors of length b
    real*8, intent(in) :: tau(nb)	!tau values
    real*8, intent(out) :: W(lda,nb)	!output matrix W
    real*8, intent(in) :: mem(nb)	!memory for a temporary matrix of size nb
    
    integer i
    
    W(1:n,1) = tau(1)*Y(1:n,1)
    do i=2,nb
        W(1:n,i) = tau(i)*Y(1:n,i)
        call DGEMV('T',n,i-1,1.d0,Y,lda,W(1,i),1,0.d0,mem,1)
	call DGEMV('N',n,i-1,-1.d0,W,lda,mem,1,1.d0,W(1,i),1)
    enddo
    
end subroutine

! --------------------------------------------------------------------------------------------------

subroutine wy_left(n, m, nb, A, lda, W, Y, mem, lda2)

    integer, intent(in) :: n		!width of the matrix A
    integer, intent(in) :: m		!length of matrix W and Y
    integer, intent(in) :: nb		!width of matrix W and Y
    integer, intent(in) :: lda		!leading dimension of A
    integer, intent(in) :: lda2		!leading dimension of W and Y
    real*8, intent(inout) :: A(lda,*)	!matrix to be transformed
    real*8, intent(in) :: W(m,nb)	!blocked transformation matrix W
    real*8, intent(in) :: Y(m,nb)	!blocked transformation matrix Y
    real*8, intent(inout) :: mem(n,nb)	!memory for a temporary matrix of size n x nb
    
    call DGEMM('T', 'N', nb, n, m, 1.d0, W, lda2, A, lda, 0.d0, mem, nb)
    call DGEMM('N', 'N', m, n, nb, -1.d0, Y, lda2, mem, nb, 1.d0, A, lda)

end subroutine

! --------------------------------------------------------------------------------------------------

subroutine wy_right(n, m, nb, A, lda, W, Y, mem, lda2)

    integer, intent(in) :: n		!height of the matrix A
    integer, intent(in) :: m		!length of matrix W and Y
    integer, intent(in) :: nb		!width of matrix W and Y
    integer, intent(in) :: lda		!leading dimension of A
    integer, intent(in) :: lda2		!leading dimension of W and Y
    real*8, intent(inout) :: A(lda,*)	!matrix to be transformed
    real*8, intent(in) :: W(m,nb)	!blocked transformation matrix W
    real*8, intent(in) :: Y(m,nb)	!blocked transformation matrix Y
    real*8, intent(inout) :: mem(n,nb)	!memory for a temporary matrix of size n x nb
    
    call DGEMM('N', 'N', n, nb, m, 1.d0, A, lda, W, lda2, 0.d0, mem, n)
    call DGEMM('N', 'T', n, m, nb, -1.d0, mem, n, Y, lda2, 1.d0, A, lda)

end subroutine

! --------------------------------------------------------------------------------------------------

subroutine wy_symm(n, nb, A, lda, W, Y, mem, mem2, lda2)

    integer, intent(in) :: n		!width/heigth of the matrix A; length of matrix W and Y
    integer, intent(in) :: nb		!width of matrix W and Y
    integer, intent(in) :: lda		!leading dimension of A
    integer, intent(in) :: lda2		!leading dimension of W and Y
    real*8, intent(inout) :: A(lda,*)	!matrix to be transformed
    real*8, intent(in) :: W(n,nb)	!blocked transformation matrix W
    real*8, intent(in) :: Y(n,nb)	!blocked transformation matrix Y
    real*8 :: mem(n,nb)			!memory for a temporary matrix of size n x nb
    real*8 :: mem2(nb,nb)		!memory for a temporary matrix of size nb x nb
    
    call DSYMM('L', 'L', n, nb, 1.d0, A, lda, W, lda2, 0.d0, mem, n)
    call DGEMM('T', 'N', nb, nb, n, 1.d0, mem, n, W, lda2, 0.d0, mem2, nb)
    call DGEMM('N', 'N', n, nb, nb, -0.5d0, Y, lda2, mem2, nb, 1.d0, mem, n)
    call DSYR2K('L', 'N', n, nb, -1.d0, Y, lda2, mem, n, 1.d0, A, lda)

end subroutine

! --------------------------------------------------------------------------------------------------

end module ELPA2

