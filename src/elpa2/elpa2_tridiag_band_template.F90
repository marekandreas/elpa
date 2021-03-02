#if 0
!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), formerly known as
!      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
!    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!      Informatik,
!    - Technische Universität München, Lehrstuhl für Informatik mit
!      Schwerpunkt Wissenschaftliches Rechnen ,
!    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!      and
!    - IBM Deutschland GmbH
!
!    This particular source code file contains additions, changes and
!    enhancements authored by Intel Corporation which is not part of
!    the ELPA consortium.
!
!    More information can be found here:
!    http://elpa.mpcdf.mpg.de/
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
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
#endif

#include "../general/sanity.F90"

subroutine tridiag_band_&
  &MATH_DATATYPE&
  &_&
  &PRECISION &
  (obj, na, nb, nblk, a_mat, lda, d, e, matrixCols, &
  hh_trans, mpi_comm_rows, mpi_comm_cols, communicator, useGPU, wantDebug, nrThreads)
  !-------------------------------------------------------------------------------
  ! tridiag_band_real/complex:
  ! Reduces a real symmetric band matrix to tridiagonal form
  !
  !  na          Order of matrix a
  !
  !  nb          Semi bandwith
  !
  !  nblk        blocksize of cyclic distribution, must be the same in both directions!
  !
  !  a_mat(lda,matrixCols)    Distributed system matrix reduced to banded form in the upper diagonal
  !
  !  lda         Leading dimension of a
  !  matrixCols  local columns of matrix a
  !
  ! hh_trans : housholder vectors
  !
  !  d(na)       Diagonal of tridiagonal matrix, set only on PE 0 (output)
  !
  !  e(na)       Subdiagonal of tridiagonal matrix, set only on PE 0 (output)
  !
  !  mpi_comm_rows
  !  mpi_comm_cols
  !              MPI-Communicators for rows/columns
  !  communicator
  !              MPI-Communicator for the total processor set
  !-------------------------------------------------------------------------------
  use elpa_abstract_impl
  use elpa2_workload
  use precision
  use, intrinsic :: iso_c_binding
  use redist
#ifdef WITH_OPENMP_TRADITIONAL
  use omp_lib
#endif
  use elpa_blas_interfaces
  use elpa_skewsymmetric_blas
  use elpa_gpu
  implicit none
#include "../general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout)   :: obj
  logical, intent(in)                          :: useGPU, wantDebug
  integer(kind=c_int)                          :: skewsymmetric
  logical                                      :: isSkewsymmetric
  integer(kind=ik), intent(in)                 :: na, nb, nblk, lda, matrixCols, mpi_comm_rows, mpi_comm_cols, communicator
#ifdef USE_ASSUMED_SIZE
  MATH_DATATYPE(kind=rck), intent(in)         :: a_mat(lda,*)
#else
  MATH_DATATYPE(kind=rck), intent(in)         :: a_mat(lda,matrixCols)
#endif
  real(kind=rk), intent(out)        :: d(na), e(na) ! set only on PE 0
  MATH_DATATYPE(kind=rck), intent(out), allocatable   :: hh_trans(:,:)

  real(kind=rk)                     :: vnorm2
  MATH_DATATYPE(kind=rck)                     :: hv(nb), tau, x, h(nb), ab_s(1+nb), hv_s(nb), hv_new(nb), tau_new, hf
  MATH_DATATYPE(kind=rck)                     :: hd(nb), hs(nb)

  integer(kind=ik)                             :: i, n, nc, nr, ns, ne, istep, iblk, nblocks_total, nblocks, nt
  integer(kind=ik)                             :: my_pe, n_pes
  integer(kind=ik)                             :: my_prow, np_rows, my_pcol, np_cols
  integer(kind=MPI_KIND)                       :: my_peMPI, n_pesMPI, mpierr
  integer(kind=MPI_KIND)                       :: my_prowMPI, np_rowsMPI, my_pcolMPI, np_colsMPI
  integer(kind=MPI_KIND)                       :: ireq_ab, ireq_hv
  integer(kind=ik)                             :: na_s, nx, num_hh_vecs, num_chunks, local_size, max_blk_size, n_off
  integer(kind=ik), intent(in)                 :: nrThreads
#ifdef WITH_OPENMP_TRADITIONAL
  integer(kind=ik)                             :: max_threads, my_thread, my_block_s, my_block_e, iter
#ifdef WITH_MPI
#endif
  integer(kind=ik), allocatable                :: global_id_tmp(:,:)
  integer(kind=ik), allocatable                :: omp_block_limits(:)
  MATH_DATATYPE(kind=rck), allocatable         :: hv_t(:,:), tau_t(:)
#endif /* WITH_OPENMP_TRADITIONAL */
  integer(kind=ik), allocatable                :: global_id(:,:), hh_cnt(:), hh_dst(:)
  integer(kind=MPI_KIND), allocatable          :: ireq_hhr(:), ireq_hhs(:)
  integer(kind=ik), allocatable                :: limits(:), snd_limits(:,:)
  integer(kind=ik), allocatable                :: block_limits(:)
  MATH_DATATYPE(kind=rck), allocatable         :: ab(:,:), hh_gath(:,:,:), hh_send(:,:,:)
  integer                                      :: istat
  integer(kind=ik)                             :: nblockEnd
  character(200)                               :: errorMessage
  character(20)                                :: gpuString

#ifndef WITH_MPI
  integer(kind=ik)                             :: startAddr
#endif
  logical                                      :: useIntelGPU

  call obj%get("is_skewsymmetric",skewsymmetric,istat)
  if (istat .ne. ELPA_OK) then
       print *,"Problem getting option for skewsymmetric settings. Aborting..."
       stop
  endif
  isSkewsymmetric = (skewsymmetric == 1)
  
  if(useGPU) then
    gpuString = "_gpu"
  else
    gpuString = ""
  endif

  useIntelGPU = .false.
  if (useGPU) then
    if (gpu_vendor() == INTEL_GPU) then
      useIntelGPU = .true.
    endif
  endif

  call obj%timer%start("tridiag_band_&
  &MATH_DATATYPE&
  &" // &
  &PRECISION_SUFFIX //&
  gpuString)

  if (wantDebug) call obj%timer%start("mpi_communication")
  call mpi_comm_rank(int(communicator,kind=MPI_KIND) ,my_peMPI ,mpierr)
  call mpi_comm_size(int(communicator,kind=MPI_KIND) ,n_pesMPI ,mpierr)

  call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND),my_prowMPI ,mpierr)
  call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND),np_rowsMPI ,mpierr)
  call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND),my_pcolMPI ,mpierr)
  call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND),np_colsMPI ,mpierr)

  my_pe = int(my_peMPI,kind=MPI_KIND)
  n_pes = int(n_pesMPI,kind=MPI_KIND)
  my_prow = int(my_prowMPI,kind=MPI_KIND)
  np_rows = int(np_rowsMPI,kind=MPI_KIND)
  my_pcol = int(my_pcolMPI,kind=MPI_KIND)
  np_cols = int(np_colsMPI,kind=MPI_KIND)
  if (wantDebug) call obj%timer%stop("mpi_communication")

  ! Get global_id mapping 2D procssor coordinates to global id

  allocate(global_id(0:np_rows-1,0:np_cols-1), stat=istat, errmsg=errorMessage)
  check_allocate("tridiag_band: global_id", istat, errorMessage)

  global_id(:,:) = 0
  global_id(my_prow, my_pcol) = my_pe

#ifdef WITH_OPENMP_TRADITIONAL
  allocate(global_id_tmp(0:np_rows-1,0:np_cols-1), stat=istat, errmsg=errorMessage)
  check_allocate("tridiag_band: global_id_tmp", istat, errorMessage)
#endif

#ifdef WITH_MPI
  if (wantDebug) call obj%timer%start("mpi_communication")
#ifndef WITH_OPENMP_TRADITIONAL
  call mpi_allreduce(mpi_in_place, global_id, int(np_rows*np_cols,kind=MPI_KIND), mpi_integer, &
                         mpi_sum, int(communicator,kind=MPI_KIND), mpierr)
#else
  global_id_tmp(:,:) = global_id(:,:)
  call mpi_allreduce(global_id_tmp, global_id, int(np_rows*np_cols,kind=MPI_KIND), mpi_integer, &
                     mpi_sum, int(communicator,kind=MPI_KIND), mpierr)
  deallocate(global_id_tmp, stat=istat, errmsg=errorMessage)
  check_deallocate("tridiag_band: global_id_tmp", istat, errorMessage)
#endif /* WITH_OPENMP_TRADITIONAL */
  if (wantDebug) call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */

  ! Total number of blocks in the band:

  nblocks_total = (na-1)/nb + 1

  ! Set work distribution

  allocate(block_limits(0:n_pes), stat=istat, errmsg=errorMessage)
  check_allocate("tridiag_band: block_limits", istat, errorMessage)

  call divide_band(obj,nblocks_total, n_pes, block_limits)

  ! nblocks: the number of blocks for my task
  nblocks = block_limits(my_pe+1) - block_limits(my_pe)

  ! allocate the part of the band matrix which is needed by this PE
  ! The size is 1 block larger than needed to avoid extensive shifts
  allocate(ab(2*nb,(nblocks+1)*nb), stat=istat, errmsg=errorMessage)
  check_allocate("tridiag_band: ab", istat, errorMessage)

  ab = 0.0_rck ! needed for lower half, the extra block should also be set to 0 for safety

  ! n_off: Offset of ab within band
  n_off = block_limits(my_pe)*nb

  ! Redistribute band in a to ab
  call redist_band_&
  &MATH_DATATYPE&
  &_&
  &PRECISION&
  &(obj,a_mat, lda, na, nblk, nb, matrixCols, mpi_comm_rows, mpi_comm_cols, communicator, ab)

  ! Calculate the workload for each sweep in the back transformation
  ! and the space requirements to hold the HH vectors

  allocate(limits(0:np_rows), stat=istat, errmsg=errorMessage)
  check_allocate("tridiag_band: limits", istat, errorMessage)

  call determine_workload(obj,na, nb, np_rows, limits)
  max_blk_size = maxval(limits(1:np_rows) - limits(0:np_rows-1))

  num_hh_vecs = 0
  num_chunks  = 0
  nx = na
  do n = 1, nblocks_total
    call determine_workload(obj, nx, nb, np_rows, limits)
    local_size = limits(my_prow+1) - limits(my_prow)
    ! add to number of householder vectors
    ! please note: for nx==1 the one and only HH Vector is 0 and is neither calculated nor send below!
    if (mod(n-1,np_cols) == my_pcol .and. local_size>0 .and. nx>1) then
      num_hh_vecs = num_hh_vecs + local_size
      num_chunks  = num_chunks+1
    endif
    nx = nx - nb
  enddo

  ! Allocate space for HH vectors

  allocate(hh_trans(nb,num_hh_vecs), stat=istat, errmsg=errorMessage)
  check_allocate("tridiag_band: hh_trans", istat, errorMessage)

  ! Allocate and init MPI requests

  allocate(ireq_hhr(num_chunks), stat=istat, errmsg=errorMessage) ! Recv requests
  check_allocate("tridiag_band: ireq_hhr", istat, errorMessage)
  allocate(ireq_hhs(nblocks), stat=istat, errmsg=errorMessage)    ! Send requests
  check_allocate("tridiag_band: ireq_hhs", istat, errorMessage)

  num_hh_vecs = 0
  num_chunks  = 0
  nx = na
  nt = 0
  nBlockEnd=1
  do n = 1, nblocks_total
    call determine_workload(obj,nx, nb, np_rows, limits)
    local_size = limits(my_prow+1) - limits(my_prow)
    if (mod(n-1,np_cols) == my_pcol .and. local_size>0 .and. nx>1) then
      num_chunks  = num_chunks+1
#ifdef WITH_MPI
      if (wantDebug) call obj%timer%start("mpi_communication")
      call mpi_irecv(hh_trans(1,num_hh_vecs+1), int(nb*local_size,kind=MPI_KIND),  MPI_MATH_DATATYPE_PRECISION_EXPL,     &
                    int(nt,kind=MPI_KIND), int(10+n-block_limits(nt),kind=MPI_KIND), &
                    int(communicator,kind=MPI_KIND), ireq_hhr(num_chunks), mpierr)
      if (wantDebug) call obj%timer%stop("mpi_communication")

#else /* WITH_MPI */
      ! carefull non-block recv data copy must be done at wait or send
      ! hh_trans(1:nb*local_size,num_hh_vecs+1) = hh_send(1:nb*hh_cnt(iblk),1,iblk)

#endif /* WITH_MPI */
      num_hh_vecs = num_hh_vecs + local_size
    endif
    nx = nx - nb
    if (n == block_limits(nt+1)) then
      nt = nt + 1
    endif
  enddo
#ifdef WITH_MPI
  ireq_hhs(:) = MPI_REQUEST_NULL
#endif
  ! Buffers for gathering/sending the HH vectors

  allocate(hh_gath(nb,max_blk_size,nblocks), stat=istat, errmsg=errorMessage) ! gathers HH vectors
  check_allocate("tridiag_band: hh_gath", istat, errorMessage)

  allocate(hh_send(nb,max_blk_size,nblocks), stat=istat, errmsg=errorMessage) ! send buffer for HH vectors
  check_allocate("tridiag_band: hh_send", istat, errorMessage)

  hh_gath(:,:,:) = 0.0_rck
  hh_send(:,:,:) = 0.0_rck

  ! Some counters

  allocate(hh_cnt(nblocks), stat=istat, errmsg=errorMessage)
  check_allocate("tridiag_band: hh_cnt", istat, errorMessage)

  allocate(hh_dst(nblocks), stat=istat, errmsg=errorMessage)
  check_allocate("tridiag_band: hh_dst", istat, errorMessage)

  hh_cnt(:) = 1 ! The first transfomation Vector is always 0 and not calculated at all
  hh_dst(:) = 0 ! PE number for receive
#ifdef WITH_MPI
  ireq_ab = MPI_REQUEST_NULL
  ireq_hv = MPI_REQUEST_NULL
#endif
  ! Limits for sending

  allocate(snd_limits(0:np_rows,nblocks), stat=istat, errmsg=errorMessage)
  check_allocate("tridiag_band: snd_limits", istat, errorMessage)

  do iblk=1,nblocks
    call determine_workload(obj, na-(iblk+block_limits(my_pe)-1)*nb, nb, np_rows, snd_limits(:,iblk))
  enddo

#ifdef WITH_OPENMP_TRADITIONAL
  ! OpenMP work distribution:
  max_threads = nrThreads
  ! For OpenMP we need at least 2 blocks for every thread
  max_threads = MIN(max_threads, nblocks/2)
  if (max_threads==0) max_threads = 1

  allocate(omp_block_limits(0:max_threads), stat=istat, errmsg=errorMessage)
  check_allocate("tridiag_band: omp_block_limits", istat, errorMessage)

  ! Get the OpenMP block limits
  call divide_band(obj,nblocks, max_threads, omp_block_limits)

  allocate(hv_t(nb,max_threads), tau_t(max_threads), stat=istat, errmsg=errorMessage)
  check_allocate("tridiag_band: hv_t", istat, errorMessage)

  hv_t = 0.0_rck
  tau_t = 0.0_rck
#endif /* WITH_OPENMP_TRADITIONAL */

  ! ---------------------------------------------------------------------------
  ! Start of calculations

  na_s = block_limits(my_pe)*nb + 1

  if (my_pe>0 .and. na_s<=na) then
    ! send first column to previous PE
    ! Only the PE owning the diagonal does that (sending 1 element of the subdiagonal block also)
    ab_s(1:nb+1) = ab(1:nb+1,na_s-n_off)
#ifdef WITH_MPI
    if (wantDebug) call obj%timer%start("mpi_communication")
    call mpi_isend(ab_s, int(nb+1,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                   int(my_pe-1,kind=MPI_KIND), 1_MPI_KIND, int(communicator,kind=MPI_KIND), ireq_ab, mpierr)
    if (wantDebug) call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */
  endif

#ifndef WITH_MPI
  startAddr = ubound(hh_trans,dim=2)
#endif /* WITH_MPI */

#ifdef WITH_OPENMP_TRADITIONAL
   do istep=1,na-nblockEnd-block_limits(my_pe)*nb
#else
   do istep=1,na-nblockEnd
#endif

     if (my_pe==0) then
       n = MIN(na-na_s,nb) ! number of rows to be reduced
       hv(:) = 0.0_rck
       hd(:) = 0.0_rck
       tau = 0.0_rck

       ! Transform first column of remaining matrix
#if REALCASE == 1
       ! The last step (istep=na-1) is only needed for sending the last HH vectors.
       ! We don't want the sign of the last element flipped (analogous to the other sweeps)
#endif
#if COMPLEXCASE == 1
       ! Opposed to the real case, the last step (istep=na-1) is needed here for making
       ! the last subdiagonal element a real number
#endif

#if REALCASE == 1
       if (istep < na-1) then
         ! Transform first column of remaining matrix
         vnorm2 = sum(ab(3:n+1,na_s-n_off)**2)
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
         vnorm2 = sum(real(ab(3:n+1,na_s-n_off),kind=rk8)**2+dimag(ab(3:n+1,na_s-n_off))**2)
#else
         vnorm2 = sum(real(ab(3:n+1,na_s-n_off),kind=rk4)**2+aimag(ab(3:n+1,na_s-n_off))**2)
#endif
         if (n<2) vnorm2 = 0.0_rk ! Safety only
#endif /* COMPLEXCASE */

          call hh_transform_&
               &MATH_DATATYPE&
               &_&
               &PRECISION &
                          (obj, ab(2,na_s-n_off), vnorm2, hf, tau, wantDebug)

          hv(1) = 1.0_rck
          hv(2:n) = ab(3:n+1,na_s-n_off)*hf
#if REALCASE == 1
       endif
#endif

#if REALCASE == 1
       if (isSkewsymmetric) then
         d(istep) = 0.0_rk
       else
         d(istep) = ab(1,na_s-n_off)
       endif
       e(istep) = ab(2,na_s-n_off)
#endif
#if COMPLEXCASE == 1
       d(istep) = real(ab(1,na_s-n_off), kind=rk)
       e(istep) = real(ab(2,na_s-n_off), kind=rk)
#endif

       if (istep == na-1) then
#if REALCASE == 1
         if (isSkewsymmetric) then
           d(na) = 0
         else
           d(na) = ab(1,na_s+1-n_off)
         endif
#endif

#if COMPLEXCASE == 1
         d(na) = real(ab(1,na_s+1-n_off),kind=rk)
#endif
         e(na) = 0.0_rck
       endif
     else
       if (na>na_s) then
         ! Receive Householder Vector from previous task, from PE owning subdiagonal

#ifdef WITH_OPENMP_TRADITIONAL

#ifdef WITH_MPI
         if (wantDebug) call obj%timer%start("mpi_communication")
         call mpi_recv(hv, int(nb,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                       int(my_pe-1,kind=MPI_KIND), 2_MPI_KIND, int(communicator,kind=MPI_KIND), &
                       MPI_STATUS_IGNORE, mpierr)
         if (wantDebug) call obj%timer%stop("mpi_communication")

#else /* WITH_MPI */

         hv(1:nb) = hv_s(1:nb)

#endif /* WITH_MPI */

#else /* WITH_OPENMP_TRADITIONAL */

#ifdef WITH_MPI
         if (wantDebug) call obj%timer%start("mpi_communication")

         call mpi_recv(hv, int(nb,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                       int(my_pe-1,kind=MPI_KIND), 2_MPI_KIND, int(communicator,kind=MPI_KIND), &
                       MPI_STATUS_IGNORE, mpierr)
         if (wantDebug) call obj%timer%stop("mpi_communication")

#else /* WITH_MPI */
         hv(1:nb) = hv_s(1:nb)
#endif /* WITH_MPI */

#endif /* WITH_OPENMP_TRADITIONAL */
         tau = hv(1)
         hv(1) = 1.0_rck
       endif
      endif

      na_s = na_s+1
      if (na_s-n_off > nb) then
        ab(:,1:nblocks*nb) = ab(:,nb+1:(nblocks+1)*nb)
        ab(:,nblocks*nb+1:(nblocks+1)*nb) = 0.0_rck
        n_off = n_off + nb
      endif

#ifdef WITH_OPENMP_TRADITIONAL
      if (max_threads > 1) then

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
          call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)

!$omp parallel do &
!$omp default(none) &
!$omp private(my_thread, my_block_s, my_block_e, iblk, ns, ne, hv, tau, &
!$omp&        nc, nr, hs, hd, vnorm2, hf, x, h, i) &
!$omp shared(max_threads, obj, ab, isSkewsymmetric, wantDebug, hh_gath, useIntelGPU, &
!$omp        hh_cnt, tau_t, hv_t, na, istep, n_off, na_s, nb, omp_block_limits, iter) &
!$omp         schedule(static,1), num_threads(max_threads)
          do my_thread = 1, max_threads

            if (iter == 1) then
              my_block_s = omp_block_limits(my_thread-1) + 1
              my_block_e = my_block_s
            else
              my_block_s = omp_block_limits(my_thread-1) + 2
              my_block_e = omp_block_limits(my_thread)
            endif

            do iblk = my_block_s, my_block_e

              ns = na_s + (iblk-1)*nb - n_off - my_thread + 1 ! first column in block
              ne = ns+nb-1                    ! last column in block

              if (istep<my_thread .or. ns+n_off>na) exit

              hv = hv_t(:,my_thread)
              tau = tau_t(my_thread)

              ! Store Householder Vector for back transformation

              hh_cnt(iblk) = hh_cnt(iblk) + 1

              hh_gath(1   ,hh_cnt(iblk),iblk) = tau
              hh_gath(2:nb,hh_cnt(iblk),iblk) = hv(2:nb)

              nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
              nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                        ! Note that nr>=0 implies that diagonal block is full (nc==nb)!

              ! Transform diagonal block
              if (useIntelGPU) then
                if (wantDebug) call obj%timer%start("mkl_offload")
#if REALCASE == 1
                if (isSkewsymmetric) then
                  hd(:) = 0.0_rk
                  call ELPA_PRECISION_SSMV(int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), hv, hd)
                else
                  call PRECISION_SYMV('L', int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), &
                                      hv, 1_BLAS_KIND, ZERO, hd, 1_BLAS_KIND)
                endif
#endif
#if COMPLEXCASE == 1
                call PRECISION_HEMV('L', int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), &
                                    hv, 1_BLAS_KIND, ZERO, hd, 1_BLAS_KIND)
#endif
                if (wantDebug) call obj%timer%stop("mkl_offload")
              else
                if (wantDebug) call obj%timer%start("blas")
#if REALCASE == 1
                if (isSkewsymmetric) then
                  hd(:) = 0.0_rk
                  call ELPA_PRECISION_SSMV(int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), hv, hd)
                else
                  call PRECISION_SYMV('L', int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), &
                                      hv, 1_BLAS_KIND, ZERO, hd, 1_BLAS_KIND)
                endif
#endif
#if COMPLEXCASE == 1
                call PRECISION_HEMV('L', int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), &
                                    hv, 1_BLAS_KIND, ZERO, hd, 1_BLAS_KIND)
#endif
                if (wantDebug) call obj%timer%stop("blas")
              endif

#if REALCASE == 1
              if (.NOT. isSkewsymmetric) then
                x = dot_product(hv(1:nc),hd(1:nc))*tau
              endif
#endif
#if COMPLEXCASE == 1
              x = dot_product(hv(1:nc),hd(1:nc))*conjg(tau)
#endif
              if (.NOT. isSkewsymmetric) then
                hd(1:nc) = hd(1:nc) - 0.5_rk*x*hv(1:nc)
              endif

              if (useIntelGPU) then
                if (wantDebug) call obj%timer%start("mkl_offload")
#if REALCASE == 1
                if (isSkewsymmetric) then
                  call ELPA_PRECISION_SSR2(int(nc,kind=BLAS_KIND), hd,  hv, ab(1,ns), &
                                           int(2*nb-1,kind=BLAS_KIND) )
                else
                  call PRECISION_SYR2('L', int(nc,kind=BLAS_KIND), -ONE, hd, 1_BLAS_KIND, &
                                    hv, 1_BLAS_KIND, ab(1,ns), int(2*nb-1,kind=BLAS_KIND))
                endif
#endif
#if COMPLEXCASE == 1
                call PRECISION_HER2('L', int(nc,kind=BLAS_KIND), -ONE, hd, 1_BLAS_KIND, &
                                  hv, 1_BLAS_KIND, ab(1,ns), int(2*nb-1,kind=BLAS_KIND))
#endif
                if (wantDebug) call obj%timer%stop("mkl_offload")
              else
                if (wantDebug) call obj%timer%start("blas")
#if REALCASE == 1
                if (isSkewsymmetric) then
                  call ELPA_PRECISION_SSR2(int(nc,kind=BLAS_KIND), hd,  hv, ab(1,ns), &
                                           int(2*nb-1,kind=BLAS_KIND) )
                else
                  call PRECISION_SYR2('L', int(nc,kind=BLAS_KIND), -ONE, hd, 1_BLAS_KIND, &
                                      hv, 1_BLAS_KIND, ab(1,ns), int(2*nb-1,kind=BLAS_KIND))
                endif
#endif
#if COMPLEXCASE == 1
                call PRECISION_HER2('L', int(nc,kind=BLAS_KIND), -ONE, hd, 1_BLAS_KIND, &
                                  hv, 1_BLAS_KIND, ab(1,ns), int(2*nb-1,kind=BLAS_KIND))
#endif
              
                if (wantDebug) call obj%timer%stop("blas")
              endif

              hv_t(:,my_thread) = 0.0_rck
              tau_t(my_thread)  = 0.0_rck
              if (nr<=0) cycle ! No subdiagonal block present any more

              ! Transform subdiagonal block
              if (useIntelGPU) then
                if (wantDebug) call obj%timer%start("mkl_offload")
                call PRECISION_GEMV('N', int(nr,kind=BLAS_KIND), int(nb,kind=BLAS_KIND), tau, &
                                    ab(nb+1,ns), int(2*nb-1,kind=BLAS_KIND), hv, 1_BLAS_KIND, &
                                    ZERO, hs, 1_BLAS_KIND)
                if (wantDebug) call obj%timer%stop("mkl_offload")
              else
                if (wantDebug) call obj%timer%start("blas")
                call PRECISION_GEMV('N', int(nr,kind=BLAS_KIND), int(nb,kind=BLAS_KIND), tau, &
                                    ab(nb+1,ns), int(2*nb-1,kind=BLAS_KIND), hv, 1_BLAS_KIND, &
                                    ZERO, hs, 1_BLAS_KIND)
                if (wantDebug) call obj%timer%stop("blas")
              endif

              if (nr>1) then

              ! complete (old) Householder transformation for first column

              ab(nb+1:nb+nr,ns) = ab(nb+1:nb+nr,ns) - hs(1:nr) ! Note: hv(1) == 1

              ! calculate new Householder transformation for first column
              ! (stored in hv_t(:,my_thread) and tau_t(my_thread))

#if REALCASE == 1
              vnorm2 = sum(ab(nb+2:nb+nr,ns)**2)
#endif
#if COMPLEXCASE == 1
#ifdef  DOUBLE_PRECISION_COMPLEX
              vnorm2 = sum(dble(ab(nb+2:nb+nr,ns))**2+dimag(ab(nb+2:nb+nr,ns))**2)
#else
              vnorm2 = sum(real(ab(nb+2:nb+nr,ns))**2+aimag(ab(nb+2:nb+nr,ns))**2)
#endif
#endif /* COMPLEXCASE */

              call hh_transform_&
              &MATH_DATATYPE&
              &_&
              &PRECISION &
              (obj, ab(nb+1,ns), vnorm2, hf, tau_t(my_thread), wantDebug)

              hv_t(1   ,my_thread) = 1.0_rck
              hv_t(2:nr,my_thread) = ab(nb+2:nb+nr,ns)*hf
              ab(nb+2:,ns) = 0.0_rck
              ! update subdiagonal block for old and new Householder transformation
              ! This way we can use a nonsymmetric rank 2 update which is (hopefully) faster
              if (useIntelGPU) then
                if (wantDebug) call obj%timer%start("mkl_offload")
                call PRECISION_GEMV(BLAS_TRANS_OR_CONJ,            &
                                    int(nr,kind=BLAS_KIND), int(nb-1,kind=BLAS_KIND), &
                                    tau_t(my_thread), ab(nb,ns+1), int(2*nb-1,kind=BLAS_KIND), &
                                    hv_t(1,my_thread), 1_BLAS_KIND, ZERO, h(2), 1_BLAS_KIND)
                if (wantDebug) call obj%timer%stop("mkl_offload")
              else
                if (wantDebug) call obj%timer%start("blas")
                call PRECISION_GEMV(BLAS_TRANS_OR_CONJ,            &
                                    int(nr,kind=BLAS_KIND), int(nb-1,kind=BLAS_KIND), &
                                    tau_t(my_thread), ab(nb,ns+1), int(2*nb-1,kind=BLAS_KIND), &
                                    hv_t(1,my_thread), 1_BLAS_KIND, ZERO, h(2), 1_BLAS_KIND)
                if (wantDebug) call obj%timer%stop("blas")
              endif

              x = dot_product(hs(1:nr),hv_t(1:nr,my_thread))*tau_t(my_thread)
              h(2:nb) = h(2:nb) - x*hv(2:nb)
              ! Unfortunately there is no BLAS routine like DSYR2 for a nonsymmetric rank 2 update ("DGER2")
              do i=2,nb
                ab(2+nb-i:1+nb+nr-i,i+ns-1) = ab(2+nb-i:1+nb+nr-i,i+ns-1) - hv_t(1:nr,my_thread)*  &
#if REALCASE == 1
                h(i) - hs(1:nr)*hv(i)
#endif
#if COMPLEXCASE == 1
                conjg(h(i)) - hs(1:nr)*conjg(hv(i))
#endif
              enddo

            else

              ! No new Householder transformation for nr=1, just complete the old one
              ab(nb+1,ns) = ab(nb+1,ns) - hs(1) ! Note: hv(1) == 1
              do i=2,nb
#if REALCASE == 1
                ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*hv(i)
#endif
#if COMPLEXCASE == 1
                ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*conjg(hv(i))
#endif
              enddo
              ! For safety: there is one remaining dummy transformation (but tau is 0 anyways)
              hv_t(1,my_thread) = 1.0_rck
            endif

          enddo

        enddo ! my_thread
        !$omp end parallel do

        call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)

        if (iter==1) then
          ! We are at the end of the first block

          ! Send our first column to previous PE
          if (my_pe>0 .and. na_s <= na) then
#ifdef WITH_MPI
            if (wantDebug) call obj%timer%start("mpi_communication")
            call mpi_wait(ireq_ab, MPI_STATUS_IGNORE, mpierr)
            if (wantDebug) call obj%timer%stop("mpi_communication")

#endif
            ab_s(1:nb+1) = ab(1:nb+1,na_s-n_off)
#ifdef WITH_MPI
            if (wantDebug) call obj%timer%start("mpi_communication")
            call mpi_isend(ab_s, int(nb+1,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                           int(my_pe-1,kind=MPI_KIND), 1_MPI_KIND, &
                           int(communicator,kind=MPI_KIND), ireq_ab, mpierr)
            if (wantDebug) call obj%timer%stop("mpi_communication")

#endif /* WITH_MPI */
          endif

          ! Request last column from next PE
          ne = na_s + nblocks*nb - (max_threads-1) - 1
#ifdef WITH_MPI
          if (wantDebug) call obj%timer%start("mpi_communication")

          if (istep>=max_threads .and. ne <= na) then
            call mpi_recv(ab(1,ne-n_off), int(nb+1,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL,  &
                          int(my_pe+1,kind=MPI_KIND), 1_MPI_KIND, int(communicator,kind=MPI_KIND), &
                          MPI_STATUS_IGNORE, mpierr)
          endif
          if (wantDebug) call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
          if (istep>=max_threads .and. ne <= na) then
            ab(1:nb+1,ne-n_off) = ab_s(1:nb+1)
          endif
#endif /* WITH_MPI */
        else
          ! We are at the end of all blocks

          ! Send last HH Vector and TAU to next PE if it has been calculated above
          ne = na_s + nblocks*nb - (max_threads-1) - 1
          if (istep>=max_threads .and. ne < na) then
#ifdef WITH_MPI
            if (wantDebug) call obj%timer%start("mpi_communication")
            call mpi_wait(ireq_hv, MPI_STATUS_IGNORE, mpierr)
            if (wantDebug) call obj%timer%stop("mpi_communication")
#endif
            hv_s(1) = tau_t(max_threads)
            hv_s(2:) = hv_t(2:,max_threads)

#ifdef WITH_MPI
            if (wantDebug) call obj%timer%start("mpi_communication")
            call mpi_isend(hv_s, int(nb,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                           int(my_pe+1,kind=MPI_KIND), 2_MPI_KIND, int(communicator,kind=MPI_KIND), &
                           ireq_hv, mpierr)
            if (wantDebug) call obj%timer%stop("mpi_communication")

#endif /* WITH_MPI */
          endif

          ! "Send" HH Vector and TAU to next OpenMP thread
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
      ! and sends the Householder Vector and the first column as early
      ! as possible.

#endif /* WITH_OPENMP_TRADITIONAL */

      do iblk=1,nblocks
        ns = na_s + (iblk-1)*nb - n_off ! first column in block
        ne = ns+nb-1                    ! last column in block

        if (ns+n_off>na) exit

        ! Store Householder Vector for back transformation

        hh_cnt(iblk) = hh_cnt(iblk) + 1

        hh_gath(1   ,hh_cnt(iblk),iblk) = tau
        hh_gath(2:nb,hh_cnt(iblk),iblk) = hv(2:nb)

#ifndef WITH_OPENMP_TRADITIONAL
        if (hh_cnt(iblk) == snd_limits(hh_dst(iblk)+1,iblk)-snd_limits(hh_dst(iblk),iblk)) then
          ! Wait for last transfer to finish
#ifdef WITH_MPI
          if (wantDebug) call obj%timer%start("mpi_communication")

          call mpi_wait(ireq_hhs(iblk), MPI_STATUS_IGNORE, mpierr)
          if (wantDebug) call obj%timer%stop("mpi_communication")
#endif
          ! Copy vectors into send buffer
          hh_send(:,1:hh_cnt(iblk),iblk) = hh_gath(:,1:hh_cnt(iblk),iblk)
          ! Send to destination

#ifdef WITH_MPI
          if (wantDebug) call obj%timer%start("mpi_communication")
          call mpi_isend(hh_send(1,1,iblk), int(nb*hh_cnt(iblk),kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                         global_id(hh_dst(iblk), mod(iblk+block_limits(my_pe)-1,np_cols)), &
                         int(10+iblk,kind=MPI_KIND), int(communicator,kind=MPI_KIND), ireq_hhs(iblk), mpierr)
          if (wantDebug) call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
          ! do the post-poned irecv here
          startAddr = startAddr - hh_cnt(iblk)
          hh_trans(1:nb,startAddr+1:startAddr+hh_cnt(iblk)) = hh_send(1:nb,1:hh_cnt(iblk),iblk)
#endif /* WITH_MPI */

          ! Reset counter and increase destination row
          hh_cnt(iblk) = 0
          hh_dst(iblk) = hh_dst(iblk)+1
        endif

        ! The following code is structured in a way to keep waiting times for
        ! other PEs at a minimum, especially if there is only one block.
        ! For this reason, it requests the last column as late as possible
        ! and sends the Householder Vector and the first column as early
        ! as possible.
#endif /* WITH_OPENMP_TRADITIONAL */
        nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
        nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                      ! Note that nr>=0 implies that diagonal block is full (nc==nb)!

        ! Multiply diagonal block and subdiagonal block with Householder Vector

        if (iblk==nblocks .and. nc==nb) then

          ! We need the last column from the next PE.
          ! First do the matrix multiplications without last column ...

          ! Diagonal block, the contribution of the last element is added below!
          ab(1,ne) = 0.0_rck

          if (useIntelGPU) then
            if (wantDebug) call obj%timer%start("mkl_offload")

#if REALCASE == 1
            if (isSkewsymmetric) then
              hd(:) = 0.0_rk
              call ELPA_PRECISION_SSMV(int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), hv, hd)
            else
              call PRECISION_SYMV('L', int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), &
                               hv, 1_BLAS_KIND, ZERO, hd, 1_BLAS_KIND)
            endif
#endif
#if COMPLEXCASE == 1
            call PRECISION_HEMV('L', int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), &
                               hv, 1_BLAS_KIND, ZERO, hd, 1_BLAS_KIND)
#endif
            ! Subdiagonal block
            if (nr>0) call PRECISION_GEMV('N', int(nr,kind=BLAS_KIND), int(nb-1,kind=BLAS_KIND), &
                                          tau, ab(nb+1,ns), int(2*nb-1,kind=BLAS_KIND), hv, 1_BLAS_KIND, &
                                          ZERO, hs, 1_BLAS_KIND)
            if (wantDebug) call obj%timer%stop("mkl_offload")

          else
            if (wantDebug) call obj%timer%start("blas")

#if REALCASE == 1
            if (isSkewsymmetric) then
              hd(:) = 0.0_rk
              call ELPA_PRECISION_SSMV(int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), hv, hd)
            else
              call PRECISION_SYMV('L', int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), &
                               hv, 1_BLAS_KIND, ZERO, hd, 1_BLAS_KIND)
            endif
#endif
#if COMPLEXCASE == 1
            call PRECISION_HEMV('L', int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), &
                             hv, 1_BLAS_KIND, ZERO, hd, 1_BLAS_KIND)
#endif
            ! Subdiagonal block
            if (nr>0) call PRECISION_GEMV('N', int(nr,kind=BLAS_KIND), int(nb-1,kind=BLAS_KIND), &
                                          tau, ab(nb+1,ns), int(2*nb-1,kind=BLAS_KIND), hv, 1_BLAS_KIND, &
                                          ZERO, hs, 1_BLAS_KIND)
            if (wantDebug) call obj%timer%stop("blas")
          endif

          ! ... then request last column ...
#ifdef WITH_MPI
          if (wantDebug) call obj%timer%start("mpi_communication")
#ifdef WITH_OPENMP_TRADITIONAL
          call mpi_recv(ab(1,ne), int(nb+1,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL,  &
                       int(my_pe+1,kind=MPI_KIND), 1_MPI_KIND, int(communicator,kind=MPI_KIND), &
                       MPI_STATUS_IGNORE, mpierr)
#else /* WITH_OPENMP_TRADITIONAL */
          call mpi_recv(ab(1,ne), int(nb+1,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL,  &
                       int(my_pe+1,kind=MPI_KIND), 1_MPI_KIND, int(communicator,kind=MPI_KIND), &
                       MPI_STATUS_IGNORE, mpierr)
#endif /* WITH_OPENMP_TRADITIONAL */
          if (wantDebug) call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */

          ab(1:nb+1,ne) = ab_s(1:nb+1)

#endif /* WITH_MPI */

          ! ... and complete the result
          hs(1:nr) = hs(1:nr) + ab(2:nr+1,ne)*tau*hv(nb)
          hd(nb) = hd(nb) + ab(1,ne)*hv(nb)*tau

        else

          ! Normal matrix multiply
          if (useIntelGPU) then
            if (wantDebug) call obj%timer%start("mkl_offload")
#if REALCASE == 1
            if (isSkewsymmetric) then
              hd(:) = 0.0_rk
              call ELPA_PRECISION_SSMV(int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), hv, hd)
            else
              call PRECISION_SYMV('L', int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), &
                                hv, 1_BLAS_KIND, ZERO, hd, 1_BLAS_KIND)
            endif
#endif
#if COMPLEXCASE == 1
            call PRECISION_HEMV('L', int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), &
                                hv, 1_BLAS_KIND, ZERO, hd, 1_BLAS_KIND)
#endif

#ifdef WITH_INTEL_GPU_VERSION
            if (nr>0) call mkl_offload_PRECISION_GEMV('N', int(nr,kind=BLAS_KIND), int(nb,kind=BLAS_KIND), tau, &
                            ab(nb+1,ns), &
                                        int(2*nb-1,kind=BLAS_KIND), hv, 1_BLAS_KIND, ZERO, hs, 1_BLAS_KIND)
#endif
            if (wantDebug) call obj%timer%stop("mkl_offload")
          else
            if (wantDebug) call obj%timer%start("blas")
#if REALCASE == 1
            if (isSkewsymmetric) then
              hd(:) = 0.0_rk
              call ELPA_PRECISION_SSMV(int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), hv, hd)
            else
              call PRECISION_SYMV('L', int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), &
                                hv, 1_BLAS_KIND, ZERO, hd, 1_BLAS_KIND)
            endif
#endif
#if COMPLEXCASE == 1
            call PRECISION_HEMV('L', int(nc,kind=BLAS_KIND), tau, ab(1,ns), int(2*nb-1,kind=BLAS_KIND), &
                               hv, 1_BLAS_KIND, ZERO, hd, 1_BLAS_KIND)
#endif
            if (nr>0) call PRECISION_GEMV('N', int(nr,kind=BLAS_KIND), int(nb,kind=BLAS_KIND), tau, ab(nb+1,ns), &
                                        int(2*nb-1,kind=BLAS_KIND), hv, 1_BLAS_KIND, ZERO, hs, 1_BLAS_KIND)
            if (wantDebug) call obj%timer%stop("blas")
          endif
        endif

        ! Calculate first column of subdiagonal block and calculate new
        ! Householder transformation for this column
        hv_new(:) = 0.0_rck ! Needed, last rows must be 0 for nr < nb
        tau_new = 0.0_rck
        if (nr>0) then

          ! complete (old) Householder transformation for first column

          ab(nb+1:nb+nr,ns) = ab(nb+1:nb+nr,ns) - hs(1:nr) ! Note: hv(1) == 1

          ! calculate new Householder transformation ...
          if (nr>1) then
#if  REALCASE == 1
            vnorm2 = sum(ab(nb+2:nb+nr,ns)**2)
#endif
#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
            vnorm2 = sum(real(ab(nb+2:nb+nr,ns),kind=rk8)**2+dimag(ab(nb+2:nb+nr,ns))**2)
#else
            vnorm2 = sum(real(ab(nb+2:nb+nr,ns),kind=rk4)**2+aimag(ab(nb+2:nb+nr,ns))**2)
#endif
#endif /* COMPLEXCASE */

            call hh_transform_&
                &MATH_DATATYPE&
                &_&
                &PRECISION &
               (obj, ab(nb+1,ns), vnorm2, hf, tau_new, wantDebug)
            hv_new(1) = 1.0_rck
            hv_new(2:nr) = ab(nb+2:nb+nr,ns)*hf
            ab(nb+2:,ns) = 0.0_rck
          endif ! nr > 1

          ! ... and send it away immediatly if this is the last block

          if (iblk==nblocks) then
#ifdef WITH_MPI
            if (wantDebug) call obj%timer%start("mpi_communication")
#ifdef WITH_OPENMP_TRADITIONAL
            call mpi_wait(ireq_hv,MPI_STATUS_IGNORE,mpierr)
#else
            call mpi_wait(ireq_hv,MPI_STATUS_IGNORE,mpierr)
#endif
            if (wantDebug) call obj%timer%stop("mpi_communication")

#endif /* WITH_MPI */
            hv_s(1) = tau_new
            hv_s(2:) = hv_new(2:)

#ifdef WITH_MPI
            if (wantDebug) call obj%timer%start("mpi_communication")
            call mpi_isend(hv_s, int(nb,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                           int(my_pe+1,kind=MPI_KIND), 2_MPI_KIND, int(communicator,kind=MPI_KIND), &
                           ireq_hv, mpierr)
            if (wantDebug) call obj%timer%stop("mpi_communication")

#endif /* WITH_MPI */
          endif

        endif

        ! Transform diagonal block
#if REALCASE == 1
        if (.NOT. isSkewsymmetric) then
          x = dot_product(hv(1:nc),hd(1:nc))*tau
        endif
#endif
#if COMPLEXCASE == 1
        x = dot_product(hv(1:nc),hd(1:nc))*conjg(tau)
#endif

#if REALCASE == 1
        if (.NOT. isSkewsymmetric) then
#endif
          hd(1:nc) = hd(1:nc) - 0.5_rk*x*hv(1:nc)
#if REALCASE == 1
        endif
#endif
        if (my_pe>0 .and. iblk==1) then

          ! The first column of the diagonal block has to be send to the previous PE
          ! Calculate first column only ...
#if REALCASE == 1
          if (isSkewsymmetric) then
            ab(1:nc,ns) = ab(1:nc,ns) - hd(1:nc)*hv(1) + hv(1:nc)*hd(1)
          else
            ab(1:nc,ns) = ab(1:nc,ns) - hd(1:nc)*hv(1) - hv(1:nc)*hd(1)
          endif
#endif
#if COMPLEXCASE == 1
          ab(1:nc,ns) = ab(1:nc,ns) - hd(1:nc)*conjg(hv(1)) - hv(1:nc)*conjg(hd(1))
#endif
          ! ... send it away ...
#ifdef WITH_MPI
          if (wantDebug) call obj%timer%start("mpi_communication")
          call mpi_wait(ireq_ab, MPI_STATUS_IGNORE, mpierr)
          if (wantDebug) call obj%timer%stop("mpi_communication")

#endif /* WITH_MPI */
          ab_s(1:nb+1) = ab(1:nb+1,ns)

#ifdef WITH_MPI
          if (wantDebug) call obj%timer%start("mpi_communication")

          call mpi_isend(ab_s, int(nb+1,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                         int(my_pe-1,kind=MPI_KIND), 1_MPI_KIND, int(communicator,kind=MPI_KIND), &
                         ireq_ab, mpierr)
          if (wantDebug) call obj%timer%stop("mpi_communication")

#endif /* WITH_MPI */
          ! ... and calculate remaining columns with rank-2 update
          if (useIntelGPU) then
            if (wantDebug) call obj%timer%start("mkl_offload")
#if REALCASE == 1
            if (isSkewsymmetric) then
              if (nc>1) call ELPA_PRECISION_SSR2(int(nc-1,kind=BLAS_KIND), hd(2), hv(2), ab(1,ns+1), int(2*nb-1,kind=BLAS_KIND) )
            else
              if (nc>1) call PRECISION_SYR2('L', int(nc-1,kind=BLAS_KIND), -ONE, hd(2), 1_BLAS_KIND, &
                                         hv(2), 1_BLAS_KIND, ab(1,ns+1), int(2*nb-1,kind=BLAS_KIND) )
            endif
#endif
#if COMPLEXCASE == 1
            if (nc>1) call PRECISION_HER2('L', int(nc-1,kind=BLAS_KIND), -ONE, hd(2), 1_BLAS_KIND, &
                                         hv(2), 1_BLAS_KIND, ab(1,ns+1), int(2*nb-1,kind=BLAS_KIND) )
#endif
            if (wantDebug) call obj%timer%stop("mkl_offload")
          else
            if (wantDebug) call obj%timer%start("blas")
#if REALCASE == 1
            if (isSkewsymmetric) then 
              if (nc>1) call ELPA_PRECISION_SSR2(int(nc-1,kind=BLAS_KIND), hd(2), hv(2), ab(1,ns+1), int(2*nb-1,kind=BLAS_KIND) )
            else
              if (nc>1) call PRECISION_SYR2('L', int(nc-1,kind=BLAS_KIND), -ONE, hd(2), 1_BLAS_KIND, &
                                         hv(2), 1_BLAS_KIND, ab(1,ns+1), int(2*nb-1,kind=BLAS_KIND) )
            endif
#endif
#if COMPLEXCASE == 1
            if (nc>1) call PRECISION_HER2('L', int(nc-1,kind=BLAS_KIND), -ONE, hd(2), 1_BLAS_KIND, &
                                       hv(2), 1_BLAS_KIND, ab(1,ns+1), int(2*nb-1,kind=BLAS_KIND) )
#endif
            if (wantDebug) call obj%timer%stop("blas")
          endif

        else
          ! No need to  send, just a rank-2 update
          if (useIntelGPU) then
            if (wantDebug) call obj%timer%start("mkl_offload")
#if REALCASE == 1
            if (isSkewsymmetric) then
              call ELPA_PRECISION_SSR2(int(nc,kind=BLAS_KIND), hd, hv, ab(1,ns), int(2*nb-1,kind=BLAS_KIND))
            else
              call PRECISION_SYR2('L', int(nc,kind=BLAS_KIND), -ONE, hd, 1_BLAS_KIND,  &
                                  hv, 1_BLAS_KIND, ab(1,ns), int(2*nb-1,kind=BLAS_KIND) )
            endif
#endif
#if COMPLEXCASE == 1
            call PRECISION_HER2('L', int(nc,kind=BLAS_KIND), -ONE, hd, 1_BLAS_KIND, hv, 1_BLAS_KIND, &
                                ab(1,ns), int(2*nb-1,kind=BLAS_KIND))
#endif
            if (wantDebug) call obj%timer%stop("mkl_offload")

          else
            if (wantDebug) call obj%timer%start("blas")
#if REALCASE == 1
            if (isSkewsymmetric) then 
              call ELPA_PRECISION_SSR2(int(nc,kind=BLAS_KIND), hd, hv, ab(1,ns), int(2*nb-1,kind=BLAS_KIND))
            else
              call PRECISION_SYR2('L', int(nc,kind=BLAS_KIND), -ONE, hd, 1_BLAS_KIND,  &
                                hv, 1_BLAS_KIND, ab(1,ns), int(2*nb-1,kind=BLAS_KIND) )
            endif
#endif
#if COMPLEXCASE == 1
            call PRECISION_HER2('L', int(nc,kind=BLAS_KIND), -ONE, hd, 1_BLAS_KIND, hv, 1_BLAS_KIND, &
                                ab(1,ns), int(2*nb-1,kind=BLAS_KIND))
#endif
            if (wantDebug) call obj%timer%stop("blas")
          endif
        endif

        ! Do the remaining double Householder transformation on the subdiagonal block cols 2 ... nb

        if (nr>0) then
          if (nr>1) then
            if (useIntelGPU) then
              if (wantDebug) call obj%timer%start("mkl_offload")
              call PRECISION_GEMV(BLAS_TRANS_OR_CONJ, int(nr,kind=BLAS_KIND), int(nb-1,kind=BLAS_KIND), &
                                  tau_new, ab(nb,ns+1), int(2*nb-1,kind=BLAS_KIND), &
                                  hv_new, 1_BLAS_KIND, ZERO, h(2), 1_BLAS_KIND)
#ifdef WITH_INTEL_GPU_VERSION
#if 0
              call mkl_offload_PRECISION_GEMV(BLAS_TRANS_OR_CONJ, int(nr,kind=BLAS_KIND), int(nb-1,kind=BLAS_KIND), &
                                  tau_new, ab(nb,ns+1), int(2*nb-1,kind=BLAS_KIND), &
                                  hv_new, 1_BLAS_KIND, ZERO, h(2), 1_BLAS_KIND)
#endif
#endif
              if (wantDebug) call obj%timer%stop("mkl_offload")
            else
              if (wantDebug) call obj%timer%start("blas")
              call PRECISION_GEMV(BLAS_TRANS_OR_CONJ, int(nr,kind=BLAS_KIND), int(nb-1,kind=BLAS_KIND), &
                                  tau_new, ab(nb,ns+1), int(2*nb-1,kind=BLAS_KIND), &
                                  hv_new, 1_BLAS_KIND, ZERO, h(2), 1_BLAS_KIND)
              if (wantDebug) call obj%timer%stop("blas")
            endif
            x = dot_product(hs(1:nr),hv_new(1:nr))*tau_new
            h(2:nb) = h(2:nb) - x*hv(2:nb)
            ! Unfortunately there is no BLAS routine like DSYR2 for a nonsymmetric rank 2 update
            do i=2,nb
#if REALCASE == 1
              ab(2+nb-i:1+nb+nr-i,i+ns-1) = ab(2+nb-i:1+nb+nr-i,i+ns-1) - hv_new(1:nr)*h(i) - hs(1:nr)*hv(i)
#endif
#if COMPLEXCASE == 1
              ab(2+nb-i:1+nb+nr-i,i+ns-1) = ab(2+nb-i:1+nb+nr-i,i+ns-1) - hv_new(1:nr)*conjg(h(i)) - hs(1:nr)*conjg(hv(i))
#endif
            enddo
          else
            ! No double Householder transformation for nr=1, just complete the row
            do i=2,nb
#if REALCASE == 1
              ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*hv(i)
#endif
#if COMPLEXCASE == 1
              ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*conjg(hv(i))
#endif
            enddo
          endif
        endif

        ! Use new HH Vector for the next block
        hv(:) = hv_new(:)
        tau = tau_new

      enddo

#ifdef WITH_OPENMP_TRADITIONAL
    endif
#endif

#ifdef WITH_OPENMP_TRADITIONAL
    do iblk = 1, nblocks

      if (hh_dst(iblk) >= np_rows) exit
      if (snd_limits(hh_dst(iblk)+1,iblk) == snd_limits(hh_dst(iblk),iblk)) exit

      if (hh_cnt(iblk) == snd_limits(hh_dst(iblk)+1,iblk)-snd_limits(hh_dst(iblk),iblk)) then
        ! Wait for last transfer to finish
#ifdef WITH_MPI
        if (wantDebug) call obj%timer%start("mpi_communication")
        call mpi_wait(ireq_hhs(iblk), MPI_STATUS_IGNORE, mpierr)
        if (wantDebug) call obj%timer%stop("mpi_communication")
#endif
        ! Copy vectors into send buffer
        hh_send(:,1:hh_cnt(iblk),iblk) = hh_gath(:,1:hh_cnt(iblk),iblk)
        ! Send to destination

#ifdef WITH_MPI
        if (wantDebug) call obj%timer%start("mpi_communication")
        call mpi_isend(hh_send(1,1,iblk), int(nb*hh_cnt(iblk),kind=MPI_KIND), &
                       MPI_MATH_DATATYPE_PRECISION_EXPL, &
                       global_id(hh_dst(iblk), mod(iblk+block_limits(my_pe)-1, np_cols)), &
                       int(10+iblk,kind=MPI_KIND), int(communicator,kind=MPI_KIND), ireq_hhs(iblk), mpierr)
        if (wantDebug) call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
        ! do the post-poned irecv here
        startAddr = startAddr - hh_cnt(iblk)
        hh_trans(1:nb,startAddr+1:startAddr+hh_cnt(iblk)) = hh_send(1:nb,1:hh_cnt(iblk),iblk)
#endif /* WITH_MPI */

        ! Reset counter and increase destination row
        hh_cnt(iblk) = 0
        hh_dst(iblk) = hh_dst(iblk)+1
      endif

    enddo
#endif /* WITH_OPENMP_TRADITIONAL */
  enddo ! istep

  ! Finish the last outstanding requests

#ifdef WITH_OPENMP_TRADITIONAL

#ifdef WITH_MPI
  if (wantDebug) call obj%timer%start("mpi_communication")
  call mpi_wait(ireq_ab,MPI_STATUS_IGNORE,mpierr)
  call mpi_wait(ireq_hv,MPI_STATUS_IGNORE,mpierr)

!  allocate(mpi_statuses(MPI_STATUS_SIZE,max(nblocks,num_chunks)), stat=istat, errmsg=errorMessage)
!  if (istat .ne. 0) then
!    print *,"tridiag_band_real: error when allocating mpi_statuses"//errorMessage
!    stop 1
!  endif

  call mpi_waitall(nblocks, ireq_hhs, MPI_STATUSES_IGNORE, mpierr)
  call mpi_waitall(num_chunks, ireq_hhr, MPI_STATUSES_IGNORE, mpierr)
! deallocate(mpi_statuses, stat=istat, errmsg=errorMessage)
! if (istat .ne. 0) then
!   print *,"tridiag_band_real: error when deallocating mpi_statuses"//errorMessage
!   stop 1
! endif
  if (wantDebug) call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */

#else /* WITH_OPENMP_TRADITIONAL */

#ifdef WITH_MPI
  if (wantDebug) call obj%timer%start("mpi_communication")
  call mpi_wait(ireq_ab,MPI_STATUS_IGNORE,mpierr)
  call mpi_wait(ireq_hv,MPI_STATUS_IGNORE,mpierr)

  call mpi_waitall(nblocks, ireq_hhs, MPI_STATUSES_IGNORE, mpierr)
  call mpi_waitall(num_chunks, ireq_hhr, MPI_STATUSES_IGNORE, mpierr)
  if (wantDebug) call obj%timer%stop("mpi_communication")
#endif

#endif /* WITH_OPENMP_TRADITIONAL */

#ifdef  WITH_MPI
  if (wantDebug) call obj%timer%start("mpi_communication")
  call mpi_barrier(int(communicator,kind=MPI_KIND),mpierr)
  if (wantDebug) call obj%timer%stop("mpi_communication")
#endif
  deallocate(ab, stat=istat, errmsg=errorMessage)
  check_deallocate("tridiag_band: ab", istat, errorMessage)

  deallocate(ireq_hhr, ireq_hhs, stat=istat, errmsg=errorMessage)
  check_deallocate("tridiag_band: ireq_hhr", istat, errorMessage)

  deallocate(hh_cnt, hh_dst, stat=istat, errmsg=errorMessage)
  check_deallocate("tridiag_band: hh_dst", istat, errorMessage)

  deallocate(hh_gath, hh_send, stat=istat, errmsg=errorMessage)
  check_deallocate("tridiag_band: hh_gath", istat, errorMessage)

  deallocate(limits, snd_limits, stat=istat, errmsg=errorMessage)
  check_deallocate("tridiag_band: limits", istat, errorMessage)

  deallocate(block_limits, stat=istat, errmsg=errorMessage)
  check_deallocate("tridiag_band: block_limits", istat, errorMessage)

  deallocate(global_id, stat=istat, errmsg=errorMessage)
  check_deallocate("tridiag_band: global_id", istat, errorMessage)

  call obj%timer%stop("tridiag_band_&
  &MATH_DATATYPE&
  &" // &
  &PRECISION_SUFFIX //&
  gpuString)

  ! intel compiler bug makes these ifdefs necessary
#if REALCASE == 1
end subroutine tridiag_band_real_&
#endif
#if COMPLEXCASE == 1
end subroutine tridiag_band_complex_&
#endif
&PRECISION

