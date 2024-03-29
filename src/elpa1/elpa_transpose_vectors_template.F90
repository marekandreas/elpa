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
! Author: Andreas Marek, MPCDF
#endif

#include "config-f90.h"
#include "../general/sanity.F90"
#include "../general/error_checking.inc"

#undef ROUTINE_NAME
#ifdef SKEW_SYMMETRIC_BUILD
#define ROUTINE_NAME elpa_transpose_vectors_ss_
#else
#define ROUTINE_NAME elpa_transpose_vectors_
#endif


subroutine ROUTINE_NAME&
&MATH_DATATYPE&
&_&
&PRECISION &
(obj, vmat_s, ld_s, comm_s, vmat_t, ld_t, comm_t, & 
 nvs, nvr, nvc, nblk, nrThreads, comm_s_isRows, success)

!-------------------------------------------------------------------------------
! This routine transposes an array of vectors which are distributed in
! communicator comm_s into its transposed form distributed in communicator comm_t.
! There must be an identical copy of vmat_s in every communicator comm_s.
! After this routine, there is an identical copy of vmat_t in every communicator comm_t.
!
! vmat_s    original array of vectors
! ld_s      leading dimension of vmat_s
! comm_s    communicator over which vmat_s is distributed
! vmat_t    array of vectors in transposed form
! ld_t      leading dimension of vmat_t
! comm_t    communicator over which vmat_t is distributed
! nvs       global index where to start in vmat_s/vmat_t
!           Please note: this is kind of a hint, some values before nvs will be
!           accessed in vmat_s/put into vmat_t
! nvr       global length of vmat_s/vmat_t
! nvc       number of columns in vmat_s/vmat_t
! nblk      block size of block cyclic distribution
!
!-------------------------------------------------------------------------------
  use precision
  use elpa_abstract_impl
#ifdef WITH_OPENMP_TRADITIONAL
  use omp_lib
#endif
  use elpa_mpi

  implicit none
  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik), intent(in)                      :: ld_s, comm_s, ld_t, comm_t, nvs, nvr, nvc, nblk
  MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(in)   :: vmat_s(ld_s,nvc)
  MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(inout):: vmat_t(ld_t,nvc)

  MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable  :: aux(:)
  integer(kind=ik)                                  :: myps, mypt, nps, npt
  integer(kind=MPI_KIND)                            :: mypsMPI, myptMPI, npsMPI, nptMPI
  integer(kind=ik)                                  :: n, lc, k, i, ips, ipt, ns, nl
  integer(kind=MPI_KIND)                            :: mpierr
  integer(kind=ik)                                  :: lcm_s_t, nblks_tot, nblks_comm, nblks_skip
  integer(kind=ik)                                  :: aux_stride
  integer(kind=ik), intent(in)                      :: nrThreads
  integer(kind=ik)                                  :: istat
  character(200)                                    :: errorMessage

  integer(kind=MPI_KIND)                            :: bcast_request1
  logical                                           :: useNonBlockingCollectives
  logical                                           :: useNonBlockingCollectivesRows
  logical                                           :: useNonBlockingCollectivesCols
  logical, intent(in)                               :: comm_s_isRows
  integer(kind=c_int)                               :: non_blocking_collectives_rows, error, &
                                                       non_blocking_collectives_cols
  logical                                           :: success

  integer(kind=ik)                                  :: mpi_comm_all, my_mpi_rank, transposed_mpi_rank, message_size, matrix_order
  integer(kind=ik)                                  :: ld_st, solver

  success = .true.

  call obj%timer%start("&
          &ROUTINE_NAME&
  &MATH_DATATYPE&
  &" // &
  &PRECISION_SUFFIX &
  )

  call obj%get("nbc_row_transpose_vectors", non_blocking_collectives_rows, error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "Problem setting option for non blocking collectives for_rows in transpose_vectors. Aborting..."
    success = .false.
     call obj%timer%stop("&
          &ROUTINE_NAME&
     &MATH_DATATYPE&
     &" // &
     &PRECISION_SUFFIX &
    )
    return
  endif

  call obj%get("nbc_col_transpose_vectors", non_blocking_collectives_cols, error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "Problem setting option for non blocking collectives for_cols in transpose_vectors. Aborting..."
    success = .false.
     call obj%timer%stop("&
          &ROUTINE_NAME&
     &MATH_DATATYPE&
     &" // &
     &PRECISION_SUFFIX &
    )
    return
  endif

  if (non_blocking_collectives_rows .eq. 1) then
    useNonBlockingCollectivesRows = .true.
  else
    useNonBlockingCollectivesRows = .false.
  endif

  if (non_blocking_collectives_cols .eq. 1) then
    useNonBlockingCollectivesCols = .true.
  else
    useNonBlockingCollectivesCols = .false.
  endif

  if (comm_s_isRows) then
    useNonBlockingCollectives = useNonBlockingCollectivesRows
  else
    useNonBlockingCollectives = useNonBlockingCollectivesCols
  endif

  call obj%timer%start("mpi_communication")
  call mpi_comm_rank(int(comm_s,kind=MPI_KIND),mypsMPI, mpierr)
  call mpi_comm_size(int(comm_s,kind=MPI_KIND),npsMPI ,mpierr)
  call mpi_comm_rank(int(comm_t,kind=MPI_KIND),myptMPI, mpierr)
  call mpi_comm_size(int(comm_t,kind=MPI_KIND),nptMPI ,mpierr)
  myps = int(mypsMPI,kind=c_int)
  nps = int(npsMPI,kind=c_int)
  mypt = int(myptMPI,kind=c_int)
  npt = int(nptMPI,kind=c_int)
  call obj%timer%stop("mpi_communication")

  ! TODO_23_11
  ! this codepath doesn't work for ELPA2, Cholesky, maybe smth else. Fix it, along with optimization of Cholesky-GPU
  ! because there nvc>1 and ld_s != ld_t (so, we can't make a contigous-memory MPI_Send call)
#if 0
#if !defined(SKEW_SYMMETRIC_BUILD)
  call obj%get("solver", solver, error)
  if (solver==ELPA_SOLVER_1STAGE .and. nps==npt .and. nvs==1  .and. .not. (nvc>1 .and. ld_s /= ld_t)) then
    call obj%get("mpi_comm_parent", mpi_comm_all, error)
    call mpi_comm_rank(int(mpi_comm_all,kind=MPI_KIND), my_mpi_rank, mpierr)
    ld_st = min(ld_s,ld_t)
    
    if (myps==mypt) then
      vmat_t(1:ld_st,1:nvc) = vmat_s(1:ld_st,1:nvc)
    else
      call obj%get("matrix_order", matrix_order, error)

      if (comm_s_isRows .and. matrix_order==COLUMN_MAJOR_ORDER .or. &
         (.not. comm_s_isRows) .and. matrix_order==ROW_MAJOR_ORDER) then
        ! my_mpi_rank = myps + mypt*nps
        if (my_mpi_rank /= myps + mypt*nps) then ! TODO_23_11 - use blacs_pnum instead and delete the check after testing
          print *, "ERROR my_mpi_rank /= myps + mypt*nps"
        endif

        transposed_mpi_rank = mypt + myps*npt
      else if (comm_s_isRows .and. matrix_order==ROW_MAJOR_ORDER .or. &
        (.not. comm_s_isRows) .and. matrix_order==COLUMN_MAJOR_ORDER) then
        ! my_mpi_rank = mypt + myps*npt
        if (my_mpi_rank /= mypt + myps*npt) then ! TODO_23_11 - use blacs_pnum instead and delete the check after testing
          print *, "ERROR my_mpi_rank /= mypt + myps*npt"
        endif

        transposed_mpi_rank = myps + mypt*nps
      else
        print *, "ERROR: matrix_order not set correctly"
      endif
      
      ! TODO_23_11 - delete after implementing and testing
      ! print *, "my_mpi_rank=", my_mpi_rank, ", transposed_mpi_rank=", transposed_mpi_rank
      ! print *, "my_mpi_rank=", my_mpi_rank, ", nvc=", nvc, ", ld_s=", ld_s, ", ld_t=", ld_t

      message_size = ld_st*nvc

#ifdef WITH_MPI      
      if (myps>mypt .and. message_size>0) then
        call MPI_Send(vmat_s, int(message_size,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, & ! TODO_23_11: implement also a non-blocking version with MPI_Isend
                      int(transposed_mpi_rank,kind=MPI_KIND), 0, int(mpi_comm_all, kind=MPI_KIND), mpierr)
        call MPI_Recv(vmat_t, int(message_size,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                      int(transposed_mpi_rank,kind=MPI_KIND), 0, int(mpi_comm_all, kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
      else if (myps<mypt  .and. message_size>0) then
        call MPI_Recv(vmat_t, int(message_size,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                      int(transposed_mpi_rank,kind=MPI_KIND), 0, int(mpi_comm_all, kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
        call MPI_Send(vmat_s, int(message_size,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, & 
                      int(transposed_mpi_rank,kind=MPI_KIND), 0, int(mpi_comm_all, kind=MPI_KIND), mpierr)
      endif
#endif      
    endif

    call obj%timer%stop("&
        &ROUTINE_NAME&
        &MATH_DATATYPE&
        &" // &
        &PRECISION_SUFFIX &
        )
    return
  endif
#endif /* !defined(SKEW_SYMMETRIC_BUILD) */     
#endif /* 0 */

  ! The basic idea of this routine is that for every block (in the block cyclic
  ! distribution), the processor within comm_t which owns the diagonal
  ! broadcasts its values of vmat_s to all processors within comm_t.
  ! Of course this has not to be done for every block separately, since
  ! the communictation pattern repeats in the global matrix after
  ! the least common multiple of (nps,npt) blocks

  lcm_s_t   = least_common_multiple(nps,npt) ! least common multiple of nps, npt

  nblks_tot = (nvr+nblk-1)/nblk ! number of blocks corresponding to nvr

  ! Get the number of blocks to be skipped at the begin.
  ! This must be a multiple of lcm_s_t (else it is getting complicated),
  ! thus some elements before nvs will be accessed/set.

  nblks_skip = ((nvs-1)/(nblk*lcm_s_t))*lcm_s_t

  allocate(aux( ((nblks_tot-nblks_skip+lcm_s_t-1)/lcm_s_t) * nblk * nvc ), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_transpose_vectors: aux", istat, errorMessage)
#ifdef WITH_OPENMP_TRADITIONAL
  !$omp parallel num_threads(nrThreads) &
  !$omp default(none) &
  !$omp private(lc, i, k, ns, nl, nblks_comm, aux_stride, ips, ipt, n, bcast_request1) &
  !$omp shared(nps, npt, lcm_s_t, mypt, nblk, myps, vmat_t, mpierr, comm_s, &
  !$omp&       obj, vmat_s, aux, nblks_skip, nblks_tot, nvc, nvr, &
#ifdef WITH_MPI
  !$omp&       MPI_STATUS_IGNORE, &
#endif
  !$omp&       useNonBlockingCollectives)
#endif
  do n = 0, lcm_s_t-1

    ips = mod(n,nps)
    ipt = mod(n,npt)

    if (mypt == ipt) then ! (mypt == ipt)

      nblks_comm = (nblks_tot-nblks_skip-n+lcm_s_t-1)/lcm_s_t
      aux_stride = nblk * nblks_comm
!      if(nblks_comm==0) cycle
      if (nblks_comm .ne. 0) then
        if (myps == ips) then
!          k = 0
#ifdef WITH_OPENMP_TRADITIONAL
          !$omp do
#endif
          do lc=1,nvc
            do i = nblks_skip+n, nblks_tot-1, lcm_s_t
              k = (i - nblks_skip - n)/lcm_s_t * nblk + (lc - 1) * aux_stride
              ns = (i/nps)*nblk ! local start of block i
              nl = min(nvr-i*nblk,nblk) ! length
              aux(k+1:k+nl) = vmat_s(ns+1:ns+nl,lc)
!              k = k+nblk
            enddo
          enddo
        endif ! (myps == ips)

#ifdef WITH_OPENMP_TRADITIONAL
        !$omp barrier
        !$omp master
#endif

#ifdef WITH_MPI
        if (useNonBlockingCollectives) then
          call obj%timer%start("mpi_nbc_communication")
          call mpi_ibcast(aux, int(nblks_comm*nblk*nvc,kind=MPI_KIND),    &
#if REALCASE == 1
                      MPI_REAL_PRECISION,    &
#endif
#if COMPLEXCASE == 1
                      MPI_COMPLEX_PRECISION, &
#endif
                      int(ips,kind=MPI_KIND), int(comm_s,kind=MPI_KIND), bcast_request1, mpierr)
          call mpi_wait(bcast_request1, MPI_STATUS_IGNORE, mpierr)
          call obj%timer%stop("mpi_nbc_communication")
        else
          call obj%timer%start("mpi_communication")
          call mpi_bcast(aux, int(nblks_comm*nblk*nvc,kind=MPI_KIND),    &
#if REALCASE == 1
                      MPI_REAL_PRECISION,    &
#endif
#if COMPLEXCASE == 1
                      MPI_COMPLEX_PRECISION, &
#endif
                      int(ips,kind=MPI_KIND), int(comm_s,kind=MPI_KIND),  mpierr)
          call obj%timer%stop("mpi_communication")
        endif
#endif /* WITH_MPI */

#ifdef WITH_OPENMP_TRADITIONAL
        !$omp end master
        !$omp barrier

        !$omp do
#endif
!        k = 0
        do lc=1,nvc
          do i = nblks_skip+n, nblks_tot-1, lcm_s_t
            k = (i - nblks_skip - n)/lcm_s_t * nblk + (lc - 1) * aux_stride
            ns = (i/npt)*nblk ! local start of block i
            nl = min(nvr-i*nblk,nblk) ! length
#ifdef SKEW_SYMMETRIC_BUILD
            vmat_t(ns+1:ns+nl,lc) = - aux(k+1:k+nl)
#else
            vmat_t(ns+1:ns+nl,lc) = aux(k+1:k+nl)
#endif
!            k = k+nblk
          enddo
        enddo
      endif
    endif ! (mypt == ipt)

  enddo
#ifdef WITH_OPENMP_TRADITIONAL
  !$omp end parallel
#endif
  deallocate(aux, stat=istat, errmsg=errorMessage)
  check_deallocate("elpa_transpose_vectors: aux", istat, errorMessage)

  call obj%timer%stop("&
  &ROUTINE_NAME&
  &MATH_DATATYPE&
  &" // &
  &PRECISION_SUFFIX &
  )

end subroutine