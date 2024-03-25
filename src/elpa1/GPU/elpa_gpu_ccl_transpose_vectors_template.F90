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
! Author: Peter Karpov, MPCDF
#endif

#include "config-f90.h"
#include "../../general/sanity.F90"
#include "../../general/error_checking.inc"

#undef USE_CCL_TRANSPOSE
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
#define USE_CCL_TRANSPOSE
#endif

subroutine elpa_gpu_ccl_transpose_vectors_&
  &MATH_DATATYPE&
  &_&
  &PRECISION &
  (obj, vmat_s_dev, ld_s, ccl_comm_s, comm_s, vmat_t_dev, ld_t, ccl_comm_t, comm_t, &
    nvs, nvr, nvc, nblk, nrThreads, comm_s_isRows, myps, mypt, nps, npt, &
    aux_transpose_dev, isSkewsymmetric, isSquareGridGPU, wantDebug, my_stream, success)
  
  !-------------------------------------------------------------------------------
  ! This is the gpu version of the routine elpa_transpose_vectors
  ! This routine transposes an array of vectors which are distributed in
  ! communicator comm_s (ccl_comm_s) into its transposed form distributed in communicator comm_t (ccl_comm_t).
  ! There must be an identical copy of vmat_s_dev in every communicator comm_s.
  ! After this routine, there is an identical copy of vmat_t_dev in every communicator comm_t.
  !
  ! vmat_s_dev  original array of vectors
  ! ld_s        leading dimension of vmat_s_dev
  ! comm_s      communicator over which vmat_s is distributed
  ! vmat_t_dev  array of vectors in transposed form
  ! ld_t        leading dimension of vmat_t
  ! comm_t      communicator over which vmat_t_dev is distributed
  ! nvs         global index where to start in vmat_s_dev/vmat_t_dev
  !             Please note: this is kind of a hint, some values before nvs will be
  !             accessed in vmat_s_dev/put into vmat_t_dev
  ! nvr         global length of vmat_s_dev/vmat_t_dev
  ! nvc         number of columns in vmat_s_dev/vmat_t_dev
  ! nblk        block size of block cyclic distribution
  !
  !-------------------------------------------------------------------------------
  
  use precision
  use elpa_abstract_impl
  use elpa_mpi
  use elpa_gpu
  use tridiag_gpu
#ifdef WITH_NVIDIA_GPU_VERSION
  use cuda_functions ! for NVTX labels
#endif
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
  use elpa_ccl_gpu
#endif
  implicit none

  class(elpa_abstract_impl_t), intent(inout)        :: obj
  integer(kind=ik), intent(in)                      :: ld_s, comm_s, ld_t, comm_t, nvs, nvr, nvc, nblk, nrThreads
  integer(kind=ik), intent(in)                      :: myps, mypt, nps, npt

  integer(kind=MPI_KIND)                            :: mypsMPI, myptMPI, npsMPI, nptMPI
  integer(kind=ik)                                  :: n, lc, k, i, ips, ipt, ns, nl
  integer(kind=MPI_KIND)                            :: mpierr
  integer(kind=ik)                                  :: lcm_s_t, nblks_tot, nblks_comm, nblks_skip
  integer(kind=ik)                                  :: aux_stride, aux_size
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
  integer(kind=ik)                                  :: ld_st, solver, sign

  integer(kind=c_intptr_t)                          :: ccl_comm_s, ccl_comm_t
  integer(kind=c_intptr_t)                          :: ccl_comm_all
  integer(kind=c_intptr_t)                          :: vmat_s_dev 
  integer(kind=c_intptr_t)                          :: vmat_t_dev 
  integer(kind=c_intptr_t)                          :: aux_transpose_dev
  logical                                           :: successGPU
  integer(kind=c_intptr_t), parameter               :: size_of_datatype = size_of_&
                                                       &PRECISION&
                                                       &_&
                                                       &MATH_DATATYPE
  integer(kind=c_int)                               :: cclDataType
  integer(kind=ik)                                  :: k_datatype
  logical, intent(in)                               :: isSkewsymmetric, isSquareGridGPU, wantDebug
  integer(kind=c_intptr_t)                          :: my_stream
  integer(kind=c_int)                               :: sm_count
  if (wantDebug) call obj%timer%start("elpa_gpu_ccl_transpose_vectors")

  success = .true.

#if   REALCASE == 1 && DOUBLE_PRECISION == 1
  cclDataType = cclDouble
  k_datatype = 1
#elif REALCASE == 1 && SINGLE_PRECISION == 1
  cclDataType = cclFloat
  k_datatype = 1
#elif COMPLEXCASE == 1 && DOUBLE_PRECISION == 1
  cclDataType = cclDouble
  k_datatype = 2
#elif COMPLEXCASE == 1 && SINGLE_PRECISION == 1
  cclDataType = cclFloat
  k_datatype = 2
#endif

  ! ! TODO_23_11: check if moving this outside speeds up the subroutine
  ! ! TODO_23_11 -- move this outside (?) and change mpi to ccl 
  ! if (wantDebug) call obj%timer%start("mpi_communication")
  ! call mpi_comm_rank(int(comm_s,kind=MPI_KIND),mypsMPI, mpierr)
  ! call mpi_comm_size(int(comm_s,kind=MPI_KIND),npsMPI ,mpierr)
  ! call mpi_comm_rank(int(comm_t,kind=MPI_KIND),myptMPI, mpierr)
  ! call mpi_comm_size(int(comm_t,kind=MPI_KIND),nptMPI ,mpierr)
  ! myps = int(mypsMPI,kind=c_int)
  ! nps = int(npsMPI,kind=c_int)
  ! mypt = int(myptMPI,kind=c_int)
  ! npt = int(nptMPI,kind=c_int)
  ! if (wantDebug) call obj%timer%stop("mpi_communication")

  ! TODO_23_11
  ! this codepath doesn't work for ELPA2 (because ld_s != ld_t)
  ! call obj%get("solver", solver, error)
  ! special square grid codepath for ELPA1
  ! if (solver==ELPA_SOLVER_1STAGE .and. (.not. isSkewsymmetric) .and. & 
  !     nps==npt .and. nvs==1  .and. .not. (nvc>1 .and. ld_s /= ld_t)) then
  if (isSquareGridGPU) then
    call obj%get("mpi_comm_parent", mpi_comm_all, error)
    call mpi_comm_rank(int(mpi_comm_all,kind=MPI_KIND), my_mpi_rank, mpierr)
    ld_st = min(ld_s,ld_t)
    
    if (myps==mypt) then
      ! vmat_t(1:ld_st,1:nvc) = vmat_s(1:ld_st,1:nvc)
#ifdef WITH_NVTX
      call nvtxRangePush("memcpy new D-D vmat_s_dev->vmat_t_dev")
#endif
      successGPU = gpu_memcpy(vmat_t_dev, vmat_s_dev, (ld_st*nvc)* size_of_datatype, gpuMemcpyDeviceToDevice)
#ifdef WITH_NVTX
      call nvtxRangePop()
#endif
    else
      call obj%get("matrix_order", matrix_order, error)

      if (comm_s_isRows .and. matrix_order==COLUMN_MAJOR_ORDER .or. &
         (.not. comm_s_isRows) .and. matrix_order==ROW_MAJOR_ORDER) then
        ! my_mpi_rank = myps + mypt*nps
        transposed_mpi_rank = mypt + myps*npt ! can be generalized for arbitrary grid mappings using blacs_pnum(icontxt, prow, pcol)
      else if (comm_s_isRows .and. matrix_order==ROW_MAJOR_ORDER .or. &
        (.not. comm_s_isRows) .and. matrix_order==COLUMN_MAJOR_ORDER) then
        ! my_mpi_rank = mypt + myps*npt
        transposed_mpi_rank = myps + mypt*nps
      else
        print *, "Error in elpa_gpu_ccl_transpose_vectors: matrix_order is set incorrectly"
      endif
      
      message_size = ld_st*nvc

      if (wantDebug) call obj%timer%start("nccl_communication")

      ccl_comm_all = obj%gpu_setup%ccl_comm_all 
      successGPU = ccl_group_start()
      if (.not. successGPU) then 
        print *,"Error in setting up nccl_group_start!"
        success = .false.
        stop 1
      endif
      
      if (myps > mypt .and. message_size > 0) then

        successGPU = successGPU .and. ccl_Send(vmat_s_dev, int(k_datatype*message_size,kind=c_size_t), &
                                                cclDataType, transposed_mpi_rank, ccl_comm_all, my_stream)
        successGPU = successGPU .and. ccl_Recv(vmat_t_dev, int(k_datatype*message_size,kind=c_size_t), &
                                                cclDataType, transposed_mpi_rank, ccl_comm_all, my_stream)
      else if (myps < mypt .and. message_size > 0) then
        successGPU = successGPU .and. ccl_Recv(vmat_t_dev, int(k_datatype*message_size,kind=c_size_t), &
                                                cclDataType, transposed_mpi_rank, ccl_comm_all, my_stream)
        successGPU = successGPU .and. ccl_Send(vmat_s_dev, int(k_datatype*message_size,kind=c_size_t), &
                                                cclDataType, transposed_mpi_rank, ccl_comm_all, my_stream)
      endif

      if (.not. successGPU) then
        print *,"Error in nccl_Send/nccl_Recv!"
        success = .false.
        stop 1
      endif
      
      successGPU = ccl_group_end()
      if (.not. successGPU) then
        print *,"Error in setting up nccl_group_end!"
        success = .false.
        stop 1
      endif

      successGPU = gpu_stream_synchronize(my_stream)
      if (wantDebug) then
        check_stream_synchronize_gpu("nccl_Send/nccl_Recv vmat_s_dev/vmat_t_dev", successGPU)
        call obj%timer%stop("nccl_communication")
      endif
    endif

    if (wantDebug) call obj%timer%stop("elpa_gpu_ccl_transpose_vectors")
    return
  endif ! isSquareGridGPU

  ! The basic idea of this routine is that for every block (in the block cyclic
  ! distribution), the processor within comm_t which owns the diagonal
  ! broadcasts its values of vmat_s to all processors within comm_t.
  ! Of course this has not to be done for every block separately, since
  ! the communictation pattern repeats in the global matrix after
  ! the least common multiple of (nps,npt) blocks

  lcm_s_t   = least_common_multiple(nps,npt)

  nblks_tot = (nvr+nblk-1)/nblk ! number of blocks corresponding to nvr

  ! Get the number of blocks to be skipped at the begin.
  ! This must be a multiple of lcm_s_t (else it is getting complicated),
  ! thus some elements before nvs will be accessed/set.

  nblks_skip = ((nvs-1)/(nblk*lcm_s_t))*lcm_s_t

  ! allocate(aux( ((nblks_tot-nblks_skip+lcm_s_t-1)/lcm_s_t) * nblk * nvc ), stat=istat, errmsg=errorMessage)
  ! check_allocate("elpa_transpose_vectors: aux", istat, errorMessage)
  ! successGPU = gpu_malloc(aux_transpose_dev, ((nblks_tot-nblks_skip+lcm_s_t-1)/lcm_s_t) * nblk * nvc * size_of_datatype)
  ! check_alloc_gpu("tridiag: aux_transpose_dev", successGPU)

! #ifdef WITH_OPENMP_TRADITIONAL
!   !$omp parallel num_threads(nrThreads) &
!   !$omp default(none) &
!   !$omp private(lc, i, k, ns, nl, nblks_comm, aux_stride, ips, ipt, n, bcast_request1) &
!   !$omp shared(nps, npt, lcm_s_t, mypt, nblk, myps, vmat_t, mpierr, comm_s, &
!   !$omp&       obj, vmat_s, aux, nblks_skip, nblks_tot, nvc, nvr, &
! #ifdef WITH_MPI
!   !$omp&       MPI_STATUS_IGNORE, &
! #endif
!   !$omp&       useNonBlockingCollectives)
! #endif


  ! for non-square grid this becomes super inefficient
  ! what about 2x1 grid?
  ! we could circumvent the problem if it was possible to use nccl from inside the gpu kernel and then rework the whole procedure to one big kernel
  ! alternatively we could make aux_transpose_dev bigger by factor lcm_s_t and break main do-loop into 3 parts
  ! this would also help for CPU version. but maybe elpa_transpose_vectors is not a bottleneck there.
  ! !? work through the cycle once with a pen and paper (or rather do it in a txt file)

  do n = 0, lcm_s_t-1

    ips = mod(n,nps)
    ipt = mod(n,npt)

    if (mypt == ipt) then

      nblks_comm = (nblks_tot-nblks_skip-n+lcm_s_t-1)/lcm_s_t
      aux_stride = nblk * nblks_comm

      if (nblks_comm .ne. 0) then
        if (myps == ips) then
          !sm_count = 32
          sm_count = obj%gpu_setup%gpuSMcount
          call gpu_transpose_reduceadd_vectors_copy_block_PRECISION (aux_transpose_dev, vmat_s_dev, & 
                                                nvc, nvr, n, nblks_skip, nblks_tot, lcm_s_t, nblk, aux_stride, nps, ld_s, &
                                                1, isSkewsymmetric, .false., wantDebug, sm_count, my_stream)
        endif ! (myps == ips)

        ! call mpi_bcast(aux, int(nblks_comm*nblk*nvc,kind=MPI_KIND),  MPI_REAL_PRECISION,    &
        ! int(ips,kind=MPI_KIND), int(comm_s,kind=MPI_KIND),  mpierr)

        if (nps>1) then
          if (wantDebug) call obj%timer%start("nccl_communication")

          aux_size = aux_stride*nvc
          successGPU = ccl_Bcast(aux_transpose_dev, aux_transpose_dev, int(k_datatype*aux_size, kind=c_size_t), &
                                  cclDataType, int(ips, kind=c_int), ccl_comm_s, my_stream)

          if (.not. successGPU) then
            print *,"Error in nccl_Bcast"
            stop 1
          endif

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("nccl_Bcast aux_transpose_dev", successGPU)

          if (wantDebug) call obj%timer%stop("nccl_communication")
        endif ! (nps>1)
        !sm_count = 32
        sm_count = obj%gpu_setup%gpuSMcount
        call gpu_transpose_reduceadd_vectors_copy_block_PRECISION (aux_transpose_dev, vmat_t_dev, &
                                              nvc, nvr, n, nblks_skip, nblks_tot, lcm_s_t, nblk, aux_stride, npt, ld_t, & 
                                              2, isSkewsymmetric, .false., wantDebug, sm_count, my_stream)
      endif ! (nblks_comm .ne. 0)
    endif ! (mypt == ipt)

  enddo ! n = 0, lcm_s_t-1

  if (wantDebug) call obj%timer%stop("elpa_gpu_ccl_transpose_vectors")

end subroutine
