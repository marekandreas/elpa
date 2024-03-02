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
! Author: Peter Karpov, MPCDF
#endif

#include "config-f90.h"
#include "../../general/sanity.F90"
#include "../../general/error_checking.inc"

#undef USE_CCL_REDUCE_ADD
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
#define USE_CCL_REDUCE_ADD
#endif

subroutine elpa_gpu_ccl_reduce_add_vectors_&
&MATH_DATATYPE&
&_&
&PRECISION &
  (obj, vmat_s_dev, ld_s, ccl_comm_s, comm_s, vmat_t_dev, ld_t, ccl_comm_t, comm_t, &
   nvr, nvc, nblk, nrThreads, aux1_reduceadd_dev, aux2_reduceadd_dev, wantDebug, my_stream, success)

!-------------------------------------------------------------------------------
! This is the gpu version of the routine elpa_reduce_add_vectors
! This routine does a reduce of all vectors in vmat_s_dev over the communicator comm_t (ccl_comm_t).
! The result of the reduce is gathered on the processes owning the diagonal
! and added to the array of vectors vmat_t_dev (which is distributed over comm_t (ccl_comm_t)).
!
! Opposed to elpa_transpose_vectors, there is NO identical copy of vmat_s_dev
! in the different members within vmat_t_dev (otherwise a reduce wouldn't be necessary).
! After this routine, an allreduce of vmat_t_dev has to be done.
!
! vmat_s_dev  array of vectors to be reduced and added
! ld_s        leading dimension of vmat_s_dev
! comm_s      communicator over which vmat_s_dev is distributed
! vmat_t_dev  array of vectors to which vmat_s_dev is added
! ld_t        leading dimension of vmat_t
! comm_t      communicator over which vmat_t is distributed
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
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
  use elpa_ccl_gpu
#endif
  implicit none

  class(elpa_abstract_impl_t), intent(inout)         :: obj
  integer(kind=ik), intent(in)                       :: ld_s, comm_s, ld_t, comm_t, nvr, nvc, nblk
  !MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(in)    :: vmat_s(ld_s,nvc)
  !MATH_DATATYPE(kind=C_DATATYPE_KIND), intent(inout) :: vmat_t(ld_t,nvc)

  !MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable   :: aux1(:), aux2(:)
  integer(kind=ik)                                   :: myps, mypt, nps, npt
  integer(kind=ik)                                   :: mpi_comm_all, my_mpi_rank
  integer(kind=MPI_KIND)                             :: mypsMPI, npsMPI, myptMPI, nptMPI
  integer(kind=ik)                                   :: n, lc, k, i, ips, ipt, ns, nl
  integer(kind=MPI_KIND)                             :: mpierr
  integer(kind=ik)                                   :: lcm_s_t, nblks_tot
  integer(kind=ik)                                   :: aux_stride, aux_size
  integer(kind=ik), intent(in)                       :: nrThreads
  integer(kind=ik)                                   :: istat
  character(200)                                     :: errorMessage
  logical                                            :: success

  logical                                            :: successGPU
  logical                                            :: isSkewsymmetric = .false.
  logical, intent(in)                                :: wantDebug
  integer(kind=c_intptr_t), parameter                :: size_of_datatype = size_of_&
                                                        &PRECISION&
                                                        &_&
                                                        &MATH_DATATYPE
  integer(kind=c_int)                                :: cclDataType
  integer(kind=ik)                                   :: k_datatype
  integer(kind=c_intptr_t)                           :: ccl_comm_s, ccl_comm_t
  integer(kind=c_intptr_t)                           :: vmat_s_dev, vmat_t_dev 
  integer(kind=c_intptr_t)                           :: aux1_reduceadd_dev, aux2_reduceadd_dev
  integer(kind=c_intptr_t)                           :: my_stream
  integer(kind=ik)                                   :: sm_count

  call obj%timer%start("elpa_gpu_ccl_reduce_add_vectors")
  
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

  call obj%timer%start("mpi_communication")
  call mpi_comm_rank(int(comm_s,kind=MPI_KIND), mypsMPI, mpierr)
  call mpi_comm_size(int(comm_s,kind=MPI_KIND), npsMPI,  mpierr)
  call mpi_comm_rank(int(comm_t,kind=MPI_KIND), myptMPI, mpierr)
  call mpi_comm_size(int(comm_t,kind=MPI_KIND), nptMPI ,mpierr)
  myps = int(mypsMPI,kind=c_int)
  nps = int(npsMPI,kind=c_int)
  mypt = int(myptMPI,kind=c_int)
  npt = int(nptMPI,kind=c_int)

  call obj%timer%stop("mpi_communication")

  ! TODO_23_11: TODO
  ! ! special square grid codepath for ELPA1
  ! if (solver==ELPA_SOLVER_1STAGE .and. nps==npt .and. nvs==1  .and. .not. (nvc>1 .and. ld_s /= ld_t)) then
  !   ...
  ! endif
  
  call obj%get("mpi_comm_parent", mpi_comm_all, mpierr)
  call mpi_comm_rank(int(mpi_comm_all,kind=MPI_KIND), my_mpi_rank, mpierr)


  ! Look to elpa_transpose_vectors for the basic idea!
  ! The difference is that here we have: nblks_skip = 0

  ! The communictation pattern repeats in the global matrix after
  ! the least common multiple of (nps,npt) blocks

  lcm_s_t   = least_common_multiple(nps,npt) ! least common multiple of nps, npt

  nblks_tot = (nvr+nblk-1)/nblk ! number of blocks corresponding to nvr
  
  if (((nblks_tot+lcm_s_t-1)/lcm_s_t) * nblk * nvc > 0) then

    ! TODO_23_11: delete after testing
    successGPU = gpu_stream_synchronize(my_stream) 
    check_stream_synchronize_gpu("check_stream_synchronize_gpu before gpu_memset", successGPU)

    ! TODO_23_11: delete after testing
! print *,"reduce1: my_mpi_rank=", my_mpi_rank,",((nblks_tot+lcm_s_t-1)/lcm_s_t)*nblk*nvc =",((nblks_tot+lcm_s_t-1)/lcm_s_t)*nblk*nvc
! print *,"reduce1 : my_mpi_rank=",my_mpi_rank,",nblks_tot=",nblks_tot,",lcm_s_t=",lcm_s_t,",nblk=",nblk,",nvc=",nvc

    successGPU = gpu_memset(aux1_reduceadd_dev, 0, (((nblks_tot+lcm_s_t-1)/lcm_s_t) * nblk * nvc ) * size_of_datatype)
    check_memcpy_gpu("elpa_gpu_ccl_reduce_add_vectors: aux1_reduceadd_dev", successGPU)

    successGPU = gpu_memset(aux2_reduceadd_dev, 0, (((nblks_tot+lcm_s_t-1)/lcm_s_t) * nblk * nvc ) * size_of_datatype)
    check_memcpy_gpu("elpa_gpu_ccl_reduce_add_vectors: aux2_reduceadd_dev", successGPU)
  endif

  ! print *,"reduce: my_mpi_rank=", my_mpi_rank, ", lcm_s_t=",lcm_s_t ! TODO_23_11: delete after testing

  do n = 0, lcm_s_t-1
    ips = mod(n,nps)
    ipt = mod(n,npt)

    aux_stride = nblk * ((nblks_tot - n + lcm_s_t - 1)/lcm_s_t)
    ! print *,"reduce: my_mpi_rank=", my_mpi_rank, ", aux_stride=",aux_stride ! TODO_23_11: delete after testing
    
    if (myps == ips) then

!       do lc=1,nvc
!         do i = n, nblks_tot-1, lcm_s_t
!           k = (i - n)/lcm_s_t * nblk + (lc - 1) * aux_stride
!           ns = (i/nps)*nblk ! local start of block i
!           nl = min(nvr-i*nblk,nblk) ! length
!           aux1(k+1:k+nl) = vmat_s(ns+1:ns+nl,lc)
! !          k = k+nblk
!         enddo
!       enddo
      !sm_count = 32
      sm_count = obj%gpu_setup%gpuSMcount
      call gpu_transpose_reduceadd_vectors_copy_block_PRECISION (aux1_reduceadd_dev, vmat_s_dev, & 
                                                nvc, nvr, n, 0, nblks_tot, lcm_s_t, nblk, aux_stride, nps, ld_s, &
                                                1, isSkewsymmetric, .true., wantDebug, sm_count, my_stream)

      aux_size = aux_stride * nvc


      ! TODO_23_11: change to a sinlge aux_  here (and to to MPI_IN_PLACE in the correspoding non-gpu routine)
      if (aux_size>0) then
        if (wantDebug) call obj%timer%start("nccl_communication")
        
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("nccl_Reduce aux1_reduceadd_dev, aux2_reduceadd_dev", successGPU)

        successGPU = ccl_group_start()
        if (.not. successGPU) then 
          print *,"Error in setting up nccl_group_start!" 
          stop 1
        endif

        successGPU = ccl_Reduce(aux1_reduceadd_dev, aux2_reduceadd_dev, int(k_datatype*aux_size, kind=c_size_t), &
                                 cclDataType, cclSum, int(ipt,kind=c_int), ccl_comm_t, my_stream)
        
        if (.not. successGPU) then
          print *, "Error in nccl_Reduce"
          stop 1
        endif

        successGPU = ccl_group_end()
        if (.not. successGPU) then
          print *,"Error in setting up nccl_group_end!"
          stop 1
        endif

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("nccl_Reduce aux1_reduceadd_dev, aux2_reduceadd_dev", successGPU)

        if (wantDebug) call obj%timer%stop("nccl_communication")
      endif ! if (aux_size>0)

      
      if (mypt == ipt) then
              !sm_count = 32
#if REALCASE == 1
          sm_count = obj%gpu_setup%gpuSMcount
          call gpu_transpose_reduceadd_vectors_copy_block_PRECISION (aux2_reduceadd_dev, vmat_t_dev, & 
                                                nvc, nvr, n, 0, nblks_tot, lcm_s_t, nblk, aux_stride, nps, ld_t, &
                                                2, isSkewsymmetric, .true., wantDebug, sm_count, my_stream)
#endif
      endif ! if (mypt == ipt)

    endif ! if (myps == ips)

  enddo


  call obj%timer%stop("elpa_gpu_ccl_reduce_add_vectors")
end subroutine


