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
#include "../general/error_checking.inc"


! transposes a row of nblk-blocks of a matrix, along with the LCM-copies of the row
! there are two versions: 1) with CCL; 2) without CCL (CPU or GPU+MPI)
subroutine elpa_transpose_row&
                            &CCL&
                            &MATH_DATATYPE&
                            &_&
                            &PRECISION &
!                            (obj, row_col_char, &
                            (obj, &
#ifdef DEVICE_POINTER
                              a_dev, at_col_dev, buf_send_dev, buf_recv_dev, buf_self_dev, &
#else /* DEVICE_POINTER */
                              a, at_col, buf_send, buf_recv, &
#endif /* DEVICE_POINTER */
                              np_fine, l_rows, l_cols, nblk_mult_rows_max, nblk_mult_cols_max)

  use, intrinsic :: iso_c_binding
  use precision
  use elpa_mpi
  use elpa_abstract_impl
  use elpa_utilities, only : least_common_multiple, check_memcpy_gpu_f
  use elpa_gpu
#if defined(USE_CCL_PXGEMM)
  use elpa_ccl_gpu
  use multiply_a_b_gpu
#endif
#ifdef WITH_NVIDIA_GPU_VERSION
  use cuda_functions ! for NVTX labels
#endif
  implicit none

#include "../../src/general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout)   :: obj
  !character(len=1)                             :: row_col_char
  integer(kind=ik), intent(in)                 :: np_fine, l_rows, l_cols, nblk_mult_rows_max, nblk_mult_cols_max
#ifdef DEVICE_POINTER
  MATH_DATATYPE(kind=rck), allocatable         :: a(:,:), at_col(:,:), buf_send(:,:), buf_recv(:,:) ! dummy variables
  integer(kind=c_intptr_t)                     :: a_dev, at_col_dev, buf_send_dev, buf_recv_dev, buf_self_dev
#else /* DEVICE_POINTER */
  MATH_DATATYPE(kind=rck)                      :: a(l_rows,l_cols), at_col(l_rows,l_cols), &
                                                  buf_send(nblk_mult_rows_max, nblk_mult_cols_max), & 
                                                  buf_recv(nblk_mult_rows_max, nblk_mult_cols_max)
  integer(kind=c_intptr_t)                     :: a_dev, at_col_dev, buf_send_dev, buf_recv_dev, buf_self_dev
#endif /* DEVICE_POINTER */

  integer(kind=ik)                             :: nblk, debug

  ! MPI-related
  integer(kind=MPI_KIND)                       :: mpierr
  integer(kind=ik)                             :: my_prow, my_pcol, np_rows, np_cols, myid
  integer(kind=ik)                             :: mpi_comm_rows, mpi_comm_cols, mpi_comm_all
  integer(kind=ik)                             :: mpi_comm_dirs ! PETERDEBUG_NEW
  integer(kind=ik)                             :: matrix_order ! PETERDEBUG_NEW --> needed only for transpose case
  integer(kind=ik)                             :: my_mpi_rank, mpi_rank_target, my_prow_target, my_pcol_target, &
                                                  mpi_rank_source, my_prow_source, my_pcol_source, & ! PETERDEBUG_NEW --> needed only for transpose
                                                  my_prow_target_deadlock

  integer(kind=ik)                             :: np_rows_fine, np_cols_fine, np_dirs_fine, np_t_fine, np_bc_fine, &
                                                  np_ab_fine, np_ab_t_fine, dnp_ab, dnp_ab_t, & ! PETERDEBUG_NEW
                                                  np_bc_fine_1, nblk_mult_cols_1, np_fine_1, nblk_mult_rows_1, & ! PETERDEBUG_NEW
                                                  m_blocks_loc_fine, n_blocks_loc_fine, m_blocks_loc_fine_1, n_blocks_loc_fine_1, &
                                                  dnp_bc_fine, np_bc_fine_1_start
  integer(kind=ik)                             :: LCM, nblk_mult_rows, nblk_mult_cols, i_block_loc_fine, j_block_loc_fine, & ! PETERDEBUG_NEW
                                                  i_block_loc, j_block_loc
  integer(kind=ik)                             :: nblk_cut_row, nblk_cut_col
  integer(kind=ik)                             :: error
  integer(kind=c_intptr_t), parameter          :: size_of_datatype = size_of_&
                                                  &PRECISION&
                                                  &_&
                                                  &MATH_DATATYPE

  ! GPU-related
  logical                                      :: successGPU, useCCL
  integer(kind=c_intptr_t)                     :: my_stream
  integer(kind=ik)                             :: SM_count
#if defined(USE_CCL_PXGEMM)
  integer(kind=c_intptr_t)                     :: ccl_comm_rows, ccl_comm_cols, ccl_comm_all, ccl_comm_dirs
  integer(kind=c_int)                          :: cclDataType
  integer(kind=ik)                             :: k_datatype
#endif

  call obj%timer%start("elpa_transpose_row")

  !   success = .true.
  useCCL = .false.
  np_bc_fine = np_fine
  
  call obj%get("matrix_order", matrix_order, error)
  if (error .ne. ELPA_OK) then
    print *, "elpa_multiply_helpers: Problem getting option matrix_order. Aborting..."
    stop 1
  endif

  !na      = obj%na
  nblk    = obj%nblk
  !lda     = obj%local_nrows
  !ldaCols = obj%local_ncols

  mpi_comm_all    = obj%mpi_setup%mpi_comm_parent
  mpi_comm_cols   = obj%mpi_setup%mpi_comm_cols
  mpi_comm_rows   = obj%mpi_setup%mpi_comm_rows

  myid    = obj%mpi_setup%myRank_comm_parent
  my_prow = obj%mpi_setup%myRank_comm_rows
  my_pcol = obj%mpi_setup%myRank_comm_cols

  np_rows = obj%mpi_setup%nRanks_comm_rows
  np_cols = obj%mpi_setup%nRanks_comm_cols

  SM_count = obj%gpu_setup%gpuSMcount

  LCM = least_common_multiple(np_rows, np_cols)*nblk
  np_rows_fine = least_common_multiple(np_rows, np_cols)
  np_cols_fine = np_rows_fine

#if defined(USE_CCL_PXGEMM)
  useCCL = .true.

  my_stream = obj%gpu_setup%my_stream
  ccl_comm_rows = obj%gpu_setup%ccl_comm_rows
  ccl_comm_cols = obj%gpu_setup%ccl_comm_cols
  ccl_comm_all  = obj%gpu_setup%ccl_comm_all

#if   REALCASE == 1 && defined(DOUBLE_PRECISION)
  cclDataType = cclDouble
  k_datatype = 1
#elif REALCASE == 1 && defined(SINGLE_PRECISION)
  cclDataType = cclFloat
  k_datatype = 1
#elif COMPLEXCASE == 1 && defined(DOUBLE_PRECISION)
  cclDataType = cclDouble
  k_datatype = 2
#elif COMPLEXCASE == 1 && defined(SINGLE_PRECISION)
  cclDataType = cclFloat
  k_datatype = 2
#endif
#endif /* defined(USE_CCL_PXGEMM) */

  call find_nblk_mult_dirs(l_rows, nblk, np_rows, np_fine   , LCM, nblk_mult_rows)
  call find_nblk_mult_dirs(l_cols, nblk, np_cols, np_bc_fine, LCM, nblk_mult_cols)

  ! if (a_transposed) then
  !   my_pdir = my_prow
  !   my_pdir_t = my_pcol
  !   np_dirs = np_rows
  !   np_dirs_t = np_cols
  !   mpi_comm_dirs = mpi_comm_rows
  ! else if (b_transposed) then
  !   my_pdir = my_pcol
  !   my_pdir_t = my_prow
  !   np_dirs = np_cols
  !   np_dirs_t = np_rows
  !   mpi_comm_dirs = mpi_comm_cols
  ! endif

#ifdef WITH_NVTX
  call nvtxRangePush("transpose row")
#endif
  ! a -> at_col: transpose block-row #np_fine of a
  ! Send
  if (mod(np_fine,np_rows) == my_prow) then

    my_pcol_target = mod(np_bc_fine, np_cols)
    
    ! we send to the process (mod(np_fine,np_rows), mod(np_bc_fine,np_cols)) in last turn
    ! to avoid the deadlock
    my_prow_target_deadlock = mod(np_fine,np_rows)
    
    np_bc_fine_1 = my_pcol
    np_bc_fine_1_start = mod(np_bc_fine_1, np_cols_fine)
    ! dry run: to find, whether there is a potential deadlock
    do np_bc_fine_1 = my_pcol, np_cols_fine-1, np_cols
      np_fine_1 = np_bc_fine_1
      my_prow_target = mod(np_fine_1, np_rows)
      if (my_prow_target==my_prow_target_deadlock) then
        np_bc_fine_1_start = mod(np_bc_fine_1+np_cols, np_cols_fine)
        exit
      endif
    enddo
    
    np_bc_fine_1 = np_bc_fine_1_start
    do ! np_bc_fine_1 periodic loop
      np_fine_1 = np_bc_fine_1
      my_prow_target = mod(np_fine_1, np_rows)

      if (matrix_order==COLUMN_MAJOR_ORDER) then
        mpi_rank_target = my_prow_target + np_rows*my_pcol_target
      else
        mpi_rank_target = my_pcol_target + np_cols*my_prow_target
      endif
      
      call find_nblk_mult_dirs(l_cols, nblk, np_cols, np_bc_fine_1, LCM, nblk_mult_cols_1)

      n_blocks_loc_fine_1 = (nblk_mult_cols_1+nblk-1)/nblk ! number of complete and incomplete blocks that with fine-grained process np_bc_fine_1
      if (useCCL) then
#ifdef USE_CCL_PXGEMM
        call gpu_ccl_copy_buf_send(PRECISION_CHAR, a_dev, buf_send_dev, l_rows, l_cols, nblk_mult_rows, nblk_mult_rows_max, &
                                    nblk, m_blocks_loc_fine, n_blocks_loc_fine_1, np_fine, np_bc_fine_1, &
                                    np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream)
#endif /* USE_CCL_PXGEMM */
      else ! useCCL
        do j_block_loc_fine = 0, n_blocks_loc_fine_1 - 1
          j_block_loc = (np_bc_fine_1 + j_block_loc_fine*np_cols_fine)/np_cols
          nblk_cut_col = min(nblk, l_cols-j_block_loc*nblk)

          m_blocks_loc_fine = (nblk_mult_rows+nblk-1)/nblk
          do i_block_loc_fine = 0, m_blocks_loc_fine - 1
            nblk_cut_row = min(nblk, nblk_mult_rows-i_block_loc_fine*nblk)
            i_block_loc = (np_fine + i_block_loc_fine*np_rows_fine)/np_rows

            buf_send(1+i_block_loc_fine*nblk: nblk_cut_row+i_block_loc_fine*nblk,   &
                      1+j_block_loc_fine*nblk: nblk_cut_col+j_block_loc_fine*nblk) = &
                    a(1+i_block_loc     *nblk: nblk_cut_row+i_block_loc     *nblk,   &
                      1+j_block_loc     *nblk: nblk_cut_col+j_block_loc     *nblk)
          enddo ! i_block_loc_fine
        enddo ! j_block_loc_fine
      endif ! useCCL

      ! PETERDEBUG: we send extra data to resolve the problem of continuity of the data.
      ! Alternatively, we could make buf_send and buf_recv to be 1D arrays of blocks (still 2D array of elements, so convenient to copy)
      if (useCCL) then
#ifdef USE_CCL_PXGEMM
        if (mpi_rank_target/=myid) then
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_pxgemm: ccl_send", successGPU)
          
          successGPU = ccl_Send(buf_send_dev, int(k_datatype*nblk_mult_rows_max*nblk_mult_cols_max,kind=c_size_t), &
                                cclDataType, mpi_rank_target, ccl_comm_all, my_stream)

          if (.not. successGPU) then
            print *,"Error in ccl_send"
            stop 1
          endif

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_pxgemm: ccl_send", successGPU)
        else
          ! PETERDEBUG: optimize memory usage - copy directly to at_col_dev (kernel needed or use gpu_ccl_copy_buf_recv)
          ! buf_self_dev = buf_send_dev
          successGPU = gpu_memcpy(buf_self_dev, buf_send_dev, nblk_mult_rows_max*nblk_mult_cols_max*size_of_datatype, &
                                  gpuMemcpyDeviceToDevice)
          check_memcpy_gpu("elpa_pxgemm: buf_self_dev <- buf_send_dev", successGPU)
        endif
#endif /* USE_CCL_PXGEMM */
      else ! useCCL
        call MPI_Send(buf_send, int(nblk_mult_rows_max*nblk_mult_cols_max, kind=MPI_KIND), &
                      MPI_MATH_DATATYPE_PRECISION, int(mpi_rank_target, kind=MPI_KIND), 0, &
                      int(mpi_comm_all, kind=MPI_KIND), mpierr)
      endif ! useCCL
      np_bc_fine_1 = mod(np_bc_fine_1+np_cols, np_cols_fine)
      if (np_bc_fine_1 == np_bc_fine_1_start) exit
    enddo ! np_bc_fine_1  periodic loop
  endif ! (mod(np_fine,np_rows) == my_prow)

  ! Recv
  if (mod(np_bc_fine,np_cols) == my_pcol) then
    my_prow_source = mod(np_fine, np_rows)

    do np_fine_1 = my_prow, np_rows_fine-1, np_rows
      np_bc_fine_1 = np_fine_1
      my_pcol_source = mod(np_bc_fine_1, np_cols)

      if (matrix_order==COLUMN_MAJOR_ORDER) then
        mpi_rank_source = my_prow_source + np_rows*my_pcol_source
      else
        mpi_rank_source = my_pcol_source + np_cols*my_prow_source
      endif

      call find_nblk_mult_dirs(l_rows, nblk, np_rows, np_fine_1, LCM, nblk_mult_rows_1)

      m_blocks_loc_fine_1 = (nblk_mult_rows_1+nblk-1)/nblk
      n_blocks_loc_fine   = (nblk_mult_cols  +nblk-1)/nblk
      
      if (useCCL) then
#ifdef USE_CCL_PXGEMM
        if (mpi_rank_source/=myid) then
          successGPU = ccl_Recv(buf_recv_dev, int(k_datatype*nblk_mult_rows_max*nblk_mult_cols_max,kind=c_size_t), &
                                cclDataType, mpi_rank_source, ccl_comm_all, my_stream)

          if (.not. successGPU) then
            print *,"Error in ccl_recv"
            stop 1
          endif

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_pxgemm: ccl_recv", successGPU)
        else
          ! buf_recv_dev = buf_self_dev
          successGPU = gpu_memcpy(buf_recv_dev, buf_self_dev, nblk_mult_rows_max*nblk_mult_cols_max*size_of_datatype, &
                                  gpuMemcpyDeviceToDevice)
          check_memcpy_gpu("elpa_pxgemm: buf_recv_dev <- buf_self_dev", successGPU)
        endif
#endif /* USE_CCL_PXGEMM */
      else ! useCCL
        call MPI_Recv(buf_recv, int(nblk_mult_rows_max*nblk_mult_cols_max, kind=MPI_KIND), &
                      MPI_MATH_DATATYPE_PRECISION, int(mpi_rank_source, kind=MPI_KIND), 0, &
                      int(mpi_comm_all, kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
      endif ! useCCL

      if (useCCL) then
#ifdef USE_CCL_PXGEMM
        call gpu_ccl_copy_buf_recv(PRECISION_CHAR, at_col_dev, buf_recv_dev, l_rows, l_cols, nblk_mult_cols, &
                                    nblk_mult_rows_max, nblk, m_blocks_loc_fine_1, n_blocks_loc_fine, np_fine_1, &
                                    np_bc_fine, np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream)
#endif /* USE_CCL_PXGEMM */
      else ! useCCL
        do i_block_loc_fine = 0, m_blocks_loc_fine_1 - 1
          i_block_loc = (np_fine_1 + i_block_loc_fine*np_rows_fine)/np_rows

          nblk_cut_row = min(nblk, l_rows-i_block_loc*nblk)

          do j_block_loc_fine = 0, n_blocks_loc_fine - 1
            nblk_cut_col = min(nblk, nblk_mult_cols-j_block_loc_fine*nblk)
            j_block_loc = (np_bc_fine + j_block_loc_fine*np_cols_fine)/np_cols
            at_col(1+i_block_loc     *nblk: nblk_cut_row+i_block_loc     *nblk,   &
                    1+j_block_loc     *nblk: nblk_cut_col+j_block_loc     *nblk) = &
transpose(buf_recv(1+j_block_loc_fine*nblk: nblk_cut_col+j_block_loc_fine*nblk,   &
                    1+i_block_loc_fine*nblk: nblk_cut_row+i_block_loc_fine*nblk))
          enddo ! j_block_loc_fine
        enddo ! i_block_loc_fine
      endif ! useCCL

    enddo ! np_fine_1
  endif ! (mod(np_bc_fine,np_cols) == my_pcol)
#ifdef WITH_NVTX
  call nvtxRangePop() ! transpose row
#endif

  

  call obj%timer%stop("elpa_transpose_row")

end subroutine

