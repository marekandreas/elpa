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


! transposes a row (col) of nblk-blocks of a matrix, along with the LCM-copies of the row
! there are two versions: 1) with CCL; 2) without CCL (CPU or GPU+MPI)
! row_col_char = R/r or C/c if we transpose a row or a column, respectively
subroutine elpa_transpose_row_or_col&
                            &CCL&
                            &MATH_DATATYPE&
                            &_&
                            &PRECISION &
                            (obj, row_col_char, &
#ifdef USE_CCL_PXGEMM
                              a_dev, at_dev, buf_send_dev, buf_recv_dev, buf_self_dev, &
#else /* USE_CCL_PXGEMM */
                              a, at, buf_send, buf_recv, buf_self, &
#endif /* USE_CCL_PXGEMM */
                              np_fine, l_rows, l_cols, nblk_mult_rows_max, nblk_mult_cols_max, debug)

  use, intrinsic :: iso_c_binding
  use precision
  use elpa_mpi
  use elpa_abstract_impl
  use elpa_utilities, only : least_common_multiple, check_memcpy_gpu_f
  use elpa_pxgemm_helpers, only : find_nblk_mult_dirs
  use elpa_gpu
  use elpa_ccl_gpu
  use pxgemm_multiply_gpu
#if defined(WITH_NVIDIA_GPU_VERSION) && defined(WITH_NVTX)
  use cuda_functions ! for NVTX labels
#elif defined(WITH_AMD_GPU_VERSION) && defined(WITH_ROCTX)
  use hip_functions  ! for ROCTX labels
#endif
  implicit none

#include "../../src/general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout)   :: obj
  character(len=1), intent(in)                 :: row_col_char
  integer(kind=ik), intent(in)                 :: np_fine, l_rows, l_cols, nblk_mult_rows_max, nblk_mult_cols_max
#ifdef USE_CCL_PXGEMM
  MATH_DATATYPE(kind=rck), allocatable         :: a(:,:), at(:,:), buf_send(:,:), buf_recv(:,:), buf_self(:,:) ! dummy variables
  integer(kind=c_intptr_t)                     :: a_dev, at_dev, buf_send_dev, buf_recv_dev, buf_self_dev
#else /* USE_CCL_PXGEMM */
  MATH_DATATYPE(kind=rck)                      :: a(l_rows,l_cols), at(l_rows,l_cols), &
                                                  buf_send(nblk_mult_rows_max, nblk_mult_cols_max), & 
                                                  buf_recv(nblk_mult_rows_max, nblk_mult_cols_max), &
                                                  buf_self(nblk_mult_rows_max, nblk_mult_cols_max)
  integer(kind=c_intptr_t)                     :: a_dev, at_dev, buf_send_dev, buf_recv_dev, buf_self_dev
#endif /* USE_CCL_PXGEMM */

  logical                                      :: row_transposed, col_transposed
  integer(kind=ik)                             :: l_dirs, l_dirs_t
  integer(kind=ik)                             :: nblk, debug

  ! MPI-related
  integer(kind=MPI_KIND)                       :: mpierr
  integer(kind=ik)                             :: my_prow, my_pcol, np_rows, np_cols, myid
  integer(kind=ik)                             :: my_pdir, my_pdir_t, np_dirs, np_dirs_t
  integer(kind=ik)                             :: mpi_comm_all
  integer(kind=ik)                             :: matrix_order
  integer(kind=ik)                             :: my_mpi_rank, mpi_rank_target, mpi_rank_source, &
                                                  my_pdir_target, my_pdir_t_target, &
                                                  my_pdir_source, my_pdir_t_source, &
                                                  my_prow_target, my_pcol_target, & 
                                                  my_prow_source, my_pcol_source, & 
                                                  my_pdir_target_deadlock

  integer(kind=ik)                             :: np_rows_fine, np_cols_fine, np_dirs_fine, np_dirs_t_fine, np_t_fine, np_bc_fine, &
                                                  np_t_fine_1, nblk_mult_dirs_1, np_fine_1, nblk_mult_dirs_t_1, &
                                                  m_blocks_loc_fine, mt_blocks_loc_fine, &
                                                  m_blocks_loc_fine_1, mt_blocks_loc_fine_1, &
                                                  np_t_fine_1_start
  integer(kind=ik)                             :: LCM, nblk_mult_rows, nblk_mult_cols, &
                                                  nblk_mult_dirs, nblk_mult_dirs_t, &
                                                  i_block_loc_fine, j_block_loc_fine, it_block_loc_fine, &
                                                  i_block_loc, j_block_loc, it_block_loc
  integer(kind=ik)                             :: nblk_cut_row, nblk_cut_col
  integer(kind=ik)                             :: nblk_cut_dir, nblk_cut_dir_t
  integer(kind=ik)                             :: lld_buf 
  integer(kind=ik)                             :: i_block_loc_fine_max, j_block_loc_fine_max 
  integer(kind=ik)                             :: np, np_t
  integer(kind=ik)                             :: error
  integer(kind=c_intptr_t), parameter          :: size_of_datatype = size_of_&
                                                  &PRECISION&
                                                  &_&
                                                  &MATH_DATATYPE

  ! GPU-related
  logical                                      :: successGPU, useCCL
  integer(kind=c_intptr_t)                     :: my_stream
  integer(kind=ik)                             :: SM_count

  integer(kind=c_intptr_t)                     :: ccl_comm_all
  integer(kind=c_int)                          :: cclDataType
  integer(kind=ik)                             :: k_datatype

  call obj%timer%start("elpa_transpose_row")

  if (row_col_char == 'R' .or. row_col_char == 'r') then
    row_transposed = .true.
    col_transposed = .false.
  else if (row_col_char == 'C' .or. row_col_char == 'c') then
    row_transposed = .false.
    col_transposed = .true.
  else
    print *, "elpa_transpose_row_or_col: row_col_char must be 'R'/'r' or 'C'/'c'. Aborting..."
    stop 1
  endif

  !   success = .true.
  useCCL = .false.
  
  call obj%get("matrix_order", matrix_order, error)
  if (error .ne. ELPA_OK) then
    print *, "elpa_pxgemm_multiply_transpose: Problem getting option matrix_order. Aborting..."
    stop 1
  endif

  !na      = obj%na
  nblk    = obj%nblk
  !lda     = obj%local_nrows
  !ldaCols = obj%local_ncols

  mpi_comm_all    = obj%mpi_setup%mpi_comm_parent

  myid    = obj%mpi_setup%myRank_comm_parent
  my_prow = obj%mpi_setup%myRank_comm_rows
  my_pcol = obj%mpi_setup%myRank_comm_cols

  np_rows = obj%mpi_setup%nRanks_comm_rows
  np_cols = obj%mpi_setup%nRanks_comm_cols

  LCM = least_common_multiple(np_rows, np_cols)*nblk

#if defined(USE_CCL_PXGEMM)
  useCCL = obj%gpu_setup%useCCL

  my_stream = obj%gpu_setup%my_stream
  ccl_comm_all  = obj%gpu_setup%ccl_comm_all
  SM_count = obj%gpu_setup%gpuSMcount
  
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

  np_bc_fine = np_fine
  nblk_mult_rows = find_nblk_mult_dirs(l_rows, nblk, np_rows, np_fine   , LCM)
  nblk_mult_cols = find_nblk_mult_dirs(l_cols, nblk, np_cols, np_bc_fine, LCM)

  if (row_transposed) then
    ! dir=row
    my_pdir   = my_prow
    my_pdir_t = my_pcol
    np_dirs   = np_rows
    np_dirs_t = np_cols
    l_dirs    = l_rows
    l_dirs_t  = l_cols
    nblk_mult_dirs   = nblk_mult_rows
    nblk_mult_dirs_t = nblk_mult_cols
  else 
    ! dir=col
    my_pdir   = my_pcol
    my_pdir_t = my_prow
    np_dirs   = np_cols
    np_dirs_t = np_rows
    l_dirs    = l_cols
    l_dirs_t  = l_rows
    nblk_mult_dirs   = nblk_mult_cols
    nblk_mult_dirs_t = nblk_mult_rows
  endif

  np_bc_fine = np_fine
  np_t_fine  = np_fine
  np_rows_fine = least_common_multiple(np_rows, np_cols)
  np_cols_fine = np_rows_fine
  np_dirs_fine = np_rows_fine
  np_dirs_t_fine = np_rows_fine

#ifdef WITH_NVTX
  call nvtxRangePush("transpose row/col")
#endif
  ! a -> at: transpose block-row (block-col) #np_fine of a
  ! Send
  if (mod(np_fine,np_dirs) == my_pdir) then

    my_pdir_t_target = mod(np_t_fine,np_dirs_t)
    
    ! we send to the process (mod(np_fine,np_rows), mod(np_bc_fine,np_cols)) in last turn
    ! to avoid the deadlock
    my_pdir_target_deadlock = mod(np_fine,np_dirs)
    
    np_t_fine_1 = my_pdir_t
    np_t_fine_1_start = mod(np_t_fine_1, np_dirs_t_fine)
    ! dry run: to find, whether there is a potential deadlock
    do np_t_fine_1 = my_pdir_t, np_dirs_t_fine-1, np_dirs_t
      np_fine_1 = np_t_fine_1
      my_pdir_target = mod(np_fine_1, np_dirs)
      if (my_pdir_target==my_pdir_target_deadlock) then
        np_t_fine_1_start = mod(np_t_fine_1+np_dirs_t, np_dirs_fine)
        exit
      endif
    enddo
    
    np_t_fine_1 = np_t_fine_1_start
    do ! np_t_fine_1 periodic loop
      np_fine_1 = np_t_fine_1
      my_pdir_target = mod(np_fine_1, np_dirs)

      if (row_transposed) then
        my_prow_target = my_pdir_target
        my_pcol_target = my_pdir_t_target
      else
        my_prow_target = my_pdir_t_target
        my_pcol_target = my_pdir_target
      endif

      if (matrix_order==COLUMN_MAJOR_ORDER) then
        mpi_rank_target = my_prow_target + np_rows*my_pcol_target
      else
        mpi_rank_target = my_pcol_target + np_cols*my_prow_target
      endif

      nblk_mult_dirs_t_1 = find_nblk_mult_dirs(l_dirs_t, nblk, np_dirs_t, np_t_fine_1, LCM)

      mt_blocks_loc_fine_1 = (nblk_mult_dirs_t_1+nblk-1)/nblk ! number of complete and incomplete blocks that with fine-grained process np_t_fine_1
      m_blocks_loc_fine    = (nblk_mult_dirs    +nblk-1)/nblk

      if (row_transposed) then
        j_block_loc_fine_max = mt_blocks_loc_fine_1 - 1
        i_block_loc_fine_max = m_blocks_loc_fine - 1
        np_t = np_t_fine_1
        np = np_fine
      else
        j_block_loc_fine_max = m_blocks_loc_fine - 1
        i_block_loc_fine_max = mt_blocks_loc_fine_1 - 1
        np_t = np_t_fine
        np = np_fine_1
      endif

      if (useCCL) then
        ! PETERDEBUG: changes needed here? Modification of kernel?
        lld_buf = nblk_mult_rows_max
        call gpu_ccl_copy_buf_send(PRECISION_CHAR, a_dev, buf_send_dev, l_rows, l_cols, lld_buf, &
                                   nblk, i_block_loc_fine_max, j_block_loc_fine_max, np, np_t, &
                                   np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream)
      else ! useCCL
        ! The nested loop is symmetric wrt to i,j, so we use the rigid order of indices for convenience of copying
        do j_block_loc_fine = 0, j_block_loc_fine_max
          j_block_loc = (np_t + j_block_loc_fine*np_cols_fine)/np_cols
          nblk_cut_col = min(nblk, l_cols-j_block_loc*nblk)

          do i_block_loc_fine = 0, i_block_loc_fine_max
            i_block_loc = (np + i_block_loc_fine*np_rows_fine)/np_rows
            nblk_cut_row = min(nblk, l_rows-i_block_loc*nblk)

            buf_send(1+ i_block_loc_fine*nblk: nblk_cut_row + i_block_loc_fine*nblk,   &
                      1+ j_block_loc_fine*nblk: nblk_cut_col + j_block_loc_fine*nblk) = &
                    a(1+ i_block_loc     *nblk: nblk_cut_row + i_block_loc     *nblk,   &
                      1+ j_block_loc     *nblk: nblk_cut_col + j_block_loc     *nblk)
          enddo ! i_block_loc_fine
        enddo ! j_block_loc_fine
      
      endif ! useCCL

      ! PETERDEBUG: we send extra data to resolve the problem of continuity of the data.
      ! Alternatively, we could make buf_send and buf_recv to be 1D arrays of blocks (still 2D array of elements, so convenient to copy)
      if (useCCL) then
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
          ! PETERDEBUG: optimize memory usage - copy directly to at_dev (kernel needed or use gpu_ccl_copy_buf_recv)
          ! buf_self_dev = buf_send_dev
          successGPU = gpu_memcpy(buf_self_dev, buf_send_dev, nblk_mult_rows_max*nblk_mult_cols_max*size_of_datatype, &
                                  gpuMemcpyDeviceToDevice)
          check_memcpy_gpu("elpa_pxgemm: buf_self_dev <- buf_send_dev", successGPU)
        endif
      else ! useCCL
        if (mpi_rank_target/=myid) then
#ifdef WITH_MPI
          call MPI_Send(buf_send, int(nblk_mult_rows_max*nblk_mult_cols_max, kind=MPI_KIND), &
                        MPI_MATH_DATATYPE_PRECISION, int(mpi_rank_target, kind=MPI_KIND), 0, &
                        int(mpi_comm_all, kind=MPI_KIND), mpierr)
#endif
        else
          buf_self = buf_send
        endif
      endif ! useCCL
      np_t_fine_1 = mod(np_t_fine_1+np_dirs_t, np_dirs_t_fine)
      if (np_t_fine_1 == np_t_fine_1_start) exit
    enddo ! np_t_fine_1  periodic loop
  endif ! (mod(np_fine,np_cols) == my_pcol)

  ! Recv
  if (mod(np_t_fine,np_dirs_t) == my_pdir_t) then
    my_pdir_source = mod(np_fine, np_dirs)

    do np_fine_1 = my_pdir, np_dirs_fine-1, np_dirs
      np_t_fine_1 = np_fine_1
      my_pdir_t_source = mod(np_t_fine_1, np_dirs_t)

      if (row_transposed) then
        my_prow_source = my_pdir_source
        my_pcol_source = my_pdir_t_source
      else
        my_prow_source = my_pdir_t_source
        my_pcol_source = my_pdir_source
      endif

      if (matrix_order==COLUMN_MAJOR_ORDER) then
        mpi_rank_source = my_prow_source + np_rows*my_pcol_source
      else
        mpi_rank_source = my_pcol_source + np_cols*my_prow_source
      endif

      nblk_mult_dirs_1 = find_nblk_mult_dirs(l_dirs, nblk, np_dirs, np_fine_1, LCM)

      m_blocks_loc_fine_1  = (nblk_mult_dirs_1+nblk-1)/nblk
      mt_blocks_loc_fine   = (nblk_mult_dirs_t+nblk-1)/nblk
      
      if (row_transposed) then
        j_block_loc_fine_max = mt_blocks_loc_fine - 1
        i_block_loc_fine_max = m_blocks_loc_fine_1 - 1
        np = np_fine_1
        np_t = np_t_fine
      else
        j_block_loc_fine_max = m_blocks_loc_fine_1 - 1
        i_block_loc_fine_max = mt_blocks_loc_fine - 1
        np = np_fine
        np_t = np_t_fine_1
      endif

      if (useCCL) then
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
      else ! useCCL
        if (mpi_rank_source/=myid) then
#ifdef WITH_MPI
          call MPI_Recv(buf_recv, int(nblk_mult_rows_max*nblk_mult_cols_max, kind=MPI_KIND), &
                        MPI_MATH_DATATYPE_PRECISION, int(mpi_rank_source, kind=MPI_KIND), 0, &
                        int(mpi_comm_all, kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
#endif
        else
          buf_recv = buf_self
        endif
      endif ! useCCL

      if (useCCL) then
        lld_buf = nblk_mult_rows_max
        call gpu_ccl_copy_buf_recv(PRECISION_CHAR, at_dev, buf_recv_dev, l_rows, l_cols, &
                                   lld_buf, nblk, i_block_loc_fine_max, j_block_loc_fine_max, np, np_t, &
                                   np_rows_fine, np_cols_fine, np_rows, np_cols, SM_count, debug, my_stream)
      else ! useCCL
        do i_block_loc_fine = 0, i_block_loc_fine_max
          i_block_loc = (np + i_block_loc_fine*np_rows_fine)/np_rows
          nblk_cut_row = min(nblk, l_rows-i_block_loc*nblk)

          do j_block_loc_fine = 0, j_block_loc_fine_max
            j_block_loc = (np_t + j_block_loc_fine*np_cols_fine)/np_cols
            nblk_cut_col = min(nblk, l_cols-j_block_loc*nblk)
            
            at(1+ i_block_loc     *nblk: nblk_cut_row + i_block_loc     *nblk,   &
               1+ j_block_loc     *nblk: nblk_cut_col + j_block_loc     *nblk) = &
#if defined (REALCASE)
            transpose(buf_recv(1+ j_block_loc_fine*nblk: nblk_cut_col + j_block_loc_fine*nblk,   &
                               1+ i_block_loc_fine*nblk: nblk_cut_row + i_block_loc_fine*nblk))
#else
      conjg(transpose(buf_recv(1+ j_block_loc_fine*nblk: nblk_cut_col + j_block_loc_fine*nblk,   &
                               1+ i_block_loc_fine*nblk: nblk_cut_row + i_block_loc_fine*nblk)) )
#endif
          enddo ! j_block_loc_fine
        enddo ! i_block_loc_fine
      
      endif ! useCCL

    enddo ! np_fine_1
  endif ! (mod(np_t_fine,np_dirs_t) == my_pdir)

  if (debug==1) successGPU = gpu_DeviceSynchronize()
  call obj%timer%stop("elpa_transpose_row")

#ifdef WITH_NVTX
  call nvtxRangePop() ! transpose row/col
#endif

end subroutine

