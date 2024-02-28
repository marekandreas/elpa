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
!
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".

#include "../general/sanity.F90"


!#if REALCASE == 1
#ifdef WITH_CUDA_AWARE_MPI
#define WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
#else
#undef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
#endif
!#endif

!#if COMPLEXCASE == 1
!#undef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
!#endif

subroutine trans_ev_tridi_to_band_&
&MATH_DATATYPE&
&_&
&PRECISION &
(obj, na, nev, nblk, nbw, q, ldq, matrixCols,         &
 hh_trans, my_pe, mpi_comm_rows, mpi_comm_cols, wantDebug, useGPU, max_threads_in, success, &
 kernel)

!-------------------------------------------------------------------------------
!  trans_ev_tridi_to_band_real/complex:
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
!  matrixCols  local columns of matrix q
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns/both
!
!-------------------------------------------------------------------------------
  use ELPA_utilities, only : error_unit
  use elpa_abstract_impl
  use elpa2_workload
  use pack_unpack_cpu
  use pack_unpack_gpu
  use compute_hh_trafo
  use elpa_gpu
  use precision
  use, intrinsic :: iso_c_binding
#ifdef WITH_OPENMP_TRADITIONAL
  ! use omp_lib
#endif
  implicit none
#include "../general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout)   :: obj
  logical, intent(in)                          :: useGPU

  integer(kind=ik), intent(in)                 :: kernel, my_pe
  integer(kind=ik), intent(in)                 :: na, nev, nblk, nbw, ldq, matrixCols, mpi_comm_rows, mpi_comm_cols

#ifdef USE_ASSUMED_SIZE
  MATH_DATATYPE(kind=rck), target              :: q(ldq,*)
#else
  MATH_DATATYPE(kind=rck), target              :: q(ldq,matrixCols)
#endif

  MATH_DATATYPE(kind=rck), intent(in),target   :: hh_trans(:,:)
  integer(kind=c_intptr_t)                     :: hh_trans_dev
  type(c_ptr)                                  :: hh_trans_mpi_dev
  MATH_DATATYPE(kind=rck), pointer             :: hh_trans_mpi_fortran_ptr(:,:)


  integer(kind=ik)                             :: np_rows, my_prow, np_cols, my_pcol
  integer(kind=MPI_KIND)                       :: np_rowsMPI, my_prowMPI, np_colsMPI, my_pcolMPI
  integer(kind=ik)                             :: i, j, ip, sweep, nbuf, l_nev, a_dim2
  integer(kind=ik)                             :: current_n, current_local_n, current_n_start, current_n_end
  integer(kind=ik)                             :: next_n, next_local_n, next_n_start, next_n_end
  integer(kind=ik)                             :: bottom_msg_length, top_msg_length, next_top_msg_length
  integer(kind=ik)                             :: stripe_width, last_stripe_width, stripe_count
#ifdef WITH_OPENMP_TRADITIONAL
  integer(kind=ik)                             :: thread_width, thread_width2, csw, b_off, b_len
#endif
  integer(kind=ik)                             :: num_result_blocks, num_result_buffers, num_bufs_recvd
  integer(kind=ik)                             :: a_off, current_tv_off, max_blk_size
  integer(kind=ik)                             :: src, src_offset, dst, offset, nfact, num_blk
  integer(kind=MPI_KIND)                       :: mpierr

  logical                                      :: flag
#ifdef WITH_OPENMP_TRADITIONAL
  MATH_DATATYPE(kind=rck), pointer             :: aIntern(:,:,:,:)
#else
  MATH_DATATYPE(kind=rck), pointer             :: aIntern(:,:,:)
#endif
  MATH_DATATYPE(kind=rck)                      :: a_var

  type(c_ptr)                                  :: aIntern_ptr

  MATH_DATATYPE(kind=rck), allocatable, target :: row(:)

  integer(kind=c_intptr_t)                     :: row_dev
  type(c_ptr)                                  :: row_mpi_dev
  MATH_DATATYPE(kind=rck), pointer             :: row_mpi_fortran_ptr(:)

  MATH_DATATYPE(kind=rck), pointer             :: row_group(:,:)

  MATH_DATATYPE(kind=rck), allocatable, target :: top_border_send_buffer(:,:)
  MATH_DATATYPE(kind=rck), allocatable, target :: top_border_recv_buffer(:,:)
  MATH_DATATYPE(kind=rck), allocatable, target :: bottom_border_send_buffer(:,:)
  MATH_DATATYPE(kind=rck), allocatable, target :: bottom_border_recv_buffer(:,:)

  integer(kind=c_intptr_t)                     :: top_border_recv_buffer_dev, top_border_send_buffer_dev
  type(c_ptr)                                  :: top_border_recv_buffer_mpi_dev, top_border_send_buffer_mpi_dev
  MATH_DATATYPE(kind=rck), pointer             :: top_border_recv_buffer_mpi_fortran_ptr(:,:), &
                                                  top_border_send_buffer_mpi_fortran_ptr(:,:)
  integer(kind=c_intptr_t)                     :: bottom_border_send_buffer_dev, bottom_border_recv_buffer_dev
  type(c_ptr)                                  :: bottom_border_send_buffer_mpi_dev, bottom_border_recv_buffer_mpi_dev
  MATH_DATATYPE(kind=rck), pointer             :: bottom_border_send_buffer_mpi_fortran_ptr(:,:), &
                                                  bottom_border_recv_buffer_mpi_fortran_ptr(:,:)
  type(c_ptr)                                  :: aIntern_mpi_dev
#ifdef WITH_OPENMP_TRADITIONAL
  MATH_DATATYPE(kind=rck), pointer             :: aIntern_mpi_fortran_ptr(:,:,:,:)
#else
  MATH_DATATYPE(kind=rck), pointer             :: aIntern_mpi_fortran_ptr(:,:,:)
#endif

  integer(kind=c_intptr_t)                     :: aIntern_dev
  integer(kind=c_intptr_t)                     :: bcast_buffer_dev

  type(c_ptr)                                  :: bcast_buffer_mpi_dev
  MATH_DATATYPE(kind=rck), pointer             :: bcast_buffer_mpi_fortran_ptr(:,:)

  integer(kind=c_intptr_t)                     :: num
  integer(kind=c_intptr_t)                     :: dev_offset, dev_offset_1
  integer(kind=c_intptr_t)                     :: row_group_dev

  type(c_ptr)                                  :: row_group_mpi_dev
  MATH_DATATYPE(kind=rck), pointer             :: row_group_mpi_fortran_ptr(:,:)

  integer(kind=c_intptr_t)                     :: q_dev
  type(c_ptr)                                  :: q_mpi_dev
  MATH_DATATYPE(kind=rck), pointer             :: q_mpi_fortran_ptr(:,:)

  integer(kind=c_intptr_t)                     :: hh_tau_dev
  MATH_DATATYPE(kind=rck), pointer             :: hh_tau_debug(:)
  integer(kind=ik)                             :: row_group_size, unpack_idx

  type(c_ptr)                                  :: row_group_host, bcast_buffer_host

  integer(kind=ik)                             :: n_times
  integer(kind=ik)                             :: chunk, this_chunk

  MATH_DATATYPE(kind=rck), allocatable,target  :: result_buffer(:,:,:)
  integer(kind=c_intptr_t)                     :: result_buffer_dev

  type(c_ptr)                                  :: result_buffer_mpi_dev
  MATH_DATATYPE(kind=rck), pointer             :: result_buffer_mpi_fortran_ptr(:,:,:)

  MATH_DATATYPE(kind=rck), pointer             :: bcast_buffer(:,:)

  integer(kind=ik)                             :: n_off

  integer(kind=MPI_KIND), allocatable          :: result_send_request(:), result_recv_request(:)
  integer(kind=ik), allocatable                :: limits(:)
  integer(kind=MPI_KIND), allocatable          :: top_send_request(:), bottom_send_request(:)
  integer(kind=MPI_KIND), allocatable          :: top_recv_request(:), bottom_recv_request(:)

  ! MPI send/recv tags, arbitrary

  integer(kind=ik), parameter                  :: bottom_recv_tag = 111
  integer(kind=ik), parameter                  :: top_recv_tag    = 222
  integer(kind=ik), parameter                  :: result_recv_tag = 333

  integer(kind=ik), intent(in)                 :: max_threads_in
  integer(kind=ik)                             :: max_threads

#ifdef WITH_OPENMP_TRADITIONAL
  integer(kind=ik)                             :: my_thread
#endif


  ! Just for measuring the kernel performance
  real(kind=c_double)                          :: kernel_time, kernel_time_recv ! MPI_WTIME always needs double
  ! long integer
  integer(kind=lik)                            :: kernel_flops, kernel_flops_recv

  logical, intent(in)                          :: wantDebug
  logical                                      :: success
  integer(kind=ik)                             :: istat, print_flops
  character(200)                               :: errorMessage
  character(20)                                :: gpuString
  logical                                      :: successGPU
#ifndef WITH_MPI
  integer(kind=ik)                             :: j1
#endif
  integer(kind=ik)                             :: error
  integer(kind=c_intptr_t), parameter          :: size_of_datatype = size_of_&
                                                                 &PRECISION&
                                                                 &_&
                                                                 &MATH_DATATYPE
#ifndef WITH_MPI
#ifdef WITH_AMD_GPU_VERSION
  logical, parameter                           :: allComputeOnGPU = .false.
#else
  logical, parameter                           :: allComputeOnGPU = .true.
#endif
#else /* WITH_MPI */
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
  logical, parameter                           :: allComputeOnGPU = .true.
#else
  logical, parameter                           :: allComputeOnGPU = .false.
#endif
#endif /* WITH_MPI */

  integer(kind=MPI_KIND)                     :: bcast_request1, allreduce_request1, allreduce_request2, &
                                                allreduce_request3, allreduce_request4
  logical                                    :: useNonBlockingCollectivesCols
  logical                                    :: useNonBlockingCollectivesRows
  integer(kind=c_int)                        :: non_blocking_collectives_rows, non_blocking_collectives_cols

  integer(kind=c_intptr_t)                   :: gpuHandle, my_stream

  if(useGPU) then
    gpuString = "_gpu"
  else
    gpuString = ""
  endif

  call obj%timer%start("trans_ev_tridi_to_band_&
  &MATH_DATATYPE&
  &" // &
  &PRECISION_SUFFIX //&
  gpuString)


#ifdef WITH_OPENMP_TRADITIONAL
  if (useGPU) then
    max_threads=1
  else
    max_threads = max_threads_in
  endif
    call omp_set_num_threads(max_threads)
#else
  max_threads = max_threads_in
#endif

  call obj%get("nbc_row_elpa2_tridi_to_band", non_blocking_collectives_rows, error)
  if (error .ne. ELPA_OK) then
    print *,"Problem setting option for non blocking collectives for rows in elpa2_tridi_to_band. Aborting..."
    stop 1
  endif

  call obj%get("nbc_col_elpa2_tridi_to_band", non_blocking_collectives_cols, error)
  if (error .ne. ELPA_OK) then
    print *,"Problem setting option for non blocking collectives for cols in elpa2_tridi_to_band. Aborting..."
    stop 1
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

  n_times = 0
  if (useGPU) then
    unpack_idx = 0
    row_group_size = 0
  endif

  success = .true.
  kernel_time = 0.0
  kernel_flops = 0

  my_prow = obj%mpi_setup%myRank_comm_rows
  my_pcol = obj%mpi_setup%myRank_comm_cols

  np_rows = obj%mpi_setup%nRanks_comm_rows
  np_cols = obj%mpi_setup%nRanks_comm_cols

  !if (wantDebug) call obj%timer%start("mpi_communication")
  !call MPI_Comm_rank(int(mpi_comm_rows,kind=MPI_KIND) , my_prowMPI , mpierr)
  !call MPI_Comm_size(int(mpi_comm_rows,kind=MPI_KIND) , np_rowsMPI , mpierr)
  !call MPI_Comm_rank(int(mpi_comm_cols,kind=MPI_KIND) , my_pcolMPI , mpierr)
  !call MPI_Comm_size(int(mpi_comm_cols,kind=MPI_KIND) , np_colsMPI , mpierr)

  !my_prow = int(my_prowMPI,kind=c_int)
  !my_pcol = int(my_pcolMPI,kind=c_int)
  !np_rows = int(np_rowsMPI,kind=c_int)
  !np_cols = int(np_colsMPI,kind=c_int)

  !if (wantDebug) call obj%timer%stop("mpi_communication")

  if (mod(nbw,nblk)/=0) then
    if (my_prow==0 .and. my_pcol==0) then
      if (wantDebug) then
        write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_&
                            &MATH_DATATYPE&
                            &: ERROR: nbw=',nbw,', nblk=',nblk
        write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_&
                            &MATH_DATATYPE&
                            &: band backtransform works only for nbw==n*nblk'
      endif
      success = .false.
      return
    endif
  endif

  nfact = nbw / nblk


  ! local number of eigenvectors
  l_nev = local_index(nev, my_pcol, np_cols, nblk, -1)

  if (l_nev==0) then
#ifdef WITH_OPENMP_TRADITIONAL
    thread_width = 0
#endif
    stripe_width = 0
    stripe_count = 0
    last_stripe_width = 0

  else ! l_nev

#ifdef WITH_OPENMP_TRADITIONAL
    ! Suggested stripe width is 48 since 48*64 real*8 numbers should fit into
    ! every primary cache
    ! Suggested stripe width is 48 - should this be reduced for the complex case ???

    thread_width = (l_nev-1)/max_threads + 1 ! number of eigenvectors per OMP thread

    if (useGPU) then
      stripe_width = 1024 ! Must be a multiple of 4
      stripe_count = (l_nev - 1) / stripe_width + 1

!#ifdef OPENMP_GPU_TRANS_EV_TRIDI
      stripe_count = (thread_width-1)/stripe_width + 1
      ! Adapt stripe width so that last one doesn't get too small
      stripe_width = (thread_width-1)/stripe_count + 1
!#endif

    else ! useGPU

#if REALCASE == 1
      call obj%get("stripewidth_real",stripe_width, error)

#ifdef DOUBLE_PRECISION_REAL
      !stripe_width = 48 ! Must be a multiple of 4
#else
      stripe_width = stripe_width * 2
      !stripe_width = 96 ! Must be a multiple of 8
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
      call obj%get("stripewidth_complex",stripe_width, error)

#ifdef DOUBLE_PRECISION_COMPLEX
      !stripe_width = 48 ! Must be a multiple of 2
#else
      stripe_width = stripe_width * 2
      !stripe_width = 48 ! Must be a multiple of 4
#endif
#endif /* COMPLEXCASE */

      stripe_count = (thread_width-1)/stripe_width + 1

      ! Adapt stripe width so that last one doesn't get too small

      stripe_width = (thread_width-1)/stripe_count + 1
    endif ! useGPU


    if (.not.(useGPU)) then
#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
      if (kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK2 .or. &
          kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK4 .or. &
          kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK6 .or. &
          kernel .eq. ELPA_2STAGE_REAL_SVE512_BLOCK2 .or. &
          kernel .eq. ELPA_2STAGE_REAL_SVE512_BLOCK4 .or. &
          kernel .eq. ELPA_2STAGE_REAL_SVE512_BLOCK6 &
          ) then

        stripe_width = ((stripe_width+7)/8)*8 ! Must be a multiple of 8 because of AVX-512 memory alignment of 64 bytes
                                              ! (8 * sizeof(double) == 64)

      else
        stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 4 because of AVX/SSE memory alignment of 32 bytes
                                              ! (4 * sizeof(double) == 32)
      endif
#else /*  DOUBLE_PRECISION_REAL */
      if (kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK2 .or. &
          kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK4 .or. &
          kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK6 .or. &
          kernel .eq. ELPA_2STAGE_REAL_SVE512_BLOCK2 .or. &
          kernel .eq. ELPA_2STAGE_REAL_SVE512_BLOCK4 .or. &
          kernel .eq. ELPA_2STAGE_REAL_SVE512_BLOCK6 &
          ) then


        stripe_width = ((stripe_width+15)/16)*16 ! Must be a multiple of 16 because of AVX-512 memory alignment of 64 bytes
                                           ! (16 * sizeof(float) == 64)

      else
        stripe_width = ((stripe_width+7)/8)*8 ! Must be a multiple of 8 because of AVX/SSE memory alignment of 32 bytes
                                         ! (8 * sizeof(float) == 32)
      endif
#endif /*  DOUBLE_PRECISION_REAL */
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX
      if (kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK1 .or. &
          kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK2 .or. &
          kernel .eq. ELPA_2STAGE_COMPLEX_SVE512_BLOCK1 .or. &
          kernel .eq. ELPA_2STAGE_COMPLEX_SVE512_BLOCK2 &
          ) then

        stripe_width = ((stripe_width+7)/8)*8 ! Must be a multiple of 4 because of AVX-512 memory alignment of 64 bytes
                                        ! (4 * sizeof(double complex) == 64)

      else

        stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 2 because of AVX/SSE memory alignment of 32 bytes
                                        ! (2 * sizeof(double complex) == 32)
      endif
#else /* DOUBLE_PRECISION_COMPLEX */

      if (kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK1 .or. &
          kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK2 .or. &
          kernel .eq. ELPA_2STAGE_COMPLEX_SVE512_BLOCK1 .or. &
          kernel .eq. ELPA_2STAGE_COMPLEX_SVE512_BLOCK2  &
          ) then

        stripe_width = ((stripe_width+7)/8)*8 ! Must be a multiple of 8 because of AVX-512 memory alignment of 64 bytes
                                        ! (8 * sizeof(float complex) == 64)

      else
        stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 4 because of AVX/SSE memory alignment of 32 bytes
                                        ! (4 * sizeof(float complex) == 32)
      endif
#endif /* DOUBLE_PRECISION_COMPLEX */
#endif /* COMPLEXCASE */
    endif ! useGPU

    if (useGPU) then
      !new
      last_stripe_width = l_nev - (stripe_count-1)*stripe_width
      ! not needed in OpenMP case
      ! last_stripe_width = l_nev - (stripe_count-1)*stripe_width
    endif ! useGPU

#else /* WITH_OPENMP_TRADITIONAL */

    ! Suggested stripe width is 48 since 48*64 real*8 numbers should fit into
    ! every primary cache
    ! Suggested stripe width is 48 - should this be reduced for the complex case ???

    if (useGPU) then
      stripe_width = 1024 ! Must be a multiple of 4
      stripe_count = (l_nev - 1) / stripe_width + 1

    else ! useGPU
#if REALCASE == 1
      call obj%get("stripewidth_real",stripe_width, error)

#ifdef DOUBLE_PRECISION_REAL
      !stripe_width = 48 ! Must be a multiple of 4
#else
      !stripe_width = 96 ! Must be a multiple of 8
      stripe_width = 2 * stripe_width
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
      call obj%get("stripewidth_complex",stripe_width, error)

#ifdef DOUBLE_PRECISION_COMPLEX
      !stripe_width = 48 ! Must be a multiple of 2
#else
      !stripe_width = 48 ! Must be a multiple of 4
#endif
#endif /* COMPLEXCASE */

      stripe_count = (l_nev-1)/stripe_width + 1

      ! Adapt stripe width so that last one doesn't get too small

      stripe_width = (l_nev-1)/stripe_count + 1

#if REALCASE == 1
#ifdef DOUBLE_PRECISION_REAL
      if (kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK2 .or. &
          kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK4 .or. &
          kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK6 .or. &
          kernel .eq. ELPA_2STAGE_REAL_SVE512_BLOCK2 .or. &
          kernel .eq. ELPA_2STAGE_REAL_SVE512_BLOCK4 .or. &
          kernel .eq. ELPA_2STAGE_REAL_SVE512_BLOCK6  &
          ) then

        stripe_width = ((stripe_width+7)/8)*8 ! Must be a multiple of 8 because of AVX-512 memory alignment of 64 bytes
                                              ! (8 * sizeof(double) == 64)

      else
        stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 4 because of AVX/SSE memory alignment of 32 bytes
                                            ! (4 * sizeof(double) == 32)
      endif
#else
      if (kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK2 .or. &
          kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK4 .or. &
          kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK6 .or. &
          kernel .eq. ELPA_2STAGE_REAL_SVE512_BLOCK2 .or. &
          kernel .eq. ELPA_2STAGE_REAL_SVE512_BLOCK4 .or. &
          kernel .eq. ELPA_2STAGE_REAL_SVE512_BLOCK6  &
          ) then


       stripe_width = ((stripe_width+15)/16)*16 ! Must be a multiple of 16 because of AVX-512 memory alignment of 64 bytes
                                               ! (16 * sizeof(float) == 64)

     else
       stripe_width = ((stripe_width+7)/8)*8 ! Must be a multiple of 8 because of AVX/SSE memory alignment of 32 bytes
                                            ! (8 * sizeof(float) == 32)
     endif
#endif
#endif /* REALCASE */

#if COMPLEXCASE == 1
#ifdef DOUBLE_PRECISION_COMPLEX

     if (kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK1 .or. &
         kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK2 .or. &
         kernel .eq. ELPA_2STAGE_COMPLEX_SVE512_BLOCK1 .or. &
         kernel .eq. ELPA_2STAGE_COMPLEX_SVE512_BLOCK2  &
     ) then

       stripe_width = ((stripe_width+7)/8)*8 ! Must be a multiple of 4 because of AVX-512 memory alignment of 64 bytes
                                       ! (4 * sizeof(double complex) == 64)

     else

       stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 2 because of AVX/SSE memory alignment of 32 bytes
                                       ! (2 * sizeof(double complex) == 32)
     endif
#else

     if (kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK1 .or. &
         kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK2 .or. &
         kernel .eq. ELPA_2STAGE_COMPLEX_SVE512_BLOCK1 .or. &
         kernel .eq. ELPA_2STAGE_COMPLEX_SVE512_BLOCK2  &
         ) then

       stripe_width = ((stripe_width+15)/16)*16 ! Must be a multiple of 8 because of AVX-512 memory alignment of 64 bytes
                                       ! (8 * sizeof(float complex) == 64)

     else
       stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 4 because of AVX/SSE memory alignment of 32 bytes
                                       ! (4 * sizeof(float complex) == 32)
     endif
#endif
#endif /* COMPLEXCASE */
   endif ! useGPU

   last_stripe_width = l_nev - (stripe_count-1)*stripe_width

#endif /* WITH_OPENMP_TRADITIONAL */
  endif ! l_nev

  ! Determine the matrix distribution at the beginning

  allocate(limits(0:np_rows), stat=istat, errmsg=errorMessage)
  check_allocate("tridi_to_band: limits", istat, errorMessage)
  call determine_workload(obj,na, nbw, np_rows, limits)

  max_blk_size = maxval(limits(1:np_rows) - limits(0:np_rows-1))

  a_dim2 = max_blk_size + nbw

  if (useGPU) then

    if (allComputeOnGPU) then
      if (wantDebug) call obj%timer%start("cuda_memcpy")

      successGPU = gpu_malloc(q_dev, ldq*matrixCols* size_of_datatype)
      check_alloc_gpu("tridi_to_band: q_dev", successGPU)

#ifdef WITH_GPU_STREAMS
      successGPU = gpu_host_register(int(loc(q),kind=c_intptr_t), &
                    ldq*matrixCols * size_of_datatype,&
                    gpuHostRegisterDefault)
      check_host_register_gpu("tridi_to_band: q", successGPU)

      my_stream = obj%gpu_setup%my_stream
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_tridi_to_band 1: q -> q_dev", successGPU)

      successGPU =  gpu_memcpy_async(q_dev, int(loc(q(1,1)),kind=c_intptr_t),  &
                               ldq*matrixCols * size_of_datatype, &
                               gpuMemcpyHostToDevice, my_stream)
      check_memcpy_gpu("trans_ev_tridi_to_band 1: q -> q_dev", successGPU)
      
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_tridi_to_band 1: q -> q_dev", successGPU)
      ! synchronize streamsPerThread; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      check_stream_synchronize_gpu("trans_ev_tridi_to_band 1: q -> q_dev", successGPU)
#else
      successGPU =  gpu_memcpy(q_dev, int(loc(q(1,1)),kind=c_intptr_t),  &
                               ldq*matrixCols * size_of_datatype, &
                               gpuMemcpyHostToDevice)
      check_memcpy_gpu("trans_ev_tridi_to_band 1: q -> q_dev", successGPU)
#endif

      ! associate with c_ptr
      q_mpi_dev = transfer(q_dev, q_mpi_dev)
      ! and associate a fortran pointer
      call c_f_pointer(q_mpi_dev, q_mpi_fortran_ptr, &
                       [ldq,matrixCols])
      if (wantDebug) call obj%timer%stop("cuda_memcpy")

      successGPU = gpu_malloc(hh_trans_dev, size(hh_trans,dim=1)*size(hh_trans,dim=2)* size_of_datatype)
      check_alloc_gpu("tridi_to_band: hh_trans_dev", successGPU)
      ! associate with c_ptr
      hh_trans_mpi_dev = transfer(hh_trans_dev, hh_trans_mpi_dev)
      ! and associate a fortran pointer
      call c_f_pointer(hh_trans_mpi_dev, hh_trans_mpi_fortran_ptr, &
                       [size(hh_trans,dim=1),size(hh_trans,dim=2)])
#ifdef WITH_GPU_STREAMS

      successGPU = gpu_host_register(int(loc(hh_trans),kind=c_intptr_t), &
                    size(hh_trans,dim=1)*size(hh_trans,dim=2) * size_of_datatype,&
                    gpuHostRegisterDefault)
      check_host_register_gpu("tridi_to_band: hh_trans", successGPU)

      my_stream = obj%gpu_setup%my_stream
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("tridi_to_band: hh_trans -> hh_trans_dev", successGPU)

      successGPU =  gpu_memcpy_async(c_loc(hh_trans_mpi_fortran_ptr(1,1)),  &
                               c_loc(hh_trans(1,1)), &
                               size(hh_trans,dim=1)*size(hh_trans,dim=2) * size_of_datatype, &
                               gpuMemcpyHostToDevice, my_stream)
      check_memcpy_gpu("tridi_to_band: hh_trans -> hh_trans_dev", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("tridi_to_band: hh_trans -> hh_trans_dev", successGPU)
      ! synchronize streamsPerThread; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      check_stream_synchronize_gpu("tridi_to_band: hh_trans -> hh_trans_dev", successGPU)
#else
      successGPU =  gpu_memcpy(c_loc(hh_trans_mpi_fortran_ptr(1,1)),  &
                               c_loc(hh_trans(1,1)), &
                               size(hh_trans,dim=1)*size(hh_trans,dim=2) * size_of_datatype, &
                               gpuMemcpyHostToDevice)
      check_memcpy_gpu("tridi_to_band: hh_trans -> hh_trans_dev", successGPU)
#endif

    endif ! allComputeOnGPU

#ifdef WITH_OPENMP_TRADITIONAL
    num = (stripe_width*a_dim2*stripe_count*max_threads)* size_of_datatype
#else
    num = (stripe_width*a_dim2*stripe_count)* size_of_datatype
#endif
    successGPU = gpu_malloc(aIntern_dev, num)
    check_alloc_gpu("tridi_to_band: aIntern_dev", successGPU)

    ! openmp loop here
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif

#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
      successGPU = gpu_memset_async(aIntern_dev , 0, num, my_stream)
      check_memset_gpu("tridi_to_band: aIntern_dev", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("tridi_to_band: aIntern_dev", successGPU)
#else
      successGPU = gpu_memset(aIntern_dev , 0, num)
      check_memset_gpu("tridi_to_band: aIntern_dev", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    else
#ifdef WITH_OPENMP_TRADITIONAL
      allocate(aIntern(stripe_width,a_dim2,stripe_count,max_threads))
      aIntern(:,:,:,:) = 0.
#else
      allocate(aIntern(stripe_width,a_dim2,stripe_count))
      aIntern(:,:,:) = 0.
#endif

      successGPU = gpu_memcpy(aIntern_dev, int(loc(aIntern),kind=c_intptr_t), &
                              num, gpuMemcpyHostToDevice)
      check_memcpy_gpu("tridi_to_band: aIntern -> aInternd_dev", successGPU)
      deallocate(aIntern)
    endif
#endif /* WITH_OPENMP_OFFLOAD_GPU_VERSION */

    if (allComputeOnGPU) then
      ! associate with c_ptr
      aIntern_mpi_dev = transfer(aIntern_dev, aIntern_mpi_dev)
      ! and associate a fortran pointer
#ifdef WITH_OPENMP_TRADITIONAL
      call c_f_pointer(aIntern_mpi_dev, aIntern_mpi_fortran_ptr, &
                       [stripe_width,a_dim2,stripe_count,max_threads])
#else
      call c_f_pointer(aIntern_mpi_dev, aIntern_mpi_fortran_ptr, &
                       [stripe_width,a_dim2,stripe_count])
#endif
    endif ! allComputeOnGPU

    ! "row_group" and "row_group_dev" are needed for GPU optimizations
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      successGPU = gpu_malloc_host(row_group_host,l_nev*nblk*size_of_datatype)
      check_host_alloc_gpu("tridi_to_band: row_group_host", successGPU)
      call c_f_pointer(row_group_host, row_group, (/l_nev,nblk/))
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    else
      allocate(row_group(l_nev,nblk))
    endif
#endif

    row_group(:, :) = 0.0_rck
    num =  (l_nev*nblk)* size_of_datatype
    successGPU = gpu_malloc(row_group_dev, num)
    check_alloc_gpu("tridi_to_band: row_group_dev", successGPU)

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif

#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
      successGPU = gpu_memset_async(row_group_dev , 0, num, my_stream)
      check_memset_gpu("tridi_to_band: row_group_dev", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("tridi_to_band: row_group_dev", successGPU)
#else
      successGPU = gpu_memset(row_group_dev , 0, num)
      check_memset_gpu("tridi_to_band: row_group_dev", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    else
      successGPU = gpu_memcpy(row_group_dev, int(loc(row_group),kind=c_intptr_t), &
                              num, gpuMemcpyHostToDevice)
      check_memcpy_gpu("tridi_to_band: row_group -> row_group_dev", successGPU)
    endif
#endif
    if (allComputeOnGPU) then
      ! associate with c_ptr
      row_group_mpi_dev = transfer(row_group_dev, row_group_mpi_dev)
      ! and associate a fortran pointer
      call c_f_pointer(row_group_mpi_dev, row_group_mpi_fortran_ptr, &
                       [l_nev,nblk])
    endif ! allComputeOnGPU

  else ! GPUs are not used

#if 0
! realcase or complexcase
!DEC$ ATTRIBUTES ALIGN: 64:: aIntern
#endif

#ifdef WITH_OPENMP_TRADITIONAL
    if (posix_memalign(aIntern_ptr, 64_c_intptr_t, stripe_width*a_dim2*stripe_count*max_threads*     &
           C_SIZEOF(a_var)) /= 0) then
      print *,"trans_ev_tridi_to_band_&
      &MATH_DATATYPE&
      &: error when allocating aIntern"//errorMessage
      stop 1
    endif

    call c_f_pointer(aIntern_ptr, aIntern, [stripe_width,a_dim2,stripe_count,max_threads])
    ! allocate(aIntern(stripe_width,a_dim2,stripe_count,max_threads), stat=istat, errmsg=errorMessage)

    ! aIntern(:,:,:,:) should be set to 0 in a parallel region, not here!

#else /* WITH_OPENMP_TRADITIONAL */

    if (posix_memalign(aIntern_ptr, 64_c_intptr_t, stripe_width*a_dim2*stripe_count*  &
        C_SIZEOF(a_var)) /= 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating aIntern"//errorMessage
      stop 1
    endif

    call c_f_pointer(aIntern_ptr, aIntern,[stripe_width,a_dim2,stripe_count] )
    !allocate(aIntern(stripe_width,a_dim2,stripe_count), stat=istat, errmsg=errorMessage)

    aIntern(:,:,:) = 0.0_rck
#endif /* WITH_OPENMP_TRADITIONAL */
  endif !useGPU

  allocate(row(l_nev), stat=istat, errmsg=errorMessage)
  check_allocate("tridi_to_band: row", istat, errorMessage)

  row(:) = 0.0_rck

  if (useGPU .and. allComputeOnGPU) then
    num =  (l_nev)* size_of_datatype
    successGPU = gpu_malloc(row_dev, num)
    check_alloc_gpu("tridi_to_band: row_dev", successGPU)
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif

#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("tridi_to_band: row_dev", successGPU)

      successGPU = gpu_memset_async(row_dev , 0, num, my_stream)
      check_memset_gpu("tridi_to_band: row_dev", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("tridi_to_band: row_dev", successGPU)
#else
      successGPU = gpu_memset(row_dev , 0, num)
      check_memset_gpu("tridi_to_band: row_dev", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    else
      successGPU = gpu_memcpy(row_dev, int(loc(row),kind=c_intptr_t), &
                              num, gpuMemcpyHostToDevice)
      check_memcpy_gpu("tridi_to_band: row -> row_dev", successGPU)
    endif
#endif

    ! associate with c_ptr
    row_mpi_dev = transfer(row_dev, row_mpi_dev)
    ! and associate a fortran pointer
    call c_f_pointer(row_mpi_dev, row_mpi_fortran_ptr, &
                     [l_nev])
  endif

  ! Copy q from a block cyclic distribution into a distribution with contiguous rows,
  ! and transpose the matrix using stripes of given stripe_width for cache blocking.

  ! The peculiar way it is done below is due to the fact that the last row should be
  ! ready first since it is the first one to start below

#ifdef WITH_OPENMP_TRADITIONAL
  ! Please note about the OMP usage below:
  ! This is not for speed, but because we want the matrix a in the memory and
  ! in the cache of the correct thread (if possible)
  if (.not.(useGPU)) then
    call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)
    !$omp parallel do num_threads(max_threads) if(max_threads > 1) &
    !$omp default(none) &
    !$omp private(my_thread) &
    !$omp shared(max_threads, aIntern) &
    !$omp schedule(static, 1)
    do my_thread = 1, max_threads
      aIntern(:,:,:,my_thread) = 0.0_rck ! if possible, do first touch allocation!
    enddo
    !$omp end parallel do
    call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)
  endif
#endif /* WITH_OPENMP_TRADITIONAL */


  if (wantDebug) call obj%timer%start("ip_loop")
  do ip = np_rows-1, 0, -1
    if (my_prow == ip) then
      ! Receive my rows which have not yet been received
      src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
      do i=limits(ip)+1,limits(ip+1)
        src = mod((i-1)/nblk, np_rows)

        if (src < my_prow) then

#ifdef WITH_OPENMP_TRADITIONAL
          if (useGPU) then
            ! An unpacking of the current row group may occur before queuing the next row

            my_stream = obj%gpu_setup%my_stream
            call unpack_and_prepare_row_group_&
            &MATH_DATATYPE&
            &_gpu_&
            &PRECISION &
                          ( obj, &
                          row_group, row_group_dev, aIntern_dev, stripe_count, &
                                      stripe_width, last_stripe_width, a_dim2, l_nev,&
                                      row_group_size, nblk, unpack_idx, &
                                       i - limits(ip), .false., wantDebug, allComputeOnGPU, my_stream)

#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
            !row_group 2
            if (wantDebug) call obj%timer%start("cuda_mpi_communication")
            call MPI_Recv(row_group_mpi_fortran_ptr(:, row_group_size), int(l_nev,kind=MPI_KIND), &
                          MPI_MATH_DATATYPE_PRECISION_EXPL, &
                          int(src,kind=MPI_KIND), 0_MPI_KIND, int(mpi_comm_rows,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
            if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
            if (wantDebug) call obj%timer%start("host_mpi_communication")
            call MPI_Recv(row_group(:, row_group_size), int(l_nev,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                          int(src,kind=MPI_KIND), 0_MPI_KIND, int(mpi_comm_rows,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
            if (wantDebug) call obj%timer%stop("host_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */

#else /* WITH_MPI */
            if (allComputeOnGPU) then
              ! memcopy row_dev -> row_group_dev
#ifdef WITH_GPU_STREAMS
              my_stream = obj%gpu_setup%my_stream
              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("tridi_to_band: row_dev", successGPU)

              successGPU =  gpu_memcpy_async(c_loc(row_group_mpi_fortran_ptr(1,row_group_size)), &
                                       c_loc(row_mpi_fortran_ptr(1)),  &
                                       l_nev* size_of_datatype,      &
                                       gpuMemcpyDeviceToDevice, my_stream)
              check_memcpy_gpu("tridi_to_band: row_dev -> row_group_dev", successGPU)

              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("tridi_to_band: row_dev -> row_group_dev", successGPU)
              ! synchronize streamsPerThread; maybe not neccessary
              successGPU = gpu_stream_synchronize()
              check_stream_synchronize_gpu("tridi_to_band: row_dev -> row_group_dev", successGPU)

#else
              successGPU =  gpu_memcpy(c_loc(row_group_mpi_fortran_ptr(1,row_group_size)), &
                                       c_loc(row_mpi_fortran_ptr(1)),  &
                                       l_nev* size_of_datatype,      &
                                       gpuMemcpyDeviceToDevice)
              check_memcpy_gpu("tridi_to_band: row_dev -> row_group_dev", successGPU)
#endif

#ifndef WITH_GPU_STREAMS
              if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
              successGPU = gpu_devicesynchronize()
              check_memcpy_gpu("tridi_to_band: device_synchronize", successGPU)
              if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
#endif
            else ! allComputeOnGPU
              row_group(1:l_nev, row_group_size) = row(1:l_nev)
            endif ! allComputeOnGPU
#endif /* WITH_MPI */

          else ! useGPU
#ifdef WITH_MPI
            if (wantDebug) call obj%timer%start("mpi_communication")
            call MPI_Recv(row, int(l_nev,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                          int(src,kind=MPI_KIND), 0_MPI_KIND, int(mpi_comm_rows,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
            if (wantDebug) call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
!            row(1:l_nev) = row(1:l_nev)
#endif /* WITH_MPI */

            call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)

            !$omp parallel do num_threads(max_threads) if(max_threads > 1) &
            !$omp default(none) &
            !$omp private(my_thread) &
            !$omp shared(max_threads, obj, aIntern, row, i, limits, ip, stripe_count, thread_width, &
            !$omp&       stripe_width, l_nev) &
            !$omp schedule(static, 1)
            do my_thread = 1, max_threads
              call unpack_row_&
                   &MATH_DATATYPE&
                   &_cpu_openmp_&
                   &PRECISION &
                                (obj, aIntern, row, i-limits(ip), my_thread, stripe_count, &
                                 thread_width, stripe_width, l_nev)

            enddo
            !$omp end parallel do

            call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)
          endif ! useGPU
#else /* WITH_OPENMP_TRADITIONAL */
          if (useGPU) then
            ! An unpacking of the current row group may occur before queuing the next row

            my_stream = obj%gpu_setup%my_stream
            call unpack_and_prepare_row_group_&
            &MATH_DATATYPE&
            &_gpu_&
            &PRECISION &
                          ( obj, &
                          row_group, row_group_dev, aIntern_dev, stripe_count, &
                                      stripe_width, last_stripe_width, a_dim2, l_nev,&
                                      row_group_size, nblk, unpack_idx, &
                                       i - limits(ip), .false., wantDebug, allComputeOnGPU, my_stream)
#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
            if (wantDebug) call obj%timer%start("cuda_mpi_communication")
            call MPI_Recv(row_group_mpi_fortran_ptr(:, row_group_size), int(l_nev,kind=MPI_KIND), &
                          MPI_MATH_DATATYPE_PRECISION_EXPL, &
                          int(src,kind=MPI_KIND), 0_MPI_KIND, int(mpi_comm_rows,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
            if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
            if (wantDebug) call obj%timer%start("host_mpi_communication")
            call MPI_Recv(row_group(:, row_group_size), int(l_nev,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                          int(src,kind=MPI_KIND), 0_MPI_KIND, int(mpi_comm_rows,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
            if (wantDebug) call obj%timer%stop("host_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#else /* WITH_MPI */
            if (allComputeOnGPU) then
              ! memcpy row_dev -> row_group_dev
#ifdef WITH_GPU_STREAMS
              my_stream = obj%gpu_setup%my_stream
              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("tridi_to_band: row_dev", successGPU)

              successGPU =  gpu_memcpy_async(c_loc(row_group_mpi_fortran_ptr(1,row_group_size)), &
                                       c_loc(row_mpi_fortran_ptr(1)),  &
                                       l_nev* size_of_datatype,      &
                                       gpuMemcpyDeviceToDevice, my_stream)
              check_memcpy_gpu("tridi_to_band: row_dev -> row_group_dev", successGPU)

              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("tridi_to_band: row_dev -> row_group_dev", successGPU)
              ! synchronize streamsPerThread; maybe not neccessary
              successGPU = gpu_stream_synchronize()
              check_stream_synchronize_gpu("tridi_to_band: row_dev -> row_group_dev", successGPU)
#else
              successGPU =  gpu_memcpy(c_loc(row_group_mpi_fortran_ptr(1,row_group_size)), &
                                       c_loc(row_mpi_fortran_ptr(1)),  &
                                       l_nev* size_of_datatype,      &
                                       gpuMemcpyDeviceToDevice)
              check_memcpy_gpu("tridi_to_band: row_dev -> row_group_dev", successGPU)
#endif

#ifndef WITH_GPU_STREAMS
              if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
              successGPU = gpu_devicesynchronize()
              check_memcpy_gpu("tridi_to_band: device_synchronize", successGPU)
              if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
#endif
            else
              row_group(1:l_nev, row_group_size) = row(1:l_nev)
            endif
#endif /* WITH_MPI */

          else ! useGPU
#ifdef WITH_MPI
            if (wantDebug) call obj%timer%start("mpi_communication")
            call MPI_Recv(row, int(l_nev,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                         int(src,kind=MPI_KIND), 0_MPI_KIND, int(mpi_comm_rows,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
            if (wantDebug) call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
!           row(1:l_nev) = row(1:l_nev)
#endif /* WITH_MPI */

            call unpack_row_&
                &MATH_DATATYPE&
                &_cpu_&
                &PRECISION &
                (obj, aIntern, row,i-limits(ip), stripe_count, stripe_width, last_stripe_width)
          endif ! useGPU
#endif /* WITH_OPENMP_TRADITIONAL */


        elseif (src == my_prow) then

          src_offset = src_offset+1

          if (useGPU) then

            ! An unpacking of the current row group may occur before queuing the next row

            call unpack_and_prepare_row_group_&
                 &MATH_DATATYPE&
                 &_gpu_&
                 &PRECISION &
             ( obj, &
                          row_group, row_group_dev, aIntern_dev, stripe_count, &
                          stripe_width, last_stripe_width, a_dim2, l_nev,&
                          row_group_size, nblk, unpack_idx, &
                          i - limits(ip), .false., wantDebug, allComputeOnGPU, my_stream)

            if (allComputeOnGPU) then
              if (wantDebug) call obj%timer%start("cuda_aware_gpublas")
              gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
              call gpublas_PRECISION_COPY(l_nev, c_loc(q_mpi_fortran_ptr(src_offset,1)), ldq, &
                                          c_loc(row_group_mpi_fortran_ptr(1,row_group_size)), 1, gpuHandle)
              if (wantDebug) call obj%timer%stop("cuda_aware_gpublas")
            else ! allComputeOnGPU
              row_group(:, row_group_size) = q(src_offset, 1:l_nev)
            endif ! allComputeOnGPU
          else ! useGPU
            row(:) = q(src_offset, 1:l_nev)
          endif ! useGPU

#ifdef WITH_OPENMP_TRADITIONAL
          if (useGPU) then
          else
            call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)

            !$omp parallel do num_threads(max_threads) if(max_threads > 1) &
            !$omp default(none) &
            !$omp private(my_thread) &
            !$omp shared(max_threads, obj, aIntern, row, i, limits, ip, stripe_count, thread_width, &
            !$omp&       stripe_width, l_nev) &
            !$omp schedule(static, 1)
            do my_thread = 1, max_threads
              call unpack_row_&
                   &MATH_DATATYPE&
                   &_cpu_openmp_&
                   &PRECISION &
                   (obj, aIntern, row, i-limits(ip), my_thread, stripe_count, thread_width, stripe_width, l_nev)

            enddo
            !$omp end parallel do

            call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)
          endif
#else /* WITH_OPENMP_TRADITIONAL */

          if (useGPU) then

          else
            call unpack_row_&
                 &MATH_DATATYPE&
                 &_cpu_&
                 &PRECISION &
                            (obj, aIntern, row,i-limits(ip),  stripe_count, stripe_width, last_stripe_width)
          endif

#endif /* WITH_OPENMP_TRADITIONAL */

        endif ! src == my_prow
      enddo ! i=limits(ip)+1,limits(ip+1)


      ! Send all rows which have not yet been send


      if (allComputeOnGPU .and. useGPU) then
        src_offset = 0
        do dst = 0, ip-1
          do i=limits(dst)+1,limits(dst+1)
            if (mod((i-1)/nblk, np_rows) == my_prow) then
              src_offset = src_offset+1

              if (wantDebug) call obj%timer%start("cuda_aware_gpublas")
              gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
              call gpublas_PRECISION_COPY(l_nev, c_loc(q_mpi_fortran_ptr(src_offset,1)), ldq, &
                                          c_loc(row_mpi_fortran_ptr(1)), 1, gpuHandle)
              if (wantDebug) call obj%timer%stop("cuda_aware_gpublas")
#ifdef WITH_GPU_STREAMS
              if (wantDebug) call obj%timer%start("cuda_aware_stream_synchronize")
              successGPU = gpu_stream_synchronize(my_stream)
              check_memcpy_gpu("tridi_to_band: stream_synchronize", successGPU)
              if (wantDebug) call obj%timer%stop("cuda_aware_stream_synchronize")
#else
              if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
              successGPU = gpu_devicesynchronize()
              check_memcpy_gpu("tridi_to_band: device_synchronize", successGPU)
              if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
#endif

#ifdef WITH_MPI
              if (wantDebug) call obj%timer%start("cuda_mpi_communication")
              call MPI_Send(row_mpi_fortran_ptr, int(l_nev,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                                int(dst,kind=MPI_KIND), 0_MPI_KIND, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
              if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#endif /* WITH_MPI */

            endif
          enddo
        enddo
      else !  allComputeOnGPU
        src_offset = 0
        do dst = 0, ip-1
          do i=limits(dst)+1,limits(dst+1)
            if (mod((i-1)/nblk, np_rows) == my_prow) then
              src_offset = src_offset+1
              row(:) = q(src_offset, 1:l_nev)

#ifdef WITH_MPI
              if (wantDebug) call obj%timer%start("host_mpi_communication")
              call MPI_Send(row, int(l_nev,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                                int(dst,kind=MPI_KIND), 0_MPI_KIND, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
              if (wantDebug) call obj%timer%stop("host_mpi_communication")
#endif /* WITH_MPI */
            endif
          enddo
        enddo
      endif ! allComputeOnGPU

    else if (my_prow < ip) then

      ! Send all rows going to PE ip
      if (allComputeOnGPU .and. useGPU) then
        src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
        do i=limits(ip)+1,limits(ip+1)
          src = mod((i-1)/nblk, np_rows)
          if (src == my_prow) then
            src_offset = src_offset+1

            if (wantDebug) call obj%timer%start("cuda_aware_gpublas")
            gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
            call gpublas_PRECISION_COPY(l_nev, c_loc(q_mpi_fortran_ptr(src_offset,1)), ldq, &
                                          c_loc(row_mpi_fortran_ptr(1)), 1, gpuHandle)
            if (wantDebug) call obj%timer%stop("cuda_aware_gpublas")

#ifdef WITH_GPU_STREAMS
            if (wantDebug) call obj%timer%start("cuda_aware_stream_synchronize")
            successGPU = gpu_stream_synchronize(my_stream)
            check_memcpy_gpu("tridi_to_band: stream_synchronize", successGPU)
            if (wantDebug) call obj%timer%stop("cuda_aware_stream_synchronize")
#else
            ! is there a way to avoid this device_synchronize ?
            if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
            successGPU = gpu_devicesynchronize()
            check_memcpy_gpu("tridi_to_band: device_synchronize", successGPU)
            if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
#endif

#ifdef WITH_MPI
            if (wantDebug) call obj%timer%start("cuda_mpi_communication")
            call MPI_Send(row_mpi_fortran_ptr, int(l_nev,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                          int(ip,kind=MPI_KIND), 0_MPI_KIND, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
            if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#endif /* WITH_MPI */

          endif
        enddo
      else ! allComputeOnGPU
        src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
        do i=limits(ip)+1,limits(ip+1)
          src = mod((i-1)/nblk, np_rows)
          if (src == my_prow) then
            src_offset = src_offset+1
            row(:) = q(src_offset, 1:l_nev)
#ifdef WITH_MPI
            if (wantDebug) call obj%timer%start("host_mpi_communication")
            call MPI_Send(row, int(l_nev,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                        int(ip,kind=MPI_KIND), 0_MPI_KIND, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
            if (wantDebug) call obj%timer%stop("host_mpi_communication")
#endif /* WITH_MPI */
          endif
        enddo
      endif  ! allComputeOnGPU

      ! Receive all rows from PE ip
      do i=limits(my_prow)+1,limits(my_prow+1)
        src = mod((i-1)/nblk, np_rows)
        if (src == ip) then
#ifdef WITH_OPENMP_TRADITIONAL
          if (useGPU) then
            ! An unpacking of the current row group may occur before queuing the next row

            call unpack_and_prepare_row_group_&
                 &MATH_DATATYPE&
                 &_gpu_&
                 &PRECISION&
                 &( obj, &
                 row_group, row_group_dev, aIntern_dev, stripe_count,  &
                 stripe_width, last_stripe_width, a_dim2, l_nev,       &
                 row_group_size, nblk, unpack_idx,                     &
                 i - limits(my_prow), .false., wantDebug, allComputeOnGPU, my_stream)

#ifdef WITH_MPI
            ! normaly recv on row_group happens
            ! it might be that we need an device synchronize to prevent a race condition
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
            if (wantDebug) call obj%timer%start("cuda_mpi_communication")
            call MPI_Recv(row_group_mpi_fortran_ptr(:, row_group_size), int(l_nev,kind=MPI_KIND), &
                          MPI_MATH_DATATYPE_PRECISION_EXPL, &
                          int(src,kind=MPI_KIND), 0_MPI_KIND, int(mpi_comm_rows,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
            if (wantDebug) call obj%timer%stop("cuda_mpi_communication")

#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
            if (wantDebug) call obj%timer%start("host_mpi_communication")
            call MPI_Recv(row_group(:, row_group_size), int(l_nev,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                          int(src,kind=MPI_KIND), 0_MPI_KIND, int(mpi_comm_rows,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
            if (wantDebug) call obj%timer%stop("host_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#else /* WITH_MPI */
            if (allComputeOnGPU) then
              ! row_dev -> row_group_dev

#ifdef WITH_GPU_STREAMS
              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("tridi_to_band: row_dev", successGPU)

              successGPU =  gpu_memcpy_async(c_loc(row_group_mpi_fortran_ptr(1,row_group_size)), &
                                       c_loc(row_mpi_fortran_ptr(1)),  &
                                       l_nev* size_of_datatype,      &
                                       gpuMemcpyDeviceToDevice, my_stream)
              check_memcpy_gpu("tridi_to_band: row_dev -> row_group_dev", successGPU)

              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("tridi_to_band: row_dev -> row_group_dev", successGPU)
              ! synchronize streamsPerThread; maybe not neccessary
              successGPU = gpu_stream_synchronize()
              check_stream_synchronize_gpu("tridi_to_band: row_dev -> row_group_dev", successGPU)
#else
              successGPU =  gpu_memcpy(c_loc(row_group_mpi_fortran_ptr(1,row_group_size)), &
                                       c_loc(row_mpi_fortran_ptr(1)),  &
                                       l_nev* size_of_datatype,      &
                                       gpuMemcpyDeviceToDevice)
              check_memcpy_gpu("tridi_to_band: row_dev -> row_group_dev", successGPU)
#endif

#ifndef WITH_GPU_STREAMS
              if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
              successGPU = gpu_devicesynchronize()
              check_memcpy_gpu("tridi_to_band: device_synchronize", successGPU)
              if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
#endif
            else ! allComputeOnGPU
              row_group(1:l_nev,row_group_size) = row(1:l_nev)
            endif ! allComputeOnGPU
#endif /* WITH_MPI */

          else ! useGPU
#ifdef WITH_MPI
            if (wantDebug) call obj%timer%start("mpi_communication")
            call MPI_Recv(row, int(l_nev,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                              int(src,kind=MPI_KIND), 0_MPI_KIND, int(mpi_comm_rows,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
            if (wantDebug) call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */

!            row(1:l_nev) = row(1:l_nev)

#endif /* WITH_MPI */

            call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)
            !$omp parallel do  num_threads(max_threads) if(max_threads > 1) &
            !$omp default(none) &
            !$omp private(my_thread) &
            !$omp shared(max_threads, obj, aIntern, row, i, limits, my_prow, stripe_count, thread_width, &
            !$omp&       stripe_width, l_nev) &
            !$omp schedule(static, 1)
            do my_thread = 1, max_threads
              call unpack_row_&
                   &MATH_DATATYPE&
                   &_cpu_openmp_&
                   &PRECISION &
                   (obj, aIntern, row, i-limits(my_prow), my_thread, stripe_count, thread_width, stripe_width, l_nev)
            enddo
            !$omp end parallel do
            call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)
          endif ! useGPU
#else /* WITH_OPENMP_TRADITIONAL */
          if (useGPU) then
            ! An unpacking of the current row group may occur before queuing the next row

            call unpack_and_prepare_row_group_&
                 &MATH_DATATYPE&
                 &_gpu_&
                 &PRECISION&
                 &( obj, &
                 row_group, row_group_dev, aIntern_dev, stripe_count,  &
                 stripe_width, last_stripe_width, a_dim2, l_nev,       &
                 row_group_size, nblk, unpack_idx,                     &
                 i - limits(my_prow), .false., wantDebug, allComputeOnGPU, my_stream)

#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
            if (wantDebug) call obj%timer%start("cuda_mpi_communication")
            call MPI_Recv(row_group_mpi_fortran_ptr(:, row_group_size), int(l_nev,kind=MPI_KIND), &
                          MPI_MATH_DATATYPE_PRECISION_EXPL, &
                          int(src,kind=MPI_KIND), 0_MPI_KIND, int(mpi_comm_rows,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
            if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
            if (wantDebug) call obj%timer%start("host_mpi_communication")
            call MPI_Recv(row_group(:, row_group_size), int(l_nev,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                          int(src,kind=MPI_KIND), 0_MPI_KIND, int(mpi_comm_rows,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
            if (wantDebug) call obj%timer%stop("host_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#else /* WITH_MPI */
            if (allComputeOnGPU) then
              ! row_dev -> row_group_dev
#ifdef WITH_GPU_STREAMS
              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("tridi_to_band: row_dev", successGPU)

              successGPU =  gpu_memcpy_async(c_loc(row_group_mpi_fortran_ptr(1,row_group_size)), &
                                       c_loc(row_mpi_fortran_ptr(1)),  &
                                       l_nev* size_of_datatype,      &
                                       gpuMemcpyDeviceToDevice, my_stream)
              check_memcpy_gpu("tridi_to_band: row_dev -> row_group_dev", successGPU)

              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("tridi_to_band: row_dev -> row_group_dev", successGPU)
              ! synchronize streamsPerThread; maybe not neccessary
              successGPU = gpu_stream_synchronize()
              check_stream_synchronize_gpu("tridi_to_band: row_dev -> row_group_dev", successGPU)
#else
              successGPU =  gpu_memcpy(c_loc(row_group_mpi_fortran_ptr(1,row_group_size)), &
                                       c_loc(row_mpi_fortran_ptr(1)),  &
                                       l_nev* size_of_datatype,      &
                                       gpuMemcpyDeviceToDevice)
              check_memcpy_gpu("tridi_to_band: row_dev -> row_group_dev", successGPU)
#endif

#ifndef WITH_GPU_STREAMS
              if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
              successGPU = gpu_devicesynchronize()
              check_memcpy_gpu("tridi_to_band: device_synchronize", successGPU)
              if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
#endif
            else
              row_group(1:l_nev,row_group_size) = row(1:l_nev)
            endif
#endif /* WITH_MPI */

          else ! useGPU
#ifdef WITH_MPI
            if (wantDebug) call obj%timer%start("mpi_communication")
            call MPI_Recv(row, int(l_nev,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                         int(src,kind=MPI_KIND), 0_MPI_KIND, int(mpi_comm_rows,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
            if (wantDebug) call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
!           row(1:l_nev) = row(1:l_nev)
#endif
            call unpack_row_&
                 &MATH_DATATYPE&
                 &_cpu_&
                 &PRECISION &
                 (obj, aIntern, row,i-limits(my_prow), stripe_count, stripe_width, last_stripe_width)
          endif ! useGPU

#endif /* WITH_OPENMP_TRADITIONAL */

        endif
      enddo ! i=limits(my_prow)+1,limits(my_prow+1)
    endif ! (my_prow < ip)
  enddo ! ip = np_rows-1, 0, -1

  if (wantDebug) call obj%timer%stop("ip_loop")

  if (wantDebug) call obj%timer%start("allocate")

  if (useGPU) then
    ! Force an unpacking of all remaining rows that haven't been unpacked yet

    call unpack_and_prepare_row_group_&
         &MATH_DATATYPE&
         &_gpu_&
         &PRECISION&
         &( obj, &
         row_group, row_group_dev, aIntern_dev, stripe_count, &
         stripe_width, last_stripe_width, &
         a_dim2, l_nev, row_group_size, nblk, unpack_idx,     &
         -1, .true., wantDebug, allComputeOnGPU, my_stream)

  endif

  ! Set up result buffer queue

  num_result_blocks = ((na-1)/nblk + np_rows - my_prow) / np_rows

  num_result_buffers = 4*nfact
  allocate(result_buffer(l_nev,nblk,num_result_buffers), stat=istat, errmsg=errorMessage)
  check_allocate("tridi_to_band: result_buffer", istat, errorMessage)

  allocate(result_send_request(num_result_buffers), stat=istat, errmsg=errorMessage)
  check_allocate("tridi_to_band: result_send_request", istat, errorMessage)

  allocate(result_recv_request(num_result_buffers), stat=istat, errmsg=errorMessage)
  check_allocate("tridi_to_band: result_recv_request", istat, errorMessage)

  if (useGPU .and. allComputeOnGPU) then
    num_result_buffers = 4*nfact
    num =  (l_nev*nblk*num_result_buffers)* size_of_datatype
    successGPU = gpu_malloc(result_buffer_dev, num* size_of_datatype)
    check_alloc_gpu("tridi_to_band: result_buffer_dev", successGPU)

    ! associate with c_ptr
    result_buffer_mpi_dev = transfer(result_buffer_dev, result_buffer_mpi_dev)
    ! and associate a fortran pointer
    call c_f_pointer(result_buffer_mpi_dev, result_buffer_mpi_fortran_ptr, &
                     [l_nev,nblk,num_result_buffers])
  endif

#ifdef WITH_MPI
  result_send_request(:) = MPI_REQUEST_NULL
  result_recv_request(:) = MPI_REQUEST_NULL
#endif

  ! Queue up buffers
#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
  if (wantDebug) call obj%timer%start("cuda_aware_mpi_communication")
#else
  if (wantDebug) call obj%timer%start("host_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
  if (my_prow > 0 .and. l_nev>0) then ! note: row 0 always sends
    do j = 1, min(num_result_buffers, num_result_blocks)
      if (useGPU) then
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
        call MPI_Irecv(result_buffer_mpi_fortran_ptr(1,1,j), int(l_nev*nblk,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                       0_MPI_KIND, int(result_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND),          &
                      result_recv_request(j), mpierr)
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
        call MPI_Irecv(result_buffer(1,1,j), int(l_nev*nblk,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL,     &
                       0_MPI_KIND, int(result_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND),          &
                      result_recv_request(j), mpierr)
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
      else ! useGPU
        call MPI_Irecv(result_buffer(1,1,j), int(l_nev*nblk,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL,     &
                       0_MPI_KIND, int(result_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND),          &
                      result_recv_request(j), mpierr)
      endif ! useGPU
    enddo ! j = 1, min(num_result_buffers, num_result_blocks)
  endif ! my_prow > 0 .and. l_nev>0
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
  if (wantDebug) call obj%timer%stop("cuda_aware_mpi_communication")
#else
  if (wantDebug) call obj%timer%stop("host_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#else /* WITH_MPI */

  ! carefull the "recv" has to be done at the corresponding wait or send
  ! result_buffer(1: l_nev*nblk,1,j) =result_buffer(1:l_nev*nblk,1,nbuf)

#endif /* WITH_MPI */

  num_bufs_recvd = 0 ! No buffers received yet

  ! Initialize top/bottom requests

  allocate(top_send_request(stripe_count), stat=istat, errmsg=errorMessage)
  check_allocate("tridi_to_band: top_send_request", istat, errorMessage)

  allocate(top_recv_request(stripe_count), stat=istat, errmsg=errorMessage)
  check_allocate("tridi_to_band: top_recv_request", istat, errorMessage)

  allocate(bottom_send_request(stripe_count), stat=istat, errmsg=errorMessage)
  check_allocate("tridi_to_band: bottom_send_request", istat, errorMessage)

  allocate(bottom_recv_request(stripe_count), stat=istat, errmsg=errorMessage)
  check_allocate("tridi_to_band: bottom_recv_request", istat, errorMessage)

#ifdef WITH_MPI
  top_send_request(:) = MPI_REQUEST_NULL
  top_recv_request(:) = MPI_REQUEST_NULL
  bottom_send_request(:) = MPI_REQUEST_NULL
  bottom_recv_request(:) = MPI_REQUEST_NULL
#endif

#ifdef WITH_OPENMP_TRADITIONAL
  allocate(top_border_send_buffer(stripe_width*nbw*max_threads, stripe_count), stat=istat, errmsg=errorMessage)
  check_allocate("tridi_to_band: top_border_send_buffer", istat, errorMessage)

  allocate(top_border_recv_buffer(stripe_width*nbw*max_threads, stripe_count), stat=istat, errmsg=errorMessage)
  check_allocate("tridi_to_band: top_border_recv_buffer", istat, errorMessage)

  allocate(bottom_border_send_buffer(stripe_width*nbw*max_threads, stripe_count), stat=istat, errmsg=errorMessage)
  check_allocate("tridi_to_band: bottom_border_send_buffer", istat, errorMessage)

  allocate(bottom_border_recv_buffer(stripe_width*nbw*max_threads, stripe_count), stat=istat, errmsg=errorMessage)
  check_allocate("tridi_to_band: bottom_border_recv_buffer", istat, errorMessage)

  top_border_send_buffer(:,:) = 0.0_rck
  top_border_recv_buffer(:,:) = 0.0_rck
  bottom_border_send_buffer(:,:) = 0.0_rck
  bottom_border_recv_buffer(:,:) = 0.0_rck

#else /* WITH_OPENMP_TRADITIONAL */

  allocate(top_border_send_buffer(stripe_width*nbw, stripe_count), stat=istat, errmsg=errorMessage)
  check_allocate("tridi_to_band: top_border_send_buffer", istat, errorMessage)

  allocate(top_border_recv_buffer(stripe_width*nbw, stripe_count), stat=istat, errmsg=errorMessage)
  check_allocate("tridi_to_band: top_border_recv_buffer", istat, errorMessage)

  allocate(bottom_border_send_buffer(stripe_width*nbw, stripe_count), stat=istat, errmsg=errorMessage)
  check_allocate("tridi_to_band: bottom_border_send_buffer", istat, errorMessage)

  allocate(bottom_border_recv_buffer(stripe_width*nbw, stripe_count), stat=istat, errmsg=errorMessage)
  check_allocate("tridi_to_band: bottom_border_recv_buffer", istat, errorMessage)

  top_border_send_buffer(:,:) = 0.0_rck
  top_border_recv_buffer(:,:) = 0.0_rck
  bottom_border_send_buffer(:,:) = 0.0_rck
  bottom_border_recv_buffer(:,:) = 0.0_rck

#endif /* WITH_OPENMP_TRADITIONAL */

  if (useGPU) then
    if (allComputeOnGPU) then
      ! top_border_recv_buffer and top_border_send_buffer
#ifdef WITH_OPENMP_TRADITIONAL
      num =  ( stripe_width*nbw*max_threads * stripe_count) * size_of_datatype
#else
      num =  ( stripe_width*nbw*stripe_count) * size_of_datatype
#endif /* WITH_OPENMP_TRADITIONAL */
      successGPU = gpu_malloc(top_border_recv_buffer_dev, num)
      check_alloc_gpu("tridi_to_band: top_border_recv_buffer_dev", successGPU)

      successGPU = gpu_malloc(top_border_send_buffer_dev, num)
      check_alloc_gpu("tridi_to_band: top_border_send_buffer_dev", successGPU)
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif

#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memset_async(top_border_recv_buffer_dev, 0, num, my_stream)
        check_memset_gpu("tridi_to_band: top_border_recv_buffer_dev", successGPU)
        successGPU = gpu_memset_async(top_border_send_buffer_dev, 0, num, my_stream)
        check_memset_gpu("tridi_to_band: top_border_send_buffer_dev", successGPU)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("tridi_to_band: top_border_send_buffer_dev", successGPU)
#else
        successGPU = gpu_memset(top_border_recv_buffer_dev, 0, num)
        check_memset_gpu("tridi_to_band: top_border_recv_buffer_dev", successGPU)
        successGPU = gpu_memset(top_border_send_buffer_dev, 0, num)
        check_memset_gpu("tridi_to_band: top_border_send_buffer_dev", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      else
        successGPU = gpu_memcpy(top_border_recv_buffer_dev, int(loc(top_border_recv_buffer),kind=c_intptr_t), &
                              num, gpuMemcpyHostToDevice)
        check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> top_border_recv_buffer_dev", successGPU)
        successGPU = gpu_memcpy(top_border_send_buffer_dev, int(loc(top_border_send_buffer),kind=c_intptr_t), &
                              num, gpuMemcpyHostToDevice)
        check_memcpy_gpu("tridi_to_band: top_border_send_buffer -> top_border_send_buffer_dev", successGPU)
      endif
#endif

      ! associate with c_ptr
      top_border_recv_buffer_mpi_dev = transfer(top_border_recv_buffer_dev, top_border_recv_buffer_mpi_dev)
      top_border_send_buffer_mpi_dev = transfer(top_border_send_buffer_dev, top_border_send_buffer_mpi_dev)
      ! and create a fortran pointer
#ifdef WITH_OPENMP_TRADITIONAL
      call c_f_pointer(top_border_recv_buffer_mpi_dev, top_border_recv_buffer_mpi_fortran_ptr, &
                       [stripe_width*nbw*max_threads, stripe_count])
      call c_f_pointer(top_border_send_buffer_mpi_dev, top_border_send_buffer_mpi_fortran_ptr, &
                       [stripe_width*nbw*max_threads, stripe_count])
#else
      call c_f_pointer(top_border_recv_buffer_mpi_dev, top_border_recv_buffer_mpi_fortran_ptr, [stripe_width*nbw, stripe_count])
      call c_f_pointer(top_border_send_buffer_mpi_dev, top_border_send_buffer_mpi_fortran_ptr, [stripe_width*nbw, stripe_count])
#endif /* WITH_OPENMP_TRADITIONAL */

      ! bottom_border_send_buffer and bottom_border_recv_buffer
#ifdef WITH_OPENMP_TRADITIONAL
      num =  ( stripe_width*nbw*max_threads * stripe_count) * size_of_datatype
#else
      num =  ( stripe_width*nbw*stripe_count) * size_of_datatype
#endif /* WITH_OPENMP_TRADITIONAL */
      successGPU = gpu_malloc(bottom_border_send_buffer_dev, num)
      check_alloc_gpu("tridi_to_band: bottom_border_send_buffer_dev", successGPU)
      successGPU = gpu_malloc(bottom_border_recv_buffer_dev, num)
      check_alloc_gpu("tridi_to_band: bottom_border_recv_buffer_dev", successGPU)

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif

#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memset_async(bottom_border_send_buffer_dev, 0, num, my_stream)
        check_memset_gpu("tridi_to_band: bottom_border_send_buffer_dev", successGPU)
        successGPU = gpu_memset_async(bottom_border_recv_buffer_dev, 0, num, my_stream)
        check_memset_gpu("tridi_to_band: bottom_border_recv_buffer_dev", successGPU)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("tridi_to_band: bottom_border_recv_buffer_dev", successGPU)
#else
        successGPU = gpu_memset(bottom_border_send_buffer_dev, 0, num)
        check_memset_gpu("tridi_to_band: bottom_border_send_buffer_dev", successGPU)
        successGPU = gpu_memset(bottom_border_recv_buffer_dev, 0, num)
        check_memset_gpu("tridi_to_band: bottom_border_recv_buffer_dev", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      else
        successGPU = gpu_memcpy(bottom_border_recv_buffer_dev, int(loc(bottom_border_recv_buffer),kind=c_intptr_t), &
                              num, gpuMemcpyHostToDevice)
        check_memcpy_gpu("tridi_to_band: b_border_recv_buffer -> b_border_recv_buffer_dev", successGPU)
        successGPU = gpu_memcpy(bottom_border_send_buffer_dev, int(loc(bottom_border_send_buffer),kind=c_intptr_t), &
                              num, gpuMemcpyHostToDevice)
        check_memcpy_gpu("tridi_to_band: b_border_send_buffer -> b_border_send_buffer_dev", successGPU)
      endif
#endif

      ! associate with c_ptr
      bottom_border_send_buffer_mpi_dev = transfer(bottom_border_send_buffer_dev, bottom_border_send_buffer_mpi_dev)
      bottom_border_recv_buffer_mpi_dev = transfer(bottom_border_recv_buffer_dev, bottom_border_recv_buffer_mpi_dev)
      ! and create a fortran pointer
#ifdef WITH_OPENMP_TRADITIONAL
      call c_f_pointer(bottom_border_send_buffer_mpi_dev, bottom_border_send_buffer_mpi_fortran_ptr, &
                       [stripe_width*nbw*max_threads, stripe_count])
      call c_f_pointer(bottom_border_recv_buffer_mpi_dev, bottom_border_recv_buffer_mpi_fortran_ptr, &
                       [stripe_width*nbw*max_threads, stripe_count])
#else
      call c_f_pointer(bottom_border_send_buffer_mpi_dev, bottom_border_send_buffer_mpi_fortran_ptr, &
                       [stripe_width*nbw, stripe_count])
      call c_f_pointer(bottom_border_recv_buffer_mpi_dev, bottom_border_recv_buffer_mpi_fortran_ptr, &
                       [stripe_width*nbw, stripe_count])
#endif /* WITH_OPENMP_TRADITIONAL */
    endif ! allComputeOnGPU

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      successGPU = gpu_host_register(int(loc(top_border_send_buffer),kind=c_intptr_t), &
                    stripe_width*nbw* stripe_count * size_of_datatype,&
                    gpuHostRegisterDefault)
      check_host_register_gpu("tridi_to_band: top_border_send_buffer", successGPU)

      successGPU = gpu_host_register(int(loc(top_border_recv_buffer),kind=c_intptr_t), &
                    stripe_width*nbw* stripe_count * size_of_datatype,&
                    gpuHostRegisterDefault)
      check_host_register_gpu("tridi_to_band: top_border_recv_buffer", successGPU)

      successGPU = gpu_host_register(int(loc(bottom_border_send_buffer),kind=c_intptr_t), &
                    stripe_width*nbw* stripe_count * size_of_datatype,&
                    gpuHostRegisterDefault)
      check_host_register_gpu("tridi_to_band: bottom_border_send_buffer", successGPU)

      successGPU = gpu_host_register(int(loc(bottom_border_recv_buffer),kind=c_intptr_t), &
                    stripe_width*nbw* stripe_count * size_of_datatype,&
                    gpuHostRegisterDefault)
      check_host_register_gpu("tridi_to_band: bottom_border_recv_buffer", successGPU)
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    endif
#endif
  endif ! useGPU


  ! Initialize broadcast buffer

  if (useGPU) then
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      successGPU = gpu_malloc_host(bcast_buffer_host,nbw*max_blk_size*size_of_datatype)
      check_host_alloc_gpu("tridi_to_band: bcast_buffer_host", successGPU)
      call c_f_pointer(bcast_buffer_host, bcast_buffer, (/nbw,max_blk_size/))
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    else
      allocate(bcast_buffer(nbw,max_blk_size))
    endif
#endif
  else
    allocate(bcast_buffer(nbw, max_blk_size), stat=istat, errmsg=errorMessage)
    check_allocate("tridi_to_band: bcast_buffer", istat, errorMessage)
  endif

  bcast_buffer = 0.0_rck

  if (useGPU) then
    num =  ( nbw * max_blk_size) * size_of_datatype
    successGPU = gpu_malloc(bcast_buffer_dev, num)
    check_alloc_gpu("tridi_to_band: bcast_buffer_dev", successGPU)

    if (allComputeOnGPU) then
      ! associate with c_ptr
      bcast_buffer_mpi_dev = transfer(bcast_buffer_dev, bcast_buffer_mpi_dev)
      call c_f_pointer(bcast_buffer_mpi_dev, bcast_buffer_mpi_fortran_ptr, &
                      [nbw, max_blk_size])
    endif ! allComputeOnGPU

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif

#ifdef WITH_GPU_STREAMS
      successGPU = gpu_memset_async( bcast_buffer_dev, 0, num, my_stream)
      check_memset_gpu("tridi_to_band: bcast_buffer_dev", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("tridi_to_band: bcast_buffer_dev", successGPU)
#else
      successGPU = gpu_memset( bcast_buffer_dev, 0, num)
      check_memset_gpu("tridi_to_band: bcast_buffer_dev", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    else
      successGPU = gpu_memcpy(bcast_buffer_dev, int(loc(bcast_buffer),kind=c_intptr_t), &
                              num, gpuMemcpyHostToDevice)
      ! looks like the original copied from an incorrect pointer there.
      !successGPU = gpu_memcpy(bcast_buffer_dev, int(loc(bcast_buffer_dev),kind=c_intptr_t), &
      !                        num, gpuMemcpyHostToDevice)
      check_memcpy_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)
    endif
#endif

    num =  (max_blk_size)* size_of_datatype
    successGPU = gpu_malloc( hh_tau_dev, num)
    check_alloc_gpu("tridi_to_band: hh_tau_dev", successGPU)

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif

#ifdef WITH_GPU_STREAMS
      successGPU = gpu_memset_async( hh_tau_dev, 0, num, my_stream)
      check_memset_gpu("tridi_to_band: hh_tau_dev", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("tridi_to_band: hh_tau_dev", successGPU)
#else
      successGPU = gpu_memset( hh_tau_dev, 0, num)
      check_memset_gpu("tridi_to_band: hh_tau_dev", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    else
      allocate(hh_tau_debug(max_blk_size))
      hh_tau_debug(:) = 0.
      successGPU = gpu_memcpy(hh_tau_dev, int(loc(hh_tau_debug),kind=c_intptr_t), &
                              num, gpuMemcpyHostToDevice)
      check_memcpy_gpu("tridi_to_band: hh_tau_debug -> hh_tau_dev", successGPU)
      deallocate(hh_tau_debug)
    endif
#endif
  endif ! useGPU

  current_tv_off = 0 ! Offset of next row to be broadcast

  ! ------------------- start of work loop -------------------

  ! Pay attention that for a_off zero indexing is assumed
  a_off = 0 ! offset in aIntern (to avoid unnecessary shifts)

  top_msg_length = 0
  bottom_msg_length = 0
  if (wantDebug) call obj%timer%stop("allocate")

  if (wantDebug) call obj%timer%start("sweep_loop")
  do sweep = 0, (na-1)/nbw

    current_n = na - sweep*nbw
    call determine_workload(obj,current_n, nbw, np_rows, limits)
    current_n_start = limits(my_prow)
    current_n_end   = limits(my_prow+1)
    current_local_n = current_n_end - current_n_start

    next_n = max(current_n - nbw, 0)
    call determine_workload(obj,next_n, nbw, np_rows, limits)
    next_n_start = limits(my_prow)
    next_n_end   = limits(my_prow+1)
    next_local_n = next_n_end - next_n_start

    if (next_n_end < next_n) then
      bottom_msg_length = current_n_end - next_n_end
    else
      bottom_msg_length = 0
    endif

    if (next_local_n > 0) then
      next_top_msg_length = current_n_start - next_n_start
    else
      next_top_msg_length = 0
    endif

    if (sweep==0 .and. current_n_end < current_n .and. l_nev > 0) then
#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
      if (wantDebug) call obj%timer%start("cuda_mpi_communication")
#else
      if (wantDebug) call obj%timer%start("host_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
      do i = 1, stripe_count

#ifdef WITH_OPENMP_TRADITIONAL
        csw = min(stripe_width, thread_width-(i-1)*stripe_width) ! "current_stripe_width"
        b_len = csw*nbw*max_threads

        if (useGPU) then
#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
          call MPI_Irecv(bottom_border_recv_buffer_mpi_fortran_ptr(1,i), int(nbw*stripe_width,kind=MPI_KIND), &
          !call MPI_Irecv(bottom_border_recv_buffer_mpi_fortran_ptr(1,i), int(b_len,kind=MPI_KIND), &
                         MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), &
                         int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND),      &
                         bottom_recv_request(i), mpierr)
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
          call MPI_Irecv(bottom_border_recv_buffer(1,i), int(nbw*stripe_width,kind=MPI_KIND), &
          !call MPI_Irecv(bottom_border_recv_buffer(1,i), int(b_len,kind=MPI_KIND), &
                         MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), &
                         int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND),      &
                         bottom_recv_request(i), mpierr)
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
        else ! useGPU
#ifdef WITH_MPI
          call MPI_Irecv(bottom_border_recv_buffer(1,i), int(b_len,kind=MPI_KIND), &
                         MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), &
                         int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), &
                         bottom_recv_request(i), mpierr)
#endif /* WITH_MPI */
        endif !useGPU
#ifndef WITH_MPI
!              carefull the "recieve" has to be done at the corresponding wait or send
!              bottom_border_recv_buffer(1:csw*nbw*max_threads,i) = top_border_send_buffer(1:csw*nbw*max_threads,i)
#endif /* WITH_MPI */


#else /* WITH_OPENMP_TRADITIONAL */

        if (useGPU) then
#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
          call MPI_Irecv(bottom_border_recv_buffer_mpi_fortran_ptr(1,i), int(nbw*stripe_width,kind=MPI_KIND), &
                         MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), &
                         int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND),      &
                         bottom_recv_request(i), mpierr)
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
          call MPI_Irecv(bottom_border_recv_buffer(1,i), int(nbw*stripe_width,kind=MPI_KIND), &
                         MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), &
                         int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND),      &
                         bottom_recv_request(i), mpierr)
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
        else !useGPU
#ifdef WITH_MPI
          call MPI_Irecv(bottom_border_recv_buffer(1,i), int(nbw*stripe_width,kind=MPI_KIND), &
                         MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), &
                         int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND),      &
                         bottom_recv_request(i), mpierr)
#endif /* WITH_MPI */
        endif !useGPU
#ifndef WITH_MPI
!            carefull the recieve has to be done at the corresponding wait or send
!            bottom_border_recv_buffer(1:nbw*stripe_width,1,i) = top_border_send_buffer(1:nbw*stripe_width,1,i)
#endif /* WITH_MPI */
#endif /* WITH_OPENMP_TRADITIONAL */

      enddo ! i = 1, stripe_count
#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
      if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#else
      if (wantDebug) call obj%timer%stop("host_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
    endif ! sweep==0 .and. current_n_end < current_n .and. l_nev > 0

    if (current_local_n > 1) then
      if (useGPU .and. allComputeOnGPU) then
        if (my_pcol == mod(sweep,np_cols)) then
          if (wantDebug) call obj%timer%start("cuda_memcpy")
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)

          successGPU =  gpu_memcpy_async(c_loc(bcast_buffer_mpi_fortran_ptr(1,1)), &
                                   c_loc(hh_trans_mpi_fortran_ptr(1,current_tv_off+1)),  &
                                     size(hh_trans,dim=1) * (current_tv_off+current_local_n-(current_tv_off+1)+1) * &   
                                     size_of_datatype, &
                                     gpuMemcpyDeviceToDevice, my_stream)
          check_memcpy_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)
          ! synchronize streamsPerThread; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)
#else
          successGPU =  gpu_memcpy(c_loc(bcast_buffer_mpi_fortran_ptr(1,1)), &
                                   c_loc(hh_trans_mpi_fortran_ptr(1,current_tv_off+1)),  &
                                     size(hh_trans,dim=1) * (current_tv_off+current_local_n-(current_tv_off+1)+1) * &
                                     size_of_datatype, &
                                     gpuMemcpyDeviceToDevice)
          check_memcpy_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)
#endif
          if (wantDebug) call obj%timer%stop("cuda_memcpy")
          current_tv_off = current_tv_off + current_local_n
        endif
      else !useGPU
        if (my_pcol == mod(sweep,np_cols)) then
          bcast_buffer(:,1:current_local_n) =    &
          hh_trans(:,current_tv_off+1:current_tv_off+current_local_n)
          current_tv_off = current_tv_off + current_local_n
        endif
      endif ! useGPU

#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
      if (.not.(allComputeOnGPU)) then
        if (wantDebug) call obj%timer%start("cuda_memcpy")
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)

        successGPU =  gpu_memcpy_async(bcast_buffer_dev, int(loc(bcast_buffer(1,1)),kind=c_intptr_t),  &
                                   nbw * current_local_n *    &
                                   size_of_datatype, &
                                   gpuMemcpyHostToDevice, my_stream)
        check_memcpy_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)
#else
        successGPU =  gpu_memcpy(bcast_buffer_dev, int(loc(bcast_buffer(1,1)),kind=c_intptr_t),  &
                                   nbw * current_local_n *    &
                                   size_of_datatype, &
                                   gpuMemcpyHostToDevice)
        check_memcpy_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)
#endif
        if (wantDebug) call obj%timer%stop("cuda_memcpy")
      endif

      if (useNonBlockingCollectivesCols) then
        if (wantDebug) call obj%timer%start("cuda_mpi_nbc_communication")
        call mpi_ibcast(bcast_buffer_mpi_fortran_ptr, int(nbw*current_local_n,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                       int(mod(sweep,np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), bcast_request1, mpierr)
        call mpi_wait(bcast_request1, MPI_STATUS_IGNORE, mpierr)
        if (wantDebug) call obj%timer%stop("cuda_mpi_nbc_communication")
      else
        if (wantDebug) call obj%timer%start("cuda_mpi_communication")
        call mpi_bcast(bcast_buffer_mpi_fortran_ptr, int(nbw*current_local_n,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                       int(mod(sweep,np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
        if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
      endif
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
      if (useNonBlockingCollectivesCols) then
        if (wantDebug) call obj%timer%start("mpi_nbc_communication")
        call mpi_ibcast(bcast_buffer, int(nbw*current_local_n,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                     int(mod(sweep,np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), bcast_request1, mpierr)
        call mpi_wait(bcast_request1, MPI_STATUS_IGNORE, mpierr)
        if (wantDebug) call obj%timer%stop("mpi_nbc_communication")
      else
        if (wantDebug) call obj%timer%start("mpi_communication")
        call mpi_bcast(bcast_buffer, int(nbw*current_local_n,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                     int(mod(sweep,np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
        if (wantDebug) call obj%timer%stop("mpi_communication")
      endif

      if (useGPU .and. .not.(allComputeOnGPU)) then
        if (wantDebug) call obj%timer%start("memcpy")
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)

        successGPU =  gpu_memcpy_async(bcast_buffer_dev, int(loc(bcast_buffer(1,1)),kind=c_intptr_t),  &
                                 nbw * current_local_n *    &
                                 size_of_datatype, &
                                 gpuMemcpyHostToDevice, my_stream)
        check_memcpy_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)
#else
        successGPU =  gpu_memcpy(bcast_buffer_dev, int(loc(bcast_buffer(1,1)),kind=c_intptr_t),  &
                                 nbw * current_local_n *    &
                                 size_of_datatype, &
                                 gpuMemcpyHostToDevice)
        check_memcpy_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)
#endif
        if (wantDebug) call obj%timer%stop("memcpy")
      endif ! useGPU
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#else /* WITH_MPI */
      if (useGPU .and. .not.(allComputeOnGPU)) then
        if (wantDebug) call obj%timer%start("memcpy")
#ifdef WITH_GPU_STREAMS
        successGPU =  gpu_memcpy_async(bcast_buffer_dev, int(loc(bcast_buffer(1,1)),kind=c_intptr_t),  &
                                 nbw * current_local_n *    &
                                 size_of_datatype, &
                                 gpuMemcpyHostToDevice, my_stream)
        check_memcpy_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)
#else
        successGPU =  gpu_memcpy(bcast_buffer_dev, int(loc(bcast_buffer(1,1)),kind=c_intptr_t),  &
                                 nbw * current_local_n *    &
                                 size_of_datatype, &
                                 gpuMemcpyHostToDevice)
        check_memcpy_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)
#endif
        if (wantDebug) call obj%timer%stop("memcpy")
      endif ! useGPU
#endif /* WITH_MPI */

      if (useGPU) then
        if (wantDebug) call obj%timer%start("extract_hh")
        call extract_hh_tau_&
             &MATH_DATATYPE&
             &_gpu_&
             &PRECISION&
             (bcast_buffer_dev, hh_tau_dev, nbw, &
             current_local_n, .false., my_stream)
        if (wantDebug) call obj%timer%stop("extract_hh")
      endif ! useGPU

    else ! (current_local_n > 1) then

      ! for current_local_n == 1 the one and only HH Vector is 0 and not stored in hh_trans_real/complex
      bcast_buffer(:,1) = 0.0_rck
      if (useGPU) then
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
        if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif

#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memset_async(bcast_buffer_dev, 0, nbw * size_of_datatype, my_stream)
          check_memset_gpu("tridi_to_band: bcast_buffer_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("tridi_to_band: bcast_buffer_dev", successGPU)
#else
          successGPU = gpu_memset(bcast_buffer_dev, 0, nbw * size_of_datatype)
          check_memset_gpu("tridi_to_band: bcast_buffer_dev", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
        else
          successGPU = gpu_memcpy(bcast_buffer_dev, int(loc(bcast_buffer(1,1)),kind=c_intptr_t), &
                              nbw*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("tridi_to_band: bcast_buffer -> bcast_buffer_dev", successGPU)
        endif
#endif

        if (wantDebug) call obj%timer%start("extract_hh")
        call extract_hh_tau_&
             &MATH_DATATYPE&
             &_gpu_&
             &PRECISION&
             &( &
             bcast_buffer_dev, hh_tau_dev, &
             nbw, 1, .true., my_stream)
        if (wantDebug) call obj%timer%stop("extract_hh")
      endif ! useGPU
    endif ! (current_local_n > 1) then

    if (l_nev == 0) cycle

    if (current_local_n > 0) then

      do i = 1, stripe_count
#ifdef WITH_OPENMP_TRADITIONAL

        ! Get real stripe width for strip i;
        ! The last OpenMP tasks may have an even smaller stripe with,
        ! but we don't care about this, i.e. we send/recv a bit too much in this case.
        ! csw: current_stripe_width

        csw = min(stripe_width, thread_width-(i-1)*stripe_width)
#endif /* WITH_OPENMP_TRADITIONAL */

        !wait_b
        if (current_n_end < current_n) then


#ifdef WITH_OPENMP_TRADITIONAL

#ifdef WITH_MPI
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
          if (wantDebug) call obj%timer%start("host_mpi_wait_bottom_recv")
#else
          if (wantDebug) call obj%timer%start("cuda_mpi_wait_bottom_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */

          call MPI_Wait(bottom_recv_request(i), MPI_STATUS_IGNORE, mpierr)

#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
          if (wantDebug) call obj%timer%stop("host_mpi_wait_bottom_recv")
#else
          if (wantDebug) call obj%timer%stop("cuda_mpi_wait_bottom_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
          n_off = current_local_n+a_off

          if (useGPU) then
            if (allComputeOnGPU) then
              if (wantDebug) call obj%timer%start("cuda_memcpy")
              !$omp parallel do num_threads(max_threads) if (max_threads>1) &
              !$omp default(none) &
              !$omp private(my_thread, n_off, b_len, b_off, successGPU, dev_offset) &
              !$omp shared(max_threads, current_local_n, a_off, csw, nbw, &
              !$omp&       i, gpuMemcpyDeviceToDevice, my_stream, &
              !$omp&       aIntern_mpi_fortran_ptr, bottom_border_recv_buffer_mpi_fortran_ptr) &
              !$omp schedule(static, 1)
              do my_thread = 1, max_threads
                n_off = current_local_n+a_off
                b_len = csw*nbw
                b_off = (my_thread-1)*b_len
                ! check this
#ifdef WITH_GPU_STREAMS
                successGPU = gpu_stream_synchronize(my_stream)
                check_stream_synchronize_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)

                successGPU =  gpu_memcpy_async(c_loc(aIntern_mpi_fortran_ptr(1,n_off+1,i,my_thread)), &
                                       c_loc(bottom_border_recv_buffer_mpi_fortran_ptr(b_off+1,i)),  &
                                       csw*nbw* size_of_datatype,      &
                                       gpuMemcpyDeviceToDevice, my_stream)
                check_memcpy_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)

                successGPU = gpu_stream_synchronize(my_stream)
                check_stream_synchronize_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)
                ! synchronize streamsPerThread; maybe not neccessary
                successGPU = gpu_stream_synchronize()
                check_stream_synchronize_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)
#else
                successGPU =  gpu_memcpy(c_loc(aIntern_mpi_fortran_ptr(1,n_off+1,i,my_thread)), &
                                       c_loc(bottom_border_recv_buffer_mpi_fortran_ptr(b_off+1,i)),  &
                                       csw*nbw* size_of_datatype,      &
                                       gpuMemcpyDeviceToDevice)
                check_memcpy_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)
#endif
              enddo
              !$omp end parallel do
              if (wantDebug) call obj%timer%stop("cuda_memcpy")
            else ! allComputeOnGPU
              if (wantDebug) call obj%timer%start("memcpy")
              !$omp parallel do num_threads(max_threads) if (max_threads>1) &
              !$omp default(none) &
              !$omp private(my_thread, n_off, b_len, b_off, successGPU, dev_offset) &
              !$omp shared(max_threads, current_local_n, a_off, csw, nbw, aIntern_dev, &
              !$omp&       i, bottom_border_recv_buffer, gpuMemcpyHostToDevice, my_stream, &
              !$omp&       stripe_width, a_dim2, stripe_count) &
              !$omp schedule(static, 1)
              do my_thread = 1, max_threads
                n_off = current_local_n+a_off
                b_len = csw*nbw
                b_off = (my_thread-1)*b_len
                ! check this
                dev_offset = (0 + (n_off * stripe_width) + ( (i-1) * stripe_width *a_dim2 ) + &
                            (my_thread-1)*stripe_width *a_dim2*stripe_count ) * size_of_datatype
#ifdef WITH_GPU_STREAMS
                successGPU = gpu_stream_synchronize(my_stream)
                check_stream_synchronize_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)

                successGPU =  gpu_memcpy_async( aIntern_dev + dev_offset , &
                                       int(loc(bottom_border_recv_buffer(b_off+1,i)),kind=c_intptr_t), &
                                       csw*nbw*  size_of_datatype,    &
                                       gpuMemcpyHostToDevice, my_stream)
                check_memcpy_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)

                successGPU = gpu_stream_synchronize(my_stream)
                check_stream_synchronize_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)
                ! synchronize streamsPerThread; maybe not neccessary
                successGPU = gpu_stream_synchronize()
                check_stream_synchronize_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)
#else
                successGPU =  gpu_memcpy( aIntern_dev + dev_offset , &
                                       int(loc(bottom_border_recv_buffer(b_off+1,i)),kind=c_intptr_t), &
                                       csw*nbw*  size_of_datatype,    &
                                       gpuMemcpyHostToDevice)
                check_memcpy_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)
#endif
              enddo
              !$omp end parallel do
              if (wantDebug) call obj%timer%stop("memcpy")
            endif ! allComputeOnGPU
          else ! useGPU
            call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)
            !$omp parallel do num_threads(max_threads) if(max_threads > 1) &
            !$omp default(none) &
            !$omp private(my_thread, n_off, b_len, b_off) &
            !$omp shared(max_threads, current_local_n, a_off, csw, nbw, aIntern, &
            !$omp&       i, bottom_border_recv_buffer) &
            !$omp schedule(static, 1)
            do my_thread = 1, max_threads
              n_off = current_local_n+a_off
              b_len = csw*nbw
              b_off = (my_thread-1)*b_len
              aIntern(1:csw,n_off+1:n_off+nbw,i,my_thread) = &
                reshape(bottom_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, nbw /))
            enddo
            !$omp end parallel do
            call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)
          endif ! useGPU
#else /* WITH_OPENMP_TRADITIONAL */

#ifdef WITH_MPI
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
          if (wantDebug) call obj%timer%start("host_mpi_wait_bottom_recv")
#else
          if (wantDebug) call obj%timer%start("cuda_mpi_wait_bottom_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
          call MPI_Wait(bottom_recv_request(i), MPI_STATUS_IGNORE, mpierr)

#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
          if (wantDebug) call obj%timer%stop("host_mpi_wait_bottom_recv")
#else
          if (wantDebug) call obj%timer%stop("cuda_mpi_wait_bottom_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
          n_off = current_local_n+a_off

          if (useGPU) then
            if (allComputeOnGPU) then
              if (wantDebug) call obj%timer%start("cuda_memcpy")
#ifdef WITH_GPU_STREAMS
              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)

              successGPU =  gpu_memcpy_async(c_loc(aIntern_mpi_fortran_ptr(1,n_off+1,i)), &
                                       c_loc(bottom_border_recv_buffer_mpi_fortran_ptr(1,i)),  &
                                       stripe_width*nbw* size_of_datatype,      &
                                       gpuMemcpyDeviceToDevice, my_stream)
              check_memcpy_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)

              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)
              ! synchronize streamsPerThread; maybe not neccessary
              successGPU = gpu_stream_synchronize()
              check_stream_synchronize_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)
#else
              successGPU =  gpu_memcpy(c_loc(aIntern_mpi_fortran_ptr(1,n_off+1,i)), &
                                       c_loc(bottom_border_recv_buffer_mpi_fortran_ptr(1,i)),  &
                                       stripe_width*nbw* size_of_datatype,      &
                                       gpuMemcpyDeviceToDevice)
              check_memcpy_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)
#endif
              if (wantDebug) call obj%timer%stop("cuda_memcpy")
            else ! allComputeOnGPU
              if (wantDebug) call obj%timer%start("memcpy")
              dev_offset = (0 + (n_off * stripe_width) + ( (i-1) * stripe_width *a_dim2 )) * size_of_datatype
#ifdef WITH_GPU_STREAMS 
              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)

              successGPU =  gpu_memcpy_async( aIntern_dev + dev_offset , &
                                      int(loc(bottom_border_recv_buffer(1,i)),kind=c_intptr_t), &
                                       stripe_width*nbw*  size_of_datatype,    &
                                       gpuMemcpyHostToDevice, my_stream)
              check_memcpy_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)

              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)
              ! synchronize streamsPerThread; maybe not neccessary
              successGPU = gpu_stream_synchronize()
              check_stream_synchronize_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)
#else
              successGPU =  gpu_memcpy( aIntern_dev + dev_offset , &
                                      int(loc(bottom_border_recv_buffer(1,i)),kind=c_intptr_t), &
                                       stripe_width*nbw*  size_of_datatype,    &
                                       gpuMemcpyHostToDevice)
              check_memcpy_gpu("tridi_to_band: bottom_border_recv_buffer -> aIntern_dev", successGPU)
#endif
              if (wantDebug) call obj%timer%stop("memcpy")
            endif ! allComputeOnGPU
          else ! useGPU
            aIntern(:,n_off+1:n_off+nbw,i) = reshape( &
            bottom_border_recv_buffer(1:stripe_width*nbw,i),(/stripe_width,nbw/))
          endif ! useGPU

#endif /* WITH_OPENMP_TRADITIONAL */

          if (next_n_end < next_n) then

#ifdef WITH_OPENMP_TRADITIONAL
            if (useGPU) then
#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
              if (wantDebug) call obj%timer%start("cuda_mpi_communication")
              call MPI_Irecv(bottom_border_recv_buffer_mpi_fortran_ptr(1,i), int(nbw*stripe_width,kind=MPI_KIND), &
              !call MPI_Irecv(bottom_border_recv_buffer_mpi_fortran_ptr(1,i), int(csw*nbw*max_threads,kind=MPI_KIND), &
                           MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), &
                           int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND),      &
                           bottom_recv_request(i), mpierr)
              if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
              if (wantDebug) call obj%timer%start("host_mpi_communication")
              call MPI_Irecv(bottom_border_recv_buffer(1,i), int(nbw*stripe_width,kind=MPI_KIND), &
              !call MPI_Irecv(bottom_border_recv_buffer(1,i), int(csw*nbw*max_threads,kind=MPI_KIND), &
                           MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), &
                           int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND),      &
                           bottom_recv_request(i), mpierr)
              if (wantDebug) call obj%timer%stop("host_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
            else !useGPU
#ifdef WITH_MPI
              if (wantDebug) call obj%timer%start("mpi_communication")
              call MPI_Irecv(bottom_border_recv_buffer(1,i), int(csw*nbw*max_threads,kind=MPI_KIND), &
                          MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), &
                          int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND),      &
                          bottom_recv_request(i), mpierr)
              if (wantDebug) call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */
            endif !useGPU
#ifndef WITH_MPI
!                carefull the recieve has to be done at the corresponding wait or send
!                bottom_border_recv_buffer(1:csw*nbw*max_threads,i) = top_border_send_buffer(1:csw*nbw*max_threads,i)
#endif /* WITH_MPI */

#else /* WITH_OPENMP_TRADITIONAL */
            if (useGPU) then
#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
              if (wantDebug) call obj%timer%start("cuda_mpi_communication")
              call MPI_Irecv(bottom_border_recv_buffer_mpi_fortran_ptr(1,i), int(nbw*stripe_width,kind=MPI_KIND), &
                            MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), &
                            int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND),      &
                            bottom_recv_request(i), mpierr)
              if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
              if (wantDebug) call obj%timer%start("host_mpi_communication")
              call MPI_Irecv(bottom_border_recv_buffer(1,i), int(nbw*stripe_width,kind=MPI_KIND), &
                            MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), &
                            int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND),      &
                            bottom_recv_request(i), mpierr)
              if (wantDebug) call obj%timer%stop("host_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
            else ! useGPU
#ifdef WITH_MPI
              if (wantDebug) call obj%timer%start("mpi_communication")
              call MPI_Irecv(bottom_border_recv_buffer(1,i), int(nbw*stripe_width,kind=MPI_KIND), &
                            MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), &
                            int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND),      &
                            bottom_recv_request(i), mpierr)
              if (wantDebug) call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */
            endif ! useGPU
#ifndef WITH_MPI
!!                carefull the recieve has to be done at the corresponding wait or send
!!                bottom_border_recv_buffer(1:stripe_width,1:nbw,i) =  top_border_send_buffer(1:stripe_width,1:nbw,i)
#endif /* WITH_MPI */

#endif /* WITH_OPENMP_TRADITIONAL */
            endif ! (next_n_end < next_n)
          endif ! (current_n_end < current_n)

          if (current_local_n <= bottom_msg_length + top_msg_length) then

            !wait_t
            if (top_msg_length>0) then

#ifdef WITH_OPENMP_TRADITIONAL
#ifdef WITH_MPI
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
              if (wantDebug) call obj%timer%start("host_mpi_wait_top_recv")
#else
              if (wantDebug) call obj%timer%start("cuda_mpi_wait_top_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
              call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
              if (wantDebug) call obj%timer%stop("host_mpi_wait_top_recv")
#else
              if (wantDebug) call obj%timer%stop("cuda_mpi_wait_top_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
              if (useGPU) then
                if (allComputeOnGPU) then
                  ! the MPI_IRECV will be done CUDA_AWARE we thus do not need a host to device copy
                  ! However, we have to copy from top_border_recv_buffer_mpi_fortran_ptr to aIntern_mpi_fortran_ptr

                  if (wantDebug) call obj%timer%start("cuda_memcpy")
                  !$omp parallel do num_threads(max_threads) if (max_threads>1) &
                  !$omp default(none) &
                  !$omp private(my_thread, b_len, b_off, successGPU) &
                  !$omp shared(max_threads, csw, top_msg_length, aIntern_mpi_fortran_ptr, a_off, i, &
                  !$omp&       top_border_recv_buffer_mpi_fortran_ptr, gpuMemcpyDeviceToDevice, my_stream) &
                  !$omp        schedule(static, 1)
                  do my_thread = 1, max_threads
                    b_len = csw*top_msg_length
                    b_off = (my_thread-1)*b_len
                    !Fortran pointer for indexing
                    ! check this
#ifdef WITH_GPU_STREAMS
                    successGPU = gpu_stream_synchronize(my_stream)
                    check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                    successGPU =  gpu_memcpy_async(c_loc(aIntern_mpi_fortran_ptr(1,a_off+1,i,my_thread)), &
                                           c_loc(top_border_recv_buffer_mpi_fortran_ptr(b_off+1,i)),  &
                                           csw*top_msg_length* size_of_datatype,      &
                                           gpuMemcpyDeviceToDevice, my_stream)
                    check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                    successGPU = gpu_stream_synchronize(my_stream)
                    check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
                    ! synchronize streamsPerThread; maybe not neccessary
                    successGPU = gpu_stream_synchronize()
                    check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
#else
                    successGPU =  gpu_memcpy(c_loc(aIntern_mpi_fortran_ptr(1,a_off+1,i,my_thread)), &
                                           c_loc(top_border_recv_buffer_mpi_fortran_ptr(b_off+1,i)),  &
                                           csw*top_msg_length* size_of_datatype,      &
                                           gpuMemcpyDeviceToDevice)
                    check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
#endif
                  enddo
                  !$omp end parallel do
                  if (wantDebug) call obj%timer%stop("cuda_memcpy")
                else ! allComputeOnGPU
                  if (wantDebug) call obj%timer%start("memcpy")
                  !$omp parallel do num_threads(max_threads) if (max_threads>1) &
                  !$omp default(none) &
                  !$omp private(my_thread, b_len, b_off, successGPU, dev_offset) &
                  !$omp shared(max_threads, csw, top_msg_length, aIntern_dev, &
                  !$omp&       a_off, stripe_width, i, a_dim2, stripe_count,  &
                  !$omp&       top_border_recv_buffer, gpuMemcpyHostToDevice, my_stream) &
                  !$omp        schedule(static, 1)
                  do my_thread = 1, max_threads
                    ! check this
                    b_len = csw*top_msg_length
                    b_off = (my_thread-1)*b_len

                    dev_offset = (0 + ((a_off) * stripe_width) + ( (i-1) * stripe_width * a_dim2 ) + &
                                  (my_thread-1)*stripe_width * a_dim2 * stripe_count) * size_of_datatype
#ifdef WITH_GPU_STREAMS
                    successGPU = gpu_stream_synchronize(my_stream)
                    check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                    successGPU =  gpu_memcpy_async( aIntern_dev+dev_offset , int(loc(top_border_recv_buffer(b_off+1,i)), &
                                  kind=c_intptr_t),  &
                                               csw*top_msg_length* size_of_datatype,      &
                                               gpuMemcpyHostToDevice, my_stream)
                    check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                    successGPU = gpu_stream_synchronize(my_stream)
                    check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
                    ! synchronize streamsPerThread; maybe not neccessary
                    successGPU = gpu_stream_synchronize()
                    check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

#else
                    successGPU =  gpu_memcpy( aIntern_dev+dev_offset , int(loc(top_border_recv_buffer(b_off+1,i)), &
                                  kind=c_intptr_t),  &
                                               csw*top_msg_length* size_of_datatype,      &
                                               gpuMemcpyHostToDevice)
                    check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
#endif
                  enddo
                  !$omp end parallel do
                  if (wantDebug) call obj%timer%stop("memcpy")
                endif ! allComputeOnGPU
              else ! useGPU
                call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)

                !$omp parallel do num_threads(max_threads) if(max_threads > 1) &
                !$omp default(none) &
                !$omp private(my_thread, n_off, b_len, b_off) &
                !$omp shared(max_threads, csw, top_msg_length, aIntern, &
                !$omp&       i, top_border_recv_buffer, a_off) &
                !$omp        schedule(static, 1)
                do my_thread = 1, max_threads
                  b_len = csw*top_msg_length
                  b_off = (my_thread-1)*b_len
                  aIntern(1:csw,a_off+1:a_off+top_msg_length,i,my_thread) = &
                  reshape(top_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, top_msg_length /))
                enddo
                !$omp end parallel do
                call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)

              endif ! useGPU

#else /* WITH_OPENMP_TRADITIONAL */

#ifdef WITH_MPI
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
              if (wantDebug) call obj%timer%start("host_mpi_wait_top_recv")
#else
              if (wantDebug) call obj%timer%start("cuda_mpi_wait_top_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
              call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
              if (wantDebug) call obj%timer%stop("host_mpi_wait_top_recv")
#else
              if (wantDebug) call obj%timer%stop("cuda_mpi_wait_top_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */

              if (useGPU) then
                if (allComputeOnGPU) then
                  ! the MPI_IRECV will be done CUDA_AWARE we thus do not need a host to device copy
                  ! However, we have to copy from top_border_recv_buffer_mpi_fortran_ptr to aIntern_mpi_fortran_ptr
                  if (wantDebug) call obj%timer%start("cuda_memcpy")

                  !Fortran pointer for indexing
#ifdef WITH_GPU_STREAMS
                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                  successGPU =  gpu_memcpy_async(c_loc(aIntern_mpi_fortran_ptr(1,a_off+1,i)), &
                                          c_loc(top_border_recv_buffer_mpi_fortran_ptr(1,i)),  &
                                          stripe_width*top_msg_length* size_of_datatype,      &
                                          gpuMemcpyDeviceToDevice, my_stream)
                  check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
                  ! synchronize streamsPerThread; maybe not neccessary
                  successGPU = gpu_stream_synchronize()
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

#else
                  successGPU =  gpu_memcpy(c_loc(aIntern_mpi_fortran_ptr(1,a_off+1,i)), &
                                          c_loc(top_border_recv_buffer_mpi_fortran_ptr(1,i)),  &
                                          stripe_width*top_msg_length* size_of_datatype,      &
                                          gpuMemcpyDeviceToDevice)
                  check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
#endif
                  if (wantDebug) call obj%timer%stop("cuda_memcpy")
                else ! allComputeOnGPU
                  if (wantDebug) call obj%timer%start("memcpy")
                  dev_offset = (0 + (a_off * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) * size_of_datatype
                  !             host_offset= (0 + (0 * stripe_width) + ( (i-1) * stripe_width * nbw ) ) * 8
#ifdef WITH_GPU_STREAMS
                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                  successGPU =  gpu_memcpy_async( aIntern_dev+dev_offset , int(loc(top_border_recv_buffer(1,i)),kind=c_intptr_t), &
                                             stripe_width*top_msg_length* size_of_datatype,      &
                                             gpuMemcpyHostToDevice, my_stream)
                  check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
                  ! synchronize streamsPerThread; maybe not neccessary
                  successGPU = gpu_stream_synchronize()
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
#else
                  successGPU =  gpu_memcpy( aIntern_dev+dev_offset , int(loc(top_border_recv_buffer(1,i)),kind=c_intptr_t),  &
                                             stripe_width*top_msg_length* size_of_datatype,      &
                                             gpuMemcpyHostToDevice)
                  check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
#endif
                  if (wantDebug) call obj%timer%stop("memcpy")
                endif ! allComputeOnGPU
              else ! useGPU
                aIntern(:,a_off+1:a_off+top_msg_length,i) = &
                reshape(top_border_recv_buffer(1:stripe_width*top_msg_length,i),(/stripe_width,top_msg_length/))
              endif ! useGPU
#endif /* WITH_OPENMP_TRADITIONAL */
            endif ! top_msg_length

            !compute
#ifdef WITH_OPENMP_TRADITIONAL
            if (useGPU) then
              if (wantDebug) call obj%timer%start("compute_hh_trafo")
              my_thread = 1 ! for the moment dummy variable
              thread_width2 = 1 ! for the moment dummy variable
              call compute_hh_trafo_&
                &MATH_DATATYPE&
                &_openmp_&
                &PRECISION&
                &(obj, my_pe, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, stripe_count, &
                max_threads, &
                l_nev, a_off, nbw, max_blk_size, bcast_buffer, bcast_buffer_dev, &
                hh_tau_dev, kernel_flops, kernel_time, n_times, 0, current_local_n, i, &
                my_thread, thread_width2, kernel, last_stripe_width=last_stripe_width, &
                my_stream=my_stream, success=success)
              if (wantDebug) call obj%timer%stop("compute_hh_trafo")
              if (.not.success) then
                success=.false.
                return
               endif      
            else ! useGPU
              call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)

              !$omp parallel do num_threads(max_threads) if(max_threads > 1) &
              !$omp default(none) &
              !$omp private(my_thread, n_off, b_len, b_off, success) &
              !$omp shared(max_threads, csw, top_msg_length, aIntern, my_pe, &
              !$omp&       a_off, i, top_border_recv_buffer, obj, useGPU, wantDebug, aIntern_dev, &
              !$omp&       stripe_width, a_dim2, stripe_count, l_nev, nbw, max_blk_size, bcast_buffer, &
              !$omp&       bcast_buffer_dev, hh_tau_dev, kernel_flops, kernel_time, n_times, current_local_n, &
              !$omp&       thread_width, kernel, my_stream) &
              !$omp        schedule(static, 1)
              do my_thread = 1, max_threads
                call compute_hh_trafo_&
                      &MATH_DATATYPE&
                      &_openmp_&
                      &PRECISION&
                      (obj, my_pe, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, &
                      stripe_count, max_threads, &
                      l_nev, a_off, nbw, max_blk_size, bcast_buffer, bcast_buffer_dev, &
                      hh_tau_dev, kernel_flops, kernel_time, n_times, 0, current_local_n, &
                      i, my_thread, thread_width, kernel, my_stream=my_stream, success=success)
              enddo
              !$omp end parallel do
              call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)

            endif ! useGPU

#else /* WITH_OPENMP_TRADITIONAL */

            if (wantDebug) call obj%timer%start("compute_hh_trafo")
            call compute_hh_trafo_&
                &MATH_DATATYPE&
                &_&
                &PRECISION&
                &(obj, my_pe, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, stripe_count, &
                max_threads, &
                a_off, nbw, max_blk_size, bcast_buffer, bcast_buffer_dev, &
                hh_tau_dev, kernel_flops, kernel_time, n_times, 0, current_local_n, i, &
                last_stripe_width, kernel, my_stream=my_stream, success=success)
            if (.not.success) then
              success=.false.
              return
            endif
            if (wantDebug) call obj%timer%stop("compute_hh_trafo")
#endif /* WITH_OPENMP_TRADITIONAL */

            !send_b        1
#ifdef WITH_MPI
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
            if (wantDebug) call obj%timer%start("host_mpi_wait_bottom_send")
#else
            if (wantDebug) call obj%timer%start("cuda_mpi_wait_bottom_send")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
            call MPI_Wait(bottom_send_request(i), MPI_STATUS_IGNORE, mpierr)
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
            if (wantDebug) call obj%timer%stop("host_mpi_wait_bottom_send")
#else
            if (wantDebug) call obj%timer%stop("cuda_mpi_wait_bottom_send")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */

            if (bottom_msg_length>0) then
              n_off = current_local_n+nbw-bottom_msg_length+a_off
#ifdef WITH_OPENMP_TRADITIONAL
              b_len = csw*bottom_msg_length*max_threads
              if (useGPU) then
                if (allComputeOnGPU) then
                  ! send should be done on GPU, send_buffer must be created and filled first
                  ! memcpy from aIntern_dev to bottom_border_send_buffer_dev
                  ! either with two offsets or with indexed pointer

                  if (wantDebug) call obj%timer%start("cuda_memcpy")
                  ! nr of threads is assumed to 1
#ifdef WITH_GPU_STREAMS
                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                  successGPU =  gpu_memcpy_async( c_loc(bottom_border_send_buffer_mpi_fortran_ptr(1,i)), &
                                           c_loc(aIntern_mpi_fortran_ptr(1,n_off+1,i,1)), &
                                            stripe_width * bottom_msg_length * size_of_datatype,      &
                                            gpuMemcpyDeviceToDevice, my_stream)
                  check_memcpy_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)

                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)
                  ! synchronize streamsPerThread; maybe not neccessary
                  successGPU = gpu_stream_synchronize()
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)
#else
                  successGPU =  gpu_memcpy( c_loc(bottom_border_send_buffer_mpi_fortran_ptr(1,i)), &
                                           c_loc(aIntern_mpi_fortran_ptr(1,n_off+1,i,1)), &
                                            stripe_width * bottom_msg_length * size_of_datatype,      &
                                            gpuMemcpyDeviceToDevice)
                  check_memcpy_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)
#endif
                  if (wantDebug) call obj%timer%stop("cuda_memcpy")
                else ! allComputeOnGPU
                  if (wantDebug) call obj%timer%start("memcpy")
                  dev_offset = (0 + (n_off * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) * size_of_datatype
#ifdef WITH_GPU_STREAMS
                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                  successGPU =  gpu_memcpy_async( int(loc(bottom_border_send_buffer(1,i)),kind=c_intptr_t), &
                                            aIntern_dev + dev_offset, &
                                            stripe_width * bottom_msg_length * size_of_datatype,      &
                                            gpuMemcpyDeviceToHost, my_stream)
                  check_memcpy_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)

                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)
                  ! synchronize streamsPerThread; maybe not neccessary
                  successGPU = gpu_stream_synchronize()
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)
#else
                  successGPU =  gpu_memcpy( int(loc(bottom_border_send_buffer(1,i)),kind=c_intptr_t), aIntern_dev + dev_offset, &
                                            stripe_width * bottom_msg_length * size_of_datatype,      &
                                            gpuMemcpyDeviceToHost)
                  check_memcpy_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)
#endif
                  if (wantDebug) call obj%timer%stop("memcpy")
                endif ! allComputeOnGPU

#ifdef WITH_MPI
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
                if (wantDebug) call obj%timer%start("host_mpi_communication")
                call MPI_Isend(bottom_border_send_buffer(1,i), int(bottom_msg_length*stripe_width,kind=MPI_KIND),  &
                !call MPI_Isend(bottom_border_send_buffer(1,i), int(b_len,kind=MPI_KIND),  &
                     MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), &
                     int(mpi_comm_rows,kind=MPI_KIND), bottom_send_request(i), mpierr)
                if (wantDebug) call obj%timer%stop("host_mpi_communication")
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
                if (wantDebug) call obj%timer%start("cuda_mpi_communication")
                call MPI_Isend(bottom_border_send_buffer_mpi_fortran_ptr(1,i), int(bottom_msg_length*stripe_width,kind=MPI_KIND), &
                !call MPI_Isend(bottom_border_send_buffer_mpi_fortran_ptr(1,i), int(b_len,kind=MPI_KIND),  &
                    MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), &
                    int(mpi_comm_rows,kind=MPI_KIND), bottom_send_request(i), mpierr)
                if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */

              else !useGPU
                bottom_border_send_buffer(1:b_len,i) = &
                reshape(aIntern(1:csw,n_off+1:n_off+bottom_msg_length,i,:), (/ b_len /))
#ifdef WITH_MPI
                if (wantDebug) call obj%timer%start("mpi_communication")
                call MPI_Isend(bottom_border_send_buffer(1,i), int(b_len,kind=MPI_KIND),  &
                     MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), &
                     int(mpi_comm_rows,kind=MPI_KIND), bottom_send_request(i), mpierr)
                if (wantDebug) call obj%timer%stop("mpi_communication")
#endif
              endif !useGPU

#ifndef WITH_MPI
              if (useGPU) then
                if (allComputeOnGPU) then
                  if (next_top_msg_length > 0) then
#ifdef WITH_GPU_STREAMS
              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                   successGPU =  gpu_memcpy_async(c_loc(top_border_recv_buffer_mpi_fortran_ptr(1,i)), &
                                             c_loc(bottom_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                             stripe_width*next_top_msg_length* size_of_datatype,      &
                                             gpuMemcpyDeviceToDevice, my_stream)
              check_memcpy_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)

              successGPU = gpu_stream_synchronize(my_stream)
     check_stream_synchronize_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)
              ! synchronize streamsPerThread; maybe not neccessary
              successGPU = gpu_stream_synchronize()
     check_stream_synchronize_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)
#else
                    successGPU =  gpu_memcpy(c_loc(top_border_recv_buffer_mpi_fortran_ptr(1,i)), &
                                             c_loc(bottom_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                             stripe_width*next_top_msg_length* size_of_datatype,      &
                                             gpuMemcpyDeviceToDevice)
              check_memcpy_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)
#endif

#ifndef WITH_GPU_STREAMS
              if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
              successGPU = gpu_devicesynchronize()
              check_memcpy_gpu("tridi_to_band: device_synchronize", successGPU)
#endif
              if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
                  endif
                else ! allComputeOnGPU
                  if (next_top_msg_length > 0) then
                    top_border_recv_buffer(1:csw*next_top_msg_length*max_threads,i) = &
                            bottom_border_send_buffer(1:csw*next_top_msg_length*max_threads,i)
                  endif
                endif ! allComputeOnGPU
              else ! useGPU
                if (next_top_msg_length > 0) then
                  top_border_recv_buffer(1:csw*next_top_msg_length*max_threads,i) = &
                          bottom_border_send_buffer(1:csw*next_top_msg_length*max_threads,i)
                endif
              endif ! useGPU
#endif /* WITH_MPI */

#else /* WITH_OPENMP_TRADITIONAL */

              if (useGPU) then
                if (allComputeOnGPU) then
                  ! send should be done on GPU, send_buffer must be created and filled first
                  ! memcpy from aIntern_dev to bottom_border_send_buffer_dev
                  ! either with two offsets or with indexed pointer

                  if (wantDebug) call obj%timer%start("cuda_memcpy")
#ifdef WITH_GPU_STREAMS
                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)
                  successGPU =  gpu_memcpy_async( c_loc(bottom_border_send_buffer_mpi_fortran_ptr(1,i)), &
                                           c_loc(aIntern_mpi_fortran_ptr(1,n_off+1,i)), &
                                            stripe_width * bottom_msg_length * size_of_datatype,      &
                                            gpuMemcpyDeviceToDevice, my_stream)
                  check_memcpy_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)

                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)
                  ! synchronize streamsPerThread; maybe not neccessary
                  successGPU = gpu_stream_synchronize()
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)

#else
                  successGPU =  gpu_memcpy( c_loc(bottom_border_send_buffer_mpi_fortran_ptr(1,i)), &
                                           c_loc(aIntern_mpi_fortran_ptr(1,n_off+1,i)), &
                                            stripe_width * bottom_msg_length * size_of_datatype,      &
                                            gpuMemcpyDeviceToDevice)
                  check_memcpy_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)
#endif
                  if (wantDebug) call obj%timer%stop("cuda_memcpy")


#ifndef WITH_GPU_STREAMS
              if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
              successGPU = gpu_devicesynchronize()
              check_memcpy_gpu("tridi_to_band: device_synchronize", successGPU)
              if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
#endif
                else ! allComputeOnGPU
                  if (wantDebug) call obj%timer%start("memcpy")
                  dev_offset = (0 + (n_off * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) * size_of_datatype
#ifdef WITH_GPU_STREAMS
                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)
                  successGPU =  gpu_memcpy_async( int(loc(bottom_border_send_buffer(1,i)),kind=c_intptr_t), &
                                            aIntern_dev + dev_offset, &
                                            stripe_width * bottom_msg_length * size_of_datatype,      &
                                            gpuMemcpyDeviceToHost, my_stream)
                  check_memcpy_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)

                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)
                  ! synchronize streamsPerThread; maybe not neccessary
                  successGPU = gpu_stream_synchronize()
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)

#else
                  successGPU =  gpu_memcpy( int(loc(bottom_border_send_buffer(1,i)),kind=c_intptr_t), &
                                            aIntern_dev + dev_offset, &
                                            stripe_width * bottom_msg_length * size_of_datatype,      &
                                            gpuMemcpyDeviceToHost)
                  check_memcpy_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)
#endif
                  if (wantDebug) call obj%timer%stop("memcpy")
                endif ! allComputeOnGPU

#ifdef WITH_MPI
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
                if (wantDebug) call obj%timer%start("host_mpi_communication")
                call MPI_Isend(bottom_border_send_buffer(1,i), int(bottom_msg_length*stripe_width,kind=MPI_KIND),  &
                      MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), &
                      int(mpi_comm_rows,kind=MPI_KIND), bottom_send_request(i), mpierr)
                if (wantDebug) call obj%timer%stop("host_mpi_communication")
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
                if (wantDebug) call obj%timer%start("cuda_mpi_communication")
                call MPI_Isend(bottom_border_send_buffer_mpi_fortran_ptr(1,i), int(bottom_msg_length*stripe_width,kind=MPI_KIND), &
                    MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), &
                    int(mpi_comm_rows,kind=MPI_KIND), bottom_send_request(i), mpierr)
                if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
              else !useGPU
                bottom_border_send_buffer(1:stripe_width*bottom_msg_length,i) = reshape(&
                aIntern(:,n_off+1:n_off+bottom_msg_length,i),(/stripe_width*bottom_msg_length/))
#ifdef WITH_MPI
                if (wantDebug) call obj%timer%start("mpi_communication")
                call MPI_Isend(bottom_border_send_buffer(1,i), int(bottom_msg_length*stripe_width,kind=MPI_KIND),  &
                     MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), &
                     int(mpi_comm_rows,kind=MPI_KIND), bottom_send_request(i), mpierr)
                if (wantDebug) call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */
              endif !useGPU
#ifndef WITH_MPI
              if (useGPU) then
                if (allComputeOnGPU) then
                  if (next_top_msg_length > 0) then
#ifdef WITH_GPU_STREAMS
                    successGPU = gpu_stream_synchronize(my_stream)
                    check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)
                    successGPU =  gpu_memcpy_async(c_loc(top_border_recv_buffer_mpi_fortran_ptr(1,i)), &
                                             c_loc(bottom_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                             stripe_width*next_top_msg_length* size_of_datatype,      &
                                             gpuMemcpyDeviceToDevice, my_stream)
                    check_memcpy_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)

                    successGPU = gpu_stream_synchronize(my_stream)
                    check_stream_synchronize_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)
                    ! synchronize streamsPerThread; maybe not neccessary
                    successGPU = gpu_stream_synchronize()
                    check_stream_synchronize_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)
#else
                    successGPU =  gpu_memcpy(c_loc(top_border_recv_buffer_mpi_fortran_ptr(1,i)), &
                                             c_loc(bottom_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                             stripe_width*next_top_msg_length* size_of_datatype,      &
                                             gpuMemcpyDeviceToDevice)
                    check_memcpy_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)
#endif
                  endif

#ifndef WITH_GPU_STREAMS
              if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
              successGPU = gpu_devicesynchronize()
              check_memcpy_gpu("tridi_to_band: device_synchronize", successGPU)
              if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
#endif
                else ! allComputeOnGPU
                  if (next_top_msg_length > 0) then
                    top_border_recv_buffer(1:stripe_width*next_top_msg_length,i) =  &
                    bottom_border_send_buffer(1:stripe_width*next_top_msg_length,i)
                  endif
                endif ! allComputeOnGPU
              else ! useGPU
                if (next_top_msg_length > 0) then
                  top_border_recv_buffer(1:stripe_width*next_top_msg_length,i) =  &
                  bottom_border_send_buffer(1:stripe_width*next_top_msg_length,i)
                endif
              endif ! useGPU
#endif /* WITH_MPI */
#endif /* WITH_OPENMP_TRADITIONAL */
            endif !(bottom_msg_length>0)

          else ! current_local_n <= bottom_msg_length + top_msg_length

            !compute
#ifdef WITH_OPENMP_TRADITIONAL
            if (useGPU) then
              my_thread = 1 ! for the moment, dummy variable
              thread_width2 = 1 ! for the moment, dummy variable
              if (wantDebug) call obj%timer%start("compute_hh_trafo")
              call compute_hh_trafo_&
                 &MATH_DATATYPE&
                 &_openmp_&
                 &PRECISION&
                 (obj, my_pe, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, stripe_count, &
                 max_threads, &
                 l_nev, a_off,  nbw, max_blk_size, bcast_buffer, bcast_buffer_dev, &
                 hh_tau_dev, kernel_flops, kernel_time, n_times, &
                 current_local_n - bottom_msg_length, bottom_msg_length, i, &
                 my_thread, thread_width2, kernel, last_stripe_width=last_stripe_width, my_stream=my_stream, &
                 success=success)
              if (.not.success) then
                success=.false.
                return
              endif
              if (wantDebug) call obj%timer%stop("compute_hh_trafo")
            else ! useGPU
              call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)

              !$omp parallel do num_threads(max_threads) if(max_threads > 1) &
              !$omp default(none) &
              !$omp private(my_thread, b_len, b_off, success) &
              !$omp shared(max_threads, obj, useGPU, wantDebug, aIntern, aIntern_dev, my_pe, &
              !$omp&       stripe_width, a_dim2, stripe_count, l_nev, a_off, &
              !$omp&       nbw, max_blk_size, bcast_buffer, bcast_buffer_dev, hh_tau_dev, &
              !$omp&       kernel_flops, kernel_time, n_times, current_local_n, &
              !$omp&       bottom_msg_length, i, thread_width, kernel, my_stream) &
              !$omp schedule(static, 1)
              do my_thread = 1, max_threads

                call compute_hh_trafo_&
                     &MATH_DATATYPE&
                     &_openmp_&
                     &PRECISION&
                     &(obj, my_pe, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, stripe_count, &
                     max_threads, l_nev, a_off, &
                     nbw, max_blk_size,  bcast_buffer, bcast_buffer_dev, &
                     hh_tau_dev, kernel_flops, kernel_time, n_times, current_local_n - bottom_msg_length, &
                     bottom_msg_length, i, my_thread, thread_width, kernel, my_stream=my_stream, success=success)
              enddo
              !$omp end parallel do
              call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)
            endif ! useGPU

            !send_b
#ifdef WITH_MPI
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
            if (wantDebug) call obj%timer%start("host_mpi_wait_bottom_send")
#else
            if (wantDebug) call obj%timer%start("cuda_mpi_wait_bottom_send")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
            call MPI_Wait(bottom_send_request(i), MPI_STATUS_IGNORE, mpierr)
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
            if (wantDebug) call obj%timer%stop("host_mpi_wait_bottom_send")
#else
            if (wantDebug) call obj%timer%stop("cuda_mpi_wait_bottom_send")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
            if (bottom_msg_length > 0) then
              n_off = current_local_n+nbw-bottom_msg_length+a_off
              b_len = csw*bottom_msg_length*max_threads
              if (useGPU) then
                if (allComputeOnGPU) then
                  ! send should be done on GPU, send_buffer must be created and filled first
                  ! memcpy from aIntern_dev to bottom_border_send_buffer_dev
                  ! either with two offsets or with indexed pointer

                  ! nr of threads is assumed to 1
                  if (wantDebug) call obj%timer%start("cuda_memcpy")
#ifdef WITH_GPU_STREAMS
                 successGPU = gpu_stream_synchronize(my_stream)
                 check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)

                  successGPU =  gpu_memcpy_async( c_loc(bottom_border_send_buffer_mpi_fortran_ptr(1,i)), &
                                            c_loc(aIntern_mpi_fortran_ptr(1,n_off+1,i,1)), &
                                             stripe_width * bottom_msg_length * size_of_datatype,      &
                                             gpuMemcpyDeviceToDevice, my_stream)
                  check_memcpy_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)

                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)
                  ! synchronize streamsPerThread; maybe not neccessary
                  successGPU = gpu_stream_synchronize()
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)
#else
                  successGPU =  gpu_memcpy( c_loc(bottom_border_send_buffer_mpi_fortran_ptr(1,i)), &
                                            c_loc(aIntern_mpi_fortran_ptr(1,n_off+1,i,1)), &
                                             stripe_width * bottom_msg_length * size_of_datatype,      &
                                             gpuMemcpyDeviceToDevice)
                  check_memcpy_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)
#endif
                  if (wantDebug) call obj%timer%stop("cuda_memcpy")
                else ! allComputeOnGPU
                  if (wantDebug) call obj%timer%start("memcpy")
                  dev_offset = (0 + (n_off * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) * size_of_datatype
#ifdef WITH_GPU_STREAMS
                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)

                  successGPU =  gpu_memcpy_async(int(loc(bottom_border_send_buffer(1,i)),kind=c_intptr_t), &
                                               aIntern_dev + dev_offset,  &
                                               stripe_width*bottom_msg_length* size_of_datatype,  &
                                               gpuMemcpyDeviceToHost, my_stream)
                  check_memcpy_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)

                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)
                  ! synchronize streamsPerThread; maybe not neccessary
                  successGPU = gpu_stream_synchronize()
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)
#else
                  successGPU =  gpu_memcpy(int(loc(bottom_border_send_buffer(1,i)),kind=c_intptr_t), aIntern_dev + dev_offset,  &
                                               stripe_width*bottom_msg_length* size_of_datatype,  &
                                               gpuMemcpyDeviceToHost)
                  check_memcpy_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)
#endif
                  if (wantDebug) call obj%timer%stop("memcpy")
                endif ! allComputeOnGPU

#ifdef WITH_MPI
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
                if (wantDebug) call obj%timer%start("host_mpi_communication")
                call MPI_Isend(bottom_border_send_buffer(1,i), int(bottom_msg_length*stripe_width,kind=MPI_KIND), &
                !call MPI_Isend(bottom_border_send_buffer(1,i), int(b_len,kind=MPI_KIND), &
                           MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), &
                           int(mpi_comm_rows,kind=MPI_KIND), bottom_send_request(i), mpierr)
                if (wantDebug) call obj%timer%stop("host_mpi_communication")
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
                if (wantDebug) call obj%timer%start("cuda_mpi_communication")
                call MPI_Isend(bottom_border_send_buffer_mpi_fortran_ptr(1,i), int(bottom_msg_length*stripe_width,kind=MPI_KIND), &
                !call MPI_Isend(bottom_border_send_buffer_mpi_fortran_ptr(1,i), int(b_len,kind=MPI_KIND),  &
                    MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), &
                    int(mpi_comm_rows,kind=MPI_KIND), bottom_send_request(i), mpierr)
                if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
              else !useGPU
                bottom_border_send_buffer(1:b_len,i) = &
                reshape(aIntern(1:csw,n_off+1:n_off+bottom_msg_length,i,:), (/ b_len /))
#ifdef WITH_MPI
                if (wantDebug) call obj%timer%start("mpi_communication")
                call MPI_Isend(bottom_border_send_buffer(1,i), int(b_len,kind=MPI_KIND), &
                           MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), &
                           int(mpi_comm_rows,kind=MPI_KIND), bottom_send_request(i), mpierr)
                if (wantDebug) call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */
              endif !useGPU

#ifndef WITH_MPI
              if (useGPU) then
                if (allComputeOnGPU) then
                  if (next_top_msg_length > 0) then
#ifdef WITH_GPU_STREAMS
                    successGPU = gpu_stream_synchronize(my_stream)
                    check_stream_synchronize_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)

                    successGPU =  gpu_memcpy_async(c_loc(top_border_recv_buffer_mpi_fortran_ptr(1,i)), &
                                             c_loc(bottom_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                             stripe_width*next_top_msg_length* size_of_datatype,      &
                                             gpuMemcpyDeviceToDevice, my_stream)
                    check_memcpy_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)

                    successGPU = gpu_stream_synchronize(my_stream)
                    check_stream_synchronize_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)
                    ! synchronize streamsPerThread; maybe not neccessary
                    successGPU = gpu_stream_synchronize()
                    check_stream_synchronize_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)

#else
                    successGPU =  gpu_memcpy(c_loc(top_border_recv_buffer_mpi_fortran_ptr(1,i)), &
                                             c_loc(bottom_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                             stripe_width*next_top_msg_length* size_of_datatype,      &
                                             gpuMemcpyDeviceToDevice)
                    check_memcpy_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)
#endif
                  endif

#ifndef WITH_GPU_STREAMS
              if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
              successGPU = gpu_devicesynchronize()
              check_memcpy_gpu("tridi_to_band: device_synchronize", successGPU)
              if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
#endif
                else ! allComputeOnGPU
                  if (next_top_msg_length > 0) then
                    top_border_recv_buffer(1:csw*next_top_msg_length*max_threads,i) = &
                            bottom_border_send_buffer(1:csw*next_top_msg_length*max_threads,i)
                  endif
                endif ! allComputeOnGPU
              else !useGPU
                if (next_top_msg_length > 0) then
                  top_border_recv_buffer(1:csw*next_top_msg_length*max_threads,i) = &
                            bottom_border_send_buffer(1:csw*next_top_msg_length*max_threads,i)
                endif
              endif !useGPU
#endif /* WITH_MPI */
            endif ! (bottom_msg_length > 0)

#else /* WITH_OPENMP_TRADITIONAL */

            if (wantDebug) call obj%timer%start("compute_hh_trafo")
            call compute_hh_trafo_&
             &MATH_DATATYPE&
             &_&
             &PRECISION&
             (obj, my_pe, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, &
             stripe_count, max_threads, &
             a_off,  nbw, max_blk_size, bcast_buffer, bcast_buffer_dev, &
             hh_tau_dev, kernel_flops, kernel_time, n_times, &
             current_local_n - bottom_msg_length, bottom_msg_length, i, &
             last_stripe_width, kernel, my_stream=my_stream, success=success)
            if (wantDebug) call obj%timer%stop("compute_hh_trafo")
            if (.not.success) then
              success=.false.
              return
            endif

            !send_b
#ifdef WITH_MPI
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
            if (wantDebug) call obj%timer%start("host_mpi_wait_bottom_send")
#else
            if (wantDebug) call obj%timer%start("cuda_mpi_wait_bottom_send")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */

            call MPI_Wait(bottom_send_request(i), MPI_STATUS_IGNORE, mpierr)
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
            if (wantDebug) call obj%timer%stop("host_mpi_wait_bottom_send")
#else
            if (wantDebug) call obj%timer%stop("cuda_mpi_wait_bottom_send")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
            if (bottom_msg_length > 0) then
              n_off = current_local_n+nbw-bottom_msg_length+a_off

              if (useGPU) then
                if (allComputeOnGPU) then
                  ! send should be done on GPU, send_buffer must be created and filled first
                  ! memcpy from aIntern_dev to bottom_border_send_buffer_dev
                  ! either with two offsets or with indexed pointer
                  if (wantDebug) call obj%timer%start("cuda_memcpy")
#ifdef WITH_GPU_STREAMS
                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)
                  successGPU =  gpu_memcpy_async( c_loc(bottom_border_send_buffer_mpi_fortran_ptr(1,i)), &
                                            c_loc(aIntern_mpi_fortran_ptr(1,n_off+1,i)), &
                                             stripe_width * bottom_msg_length * size_of_datatype,      &
                                             gpuMemcpyDeviceToDevice, my_stream)
                  check_memcpy_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)

                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)
                  ! synchronize streamsPerThread; maybe not neccessary
                  successGPU = gpu_stream_synchronize()
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)
#else
                  successGPU =  gpu_memcpy( c_loc(bottom_border_send_buffer_mpi_fortran_ptr(1,i)), &
                                            c_loc(aIntern_mpi_fortran_ptr(1,n_off+1,i)), &
                                             stripe_width * bottom_msg_length * size_of_datatype,      &
                                             gpuMemcpyDeviceToDevice)
                  check_memcpy_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer_dev", successGPU)
#endif
                  if (wantDebug) call obj%timer%stop("cuda_memcpy")
                else ! allComputeOnGPU
                  if (wantDebug) call obj%timer%start("memcpy")
                  dev_offset = (0 + (n_off * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) * size_of_datatype
#ifdef WITH_GPU_STREAMS
                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)

                  successGPU =  gpu_memcpy_async(int(loc(bottom_border_send_buffer(1,i)),kind=c_intptr_t), &
                                           aIntern_dev + dev_offset,  &
                                           stripe_width*bottom_msg_length* size_of_datatype,  &
                                           gpuMemcpyDeviceToHost, my_stream)
                  check_memcpy_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)

                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)
                  ! synchronize streamsPerThread; maybe not neccessary
                  successGPU = gpu_stream_synchronize()
                  check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)
#else
                  successGPU =  gpu_memcpy(int(loc(bottom_border_send_buffer(1,i)),kind=c_intptr_t), aIntern_dev + dev_offset, &
                                           stripe_width*bottom_msg_length* size_of_datatype,  &
                                           gpuMemcpyDeviceToHost)
                  check_memcpy_gpu("tridi_to_band: aIntern_dev -> bottom_border_send_buffer", successGPU)
#endif
                  if (wantDebug) call obj%timer%stop("memcpy")
                endif ! allComputeOnGPU

#ifdef WITH_MPI
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
                if (wantDebug) call obj%timer%start("host_mpi_communication")
                call MPI_Isend(bottom_border_send_buffer(1,i), int(bottom_msg_length*stripe_width,kind=MPI_KIND), &
                           MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), &
                           int(mpi_comm_rows,kind=MPI_KIND), bottom_send_request(i), mpierr)
                if (wantDebug) call obj%timer%stop("host_mpi_communication")
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
                if (wantDebug) call obj%timer%start("cuda_mpi_communication")
                call MPI_Isend(bottom_border_send_buffer_mpi_fortran_ptr(1,i), int(bottom_msg_length*stripe_width,kind=MPI_KIND), &
                MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), &
                int(mpi_comm_rows,kind=MPI_KIND), bottom_send_request(i), mpierr)
                if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */

#endif /* WITH_MPI */
              else !useGPU
                bottom_border_send_buffer(1:stripe_width*bottom_msg_length,i) = reshape(&
                aIntern(:,n_off+1:n_off+bottom_msg_length,i),(/stripe_width*bottom_msg_length/))
#ifdef WITH_MPI
                if (wantDebug) call obj%timer%start("mpi_communication")
                call MPI_Isend(bottom_border_send_buffer(1,i), int(bottom_msg_length*stripe_width,kind=MPI_KIND), &
                           MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow+1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), &
                           int(mpi_comm_rows,kind=MPI_KIND), bottom_send_request(i), mpierr)
                if (wantDebug) call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */
              endif !useGPU

#ifndef WITH_MPI
              if (useGPU) then
                if (allComputeOnGPU) then
                  if (next_top_msg_length > 0) then
#ifdef WITH_GPU_STREAMS
                    successGPU = gpu_stream_synchronize(my_stream)
                    check_stream_synchronize_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)
                    successGPU =  gpu_memcpy_async(c_loc(top_border_recv_buffer_mpi_fortran_ptr(1,i)), &
                                             c_loc(bottom_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                             stripe_width*next_top_msg_length* size_of_datatype,      &
                                             gpuMemcpyDeviceToDevice, my_stream)
                    check_memcpy_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)

                    successGPU = gpu_stream_synchronize(my_stream)
                    check_stream_synchronize_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)
                    ! synchronize streamsPerThread; maybe not neccessary
                    successGPU = gpu_stream_synchronize()
                    check_stream_synchronize_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)
#else
                    successGPU =  gpu_memcpy(c_loc(top_border_recv_buffer_mpi_fortran_ptr(1,i)), &
                                             c_loc(bottom_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                             stripe_width*next_top_msg_length* size_of_datatype,      &
                                             gpuMemcpyDeviceToDevice)
                    check_memcpy_gpu("tridi_to_band: bottom_border_send_dev -> top_border_recv_dev", successGPU)
#endif
                  endif

#ifndef WITH_GPU_STREAMS
              if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
              successGPU = gpu_devicesynchronize()
              check_memcpy_gpu("tridi_to_band: device_synchronize", successGPU)
              if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
#endif
                else ! allComputeOnGPU
                  if (next_top_msg_length > 0) then
                    top_border_recv_buffer(1:stripe_width*next_top_msg_length,i) =  &
                    bottom_border_send_buffer(1:stripe_width*next_top_msg_length,i)
                  endif
                endif ! allComputeOnGPU
              else ! useGPU
                if (next_top_msg_length > 0) then
                  top_border_recv_buffer(1:stripe_width*next_top_msg_length,i) =  &
                  bottom_border_send_buffer(1:stripe_width*next_top_msg_length,i)
                endif
              endif ! useGPU
#endif /* WITH_MPI */

            endif ! (bottom_msg_length > 0)

#endif /* WITH_OPENMP_TRADITIONAL */

            !compute
#ifdef WITH_OPENMP_TRADITIONAL

            if (useGPU) then
              if (wantDebug) call obj%timer%start("compute_hh_trafo")
              my_thread = 1 ! at the moment only dummy variable
              thread_width2 = 1 ! for the moment dummy variable
              call compute_hh_trafo_&
                   &MATH_DATATYPE&
                   &_openmp_&
                   &PRECISION&
                   (obj, my_pe, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, &
                   stripe_count, max_threads, &
                   l_nev, a_off,  nbw, max_blk_size, bcast_buffer, bcast_buffer_dev, &
                   hh_tau_dev, kernel_flops, kernel_time, n_times, top_msg_length, &
                   current_local_n-top_msg_length-bottom_msg_length, i, &
                   my_thread, thread_width2, kernel, last_stripe_width=last_stripe_width, my_stream=my_stream, &
                   success=success)
              if (wantDebug) call obj%timer%stop("compute_hh_trafo")
              if (.not.success) then
                success=.false.
                return
              endif
            else ! useGPU
              call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)

              !$omp parallel do num_threads(max_threads) if(max_threads > 1) &
              !$omp default(none) &
              !$omp private(my_thread) &
              !$omp shared(max_threads, obj, useGPU, wantDebug, aIntern, aIntern_dev, my_pe, &
              !$omp&       stripe_width, a_dim2, stripe_count, l_nev, a_off, &
              !$omp&       nbw, max_blk_size, bcast_buffer, bcast_buffer_dev, hh_tau_dev, &
              !$omp&       kernel_flops, kernel_time, n_times, top_msg_length, current_local_n, &
              !$omp&       bottom_msg_length, i, thread_width, kernel, success, my_stream) &
              !$omp schedule(static, 1)
              do my_thread = 1, max_threads
                call compute_hh_trafo_&
                &MATH_DATATYPE&
                &_openmp_&
                &PRECISION&
                (obj, my_pe, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width ,a_dim2, stripe_count, &
                max_threads, l_nev, a_off, &
                nbw, max_blk_size, bcast_buffer, bcast_buffer_dev, &
                hh_tau_dev, kernel_flops, kernel_time, n_times, top_msg_length, &
                current_local_n-top_msg_length-bottom_msg_length, i, my_thread, thread_width, &
                kernel, my_stream=my_stream, success=success)
              enddo
              !$omp end parallel do
              call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)
            endif !useGPU

#else /* WITH_OPENMP_TRADITIONAL */

            if (wantDebug) call obj%timer%start("compute_hh_trafo")
            call compute_hh_trafo_&
             &MATH_DATATYPE&
             &_&
             &PRECISION&
             (obj, my_pe, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, &
             stripe_count, max_threads, &
             a_off,  nbw, max_blk_size, bcast_buffer, bcast_buffer_dev, &
             hh_tau_dev, kernel_flops, kernel_time, n_times, top_msg_length, &
             current_local_n-top_msg_length-bottom_msg_length, i, &
             last_stripe_width, kernel, my_stream=my_stream, success=success)
            if (wantDebug) call obj%timer%stop("compute_hh_trafo")
            if (.not.success) then
              success=.false.
              return
            endif
#endif /* WITH_OPENMP_TRADITIONAL */

            !wait_t
            if (top_msg_length>0) then
#ifdef WITH_OPENMP_TRADITIONAL

#ifdef WITH_MPI
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
              if (wantDebug) call obj%timer%start("host_mpi_wait_top_recv")
#else
              if (wantDebug) call obj%timer%start("cuda_mpi_wait_top_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
              call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
              if (wantDebug) call obj%timer%stop("host_mpi_wait_top_recv")
#else
              if (wantDebug) call obj%timer%stop("cuda_mpi_wait_top_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
              if (useGPU) then
                if (allComputeOnGPU) then
                  if (wantDebug) call obj%timer%start("cuda_memcpy")
                  !$omp parallel do num_threads(max_threads) if (max_threads>1) &
                  !$omp default(none) &
                  !$omp private(my_thread, b_len, b_off, dev_offset, successGPU) &
                  !$omp shared(max_threads, a_off, stripe_width, i, a_dim2, stripe_count, csw, &
                  !$omp&       aIntern_mpi_fortran_ptr, top_border_recv_buffer_mpi_fortran_ptr, &
                  !$omp&       top_msg_length, gpuMemcpyDeviceToDevice, my_stream) &
                  !$omp schedule(static, 1)
                  do my_thread = 1, max_threads
                    b_len = csw*top_msg_length
                    b_off = (my_thread-1)*b_len
                    ! check this
#ifdef WITH_GPU_STREAMS
                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                    successGPU =  gpu_memcpy_async(c_loc(aIntern_mpi_fortran_ptr(1,a_off+1,i,my_thread)), &
                                           c_loc(top_border_recv_buffer_mpi_fortran_ptr(b_off+1,i)),  &
                                           csw* top_msg_length* size_of_datatype,      &
                                           gpuMemcpyDeviceToDevice, my_stream)
                  check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
                  ! synchronize streamsPerThread; maybe not neccessary
                  successGPU = gpu_stream_synchronize()
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
#else
                    successGPU =  gpu_memcpy(c_loc(aIntern_mpi_fortran_ptr(1,a_off+1,i,my_thread)), &
                                           c_loc(top_border_recv_buffer_mpi_fortran_ptr(b_off+1,i)),  &
                                           csw* top_msg_length* size_of_datatype,      &
                                           gpuMemcpyDeviceToDevice)
                  check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
#endif
                  enddo
                  !$omp end parallel do
                  if (wantDebug) call obj%timer%stop("cuda_memcpy")
                else ! allComputeOnGPU
                  if (wantDebug) call obj%timer%start("memcpy")
                  !$omp parallel do num_threads(max_threads) if (max_threads>1) &
                  !$omp default(none) &
                  !$omp private(my_thread, b_len, b_off, dev_offset, successGPU) &
                  !$omp shared(max_threads, a_off, stripe_width, i, a_dim2, stripe_count, csw, &
                  !$omp&       aIntern_dev, top_border_recv_buffer, top_msg_length, gpuMemcpyHostToDevice, my_stream) &
                  !$omp schedule(static, 1)
                  do my_thread = 1, max_threads
                    b_len = csw*top_msg_length
                    b_off = (my_thread-1)*b_len
                    ! copy top_border_recv_buffer to aIntern_dev, maybe not necessary if CUDA_AWARE MPI_IRECV
                    ! check this
                    dev_offset = (0 + (a_off * stripe_width) + ( (i-1) * stripe_width * a_dim2 ) + &
                                  (my_thread-1)*stripe_width * a_dim2 * stripe_count) * size_of_datatype
#ifdef WITH_GPU_STREAMS
                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                    successGPU = gpu_memcpy_async( aIntern_dev + dev_offset ,int(loc( top_border_recv_buffer(b_off+1,i)),&
                                  kind=c_intptr_t),  &
                                          csw * top_msg_length * size_of_datatype,   &
                                          gpuMemcpyHostToDevice, my_stream)
                    check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                    successGPU = gpu_stream_synchronize(my_stream)
                    check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
                    ! synchronize streamsPerThread; maybe not neccessary
                    successGPU = gpu_stream_synchronize()
                    check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
#else
                    successGPU = gpu_memcpy( aIntern_dev + dev_offset ,int(loc( top_border_recv_buffer(b_off+1,i)),&
                                  kind=c_intptr_t),  &
                                          csw * top_msg_length * size_of_datatype,   &
                                          gpuMemcpyHostToDevice)
                    check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
#endif
                  enddo
                  !$omp end parallel do
                  if (wantDebug) call obj%timer%stop("memcpy")
                endif ! allComputeOnGPU
              else ! useGPU
                call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)

                !$omp parallel do num_threads(max_threads) if(max_threads > 1) &
                !$omp default(none) &
                !$omp private(my_thread, b_len, b_off) &
                !$omp shared(obj, max_threads, top_msg_length, csw, aIntern, a_off, &
                !$omp&       top_border_recv_buffer, useGPU, wantDebug, aIntern_dev, stripe_width, &
                !$omp&       a_dim2, stripe_count, l_nev, nbw, max_blk_size, bcast_buffer, bcast_buffer_dev, &
                !$omp&       hh_tau_dev, kernel_flops, kernel_time, n_times, i, thread_width, kernel) &
                !$omp schedule(static, 1)
                do my_thread = 1, max_threads
                  if (top_msg_length>0) then
                    b_len = csw*top_msg_length
                    b_off = (my_thread-1)*b_len
                    aIntern(1:csw,a_off+1:a_off+top_msg_length,i,my_thread) = &
                      reshape(top_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, top_msg_length /))
                  endif
                enddo
                !$omp end parallel do
                call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)
              endif ! useGPU

#else /* WITH_OPENMP_TRADITIONAL */

#ifdef WITH_MPI
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
              if (wantDebug) call obj%timer%start("host_mpi_wait_top_recv")
#else
              if (wantDebug) call obj%timer%start("cuda_mpi_wait_top_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
              call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
              if (wantDebug) call obj%timer%stop("host_mpi_wait_top_recv")
#else
              if (wantDebug) call obj%timer%stop("cuda_mpi_wait_top_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
              if (useGPU) then
                if (allComputeOnGPU) then
                  if (wantDebug) call obj%timer%start("cuda_memcpy")
#ifdef WITH_GPU_STREAMS
                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                  successGPU =  gpu_memcpy_async(c_loc(aIntern_mpi_fortran_ptr(1,a_off+1,i)), &
                                           c_loc(top_border_recv_buffer_mpi_fortran_ptr(1,i)),  &
                                           stripe_width* top_msg_length* size_of_datatype,      &
                                           gpuMemcpyDeviceToDevice, my_stream)
                  check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
                  ! synchronize streamsPerThread; maybe not neccessary
                  successGPU = gpu_stream_synchronize()
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
#else
                  successGPU =  gpu_memcpy(c_loc(aIntern_mpi_fortran_ptr(1,a_off+1,i)), &
                                           c_loc(top_border_recv_buffer_mpi_fortran_ptr(1,i)),  &
                                           stripe_width* top_msg_length* size_of_datatype,      &
                                           gpuMemcpyDeviceToDevice)
                  check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
#endif
                  if (wantDebug) call obj%timer%stop("cuda_memcpy")
                else ! allComputeOnGPU
                  if (wantDebug) call obj%timer%start("memcpy")
                  ! copy top_border_recv_buffer to aIntern_dev, maybe not necessary if CUDA_AWARE IRECV
                  dev_offset = (0 + (a_off * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) * size_of_datatype
#ifdef WITH_GPU_STREAMS
                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                  successGPU =  gpu_memcpy_async(aIntern_dev + dev_offset ,int(loc( top_border_recv_buffer(:,i)),kind=c_intptr_t), &
                                        stripe_width * top_msg_length * size_of_datatype,   &
                                        gpuMemcpyHostToDevice, my_stream)
                  check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

                  successGPU = gpu_stream_synchronize(my_stream)
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
                  ! synchronize streamsPerThread; maybe not neccessary
                  successGPU = gpu_stream_synchronize()
                  check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
#else
                  successGPU =  gpu_memcpy(aIntern_dev + dev_offset ,int(loc( top_border_recv_buffer(:,i)),kind=c_intptr_t),  &
                                        stripe_width * top_msg_length * size_of_datatype,   &
                                        gpuMemcpyHostToDevice)
                  check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
#endif
                  if (wantDebug) call obj%timer%stop("memcpy")
                endif ! allComputeOnGPU
              else ! useGPU
                aIntern(:,a_off+1:a_off+top_msg_length,i) = &
                reshape(top_border_recv_buffer(1:stripe_width*top_msg_length,i),(/stripe_width,top_msg_length/))
              endif ! useGPU
#endif /* WITH_OPENMP_TRADITIONAL */
           endif

           !compute
#ifdef WITH_OPENMP_TRADITIONAL
           if (useGPU) then
             my_thread = 1 ! for the momment dummy variable
             thread_width2 = 1 ! for the moment dummy variable
             if (wantDebug) call obj%timer%start("compute_hh_trafo")
             call compute_hh_trafo_&
               &MATH_DATATYPE&
               &_openmp_&
               &PRECISION&
               (obj, my_pe, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, &
               stripe_count, max_threads, &
               l_nev, a_off, nbw, max_blk_size,  bcast_buffer, bcast_buffer_dev, &
               hh_tau_dev, kernel_flops, kernel_time, n_times, 0, top_msg_length, i, &
               my_thread, thread_width2, kernel, last_stripe_width=last_stripe_width, &
               my_stream=my_stream, success=success)
             if (wantDebug) call obj%timer%stop("compute_hh_trafo")
             if (.not.success) then
               success=.false.
               return
             endif
           else ! useGPU
             call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)

             !$omp parallel do num_threads(max_threads) if(max_threads > 1) &
             !$omp default(none) &
             !$omp private(my_thread, b_len, b_off) &
             !$omp shared(obj, max_threads, top_msg_length, csw, aIntern, a_off, my_pe, &
             !$omp&       top_border_recv_buffer, useGPU, wantDebug, aIntern_dev, stripe_width, &
             !$omp&       a_dim2, stripe_count, l_nev, nbw, max_blk_size, bcast_buffer, bcast_buffer_dev, &
             !$omp&       hh_tau_dev, kernel_flops, kernel_time, n_times, i, thread_width, kernel, success, my_stream) &
             !$omp schedule(static, 1)
             do my_thread = 1, max_threads
               call compute_hh_trafo_&
                    &MATH_DATATYPE&
                    &_openmp_&
                    &PRECISION&
                    (obj, my_pe, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, stripe_count, max_threads, &
                    l_nev, a_off, &
                    nbw, max_blk_size,  bcast_buffer, bcast_buffer_dev, &
                    hh_tau_dev, kernel_flops, kernel_time, n_times, 0, top_msg_length, i, my_thread, &
                    thread_width, kernel, my_stream=my_stream, success=success)
             enddo
             !$omp end parallel do
             call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)
           endif ! useGPU
#else /* WITH_OPENMP_TRADITIONAL */

           if (wantDebug) call obj%timer%start("compute_hh_trafo")
           call compute_hh_trafo_&
             &MATH_DATATYPE&
             &_&
             &PRECISION&
             (obj, my_pe, useGPU, wantDebug, aIntern, aIntern_dev, stripe_width, a_dim2, stripe_count, max_threads, &
             a_off, nbw, max_blk_size,  bcast_buffer, bcast_buffer_dev, &
             hh_tau_dev, kernel_flops, kernel_time, n_times, 0, top_msg_length, i, &
             last_stripe_width, kernel, my_stream=my_stream, success=success)
           if (wantDebug) call obj%timer%stop("compute_hh_trafo")
           if (.not.success) then
             success=.false.
             return
           endif

#endif /* WITH_OPENMP_TRADITIONAL */
         endif ! which if branch

         if (next_top_msg_length > 0) then
           !request top_border data
#ifdef WITH_OPENMP_TRADITIONAL
           b_len = csw*next_top_msg_length*max_threads
           if (useGPU) then
#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
             if (wantDebug) call obj%timer%start("cuda_mpi_communication")
             call MPI_Irecv(top_border_recv_buffer_mpi_fortran_ptr(1,i), int(next_top_msg_length*stripe_width,kind=MPI_KIND), &
             !call MPI_Irecv(top_border_recv_buffer_mpi_fortran_ptr(1,i), int(b_len,kind=MPI_KIND), &
                         MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow-1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), &
                         int(mpi_comm_rows,kind=MPI_KIND), top_recv_request(i), mpierr)
             if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
             if (wantDebug) call obj%timer%start("host_mpi_communication")
             call MPI_Irecv(top_border_recv_buffer(1,i), int(next_top_msg_length*stripe_width,kind=MPI_KIND), &
             !call MPI_Irecv(top_border_recv_buffer(1,i), int(b_len,kind=MPI_KIND), &
                         MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow-1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), &
                         int(mpi_comm_rows,kind=MPI_KIND), top_recv_request(i), mpierr)
             if (wantDebug) call obj%timer%stop("host_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */

#endif /* WITH_MPI */
           else !useGPU
#ifdef WITH_MPI
             if (wantDebug) call obj%timer%start("mpi_communication")
             call MPI_Irecv(top_border_recv_buffer(1,i), int(b_len,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                         int(my_prow-1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), &
                         top_recv_request(i), mpierr)
             if (wantDebug) call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */
           endif !useGPU
#ifndef WITH_MPI
!             carefull the "recieve" has to be done at the corresponding wait or send
!              top_border_recv_buffer(1:csw*next_top_msg_length*max_threads,i) = &
!                                     bottom_border_send_buffer(1:csw*next_top_msg_length*max_threads,i)
#endif /* WITH_MPI */

#else /* WITH_OPENMP_TRADITIONAL */

#ifdef WITH_MPI
           if (useGPU) then
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
             if (wantDebug) call obj%timer%start("cuda_mpi_communication")
             call MPI_Irecv(top_border_recv_buffer_mpi_fortran_ptr(1,i), int(next_top_msg_length*stripe_width,kind=MPI_KIND), &
                         MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow-1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), &
                         int(mpi_comm_rows,kind=MPI_KIND), top_recv_request(i), mpierr)
             if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
             if (wantDebug) call obj%timer%start("host_mpi_communication")
             call MPI_Irecv(top_border_recv_buffer(1,i), int(next_top_msg_length*stripe_width,kind=MPI_KIND), &
                         MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow-1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), &
                         int(mpi_comm_rows,kind=MPI_KIND), top_recv_request(i), mpierr)
             if (wantDebug) call obj%timer%stop("host_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
           else !useGPU
             if (wantDebug) call obj%timer%start("mpi_communication")
             call MPI_Irecv(top_border_recv_buffer(1,i), int(next_top_msg_length*stripe_width,kind=MPI_KIND), &
                         MPI_MATH_DATATYPE_PRECISION_EXPL, int(my_prow-1,kind=MPI_KIND), int(top_recv_tag,kind=MPI_KIND), &
                         int(mpi_comm_rows,kind=MPI_KIND), top_recv_request(i), mpierr)
             if (wantDebug) call obj%timer%stop("mpi_communication")
           endif !useGPU
#else /* WITH_MPI */
!             carefull the "recieve" has to be done at the corresponding wait or send
!              top_border_recv_buffer(1:stripe_width,1:next_top_msg_length,i) =  &
!               bottom_border_send_buffer(1:stripe_width,1:next_top_msg_length,i)
#endif /* WITH_MPI */

#endif /* WITH_OPENMP_TRADITIONAL */

         endif ! next_top_msg_length > 0

         !send_t
         if (my_prow > 0) then
#ifdef WITH_OPENMP_TRADITIONAL

#ifdef WITH_MPI
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
           if (wantDebug) call obj%timer%start("host_mpi_wait_top_send")
#else
           if (wantDebug) call obj%timer%start("cuda_mpi_wait_top_send")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
           call MPI_Wait(top_send_request(i), MPI_STATUS_IGNORE, mpierr)
#ifndef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
           if (wantDebug) call obj%timer%stop("host_mpi_wait_top_send")
#else
           if (wantDebug) call obj%timer%stop("cuda_mpi_wait_top_send")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
           b_len = csw*nbw*max_threads
           if (useGPU) then
             if (allComputeOnGPU) then
               ! my_thread is assumed 1!
               ! this should be updated
               if (wantDebug) call obj%timer%start("cuda_memcpy")
#ifdef WITH_GPU_STREAMS
               successGPU = gpu_stream_synchronize(my_stream)
               check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

               successGPU =  gpu_memcpy_async(c_loc(top_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                        c_loc(aIntern_mpi_fortran_ptr(1,a_off+1,i,my_thread)), &
                                        stripe_width* nbw* size_of_datatype,      &
                                        gpuMemcpyDeviceToDevice, my_stream)
               check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

               successGPU = gpu_stream_synchronize(my_stream)
               check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
               ! synchronize streamsPerThread; maybe not neccessary
               successGPU = gpu_stream_synchronize()
               check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
#else
               successGPU =  gpu_memcpy(c_loc(top_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                        c_loc(aIntern_mpi_fortran_ptr(1,a_off+1,i,my_thread)), &
                                        stripe_width* nbw* size_of_datatype,      &
                                        gpuMemcpyDeviceToDevice)
               check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
#endif
               if (wantDebug) call obj%timer%stop("cuda_memcpy")
             else ! allComputeOnGPU
               if (wantDebug) call obj%timer%start("memcpy")
               dev_offset = (0 + (a_off * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) * size_of_datatype
#ifdef WITH_GPU_STREAMS
               successGPU = gpu_stream_synchronize(my_stream)
               check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> top_border_send_buffer", successGPU)

               successGPU =  gpu_memcpy_async(int(loc(top_border_send_buffer(:,i)),kind=c_intptr_t), aIntern_dev + dev_offset, &
                                      stripe_width*nbw * size_of_datatype, &
                                      gpuMemcpyDeviceToHost, my_stream)
               check_memcpy_gpu("tridi_to_band: aIntern_dev -> top_border_send_buffer", successGPU)

               successGPU = gpu_stream_synchronize(my_stream)
               check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> top_border_send_buffer", successGPU)
               ! synchronize streamsPerThread; maybe not neccessary
               successGPU = gpu_stream_synchronize()
               check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> top_border_send_buffer", successGPU)
#else
               successGPU =  gpu_memcpy(int(loc(top_border_send_buffer(:,i)),kind=c_intptr_t), aIntern_dev + dev_offset, &
                                      stripe_width*nbw * size_of_datatype, &
                                      gpuMemcpyDeviceToHost)
               check_memcpy_gpu("tridi_to_band: aIntern_dev -> top_border_send_buffer", successGPU)
#endif
               if (wantDebug) call obj%timer%stop("memcpy")
             endif ! allComputeOnGPU
           else ! useGPU
             top_border_send_buffer(1:b_len,i) = reshape(aIntern(1:csw,a_off+1:a_off+nbw,i,:), (/ b_len /))
           endif ! useGPU

           if (useGPU) then
#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
             if (wantDebug) call obj%timer%start("cuda_mpi_communication")
             call MPI_Isend(top_border_send_buffer_mpi_fortran_ptr(1,i), int(nbw*stripe_width,kind=MPI_KIND), &
             !call MPI_Isend(top_border_send_buffer_mpi_fortran_ptr(1,i), int(b_len,kind=MPI_KIND), &
                            MPI_MATH_DATATYPE_PRECISION_EXPL, &
                            int(my_prow-1,kind=MPI_KIND), int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), &
                            top_send_request(i), mpierr)
             if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
             if (wantDebug) call obj%timer%start("host_mpi_communication")
             call MPI_Isend(top_border_send_buffer(1,i), int(nbw*stripe_width,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
             !call MPI_Isend(top_border_send_buffer(1,i), int(b_len,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                         int(my_prow-1,kind=MPI_KIND), int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND),   &
                         top_send_request(i), mpierr)
             if (wantDebug) call obj%timer%stop("host_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
           else ! useGPU
#ifdef WITH_MPI
             if (wantDebug) call obj%timer%start("mpi_communication")
             call MPI_Isend(top_border_send_buffer(1,i), int(b_len,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                         int(my_prow-1,kind=MPI_KIND), int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), &
                         top_send_request(i), mpierr)
             if (wantDebug) call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */
           endif ! useGPU

#ifndef WITH_MPI
           if (useGPU) then
             if (allComputeOnGPU) then
               ! my_thread is assumed 1 !
               ! this should be updated
               if (sweep==0 .and. current_n_end < current_n .and. l_nev > 0) then
#ifdef WITH_GPU_STREAMS
                 successGPU = gpu_stream_synchronize(my_stream)
                 check_stream_synchronize_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)
                 successGPU =  gpu_memcpy_async(c_loc(bottom_border_recv_buffer_mpi_fortran_ptr(1,i)), &
                                          c_loc(top_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                          nbw*stripe_width* size_of_datatype,      &
                                           gpuMemcpyDeviceToDevice, my_stream)
                 check_memcpy_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)

                 successGPU = gpu_stream_synchronize(my_stream)
                 check_stream_synchronize_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)
                 ! synchronize streamsPerThread; maybe not neccessary
                 successGPU = gpu_stream_synchronize()
                 check_stream_synchronize_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)

#else
                 successGPU =  gpu_memcpy(c_loc(bottom_border_recv_buffer_mpi_fortran_ptr(1,i)), &
                                          c_loc(top_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                          nbw*stripe_width* size_of_datatype,      &
                                           gpuMemcpyDeviceToDevice)
                 check_memcpy_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)
#endif
               endif
#ifndef WITH_GPU_STREAMS
              if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
              successGPU = gpu_devicesynchronize()
              check_memcpy_gpu("tridi_to_band: device_synchronize", successGPU)
              if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
#endif
               if (next_n_end < next_n) then
#ifdef WITH_GPU_STREAMS
                 successGPU = gpu_stream_synchronize(my_stream)
                 check_stream_synchronize_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)
                 successGPU =  gpu_memcpy_async(c_loc(bottom_border_recv_buffer_mpi_fortran_ptr(1,i)), &
                                          c_loc(top_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                          nbw*stripe_width* size_of_datatype,      &
                                           gpuMemcpyDeviceToDevice, my_stream)
                 check_memcpy_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)

                 successGPU = gpu_stream_synchronize(my_stream)
                 check_stream_synchronize_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)
                 ! synchronize streamsPerThread; maybe not neccessary
                 successGPU = gpu_stream_synchronize()
                 check_stream_synchronize_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)

#else
                 successGPU =  gpu_memcpy(c_loc(bottom_border_recv_buffer_mpi_fortran_ptr(1,i)), &
                                          c_loc(top_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                          nbw*stripe_width* size_of_datatype,      &
                                           gpuMemcpyDeviceToDevice)
                 check_memcpy_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)
#endif
               endif
#ifndef WITH_GPU_STREAMS
              if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
              successGPU = gpu_devicesynchronize()
              check_memcpy_gpu("tridi_to_band: device_synchronize", successGPU)
              if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
#endif
             else ! allComputeOnGPU
               if (sweep==0 .and. current_n_end < current_n .and. l_nev > 0) then
                 bottom_border_recv_buffer(1:nbw*stripe_width,i) = top_border_send_buffer(1:nbw*stripe_width,i)
                 !bottom_border_recv_buffer(1:csw*nbw*max_threads,i) = top_border_send_buffer(1:csw*nbw*max_threads,i)
               endif
               if (next_n_end < next_n) then
                 bottom_border_recv_buffer(1:stripe_width*nbw,i) =  top_border_send_buffer(1:stripe_width*nbw,i)
                 !bottom_border_recv_buffer(1:csw*nbw*max_threads,i) = top_border_send_buffer(1:csw*nbw*max_threads,i)
               endif
             endif ! allComputeOnGPU
           else ! useGPU
             if (sweep==0 .and. current_n_end < current_n .and. l_nev > 0) then
               bottom_border_recv_buffer(1:csw*nbw*max_threads,i) = top_border_send_buffer(1:csw*nbw*max_threads,i)
             endif
             if (next_n_end < next_n) then
               bottom_border_recv_buffer(1:csw*nbw*max_threads,i) = top_border_send_buffer(1:csw*nbw*max_threads,i)
             endif
           endif ! useGPU
#endif /* WITH_MPI */

#else /* WITH_OPENMP_TRADITIONAL */

#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
           if (wantDebug) call obj%timer%start("cuda_mpi_wait_top_send")
#else
           if (wantDebug) call obj%timer%start("host_mpi_wait_top_send")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
           call MPI_Wait(top_send_request(i), MPI_STATUS_IGNORE, mpierr)
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
           if (wantDebug) call obj%timer%stop("cuda_mpi_wait_top_send")
#else
           if (wantDebug) call obj%timer%stop("host_mpi_wait_top_send")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */

#endif /* WITH_MPI */
           if (useGPU) then
             if (allComputeOnGPU) then
               if (wantDebug) call obj%timer%start("cuda_memcpy")
#ifdef WITH_GPU_STREAMS
               successGPU = gpu_stream_synchronize(my_stream)
               check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

               successGPU =  gpu_memcpy_async(c_loc(top_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                        c_loc(aIntern_mpi_fortran_ptr(1,a_off+1,i)), &
                                        stripe_width* nbw* size_of_datatype,      &
                                        gpuMemcpyDeviceToDevice, my_stream)
               check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

               successGPU = gpu_stream_synchronize(my_stream)
               check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
               ! synchronize streamsPerThread; maybe not neccessary
               successGPU = gpu_stream_synchronize()
               check_stream_synchronize_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)

#else
               successGPU =  gpu_memcpy(c_loc(top_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                        c_loc(aIntern_mpi_fortran_ptr(1,a_off+1,i)), &
                                        stripe_width* nbw* size_of_datatype,      &
                                        gpuMemcpyDeviceToDevice)
               check_memcpy_gpu("tridi_to_band: top_border_recv_buffer -> aIntern_dev", successGPU)
#endif
               if (wantDebug) call obj%timer%stop("cuda_memcpy")
             else ! allComputeOnGPU
               if (wantDebug) call obj%timer%start("memcpy")
               dev_offset = (0 + (a_off * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) * size_of_datatype
#ifdef WITH_GPU_STREAMS
               successGPU = gpu_stream_synchronize(my_stream)
               check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> top_border_send_buffer", successGPU)

               successGPU =  gpu_memcpy_async(int(loc(top_border_send_buffer(:,i)),kind=c_intptr_t), aIntern_dev + dev_offset, &
                                         stripe_width*nbw * size_of_datatype, &
                                         gpuMemcpyDeviceToHost, my_stream)
               check_memcpy_gpu("tridi_to_band: aIntern_dev -> top_border_send_buffer", successGPU)

               successGPU = gpu_stream_synchronize(my_stream)
               check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> top_border_send_buffer", successGPU)
               ! synchronize streamsPerThread; maybe not neccessary
               successGPU = gpu_stream_synchronize()
               check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> top_border_send_buffer", successGPU)

#else
               successGPU =  gpu_memcpy(int(loc(top_border_send_buffer(:,i)),kind=c_intptr_t), aIntern_dev + dev_offset, &
                                         stripe_width*nbw * size_of_datatype, &
                                         gpuMemcpyDeviceToHost)
               check_memcpy_gpu("tridi_to_band: aIntern_dev -> top_border_send_buffer", successGPU)
#endif
               if (wantDebug) call obj%timer%stop("memcpy")
             endif ! allComputeOnGPU
           else ! useGPU
             top_border_send_buffer(:,i) = reshape(aIntern(:,a_off+1:a_off+nbw,i),(/stripe_width*nbw/))
           endif ! useGPU
#ifdef WITH_MPI
           if (useGPU) then
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
             if (wantDebug) call obj%timer%start("cuda_mpi_communication")
             call MPI_Isend(top_border_send_buffer_mpi_fortran_ptr(1,i), int(nbw*stripe_width,kind=MPI_KIND), &
                            MPI_MATH_DATATYPE_PRECISION_EXPL, &
                            int(my_prow-1,kind=MPI_KIND), int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND),  &
                            top_send_request(i), mpierr)
             if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
             if (wantDebug) call obj%timer%start("host_mpi_communication")
             call MPI_Isend(top_border_send_buffer(1,i), int(nbw*stripe_width,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                            int(my_prow-1,kind=MPI_KIND), int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), &
                            top_send_request(i), mpierr)
             if (wantDebug) call obj%timer%stop("host_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
           else ! useGPU
             if (wantDebug) call obj%timer%start("mpi_communication")
             call MPI_Isend(top_border_send_buffer(1,i), int(nbw*stripe_width,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                            int(my_prow-1,kind=MPI_KIND), int(bottom_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), &
                            top_send_request(i), mpierr)
             if (wantDebug) call obj%timer%stop("mpi_communication")
           endif ! useGPU
#else /* WITH_MPI */
           if (useGPU) then
             if (allComputeOnGPU) then
               if (sweep==0 .and. current_n_end < current_n .and. l_nev > 0) then
#ifdef WITH_GPU_STREAMS
                 successGPU = gpu_stream_synchronize(my_stream)
                 check_stream_synchronize_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)
                 successGPU =  gpu_memcpy_async(c_loc(bottom_border_recv_buffer_mpi_fortran_ptr(1,i)), &
                                          c_loc(top_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                          nbw*stripe_width* size_of_datatype,      &
                                           gpuMemcpyDeviceToDevice, my_stream)
                 check_memcpy_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)

                 successGPU = gpu_stream_synchronize(my_stream)
                 check_stream_synchronize_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)
                 ! synchronize streamsPerThread; maybe not neccessary
                 successGPU = gpu_stream_synchronize()
                 check_stream_synchronize_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)
#else
                 successGPU =  gpu_memcpy(c_loc(bottom_border_recv_buffer_mpi_fortran_ptr(1,i)), &
                                          c_loc(top_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                          nbw*stripe_width* size_of_datatype,      &
                                           gpuMemcpyDeviceToDevice)
                 check_memcpy_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)
#endif
               endif
#ifndef WITH_GPU_STREAMS
              if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
              successGPU = gpu_devicesynchronize()
              check_memcpy_gpu("tridi_to_band: device_synchronize", successGPU)
              if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
#endif
               if (next_n_end < next_n) then
#ifdef WITH_GPU_STREAMS
                 successGPU = gpu_stream_synchronize(my_stream)
                 check_stream_synchronize_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)
                 successGPU =  gpu_memcpy_async(c_loc(bottom_border_recv_buffer_mpi_fortran_ptr(1,i)), &
                                          c_loc(top_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                          nbw*stripe_width* size_of_datatype,      &
                                           gpuMemcpyDeviceToDevice, my_stream)
                 check_memcpy_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)

                 successGPU = gpu_stream_synchronize(my_stream)
                 check_stream_synchronize_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)
                 ! synchronize streamsPerThread; maybe not neccessary
                 successGPU = gpu_stream_synchronize()
                 check_stream_synchronize_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)
#else
                 successGPU =  gpu_memcpy(c_loc(bottom_border_recv_buffer_mpi_fortran_ptr(1,i)), &
                                          c_loc(top_border_send_buffer_mpi_fortran_ptr(1,i)),  &
                                          nbw*stripe_width* size_of_datatype,      &
                                           gpuMemcpyDeviceToDevice)
                 check_memcpy_gpu("tridi_to_band: top_border_send_dev -> bottom_border_recv_dev", successGPU)
#endif
               endif
#ifndef WITH_GPU_STREAMS
              if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
              successGPU = gpu_devicesynchronize()
              check_memcpy_gpu("tridi_to_band: device_synchronize", successGPU)
              if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
#endif
             else ! allComputeOnGPU
               if (sweep==0 .and. current_n_end < current_n .and. l_nev > 0) then
                 bottom_border_recv_buffer(1:nbw*stripe_width,i) = top_border_send_buffer(1:nbw*stripe_width,i)
               endif
               if (next_n_end < next_n) then
                 bottom_border_recv_buffer(1:stripe_width*nbw,i) =  top_border_send_buffer(1:stripe_width*nbw,i)
               endif
             endif ! allComputeOnGPU
           else ! useGPU
             if (sweep==0 .and. current_n_end < current_n .and. l_nev > 0) then
               bottom_border_recv_buffer(1:nbw*stripe_width,i) = top_border_send_buffer(1:nbw*stripe_width,i)
             endif
             if (next_n_end < next_n) then
               bottom_border_recv_buffer(1:stripe_width*nbw,i) =  top_border_send_buffer(1:stripe_width*nbw,i)
             endif
           endif ! useGPU
#endif /* WITH_MPI */

#endif /* WITH_OPENMP_TRADITIONAL */
         endif ! my_prow > 0

         ! Care that there are not too many outstanding top_recv_request's
         if (stripe_count > 1) then
           if (i > 1) then
#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
             if (wantDebug) call obj%timer%start("cuda_mpi_wait_top_recv")
#else
             if (wantDebug) call obj%timer%start("host_mpi_wait_top_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
             call MPI_Wait(top_recv_request(i-1), MPI_STATUS_IGNORE, mpierr)
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
             if (wantDebug) call obj%timer%stop("cuda_mpi_wait_top_recv")
#else
             if (wantDebug) call obj%timer%stop("host_mpi_wait_top_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
           else ! i > 1
#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
             if (wantDebug) call obj%timer%start("cuda_mpi_wait_top_recv")
#else
             if (wantDebug) call obj%timer%start("host_mpi_wait_top_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
            call MPI_Wait(top_recv_request(stripe_count), MPI_STATUS_IGNORE, mpierr)
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
             if (wantDebug) call obj%timer%stop("cuda_mpi_wait_top_recv")
#else
             if (wantDebug) call obj%timer%stop("host_mpi_wait_top_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
          endif ! i > 1
        endif ! stripe_count > 1
      enddo ! i = 1, stripe_count

      top_msg_length = next_top_msg_length

    else ! current_local_n > 0
      ! wait for last top_send_request

#ifdef WITH_MPI
      do i = 1, stripe_count
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
        if (wantDebug) call obj%timer%start("cuda_mpi_wait_top_send")
#else
        if (wantDebug) call obj%timer%start("host_mpi_wait_top_send")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
        call MPI_Wait(top_send_request(i), MPI_STATUS_IGNORE, mpierr)
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
        if (wantDebug) call obj%timer%stop("cuda_mpi_wait_top_send")
#else
        if (wantDebug) call obj%timer%stop("host_mpi_wait_top_send")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
      enddo ! i = 1, stripe_count
#endif /* WITH_MPI */
    endif  ! current_local_n > 0

    ! Care about the result

    if (my_prow == 0) then

      ! topmost process sends nbw rows to destination processes

      do j=0, nfact-1
        num_blk = sweep*nfact+j ! global number of destination block, 0 based
        if (num_blk*nblk >= na) exit

        nbuf = mod(num_blk, num_result_buffers) + 1 ! buffer number to get this block

#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
        if (wantDebug) call obj%timer%start("cuda_mpi_wait_result_send")
#else
        if (wantDebug) call obj%timer%start("host_mpi_wait_result_send")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
        call MPI_Wait(result_send_request(nbuf), MPI_STATUS_IGNORE, mpierr)
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
        if (wantDebug) call obj%timer%stop("cuda_mpi_wait_result_send")
#else
        if (wantDebug) call obj%timer%stop("host_mpi_wait_result_send")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
        dst = mod(num_blk, np_rows)

        if (dst == 0) then
          if (useGPU) then
            row_group_size = min(na - num_blk*nblk, nblk)

            call pack_row_group_&
                 &MATH_DATATYPE&
                 &_gpu_&
                 &PRECISION&
                 &(obj, row_group_dev, aIntern_dev, stripe_count, stripe_width, last_stripe_width, a_dim2, l_nev, &
                         row_group(:, :), j * nblk + a_off, row_group_size, &
                         result_buffer_dev, nblk, num_result_buffers, nbuf, .false., wantDebug, allComputeOnGPU, my_stream)

            if (allComputeOnGPU) then
              ! memcpy DeviceToDevice row_group_dev -> q_dev
              do i = 1, row_group_size
               if (wantDebug) call obj%timer%start("cuda_aware_gpublas")
               gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
               call gpublas_PRECISION_COPY(l_nev, c_loc(row_group_mpi_fortran_ptr(1,i)), 1, &
                                           c_loc(q_mpi_fortran_ptr((num_blk / np_rows) * nblk + i,1)), ldq, gpuHandle)
               if (wantDebug) call obj%timer%stop("cuda_aware_gpublas")
              enddo
            else ! allComputeOnGPU
              do i = 1, row_group_size
                q((num_blk / np_rows) * nblk + i, 1 : l_nev) = row_group(:, i)
              enddo
            endif ! allComputeOnGPU
          else ! useGPU

            do i = 1, min(na - num_blk*nblk, nblk)
#ifdef WITH_OPENMP_TRADITIONAL
              call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)
              !$omp parallel do num_threads(max_threads) if (max_threads>1) &
              !$omp default(none) &
              !$omp private(my_thread) &
              !$omp shared(max_threads, obj, aIntern, row, i, limits, ip, stripe_count, thread_width, &
              !$omp&       stripe_width, l_nev, j, nblk, a_off) &
              !$omp schedule(static, 1)
              do my_thread = 1, max_threads
                call pack_row_&
                   &MATH_DATATYPE&
                   &_cpu_openmp_&
                   &PRECISION&
                   &(obj, aIntern, row, j*nblk+i+a_off, stripe_width, stripe_count, my_thread, thread_width, l_nev)
              enddo
              !$omp end parallel do
              call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)
#else /* WITH_OPENMP_TRADITIONAL */
              call pack_row_&
                   &MATH_DATATYPE&
                   &_cpu_&
                   &PRECISION&
                   &(obj, aIntern, row, j*nblk+i+a_off, stripe_width, last_stripe_width, stripe_count)
#endif /* WITH_OPENMP_TRADITIONAL */
              q((num_blk/np_rows)*nblk+i,1:l_nev) = row(:)
            enddo
          endif ! useGPU

        else ! (dst == 0)

          if (useGPU) then

            call pack_row_group_&
                 &MATH_DATATYPE&
                 &_gpu_&
                 &PRECISION&
                 &(obj, row_group_dev, aIntern_dev, stripe_count, stripe_width, &
                   last_stripe_width, a_dim2, l_nev, &
                   result_buffer(:, :, nbuf), j * nblk + a_off, nblk, &
                   result_buffer_dev, nblk, num_result_buffers, nbuf, .true., wantDebug, allComputeOnGPU, my_stream)

          else  ! useGPU
            do i = 1, nblk
#ifdef WITH_OPENMP_TRADITIONAL
              call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)
              !$omp parallel do num_threads(max_threads) if (max_threads>1) &
              !$omp default(none) &
              !$omp private(my_thread) &
              !$omp shared(max_threads, obj, aIntern, result_buffer, i, limits, ip, stripe_count, thread_width, &
              !$omp&       stripe_width, l_nev, j, nblk, a_off, nbuf) &
              !$omp schedule(static, 1)
              do my_thread = 1, max_threads
                call pack_row_&
                   &MATH_DATATYPE&
                   &_cpu_openmp_&
                   &PRECISION&
                   &(obj, aIntern, result_buffer(:,i,nbuf), j*nblk+i+a_off, stripe_width, stripe_count, &
                     my_thread, thread_width, l_nev)
              enddo
              !$omp end parallel do
              call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)
#else /* WITH_OPENMP_TRADITIONAL */
              call pack_row_&
                   &MATH_DATATYPE&
                   &_cpu_&
                   &PRECISION&
                   &(obj, aIntern, result_buffer(:,i,nbuf),j*nblk+i+a_off, stripe_width, last_stripe_width, stripe_count)
#endif /* WITH_OPENMP_TRADITIONAL */
            enddo
          endif ! useGPU

#ifdef WITH_MPI
          if (useGPU) then
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
            if (wantDebug) call obj%timer%start("cuda_mpi_communication")
            call MPI_Isend(result_buffer_mpi_fortran_ptr(1,1,nbuf), int(l_nev*nblk,kind=MPI_KIND), &
                           MPI_MATH_DATATYPE_PRECISION_EXPL, &
                           int(dst,kind=MPI_KIND), int(result_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), &
                           result_send_request(nbuf), mpierr)
            if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
            if (wantDebug) call obj%timer%start("host_mpi_communication")
            call MPI_Isend(result_buffer(1,1,nbuf), int(l_nev*nblk,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                           int(dst,kind=MPI_KIND), int(result_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), &
                           result_send_request(nbuf), mpierr)
            if (wantDebug) call obj%timer%stop("host_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
          else ! useGPU
            if (wantDebug) call obj%timer%start("mpi_communication")
            call MPI_Isend(result_buffer(1,1,nbuf), int(l_nev*nblk,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                           int(dst,kind=MPI_KIND), int(result_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), &
                           result_send_request(nbuf), mpierr)
            if (wantDebug) call obj%timer%stop("mpi_communication")
          endif ! useGPU
#else /* WITH_MPI */
#endif /* WITH_MPI */
        endif ! (dst == 0)
      enddo  !j=0, nfact-1

    else ! (my_prow == 0)

      ! receive and store final result

      do j = num_bufs_recvd, num_result_blocks-1

        nbuf = mod(j, num_result_buffers) + 1 ! buffer number to get this block

        ! If there is still work to do, just test for the next result request
        ! and leave the loop if it is not ready, otherwise wait for all
        ! outstanding requests

        if (next_local_n > 0) then
            ! needed
            !successGPU = gpu_devicesynchronize()
#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
          if (wantDebug) call obj%timer%start("cuda_mpi_test")
#else
          if (wantDebug) call obj%timer%start("host_mpi_test")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
          call MPI_Test(result_recv_request(nbuf), flag, MPI_STATUS_IGNORE, mpierr)
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
          if (wantDebug) call obj%timer%stop("cuda_mpi_test")
#else
          if (wantDebug) call obj%timer%stop("host_mpi_test")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#else /* WITH_MPI */
          flag = .true.
#endif /* WITH_MPI */

          if (.not.flag) exit

        else ! (next_local_n > 0)
#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
          if (wantDebug) call obj%timer%start("cuda_mpi_wait_result_recv")
#else
          if (wantDebug) call obj%timer%start("host_mpi_wait_result_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */

          call MPI_Wait(result_recv_request(nbuf), MPI_STATUS_IGNORE, mpierr)
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
          if (wantDebug) call obj%timer%stop("cuda_mpi_wait_result_recv")
#else
          if (wantDebug) call obj%timer%stop("host_mpi_wait_result_recv")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
        endif ! (next_local_n > 0)

        ! Fill result buffer into q
        num_blk = j*np_rows + my_prow ! global number of current block, 0 based
        if (useGPU) then
          if (allComputeOnGPU) then
            do i = 1, min(na - num_blk*nblk, nblk)
              if (wantDebug) call obj%timer%start("cuda_aware_gpublas")
              gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
              call gpublas_PRECISION_COPY(l_nev, c_loc(result_buffer_mpi_fortran_ptr(1,i,nbuf)), 1, &
                                          c_loc(q_mpi_fortran_ptr(j*nblk + i,1)), ldq, gpuHandle)
              if (wantDebug) call obj%timer%stop("cuda_aware_gpublas")
            enddo
          else ! allComputeOnGPU
            do i = 1, min(na - num_blk*nblk, nblk)
              q(j*nblk+i, 1:l_nev) = result_buffer(1:l_nev, i, nbuf)
            enddo
          endif ! allComputeOnGPU
        else ! useGPU
          do i = 1, min(na - num_blk*nblk, nblk)
            q(j*nblk+i, 1:l_nev) = result_buffer(1:l_nev, i, nbuf)
          enddo
        endif ! useGPU
        ! Queue result buffer again if there are outstanding blocks left
#ifdef WITH_MPI

        if (useGPU) then
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
          ! result_buffer 4
          if (wantDebug) call obj%timer%start("cuda_mpi_communication")

          if (j+num_result_buffers < num_result_blocks) then
            call MPI_Irecv(result_buffer_mpi_fortran_ptr(1,1,nbuf), int(l_nev*nblk,kind=MPI_KIND), &
                         MPI_MATH_DATATYPE_PRECISION_EXPL, &
                         0_MPI_KIND, int(result_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), &
                         result_recv_request(nbuf), mpierr)
          endif

          ! carefull the "recieve" has to be done at the corresponding wait or send
          !         if (j+num_result_buffers < num_result_blocks) &
          !                result_buffer(1:l_nev*nblk,1,nbuf) =  result_buffer(1:l_nev*nblk,1,nbuf)
          if (wantDebug) call obj%timer%stop("cuda_mpi_communication")
#else /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
          if (wantDebug) call obj%timer%start("host_mpi_communication")

          if (j+num_result_buffers < num_result_blocks) then
            call MPI_Irecv(result_buffer(1,1,nbuf), int(l_nev*nblk,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                           0_MPI_KIND, int(result_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), &
                           result_recv_request(nbuf), mpierr)
          endif

          ! carefull the "recieve" has to be done at the corresponding wait or send
          !         if (j+num_result_buffers < num_result_blocks) &
          !                result_buffer(1:l_nev*nblk,1,nbuf) =  result_buffer(1:l_nev*nblk,1,nbuf)
          if (wantDebug) call obj%timer%stop("host_mpi_communication")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
        else ! useGPU
          if (wantDebug) call obj%timer%start("mpi_communication")

          if (j+num_result_buffers < num_result_blocks) then
            call MPI_Irecv(result_buffer(1,1,nbuf), int(l_nev*nblk,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION_EXPL, &
                           0_MPI_KIND, int(result_recv_tag,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), &
                           result_recv_request(nbuf), mpierr)
          endif

          ! carefull the "recieve" has to be done at the corresponding wait or send
          !         if (j+num_result_buffers < num_result_blocks) &
          !                result_buffer(1:l_nev*nblk,1,nbuf) =  result_buffer(1:l_nev*nblk,1,nbuf)
          if (wantDebug) call obj%timer%stop("mpi_communication")
        endif !useGPU
#else /* WITH_MPI */

#endif /* WITH_MPI */

      enddo ! j = num_bufs_recvd, num_result_blocks-1
      num_bufs_recvd = j

    endif ! (my_prow == 0)

    ! Shift the remaining rows to the front of aIntern (if necessary)

    offset = nbw - top_msg_length
    if (offset<0) then
      if (wantDebug) write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_&
                                         &MATH_DATATYPE&
                                         &: internal error, offset for shifting = ',offset
      success = .false.
      return
    endif

    a_off = a_off + offset
    if (a_off + next_local_n + nbw >= a_dim2) then
#ifdef WITH_OPENMP_TRADITIONAL
      if (useGPU) then
        do i = 1, stripe_count
          chunk = min(next_local_n,a_off)

          if (chunk < 1) exit

          if (wantDebug) call obj%timer%start("normal_memcpy")
          !$omp parallel do num_threads(max_threads) if (max_threads>1) &
          !$omp default(none) &
          !$omp private(my_thread, j, this_chunk, dev_offset, dev_offset_1, num, successGPU) &
          !$omp shared(max_threads, top_msg_length, next_local_n, chunk, i, a_off, &
          !$omp&       stripe_width, a_dim2, stripe_count, aIntern_dev, gpuMemcpyDeviceToDevice, my_stream) &
          !$omp schedule(static, 1)
          do my_thread = 1, max_threads
            ! check this
            do j = top_msg_length+1, top_msg_length+next_local_n, chunk
              this_chunk = min(j+chunk-1,top_msg_length+next_local_n)-j+1
              dev_offset = ((j-1)*stripe_width+(i-1)*stripe_width*a_dim2+&
                            (my_thread-1)*stripe_width*a_dim2*stripe_count)*size_of_datatype
              dev_offset_1 = ((j+a_off-1)*stripe_width+(i-1)*stripe_width*a_dim2+&
                              (my_thread-1)*stripe_width*a_dim2*stripe_count)*size_of_datatype
              num = stripe_width*this_chunk*size_of_datatype
#ifdef WITH_GPU_STREAMS
              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> aIntern_dev", successGPU)

              successGPU = gpu_memcpy_async(aIntern_dev+dev_offset, aIntern_dev+dev_offset_1, num, &
                      gpuMemcpyDeviceToDevice, my_stream)
              check_memcpy_gpu("tridi_to_band: aIntern_dev -> aIntern_dev", successGPU)

              successGPU = gpu_stream_synchronize(my_stream)
              check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> aIntern_dev", successGPU)
              ! synchronize streamsPerThread; maybe not neccessary
              successGPU = gpu_stream_synchronize()
              check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> aIntern_dev", successGPU)
#else
              successGPU = gpu_memcpy(aIntern_dev+dev_offset, aIntern_dev+dev_offset_1, num, gpuMemcpyDeviceToDevice)
              check_memcpy_gpu("tridi_to_band: aIntern_dev -> aIntern_dev", successGPU)
#endif
            enddo
          enddo
          !$omp end parallel do
          if (wantDebug) call obj%timer%stop("normal_memcpy")
        end do
      else ! useGPU
        call obj%timer%start("OpenMP parallel" // PRECISION_SUFFIX)
        !$omp parallel do num_threads(max_threads) if(max_threads > 1) &
        !$omp default(none) &
        !$omp private(my_thread, i, j) &
        !$omp shared(max_threads, stripe_count, top_msg_length, next_local_n, &
        !$omp&       aIntern, a_off) &
        !$omp schedule(static, 1)
        do my_thread = 1, max_threads
          do i = 1, stripe_count
            do j = top_msg_length+1, top_msg_length+next_local_n
              aIntern(:,j,i,my_thread) = aIntern(:,j+a_off,i,my_thread)
            enddo
          enddo
        enddo
        !$omp end parallel do
        call obj%timer%stop("OpenMP parallel" // PRECISION_SUFFIX)
      endif ! useGPU
#else /* WITH_OPENMP_TRADITIONAL */
      do i = 1, stripe_count
        if (useGPU) then
          chunk = min(next_local_n,a_off)

          if (chunk < 1) exit

          if (wantDebug) call obj%timer%start("normal_memcpy")
          do j = top_msg_length+1, top_msg_length+next_local_n, chunk
            this_chunk = min(j+chunk-1,top_msg_length+next_local_n)-j+1
            dev_offset = ((j-1)*stripe_width+(i-1)*stripe_width*a_dim2)*size_of_datatype
            dev_offset_1 = ((j+a_off-1)*stripe_width+(i-1)*stripe_width*a_dim2)*size_of_datatype
            num = stripe_width*this_chunk*size_of_datatype
#ifdef WITH_GPU_STREAMS
            successGPU = gpu_stream_synchronize(my_stream)
            check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> aIntern_dev", successGPU)

            successGPU = gpu_memcpy_async(aIntern_dev+dev_offset, aIntern_dev+dev_offset_1, num, &
                    gpuMemcpyDeviceToDevice, my_stream)

            check_memcpy_gpu("tridi_to_band: aIntern_dev -> aIntern_dev", successGPU)

            successGPU = gpu_stream_synchronize(my_stream)
            check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> aIntern_dev", successGPU)
            ! synchronize streamsPerThread; maybe not neccessary
            successGPU = gpu_stream_synchronize()
            check_stream_synchronize_gpu("tridi_to_band: aIntern_dev -> aIntern_dev", successGPU)
#else
            successGPU = gpu_memcpy(aIntern_dev+dev_offset, aIntern_dev+dev_offset_1, num, gpuMemcpyDeviceToDevice)

            check_memcpy_gpu("tridi_to_band: aIntern_dev -> aIntern_dev", successGPU)
#endif
          end do
          if (wantDebug) call obj%timer%stop("normal_memcpy")
        else ! not useGPU
          do j = top_msg_length+1, top_msg_length+next_local_n
            aIntern(:,j,i) = aIntern(:,j+a_off,i)
          end do
        end if
      end do ! stripe_count
#endif /* WITH_OPENMP_TRADITIONAL */

      a_off = 0
    end if
  end do ! sweep
  if (wantDebug) call obj%timer%stop("sweep_loop")

  ! Just for safety:
#ifdef WITH_MPI
  if (ANY(top_send_request    /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR top_send_request ***',my_prow,my_pcol
  if (ANY(bottom_send_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR bottom_send_request ***',my_prow,my_pcol
  if (ANY(top_recv_request    /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR top_recv_request ***',my_prow,my_pcol
  if (ANY(bottom_recv_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR bottom_recv_request ***',my_prow,my_pcol
#endif

  if (my_prow == 0) then

#ifdef WITH_MPI
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
    if (wantDebug) call obj%timer%start("cuda_mpi_waitall")
#else
    if (wantDebug) call obj%timer%start("host_mpi_waitall")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
    call MPI_Waitall(num_result_buffers, result_send_request, MPI_STATUSES_IGNORE, mpierr)
#ifdef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
    if (wantDebug) call obj%timer%stop("cuda_mpi_waitall")
#else
    if (wantDebug) call obj%timer%stop("host_mpi_waitall")
#endif /* WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND */
#endif /* WITH_MPI */
  endif

#ifdef WITH_MPI
  if (ANY(result_send_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR result_send_request ***',my_prow,my_pcol
  if (ANY(result_recv_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR result_recv_request ***',my_prow,my_pcol
#endif

  call obj%get("print_flops",print_flops,error)

#ifdef WITH_MPI
#ifdef HAVE_DETAILED_TIMINGS
  if (print_flops == 1) then

    if (useNonBlockingCollectivesRows) then
      call mpi_iallreduce(kernel_flops, kernel_flops_recv, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_ROWS, &
                          allreduce_request1, mpierr)

      call mpi_iallreduce(kernel_time, kernel_time_recv, 1, MPI_REAL8, MPI_MAX, MPI_COMM_ROWS, &
                          allreduce_request3, mpierr)

      call mpi_wait(allreduce_request1, MPI_STATUS_IGNORE, mpierr)
      kernel_flops = kernel_flops_recv

      call mpi_wait(allreduce_request3, MPI_STATUS_IGNORE, mpierr)
      kernel_time_recv = kernel_time
    else
      call mpi_allreduce(kernel_flops, kernel_flops_recv, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_ROWS, &
                          mpierr)
      kernel_flops = kernel_flops_recv

      call mpi_allreduce(kernel_time, kernel_time_recv, 1, MPI_REAL8, MPI_MAX, MPI_COMM_ROWS, &
                          mpierr)
      kernel_time_recv = kernel_time
    endif

    if (useNonBlockingCollectivesCols) then
      call mpi_iallreduce(kernel_flops, kernel_flops_recv, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_COLS, &
                          allreduce_request2, mpierr)

      call mpi_iallreduce(kernel_time, kernel_time_recv, 1, MPI_REAL8, MPI_MAX, MPI_COMM_COLS, &
                          allreduce_request4, mpierr)

      call mpi_wait(allreduce_request2, MPI_STATUS_IGNORE, mpierr)
      kernel_flops = kernel_flops_recv

      call mpi_wait(allreduce_request4, MPI_STATUS_IGNORE, mpierr)
      kernel_time_recv = kernel_time
    else
      call mpi_allreduce(kernel_flops, kernel_flops_recv, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_COLS, &
                         mpierr)
      kernel_flops = kernel_flops_recv

      call mpi_allreduce(kernel_time, kernel_time_recv, 1, MPI_REAL8, MPI_MAX, MPI_COMM_COLS, &
                          mpierr)
      kernel_time_recv = kernel_time
    endif
  endif
#endif
#endif /* WITH_MPI */

  if (useGPU .and. allComputeOnGPU) then
    ! finally copy q_dev to q
    if (wantDebug) call obj%timer%start("cuda_memcpy")
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("trans_ev_tridi_to_band 1: q_dev -> q", successGPU)

    successGPU =  gpu_memcpy_async(int(loc(q(1,1)),kind=c_intptr_t),  &
                             q_dev, &
                             ldq*matrixCols * size_of_datatype, &
                             gpuMemcpyDeviceToHost, my_stream)
    check_memcpy_gpu("trans_ev_tridi_to_band 1: q_dev -> q", successGPU)

    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("trans_ev_tridi_to_band 1: q_dev -> q", successGPU)
    ! synchronize streamsPerThread; maybe not neccessary
    successGPU = gpu_stream_synchronize()
    check_stream_synchronize_gpu("trans_ev_tridi_to_band 1: q_dev -> q", successGPU)
#else
    successGPU =  gpu_memcpy(int(loc(q(1,1)),kind=c_intptr_t),  &
                             q_dev, &
                             ldq*matrixCols * size_of_datatype, &
                             gpuMemcpyDeviceToHost)
    check_memcpy_gpu("trans_ev_tridi_to_band 1: q_dev -> q", successGPU)
#endif
    if (wantDebug) call obj%timer%stop("cuda_memcpy")

  endif

  if (my_prow==0 .and. my_pcol==0 .and.print_flops == 1) then
      write(error_unit,'(" Kernel time:",f10.3," MFlops: ",es12.5)')  kernel_time, kernel_flops/kernel_time*1.d-6
  endif

  ! deallocate all working space

  if (.not.(useGPU)) then
    nullify(aIntern)
    call free(aIntern_ptr)
  endif

  deallocate(row, stat=istat, errmsg=errorMessage)
  check_deallocate("tridi_to_band: row", istat, errorMessage)

  deallocate(limits, stat=istat, errmsg=errorMessage)
  check_deallocate("tridi_to_band: limits", istat, errorMessage)

  deallocate(result_send_request, stat=istat, errmsg=errorMessage)
  check_deallocate("tridi_to_band: result_send_request", istat, errorMessage)

  deallocate(result_recv_request, stat=istat, errmsg=errorMessage)
  check_deallocate("tridi_to_band: result_recv_request", istat, errorMessage)

  deallocate(result_buffer, stat=istat, errmsg=errorMessage)
  check_deallocate("tridi_to_band: result_buffer", istat, errorMessage)

  if (useGPU) then
    if (allComputeOnGPU) then
      successGPU = gpu_free(result_buffer_dev)
      check_dealloc_gpu("tridi_to_band: result_buffer_dev", successGPU)
      nullify(result_buffer_mpi_fortran_ptr)
    endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      nullify(bcast_buffer)

      successGPU = gpu_free_host(bcast_buffer_host)
      check_host_dealloc_gpu("tridi_to_band: bcast_buffer_host", successGPU)
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    else
      deallocate(bcast_buffer)
    endif
#endif
  else ! useGPU
    deallocate(bcast_buffer, stat=istat, errmsg=errorMessage)
    check_deallocate("tridi_to_band: bcast_buffer", istat, errorMessage)
  endif ! useGPU


  if (useGPU) then
    successGPU = gpu_free(aIntern_dev)
    check_dealloc_gpu("tridi_to_band: aIntern_dev", successGPU)

    if (allComputeOnGPU) then
      successGPU = gpu_free(q_dev)
      check_dealloc_gpu("tridi_to_band: q_dev", successGPU)
      nullify(q_mpi_fortran_ptr)

      successGPU = gpu_free(hh_trans_dev)
      check_dealloc_gpu("tridi_to_band: hh_trans_dev", successGPU)
      nullify(hh_trans_mpi_fortran_ptr)

      successGPU = gpu_free(top_border_recv_buffer_dev)
      check_dealloc_gpu("tridi_to_band: top_border_recv_buffer_dev", successGPU)
      nullify(top_border_recv_buffer_mpi_fortran_ptr)

      successGPU = gpu_free(top_border_send_buffer_dev)
      check_dealloc_gpu("tridi_to_band: top_border_send_buffer_dev", successGPU)
      nullify(top_border_send_buffer_mpi_fortran_ptr)

      successGPU = gpu_free(bottom_border_send_buffer_dev)
      check_dealloc_gpu("tridi_to_band: bottom_border_send_buffer_dev", successGPU)
      nullify(bottom_border_send_buffer_mpi_fortran_ptr)

      successGPU = gpu_free(bottom_border_recv_buffer_dev)
      check_dealloc_gpu("tridi_to_band: bottom_border_recv_buffer_dev", successGPU)
      nullify(bottom_border_recv_buffer_mpi_fortran_ptr)

      nullify(aIntern_mpi_fortran_ptr)
    endif ! allComputeOnGPU

    successGPU = gpu_free(hh_tau_dev)
    check_dealloc_gpu("tridi_to_band: hh_tau_dev", successGPU)

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      nullify(row_group)

      successGPU = gpu_free_host(row_group_host)
      check_host_dealloc_gpu("tridi_to_band: row_group_host", successGPU)
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    else
      deallocate(row_group)
    endif
#endif

    successGPU = gpu_free(row_group_dev)
    check_dealloc_gpu("tridi_to_band: row_group_dev", successGPU)

    if (allComputeOnGPU) then
      nullify(row_group_mpi_fortran_ptr)

      successGPU = gpu_free(row_dev)
      check_dealloc_gpu("tridi_to_band: row_dev", successGPU)
      nullify(row_mpi_fortran_ptr)
    endif ! allComputeOnGPU

    successGPU =  gpu_free(bcast_buffer_dev)
    check_dealloc_gpu("tridi_to_band: bcast_buffer_dev", successGPU)

    if (allComputeOnGPU) then
      nullify(bcast_buffer_mpi_fortran_ptr)
    endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      successGPU = gpu_host_unregister(int(loc(top_border_send_buffer),kind=c_intptr_t))
      check_host_unregister_gpu("tridi_to_band: top_border_send_buffer", successGPU)

      successGPU = gpu_host_unregister(int(loc(top_border_recv_buffer),kind=c_intptr_t))
      check_host_unregister_gpu("tridi_to_band: top_border_recv_buffer", successGPU)

      successGPU = gpu_host_unregister(int(loc(bottom_border_send_buffer),kind=c_intptr_t))
      check_host_unregister_gpu("tridi_to_band: bottom_border_send_buffer", successGPU)

      successGPU = gpu_host_unregister(int(loc(bottom_border_recv_buffer),kind=c_intptr_t))
      check_host_unregister_gpu("tridi_to_band: bottom_border_recv_buffer", successGPU)
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    endif
#endif

#ifdef WITH_GPU_STREAMS
   if (allComputeOnGPU) then
     successGPU = gpu_host_unregister(int(loc(q),kind=c_intptr_t))
     check_host_unregister_gpu("tridi_to_band: q", successGPU)

     successGPU = gpu_host_unregister(int(loc(hh_trans),kind=c_intptr_t))
     check_host_unregister_gpu("tridi_to_band: hh_trans", successGPU)
   endif

   !successGPU = gpu_host_unregister(int(loc(aIntern),kind=c_intptr_t))
   !check_host_unregister_gpu("tridi_to_band: aIntern", successGPU)
#endif
  endif ! useGPU

  deallocate(top_border_send_buffer, stat=istat, errmsg=errorMessage)
  check_deallocate("tridi_to_band: top_border_send_buffer", istat, errorMessage)

  deallocate(top_border_recv_buffer, stat=istat, errmsg=errorMessage)
  check_deallocate("tridi_to_band: top_border_recv_buffer", istat, errorMessage)

  deallocate(bottom_border_send_buffer, stat=istat, errmsg=errorMessage)
  check_deallocate("tridi_to_band: bottom_border_send_buffer", istat, errorMessage)

  deallocate(bottom_border_recv_buffer, stat=istat, errmsg=errorMessage)
  check_deallocate("tridi_to_band: bottom_border_recv_buffer", istat, errorMessage)

  deallocate(top_send_request, stat=istat, errmsg=errorMessage)
  check_deallocate("tridi_to_band: top_send_request", istat, errorMessage)

  deallocate(top_recv_request, stat=istat, errmsg=errorMessage)
  check_deallocate("tridi_to_band: top_recv_request", istat, errorMessage)

  deallocate(bottom_send_request, stat=istat, errmsg=errorMessage)
  check_deallocate("tridi_to_band: bottom_send_request", istat, errorMessage)

  deallocate(bottom_recv_request, stat=istat, errmsg=errorMessage)
  check_deallocate("tridi_to_band: bottom_recv_request", istat, errorMessage)

#ifdef WITH_OPENMP_TRADITIONAL
  if (useGPU) then
    call omp_set_num_threads(max_threads_in)
  endif
#endif
  call obj%timer%stop("trans_ev_tridi_to_band_&
                      &MATH_DATATYPE&
                      &" // &
                      &PRECISION_SUFFIX //&
                      gpuString)
  return

end subroutine

! vim: syntax=fortran
