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
!
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
#endif

#include "../general/sanity.F90"

!> \brief Transforms the eigenvectors of a tridiagonal matrix back
!>                     to the eigenvectors of the original matrix
!>                     (like Scalapack Routine PDORMTR)
!>
!  Parameters
!
!> \param na          Order of matrix a_mat, number of rows of matrix q_mat
!>
!> \param nqc         Number of columns of matrix q_mat
!>
!> \param a_mat(lda,matrixCols)  Matrix containing the Householder vectors (i.e. matrix a after tridiag_real)
!>                           Distribution is like in Scalapack.
!>
!> \param lda         Leading dimension of a_mat
!>
!> \param tau(na)     Factors of the Householder vectors
!>
!> \param q_mat       On input: Eigenvectors of tridiagonal matrix
!>                    On output: Transformed eigenvectors
!>                    Distribution is like in Scalapack.
!>
!> \param ldq         Leading dimension of q_mat
!>
!> \param nblk        blocksize of cyclic distribution, must be the same in both directions!
!>
!> \param matrixCols  local columns of matrix a_mat and q_mat
!>
!> \param mpi_comm_rows        MPI-Communicator for rows
!>
!> \param mpi_comm_cols        MPI-Communicator for columns
!>
!> \param useGPU      If true,  GPU version of the subroutine will be used
!>

#undef USE_CCL_TRANS_EV
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
#define USE_CCL_TRANS_EV
#endif

#ifdef TRANS_EV_GPU
subroutine trans_ev_gpu_&
        &MATH_DATATYPE&
        &_&
        &PRECISION &
        (obj, na, nqc, a_dev, lda, tau_dev, q_dev, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug, success)
#else
subroutine trans_ev_cpu_&
        &MATH_DATATYPE&
        &_&
        &PRECISION &
        (obj, na, nqc, a_mat, lda, tau, q_mat, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug, success)
#endif

  use, intrinsic :: iso_c_binding
  use precision
  use elpa_abstract_impl
  use elpa_blas_interfaces
  use elpa_gpu
  use elpa_gpu_util
  use trans_ev_gpu
#if defined(WITH_NVIDIA_GPU_VERSION) && defined(WITH_NVTX)
  use cuda_functions ! for NVTX labels
#elif defined(WITH_AMD_GPU_VERSION) && defined(WITH_ROCTX)
  use hip_functions  ! for ROCTX labels
#endif
  use elpa_ccl_gpu

  implicit none
#include "../general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout)    :: obj
  integer(kind=ik), intent(in)                  :: na, nqc, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols

  integer(kind=c_intptr_t)                      :: a_dev, tau_dev, q_dev
#ifndef TRANS_EV_GPU
  MATH_DATATYPE(kind=rck), intent(in)           :: tau(na)

#ifdef USE_ASSUMED_SIZE
  MATH_DATATYPE(kind=rck), intent(inout)        :: a_mat(lda,*)
  MATH_DATATYPE(kind=rck), intent(inout)        :: q_mat(ldq,*)
#else
  MATH_DATATYPE(kind=rck), intent(inout)        :: a_mat(lda,matrixCols)
  MATH_DATATYPE(kind=rck), intent(inout)        :: q_mat(ldq,matrixCols)
#endif
#else /* TRANS_EV_GPU */
  MATH_DATATYPE(kind=rck)                       :: tau(na)
  MATH_DATATYPE(kind=rck)                       :: a_mat(lda,matrixCols)
  MATH_DATATYPE(kind=rck)                       :: q_mat(ldq,matrixCols)
#endif /* TRANS_EV_GPU */

  integer(kind=ik)                              :: max_stored_rows, max_stored_rows_fac

  integer(kind=ik)                              :: my_prow, my_pcol, np_rows, np_cols
  integer(kind=MPI_KIND)                        :: mpierr, my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI
  integer(kind=ik)                              :: totalblocks, max_blocks_row, max_blocks_col, max_local_rows, max_local_cols
  integer(kind=ik)                              :: l_cols, l_rows, l_colh, nstor
  integer(kind=ik)                              :: istep, n, nc, ic, ics, ice, nb, cur_pcol
  integer(kind=ik)                              :: blockStep

  MATH_DATATYPE(kind=rck), allocatable          :: hvb(:), hvm(:,:)
  MATH_DATATYPE(kind=rck), pointer              :: tmp(:)
  MATH_DATATYPE(kind=rck), allocatable          :: h(:), tmp_debug(:)
  MATH_DATATYPE(kind=rck), pointer              :: tmat(:,:)
  MATH_DATATYPE(kind=rck), pointer              :: hvm1(:)
  
  type(c_ptr)                                   :: tmp_host, hvm1_host, tmat_host
  integer(kind=c_intptr_t)                      :: tmp_dev, hvb_dev, hvm_dev, tmat_dev, h_dev, h1_buffer_dev
  integer(kind=c_intptr_t)                      :: shift_dev, shift_h_dev
  integer(kind=c_intptr_t)                      :: num, num_el
  
  character(200)                                :: errorMessage
  integer(kind=ik)                              :: istat, error
  integer(kind=MPI_KIND)                        :: bcast_request1, allreduce_request1, allreduce_request2
  logical                                       :: useNonBlockingCollectivesCols
  logical                                       :: useNonBlockingCollectivesRows
  integer(kind=c_int)                           :: non_blocking_collectives_rows, non_blocking_collectives_cols
  logical                                       :: success

  logical                                       :: useGPU
  logical                                       :: successGPU, wantDebug
  integer(kind=c_intptr_t), parameter           :: size_of_datatype = size_of_&
                                                                      &PRECISION&
                                                                      &_&
                                                                      &MATH_DATATYPE
  integer(kind=c_intptr_t)                      :: gpublasHandle, my_stream
  character(20)                                 :: gpuString
  integer(kind=ik)                              :: SM_count, debug, useCCL_int

  logical                                       :: useCCL
  integer(kind=c_intptr_t)                      :: ccl_comm_rows, ccl_comm_cols
  integer(kind=c_int)                           :: cclDataType
  integer(kind=ik)                              :: k_datatype
  integer(kind=ik)                              :: i,j ! PETERDEBUG: only for debugging, cleanup

  success = .true.

  debug = 0
  if (wantDebug) debug = 1

  useGPU = .false.
#ifdef TRANS_EV_GPU
  useGPU = .true.
  gpublasHandle = obj%gpu_setup%gpublasHandleArray(0)
  SM_count = obj%gpu_setup%gpuSMcount
#endif

#ifdef WITH_GPU_STREAMS
  my_stream = obj%gpu_setup%my_stream
#endif

  if(useGPU) then
    gpuString = "_gpu"
  else
    gpuString = ""
  endif

  useCCL = .false.
  useCCL_int = 0
#if defined(USE_CCL_TRANS_EV)
  if (useGPU) then
    useCCL = .true.
    useCCL_int = 1
  
    ccl_comm_rows = obj%gpu_setup%ccl_comm_rows
    ccl_comm_cols = obj%gpu_setup%ccl_comm_cols

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
  endif ! useGPU
#endif /* defined(USE_CCL_TRANS_EV) */

  call obj%timer%start("trans_ev_&
  &MATH_DATATYPE&
  &" // &
  &PRECISION_SUFFIX //&
  gpuString)

  if (useGPU) then

    num = lda * matrixCols * size_of_datatype
#ifdef WITH_GPU_STREAMS
    call gpu_memcpy_async_and_stream_synchronize &
         ("trans_ev a_dev -> a_mat", a_dev, 0_c_intptr_t, &
                            a_mat(1:lda,1:matrixCols), &
                            1, 1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
    successGPU = gpu_memcpy(int(loc(a_mat(1,1)),kind=c_intptr_t), a_dev, num, gpuMemcpyDeviceToHost)
    check_memcpy_gpu("trans_ev", successGPU)
#endif

    num = na * size_of_datatype
#ifdef WITH_GPU_STREAMS
    call gpu_memcpy_async_and_stream_synchronize &
         ("trans_ev tau_dev -> tau", tau_dev, 0_c_intptr_t, &
                            tau(1:na), &
                            1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
    successGPU = gpu_memcpy(int(loc(tau(1)),kind=c_intptr_t), tau_dev, num, gpuMemcpyDeviceToHost)
    check_memcpy_gpu("trans_ev", successGPU)
#endif
  endif ! useGPU


  call obj%get("nbc_row_elpa1_tridi_to_full", non_blocking_collectives_rows, error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "Problem setting option for non blocking collectives for rows in elpa1_tridi_to_full. Aborting..."
    call obj%timer%stop("trans_ev_&
    &MATH_DATATYPE&
    &" // &
    &PRECISION_SUFFIX //&
    gpuString)
    success = .false.
    return
  endif

  call obj%get("nbc_col_elpa1_tridi_to_full", non_blocking_collectives_cols, error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "Problem setting option for non blocking collectives for cols in elpa1_tridi_to_full. Aborting..."
    call obj%timer%stop("trans_ev_&
    &MATH_DATATYPE&
    &" // &
    &PRECISION_SUFFIX //&
    gpuString)
    success = .false.
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

  my_prow = obj%mpi_setup%myRank_comm_rows
  my_pcol = obj%mpi_setup%myRank_comm_cols

  np_rows = obj%mpi_setup%nRanks_comm_rows
  np_cols = obj%mpi_setup%nRanks_comm_cols

  call obj%get("max_stored_rows",max_stored_rows_fac, error)  ! PETERDEBUG: default 256. 
                                                              ! check whether for GPU it should be increased
                                                              ! size for GEMM should be at least 5000 (A100)

  totalblocks = (na-1)/nblk + 1
  max_blocks_row = (totalblocks-1)/np_rows + 1
  max_blocks_col = ((nqc-1)/nblk)/np_cols + 1  ! Columns of q_mat!

  max_local_rows = max_blocks_row*nblk
  max_local_cols = max_blocks_col*nblk

  max_stored_rows = (max_stored_rows_fac/nblk+1)*nblk
  
  !print *, "max_stored_rows=", max_stored_rows ! PETERDEBUG

  if (.not. useGPU) then
    allocate(tmat(max_stored_rows,max_stored_rows), stat=istat, errmsg=errorMessage)
    call check_alloc("trans_ev", "tmat", istat, errorMessage)

    allocate(tmp(max_local_cols*max_stored_rows), stat=istat, errmsg=errorMessage)
    call check_alloc("trans_ev", "tmp", istat, errorMessage)
  endif

  allocate(h(max_stored_rows*max_stored_rows), stat=istat, errmsg=errorMessage)
  call check_alloc("trans_ev", "h", istat, errorMessage)

  allocate(hvb(max_local_rows*nblk), stat=istat, errmsg=errorMessage)
  call check_alloc("trans_ev", "hvb", istat, errorMessage)

  allocate(hvm(max_local_rows,max_stored_rows), stat=istat, errmsg=errorMessage)
  call check_alloc("trans_ev", "hvm", istat, errorMessage)
 

  if (useGPU) then
    ! todo: this is used only for copying hmv to device.. it should be possible to go without it
    !allocate(hvm1(max_local_rows*max_stored_rows), stat=istat, errmsg=errorMessage)
    !call check_alloc("trans_ev_&
    !&MATH_DATATYPE&
    !&", "hvm1", istat, errorMessage)

    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
      num = (max_local_rows*max_stored_rows) * size_of_datatype
      successGPU = gpu_malloc_host(hvm1_host,num)
      check_alloc_gpu("trans_ev: hvm1_host", successGPU)
      call c_f_pointer(hvm1_host,hvm1,(/(max_local_rows*max_stored_rows)/))
    else
      allocate(hvm1(max_local_rows*max_stored_rows))
    endif

    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
      num = (max_stored_rows*max_stored_rows) * size_of_datatype
      successGPU = gpu_malloc_host(tmat_host,num)
      check_alloc_gpu("trans_ev: tmat_host", successGPU)
      call c_f_pointer(tmat_host,tmat,(/max_stored_rows,max_stored_rows/))
    else
      allocate(tmat(max_stored_rows,max_stored_rows))
    endif

    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
      num = (max_local_cols*max_stored_rows) * size_of_datatype
      successGPU = gpu_malloc_host(tmp_host,num)
      check_alloc_gpu("trans_ev: tmp_host", successGPU)
      call c_f_pointer(tmp_host, tmp, (/(max_local_cols*max_stored_rows)/))
    else
      allocate(tmp(max_local_cols*max_stored_rows))
    endif

    successGPU = gpu_malloc(tmat_dev, max_stored_rows * max_stored_rows * size_of_datatype)
    check_alloc_gpu("trans_ev", successGPU)

    successGPU = gpu_malloc(hvb_dev, max_local_rows * nblk * size_of_datatype)
    check_alloc_gpu("trans_ev", successGPU)

    successGPU = gpu_malloc(hvm_dev, max_local_rows * max_stored_rows * size_of_datatype)
    check_alloc_gpu("trans_ev", successGPU)

    successGPU = gpu_malloc(tmp_dev, max_local_cols * max_stored_rows * size_of_datatype)
    check_alloc_gpu("trans_ev", successGPU)

    successGPU = gpu_malloc(h_dev, max_stored_rows * max_stored_rows * size_of_datatype)
    check_alloc_gpu("trans_ev", successGPU)

    successGPU = gpu_malloc(h1_buffer_dev, max_stored_rows * size_of_datatype)
    check_alloc_gpu("trans_ev", successGPU)

    !if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
    !  successGPU = gpu_host_register(int(loc(q_mat),kind=c_intptr_t),num,&
    !              gpuHostRegisterDefault)
    !  check_host_register_gpu("trans_ev: q_mat", successGPU)
    !endif
  endif  ! useGPU


  hvb = 0   ! Safety only ! PETERDEBUG: check whether it is really needed
  hvm = 0   ! Must be set to 0 !!!

  ! PETERDEBUG: check whether it is really needed
  if (useGPU) then
    num = max_local_rows*max_stored_rows * size_of_datatype
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    successGPU = gpu_memset_async(hvm_dev, 0, num, my_stream)
    check_memcpy_gpu("trans_ev: hvm_dev", successGPU)
#else
    successGPU = gpu_memset(hvm_dev, 0, num)
#endif
  endif

  blockStep = nblk
  l_cols = local_index(nqc, my_pcol, np_cols, nblk, -1) ! Local columns of q_mat

  nstor = 0


#if COMPLEXCASE == 1
  ! In the complex case tau(2) /= 0
  if (useGPU) then
    if (my_prow == prow(1, nblk, np_rows)) then
      call GPU_SCALE_QMAT_PRECISION(ldq, l_cols, q_dev, tau_dev, my_stream)
     endif
  else
    if (my_prow == prow(1, nblk, np_rows)) then
      q_mat(1,1:l_cols) = q_mat(1,1:l_cols)*(ONE-tau(2))
    endif
  endif
#endif

  do istep = 1, na, blockStep
    NVTX_RANGE_PUSH("main_loop")
    call obj%timer%start("main_loop")

    ics = MAX(istep,3)
    ice = MIN(istep+nblk-1,na)
    if (ice<ics) cycle

    cur_pcol = pcol(istep, nblk, np_cols)

    if (useGPU) then
      if (my_pcol==cur_pcol) then
        call obj%timer%start("gpu_copy_hvb_a_kernel")
        NVTX_RANGE_PUSH("gpu_copy_hvb_a")
        call gpu_copy_hvb_a(PRECISION_CHAR, hvb_dev, a_dev, max_local_rows, lda, my_prow, np_rows, my_pcol, np_cols, &
                                                  nblk, ics, ice, SM_count, debug, my_stream)
        NVTX_RANGE_POP("gpu_copy_hvb_a")
        call obj%timer%stop("gpu_copy_hvb_a_kernel")
      endif

      !dry run to find nb, number of used elements in hvb_dev
      ! do ic = ics, ice
      !   l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1)
      !   nb = nb+l_rows
      ! enddo
      nb = max_local_rows*nblk ! no compression
    else ! useGPU
      NVTX_RANGE_PUSH("loop: copy hvb <- a_mat")
      nb = 0
      ! PETERDEBUG: typically, ice-ics = nblk-1, hence nblk steps
      do ic = ics, ice
        l_colh = local_index(ic  , my_pcol, np_cols, nblk, -1) ! Column of Householder Vector
        l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1) ! # rows of Householder Vector

        if (my_pcol == cur_pcol) then
          hvb(nb+1:nb+l_rows) = a_mat(1:l_rows,l_colh)
          if (my_prow == prow(ic-1, nblk, np_rows)) then
            hvb(nb+l_rows) = 1.
          endif
        endif

        nb = nb+l_rows
      enddo
      NVTX_RANGE_POP("loop: copy hvb <- a_mat")
    endif

    ! PETERDEBUG: do we need at all this compression-decompression?
    ! max_local_rows -> "l_rows_max"

    if (useGPU .and. .not. useCCL) then
      num = nb * size_of_datatype
#ifdef WITH_GPU_STREAMS
      call gpu_memcpy_async_and_stream_synchronize &
              ("trans_ev hvb_dev -> hvb", hvb_dev, 0_c_intptr_t, &
              hvb(1:nb), 1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
      successGPU = gpu_memcpy(int(loc(hvb(1)),kind=c_intptr_t), hvb_dev, num, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("trans_ev", successGPU)
#endif
    endif ! useGPU

#ifdef WITH_MPI
    if (nb > 0) then
      if (useCCL) then
        NVTX_RANGE_PUSH("ccl_bcast")
        call obj%timer%start("ccl_bcast")

        successGPU = ccl_bcast(hvb_dev, hvb_dev, int(k_datatype*nb,kind=c_size_t), cclDatatype, &
                               int(cur_pcol,kind=c_int), ccl_comm_cols, my_stream)

        if (.not. successGPU) then
          print *,"Error in ccl_bcast"
          stop 1
        endif

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("trans_ev: ccl_bcast", successGPU)
        
        call obj%timer%stop("ccl_bcast")
        NVTX_RANGE_POP("ccl_bcast")
      else ! useCCL
        NVTX_RANGE_PUSH("mpi_bcast")
        if (useNonBlockingCollectivesCols) then
          call obj%timer%start("mpi_nbc_communication")
          call mpi_ibcast(hvb, int(nb,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION , int(cur_pcol,kind=MPI_KIND), &
                          int(mpi_comm_cols,kind=MPI_KIND), bcast_request1, mpierr)
          call mpi_wait(bcast_request1, MPI_STATUS_IGNORE, mpierr)
          call obj%timer%stop("mpi_nbc_communication")
        else
          call obj%timer%start("mpi_communication")
          call mpi_bcast (hvb, int(nb,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION , int(cur_pcol,kind=MPI_KIND), &
                          int(mpi_comm_cols,kind=MPI_KIND), mpierr)
          call obj%timer%stop("mpi_communication")
        endif
      NVTX_RANGE_POP("mpi_bcast")
      endif ! useCCL
    endif ! (nb > 0)
#endif /* WITH_MPI */

    if (useGPU .and. .not. useCCL) then
#ifdef WITH_GPU_STREAMS
      num_el = max_local_rows*nblk
      call gpu_memcpy_async_and_stream_synchronize("trans_ev hvb -> hvb_dev", &
                hvb_dev, 0_c_intptr_t, hvb(1:num_el), 1, &
                num_el*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
      successGPU = gpu_memcpy(hvb_dev, int(loc(hvb(1)),kind=c_intptr_t), num_el*size_of_datatype, gpuMemcpyHostToDevice)
      check_memcpy_gpu("trans_ev", successGPU)
#endif
    endif ! useGPU

    if (useGPU) then
      call obj%timer%start("gpu_copy_hvm_hvb_kernel")
      NVTX_RANGE_PUSH("gpu_copy_hvm_hvb")
      call gpu_copy_hvm_hvb(PRECISION_CHAR, hvm_dev, hvb_dev, max_local_rows, max_local_rows, my_prow, np_rows, &
                            nstor, nblk, ics, ice, SM_count, debug, my_stream)
      NVTX_RANGE_POP("gpu_copy_hvm_hvb")
      call obj%timer%stop("gpu_copy_hvm_hvb_kernel")

      l_rows = local_index(ice-1, my_prow, np_rows, nblk, -1) ! last l_rows
      nstor =  nstor + (ice-ics+1)
    else ! useGPU
      nb = 0
      NVTX_RANGE_PUSH("loop: copy hvm <- hvb")
      do ic = ics, ice
        l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1) ! # rows of Householder Vector
        hvm(1:l_rows,nstor+1) = hvb(nb+1:nb+l_rows)
        nstor = nstor+1
        nb = nb+l_rows
      enddo
      NVTX_RANGE_POP("loop: copy hvm <- hvb")
    endif ! useGPU

    ! PETERDEBUG: this memcopy can be cleaned up?
    if (useGPU .and. .not. useCCL) then
      num = max_local_rows * max_stored_rows * size_of_datatype
#ifdef WITH_GPU_STREAMS
      call gpu_memcpy_async_and_stream_synchronize &
              ("trans_ev hvm_dev -> hvm", hvm_dev, 0_c_intptr_t, &
              hvm(1:max_local_rows,1:max_stored_rows), &
              1, 1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
      successGPU = gpu_memcpy(int(loc(hvm(1,1)),kind=c_intptr_t), hvm_dev, num, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("trans_ev hvm_dev -> hvm", successGPU)
#endif
    endif ! useGPU

    ! PETERDEBUG: print array
  !   print *, "istep=", istep, "nstor=", nstor, "max_stored_rows=", max_stored_rows ! PETEDEBUG
  !   do i = 1, size(hvm, 1)
  !     do j = 1, MIN(size(hvm, 2), na)
  !        write(*,'(F8.2)', advance='no') hvm(i,j)
  !     end do
  !     write(*,*)  ! move to next line
  !  end do
    
    ! PETERDEBUG: for GPU, and big max_stored_rows, isn't it more efficient to do one MPI_send instead of many?
    ! Please note: for smaller matix sizes (na/np_rows<=256), a value of 32 for nstor is enough!
    if (nstor+nblk > max_stored_rows .or. istep+nblk > na .or. (na/np_rows <= 256 .and. nstor >= 32)) then

      ! Calculate scalar products of stored vectors.
      ! This can be done in different ways, we use dsyrk or zherk

      if (useGPU) then
        call obj%timer%start("gpu_memset")
        ! PETERDEBUG: is this really needed for GPU?
        num_el = max_stored_rows*max_stored_rows
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memset_async(tmat_dev, 0, num_el*size_of_datatype, my_stream)
        if (wantDebug) successGPU = gpu_DeviceSynchronize()
#else
        successGPU = gpu_memset(tmat_dev, 0, num_el*size_of_datatype)
#endif
        check_memcpy_gpu("trans_ev: tmat_dev", successGPU)
        call obj%timer%stop("gpu_memset")
      else
        tmat = 0
      endif

      if (l_rows>0) then
        if (useGPU) then
          !successGPU = gpu_memset(tmat_dev, 0, max_stored_rows*max_stored_rows * size_of_datatype) ! PETERDEBUG

          call obj%timer%start("gpublas_syrk")
          NVTX_RANGE_PUSH("gpublas_syrk")
          call gpublas_PRECISION_SYRK_HERK('U', BLAS_TRANS_OR_CONJ, &
                                           nstor, l_rows, ONE, &
                                           hvm_dev, max_local_rows, ZERO, &
                                           tmat_dev, max_stored_rows, gpublasHandle)
          if (wantDebug) successGPU = gpu_DeviceSynchronize()
          NVTX_RANGE_POP("gpublas_syrk")
          call obj%timer%stop("gpublas_syrk")
        else ! useGPU
          call obj%timer%start("blas_syrk")
          NVTX_RANGE_PUSH("blas_syrk")
#if REALCASE == 1
          call PRECISION_SYRK &
#elif COMPLEXCASE == 1
          call PRECISION_HERK &
#endif
                            ('U', BLAS_TRANS_OR_CONJ, &
                            int(nstor,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), ONE, &
                            hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), ZERO, &
                            tmat, int(max_stored_rows,kind=BLAS_KIND))
          
          NVTX_RANGE_POP("blas_syrk")
          call obj%timer%stop("blas_syrk")
        endif ! useGPU
      endif ! (l_rows>0)

      if (useGPU .and. .not. useCCL) then
        num_el = max_stored_rows*max_stored_rows
#ifdef WITH_GPU_STREAMS
        call gpu_memcpy_async_and_stream_synchronize &
                ("trans_ev: tmat_dev -> tmat", tmat_dev, 0_c_intptr_t, &
                tmat(1:max_stored_rows,1:max_stored_rows), &
                1, 1, num_el*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else              
        successGPU = gpu_memcpy(int(loc(tmat(1,1)),kind=c_intptr_t), tmat_dev, num_el*size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("trans_ev", successGPU)
#endif
      endif ! useGPU
      
      if (useCCL) then
        ! no compression
        !nc = max_stored_rows*max_stored_rows
        nc = max_stored_rows*nstor
        if (nc>0) then
          if (wantDebug) call obj%timer%start("gpu_memcpy")
          ! PETERDEBUG: add streamed version
          successGPU = gpu_memcpy(h_dev, tmat_dev, nc*size_of_datatype, gpuMemcpyDeviceToDevice)
          check_memcpy_gpu("elpa_trans_ev: h_dev <- tmat_dev", successGPU)
          if (wantDebug) call obj%timer%stop("gpu_memcpy")
        endif
      else  ! useCCL
        ! compression
        nc = 0
        do n = 1, nstor-1
          h(nc+1:nc+n) = tmat(1:n,n+1)
          nc = nc+n ! PETERDEBUG: on GPU: nc += max_stored_rows 
        enddo
      endif ! useCCL

#ifdef WITH_MPI
      if (nc > 0) then
        if (useCCL) then
          NVTX_RANGE_PUSH("ccl_allreduce")
          call obj%timer%start("ccl_allreduce")
          successGPU = ccl_allreduce(h_dev, h_dev, int(k_datatype*nc,kind=c_size_t), &
                                     cclDatatype, cclSum, ccl_comm_rows, my_stream)
          
          if (.not. successGPU) then
            print *,"Error in ccl_allreduce"
            stop 1
          endif

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev", successGPU)
          call obj%timer%stop("ccl_allreduce")
          NVTX_RANGE_POP("ccl_allreduce")
        else ! useCCL
          if (useNonBlockingCollectivesRows) then
            call obj%timer%start("mpi_nbc_communication")
            call mpi_iallreduce(MPI_IN_PLACE, h, int(nc,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                                int(mpi_comm_rows,kind=MPI_KIND), allreduce_request1, mpierr)
            call mpi_wait(allreduce_request1, MPI_STATUS_IGNORE, mpierr)
            call obj%timer%stop("mpi_nbc_communication")
          else
            call obj%timer%start("mpi_communication")
            call mpi_allreduce (MPI_IN_PLACE, h, int(nc,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                                int(mpi_comm_rows,kind=MPI_KIND), mpierr)
            call obj%timer%stop("mpi_communication")
          endif
        endif ! useCCL
      endif ! (nc > 0)
#endif /* WITH_MPI */

      if (useGPU .and. .not. useCCL) then
        num_el = nc
#ifdef WITH_GPU_STREAMS
        call gpu_memcpy_async_and_stream_synchronize("trans_ev: h -> h_dev", h_dev, 0_c_intptr_t, &
                                                      h(1:num_el), 1, num_el*size_of_datatype, &
                                                      gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else /* WITH_GPU_STREAMS */
        successGPU = gpu_memcpy(h_dev, int(loc(h(1)),kind=c_intptr_t), num_el*size_of_datatype, gpuMemcpyHostToDevice)
        check_memcpy_gpu("trans_ev", successGPU)
#endif /* WITH_GPU_STREAMS */
      endif ! useGPU

      ! Calculate triangular matrix T. 
      ! What was previously stored in (upper part of) tmat is now stored in h, old values of tmat are not needed anymore

      ! PETERDEBUG: we can write a kernel here
      ! Also, we can use another stream for this, which can be completely hidden by other operations
      if (useGPU) then

        ! PETERDEBUG: check whether this can be cleaned up after testing gpu_update_h_trmv_kernel
        call obj%timer%start("gpu_memset")
        successGPU = gpu_memset(tmat_dev, 0, max_stored_rows*max_stored_rows * size_of_datatype)
        call obj%timer%stop("gpu_memset")

        nc = 0
        n = 0
        shift_dev = (ice-nstor+n)*size_of_datatype
        shift_h_dev = 0
        call gpu_update_tmat(PRECISION_CHAR, tmat_dev, h_dev, tau_dev + shift_dev, &
                             max_stored_rows, nc, n, SM_count, debug, my_stream)
        
        !NVTX_RANGE_PUSH("gpublas_trmv+gpu_update_tmat loop")
        
        NVTX_RANGE_PUSH("gpu_trmv")
        call gpu_trmv(PRECISION_CHAR, tmat_dev, h_dev+shift_h_dev, h1_buffer_dev, tau_dev+shift_dev, &
                      max_stored_rows, n, SM_count, debug, my_stream)
        if (wantDebug) successGPU = gpu_DeviceSynchronize()
        NVTX_RANGE_POP("gpu_trmv")

        call obj%timer%start("gpu_trmv_kernel_loop")
        NVTX_RANGE_PUSH("trmv_loop")
        ! call gpu_trmv_loop(PRECISION_CHAR, tmat_dev, h_dev, h1_buffer_dev, tau_dev, &
        !                   max_stored_rows, nstor, ice, SM_count, useCCL_int, debug, my_stream)

        do n = 1, nstor-1
          !shift_dev = nc*size_of_datatype
          !h_dev <- tmat_dev*h_dev

         if (useCCL) then
           shift_h_dev = n*max_stored_rows*size_of_datatype
         else
           shift_h_dev = nc*size_of_datatype
           nc = nc+n
         endif

          ! NVTX_RANGE_PUSH("gpublas_trmv")
          ! call gpublas_PRECISION_TRMV('L', BLAS_TRANS_OR_CONJ, 'N', n, &
          !                             tmat_dev, max_stored_rows, &
          !                             h_dev + shift_h_dev, 1, gpublasHandle)
          ! if (wantDebug) successGPU = gpu_DeviceSynchronize()
          ! NVTX_RANGE_POP("gpublas_trmv")

          ! non-transposed matrix tmat_dev here, transposition later in TRMM
          ! NVTX_RANGE_PUSH("gpublas_trmv")
          ! call gpublas_PRECISION_TRMV('U', 'N', 'N', n, &
          !                             tmat_dev, max_stored_rows, &
          !                             h_dev + shift_h_dev, 1, gpublasHandle)
          ! if (wantDebug) successGPU = gpu_DeviceSynchronize()
          ! NVTX_RANGE_POP("gpublas_trmv")

          ! shift_dev = (ice-nstor+n)*size_of_datatype
          ! NVTX_RANGE_PUSH("gpu_update_tmat")
          ! call gpu_update_tmat(PRECISION_CHAR, tmat_dev, h_dev+shift_h_dev, tau_dev+shift_dev, &
          !                      max_stored_rows, nc, n, SM_count, debug, my_stream)
          ! NVTX_RANGE_POP("gpu_update_tmat")

          shift_dev = (ice-nstor+n)*size_of_datatype
          NVTX_RANGE_PUSH("gpu_trmv")
          call gpu_trmv(PRECISION_CHAR, tmat_dev, h_dev+shift_h_dev, h1_buffer_dev, tau_dev+shift_dev, &
                        max_stored_rows, n, SM_count, debug, my_stream)
          NVTX_RANGE_POP("gpu_trmv")

          ! shift_dev = (ice-nstor+n)*size_of_datatype
          ! NVTX_RANGE_PUSH("gpu_update_tmat")
          ! call gpu_update_tmat(PRECISION_CHAR, tmat_dev, h1_buffer_dev, tau_dev+shift_dev, &
          !                      max_stored_rows, nc, n, SM_count, debug, my_stream)
          ! NVTX_RANGE_POP("gpu_update_tmat")
        enddo
        if (wantDebug) successGPU = gpu_DeviceSynchronize()
        call obj%timer%stop("gpu_trmv_kernel_loop")
        NVTX_RANGE_POP("trmv_loop")
        
        !NVTX_RANGE_POP("gpublas_trmv+gpu_update_tmat loop")
      else ! useGPU
        nc = 0
        tmat(1,1) = tau(ice-nstor+1)
        do n = 1, nstor-1
          ! h = tmat*h
          call obj%timer%start("blas")
          NVTX_RANGE_PUSH("blas_trmv") ! PETERDEBUG: transform, similarly to GPU branch, if doesn't lose performance
          call PRECISION_TRMV('L', BLAS_TRANS_OR_CONJ, 'N', int(n,kind=BLAS_KIND), &
                              tmat, int(max_stored_rows,kind=BLAS_KIND), &
                              h(nc+1), 1_BLAS_KIND)
          NVTX_RANGE_POP("blas_trmv")
          call obj%timer%stop("blas")

          ! update tmat for next iteration
#if REALCASE == 1
          tmat(n+1,1:n) = -h(nc+1:nc+n) *tau(ice-nstor+n+1)
#elif COMPLEXCASE == 1
          tmat(n+1,1:n) = -conjg(h(nc+1:nc+n)) *tau(ice-nstor+n+1)
#endif
          
          tmat(n+1,n+1) = tau(ice-nstor+n+1)
          nc = nc+n
        enddo
      endif ! useGPU

#if !defined(WITH_GPU_STREAMS)
      if (useGPU) successGPU = gpu_DeviceSynchronize()
#endif

      ! Q = Q - V * T * V**T * Q

      if (l_rows>0) then
        if (useGPU) then
          call obj%timer%start("gpublas_gemm")
          NVTX_RANGE_PUSH("gpublas_gemm")
          call gpublas_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',   &
                                      nstor, l_cols, l_rows, ONE, &
                                      hvm_dev, max_local_rows,  &
                                      q_dev, ldq, ZERO, &
                                      tmp_dev, nstor, gpublasHandle)
          if (wantDebug) successGPU = gpu_DeviceSynchronize()
          NVTX_RANGE_POP("gpublas_gemm")
          call obj%timer%stop("gpublas_gemm")
        else ! useGPU
          call obj%timer%start("blas")
          call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',  &
                              int(nstor,kind=BLAS_KIND), &
                              int(l_cols,kind=BLAS_KIND),&
                              int(l_rows,kind=BLAS_KIND), ONE, &
                              hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), &
                              q_mat, int(ldq,kind=BLAS_KIND), ZERO, &
                              tmp, int(nstor,kind=BLAS_KIND))
          call obj%timer%stop("blas")
        endif ! useGPU

      else ! (l_rows>0)

        if (useGPU) then
          if (wantDebug) call obj%timer%start("gpu_memset")
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          num = l_cols * nstor * size_of_datatype
          successGPU = gpu_memset_async(tmp_dev, 0, num, my_stream)
          check_memcpy_gpu("trans_ev: tmp_dev", successGPU)
          !successGPU = gpu_stream_synchronize(my_stream)
          !check_stream_synchronize_gpu("trans_ev", successGPU)
          if (wantDebug) successGPU = gpu_DeviceSynchronize()
#else
          successGPU = gpu_memset(tmp_dev, 0, l_cols * nstor * size_of_datatype)
#endif
          check_memcpy_gpu("trans_ev", successGPU)
          if (wantDebug) call obj%timer%stop("gpu_memset")
        else
          tmp(1:l_cols*nstor) = 0
        endif
      endif  ! (l_rows>0)

#ifdef WITH_MPI
      if (useGPU .and. .not. useCCL) then
        ! In the legacy GPU version, this allreduce was ommited. But probably it has to be done for GPU + MPI
        ! todo: does it need to be copied whole? Wouldn't be a part sufficient?

        ! copy tmp_dev -> tmp if needed
#ifdef WITH_GPU_STREAMS
        num = max_local_cols * max_stored_rows * size_of_datatype
        call gpu_memcpy_async_and_stream_synchronize &
            ("trans_ev tmp_dev -> tmp", tmp_dev, 0_c_intptr_t, &
                                                 tmp(1:max_local_cols*max_stored_rows), &
                                                 1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
        successGPU = gpu_memcpy(int(loc(tmp(1)),kind=c_intptr_t), tmp_dev,  &
                      max_local_cols * max_stored_rows * size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("trans_ev", successGPU)
#endif
      endif ! (useGPU .and. .not. useCCL)

      if (useCCL) then
        call obj%timer%start("ccl_allreduce")
        NVTX_RANGE_PUSH("ccl_allreduce")
        successGPU = ccl_Allreduce(tmp_dev, tmp_dev, int(k_datatype*nstor*l_cols,kind=c_size_t), &
                                     cclDataType, cclSum, ccl_comm_rows, my_stream)

        if (.not. successGPU) then
          print *,"Error in ccl_allreduce"
          stop
        endif
          
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("trans_ev", successGPU)
        NVTX_RANGE_POP("ccl_allreduce")
        call obj%timer%stop("ccl_allreduce")
      else ! useCCL
        if (useNonBlockingCollectivesRows) then
          call obj%timer%start("mpi_nbc_communication")
          call mpi_iallreduce(MPI_IN_PLACE, tmp, int(nstor*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                             int(mpi_comm_rows,kind=MPI_KIND), allreduce_request2, mpierr)
          call mpi_wait(allreduce_request2, MPI_STATUS_IGNORE, mpierr)
          call obj%timer%stop("mpi_nbc_communication")
        else ! useNonBlockingCollectivesRows
          call obj%timer%start("mpi_communication")
          call mpi_allreduce(MPI_IN_PLACE, tmp, int(nstor*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                             int(mpi_comm_rows,kind=MPI_KIND), mpierr)
          call obj%timer%stop("mpi_communication")
        endif ! useNonBlockingCollectivesRows
      endif ! useCCL

      ! copy back to tmp -> tmp_dev if needed
      if (useGPU .and. .not. useCCL) then
#ifdef WITH_GPU_STREAMS
          num = max_local_cols * max_stored_rows * size_of_datatype
          call gpu_memcpy_async_and_stream_synchronize &
            ("trans_ev tmp -> tmp_dev", tmp_dev, 0_c_intptr_t, &
                                                 tmp(1:max_local_cols*max_stored_rows), &
                                                 1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else /* WITH_GPU_STREAMS */
        successGPU = gpu_memcpy(tmp_dev, int(loc(tmp(1)),kind=c_intptr_t),  &
                      max_local_cols * max_stored_rows * size_of_datatype, gpuMemcpyHostToDevice)
        check_memcpy_gpu("trans_ev", successGPU)
#endif /* WITH_GPU_STREAMS */
      endif ! useGPU

#endif /* WITH_MPI */

      if (l_rows > 0) then
        if (useGPU) then
          
          ! tmp_dev = tmat_dev*tmp_dev
          call obj%timer%start("gpublas_trmm")
          NVTX_RANGE_PUSH("gpublas_trmm")
          call gpublas_PRECISION_TRMM('L', 'L', 'N', 'N',     &
          !call gpublas_PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N',     &
                                      nstor, l_cols, ONE, &
                                      tmat_dev, max_stored_rows,  &
                                      tmp_dev, nstor, gpublasHandle)
          if (wantDebug) successGPU = gpu_DeviceSynchronize()
          call obj%timer%stop("gpublas_trmm")
          NVTX_RANGE_POP("gpublas_trmm")

          NVTX_RANGE_PUSH("gpublas_gemm")
          call obj%timer%start("gpublas_gemm")
          call gpublas_PRECISION_GEMM('N', 'N', &
                                      l_rows, l_cols, nstor, -ONE, &
                                      hvm_dev, max_local_rows, &
                                      tmp_dev, nstor, ONE, &
                                      q_dev, ldq, gpublasHandle)
          if (wantDebug) successGPU = gpu_DeviceSynchronize()
          call obj%timer%stop("gpublas_gemm")
          NVTX_RANGE_POP("gpublas_gemm")
        else !useGPU
          call obj%timer%start("blas")

          ! tmp = tmat * tmp
          call PRECISION_TRMM('L', 'L', 'N', 'N', &
                              int(nstor,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), ONE, &
                              tmat, int(max_stored_rows,kind=BLAS_KIND), &
                              tmp, int(nstor,kind=BLAS_KIND))
          !q_mat = q_mat - hvm*tmp
          call PRECISION_GEMM('N', 'N', &
                              int(l_rows,kind=BLAS_KIND), &
                              int(l_cols,kind=BLAS_KIND), &
                              int(nstor,kind=BLAS_KIND), -ONE, &
                              hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), &
                              tmp, int(nstor,kind=BLAS_KIND), &
                              ONE, q_mat, int(ldq,kind=BLAS_KIND))
          call obj%timer%stop("blas")
        endif ! useGPU
      endif  ! l_rows>0
      nstor = 0
    endif  ! (nstor+nblk>max_stored_rows .or. istep+nblk>na .or. (na/np_rows<=256 .and. nstor>=32))

    ! PETERDEBUG q_mat: ldq,matrixCols
    ! if (my_prow==0 .and. my_pcol==0) then
    !   print *, "istep=", istep, "nstor=", nstor, "max_stored_rows=", max_stored_rows ! PETEDEBUG
    !   do i = 1, size(q_mat, 1)
    !     !do j = 1, MIN(size(q_mat, 2), na)
    !     do j = 1, size(q_mat, 2)
    !         write(*,'(F8.2)', advance='no') q_mat(i,j)
    !     end do
    !     write(*,*)  ! move to next line
    !   end do
    ! endif

    NVTX_RANGE_POP("main_loop")
    call obj%timer%stop("main_loop")
  enddo ! istep = 1, na, blockStep

  deallocate(h, hvb, hvm, stat=istat, errmsg=errorMessage)
  check_deallocate("trans_ev: h, hvb, hvm", istat, errorMessage)

  if (useGPU) then

    num = lda * matrixCols * size_of_datatype
#ifdef WITH_GPU_STREAMS
    ! at least in the real case this memory copy could be done before calling this step
    my_stream = obj%gpu_setup%my_stream
    call gpu_memcpy_async_and_stream_synchronize &
            ("trans_ev a_mat -> a_dev", a_dev, 0_c_intptr_t, &
                                                 a_mat(1:lda,1:matrixCols), &
                                                 1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
    successGPU = gpu_memcpy(a_dev, int(loc(a_mat(1,1)),kind=c_intptr_t), &
                  num, gpuMemcpyHostToDevice)
    check_memcpy_gpu("trans_ev", successGPU)
#endif

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    !if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
    !  successGPU = gpu_host_unregister(int(loc(q_mat),kind=c_intptr_t))
    !  check_host_unregister_gpu("trans_ev: q_mat", successGPU)
    !endif

    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
      successGPU = gpu_free_host(hvm1_host)
      check_host_dealloc_gpu("trans_ev: hvm1_host", successGPU)
      nullify(hvm1)

      successGPU = gpu_free_host(tmat_host)
      check_host_dealloc_gpu("trans_ev: tmat_host", successGPU)
      nullify(tmat)

      successGPU = gpu_free_host(tmp_host)
      check_host_dealloc_gpu("trans_ev: tmp_host", successGPU)
      nullify(tmp)

    else
      deallocate(hvm1)
      deallocate(tmat)
      deallocate(tmp)
    endif
#endif
    !deallocate(hvm1, stat=istat, errmsg=errorMessage)
    !if (istat .ne. 0) then
    !  print *,"trans_ev_&
    !  &MATH_DATATYPE&
    !  &: error when deallocating hvm1 "//errorMessage
    !  stop 1
    !endif

    successGPU = gpu_free(tmp_dev)
    check_dealloc_gpu("trans_ev", successGPU)

    successGPU = gpu_free(hvm_dev)
    check_dealloc_gpu("trans_ev", successGPU)

    successGPU = gpu_free(tmat_dev)
    check_dealloc_gpu("trans_ev", successGPU)

    successGPU = gpu_free(h_dev)
    check_dealloc_gpu("trans_ev", successGPU)

    successGPU = gpu_free(h1_buffer_dev)
    check_dealloc_gpu("trans_ev", successGPU)
  else ! useGPU
    deallocate(tmat, tmp, stat=istat, errmsg=errorMessage)
    check_deallocate("trans_ev: tmat, tmp", istat, errorMessage)
  endif ! useGPU

  call obj%timer%stop("trans_ev_&
  &MATH_DATATYPE&
  &" // &
  &PRECISION_SUFFIX // &
  gpuString )

end

