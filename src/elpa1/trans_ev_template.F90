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
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL) || defined(WITH_ONEAPI_ONECCL)
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
        (obj, na, nqc, a_mat, lda, tau,     q_mat, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, wantDebug, success)
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
#ifdef WITH_SYCL_GPU_VERSION
  use sycl_functions ! for sycl_getiscpudevice
#endif

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
  ! workaround for SYCL CPU devices: SYRK/HERK produces wrong results
  logical :: is_sycl_cpu
  is_sycl_cpu = .false.
#if defined(WITH_SYCL_GPU_VERSION)
  success = sycl_getiscpudevice(is_sycl_cpu)
#endif

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

  useCCL_int = 0
  useCCL = obj%gpu_setup%useCCL


#if defined(USE_CCL_TRANS_EV)
  if (useGPU) then
    if (useCCL) then
      useCCL_int = 1
    endif
  
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

  call obj%get("max_stored_rows", max_stored_rows_fac, error)
  ! Use an alternative default for GPU run, if not enforced by user
  if (useGPU .and. obj%is_set("max_stored_rows")==0) then
    max_stored_rows_fac = 1024
  endif
  max_stored_rows = max((max_stored_rows_fac/nblk)*nblk, nblk)
  if (wantDebug .and. obj%mpi_setup%myRank_comm_parent==0) print *, "ELPA trans_ev: max_stored_rows=", max_stored_rows

  totalblocks = (na-1)/nblk + 1
  max_blocks_row = (totalblocks-1)/np_rows + 1
  max_blocks_col = ((nqc-1)/nblk)/np_cols + 1  ! Columns of q_mat!

  max_local_rows = max_blocks_row*nblk
  max_local_cols = max_blocks_col*nblk

  if (.not. useGPU) then
    allocate(tmat(max_stored_rows,max_stored_rows), stat=istat, errmsg=errorMessage)
    call check_alloc("trans_ev", "tmat", istat, errorMessage)

    allocate(tmp(max_local_cols*max_stored_rows), stat=istat, errmsg=errorMessage)
    call check_alloc("trans_ev", "tmp", istat, errorMessage)
  endif

  if (.not. useCCL) then
    allocate(h(max_stored_rows*max_stored_rows), stat=istat, errmsg=errorMessage)
    call check_alloc("trans_ev", "h", istat, errorMessage)

    allocate(hvb(max_local_rows*nblk), stat=istat, errmsg=errorMessage)
    call check_alloc("trans_ev", "hvb", istat, errorMessage)

    allocate(hvm(max_local_rows,max_stored_rows), stat=istat, errmsg=errorMessage)
    call check_alloc("trans_ev", "hvm", istat, errorMessage)
  endif

  if (useGPU) then
    ! todo: this is used only for copying hmv to device.. it should be possible to go without it
    !allocate(hvm1(max_local_rows*max_stored_rows), stat=istat, errmsg=errorMessage)
    !call check_alloc("trans_ev_&
    !&MATH_DATATYPE&
    !&", "hvm1", istat, errorMessage)

    if (.not. useCCL) then
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
    endif ! (.not. useCCL) 

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

  if (.not. useCCL) then
    hvm = 0   ! Must be set to 0 !!!
  endif

  if (useGPU) then
    num = max_local_rows*max_stored_rows * size_of_datatype
#ifdef WITH_GPU_STREAMS
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

  if (useGPU) then
    call obj%timer%start("gpu_memset")
    num_el = max_stored_rows*max_stored_rows
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_memset_async(tmat_dev, 0, num_el*size_of_datatype, my_stream)
    if (wantDebug) successGPU = gpu_stream_synchronize(my_stream)
#else
    successGPU = gpu_memset(tmat_dev, 0, num_el*size_of_datatype)
#endif
    check_memcpy_gpu("trans_ev: tmat_dev", successGPU)
    call obj%timer%stop("gpu_memset")
  else
    tmat = 0
  endif

  do istep = 1, na, blockStep
    ics = MAX(istep,3)
    ice = MIN(istep+nblk-1,na)
    if (ice<ics) cycle

    NVTX_RANGE_PUSH("main_loop")
    call obj%timer%start("main_loop_trans_ev")

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

      nb = max_local_rows*nblk ! no compression
    else ! useGPU
      NVTX_RANGE_PUSH("loop: copy hvb <- a_mat")
      nb = 0
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


#ifdef WITH_MPI
    if (useGPU .and. .not. useCCL) then
      num = nb * size_of_datatype
#ifdef WITH_GPU_STREAMS
      successGPU = gpu_memcpy_async(int(loc(hvb(1)),kind=c_intptr_t), hvb_dev, num, gpuMemcpyDeviceToHost, my_stream)
      successGPU = successGPU .and. gpu_stream_synchronize(my_stream)
#else
      successGPU = gpu_memcpy      (int(loc(hvb(1)),kind=c_intptr_t), hvb_dev, num, gpuMemcpyDeviceToHost)
#endif
      check_memcpy_gpu("trans_ev", successGPU)
    endif ! useGPU

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

    if (useGPU .and. .not. useCCL) then
      num_el = nb ! = max_local_rows*nblk, no compression
#ifdef WITH_GPU_STREAMS
      successGPU = gpu_memcpy_async(hvb_dev, int(loc(hvb(1)),kind=c_intptr_t), num_el*size_of_datatype, &
                                    gpuMemcpyHostToDevice, my_stream)
      if (wantDebug) successGPU = successGPU .and. gpu_stream_synchronize(my_stream)
#else
      successGPU = gpu_memcpy      (hvb_dev, int(loc(hvb(1)),kind=c_intptr_t), num_el*size_of_datatype, &
                                    gpuMemcpyHostToDevice)
#endif
      check_memcpy_gpu("trans_ev", successGPU)
    endif ! useGPU
#endif /* WITH_MPI */

    if (useGPU) then
      call obj%timer%start("gpu_copy_hvm_hvb_kernel")
      NVTX_RANGE_PUSH("gpu_copy_hvm_hvb")
      call gpu_copy_hvm_hvb(PRECISION_CHAR, hvm_dev, hvb_dev, tau_dev, &
                            max_local_rows, max_local_rows, my_prow, np_rows, &
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
        
        ! if tau==0, reflector is identity and the column inactive (hvm(1:l_rows,nstor+1) = 0)
        if (tau(ic) /= ZERO) then
          hvm(1:l_rows, nstor+1) = hvb(nb+1:nb+l_rows)
        endif
        
        nstor = nstor+1
        nb = nb+l_rows
      enddo
      NVTX_RANGE_POP("loop: copy hvm <- hvb")
    endif ! useGPU

    ! Please note: for smaller matix sizes (na/np_rows<=256), a value of 32 for nstor is enough!
    if (nstor+nblk > max_stored_rows .or. istep+nblk > na .or. (na/np_rows <= 256 .and. nstor >= 32)) then

      ! Calculate scalar products of stored vectors.
      ! This can be done in different ways, we use dsyrk or zherk
      if (l_rows>0) then
        if (useGPU .and. .not. is_sycl_cpu) then
          call obj%timer%start("gpublas_syrk")
          NVTX_RANGE_PUSH("gpublas_syrk")

          call gpublas_PRECISION_SYRK_HERK('L', BLAS_TRANS_OR_CONJ, &
                                           nstor, l_rows, ONE, &
                                           hvm_dev, max_local_rows, ZERO, &
                                           tmat_dev, max_stored_rows, gpublasHandle)
#ifdef WITH_GPU_STREAMS
          if (wantDebug) successGPU = gpu_stream_synchronize(my_stream)
#else
          if (wantDebug) successGPU = gpu_DeviceSynchronize()
#endif

          NVTX_RANGE_POP("gpublas_syrk")
          call obj%timer%stop("gpublas_syrk")
        else ! useGPU
          if(is_sycl_cpu) then
            num = max_local_rows*max_stored_rows * size_of_datatype
#ifdef WITH_GPU_STREAMS
            successGPU = gpu_memcpy_async(int(loc(hvm(1,1)),kind=c_intptr_t), hvm_dev, num, gpuMemcpyDeviceToHost, my_stream)
            successGPU = successGPU .and. gpu_stream_synchronize(my_stream)
#else
            successGPU = gpu_memcpy      (int(loc(hvm(1,1)),kind=c_intptr_t), hvm_dev, num, gpuMemcpyDeviceToHost)
#endif
            check_memcpy_gpu("trans_ev hvm_dev -> hvm", successGPU)
            
            tmat = 0
          endif ! is_sycl_cpu

          call obj%timer%start("blas_syrk")
          NVTX_RANGE_PUSH("blas_syrk")
#if REALCASE == 1
          call PRECISION_SYRK &
#elif COMPLEXCASE == 1
          call PRECISION_HERK &
#endif
                            ('L', BLAS_TRANS_OR_CONJ, &
                            int(nstor,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), ONE, &
                            hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), ZERO, &
                            tmat, int(max_stored_rows,kind=BLAS_KIND))
          
          NVTX_RANGE_POP("blas_syrk")
          call obj%timer%stop("blas_syrk")

          if(is_sycl_cpu) then
            num_el = max_stored_rows*max_stored_rows
#ifdef WITH_GPU_STREAMS
            successGPU = gpu_memcpy_async(tmat_dev, int(loc(tmat(1,1)),kind=c_intptr_t), num_el*size_of_datatype, &
                                          gpuMemcpyHostToDevice, my_stream)
#else
            successGPU = gpu_memcpy      (tmat_dev, int(loc(tmat(1,1)),kind=c_intptr_t), num_el*size_of_datatype, &
                                          gpuMemcpyHostToDevice)
#endif
            check_memcpy_gpu("trans_ev", successGPU)
          endif
        endif ! useGPU
      endif ! (l_rows>0)

      if (useGPU .and. .not. useCCL) then
        num_el = max_stored_rows*max_stored_rows
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(int(loc(tmat(1,1)),kind=c_intptr_t), tmat_dev, num_el*size_of_datatype, &
                                      gpuMemcpyDeviceToHost, my_stream)                
        successGPU = successGPU .and. gpu_stream_synchronize(my_stream)
#else
        successGPU = gpu_memcpy      (int(loc(tmat(1,1)),kind=c_intptr_t), tmat_dev, num_el*size_of_datatype, &
                                      gpuMemcpyDeviceToHost)
#endif
        check_memcpy_gpu("trans_ev", successGPU)
      endif ! useGPU


      if (useCCL) then
        ! no compression
        nc = max_stored_rows*nstor
        if (nc>0) then
          if (wantDebug) call obj%timer%start("gpu_memcpy")
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memcpy_async(h_dev, tmat_dev, nc*size_of_datatype, gpuMemcpyDeviceToDevice, my_stream)
          if (wantDebug) successGPU = successGPU .and. gpu_stream_synchronize(my_stream)
#else
          successGPU = gpu_memcpy      (h_dev, tmat_dev, nc*size_of_datatype, gpuMemcpyDeviceToDevice)
#endif
          check_memcpy_gpu("elpa_trans_ev: h_dev <- tmat_dev", successGPU)
          if (wantDebug) call obj%timer%stop("gpu_memcpy")
        endif
      else  ! useCCL
        ! compression
        nc = 0
        do n = 1, nstor
          h(nc+1:nc+n) = tmat(n,1:n) ! lower triangular part, takes into account diagonal
          nc = nc+n ! on GPU: nc += max_stored_rows per iteration
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
        successGPU = gpu_memcpy_async(h_dev, int(loc(h(1)),kind=c_intptr_t), num_el*size_of_datatype, &
                                      gpuMemcpyHostToDevice, my_stream)
        if (wantDebug) successGPU = successGPU .and. gpu_stream_synchronize(my_stream)
#else /* WITH_GPU_STREAMS */
        successGPU = gpu_memcpy      (h_dev, int(loc(h(1)),kind=c_intptr_t), num_el*size_of_datatype, &
                                      gpuMemcpyHostToDevice)
#endif /* WITH_GPU_STREAMS */
        check_memcpy_gpu("trans_ev", successGPU)
      endif ! useGPU


      ! Calculate triangular matrix T. 
      ! What was previously stored in (upper part of) tmat is now stored in h, old values of tmat are not needed anymore
      if (useCCL) then
        ! no compression
        nc = max_stored_rows*nstor
        if (nc>0) then
          if (wantDebug) call obj%timer%start("gpu_memcpy")
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memcpy_async(tmat_dev, h_dev, nc*size_of_datatype, gpuMemcpyDeviceToDevice, my_stream)
          if (wantDebug) successGPU = gpu_stream_synchronize(my_stream)
#else
          successGPU = gpu_memcpy      (tmat_dev, h_dev, nc*size_of_datatype, gpuMemcpyDeviceToDevice)
#endif
          check_memcpy_gpu("elpa_trans_ev: tmat_dev <- h_dev", successGPU)
          if (wantDebug) call obj%timer%stop("gpu_memcpy")
        endif
      else ! useCCL
        ! decompression
        nc = 0

        do n = 1, nstor
          tmat(n,1:n) = h(nc+1:nc+n)
          nc = nc+n
        enddo

        if (useGPU) then
          num = max_stored_rows*nstor*size_of_datatype
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memcpy_async(tmat_dev, int(loc(tmat(1,1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
#else
          successGPU = gpu_memcpy      (tmat_dev, int(loc(tmat(1,1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
#endif
        endif
      endif ! useCCL

      if (useGPU) then
        call gpu_set_tmat_diag_from_tau(PRECISION_CHAR, tmat_dev, tau_dev, &
                                        int(max_stored_rows,kind=c_int), int(nstor,kind=c_int), &
                                        int(ice-nstor,kind=c_int), int(SM_count,kind=c_int), &
                                        int(debug,kind=c_int), my_stream)
      else ! useGPU
        do n = 1, nstor
          ic = ice-nstor+n
          if (tau(ic) == ZERO) then
            tmat(n,n) = ONE
          else
#ifdef REALCASE
            tmat(n,n) = tmat(n,n)/2 ! a special trick for real case
#elif COMPLEXCASE
            tmat(n,n) = ONE/tau(ic) ! general for both real and complex
#endif
          endif ! (tau(ic) == ZERO)
        enddo
      endif ! useGPU

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
#ifdef WITH_GPU_STREAMS
          if (wantDebug) successGPU = gpu_stream_synchronize(my_stream)
#else
          if (wantDebug) successGPU = gpu_DeviceSynchronize()
#endif
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
          num = l_cols * nstor * size_of_datatype
          successGPU = gpu_memset_async(tmp_dev, 0, num, my_stream)

          if (wantDebug) successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev", successGPU)
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
      num_el = nstor*l_cols

      if (useGPU .and. .not. useCCL) then
        ! copy tmp_dev -> tmp if needed
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(int(loc(tmp(1)),kind=c_intptr_t), tmp_dev, num_el*size_of_datatype, &
                                      gpuMemcpyDeviceToHost, my_stream)
        successGPU = successGPU .and. gpu_stream_synchronize(my_stream)
#else
        successGPU = gpu_memcpy      (int(loc(tmp(1)),kind=c_intptr_t), tmp_dev, num_el*size_of_datatype, &
                                      gpuMemcpyDeviceToHost)
#endif
        check_memcpy_gpu("trans_ev", successGPU)
      endif ! (useGPU .and. .not. useCCL)


      if (useCCL) then
        call obj%timer%start("ccl_allreduce")
        NVTX_RANGE_PUSH("ccl_allreduce")
        successGPU = ccl_Allreduce(tmp_dev, tmp_dev, int(k_datatype*num_el,kind=c_size_t), &
                                     cclDataType, cclSum, ccl_comm_rows, my_stream)

        if (.not. successGPU) then
          print *,"Error in ccl_allreduce"
          stop 1
        endif

        if (wantDebug) then
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev", successGPU)
        endif
        NVTX_RANGE_POP("ccl_allreduce")
        call obj%timer%stop("ccl_allreduce")
      else ! useCCL
        if (useNonBlockingCollectivesRows) then
          call obj%timer%start("mpi_nbc_communication")
          call mpi_iallreduce(MPI_IN_PLACE, tmp, int(num_el,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                             int(mpi_comm_rows,kind=MPI_KIND), allreduce_request2, mpierr)
          call mpi_wait(allreduce_request2, MPI_STATUS_IGNORE, mpierr)
          call obj%timer%stop("mpi_nbc_communication")
        else ! useNonBlockingCollectivesRows
          call obj%timer%start("mpi_communication")
          call mpi_allreduce(MPI_IN_PLACE, tmp, int(num_el,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                             int(mpi_comm_rows,kind=MPI_KIND), mpierr)
          call obj%timer%stop("mpi_communication")
        endif ! useNonBlockingCollectivesRows
      endif ! useCCL

      ! copy back to tmp -> tmp_dev if needed
      if (useGPU .and. .not. useCCL) then
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(tmp_dev, int(loc(tmp(1)),kind=c_intptr_t), num_el*size_of_datatype, &
                                      gpuMemcpyHostToDevice, my_stream)
        if (wantDebug) successGPU = successGPU .and. gpu_stream_synchronize(my_stream)
#else /* WITH_GPU_STREAMS */
        successGPU = gpu_memcpy      (tmp_dev, int(loc(tmp(1)),kind=c_intptr_t), num_el*size_of_datatype, &
                                      gpuMemcpyHostToDevice)
#endif /* WITH_GPU_STREAMS */
        check_memcpy_gpu("trans_ev", successGPU)
      endif ! useGPU

#endif /* WITH_MPI */

      if (l_rows > 0) then
        if (useGPU) then
          ! tmp_dev = tmat_dev*tmp_dev
          call obj%timer%start("gpublas_trsm")
          NVTX_RANGE_PUSH("gpublas_trsm")

          call gpublas_PRECISION_TRSM('L', 'L', 'N', 'N', &
                                      nstor, l_cols, ONE, &
                                      tmat_dev, max_stored_rows, &
                                      tmp_dev, nstor, gpublasHandle)
#ifdef WITH_GPU_STREAMS
          if (wantDebug) successGPU = gpu_stream_synchronize(my_stream)
#else
          if (wantDebug) successGPU = gpu_DeviceSynchronize()
#endif
          call obj%timer%stop("gpublas_trsm")
          NVTX_RANGE_POP("gpublas_trsm")

          NVTX_RANGE_PUSH("gpublas_gemm")
          call obj%timer%start("gpublas_gemm")
          call gpublas_PRECISION_GEMM('N', 'N', &
                                      l_rows, l_cols, nstor, -ONE, &
                                      hvm_dev, max_local_rows, &
                                      tmp_dev, nstor, ONE, &
                                      q_dev, ldq, gpublasHandle)
#ifdef WITH_GPU_STREAMS
          if (wantDebug) successGPU = gpu_stream_synchronize(my_stream)
#else
          if (wantDebug) successGPU = gpu_DeviceSynchronize()
#endif
          call obj%timer%stop("gpublas_gemm")
          NVTX_RANGE_POP("gpublas_gemm")
        else !useGPU
          call obj%timer%start("blas")

          ! tmp = tmat^{-1} * tmp
          call PRECISION_TRSM('L', 'L', 'N', 'N', &
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

    NVTX_RANGE_POP("main_loop")
    call obj%timer%stop("main_loop_trans_ev")
  enddo ! istep = 1, na, blockStep

  if (.not. useCCL) then
    deallocate(h, hvb, hvm, stat=istat, errmsg=errorMessage)
    check_deallocate("trans_ev: h, hvb, hvm", istat, errorMessage)
  endif

  if (useGPU) then
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    !if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
    !  successGPU = gpu_host_unregister(int(loc(q_mat),kind=c_intptr_t))
    !  check_host_unregister_gpu("trans_ev: q_mat", successGPU)
    !endif

    if (.not. useCCL) then
      if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
        successGPU = gpu_free_host(hvm1_host)
        check_host_dealloc_gpu("trans_ev: hvm1_host", successGPU)
        nullify(hvm1)

        successGPU = gpu_free_host(tmat_host)
        check_host_dealloc_gpu("trans_ev: tmat_host", successGPU)
        nullify(tmat)
      else
        deallocate(hvm1)
        deallocate(tmat)
      endif

      if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
        successGPU = gpu_free_host(tmp_host)
        check_host_dealloc_gpu("trans_ev: tmp_host", successGPU)
        nullify(tmp)
      else
        deallocate(tmp)
      endif
    endif ! (.not. useCCL)
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
