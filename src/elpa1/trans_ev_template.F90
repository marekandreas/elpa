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
!> \param q_mat           On input: Eigenvectors of tridiagonal matrix
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
        (obj, na, nqc, a_dev, lda, tau_dev, q_dev, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, success)
#else
subroutine trans_ev_cpu_&
        &MATH_DATATYPE&
        &_&
        &PRECISION &
        (obj, na, nqc, a_mat, lda, tau, q_mat, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, success)
#endif

  use, intrinsic :: iso_c_binding
  use precision
  use elpa_abstract_impl
  use elpa_blas_interfaces
  use elpa_gpu
  use elpa_gpu_util
  use trans_ev_gpu
#ifdef WITH_NVIDIA_GPU_VERSION
  use cuda_functions
#endif
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
  use elpa_ccl_gpu
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

  logical                                       :: useGPU
  integer(kind=ik)                              :: max_stored_rows, max_stored_rows_fac

  integer(kind=ik)                              :: my_prow, my_pcol, np_rows, np_cols
  integer(kind=MPI_KIND)                        :: mpierr, my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI
  integer(kind=ik)                              :: totalblocks, max_blocks_row, max_blocks_col, max_local_rows, max_local_cols
  integer(kind=ik)                              :: l_cols, l_rows, l_colh, nstor
  integer(kind=ik)                              :: istep, n, nc, ic, ics, ice, nb, cur_pcol
  integer(kind=ik)                              :: hvn_ubnd, hvm_ubnd

  MATH_DATATYPE(kind=rck), allocatable          :: hvb(:), hvm(:,:)
  MATH_DATATYPE(kind=rck), pointer              :: tmp1(:), tmp2(:)
  MATH_DATATYPE(kind=rck), allocatable          :: h1(:), h2(:), tmp_debug(:)
  MATH_DATATYPE(kind=rck), pointer              :: tmat(:,:)
  MATH_DATATYPE(kind=rck), pointer              :: hvm1(:)
  type(c_ptr)                                   :: tmp1_host, tmp2_host
  type(c_ptr)                                   :: hvm1_host, tmat_host

  integer(kind=ik)                              :: istat
  character(200)                                :: errorMessage
  character(20)                                 :: gpuString

  integer(kind=c_intptr_t)                      :: num
  integer(kind=C_intptr_T)                      :: tmp_dev, hvm_dev, tmat_dev
#ifdef WITH_CUDA_AWARE_MPI
  type(c_ptr)                                   :: tmp_mpi_dev
  MATH_DATATYPE(kind=rck), pointer              :: tmp_mpi(:)
#endif

  integer(kind=ik)                              :: blockStep
  logical                                       :: successGPU
  integer(kind=c_intptr_t), parameter           :: size_of_datatype = size_of_&
                                                                      &PRECISION&
                                                                      &_&
                                                                      &MATH_DATATYPE
  integer(kind=ik)                              :: error
  integer(kind=MPI_KIND)                        :: bcast_request1, allreduce_request1, allreduce_request2
  logical                                       :: useNonBlockingCollectivesCols
  logical                                       :: useNonBlockingCollectivesRows
  integer(kind=c_int)                           :: non_blocking_collectives_rows, non_blocking_collectives_cols
  logical                                       :: success
  integer(kind=c_intptr_t)                      :: gpuHandle, my_stream
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
  integer(kind=c_intptr_t)                      :: ccl_comm_rows, ccl_comm_cols
#endif

 useGPU = .false.
#ifdef TRANS_EV_GPU
 useGPU = .true.
#endif


  success = .true.

  if(useGPU) then
    gpuString = "_gpu"
  else
    gpuString = ""
  endif

  call obj%timer%start("trans_ev_&
  &MATH_DATATYPE&
  &" // &
  &PRECISION_SUFFIX //&
  gpuString)

  if (useGPU) then

    num = lda * matrixCols * size_of_datatype
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    num = lda * matrixCols * size_of_datatype
    call gpu_memcpy_async_and_stream_synchronize &
         ("trans_ev a_dev -> a_mat", a_dev, 0_c_intptr_t, &
                            a_mat(1:lda,1:matrixCols), &
                            1, 1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
    successGPU = gpu_memcpy(int(loc(a_mat(1,1)),kind=c_intptr_t), &
                  a_dev, num, gpuMemcpyDeviceToHost)
    check_memcpy_gpu("trans_ev", successGPU)
#endif

    num = na * size_of_datatype
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    call gpu_memcpy_async_and_stream_synchronize &
         ("trans_ev tau_dev -> tau", tau_dev, 0_c_intptr_t, &
                            tau(1:na), &
                            1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
    successGPU = gpu_memcpy(int(loc(tau(1)),kind=c_intptr_t), &
                  tau_dev, num, gpuMemcpyDeviceToHost)
    check_memcpy_gpu("trans_ev", successGPU)
#endif



  endif


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


  !call obj%timer%start("mpi_communication")
  !call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND) ,my_prowMPI, mpierr)
  !call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND) ,np_rowsMPI, mpierr)
  !call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND) ,my_pcolMPI, mpierr)
  !call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND) ,np_colsMPI, mpierr)

  !my_prow = int(my_prowMPI, kind=c_int)
  !np_rows = int(np_rowsMPI, kind=c_int)
  !my_pcol = int(my_pcolMPI, kind=c_int)
  !np_cols = int(np_colsMPI, kind=c_int)
  !call obj%timer%stop("mpi_communication")

  call obj%get("max_stored_rows",max_stored_rows_fac, error)

  totalblocks = (na-1)/nblk + 1
  max_blocks_row = (totalblocks-1)/np_rows + 1
  max_blocks_col = ((nqc-1)/nblk)/np_cols + 1  ! Columns of q_mat!

  max_local_rows = max_blocks_row*nblk
  max_local_cols = max_blocks_col*nblk

  max_stored_rows = (max_stored_rows_fac/nblk+1)*nblk
 
  if (.not.(useGPU)) then
    allocate(tmat(max_stored_rows,max_stored_rows), stat=istat, errmsg=errorMessage)
    call check_alloc("trans_ev_&
    &MATH_DATATYPE&
    &", "tmat", istat, errorMessage)

    allocate(tmp1(max_local_cols*max_stored_rows), stat=istat, errmsg=errorMessage)
    call check_alloc("trans_ev_&
    &MATH_DATATYPE&
    &", "tmp1", istat, errorMessage)

    allocate(tmp2(max_local_cols*max_stored_rows), stat=istat, errmsg=errorMessage)
    call check_alloc("trans_ev_&
    &MATH_DATATYPE&
    &", "tmp2", istat, errorMessage)
  endif

  allocate(h1(max_stored_rows*max_stored_rows), stat=istat, errmsg=errorMessage)
  call check_alloc("trans_ev_&
  &MATH_DATATYPE&
  &", "h1", istat, errorMessage)

  allocate(h2(max_stored_rows*max_stored_rows), stat=istat, errmsg=errorMessage)
  call check_alloc("trans_ev_&
  &MATH_DATATYPE&
  &", "h2", istat, errorMessage)

  allocate(hvb(max_local_rows*nblk), stat=istat, errmsg=errorMessage)
  call check_alloc("trans_ev_&
  &MATH_DATATYPE&
  &", "hvn", istat, errorMessage)

  allocate(hvm(max_local_rows,max_stored_rows), stat=istat, errmsg=errorMessage)
  call check_alloc("trans_ev_&
  &MATH_DATATYPE&
  &", "hvm", istat, errorMessage)

  hvm = 0   ! Must be set to 0 !!!
  hvb = 0   ! Safety only
  blockStep = nblk

  l_cols = local_index(nqc, my_pcol, np_cols, nblk, -1) ! Local columns of q_mat

  nstor = 0
  if (useGPU) then
    hvn_ubnd = 0
  endif

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
    ! todo: this is used only for copying hmv to device.. it should be possible to go without it
    !allocate(hvm1(max_local_rows*max_stored_rows), stat=istat, errmsg=errorMessage)
    !call check_alloc("trans_ev_&
    !&MATH_DATATYPE&
    !&", "hvm1", istat, errorMessage)
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
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
      successGPU = gpu_malloc_host(tmp1_host,num)
      check_alloc_gpu("trans_ev: tmp1_host", successGPU)
      call c_f_pointer(tmp1_host,tmp1,(/(max_local_cols*max_stored_rows)/))
    else
      allocate(tmp1(max_local_cols*max_stored_rows))
    endif

    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
      num = (max_local_cols*max_stored_rows) * size_of_datatype
      successGPU = gpu_malloc_host(tmp2_host,num)
      check_alloc_gpu("trans_ev: tmp2_host", successGPU)
      call c_f_pointer(tmp2_host,tmp2,(/(max_local_cols*max_stored_rows)/))
    else
      allocate(tmp2(max_local_cols*max_stored_rows))
    endif
#endif
    successGPU = gpu_malloc(tmat_dev, max_stored_rows * max_stored_rows * size_of_datatype)
    check_alloc_gpu("trans_ev", successGPU)

    successGPU = gpu_malloc(hvm_dev, max_local_rows * max_stored_rows * size_of_datatype)
    check_alloc_gpu("trans_ev", successGPU)

    successGPU = gpu_malloc(tmp_dev, max_local_cols * max_stored_rows * size_of_datatype)
    check_alloc_gpu("trans_ev", successGPU)

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)

    !if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
    !  successGPU = gpu_host_register(int(loc(q_mat),kind=c_intptr_t),num,&
    !              gpuHostRegisterDefault)
    !  check_host_register_gpu("trans_ev: q_mat", successGPU)
    !endif
#endif

  endif  ! useGPU

  do istep = 1, na, blockStep

#ifdef WITH_NVTX
    call nvtxRangePush("trans_ev_cycle")
#endif

    ics = MAX(istep,3)
    ice = MIN(istep+nblk-1,na)
    if (ice<ics) cycle

    cur_pcol = pcol(istep, nblk, np_cols)

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

#ifdef WITH_MPI
    if (nb > 0) then
      if (useNonBlockingCollectivesCols) then
        call obj%timer%start("mpi_nbc_communication")
        call mpi_ibcast(hvb, int(nb,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION , int(cur_pcol,kind=MPI_KIND), &
                   int(mpi_comm_cols,kind=MPI_KIND), bcast_request1, mpierr)
        call mpi_wait(bcast_request1, MPI_STATUS_IGNORE, mpierr)
        call obj%timer%stop("mpi_nbc_communication")
       else
        call obj%timer%start("mpi_communication")
        call mpi_bcast(hvb, int(nb,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION , int(cur_pcol,kind=MPI_KIND), &
                   int(mpi_comm_cols,kind=MPI_KIND), mpierr)
        call obj%timer%stop("mpi_communication")
      endif
    endif
#endif /* WITH_MPI */

    nb = 0
    do ic = ics, ice
      l_rows = local_index(ic-1, my_prow, np_rows, nblk, -1) ! # rows of Householder Vector
      hvm(1:l_rows,nstor+1) = hvb(nb+1:nb+l_rows)
      if (useGPU) then
        hvm_ubnd = l_rows
      endif
      nstor = nstor+1
      nb = nb+l_rows
    enddo

    ! Please note: for smaller matix sizes (na/np_rows<=256), a value of 32 for nstor is enough!
    if (nstor+nblk > max_stored_rows .or. istep+nblk > na .or. (na/np_rows <= 256 .and. nstor >= 32)) then

      ! Calculate scalar products of stored vectors.
      ! This can be done in different ways, we use dsyrk or zherk

      tmat = 0
      call obj%timer%start("blas")
      if (l_rows>0) then
#if REALCASE == 1
        call PRECISION_SYRK('U', 'T',   &
#endif
#if COMPLEXCASE == 1
        call PRECISION_HERK('U', 'C',   &
#endif
                         int(nstor,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), ONE, &
                         hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), ZERO, tmat, int(max_stored_rows,kind=BLAS_KIND))
      endif
      call obj%timer%stop("blas")
      nc = 0
      do n = 1, nstor-1
        h1(nc+1:nc+n) = tmat(1:n,n+1)
        nc = nc+n
      enddo
#ifdef WITH_MPI
      if (nc > 0) then
        if (useNonBlockingCollectivesRows) then
          call obj%timer%start("mpi_nbc_communication")
          call mpi_iallreduce( h1, h2, int(nc,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                                   int(mpi_comm_rows,kind=MPI_KIND), allreduce_request1, mpierr)
          call mpi_wait(allreduce_request1, MPI_STATUS_IGNORE, mpierr)
          call obj%timer%stop("mpi_nbc_communication")
        else
          call obj%timer%start("mpi_communication")
          call mpi_allreduce( h1, h2, int(nc,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                                   int(mpi_comm_rows,kind=MPI_KIND), mpierr)
          call obj%timer%stop("mpi_communication")
        endif
      endif
#else /* WITH_MPI */

      if (nc > 0) h2 = h1

#endif /* WITH_MPI */
      ! Calculate triangular matrix T

      nc = 0
      tmat(1,1) = tau(ice-nstor+1)
      do n = 1, nstor-1
        call obj%timer%start("blas")
        call PRECISION_TRMV('L', BLAS_TRANS_OR_CONJ , 'N', int(n,kind=BLAS_KIND), tmat, &
                            int(max_stored_rows,kind=BLAS_KIND), h2(nc+1), 1_BLAS_KIND)
        call obj%timer%stop("blas")

        tmat(n+1,1:n) = &
#if REALCASE == 1
        -h2(nc+1:nc+n)  &
#endif
#if COMPLEXCASE == 1
        -conjg(h2(nc+1:nc+n)) &
#endif
        *tau(ice-nstor+n+1)

        tmat(n+1,n+1) = tau(ice-nstor+n+1)
        nc = nc+n
      enddo

      if (useGPU) then
        ! todo: is this reshape really neccessary?
        hvm1(1:hvm_ubnd*nstor) = reshape(hvm(1:hvm_ubnd,1:nstor), (/ hvm_ubnd*nstor /))

        !hvm_dev(1:hvm_ubnd*nstor) = hvm1(1:hvm_ubnd*nstor)
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        num = hvm_ubnd * nstor * size_of_datatype
        call gpu_memcpy_async_and_stream_synchronize &
            ("trans_ev hvm1 -> hvm_dev", hvm_dev, 0_c_intptr_t, &
                                                 hvm1(1:max_local_rows*max_stored_rows), &
                                                 1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)


        my_stream = obj%gpu_setup%my_stream
        num = max_stored_rows * max_stored_rows * size_of_datatype
        call gpu_memcpy_async_and_stream_synchronize &
            ("trans_ev tmat -> tmat_dev", tmat_dev, 0_c_intptr_t, &
                                                 tmat(1:max_local_rows,1:max_stored_rows), &
                                                 1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
        successGPU = gpu_memcpy(hvm_dev, int(loc(hvm1(1)),kind=c_intptr_t),   &
                      hvm_ubnd * nstor * size_of_datatype, gpuMemcpyHostToDevice)

        check_memcpy_gpu("trans_ev", successGPU)

        !tmat_dev = tmat
        successGPU = gpu_memcpy(tmat_dev, int(loc(tmat(1,1)),kind=c_intptr_t),   &
                      max_stored_rows * max_stored_rows * size_of_datatype, gpuMemcpyHostToDevice)
        check_memcpy_gpu("trans_ev", successGPU)
#endif
      endif


      ! Q = Q - V * T * V**T * Q

      if (l_rows > 0) then
        if (useGPU) then
          call obj%timer%start("gpublas")
          gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
          call gpublas_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',   &
                                   nstor, l_cols, l_rows, ONE, hvm_dev, hvm_ubnd,  &
                                   q_dev, ldq, ZERO, tmp_dev, nstor, gpuHandle)
          call obj%timer%stop("gpublas")
        else ! useGPU

          call obj%timer%start("blas")
          call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',  &
                              int(nstor,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), &
                              int(l_rows,kind=BLAS_KIND), ONE, hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), &
                              q_mat, int(ldq,kind=BLAS_KIND), ZERO, tmp1, int(nstor,kind=BLAS_KIND))
          call obj%timer%stop("blas")
        endif ! useGPU

      else !l_rows>0

        if (useGPU) then
          if (gpu_vendor() /= OPENMP_OFFLOAD_GPU) then
#ifdef WITH_GPU_STREAMS
            my_stream = obj%gpu_setup%my_stream
            num = l_cols * nstor * size_of_datatype
            successGPU = gpu_memset_async(tmp_dev, 0, num, my_stream)
            check_memcpy_gpu("trans_ev: tmp_dev", successGPU)
            !successGPU = gpu_stream_synchronize(my_stream)
            !check_stream_synchronize_gpu("trans_ev", successGPU)
#else
            successGPU = gpu_memset(tmp_dev, 0, l_cols * nstor * size_of_datatype)
#endif
            check_memcpy_gpu("trans_ev", successGPU)
          else
            allocate(tmp_debug(l_cols * nstor))
            tmp_debug(:) = 0.
            successGPU = gpu_memcpy(tmp_dev, int(loc(tmp_debug),kind=c_intptr_t), &
                                    l_cols*nstor*size_of_datatype, gpuMemcpyHostToDevice)
            check_memcpy_gpu("trans_ev", successGPU)
            deallocate(tmp_debug)
          endif
        else
          tmp1(1:l_cols*nstor) = 0
        endif
      endif  !l_rows>0

#ifdef WITH_MPI

      if (useGPU) then
#ifndef WITH_CUDA_AWARE_MPI
        ! In the legacy GPU version, this allreduce was ommited. But probably it has to be done for GPU + MPI
        ! todo: does it need to be copied whole? Wouldn't be a part sufficient?
#ifdef WITH_GPU_STREAMS
#ifdef USE_CCL_TRANS_EV
       ! no memory transfers needed
#else /* USE_CCL_TRANS_EV */
        my_stream = obj%gpu_setup%my_stream
        num = max_local_cols * max_stored_rows * size_of_datatype
        call gpu_memcpy_async_and_stream_synchronize &
            ("trans_ev tmp_dev -> tmp1", tmp_dev, 0_c_intptr_t, &
                                                 tmp1(1:max_local_cols*max_stored_rows), &
                                                 1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#endif /* USE_CCL_TRANS_EV */
#else /* WITH_GPU_STREAMS */
        successGPU = gpu_memcpy(int(loc(tmp1(1)),kind=c_intptr_t), tmp_dev,  &
                      max_local_cols * max_stored_rows * size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("trans_ev", successGPU)
#endif /* WITH_GPU_STREAMS */

#else /* WITH_CUDA_AWARE_MPI */
        ! in case of CUDA_AWARE MPI
        ! associate devicePointer with a fortran pointer
        ! do MPI
        tmp_mpi_dev = transfer(tmp_dev, tmp_mpi_dev)
        call c_f_pointer(tmp_mpi_dev,tmp_mpi,(/(max_local_cols*max_stored_rows)/))

#endif /* WITH_CUDA_AWARE_MPI */
      endif


      if (useNonBlockingCollectivesRows) then
        call obj%timer%start("mpi_nbc_communication")
#ifndef WITH_CUDA_AWARE_MPI

#ifdef USE_CCL_TRANS_EV
        if (useGPU) then
          ccl_comm_rows = obj%gpu_setup%ccl_comm_rows
          successGPU = ccl_group_start()
          if (.not.successGPU) then
            print *,"Error in setting up nccl_group_start!"
            stop
          endif
          successGPU = ccl_Allreduce(tmp_dev, tmp_dev, &
#if REALCASE == 1
                                      int(nstor*l_cols,kind=c_size_t), &
#endif
#if COMPLEXCASE == 1
                                      int(2*nstor*l_cols,kind=c_size_t), &
#endif
#if REALCASE == 1
#if DOUBLE_PRECISION == 1
                                   cclDouble, &
#endif
#if SINGLE_PRECISION == 1
                                   cclFloat, &
#endif
#endif /* REALCASE */
#if COMPLEXCASE == 1
#if DOUBLE_PRECISION == 1
                                   cclDouble, &
#endif
#if SINGLE_PRECISION == 1
                                   cclFloat, &
#endif
#endif /* COMPLEXCASE */
                                   cclSum, ccl_comm_rows, my_stream)

          if (.not.successGPU) then
            print *,"Error in nccl_allreduce"
            stop
          endif
          successGPU = ccl_group_end()
          if (.not.successGPU) then
            print *,"Error in setting up nccl_group_end!"
            stop
          endif
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev", successGPU)
        else ! use GPU
          call mpi_iallreduce(tmp1, tmp2, int(nstor*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                         int(mpi_comm_rows,kind=MPI_KIND), allreduce_request2, mpierr)
        endif ! useGPU
#else /* USE_CCL_TRANS_EV */
        call mpi_iallreduce(tmp1, tmp2, int(nstor*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                         int(mpi_comm_rows,kind=MPI_KIND), allreduce_request2, mpierr)
#endif /* USE_CCL_TRANS_EV */

#else /* WITH_CUDA_AWARE_MPI */
        call mpi_iallreduce(mpi_in_place, tmp_mpi, int(nstor*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                         int(mpi_comm_rows,kind=MPI_KIND), allreduce_request2, mpierr)
#endif /* WITH_CUDA_AWARE_MPI */
        call mpi_wait(allreduce_request2, MPI_STATUS_IGNORE, mpierr)
        call obj%timer%stop("mpi_nbc_communication")
      else ! useNonBlockingCollectivesRows
        call obj%timer%start("mpi_communication")
#ifndef WITH_CUDA_AWARE_MPI

#ifdef USE_CCL_TRANS_EV
        if (useGPU) then
          ccl_comm_rows = obj%gpu_setup%ccl_comm_rows
          success = ccl_group_start()
          if (.not.success) then
            print *,"Error in setting up nccl_group_start!"
            stop
          endif

          success = ccl_Allreduce(tmp_dev, tmp_dev, &
#if REALCASE == 1
                                   int(nstor*l_cols,kind=c_size_t), &
#endif
#if COMPLEXCASE == 1
                                   int(2*nstor*l_cols,kind=c_size_t), &
#endif
#if REALCASE == 1
#if DOUBLE_PRECISION == 1
                                   cclDouble, &
#endif
#if SINGLE_PRECISION == 1
                                   cclFloat, &
#endif
#endif /* REALCASE */
#if COMPLEXCASE == 1
#if DOUBLE_PRECISION == 1
                                   cclDouble, &
#endif
#if SINGLE_PRECISION == 1
                                   cclFloat, &
#endif
#endif /* COMPLEXCASE */
                                   cclSum, ccl_comm_rows, my_stream)

          if (.not.success) then
            print *,"Error in nccl_allreduce"
            stop
          endif
          success = ccl_group_end()
          if (.not.success) then
            print *,"Error in setting up nccl_group_end!"
            stop
          endif
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev", successGPU)
        else ! useGPU
          call mpi_allreduce(tmp1, tmp2, int(nstor*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                         int(mpi_comm_rows,kind=MPI_KIND), mpierr)
        endif ! useGPU
#else /* USE_CCL_TRANS_EV */
        call mpi_allreduce(tmp1, tmp2, int(nstor*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                         int(mpi_comm_rows,kind=MPI_KIND), mpierr)
#endif /* USE_CCL_TRANS_EV */

#else /* WITH_CUDA_AWARE_MPI */
        call mpi_allreduce(mpi_in_place, tmp_mpi, int(nstor*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                         int(mpi_comm_rows,kind=MPI_KIND), mpierr)
#endif /* WITH_CUDA_AWARE_MPI */
        call obj%timer%stop("mpi_communication")
      endif ! useNonBlockingCollectivesRows

      if (useGPU) then
#ifndef WITH_CUDA_AWARE_MPI
        ! copy back tmp2 - after reduction...


#ifdef USE_CCL_TRANS_EV
        ! no memory copy needed
#else /* USE_CCL_TRANS_EV */

#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          num = max_local_cols * max_stored_rows * size_of_datatype
          call gpu_memcpy_async_and_stream_synchronize &
            ("trans_ev tmp1 -> tmp_dev", tmp_dev, 0_c_intptr_t, &
                                                 tmp2(1:max_local_cols*max_stored_rows), &
                                                 1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else /* WITH_GPU_STREAMS */
        successGPU = gpu_memcpy(tmp_dev, int(loc(tmp2(1)),kind=c_intptr_t),  &
                      max_local_cols * max_stored_rows * size_of_datatype, gpuMemcpyHostToDevice)
        check_memcpy_gpu("trans_ev", successGPU)
#endif /* WITH_GPU_STREAMS */
#endif /* USE_CCL_TRANS_EV */

#else /* WITH_CUDA_AWARE_MPI */
        
        tmp_dev = transfer(tmp_mpi_dev, tmp_dev)
        tmp_mpi_dev = C_NULL_PTR
        tmp_mpi => null()
#endif /* WITH_CUDA_AWARE_MPI */
      endif ! useGPU

#else /* WITH_MPI */
!     tmp2 = tmp1
#endif /* WITH_MPI */

      if (l_rows > 0) then
        if (useGPU) then
          call obj%timer%start("gpublas")
          gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
          call gpublas_PRECISION_TRMM('L', 'L', 'N', 'N',     &
                                   nstor, l_cols, ONE, tmat_dev, max_stored_rows,  &
                                   tmp_dev, nstor, gpuHandle)

          call gpublas_PRECISION_GEMM('N', 'N' ,l_rows ,l_cols ,nstor,  &
                                   -ONE, hvm_dev, hvm_ubnd, tmp_dev, nstor,   &
                                   ONE, q_dev, ldq, gpuHandle)
          call obj%timer%stop("gpublas")
        else !useGPU
#ifdef WITH_MPI
          ! tmp2 = tmat * tmp2
          call obj%timer%start("blas")
          call PRECISION_TRMM('L', 'L', 'N', 'N', int(nstor,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND),   &
                             ONE, tmat, int(max_stored_rows,kind=BLAS_KIND), tmp2, int(nstor,kind=BLAS_KIND))
          !q_mat = q_mat - hvm*tmp2
          call PRECISION_GEMM('N', 'N', int(l_rows,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), int(nstor,kind=BLAS_KIND),   &
                              -ONE, hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), tmp2, int(nstor,kind=BLAS_KIND), &
                              ONE, q_mat, int(ldq,kind=BLAS_KIND))
          call obj%timer%stop("blas")
#else /* WITH_MPI */
          call obj%timer%start("blas")

          call PRECISION_TRMM('L', 'L', 'N', 'N', int(nstor,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND),   &
                              ONE, tmat, int(max_stored_rows,kind=BLAS_KIND), tmp1, int(nstor,kind=BLAS_KIND))
          call PRECISION_GEMM('N', 'N', int(l_rows,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), &
                              int(nstor,kind=BLAS_KIND), -ONE, hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), &
                              tmp1, int(nstor,kind=BLAS_KIND), ONE, q_mat, int(ldq,kind=BLAS_KIND))
          call obj%timer%stop("blas")
#endif /* WITH_MPI */
        endif ! useGPU
      endif  ! l_rows>0
      nstor = 0
    endif  ! (nstor+nblk>max_stored_rows .or. istep+nblk>na .or. (na/np_rows<=256 .and. nstor>=32))

#ifdef WITH_NVTX    
    call nvtxRangePop()
#endif
    
  enddo ! istep = 1, na, blockStep

  deallocate(h1, h2, hvb, hvm, stat=istat, errmsg=errorMessage)
  check_deallocate("trans_ev_&
    &MATH_DATATYPE&
    &: h1, h2, hvb, hvm", istat, errorMessage)

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

      successGPU = gpu_free_host(tmp1_host)
      check_host_dealloc_gpu("trans_ev: tmp1_host", successGPU)
      nullify(tmp1)

      successGPU = gpu_free_host(tmp2_host)
      check_host_dealloc_gpu("trans_ev: tmp2_host", successGPU)
      nullify(tmp2)
    else
      deallocate(hvm1)
      deallocate(tmat)
      deallocate(tmp1)
      deallocate(tmp2)
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
  else ! useGPU
    deallocate(tmat, tmp1, tmp2, stat=istat, errmsg=errorMessage)
    check_deallocate("trans_ev_&
    &MATH_DATATYPE&
    &: tmat, tmp1, tmp2", istat, errorMessage)
  endif ! useGPU

  call obj%timer%stop("trans_ev_&
  &MATH_DATATYPE&
  &" // &
  &PRECISION_SUFFIX // &
  gpuString )

end

