#if 0
!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), fomerly known as
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



! ELPA2 -- 2-stage solver for ELPA
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
#endif


! - works with mimic loop
! - is it the sharing of device pointers?


subroutine bandred_&
&MATH_DATATYPE&
&_&
&PRECISION &
(obj, na, a_mat, matrixRows, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols, tmat, &
wantDebug, useGPU, success, &
#if REALCASE == 1
useQR, &
#endif
max_threads, isSkewsymmetric)

!-------------------------------------------------------------------------------
!  bandred_real/complex: Reduces a distributed symmetric matrix to band form
!
!  Parameters
!
!  na          Order of matrix
!
!  a_mat(matrixRows,matrixCols)    Distributed matrix which should be reduced.
!              Distribution is like in Scalapack.
!              Opposed to Scalapack, a_mat(:,:) must be set completely (upper and lower half)
!              a_mat(:,:) is overwritten on exit with the band and the Householder vectors
!              in the upper half.
!
!  matrixRows         Leading dimension of a_mat
!  matrixCols  local columns of matrix a_mat
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  nbw         semi bandwith of output matrix
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!
!  tmat(nbw,nbw,numBlocks)    where numBlocks = (na-1)/nbw + 1
!              Factors for the Householder vectors (returned), needed for back transformation
!
!-------------------------------------------------------------------------------

  use elpa_gpu
  use, intrinsic :: iso_c_binding
  use elpa1_compute
#ifdef WITH_OPENMP_TRADITIONAL
  use omp_lib
#endif
  use precision
  use elpa_blas_interfaces
#ifdef WITH_MPI
  use elpa_scalapack_interfaces
#endif
  use elpa_abstract_impl
  !use cuda_functions
  !use hip_functions
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
  use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
  use sycl_functions
#endif

  implicit none
#include "../general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout)  :: obj
  integer(kind=ik)                            :: na, matrixRows, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols

#ifdef USE_ASSUMED_SIZE
  MATH_DATATYPE(kind=rck)                     :: a_mat(matrixRows,*)
  MATH_DATATYPE(kind=rck)                     :: tmat(nbw,nbw,*)
#else
  MATH_DATATYPE(kind=rck)                     :: a_mat(matrixRows,matrixCols)
  MATH_DATATYPE(kind=rck)                     :: tmat(nbw,nbw,numBlocks)
#endif

#if REALCASE == 1
  real(kind=rk)                               :: eps
#endif
  logical, intent(in)                         :: useGPU
  logical, intent(in)                         :: isSkewsymmetric
  character(20)                               :: gpuString

  integer(kind=ik)                            :: my_prow, my_pcol, np_rows, np_cols
  integer(kind=MPI_KIND)                      :: mpierr,  my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI
  integer(kind=ik)                            :: l_cols, l_rows, max_l_rows, max_l_cols
  integer(kind=ik),allocatable                :: blockinfo(:,:)
#if REALCASE == 1
  integer(kind=ik)                            :: vmrCols
#endif
#ifdef WITH_OPENMP_TRADITIONAL
  integer(kind=ik)                            :: lrs, transformChunkSize
#endif
  integer(kind=ik)                            :: i, j, lcs, lce, lre, lc, lr, cur_pcol, n_cols, nrow
  integer(kind=ik)                            :: istep, ncol, lch, lcx, iblock, nblocks, c_start, &
                                                blc_start, blc_end, blc_len
  integer(kind=ik)                            :: tile_size, l_rows_tile, l_cols_tile

  MATH_DATATYPE(kind=rck)                    :: vrl, tau
  MATH_DATATYPE(kind=rck)                    :: vav(nbw,nbw)

  MATH_DATATYPE(kind=rck), allocatable        :: tmpGPU(:)
  MATH_DATATYPE(kind=rck), pointer            :: vmrGPU(:), umcGPU(:)
  MATH_DATATYPE(kind=rck), pointer            :: vmrGPU_2d(:,:), umcGPU_2d(:,:)
  MATH_DATATYPE(kind=rck), allocatable        :: vmrCPU(:,:), umcCPU(:,:), vmrCPU_qr(:,:)
  MATH_DATATYPE(kind=rck), allocatable        :: vr(:)
  MATH_DATATYPE(kind=rck)                     :: taublock(nbw), vrlblock(nbw)
  MATH_DATATYPE(kind=rck), allocatable, target:: ex_buff(:)
  MATH_DATATYPE(kind=rck), pointer, contiguous:: ex_buff2d(:,:)

#if REALCASE == 1
  ! needed for blocked QR decomposition
  integer(kind=ik)                            :: PQRPARAM(11), work_size
  real(kind=rk)                               :: dwork_size(1)
  real(kind=rk), allocatable                  :: work_blocked(:), tauvector(:), blockheuristic(:)
#endif
  integer(kind=C_intptr_T)                    :: a_dev, vmr_dev, umc_dev, tmat_dev, vav_dev
  integer(kind=C_intptr_T)                    :: a_dev0, a_dev1, vmr_dev0, vmr_dev1, umc_dev0, umc_dev1
  type(c_ptr)                                 :: a_dev_ptr, vmr_dev_ptr, umc_dev_ptr
  MATH_DATATYPE(kind=rck), pointer            :: vmr_dev_fortran_ptr, umc_dev_fortran_ptr, a_dev_fortran_ptr
  type(c_ptr)                                 :: vmr_host, umc_host
  MATH_DATATYPE(kind=rck), pointer            :: vmr_debug(:), umc_debug(:)
#ifdef WITH_MPI
  !integer(kind=ik), external                  :: numroc -> use elpa_scalapack
#endif
  integer(kind=ik)                            :: ierr
  integer(kind=ik)                            :: cur_l_rows, cur_l_cols
#ifndef WITH_OPENMP_TRADITIONAL
  integer(kind=ik)                            :: vmr_size, umc_size
  integer(kind=ik)                            :: l_rows2, vmr_size2, umc_size2
#endif
  integer(kind=c_intptr_t)                    :: lc_start, lc_end
#if COMPLEXCASE == 1
  integer(kind=c_intptr_t)                    :: lce_1, lcs_1, lre_1
#endif
  integer(kind=ik)                            :: lr_end
  !integer(kind=ik)                            :: na_cols
  !integer(kind=BLAS_KIND)                     :: na_colsBLAS
#if COMPLEXCASE == 1
  integer(kind=ik)                            :: na_rows
  integer(kind=BLAS_KIND)                     :: na_rowsBLAS
#endif

  logical, intent(in)                         :: wantDebug
  logical, intent(out)                        :: success
  logical                                     :: successGPU
  integer(kind=ik)                            :: istat
  character(200)                              :: errorMessage
  integer(kind=ik)                            :: min_tile_size, error

#if REALCASE == 1
  logical, intent(in)                         :: useQR
#endif
  integer(kind=ik)                            :: mystart, myend, m_way, n_way, work_per_thread, m_id, n_id, n_threads, &
                                                ii, off, lrex
  integer(kind=c_intptr_t), parameter           :: size_of_datatype = size_of_&
                                                                    &PRECISION&
                                                                    &_&
                                                                    &MATH_DATATYPE

  logical                                     :: useGPU_reduction_lower_block_to_tridiagonal
  integer(kind=ik),intent(in)                 :: max_threads
  integer(kind=ik)                            :: max_threads_used
  logical                                     :: do_memcpy
  integer(kind=ik)                            :: i_blk,blk_off, blk_end

  integer(kind=MPI_KIND)                      :: bcast_request, allreduce_request1, allreduce_request2, &
                                                 allreduce_request3, allreduce_request4, allreduce_request5, &
                                                 allreduce_request6
  integer(kind=MPI_KIND), allocatable         :: breq(:)
  
  logical                                     :: useNonBlockingCollectivesCols
  logical                                     :: useNonBlockingCollectivesRows
  integer(kind=c_int)                         :: non_blocking_collectives_rows, non_blocking_collectives_cols
  integer(kind=c_int)                         :: myThreadID, mimick
  integer(kind=c_int)                         :: memcols

  integer(kind=c_intptr_t)                    :: gpuHandle, my_stream


  max_threads_used = max_threads
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
  if (useGPU) then
    if (gpu_vendor() == OPENMP_OFFLOAD_GPU) then
      max_threads_used=1
    endif
  endif
#endif

  if(useGPU) then
    gpuString = "_gpu"
  else
    gpuString = ""
  endif

  call obj%timer%start("bandred_&
  &MATH_DATATYPE&
  &" // &
  PRECISION_SUFFIX // &
  gpuString )

  call obj%get("nbc_row_elpa2_full_to_band", non_blocking_collectives_rows, error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "Problem setting option for non blocking collectives for rows in elpa2_bandred. Aborting..."
    success = .false.
    return
  endif

  call obj%get("nbc_col_elpa2_full_to_band", non_blocking_collectives_cols, error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "Problem setting option for non blocking collectives for cols in elpa2_bandred. Aborting..."
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

  useGPU_reduction_lower_block_to_tridiagonal = .false.
 
  if (useGPU) then
    useGPU_reduction_lower_block_to_tridiagonal = .true.
#if REALCASE == 1
    if (useQR) then
      !in this case switch off GPU usage for step "reduce current block to lower triangular form"
      ! since this is done by QR decomposition
      useGPU_reduction_lower_block_to_tridiagonal = .false.
    endif
#endif
  endif

  my_prow = obj%mpi_setup%myRank_comm_rows
  my_pcol = obj%mpi_setup%myRank_comm_cols

  np_rows = obj%mpi_setup%nRanks_comm_rows
  np_cols = obj%mpi_setup%nRanks_comm_cols

  !if (wantDebug) call obj%timer%start("mpi_communication")

  !call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND) ,my_prowMPI ,mpierr)
  !call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND) ,np_rowsMPI ,mpierr)
  !call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND) ,my_pcolMPI ,mpierr)
  !call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND) ,np_colsMPI ,mpierr)

  !my_prow = int(my_prowMPI,kind=c_int)
  !np_rows = int(np_rowsMPI,kind=c_int)
  !my_pcol = int(my_pcolMPI,kind=c_int)
  !np_cols = int(np_colsMPI,kind=c_int)

  !if (wantDebug) call obj%timer%stop("mpi_communication")
  success = .true.


  ! Semibandwith nbw must be a multiple of blocksize nblk
  if (mod(nbw,nblk)/=0) then
    if (my_prow==0 .and. my_pcol==0) then
      if (wantDebug) then
        write(error_unit,*) 'ELPA2_bandred_&
                             &MATH_DATATYPE&
                             &: ERROR: nbw=',nbw,', nblk=',nblk
        write(error_unit,*) 'ELPA2_bandred_&
                             &MATH_DATATYPE&
                             &: ELPA2 works only for nbw==n*nblk'
      endif
      success = .false.
      return
    endif
  endif

  ! na_rows in used nowhere; only na_cols
  if (useGPU) then

    ! Here we convert the regular host array into a pinned host array
    successGPU = gpu_malloc(a_dev, matrixRows*matrixCols* size_of_datatype)
    check_alloc_gpu("bandred: a_dev", successGPU)

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      successGPU = gpu_host_register(int(loc(vav),kind=c_intptr_t), &
                  nbw * nbw * size_of_datatype,&
                  gpuHostRegisterDefault)
      check_host_register_gpu("bandred: vav", successGPU)
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    endif
#endif

    successGPU = gpu_malloc(vav_dev, nbw*nbw* size_of_datatype)
    check_alloc_gpu("bandred: vav_dev", successGPU)
  endif ! useGPU

  ! Matrix is split into tiles; work is done only for tiles on the diagonal or above

  tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size

  ! make tile_size a smallest possible multiple of previously defined tile size, such that it is
  ! larger or equal to min_tile_size
  ! min_tile_size has been originally hardcoded as 128 * max(np_rows, np_cols), so it is now the implicit value
  ! it can, however, be set by the user
  call obj%get("min_tile_size", min_tile_size ,error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "Problem setting option for min_tile_size. Aborting..."
    success = .false.
    return
  endif
  if(min_tile_size == 0) then
    ! not set by the user, use the default value
    min_tile_size = 128*max(np_rows, np_cols)
  endif
  tile_size = ((min_tile_size-1)/tile_size+1)*tile_size

  l_rows_tile = tile_size/np_rows ! local rows of a tile
  l_cols_tile = tile_size/np_cols ! local cols of a tile

#if REALCASE == 1
  if (useQR) then

    if (which_qr_decomposition == 1) then
      call qr_pqrparam_init(obj,pqrparam(1:11),    nblk,'M',0,   nblk,'M',0,   nblk,'M',1,'s')
      allocate(tauvector(na), stat=istat, errmsg=errorMessage)
      check_allocate("bandred: tauvector", istat, errorMessage)

      allocate(blockheuristic(nblk), stat=istat, errmsg=errorMessage)
      check_allocate("bandred: blockheuristic", istat, errorMessage)

      l_rows = local_index(na, my_prow, np_rows, nblk, -1)
      allocate(vmrCPU_qr(max(l_rows,1),na), stat=istat, errmsg=errorMessage)
      check_allocate("bandred: vmrCPU_qr", istat, errorMessage)

      vmrCols = na

#ifdef USE_ASSUMED_SIZE_QR
      call qr_pdgeqrf_2dcomm_&
           &PRECISION&
           &(obj, a_mat, matrixRows, matrixCols, vmrCPU_qr, max(l_rows,1), vmrCols, tauvector(1), na, tmat(1,1,1), &
                             nbw, nbw, dwork_size, 1, -1, na, nbw, nblk, nblk, na, na, 1, 0, PQRPARAM(1:11), &
                             mpi_comm_rows, mpi_comm_cols, blockheuristic)

#else
      call qr_pdgeqrf_2dcomm_&
           &PRECISION&
           &(obj, a_mat(1:matrixRows,1:matrixCols), matrixCols, matrixRows, vmrCPU_qr(1:max(l_rows,1),1:vmrCols), max(l_rows,1), &
                             vmrCols, tauvector(1:na), na, tmat(1:nbw,1:nbw,1), nbw, &
                             nbw, dwork_size(1:1), 1, -1, na, nbw, nblk, nblk, na, na, 1, 0, PQRPARAM(1:11), &
                             mpi_comm_rows, mpi_comm_cols, blockheuristic)
#endif

      work_size = int(dwork_size(1))
      allocate(work_blocked(work_size), stat=istat, errmsg=errorMessage)
      check_allocate("bandred: work_blocked", istat, errorMessage)
      work_blocked = 0.0_rk
      deallocate(vmrCPU_qr, stat=istat, errmsg=errorMessage)
      check_deallocate("bandred: vmrCPU_qr", istat, errorMessage)

    endif ! which_qr_decomposition

  endif ! useQr
#endif /* REALCASE */

  blk_end = (na-1)/nbw
  if (useGPU) then
 
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      successGPU = gpu_host_register(int(loc(a_mat),kind=c_intptr_t), &
                  matrixRows*matrixCols*size_of_datatype, gpuHostRegisterDefault)
      check_host_register_gpu("bandred: a_mat", successGPU)
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    endif
#endif

#ifndef WITH_OPENMP_TRADITIONAL
    cur_l_rows = 0
    cur_l_cols = 0
#endif

#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("bandred: a_dev", successGPU)

    successGPU = gpu_memcpy_async(a_dev, int(loc(a_mat),kind=c_intptr_t), &
                  matrixRows*matrixCols*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
    check_memcpy_gpu("bandred: a_dev", successGPU)

    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("bandred: a_dev", successGPU)
    ! synchronize streamsPerThread; maybe not neccessary
    successGPU = gpu_stream_synchronize()
    check_stream_synchronize_gpu("bandred: a_dev", successGPU)
#else
    successGPU = gpu_memcpy(a_dev, int(loc(a_mat),kind=c_intptr_t), &
                  matrixRows*matrixCols*size_of_datatype, gpuMemcpyHostToDevice)
    check_memcpy_gpu("bandred: a_dev", successGPU)
#endif

    successGPU = gpu_malloc(tmat_dev, nbw*nbw*size_of_datatype)
    check_alloc_gpu("bandred: tmat_dev", successGPU)

#ifdef WITH_GPU_STREAMS
    successGPU = gpu_host_register(int(loc(tmat),kind=c_intptr_t), &
                 nbw *nbw *numBlocks * size_of_datatype,&
                  gpuHostRegisterDefault)
    check_host_register_gpu("bandred: tmat", successGPU)
#endif




#ifndef WITH_OPENMP_TRADITIONAL
    istep = (na-1)/nbw
    blk_end = (na-1)/nbw
    n_cols = min(na,(istep+1)*nbw)-istep*nbw
    l_cols = local_index(istep*nbw, my_pcol, np_cols, nblk, -1)
    l_rows = local_index(istep*nbw, my_prow, np_rows, nblk, -1)
    cur_l_rows = max(l_rows,1)
    cur_l_cols = max(l_cols,1)
    vmr_size = cur_l_rows*2*n_cols
    umc_size = cur_l_cols*2*n_cols

    istep = (na-1)/nbw - 1
    n_cols = min(na,(istep+1)*nbw)-istep*nbw
    l_cols = local_index(istep*nbw, my_pcol, np_cols, nblk, -1)
    l_rows2 = local_index(istep*nbw, my_prow, np_rows, nblk, -1)
    cur_l_rows = max(l_rows2,1)
    cur_l_cols = max(l_cols,1)
    vmr_size2 = cur_l_rows*2*n_cols
    umc_size2 = cur_l_cols*2*n_cols

    l_rows = max(l_rows,l_rows2)
    vmr_size = max(vmr_size,vmr_size2)
    umc_size = max(umc_size,umc_size2)

    allocate(vr(l_rows + 1), stat=istat, errmsg=errorMessage)
    check_allocate("bandred: vr", istat, errorMessage)

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      successGPU = gpu_malloc_host(vmr_host,vmr_size*size_of_datatype)
      check_host_alloc_gpu("bandred: vmr_host", successGPU)
      call c_f_pointer(vmr_host, vmrGPU, (/vmr_size/))
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    else
      allocate(vmrGPU(vmr_size))
    endif
#endif

    successGPU = gpu_malloc(vmr_dev, vmr_size*size_of_datatype)
    check_alloc_gpu("bandred: vmr_dev", successGPU)


#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      successGPU = gpu_malloc_host(umc_host,umc_size*size_of_datatype)
      check_host_alloc_gpu("bandred: umc_host", successGPU)
      call c_f_pointer(umc_host, umcGPU, (/umc_size/))
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    else
      allocate(umcGPU(umc_size))
    endif
#endif

    successGPU = gpu_malloc(umc_dev, umc_size*size_of_datatype)
    check_alloc_gpu("bandred: umc_dev", successGPU)


#endif /* WITH_OPENMP_TRADITIONAL */

  endif ! useGPU

  do istep = blk_end, 1, -1

    n_cols = MIN(na,(istep+1)*nbw) - istep*nbw ! Number of columns in current step

    ! Number of local columns/rows of remaining matrix
    l_cols = local_index(istep*nbw, my_pcol, np_cols, nblk, -1)
    l_rows = local_index(istep*nbw, my_prow, np_rows, nblk, -1)


    max_l_rows = max(l_rows,1)
    max_l_cols = max(l_cols,1)

    ! Allocate vmr and umc to their exact sizes so that they can be used in bcasts and reduces

    if (useGPU) then
#ifndef WITH_OPENMP_TRADITIONAL
      vmr_size = max_l_rows * 2 * n_cols
      umc_size = max_l_cols * 2 * n_cols
#else
      allocate(vr(l_rows + 1), stat=istat, errmsg=errorMessage)
      check_allocate("bandred: vr", istat, errorMessage)

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
        successGPU = gpu_malloc_host(vmr_host, max_l_rows*2*n_cols*size_of_datatype)
        check_host_alloc_gpu("bandred: vmr_host", successGPU)
        call c_f_pointer(vmr_host, vmrGPU, [max_l_rows*2*n_cols])
        call c_f_pointer(vmr_host, vmrGPU_2d, [max_l_rows,2*n_cols])
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      else
        allocate(vmrGPU(max_l_rows*2*n_cols))
        allocate(vmrGPU_2d(max_l_rows,2*n_cols))
      endif
#endif

      successGPU = gpu_malloc(vmr_dev, max_l_rows*2*n_cols*size_of_datatype)
      check_alloc_gpu("bandred: vmr_dev", successGPU)


#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
        successGPU = gpu_malloc_host(umc_host,max_l_cols*2*n_cols*size_of_datatype)
        check_host_alloc_gpu("bandred: umc_host", successGPU)
        call c_f_pointer(umc_host, umcGPU, [max_l_cols*2*n_cols])
        call c_f_pointer(umc_host, umcGPU_2d, [max_l_cols,2*n_cols])
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      else
        allocate(umcGPU(max_l_cols*2*n_cols))
        allocate(umcGPU_2d(max_l_cols,2*n_cols))
      endif
#endif

      successGPU = gpu_malloc(umc_dev, max_l_cols*2*n_cols*size_of_datatype)
      check_alloc_gpu("bandred: umc_dev", successGPU)
#endif
    else ! GPU not used

      ! unify the the name vmr and vmrCPU, as well as vmrGPU
      ! the same for umcCPU and umcGPU
      ! Allocate vmr and umcCPU to their exact sizes so that they can be used in bcasts and reduces

      allocate(vmrCPU(max_l_rows,2*n_cols), stat=istat, errmsg=errorMessage)
      check_allocate("bandred: vmrCPU", istat, errorMessage)

      allocate(umcCPU(max_l_cols,2*n_cols), stat=istat, errmsg=errorMessage)
      check_allocate("bandred: umcCPU", istat, errorMessage)

      allocate(vr(l_rows+1), stat=istat, errmsg=errorMessage)
      check_allocate("bandred: vr", istat, errorMessage)

    endif ! use GPU

    if (useGPU) then
      vmrGPU(1 : max_l_rows * n_cols) = 0.0_rck
#ifndef WITH_OPENMP_TRADITIONAL
      umcGPU(1 : umc_size) = 0.0_rck
#else
      umcGPU(1: max_l_cols*2*n_cols) = 0.0_rck
#endif
    else ! useGPU
      vmrCPU(1:l_rows,1:n_cols) = 0.0_rck
    endif ! useGPU


    vr(:) = 0.0_rck
    tmat(:,:,istep) = 0.0_rck
    if (useGPU) then
      lc_start = local_index(istep*nbw+1, my_pcol, np_cols, nblk, -1)
      lc_end   = local_index(istep*nbw+n_cols, my_pcol, np_cols, nblk, -1)
      lr_end   = local_index((istep-1)*nbw + n_cols, my_prow, np_rows, nblk, -1)

      if (lc_start .le. 0) lc_start = 1

      do_memcpy = .false.

      ! Note: mod(nbw,nblk) == 0
      do i_blk = 1, nbw/nblk
        blk_off = (i_blk-1) * nblk
        cur_pcol = pcol(istep*nbw+1+blk_off, nblk, np_cols)

        if (my_pcol == cur_pcol) then
          do_memcpy = .true.
        endif
      enddo

      if (do_memcpy) then
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
        if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif

#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("bandred: a_dev -> a_mat", successGPU)

          successGPU = gpu_memcpy2d_async(int(loc(a_mat(1, lc_start)),kind=c_intptr_t), &
                        int((matrixRows*size_of_datatype),kind=c_intptr_t), &
                        (a_dev + int( ( (lc_start-1) * matrixRows*size_of_datatype),kind=c_intptr_t )), &
                        int(matrixRows*size_of_datatype,kind=c_intptr_t), &
                        int(lr_end*size_of_datatype,kind=c_intptr_t), &
                        int((lc_end - lc_start+1),kind=c_intptr_t),int(gpuMemcpyDeviceToHost,kind=c_int), &
                        my_stream)
          check_memcpy_gpu("bandred: a_dev -> a_mat", successGPU)
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("bandred: a_dev -> a_mat", successGPU)
          ! sychronize streamsPerThread; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("bandred: a_dev -> a_mat", successGPU)
#else
          successGPU = gpu_memcpy2d(int(loc(a_mat(1, lc_start)),kind=c_intptr_t), &
                        int((matrixRows*size_of_datatype),kind=c_intptr_t), &
                        (a_dev + int( ( (lc_start-1) * matrixRows*size_of_datatype),kind=c_intptr_t )), &
                        int(matrixRows*size_of_datatype,kind=c_intptr_t), &
                        int(lr_end*size_of_datatype,kind=c_intptr_t), &
                        int((lc_end - lc_start+1),kind=c_intptr_t),int(gpuMemcpyDeviceToHost,kind=c_int))
          check_memcpy_gpu("bandred: a_dev -> a_mat", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
        else
          do memcols = lc_start, lc_end
            successGPU = gpu_memcpy(int(loc(a_mat(1,memcols)),kind=c_intptr_t), &
                                    a_dev + int(((memcols-1) * matrixRows*size_of_datatype),kind=c_intptr_t), &
                                    int(lr_end*size_of_datatype,kind=c_intptr_t), gpuMemcpyDeviceToHost)
            check_memcpy_gpu("bandred: a_dev -> a_mat (loop)", successGPU)
          enddo
        endif
#endif
      endif ! do_memcpy
    endif ! useGPU

    ! Reduce current block to lower triangular form
#if REALCASE == 1
    if (useQR) then
      if (which_qr_decomposition == 1) then
        vmrCols = 2*n_cols
#ifdef USE_ASSUMED_SIZE_QR
        call qr_pdgeqrf_2dcomm_&
             &PRECISION&
             &(obj, a_mat, matrixRows, matrixCols, vmrCPU, max_l_rows, vmrCols, tauvector(1), &
                               na, tmat(1,1,istep), nbw, nbw, work_blocked, work_size,        &
                                 work_size, na, n_cols, nblk, nblk,        &
                                 istep*nbw+n_cols-nbw, istep*nbw+n_cols, 1,&
                                 0, PQRPARAM(1:11), mpi_comm_rows, mpi_comm_cols,&
                                 blockheuristic)

#else
        call qr_pdgeqrf_2dcomm_&
             &PRECISION&
             &(obj, a_mat(1:matrixRows,1:matrixCols), matrixRows, matrixCols, vmrCPU(1:max_l_rows,1:vmrCols) ,   &
                                max_l_rows, vmrCols, tauvector(1:na), na, &
                                 tmat(1:nbw,1:nbw,istep), nbw, nbw, work_blocked(1:work_size), work_size, &
                                 work_size, na, n_cols, nblk, nblk,        &
                                 istep*nbw+n_cols-nbw, istep*nbw+n_cols, 1,&
                                 0, PQRPARAM(1:11), mpi_comm_rows, mpi_comm_cols,&
                                 blockheuristic)
#endif
      endif

    else !useQR
#endif /* REALCASE == 1 */

       call obj%timer%start("hh_block")
       
       allocate(blockinfo(4,n_cols/nblk+1))
       iblock=0
       do lc = n_cols, 1, -1
          ncol = istep*nbw + lc
          if((lc.eq.n_cols).or.(mod(ncol,nblk).eq.0)) then
             cur_pcol = pcol(ncol, nblk, np_cols) ! Processor column owning current block
             !new block
             iblock=iblock+1
             lch = local_index(ncol, my_pcol, np_cols, nblk, -1) ! HV local column number
             blockinfo(1,iblock)=cur_pcol !owner of this block
             blockinfo(2,iblock)=((lch-1)/nblk)*nblk+1 !first a_mat index of this block
             blockinfo(3,iblock)=mod(ncol-1,nblk)+1 !length of block
             blockinfo(4,iblock)=lc !last local cell indes of this block
          end if
       end do
       nblocks=iblock
          
       allocate(ex_buff(l_rows*n_cols))
       lrex  = l_rows
       ex_buff2d(1:lrex,1:n_cols) => ex_buff
       do iblock=1,nblocks
          c_start = blockinfo(2,iblock)
          blc_end = blockinfo(4,iblock)
          blc_len=blockinfo(3,iblock)
          blc_start=blc_end-blc_len+1
          cur_pcol = blockinfo(1,iblock)
          
          if(my_pcol.eq.cur_pcol) then
#ifdef WITH_OPENMP_TRADITIONAL
             !$omp parallel do private(off)
#endif
             do off=1,blc_len
                ex_buff2d(1:lrex,blc_start+off-1)=a_mat(1:lrex,c_start+off-1)
             end do
#ifdef WITH_OPENMP_TRADITIONAL
             !$omp end parallel do
#endif
          end if
       end do
#ifdef WITH_MPI
       call obj%timer%start("bcast_multi")
       if(lrex.gt.0 .and. np_rows*np_cols.gt.1) then
          allocate(breq(0:nblocks-1))
          do j=0,nblocks-1
             !mix bcasts with different root to level the stess on the network
             iblock=mod(j+my_prow,nblocks)+1
             c_start = blockinfo(2,iblock)
             blc_end = blockinfo(4,iblock)
             blc_len=blockinfo(3,iblock)
             blc_start=blc_end-blc_len+1
             cur_pcol = blockinfo(1,iblock)
             call mpi_ibcast(ex_buff2d(1,blc_start), int(lrex*blc_len,kind=MPI_KIND), &
                  MPI_MATH_DATATYPE_PRECISION, int(cur_pcol,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), &
                  breq(j),mpierr)     
          end do
          call mpi_waitall(nblocks, breq, MPI_STATUSES_IGNORE, mpierr)
          deallocate(breq)
       end if
       call obj%timer%stop("bcast_multi")
#endif
       call obj%timer%start("hh_trans")
       off=0
       do lc = n_cols, 1, -1
          ncol = istep*nbw + lc ! absolute column number of householder Vector
          nrow = ncol - nbw ! Absolute number of pivot row  
          if (nrow == 1) then !done
             taublock(1)=0. 
             exit
          end if
          
          lr  = local_index(nrow, my_prow, np_rows, nblk, -1) ! current row length
          off=off+1
          call get_hh_vec(ex_buff2d(1:lr,n_cols-off+1),vr,tau,vrl)
          
          call apply_ht(tau,vr,ex_buff2d(:,1:n_cols-off))
          if (useGPU_reduction_lower_block_to_tridiagonal) then
             vmrGPU(max_l_rows * (lc - 1) + 1 : max_l_rows * (lc - 1) + lr) = vr(1:lr)
          else
             vmrCPU(1:lr,lc) = vr(1:lr)
          endif
#if REALCASE == 1
          taublock(lc) = tau
#else
          taublock(lc) = conjg(tau)
#endif
          vrlblock(lc)=vrl
       end do
       call obj%timer%stop("hh_trans")
          
       do iblock=1,nblocks
          c_start = blockinfo(2,iblock)
          blc_end = blockinfo(4,iblock)
          blc_len=blockinfo(3,iblock)
          blc_start=blc_end-blc_len+1
          cur_pcol = blockinfo(1,iblock)
          
          if(my_pcol.eq.cur_pcol) then
#ifdef WITH_OPENMP_TRADITIONAL
             !$omp  parallel do private(off,lc,lch,ncol,nrow,lr)
#endif
             do off=1,blc_len
                lc=blc_start+off-1
                lch=c_start+off-1
                ncol = istep*nbw + lc ! absolute column number of householder Vector
                nrow = ncol - nbw ! Absolute number of pivot row   
                lr  = local_index(nrow, my_prow, np_rows, nblk, -1) ! current row length

                if (nrow.gt.1) then
                   if (useGPU_reduction_lower_block_to_tridiagonal) then
                      a_mat(1:lr,lch)=vmrGPU(max_l_rows * (lc - 1) + 1 : max_l_rows * (lc - 1) + lr) 
                   else
                      a_mat(1:lr,lch)=vmrCPU(1:lr,lc)  
                   endif
                   if (my_prow==prow(nrow, nblk, np_rows)) a_mat(lr,lch) = vrlblock(lc)
                   a_mat(lr+1:lrex,c_start+off-1)=ex_buff2d(lr+1:lrex,blc_start+off-1)
                else
                   a_mat(1:lrex,c_start+off-1)=ex_buff2d(1:lrex,blc_start+off-1)
                end if
             end do
#ifdef WITH_OPENMP_TRADITIONAL
             !$omp end parallel do
#endif             
          end if
       end do

          
       deallocate(blockinfo)
       deallocate(ex_buff)

       call obj%timer%stop("hh_block")

       if (useGPU_reduction_lower_block_to_tridiagonal) then
        ! store column tiles back to GPU
        if (do_memcpy) then
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
          if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif

#ifdef WITH_GPU_STREAMS
            my_stream = obj%gpu_setup%my_stream
            successGPU = gpu_stream_synchronize(my_stream)
            check_stream_synchronize_gpu("bandred: a_mat -> a_dev", successGPU)

            successGPU = gpu_memcpy2d_async((a_dev+ &
                         int(((lc_start-1)*matrixRows*size_of_datatype),kind=c_intptr_t)), &
                         int(matrixRows*size_of_datatype,kind=c_intptr_t), int(loc(a_mat(1,lc_start)),kind=c_intptr_t), &
                         int(matrixRows*size_of_datatype,kind=c_intptr_t), &
                         int(lr_end*size_of_datatype,kind=c_intptr_t), &
                         int((lc_end - lc_start+1),kind=c_intptr_t), &
                         int(gpuMemcpyHostToDevice,kind=c_int), my_stream)
            check_memcpy_gpu("bandred: a_mat -> a_dev", successGPU)
            successGPU = gpu_stream_synchronize(my_stream)
            check_stream_synchronize_gpu("bandred: a_mat -> a_dev", successGPU)
            ! sychronize streamsPerThread; maybe not neccessary
            successGPU = gpu_stream_synchronize()
            check_stream_synchronize_gpu("bandred: a_mat -> a_dev", successGPU)
#else
            successGPU = gpu_memcpy2d((a_dev+ &
                         int(((lc_start-1)*matrixRows*size_of_datatype),kind=c_intptr_t)), &
                         int(matrixRows*size_of_datatype,kind=c_intptr_t), int(loc(a_mat(1,lc_start)),kind=c_intptr_t), &
                         int(matrixRows*size_of_datatype,kind=c_intptr_t), &
                         int(lr_end*size_of_datatype,kind=c_intptr_t), &
                         int((lc_end - lc_start+1),kind=c_intptr_t), &
                         int(gpuMemcpyHostToDevice,kind=c_int))
            check_memcpy_gpu("bandred: a_mat -> a_dev", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
          else
            do memcols = lc_start, lc_end
              successGPU = gpu_memcpy(a_dev + int(((memcols-1) * matrixRows*size_of_datatype),kind=c_intptr_t), &
                                     int(loc(a_mat(1,memcols)),kind=c_intptr_t), &
                                     int(lr_end*size_of_datatype,kind=c_intptr_t), gpuMemcpyHostToDevice)
              check_memcpy_gpu("bandred: a_dev -> a_mat (loop)", successGPU)
            enddo
          endif
#endif
        endif ! do_memcopy
      endif ! (useGPU_reduction_lower_block_to_tridiagonal

      ! Calculate scalar products of stored Householder vectors.
      ! This can be done in different ways, we use dsyrk

      vav = 0
      call obj%timer%start("blas0")
      if (useGPU_reduction_lower_block_to_tridiagonal) then
        if (l_rows > 0) then
#if REALCASE == 1
          call PRECISION_SYRK('U', 'T',            &
#endif
#if COMPLEXCASE == 1
          call PRECISION_HERK('U', 'C',            &
#endif
                           int(n_cols,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), ONE, &
                           vmrGPU, int(max(l_rows, 1),kind=BLAS_KIND), &
                           ZERO, vav, int(nbw,kind=BLAS_KIND))
        endif
      else ! useGPU_reduction_to_tridiagonal
        if (l_rows > 0) then
#if REALCASE == 1
          call PRECISION_SYRK('U', 'T',           &
#endif
#if COMPLEXCASE == 1
          call PRECISION_HERK('U', 'C',           &
#endif
                            int(n_cols,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), ONE, vmrCPU, &
                            int(max(l_rows, 1),kind=BLAS_KIND), ZERO, vav, int(nbw,kind=BLAS_KIND))
        endif
      endif
      call obj%timer%stop("blas0")
#if REALCASE == 1
      call symm_matrix_allreduce_&
#endif
#if COMPLEXCASE == 1
      call herm_matrix_allreduce_&
#endif
         &PRECISION &
                         (obj, n_cols,vav, nbw, nbw,mpi_comm_rows, .true., success)
      if (.not.(success)) then
        write(error_unit,*) "Error when calling symm/herm_allreduce. Aborting..."
        return
      endif

         ! Calculate triangular matrix T for block Householder Transformation
      call obj%timer%start("blas1")
      do lc=n_cols,1,-1
         tau = taublock(lc)
         tmat(lc,lc,istep)=tau
         if (lc < n_cols) then
            call PRECISION_TRMV('U', BLAS_TRANS_OR_CONJ, 'N',&
                 int(n_cols-lc,kind=BLAS_KIND), tmat(lc+1,lc+1,istep), &
                 int(nbw,kind=BLAS_KIND), vav(lc+1,lc), 1_BLAS_KIND)
            
#if REALCASE == 1
            tmat(lc,lc+1:n_cols,istep) = -tau * vav(lc+1:n_cols,lc)
#endif
#if COMPLEXCASE == 1
            tmat(lc,lc+1:n_cols,istep) = -tau * conjg(vav(lc+1:n_cols,lc))
#endif
         endif
      enddo
      call obj%timer%stop("blas1")
#if REALCASE == 1
    endif !useQR
#endif

#if REALCASE == 1
    if (useGPU .and. useQR ) then
      ! copy the data for furhter usage
      ! qr worked on *CPU arrarys
      !vmrGPU(1:max_l_rows * n_cols) = vmrCPU(1:max_l_rows,1:n_cols)
      if (do_memcpy) then
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
        if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("bandred: a_mat -> a_dev", successGPU)

          successGPU = gpu_memcpy2d_async((a_dev+ &
                       int(((lc_start-1)*matrixRows*size_of_datatype),kind=c_intptr_t)), &
                       int(matrixRows*size_of_datatype,kind=c_intptr_t), int(loc(a_mat(1,lc_start)),kind=c_intptr_t), &
                       int(matrixRows*size_of_datatype,kind=c_intptr_t), &
                       int(lr_end*size_of_datatype,kind=c_intptr_t), &
                       int((lc_end - lc_start+1),kind=c_intptr_t), &
                       int(gpuMemcpyHostToDevice,kind=c_int), my_stream)
          check_memcpy_gpu("bandred: a_mat -> a_dev", successGPU)
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("bandred: a_mat -> a_dev", successGPU)
          ! sychronize streamsPerThread; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("bandred: a_mat -> a_dev", successGPU)
#else
          successGPU = gpu_memcpy2d((a_dev+ &
                       int(((lc_start-1)*matrixRows*size_of_datatype),kind=c_intptr_t)), &
                       int(matrixRows*size_of_datatype,kind=c_intptr_t), int(loc(a_mat(1,lc_start)),kind=c_intptr_t), &
                       int(matrixRows*size_of_datatype,kind=c_intptr_t), &
                       int(lr_end*size_of_datatype,kind=c_intptr_t), &
                       int((lc_end - lc_start+1),kind=c_intptr_t), &
                       int(gpuMemcpyHostToDevice,kind=c_int))
          check_memcpy_gpu("bandred: a_mat -> a_dev", successGPU)
#endif
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
        else
          do memcols = lc_start, lc_end
            successGPU = gpu_memcpy(a_dev + int(((memcols-1) * matrixRows*size_of_datatype),kind=c_intptr_t), &
                                   int(loc(a_mat(1,memcols)),kind=c_intptr_t), &
                                   int(lr_end*size_of_datatype,kind=c_intptr_t), gpuMemcpyHostToDevice)
            check_memcpy_gpu("bandred: a_dev -> a_mat (loop)", successGPU)
          enddo
        endif
#endif
      endif ! do_memcpy
    endif ! useGPU .and. useQR
#endif

    ! Transpose vmr -> vmc (stored in umc, second half)
    if (useGPU) then
      call elpa_transpose_vectors_&
           &MATH_DATATYPE&
           &_&
           &PRECISION &
                        (obj, vmrGPU(:), max_l_rows, mpi_comm_rows, &
                         umcGPU(max_l_cols * n_cols + 1:), max_l_cols, &
                         mpi_comm_cols, 1, istep*nbw, n_cols, nblk, max_threads_used, .true., &
                         success)
      if (.not.(success)) then
        write(error_unit,*) "Error in elpa_transpose_vectors. Aborting..."
        return
      endif
    else ! useGPU
      call elpa_transpose_vectors_&
           &MATH_DATATYPE&
           &_&
           &PRECISION &
                                        (obj, vmrCPU, max_l_rows, mpi_comm_rows, &
                                         umcCPU(1,n_cols+1), max_l_cols, mpi_comm_cols, &
                                         1, istep*nbw, n_cols, nblk, max_threads_used, .true., &
      success)
      if (.not.(success)) then
        write(error_unit,*) "Error in elpa_transpose_vectors. Aborting..."
        return
      endif
    endif ! useGPU

    ! Calculate umc = A**T * vmr
    ! Note that the distributed A has to be transposed
    ! Opposed to direct tridiagonalization there is no need to use the cache locality
    ! of the tiles, so we can use strips of the matrix


    !Code for Algorithm 4

    ! n_way is actually a branch for the number of OpenMP threads
    n_way = 1
#ifdef WITH_OPENMP_TRADITIONAL
    n_way = max_threads_used
    if (n_way > 1) then
      if (useGPU) then
        !$omp parallel do num_threads(max_threads_used) &
        !$omp default(none) &
        !$omp private(i) &
        !$omp shared(l_cols_tile, l_cols, umcGPU_2d, n_cols)
        do i=1,min(l_cols_tile, l_cols)
          umcGPU_2d(i,1:n_cols) = 0.0_rck
        enddo
      
        !$omp parallel do num_threads(max_threads_used) &
        !$omp default(none) &
        !$omp private(i) &
        !$omp shared(l_rows, vmrGPU_2d, n_cols)
        do i=1,l_rows
          vmrGPU_2d(i,n_cols+1:2*n_cols) = 0.0_rck
        enddo
      else ! useGPU
              !$omp parallel do num_threads(max_threads_used) &
        !$omp default(none) &
        !$omp private(i) &
        !$omp shared(l_cols_tile, l_cols, umcCPU, n_cols)
        do i=1,min(l_cols_tile, l_cols)
          umcCPU(i,1:n_cols) = 0.0_rck
        enddo

        !$omp parallel do num_threads(max_threads_used) &
        !$omp default(none) &
        !$omp private(i) &
        !$omp shared(l_rows, vmrCPU, n_cols)
        do i=1,l_rows
          vmrCPU(i,n_cols+1:2*n_cols) = 0.0_rck
        enddo
      endif ! useGPU

      if (useGPU) then
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("bandred: vmr_dev", successGPU)

        successGPU = gpu_memcpy_async(vmr_dev, int(loc(vmrGPU_2d(1,1)),kind=c_intptr_t), &
                     max_l_rows*2*n_cols*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
        check_memcpy_gpu("bandred: vmrGPU_2d -> vmr_dev", successGPU)

        successGPU = gpu_memcpy_async(umc_dev, int(loc(umcGPU_2d(1,1)), kind=c_intptr_t), &
                     max_l_cols*2*n_cols*size_of_datatype, &
                        gpuMemcpyHostToDevice, my_stream)
        check_memcpy_gpu("bandred: umcGPU -> umc_dev", successGPU)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("bandred: umcGPU -> umc_dev", successGPU)
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu("bandred: umcGPU -> umc_dev", successGPU)
#else
        successGPU = gpu_memcpy(vmr_dev, int(loc(vmrGPU_2d(1,1)),kind=c_intptr_t), &
                     max_l_rows*2*n_cols*size_of_datatype, gpuMemcpyHostToDevice)
        check_memcpy_gpu("bandred: vmrGPU_2d -> vmr_dev", successGPU)

        successGPU = gpu_memcpy(umc_dev, int(loc(umcGPU_2d(1,1)), kind=c_intptr_t), &
                     max_l_cols*2*n_cols*size_of_datatype, &
                        gpuMemcpyHostToDevice)
        check_memcpy_gpu("bandred: umcGPU -> umc_dev", successGPU)
#endif
      endif ! useGPU


      if (l_cols > 0 .and. l_rows > 0) then

        !SYMM variant 4
        !Partitioned Matrix Expression:
        ! Ct = Atl Bt + Atr Bb
        ! Cb = Atr' Bt + Abl Bb
        !
        !Loop invariant:
        ! Ct = Atl Bt + Atr Bb
        !
        !Update:
        ! C1 = A10'B0 + A11B1 + A21 B2
        !
        !This algorithm chosen because in this algoirhtm, the loop around the dgemm calls
        !is easily parallelized, and regardless of choise of algorithm,
        !the startup cost for parallelizing the dgemms inside the loop is too great
        !$omp  parallel do schedule(static,1) num_threads(max_threads_used) &
        !$omp  default(none) &
        !$omp  private(i, lcs, lce, lrs, lre, myThreadID, gpuHandle, successGPU) &
        !$omp  shared(istep, nbw, tile_size, obj, l_cols, l_cols_tile, l_rows, isSkewsymmetric, &
        !$omp&       n_cols, l_rows_tile, umcCPU, vmrCPU, a_mat, matrixRows, max_l_cols, max_l_rows, &
        !$omp&       useGPU, a_dev, umc_dev, vmr_dev)
        do i=0,(istep*nbw-1)/tile_size
          myThreadID=omp_get_thread_num()
          if (useGPU) then
            successGPU = gpu_setdevice(obj%gpu_setup%gpuDeviceArray(myThreadID))
          endif
          lcs = i*l_cols_tile+1                   ! local column start
          lce = min(l_cols, (i+1)*l_cols_tile)    ! local column end

          lrs = i*l_rows_tile+1                   ! local row start
          lre = min(l_rows, (i+1)*l_rows_tile)    ! local row end

          !C1 += [A11 A12] [B1
          !                 B2]
          if ( lre > lrs .and. l_cols > lcs ) then
            if (useGPU) then
              call obj%timer%start("gpublas")
              gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
              if (isSkewsymmetric) then
                call gpublas_PRECISION_GEMM('N', 'N', lre-lrs+1, n_cols, l_cols-lcs+1, &
                                    -ONE, a_dev + (lrs-1 + (lcs-1)*matrixRows)*size_of_datatype, matrixRows, &
                                    umc_dev + (lcs-1 + (n_cols+1+1)*max_l_cols)*size_of_datatype, max_l_cols, &
                                    ZERO, vmr_dev + (lrs-1+(n_cols+1-1)*max_l_rows)*size_of_datatype, max_l_rows, gpuHandle)
              else
                call gpublas_PRECISION_GEMM('N', 'N', lre-lrs+1, n_cols, l_cols-lcs+1, &
                                    ONE, a_dev + (lrs-1 +(lcs-1)*matrixRows)*size_of_datatype, matrixRows, &
                                    umc_dev+(lcs-1+(n_cols+1-1)*max_l_cols)*size_of_datatype, max_l_cols,    &
                                    ZERO, vmr_dev + (lrs-1 +(n_cols+1-1)*max_l_rows)*size_of_datatype, max_l_rows, gpuHandle)
              endif
              call obj%timer%stop("gpublas")
            else ! useGPU
              call obj%timer%start("blas")
              if (isSkewsymmetric) then
                call PRECISION_GEMM('N', 'N', int(lre-lrs+1,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), &
                                    int(l_cols-lcs+1,kind=BLAS_KIND),                                    &
                                    -ONE, a_mat(lrs,lcs), int(matrixRows,kind=BLAS_KIND),       &
                                    umcCPU(lcs,n_cols+1), int(max_l_cols,kind=BLAS_KIND),      &
                                    ZERO, vmrCPU(lrs,n_cols+1), int(max_l_rows,kind=BLAS_KIND) )
              else
                call PRECISION_GEMM('N', 'N', int(lre-lrs+1,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), &
                                    int(l_cols-lcs+1,kind=BLAS_KIND),                                    &
                                    ONE, a_mat(lrs,lcs), int(matrixRows,kind=BLAS_KIND),        &
                                    umcCPU(lcs,n_cols+1), int(max_l_cols,kind=BLAS_KIND),      &
                                    ZERO, vmrCPU(lrs,n_cols+1), int(max_l_rows,kind=BLAS_KIND) )

              endif
              call obj%timer%stop("blas")
            endif ! useGPU
          endif ! lre > lrs .and. l_cols > lcs

          ! C1 += A10' B0
          if ( lce > lcs .and. i > 0 ) then
            if (useGPU) then
              call obj%timer%start("gpublas")
              gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
              call gpublas_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', lce-lcs+1, n_cols, lrs-1,    &
                                   ONE, a_dev+(1-1+(lcs-1)*matrixRows)*size_of_datatype, matrixRows, &
                                   vmr_dev + (1-1 +(1-1)*max_l_rows)*size_of_datatype, max_l_rows, &
                                   ZERO, umc_dev+(lcs-1 +(1-1)*max_l_cols)*size_of_datatype, max_l_cols, gpuHandle)
              call obj%timer%stop("gpublas")
            else ! useGPU
              call obj%timer%start("blas")
              call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',     &
                                  int(lce-lcs+1,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(lrs-1,kind=BLAS_KIND), &
                                  ONE, a_mat(1,lcs), int(matrixRows,kind=BLAS_KIND),      &
                                  vmrCPU(1,1), int(max_l_rows,kind=BLAS_KIND),   &
                                  ZERO, umcCPU(lcs,1), int(max_l_cols,kind=BLAS_KIND) )
              call obj%timer%stop("blas")
            endif !useGPU
          endif ! lce > lcs .and. i > 0
        enddo ! i=0,(istep*nbw-1)/tile_size
      endif ! l_cols>0 .and. l_rows>0

      if (useGPU) then
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("bandred: vmr_dev", successGPU)

        successGPU = gpu_memcpy_async(int(loc(vmrGPU(1)),kind=c_intptr_t), vmr_dev, &
                     max_l_rows*2*n_cols*size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
        check_memcpy_gpu("bandred: vmr_dev -> vmrGPU", successGPU)

        successGPU = gpu_memcpy_async(int(loc(umcGPU(1)), kind=c_intptr_t), umc_dev, &
                     max_l_cols*2*n_cols*size_of_datatype, &
                        gpuMemcpyDeviceToHost, my_stream)
        check_memcpy_gpu("bandred: umc_dev -> umcGPU", successGPU)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("bandred: umc_dev -> umcGPU", successGPU)
        ! synchronize streamsPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu("bandred: umc_dev -> umcGPU", successGPU)

#else
        successGPU = gpu_memcpy(int(loc(vmrGPU(1)),kind=c_intptr_t), vmr_dev, &
                     max_l_rows*2*n_cols*size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("bandred: vmr_dev -> vmrGPU", successGPU)

        successGPU = gpu_memcpy(int(loc(umcGPU(1)), kind=c_intptr_t), umc_dev, &
                     max_l_cols*2*n_cols*size_of_datatype, &
                        gpuMemcpyDeviceToHost)
        check_memcpy_gpu("bandred: umc_dev -> umcGPU", successGPU)
#endif
      endif ! useGPU

    else ! n_way > 1
#endif /* WITH_OPENMP_TRADITIONAL */

      if (.not.useGPU) then
        umcCPU(1:l_cols,1:n_cols) = 0.0_rck
        vmrCPU(1:l_rows,n_cols+1:2*n_cols) = 0.0_rck
      endif ! useGPU

      if (l_cols > 0 .and. l_rows > 0) then

        if (useGPU) then
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
          if (gpu_vendor() /= OPENMP_OFFLOAD_GPU) then
#endif

#ifdef WITH_GPU_STREAMS
            my_stream = obj%gpu_setup%my_stream
            successGPU = gpu_stream_synchronize(my_stream)
            check_stream_synchronize_gpu("bandred: vmr_dev", successGPU)

            successGPU = gpu_memset_async(vmr_dev+max_l_rows*n_cols*size_of_datatype, &
                        0, max_l_rows*n_cols*size_of_datatype, my_stream)
            check_memset_gpu("bandred: vmr_dev", successGPU)

            successGPU = gpu_stream_synchronize(my_stream)
            check_stream_synchronize_gpu("bandred: vmr_dev", successGPU)
#else
            successGPU = gpu_memset(vmr_dev+max_l_rows*n_cols*size_of_datatype, &
                        0, max_l_rows*n_cols*size_of_datatype)
            check_memset_gpu("bandred: vmr_dev", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
          else
            allocate(vmr_debug(max_l_rows*n_cols))
            vmr_debug(:) = 0.
            successGPU = gpu_memcpy(vmr_dev+max_l_rows*n_cols*size_of_datatype, &
                                    int(loc(vmr_debug),kind=c_intptr_t), &
                                    max_l_rows*n_cols*size_of_datatype, gpuMemcpyHostToDevice)
            check_memcpy_gpu("bandred: vmr_debug -> vmr_dev", successGPU)
            deallocate(vmr_debug)
          endif
#endif

#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("bandred: vmrGPU", successGPU)

          successGPU = gpu_memcpy_async(vmr_dev, int(loc(vmrGPU(1)),kind=c_intptr_t), &
                        max_l_rows*n_cols*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("bandred: vmrGPU -> vmr_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("bandred: vmrGPU -> vmr_dev", successGPU)
          ! synchronize streamsPerThread; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("bandred: vmrGPU -> vmr_dev", successGPU)
#else
          successGPU = gpu_memcpy(vmr_dev, int(loc(vmrGPU(1)),kind=c_intptr_t), &
                        max_l_rows*n_cols*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("bandred: vmrGPU -> vmr_dev", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
          if (gpu_vendor() /= OPENMP_OFFLOAD_GPU) then
#endif

#ifdef WITH_GPU_STREAMS
            successGPU = gpu_stream_synchronize(my_stream)
            check_stream_synchronize_gpu("bandred: umc_dev", successGPU)

            successGPU = gpu_memset_async(umc_dev, 0, l_cols*n_cols*size_of_datatype, my_stream)
            check_memset_gpu("bandred: umc_dev", successGPU)

            successGPU = gpu_stream_synchronize(my_stream)
            check_stream_synchronize_gpu("bandred: umc_dev", successGPU)
#else
            successGPU = gpu_memset(umc_dev, 0, l_cols*n_cols*size_of_datatype)
            check_memset_gpu("bandred: umc_dev", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
          else
            allocate(umc_debug(l_cols*n_cols))
            umc_debug(:) = 0.
            successGPU = gpu_memcpy(umc_dev, int(loc(umc_debug),kind=c_intptr_t), &
                                    l_cols*n_cols*size_of_datatype, gpuMemcpyHostToDevice)
            check_memcpy_gpu("bandred: umc_debug -> umc_dev", successGPU)
            deallocate(umc_debug)
          endif
#endif


#ifdef WITH_GPU_STREAMS
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("bandred: umcGPU", successGPU)

          successGPU = gpu_memcpy_async(umc_dev+l_cols*n_cols*size_of_datatype, &
                        int(loc(umcGPU(1+l_cols*n_cols)),kind=c_intptr_t), &
#ifndef WITH_OPENMP_TRADITIONAL
                        (umc_size-l_cols*n_cols)*size_of_datatype, &
#else
                        (max_l_cols*2*n_cols-l_cols*n_cols)*size_of_datatype, &
#endif
                        gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("bandred: umcGPU -> umc_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("bandred: umcGPU -> umc_dev", successGPU)
          ! synchronize streamsPerThread; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("bandred: umcGPU -> umc_dev", successGPU)
#else
          successGPU = gpu_memcpy(umc_dev+l_cols*n_cols*size_of_datatype, &
                        int(loc(umcGPU(1+l_cols*n_cols)),kind=c_intptr_t), &
#ifndef WITH_OPENMP_TRADITIONAL
                        (umc_size-l_cols*n_cols)*size_of_datatype, &
#else
                        (max_l_cols*2*n_cols-l_cols*n_cols)*size_of_datatype, &
#endif
                        gpuMemcpyHostToDevice)
          check_memcpy_gpu("bandred: umcGPU -> umc_dev", successGPU)
#endif /* WITH_GPU_STREAMS */
        endif ! useGPU

        do i=0,(istep*nbw-1)/tile_size

          lcs = i*l_cols_tile+1
          lce = min(l_cols,(i+1)*l_cols_tile)
          if (lce<lcs) cycle
          lre = min(l_rows,(i+1)*l_rows_tile)

          if (useGPU) then
            call obj%timer%start("gpublas")
            gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
#ifndef WITH_OPENMP_TRADITIONAL
            call gpublas_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',                   &
                                       lce-lcs+1, n_cols, lre,     &
                                       ONE, (a_dev + ((lcs-1)*matrixRows* &
                                       size_of_datatype)),         &
                                       matrixRows, vmr_dev,max_l_rows,    &
                                       ONE, (umc_dev+ (lcs-1)*     &
                                           size_of_datatype),      &
                                       max_l_cols, gpuHandle)
#else
            call gpublas_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',                   &
                                       lce-lcs+1, n_cols, lre,     &
                                       ONE, (a_dev + (1-1+(lcs-1)*matrixRows* &
                                       size_of_datatype)),         &
                                       matrixRows, vmr_dev,max_l_rows,    &
                                       ONE, (umc_dev+ (lcs-1)*     &
                                           size_of_datatype),      &
                                       max_l_cols, gpuHandle)
#endif

            call obj%timer%stop("gpublas")

            if(i == 0) cycle
            call obj%timer%start("gpublas")

            lre = min(l_rows,i*l_rows_tile)
            gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
            if (isSkewsymmetric) then
              call gpublas_PRECISION_GEMM('N', 'N', lre,n_cols, lce-lcs+1, -ONE, &
                            (a_dev+ ((lcs-1)*matrixRows*                 &
                                  size_of_datatype)),             &
                       matrixRows, (umc_dev+(max_l_cols * n_cols+lcs-1)* &
                              size_of_datatype),              &
                              max_l_cols, ONE, (vmr_dev+(max_l_rows * n_cols)* &
                            size_of_datatype),              &
                              max_l_rows, gpuHandle)
            else
              call gpublas_PRECISION_GEMM('N', 'N', lre,n_cols, lce-lcs+1, ONE, &
                                          (a_dev+ ((lcs-1)*matrixRows*                 &
                                                size_of_datatype)),             &
                                     matrixRows, (umc_dev+(max_l_cols * n_cols+lcs-1)* &
                                            size_of_datatype),              &
                                            max_l_cols, ONE, (vmr_dev+(max_l_rows * n_cols)* &
                                          size_of_datatype),              &
                                            max_l_rows, gpuHandle)
            endif
            call obj%timer%stop("gpublas")
          else ! useGPU

            call obj%timer%start("blas")
            call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',       &
                                int(lce-lcs+1,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(lre,kind=BLAS_KIND), &
                                ONE, a_mat(1,lcs), int(matrixRows,kind=BLAS_KIND), &
                                vmrCPU, int(max_l_rows,kind=BLAS_KIND), ONE, umcCPU(lcs,1), &
                                int(max_l_cols,kind=BLAS_KIND) )
            call obj%timer%stop("blas")
            if (i == 0) cycle
            lre = min(l_rows,i*l_rows_tile)
            call obj%timer%start("blas")

            if (isSkewsymmetric) then
              call PRECISION_GEMM('N', 'N', int(lre,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(lce-lcs+1,kind=BLAS_KIND), &
                                  -ONE, a_mat(1,lcs), int(matrixRows,kind=BLAS_KIND),                                           &
                                  umcCPU(lcs,n_cols+1), int(max_l_cols,kind=BLAS_KIND), ONE,                          &
                                  vmrCPU(1,n_cols+1), int(max_l_rows, kind=BLAS_KIND) )

            else
              call PRECISION_GEMM('N', 'N', int(lre,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(lce-lcs+1,kind=BLAS_KIND), &
                                  ONE, a_mat(1,lcs), int(matrixRows,kind=BLAS_KIND),                                            &
                                  umcCPU(lcs,n_cols+1), int(max_l_cols,kind=BLAS_KIND), ONE,                          &
                                  vmrCPU(1,n_cols+1), int(max_l_rows, kind=BLAS_KIND) )
            endif
            call obj%timer%stop("blas")
          endif ! useGPU
        enddo ! i=0,(istep*nbw-1)/tile_size

        if (useGPU) then
          if (tile_size < istep*nbw .or. n_way > 1) then
#ifdef WITH_GPU_STREAMS
            successGPU = gpu_stream_synchronize(my_stream)
            check_stream_synchronize_gpu("bandred: vmr_dev", successGPU)

            successGPU = gpu_memcpy_async(int(loc(vmrGPU(1+max_l_rows*n_cols)),kind=c_intptr_t), &
                          vmr_dev+max_l_rows*n_cols*size_of_datatype, &
#ifndef WITH_OPENMP_TRADITIONAL
                          (vmr_size-max_l_rows*n_cols)*size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
#else
                          (max_l_rows*2*n_cols-max_l_rows*n_cols)*size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
#endif
            check_memcpy_gpu("bandred: vmr_dev -> vmrGPU", successGPU)

            successGPU = gpu_stream_synchronize(my_stream)
            check_stream_synchronize_gpu("bandred: vmr_dev -> vmrGPU", successGPU)
            ! synchronize streamsPerThread; maybe not neccessary
            successGPU = gpu_stream_synchronize()
            check_stream_synchronize_gpu("bandred: vmr_dev -> vmrGPU", successGPU)

#else /* WITH_GPU_STREAMS */
            successGPU = gpu_memcpy(int(loc(vmrGPU(1+max_l_rows*n_cols)),kind=c_intptr_t), &
                          vmr_dev+max_l_rows*n_cols*size_of_datatype, &
#ifndef WITH_OPENMP_TRADITIONAL
                          (vmr_size-max_l_rows*n_cols)*size_of_datatype, gpuMemcpyDeviceToHost)
#else
                          (max_l_rows*2*n_cols-max_l_rows*n_cols)*size_of_datatype, gpuMemcpyDeviceToHost)
#endif
            check_memcpy_gpu("bandred: vmr_dev -> vmrGPU", successGPU)
#endif /* WITH_GPU_STREAMS */
          endif

#ifdef WITH_GPU_STREAMS
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("bandred: umc_dev", successGPU)

          successGPU = gpu_memcpy_async(int(loc(umcGPU(1)),kind=c_intptr_t), &
                        umc_dev, l_cols*n_cols*size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("bandred: umc_dev -> umcGPU", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("bandred: umc_dev -> umcGPU", successGPU)
          ! synchronize streamsPerThread; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("bandred: umc_dev -> umcGPU", successGPU)
#else
          successGPU = gpu_memcpy(int(loc(umcGPU(1)),kind=c_intptr_t), &
                        umc_dev, l_cols*n_cols*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("bandred: umc_dev -> umcGPU", successGPU)
#endif
        endif ! useGPU
      endif ! l_cols>0 .and. l_rows>0

#ifdef WITH_OPENMP_TRADITIONAL
    endif ! n_way > 1
#endif
    ! Sum up all ur(:) parts along rows and add them to the uc(:) parts
    ! on the processors containing the diagonal
    ! This is only necessary if ur has been calculated, i.e. if the
    ! global tile size is smaller than the global remaining matrix

    ! Or if we used the Algorithm 4
    if (tile_size < istep*nbw .or. n_way > 1) then

      if (useGPU) then
        call elpa_reduce_add_vectors_&
             &MATH_DATATYPE&
             &_&
             &PRECISION &
                             (obj, vmrGPU(max_l_rows * n_cols + 1:),max_l_rows,  &
                              mpi_comm_rows, umcGPU,                            &
                              max_l_cols, mpi_comm_cols, istep*nbw, n_cols, nblk, max_threads_used)
      else ! useGPU
        call elpa_reduce_add_vectors_&
        &MATH_DATATYPE&
        &_&
        &PRECISION &
                                         (obj, vmrCPU(1,n_cols+1),max_l_rows,mpi_comm_rows, &
                                          umcCPU, max_l_cols, mpi_comm_cols, &
                                          istep*nbw, n_cols, nblk, max_threads_used)
      endif ! useGPU
    endif ! tile_size < istep*nbw .or. n_way > 1

    if (l_cols > 0) then

      if (useGPU) then
#ifdef WITH_MPI
        allocate(tmpGPU(l_cols * n_cols), stat=istat, errmsg=errorMessage)
        check_allocate("bandred: tmpGPU", istat, errorMessage)
        if (useNonBlockingCollectivesRows) then
          if (wantDebug) call obj%timer%start("mpi_nbc_communication")

          call mpi_iallreduce(umcGPU, tmpGPU, int(l_cols*n_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                         MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), allreduce_request5, mpierr)
          call mpi_wait(allreduce_request5, MPI_STATUS_IGNORE, mpierr)

          umcGPU(1 : l_cols * n_cols) = tmpGPU(1 : l_cols * n_cols)
          if (wantDebug) call obj%timer%stop("mpi_nbc_communication")
        else
          if (wantDebug) call obj%timer%start("mpi_communication")

          call mpi_allreduce(umcGPU, tmpGPU, int(l_cols*n_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                         MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
          umcGPU(1 : l_cols * n_cols) = tmpGPU(1 : l_cols * n_cols)
          if (wantDebug) call obj%timer%stop("mpi_communication")
        endif
#endif /* WITH_MPI */

        if (allocated(tmpGPU)) then
          deallocate(tmpGPU, stat=istat, errmsg=errorMessage)
          check_deallocate("bandred: tmpGPU", istat, errorMessage)
        endif

      else ! useGPU

#ifdef WITH_MPI
        if (useNonBlockingCollectivesRows) then
          if (wantDebug) call obj%timer%start("mpi_nbc_communication")
          call mpi_iallreduce(MPI_IN_PLACE, umcCPU, int(l_cols*n_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,    &
                           MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), allreduce_request6, mpierr)
          call mpi_wait(allreduce_request6, MPI_STATUS_IGNORE, mpierr)
          if (wantDebug) call obj%timer%stop("mpi_nbc_communication")
        else
          if (wantDebug) call obj%timer%start("mpi_communication")
          call mpi_allreduce(MPI_IN_PLACE, umcCPU, int(l_cols*n_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,    &
                           MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
          if (wantDebug) call obj%timer%stop("mpi_communication")
        endif
#endif /* WITH_MPI */

      endif ! useGPU
    endif ! l_cols > 0
    ! U = U * Tmat**T

    if (useGPU) then
#ifdef WITH_GPU_STREAMS
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("bandred: umcGPU", successGPU)

      successGPU = gpu_memcpy_async(umc_dev, int(loc(umcGPU(1)),kind=c_intptr_t), &
                    l_cols*n_cols*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
      check_memcpy_gpu("bandred: umcGPU -> umc_dev ", successGPU)

      successGPU = gpu_memcpy_async(tmat_dev,int(loc(tmat(1,1,istep)),kind=c_intptr_t), &
                    nbw*nbw*size_of_datatype,gpuMemcpyHostToDevice, my_stream)
      check_memcpy_gpu("bandred: tmat -> tmat_dev ", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("bandred: tmat -> tmat_dev ", successGPU)
      ! synchronize streamsPerThread; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      check_stream_synchronize_gpu("bandred: tmat -> tmat_dev ", successGPU)
#else
      successGPU = gpu_memcpy(umc_dev, int(loc(umcGPU(1)),kind=c_intptr_t), &
                    l_cols*n_cols*size_of_datatype, gpuMemcpyHostToDevice)
      check_memcpy_gpu("bandred: umcGPU -> umc_dev ", successGPU)

      successGPU = gpu_memcpy(tmat_dev,int(loc(tmat(1,1,istep)),kind=c_intptr_t), &
                    nbw*nbw*size_of_datatype,gpuMemcpyHostToDevice)
      check_memcpy_gpu("bandred: tmat -> tmat_dev ", successGPU)
#endif

      call obj%timer%start("gpublas")
      gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
      call gpublas_PRECISION_TRMM('Right', 'Upper', BLAS_TRANS_OR_CONJ, 'Nonunit',  &
                          l_cols, n_cols, ONE, tmat_dev, nbw, umc_dev, max_l_cols, gpuHandle)
      call obj%timer%stop("gpublas")

      ! VAV = Tmat * V**T * A * V * Tmat**T = (U*Tmat**T)**T * V * Tmat**T
      call obj%timer%start("gpublas")
      gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
      call gpublas_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',             &
                               n_cols, n_cols, l_cols, ONE, umc_dev, max_l_cols, &
                               (umc_dev+(max_l_cols * n_cols )*size_of_datatype),max_l_cols, &
                               ZERO, vav_dev, nbw, gpuHandle)

      call gpublas_PRECISION_TRMM('Right', 'Upper', BLAS_TRANS_OR_CONJ, 'Nonunit',    &
         n_cols, n_cols, ONE, tmat_dev, nbw, vav_dev, nbw, gpuHandle)
      call obj%timer%stop("gpublas")

#ifdef WITH_GPU_STREAMS
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("bandred: vav_dev", successGPU)

      successGPU = gpu_memcpy_async(int(loc(vav),kind=c_intptr_t), &
                  vav_dev, nbw*nbw*size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
      check_memcpy_gpu("bandred: vav_dev -> vav ", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("bandred: vav_dev -> vav ", successGPU)
      ! synchronize streamsPerThread; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      check_stream_synchronize_gpu("bandred: vav_dev -> vav ", successGPU)
#else
      successGPU = gpu_memcpy(int(loc(vav),kind=c_intptr_t), &
                  vav_dev, nbw*nbw*size_of_datatype, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("bandred: vav_dev -> vav ", successGPU)
#endif
    else ! useGPU

      call obj%timer%start("blas")

      call PRECISION_TRMM('Right', 'Upper', BLAS_TRANS_OR_CONJ, 'Nonunit',     &
                          int(l_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), ONE, tmat(1,1,istep), &
                          int(nbw,kind=BLAS_KIND), &
                          umcCPU, int(max_l_cols,kind=BLAS_KIND))

      ! VAV = Tmat * V**T * A * V * Tmat**T = (U*Tmat**T)**T * V * Tmat**T

      call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',              &
                          int(n_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), &
                          ONE, umcCPU, int(max_l_cols,kind=BLAS_KIND), umcCPU(1,n_cols+1), &
                          int(max_l_cols,kind=BLAs_KIND), ZERO, vav, int(nbw,kind=BLAS_KIND))

      call PRECISION_TRMM('Right', 'Upper', BLAS_TRANS_OR_CONJ, 'Nonunit',    &
                          int(n_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), ONE, tmat(1,1,istep),    &
                          int(nbw,kind=BLAS_KIND), vav, int(nbw,kind=BLAS_KIND) )
      call obj%timer%stop("blas")

    endif ! useGPU

#if REALCASE == 1
    if (isSkewsymmetric) then
      call ssymm_matrix_allreduce_&
      &PRECISION &
      (obj, n_cols,vav, nbw, nbw ,mpi_comm_cols, .false., success)
      if (.not.(success)) then
        write(error_unit,*) "Error when calling ssymm_matrix_allreduce"
        return
      endif
    else
      call symm_matrix_allreduce_&
      &PRECISION &
      (obj, n_cols,vav, nbw, nbw ,mpi_comm_cols, .false., success)
      if (.not.(success)) then
        write(error_unit,*) "Error when calling symm_matrix_allreduce"
        return
      endif
    endif
#endif /* REALCASE */
#if COMPLEXCASE == 1
    call herm_matrix_allreduce_&
         &PRECISION &
         (obj, n_cols,vav, nbw, nbw ,mpi_comm_cols, .false., success)
    if (.not.(success)) then
      write(error_unit,*) "Error when calling symm/herm_allreduce. Aborting..."
      return
    endif
#endif

    if (useGPU) then
#ifdef WITH_GPU_STREAMS
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("bandred: vav", successGPU)

      successGPU = gpu_memcpy_async(vav_dev, int(loc(vav),kind=c_intptr_t), &
                       nbw*nbw*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
      check_memcpy_gpu("bandred: vav -> vav_dev ", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("bandred: vav -> vav_dev ", successGPU)
      ! synchronize streamsPerThread; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      check_stream_synchronize_gpu("bandred: vav -> vav_dev ", successGPU)
#else
      successGPU = gpu_memcpy(vav_dev, int(loc(vav),kind=c_intptr_t), &
                       nbw*nbw*size_of_datatype, gpuMemcpyHostToDevice)
      check_memcpy_gpu("bandred: vav -> vav_dev ", successGPU)
#endif
    endif


    ! U = U - 0.5 * V * VAV

    if (useGPU) then
      call obj%timer%start("gpublas")
      gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
      if (isSkewsymmetric) then
        call gpublas_PRECISION_GEMM('N', 'N', l_cols, n_cols, n_cols,&
#if REALCASE == 1
                                  0.5_rk,                      &
#endif
#if COMPLEXCASE == 1
                                  (0.5_rk, 0.0_rk), &
#endif
                                  (umc_dev+(max_l_cols * n_cols )* &
                                  size_of_datatype),   &
                                  max_l_cols, vav_dev,nbw,        &
                                  ONE, umc_dev, max_l_cols, gpuHandle)
      else
        call gpublas_PRECISION_GEMM('N', 'N', l_cols, n_cols, n_cols,&
#if REALCASE == 1
                                 -0.5_rk,                      &
#endif
#if COMPLEXCASE == 1
                                 (-0.5_rk, 0.0_rk), &
#endif
                                 (umc_dev+(max_l_cols * n_cols )* &
                                 size_of_datatype),   &
                                 max_l_cols, vav_dev,nbw,        &
                                 ONE, umc_dev, max_l_cols, gpuHandle)
      endif
      call obj%timer%stop("gpublas")

#ifdef WITH_GPU_STREAMS
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("bandred: umc_dev", successGPU)

      successGPU = gpu_memcpy_async(int(loc(umcGPU(1)),kind=c_intptr_t), &
#ifndef WITH_OPENMP_TRADITIONAL
                  umc_dev, umc_size*size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
#else
                  umc_dev, max_l_cols*2*n_cols*size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
#endif
      check_memcpy_gpu("bandred: umc_dev -> umcGPU ", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("bandred: umc_dev -> umcGPU ", successGPU)
      ! synchronize streamsPerThread; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      check_stream_synchronize_gpu("bandred: umc_dev -> umcGPU ", successGPU)
#else /* WITH_GPU_STREAMS */
      successGPU = gpu_memcpy(int(loc(umcGPU(1)),kind=c_intptr_t), &
#ifndef WITH_OPENMP_TRADITIONAL
                  umc_dev, umc_size*size_of_datatype, gpuMemcpyDeviceToHost)
#else
                  umc_dev, max_l_cols*2*n_cols*size_of_datatype, gpuMemcpyDeviceToHost)
#endif
      check_memcpy_gpu("bandred: umc_dev -> umcGPU ", successGPU)
#endif /* WITH_GPU_STREAMS */

      ! Transpose umc -> umr (stored in vmr, second half)
      if (isSkewsymmetric) then
        call elpa_transpose_vectors_ss_&
           &MATH_DATATYPE&
           &_&
           &PRECISION &
                       (obj, umcGPU(:), max_l_cols, mpi_comm_cols, &
                        vmrGPU(max_l_rows * n_cols + 1:), max_l_rows, mpi_comm_rows, &
                        1, istep*nbw, n_cols, nblk, max_threads_used, .false., &
                        success)
        if (.not.(success)) then
          write(error_unit,*) "Error in elpa_transpose_vectors_ss. Aborting..."
          return
        endif
      else
        call elpa_transpose_vectors_&
           &MATH_DATATYPE&
           &_&
           &PRECISION &
                       (obj, umcGPU, max_l_cols, mpi_comm_cols, &
                        vmrGPU(max_l_rows * n_cols + 1:), max_l_rows, mpi_comm_rows, &
                        1, istep*nbw, n_cols, nblk, max_threads_used, .false., success)
        if (.not.(success)) then
          write(error_unit,*) "Error in elpa_transpose_vectors. Aborting..."
          return
        endif
      endif

#ifdef WITH_GPU_STREAMS
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("bandred: vmr", successGPU)

      successGPU = gpu_memcpy_async(vmr_dev+max_l_rows*n_cols*size_of_datatype, &
                  int(loc(vmrGPU(1+max_l_rows*n_cols)),kind=c_intptr_t), &
#ifndef WITH_OPENMP_TRADITIONAL
                  (vmr_size-max_l_rows*n_cols)*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
#else
                  (max_l_rows*2*n_cols-max_l_rows*n_cols)*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
#endif
      check_memcpy_gpu("bandred: vmr -> vmrGPU ", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("bandred: vmr -> vmrGPU ", successGPU)
      ! synchronize streamsPerThread; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      check_stream_synchronize_gpu("bandred: vmr -> vmrGPU ", successGPU)
#else /* WITH_GPU_STREAMS */
      successGPU = gpu_memcpy(vmr_dev+max_l_rows*n_cols*size_of_datatype, &
                  int(loc(vmrGPU(1+max_l_rows*n_cols)),kind=c_intptr_t), &
#ifndef WITH_OPENMP_TRADITIONAL
                  (vmr_size-max_l_rows*n_cols)*size_of_datatype, gpuMemcpyHostToDevice)
#else
                  (max_l_rows*2*n_cols-max_l_rows*n_cols)*size_of_datatype, gpuMemcpyHostToDevice)
#endif
      check_memcpy_gpu("bandred: vmr -> vmrGPU ", successGPU)
#endif /* WITH_GPU_STREAMS */
    else ! useGPU
      call obj%timer%start("blas")
#if REALCASE == 1
      if (isSkewsymmetric) then
        call PRECISION_GEMM('N', 'N', int(l_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND),     &
                            0.5_rk, umcCPU(1,n_cols+1), int(max_l_cols,kind=BLAS_KIND), vav,                        &
                            int(nbw,kind=BLAS_KIND), ONE, umcCPU, int(max_l_cols,kind=BLAS_KIND) )
      else
        call PRECISION_GEMM('N', 'N', int(l_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND),     &
                            -0.5_rk, umcCPU(1,n_cols+1), int(max_l_cols,kind=BLAS_KIND), vav,                       &
                            int(nbw,kind=BLAS_KIND), ONE, umcCPU, int(max_l_cols,kind=BLAS_KIND) )
      endif
#endif
#if COMPLEXCASE == 1
      call PRECISION_GEMM('N', 'N', int(l_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND),     &
                         (-0.5_rk, 0.0_rk),     &
                         umcCPU(1,n_cols+1), int(max_l_cols,kind=BLAS_KIND), vav, &
                         int(nbw,kind=BLAS_KIND), ONE, umcCPU, int(max_l_cols,kind=BLAS_KIND))
#endif

      call obj%timer%stop("blas")

      ! Transpose umc -> umr (stored in vmr, second half)
      if (isSkewsymmetric) then
        call elpa_transpose_vectors_ss_&
          &MATH_DATATYPE&
        &_&
        &PRECISION &
                                 (obj, umcCPU, max_l_cols, mpi_comm_cols, &
                                        vmrCPU(1,n_cols+1), max_l_rows, mpi_comm_rows, &
                                        1, istep*nbw, n_cols, nblk, max_threads_used, .false., &
                                        success)
          if (.not.(success)) then
            write(error_unit,*) "Error in elpa_transpose_vectors_ss. Aborting..."
            return
          endif
      else
       call elpa_transpose_vectors_&
       &MATH_DATATYPE&
       &_&
       &PRECISION &
                                (obj, umcCPU, max_l_cols, mpi_comm_cols, &
                                          vmrCPU(1,n_cols+1), max_l_rows, mpi_comm_rows, &
                                          1, istep*nbw, n_cols, nblk, max_threads_used, .false., &
                                          success)
          if (.not.(success)) then
            write(error_unit,*) "Error in elpa_transpose_vectors. Aborting..."
            return
          endif
      endif
    endif  ! useGPU

    ! A = A - V*U**T - U*V**T

#ifdef WITH_OPENMP_TRADITIONAL

!$omp parallel num_threads(max_threads_used) &
    !$omp default(none) &
    !$omp private( ii, i, lcs, lce, lre, n_way, m_way, m_id, n_id, work_per_thread, mystart, myend, myThreadID, &
    !$omp&         a_dev0, a_dev1, vmr_dev0, vmr_dev1, umc_dev0, umc_dev1, gpuHandle ) &
    !$omp shared(a_mat, n_threads, istep, tile_size, nbw, n_cols, obj, vmrcpu, l_cols_tile, l_rows, l_rows_tile, &
    !$omp&       gpuMemcpyDeviceToHost, gpuMemcpyHostToDevice, successGPU, max_threads_used, &
#ifndef WITH_OPENMP_TRADITIONAL
    !$omp&       umc_size, vmr_size, &
#endif
    !$omp&       matrixCols, &
    !$omp&       umccpu, l_cols, a_dev, vmr_dev, useGPU, max_l_rows, umc_dev, max_l_cols, matrixRows)
    n_threads  = omp_get_num_threads()
    myThreadID = omp_get_thread_num()
    if (useGPU) then
      successGPU = gpu_setdevice(obj%gpu_setup%gpuDeviceArray(myThreadID))
    endif

    if (mod(n_threads, 2) == 0) then
      n_way = 2
    else
      n_way = 1
    endif

    m_way = n_threads / n_way

    m_id = mod(myThreadID,  m_way)
    n_id = myThreadID / m_way


    do ii=n_id*tile_size,(istep*nbw-1),tile_size*n_way
      i = ii / tile_size
      lcs = i*l_cols_tile+1
      lce = min(l_cols,(i+1)*l_cols_tile)
      lre = min(l_rows,(i+1)*l_rows_tile)
      if (lce < lcs .or. lre < 1) cycle

      !Figure out this thread's range
      work_per_thread = lre / m_way
      if (work_per_thread * m_way < lre) work_per_thread = work_per_thread + 1
      mystart = m_id * work_per_thread + 1
      myend   = mystart + work_per_thread - 1
      if ( myend > lre ) myend = lre
      if ( myend-mystart+1 < 1) cycle
      if (useGPU) then
        call obj%timer%start("gpublas")
        gpuHandle = obj%gpu_setup%gpublasHandleArray(myThreadID)
        ! original
        !call gpublas_PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, myend-mystart+1,    &
        !                         lce-lcs+1, 2*n_cols, -ONE, &
        !                         vmr_dev, max_l_rows, (umc_dev +(lcs-1)*  &
        !                         size_of_datatype), &
        !                         max_l_cols, ONE, (a_dev+(lcs-1)*matrixRows* &
        !                         size_of_datatype), matrixRows)

        ! correct indices

        call gpublas_PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, myend-mystart+1,    &
                                 lce-lcs+1, 2*n_cols, -ONE, &
                                 vmr_dev+(mystart-1+(1-1)*max_l_rows)*size_of_datatype, max_l_rows, umc_dev +(lcs-1+  &
                                 (1-1)*max_l_cols)*size_of_datatype, &
                                 max_l_cols, ONE, a_dev+(mystart-1+(lcs-1)*matrixRows)* &
                                 size_of_datatype, matrixRows, gpuHandle)

        call obj%timer%stop("gpublas")
      else ! useGPU
        call obj%timer%start("blas")
        call PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, int(myend-mystart+1,kind=BLAS_KIND), &
                            int(lce-lcs+1,kind=BLAS_KIND), int(2*n_cols,kind=BLAS_KIND), -ONE, &
                            vmrCPU(mystart, 1), int(max_l_rows,kind=BLAS_KIND), &
                            umcCPU(lcs,1), int(max_l_cols,kind=BLAS_KIND), &
                            ONE, a_mat(mystart,lcs), int(matrixRows,kind=BLAS_KIND) )
        call obj%timer%stop("blas")
      endif ! useGPU
    enddo
    !$omp end parallel
#else /* WITH_OPENMP_TRADITIONAL */

    do i=0,(istep*nbw-1)/tile_size
      lcs = i*l_cols_tile+1
      lce = min(l_cols,(i+1)*l_cols_tile)
      lre = min(l_rows,(i+1)*l_rows_tile)
      if (lce<lcs .or. lre<1) cycle

      if (useGPU) then
        call obj%timer%start("gpublas")

        gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
        call gpublas_PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ,     &
                                   lre, lce-lcs+1, 2*n_cols, -ONE, &
                                   vmr_dev, max_l_rows, (umc_dev +(lcs-1)*  &
                                   size_of_datatype), &
                                   max_l_cols, ONE, (a_dev+(lcs-1)*matrixRows* &
                                   size_of_datatype), matrixRows, gpuHandle)
        call obj%timer%stop("gpublas")
      else ! useGPU

        call obj%timer%start("blas")
        call PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, int(lre,kind=BLAS_KIND),int(lce-lcs+1,kind=BLAS_KIND), &
                            int(2*n_cols,kind=BLAS_KIND), &
                            -ONE, &
                            vmrCPU, int(max_l_rows,kind=BLAS_KIND), umcCPU(lcs,1), &
                            int(max_l_cols,kind=BLAS_KIND), &
                            ONE, a_mat(1,lcs), int(matrixRows,kind=BLAS_KIND))
        call obj%timer%stop("blas")
     endif ! useGPU
    enddo ! i=0,(istep*nbw-1)/tile_size
#endif /* WITH_OPENMP_TRADITIONAL */

    if (.not.(useGPU)) then
      if (allocated(vr)) then
        deallocate(vr, stat=istat, errmsg=errorMessage)
        check_deallocate("bandred: vr", istat, errorMessage)
      endif

      if (allocated(umcCPU)) then
        deallocate(umcCPU, stat=istat, errmsg=errorMessage)
        check_deallocate("bandred: umcCPU", istat, errorMessage)
      endif

      if (allocated(vmrCPU)) then
        deallocate(vmrCPU, stat=istat, errmsg=errorMessage)
        check_deallocate("bandred: vmrCPU", istat, errorMessage)
      endif
    endif !useGPU

#ifdef WITH_OPENMP_TRADITIONAL
    if (allocated(vr)) then
      deallocate(vr, stat=istat, errmsg=errorMessage)
      check_deallocate("bandred: vr", istat, errorMessage)
    endif

    if (useGPU) then

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
        if (associated(umcGPU_2d)) then
          nullify(umcGPU_2d)
        endif

        if (associated(vmrGPU_2d)) then
          nullify(vmrGPU_2d)
        endif
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      else
        deallocate(umcGPU_2d)
        deallocate(vmrGPU_2d)
      endif

      if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
        if (associated(umcGPU)) then
          nullify(umcGPU)

          successGPU = gpu_free_host(umc_host)
          check_host_dealloc_gpu("bandred: umc_host ", successGPU)

          successGPU = gpu_free(umc_dev)
          check_dealloc_gpu("bandred: umc_dev ", successGPU)
        endif

        if (associated(vmrGPU)) then
          nullify(vmrGPU)

          successGPU = gpu_free_host(vmr_host)
          check_host_dealloc_gpu("bandred: vmr_host ", successGPU)

          successGPU = gpu_free(vmr_dev)
          check_dealloc_gpu("bandred: vmr_dev ", successGPU)
        endif
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
      else 
        if (associated(umcGPU)) then
          deallocate(umcGPU)

          successGPU = gpu_free(umc_dev)
          check_dealloc_gpu("bandred: umc_dev ", successGPU)
        endif
        if (associated(vmrGPU)) then
          deallocate(vmrGPU)

          successGPU = gpu_free(vmr_dev)
          check_dealloc_gpu("bandred: vmr_dev ", successGPU)
        endif
      endif
#endif

    endif
#endif

 enddo ! istep - loop

  if (useGPU) then

    ! copy a_dev to a_mat
    ! we do it here, since a is needed on the host in the following routine
    ! (band to tridi). Previously, a has been kept on the device and then
    ! copied in redist_band (called from tridiag_band). However, it seems to
    ! be easier to do it here.
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("bandred: a_dev", successGPU)

    successGPU = gpu_memcpy_async(int(loc(a_mat),kind=c_intptr_t), &
                  int(a_dev,kind=c_intptr_t), &
                  int(matrixRows*matrixCols* size_of_datatype, kind=c_intptr_t), &
                  gpuMemcpyDeviceToHost, my_stream)
    check_memcpy_gpu("bandred: a_dev -> a_mat ", successGPU)

    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("bandred: a_dev -> a_mat ", successGPU)
    ! synchronize streamsPerThread; maybe not neccessary
    successGPU = gpu_stream_synchronize()
    check_stream_synchronize_gpu("bandred: a_dev -> a_mat ", successGPU)
#else
    successGPU = gpu_memcpy(int(loc(a_mat),kind=c_intptr_t), &
                  int(a_dev,kind=c_intptr_t), &
                  int(matrixRows*matrixCols* size_of_datatype, kind=c_intptr_t), &
                  gpuMemcpyDeviceToHost)
    check_memcpy_gpu("bandred: a_dev -> a_mat ", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      successGPU = gpu_host_unregister(int(loc(a_mat),kind=c_intptr_t))
      check_host_unregister_gpu("bandred: a_mat ", successGPU)
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    endif
#endif

#ifdef WITH_GPU_STREAMS
    successGPU = gpu_host_unregister(int(loc(tmat),kind=c_intptr_t))
    check_host_unregister_gpu("bandred: tmat ", successGPU)
#endif

    successGPU = gpu_free(a_dev)
    check_dealloc_gpu("bandred: a_dev ", successGPU)

    successGPU = gpu_free(vav_dev)
    check_dealloc_gpu("bandred: vav_dev ", successGPU)

    successGPU = gpu_free(tmat_dev)
    check_dealloc_gpu("bandred: tmat_dev ", successGPU)

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      successGPU = gpu_host_unregister(int(loc(vav),kind=c_intptr_t))
      check_host_unregister_gpu("bandred: vav", successGPU)
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    endif
#endif

#ifndef WITH_OPENMP_TRADITIONAL
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      if (associated(umcGPU)) then
        nullify(umcGPU)

        successGPU = gpu_free_host(umc_host)
        check_host_dealloc_gpu("bandred: umc_host ", successGPU)
        successGPU = gpu_free(umc_dev)
        check_dealloc_gpu("bandred: umc_dev ", successGPU)
      endif

      if (associated(vmrGPU)) then
        nullify(vmrGPU)

        successGPU = gpu_free_host(vmr_host)
        check_host_dealloc_gpu("bandred: vmr_host ", successGPU)

        successGPU = gpu_free(vmr_dev)
        check_dealloc_gpu("bandred: vmr_dev ", successGPU)
      endif
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    else
      if (associated(umcGPU)) then
        deallocate(umcGPU)
        successGPU = gpu_free(umc_dev)
        check_dealloc_gpu("bandred: umc_dev ", successGPU)
      endif
      if (associated(vmrGPU)) then
        deallocate(vmrGPU)
        successGPU = gpu_free(vmr_dev)
        check_dealloc_gpu("bandred: vmr_dev ", successGPU)
      endif
    endif
#endif

#endif /* WITH_OPENMP_TRADITIONAL */
  endif ! useGPU
  
#ifndef WITH_OPENMP_TRADITIONAL
  if (allocated(vr)) then
    deallocate(vr, stat=istat, errmsg=errorMessage)
    check_deallocate("bandred: vr", istat, errorMessage)
  endif
#endif

  if (allocated(umcCPU)) then
    deallocate(umcCPU, stat=istat, errmsg=errorMessage)
    check_deallocate("bandred: umcCPU", istat, errorMessage)
  endif

  if (allocated(vmrCPU)) then
    deallocate(vmrCPU, stat=istat, errmsg=errorMessage)
    check_deallocate("bandred: vmrCPU", istat, errorMessage)
  endif

#if REALCASE == 1
  if (useQR) then
    if (which_qr_decomposition == 1) then
      deallocate(work_blocked, stat=istat, errmsg=errorMessage)
      check_deallocate("bandred: work_blocked", istat, errorMessage)

      deallocate(tauvector, stat=istat, errmsg=errorMessage)
      check_deallocate("bandred: tauvector", istat, errorMessage)
    endif
  endif
#endif
  
  call obj%timer%stop("bandred_&
  &MATH_DATATYPE&
  &" // &
  &PRECISION_SUFFIX //&
  gpuString)

contains
  subroutine get_hh_vec(vec_in,vr,tau,vrl)
    MATH_DATATYPE(kind=rck):: vr(:), vec_in(:), tau, vrl
    MATH_DATATYPE(kind=rck):: aux1(2), xf
    real(kind=rk):: vnorm2
    ! Get Vector to be transformed; distribute last element and norm of
    ! remaining elements to all procs in current column
    
    if (my_prow==prow(nrow, nblk, np_rows)) then
       aux1(1) = dot_product(vec_in(1:lr-1),vec_in(1:lr-1))
       aux1(2) = vec_in(lr)
    else
       aux1(1) = dot_product(vec_in(1:lr),vec_in(1:lr))
       aux1(2) = 0.0_rck
    endif

#ifdef WITH_MPI
    if (useNonBlockingCollectivesRows) then
       if (wantDebug) call obj%timer%start("mpi_nbc_communication")
       call mpi_iallreduce(MPI_IN_PLACE, aux1, 2_MPI_KIND, MPI_MATH_DATATYPE_PRECISION, &
            MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), &
            allreduce_request1, mpierr)
       
       call mpi_wait(allreduce_request1, MPI_STATUS_IGNORE, mpierr)
       
       if (wantDebug) call obj%timer%stop("mpi_nbc_communication")
    else
       !             if (wantDebug)             call obj%timer%start("mpi_comm")
       call mpi_allreduce(MPI_IN_PLACE, aux1, 2_MPI_KIND, MPI_MATH_DATATYPE_PRECISION, &
            MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), &
            mpierr)
       
       !            if (wantDebug)            call obj%timer%stop("mpi_comm")
    endif
    
#endif

#if REALCASE == 1
    vnorm2 = aux1(1)
#endif
#if COMPLEXCASE == 1
    vnorm2 = real(aux1(1),kind=rk)
#endif
    vrl    = aux1(2)

    ! Householder transformation
    call hh_transform_&
         &MATH_DATATYPE&
         &_&
         &PRECISION &
         (obj, vrl, vnorm2, xf, tau, wantDebug)
    ! Scale vr and store Householder Vector for back transformation

    vr(1:lr) = vec_in(1:lr) * xf
    if (my_prow==prow(nrow, nblk, np_rows)) vr(lr) = 1.0_rck

  end subroutine get_hh_vec
  

  subroutine apply_ht(tau,vr,ex_buff2d)
    MATH_DATATYPE(kind=rck):: tau, vr(:), ex_buff2d(:,:)
    MATH_DATATYPE(kind=rck):: tauc
    MATH_DATATYPE(kind=rck):: aux1(nbw)
    integer:: nlc, imax
    logical:: use_blas

    imax=ubound(ex_buff2d,2)
    
    if((imax.lt.3).or.(max_threads.gt.1)) then
       !don't use BLAS for very small imax because overhead is too high
       !don't use BLAS with OpenMP because measurements showed that threading is not effective for these routines
       use_blas=.false.
    else
       use_blas=.true.
    end if
    
    !we need to transform the remaining ex_buff
    if (lr>0) then
       if(use_blas) then !note that aux1 is conjg between > and < thresh_blas!!
          call PRECISION_GEMV(BLAS_TRANS_OR_CONJ,int(lr,kind=BLAS_KIND),int(imax,kind=BLAS_KIND), &
               ONE, ex_buff2d, size(ex_buff2d,1,kind=BLAS_KIND), vr, 1_BLAS_KIND, ZERO, aux1, &
               1_BLAS_KIND)
       else
#ifdef WITH_OPENMP_TRADITIONAL
          !$omp  parallel do private(nlc)
#endif
          do nlc=1,imax
             aux1(nlc) = dot_product(vr(1:lr),ex_buff2d(1:lr,nlc))
          end do
#ifdef WITH_OPENMP_TRADITIONAL
          !$omp end parallel do
#endif         
       end if
    else
       aux1(1:imax) = 0.
    end if

    ! Get global dot products
#ifdef WITH_MPI
    if (useNonBlockingCollectivesRows) then
       if (wantDebug) call obj%timer%start("mpi_nbc_communication")
       if (imax > 0) then
          call mpi_iallreduce(MPI_IN_PLACE, aux1, int(imax,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
               MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), &
               allreduce_request3, mpierr)
          call mpi_wait(allreduce_request3, MPI_STATUS_IGNORE, mpierr)
       endif
       if (wantDebug) call obj%timer%stop("mpi_nbc_communication")
    else
       if (wantDebug) call obj%timer%start("mpi_communication")
       if (imax>0) then
          call mpi_allreduce(MPI_IN_PLACE, aux1, int(imax,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
               MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), &
               mpierr)
       endif
       if (wantDebug) call obj%timer%stop("mpi_communication")
    endif
#endif /* WITH_MPI */

    if(lr.le.0) return !no data on this processor

    ! Transform
#if REALCASE == 1
    tauc=-tau
#else
    tauc=-conjg(tau)
#endif 
    if(use_blas) then
       call PRECISION_GERC(int(lr,kind=BLAS_KIND),int(imax,kind=BLAS_KIND),tauc,vr,1_BLAS_KIND,&
            aux1,1_BLAS_KIND,ex_buff2d,ubound(ex_buff2d,1,kind=BLAS_KIND))
    else
#ifdef WITH_OPENMP_TRADITIONAL 
       !$omp  parallel do private(nlc)
#endif
       do nlc=1,imax         
          ex_buff2d(1:lr,nlc) = ex_buff2d(1:lr,nlc) + tauc*aux1(nlc)*vr(1:lr)
       end do
#ifdef WITH_OPENMP_TRADITIONAL 
       !$omp end parallel do
#endif       
    end if

  end subroutine apply_ht
  
end subroutine bandred_&
&MATH_DATATYPE&
&_&
&PRECISION

