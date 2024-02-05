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
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
#endif

#include "../general/sanity.F90"

#ifdef WITH_CUDA_AWARE_MPI
#define CUDA_AWARE_MPI_BAND_TO_FULL
#else
#undef CUDA_AWARE_MPI_BAND_TO_FULL
#endif

#ifdef CUDA_AWARE_MPI_BAND_TO_FULL
#define MORE_GPUBLAS
#else
#undef MORE_GPUBLAS
#endif

subroutine trans_ev_band_to_full_&
    &MATH_DATATYPE&
    &_&
    &PRECISION &
    (obj, na, nqc, nblk, nbw, a_mat, lda, tmat, q_mat, &
     ldq, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols, useGPU, &
#if REALCASE == 1
     useQr, success)
#endif
#if COMPLEXCASE == 1
     success)
#endif

!-------------------------------------------------------------------------------
!  trans_ev_band_to_full_real/complex:
!  Transforms the eigenvectors of a band matrix back to the eigenvectors of the original matrix
!
!  Parameters
!
!  na          Order of matrix a_mat, number of rows of matrix q_mat
!
!  nqc         Number of columns of matrix q_mat
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  nbw         semi bandwith
!
!  a_mat(lda,matrixCols)    Matrix containing the Householder vectors (i.e. matrix a_mat after bandred_real/complex)
!              Distribution is like in Scalapack.
!
!  lda         Leading dimension of a_mat
!  matrixCols  local columns of matrix a_mat and q_mat
!
!  tmat(nbw,nbw,numBlocks) Factors returned by bandred_real/complex
!
!  q_mat           On input: Eigenvectors of band matrix
!              On output: Transformed eigenvectors
!              Distribution is like in Scalapack.
!
!  ldq         Leading dimension of q_mat
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!
!-------------------------------------------------------------------------------
  use precision
  use elpa_gpu
  use, intrinsic :: iso_c_binding
  use elpa_abstract_impl
  use elpa_blas_interfaces

  implicit none
#include "../general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout)     :: obj
  logical, intent(in)                            :: useGPU
#if REALCASE == 1
  logical, intent(in)                            :: useQR
#endif
  integer(kind=ik)                               :: na, nqc, lda, ldq, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, &
                                                    mpi_comm_cols
#ifdef USE_ASSUMED_SIZE
  MATH_DATATYPE(kind=rck)                        :: a_mat(lda,*)
  MATH_DATATYPE(kind=rck)                        :: q_mat(ldq,*), tmat(nbw,nbw,*)
#else
  MATH_DATATYPE(kind=rck)                        :: a_mat(lda,matrixCols)
  MATH_DATATYPE(kind=rck)                        :: q_mat(ldq,matrixCols), tmat(nbw, nbw, numBlocks)
#endif

  integer(kind=ik)                               :: my_prow, my_pcol, np_rows, np_cols
  integer(kind=MPI_KIND)                         :: my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI, mpierr
  integer(kind=ik)                               :: max_blocks_row, max_blocks_col, max_local_rows, &
                                                    max_local_cols
  integer(kind=ik)                               :: l_cols, l_rows, l_colh, n_cols
  integer(kind=ik)                               :: istep, lc, ncol, nrow, nb, ns

  MATH_DATATYPE(kind=rck), allocatable           :: hvb(:)
  MATH_DATATYPE(kind=rck), pointer               :: hvm(:,:), tmp1(:), tmp2(:)
  MATH_DATATYPE(kind=rck), pointer               :: tmp_debug(:)
  ! hvm_dev is fist used and set in this routine
  ! q_mat is changed in trans_ev_tridi on the host, copied to device and passed here. this can be adapted
  ! tmp_dev is first used in this routine
  ! tmat_complete_dev is not passed along from bandred_real
  integer(kind=C_intptr_T)                       :: hvm_dev, q_dev, tmp_dev, tmat_complete_dev, dev_offset
#ifdef MORE_GPUBLAS
  integer(kind=C_intptr_T)                       :: t_tmp_dev
#endif

  type(c_ptr)                                    :: hvm_host, tmp1_host, tmp2_host

#ifdef MORE_GPUBLAS
  type(c_ptr)                                    :: t_tmp_gpu_dev
  MATH_DATATYPE(kind=rck), pointer               :: t_tmp_gpu_deviceptr(:)

  type(c_ptr)                                    :: tmat_gpu_dev
  MATH_DATATYPE(kind=rck), pointer               :: tmat_gpu_deviceptr(:,:)

  type(c_ptr)                                    :: hvm_gpu_dev
  MATH_DATATYPE(kind=rck), pointer               :: hvm_gpu_deviceptr(:,:)
#endif

#ifdef CUDA_AWARE_MPI_BAND_TO_FULL
  integer(kind=c_intptr_t)                       :: t_tmp2_dev
  type(c_ptr)                                    :: t_tmp2_gpu_dev
  MATH_DATATYPE(kind=rck), pointer               :: t_tmp2_gpu_deviceptr(:)

  type(c_ptr)                                    :: tmp1_mpi_dev, tmp2_mpi_dev
  MATH_DATATYPE(kind=rck), pointer               :: tmp1_mpi_deviceptr(:), tmp2_mpi_deviceptr(:)
  integer(kind=c_intptr_t)                       :: tmp2_dev
#endif

  integer(kind=ik)                               :: i

  MATH_DATATYPE(kind=rck), allocatable, target   :: tmat_complete(:,:), t_tmp(:,:), t_tmp2(:,:)
  integer(kind=ik)                               :: t_cols, t_rows, ii, jj
  integer(kind=ik)                               :: cwy_blocking

  integer(kind=ik)                               :: istat
  character(200)                                 :: errorMessage
  character(20)                                  :: gpuString
  logical                                        :: successGPU
  integer(kind=c_intptr_t), parameter            :: size_of_datatype = size_of_&
                                                                       &PRECISION&
                                                                       &_&
                                                                       &MATH_DATATYPE
  integer(kind=ik)                               :: blocking_factor, error, blk_end
  integer(kind=MPI_KIND)                         :: bcast_request1, allreduce_request1, allreduce_request2
  logical                                        :: useNonBlockingCollectivesCols
  logical                                        :: useNonBlockingCollectivesRows
  integer(kind=c_int)                            :: non_blocking_collectives_rows, non_blocking_collectives_cols
  logical                                        :: success
  integer(kind=MPI_KIND), allocatable            :: ibreq(:)
  integer(kind=ik)                               :: nblocks, bc_counter
  integer(kind=c_intptr_t)                       :: gpuHandle, my_stream

  success = .true.

  if(useGPU) then
    gpuString = "_gpu"
  else
    gpuString = ""
  endif

  call obj%timer%start("trans_ev_band_to_full_&
  &MATH_DATATYPE&
  &" // &
  &PRECISION_SUFFIX //&
  gpuString)

  call obj%get("nbc_row_elpa2_band_to_full", non_blocking_collectives_rows, error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "Problem setting option for non blocking collectives for rows in elpa2_band_to_full. Aborting..."
    call obj%timer%stop("trans_ev_band_to_full_&
    &MATH_DATATYPE&
    &" // &
    &PRECISION_SUFFIX //&
    gpuString)
    success = .false.
    return
  endif

  call obj%get("nbc_col_elpa2_band_to_full", non_blocking_collectives_cols, error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "Problem setting option for non blocking collectives for cols in elpa2_band_to_full. Aborting..."
    call obj%timer%stop("trans_ev_band_to_full_&
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

#ifdef BAND_TO_FULL_BLOCKING
  call obj%get("blocking_in_band_to_full",blocking_factor,error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "Problem getting option for blocking_in_band_to_full. Aborting..."
    call obj%timer%stop("trans_ev_band_to_full_&
    &MATH_DATATYPE&
    &" // &
    &PRECISION_SUFFIX //&
    gpuString)
    success = .false.
    return
  endif
#else
  blocking_factor = 1
#endif


  my_prow = obj%mpi_setup%myRank_comm_rows
  my_pcol = obj%mpi_setup%myRank_comm_cols

  np_rows = obj%mpi_setup%nRanks_comm_rows
  np_cols = obj%mpi_setup%nRanks_comm_cols

  !call obj%timer%start("mpi_communication")
  !call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND) ,my_prowMPI ,mpierr)
  !call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND) ,np_rowsMPI ,mpierr)
  !call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND) ,my_pcolMPI ,mpierr)
  !call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND) ,np_colsMPI ,mpierr)

  !my_prow = int(my_prowMPI,kind=c_int)
  !my_pcol = int(my_pcolMPI,kind=c_int)
  !np_rows = int(np_rowsMPI,kind=c_int)
  !np_cols = int(np_colsMPI,kind=c_int)
  !call obj%timer%stop("mpi_communication")

  max_blocks_row = ((na -1)/nblk)/np_rows + 1 ! Rows of a_mat
  max_blocks_col = ((nqc-1)/nblk)/np_cols + 1 ! Columns of q_mat!

  max_local_rows = max_blocks_row*nblk
  max_local_cols = max_blocks_col*nblk

  cwy_blocking = blocking_factor * nbw

  if (useGPU) then
    ! copy q_mat to q_dev
    successGPU = gpu_malloc(q_dev,ldq*matrixCols*size_of_datatype)
    check_alloc_gpu("trans_ev_band_to_full: q_dev", successGPU)
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      successGPU = gpu_host_register(int(loc(q_mat),kind=c_intptr_t),&
                    ldq*matrixCols*size_of_datatype, gpuHostRegisterDefault)
      check_host_register_gpu("trans_ev_band_to_full: q_mat", successGPU)
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    endif
#endif

#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("trans_ev_band_to_full: q_mat -> q_dev", successGPU)

    successGPU = gpu_memcpy_async(q_dev,int(loc(q_mat),kind=c_intptr_t),&
                  ldq*matrixCols*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
    check_memcpy_gpu("trans_ev_band_to_full: q_mat -> q_dev", successGPU)

    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("trans_ev_band_to_full: q_mat -> q_dev", successGPU)
    ! synchronize streamPerThread; maybe not neccessary
    successGPU = gpu_stream_synchronize()
    check_stream_synchronize_gpu("trans_ev_band_to_full: q_mat -> q_dev", successGPU)
#else
    successGPU = gpu_memcpy(q_dev,int(loc(q_mat),kind=c_intptr_t),&
                  ldq*matrixCols*size_of_datatype, gpuMemcpyHostToDevice)
    check_memcpy_gpu("trans_ev_band_to_full: q_mat -> q_dev", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      successGPU = gpu_malloc_host(tmp1_host,max_local_cols*cwy_blocking*size_of_datatype)
      check_host_alloc_gpu("trans_ev_band_to_full: tmp1_host", successGPU)
      call c_f_pointer(tmp1_host, tmp1, (/max_local_cols*cwy_blocking/))

      successGPU = gpu_malloc_host(tmp2_host,max_local_cols*cwy_blocking*size_of_datatype)
      check_host_alloc_gpu("trans_ev_band_to_full: tmp2_host", successGPU)
      call c_f_pointer(tmp2_host, tmp2, (/max_local_cols*cwy_blocking/))

      successGPU = gpu_malloc_host(hvm_host,max_local_rows*cwy_blocking*size_of_datatype)
      check_host_alloc_gpu("trans_ev_band_to_full: hvm_host", successGPU)
      call c_f_pointer(hvm_host, hvm, (/max_local_rows,cwy_blocking/))
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    else
      allocate(tmp1(max_local_cols*cwy_blocking))
      allocate(tmp2(max_local_cols*cwy_blocking))
      allocate(hvm(max_local_rows,cwy_blocking))
    endif
#endif
  else ! useGPU
    allocate(tmp1(max_local_cols*cwy_blocking), stat=istat, errmsg=errorMessage)
    check_allocate("trans_ev_band_to_full: tmp1", istat, errorMessage)

    allocate(tmp2(max_local_cols*cwy_blocking), stat=istat, errmsg=errorMessage)
    check_allocate("trans_ev_band_to_full: tmp2", istat, errorMessage)

    allocate(hvm(max_local_rows,cwy_blocking), stat=istat, errmsg=errorMessage)
    check_allocate("trans_ev_band_to_full: hvm", istat, errorMessage)
  endif !useGPU

  allocate(hvb(max_local_rows*cwy_blocking), stat=istat, errmsg=errorMessage)
  check_allocate("trans_ev_band_to_full: hvb", istat, errorMessage)

  allocate(tmat_complete(cwy_blocking,cwy_blocking), stat=istat, errmsg=errorMessage)
  check_allocate("trans_ev_band_to_full: tmat_complete", istat, errorMessage)

  if (useGPU) then
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      successGPU = gpu_host_register(int(loc(tmat_complete),kind=c_intptr_t), &
                    cwy_blocking * cwy_blocking * size_of_datatype,&
                    gpuHostRegisterDefault)
      check_host_register_gpu("trans_ev_band_to_full: tmat_complete", successGPU)
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    endif
#endif
  endif

#ifdef MORE_GPUBLAS
  ! could be used alwasy if beneficial
  if (useGPU) then
    if (blocking_factor > 1) then
      successGPU = gpu_malloc(t_tmp_dev,cwy_blocking*nbw*size_of_datatype)
      check_alloc_gpu("trans_ev_band_to_full: t_tmp_dev", successGPU)

#ifdef CUDA_AWARE_MPI_BAND_TO_FULL
      successGPU = gpu_malloc(t_tmp2_dev,cwy_blocking*nbw*size_of_datatype)
      check_alloc_gpu("trans_ev_band_to_full: t_tmp2_dev", successGPU)
#endif

    endif
  endif
#endif /* MORE_GPUBLAS */

  if (blocking_factor > 1) then
    allocate(t_tmp(cwy_blocking,nbw), stat=istat, errmsg=errorMessage)
    check_allocate("trans_ev_band_to_full: t_tmp", istat, errorMessage)

    allocate(t_tmp2(cwy_blocking,nbw), stat=istat, errmsg=errorMessage)
    check_allocate("trans_ev_band_to_full: t_tmp2", istat, errorMessage)

#ifdef WITH_GPU_STREAMS
    if (useGPU) then
      successGPU = gpu_host_register(int(loc(t_tmp),kind=c_intptr_t),&
                    cwy_blocking*nbw*size_of_datatype, gpuHostRegisterDefault)
      check_host_register_gpu("trans_ev_band_to_full: t_tmp", successGPU)

      successGPU = gpu_host_register(int(loc(t_tmp2),kind=c_intptr_t),&
                    cwy_blocking*nbw*size_of_datatype, gpuHostRegisterDefault)
      check_host_register_gpu("trans_ev_band_to_full: t_tmp2", successGPU)
    endif
#endif
  endif

  if (useGPU) then
    successGPU = gpu_malloc(hvm_dev,max_local_rows*cwy_blocking*size_of_datatype)
    check_alloc_gpu("trans_ev_band_to_full: hvm_dev", successGPU)

    successGPU = gpu_malloc(tmp_dev,max_local_cols*cwy_blocking*size_of_datatype)
    check_alloc_gpu("trans_ev_band_to_full: tmp_dev", successGPU)

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif

#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp_dev", successGPU)

      successGPU = gpu_memset_async(tmp_dev, 0, max_local_cols*cwy_blocking*size_of_datatype, my_stream)
      check_memset_gpu("trans_ev_band_to_full: tmp_dev", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp_dev", successGPU)
#else
      successGPU = gpu_memset(tmp_dev, 0, max_local_cols*cwy_blocking*size_of_datatype)
      check_memset_gpu("trans_ev_band_to_full: tmp_dev", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    else
      allocate(tmp_debug(max_local_cols*cwy_blocking))
      tmp_debug(:) = 0.
      successGPU = gpu_memcpy(tmp_dev, int(loc(tmp_debug),kind=c_intptr_t), &
                             max_local_cols*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice)
      check_memcpy_gpu("trans_ev_band_to_full: tmp_debug -> tmp_dev", successGPU)
   endif
#endif

#ifdef CUDA_AWARE_MPI_BAND_TO_FULL
    successGPU = gpu_malloc(tmp2_dev,max_local_cols*cwy_blocking*size_of_datatype)
    check_alloc_gpu("trans_ev_band_to_full: tmp2_dev", successGPU)
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif

#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp2_dev", successGPU)

      successGPU = gpu_memset_async(tmp2_dev, 0, max_local_cols*cwy_blocking*size_of_datatype, my_stream)
      check_memset_gpu("trans_ev_band_to_full: tmp2_dev", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp2_dev", successGPU)
#else
      successGPU = gpu_memset(tmp2_dev, 0, max_local_cols*cwy_blocking*size_of_datatype)
      check_memset_gpu("trans_ev_band_to_full: tmp2_dev", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    else
     allocate(tmp_debug(max_local_cols*cwy_blocking))
     tmp_debug(:) = 0.
     successGPU = gpu_memcpy(tmp2_dev, int(loc(tmp_debug),kind=c_intptr_t), &
                             max_local_cols*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice)
     check_memcpy_gpu("trans_ev_band_to_full: tmp_debug -> tmp_dev", successGPU)
   endif
#endif
#endif /* CUDA_AWARE_MPI_BAND_TO_FULL */

    successGPU = gpu_malloc(tmat_complete_dev,cwy_blocking*cwy_blocking*size_of_datatype)
    check_alloc_gpu("trans_ev_band_to_full: tmat_complete_dev", successGPU)
  endif


  hvm = 0.0_rck ! Must be set to 0 !!!
  hvb = 0.0_rck ! Safety only
  tmp1 = 0.0_rck
  tmp2 = 0.0_rck
  tmat_complete = 0.0_rck
  if (blocking_factor > 1) then
     t_tmp = 0.0_rck ! Must be set to 0 !!!
     t_tmp2 = 0.0_rck

#ifdef MORE_GPUBLAS
     ! could be used always if beneficial
     if (useGPU) then
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
       if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif

#ifdef WITH_GPU_STREAMS
         my_stream = obj%gpu_setup%my_stream
         successGPU = gpu_stream_synchronize(my_stream)
         check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp_dev", successGPU)

         successGPU = gpu_memset_async(t_tmp_dev, 0, cwy_blocking*nbw*size_of_datatype, my_stream)
         check_memset_gpu("trans_ev_band_to_full: t_tmp_dev", successGPU)

         successGPU = gpu_stream_synchronize(my_stream)
         check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp_dev", successGPU)
#else
         successGPU = gpu_memset(t_tmp_dev, 0, cwy_blocking*nbw*size_of_datatype)
         check_memset_gpu("trans_ev_band_to_full: t_tmp_dev", successGPU)
#endif

#ifdef CUDA_AWARE_MPI_BAND_TO_FULL
#ifdef WITH_GPU_STREAMS
         my_stream = obj%gpu_setup%my_stream
         successGPU = gpu_stream_synchronize(my_stream)
         check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp2_dev", successGPU)

         successGPU = gpu_memset_async(t_tmp2_dev, 0, cwy_blocking*nbw*size_of_datatype, my_stream)
         check_memset_gpu("trans_ev_band_to_full: t_tmp2_dev", successGPU)

         successGPU = gpu_stream_synchronize(my_stream)
         check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp2_dev", successGPU)
#else
         successGPU = gpu_memset(t_tmp2_dev, 0, cwy_blocking*nbw*size_of_datatype)
         check_memset_gpu("trans_ev_band_to_full: t_tmp2_dev", successGPU)

#endif
#endif /* CUDA_AWARE_MPI_BAND_TO_FULL */
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
       else
         allocate(tmp_debug(nbw*cwy_blocking))
         tmp_debug(:) = 0.
         successGPU = gpu_memcpy(t_tmp_dev, int(loc(tmp_debug),kind=c_intptr_t), &
                                 nbw*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice)
         check_memcpy_gpu("trans_ev_band_to_full: tmp_debug -> t_tmp_dev", successGPU)
#ifdef CUDA_AWARE_MPI_BAND_TO_FULL
         successGPU = gpu_memcpy(t_tmp2_dev, int(loc(tmp_debug),kind=c_intptr_t), &
                                 nbw*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice)
         check_memcpy_gpu("trans_ev_band_to_full: tmp_debug -> t_tmp_dev", successGPU)
#endif
         deallocate(tmp_debug)
       endif
#endif /* defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION) */
     endif ! useGPU
#endif /* MORE_GPUBLAS */
  endif
  l_cols = local_index(nqc, my_pcol, np_cols, nblk, -1) ! Local columns of q_mat

  blk_end = ((na-1)/nbw-1)/blocking_factor + 1
  do istep=1, blk_end

    ! This the call when using na >= ((blocking_factor+1)*nbw)
    ! n_cols = MIN(na,istep*cwy_blocking+nbw) - (istep-1)*cwy_blocking - nbw
    ! Number of columns in current step
    ! As an alternative we add some special case handling if na < cwy_blocking
    if (na < cwy_blocking) then
      n_cols = MAX(0, na-nbw)
      if ( n_cols .eq. 0 ) then
        exit
      end if
    else
      n_cols = MIN(na,istep*cwy_blocking+nbw) - (istep-1)*cwy_blocking - nbw ! Number of columns in current step
    end if

    ! Broadcast all Householder vectors for current step compressed in hvb

    nb = 0
    ns = 0
    bc_counter=0
    nblocks = n_cols/nblk;
    allocate(ibreq(0:nblocks-1))

    do lc = 1, n_cols
      ncol = (istep-1)*cwy_blocking + nbw + lc ! absolute column number of householder Vector
      nrow = ncol - nbw ! absolute number of pivot row

      l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast
      l_colh = local_index(ncol , my_pcol, np_cols, nblk, -1) ! HV local column number

      if (my_pcol==pcol(ncol, nblk, np_cols)) hvb(nb+1:nb+l_rows) = a_mat(1:l_rows,l_colh)

      nb = nb+l_rows

      if (lc==n_cols .or. mod(ncol,nblk)==0) then
#ifdef WITH_MPI
        if (useNonBlockingCollectivesCols) then
          call mpi_ibcast(hvb(ns+1), int(nb-ns,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,&
                         int(pcol(ncol, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), &
                         ibreq(bc_counter), mpierr)
          bc_counter = bc_counter + 1  
        else
          call obj%timer%start("mpi_communication")
          call mpi_bcast(hvb(ns+1), int(nb-ns,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,&
                         int(pcol(ncol, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), &
                         mpierr)
          call obj%timer%stop("mpi_communication")
        endif
#endif /* WITH_MPI */
        ns = nb
      endif
    enddo ! lc

#ifdef  WITH_MPI
  if(useNonBlockingCollectivesCols) then
    call obj%timer%start("mpi_nbc_communication")
    call mpi_waitall(nblocks, ibreq, MPI_STATUSES_IGNORE, mpierr)
    call obj%timer%stop("mpi_nbc_communication")
  endif
#endif
  deallocate(ibreq)

    ! Expand compressed Householder vectors into matrix hvm

    nb = 0
    do lc = 1, n_cols
      nrow = (istep-1)*cwy_blocking + lc ! absolute number of pivot row
      l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast

      ! could maybe also done on GPU
      hvm(1:l_rows,lc) = hvb(nb+1:nb+l_rows)
      if (my_prow==prow(nrow, nblk, np_rows)) hvm(l_rows+1,lc) = 1.0_rck
      nb = nb+l_rows
    enddo

    l_rows = local_index(MIN(na,(istep+1)*cwy_blocking), my_prow, np_rows, nblk, -1)

    ! compute tmat2 out of tmat(:,:,)
    tmat_complete = 0
    do i = 1, blocking_factor
      t_cols = MIN(nbw, n_cols - (i-1)*nbw)
      if (t_cols <= 0) exit
      t_rows = (i - 1) * nbw
      tmat_complete(t_rows+1:t_rows+t_cols,t_rows+1:t_rows+t_cols) = tmat(1:t_cols,1:t_cols,(istep-1)*blocking_factor + i)

      if (i > 1) then
        if (useGPU) then
#ifdef MORE_GPUBLAS

#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: hvm -> hvm_dev", successGPU)

          successGPU = gpu_memcpy_async(hvm_dev, int(loc(hvm),kind=c_intptr_t), &
                          max_local_rows*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("trans_ev_band_to_full: hvm -> hvm_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: hvm -> hvm_dev", successGPU)
          ! synchronize streamPerThread; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("trans_ev_band_to_full: hvm -> hvm_dev", successGPU)
#else
          successGPU = gpu_memcpy(hvm_dev, int(loc(hvm),kind=c_intptr_t), &
                          max_local_rows*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("trans_ev_band_to_full: hvm -> hvm_dev", successGPU)
#endif


          !create a fortran pointer and use this offset
          hvm_gpu_dev = transfer(hvm_dev, hvm_gpu_dev)
          call c_f_pointer(hvm_gpu_dev,hvm_gpu_deviceptr, [max_local_rows,cwy_blocking])

          call obj%timer%start("gpublas")
          gpuHandle = obj%gpu_setup%gpublasHandleArray(0)

          call gpublas_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', &
                                       t_rows, t_cols, l_rows, ONE, hvm_dev, max_local_rows, &
                                       c_loc(hvm_gpu_deviceptr(:,(i-1)*nbw+1:)), max_local_rows , ZERO, t_tmp_dev, &
                                       cwy_blocking, gpuHandle)


          call obj%timer%stop("gpublas")

          t_tmp_gpu_dev = transfer(t_tmp_dev, t_tmp_gpu_dev)
          call c_f_pointer(t_tmp_gpu_dev,t_tmp_gpu_deviceptr,(/(cwy_blocking*nbw)/))

#ifdef CUDA_AWARE_MPI_BAND_TO_FULL
          t_tmp2_gpu_dev = transfer(t_tmp2_dev, t_tmp2_gpu_dev)
          call c_f_pointer(t_tmp2_gpu_dev,t_tmp2_gpu_deviceptr,(/(cwy_blocking*nbw)/))
#endif

#else /* MORE_GPUBLAS */
          call obj%timer%start("blas")
          call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', &
                            int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), ONE, hvm, &
                            int(max_local_rows,kind=BLAS_KIND), hvm(:,(i-1)*nbw+1:), &
                            int(max_local_rows,kind=BLAS_KIND), ZERO, t_tmp, int(cwy_blocking, kind=BLAS_KIND))
          call obj%timer%stop("blas")
#endif /* MORE_GPUBLAS */
        else ! useGPU
          call obj%timer%start("blas")
          call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', &
                            int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), ONE, hvm, &
                            int(max_local_rows,kind=BLAS_KIND), hvm(:,(i-1)*nbw+1:), &
                            int(max_local_rows,kind=BLAS_KIND), ZERO, t_tmp, int(cwy_blocking, kind=BLAS_KIND))
          call obj%timer%stop("blas")
        endif ! useGPU


#ifdef WITH_MPI
        if (useNonBlockingCollectivesRows) then
#ifndef CUDA_AWARE_MPI_BAND_TO_FULL

#ifdef MORE_GPUBLAS
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp", successGPU)

          successGPU = gpu_memcpy_async(int(loc(t_tmp),kind=c_intptr_t), &
                                  t_tmp_dev, cwy_blocking*nbw*size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp", successGPU)
          ! synchronize streamPerThread; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp", successGPU)
#else
          successGPU = gpu_memcpy(int(loc(t_tmp),kind=c_intptr_t), &
                                  t_tmp_dev, cwy_blocking*nbw*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp", successGPU)
#endif
#endif /* MORE_GPUBLAS */
          call obj%timer%start("mpi_nbc_communication")
          call mpi_iallreduce(t_tmp, t_tmp2, int(cwy_blocking*nbw,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                         MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), allreduce_request1, mpierr)
          call mpi_wait(allreduce_request1, MPI_STATUS_IGNORE, mpierr)
          call obj%timer%stop("mpi_nbc_communication")

#ifdef MORE_GPUBLAS
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp -> t_tmp_dev", successGPU)

          successGPU = gpu_memcpy_async(t_tmp_dev, int(loc(t_tmp2),kind=c_intptr_t), &
                                  cwy_blocking*nbw*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("trans_ev_band_to_full: t_tmp -> t_tmp_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp -> t_tmp_dev", successGPU)
          ! synchronize streamPerThread; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp -> t_tmp_dev", successGPU)
#else
          successGPU = gpu_memcpy(t_tmp_dev, int(loc(t_tmp2),kind=c_intptr_t), &
                                  cwy_blocking*nbw*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("trans_ev_band_to_full: t_tmp -> t_tmp_dev", successGPU)
#endif
#endif /* MORE_GPUBLAS */

#else /* CUDA_AWARE_MPI_BAND_TO_FULL */
          call obj%timer%start("cuda_mpi_nbc_communication")
          call mpi_iallreduce(t_tmp_gpu_deviceptr, t_tmp2_gpu_deviceptr, int(cwy_blocking*nbw,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                           MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), allreduce_request1, mpierr)
          call mpi_wait(allreduce_request1, MPI_STATUS_IGNORE, mpierr)
          call obj%timer%stop("cuda_mpi_nbc_communication")
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp2_dev -> t_tmp_dev", successGPU)

          successGPU = gpu_memcpy_async(t_tmp_dev, t_tmp2_dev, &
                                  cwy_blocking*nbw*size_of_datatype, gpuMemcpyDeviceToDevice, my_stream)
          check_memcpy_gpu("trans_ev_band_to_full: t_tmp2_dev -> t_tmp_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp2_dev -> t_tmp_dev", successGPU)
          ! synchronize streamPerThread; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp2_dev -> t_tmp_dev", successGPU)
#else
          successGPU = gpu_memcpy(t_tmp_dev, t_tmp2_dev, &
                                  cwy_blocking*nbw*size_of_datatype, gpuMemcpyDeviceToDevice)
          check_memcpy_gpu("trans_ev_band_to_full: t_tmp2_dev -> t_tmp_dev", successGPU)
#endif
#endif /* CUDA_AWARE_MPI_BAND_TO_FULL */
        else ! useNonBlockingCollectivesRows
#ifndef CUDA_AWARE_MPI_BAND_TO_FULL

#ifdef MORE_GPUBLAS
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp", successGPU)

          successGPU = gpu_memcpy_async(int(loc(t_tmp),kind=c_intptr_t), &
                                  t_tmp_dev, cwy_blocking*nbw*size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp", successGPU)
          ! synchronize streamPerThread; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp", successGPU)
#else
          successGPU = gpu_memcpy(int(loc(t_tmp),kind=c_intptr_t), &
                                  t_tmp_dev, cwy_blocking*nbw*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp", successGPU)
#endif
#endif /* MORE_GPUBLAS */
          call obj%timer%start("mpi_communication")
          call mpi_allreduce(t_tmp, t_tmp2, int(cwy_blocking*nbw,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                           MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
          call obj%timer%stop("mpi_communication")

#ifdef MORE_GPUBLAS
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp -> t_tmp_dev", successGPU)

          successGPU = gpu_memcpy_async(t_tmp_dev, int(loc(t_tmp2),kind=c_intptr_t), &
                                  cwy_blocking*nbw*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("trans_ev_band_to_full: t_tmp -> t_tmp_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp -> t_tmp_dev", successGPU)
          ! synchronize streamPerThread; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp -> t_tmp_dev", successGPU)
#else
          successGPU = gpu_memcpy(t_tmp_dev, int(loc(t_tmp2),kind=c_intptr_t), &
                                  cwy_blocking*nbw*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("trans_ev_band_to_full: t_tmp -> t_tmp_dev", successGPU)
#endif
#endif /* MORE_GPUBLAS */

#else /* CUDA_AWARE_MPI_BAND_TO_FULL */
          call obj%timer%start("cuda_mpi_communication")
          call mpi_allreduce(t_tmp_gpu_deviceptr, t_tmp2_gpu_deviceptr, int(cwy_blocking*nbw,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                           MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
          call obj%timer%stop("cuda_mpi_communication")
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp2_dev -> t_tmp_dev", successGPU)

          successGPU = gpu_memcpy_async(t_tmp_dev, t_tmp2_dev, &
                                  cwy_blocking*nbw*size_of_datatype, gpuMemcpyDeviceToDevice, my_stream)
          check_memcpy_gpu("trans_ev_band_to_full: t_tmp2_dev -> t_tmp_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp2_dev -> t_tmp_dev", successGPU)
          ! synchronize streamPerThread; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp2_dev -> t_tmp_dev", successGPU)
#else
          successGPU = gpu_memcpy(t_tmp_dev, t_tmp2_dev, &
                                  cwy_blocking*nbw*size_of_datatype, gpuMemcpyDeviceToDevice)
          check_memcpy_gpu("trans_ev_band_to_full: t_tmp2_dev -> t_tmp_dev", successGPU)
#endif
#endif /* CUDA_AWARE_MPI_BAND_TO_FULL */
        endif ! useNonBlockingCollectivesRows

        if (useGPU) then
#ifdef MORE_GPUBLAS
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)

          successGPU = gpu_memcpy_async(tmat_complete_dev, int(loc(tmat_complete),kind=c_intptr_t), &
                          cwy_blocking*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)
          ! synchronize streamPerThread; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)
#else
          successGPU = gpu_memcpy(tmat_complete_dev, int(loc(tmat_complete),kind=c_intptr_t), &
                          cwy_blocking*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("trans_ev_band_to_full: tmat -> tmat_complete_dev", successGPU)
#endif

          call obj%timer%start("gpublas")
          gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
          call gpublas_PRECISION_TRMM('L', 'U', 'N', 'N', &
                                   t_rows, t_cols, ONE, tmat_complete_dev, cwy_blocking, t_tmp_dev, cwy_blocking, gpuHandle)

          t_tmp_gpu_dev = transfer(t_tmp_dev, t_tmp_gpu_dev)
          tmat_gpu_dev = transfer(tmat_complete_dev, tmat_gpu_dev)
          call c_f_pointer(tmat_gpu_dev,tmat_gpu_deviceptr, [cwy_blocking,cwy_blocking])
          call c_f_pointer(t_tmp_gpu_dev,t_tmp_gpu_deviceptr, [cwy_blocking*nbw])

          gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
          call gpublas_PRECISION_TRMM('R', 'U', 'N', 'N', &
                                   t_rows, t_cols, -ONE, c_loc(tmat_gpu_deviceptr(t_rows+1,t_rows+1)), cwy_blocking, &
                                   t_tmp_gpu_dev, cwy_blocking, gpuHandle)
          call obj%timer%stop("gpublas")

#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp2", successGPU)

          successGPU = gpu_memcpy_async(int(loc(t_tmp2),kind=c_intptr_t), t_tmp_dev, &
                          cwy_blocking*nbw*size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp2", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp2", successGPU)
          ! synchronize streamPerThread; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp2", successGPU)
#else
          successGPU = gpu_memcpy(int(loc(t_tmp2),kind=c_intptr_t), t_tmp_dev, &
                          cwy_blocking*nbw*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp2", successGPU)
#endif
          tmat_complete(1:t_rows,t_rows+1:t_rows+t_cols) = t_tmp2(1:t_rows,1:t_cols)

#else /* MORE_GPUBLAS */
          call obj%timer%start("blas")
          call PRECISION_TRMM('L', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), ONE, tmat_complete, &
                            int(cwy_blocking,kind=BLAS_KIND), t_tmp2, int(cwy_blocking,kind=BLAS_KIND))
          call PRECISION_TRMM('R', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), -ONE, &
                            tmat_complete(t_rows+1,t_rows+1), &
                            int(cwy_blocking,kind=BLAS_KIND), t_tmp2, int(cwy_blocking,kind=BLAS_KIND))
          call obj%timer%stop("blas")
          tmat_complete(1:t_rows,t_rows+1:t_rows+t_cols) = t_tmp2(1:t_rows,1:t_cols)
            
#endif /* MORE_GPUBLAS */

        else ! useGPU
          call obj%timer%start("blas")
          call PRECISION_TRMM('L', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), ONE, tmat_complete, &
                            int(cwy_blocking,kind=BLAS_KIND), t_tmp2, int(cwy_blocking,kind=BLAS_KIND))
          call PRECISION_TRMM('R', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), -ONE, &
                            tmat_complete(t_rows+1,t_rows+1), &
                            int(cwy_blocking,kind=BLAS_KIND), t_tmp2, int(cwy_blocking,kind=BLAS_KIND))
          call obj%timer%stop("blas")
          tmat_complete(1:t_rows,t_rows+1:t_rows+t_cols) = t_tmp2(1:t_rows,1:t_cols)
        endif ! useGPU

#else /* WITH_MPI */
        if (useGPU) then
          ! remove cuda_aware section here, does not make sense without MPI add MORE_GPUBLAS instead
#ifdef MORE_GPUBLAS

#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)

          successGPU = gpu_memcpy_async(tmat_complete_dev, int(loc(tmat_complete),kind=c_intptr_t), &
                          cwy_blocking*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)
          ! synchronize streamPerThread; maybe not neccessary
          successGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)
#else
          successGPU = gpu_memcpy(tmat_complete_dev, int(loc(tmat_complete),kind=c_intptr_t), &
                          cwy_blocking*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)
#endif

          call obj%timer%start("gpublas")
          gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
          call gpublas_PRECISION_TRMM('L', 'U', 'N', 'N', &
                                   t_rows, t_cols, ONE, tmat_complete_dev, cwy_blocking, t_tmp_dev, cwy_blocking, gpuHandle)

          t_tmp_gpu_dev = transfer(t_tmp_dev, t_tmp_gpu_dev)
          tmat_gpu_dev = transfer(tmat_complete_dev, tmat_gpu_dev)
          call c_f_pointer(tmat_gpu_dev,tmat_gpu_deviceptr, [cwy_blocking,cwy_blocking])
          call c_f_pointer(t_tmp_gpu_dev,t_tmp_gpu_deviceptr, [cwy_blocking*nbw])

          gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
          call gpublas_PRECISION_TRMM('R', 'U', 'N', 'N', &
                                   t_rows, t_cols, -ONE, c_loc(tmat_gpu_deviceptr(t_rows+1,t_rows+1)), cwy_blocking, &
                                   t_tmp_gpu_dev, cwy_blocking, gpuHandle)
          call obj%timer%stop("gpublas")
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          succcessGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp2", successGPU)

          successGPU = gpu_memcpy_async(int(loc(t_tmp),kind=c_intptr_t), t_tmp_dev, &
                          cwy_blocking*nbw*size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
          check_memcpy_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp2", successGPU)

          succcessGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp2", successGPU)
          ! synchronize streamPerThread; maybe not neccessary
          succcessGPU = gpu_stream_synchronize()
          check_stream_synchronize_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp2", successGPU)
#else
          successGPU = gpu_memcpy(int(loc(t_tmp),kind=c_intptr_t), t_tmp_dev, &
                          cwy_blocking*nbw*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("trans_ev_band_to_full: t_tmp_dev -> t_tmp2", successGPU)
#endif
          tmat_complete(1:t_rows,t_rows+1:t_rows+t_cols) = t_tmp(1:t_rows,1:t_cols)

#else /* MORE_GPUBLAS */
          call obj%timer%start("blas")
          call PRECISION_TRMM('L', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), ONE, tmat_complete, &
                            int(cwy_blocking,kind=BLAS_KIND), t_tmp, int(cwy_blocking,kind=BLAS_KIND))
          call PRECISION_TRMM('R', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), -ONE, &
                              tmat_complete(t_rows+1,t_rows+1), &
                              int(cwy_blocking,kind=BLAS_KIND), t_tmp, int(cwy_blocking,kind=BLAS_KIND))
          call obj%timer%stop("blas")
          tmat_complete(1:t_rows,t_rows+1:t_rows+t_cols) = t_tmp(1:t_rows,1:t_cols)
#endif /* MORE_GPUBLAS */
        else !useGPU
          call obj%timer%start("blas")
          call PRECISION_TRMM('L', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), ONE, tmat_complete, &
                            int(cwy_blocking,kind=BLAS_KIND), t_tmp, int(cwy_blocking,kind=BLAS_KIND))
          call PRECISION_TRMM('R', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), -ONE, &
                              tmat_complete(t_rows+1,t_rows+1), &
                              int(cwy_blocking,kind=BLAS_KIND), t_tmp, int(cwy_blocking,kind=BLAS_KIND))
          call obj%timer%stop("blas")
          tmat_complete(1:t_rows,t_rows+1:t_rows+t_cols) = t_tmp(1:t_rows,1:t_cols)
        endif !useGPU

#endif /* WITH_MPI */

      endif
    enddo

    ! Q = Q - V * T**T * V**T * Q

    if (l_rows > 0) then
      if (useGPU) then
#ifndef MORE_GPUBAS
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("trans_ev_band_to_full: hvm -> hvm_dev", successGPU)

        successGPU = gpu_memcpy_async(hvm_dev, int(loc(hvm),kind=c_intptr_t), &
                        max_local_rows*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
        check_memcpy_gpu("trans_ev_band_to_full: hvm -> hvm_dev", successGPU)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("trans_ev_band_to_full: hvm -> hvm_dev", successGPU)
        ! synchronize streamPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu("trans_ev_band_to_full: hvm -> hvm_dev", successGPU)
#else
        successGPU = gpu_memcpy(hvm_dev, int(loc(hvm),kind=c_intptr_t), &
                        max_local_rows*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice)
        check_memcpy_gpu("trans_ev_band_to_full: hvm -> hvm_dev", successGPU)
#endif
#endif /* MORE_GPUBAS */
        call obj%timer%start("gpublas")
        gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
        call gpublas_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', &
                                     n_cols, l_cols, l_rows, ONE, hvm_dev, max_local_rows, &
                                     q_dev, ldq , ZERO, tmp_dev, n_cols, gpuHandle)
        call obj%timer%stop("gpublas")

#ifdef WITH_MPI
#ifndef CUDA_AWARE_MPI_BAND_TO_FULL
        ! copy data from device to host for a later MPI_ALLREDUCE
#ifdef WITH_GPU_STREAMS  
        my_stream = obj%gpu_setup%my_stream
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)

        successGPU = gpu_memcpy_async(int(loc(tmp1),kind=c_intptr_t), &
                      tmp_dev, l_cols*n_cols*size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
        check_memcpy_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)
        ! synchronize streamPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)

#else
        successGPU = gpu_memcpy(int(loc(tmp1),kind=c_intptr_t), &
                      tmp_dev, l_cols*n_cols*size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)
#endif
#else /* CUDA_AWARE_MPI_BAND_TO_FULL */
        tmp1_mpi_dev = transfer(tmp_dev, tmp1_mpi_dev)
        call c_f_pointer(tmp1_mpi_dev,tmp1_mpi_deviceptr,(/(l_cols*n_cols)/))

#endif /* CUDA_AWARE_MPI_BAND_TO_FULL */
#endif /* WITH_MPI */
      else ! useGPU
        call obj%timer%start("blas")
        call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', &
                            int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), ONE, &
                            hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), q_mat, int(ldq,kind=BLAS_KIND), ZERO, tmp1, &
                           int(n_cols,kind=BLAS_KIND))
        call obj%timer%stop("blas")
      endif ! useGPU
    else ! l_rows>0
#ifdef MORE_GPUBLAS
      if (useGPU) then
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
        if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif

#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: tmp_dev", successGPU)

          successGPU = gpu_memset_async(tmp_dev, 0, l_cols*n_cols*size_of_datatype, my_stream)
          check_memset_gpu("trans_ev_band_to_full: tmp_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("trans_ev_band_to_full: tmp_dev", successGPU)
#else
          successGPU = gpu_memset(tmp_dev, 0, l_cols*n_cols*size_of_datatype)
          check_memset_gpu("trans_ev_band_to_full: tmp_dev", successGPU)
#endif

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
        else
          allocate(tmp_debug(l_cols*n_cols))
          tmp_debug(:) = 0.
          successGPU = gpu_memcpy(tmp_dev, int(loc(tmp_debug),kind=c_intptr_t), &
                                 l_cols*n_cols*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("trans_ev_band_to_full: tmp_debug -> tmp_dev", successGPU)
          deallocate(tmp_debug)
       endif
#endif /* defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION) */
      else ! useGPU
#endif /* MORE_GPUBLAS */
        tmp1(1:l_cols*n_cols) = 0.0_rck
#ifdef MORE_GPUBLAS
      endif ! useGPU
#endif
    endif ! l_rows>0

#ifdef WITH_MPI
    if (useGPU) then
#ifndef MORE_GPUBLAS
#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)

      successGPU = gpu_memcpy_async(int(loc(tmp1),kind=c_intptr_t), &
                   tmp_dev, l_cols*n_cols*size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
      check_memcpy_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)
      ! synchronize streamPerThread; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)
#else
      successGPU = gpu_memcpy(int(loc(tmp1),kind=c_intptr_t), &
                   tmp_dev, l_cols*n_cols*size_of_datatype, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)
#endif

#else /* MORE_GPUBLAS */
#ifdef CUDA_AWARE_MPI_BAND_TO_FULL
      tmp1_mpi_dev = transfer(tmp_dev, tmp1_mpi_dev)
      call c_f_pointer(tmp1_mpi_dev,tmp1_mpi_deviceptr,(/(l_cols*n_cols)/))
      tmp2_mpi_dev = transfer(tmp2_dev, tmp2_mpi_dev)
      call c_f_pointer(tmp2_mpi_dev,tmp2_mpi_deviceptr,(/(l_cols*n_cols)/))
#endif

#endif /* MORE_GPUBLAS */
    endif
#endif /* WITH_MPI */

#ifdef WITH_MPI
    if (useNonBlockingCollectivesRows) then
#ifndef CUDA_AWARE_MPI_BAND_TO_FULL

#ifdef MORE_GPUBLAS
#ifdef WITH_GPU_STREAMS  
      my_stream = obj%gpu_setup%my_stream
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)

      successGPU = gpu_memcpy_async(int(loc(tmp1),kind=c_intptr_t), &
                              tmp_dev, l_cols*n_cols*size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
      check_memcpy_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)
      ! synchronize streamPerThread; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)
#else
      successGPU = gpu_memcpy(int(loc(tmp1),kind=c_intptr_t), &
                              tmp_dev, l_cols*n_cols*size_of_datatype, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)
#endif
#endif /* MORE_GPUBLAS */
      call obj%timer%start("mpi_nbc_communication")
      call mpi_iallreduce(tmp1, tmp2, int(n_cols*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                       int(mpi_comm_rows,kind=MPI_KIND), allreduce_request2, mpierr)
      call mpi_wait(allreduce_request2, MPI_STATUS_IGNORE, mpierr)
      call obj%timer%stop("mpi_nbc_communication")

#ifdef MORE_GPUBLAS
#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp2 -> tmp_dev", successGPU)

      successGPU = gpu_memcpy_async(tmp_dev, int(loc(tmp2),kind=c_intptr_t), &
                              l_cols*n_cols*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
      check_memcpy_gpu("trans_ev_band_to_full: tmp2 -> tmp_dev", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp2 -> tmp_dev", successGPU)
      ! synchronize streamPerThread; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp2 -> tmp_dev", successGPU)
#else
      successGPU = gpu_memcpy(tmp_dev, int(loc(tmp2),kind=c_intptr_t), &
                              l_cols*n_cols*size_of_datatype, gpuMemcpyHostToDevice)
      check_memcpy_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)
#endif
#endif /* MORE_GPUBLAS */


#else /* CUDA_AWARE_MPI_BAND_TO_FULL */
      call obj%timer%start("cuda_mpi_nbc_communication")
      call mpi_iallreduce(tmp1_mpi_deviceptr, tmp2_mpi_deviceptr, int(n_cols*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                          MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), allreduce_request2, mpierr)
      call mpi_wait(allreduce_request2, MPI_STATUS_IGNORE, mpierr)
      call obj%timer%stop("cuda_mpi_nbc_communication")
#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp2_dev -> tmp_dev", successGPU)

      successGPU = gpu_memcpy_async(tmp1_mpi_dev, tmp2_mpi_dev, &
                              l_cols*n_cols*size_of_datatype, gpuMemcpyDeviceToDevice, my_stream)
      check_memcpy_gpu("trans_ev_band_to_full: tmp2_dev -> tmp_dev", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp2_dev -> tmp_dev", successGPU)
      ! synchronize streamPerThread; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp2_dev -> tmp_dev", successGPU)

#else
      successGPU = gpu_memcpy(tmp1_mpi_dev, tmp2_mpi_dev, &
                              l_cols*n_cols*size_of_datatype, gpuMemcpyDeviceToDevice)
      check_memcpy_gpu("trans_ev_band_to_full: tmp2_dev -> tmp_dev", successGPU)
#endif
#endif /* CUDA_AWARE_MPI_BAND_TO_FULL */
    else
#ifndef CUDA_AWARE_MPI_BAND_TO_FULL

#ifdef MORE_GPUBLAS
#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)

      successGPU = gpu_memcpy_async(int(loc(tmp1),kind=c_intptr_t), &
                              tmp_dev, l_cols*n_cols*size_of_datatype, gpuMemcpyDeviceToHost, my_stream)
      check_memcpy_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)
      ! synchronize streamPerThread; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)
#else
      successGPU = gpu_memcpy(int(loc(tmp1),kind=c_intptr_t), &
                              tmp_dev, l_cols*n_cols*size_of_datatype, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)
#endif
#endif /* MORE_GPUBLAS */

      call obj%timer%start("mpi_communication")
      call mpi_allreduce(tmp1, tmp2, int(n_cols*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                       int(mpi_comm_rows,kind=MPI_KIND), mpierr)
      call obj%timer%stop("mpi_communication")

#ifdef MORE_GPUBLAS
#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp2 -> tmp_dev", successGPU)

      successGPU = gpu_memcpy_async(tmp_dev, int(loc(tmp2),kind=c_intptr_t), &
                              l_cols*n_cols*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
      check_memcpy_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp2 -> tmp_dev", successGPU)
      ! synchronize streamPerThread; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp2 -> tmp_dev", successGPU)
#else
      successGPU = gpu_memcpy(tmp_dev, int(loc(tmp2),kind=c_intptr_t), &
                              l_cols*n_cols*size_of_datatype, gpuMemcpyHostToDevice)
      check_memcpy_gpu("trans_ev_band_to_full: tmp_dev -> tmp2", successGPU)
#endif
#endif /* MORE_GPUBLAS */

#else /* CUDA_AWARE_MPI_BAND_TO_FULL */
      call obj%timer%start("cuda_mpi_communication")
      call mpi_allreduce(tmp1_mpi_deviceptr, tmp2_mpi_deviceptr, int(n_cols*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                          MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
      call obj%timer%stop("cuda_mpi_communication")
#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp2_dev -> tmp_dev", successGPU)

      successGPU = gpu_memcpy_async(tmp1_mpi_dev, tmp2_mpi_dev, &
                              l_cols*n_cols*size_of_datatype, gpuMemcpyDeviceToDevice, my_stream)
      check_memcpy_gpu("trans_ev_band_to_full: tmp2_dev -> tmp_dev", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp2_dev -> tmp_dev", successGPU)
      ! synchronize streamPerThread; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      check_stream_synchronize_gpu("trans_ev_band_to_full: tmp2_dev -> tmp_dev", successGPU)
#else
      successGPU = gpu_memcpy(tmp1_mpi_dev, tmp2_mpi_dev, &
                              l_cols*n_cols*size_of_datatype, gpuMemcpyDeviceToDevice)
      check_memcpy_gpu("trans_ev_band_to_full: tmp2_dev -> tmp_dev", successGPU)
#endif
#endif /* CUDA_AWARE_MPI_BAND_TO_FULL */
    endif

    if (l_rows>0) then
      if (useGPU) then
#ifndef MORE_GPUBLAS
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("trans_ev_band_to_full: tmp2 -> tmp_dev", successGPU)

        successGPU = gpu_memcpy_async(tmp_dev, int(loc(tmp2),kind=c_intptr_t), &
                      l_cols*n_cols*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
        check_memcpy_gpu("trans_ev_band_to_full: tmp2 -> tmp_dev", successGPU)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("trans_ev_band_to_full: tmp2 -> tmp_dev", successGPU)
        ! synchronize streamPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu("trans_ev_band_to_full: tmp2 -> tmp_dev", successGPU)
#else
        successGPU = gpu_memcpy(tmp_dev, int(loc(tmp2),kind=c_intptr_t), &
                      l_cols*n_cols*size_of_datatype, gpuMemcpyHostToDevice)
        check_memcpy_gpu("trans_ev_band_to_full: tmp2 -> tmp_dev", successGPU)
#endif
#endif /* MORE_GPUBLAS */

        ! needed: as long as not device to device copy
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)

        successGPU = gpu_memcpy_async(tmat_complete_dev, int(loc(tmat_complete),kind=c_intptr_t), &
                      cwy_blocking*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
        check_memcpy_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)
        ! synchronize streamPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)
#else
        successGPU = gpu_memcpy(tmat_complete_dev, int(loc(tmat_complete),kind=c_intptr_t), &
                      cwy_blocking*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice)
        check_memcpy_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)
#endif

        call obj%timer%start("gpublas")
        gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
        call gpublas_PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N', &
                                 n_cols, l_cols, ONE, tmat_complete_dev, cwy_blocking, tmp_dev, n_cols, gpuHandle)
        call gpublas_PRECISION_GEMM('N', 'N', l_rows, l_cols, n_cols, -ONE, hvm_dev, max_local_rows, tmp_dev, &
                                   n_cols, ONE, q_dev, ldq, gpuHandle)
        call obj%timer%stop("gpublas")
      else ! useGPU
        call obj%timer%start("blas")
        call PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N', &
                            int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), ONE, tmat_complete, &
                            int(cwy_blocking,kind=BLAS_KIND), tmp2, int(n_cols,kind=BLAS_KIND))
        call PRECISION_GEMM('N', 'N', int(l_rows,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), &
                            int(n_cols,kind=BLAS_KIND), -ONE, hvm, &
                            int(ubound(hvm,dim=1),kind=BLAS_KIND), tmp2, int(n_cols,kind=BLAS_KIND), ONE, &
                            q_mat, int(ldq,kind=BLAS_KIND))
        call obj%timer%stop("blas")
      endif ! useGPU

    endif
#else /* WITH_MPI */
    if (l_rows > 0) then
      if (useGPU) then
        ! needed as long as not device to device copy
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)

        successGPU = gpu_memcpy_async(tmat_complete_dev, int(loc(tmat_complete),kind=c_intptr_t), &
                      cwy_blocking*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice, my_stream)
        check_memcpy_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)
        ! synchronize streamPerThread; maybe not neccessary
        successGPU = gpu_stream_synchronize()
        check_stream_synchronize_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)
#else
        successGPU = gpu_memcpy(tmat_complete_dev, int(loc(tmat_complete),kind=c_intptr_t), &
                      cwy_blocking*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice)
        check_memcpy_gpu("trans_ev_band_to_full: tmat_complete -> tmat_complete_dev", successGPU)
#endif

        call obj%timer%start("gpublas")
        gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
        call gpublas_PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N', &
                                   n_cols, l_cols, ONE, tmat_complete_dev, cwy_blocking, &
                                   tmp_dev, n_cols, gpuHandle)
        call gpublas_PRECISION_GEMM('N', 'N', l_rows, l_cols, n_cols, &
                                    -ONE, hvm_dev, max_local_rows, tmp_dev, n_cols, ONE, q_dev, ldq, gpuHandle)
        call obj%timer%stop("gpublas")
      else ! useGPU
        call obj%timer%start("blas")
        call PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N', &
                            int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), ONE, tmat_complete, &
                            int(cwy_blocking,kind=BLAS_KIND), &
                            tmp1, int(n_cols,kind=BLAS_KIND))
        call PRECISION_GEMM('N', 'N', int(l_rows,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), &
                            -ONE, hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), tmp1, int(n_cols,kind=BLAS_KIND), ONE, q_mat, &
                            int(ldq,kind=BLAS_KIND))
        call obj%timer%stop("blas")
      endif ! useGPU
    endif
#endif /* WITH_MPI */

  enddo ! istep

  deallocate(hvb, stat=istat, errmsg=errorMessage)
  check_deallocate("trans_ev_band_to_full: hvb", istat, errorMessage)

  if (useGPU) then
    successGPU = gpu_free(hvm_dev)
    check_dealloc_gpu("trans_ev_band_to_full: hvm_dev", successGPU)

    successGPU = gpu_free(tmp_dev)
    check_dealloc_gpu("trans_ev_band_to_full: tmp_dev", successGPU)
#ifdef CUDA_AWARE_MPI_BAND_TO_FULL
    successGPU = gpu_free(tmp2_dev)
    check_dealloc_gpu("trans_ev_band_to_full: tmp2_dev", successGPU)
#endif

    successGPU = gpu_free(tmat_complete_dev)
    check_dealloc_gpu("trans_ev_band_to_full: tmat_complete_dev", successGPU)

    ! final transfer of q_dev
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("trans_ev_band_to_full: q_dev -> q_mat", successGPU)

    successGPU = gpu_memcpy_async(int(loc(q_mat),kind=c_intptr_t), q_dev, ldq*matrixCols*size_of_datatype, &
                  gpuMemcpyDeviceToHost, my_stream)
    check_memcpy_gpu("trans_ev_band_to_full: q_dev -> q_mat", successGPU)

    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("trans_ev_band_to_full: q_dev -> q_mat", successGPU)
    ! synchronize streamPerThread; maybe not neccessary
    successGPU = gpu_stream_synchronize()
    check_stream_synchronize_gpu("trans_ev_band_to_full: q_dev -> q_mat", successGPU)
#else
    successGPU = gpu_memcpy(int(loc(q_mat),kind=c_intptr_t), q_dev, ldq*matrixCols*size_of_datatype, &
                  gpuMemcpyDeviceToHost)
    check_memcpy_gpu("trans_ev_band_to_full: q_dev -> q_mat", successGPU)
#endif

    successGPU = gpu_free(q_dev)
    check_dealloc_gpu("trans_ev_band_to_full: q_dev", successGPU)

#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      successGPU = gpu_host_unregister(int(loc(q_mat),kind=c_intptr_t))
      check_host_unregister_gpu("trans_ev_band_to_full: q_mat", successGPU)
      nullify(tmp1)
      nullify(tmp2)
      nullify(hvm)
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    else
      deallocate(tmp1, tmp2, hvm)
    endif
#endif

    ! take care of new pointers nullify them
#ifdef MORE_GPUBLAS
    nullify(tmat_gpu_deviceptr)
    nullify(hvm_gpu_deviceptr)
    nullify(t_tmp_gpu_deviceptr)
#endif
#ifdef CUDA_AWARE_MPI_BAND_TO_FULL
    nullify(tmp1_mpi_deviceptr)
#endif


#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
#endif
      successGPU = gpu_free_host(tmp1_host)
      check_host_dealloc_gpu("trans_ev_band_to_full: tmp1_host", successGPU)

      successGPU = gpu_free_host(tmp2_host)
      check_host_dealloc_gpu("trans_ev_band_to_full: tmp2_host", successGPU)

      successGPU = gpu_free_host(hvm_host)
      check_host_dealloc_gpu("trans_ev_band_to_full: hvm_host", successGPU)

      successGPU = gpu_host_unregister(int(loc(tmat_complete),kind=c_intptr_t))
      check_host_unregister_gpu("trans_ev_band_to_full: tmat_complete", successGPU)
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    endif
#endif
  else ! useGPU
    deallocate(tmp1, stat=istat, errmsg=errorMessage)
    check_deallocate("trans_ev_band_to_full: tmp1", istat, errorMessage)

    deallocate(tmp2, stat=istat, errmsg=errorMessage)
    check_deallocate("trans_ev_band_to_full: tmp2", istat, errorMessage)

    deallocate(hvm, stat=istat, errmsg=errorMessage)
    check_deallocate("trans_ev_band_to_full: hvm", istat, errorMessage)
  endif ! useGPU

  deallocate(tmat_complete, stat=istat, errmsg=errorMessage)
  check_deallocate("trans_ev_band_to_full: tmat_complete", istat, errorMessage)

#ifdef MORE_GPUBLAS
  if (useGPU) then
    if (blocking_factor > 1) then
      successGPU = gpu_free(t_tmp_dev)
      check_dealloc_gpu("trans_ev_band_to_full: t_tmp_dev", successGPU)
#ifdef CUDA_AWARE_MPI_BAND_TO_FULL
      successGPU = gpu_free(t_tmp2_dev)
      check_dealloc_gpu("trans_ev_band_to_full: t_tmp2_dev", successGPU)
#endif
    endif
  endif
#endif /* MORE_GPUBLAS */

  if (blocking_factor > 1) then
#ifdef WITH_GPU_STREAMS
    if (useGPU) then
      successGPU = gpu_host_unregister(int(loc(t_tmp),kind=c_intptr_t))
      check_host_unregister_gpu("trans_ev_band_to_full: t_tmp", successGPU)

      successGPU = gpu_host_unregister(int(loc(t_tmp2),kind=c_intptr_t))
      check_host_unregister_gpu("trans_ev_band_to_full: t_tmp2", successGPU)
    endif
#endif

    deallocate(t_tmp, stat=istat, errmsg=errorMessage)
    check_deallocate("trans_ev_band_to_full: t_tmp", istat, errorMessage)

    deallocate(t_tmp2, stat=istat, errmsg=errorMessage)
    check_deallocate("trans_ev_band_to_full: t_tmp2", istat, errorMessage)
  endif

  call obj%timer%stop("trans_ev_band_to_full_&
  &MATH_DATATYPE&
  &" // &
  &PRECISION_SUFFIX //&
  gpuString)

end subroutine trans_ev_band_to_full_&
&MATH_DATATYPE&
    &_&
    &PRECISION

