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

subroutine trans_ev_band_to_full_&
    &MATH_DATATYPE&
    &_&
    &PRECISION &
    (obj, na, nqc, nblk, nbw, a_mat, lda, tmat, q_mat, &
     ldq, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols, useGPU &
#if REALCASE == 1
     ,useQr)
#endif
#if COMPLEXCASE == 1
     )
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
  class(elpa_abstract_impl_t), intent(inout) :: obj
  logical, intent(in)                    :: useGPU
#if REALCASE == 1
  logical, intent(in)                     :: useQR
#endif
  integer(kind=ik)                       :: na, nqc, lda, ldq, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols
#ifdef USE_ASSUMED_SIZE
  MATH_DATATYPE(kind=rck)                :: a_mat(lda,*)
  MATH_DATATYPE(kind=rck)                :: q_mat(ldq,*), tmat(nbw,nbw,*)
#else
  MATH_DATATYPE(kind=rck)                :: a_mat(lda,matrixCols)
  MATH_DATATYPE(kind=rck)                :: q_mat(ldq,matrixCols), tmat(nbw, nbw, numBlocks)
#endif

  integer(kind=ik)                       :: my_prow, my_pcol, np_rows, np_cols
  integer(kind=MPI_KIND)                 :: my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI, mpierr
  integer(kind=ik)                       :: max_blocks_row, max_blocks_col, max_local_rows, &
                                            max_local_cols
  integer(kind=ik)                       :: l_cols, l_rows, l_colh, n_cols
  integer(kind=ik)                       :: istep, lc, ncol, nrow, nb, ns

  MATH_DATATYPE(kind=rck), allocatable   :: hvb(:)
  MATH_DATATYPE(kind=rck), pointer       :: hvm(:,:), tmp1(:), tmp2(:)
  ! hvm_dev is fist used and set in this routine
  ! q_mat is changed in trans_ev_tridi on the host, copied to device and passed here. this can be adapted
  ! tmp_dev is first used in this routine
  ! tmat_dev is not passed along from bandred_real
  integer(kind=C_intptr_T)               :: hvm_dev, q_dev, tmp_dev, tmat_dev
  type(c_ptr)                            :: hvm_host, tmp1_host, tmp2_host

  integer(kind=ik)                       :: i

  MATH_DATATYPE(kind=rck), allocatable   :: tmat_complete(:,:), t_tmp(:,:), t_tmp2(:,:)
  integer(kind=ik)                       :: t_cols, t_rows
  integer(kind=ik)                       :: cwy_blocking

  integer(kind=ik)                       :: istat
  character(200)                         :: errorMessage
  character(20)                          :: gpuString
  logical                                :: successGPU
  integer(kind=c_intptr_t), parameter    :: size_of_datatype = size_of_&
                                                               &PRECISION&
                                                               &_&
                                                               &MATH_DATATYPE
  integer(kind=ik)                       :: blocking_factor, error, blk_end
  logical                                :: useIntelGPU

  if(useGPU) then
    gpuString = "_gpu"
  else
    gpuString = ""
  endif

  useIntelGPU = .false.
  if (useGPU) then
    if (gpu_vendor() == INTEL_GPU) then
      useIntelGPU = .true.
    endif
  endif


  call obj%timer%start("trans_ev_band_to_full_&
  &MATH_DATATYPE&
  &" // &
  &PRECISION_SUFFIX //&
  gpuString)

#ifdef BAND_TO_FULL_BLOCKING
  call obj%get("blocking_in_band_to_full",blocking_factor,error)
  if (error .ne. ELPA_OK) then
    print *,"Problem getting option for blocking_in_band_to_full. Aborting..."
    stop
  endif
#else
  blocking_factor = 1
#endif


  call obj%timer%start("mpi_communication")
  call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND) ,my_prowMPI ,mpierr)
  call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND) ,np_rowsMPI ,mpierr)
  call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND) ,my_pcolMPI ,mpierr)
  call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND) ,np_colsMPI ,mpierr)

  my_prow = int(my_prowMPI,kind=c_int)
  my_pcol = int(my_pcolMPI,kind=c_int)
  np_rows = int(np_rowsMPI,kind=c_int)
  np_cols = int(np_colsMPI,kind=c_int)
  call obj%timer%stop("mpi_communication")

  max_blocks_row = ((na -1)/nblk)/np_rows + 1 ! Rows of a_mat
  max_blocks_col = ((nqc-1)/nblk)/np_cols + 1 ! Columns of q_mat!

  max_local_rows = max_blocks_row*nblk
  max_local_cols = max_blocks_col*nblk

  cwy_blocking = blocking_factor * nbw

  if (useGPU) then
    if (useIntelGPU) then
      allocate(tmp1(max_local_cols*cwy_blocking), stat=istat, errmsg=errorMessage)
      check_allocate("trans_ev_band_to_full: tmp1", istat, errorMessage)
  
      allocate(tmp2(max_local_cols*cwy_blocking), stat=istat, errmsg=errorMessage)
      check_allocate("trans_ev_band_to_full: tmp2", istat, errorMessage)
  
      allocate(hvm(max_local_rows,cwy_blocking), stat=istat, errmsg=errorMessage)
      check_allocate("trans_ev_band_to_full: hvm", istat, errorMessage)

    else
      ! copy q_mat to q_dev
      successGPU = gpu_malloc(q_dev,ldq*matrixCols*size_of_datatype)
      check_alloc_gpu("trans_ev_band_to_full: q_dev", successGPU)

      successGPU = gpu_host_register(int(loc(q_mat),kind=c_intptr_t),&
                    ldq*matrixCols*size_of_datatype, gpuHostRegisterDefault)
      check_host_register_gpu("trans_ev_band_to_full: q_mat", successGPU)

      successGPU = gpu_memcpy(q_dev,int(loc(q_mat),kind=c_intptr_t),&
                    ldq*matrixCols*size_of_datatype, gpuMemcpyHostToDevice)
      check_memcpy_gpu("trans_ev_band_to_full: q_mat -> q_dev", successGPU)

      successGPU = gpu_malloc_host(tmp1_host,max_local_cols*cwy_blocking*size_of_datatype)
      check_host_alloc_gpu("trans_ev_band_to_full: tmp1_host", successGPU)
      call c_f_pointer(tmp1_host, tmp1, (/max_local_cols*cwy_blocking/))

      successGPU = gpu_malloc_host(tmp2_host,max_local_cols*cwy_blocking*size_of_datatype)
      check_host_alloc_gpu("trans_ev_band_to_full: tmp2_host", successGPU)
      call c_f_pointer(tmp2_host, tmp2, (/max_local_cols*cwy_blocking/))

      successGPU = gpu_malloc_host(hvm_host,max_local_rows*cwy_blocking*size_of_datatype)
      check_host_alloc_gpu("trans_ev_band_to_full: hvm_host", successGPU)
      call c_f_pointer(hvm_host, hvm, (/max_local_rows,cwy_blocking/))
    endif
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

  if (useGPU .and. .not.(useIntelGPU)) then
    successGPU = gpu_host_register(int(loc(tmat_complete),kind=c_intptr_t), &
                  cwy_blocking * cwy_blocking * size_of_datatype,&
                  gpuHostRegisterDefault)
    check_host_register_gpu("trans_ev_band_to_full: tmat_complete", successGPU)
  endif
  !if (useIntelGPU) then
  !  ! needed later
  !endif

  if (blocking_factor > 1) then
    allocate(t_tmp(cwy_blocking,nbw), stat=istat, errmsg=errorMessage)
    check_allocate("trans_ev_band_to_full: t_tmp", istat, errorMessage)

    allocate(t_tmp2(cwy_blocking,nbw), stat=istat, errmsg=errorMessage)
    check_allocate("trans_ev_band_to_full: t_tmp2", istat, errorMessage)
  endif

  if (useGPU .and. .not.(useIntelGPU)) then
    successGPU = gpu_malloc(hvm_dev,max_local_rows*cwy_blocking*size_of_datatype)
    check_alloc_gpu("trans_ev_band_to_full: hvm_dev", successGPU)

    successGPU = gpu_malloc(tmp_dev,max_local_cols*cwy_blocking*size_of_datatype)
    check_alloc_gpu("trans_ev_band_to_full: tmp_dev", successGPU)

    successGPU = gpu_malloc(tmat_dev,cwy_blocking*cwy_blocking*size_of_datatype)
    check_alloc_gpu("trans_ev_band_to_full: tmat_dev", successGPU)
  endif

  !if (useIntelGPU) then
  !  ! needed later
  !endif


  hvm = 0.0_rck ! Must be set to 0 !!!
  hvb = 0.0_rck ! Safety only
  tmp1 = 0.0_rck
  tmp2 = 0.0_rck
  tmat_complete = 0.0_rck
  if (blocking_factor > 1) then
     t_tmp = 0.0_rck ! Must be set to 0 !!!
     t_tmp2 = 0.0_rck
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

    do lc = 1, n_cols
      ncol = (istep-1)*cwy_blocking + nbw + lc ! absolute column number of householder Vector
      nrow = ncol - nbw ! absolute number of pivot row

      l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast
      l_colh = local_index(ncol , my_pcol, np_cols, nblk, -1) ! HV local column number

      if (my_pcol==pcol(ncol, nblk, np_cols)) hvb(nb+1:nb+l_rows) = a_mat(1:l_rows,l_colh)

      nb = nb+l_rows

      if (lc==n_cols .or. mod(ncol,nblk)==0) then
#ifdef WITH_MPI
        call obj%timer%start("mpi_communication")
        call MPI_Bcast(hvb(ns+1), int(nb-ns,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,&
                         int(pcol(ncol, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)

        call obj%timer%stop("mpi_communication")

#endif /* WITH_MPI */
        ns = nb
      endif
    enddo ! lc

    ! Expand compressed Householder vectors into matrix hvm

    nb = 0
    do lc = 1, n_cols
      nrow = (istep-1)*cwy_blocking + lc ! absolute number of pivot row
      l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast

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
        if (useIntelGPU) then
          !call obj%timer%start("mkl_offload")
#if 0
          call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', &
                              int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), ONE, hvm, &
                              int(max_local_rows,kind=BLAS_KIND), hvm(:,(i-1)*nbw+1:), &
                              int(max_local_rows,kind=BLAS_KIND), ZERO, t_tmp, int(cwy_blocking, kind=BLAS_KIND))
#endif
#ifdef WITH_INTEL_GPU_VERSION
          call mkl_offload_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', &
                              int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), ONE, hvm, &
                              int(max_local_rows,kind=BLAS_KIND), hvm(:,(i-1)*nbw+1:), &
                              int(max_local_rows,kind=BLAS_KIND), ZERO, t_tmp, int(cwy_blocking, kind=BLAS_KIND))
#endif
          !call obj%timer%stop("mkl_offload")

        else
          call obj%timer%start("blas")
          call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', &
                            int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), ONE, hvm, &
                            int(max_local_rows,kind=BLAS_KIND), hvm(:,(i-1)*nbw+1:), &
                            int(max_local_rows,kind=BLAS_KIND), ZERO, t_tmp, int(cwy_blocking, kind=BLAS_KIND))
          call obj%timer%stop("blas")
        endif


#ifdef WITH_MPI
        call obj%timer%start("mpi_communication")
        call mpi_allreduce(t_tmp, t_tmp2, int(cwy_blocking*nbw,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                           MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
        call obj%timer%stop("mpi_communication")

        if (useIntelGPU) then
          !call obj%timer%start("mkl_offload")
#if 0
          call PRECISION_TRMM('L', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), ONE, tmat_complete, &
                              int(cwy_blocking,kind=BLAS_KIND), t_tmp2, int(cwy_blocking,kind=BLAS_KIND))
          call PRECISION_TRMM('R', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), -ONE, &
                              tmat_complete(t_rows+1,t_rows+1), &
                              int(cwy_blocking,kind=BLAS_KIND), t_tmp2, int(cwy_blocking,kind=BLAS_KIND))
#endif
#ifdef WITH_INTEL_GPU_VERSION
          call mkl_offload_PRECISION_TRMM('L', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), ONE, &
                  tmat_complete, &
                              int(cwy_blocking,kind=BLAS_KIND), t_tmp2, int(cwy_blocking,kind=BLAS_KIND))
          call mkl_offload_PRECISION_TRMM('R', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), -ONE, &
                              tmat_complete(t_rows+1,t_rows+1), &
                              int(cwy_blocking,kind=BLAS_KIND), t_tmp2, int(cwy_blocking,kind=BLAS_KIND))
#endif
          !call obj%timer%stop("mkl_offload")
        else
          call obj%timer%start("blas")
          call PRECISION_TRMM('L', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), ONE, tmat_complete, &
                            int(cwy_blocking,kind=BLAS_KIND), t_tmp2, int(cwy_blocking,kind=BLAS_KIND))
          call PRECISION_TRMM('R', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), -ONE, &
                            tmat_complete(t_rows+1,t_rows+1), &
                            int(cwy_blocking,kind=BLAS_KIND), t_tmp2, int(cwy_blocking,kind=BLAS_KIND))
          call obj%timer%stop("blas")
        endif
        tmat_complete(1:t_rows,t_rows+1:t_rows+t_cols) = t_tmp2(1:t_rows,1:t_cols)

#else /* WITH_MPI */
        if (useIntelGPU) then
          !call obj%timer%start("mkl_offload")
#if 0
          call PRECISION_TRMM('L', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), ONE, tmat_complete, &
                              int(cwy_blocking,kind=BLAS_KIND), t_tmp, int(cwy_blocking,kind=BLAS_KIND))
          call PRECISION_TRMM('R', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), -ONE, &
                              tmat_complete(t_rows+1,t_rows+1), &
                              int(cwy_blocking,kind=BLAS_KIND), t_tmp, int(cwy_blocking,kind=BLAS_KIND))
#endif
#ifdef WITH_INTEL_GPU_VERSION
          call mkl_offload_PRECISION_TRMM('L', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), ONE, &
                                          tmat_complete, &
                              int(cwy_blocking,kind=BLAS_KIND), t_tmp, int(cwy_blocking,kind=BLAS_KIND))
          call mkl_offload_PRECISION_TRMM('R', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), -ONE, &
                              tmat_complete(t_rows+1,t_rows+1), &
                              int(cwy_blocking,kind=BLAS_KIND), t_tmp, int(cwy_blocking,kind=BLAS_KIND))
#endif
          !call obj%timer%stop("mkl_offload")

        else
          call obj%timer%start("blas")
          call PRECISION_TRMM('L', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), ONE, tmat_complete, &
                            int(cwy_blocking,kind=BLAS_KIND), t_tmp, int(cwy_blocking,kind=BLAS_KIND))
          call PRECISION_TRMM('R', 'U', 'N', 'N', int(t_rows,kind=BLAS_KIND), int(t_cols,kind=BLAS_KIND), -ONE, &
                              tmat_complete(t_rows+1,t_rows+1), &
                              int(cwy_blocking,kind=BLAS_KIND), t_tmp, int(cwy_blocking,kind=BLAS_KIND))
          call obj%timer%stop("blas")
        endif
        tmat_complete(1:t_rows,t_rows+1:t_rows+t_cols) = t_tmp(1:t_rows,1:t_cols)

#endif /* WITH_MPI */

      endif
    enddo

    ! Q = Q - V * T**T * V**T * Q

    if (l_rows>0) then
      if (useGPU) then
        if (useIntelGPU) then
          !call obj%timer%start("mkl_offload")
#if 0
          call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', &
                              int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), ONE, &
                              hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), q_mat, int(ldq,kind=BLAS_KIND), ZERO, tmp1, &
                              int(n_cols,kind=BLAS_KIND))
#endif
#ifdef WITH_INTEL_GPU_VERSION
          call mkl_offload_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', &
                            int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), ONE, &
                            hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), q_mat, int(ldq,kind=BLAS_KIND), ZERO, tmp1, &
                            int(n_cols,kind=BLAS_KIND))
#endif
          !call obj%timer%stop("mkl_offload")

        else
          successGPU = gpu_memcpy(hvm_dev, int(loc(hvm),kind=c_intptr_t), &
                          max_local_rows*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("trans_ev_band_to_full: hvm -> hvm_dev", successGPU)

          call obj%timer%start("gpublas")
          call gpublas_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', &
                                       n_cols, l_cols, l_rows, ONE, hvm_dev, max_local_rows, &
                                       q_dev, ldq , ZERO, tmp_dev, n_cols)
          call obj%timer%stop("gpublas")

#ifdef WITH_MPI
          ! copy data from device to host for a later MPI_ALLREDUCE
          successGPU = gpu_memcpy(int(loc(tmp1),kind=c_intptr_t), &
                        tmp_dev, l_cols*n_cols*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("trans_ev_band_to_full: tmp_dev -> tmp1", successGPU)
#endif /* WITH_MPI */
        endif
      else
        call obj%timer%start("blas")
        call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', &
                            int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), ONE, &
                            hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), q_mat, int(ldq,kind=BLAS_KIND), ZERO, tmp1, &
                           int(n_cols,kind=BLAS_KIND))
        call obj%timer%stop("blas")
      endif ! useGPU
    else ! l_rows>0
      tmp1(1:l_cols*n_cols) = 0.0_rck
    endif ! l_rows>0

#ifdef WITH_MPI
    call obj%timer%start("mpi_communication")
    call mpi_allreduce(tmp1, tmp2, int(n_cols*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, MPI_SUM, &
                       int(mpi_comm_rows,kind=MPI_KIND), mpierr)
    call obj%timer%stop("mpi_communication")

    if (l_rows>0) then
      if (useGPU) then
        if (useIntelGPU) then
          !call obj%timer%start("mkl_offload")
          call PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N', &
                              int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), ONE, tmat_complete, &
                              int(cwy_blocking,kind=BLAS_KIND), tmp2, int(n_cols,kind=BLAS_KIND))
          call PRECISION_GEMM('N', 'N', int(l_rows,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), &
                              int(n_cols,kind=BLAS_KIND), -ONE, hvm, &
                              int(ubound(hvm,dim=1),kind=BLAS_KIND), tmp2, int(n_cols,kind=BLAS_KIND), ONE, &
                              q_mat, int(ldq,kind=BLAS_KIND))
          !call obj%timer%stop("mkl_offload")

        else
          successGPU = gpu_memcpy(tmp_dev, int(loc(tmp2),kind=c_intptr_t), &
                        l_cols*n_cols*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("trans_ev_band_to_full: tmp2 -> tmp_dev", successGPU)

          successGPU = gpu_memcpy(tmat_dev, int(loc(tmat_complete),kind=c_intptr_t), &
                        cwy_blocking*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("trans_ev_band_to_full: tmat_complete -> tmat_dev", successGPU)

          call obj%timer%start("gpublas")
          call gpublas_PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N', &
                                   n_cols, l_cols, ONE, tmat_dev, cwy_blocking, tmp_dev, n_cols)
          call gpublas_PRECISION_GEMM('N', 'N', l_rows, l_cols, n_cols, -ONE, hvm_dev, max_local_rows, tmp_dev, &
                                     n_cols, ONE, q_dev, ldq)
          call obj%timer%stop("gpublas")
        endif
      else
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
    if (l_rows>0) then
      if (useGPU) then
        if (useIntelGPU) then
#if 0
          !call obj%timer%start("mkl_offload")
          call PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N', &
                            int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), ONE, tmat_complete, &
                            int(cwy_blocking,kind=BLAS_KIND), &
                            tmp1, int(n_cols,kind=BLAS_KIND))
          call PRECISION_GEMM('N', 'N', int(l_rows,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), &
                            -ONE, hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), tmp1, int(n_cols,kind=BLAS_KIND), ONE, q_mat, &
                            int(ldq,kind=BLAS_KIND))
#endif
#ifdef WITH_INTEL_GPU_VERSION
          call mkl_offload_PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N', &
                            int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), ONE, tmat_complete, &
                            int(cwy_blocking,kind=BLAS_KIND), &
                            tmp1, int(n_cols,kind=BLAS_KIND))
          call mkl_offload_PRECISION_GEMM('N', 'N', int(l_rows,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), &
                            -ONE, hvm, int(ubound(hvm,dim=1),kind=BLAS_KIND), tmp1, int(n_cols,kind=BLAS_KIND), ONE, q_mat, &
                            int(ldq,kind=BLAS_KIND))
#endif
          !call obj%timer%stop("mkl_offload")
        else
          successGPU = gpu_memcpy(tmat_dev, int(loc(tmat_complete),kind=c_intptr_t), &
                        cwy_blocking*cwy_blocking*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("trans_ev_band_to_full: tmat_complete -> tmat_dev", successGPU)

          call obj%timer%start("gpublas")
          call gpublas_PRECISION_TRMM('L', 'U', BLAS_TRANS_OR_CONJ, 'N', &
                                     n_cols, l_cols, ONE, tmat_dev, cwy_blocking, &
                                     tmp_dev, n_cols)
          call gpublas_PRECISION_GEMM('N', 'N', l_rows, l_cols, n_cols, &
                                      -ONE, hvm_dev, max_local_rows, tmp_dev, n_cols, ONE, q_dev, ldq)
          call obj%timer%stop("gpublas")
        endif
      else
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
    if (useIntelGPU) then
      deallocate(tmp1, stat=istat, errmsg=errorMessage)
      check_deallocate("trans_ev_band_to_full: tmp1", istat, errorMessage)

      deallocate(tmp2, stat=istat, errmsg=errorMessage)
      check_deallocate("trans_ev_band_to_full: tmp2", istat, errorMessage)

      deallocate(hvm, stat=istat, errmsg=errorMessage)
      check_deallocate("trans_ev_band_to_full: hvm", istat, errorMessage)

    else
      successGPU = gpu_free(hvm_dev)
      check_dealloc_gpu("trans_ev_band_to_full: hvm_dev", successGPU)

      successGPU = gpu_free(tmp_dev)
      check_dealloc_gpu("trans_ev_band_to_full: tmp_dev", successGPU)

      successGPU = gpu_free(tmat_dev)
      check_dealloc_gpu("trans_ev_band_to_full: tmat_dev", successGPU)

      ! final transfer of q_dev
      successGPU = gpu_memcpy(int(loc(q_mat),kind=c_intptr_t), q_dev, ldq*matrixCols*size_of_datatype, &
                    gpuMemcpyDeviceToHost)
      check_memcpy_gpu("trans_ev_band_to_full: q_dev -> q_mat", successGPU)

      successGPU = gpu_free(q_dev)
      check_dealloc_gpu("trans_ev_band_to_full: q_dev", successGPU)

      successGPU = gpu_host_unregister(int(loc(q_mat),kind=c_intptr_t))
      check_host_unregister_gpu("trans_ev_band_to_full: q_mat", successGPU)
      nullify(tmp1)
      nullify(tmp2)
      nullify(hvm)

      successGPU = gpu_free_host(tmp1_host)
      check_host_dealloc_gpu("trans_ev_band_to_full: tmp1_host", successGPU)

      successGPU = gpu_free_host(tmp2_host)
      check_host_dealloc_gpu("trans_ev_band_to_full: tmp2_host", successGPU)

      successGPU = gpu_free_host(hvm_host)
      check_host_dealloc_gpu("trans_ev_band_to_full: hvm_host", successGPU)

      successGPU = gpu_host_unregister(int(loc(tmat_complete),kind=c_intptr_t))
      check_host_unregister_gpu("trans_ev_band_to_full: tmat_complete", successGPU)
    endif
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

  if (blocking_factor > 1) then
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

