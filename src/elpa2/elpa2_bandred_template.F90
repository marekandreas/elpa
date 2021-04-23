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
subroutine bandred_&
&MATH_DATATYPE&
&_&
&PRECISION &
(obj, na, a_mat, lda, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols, tmat, &
wantDebug, useGPU, success, &
#if REALCASE == 1
useQR, &
#endif
max_threads)

!-------------------------------------------------------------------------------
!  bandred_real/complex: Reduces a distributed symmetric matrix to band form
!
!  Parameters
!
!  na          Order of matrix
!
!  a_mat(lda,matrixCols)    Distributed matrix which should be reduced.
!              Distribution is like in Scalapack.
!              Opposed to Scalapack, a_mat(:,:) must be set completely (upper and lower half)
!              a_mat(:,:) is overwritten on exit with the band and the Householder vectors
!              in the upper half.
!
!  lda         Leading dimension of a_mat
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

  implicit none
#include "../general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik)                            :: na, lda, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols

#ifdef USE_ASSUMED_SIZE
  MATH_DATATYPE(kind=rck)                     :: a_mat(lda,*)
  MATH_DATATYPE(kind=rck)                     :: tmat(nbw,nbw,*)
#else
  MATH_DATATYPE(kind=rck)                     :: a_mat(lda,matrixCols)
  MATH_DATATYPE(kind=rck)                     :: tmat(nbw,nbw,numBlocks)
#endif

#if REALCASE == 1
  real(kind=rk)                               :: eps
#endif
  logical, intent(in)                         :: useGPU
  integer(kind=c_int)                         :: skewsymmetric
  logical                                     :: isSkewsymmetric
  character(20)                               :: gpuString

  integer(kind=ik)                            :: my_prow, my_pcol, np_rows, np_cols
  integer(kind=MPI_KIND)                      :: mpierr,  my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI
  integer(kind=ik)                            :: l_cols, l_rows
#if REALCASE == 1
  integer(kind=ik)                            :: vmrCols
#endif
#ifdef WITH_OPENMP_TRADITIONAL
  integer(kind=ik)                            :: mynlc, lrs, transformChunkSize
#endif
  integer(kind=ik)                            :: i, j, lcs, lce, lre, lc, lr, cur_pcol, n_cols, nrow
  integer(kind=ik)                            :: istep, ncol, lch, lcx, nlc
  integer(kind=ik)                            :: tile_size, l_rows_tile, l_cols_tile

  real(kind=rk)                              :: vnorm2
  MATH_DATATYPE(kind=rck)                    :: xf, aux1(nbw), aux2(nbw), vrl, tau
  MATH_DATATYPE(kind=rck)                    :: vav(nbw,nbw)

  MATH_DATATYPE(kind=rck), allocatable :: tmpGPU(:)
  MATH_DATATYPE(kind=rck), pointer     :: vmrGPU(:), umcGPU(:)
  MATH_DATATYPE(kind=rck), allocatable :: tmpCPU(:,:), vmrCPU(:,:), umcCPU(:,:)
  MATH_DATATYPE(kind=rck), allocatable :: vr(:)

#if REALCASE == 1
  ! needed for blocked QR decomposition
  integer(kind=ik)                            :: PQRPARAM(11), work_size
  real(kind=rk)                    :: dwork_size(1)
  real(kind=rk), allocatable       :: work_blocked(:), tauvector(:), blockheuristic(:)
#endif
  integer(kind=C_intptr_T)                    :: a_dev, vmr_dev, umc_dev, tmat_dev, vav_dev
  type(c_ptr)                                 :: vmr_host, umc_host
#ifdef WITH_MPI
  !integer(kind=ik), external                  :: numroc -> use elpa_scalapack
#endif
  integer(kind=ik)                            :: ierr
  integer(kind=ik)                            :: cur_l_rows, cur_l_cols, vmr_size, umc_size
  integer(kind=ik)                            :: l_rows2, vmr_size2, umc_size2
  integer(kind=c_intptr_t)                    :: lc_start, lc_end
#if COMPLEXCASE == 1
  integer(kind=c_intptr_t)                    :: lce_1, lcs_1, lre_1
#endif
  integer(kind=ik)                            :: lr_end
  integer(kind=ik)                            :: na_cols
  integer(kind=BLAS_KIND)                     :: na_colsBLAS
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
                                                ii, pp
  integer(kind=c_intptr_t), parameter           :: size_of_datatype = size_of_&
                                                                    &PRECISION&
                                                                    &_&
                                                                    &MATH_DATATYPE

  logical                                     :: useGPU_reduction_lower_block_to_tridiagonal
  integer(kind=ik), intent(in)                :: max_threads
  logical                                     :: do_memcpy
  integer(kind=ik)                            :: i_blk,blk_off, blk_end
  logical                                     :: useIntelGPU

  call obj%get("is_skewsymmetric",skewsymmetric,error)
  if (error .ne. ELPA_OK) then
       print *,"Problem getting option for skewsymmetric settings. Aborting..."
       stop
  endif
  isSkewsymmetric = (skewsymmetric == 1)

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

  call obj%timer%start("bandred_&
  &MATH_DATATYPE&
  &" // &
  PRECISION_SUFFIX // &
  gpuString )

  useGPU_reduction_lower_block_to_tridiagonal = .false.

  if (useGPU .and. .not.(useIntelGPU)) then
    useGPU_reduction_lower_block_to_tridiagonal = .true.
#if REALCASE == 1
    if (useQR) then
      !in this case switch off GPU usage for step "reduce current block to lower triangular form"
      ! since this is done by QR decomposition
      useGPU_reduction_lower_block_to_tridiagonal = .false.
    endif
#endif
  endif

  if (useIntelGPU) then
    useGPU_reduction_lower_block_to_tridiagonal = .true.
#if REALCASE == 1
    if (useQR) then
      !in this case switch off GPU usage for step "reduce current block to lower triangular form"
      ! since this is done by QR decomposition
      useGPU_reduction_lower_block_to_tridiagonal = .false.
    endif
#endif
  endif

  if (wantDebug) call obj%timer%start("mpi_communication")

  call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND) ,my_prowMPI ,mpierr)
  call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND) ,np_rowsMPI ,mpierr)
  call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND) ,my_pcolMPI ,mpierr)
  call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND) ,np_colsMPI ,mpierr)

  my_prow = int(my_prowMPI,kind=c_int)
  np_rows = int(np_rowsMPI,kind=c_int)
  my_pcol = int(my_pcolMPI,kind=c_int)
  np_cols = int(np_colsMPI,kind=c_int)

  if (wantDebug) call obj%timer%stop("mpi_communication")
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
  if (useGPU .and. .not.(useIntelGPU)) then
#ifdef WITH_MPI
#if COMPLEXCASE == 1
    na_rowsBLAS = numroc(int(na,kind=BLAS_KIND), int(nblk,kind=BLAS_KIND), &
                         int(my_prow,kind=BLAS_KIND), 0_BLAS_KIND, int(np_rows,kind=BLAS_KIND))
    na_rows = int(na_rowsBLAS,kind=c_int)
#endif
    na_colsBLAS = numroc(int(na,kind=BLAS_KIND), int(nblk,kind=BLAS_KIND), &
                         int(my_pcol,kind=BLAS_KIND), 0_BLAS_KIND, int(np_cols,kind=BLAS_KIND))
    na_cols = int(na_colsBLAS,kind=c_int)
#else
#if COMPLEXCASE == 1
    na_rows = na
#endif
    na_cols = na
#endif /* WITH_MPI */

    ! Here we convert the regular host array into a pinned host array
    successGPU = gpu_malloc(a_dev, lda*na_cols* size_of_datatype)
    check_alloc_gpu("bandred: a_dev", successGPU)

    successGPU = gpu_host_register(int(loc(vav),kind=c_intptr_t), &
                  nbw * nbw * size_of_datatype,&
                  gpuHostRegisterDefault)
    check_host_register_gpu("bandred: vav", successGPU)

    successGPU = gpu_malloc(vav_dev, nbw*nbw* size_of_datatype)
    check_alloc_gpu("bandred: vav_dev", successGPU)
  endif ! useGPU

  !if (useIntelGPU) then
  !  ! needed later when explicit copy
  !endif ! useIntelGPU

  ! Matrix is split into tiles; work is done only for tiles on the diagonal or above

  tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size

  ! make tile_size a smallest possible multiple of previously defined tile size, such that it is
  ! larger or equal to min_tile_size
  ! min_tile_size has been originally hardcoded as 128 * max(np_rows, np_cols), so it is now the implicit value
  ! it can, however, be set by the user
  call obj%get("min_tile_size", min_tile_size ,error)
  if (error .ne. ELPA_OK) then
    print *,"Problem setting option for min_tile_size. Aborting..."
    stop
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
      allocate(vmrCPU(max(l_rows,1),na), stat=istat, errmsg=errorMessage)
      check_allocate("bandred: vmrCPU", istat, errorMessage)

      vmrCols = na

#ifdef USE_ASSUMED_SIZE_QR
      call qr_pdgeqrf_2dcomm_&
           &PRECISION&
           &(obj, a_mat, lda, matrixCols, vmrCPU, max(l_rows,1), vmrCols, tauvector(1), na, tmat(1,1,1), &
                             nbw, nbw, dwork_size, 1, -1, na, nbw, nblk, nblk, na, na, 1, 0, PQRPARAM(1:11), &
                             mpi_comm_rows, mpi_comm_cols, blockheuristic)

#else
      call qr_pdgeqrf_2dcomm_&
           &PRECISION&
           &(obj, a_mat(1:lda,1:matrixCols), matrixCols, lda, vmrCPU(1:max(l_rows,1),1:vmrCols), max(l_rows,1), &
                             vmrCols, tauvector(1:na), na, tmat(1:nbw,1:nbw,1), nbw, &
                             nbw, dwork_size(1:1), 1, -1, na, nbw, nblk, nblk, na, na, 1, 0, PQRPARAM(1:11), &
                             mpi_comm_rows, mpi_comm_cols, blockheuristic)
#endif

      work_size = int(dwork_size(1))
      allocate(work_blocked(work_size), stat=istat, errmsg=errorMessage)
      check_allocate("bandred: work_blocked", istat, errorMessage)
      work_blocked = 0.0_rk
      deallocate(vmrCPU, stat=istat, errmsg=errorMessage)
      check_deallocate("bandred: vmrCPU", istat, errorMessage)

    endif ! which_qr_decomposition

  endif ! useQr
#endif /* REALCASE */

  blk_end = (na-1)/nbw
  if (useGPU .and. .not.(useIntelGPU)) then

    successGPU = gpu_host_register(int(loc(a_mat),kind=c_intptr_t), &
                  lda*na_cols*size_of_datatype, gpuHostRegisterDefault)
    check_host_register_gpu("bandred: a_mat", successGPU)

    cur_l_rows = 0
    cur_l_cols = 0

    successGPU = gpu_memcpy(a_dev, int(loc(a_mat),kind=c_intptr_t), &
                  lda*na_cols*size_of_datatype, gpuMemcpyHostToDevice)
    check_memcpy_gpu("bandred: a_dev", successGPU)

    successGPU = gpu_malloc(tmat_dev, nbw*nbw*size_of_datatype)
    check_alloc_gpu("bandred: tmat_dev", successGPU)

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
    if (istat .ne. 0) then
      print *,"bandred_&
              &MATH_DATATYPE&
              &: error when allocating vr "//errorMessage
      stop 1
    endif

    successGPU = gpu_malloc_host(vmr_host,vmr_size*size_of_datatype)
    check_host_alloc_gpu("bandred: vmr_host", successGPU)
    call c_f_pointer(vmr_host, vmrGPU, (/vmr_size/))

    successGPU = gpu_malloc(vmr_dev, vmr_size*size_of_datatype)
    check_alloc_gpu("bandred: vmr_dev", successGPU)

    successGPU = gpu_malloc_host(umc_host,umc_size*size_of_datatype)
    check_host_alloc_gpu("bandred: umc_host", successGPU)
    call c_f_pointer(umc_host, umcGPU, (/umc_size/))

    successGPU = gpu_malloc(umc_dev, umc_size*size_of_datatype)
    check_alloc_gpu("bandred: umc_dev", successGPU)

  endif ! useGPU

  !if (useIntelGPU) then
     ! needed later when explict mem copy
  !endif ! useIntelGPU


  do istep = blk_end, 1, -1

    n_cols = MIN(na,(istep+1)*nbw) - istep*nbw ! Number of columns in current step

    ! Number of local columns/rows of remaining matrix
    l_cols = local_index(istep*nbw, my_pcol, np_cols, nblk, -1)
    l_rows = local_index(istep*nbw, my_prow, np_rows, nblk, -1)

    ! Allocate vmr and umc to their exact sizes so that they can be used in bcasts and reduces

    if (useGPU) then
      if (useIntelGPU) then
        ! unify the the name vmr and vmrCPU, as well as vmrGPU
        ! the same for umcCPU and umcGPU
        ! Allocate vmr and umcCPU to their exact sizes so that they can be used in bcasts and reduces

        allocate(vmrCPU(max(l_rows,1),2*n_cols), stat=istat, errmsg=errorMessage)
        check_allocate("bandred: vmrCPU", istat, errorMessage)

        allocate(umcCPU(max(l_cols,1),2*n_cols), stat=istat, errmsg=errorMessage)
        check_allocate("bandred: umcCPU", istat, errorMessage)

        allocate(vr(l_rows+1), stat=istat, errmsg=errorMessage)
        check_allocate("bandred: vr", istat, errorMessage)
      else
        cur_l_rows = max(l_rows, 1)
        cur_l_cols = max(l_cols, 1)
        vmr_size = cur_l_rows * 2 * n_cols
        umc_size = cur_l_cols * 2 * n_cols
      endif
    else ! GPU not used

      ! unify the the name vmr and vmrCPU, as well as vmrGPU
      ! the same for umcCPU and umcGPU
      ! Allocate vmr and umcCPU to their exact sizes so that they can be used in bcasts and reduces

      allocate(vmrCPU(max(l_rows,1),2*n_cols), stat=istat, errmsg=errorMessage)
      check_allocate("bandred: vmrCPU", istat, errorMessage)

      allocate(umcCPU(max(l_cols,1),2*n_cols), stat=istat, errmsg=errorMessage)
      check_allocate("bandred: umcCPU", istat, errorMessage)

      allocate(vr(l_rows+1), stat=istat, errmsg=errorMessage)
      check_allocate("bandred: vr", istat, errorMessage)

    endif ! use GPU

    if (useGPU) then
      if (useIntelGPU) then
        vmrCPU(1:l_rows,1:n_cols) = 0.0_rck
      else
        vmrGPU(1 : cur_l_rows * n_cols) = 0.0_rck
        umcGPU(1 : umc_size) = 0.0_rck
      endif
    else
      vmrCPU(1:l_rows,1:n_cols) = 0.0_rck
    endif ! useGPU

    vr(:) = 0.0_rck
    tmat(:,:,istep) = 0.0_rck
    if (useGPU .and. .not.(useIntelGPU)) then
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
        successGPU = gpu_memcpy2d(int(loc(a_mat(1, lc_start)),kind=c_intptr_t), &
                      int((lda*size_of_datatype),kind=c_intptr_t), &
                      (a_dev + int( ( (lc_start-1) * lda*size_of_datatype),kind=c_intptr_t )), &
                      int(lda*size_of_datatype,kind=c_intptr_t), &
                      int(lr_end*size_of_datatype,kind=c_intptr_t), &
                      int((lc_end - lc_start+1),kind=c_intptr_t),int(gpuMemcpyDeviceToHost,kind=c_int))

        check_memcpy_gpu("bandred: a_dev -> a_mat", successGPU)
      endif
    endif ! useGPU

    !if (useIntelGPU) then
       ! needed later when explict mem copy
    !endif ! useIntelGPU


    ! Reduce current block to lower triangular form
#if REALCASE == 1
    if (useQR) then
      if (which_qr_decomposition == 1) then
        vmrCols = 2*n_cols
#ifdef USE_ASSUMED_SIZE_QR
        call qr_pdgeqrf_2dcomm_&
             &PRECISION&
             &(obj, a_mat, lda, matrixCols, vmrCPU, max(l_rows,1), vmrCols, tauvector(1), &
                               na, tmat(1,1,istep), nbw, nbw, work_blocked, work_size,        &
                                 work_size, na, n_cols, nblk, nblk,        &
                                 istep*nbw+n_cols-nbw, istep*nbw+n_cols, 1,&
                                 0, PQRPARAM(1:11), mpi_comm_rows, mpi_comm_cols,&
                                 blockheuristic)

#else
        call qr_pdgeqrf_2dcomm_&
             &PRECISION&
             &(obj, a_mat(1:lda,1:matrixCols), lda, matrixCols, vmrCPU(1:max(l_rows,1),1:vmrCols) ,   &
                                max(l_rows,1), vmrCols, tauvector(1:na), na, &
                                 tmat(1:nbw,1:nbw,istep), nbw, nbw, work_blocked(1:work_size), work_size, &
                                 work_size, na, n_cols, nblk, nblk,        &
                                 istep*nbw+n_cols-nbw, istep*nbw+n_cols, 1,&
                                 0, PQRPARAM(1:11), mpi_comm_rows, mpi_comm_cols,&
                                 blockheuristic)
#endif
      endif

    else !useQR
#endif /* REALCASE == 1 */
      do lc = n_cols, 1, -1

        ncol = istep*nbw + lc ! absolute column number of householder Vector
        nrow = ncol - nbw ! Absolute number of pivot row

        lr  = local_index(nrow, my_prow, np_rows, nblk, -1) ! current row length
        lch = local_index(ncol, my_pcol, np_cols, nblk, -1) ! HV local column number

        tau = 0

        if (nrow == 1) exit ! Nothing to do

        cur_pcol = pcol(ncol, nblk, np_cols) ! Processor column owning current block

        if (my_pcol==cur_pcol) then

          ! Get Vector to be transformed; distribute last element and norm of
          ! remaining elements to all procs in current column

          vr(1:lr) = a_mat(1:lr,lch) ! Vector to be transformed

          if (my_prow==prow(nrow, nblk, np_rows)) then
            aux1(1) = dot_product(vr(1:lr-1),vr(1:lr-1))
            aux1(2) = vr(lr)
          else
            aux1(1) = dot_product(vr(1:lr),vr(1:lr))
            aux1(2) = 0.0_rck
          endif

#ifdef WITH_MPI
          if (wantDebug) call obj%timer%start("mpi_communication")
          call mpi_allreduce(aux1, aux2, 2_MPI_KIND, MPI_MATH_DATATYPE_PRECISION, &
                             MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
          if (wantDebug) call obj%timer%stop("mpi_communication")

#else /* WITH_MPI */
          aux2 = aux1 ! this should be optimized
#endif

#if REALCASE == 1
          vnorm2 = aux2(1)
#endif
#if COMPLEXCASE == 1
          vnorm2 = real(aux2(1),kind=rk)
#endif
          vrl    = aux2(2)

          ! Householder transformation
          call hh_transform_&
             &MATH_DATATYPE&
             &_&
             &PRECISION &
                         (obj, vrl, vnorm2, xf, tau, wantDebug)
          ! Scale vr and store Householder Vector for back transformation

          vr(1:lr) = vr(1:lr) * xf
          if (my_prow==prow(nrow, nblk, np_rows)) then
            a_mat(1:lr-1,lch) = vr(1:lr-1)
            a_mat(lr,lch) = vrl
            vr(lr) = 1.0_rck
          else
            a_mat(1:lr,lch) = vr(1:lr)
          endif

        endif

        ! Broadcast Householder Vector and tau along columns

        vr(lr+1) = tau
#ifdef WITH_MPI
        if (wantDebug) call obj%timer%start("mpi_communication")
        call MPI_Bcast(vr, int(lr+1,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                      int(cur_pcol,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
        if (wantDebug) call obj%timer%stop("mpi_communication")

#endif /* WITH_MPI */

        if (useGPU_reduction_lower_block_to_tridiagonal .and. .not.(useIntelGPU)) then
          vmrGPU(cur_l_rows * (lc - 1) + 1 : cur_l_rows * (lc - 1) + lr) = vr(1:lr)
        else
          vmrCPU(1:lr,lc) = vr(1:lr)
        endif
        tau = vr(lr+1)

#if REALCASE == 1
        tmat(lc,lc,istep) = tau ! Store tau in diagonal of tmat
#endif
#if COMPLEXCASE == 1
        tmat(lc,lc,istep) = conjg(tau) ! Store tau in diagonal of tmat
#endif
        ! Transform remaining columns in current block with Householder Vector
        ! Local dot product

        aux1 = 0.0_rck

#ifdef WITH_OPENMP_TRADITIONAL
!#if 0
! ! original complex implementation without openmp. check performance
!        nlc = 0 ! number of local columns
!        do j=1,lc-1
!          lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
!          if (lcx>0) then
!            nlc = nlc+1
!            aux1(nlc) = dot_product(vr(1:lr),a_mat(1:lr,lcx))
!          endif
!        enddo
!
!        ! Get global dot products
!#ifdef WITH_MPI
!        if (wantDebug) call obj%timer%start("mpi_communication")
!        if (nlc>0) call mpi_allreduce(aux1, aux2, int(nlc,kind=MPI_KIND), MPI_COMPLEX_PRECISION, MPI_SUM, &
!                                         int(mpi_comm_rows,kind=MPI_KIND), mpierr)
!
!        ! Transform
!
!        nlc = 0
!        do j=1,lc-1
!          lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
!          if (lcx>0) then
!            nlc = nlc+1
!            a_mat(1:lr,lcx) = a_mat(1:lr,lcx) - conjg(tau)*aux2(nlc)*vr(1:lr)
!
!          endif
!        enddo
!
!
!        if (wantDebug) call obj%timer%stop("mpi_communication")
!
!#else /* WITH_MPI */
!
!        ! Transform
!
!        nlc = 0
!        do j=1,lc-1
!          lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
!          if (lcx>0) then
!            nlc = nlc+1
!            a_mat(1:lr,lcx) = a_mat(1:lr,lcx) - conjg(tau)*aux1(nlc)*vr(1:lr)
!          endif
!        enddo
!
!#endif /* WITH_MPI */
!!
!!       ! Transform
!!
!!       nlc = 0
!!       do j=1,lc-1
!!         lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
!!         if (lcx>0) then
!!           nlc = nlc+1
!!           a_mat(1:lr,lcx) = a_mat(1:lr,lcx) - conjg(tau)*aux2(nlc)*vr(1:lr)
!
!!         endif
!!       enddo
!#endif /* if 0 */

        !Open up one omp region to avoid paying openmp overhead.
        !This does not help performance due to the addition of two openmp barriers around the MPI call,
        !But in the future this may be beneficial if these barriers are replaced with a faster implementation

        !$omp  parallel &
        !$omp  default(none) &
        !$omp  shared(lc, istep, nbw, my_pcol, np_cols, nblk, &
        !$omp& lr, vr, a_mat, transformChunkSize, tau, aux1, aux2, wantDebug, mpi_comm_rows, obj) &
        !$omp private(mynlc, j, lcx, ii, pp, mpierr )        
        mynlc = 0 ! number of local columns

        !This loop does not have independent iterations,
        !'mynlc' is incremented each iteration, and it is difficult to remove this dependency
        !Thus each thread executes every iteration of the loop, except it only does the work if it 'owns' that iteration
        !That is, a thread only executes the work associated with an iteration if its thread id is congruent to
        !the iteration number modulo the number of threads
        do j=1,lc-1
          lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
          if (lcx>0 ) then
            mynlc = mynlc+1
            if ( mod((j-1), omp_get_num_threads()) .eq. omp_get_thread_num() ) then
                if (lr>0) aux1(mynlc) = dot_product(vr(1:lr),a_mat(1:lr,lcx))
            endif
          endif
        enddo

        ! Get global dot products

        !$omp barrier
        !$omp single
#ifdef WITH_MPI
        if (wantDebug) call obj%timer%start("mpi_communication")
        if (mynlc>0) call mpi_allreduce(aux1, aux2, int(mynlc,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                                        MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
        if (wantDebug) call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
        if (mynlc>0) aux2 = aux1
#endif /* WITH_MPI */
        !$omp end single
        !$omp barrier

        ! Transform
        transformChunkSize=32
        mynlc = 0
        do j=1,lc-1
          lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
          if (lcx>0) then
            mynlc = mynlc+1
            !This loop could be parallelized with an openmp pragma with static scheduling and chunk size 32
            !However, for some reason this is slower than doing it manually, so it is parallelized as below.
            do ii=omp_get_thread_num()*transformChunkSize,lr,omp_get_num_threads()*transformChunkSize
              do pp = 1,transformChunkSize
                if (pp + ii > lr) exit
#if REALCASE == 1
                a_mat(ii+pp,lcx) = a_mat(ii+pp,lcx) - tau*aux2(mynlc)*vr(ii+pp)
#endif
#if COMPLEXCASE == 1
                a_mat(ii+pp,lcx) = a_mat(ii+pp,lcx) - conjg(tau)*aux2(mynlc)*vr(ii+pp)
#endif
              enddo
            enddo
          endif
        enddo
        !$omp end parallel

#else /* WITH_OPENMP_TRADITIONAL */

        nlc = 0 ! number of local columns
        do j=1,lc-1
          lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
          if (lcx>0) then
            nlc = nlc+1
            if (lr>0) aux1(nlc) = dot_product(vr(1:lr),a_mat(1:lr,lcx))
          endif
        enddo

        ! Get global dot products
#ifdef WITH_MPI
        if (wantDebug) call obj%timer%start("mpi_communication")
        if (nlc>0) call mpi_allreduce(aux1, aux2, int(nlc,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                                      MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
        if (wantDebug) call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
        if (nlc>0) aux2=aux1
#endif /* WITH_MPI */
        ! Transform

        nlc = 0
        do j=1,lc-1
          lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
          if (lcx>0) then
            nlc = nlc+1
#if REALCASE == 1
            a_mat(1:lr,lcx) = a_mat(1:lr,lcx) - tau*aux2(nlc)*vr(1:lr)
#endif
#if COMPLEXCASE == 1
            a_mat(1:lr,lcx) = a_mat(1:lr,lcx) - conjg(tau)*aux2(nlc)*vr(1:lr)
#endif
          endif
        enddo
#endif /* WITH_OPENMP_TRADITIONAL */
      enddo ! lc

      if (useGPU_reduction_lower_block_to_tridiagonal .and. .not.(useIntelGPU)) then
        ! store column tiles back to GPU
        if (do_memcpy) then
          successGPU = gpu_memcpy2d((a_dev+ &
                        int(((lc_start-1)*lda*size_of_datatype),kind=c_intptr_t)), &
                        int(lda*size_of_datatype,kind=c_intptr_t), int(loc(a_mat(1,lc_start)),kind=c_intptr_t), &
                        int(lda*size_of_datatype,kind=c_intptr_t), &
                        int(lr_end*size_of_datatype,kind=c_intptr_t), &
                        int((lc_end - lc_start+1),kind=c_intptr_t), &
                        int(gpuMemcpyHostToDevice,kind=c_int))
          check_memcpy_gpu("bandred: a_mat -> a_dev", successGPU)
        endif
      endif

      if (useGPU_reduction_lower_block_to_tridiagonal .and. useIntelGPU) then
        ! store column tiles back to GPU
      endif

      ! Calculate scalar products of stored Householder vectors.
      ! This can be done in different ways, we use dsyrk

      vav = 0
      call obj%timer%start("blas")
      if (useGPU_reduction_lower_block_to_tridiagonal .and. .not.(useIntelGPU)) then
        if (l_rows>0) &
#if REALCASE == 1
        call PRECISION_SYRK('U', 'T',            &
#endif
#if COMPLEXCASE == 1
        call PRECISION_HERK('U', 'C',            &
#endif
                           int(n_cols,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), ONE, &
                           vmrGPU, int(cur_l_rows,kind=BLAS_KIND), &
                           ZERO, vav, int(ubound(vav,dim=1),kind=BLAS_KIND))

      else ! useGPU_reduction_to_tridiagonal
        if (l_rows>0) &
#if REALCASE == 1
        call PRECISION_SYRK('U', 'T',           &
#endif
#if COMPLEXCASE == 1
        call PRECISION_HERK('U', 'C',           &
#endif
                            int(n_cols,kind=BLAS_KIND), int(l_rows,kind=BLAS_KIND), ONE, vmrCPU, &
                            int(ubound(vmrCPU,dim=1),kind=BLAS_KIND), ZERO, vav, int(ubound(vav,dim=1),kind=BLAS_KIND))
      endif
      call obj%timer%stop("blas")
#if REALCASE == 1
      call symm_matrix_allreduce_&
#endif
#if COMPLEXCASE == 1
      call herm_matrix_allreduce_&
#endif
         &PRECISION &
                         (obj, n_cols,vav, nbw, nbw,mpi_comm_rows)
         ! Calculate triangular matrix T for block Householder Transformation
      call obj%timer%start("blas")
      do lc=n_cols,1,-1
        tau = tmat(lc,lc,istep)
        if (lc<n_cols) then
          call PRECISION_TRMV('U', BLAS_TRANS_OR_CONJ, 'N',&
                              int(n_cols-lc,kind=BLAS_KIND), tmat(lc+1,lc+1,istep), &
                              int(ubound(tmat,dim=1),kind=BLAS_KIND), vav(lc+1,lc), 1_BLAS_KIND)

#if REALCASE == 1
          tmat(lc,lc+1:n_cols,istep) = -tau * vav(lc+1:n_cols,lc)
#endif
#if COMPLEXCASE == 1
          tmat(lc,lc+1:n_cols,istep) = -tau * conjg(vav(lc+1:n_cols,lc))
#endif
        endif
      enddo
      call obj%timer%stop("blas")
#if REALCASE == 1
    endif !useQR
#endif

#if REALCASE == 1
    if (useGPU .and. useQR ) then
      if (useIntelGPU) then
        ! copy the data for furhter usage
        ! qr worked on *CPU arrarys
        !vmrCUDA(1:cur_l_rows * n_cols) = vmrCPU(1:cur_l_rows,1:n_cols)
      else     
        ! copy the data for furhter usage
        ! qr worked on *CPU arrarys
        !vmrGPU(1:cur_l_rows * n_cols) = vmrCPU(1:cur_l_rows,1:n_cols)
        if (do_memcpy) then
          successGPU = gpu_memcpy2d((a_dev+ &
                        int(((lc_start-1)*lda*size_of_datatype),kind=c_intptr_t)), &
                        int(lda*size_of_datatype,kind=c_intptr_t), int(loc(a_mat(1,lc_start)),kind=c_intptr_t), &
                        int(lda*size_of_datatype,kind=c_intptr_t), &
                        int(lr_end*size_of_datatype,kind=c_intptr_t), &
                        int((lc_end - lc_start+1),kind=c_intptr_t), &
                        int(gpuMemcpyHostToDevice,kind=c_int))
          check_memcpy_gpu("bandred: a_mat -> a_dev", successGPU)
        endif
      endif
    endif
#endif

    ! Transpose vmr -> vmc (stored in umc, second half)
    if (useGPU) then
      if (useIntelGPU) then
        call elpa_transpose_vectors_&
             &MATH_DATATYPE&
             &_&
             &PRECISION &
                                          (obj, vmrCPU, ubound(vmrCPU,dim=1), mpi_comm_rows, &
                                           umcCPU(1,n_cols+1), ubound(umcCPU,dim=1), mpi_comm_cols, &
                                           1, istep*nbw, n_cols, nblk, max_threads)

      else
        call elpa_transpose_vectors_&
             &MATH_DATATYPE&
             &_&
             &PRECISION &
                          (obj, vmrGPU(:), cur_l_rows, mpi_comm_rows, &
                           umcGPU(cur_l_cols * n_cols + 1:), cur_l_cols, &
                           mpi_comm_cols, 1, istep*nbw, n_cols, nblk, max_threads)
      endif
    else ! useGPU
      call elpa_transpose_vectors_&
           &MATH_DATATYPE&
           &_&
           &PRECISION &
                                        (obj, vmrCPU, ubound(vmrCPU,dim=1), mpi_comm_rows, &
                                         umcCPU(1,n_cols+1), ubound(umcCPU,dim=1), mpi_comm_cols, &
                                         1, istep*nbw, n_cols, nblk, max_threads)
    endif

    ! Calculate umc = A**T * vmr
    ! Note that the distributed A has to be transposed
    ! Opposed to direct tridiagonalization there is no need to use the cache locality
    ! of the tiles, so we can use strips of the matrix


!#if 0
!    ! original complex implemetation check for performance
!    umcCPU(1:l_cols,1:n_cols) = 0.0_rck
!    vmrCPU(1:l_rows,n_cols+1:2*n_cols) = 0.0_rck
!
!    if (l_cols>0 .and. l_rows>0) then
!      do i=0,(istep*nbw-1)/tile_size
!
!        lcs = i*l_cols_tile+1
!        lce = min(l_cols,(i+1)*l_cols_tile)
!        if (lce<lcs) cycle
!
!        lre = min(l_rows,(i+1)*l_rows_tile)
!
!          call obj%timer%start("blas")
!          call PRECISION_GEMM('C', 'N', lce-lcs+1, n_cols, lre, ONE, a_mat(1,lcs), ubound(a_mat,dim=1), &
!                     vmrCPU, ubound(vmrCPU,dim=1), ONE, umcCPU(lcs,1), ubound(umcCPU,dim=1))
!          call obj%timer%stop("blas")
!
!        if (i==0) cycle
!        lre = min(l_rows,i*l_rows_tile)
!          call obj%timer%start("blas")
!          call PRECISION_GEMM('N', 'N', lre, n_cols, lce-lcs+1, ONE, a_mat(1,lcs), lda, &
!                     umcCPU(lcs,n_cols+1), ubound(umcCPU,dim=1), ONE, vmrCPU(1,n_cols+1), ubound(vmrCPU,dim=1))
!          call obj%timer%stop("blas")
!      enddo
!
!    endif ! (l_cols>0 .and. l_rows>0)
!#endif /* if 0 */

    !Code for Algorithm 4

    ! n_way is actually a branch for the number of OpenMP threads
    n_way = 1
#ifdef WITH_OPENMP_TRADITIONAL

    n_way = max_threads
    if (n_way > 1) then
      !$omp parallel do &
      !$omp default(none) &
      !$omp private(i) &
      !$omp shared(l_cols_tile, l_cols, umcCPU, n_cols)
      do i=1,min(l_cols_tile, l_cols)
        umcCPU(i,1:n_cols) = 0.0_rck
      enddo

      !$omp parallel do &
      !$omp default(none) &
      !$omp private(i) &
      !$omp shared(l_rows, vmrCPU, n_cols)
      do i=1,l_rows
        vmrCPU(i,n_cols+1:2*n_cols) = 0.0_rck
      enddo

      if (l_cols>0 .and. l_rows>0) then

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
        !$omp  parallel do schedule(static,1) &
        !$omp  default(none) &
        !$omp  private(i, lcs, lce, lrs, lre) &
        !$omp  shared(istep, nbw, tile_size, obj, l_cols, l_cols_tile, l_rows, isSkewsymmetric, &
        !$omp&       n_cols, l_rows_tile, umcCPU, vmrCPU, a_mat)
        do i=0,(istep*nbw-1)/tile_size
          lcs = i*l_cols_tile+1                   ! local column start
          lce = min(l_cols, (i+1)*l_cols_tile)    ! local column end

          lrs = i*l_rows_tile+1                   ! local row start
          lre = min(l_rows, (i+1)*l_rows_tile)    ! local row end

          !C1 += [A11 A12] [B1
          !                 B2]
          if ( lre > lrs .and. l_cols > lcs ) then
            call obj%timer%start("blas")
            if (isSkewsymmetric) then
              call PRECISION_GEMM('N', 'N', int(lre-lrs+1,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), &
                                  int(l_cols-lcs+1,kind=BLAS_KIND),                                    &
                                  -ONE, a_mat(lrs,lcs), int(ubound(a_mat,dim=1),kind=BLAS_KIND),       &
                                  umcCPU(lcs,n_cols+1), int(ubound(umcCPU,dim=1),kind=BLAS_KIND),      &
                                  ZERO, vmrCPU(lrs,n_cols+1), int(ubound(vmrCPU,dim=1),kind=BLAS_KIND) )
            else
              call PRECISION_GEMM('N', 'N', int(lre-lrs+1,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), &
                                  int(l_cols-lcs+1,kind=BLAS_KIND),                                    &
                                  ONE, a_mat(lrs,lcs), int(ubound(a_mat,dim=1),kind=BLAS_KIND),        &
                                  umcCPU(lcs,n_cols+1), int(ubound(umcCPU,dim=1),kind=BLAS_KIND),      &
                                  ZERO, vmrCPU(lrs,n_cols+1), int(ubound(vmrCPU,dim=1),kind=BLAS_KIND) )

            endif
            call obj%timer%stop("blas")
          endif

          ! C1 += A10' B0
          if ( lce > lcs .and. i > 0 ) then
            call obj%timer%start("blas")
            call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',     &
                                int(lce-lcs+1,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(lrs-1,kind=BLAS_KIND), &
                                 ONE, a_mat(1,lcs), int(ubound(a_mat,dim=1),kind=BLAS_KIND),      &
                                 vmrCPU(1,1), int(ubound(vmrCPU,dim=1),kind=BLAS_KIND),   &
                                 ZERO, umcCPU(lcs,1), int(ubound(umcCPU,dim=1),kind=BLAS_KIND) )
            call obj%timer%stop("blas")
          endif
        enddo
      endif ! l_cols>0 .and. l_rows>0

    else ! n_way > 1
#endif /* WITH_OPENMP_TRADITIONAL */

      if (.not. useGPU .or. useIntelGPU) then
        umcCPU(1:l_cols,1:n_cols) = 0.0_rck
        vmrCPU(1:l_rows,n_cols+1:2*n_cols) = 0.0_rck
      endif ! useGPU

      if (l_cols>0 .and. l_rows>0) then

        if (useGPU .and. .not.(useIntelGPU)) then
          successGPU = gpu_memset(vmr_dev+cur_l_rows*n_cols*size_of_datatype, &
                        0, cur_l_rows*n_cols*size_of_datatype)
          check_memset_gpu("bandred: vmr_dev", successGPU)

          successGPU = gpu_memcpy(vmr_dev, int(loc(vmrGPU(1)),kind=c_intptr_t), &
                        cur_l_rows*n_cols*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("bandred: vmrGPU -> vmr_dev", successGPU)

          successGPU = gpu_memset(umc_dev, 0, l_cols*n_cols*size_of_datatype)
          check_memset_gpu("bandred: umc_dev", successGPU)

          successGPU = gpu_memcpy(umc_dev+l_cols*n_cols*size_of_datatype, &
                        int(loc(umcGPU(1+l_cols*n_cols)),kind=c_intptr_t), &
                        (umc_size-l_cols*n_cols)*size_of_datatype, &
                        gpuMemcpyHostToDevice)
          check_memcpy_gpu("bandred: umcGPU -> umc_dev", successGPU)
        endif ! useGPU

        do i=0,(istep*nbw-1)/tile_size

          lcs = i*l_cols_tile+1
          lce = min(l_cols,(i+1)*l_cols_tile)
          if (lce<lcs) cycle
          lre = min(l_rows,(i+1)*l_rows_tile)

          if (useGPU) then
            if (useIntelGPU) then
              call obj%timer%start("mkl_offload")
#if 0
              call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',       &
                                  int(lce-lcs+1,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(lre,kind=BLAS_KIND), &
                                  ONE, a_mat(1,lcs), int(ubound(a_mat,dim=1),kind=BLAS_KIND), &
                                  vmrCPU, int(ubound(vmrCPU,dim=1),kind=BLAS_KIND), ONE, umcCPU(lcs,1), &
                                  int(ubound(umcCPU,dim=1),kind=BLAS_KIND) )
#endif
#ifdef WITH_INTEL_GPU_VERSION
              call mkl_offload_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',       &
                                  int(lce-lcs+1,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(lre,kind=BLAS_KIND), &
                                  ONE, a_mat(1,lcs), int(ubound(a_mat,dim=1),kind=BLAS_KIND), &
                                  vmrCPU, int(ubound(vmrCPU,dim=1),kind=BLAS_KIND), ONE, umcCPU(lcs,1), &
                                  int(ubound(umcCPU,dim=1),kind=BLAS_KIND) )
#endif
              call obj%timer%stop("mkl_offload")
              if (i==0) cycle
              lre = min(l_rows,i*l_rows_tile)
              call obj%timer%start("mkl_offload")

              if (isSkewsymmetric) then
#if 0
                call PRECISION_GEMM('N', 'N', int(lre,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(lce-lcs+1,kind=BLAS_KIND), &
                                    -ONE, a_mat(1,lcs), int(lda,kind=BLAS_KIND),                                                   &
                                    umcCPU(lcs,n_cols+1), int(ubound(umcCPU,dim=1),kind=BLAS_KIND), ONE,                          &
                                    vmrCPU(1,n_cols+1), int(ubound(vmrCPU,dim=1), kind=BLAS_KIND) )
#endif
#ifdef WITH_INTEL_GPU_VERSION
                call mkl_offload_PRECISION_GEMM('N', 'N', int(lre,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), &
                        int(lce-lcs+1,kind=BLAS_KIND), &
                                    -ONE, a_mat(1,lcs), int(lda,kind=BLAS_KIND),                                                   &
                                    umcCPU(lcs,n_cols+1), int(ubound(umcCPU,dim=1),kind=BLAS_KIND), ONE,                          &
                                    vmrCPU(1,n_cols+1), int(ubound(vmrCPU,dim=1), kind=BLAS_KIND) )
#endif

              else
#if 0
                call PRECISION_GEMM('N', 'N', int(lre,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(lce-lcs+1,kind=BLAS_KIND), &
                                    ONE, a_mat(1,lcs), int(lda,kind=BLAS_KIND),                                                   &
                                    umcCPU(lcs,n_cols+1), int(ubound(umcCPU,dim=1),kind=BLAS_KIND), ONE,                          &
                                    vmrCPU(1,n_cols+1), int(ubound(vmrCPU,dim=1), kind=BLAS_KIND) )
#endif
#ifdef WITH_INTEL_GPU_VERSION
                call mkl_offload_PRECISION_GEMM('N', 'N', int(lre,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), &
                        int(lce-lcs+1,kind=BLAS_KIND), &
                                    ONE, a_mat(1,lcs), int(lda,kind=BLAS_KIND),                                                   &
                                    umcCPU(lcs,n_cols+1), int(ubound(umcCPU,dim=1),kind=BLAS_KIND), ONE,                          &
                                    vmrCPU(1,n_cols+1), int(ubound(vmrCPU,dim=1), kind=BLAS_KIND) )
#endif
              endif
              call obj%timer%stop("mkl_offload")

            else
              call obj%timer%start("gpublas")
              call gpublas_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',                   &
                                         lce-lcs+1, n_cols, lre,     &
                                         ONE, (a_dev + ((lcs-1)*lda* &
                                         size_of_datatype)),         &
                                         lda, vmr_dev,cur_l_rows,    &
                                         ONE, (umc_dev+ (lcs-1)*     &
                                             size_of_datatype),      &
                                         cur_l_cols)

              call obj%timer%stop("gpublas")

              if(i==0) cycle
              call obj%timer%start("gpublas")

              lre = min(l_rows,i*l_rows_tile)
              if (isSkewsymmetric) then
                call gpublas_PRECISION_GEMM('N', 'N', lre,n_cols, lce-lcs+1, -ONE, &
                              (a_dev+ ((lcs-1)*lda*                 &
                                    size_of_datatype)),             &
                         lda, (umc_dev+(cur_l_cols * n_cols+lcs-1)* &
                                size_of_datatype),              &
                                cur_l_cols, ONE, (vmr_dev+(cur_l_rows * n_cols)* &
                              size_of_datatype),              &
                                cur_l_rows)
              else
                call gpublas_PRECISION_GEMM('N', 'N', lre,n_cols, lce-lcs+1, ONE, &
                                            (a_dev+ ((lcs-1)*lda*                 &
                                                  size_of_datatype)),             &
                                       lda, (umc_dev+(cur_l_cols * n_cols+lcs-1)* &
                                              size_of_datatype),              &
                                              cur_l_cols, ONE, (vmr_dev+(cur_l_rows * n_cols)* &
                                            size_of_datatype),              &
                                              cur_l_rows)
              endif
              call obj%timer%stop("gpublas")
            endif
          else ! useGPU

            call obj%timer%start("blas")
            call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',       &
                                int(lce-lcs+1,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(lre,kind=BLAS_KIND), &
                                ONE, a_mat(1,lcs), int(ubound(a_mat,dim=1),kind=BLAS_KIND), &
                                vmrCPU, int(ubound(vmrCPU,dim=1),kind=BLAS_KIND), ONE, umcCPU(lcs,1), &
                                int(ubound(umcCPU,dim=1),kind=BLAS_KIND) )
            call obj%timer%stop("blas")
            if (i==0) cycle
            lre = min(l_rows,i*l_rows_tile)
            call obj%timer%start("blas")

            if (isSkewsymmetric) then
              call PRECISION_GEMM('N', 'N', int(lre,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(lce-lcs+1,kind=BLAS_KIND), &
                                  -ONE, a_mat(1,lcs), int(lda,kind=BLAS_KIND),                                                   &
                                  umcCPU(lcs,n_cols+1), int(ubound(umcCPU,dim=1),kind=BLAS_KIND), ONE,                          &
                                  vmrCPU(1,n_cols+1), int(ubound(vmrCPU,dim=1), kind=BLAS_KIND) )

            else
              call PRECISION_GEMM('N', 'N', int(lre,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(lce-lcs+1,kind=BLAS_KIND), &
                                  ONE, a_mat(1,lcs), int(lda,kind=BLAS_KIND),                                                   &
                                  umcCPU(lcs,n_cols+1), int(ubound(umcCPU,dim=1),kind=BLAS_KIND), ONE,                          &
                                  vmrCPU(1,n_cols+1), int(ubound(vmrCPU,dim=1), kind=BLAS_KIND) )
            endif
            call obj%timer%stop("blas")
          endif ! useGPU
        enddo ! i=0,(istep*nbw-1)/tile_size

        if (useGPU .and. .not.(useIntelGPU)) then
          if (tile_size < istep*nbw .or. n_way > 1) then
            successGPU = gpu_memcpy(int(loc(vmrGPU(1+cur_l_rows*n_cols)),kind=c_intptr_t), &
                          vmr_dev+cur_l_rows*n_cols*size_of_datatype, &
                          (vmr_size-cur_l_rows*n_cols)*size_of_datatype, gpuMemcpyDeviceToHost)
            check_memcpy_gpu("bandred: vmr_dev -> vmrGPU", successGPU)
          endif

          successGPU = gpu_memcpy(int(loc(umcGPU(1)),kind=c_intptr_t), &
                        umc_dev, l_cols*n_cols*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("bandred: umc_dev -> umcGPU", successGPU)
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
        if (useIntelGPU) then
          call elpa_reduce_add_vectors_&
          &MATH_DATATYPE&
          &_&
          &PRECISION &
                                      (obj, vmrCPU(1,n_cols+1),ubound(vmrCPU,dim=1),mpi_comm_rows, &
                                       umcCPU, ubound(umcCPU,dim=1), mpi_comm_cols, &
                                      istep*nbw, n_cols, nblk, max_threads)

        else

          call elpa_reduce_add_vectors_&
               &MATH_DATATYPE&
               &_&
               &PRECISION &
                               (obj, vmrGPU(cur_l_rows * n_cols + 1:),cur_l_rows,  &
                                mpi_comm_rows, umcGPU,                            &
                                cur_l_cols, mpi_comm_cols, istep*nbw, n_cols, nblk, max_threads)
        endif
      else ! useGPU

        call elpa_reduce_add_vectors_&
        &MATH_DATATYPE&
        &_&
        &PRECISION &
                                         (obj, vmrCPU(1,n_cols+1),ubound(vmrCPU,dim=1),mpi_comm_rows, &
                                          umcCPU, ubound(umcCPU,dim=1), mpi_comm_cols, &
                                          istep*nbw, n_cols, nblk, max_threads)
      endif ! useGPU
    endif ! tile_size < istep*nbw .or. n_way > 1

    if (l_cols>0) then

      if (useGPU) then
        if (useIntelGPU) then
          allocate(tmpCPU(l_cols,n_cols), stat=istat, errmsg=errorMessage)
          check_allocate("bandred: tmpCPU", istat, errorMessage)

#ifdef WITH_MPI
          if (wantDebug) call obj%timer%start("mpi_communication")
          call mpi_allreduce(umcCPU, tmpCPU, int(l_cols*n_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,    &
                           MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
          umcCPU(1:l_cols,1:n_cols) = tmpCPU(1:l_cols,1:n_cols)
          if (wantDebug) call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */

          deallocate(tmpCPU, stat=istat, errmsg=errorMessage)
          check_deallocate("bandred: tmpCPU", istat, errorMessage)
        else
#ifdef WITH_MPI
          allocate(tmpGPU(l_cols * n_cols), stat=istat, errmsg=errorMessage)
          check_allocate("bandred: tmpGPU", istat, errorMessage)

          if (wantDebug) call obj%timer%start("mpi_communication")

          call mpi_allreduce(umcGPU, tmpGPU, int(l_cols*n_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                           MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), ierr)

          umcGPU(1 : l_cols * n_cols) = tmpGPU(1 : l_cols * n_cols)
          if (wantDebug) call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */

          if (allocated(tmpGPU)) then
            deallocate(tmpGPU, stat=istat, errmsg=errorMessage)
            check_deallocate("bandred: tmpGPU", istat, errorMessage)
          endif
        endif

      else ! useGPU

        allocate(tmpCPU(l_cols,n_cols), stat=istat, errmsg=errorMessage)
        check_allocate("bandred: tmpCPU", istat, errorMessage)

#ifdef WITH_MPI
        if (wantDebug) call obj%timer%start("mpi_communication")
        call mpi_allreduce(umcCPU, tmpCPU, int(l_cols*n_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,    &
                           MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
        umcCPU(1:l_cols,1:n_cols) = tmpCPU(1:l_cols,1:n_cols)
        if (wantDebug) call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */

        deallocate(tmpCPU, stat=istat, errmsg=errorMessage)
        check_deallocate("bandred: tmpCPU", istat, errorMessage)
      endif ! useGPU
    endif ! l_cols > 0

    ! U = U * Tmat**T

    if (useGPU) then
      if (useIntelGPU) then
        call obj%timer%start("mkl_offload")

        call PRECISION_TRMM('Right', 'Upper', BLAS_TRANS_OR_CONJ, 'Nonunit',     &
                          int(l_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), ONE, tmat(1,1,istep), &
                          int(ubound(tmat,dim=1),kind=BLAS_KIND), &
                          umcCPU, int(ubound(umcCPU,dim=1),kind=BLAS_KIND))

        ! VAV = Tmat * V**T * A * V * Tmat**T = (U*Tmat**T)**T * V * Tmat**T

        call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',              &
                          int(n_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), &
                          ONE, umcCPU, int(ubound(umcCPU,dim=1),kind=BLAS_KIND), umcCPU(1,n_cols+1), &
                          int(ubound(umcCPU,dim=1),kind=BLAs_KIND), ZERO, vav, int(ubound(vav,dim=1),kind=BLAS_KIND))

        call PRECISION_TRMM('Right', 'Upper', BLAS_TRANS_OR_CONJ, 'Nonunit',    &
                          int(n_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), ONE, tmat(1,1,istep),    &
                          int(ubound(tmat,dim=1),kind=BLAS_KIND), vav, int(ubound(vav,dim=1),kind=BLAS_KIND) )
        call obj%timer%stop("mkl_offload")
#ifdef WITH_INTEL_GPU_VERSION
#if 0
        call obj%timer%start("mkl_offload")

        call mkl_offload_PRECISION_TRMM('Right', 'Upper', BLAS_TRANS_OR_CONJ, 'Nonunit',     &
                          int(l_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), ONE, tmat(1,1,istep), &
                          int(ubound(tmat,dim=1),kind=BLAS_KIND), &
                          umcCPU, int(ubound(umcCPU,dim=1),kind=BLAS_KIND))

        ! VAV = Tmat * V**T * A * V * Tmat**T = (U*Tmat**T)**T * V * Tmat**T

        call mkl_offload_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',              &
                          int(n_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), &
                          ONE, umcCPU, int(ubound(umcCPU,dim=1),kind=BLAS_KIND), umcCPU(1,n_cols+1), &
                          int(ubound(umcCPU,dim=1),kind=BLAs_KIND), ZERO, vav, int(ubound(vav,dim=1),kind=BLAS_KIND))

        call mkl_offload_PRECISION_TRMM('Right', 'Upper', BLAS_TRANS_OR_CONJ, 'Nonunit',    &
                          int(n_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), ONE, tmat(1,1,istep),    &
                          int(ubound(tmat,dim=1),kind=BLAS_KIND), vav, int(ubound(vav,dim=1),kind=BLAS_KIND) )
         call obj%timer%stop("mkl_offload")
#endif
#endif

      else
        successGPU = gpu_memcpy(umc_dev, int(loc(umcGPU(1)),kind=c_intptr_t), &
                      l_cols*n_cols*size_of_datatype, gpuMemcpyHostToDevice)
        check_memcpy_gpu("bandred: umcGPU -> umc_dev ", successGPU)

        successGPU = gpu_memcpy(tmat_dev,int(loc(tmat(1,1,istep)),kind=c_intptr_t), &
                      nbw*nbw*size_of_datatype,gpuMemcpyHostToDevice)
        check_memcpy_gpu("bandred: tmat -> tmat_dev ", successGPU)

        call obj%timer%start("gpublas")
        call gpublas_PRECISION_TRMM('Right', 'Upper', BLAS_TRANS_OR_CONJ, 'Nonunit',  &
                            l_cols, n_cols, ONE, tmat_dev, nbw, umc_dev, cur_l_cols)
        call obj%timer%stop("gpublas")

        ! VAV = Tmat * V**T * A * V * Tmat**T = (U*Tmat**T)**T * V * Tmat**T
        call obj%timer%start("gpublas")
        call gpublas_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',             &
                                 n_cols, n_cols, l_cols, ONE, umc_dev, cur_l_cols, &
                                 (umc_dev+(cur_l_cols * n_cols )*size_of_datatype),cur_l_cols, &
                                 ZERO, vav_dev, nbw)

        call gpublas_PRECISION_TRMM('Right', 'Upper', BLAS_TRANS_OR_CONJ, 'Nonunit',    &
           n_cols, n_cols, ONE, tmat_dev, nbw, vav_dev, nbw)
        call obj%timer%stop("gpublas")

        successGPU = gpu_memcpy(int(loc(vav),kind=c_intptr_t), &
                    vav_dev, nbw*nbw*size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("bandred: vav_dev -> vav ", successGPU)
      endif
    else ! useGPU

      call obj%timer%start("blas")

      call PRECISION_TRMM('Right', 'Upper', BLAS_TRANS_OR_CONJ, 'Nonunit',     &
                          int(l_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), ONE, tmat(1,1,istep), &
                          int(ubound(tmat,dim=1),kind=BLAS_KIND), &
                          umcCPU, int(ubound(umcCPU,dim=1),kind=BLAS_KIND))

      ! VAV = Tmat * V**T * A * V * Tmat**T = (U*Tmat**T)**T * V * Tmat**T

      call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N',              &
                          int(n_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), &
                          ONE, umcCPU, int(ubound(umcCPU,dim=1),kind=BLAS_KIND), umcCPU(1,n_cols+1), &
                          int(ubound(umcCPU,dim=1),kind=BLAs_KIND), ZERO, vav, int(ubound(vav,dim=1),kind=BLAS_KIND))

      call PRECISION_TRMM('Right', 'Upper', BLAS_TRANS_OR_CONJ, 'Nonunit',    &
                          int(n_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), ONE, tmat(1,1,istep),    &
                          int(ubound(tmat,dim=1),kind=BLAS_KIND), vav, int(ubound(vav,dim=1),kind=BLAS_KIND) )
      call obj%timer%stop("blas")

    endif ! useGPU

#if REALCASE == 1
#ifdef HAVE_SKEWSYMMETRIC
    if (isSkewsymmetric) then
      call ssymm_matrix_allreduce_&
      &PRECISION &
      (obj, n_cols,vav, nbw, nbw ,mpi_comm_cols)
    else
#endif
      call symm_matrix_allreduce_&
      &PRECISION &
      (obj, n_cols,vav, nbw, nbw ,mpi_comm_cols)
#ifdef HAVE_SKEWSYMMETRIC
    endif
#endif
#endif /* REALCASE */
#if COMPLEXCASE == 1
    call herm_matrix_allreduce_&
         &PRECISION &
         (obj, n_cols,vav, nbw, nbw ,mpi_comm_cols)
#endif

    if (useGPU .and. .not.(useIntelGPU)) then
      successGPU = gpu_memcpy(vav_dev, int(loc(vav),kind=c_intptr_t), &
                       nbw*nbw*size_of_datatype, gpuMemcpyHostToDevice)
      check_memcpy_gpu("bandred: vav -> vav_dev ", successGPU)
    endif
    !if (useIntelGPU) then
      ! needed later
    !endif


    ! U = U - 0.5 * V * VAV

    if (useGPU) then
      if (useIntelGPU) then
      call obj%timer%start("mkl_offload")
#if REALCASE == 1
      if (isSkewsymmetric) then
        call PRECISION_GEMM('N', 'N', int(l_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND),     &
                            0.5_rk, umcCPU(1,n_cols+1), int(ubound(umcCPU,dim=1),kind=BLAS_KIND), vav,                        &
                            int(ubound(vav,dim=1),kind=BLAS_KIND), ONE, umcCPU, int(ubound(umcCPU,dim=1),kind=BLAS_KIND) )
      else
        call PRECISION_GEMM('N', 'N', int(l_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND),     &
                            -0.5_rk, umcCPU(1,n_cols+1), int(ubound(umcCPU,dim=1),kind=BLAS_KIND), vav,                       &
                            int(ubound(vav,dim=1),kind=BLAS_KIND), ONE, umcCPU, int(ubound(umcCPU,dim=1),kind=BLAS_KIND) )
      endif
#endif
#if COMPLEXCASE == 1
      call PRECISION_GEMM('N', 'N', int(l_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND),     &
                         (-0.5_rk, 0.0_rk),     &
                         umcCPU(1,n_cols+1), int(ubound(umcCPU,dim=1),kind=BLAS_KIND), vav, &
                         int(ubound(vav,dim=1),kind=BLAS_KIND), ONE, umcCPU, int(ubound(umcCPU,dim=1),kind=BLAS_KIND))
#endif

      call obj%timer%stop("mkl_offload")

      ! Transpose umc -> umr (stored in vmr, second half)
      if (isSkewsymmetric) then
        call elpa_transpose_vectors_ss_&
          &MATH_DATATYPE&
        &_&
        &PRECISION &
                                 (obj, umcCPU, ubound(umcCPU,dim=1), mpi_comm_cols, &
                                        vmrCPU(1,n_cols+1), ubound(vmrCPU,dim=1), mpi_comm_rows, &
                                        1, istep*nbw, n_cols, nblk, max_threads)
      else
       call elpa_transpose_vectors_&
       &MATH_DATATYPE&
       &_&
       &PRECISION &
                                (obj, umcCPU, ubound(umcCPU,dim=1), mpi_comm_cols, &
                                          vmrCPU(1,n_cols+1), ubound(vmrCPU,dim=1), mpi_comm_rows, &
                                          1, istep*nbw, n_cols, nblk, max_threads)
      endif

      else
        call obj%timer%start("gpublas")
        if (isSkewsymmetric) then
          call gpublas_PRECISION_GEMM('N', 'N', l_cols, n_cols, n_cols,&
#if REALCASE == 1
                                    0.5_rk,                      &
#endif
#if COMPLEXCASE == 1
                                    (0.5_rk, 0.0_rk), &
#endif
                                    (umc_dev+(cur_l_cols * n_cols )* &
                                    size_of_datatype),   &
                                    cur_l_cols, vav_dev,nbw,        &
                                    ONE, umc_dev, cur_l_cols)
        else
          call gpublas_PRECISION_GEMM('N', 'N', l_cols, n_cols, n_cols,&
#if REALCASE == 1
                                   -0.5_rk,                      &
#endif
#if COMPLEXCASE == 1
                                   (-0.5_rk, 0.0_rk), &
#endif
                                   (umc_dev+(cur_l_cols * n_cols )* &
                                   size_of_datatype),   &
                                   cur_l_cols, vav_dev,nbw,        &
                                   ONE, umc_dev, cur_l_cols)
        endif
        call obj%timer%stop("gpublas")

        successGPU = gpu_memcpy(int(loc(umcGPU(1)),kind=c_intptr_t), &
                    umc_dev, umc_size*size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("bandred: umc_dev -> umcGPU ", successGPU)

        ! Transpose umc -> umr (stored in vmr, second half)
        if (isSkewsymmetric) then
          call elpa_transpose_vectors_ss_&
             &MATH_DATATYPE&
             &_&
             &PRECISION &
                         (obj, umcGPU(:), cur_l_cols, mpi_comm_cols, &
                          vmrGPU(cur_l_rows * n_cols + 1:), cur_l_rows, mpi_comm_rows, &
                          1, istep*nbw, n_cols, nblk, max_threads)
        else
          call elpa_transpose_vectors_&
             &MATH_DATATYPE&
             &_&
             &PRECISION &
                         (obj, umcGPU, cur_l_cols, mpi_comm_cols, &
                          vmrGPU(cur_l_rows * n_cols + 1:), cur_l_rows, mpi_comm_rows, &
                          1, istep*nbw, n_cols, nblk, max_threads)
        endif

        successGPU = gpu_memcpy(vmr_dev+cur_l_rows*n_cols*size_of_datatype, &
                    int(loc(vmrGPU(1+cur_l_rows*n_cols)),kind=c_intptr_t), &
                    (vmr_size-cur_l_rows*n_cols)*size_of_datatype, gpuMemcpyHostToDevice)
        check_memcpy_gpu("bandred: vmr -> vmrGPU ", successGPU)
      endif

    else ! useGPU
      call obj%timer%start("blas")
#if REALCASE == 1
      if (isSkewsymmetric) then
        call PRECISION_GEMM('N', 'N', int(l_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND),     &
                            0.5_rk, umcCPU(1,n_cols+1), int(ubound(umcCPU,dim=1),kind=BLAS_KIND), vav,                        &
                            int(ubound(vav,dim=1),kind=BLAS_KIND), ONE, umcCPU, int(ubound(umcCPU,dim=1),kind=BLAS_KIND) )
      else
        call PRECISION_GEMM('N', 'N', int(l_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND),     &
                            -0.5_rk, umcCPU(1,n_cols+1), int(ubound(umcCPU,dim=1),kind=BLAS_KIND), vav,                       &
                            int(ubound(vav,dim=1),kind=BLAS_KIND), ONE, umcCPU, int(ubound(umcCPU,dim=1),kind=BLAS_KIND) )
      endif
#endif
#if COMPLEXCASE == 1
      call PRECISION_GEMM('N', 'N', int(l_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND), int(n_cols,kind=BLAS_KIND),     &
                         (-0.5_rk, 0.0_rk),     &
                         umcCPU(1,n_cols+1), int(ubound(umcCPU,dim=1),kind=BLAS_KIND), vav, &
                         int(ubound(vav,dim=1),kind=BLAS_KIND), ONE, umcCPU, int(ubound(umcCPU,dim=1),kind=BLAS_KIND))
#endif

      call obj%timer%stop("blas")

      ! Transpose umc -> umr (stored in vmr, second half)
      if (isSkewsymmetric) then
        call elpa_transpose_vectors_ss_&
          &MATH_DATATYPE&
        &_&
        &PRECISION &
                                 (obj, umcCPU, ubound(umcCPU,dim=1), mpi_comm_cols, &
                                        vmrCPU(1,n_cols+1), ubound(vmrCPU,dim=1), mpi_comm_rows, &
                                        1, istep*nbw, n_cols, nblk, max_threads)
      else
       call elpa_transpose_vectors_&
       &MATH_DATATYPE&
       &_&
       &PRECISION &
                                (obj, umcCPU, ubound(umcCPU,dim=1), mpi_comm_cols, &
                                          vmrCPU(1,n_cols+1), ubound(vmrCPU,dim=1), mpi_comm_rows, &
                                          1, istep*nbw, n_cols, nblk, max_threads)
      endif
    endif  ! useGPU

    ! A = A - V*U**T - U*V**T

#ifdef WITH_OPENMP_TRADITIONAL
    !$omp parallel &
    !$omp default(none) &
    !$omp private( ii, i, lcs, lce, lre, n_way, m_way, m_id, n_id, work_per_thread, mystart, myend  ) &
    !$omp shared(a_mat, n_threads, istep, tile_size, nbw, n_cols, obj, vmrcpu, l_cols_tile, l_rows, l_rows_tile, &
    !$omp&       umccpu, l_cols, a_dev, vmr_dev, useGPU, cur_l_rows, umc_dev, cur_l_cols, lda, useIntelGPU )
    n_threads = omp_get_num_threads()

    if (mod(n_threads, 2) == 0) then
      n_way = 2
    else
      n_way = 1
    endif

    m_way = n_threads / n_way

    m_id = mod(omp_get_thread_num(),  m_way)
    n_id = omp_get_thread_num() / m_way

    do ii=n_id*tile_size,(istep*nbw-1),tile_size*n_way
      i = ii / tile_size
      lcs = i*l_cols_tile+1
      lce = min(l_cols,(i+1)*l_cols_tile)
      lre = min(l_rows,(i+1)*l_rows_tile)
      if (lce<lcs .or. lre<1) cycle

      !Figure out this thread's range
      work_per_thread = lre / m_way
      if (work_per_thread * m_way < lre) work_per_thread = work_per_thread + 1
      mystart = m_id * work_per_thread + 1
      myend   = mystart + work_per_thread - 1
      if ( myend > lre ) myend = lre
      if ( myend-mystart+1 < 1) cycle
      if (useGPU) then
        if (useIntelGPU) then
          call obj%timer%start("mkl_offload")
          call PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, int(myend-mystart+1,kind=BLAS_KIND), &
                            int(lce-lcs+1,kind=BLAS_KIND), int(2*n_cols,kind=BLAS_KIND), -ONE, &
                            vmrCPU(mystart, 1), int(ubound(vmrCPU,1),kind=BLAS_KIND), &
                            umcCPU(lcs,1), int(ubound(umcCPU,1),kind=BLAS_KIND), &
                            ONE, a_mat(mystart,lcs), int(ubound(a_mat,1),kind=BLAS_KIND) )

          call obj%timer%stop("mkl_offload")

        else
          if (n_way .gt. 1) then
            print *,"error more than 1 openmp thread used in GPU part of elpa2_bandred"
            print *,"this should never happen"
            stop
          endif
          call obj%timer%start("gpublas")

          call gpublas_PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, myend-mystart+1,    &
                                   lce-lcs+1, 2*n_cols, -ONE, &
                                   vmr_dev, cur_l_rows, (umc_dev +(lcs-1)*  &
                                   size_of_datatype), &
                                   cur_l_cols, ONE, (a_dev+(lcs-1)*lda* &
                                   size_of_datatype), lda)
          call obj%timer%stop("gpublas")
        endif
      else
        call obj%timer%start("blas")
        call PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, int(myend-mystart+1,kind=BLAS_KIND), &
                            int(lce-lcs+1,kind=BLAS_KIND), int(2*n_cols,kind=BLAS_KIND), -ONE, &
                            vmrCPU(mystart, 1), int(ubound(vmrCPU,1),kind=BLAS_KIND), &
                            umcCPU(lcs,1), int(ubound(umcCPU,1),kind=BLAS_KIND), &
                            ONE, a_mat(mystart,lcs), int(ubound(a_mat,1),kind=BLAS_KIND) )
        call obj%timer%stop("blas")
      endif
    enddo
    !$omp end parallel

#else /* WITH_OPENMP_TRADITIONAL */

    do i=0,(istep*nbw-1)/tile_size
      lcs = i*l_cols_tile+1
      lce = min(l_cols,(i+1)*l_cols_tile)
      lre = min(l_rows,(i+1)*l_rows_tile)
      if (lce<lcs .or. lre<1) cycle

      if (useGPU) then
        if (useIntelGPU) then
          call obj%timer%start("mkl_offload")
          call PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, int(lre,kind=BLAS_KIND),int(lce-lcs+1,kind=BLAS_KIND), &
                              int(2*n_cols,kind=BLAS_KIND), &
                              -ONE, &
                              vmrCPU, int(ubound(vmrCPU,dim=1),kind=BLAS_KIND), umcCPU(lcs,1), &
                              int(ubound(umcCPU,dim=1),kind=BLAS_KIND), &
                              ONE, a_mat(1,lcs), int(lda,kind=BLAS_KIND))
          call obj%timer%stop("mkl_offload")

        else
          call obj%timer%start("gpublas")

          call gpublas_PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ,     &
                                     lre, lce-lcs+1, 2*n_cols, -ONE, &
                                     vmr_dev, cur_l_rows, (umc_dev +(lcs-1)*  &
                                     size_of_datatype), &
                                     cur_l_cols, ONE, (a_dev+(lcs-1)*lda* &
                                     size_of_datatype), lda)
          call obj%timer%stop("gpublas")
        endif
      else ! useGPU

        call obj%timer%start("blas")
        call PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, int(lre,kind=BLAS_KIND),int(lce-lcs+1,kind=BLAS_KIND), &
                            int(2*n_cols,kind=BLAS_KIND), &
                            -ONE, &
                            vmrCPU, int(ubound(vmrCPU,dim=1),kind=BLAS_KIND), umcCPU(lcs,1), &
                            int(ubound(umcCPU,dim=1),kind=BLAS_KIND), &
                            ONE, a_mat(1,lcs), int(lda,kind=BLAS_KIND))
        call obj%timer%stop("blas")
      endif ! useGPU
    enddo ! i=0,(istep*nbw-1)/tile_size
#endif /* WITH_OPENMP_TRADITIONAL */

    if (.not.(useGPU) .or. useIntelGPU) then
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

  enddo ! istep - loop

  if (useGPU .and. .not.(useIntelGPU)) then
    ! copy a_dev to a_mat
    ! we do it here, since a is needed on the host in the following routine
    ! (band to tridi). Previously, a has been kept on the device and then
    ! copied in redist_band (called from tridiag_band). However, it seems to
    ! be easier to do it here.
    successGPU = gpu_memcpy(int(loc(a_mat),kind=c_intptr_t), &
                  int(a_dev,kind=c_intptr_t), &
                  int(lda*matrixCols* size_of_datatype, kind=c_intptr_t), &
                  gpuMemcpyDeviceToHost)
    check_memcpy_gpu("bandred: a_dev -> a_mat ", successGPU)

    successGPU = gpu_host_unregister(int(loc(a_mat),kind=c_intptr_t))
    check_host_unregister_gpu("bandred: a_mat ", successGPU)

    successGPU = gpu_free(a_dev)
    check_dealloc_gpu("bandred: a_dev ", successGPU)

    successGPU = gpu_free(vav_dev)
    check_dealloc_gpu("bandred: vav_dev ", successGPU)

    successGPU = gpu_free(tmat_dev)
    check_dealloc_gpu("bandred: tmat_dev ", successGPU)

    successGPU = gpu_host_unregister(int(loc(vav),kind=c_intptr_t))
    check_host_unregister_gpu("bandred: vav", successGPU)

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
  endif ! useGPU

  !if (useIntelGPU) then
  !   ! needed later
  !endif

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

end subroutine bandred_&
&MATH_DATATYPE&
&_&
&PRECISION

