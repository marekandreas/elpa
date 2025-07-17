!    Copyright 2024, P. Karpov, MPCDF
!
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

!    This file was written by P. Karpov, MPCDF

#include "../general/sanity.F90"
#include "../general/error_checking.inc"


#undef USE_CCL_PXGEMM
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
#define USE_CCL_PXGEMM
#endif

  use elpa1_compute
  use elpa_mpi
  use precision
  use elpa_abstract_impl
  use, intrinsic :: iso_c_binding
  use elpa_gpu
  use mod_check_for_gpu
  use elpa_blas_interfaces
  use ELPA_utilities, only : local_index, greatest_common_divisor, least_common_multiple, &
                             check_deallocate_f, check_dealloc_gpu_f, &
                             check_host_dealloc_gpu_f, check_alloc_gpu_f, check_host_alloc_gpu_f, &
                             check_host_unregister_gpu_f, check_memcpy_gpu_f, check_allocate_f, &
                             check_host_register_gpu_f, check_alloc, error_unit
  use elpa_pxgemm_transpose
  use elpa_pxgemm_helpers
  use mod_query_gpu_usage
#ifdef WITH_GPU_STREAMS
  use elpa_gpu_util
#endif
#if defined(WITH_NVIDIA_GPU_VERSION) && defined(WITH_NVTX)
  use cuda_functions ! for NVTX labels
#elif defined(WITH_AMD_GPU_VERSION) && defined(WITH_ROCTX)
  use hip_functions  ! for ROCTX labels
#endif
  use elpa_ccl_gpu
  use multiply_a_b_gpu
  use pxgemm_multiply_gpu
  implicit none

#include "../../src/general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout)   :: obj

  character*1                                  :: trans_a, trans_b
  character(1, c_char)                         :: trans_a_cchar, trans_b_cchar
  integer(kind=ik), intent(in)                 :: ldb, ldbCols, ldc, ldcCols
  integer(kind=ik)                             :: lda, ldaCols
  integer(kind=ik)                             :: na, ncb
#ifdef DEVICE_POINTER
  type(c_ptr)                                  :: aDev, bDev, cDev
  MATH_DATATYPE(kind=rck), allocatable         :: a(:,:), b(:,:), c(:,:) ! used for d_ptr and (.not. USE_CCL_PXGEMM)
#else /* DEVICE_POINTER */
#ifdef USE_ASSUMED_SIZE
  MATH_DATATYPE(kind=rck)                      :: a(obj%local_nrows,*), b(ldb,*), c(ldc,*)
#else
  MATH_DATATYPE(kind=rck)                      :: a(obj%local_nrows,obj%local_ncols), b(ldb,ldbCols), c(ldc,ldcCols)
#endif
#endif /* DEVICE_POINTER */
  MATH_DATATYPE(kind=rck), allocatable         :: at_full(:,:), bt_full(:,:)
  MATH_DATATYPE(kind=rck), allocatable         :: buf_send(:,:), buf_recv(:,:), buf_self(:,:), at_col(:,:), bt_row(:,:) ! needed for TT case

  integer(kind=ik)                             :: my_pdir, my_pdir_t, np_dirs, np_dirs_t
  integer(kind=ik)                             :: np_rows_fine, np_cols_fine, np_dirs_fine, np_fine, np_t_fine, np_bc_fine, &
                                                  np_ab_fine, np_ab_t_fine, dnp_ab, dnp_ab_t
  integer(kind=ik)                             :: LCM, nblk_mult_rows, nblk_mult_cols, i_block_loc_fine, j_block_loc_fine
  integer(kind=ik)                             :: nblk_mult_rows_max, nblk_mult_cols_max, nblk_rows_cut, nblk_cols_cut
  integer(kind=ik)                             :: nblk_mult_max, nblk_mult_min
  integer(kind=ik)                             :: l_cols, l_rows
  integer(kind=ik)                             :: l_rows_max, l_cols_max, l_rows_min, l_cols_min
  integer(kind=ik)                             :: l_rows_source, l_cols_source
  integer(kind=ik)                             :: np, nb, nblk_mult, lrs, lre, lcs, lce
  integer(kind=ik)                             :: i_block_loc, j_block_loc
  integer(kind=ik)                             :: np_bc
  integer(kind=ik)                             :: np_t
  integer(kind=ik), allocatable                :: lrs_save(:), lre_save(:)

  logical                                      :: a_transposed, b_transposed
  integer(kind=ik)                             :: a_transposed_int, b_transposed_int

  MATH_DATATYPE(kind=rck)                      :: beta
  integer(kind=ik)                             :: beta_int
  MATH_DATATYPE(kind=rck), pointer, contiguous :: aux_a_full(:,:), aux_b_full(:,:), tmp1_full(:,:)
  logical                                      :: wantDebug
  integer(kind=ik)                             :: istat, debug
  character(200)                               :: errorMessage
  character(20)                                :: gpuString, tnString
  logical                                      :: success, successGPU, successGPU2
  logical                                      :: useGPU
  logical                                      :: isSquareGrid, first_call
  integer(kind=c_int)                          :: numGPU, blocking, SM_count

  ! MPI-related
  integer(kind=MPI_KIND)                       :: mpierr
  integer(kind=ik)                             :: my_prow, my_pcol, np_rows, np_cols, myid
  integer(kind=ik)                             :: mpi_comm_rows, mpi_comm_cols, mpi_comm_all
  integer(kind=ik)                             :: mpi_comm_dirs

  integer(kind=ik)                             :: nblk, nblk_cut, nblk_cut_row, nblk_cut_col
  integer(kind=ik)                             :: error
  integer(kind=c_intptr_t)                     :: aux_a_full_dev, aux_b_full_dev, tmp1_full_dev
  integer(kind=c_intptr_t)                     :: at_col_dev, bt_row_dev
  integer(kind=c_intptr_t)                     :: buf_send_dev, buf_recv_dev, buf_self_dev

  integer(kind=c_intptr_t)                     :: a_dev
  integer(kind=c_intptr_t)                     :: b_dev
  integer(kind=c_intptr_t)                     :: c_dev

  integer(kind=c_intptr_t)                     :: num, num_a, num_b, num_r, num_c
  integer(kind=c_intptr_t)                     :: aux_off, b_off
  integer(kind=c_intptr_t), parameter          :: size_of_datatype = size_of_&
                                                                    &PRECISION&
                                                                    &_&
                                                                    &MATH_DATATYPE

  integer(kind=c_intptr_t)                     :: gpuHandle, my_stream
  integer(kind=c_int)                          :: gpu_pxgemm_multiply

  logical                                      :: useCCL
  integer(kind=c_intptr_t)                     :: ccl_comm_rows, ccl_comm_cols, ccl_comm_all, ccl_comm_dirs
  integer(kind=c_int)                          :: cclDataType
  integer(kind=ik)                             :: k_datatype

  NVTX_RANGE_PUSH("elpa_pxgemm_multiply")

  success = .true.
  useGPU = .false.

  a_transposed = .false.
  b_transposed = .false.
  a_transposed_int = 0
  b_transposed_int = 0
  trans_a_cchar = 'N'
  trans_b_cchar = 'N'
  if (trans_a=='t' .or. trans_a=='T' .or. trans_a=='c' .or. trans_a=='C') then
    a_transposed = .true.
    a_transposed_int = 1
    trans_a_cchar = BLAS_TRANS_OR_CONJ
  endif
  if (trans_b=='t' .or. trans_b=='T' .or. trans_b=='c' .or. trans_b=='C') then
    b_transposed = .true.
    b_transposed_int = 1
    trans_b_cchar = BLAS_TRANS_OR_CONJ
  endif

  call obj%get("debug", debug, error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "elpa_pxgemm_multiply: Problem getting option for debug settings. Aborting..."
    success = .false.
    return
  endif
  if (debug == 1) then
    wantDebug = .true.
  else
    wantDebug = .false.
  endif

#if defined(DEVICE_POINTER)
  useGPU = .true.
#else /* DEVICE_POINTER */
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
  if (.not.(query_gpu_usage(obj, "elpa_pxgemm_multiply", useGPU))) then
    print *,"elpa_pxgemm_multiply: Problem querrying settings for GPU Aborting..."
    stop 1
  endif
#endif

  ! check whether the above setting should be overriden
  if (obj%is_set("gpu_pxgemm_multiply") == 1) then
    call obj%get("gpu_pxgemm_multiply", gpu_pxgemm_multiply, error)
    if (error .ne. ELPA_OK) then
      print *,"Problem getting option for gpu_hermitian_mutltiply. Aborting..."
      stop 1
    endif
    if (useGPU .and. gpu_pxgemm_multiply .eq. 0) then
      useGPU = .false.
    else if (.not.(useGPU) .and. gpu_pxgemm_multiply .eq. 1) then
      useGPU = .true.
    else
    endif
  else
    ! no override by user
    ! keep setting as found before
  endif
#endif /* DEVICE_POINTER */

  if(useGPU) then
    gpuString = "_gpu"
  else
    gpuString = ""
  endif

  if (a_transposed .and. b_transposed) then
    tnString = "_tt"
  else if (.not. a_transposed .and. b_transposed) then
    tnString = "_nt"
  else if (a_transposed .and. .not. b_transposed) then
    tnString = "_tn"
  else if (.not. a_transposed .and. .not. b_transposed) then
    tnString = "_nn"
  endif

  call obj%timer%start("elpa_pxgemm_multiply"//trim(tnString)//trim(gpuString))

  na      = obj%na
  nblk    = obj%nblk
  lda     = obj%local_nrows
  ldaCols = obj%local_ncols

  mpi_comm_all    = obj%mpi_setup%mpi_comm_parent
  mpi_comm_cols   = obj%mpi_setup%mpi_comm_cols
  mpi_comm_rows   = obj%mpi_setup%mpi_comm_rows

  myid    = obj%mpi_setup%myRank_comm_parent
  my_prow = obj%mpi_setup%myRank_comm_rows
  my_pcol = obj%mpi_setup%myRank_comm_cols

  np_rows = obj%mpi_setup%nRanks_comm_rows
  np_cols = obj%mpi_setup%nRanks_comm_cols

  l_rows = local_index(na,  my_prow, np_rows, nblk, -1) ! Local rows of a and b
  l_cols = local_index(ncb, my_pcol, np_cols, nblk, -1) ! Local cols of b

  useCCL = .false.
  if (useGPU) then
    call obj%timer%start("check_for_gpu")
    if (check_for_gpu(obj, myid, numGPU)) then
      ! set the neccessary parameters
      call set_gpu_parameters()
    else
      print *,"GPUs are requested but not detected! Aborting..."
      success = .false.
      return
    endif
    call obj%timer%stop("check_for_gpu")

    SM_count = obj%gpu_setup%gpuSMcount
    gpuHandle = obj%gpu_setup%gpublasHandleArray(0)

#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
#endif

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


#if defined(DEVICE_POINTER)
    a_dev = transfer(aDev, a_dev)
    b_dev = transfer(bDev, b_dev)
    c_dev = transfer(cDev, c_dev)

#else /* DEVICE_POINTER */
    successGPU = gpu_malloc(a_dev, lda*ldaCols*size_of_datatype)
    check_alloc_gpu("elpa_pxgemm_multiply: a_dev", successGPU)

    successGPU = gpu_malloc(b_dev, ldb*ldbCols*size_of_datatype)
    check_alloc_gpu("elpa_pxgemm_multiply: b_dev", successGPU) 

    successGPU = gpu_malloc(c_dev, ldc*ldcCols*size_of_datatype)
    check_alloc_gpu("elpa_pxgemm_multiply: c_dev", successGPU)

    call obj%timer%start("gpu_memcpy")
    NVTX_RANGE_PUSH("gpu_memcpy: a->a_dev, b->b_dev")
    ! copy a to a_dev
    num = lda*ldaCols*size_of_datatype
#ifdef WITH_GPU_STREAMS
    call gpu_memcpy_async_and_stream_synchronize &
    ("elpa_pxgemm_multiply: a to a_dev", a_dev, 0_c_intptr_t, &
                                       a(1:lda,1:ldaCols), &
                                       1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
    successGPU = gpu_memcpy(a_dev, int(loc(a),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
    check_memcpy_gpu("elpa_pxgemm_multiply: a to a_dev", successGPU)
#endif

    ! copy b to b_dev
#ifdef WITH_GPU_STREAMS
    call gpu_memcpy_async_and_stream_synchronize &
    ("elpa_pxgemm_multiply: b to b_dev", b_dev, 0_c_intptr_t, &
                                       b(1:ldb,1:ldbCols), &
                                       1, 1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
    successGPU = gpu_memcpy(b_dev,int(loc(b),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
    check_memcpy_gpu("elpa_pxgemm_multiply: b to b_dev", successGPU)
#endif

    NVTX_RANGE_POP("gpu_memcpy: a->a_dev, b->b_dev")
    call obj%timer%stop("gpu_memcpy")

    ! this is needed to protect from NaNs, until we introduce beta parameter and copy c to c_dev (c = alpha*a*b + beta*c)
#ifdef WITH_GPU_STREAMS
    successGPU = gpu_memset_async(c_dev, 0, ldc*ldcCols*size_of_datatype, my_stream)
#else
    successGPU = gpu_memset(c_dev, 0, ldc*ldcCols*size_of_datatype)
#endif
    check_memcpy_gpu("elpa_pxgemm_multiply: memset c_dev", successGPU)
#endif /* DEVICE_POINTER */
  endif ! useGPU

  isSquareGrid = .false.
  if (np_rows == np_cols) isSquareGrid = .true.

#if WITH_MPI
  ! overlap async memcopy to GPU with MPI
#ifdef WITH_GPU_STREAMS
  if (useGPU) then  
    successGPU = gpu_stream_synchronize(my_stream)
    check_stream_synchronize_gpu("elpa_pxgemm_multiply: stream synchronize", successGPU)
  endif
#endif

  ! l_rows_max = l_rows
  call mpi_allreduce(l_rows, l_rows_max, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
  call mpi_allreduce(l_cols, l_cols_max, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
  call mpi_allreduce(l_rows, l_rows_min, 1_MPI_KIND, MPI_INTEGER, MPI_MIN, int(mpi_comm_all,kind=MPI_KIND), mpierr)
  call mpi_allreduce(l_cols, l_cols_min, 1_MPI_KIND, MPI_INTEGER, MPI_MIN, int(mpi_comm_all,kind=MPI_KIND), mpierr)
#else /* WITH_MPI */
  l_rows_max = l_rows
  l_cols_max = l_cols
  l_rows_min = l_rows
  l_cols_min = l_cols
#endif /* WITH_MPI */

  nblk_mult = greatest_common_divisor(l_rows_max, l_cols_max)
!______________________________________________________________________________________________

  if (isSquareGrid) then
    !nblk_mult = l_rows_max ! = l_cols_max
!_______________________________________________

    if (.not. a_transposed .and. b_transposed .or. &
        a_transposed .and. .not. b_transposed ) then
      if (wantDebug .and. myid==0) print *, "elpa_pxgemm_multiply NEW: SQUARE_GRID start: (a_transposed XOR b_transposed)" ! PETERDEBUG
      
      allocate(aux_a_full(l_rows_max, nblk_mult), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_pxgemm_multiply: aux_a_full", istat, errorMessage)
    
      allocate(aux_b_full(nblk_mult, l_cols_max), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_pxgemm_multiply: aux_b_full", istat, errorMessage)
      
      allocate(tmp1_full(l_rows_max, l_cols_max), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_pxgemm_multiply: tmp1_full", istat, errorMessage)
  
      ! PETERDEBUG: is it possible to use the original GPU memory, without copying?
      if (useGPU) then
        successGPU = gpu_malloc(aux_a_full_dev, nblk_mult*nblk_mult*size_of_datatype)
        check_alloc_gpu("elpa_pxgemm_multiply: aux_a_full_dev", successGPU)
  
        successGPU = gpu_malloc(aux_b_full_dev, nblk_mult*nblk_mult*size_of_datatype)
        check_alloc_gpu("elpa_pxgemm_multiply: aux_b_full_dev", successGPU)
  
        successGPU = gpu_malloc(tmp1_full_dev, nblk_mult*nblk_mult*size_of_datatype)
        check_alloc_gpu("elpa_pxgemm_multiply: tmp1_full_dev", successGPU)
      endif

      ! dir = row/col for TN/NT
      if (a_transposed) then
        my_pdir = my_prow
        np_dirs = np_rows
        mpi_comm_dirs = mpi_comm_rows
        ccl_comm_dirs = obj%gpu_setup%ccl_comm_rows
      else if (b_transposed) then
        my_pdir = my_pcol
        np_dirs = np_cols
        mpi_comm_dirs = mpi_comm_cols
        ccl_comm_dirs = obj%gpu_setup%ccl_comm_cols
      endif

      call obj%timer%start("main_loop_square_grid_tn_nt")

      ! main loop: build up the result matrix by processor rows/cols for TN/NT
      do np = 0, np_dirs-1
        NVTX_RANGE_PUSH("do np = 0, np_dirs-1")
        if (wantDebug .and. myid==0)  print *, "np = ", np ! PETERDEBUG

        ! In this turn, procs of row/col np assemble the result for TN/NT case
        
        np_t=np ! np, that posesses the given "col of a"/"row of b" for TN/NT

        if (a_transposed) then
          if (useGPU) then
            if (np_t == my_pcol) call gpu_copy_and_set_zeros_aux_full(PRECISION_CHAR, a_dev, aux_a_full_dev, l_rows, l_cols, &
                                                                      nblk_mult, debug, my_stream)
            call gpu_copy_and_set_zeros_aux_full(PRECISION_CHAR, b_dev, aux_b_full_dev, l_rows, l_cols, &
                                                 nblk_mult, debug, my_stream)
          else ! useGPU
            if (np_t == my_pcol) then
              aux_a_full(1:l_rows,1:l_cols) = a(1:l_rows,1:l_cols)
              if (l_rows < nblk_mult) aux_a_full(l_rows+1:nblk_mult,1:l_cols) = 0
              if (l_cols < nblk_mult) aux_a_full(1:l_rows,l_cols+1:nblk_mult) = 0
              if (l_rows < nblk_mult .and. l_cols < nblk_mult) aux_a_full(l_rows+1:nblk_mult,l_cols+1:nblk_mult) = 0
            endif
            aux_b_full(1:l_rows,1:l_cols) = b(1:l_rows,1:l_cols)
            if (l_rows < nblk_mult) aux_b_full(l_rows+1:nblk_mult,1:l_cols) = 0
            if (l_cols < nblk_mult) aux_b_full(1:l_rows,l_cols+1:nblk_mult) = 0
            if (l_rows < nblk_mult .and. l_cols < nblk_mult) aux_b_full(l_rows+1:nblk_mult,l_cols+1:nblk_mult) = 0
          endif ! useGPU
        else if (b_transposed) then
          if (useGPU) then
            if (np_t == my_prow) call gpu_copy_and_set_zeros_aux_full(PRECISION_CHAR, b_dev, aux_b_full_dev, l_rows, l_cols, &
                                                                      nblk_mult, debug, my_stream)
            call gpu_copy_and_set_zeros_aux_full(PRECISION_CHAR, a_dev, aux_a_full_dev, l_rows, l_cols, &
                                                 nblk_mult, debug, my_stream)
          
          else
            if (np_t == my_prow) then
              aux_b_full(1:l_rows,1:l_cols) = b(1:l_rows,1:l_cols)
              if (l_rows < nblk_mult) aux_b_full(l_rows+1:nblk_mult,1:l_cols) = 0
              if (l_cols < nblk_mult) aux_b_full(1:l_rows,l_cols+1:nblk_mult) = 0
              if (l_rows < nblk_mult .and. l_cols < nblk_mult) aux_b_full(l_rows+1:nblk_mult,l_cols+1:nblk_mult) = 0
            endif
            aux_a_full(1:l_rows,1:l_cols) = a(1:l_rows,1:l_cols) ! aux_a_full -> aux_ab_nontransposed_full
            if (l_rows < nblk_mult) aux_a_full(l_rows+1:nblk_mult,1:l_cols) = 0
            if (l_cols < nblk_mult) aux_a_full(1:l_rows,l_cols+1:nblk_mult) = 0
            if (l_rows < nblk_mult .and. l_cols < nblk_mult) aux_a_full(l_rows+1:nblk_mult,l_cols+1:nblk_mult) = 0
          endif ! useGPU
        endif

        ! aux_a_full/aux_b_full -> aux_ab_trans (aux_ab) ! auxillary buffer for a matrix that is     to be transposed: a/b in TN/NT case
        ! aux_b_full/aux_a_full -> aux_ab_nontr (aux_ba) ! auxillary buffer for a matrix that is not to be transposed: b/a in TN/NT case

#ifdef WITH_MPI
        ! copy data to host for bcast, if needed
        if (useGPU .and. .not. useCCL) then
          num = nblk_mult*nblk_mult
#ifdef WITH_GPU_STREAMS
          if (a_transposed) then
            call gpu_memcpy_async_and_stream_synchronize("elpa_pxgemm_multiply: aux_a_full_dev -> aux_a_full",  &
                  aux_a_full_dev, 0_c_intptr_t, aux_a_full(1:nblk_mult,1:nblk_mult), &
                  1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .false., .false.)
          else if (b_transposed) then
            call gpu_memcpy_async_and_stream_synchronize("elpa_pxgemm_multiply: aux_b_full_dev -> aux_b_full", &
                  aux_b_full_dev, 0_c_intptr_t, aux_b_full(1:nblk_mult,1:nblk_mult), &
                  1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
          endif
#else
          if (a_transposed) then
            successGPU = gpu_memcpy(int(loc(aux_a_full),kind=c_intptr_t), aux_a_full_dev, &
                                    num*size_of_datatype, gpuMemcpyDeviceToHost)
            check_memcpy_gpu("elpa_pxgemm_multiply: aux_a_full_dev -> aux_a_full", successGPU)
          else if (b_transposed) then
            successGPU = gpu_memcpy(int(loc(aux_b_full),kind=c_intptr_t), aux_b_full_dev, &
                                    num*size_of_datatype, gpuMemcpyDeviceToHost)
            check_memcpy_gpu("elpa_pxgemm_multiply: aux_b_full_dev -> aux_b_full", successGPU)
          endif
#endif  
        endif ! (useGPU .and. .not. useCCL)


        ! Broadcast processor column
        if (useCCL) then
          call obj%timer%start("ccl_bcast")
          NVTX_RANGE_PUSH("ccl_bcast aux_a/b_full")

          if (a_transposed) then
            successGPU = ccl_bcast(aux_a_full_dev, aux_a_full_dev, int(k_datatype*nblk_mult*nblk_mult,kind=c_size_t), &
                                   cclDatatype, int(np_t,kind=c_int), ccl_comm_cols, my_stream)
          else if (b_transposed) then
            successGPU = ccl_bcast(aux_b_full_dev, aux_b_full_dev, int(k_datatype*nblk_mult*nblk_mult,kind=c_size_t), &
                                   cclDatatype, int(np_t,kind=c_int), ccl_comm_rows, my_stream)
          endif

          if (.not. successGPU) then
            print *,"Error in ccl_bcast"
            stop 1
          endif

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_pxgemm_multiply: ccl_bcast", successGPU)

          NVTX_RANGE_POP("ccl_bcast aux_a/b_full")
          call obj%timer%stop("ccl_bcast")
        else ! useCCL
          call obj%timer%start("mpi_communication")
          NVTX_RANGE_PUSH("MPI_Bcast(aux_a/b_full)")

          if (a_transposed) then
            call MPI_Bcast(aux_a_full, int(nblk_mult*nblk_mult,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                          int(np_t,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
          else if (b_transposed) then
            call MPI_Bcast(aux_b_full, int(nblk_mult*nblk_mult,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                          int(np_t,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
          endif

          NVTX_RANGE_POP("MPI_Bcast(aux_a/b_full)")
          call obj%timer%stop("mpi_communication")
        endif ! useCCL

        ! copy data back to device, if needed
        if (useGPU .and. .not. useCCL) then
          num = nblk_mult*nblk_mult
#ifdef WITH_GPU_STREAMS
          if (a_transposed) then
            call gpu_memcpy_async_and_stream_synchronize("elpa_pxgemm_multiply: aux_a_full -> aux_a_full_dev", &
                  aux_a_full_dev, 0_c_intptr_t, aux_a_full(1:nblk_mult,1:nblk_mult), &
                  1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
          else if (b_transposed) then
            call gpu_memcpy_async_and_stream_synchronize("elpa_pxgemm_multiply: aux_b_full -> aux_b_full_dev", &
                  aux_b_full_dev, 0_c_intptr_t, aux_b_full(1:nblk_mult,1:nblk_mult), &
                  1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
          endif
#else
          if (a_transposed) then
            successGPU = gpu_memcpy(aux_a_full_dev, int(loc(aux_a_full),kind=c_intptr_t), &
                                    num*size_of_datatype, gpuMemcpyHostToDevice)
            check_memcpy_gpu("elpa_pxgemm_multiply: aux_a_full -> aux_a_full_dev", successGPU)
          else if (b_transposed) then
            successGPU = gpu_memcpy(aux_b_full_dev, int(loc(aux_b_full),kind=c_intptr_t), &
                                  num*size_of_datatype, gpuMemcpyHostToDevice)
            check_memcpy_gpu("elpa_pxgemm_multiply: aux_b_full -> aux_b_full_dev", successGPU)
          endif
#endif
        endif ! (useGPU .and. .not. useCCL)
#endif /* WITH_MPI */

        if (useGPU) then
          call obj%timer%start("gpublas")
          NVTX_RANGE_PUSH("gpublas")
          call gpublas_PRECISION_GEMM(trans_a_cchar, trans_b_cchar, &
                            nblk_mult, nblk_mult, nblk_mult, ONE, &
                            aux_a_full_dev, nblk_mult, &
                            aux_b_full_dev, nblk_mult, ZERO, &
                            tmp1_full_dev , nblk_mult, gpuHandle)
          if (wantDebug) successGPU = gpu_DeviceSynchronize()
          NVTX_RANGE_POP("gpublas")
          call obj%timer%stop("gpublas")
        else ! useGPU
          call obj%timer%start("blas")
          call PRECISION_GEMM(trans_a, trans_b, &
                              int(nblk_mult, kind=BLAS_KIND), &
                              int(nblk_mult, kind=BLAS_KIND), &
                              int(nblk_mult, kind=BLAS_KIND), ONE, &
                              aux_a_full, int(nblk_mult, kind=BLAS_KIND), &
                              aux_b_full, int(nblk_mult, kind=BLAS_KIND), ZERO, &
                              tmp1_full , int(nblk_mult, kind=BLAS_KIND))
          call obj%timer%stop("blas")
        endif ! useGPU

#ifdef WITH_MPI
        num = nblk_mult*nblk_mult

        ! copy data to host for mpi_reduce, if needed
        if (useGPU .and. .not. useCCL) then
          call obj%timer%start("gpu_memcpy")
#ifdef WITH_GPU_STREAMS
          call gpu_memcpy_async_and_stream_synchronize("elpa_pxgemm_multiply: tmp1_full_dev -> tmp1_full",  &
                tmp1_full_dev, 0_c_intptr_t, tmp1_full(1:nblk_mult,1:nblk_mult), &
                1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .false., .false.)
#else
          successGPU = gpu_memcpy(int(loc(tmp1_full),kind=c_intptr_t), tmp1_full_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_pxgemm_multiply: tmp1_full_dev -> tmp1_full", successGPU)
#endif
          call obj%timer%stop("gpu_memcpy")
        endif ! (useGPU .and. .not. useCCL)


        if (useCCL) then
          call obj%timer%start("ccl_reduce")
          NVTX_RANGE_PUSH("ccl_reduce tmp1_full_dev")

          successGPU  = ccl_reduce(tmp1_full_dev, tmp1_full_dev, int(k_datatype*num,kind=c_size_t), cclDatatype, &
                                   cclSum, int(np, kind=c_int), ccl_comm_dirs, my_stream)
          
          if (.not. successGPU) then
            print *,"Error in ccl_reduce"
            stop 1
          endif

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_pxgemm_multiply: ccl_bcast", successGPU)

          NVTX_RANGE_POP("ccl_bcast aux_a_full_dev, aux_b_full_dev")
          call obj%timer%stop("ccl_reduce")    
        else ! useCCL
          call obj%timer%start("mpi_communication")
          if (my_pdir==np) then
            call MPI_Reduce(MPI_IN_PLACE, tmp1_full, int(num,kind=MPI_KIND),  MPI_MATH_DATATYPE_PRECISION, &
                            MPI_SUM, int(np,kind=MPI_KIND), int(mpi_comm_dirs,kind=MPI_KIND), mpierr)
          else
            call MPI_Reduce(tmp1_full   , tmp1_full, int(num,kind=MPI_KIND),  MPI_MATH_DATATYPE_PRECISION, &
                            MPI_SUM, int(np,kind=MPI_KIND), int(mpi_comm_dirs,kind=MPI_KIND), mpierr)
          endif
          call obj%timer%stop("mpi_communication")
        endif ! useCCL

        if (useGPU .and. .not. useCCL) then
          call obj%timer%start("gpu_memcpy")
#ifdef WITH_GPU_STREAMS
          call gpu_memcpy_async_and_stream_synchronize &
              ("elpa_pxgemm_multiply: tmp1_full -> tmp1_full_dev", 0_c_intptr_t, tmp1_full_dev, &
                tmp1_full(1:nblk_mult,1:nblk_mult), &
                1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
          successGPU = gpu_memcpy(tmp1_full_dev, int(loc(tmp1_full),kind=c_intptr_t), &
                                  num*size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_pxgemm_multiply: tmp1_full -> tmp1_full_dev", successGPU)
#endif
          call obj%timer%stop("gpu_memcpy")
        endif ! (useGPU .and. .not. useCCL)
#endif /* WITH_MPI */

        ! Put the result into C
        if (my_pdir==np) then
          if (useGPU) then
            NVTX_RANGE_PUSH("gpu_copy_aux_full")
            call gpu_copy_aux_full(PRECISION_CHAR, c_dev, tmp1_full_dev, l_rows, l_cols, ldc, nblk_mult, debug, my_stream)
            NVTX_RANGE_POP ("gpu_copy_aux_full")
          else  ! useGPU
            c(1:l_rows,1:l_cols) = tmp1_full(1:l_rows,1:l_cols)
          endif ! useGPU
        endif

        NVTX_RANGE_POP("do np = 0, np_dirs-1")
      enddo ! np = 0, np_dirs-1
      call obj%timer%stop("main_loop_square_grid_tn_nt")

#if !defined(DEVICE_POINTER)
      if (useGPU) then
        call obj%timer%start("gpu_memcpy")
        NVTX_RANGE_PUSH("gpu_memcpy: c_dev->c")
        num = ldc*ldcCols
#ifdef WITH_GPU_STREAMS
        call gpu_memcpy_async_and_stream_synchronize &
            ("elpa_pxgemm_multiply: c_dev -> c", c_dev, 0_c_intptr_t, c(1:ldc,1:ldcCols), &
              1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
        successGPU = gpu_memcpy(int(loc(c),kind=c_intptr_t), c_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("elpa_pxgemm_multiply: c_dev -> c", successGPU)
#endif
        NVTX_RANGE_POP("gpu_memcpy: c_dev->c")
        call obj%timer%stop("gpu_memcpy")
      endif ! useGPU
#endif /* !defined(DEVICE_POINTER) */

      deallocate(tmp1_full,  stat=istat, errmsg=errorMessage)
      call check_alloc("elpa_pxgemm_multiply", "tmp1_full", istat, errorMessage)

      if (useGPU) then
        successGPU = gpu_free(tmp1_full_dev)
        check_dealloc_gpu("elpa_pxgemm_multiply: tmp1_full_dev", successGPU)
      endif

    endif ! (.not. a_transposed .and. b_transposed)

!_______________________________________________

    if ((.not. a_transposed) .and. (.not. b_transposed) .or. &
               a_transposed  .and.        b_transposed) then
       if (wantDebug .and. myid==0) print *, "elpa_pxgemm_multiply NEW: SQUARE_GRID start: NN or TT" ! PETERDEBUG
   
      allocate(aux_a_full(l_rows, nblk_mult), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_pxgemm_multiply: aux_a_full", istat, errorMessage)
    
      allocate(aux_b_full(nblk_mult, l_cols), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_pxgemm_multiply: aux_b_full", istat, errorMessage)
      
      if (useGPU .and. l_rows /= ldc) then
        print *, "elpa_pxgemm_multiply: Error: case ldc != lda is not implemented yet for NN and TT on GPU"
        stop 1
      endif
      
      ! PETERDEBUG: is it possible to use the original GPU memory, without copying?
      if (useGPU) then
        successGPU = gpu_malloc(aux_a_full_dev, l_rows*nblk_mult*size_of_datatype)
        check_alloc_gpu("elpa_pxgemm_multiply: aux_a_full_dev", successGPU)
  
        successGPU = gpu_malloc(aux_b_full_dev, nblk_mult*l_cols*size_of_datatype)
        check_alloc_gpu("elpa_pxgemm_multiply: aux_b_full_dev", successGPU)
      endif

      if (a_transposed) then
        if (.not. useCCL) then
          allocate(at_col(l_rows, l_cols), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: at_col", istat, errorMessage)
        endif

        if (useGPU) then
          successGPU = gpu_malloc(at_col_dev, l_rows*l_cols*size_of_datatype)
          check_alloc_gpu("elpa_pxgemm_multiply: at_col_dev", successGPU)
        endif
      endif

      if (b_transposed) then
        if (.not. useCCL) then
          allocate(bt_row(l_rows, l_cols), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: bt_row", istat, errorMessage)
        endif

        if (useGPU) then
          successGPU = gpu_malloc(bt_row_dev, l_rows*l_cols*size_of_datatype)
          check_alloc_gpu("elpa_pxgemm_multiply: bt_row_dev", successGPU)
        endif
      endif

      if (a_transposed .or. b_transposed) then
        LCM = least_common_multiple(np_rows, np_cols)*nblk
        ! max nblk_mult_rows, nblk_mult_cols is achieved at the 0-th MPI process
        nblk_mult_rows_max = find_nblk_mult_dirs(l_rows_max, nblk, np_rows, 0, LCM)
        nblk_mult_cols_max = find_nblk_mult_dirs(l_cols_max, nblk, np_cols, 0, LCM)

        if (useCCL) then
          successGPU = gpu_malloc(buf_send_dev, nblk_mult_rows_max*nblk_mult_cols_max*size_of_datatype)
          check_alloc_gpu("elpa_pxgemm_multiply: buf_send_dev", successGPU)

          successGPU = gpu_malloc(buf_recv_dev, nblk_mult_rows_max*nblk_mult_cols_max*size_of_datatype)
          check_alloc_gpu("elpa_pxgemm_multiply: buf_recv_dev", successGPU)

          successGPU = gpu_malloc(buf_self_dev, nblk_mult_rows_max*nblk_mult_cols_max*size_of_datatype)
          check_alloc_gpu("elpa_pxgemm_multiply: buf_self_dev", successGPU)
        else ! useCCL
          allocate(buf_send(nblk_mult_rows_max, nblk_mult_cols_max), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: buf_send", istat, errorMessage)

          allocate(buf_recv(nblk_mult_rows_max, nblk_mult_cols_max), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: buf_recv", istat, errorMessage)

          allocate(buf_self(nblk_mult_rows_max, nblk_mult_cols_max), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: buf_self", istat, errorMessage)

#if defined(DEVICE_POINTER)
          allocate(a(obj%local_nrows, obj%local_ncols), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: a", istat, errorMessage)

          allocate(b(ldb, ldbCols), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: b", istat, errorMessage)

          successGPU = gpu_memcpy(int(loc(a),kind=c_intptr_t), a_dev, &
                                  obj%local_nrows*obj%local_ncols*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_pxgemm_multiply: a_dev -> a", successGPU)

          successGPU = gpu_memcpy(int(loc(b),kind=c_intptr_t), b_dev, &
                                  ldb*ldbCols*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_pxgemm_multiply: b_dev -> b", successGPU)
#endif /* defined(DEVICE_POINTER) */          
        endif ! useCCL
      endif ! (a_transposed .or. b_transposed)

      ! main loop: iterate through np, which are process rows for matrix A and process cols for matrix B
      call obj%timer%start("main_loop_square_grid_nn_tt")

      do np = 0, np_rows-1 ! np_rows=np_cols
        NVTX_RANGE_PUSH("np = 0, np_rows-1")

        ! In this turn, procs of row np assemble the result
        np_bc=np ! np, that posesses the given column of a

        if (a_transposed) then
          if (useCCL) then
            call elpa_transpose_row_or_col_ccl_&
                  &MATH_DATATYPE&
                  &_&
                  &PRECISION&
                  (obj, 'R', a_dev, at_col_dev, buf_send_dev, buf_recv_dev, buf_self_dev, np, l_rows, l_cols, &
                  nblk_mult_rows_max, nblk_mult_cols_max, debug)
          else
            call elpa_transpose_row_or_col_&
                  &MATH_DATATYPE&
                  &_&
                  &PRECISION&
                  (obj, 'R', a, at_col, buf_send, buf_recv, buf_self, np, l_rows, l_cols, &
                  nblk_mult_rows_max, nblk_mult_cols_max, debug)
          endif
        endif

        if (b_transposed) then
          if (useCCL) then
            call elpa_transpose_row_or_col_ccl_&
                  &MATH_DATATYPE&
                  &_&
                  &PRECISION&
                  (obj, 'C', b_dev, bt_row_dev, buf_send_dev, buf_recv_dev, buf_self_dev, np, l_rows, l_cols, &
                  nblk_mult_rows_max, nblk_mult_cols_max, debug)
          else
            call elpa_transpose_row_or_col_&
                  &MATH_DATATYPE&
                  &_&
                  &PRECISION&
                  (obj, 'C', b, bt_row, buf_send, buf_recv, buf_self, np, l_rows, l_cols, &
                  nblk_mult_rows_max, nblk_mult_cols_max, debug)
          endif
        endif

        if (np_bc == my_pcol) then
          if (useGPU) then
            if (a_transposed) then
              if (.not. useCCL) then
                num = l_rows*l_cols
#ifdef WITH_GPU_STREAMS
                call gpu_memcpy_async_and_stream_synchronize &
                    ("elpa_pxgemm_multiply: at_col -> at_col_dev", at_col_dev, 0_c_intptr_t, at_col(1:l_rows,1:l_cols), &
                      1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
                successGPU = gpu_memcpy(at_col_dev, int(loc(at_col),kind=c_intptr_t), num*size_of_datatype, gpuMemcpyHostToDevice)
                check_memcpy_gpu("elpa_pxgemm_multiply: at_col -> at_col_dev", successGPU)
#endif
              endif ! (.not. useCCL)

              call gpu_copy_aux_full(PRECISION_CHAR, aux_a_full_dev, at_col_dev, &
                                    l_rows, l_cols, l_rows, obj%local_nrows, debug, my_stream)
            else
              call gpu_copy_aux_full(PRECISION_CHAR, aux_a_full_dev, a_dev, &
                                    l_rows, l_cols, l_rows, obj%local_nrows, debug, my_stream)
              ! call gpu_copy_and_set_zeros_aux_full(PRECISION_CHAR, a_dev, aux_a_full_dev, & ! PETERDEBUG: cleanup this kernel
              !                                       l_rows, l_cols, l_rows, debug, my_stream)
                                                  !l_rows, l_cols, nblk_mult, debug, my_stream)
            endif
          else ! useGPU
            if (a_transposed) then
              aux_a_full(1:l_rows,1:l_cols) = at_col(1:l_rows,1:l_cols)
            else
              aux_a_full(1:l_rows,1:l_cols) = a(1:l_rows,1:l_cols)
            endif
          endif ! useGPU
        endif
        if (np    == my_prow) then
          if (useGPU) then
            if (b_transposed) then
              if (.not. useCCL) then
                num = l_rows*l_cols
#ifdef WITH_GPU_STREAMS
                call gpu_memcpy_async_and_stream_synchronize &
                    ("elpa_pxgemm_multiply: bt_row -> bt_row_dev", bt_row_dev, 0_c_intptr_t, bt_row(1:l_rows,1:l_cols), &
                      1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
                successGPU = gpu_memcpy(bt_row_dev, int(loc(bt_row),kind=c_intptr_t), num*size_of_datatype, gpuMemcpyHostToDevice)
                check_memcpy_gpu("elpa_pxgemm_multiply: bt_row -> bt_row_dev", successGPU)
#endif
              endif ! (.not. useCCL)
              
              call gpu_copy_aux_full(PRECISION_CHAR, aux_b_full_dev, bt_row_dev, &
                                    l_rows, l_cols, nblk_mult, ldb, debug, my_stream)
            else
              call gpu_copy_aux_full(PRECISION_CHAR, aux_b_full_dev, b_dev, &
                                    l_rows, l_cols, nblk_mult, ldb, debug, my_stream)
              ! call gpu_copy_and_set_zeros_aux_full(PRECISION_CHAR, b_dev, aux_b_full_dev, & ! PETERDEBUG
              !                                       l_rows, l_cols, l_rows, debug, my_stream)
                                                  !l_rows, l_cols, nblk_mult, debug, my_stream)
            endif
          else ! useGPU
            if (b_transposed) then
              aux_b_full(1:l_rows,1:l_cols) = bt_row(1:l_rows,1:l_cols)
            else
              aux_b_full(1:l_rows,1:l_cols) = b(1:l_rows,1:l_cols)
            endif
          endif ! useGPU
        endif

#ifdef WITH_MPI
        ! copy data to host for bcast, if needed
        if (useGPU .and. .not. useCCL) then
          !num = nblk_mult*nblk_mult
          num_a = l_rows*nblk_mult
          num_b = nblk_mult*l_cols
#ifdef WITH_GPU_STREAMS
          call gpu_memcpy_async_and_stream_synchronize("elpa_pxgemm_multiply: aux_a_full_dev -> aux_a_full", &
                aux_a_full_dev, 0_c_intptr_t, aux_a_full(1:l_rows,1:nblk_mult), &
                1, 1, num_a*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .false., .false.)
          call gpu_memcpy_async_and_stream_synchronize("elpa_pxgemm_multiply: aux_b_full_dev -> aux_b_full", &
                aux_b_full_dev, 0_c_intptr_t, aux_b_full(1:nblk_mult,1:l_cols), &
                1, 1, num_b*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
          successGPU = gpu_memcpy(int(loc(aux_a_full),kind=c_intptr_t),aux_a_full_dev,num_a*size_of_datatype,gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_pxgemm_multiply: aux_a_full_dev -> aux_a_full", successGPU)

          successGPU = gpu_memcpy(int(loc(aux_b_full),kind=c_intptr_t),aux_b_full_dev,num_b*size_of_datatype,gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_pxgemm_multiply: aux_b_full_dev -> aux_b_full", successGPU)
#endif  
        endif ! (useGPU .and. .not. useCCL)

        ! Broadcast processor column
        if (useCCL) then
          call obj%timer%start("ccl_bcast")
          NVTX_RANGE_PUSH("ccl_bcast aux_a_full_dev, aux_b_full_dev")

          successGPU  = ccl_bcast(aux_a_full_dev, aux_a_full_dev, int(k_datatype*l_rows*nblk_mult,kind=c_size_t), cclDatatype, &
                                  int(np_bc,kind=c_int), ccl_comm_cols, my_stream)

          successGPU2 = ccl_bcast(aux_b_full_dev, aux_b_full_dev, int(k_datatype*nblk_mult*l_cols,kind=c_size_t), cclDatatype, &
                                  int(np   ,kind=c_int), ccl_comm_rows, my_stream)
          
          if (.not. (successGPU .and. successGPU2)) then
            print *,"Error in ccl_bcast"
            stop 1
          endif

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_pxgemm_multiply: ccl_bcast", successGPU)

          NVTX_RANGE_POP("ccl_bcast aux_a_full_dev, aux_b_full_dev")
          call obj%timer%stop("ccl_bcast")
        else ! useCCL
          call obj%timer%start("mpi_communication")
          NVTX_RANGE_PUSH("MPI_Bcast: aux_a_full, aux_b_full")

          call MPI_Bcast(aux_a_full, int(l_rows*nblk_mult,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                        int(np_bc,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
          call MPI_Bcast(aux_b_full, int(nblk_mult*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                        int(np   ,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)

          NVTX_RANGE_POP("MPI_Bcast: aux_a_full, aux_b_full")
          call obj%timer%stop("mpi_communication")
        endif ! useCCL

        call obj%timer%start("mpi_communication")
        NVTX_RANGE_PUSH("MPI_Bcast: l_cols_source, l_rows_source")
        l_cols_source = l_cols
        l_rows_source = l_rows
        
        call MPI_Bcast(l_cols_source, int(1,kind=MPI_KIND), MPI_INTEGER, &
                      int(np_bc,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
        call MPI_Bcast(l_rows_source, int(1,kind=MPI_KIND), MPI_INTEGER, &
                      int(np   ,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)

        nblk_mult_min = min(l_rows_source, l_cols_source)
        NVTX_RANGE_POP("MPI_Bcast: l_cols_source, l_rows_source")
        call obj%timer%stop("mpi_communication")

        ! copy data back to device, if needed
        if (useGPU .and. .not. useCCL) then
          num_a = l_rows*nblk_mult
          num_b = nblk_mult*l_cols
#ifdef WITH_GPU_STREAMS
          call gpu_memcpy_async_and_stream_synchronize("elpa_pxgemm_multiply: aux_a_full -> aux_a_full_dev", &
                aux_a_full_dev, 0_c_intptr_t, aux_a_full(1:l_rows,1:nblk_mult), &
                1, 1, num_a*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
          call gpu_memcpy_async_and_stream_synchronize("elpa_pxgemm_multiply: aux_b_full -> aux_b_full_dev", &
                 aux_b_full_dev, 0_c_intptr_t, aux_b_full(1:nblk_mult,1:l_cols), &
                 1, 1, num_b*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
          successGPU = gpu_memcpy(aux_a_full_dev,int(loc(aux_a_full),kind=c_intptr_t),num_a*size_of_datatype,gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_pxgemm_multiply: aux_a_full -> aux_a_full_dev", successGPU)
          
          successGPU = gpu_memcpy(aux_b_full_dev,int(loc(aux_b_full),kind=c_intptr_t),num_b*size_of_datatype,gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_pxgemm_multiply: aux_b_full -> aux_b_full_dev", successGPU)
#endif
        endif ! (useGPU .and. .not. useCCL)
#else /* WITH_MPI */
        nblk_mult_min= l_rows
#endif /* WITH_MPI */

        beta = ZERO
        if (np>0) beta = ONE
        if (useGPU) then
          call obj%timer%start("gpublas")
          NVTX_RANGE_PUSH("gpublas")
          call gpublas_PRECISION_GEMM('N', 'N', &
                            l_rows, l_cols, nblk_mult_min, ONE, &
                            aux_a_full_dev, l_rows, &
                            aux_b_full_dev, nblk_mult, beta, &
                            c_dev , l_rows, gpuHandle)
          if (wantDebug) successGPU = gpu_DeviceSynchronize()
          NVTX_RANGE_POP("gpublas")
          call obj%timer%stop("gpublas")
        else ! useGPU
          call obj%timer%start("blas")
          call PRECISION_GEMM('N', 'N', &
                              int(l_rows, kind=BLAS_KIND), &
                              int(l_cols, kind=BLAS_KIND), &
                              int(nblk_mult_min, kind=BLAS_KIND), ONE, &
                              aux_a_full(1:l_rows, 1:nblk_mult_min), int(l_rows,kind=BLAS_KIND), &
                              aux_b_full(1:nblk_mult_min, 1:l_cols), int(nblk_mult_min,kind=BLAS_KIND), beta, &
                              c(1:l_rows, 1:l_cols), int(l_rows,kind=BLAS_KIND))
          call obj%timer%stop("blas")
        endif ! useGPU

        NVTX_RANGE_POP("np = 0, np_rows-1")
      enddo ! np = 0, np_rows-1

      call obj%timer%stop("main_loop_square_grid_nn_tt")
    
#if !defined(DEVICE_POINTER)
      ! Put the result into C
      if (useGPU) then
        call obj%timer%start("gpu_memcpy")
        NVTX_RANGE_PUSH("gpu_memcpy: c_dev->c")
        num = l_rows*l_cols
#ifdef WITH_GPU_STREAMS
        call gpu_memcpy_async_and_stream_synchronize &
            ("elpa_pxgemm_multiply: c_dev -> c", c_dev, 0_c_intptr_t, c(1:l_rows,1:l_cols), &
              1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
        successGPU = gpu_memcpy(int(loc(c),kind=c_intptr_t), c_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("elpa_pxgemm_multiply: c_dev -> c", successGPU)
#endif
        NVTX_RANGE_POP("gpu_memcpy: c_dev->c")
        call obj%timer%stop("gpu_memcpy")
      endif ! useGPU
#endif /* !defined(DEVICE_POINTER) */

      if (a_transposed) then
        if (.not. useCCL) then
          deallocate(at_col, stat=istat, errmsg=errorMessage)
          call check_alloc("elpa_pxgemm_multiply", "at_col", istat, errorMessage)
        endif

        if (useGPU) then
          successGPU = gpu_free(at_col_dev)
          check_dealloc_gpu("elpa_pxgemm_multiply: at_col_dev", successGPU)
        endif
      endif

      if (b_transposed) then
        if (.not. useCCL) then
          deallocate(bt_row, stat=istat, errmsg=errorMessage)
          call check_alloc("elpa_pxgemm_multiply", "bt_row", istat, errorMessage)
        endif

        if (useGPU) then
          successGPU = gpu_free(bt_row_dev)
          check_dealloc_gpu("elpa_pxgemm_multiply: bt_row_dev", successGPU)
        endif
      endif

      if (a_transposed .or. b_transposed) then
        if (useCCL) then
          successGPU = gpu_free(buf_send_dev)
          check_dealloc_gpu("elpa_pxgemm_multiply: buf_send_dev", successGPU)

          successGPU = gpu_free(buf_recv_dev)
          check_dealloc_gpu("elpa_pxgemm_multiply: buf_recv_dev", successGPU)

          successGPU = gpu_free(buf_self_dev)
          check_dealloc_gpu("elpa_pxgemm_multiply: buf_self_dev", successGPU)
        else
          deallocate(buf_send, buf_recv, buf_self, stat=istat, errmsg=errorMessage)
          call check_alloc("elpa_pxgemm_multiply", "buf_send, buf_recv, buf_self", istat, errorMessage)

#if defined(DEVICE_POINTER)
          deallocate(a, b, stat=istat, errmsg=errorMessage)
          call check_alloc("elpa_pxgemm_multiply", "a, b", istat, errorMessage)
#endif
        endif
      endif ! (a_transposed .or. b_transposed)

    endif ! ((.not. a_transposed) .and. (.not. b_transposed) .or. a_transposed .and. b_transposed)

!_______________________________________________

  endif ! isSquareGrid

! _________________________________________________________________________________________________________________________________

  if (.not. isSquareGrid) then

    LCM = least_common_multiple(np_rows, np_cols)*nblk
    nblk_mult_rows_max = (l_rows_max*np_rows+LCM-1)/LCM*nblk
    nblk_mult_cols_max = (l_cols_max*np_cols+LCM-1)/LCM*nblk
    nblk_mult = max(nblk_mult_rows_max, nblk_mult_cols_max)
    
    if (wantDebug .and. myid==0) then ! PETERDEBUG
      print *, "LCM = ", LCM
      print *, "l_rows_max", l_rows_max
      print *, "l_cols_max", l_cols_max
      print *, "l_rows", l_rows
      print *, "l_cols", l_cols
      print *, "np_rows", np_rows
      print *, "np_cols", np_cols
      print *, "nblk_mult = ", nblk_mult
    endif

    np_rows_fine = least_common_multiple(np_rows, np_cols)
    np_cols_fine = np_rows_fine

! ___________________________________________________________________

    if ((.not. a_transposed) .and. (.not. b_transposed) .or. &
               a_transposed  .and.        b_transposed) then
      if (wantDebug .and. myid==0) print *, "elpa_pxgemm_multiply NEW: NON-SQUARE_GRID start: NN or TT or universal" ! PETERDEBUG

      allocate(aux_a_full(l_rows, nblk_mult), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_pxgemm_multiply: aux_a_full", istat, errorMessage)
    
      allocate(aux_b_full(nblk_mult, l_cols), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_pxgemm_multiply: aux_b_full", istat, errorMessage)
      
      if (useGPU) then
        successGPU = gpu_malloc(aux_a_full_dev, l_rows*nblk_mult*size_of_datatype)
        check_alloc_gpu("elpa_pxgemm_multiply: aux_a_full_dev", successGPU)
  
        successGPU = gpu_malloc(aux_b_full_dev, nblk_mult*l_cols*size_of_datatype)
        check_alloc_gpu("elpa_pxgemm_multiply: aux_b_full_dev", successGPU)
      endif

      if (useGPU .and. l_rows /= ldc) then
        print *, "elpa_pxgemm_multiply: Error: case ldc != lda is not implemented yet for NN and TT on GPU"
        stop 1
      endif
      
      if (a_transposed) then
        ! PETERDEBUG: this is as memory consuming as the whole matrix. Can we do smth about it?
        if (.not. useCCL) then
          allocate(at_col(l_rows, l_cols), stat=istat, errmsg=errorMessage) ! l_rows*nblk_mult as aux_a_full (maybe they are the same??)
          check_allocate("elpa_pxgemm_multiply: at_col", istat, errorMessage)
          ! PETERDEBUG: it seems we can't do much about it. Consider 2x2 grid. 
          ! Process (0,1) sends data to process (1,0), hence the latter should contain the whole local matrix
          ! Of course, process (1,1) doesn't need the space for the matrix, but that's not signficant
          ! due to the balance considerations
        endif

        if (useGPU) then
          ! PETERDEBUG: this is as memory consuming as the whole matrix. Can we do smth about it?
          successGPU = gpu_malloc(at_col_dev, l_rows*l_cols*size_of_datatype) ! l_rows*nblk_mult as aux_a_full_dev
          check_alloc_gpu("elpa_pxgemm_multiply: at_col_dev", successGPU)
        endif
      endif

      if (b_transposed) then
        if (.not. useCCL) then
          ! PETERDEBUG: this is as memory consuming as the whole matrix. Can we do smth about it?
          allocate(bt_row(l_rows, l_cols), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: bt_row", istat, errorMessage)
        endif

        if (useGPU) then
          ! PETERDEBUG: this is as memory consuming as the whole matrix. Can we do smth about it?
          successGPU = gpu_malloc(bt_row_dev, l_rows*l_cols*size_of_datatype)
          check_alloc_gpu("elpa_pxgemm_multiply: bt_row_dev", successGPU)
        endif
      endif

      if (a_transposed .or. b_transposed) then
        ! max nblk_mult_rows, nblk_mult_cols is achieved at the 0-th MPI process
        nblk_mult_rows_max = find_nblk_mult_dirs(l_rows_max, nblk, np_rows, 0, LCM)
        nblk_mult_cols_max = find_nblk_mult_dirs(l_cols_max, nblk, np_cols, 0, LCM)

        if (useCCL) then
          successGPU = gpu_malloc(buf_send_dev, nblk_mult_rows_max*nblk_mult_cols_max*size_of_datatype)
          check_alloc_gpu("elpa_pxgemm_multiply: buf_send_dev", successGPU)

          successGPU = gpu_malloc(buf_recv_dev, nblk_mult_rows_max*nblk_mult_cols_max*size_of_datatype)
          check_alloc_gpu("elpa_pxgemm_multiply: buf_recv_dev", successGPU)

          successGPU = gpu_malloc(buf_self_dev, nblk_mult_rows_max*nblk_mult_cols_max*size_of_datatype)
          check_alloc_gpu("elpa_pxgemm_multiply: buf_self_dev", successGPU)
        else ! useCCL
          allocate(buf_send(nblk_mult_rows_max, nblk_mult_cols_max), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: buf_send", istat, errorMessage)

          allocate(buf_recv(nblk_mult_rows_max, nblk_mult_cols_max), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: buf_recv", istat, errorMessage)

          allocate(buf_self(nblk_mult_rows_max, nblk_mult_cols_max), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: buf_self", istat, errorMessage)

#if defined(DEVICE_POINTER)
          allocate(a(obj%local_nrows, obj%local_ncols), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: a", istat, errorMessage)

          allocate(b(ldb, ldbCols), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: b", istat, errorMessage)

          successGPU = gpu_memcpy(int(loc(a),kind=c_intptr_t), a_dev, &
                                  obj%local_nrows*obj%local_ncols*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_pxgemm_multiply: a_dev -> a", successGPU)

          successGPU = gpu_memcpy(int(loc(b),kind=c_intptr_t), b_dev, &
                                  ldb*ldbCols*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_pxgemm_multiply: b_dev -> b", successGPU)
#endif /* defined(DEVICE_POINTER) */
        endif ! useCCL
      endif

      call obj%timer%start("main_loop_nn_tt")

      ! main loop: iterate through np_fine, which are "virtual" process rows for matrix A and process cols for matrix B
      do np_fine = 0, np_rows_fine-1 ! np_rows_fine=np_cols_fine
        NVTX_RANGE_PUSH("np_fine = 0, np_rows_fine-1")
        if (wantDebug .and. myid==0) print *, "np_fine = ", np_fine ! PETERDEBUG

        ! In this turn, procs of row np_fine assemble the result
        
        np_bc_fine=np_fine ! np_fine, that posesses the given column of a

        np    = mod(np_fine, np_rows)
        np_bc = mod(np_bc_fine, np_cols)

        nblk_mult_rows = find_nblk_mult_dirs(l_rows, nblk, np_rows, np_fine   , LCM)
        nblk_mult_cols = find_nblk_mult_dirs(l_cols, nblk, np_cols, np_bc_fine, LCM)

        if (a_transposed) then
          if (useCCL) then
            call elpa_transpose_row_or_col_ccl_&
                  &MATH_DATATYPE&
                  &_&
                  &PRECISION&
                  (obj, 'R', a_dev, at_col_dev, buf_send_dev, buf_recv_dev, buf_self_dev, np_fine, l_rows, l_cols, &
                  nblk_mult_rows_max, nblk_mult_cols_max, debug)
          else
            call elpa_transpose_row_or_col_&
                  &MATH_DATATYPE&
                  &_&
                  &PRECISION&
                  (obj, 'R', a, at_col, buf_send, buf_recv, buf_self, np_fine, l_rows, l_cols, &
                  nblk_mult_rows_max, nblk_mult_cols_max, debug)
          endif
        endif

        if (b_transposed) then
          if (useCCL) then
            call elpa_transpose_row_or_col_ccl_&
                  &MATH_DATATYPE&
                  &_&
                  &PRECISION&
                  (obj, 'C', b_dev, bt_row_dev, buf_send_dev, buf_recv_dev, buf_self_dev, np_fine, l_rows, l_cols, &
                  nblk_mult_rows_max, nblk_mult_cols_max, debug)
          else
            call elpa_transpose_row_or_col_&
                  &MATH_DATATYPE&
                  &_&
                  &PRECISION&
                  (obj, 'C', b, bt_row, buf_send, buf_recv, buf_self, np_fine, l_rows, l_cols, &
                  nblk_mult_rows_max, nblk_mult_cols_max, debug)
          endif
        endif

        if (mod(np_bc_fine,np_cols) == my_pcol) then
          if (useGPU) then
            if (a_transposed) then
              if (.not. useCCL) then
                num = l_rows*l_cols
#ifdef WITH_GPU_STREAMS
                call gpu_memcpy_async_and_stream_synchronize &
                    ("elpa_pxgemm_multiply: at_col -> at_col_dev", at_col_dev, 0_c_intptr_t, at_col(1:l_rows,1:l_cols), &
                      1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
                successGPU = gpu_memcpy(at_col_dev, int(loc(at_col),kind=c_intptr_t), num*size_of_datatype, gpuMemcpyHostToDevice)
                check_memcpy_gpu("elpa_pxgemm_multiply: at_col -> at_col_dev", successGPU)
#endif
              endif ! (.not. useCCL)

              call gpu_copy_and_set_zeros_aux_a_full(PRECISION_CHAR, at_col_dev, aux_a_full_dev, l_rows, &
                                                    nblk_mult_cols, nblk, np_bc_fine, np_cols_fine, np_cols, debug, my_stream)
            else ! (a_transposed .and. b_transposed)
              call gpu_copy_and_set_zeros_aux_a_full(PRECISION_CHAR, a_dev     , aux_a_full_dev, l_rows, &
                                                    nblk_mult_cols, nblk, np_bc_fine, np_cols_fine, np_cols, debug, my_stream)
            endif ! a_transposed
          else ! useGPU
            do j_block_loc_fine = 0, (nblk_mult_cols+nblk-1)/nblk - 1
              nblk_cut = min(nblk, nblk_mult_cols-j_block_loc_fine*nblk)
              j_block_loc = (np_bc_fine + j_block_loc_fine*np_cols_fine)/np_cols
              if (a_transposed) then
                aux_a_full(1:l_rows, 1+j_block_loc_fine*nblk: nblk_cut+j_block_loc_fine*nblk) = &
                    at_col(1:l_rows, 1+j_block_loc*nblk     : nblk_cut+j_block_loc*nblk)
              else ! a_transposed
                aux_a_full(1:l_rows, 1+j_block_loc_fine*nblk: nblk_cut+j_block_loc_fine*nblk) = &
                         a(1:l_rows, 1+j_block_loc*nblk     : nblk_cut+j_block_loc*nblk)
              endif ! a_transposed
            enddo ! j_block_loc_fine
          endif ! useGPU
        endif ! (mod(np_bc_fine,np_cols) == my_pcol)

        if (mod(np_fine,np_rows) == my_prow) then
          if (useGPU) then
            if (b_transposed) then
              if (.not. useCCL) then
                num = l_rows*l_cols
#ifdef WITH_GPU_STREAMS
                call gpu_memcpy_async_and_stream_synchronize &
                    ("elpa_pxgemm_multiply: bt_row -> bt_row_dev", bt_row_dev, 0_c_intptr_t, bt_row(1:l_rows,1:l_cols), &
                      1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
                successGPU = gpu_memcpy(bt_row_dev, int(loc(bt_row),kind=c_intptr_t), num*size_of_datatype, gpuMemcpyHostToDevice)
                check_memcpy_gpu("elpa_pxgemm_multiply: bt_row -> bt_row_dev", successGPU)
#endif
              endif ! (.not. useCCL)
              
              call gpu_copy_and_set_zeros_aux_b_full(PRECISION_CHAR, bt_row_dev, aux_b_full_dev, l_rows, l_cols, nblk_mult, &
                                                  nblk_mult_rows, nblk, np_fine, np_rows_fine, np_rows, SM_count, debug, my_stream)
            else  ! b_transposed
              call gpu_copy_and_set_zeros_aux_b_full(PRECISION_CHAR, b_dev, aux_b_full_dev, l_rows, l_cols, nblk_mult, &
                                                  nblk_mult_rows, nblk, np_fine, np_rows_fine, np_rows, SM_count, debug, my_stream)
            endif ! b_transposed
          else ! useGPU
            do i_block_loc_fine = 0, (nblk_mult_rows+nblk-1)/nblk - 1
              nblk_cut = min(nblk, nblk_mult_rows-i_block_loc_fine*nblk)
              i_block_loc = (np_fine + i_block_loc_fine*np_rows_fine)/np_rows
              if (b_transposed) then
                aux_b_full(1+i_block_loc_fine*nblk: nblk_cut+i_block_loc_fine*nblk, 1:l_cols) = &
                    bt_row(1+i_block_loc*nblk     : nblk_cut+i_block_loc*nblk     , 1:l_cols)
              else  ! b_transposed
                aux_b_full(1+i_block_loc_fine*nblk: nblk_cut+i_block_loc_fine*nblk, 1:l_cols) = &
                         b(1+i_block_loc*nblk     : nblk_cut+i_block_loc*nblk     , 1:l_cols)
              endif ! b_transposed
            enddo ! i_block_loc_fine
          endif ! useGPU
        endif ! (mod(np_fine,np_rows) == my_prow)


        ! copy data to host for bcast, if needed
        num_a = l_rows*nblk_mult
        num_b = nblk_mult*l_cols
        if (useGPU .and. .not. useCCL) then
#ifdef WITH_GPU_STREAMS
          call gpu_memcpy_async_and_stream_synchronize("elpa_pxgemm_multiply: aux_a_full_dev -> aux_a_full",  &
                aux_a_full_dev, 0_c_intptr_t, aux_a_full(1:l_rows,1:nblk_mult), &
                1, 1, num_a*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .false., .false.)
          call gpu_memcpy_async_and_stream_synchronize ("elpa_pxgemm_multiply: aux_b_full_dev -> aux_b_full", &
                aux_b_full_dev, 0_c_intptr_t, aux_b_full(1:nblk_mult,1:l_cols), &
                1, 1, num_b*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
          successGPU = gpu_memcpy(int(loc(aux_a_full),kind=c_intptr_t), aux_a_full_dev, num_a*size_of_datatype, & 
                                  gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_pxgemm_multiply: aux_a_full_dev -> aux_a_full", successGPU)

          successGPU = gpu_memcpy(int(loc(aux_b_full),kind=c_intptr_t), aux_b_full_dev, num_b*size_of_datatype, & 
                                  gpuMemcpyDeviceToHost)
          check_memcpy_gpu("elpa_pxgemm_multiply: aux_b_full_dev -> aux_b_full", successGPU)
#endif  
        endif ! (useGPU .and. .not. useCCL)

#ifdef WITH_MPI
        ! Broadcast processor column
        if (useCCL) then
          call obj%timer%start("ccl_bcast")
          NVTX_RANGE_PUSH("ccl_bcast aux_a_full_dev, aux_b_full_dev")

          successGPU  = ccl_bcast(aux_a_full_dev, aux_a_full_dev, int(k_datatype*l_rows*nblk_mult,kind=c_size_t), cclDatatype, &
                                  int(np_bc,kind=c_int), ccl_comm_cols, my_stream)

          successGPU2 = ccl_bcast(aux_b_full_dev, aux_b_full_dev, int(k_datatype*nblk_mult*l_cols,kind=c_size_t), cclDatatype, &
                                  int(np   ,kind=c_int), ccl_comm_rows, my_stream)
          
          if (.not. (successGPU .and. successGPU2)) then
            print *,"Error in ccl_bcast"
            stop 1
          endif

          call MPI_Bcast(nblk_mult_rows, 1_MPI_KIND, MPI_INTEGER, &
                         int(np   ,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("elpa_pxgemm_multiply: ccl_bcast", successGPU)

          NVTX_RANGE_POP("ccl_bcast aux_a_full_dev, aux_b_full_dev")
          call obj%timer%stop("ccl_bcast")
        else ! useCCL
          call obj%timer%start("mpi_communication")
          NVTX_RANGE_PUSH("MPI_Bcast: aux_a_full, aux_b_full")

          call MPI_Bcast(aux_a_full, int(l_rows*nblk_mult,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                        int(np_bc,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)

          call MPI_Bcast(aux_b_full, int(nblk_mult*l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                        int(np   ,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
          call MPI_Bcast(nblk_mult_rows, 1_MPI_KIND, MPI_INTEGER, &
                        int(np   ,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)

          NVTX_RANGE_POP("MPI_Bcast: aux_a_full, aux_b_full")
          call obj%timer%stop("mpi_communication")
        endif ! useCCL


        ! copy data back to device, if needed
        if (useGPU .and. .not. useCCL) then
          num_a = l_rows*nblk_mult
          num_b = nblk_mult*l_cols
#ifdef WITH_GPU_STREAMS
          call gpu_memcpy_async_and_stream_synchronize("elpa_pxgemm_multiply: aux_a_full -> aux_a_full_dev", &
                aux_a_full_dev, 0_c_intptr_t, aux_a_full(1:l_rows,1:nblk_mult), &
                1, 1, num_a*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
          call gpu_memcpy_async_and_stream_synchronize("elpa_pxgemm_multiply: aux_b_full -> aux_b_full_dev", &
                aux_b_full_dev, 0_c_intptr_t, aux_b_full(1:nblk_mult,1:l_cols), &
                1, 1, num_b*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
          successGPU = gpu_memcpy(aux_a_full_dev,int(loc(aux_a_full),kind=c_intptr_t), num_a*size_of_datatype, &
                                  gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_pxgemm_multiply: aux_a_full -> aux_a_full_dev", successGPU)
          
          successGPU = gpu_memcpy(aux_b_full_dev,int(loc(aux_b_full),kind=c_intptr_t), num_b*size_of_datatype, &
                                  gpuMemcpyHostToDevice)
          check_memcpy_gpu("elpa_pxgemm_multiply: aux_b_full -> aux_b_full_dev", successGPU)
#endif
        endif ! (useGPU .and. .not. useCCL)
#endif /* WITH_MPI */
        
        beta = ZERO
        if (np_fine>0) beta = ONE
        if (useGPU) then
          call obj%timer%start("gpublas")
          NVTX_RANGE_PUSH("gpublas")
          call gpublas_PRECISION_GEMM('N', 'N', &
                            l_rows, l_cols, nblk_mult_rows, ONE, &
                            aux_a_full_dev, l_rows, &
                            aux_b_full_dev, nblk_mult, beta, &
                            c_dev , l_rows, gpuHandle)
          if (wantDebug) successGPU = gpu_DeviceSynchronize()
          NVTX_RANGE_POP("gpublas")
          call obj%timer%stop("gpublas")
        else ! useGPU
          call obj%timer%start("blas")
          call PRECISION_GEMM('N', 'N', &
                              int(l_rows, kind=BLAS_KIND), &
                              int(l_cols, kind=BLAS_KIND), &
                              int(nblk_mult_rows, kind=BLAS_KIND), ONE, &
                              aux_a_full(1:l_rows, 1:nblk_mult), int(l_rows,kind=BLAS_KIND), &
                              aux_b_full(1:nblk_mult, 1:l_cols), int(nblk_mult,kind=BLAS_KIND), beta, &
                              c(1:l_rows, 1:l_cols) , int(l_rows,kind=BLAS_KIND))
          call obj%timer%stop("blas")
        endif ! useGPU

        NVTX_RANGE_POP("np_fine = 0, np_rows_fine-1")
      enddo ! np_fine = 0, np_rows_fine-1
      
      call obj%timer%stop("main_loop_nn_tt")

#if !defined(DEVICE_POINTER)
      ! Put the result into C
      if (useGPU) then
        call obj%timer%start("gpu_memcpy")
        NVTX_RANGE_PUSH("gpu_memcpy: c_dev->c")
        num = l_rows*l_cols
#ifdef WITH_GPU_STREAMS
        call gpu_memcpy_async_and_stream_synchronize &
            ("elpa_pxgemm_multiply: c_dev -> c", c_dev, 0_c_intptr_t, c(1:l_rows,1:l_cols), &
              1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
        successGPU = gpu_memcpy(int(loc(c),kind=c_intptr_t), c_dev, num*size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("elpa_pxgemm_multiply: c_dev -> c", successGPU)
#endif
        NVTX_RANGE_POP("gpu_memcpy: c_dev->c")
        call obj%timer%stop("gpu_memcpy")
      endif ! useGPU
#endif /* DEVICE_POINTER */

      
      if (a_transposed) then
        if (.not. useCCL) then
          deallocate(at_col, stat=istat, errmsg=errorMessage)
          call check_alloc("elpa_pxgemm_multiply", "at_col", istat, errorMessage)
        endif

        if (useGPU) then
          successGPU = gpu_free(at_col_dev)
          check_dealloc_gpu("elpa_pxgemm_multiply: at_col_dev", successGPU)
        endif
      endif

      if (b_transposed) then
        if (.not. useCCL) then
          deallocate(bt_row, stat=istat, errmsg=errorMessage)
          call check_alloc("elpa_pxgemm_multiply", "bt_row", istat, errorMessage)
        endif

        if (useGPU) then
          successGPU = gpu_free(bt_row_dev)
          check_dealloc_gpu("elpa_pxgemm_multiply: bt_row_dev", successGPU)
        endif
      endif

      if (a_transposed .or. b_transposed) then
        if (useCCL) then
          successGPU = gpu_free(buf_send_dev)
          check_dealloc_gpu("elpa_pxgemm_multiply: buf_send_dev", successGPU)

          successGPU = gpu_free(buf_recv_dev)
          check_dealloc_gpu("elpa_pxgemm_multiply: buf_recv_dev", successGPU)

          successGPU = gpu_free(buf_self_dev)
          check_dealloc_gpu("elpa_pxgemm_multiply: buf_self_dev", successGPU)
        else
          deallocate(buf_send, buf_recv, buf_self, stat=istat, errmsg=errorMessage)
          call check_alloc("elpa_pxgemm_multiply", "buf_send, buf_recv, buf_self", istat, errorMessage)

#if defined(DEVICE_POINTER)
          deallocate(a, b, stat=istat, errmsg=errorMessage)
          call check_alloc("elpa_pxgemm_multiply", "a, b", istat, errorMessage)
#endif          
        endif
      endif ! (a_transposed .or. b_transposed)

    endif ! universal

! ___________________________________________________________________
    if (.not. a_transposed .and.       b_transposed .or. &
              a_transposed .and. .not. b_transposed) then
      if (wantDebug .and. myid==0) print *, "elpa_pxgemm_multiply NEW: NON-SQUARE_GRID start: TN or NT" ! PETERDEBUG
      
      ! dir = row/col for TN/NT
      if (a_transposed) then
        my_pdir = my_prow
        my_pdir_t = my_pcol
        np_dirs = np_rows
        np_dirs_t = np_cols
        mpi_comm_dirs = mpi_comm_rows
        ccl_comm_dirs = obj%gpu_setup%ccl_comm_rows
      else if (b_transposed) then
        my_pdir = my_pcol
        my_pdir_t = my_prow
        np_dirs = np_cols
        np_dirs_t = np_rows
        mpi_comm_dirs = mpi_comm_cols
        ccl_comm_dirs = obj%gpu_setup%ccl_comm_cols
      endif

      np_dirs_fine = least_common_multiple(np_rows, np_cols)

      nblk_mult_rows = nblk_mult
      nblk_mult_cols = nblk_mult

      nblk_mult_max      = (na+LCM-1)/LCM * nblk
      !nblk_mult_cols_max = (na+LCM-1)/LCM * nblk
      !nblk_mult_rows_max = (na+LCM-1)/LCM * nblk

      if (a_transposed) then
          allocate(aux_a_full(nblk_mult, nblk_mult), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: aux_a_full", istat, errorMessage)
        
          allocate(aux_b_full(nblk_mult, nblk_mult_max*(np_dirs_fine/np_dirs_t)), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: aux_b_full", istat, errorMessage)
          
          allocate(tmp1_full(nblk_mult, nblk_mult_max*(np_dirs_fine/np_dirs_t)), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: tmp1_full", istat, errorMessage)
        
        if (useGPU) then
          successGPU = gpu_malloc(aux_a_full_dev, nblk_mult*nblk_mult*size_of_datatype)
          check_alloc_gpu("elpa_pxgemm_multiply: aux_a_full_dev", successGPU)

          successGPU = gpu_malloc(aux_b_full_dev, nblk_mult*nblk_mult_max*(np_dirs_fine/np_dirs_t)*size_of_datatype)
          check_alloc_gpu("elpa_pxgemm_multiply: aux_b_full_dev", successGPU)

          successGPU = gpu_malloc(tmp1_full_dev , nblk_mult*nblk_mult_max*(np_dirs_fine/np_dirs_t)*size_of_datatype)
          check_alloc_gpu("elpa_pxgemm_multiply: tmp1_full_dev", successGPU)
        endif ! useGPU
      else if (b_transposed) then
          allocate(aux_a_full(nblk_mult_max*(np_dirs_fine/np_dirs_t), nblk_mult), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: aux_a_full", istat, errorMessage)
        
          allocate(aux_b_full(nblk_mult, nblk_mult), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: aux_b_full", istat, errorMessage)
          
          allocate(tmp1_full(nblk_mult_max*(np_dirs_fine/np_dirs_t), nblk_mult), stat=istat, errmsg=errorMessage)
          check_allocate("elpa_pxgemm_multiply: tmp1_full", istat, errorMessage)
        if (useGPU) then
          successGPU = gpu_malloc(aux_a_full_dev, nblk_mult_max*(np_dirs_fine/np_dirs_t)*nblk_mult*size_of_datatype)
          check_alloc_gpu("elpa_pxgemm_multiply: aux_a_full_dev", successGPU)

          successGPU = gpu_malloc(aux_b_full_dev, nblk_mult*nblk_mult*size_of_datatype)
          check_alloc_gpu("elpa_pxgemm_multiply: aux_b_full_dev", successGPU)

          successGPU = gpu_malloc(tmp1_full_dev , nblk_mult_max*(np_dirs_fine/np_dirs_t)*nblk_mult*size_of_datatype)
          check_alloc_gpu("elpa_pxgemm_multiply: tmp1_full_dev", successGPU)
        endif ! useGPU
      endif ! b_transposed
      
      call obj%timer%start("main_loop_tn_nt")

      ! main loop: build up the result matrix C by the (virtual) process rows/cols for TN/NT
      do np_fine = 0, np_dirs_fine-1
        NVTX_RANGE_PUSH("do np_fine")
        if (wantDebug .and. myid==0) print *, "np_fine = ", np_fine ! PETERDEBUG

        ! In this turn, procs of row/col np assemble the result for TN/NT case
        
        np_t_fine=np_fine ! np, that posesses the given "col of a"/"row of b" for TN/NT

        !np   = mod(np_fine, np_rows)
        !np_t = mod(np_t_fine, np_cols)

        np   = mod(np_fine, np_dirs)
        np_t = mod(np_t_fine, np_dirs_t)

        ! For non-square grids we work on LCMxLCM blocks. 
        ! One block consists of nblk x nblk block in the first LCM*LCM block and all its copies in the other LCM*LCM blocks.
        ! TN: loop over fine rows of a
        ! NT: loop over fine cols of b
        do np_ab_fine = 0, np_dirs_fine-1
        !do dnp_ab = 0, np_dirs_fine/np_dirs-1
          
          ! consider only the fine rows/cols that belong to the current process
          if (mod(np_ab_fine,np_dirs)/=my_pdir) cycle

          if (useGPU) then
            call obj%timer%start("gpu_copy_and_set_zeros_aux_ab_full_tn")
            call gpu_copy_and_set_zeros_aux_ab_full_tn_nt (PRECISION_CHAR, a_transposed_int, &
                                                a_dev, b_dev, aux_a_full_dev, aux_b_full_dev, &
                                                l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                                np_ab_fine, np_rows, my_prow, &
                                                np_t_fine , np_cols, my_pcol, &
                                                np_dirs_fine, SM_count, debug, my_stream)
            call obj%timer%stop("gpu_copy_and_set_zeros_aux_ab_full_tn")
          else ! useGPU
            call obj%timer%start("copy_and_set_zeros_aux_ab_full_tn")

            if (a_transposed) then
              if (mod(np_t_fine,np_cols) == my_pcol) then
                do j_block_loc_fine = 0, nblk_mult_max/nblk-1
                  j_block_loc = (np_t_fine + j_block_loc_fine*np_cols_fine)/np_cols
                  
                  do i_block_loc_fine = 0, nblk_mult/nblk-1
                    i_block_loc = (np_ab_fine + i_block_loc_fine*np_rows_fine)/np_rows
                    
                    nblk_cols_cut = min(nblk, l_cols - j_block_loc*nblk)
                    nblk_rows_cut = min(nblk, l_rows - i_block_loc*nblk)

                    if (nblk_rows_cut>0 .and. nblk_cols_cut>0) then
                      aux_a_full(1+i_block_loc_fine*nblk : nblk_rows_cut+i_block_loc_fine*nblk, &
                                 1+j_block_loc_fine*nblk : nblk_cols_cut+j_block_loc_fine*nblk) = &
                              a (1+i_block_loc*nblk      : nblk_rows_cut+i_block_loc*nblk, &
                                 1+j_block_loc*nblk      : nblk_cols_cut+j_block_loc*nblk)
                    endif

                    call set_zeros_in_unused_block_part_&
                                  &MATH_DATATYPE&
                                  &_&
                                  &PRECISION &
                                  (aux_a_full, nblk, nblk_rows_cut, nblk_cols_cut, &
                                  i_block_loc_fine, j_block_loc_fine, 0, 0)

                    
                  enddo ! i_block_loc_fine
                enddo ! j_block_loc_fine
              endif ! (mod(np_t_fine,np_cols) == my_pcol)

              do dnp_ab_t = 0, np_dirs_fine/np_dirs_t-1
                np_ab_t_fine = dnp_ab_t*np_dirs_t + my_pdir_t

                do j_block_loc_fine = 0, nblk_mult_max/nblk-1
                  j_block_loc = (np_ab_t_fine + j_block_loc_fine*np_cols_fine)/np_cols
                  
                  do i_block_loc_fine = 0, nblk_mult/nblk-1
                    i_block_loc = (np_ab_fine + i_block_loc_fine*np_rows_fine)/np_rows

                    nblk_rows_cut = min(nblk, l_rows - i_block_loc*nblk)
                    nblk_cols_cut = min(nblk, l_cols - j_block_loc*nblk)
                    
                    if (nblk_rows_cut>0 .and. nblk_cols_cut>0) then
                      aux_b_full(1+i_block_loc_fine*nblk : nblk_rows_cut+i_block_loc_fine*nblk, &
                                 1            +j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max : &
                                 nblk_cols_cut+j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max) = &
                               b(1+i_block_loc*nblk      :nblk_rows_cut+i_block_loc*nblk, &
                                 1+j_block_loc*nblk      :nblk_cols_cut+j_block_loc*nblk)
                    endif

                    call set_zeros_in_unused_block_part_&
                                  &MATH_DATATYPE&
                                  &_&
                                  &PRECISION &
                                  (aux_b_full, nblk, nblk_rows_cut, nblk_cols_cut, &
                                  i_block_loc_fine, j_block_loc_fine, 0, dnp_ab_t*nblk_mult_max)

                  enddo ! i_block_loc_fine
                enddo ! j_block_loc_fine
              enddo ! np_ab_t_fine
            else if (b_transposed) then

              if (mod(np_t_fine, np_rows) == my_prow) then
                do i_block_loc_fine = 0, nblk_mult_max/nblk-1
                  i_block_loc = (np_t_fine + i_block_loc_fine*np_rows_fine)/np_rows
          
                  do j_block_loc_fine = 0, nblk_mult/nblk-1
                    j_block_loc = (np_ab_fine + j_block_loc_fine*np_cols_fine)/np_cols
          
                    nblk_rows_cut = min(nblk, l_rows - i_block_loc*nblk)
                    nblk_cols_cut = min(nblk, l_cols - j_block_loc*nblk)
                    
                    if (nblk_rows_cut>0 .and. nblk_cols_cut>0) then
                      aux_b_full(1+i_block_loc_fine*nblk : nblk_rows_cut+i_block_loc_fine*nblk, &
                                1+j_block_loc_fine*nblk : nblk_cols_cut+j_block_loc_fine*nblk) = &
                              b(1+i_block_loc*nblk      : nblk_rows_cut+i_block_loc*nblk, &
                                1+j_block_loc*nblk      : nblk_cols_cut+j_block_loc*nblk)
                    endif

                    call set_zeros_in_unused_block_part_&
                                  &MATH_DATATYPE&
                                  &_&
                                  &PRECISION &
                                  (aux_b_full, nblk, nblk_rows_cut, nblk_cols_cut, &
                                  i_block_loc_fine, j_block_loc_fine, 0, 0)

                  enddo ! j_block_loc_fine
                enddo ! i_block_loc_fine
              endif ! (mod(np_t_fine, np_rows) == my_prow)
          
              do dnp_ab_t = 0, np_dirs_fine/np_dirs_t-1
                np_ab_t_fine = dnp_ab_t*np_dirs_t + my_pdir_t
          
                do i_block_loc_fine = 0, nblk_mult_max/nblk-1
                  i_block_loc = (np_ab_t_fine + i_block_loc_fine*np_rows_fine)/np_rows
          
                  do j_block_loc_fine = 0, nblk_mult/nblk-1
                    j_block_loc = (np_ab_fine + j_block_loc_fine*np_cols_fine)/np_cols
          
                    nblk_rows_cut = min(nblk, l_rows - i_block_loc*nblk)
                    nblk_cols_cut = min(nblk, l_cols - j_block_loc*nblk)
                    
                    if (nblk_rows_cut>0 .and. nblk_cols_cut>0) then
                      aux_a_full(1            +i_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max : &
                                 nblk_rows_cut+i_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max, &
                                  1+j_block_loc_fine*nblk : nblk_cols_cut+j_block_loc_fine*nblk) = &
                                a(1+i_block_loc*nblk      : nblk_rows_cut+i_block_loc*nblk, &
                                  1+j_block_loc*nblk      : nblk_cols_cut+j_block_loc*nblk)
                    endif
                    
                    call set_zeros_in_unused_block_part_&
                                  &MATH_DATATYPE&
                                  &_&
                                  &PRECISION &
                                  (aux_a_full, nblk, nblk_rows_cut, nblk_cols_cut, &
                                  i_block_loc_fine, j_block_loc_fine, dnp_ab_t*nblk_mult_max, 0)

                  enddo ! j_block_loc_fine
                enddo ! i_block_loc_fine
              enddo ! dnp_ab_t
            endif ! b_transposed
            call obj%timer%stop("copy_and_set_zeros_aux_ab_full_tn")
          endif ! useGPU

          ! aux_a_full/aux_b_full -> aux_ab_trans (aux_ab) ! auxillary buffer for a matrix that is     to be transposed: a/b in TN/NT case
          ! aux_b_full/aux_a_full -> aux_ab_nontr (aux_ba) ! auxillary buffer for a matrix that is not to be transposed: b/a in TN/NT case
          !enddo ! dnp_ab -> later

#ifdef WITH_MPI
          if (useGPU .and. .not. useCCL) then
            if (my_pdir_t == np_t) then
              call obj%timer%start("gpu_memcpy")
              num = nblk_mult*nblk_mult
              if (a_transposed) then
#ifdef WITH_GPU_STREAMS
                call gpu_memcpy_async_and_stream_synchronize("elpa_pxgemm_multiply: aux_a_full_dev -> aux_a_full", &
                      aux_a_full_dev, 0_c_intptr_t, aux_a_full(1:nblk_mult,1:nblk_mult),&
                      1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
                successGPU = gpu_memcpy(int(loc(aux_a_full),kind=c_intptr_t), aux_a_full_dev, &
                                        num*size_of_datatype, gpuMemcpyDeviceToHost)
                check_memcpy_gpu("elpa_pxgemm_multiply: aux_a_full_dev -> aux_a_full", successGPU)
#endif
              else if (b_transposed) then
#ifdef WITH_GPU_STREAMS
                call gpu_memcpy_async_and_stream_synchronize("elpa_pxgemm_multiply: aux_b_full_dev -> aux_b_full", &
                      aux_b_full_dev, 0_c_intptr_t, aux_b_full(1:nblk_mult,1:nblk_mult),&
                      1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
                successGPU = gpu_memcpy(int(loc(aux_b_full),kind=c_intptr_t), aux_b_full_dev, &
                                        nblk_mult*nblk_mult*size_of_datatype, gpuMemcpyDeviceToHost)
                check_memcpy_gpu("elpa_pxgemm_multiply: aux_b_full_dev -> aux_b_full", successGPU)
#endif
              endif ! b_transposed
              call obj%timer%stop("gpu_memcpy")
            endif ! (my_pdir_t == np_t)
          endif ! (useGPU .and. .not. useCCL)

          ! Broadcast processor column/row
          if (useCCL) then
            call obj%timer%start("ccl_bcast")
            NVTX_RANGE_PUSH("ccl_bcast")

            if (a_transposed) then
              successGPU  = ccl_bcast(aux_a_full_dev, aux_a_full_dev, int(k_datatype*nblk_mult*nblk_mult,kind=c_size_t), &
                                      cclDatatype, int(np_t,kind=c_int), ccl_comm_cols, my_stream)
            else if (b_transposed) then
              successGPU  = ccl_bcast(aux_b_full_dev, aux_b_full_dev, int(k_datatype*nblk_mult*nblk_mult,kind=c_size_t), &
                                      cclDatatype, int(np_t,kind=c_int), ccl_comm_rows, my_stream)
            endif

            if (.not. successGPU) then
              print *,"Error in ccl_bcast"
              stop 1
            endif

            successGPU = gpu_stream_synchronize(my_stream)
            check_stream_synchronize_gpu("elpa_pxgemm_multiply: ccl_bcast", successGPU)

            NVTX_RANGE_POP("ccl_bcast aux_a_full_dev, aux_b_full_dev")
            call obj%timer%stop("ccl_bcast")
          else ! useCCL
            call obj%timer%start("mpi_bcast")
            NVTX_RANGE_PUSH("MPI_Bcast(aux_a/b_full)")

            if (a_transposed) then
              call MPI_Bcast(aux_a_full, int(nblk_mult*nblk_mult,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                            int(np_t,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
            else if (b_transposed) then
              call MPI_Bcast(aux_b_full, int(nblk_mult*nblk_mult,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION, &
                            int(np_t,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
            endif

            NVTX_RANGE_POP("MPI_Bcast(aux_a/b_full)")
            call obj%timer%stop("mpi_bcast")
          endif ! useCCL

          if (useGPU .and. .not. useCCL) then
            call obj%timer%start("gpu_memcpy")
            num = nblk_mult*nblk_mult
            if (a_transposed) then
#ifdef WITH_GPU_STREAMS
              call gpu_memcpy_async_and_stream_synchronize("elpa_pxgemm_multiply: aux_a_full -> aux_a_full_dev", &
                    aux_a_full_dev, 0_c_intptr_t, aux_a_full(1:nblk_mult,1:nblk_mult), &
                    1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
              successGPU = gpu_memcpy(aux_a_full_dev, int(loc(aux_a_full),kind=c_intptr_t), &
                                      nblk_mult*nblk_mult*size_of_datatype, gpuMemcpyHostToDevice)
              check_memcpy_gpu("elpa_pxgemm_multiply: aux_a_full -> aux_a_full_dev", successGPU)
#endif
            else if (b_transposed) then
#ifdef WITH_GPU_STREAMS
              call gpu_memcpy_async_and_stream_synchronize ("elpa_pxgemm_multiply: aux_b_full -> aux_b_full_dev", &
                    aux_b_full_dev, 0_c_intptr_t, aux_b_full(1:nblk_mult,1:nblk_mult), &
                    1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
              successGPU = gpu_memcpy(aux_b_full_dev, int(loc(aux_b_full),kind=c_intptr_t), &
                                    nblk_mult*nblk_mult*size_of_datatype, gpuMemcpyHostToDevice)
              check_memcpy_gpu("elpa_pxgemm_multiply: aux_b_full -> aux_b_full_dev", successGPU)
#endif
            endif ! b_transposed
            call obj%timer%stop("gpu_memcpy")
          endif ! (useGPU .and. .not. useCCL)
#endif /* WITH_MPI */
   
          if (useGPU) then
            call obj%timer%start("gpublas")
            NVTX_RANGE_PUSH("gpublas")
            if (a_transposed) then
              call gpublas_PRECISION_GEMM(trans_a_cchar, trans_b_cchar, &
                                          nblk_mult, &
                                          nblk_mult_max*(np_dirs_fine/np_dirs_t), &
                                          nblk_mult, ONE, &
                                          aux_a_full_dev, nblk_mult, &
                                          aux_b_full_dev, nblk_mult, ZERO, &
                                          tmp1_full_dev , nblk_mult, gpuHandle)
            else if (b_transposed) then
              call gpublas_PRECISION_GEMM(trans_a_cchar, trans_b_cchar, &
                                          nblk_mult_max*(np_dirs_fine/np_dirs_t), &
                                          nblk_mult, &
                                          nblk_mult, ONE, &
                                          aux_a_full_dev, nblk_mult_max*(np_dirs_fine/np_dirs_t), &
                                          aux_b_full_dev, nblk_mult, ZERO, &
                                          tmp1_full_dev , nblk_mult_max*(np_dirs_fine/np_dirs_t), gpuHandle)
            endif ! b_transposed
            if (wantDebug) successGPU = gpu_DeviceSynchronize()
            NVTX_RANGE_POP("gpublas")
            call obj%timer%stop("gpublas")
          else ! useGPU
            call obj%timer%start("blas")
            if (a_transposed) then
              call PRECISION_GEMM(trans_a, trans_b, &
                                  int(nblk_mult, kind=BLAS_KIND), &
                                  int(nblk_mult_max*(np_dirs_fine/np_dirs_t), kind=BLAS_KIND), &
                                  int(nblk_mult, kind=BLAS_KIND), ONE, &
                                  aux_a_full(1:nblk_mult,1:nblk_mult), int(nblk_mult, kind=BLAS_KIND), &
                                  aux_b_full(1:nblk_mult,1:nblk_mult_max*(np_dirs_fine/np_dirs_t)), &
                                  int(nblk_mult, kind=BLAS_KIND), ZERO, &
                                  tmp1_full (1:nblk_mult,1:nblk_mult_max*(np_dirs_fine/np_dirs_t)), &
                                  int(nblk_mult, kind=BLAS_KIND))
            else if (b_transposed) then
              call PRECISION_GEMM(trans_a, trans_b, &
                                  int(nblk_mult_max*(np_dirs_fine/np_dirs_t), kind=BLAS_KIND), &
                                  int(nblk_mult, kind=BLAS_KIND), &
                                  int(nblk_mult, kind=BLAS_KIND), ONE, &
                                  aux_a_full(1:nblk_mult_max*(np_dirs_fine/np_dirs_t),1:nblk_mult),&
                                  int(nblk_mult_max*(np_dirs_fine/np_dirs_t), kind=BLAS_KIND), &
                                  aux_b_full(1:nblk_mult,1:nblk_mult), int(nblk_mult, kind=BLAS_KIND), ZERO, &
                                  tmp1_full (1:nblk_mult_max*(np_dirs_fine/np_dirs_t),1:nblk_mult), &
                                  int(nblk_mult_max*(np_dirs_fine/np_dirs_t), kind=BLAS_KIND))
            endif ! b_transposed
            call obj%timer%stop("blas")
          endif ! useGPU

#ifdef WITH_MPI
          if (a_transposed) then
            num_r = nblk_mult
            num_c = nblk_mult_max*(np_dirs_fine/np_dirs_t)
          else if (b_transposed) then
            num_r = nblk_mult_max*(np_dirs_fine/np_dirs_t)
            num_c = nblk_mult
          endif

          num = nblk_mult*nblk_mult_max*(np_dirs_fine/np_dirs_t)
          
          if (useGPU .and. .not. useCCL) then
            call obj%timer%start("gpu_memcpy")
#ifdef WITH_GPU_STREAMS
            call gpu_memcpy_async_and_stream_synchronize &
                ("elpa_pxgemm_multiply: tmp1_full_dev -> tmp1_full", tmp1_full_dev, 0_c_intptr_t, &
                  tmp1_full(1:num_r,1:num_c), &
                  1, 1, num*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
            successGPU = gpu_memcpy(int(loc(tmp1_full),kind=c_intptr_t), tmp1_full_dev, &
                                    num*size_of_datatype, gpuMemcpyDeviceToHost)
            check_memcpy_gpu("elpa_pxgemm_multiply: tmp1_full_dev -> tmp1_full", successGPU)
#endif
            call obj%timer%stop("gpu_memcpy")
          endif ! (useGPU .and. .not. useCCL) 
          
          if (useCCL) then
            call obj%timer%start("ccl_reduce")
            NVTX_RANGE_PUSH("ccl_reduce tmp1_full_dev")

            successGPU  = ccl_reduce(tmp1_full_dev, tmp1_full_dev, int(k_datatype*num, kind=c_size_t), &
                                     cclDatatype, cclSum, int(np, kind=c_int), ccl_comm_dirs, my_stream)
          
            if (.not. successGPU) then
              print *,"Error in ccl_reduce"
              stop 1
            endif

            successGPU = gpu_stream_synchronize(my_stream)
            check_stream_synchronize_gpu("elpa_pxgemm_multiply: ccl_bcast", successGPU)

            NVTX_RANGE_POP("ccl_bcast aux_a_full_dev, aux_b_full_dev")
            call obj%timer%stop("ccl_reduce")
          else ! useCCL
            call obj%timer%start("mpi_reduce")
            if (my_pdir==np) then
              call MPI_Reduce(MPI_IN_PLACE, tmp1_full, int(num, kind=MPI_KIND), & 
                              MPI_MATH_DATATYPE_PRECISION, MPI_SUM, int(np,kind=MPI_KIND), int(mpi_comm_dirs,kind=MPI_KIND), mpierr)
            else
              call MPI_Reduce(tmp1_full   , tmp1_full, int(num, kind=MPI_KIND), &
                              MPI_MATH_DATATYPE_PRECISION, MPI_SUM, int(np,kind=MPI_KIND), int(mpi_comm_dirs,kind=MPI_KIND), mpierr)
            endif
            call obj%timer%stop("mpi_reduce")
          endif ! useCCL

          if (useGPU .and. .not. useCCL) then
            call obj%timer%start("gpu_memcpy")
#ifdef WITH_GPU_STREAMS
            call gpu_memcpy_async_and_stream_synchronize &
                ("elpa_pxgemm_multiply: tmp1_full -> tmp1_full_dev", 0_c_intptr_t, tmp1_full_dev, &
                  tmp1_full(1:num_r,1:num_c), &
                  1, 1, num*size_of_datatype, gpuMemcpyHostToDevice, my_stream, .false., .true., .false.)
#else
            successGPU = gpu_memcpy(tmp1_full_dev, int(loc(tmp1_full),kind=c_intptr_t), &
                                    num*size_of_datatype, gpuMemcpyHostToDevice)
            check_memcpy_gpu("elpa_pxgemm_multiply: tmp1_full -> tmp1_full_dev", successGPU)
#endif
            call obj%timer%stop("gpu_memcpy")
          endif ! (useGPU .and. .not. useCCL)
#endif /* WITH_MPI */


          if (my_pdir==np) then
            if (useGPU) then
              call obj%timer%start("gpu_update_c_tn_nt")
              beta_int = 1
              dnp_ab = np_ab_fine/np_dirs
              if (dnp_ab == 0) beta_int = 0
              ! Peter: for ELPA: ldc=l_rows (i.e. ld of c_dev and ld of tmp1_full_dev are same)
              ! but generally it's possible that ldc is not equal to l_rows --> extra parameter in
              ! kernel is needed
              call gpu_update_c_tn_nt(PRECISION_CHAR, a_transposed_int, &
                                      c_dev, tmp1_full_dev, beta_int, &
                                      l_rows, l_cols, nblk_mult_max, nblk_mult, nblk, &
                                      np_rows, np_cols, np_dirs_fine, &
                                      np_dirs_t, my_pdir_t, np_fine, &
                                      SM_count, debug, my_stream)
              call obj%timer%stop("gpu_update_c_tn_nt")
            else
              ! Put the result into C
              call obj%timer%start("update_c_tn_nt")
              first_call = .false.
              dnp_ab = np_ab_fine/np_dirs
              if (dnp_ab == 0) first_call = .true.

              do dnp_ab_t = 0, np_dirs_fine/np_dirs_t-1
                np_ab_t_fine = dnp_ab_t*np_dirs_t + my_pdir_t
                
                if (a_transposed) then
                  do j_block_loc_fine = 0, nblk_mult_max/nblk-1
                    j_block_loc = (np_ab_t_fine + j_block_loc_fine*np_cols_fine)/np_cols
                    
                    do i_block_loc_fine = 0, nblk_mult/nblk-1
                      i_block_loc = (np_fine + i_block_loc_fine*np_rows_fine)/np_rows

                      nblk_rows_cut = min(nblk, l_rows - i_block_loc*nblk)
                      nblk_cols_cut = min(nblk, l_cols - j_block_loc*nblk)
                      
                      if (nblk_rows_cut>0 .and. nblk_cols_cut>0) then
                        if (first_call) then
                          c(1+i_block_loc*nblk:nblk_rows_cut+i_block_loc*nblk, &
                            1+j_block_loc*nblk:nblk_cols_cut+j_block_loc*nblk) = &
                          tmp1_full(1+i_block_loc_fine*nblk :nblk_rows_cut+i_block_loc_fine*nblk, &
                                    1            +j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max : &
                                    nblk_cols_cut+j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max)
                        else
                          c(1+i_block_loc*nblk:nblk_rows_cut+i_block_loc*nblk, &
                            1+j_block_loc*nblk:nblk_cols_cut+j_block_loc*nblk) = &
                          c(1+i_block_loc*nblk:nblk_rows_cut+i_block_loc*nblk, &
                            1+j_block_loc*nblk:nblk_cols_cut+j_block_loc*nblk) + &
                          tmp1_full(1+i_block_loc_fine*nblk :nblk_rows_cut+i_block_loc_fine*nblk, &
                                    1            +j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max : &
                                    nblk_cols_cut+j_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max)
                        endif
                      endif                    
                    enddo ! i_block_loc_fine
                  enddo ! j_block_loc_fine
                else if (b_transposed) then
                  do i_block_loc_fine = 0, nblk_mult_max/nblk-1
                    i_block_loc = (np_ab_t_fine + i_block_loc_fine*np_rows_fine)/np_rows
            
                    do j_block_loc_fine = 0, nblk_mult/nblk-1
                      j_block_loc = (np_fine + j_block_loc_fine*np_cols_fine)/np_cols
            
                      nblk_rows_cut = min(nblk, l_rows - i_block_loc*nblk)
                      nblk_cols_cut = min(nblk, l_cols - j_block_loc*nblk)
                      
                      if (nblk_rows_cut>0 .and. nblk_cols_cut>0) then
                        if (first_call) then
                          c(1+i_block_loc*nblk : nblk_rows_cut+i_block_loc*nblk, &
                            1+j_block_loc*nblk : nblk_cols_cut+j_block_loc*nblk) = &
                          tmp1_full(1            +i_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max : &
                                    nblk_rows_cut+i_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max, &
                                    1+j_block_loc_fine*nblk : nblk_cols_cut+j_block_loc_fine*nblk)
                        else
                          c(1+i_block_loc*nblk : nblk_rows_cut+i_block_loc*nblk, &
                            1+j_block_loc*nblk : nblk_cols_cut+j_block_loc*nblk) = &
                          c(1+i_block_loc*nblk : nblk_rows_cut+i_block_loc*nblk, &
                            1+j_block_loc*nblk : nblk_cols_cut+j_block_loc*nblk) + &
                          tmp1_full(1            +i_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max : &
                                    nblk_rows_cut+i_block_loc_fine*nblk+dnp_ab_t*nblk_mult_max, &
                                    1+j_block_loc_fine*nblk : nblk_cols_cut+j_block_loc_fine*nblk)
                        endif
                      endif
                    enddo ! j_block_loc_fine
                  enddo ! i_block_loc_fine
                endif
              enddo ! dnp_ab_t = 0, np_dirs_fine/np_dirs_t-1
              call obj%timer%stop("update_c_tn_nt")
            endif ! useGPU
          endif ! (my_pdir==np)
          

        enddo ! do np_ab_fine = 0, np_dirs_fine-1
        
        NVTX_RANGE_POP("do np = 0, np_dirs-1")
      enddo ! np = 0, np_dirs-1
      call obj%timer%stop("main_loop_tn_nt")

#if !defined(DEVICE_POINTER)
      if (useGPU) then
        call obj%timer%start("gpu_memcpy")
        NVTX_RANGE_PUSH("gpu_memcpy: c_dev->c")
#ifdef WITH_GPU_STREAMS
        call gpu_memcpy_async_and_stream_synchronize &
          ("elpa_pxgemm_multiply: c_dev -> c", c_dev, 0_c_intptr_t, c(1:ldc,1:ldcCols), &
            1, 1, ldc*ldcCols*size_of_datatype, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
        successGPU = gpu_memcpy(int(loc(c),kind=c_intptr_t), c_dev, &
                                ldc*ldcCols*size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("elpa_pxgemm_multiply: c_dev -> c", successGPU)
#endif
        NVTX_RANGE_POP("gpu_memcpy: c_dev->c")
        call obj%timer%stop("gpu_memcpy")
      endif ! useGPU 
#endif /* DEVICE_POINTER */

      deallocate(tmp1_full,  stat=istat, errmsg=errorMessage)
      call check_alloc("elpa_pxgemm_multiply", "tmp1_full", istat, errorMessage)

      if (useGPU) then
        successGPU = gpu_free(tmp1_full_dev)
        check_dealloc_gpu("elpa_pxgemm_multiply: tmp1_full_dev", successGPU)
      endif
    endif ! (.not. a_transposed .and. b_transposed)

! ___________________________________________________________________ 

  endif ! (.not. isSquareGrid)
  
!______________________________________________________________________________________________

    deallocate(aux_a_full, stat=istat, errmsg=errorMessage)
    call check_alloc("elpa_pxgemm_multiply", "aux_a_full", istat, errorMessage)

    deallocate(aux_b_full, stat=istat, errmsg=errorMessage)
    call check_alloc("elpa_pxgemm_multiply", "aux_b_full", istat, errorMessage)
    
    if (useGPU) then
      successGPU = gpu_free(aux_a_full_dev)
      check_dealloc_gpu("elpa_pxgemm_multiply: aux_a_full_dev", successGPU)

      successGPU = gpu_free(aux_b_full_dev)
      check_dealloc_gpu("elpa_pxgemm_multiply: aux_b_full_dev", successGPU)
    endif

!______________________________________________________________________________________________

  if (useGPU) then
#if !defined(DEVICE_POINTER)
  successGPU = gpu_free(a_dev)
  check_dealloc_gpu("elpa_pxgemm_multiply: a_dev", successGPU)

  successGPU = gpu_free(b_dev)
  check_dealloc_gpu("elpa_pxgemm_multiply: b_dev", successGPU)

  successGPU = gpu_free(c_dev)
  check_dealloc_gpu("elpa_pxgemm_multiply: c_dev", successGPU)
#endif /* DEVICE_POINTER */
  endif ! useGPU

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION)
  if (useGPU) then
    successGPU = gpu_get_last_error()
    if (.not. successGPU) then
      write(error_unit,*) "elpa1_template: GPU error detected via gpu_get_last_error(). Aborting..."
      write(error_unit,*) "Rerun the program with the debug option e.g. 'export ELPA_DEFAULT_debug=1'"
      stop 1
    endif
  endif
#endif

  call obj%timer%stop("elpa_pxgemm_multiply"//trim(tnString)//trim(gpuString))

  NVTX_RANGE_POP("elpa_pxgemm_multiply")