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

#undef USE_CCL_TRIDIAG
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
#define USE_CCL_TRIDIAG
#endif

#undef SAVE_MATR
#ifdef DOUBLE_PRECISION_REAL
#define SAVE_MATR(name, iteration) \
call prmat(na, useGpu, a_mat, a_dev, matrixRows, matrixCols, nblk, my_prow, my_pcol, np_rows, np_cols, name, iteration)
#else
#define SAVE_MATR(name, iteration)
#endif

!> \brief Reduces a distributed symmetric matrix to tridiagonal form (like Scalapack Routine PDSYTRD)
!>
!  Parameters
!
!> \param obj	      object of elpa_type
!> \param na          Order of matrix
!>
!> \param a_mat(matrixRows,matrixCols)    Distributed matrix which should be reduced.
!>              Distribution is like in Scalapack.
!>              Opposed to PDSYTRD, a(:,:) must be set completely (upper and lower half)
!>              a(:,:) is overwritten on exit with the Householder vectors
!>
!> \param matrixRows         Leading dimension of a
!>
!> \param nblk        blocksize of cyclic distribution, must be the same in both directions!
!>
!> \param matrixCols  local columns of matrix
!>
!> \param mpi_comm_rows        MPI-Communicator for rows
!> \param mpi_comm_cols        MPI-Communicator for columns
!>
!> \param d_vec(na)       Diagonal elements (returned), identical on all processors
!>
!> \param e_vec(na)       Off-Diagonal elements (returned), identical on all processors
!>
!> \param tau(na)     Factors for the Householder vectors (returned), needed for back transformation
!>
!> \param useGPU      If true,  GPU version of the subroutine will be used
!> \param wantDebug   if true more debug information
!>

#ifdef TRIDIAG_GPU_BUILD
subroutine tridiag_gpu_&
#else
subroutine tridiag_cpu_&
#endif
   &MATH_DATATYPE&
   &_&
   &PRECISION &
  (obj, na, &
#ifdef TRIDIAG_GPU_BUILD
   a_dev, &  
#else
   a_mat, &
#endif
  matrixRows, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, &
#ifdef TRIDIAG_GPU_BUILD
  d_vec_dev, e_vec_dev, tau_dev, &
#else
  d_vec, e_vec, tau, &
#endif
  wantDebug, max_threads_in, isSkewsymmetric, success)

  use, intrinsic :: iso_c_binding
  use precision
  use elpa_abstract_impl
  use matrix_plot
  use elpa_omp
  use elpa_blas_interfaces
  use elpa_gpu
  use elpa_gpu_util
  use tridiag_gpu
#ifdef WITH_NVIDIA_GPU_VERSION
  use cuda_functions ! for NVTX labels
#endif
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
  use elpa_ccl_gpu
#endif


  implicit none
#include "../general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout)    :: obj
  integer(kind=ik), intent(in)                  :: na, matrixRows, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
  logical, intent(in)                           :: wantDebug
  logical, intent(in)                           :: isSkewsymmetric

  logical                                       :: useCCL=.false.

  ! input/output GPU pointers
  integer(kind=C_intptr_T)                      :: tau_dev, a_dev, d_vec_dev, e_vec_dev 
#ifndef TRIDIAG_GPU_BUILD
  MATH_DATATYPE(kind=rck), intent(out)          :: tau(na)
#ifdef USE_ASSUMED_SIZE
  MATH_DATATYPE(kind=rck), intent(inout)        :: a_mat(matrixRows,*)
#else
  MATH_DATATYPE(kind=rck), intent(inout)        :: a_mat(matrixRows,matrixCols)
#endif
  real(kind=rk), intent(out)                    :: d_vec(na)
  real(kind=rk), intent(out)                    :: e_vec(na)

#else /* TRIDIAG_GPU_BUILD */
  MATH_DATATYPE(kind=rck)                       :: tau(na)
  MATH_DATATYPE(kind=rck)                       :: a_mat(matrixRows,matrixCols)
  real(kind=rk)                                 :: d_vec(na)
  real(kind=rk)                                 :: e_vec(na)
#endif /* TRIDIAG_GPU_BUILD */


 
  integer(kind=ik)                              :: max_stored_uv = 32 ! TODO_23_11 - make it tunable instead of hard-coded
  logical,          parameter                   :: mat_vec_as_one_block = .true.

  ! id in processor row and column and total numbers of processor rows and columns
  integer(kind=ik)                              :: my_prow, my_pcol, np_rows, np_cols
  integer(kind=MPI_KIND)                        :: my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI
  integer(kind=MPI_KIND)                        :: mpierr
  integer(kind=ik)                              :: totalblocks, max_loc_block_rows, max_loc_block_cols, max_local_rows, &
                                                   max_local_cols
  ! updated after each istep (in the main cycle) to contain number of
  ! local columns and rows of the remaining part of the matrix
  integer(kind=ik)                              :: l_cols, l_rows
  integer(kind=ik)                              :: n_stored_vecs
  logical                                       :: isOurProcessRow, isOurProcessCol, isOurProcessCol_prev


  integer(kind=C_intptr_T)                      :: v_row_dev, v_col_dev, u_row_dev, u_col_dev, vu_stored_rows_dev, &
                                                   uv_stored_cols_dev
  logical                                       :: successGPU

  integer(kind=ik)                              :: istep, i, j, l_col_beg, l_col_end, l_row_beg, l_row_end
  integer(kind=ik)                              :: tile_size, l_rows_per_tile, l_cols_per_tile
  integer(kind=c_intptr_t)                      :: offset_dev

  integer(kind=ik), intent(in)                  :: max_threads_in
  integer(kind=ik)                              :: max_threads
#ifdef WITH_OPENMP_TRADITIONAL
  integer(kind=ik)                              :: my_thread, n_threads, n_iter
#endif

  real(kind=rk)                                 :: vnorm2
  MATH_DATATYPE(kind=rck)                       :: vav, x, aux1(2), vrl, xf, conjg_tau, dot_prod
  MATH_DATATYPE(kind=rck), allocatable          :: aux(:)

  integer(kind=c_intptr_t)                      :: aux_dev, aux1_dev, aux_complex_dev, vav_dev, vav_host_or_dev, dot_prod_dev, & 
                                                   xf_dev, a_updated_element_dev, xf_host_or_dev, tau_istep_host_or_dev
  integer(kind=c_intptr_t)                      :: vnorm2_dev, vrl_dev ! alias pointers for aux1_dev(1), aux1_dev(2)

#if COMPLEXCASE == 1
  complex(kind=rck)                             :: aux3(1)
#endif

  integer(kind=c_intptr_t)                      :: num
  MATH_DATATYPE(kind=rck), pointer              :: v_row(:), & ! used to store calculated Householder Vector
                                                   v_col(:)   ! the same Vector, but transposed 
  MATH_DATATYPE(kind=rck), pointer              :: u_row_debug(:), & ! used to store calculated Householder Vector
                                                   u_col_debug(:)   ! the same Vector, but transposed 
  MATH_DATATYPE(kind=rck), pointer              :: u_col(:), u_row(:)

  ! the following two matrices store pairs of vectors v and u calculated in each step
  ! at most max_stored_uv Vector pairs are stored, than the matrix A_i is explicitli updated
  ! u and v are stored both in row and Vector forms
  ! pattern: v1,u1,v2,u2,v3,u3,....
  ! todo: It is little bit confusing, I think, that variables _row actually store columns and vice versa
  !MATH_DATATYPE(kind=rck), pointer             :: vu_stored_rows(:,:)
  MATH_DATATYPE(kind=rck), allocatable         :: vu_stored_rows(:,:)
  ! pattern: u1,v1,u2,v2,u3,v3,....
  MATH_DATATYPE(kind=rck), allocatable         :: uv_stored_cols(:,:)

#ifdef WITH_OPENMP_TRADITIONAL
  MATH_DATATYPE(kind=rck), allocatable         :: ur_p(:,:), uc_p(:,:)
#endif

  type(c_ptr)                                   :: v_row_host, v_col_host
  type(c_ptr)                                   :: u_row_host, u_col_host
  !type(c_ptr)                                   :: vu_stored_rows_host, uv_stored_cols_host
  real(kind=rk), allocatable                    :: tmp_real(:)
  integer(kind=ik)                              :: min_tile_size, error
  integer(kind=ik)                              :: istat
  character(200)                                :: errorMessage
  character(20)                                 :: gpuString
  integer(kind=ik)                              :: nblockEnd
  integer(kind=c_intptr_t), parameter           :: size_of_datatype = size_of_&
                                                                      &PRECISION&
                                                                      &_&
                                                                      &MATH_DATATYPE
  integer(kind=c_intptr_t), parameter           :: size_of_datatype_real = size_of_&
                                                                      &PRECISION&
                                                                      &_real
  integer(kind=MPI_KIND)                        :: bcast_request1, bcast_request2, bcast_request3
  integer(kind=MPI_KIND)                        :: allreduce_request1, allreduce_request2, allreduce_request3
  integer(kind=MPI_KIND)                        :: allreduce_request4, allreduce_request5, allreduce_request6, &
                                                   allreduce_request7
  logical                                       :: useNonBlockingCollectivesCols
  logical                                       :: useNonBlockingCollectivesRows
  integer(kind=c_int)                           :: non_blocking_collectives_rows, non_blocking_collectives_cols
  logical                                       :: success

  integer(kind=c_intptr_t)                      :: gpuHandle, my_stream
#if defined(USE_CCL_TRIDIAG)
  integer(kind=c_intptr_t)                      :: ccl_comm_rows, ccl_comm_cols
  integer(kind=ik)                              :: nvs, nvr, nvc, lcm_s_t, nblks_tot, nblks_comm, nblks_skip
  logical                                       :: isSquareGridGPU = .false.
  integer(kind=c_intptr_t)                      :: aux_transpose_dev
  integer(kind=c_int)                           :: cclDataType
  integer(kind=ik)                              :: k_datatype
#endif
  logical                                       :: useGPU
  integer(kind=c_int)                           :: pointerMode


  integer(kind=ik)                              :: string_length, sm_count

  useGPU = .false.
#ifdef TRIDIAG_GPU_BUILD
  useGPU = .true.
#endif

#if defined(USE_CCL_TRIDIAG)
  if (useGPU) then
    useCCL = .true.
  
    ccl_comm_rows = obj%gpu_setup%ccl_comm_rows
    ccl_comm_cols = obj%gpu_setup%ccl_comm_cols
  
#if   REALCASE == 1 && DOUBLE_PRECISION == 1
    cclDataType = cclDouble
    k_datatype = 1
#elif REALCASE == 1 && SINGLE_PRECISION == 1
    cclDataType = cclFloat
    k_datatype = 1
#elif COMPLEXCASE == 1 && DOUBLE_PRECISION == 1
    cclDataType = cclDouble
    k_datatype = 2
#elif COMPLEXCASE == 1 && SINGLE_PRECISION == 1
    cclDataType = cclFloat
    k_datatype = 2
#endif
 
  endif 
#endif /* defined(USE_CCL_TRIDIAG) */

  allocate(aux(2*max_stored_uv), stat=istat, errmsg=errorMessage)

  success = .true.

  if(useGPU) then
    gpuString = "_gpu"
  else
    gpuString = ""
  endif

  if (useGPU) then
    max_threads=1
  else
    max_threads=max_threads_in
  endif

  call obj%timer%start("tridiag_&
       &MATH_DATATYPE&
       &" // &
       PRECISION_SUFFIX // &
       gpuString )

  call obj%get("nbc_row_elpa1_full_to_tridi", non_blocking_collectives_rows, error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "Problem setting option for non blocking collectives for rows in elpa1_tridiag. Aborting..."
    success = .false.
    call obj%timer%stop("tridiag_&
    &MATH_DATATYPE&
    &" // &
    PRECISION_SUFFIX // &
    gpuString )
    return
  endif

  call obj%get("nbc_col_elpa1_full_to_tridi", non_blocking_collectives_cols, error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "Problem setting option for non blocking collectives for cols in elpa1_tridiag. Aborting..."
    success = .false.
    call obj%timer%stop("tridiag_&
    &MATH_DATATYPE&
    &" // &
    PRECISION_SUFFIX // &
    gpuString )
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


  !mpi_comm_all    = obj%mpi_setup%mpi_comm_parent
  !mpi_comm_cols   = obj%mpi_setup%mpi_comm_cols
  !mpi_comm_rows   = obj%mpi_setup%mpi_comm_rows

  my_prow = obj%mpi_setup%myRank_comm_rows
  my_pcol = obj%mpi_setup%myRank_comm_cols

  np_rows = obj%mpi_setup%nRanks_comm_rows
  np_cols = obj%mpi_setup%nRanks_comm_cols


  !if (wantDebug) call obj%timer%start("mpi_communication")
  !call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND), my_prowMPI, mpierr)
  !call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)
  !call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND), my_pcolMPI, mpierr)
  !call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND), np_colsMPI, mpierr)


  !my_prow = int(my_prowMPI, kind=c_int)
  !np_rows = int(np_rowsMPI, kind=c_int)
  !my_pcol = int(my_pcolMPI, kind=c_int)
  !np_cols = int(np_colsMPI, kind=c_int)
  !if (wantDebug) call obj%timer%stop("mpi_communication")

  ! Matrix is split into tiles; work is done only for tiles on the diagonal or above
  ! seems that tile is a square submatrix, consisting by several blocks
  ! it is a smallest possible square submatrix, where blocks being distributed among
  ! processors are "aligned" in both rows and columns
  !  -----------------
  ! | 1 4 | 1 4 | 1 4 | ...
  ! | 2 5 | 2 5 | 2 5 | ...
  ! | 3 6 | 3 6 | 3 6 | ...
  !  ----------------- ...
  ! | 1 4 | 1 4 | 1 4 | ...
  ! | 2 5 | 2 5 | 2 5 | ...
  ! | 3 6 | 3 6 | 3 6 | ...
  !  ----------------- .
  !   : :   : :   : :    .
  !   : :   : :   : :      .
  !
  ! this is a tile, where each number represents block, assigned to a processor with the shown number
  ! size of this small block is nblk
  ! Image is for situation with 6 processors, 3 processor rows and 2 columns
  ! tile_size is thus nblk * 6
  !
  tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size

  ! make tile_size a smallest possible multiple of previously defined tile size, such that it is
  ! larger or equal to min_tile_size
  ! min_tile_size has been originally hardcoded as 128 * max(np_rows, np_cols), so it is now the implicit value
  ! it can, however, be set by the user
  call obj%get("min_tile_size", min_tile_size ,error)
  if (error .ne. ELPA_OK) then
    write(error_unit,*) "Problem setting option for min_tile_size. Aborting..."
    success = .false.
    call obj%timer%stop("tridiag_&
    &MATH_DATATYPE&
    &" // &
    PRECISION_SUFFIX // &
    gpuString )
    return
  endif
  if(min_tile_size == 0) then
    ! not set by the user, use the default value
    min_tile_size = 128*max(np_rows, np_cols)
  endif
  tile_size = ((min_tile_size-1)/tile_size+1)*tile_size

  nblockEnd = 3

  l_rows_per_tile = tile_size/np_rows ! local rows of a tile
  l_cols_per_tile = tile_size/np_cols ! local cols of a tile

  totalblocks = (na-1)/nblk + 1
  max_loc_block_rows = (totalblocks-1)/np_rows + 1
  max_loc_block_cols = (totalblocks-1)/np_cols + 1

  ! localy owned submatrix has size at most max_local_rows x max_local_cols at each processor
  max_local_rows = max_loc_block_rows*nblk
  max_local_cols = max_loc_block_cols*nblk

  ! allocate memmory for vectors
  ! todo: It is little bit confusing, I think, that variables _row actually store columns and vice versa
  ! todo: if something has length max_local_rows, it is actually a column, no?
  ! todo: probably one should read it as v_row = Vector v distributed among rows

  allocate(uv_stored_cols(max_local_cols,2*max_stored_uv), stat=istat, errmsg=errorMessage)
  call check_alloc("tridiag_&
       &MATH_DATATYPE ", "uv_stored_cols", istat, errorMessage)

  allocate(vu_stored_rows(max_local_rows,2*max_stored_uv), stat=istat, errmsg=errorMessage)
  call check_alloc("tridiag_&
       &MATH_DATATYPE ", "vu_stored_rows", istat, errorMessage)

  if (useGPU) then
#ifdef WITH_GPU_STREAMS
#if COMPLEXCASE == 1
    num = 1 * size_of_datatype
    successGPU = gpu_host_register(int(loc(aux3),kind=c_intptr_t), num,&
                  gpuHostRegisterDefault)
    check_host_register_gpu("tridiag: aux3", successGPU)
#endif
#endif

    ! allocate v_row 1 element longer to allow store and broadcast tau together with it
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
      num = (max_local_rows+1) * size_of_datatype
      successGPU = gpu_malloc_host(v_row_host, num)
      check_host_alloc_gpu("tridiag: v_row_host", successGPU)
      call c_f_pointer(v_row_host,v_row,(/(max_local_rows+1)/))
    else
      allocate(v_row(max_local_rows+1))
    endif

    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
      num = (max_local_cols) * size_of_datatype
      successGPU = gpu_malloc_host(v_col_host,num)
      check_host_alloc_gpu("tridiag: v_col_host", successGPU)
      call c_f_pointer(v_col_host,v_col,(/(max_local_cols)/))
    else
      allocate(v_col(max_local_cols))
    endif

    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
      num = (max_local_cols) * size_of_datatype
      successGPU = gpu_malloc_host(u_col_host,num)
      check_host_alloc_gpu("tridiag: u_col_host", successGPU)
      call c_f_pointer(u_col_host,u_col,(/(max_local_cols)/))
    else
      allocate(u_col(max_local_cols))
    endif

    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
      num = (max_local_rows) * size_of_datatype
      successGPU = gpu_malloc_host(u_row_host,num)
      check_host_alloc_gpu("tridiag: u_row_host", successGPU)
      call c_f_pointer(u_row_host,u_row,(/(max_local_rows)/))
    else
      allocate(u_row(max_local_rows))
    endif

    
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
      num = (max_local_rows * 2*max_stored_uv) * size_of_datatype
      successGPU = gpu_host_register(int(loc(vu_stored_rows),kind=c_intptr_t), num, gpuHostRegisterDefault)
      check_host_register_gpu("tridiag: vu_stored_rows", successGPU)

      num = (max_local_cols * 2*max_stored_uv) * size_of_datatype
      successGPU = gpu_host_register(int(loc(uv_stored_cols),kind=c_intptr_t), num, gpuHostRegisterDefault)
      check_host_register_gpu("tridiag: uv_stored_cols", successGPU)

      num = (1 * 2*max_stored_uv) * size_of_datatype
      successGPU = gpu_host_register(int(loc(aux),kind=c_intptr_t), num, gpuHostRegisterDefault)
      check_host_register_gpu("tridiag: aux", successGPU)

      num = na * size_of_datatype_real
      successGPU = gpu_host_register(int(loc(d_vec),kind=c_intptr_t), num, gpuHostRegisterDefault)
      check_host_register_gpu("tridiag: d_vec", successGPU)

      num = na * size_of_datatype_real
      successGPU = gpu_host_register(int(loc(e_vec),kind=c_intptr_t), num, gpuHostRegisterDefault)
      check_host_register_gpu("tridiag: e_vec", successGPU)

      num = na * size_of_datatype
      successGPU = gpu_host_register(int(loc(tau),kind=c_intptr_t), num, gpuHostRegisterDefault)
      check_host_register_gpu("tridiag: tau", successGPU)

      num = 2 * size_of_datatype
      successGPU = gpu_host_register(int(loc(aux1),kind=c_intptr_t), num, gpuHostRegisterDefault)
      check_host_register_gpu("tridiag: aux1", successGPU)

      num = 1 * size_of_datatype
      successGPU = gpu_host_register(int(loc(vav),kind=c_intptr_t), num, gpuHostRegisterDefault)
      check_host_register_gpu("tridiag: vav", successGPU)

      num = 1 * size_of_datatype
      successGPU = gpu_host_register(int(loc(dot_prod),kind=c_intptr_t), num, gpuHostRegisterDefault)
      check_host_register_gpu("tridiag: dot_prod", successGPU)

      num = 1 * size_of_datatype
      successGPU = gpu_host_register(int(loc(xf),kind=c_intptr_t), num, gpuHostRegisterDefault)
      check_host_register_gpu("tridiag: xf", successGPU)
    endif ! gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU
  else ! useGPU

    allocate(v_row(max_local_rows+1), stat=istat, errmsg=errorMessage)
    call check_alloc("tridiag_&
    &MATH_DATATYPE ", "v_row", istat, errorMessage)

    allocate(v_col(max_local_cols), stat=istat, errmsg=errorMessage)
    call check_alloc("tridiag_&
     &MATH_DATATYPE ", "v_col", istat, errorMessage)

    allocate(u_col(max_local_cols), stat=istat, errmsg=errorMessage)
    call check_alloc("tridiag_&
    &MATH_DATATYPE ", "u_col", istat, errorMessage)

    allocate(u_row(max_local_rows), stat=istat, errmsg=errorMessage)
    call check_alloc("tridiag_&
    &MATH_DATATYPE ", "u_row", istat, errorMessage)
      
  endif ! useGPU

#ifdef WITH_OPENMP_TRADITIONAL
  allocate(ur_p(max_local_rows,0:max_threads-1), stat=istat, errmsg=errorMessage)
  call check_alloc("tridiag_&
  &MATH_DATATYPE ", "ur_p", istat, errorMessage)

  allocate(uc_p(max_local_cols,0:max_threads-1), stat=istat, errmsg=errorMessage)
  call check_alloc("tridiag_&
  &MATH_DATATYPE ", "uc_p", istat, errorMessage)
#endif /* WITH_OPENMP_TRADITIONAL */

  v_row = 0
  u_row = 0
  v_col = 0
  u_col = 0

  if (useGPU) then
    successGPU = gpu_malloc(v_row_dev, (max_local_rows+1) * size_of_datatype)
    check_alloc_gpu("tridiag: v_row_dev", successGPU)

    successGPU = gpu_malloc(u_row_dev, max_local_rows * size_of_datatype)

    check_alloc_gpu("tridiag: u_row_dev", successGPU)

    successGPU = gpu_malloc(v_col_dev, max_local_cols * size_of_datatype)
    check_alloc_gpu("tridiag: v_col_dev", successGPU)

    successGPU = gpu_malloc(u_col_dev, max_local_cols * size_of_datatype)
    check_alloc_gpu("tridiag: u_col_dev", successGPU)

    successGPU = gpu_malloc(vu_stored_rows_dev, max_local_rows * 2 * max_stored_uv * size_of_datatype)
    check_alloc_gpu("tridiag: vu_stored_rows_dev", successGPU)

    successGPU = gpu_malloc(uv_stored_cols_dev, max_local_cols * 2 * max_stored_uv * size_of_datatype)
    check_alloc_gpu("tridiag: uv_stored_cols_dev", successGPU)

    !successGPU = gpu_malloc(d_vec_dev, na * size_of_datatype_real)
    !check_alloc_gpu("tridiag: d_vec_dev", successGPU)

    !successGPU = gpu_malloc(e_vec_dev, na * size_of_datatype_real)
    !check_alloc_gpu("tridiag: e_vec_dev", successGPU)

    !successGPU = gpu_malloc(tau_dev, na * size_of_datatype)
    !check_alloc_gpu("tridiag: tau_dev", successGPU)

    successGPU = gpu_malloc(aux_dev, 2*max_stored_uv * size_of_datatype)
    check_alloc_gpu("tridiag: aux_dev", successGPU)

    successGPU = gpu_malloc(aux1_dev, 2 * size_of_datatype)
    check_alloc_gpu("tridiag: aux1_dev", successGPU)

    successGPU = gpu_malloc(aux_complex_dev, 2 *max_stored_uv* size_of_datatype)
    check_alloc_gpu("tridiag: aux_complex_dev", successGPU)

    successGPU = gpu_malloc(vav_dev, 1 * size_of_datatype)
    check_alloc_gpu("tridiag: vav_dev", successGPU)

    successGPU = gpu_malloc(dot_prod_dev, 1 * size_of_datatype)
    check_alloc_gpu("tridiag: dot_prod_dev", successGPU)

    successGPU = gpu_malloc(xf_dev, 1 * size_of_datatype)
    check_alloc_gpu("tridiag: xf_dev", successGPU)

#ifdef USE_CCL_TRIDIAG
    ! for gpu_transpose_vectors on non-square grids
    if (np_rows==np_cols .and. (.not. isSkewsymmetric)) then
      ! isSquareGridGPU = .true. ! TODO_23_11 - switched off for now, add test for arbitrary grid mapping and switch back on
    endif

    if (.not. isSquareGridGPU) then
      lcm_s_t   = least_common_multiple(np_rows,np_cols)
      nvs = 1 ! global index where to start in vmat_s/vmat_t
      nvr  = na-1 ! global length of v_col_dev/v_row_dev(without last tau-element), max(istep-1)
      nvc = 1 ! number of columns in v_col_dev/v_row_dev
      nblks_tot = (nvr+nblk-1)/nblk ! number of blocks corresponding to nvr
      ! Get the number of blocks to be skipped at the beginning
      ! This must be a multiple of lcm_s_t (else it is getting complicated),
      ! thus some elements before nvs will be accessed/set.
      nblks_skip = ((nvs-1)/(nblk*lcm_s_t))*lcm_s_t
    
      successGPU = gpu_malloc(aux_transpose_dev, ((nblks_tot-nblks_skip+lcm_s_t-1)/lcm_s_t) * nblk * nvc * size_of_datatype)
      check_alloc_gpu("tridiag: aux_transpose_dev", successGPU)
    endif ! .not. isSquareGridGPU
#endif /* USE_CCL_TRIDIAG */
  endif !useGPU

  d_vec(:) = 0
  e_vec(:) = 0
  tau(:) = 0

  if (useGPU) then
    successGPU = gpu_memset(d_vec_dev, 0, na * size_of_datatype_real)
    check_memcpy_gpu("tridiag: d_vec_dev", successGPU)

    successGPU = gpu_memset(e_vec_dev, 0, na * size_of_datatype_real)
    check_memcpy_gpu("tridiag: e_vec_dev", successGPU)

    successGPU = gpu_memset(tau_dev, 0, na * size_of_datatype)
    check_memcpy_gpu("tridiag: tau_dev", successGPU)
  endif

  n_stored_vecs = 0

  l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a_mat
  l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local cols of a_mat

  if (my_prow == prow(na, nblk, np_rows) .and. my_pcol == pcol(na, nblk, np_cols)) then
    if (useGPU) then
      my_stream = obj%gpu_setup%my_stream
      num = 1 * size_of_datatype_real
      offset_dev = (l_rows-1 + (l_cols-1)*matrixRows) * size_of_datatype
      successGPU = gpu_memcpy(int(loc(d_vec(na)),kind=c_intptr_t), a_dev + offset_dev, &
                                num, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("tridiag: a_dev", successGPU)
    else ! useGPU
#if COMPLEXCASE == 1
      d_vec(na) = real(a_mat(l_rows,l_cols), kind=rk)
#endif
#if REALCASE == 1
      d_vec(na) = a_mat(l_rows,l_cols)
#endif
    endif ! useGPU
  endif

  ! main cycle of tridiagonalization
  ! in each step, 1 Householder Vector is calculated
  do istep = na, nblockEnd ,-1

#ifdef WITH_NVTX
    call nvtxRangePush("tridi cycle")
#endif

    ! Calculate number of local rows and columns of the still remaining matrix
    ! on the local processor
    l_rows = local_index(istep-1, my_prow, np_rows, nblk, -1)
    l_cols = local_index(istep-1, my_pcol, np_cols, nblk, -1)

    ! Calculate Vector for Householder transformation on all procs
    ! owning column istep

    if (useGPU) then 
#ifdef WITH_NVTX
      call nvtxRangePush("kernel: gpu_copy_and_set_zeros a_dev(:,l_cols+1)->v_row_dev, aux1_dev=0,vav_dev=0")
#endif
      ! copy l_cols + 1 column of a_dev to v_row_dev
      isOurProcessRow      = (my_prow == prow(istep-1, nblk, np_rows))
      isOurProcessCol      = (my_pcol == pcol(istep-1, nblk, np_cols))
      isOurProcessCol_prev = (my_pcol == pcol(istep  , nblk, np_cols)) ! isOurProcessCol from the previous step
      my_stream = obj%gpu_setup%my_stream
      call gpu_copy_and_set_zeros_PRECISION(v_row_dev, a_dev, l_rows, l_cols, matrixRows, istep, &
                                            aux1_dev, vav_dev, d_vec_dev, &
                                            isOurProcessRow, isOurProcessCol, isOurProcessCol_prev, &
                                            isSkewsymmetric, useCCL, wantDebug, my_stream)
      if (gpu_vendor() /= SYCL_GPU) successGPU = gpu_DeviceSynchronize()
#ifdef WITH_NVTX
        call nvtxRangePop()
#endif
    endif ! useGPU

    if (my_pcol == pcol(istep, nblk, np_cols)) then

      ! Get Vector to be transformed; distribute last element and norm of
      ! remaining elements to all procs in current column

!             ! copy l_cols + 1 column of A to v_row
      ! if (useGPU) then

! #ifdef WITH_NVTX
!         call nvtxRangePush("memcpy new D-D a_dev(:,l_cols+1)->v_row_dev")
! #endif
!         ! TODO_23_11:  create a dev-dev copy kernel or merge it to another kernel
!         offset_dev = l_cols * matrixRows * size_of_datatype
!         successGPU = gpu_memcpy(v_row_dev, a_dev + offset_dev, (l_rows)* size_of_datatype, gpuMemcpyDeviceToDevice)
!         check_memcpy_gpu("tridiag a_dev 1", successGPU)

! #ifdef WITH_NVTX
!         call nvtxRangePop()
! #endif

!       else ! useGPU
!         v_row(1:l_rows) = a_mat(1:l_rows,l_cols+1)
!       endif ! useGPU

      ! copy l_cols + 1 column of A to v_row
      if (.not. useGPU) then
        v_row(1:l_rows) = a_mat(1:l_rows,l_cols+1)
      endif ! useGPU

      if (n_stored_vecs > 0 .and. l_rows > 0) then
        if (useGPU) then
          if (wantDebug) call obj%timer%start("gpublas gemv skinny with copying")
#ifdef WITH_NVTX
          call nvtxRangePush("gpublas gemv skinny  v_row_dev+=vu_stored_rows_dev*uv_stored_cols_dev")
#endif
          ! v_row_dev = vu_stored_rows_dev * uv_stored_cols_dev(l_cols+1,1:2*n_stored_vecs) + v_row_dev
          gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
          call gpublas_PRECISION_GEMV('N', l_rows, 2*n_stored_vecs,  &
                                    ONE, vu_stored_rows_dev, max_local_rows,  &
#if REALCASE == 1
                                    uv_stored_cols_dev+(l_cols+1-1 +max_local_cols*(1-1))*size_of_datatype , max_local_cols, &
#endif
#if COMPLEXCASE == 1
                                    aux_complex_dev, 1,   &
#endif
                                    ONE, v_row_dev, 1, gpuHandle)

          if (wantDebug .and. gpu_vendor() /= SYCL_GPU) successGPU = gpu_DeviceSynchronize()
#ifdef WITH_NVTX
          call nvtxRangePop()
#endif
          if (wantDebug) call obj%timer%stop("gpublas gemv skinny with copying")

        else ! useGPU
#if COMPLEXCASE == 1
          aux(1:2*n_stored_vecs) = conjg(uv_stored_cols(l_cols+1,1:2*n_stored_vecs))
#endif

          if (wantDebug) call obj%timer%start("blas")
          ! v_row = vu_stored_rows * uv_stored_cols(l_cols+1,1:2*n_stored_vecs) + v_row
          call PRECISION_GEMV('N',   &
                            int(l_rows,kind=BLAS_KIND), int(2*n_stored_vecs,kind=BLAS_KIND), &
                            ONE, vu_stored_rows, int(ubound(vu_stored_rows,dim=1),kind=BLAS_KIND), &
#if REALCASE == 1
                            uv_stored_cols(l_cols+1,1), &
                            int(ubound(uv_stored_cols,dim=1),kind=BLAS_KIND), &
#endif
#if COMPLEXCASE == 1
                            aux, 1_BLAS_KIND,  &
#endif
                            ONE, v_row, 1_BLAS_KIND)
          if (wantDebug) call obj%timer%stop("blas")
        endif ! useGPU
      endif ! (n_stored_vecs > 0 .and. l_rows > 0)

      if (useGPU) then
        isOurProcessRow = (my_prow == prow(istep-1, nblk, np_rows))

        my_stream = obj%gpu_setup%my_stream
#ifdef WITH_NVTX
        call nvtxRangePush("kernel: gpu_dot_product_and_assign v_row_dev*v_row_dev,aux1_dev")
#endif
        call gpu_dot_product_and_assign_PRECISION(v_row_dev, l_rows, isOurProcessRow, aux1_dev, wantDebug, my_stream)
#ifdef WITH_NVTX
        call nvtxRangePop()
#endif
        if (.not. useCCL) then

#ifdef WITH_NVTX
          call nvtxRangePush("memcpy new D-H aux1_dev->aux1")
#endif

#ifdef WITH_GPU_STREAMS
          call gpu_memcpy_async_and_stream_synchronize("tridiag aux1_dev->aux1", aux1_dev,  0_c_intptr_t, &
                                                      aux1(1:2), 1, 2*size_of_datatype, &
                                                      gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
          successGPU = gpu_memcpy(int(loc(aux1),kind=c_intptr_t), aux1_dev, 2*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("tridiag: aux1_dev -> aux1", successGPU)
#endif


#ifdef WITH_NVTX
          call nvtxRangePop()
#endif
        endif ! .not. useCCL 
      endif ! useGPU  


      if (.not. useGPU) then
#ifdef WITH_NVTX
        call nvtxRangePush("cpu_dot v_row*v_row, aux1(2)=v_row")
#endif
        if (my_prow == prow(istep-1, nblk, np_rows)) then
          aux1(1) = dot_product(v_row(1:l_rows-1),v_row(1:l_rows-1)) ! = "q"
          aux1(2) = v_row(l_rows) ! = "a_11" (or rather a_nn)
        else
          aux1(1) = dot_product(v_row(1:l_rows),v_row(1:l_rows))
          aux1(2) = 0.
        endif
#ifdef WITH_NVTX
        call nvtxRangePop()
#endif
      endif ! .not. useGPU 

#ifdef WITH_MPI
      if (useCCL) then
#if defined(USE_CCL_TRIDIAG)
        successGPU = ccl_Allreduce(aux1_dev, aux1_dev, int(k_datatype*2, kind=c_size_t), &
                                    cclDataType, cclSum, ccl_comm_rows, my_stream)
        if (.not. successGPU) then
          print *,"Error in ccl_Allreduce"
          stop 1
        endif

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("ccl_Allreduce aux1_dev", successGPU)
#endif /* USE_CCL_TRIDIAG */
      else ! useCCL

        if (useNonBlockingCollectivesRows) then
          if (wantDebug) call obj%timer%start("mpi_communication_non_blocking")
          call mpi_iallreduce(MPI_IN_PLACE, aux1, 2_MPI_KIND, MPI_MATH_DATATYPE_PRECISION, &
                            MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), allreduce_request1, mpierr)
          call mpi_wait(allreduce_request1, MPI_STATUS_IGNORE, mpierr)
          if (wantDebug) call obj%timer%stop("mpi_communication_non_blocking")
        else ! useNonBlockingCollectivesRows
          if (wantDebug) call obj%timer%start("mpi_communication")
          call mpi_allreduce(MPI_IN_PLACE, aux1, 2_MPI_KIND, MPI_MATH_DATATYPE_PRECISION, &
                            MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
          if (wantDebug) call obj%timer%stop("mpi_communication")
        endif ! useNonBlockingCollectivesRows
      endif ! useCCL
#endif /* WITH_MPI */

      if (useCCL) then
#if defined(USE_CCL_TRIDIAG)
        vnorm2_dev = aux1_dev
        vrl_dev = aux1_dev + 1*size_of_datatype
#endif
      else ! useCCL
#if REALCASE == 1
        vnorm2 = aux1(1)
#endif
#if COMPLEXCASE == 1
        vnorm2 = real(aux1(1),kind=rk)
#endif
        vrl    = aux1(2)
      endif ! useCCL


      ! Householder transformation
      if (useCCL) then
#if defined(USE_CCL_TRIDIAG)
#ifdef WITH_NVTX
        call nvtxRangePush("gpu_hh_transform")
#endif
        call gpu_hh_transform_PRECISION(obj, vrl_dev, vnorm2_dev, xf_dev, tau_dev+(istep-1)*size_of_datatype, & 
                                        wantDebug, my_stream)
#ifdef WITH_NVTX
        call nvtxRangePop()
#endif
#endif /* defined(USE_CCL_TRIDIAG) */
      else ! useCCL
#ifdef WITH_NVTX
        call nvtxRangePush("hh_transform")
#endif
#if REALCASE == 1
        call hh_transform_real_&
#endif
#if COMPLEXCASE == 1
        call hh_transform_complex_&
#endif
                &PRECISION &
                (obj, vrl, vnorm2, xf, tau(istep), wantDebug)
#ifdef WITH_NVTX
        call nvtxRangePop()
#endif
      endif ! useCCL
      
      ! vrl is newly computed off-diagonal element of the final tridiagonal matrix
      if (my_prow == prow(istep-1, nblk, np_rows)) then
        if (.not. useCCL) then
#if REALCASE == 1
          e_vec(istep-1) = vrl
#endif
#if COMPLEXCASE == 1
          e_vec(istep-1) = real(vrl,kind=rk)
#endif
        endif ! .not. useCCL
      endif

      if (.not. useGPU) then
#ifdef WITH_NVTX
        call nvtxRangePush("scale v_row *= xf")
#endif
        ! Scale v_row and store Householder Vector for back transformation
        v_row(1:l_rows) = v_row(1:l_rows) * xf
#ifdef WITH_NVTX
        call nvtxRangePop()
#endif

        if (my_prow == prow(istep-1, nblk, np_rows)) then
          v_row(l_rows) = 1.
        endif

        ! store Householder Vector for back transformation
#ifdef WITH_NVTX
        call nvtxRangePush("cpu copy: v_row->a_mat")
#endif
        ! update a_mat
        a_mat(1:l_rows,l_cols+1) = v_row(1:l_rows)
#ifdef WITH_NVTX
        call nvtxRangePop()
#endif
      endif ! .not. useGPU 

      if (.not. useCCL) then
        ! add tau after the end of actuall v_row, to be broadcasted with it
        v_row(l_rows+1) = tau(istep)
      endif

      
      if (useGPU) then
        if (useCCL) then
          xf_host_or_dev = xf_dev
        else 
          xf_host_or_dev = int(loc(xf),kind=c_intptr_t)
        endif       
        
#ifdef WITH_NVTX
        call nvtxRangePush("kernel gpu_set_e_vec_scale_set_one_store_v_row")
#endif
        isOurProcessRow = (my_prow == prow(istep-1, nblk, np_rows))
        call gpu_set_e_vec_scale_set_one_store_v_row_PRECISION(e_vec_dev, vrl_dev, a_dev, v_row_dev, tau_dev, xf_host_or_dev, & 
                                                  l_rows, l_cols, matrixRows, istep, isOurProcessRow, useCCL, wantDebug, my_stream)
#ifdef WITH_NVTX
        call nvtxRangePop()
#endif
      endif ! useGPU  


      if (useGPU .and. .not. useCCL) then      
        !v_row_dev -> v_row
#ifdef WITH_NVTX
        call nvtxRangePush("memcpy new D-H v_row_dev->v_row")
#endif

#ifdef WITH_GPU_STREAMS
        call gpu_memcpy_async_and_stream_synchronize ("tridiag v_row_dev -> v_row", v_row_dev, 0_c_intptr_t, &
                                                v_row(1:l_rows), 1, l_rows*size_of_datatype, &
                                                gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
        successGPU = gpu_memcpy(int(loc(v_row),kind=c_intptr_t), v_row_dev, l_rows*size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("tridiag: v_row_dev -> v_row", successGPU)
#endif

#ifdef WITH_NVTX
        call nvtxRangePop()
#endif
      endif ! useGPU .and. .not. useCCL

    endif !(my_pcol == pcol(istep, nblk, np_cols))


#ifdef WITH_MPI
    if (useCCL .and. np_cols>1) then
#ifdef USE_CCL_TRIDIAG
      successGPU = ccl_Bcast(v_row_dev, v_row_dev, int(k_datatype*(l_rows+1), kind=c_size_t), cclDataType, &
                              int(pcol(istep, nblk, np_cols),kind=c_int), ccl_comm_cols, my_stream)
      if (.not. successGPU) then
        print *,"Error in ccl_Bcast"
        stop 1
      endif

      successGPU = gpu_stream_synchronize(my_stream) ! TODO_23_11: do we need ccl_group_start() here and before ?
      check_stream_synchronize_gpu("ccl_Bcast v_row_dev", successGPU)
#endif /* USE_CCL_TRIDIAG */
    else ! useCCL
      if (useNonBlockingCollectivesCols) then
        if (wantDebug) call obj%timer%start("mpi_nbc_communication")
        ! Broadcast the Householder Vector (and tau) along columns
        call mpi_ibcast(v_row, int(l_rows+1,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,    &
                    int(pcol(istep, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), &
                    bcast_request1, mpierr)
        call mpi_wait(bcast_request1, MPI_STATUS_IGNORE, mpierr)
        if (wantDebug) call obj%timer%stop("mpi_nbc_communication")
      else
        if (wantDebug) call obj%timer%start("mpi_communication")
        call mpi_bcast(v_row, int(l_rows+1,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,    &
                    int(pcol(istep, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), &
                    mpierr)
        if (wantDebug) call obj%timer%stop("mpi_communication")
      endif
    endif ! useCCL .and. np_cols>1
#endif /* WITH_MPI */

    !recover tau, which has been broadcasted together with v_row
    if (.not. useCCL) then
      tau(istep) =  v_row(l_rows+1)
    endif

    ! Transpose Householder Vector v_row -> v_col
    if (useCCL) then
#ifdef WITH_NVTX
      call nvtxRangePush("elpa_gpu_ccl_transpose_vectors v_row_dev->v_col_dev")
#endif
#if defined(USE_CCL_TRIDIAG)
      call elpa_gpu_ccl_transpose_vectors_&
          &MATH_DATATYPE&
          &_&
          &PRECISION &
                (obj, v_row_dev, max_local_rows+1, ccl_comm_rows, mpi_comm_rows, v_col_dev, max_local_cols, &
                ccl_comm_cols, mpi_comm_cols, 1, istep-1, 1, nblk, max_threads, .true., my_prow, my_pcol, np_rows, np_cols, &
                aux_transpose_dev, isSkewsymmetric, isSquareGridGPU, wantDebug, my_stream, success)
#endif /* USE_CCL_TRIDIAG */
#ifdef WITH_NVTX
      call nvtxRangePop()
#endif
    else ! useCCL
#ifdef WITH_NVTX
      call nvtxRangePush("elpa_transpose_vectors v_row -> v_col")
#endif
      call elpa_transpose_vectors_&
          &MATH_DATATYPE&
          &_&
          &PRECISION &
                (obj, v_row, ubound(v_row,dim=1), mpi_comm_rows, v_col, ubound(v_col,dim=1), mpi_comm_cols, &
                1, istep-1, 1, nblk, max_threads, .true., success)
      if (wantDebug .and. gpu_vendor() /= SYCL_GPU) successGPU = gpu_DeviceSynchronize()
#ifdef WITH_NVTX
      call nvtxRangePop()
#endif
      if (.not.(success)) then
        write(error_unit,*) "Error in elpa_transpose_vectors. Aborting!"
        return
      endif
    endif ! useCCL

    ! Calculate u = (A + VU**T + UV**T)*v // Dongarra 1987: "y = (A - UV**T - VU**T)*u"

    ! For cache efficiency, we use only the upper half of the matrix tiles for this,
    ! thus the result is partly in u_col(:) and partly in u_row(:)

    if (.not. useGPU) then
#ifdef WITH_NVTX
      call nvtxRangePush("cpu: set u_col,u_row=0")
#endif
      u_col(1:l_cols) = 0
      u_row(1:l_rows) = 0
#ifdef WITH_NVTX
      call nvtxRangePop()
#endif
    endif

    if (useGPU .and. useCCL) then
#ifdef WITH_NVTX
      call nvtxRangePush("gpu-ccl: set u_col_dev=0")
#endif
      successGPU = gpu_memset(u_col_dev, 0, l_cols * size_of_datatype) ! TODO_23_11: omit this, but change gpublas_gemm to u_col_dev=a_dev^T*v_row_dev+0*u_col_dev?
      check_memcpy_gpu("tridiag: u_col_dev", successGPU)
#ifdef WITH_NVTX
      call nvtxRangePop()
#endif
    endif

    if (l_rows>0 .and. l_cols>0) then
      if (useGPU) then
#if defined(WITH_OPENMP_OFFLOAD_GPU_VERSION)
        if (gpu_vendor() == OPENMP_OFFLOAD_GPU) then
          ! debug
          allocate(u_col_debug(l_cols))
          u_col_debug(:) = 0.
          
          successGPU = gpu_memcpy(u_col_dev, int(loc(u_col_debug(1)),kind=c_intptr_t), &
                      l_cols * size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("tridiag: u_col_dev", successGPU)

          deallocate(u_col_debug)
          allocate(u_row_debug(l_rows))
          u_row_debug(:) = 0.

          successGPU = gpu_memcpy(u_row_dev, int(loc(u_row_debug(1)),kind=c_intptr_t), &
                      l_rows * size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("tridiag: u_row_dev", successGPU)

          deallocate(u_row_debug)
        endif ! (gpu_vendor() == OPENMP_OFFLOAD_GPU)
#endif

        if (.not. mat_vec_as_one_block) then
#ifdef WITH_NVTX
          call nvtxRangePush("memcpy H-D v_col->v_col_dev")
#endif

#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          num = l_cols * size_of_datatype
          call gpu_memcpy_async_and_stream_synchronize &
              ("tridiag v_col -> v_col_dev",  v_col_dev, 0_c_intptr_t, &
                                              v_col(1:max_local_cols), &
                                              1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
          successGPU = gpu_memcpy(v_col_dev, int(loc(v_col(1)),kind=c_intptr_t), &
                        l_cols * size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("tridiag: v_col_dev", successGPU)
#endif

#ifdef WITH_NVTX
          call nvtxRangePop()
#endif

        endif ! .not. mat_vec_as_one_block

        if (.not. useCCL) then
#ifdef WITH_NVTX
          call nvtxRangePush("memcpy H-D v_row->v_row_dev")
#endif

#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          num = l_rows * size_of_datatype
          call gpu_memcpy_async_and_stream_synchronize &
              ("tridiag v_row -> v_row_dev", v_row_dev, 0_c_intptr_t, &
                                                    v_row(1:max_local_rows+1), &
                                                    1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
          successGPU = gpu_memcpy(v_row_dev, int(loc(v_row(1)),kind=c_intptr_t), &
                                    l_rows * size_of_datatype, gpuMemcpyHostToDevice)
          check_memcpy_gpu("tridiag: v_row_dev", successGPU)
#endif

#ifdef WITH_NVTX
          call nvtxRangePop()
#endif

        endif ! .not. useCCL
      endif ! useGPU

#ifdef WITH_OPENMP_TRADITIONAL
      call obj%timer%start("OpenMP parallel")
!todo : check whether GPU implementation with large matrix multiply is beneficial
!       for a larger number of threads; could be addressed with autotuning if this
!       is the case
!$omp parallel &
!$omp num_threads(max_threads) &
!$omp default(none) &
!$omp private(my_thread, n_threads, n_iter, i, l_col_beg, l_col_end, j, l_row_beg, l_row_end, my_stream, num) &
!$omp shared(obj, gpuHandle, useGPU, isSkewsymmetric, gpuMemcpyDeviceToHost, successGPU, u_row, u_row_dev, &
!$omp &      v_row, v_row_dev, v_col, v_col_dev, u_col, u_col_dev, a_dev, offset_dev, &
!$omp&       max_local_cols, max_local_rows, wantDebug, l_rows_per_tile, l_cols_per_tile, &
!$omp&       matrixRows, istep, tile_size, l_rows, l_cols, ur_p, uc_p, a_mat, &
!$omp&       matrixCols)
      my_thread = omp_get_thread_num()
          
      n_threads = omp_get_num_threads()

      n_iter = 0

      ! first calculate A*v part of (A + VU**T + UV**T)*v
      uc_p(1:l_cols,my_thread) = 0.
      ur_p(1:l_rows,my_thread) = 0.
#endif /* WITH_OPENMP_TRADITIONAL */

#ifdef WITH_NVTX
      call nvtxRangePush("cpu-only nested loop")
#endif
      if (.not. useGPU) then
        do i=0, (istep-2)/tile_size ! iteration over tiles
          l_col_beg = i*l_cols_per_tile+1
          l_col_end = min(l_cols,(i+1)*l_cols_per_tile)
          if (l_col_end < l_col_beg) cycle
          do j = 0, i
            l_row_beg = j*l_rows_per_tile+1
            l_row_end = min(l_rows,(j+1)*l_rows_per_tile)
            if (l_row_end < l_row_beg) cycle
#ifdef WITH_OPENMP_TRADITIONAL
            if (mod(n_iter,n_threads) == my_thread) then
              if (wantDebug) call obj%timer%start("blas")
              call PRECISION_GEMV(BLAS_TRANS_OR_CONJ, &
                  int(l_row_end-l_row_beg+1,kind=BLAS_KIND), int(l_col_end-l_col_beg+1,kind=BLAS_KIND), &
                  ONE, a_mat(l_row_beg,l_col_beg), int(matrixRows,kind=BLAS_KIND),         &
                  v_row(l_row_beg:max_local_rows+1), 1_BLAS_KIND, ONE, uc_p(l_col_beg,my_thread), 1_BLAS_KIND)

              if (i/=j) then
                if (isSkewsymmetric) then
                  call PRECISION_GEMV('N', int(l_row_end-l_row_beg+1,kind=BLAS_KIND), &
                                      int(l_col_end-l_col_beg+1,kind=BLAS_KIND), &
                                      -ONE, a_mat(l_row_beg,l_col_beg), int(matrixRows,kind=BLAS_KIND), &
                                      v_col(l_col_beg:max_local_cols), 1_BLAS_KIND,  &
                                      ONE, ur_p(l_row_beg,my_thread), 1_BLAS_KIND)

                else
                  call PRECISION_GEMV('N', int(l_row_end-l_row_beg+1,kind=BLAS_KIND), &
                                      int(l_col_end-l_col_beg+1,kind=BLAS_KIND), &
                                      ONE, a_mat(l_row_beg,l_col_beg), int(matrixRows,kind=BLAS_KIND), &
                                      v_col(l_col_beg:max_local_cols), 1_BLAS_KIND,  &
                                      ONE, ur_p(l_row_beg,my_thread), 1_BLAS_KIND)
                endif
              endif
              if (wantDebug) call obj%timer%stop("blas")
            endif
            n_iter = n_iter+1
#else /* WITH_OPENMP_TRADITIONAL */

            ! multiplication by blocks is efficient only for CPU
            ! for GPU we introduced 2 other ways, either by stripes (more simmilar to the original
            ! CPU implementation) or by one large matrix Vector multiply
            if (wantDebug) call obj%timer%start("blas")
            ! u_col = a_mat*v_row + u_col(=0)
            call PRECISION_GEMV(BLAS_TRANS_OR_CONJ,  &
                        int(l_row_end-l_row_beg+1,kind=BLAS_KIND), int(l_col_end-l_col_beg+1,kind=BLAS_KIND), &
                        ONE, a_mat(l_row_beg, l_col_beg), int(matrixRows,kind=BLAS_KIND),         &
                        v_row(l_row_beg:max_local_rows+1), 1_BLAS_KIND,                           &
                        ONE, u_col(l_col_beg:max_local_cols), 1_BLAS_KIND)

            if (i/=j) then
              if (isSkewsymmetric) then
                call PRECISION_GEMV('N',int(l_row_end-l_row_beg+1,kind=BLAS_KIND), int(l_col_end-l_col_beg+1,kind=BLAS_KIND), &
                                    -ONE, a_mat(l_row_beg,l_col_beg), int(matrixRows,kind=BLAS_KIND),               &
                                    v_col(l_col_beg:max_local_cols), 1_BLAS_KIND, ONE, u_row(l_row_beg:max_local_rows), &
                                    1_BLAS_KIND)

              else
                ! u_row = a_mat*v_col + u_row(=0)
                call PRECISION_GEMV('N',int(l_row_end-l_row_beg+1,kind=BLAS_KIND), int(l_col_end-l_col_beg+1,kind=BLAS_KIND),  &
                                    ONE, a_mat(l_row_beg,l_col_beg), int(matrixRows,kind=BLAS_KIND),               &
                                    v_col(l_col_beg:max_local_cols), 1_BLAS_KIND, ONE, u_row(l_row_beg:max_local_rows), &
                                    1_BLAS_KIND)
              endif
            endif
            if (wantDebug) call obj%timer%stop("blas")

#endif /* WITH_OPENMP_TRADITIONAL */
          enddo  ! j=0,i
        enddo  ! i=0,(istep-2)/tile_size
      endif ! .not. useGPU
#ifdef WITH_NVTX
      call nvtxRangePop() ! cpu-only nested loop
#endif

      if (useGPU) then
        if (mat_vec_as_one_block) then
          ! Unlike for CPU, we (for each MPI thread) do just one large mat-vec multiplication
          ! this requires altering of the algorithm when later explicitly updating the matrix
          ! after max_stored_uv is reached : we need to update all tiles, not only those above diagonal
          if (wantDebug) call obj%timer%start("gpublas")
          
          gpuHandle = obj%gpu_setup%gpublasHandleArray(0) ! 0.3-0.5 us

#ifdef WITH_NVTX
          call nvtxRangePush("gpublas_gemv: u_col_dev=a_dev^T*v_row_dev")
#endif
              
          ! u_col_dev = a_dev^T*v_row_dev
          call gpublas_PRECISION_GEMV(BLAS_TRANS_OR_CONJ, l_rows,l_cols,  &
                                    ONE, a_dev, matrixRows,                   &
                                    v_row_dev , 1,                          &
                                    ZERO, u_col_dev, 1, gpuHandle)
              
          if (wantDebug .and. gpu_vendor() /= SYCL_GPU) successGPU = gpu_DeviceSynchronize()
#ifdef WITH_NVTX
          call nvtxRangePop()
#endif
              
       ! todo: try with non transposed!!!
!                 if(i/=j) then
!                   call gpublas_PRECISION_GEMV('N', l_row_end-l_row_beg+1,l_col_end-l_col_beg+1,  &
!                                             ONE, a_dev + offset_dev, matrixRows,                        &
!                                             v_col_dev + (l_col_beg - 1) *                      &
!                                             size_of_datatype, 1,                          &
!                                             ONE, u_row_dev + (l_row_beg - 1) *                 &
!                                             size_of_datatype, 1)
!                 endif
          if (wantDebug) call obj%timer%stop("gpublas")
        else  ! mat_vec_as_one_block
          !perform multiplication by stripes - it is faster than by blocks, since we call cublas with
          !larger matrices. In general, however, this algorithm is very simmilar to the one with CPU
          do i=0,(istep-2)/tile_size
            l_col_beg = i*l_cols_per_tile+1
            l_col_end = min(l_cols,(i+1)*l_cols_per_tile)
            if (l_col_end<l_col_beg) cycle

            l_row_beg = 1
            l_row_end = min(l_rows,(i+1)*l_rows_per_tile)
                  
            offset_dev = ((l_row_beg-1) + (l_col_beg - 1) * matrixRows) * size_of_datatype
  
            gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
            call gpublas_PRECISION_GEMV(BLAS_TRANS_OR_CONJ, &
                          l_row_end-l_row_beg+1, l_col_end-l_col_beg+1, &
                          ONE, a_dev + offset_dev, matrixRows,  &
                          v_row_dev + (l_row_beg - 1) * size_of_datatype, 1,  &
                          ONE, u_col_dev + (l_col_beg - 1) * size_of_datatype, 1, gpuHandle)
          enddo !i=0,(istep-2)/tile_size

          do i=0,(istep-2)/tile_size
            l_col_beg = i*l_cols_per_tile+1
            l_col_end = min(l_cols,(i+1)*l_cols_per_tile)
            if (l_col_end<l_col_beg) cycle

            l_row_beg = 1
            l_row_end = min(l_rows,i*l_rows_per_tile)
              
            offset_dev = ((l_row_beg-1) + (l_col_beg - 1) * matrixRows) * size_of_datatype

            if (isSkewsymmetric) then
                gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
                call gpublas_PRECISION_GEMV('N', l_row_end-l_row_beg+1, l_col_end-l_col_beg+1, &
                            -ONE, a_dev + offset_dev, matrixRows, &
                            v_col_dev + (l_col_beg - 1) * size_of_datatype,1, &
                            ONE, u_row_dev + (l_row_beg - 1) * size_of_datatype, 1, gpuHandle)
            else
                gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
                call gpublas_PRECISION_GEMV('N', l_row_end-l_row_beg+1, l_col_end-l_col_beg+1, &
                            ONE, a_dev + offset_dev, matrixRows, &
                            v_col_dev + (l_col_beg - 1) * size_of_datatype,1, &
                            ONE, u_row_dev + (l_row_beg - 1) * size_of_datatype, 1, gpuHandle)
            endif
          enddo ! i=0,(istep-2)/tile_size
        end if ! mat_vec_as_one_block / per stripes


        if (.not. mat_vec_as_one_block) then
#ifdef WITH_NVTX
          call nvtxRangePush("memcpy D-H u_row_dev->u_row")
#endif

#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          num = l_rows * size_of_datatype
          call gpu_memcpy_async_and_stream_synchronize &
              ("tridiag u_row_dev -> u_row", u_row_dev, 0_c_intptr_t, &
                                                    u_row(1:max_local_rows), &
                                                    1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
          successGPU = gpu_memcpy(int(loc(u_row(1)),kind=c_intptr_t), u_row_dev, l_rows*size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("tridiag: u_row_dev 1", successGPU)
#endif

#ifdef WITH_NVTX
          call nvtxRangePop()
#endif

        endif ! .not. mat_vec_as_one_block
      endif ! useGPU

#ifdef WITH_OPENMP_TRADITIONAL
!$OMP END PARALLEL
      call obj%timer%stop("OpenMP parallel")
      if (.not.(useGPU)) then
        do i=0,max_threads-1
          u_col(1:l_cols) = u_col(1:l_cols) + uc_p(1:l_cols,i)
          u_row(1:l_rows) = u_row(1:l_rows) + ur_p(1:l_rows,i)
        enddo
      endif
#endif /* WITH_OPENMP_TRADITIONAL */

      ! second calculate (VU**T + UV**T)*v part of (A + VU**T + UV**T)*v
      if (n_stored_vecs > 0) then
        if (.not. useGPU) then
          if (wantDebug) call obj%timer%start("blas")

#ifdef WITH_NVTX
          call nvtxRangePush("cpu gemv_x2 aux=vu_stored_rows^T*v_row,u_col+=uv_stored_cols*aux")
#endif
      
          ! aux = vu_stored_rows^T*v_row
          call PRECISION_GEMV(BLAS_TRANS_OR_CONJ,     &
                              int(l_rows,kind=BLAS_KIND), int(2*n_stored_vecs,kind=BLAS_KIND),   &
                              ONE, vu_stored_rows, int(ubound(vu_stored_rows,dim=1),kind=BLAS_KIND),   &
                              v_row,  1_BLAS_KIND, ZERO, aux, 1_BLAS_KIND)
              
          ! u_col = uv_stored_cols*aux + u_col
          call PRECISION_GEMV('N', int(l_cols,kind=BLAS_KIND), int(2*n_stored_vecs,kind=BLAS_KIND),   &
                              ONE, uv_stored_cols, int(ubound(uv_stored_cols,dim=1),kind=BLAS_KIND),   &
                              aux, 1_BLAS_KIND, ONE, u_col,  1_BLAS_KIND)
#ifdef WITH_NVTX
          call nvtxRangePop()
#endif
          if (wantDebug) call obj%timer%stop("blas")
        else ! .not. useGPU
          if (wantDebug) call obj%timer%start("gpublas gemv x2 skinny")

#ifdef WITH_NVTX
          call nvtxRangePush("gpublas gemv_x2 skinny aux_dev=vu_stored_rows_dev^T*v_row_dev,u_col_dev+=uv_stored_cols_dev*aux_dev")
#endif

          ! aux_dev = vu_stored_rows_dev^T*v_row_dev
          gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
          call gpublas_PRECISION_GEMV(BLAS_TRANS_OR_CONJ, l_rows, 2*n_stored_vecs, &
                                      ONE, vu_stored_rows_dev, max_local_rows,   &
                                      v_row_dev,  1, ZERO, aux_dev, 1, gpuHandle)
              
          ! u_col_dev = uv_stored_cols_dev*aux_dev + u_col_dev
          call gpublas_PRECISION_GEMV('N', l_cols, 2*n_stored_vecs, &
                                      ONE, uv_stored_cols_dev, max_local_cols,   &
                                      aux_dev, 1, ONE, u_col_dev, 1, gpuHandle)

          if (wantDebug .and. gpu_vendor() /= SYCL_GPU) successGPU = gpu_DeviceSynchronize()
#ifdef WITH_NVTX
          call nvtxRangePop()
#endif

          if (wantDebug) call obj%timer%stop("gpublas gemv x2 skinny")  
        endif ! .not. useGPU
      endif ! n_stored_vecs > 0

    endif  ! (l_rows>0 .and. l_cols>0)

    if (useGPU .and. l_cols>0 .and. (.not. useCCL)) then
#ifdef WITH_NVTX
      call nvtxRangePush("memcpy D-H u_col_dev->u_col")
#endif

#ifdef WITH_GPU_STREAMS
      call gpu_memcpy_async_and_stream_synchronize ("tridiag u_col_dev -> u_col", u_col_dev, 0_c_intptr_t, &
                                                u_col(1:max_local_cols), 1, l_cols*size_of_datatype, &
                                                gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
      successGPU = gpu_memcpy(int(loc(u_col(1)),kind=c_intptr_t), u_col_dev, l_cols*size_of_datatype, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("tridiag: u_col_dev 1", successGPU)
#endif

#ifdef WITH_NVTX
      call nvtxRangePop()
#endif
    endif ! useGPU


    if (useGPU .and. .not. useCCL) then 
        if (l_rows==0) then
#ifdef WITH_NVTX
        call nvtxRangePush("cpu: set u_col=0")
#endif
        u_col(1:l_cols) = 0
#ifdef WITH_NVTX
        call nvtxRangePop()
#endif
      endif
    endif ! useGPU

    ! Sum up all u_row(:) parts along rows and add them to the u_col(:) parts
    ! on the processors containing the diagonal
    ! This is only necessary if u_row has been calculated, i.e. if the
    ! global tile size is smaller than the global remaining matrix
    
    ! in GPU case, u_row_dev=0 up to now, so elpa_reduce_add_vectors is skipped

    if (tile_size < istep-1 .and. .not. useGPU) then
#ifdef WITH_NVTX
      call nvtxRangePush("elpa_reduce_add_vectors u_row,u_col")
#endif
      call elpa_reduce_add_vectors_&
            &MATH_DATATYPE&
            &_&
            &PRECISION &
            (obj, u_row, ubound(u_row,dim=1), mpi_comm_rows, u_col, ubound(u_col,dim=1), &
            mpi_comm_cols, istep-1, 1, nblk, max_threads)
#ifdef WITH_NVTX
      call nvtxRangePop()
#endif
    endif ! (tile_size < istep-1 .and. .not. useGPU)

    ! Sum up all the u_col(:) parts, transpose u_col -> u_row

    if (l_cols > 0) then
#ifdef WITH_MPI
      if (useCCL) then
#if defined(USE_CCL_TRIDIAG)
#ifdef WITH_NVTX
        call nvtxRangePush("ccl_Allreduce u_col_dev") ! TODO_23_11: can we use Bcast instead of Allreauce and not set u_col_dev=0 above?
#endif
        successGPU = ccl_Allreduce(u_col_dev, u_col_dev, int(k_datatype*l_cols, kind=c_size_t), &
                                    cclDataType, cclSum, ccl_comm_rows, my_stream)
        if (.not. successGPU) then
          print *,"Error in ccl_allreduce"
          stop 1
        endif

        successGPU = gpu_stream_synchronize(my_stream) ! TODO_23_11: do we need it here and before ccl_group_start()?
        check_stream_synchronize_gpu("ccl_Allreduce u_col_dev", successGPU)
#ifdef WITH_NVTX
        call nvtxRangePop()
#endif
#endif /* USE_CCL_TRIDIAG */

      else ! useCCL
          if (useNonBlockingCollectivesRows) then
            if (wantDebug) call obj%timer%start("mpi_nbc_communication")
            call mpi_iallreduce(MPI_IN_PLACE, u_col, int(l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,    &
                              MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), allreduce_request2, mpierr)
            call mpi_wait(allreduce_request2, MPI_STATUS_IGNORE, mpierr)
            if (wantDebug) call obj%timer%stop("mpi_nbc_communication")
          else
            if (wantDebug) call obj%timer%start("mpi_communication")
            call mpi_allreduce(MPI_IN_PLACE, u_col, int(l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,    &
                              MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
            if (wantDebug) call obj%timer%stop("mpi_communication")
          endif

      endif ! useCCL
#endif /* WITH_MPI */
    endif ! (l_cols > 0)
    
    ! Transpose Householder Vector u_col -> u_row
    if (useCCL) then
#if defined(USE_CCL_TRIDIAG)
#ifdef WITH_NVTX
      call nvtxRangePush("elpa_gpu_ccl_transpose_vectors u_col_dev->u_row_dev")
#endif
      call elpa_gpu_ccl_transpose_vectors_&
          &MATH_DATATYPE&
          &_&
          &PRECISION &
                (obj, u_col_dev, max_local_cols, ccl_comm_cols, mpi_comm_cols, u_row_dev, max_local_rows+1, &
                ccl_comm_rows, mpi_comm_rows, 1, istep-1, 1, nblk, max_threads, .false., my_pcol, my_prow, np_cols, np_rows, &
                aux_transpose_dev, isSkewsymmetric, isSquareGridGPU, wantDebug, my_stream, success)
#ifdef WITH_NVTX
      call nvtxRangePop()
#endif
#endif /* USE_CCL_TRIDIAG */
    else ! useCCL
      if (isSkewsymmetric) then
        call elpa_transpose_vectors_ss_&
        &MATH_DATATYPE&
        &_&
        &PRECISION &
        (obj, u_col, ubound(u_col,dim=1), mpi_comm_cols, u_row, ubound(u_row,dim=1), &
          mpi_comm_rows, 1, istep-1, 1, nblk, max_threads, .false., success)
        if (.not.(success)) then
          write(error_unit,*) "Error in elpa_transpose_vectors_ss. Aborting!"
          return
        endif
      else ! isSkewsymmetric
#ifdef WITH_NVTX
        call nvtxRangePush("elpa_transpose_vectors u_col->u_row")
#endif
        call elpa_transpose_vectors_&
        &MATH_DATATYPE&
        &_&
        &PRECISION &
        (obj, u_col, ubound(u_col,dim=1), mpi_comm_cols, u_row, ubound(u_row,dim=1), &
        mpi_comm_rows, 1, istep-1, 1, nblk, max_threads, .false., success)
#ifdef WITH_NVTX
        call nvtxRangePop()
#endif
        if (.not.(success)) then
          write(error_unit,*) "Error in elpa_transpose_vectors. Aborting!"
          return
        endif
      endif ! isSkewsymmetric
    endif ! useCCL


    if (useGPU .and. .not. useCCL) then
#ifdef WITH_NVTX
      call nvtxRangePush("memcpy new-2 H-D v_col->v_col_dev")
#endif

#ifdef WITH_GPU_STREAMS
      call gpu_memcpy_async_and_stream_synchronize ("tridiag v_col->v_col_dev", v_col_dev, 0_c_intptr_t, &
                                                 v_col(1:l_cols), 1, l_cols*size_of_datatype, &
                                                 gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
      successGPU = gpu_memcpy(v_col_dev, int(loc(v_col(1)),kind=c_intptr_t), l_cols*size_of_datatype, gpuMemcpyHostToDevice)
      check_memcpy_gpu("tridiag: v_col_dev", successGPU)
#endif

#ifdef WITH_NVTX
      call nvtxRangePop()
 
      call nvtxRangePush("memcpy new-2 H-D u_col->u_col_dev")
#endif

#ifdef WITH_GPU_STREAMS
      call gpu_memcpy_async_and_stream_synchronize ("tridiag u_col->u_col_dev", u_col_dev, 0_c_intptr_t, &
                                                 u_col(1:l_cols), 1, l_cols*size_of_datatype, &
                                                 gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
      successGPU = gpu_memcpy(u_col_dev, int(loc(u_col(1)),kind=c_intptr_t), l_cols*size_of_datatype, gpuMemcpyHostToDevice)
      check_memcpy_gpu("tridiag: u_col_dev", successGPU)
#endif

#ifdef WITH_NVTX
      call nvtxRangePop()
#endif
    endif ! (useGPU .and. .not. useCCL)

#if defined(USE_CCL_TRIDIAG)
    if (useGPU .and. useCCL) then
#ifdef WITH_NVTX
      call nvtxRangePush("kernel: gpu_dot_product_double vav_dev=v_col_dev*u_col_dev")
#endif
      sm_count = obj%gpu_setup%gpuSMcount
      !sm_count=32
      call gpu_dot_product_PRECISION(l_cols, v_col_dev, 1, u_col_dev, 1, vav_dev, wantDebug, sm_count, my_stream)
      !call gpu_dot_product_PRECISION(l_cols, v_col_dev, 1, u_col_dev, 1, vav_dev, wantDebug, my_stream)
#ifdef WITH_NVTX
      call nvtxRangePop()
#endif
    endif ! useGPU .and. useCCL
#endif /* USE_CCL_TRIDIAG */


    if (useGPU .and. .not. useCCL) then
#ifdef WITH_NVTX
      call nvtxRangePush("memcpy new-2 H-D u_row->u_row_dev")
#endif

#ifdef WITH_GPU_STREAMS
      call gpu_memcpy_async_and_stream_synchronize ("tridiag u_row->u_row_dev", u_row_dev, 0_c_intptr_t, &
                                                  u_row(1:l_rows), 1, l_rows*size_of_datatype, &
                                                  gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
      successGPU = gpu_memcpy(u_row_dev, int(loc(u_row(1)),kind=c_intptr_t), l_rows*size_of_datatype, gpuMemcpyHostToDevice)
      check_memcpy_gpu("tridiag: u_row_dev", successGPU)
#endif

#ifdef WITH_NVTX
      call nvtxRangePop()
      call nvtxRangePush("memcpy new-2 H-D v_row->v_row_dev")
#endif

#ifdef WITH_GPU_STREAMS
      call gpu_memcpy_async_and_stream_synchronize ("tridiag v_row->v_row_dev", v_row_dev, 0_c_intptr_t, &
                                                  v_row(1:l_rows), 1, l_rows*size_of_datatype, &
                                                  gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
      successGPU = gpu_memcpy(v_row_dev, int(loc(v_row(1)),kind=c_intptr_t), l_rows*size_of_datatype, gpuMemcpyHostToDevice)
      check_memcpy_gpu("tridiag: v_row_dev", successGPU)
#endif

#ifdef WITH_NVTX
      call nvtxRangePop()
#endif
    endif ! (useGPU .and. .not. useCCL)


    ! calculate u**T * v (same as v**T * (A + VU**T + UV**T) * v )
    if (.not. useGPU .or. (useGPU .and. .not. useCCL)) then
      vav = 0 ! x=0
#ifdef WITH_NVTX
      call nvtxRangePush("cpu_dot v_col*u_col")
#endif
      if (l_cols>0) vav = dot_product(v_col(1:l_cols), u_col(1:l_cols))
#ifdef WITH_NVTX
      call nvtxRangePop()
#endif
    endif

#ifdef WITH_MPI
    if (useCCL) then
#if defined(USE_CCL_TRIDIAG)
      successGPU = ccl_Allreduce(vav_dev, vav_dev, int(k_datatype*1, kind=c_size_t), &
                                 cclDataType, cclSum, ccl_comm_cols, my_stream)
      if (.not. successGPU) then
        print *,"Error in ccl_allreduce"
        stop 1
      endif

      successGPU = gpu_stream_synchronize(my_stream) ! TODO_23_11: do we need it here and before ccl_group_start()?
      check_stream_synchronize_gpu("ccl_Allreduce vav_dev", successGPU)
#endif /* USE_CCL_TRIDIAG */

    else ! useCCL
      if (useNonBlockingCollectivesCols) then
        if (wantDebug) call obj%timer%start("mpi_nbc_communication")
        call mpi_iallreduce(MPI_IN_PLACE, vav, 1_MPI_KIND, MPI_MATH_DATATYPE_PRECISION, MPI_SUM, int(mpi_comm_cols,kind=MPI_KIND), &
              allreduce_request3, mpierr)
        call mpi_wait(allreduce_request3, MPI_STATUS_IGNORE, mpierr)
        if (wantDebug) call obj%timer%stop("mpi_nbc_communication")
      else ! useNonBlockingCollectivesCols
        if (wantDebug) call obj%timer%start("mpi_communication")
        call mpi_allreduce(MPI_IN_PLACE, vav, 1_MPI_KIND, MPI_MATH_DATATYPE_PRECISION, MPI_SUM, int(mpi_comm_cols,kind=MPI_KIND), &
                mpierr)
        if (wantDebug) call obj%timer%stop("mpi_communication")
      endif ! useNonBlockingCollectivesCols
    endif ! useCCL
#endif /* WITH_MPI */


    ! store u and v in the matrices U and V
    ! these matrices are stored combined in one here

   if (.not. useCCL) then
#if REALCASE == 1
      conjg_tau = tau(istep)
#endif
#if COMPLEXCASE == 1
      conjg_tau = conjg(tau(istep))
#endif
    endif ! .not. useCCL
    
    if (.not. useGPU) then
#ifdef WITH_NVTX
      call nvtxRangePush("store u,v in U,V")
#endif
      if (l_rows > 0) then
        ! update vu_stored_rows
        vu_stored_rows(1:l_rows,2*n_stored_vecs+1) = conjg_tau*v_row(1:l_rows)
        vu_stored_rows(1:l_rows,2*n_stored_vecs+2) = 0.5*conjg_tau*vav*v_row(1:l_rows) - u_row(1:l_rows)
      endif
      if (l_cols > 0) then
        ! update uv_stored_cols
        uv_stored_cols(1:l_cols,2*n_stored_vecs+1) = 0.5*conjg_tau*vav*v_col(1:l_cols) - u_col(1:l_cols)
        uv_stored_cols(1:l_cols,2*n_stored_vecs+2) = conjg_tau*v_col(1:l_cols)
      endif
#ifdef WITH_NVTX
      call nvtxRangePop()
#endif
    endif ! .not. useGPU


    if (useGPU) then

      ! kernel: update vu_stored_rows_dev, uv_stored_cols
      ! then cpu's "store u,v in U,V" can be deleted. But we should take care of dot_prod below, where vu_stored_rows and uv_stored_cols are used
#ifdef WITH_NVTX
      call nvtxRangePush("kernel gpu_store_u_v_in_uv_vu")
#endif
      if (useCCL) then
        vav_host_or_dev = vav_dev
        !tau_istep_host_or_dev = tau_dev + (istep-1)*size_of_datatype
        tau_istep_host_or_dev = v_row_dev + (l_rows+1-1)*size_of_datatype
      else ! useCCL
        vav_host_or_dev = int(loc(vav),kind=c_intptr_t)
        tau_istep_host_or_dev = int(loc(tau(istep)), kind=c_intptr_t)
      endif ! useCCL
      call gpu_store_u_v_in_uv_vu_PRECISION(vu_stored_rows_dev, uv_stored_cols_dev, v_row_dev, u_row_dev, &
                                          v_col_dev, u_col_dev, tau_dev, aux_complex_dev, &
                                          vav_host_or_dev, tau_istep_host_or_dev, &
                                          l_rows, l_cols, n_stored_vecs,  max_local_rows, max_local_cols, istep, &
                                          useCCL, wantDebug, my_stream)
#ifdef WITH_NVTX
      call nvtxRangePop()
#endif
    endif ! useGPU


    ! We have calculated another Householder Vector, number of implicitly stored increased
    n_stored_vecs = n_stored_vecs+1

    ! If the limit of max_stored_uv is reached, calculate A + VU**T + UV**T
    if (n_stored_vecs == max_stored_uv .or. istep == 3) then        
      
      if (.not. useGPU .OR. .not. mat_vec_as_one_block) then
        do i = 0, (istep-2)/tile_size
          ! go over tiles above (or on) the diagonal
          l_col_beg = i*l_cols_per_tile+1
          l_col_end = min(l_cols,(i+1)*l_cols_per_tile)
          l_row_beg = 1
          l_row_end = min(l_rows,(i+1)*l_rows_per_tile)
          if (l_col_end<l_col_beg .or. l_row_end<l_row_beg) then
            cycle
          endif

          if (useGPU) then
            if (.not. mat_vec_as_one_block) then
              ! if using mat-vec multiply by stripes, it is enough to update tiles above (or on) the diagonal only
              ! we than use the same calls as for CPU version
              if (wantDebug) call obj%timer%start("gpublas_gemm")
              ! update a_dev
              ! a_dev = vu_stored_rows_dev*uv_stored_cols_dev + a_dev
              gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
              call gpublas_PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ,     &
                                      l_row_end-l_row_beg+1, l_col_end-l_col_beg+1, 2*n_stored_vecs,                      &
                                      ONE, vu_stored_rows_dev + (l_row_beg - 1) *                                         &
                                      size_of_datatype,  &
                                      max_local_rows, uv_stored_cols_dev + (l_col_beg - 1) *                              &
                                      size_of_datatype,  &
                                      max_local_cols, ONE, a_dev + ((l_row_beg - 1) + (l_col_beg - 1) * matrixRows) *     &
                                      size_of_datatype , matrixRows, gpuHandle)
              if (wantDebug .and. gpu_vendor() /= SYCL_GPU) successGPU = gpu_DeviceSynchronize()
              if (wantDebug) call obj%timer%stop("gpublas_gemm")
            endif ! .not. mat_vec_as_one_block
          else ! useGPU
            if (wantDebug) call obj%timer%start("blas_gemm")
            ! update a_mat
            ! a_mat = vu_stored_rows*uv_stored_cols + a_mat
            call PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ,                &
                                int(l_row_end-l_row_beg+1,kind=BLAS_KIND), int(l_col_end-l_col_beg+1,kind=BLAS_KIND), &
                                int(2*n_stored_vecs,kind=BLAS_KIND),    &
                                ONE, vu_stored_rows(l_row_beg:max_local_rows,1:2*max_stored_uv), &
                                int(ubound(vu_stored_rows,dim=1),kind=BLAS_KIND),   &
                                uv_stored_cols(l_col_beg,1), &
                                int(ubound(uv_stored_cols,dim=1),kind=BLAS_KIND),        &
                                ONE, a_mat(l_row_beg,l_col_beg), int(matrixRows,kind=BLAS_KIND))
            if (wantDebug) call obj%timer%stop("blas_gemm")
          endif !useGPU
        enddo ! i = 0, (istep-2)/tile_size

      else !.not. useGPU or .not. mat_vec_as_one_block (i.e. useGPU and mat_vec_as_one_block)

        !update whole (remaining) part of matrix, including tiles below diagonal
        !we can do that in one large cublas call
        if (wantDebug) call obj%timer%start("gpublas_gemm")

#ifdef WITH_NVTX
        call nvtxRangePush("gpublas_gemm a_dev+=vu_stored_rows_dev*uv_stored_cols_dev")
#endif

        gpuHandle = obj%gpu_setup%gpublasHandleArray(0)
        ! update a_dev
        ! a_dev = vu_stored_rows_dev*uv_stored_cols_dev + a_dev
        call gpublas_PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, l_rows, l_cols, 2*n_stored_vecs,   &
                                  ONE, vu_stored_rows_dev, max_local_rows, &
                                  uv_stored_cols_dev, max_local_cols,  &
                                  ONE, a_dev, matrixRows, gpuHandle)
#ifdef WITH_NVTX                        
        call nvtxRangePop()
#endif

        if (wantDebug .and. gpu_vendor() /= SYCL_GPU) successGPU = gpu_DeviceSynchronize()
        if (wantDebug) call obj%timer%stop("gpublas_gemm")
        
        ! copy real(a_dev(l_rows,l_cols)) -> d_vec_dev(istep-1) for correct initial value of d_vec_dev before atomicAdd
        if ((.not. isSkewsymmetric) .and. &
            (my_prow == prow(istep-1, nblk, np_rows) .and. my_pcol == pcol(istep-1, nblk, np_cols))) then
          offset_dev = ((l_rows-1) + (l_cols-1)*matrixRows) * size_of_datatype
          successGPU = gpu_memcpy(d_vec_dev + (istep-2)*size_of_datatype_real, &
                                  a_dev + offset_dev, 1*size_of_datatype_real, gpuMemcpyDeviceToDevice)
          check_memcpy_gpu("tridiag a_dev->d_vec_dev", successGPU)
        endif

      endif !.not. useGPU or .not. mat_vec_as_one_block

      n_stored_vecs = 0
    endif ! (n_stored_vecs == max_stored_uv .or. istep == 3)

    if (my_prow == prow(istep-1, nblk, np_rows) .and. my_pcol == pcol(istep-1, nblk, np_cols)) then

      if (useGPU) then
        my_stream = obj%gpu_setup%my_stream
#ifdef WITH_NVTX
        call nvtxRangePush("kernel: gpu_update_matrix_element_add")
#endif
        call gpu_update_matrix_element_add_PRECISION(vu_stored_rows_dev, uv_stored_cols_dev, a_dev, d_vec_dev, &
                                            l_rows, l_cols, matrixRows, max_local_rows, max_local_cols, istep, n_stored_vecs, &
                                            isSkewsymmetric, wantDebug, my_stream)
#ifdef WITH_NVTX
        call nvtxRangePop()
#endif

      else ! useGPU
        if (n_stored_vecs > 0) then
          ! update a_mat (only one elememt!)
          dot_prod = dot_product(vu_stored_rows(l_rows,1:2*n_stored_vecs), uv_stored_cols(l_cols,1:2*n_stored_vecs))
          a_mat(l_rows,l_cols) = a_mat(l_rows,l_cols) + dot_prod
        endif
#if REALCASE == 1
        if (isSkewsymmetric) then
          d_vec(istep-1) = 0.0_rk
        else
          d_vec(istep-1) = a_mat(l_rows,l_cols)
        endif
#endif
#if COMPLEXCASE == 1
        d_vec(istep-1) = real(a_mat(l_rows,l_cols),kind=rk)
#endif
      endif ! useGPU

    endif ! (my_prow == prow(istep-1, nblk, np_rows) .and. my_pcol == pcol(istep-1, nblk, np_cols))

#ifdef WITH_NVTX
    call nvtxRangePop() ! tridi main cycle
#endif

  enddo ! main cycle over istep=na,3,-1

#if COMPLEXCASE == 1
  ! Store e_vec(1) and d_vec(1)

  if (my_pcol==pcol(2, nblk, np_cols)) then
    if (my_prow==prow(1, nblk, np_rows)) then
      ! We use last l_cols value of loop above
      if (useGPU) then
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        num =  1 * size_of_datatype
        call gpu_memcpy_async_and_stream_synchronize &
              ("a_dev -> aux3", a_dev, (matrixRows * (l_cols - 1)) * size_of_datatype, &
                                                  aux3(1:1), &
                                                  1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
        successGPU = gpu_memcpy(int(loc(aux3(1)),kind=c_intptr_t), a_dev + (matrixRows * (l_cols - 1)) * size_of_datatype, &
                                1 * size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("tridiag: a_dev 5", successGPU)
#endif
        vrl = aux3(1)
      else !useGPU
        vrl = a_mat(1,l_cols)
      endif !useGPU

      call hh_transform_complex_&
            &PRECISION &
            (obj, vrl, 0.0_rk, xf, tau(2), wantDebug)
      e_vec(1) = real(vrl,kind=rk)
      a_mat(1,l_cols) = 1. ! for consistency only
    endif ! (my_prow==prow(1, nblk, np_rows))

#ifdef WITH_MPI
    if (useNonBlockingCollectivesRows) then
      if (wantDebug) call obj%timer%start("mpi_nbc_communication")
      call mpi_ibcast(tau(2), 1_MPI_KIND, MPI_COMPLEX_PRECISION, int(prow(1, nblk, np_rows),kind=MPI_KIND), &
                  int(mpi_comm_rows,kind=MPI_KIND), bcast_request2, mpierr)
      call mpi_wait(bcast_request2, MPI_STATUS_IGNORE, mpierr)
      if (wantDebug) call obj%timer%stop("mpi_nbc_communication")
    else
      if (wantDebug) call obj%timer%start("mpi_communication")
      call mpi_bcast(tau(2), 1_MPI_KIND, MPI_COMPLEX_PRECISION, int(prow(1, nblk, np_rows),kind=MPI_KIND), &
                  int(mpi_comm_rows,kind=MPI_KIND),  mpierr)
      if (wantDebug) call obj%timer%stop("mpi_communication")
    endif
#endif /* WITH_MPI */

  endif ! (my_pcol==pcol(2, nblk, np_cols))

#ifdef WITH_MPI
  if (useNonBlockingCollectivesCols) then
    if (wantDebug) call obj%timer%start("mpi_nbc_communication")
    call mpi_ibcast(tau(2), 1_MPI_KIND, MPI_COMPLEX_PRECISION, int(pcol(2, nblk, np_cols),kind=MPI_KIND), &
                int(mpi_comm_cols,kind=MPI_KIND), bcast_request3, mpierr)
    call mpi_wait(bcast_request3, MPI_STATUS_IGNORE, mpierr)
    if (wantDebug) call obj%timer%stop("mpi_nbc_communication")
  else
    if (wantDebug) call obj%timer%start("mpi_communication")
    call mpi_bcast(tau(2), 1_MPI_KIND, MPI_COMPLEX_PRECISION, int(pcol(2, nblk, np_cols),kind=MPI_KIND), &
                int(mpi_comm_cols,kind=MPI_KIND), mpierr)
    if (wantDebug) call obj%timer%stop("mpi_communication")
  endif
#endif /* WITH_MPI */

  if (my_prow == prow(1, nblk, np_rows) .and. my_pcol == pcol(1, nblk, np_cols))  then
    if (useGPU) then
#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
      num =  1 * size_of_datatype
      call gpu_memcpy_async_and_stream_synchronize &
            ("a_dev -> aux3", a_dev, 0_c_intptr_t, &
                                                 aux3(1:1), &
                                                1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else
      successGPU = gpu_memcpy(int(loc(aux3(1)),kind=c_intptr_t), a_dev, &
                             1 * size_of_datatype, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("tridiag: a_dev 6", successGPU)
#endif
      d_vec(1) = PRECISION_REAL(aux3(1))
    else !useGPU
      d_vec(1) = PRECISION_REAL(a_mat(1,1))
    endif !useGPU
  endif ! (my_prow == prow(1, nblk, np_rows) .and. my_pcol == pcol(1, nblk, np_cols))
#endif /* COMPLEXCASE == 1 */

#if REALCASE == 1
  ! Store e_vec(1)

  if (my_prow==prow(1, nblk, np_rows) .and. my_pcol==pcol(2, nblk, np_cols)) then
    if (useGPU) then
#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
      num =  1 * size_of_datatype
      call gpu_memcpy_async_and_stream_synchronize &
            ("a_dev -> e_vec", a_dev,(matrixRows * (l_cols - 1)) * size_of_datatype, &
                                                e_vec(1:na), &
                                                1, num, gpuMemcpyDeviceToHost, my_stream, .false., .true., .false.)
#else /* WITH_GPU_STREAMS */
      successGPU = gpu_memcpy(int(loc(e_vec(1)),kind=c_intptr_t), a_dev + (matrixRows * (l_cols - 1)) * size_of_datatype, &
                              1 * size_of_datatype, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("tridiag: a_dev 7", successGPU)
#endif /* WITH_GPU_STREAMS */
    else !useGPU
      e_vec(1) = a_mat(1,l_cols) ! use last l_cols value of loop above
    endif !useGPU
  endif ! if (my_prow==prow(1, nblk, np_rows) .and. my_pcol==pcol(2, nblk, np_cols))

  ! Store d_vec(1)
  if (my_prow==prow(1, nblk, np_rows) .and. my_pcol==pcol(1, nblk, np_cols)) then
    if(useGPU) then
#ifdef WITH_GPU_STREAMS
      successGPU = gpu_memcpy_async(int(loc(d_vec(1)),kind=c_intptr_t), a_dev, 1 * size_of_datatype, &
                   gpuMemcpyDeviceToHost, my_stream)
      check_memcpy_gpu("tridiag: a_dev 8", successGPU)

      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("tridiag: a_dev 8", successGPU)
#else /* WITH_GPU_STREAMS */
      successGPU = gpu_memcpy(int(loc(d_vec(1)),kind=c_intptr_t), a_dev, 1 * size_of_datatype, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("tridiag: a_dev 8", successGPU)
#endif /* WITH_GPU_STREAMS */
    else !useGPU
      if (isSkewsymmetric) then
        d_vec(1) = 0.0_rk
      else
        d_vec(1) = a_mat(1,1)
      endif
    endif !useGPU
  endif ! (my_prow==prow(1, nblk, np_rows) .and. my_pcol==pcol(1, nblk, np_cols))
#endif /* REALCASE */

  if (useGPU) then
    offset_dev = 1 * size_of_datatype_real
    ! first and last elements of d_vec are treated separately
    successGPU = gpu_memcpy(int(loc(d_vec(2)),kind=c_intptr_t), &
                            d_vec_dev + offset_dev, (na-2) * size_of_datatype_real, gpuMemcpyDeviceToHost)
    check_memcpy_gpu("tridiag: d_vec", successGPU)

    if (useCCL) then
      ! e_vec(1) is treated separately
      offset_dev = 1 * size_of_datatype_real
      successGPU = gpu_memcpy(int(loc(e_vec(2)),kind=c_intptr_t), &
                              e_vec_dev + offset_dev, (na-1) * size_of_datatype_real, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("tridiag: e_vec", successGPU)

      ! tau(2) is treated separately, tau(1) is not used
      offset_dev = 2 * size_of_datatype
      successGPU = gpu_memcpy(int(loc(tau(3)),kind=c_intptr_t), &
                              tau_dev + offset_dev, (na-2) * size_of_datatype, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("tridiag: tau", successGPU)
    endif

    ! todo: should we leave a_mat on the device for further use?
    !successGPU = gpu_free(a_dev)
    !check_dealloc_gpu("tridiag: a_dev 9", successGPU)

    successGPU = gpu_free(v_row_dev)
    check_dealloc_gpu("tridiag: v_row_dev", successGPU)

    successGPU = gpu_free(u_row_dev)
    check_dealloc_gpu("tridiag: (u_row_dev", successGPU)

    successGPU = gpu_free(v_col_dev)
    check_dealloc_gpu("tridiag: v_col_dev", successGPU)

    successGPU = gpu_free(u_col_dev)
    check_dealloc_gpu("tridiag: u_col_dev ", successGPU)

    successGPU = gpu_free(vu_stored_rows_dev)
    check_dealloc_gpu("tridiag: vu_stored_rows_dev ", successGPU)

    successGPU = gpu_free(uv_stored_cols_dev)
    check_dealloc_gpu("tridiag:uv_stored_cols_dev ", successGPU)

    !successGPU = gpu_free(d_vec_dev)
    !check_dealloc_gpu("tridiag: d_vec_dev", successGPU)

    !successGPU = gpu_free(e_vec_dev)
    !check_dealloc_gpu("tridiag: e_vec_dev", successGPU)

    !successGPU = gpu_free(tau_dev)
    !check_dealloc_gpu("tridiag: tau_dev", successGPU)

    successGPU = gpu_free(aux_dev)
    check_dealloc_gpu("tridiag: aux_dev", successGPU)

    successGPU = gpu_free(aux1_dev)
    check_dealloc_gpu("tridiag: aux1_dev", successGPU)

    successGPU = gpu_free(aux_complex_dev)
    check_dealloc_gpu("tridiag: aux_complex_dev", successGPU)

    successGPU = gpu_free(vav_dev)
    check_dealloc_gpu("tridiag: vav_dev", successGPU)

    successGPU = gpu_free(dot_prod_dev)
    check_dealloc_gpu("tridiag: dot_prod_dev", successGPU)

    successGPU = gpu_free(xf_dev)
    check_dealloc_gpu("tridiag: xf_dev", successGPU)

#ifdef USE_CCL_TRIDIAG
    if (.not. isSquareGridGPU) then
      successGPU = gpu_free(aux_transpose_dev)
      check_dealloc_gpu("tridiag: aux_transpose_dev", successGPU)
    endif
#endif
  endif ! useGPU

  ! distribute the arrays d_vec and e_vec to all processors

  allocate(tmp_real(na), stat=istat, errmsg=errorMessage)
  check_allocate("tridiag: tmp_real", istat, errorMessage)

#ifdef WITH_MPI
  if (useNonBlockingCollectivesRows) then
    if (wantDebug) call obj%timer%start("mpi_nbc_communication")
    tmp_real = d_vec
    call mpi_iallreduce(tmp_real, d_vec, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_SUM, &
                       int(mpi_comm_rows,kind=MPI_KIND), allreduce_request4, mpierr)
    call mpi_wait(allreduce_request4, MPI_STATUS_IGNORE, mpierr)
    tmp_real = e_vec
    call mpi_iallreduce(tmp_real, e_vec, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_SUM, &
                       int(mpi_comm_rows,kind=MPI_KIND), allreduce_request6,mpierr)
    call mpi_wait(allreduce_request6, MPI_STATUS_IGNORE, mpierr)
    if (wantDebug) call obj%timer%stop("mpi_nbc_communication")
  else
    if (wantDebug) call obj%timer%start("mpi_communication")
    tmp_real = d_vec
    call mpi_allreduce(tmp_real, d_vec, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_SUM, &
                       int(mpi_comm_rows,kind=MPI_KIND), mpierr)
    tmp_real = e_vec
    call mpi_allreduce(tmp_real, e_vec, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_SUM, &
                       int(mpi_comm_rows,kind=MPI_KIND), mpierr) ! TODO_23_11: change to MPI_IN_PLACE, get rid of tmp_real
    if (wantDebug) call obj%timer%stop("mpi_communication")
  endif
  if (useNonBlockingCollectivesCols) then
    if (wantDebug) call obj%timer%start("mpi_nbc_communication")
    tmp_real = d_vec
    call mpi_iallreduce(tmp_real, d_vec, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_SUM, &
                       int(mpi_comm_cols,kind=MPI_KIND), allreduce_request5, mpierr)
    call mpi_wait(allreduce_request5, MPI_STATUS_IGNORE, mpierr)

    tmp_real = e_vec
    call mpi_iallreduce(tmp_real, e_vec, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_SUM, &
                       int(mpi_comm_cols,kind=MPI_KIND), allreduce_request7, mpierr)
    call mpi_wait(allreduce_request7, MPI_STATUS_IGNORE, mpierr)
    if (wantDebug) call obj%timer%stop("mpi_nbc_communication")
  else
    if (wantDebug) call obj%timer%start("mpi_communication")
    tmp_real = d_vec
    call mpi_allreduce(tmp_real, d_vec, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_SUM, &
                       int(mpi_comm_cols,kind=MPI_KIND), mpierr)
    tmp_real = e_vec
    call mpi_allreduce(tmp_real, e_vec, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_SUM, &
                       int(mpi_comm_cols,kind=MPI_KIND), mpierr)
    if (wantDebug) call obj%timer%stop("mpi_communication")
  endif
#endif /* WITH_MPI */

  deallocate(tmp_real, stat=istat, errmsg=errorMessage)
  check_deallocate("tridiag: tmp_real", istat, errorMessage)

  if (useGPU) then
#ifdef WITH_GPU_STREAMS
#if COMPLEXCASE == 1
    successGPU = gpu_host_unregister(int(loc(aux3),kind=c_intptr_t))
    check_host_unregister_gpu("tridiag: aux3", successGPU)
#endif
#endif

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
      successGPU = gpu_free_host(v_row_host)
      check_host_dealloc_gpu("tridiag: v_row_host", successGPU)
      nullify(v_row)

      successGPU = gpu_free_host(v_col_host)
      check_host_dealloc_gpu("tridiag: v_col_host", successGPU)
      nullify(v_col)

      successGPU = gpu_free_host(u_col_host)
      check_host_dealloc_gpu("tridiag: u_col_host", successGPU)
      nullify(u_col)

      successGPU = gpu_free_host(u_row_host)
      check_host_dealloc_gpu("tridiag: u_row_host", successGPU)
      nullify(u_row)
    else
      deallocate(v_row)
      deallocate(v_col)
      deallocate(u_row)
      deallocate(u_col)
    endif

    if (gpu_vendor() /= OPENMP_OFFLOAD_GPU .and. gpu_vendor() /= SYCL_GPU) then
      successGPU = gpu_host_unregister(int(loc(uv_stored_cols),kind=c_intptr_t))
      check_host_unregister_gpu("tridiag: uv_stored_cols", successGPU)

      successGPU = gpu_host_unregister(int(loc(vu_stored_rows),kind=c_intptr_t))
      check_host_unregister_gpu("tridiag: vu_stored_rows", successGPU)

      successGPU = gpu_host_unregister(int(loc(d_vec),kind=c_intptr_t))
      check_host_unregister_gpu("tridiag: d_vec", successGPU)

      successGPU = gpu_host_unregister(int(loc(e_vec),kind=c_intptr_t))
      check_host_unregister_gpu("tridiag: e_vec", successGPU)

      successGPU = gpu_host_unregister(int(loc(tau),kind=c_intptr_t))
      check_host_unregister_gpu("tridiag: tau", successGPU)

      successGPU = gpu_host_unregister(int(loc(aux),kind=c_intptr_t))
      check_host_unregister_gpu("tridiag: aux", successGPU)

      successGPU = gpu_host_unregister(int(loc(aux1),kind=c_intptr_t))
      check_host_unregister_gpu("tridiag: aux1", successGPU)

      successGPU = gpu_host_unregister(int(loc(vav),kind=c_intptr_t))
      check_host_unregister_gpu("tridiag: vav", successGPU)

      successGPU = gpu_host_unregister(int(loc(dot_prod),kind=c_intptr_t))
      check_host_unregister_gpu("tridiag: dot_prod", successGPU)

      successGPU = gpu_host_unregister(int(loc(xf),kind=c_intptr_t))
      check_host_unregister_gpu("tridiag: xf", successGPU)
    endif
#endif
  else ! useGPU
    deallocate(v_row, v_col, u_row, u_col, stat=istat, errmsg=errorMessage)
    check_deallocate("tridiag: v_row, v_col, u_row, u_col", istat, errorMessage)
  endif ! useGPU

  deallocate(vu_stored_rows, uv_stored_cols, stat=istat, errmsg=errorMessage)
  check_deallocate("tridiag: vu_stored_rows, uv_stored_cols", istat, errorMessage)

  deallocate(aux, stat=istat, errmsg=errorMessage)
  check_deallocate("tridiag: aux", istat, errorMessage)



! copy to device
  if (useGPU) then
    num = na * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    call gpu_memcpy_async_and_stream_synchronize &
            ("tridiag e_vec -> e_vec_dev", e_vec_dev, 0_c_intptr_t, &
                                                 e_vec(1:na), &
                                                 1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
    successGPU = gpu_memcpy(e_vec_dev, int(loc(e_vec(1)),kind=c_intptr_t),  &
                              num, gpuMemcpyHostToDevice)
    check_memcpy_gpu("tridiag: e_vec_dev", successGPU)
#endif

    num = na * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    call gpu_memcpy_async_and_stream_synchronize &
            ("tridiag d_vec -> d_vec_dev", d_vec_dev, 0_c_intptr_t, &
                                                 d_vec(1:na), &
                                                 1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
    successGPU = gpu_memcpy(d_vec_dev, int(loc(d_vec(1)),kind=c_intptr_t),  &
                              num, gpuMemcpyHostToDevice)
    check_memcpy_gpu("tridiag: d_vec_dev", successGPU)
#endif

    num = na * size_of_datatype
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
    call gpu_memcpy_async_and_stream_synchronize &
            ("tridiag tau -> tau_dev", tau_dev, 0_c_intptr_t, &
                                                 tau(1:na), &
                                                 1, num, gpuMemcpyHostToDevice, my_stream, .false., .false., .false.)
#else
    successGPU = gpu_memcpy(tau_dev, int(loc(tau(1)),kind=c_intptr_t),  &
                              num, gpuMemcpyHostToDevice)
    check_memcpy_gpu("tridiag: d_vec_dev", successGPU)
#endif

  endif ! useGPU


  call obj%timer%stop("tridiag_&
       &MATH_DATATYPE&
       &" // &
       PRECISION_SUFFIX // &
       gpuString )

end
