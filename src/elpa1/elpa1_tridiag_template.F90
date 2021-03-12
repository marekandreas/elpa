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
subroutine tridiag_&
  &MATH_DATATYPE&
  &_&
  &PRECISION &
  (obj, na, a_mat, matrixRows, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, d_vec, e_vec, tau, useGPU, wantDebug, max_threads)
  use, intrinsic :: iso_c_binding
  use precision
  use elpa_abstract_impl
  use matrix_plot
  use elpa_omp
  use elpa_blas_interfaces
  use elpa_gpu

  implicit none
#include "../general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout)    :: obj
  integer(kind=ik), intent(in)                  :: na, matrixRows, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
  logical, intent(in)                           :: useGPU, wantDebug
  integer(kind=c_int)                           :: skewsymmetric
  logical                                       :: isSkewsymmetric

  MATH_DATATYPE(kind=rck), intent(out)          :: tau(na)
#ifdef USE_ASSUMED_SIZE
  MATH_DATATYPE(kind=rck), intent(inout)        :: a_mat(matrixRows,*)
#else
  MATH_DATATYPE(kind=rck), intent(inout)        :: a_mat(matrixRows,matrixCols)
#endif
  real(kind=rk), intent(out)                    :: d_vec(na)
  real(kind=rk), intent(out)                    :: e_vec(na)
  integer(kind=ik), parameter                   :: max_stored_uv = 32
  logical,          parameter                   :: mat_vec_as_one_block = .true.

  ! id in processor row and column and total numbers of processor rows and columns
  integer(kind=ik)                              :: my_prow, my_pcol, np_rows, np_cols
  integer(kind=MPI_KIND)                        :: my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI
  integer(kind=MPI_KIND)                        :: mpierr
  integer(kind=ik)                              :: totalblocks, max_loc_block_rows, max_loc_block_cols, max_local_rows, &
                                                   max_local_cols
  ! updated after each istep (in the main cycle) to contain number of
  ! local columns and rows of the remaining part of the matrix
  !integer(kind=ik)                             :: l_cols, l_rows
  integer(kind=ik)                              :: l_cols, l_rows
  integer(kind=ik)                              :: n_stored_vecs

  integer(kind=C_intptr_T)                      :: a_dev, v_row_dev, v_col_dev, u_row_dev, u_col_dev, vu_stored_rows_dev, &
                                                   uv_stored_cols_dev
  logical                                       :: successGPU

  integer(kind=ik)                              :: istep, i, j, l_col_beg, l_col_end, l_row_beg, l_row_end
  integer(kind=ik)                              :: tile_size, l_rows_per_tile, l_cols_per_tile
  integer(kind=c_intptr_t)                      :: a_offset

  integer(kind=ik), intent(in)                  :: max_threads
#ifdef WITH_OPENMP_TRADITIONAL
  integer(kind=ik)                              :: my_thread, n_threads, n_iter
#endif

  real(kind=rk)                                 :: vnorm2
  MATH_DATATYPE(kind=rck)                       :: vav, x, aux(2*max_stored_uv), aux1(2), aux2(2), vrl, xf
#if COMPLEXCASE == 1
  complex(kind=rck)                             :: aux3(1)
#endif

  integer(kind=c_intptr_t)                      :: num
  MATH_DATATYPE(kind=rck), allocatable          :: tmp(:)
  MATH_DATATYPE(kind=rck), pointer              :: v_row(:), & ! used to store calculated Householder Vector
                                                   v_col(:)   ! the same Vector, but transposed 
  MATH_DATATYPE(kind=rck), pointer              :: u_col(:), u_row(:)

  ! the following two matrices store pairs of vectors v and u calculated in each step
  ! at most max_stored_uv Vector pairs are stored, than the matrix A_i is explicitli updated
  ! u and v are stored both in row and Vector forms
  ! pattern: v1,u1,v2,u2,v3,u3,....
  ! todo: It is little bit confusing, I think, that variables _row actually store columns and vice versa
  MATH_DATATYPE(kind=rck), pointer             :: vu_stored_rows(:,:)
  ! pattern: u1,v1,u2,v2,u3,v3,....
  MATH_DATATYPE(kind=rck), allocatable         :: uv_stored_cols(:,:)

#ifdef WITH_OPENMP_TRADITIONAL
  MATH_DATATYPE(kind=rck), allocatable         :: ur_p(:,:), uc_p(:,:)
#endif

  type(c_ptr)                                   :: v_row_host, v_col_host
  type(c_ptr)                                   :: u_row_host, u_col_host
  type(c_ptr)                                   :: vu_stored_rows_host, uv_stored_cols_host
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
  logical                                       :: useIntelGPU

  call obj%get("is_skewsymmetric",skewsymmetric,istat)
  if (istat .ne. ELPA_OK) then
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

  call obj%timer%start("tridiag_&
  &MATH_DATATYPE&
  &" // &
  PRECISION_SUFFIX // &
  gpuString )

  if (wantDebug) call obj%timer%start("mpi_communication")
  call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND), my_prowMPI, mpierr)
  call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)
  call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND), my_pcolMPI, mpierr)
  call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND), np_colsMPI, mpierr)

  my_prow = int(my_prowMPI, kind=c_int)
  np_rows = int(np_rowsMPI, kind=c_int)
  my_pcol = int(my_pcolMPI, kind=c_int)
  np_cols = int(np_colsMPI, kind=c_int)
  if (wantDebug) call obj%timer%stop("mpi_communication")

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
    print *,"Problem setting option for min_tile_size. Aborting..."
    stop
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
  !
  allocate(tmp(MAX(max_local_rows,max_local_cols)), stat=istat, errmsg=errorMessage)
  call check_alloc("tridiag_&
  &MATH_DATATYPE ", "tmp", istat, errorMessage)

  ! allocate v_row 1 element longer to allow store and broadcast tau together with it
  allocate(uv_stored_cols(max_local_cols,2*max_stored_uv), stat=istat, errmsg=errorMessage)
  call check_alloc("tridiag_&
       &MATH_DATATYPE ", "uv_stored_cols", istat, errorMessage)

  allocate(vu_stored_rows(max_local_rows,2*max_stored_uv), stat=istat, errmsg=errorMessage)
  call check_alloc("tridiag_&
       &MATH_DATATYPE ", "vu_stored_rows", istat, errorMessage)

  if (useGPU) then
    if (useIntelGPU) then
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
    else

      num = (max_local_rows+1) * size_of_datatype
      successGPU = gpu_malloc_host(v_row_host, num)
      check_host_alloc_gpu("tridiag: v_row_host", successGPU)
      call c_f_pointer(v_row_host,v_row,(/(max_local_rows+1)/))

      num = (max_local_cols) * size_of_datatype
      successGPU = gpu_malloc_host(v_col_host,num)
      check_host_alloc_gpu("tridiag: v_col_host", successGPU)
      call c_f_pointer(v_col_host,v_col,(/(max_local_cols)/))

      num = (max_local_cols) * size_of_datatype
      successGPU = gpu_malloc_host(u_col_host,num)
      check_host_alloc_gpu("tridiag: u_col_host", successGPU)
      call c_f_pointer(u_col_host,u_col,(/(max_local_cols)/))

      num = (max_local_rows) * size_of_datatype
      successGPU = gpu_malloc_host(u_row_host,num)
      check_host_alloc_gpu("tridiag: u_row_host", successGPU)
      call c_f_pointer(u_row_host,u_row,(/(max_local_rows)/))

      num = (max_local_rows * 2*max_stored_uv) * size_of_datatype
      successGPU = gpu_host_register(int(loc(vu_stored_rows),kind=c_intptr_t),num,&
                    gpuHostRegisterDefault)
      check_host_register_gpu("tridiag: vu_stored_roes", successGPU)

      num = (max_local_cols * 2*max_stored_uv) * size_of_datatype
      successGPU = gpu_host_register(int(loc(uv_stored_cols),kind=c_intptr_t),num,&
                    gpuHostRegisterDefault)
      check_host_register_gpu("tridiag: uv_stored_cols", successGPU)

#if defined(DOUBLE_PRECISION_REAL) || defined(DOUBLE_PRECISION_COMPLEX)
      num = na * 8
#else
      num = na * 4
#endif
      successGPU = gpu_host_register(int(loc(e_vec),kind=c_intptr_t),num,&
                        gpuHostRegisterDefault)
      check_host_register_gpu("tridiag: e_vec", successGPU)

#if defined(DOUBLE_PRECISION_REAL) || defined(DOUBLE_PRECISION_COMPLEX)
      num = na * 8
#else
      num = na * 4
#endif
      successGPU = gpu_host_register(int(loc(d_vec),kind=c_intptr_t),num,&
                        gpuHostRegisterDefault)
      check_host_register_gpu("tridiag: d_vec", successGPU)
    endif
  else
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
      
  endif

#ifdef WITH_OPENMP_TRADITIONAL
  allocate(ur_p(max_local_rows,0:max_threads-1), stat=istat, errmsg=errorMessage)
  call check_alloc("tridiag_&
  &MATH_DATATYPE ", "ur_p", istat, errorMessage)

  allocate(uc_p(max_local_cols,0:max_threads-1), stat=istat, errmsg=errorMessage)
  call check_alloc("tridiag_&
  &MATH_DATATYPE ", "uc_p", istat, errorMessage)
#endif /* WITH_OPENMP_TRADITIONAL */

  tmp = 0
  v_row = 0
  u_row = 0
  v_col = 0
  u_col = 0

  if (useGPU .and. .not.(useIntelGPU) ) then
     successGPU = gpu_malloc(v_row_dev, max_local_rows * size_of_datatype)
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
     check_alloc_gpu("tridiag: vu_stored_rows_dev", successGPU)
  endif !useGPU

  !if (useIntelGPU) then
  !  ! needed later
  !endif


  d_vec(:) = 0
  e_vec(:) = 0
  tau(:) = 0

  n_stored_vecs = 0

  l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a_mat
  l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local cols of a_mat

  if (my_prow == prow(na, nblk, np_rows) .and. my_pcol == pcol(na, nblk, np_cols)) &
#if COMPLEXCASE == 1
  d_vec(na) = real(a_mat(l_rows,l_cols), kind=rk)
#endif
#if REALCASE == 1
  d_vec(na) = a_mat(l_rows,l_cols)
#endif

  if (useGPU .and. .not.(useIntelGPU)) then
    ! allocate memmory for matrix A on the device and than copy the matrix

    num = matrixRows * matrixCols * size_of_datatype

    successGPU = gpu_malloc(a_dev, num)
    check_alloc_gpu("tridiag: a_dev", successGPU)

    successGPU = gpu_host_register(int(loc(a_mat),kind=c_intptr_t),num,&
                  gpuHostRegisterDefault)
    check_host_register_gpu("tridiag: a_mat", successGPU)

    successGPU = gpu_memcpy(a_dev, int(loc(a_mat(1,1)),kind=c_intptr_t), &
                              num, gpuMemcpyHostToDevice)
    check_memcpy_gpu("tridiag: a_dev", successGPU)
  endif

  !if (useIntelGPU) then
  !  ! needed later
  !endif

  ! main cycle of tridiagonalization
  ! in each step, 1 Householder Vector is calculated
  do istep = na, nblockEnd ,-1

    ! Calculate number of local rows and columns of the still remaining matrix
    ! on the local processor
    l_rows = local_index(istep-1, my_prow, np_rows, nblk, -1)
    l_cols = local_index(istep-1, my_pcol, np_cols, nblk, -1)

    ! Calculate Vector for Householder transformation on all procs
    ! owning column istep

    if (my_pcol == pcol(istep, nblk, np_cols)) then

      ! Get Vector to be transformed; distribute last element and norm of
      ! remaining elements to all procs in current column

      ! copy l_cols + 1 column of A to v_row
      if (useGPU) then
        if (useIntelGPU) then
          v_row(1:l_rows) = a_mat(1:l_rows,l_cols+1)
        else
          a_offset = l_cols * matrixRows * size_of_datatype
          ! we use v_row on the host at the moment! successGPU = gpu_memcpy(v_row_dev, a_dev + a_offset, 
          ! (l_rows)*size_of_PRECISION_real, gpuMemcpyDeviceToDevice)

          successGPU = gpu_memcpy(int(loc(v_row),kind=c_intptr_t), &
                                    a_dev + a_offset, (l_rows)* size_of_datatype, gpuMemcpyDeviceToHost)
          check_memcpy_gpu("tridiag a_dev 1", successGPU)
        endif
      else
        v_row(1:l_rows) = a_mat(1:l_rows,l_cols+1)
      endif

      if (n_stored_vecs > 0 .and. l_rows > 0) then
#if COMPLEXCASE == 1
        aux(1:2*n_stored_vecs) = conjg(uv_stored_cols(l_cols+1,1:2*n_stored_vecs))
#endif
        if (useIntelGPU) then
                print *,"intel phase aaaaaaaaaaaaaaaaaaaaaaaaaa"
          if (wantDebug) call obj%timer%start("mkl_offload")
#if REALCASE == 1
          aux(1:2*n_stored_vecs) = uv_stored_cols(l_cols+1,1:2*n_stored_vecs)
#endif

#if 0
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
#endif /* 0 */

#ifdef WITH_INTEL_GPU_VERSION
          ! check why the copy to aux is necessary
          call mkl_offload_PRECISION_GEMV('N',   &
                              int(l_rows,kind=BLAS_KIND), int(2*n_stored_vecs,kind=BLAS_KIND), &
                              ONE, vu_stored_rows, int(ubound(vu_stored_rows,dim=1),kind=BLAS_KIND), &
#if REALCASE == 1
                              !uv_stored_cols(l_cols+1,1), &
                              !uv_stored_cols(l_cols+1,1), int((2*max_stored_uv) * &
                              !(max_local_cols-(l_cols+1)+1),kind=BLAS_KIND), &
                              !int(ubound(uv_stored_cols,dim=1),kind=BLAS_KIND), &
                              aux, 1_BLAS_KIND,  &
#endif
#if COMPLEXCASE == 1
                              !aux, int(2*max_stored_uv,kind=BLAS_KIND), 1_BLAS_KIND,  &
                              aux, 1_BLAS_KIND,  &
#endif
                              !ONE, v_row, int(max_local_rows+1,kind=BLAS_KIND), 1_BLAS_KIND)
                              ONE, v_row, 1_BLAS_KIND)
#endif /* WITH_INTEL_GPU_VERSION */

          if (wantDebug) call obj%timer%stop("mkl_offload")
        else
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
        endif
      endif

      if (my_prow == prow(istep-1, nblk, np_rows)) then
        aux1(1) = dot_product(v_row(1:l_rows-1),v_row(1:l_rows-1))
        aux1(2) = v_row(l_rows)
      else
        aux1(1) = dot_product(v_row(1:l_rows),v_row(1:l_rows))
        aux1(2) = 0.
      endif

#ifdef WITH_MPI
      if (wantDebug) call obj%timer%start("mpi_communication")
      call mpi_allreduce(aux1, aux2, 2_MPI_KIND, MPI_MATH_DATATYPE_PRECISION, &
                           MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
      if (wantDebug) call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
      aux2 = aux1
#endif /* WITH_MPI */

#if REALCASE == 1
      vnorm2 = aux2(1)
#endif
#if COMPLEXCASE == 1
      vnorm2 = real(aux2(1),kind=rk)
#endif
      vrl    = aux2(2)

      ! Householder transformation
#if REALCASE == 1
      call hh_transform_real_&
#endif
#if COMPLEXCASE == 1
      call hh_transform_complex_&
#endif
               &PRECISION &
               (obj, vrl, vnorm2, xf, tau(istep), wantDebug)
      ! Scale v_row and store Householder Vector for back transformation

      v_row(1:l_rows) = v_row(1:l_rows) * xf
      if (my_prow == prow(istep-1, nblk, np_rows)) then
        v_row(l_rows) = 1.

        ! vrl is newly computed off-diagonal element of the final tridiagonal matrix
#if REALCASE == 1
        e_vec(istep-1) = vrl
#endif
#if COMPLEXCASE == 1
        e_vec(istep-1) = real(vrl,kind=rk)
#endif
      endif

      ! store Householder Vector for back transformation
      a_mat(1:l_rows,l_cols+1) = v_row(1:l_rows)

      ! add tau after the end of actuall v_row, to be broadcasted with it
      v_row(l_rows+1) = tau(istep)
    endif !(my_pcol == pcol(istep, nblk, np_cols))

!          SAVE_MATR("HH vec stored", na - istep + 1)

#ifdef WITH_MPI
    if (wantDebug) call obj%timer%start("mpi_communication")
    ! Broadcast the Householder Vector (and tau) along columns
    call MPI_Bcast(v_row, int(l_rows+1,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,    &
                   int(pcol(istep, nblk, np_cols),kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
    if (wantDebug) call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */

    !recover tau, which has been broadcasted together with v_row
    tau(istep) =  v_row(l_rows+1)

    ! Transpose Householder Vector v_row -> v_col
    call elpa_transpose_vectors_&
        &MATH_DATATYPE&
        &_&
        &PRECISION &
              (obj, v_row, ubound(v_row,dim=1), mpi_comm_rows, v_col, ubound(v_col,dim=1), mpi_comm_cols, &
               1, istep-1, 1, nblk, max_threads)

    ! Calculate u = (A + VU**T + UV**T)*v

    ! For cache efficiency, we use only the upper half of the matrix tiles for this,
    ! thus the result is partly in u_col(:) and partly in u_row(:)

    u_col(1:l_cols) = 0
    u_row(1:l_rows) = 0
    if (l_rows > 0 .and. l_cols> 0 ) then
     if (useGPU .and. .not.(useIntelGPU)) then
       successGPU = gpu_memset(u_col_dev, 0, l_cols * size_of_datatype)
       check_memcpy_gpu("tridiag: u_col_dev", successGPU)

       successGPU = gpu_memset(u_row_dev, 0, l_rows * size_of_datatype)
       check_memcpy_gpu("tridiag: u_row_dev", successGPU)

       successGPU = gpu_memcpy(v_col_dev, int(loc(v_col(1)),kind=c_intptr_t), &
                     l_cols * size_of_datatype, gpuMemcpyHostToDevice)

       check_memcpy_gpu("tridiag: v_col_dev", successGPU)

       successGPU = gpu_memcpy(v_row_dev, int(loc(v_row(1)),kind=c_intptr_t), &
                                 l_rows * size_of_datatype, gpuMemcpyHostToDevice)
       check_memcpy_gpu("tridiag: v_row_dev", successGPU)
     endif ! useGPU

     !if (useIntelGPU) then
     !  ! needed later when we can do explicit memcopy
     !endif

#ifdef WITH_OPENMP_TRADITIONAL
     call obj%timer%start("OpenMP parallel")
!todo : check whether GPU implementation with large matrix multiply is beneficial
!       for a larger number of threads; could be addressed with autotuning if this
!       is the case
!$omp parallel &
!$omp num_threads(max_threads) &
!$omp default(none) &
!$omp private(my_thread,n_threads,n_iter,i,l_col_beg,l_col_end,j,l_row_beg,l_row_end) &
!$omp shared(useGPU, isSkewsymmetric, gpuMemcpyDeviceToHost, successGPU, u_row, u_row_dev, &
!$omp &      v_row, v_row_dev, v_col, v_col_dev, u_col, u_col_dev, a_dev, a_offset, &
!$omp&       max_local_cols, max_local_rows, obj, wantDebug, l_rows_per_tile, l_cols_per_tile, &
!$omp&       matrixRows, istep, tile_size, l_rows, l_cols, ur_p, uc_p, a_mat, useIntelGPU, &
!$omp&       matrixCols)
     my_thread = omp_get_thread_num()
          
     n_threads = omp_get_num_threads()

     n_iter = 0

     ! first calculate A*v part of (A + VU**T + UV**T)*v
     uc_p(1:l_cols,my_thread) = 0.
     ur_p(1:l_rows,my_thread) = 0.
#endif /* WITH_OPENMP_TRADITIONAL */
     do i= 0, (istep-2)/tile_size
       l_col_beg = i*l_cols_per_tile+1
       l_col_end = min(l_cols,(i+1)*l_cols_per_tile)
       if (l_col_end < l_col_beg) cycle
       do j = 0, i
         l_row_beg = j*l_rows_per_tile+1
         l_row_end = min(l_rows,(j+1)*l_rows_per_tile)
         if (l_row_end < l_row_beg) cycle
#ifdef WITH_OPENMP_TRADITIONAL
         if (mod(n_iter,n_threads) == my_thread) then
           if (.not. useGPU) then
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
           endif
           if (wantDebug) call obj%timer%stop("blas")
         endif
         n_iter = n_iter+1
#else /* WITH_OPENMP_TRADITIONAL */

         ! multiplication by blocks is efficient only for CPU
         ! for GPU we introduced 2 other ways, either by stripes (more simmilar to the original
         ! CPU implementation) or by one large matrix Vector multiply
         if (.not. useGPU) then
           if (wantDebug) call obj%timer%start("blas")
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
               call PRECISION_GEMV('N',int(l_row_end-l_row_beg+1,kind=BLAS_KIND), int(l_col_end-l_col_beg+1,kind=BLAS_KIND),  &
                                   ONE, a_mat(l_row_beg,l_col_beg), int(matrixRows,kind=BLAS_KIND),               &
                                   v_col(l_col_beg:max_local_cols), 1_BLAS_KIND, ONE, u_row(l_row_beg:max_local_rows), &
                                   1_BLAS_KIND)
             endif
           endif
           if (wantDebug) call obj%timer%stop("blas")
         endif ! not useGPU

#endif /* WITH_OPENMP_TRADITIONAL */
            enddo  ! j=0,i
          enddo  ! i=0,(istep-2)/tile_size

          if (useGPU) then
            if (mat_vec_as_one_block) then
              if (useIntelGPU) then
                 if (wantDebug) call obj%timer%start("mkl_offload")
#if 0
                call PRECISION_GEMV(BLAS_TRANS_OR_CONJ, int(l_rows,kind=BLAS_KIND),int(l_cols,kind=BLAS_KIND),  &
                                          ONE, a_mat, int(matrixRows,kind=BLAS_KIND),       &
                                          v_row , 1_BLAS_KIND,          &
                                          ONE, u_col, 1_BLAS_KIND)
#endif
#ifdef WITH_INTEL_GPU_VERSION
                call mkl_offload_PRECISION_GEMV(BLAS_TRANS_OR_CONJ, int(l_rows,kind=BLAS_KIND),int(l_cols,kind=BLAS_KIND),  &
                                          ONE, a_mat, int(matrixRows,kind=BLAS_KIND),       &
                                          v_row , 1_BLAS_KIND,          &
                                          ONE, u_col, 1_BLAS_KIND)
#endif

                if (wantDebug) call obj%timer%stop("mkl_offload")

              else
                ! Unlike for CPU, we (for each MPI thread) do just one large mat-vec multiplication
                ! this requires altering of the algorithm when later explicitly updating the matrix
                ! after max_stored_uv is reached : we need to update all tiles, not only those above diagonal
                if (wantDebug) call obj%timer%start("gpublas")
                call gpublas_PRECISION_GEMV(BLAS_TRANS_OR_CONJ, l_rows,l_cols,  &
                                          ONE, a_dev, matrixRows,                   &
                                          v_row_dev , 1,                          &
                                          ONE, u_col_dev, 1)

       ! todo: try with non transposed!!!
!                 if(i/=j) then
!                   call gpublas_PRECISION_GEMV('N', l_row_end-l_row_beg+1,l_col_end-l_col_beg+1,  &
!                                             ONE, a_dev + a_offset, matrixRows,                        &
!                                             v_col_dev + (l_col_beg - 1) *                      &
!                                             size_of_datatype, 1,                          &
!                                             ONE, u_row_dev + (l_row_beg - 1) *                 &
!                                             size_of_datatype, 1)
!                 endif
                if (wantDebug) call obj%timer%stop("gpublas")
              endif
            else  ! mat_vec_as_one_block
              !perform multiplication by stripes - it is faster than by blocks, since we call cublas with
              !larger matrices. In general, however, this algorithm is very simmilar to the one with CPU
              do i=0,(istep-2)/tile_size
                  l_col_beg = i*l_cols_per_tile+1
                  l_col_end = min(l_cols,(i+1)*l_cols_per_tile)
                  if(l_col_end<l_col_beg) cycle

                  l_row_beg = 1
                  l_row_end = min(l_rows,(i+1)*l_rows_per_tile)
                  
                  if (useIntelGPU) then
                    if (wantDebug) call obj%timer%start("mkl_offload")
#if 0
                    call PRECISION_GEMV(BLAS_TRANS_OR_CONJ, &
                              int(l_row_end-l_row_beg+1,kind=BLAS_KIND), int(l_col_end-l_col_beg+1,kind=BLAS_KIND), &
                              ONE, a_mat(l_row_beg,l_col_beg), int(matrixRows,kind=BLAS_KIND),  &
                              v_row(l_row_beg:max_local_rows+1), 1_BLAS_KIND,  &
                              ONE, u_col(l_col_beg:max_local_cols), 1_BLAS_KIND)
#endif
#ifdef WITH_INTEL_GPU_VERSION
                    call mkl_offload_PRECISION_GEMV(BLAS_TRANS_OR_CONJ, &
                              int(l_row_end-l_row_beg+1,kind=BLAS_KIND), int(l_col_end-l_col_beg+1,kind=BLAS_KIND), &
                              ONE, a_mat(l_row_beg:matrixRows,l_col_beg:matrixCols), int(matrixRows,kind=BLAS_KIND),  &
                              v_row(l_row_beg:max_local_rows+1), 1_BLAS_KIND,  &
                              ONE, u_col(l_col_beg:max_local_cols), 1_BLAS_KIND)
#endif
                    if (wantDebug) call obj%timer%stop("mkl_offload")

                  else
                    a_offset = ((l_row_beg-1) + (l_col_beg - 1) * matrixRows) * &
                            size_of_datatype

                    call gpublas_PRECISION_GEMV(BLAS_TRANS_OR_CONJ, &
                                l_row_end-l_row_beg+1, l_col_end-l_col_beg+1, &
                                ONE, a_dev + a_offset, matrixRows,  &
                                v_row_dev + (l_row_beg - 1) * size_of_datatype, 1,  &
                                ONE, u_col_dev + (l_col_beg - 1) * size_of_datatype, 1)
                endif
              enddo

              do i=0,(istep-2)/tile_size
                  l_col_beg = i*l_cols_per_tile+1
                  l_col_end = min(l_cols,(i+1)*l_cols_per_tile)
                  if(l_col_end<l_col_beg) cycle

                  l_row_beg = 1
                  l_row_end = min(l_rows,i*l_rows_per_tile)
                  
                  if (useIntelGPU) then
                    if (wantDebug) call obj%timer%start("mkl_offload")
#if 0
                    if (isSkewsymmetric) then
                       call PRECISION_GEMV('N', int(l_row_end-l_row_beg+1,kind=BLAS_KIND), &
                                                int(l_col_end-l_col_beg+1,kind=BLAS_KIND), &
                                  -ONE, a_mat(l_row_beg,l_col_beg), int(matrixRows,kind=BLAS_KIND), &
                                  v_col(l_col_beg:max_local_cols), 1_BLAS_KIND, &
                                   ONE, u_row(l_row_beg:max_local_rows), 1_BLAS_KIND)
                    else
                       call PRECISION_GEMV('N', int(l_row_end-l_row_beg+1,kind=BLAS_KIND), &
                                              int(l_col_end-l_col_beg+1,kind=BLAS_KIND), &
                                 ONE, a_mat(l_row_beg,l_col_beg), int(matrixRows,kind=BLAS_KIND), &
                                 v_col(l_col_beg:max_local_cols), 1_BLAS_KIND, &
                                 ONE, u_row(l_row_beg:max_local_rows), 1_BLAS_KIND)
                    endif
#endif
#ifdef WITH_INTEL_GPU_VERSION
                    if (isSkewsymmetric) then
                       call mkl_offload_PRECISION_GEMV('N', int(l_row_end-l_row_beg+1,kind=BLAS_KIND), &
                                              int(l_col_end-l_col_beg+1,kind=BLAS_KIND), &
                                   -ONE, a_mat(l_row_beg:matrixRows,l_col_beg:matrixCols), int(matrixRows,kind=BLAS_KIND), &
                                 v_col(l_col_beg:max_local_cols), 1_BLAS_KIND, &
                                   ONE, u_row(l_row_beg:max_local_rows), &
                                 1_BLAS_KIND)
                    else
                       call mkl_offload_PRECISION_GEMV('N', int(l_row_end-l_row_beg+1,kind=BLAS_KIND), &
                                                int(l_col_end-l_col_beg+1,kind=BLAS_KIND), &
                                 ONE, a_mat(l_row_beg:matrixRows,l_col_beg:matrixCols), int(matrixRows,kind=BLAS_KIND), &
                                   v_col(l_col_beg:max_local_cols), 1_BLAS_KIND, &
                                   ONE, u_row(l_row_beg:max_local_rows),  &
                                 1_BLAS_KIND)
                    endif
#endif
                    if (wantDebug) call obj%timer%stop("mkl_offload")


                  else
                    a_offset = ((l_row_beg-1) + (l_col_beg - 1) * matrixRows) * &
                            size_of_datatype
                    if (isSkewsymmetric) then
                       call gpublas_PRECISION_GEMV('N', l_row_end-l_row_beg+1, l_col_end-l_col_beg+1, &
                                   -ONE, a_dev + a_offset, matrixRows, &
                                   v_col_dev + (l_col_beg - 1) * size_of_datatype,1, &
                                   ONE, u_row_dev + (l_row_beg - 1) * size_of_datatype, 1)
                    else
                       call gpublas_PRECISION_GEMV('N', l_row_end-l_row_beg+1, l_col_end-l_col_beg+1, &
                                   ONE, a_dev + a_offset, matrixRows, &
                                   v_col_dev + (l_col_beg - 1) * size_of_datatype,1, &
                                   ONE, u_row_dev + (l_row_beg - 1) * size_of_datatype, 1)
                   endif
                endif
              enddo
            end if !multiplication as one block / per stripes

            if (.not.(useIntelGPU)) then
              successGPU = gpu_memcpy(int(loc(u_col(1)),kind=c_intptr_t), &
                          u_col_dev, l_cols * size_of_datatype, gpuMemcpyDeviceToHost)
              check_memcpy_gpu("tridiag: u_col_dev 1", successGPU)

              successGPU = gpu_memcpy(int(loc(u_row(1)),kind=c_intptr_t), &
                          u_row_dev, l_rows * size_of_datatype, gpuMemcpyDeviceToHost)
              check_memcpy_gpu("tridiag: u_row_dev 1", successGPU)
            endif
            !if (useIntelGPU) then
            !  
            !endif

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
           if (wantDebug) call obj%timer%start("blas")
#if REALCASE == 1
           call PRECISION_GEMV('T',     &
#endif
#if COMPLEXCASE == 1
           call PRECISION_GEMV('C',     &
#endif
                               int(l_rows,kind=BLAS_KIND), int(2*n_stored_vecs,kind=BLAS_KIND),   &
                               ONE, vu_stored_rows, int(ubound(vu_stored_rows,dim=1),kind=BLAS_KIND),   &
                               v_row,  1_BLAS_KIND, ZERO, aux, 1_BLAS_KIND)

           call PRECISION_GEMV('N', int(l_cols,kind=BLAS_KIND), int(2*n_stored_vecs,kind=BLAS_KIND),   &
                               ONE, uv_stored_cols, int(ubound(uv_stored_cols,dim=1),kind=BLAS_KIND),   &
                               aux, 1_BLAS_KIND, ONE, u_col,  1_BLAS_KIND)
           if (wantDebug) call obj%timer%stop("blas")
         endif

       endif  ! (l_rows>0 .and. l_cols>0)

       ! Sum up all u_row(:) parts along rows and add them to the u_col(:) parts
       ! on the processors containing the diagonal
       ! This is only necessary if u_row has been calculated, i.e. if the
       ! global tile size is smaller than the global remaining matrix

       if (tile_size < istep-1) then

         call elpa_reduce_add_vectors_&
         &MATH_DATATYPE&
         &_&
         &PRECISION &
         (obj, u_row, ubound(u_row,dim=1), mpi_comm_rows, u_col, ubound(u_col,dim=1), &
         mpi_comm_cols, istep-1, 1, nblk, max_threads)

       endif

       ! Sum up all the u_col(:) parts, transpose u_col -> u_row

       if (l_cols>0) then
         tmp(1:l_cols) = u_col(1:l_cols)
#ifdef WITH_MPI
         if (wantDebug) call obj%timer%start("mpi_communication")
         call mpi_allreduce(tmp, u_col, int(l_cols,kind=MPI_KIND), MPI_MATH_DATATYPE_PRECISION,    &
                            MPI_SUM, int(mpi_comm_rows,kind=MPI_KIND), mpierr)
         if (wantDebug) call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */
         u_col = tmp
#endif /* WITH_MPI */
       endif
       if (isSkewsymmetric) then
          call elpa_transpose_vectors_ss_&
          &MATH_DATATYPE&
          &_&
          &PRECISION &
          (obj, u_col, ubound(u_col,dim=1), mpi_comm_cols, u_row, ubound(u_row,dim=1), &
           mpi_comm_rows, 1, istep-1, 1, nblk, max_threads)
       else
          call elpa_transpose_vectors_&
          &MATH_DATATYPE&
          &_&
          &PRECISION &
          (obj, u_col, ubound(u_col,dim=1), mpi_comm_cols, u_row, ubound(u_row,dim=1), &
           mpi_comm_rows, 1, istep-1, 1, nblk, max_threads)
       endif

       ! calculate u**T * v (same as v**T * (A + VU**T + UV**T) * v )
       x = 0
       if (l_cols>0)  &
       x = dot_product(v_col(1:l_cols),u_col(1:l_cols))

#ifdef WITH_MPI
       if (wantDebug) call obj%timer%start("mpi_communication")
       call mpi_allreduce(x, vav, 1_MPI_KIND, MPI_MATH_DATATYPE_PRECISION, MPI_SUM, int(mpi_comm_cols,kind=MPI_KIND), mpierr)
       if (wantDebug) call obj%timer%stop("mpi_communication")
#else /* WITH_MPI */

       vav = x

#endif /* WITH_MPI */

       ! store u and v in the matrices U and V
       ! these matrices are stored combined in one here

       do j=1,l_rows
#if REALCASE == 1
         vu_stored_rows(j,2*n_stored_vecs+1) = tau(istep)*v_row(j)
         vu_stored_rows(j,2*n_stored_vecs+2) = 0.5*tau(istep)*vav*v_row(j) - u_row(j)
#endif
#if COMPLEXCASE == 1
         vu_stored_rows(j,2*n_stored_vecs+1) = conjg(tau(istep))*v_row(j)
         vu_stored_rows(j,2*n_stored_vecs+2) = 0.5*conjg(tau(istep))*vav*v_row(j) - u_row(j)
#endif
       enddo
       do j=1,l_cols
#if REALCASE == 1
         uv_stored_cols(j,2*n_stored_vecs+1) = 0.5*tau(istep)*vav*v_col(j) - u_col(j)
         uv_stored_cols(j,2*n_stored_vecs+2) = tau(istep)*v_col(j)
#endif
#if COMPLEXCASE == 1
         uv_stored_cols(j,2*n_stored_vecs+1) = 0.5*conjg(tau(istep))*vav*v_col(j) - u_col(j)
         uv_stored_cols(j,2*n_stored_vecs+2) = conjg(tau(istep))*v_col(j)
#endif
       enddo

       ! We have calculated another Hauseholder Vector, number of implicitly stored increased
       n_stored_vecs = n_stored_vecs+1

       ! If the limit of max_stored_uv is reached, calculate A + VU**T + UV**T
       if (n_stored_vecs == max_stored_uv .or. istep == 3) then

         if (useGPU .and. .not.(useIntelGPU)) then
           successGPU = gpu_memcpy(vu_stored_rows_dev, int(loc(vu_stored_rows(1,1)),kind=c_intptr_t), &
                                     max_local_rows * 2 * max_stored_uv *          &
                                     size_of_datatype, gpuMemcpyHostToDevice)
           check_memcpy_gpu("tridiag: uv_stored_rows_dev", successGPU)

           successGPU = gpu_memcpy(uv_stored_cols_dev, int(loc(uv_stored_cols(1,1)),kind=c_intptr_t), &
                                     max_local_cols * 2 * max_stored_uv *          &
                                     size_of_datatype, gpuMemcpyHostToDevice)
           check_memcpy_gpu("tridiag: uv_stored_cols_dev", successGPU)
         endif
          !if (useIntelGPU) then
          !  ! needed later when we can do explicit offloads
          !endif

         do i = 0, (istep-2)/tile_size
           ! go over tiles above (or on) the diagonal
           l_col_beg = i*l_cols_per_tile+1
           l_col_end = min(l_cols,(i+1)*l_cols_per_tile)
           l_row_beg = 1
           l_row_end = min(l_rows,(i+1)*l_rows_per_tile)
           if (l_col_end<l_col_beg .or. l_row_end<l_row_beg) &
           cycle


           if (useGPU) then
             if (.not. mat_vec_as_one_block) then
               if (useIntelGPU) then
                  if (wantDebug) call obj%timer%start("mkl_offload")

                  call PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ,     &
                                       int(l_row_end-l_row_beg+1,kind=BLAS_KIND), &
                                       int(l_col_end-l_col_beg+1,kind=BLAS_KIND), int(2*n_stored_vecs,kind=BLAS_KIND),     &
                                       ONE, vu_stored_rows(l_row_beg:max_local_rows,1:2*max_stored_uv),                    &
                                       int(max_local_rows,kind=BLAS_KIND), uv_stored_cols(l_col_beg,1),                    &
                                       int(max_local_cols,kind=BLAS_KIND), ONE, a_mat(l_row_beg,l_col_beg),                &
                                       int(matrixRows,kind=BLAS_KIND))
#ifdef WITH_INTEL_GPU_VERSION
#if 0
                  ! offload fails, check this
                  call mkl_offload_PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ,     &
                                       int(l_row_end-l_row_beg+1,kind=BLAS_KIND), &
                                       int(l_col_end-l_col_beg+1,kind=BLAS_KIND), int(2*n_stored_vecs,kind=BLAS_KIND),     &
                                       ONE, vu_stored_rows(l_row_beg:max_local_rows,1:2*max_stored_uv),                    &
                                       int(max_local_rows,kind=BLAS_KIND), &
                                       uv_stored_cols(l_col_beg:max_local_cols,1:2*max_stored_uv),                    &
                                       int(max_local_cols,kind=BLAS_KIND), ONE, &
                                       a_mat(l_row_beg:matrixRows,l_col_beg:matrixCols),                &
                                       int(matrixRows,kind=BLAS_KIND))
#endif
#endif
                  if (wantDebug) call obj%timer%stop("mkl_offload")

               else
                 ! if using mat-vec multiply by stripes, it is enough to update tiles above (or on) the diagonal only
                 ! we than use the same calls as for CPU version
                 if (wantDebug) call obj%timer%start("gpublas")
                 call gpublas_PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ,     &
                                         l_row_end-l_row_beg+1, l_col_end-l_col_beg+1, 2*n_stored_vecs,                      &
                                         ONE, vu_stored_rows_dev + (l_row_beg - 1) *                                         &
                                         size_of_datatype,  &
                                         max_local_rows, uv_stored_cols_dev + (l_col_beg - 1) *                              &
                                         size_of_datatype,  &
                                         max_local_cols, ONE, a_dev + ((l_row_beg - 1) + (l_col_beg - 1) * matrixRows) *     &
                                         size_of_datatype , matrixRows)
                 if (wantDebug) call obj%timer%stop("gpublas")
               endif
             endif
           else !useGPU
             if (wantDebug) call obj%timer%start("blas")
             call PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ,                &
                                  int(l_row_end-l_row_beg+1,kind=BLAS_KIND), int(l_col_end-l_col_beg+1,kind=BLAS_KIND), &
                                  int(2*n_stored_vecs,kind=BLAS_KIND),    &
                                  ONE, vu_stored_rows(l_row_beg:max_local_rows,1:2*max_stored_uv), &
                                  int(ubound(vu_stored_rows,dim=1),kind=BLAS_KIND),   &
                                  uv_stored_cols(l_col_beg,1), &
                                  int(ubound(uv_stored_cols,dim=1),kind=BLAS_KIND),        &
                                  ONE, a_mat(l_row_beg,l_col_beg), int(matrixRows,kind=BLAS_KIND))
             if (wantDebug) call obj%timer%stop("blas")
           endif !useGPU
         enddo

         if (useGPU) then
           if (mat_vec_as_one_block) then
             if (useIntelGPU) then
                call PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, int(l_rows,kind=BLAS_KIND), int(l_cols,kind=BLAS_KIND), &
                                    int(2*n_stored_vecs, kind=BLAS_KIND), ONE,  &
                                    vu_stored_rows, int(max_local_rows,kind=BLAS_KIND), &
                                    uv_stored_cols, int(max_local_cols,kind=BLAS_KIND),  &
                                    ONE, a_mat, int(matrixRows,kind=BLAS_KIND))
#ifdef WITH_INTEL_GPU_VERSION
#if 0
                ! offload fails, check this
                call mkl_offload_PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, int(l_rows,kind=BLAS_KIND), &
                                    int(l_cols,kind=BLAS_KIND), &
                                    int(2*n_stored_vecs, kind=BLAS_KIND), ONE,  &
                                    vu_stored_rows(1:max_local_rows,1:2*max_stored_uv), &
                                    int(max_local_rows,kind=BLAS_KIND), &
                                    uv_stored_cols(1:max_local_cols,1:2*max_stored_uv), &
                                    int(max_local_cols,kind=BLAS_KIND),  &
                                    ONE, a_mat(1:matrixRows,1:matrixCols), int(matrixRows,kind=BLAS_KIND))
#endif
#endif
                if (wantDebug) call obj%timer%stop("mkl_offload")
             else
               !update whole (remaining) part of matrix, including tiles below diagonal
               !we can do that in one large cublas call
               if (wantDebug) call obj%timer%start("gpublas")
               call gpublas_PRECISION_GEMM('N', BLAS_TRANS_OR_CONJ, l_rows, l_cols, 2*n_stored_vecs,   &
                                         ONE, vu_stored_rows_dev, max_local_rows, &
                                         uv_stored_cols_dev, max_local_cols,  &
                                         ONE, a_dev, matrixRows)
               if (wantDebug) call obj%timer%stop("gpublas")
             endif
           endif
         endif

         n_stored_vecs = 0
       endif

       if (my_prow == prow(istep-1, nblk, np_rows) .and. my_pcol == pcol(istep-1, nblk, np_cols)) then
         if (useGPU) then
           if (useIntelGPU) then
                       ! if (useIntelGPU) then
          ! needed at a later time when we can do explcit mem copys
          ! endif

           else
             !a_mat(l_rows,l_cols) = a_dev(l_rows,l_cols)
              a_offset = ((l_rows - 1) + matrixRows * (l_cols - 1)) * size_of_datatype

              successGPU = gpu_memcpy(int(loc(a_mat(l_rows, l_cols)),kind=c_intptr_t), a_dev + a_offset, &
                                      1 *  size_of_datatype, gpuMemcpyDeviceToHost)
              check_memcpy_gpu("tridiag: a_dev 3", successGPU)
           endif
         endif
         if (n_stored_vecs > 0) then
           a_mat(l_rows,l_cols) = a_mat(l_rows,l_cols) &
                       + dot_product(vu_stored_rows(l_rows,1:2*n_stored_vecs),uv_stored_cols(l_cols,1:2*n_stored_vecs))
         end if
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

         if (useGPU) then
           if (useIntelGPU) then
          ! if (useIntelGPU) then
          ! needed at a later time when we can expicit mem copy
          ! endif
           else
             !a_dev(l_rows,l_cols) = a_mat(l_rows,l_cols)
             !successGPU = gpu_threadsynchronize()
             !check_memcpy_gpu("tridiag: a_dev 4a5a", successGPU)

             successGPU = gpu_memcpy(a_dev + a_offset, int(loc(a_mat(l_rows, l_cols)),kind=c_intptr_t), &
                                     int(1 * size_of_datatype, kind=c_intptr_t), gpuMemcpyHostToDevice)
             check_memcpy_gpu("tridiag: a_dev 4", successGPU)
           endif
         endif
       endif

     enddo ! main cycle over istep=na,3,-1

#if COMPLEXCASE == 1
     ! Store e_vec(1) and d_vec(1)

     if (my_pcol==pcol(2, nblk, np_cols)) then
      if (my_prow==prow(1, nblk, np_rows)) then
       ! We use last l_cols value of loop above
       if (useGPU) then
         if (useIntelGPU) then
            vrl = a_mat(1,l_cols)
         else
           successGPU = gpu_memcpy(int(loc(aux3(1)),kind=c_intptr_t), a_dev + (matrixRows * (l_cols - 1)) * size_of_datatype, &
                                   1 * size_of_datatype, gpuMemcpyDeviceToHost)
           check_memcpy_gpu("tridiag: a_dev 5", successGPU)
           vrl = aux3(1)
         endif
       else !useGPU
         vrl = a_mat(1,l_cols)
       endif !useGPU
       call hh_transform_complex_&
       &PRECISION &
       (obj, vrl, 0.0_rk, xf, tau(2), wantDebug)
#if REALCASE == 1
       e_vec(1) = vrl
#endif
#if COMPLEXCASE == 1
       e_vec(1) = real(vrl,kind=rk)
#endif
       a_mat(1,l_cols) = 1. ! for consistency only
     endif
#ifdef WITH_MPI
     if (wantDebug) call obj%timer%start("mpi_communication")
     call mpi_bcast(tau(2), 1_MPI_KIND, MPI_COMPLEX_PRECISION, int(prow(1, nblk, np_rows),kind=MPI_KIND), &
                   int(mpi_comm_rows,kind=MPI_KIND), mpierr)
     if (wantDebug) call obj%timer%stop("mpi_communication")

#endif /* WITH_MPI */
   endif

#ifdef WITH_MPI
   if (wantDebug) call obj%timer%start("mpi_communication")
   call mpi_bcast(tau(2), 1_MPI_KIND, MPI_COMPLEX_PRECISION, int(pcol(2, nblk, np_cols),kind=MPI_KIND), &
                  int(mpi_comm_cols,kind=MPI_KIND), mpierr)
   if (wantDebug) call obj%timer%stop("mpi_communication")

#endif /* WITH_MPI */
  if (my_prow == prow(1, nblk, np_rows) .and. my_pcol == pcol(1, nblk, np_cols))  then
    if (useGPU) then
      if (useIntelGPU) then
        d_vec(1) = PRECISION_REAL(a_mat(1,1))
      else
        successGPU = gpu_memcpy(int(loc(aux3(1)),kind=c_intptr_t), a_dev, &
                               1 * size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("tridiag: a_dev 6", successGPU)
        d_vec(1) = PRECISION_REAL(aux3(1))
      endif
    else !useGPU
      d_vec(1) = PRECISION_REAL(a_mat(1,1))
    endif !useGPU
  endif

#endif /* COMPLEXCASE == 1 */

#if REALCASE == 1
  ! Store e_vec(1)

  if (my_prow==prow(1, nblk, np_rows) .and. my_pcol==pcol(2, nblk, np_cols)) then
    if (useGPU) then
      if (useIntelGPU) then
        e_vec(1) = a_mat(1,l_cols) ! use last l_cols value of loop above
      else
        successGPU = gpu_memcpy(int(loc(e_vec(1)),kind=c_intptr_t), a_dev + (matrixRows * (l_cols - 1)) * size_of_datatype, &
                                1 * size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("tridiag: a_dev 7", successGPU)
      endif
    else !useGPU
      e_vec(1) = a_mat(1,l_cols) ! use last l_cols value of loop above
    endif !useGPU
  endif

  ! Store d_vec(1)
  if (my_prow==prow(1, nblk, np_rows) .and. my_pcol==pcol(1, nblk, np_cols)) then
    if(useGPU) then
      if (useIntelGPU) then
        if (isSkewsymmetric) then
          d_vec(1) = 0.0_rk
        else
          d_vec(1) = a_mat(1,1)
        endif
      else
        successGPU = gpu_memcpy(int(loc(d_vec(1)),kind=c_intptr_t), a_dev, 1 * size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("tridiag: a_dev 8", successGPU)
      endif
    else !useGPU
      if (isSkewsymmetric) then
        d_vec(1) = 0.0_rk
      else
        d_vec(1) = a_mat(1,1)
      endif
    endif !useGPU
  endif
#endif

  deallocate(tmp, stat=istat, errmsg=errorMessage)
  check_deallocate("tridiag: tmp", istat, errorMessage)

  if (useGPU .and. .not.(useIntelGPU)) then
    ! todo: should we leave a_mat on the device for further use?
    successGPU = gpu_free(a_dev)
    check_dealloc_gpu("tridiag: a_dev 9", successGPU)

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
  endif
  ! if (useIntelGPU) then
  ! needed at a later time when we can do explicit frees
  ! endif


  ! distribute the arrays d_vec and e_vec to all processors

  allocate(tmp_real(na), stat=istat, errmsg=errorMessage)
  check_allocate("tridiag: tmp_real", istat, errorMessage)

#ifdef WITH_MPI
  if (wantDebug) call obj%timer%start("mpi_communication")
  tmp_real = d_vec
  call mpi_allreduce(tmp_real, d_vec, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_SUM, &
                     int(mpi_comm_rows,kind=MPI_KIND), mpierr)
  tmp_real = d_vec
  call mpi_allreduce(tmp_real, d_vec, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_SUM, &
                     int(mpi_comm_cols,kind=MPI_KIND), mpierr)
  tmp_real = e_vec
  call mpi_allreduce(tmp_real, e_vec, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_SUM, &
                     int(mpi_comm_rows,kind=MPI_KIND), mpierr)
  tmp_real = e_vec
  call mpi_allreduce(tmp_real, e_vec, int(na,kind=MPI_KIND), MPI_REAL_PRECISION, MPI_SUM, &
                     int(mpi_comm_cols,kind=MPI_KIND), mpierr)
  if (wantDebug) call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */

  deallocate(tmp_real, stat=istat, errmsg=errorMessage)
  check_deallocate("tridiag: tmp_real", istat, errorMessage)

  if (useGPU) then
    if (useIntelGPU) then
           deallocate(v_row, v_col, u_row, u_col, stat=istat, errmsg=errorMessage)
     check_deallocate("tridiag: v_row, v_col, u_row, u_col", istat, errorMessage)
    else
      successGPU = gpu_host_unregister(int(loc(a_mat),kind=c_intptr_t))
      check_host_unregister_gpu("tridiag: a_mat", successGPU)

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

      successGPU = gpu_host_unregister(int(loc(uv_stored_cols),kind=c_intptr_t))
      check_host_unregister_gpu("tridiag: uv_stored_cols", successGPU)

      successGPU = gpu_host_unregister(int(loc(vu_stored_rows),kind=c_intptr_t))
      check_host_unregister_gpu("tridiag: vu_stored_rows", successGPU)

      successGPU = gpu_host_unregister(int(loc(e_vec),kind=c_intptr_t))
      check_host_unregister_gpu("tridiag: e_vec", successGPU)

      successGPU = gpu_host_unregister(int(loc(d_vec),kind=c_intptr_t))
      check_host_unregister_gpu("tridiag: d_vec", successGPU)
    endif
  else
    deallocate(v_row, v_col, u_row, u_col, stat=istat, errmsg=errorMessage)
    check_deallocate("tridiag: v_row, v_col, u_row, u_col", istat, errorMessage)
  endif

  deallocate(vu_stored_rows, uv_stored_cols, stat=istat, errmsg=errorMessage)
  check_deallocate("tridiag: vu_stored_rows, uv_stored_cols", istat, errorMessage)

  call obj%timer%stop("tridiag_&
  &MATH_DATATYPE&
  &" // &
  PRECISION_SUFFIX // &
  gpuString )

end subroutine tridiag_&
&MATH_DATATYPE&
&_&
&PRECISION
