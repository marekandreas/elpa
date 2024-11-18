!
!    Copyright 2017, L. Hüdepohl and A. Marek, MPCDF
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
!

! PETERDEBUG: update the text below after benchmarks
! using elpa internal Hermitian multiply is faster then scalapack multiply, but we need an extra
! temporary matrix.
! using cannon algorithm should be the fastest. After this is verified, the other options should be removed
! however, we need the extra temporary matrix as well.

#include "general/error_checking.inc"

#ifdef DEVICE_POINTER
subroutine elpa_transform_generalized_d_ptr_&
          &ELPA_IMPL_SUFFIX&
          &(self, aDev, bDev, is_already_decomposed, error)
#else
subroutine elpa_transform_generalized_a_h_a_&
          &ELPA_IMPL_SUFFIX&
          &(self, a, b, is_already_decomposed, error)
#endif

  use precision
  use elpa_utilities, only : error_unit
  use mod_query_gpu_usage
  use elpa_gpu
#if defined (WITH_NVIDIA_GPU_VERSION) && defined (WITH_NVTX)
  use cuda_functions ! for NVTX labels
#endif
  implicit none
#include "general/precision_kinds.F90"
  class(elpa_impl_t)  :: self

#ifdef DEVICE_POINTER
  type(c_ptr)              :: aDev, bDev, tmpDev
  MATH_DATATYPE(kind=rck)  :: a(self%local_nrows, self%local_ncols), b(self%local_nrows, self%local_ncols) ! dummy variables
#else /* DEVICE_POINTER */  
#ifdef USE_ASSUMED_SIZE
  MATH_DATATYPE(kind=rck)  :: a(self%local_nrows, *), b(self%local_nrows, *)
#else
  MATH_DATATYPE(kind=rck)  :: a(self%local_nrows, self%local_ncols), b(self%local_nrows, self%local_ncols)
#endif
#endif /* DEVICE_POINTER */

  logical                  :: is_already_decomposed
  integer(kind=ik)         :: error
  
  integer(kind=ik)         :: istat
  character(200)           :: errorMessage
  integer                  :: sc_desc(SC_DESC_LEN)
  integer(kind=ik)         :: myid, my_prow, my_pcol, np_rows, np_cols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
  integer(kind=ik)         :: i, j
  integer(kind=MPI_KIND)   :: my_pMPI, my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI, mpierr
  integer(kind=ik)         :: BuffLevelInt
  integer(kind=c_int)      :: cannon_for_generalized, pxgemm_for_generalized, pxtrmm_for_generalized, debug, gpu_cannon

  logical                  :: useGPU, successGPU
  integer(kind=c_intptr_t) :: gpublasHandle
  integer(kind=c_intptr_t), parameter ::  size_of_datatype = size_of_&
                                        &PRECISION&
                                        &_&
                                        &MATH_DATATYPE

  MATH_DATATYPE(kind=rck), allocatable :: tmp(:,:)
  logical, save            :: firstCall = .true.

  call self%get("mpi_comm_rows"  , mpi_comm_rows, error)
  call self%get("mpi_comm_cols"  , mpi_comm_cols, error)
  call self%get("mpi_comm_parent", mpi_comm_all , error)

  call mpi_comm_rank(int(mpi_comm_all ,kind=MPI_KIND), my_pMPI   , mpierr)
  call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND), my_prowMPI, mpierr)
  call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)
  call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND), my_pcolMPI, mpierr)
  call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND), np_colsMPI, mpierr)

  myid    = int(my_pMPI   , kind=c_int)
  my_prow = int(my_prowMPI, kind=c_int)
  np_rows = int(np_rowsMPI, kind=c_int)
  my_pcol = int(my_pcolMPI, kind=c_int)
  np_cols = int(np_colsMPI, kind=c_int)

  call self%get("cannon_for_generalized", cannon_for_generalized, error)
  call self%get("pxtrmm_for_generalized", pxtrmm_for_generalized, error)
  call self%get("pxgemm_for_generalized", pxgemm_for_generalized, error)
  call self%get("debug", debug, error)

  useGPU = .false.
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
  if (.not.(query_gpu_usage(self, "Generalized_transform", useGPU))) then
    write(error_unit,*) "Generalized transform: Problem getting options for GPU. Aborting..."
    error = ELPA_ERROR
    return
  endif
#endif

  gpu_cannon = 0
  gpublasHandle=0
  if (useGPU) then
    call self%get("gpu_cannon", gpu_cannon, error)

    pxtrmm_for_generalized = 0 ! default for GPU
    if (self%is_set("pxtrmm_for_generalized") == 1) then ! if user enforces pxtrmm, use it
      call self%get("pxtrmm_for_generalized", pxtrmm_for_generalized, error)
    endif
  endif


#if !defined(WITH_MPI)
  if (cannon_for_generalized==1 .and. (myid == 0) .and. firstCall) then
    write(error_uni,*) "Cannons algorithm can only be used with MPI. Switching it off."
    firstCall = .false.
  end if
  cannon_for_generalized = 0
#endif

if (mod(np_cols, np_rows) /= 0) then
  if (cannon_for_generalized==1 .and. (myid == 0) .and. firstCall) then
    write(error_unit,*) "To use Cannons algorithm, np_cols must be a multiple of np_rows. Switching it off."
    firstCall = .false.
  endif
  cannon_for_generalized = 0
endif

#ifdef DEVICE_POINTER
  if (.not. useGPU) then
    write(error_unit,*) "Device pointer interface is not available if gpu keywords weren't switched on. Aborting..."
    error = ELPA_ERROR
    return
  endif

  if (pxgemm_for_generalized /= 1) then
    write(error_unit,*) "Device pointer interface is available only for pxgemm_for_generalized=1. Switching it on."
    pxgemm_for_generalized = 1
  endif
#endif

  call self%timer_start("transform_generalized()")

#ifdef WITH_NVTX
  call nvtxRangePush("transform_generalized")
#endif

  if (.not. is_already_decomposed) then
#ifdef WITH_NVTX
    call nvtxRangePush("cholesky: B = U^T*U, B <- U")
#endif

    ! B = U^T*U, B <- U
#ifdef DEVICE_POINTER
    call self%elpa_cholesky_d_ptr_&
        &ELPA_IMPL_SUFFIX&
        &(bDev, error)
#else
    call self%elpa_cholesky_a_h_a_&
        &ELPA_IMPL_SUFFIX&
        &(b, error)
#endif
    if(error .NE. ELPA_OK) return

#ifdef WITH_NVTX
    call nvtxRangePop() ! cholesky: B = U^T*U, B <- U
#endif

#ifdef WITH_NVTX
    call nvtxRangePush("invert_trm: B <- inv(U)")
#endif

    ! B <- inv(U)
#ifdef DEVICE_POINTER
    call self%elpa_invert_trm_d_ptr_&
        &ELPA_IMPL_SUFFIX&
        &(bDev, error)
#else
    call self%elpa_invert_trm_a_h_a_&
        &ELPA_IMPL_SUFFIX&
        &(b, error)
#endif
    if(error .NE. ELPA_OK) return

#ifdef WITH_NVTX
    call nvtxRangePop() ! invert_trm: B <- inv(U)
#endif

  endif ! (.not. is_already_decomposed)

  if (pxgemm_for_generalized == 1) then
#ifdef DEVICE_POINTER
    successGPU = gpu_malloc(tmpDev, self%local_nrows*self%local_ncols*size_of_datatype)
    check_alloc_gpu("elpa_transform_generalized: tmpDev", successGPU)
#else
    allocate(tmp(self%local_nrows, self%local_ncols), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_transform_generalized: tmp", istat, errorMessage)
#endif

    ! tmp <- B^T * A = inv(U^T) * A (we have to use temporary variable)
#ifdef DEVICE_POINTER
    ! PETERDEBUG: use qDev (yet empty) for tmpDev here, and use aDev (already destroyed) for tmpDev in backtransform
    call self%elpa_pxgemm_multiply_d_ptr_&
          &ELPA_IMPL_SUFFIX&
          &(BLAS_TRANS_OR_CONJ,'N', self%na, bDev, aDev, self%local_nrows, self%local_ncols, tmpDev, &
                                    self%local_nrows, self%local_ncols, error)
#else
    call self%elpa_pxgemm_multiply_a_h_a_&
          &ELPA_IMPL_SUFFIX&
          &(BLAS_TRANS_OR_CONJ,'N', self%na, b, a, self%local_nrows, self%local_ncols, tmp, &
                                    self%local_nrows, self%local_ncols, error)
#endif
    if(error .NE. ELPA_OK) return

    ! A <- tmp * inv(U) = inv(U)^T * A * inv(U)
#ifdef DEVICE_POINTER
    call self%elpa_pxgemm_multiply_d_ptr_&
          &ELPA_IMPL_SUFFIX&
          &('N','N', self%na, tmpDev, bDev, self%local_nrows, self%local_ncols, aDev, &
          self%local_nrows, self%local_ncols, error)
#else
    call self%elpa_pxgemm_multiply_a_h_a_&
          &ELPA_IMPL_SUFFIX&
          &('N','N', self%na, tmp, b, self%local_nrows, self%local_ncols, a, &
          self%local_nrows, self%local_ncols, error)
#endif
    if(error .NE. ELPA_OK) return

#ifdef DEVICE_POINTER
    successGPU = gpu_free(tmpDev)
    check_dealloc_gpu("elpa_transform_generalized: tmpDev", successGPU)
#else
    deallocate(tmp, stat=istat, errmsg=errorMessage)
    call check_alloc("elpa_transform_generalized", "tmp", istat, errorMessage)
#endif
  else if (cannon_for_generalized == 1) then
    allocate(tmp(self%local_nrows, self%local_ncols), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_transform_generalized: tmp", istat, errorMessage)

    call self%get("cannon_buffer_size", BuffLevelInt, error)
    if (gpu_cannon==1 .and. BuffLevelInt>0) then
      write(error_unit,*) "Warning: cannon_buffer_size>0 is not supported with GPUs. Using cannon_buffer_size=0"
      BuffLevelInt = 0
    endif
    call self%timer_start("cannons_reduction")
    ! BEWARE! even though tmp is output from the routine, it has to be zero on input!
#ifdef WITH_NVTX
    call nvtxRangePush("tmp = 0")
#endif
    tmp = 0.0_rck
#ifdef WITH_NVTX
    call nvtxRangePop()
#endif
#ifdef WITH_MPI
#ifdef WITH_NVTX
    call nvtxRangePush("cannons_reduction")
#endif
    if (useGPU) then
      gpublasHandle = self%gpu_setup%gpublasHandleArray(0)
    endif 
    
    error = self%construct_scalapack_descriptor(sc_desc, .false.)
    if(error .NE. ELPA_OK) return

    call cannons_reduction_&
      &ELPA_IMPL_SUFFIX&
      &(a, b, self%local_nrows, self%local_ncols, &
        int(sc_desc,kind=BLAS_KIND), tmp, int(BuffLevelInt,kind=MPI_KIND),   &
        int(mpi_comm_rows,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), debug, gpu_cannon, gpublasHandle)
#ifdef WITH_NVTX
    call nvtxRangePop()
#endif
#endif /* WITH_MPI */
    call self%timer_stop("cannons_reduction")

    a(1:self%local_nrows, 1:self%local_ncols) = tmp(1:self%local_nrows, 1:self%local_ncols)

    deallocate(tmp, stat=istat, errmsg=errorMessage)
    call check_alloc("elpa_transform_generalized", "tmp", istat, errorMessage)
  else ! not ((pxgemm_for_generalized == 1) .or. (cannon_for_generalized == 1))
    allocate(tmp(self%local_nrows, self%local_ncols), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_transform_generalized: tmp", istat, errorMessage)

    ! tmp <- B^T * A = inv(U^T) * A (we have to use temporary variable)
    call self%elpa_hermitian_multiply_a_h_a_&
        &ELPA_IMPL_SUFFIX&
        &('U','F', self%na, b, a, self%local_nrows, self%local_ncols, tmp, &
                              self%local_nrows, self%local_ncols, error)
    if(error .NE. ELPA_OK) return

    ! A <- tmp * inv(U) = inv(U)^T * A * inv(U)
    if (pxtrmm_for_generalized == 1) then ! CPU-only codepath
      ! A <- inv(U)^T * A
#ifdef WITH_NVTX
      call nvtxRangePush("copy: tmp -> a")
#endif
      a(1:self%local_nrows, 1:self%local_ncols) = tmp(1:self%local_nrows, 1:self%local_ncols)
#ifdef WITH_NVTX
      call nvtxRangePop()
#endif

      error = self%construct_scalapack_descriptor(sc_desc, .false.)
      if(error .NE. ELPA_OK) return

      call self%timer_start("scalapack multiply A * inv(U)")
#ifdef WITH_NVTX
      call nvtxRangePush("scalapack multiply A * inv(U)")
#endif
#ifdef WITH_MPI
      call p&
            &BLAS_CHAR&
            &trmm("R", "U", "N", "N", int(self%na,kind=BLAS_KIND), int(self%na,kind=BLAS_KIND), &
                  ONE, b, 1_BLAS_KIND, 1_BLAS_KIND, int(sc_desc,kind=BLAS_KIND), &
                       a, 1_BLAS_KIND, 1_BLAS_KIND, int(sc_desc,kind=BLAS_KIND))
#else /* WITH_MPI */
      call BLAS_CHAR&
            &trmm("R", "U", "N", "N", int(self%na,kind=BLAS_KIND), int(self%na,kind=BLAS_KIND), &
                  ONE, b, int(self%na,kind=BLAS_KIND), a, int(self%na,kind=BLAS_KIND))
#endif /* WITH_MPI */

#ifdef WITH_NVTX
      call nvtxRangePop()
#endif
      call self%timer_stop("scalapack multiply A * inv(U)")
    
    else ! (pxtrmm_for_generalized == 1)

      ! A <- tmp^T
      call self%timer_start("PxTRAN")
#ifdef WITH_MPI
      call p&
            &BLAS_CHAR&
#if REALCASE == 1
            &tran&
#else
            &tranc&
#endif
            &(self%na, self%na, ONE , tmp, 1_BLAS_KIND, 1_BLAS_KIND, int(sc_desc,kind=BLAS_KIND), &
                                ZERO,   a, 1_BLAS_KIND, 1_BLAS_KIND, int(sc_desc,kind=BLAS_KIND))
#else /* WITH_MPI */
#if REALCASE == 1
      a(1:self%na, 1:self%na) =       transpose(tmp(1:self%na, 1:self%na))
#else
      a(1:self%na, 1:self%na) = conjg(transpose(tmp(1:self%na, 1:self%na)))
#endif
#endif /* WITH_MPI */
      call self%timer_stop("PxTRAN")
      
      ! tmp <- A^T * inv(U) = (tmp^T)^T * inv(U)
      call self%elpa_hermitian_multiply_a_h_a_&
            &ELPA_IMPL_SUFFIX&
            &('F','F', self%na, a, b, self%local_nrows, self%local_ncols, tmp, &
            self%local_nrows, self%local_ncols, error)
      if(error .NE. ELPA_OK) return

      ! a <- tmp
      call self%timer_start("copy")
      a(1:self%local_nrows, 1:self%local_ncols) = tmp(1:self%local_nrows, 1:self%local_ncols)
      call self%timer_stop("copy")

    endif ! (pxtrmm_for_generalized == 1)

    deallocate(tmp, stat=istat, errmsg=errorMessage)
    call check_alloc("elpa_transform_generalized", "tmp", istat, errorMessage)
  endif ! (pxgemm_for_generalized == 1) .or. (cannon_for_generalized == 1) .or. else

#ifdef WITH_NVTX
  call nvtxRangePop() ! transform_generalized
#endif

  call self%timer_stop("transform_generalized()")
end subroutine

! _________________________________________________________________________________________________________________________________

#ifdef DEVICE_POINTER
subroutine elpa_transform_back_generalized_d_ptr_&
            &ELPA_IMPL_SUFFIX&
            &(self, bDev, qDev, error)
#else
subroutine elpa_transform_back_generalized_a_h_a_&
            &ELPA_IMPL_SUFFIX&
            &(self, b, q, error)
#endif
  use mod_query_gpu_usage
  use elpa_utilities , only : check_alloc, check_allocate_f, error_unit
  use elpa_gpu
#ifdef WITH_NVIDIA_GPU_VERSION
  use cuda_functions ! for NVTX labels
#endif
  implicit none
#include "general/precision_kinds.F90"
  class(elpa_impl_t)  :: self

#ifdef DEVICE_POINTER
  type(c_ptr)              :: bDev, qDev, tmpDev
  MATH_DATATYPE(kind=rck),  allocatable :: b(:,:), q(:,:) ! dummy variables
#else /* DEVICE_POINTER */
#ifdef USE_ASSUMED_SIZE
  MATH_DATATYPE(kind=rck)  :: b(self%local_nrows, *), q(self%local_nrows, *)
#else
  MATH_DATATYPE(kind=rck)  :: b(self%local_nrows, self%local_ncols), q(self%local_nrows, self%local_ncols)
#endif
#endif /* DEVICE_POINTER */

  integer                  :: error
  
  integer(kind=ik)         :: istat
  character(200)           :: errorMessage
  integer(kind=ik)         :: myid, my_prow, my_pcol, np_rows, np_cols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
  integer(kind=ik)         :: i, j
  integer(kind=MPI_KIND)   :: mpierr, my_pMPI, my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI
  integer                  :: sc_desc(SC_DESC_LEN)
  integer                  :: sc_desc_ev(SC_DESC_LEN)
  integer(kind=c_int)      :: cannon_for_generalized, pxgemm_for_generalized, pxtrmm_for_generalized, debug, gpu_cannon
  
  logical                  :: useGPU, successGPU
  integer(kind=c_intptr_t) :: gpublasHandle
  integer(kind=c_intptr_t), parameter ::  size_of_datatype = size_of_&
                                          &PRECISION&
                                          &_&
                                          &MATH_DATATYPE

  MATH_DATATYPE(kind=rck), allocatable :: tmp(:,:)
  MATH_DATATYPE(kind=rck), allocatable :: bt(:,:)

  call self%get("mpi_comm_rows",mpi_comm_rows,error)
  call self%get("mpi_comm_cols",mpi_comm_cols,error)
  call self%get("mpi_comm_parent", mpi_comm_all,error)

  call mpi_comm_rank(int(mpi_comm_all,kind=MPI_KIND), my_pMPI,mpierr)
  call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND),my_prowMPI,mpierr)
  call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND),np_rowsMPI,mpierr)
  call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND),my_pcolMPI,mpierr)
  call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND),np_colsMPI,mpierr)

  myid = int(my_pMPI,kind=c_int)
  my_prow = int(my_prowMPI,kind=c_int)
  np_rows = int(np_rowsMPI,kind=c_int)
  my_pcol = int(my_pcolMPI,kind=c_int)
  np_cols = int(np_colsMPI,kind=c_int)

  call self%get("cannon_for_generalized", cannon_for_generalized, error)
  call self%get("pxgemm_for_generalized", pxgemm_for_generalized, error)
  call self%get("pxtrmm_for_generalized", pxtrmm_for_generalized, error)
  call self%get("debug", debug, error)
  
  useGPU = .false.
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
  if (.not.(query_gpu_usage(self, "elpa_transform_back_generalized", useGPU))) then
    write(error_unit,*) "elpa_transform_back_generalized: Problem getting options for GPU. Aborting..."
    error = ELPA_ERROR
    return
  endif
#endif

  gpu_cannon = 0
  gpublasHandle=0
  if (useGPU) then
    call self%get("gpu_cannon", gpu_cannon, error)

    pxtrmm_for_generalized = 0 ! default for GPU
    if (self%is_set("pxtrmm_for_generalized") == 1) then ! if user enforces pxtrmm, use it
      call self%get("pxtrmm_for_generalized", pxtrmm_for_generalized, error)
    endif
  endif

#if !defined(WITH_MPI)
  cannon_for_generalized = 0
#endif

  if (mod(np_cols, np_rows) /= 0) then
    cannon_for_generalized = 0
  endif

  error = self%construct_scalapack_descriptor(sc_desc, .false.)
  error = self%construct_scalapack_descriptor(sc_desc_ev, .true.)
  if(error .NE. ELPA_OK) return

#ifdef DEVICE_POINTER
  if (.not. useGPU) then
    write(*,*) "Device pointer interface is not available if gpu keywords weren't switched on. Aborting..."
    error = ELPA_ERROR
    return
  endif

  if (pxgemm_for_generalized /= 1) then
    write(*,*) "Device pointer interface is available only for pxgemm_for_generalized=1. Switching it on."
    pxgemm_for_generalized = 1
  endif
#endif

  call self%timer_start("transform_back_generalized()")

#ifdef WITH_NVTX
  call nvtxRangePush("transform_back_generalized") ! PETERDEBUG: change to short notation
#endif

  if (pxgemm_for_generalized == 1) then
#ifdef DEVICE_POINTER
    successGPU = gpu_malloc(tmpDev, self%local_nrows*self%local_ncols*size_of_datatype)
    check_alloc_gpu("elpa_transform_back_generalized: tmpDev", successGPU)
#else
    allocate(tmp(self%local_nrows, self%local_ncols), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_transform_back_generalized: tmp", istat, errorMessage)
#endif

    ! tmp <- b Q = inv(U) Q
#ifdef DEVICE_POINTER
    call self%elpa_pxgemm_multiply_d_ptr_&
          &ELPA_IMPL_SUFFIX&
          &('N','N', self%na, bDev, qDev, self%local_nrows, self%local_ncols, tmpDev, &
          self%local_nrows, self%local_ncols, error)
#else
    call self%elpa_pxgemm_multiply_a_h_a_&
          &ELPA_IMPL_SUFFIX&
          &('N','N', self%na, b, q, self%local_nrows, self%local_ncols, tmp, &
          self%local_nrows, self%local_ncols, error)
#endif
    if(error .NE. ELPA_OK) return

    ! q <- tmp
    call self%timer_start("copy")
#ifdef DEVICE_POINTER
    successGPU = gpu_memcpy(qDev, tmpDev, self%local_nrows*self%local_ncols*size_of_datatype, gpuMemcpyDeviceToDevice)
    check_memcpy_gpu("tridiag qDev<-tmpDev", successGPU)
#else
    q(1:self%local_nrows, 1:self%local_ncols) = tmp(1:self%local_nrows, 1:self%local_ncols)
#endif
    call self%timer_stop("copy")

#ifdef DEVICE_POINTER
    successGPU = gpu_free(tmpDev)
    check_dealloc_gpu("elpa_transform_back_generalized: tmpDev", successGPU)
#else
    deallocate(tmp, stat=istat, errmsg=errorMessage)
    call check_alloc("elpa_transform_back_generalized", "tmp", istat, errorMessage)
#endif

  else if (cannon_for_generalized == 1) then
    call self%timer_start("cannons_triang_rectangular")
#ifdef WITH_NVTX
    call nvtxRangePush("cannons_triang_rectangular")
#endif
#ifdef WITH_MPI
    if (useGPU) then
      gpublasHandle = self%gpu_setup%gpublasHandleArray(0)
    endif

    allocate(tmp(self%local_nrows, self%local_ncols), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_transform_back_generalized: tmp", istat, errorMessage)

    call cannons_triang_rectangular_&
      &ELPA_IMPL_SUFFIX&
      &(b, q, self%local_nrows, self%local_ncols, &
        int(sc_desc,kind=BLAS_KIND), int(sc_desc_ev,kind=BLAS_KIND), tmp,  &
        int(mpi_comm_rows,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), debug, gpu_cannon, gpublasHandle)
#endif
#ifdef WITH_NVTX
    call nvtxRangePop()
#endif
    call self%timer_stop("cannons_triang_rectangular")

    q(1:self%local_nrows, 1:self%local_ncols) = tmp(1:self%local_nrows, 1:self%local_ncols)

    deallocate(tmp, stat=istat, errmsg=errorMessage)
    call check_alloc("elpa_transform_back_generalized", "tmp", istat, errorMessage)
  else
    if (pxtrmm_for_generalized == 1) then
      call self%timer_start("scalapack multiply inv(U) * Q")
#ifdef WITH_NVTX
      call nvtxRangePush("scalapack multiply: Q <- inv(U) * Q")
#endif

#ifdef WITH_MPI
      ! Q <- inv(U) * Q
      call p&
          &BLAS_CHAR&
          &trmm("L", "U", "N", "N", int(self%na,kind=BLAS_KIND), int(self%nev,kind=BLAS_KIND), &
                ONE, b, 1_BLAS_KIND, 1_BLAS_KIND, int(sc_desc,kind=BLAS_KIND),  &
                q, 1_BLAS_KIND, 1_BLAS_KIND, int(sc_desc,kind=BLAS_KIND))
#else
      call BLAS_CHAR&
          &trmm("L", "U", "N", "N", int(self%na,kind=BLAS_KIND), int(self%nev,kind=BLAS_KIND), &
                ONE, b, int(self%na,kind=BLAS_KIND), q, int(self%na,kind=BLAS_KIND))
#endif

#ifdef WITH_NVTX
      call nvtxRangePop()
#endif  
      call self%timer_stop("scalapack multiply inv(U) * Q")
    
    else ! (pxtrmm_for_generalized == 1)
      
      ! two additional temp arrays bt amd temp are needed, since we can't modify b: it might be used later if(is_already_decomposed)
      allocate(tmp(self%local_nrows, self%local_ncols), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_transform_back_generalized: tmp", istat, errorMessage)
      
      allocate(bt(self%local_nrows, self%local_ncols), stat=istat, errmsg=errorMessage)
      check_allocate("elpa_transform_back_generalized: bt", istat, errorMessage)

      ! bt <- b^T
      call self%timer_start("PxTRAN")
#ifdef WITH_MPI
      call p&
            &BLAS_CHAR&
#if REALCASE == 1
            &tran&
#else
            &tranc&
#endif
            &(self%na, self%na, ONE , b , 1_BLAS_KIND, 1_BLAS_KIND, int(sc_desc,kind=BLAS_KIND), &
                                ZERO, bt, 1_BLAS_KIND, 1_BLAS_KIND, int(sc_desc,kind=BLAS_KIND))
#else /* WITH_MPI */
#if REALCASE == 1
      bt(1:self%na, 1:self%na) =       transpose(b(1:self%na, 1:self%na))
#else
      bt(1:self%na, 1:self%na) = conjg(transpose(b(1:self%na, 1:self%na)))
#endif
#endif /* WITH_MPI */
      call self%timer_stop("PxTRAN")
      
      ! tmp <- bt^T Q = inv(U) Q
      call self%elpa_hermitian_multiply_a_h_a_&
            &ELPA_IMPL_SUFFIX&
            &('F','F', self%na, bt, q, self%local_nrows, self%local_ncols, tmp, &
            self%local_nrows, self%local_ncols, error)
      if(error .NE. ELPA_OK) return

      ! q <- tmp
      call self%timer_start("copy")
      q(1:self%local_nrows, 1:self%local_ncols) = tmp(1:self%local_nrows, 1:self%local_ncols)
      call self%timer_stop("copy")

      deallocate(tmp, stat=istat, errmsg=errorMessage)
      call check_alloc("elpa_transform_back_generalized", "tmp", istat, errorMessage)

      deallocate(bt, stat=istat, errmsg=errorMessage)
      call check_alloc("elpa_transform_back_generalized", "bt", istat, errorMessage)
    endif ! (pxtrmm_for_generalized == 1)
  endif

#ifdef WITH_NVTX
  call nvtxRangePop() ! transform_back_generalized
#endif
  call self%timer_stop("transform_back_generalized()")

end subroutine

