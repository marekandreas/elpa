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
#include "../general/error_checking.inc"

#ifdef DEVICE_POINTER
#ifdef ACTIVATE_SKEW
function elpa_solve_skew_evp_&
         &MATH_DATATYPE&
   &_1stage_d_ptr_&
   &PRECISION&
   &_impl (obj, &
#else /* ACTIVATE_SKEW */
function elpa_solve_evp_&
         &MATH_DATATYPE&
   &_1stage_d_ptr_&
   &PRECISION&
   &_impl (obj, &
#endif /* ACTIVATE_SKEW */
   aExtern, &
   evExtern, &
   qExtern) result(success)
#else /* DEVICE_POINTER */

#ifdef ACTIVATE_SKEW
function elpa_solve_skew_evp_&
         &MATH_DATATYPE&
   &_1stage_a_h_a_&
   &PRECISION&
   &_impl (obj, &
#else /* ACTIVATE_SKEW */
function elpa_solve_evp_&
         &MATH_DATATYPE&
   &_1stage_a_h_a_&
   &PRECISION&
   &_impl (obj, &
#endif /* ACTIVATE_SKEW */
   aExtern, &
   evExtern, &
   qExtern) result(success)

#endif /* DEVICE_POINTER */

   use precision
#ifdef WITH_NVIDIA_GPU_VERSION
   use cuda_functions
#endif
#ifdef WITH_AMD_GPU_VERSION
   use hip_functions
#endif
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
   use openmp_offload_functions
#endif
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
   use elpa_gpu
   use mod_check_for_gpu
#endif
   use, intrinsic :: iso_c_binding
   use elpa_abstract_impl
   use elpa_mpi
   use elpa1_compute
   use elpa_omp
#ifdef REDISTRIBUTE_MATRIX
   use elpa_scalapack_interfaces
#endif
   use solve_tridi
#ifdef HAVE_AFFINITY_CHECKING
   use thread_affinity
#endif
   use elpa_utilities, only : error_unit

   use mod_query_gpu_usage

   implicit none
#include "../general/precision_kinds.F90"
   class(elpa_abstract_impl_t), intent(inout)                         :: obj
#ifdef DEVICE_POINTER
   type(c_ptr)                                                        :: evExtern
#else
   real(kind=REAL_DATATYPE), target, intent(out)                      :: evExtern(obj%na)
#endif

   real(kind=REAL_DATATYPE), pointer                                  :: ev(:)

#ifdef DEVICE_POINTER

!#ifdef REDISTRIBUTE_MATRIX
   type(c_ptr)                                                         :: aExtern
   type(c_ptr), optional                                             :: qExtern
!#else /* REDISTRIBUTE_MATRIX */
!   type(c_ptr)                                                        :: a, q
!#endif /* REDISTRIBUTE_MATRIX */

#else /* DEVICE_POINTER */

!#ifdef REDISTRIBUTE_MATRIX

#ifdef USE_ASSUMED_SIZE
   MATH_DATATYPE(kind=rck), intent(inout), target                     :: aExtern(obj%local_nrows,*)
   MATH_DATATYPE(kind=rck), optional,target,intent(out)               :: qExtern(obj%local_nrows,*)
#else
   MATH_DATATYPE(kind=rck), intent(inout), target                     :: aExtern(1:obj%local_nrows,1:obj%local_ncols)
#ifdef ACTIVATE_SKEW
   MATH_DATATYPE(kind=C_DATATYPE_KIND), optional, target, intent(out) :: qExtern(1:obj%local_nrows,1:2*obj%local_ncols)
#else
   MATH_DATATYPE(kind=C_DATATYPE_KIND), optional, target, intent(out) :: qExtern(1:obj%local_nrows,1:obj%local_ncols)
#endif
#endif /* USE_ASSUMED_SIZE */

!#else /* REDISTRIBUTE_MATRIX */
!
!#ifdef USE_ASSUMED_SIZE
!   MATH_DATATYPE(kind=rck), intent(inout), target                     :: a(obj%local_nrows,*)
!   MATH_DATATYPE(kind=rck), optional,target,intent(out)               :: q(obj%local_nrows,*)
!#else
!   MATH_DATATYPE(kind=rck), intent(inout), target                     :: a(obj%local_nrows,obj%local_ncols)
!#ifdef ACTIVATE_SKEW
!   MATH_DATATYPE(kind=C_DATATYPE_KIND), optional, target, intent(out) :: q(obj%local_nrows,2*obj%local_ncols)
!#else
!   MATH_DATATYPE(kind=C_DATATYPE_KIND), optional, target, intent(out) :: q(obj%local_nrows,obj%local_ncols)
!#endif
!#endif /* USE_ASSUMED_SIZE */

!#endif /* REDISTRIBUTE_MATRIX */

#endif /* DEVICE_POINTER */

  MATH_DATATYPE(kind=rck), pointer                                   :: a(:,:)
  MATH_DATATYPE(kind=rck), pointer                                   :: q(:,:)

#if REALCASE == 1
   real(kind=C_DATATYPE_KIND), allocatable           :: tau(:)
   real(kind=C_DATATYPE_KIND), allocatable, target   :: q_dummy(:,:)
   real(kind=C_DATATYPE_KIND), pointer               :: q_actual(:,:)
#endif /* REALCASE */

#if COMPLEXCASE == 1
   real(kind=REAL_DATATYPE), allocatable             :: q_real(:,:)
   complex(kind=C_DATATYPE_KIND), allocatable        :: tau(:)
   complex(kind=C_DATATYPE_KIND), allocatable,target :: q_dummy(:,:)
   complex(kind=C_DATATYPE_KIND), pointer            :: q_actual(:,:)
#endif /* COMPLEXCASE */


   integer(kind=c_int)                             :: l_cols, l_rows, l_cols_nev, np_rows, np_cols
   integer(kind=MPI_KIND)                          :: np_rowsMPI, np_colsMPI

   logical                                         :: useGPU
   integer(kind=c_int)                             :: skewsymmetric
   logical                                         :: isSkewsymmetric
   logical                                         :: success

   logical                                         :: do_useGPU, do_useGPU_tridiag, &
                                                      do_useGPU_solve_tridi, do_useGPU_trans_ev
   integer(kind=ik)                                :: numberOfGPUDevices

   integer(kind=c_int)                             :: my_pe, n_pes, my_prow, my_pcol
   integer(kind=MPI_KIND)                          :: mpierr, my_peMPI, n_pesMPI, my_prowMPI, my_pcolMPI
   real(kind=C_DATATYPE_KIND), allocatable         :: e(:)
   logical                                         :: wantDebug
   integer(kind=c_int)                             :: istat, debug, gpu
   character(200)                                  :: errorMessage
   integer(kind=ik)                                :: na, nev, nblk, matrixCols, &
                                                      mpi_comm_rows, mpi_comm_cols,        &
                                                      mpi_comm_all, check_pd, i, error, matrixRows
   real(kind=C_DATATYPE_KIND)                      :: thres_pd

#ifdef REDISTRIBUTE_MATRIX
   integer(kind=ik)                                :: nblkInternal, matrixOrder
   character(len=1)                                :: layoutInternal, layoutExternal
   integer(kind=c_int)                             :: external_blacs_ctxt
   integer(kind=BLAS_KIND)                         :: external_blacs_ctxt_
   integer(kind=BLAS_KIND)                         :: np_rows_, np_cols_, my_prow_, my_pcol_
   integer(kind=BLAS_KIND)                         :: np_rows__, np_cols__, my_prow__, my_pcol__
   integer(kind=BLAS_KIND)                         :: sc_desc_(1:9), sc_desc(1:9)
   integer(kind=BLAS_KIND)                         :: na_rows_, na_cols_, info_, blacs_ctxt_
   integer(kind=ik)                                :: mpi_comm_rows_, mpi_comm_cols_
   integer(kind=MPI_KIND)                          :: mpi_comm_rowsMPI_, mpi_comm_colsMPI_
   character(len=1), parameter                     :: matrixLayouts(2) = [ 'C', 'R' ]

   MATH_DATATYPE(kind=rck), allocatable, target               :: aIntern(:,:)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), allocatable, target   :: qIntern(:,:)
   real(kind=REAL_DATATYPE), pointer                          :: evIntern(:)
#else
   MATH_DATATYPE(kind=rck), pointer                           :: aIntern(:,:)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer               :: qIntern(:,:)
   real(kind=REAL_DATATYPE), pointer                          :: evIntern(:)
#endif
   integer(kind=c_int)                             :: pinningInfo

   logical                                         :: do_tridiag, do_solve, do_trans_ev
   integer(kind=ik)                                :: nrThreads, limitThreads
   integer(kind=ik)                                :: global_index

   logical                                         :: reDistributeMatrix, doRedistributeMatrix
   integer(kind=ik)                                :: gpu_old, gpu_new

   logical                                         :: successGPU

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
   integer(kind=c_intptr_t), parameter             :: size_of_datatype = size_of_&
                                                                      &PRECISION&
                                                                      &_&
                                                                      &MATH_DATATYPE
   integer(kind=c_intptr_t), parameter             :: size_of_real_datatype = size_of_&
                                                                      &PRECISION&
                                                                      &_real
#endif
#ifdef WITH_GPU_STREAMS
   !integer(kind=c_intptr_t)                        :: num
#endif
   
#ifdef ACTIVATE_SKEW
   call obj%timer%start("elpa_solve_skew_evp_&
#else
   call obj%timer%start("elpa_solve_evp_&
#endif
   &MATH_DATATYPE&
   &_1stage_&
   &PRECISION&
   &")

   call obj%get("debug",debug, error)
   if (error .ne. ELPA_OK) then
     write(error_unit,*) "ELPA1: Problem getting option for debug settings. Aborting..."
#include "./elpa1_aborting_template.F90"
   endif

   wantDebug = debug == 1

#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
    ! check legacy GPU setings
#ifdef ACTIVATE_SKEW
    if (.not.(query_gpu_usage(obj, "ELPA1_SKEW", useGPU))) then
      call obj%timer%stop("elpa_solve_skew_evp_&
      &MATH_DATATYPE&
      &_1stage_&
      &PRECISION&
      &")
#else
    if (.not.(query_gpu_usage(obj, "ELPA1", useGPU))) then
      call obj%timer%stop("elpa_solve_evp_&
      &MATH_DATATYPE&
      &_1stage_&
      &PRECISION&
      &")
#endif
      write(error_unit,*) "ELPA1: Problem getting options for GPU. Aborting..."
#include "./elpa1_aborting_template.F90"      
    endif
#endif /* defined(WITH_NVIDIA_GPU_VERSION) ... */

    do_useGPU = .false.     

   call obj%get("mpi_comm_parent", mpi_comm_all, error)
   if (error .ne. ELPA_OK) then
     write(error_unit, *) "ELPA1: Problem getting mpi_comm_all. Aborting..."
#include "./elpa1_aborting_template.F90"
   endif

   call mpi_comm_rank(int(mpi_comm_all,kind=MPI_KIND), my_peMPI, mpierr)
   my_pe = int(my_peMPI,kind=c_int)

    ! openmp setting
#include "../helpers/elpa_openmp_settings_template.F90"


#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
   if (useGPU) then
     call obj%timer%start("check_for_gpu")

     if (check_for_gpu(obj, my_pe, numberOfGPUDevices, wantDebug=wantDebug)) then
       do_useGPU = .true.
       ! set the neccessary parameters
       call set_gpu_parameters()
     else
       write(error_unit, *) "GPUs are requested but not detected! Aborting..."
       call obj%timer%stop("check_for_gpu")
#include "./elpa1_aborting_template.F90"
     endif
     call obj%timer%stop("check_for_gpu")
   endif ! useGPU
#endif


   do_useGPU_tridiag = do_useGPU
   do_useGPU_solve_tridi = do_useGPU
   do_useGPU_trans_ev = do_useGPU
   ! only if we want (and can) use GPU in general, look what are the
   ! requirements for individual routines. Implicitly they are all set to 1, so
   ! unles specified otherwise by the user, GPU versions of all individual
   ! routines should be used
   if(do_useGPU) then
     call obj%get("gpu_tridiag", gpu, error)
     if (error .ne. ELPA_OK) then
       write(error_unit, *) "ELPA1: Problem getting option for gpu_tridiag. Aborting..."
#include "./elpa1_aborting_template.F90"
     endif
     do_useGPU_tridiag = (gpu == 1)

     call obj%get("gpu_solve_tridi", gpu, error)
     if (error .ne. ELPA_OK) then
       write(error_unit, *) "ELPA1: Problem getting option for gpu_solve_tridi. Aborting..."
#include "./elpa1_aborting_template.F90"
     endif
     do_useGPU_solve_tridi = (gpu == 1)

     call obj%get("gpu_trans_ev", gpu, error)
     if (error .ne. ELPA_OK) then
       write(error_unit, *) "ELPA1: Problem getting option for gpu_trans_ev. Aborting..."
#include "./elpa1_aborting_template.F90"
     endif
     do_useGPU_trans_ev = (gpu == 1)
   endif
   ! for elpa1 the easy thing is, that the individual phases of the algorithm
   ! do not share any data on the GPU.

   reDistributeMatrix = .false.

   na         = obj%na
   nev        = obj%nev
   matrixRows = obj%local_nrows
   nblk       = obj%nblk
   matrixCols = obj%local_ncols

   call obj%get("mpi_comm_rows",mpi_comm_rows,error)
   if (error .ne. ELPA_OK) then
     write(error_unit, *) "ELPA1: Problem getting mpi_comm_rows. Aborting..."
#include "./elpa1_aborting_template.F90"
   endif
   call obj%get("mpi_comm_cols",mpi_comm_cols,error)
   if (error .ne. ELPA_OK) then
     write(error_unit, *) "ELPA1 Problem getting mpi_comm_cols. Aborting..."
#include "./elpa1_aborting_template.F90"
   endif

#ifdef REDISTRIBUTE_MATRIX
#ifndef DEVICE_POINTER
   ! if a matrix redistribution is done then
   ! - aIntern, qIntern are getting allocated for the new distribution
   ! - nblk, matrixCols, matrixRows, mpi_comm_cols, mpi_comm_rows are getting updated
   ! TODO: make sure that nowhere in ELPA the communicators are getting "getted",
   ! and the variables obj%local_nrows,1:obj%local_ncols are being used
   ! - a points then to aIntern, q points to qIntern
#include "../helpers/elpa_redistribute_template.F90"
   ! ev still has to be assigned
#else
   ! at the moment no redistribute if dptr !!
#endif /* DEVICE_POINTER */
#endif /* REDISTRIBUTE_MATRIX */

#ifdef DEVICE_POINTER

#ifdef REDISTRIBUTE_MATRIX
   doRedistributeMatrix = .false.
! do the same as if not redistribute!!
   allocate(aIntern(1:matrixRows,1:matrixCols), stat=istat, errmsg=errorMessage)
   check_allocate("elpa1_template: aIntern", istat, errorMessage)

   a       => aIntern(1:matrixRows,1:matrixCols)

   allocate(evIntern(1:obj%na), stat=istat, errmsg=errorMessage)
   check_allocate("elpa1_template: evIntern", istat, errorMessage)
   ev      => evIntern(1:obj%na)

   if (present(qExtern)) then
#ifdef ACTIVATE_SKEW
     allocate(qIntern(1:matrixRows,1:2*matrixCols), stat=istat, errmsg=errorMessage)
     check_allocate("elpa1_template: qIntern", istat, errorMessage)
#else
     allocate(qIntern(1:matrixRows,1:matrixCols), stat=istat, errmsg=errorMessage)
     check_allocate("elpa1_template: qIntern", istat, errorMessage)
#endif
   endif

!   ! and associate pointer
!   ! no redistribution happend
!   a => aExtern(1:matrixRows,1:matrixCols)
!   if (present(qExtern)) then
!#ifdef ACTIVATE_SKEW
!     q => qExtern(1:matrixRows,1:2*matrixCols)
!#else
!     q => qExtern(1:matrixRows,1:matrixCols)
!#endif
!   endif

#else /* REDISTRIBUTE_MATRIX */

   allocate(aIntern(1:matrixRows,1:matrixCols), stat=istat, errmsg=errorMessage)
   check_allocate("elpa1_template: aIntern", istat, errorMessage)

   a       => aIntern(1:matrixRows,1:matrixCols)

   allocate(evIntern(1:obj%na), stat=istat, errmsg=errorMessage)
   check_allocate("elpa1_template: evIntern", istat, errorMessage)
   ev      => evIntern(1:obj%na)

   if (present(qExtern)) then
#ifdef ACTIVATE_SKEW
     allocate(qIntern(1:matrixRows,1:2*matrixCols), stat=istat, errmsg=errorMessage)
     check_allocate("elpa1_template: qIntern", istat, errorMessage)
#else
     allocate(qIntern(1:matrixRows,1:matrixCols), stat=istat, errmsg=errorMessage)
     check_allocate("elpa1_template: qIntern", istat, errorMessage)
#endif
   endif
#endif /* REDISTRIBUTE_MATRIX */


#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
   !TODO: intel gpu
   ! in case of devcice pointer _AND_ redistribute
   ! 1. copy aExtern to aIntern_dummy
   ! 2. redistribute aIntern_dummy to aIntern

   successGPU = gpu_memcpy(c_loc(aIntern(1,1)), aExtern, matrixRows*matrixCols*size_of_datatype, &
                             gpuMemcpyDeviceToHost)
   check_memcpy_gpu("elpa1: aExtern -> aIntern", successGPU)
#endif


#else /* DEVICE_POINTER */

#ifdef REDISTRIBUTE_MATRIX
  ! a and q point already to the allocated arrays aIntern, qIntern
  if (.not.(doRedistributeMatrix)) then
    ! no redistribution happend
    a => aExtern(1:matrixRows,1:matrixCols)
    if (present(qExtern)) then
#ifdef ACTIVATE_SKEW
      q => qExtern(1:matrixRows,1:2*matrixCols)
#else
      q => qExtern(1:matrixRows,1:matrixCols)
#endif
    endif
  endif
#else /* REDISTRIBUTE_MATRIX */
   ! aIntern, qIntern, are normally pointers since no matrix redistribution is used
   ! point them to  the external arrays
   aIntern => aExtern(1:matrixRows,1:matrixCols)
   if (present(qExtern)) then
#ifdef ACTIVATE_SKEW
     qIntern => qExtern(1:matrixRows,1:2*matrixCols)
#else
     qIntern => qExtern(1:matrixRows,1:matrixCols)
#endif
   endif
#endif /* REDISTRIBUTE_MATRIX */

  ! whether a matrix redistribution or not is happening we still
  ! have to point ev
   evIntern => evExtern(1:obj%na)

#endif /* DEVICE_POINTER */


#ifdef REDISTRIBUTE_MATRIX
   if (doRedistributeMatrix) then
#endif
     a       => aIntern(1:matrixRows,1:matrixCols)
     if (present(qExtern)) then
#ifdef ACTIVATE_SKEW
       q       => qIntern(1:matrixRows,1:2*matrixCols)
#else
       q       => qIntern(1:matrixRows,1:matrixCols)
#endif
     endif
#ifdef REDISTRIBUTE_MATRIX
   endif
#endif

   ev      => evIntern(1:obj%na)

#ifdef WITH_GPU_STREAMS
   !page lock ev (or evIntern, or evExtern)
   !num = (obj%na) * size_of_&
   !                 &PRECISION&
   !                 &_real
   !successGPU = gpu_host_register(int(loc(ev),kind=c_intptr_t), num,&
   !             gpuHostRegisterDefault)
   !check_host_register_gpu("elpa1_template: ev", successGPU)

   !!page lock a (or aInter, or aExtern)
   !num = (matrixRows*matrixCols) * size_of_datatype

   !successGPU = gpu_host_register(int(loc(a),kind=c_intptr_t), num,&
   !             gpuHostRegisterDefault)
   !check_host_register_gpu("elpa1_template: a", successGPU)

#endif /* WITH_GPU_STREAMS */


#ifdef WITH_NVTX
   call nvtxRangePush("elpa1")
#endif
   call obj%get("output_pinning_information", pinningInfo, error)
   if (error .ne. ELPA_OK) then
     write(error_unit, *) "ELPA1 Problem setting option for output_pinning_information. Aborting..."
#include "./elpa1_aborting_template.F90"
   endif

#ifdef HAVE_AFFINITY_CHECKING  
   if (pinningInfo .eq. 1) then
     call init_thread_affinity(nrThreads)

     call check_thread_affinity()
     if (my_pe .eq. 0) call print_thread_affinity(my_pe)
     call cleanup_thread_affinity()
   endif
#endif
   success = .true.

   if (present(qExtern)) then
     obj%eigenvalues_only = .false.
   else
     obj%eigenvalues_only = .true.
   endif

   ! special case na = 1
   if (na .eq. 1) then
#if REALCASE == 1
     ev(1) = a(1,1)
#endif
#if COMPLEXCASE == 1
     ev(1) = real(a(1,1))
#endif
     if (.not.(obj%eigenvalues_only)) then
       q(1,1) = ONE
     endif

     ! restore original OpenMP settings
#ifdef WITH_OPENMP_TRADITIONAL
     call omp_set_num_threads(omp_threads_caller)
#endif
#ifdef ACTIVATE_SKEW
     call obj%timer%stop("elpa_solve_skew_evp_&
#else
     call obj%timer%stop("elpa_solve_evp_&
#endif
     &MATH_DATATYPE&
     &_1stage_&
     &PRECISION&
     &")
     success = .true.
     return
   endif

   if (nev == 0) then
     nev = 1
     obj%eigenvalues_only = .true.
   endif

#ifdef ACTIVATE_SKEW
   !call obj%get("is_skewsymmetric",skewsymmetric,error)
   !if (error .ne. ELPA_OK) then
   !  print *,"Problem getting option for skewsymmetric. Aborting..."
   !  stop
   !endif
   !isSkewsymmetric = (skewsymmetric == 1)
   isSkewsymmetric = .true.
#else
   isSkewsymmetric = .false.
#endif

   call obj%timer%start("mpi_communication")

   call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND), my_prowMPI, mpierr)
   call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND), my_pcolMPI, mpierr)

   my_prow = int(my_prowMPI,kind=c_int)
   my_pcol = int(my_pcolMPI,kind=c_int)

   call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)
   call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND), np_colsMPI, mpierr)

   np_rows = int(np_rowsMPI,kind=c_int)
   np_cols = int(np_colsMPI,kind=c_int)

   call obj%timer%stop("mpi_communication")

   ! allocate a dummy q_intern, if eigenvectors should not be commputed and thus q is NOT present
   if (.not.(obj%eigenvalues_only)) then
     q_actual => q(1:matrixRows,1:matrixCols)
   else
     allocate(q_dummy(1:matrixRows,1:matrixCols), stat=istat, errmsg=errorMessage)
     check_allocate("elpa1_template: q_dummy", istat, errorMessage)
     q_actual => q_dummy
   endif

#if COMPLEXCASE == 1
   l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a and q
   l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local columns of q

   l_cols_nev = local_index(nev, my_pcol, np_cols, nblk, -1) ! Local columns corresponding to nev

   allocate(q_real(l_rows,l_cols), stat=istat, errmsg=errorMessage)
   check_allocate("elpa1_template: q_real", istat, errorMessage)
#endif
   allocate(e(na), tau(na), stat=istat, errmsg=errorMessage)
   check_allocate("elpa1_template: e, tau", istat, errorMessage)

#ifdef WITH_GPU_STREAMS
! currenly only needed in tridiag and done there

!   !page lock ev (or evIntern, or evExtern)
!   num = (obj%na) * size_of_&
!                    &PRECISION&
!                    &_real
!   successGPU = gpu_host_register(int(loc(e),kind=c_intptr_t), num,&
!                gpuHostRegisterDefault)
!   check_host_register_gpu("elpa1_template: e", successGPU)
#endif



   ! start the computations
   ! as default do all three steps (this might change at some point)
   do_tridiag  = .true.
   do_solve    = .true.
   do_trans_ev = .true.

   if (do_tridiag) then
     call obj%autotune_timer%start("full_to_tridi")
     call obj%timer%start("forward")
#ifdef HAVE_LIKWID
     call likwid_markerStartRegion("tridi")
#endif
#ifdef WITH_NVTX
     call nvtxRangePush("tridi")
#endif
     call tridiag_&
     &MATH_DATATYPE&
     &_&
     &PRECISION&
     & (obj, na, a, matrixRows, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, ev, e, tau, do_useGPU_tridiag, wantDebug, &
        nrThreads, isSkewsymmetric, success)
     if (.not.(success)) then
       write(error_unit,*) "Error in tridiag. Aborting..."
       return
     endif

#ifdef WITH_NVTX
     call nvtxRangePop()
#endif
#ifdef HAVE_LIKWID
     call likwid_markerStopRegion("tridi")
#endif
     call obj%timer%stop("forward")
     call obj%autotune_timer%stop("full_to_tridi")
    endif  !do_tridiag

    if (do_solve) then
     call obj%autotune_timer%start("solve")
     call obj%timer%start("solve")

#ifdef HAVE_LIKWID
     call likwid_markerStartRegion("solve")
#endif
#ifdef WITH_NVTX
     call nvtxRangePush("solve")
#endif
     call solve_tridi_&
     &PRECISION&
     & (obj, na, nev, ev, e,  &
#if REALCASE == 1
        q_actual, matrixRows,          &
#endif
#if COMPLEXCASE == 1
        q_real, l_rows,  &
#endif
        nblk, matrixCols, mpi_comm_all, mpi_comm_rows, mpi_comm_cols, do_useGPU_solve_tridi, wantDebug, &
                success, nrThreads)

#ifdef WITH_NVTX
     call nvtxRangePop()
#endif
#ifdef HAVE_LIKWID
     call likwid_markerStopRegion("solve")
#endif
     call obj%timer%stop("solve")
     call obj%autotune_timer%stop("solve")
     if (.not.(success)) then
       write(error_unit, *) "ELPA1: solve step encountered an error. Aborting..."
#include "./elpa1_aborting_template.F90"
     endif
   endif !do_solve

   if (obj%eigenvalues_only) then
     do_trans_ev = .false.
   else
     call obj%get("check_pd",check_pd,error)
     if (error .ne. ELPA_OK) then
#include "./elpa1_aborting_template.F90"
     endif
     if (check_pd .eq. 1) then
       call obj%get("thres_pd_&
       &PRECISION&
       &",thres_pd,error)
       if (error .ne. ELPA_OK) then
         write(error_unit, *) "ELPA1 Problem setting option for thres_pd_&
         &PRECISION&
         &. Aborting..."
#include "./elpa1_aborting_template.F90"
       endif

       check_pd = 0
       do i = 1, na
         if (ev(i) .gt. thres_pd) then
           check_pd = check_pd + 1
         endif
       enddo
       if (check_pd .lt. na) then
         ! not positiv definite => eigenvectors needed
         do_trans_ev = .true.
       else
         do_trans_ev = .false.
       endif
     endif ! check_pd
   endif ! eigenvalues_only

   if (do_trans_ev) then
    ! q must be given thats why from here on we can use q and not q_actual
#if COMPLEXCASE == 1
     q(1:l_rows,1:l_cols_nev) = q_real(1:l_rows,1:l_cols_nev)
#endif
     if (isSkewsymmetric) then
     ! Extra transformation step for skew-symmetric matrix. Multiplication with diagonal complex matrix D.
     ! This makes the eigenvectors complex.
     ! For now real part of eigenvectors is generated in first half of q, imaginary part in second part.
       q(1:matrixRows, matrixCols+1:2*matrixCols) = 0.0
       do i = 1, matrixRows
!        global_index = indxl2g(i, nblk, my_prow, 0, np_rows)
         global_index = np_rows*nblk*((i-1)/nblk) + MOD(i-1,nblk) + MOD(np_rows+my_prow-0, np_rows)*nblk + 1
         if (mod(global_index-1,4) .eq. 0) then
            ! do nothing
         end if
         if (mod(global_index-1,4) .eq. 1) then
            q(i,matrixCols+1:2*matrixCols) = q(i,1:matrixCols)
            q(i,1:matrixCols) = 0
         end if
         if (mod(global_index-1,4) .eq. 2) then
            q(i,1:matrixCols) = -q(i,1:matrixCols)
         end if
         if (mod(global_index-1,4) .eq. 3) then
            q(i,matrixCols+1:2*matrixCols) = -q(i,1:matrixCols)
            q(i,1:matrixCols) = 0
         end if
       end do
     endif

     call obj%autotune_timer%start("tridi_to_full")
     call obj%timer%start("back")
#ifdef HAVE_LIKWID
     call likwid_markerStartRegion("trans_ev")
#endif
#ifdef WITH_NVTX
     call nvtxRangePush("trans_ev")
#endif

     ! In the skew-symmetric case this transforms the real part
     call trans_ev_&
     &MATH_DATATYPE&
     &_&
     &PRECISION&
     & (obj, na, nev, a, matrixRows, tau, q, matrixRows, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, do_useGPU_trans_ev, &
        success)
     if (.not.(success)) then
       write(error_unit,*) "Error in trans_ev. Aborting..."
       return
     endif
     if (isSkewsymmetric) then
       ! Transform imaginary part
       ! Transformation of real and imaginary part could also be one call of trans_ev_tridi acting on the n x 2n matrix.
       call trans_ev_&
             &MATH_DATATYPE&
             &_&
             &PRECISION&
             & (obj, na, nev, a, matrixRows, tau, q(1:matrixRows, matrixCols+1:2*matrixCols), matrixRows, nblk, matrixCols, &
                mpi_comm_rows, mpi_comm_cols, do_useGPU_trans_ev, success)
        if (.not.(success)) then
          write(error_unit,*) "Error in trans_ev. Aborting..."
          return
        endif
       endif

#ifdef WITH_NVTX
     call nvtxRangePop()
#endif
#ifdef HAVE_LIKWID
     call likwid_markerStopRegion("trans_ev")
#endif
     call obj%timer%stop("back")
     call obj%autotune_timer%stop("tridi_to_full")
   endif ! do_trans_ev

#if COMPLEXCASE == 1
    deallocate(q_real, stat=istat, errmsg=errorMessage)
    check_deallocate("elpa1_template: q_real", istat, errorMessage)
#endif

   deallocate(tau, stat=istat, errmsg=errorMessage)
   check_deallocate("elpa1_template: e, tau", istat, errorMessage)

   if (obj%eigenvalues_only) then
     deallocate(q_dummy, stat=istat, errmsg=errorMessage)
     check_deallocate("elpa1_template: q_dummy", istat, errorMessage)
   endif

#ifdef WITH_NVTX
   call nvtxRangePop()
#endif
   ! restore original OpenMP settings
#ifdef WITH_OPENMP_TRADITIONAL
   call omp_set_num_threads(omp_threads_caller)
#endif

#ifdef REDISTRIBUTE_MATRIX
   ! redistribute back if necessary
   if (doRedistributeMatrix) then

     !if (layoutInternal /= layoutExternal) then
     !  ! maybe this can be skiped I now the process grid
     !  ! and np_rows and np_cols

     !  call obj%get("mpi_comm_rows",mpi_comm_rows,error)
     !  call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)
     !  call obj%get("mpi_comm_cols",mpi_comm_cols,error)
     !  call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND), np_colsMPI, mpierr)

     !  np_rows = int(np_rowsMPI,kind=c_int)
     !  np_cols = int(np_colsMPI,kind=c_int)

     !  ! we get new blacs context and the local process grid coordinates
     !  call BLACS_Gridinit(external_blacs_ctxt, layoutInternal, int(np_rows,kind=BLAS_KIND), int(np_cols,kind=BLAS_KIND))
     !  call BLACS_Gridinfo(int(external_blacs_ctxt,KIND=BLAS_KIND), np_rows__, &
     !                      np_cols__, my_prow__, my_pcol__)

     !endif

     !call scal_PRECISION_GEMR2D &
     !(int(na,kind=BLAS_KIND), int(na,kind=BLAS_KIND), aIntern, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc_, aExtern, &
     !1_BLAS_KIND, 1_BLAS_KIND, sc_desc, external_blacs_ctxt)

     call scal_PRECISION_GEMR2D &
     (int(na,kind=BLAS_KIND), int(na,kind=BLAS_KIND), qIntern, 1_BLAS_KIND, 1_BLAS_KIND, sc_desc_, qExtern, &
     1_BLAS_KIND, 1_BLAS_KIND, sc_desc, int(external_blacs_ctxt,kind=BLAS_KIND))


     !clean MPI communicators and blacs grid
     !of the internal re-distributed matrix
     call mpi_comm_free(mpi_comm_rowsMPI_, mpierr)
     call mpi_comm_free(mpi_comm_colsMPI_, mpierr)
     call blacs_gridexit(blacs_ctxt_)
   endif
#endif /* REDISTRIBUTE_MATRIX */

#if defined(DEVICE_POINTER) || defined(REDISTRIBUTE_MATRIX)

#ifdef DEVICE_POINTER
  
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
   !copy qIntern and ev to provided device pointers
   if (present(qExtern)) then
   successGPU = gpu_memcpy(qExtern, c_loc(qIntern(1,1)), matrixRows*matrixCols*size_of_datatype, &
                             gpuMemcpyHostToDevice)
   endif
   check_memcpy_gpu("elpa1: qIntern -> qExtern", successGPU)
   successGPU = gpu_memcpy(evExtern, c_loc(ev(1)), obj%na*size_of_real_datatype, &
                             gpuMemcpyHostToDevice)
   check_memcpy_gpu("elpa1: ev -> evExtern", successGPU)
#endif
#endif

#if defined(REDISTRIBUTE_MATRIX)
   if (doRedistributeMatrix) then
#endif
     deallocate(aIntern, stat=istat, errmsg=errorMessage)
     check_deallocate("elpa1_template: aIntern", istat, errorMessage)
     !deallocate(evIntern)
     nullify(evIntern)
     if (present(qExtern)) then
       deallocate(qIntern, stat=istat, errmsg=errorMessage)
       check_deallocate("elpa1_template: qIntern", istat, errorMessage)
     endif
#if defined(REDISTRIBUTE_MATRIX)
   endif
#endif

#endif /* defined(DEVICE_POINTER) || defined(REDISTRIBUTE_MATRIX) */

#ifdef WITH_GPU_STREAMS
   !successGPU = gpu_host_unregister(int(loc(a),kind=c_intptr_t))
   !check_host_unregister_gpu("elpa1_template: a", successGPU)
#endif


#if !defined(DEVICE_POINTER) && !defined(REDISTRIBUTE_MATRIX)
   nullify(aIntern)
   nullify(evIntern)
   if (present(qExtern)) then
     nullify(qIntern)
   endif
#endif
 
   deallocate(e, stat=istat, errmsg=errorMessage)
   check_deallocate("elpa1_template: e", istat, errorMessage)
#ifdef WITH_GPU_STREAMS
   !successGPU = gpu_host_unregister(int(loc(ev),kind=c_intptr_t))
   !check_host_unregister_gpu("elpa1_template: ev", successGPU)

   !successGPU = gpu_host_unregister(int(loc(e),kind=c_intptr_t))
   !check_host_unregister_gpu("elpa1_template: e", successGPU)
#endif

   nullify(ev)
   nullify(a)
   nullify(q)

   nullify(q_actual)

#ifdef ACTIVATE_SKEW
   call obj%timer%stop("elpa_solve_skew_evp_&
#else
   call obj%timer%stop("elpa_solve_evp_&
#endif
   &MATH_DATATYPE&
   &_1stage_&
   &PRECISION&
   &")
end function


