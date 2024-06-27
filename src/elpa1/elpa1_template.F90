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


! 1. ELPA should only work with arrays /pointers called *Intern

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
   aDevExtern, &
   evDevExtern, &
   qDevExtern) result(success)
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
   use elpa_gpu
   use elpa1_gpu
   use elpa_gpu_util
#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION) || defined(WITH_OPENMP_OFFLOAD_GPU_VERSION) || defined(WITH_SYCL_GPU_VERSION)
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
   type(c_ptr)                                                        :: evDevExtern
#ifdef REDISTRIBUTE_MATRIX
   real(kind=REAL_DATATYPE), allocatable                              :: evExtern(:)
#endif /* REDISTRIBUTE_MATRIX */
#else /* DEVICE_POINTER */
   real(kind=REAL_DATATYPE), target, intent(out)                      :: evExtern(obj%na)
#endif /* DEVICE_POINTER */

   real(kind=REAL_DATATYPE), pointer                                  :: ev(:)

#ifdef DEVICE_POINTER
   type(c_ptr)                                                        :: aDevExtern
   type(c_ptr), optional                                              :: qDevExtern

#ifdef REDISTRIBUTE_MATRIX
   MATH_DATATYPE(kind=rck), allocatable                               :: aExtern(:,:)
   MATH_DATATYPE(kind=rck), allocatable                               :: qExtern(:,:)
#endif /* REDISTRIBUTE_MATRIX */


#else /* DEVICE_POINTER */

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
   integer(kind=BLAS_KIND)                         :: external_blacs_ctxtBLAS
   integer(kind=BLAS_KIND)                         :: np_rowsInternal, np_colsInternal, my_prowInternal, my_pcolInternal
   integer(kind=BLAS_KIND)                         :: np_rowsExt, np_colsExt, my_prowExt, my_pcolExt
   integer(kind=BLAS_KIND)                         :: sc_descInternal(1:9), sc_desc(1:9), sc_descExt(1:9)
   integer(kind=BLAS_KIND)                         :: na_rowsInternal, na_colsInternal, info_, blacs_ctxtInternal
   integer(kind=BLAS_KIND)                         :: na_rowsExt, na_colsExt
   integer(kind=ik)                                :: mpi_comm_rowsInternal, mpi_comm_colsInternal
   integer(kind=MPI_KIND)                          :: mpi_comm_rowsMPIInternal, mpi_comm_colsMPIInternal
   character(len=1), parameter                     :: matrixLayouts(2) = [ 'C', 'R' ]

   MATH_DATATYPE(kind=rck),  pointer  :: aIntern(:,:)
   MATH_DATATYPE(kind=C_DATATYPE_KIND), pointer   :: qIntern(:,:)
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

   integer(kind=c_intptr_t), parameter             :: size_of_datatype = size_of_&
                                                                      &PRECISION&
                                                                      &_&
                                                                      &MATH_DATATYPE
   integer(kind=c_intptr_t), parameter             :: size_of_real_datatype = size_of_&
                                                                      &PRECISION&
                                                                      &_real
   integer(kind=c_intptr_t)                        :: num
   integer(kind=c_intptr_t)                        :: tau_dev, q_part2_dev
   integer(kind=ik)                                :: negative_or_positive
#ifdef WITH_GPU_STREAMS
   integer(kind=c_intptr_t)                        :: my_stream
#endif
   integer(kind=c_intptr_t)                         :: a_devIntern, tau_devIntern, ev_devIntern, e_devIntern
   integer(kind=c_intptr_t)                         :: q_real_devIntern, q_devIntern

   integer(kind=c_intptr_t)                         :: a_dev, q_dev, ev_dev, q_dev_actual, q_dev_dummy, &
                                                       q_dev_real, e_dev
   integer(kind=ik)                                 :: success_int
   logical                                                            :: useNonBlockingCollectivesAll
   integer(kind=ik)                                                   :: non_blocking_collectives_all
   integer(kind=MPI_KIND)                                             :: allreduce_request1, &
                                                                         allreduce_request2, allreduce_request3, &
                                                                         allreduce_request4, allreduce_request

   useGPU = .false.

   ! as default do all three steps (this might change at some point)
   do_tridiag  = .true.
   do_solve    = .true.
   do_trans_ev = .true.
   ! to implement a possibiltiy to set this                             
   useNonBlockingCollectivesAll = .false.

   ! routine preperation
   na         = obj%na
   nev        = obj%nev
   matrixRows = obj%local_nrows
   nblk       = obj%nblk
   matrixCols = obj%local_ncols


   call obj%get("nbc_all_elpa2_main", non_blocking_collectives_all, error)
   if (error .ne. ELPA_OK) then
     write(error_unit,*) "ELPA1: Problem getting option for non blocking collectives. Aborting..."
#include "./elpa1_aborting_template.F90"
   endif                                                                

   if (non_blocking_collectives_all .eq. 1) then                     
     useNonBlockingCollectivesAll = .true.
   else
     useNonBlockingCollectivesAll = .false.
   endif

   ! skew?  
#ifdef ACTIVATE_SKEW
   isSkewsymmetric = .true.
#else
   isSkewsymmetric = .false.
#endif
   ! eigenvalues or eigenvectors?
#ifndef DEVICE_POINTER
   if (present(qExtern)) then
#else
   if (present(qDevExtern)) then
#endif
     obj%eigenvalues_only = .false.
   else
     obj%eigenvalues_only = .true.
   endif
   if (nev == 0) then
     nev = 1
     obj%eigenvalues_only = .true.
   endif


  if (obj%eigenvalues_only) then
    do_trans_ev = .false.
  endif

#ifdef ACTIVATE_SKEW
   call obj%timer%start("elpa_solve_skew_evp_&
#else
   call obj%timer%start("elpa_solve_evp_&
#endif
   &MATH_DATATYPE&
   &_1stage_&
   &PRECISION&
   &") ! "

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
   ! if a matrix redistribution is done then
   ! - aIntern, qIntern are getting allocated for the new distribution
   ! - nblk, matrixCols, matrixRows, mpi_comm_cols, mpi_comm_rows are getting updated
   ! TODO: make sure that nowhere in ELPA the communicators are getting "getted",
   ! and the variables obj%local_nrows,1:obj%local_ncols are being used
   ! - a points then to aIntern, q points to qIntern
#include "../helpers/elpa_redistribute_template.F90"
#endif /* REDISTRIBUTE_MATRIX */
!

   my_pe    = obj%mpi_setup%myRank_comm_parent
   my_prow = obj%mpi_setup%myRank_comm_rows
   my_pcol = obj%mpi_setup%myRank_comm_cols

   np_rows = obj%mpi_setup%nRanks_comm_rows
   np_cols = obj%mpi_setup%nRanks_comm_cols
   n_pes   = obj%mpi_setup%nRanks_comm_parent

#if COMPLEXCASE == 1
   l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a and q
   l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local columns of q
   l_cols_nev = local_index(nev, my_pcol, np_cols, nblk, -1) ! Local columns corresponding to nev
#endif

#ifndef DEVICE_POINTER
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! no device pointer
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   allocate(e(na), tau(na), stat=istat, errmsg=errorMessage)
   check_allocate("elpa1_template: e, tau", istat, errorMessage)

#ifndef REDISTRIBUTE_MATRIX
   aIntern => aExtern(1:matrixRows,1:matrixCols)
   a => aExtern(1:matrixRows,1:matrixCols)
   if (present(qExtern)) then
     if (isSkewsymmetric) then
       qIntern => qExtern(1:matrixRows,1:2*matrixCols)
       q => qExtern(1:matrixRows,1:2*matrixCols)
     else
       qIntern => qExtern(1:matrixRows,1:matrixCols)
       q => qExtern(1:matrixRows,1:matrixCols)
     endif
   endif
   evIntern => evExtern(1:obj%na)
   ev => evExtern(1:obj%na)

   ! allocate a dummy q_intern, if eigenvectors should not be commputed and thus q is NOT present
   if (.not.(obj%eigenvalues_only)) then
     q_actual => q(1:matrixRows,1:matrixCols)
   else
     allocate(q_dummy(1:matrixRows,1:matrixCols), stat=istat, errmsg=errorMessage)
     check_allocate("elpa1_template: q_dummy", istat, errorMessage)
     q_actual => q_dummy
   endif

#if COMPLEXCASE == 1
   allocate(q_real(l_rows,l_cols), stat=istat, errmsg=errorMessage)
   check_allocate("elpa1_template: q_real", istat, errorMessage)
#endif /* COMPLEXCASE */

#else /* REDISTRIBUTE_MATRIX */

   if (doRedistributeMatrix) then
     !aIntern => aExtern(1:matrixRows,1:matrixCols)
     a => aIntern(1:matrixRows,1:matrixCols)
     if (present(qExtern)) then
       if (isSkewsymmetric) then
         !qIntern => qExtern(1:matrixRows,1:2*matrixCols)
         q => qIntern(1:matrixRows,1:2*matrixCols)
       else
         !qIntern => qExtern(1:matrixRows,1:matrixCols)
         q => qIntern(1:matrixRows,1:matrixCols)
       endif
     endif
     ! ev never changes
     evIntern => evExtern(1:obj%na)
     ev => evIntern(1:obj%na)

     ! allocate a dummy q_intern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
       q_actual => q(1:matrixRows,1:matrixCols)
     else
       allocate(q_dummy(1:matrixRows,1:matrixCols), stat=istat, errmsg=errorMessage)
       check_allocate("elpa1_template: q_dummy", istat, errorMessage)
       q_actual => q_dummy
     endif

#if COMPLEXCASE == 1
     allocate(q_real(l_rows,l_cols), stat=istat, errmsg=errorMessage)
     check_allocate("elpa1_template: q_real", istat, errorMessage)
#endif /* COMPLEXCASE */

   else ! doRedistributeMatrix
     aIntern => aExtern(1:matrixRows,1:matrixCols)
     a => aExtern(1:matrixRows,1:matrixCols)
     if (present(qExtern)) then
       if (isSkewsymmetric) then
         qIntern => qExtern(1:matrixRows,1:2*matrixCols)
         q => qExtern(1:matrixRows,1:2*matrixCols)
       else
         qIntern => qExtern(1:matrixRows,1:matrixCols)
         q => qExtern(1:matrixRows,1:matrixCols)
       endif
     endif
     evIntern => evExtern(1:obj%na)
     ev => evExtern(1:obj%na)

     ! allocate a dummy q_intern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
       q_actual => q(1:matrixRows,1:matrixCols)
     else
       allocate(q_dummy(1:matrixRows,1:matrixCols), stat=istat, errmsg=errorMessage)
       check_allocate("elpa1_template: q_dummy", istat, errorMessage)
       q_actual => q_dummy
     endif

#if COMPLEXCASE == 1
     allocate(q_real(l_rows,l_cols), stat=istat, errmsg=errorMessage)
     check_allocate("elpa1_template: q_real", istat, errorMessage)
#endif /* COMPLEXCASE */
   endif ! doRedistributeMatrix


#endif /* REDISTRIBUTE_MATRIX */

   if (useGPU) then
     num = (na) * size_of_real_datatype
     successGPU = gpu_malloc(e_dev, num)
     check_alloc_gpu("elpa1_template e_dev", successGPU)

     num = (na) * size_of_datatype
     successGPU = gpu_malloc(tau_dev, num)
     check_alloc_gpu("elpa1_template tau_devIntern", successGPU)

#ifndef REDISTRIBUTE_MATRIX
     ! alloc a_devIntern, q_devIntern, ev_devIntern
     num = (matrixRows* matrixCols) * size_of_datatype
     successGPU = gpu_malloc(a_devIntern, num)
     check_alloc_gpu("elpa1_template a_devIntern", successGPU)

     successGPU = gpu_memcpy(a_devIntern, int(loc(a(1,1)),kind=c_intptr_t), &
                 num, gpuMemcpyHostToDevice)
     check_memcpy_gpu("elpa1_template a -> a_devIntern", successGPU)

     num = (na) * size_of_real_datatype
     successGPU = gpu_malloc(ev_devIntern, num)
     check_alloc_gpu("elpa1_template ev_devIntern", successGPU)

     if (present(qExtern)) then
       if (isSkewsymmetric) then
         num = (matrixRows* 2*matrixCols) * size_of_datatype
       else
         num = (matrixRows* matrixCols) * size_of_datatype
       endif
       successGPU = gpu_malloc(q_devIntern, num)
       check_alloc_gpu("elpa1_template q_devIntern", successGPU)
     endif

     ! associate a_dev, q_dev, ev_dev
     a_dev = transfer(a_devIntern, a_dev)
     ev_dev = transfer(ev_devIntern, ev_dev)
     if (present(qExtern)) then
       q_dev = transfer(q_devIntern, q_dev)
     endif
     
     ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
       q_dev_actual = transfer(q_dev, q_dev_actual)
     else
       num = (matrixRows* matrixCols) * size_of_datatype
       successGPU = gpu_malloc(q_dev_dummy, num)
       check_alloc_gpu("elpa1_template q_dev_dummy", successGPU)
       q_dev_actual = transfer(q_dev_dummy, q_dev_actual)
     endif

#if COMPLEXCASE == 1
     num = (l_rows* l_cols) * size_of_real_datatype 
     successGPU = gpu_malloc(q_dev_real, num)
     check_alloc_gpu("elpa1_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */

#else /* REDISTRIBUTE_MATRIX */


   if (doRedistributeMatrix) then

     !  a_devIntern, q_devIntern already allocated
     !
     !! alloc a_devIntern, q_devIntern, ev_devIntern
     !num = (matrixRows* matrixCols) * size_of_datatype
     !successGPU = gpu_malloc(a_devIntern, num)
     !check_alloc_gpu("elpa1_template a_devIntern", successGPU)

     !successGPU = gpu_memcpy(a_devIntern, int(loc(a(1,1)),kind=c_intptr_t), &
     !            num, gpuMemcpyHostToDevice)
     !check_memcpy_gpu("elpa1_template a -> a_devIntern", successGPU)

     num = (na) * size_of_real_datatype
     successGPU = gpu_malloc(ev_devIntern, num)
     check_alloc_gpu("elpa1_template ev_devIntern", successGPU)
     !successGPU = gpu_memcpy(ev_devIntern, int(loc(ev(1)),kind=c_intptr_t), &
     !            num, gpuMemcpyHostToDevice)
     !check_memcpy_gpu("elpa1_template ev -> ev_devIntern", successGPU)
     !if (present(qExtern)) then
     !  if (isSkewsymmetric) then
     !    num = (matrixRows* 2*matrixCols) * size_of_datatype
     !  else
     !    num = (matrixRows* matrixCols) * size_of_datatype
     !  endif
     !  successGPU = gpu_malloc(q_devIntern, num)
     !  check_alloc_gpu("elpa1_template q_devIntern", successGPU)
     !endif


     allocate(aIntern(matrixRows,matrixCols), stat=istat, errmsg=errorMessage)
     check_allocate("elpa1_template: aIntern", istat, errorMessage)
     if (isSkewsymmetric) then
       allocate(qIntern(matrixRows,2*matrixCols), stat=istat, errmsg=errorMessage)
     else
       allocate(qIntern(matrixRows,matrixCols), stat=istat, errmsg=errorMessage)
     endif
     check_allocate("elpa1_template: qIntern", istat, errorMessage)

     if (isSkewsymmetric) then
       q => qIntern(1:matrixRows,1:2*matrixCols)
     else
       q => qIntern(1:matrixRows,1:matrixCols)
     endif
     a => aIntern(1:matrixRows,1:matrixCols)

     ! associate a_dev, q_dev, ev_dev
     a_dev = transfer(a_devIntern, a_dev)
     ev_dev = transfer(ev_devIntern, ev_dev)
     if (present(qExtern)) then
       q_dev = transfer(q_devIntern, q_dev)
     endif
     
     ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
       q_dev_actual = transfer(q_dev, q_dev_actual)
     else
       num = (matrixRows* matrixCols) * size_of_datatype
       successGPU = gpu_malloc(q_dev_dummy, num)
       check_alloc_gpu("elpa1_template q_dev_dummy", successGPU)
       q_dev_actual = transfer(q_dev_dummy, q_dev_actual)
     endif

#if COMPLEXCASE == 1
     num = (l_rows* l_cols) * size_of_real_datatype 
     successGPU = gpu_malloc(q_dev_real, num)
     check_alloc_gpu("elpa1_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */

   else ! doRedistributeMatrix
     ! alloc a_devIntern, q_devIntern, ev_devIntern
     num = (matrixRows* matrixCols) * size_of_datatype
     successGPU = gpu_malloc(a_devIntern, num)
     check_alloc_gpu("elpa1_template a_devIntern", successGPU)

     successGPU = gpu_memcpy(a_devIntern, int(loc(a(1,1)),kind=c_intptr_t), &
                 num, gpuMemcpyHostToDevice)
     check_memcpy_gpu("elpa1_template a -> a_devIntern", successGPU)

     num = (na) * size_of_real_datatype
     successGPU = gpu_malloc(ev_devIntern, num)
     check_alloc_gpu("elpa1_template ev_devIntern", successGPU)
     !successGPU = gpu_memcpy(ev_devIntern, int(loc(ev(1)),kind=c_intptr_t), &
     !            num, gpuMemcpyHostToDevice)
     !check_memcpy_gpu("elpa1_template ev -> ev_devIntern", successGPU)
     if (present(qExtern)) then
       if (isSkewsymmetric) then
         num = (matrixRows* 2*matrixCols) * size_of_datatype
       else
         num = (matrixRows* matrixCols) * size_of_datatype
       endif
       successGPU = gpu_malloc(q_devIntern, num)
       check_alloc_gpu("elpa1_template q_devIntern", successGPU)
     endif

     ! associate a_dev, q_dev, ev_dev
     a_dev = transfer(a_devIntern, a_dev)
     ev_dev = transfer(ev_devIntern, ev_dev)
     if (present(qExtern)) then
       q_dev = transfer(q_devIntern, q_dev)
     endif
     
     ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
       q_dev_actual = transfer(q_dev, q_dev_actual)
     else
       num = (matrixRows* matrixCols) * size_of_datatype
       successGPU = gpu_malloc(q_dev_dummy, num)
       check_alloc_gpu("elpa1_template q_dev_dummy", successGPU)
       q_dev_actual = transfer(q_dev_dummy, q_dev_actual)
     endif

#if COMPLEXCASE == 1
     num = (l_rows* l_cols) * size_of_real_datatype 
     successGPU = gpu_malloc(q_dev_real, num)
     check_alloc_gpu("elpa1_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */
   endif ! doRedistributeMatrix

#endif /* REDISTRIBUTE_MATRIX */
   endif ! useGPU
#else /* DEVICE_POINTER */
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !
   ! DEVICE POINTER
   !
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !
   if (useGPU) then
     num = (na) * size_of_real_datatype
     successGPU = gpu_malloc(e_dev, num)
     check_alloc_gpu("elpa1_template e_dev", successGPU)

     num = (na) * size_of_datatype
     successGPU = gpu_malloc(tau_dev, num)
     check_alloc_gpu("elpa1_template tau_devIntern", successGPU)

#ifndef REDISTRIBUTE_MATRIX
     ! associate a_dev, q_dev, ev_dev
     a_devIntern = transfer(aDevExtern, a_devIntern)
     a_dev = transfer(a_devIntern, a_dev)
     ev_dev = transfer(evDevExtern, ev_dev)
     ev_devIntern = transfer(evDevExtern, ev_devIntern)
     if (present(qDevExtern)) then
       q_dev = transfer(qDevExtern, q_dev)
       q_devIntern = transfer(qDevExtern, q_devIntern)
     endif

     ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
       q_dev_actual = transfer(q_dev, q_dev_actual)
     else
       num = (matrixRows* matrixCols) * size_of_datatype
       successGPU = gpu_malloc(q_dev_dummy, num)
       check_alloc_gpu("elpa1_template q_dev_dummy", successGPU)
       q_dev_actual = transfer(q_dev_dummy, q_dev_actual)
     endif

#if COMPLEXCASE == 1
     num = (l_rows* l_cols) * size_of_real_datatype 
     successGPU = gpu_malloc(q_dev_real, num)
     check_alloc_gpu("elpa1_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */

#else /* REDISTRIBUTE_MATRIX */
     if (doRedistributeMatrix) then
       ! associate a_dev, q_dev, ev_dev
       !a_devIntern = transfer(aDevExtern, a_devIntern)
       a_dev = transfer(a_devIntern, a_dev)
       ev_dev = transfer(evDevExtern, ev_dev)
       ev_devIntern = transfer(evDevExtern, ev_devIntern)
       if (present(qDevExtern)) then
         q_dev = transfer(q_devIntern, q_dev)
       endif

       ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
       if (.not.(obj%eigenvalues_only)) then
         q_dev_actual = transfer(q_dev, q_dev_actual)
       else
         num = (matrixRows* matrixCols) * size_of_datatype
         successGPU = gpu_malloc(q_dev_dummy, num)
         check_alloc_gpu("elpa1_template q_dev_dummy", successGPU)
         q_dev_actual = transfer(q_dev_dummy, q_dev_actual)
       endif

#if COMPLEXCASE == 1
       num = (l_rows* l_cols) * size_of_real_datatype 
       successGPU = gpu_malloc(q_dev_real, num)
       check_alloc_gpu("elpa1_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */

     else ! doRedistributeMatrix
       ! associate a_dev, q_dev, ev_dev
       a_devIntern = transfer(aDevExtern, a_devIntern)
       a_dev = transfer(a_devIntern, a_dev)
       ev_dev = transfer(evDevExtern, ev_dev)
       ev_devIntern = transfer(evDevExtern, ev_devIntern)
       if (present(qDevExtern)) then
         q_devIntern = transfer(qDevExtern, q_devIntern)
         q_dev = transfer(q_devIntern, q_dev)
       endif

       ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
       if (.not.(obj%eigenvalues_only)) then
         q_dev_actual = transfer(q_dev, q_dev_actual)
       else
         num = (matrixRows* matrixCols) * size_of_datatype
         successGPU = gpu_malloc(q_dev_dummy, num)
         check_alloc_gpu("elpa1_template q_dev_dummy", successGPU)
         q_dev_actual = transfer(q_dev_dummy, q_dev_actual)
       endif

#if COMPLEXCASE == 1
       num = (l_rows* l_cols) * size_of_real_datatype 
       successGPU = gpu_malloc(q_dev_real, num)
       check_alloc_gpu("elpa1_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */
     endif ! doRedistributeMatrix

#endif /* REDISTRIBUTE_MATRIX */
   endif ! useGPU

#endif /* DEVICE_POINTER */

! no allocate after here


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

#ifndef DEVICE_POINTER
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
     if (useGPU) then
       ! nothing to do since no compute yet
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
     &") ! "
     success = .true.
     return
   endif
#endif /* DEVICE_POINTER */

   ! start the computations
   if (do_tridiag) then
     call obj%autotune_timer%start("full_to_tridi")
     call obj%timer%start("forward")
#ifdef HAVE_LIKWID
     call likwid_markerStartRegion("tridi")
#endif
#ifdef WITH_NVTX
     call nvtxRangePush("tridi")
#endif

     if (do_useGPU_tridiag) then
       call tridiag_gpu_&
       &MATH_DATATYPE&
       &_&
       &PRECISION&
       & (obj, na, a_dev, matrixRows, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, ev_dev, e_dev, &
         tau_dev, wantDebug, nrThreads, isSkewsymmetric, success)
     else
       call tridiag_cpu_&
       &MATH_DATATYPE&
       &_&
       &PRECISION&
       & (obj, na, a, matrixRows, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, ev, e, tau, wantDebug, &
          nrThreads, isSkewsymmetric, success)
     endif
     if (success) then
       success_int = 0
     else
       success_int = 1
     endif
#ifdef WITH_MPI
     if (useNonBlockingCollectivesAll) then
       call mpi_iallreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), &
       allreduce_request1, mpierr)
       call mpi_wait(allreduce_request1, MPI_STATUS_IGNORE, mpierr)
     else
       call mpi_allreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
     endif
#endif
     if (success_int .eq. 1) then
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

     if (do_useGPU_solve_tridi) then
       call solve_tridi_gpu_&
       &PRECISION&
       & (obj, na, nev, ev_dev, e_dev,  &
#if REALCASE == 1
        q_dev_actual, matrixRows,          &
#endif
#if COMPLEXCASE == 1
        q_dev_real, l_rows,  &
#endif
        nblk, matrixCols, mpi_comm_all, mpi_comm_rows, mpi_comm_cols, wantDebug, &
                success, nrThreads)
     else
       call solve_tridi_cpu_&
       &PRECISION&
       & (obj, na, nev, ev, e,  &
#if REALCASE == 1
        q_actual, matrixRows,          &
#endif
#if COMPLEXCASE == 1
        q_real, l_rows,  &
#endif
        nblk, matrixCols, mpi_comm_all, mpi_comm_rows, mpi_comm_cols, wantDebug, &
                success, nrThreads)
    endif

#ifdef WITH_NVTX
     call nvtxRangePop()
#endif
#ifdef HAVE_LIKWID
     call likwid_markerStopRegion("solve")
#endif
     call obj%timer%stop("solve")
     call obj%autotune_timer%stop("solve")
     if (success) then
       success_int = 0
     else
       success_int = 1
     endif
#ifdef WITH_MPI
     if (useNonBlockingCollectivesAll) then
       call mpi_iallreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), &
       allreduce_request2, mpierr)
       call mpi_wait(allreduce_request2, MPI_STATUS_IGNORE, mpierr)
     else
       call mpi_allreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
     endif
#endif
     if (success_int .eq. 1) then
       write(error_unit,*) "Error in solve. Aborting..."
       return
     endif


   endif !do_solve

   if (do_trans_ev) then
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
       if (do_useGPU_solve_tridi) then
         num = (na) * size_of_real_datatype
         successGPU = gpu_memcpy(int(loc(ev(1)),kind=c_intptr_t), ev_dev, &
                 num, gpuMemcpyDeviceToHost)
         check_memcpy_gpu("elpa1_template ev_dev -> ev", successGPU)
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
   endif ! do_trans_ev

   if (do_trans_ev) then

    ! q must be given thats why from here on we can use q and not q_actual
#if COMPLEXCASE == 1
     if (do_useGPU_trans_ev) then
#ifdef WITH_GPU_STREAMS
       my_stream = obj%gpu_setup%my_stream
       call GPU_COPY_REAL_PART_TO_Q_PRECISION(q_dev, q_dev_real, matrixRows, l_rows, l_cols_nev, my_stream)
#else
       call GPU_COPY_REAL_PART_TO_Q_PRECISION(q_dev, q_dev_real, matrixRows, l_rows, l_cols_nev)
#endif
     else
       q(1:l_rows,1:l_cols_nev) = q_real(1:l_rows,1:l_cols_nev)
     endif
#endif /* COMPLEXCASE */


     if (isSkewsymmetric) then

       if (do_useGPU_trans_ev) then
#ifdef WITH_GPU_STREAMS
         my_stream = obj%gpu_setup%my_stream
         call GPU_ZERO_SKEWSYMMETRIC_Q_PRECISION(q_dev, matrixRows, matrixCols, my_stream)
#else
         call GPU_ZERO_SKEWSYMMETRIC_Q_PRECISION(q_dev, matrixRows, matrixCols)
#endif
         do i = 1, matrixRows
           global_index = np_rows*nblk*((i-1)/nblk) + MOD(i-1,nblk) + MOD(np_rows+my_prow-0, np_rows)*nblk + 1
           if (mod(global_index-1,4) .eq. 0) then
             ! do nothing
           end if
           if (mod(global_index-1,4) .eq. 1) then
             negative_or_positive = 1
#ifdef WITH_GPU_STREAMS
             my_stream = obj%gpu_setup%my_stream
             call GPU_COPY_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION(q_dev, i, matrixRows, &
                     matrixCols, negative_or_positive, my_stream)
#else
             call GPU_COPY_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION(q_dev, i, matrixRows, &
                     matrixCols, negative_or_positive)
#endif
           end if
           if (mod(global_index-1,4) .eq. 2) then
             negative_or_positive = -1
#ifdef WITH_GPU_STREAMS
             my_stream = obj%gpu_setup%my_stream
             call GPU_COPY_SKEWSYMMETRIC_FIRST_HALF_Q_PRECISION(q_dev, i, matrixRows, &
                     matrixCols, negative_or_positive, my_stream)
#else
             call GPU_COPY_SKEWSYMMETRIC_FIRST_HALF_Q_PRECISION(q_dev, i, matrixRows, &
                     matrixCols, negative_or_positive)
#endif
           end if
           if (mod(global_index-1,4) .eq. 3) then
             negative_or_positive = -1
#ifdef WITH_GPU_STREAMS
             my_stream = obj%gpu_setup%my_stream
             call GPU_COPY_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION(q_dev, i, matrixRows, matrixCols, &
                          negative_or_positive, my_stream)
#else
             call GPU_COPY_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION(q_dev, i, matrixRows, matrixCols, &
                          negative_or_positive)
#endif
           end if
         enddo
       else ! do_useGPU_trans_ev
         ! Extra transformation step for skew-symmetric matrix. Multiplication with diagonal complex matrix D.
         ! This makes the eigenvectors complex.
         ! For now real part of eigenvectors is generated in first half of q, imaginary part in second part.
         q(1:matrixRows, matrixCols+1:2*matrixCols) = 0.0
         do i = 1, matrixRows
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
       endif ! do_useGPU_trans_ev

     endif ! isSkewsymmetric

     call obj%autotune_timer%start("tridi_to_full")
     call obj%timer%start("back")
#ifdef HAVE_LIKWID
     call likwid_markerStartRegion("trans_ev")
#endif
#ifdef WITH_NVTX
     call nvtxRangePush("trans_ev")
#endif


     ! In the skew-symmetric case this transforms the real part
     if (do_useGPU_trans_ev) then
       call trans_ev_gpu_&
       &MATH_DATATYPE&
       &_&
       &PRECISION&
       & (obj, na, nev, a_dev, matrixRows, tau_dev, q_dev, matrixRows, nblk, matrixCols, &
          mpi_comm_rows, mpi_comm_cols, success)
     else
       call trans_ev_cpu_&
       &MATH_DATATYPE&
       &_&
       &PRECISION&
       & (obj, na, nev, a, matrixRows, tau, q, matrixRows, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, &
        success)
     endif

     if (success) then
       success_int = 0
     else
       success_int = 1
     endif
#ifdef WITH_MPI
     if (useNonBlockingCollectivesAll) then
       call mpi_iallreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), &
       allreduce_request3, mpierr)
       call mpi_wait(allreduce_request3, MPI_STATUS_IGNORE, mpierr)
     else
       call mpi_allreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
     endif
#endif
     if (success_int .eq. 1) then
       write(error_unit,*) "Error in trans_ev (real). Aborting..."
       return
     endif

     if (isSkewsymmetric) then

       if (.not.(do_useGPU_trans_ev)) then
         ! Transform imaginary part
         ! Transformation of real and imaginary part could also be one call of trans_ev_tridi acting on the n x 2n matrix.
         call trans_ev_cpu_&
               &MATH_DATATYPE&
               &_&
               &PRECISION&
               & (obj, na, nev, a, matrixRows, tau, q(1:matrixRows, matrixCols+1:2*matrixCols), matrixRows, nblk, matrixCols, &
                  mpi_comm_rows, mpi_comm_cols, success)
       else ! do_useGPU_trans_ev
         num = matrixRows*matrixCols*size_of_datatype
         successGPU = gpu_malloc(q_part2_dev, num)
         check_alloc_gpu("elpa1_template q_dev", successGPU)

         ! copy q_part2(1:matrixRows,1:matrixCols) = q(1:matrixRows, matrixCols+1:2*matrixCols)
#ifdef WITH_GPU_STREAMS
         my_stream = obj%gpu_setup%my_stream
         call GPU_GET_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION(q_dev, q_part2_dev, matrixRows, matrixCols, &
                                                                my_stream)
#else
         call GPU_GET_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION(q_dev, q_part2_dev, matrixRows, matrixCols)
#endif

         call trans_ev_gpu_&
         &MATH_DATATYPE&
         &_&
         &PRECISION&
         & (obj, na, nev, a_dev, matrixRows, tau_dev, q_part2_dev, matrixRows, nblk, matrixCols, &
            mpi_comm_rows, mpi_comm_cols, success)
       endif ! do_useGPU_trans_ev
       if (success) then
         success_int = 0
       else
         success_int = 1
       endif
#ifdef WITH_MPI
       if (useNonBlockingCollectivesAll) then
         call mpi_iallreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), &
         allreduce_request4, mpierr)
         call mpi_wait(allreduce_request4, MPI_STATUS_IGNORE, mpierr)
       else
         call mpi_allreduce(mpi_in_place, success_int, 1_MPI_KIND, MPI_INTEGER, MPI_MAX, int(mpi_comm_all,kind=MPI_KIND), mpierr)
       endif
#endif
       if (success_int .eq. 1) then
         write(error_unit,*) "Error in trans_ev (imag). Aborting..."
         return
       endif
     endif ! isSkewsymmetric

     if (isSkewsymmetric) then
       if (do_useGPU_trans_ev) then
#ifdef WITH_GPU_STREAMS
         my_stream = obj%gpu_setup%my_stream
         call GPU_PUT_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION(q_dev, q_part2_dev, matrixRows, matrixCols, &
                                                                 my_stream)
#else
         call GPU_PUT_SKEWSYMMETRIC_SECOND_HALF_Q_PRECISION(q_dev, q_part2_dev, matrixRows, matrixCols)
#endif
       endif
     endif

#ifndef DEVICE_POINTER
     if (useGPU) then
       ! copy back
       if (isSkewsymmetric) then
         num = (matrixRows* 2*matrixCols) * size_of_datatype
       else
         num = (matrixRows* matrixCols) * size_of_datatype
       endif
       successGPU = gpu_memcpy(int(loc(q(1,1)),kind=c_intptr_t), q_dev, &
                    num, gpuMemcpyDeviceToHost)
       check_memcpy_gpu("elpa1_template q_dev -> q", successGPU)
     endif
#endif /* DEVICE_POINTER */

     if (isSkewsymmetric) then
       if (do_useGPU_trans_ev) then
         successGPU = gpu_free(q_part2_dev)
         check_dealloc_gpu("elpa1_template q_part2_dev", successGPU)
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


#ifndef DEVICE_POINTER
     if (useGPU) then
       ! copy back always
       num = (na) * size_of_real_datatype
       successGPU = gpu_memcpy(int(loc(ev(1)),kind=c_intptr_t), ev_dev, &
                 num, gpuMemcpyDeviceToHost)
       check_memcpy_gpu("elpa1_template ev_dev -> ev", successGPU)
     endif
#endif /* DEVICE_POINTER */


#ifdef WITH_NVTX
   call nvtxRangePop()
#endif
   ! restore original OpenMP settings
#ifdef WITH_OPENMP_TRADITIONAL
   call omp_set_num_threads(omp_threads_caller)
#endif

#ifdef REDISTRIBUTE_MATRIX
#include "../helpers/elpa_redistribute_back_template.F90"
#endif /* REDISTRIBUTE_MATRIX */

#ifndef DEVICE_POINTER
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! no device pointer
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   deallocate(e, tau, stat=istat, errmsg=errorMessage)
   check_deallocate("elpa1_template: e, tau", istat, errorMessage)

#ifndef REDISTRIBUTE_MATRIX
   nullify(aIntern)
   nullify(a)
   if (present(qExtern)) then
     nullify(qIntern)
     nullify(q)
   endif
   nullify(evIntern)
   nullify(ev)

   if (.not.(obj%eigenvalues_only)) then
     nullify(q_actual)
   else
     deallocate(q_dummy, stat=istat, errmsg=errorMessage)
     check_deallocate("elpa1_template: q_dummy", istat, errorMessage)
     nullify(q_actual)
   endif

#if COMPLEXCASE == 1
   deallocate(q_real, stat=istat, errmsg=errorMessage)
   check_deallocate("elpa1_template: q_real", istat, errorMessage)
#endif /* COMPLEXCASE */

#else /* REDISTRIBUTE_MATRIX */

   if (doRedistributeMatrix) then
     nullify(a)
     ! deaclloacte aIntern!!
     if (present(qExtern)) then
       nullify(q)
       ! deallocate qIntern!!!
     endif
     ! ev never changes
     nullify(evIntern)
     nullify(ev)

     if (.not.(obj%eigenvalues_only)) then
       nullify(q_actual)
     else
       deallocate(q_dummy, stat=istat, errmsg=errorMessage)
       check_deallocate("elpa1_template: q_dummy", istat, errorMessage)
       nullify(q_actual)
     endif

#if COMPLEXCASE == 1
     deallocate(q_real, stat=istat, errmsg=errorMessage)
     check_deallocate("elpa1_template: q_real", istat, errorMessage)
#endif /* COMPLEXCASE */

   else ! doRedistributeMatrix
     nullify(aIntern)
     nullify(a)
     if (present(qExtern)) then
       nullify(qIntern)
       nullify(q)
     endif
     nullify(evIntern)
     nullify(ev)

     ! allocate a dummy q_intern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
       nullify(q_actual)
     else
       deallocate(q_dummy, stat=istat, errmsg=errorMessage)
       check_deallocate("elpa1_template: q_dummy", istat, errorMessage)
       nullify(q_actual)
     endif

#if COMPLEXCASE == 1
     deallocate(q_real, stat=istat, errmsg=errorMessage)
     check_deallocate("elpa1_template: q_real", istat, errorMessage)
#endif /* COMPLEXCASE */
   endif ! doRedistributeMatrix


#endif /* REDISTRIBUTE_MATRIX */

   if (useGPU) then
     successGPU = gpu_free(e_dev)
     check_dealloc_gpu("elpa1_template e_dev", successGPU)

     successGPU = gpu_free(tau_dev)
     check_dealloc_gpu("elpa1_template tau_dev", successGPU)

#ifndef REDISTRIBUTE_MATRIX
     ! alloc a_devIntern, q_devIntern, ev_devIntern
     successGPU = gpu_free(a_devIntern)
     check_dealloc_gpu("elpa1_template a_devIntern", successGPU)

     successGPU = gpu_free(ev_devIntern)
     check_dealloc_gpu("elpa1_template ev_devIntern", successGPU)

     if (present(qExtern)) then
       successGPU = gpu_free(q_devIntern)
       check_dealloc_gpu("elpa1_template q_devIntern 1", successGPU)
     endif

     ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
     else
       successGPU = gpu_free(q_dev_dummy)
       check_dealloc_gpu("elpa1_template q_dev_dummy", successGPU)
     endif

#if COMPLEXCASE == 1
     successGPU = gpu_free(q_dev_real)
     check_dealloc_gpu("elpa1_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */

#else /* REDISTRIBUTE_MATRIX */


   if (doRedistributeMatrix) then

     successGPU = gpu_free(ev_devIntern)
     check_dealloc_gpu("elpa1_template ev_devIntern", successGPU)

     deallocate(aIntern, stat=istat, errmsg=errorMessage)
     check_deallocate("elpa1_template: aIntern", istat, errorMessage)
     deallocate(qIntern, stat=istat, errmsg=errorMessage)
     check_deallocate("elpa1_template: qIntern", istat, errorMessage)

     nullify(q)
     nullify(a)

     ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
     else
       successGPU = gpu_free(q_dev_dummy)
       check_dealloc_gpu("elpa1_template q_dev_dummy", successGPU)
     endif

#if COMPLEXCASE == 1
     successGPU = gpu_free(q_dev_real)
     check_dealloc_gpu("elpa1_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */

   else ! doRedistributeMatrix
     successGPU = gpu_free(a_devIntern)
     check_dealloc_gpu("elpa1_template a_devIntern", successGPU)

     successGPU = gpu_free(ev_devIntern)
     check_dealloc_gpu("elpa1_template ev_devIntern", successGPU)
     !check_memcpy_gpu("elpa1_template ev -> ev_devIntern", successGPU)
     if (present(qExtern)) then
       successGPU = gpu_free(q_devIntern)
       check_dealloc_gpu("elpa1_template q_devIntern 2", successGPU)
     endif

     
     ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
     else
       successGPU = gpu_free(q_dev_dummy)
       check_dealloc_gpu("elpa1_template q_dev_dummy", successGPU)
     endif

#if COMPLEXCASE == 1
     successGPU = gpu_free(q_dev_real)
     check_dealloc_gpu("elpa1_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */
   endif ! doRedistributeMatrix

#endif /* REDISTRIBUTE_MATRIX */
   endif ! useGPU
#else /* DEVICE_POINTER */
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !
   ! DEVICE POINTER
   !
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !
   if (useGPU) then
     successGPU = gpu_free(e_dev)
     check_dealloc_gpu("elpa1_template e_dev", successGPU)

     successGPU = gpu_free(tau_dev)
     check_dealloc_gpu("elpa1_template tau_devIntern", successGPU)

#ifndef REDISTRIBUTE_MATRIX
     ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
     if (.not.(obj%eigenvalues_only)) then
     else
       successGPU = gpu_free(q_dev_dummy)
       check_dealloc_gpu("elpa1_template q_dev_dummy", successGPU)
     endif

#if COMPLEXCASE == 1
     successGPU = gpu_free(q_dev_real)
     check_dealloc_gpu("elpa1_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */

#else /* REDISTRIBUTE_MATRIX */
     if (doRedistributeMatrix) then
       if (present(qDevExtern)) then
       endif

       ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
       if (.not.(obj%eigenvalues_only)) then
       else
         successGPU = gpu_free(q_dev_dummy)
         check_dealloc_gpu("elpa1_template q_dev_dummy", successGPU)
       endif

#if COMPLEXCASE == 1
       successGPU = gpu_free(q_dev_real)
       check_dealloc_gpu("elpa1_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */

     else ! doRedistributeMatrix

       ! allocate dummy q_devIntern, if eigenvectors should not be commputed and thus q is NOT present
       if (.not.(obj%eigenvalues_only)) then
       else
         successGPU = gpu_free(q_dev_dummy)
         check_dealloc_gpu("elpa1_template q_dev_dummy", successGPU)
       endif

#if COMPLEXCASE == 1
       successGPU = gpu_free(q_dev_real)
       check_dealloc_gpu("elpa1_template q_dev_real", successGPU)
#endif /* COMPLEXCASE */
     endif ! doRedistributeMatrix

#endif /* REDISTRIBUTE_MATRIX */
   endif ! useGPU

#endif /* DEVICE_POINTER */





#if defined(WITH_NVIDIA_GPU_VERSION) || defined(WITH_AMD_GPU_VERSION)
   successGPU = gpu_get_last_error()
   if (.not.successGPU) then
    print *,"elpa1_template: GPU error detected via gpu_get_last_error(). Aborting..."
    print *,"Rerun the program with the debug option e.g. 'export ELPA_DEFAULT_debug=1'"
    stop 1
  endif
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
end function


