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

#undef USE_CCL_SOLVE_TRIDI
#if defined(WITH_NVIDIA_NCCL) || defined(WITH_AMD_RCCL)
#define USE_CCL_SOLVE_TRIDI                   
#endif

#ifdef SOLVE_TRIDI_GPU_BUILD
  subroutine solve_tridi_col_gpu_&
                                 &PRECISION_AND_SUFFIX &
                                ( obj, na_local, nev, nqoff, &
                                  d_dev, e_dev, q_dev, &
                                  ldq, nblk, matrixCols, mpi_comm_rows, wantDebug, success, max_threads )

!    subroutine solve_tridi_col_gpu_&
!    &PRECISION_AND_SUFFIX &
!      ( obj, na, nev, nqoff, d_dev, e, q_dev, ldq, nblk, matrixCols, mpi_comm_rows, useGPU, wantDebug, success, max_threads )
#else
  subroutine solve_tridi_col_cpu_&
                                 &PRECISION_AND_SUFFIX &
                                ( obj, na_local, nev, nqoff, &
                                  d, e, q, &
                                  ldq, nblk, matrixCols, mpi_comm_rows, wantDebug, success, max_threads )
#endif

   ! Solves the symmetric, tridiagonal eigenvalue problem on one processor column
   ! with the divide and conquer method.
   ! Works best if the number of processor rows is a power of 2!
      use precision
      use elpa_abstract_impl
      use elpa_mpi
      use merge_systems
      use ELPA_utilities
      use distribute_global_column
      use elpa_gpu
      !use single_problem
      use tridi_col_gpu

      use solve_tridi_col_cuda

#if defined(USE_CCL_SOLVE_TRIDI)
      use elpa_ccl_gpu
#endif
#if defined(WITH_NVIDIA_GPU_VERSION) && defined(WITH_NVTX)
      use cuda_functions ! for NVTX labels
#elif defined(WITH_AMD_GPU_VERSION) && defined(WITH_ROCTX)
      use hip_functions  ! for ROCTX labels
#endif
      use distribute_global_column_gpu
      use distribute_global_column_gpu_new
      implicit none
      class(elpa_abstract_impl_t), intent(inout) :: obj

      integer(kind=ik)              :: na_local, nev, nqoff, ldq, nblk, matrixCols, mpi_comm_rows
      real(kind=REAL_DATATYPE)      :: d(na_local), e(na_local)
#ifdef USE_ASSUMED_SIZE
#ifdef SOLVE_TRIDI_GPU_BUILD
      real(kind=REAL_DATATYPE)      :: q(ldq,matrixCols)
#else
      real(kind=REAL_DATATYPE)      :: q(ldq,*)
#endif
#else
      real(kind=REAL_DATATYPE)      :: q(ldq,matrixCols)
#endif

      integer(kind=ik), parameter   :: min_submatrix_size = 16 ! Minimum size of the submatrices to be used

      real(kind=REAL_DATATYPE), allocatable    :: qmat1(:,:), qmat2(:,:) ! qmat2 is needed since for ndiv>1, so qmat1 stays intact  
      integer(kind=ik)              :: i, n, np
      integer(kind=ik)              :: ndiv, noff, nmid, nlen, max_size
      integer(kind=ik)              :: my_prow, np_rows
      integer(kind=MPI_KIND)        :: mpierr, my_prowMPI, np_rowsMPI

      integer(kind=ik), allocatable :: limits(:), l_col(:), p_col_i(:), p_col_o(:)
      logical, intent(in)           :: wantDebug
      logical                       :: useGPU
      logical                       :: success
      integer(kind=ik)              :: istat, debug
      character(200)                :: errorMessage

      integer(kind=ik), intent(in)  :: max_threads

      integer(kind=MPI_KIND)        :: bcast_request1, bcast_request2
      logical                       :: useNonBlockingCollectivesRows
      integer(kind=c_int)           :: non_blocking_collectives, error

      integer(kind=c_intptr_t)      :: d_dev, e_dev ! shifted
      integer(kind=c_intptr_t)      :: q_dev, qtmp_dev, qmat1_dev, qmat2_dev
                                       
      type(c_ptr)                   :: limits_dev

      logical                       :: successGPU
      integer(kind=c_intptr_t), parameter        :: size_of_datatype_real = size_of_&
                                                                      &PRECISION&
                                                                      &_real

      integer(kind=c_intptr_t)      :: gpusolverHandle
      integer(kind=c_intptr_t)      :: num, my_stream, offset1, offset2

      logical                        :: useCCL
      integer(kind=c_intptr_t)       :: ccl_comm_rows, ccl_comm_cols, ccl_comm_all
      integer(kind=c_int)            :: cclDataType_real

      success = .true.

      call obj%timer%start("solve_tridi_col" // PRECISION_SUFFIX)
      
      debug = 0
      if (wantDebug) debug = 1

      useGPU = .false.
#ifdef SOLVE_TRIDI_GPU_BUILD
      useGPU = .true.
#endif

#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
#endif

      useCCL = .false.
#if defined(USE_CCL_SOLVE_TRIDI)                
      if (useGPU) then
        useCCL = obj%gpu_setup%useCCL
        ccl_comm_rows = obj%gpu_setup%ccl_comm_rows 
        ccl_comm_cols = obj%gpu_setup%ccl_comm_cols 

#if defined(DOUBLE_PRECISION)
       cclDataType_real = cclDouble                  
#elif defined(SINGLE_PRECISION)
       cclDataType_real = cclFloat
#endif
      endif
#endif

      call obj%get("nbc_row_solve_tridi", non_blocking_collectives, error)
      if (error .ne. ELPA_OK) then
        write(error_unit,*) "Problem setting option for non blocking collectives for rows in solve_tridi. Aborting..."
        success = .false.

        call obj%timer%stop("solve_tridi_col" // PRECISION_SUFFIX)
        return
      endif

      if (useGPU) then
        num = na_local * size_of_datatype_real
        successGPU = gpu_malloc(qtmp_dev, num)
        check_alloc_gpu("solve_tridi_col (qtmp_dev: ", successGPU)
      endif


      if (non_blocking_collectives .eq. 1) then
        useNonBlockingCollectivesRows = .true.
      else
        useNonBlockingCollectivesRows = .false.
      endif

      call obj%timer%start("mpi_communication")
      call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND), my_prowMPI, mpierr)
      call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND), np_rowsMPI, mpierr)

      my_prow = int(my_prowMPI,kind=c_int)
      np_rows = int(np_rowsMPI,kind=c_int)
      call obj%timer%stop("mpi_communication")
      success = .true.
      ! Calculate the number of subdivisions needed.

      n = na_local
      ndiv = 1
      do while(2*ndiv<=np_rows .and. n>2*min_submatrix_size)
        n = ((n+3)/4)*2 ! the bigger one of the two halves, we want EVEN boundaries
        ndiv = ndiv*2
      enddo

      ! If there is only 1 processor row and not all eigenvectors are needed
      ! and the matrix size is big enough, then use 2 subdivisions
      ! so that merge_systems is called once and only the needed
      ! eigenvectors are calculated for the final problem.
      
      ! In the current implementation, GPU codebranch doesn't have special handling for np_rows=1. So we switch it off here
      if (.not. useGPU .and. np_rows==1 .and. nev<na_local .and. na_local>2*min_submatrix_size) ndiv = 2

      allocate(limits(0:ndiv), stat=istat, errmsg=errorMessage)
      check_deallocate("solve_tridi_col: limits", istat, errorMessage)

      limits(0) = 0
      limits(ndiv) = na_local

      n = ndiv
      do while(n>1)
        n = n/2 ! n is always a power of 2
        do i=0,ndiv-1,2*n
          ! We want to have even boundaries (for cache line alignments)
          limits(i+n) = limits(i) + ((limits(i+2*n)-limits(i)+3)/4)*2
        enddo
      enddo

      ! Calculate the maximum size of a subproblem

      max_size = 0
      do i=1,ndiv
        max_size = MAX(max_size,limits(i)-limits(i-1))
      enddo

      ! Subdivide matrix by subtracting rank 1 modifications

      if (useGPU) then
        num = (ndiv) * size_of_int
        successGPU = gpu_malloc(limits_dev, num)
        check_alloc_gpu("solve_tridi_col limits_dev: ", successGPU)

        num = (ndiv) * size_of_int
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(limits_dev, int(loc(limits(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
#else
        successGPU = gpu_memcpy      (limits_dev, int(loc(limits(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
#endif
        check_memcpy_gpu("solve_tridi_col: limits_dev", successGPU)
        
        call GPU_UPDATE_D_PRECISION (limits_dev, d_dev, e_dev, ndiv, na_local, my_stream)
        if (wantDebug) successGPU = gpu_DeviceSynchronize()

        successGPU = gpu_free(limits_dev)
        check_dealloc_gpu("solve_tridi_col: limits_dev", successGPU)


      else
        do i=1,ndiv-1
          n = limits(i)
          d(n) = d(n)-abs(e(n))
          d(n+1) = d(n+1)-abs(e(n))
        enddo
      endif

      if (np_rows==1 .and. .not. useGPU) then
        ! For 1 processor row there may be 1 or 2 subdivisions
        do n=0,ndiv-1
          noff = limits(n)        ! Start of subproblem
          nlen = limits(n+1)-noff ! Size of subproblem

          ! if (useGPU) then
          !   NVTX_RANGE_PUSH("solve_tridi_single_problem_gpu")
          !   call solve_tridi_single_problem_gpu_&
          !   &PRECISION_AND_SUFFIX &
          !                             (obj, nlen, d_dev + (noff+1-1)*size_of_datatype_real, &
          !                                         e_dev + (noff+1-1)*size_of_datatype_real, &
          !                                         q_dev + ((nqoff+noff+1-1)+ldq*(noff+1-1))*size_of_datatype_real, &
          !                               ldq, qtmp_dev, wantDebug, success)
          !   NVTX_RANGE_POP("solve_tridi_single_problem_gpu")
          ! else ! useGPU
          
            call solve_tridi_single_problem_cpu_&
            &PRECISION_AND_SUFFIX &
                                    (obj, nlen,d(noff+1),e(noff+1), &
                                      q(nqoff+noff+1,noff+1), ldq, wantDebug, success)
          ! endif ! useGPU

          if (.not.(success)) then
               print *,"solve_tridi single failed"  
            call obj%timer%stop("solve_tridi_col" // PRECISION_SUFFIX)
            return
          endif
        enddo ! n=0,ndiv-1

      else ! (np_rows == 1 and. .not. useGPU)

        ! Solve sub problems in parallel with solve_tridi_single
        ! There is at maximum 1 subproblem per processor

        allocate(qmat1(max_size,max_size), stat=istat, errmsg=errorMessage)
        check_deallocate("solve_tridi_col: qmat1", istat, errorMessage)

        allocate(qmat2(max_size,max_size), stat=istat, errmsg=errorMessage)
        check_deallocate("solve_tridi_col: qmat2", istat, errorMessage)

        if (.not. useGPU) then
          qmat1 = 0 ! Make sure that all elements are defined
        endif
        
        if (useGPU) then
          num = max_size*max_size * size_of_datatype_real
          successGPU = gpu_malloc(qmat1_dev, num)
          check_alloc_gpu("solve_tridi_single qmat1_dev: ", successGPU)

          num = max_size*max_size * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
          successGPU = gpu_memset_async(qmat1_dev , 0, num, my_stream)
#else
          successGPU = gpu_memset      (qmat1_dev , 0, num)
#endif
          check_memset_gpu("solve_tridi_col: 1 qmat1_dev", successGPU)
        endif

        if (my_prow < ndiv) then

          noff = limits(my_prow)        ! Start of subproblem
          nlen = limits(my_prow+1)-noff ! Size of subproblem

          if (useGPU) then
            NVTX_RANGE_PUSH("solve_tridi_single_problem_gpu")
            call solve_tridi_single_problem_gpu_&
            &PRECISION_AND_SUFFIX &
                                    (obj, nlen, d_dev + (noff+1-1)*size_of_datatype_real, &
                                                e_dev + (noff+1-1)*size_of_datatype_real, &
                                                        qmat1_dev, &
                                      max_size, qtmp_dev, wantDebug, success)
            NVTX_RANGE_POP("solve_tridi_single_problem_gpu")
          else
            call solve_tridi_single_problem_cpu_&
            &PRECISION_AND_SUFFIX &
                                      (obj, nlen,d(noff+1),e(noff+1), qmat1, &
                                       max_size, wantDebug, success)
          endif

          if (.not.(success)) then
            print *,"solve_tridie single 2 failed"
            call obj%timer%stop("solve_tridi_col" // PRECISION_SUFFIX)
            return
          endif
        endif ! (my_prow < ndiv)


        if (useGPU) then
          num = (max_size*max_size) * size_of_datatype_real
          successGPU = gpu_malloc(qmat2_dev, num)
          check_alloc_gpu("solve_tridi_col qmat2_dev: ", successGPU)
        endif


        do np = 0, ndiv-1

          noff = limits(np)
          nlen = limits(np+1)-noff

#ifdef WITH_MPI
          if (useGPU .and. .not. useCCL) then
            num = (na_local) * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
            successGPU = gpu_memcpy_async(int(loc(d(1)),kind=c_intptr_t), d_dev, num, gpuMemcpyDeviceToHost, my_stream)
#else
            successGPU = gpu_memcpy      (int(loc(d(1)),kind=c_intptr_t), d_dev, num, gpuMemcpyDeviceToHost)
#endif
            check_memcpy_gpu("solve_tridi_col: d_dev", successGPU)

            num = (max_size*max_size) * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
            successGPU = gpu_memcpy_async(int(loc(qmat1(1,1)),kind=c_intptr_t), qmat1_dev, num, gpuMemcpyDeviceToHost, my_stream)
#else
            successGPU = gpu_memcpy      (int(loc(qmat1(1,1)),kind=c_intptr_t), qmat1_dev, num, gpuMemcpyDeviceToHost)
#endif
            check_memcpy_gpu("solve_tridi_col: qmat1_dev", successGPU)

#ifdef WITH_GPU_STREAMS
            successGPU = gpu_stream_synchronize(my_stream)
            check_stream_synchronize_gpu("solve_tridi_col: d_dev -> d", successGPU)
#endif
          endif ! (useGPU .and. .not. useCCL)

          if (useCCL) then
#if defined(USE_CCL_SOLVE_TRIDI)
            successGPU = ccl_bcast(d_dev + (noff+1-1) * size_of_datatype_real,   &
                                  d_dev + (noff+1-1) * size_of_datatype_real,  & 
                                  int(nlen,kind=c_size_t), cclDataType_real, &
                                  int(np,kind=c_int), ccl_comm_rows, my_stream)

            if (.not. successGPU) then
              print *,"Error in ccl_bcast"
              stop 1
            endif

            successGPU = gpu_stream_synchronize(my_stream)
            check_stream_synchronize_gpu("solve_tridi_col: ccl_bcast", successGPU)

            call GPU_COPY_QMAT1_TO_QMAT2_PRECISION (qmat1_dev, qmat2_dev, max_size, my_stream)
      
            successGPU = ccl_bcast(qmat1_dev, qmat2_dev, int(max_size*max_size,kind=c_size_t), cclDataType_real, &
                                  int(np,kind=c_int), ccl_comm_rows, my_stream)

            if (.not. successGPU) then
              print *,"Error in ccl_bcast"
              stop 1
            endif

            successGPU = gpu_stream_synchronize(my_stream)
            check_stream_synchronize_gpu("solve_tridi_col: ccl_bcast", successGPU)
#endif /* defined(USE_CCL_SOLVE_TRIDI) */
          else ! useCCL
            if (useNonBlockingCollectivesRows) then
              call obj%timer%start("mpi_nbc_communication")
              call mpi_ibcast(d(noff+1), int(nlen,kind=MPI_KIND), MPI_REAL_PRECISION, int(np,kind=MPI_KIND), &
                          int(mpi_comm_rows,kind=MPI_KIND), bcast_request1, mpierr)
              call mpi_wait(bcast_request1, MPI_STATUS_IGNORE, mpierr)

              qmat2 = qmat1
              call mpi_ibcast(qmat2, int(max_size*max_size,kind=MPI_KIND), MPI_REAL_PRECISION, int(np,kind=MPI_KIND), &
                          int(mpi_comm_rows,kind=MPI_KIND), bcast_request2, mpierr)
              call mpi_wait(bcast_request2, MPI_STATUS_IGNORE, mpierr)
              call obj%timer%stop("mpi_nbc_communication")
            else
              call obj%timer%start("mpi_communication")
              call mpi_bcast(d(noff+1), int(nlen,kind=MPI_KIND), MPI_REAL_PRECISION, int(np,kind=MPI_KIND), &
                          int(mpi_comm_rows,kind=MPI_KIND), mpierr)

              qmat2 = qmat1
              call mpi_bcast(qmat2, int(max_size*max_size,kind=MPI_KIND), MPI_REAL_PRECISION, int(np,kind=MPI_KIND), &
                          int(mpi_comm_rows,kind=MPI_KIND), mpierr)
              call obj%timer%stop("mpi_communication")
            endif
          endif ! useCCL

          if (useGPU .and. .not. useCCL) then
            num = (na_local) * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
            successGPU = gpu_memcpy_async(d_dev, int(loc(d(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
#else
            successGPU = gpu_memcpy      (d_dev, int(loc(d(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
#endif
            check_memcpy_gpu("solve_tridi_col: d_dev", successGPU)

            num = (max_size*max_size) * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
            successGPU = gpu_memcpy_async(qmat2_dev, int(loc(qmat2(1,1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
#else
            successGPU = gpu_memcpy      (qmat2_dev, int(loc(qmat2(1,1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
#endif
            check_memcpy_gpu("solve_tridi_col: qmat2_dev", successGPU)
          endif ! (useGPU .and. .not. useCCL)

#else /* WITH_MPI */

! what to do here??
!          qmat2 = qmat1 ! is this correct

#endif /* WITH_MPI */




#ifdef WITH_MPI
          if (useGPU) then
            call gpu_distribute_global_column(PRECISION_CHAR, qmat2_dev, q_dev, max_size, max_size, ldq, matrixCols, &
                                              noff, nqoff+noff, nlen, my_prow, np_rows, nblk, debug, my_stream)
          else
            do i=1,nlen
              call distribute_global_column_4_&
              &PRECISION &
                    (obj, qmat2(1:max_size,1:max_size), q(1:ldq,1:matrixCols), max_size, max_size, ldq, matrixCols, &
                      noff, nqoff+noff, nlen, my_prow, np_rows, nblk)
            enddo ! i=1,nlen
          endif
#else /* WITH_MPI */
          if (useGPU) then
            call gpu_distribute_global_column(PRECISION_CHAR, qmat1_dev, q_dev, max_size, max_size, ldq, matrixCols, &
                                              noff, nqoff+noff, nlen, my_prow, np_rows, nblk, debug, my_stream)
          else
            do i=1,nlen
                  call distribute_global_column_4_&
                  &PRECISION &
                      (obj, qmat1(1:max_size,1:max_size), q(1:ldq,1:matrixCols), max_size, max_size, ldq, matrixCols, &
                        noff, nqoff+noff, nlen, my_prow, np_rows, nblk)

            enddo ! i=1,nlen
          endif
#endif /* WITH_MPI */

          ! assume d is on the host
        enddo ! np = 0, ndiv-1

        if (useGPU) then
          successGPU = gpu_free(qmat1_dev)
          check_dealloc_gpu("solve_tridi_col: qmat1_dev", successGPU)

          successGPU = gpu_free(qmat2_dev)
          check_dealloc_gpu("solve_tridi_col: qmat2_dev", successGPU)
        endif


        deallocate(qmat1, qmat2, stat=istat, errmsg=errorMessage)
        check_deallocate("solve_tridi_col: qmat1, qmat2", istat, errorMessage)

      endif ! (np_rows == 1 and. .not. useGPU)


      ! Allocate and set index arrays l_col and p_col

      allocate(l_col(na_local), p_col_i(na_local),  p_col_o(na_local), stat=istat, errmsg=errorMessage)
      check_deallocate("solve_tridi_col: l_col, p_col_i, p_col_o", istat, errorMessage)

      do i=1,na_local
        l_col(i) = i
        p_col_i(i) = 0
        p_col_o(i) = 0
      enddo

      ! assume that d is on the host !!!


      if (useGPU) then
        num = na_local * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(int(loc(e(1)),kind=c_intptr_t), e_dev, num, gpuMemcpyDeviceToHost, my_stream)
#else
        successGPU = gpu_memcpy      (int(loc(e(1)),kind=c_intptr_t), e_dev, num, gpuMemcpyDeviceToHost)
#endif
        check_memcpy_gpu("solve_tridi_col: e_dev2", successGPU)
        
        num = na_local * size_of_datatype_real
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(int(loc(d(1)),kind=c_intptr_t), d_dev, num, gpuMemcpyDeviceToHost, my_stream)
#else
        successGPU = gpu_memcpy      (int(loc(d(1)),kind=c_intptr_t), d_dev, num, gpuMemcpyDeviceToHost)
#endif
        check_memcpy_gpu("solve_tridi_col: d_dev", successGPU)

#ifdef WITH_GPU_STREAMS
        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("solve_tridi_col: d_dev -> d", successGPU)
#endif
      endif

      ! Merge subproblems
      n = 1
      do while(n<ndiv) ! if ndiv==1, the problem was solved by single call to solve_tridi_single

        do i=0,ndiv-1,2*n

          noff = limits(i)
          nmid = limits(i+n) - noff
          nlen = limits(i+2*n) - noff

          if (nlen == na_local) then
            ! Last merge, set p_col_o=-1 for unneeded (output) eigenvectors
            p_col_o(nev+1:na_local) = -1
          endif
          if (useGPU) then
            NVTX_RANGE_PUSH("merge_systems_gpu")
            call merge_systems_gpu_&
            &PRECISION &
                                (obj, nlen, nmid, d(noff+1), e(noff+nmid), q_dev, ldq, nqoff+noff, nblk, &
                                 matrixCols, int(mpi_comm_rows,kind=ik), int(mpi_comm_self,kind=ik), &
                                 l_col(noff+1), p_col_i(noff+1), &
                                 l_col(noff+1), p_col_o(noff+1), 0, 1, useGPU, wantDebug, success, max_threads)
            NVTX_RANGE_POP("merge_systems_gpu")
          else
            call merge_systems_cpu_&
            &PRECISION &
                                (obj, nlen, nmid, d(noff+1), e(noff+nmid), q, ldq, nqoff+noff, nblk, &
                                 matrixCols, int(mpi_comm_rows,kind=ik), int(mpi_comm_self,kind=ik), &
                                 l_col(noff+1), p_col_i(noff+1), &
                                 l_col(noff+1), p_col_o(noff+1), 0, 1, useGPU, wantDebug, success, max_threads)
          endif

          if (.not.(success)) then
            print *,"returning early from merge_systems"
            call obj%timer%stop("solve_tridi_col" // PRECISION_SUFFIX)
            return
          endif
        enddo ! i=0,ndiv-1,2*n

        n = 2*n

      enddo ! do while

      if (useGPU) then
        num = (na_local)* size_of_datatype_real
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(d_dev, int(loc(d(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
#else
        successGPU = gpu_memcpy      (d_dev, int(loc(d(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
#endif
        check_memcpy_gpu("solve_tridi_col: d_dev", successGPU)
      endif


      deallocate(limits, l_col, p_col_i, p_col_o, stat=istat, errmsg=errorMessage)
      check_deallocate("solve_tridi_col: limits, l_col, p_col_i, p_col_o", istat, errorMessage)


      if (useGPU) then
        num = (na_local)* size_of_datatype_real
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(d_dev, int(loc(d(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
#else
        successGPU = gpu_memcpy(d_dev, int(loc(d(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
#endif
        check_memcpy_gpu("solve_tridi_col: d_dev", successGPU)

        num = (na_local)* size_of_datatype_real
#ifdef WITH_GPU_STREAMS
        successGPU = gpu_memcpy_async(e_dev, int(loc(e(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
#else
        successGPU = gpu_memcpy      (e_dev, int(loc(e(1)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
#endif
        check_memcpy_gpu("solve_tridi_col: e_dev 3", successGPU)

        successGPU = gpu_free(qtmp_dev)
        check_dealloc_gpu("solve_tridi_col: qtmp_dev", successGPU)
      endif ! useGPU


      call obj%timer%stop("solve_tridi_col" // PRECISION_SUFFIX)

    end

!#include "./solve_tridi_single_problem_template.F90"

