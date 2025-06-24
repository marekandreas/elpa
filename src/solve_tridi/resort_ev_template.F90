#include "../general/error_checking.inc"

#ifdef SOLVE_TRIDI_GPU_BUILD
  subroutine resort_ev_gpu_&
                           &PRECISION&
                           &(obj, idx_ev, nLength, na, p_col_out, q_dev, ldq, matrixCols, l_rows, l_rqe, l_rqs, &
                           mpi_comm_cols_self, p_col, l_col, l_col_out)
#else
  subroutine resort_ev_cpu_&
                           &PRECISION&
                           &(obj, idx_ev, nLength, na, p_col_out, q,     ldq, matrixCols, l_rows, l_rqe, l_rqs, &
                           mpi_comm_cols_self, p_col, l_col, l_col_out)
#endif
    use precision
#ifdef WITH_OPENMP_TRADITIONAL
    use elpa_omp
#endif
    use elpa_mpi
    use ELPA_utilities
    use elpa_abstract_impl
    use elpa_gpu
    use elpa_ccl_gpu
    implicit none
    class(elpa_abstract_impl_t), intent(inout) :: obj
    integer(kind=ik), intent(in)               :: nLength, na
    integer(kind=ik), intent(in)               :: ldq, matrixCols, l_rows, l_rqe, l_rqs
    integer(kind=ik), intent(in)               :: mpi_comm_cols_self
    integer(kind=ik), intent(in)               :: p_col(na), l_col(na), l_col_out(na)
#ifdef WITH_MPI
    integer(kind=MPI_KIND)                     :: mpierrMPI, my_pcolMPI
    integer(kind=ik)                           :: mpierr
#endif
    integer(kind=ik)                           :: my_pcol

    integer(kind=c_intptr_t)                   :: q_dev, qtmp_dev

#if defined(USE_ASSUMED_SIZE) && !defined(SOLVE_TRIDI_GPU_BUILD)
    real(kind=REAL_DATATYPE)                   :: q(ldq,*)
#else
    real(kind=REAL_DATATYPE)                   :: q(ldq,matrixCols)
#endif

    integer(kind=ik), intent(in)               :: p_col_out(na)
    integer(kind=ik)                           :: idx_ev(nLength)
    integer(kind=ik)                           :: i, nc, pc1, pc2, lc1, lc2, l_cols_out

    real(kind=REAL_DATATYPE), allocatable      :: qtmp(:,:)
    integer(kind=ik)                           :: istat
    character(200)                             :: errorMessage

    logical                                    :: useGPU, useCCL
    logical                                    :: successGPU
    integer(kind=c_intptr_t)                   :: ccl_comm_rows, ccl_comm_cols_self
    integer(kind=c_int)                        :: cclDataType
    integer(kind=c_intptr_t)                   :: my_stream
    integer(kind=c_intptr_t)                   :: num
    integer(kind=c_intptr_t), parameter        :: size_of_datatype = size_of_&
                                                                    &PRECISION&
                                                                    &_real
    if (l_rows==0) return ! My processor column has no work to do

#ifdef WITH_MPI
    call mpi_comm_rank(int(mpi_comm_cols_self,kind=MPI_KIND), my_pcolMPI, mpierr)
    my_pcol = int(my_pcolMPI,kind=c_int)
#endif

#ifdef SOLVE_TRIDI_GPU_BUILD
    useGPU = .true.
#else
    useGPU = .false.
#endif
    
    useCCL = obj%gpu_setup%useCCL

    if (useGPU) then
#ifdef WITH_GPU_STREAMS
      my_stream = obj%gpu_setup%my_stream
#endif
      
      if (useCCL) then
        my_stream = obj%gpu_setup%my_stream
        ccl_comm_rows = obj%gpu_setup%ccl_comm_rows
        ccl_comm_cols_self = obj%gpu_setup%ccl_comm_cols
        if (mpi_comm_cols_self==mpi_comm_self) then
          ccl_comm_cols_self = obj%gpu_setup%ccl_comm_self
        endif

#if defined(DOUBLE_PRECISION)
        cclDataType = cclDouble
#endif      
#if defined(SINGLE_PRECISION)
        cclDataType = cclFloat
#endif
      endif ! useCCL
    endif ! useGPU

    ! Resorts eigenvectors so that q_new(:,i) = q_old(:,idx_ev(i))

    l_cols_out = COUNT(p_col_out(1:na)==my_pcol)

    if (useGPU) then
      num = (l_rows*l_cols_out) * size_of_datatype
      successGPU = gpu_malloc(qtmp_dev, num)
      check_alloc_gpu("resort_ev: qtmp_dev", successGPU)
    endif

    if (.not. useCCL) then
      allocate(qtmp(l_rows,l_cols_out), stat=istat, errmsg=errorMessage)
      check_allocate("resort_ev: qtmp",istat, errorMessage)
    endif

    if (useCCL) then
      successGPU = ccl_group_start()
      if (.not. successGPU) then
        print *, "resort_ev: Error in setting up ccl_group_start!"
        stop 1
      endif
    endif

    nc = 0

    do i=1,na

      pc1 = p_col(idx_ev(i))
      lc1 = l_col(idx_ev(i))
      pc2 = p_col_out(i)

      if (pc2<0) cycle ! This column is not needed in output

      if (pc2==my_pcol) nc = nc+1 ! Counter for output columns

      if (pc1==my_pcol) then
        if (pc2==my_pcol) then
          ! send and recieve column are local
          if (useGPU) then
            num = l_rows*size_of_datatype
#ifdef WITH_GPU_STREAMS
            my_stream = obj%gpu_setup%my_stream
            successGPU = gpu_memcpy_async(qtmp_dev+(nc-1)*l_rows*size_of_datatype, &
                                          q_dev+(l_rqs-1+(lc1-1)*ldq)*size_of_datatype, num, gpuMemcpyDeviceToDevice, my_stream)
            check_memcpy_gpu("resort_ev: qtmp_dev <- q_dev", successGPU)
#else
            successGPU = gpu_memcpy(qtmp_dev+(nc-1)*l_rows*size_of_datatype, &
                                    q_dev+(l_rqs-1+(lc1-1)*ldq)*size_of_datatype, num, gpuMemcpyDeviceToDevice)
            check_memcpy_gpu("resort_ev: qtmp_dev <- q_dev", successGPU)
#endif      
          else
            qtmp(1:l_rows,nc) = q(l_rqs:l_rqe,lc1)
          endif
        else
#ifdef WITH_MPI
          if (useGPU .and. .not. useCCL) then
            num = l_rows*size_of_datatype
#ifdef WITH_GPU_STREAMS
            my_stream = obj%gpu_setup%my_stream
            successGPU = gpu_memcpy_async(int(loc(q(l_rqs,lc1)),kind=c_intptr_t), &
                                          q_dev+(l_rqs-1+(lc1-1)*ldq)*size_of_datatype, num, gpuMemcpyDeviceToHost, my_stream)
            check_memcpy_gpu("resort_ev: q <- q_dev", successGPU)

            successGPU = gpu_stream_synchronize(my_stream)
            check_stream_synchronize_gpu("resort_ev: q <- q_dev", successGPU)
#else
            successGPU = gpu_memcpy(int(loc(q(l_rqs,lc1)),kind=c_intptr_t), &
                                    q_dev+(l_rqs-1+(lc1-1)*ldq)*size_of_datatype, num, gpuMemcpyDeviceToHost)
            check_memcpy_gpu("resort_ev: q <- q_dev", successGPU)
#endif
          endif ! (useGPU .and. .not. useCCL)

          if (useCCL) then
            successGPU = ccl_Send(q_dev+(l_rqs-1+(lc1-1)*ldq)*size_of_datatype, int(l_rows,kind=c_size_t), &
                                  cclDataType, pc2, ccl_comm_cols_self, my_stream)
          else
            call obj%timer%start("mpi_communication")
            call mpi_send(q(l_rqs,lc1), int(l_rows,kind=MPI_KIND), MPI_REAL_PRECISION, pc2, int(mod(i,4096),kind=MPI_KIND), &
                                int(mpi_comm_cols_self,kind=MPI_KIND), mpierr)
            call obj%timer%stop("mpi_communication")
          endif
#else /* WITH_MPI */
#endif /* WITH_MPI */
        endif
      else if (pc2==my_pcol) then
#ifdef WITH_MPI
        if (useCCL) then
          successGPU = ccl_Recv(qtmp_dev+(nc-1)*l_rows*size_of_datatype, int(l_rows,kind=c_size_t), &
                                cclDataType, pc1, ccl_comm_cols_self, my_stream)
        else
          call obj%timer%start("mpi_communication")
          call mpi_recv(qtmp(1,nc), int(l_rows,kind=MPI_KIND), MPI_REAL_PRECISION, pc1, int(mod(i,4096),kind=MPI_KIND), &
                        int(mpi_comm_cols_self,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
          call obj%timer%stop("mpi_communication")
        endif

        if (useGPU .and. .not. useCCL) then
          num = l_rows*size_of_datatype
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_memcpy_async(qtmp_dev+(nc-1)*l_rows*size_of_datatype, &
                                        int(loc(qtmp(1,nc)),kind=c_intptr_t), num, gpuMemcpyHostToDevice, my_stream)
          check_memcpy_gpu("resort_ev: qtmp_dev <- qtmp", successGPU)
#else
          successGPU = gpu_memcpy      (qtmp_dev+(nc-1)*l_rows*size_of_datatype, &
                                        int(loc(qtmp(1,nc)),kind=c_intptr_t), num, gpuMemcpyHostToDevice)
          check_memcpy_gpu("resort_ev: qtmp_dev <- qtmp", successGPU)
#endif
        endif ! (useGPU .and. .not. useCCL)

#else /* WITH_MPI */
        if (useGPU) then
          num = l_rows*size_of_datatype
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_memcpy_async(qtmp_dev+(nc-1)*l_rows*size_of_datatype, &
                                        q_dev+(l_rqs-1+(lc1-1)*ldq)*size_of_datatype, num, gpuMemcpyDeviceToDevice, my_stream)
          check_memcpy_gpu("resort_ev: qtmp_dev <- q_dev", successGPU)

          successGPU = gpu_stream_synchronize(my_stream)
          check_stream_synchronize_gpu("resort_ev: qtmp_dev <- q_dev", successGPU)
#else
          successGPU = gpu_memcpy      (qtmp_dev+(nc-1)*l_rows*size_of_datatype, &
                                        q_dev+(l_rqs-1+(lc1-1)*ldq)*size_of_datatype, num, gpuMemcpyDeviceToDevice)
          check_memcpy_gpu("resort_ev: qtmp_dev <- q_dev", successGPU)
#endif
        else
          qtmp(1:l_rows,nc) = q(l_rqs:l_rqe,lc1)
        endif
#endif /* WITH_MPI */
      endif
    enddo

    if (useCCL) then
      successGPU = ccl_group_end()
      if (.not. successGPU) then
        print *, "resort_ev: Error in setting up ccl_group_end!"
        stop 1
      endif
    endif

    ! Insert qtmp into (output) q

    nc = 0

    do i=1,na

      pc2 = p_col_out(i)
      lc2 = l_col_out(i)

      if (pc2==my_pcol) then
        nc = nc+1

        if (useGPU) then
          num = l_rows*size_of_datatype
#ifdef WITH_GPU_STREAMS
          my_stream = obj%gpu_setup%my_stream
          successGPU = gpu_memcpy_async(q_dev+(l_rqs-1+(lc2-1)*ldq)*size_of_datatype, &
                                        qtmp_dev+(nc-1)*l_rows*size_of_datatype, num, gpuMemcpyDeviceToDevice, my_stream)
          check_memcpy_gpu("resort_ev: q_dev <- qtmp_dev", successGPU)
#else
          successGPU = gpu_memcpy(q_dev+(l_rqs-1+(lc2-1)*ldq)*size_of_datatype, &
                                  qtmp_dev+(nc-1)*l_rows*size_of_datatype, num, gpuMemcpyDeviceToDevice)
          check_memcpy_gpu("resort_ev: q_dev <- qtmp_dev", successGPU)
#endif
        else  
          q(l_rqs:l_rqe,lc2) = qtmp(1:l_rows,nc)
        endif
      endif
    enddo

    if (useGPU) then
      successGPU = gpu_free(qtmp_dev)
      check_dealloc_gpu("resort_ev: qtmp_dev", successGPU)
    endif
    
    if (.not. useCCL) then
      deallocate(qtmp, stat=istat, errmsg=errorMessage)
      check_deallocate("resort_ev: qtmp",istat, errorMessage)
    endif

  end subroutine

