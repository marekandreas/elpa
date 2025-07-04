#define NOT_OK_OUCH block; if (.not. ok) then; print *, "Error at line ", __FILE__, __LINE__; stop; end if; end block


module oneccl_tests
  implicit none
contains
  subroutine test_sanity(num_elements)
    use, intrinsic :: iso_c_binding
    use omp_lib, only: omp_get_wtime
    use elpa_gpu
    use sycl_functions
    implicit none

    double precision t_begin, t_end
    double precision init_time, transfer_time, compute_time
    integer(kind=c_intptr_t) :: num_elements, num_bytes
    integer(kind=c_intptr_t) :: gpu_buffer, host_ptr
    integer(kind=c_int) :: ok, n_gpu
    real(kind=c_double), dimension(:), allocatable :: host_buffer

    init_time = t_end - t_begin
    ok = sycl_getdevicecount(n_gpu)

    if(ok == 0) then
      write(*,"(2X,A)") "Error: gpu_getdevicecount"
      stop
    end if
    if(n_gpu > 0) then
      ok = gpu_setdevice(0)
      if(ok == 0) then
        write(*,"(2X,A)") "Error: gpu_setdevice"
        stop
      end if
    else
      write(*,"(2X,A)") "Error: no GPU"
      stop
    end if

    print *, "found ", n_gpu, " GPUs"

    allocate(host_buffer(num_elements))
    num_bytes = num_elements * c_double
    host_ptr = int(loc(host_buffer),c_intptr_t)
    ok = gpu_malloc(gpu_buffer,num_bytes)
    ok = gpu_memcpy(gpu_buffer,host_ptr,num_bytes,gpuMemcpyHostToDevice)
    t_end = omp_get_wtime()
    transfer_time = t_end - t_begin
    print *, "transfer time: ", transfer_time

    t_begin = omp_get_wtime()
    ok = gpu_free(gpu_buffer)
    t_end = omp_get_wtime()

    transfer_time = transfer_time + (t_end - t_begin)
    deallocate(host_buffer)
    print *, "back-transfer time: ", transfer_time
  end subroutine test_sanity



  subroutine get_mpi_local_ranks(ierr, n_local_ranks)
    use, intrinsic :: iso_c_binding
    use mpi
    implicit none
    integer(kind=c_int), intent(out) :: ierr
    integer(kind=c_int), intent(out) :: n_local_ranks
    integer(kind=c_int) :: n_ranks, mpi_local_comm

    call mpi_comm_size(mpi_comm_world, n_ranks, ierr)
    if (ierr /= MPI_SUCCESS) then
      print *, "Error getting MPI size"
      n_local_ranks = -1
      return
    end if 
    call mpi_comm_split_type(mpi_comm_world, mpi_comm_type_shared, 0, mpi_info_null, mpi_local_comm, ierr)
    if (ierr /= MPI_SUCCESS) then
      print *, "Error splitting MPI communicator"
      n_local_ranks = -1
      return
    end if
    call mpi_comm_size(mpi_local_comm, n_local_ranks, ierr)
    if (ierr /= MPI_SUCCESS) then
      print *, "Error getting local MPI size"
      n_local_ranks = -1
      return
    end if
    call mpi_comm_free(mpi_local_comm, ierr)
    if (ierr /= MPI_SUCCESS) then
      print *, "Error freeing local MPI communicator"
      n_local_ranks = -1
      return
    end if
  end subroutine get_mpi_local_ranks



  subroutine test_ccl_setup(ccl_unique_id_val, ccl_comm, my_stream)
    use, intrinsic :: iso_c_binding
    use mpi
    use elpa_gpu
    use elpa_ccl_gpu
    use sycl_functions
    implicit none

    integer(kind=c_intptr_t), intent(out) :: ccl_comm
    integer(kind=c_intptr_t), intent(out) :: my_stream
    
    type(onecclUniqueId), intent(inout) :: ccl_unique_id_val
    integer(kind=c_int) :: my_rank, my_gpu
    integer(kind=c_int) :: n_local_ranks, n_ranks, n_local_gpus, ierr, ok
    integer :: mpi_local_comm

    call mpi_comm_rank(mpi_comm_world, my_rank, ierr)
    call mpi_comm_size(mpi_comm_world, n_ranks, ierr)

    print *, "Rank ", my_rank, " of ", n_ranks, " is starting the CCL test"

    call get_mpi_local_ranks(ierr, n_local_ranks)

    ok = sycl_state_initialize(0)
    if (n_local_ranks < 0) then
      write(*,"(2X,A)") "Error getting local ranks"
      stop
    else if (my_rank == 0) then
      print *, "found nLocalRanks: ", n_local_ranks
    end if
    ok = gpu_vendor_internal()
    if (ok == 0) then
      write(*,"(2X,A)") "Error: gpu_vendor_internal"
      stop
    end if

    call set_gpu_parameters()
    ok = sycl_getdevicecount(n_local_gpus)
    if (ok == 0 .or. n_local_gpus == 0) then
      write(*,"(2X,A)") "Issue with the GPUs. Ok:", ok, "nGpus:", n_local_gpus
      stop
    end if

    print *, "Found nLocalGpus: ", n_local_gpus
    my_gpu = mod(mod(my_rank, n_local_ranks), n_local_gpus)
    ok = gpu_setdevice(my_gpu)
    if (ok == 0) then
      write(*,"(2X,A)") "Error: gpu_setdevice"
      stop
    end if
    print *, "Rank", my_rank, "using GPU: ", my_gpu
    ok = sycl_stream_create(my_stream)
    if (ok == 0) then
      write(*,"(2X,A)") "Error: sycl_stream_create"
      stop
    end if
    print *, "Rank", my_rank, "created SYCL stream: ", my_stream

    print *, "Rank", my_rank, "Found nGpus: ", n_local_gpus, "Chose GPU: ", my_gpu

    if (my_rank == 0) then
      call sycl_printdevices()
    endif

    if (my_rank == 0) then
      ok = ccl_get_unique_id(ccl_unique_id_val)
      if (ok == 0) then
        write(*,"(2X,A)") "Error: CCL_get_unique_id"
        stop
      end if
      call mpi_bcast(ccl_unique_id_val, 256, mpi_byte, 0, mpi_comm_world, ierr)
    else
      call mpi_bcast(ccl_unique_id_val, 256, mpi_byte, 0, mpi_comm_world, ierr)
    end if
    ok = ccl_comm_init_rank(ccl_comm, n_ranks, ccl_unique_id_val, my_rank)
    if (ok == 0) then
      write(*,"(2X,A)") "Error: CCL_comm_init_rank"
      stop
    end if
    print *, "init success"
  end subroutine test_ccl_setup



  subroutine test_ccl_cleanup(ccl_comm, my_stream)
    use, intrinsic :: iso_c_binding
    use elpa_gpu
    use elpa_ccl_gpu
    use sycl_functions
    implicit none

    integer(kind=c_intptr_t), intent(in) :: ccl_comm
    integer(kind=c_intptr_t), intent(in) :: my_stream
    integer(kind=c_int) :: ok

    ok = ccl_comm_destroy(ccl_comm)
    if  (ok == 0) then
      write(*,"(2X,A)") "Error: CCL_comm_destroy"
      stop
    else 
      print *, "CCL communicator destroyed successfully"
    end if
  end subroutine test_ccl_cleanup


  subroutine verify_allreduce_result(result_data, n_ranks, reduction_operation, is_correct)
    use, intrinsic :: iso_c_binding
    use elpa_ccl_gpu
    use elpa_gpu
    implicit none

    real(kind=c_double), intent(inout) :: result_data(:)
    integer(kind=c_int), intent(in), value :: n_ranks, reduction_operation
    logical, intent(out) :: is_correct
    integer(kind=c_int) :: i
    real(kind=c_double) :: expected_value

    expected_value = 0
    is_correct = .true.
    if (reduction_operation == ccl_redOp_cclSum()) then
      expected_value = 0
      do i = 0, n_ranks - 1
        expected_value = expected_value + (1000.0 + i)
      end do
    elseif (reduction_operation == ccl_redOp_cclProd()) then
      expected_value = 1
      do i = 0, n_ranks - 1
        expected_value = expected_value * (1000.0 + i)
      end do
    else if (reduction_operation == ccl_redOp_cclMax()) then
      expected_value = 1000.0 + n_ranks - 1
    else if (reduction_operation == ccl_redOp_cclMin()) then
      expected_value = 1000.0
    end if

    do i = 1, size(result_data)
      if (result_data(i) /= expected_value) then
        print *, "expected value: ", expected_value, "got", result_data(i)
        is_correct = .false.
        exit
      end if
    end do
  end subroutine verify_allreduce_result



  subroutine test_ccl_allreduce(ccl_comm, num_elements, onecclStream)
    use, intrinsic :: iso_c_binding
    use elpa_gpu
    use elpa_ccl_gpu
    use oneccl_functions
    use sycl_functions
    use mpi
    implicit none

    integer(kind=c_intptr_t), intent(in) :: ccl_comm
    integer(kind=c_intptr_t), intent(in) :: onecclStream
    integer(kind=c_intptr_t), intent(in), value :: num_elements

    integer(kind=c_size_t) :: num_ccl_elements
    integer(kind=c_int) :: n_ranks, my_rank, ierr
    integer(kind=c_int) :: i
    logical :: ok, is_correct

    real(kind=c_double) :: original_data(num_elements), result_data(num_elements)
    integer(kind=c_intptr_t) :: original_data_gpu, result_data_gpu


    call mpi_comm_rank(mpi_comm_world, my_rank, ierr)
    call mpi_comm_size(mpi_comm_world, n_ranks, ierr)

    num_ccl_elements = num_elements

    do i = 1, num_elements
      original_data(i) = 1000.0 + my_rank
    end do

    block
      character(len=20), dimension(4) :: reduction_names = &
          ['ccl_redOp_cclSum()', 'ccl_redOp_cclProd()', 'ccl_redOp_cclMax()', 'ccl_redOp_cclMin()']
      integer(kind=c_int) :: reduction_operations(4)
      integer(kind=c_int) :: j

      reduction_operations = [ccl_redOp_cclSum(), ccl_redOp_cclProd(), ccl_redOp_cclMax(), ccl_redOp_cclMin()]

      ok = sycl_malloc(original_data_gpu, num_elements * c_double)
      NOT_OK_OUCH
      ok = gpu_malloc(result_data_gpu, num_elements * c_double)
      NOT_OK_OUCH
      ok = gpu_memcpy(original_data_gpu, int(loc(original_data),c_intptr_t), num_elements * c_double, gpuMemcpyHostToDevice)
      NOT_OK_OUCH

      do j = 1, 4
        ok = gpu_memset(result_data_gpu, 0, num_elements * c_double)
        NOT_OK_OUCH
        if (my_rank == 0) then
          print *, " - test AllReduce ", reduction_names(j)
        end if
        call mpi_barrier(mpi_comm_world, ierr)
        ok = ccl_allreduce(original_data_gpu, result_data_gpu, num_ccl_elements, &
                  ccl_dataType_cclDouble(), reduction_operations(j), ccl_comm, oneCCLStream)
        NOT_OK_OUCH

        ok = gpu_memcpy(loc(result_data), result_data_gpu, num_elements * c_double, gpuMemcpyDeviceToHost)
        NOT_OK_OUCH
        if (my_rank == 0) then
          call verify_allreduce_result(result_data, n_ranks, reduction_operations(j), is_correct)
          print *, "  -> Result ", reduction_names(j), "correct? -", is_correct
        endif
        call mpi_barrier(mpi_comm_world, ierr)
      enddo
    end block

    ok = gpu_free(original_data_gpu)
    NOT_OK_OUCH
    ok = gpu_free(result_data_gpu)
    NOT_OK_OUCH
  end subroutine test_ccl_allreduce



  subroutine test_ccl_reduce(ccl_comm, num_elements, onecclStream)
    use, intrinsic :: iso_c_binding
    use elpa_gpu
    use elpa_ccl_gpu
    use oneccl_functions
    use sycl_functions
    use mpi
    implicit none

    integer(kind=c_intptr_t), intent(in) :: ccl_comm
    integer(kind=c_intptr_t), intent(in), value :: num_elements
    integer(kind=c_intptr_t), intent(in) :: onecclStream

    integer(kind=c_size_t) :: num_ccl_elements
    integer(kind=c_int) :: n_ranks, my_rank, ierr
    integer(kind=c_int) :: i
    logical :: ok, is_correct

    real(kind=c_double) :: original_data(num_elements), result_data(num_elements)
    integer(kind=c_intptr_t) :: original_data_gpu, result_data_gpu


    call mpi_comm_rank(mpi_comm_world, my_rank, ierr)
    call mpi_comm_size(mpi_comm_world, n_ranks, ierr)

    num_ccl_elements = num_elements

    do i = 1, num_elements
      original_data(i) = 1000.0 + my_rank
    end do

    block
      character(len=20), dimension(4) :: reduction_names = &
          ['ccl_redOp_cclSum()', 'ccl_redOp_cclProd()', 'ccl_redOp_cclMax()', 'ccl_redOp_cclMin()']
      integer(kind=c_int) :: reduction_operations(4)
      integer(kind=c_int) :: j
      integer(kind=c_int) :: destination_rank

      reduction_operations = [ccl_redOp_cclSum(), ccl_redOp_cclProd(), ccl_redOp_cclMax(), ccl_redOp_cclMin()]
      destination_rank = 0

      ok = sycl_malloc(original_data_gpu, num_elements * c_double); NOT_OK_OUCH
      ok = gpu_malloc(result_data_gpu, num_elements * c_double); NOT_OK_OUCH
      ok = gpu_memcpy(original_data_gpu, int(loc(original_data),c_intptr_t), num_elements * c_double, gpuMemcpyHostToDevice)
      NOT_OK_OUCH

      do j = 1, 4
        ok = gpu_memset(result_data_gpu, 0, num_elements * c_double); NOT_OK_OUCH
        if (my_rank == 0) then
          print *, " - test Reduce ", reduction_names(j)
        end if
        call mpi_barrier(mpi_comm_world, ierr)
        !  ccl_reduce_intptr(sendbuff, recvbuff, nrElements, cclDatatype, cclOp, root, cclComm, gpuStream)
        ok = ccl_reduce(original_data_gpu, result_data_gpu, num_ccl_elements, &
                  ccl_dataType_cclDouble(), reduction_operations(j), destination_rank, ccl_comm, oneCCLStream)
        NOT_OK_OUCH

        ok = gpu_memcpy(loc(result_data), result_data_gpu, num_elements * c_double, gpuMemcpyDeviceToHost); NOT_OK_OUCH
        if (my_rank == 0) then
          call verify_allreduce_result(result_data, n_ranks, reduction_operations(j), is_correct)
          print *, "  -> Result ", reduction_names(j), "correct? -", is_correct
        endif
        call mpi_barrier(mpi_comm_world, ierr)
      enddo
    end block

    ok = gpu_free(original_data_gpu)
    NOT_OK_OUCH
    ok = gpu_free(result_data_gpu)
    NOT_OK_OUCH
  end subroutine test_ccl_reduce



  subroutine test_ccl_broadcast(ccl_comm, num_elements, onecclStream)
    use, intrinsic :: iso_c_binding
    use elpa_gpu
    use elpa_ccl_gpu
    use oneccl_functions
    use sycl_functions
    use mpi
    implicit none

    integer(kind=c_intptr_t), intent(in) :: ccl_comm
    integer(kind=c_intptr_t), intent(in), value :: num_elements
    integer(kind=c_intptr_t), intent(in) :: onecclStream

    integer(kind=c_size_t) :: num_ccl_elements
    integer(kind=c_int) :: n_ranks, my_rank, ierr
    integer(kind=c_int) :: i
    integer(kind=c_int) :: destination_rank
    logical :: ok, is_correct

    real(kind=c_double) :: original_data(num_elements), result_data(num_elements)
    integer(kind=c_intptr_t) :: original_data_gpu, result_data_gpu


    call mpi_comm_rank(mpi_comm_world, my_rank, ierr)
    call mpi_comm_size(mpi_comm_world, n_ranks, ierr)

    num_ccl_elements = num_elements

    do i = 1, num_elements
      original_data(i) = 1000.0 + my_rank
    end do

    destination_rank = 0

    ok = gpu_malloc(original_data_gpu, num_elements * c_double); NOT_OK_OUCH
    ok = gpu_malloc(result_data_gpu, num_elements * c_double); NOT_OK_OUCH
    ok = gpu_memcpy(original_data_gpu, int(loc(original_data),c_intptr_t), num_elements * c_double, gpuMemcpyHostToDevice)
    NOT_OK_OUCH

    do destination_rank = 0, n_ranks-1
      ok = gpu_memset(result_data_gpu, 0, num_elements * c_double); NOT_OK_OUCH
      if (my_rank == 0) then
        print *, " - test Broadcast to rank ", destination_rank
      end if
      call mpi_barrier(mpi_comm_world, ierr)
      ! ccl_bcast_intptr(sendbuff, recvbuff, nrElements, cclDatatype, root, cclComm, gpuStream)
      ok = ccl_bcast(original_data_gpu, result_data_gpu, num_ccl_elements, &
                ccl_dataType_cclDouble(), destination_rank, ccl_comm, onecclStream)
      NOT_OK_OUCH

      ok = gpu_memcpy(loc(result_data), result_data_gpu, num_elements * c_double, gpuMemcpyDeviceToHost); NOT_OK_OUCH
      is_correct = (result_data(1) == 1000.0 + destination_rank)
      if (my_rank == 0) then
        print *, "  -> Result correct on rank 0? -", is_correct
      elseif (.not. is_correct) then
        print *, "  -> Result INCORRECT on rank ", my_rank, "? -", "expected", (1000.0 + destination_rank), "got", result_data(1)

      endif
      call mpi_barrier(mpi_comm_world, ierr)
    enddo

    ok = gpu_free(original_data_gpu)
    NOT_OK_OUCH
    ok = gpu_free(result_data_gpu)
    NOT_OK_OUCH
  end subroutine test_ccl_broadcast



  subroutine test_ccl_sendrecv(ccl_comm, num_elements, oneccl_stream)
    use, intrinsic :: iso_c_binding
    use elpa_gpu
    use elpa_ccl_gpu
    use oneccl_functions
    use sycl_functions
    use mpi
    implicit none

    integer(kind=c_intptr_t), intent(in) :: ccl_comm
    integer(kind=c_intptr_t), intent(in), value :: num_elements
    integer(kind=c_intptr_t), intent(in) :: oneccl_stream

    integer(kind=c_size_t) :: num_ccl_elements
    integer(kind=c_int) :: n_ranks, my_rank, ierr, send_partner, recv_partner
    integer(kind=c_int) :: i, is_correct
    logical :: ok

    real(kind=c_double) :: original_data(num_elements), result_data(num_elements)
    integer(kind=c_intptr_t) :: original_data_gpu, result_data_gpu

    call MPI_comm_rank(MPI_COMM_WORLD, my_rank, ierr)
    call MPI_comm_size(MPI_COMM_WORLD, n_ranks, ierr)

    send_partner = mod(my_rank + n_ranks + 1, n_ranks)
    recv_partner = mod(my_rank + n_ranks - 1, n_ranks)

    num_ccl_elements = num_elements

    do i = 1, num_elements
      original_data(i) = 1000.0 + my_rank
    end do

    ok = gpu_malloc(original_data_gpu, num_elements * c_double); NOT_OK_OUCH
    ok = gpu_malloc(result_data_gpu, num_elements * c_double); NOT_OK_OUCH
    ok = gpu_memcpy(original_data_gpu, int(loc(original_data),c_intptr_t), num_elements * c_double, gpuMemcpyHostToDevice)
    NOT_OK_OUCH

    if (my_rank == 0) then
      print *, " - test Send/Recv to rank ", send_partner
    end if
    !  function ccl_send_intptr(sendbuff, nrElements, cclDatatype, peer, cclComm, gpuStream) result(success)
    if (mod(n_ranks, 2) /= 0) then
      print *, "The send test only works with even rank numbers!"
      return
    else
      if (mod(my_rank, 2) == 0) then
        ok = ccl_send(original_data_gpu, num_ccl_elements, ccl_dataType_cclDouble(), send_partner, ccl_comm, oneccl_stream); NOT_OK_OUCH
        ok = ccl_recv(result_data_gpu,   num_ccl_elements, ccl_dataType_cclDouble(), recv_partner, ccl_comm, oneccl_stream); NOT_OK_OUCH
      else
        ok = ccl_recv(result_data_gpu,   num_ccl_elements, ccl_dataType_cclDouble(), recv_partner, ccl_comm, oneccl_stream); NOT_OK_OUCH
        ok = ccl_send(original_data_gpu, num_ccl_elements, ccl_dataType_cclDouble(), send_partner, ccl_comm, oneccl_stream); NOT_OK_OUCH
      endif
    endif
    ! ok = oneccl_stream_synchronize(oneccl_stream); NOT_OK_OUCH

    ok = gpu_memcpy(loc(result_data), result_data_gpu, num_elements * c_double, gpuMemcpyDeviceToHost); NOT_OK_OUCH
    is_correct = 1
    if (result_data(1) /= 1000.0 + recv_partner) then
      print *, "  -> Result INCORRECT on rank ", my_rank, "? -", "expected", (1000.0 + recv_partner), "got", result_data(1)
      is_correct = 0
    endif

    call MPI_Allreduce(MPI_IN_PLACE, is_correct, 1, MPI_INTEGER, MPI_LAND, MPI_COMM_WORLD, ierr)

    if (my_rank == 0 .and. is_correct == 1) then
      print *, "  -> Result CORRECT on all ranks"
    endif

  ! int expectedResult = (recvPartner + 1)  * 1000;
  ! bool correct = recvBuffer[0] == static_cast<DT>(expectedResult);
  ! std::cout << "  Result: " << recvBuffer[0] << " is " << (correct ? "CORRECT" : "\033[1;31mINCORRECT") << "\033[0m" << std::endl;
  end subroutine test_ccl_sendrecv
end module oneccl_tests
