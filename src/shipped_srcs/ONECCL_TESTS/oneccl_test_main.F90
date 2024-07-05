program hh_test
  use, intrinsic :: iso_c_binding
  use elpa_gpu
  use elpa_ccl_gpu
  use oneccl_tests
  use mpi
  implicit none
  integer(kind=c_intptr_t), parameter :: num_elements = 1024
  type(onecclUniqueId) :: ccl_unique_id
  integer(kind=c_intptr_t) :: ccl_comm
  integer(kind=c_int) :: ierr, mpi_provided_thread_level
  
  call mpi_init_thread(mpi_thread_multiple, mpi_provided_thread_level, ierr)
  call test_ccl_setup(ccl_unique_id, ccl_comm)
  call test_ccl_allreduce(ccl_comm, num_elements)
  call test_ccl_reduce(ccl_comm, num_elements)
  !call test_ccl_broadcast(ccl_comm, num_elements)
  call test_ccl_sendrecv(ccl_comm,num_elements)
  
  call mpi_finalize(ierr)
end program
