module compute_hh_wrapper

  use iso_c_binding, only: c_double,c_intptr_t

  implicit none

  contains

  subroutine compute_hh_gpu(nn,nc,nbw,q,hh,tau)

    use cuda_f_interface

    implicit none

    integer, intent(in) :: nn ! (n)
    integer, intent(in) :: nc ! (N_C)
    integer, intent(in) :: nbw ! (b)
    real(c_double), intent(inout) :: q(:,:) ! (X), dimension(nc,nn+nbw-1)
    real(c_double), intent(in) :: hh(:,:) ! (v), dimension(nbw,nn)
    real(c_double), intent(in) :: tau(:) ! (tau), dimension(nn)

    integer :: ok
    integer(c_intptr_t) :: q_dev
    integer(c_intptr_t) :: hh_dev
    integer(c_intptr_t) :: tau_dev
    integer(c_intptr_t) :: num
    integer(c_intptr_t) :: host_ptr

    integer, parameter :: size_of_double = 8

    call gpu_init()

    ! Copy q to GPU
    num = nc*(nn+nbw-1)*size_of_double
    host_ptr = int(loc(q),c_intptr_t)
    ok = cuda_malloc(q_dev,num)
    ok = cuda_memcpy(q_dev,host_ptr,num,cudaMemcpyHostToDevice)

    ! Copy hh to GPU
    num = nbw*nn*size_of_double
    host_ptr = int(loc(hh),c_intptr_t)
    ok = cuda_malloc(hh_dev,num)
    ok = cuda_memcpy(hh_dev,host_ptr,num,cudaMemcpyHostToDevice)

    ! Copy tau to GPU
    num = nn*size_of_double
    host_ptr = int(loc(tau),c_intptr_t)
    ok = cuda_malloc(tau_dev,num)
    ok = cuda_memcpy(tau_dev,host_ptr,num,cudaMemcpyHostToDevice)

    ! Compute
    call compute_hh_gpu_kernel(q_dev,hh_dev,tau_dev,nc,nbw,nc,nn)

    ! Copy q to CPU
    num = nc*(nn+nbw-1)*size_of_double
    host_ptr = int(loc(q),c_intptr_t)
    ok = cuda_memcpy(host_ptr,q_dev,num,cudaMemcpyDeviceToHost)

    ok = cuda_free(q_dev)
    ok = cuda_free(hh_dev)
    ok = cuda_free(tau_dev)

  end subroutine

  ! Householder transformation
  ! (I - tau * hh * hh^T) * q = q - tau * hh * hh^T * q
  subroutine compute_hh_cpu(nn,nc,nbw,q,hh,tau)

    implicit none

    integer, intent(in) :: nn ! (n)
    integer, intent(in) :: nc ! (N_C)
    integer, intent(in) :: nbw ! (b)
    real(c_double), intent(inout) :: q(:,:) ! (X), dimension(nc,nn+nbw-1)
    real(c_double), intent(in) :: hh(:,:) ! (v), dimension(nbw,nn)
    real(c_double), intent(in) :: tau(:) ! (tau), dimension(nn)

    integer :: j
    integer :: i
    real(c_double) :: dotp

    do j = nn,1,-1
      do i = 1,nc
        dotp = dot_product(q(i,j:j+nbw-1),hh(:,j))
        q(i,j:j+nbw-1) = q(i,j:j+nbw-1)-tau(j)*dotp*hh(:,j)
      end do
    end do

  end subroutine

end module
