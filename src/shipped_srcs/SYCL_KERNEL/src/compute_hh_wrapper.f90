module compute_hh_wrapper

  use iso_c_binding, only: c_double,c_intptr_t

  implicit none

  contains

  subroutine compute_hh_gpu(nn,nc,nbw,q,hh,tau, init_time, transfer_time, compute_time)

    use sycl_f_interface
    use omp_lib, only: omp_get_wtime

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

    double precision, intent(out) :: transfer_time, compute_time, init_time
    double precision :: t_begin, t_end

    integer, parameter :: size_of_double = 8

    t_begin = omp_get_wtime()
    call gpu_init()
    t_end = omp_get_wtime()
    init_time = t_end - t_begin


    t_begin = omp_get_wtime()

    ! Copy q to GPU
    num = nc*(nn+nbw-1)*size_of_double
    host_ptr = int(loc(q),c_intptr_t)
    ok = sycl_malloc(q_dev,num)
    ok = sycl_memcpy(q_dev,host_ptr,num,syclMemcpyHostToDevice)

    ! Copy hh to GPU
    num = nbw*nn*size_of_double
    host_ptr = int(loc(hh),c_intptr_t)
    ok = sycl_malloc(hh_dev,num)
    ok = sycl_memcpy(hh_dev,host_ptr,num,syclMemcpyHostToDevice)

    ! Copy tau to GPU
    num = nn*size_of_double
    host_ptr = int(loc(tau),c_intptr_t)
    ok = sycl_malloc(tau_dev,num)
    ok = sycl_memcpy(tau_dev,host_ptr,num,syclMemcpyHostToDevice)

    t_end = omp_get_wtime()
    transfer_time = t_end - t_begin

    ! Compute
    t_begin = omp_get_wtime()
    call compute_hh_gpu_kernel(q_dev,hh_dev,tau_dev,nc,nbw,nc,nn)
    t_end = omp_get_wtime()
    compute_time = t_end - t_begin

    t_begin = omp_get_wtime()

    ! Copy q to CPU
    num = nc*(nn+nbw-1)*size_of_double
    host_ptr = int(loc(q),c_intptr_t)
    ok = sycl_memcpy(host_ptr,q_dev,num,syclMemcpyDeviceToHost)

    ok = sycl_free(q_dev)
    ok = sycl_free(hh_dev)
    ok = sycl_free(tau_dev)

    t_end = omp_get_wtime()
    transfer_time = transfer_time + (t_end - t_begin)
  end subroutine

  subroutine compute_hh_gpu_complex(nn,nc,nbw,q,hh,tau, init_time, transfer_time, compute_time)

    use sycl_f_interface
    use omp_lib, only: omp_get_wtime

    implicit none

    integer, intent(in) :: nn ! (n)
    integer, intent(in) :: nc ! (N_C)
    integer, intent(in) :: nbw ! (b)
    complex(c_double), intent(inout) :: q(:,:) ! (X), dimension(nc,nn+nbw-1)
    complex(c_double), intent(in) :: hh(:,:) ! (v), dimension(nbw,nn)
    complex(c_double), intent(in) :: tau(:) ! (tau), dimension(nn)

    integer :: ok
    integer(c_intptr_t) :: q_dev
    integer(c_intptr_t) :: hh_dev
    integer(c_intptr_t) :: tau_dev
    integer(c_intptr_t) :: num
    integer(c_intptr_t) :: host_ptr

    double precision, intent(out) :: transfer_time, compute_time, init_time
    double precision :: t_begin, t_end

    integer, parameter :: size_of_double_complex = 16

    t_begin = omp_get_wtime()
    call gpu_init()
    t_end = omp_get_wtime()
    init_time = t_end - t_begin


    t_begin = omp_get_wtime()

    ! Copy q to GPU
    num = nc*(nn+nbw-1)*size_of_double_complex
    host_ptr = int(loc(q),c_intptr_t)
    ok = sycl_malloc(q_dev,num)
    ok = sycl_memcpy(q_dev,host_ptr,num,syclMemcpyHostToDevice)

    ! Copy hh to GPU
    num = nbw*nn*size_of_double_complex
    host_ptr = int(loc(hh),c_intptr_t)
    ok = sycl_malloc(hh_dev,num)
    ok = sycl_memcpy(hh_dev,host_ptr,num,syclMemcpyHostToDevice)

    ! Copy tau to GPU
    num = nn*size_of_double_complex
    host_ptr = int(loc(tau),c_intptr_t)
    ok = sycl_malloc(tau_dev,num)
    ok = sycl_memcpy(tau_dev,host_ptr,num,syclMemcpyHostToDevice)

    t_end = omp_get_wtime()
    transfer_time = t_end - t_begin

    ! Compute
    t_begin = omp_get_wtime()
    call compute_hh_gpu_kernel_complex(q_dev,hh_dev,tau_dev,nc,nbw,nc,nn)
    t_end = omp_get_wtime()
    compute_time = t_end - t_begin

    t_begin = omp_get_wtime()

    ! Copy q to CPU
    num = nc*(nn+nbw-1)*size_of_double_complex
    host_ptr = int(loc(q),c_intptr_t)
    ok = sycl_memcpy(host_ptr,q_dev,num,syclMemcpyDeviceToHost)

    ok = sycl_free(q_dev)
    ok = sycl_free(hh_dev)
    ok = sycl_free(tau_dev)

    t_end = omp_get_wtime()
    transfer_time = transfer_time + (t_end - t_begin)
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

  ! Householder transformation
  ! (I - tau * hh * hh^T) * q = q - tau * hh * hh^T * q
  subroutine compute_hh_cpu_complex(nn,nc,nbw,q,hh,tau)
    implicit none

    integer, intent(in) :: nn ! (n)
    integer, intent(in) :: nc ! (N_C)
    integer, intent(in) :: nbw ! (b)
    complex(c_double), intent(inout) :: q(:,:) ! (X), dimension(nc,nn+nbw-1)
    complex(c_double), intent(in) :: hh(:,:) ! (v), dimension(nbw,nn)
    complex(c_double), intent(in) :: tau(:) ! (tau), dimension(nn)

    integer :: j
    integer :: i
    integer :: k, kbw
    complex(c_double) :: dotp, hh_conj

    do j = nn,1,-1
      do i = 1,nc
        dotp = 0.0
        do k = j, j+nbw-1
          kbw = k - j + 1
          hh_conj = conjg(hh(kbw,j))
          dotp = dotp + hh_conj * q(i, k)
        end do
        q(i,j:j+nbw-1) = q(i,j:j+nbw-1)-tau(j)*dotp*hh(:,j)
      end do
    end do
  end subroutine
end module
