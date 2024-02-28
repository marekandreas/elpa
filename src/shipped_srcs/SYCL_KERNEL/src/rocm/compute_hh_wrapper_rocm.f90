module compute_hh_wrapper_rocm

  use iso_c_binding, only: c_double,c_intptr_t

  implicit none

  contains

  subroutine compute_hh_rocm_gpu(nn,nc,nbw,q,hh,tau, init_time, transfer_time, compute_time)

    use rocm_f_interface
    use omp_lib, only: omp_get_wtime

    implicit none

    double precision, intent(out) :: init_time, transfer_time, compute_time

    integer, intent(in) :: nn ! (n)
    integer, intent(in) :: nc ! (N_C)
    integer, intent(in) :: nbw ! (b)
    real(c_double), intent(inout) :: q(:,:) ! (X), dimension(nc,nn+nbw-1)
    real(c_double), intent(in) :: hh(:,:) ! (v), dimension(nbw,nn)
    real(c_double), intent(in) :: tau(:) ! (tau), dimension(nn)

    double precision :: t_begin, t_end
    integer :: ok
    integer(c_intptr_t) :: q_dev
    integer(c_intptr_t) :: hh_dev
    integer(c_intptr_t) :: tau_dev
    integer(c_intptr_t) :: num
    integer(c_intptr_t) :: host_ptr

    integer, parameter :: size_of_double = 8

    t_begin = omp_get_wtime()
    call gpu_init()
    t_end = omp_get_wtime()
    init_time = t_end - t_begin

    t_begin = omp_get_wtime()
    ! Copy q to GPU
    num = nc*(nn+nbw-1)*size_of_double
    host_ptr = int(loc(q),c_intptr_t)
    ok = hip_malloc(q_dev,num)
    ok = hip_memcpy(q_dev,host_ptr,num,hipMemcpyHostToDevice)

    ! Copy hh to GPU
    num = nbw*nn*size_of_double
    host_ptr = int(loc(hh),c_intptr_t)
    ok = hip_malloc(hh_dev,num)
    ok = hip_memcpy(hh_dev,host_ptr,num,hipMemcpyHostToDevice)

    ! Copy tau to GPU
    num = nn*size_of_double
    host_ptr = int(loc(tau),c_intptr_t)
    ok = hip_malloc(tau_dev,num)
    ok = hip_memcpy(tau_dev,host_ptr,num,hipMemcpyHostToDevice)
    t_end = omp_get_wtime()
    transfer_time = t_end - t_begin

    ! Compute
    t_begin = omp_get_wtime()
    call compute_hh_hip_gpu_kernel(q_dev,hh_dev,tau_dev,nc,nbw,nc,nn)
    ok = hip_device_synchronize()

    t_end = omp_get_wtime()
    compute_time = t_end - t_begin

    ! Copy q to CPU
    t_begin = omp_get_wtime()
    num = nc*(nn+nbw-1)*size_of_double
    host_ptr = int(loc(q),c_intptr_t)
    ok = hip_memcpy(host_ptr,q_dev,num,hipMemcpyDeviceToHost)

    ok = hip_free(q_dev)
    ok = hip_free(hh_dev)
    ok = hip_free(tau_dev)

    t_end = omp_get_wtime()
    transfer_time = transfer_time + (t_end - t_begin)
  end subroutine

  subroutine compute_hh_rocm_gpu_complex(nn,nc,nbw,q,hh,tau, init_time, transfer_time, compute_time)

    use rocm_f_interface
    use omp_lib, only: omp_get_wtime

    implicit none

    double precision, intent(out) :: init_time, transfer_time, compute_time

    integer, intent(in) :: nn ! (n)
    integer, intent(in) :: nc ! (N_C)
    integer, intent(in) :: nbw ! (b)
    complex(c_double), intent(inout) :: q(:,:) ! (X), dimension(nc,nn+nbw-1)
    complex(c_double), intent(in)    :: hh(:,:) ! (v), dimension(nbw,nn)
    complex(c_double), intent(in)    :: tau(:) ! (tau), dimension(nn)

    double precision :: t_begin, t_end
    integer :: ok
    integer(c_intptr_t) :: q_dev
    integer(c_intptr_t) :: hh_dev
    integer(c_intptr_t) :: tau_dev
    integer(c_intptr_t) :: num
    integer(c_intptr_t) :: host_ptr

    integer, parameter :: size_of_double_complex = 8 * 2

    t_begin = omp_get_wtime()
    call gpu_init()
    t_end = omp_get_wtime()
    init_time = t_end - t_begin

    t_begin = omp_get_wtime()
    ! Copy q to GPU
    num = nc*(nn+nbw-1)*size_of_double_complex
    host_ptr = int(loc(q),c_intptr_t)
    ok = hip_malloc(q_dev,num)
    ok = hip_memcpy(q_dev,host_ptr,num,hipMemcpyHostToDevice)

    ! Copy hh to GPU
    num = nbw*nn*size_of_double_complex
    host_ptr = int(loc(hh),c_intptr_t)
    ok = hip_malloc(hh_dev,num)
    ok = hip_memcpy(hh_dev,host_ptr,num,hipMemcpyHostToDevice)

    ! Copy tau to GPU
    num = nn*size_of_double_complex
    host_ptr = int(loc(tau),c_intptr_t)
    ok = hip_malloc(tau_dev,num)
    ok = hip_memcpy(tau_dev,host_ptr,num,hipMemcpyHostToDevice)
    t_end = omp_get_wtime()
    transfer_time = t_end - t_begin

    ! Compute
    t_begin = omp_get_wtime()
    call compute_hh_hip_gpu_complex_kernel(q_dev,hh_dev,tau_dev,nc,nbw,nc,nn)
    ok = hip_device_synchronize()
    t_end = omp_get_wtime()
    compute_time = t_end - t_begin

    ! Copy q to CPU
    t_begin = omp_get_wtime()
    num = nc*(nn+nbw-1)*size_of_double_complex
    host_ptr = int(loc(q),c_intptr_t)
    ok = hip_memcpy(host_ptr,q_dev,num,hipMemcpyDeviceToHost)

    ok = hip_free(q_dev)
    ok = hip_free(hh_dev)
    ok = hip_free(tau_dev)

    t_end = omp_get_wtime()
    transfer_time = transfer_time + (t_end - t_begin)
  end subroutine
end module
