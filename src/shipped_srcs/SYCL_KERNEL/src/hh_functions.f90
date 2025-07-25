
#ifdef ENABLE_REFERENCE_CUDA_IMPLEMENTATION
#define USE_COMPUTE_HH_WRAPPER_CUDA use compute_hh_wrapper_cuda
#define COMPUTE_HH_CUDA_GPU_COMPLEX(nn, nc, nbw, evec2, hh, tau, wa, wb, wc) \
   call compute_hh_cuda_gpu_complex(nn, nc, nbw, evec2, hh, tau, wa, wb, wc)
#define COMPUTE_HH_CUDA_GPU(nn, nc, nbw, evec2, hh, tau, wa, wb, wc) \
   call compute_hh_cuda_gpu(nn, nc, nbw, evec2, hh, tau, wa, wb, wc)
#else
#define USE_COMPUTE_HH_WRAPPER_CUDA
#define COMPUTE_HH_CUDA_GPU_COMPLEX(nn, nc, nbw, evec2, hh, tau, wa, wb, wc) stop
#define COMPUTE_HH_CUDA_GPU(nn, nc, nbw, evec2, hh, tau, wa, wb, wc) stop
#endif

#ifdef ENABLE_REFERENCE_ROCM_IMPLEMENTATION
#define COMPUTE_HH_ROCM_GPU_COMPLEX(nn, nc, nbw, evec2, hh, tau, wa, wb, wc) \
   call compute_hh_rocm_gpu_complex(nn, nc, nbw, evec2, hh, tau, wa, wb, wc)
#define COMPUTE_HH_ROCM_GPU(nn, nc, nbw, evec2, hh, tau, wa, wb, wc) \
   call compute_hh_rocm_gpu(nn, nc, nbw, evec2, hh, tau, wa, wb, wc)
#else
#define COMPUTE_HH_ROCM_GPU_COMPLEX(nn, nc, nbw, evec2, hh, tau, wa, wb, wc) stop
#define COMPUTE_HH_ROCM_GPU(nn, nc, nbw, evec2, hh, tau, wa, wb, wc) stop
#endif

module hh_functions
  implicit none
contains
  subroutine perform_hh_test_complex(nbw, nn, nr, nc, backend)
    use compute_hh_wrapper
#ifdef ENABLE_REFERENCE_ROCM_IMPLEMENTATION
    use compute_hh_wrapper_rocm
#endif
#ifdef ENABLE_REFERENCE_CUDA_IMPLEMENTATION
    use compute_hh_wrapper_cuda
#endif
    use omp_lib, only: omp_get_wtime
    use iso_c_binding, only: c_double,c_intptr_t
    implicit none

    integer, intent(in), value :: nbw
    integer, intent(in), value :: nn
    integer, intent(in), value :: nr
    integer, intent(in), value :: nc
    integer, intent(in), value :: backend

    integer, parameter :: syclBe = 1, cudaBe = 2, rocmBe = 3

    integer :: n_rand
    integer :: i
    complex(c_double) :: dotp
    complex(c_double) :: err

    integer, allocatable :: seed(:)
    complex(c_double), allocatable :: evec1(:,:) ! Eigenvector matrix (X)
    complex(c_double), allocatable :: evec2(:,:) ! Eigenvector matrix (X)
    complex(c_double), allocatable :: hh(:,:) ! Householder vectors (v)
    complex(c_double), allocatable :: tau(:) ! (tau)

    double precision :: wa, wb, wc

    ! Generate random data
    call random_seed(size=n_rand)

    ! Note: evec here is the transpose of X in the paper
    allocate(seed(n_rand))
    allocate(evec1(nc,nr))
    allocate(evec2(nc,nr))
    allocate(hh(nbw,nn))
    allocate(tau(nn))

    seed(:) = 20191015

    print *, "warm up GPU"
    if (backend == cudaBe) then
      COMPUTE_HH_CUDA_GPU_COMPLEX(nn, nc, nbw, evec2, hh, tau, wa, wb, wc)
    elseif (backend == rocmBe) then
      COMPUTE_HH_ROCM_GPU_COMPLEX(nn, nc, nbw, evec2, hh, tau, wa, wb, wc)
    else
      call compute_hh_gpu_complex(nn, nc, nbw, evec2, hh, tau, wa, wb, wc)
    endif
    print *, "...done."


    call random_seed(put=seed)
    call random_number_complex(evec1, nc, nr)
    call random_number_complex(hh, nbw, nn)


    ! Normalize
    do i = 1,nc
    dotp = dot_product(evec1(i,:),evec1(i,:))
    evec1(i,:) = evec1(i,:)/sqrt(abs(dotp))
    end do

    do i = 1,nn
    dotp = dot_product(hh(:,i),hh(:,i))
    hh(:,i) = hh(:,i)/sqrt(abs(dotp))
    end do

    evec2(:,:) = evec1
    tau(:) = hh(1,:)
    hh(1,:) = 1.0

    ! put code to test here

    ! Start testing CPU reference code
    block
      double precision :: start, finish

      start = omp_get_wtime()
      call compute_hh_cpu_complex(nn,nc,nbw,evec1,hh,tau)
      finish = omp_get_wtime()

      print *, "CPU version finished"
      print '("Time CPU = ",f8.4," seconds.")',finish-start
    end block
    ! Start testing GPU code
    block
      double precision :: gpu_start, gpu_finish, init_time, transfer_time, compute_time

      gpu_start = omp_get_wtime()
      if (backend == cudaBe) then
        COMPUTE_HH_CUDA_GPU_COMPLEX(nn, nc, nbw, evec2, hh, tau, init_time, transfer_time, compute_time)
      elseif (backend == rocmBe) then
        COMPUTE_HH_ROCM_GPU_COMPLEX(nn, nc, nbw, evec2, hh, tau, init_time, transfer_time, compute_time)
      else
        call compute_hh_gpu_complex(nn, nc, nbw, evec2, hh, tau, init_time, transfer_time, compute_time)
      endif
      gpu_finish = omp_get_wtime()

      write(*,"(2X,A)") "GPU version finished"
      print '("Init     Time GPU = ",f8.4," seconds.")', init_time
      print '("Transfer Time GPU = ",f8.4," seconds.")', transfer_time
      print '("Compute  Time GPU = ",f8.4," seconds.")', compute_time
      print '("---------------------------------------------")'
      print '("Total Time GPU = ",f8.4," seconds.")', gpu_finish - gpu_start
    end block

    ! Compare results
    err = maxval(abs(evec1-evec2))

    write(*,"(2X,A,E10.2)") "| Error :",err

    deallocate(seed)
    deallocate(evec1)
    deallocate(evec2)
    deallocate(hh)
    deallocate(tau)
  end subroutine perform_hh_test_complex


  subroutine perform_hh_test_real(nbw, nn, nr, nc, backend)
    use compute_hh_wrapper
#ifdef ENABLE_REFERENCE_ROCM_IMPLEMENTATION
    use compute_hh_wrapper_rocm
#endif
#ifdef ENABLE_REFERENCE_CUDA_IMPLEMENTATION
    use compute_hh_wrapper_cuda
#endif
    use omp_lib, only: omp_get_wtime
    use iso_c_binding, only: c_double,c_intptr_t
    implicit none

    integer, intent(in), value :: nbw
    integer, intent(in), value :: nn
    integer, intent(in), value :: nr
    integer, intent(in), value :: nc
    integer, intent(in), value :: backend

    integer, parameter :: syclBe = 1, cudaBe = 2, rocmBe = 3

    integer :: n_rand
    integer :: i
    real(c_double) :: dotp
    real(c_double) :: err

    integer, allocatable :: seed(:)
    real(c_double), allocatable :: evec1(:,:) ! Eigenvector matrix (X)
    real(c_double), allocatable :: evec2(:,:) ! Eigenvector matrix (X)
    real(c_double), allocatable :: hh(:,:) ! Householder vectors (v)
    real(c_double), allocatable :: tau(:) ! (tau)

    double precision :: wa, wb, wc

    ! Generate random data
    call random_seed(size=n_rand)

    ! Note: evec here is the transpose of X in the paper
    allocate(seed(n_rand))
    allocate(evec1(nc,nr))
    allocate(evec2(nc,nr))
    allocate(hh(nbw,nn))
    allocate(tau(nn))

    seed(:) = 20191015

    call random_seed(put=seed)
    call random_number(hh)
    call random_number(evec1)

    print *, "start gpu test"
    if (backend == cudaBe) then
      COMPUTE_HH_CUDA_GPU(nn, nc, nbw, evec2, hh, tau, wa, wb, wc)
    elseif (backend == rocmBe) then
      COMPUTE_HH_ROCM_GPU(nn, nc, nbw, evec2, hh, tau, wa, wb, wc)
    else
      call compute_hh_gpu(nn, nc, nbw, evec2, hh, tau, wa,wb,wc)
    endif
    print *, "end gpu test"

    ! Normalize
    do i = 1,nc
      dotp = dot_product(evec1(i,:),evec1(i,:))
      evec1(i,:) = evec1(i,:)/sqrt(abs(dotp))
    end do

    do i = 1,nn
      dotp = dot_product(hh(:,i),hh(:,i))
      hh(:,i) = hh(:,i)/sqrt(abs(dotp))
    end do

    evec2(:,:) = evec1
    tau(:) = hh(1,:)
    hh(1,:) = 1.0

    ! put code to test here

    ! Start testing CPU reference code
    block
      double precision :: start, finish

      start = omp_get_wtime()
      call compute_hh_cpu(nn,nc,nbw,evec1,hh,tau)
      finish = omp_get_wtime()

      write(*,"(2X,A)") "CPU version finished"
      print '("Time CPU = ",f8.4," seconds.")',finish-start
    end block
    ! Start testing GPU code
    block
      double precision :: gpu_start, gpu_finish, init_time, transfer_time, compute_time

      gpu_start = omp_get_wtime()
      if (backend == cudaBe) then
        COMPUTE_HH_CUDA_GPU(nn, nc, nbw, evec2, hh, tau, init_time, transfer_time, compute_time)
      elseif (backend == rocmBe) then
        COMPUTE_HH_ROCM_GPU(nn, nc, nbw, evec2, hh, tau, init_time, transfer_time, compute_time)
      else
        call compute_hh_gpu(nn, nc, nbw, evec2, hh, tau, init_time, transfer_time, compute_time)
      endif
      gpu_finish = omp_get_wtime()

      write(*,"(2X,A)") "GPU version finished"
      print '("Init     Time GPU = ",f8.4," seconds.")', init_time
      print '("Transfer Time GPU = ",f8.4," seconds.")', transfer_time
      print '("Compute  Time GPU = ",f8.4," seconds.")', compute_time
      print '("---------------------------------------------")'
      print '("Total Time GPU = ",f8.4," seconds.")', gpu_finish - gpu_start
    end block


    ! Compare results
    err = maxval(abs(evec1-evec2))

    write(*,"(2X,A,E10.2)") "| Error :",err

    deallocate(seed)
    deallocate(evec1)
    deallocate(evec2)
    deallocate(hh)
    deallocate(tau)
  end subroutine perform_hh_test_real


  subroutine random_number_complex(array, dim1, dim2)
    use iso_c_binding, only: c_double
    implicit none

    complex (kind=c_double), intent(inout) :: array(:,:)
    complex (kind=c_double) :: tmp
    integer, intent(in), value :: dim1, dim2
    real (kind=c_double) :: realc, imagc
    integer :: i, j

    do i = 1, dim1
      do j = 1, dim2
        call random_number(realc)
        call random_number(imagc)
        tmp = cmplx(realc, imagc, c_double)
        array(i, j) = tmp
      end do
    end do
  end subroutine random_number_complex

end module
