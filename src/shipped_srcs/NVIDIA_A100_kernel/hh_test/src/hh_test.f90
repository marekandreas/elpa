!
! This program tests the CUDA kernel for Householder transformations.
! It is separeted from the 4th step of the ELPA2 eigensolver.
!
program hh_test

  use iso_c_binding, only: c_double
  use compute_hh_wrapper, only: compute_hh_cpu,compute_hh_gpu

  implicit none

  character(10) :: arg
  integer :: nbw ! Length of Householder vectors (b==nbw)
  integer :: nn ! Length of eigenvectors (n)
  integer :: nr ! (N_R==n+b-1)
  integer :: n_rand
  integer :: i
  real(c_double) :: dotp
  real(c_double) :: err

  integer, allocatable :: seed(:)
  real(c_double), allocatable :: evec1(:,:) ! Eigenvector matrix (X)
  real(c_double), allocatable :: evec2(:,:) ! Eigenvector matrix (X)
  real(c_double), allocatable :: hh(:,:) ! Householder vectors (v)
  real(c_double), allocatable :: tau(:) ! (tau)

  integer, parameter :: nc = 1024 ! Number of eigenvectors (N_C)

  ! Read command line arguments
  if(command_argument_count() == 2) then
    call get_command_argument(1,arg)

    read(arg,*) nbw

    ! Must be 2^n
    if(nbw <= 2) then
      nbw = 2
    else if(nbw <= 4) then
      nbw = 4
    else if(nbw <= 8) then
      nbw = 8
    else if(nbw <= 16) then
      nbw = 16
    else if(nbw <= 32) then
      nbw = 32
    else if(nbw <= 64) then
      nbw = 64
    else if(nbw <= 128) then
      nbw = 128
    else if(nbw <= 256) then
      nbw = 256
    else if(nbw <= 512) then
      nbw = 512
    else
      nbw = 1024
    end if

    call get_command_argument(2,arg)

    read(arg,*) nn

    if(nn <= 0) then
      nn = 1000
    end if

    nr = nn+nbw-1

    write(*,"(2X,A)") "Test parameters:"
    write(*,"(2X,A,I10)") "| b  : ",nbw
    write(*,"(2X,A,I10)") "| n  : ",nn
    write(*,"(2X,A,I10)") "| nr : ",nr
    write(*,"(2X,A,I10)") "| nc : ",nc
  else
    write(*,"(2X,A)") "################################################"
    write(*,"(2X,A)") "##  Wrong number of command line arguments!!  ##"
    write(*,"(2X,A)") "##  Arg#1: Length of Householder vector       ##"
    write(*,"(2X,A)") "##         (must be 2^n, n = 1,2,...,10)      ##"
    write(*,"(2X,A)") "##  Arg#2: Length of eigenvectors             ##"
    write(*,"(2X,A)") "################################################"

    stop
  end if

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

  ! Start testing CPU reference code
  call compute_hh_cpu(nn,nc,nbw,evec1,hh,tau)

  write(*,"(2X,A)") "CPU version finished"

  ! Start testing GPU code
  call compute_hh_gpu(nn,nc,nbw,evec2,hh,tau)

  write(*,"(2X,A)") "GPU version finished"

  ! Compare results
  err = maxval(abs(evec1-evec2))

  write(*,"(2X,A,E10.2)") "| Error :",err

  deallocate(seed)
  deallocate(evec1)
  deallocate(evec2)
  deallocate(hh)
  deallocate(tau)

end program
