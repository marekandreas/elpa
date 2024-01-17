!
! This program tests the CUDA kernel for Householder transformations.
! It is separeted from the 4th step of the ELPA2 eigensolver.
!
program hh_test
  use hh_functions

  implicit none

  character(10) :: arg
  integer            :: backend, numberType
  integer, parameter :: syclBe = 1, cudaBe = 2, realNt = 3, complexNt = 4, invalidConfig = 5
  integer            :: nbw       ! Length of Householder vectors (b==nbw)
  integer            :: nn        ! Length of eigenvectors (n)
  integer            :: nr        ! (N_R==n+b-1)
  integer, parameter :: nc = 1024 ! Number of eigenvectors (N_C)

  ! Read command line arguments
  if(command_argument_count() == 4) then
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

    call get_command_argument(3, arg)
    if (index(arg, "R") > 0) then
      numberType = realNt
    else if (index(arg, "C") > 0) then
      numberType = complexNt
    else
      write(*,"(2X,A)") "################################################"
      write(*,"(2X,A)") "##  Incorrect Datatype for Arg #3             ##"
      write(*,"(2X,A)") "##  Arg#3: Datatype: R -> Real C -> Complex   ##"
      write(*,"(2X,A)") "################################################"
      stop
    endif

    call get_command_argument(4, arg)
    if (index(arg, "N") > 0) then
      backend = cudaBe
    else if (index(arg, "S") > 0) then
      backend = syclBe
    else
      write(*,"(2X,A)") "################################################"
      write(*,"(2X,A)") "##  Incorrect Backend for Arg #4              ##"
      write(*,"(2X,A)") "##  Arg#4: Backend: N -> CUDA, S -> SYCL      ##"
      write(*,"(2X,A)") "################################################"
      stop
    endif

    write(*,"(2X,A)") "Test parameters:"
    write(*,"(2X,A,I10)") "| b  : ",nbw
    write(*,"(2X,A,I10)") "| n  : ",nn
    write(*,"(2X,A,I10)") "| nr : ",nr
    write(*,"(2X,A,I10)") "| nc : ",nc
    write(*,"(2X,A,A10)") "| DT : ",arg

    if (backend == syclBe .and. numberType == realNt) then
      call perform_hh_test_real(nbw, nn, nr, nc, .false.)
    else if (backend == syclBe .and. numberType == complexNt) then
      call perform_hh_test_complex(nbw, nn, nr, nc, .false.)
    else if (backend == cudaBe .and. numberType == realNt) then
      call perform_hh_test_real(nbw, nn, nr, nc, .true.)
    else if (backend == cudaBe .and. numberType == complexNt) then
      print *, "Not implemented yet, using SYCL!"
      call perform_hh_test_complex(nbw, nn, nr, nc, .true.)
      print *, "Not implemented"
    endif

  else
    write(*,"(2X,A)") "################################################"
    write(*,"(2X,A)") "##  Wrong number of command line arguments!!  ##"
    write(*,"(2X,A)") "##  Arg#1: Length of Householder vector       ##"
    write(*,"(2X,A)") "##         (must be 2^n, n = 1,2,...,10)      ##"
    write(*,"(2X,A)") "##  Arg#2: Length of eigenvectors             ##"
    write(*,"(2X,A)") "##  Arg#3: Datatype: R -> Real C -> Complex   ##"
    write(*,"(2X,A)") "##  Arg#4: Backend: N -> CUDA, S -> SYCL      ##"
    write(*,"(2X,A)") "##         ONLY use CUDA with NVIDIA GPUs!    ##"
    write(*,"(2X,A)") "################################################"
    stop
  end if

end program
