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
!
! Author: A. Marek, MPCDF

#include "../general/sanity.F90"
#include "../general/error_checking.inc"

  use elpa1_compute
  use elpa_mpi
  use precision
  use elpa_abstract_impl
  use, intrinsic :: iso_c_binding
  use elpa_gpu
  use mod_check_for_gpu
  use elpa_blas_interfaces
  implicit none

#include "../../src/general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout) :: obj

  character*1                   :: uplo_a, uplo_c

  integer(kind=ik), intent(in)  :: ldb, ldbCols, ldc, ldcCols
  integer(kind=ik)              :: na, ncb
#ifdef USE_ASSUMED_SIZE
  MATH_DATATYPE(kind=rck)       :: a(obj%local_nrows,*), b(ldb,*), c(ldc,*)
#else
  MATH_DATATYPE(kind=rck)       :: a(obj%local_nrows,obj%local_ncols), b(ldb,ldbCols), c(ldc,ldcCols)
#endif
  integer(kind=ik)              :: my_prow, my_pcol, np_rows, np_cols, myid
  integer(kind=MPI_KIND)        :: my_prowMPI, my_pcolMPI, np_rowsMPI, np_colsMPI
  integer(kind=MPI_KIND)        :: mpierr, myidMPI
  integer(kind=ik)              :: l_cols, l_rows, l_rows_np
  integer(kind=ik)              :: np, n, nb, nblk_mult, lrs, lre, lcs, lce
  integer(kind=ik)              :: gcol_min, gcol, goff
  integer(kind=ik)              :: nstor, nr_done, noff, np_bc, n_aux_bc, nvals
  integer(kind=ik), allocatable :: lrs_save(:), lre_save(:)

  logical                       :: a_lower, a_upper, c_lower, c_upper
  MATH_DATATYPE(kind=rck), pointer, contiguous :: aux_mat(:,:), tmp1(:,:)
  MATH_DATATYPE(kind=rck), allocatable :: aux_bc(:), tmp2(:,:)
  integer(kind=ik)              :: istat
  character(200)                :: errorMessage
  character(20)                 :: gpuString
  logical                       :: success
  logical                       :: successGPU
  logical                       :: useGPU
  integer(kind=c_int)           :: gpu, numGPU
  integer(kind=ik)              :: mpi_comm_rows, mpi_comm_cols, mpi_comm_all
  integer(kind=ik)              :: nblk, matrixRows, matrixCols, error
  integer(kind=c_intptr_t)      :: aux_dev, b_dev, tmp1_dev
  type(c_ptr)                   :: aux_host, tmp1_host
  integer(kind=c_intptr_t)      :: num
  integer(kind=c_intptr_t)      :: aux_off, b_off
  integer(kind=c_intptr_t), parameter :: size_of_datatype = size_of_&
                                                            &PRECISION&
                                                            &_&
                                                            &MATH_DATATYPE

  success = .true.

  ! GPU settings
  if (gpu_vendor() == NVIDIA_GPU) then
    call obj%get("nvidia-gpu",gpu,error)
    if (error .ne. ELPA_OK) then
      print *,"Problem getting option for NVIDIA GPU. Aborting..."
      stop
    endif
  else if (gpu_vendor() == AMD_GPU) then
    call obj%get("amd-gpu",gpu,error)
    if (error .ne. ELPA_OK) then
      print *,"Problem getting option for AMD GPU. Aborting..."
      stop
    endif
  else
    gpu = 0
  endif


  useGPU = (gpu == 1)

  if(useGPU) then
    gpuString = "_gpu"
  else
    gpuString = ""
  endif

  call obj%timer%start("elpa_mult_at_b_&
  &MATH_DATATYPE&
  &_&
  &PRECISION&
  &"//gpuString)

  na          = obj%na
  nblk        = obj%nblk
  matrixRows  = obj%local_nrows
  matrixCols  = obj%local_ncols

  call obj%get("mpi_comm_rows",mpi_comm_rows,error)
  if (error .ne. ELPA_OK) then
    print *,"Problem getting option for mpi_comm_rows. Aborting..."
    stop
  endif
  call obj%get("mpi_comm_cols",mpi_comm_cols,error)
  if (error .ne. ELPA_OK) then
    print *,"Problem getting option for mpi_comm_cols. Aborting..."
    stop
  endif
  call obj%get("mpi_comm_parent",mpi_comm_all,error)
  if (error .ne. ELPA_OK) then
    print *,"Problem getting option for mpi_comm_parent. Aborting..."
    stop
  endif

  call obj%timer%start("mpi_communication")
  call mpi_comm_rank(int(mpi_comm_rows,kind=MPI_KIND) ,my_prowMPI ,mpierr)
  call mpi_comm_size(int(mpi_comm_rows,kind=MPI_KIND) ,np_rowsMPI ,mpierr)
  call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND) ,my_pcolMPI ,mpierr)
  call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND) ,np_colsMPI ,mpierr)
  call mpi_comm_rank(int(mpi_comm_all, kind=MPI_KIND) ,myidMPI ,mpierr)

  my_prow = int(my_prowMPI,kind=c_int)
  np_rows = int(np_rowsMPI,kind=c_int)
  my_pcol = int(my_pcolMPI,kind=c_int)
  np_cols = int(np_colsMPI,kind=c_int)
  myid    = int(myidMPI,kind=c_int)
  call obj%timer%stop("mpi_communication")
  l_rows = local_index(na,  my_prow, np_rows, nblk, -1) ! Local rows of a and b
  l_cols = local_index(ncb, my_pcol, np_cols, nblk, -1) ! Local cols of b

  ! Block factor for matrix multiplications, must be a multiple of nblk

  if (na/np_rows <= 256) then
     nblk_mult = (31/nblk+1)*nblk
  else
     nblk_mult = (63/nblk+1)*nblk
  endif

  if (useGPU) then
    call obj%timer%start("check_for_gpu")
    if (check_for_gpu(obj, myid, numGPU)) then
      ! set the neccessary parameters
      call set_gpu_parameters()
    else
      print *,"GPUs are requested but not detected! Aborting..."
      success = .false.
      return
    endif
    call obj%timer%stop("check_for_gpu")

    ! copy b to b_dev
    num = ldb*ldbCols*size_of_datatype
    successGPU = gpu_malloc(b_dev,num)
    check_alloc_gpu("elpa_mult_at_b: b_dev", successGPU)

    successGPU = gpu_host_register(int(loc(b),kind=c_intptr_t),num,&
                  gpuHostRegisterDefault)

    check_host_register_gpu("elpa_mult_at_b: b", successGPU)

    successGPU = gpu_memcpy(b_dev,int(loc(b),kind=c_intptr_t),num,&
                  gpuMemcpyHostToDevice)
    check_memcpy_gpu("elpa_mult_at_b: b to b_dev", successGPU)

    num = l_rows*nblk_mult*size_of_datatype
    successGPU = gpu_malloc_host(aux_host,num)
    check_host_alloc_gpu("elpa_mult_at_b: aux_host", successGPU)

    call c_f_pointer(aux_host,aux_mat,(/l_rows,nblk_mult/))

    successGPU = gpu_malloc(aux_dev,num)
    check_alloc_gpu("elpa_mult_at_b: aux_dev", successGPU)

    num = nblk_mult*l_cols*size_of_datatype
    successGPU = gpu_malloc_host(tmp1_host,num)
    check_host_alloc_gpu("elpa_mult_at_b: tmp1_host", successGPU)

    call c_f_pointer(tmp1_host,tmp1,(/nblk_mult,l_cols/))

    successGPU = gpu_malloc(tmp1_dev,num)
    check_alloc_gpu("elpa_mult_at_b: tmp1_dev", successGPU)
  else ! useGPU
    allocate(aux_mat(l_rows,nblk_mult), stat=istat, errmsg=errorMessage)
    check_allocate("elpa_mult_at_b: aux_mat", istat, errorMessage)
  endif ! useGPU

  allocate(aux_bc(l_rows*nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_mult_at_b: aux_bc", istat, errorMessage)

  allocate(lrs_save(nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_mult_at_b: lrs_save", istat, errorMessage)

  allocate(lre_save(nblk), stat=istat, errmsg=errorMessage)
  check_allocate("elpa_mult_at_b: lre_save", istat, errorMessage)

  a_lower = .false.
  a_upper = .false.
  c_lower = .false.
  c_upper = .false.

  if (uplo_a=='u' .or. uplo_a=='U') a_upper = .true.
  if (uplo_a=='l' .or. uplo_a=='L') a_lower = .true.
  if (uplo_c=='u' .or. uplo_c=='U') c_upper = .true.
  if (uplo_c=='l' .or. uplo_c=='L') c_lower = .true.

  ! Build up the result matrix by processor rows

  do np = 0, np_rows-1

    ! In this turn, procs of row np assemble the result

    l_rows_np = local_index(na, np, np_rows, nblk, -1) ! local rows on receiving processors

    nr_done = 0 ! Number of rows done
    aux_mat = 0
    nstor = 0   ! Number of columns stored in aux_mat

    ! Loop over the blocks on row np

    do nb=0,(l_rows_np-1)/nblk

      goff  = nb*np_rows + np ! Global offset in blocks corresponding to nb

      ! Get the processor column which owns this block (A is transposed, so we need the column)
      ! and the offset in blocks within this column.
      ! The corresponding block column in A is then broadcast to all for multiplication with B

      np_bc = MOD(goff,np_cols)
      noff = goff/np_cols
      n_aux_bc = 0

      ! Gather up the complete block column of A on the owner

      do n = 1, min(l_rows_np-nb*nblk,nblk) ! Loop over columns to be broadcast

        gcol = goff*nblk + n ! global column corresponding to n
        if (nstor==0 .and. n==1) gcol_min = gcol

        lrs = 1       ! 1st local row number for broadcast
        lre = l_rows  ! last local row number for broadcast
        if (a_lower) lrs = local_index(gcol, my_prow, np_rows, nblk, +1)
        if (a_upper) lre = local_index(gcol, my_prow, np_rows, nblk, -1)

        if (lrs<=lre) then
          nvals = lre-lrs+1
          if (my_pcol == np_bc) aux_bc(n_aux_bc+1:n_aux_bc+nvals) = a(lrs:lre,noff*nblk+n)
          n_aux_bc = n_aux_bc + nvals
        endif

        lrs_save(n) = lrs
        lre_save(n) = lre

      enddo

      ! Broadcast block column
#ifdef WITH_MPI
      call obj%timer%start("mpi_communication")
#if REALCASE == 1
      call MPI_Bcast(aux_bc, int(n_aux_bc,kind=MPI_KIND),    &
                     MPI_REAL_PRECISION,  &
                     int(np_bc,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
#endif
#if COMPLEXCASE == 1
      call MPI_Bcast(aux_bc, int(n_aux_bc,kind=MPI_KIND),    &
                     MPI_COMPLEX_PRECISION,  &
                     int(np_bc,kind=MPI_KIND), int(mpi_comm_cols,kind=MPI_KIND), mpierr)
#endif
      call obj%timer%stop("mpi_communication")
#endif /* WITH_MPI */
      ! Insert what we got in aux_mat

      n_aux_bc = 0
      do n = 1, min(l_rows_np-nb*nblk,nblk)
        nstor = nstor+1
        lrs = lrs_save(n)
        lre = lre_save(n)
        if (lrs<=lre) then
          nvals = lre-lrs+1
          aux_mat(lrs:lre,nstor) = aux_bc(n_aux_bc+1:n_aux_bc+nvals)
          n_aux_bc = n_aux_bc + nvals
        endif
      enddo

      ! If we got nblk_mult columns in aux_mat or this is the last block
      ! do the matrix multiplication

      if (nstor==nblk_mult .or. nb*nblk+nblk >= l_rows_np) then

        lrs = 1       ! 1st local row number for multiply
        lre = l_rows  ! last local row number for multiply
        if (a_lower) lrs = local_index(gcol_min, my_prow, np_rows, nblk, +1)
        if (a_upper) lre = local_index(gcol, my_prow, np_rows, nblk, -1)

        lcs = 1       ! 1st local col number for multiply
        lce = l_cols  ! last local col number for multiply
        if (c_upper) lcs = local_index(gcol_min, my_pcol, np_cols, nblk, +1)
        if (c_lower) lce = MIN(local_index(gcol, my_pcol, np_cols, nblk, -1),l_cols)

        if (lcs<=lce) then
          allocate(tmp1(nstor,lcs:lce), tmp2(nstor,lcs:lce), stat=istat, errmsg=errorMessage)
          call check_alloc("elpa_mult_at_b_&
          &MATH_DATATYPE ", "tmp1", istat, errorMessage)

          if (lrs<=lre) then
            if (useGPU) then
              num = l_rows*nblk_mult*size_of_datatype
              successGPU = gpu_memcpy(aux_dev, int(loc(aux_mat),kind=c_intptr_t), &
                            num, gpuMemcpyHostToDevice)
              check_memcpy_gpu("elpa_mult_at_b: aux_mat to aux_dev", successGPU)

              aux_off = (lrs-1)*size_of_datatype
              b_off = ((lcs-1)*ldb+lrs-1)*size_of_datatype

              call obj%timer%start("gpublas")
              call gpublas_PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', nstor, lce-lcs+1, &
                   lre-lrs+1, ONE, aux_dev+aux_off, l_rows, b_dev+b_off, ldb, ZERO, &
                   tmp1_dev, nstor)
              call obj%timer%stop("gpublas")

              num = nstor*(lce-lcs+1)*size_of_datatype
              successGPU = gpu_memcpy(int(loc(tmp1),kind=c_intptr_t), &
                            tmp1_dev, num, gpuMemcpyDeviceToHost)
              check_memcpy_gpu("elpa_mult_at_b: tmp1_dev to tmp1", successGPU)
            else ! useGPU
              call obj%timer%start("blas")
              call PRECISION_GEMM(BLAS_TRANS_OR_CONJ, 'N', int(nstor,kind=BLAS_KIND), &
                                int(lce-lcs+1,kind=BLAS_KIND), int(lre-lrs+1,kind=BLAS_KIND), &
                                ONE, aux_mat(lrs:lre,1:nstor), int(lre-lrs+1,kind=BLAS_KIND), &
                                b(lrs,lcs), int(ldb,kind=BLAS_KIND), ZERO, tmp1, &
                                int(nstor,kind=BLAS_KIND))
              call obj%timer%stop("blas")
            endif ! useGPU
          else
            tmp1 = 0
          endif

          ! Sum up the results and send to processor row np
#ifdef WITH_MPI
          call obj%timer%start("mpi_communication")
          call mpi_reduce(tmp1, tmp2, int(nstor*(lce-lcs+1),kind=MPI_KIND),  MPI_MATH_DATATYPE_PRECISION, &
                          MPI_SUM, int(np,kind=MPI_KIND), int(mpi_comm_rows,kind=MPI_KIND), mpierr)
          call obj%timer%stop("mpi_communication")
          ! Put the result into C
          if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp2(1:nstor,lcs:lce)

#else /* WITH_MPI */
!          tmp2 = tmp1
          ! Put the result into C
          if (my_prow==np) c(nr_done+1:nr_done+nstor,lcs:lce) = tmp1(1:nstor,lcs:lce)

#endif /* WITH_MPI */

          deallocate(tmp1,tmp2, stat=istat, errmsg=errorMessage)
          check_deallocate("elpa_mult_at_b: tmp1, tmp2", istat, errorMessage)
        endif

        nr_done = nr_done+nstor
        nstor=0
        aux_mat(:,:)=0
      endif
    enddo
  enddo

  if (useGPU) then
    successGPU = gpu_free(b_dev)
    check_dealloc_gpu("elpa_multiply_a_b: b_dev", successGPU)

    successGPU = gpu_host_unregister(int(loc(b),kind=c_intptr_t))
    check_host_unregister_gpu("elpa_multiply_a_b: b", successGPU)

    nullify(aux_mat)
    nullify(tmp1)

    successGPU = gpu_free_host(aux_host)
    check_host_dealloc_gpu("elpa_multiply_a_b: aux_host", successGPU)

    successGPU = gpu_free(aux_dev)
    check_dealloc_gpu("elpa_multiply_a_b: aux_dev", successGPU)

    successGPU = gpu_free_host(tmp1_host)
    check_host_dealloc_gpu("elpa_multiply_a_b: tmp1_host", successGPU)

    successGPU = gpu_free(tmp1_dev)
    check_dealloc_gpu("elpa_multiply_a_b: tmp1_dev", successGPU)
  else ! useGPU
    deallocate(aux_mat, stat=istat, errmsg=errorMessage)
    check_deallocate("elpa_mult_at_b: aux_mat", istat, errorMessage)
  endif ! useGPU

  deallocate(aux_bc, lrs_save, lre_save, stat=istat, errmsg=errorMessage)
  check_deallocate("elpa_mult_at_b: aux_bc, lrs_save, lre_save", istat, errorMessage)


  call obj%timer%stop("elpa_mult_at_b_&
  &MATH_DATATYPE&
  &_&
  &PRECISION&
  &"//gpuString)
