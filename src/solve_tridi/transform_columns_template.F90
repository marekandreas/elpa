#if 0
!    This file is part of ELPA.
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
#endif

#include "../general/sanity.F90"
#include "../general/error_checking.inc"


#ifdef SOLVE_TRIDI_GPU_BUILD
subroutine transform_columns_gpu_&
                                 &PRECISION&
                                 &(obj, col1, col2, na, tmp, l_rqs, l_rqe, q_dev, ldq, matrixCols, &
                                   l_rows, mpi_comm_cols, p_col, l_col, qtrans_dev, &
                                   tmp_dev, zero_dev, one_dev, debug, my_stream)
#else
subroutine transform_columns_cpu_&
                                 &PRECISION&
                                 &(obj, col1, col2, na, tmp, l_rqs, l_rqe, q    , ldq, matrixCols, &
                                   l_rows, mpi_comm_cols, p_col, l_col, qtrans)
#endif
  use precision
  use elpa_abstract_impl
#ifdef WITH_OPENMP_TRADITIONAL
  use elpa_omp
#endif
  use elpa_mpi
  use elpa_gpu
  use elpa_ccl_gpu
  use transform_columns_gpu
  use elpa_utilities, only : check_memcpy_gpu_f
  implicit none
  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik), intent(in)               :: na, l_rqs, l_rqe, ldq, matrixCols
  integer(kind=ik), intent(in)               :: l_rows, mpi_comm_cols
  integer(kind=ik), intent(in)               :: p_col(na), l_col(na)

  integer(kind=c_intptr_t)                   :: q_dev, tmp_dev, shift_dev, qtrans_dev, zero_dev, one_dev

#if defined(USE_ASSUMED_SIZE) && !defined(SOLVE_TRIDI_GPU_BUILD)
  real(kind=REAL_DATATYPE), intent(inout)    :: q(ldq,*)
#else
  real(kind=REAL_DATATYPE)                   :: q(ldq,matrixCols)
#endif

  real(kind=REAL_DATATYPE)                   :: qtrans(2,2)
#ifdef WITH_MPI
  integer(kind=MPI_KIND)                     :: mpierrMPI, my_pcolMPI
  integer(kind=ik)                           :: mpierr
#endif
  integer(kind=ik)                           :: my_pcol
  integer(kind=ik)                           :: col1, col2
  real(kind=REAL_DATATYPE)                   :: tmp(na) ! PETERDEBUG111 why na, not l_rows? 
  integer(kind=ik)                           :: pc1, pc2, lc1, lc2

  logical                                    :: useGPU, successGPU
  integer(kind=ik)                           :: debug
  integer(kind=c_intptr_t)                   :: my_stream
  integer(kind=c_intptr_t), parameter        :: size_of_datatype = size_of_&
                                                                           &PRECISION&
                                                                           &_real

  logical                                    :: useCCL
  integer(kind=c_intptr_t)                   :: ccl_comm_cols
  integer(kind=c_int)                        :: cclDataType

  if (l_rows==0) return ! My processor column has no work to do
  
  useGPU = .false.
#ifdef SOLVE_TRIDI_GPU_BUILD
  useGPU = .true.
#endif

  useCCL = obj%gpu_setup%useCCL

  if (useGPU) then
#ifdef WITH_GPU_STREAMS
    my_stream = obj%gpu_setup%my_stream
#endif
    !SM_count = obj%gpu_setup%gpuSMcount ! PETERDEBUG111: cleanup if not needed
    
    if (useCCL) then
      my_stream = obj%gpu_setup%my_stream
      ccl_comm_cols = obj%gpu_setup%ccl_comm_cols
#if defined(DOUBLE_PRECISION)
      cclDataType = cclDouble
#endif      
#if defined(SINGLE_PRECISION)
      cclDataType = cclFloat
#endif
    endif ! useCCL
  endif ! useGPU

#ifdef WITH_MPI
  call obj%timer%start("mpi_communication")
  call mpi_comm_rank(int(mpi_comm_cols,kind=MPI_KIND) ,my_pcolMPI, mpierr)
  !call mpi_comm_size(int(mpi_comm_cols,kind=MPI_KIND) ,np_colsMPI, mpierr)

  my_pcol = int(my_pcolMPI,kind=c_int)
  !np_cols = int(np_colsMPI,kind=c_int)

  call obj%timer%stop("mpi_communication")
#else
#endif
  pc1 = p_col(col1)
  lc1 = l_col(col1)
  pc2 = p_col(col2)
  lc2 = l_col(col2)

  if (pc1==my_pcol) then
    if (pc2==my_pcol) then
      ! both columns are local
      if (useGPU) then
        call gpu_transform_two_columns(PRECISION_CHAR, q_dev, qtrans_dev, tmp_dev, ldq, l_rows, l_rqs, l_rqe, lc1, lc2, &
                                       debug, my_stream)
      else  ! useGPU
        tmp(1:l_rows)      = q(l_rqs:l_rqe,lc1)*qtrans(1,1) + q(l_rqs:l_rqe,lc2)*qtrans(2,1)
        q(l_rqs:l_rqe,lc2) = q(l_rqs:l_rqe,lc1)*qtrans(1,2) + q(l_rqs:l_rqe,lc2)*qtrans(2,2)
        q(l_rqs:l_rqe,lc1) = tmp(1:l_rows)
      endif ! useGPU
    else ! (pc2==my_pcol)
#ifdef WITH_MPI
      if (useGPU .and. .not. useCCL) then
        ! memcopy GPU->CPU
        ! PETERDEBUG111 streamed version
        shift_dev = (l_rqs-1 + (lc1-1)*ldq)*size_of_datatype
        successGPU = gpu_memcpy(int(loc(q(l_rqs,lc1)),kind=c_intptr_t), q_dev + shift_dev, &
                                l_rows*size_of_datatype, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("transform_columns: q_dev, lc1", successGPU)
      endif

      if (useCCL) then
        call obj%timer%start("ccl_send_recv")
        successGPU = ccl_group_start()

        successGPU = successGPU .and. ccl_send (q_dev, int(l_rows,kind=c_size_t), &
                                                cclDataType, pc2, ccl_comm_cols, my_stream)

        successGPU = successGPU .and. ccl_recv (tmp_dev, int(l_rows,kind=c_size_t), &
                                                cclDataType, pc2, ccl_comm_cols, my_stream)

        successGPU = ccl_group_end()

        if (.not. successGPU) then
          print *,"Error in ccl_group_start/ccl_group_end/ccl_send/ccl_recv!"
          stop 1
        endif

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("transform_columns", successGPU)
        call obj%timer%stop("ccl_send_recv")
      else  ! useCCL
        call obj%timer%start("mpi_communication")
        call mpi_sendrecv(q(l_rqs,lc1), int(l_rows,kind=MPI_KIND), MPI_REAL_PRECISION, pc2, 1_MPI_KIND, &
                          tmp, int(l_rows,kind=MPI_KIND), MPI_REAL_PRECISION, pc2, 1_MPI_KIND,          &
                          int(mpi_comm_cols,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
        call obj%timer%stop("mpi_communication")
      endif ! useCCL

      if (useGPU .and. .not. useCCL) then
        ! memcopy CPU->GPU
        ! PETERDEBUG111 streamed version
        !shift_dev = (l_rqs-1 + (lc1-1)*ldq)*size_of_datatype
        successGPU = gpu_memcpy(tmp_dev, int(loc(tmp(1)),kind=c_intptr_t), &
                                l_rows*size_of_datatype, gpuMemcpyHostToDevice)
        check_memcpy_gpu("transform_columns: tmp_dev", successGPU)
      endif
#else /* WITH_MPI */
#endif /* WITH_MPI */

      if (useGPU) then
        shift_dev = (l_rqs-1 + (lc1-1)*ldq)*size_of_datatype
        call gpu_transform_one_column(PRECISION_CHAR, q_dev+shift_dev, tmp_dev, q_dev+shift_dev, &
                                      qtrans_dev, qtrans_dev + 1*size_of_datatype, l_rows, debug, my_stream)
      else
        q(l_rqs:l_rqe,lc1) = q(l_rqs:l_rqe,lc1)*qtrans(1,1) + tmp(1:l_rows)*qtrans(2,1)
      endif
    endif ! (pc2==my_pcol)
  else if (pc2==my_pcol) then ! (pc1==my_pcol)
#ifdef WITH_MPI
    if (useGPU .and. .not. useCCL) then
      ! PETERDEBUG111 streamed version
      shift_dev = (l_rqs-1 + (lc2-1)*ldq)*size_of_datatype
      successGPU = gpu_memcpy(int(loc(q(l_rqs,lc2)),kind=c_intptr_t), q_dev + shift_dev, &
                              l_rows*size_of_datatype, gpuMemcpyDeviceToHost)
      check_memcpy_gpu("transform_columns: q_dev, lc2", successGPU)
    endif

    if (useCCL) then
      call obj%timer%start("ccl_send_recv")
      successGPU = ccl_group_start()
    
      successGPU = successGPU .and. ccl_send (q_dev + shift_dev, int(l_rows,kind=c_size_t), &
                                              cclDataType, pc1, ccl_comm_cols, my_stream)
    
      successGPU = successGPU .and. ccl_recv (tmp_dev, int(l_rows,kind=c_size_t), &
                                              cclDataType, pc1, ccl_comm_cols, my_stream)
    
      successGPU = successGPU .and. ccl_group_end()
      if (.not. successGPU) then
        print *,"Error in ccl_group_start/ccl_group_end/ccl_send/ccl_recv!"
        stop 1
      endif
    
      successGPU = gpu_stream_synchronize(my_stream)
      check_stream_synchronize_gpu("transform_columns", successGPU)
      call obj%timer%stop("ccl_send_recv")
    else ! useCCL
      call obj%timer%start("mpi_communication")
      call mpi_sendrecv(q(l_rqs,lc2), int(l_rows,kind=MPI_KIND), MPI_REAL_PRECISION, pc1, 1_MPI_KIND, &
                        tmp, int(l_rows,kind=MPI_KIND), MPI_REAL_PRECISION, pc1, 1_MPI_KIND,          &
                        int(mpi_comm_cols,kind=MPI_KIND), MPI_STATUS_IGNORE, mpierr)
      call obj%timer%stop("mpi_communication")
    endif ! useCCL

    if (useGPU .and. .not. useCCL) then
      ! PETERDEBUG111 streamed version
      !shift_dev = (l_rqs-1 + (lc2-1)*ldq)*size_of_datatype
      successGPU = gpu_memcpy(tmp_dev, int(loc(tmp(1)),kind=c_intptr_t), &
                              l_rows*size_of_datatype, gpuMemcpyHostToDevice)
      check_memcpy_gpu("transform_columns: tmp_dev", successGPU)
    endif
#else /* WITH_MPI */
    if (useGPU) then
      shift_dev = (l_rqs-1 + (lc2-1)*ldq)*size_of_datatype
      call gpu_transform_one_column(PRECISION_CHAR, q_dev+shift_dev, tmp_dev, tmp_dev, &
                                    one_dev, zero_dev, l_rows, debug, my_stream)
    else
      tmp(1:l_rows) = q(l_rqs:l_rqe,lc2)
    endif
#endif /* WITH_MPI */

    if (useGPU) then
      shift_dev = (l_rqs-1 + (lc2-1)*ldq)*size_of_datatype
      call gpu_transform_one_column(PRECISION_CHAR, tmp_dev, q_dev+shift_dev, q_dev+shift_dev, &
                                    qtrans_dev+2*size_of_datatype, qtrans_dev+3*size_of_datatype, &
                                    l_rows, debug, my_stream)
    else
      q(l_rqs:l_rqe,lc2) = tmp(1:l_rows)*qtrans(1,2) + q(l_rqs:l_rqe,lc2)*qtrans(2,2)
    endif
  endif

end subroutine
