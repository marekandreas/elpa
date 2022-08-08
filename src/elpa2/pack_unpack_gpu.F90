#if 0
!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
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
!
!    More information can be found here:
!    https://elpa.mpcdf.mpg.de/
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
! This file was written by A. Marek, MPCDF
#endif

! Pack a filled row group (i.e. an array of consecutive rows)

!#if REALCASE == 1
#ifdef WITH_CUDA_AWARE_MPI
#define WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
#endif
!#endif

!#if COMPLEXCASE == 1
!#undef WITH_CUDA_AWARE_MPI_TRANS_TRIDI_TO_BAND
!#endif

subroutine pack_row_group_&
&MATH_DATATYPE&
&_gpu_&
&PRECISION &
(obj, row_group_dev, a_dev, stripe_count, stripe_width, last_stripe_width, a_dim2, l_nev, &
 rows, n_offset, row_count, result_buffer_dev, nblk, num_result_buffers, nbuf, doCopyResult, &
 wantDebug, allComputeOnGPU, my_stream)
  use gpu_c_kernel
  use elpa_gpu, only : gpu_stream_synchronize, gpu_memcpy, gpu_memcpy_async, SIZE_OF_DOUBLE_REAL, &
#ifdef WANT_SINGLE_PRECISION_REAL
          SIZE_OF_SINGLE_REAL, &
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
          SIZE_OF_SINGLE_COMPLEX, &
#endif
          SIZE_OF_DOUBLE_COMPLEX, gpuMemcpyDeviceToHost, gpuMemcpyDeviceToDevice
  use elpa_abstract_impl
  use precision
  use, intrinsic :: iso_c_binding
  !use elpa_utilities, only : check_stream_synchronize_gpu_f
  implicit none
  class(elpa_abstract_impl_t), intent(inout)   :: obj
  integer(kind=c_intptr_t)                     :: row_group_dev, a_dev, result_buffer_dev

  integer(kind=ik), intent(in)                 :: stripe_count, stripe_width, last_stripe_width, a_dim2, l_nev
  integer(kind=ik), intent(in)                 :: n_offset, row_count
#if REALCASE == 1
  real(kind=C_DATATYPE_KIND)                   :: rows(:,:)
#endif
#if COMPLEXCASE == 1
  complex(kind=C_DATATYPE_KIND)                :: rows(:,:)
#endif
  integer(kind=ik)                             :: max_idx
  logical                                      :: successGPU
  logical, intent(in)                          :: doCopyResult, wantDebug, allComputeOnGPU
  integer(kind=ik), intent(in)                 :: nblk, nbuf
  integer(kind=ik), intent(in)                 :: num_result_buffers
  type(c_ptr)                                  :: result_buffer_mpi_dev
#if REALCASE == 1
  real(kind=C_DATATYPE_KIND), pointer          :: result_buffer_mpi_fortran_ptr(:,:,:)
#endif
#if COMPLEXCASE == 1
  complex(kind=C_DATATYPE_KIND), pointer       :: result_buffer_mpi_fortran_ptr(:,:,:)
#endif
  integer(kind=c_intptr_t)                     :: my_stream
  

  if (wantDebug) call obj%timer%start("pack_row_group")

  if (allComputeOnGPU) then 
    ! associate with c_ptr
    result_buffer_mpi_dev = transfer(result_buffer_dev, result_buffer_mpi_dev)
    ! and associate a fortran pointer
    call c_f_pointer(result_buffer_mpi_dev, result_buffer_mpi_fortran_ptr, &
                    [l_nev,nblk,num_result_buffers])
  endif

  ! Use many blocks for higher GPU occupancy
  max_idx = (stripe_count - 1) * stripe_width + last_stripe_width

  ! Use one kernel call to pack the entire row group
  ! writes to row_group_dev
  call launch_my_pack_gpu_kernel_&
  &MATH_DATATYPE&
  &_&
  &PRECISION &
  (row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev, my_stream)

  if (.not.(allComputeOnGPU)) then
#ifdef WITH_GPU_STREAMS
   successGPU = gpu_stream_synchronize(my_stream)
   !check_stream_synchronize_gpu("pack_row_group", successGPU)
   if (.not.(successGPU)) then
     print *,"pack_row_group_&
     &MATH_DATATYPE&
     &_gpu_&
     &PRECISION&
     &: error in stream_synchronize"
     stop 1
   endif

    successGPU =  gpu_memcpy_async(int(loc(rows(:, 1: row_count)),kind=c_intptr_t), row_group_dev , row_count * l_nev * size_of_&
    &PRECISION&
    &_&
    &MATH_DATATYPE&
    & , gpuMemcpyDeviceToHost, my_stream)
    if (.not.(successGPU)) then
      print *,"pack_row_group_&
      &MATH_DATATYPE&
      &_gpu_&
      &PRECISION&
      &: error in cudaMemcpy"
      stop 1
    endif

    successGPU = gpu_stream_synchronize(my_stream)
    if (.not.(successGPU)) then
      print *,"pack_row_group_&
      &MATH_DATATYPE&
      &_gpu_&
      &PRECISION&
      &: error in stream_synchronize"
      stop 1
    endif
    ! synchronize streamsPerThread; maybe not neccessary
    successGPU = gpu_stream_synchronize()
    if (.not.(successGPU)) then
      print *,"pack_row_group_&
      &MATH_DATATYPE&
      &_gpu_&
      &PRECISION&
      &: error in stream_synchronize"
      stop 1
    endif
#else /* WITH_GPU_STREAMS */
    successGPU =  gpu_memcpy(int(loc(rows(:, 1: row_count)),kind=c_intptr_t), row_group_dev , row_count * l_nev * size_of_&
    &PRECISION&
    &_&
    &MATH_DATATYPE&
    & , gpuMemcpyDeviceToHost)
    if (.not.(successGPU)) then
      print *,"pack_row_group_&
      &MATH_DATATYPE&
      &_gpu_&
      &PRECISION&
      &: error in cudaMemcpy"
      stop 1
    endif
#endif /* WITH_GPU_STREAMS */
  else ! allComputeOnGPU
    if (doCopyResult) then
      ! need to copy row_group_dev -> result_buffer_dev
#ifdef WITH_GPU_STREAMS  
      successGPU = gpu_stream_synchronize(my_stream)
      !check_stream_synchronize_gpu("pack_row_group 1", successGPU)
      if (.not.(successGPU)) then
        print *,"pack_row_group_&
        &MATH_DATATYPE&
        &_gpu_&
        &PRECISION&
        &: error in stream_synchronize"
        stop 1
      endif

      successGPU =  gpu_memcpy_async(c_loc(result_buffer_mpi_fortran_ptr(1, 1, nbuf)), &
                    row_group_dev , row_count * l_nev * size_of_&
                    &PRECISION&
                    &_&
                    &MATH_DATATYPE&
                    & , gpuMemcpyDeviceToDevice, my_stream)
      if (.not.(successGPU)) then
        print *,"pack_row_group_&
        &MATH_DATATYPE&
        &_gpu_&
        &PRECISION&
        &: error in cudaMemcpy"
        stop 1
      endif

      successGPU = gpu_stream_synchronize(my_stream)
      if (.not.(successGPU)) then
        print *,"pack_row_group_&
        &MATH_DATATYPE&
        &_gpu_&
        &PRECISION&
        &: error in stream_synchronize"
        stop 1
      endif
      ! synchronize streamsPerThread; maybe not neccessary
      successGPU = gpu_stream_synchronize()
      if (.not.(successGPU)) then
        print *,"pack_row_group_&
        &MATH_DATATYPE&
        &_gpu_&
        &PRECISION&
        &: error in stream_synchronize"
        stop 1
      endif

#else /* WITH_GPU_STREAMS */
      successGPU =  gpu_memcpy(c_loc(result_buffer_mpi_fortran_ptr(1, 1, nbuf)), &
                    row_group_dev , row_count * l_nev * size_of_&
                    &PRECISION&
                    &_&
                    &MATH_DATATYPE&
                    & , gpuMemcpyDeviceToDevice)
      if (.not.(successGPU)) then
        print *,"pack_row_group_&
        &MATH_DATATYPE&
        &_gpu_&
        &PRECISION&
        &: error in cudaMemcpy"
        stop 1
      endif
#endif /* WITH_GPU_STREAMS */

    endif
  endif ! allComputeOnGPU
  if (wantDebug) call obj%timer%stop("pack_row_group")

end subroutine


! Unpack a filled row group (i.e. an array of consecutive rows)
subroutine unpack_row_group_&
  &MATH_DATATYPE&
  &_gpu_&
  &PRECISION &
  (obj, row_group_dev, a_dev, stripe_count, stripe_width, last_stripe_width, &
   a_dim2, l_nev, rows, n_offset, row_count, wantDebug, allComputeOnGPU, my_stream)
  use gpu_c_kernel
  use elpa_abstract_impl

  !use elpa_utilities, only : check_stream_synchronize_gpu_f
  use precision
  use, intrinsic :: iso_c_binding
  use elpa_gpu, only : gpu_devicesynchronize, gpu_memcpy_async, gpuMemcpyHostToDevice, gpu_memcpy, &
#ifdef WANT_SINGLE_PRECISION_REAL
          SIZE_OF_SINGLE_REAL, &
#endif
#ifdef WANT_SINGLE_PRECISION_COMPLEX
          SIZE_OF_SINGLE_COMPLEX, &
#endif
          SIZE_OF_DOUBLE_REAL, SIZE_OF_DOUBLE_COMPLEX, gpu_stream_synchronize
  implicit none
  class(elpa_abstract_impl_t), intent(inout)   :: obj
  integer(kind=c_intptr_t)                     :: row_group_dev, a_dev
  integer(kind=ik), intent(in)                 :: stripe_count, stripe_width, last_stripe_width, a_dim2, l_nev
  integer(kind=ik), intent(in)                 :: n_offset, row_count
#if REALCASE == 1
  real(kind=C_DATATYPE_KIND), intent(in) :: rows(:, :)
#endif
#if COMPLEXCASE == 1
  complex(kind=C_DATATYPE_KIND), intent(in) :: rows(:, :)
#endif

  integer(kind=ik)                             :: max_idx
  logical                                      :: successGPU
  logical, intent(in)                          :: wantDebug, allComputeOnGPU
  integer(kind=c_intptr_t)                     :: my_stream

  if (wantDebug) call obj%timer%start("unpack_row_group")
  ! Use many blocks for higher GPU occupancy
  max_idx = (stripe_count - 1) * stripe_width + last_stripe_width

  if (.not.(allComputeOnGPU)) then
#ifdef WITH_GPU_STREAMS
    successGPU =  gpu_memcpy_async(row_group_dev , int(loc(rows(1, 1)),kind=c_intptr_t),row_count * l_nev * &
                             size_of_&
                             &PRECISION&
                             &_&
                             &MATH_DATATYPE&
                             &, gpuMemcpyHostToDevice, my_stream)
    if (.not.(successGPU)) then
        print *,"unpack_row_group_&
        &MATH_DATATYPE&
        &_gpu_&
        &PRECISION&
        &: error in cudaMemcpy"
        stop 1
    endif
    successGPU = gpu_stream_synchronize(my_stream)
    if (.not.(successGPU)) then
        print *,"unpack_row_group_&
        &MATH_DATATYPE&
        &_gpu_&
        &PRECISION&
        &: error in cudaStreamSynchronize"
        stop 1
    endif
    ! synchronize streamsPerThread; maybe not neccessary
    successGPU = gpu_stream_synchronize()
    if (.not.(successGPU)) then
        print *,"unpack_row_group_&
        &MATH_DATATYPE&
        &_gpu_&
        &PRECISION&
        &: error in cudaStreamSynchronize"
        stop 1
    endif
#else
    successGPU =  gpu_memcpy(row_group_dev , int(loc(rows(1, 1)),kind=c_intptr_t),row_count * l_nev * &
                             size_of_&
                             &PRECISION&
                             &_&
                             &MATH_DATATYPE&
                             &, gpuMemcpyHostToDevice)
    if (.not.(successGPU)) then
        print *,"unpack_row_group_&
        &MATH_DATATYPE&
        &_gpu_&
        &PRECISION&
        &: error in cudaMemcpy"
        stop 1
    endif
#endif
  endif ! allComputeOnGPU

  ! only read access to row_group_dev
  call launch_my_unpack_gpu_kernel_&
  &MATH_DATATYPE&
  &_&
  &PRECISION &
  ( row_count, n_offset, max_idx,stripe_width,a_dim2, stripe_count, l_nev, &
                                          row_group_dev, a_dev, my_stream)

  if (allComputeOnGPU) then
    if (wantDebug) call obj%timer%start("cuda_aware_device_synchronize")
    successGPU = gpu_devicesynchronize()
    if (.not.(successGPU)) then
      print *,"unpack_row_group_&
      &MATH_DATATYPE&
      &_gpu_&
      &PRECISION&
      &: error in cudaMemcpy"
      stop 1
    endif
    if (wantDebug) call obj%timer%stop("cuda_aware_device_synchronize")
  endif ! allComputeOnGPU
  if (wantDebug) call obj%timer%stop("unpack_row_group")

end subroutine

! This subroutine must be called before queuing the next row for unpacking; it ensures that an unpacking of the current row group
! occurs when the queue is full or when the next row belongs to another group
subroutine unpack_and_prepare_row_group_&
  &MATH_DATATYPE&
  &_gpu_&
  &PRECISION &
  (obj, row_group, row_group_dev, a_dev, stripe_count, stripe_width, &
   last_stripe_width, a_dim2, l_nev, row_group_size, nblk,      &
   unpack_idx, next_unpack_idx, force, wantDebug, allComputeOnGPU, my_stream)

  use, intrinsic :: iso_c_binding
  use precision
  use gpu_c_kernel
  use elpa_abstract_impl
  !use elpa_utilities, only : check_stream_synchronize_gpu_f

  implicit none

  class(elpa_abstract_impl_t), intent(inout)   :: obj
#if REALCASE == 1
  real(kind=C_DATATYPE_KIND)                   :: row_group(:,:)
#endif
#if COMPLEXCASE == 1
  complex(kind=C_DATATYPE_KIND)                :: row_group(:,:)
#endif
  integer(kind=c_intptr_t)                     :: row_group_dev, a_dev
  integer(kind=ik), intent(in)                 :: stripe_count, stripe_width, last_stripe_width, a_dim2, l_nev
  integer(kind=ik), intent(inout)              :: row_group_size
  integer(kind=ik), intent(in)                 :: nblk
  integer(kind=ik), intent(inout)              :: unpack_idx
  integer(kind=ik), intent(in)                 :: next_unpack_idx
  logical, intent(in)                          :: force, wantDebug, allComputeOnGPU
  integer(kind=c_intptr_t)                     :: my_stream

  if (wantDebug) call obj%timer%start("unpack_and_prepare_row_group")

  if (row_group_size == 0) then
    ! Nothing to flush, just prepare for the upcoming row
    row_group_size = 1
  else
    if (force .or. (row_group_size == nblk) .or. (unpack_idx + 1 /= next_unpack_idx)) then
      ! A flush and a reset must be performed
      call unpack_row_group_&
      &MATH_DATATYPE&
      &_gpu_&
      &PRECISION&
      (obj, row_group_dev, a_dev, stripe_count, stripe_width, last_stripe_width, &
       a_dim2, l_nev, row_group(:, :), unpack_idx - row_group_size, row_group_size, &
       wantDebug, allComputeOnGPU, my_stream)
      row_group_size = 1
    else
      ! Just prepare for the upcoming row
      row_group_size = row_group_size + 1
    endif
  endif
  ! Always update the index for the upcoming row
  unpack_idx = next_unpack_idx

  if (wantDebug) call obj%timer%stop("unpack_and_prepare_row_group")
end subroutine

! The host wrapper for extracting "tau" from the HH reflectors (see the kernel below)
subroutine extract_hh_tau_&
  &MATH_DATATYPE&
  &_gpu_&
  &PRECISION&
  & (bcast_buffer_dev, hh_tau_dev, nbw, n, is_zero, my_stream)
  use gpu_c_kernel
  use precision
  use, intrinsic :: iso_c_binding
  implicit none
  integer(kind=c_intptr_t) :: bcast_buffer_dev, hh_tau_dev
  integer(kind=ik), value  :: nbw, n
  logical, value           :: is_zero
  integer(kind=ik)         :: val_is_zero
  integer(kind=c_intptr_t) :: my_stream
  if (is_zero) then
   val_is_zero = 1
  else
   val_is_zero = 0
  endif

  call launch_extract_hh_tau_gpu_kernel_&
  &MATH_DATATYPE&
  &_&
  &PRECISION&
  & (bcast_buffer_dev, hh_tau_dev, nbw, n, val_is_zero, my_stream)
end subroutine
