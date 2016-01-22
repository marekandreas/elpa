module pack_unpack_real_gpu
#include "config-f90.h"
  implicit none

  public pack_row_group_real_gpu, unpack_row_group_real_gpu, &
         unpack_and_prepare_row_group_real_gpu, compute_hh_dot_products_real_gpu, &
         extract_hh_tau_real_gpu
  contains

    ! Pack a filled row group (i.e. an array of consecutive rows)
    subroutine pack_row_group_real_gpu(row_group_dev, a_dev, stripe_count, stripe_width, last_stripe_width, a_dim2, l_nev, &
                                       rows, n_offset, row_count)
      use cuda_c_kernel
      use cuda_functions
      use precision
      use iso_c_binding
      implicit none
      integer(kind=c_intptr_t)     :: row_group_dev, a_dev
      integer(kind=ik), intent(in) :: stripe_count, stripe_width, last_stripe_width, a_dim2, l_nev
      integer(kind=ik), intent(in) :: n_offset, row_count
      real(kind=rk)                :: rows(:,:)
      integer(kind=ik)             :: max_idx
      logical                      :: successCUDA

      ! Use many blocks for higher GPU occupancy
      max_idx = (stripe_count - 1) * stripe_width + last_stripe_width

      ! Use one kernel call to pack the entire row group

!      call my_pack_kernel<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, a_dev, row_group_dev)

      call launch_my_pack_c_kernel_real(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, &
                                        l_nev, a_dev, row_group_dev)

      ! Issue one single transfer call for all rows (device to host)
!        rows(:, 1 : row_count) = row_group_dev(:, 1 : row_count)

      successCUDA =  cuda_memcpy( loc(rows(:, 1: row_count)), row_group_dev , row_count * l_nev * size_of_real_datatype , &
                                 cudaMemcpyDeviceToHost)
      if (.not.(successCUDA)) then
        print *,"pack_row_group_real_gpu: error in cudaMemcpy"
        stop
      endif
      !write(*,*) cudaGetErrorString(istat)

    end subroutine


    ! Unpack a filled row group (i.e. an array of consecutive rows)
    subroutine unpack_row_group_real_gpu(row_group_dev, a_dev, stripe_count, stripe_width, last_stripe_width, &
                                         a_dim2, l_nev, rows, n_offset, row_count)
      use cuda_c_kernel
      use precision
      use iso_c_binding
      use cuda_functions
      implicit none
      integer(kind=c_intptr_t)     :: row_group_dev, a_dev
      integer(kind=ik), intent(in) :: stripe_count, stripe_width, last_stripe_width, a_dim2, l_nev
      integer(kind=ik), intent(in) :: n_offset, row_count
      real(kind=rk), intent(in)    :: rows(:, :)
      integer(kind=ik)             :: max_idx
      integer(kind=ik)             :: i
      logical                      :: successCUDA

      ! Use many blocks for higher GPU occupancy
      max_idx = (stripe_count - 1) * stripe_width + last_stripe_width

      ! Issue one single transfer call for all rows (host to device)
!      row_group_dev(:, 1 : row_count) = rows(:, 1 : row_count)

       !istat =  cuda_memcpy( row_group_dev , loc(rows(:, 1: row_count)),row_count * l_nev * size_of_real_datatype , &
       !      cudaMemcpyHostToDevice)

      successCUDA =  cuda_memcpy( row_group_dev , loc(rows(1, 1)),row_count * l_nev * &
                                 size_of_real_datatype ,cudaMemcpyHostToDevice)
      if (.not.(successCUDA)) then
        print *,"unpack_row_group_real_gpu: error in cudaMemcpy"
        stop
      endif
      !write(*,*) cudaGetErrorString(istat)

      ! Use one kernel call to pack the entire row group
      !        call my_unpack_kernel<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, row_group_dev, a_dev)

      call launch_my_unpack_c_kernel_real( row_count, n_offset, max_idx,stripe_width,a_dim2, stripe_count, l_nev, &
                                          row_group_dev,a_dev)

    end subroutine

    ! This subroutine must be called before queuing the next row for unpacking; it ensures that an unpacking of the current row group
    ! occurs when the queue is full or when the next row belongs to another group
    subroutine unpack_and_prepare_row_group_real_gpu(row_group, row_group_dev, a_dev, stripe_count, stripe_width, &
                                                     last_stripe_width, a_dim2, l_nev, row_group_size, nblk,      &
                                                     unpack_idx, next_unpack_idx, force)

      use iso_c_binding
      use precision
      implicit none
      real(kind=rk)                   :: row_group(:,:)
      integer(kind=c_intptr_t)        :: row_group_dev, a_dev
      integer(kind=ik), intent(in)    :: stripe_count, stripe_width, last_stripe_width, a_dim2, l_nev
      integer(kind=ik), intent(inout) :: row_group_size
      integer(kind=ik), intent(in)    :: nblk
      integer(kind=ik), intent(inout) :: unpack_idx
      integer(kind=ik), intent(in)    :: next_unpack_idx
      logical, intent(in)             :: force

      if (row_group_size == 0) then
        ! Nothing to flush, just prepare for the upcoming row
        row_group_size = 1
      else
        if (force .or. (row_group_size == nblk) .or. (unpack_idx + 1 /= next_unpack_idx)) then
          ! A flush and a reset must be performed
          call unpack_row_group_real_gpu(row_group_dev, a_dev, stripe_count, stripe_width, last_stripe_width, &
                                         a_dim2, l_nev, row_group(:, :), unpack_idx - row_group_size, row_group_size)
          row_group_size = 1
        else
          ! Just prepare for the upcoming row
          row_group_size = row_group_size + 1
        endif
      endif
      ! Always update the index for the upcoming row
      unpack_idx = next_unpack_idx
    end subroutine

    ! The host wrapper for computing the dot products between consecutive HH reflectors (see the kernel below)
    subroutine compute_hh_dot_products_real_gpu(bcast_buffer_dev, hh_dot_dev, nbw, n)
      use cuda_c_kernel
      use precision
      use iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: bcast_buffer_dev, hh_dot_dev
      integer(kind=ik), value  :: nbw, n

      if (n .le. 1) return
      call launch_compute_hh_dotp_c_kernel_real( bcast_buffer_dev, hh_dot_dev, nbw, n)
    end subroutine

    ! The host wrapper for extracting "tau" from the HH reflectors (see the kernel below)
    subroutine extract_hh_tau_real_gpu(bcast_buffer_dev, hh_tau_dev, nbw, n, is_zero)
      use cuda_c_kernel
      use precision
      use iso_c_binding
      implicit none
      integer(kind=c_intptr_t) :: bcast_buffer_dev, hh_tau_dev
      integer(kind=ik), value  :: nbw, n
      logical, value           :: is_zero
      integer(kind=ik)         :: val_is_zero
      if (is_zero) then
      val_is_zero = 1
      else
       val_is_zero = 0
      endif

      call launch_extract_hh_tau_c_kernel_real(bcast_buffer_dev, hh_tau_dev, nbw, n, val_is_zero)
    end subroutine

    ! -------------------------------------------
    ! Fortran back-transformation support kernels
    ! -------------------------------------------

    ! Reset a reduction block
    ! Limitation: the thread-block size must be a divider of the reduction block's size
    ! Reset 2 reduction blocks without an explicit synchronization at the end
    ! Limitation: : the thread-block size must be a divider of the reduction block's size
    ! Perform a reduction on an initialized, 128-element shared block
    ! Compute the dot-product between 2 consecutive HH vectors
    ! Limitation 1: the size of the thread block must be at most 128 and a power-of-2
    ! Limitation 2: the size of the warp must be equal to 32
    !
    ! Extract "tau" from the HH matrix and replace it with 1.0 or 0.0 (depending on case)
    ! Having "tau" as the first element in a HH reflector reduces space requirements, but causes undesired branching in the kernels
    !
    ! -------------------------------------------
    ! Fortran back-transformation support kernels
    ! -------------------------------------------
    !
    ! This is the simplest and slowest available backtransformation kernel
    !
    ! This is an improved version of the simple backtransformation kernel; here, we halve the number of iterations and apply
    ! 2 Householder reflectors per iteration
    !
    ! ---------------------------------
    ! Row packing and unpacking kernels
    ! ---------------------------------
    !
    ! The row group packing kernel

        ! Host wrapper for the Householder backtransformation step. Several kernels are available. Performance note:
        ! - "compute_hh_trafo_c_kernel" is the C kernel for the backtransformation (this exhibits best performance)
        ! - "compute_hh_trafo_kernel" is the Fortran equivalent of the C kernel
        ! - "compute_hh_trafo_single_kernel" is the reference Fortran kernel


end module
