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
! This file was written by A. Marek, MPCDF
#endif

subroutine compute_hh_trafo_&
&MATH_DATATYPE&
#ifdef WITH_OPENMP_TRADITIONAL
&_openmp_&
#else
&_&
#endif
&PRECISION &
(obj, useGPU, wantDebug, a, a_dev, stripe_width, a_dim2, stripe_count, max_threads, &
#ifdef WITH_OPENMP_TRADITIONAL
l_nev, &
#endif
a_off, nbw, max_blk_size, bcast_buffer, bcast_buffer_dev, &
hh_tau_dev, kernel_flops, kernel_time, n_times, off, ncols, istripe, &
#ifdef WITH_OPENMP_TRADITIONAL
my_thread, thread_width, kernel, last_stripe_width)
#else
last_stripe_width, kernel)
#endif

  use precision
  use elpa_abstract_impl
  use, intrinsic :: iso_c_binding
#if REALCASE == 1
  use single_hh_trafo_real
#if defined(WITH_REAL_GENERIC_SIMPLE_KERNEL) && !(defined(USE_ASSUMED_SIZE))
  use real_generic_simple_kernel !, only : double_hh_trafo_generic_simple
#endif

#if defined(WITH_REAL_GENERIC_SIMPLE_BLOCK4_KERNEL) && !(defined(USE_ASSUMED_SIZE))
  use real_generic_simple_block4_kernel !, only : double_hh_trafo_generic_simple
#endif

!#if defined(WITH_REAL_GENERIC_SIMPLE_BLOCK6_KERNEL) && !(defined(USE_ASSUMED_SIZE))
!  use real_generic_simple_block6_kernel !, only : double_hh_trafo_generic_simple
!#endif

#if defined(WITH_REAL_GENERIC_KERNEL) && !(defined(USE_ASSUMED_SIZE))
  use real_generic_kernel !, only : double_hh_trafo_generic
#endif

#if defined(WITH_REAL_BGP_KERNEL)
  use real_bgp_kernel !, only : double_hh_trafo_bgp
#endif

#if defined(WITH_REAL_BGQ_KERNEL)
  use real_bgq_kernel !, only : double_hh_trafo_bgq
#endif

#endif /* REALCASE */

#if COMPLEXCASE == 1

#if defined(WITH_COMPLEX_GENERIC_SIMPLE_KERNEL) && !(defined(USE_ASSUMED_SIZE))
  use complex_generic_simple_kernel !, only : single_hh_trafo_complex_generic_simple
#endif
#if defined(WITH_COMPLEX_GENERIC_KERNEL) && !(defined(USE_ASSUMED_SIZE))
  use complex_generic_kernel !, only : single_hh_trafo_complex_generic
#endif

#endif /* COMPLEXCASE */

  !use cuda_c_kernel
  !use cuda_functions
  !use hip_functions
  use gpu_c_kernel
  use elpa_gpu

  use elpa_generated_fortran_interfaces

  implicit none
  class(elpa_abstract_impl_t), intent(inout) :: obj
  logical, intent(in)                        :: useGPU, wantDebug
  real(kind=c_double), intent(inout)         :: kernel_time  ! MPI_WTIME always needs double
  integer(kind=lik)                          :: kernel_flops
  integer(kind=ik), intent(in)               :: nbw, max_blk_size
#if REALCASE == 1
  real(kind=C_DATATYPE_KIND)                 :: bcast_buffer(nbw,max_blk_size)
#endif
#if COMPLEXCASE == 1
  complex(kind=C_DATATYPE_KIND)              :: bcast_buffer(nbw,max_blk_size)
#endif
  integer(kind=ik), intent(in)               :: a_off

  integer(kind=ik), intent(in)               :: stripe_width,a_dim2,stripe_count

  integer(kind=ik), intent(in)               :: max_threads
#ifndef WITH_OPENMP_TRADITIONAL
  integer(kind=ik), intent(in)               :: last_stripe_width
#if REALCASE == 1
!  real(kind=C_DATATYPE_KIND)                :: a(stripe_width,a_dim2,stripe_count)
  real(kind=C_DATATYPE_KIND), pointer        :: a(:,:,:)
#endif
#if COMPLEXCASE == 1
!  complex(kind=C_DATATYPE_KIND)            :: a(stripe_width,a_dim2,stripe_count)
  complex(kind=C_DATATYPE_KIND),pointer     :: a(:,:,:)
#endif

#else /* WITH_OPENMP_TRADITIONAL */
  integer(kind=ik), intent(in)               :: l_nev, thread_width
  integer(kind=ik), intent(in), optional     :: last_stripe_width
#if REALCASE == 1
!  real(kind=C_DATATYPE_KIND)                :: a(stripe_width,a_dim2,stripe_count,max_threads)
  real(kind=C_DATATYPE_KIND), pointer        :: a(:,:,:,:)
#endif
#if COMPLEXCASE == 1
!  complex(kind=C_DATATYPE_KIND)            :: a(stripe_width,a_dim2,stripe_count,max_threads)
  complex(kind=C_DATATYPE_KIND),pointer     :: a(:,:,:,:)
#endif

#endif /* WITH_OPENMP_TRADITIONAL */

  integer(kind=ik), intent(in)               :: kernel

  integer(kind=c_intptr_t)                   :: a_dev
  integer(kind=c_intptr_t)                   :: bcast_buffer_dev
  integer(kind=c_intptr_t)                   :: hh_tau_dev
  integer(kind=c_intptr_t)                   :: dev_offset, dev_offset_1, dev_offset_2

  ! Private variables in OMP regions (my_thread) should better be in the argument list!
  integer(kind=ik)                           :: off, ncols, istripe
#ifdef WITH_OPENMP_TRADITIONAL
  integer(kind=ik)                           :: my_thread, noff
#endif
  integer(kind=ik)                           :: j, nl, jj, jjj, n_times
#if REALCASE == 1
  real(kind=C_DATATYPE_KIND)                 :: w(nbw,6)
#endif
#if COMPLEXCASE == 1
  complex(kind=C_DATATYPE_KIND)              :: w(nbw,2)
#endif
  real(kind=c_double)                        :: ttt ! MPI_WTIME always needs double

  integer(kind=c_intptr_t), parameter        :: size_of_datatype = size_of_&
                                                                 &PRECISION&
                                                                 &_&
                                                                 &MATH_DATATYPE


  j = -99

  if (wantDebug) then
#ifdef WITH_NVIDIA_GPU_VERSION
    if (useGPU .and. &
#if REALCASE == 1
      ( kernel .ne. ELPA_2STAGE_REAL_NVIDIA_GPU)) then
#endif
#if COMPLEXCASE == 1
      ( kernel .ne. ELPA_2STAGE_COMPLEX_NVIDIA_GPU)) then
#endif
      print *,"ERROR: useGPU is set in compute_hh_trafo but not a NVIDIA GPU kernel!"
      stop
    endif
#endif
#ifdef WITH_AMD_GPU_VERSION
    if (useGPU .and. &
#if REALCASE == 1
      ( kernel .ne. ELPA_2STAGE_REAL_AMD_GPU)) then
#endif
#if COMPLEXCASE == 1
      ( kernel .ne. ELPA_2STAGE_COMPLEX_AMD_GPU)) then
#endif
      print *,"ERROR: useGPU is set in compute_hh_trafo but not a AMD GPU kernel!"
      stop
    endif
#endif
  endif

  ! intel missing
#if REALCASE == 1
  if (kernel .eq. ELPA_2STAGE_REAL_NVIDIA_GPU .or. kernel .eq. ELPA_2STAGE_REAL_AMD_GPU) then
#endif
#if COMPLEXCASE == 1
  if (kernel .eq. ELPA_2STAGE_COMPLEX_NVIDIA_GPU .or. kernel .eq. ELPA_2STAGE_COMPLEX_AMD_GPU) then
#endif
    ! ncols - indicates the number of HH reflectors to apply; at least 1 must be available
    if (ncols < 1) then
      if (wantDebug) then
        print *, "Returning early from compute_hh_trafo"
      endif
      return
    endif
  endif

  if (wantDebug) call obj%timer%start("compute_hh_trafo_&
  &MATH_DATATYPE&
#ifdef WITH_OPENMP_TRADITIONAL
  &_openmp" // &
#else
  &" // &
#endif
  &PRECISION_SUFFIX &
  )


#ifdef WITH_OPENMP_TRADITIONAL
  if (my_thread==1) then ! in the calling routine threads go form 1 .. max_threads
#endif
    ttt = mpi_wtime()
#ifdef WITH_OPENMP_TRADITIONAL
  endif
#endif


#ifndef WITH_OPENMP_TRADITIONAL
  nl = merge(stripe_width, last_stripe_width, istripe<stripe_count)
#else /* WITH_OPENMP_TRADITIONAL */

  if (present(last_stripe_width)) then
    nl = merge(stripe_width, last_stripe_width, istripe<stripe_count)
  else
    if (istripe<stripe_count) then
      nl = stripe_width
    else
      noff = (my_thread-1)*thread_width + (istripe-1)*stripe_width
      nl = min(my_thread*thread_width-noff, l_nev-noff)
      if (nl<=0) then
        if (wantDebug) call obj%timer%stop("compute_hh_trafo_&
        &MATH_DATATYPE&
#ifdef WITH_OPENMP_TRADITIONAL
        &_openmp" // &
#else
        &" // &
#endif
        &PRECISION_SUFFIX &
        )

        return
      endif
    endif
  endif
#endif /* not WITH_OPENMP_TRADITIONAL */

#if REALCASE == 1
! GPU kernel real
  if (kernel .eq. ELPA_2STAGE_REAL_NVIDIA_GPU .or. kernel .eq. ELPA_2STAGE_REAL_AMD_GPU) then
#endif
#if COMPLEXCASE == 1
! GPU kernel complex
  if (kernel .eq. ELPA_2STAGE_COMPLEX_NVIDIA_GPU .or. kernel .eq. ELPA_2STAGE_COMPLEX_AMD_GPU) then
#endif
    if (wantDebug) then
      call obj%timer%start("compute_hh_trafo: GPU")
    endif

    dev_offset = ((a_off+off)*stripe_width+(istripe-1)*stripe_width*a_dim2)*size_of_datatype

    dev_offset_1 = off*nbw*size_of_datatype

    dev_offset_2 = off*size_of_datatype

    call launch_compute_hh_trafo_gpu_kernel_&
         &MATH_DATATYPE&
         &_&
         &PRECISION&
         &(a_dev + dev_offset, bcast_buffer_dev + dev_offset_1, &
         hh_tau_dev + dev_offset_2, nl, nbw,stripe_width, ncols)

    if (wantDebug) then
      call obj%timer%stop("compute_hh_trafo: GPU")
    endif

  else ! not CUDA kernel

    if (wantDebug) then
      call obj%timer%start("compute_hh_trafo: CPU")
    endif
#if REALCASE == 1
#ifndef WITH_FIXED_REAL_KERNEL
    if (kernel .eq. ELPA_2STAGE_REAL_AVX_BLOCK2 .or. &
        kernel .eq. ELPA_2STAGE_REAL_AVX2_BLOCK2 .or. &
        kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK2 .or. &
        kernel .eq. ELPA_2STAGE_REAL_SSE_BLOCK2 .or. &
        kernel .eq. ELPA_2STAGE_REAL_SPARC64_BLOCK2 .or. &
        kernel .eq. ELPA_2STAGE_REAL_NEON_ARCH64_BLOCK2 .or. &
        kernel .eq. ELPA_2STAGE_REAL_VSX_BLOCK2 .or. &
        kernel .eq. ELPA_2STAGE_REAL_SVE128_BLOCK2 .or. &
        kernel .eq. ELPA_2STAGE_REAL_SVE256_BLOCK2 .or. &
        kernel .eq. ELPA_2STAGE_REAL_SVE512_BLOCK2 .or. &
        kernel .eq. ELPA_2STAGE_REAL_GENERIC    .or. &
        kernel .eq. ELPA_2STAGE_REAL_GENERIC_SIMPLE .or. &
        kernel .eq. ELPA_2STAGE_REAL_SSE_ASSEMBLY .or. &
        kernel .eq. ELPA_2STAGE_REAL_BGP .or.        &
        kernel .eq. ELPA_2STAGE_REAL_BGQ) then
#endif /* not WITH_FIXED_REAL_KERNEL */

#endif /* REALCASE */

      !FORTRAN CODE / X86 INRINISIC CODE / BG ASSEMBLER USING 2 HOUSEHOLDER VECTORS
#if REALCASE == 1
      ! generic kernel real case
#if defined(WITH_REAL_GENERIC_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
      if (kernel .eq. ELPA_2STAGE_REAL_GENERIC) then
#endif /* not WITH_FIXED_REAL_KERNEL */

        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)

#ifdef WITH_OPENMP_TRADITIONAL

#ifdef USE_ASSUMED_SIZE
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_generic_&
          &PRECISION&
          & (a(1,j+off+a_off-1,istripe,my_thread), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_generic_&
          &PRECISION&
          & (a(1:stripe_width,j+off+a_off-1:j+off+a_off+nbw-1, istripe,my_thread), w(1:nbw,1:6), &
          nbw, nl, stripe_width, nbw)
#endif

#else /* WITH_OPENMP_TRADITIONAL */

#ifdef USE_ASSUMED_SIZE
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_generic_&
          &PRECISION&
          & (a(1,j+off+a_off-1,istripe),w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_generic_&
          &PRECISION&
          & (a(1:stripe_width,j+off+a_off-1:j+off+a_off+nbw-1,istripe),w(1:nbw,1:6), nbw, nl, stripe_width, nbw)
#endif
#endif /* WITH_OPENMP_TRADITIONAL */

        enddo

#ifndef WITH_FIXED_REAL_KERNEL
      endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_GENERIC_KERNEL */

#endif /* REALCASE == 1 */

#if COMPLEXCASE == 1
      ! generic kernel complex case
#if defined(WITH_COMPLEX_GENERIC_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
      if (kernel .eq. ELPA_2STAGE_COMPLEX_GENERIC .or. &
          kernel .eq. ELPA_2STAGE_COMPLEX_BGP .or. &
          kernel .eq. ELPA_2STAGE_COMPLEX_BGQ ) then
#endif /* not WITH_FIXED_COMPLEX_KERNEL */
          ttt = mpi_wtime()
          do j = ncols, 1, -1
#ifdef WITH_OPENMP_TRADITIONAL
#ifdef USE_ASSUMED_SIZE

            call single_hh_trafo_&
                 &MATH_DATATYPE&
                 &_generic_&
                 &PRECISION&
                 & (a(1,j+off+a_off,istripe,my_thread), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
            call single_hh_trafo_&
                 &MATH_DATATYPE&
                 &_generic_&
                 &PRECISION&
                 & (a(1:stripe_width,j+off+a_off:j+off+a_off+nbw-1,istripe,my_thread), &
                 bcast_buffer(1:nbw,j+off), nbw, nl, stripe_width)
#endif

#else /* WITH_OPENMP_TRADITIONAL */

#ifdef USE_ASSUMED_SIZE
            call single_hh_trafo_&
                 &MATH_DATATYPE&
                 &_generic_&
                 &PRECISION&
                 & (a(1,j+off+a_off,istripe), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
            call single_hh_trafo_&
                 &MATH_DATATYPE&
                 &_generic_&
                 &PRECISION&
                 & (a(1:stripe_width,j+off+a_off:j+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,j+off), &
                 nbw, nl, stripe_width)
#endif
#endif /* WITH_OPENMP_TRADITIONAL */

          enddo
#ifndef WITH_FIXED_COMPLEX_KERNEL
        endif ! (kernel .eq. ELPA_2STAGE_COMPLEX_GENERIC .or. kernel .eq. ELPA_2STAGE_COMPLEX_BGP .or. kernel .eq. ELPA_2STAGE_COMPLEX_BGQ )
#endif /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_GENERIC_KERNEL */

#endif /* COMPLEXCASE */

#if REALCASE == 1
        ! generic simple real kernel
#if defined(WITH_REAL_GENERIC_SIMPLE_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
        if (kernel .eq. ELPA_2STAGE_REAL_GENERIC_SIMPLE) then
#endif /* not WITH_FIXED_REAL_KERNEL */
          do j = ncols, 2, -2
            w(:,1) = bcast_buffer(1:nbw,j+off)
            w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL

#ifdef USE_ASSUMED_SIZE
            call double_hh_trafo_&
                 &MATH_DATATYPE&
                 &_generic_simple_&
                 &PRECISION&
                 & (a(1,j+off+a_off-1,istripe,my_thread), w, nbw, nl, stripe_width, nbw)
#else
            call double_hh_trafo_&
                 &MATH_DATATYPE&
                 &_generic_simple_&
                 &PRECISION&
                 & (a(1:stripe_width,j+off+a_off-1:j+off+a_off-1+nbw,istripe,my_thread), w, nbw, nl, stripe_width, nbw)

#endif

#else /* WITH_OPENMP_TRADITIONAL */

#ifdef USE_ASSUMED_SIZE
            call double_hh_trafo_&
                 &MATH_DATATYPE&
                 &_generic_simple_&
                 &PRECISION&
                 & (a(1,j+off+a_off-1,istripe), w, nbw, nl, stripe_width, nbw)
#else
            call double_hh_trafo_&
                 &MATH_DATATYPE&
                 &_generic_simple_&
                 &PRECISION&
                 & (a(1:stripe_width,j+off+a_off-1:j+off+a_off-1+nbw,istripe), w, nbw, nl, stripe_width, nbw)
#endif

#endif /* WITH_OPENMP_TRADITIONAL */

          enddo
#ifndef WITH_FIXED_REAL_KERNEL
        endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_GENERIC_SIMPLE_KERNEL */

#endif /* REALCASE */

#if COMPLEXCASE == 1
        ! generic simple complex case

#if defined(WITH_COMPLEX_GENERIC_SIMPLE_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
        if (kernel .eq. ELPA_2STAGE_COMPLEX_GENERIC_SIMPLE) then
#endif /* not WITH_FIXED_COMPLEX_KERNEL */
          ttt = mpi_wtime()
          do j = ncols, 1, -1
#ifdef WITH_OPENMP_TRADITIONAL
#ifdef USE_ASSUMED_SIZE
            call single_hh_trafo_&
                 &MATH_DATATYPE&
                 &_generic_simple_&
                 &PRECISION&
                 & (a(1,j+off+a_off,istripe,my_thread), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
            call single_hh_trafo_&
                 &MATH_DATATYPE&
                 &_generic_simple_&
                 &PRECISION&
                 & (a(1:stripe_width, j+off+a_off:j+off+a_off+nbw-1,istripe,my_thread), bcast_buffer(1:nbw,j+off), &
                 nbw, nl, stripe_width)
#endif

#else /* WITH_OPENMP_TRADITIONAL */

#ifdef USE_ASSUMED_SIZE
            call single_hh_trafo_&
                 &MATH_DATATYPE&
                 &_generic_simple_&
                 &PRECISION&
                 & (a(1,j+off+a_off,istripe), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
            call single_hh_trafo_&
                 &MATH_DATATYPE&
                 &_generic_simple_&
                 &PRECISION&
                 & (a(1:stripe_width,j+off+a_off:j+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,j+off), &
                 nbw, nl, stripe_width)
#endif

#endif /* WITH_OPENMP_TRADITIONAL */
          enddo
#ifndef WITH_FIXED_COMPLEX_KERNEL
        endif ! (kernel .eq. ELPA_2STAGE_COMPLEX_GENERIC_SIMPLE)
#endif /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_GENERIC_SIMPLE_KERNEL */

#endif /* COMPLEXCASE */

#if REALCASE == 1
        ! sse assembly kernel real case
#if defined(WITH_REAL_SSE_ASSEMBLY_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
        if (kernel .eq. ELPA_2STAGE_REAL_SSE_ASSEMBLY) then

#endif /* not WITH_FIXED_REAL_KERNEL */
          do j = ncols, 2, -2
            w(:,1) = bcast_buffer(1:nbw,j+off)
            w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
            call double_hh_trafo_&
            &MATH_DATATYPE&
            &_&
            &PRECISION&
            &_sse_assembly&
            & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
            call double_hh_trafo_&
            &MATH_DATATYPE&
            &_&
            &PRECISION&
            &_sse_assembly&
            & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
          enddo
#ifndef WITH_FIXED_REAL_KERNEL
        endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_SSE_ASSEMBLY_KERNEL */

#endif /* REALCASE */

#if COMPLEXCASE == 1

        ! sse assembly kernel complex case
#if defined(WITH_COMPLEX_SSE_ASSEMBLY_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
        if (kernel .eq. ELPA_2STAGE_COMPLEX_SSE_ASSEMBLY) then
#endif /* not WITH_FIXED_COMPLEX_KERNEL */
          ttt = mpi_wtime()
          do j = ncols, 1, -1
#ifdef WITH_OPENMP_TRADITIONAL
            call single_hh_trafo_&
            &MATH_DATATYPE&
            &_&
            &PRECISION&
            &_sse_assembly&
            & (c_loc(a(1,j+off+a_off,istripe,my_thread)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
            call single_hh_trafo_&
            &MATH_DATATYPE&
            &_&
            &PRECISION&
            &_sse_assembly&
            & (c_loc(a(1,j+off+a_off,istripe)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#endif
          enddo
#ifndef WITH_FIXED_COMPLEX_KERNEL
        endif ! (kernel .eq. ELPA_2STAGE_COMPLEX_SSE)
#endif /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_SSE_ASSEMBLY_KERNEL */
#endif /* COMPLEXCASE */

#if REALCASE == 1
        ! no sse, vsx, sparc64 sve block1 real kernel
#endif

#if COMPLEXCASE == 1

        ! sparc64 block1 complex kernel
#if defined(WITH_COMPLEX_SPARC64_BLOCK1_KERNEL)
!#ifndef WITH_FIXED_COMPLEX_KERNEL
!        if (kernel .eq. ELPA_2STAGE_COMPLEX_SPARC64_BLOCK1) then
!#endif /* not WITH_FIXED_COMPLEX_KERNEL */
!
!#if (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_SPARC64_BLOCK2_KERNEL))
!        ttt = mpi_wtime()
!        do j = ncols, 1, -1
!#ifdef WITH_OPENMP_TRADITIONAL
!          call single_hh_trafo_&
!          &MATH_DATATYPE&
!          &_sparc64_1hv_&
!          &PRECISION&
!          & (c_loc(a(1,j+off+a_off,istripe,my_thread)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
!#else
!          call single_hh_trafo_&
!          &MATH_DATATYPE&
!          &_sparc64_1hv_&
!          &PRECISION&
!          & (c_loc(a(1,j+off+a_off,istripe)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
!#endif
!        enddo
!#endif /* (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_SPARC64_BLOCK2_KERNEL)) */
!
!#ifndef WITH_FIXED_COMPLEX_KERNEL
!      endif ! (kernel .eq. ELPA_2STAGE_COMPLEX_SPARC64_BLOCK1)
!#endif /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_SPARC64_BLOCK1_KERNEL */

#endif /* COMPLEXCASE */


#if COMPLEXCASE == 1

      ! vsx block1 complex kernel
#if defined(WITH_COMPLEX_VSX_BLOCK1_KERNEL)
!#ifndef WITH_FIXED_COMPLEX_KERNEL
!      if (kernel .eq. ELPA_2STAGE_COMPLEX_VSX_BLOCK1) then
!#endif /* not WITH_FIXED_COMPLEX_KERNEL */
!
!#if (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_VSX_BLOCK2_KERNEL))
!        ttt = mpi_wtime()
!        do j = ncols, 1, -1
!#ifdef WITH_OPENMP_TRADITIONAL
!          call single_hh_trafo_&
!          &MATH_DATATYPE&
!          &_vsx_1hv_&
!          &PRECISION&
!          & (c_loc(a(1,j+off+a_off,istripe,my_thread)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
!#else
!          call single_hh_trafo_&
!          &MATH_DATATYPE&
!          &_vsx_1hv_&
!          &PRECISION&
!          & (c_loc(a(1,j+off+a_off,istripe)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
!#endif
!        enddo
!#endif /* (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_VSX_BLOCK2_KERNEL)) */
!
!#ifndef WITH_FIXED_COMPLEX_KERNEL
!      endif ! (kernel .eq. ELPA_2STAGE_COMPLEX_VSX_BLOCK1)
!#endif /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_VSX_BLOCK1_KERNEL */

#endif /* COMPLEXCASE */


#if COMPLEXCASE == 1

      ! sse block1 complex kernel
#if defined(WITH_COMPLEX_SSE_BLOCK1_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
      if (kernel .eq. ELPA_2STAGE_COMPLEX_SSE_BLOCK1) then
#endif /* not WITH_FIXED_COMPLEX_KERNEL */

#if (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_SSE_BLOCK2_KERNEL))
        ttt = mpi_wtime()
        do j = ncols, 1, -1
#ifdef WITH_OPENMP_TRADITIONAL
          call single_hh_trafo_&
          &MATH_DATATYPE&
          &_sse_1hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off,istripe,my_thread)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
          call single_hh_trafo_&
          &MATH_DATATYPE&
          &_sse_1hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off,istripe)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_SSE_BLOCK2_KERNEL)) */

#ifndef WITH_FIXED_COMPLEX_KERNEL
      endif ! (kernel .eq. ELPA_2STAGE_COMPLEX_SSE_BLOCK1)
#endif /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_SSE_BLOCK1_KERNEL */

      ! neon_arch64 block1 complex kernel
#if defined(WITH_COMPLEX_NEON_ARCH64_BLOCK1_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
      if (kernel .eq. ELPA_2STAGE_COMPLEX_NEON_ARCH64_BLOCK1) then
#endif /* not WITH_FIXED_COMPLEX_KERNEL */

#if (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_NEON_ARCH64_BLOCK2_KERNEL))
        ttt = mpi_wtime()
        do j = ncols, 1, -1
#ifdef WITH_OPENMP_TRADITIONAL
          call single_hh_trafo_&
          &MATH_DATATYPE&
          &_neon_arch64_1hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off,istripe,my_thread)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
          call single_hh_trafo_&
          &MATH_DATATYPE&
          &_neon_arch64_1hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off,istripe)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_NEON_ARCH64_BLOCK2_KERNEL)) */

#ifndef WITH_FIXED_COMPLEX_KERNEL
      endif ! (kernel .eq. ELPA_2STAGE_COMPLEX_NEON_ARCH64_BLOCK1)
#endif /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_NEON_ARCH64_BLOCK1_KERNEL */

      ! sve128 block1 complex kernel
#if defined(WITH_COMPLEX_SVE128_BLOCK1_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
      if (kernel .eq. ELPA_2STAGE_COMPLEX_SVE128_BLOCK1) then
#endif /* not WITH_FIXED_COMPLEX_KERNEL */

#if (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_SVE128_BLOCK2_KERNEL))
        ttt = mpi_wtime()
        do j = ncols, 1, -1
#ifdef WITH_OPENMP_TRADITIONAL
          call single_hh_trafo_&
          &MATH_DATATYPE&
          &_sve128_1hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off,istripe,my_thread)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
          call single_hh_trafo_&
          &MATH_DATATYPE&
          &_sve128_1hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off,istripe)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_SVE128_BLOCK2_KERNEL)) */

#ifndef WITH_FIXED_COMPLEX_KERNEL
      endif ! (kernel .eq. ELPA_2STAGE_COMPLEX_SVE128_BLOCK1)
#endif /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_SVE128_BLOCK1_KERNEL */

#endif /* COMPLEXCASE */

#if REALCASE == 1
      !no avx block1 real kernel
#endif /* REALCASE */

#if COMPLEXCASE == 1

      ! avx block1 complex kernel
#if defined(WITH_COMPLEX_AVX_BLOCK1_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
      if ((kernel .eq. ELPA_2STAGE_COMPLEX_AVX_BLOCK1)) then
#endif /* not WITH_FIXED_COMPLEX_KERNEL */

#if (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_AVX_BLOCK2_KERNEL) )
        ttt = mpi_wtime()
        do j = ncols, 1, -1
#ifdef WITH_OPENMP_TRADITIONAL
          call single_hh_trafo_&
          &MATH_DATATYPE&
          &_avx_1hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off,istripe,my_thread)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
          call single_hh_trafo_&
          &MATH_DATATYPE&
          &_avx_1hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off,istripe)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_AVX_BLOCK2_KERNEL)) */

#ifndef WITH_FIXED_COMPLEX_KERNEL
      endif ! ((kernel .eq. ELPA_2STAGE_COMPLEX_AVX_BLOCK1) )
#endif /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_AVX_BLOCK1_KERNEL */

#if defined(WITH_COMPLEX_AVX2_BLOCK1_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
      if ((kernel .eq. ELPA_2STAGE_COMPLEX_AVX2_BLOCK1)) then
#endif /* not WITH_FIXED_COMPLEX_KERNEL */

#if (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_AVX2_BLOCK2_KERNEL))
        ttt = mpi_wtime()
        do j = ncols, 1, -1
#ifdef WITH_OPENMP_TRADITIONAL
          call single_hh_trafo_&
          &MATH_DATATYPE&
          &_avx2_1hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off,istripe,my_thread)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
          call single_hh_trafo_&
          &MATH_DATATYPE&
          &_avx2_1hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off,istripe)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_AVX2_BLOCK2_KERNEL)) */

#ifndef WITH_FIXED_COMPLEX_KERNEL
      endif ! ((kernel .eq. ELPA_2STAGE_COMPLEX_AVX2_BLOCK1))
#endif /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_AVX2_BLOCK1_KERNEL */

#if defined(WITH_COMPLEX_SVE256_BLOCK1_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
      if ((kernel .eq. ELPA_2STAGE_COMPLEX_SVE256_BLOCK1)) then
#endif /* not WITH_FIXED_COMPLEX_KERNEL */

#if (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_SVE256_BLOCK2_KERNEL))
        ttt = mpi_wtime()
        do j = ncols, 1, -1
#ifdef WITH_OPENMP_TRADITIONAL
          call single_hh_trafo_&
          &MATH_DATATYPE&
          &_sve256_1hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off,istripe,my_thread)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
          call single_hh_trafo_&
          &MATH_DATATYPE&
          &_sve256_1hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off,istripe)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_SVE256_BLOCK2_KERNEL)) */

#ifndef WITH_FIXED_COMPLEX_KERNEL
      endif ! ((kernel .eq. ELPA_2STAGE_COMPLEX_SVE256_BLOCK1))
#endif /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_SVE256_BLOCK1_KERNEL */

#endif /* COMPLEXCASE */

#if REALCASE == 1
      ! no avx512 block1 real kernel
      ! no sve512 block1 real kernel
#endif /* REALCASE */

#if COMPLEXCASE == 1

      ! avx512 block1 complex kernel
#if defined(WITH_COMPLEX_AVX512_BLOCK1_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
      if ((kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK1)) then
#endif /* not WITH_FIXED_COMPLEX_KERNEL */

#if (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_AVX512_BLOCK2_KERNEL) )
        ttt = mpi_wtime()
        do j = ncols, 1, -1
#ifdef WITH_OPENMP_TRADITIONAL
          call single_hh_trafo_&
          &MATH_DATATYPE&
          &_avx512_1hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off,istripe,my_thread)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
          call single_hh_trafo_&
          &MATH_DATATYPE&
          &_avx512_1hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off,istripe)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_AVX512_BLOCK2_KERNEL) ) */

#ifndef WITH_FIXED_COMPLEX_KERNEL
      endif ! ((kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK1))
#endif /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_AVX512_BLOCK1_KERNEL  */

      ! sve512 block1 complex kernel
#if defined(WITH_COMPLEX_SVE512_BLOCK1_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
      if ((kernel .eq. ELPA_2STAGE_COMPLEX_SVE512_BLOCK1)) then
#endif /* not WITH_FIXED_COMPLEX_KERNEL */

#if (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_SVE512_BLOCK2_KERNEL) )
        ttt = mpi_wtime()
        do j = ncols, 1, -1
#ifdef WITH_OPENMP_TRADITIONAL
          call single_hh_trafo_&
          &MATH_DATATYPE&
          &_sve512_1hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off,istripe,my_thread)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
          call single_hh_trafo_&
          &MATH_DATATYPE&
          &_sve512_1hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off,istripe)), bcast_buffer(1,j+off),nbw,nl,stripe_width)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_COMPLEX_KERNEL)) || (defined(WITH_FIXED_COMPLEX_KERNEL) && !defined(WITH_COMPLEX_SVE512_BLOCK2_KERNEL) ) */

#ifndef WITH_FIXED_COMPLEX_KERNEL
      endif ! ((kernel .eq. ELPA_2STAGE_COMPLEX_SVE512_BLOCK1))
#endif /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_SVE512_BLOCK1_KERNEL  */

#endif /* COMPLEXCASE */

#if REALCASE == 1
      ! implementation of sparc64 block 2 real case
#if defined(WITH_REAL_SPARC64_BLOCK2_KERNEL)

#ifndef WITH_FIXED_REAL_KERNEL
      if (kernel .eq. ELPA_2STAGE_REAL_SPARC64_BLOCK2) then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_SPARC64_BLOCK6_KERNEL) && !defined(WITH_REAL_SPARC64_BLOCK4_KERNEL))
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sparc64_2hv_&
          &PRECISION &
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sparc64_2hv_&
          &PRECISION &
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_SPARC64_BLOCK6_KERNEL) && !defined(WITH_REAL_SPARC64_BLOCK4_KERNEL)) */

#ifndef WITH_FIXED_REAL_KERNEL
      endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_SPARC64_BLOCK2_KERNEL */

#endif /* REALCASE == 1 */

#if REALCASE == 1
      ! implementation of neon_arch64 block 2 real case
#if defined(WITH_REAL_NEON_ARCH64_BLOCK2_KERNEL)

#ifndef WITH_FIXED_REAL_KERNEL
      if (kernel .eq. ELPA_2STAGE_REAL_NEON_ARCH64_BLOCK2) then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_NEON_ARCH64_BLOCK6_KERNEL) && !defined(WITH_REAL_NEON_ARCH64_BLOCK4_KERNEL))
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_neon_arch64_2hv_&
          &PRECISION &
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_neon_arch64_2hv_&
          &PRECISION &
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_NEON_ARCH64_BLOCK6_KERNEL) && !defined(WITH_REAL_NEON_ARCH64_BLOCK4_KERNEL)) */

#ifndef WITH_FIXED_REAL_KERNEL
      endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_NEON_ARCH64_BLOCK2_KERNEL */

      ! implementation of neon_arch64 block 2 real case
#if defined(WITH_REAL_SVE128_BLOCK2_KERNEL)

#ifndef WITH_FIXED_REAL_KERNEL
      if (kernel .eq. ELPA_2STAGE_REAL_SVE128_BLOCK2) then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_SVE128_BLOCK6_KERNEL) && !defined(WITH_REAL_SVE128_BLOCK4_KERNEL))
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sve128_2hv_&
          &PRECISION &
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sve128_2hv_&
          &PRECISION &
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_SVE128_BLOCK6_KERNEL) && !defined(WITH_REAL_SVE128_BLOCK4_KERNEL)) */

#ifndef WITH_FIXED_REAL_KERNEL
      endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_SVE128_BLOCK2_KERNEL */

#endif /* REALCASE == 1 */


#if REALCASE == 1
      ! implementation of vsx block 2 real case
#if defined(WITH_REAL_VSX_BLOCK2_KERNEL)

#ifndef WITH_FIXED_REAL_KERNEL
      if (kernel .eq. ELPA_2STAGE_REAL_VSX_BLOCK2) then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_VSX_BLOCK6_KERNEL) && !defined(WITH_REAL_VSX_BLOCK4_KERNEL))
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_vsx_2hv_&
          &PRECISION &
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_vsx_2hv_&
          &PRECISION &
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_VSX_BLOCK6_KERNEL) && !defined(WITH_REAL_VSX_BLOCK4_KERNEL)) */

#ifndef WITH_FIXED_REAL_KERNEL
      endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_VSX_BLOCK2_KERNEL */

#endif /* REALCASE == 1 */

#if REALCASE == 1
      ! implementation of sse block 2 real case
#if defined(WITH_REAL_SSE_BLOCK2_KERNEL)

#ifndef WITH_FIXED_REAL_KERNEL
      if (kernel .eq. ELPA_2STAGE_REAL_SSE_BLOCK2) then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_SSE_BLOCK6_KERNEL) && !defined(WITH_REAL_SSE_BLOCK4_KERNEL))
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sse_2hv_&
          &PRECISION &
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sse_2hv_&
          &PRECISION &
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_SSE_BLOCK6_KERNEL) && !defined(WITH_REAL_SSE_BLOCK4_KERNEL)) */

#ifndef WITH_FIXED_REAL_KERNEL
      endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_SSE_BLOCK2_KERNEL */

#endif /* REALCASE == 1 */


#if COMPLEXCASE == 1
      ! implementation of sparc64 block 2 complex case

#if defined(WITH_COMPLEX_SPARC64_BLOCK2_KERNEL)
!#ifndef WITH_FIXED_COMPLEX_KERNEL
!      if (kernel .eq. ELPA_2STAGE_COMPLEX_SPARC64_BLOCK2) then
!#endif  /* not WITH_FIXED_COMPLEX_KERNEL */
!
!        ttt = mpi_wtime()
!        do j = ncols, 2, -2
!          w(:,1) = bcast_buffer(1:nbw,j+off)
!          w(:,2) = bcast_buffer(1:nbw,j+off-1)
!#ifdef WITH_OPENMP_TRADITIONAL
!          call double_hh_trafo_&
!          &MATH_DATATYPE&
!          &_sparc64_2hv_&
!          &PRECISION&
!          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
!#else
!          call double_hh_trafo_&
!          &MATH_DATATYPE&
!          &_sparc64_2hv_&
!          &PRECISION&
!          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
!#endif
!        enddo
!#ifdef WITH_OPENMP_TRADITIONAL
!        if (j==1) call single_hh_trafo_&
!        &MATH_DATATYPE&
!        &_sparc64_1hv_&
!        &PRECISION&
!        & (c_loc(a(1,1+off+a_off,istripe,my_thread)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
!#else
!        if (j==1) call single_hh_trafo_&
!        &MATH_DATATYPE&
!        &_sparc64_1hv_&
!        &PRECISION&
!        & (c_loc(a(1,1+off+a_off,istripe)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
!#endif
!
!#ifndef WITH_FIXED_COMPLEX_KERNEL
!      endif ! (kernel .eq. ELPA_2STAGE_COMPLEX_SPARC64_BLOCK2)
!#endif  /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_SPARC64_BLOCK2_KERNEL */
#endif /* COMPLEXCASE == 1 */


#if COMPLEXCASE == 1
      ! implementation of vsx block 2 complex case

#if defined(WITH_COMPLEX_VSX_BLOCK2_KERNEL)
!#ifndef WITH_FIXED_COMPLEX_KERNEL
!      if (kernel .eq. ELPA_2STAGE_COMPLEX_VSX_BLOCK2) then
!#endif  /* not WITH_FIXED_COMPLEX_KERNEL */
!
!        ttt = mpi_wtime()
!        do j = ncols, 2, -2
!          w(:,1) = bcast_buffer(1:nbw,j+off)
!          w(:,2) = bcast_buffer(1:nbw,j+off-1)
!#ifdef WITH_OPENMP_TRADITIONAL
!          call double_hh_trafo_&
!          &MATH_DATATYPE&
!          &_vsx_2hv_&
!          &PRECISION&
!          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
!#else
!          call double_hh_trafo_&
!          &MATH_DATATYPE&
!          &_vsx_2hv_&
!          &PRECISION&
!          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
!#endif
!        enddo
!#ifdef WITH_OPENMP_TRADITIONAL
!        if (j==1) call single_hh_trafo_&
!        &MATH_DATATYPE&
!        &_vsx_1hv_&
!        &PRECISION&
!        & (c_loc(a(1,1+off+a_off,istripe,my_thread)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
!#else
!        if (j==1) call single_hh_trafo_&
!        &MATH_DATATYPE&
!        &_vsx_1hv_&
!        &PRECISION&
!        & (c_loc(a(1,1+off+a_off,istripe)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
!#endif
!
!#ifndef WITH_FIXED_COMPLEX_KERNEL
!      endif ! (kernel .eq. ELPA_2STAGE_COMPLEX_VSX_BLOCK2)
!#endif  /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_VSX_BLOCK2_KERNEL */
#endif /* COMPLEXCASE == 1 */

#if COMPLEXCASE == 1
      ! implementation of sse block 2 complex case

#if defined(WITH_COMPLEX_SSE_BLOCK2_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
      if (kernel .eq. ELPA_2STAGE_COMPLEX_SSE_BLOCK2) then
#endif  /* not WITH_FIXED_COMPLEX_KERNEL */

        ttt = mpi_wtime()
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sse_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sse_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#ifdef WITH_OPENMP_TRADITIONAL
        if (j==1) call single_hh_trafo_&
        &MATH_DATATYPE&
        &_sse_1hv_&
        &PRECISION&
        & (c_loc(a(1,1+off+a_off,istripe,my_thread)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
#else
        if (j==1) call single_hh_trafo_&
        &MATH_DATATYPE&
        &_sse_1hv_&
        &PRECISION&
        & (c_loc(a(1,1+off+a_off,istripe)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
#endif

#ifndef WITH_FIXED_COMPLEX_KERNEL
      endif ! (kernel .eq. ELPA_2STAGE_COMPLEX_SSE_BLOCK2)
#endif  /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_SSE_BLOCK2_KERNEL */

      ! implementation of neon_arch64 block 2 complex case

#if defined(WITH_COMPLEX_NEON_ARCH64_BLOCK2_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
      if (kernel .eq. ELPA_2STAGE_COMPLEX_NEON_ARCH64_BLOCK2) then
#endif  /* not WITH_FIXED_COMPLEX_KERNEL */

        ttt = mpi_wtime()
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_neon_arch64_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_neon_arch64_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#ifdef WITH_OPENMP_TRADITIONAL
        if (j==1) call single_hh_trafo_&
        &MATH_DATATYPE&
        &_neon_arch64_1hv_&
        &PRECISION&
        & (c_loc(a(1,1+off+a_off,istripe,my_thread)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
#else
        if (j==1) call single_hh_trafo_&
        &MATH_DATATYPE&
        &_neon_arch64_1hv_&
        &PRECISION&
        & (c_loc(a(1,1+off+a_off,istripe)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
#endif

#ifndef WITH_FIXED_COMPLEX_KERNEL
      endif ! (kernel .eq. ELPA_2STAGE_COMPLEX_NEON_ARCH64_BLOCK2)
#endif  /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_NEON_ARCH64_BLOCK2_KERNEL */

      ! implementation of sve128 block 2 complex case

#if defined(WITH_COMPLEX_SVE128_BLOCK2_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
      if (kernel .eq. ELPA_2STAGE_COMPLEX_SVE128_BLOCK2) then
#endif  /* not WITH_FIXED_COMPLEX_KERNEL */

        ttt = mpi_wtime()
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sve128_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sve128_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#ifdef WITH_OPENMP_TRADITIONAL
        if (j==1) call single_hh_trafo_&
        &MATH_DATATYPE&
        &_sve128_1hv_&
        &PRECISION&
        & (c_loc(a(1,1+off+a_off,istripe,my_thread)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
#else
        if (j==1) call single_hh_trafo_&
        &MATH_DATATYPE&
        &_sve128_1hv_&
        &PRECISION&
        & (c_loc(a(1,1+off+a_off,istripe)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
#endif

#ifndef WITH_FIXED_COMPLEX_KERNEL
      endif ! (kernel .eq. ELPA_2STAGE_COMPLEX_SVE128_BLOCK2)
#endif  /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_SVE128_BLOCK2_KERNEL */

#endif /* COMPLEXCASE == 1 */

#if REALCASE == 1
      ! implementation of avx block 2 real case

#if defined(WITH_REAL_AVX_BLOCK2_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL

      if ((kernel .eq. ELPA_2STAGE_REAL_AVX_BLOCK2))  then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_AVX_BLOCK6_KERNEL) && !defined(WITH_REAL_AVX_BLOCK4_KERNEL) )
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL

          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_avx_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_avx_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) ... */

#ifndef WITH_FIXED_REAL_KERNEL
      endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_AVX_BLOCK2_KERNEL */

#if defined(WITH_REAL_AVX2_BLOCK2_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL

      if ((kernel .eq. ELPA_2STAGE_REAL_AVX2_BLOCK2))  then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_AVX2_BLOCK6_KERNEL) && !defined(WITH_REAL_AVX2_BLOCK4_KERNEL))
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL

          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_avx2_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_avx2_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) ... */

#ifndef WITH_FIXED_REAL_KERNEL
      endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_AVX2_BLOCK2_KERNEL */

#if defined(WITH_REAL_SVE256_BLOCK2_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL

      if ((kernel .eq. ELPA_2STAGE_REAL_SVE256_BLOCK2))  then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_SVE256_BLOCK6_KERNEL) && !defined(WITH_REAL_SVE256_BLOCK4_KERNEL))
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL

          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sve256_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sve256_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) ... */

#ifndef WITH_FIXED_REAL_KERNEL
      endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_SVE256_BLOCK2_KERNEL */

#endif /* REALCASE */

#if COMPLEXCASE == 1

      ! implementation of avx block 2 complex case
#if defined(WITH_COMPLEX_AVX_BLOCK2_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
      if ( (kernel .eq. ELPA_2STAGE_COMPLEX_AVX_BLOCK2) ) then
#endif  /* not WITH_FIXED_COMPLEX_KERNEL */

        ttt = mpi_wtime()
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_avx_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_avx_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#ifdef WITH_OPENMP_TRADITIONAL
        if (j==1) call single_hh_trafo_&
        &MATH_DATATYPE&
        &_avx_1hv_&
        &PRECISION&
        & (c_loc(a(1,1+off+a_off,istripe,my_thread)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
#else
        if (j==1) call single_hh_trafo_&
        &MATH_DATATYPE&
        &_avx_1hv_&
        &PRECISION&
        & (c_loc(a(1,1+off+a_off,istripe)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
#endif

#ifndef WITH_FIXED_COMPLEX_KERNEL
      endif ! ( (kernel .eq. ELPA_2STAGE_COMPLEX_AVX_BLOCK2) )
#endif  /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_AVX_BLOCK2_KERNEL */

      ! implementation of avx2 block 2 complex case
#if defined(WITH_COMPLEX_AVX2_BLOCK2_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
      if ( (kernel .eq. ELPA_2STAGE_COMPLEX_AVX2_BLOCK2) ) then
#endif  /* not WITH_FIXED_COMPLEX_KERNEL */

        ttt = mpi_wtime()
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_avx2_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_avx2_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#ifdef WITH_OPENMP_TRADITIONAL
        if (j==1) call single_hh_trafo_&
        &MATH_DATATYPE&
        &_avx2_1hv_&
        &PRECISION&
        & (c_loc(a(1,1+off+a_off,istripe,my_thread)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
#else
        if (j==1) call single_hh_trafo_&
        &MATH_DATATYPE&
        &_avx2_1hv_&
        &PRECISION&
        & (c_loc(a(1,1+off+a_off,istripe)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
#endif

#ifndef WITH_FIXED_COMPLEX_KERNEL
      endif ! ( (kernel .eq. ELPA_2STAGE_COMPLEX_AVX2_BLOCK2) )
#endif  /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /*  WITH_COMPLEX_AVX2_BLOCK2_KERNEL */

      ! implementation of sve256 block 2 complex case
#if defined(WITH_COMPLEX_SVE256_BLOCK2_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
      if ( (kernel .eq. ELPA_2STAGE_COMPLEX_SVE256_BLOCK2) ) then
#endif  /* not WITH_FIXED_COMPLEX_KERNEL */

        ttt = mpi_wtime()
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sve256_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sve256_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#ifdef WITH_OPENMP_TRADITIONAL
        if (j==1) call single_hh_trafo_&
        &MATH_DATATYPE&
        &_sve256_1hv_&
        &PRECISION&
        & (c_loc(a(1,1+off+a_off,istripe,my_thread)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
#else
        if (j==1) call single_hh_trafo_&
        &MATH_DATATYPE&
        &_sve256_1hv_&
        &PRECISION&
        & (c_loc(a(1,1+off+a_off,istripe)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
#endif

#ifndef WITH_FIXED_COMPLEX_KERNEL
      endif ! ( (kernel .eq. ELPA_2STAGE_COMPLEX_SVE256_BLOCK2) )
#endif  /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /*  WITH_COMPLEX_SVE256_BLOCK2_KERNEL */

#endif /* COMPLEXCASE */

#if REALCASE == 1
      ! implementation of avx512 block 2 real case

#if defined(WITH_REAL_AVX512_BLOCK2_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL

      if ((kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK2)) then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_AVX512_BLOCK6_KERNEL) && !defined(WITH_REAL_AVX512_BLOCK4_KERNEL))
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL

          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_avx512_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_avx512_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) ... */

#ifndef WITH_FIXED_REAL_KERNEL
      endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_AVX512_BLOCK2_KERNEL */


! implementation of sve512 block 2 real case

#if defined(WITH_REAL_SVE512_BLOCK2_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL

      if ((kernel .eq. ELPA_2STAGE_REAL_SVE512_BLOCK2)) then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_SVE512_BLOCK6_KERNEL) && !defined(WITH_REAL_SVE512_BLOCK4_KERNEL))
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL

          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sve512_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sve512_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) ... */

#ifndef WITH_FIXED_REAL_KERNEL
      endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_SVE512_BLOCK2_KERNEL */


#endif /* REALCASE */

#if COMPLEXCASE == 1

! implementation of avx512 block 2 complex case
#if defined(WITH_COMPLEX_AVX512_BLOCK2_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
      if ( (kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK2)) then
#endif  /* not WITH_FIXED_COMPLEX_KERNEL */

        ttt = mpi_wtime()
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_avx512_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_avx512_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#ifdef WITH_OPENMP_TRADITIONAL
        if (j==1) call single_hh_trafo_&
        &MATH_DATATYPE&
        &_avx512_1hv_&
        &PRECISION&
        & (c_loc(a(1,1+off+a_off,istripe,my_thread)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
#else
        if (j==1) call single_hh_trafo_&
        &MATH_DATATYPE&
        &_avx512_1hv_&
        &PRECISION&
        & (c_loc(a(1,1+off+a_off,istripe)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
#endif

#ifndef WITH_FIXED_COMPLEX_KERNEL
      endif ! ( (kernel .eq. ELPA_2STAGE_COMPLEX_AVX512_BLOCK2))
#endif  /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_AVX512_BLOCK2_KERNEL */

! implementation of vse512 block 2 complex case
#if defined(WITH_COMPLEX_SVE512_BLOCK2_KERNEL)
#ifndef WITH_FIXED_COMPLEX_KERNEL
      if ( (kernel .eq. ELPA_2STAGE_COMPLEX_SVE512_BLOCK2)) then
#endif  /* not WITH_FIXED_COMPLEX_KERNEL */

        ttt = mpi_wtime()
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sve512_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_&
          &MATH_DATATYPE&
          &_sve512_2hv_&
          &PRECISION&
          & (c_loc(a(1,j+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#ifdef WITH_OPENMP_TRADITIONAL
        if (j==1) call single_hh_trafo_&
        &MATH_DATATYPE&
        &_sve512_1hv_&
        &PRECISION&
        & (c_loc(a(1,1+off+a_off,istripe,my_thread)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
#else
        if (j==1) call single_hh_trafo_&
        &MATH_DATATYPE&
        &_sve512_1hv_&
        &PRECISION&
        & (c_loc(a(1,1+off+a_off,istripe)), bcast_buffer(1,off+1), nbw, nl, stripe_width)
#endif

#ifndef WITH_FIXED_COMPLEX_KERNEL
      endif ! ( (kernel .eq. ELPA_2STAGE_COMPLEX_SVE512_BLOCK2))
#endif  /* not WITH_FIXED_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_SVE512_BLOCK2_KERNEL */

#endif /* COMPLEXCASE */


#if REALCASE == 1

#if defined(WITH_REAL_BGP_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
      if (kernel .eq. ELPA_2STAGE_REAL_BGP) then

#endif /* not WITH_FIXED_REAL_KERNEL */
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
          call double_hh_trafo_bgp_&
          &PRECISION&
          & (a(1,j+off+a_off-1,istripe,my_thread), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_bgp_&
          &PRECISION&
          & (a(1,j+off+a_off-1,istripe), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#ifndef WITH_FIXED_REAL_KERNEL
      endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_BGP_KERNEL */

#if defined(WITH_REAL_BGQ_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
      if (kernel .eq. ELPA_2STAGE_REAL_BGQ) then

#endif /* not WITH_FIXED_REAL_KERNEL */
        do j = ncols, 2, -2
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
          call double_hh_trafo_bgq_&
          &PRECISION&
          & (a(1,j+off+a_off-1,istripe,my_thread), w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_bgq_&PRECISION&
          & (a(1,j+off+a_off-1,istripe), w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#ifndef WITH_FIXED_REAL_KERNEL
      endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_BGQ_KERNEL */

#endif /* REALCASE */

#if COMPLEXCASE == 1
      ! complex bgp/bgq kernel implemented
#endif


#if REALCASE == 1
#ifdef WITH_OPENMP_TRADITIONAL
      if (j==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width, 1+off+a_off:1+off+a_off+nbw-1,istripe,my_thread), &
               bcast_buffer(1:nbw,off+1), nbw, nl,stripe_width)
#else
      if (j==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl,&
               stripe_width)
#endif

#endif /* REALCASE == 1 */

#if REALCASE == 1
#ifndef WITH_FIXED_REAL_KERNEL
    endif !
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* REALCASE == 1 */

#if REALCASE == 1
    ! generic simple block4 real kernel

#if defined(WITH_REAL_GENERIC_SIMPLE_BLOCK4_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if (kernel .eq. ELPA_2STAGE_REAL_GENERIC_SIMPLE_BLOCK4) then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_GENERIC_SIMPLE_BLOCK6_KERNEL))
      do j = ncols, 4, -4
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
#ifdef WITH_OPENMP_TRADITIONAL

#ifdef USE_ASSUMED_SIZE
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_4hv_&
        &PRECISION&
        & (a(1,j+off+a_off-3,istripe,my_thread), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_4hv_&
        &PRECISION&
        & (a(1:stripe_width,j+off+a_off-3:j+off+a_off+nbw-1,istripe,my_thread), w(1:nbw,1:6), nbw, nl, &
           stripe_width, nbw)
#endif

#else

#ifdef USE_ASSUMED_SIZE
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_4hv_&
        &PRECISION&
        & (a(1,j+off+a_off-3,istripe), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_4hv_&
        &PRECISION&
        & (a(1:stripe_width,j+off+a_off-3:j+off+a_off+nbw-1,istripe), w(1:nbw,1:6), nbw, nl, &
           stripe_width, nbw)
#endif

#endif
      enddo

      do jj = j, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL

#ifdef USE_ASSUMED_SIZE
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_&
        &PRECISION&
        & (a(1,jj+off+a_off-1,istripe,my_thread), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_&
        &PRECISION&
        & (a(1:stripe_width,jj+off+a_off-1:jj+off+a_off-1+nbw,istripe,my_thread), w(1:nbw,1:6), nbw, &
           nl, stripe_width, nbw)
#endif

#else

#ifdef USE_ASSUMED_SIZE
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_&
        &PRECISION&
        & (a(1,jj+off+a_off-1,istripe), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_&
        &PRECISION&
        & (a(1:stripe_width,jj+off+a_off-1:jj+off+a_off-1+nbw,istripe), w(1:nbw,1:6), &
           nbw, nl, stripe_width, nbw)
#endif

#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL

      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
               bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)

#else

      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), &
         nbw, nl, stripe_width)
#endif
#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_GENERIC_SIMPLE_BLOCK6_KERNEL)) */

#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_GENERIC_SIMPLE_BLOCK4_KERNEL */

#endif /* REALCASE */

#if REALCASE == 1
    !real generic simple block6 kernel
#if defined(WITH_REAL_GENERIC_SIMPLE_BLOCK6_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if (kernel .eq. ELPA_2STAGE_REAL_GENERIC_SIMPLE_BLOCK6) then

#endif /* not WITH_FIXED_REAL_KERNEL */
      ! X86 INTRINSIC CODE, USING 6 HOUSEHOLDER VECTORS
      do j = ncols, 6, -6
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
        w(:,5) = bcast_buffer(1:nbw,j+off-4)
        w(:,6) = bcast_buffer(1:nbw,j+off-5)

#ifdef WITH_OPENMP_TRADITIONAL

!#ifdef USE_ASSUMED_SIZE
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_6hv_&
        &PRECISION&
        & (a(1,j+off+a_off-5,istripe,my_thread), w, nbw, nl, stripe_width, nbw)
!#else
!        call hexa_hh_trafo_&
!        &MATH_DATATYPE&
!        &_generic_simple_6hv_&
!        &PRECISION&
!        & (a(1:stripe_width,j+off+a_off-5:j+off+a_off-1,istripe,my_thread), w(1:nbw,1:6), &
!           nbw, nl, stripe_width, nbw)
!#endif

#else /* WITH_OPENMP_TRADITIONAL */
!#ifdef USE_ASSUMED_SIZE
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_6hv_&
        &PRECISION&
        & (a(1,j+off+a_off-5,istripe), w, nbw, nl, stripe_width, nbw)
!#else
!        call hexa_hh_trafo_&
!        &MATH_DATATYPE&
!        &_generic_simple_6hv_&
!        &PRECISION&
!        & (a(1:stripe_width,j+off+a_off-5:j+off+a_off+nbw-1,istripe), w(1:nbw,1:6), &
!           nbw, nl, stripe_width, nbw)
!#endif
#endif /* WITH_OPENMP_TRADITIONAL */
      enddo
      do jj = j, 4, -4
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
        w(:,3) = bcast_buffer(1:nbw,jj+off-2)
        w(:,4) = bcast_buffer(1:nbw,jj+off-3)
#ifdef WITH_OPENMP_TRADITIONAL

#ifdef USE_ASSUMED_SIZE
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_4hv_&
        &PRECISION&
        & (a(1,jj+off+a_off-3,istripe,my_thread), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_4hv_&
        &PRECISION&
        & (a(1:stripe_width,jj+off+a_off-3:jj+off+a_off+nbw-1,istripe,my_thread), &
           w(1:nbw,1:6), nbw, nl, stripe_width, nbw)
#endif

#else /* WITH_OPENMP_TRADITIONAL */

#ifdef USE_ASSUMED_SIZE
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_4hv_&
        &PRECISION&
        & (a(1,jj+off+a_off-3,istripe), w, &
                                      nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_4hv_&
        &PRECISION&
        & (a(1:stripe_width,jj+off+a_off-3:jj+off+a_off+nbw-1,istripe), &
           w(1:nbw,1:6), nbw, nl, stripe_width, nbw)
#endif

#endif /* WITH_OPENMP_TRADITIONAL */
      enddo
      do jjj = jj, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jjj+off)
        w(:,2) = bcast_buffer(1:nbw,jjj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL

#ifdef USE_ASSUMED_SIZE
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_&
        &PRECISION&
        & (a(1,jjj+off+a_off-1,istripe,my_thread), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_&
        &PRECISION&
        & (a(1:stripe_width,jj+off+a_off-1:jj+off+a_off-1+nbw,istripe,my_thread), w(1:nbw,1:6), nbw, &
           nl, stripe_width, nbw)
#endif

#else /* WITH_OPENMP_TRADITIONAL */

#ifdef USE_ASSUMED_SIZE
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_&
        &PRECISION&
        & (a(1,jjj+off+a_off-1,istripe), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_generic_simple_&
        &PRECISION&
        & (a(1:stripe_width,jj+off+a_off-1:jj+off+a_off-1+nbw,istripe), w(1:nbw,1:6), nbw, nl, &
           stripe_width, nbw)
#endif

#endif /* WITH_OPENMP_TRADITIONAL */
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif
#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_GENERIC_SIMPLE_BLOCK6_KERNEL */

#endif /* REALCASE */


#if REALCASE == 1
    ! sparc64 block 4 real kernel

#if defined(WITH_REAL_SPARC64_BLOCK4_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if (kernel .eq. ELPA_2STAGE_REAL_SPARC64_BLOCK4) then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_SPARC64_BLOCK6_KERNEL))
      do j = ncols, 4, -4
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sparc64_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sparc64_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sparc64_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sparc64_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
               bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif

#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_SPARC64_BLOCK6_KERNEL)) */

#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_SPARC64_BLOCK4_KERNEL */

#endif /* REALCASE */


#if REALCASE == 1
    ! neon_arch64 block 4 real kernel

#if defined(WITH_REAL_NEON_ARCH64_BLOCK4_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if (kernel .eq. ELPA_2STAGE_REAL_NEON_ARCH64_BLOCK4) then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_NEON_ARCH64_BLOCK6_KERNEL))
      do j = ncols, 4, -4
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_neon_arch64_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_neon_arch64_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_neon_arch64_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_neon_arch64_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
               bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif

#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_NEON_ARCH64_BLOCK6_KERNEL)) */

#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_NEON_ARCH64_BLOCK4_KERNEL */

#endif /* REALCASE */

#if REALCASE == 1
    ! sve128 block 4 real kernel

#if defined(WITH_REAL_SVE128_BLOCK4_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if (kernel .eq. ELPA_2STAGE_REAL_SVE128_BLOCK4) then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_SVE128_BLOCK6_KERNEL))
      do j = ncols, 4, -4
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sve128_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sve128_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sve128_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sve128_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
               bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif

#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_SVE128_BLOCK6_KERNEL)) */

#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_SVE128_BLOCK4_KERNEL */

#endif /* REALCASE */

#if REALCASE == 1
    ! vsx block4 real kernel

#if defined(WITH_REAL_VSX_BLOCK4_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if (kernel .eq. ELPA_2STAGE_REAL_VSX_BLOCK4) then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_VSX_BLOCK6_KERNEL))
      ! X86 INTRINSIC CODE, USING 4 HOUSEHOLDER VECTORS
      do j = ncols, 4, -4
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_vsx_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_vsx_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_vsx_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_vsx_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
               bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif

#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_VSX_BLOCK6_KERNEL)) */

#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_VSX_BLOCK4_KERNEL */

#endif /* REALCASE */

#if REALCASE == 1
    ! sse block4 real kernel

#if defined(WITH_REAL_SSE_BLOCK4_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if (kernel .eq. ELPA_2STAGE_REAL_SSE_BLOCK4) then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_SSE_BLOCK6_KERNEL))
      ! X86 INTRINSIC CODE, USING 4 HOUSEHOLDER VECTORS
      do j = ncols, 4, -4
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sse_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sse_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sse_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sse_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
               bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif

#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_SSE_BLOCK6_KERNEL)) */

#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_SSE_BLOCK4_KERNEL */

#endif /* REALCASE */

#if COMPLEXCASE == 1
    !no sse block4 complex kernel
#endif /* COMPLEXCASE */

#if REALCASE == 1
    ! avx block4 real kernel
#if defined(WITH_REAL_AVX_BLOCK4_KERNEL) 
#ifndef WITH_FIXED_REAL_KERNEL
    if ((kernel .eq. ELPA_2STAGE_REAL_AVX_BLOCK4)) then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_AVX_BLOCK6_KERNEL) )
      ! X86 INTRINSIC CODE, USING 4 HOUSEHOLDER VECTORS
      do j = ncols, 4, -4
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_avx_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_avx_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_avx_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe,my_thread)),w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_avx_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif

#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_AVX_BLOCK6_KERNEL)) */

#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_AVX_BLOCK4_KERNEL */

    ! avx2 block4 real kernel
#if defined(WITH_REAL_AVX2_BLOCK4_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if ((kernel .eq. ELPA_2STAGE_REAL_AVX2_BLOCK4)) then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_AVX2_BLOCK6_KERNEL))
      ! X86 INTRINSIC CODE, USING 4 HOUSEHOLDER VECTORS
      do j = ncols, 4, -4
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_avx2_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_avx2_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_avx2_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe,my_thread)),w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_avx2_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif

#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_AVX2_BLOCK6_KERNEL)) */

#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_AVX2_BLOCK4_KERNEL */

    ! sve256 block4 real kernel
#if defined(WITH_REAL_SVE256_BLOCK4_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if ((kernel .eq. ELPA_2STAGE_REAL_SVE256_BLOCK4)) then

#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_SVE256_BLOCK6_KERNEL))
      ! X86 INTRINSIC CODE, USING 4 HOUSEHOLDER VECTORS
      do j = ncols, 4, -4
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sve256_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sve256_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sve256_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe,my_thread)),w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sve256_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif

#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_SVE256_BLOCK6_KERNEL)) */

#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_SVE256_BLOCK4_KERNEL */

#endif /* REALCASE */

#if COMPLEXCASE == 1
    !no avx block4 complex kernel
#endif

#if REALCASE == 1
    ! avx512 block4 real kernel

#if defined(WITH_REAL_AVX512_BLOCK4_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if (kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK4) then
#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_AVX512_BLOCK6_KERNEL))
      ! X86 INTRINSIC CODE, USING 4 HOUSEHOLDER VECTORS
      do j = ncols, 4, -4
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_avx512_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_avx512_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_avx512_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_avx512_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif

#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_AVX_BLOCK6_KERNEL) ) */

#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_AVX512_BLOCK4_KERNEL */


! sve512 block4 real kernel

#if defined(WITH_REAL_SVE512_BLOCK4_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if (kernel .eq. ELPA_2STAGE_REAL_SVE512_BLOCK4) then
#endif /* not WITH_FIXED_REAL_KERNEL */

#if (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_SVE512_BLOCK6_KERNEL))
      ! X86 INTRINSIC CODE, USING 4 HOUSEHOLDER VECTORS
      do j = ncols, 4, -4
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sve512_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sve512_4hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-3,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sve512_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sve512_2hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif

#endif /* (!defined(WITH_FIXED_REAL_KERNEL)) || (defined(WITH_FIXED_REAL_KERNEL) && !defined(WITH_REAL_AVX_BLOCK6_KERNEL) ) */

#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_SVE512_BLOCK4_KERNEL */

#endif /* REALCASE */

#if COMPLEXCASE == 1
    !no avx512 block4 complex kernel
    !no sve512 block4 complex kernel
#endif /* COMPLEXCASE */


#if REALCASE == 1
    !sparc64 block6 real kernel
#if defined(WITH_REAL_SPARC64_BLOCK6_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if (kernel .eq. ELPA_2STAGE_REAL_SPARC64_BLOCK6) then

#endif /* not WITH_FIXED_REAL_KERNEL */
      ! X86 INTRINSIC CODE, USING 6 HOUSEHOLDER VECTORS
      do j = ncols, 6, -6
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
        w(:,5) = bcast_buffer(1:nbw,j+off-4)
        w(:,6) = bcast_buffer(1:nbw,j+off-5)
#ifdef WITH_OPENMP_TRADITIONAL
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_sparc64_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_sparc64_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 4, -4
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
        w(:,3) = bcast_buffer(1:nbw,jj+off-2)
        w(:,4) = bcast_buffer(1:nbw,jj+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sparc64_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sparc64_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe)), w, &
                                                  nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jjj = jj, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jjj+off)
        w(:,2) = bcast_buffer(1:nbw,jjj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sparc64_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sparc64_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
                                bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif
#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_SPARC64_BLOCK6_KERNEL */

#endif /* REALCASE */

#if REALCASE == 1
    !neon_arch64 block6 real kernel
#if defined(WITH_REAL_NEON_ARCH64_BLOCK6_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if (kernel .eq. ELPA_2STAGE_REAL_NEON_ARCH64_BLOCK6) then

#endif /* not WITH_FIXED_REAL_KERNEL */
      ! X86 INTRINSIC CODE, USING 6 HOUSEHOLDER VECTORS
      do j = ncols, 6, -6
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
        w(:,5) = bcast_buffer(1:nbw,j+off-4)
        w(:,6) = bcast_buffer(1:nbw,j+off-5)
#ifdef WITH_OPENMP_TRADITIONAL
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_neon_arch64_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_neon_arch64_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 4, -4
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
        w(:,3) = bcast_buffer(1:nbw,jj+off-2)
        w(:,4) = bcast_buffer(1:nbw,jj+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_neon_arch64_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_neon_arch64_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe)), w, &
                                      nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jjj = jj, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jjj+off)
        w(:,2) = bcast_buffer(1:nbw,jjj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_neon_arch64_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_neon_arch64_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif
#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_NEON_ARCH64_BLOCK6_KERNEL */

#endif /* REALCASE */


#if REALCASE == 1
    !sve128 block6 real kernel
#if defined(WITH_REAL_SVE128_BLOCK6_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if (kernel .eq. ELPA_2STAGE_REAL_SVE128_BLOCK6) then

#endif /* not WITH_FIXED_REAL_KERNEL */
      ! X86 INTRINSIC CODE, USING 6 HOUSEHOLDER VECTORS
      do j = ncols, 6, -6
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
        w(:,5) = bcast_buffer(1:nbw,j+off-4)
        w(:,6) = bcast_buffer(1:nbw,j+off-5)
#ifdef WITH_OPENMP_TRADITIONAL
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_SVE128_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_SVE128_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 4, -4
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
        w(:,3) = bcast_buffer(1:nbw,jj+off-2)
        w(:,4) = bcast_buffer(1:nbw,jj+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sve128_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sve128_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe)), w, &
                                      nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jjj = jj, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jjj+off)
        w(:,2) = bcast_buffer(1:nbw,jjj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sve128_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sve128_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif
#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_SVE128_BLOCK6_KERNEL */

#endif /* REALCASE */

#if REALCASE == 1
    !vsx block6 real kernel
#if defined(WITH_REAL_VSX_BLOCK6_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if (kernel .eq. ELPA_2STAGE_REAL_VSX_BLOCK6) then

#endif /* not WITH_FIXED_REAL_KERNEL */
      ! X86 INTRINSIC CODE, USING 6 HOUSEHOLDER VECTORS
      do j = ncols, 6, -6
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
        w(:,5) = bcast_buffer(1:nbw,j+off-4)
        w(:,6) = bcast_buffer(1:nbw,j+off-5)
#ifdef WITH_OPENMP_TRADITIONAL
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_vsx_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_vsx_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 4, -4
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
        w(:,3) = bcast_buffer(1:nbw,jj+off-2)
        w(:,4) = bcast_buffer(1:nbw,jj+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_vsx_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_vsx_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe)), w, &
                                      nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jjj = jj, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jjj+off)
        w(:,2) = bcast_buffer(1:nbw,jjj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_vsx_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_vsx_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif
#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_VSX_BLOCK6_KERNEL */

#endif /* REALCASE */

#if REALCASE == 1
    !sse block6 real kernel
#if defined(WITH_REAL_SSE_BLOCK6_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if (kernel .eq. ELPA_2STAGE_REAL_SSE_BLOCK6) then

#endif /* not WITH_FIXED_REAL_KERNEL */
      ! X86 INTRINSIC CODE, USING 6 HOUSEHOLDER VECTORS
      do j = ncols, 6, -6
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
        w(:,5) = bcast_buffer(1:nbw,j+off-4)
        w(:,6) = bcast_buffer(1:nbw,j+off-5)
#ifdef WITH_OPENMP_TRADITIONAL
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_sse_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_sse_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 4, -4
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
        w(:,3) = bcast_buffer(1:nbw,jj+off-2)
        w(:,4) = bcast_buffer(1:nbw,jj+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sse_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sse_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe)), w, &
                                      nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jjj = jj, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jjj+off)
        w(:,2) = bcast_buffer(1:nbw,jjj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sse_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sse_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif
#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_SSE_BLOCK6_KERNEL */

#endif /* REALCASE */

#if COMPLEXCASE == 1
    ! no sse block6 complex kernel
#endif

#if REALCASE == 1
    ! avx block6 real kernel

#if defined(WITH_REAL_AVX_BLOCK6_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if ((kernel .eq. ELPA_2STAGE_REAL_AVX_BLOCK6)) then

#endif /* not WITH_FIXED_REAL_KERNEL */
      ! X86 INTRINSIC CODE, USING 6 HOUSEHOLDER VECTORS
      do j = ncols, 6, -6
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
        w(:,5) = bcast_buffer(1:nbw,j+off-4)
        w(:,6) = bcast_buffer(1:nbw,j+off-5)
#ifdef WITH_OPENMP_TRADITIONAL
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_avx_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_avx_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 4, -4
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
        w(:,3) = bcast_buffer(1:nbw,jj+off-2)
        w(:,4) = bcast_buffer(1:nbw,jj+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_avx_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_avx_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jjj = jj, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jjj+off)
        w(:,2) = bcast_buffer(1:nbw,jjj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_avx_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_avx_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif
#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_AVX_BLOCK6_KERNEL */

    ! avx2 block6 real kernel

#if defined(WITH_REAL_AVX2_BLOCK6_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if ((kernel .eq. ELPA_2STAGE_REAL_AVX2_BLOCK6)) then

#endif /* not WITH_FIXED_REAL_KERNEL */
      ! X86 INTRINSIC CODE, USING 6 HOUSEHOLDER VECTORS
      do j = ncols, 6, -6
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
        w(:,5) = bcast_buffer(1:nbw,j+off-4)
        w(:,6) = bcast_buffer(1:nbw,j+off-5)
#ifdef WITH_OPENMP_TRADITIONAL
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_avx2_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_avx2_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 4, -4
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
        w(:,3) = bcast_buffer(1:nbw,jj+off-2)
        w(:,4) = bcast_buffer(1:nbw,jj+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_avx2_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_avx2_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jjj = jj, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jjj+off)
        w(:,2) = bcast_buffer(1:nbw,jjj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_avx2_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_avx2_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif
#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_AVX2_BLOCK6_KERNEL */

    ! sve256 block6 real kernel

#if defined(WITH_REAL_SVE256_BLOCK6_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if ((kernel .eq. ELPA_2STAGE_REAL_SVE256_BLOCK6)) then

#endif /* not WITH_FIXED_REAL_KERNEL */
      ! X86 INTRINSIC CODE, USING 6 HOUSEHOLDER VECTORS
      do j = ncols, 6, -6
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
        w(:,5) = bcast_buffer(1:nbw,j+off-4)
        w(:,6) = bcast_buffer(1:nbw,j+off-5)
#ifdef WITH_OPENMP_TRADITIONAL
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_sve256_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_sve256_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 4, -4
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
        w(:,3) = bcast_buffer(1:nbw,jj+off-2)
        w(:,4) = bcast_buffer(1:nbw,jj+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sve256_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sve256_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jjj = jj, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jjj+off)
        w(:,2) = bcast_buffer(1:nbw,jjj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sve256_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sve256_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), &
                              bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif
#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_SVE256_BLOCK6_KERNEL */

#endif /* REALCASE */

#if COMPLEXCASE == 1
    !no avx block6 complex kernel
#endif

#if REALCASE == 1
    ! avx512 block6 kernel
#if defined(WITH_REAL_AVX512_BLOCK6_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if ((kernel .eq. ELPA_2STAGE_REAL_AVX512_BLOCK6)) then
#endif /* not WITH_FIXED_REAL_KERNEL */
      ! X86 INTRINSIC CODE, USING 6 HOUSEHOLDER VECTORS

      do j = ncols, 6, -6
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
        w(:,5) = bcast_buffer(1:nbw,j+off-4)
        w(:,6) = bcast_buffer(1:nbw,j+off-5)
#ifdef WITH_OPENMP_TRADITIONAL
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_avx512_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_avx512_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 4, -4
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
        w(:,3) = bcast_buffer(1:nbw,jj+off-2)
        w(:,4) = bcast_buffer(1:nbw,jj+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_avx512_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_avx512_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jjj = jj, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jjj+off)
        w(:,2) = bcast_buffer(1:nbw,jjj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_avx512_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_avx512_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
                             bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif
#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_AVX512_BLOCK6_KERNEL */


! sve512 block6 kernel
#if defined(WITH_REAL_SVE512_BLOCK6_KERNEL)
#ifndef WITH_FIXED_REAL_KERNEL
    if ((kernel .eq. ELPA_2STAGE_REAL_SVE512_BLOCK6)) then
#endif /* not WITH_FIXED_REAL_KERNEL */
      ! X86 INTRINSIC CODE, USING 6 HOUSEHOLDER VECTORS

      do j = ncols, 6, -6
        w(:,1) = bcast_buffer(1:nbw,j+off)
        w(:,2) = bcast_buffer(1:nbw,j+off-1)
        w(:,3) = bcast_buffer(1:nbw,j+off-2)
        w(:,4) = bcast_buffer(1:nbw,j+off-3)
        w(:,5) = bcast_buffer(1:nbw,j+off-4)
        w(:,6) = bcast_buffer(1:nbw,j+off-5)
#ifdef WITH_OPENMP_TRADITIONAL
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_sve512_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call hexa_hh_trafo_&
        &MATH_DATATYPE&
        &_sve512_6hv_&
        &PRECISION&
        & (c_loc(a(1,j+off+a_off-5,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jj = j, 4, -4
        w(:,1) = bcast_buffer(1:nbw,jj+off)
        w(:,2) = bcast_buffer(1:nbw,jj+off-1)
        w(:,3) = bcast_buffer(1:nbw,jj+off-2)
        w(:,4) = bcast_buffer(1:nbw,jj+off-3)
#ifdef WITH_OPENMP_TRADITIONAL
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sve512_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call quad_hh_trafo_&
        &MATH_DATATYPE&
        &_sve512_4hv_&
        &PRECISION&
        & (c_loc(a(1,jj+off+a_off-3,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
      do jjj = jj, 2, -2
        w(:,1) = bcast_buffer(1:nbw,jjj+off)
        w(:,2) = bcast_buffer(1:nbw,jjj+off-1)
#ifdef WITH_OPENMP_TRADITIONAL
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sve512_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe,my_thread)), w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_&
        &MATH_DATATYPE&
        &_sve512_2hv_&
        &PRECISION&
        & (c_loc(a(1,jjj+off+a_off-1,istripe)), w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP_TRADITIONAL
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_openmp_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1, istripe,my_thread), &
                             bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#else
      if (jjj==1) call single_hh_trafo_&
      &MATH_DATATYPE&
      &_cpu_&
      &PRECISION&
      & (a(1:stripe_width,1+off+a_off:1+off+a_off+nbw-1,istripe), bcast_buffer(1:nbw,off+1), nbw, nl, stripe_width)
#endif
#ifndef WITH_FIXED_REAL_KERNEL
    endif
#endif /* not WITH_FIXED_REAL_KERNEL */
#endif /* WITH_REAL_SVE512_BLOCK6_KERNEL */

#endif /* REALCASE */

#if COMPLEXCASE == 1
    !no avx512 block6 complex kernel
    !no sve512 block6 complex kernel
#endif /* COMPLEXCASE */

    if (wantDebug) then
      call obj%timer%stop("compute_hh_trafo: CPU")
    endif
  endif ! GPU_KERNEL

#ifdef WITH_OPENMP_TRADITIONAL
  if (my_thread==1) then
#endif
    kernel_flops = kernel_flops + 4*int(nl,8)*int(ncols,8)*int(nbw,8)
    kernel_time = kernel_time + mpi_wtime()-ttt
    n_times = n_times + 1
#ifdef WITH_OPENMP_TRADITIONAL
  endif
#endif

  if (wantDebug) call obj%timer%stop("compute_hh_trafo_&
  &MATH_DATATYPE&
#ifdef WITH_OPENMP_TRADITIONAL
  &_openmp" // &
#else
  &" // &
#endif
  &PRECISION_SUFFIX &
  )

end subroutine

! vim: syntax=fortran
