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
  subroutine solve_tridi_single_problem_gpu_&
  &PRECISION_AND_SUFFIX &
  (obj, nlen, d_dev, e_dev, q_dev, ldq, qtmp_dev, wantDebug, success)
#else
  subroutine solve_tridi_single_problem_cpu_&
  &PRECISION_AND_SUFFIX &
  (obj, nlen, d, e, q, ldq, wantDebug, success)
#endif
    ! solve_tridi_single_problem is called from solve_trodi_col: with parameters q_dev=qmat1_dev, ldq=max_size
    ! qmat1(max_size, max_size) -> q(ldq, ldq), and not q(ldq,nlen)!! same for q_dev
    ! but that's fine if nlen<=ldq, then not the whole matrix is used. otherwise can lead to errors
    ! Now: called from two places differently: with np_rows==1 and np_rows>1 and should be treated with care  

    ! Solves the symmetric, tridiagonal eigenvalue problem on a single processor.
    ! Takes precautions if DSTEDC fails or if the eigenvalues are not ordered correctly.
    use precision
    use elpa_abstract_impl
    use elpa_blas_interfaces
    use ELPA_utilities
    use elpa_gpu
    use solve_single_problem_gpu
#if defined(WITH_NVIDIA_GPU_VERSION) && defined(WITH_NVTX)
    use cuda_functions ! for NVTX labels
#elif defined(WITH_AMD_GPU_VERSION) && defined(WITH_ROCTX)
    use hip_functions  ! for ROCTX labels
#endif
    implicit none
    class(elpa_abstract_impl_t), intent(inout) :: obj
    logical                                    :: useGPU, useGPUsolver
    integer(kind=ik)                           :: nlen, ldq
    real(kind=REAL_DATATYPE)                   :: d(nlen), e(nlen), q(ldq,nlen)

    real(kind=REAL_DATATYPE), allocatable      :: work(:), qtmp(:), ds(:), es(:)
    real(kind=REAL_DATATYPE)                   :: dtmp

    integer(kind=ik)                           :: i, j, lwork, liwork, info
    integer(kind=BLAS_KIND)                    :: infoBLAS
    integer(kind=ik), allocatable              :: iwork(:)
    !real(kind=REAL_DATATYPE), allocatable      :: mat(:,:)

    logical, intent(in)                        :: wantDebug
    logical, intent(out)                       :: success
    logical                                    :: has_nans
    integer(kind=c_int)                        :: debug
    integer(kind=ik)                           :: istat
    character(200)                             :: errorMessage
    integer(kind=c_intptr_t)                   :: num, my_stream
    integer(kind=c_intptr_t)                   :: q_dev, d_dev, info_dev, qtmp_dev, e_dev
    logical                                    :: successGPU
    integer(kind=c_intptr_t), parameter        :: size_of_datatype = size_of_&
                                                                    &PRECISION&
                                                                    &_real

    integer(kind=c_intptr_t)                    :: gpusolverHandle

    debug = 0
    if (wantDebug) debug = 1
    
    useGPU =.false.
    useGPUsolver =.false.
#ifdef SOLVE_TRIDI_GPU_BUILD
    useGPU =.true.

#if defined(WITH_NVIDIA_CUSOLVER)
    useGPUsolver =.true.
#endif
#if defined(WITH_AMD_ROCSOLVER)
    ! As of ELPA 2025.01 release, rocsolver_?stedc/rocsolver_?syevd showed bad performance (worse than on CPU).
    ! Hopefully, this will be fixed by AMD and then we can enable it.
    useGPUsolver =.false.
#endif
#endif /* SOLVE_TRIDI_GPU_BUILD */

    call obj%timer%start("solve_tridi_single" // PRECISION_SUFFIX)

    success = .true.
    allocate(ds(nlen), es(nlen), stat=istat, errmsg=errorMessage)
    check_allocate("solve_tridi_single: ds, es", istat, errorMessage)

    if (useGPUsolver) then
      num = 1 * size_of_int
      successGPU = gpu_malloc(info_dev, num)
      check_alloc_gpu("solve_tridi_single info_dev: ", successGPU)
      
      my_stream = obj%gpu_setup%my_stream
      call gpu_construct_tridi_matrix(PRECISION_CHAR, q_dev, d_dev, e_dev, nlen, ldq, debug, my_stream)
    endif

     ! Save d and e for the case that dstedc fails
#if defined(WITH_NVIDIA_CUSOLVER) || defined(WITH_AMD_ROCSOLVER)
    if (.not.useGPU) then
      ds(:) = d(:)
      es(:) = e(:)
    endif
#endif

    if (useGPU) then
      if (.not. useGPUsolver) then
        ! use CPU solver

        ! copy to host
        num = nlen * size_of_datatype
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        successGPU = gpu_memcpy_async(int(loc(d(1)),kind=c_intptr_t), d_dev, &
                        num, gpuMemcpyDeviceToHost, my_stream)
        check_memcpy_gpu("solve_tridi_single: d_dev ", successGPU)
#else
        successGPU = gpu_memcpy(int(loc(d(1)),kind=c_intptr_t), d_dev, &
                            num, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("solve_tridi_single: d_dev", successGPU)
#endif
        num = nlen * size_of_datatype
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        successGPU = gpu_memcpy_async(int(loc(e(1)),kind=c_intptr_t), e_dev, &
                        num, gpuMemcpyDeviceToHost, my_stream)
        check_memcpy_gpu("solve_tridi_single: e_dev ", successGPU)
#else
        successGPU = gpu_memcpy(int(loc(e(1)),kind=c_intptr_t), e_dev, &
                            num, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("solve_tridi_single: e_dev", successGPU)
#endif

        ds(:) = d(:)
        es(:) = e(:)

#include "./solve_tridi_single_problem_include.F90"

        ! copy back
        num = nlen * size_of_datatype
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        successGPU = gpu_memcpy_async(d_dev, int(loc(d(1)),kind=c_intptr_t),  &
                        num, gpuMemcpyHostToDevice, my_stream)
        check_memcpy_gpu("solve_tridi_single: d_dev ", successGPU)
#else
        successGPU = gpu_memcpy(d_dev, int(loc(d(1)),kind=c_intptr_t),  &
                            num, gpuMemcpyHostToDevice)
        check_memcpy_gpu("solve_tridi_single: d_dev", successGPU)
#endif
        num = nlen * size_of_datatype
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        successGPU = gpu_memcpy_async(e_dev, int(loc(e(1)),kind=c_intptr_t), &
                        num, gpuMemcpyHostToDevice, my_stream)
        check_memcpy_gpu("solve_tridi_single: e_dev ", successGPU)
#else
        successGPU = gpu_memcpy(e_dev, int(loc(e(1)),kind=c_intptr_t),  &
                            num, gpuMemcpyHostToDevice)
        check_memcpy_gpu("solve_tridi_single: e_dev", successGPU)
#endif

! fails
        num = ldq*nlen * size_of_datatype
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        successGPU = gpu_memcpy_async(q_dev, int(loc(q(1,1)),kind=c_intptr_t), &
                        num, gpuMemcpyHostToDevice, my_stream)
        check_memcpy_gpu("solve_tridi_single: q_dev1 ", successGPU)
#else
        successGPU = gpu_memcpy(q_dev, int(loc(q(1,1)),kind=c_intptr_t),  &
                            num, gpuMemcpyHostToDevice)
        check_memcpy_gpu("solve_tridi_single: q_dev1", successGPU)
#endif

      else ! (.not. useGPUsolver)
        
        call obj%timer%start("gpusolver_syevd")
        NVTX_RANGE_PUSH("gpusolver_syevd")
        gpusolverHandle = obj%gpu_setup%gpusolverHandleArray(0)
        call gpusolver_PRECISION_syevd (nlen, q_dev, ldq, d_dev, info_dev, gpusolverHandle)
        if (wantDebug) successGPU = gpu_DeviceSynchronize()
        NVTX_RANGE_POP("gpusolver_syevd")
        call obj%timer%stop("gpusolver_syevd")

        num = 1 * size_of_int
#ifdef WITH_GPU_STREAMS
        my_stream = obj%gpu_setup%my_stream
        successGPU = gpu_memcpy_async(int(loc(info),kind=c_intptr_t), info_dev, &
                        num, gpuMemcpyDeviceToHost, my_stream)
        check_memcpy_gpu("solve_tridi_single: ", successGPU)

        successGPU = gpu_stream_synchronize(my_stream)
        check_stream_synchronize_gpu("solve_tridi_single: info_dev -> info", successGPU)
#else
        successGPU = gpu_memcpy(int(loc(info),kind=c_intptr_t), info_dev, &
                            num, gpuMemcpyDeviceToHost)
        check_memcpy_gpu("solve_tridi_single: info_dev", successGPU)
#endif

        if (info .ne. 0) then
          write(error_unit,'(a,i8,a)') "Error in gpusolver_PRECISION_syevd, info=", info, ", aborting..."
          stop 1
        endif
      endif ! (.not. useGPUsolver)
!!
!!       else
!         !copy to host
!         num = nlen * size_of_datatype
!#ifdef WITH_GPU_STREAMS
!         my_stream = obj%gpu_setup%my_stream
!         successGPU = gpu_memcpy_async(int(loc(d(1)),kind=c_intptr_t), d_dev, &
!                          num, gpuMemcpyDeviceToHost, my_stream)
!         check_memcpy_gpu("solve_tridi_single: d_dev ", successGPU)
!#else
!         successGPU = gpu_memcpy(int(loc(d(1)),kind=c_intptr_t), d_dev, &
!                             num, gpuMemcpyDeviceToHost)
!         check_memcpy_gpu("solve_tridi_single: d_dev", successGPU)
!#endif
!         num = nlen * size_of_datatype
!#ifdef WITH_GPU_STREAMS
!         my_stream = obj%gpu_setup%my_stream
!         successGPU = gpu_memcpy_async(int(loc(e(1)),kind=c_intptr_t), e_dev, &
!                          num, gpuMemcpyDeviceToHost, my_stream)
!         check_memcpy_gpu("solve_tridi_single: e_dev ", successGPU)
!#else
!         successGPU = gpu_memcpy(int(loc(e(1)),kind=c_intptr_t), e_dev, &
!                             num, gpuMemcpyDeviceToHost)
!         check_memcpy_gpu("solve_tridi_single: e_dev", successGPU)
!#endif
!
!         ds(:) = d(:)
!         es(:) = e(:)
!
!#endif /* defined(WITH_NVIDIA_CUSOLVER) || defined(WITH_AMD_ROCSOLVER) */
!
!#include "./solve_tridi_single_problem_include.F90"
!
!
!#if defined(WITH_NVIDIA_CUSOLVER) || defined(WITH_AMD_ROCSOLVER)
!         !copy back
!         num = nlen * size_of_datatype
!#ifdef WITH_GPU_STREAMS
!         my_stream = obj%gpu_setup%my_stream
!         successGPU = gpu_memcpy_async(d_dev, int(loc(d(1)),kind=c_intptr_t),  &
!                          num, gpuMemcpyHostToDevice, my_stream)
!         check_memcpy_gpu("solve_tridi_single: d_dev ", successGPU)
!#else
!         successGPU = gpu_memcpy(d_dev, int(loc(d(1)),kind=c_intptr_t),  &
!                             num, gpuMemcpyHostToDevice)
!         check_memcpy_gpu("solve_tridi_single: d_dev", successGPU)
!#endif
!         num = nlen * size_of_datatype
!#ifdef WITH_GPU_STREAMS
!         my_stream = obj%gpu_setup%my_stream
!         successGPU = gpu_memcpy_async(e_dev, int(loc(e(1)),kind=c_intptr_t), &
!                          num, gpuMemcpyHostToDevice, my_stream)
!         check_memcpy_gpu("solve_tridi_single: e_dev ", successGPU)
!#else
!         successGPU = gpu_memcpy(e_dev, int(loc(e(1)),kind=c_intptr_t),  &
!                             num, gpuMemcpyHostToDevice)
!         check_memcpy_gpu("solve_tridi_single: e_dev", successGPU)
!#endif
!
!! fails
!         num = ldq*nlen * size_of_datatype
!#ifdef WITH_GPU_STREAMS
!         my_stream = obj%gpu_setup%my_stream
!         successGPU = gpu_memcpy_async(q_dev, int(loc(q(1,1)),kind=c_intptr_t), &
!                          num, gpuMemcpyHostToDevice, my_stream)
!         check_memcpy_gpu("solve_tridi_single: q_dev1 ", successGPU)
!#else
!         successGPU = gpu_memcpy(q_dev, int(loc(q(1,1)),kind=c_intptr_t),  &
!                             num, gpuMemcpyHostToDevice)
!         check_memcpy_gpu("solve_tridi_single: q_dev1", successGPU)
!#endif
!!       endif ! nlen


    else ! useGPU
#include "./solve_tridi_single_problem_include.F90"
    endif ! useGPU

    if (useGPUsolver) then
      successGPU = gpu_free(info_dev)
      check_dealloc_gpu("solve_tridi_single: info_dev", successGPU)
    endif

    ! Check if eigenvalues are monotonically increasing
    ! This seems to be not always the case  (in the IBM implementation of dstedc ???)

    if (useGPU) then
      my_stream = obj%gpu_setup%my_stream
      call gpu_check_monotony(PRECISION_CHAR, d_dev, q_dev, qtmp_dev, nlen, ldq, debug, my_stream)
    else
      do i=1,nlen-1
        if (d(i+1)<d(i)) then
#ifdef DOUBLE_PRECISION_REAL
          if (abs(d(i+1) - d(i)) / abs(d(i+1) + d(i)) > 1e-14_rk8) then
#else
          if (abs(d(i+1) - d(i)) / abs(d(i+1) + d(i)) > 1e-14_rk4) then
#endif
            write(error_unit,'(a,i8,2g25.16)') '***WARNING: Monotony error dste**:',i+1,d(i),d(i+1)
          else
            write(error_unit,'(a,i8,2g25.16)') 'Info: Monotony error dste{dc,qr}:',i+1,d(i),d(i+1)
            write(error_unit,'(a)') 'The eigenvalues from a lapack call are not sorted to machine precision.'
            write(error_unit,'(a)') 'In this extent, this is completely harmless.'
            write(error_unit,'(a)') 'Still, we keep this info message just in case.'
          end if
          allocate(qtmp(nlen), stat=istat, errmsg=errorMessage)
          check_allocate("solve_tridi_single: qtmp", istat, errorMessage)

          dtmp = d(i+1)
          qtmp(1:nlen) = q(1:nlen,i+1)
          do j=i,1,-1
            if (dtmp<d(j)) then
              d(j+1)        = d(j)
              q(1:nlen,j+1) = q(1:nlen,j)
            else
              exit ! Loop
            endif
          enddo
          d(j+1)        = dtmp
          q(1:nlen,j+1) = qtmp(1:nlen)
          deallocate(qtmp, stat=istat, errmsg=errorMessage)
          check_deallocate("solve_tridi_single: qtmp", istat, errorMessage)
        endif
      enddo
    endif ! useGPU

    call obj%timer%stop("solve_tridi_single" // PRECISION_SUFFIX)

#ifdef SOLVE_TRIDI_GPU_BUILD
  end subroutine solve_tridi_single_problem_gpu_&
  &PRECISION_AND_SUFFIX
#else
  end subroutine solve_tridi_single_problem_cpu_&
  &PRECISION_AND_SUFFIX
#endif
