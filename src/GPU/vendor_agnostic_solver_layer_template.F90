#if 0
!    Copyright 2021, A. Marek, MPCDF
!
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
! This file is the generated version. Do NOT edit
#endif


  contains

    subroutine gpusolver_Dtrtri(uplo, diag, n, a, lda, info, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_intptr_t)        :: handle

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Dtrtri(uplo, diag, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
#ifndef WITH_AMD_HIPSOLVER_API
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_Dtrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_Dtrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
!      if (use_gpu_vendor == sycl_gpu) then
!        call mkl_sycl_Dtrtri(uplo, diag, n, a, lda, info, handle)
!      endif
#endif
    end subroutine

    subroutine gpusolver_Dpotrf(uplo, n, a_dev, lda, info_dev, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=c_intptr_t)        :: handle

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Dpotrf(uplo, n, a_dev, lda, info_dev, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_Dpotrf(uplo, n, a_dev, lda, info_dev, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_Dpotrf(uplo, n, a_dev, lda, info_dev, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
!      if (use_gpu_vendor == sycl_gpu) then
!        call mkl_sycl_Dpotrf(uplo, n, a_dev, lda, info_dev, handle)
!      endif
#endif
    end subroutine

    subroutine gpusolver_Strtri(uplo, diag, n, a, lda, info, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_intptr_t)        :: handle

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Strtri(uplo, diag, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
#ifndef WITH_AMD_HIPSOLVER_API
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_Strtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_Strtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
!      if (use_gpu_vendor == sycl_gpu) then
!        call mkl_sycl_Strtri(uplo, diag, n, a, lda, info, handle)
!      endif
#endif
    end subroutine

    subroutine gpusolver_Spotrf(uplo, n, a_dev, lda, info_dev, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=c_intptr_t)        :: handle

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Spotrf(uplo, n, a_dev, lda, info_dev, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_Spotrf(uplo, n, a_dev, lda, info_dev, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_Spotrf(uplo, n, a_dev, lda, info_dev, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
!      if (use_gpu_vendor == sycl_gpu) then
!        call mkl_sycl_Spotrf(uplo, n, a_dev, lda, info_dev, handle)
!      endif
#endif
    end subroutine

    subroutine gpusolver_Ztrtri(uplo, diag, n, a, lda, info, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_intptr_t)        :: handle

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Ztrtri(uplo, diag, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
#ifndef WITH_AMD_HIPSOLVER_API
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_Ztrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_Ztrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
!      if (use_gpu_vendor == sycl_gpu) then
!        call mkl_sycl_Ztrtri(uplo, diag, n, a, lda, info, handle)
!      endif
#endif
    end subroutine

    subroutine gpusolver_Zpotrf(uplo, n, a_dev, lda, info_dev, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=c_intptr_t)        :: handle

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Zpotrf(uplo, n, a_dev, lda, info_dev, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_Zpotrf(uplo, n, a_dev, lda, info_dev, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_Zpotrf(uplo, n, a_dev, lda, info_dev, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
!      if (use_gpu_vendor == sycl_gpu) then
!        call mkl_sycl_Zpotrf(uplo, n, a_dev, lda, info_dev, handle)
!      endif
#endif
    end subroutine

    subroutine gpusolver_Ctrtri(uplo, diag, n, a, lda, info, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=c_intptr_t)        :: handle

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Ctrtri(uplo, diag, n, a, lda, info, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
#ifndef WITH_AMD_HIPSOLVER_API
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_Ctrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_Ctrtri(uplo, diag, n, a, lda, info, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
!      if (use_gpu_vendor == sycl_gpu) then
!        call mkl_sycl_Ctrtri(uplo, diag, n, a, lda, info, handle)
!      endif
#endif
    end subroutine

    subroutine gpusolver_Cpotrf(uplo, n, a_dev, lda, info_dev, handle)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=c_intptr_t)        :: handle

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Cpotrf(uplo, n, a_dev, lda, info_dev, handle)
      endif
#ifdef WITH_AMD_GPU_VERSION
      if (use_gpu_vendor == amd_gpu) then
        call rocsolver_Cpotrf(uplo, n, a_dev, lda, info_dev, handle)
      endif
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      if (use_gpu_vendor == openmp_offload_gpu) then
        call mkl_openmp_offload_Cpotrf(uplo, n, a_dev, lda, info_dev, handle)
      endif
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
!      if (use_gpu_vendor == sycl_gpu) then
!        call mkl_sycl_Cpotrf(uplo, n, a_dev, lda, info_dev, handle)
!      endif
#endif
    end subroutine

    ! Xpotrf_bufferSize

    subroutine gpusolver_Xpotrf_bufferSize(cusolverHandle, uplo, n, dataType, a_dev, lda, &
                                           workspaceInBytesOnDevice, workspaceInBytesOnHost)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)       :: cusolverHandle
      character(1, c_char), value    :: uplo, dataType
      integer(kind=c_int)            :: n, lda
      integer(kind=c_intptr_t)       :: a_dev
      integer(kind=c_size_t)         :: workspaceInBytesOnDevice, workspaceInBytesOnHost

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Xpotrf_bufferSize(cusolverHandle, uplo, n, dataType, a_dev, lda, &
                                        workspaceInBytesOnDevice, workspaceInBytesOnHost)
      endif
#ifdef WITH_AMD_GPU_VERSION
! not yet available in roc
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
! not yet available in openmp offload
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
#endif
    end subroutine

    ! Xpotrf

    subroutine gpusolver_Xpotrf(cusolverHandle, uplo, n, dataType, a_dev, lda, &
                                buffer_dev , workspaceInBytesOnDevice, &
                                buffer_host, workspaceInBytesOnHost, info_dev)
      use, intrinsic :: iso_c_binding
      use cuda_functions
#ifdef WITH_AMD_GPU_VERSION
      use hip_functions
#endif
#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
      use openmp_offload_functions
#endif
#ifdef WITH_SYCL_GPU_VERSION
      use sycl_functions
#endif

      implicit none
      integer(kind=c_intptr_t)       :: cusolverHandle
      character(1, c_char), value    :: uplo, dataType
      integer(kind=c_int)            :: n, lda
      integer(kind=c_intptr_t)       :: a_dev, buffer_dev, info_dev
      integer(kind=c_intptr_t)       :: buffer_host
      integer(kind=c_size_t)         :: workspaceInBytesOnDevice, workspaceInBytesOnHost

      if (use_gpu_vendor == nvidia_gpu) then
        call cusolver_Xpotrf(cusolverHandle, uplo, n, dataType, a_dev, lda, &
                             buffer_dev , workspaceInBytesOnDevice, &
                             buffer_host, workspaceInBytesOnHost, info_dev)
      endif
#ifdef WITH_AMD_GPU_VERSION
! not yet available in roc
#endif

#ifdef WITH_OPENMP_OFFLOAD_GPU_VERSION
! not yet available in openmp offload
#endif

#ifdef WITH_SYCL_GPU_VERSION
! not yet available in mkl
#endif
    end subroutine

