!    Copyright 2014 - 2023, A. Marek
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
! Author: Peter Karpov, MPCDF
! This file is the generated version. Do NOT edit


  interface
    function cusolver_set_stream_c(cusolverHandle, cudaStream) result(istat) &
             bind(C, name="cusolverSetStreamFromC")
      use, intrinsic :: iso_c_binding
      implicit none

      integer(kind=C_intptr_T), value  :: cusolverHandle
      integer(kind=C_intptr_T), value  :: cudaStream
      integer(kind=C_INT)              :: istat
    end function
  end interface

  interface
    function cusolver_create_c(cusolverHandle) result(istat) &
             bind(C, name="cusolverCreateFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: cusolverHandle
      integer(kind=C_INT)      :: istat
    end function
  end interface

  interface
    function cusolver_destroy_c(cusolverHandle) result(istat) &
             bind(C, name="cusolverDestroyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T), value :: cusolverHandle
      integer(kind=C_INT)      :: istat
    end function
  end interface

  ! cusolver_?trtri_c

  interface
    subroutine cusolver_Dtrtri_c(cusolverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="cusolverDtrtri_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: cusolverHandle
    end subroutine
  end interface

  interface
    subroutine cusolver_Strtri_c(cusolverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="cusolverStrtri_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: cusolverHandle
    end subroutine
  end interface

  interface
    subroutine cusolver_Ztrtri_c(cusolverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="cusolverZtrtri_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: cusolverHandle
    end subroutine
  end interface

  interface
    subroutine cusolver_Ctrtri_c(cusolverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="cusolverCtrtri_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: cusolverHandle
    end subroutine
  end interface

  ! cusolver_?potrf_c

  interface
    subroutine cusolver_Dpotrf_c(cusolverHandle, uplo, n, a_dev, lda, info_dev) &
                              bind(C,name="cusolverDpotrf_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a_dev, info_dev
      integer(kind=C_intptr_T), value           :: cusolverHandle
    end subroutine
  end interface

  interface
    subroutine cusolver_Spotrf_c(cusolverHandle, uplo, n, a_dev, lda, info_dev) &
                              bind(C,name="cusolverSpotrf_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a_dev, info_dev
      integer(kind=C_intptr_T), value           :: cusolverHandle
    end subroutine
  end interface

  interface
    subroutine cusolver_Zpotrf_c(cusolverHandle, uplo, n, a_dev, lda, info_dev) &
                              bind(C,name="cusolverZpotrf_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a_dev, info_dev
      integer(kind=C_intptr_T), value           :: cusolverHandle
    end subroutine
  end interface

  interface
    subroutine cusolver_Cpotrf_c(cusolverHandle, uplo, n, a_dev, lda, info_dev) &
                              bind(C,name="cusolverCpotrf_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a_dev, info_dev
      integer(kind=C_intptr_T), value           :: cusolverHandle
    end subroutine
  end interface

  ! cusolver_Xpotrf_buffereSize_c

  interface
    subroutine cusolver_Xpotrf_bufferSize_c(cusolverHandle, uplo, n, dataType, a_dev, lda, &
                                            workspaceInBytesOnDevice, workspaceInBytesOnHost) &
                                            bind(C,name="cusolverXpotrf_bufferSize_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value          :: cusolverHandle
      character(1, c_char), value              :: uplo, dataType
      integer(kind=c_int), intent(in), value   :: n, lda
      integer(kind=c_intptr_t), value          :: a_dev
      integer(kind=c_size_t)                   :: workspaceInBytesOnDevice, workspaceInBytesOnHost
    end subroutine
  end interface

  ! cusolver_Xpotrf_c

  interface
    subroutine cusolver_Xpotrf_c(cusolverHandle, uplo, n, dataType, a_dev, lda, &
                                 buffer_dev , workspaceInBytesOnDevice, &
                                 buffer_host, workspaceInBytesOnHost, info_dev) &
                                 bind(C,name="cusolverXpotrf_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t), value          :: cusolverHandle
      character(1, c_char), value              :: uplo, dataType
      integer(kind=c_int), intent(in), value   :: n, lda
      integer(kind=c_intptr_t), value          :: a_dev, buffer_dev, info_dev
      integer(kind=c_intptr_t), value          :: buffer_host
      integer(kind=c_size_t)                   :: workspaceInBytesOnDevice, workspaceInBytesOnHost
    end subroutine
  end interface


  contains


    function cusolver_set_stream(cusolverHandle, cudaStream) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: cusolverHandle
      integer(kind=C_intptr_t)                  :: cudaStream
      logical                                   :: success

#ifdef WITH_NVIDIA_CUSOLVER
      success = cusolver_set_stream_c(cusolverHandle, cudaStream) /= 0
#else
      success = .true.
#endif
    end function

    function cusolver_create(cusolverHandle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: cusolverHandle
      logical                                   :: success
#ifdef WITH_NVIDIA_CUSOLVER
      success = cusolver_create_c(cusolverHandle) /= 0
#else
      success = .true.
#endif
    end function

    function cusolver_destroy(cusolverHandle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: cusolverHandle
      logical                                   :: success
#ifdef WITH_NVIDIA_CUSOLVER
      success = cusolver_destroy_c(cusolverHandle) /= 0
#else
      success = .true.
#endif
    end function

    ! cusolver_?trtri

    subroutine cusolver_Dtrtri(uplo, diag, n, a, lda, info, cusolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: cusolverHandle
#ifdef WITH_NVIDIA_CUSOLVER
      call cusolver_Dtrtri_c(cusolverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    subroutine cusolver_Strtri(uplo, diag, n, a, lda, info, cusolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: cusolverHandle
#ifdef WITH_NVIDIA_CUSOLVER
      call cusolver_Strtri_c(cusolverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    subroutine cusolver_Ztrtri(uplo, diag, n, a, lda, info, cusolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: cusolverHandle
#ifdef WITH_NVIDIA_CUSOLVER
      call cusolver_Ztrtri_c(cusolverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    subroutine cusolver_Ctrtri(uplo, diag, n, a, lda, info, cusolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: cusolverHandle
#ifdef WITH_NVIDIA_CUSOLVER
      call cusolver_Ctrtri_c(cusolverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    ! cusolver_?potrf

    subroutine cusolver_Dpotrf(uplo, n, a_dev, lda, info_dev, cusolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=C_intptr_T)        :: cusolverHandle
#ifdef WITH_NVIDIA_CUSOLVER
      call cusolver_Dpotrf_c(cusolverHandle, uplo, n, a_dev, lda, info_dev)
#endif
    end subroutine

    subroutine cusolver_Spotrf(uplo, n, a_dev, lda, info_dev, cusolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=C_intptr_T)        :: cusolverHandle
#ifdef WITH_NVIDIA_CUSOLVER
      call cusolver_Spotrf_c(cusolverHandle, uplo, n, a_dev, lda, info_dev)
#endif
    end subroutine

    subroutine cusolver_Zpotrf(uplo, n, a_dev, lda, info_dev, cusolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=C_intptr_T)        :: cusolverHandle
#ifdef WITH_NVIDIA_CUSOLVER
      call cusolver_Zpotrf_c(cusolverHandle, uplo, n, a_dev, lda, info_dev)
#endif
    end subroutine

    subroutine cusolver_Cpotrf(uplo, n, a_dev, lda, info_dev, cusolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=C_intptr_T)        :: cusolverHandle
#ifdef WITH_NVIDIA_CUSOLVER
      call cusolver_Cpotrf_c(cusolverHandle, uplo, n, a_dev, lda, info_dev)
#endif
    end subroutine

    ! cusolver_Xpotrf_bufferSize

    subroutine cusolver_Xpotrf_bufferSize(cusolverHandle, uplo, n, dataType, a_dev, lda, &
                                           workspaceInBytesOnDevice, workspaceInBytesOnHost)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t)        :: cusolverHandle
      character(1, c_char), value     :: uplo, dataType
      integer(kind=c_int)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev
      integer(kind=c_size_t)          :: workspaceInBytesOnDevice, workspaceInBytesOnHost

#ifdef WITH_NVIDIA_CUSOLVER
      call cusolver_Xpotrf_bufferSize_c(cusolverHandle, uplo, n, dataType, a_dev, lda, &
                                        workspaceInBytesOnDevice, workspaceInBytesOnHost)
#endif
    end subroutine

    ! cusolver_Xpotrf

    subroutine cusolver_Xpotrf(cusolverHandle, uplo, n, dataType, a_dev, lda, &
                               buffer_dev , workspaceInBytesOnDevice, &
                               buffer_host, workspaceInBytesOnHost, info_dev)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=c_intptr_t)        :: cusolverHandle
      character(1, c_char), value     :: uplo, dataType
      integer(kind=c_int)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, buffer_dev, info_dev
      integer(kind=c_intptr_t)        :: buffer_host
      integer(kind=c_size_t)          :: workspaceInBytesOnDevice, workspaceInBytesOnHost

#ifdef WITH_NVIDIA_CUSOLVER
      call cusolver_Xpotrf_c(cusolverHandle, uplo, n, dataType, a_dev, lda, &
                             buffer_dev , workspaceInBytesOnDevice, &
                             buffer_host, workspaceInBytesOnHost, info_dev)
#endif
    end subroutine

