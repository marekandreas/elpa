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


!  interface
!    function rocsolver_set_stream_c(rocsolverHandle, hipStream) result(istat) &
!             bind(C, name="rocsolverSetStreamFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!
!      integer(kind=C_intptr_T), value  :: rocsolverHandle
!      integer(kind=C_intptr_T), value  :: hipStream
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface

!  interface
!    function rocsolver_create_c(rocsolverHandle) result(istat) &
!             bind(C, name="rocsolverCreateFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T) :: rocsolverHandle
!      integer(kind=C_INT)      :: istat
!    end function
!  end interface
!
!  interface
!    function rocsolver_destroy_c(rocsolverHandle) result(istat) &
!             bind(C, name="rocsolverDestroyFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_T), value :: rocsolverHandle
!      integer(kind=C_INT)      :: istat
!    end function
!  end interface

  ! rocsolver_?trtri_c

#ifndef WITH_AMD_HIPSOLVER_API
  interface
    subroutine rocsolver_Dtrtri_c(rocsolverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="rocsolverDtrtri_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: rocsolverHandle
    end subroutine
  end interface
#endif

#ifndef WITH_AMD_HIPSOLVER_API
  interface
    subroutine rocsolver_Strtri_c(rocsolverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="rocsolverStrtri_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: rocsolverHandle
    end subroutine
  end interface
#endif

#ifndef WITH_AMD_HIPSOLVER_API
  interface
    subroutine rocsolver_Ztrtri_c(rocsolverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="rocsolverZtrtri_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: rocsolverHandle
    end subroutine
  end interface
#endif

#ifndef WITH_AMD_HIPSOLVER_API
  interface
    subroutine rocsolver_Ctrtri_c(rocsolverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="rocsolverCtrtri_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: rocsolverHandle
    end subroutine
  end interface
#endif

  ! rocsolver_?potrf_c

  interface
    subroutine rocsolver_Dpotrf_c(rocsolverHandle, uplo, n, a_dev, lda, info_dev) &
                              bind(C,name="rocsolverDpotrf_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a_dev, info_dev
      integer(kind=C_intptr_T), value           :: rocsolverHandle
    end subroutine
  end interface

  interface
    subroutine rocsolver_Spotrf_c(rocsolverHandle, uplo, n, a_dev, lda, info_dev) &
                              bind(C,name="rocsolverSpotrf_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a_dev, info_dev
      integer(kind=C_intptr_T), value           :: rocsolverHandle
    end subroutine
  end interface

  interface
    subroutine rocsolver_Zpotrf_c(rocsolverHandle, uplo, n, a_dev, lda, info_dev) &
                              bind(C,name="rocsolverZpotrf_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a_dev, info_dev
      integer(kind=C_intptr_T), value           :: rocsolverHandle
    end subroutine
  end interface

  interface
    subroutine rocsolver_Cpotrf_c(rocsolverHandle, uplo, n, a_dev, lda, info_dev) &
                              bind(C,name="rocsolverCpotrf_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a_dev, info_dev
      integer(kind=C_intptr_T), value           :: rocsolverHandle
    end subroutine
  end interface

  ! rocsolver_Xpotrf_buffereSize_c

!  interface
!    subroutine rocsolver_Xpotrf_bufferSize_c(rocsolverHandle, uplo, n, dataType, a_dev, lda, &
!                                            workspaceInBytesOnDevice, workspaceInBytesOnHost) &
!                                            bind(C,name="rocsolverXpotrf_bufferSize_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t), value          :: rocsolverHandle
!      character(1, c_char), value              :: uplo, dataType
!      integer(kind=c_int), intent(in), value   :: n, lda
!      integer(kind=c_intptr_t), value          :: a_dev
!      integer(kind=c_size_t)                   :: workspaceInBytesOnDevice, workspaceInBytesOnHost
!    end subroutine
!  end interface

  ! rocsolver_Xpotrf_c

!  interface
!    subroutine rocsolver_Xpotrf_c(rocsolverHandle, uplo, n, dataType, a_dev, lda, &
!                                 buffer_dev , workspaceInBytesOnDevice, &
!                                 buffer_host, workspaceInBytesOnHost, info_dev) &
!                                 bind(C,name="rocsolverXpotrf_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t), value          :: rocsolverHandle
!      character(1, c_char), value              :: uplo, dataType
!      integer(kind=c_int), intent(in), value   :: n, lda
!      integer(kind=c_intptr_t), value          :: a_dev, buffer_dev, info_dev
!      integer(kind=c_intptr_t), value          :: buffer_host
!      integer(kind=c_size_t)                   :: workspaceInBytesOnDevice, workspaceInBytesOnHost
!    end subroutine
!  end interface


  contains


!    function rocsolver_set_stream(rocsolverHandle, hipStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: rocsolverHandle
!      integer(kind=C_intptr_t)                  :: hipStream
!      logical                                   :: success
!
!#ifdef WITH_AMD_ROCSOLVER
!      success = rocsolver_set_stream_c(rocsolverHandle, hipStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function rocsolver_create(rocsolverHandle) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: rocsolverHandle
!      logical                                   :: success
!#ifdef WITH_AMD_ROCSOLVER
!      success = rocsolver_create_c(rocsolverHandle) /= 0
!#else
!      success = .true.
!#endif
!    end function

!    function rocsolver_destroy(rocsolverHandle) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: rocsolverHandle
!      logical                                   :: success
!#ifdef WITH_AMD_ROCSOLVER
!      success = rocsolver_destroy_c(rocsolverHandle) /= 0
!#else
!      success = .true.
!#endif
!    end function

    ! rocsolver_?trtri

#ifndef WITH_AMD_HIPSOLVER_API
    subroutine rocsolver_Dtrtri(uplo, diag, n, a, lda, info, rocsolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: rocsolverHandle
#ifdef WITH_AMD_ROCSOLVER
      call rocsolver_Dtrtri_c(rocsolverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine
#endif

#ifndef WITH_AMD_HIPSOLVER_API
    subroutine rocsolver_Strtri(uplo, diag, n, a, lda, info, rocsolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: rocsolverHandle
#ifdef WITH_AMD_ROCSOLVER
      call rocsolver_Strtri_c(rocsolverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine
#endif

#ifndef WITH_AMD_HIPSOLVER_API
    subroutine rocsolver_Ztrtri(uplo, diag, n, a, lda, info, rocsolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: rocsolverHandle
#ifdef WITH_AMD_ROCSOLVER
      call rocsolver_Ztrtri_c(rocsolverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine
#endif

#ifndef WITH_AMD_HIPSOLVER_API
    subroutine rocsolver_Ctrtri(uplo, diag, n, a, lda, info, rocsolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: rocsolverHandle
#ifdef WITH_AMD_ROCSOLVER
      call rocsolver_Ctrtri_c(rocsolverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine
#endif

    ! rocsolver_?potrf

    subroutine rocsolver_Dpotrf(uplo, n, a_dev, lda, info_dev, rocsolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=C_intptr_T)        :: rocsolverHandle
#ifdef WITH_AMD_ROCSOLVER
      call rocsolver_Dpotrf_c(rocsolverHandle, uplo, n, a_dev, lda, info_dev)
#endif
    end subroutine

    subroutine rocsolver_Spotrf(uplo, n, a_dev, lda, info_dev, rocsolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=C_intptr_T)        :: rocsolverHandle
#ifdef WITH_AMD_ROCSOLVER
      call rocsolver_Spotrf_c(rocsolverHandle, uplo, n, a_dev, lda, info_dev)
#endif
    end subroutine

    subroutine rocsolver_Zpotrf(uplo, n, a_dev, lda, info_dev, rocsolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=C_intptr_T)        :: rocsolverHandle
#ifdef WITH_AMD_ROCSOLVER
      call rocsolver_Zpotrf_c(rocsolverHandle, uplo, n, a_dev, lda, info_dev)
#endif
    end subroutine

    subroutine rocsolver_Cpotrf(uplo, n, a_dev, lda, info_dev, rocsolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=C_intptr_T)        :: rocsolverHandle
#ifdef WITH_AMD_ROCSOLVER
      call rocsolver_Cpotrf_c(rocsolverHandle, uplo, n, a_dev, lda, info_dev)
#endif
    end subroutine

    ! rocsolver_Xpotrf_bufferSize

!    subroutine rocsolver_Xpotrf_bufferSize(rocsolverHandle, uplo, n, dataType, a_dev, lda, &
!                                           workspaceInBytesOnDevice, workspaceInBytesOnHost)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t)        :: rocsolverHandle
!      character(1, c_char), value     :: uplo, dataType
!      integer(kind=c_int)             :: n, lda
!      integer(kind=c_intptr_t)        :: a_dev
!      integer(kind=c_size_t)          :: workspaceInBytesOnDevice, workspaceInBytesOnHost
!
!#ifdef WITH_AMD_ROCSOLVER
!      call rocsolver_Xpotrf_bufferSize_c(rocsolverHandle, uplo, n, dataType, a_dev, lda, &
!                                        workspaceInBytesOnDevice, workspaceInBytesOnHost)
!#endif
!    end subroutine

    ! rocsolver_Xpotrf

!    subroutine rocsolver_Xpotrf(rocsolverHandle, uplo, n, dataType, a_dev, lda, &
!                               buffer_dev , workspaceInBytesOnDevice, &
!                               buffer_host, workspaceInBytesOnHost, info_dev)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t)        :: rocsolverHandle
!      character(1, c_char), value     :: uplo, dataType
!      integer(kind=c_int)             :: n, lda
!      integer(kind=c_intptr_t)        :: a_dev, buffer_dev, info_dev
!      integer(kind=c_intptr_t)        :: buffer_host
!      integer(kind=c_size_t)          :: workspaceInBytesOnDevice, workspaceInBytesOnHost
!
!#ifdef WITH_AMD_ROCSOLVER
!      call rocsolver_Xpotrf_c(rocsolverHandle, uplo, n, dataType, a_dev, lda, &
!                             buffer_dev , workspaceInBytesOnDevice, &
!                             buffer_host, workspaceInBytesOnHost, info_dev)
!#endif
!    end subroutine

