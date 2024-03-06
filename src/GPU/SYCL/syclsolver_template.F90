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
!    function syclsolver_set_stream_c(syclsolverHandle, syclStream) result(istat) &
!             bind(C, name="syclsolverSetStreamFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!
!      integer(kind=C_intptr_T), value  :: syclsolverHandle
!      integer(kind=C_intptr_T), value  :: syclStream
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface

  interface
    function syclsolver_create_c(syclsolverHandle) result(istat) &
             bind(C, name="syclsolverCreateFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: syclsolverHandle
      integer(kind=C_INT)      :: istat
    end function
  end interface

  interface
    function syclsolver_destroy_c(syclsolverHandle) result(istat) &
             bind(C, name="syclsolverDestroyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: syclsolverHandle
      integer(kind=C_INT)      :: istat
    end function
  end interface

  ! syclsolver_?trtri_c

  interface
    subroutine syclsolver_Dtrtri_c(syclsolverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="syclsolverDtrtri_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: syclsolverHandle
    end subroutine
  end interface

  interface
    subroutine syclsolver_Strtri_c(syclsolverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="syclsolverStrtri_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: syclsolverHandle
    end subroutine
  end interface

  interface
    subroutine syclsolver_Ztrtri_c(syclsolverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="syclsolverZtrtri_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: syclsolverHandle
    end subroutine
  end interface

  interface
    subroutine syclsolver_Ctrtri_c(syclsolverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="syclsolverCtrtri_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: syclsolverHandle
    end subroutine
  end interface

  ! syclsolver_?potrf_c

  interface
    subroutine syclsolver_Dpotrf_c(syclsolverHandle, uplo, n, a_dev, lda, info_dev) &
                              bind(C,name="syclsolverDpotrf_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a_dev, info_dev
      integer(kind=C_intptr_T), value           :: syclsolverHandle
    end subroutine
  end interface

  interface
    subroutine syclsolver_Spotrf_c(syclsolverHandle, uplo, n, a_dev, lda, info_dev) &
                              bind(C,name="syclsolverSpotrf_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a_dev, info_dev
      integer(kind=C_intptr_T), value           :: syclsolverHandle
    end subroutine
  end interface

  interface
    subroutine syclsolver_Zpotrf_c(syclsolverHandle, uplo, n, a_dev, lda, info_dev) &
                              bind(C,name="syclsolverZpotrf_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a_dev, info_dev
      integer(kind=C_intptr_T), value           :: syclsolverHandle
    end subroutine
  end interface

  interface
    subroutine syclsolver_Cpotrf_c(syclsolverHandle, uplo, n, a_dev, lda, info_dev) &
                              bind(C,name="syclsolverCpotrf_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a_dev, info_dev
      integer(kind=C_intptr_T), value           :: syclsolverHandle
    end subroutine
  end interface

  ! syclsolver_Xpotrf_buffereSize_c

!  interface
!    subroutine syclsolver_Xpotrf_bufferSize_c(syclsolverHandle, uplo, n, dataType, a_dev, lda, &
!                                            workspaceInBytesOnDevice, workspaceInBytesOnHost) &
!                                            bind(C,name="syclsolverXpotrf_bufferSize_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t), value          :: syclsolverHandle
!      character(1, c_char), value              :: uplo, dataType
!      integer(kind=c_int), intent(in), value   :: n, lda
!      integer(kind=c_intptr_t), value          :: a_dev
!      integer(kind=c_size_t)                   :: workspaceInBytesOnDevice, workspaceInBytesOnHost
!    end subroutine
!  end interface

  ! syclsolver_Xpotrf_c

!  interface
!    subroutine syclsolver_Xpotrf_c(syclsolverHandle, uplo, n, dataType, a_dev, lda, &
!                                 buffer_dev , workspaceInBytesOnDevice, &
!                                 buffer_host, workspaceInBytesOnHost, info_dev) &
!                                 bind(C,name="syclsolverXpotrf_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t), value          :: syclsolverHandle
!      character(1, c_char), value              :: uplo, dataType
!      integer(kind=c_int), intent(in), value   :: n, lda
!      integer(kind=c_intptr_t), value          :: a_dev, buffer_dev, info_dev
!      integer(kind=c_intptr_t), value          :: buffer_host
!      integer(kind=c_size_t)                   :: workspaceInBytesOnDevice, workspaceInBytesOnHost
!    end subroutine
!  end interface


  contains


!    function syclsolver_set_stream(syclsolverHandle, syclStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: syclsolverHandle
!      integer(kind=C_intptr_t)                  :: syclStream
!      logical                                   :: success
!
!#ifdef WITH_SYCL_SOLVER
!      success = syclsolver_set_stream_c(syclsolverHandle, syclStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

    function syclsolver_create(syclsolverHandle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: syclsolverHandle
      logical                                   :: success
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      success = syclsolver_create_c(syclsolverHandle) /= 0
#else
      success = .true.
#endif
    end function

    function syclsolver_destroy(syclsolverHandle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: syclsolverHandle
      logical                                   :: success
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      success = syclsolver_destroy_c(syclsolverHandle) /= 0
#else
      success = .true.
#endif
    end function

    ! syclsolver_?trtri

    subroutine syclsolver_Dtrtri(uplo, diag, n, a, lda, info, syclsolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: syclsolverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call syclsolver_Dtrtri_c(syclsolverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    subroutine syclsolver_Strtri(uplo, diag, n, a, lda, info, syclsolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: syclsolverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call syclsolver_Strtri_c(syclsolverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    subroutine syclsolver_Ztrtri(uplo, diag, n, a, lda, info, syclsolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: syclsolverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call syclsolver_Ztrtri_c(syclsolverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    subroutine syclsolver_Ctrtri(uplo, diag, n, a, lda, info, syclsolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: syclsolverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call syclsolver_Ctrtri_c(syclsolverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    ! syclsolver_?potrf

    subroutine syclsolver_Dpotrf(uplo, n, a_dev, lda, info_dev, syclsolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=C_intptr_T)        :: syclsolverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call syclsolver_Dpotrf_c(syclsolverHandle, uplo, n, a_dev, lda, info_dev)
#endif
    end subroutine

    subroutine syclsolver_Spotrf(uplo, n, a_dev, lda, info_dev, syclsolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=C_intptr_T)        :: syclsolverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call syclsolver_Spotrf_c(syclsolverHandle, uplo, n, a_dev, lda, info_dev)
#endif
    end subroutine

    subroutine syclsolver_Zpotrf(uplo, n, a_dev, lda, info_dev, syclsolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=C_intptr_T)        :: syclsolverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call syclsolver_Zpotrf_c(syclsolverHandle, uplo, n, a_dev, lda, info_dev)
#endif
    end subroutine

    subroutine syclsolver_Cpotrf(uplo, n, a_dev, lda, info_dev, syclsolverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=C_intptr_T)        :: syclsolverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call syclsolver_Cpotrf_c(syclsolverHandle, uplo, n, a_dev, lda, info_dev)
#endif
    end subroutine

    ! syclsolver_Xpotrf_bufferSize

!    subroutine syclsolver_Xpotrf_bufferSize(syclsolverHandle, uplo, n, dataType, a_dev, lda, &
!                                           workspaceInBytesOnDevice, workspaceInBytesOnHost)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t)        :: syclsolverHandle
!      character(1, c_char), value     :: uplo, dataType
!      integer(kind=c_int)             :: n, lda
!      integer(kind=c_intptr_t)        :: a_dev
!      integer(kind=c_size_t)          :: workspaceInBytesOnDevice, workspaceInBytesOnHost
!
!#ifdef WITH_OPENMP_OFFLOAD_SOLVER
!      call syclsolver_Xpotrf_bufferSize_c(syclsolverHandle, uplo, n, dataType, a_dev, lda, &
!                                        workspaceInBytesOnDevice, workspaceInBytesOnHost)
!#endif
!    end subroutine

    ! syclsolver_Xpotrf

!    subroutine syclsolver_Xpotrf(syclsolverHandle, uplo, n, dataType, a_dev, lda, &
!                               buffer_dev , workspaceInBytesOnDevice, &
!                               buffer_host, workspaceInBytesOnHost, info_dev)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t)        :: syclsolverHandle
!      character(1, c_char), value     :: uplo, dataType
!      integer(kind=c_int)             :: n, lda
!      integer(kind=c_intptr_t)        :: a_dev, buffer_dev, info_dev
!      integer(kind=c_intptr_t)        :: buffer_host
!      integer(kind=c_size_t)          :: workspaceInBytesOnDevice, workspaceInBytesOnHost
!
!#ifdef WITH_OPENMP_OFFLOAD_SOLVER
!      call syclsolver_Xpotrf_c(syclsolverHandle, uplo, n, dataType, a_dev, lda, &
!                             buffer_dev , workspaceInBytesOnDevice, &
!                             buffer_host, workspaceInBytesOnHost, info_dev)
!#endif
!    end subroutine

