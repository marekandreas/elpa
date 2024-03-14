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
!    function openmp_offload_solver_set_stream_c(openmp_offload_solverHandle, openmp_offloadStream) result(istat) &
!             bind(C, name="openmpOffloadsolverSetStreamFromC")
!      use, intrinsic :: iso_c_binding
!      implicit none
!
!      integer(kind=C_intptr_T), value  :: openmp_offload_solverHandle
!      integer(kind=C_intptr_T), value  :: openmp_offloadStream
!      integer(kind=C_INT)              :: istat
!    end function
!  end interface

  interface
    function openmp_offload_solver_create_c(openmp_offload_solverHandle) result(istat) &
             bind(C, name="openmpOffloadsolverCreateFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: openmp_offload_solverHandle
      integer(kind=C_INT)      :: istat
    end function
  end interface

  interface
    function openmp_offload_solver_destroy_c(openmp_offload_solverHandle) result(istat) &
             bind(C, name="openmpOffloadsolverDestroyFromC")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_T) :: openmp_offload_solverHandle
      integer(kind=C_INT)      :: istat
    end function
  end interface

  ! openmp_offload_solver_?trtri_c

  interface
    subroutine openmp_offload_solver_Dtrtri_c(openmp_offload_solverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="openmpOffloadsolverDtrtri_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: openmp_offload_solverHandle
    end subroutine
  end interface

  interface
    subroutine openmp_offload_solver_Strtri_c(openmp_offload_solverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="openmpOffloadsolverStrtri_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: openmp_offload_solverHandle
    end subroutine
  end interface

  interface
    subroutine openmp_offload_solver_Ztrtri_c(openmp_offload_solverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="openmpOffloadsolverZtrtri_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: openmp_offload_solverHandle
    end subroutine
  end interface

  interface
    subroutine openmp_offload_solver_Ctrtri_c(openmp_offload_solverHandle, uplo, diag, n, a, lda, info) &
                              bind(C,name="openmpOffloadsolverCtrtri_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo, diag
      integer(kind=C_INT64_T), intent(in),value :: n, lda
      integer(kind=C_intptr_T), value           :: a
      integer(kind=C_INT)                       :: info
      integer(kind=C_intptr_T), value           :: openmp_offload_solverHandle
    end subroutine
  end interface

  ! openmp_offload_solver_?potrf_c

  interface
    subroutine openmp_offload_solver_Dpotrf_c(openmp_offload_solverHandle, uplo, n, a_dev, lda, info_dev) &
                              bind(C,name="openmpOffloadsolverDpotrf_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a_dev, info_dev
      integer(kind=C_intptr_T), value           :: openmp_offload_solverHandle
    end subroutine
  end interface

  interface
    subroutine openmp_offload_solver_Spotrf_c(openmp_offload_solverHandle, uplo, n, a_dev, lda, info_dev) &
                              bind(C,name="openmpOffloadsolverSpotrf_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a_dev, info_dev
      integer(kind=C_intptr_T), value           :: openmp_offload_solverHandle
    end subroutine
  end interface

  interface
    subroutine openmp_offload_solver_Zpotrf_c(openmp_offload_solverHandle, uplo, n, a_dev, lda, info_dev) &
                              bind(C,name="openmpOffloadsolverZpotrf_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a_dev, info_dev
      integer(kind=C_intptr_T), value           :: openmp_offload_solverHandle
    end subroutine
  end interface

  interface
    subroutine openmp_offload_solver_Cpotrf_c(openmp_offload_solverHandle, uplo, n, a_dev, lda, info_dev) &
                              bind(C,name="openmpOffloadsolverCpotrf_elpa_wrapper")
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value                 :: uplo
      integer(kind=C_INT), intent(in),value     :: n, lda
      integer(kind=C_intptr_T), value           :: a_dev, info_dev
      integer(kind=C_intptr_T), value           :: openmp_offload_solverHandle
    end subroutine
  end interface

  ! openmp_offload_solver_Xpotrf_buffereSize_c

!  interface
!    subroutine openmp_offload_solver_Xpotrf_bufferSize_c(openmp_offload_solverHandle, uplo, n, dataType, a_dev, lda, &
!                                            workspaceInBytesOnDevice, workspaceInBytesOnHost) &
!                                            bind(C,name="openmpOffloadsolverXpotrf_bufferSize_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t), value          :: openmp_offload_solverHandle
!      character(1, c_char), value              :: uplo, dataType
!      integer(kind=c_int), intent(in), value   :: n, lda
!      integer(kind=c_intptr_t), value          :: a_dev
!      integer(kind=c_size_t)                   :: workspaceInBytesOnDevice, workspaceInBytesOnHost
!    end subroutine
!  end interface

  ! openmp_offload_solver_Xpotrf_c

!  interface
!    subroutine openmp_offload_solver_Xpotrf_c(openmp_offload_solverHandle, uplo, n, dataType, a_dev, lda, &
!                                 buffer_dev , workspaceInBytesOnDevice, &
!                                 buffer_host, workspaceInBytesOnHost, info_dev) &
!                                 bind(C,name="openmpOffloadsolverXpotrf_elpa_wrapper")
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t), value          :: openmp_offload_solverHandle
!      character(1, c_char), value              :: uplo, dataType
!      integer(kind=c_int), intent(in), value   :: n, lda
!      integer(kind=c_intptr_t), value          :: a_dev, buffer_dev, info_dev
!      integer(kind=c_intptr_t), value          :: buffer_host
!      integer(kind=c_size_t)                   :: workspaceInBytesOnDevice, workspaceInBytesOnHost
!    end subroutine
!  end interface


  contains


!    function openmp_offload_solver_set_stream(openmp_offload_solverHandle, openmpOffloadStream) result(success)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=C_intptr_t)                  :: openmp_offload_solverHandle
!      integer(kind=C_intptr_t)                  :: openmpOffloadStream
!      logical                                   :: success
!
!#ifdef WITH_OPENMP_OFFLOAD_OPENMP_OFFLOAD_SOLVER
!      success = openmp_offload_solver_set_stream_c(openmp_offload_solverHandle, openmpOffloadStream) /= 0
!#else
!      success = .true.
!#endif
!    end function

    function openmp_offload_solver_create(openmp_offload_solverHandle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: openmp_offload_solverHandle
      logical                                   :: success
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      success = openmp_offload_solver_create_c(openmp_offload_solverHandle) /= 0
#else
      success = .true.
#endif
    end function

    function openmp_offload_solver_destroy(openmp_offload_solverHandle) result(success)
      use, intrinsic :: iso_c_binding
      implicit none
      integer(kind=C_intptr_t)                  :: openmp_offload_solverHandle
      logical                                   :: success
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      success = openmp_offload_solver_destroy_c(openmp_offload_solverHandle) /= 0
#else
      success = .true.
#endif
    end function

    ! openmp_offload_solver_?trtri

    subroutine openmp_offload_solver_Dtrtri(uplo, diag, n, a, lda, info, openmp_offload_solverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: openmp_offload_solverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call openmp_offload_solver_Dtrtri_c(openmp_offload_solverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    subroutine openmp_offload_solver_Strtri(uplo, diag, n, a, lda, info, openmp_offload_solverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: openmp_offload_solverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call openmp_offload_solver_Strtri_c(openmp_offload_solverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    subroutine openmp_offload_solver_Ztrtri(uplo, diag, n, a, lda, info, openmp_offload_solverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: openmp_offload_solverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call openmp_offload_solver_Ztrtri_c(openmp_offload_solverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    subroutine openmp_offload_solver_Ctrtri(uplo, diag, n, a, lda, info, openmp_offload_solverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo, diag
      integer(kind=C_INT64_T)         :: n, lda
      integer(kind=c_intptr_t)        :: a
      integer(kind=c_int)             :: info
      integer(kind=C_intptr_T)        :: openmp_offload_solverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call openmp_offload_solver_Ctrtri_c(openmp_offload_solverHandle, uplo, diag, n, a, lda, info)
#endif
    end subroutine

    ! openmp_offload_solver_?potrf

    subroutine openmp_offload_solver_Dpotrf(uplo, n, a_dev, lda, info_dev, openmp_offload_solverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=C_intptr_T)        :: openmp_offload_solverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call openmp_offload_solver_Dpotrf_c(openmp_offload_solverHandle, uplo, n, a_dev, lda, info_dev)
#endif
    end subroutine

    subroutine openmp_offload_solver_Spotrf(uplo, n, a_dev, lda, info_dev, openmp_offload_solverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=C_intptr_T)        :: openmp_offload_solverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call openmp_offload_solver_Spotrf_c(openmp_offload_solverHandle, uplo, n, a_dev, lda, info_dev)
#endif
    end subroutine

    subroutine openmp_offload_solver_Zpotrf(uplo, n, a_dev, lda, info_dev, openmp_offload_solverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=C_intptr_T)        :: openmp_offload_solverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call openmp_offload_solver_Zpotrf_c(openmp_offload_solverHandle, uplo, n, a_dev, lda, info_dev)
#endif
    end subroutine

    subroutine openmp_offload_solver_Cpotrf(uplo, n, a_dev, lda, info_dev, openmp_offload_solverHandle)
      use, intrinsic :: iso_c_binding
      implicit none
      character(1,C_CHAR),value       :: uplo
      integer(kind=C_INT)             :: n, lda
      integer(kind=c_intptr_t)        :: a_dev, info_dev
      integer(kind=C_intptr_T)        :: openmp_offload_solverHandle
#ifdef WITH_OPENMP_OFFLOAD_SOLVER
      call openmp_offload_solver_Cpotrf_c(openmp_offload_solverHandle, uplo, n, a_dev, lda, info_dev)
#endif
    end subroutine

    ! openmp_offload_solver_Xpotrf_bufferSize

!    subroutine openmp_offload_solver_Xpotrf_bufferSize(openmp_offload_solverHandle, uplo, n, dataType, a_dev, lda, &
!                                           workspaceInBytesOnDevice, workspaceInBytesOnHost)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t)        :: openmp_offload_solverHandle
!      character(1, c_char), value     :: uplo, dataType
!      integer(kind=c_int)             :: n, lda
!      integer(kind=c_intptr_t)        :: a_dev
!      integer(kind=c_size_t)          :: workspaceInBytesOnDevice, workspaceInBytesOnHost
!
!#ifdef WITH_OPENMP_OFFLOAD_SOLVER
!      call openmp_offload_solver_Xpotrf_bufferSize_c(openmp_offload_solverHandle, uplo, n, dataType, a_dev, lda, &
!                                        workspaceInBytesOnDevice, workspaceInBytesOnHost)
!#endif
!    end subroutine

    ! openmp_offload_solver_Xpotrf

!    subroutine openmp_offload_solver_Xpotrf(openmp_offload_solverHandle, uplo, n, dataType, a_dev, lda, &
!                               buffer_dev , workspaceInBytesOnDevice, &
!                               buffer_host, workspaceInBytesOnHost, info_dev)
!      use, intrinsic :: iso_c_binding
!      implicit none
!      integer(kind=c_intptr_t)        :: openmp_offload_solverHandle
!      character(1, c_char), value     :: uplo, dataType
!      integer(kind=c_int)             :: n, lda
!      integer(kind=c_intptr_t)        :: a_dev, buffer_dev, info_dev
!      integer(kind=c_intptr_t)        :: buffer_host
!      integer(kind=c_size_t)          :: workspaceInBytesOnDevice, workspaceInBytesOnHost
!
!#ifdef WITH_OPENMP_OFFLOAD_SOLVER
!      call openmp_offload_solver_Xpotrf_c(openmp_offload_solverHandle, uplo, n, dataType, a_dev, lda, &
!                             buffer_dev , workspaceInBytesOnDevice, &
!                             buffer_host, workspaceInBytesOnHost, info_dev)
!#endif
!    end subroutine

