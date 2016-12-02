!   This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), fomerly known as
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



! ELPA2 -- 2-stage solver for ELPA
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".


#include "config-f90.h"
!> \brief Fortran module which provides the routines to use the 2-stage ELPA solver
module ELPA2

! Version 1.1.2, 2011-02-21

  use elpa_utilities
  use elpa1, only : elpa_print_times, time_evp_back, time_evp_fwd, time_evp_solve
  use elpa2_utilities

  implicit none

  PRIVATE ! By default, all routines contained are private

  ! The following routines are public:

  public :: solve_evp_real_2stage_double               !< old, deprecated interface: Driver routine for real double-precision eigenvalue problem. will be deleted at some point
  public :: solve_evp_complex_2stage_double            !< old, deprecated interface: Driver routine for complex double-precision eigenvalue problem. will be deleted at some point
  public :: elpa_solve_evp_real_2stage_double          !< Driver routine for real double-precision 2-stage eigenvalue problem
  public :: elpa_solve_evp_complex_2stage_double       !< Driver routine for complex double-precision 2-stage eigenvalue problem


!-------------------------------------------------------------------------------
!>  \brief solve_evp_real_2stage: Old, deprecated interface for elpa_solve_evp_real_2stage_double
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param THIS_REAL_ELPA_KERNEL_API (optional) specify used ELPA2 kernel via API
!>
!>  \param useQR (optional)                     use QR decomposition
!>  \param useGPU (optional)                    decide whether to use GPUs or not

!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
  interface solve_evp_real_2stage
    module procedure solve_evp_real_2stage_double
  end interface

!-------------------------------------------------------------------------------
!>  \brief elpa_solve_evp_real_2stage_double: Fortran function to solve the real double-precision eigenvalue problem with a 2 stage approach. This is called by "elpa_solve_evp_real_double"
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param THIS_REAL_ELPA_KERNEL_API (optional) specify used ELPA2 kernel via API
!>
!>  \param useQR (optional)                     use QR decomposition
!>  \param useGPU (optional)                    decide whether to use GPUs or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
  interface elpa_solve_evp_real_2stage_double
    module procedure solve_evp_real_2stage_double
  end interface

!-------------------------------------------------------------------------------
!>  \brief solve_evp_complex_2stage: Old, deprecated interface for elpa_solve_evp_complex_2stage_double
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param THIS_REAL_ELPA_KERNEL_API (optional) specify used ELPA2 kernel via API
!>
!>  \param useGPU (optional)                    decide whether to use GPUs or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
  interface solve_evp_complex_2stage
    module procedure solve_evp_complex_2stage_double
  end interface

!-------------------------------------------------------------------------------
!>  \brief elpa_solve_evp_complex_2stage_double: Fortran function to solve the complex double-precision eigenvalue problem with a 2 stage approach. This is called by "elpa_solve_evp_complex_double"
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param THIS_REAL_ELPA_KERNEL_API (optional) specify used ELPA2 kernel via API
!>
!>  \param useGPU (optional)                    decide whether to use GPUs or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
  interface elpa_solve_evp_complex_2stage_double
    module procedure solve_evp_complex_2stage_double
  end interface

#ifdef WANT_SINGLE_PRECISION_REAL
  public :: solve_evp_real_2stage_single
  public :: elpa_solve_evp_real_2stage_single
#endif

#ifdef WANT_SINGLE_PRECISION_COMPLEX
  public :: solve_evp_complex_2stage_single
  public :: elpa_solve_evp_complex_2stage_single
#endif

#ifdef WANT_SINGLE_PRECISION_REAL
!-------------------------------------------------------------------------------
!>  \brief elpa_solve_evp_real_2stage_single: Fortran function to solve the real single-precision eigenvalue problem with a 2 stage approach. This is called by "elpa_solve_evp_real_single"
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param THIS_REAL_ELPA_KERNEL_API (optional) specify used ELPA2 kernel via API
!>
!>  \param useQR (optional)                     use QR decomposition
!>  \param useGPU (optional)                    decide whether to use GPUs or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
  interface elpa_solve_evp_real_2stage_single
    module procedure solve_evp_real_2stage_single
  end interface
#endif

#ifdef WANT_SINGLE_PRECISION_COMPLEX
!-------------------------------------------------------------------------------
!>  \brief elpa_solve_evp_complex_2stage_single: Fortran function to solve the complex double-precision eigenvalue problem with a 2 stage approach. This is called by "elpa_solve_evp_complex_single"
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param THIS_REAL_ELPA_KERNEL_API (optional) specify used ELPA2 kernel via API
!>
!>  \param useGPU (optional)                    decide whether to use GPUs or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
  interface elpa_solve_evp_complex_2stage_single
    module procedure solve_evp_complex_2stage_single
  end interface
#endif

  contains
!-------------------------------------------------------------------------------
!>  \brief solve_evp_real_2stage_double: Fortran function to solve the double-precision real eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param THIS_REAL_ELPA_KERNEL_API (optional) specify used ELPA2 kernel via API
!>
!>  \param useQR (optional)                     use QR decomposition
!>  \param useGPU (optional)                    decide whether to use GPUs or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------

#define DOUBLE_PRECISION_REAL

#ifdef DOUBLE_PRECISION_REAL
  function solve_evp_real_2stage_double(na, nev, a, lda, ev, q, ldq, nblk,        &
                               matrixCols,                               &
                                 mpi_comm_rows, mpi_comm_cols,           &
                                 mpi_comm_all, THIS_REAL_ELPA_KERNEL_API,&
                                 useQR, useGPU) result(success)
#else
  function solve_evp_real_2stage_single(na, nev, a, lda, ev, q, ldq, nblk,        &
                               matrixCols,                               &
                                 mpi_comm_rows, mpi_comm_cols,           &
                                 mpi_comm_all, THIS_REAL_ELPA_KERNEL_API,&
                                 useQR, useGPU) result(success)
#endif


#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif
   use elpa1_utilities, only : gpu_usage_via_environment_variable
   use elpa1_compute
   use elpa2_compute
   use elpa_mpi
   use cuda_functions
   use mod_check_for_gpu
   use iso_c_binding
   implicit none
   logical, intent(in), optional             :: useQR, useGPU
   logical                                   :: useQRActual, useQREnvironment
   integer(kind=c_int), intent(in), optional :: THIS_REAL_ELPA_KERNEL_API
   integer(kind=c_int)                       :: THIS_REAL_ELPA_KERNEL

   integer(kind=c_int), intent(in)           :: na, nev, lda, ldq, matrixCols, mpi_comm_rows, &
                                                mpi_comm_cols, mpi_comm_all
   integer(kind=c_int), intent(in)           :: nblk
   real(kind=c_double), intent(inout)        :: ev(na)
#ifdef USE_ASSUMED_SIZE
   real(kind=c_double), intent(inout)        :: a(lda,*), q(ldq,*)
#else
   real(kind=c_double), intent(inout)        :: a(lda,matrixCols), q(ldq,matrixCols)
#endif
   real(kind=c_double), allocatable          :: hh_trans_real(:,:)

   integer(kind=c_int)                       :: my_pe, n_pes, my_prow, my_pcol, np_rows, np_cols, mpierr
   integer(kind=c_int)                       :: nbw, num_blocks
   real(kind=c_double), allocatable          :: tmat(:,:,:), e(:)
   integer(kind=c_intptr_t)                  :: tmat_dev, q_dev, a_dev
   real(kind=c_double)                       :: ttt0, ttt1, ttts  ! MPI_WTIME always needs double
   integer(kind=c_int)                       :: i
   logical                                   :: success
   logical, save                             :: firstCall = .true.
   logical                                   :: wantDebug
   integer(kind=c_int)                       :: istat
   character(200)                            :: errorMessage
   logical                                   :: do_useGPU
   integer(kind=c_int)                       :: numberOfGPUDevices

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("solve_evp_real_2stage_double")
#endif

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("mpi_communication")
#endif
    call mpi_comm_rank(mpi_comm_all,my_pe,mpierr)
    call mpi_comm_size(mpi_comm_all,n_pes,mpierr)

    call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
    call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
    call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
    call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("mpi_communication")
#endif

    wantDebug = .false.
    if (firstCall) then
      ! are debug messages desired?
      wantDebug = debug_messages_via_environment_variable()
      firstCall = .false.
    endif

    success = .true.

    useQRActual = .false.
    do_useGPU      = .false.

    ! set usage of qr decomposition via API call
    if (present(useQR)) then
      if (useQR) useQRActual = .true.
        if (.not.(useQR)) useQRACtual = .false.
    endif

    ! overwrite this with environment variable settings
    if (qr_decomposition_via_environment_variable(useQREnvironment)) then
      useQRActual = useQREnvironment
    endif

    if (useQRActual) then
      if (mod(na,2) .ne. 0) then
        if (wantDebug) then
          write(error_unit,*) "solve_evp_real_2stage: QR-decomposition: blocksize does not fit with matrixsize"
        endif
        print *, "Do not use QR-decomposition for this matrix and blocksize."
        success = .false.
        return
      endif
    endif

    if (present(useGPU)) then
      if (useGPU) then
        if (check_for_gpu(my_pe,numberOfGPUDevices, wantDebug=wantDebug)) then

           do_useGPU = .true.
           ! set the neccessary parameters
           cudaMemcpyHostToDevice   = cuda_memcpyHostToDevice()
           cudaMemcpyDeviceToHost   = cuda_memcpyDeviceToHost()
           cudaMemcpyDeviceToDevice = cuda_memcpyDeviceToDevice()
           cudaHostRegisterPortable = cuda_hostRegisterPortable()
           cudaHostRegisterMapped   = cuda_hostRegisterMapped()
        else
          print *,"GPUs are requested but not detected! Aborting..."
          success = .false.
          return
        endif
      endif
    else
      ! check whether set by environment variable
      do_useGPU = gpu_usage_via_environment_variable()
   endif

    if (present(THIS_REAL_ELPA_KERNEL_API)) then
      ! user defined kernel via the optional argument in the API call
      THIS_REAL_ELPA_KERNEL = THIS_REAL_ELPA_KERNEL_API
    else

      ! if kernel is not choosen via api
      ! check whether set by environment variable
      THIS_REAL_ELPA_KERNEL = get_actual_real_kernel()
    endif

    ! check whether choosen kernel is allowed: function returns true if NOT allowed! change this
    if (check_allowed_real_kernels(THIS_REAL_ELPA_KERNEL)) then

      if (my_pe == 0) then
        write(error_unit,*) " "
        write(error_unit,*) "The choosen kernel ",REAL_ELPA_KERNEL_NAMES(THIS_REAL_ELPA_KERNEL)
        write(error_unit,*) "is not in the list of the allowed kernels!"
        write(error_unit,*) " "
        write(error_unit,*) "Allowed kernels are:"
        do i=1,size(REAL_ELPA_KERNEL_NAMES(:))
          if (AVAILABLE_REAL_ELPA_KERNELS(i) .ne. 0) then
            write(error_unit,*) REAL_ELPA_KERNEL_NAMES(i)
          endif
        enddo

        write(error_unit,*) " "
        ! check whether generic kernel is defined
         if (AVAILABLE_REAL_ELPA_KERNELS(REAL_ELPA_KERNEL_GENERIC) .eq. 1) then
           write(error_unit,*) "The default kernel REAL_ELPA_KERNEL_GENERIC will be used !"
         else
           write(error_unit,*) "As default kernel ",REAL_ELPA_KERNEL_NAMES(DEFAULT_REAL_ELPA_KERNEL)," will be used"
         endif
      endif  ! my_pe == 0
      if (AVAILABLE_REAL_ELPA_KERNELS(REAL_ELPA_KERNEL_GENERIC) .eq. 1) then
        THIS_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_GENERIC
      else
        THIS_REAL_ELPA_KERNEL = DEFAULT_REAL_ELPA_KERNEL
      endif
    endif

    ! check consistency between request for GPUs and defined kernel
    if (do_useGPU) then
      if (THIS_REAL_ELPA_KERNEL .ne. REAL_ELPA_KERNEL_GPU) then
        write(error_unit,*) "GPU usage has been requested but compute kernel is defined as non-GPU! Aborting..."
        success = .false.
        return
      endif
    endif

    ! Choose bandwidth, must be a multiple of nblk, set to a value >= 32
    ! On older systems (IBM Bluegene/P, Intel Nehalem) a value of 32 was optimal.
    ! For Intel(R) Xeon(R) E5 v2 and v3, better use 64 instead of 32!
    ! For IBM Bluegene/Q this is not clear at the moment. We have to keep an eye
    ! on this and maybe allow a run-time optimization here
    if (do_useGPU) then
      nbw = nblk
    else
      nbw = (63/nblk+1)*nblk
    endif

    num_blocks = (na-1)/nbw + 1

    allocate(tmat(nbw,nbw,num_blocks), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"solve_evp_real_2stage: error when allocating tmat "//errorMessage
      stop
    endif

    ! Reduction full -> band

    ttt0 = MPI_Wtime()
    ttts = ttt0
#ifdef DOUBLE_PRECISION_REAL
    call bandred_real_double(na, a, a_dev, lda, nblk, nbw, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, &
        tmat, tmat_dev, wantDebug, do_useGPU, success, useQRActual)
#else
    call bandred_real_single(na, a, a_dev, lda, nblk, nbw, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, &
        tmat, tmat_dev, wantDebug, do_useGPU, success, useQRActual)
#endif
    if (.not.(success)) return
    ttt1 = MPI_Wtime()
    if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
       write(error_unit,*) 'Time bandred_real               :',ttt1-ttt0

     ! Reduction band -> tridiagonal

     allocate(e(na), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"solve_evp_real_2stage: error when allocating e "//errorMessage
       stop
     endif

     ttt0 = MPI_Wtime()
#ifdef DOUBLE_PRECISION_REAL
     call tridiag_band_real_double(na, nbw, nblk, a, lda, ev, e, matrixCols, hh_trans_real, &
                          mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
#else
     call tridiag_band_real_single(na, nbw, nblk, a, lda, ev, e, matrixCols, hh_trans_real, &
                          mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
#endif

     ttt1 = MPI_Wtime()
     if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
       write(error_unit,*) 'Time tridiag_band_real          :',ttt1-ttt0

#ifdef WITH_MPI
#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("mpi_communication")
#endif
#ifdef DOUBLE_PRECISION_REAL
     call mpi_bcast(ev,na,MPI_REAL8,0,mpi_comm_all,mpierr)
     call mpi_bcast(e,na,MPI_REAL8,0,mpi_comm_all,mpierr)
#else
     call mpi_bcast(ev,na,MPI_REAL4,0,mpi_comm_all,mpierr)
     call mpi_bcast(e,na,MPI_REAL4,0,mpi_comm_all,mpierr)
#endif
#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("mpi_communication")
#endif
#endif /* WITH_MPI */
     ttt1 = MPI_Wtime()
     time_evp_fwd = ttt1-ttts

     ! Solve tridiagonal system

     ttt0 = MPI_Wtime()
#ifdef DOUBLE_PRECISION_REAL
     call solve_tridi_double(na, nev, ev, e, q, ldq, nblk, matrixCols, mpi_comm_rows,  &
                      mpi_comm_cols, wantDebug, success)
#else
     call solve_tridi_single(na, nev, ev, e, q, ldq, nblk, matrixCols, mpi_comm_rows,  &
                      mpi_comm_cols, wantDebug, success)
#endif
     if (.not.(success)) return

     ttt1 = MPI_Wtime()
     if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
     write(error_unit,*) 'Time solve_tridi                :',ttt1-ttt0
     time_evp_solve = ttt1-ttt0
     ttts = ttt1

     deallocate(e, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"solve_evp_real_2stage: error when deallocating e "//errorMessage
       stop
     endif
     ! Backtransform stage 1

     ttt0 = MPI_Wtime()
#ifdef DOUBLE_PRECISION_REAL
     call trans_ev_tridi_to_band_real_double(na, nev, nblk, nbw, q, q_dev, ldq, matrixCols, hh_trans_real, &
         mpi_comm_rows, mpi_comm_cols, wantDebug, do_useGPU, success,      &
                                    THIS_REAL_ELPA_KERNEL)
#else
     call trans_ev_tridi_to_band_real_single(na, nev, nblk, nbw, q, q_dev, ldq, matrixCols, hh_trans_real, &
         mpi_comm_rows, mpi_comm_cols, wantDebug, do_useGPU, success,      &
                                    THIS_REAL_ELPA_KERNEL)
#endif

     if (.not.(success)) return
     ttt1 = MPI_Wtime()
     if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
       write(error_unit,*) 'Time trans_ev_tridi_to_band_real:',ttt1-ttt0

     ! We can now deallocate the stored householder vectors
     deallocate(hh_trans_real, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"solve_evp_real_2stage: error when deallocating hh_trans_real "//errorMessage
       stop
     endif


     ! Backtransform stage 2
     ttt0 = MPI_Wtime()
#ifdef DOUBLE_PRECISION_REAL
     call trans_ev_band_to_full_real_double(na, nev, nblk, nbw, a, a_dev, lda, tmat, tmat_dev, q, q_dev, ldq, &
                                            matrixCols, num_blocks, mpi_comm_rows, &
                                            mpi_comm_cols, do_useGPU, useQRActual)
#else
     call trans_ev_band_to_full_real_single(na, nev, nblk, nbw, a, a_dev, lda, tmat, tmat_dev, q, q_dev, ldq, &
                                            matrixCols, num_blocks, mpi_comm_rows, &
                                            mpi_comm_cols, do_useGPU, useQRActual)
#endif

     ttt1 = MPI_Wtime()
     if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
       write(error_unit,*) 'Time trans_ev_band_to_full_real :',ttt1-ttt0
     time_evp_back = ttt1-ttts

     deallocate(tmat, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"solve_evp_real_2stage: error when deallocating tmat"//errorMessage
       stop
     endif

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("solve_evp_real_2stage_double")
#endif
1    format(a,f10.3)

#ifdef DOUBLE_PRECISION_REAL
   end function solve_evp_real_2stage_double
#else
   end function solve_evp_real_2stage_single
#endif

#ifdef WANT_SINGLE_PRECISION_REAL
#undef DOUBLE_PRECISION_REAL
!-------------------------------------------------------------------------------
!>  \brief solve_evp_real_2stage_single: Fortran function to solve the single-precision real eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param THIS_REAL_ELPA_KERNEL_API (optional) specify used ELPA2 kernel via API
!>
!>  \param useQR (optional)                     use QR decomposition
!>  \param useGPU (optional)                    decide whether GPUs should be used or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------

#ifdef DOUBLE_PRECISION_REAL
  function solve_evp_real_2stage_double(na, nev, a, lda, ev, q, ldq, nblk,        &
                               matrixCols,                               &
                                 mpi_comm_rows, mpi_comm_cols,           &
                                 mpi_comm_all, THIS_REAL_ELPA_KERNEL_API,&
                                 useQR, useGPU) result(success)
#else
  function solve_evp_real_2stage_single(na, nev, a, lda, ev, q, ldq, nblk,        &
                               matrixCols,                               &
                                 mpi_comm_rows, mpi_comm_cols,           &
                                 mpi_comm_all, THIS_REAL_ELPA_KERNEL_API,&
                                 useQR, useGPU) result(success)
#endif

#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif
   use elpa1_utilities, only : gpu_usage_via_environment_variable

   use cuda_functions
   use mod_check_for_gpu
   use iso_c_binding
   use elpa1_compute
   use elpa2_compute
   use elpa_mpi
   implicit none
   logical, intent(in), optional             :: useQR, useGPU
   logical                                   :: useQRActual, useQREnvironment
   integer(kind=c_int), intent(in), optional :: THIS_REAL_ELPA_KERNEL_API
   integer(kind=c_int)                       :: THIS_REAL_ELPA_KERNEL

   integer(kind=c_int), intent(in)           :: na, nev, lda, ldq, matrixCols, mpi_comm_rows, &
                                                mpi_comm_cols, mpi_comm_all
   integer(kind=c_int), intent(in)           :: nblk
   real(kind=c_float), intent(inout)         :: ev(na)
#ifdef USE_ASSUMED_SIZE
   real(kind=c_float), intent(inout)         :: a(lda,*),  q(ldq,*)

#else
   real(kind=c_float), intent(inout)         :: a(lda,matrixCols),  q(ldq,matrixCols)
#endif
   real(kind=c_float), allocatable           :: hh_trans_real(:,:)

   integer(kind=c_int)                       :: my_pe, n_pes, my_prow, my_pcol, np_rows, np_cols, mpierr
   integer(kind=c_int)                       :: nbw, num_blocks
   real(kind=c_float), allocatable           :: tmat(:,:,:), e(:)
   integer(kind=c_intptr_t)                  :: tmat_dev, q_dev, a_dev
   real(kind=c_double)                       :: ttt0, ttt1, ttts  ! MPI_WTIME always needs double
   integer(kind=c_int)                       :: i
   logical                                   :: success
   logical, save                             :: firstCall = .true.
   logical                                   :: wantDebug
   integer(kind=c_int)                       :: istat
   character(200)                            :: errorMessage
   logical                                   :: do_useGPU
   integer(kind=c_int)                       :: numberOfGPUDevices

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("solve_evp_real_2stage_single")
#endif
#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("mpi_communication")
#endif
    call mpi_comm_rank(mpi_comm_all,my_pe,mpierr)
    call mpi_comm_size(mpi_comm_all,n_pes,mpierr)

    call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
    call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
    call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
    call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("mpi_communication")
#endif
    wantDebug = .false.
    if (firstCall) then
      ! are debug messages desired?
      wantDebug = debug_messages_via_environment_variable()
      firstCall = .false.
    endif

    success = .true.

    useQRActual = .false.
    do_useGPU   = .false.

    ! set usage of qr decomposition via API call
    if (present(useQR)) then
      if (useQR) useQRActual = .true.
        if (.not.(useQR)) useQRACtual = .false.
    endif

    ! overwrite this with environment variable settings
    if (qr_decomposition_via_environment_variable(useQREnvironment)) then
      useQRActual = useQREnvironment
    endif

    if (useQRActual) then
      if (mod(na,2) .ne. 0) then
        if (wantDebug) then
          write(error_unit,*) "solve_evp_real_2stage: QR-decomposition: blocksize does not fit with matrixsize"
        endif
        print *, "Do not use QR-decomposition for this matrix and blocksize."
        success = .false.
        return
      endif
    endif

    if (present(useGPU)) then
      if (useGPU) then
        if (check_for_gpu(my_pe,numberOfGPUDevices, wantDebug=wantDebug)) then

           do_useGPU = .true.
           ! set the neccessary parameters
           cudaMemcpyHostToDevice   = cuda_memcpyHostToDevice()
           cudaMemcpyDeviceToHost   = cuda_memcpyDeviceToHost()
           cudaMemcpyDeviceToDevice = cuda_memcpyDeviceToDevice()
           cudaHostRegisterPortable = cuda_hostRegisterPortable()
           cudaHostRegisterMapped   = cuda_hostRegisterMapped()
        else
          write(error_unit,*) "GPUs are requested but not detected! Aborting..."
          success = .false.
          return
        endif
      endif
    else
      ! check whether set by environment variable
      do_useGPU = gpu_usage_via_environment_variable()
   endif


    if (present(THIS_REAL_ELPA_KERNEL_API)) then
      ! user defined kernel via the optional argument in the API call
      THIS_REAL_ELPA_KERNEL = THIS_REAL_ELPA_KERNEL_API
    else

      ! if kernel is not choosen via api
      ! check whether set by environment variable
      THIS_REAL_ELPA_KERNEL = get_actual_real_kernel()
    endif

    ! check whether choosen kernel is allowed
    if (check_allowed_real_kernels(THIS_REAL_ELPA_KERNEL)) then

      if (my_pe == 0) then
        write(error_unit,*) " "
        write(error_unit,*) "The choosen kernel ",REAL_ELPA_KERNEL_NAMES(THIS_REAL_ELPA_KERNEL)
        write(error_unit,*) "is not in the list of the allowed kernels!"
        write(error_unit,*) " "
        write(error_unit,*) "Allowed kernels are:"
        do i=1,size(REAL_ELPA_KERNEL_NAMES(:))
          if (AVAILABLE_REAL_ELPA_KERNELS(i) .ne. 0) then
            write(error_unit,*) REAL_ELPA_KERNEL_NAMES(i)
          endif
        enddo

        write(error_unit,*) " "
        ! check whether generic kernel is defined
         if (AVAILABLE_REAL_ELPA_KERNELS(REAL_ELPA_KERNEL_GENERIC) .eq. 1) then
           write(error_unit,*) "The default kernel REAL_ELPA_KERNEL_GENERIC will be used !"
         else
           write(error_unit,*) "As default kernel ",REAL_ELPA_KERNEL_NAMES(DEFAULT_REAL_ELPA_KERNEL)," will be used"
         endif
      endif  ! my_pe == 0
      if (AVAILABLE_REAL_ELPA_KERNELS(REAL_ELPA_KERNEL_GENERIC) .eq. 1) then
        THIS_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_GENERIC
      else
        THIS_REAL_ELPA_KERNEL = DEFAULT_REAL_ELPA_KERNEL
      endif
    endif

    ! check consistency between request for GPUs and defined kernel
    if (do_useGPU) then
      if (THIS_REAL_ELPA_KERNEL .ne. REAL_ELPA_KERNEL_GPU) then
        write(error_unit,*) "GPU usage has been requested but compute kernel is defined as non-GPU! Aborting..."
        success = .false.
        return
      endif
    endif

    ! Choose bandwidth, must be a multiple of nblk, set to a value >= 32
    ! On older systems (IBM Bluegene/P, Intel Nehalem) a value of 32 was optimal.
    ! For Intel(R) Xeon(R) E5 v2 and v3, better use 64 instead of 32!
    ! For IBM Bluegene/Q this is not clear at the moment. We have to keep an eye
    ! on this and maybe allow a run-time optimization here
    if (do_useGPU) then
      nbw = nblk
    else
      nbw = (63/nblk+1)*nblk
    endif

    num_blocks = (na-1)/nbw + 1

    allocate(tmat(nbw,nbw,num_blocks), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"solve_evp_real_2stage: error when allocating tmat "//errorMessage
      stop
    endif

    ! Reduction full -> band

    ttt0 = MPI_Wtime()
    ttts = ttt0
#ifdef DOUBLE_PRECISION_REAL
    call bandred_real_double(na, a, a_dev, lda, nblk, nbw, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, &
        tmat, tmat_dev, wantDebug, do_useGPU, success, useQRActual)
#else
    call bandred_real_single(na, a, a_dev, lda, nblk, nbw, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, &
        tmat, tmat_dev, wantDebug, do_useGPU, success, useQRActual)
#endif
    if (.not.(success)) return
    ttt1 = MPI_Wtime()
    if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
       write(error_unit,*) 'Time bandred_real               :',ttt1-ttt0

     ! Reduction band -> tridiagonal

     allocate(e(na), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"solve_evp_real_2stage: error when allocating e "//errorMessage
       stop
     endif

     ttt0 = MPI_Wtime()
#ifdef DOUBLE_PRECISION_REAL
     call tridiag_band_real_double(na, nbw, nblk, a, lda, ev, e, matrixCols, hh_trans_real, &
                          mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
#else
     call tridiag_band_real_single(na, nbw, nblk, a, lda, ev, e, matrixCols, hh_trans_real, &
                          mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
#endif

     ttt1 = MPI_Wtime()
     if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
       write(error_unit,*) 'Time tridiag_band_real          :',ttt1-ttt0

#ifdef WITH_MPI
#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("mpi_communication")
#endif
#ifdef DOUBLE_PRECISION_REAL
     call mpi_bcast(ev,na,MPI_REAL8,0,mpi_comm_all,mpierr)
     call mpi_bcast(e,na,MPI_REAL8,0,mpi_comm_all,mpierr)
#else
     call mpi_bcast(ev,na,MPI_REAL4,0,mpi_comm_all,mpierr)
     call mpi_bcast(e,na,MPI_REAL4,0,mpi_comm_all,mpierr)
#endif
#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("mpi_communication")
#endif
#endif /* WITH_MPI */
     ttt1 = MPI_Wtime()
     time_evp_fwd = ttt1-ttts

     ! Solve tridiagonal system

     ttt0 = MPI_Wtime()
#ifdef DOUBLE_PRECISION_REAL
     call solve_tridi_double(na, nev, ev, e, q, ldq, nblk, matrixCols, mpi_comm_rows,  &
                      mpi_comm_cols, wantDebug, success)
#else
     call solve_tridi_single(na, nev, ev, e, q, ldq, nblk, matrixCols, mpi_comm_rows,  &
                      mpi_comm_cols, wantDebug, success)
#endif
     if (.not.(success)) return

     ttt1 = MPI_Wtime()
     if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
     write(error_unit,*) 'Time solve_tridi                :',ttt1-ttt0
     time_evp_solve = ttt1-ttt0
     ttts = ttt1

     deallocate(e, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"solve_evp_real_2stage: error when deallocating e "//errorMessage
       stop
     endif
     ! Backtransform stage 1

     ttt0 = MPI_Wtime()
#ifdef DOUBLE_PRECISION_REAL
     call trans_ev_tridi_to_band_real_double(na, nev, nblk, nbw, q, q_dev, ldq, matrixCols, hh_trans_real, &
                                            mpi_comm_rows, mpi_comm_cols, wantDebug, do_useGPU, success,      &
                                    THIS_REAL_ELPA_KERNEL)
#else
     call trans_ev_tridi_to_band_real_single(na, nev, nblk, nbw, q, q_dev, ldq, matrixCols, hh_trans_real, &
                                             mpi_comm_rows, mpi_comm_cols, wantDebug, do_useGPU, success,      &
                                    THIS_REAL_ELPA_KERNEL)
#endif

     if (.not.(success)) return
     ttt1 = MPI_Wtime()
     if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
       write(error_unit,*) 'Time trans_ev_tridi_to_band_real:',ttt1-ttt0

     ! We can now deallocate the stored householder vectors
     deallocate(hh_trans_real, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"solve_evp_real_2stage: error when deallocating hh_trans_real "//errorMessage
       stop
     endif


     ! Backtransform stage 2
     ttt0 = MPI_Wtime()
#ifdef DOUBLE_PRECISION_REAL
     call trans_ev_band_to_full_real_double(na, nev, nblk, nbw, a, a_dev, lda, tmat, tmat_dev, q, q_dev, ldq, &
                                            matrixCols, num_blocks, mpi_comm_rows, &
                                            mpi_comm_cols, do_useGPU, useQRActual)
#else
     call trans_ev_band_to_full_real_single(na, nev, nblk, nbw, a, a_dev, lda, tmat, tmat_dev, q, q_dev, ldq, &
                                            matrixCols, num_blocks, mpi_comm_rows, &
                                            mpi_comm_cols, do_useGPU, useQRActual)
#endif

     ttt1 = MPI_Wtime()
     if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
       write(error_unit,*) 'Time trans_ev_band_to_full_real :',ttt1-ttt0
     time_evp_back = ttt1-ttts

     deallocate(tmat, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"solve_evp_real_2stage: error when deallocating tmat"//errorMessage
       stop
     endif

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("solve_evp_real_2stage_single")
#endif
1    format(a,f10.3)

#ifdef DOUBLE_PRECISION_REAL
   end function solve_evp_real_2stage_double
#else
   end function solve_evp_real_2stage_single
#endif

#endif /* WANT_SINGLE_PRECISION_REAL */

!>  \brief solve_evp_complex_2stage_double: Fortran function to solve the double-precision complex eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param THIS_REAL_ELPA_KERNEL_API (optional) specify used ELPA2 kernel via API
!>  \param useGPU (optional)                    decide whether GPUs should be used or not
!>
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
#define DOUBLE_PRECISION_COMPLEX 1

#ifdef DOUBLE_PRECISION_COMPLEX
function solve_evp_complex_2stage_double(na, nev, a, lda, ev, q, ldq, nblk, &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols,      &
                                    mpi_comm_all, THIS_COMPLEX_ELPA_KERNEL_API, useGPU) result(success)
#else
function solve_evp_complex_2stage_single(na, nev, a, lda, ev, q, ldq, nblk, &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols,      &
                                    mpi_comm_all, THIS_COMPLEX_ELPA_KERNEL_API, useGPU) result(success)
#endif


#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   use elpa1_utilities, only : gpu_usage_via_environment_variable

   use elpa1_compute
   use elpa2_compute
   use elpa_mpi
   use cuda_functions
   use mod_check_for_gpu
   use iso_c_binding
   implicit none
   logical, intent(in), optional             :: useGPU
   integer(kind=c_int), intent(in), optional :: THIS_COMPLEX_ELPA_KERNEL_API
   integer(kind=c_int)                       :: THIS_COMPLEX_ELPA_KERNEL
   integer(kind=c_int), intent(in)           :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
   real(kind=c_double), intent(inout)        :: ev(na)
#ifdef USE_ASSUMED_SIZE
   complex(kind=c_double), intent(inout)     :: a(lda,*), q(ldq,*)
#else
   complex(kind=c_double), intent(inout)     :: a(lda,matrixCols), q(ldq,matrixCols)
#endif
   complex(kind=c_double), allocatable       :: hh_trans_complex(:,:)

   integer(kind=c_int)                       :: my_prow, my_pcol, np_rows, np_cols, mpierr, my_pe, n_pes
   integer(kind=c_int)                       :: l_cols, l_rows, l_cols_nev, nbw, num_blocks
   complex(kind=c_double), allocatable       :: tmat(:,:,:)
   real(kind=c_double), allocatable          :: q_real(:,:), e(:)
   real(kind=c_double)                       :: ttt0, ttt1, ttts  ! MPI_WTIME always needs double
   integer(kind=c_int)                       :: i

   logical                                   :: success, wantDebug
   logical, save                             :: firstCall = .true.
   integer(kind=c_int)                       :: istat
   character(200)                            :: errorMessage
   logical                                   :: do_useGPU
   integer(kind=c_int)                       :: numberOfGPUDevices

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("solve_evp_complex_2stage_double")
#endif
#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("mpi_communication")
#endif
    call mpi_comm_rank(mpi_comm_all,my_pe,mpierr)
    call mpi_comm_size(mpi_comm_all,n_pes,mpierr)

    call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
    call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
    call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
    call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("mpi_communication")
#endif

    do_useGPU = .false.
    wantDebug = .false.
    if (firstCall) then
      ! are debug messages desired?
      wantDebug = debug_messages_via_environment_variable()
      firstCall = .false.
    endif

    success = .true.

    if (present(useGPU)) then
      if (useGPU) then
        if (check_for_gpu(my_pe,numberOfGPUDevices, wantDebug=wantDebug)) then

           do_useGPU = .true.
           ! set the neccessary parameters
           cudaMemcpyHostToDevice   = cuda_memcpyHostToDevice()
           cudaMemcpyDeviceToHost   = cuda_memcpyDeviceToHost()
           cudaMemcpyDeviceToDevice = cuda_memcpyDeviceToDevice()
           cudaHostRegisterPortable = cuda_hostRegisterPortable()
           cudaHostRegisterMapped   = cuda_hostRegisterMapped()
        else
          print *,"GPUs are requested but not detected! Aborting..."
          success = .false.
          return
        endif
      endif
    else
      ! check whether set by environment variable
      do_useGPU = gpu_usage_via_environment_variable()
    endif

    if (present(THIS_COMPLEX_ELPA_KERNEL_API)) then
      ! user defined kernel via the optional argument in the API call
      THIS_COMPLEX_ELPA_KERNEL = THIS_COMPLEX_ELPA_KERNEL_API
    else
      ! if kernel is not choosen via api
      ! check whether set by environment variable
      THIS_COMPLEX_ELPA_KERNEL = get_actual_complex_kernel()
    endif

    ! check whether choosen kernel is allowed
    if (check_allowed_complex_kernels(THIS_COMPLEX_ELPA_KERNEL)) then

      if (my_pe == 0) then
        write(error_unit,*) " "
        write(error_unit,*) "The choosen kernel ",COMPLEX_ELPA_KERNEL_NAMES(THIS_COMPLEX_ELPA_KERNEL)
        write(error_unit,*) "is not in the list of the allowed kernels!"
        write(error_unit,*) " "
        write(error_unit,*) "Allowed kernels are:"
        do i=1,size(COMPLEX_ELPA_KERNEL_NAMES(:))
          if (AVAILABLE_COMPLEX_ELPA_KERNELS(i) .ne. 0) then
            write(error_unit,*) COMPLEX_ELPA_KERNEL_NAMES(i)
          endif
        enddo

        write(error_unit,*) " "
        write(error_unit,*) "The defaul kernel COMPLEX_ELPA_KERNEL_GENERIC will be used !"
      endif
      THIS_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_GENERIC
    endif

    ! check consistency between request for GPUs and defined kernel
    if (do_useGPU) then
      if (THIS_COMPLEX_ELPA_KERNEL .ne. COMPLEX_ELPA_KERNEL_GPU) then
        write(error_unit,*) "GPU usage has been requested but compute kernel is defined as non-GPU! Aborting..."
        success = .false.
        return
      endif
    endif

    ! Choose bandwidth, must be a multiple of nblk, set to a value >= 32

    nbw = (31/nblk+1)*nblk

    num_blocks = (na-1)/nbw + 1

    allocate(tmat(nbw,nbw,num_blocks), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"solve_evp_complex_2stage: error when allocating tmat"//errorMessage
      stop
    endif
    ! Reduction full -> band

    ttt0 = MPI_Wtime()
    ttts = ttt0
#ifdef DOUBLE_PRECISION_COMPLEX
    call bandred_complex_double(na, a, lda, nblk, nbw, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, &
        tmat, wantDebug, do_useGPU, success)
#else
    call bandred_complex_single(na, a, lda, nblk, nbw, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, &
        tmat, wantDebug, do_useGPU, success)
#endif
    if (.not.(success)) then

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("solve_evp_complex_2stage_double")
#endif
      return
    endif
    ttt1 = MPI_Wtime()
    if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
       write(error_unit,*) 'Time bandred_complex               :',ttt1-ttt0

    ! Reduction band -> tridiagonal

    allocate(e(na), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"solve_evp_complex_2stage: error when allocating e"//errorMessage
      stop
    endif


    ttt0 = MPI_Wtime()
#ifdef DOUBLE_PRECISION_COMPLEX
   call tridiag_band_complex_double(na, nbw, nblk, a, lda, ev, e, matrixCols, hh_trans_complex, &
                             mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
#else
   call tridiag_band_complex_single(na, nbw, nblk, a, lda, ev, e, matrixCols, hh_trans_complex, &
                             mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
#endif

    ttt1 = MPI_Wtime()
    if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
       write(error_unit,*) 'Time tridiag_band_complex          :',ttt1-ttt0

#ifdef WITH_MPI
#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("mpi_communication")
#endif
#ifdef DOUBLE_PRECISION_COMPLEX
    call mpi_bcast(ev, na, mpi_real8, 0, mpi_comm_all, mpierr)
    call mpi_bcast(e, na, mpi_real8, 0, mpi_comm_all, mpierr)
#else
    call mpi_bcast(ev, na, mpi_real4, 0, mpi_comm_all, mpierr)
    call mpi_bcast(e, na, mpi_real4, 0, mpi_comm_all, mpierr)
#endif
#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("mpi_communication")
#endif
#endif /* WITH_MPI */
    ttt1 = MPI_Wtime()
    time_evp_fwd = ttt1-ttts

    l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a and q
    l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local columns of q
    l_cols_nev = local_index(nev, my_pcol, np_cols, nblk, -1) ! Local columns corresponding to nev

    allocate(q_real(l_rows,l_cols), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"solve_evp_complex_2stage: error when allocating q_real"//errorMessage
      stop
    endif

    ! Solve tridiagonal system

    ttt0 = MPI_Wtime()
#ifdef DOUBLE_PRECISION_COMPLEX
    call solve_tridi_double(na, nev, ev, e, q_real, ubound(q_real,dim=1), nblk, matrixCols, &
                     mpi_comm_rows, mpi_comm_cols, wantDebug, success)
#else
    call solve_tridi_single(na, nev, ev, e, q_real, ubound(q_real,dim=1), nblk, matrixCols, &
                     mpi_comm_rows, mpi_comm_cols, wantDebug, success)
#endif
    if (.not.(success)) return

    ttt1 = MPI_Wtime()
    if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times)  &
       write(error_unit,*) 'Time solve_tridi                   :',ttt1-ttt0
    time_evp_solve = ttt1-ttt0
    ttts = ttt1

    q(1:l_rows,1:l_cols_nev) = q_real(1:l_rows,1:l_cols_nev)

    deallocate(e, q_real, stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"solve_evp_complex_2stage: error when deallocating e, q_real"//errorMessage
      stop
    endif


    ! Backtransform stage 1

    ttt0 = MPI_Wtime()
#ifdef DOUBLE_PRECISION_COMPLEX
    call trans_ev_tridi_to_band_complex_double(na, nev, nblk, nbw, q, ldq,  &
                                       matrixCols, hh_trans_complex, &
                                       mpi_comm_rows, mpi_comm_cols, &
                                       wantDebug, do_useGPU, success,THIS_COMPLEX_ELPA_KERNEL)
#else
    call trans_ev_tridi_to_band_complex_single(na, nev, nblk, nbw, q, ldq,  &
                                       matrixCols, hh_trans_complex, &
                                       mpi_comm_rows, mpi_comm_cols, &
                                       wantDebug, do_useGPU, success,THIS_COMPLEX_ELPA_KERNEL)
#endif
    if (.not.(success)) return
    ttt1 = MPI_Wtime()
    if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
       write(error_unit,*) 'Time trans_ev_tridi_to_band_complex:',ttt1-ttt0

    ! We can now deallocate the stored householder vectors
    deallocate(hh_trans_complex, stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"solve_evp_complex_2stage: error when deallocating hh_trans_complex"//errorMessage
      stop
    endif

    ! Backtransform stage 2

    ttt0 = MPI_Wtime()
#ifdef DOUBLE_PRECISION_COMPLEX
   call trans_ev_band_to_full_complex_double(na, nev, nblk, nbw, a, lda, tmat, q, ldq, matrixCols, num_blocks, &
       mpi_comm_rows, mpi_comm_cols, do_useGPU)
#else
   call trans_ev_band_to_full_complex_single(na, nev, nblk, nbw, a, lda, tmat, q, ldq, matrixCols, num_blocks, &
       mpi_comm_rows, mpi_comm_cols, do_useGPU)
#endif
    ttt1 = MPI_Wtime()
    if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
       write(error_unit,*) 'Time trans_ev_band_to_full_complex :',ttt1-ttt0
    time_evp_back = ttt1-ttts

    deallocate(tmat, stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"solve_evp_complex_2stage: error when deallocating tmat "//errorMessage
      stop
    endif

#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("solve_evp_complex_2stage_double")
#endif

1   format(a,f10.3)
#ifdef DOUBLE_PRECISION_COMPLEX
end function solve_evp_complex_2stage_double
#else
end function solve_evp_complex_2stage_single
#endif

#ifdef WANT_SINGLE_PRECISION_COMPLEX
#undef DOUBLE_PRECISION_COMPLEX

!>  \brief solve_evp_complex_2stage_single: Fortran function to solve the single-precision complex eigenvalue problem with a 2 stage approach
!>
!>  Parameters
!>
!>  \param na                                   Order of matrix a
!>
!>  \param nev                                  Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                    Distributed matrix for which eigenvalues are to be computed.
!>                                              Distribution is like in Scalapack.
!>                                              The full matrix must be set (not only one half like in scalapack).
!>                                              Destroyed on exit (upper and lower half).
!>
!>  \param lda                                  Leading dimension of a
!>
!>  \param ev(na)                               On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                    On output: Eigenvectors of a
!>                                              Distribution is like in Scalapack.
!>                                              Must be always dimensioned to the full size (corresponding to (na,na))
!>                                              even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                  Leading dimension of q
!>
!>  \param nblk                                 blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                           local columns of matrix a and q
!>
!>  \param mpi_comm_rows                        MPI communicator for rows
!>  \param mpi_comm_cols                        MPI communicator for columns
!>  \param mpi_comm_all                         MPI communicator for the total processor set
!>
!>  \param THIS_COMPLEX_ELPA_KERNEL_API (optional) specify used ELPA2 kernel via API
!>  \param useGPU (optional)                   decide whether GPUs should be used or not
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------

#ifdef DOUBLE_PRECISION_COMPLEX
function solve_evp_complex_2stage_double(na, nev, a, lda, ev, q, ldq, nblk, &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols,      &
                                    mpi_comm_all, THIS_COMPLEX_ELPA_KERNEL_API, useGPU) result(success)
#else
function solve_evp_complex_2stage_single(na, nev, a, lda, ev, q, ldq, nblk, &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols,      &
                                    mpi_comm_all, THIS_COMPLEX_ELPA_KERNEL_API, useGPU) result(success)
#endif


#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   use elpa1_utilities, only : gpu_usage_via_environment_variable

   use cuda_functions
   use mod_check_for_gpu
   use elpa1_compute
   use elpa2_compute
   use elpa_mpi
   use iso_c_binding
   implicit none
   logical, intent(in), optional             :: useGPU
   integer(kind=c_int), intent(in), optional :: THIS_COMPLEX_ELPA_KERNEL_API
   integer(kind=c_int)                       :: THIS_COMPLEX_ELPA_KERNEL
   integer(kind=c_int), intent(in)           :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
#ifdef USE_ASSUMED_SIZE
   complex(kind=c_float), intent(inout)      :: a(lda,*), q(ldq,*)
#else
   complex(kind=c_float), intent(inout)      :: a(lda,matrixCols), q(ldq,matrixCols)
#endif
   real(kind=c_float), intent(inout)         :: ev(na)
   complex(kind=c_float), allocatable        :: hh_trans_complex(:,:)

   integer(kind=c_int)                       :: my_prow, my_pcol, np_rows, np_cols, mpierr, my_pe, n_pes
   integer(kind=c_int)                       :: l_cols, l_rows, l_cols_nev, nbw, num_blocks
   complex(kind=c_float), allocatable        :: tmat(:,:,:)
   real(kind=c_float), allocatable           :: q_real(:,:), e(:)
   real(kind=c_double)                       :: ttt0, ttt1, ttts  ! MPI_WTIME always needs double
   integer(kind=c_int)                       :: i

   logical                                   :: success, wantDebug
   logical, save                             :: firstCall = .true.
   integer(kind=c_int)                       :: istat
   character(200)                            :: errorMessage
   logical                                   :: do_useGPU
   integer(kind=c_int)                       :: numberOfGPUDevices

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("solve_evp_complex_2stage_single")
#endif
#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("mpi_communication")
#endif
    call mpi_comm_rank(mpi_comm_all,my_pe,mpierr)
    call mpi_comm_size(mpi_comm_all,n_pes,mpierr)

    call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
    call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
    call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
    call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("mpi_communication")
#endif

    do_useGPU = .false.
    wantDebug = .false.
    if (firstCall) then
      ! are debug messages desired?
      wantDebug = debug_messages_via_environment_variable()
      firstCall = .false.
    endif


    success = .true.

    if (present(useGPU)) then
      if (useGPU) then
        if (check_for_gpu(my_pe,numberOfGPUDevices, wantDebug=wantDebug)) then

           do_useGPU = .true.
           ! set the neccessary parameters
           cudaMemcpyHostToDevice   = cuda_memcpyHostToDevice()
           cudaMemcpyDeviceToHost   = cuda_memcpyDeviceToHost()
           cudaMemcpyDeviceToDevice = cuda_memcpyDeviceToDevice()
           cudaHostRegisterPortable = cuda_hostRegisterPortable()
           cudaHostRegisterMapped   = cuda_hostRegisterMapped()
        else
          print *,"GPUs are requested but not detected! Aborting..."
          success = .false.
          return
        endif
      endif
    else
      ! check whether set by environment variable
      do_useGPU = gpu_usage_via_environment_variable()
   endif

    if (present(THIS_COMPLEX_ELPA_KERNEL_API)) then
      ! user defined kernel via the optional argument in the API call
      THIS_COMPLEX_ELPA_KERNEL = THIS_COMPLEX_ELPA_KERNEL_API
    else
      ! if kernel is not choosen via api
      ! check whether set by environment variable
      THIS_COMPLEX_ELPA_KERNEL = get_actual_complex_kernel()
    endif

    ! check whether choosen kernel is allowed
    if (check_allowed_complex_kernels(THIS_COMPLEX_ELPA_KERNEL)) then

      if (my_pe == 0) then
        write(error_unit,*) " "
        write(error_unit,*) "The choosen kernel ",COMPLEX_ELPA_KERNEL_NAMES(THIS_COMPLEX_ELPA_KERNEL)
        write(error_unit,*) "is not in the list of the allowed kernels!"
        write(error_unit,*) " "
        write(error_unit,*) "Allowed kernels are:"
        do i=1,size(COMPLEX_ELPA_KERNEL_NAMES(:))
          if (AVAILABLE_COMPLEX_ELPA_KERNELS(i) .ne. 0) then
            write(error_unit,*) COMPLEX_ELPA_KERNEL_NAMES(i)
          endif
        enddo

        write(error_unit,*) " "
        write(error_unit,*) "The defaul kernel COMPLEX_ELPA_KERNEL_GENERIC will be used !"
      endif
      THIS_COMPLEX_ELPA_KERNEL = COMPLEX_ELPA_KERNEL_GENERIC
    endif

    ! check consistency between request for GPUs and defined kernel
    if (do_useGPU) then
      if (THIS_COMPLEX_ELPA_KERNEL .ne. COMPLEX_ELPA_KERNEL_GPU) then
        write(error_unit,*) "GPU usage has been requested but compute kernel is defined as non-GPU! Aborting..."
        success = .false.
        return
      endif
    endif
    ! Choose bandwidth, must be a multiple of nblk, set to a value >= 32

    nbw = (31/nblk+1)*nblk

    num_blocks = (na-1)/nbw + 1

    allocate(tmat(nbw,nbw,num_blocks), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"solve_evp_complex_2stage: error when allocating tmat"//errorMessage
      stop
    endif
    ! Reduction full -> band

    ttt0 = MPI_Wtime()
    ttts = ttt0
#ifdef DOUBLE_PRECISION_COMPLEX
    call bandred_complex_double(na, a, lda, nblk, nbw, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, &
        tmat, wantDebug, do_useGPU, success)
#else
    call bandred_complex_single(na, a, lda, nblk, nbw, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, &
        tmat, wantDebug, do_useGPU, success)
#endif
    if (.not.(success)) then

#ifdef HAVE_DETAILED_TIMINGS
      call timer%stop("solve_evp_complex_2stage_single")
#endif
      return
    endif
    ttt1 = MPI_Wtime()
    if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
       write(error_unit,*) 'Time bandred_complex               :',ttt1-ttt0

    ! Reduction band -> tridiagonal

    allocate(e(na), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"solve_evp_complex_2stage: error when allocating e"//errorMessage
      stop
    endif


    ttt0 = MPI_Wtime()
#ifdef DOUBLE_PRECISION_COMPLEX
   call tridiag_band_complex_double(na, nbw, nblk, a, lda, ev, e, matrixCols, hh_trans_complex, &
                             mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
#else
   call tridiag_band_complex_single(na, nbw, nblk, a, lda, ev, e, matrixCols, hh_trans_complex, &
                             mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
#endif

    ttt1 = MPI_Wtime()
    if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
       write(error_unit,*) 'Time tridiag_band_complex          :',ttt1-ttt0

#ifdef WITH_MPI
#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("mpi_communication")
#endif
#ifdef DOUBLE_PRECISION_COMPLEX
    call mpi_bcast(ev, na, mpi_real8, 0, mpi_comm_all, mpierr)
    call mpi_bcast(e, na, mpi_real8, 0, mpi_comm_all, mpierr)
#else
    call mpi_bcast(ev, na, mpi_real4, 0, mpi_comm_all, mpierr)
    call mpi_bcast(e, na, mpi_real4, 0, mpi_comm_all, mpierr)
#endif
#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("mpi_communication")
#endif
#endif /* WITH_MPI */
    ttt1 = MPI_Wtime()
    time_evp_fwd = ttt1-ttts

    l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a and q
    l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local columns of q
    l_cols_nev = local_index(nev, my_pcol, np_cols, nblk, -1) ! Local columns corresponding to nev

    allocate(q_real(l_rows,l_cols), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"solve_evp_complex_2stage: error when allocating q_real"//errorMessage
      stop
    endif

    ! Solve tridiagonal system

    ttt0 = MPI_Wtime()
#ifdef DOUBLE_PRECISION_COMPLEX
    call solve_tridi_double(na, nev, ev, e, q_real, ubound(q_real,dim=1), nblk, matrixCols, &
                     mpi_comm_rows, mpi_comm_cols, wantDebug, success)
#else
    call solve_tridi_single(na, nev, ev, e, q_real, ubound(q_real,dim=1), nblk, matrixCols, &
                     mpi_comm_rows, mpi_comm_cols, wantDebug, success)
#endif
    if (.not.(success)) return

    ttt1 = MPI_Wtime()
    if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times)  &
       write(error_unit,*) 'Time solve_tridi                   :',ttt1-ttt0
    time_evp_solve = ttt1-ttt0
    ttts = ttt1

    q(1:l_rows,1:l_cols_nev) = q_real(1:l_rows,1:l_cols_nev)

    deallocate(e, q_real, stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"solve_evp_complex_2stage: error when deallocating e, q_real"//errorMessage
      stop
    endif


    ! Backtransform stage 1

    ttt0 = MPI_Wtime()
#ifdef DOUBLE_PRECISION_COMPLEX
    call trans_ev_tridi_to_band_complex_double(na, nev, nblk, nbw, q, ldq,  &
                                       matrixCols, hh_trans_complex, &
                                       mpi_comm_rows, mpi_comm_cols, &
                                       wantDebug, do_useGPU, success,THIS_COMPLEX_ELPA_KERNEL)
#else
    call trans_ev_tridi_to_band_complex_single(na, nev, nblk, nbw, q, ldq,  &
                                       matrixCols, hh_trans_complex, &
                                       mpi_comm_rows, mpi_comm_cols, &
                                       wantDebug, do_useGPU, success,THIS_COMPLEX_ELPA_KERNEL)
#endif
    if (.not.(success)) return
    ttt1 = MPI_Wtime()
    if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
       write(error_unit,*) 'Time trans_ev_tridi_to_band_complex:',ttt1-ttt0

    ! We can now deallocate the stored householder vectors
    deallocate(hh_trans_complex, stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"solve_evp_complex_2stage: error when deallocating hh_trans_complex"//errorMessage
      stop
    endif

    ! Backtransform stage 2

    ttt0 = MPI_Wtime()
#ifdef DOUBLE_PRECISION_COMPLEX
   call trans_ev_band_to_full_complex_double(na, nev, nblk, nbw, a, lda, tmat, q, ldq, matrixCols, num_blocks, &
       mpi_comm_rows, mpi_comm_cols, do_useGPU)
#else
   call trans_ev_band_to_full_complex_single(na, nev, nblk, nbw, a, lda, tmat, q, ldq, matrixCols, num_blocks, &
       mpi_comm_rows, mpi_comm_cols, do_useGPU)
#endif
    ttt1 = MPI_Wtime()
    if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
       write(error_unit,*) 'Time trans_ev_band_to_full_complex :',ttt1-ttt0
    time_evp_back = ttt1-ttts

    deallocate(tmat, stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"solve_evp_complex_2stage: error when deallocating tmat "//errorMessage
      stop
    endif

#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("solve_evp_complex_2stage_single")
#endif

1   format(a,f10.3)
#ifdef DOUBLE_PRECISION_COMPLEX
end function solve_evp_complex_2stage_double
#else
end function solve_evp_complex_2stage_single
#endif

#endif /* WANT_SINGLE_PRECISION_COMPLEX */

end module ELPA2
