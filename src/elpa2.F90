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
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaftrn,
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!      and
!    - IBM Deutschland GmbH
!
!
!    More information can be found here:
!    http://elpa.rzg.mpg.de/
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

module ELPA2

! Version 1.1.2, 2011-02-21

  use elpa_utilities
  USE ELPA1
  use elpa2_utilities
  use elpa_pdgeqrf

#ifdef WITH_GPU_VERSION
  use cuda_routines
  use cuda_c_kernel
  use iso_c_binding
#endif
  implicit none

  PRIVATE ! By default, all routines contained are private

  ! The following routines are public:

  public :: solve_evp_real_2stage
  public :: solve_evp_complex_2stage

  public :: bandred_real
  public :: tridiag_band_real
  public :: trans_ev_tridi_to_band_real
  public :: trans_ev_band_to_full_real

  public :: bandred_complex
  public :: tridiag_band_complex
  public :: trans_ev_tridi_to_band_complex
  public :: trans_ev_band_to_full_complex

  public :: band_band_real
  public :: divide_band

  integer, public :: which_qr_decomposition = 1     ! defines, which QR-decomposition algorithm will be used
                                                    ! 0 for unblocked
                                                    ! 1 for blocked (maxrank: nblk)
!-------------------------------------------------------------------------------

  ! The following array contains the Householder vectors of the
  ! transformation band -> tridiagonal.
  ! It is allocated and set in tridiag_band_real and used in
  ! trans_ev_tridi_to_band_real.
  ! It must be deallocated by the user after trans_ev_tridi_to_band_real!

  real*8, allocatable :: hh_trans_real(:,:)
  complex*16, allocatable :: hh_trans_complex(:,:)

!-------------------------------------------------------------------------------

  include 'mpif.h'


!******
contains

function solve_evp_real_2stage(na, nev, a, lda, ev, q, ldq, nblk,        &
                               matrixCols,                               &
                                 mpi_comm_rows, mpi_comm_cols,           &
                                 mpi_comm_all, THIS_REAL_ELPA_KERNEL_API,&
                                 useQR) result(success)

!-------------------------------------------------------------------------------
!  solve_evp_real_2stage: Solves the real eigenvalue problem with a 2 stage approach
!
!  Parameters
!
!  na          Order of matrix a
!
!  nev         Number of eigenvalues needed
!
!  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!              Distribution is like in Scalapack.
!              The full matrix must be set (not only one half like in scalapack).
!              Destroyed on exit (upper and lower half).
!
!  lda         Leading dimension of a
!  matrixCols  local columns of matrix a and q
!
!  ev(na)      On output: eigenvalues of a, every processor gets the complete set
!
!  q(ldq,matrixCols)    On output: Eigenvectors of a
!              Distribution is like in Scalapack.
!              Must be always dimensioned to the full size (corresponding to (na,na))
!              even if only a part of the eigenvalues is needed.
!
!  ldq         Leading dimension of q
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!  mpi_comm_all
!              MPI-Communicator for the total processor set
!
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
   implicit none
   logical, intent(in), optional :: useQR
   logical                       :: useQRActual, useQREnvironment
   integer, intent(in), optional :: THIS_REAL_ELPA_KERNEL_API
   integer                       :: THIS_REAL_ELPA_KERNEL

   integer, intent(in)           :: na, nev, lda, ldq, matrixCols, mpi_comm_rows, &
                                    mpi_comm_cols, mpi_comm_all
   integer, intent(in)           :: nblk
   real*8, intent(inout)         :: a(lda,matrixCols), ev(na), q(ldq,matrixCols)

   integer                       :: my_pe, n_pes, my_prow, my_pcol, np_rows, np_cols, mpierr
   integer                       :: nbw, num_blocks
   real*8, allocatable           :: tmat(:,:,:), e(:)
   real*8                        :: ttt0, ttt1, ttts
   integer                       :: i
   logical                       :: success
   logical, save                 :: firstCall = .true.
   logical                       :: wantDebug
   integer                       :: istat
   character(200)                :: errorMessage

#ifdef WITH_GPU_VERSION
   if (nblk .ne. 128) then
     print *,"At the moment GPU version needs blocksize 128"
     stop
   endif
#endif

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("solve_evp_real_2stage")
#endif
   call mpi_comm_rank(mpi_comm_all,my_pe,mpierr)
   call mpi_comm_size(mpi_comm_all,n_pes,mpierr)

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)


   wantDebug = .false.
   if (firstCall) then
     ! are debug messages desired?
     wantDebug = debug_messages_via_environment_variable()
     firstCall = .false.
   endif

   success = .true.

   useQRActual = .false.

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
     if (mod(na,nblk) .ne. 0) then
       if (wantDebug) then
         write(error_unit,*) "solve_evp_real_2stage: QR-decomposition: blocksize does not fit with matrixsize"
       endif
     print *, "Do not use QR-decomposition for this matrix and blocksize."
     success = .false.
     return
     endif
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
       write(error_unit,*) "The defaul kernel REAL_ELPA_KERNEL_GENERIC will be used !"
     endif
     THIS_REAL_ELPA_KERNEL = REAL_ELPA_KERNEL_GENERIC

   endif

   ! Choose bandwidth, must be a multiple of nblk, set to a value >= 32

#ifdef WITH_GPU_VERSION
   nbw = nblk
#else
   nbw = (31/nblk+1)*nblk
#endif
   num_blocks = (na-1)/nbw + 1

   allocate(tmat(nbw,nbw,num_blocks), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"solve_evp_real_2stage: error when allocating tmat "//errorMessage
     stop
   endif

   ! Reduction full -> band

   ttt0 = MPI_Wtime()
   ttts = ttt0
   call bandred_real(na, a, lda, nblk, nbw, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, &
                     tmat, wantDebug, success, useQRActual)
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
   call tridiag_band_real(na, nbw, nblk, a, lda, ev, e, matrixCols, mpi_comm_rows, &
                          mpi_comm_cols, mpi_comm_all)
   ttt1 = MPI_Wtime()
   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      write(error_unit,*) 'Time tridiag_band_real          :',ttt1-ttt0

   call mpi_bcast(ev,na,MPI_REAL8,0,mpi_comm_all,mpierr)
   call mpi_bcast(e,na,MPI_REAL8,0,mpi_comm_all,mpierr)

   ttt1 = MPI_Wtime()
   time_evp_fwd = ttt1-ttts

   ! Solve tridiagonal system

   ttt0 = MPI_Wtime()
   call solve_tridi(na, nev, ev, e, q, ldq, nblk, matrixCols, mpi_comm_rows,  &
                    mpi_comm_cols, wantDebug, success)
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
   call trans_ev_tridi_to_band_real(na, nev, nblk, nbw, q, ldq, matrixCols, mpi_comm_rows, &
                                    mpi_comm_cols, wantDebug, success, THIS_REAL_ELPA_KERNEL)
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
   call trans_ev_band_to_full_real(na, nev, nblk, nbw, a, lda, tmat, q, ldq, matrixCols, num_blocks, mpi_comm_rows, &
                                   mpi_comm_cols, useQRActual)
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
   call timer%stop("solve_evp_real_2stage")
#endif
1  format(a,f10.3)

end function solve_evp_real_2stage

!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------

function solve_evp_complex_2stage(na, nev, a, lda, ev, q, ldq, nblk, &
                                  matrixCols, mpi_comm_rows, mpi_comm_cols,      &
                                    mpi_comm_all, THIS_COMPLEX_ELPA_KERNEL_API) result(success)

!-------------------------------------------------------------------------------
!  solve_evp_complex_2stage: Solves the complex eigenvalue problem with a 2 stage approach
!
!  Parameters
!
!  na          Order of matrix a
!
!  nev         Number of eigenvalues needed
!
!  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!              Distribution is like in Scalapack.
!              The full matrix must be set (not only one half like in scalapack).
!              Destroyed on exit (upper and lower half).
!
!  lda         Leading dimension of a
!  matrixCols  local columns of matrix a and q
!
!  ev(na)      On output: eigenvalues of a, every processor gets the complete set
!
!  q(ldq,matrixCols)    On output: Eigenvectors of a
!              Distribution is like in Scalapack.
!              Must be always dimensioned to the full size (corresponding to (na,na))
!              even if only a part of the eigenvalues is needed.
!
!  ldq         Leading dimension of q
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!  mpi_comm_all
!              MPI-Communicator for the total processor set
!
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
   implicit none
   integer, intent(in), optional :: THIS_COMPLEX_ELPA_KERNEL_API
   integer                       :: THIS_COMPLEX_ELPA_KERNEL
   integer, intent(in)           :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all
   complex*16, intent(inout)     :: a(lda,matrixCols), q(ldq,matrixCols)
   real*8, intent(inout)         :: ev(na)

   integer                       :: my_prow, my_pcol, np_rows, np_cols, mpierr, my_pe, n_pes
   integer                       :: l_cols, l_rows, l_cols_nev, nbw, num_blocks
   complex*16, allocatable       :: tmat(:,:,:)
   real*8, allocatable           :: q_real(:,:), e(:)
   real*8                        :: ttt0, ttt1, ttts
   integer                       :: i

   logical                       :: success, wantDebug
   logical, save                 :: firstCall = .true.
   integer                       :: istat
   character(200)                :: errorMessage


#ifdef WITH_GPU_VERSION
   if (nblk .ne. 128) then
     print *,"At the moment GPU version needs blocksize 128"
     stop
   endif
#endif

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("solve_evp_complex_2stage")
#endif
   call mpi_comm_rank(mpi_comm_all,my_pe,mpierr)
   call mpi_comm_size(mpi_comm_all,n_pes,mpierr)

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   wantDebug = .false.
   if (firstCall) then
     ! are debug messages desired?
     wantDebug = debug_messages_via_environment_variable()
     firstCall = .false.
   endif


   success = .true.

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
   call bandred_complex(na, a, lda, nblk, nbw, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols, &
                        tmat, wantDebug, success)
   if (.not.(success)) then
#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop()
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
   call tridiag_band_complex(na, nbw, nblk, a, lda, ev, e, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm_all)
   ttt1 = MPI_Wtime()
   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
      write(error_unit,*) 'Time tridiag_band_complex          :',ttt1-ttt0

   call mpi_bcast(ev,na,MPI_REAL8,0,mpi_comm_all,mpierr)
   call mpi_bcast(e,na,MPI_REAL8,0,mpi_comm_all,mpierr)

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
   call solve_tridi(na, nev, ev, e, q_real, ubound(q_real,dim=1), nblk, matrixCols, &
                    mpi_comm_rows, mpi_comm_cols, wantDebug, success)
   if (.not.(success)) return

   ttt1 = MPI_Wtime()
   if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times)  &
      write(error_unit,*) 'Time solve_tridi                   :',ttt1-ttt0
   time_evp_solve = ttt1-ttt0
   ttts = ttt1

   q(1:l_rows,1:l_cols_nev) = q_real(1:l_rows,1:l_cols_nev)

   deallocate(e, q_real, stat=istat, errmsg=errorMessage))
   if (istat .ne. 0) then
     print *,"solve_evp_complex_2stage: error when deallocating e, q_real"//errorMessage
     stop
   endif


   ! Backtransform stage 1

   ttt0 = MPI_Wtime()
   call trans_ev_tridi_to_band_complex(na, nev, nblk, nbw, q, ldq,  &
                                       matrixCols, mpi_comm_rows, mpi_comm_cols,&
                                       wantDebug, success,THIS_COMPLEX_ELPA_KERNEL)
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
   call trans_ev_band_to_full_complex(na, nev, nblk, nbw, a, lda, tmat, q, ldq, matrixCols, num_blocks, mpi_comm_rows, mpi_comm_cols)
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
   call timer%stop("solve_evp_complex_2stage")
#endif

1  format(a,f10.3)

end function solve_evp_complex_2stage

!-------------------------------------------------------------------------------


subroutine bandred_real(na, a, lda, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols, &
                        tmat, wantDebug, success, useQR)

!-------------------------------------------------------------------------------
!  bandred_real: Reduces a distributed symmetric matrix to band form
!
!  Parameters
!
!  na          Order of matrix
!
!  a(lda,matrixCols)    Distributed matrix which should be reduced.
!              Distribution is like in Scalapack.
!              Opposed to Scalapack, a(:,:) must be set completely (upper and lower half)
!              a(:,:) is overwritten on exit with the band and the Householder vectors
!              in the upper half.
!
!  lda         Leading dimension of a
!  matrixCols  local columns of matrix a
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  nbw         semi bandwith of output matrix
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!
!  tmat(nbw,nbw,numBlocks)    where numBlocks = (na-1)/nbw + 1
!              Factors for the Householder vectors (returned), needed for back transformation
!
!-------------------------------------------------------------------------------

#ifdef WITH_GPU_VERSION
  use cuda_routines
  use iso_c_binding
#endif

#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif

   implicit none

   integer             :: na, lda, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols
   real*8              :: a(lda,matrixCols), tmat(nbw,nbw,numBlocks)

   integer             :: my_prow, my_pcol, np_rows, np_cols, mpierr
   integer             :: l_cols, l_rows
   integer             :: i, j, lcs, lce, lre, lc, lr, cur_pcol, n_cols, nrow
   integer             :: istep, ncol, lch, lcx, nlc
   integer             :: tile_size, l_rows_tile, l_cols_tile

#ifdef WITH_GPU_VERSION
   real*8              :: eps
#endif

   real*8              :: vnorm2, xf, aux1(nbw), aux2(nbw), vrl, tau, vav(nbw,nbw)

#ifdef WITH_GPU_VERSION
   real*8, allocatable :: tmp(:), vr(:), vmr(:), umc(:)
#else

   real*8, allocatable :: tmp(:,:), vr(:), vmr(:,:), umc(:,:)
#endif

   ! needed for blocked QR decomposition
   integer             :: PQRPARAM(11), work_size
   real*8              :: dwork_size(1)
   real*8, allocatable :: work_blocked(:), tauvector(:), blockheuristic(:)

#ifdef WITH_GPU_VERSION
   integer(C_SIZE_T)   :: a_dev, vmr_dev, umc_dev, tmat_dev, vav_dev
   integer, external   :: numroc
   integer             :: ierr
   integer             :: cur_l_rows, cur_l_cols, vmr_size, umc_size
   integer(C_SIZE_T)   :: lc_start, lc_end
   integer             :: lr_end
   integer             :: na_rows, na_cols
#endif
   logical, intent(in) :: wantDebug
   logical, intent(out):: success

   logical, intent(in) :: useQR
   integer             :: istat
   character(200)      :: errorMessage

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("bandred_real")
#endif
   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)
   success = .true.


   ! Semibandwith nbw must be a multiple of blocksize nblk
   if (mod(nbw,nblk)/=0) then
     if (my_prow==0 .and. my_pcol==0) then
       if (wantDebug) then
         write(error_unit,*) 'ELPA2_bandred_real: ERROR: nbw=',nbw,', nblk=',nblk
         write(error_unit,*) 'ELPA2_bandred_real: ELPA2 works only for nbw==n*nblk'
       endif
       success = .false.
       return
     endif
   endif

#ifdef WITH_GPU_VERSION
   na_rows = numroc(na, nblk, my_prow, 0, np_rows)
   na_cols = numroc(na, nblk, my_pcol, 0, np_cols)
#endif

   ! Matrix is split into tiles; work is done only for tiles on the diagonal or above

   tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size
   tile_size = ((128*max(np_rows,np_cols)-1)/tile_size+1)*tile_size ! make local tiles at least 128 wide

   l_rows_tile = tile_size/np_rows ! local rows of a tile
   l_cols_tile = tile_size/np_cols ! local cols of a tile

   if (useQR) then
#ifdef WITH_GPU_VERSION
     print *,"qr decomposition at the moment not supported with GPU"
     stop
#else
     if (which_qr_decomposition == 1) then
       call qr_pqrparam_init(pqrparam,    nblk,'M',0,   nblk,'M',0,   nblk,'M',1,'s')
       allocate(tauvector(na), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_real: error when allocating tauvector "//errorMessage
         stop
       endif

       allocate(blockheuristic(nblk), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_real: error when allocating blockheuristic "//errorMessage
         stop
       endif

       l_rows = local_index(na, my_prow, np_rows, nblk, -1)
       allocate(vmr(max(l_rows,1),na), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_real: error when allocating vmr "//errorMessage
         stop
       endif

       call qr_pdgeqrf_2dcomm(a, lda, vmr, max(l_rows,1), tauvector, tmat(1,1,1), nbw, dwork_size(1), -1, na, &
                             nbw, nblk, nblk, na, na, 1, 0, PQRPARAM, mpi_comm_rows, mpi_comm_cols, blockheuristic)
       work_size = dwork_size(1)
       allocate(work_blocked(work_size), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_real: error when allocating work_blocked "//errorMessage
         stop
       endif

       work_blocked = 0.0d0
       deallocate(vmr, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_real: error when deallocating vmr "//errorMessage
         stop
       endif

     endif
#endif
   endif ! useQr

#ifdef WITH_GPU_VERSION
   ! Here we convert the regular host array into a pinned host array
   istat = cuda_malloc(a_dev, lda*na_cols*8_8)
   if (istat .ne. 0) then
     print *,"bandred_real: error in cudaMalloc"
     stop
   endif

   istat = cuda_malloc(tmat_dev, nbw*nbw*8_8)
   if (istat .ne. 0) then
     print *,"bandred_real: error in cudaMalloc"
     stop
   endif

   istat = cuda_malloc(vav_dev, nbw*nbw*8_8)
   if (istat .ne. 0) then
     print *,"bandred_real: error in cudaMalloc"
     stop
   endif

   cur_l_rows = 0
   cur_l_cols = 0

   istat = cuda_memcpy(a_dev, loc(a(1,1)), (lda)*(na_cols)*8_8,cudaMemcpyHostToDevice)
   if (istat .ne. 0) then
     print *,"bandred_real: error in cudaMemcpy"
     stop
   endif

#endif

   do istep = (na-1)/nbw, 1, -1

     n_cols = MIN(na,(istep+1)*nbw) - istep*nbw ! Number of columns in current step

     ! Number of local columns/rows of remaining matrix
     l_cols = local_index(istep*nbw, my_pcol, np_cols, nblk, -1)
     l_rows = local_index(istep*nbw, my_prow, np_rows, nblk, -1)

#ifdef WITH_GPU_VERSION
     cur_l_rows = max(l_rows, 1)
     cur_l_cols = max(l_cols, 1)

     vmr_size = cur_l_rows * 2 * n_cols
     umc_size = cur_l_cols * 2 * n_cols

     ! Allocate vmr and umc only if the inew size exceeds their current capacity
     ! Added for FORTRAN CALLS
     if ((.not. allocated(vr)) .or. (l_rows + 1 .gt. ubound(vr, dim=1))) then
       if (allocated(vr)) then
         deallocate(vr, stat=istat, errmsg=errorMessage)
         if (istat .ne. 0) then
           print *,"bandred_real: error when deallocating vr "//errorMessage
           stop
         endif
       endif
       allocate(vr(l_rows + 1), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_real: error when allocating vr "//errorMessage
         stop
       endif

     endif

     if ((.not. allocated(vmr)) .or. (vmr_size .gt. ubound(vmr, dim=1))) then
       if (allocated(vmr)) then
         deallocate(vmr, stat=istat, errmsg=errorMessage)
         if (istat .ne. 0) then
           print *,"bandred_real: error when allocating vmr "//errorMessage
           stop
         endif

         istat = cuda_free(vmr_dev)
         if (istat .ne. 0) then
           print *,"bandred_real: error in cuda_free"
           stop
         endif
       endif

       allocate(vmr(vmr_size), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_real: error when allocating vmr "//errorMessage
         stop
       endif

       istat = cuda_malloc(vmr_dev, vmr_size*8_8)
       if (istat .ne. 0) then
         print *,"bandred_real: error in cudaMalloc"
         stop
       endif

     endif



     if ((.not. allocated(umc)) .or. (umc_size .gt. ubound(umc, dim=1))) then
       if (allocated(umc)) then
         deallocate(umc, stat=istat, errmsg=errorMessage)
         if (istat .ne. 0) then
           print *,"bandred_real: error when deallocating umc "//errorMessage
           stop
         endif

         istat = cuda_free(umc_dev)
         if (istat .ne. 0) then
            print *,"bandred_real: error in cudaFree"
            stop
         endif

       endif

       allocate(umc(umc_size), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_real: error when deallocating umc "//errorMessage
         stop
       endif

       istat = cuda_malloc(umc_dev, umc_size*8_8)
       if (istat .ne. 0) then
         print *,"bandred_real: error in cudaMalloc"
         stop
       endif

     endif
#else
     ! Allocate vmr and umc to their exact sizes so that they can be used in bcasts and reduces

     allocate(vmr(max(l_rows,1),2*n_cols), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"bandred_real: error when allocating vmr "//errorMessage
       stop
     endif

     allocate(umc(max(l_cols,1),2*n_cols), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"bandred_real: error when allocating umc "//errorMessage
       stop
     endif

     allocate(vr(l_rows+1), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"bandred_real: error when allocating vr "//errorMessage
       stop
     endif

#endif

#ifdef WITH_GPU_VERSION
     vmr(1 : cur_l_rows * n_cols) = 0.
#else
     vmr(1:l_rows,1:n_cols) = 0.
#endif
     vr(:) = 0
     tmat(:,:,istep) = 0

#ifdef WITH_GPU_VERSION
     umc(1 : umc_size) = 0.

     lc_start = local_index(istep*nbw+1, my_pcol, np_cols, nblk, -1)
     lc_end   = local_index(istep*nbw+n_cols, my_pcol, np_cols, nblk, -1)
     lr_end   = local_index((istep-1)*nbw + n_cols, my_prow, np_rows, nblk, -1)

     if(lc_start .le. 0) lc_start = 1

     ! Here we assume that the processor grid and the block grid are aligned
     cur_pcol = pcol(istep*nbw+1, nblk, np_cols)

     if(my_pcol == cur_pcol) then

       istat = cuda_memcpy2d(loc(a(1, lc_start)), lda*8_8, (a_dev + ((lc_start-1) * lda*8_8)), lda*8_8, &
                            lr_end*8_8, (lc_end - lc_start+1), cudaMemcpyDeviceToHost)
       if (istat .ne. 0) then
         print *,"error in cudaMemcpy2d"
         stop
       endif

     endif
#endif

     ! Reduce current block to lower triangular form

     if (useQR) then
       if (which_qr_decomposition == 1) then
         call qr_pdgeqrf_2dcomm(a, lda, vmr, max(l_rows,1), tauvector(1), &
                                  tmat(1,1,istep), nbw, work_blocked,       &
                                  work_size, na, n_cols, nblk, nblk,        &
                                  istep*nbw+n_cols-nbw, istep*nbw+n_cols, 1,&
                                  0, PQRPARAM, mpi_comm_rows, mpi_comm_cols,&
                                  blockheuristic)
       endif
     else

       do lc = n_cols, 1, -1

         ncol = istep*nbw + lc ! absolute column number of householder vector
         nrow = ncol - nbw ! Absolute number of pivot row

         lr  = local_index(nrow, my_prow, np_rows, nblk, -1) ! current row length
         lch = local_index(ncol, my_pcol, np_cols, nblk, -1) ! HV local column number

         tau = 0

         if (nrow == 1) exit ! Nothing to do

         cur_pcol = pcol(ncol, nblk, np_cols) ! Processor column owning current block

         if (my_pcol==cur_pcol) then

           ! Get vector to be transformed; distribute last element and norm of
           ! remaining elements to all procs in current column

           vr(1:lr) = a(1:lr,lch) ! vector to be transformed

           if (my_prow==prow(nrow, nblk, np_rows)) then
             aux1(1) = dot_product(vr(1:lr-1),vr(1:lr-1))
             aux1(2) = vr(lr)
           else
             aux1(1) = dot_product(vr(1:lr),vr(1:lr))
             aux1(2) = 0.
           endif

           call mpi_allreduce(aux1,aux2,2,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)

           vnorm2 = aux2(1)
           vrl    = aux2(2)

           ! Householder transformation

           call hh_transform_real(vrl, vnorm2, xf, tau)

           ! Scale vr and store Householder vector for back transformation

           vr(1:lr) = vr(1:lr) * xf
           if (my_prow==prow(nrow, nblk, np_rows)) then
             a(1:lr-1,lch) = vr(1:lr-1)
             a(lr,lch) = vrl
             vr(lr) = 1.
           else
             a(1:lr,lch) = vr(1:lr)
           endif

         endif

         ! Broadcast Householder vector and tau along columns

         vr(lr+1) = tau
         call MPI_Bcast(vr,lr+1,MPI_REAL8,cur_pcol,mpi_comm_cols,mpierr)
#ifdef WITH_GPU_VERSION
         vmr(cur_l_rows * (lc - 1) + 1 : cur_l_rows * (lc - 1) + lr) = vr(1:lr)
#else
         vmr(1:lr,lc) = vr(1:lr)
#endif
         tau = vr(lr+1)
         tmat(lc,lc,istep) = tau ! Store tau in diagonal of tmat

         ! Transform remaining columns in current block with Householder vector
         ! Local dot product

         aux1 = 0

         nlc = 0 ! number of local columns
         do j=1,lc-1
           lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
           if (lcx>0) then
             nlc = nlc+1
             if (lr>0) aux1(nlc) = dot_product(vr(1:lr),a(1:lr,lcx))
           endif
         enddo

         ! Get global dot products
         if (nlc>0) call mpi_allreduce(aux1,aux2,nlc,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)

         ! Transform

         nlc = 0
         do j=1,lc-1
           lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
           if (lcx>0) then
             nlc = nlc+1
             a(1:lr,lcx) = a(1:lr,lcx) - tau*aux2(nlc)*vr(1:lr)
           endif
         enddo

       enddo

#ifdef WITH_GPU_VERSION
      ! store column tiles back to GPU
      cur_pcol = pcol(istep*nbw+1, nblk, np_cols)
      if (my_pcol == cur_pcol) then
        istat = cuda_memcpy2d((a_dev+((lc_start-1)*lda*8_8)), lda*8_8, loc(a(1, lc_start)), lda*8_8,  lr_end*8_8, &
                                  (lc_end - lc_start+1),cudaMemcpyHostToDevice)
        if (istat .ne. 0) then
          print *,"error in cudaMemcpy2d"
          stop
        endif

      endif
#endif
       ! Calculate scalar products of stored Householder vectors.
       ! This can be done in different ways, we use dsyrk

       vav = 0
#ifdef WITH_GPU_VERSION
       if (l_rows>0) &
         call dsyrk('U','T',n_cols,l_rows,1.d0,vmr,cur_l_rows,0.d0,vav,ubound(vav,dim=1))
#else
       if (l_rows>0) &
         call dsyrk('U','T',n_cols,l_rows,1.d0,vmr,ubound(vmr,dim=1),0.d0,vav,ubound(vav,dim=1))

#endif
       call symm_matrix_allreduce(n_cols,vav, nbw, nbw,mpi_comm_rows)

       ! Calculate triangular matrix T for block Householder Transformation

       do lc=n_cols,1,-1
         tau = tmat(lc,lc,istep)
         if (lc<n_cols) then
           call dtrmv('U','T','N',n_cols-lc,tmat(lc+1,lc+1,istep),ubound(tmat,dim=1),vav(lc+1,lc),1)
           tmat(lc,lc+1:n_cols,istep) = -tau * vav(lc+1:n_cols,lc)
         endif
       enddo
     endif

    ! Transpose vmr -> vmc (stored in umc, second half)
#ifdef WITH_GPU_VERSION
    call elpa_transpose_vectors_real  (vmr, cur_l_rows, mpi_comm_rows, &
                                    umc(cur_l_cols * n_cols + 1), cur_l_cols, mpi_comm_cols, &
                                    1, istep*nbw, n_cols, nblk)
#else
    call elpa_transpose_vectors_real  (vmr, ubound(vmr,dim=1), mpi_comm_rows, &
                                    umc(1,n_cols+1), ubound(umc,dim=1), mpi_comm_cols, &
                                    1, istep*nbw, n_cols, nblk)
#endif

    ! Calculate umc = A**T * vmr
    ! Note that the distributed A has to be transposed
    ! Opposed to direct tridiagonalization there is no need to use the cache locality
    ! of the tiles, so we can use strips of the matrix
#ifdef WITH_GPU_VERSION
    umc(1 : l_cols * n_cols) = 0.d0
    vmr(cur_l_rows * n_cols + 1 : cur_l_rows * n_cols * 2) = 0
#else
    umc(1:l_cols,1:n_cols) = 0.d0
    vmr(1:l_rows,n_cols+1:2*n_cols) = 0
#endif
    if (l_cols>0 .and. l_rows>0) then

#ifdef WITH_GPU_VERSION
      istat = cuda_memcpy(vmr_dev, loc(vmr(1)), vmr_size*8_8,cudaMemcpyHostToDevice)
      if (istat .ne. 0) then
        print *,"error in cudaMemcpy"
        stop
      endif

      istat = cuda_memcpy(umc_dev, loc(umc(1)), umc_size*8_8,cudaMemcpyHostToDevice)
      if (istat .ne. 0) then
        print *,"error in cudaMemcpy"
        stop
      endif

#endif
      do i=0,(istep*nbw-1)/tile_size

        lcs = i*l_cols_tile+1
        lce = min(l_cols,(i+1)*l_cols_tile)
        if (lce<lcs) cycle

        lre = min(l_rows,(i+1)*l_rows_tile)

#ifdef WITH_GPU_VERSION
        call cublas_dgemm('T','N',lce-lcs+1,n_cols,lre, &
                        1.d0, (a_dev + ((lcs-1)*lda*8_8)), lda, vmr_dev,cur_l_rows, &
                        1.d0, (umc_dev+ (lcs-1)*8_8), cur_l_cols)

        if(i==0) cycle
        lre = min(l_rows,i*l_rows_tile)

        call cublas_dgemm('N','N',lre,n_cols,lce-lcs+1,&
                   1.d0, (a_dev+ ((lcs-1)*lda*8_8)),lda,(umc_dev+(cur_l_cols * n_cols+lcs-1)*8_8), cur_l_cols, &
                   1.d0, (vmr_dev+(cur_l_rows * n_cols)*8_8), cur_l_rows)
#else
        call DGEMM('T','N',lce-lcs+1,n_cols,lre,1.d0,a(1,lcs),ubound(a,dim=1), &
                     vmr,ubound(vmr,dim=1),1.d0,umc(lcs,1),ubound(umc,dim=1))

        if (i==0) cycle
        lre = min(l_rows,i*l_rows_tile)
        call DGEMM('N','N',lre,n_cols,lce-lcs+1,1.d0,a(1,lcs),lda, &
                     umc(lcs,n_cols+1),ubound(umc,dim=1),1.d0,vmr(1,n_cols+1),ubound(vmr,dim=1))

#endif
      enddo
#ifdef WITH_GPU_VERSION
      istat = cuda_memcpy(loc(vmr(1)), vmr_dev,vmr_size*8_8,cudaMemcpyDeviceToHost)
      if (istat .ne. 0) then
        print *,"error in cudaMemcpy"
        stop
      endif

      istat = cuda_memcpy(loc(umc(1)), umc_dev, umc_size*8_8,cudaMemcpyDeviceToHost)
      if (istat .ne. 0) then
        print *,"error in cudaMemcpy"
        stop
      endif

#endif
    endif

    ! Sum up all ur(:) parts along rows and add them to the uc(:) parts
    ! on the processors containing the diagonal
    ! This is only necessary if ur has been calculated, i.e. if the
    ! global tile size is smaller than the global remaining matrix

    if (tile_size < istep*nbw) then
#ifdef WITH_GPU_VERSION
       call elpa_reduce_add_vectors_real  (vmr(cur_l_rows * n_cols + 1),cur_l_rows,mpi_comm_rows, &
                                        umc, cur_l_cols, mpi_comm_cols, &
                                        istep*nbw, n_cols, nblk)
#else
       call elpa_reduce_add_vectors_real  (vmr(1,n_cols+1),ubound(vmr,dim=1),mpi_comm_rows, &
                                      umc, ubound(umc,dim=1), mpi_comm_cols, &
                                      istep*nbw, n_cols, nblk)
#endif
    endif

    if (l_cols>0) then
#ifdef WITH_GPU_VERSION
      allocate(tmp(l_cols * n_cols), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"bandred_real: error when allocating tmp "//errorMessage
       stop
     endif

      call mpi_allreduce(umc,tmp,l_cols*n_cols,MPI_REAL8,MPI_SUM,mpi_comm_rows,ierr)
      umc(1 : l_cols * n_cols) = tmp(1 : l_cols * n_cols)
#else
      allocate(tmp(l_cols,n_cols), stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"bandred_real: error when allocating tmp "//errorMessage
        stop
      endif

      call mpi_allreduce(umc,tmp,l_cols*n_cols,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)
      umc(1:l_cols,1:n_cols) = tmp(1:l_cols,1:n_cols)
#endif
      deallocate(tmp, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"bandred_real: error when deallocating tmp "//errorMessage
        stop
      endif

    endif

    ! U = U * Tmat**T
#ifdef WITH_GPU_VERSION
    istat = cuda_memcpy(umc_dev, loc(umc(1)), umc_size*8_8, cudaMemcpyHostToDevice)
    if (istat .ne. 0) then
      print *,"bandred_real: error in cudaMemcpy"
      stop
    endif

    istat = cuda_memcpy(tmat_dev,loc(tmat(1,1,istep)),nbw*nbw*8_8,cudaMemcpyHostToDevice)
    if (istat .ne. 0) then
      print *,"bandred_real: error in cudaMemcpy"
      stop
    endif

    call cublas_dtrmm('Right','Upper','Trans','Nonunit',l_cols,n_cols, &
                  1.d0, tmat_dev,nbw,umc_dev,cur_l_cols)

    ! VAV = Tmat * V**T * A * V * Tmat**T = (U*Tmat**T)**T * V * Tmat**T

    istat = cuda_memcpy(vav_dev,loc(vav(1,1)), nbw*nbw*8_8,cudaMemcpyHostToDevice)
    if(istat .ne. 0) print *, " cuad memcpy failed vav_dev ", istat


    call cublas_dgemm('T','N',n_cols,n_cols,l_cols, &
                  1.d0, umc_dev,cur_l_cols,(umc_dev+(cur_l_cols * n_cols )*8_8),cur_l_cols, &
                  0.d0, vav_dev,nbw)

    call cublas_dtrmm('Right','Upper','Trans','Nonunit',n_cols,n_cols, &
                   1.d0, tmat_dev,nbw, vav_dev, nbw)


    istat = cuda_memcpy(loc(vav(1,1)), vav_dev, nbw*nbw*8_8, cudaMemcpyDeviceToHost)
    if (istat .ne. 0) then
      print *,"bandred_real: error in cudaMemcpy"
      stop
    endif

    call symm_matrix_allreduce(n_cols,vav, nbw,nbw,mpi_comm_cols)

    istat = cuda_memcpy(vav_dev, loc(vav(1,1)), nbw*nbw*8_8,cudaMemcpyHostToDevice)
    if (istat .ne. 0) then
      print *,"bandred_real: error in cudaMemcpy"
      stop
    endif

#else
    call dtrmm('Right','Upper','Trans','Nonunit',l_cols,n_cols,1.d0,tmat(1,1,istep),ubound(tmat,dim=1),umc,ubound(umc,dim=1))

    ! VAV = Tmat * V**T * A * V * Tmat**T = (U*Tmat**T)**T * V * Tmat**T

    call dgemm('T','N',n_cols,n_cols,l_cols,1.d0,umc,ubound(umc,dim=1),umc(1,n_cols+1),ubound(umc,dim=1),0.d0,vav,ubound(vav,dim=1))
    call dtrmm('Right','Upper','Trans','Nonunit',n_cols,n_cols,1.d0,tmat(1,1,istep),ubound(tmat,dim=1),vav,ubound(vav,dim=1))

    call symm_matrix_allreduce(n_cols,vav,ubound(vav,dim=1),mpi_comm_cols)
#endif

    ! U = U - 0.5 * V * VAV
#ifdef WITH_GPU_VERSION
    call cublas_dgemm('N','N',l_cols,n_cols,n_cols,&
                    -0.5d0, (umc_dev+(cur_l_cols * n_cols )*8_8),cur_l_cols, vav_dev,nbw,&
                     1.0d0, umc_dev,cur_l_cols)

    istat = cuda_memcpy(loc(umc(1)), umc_dev, umc_size*8_8, cudaMemcpyDeviceToHost)
    if (istat .ne. 0) then
      print *,"bandred_real: error in cudaMemcpy"
      stop
    endif

    ! Transpose umc -> umr (stored in vmr, second half)

    call elpa_transpose_vectors_real  (umc, cur_l_cols, mpi_comm_cols, &
                                   vmr(cur_l_rows * n_cols + 1), cur_l_rows, mpi_comm_rows, &
                                   1, istep*nbw, n_cols, nblk)
    istat = cuda_memcpy(vmr_dev, loc(vmr(1)), vmr_size*8_8, cudaMemcpyHostToDevice)
    if (istat .ne. 0) then
      print *,"bandred_real: error in cudaMemcpy"
      stop
    endif

    istat = cuda_memcpy(umc_dev, loc(umc(1)), umc_size*8_8, cudaMemcpyHostToDevice)
    if (istat .ne. 0) then
      print *,"bandred_real: error in cudaMemcpy"
      stop
    endif

#else
    call dgemm('N','N',l_cols,n_cols,n_cols,-0.5d0,umc(1,n_cols+1),ubound(umc,dim=1),vav,ubound(vav,dim=1),1.d0,umc,ubound(umc,dim=1))

    ! Transpose umc -> umr (stored in vmr, second half)

    call elpa_transpose_vectors_real  (umc, ubound(umc,dim=1), mpi_comm_cols, &
                                   vmr(1,n_cols+1), ubound(vmr,dim=1), mpi_comm_rows, &
                                   1, istep*nbw, n_cols, nblk)
#endif

    ! A = A - V*U**T - U*V**T

    do i=0,(istep*nbw-1)/tile_size
      lcs = i*l_cols_tile+1
      lce = min(l_cols,(i+1)*l_cols_tile)
      lre = min(l_rows,(i+1)*l_rows_tile)
      if (lce<lcs .or. lre<1) cycle
#ifdef WITH_GPU_VERSION
      call cublas_dgemm('N', 'T', lre, lce-lcs+1, 2*n_cols, -1.d0, &
                 vmr_dev,cur_l_rows,(umc_dev +(lcs-1)*8_8),cur_l_cols, &
                 1.d0,(a_dev+(lcs-1)*lda*8_8),lda)
#else
      call dgemm('N','T',lre,lce-lcs+1,2*n_cols,-1.d0, &
                  vmr,ubound(vmr,dim=1),umc(lcs,1),ubound(umc,dim=1), &
                  1.d0,a(1,lcs),lda)
#endif
    enddo

#ifdef WITH_GPU_VERSION
  enddo
  istat = cuda_memcpy ( loc (a), a_dev, lda*na_cols*8_8,cudaMemcpyDeviceToHost)
  if (istat .ne. 0) then
    print *,"error in cudaMemcpy"
    stop
  endif

  istat = cuda_free(a_dev)
  if (istat .ne. 0) then
    print *,"bandred_real: error in cudaFree"
    stop
  endif

  istat = cuda_free(tmat_dev)
  if (istat .ne. 0) then
    print *,"bandred_real: error in cudaFree"
    stop
  endif

  istat = cuda_free(vav_dev)
  if (istat .ne. 0) then
    print *,"bandred_real: error in cudaFree"
    stop
  endif
#endif

    if (allocated(vr)) then
      deallocate(vr, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"bandred_real: error when deallocating vr "//errorMessage
        stop
      endif
    endif

    if (allocated(vmr)) then
      deallocate(vmr, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"bandred_real: error when deallocating vmr "//errorMessage
        stop
      endif

#if WITH_GPU_VERSION
      istat = cuda_free(vmr_dev)
      if (istat .ne. 0) then
        print *,"bandred_real: error in cudaFree"
        stop
      endif
#endif
    endif

    if (allocated(umc)) then
      deallocate(umc, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"bandred_real: error when deallocating umc "//errorMessage
        stop
      endif

#if WITH_GPU_VERSION
      istat = cuda_free(umc_dev)
      if (istat .ne. 0) then
        print *,"bandred_real: error in cudaFree"
        stop
      endif

#endif
    endif
#ifndef WITH_GPU_VERSION
  enddo
#endif

  if (useQR) then
    if (which_qr_decomposition == 1) then
      deallocate(work_blocked, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"bandred_real: error when deallocating work_blocked "//errorMessage
        stop
      endif

      deallocate(tauvector, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"bandred_real: error when deallocating tauvector "//errorMessage
        stop
      endif

    endif
  endif

#ifdef HAVE_DETAILED_TIMINGS
  call timer%stop("bandred_real")
#endif
end subroutine bandred_real ! slower for gpu on 10000 10000 ???

!-------------------------------------------------------------------------------

subroutine symm_matrix_allreduce(n,a,lda,ldb,comm)

!-------------------------------------------------------------------------------
!  symm_matrix_allreduce: Does an mpi_allreduce for a symmetric matrix A.
!  On entry, only the upper half of A needs to be set
!  On exit, the complete matrix is set
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
   implicit none
   integer  :: n, lda, ldb, comm
   real*8   :: a(lda,ldb)

   integer  :: i, nc, mpierr
   real*8   :: h1(n*n), h2(n*n)

#ifdef HAVE_DETAILED_TIMINGS
  call timer%start("symm_matrix_allreduce")
#endif

   nc = 0
   do i=1,n
     h1(nc+1:nc+i) = a(1:i,i)
     nc = nc+i
   enddo

   call mpi_allreduce(h1,h2,nc,MPI_REAL8,MPI_SUM,comm,mpierr)

   nc = 0
   do i=1,n
     a(1:i,i) = h2(nc+1:nc+i)
     a(i,1:i-1) = a(1:i-1,i)
     nc = nc+i
   enddo

#ifdef HAVE_DETAILED_TIMINGS
  call timer%stop("symm_matrix_allreduce")
#endif

end subroutine symm_matrix_allreduce

!-------------------------------------------------------------------------------

subroutine trans_ev_band_to_full_real(na, nqc, nblk, nbw, a, lda, tmat, q, ldq, matrixCols, numBlocks, mpi_comm_rows, &
                                      mpi_comm_cols, useQR)


!-------------------------------------------------------------------------------
!  trans_ev_band_to_full_real:
!  Transforms the eigenvectors of a band matrix back to the eigenvectors of the original matrix
!
!  Parameters
!
!  na          Order of matrix a, number of rows of matrix q
!
!  nqc         Number of columns of matrix q
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  nbw         semi bandwith
!
!  a(lda,matrixCols)    Matrix containing the Householder vectors (i.e. matrix a after bandred_real)
!              Distribution is like in Scalapack.
!
!  lda         Leading dimension of a
!  matrixCols  local columns of matrix a and q
!
!  tmat(nbw,nbw,numBlocks) Factors returned by bandred_real
!
!  q           On input: Eigenvectors of band matrix
!              On output: Transformed eigenvectors
!              Distribution is like in Scalapack.
!
!  ldq         Leading dimension of q
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif

#ifdef WITH_GPU_VERSION
 use cuda_routines
 use iso_c_binding
#endif
   implicit none
   integer              :: na, nqc, lda, ldq, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols
   real*8               :: a(lda,matrixCols), q(ldq,matrixCols), tmat(nbw, nbw, numBlocks)

   real*8, allocatable  :: q_temp(:,:), tmat_temp(:,:)
   integer              :: my_prow, my_pcol, np_rows, np_cols, mpierr
   integer              :: max_blocks_row, max_blocks_col, max_local_rows, &
                           max_local_cols
   integer              :: l_cols, l_rows, l_colh, n_cols
   integer              :: istep, lc, ncol, nrow, nb, ns

   real*8, allocatable  :: tmp1(:), tmp2(:), hvb(:), hvm(:,:)
#ifdef WITH_GPU_VERSION
   integer(C_SIZE_T)    :: hvm_dev, q_dev, tmp_dev, tmat_dev
#endif

   integer              :: i

   real*8, allocatable  :: tmat_complete(:,:), t_tmp(:,:), t_tmp2(:,:)
   integer              :: cwy_blocking, t_blocking, t_cols, t_rows
   logical, intent(in)  :: useQR
   integer              :: istat
   character(200)       :: errorMessage

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("trans_ev_band_to_full_real")
#endif

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   max_blocks_row = ((na -1)/nblk)/np_rows + 1  ! Rows of A
   max_blocks_col = ((nqc-1)/nblk)/np_cols + 1  ! Columns of q!

   max_local_rows = max_blocks_row*nblk
   max_local_cols = max_blocks_col*nblk

   if (useQR) then
#ifdef WITH_GPU_VERSION
     print *,"no QR with GPU"
     stop
#endif
     t_blocking = 2 ! number of matrices T (tmat) which are aggregated into a new (larger) T matrix (tmat_complete) and applied at once
     cwy_blocking = t_blocking * nbw

     allocate(tmp1(max_local_cols*cwy_blocking), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error when allocating tmp1 "//errorMessage
       stop
     endif

     allocate(tmp2(max_local_cols*cwy_blocking), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error when allocating tmp2 "//errorMessage
       stop
     endif

     allocate(hvb(max_local_rows*cwy_blocking), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error when allocating hvb "//errorMessage
       stop
     endif

     allocate(hvm(max_local_rows,cwy_blocking), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error when allocating hvm "//errorMessage
       stop
     endif

     allocate(tmat_complete(cwy_blocking,cwy_blocking), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error when allocating tmat_complete "//errorMessage
       stop
     endif

     allocate(t_tmp(cwy_blocking,nbw), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error when allocating t_tmp "//errorMessage
       stop
     endif

     allocate(t_tmp2(cwy_blocking,nbw), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error when allocating t_tmp2 "//errorMessage
       stop
     endif

   else ! no QR
     allocate(tmp1(max_local_cols*nbw), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error when allocating tmp1 "//errorMessage
       stop
     endif

     allocate(tmp2(max_local_cols*nbw), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error when allocating tmp2 "//errorMessage
       stop
     endif

     allocate(hvb(max_local_rows*nbw), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error when allocating hvb "//errorMessage
       stop
     endif

     allocate(hvm(max_local_rows,nbw), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error when allocating hvm "//errorMessage
       stop
     endif

#ifdef WITH_GPU_VERSION
!     allocate(q_temp(ldq,max_local_cols), stat=istat, errmsg=errorMessage)
!     if (istat .ne. 0) then
!       print *,"error when allocating q_temp "//errorMessage
!       stop
!     endif

     allocate(tmat_temp(nbw,nbw), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error when allocating tmat_temp "//errorMessage
       stop
     endif

     istat = cuda_malloc(hvm_dev, (max_local_rows)*nbw*8_8)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error in cudaMalloc"
       stop
     endif

     istat = cuda_malloc(tmp_dev, (max_local_cols)*nbw*8_8)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error in cudaMalloc"
       stop
     endif

     istat = cuda_malloc(tmat_dev, nbw*nbw*8_8)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error in cudaMalloc"
       stop
     endif

     istat = cuda_malloc(q_dev, ldq*matrixCols*8_8)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error in cudaMalloc"
       stop
     endif

!     q_temp(:,:) = 0.0
!     q_temp(1:ldq,1:na_cols) = q(1:ldq,1:na_cols)

     istat = cuda_memcpy(q_dev, loc(q), (ldq)*(matrixCols)*8_8, cudaMemcpyHostToDevice)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error in cudaMalloc"
       stop
     endif

     istat = cuda_memset(hvm_dev, 0, (max_local_rows)*(nbw)*8_8)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error in cudaMalloc"
       stop
     endif

#endif


   endif ! QR

   hvm = 0.0   ! Must be set to 0 !!!

   hvb = 0.0   ! Safety only



   l_cols = local_index(nqc, my_pcol, np_cols, nblk, -1) ! Local columns of q

   if (useQR) then
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_band_to_full_real: no QR with GPU"
     stop
#endif
     do istep=1,((na-1)/nbw-1)/t_blocking + 1
       n_cols = MIN(na,istep*cwy_blocking+nbw) - (istep-1)*cwy_blocking - nbw ! Number of columns in current step

       ! Broadcast all Householder vectors for current step compressed in hvb

       nb = 0
       ns = 0

       do lc = 1, n_cols
         ncol = (istep-1)*cwy_blocking + nbw + lc ! absolute column number of householder vector
         nrow = ncol - nbw ! absolute number of pivot row

         l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast
         l_colh = local_index(ncol  , my_pcol, np_cols, nblk, -1) ! HV local column number

         if (my_pcol==pcol(ncol, nblk, np_cols)) hvb(nb+1:nb+l_rows) = a(1:l_rows,l_colh)

         nb = nb+l_rows

         if (lc==n_cols .or. mod(ncol,nblk)==0) then
           call MPI_Bcast(hvb(ns+1),nb-ns,MPI_REAL8,pcol(ncol, nblk, np_cols),mpi_comm_cols,mpierr)
           ns = nb
         endif
       enddo

       ! Expand compressed Householder vectors into matrix hvm

       nb = 0
       do lc = 1, n_cols
         nrow = (istep-1)*cwy_blocking + lc ! absolute number of pivot row
         l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast

         hvm(1:l_rows,lc) = hvb(nb+1:nb+l_rows)
         if (my_prow==prow(nrow, nblk, np_rows)) hvm(l_rows+1,lc) = 1.

         nb = nb+l_rows
       enddo

       l_rows = local_index(MIN(na,(istep+1)*cwy_blocking), my_prow, np_rows, nblk, -1)

       ! compute tmat2 out of tmat(:,:,)
       tmat_complete = 0
       do i = 1, t_blocking
         t_cols = MIN(nbw, n_cols - (i-1)*nbw)
         if (t_cols <= 0) exit
         t_rows = (i - 1) * nbw
         tmat_complete(t_rows+1:t_rows+t_cols,t_rows+1:t_rows+t_cols) = tmat(1:t_cols,1:t_cols,(istep-1)*t_blocking + i)
         if (i > 1) then
           call dgemm('T', 'N', t_rows, t_cols, l_rows, 1.d0, hvm(1,1), max_local_rows, hvm(1,(i-1)*nbw+1), &
                     max_local_rows, 0.d0, t_tmp, cwy_blocking)
           call mpi_allreduce(t_tmp,t_tmp2,cwy_blocking*nbw,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)
           call dtrmm('L','U','N','N',t_rows,t_cols,1.0d0,tmat_complete,cwy_blocking,t_tmp2,cwy_blocking)
           call dtrmm('R','U','N','N',t_rows,t_cols,-1.0d0,tmat_complete(t_rows+1,t_rows+1),cwy_blocking,t_tmp2,cwy_blocking)
           tmat_complete(1:t_rows,t_rows+1:t_rows+t_cols) = t_tmp2(1:t_rows,1:t_cols)
         endif
       enddo

       ! Q = Q - V * T**T * V**T * Q

       if (l_rows>0) then
         call dgemm('T','N',n_cols,l_cols,l_rows,1.d0,hvm,ubound(hvm,dim=1), &
                    q,ldq,0.d0,tmp1,n_cols)
       else
         tmp1(1:l_cols*n_cols) = 0
       endif
       call mpi_allreduce(tmp1,tmp2,n_cols*l_cols,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)


       if (l_rows>0) then
         call dtrmm('L','U','T','N',n_cols,l_cols,1.0d0,tmat_complete,cwy_blocking,tmp2,n_cols)
         call dgemm('N','N',l_rows,l_cols,n_cols,-1.d0,hvm,ubound(hvm,dim=1), tmp2,n_cols,1.d0,q,ldq)
       endif
     enddo

   else !  do not useQR

     do istep=1,(na-1)/nbw

       n_cols = MIN(na,(istep+1)*nbw) - istep*nbw ! Number of columns in current step

       ! Broadcast all Householder vectors for current step compressed in hvb

       nb = 0
       ns = 0

       do lc = 1, n_cols
         ncol = istep*nbw + lc ! absolute column number of householder vector
         nrow = ncol - nbw ! absolute number of pivot row

         l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast
         l_colh = local_index(ncol  , my_pcol, np_cols, nblk, -1) ! HV local column number

         if (my_pcol==pcol(ncol, nblk, np_cols)) hvb(nb+1:nb+l_rows) = a(1:l_rows,l_colh)

         nb = nb+l_rows

         if (lc==n_cols .or. mod(ncol,nblk)==0) then
           call MPI_Bcast(hvb(ns+1),nb-ns,MPI_REAL8,pcol(ncol, nblk, np_cols),mpi_comm_cols,mpierr)
           ns = nb
         endif
       enddo

       ! Expand compressed Householder vectors into matrix hvm

       nb = 0
       do lc = 1, n_cols
         nrow = (istep-1)*nbw+lc ! absolute number of pivot row
         l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast

         hvm(1:l_rows,lc) = hvb(nb+1:nb+l_rows)
         if (my_prow==prow(nrow, nblk, np_rows)) hvm(l_rows+1,lc) = 1.

         nb = nb+l_rows
       enddo

#ifdef WITH_GPU_VERSION
       istat = cuda_memcpy(hvm_dev, loc(hvm), ((max_local_rows)*nbw*8_8),cudaMemcpyHostToDevice)
       if (istat .ne. 0) then
         print *,"trans_ev_band_to_full_real: error in cudaMemcpy"
         stop
       endif
#endif
       l_rows = local_index(MIN(na,(istep+1)*nbw), my_prow, np_rows, nblk, -1)

       ! Q = Q - V * T**T * V**T * Q

       if (l_rows>0) then
#ifdef WITH_GPU_VERSION
         call cublas_dgemm('T','N',n_cols,l_cols,l_rows,1.d0,hvm_dev,max_local_rows, &
                    q_dev,ldq ,0.d0,tmp_dev,n_cols)
         istat = cuda_memcpy(loc(tmp1), tmp_dev, l_cols*n_cols*8_8, cudaMemcpyDeviceToHost)
         if (istat .ne. 0) then
           print *,"trans_ev_band_to_full_real: error in cudaMemcpy"
           stop
         endif
#else
         call dgemm('T','N',n_cols,l_cols,l_rows,1.d0,hvm,ubound(hvm,dim=1), &
                    q,ldq,0.d0,tmp1,n_cols)
#endif
       else
!#ifdef WITH_GPU_VERSION
!         istat = cuda_memset(tmp_dev, 0, l_cols*n_cols*8_8)
!         if (istat .ne. 0) then
!           print *,"trans_ev_band_to_full_real: error in cudaMemset"
!           stop
!         endif
!
!#else
         tmp1(1:l_cols*n_cols) = 0
!#endif
       endif

!#ifdef WITH_GPU_VERSION
!       istat = cuda_memcpy(loc(tmp1), tmp_dev, max_local_cols*nbw*8_8,cudaMemcpyDeviceToHost)
!       if (istat .ne. 0) then
!         print *,"error in cudaMemcpy"
!         stop
!       endif
!#endif
       call mpi_allreduce(tmp1,tmp2,n_cols*l_cols,MPI_REAL8,MPI_SUM,mpi_comm_rows,mpierr)

!#ifdef WITH_GPU_VERSION
!       istat = cuda_memcpy(tmp_dev, loc(tmp2), max_local_cols*nbw*8_8,cudaMemcpyHostToDevice)
!       if (istat .ne. 0) then
!         print *,"error in cudaMemcpy"
!         stop
!       endif
!#endif

       if (l_rows>0) then
#ifdef WITH_GPU_VERSION
         istat = cuda_memcpy(tmp_dev, loc(tmp2), n_cols*l_cols*8_8,cudaMemcpyHostToDevice)
         if (istat .ne. 0) then
           print *,"trans_ev_band_to_full_real: error in cudaMemcpy"
           stop
         endif

         istat = cuda_memcpy(tmat_dev, loc(tmat(1,1,istep)), nbw*nbw*8_8,cudaMemcpyHostToDevice)
         if (istat .ne. 0) then
           print *,"trans_ev_band_to_full_real: error in cudaMemcpy"
           stop
         endif

         call cublas_dtrmm('L','U','T','N',n_cols,l_cols,1.0d0, tmat_dev, nbw, tmp_dev, n_cols)
         call cublas_dgemm('N','N',l_rows,l_cols,n_cols,-1.d0,hvm_dev,max_local_rows, &
                    tmp_dev,n_cols,1.d0,q_dev,ldq)

         istat = cuda_memcpy(loc(hvm), hvm_dev, ((max_local_rows)*nbw*8_8),cudaMemcpyDeviceToHost)
         if (istat .ne. 0) then
           print *,"trans_ev_band_to_full_real: error in cudaMemcpy"
           stop
         endif

#else
         call dtrmm('L','U','T','N',n_cols,l_cols,1.0d0,tmat(1,1,istep),ubound(tmat,dim=1),tmp2,n_cols)
         call dgemm('N','N',l_rows,l_cols,n_cols,-1.d0,hvm,ubound(hvm,dim=1), &
                    tmp2,n_cols,1.d0,q,ldq)
#endif
       endif
!#ifdef WITH_GPU_VERSION
!       istat = cuda_memcpy(loc(hvm), hvm_dev, ((max_local_rows)*nbw*8_8),cudaMemcpyDeviceToHost)
!       if (istat .ne. 0) then
!         print *,"error in cudaMemcpy"
!         stop
!       endif
!
!#endif
     enddo
   endif ! endQR

   deallocate(tmp1, tmp2, hvb, stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_real: error when deallocating tmp1 tmp2 hvb "//errorMessage
     stop
   endif

#ifdef WITH_GPU_VERSION
   istat = cuda_free(hvm_dev)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_real: error in cudaFree"
     stop
   endif

   istat = cuda_free(tmp_dev)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_real: error in cudaFree"
     stop
   endif

   istat = cuda_free(tmat_dev)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_real: error in cudaFree"
     stop
   endif

   istat = cuda_memcpy(loc(q), q_dev, ldq*matrixCols*8_8, cudaMemcpyDeviceToHost)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_real: error in cudaFree"
     stop
   endif

!   q(1:ldq,1:na_cols) = q_temp(1:ldq,1:na_cols)

   istat = cuda_free(q_dev)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_real: error in cudaFree"
     stop
   endif

!   deallocate(q_temp, stat=istat, errmsg=errorMessage)
!   if (istat .ne. 0) then
!     print *,"error when deallocating q_temp "//errorMessage
!     stop
!   endif

   deallocate(tmat_temp, stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_real: error when deallocating tmat_temp "//errorMessage
     stop
   endif

#endif
   deallocate(hvm, stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_real: error when deallocating hvm "//errorMessage
     stop
   endif

   if (useQr) then
     deallocate(tmat_complete, t_tmp, t_tmp2, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_real: error when deallocating tmat_complete, t_tmp, t_tmp2 "//errorMessage
       stop
     endif

   endif

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("trans_ev_band_to_full_real")
#endif
end subroutine trans_ev_band_to_full_real

! --------------------------------------------------------------------------------------------------

subroutine tridiag_band_real(na, nb, nblk, a, lda, d, e, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm)

!-------------------------------------------------------------------------------
! tridiag_band_real:
! Reduces a real symmetric band matrix to tridiagonal form
!
!  na          Order of matrix a
!
!  nb          Semi bandwith
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  a(lda,matrixCols)    Distributed system matrix reduced to banded form in the upper diagonal
!
!  lda         Leading dimension of a
!  matrixCols  local columns of matrix a
!
!  d(na)       Diagonal of tridiagonal matrix, set only on PE 0 (output)
!
!  e(na)       Subdiagonal of tridiagonal matrix, set only on PE 0 (output)
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!  mpi_comm
!              MPI-Communicator for the total processor set
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
   implicit none

   integer, intent(in) ::  na, nb, nblk, lda, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm
   real*8, intent(in)  :: a(lda,matrixCols)
   real*8, intent(out) :: d(na), e(na) ! set only on PE 0


   real*8               :: vnorm2, hv(nb), tau, x, h(nb), ab_s(1+nb), hv_s(nb), hv_new(nb), tau_new, hf
   real*8               :: hd(nb), hs(nb)

   integer              :: i, j, n, nc, nr, ns, ne, istep, iblk, nblocks_total, nblocks, nt
   integer              :: my_pe, n_pes, mpierr
   integer              :: my_prow, np_rows, my_pcol, np_cols
   integer              :: ireq_ab, ireq_hv
   integer              :: na_s, nx, num_hh_vecs, num_chunks, local_size, max_blk_size, n_off
#ifdef WITH_OPENMP
   integer              :: max_threads, my_thread, my_block_s, my_block_e, iter
   integer              :: mpi_status(MPI_STATUS_SIZE)
   integer, allocatable :: mpi_statuses(:,:), global_id_tmp(:,:)
   integer, allocatable :: omp_block_limits(:)
   real*8, allocatable  :: hv_t(:,:), tau_t(:)
#endif
   integer, allocatable :: ireq_hhr(:), ireq_hhs(:), global_id(:,:), hh_cnt(:), hh_dst(:)
   integer, allocatable :: limits(:), snd_limits(:,:)
   integer, allocatable :: block_limits(:)
   real*8, allocatable :: ab(:,:), hh_gath(:,:,:), hh_send(:,:,:)
!   ! dummies for calling redist_band
!   complex*16 :: c_a(1,1), c_ab(1,1)

#ifdef WITH_OPENMP
   integer              :: omp_get_max_threads
#endif
   integer              :: istat
   character(200)       :: errorMessage

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("tridiag_band_real")
#endif

   call mpi_comm_rank(mpi_comm,my_pe,mpierr)
   call mpi_comm_size(mpi_comm,n_pes,mpierr)

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   ! Get global_id mapping 2D procssor coordinates to global id

   allocate(global_id(0:np_rows-1,0:np_cols-1), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when allocating global_id "//errorMessage
     stop
   endif


   global_id(:,:) = 0
   global_id(my_prow, my_pcol) = my_pe
#ifdef WITH_OPENMP
   allocate(global_id_tmp(0:np_rows-1,0:np_cols-1), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when allocating global_id_tmp "//errorMessage
     stop
   endif

#endif

#ifndef WITH_OPENMP
   call mpi_allreduce(mpi_in_place, global_id, np_rows*np_cols, mpi_integer, mpi_sum, mpi_comm, mpierr)
#else
    global_id_tmp(:,:) = global_id(:,:)
    call mpi_allreduce(global_id_tmp, global_id, np_rows*np_cols, mpi_integer, mpi_sum, mpi_comm, mpierr)
    deallocate(global_id_tmp, stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when deallocating global_id_tmp "//errorMessage
     stop
   endif

#endif

   ! Total number of blocks in the band:

   nblocks_total = (na-1)/nb + 1

   ! Set work distribution

   allocate(block_limits(0:n_pes), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when allocating block_limits"//errorMessage
     stop
   endif


   call divide_band(nblocks_total, n_pes, block_limits)

   ! nblocks: the number of blocks for my task
   nblocks = block_limits(my_pe+1) - block_limits(my_pe)

   ! allocate the part of the band matrix which is needed by this PE
   ! The size is 1 block larger than needed to avoid extensive shifts
   allocate(ab(2*nb,(nblocks+1)*nb), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when allocating ab"//errorMessage
     stop
   endif

   ab = 0 ! needed for lower half, the extra block should also be set to 0 for safety

   ! n_off: Offset of ab within band
   n_off = block_limits(my_pe)*nb

   ! Redistribute band in a to ab
   call redist_band_real(a, lda, na, nblk, nb, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm, ab)

   ! Calculate the workload for each sweep in the back transformation
   ! and the space requirements to hold the HH vectors

   allocate(limits(0:np_rows), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when allocating limits"//errorMessage
     stop
   endif

   call determine_workload(na, nb, np_rows, limits)
   max_blk_size = maxval(limits(1:np_rows) - limits(0:np_rows-1))

   num_hh_vecs = 0
   num_chunks  = 0
   nx = na
   do n = 1, nblocks_total
     call determine_workload(nx, nb, np_rows, limits)
     local_size = limits(my_prow+1) - limits(my_prow)
     ! add to number of householder vectors
     ! please note: for nx==1 the one and only HH vector is 0 and is neither calculated nor send below!
     if (mod(n-1,np_cols) == my_pcol .and. local_size>0 .and. nx>1) then
       num_hh_vecs = num_hh_vecs + local_size
       num_chunks  = num_chunks+1
     endif
     nx = nx - nb
   enddo

   ! Allocate space for HH vectors

   allocate(hh_trans_real(nb,num_hh_vecs), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when allocating hh_trans_real"//errorMessage
     stop
   endif


   ! Allocate and init MPI requests

   allocate(ireq_hhr(num_chunks), stat=istat, errmsg=errorMessage) ! Recv requests
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when allocating ireq_hhr"//errorMessage
     stop
   endif
   allocate(ireq_hhs(nblocks), stat=istat, errmsg=errorMessage)    ! Send requests
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when allocating ireq_hhs"//errorMessage
     stop
   endif

   num_hh_vecs = 0
   num_chunks  = 0
   nx = na
   nt = 0
   do n = 1, nblocks_total
     call determine_workload(nx, nb, np_rows, limits)
     local_size = limits(my_prow+1) - limits(my_prow)
     if (mod(n-1,np_cols) == my_pcol .and. local_size>0 .and. nx>1) then
       num_chunks  = num_chunks+1
       call mpi_irecv(hh_trans_real(1,num_hh_vecs+1), nb*local_size, mpi_real8, nt, &
                        10+n-block_limits(nt), mpi_comm, ireq_hhr(num_chunks), mpierr)
       num_hh_vecs = num_hh_vecs + local_size
     endif
     nx = nx - nb
     if (n == block_limits(nt+1)) then
       nt = nt + 1
     endif
   enddo

   ireq_hhs(:) = MPI_REQUEST_NULL

   ! Buffers for gathering/sending the HH vectors

   allocate(hh_gath(nb,max_blk_size,nblocks), stat=istat, errmsg=errorMessage) ! gathers HH vectors
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when allocating hh_gath"//errorMessage
     stop
   endif

   allocate(hh_send(nb,max_blk_size,nblocks), stat=istat, errmsg=errorMessage) ! send buffer for HH vectors
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when allocating hh_send"//errorMessage
     stop
   endif
   hh_gath(:,:,:) = 0
   hh_send(:,:,:) = 0

   ! Some counters

   allocate(hh_cnt(nblocks), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when allocating hh_cnt"//errorMessage
     stop
   endif

   allocate(hh_dst(nblocks), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when allocating hh_dst"//errorMessage
     stop
   endif


   hh_cnt(:) = 1 ! The first transfomation vector is always 0 and not calculated at all
   hh_dst(:) = 0 ! PE number for receive

   ireq_ab = MPI_REQUEST_NULL
   ireq_hv = MPI_REQUEST_NULL

   ! Limits for sending

   allocate(snd_limits(0:np_rows,nblocks), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when allocating snd_limits"//errorMessage
     stop
   endif
   do iblk=1,nblocks
     call determine_workload(na-(iblk+block_limits(my_pe)-1)*nb, nb, np_rows, snd_limits(:,iblk))
   enddo

#ifdef WITH_OPENMP
   ! OpenMP work distribution:

   max_threads = 1
   max_threads = omp_get_max_threads()

   ! For OpenMP we need at least 2 blocks for every thread
   max_threads = MIN(max_threads, nblocks/2)
   if (max_threads==0) max_threads = 1

   allocate(omp_block_limits(0:max_threads), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when allocating omp_block_limits"//errorMessage
     stop
   endif

   ! Get the OpenMP block limits
   call divide_band(nblocks, max_threads, omp_block_limits)

   allocate(hv_t(nb,max_threads), tau_t(max_threads), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when allocating hv_t, tau_t"//errorMessage
     stop
   endif

   hv_t = 0
   tau_t = 0
#endif

   ! ---------------------------------------------------------------------------
   ! Start of calculations

   na_s = block_limits(my_pe)*nb + 1

   if (my_pe>0 .and. na_s<=na) then
     ! send first column to previous PE
     ! Only the PE owning the diagonal does that (sending 1 element of the subdiagonal block also)
     ab_s(1:nb+1) = ab(1:nb+1,na_s-n_off)
     call mpi_isend(ab_s,nb+1,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
   endif

#ifdef WITH_OPENMP
   do istep=1,na-1-block_limits(my_pe)*nb
#else
   do istep=1,na-1
#endif

     if (my_pe==0) then
       n = MIN(na-na_s,nb) ! number of rows to be reduced
       hv(:) = 0
       tau = 0
       ! The last step (istep=na-1) is only needed for sending the last HH vectors.
       ! We don't want the sign of the last element flipped (analogous to the other sweeps)
       if (istep < na-1) then
         ! Transform first column of remaining matrix
         vnorm2 = sum(ab(3:n+1,na_s-n_off)**2)
         call hh_transform_real(ab(2,na_s-n_off),vnorm2,hf,tau)
         hv(1) = 1
         hv(2:n) = ab(3:n+1,na_s-n_off)*hf
       endif
       d(istep) = ab(1,na_s-n_off)
       e(istep) = ab(2,na_s-n_off)
       if (istep == na-1) then
         d(na) = ab(1,na_s+1-n_off)
         e(na) = 0
       endif
     else
       if (na>na_s) then
         ! Receive Householder vector from previous task, from PE owning subdiagonal
#ifdef WITH_OPENMP
         call mpi_recv(hv,nb,mpi_real8,my_pe-1,2,mpi_comm,MPI_STATUS,mpierr)
#else
         call mpi_recv(hv,nb,mpi_real8,my_pe-1,2,mpi_comm,MPI_STATUS_IGNORE,mpierr)
#endif
         tau = hv(1)
         hv(1) = 1.
       endif
     endif

     na_s = na_s+1
     if (na_s-n_off > nb) then
       ab(:,1:nblocks*nb) = ab(:,nb+1:(nblocks+1)*nb)
       ab(:,nblocks*nb+1:(nblocks+1)*nb) = 0
       n_off = n_off + nb
     endif

#ifdef WITH_OPENMP
     if (max_threads > 1) then

       ! Codepath for OpenMP

       ! Please note that in this case it is absolutely necessary to have at least 2 blocks per thread!
       ! Every thread is one reduction cycle behind its predecessor and thus starts one step later.
       ! This simulates the behaviour of the MPI tasks which also work after each other.
       ! The code would be considerably easier, if the MPI communication would be made within
       ! the parallel region - this is avoided here since this would require
       ! MPI_Init_thread(MPI_THREAD_MULTIPLE) at the start of the program.

       hv_t(:,1) = hv
       tau_t(1) = tau

       do iter = 1, 2

         ! iter=1 : work on first block
         ! iter=2 : work on remaining blocks
         ! This is done in 2 iterations so that we have a barrier in between:
         ! After the first iteration, it is guaranteed that the last row of the last block
         ! is completed by the next thread.
         ! After the first iteration it is also the place to exchange the last row
         ! with MPI calls
#ifdef HAVE_DETAILED_TIMINGS
         call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, my_block_s, my_block_e, iblk, ns, ne, hv, tau, &
!$omp&                    nc, nr, hs, hd, vnorm2, hf, x, h, i), schedule(static,1), num_threads(max_threads)
         do my_thread = 1, max_threads

           if (iter == 1) then
             my_block_s = omp_block_limits(my_thread-1) + 1
             my_block_e = my_block_s
           else
             my_block_s = omp_block_limits(my_thread-1) + 2
             my_block_e = omp_block_limits(my_thread)
           endif

           do iblk = my_block_s, my_block_e

             ns = na_s + (iblk-1)*nb - n_off - my_thread + 1 ! first column in block
             ne = ns+nb-1                    ! last column in block

             if (istep<my_thread .or. ns+n_off>na) exit

             hv = hv_t(:,my_thread)
             tau = tau_t(my_thread)

             ! Store Householder vector for back transformation

             hh_cnt(iblk) = hh_cnt(iblk) + 1

             hh_gath(1   ,hh_cnt(iblk),iblk) = tau
             hh_gath(2:nb,hh_cnt(iblk),iblk) = hv(2:nb)

             nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
             nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                       ! Note that nr>=0 implies that diagonal block is full (nc==nb)!

             ! Transform diagonal block

             call DSYMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,0.d0,hd,1)

             x = dot_product(hv(1:nc),hd(1:nc))*tau
             hd(1:nc) = hd(1:nc) - 0.5*x*hv(1:nc)

             call DSYR2('L',nc,-1.d0,hd,1,hv,1,ab(1,ns),2*nb-1)

             hv_t(:,my_thread) = 0
             tau_t(my_thread)  = 0

             if (nr<=0) cycle ! No subdiagonal block present any more

             ! Transform subdiagonal block

             call DGEMV('N',nr,nb,tau,ab(nb+1,ns),2*nb-1,hv,1,0.d0,hs,1)

             if (nr>1) then

               ! complete (old) Householder transformation for first column

               ab(nb+1:nb+nr,ns) = ab(nb+1:nb+nr,ns) - hs(1:nr) ! Note: hv(1) == 1

               ! calculate new Householder transformation for first column
               ! (stored in hv_t(:,my_thread) and tau_t(my_thread))

               vnorm2 = sum(ab(nb+2:nb+nr,ns)**2)
               call hh_transform_real(ab(nb+1,ns),vnorm2,hf,tau_t(my_thread))
               hv_t(1   ,my_thread) = 1.
               hv_t(2:nr,my_thread) = ab(nb+2:nb+nr,ns)*hf
               ab(nb+2:,ns) = 0

               ! update subdiagonal block for old and new Householder transformation
               ! This way we can use a nonsymmetric rank 2 update which is (hopefully) faster

               call DGEMV('T',nr,nb-1,tau_t(my_thread),ab(nb,ns+1),2*nb-1,hv_t(1,my_thread),1,0.d0,h(2),1)
               x = dot_product(hs(1:nr),hv_t(1:nr,my_thread))*tau_t(my_thread)
               h(2:nb) = h(2:nb) - x*hv(2:nb)
               ! Unfortunately there is no BLAS routine like DSYR2 for a nonsymmetric rank 2 update ("DGER2")
               do i=2,nb
                 ab(2+nb-i:1+nb+nr-i,i+ns-1) = ab(2+nb-i:1+nb+nr-i,i+ns-1) - hv_t(1:nr,my_thread)*h(i) - hs(1:nr)*hv(i)
               enddo

             else

               ! No new Householder transformation for nr=1, just complete the old one
               ab(nb+1,ns) = ab(nb+1,ns) - hs(1) ! Note: hv(1) == 1
               do i=2,nb
                 ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*hv(i)
               enddo
               ! For safety: there is one remaining dummy transformation (but tau is 0 anyways)
               hv_t(1,my_thread) = 1.

             endif

           enddo

         enddo ! my_thread
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
         call timer%stop("OpenMP parallel")
#endif

         if (iter==1) then
           ! We are at the end of the first block

           ! Send our first column to previous PE
           if (my_pe>0 .and. na_s <= na) then
             call mpi_wait(ireq_ab,mpi_status,mpierr)
             ab_s(1:nb+1) = ab(1:nb+1,na_s-n_off)
             call mpi_isend(ab_s,nb+1,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
           endif

           ! Request last column from next PE
           ne = na_s + nblocks*nb - (max_threads-1) - 1
           if (istep>=max_threads .and. ne <= na) then
             call mpi_recv(ab(1,ne-n_off),nb+1,mpi_real8,my_pe+1,1,mpi_comm,mpi_status,mpierr)
           endif

         else
           ! We are at the end of all blocks

           ! Send last HH vector and TAU to next PE if it has been calculated above
           ne = na_s + nblocks*nb - (max_threads-1) - 1
           if (istep>=max_threads .and. ne < na) then
             call mpi_wait(ireq_hv,mpi_status,mpierr)
             hv_s(1) = tau_t(max_threads)
             hv_s(2:) = hv_t(2:,max_threads)
             call mpi_isend(hv_s,nb,mpi_real8,my_pe+1,2,mpi_comm,ireq_hv,mpierr)
           endif

           ! "Send" HH vector and TAU to next OpenMP thread
           do my_thread = max_threads, 2, -1
             hv_t(:,my_thread) = hv_t(:,my_thread-1)
             tau_t(my_thread)  = tau_t(my_thread-1)
           enddo

         endif
       enddo ! iter

     else

       ! Codepath for 1 thread without OpenMP

       ! The following code is structured in a way to keep waiting times for
       ! other PEs at a minimum, especially if there is only one block.
       ! For this reason, it requests the last column as late as possible
       ! and sends the Householder vector and the first column as early
       ! as possible.

#endif /* WITH_OPENMP */

       do iblk=1,nblocks

         ns = na_s + (iblk-1)*nb - n_off ! first column in block
         ne = ns+nb-1                    ! last column in block

         if (ns+n_off>na) exit

         ! Store Householder vector for back transformation

         hh_cnt(iblk) = hh_cnt(iblk) + 1

         hh_gath(1   ,hh_cnt(iblk),iblk) = tau
         hh_gath(2:nb,hh_cnt(iblk),iblk) = hv(2:nb)

#ifndef WITH_OPENMP
         if (hh_cnt(iblk) == snd_limits(hh_dst(iblk)+1,iblk)-snd_limits(hh_dst(iblk),iblk)) then
           ! Wait for last transfer to finish

           call mpi_wait(ireq_hhs(iblk), MPI_STATUS_IGNORE, mpierr)

           ! Copy vectors into send buffer
           hh_send(:,1:hh_cnt(iblk),iblk) = hh_gath(:,1:hh_cnt(iblk),iblk)
           ! Send to destination
           call mpi_isend(hh_send(1,1,iblk), nb*hh_cnt(iblk), mpi_real8, &
                        global_id(hh_dst(iblk),mod(iblk+block_limits(my_pe)-1,np_cols)), &
                        10+iblk, mpi_comm, ireq_hhs(iblk), mpierr)
         ! Reset counter and increase destination row
           hh_cnt(iblk) = 0
           hh_dst(iblk) = hh_dst(iblk)+1
         endif

         ! The following code is structured in a way to keep waiting times for
         ! other PEs at a minimum, especially if there is only one block.
         ! For this reason, it requests the last column as late as possible
         ! and sends the Householder vector and the first column as early
         ! as possible.
#endif
         nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
         nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                       ! Note that nr>=0 implies that diagonal block is full (nc==nb)!

         ! Multiply diagonal block and subdiagonal block with Householder vector

         if (iblk==nblocks .and. nc==nb) then

           ! We need the last column from the next PE.
           ! First do the matrix multiplications without last column ...

           ! Diagonal block, the contribution of the last element is added below!
           ab(1,ne) = 0
           call DSYMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,0.d0,hd,1)

           ! Subdiagonal block
           if (nr>0) call DGEMV('N',nr,nb-1,tau,ab(nb+1,ns),2*nb-1,hv,1,0.d0,hs,1)

           ! ... then request last column ...
#ifdef WITH_OPENMP
           call mpi_recv(ab(1,ne),nb+1,mpi_real8,my_pe+1,1,mpi_comm,MPI_STATUS,mpierr)
#else
           call mpi_recv(ab(1,ne),nb+1,mpi_real8,my_pe+1,1,mpi_comm,MPI_STATUS_IGNORE,mpierr)
#endif

           ! ... and complete the result
           hs(1:nr) = hs(1:nr) + ab(2:nr+1,ne)*tau*hv(nb)
           hd(nb) = hd(nb) + ab(1,ne)*hv(nb)*tau

         else

           ! Normal matrix multiply
           call DSYMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,0.d0,hd,1)
           if (nr>0) call DGEMV('N',nr,nb,tau,ab(nb+1,ns),2*nb-1,hv,1,0.d0,hs,1)

         endif

         ! Calculate first column of subdiagonal block and calculate new
         ! Householder transformation for this column

         hv_new(:) = 0 ! Needed, last rows must be 0 for nr < nb
         tau_new = 0

         if (nr>0) then

           ! complete (old) Householder transformation for first column

           ab(nb+1:nb+nr,ns) = ab(nb+1:nb+nr,ns) - hs(1:nr) ! Note: hv(1) == 1

           ! calculate new Householder transformation ...
           if (nr>1) then
             vnorm2 = sum(ab(nb+2:nb+nr,ns)**2)
             call hh_transform_real(ab(nb+1,ns),vnorm2,hf,tau_new)
             hv_new(1) = 1.
             hv_new(2:nr) = ab(nb+2:nb+nr,ns)*hf
             ab(nb+2:,ns) = 0
           endif

           ! ... and send it away immediatly if this is the last block

           if (iblk==nblocks) then
#ifdef WITH_OPENMP
             call mpi_wait(ireq_hv,MPI_STATUS,mpierr)
#else
             call mpi_wait(ireq_hv,MPI_STATUS_IGNORE,mpierr)
#endif
             hv_s(1) = tau_new
             hv_s(2:) = hv_new(2:)
             call mpi_isend(hv_s,nb,mpi_real8,my_pe+1,2,mpi_comm,ireq_hv,mpierr)
           endif

         endif

         ! Transform diagonal block
         x = dot_product(hv(1:nc),hd(1:nc))*tau
         hd(1:nc) = hd(1:nc) - 0.5*x*hv(1:nc)

         if (my_pe>0 .and. iblk==1) then

           ! The first column of the diagonal block has to be send to the previous PE
           ! Calculate first column only ...

           ab(1:nc,ns) = ab(1:nc,ns) - hd(1:nc)*hv(1) - hv(1:nc)*hd(1)

           ! ... send it away ...

#ifdef WITH_OPENMP
           call mpi_wait(ireq_ab,MPI_STATUS,mpierr)
#else
           call mpi_wait(ireq_ab,MPI_STATUS_IGNORE,mpierr)
#endif
           ab_s(1:nb+1) = ab(1:nb+1,ns)
           call mpi_isend(ab_s,nb+1,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)

           ! ... and calculate remaining columns with rank-2 update
           if (nc>1) call DSYR2('L',nc-1,-1.d0,hd(2),1,hv(2),1,ab(1,ns+1),2*nb-1)
         else
           ! No need to  send, just a rank-2 update
           call DSYR2('L',nc,-1.d0,hd,1,hv,1,ab(1,ns),2*nb-1)
         endif

         ! Do the remaining double Householder transformation on the subdiagonal block cols 2 ... nb

         if (nr>0) then
           if (nr>1) then
             call DGEMV('T',nr,nb-1,tau_new,ab(nb,ns+1),2*nb-1,hv_new,1,0.d0,h(2),1)
             x = dot_product(hs(1:nr),hv_new(1:nr))*tau_new
             h(2:nb) = h(2:nb) - x*hv(2:nb)
             ! Unfortunately there is no BLAS routine like DSYR2 for a nonsymmetric rank 2 update
             do i=2,nb
               ab(2+nb-i:1+nb+nr-i,i+ns-1) = ab(2+nb-i:1+nb+nr-i,i+ns-1) - hv_new(1:nr)*h(i) - hs(1:nr)*hv(i)
             enddo
           else
             ! No double Householder transformation for nr=1, just complete the row
             do i=2,nb
               ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*hv(i)
             enddo
           endif
         endif

         ! Use new HH vector for the next block
         hv(:) = hv_new(:)
         tau = tau_new

       enddo

#ifdef WITH_OPENMP
     endif


     do iblk = 1, nblocks

      if (hh_dst(iblk) >= np_rows) exit
      if (snd_limits(hh_dst(iblk)+1,iblk) == snd_limits(hh_dst(iblk),iblk)) exit

      if (hh_cnt(iblk) == snd_limits(hh_dst(iblk)+1,iblk)-snd_limits(hh_dst(iblk),iblk)) then
        ! Wait for last transfer to finish
        call mpi_wait(ireq_hhs(iblk), mpi_status, mpierr)
        ! Copy vectors into send buffer
        hh_send(:,1:hh_cnt(iblk),iblk) = hh_gath(:,1:hh_cnt(iblk),iblk)
        ! Send to destination
        call mpi_isend(hh_send(1,1,iblk), nb*hh_cnt(iblk), mpi_real8, &
              global_id(hh_dst(iblk),mod(iblk+block_limits(my_pe)-1,np_cols)), &
              10+iblk, mpi_comm, ireq_hhs(iblk), mpierr)
        ! Reset counter and increase destination row
        hh_cnt(iblk) = 0
        hh_dst(iblk) = hh_dst(iblk)+1
      endif

    enddo
#endif
  enddo

  ! Finish the last outstanding requests
#ifdef WITH_OPENMP
  call mpi_wait(ireq_ab,MPI_STATUS,mpierr)
  call mpi_wait(ireq_hv,MPI_STATUS,mpierr)

  allocate(mpi_statuses(MPI_STATUS_SIZE,max(nblocks,num_chunks)), stat=istat, errmsg=errorMessage)
  if (istat .ne. 0) then
    print *,"tridiag_band_real: error when allocating mpi_statuses"//errorMessage
    stop
  endif

  call mpi_waitall(nblocks, ireq_hhs, MPI_STATUSES, mpierr)
  call mpi_waitall(num_chunks, ireq_hhr, MPI_STATUSES, mpierr)
  deallocate(mpi_statuses, stat=istat, errmsg=errorMessage)
  if (istat .ne. 0) then
    print *,"tridiag_band_real: error when deallocating mpi_statuses"//errorMessage
    stop
  endif

#else
  call mpi_wait(ireq_ab,MPI_STATUS_IGNORE,mpierr)
  call mpi_wait(ireq_hv,MPI_STATUS_IGNORE,mpierr)

  call mpi_waitall(nblocks, ireq_hhs, MPI_STATUSES_IGNORE, mpierr)
  call mpi_waitall(num_chunks, ireq_hhr, MPI_STATUSES_IGNORE, mpierr)
#endif

  call mpi_barrier(mpi_comm,mpierr)

  deallocate(ab, stat=istat, errmsg=errorMessage)
  if (istat .ne. 0) then
    print *,"tridiag_band_real: error when deallocating ab"//errorMessage
    stop
  endif

  deallocate(ireq_hhr, ireq_hhs, stat=istat, errmsg=errorMessage)
  if (istat .ne. 0) then
    print *,"tridiag_band_real: error when deallocating ireq_hhr, ireq_hhs"//errorMessage
    stop
  endif

  deallocate(hh_cnt, hh_dst, stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when deallocating hh_cnt, hh_dst"//errorMessage
     stop
   endif

  deallocate(hh_gath, hh_send, stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when deallocating hh_gath, hh_send"//errorMessage
     stop
   endif

  deallocate(limits, snd_limits, stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when deallocating limits, send_limits"//errorMessage
     stop
   endif

  deallocate(block_limits, stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when deallocating block_limits"//errorMessage
     stop
   endif

  deallocate(global_id, stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_real: error when allocating global_id"//errorMessage
     stop
   endif

#ifdef HAVE_DETAILED_TIMINGS
  call timer%stop("tridiag_band_real")
#endif

 end subroutine tridiag_band_real

! --------------------------------------------------------------------------------------------------


subroutine trans_ev_tridi_to_band_real(na, nev, nblk, nbw, q, ldq, matrixCols, &
                                       mpi_comm_rows, mpi_comm_cols, wantDebug, success, &
                                       THIS_REAL_ELPA_KERNEL)
!-------------------------------------------------------------------------------
!  trans_ev_tridi_to_band_real:
!  Transforms the eigenvectors of a tridiagonal matrix back to the eigenvectors of the band matrix
!
!  Parameters
!
!  na          Order of matrix a, number of rows of matrix q
!
!  nev         Number eigenvectors to compute (= columns of matrix q)
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  nb          semi bandwith
!
!  q           On input: Eigenvectors of tridiagonal matrix
!              On output: Transformed eigenvectors
!              Distribution is like in Scalapack.
!
!  ldq         Leading dimension of q
!  matrixCols  local columns of matrix q
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns/both
!
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
    implicit none

    integer, intent(in) :: THIS_REAL_ELPA_KERNEL
    integer, intent(in) :: na, nev, nblk, nbw, ldq, matrixCols, mpi_comm_rows, mpi_comm_cols
    real*8              :: q(ldq,matrixCols)

    integer np_rows, my_prow, np_cols, my_pcol

    integer i, j, ip, sweep, nbuf, l_nev, a_dim2
    integer current_n, current_local_n, current_n_start, current_n_end
    integer next_n, next_local_n, next_n_start, next_n_end
    integer bottom_msg_length, top_msg_length, next_top_msg_length
    integer stripe_width, last_stripe_width, stripe_count
#ifdef WITH_OPENMP
    integer thread_width, csw, b_off, b_len
#endif
    integer num_result_blocks, num_result_buffers, num_bufs_recvd
    integer a_off, current_tv_off, max_blk_size
    integer mpierr, src, src_offset, dst, offset, nfact, num_blk
#ifdef WITH_OPENMP
    integer mpi_status(MPI_STATUS_SIZE)
#endif
    logical flag

#ifdef WITH_OPENMP
    real*8, allocatable :: a(:,:,:,:), row(:)
#else
    real*8, allocatable :: a(:,:,:), row(:)
#endif

#ifdef WITH_GPU_VERSION
    real*8, allocatable :: row_group(:,:)

#endif

#ifdef WITH_OPENMP
    real*8, allocatable :: top_border_send_buffer(:,:), top_border_recv_buffer(:,:)
    real*8, allocatable :: bottom_border_send_buffer(:,:), bottom_border_recv_buffer(:,:)
#else
    real*8, allocatable :: top_border_send_buffer(:,:,:), top_border_recv_buffer(:,:,:)
    real*8, allocatable :: bottom_border_send_buffer(:,:,:), bottom_border_recv_buffer(:,:,:)
#endif
    real*8, allocatable :: result_buffer(:,:,:)
    real*8, allocatable :: bcast_buffer(:,:)
    integer             :: tmp
#ifdef WITH_GPU_VERSION
!    real*8, allocatable, device :: a_dev(:,:,:)
!    real*8, allocatable, device :: bcast_buffer_dev(:,:)
!    real*8, allocatable, device :: row_dev(:)
!    real*8, allocatable, device :: row_group_dev(:,:)
!    real*8, allocatable, device :: hh_dot_dev(:)
!    real*8, allocatable, device :: hh_tau_dev(:)

    integer(c_size_t)  :: a_dev
    integer(c_size_t)  :: bcast_buffer_dev
    integer(c_size_t)  :: num
    integer(c_size_t)  :: dev_offset, dev_offset_1


    integer(c_size_t)  :: row_dev
    integer(c_size_t)  :: row_group_dev
    integer(c_size_t)  :: hh_dot_dev
    integer(c_size_t)  :: hh_tau_dev

    Integer            :: top, chunk, this_chunk
    integer            :: row_group_size, unpack_idx
#endif

    integer              :: n_off
    integer, allocatable :: result_send_request(:), result_recv_request(:), limits(:)
    integer, allocatable :: top_send_request(:), bottom_send_request(:)
    integer, allocatable :: top_recv_request(:), bottom_recv_request(:)
#ifdef WITH_OPENMP
    integer, allocatable :: mpi_statuses(:,:)
#endif
    ! MPI send/recv tags, arbitrary

    integer, parameter  :: bottom_recv_tag = 111
    integer, parameter  :: top_recv_tag    = 222
    integer, parameter  :: result_recv_tag = 333

    ! Just for measuring the kernel performance
    real*8              :: kernel_time
    integer*8           :: kernel_flops

#ifdef WITH_OPENMP
    integer             :: max_threads, my_thread
    integer             :: omp_get_max_threads
#endif

    logical, intent(in) :: wantDebug
    logical             :: success
    integer             :: istat
    character(200)      :: errorMessage


#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("trans_ev_tridi_to_band_real")
#endif

#ifdef WITH_GPU_VERSION
    unpack_idx = 0
    row_group_size = 0
#endif
    success = .true.
    kernel_time = 1.d-100
    kernel_flops = 0

#ifdef WITH_OPENMP
    max_threads = 1
    max_threads = omp_get_max_threads()
#endif

    call MPI_Comm_rank(mpi_comm_rows, my_prow, mpierr)
    call MPI_Comm_size(mpi_comm_rows, np_rows, mpierr)
    call MPI_Comm_rank(mpi_comm_cols, my_pcol, mpierr)
    call MPI_Comm_size(mpi_comm_cols, np_cols, mpierr)

    if (mod(nbw,nblk)/=0) then
      if (my_prow==0 .and. my_pcol==0) then
        if (wantDebug) then
          write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_real: ERROR: nbw=',nbw,', nblk=',nblk
          write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_real: band backtransform works only for nbw==n*nblk'
        endif
        success = .false.
        return
      endif
    endif

    nfact = nbw / nblk


    ! local number of eigenvectors
    l_nev = local_index(nev, my_pcol, np_cols, nblk, -1)

    if (l_nev==0) then
#ifdef WITH_OPENMP
      thread_width = 0
#endif
      stripe_width = 0
      stripe_count = 0
      last_stripe_width = 0
    else

      ! Suggested stripe width is 48 since 48*64 real*8 numbers should fit into
      ! every primary cache
#ifndef WITH_GPU_VERSION

#ifdef WITH_OPENMP
      thread_width = (l_nev-1)/max_threads + 1 ! number of eigenvectors per OMP thread
#endif
      stripe_width = 48 ! Must be a multiple of 4
#ifdef WITH_OPENMP
      stripe_count = (thread_width-1)/stripe_width + 1
#else
      stripe_count = (l_nev-1)/stripe_width + 1
#endif
      ! Adapt stripe width so that last one doesn't get too small
#ifdef WITH_OPENMP
      stripe_width = (thread_width-1)/stripe_count + 1
#else
      stripe_width = (l_nev-1)/stripe_count + 1
#endif
      stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 4 !!!

#else /* WITH_GPU_VERSION */
      stripe_width = 256 ! Must be a multiple of 4
      stripe_count = (l_nev - 1) / stripe_width + 1
#endif /* WITH_GPU_VERSION */


      last_stripe_width = l_nev - (stripe_count-1)*stripe_width
    endif

    ! Determine the matrix distribution at the beginning

    allocate(limits(0:np_rows), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating limits"//errorMessage
      stop
    endif
   call determine_workload(na, nbw, np_rows, limits)

    max_blk_size = maxval(limits(1:np_rows) - limits(0:np_rows-1))

    a_dim2 = max_blk_size + nbw

#ifdef WITH_GPU_VERSION
    num =  (stripe_width*a_dim2*stripe_count)*8_8
    istat = cuda_malloc(a_dev, stripe_width*a_dim2*stripe_count*8_8)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error in cudaMalloc"//errorMessage
      stop
    endif

    istat = cuda_memset(a_dev , 0, num)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error in cudaMemset"//errorMessage
      stop
    endif

#else /* WITH_GPU_VERSION */

!DEC$ ATTRIBUTES ALIGN: 64:: a
#ifdef WITH_OPENMP
    allocate(a(stripe_width,a_dim2,stripe_count,max_threads), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating a"//errorMessage
      stop
    endif

    ! a(:,:,:,:) should be set to 0 in a parallel region, not here!
#else
    allocate(a(stripe_width,a_dim2,stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating a"//errorMessage
      stop
    endif

    a(:,:,:) = 0
#endif

#endif /* WITH_GPU_VERSION */

    allocate(row(l_nev), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating row"//errorMessage
      stop
    endif

    row(:) = 0

#ifdef WITH_GPU_VERSION
    num =  (l_nev)*8_8
    istat = cuda_malloc( row_dev,l_nev*8_8)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error in cudaMalloc "//errorMessage
      stop
    endif

    istat = cuda_memset(row_dev , 0, num)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error in cudaMemset"//errorMessage
      stop
    endif

    ! "row_group" and "row_group_dev" are needed for GPU optimizations
    allocate(row_group(l_nev, nblk), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating row_group"//errorMessage
      stop
    endif
    row_group(:, :) = 0

    num =  (l_nev*nblk)*8_8
!    call cuda_malloc2d( row_group_dev,l_nev*8_8,nblk*8_8)
    istat = cuda_malloc(row_group_dev, l_nev*nblk*8_8)
   if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error in cudaMalloc"//errorMessage
      stop
    endif
    istat = cuda_memset(row_group_dev , 0, num)
   if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error in cudaMemset"//errorMessage
      stop
    endif
#endif

    ! Copy q from a block cyclic distribution into a distribution with contiguous rows,
    ! and transpose the matrix using stripes of given stripe_width for cache blocking.

    ! The peculiar way it is done below is due to the fact that the last row should be
    ! ready first since it is the first one to start below

#ifdef WITH_OPENMP
    ! Please note about the OMP usage below:
    ! This is not for speed, but because we want the matrix a in the memory and
    ! in the cache of the correct thread (if possible)
#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
    do my_thread = 1, max_threads
      a(:,:,:,my_thread) = 0 ! if possible, do first touch allocation!
    enddo
    !$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("OpenMP parallel")
#endif
#endif

   do ip = np_rows-1, 0, -1
     if (my_prow == ip) then
       ! Receive my rows which have not yet been received
       src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
       do i=limits(ip)+1,limits(ip+1)
         src = mod((i-1)/nblk, np_rows)
         if (src < my_prow) then
#ifdef WITH_OPENMP

#ifdef WITH_GPU_VERSION
           print *,"trans_ev_tridi_to_band_real: not yet implemented"
           stop
#endif
           call MPI_Recv(row, l_nev, MPI_REAL8, src, 0, mpi_comm_rows, MPI_STATUS, mpierr)
#ifdef HAVE_DETAILED_TIMINGS
           call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
           do my_thread = 1, max_threads
             call unpack_row(row,i-limits(ip),my_thread)
           enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
           call timer%stop("OpenMP parallel")
#endif

#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
           ! An unpacking of the current row group may occur before queuing the next row
           call unpack_and_prepare_row_group(i - limits(ip), .false.)
           call MPI_Recv(row_group(:, row_group_size), l_nev, MPI_REAL8, src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)
#else
           call MPI_Recv(row, l_nev, MPI_REAL8, src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)
           call unpack_row(row,i-limits(ip))
#endif /* WITH_GPU_VERSION */

#endif /* WITH_OPENMP */
         elseif (src==my_prow) then
           src_offset = src_offset+1
#ifndef WITH_GPU_VERSION
           row(:) = q(src_offset, 1:l_nev)
#endif

#ifdef WITH_OPENMP

#ifdef HAVE_DETAILED_TIMINGS
           call timer%start("OpenMP parallel")
#endif

#ifdef WITH_GPU_VERSION
           print *,"trans_ev_tridi_to_band_real: not yet implemented"
           stop
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
           do my_thread = 1, max_threads
             call unpack_row(row,i-limits(ip),my_thread)
           enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
           call timer%stop("OpenMP parallel")
#endif

#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
           ! An unpacking of the current row group may occur before queuing the next row
           call unpack_and_prepare_row_group(i - limits(ip), .false.)
           row_group(:, row_group_size) = q(src_offset, 1:l_nev)
#else
           call unpack_row(row,i-limits(ip))
#endif
#endif /* WITH_OPENMP */

         endif
       enddo
       ! Send all rows which have not yet been send
       src_offset = 0
       do dst = 0, ip-1
         do i=limits(dst)+1,limits(dst+1)
           if (mod((i-1)/nblk, np_rows) == my_prow) then
             src_offset = src_offset+1
             row(:) = q(src_offset, 1:l_nev)
             call MPI_Send(row, l_nev, MPI_REAL8, dst, 0, mpi_comm_rows, mpierr)
           endif
         enddo
       enddo
     else if (my_prow < ip) then
       ! Send all rows going to PE ip
       src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
       do i=limits(ip)+1,limits(ip+1)
         src = mod((i-1)/nblk, np_rows)
         if (src == my_prow) then
           src_offset = src_offset+1
           row(:) = q(src_offset, 1:l_nev)
           call MPI_Send(row, l_nev, MPI_REAL8, ip, 0, mpi_comm_rows, mpierr)
         endif
       enddo
       ! Receive all rows from PE ip
       do i=limits(my_prow)+1,limits(my_prow+1)
         src = mod((i-1)/nblk, np_rows)
         if (src == ip) then
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
           print *,"trans_ev_tridi_to_band_real: not yet implemented"
           stop
#endif
           call MPI_Recv(row, l_nev, MPI_REAL8, src, 0, mpi_comm_rows, MPI_STATUS, mpierr)
#ifdef HAVE_DETAILED_TIMINGS
           call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
           do my_thread = 1, max_threads
             call unpack_row(row,i-limits(my_prow),my_thread)
           enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
           call timer%stop("OpenMP parallel")
#endif

#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
           ! An unpacking of the current row group may occur before queuing the next row
           call unpack_and_prepare_row_group(i - limits(my_prow), .false.)
           call MPI_Recv(row_group(:, row_group_size), l_nev, MPI_REAL8, src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)
#else
           call MPI_Recv(row, l_nev, MPI_REAL8, src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)
           call unpack_row(row,i-limits(my_prow))
#endif

#endif /* WITH_OPENMP */

         endif
       enddo
     endif
   enddo

#ifdef WITH_GPU_VERSION
   ! Force an unpacking of all remaining rows that haven't been unpacked yet
   call unpack_and_prepare_row_group(-1, .true.)
   istat = cuda_devicesynchronize()

    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error in cudaDeviceSynchronize"//errorMessage
      stop
    endif

#endif

   ! Set up result buffer queue

   num_result_blocks = ((na-1)/nblk + np_rows - my_prow) / np_rows

   num_result_buffers = 4*nfact
   allocate(result_buffer(l_nev,nblk,num_result_buffers), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"trans_ev_tridi_to_band_real: error when allocating result_buffer"//errorMessage
     stop
   endif

   allocate(result_send_request(num_result_buffers), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"trans_ev_tridi_to_band_real: error when allocating result_send_request"//errorMessage
     stop
   endif

   allocate(result_recv_request(num_result_buffers), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"trans_ev_tridi_to_band_real: error when allocating result_recv_request"//errorMessage
     stop
   endif

   result_send_request(:) = MPI_REQUEST_NULL
   result_recv_request(:) = MPI_REQUEST_NULL

   ! Queue up buffers

   if (my_prow > 0 .and. l_nev>0) then ! note: row 0 always sends
     do j = 1, min(num_result_buffers, num_result_blocks)
       call MPI_Irecv(result_buffer(1,1,j), l_nev*nblk, MPI_REAL8, 0, result_recv_tag, &
                           mpi_comm_rows, result_recv_request(j), mpierr)
     enddo
   endif

   num_bufs_recvd = 0 ! No buffers received yet

   ! Initialize top/bottom requests

   allocate(top_send_request(stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating top_send_request"//errorMessage
      stop
    endif

   allocate(top_recv_request(stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating top_recv_request"//errorMessage
      stop
    endif

   allocate(bottom_send_request(stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating bottom_send_request"//errorMessage
      stop
    endif

   allocate(bottom_recv_request(stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating bottom_recv_request"//errorMessage
      stop
    endif

   top_send_request(:) = MPI_REQUEST_NULL
   top_recv_request(:) = MPI_REQUEST_NULL
   bottom_send_request(:) = MPI_REQUEST_NULL
   bottom_recv_request(:) = MPI_REQUEST_NULL

#ifdef WITH_OPENMP
   allocate(top_border_send_buffer(stripe_width*nbw*max_threads, stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating top_border_send_buffer"//errorMessage
      stop
    endif

   allocate(top_border_recv_buffer(stripe_width*nbw*max_threads, stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating top_border_recv_buffer"//errorMessage
      stop
    endif

   allocate(bottom_border_send_buffer(stripe_width*nbw*max_threads, stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating bottom_border_send_buffer"//errorMessage
      stop
    endif

   allocate(bottom_border_recv_buffer(stripe_width*nbw*max_threads, stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating bottom_border_recv_buffer"//errorMessage
      stop
    endif
   top_border_send_buffer(:,:) = 0
   top_border_recv_buffer(:,:) = 0
   bottom_border_send_buffer(:,:) = 0
   bottom_border_recv_buffer(:,:) = 0

   ! Initialize broadcast buffer
#else
   allocate(top_border_send_buffer(stripe_width, nbw, stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating top_border_send_bufer"//errorMessage
      stop
    endif

   allocate(top_border_recv_buffer(stripe_width, nbw, stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating top_border_recv_buffer"//errorMessage
      stop
    endif

   allocate(bottom_border_send_buffer(stripe_width, nbw, stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating bottom_border_send_buffer"//errorMessage
      stop
    endif

   allocate(bottom_border_recv_buffer(stripe_width, nbw, stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating bottom_border_recv_buffer"//errorMessage
      stop
    endif

   top_border_send_buffer(:,:,:) = 0
   top_border_recv_buffer(:,:,:) = 0
   bottom_border_send_buffer(:,:,:) = 0
   bottom_border_recv_buffer(:,:,:) = 0
#endif

   allocate(bcast_buffer(nbw, max_blk_size), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error when allocating bcast_buffer"//errorMessage
      stop
    endif

   bcast_buffer = 0

#ifdef WITH_GPU_VERSION
   num =  ( nbw * max_blk_size) * 8_8
   istat = cuda_malloc(bcast_buffer_dev, nbw * max_blk_size * 8_8)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error in cudaMalloc"
      stop
    endif

   istat = cuda_memset( bcast_buffer_dev, 0, num)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error in cudaMemset"
      stop
    endif

   num =  ((max_blk_size-1))*8_8
   istat = cuda_malloc( hh_dot_dev, (max_blk_size -1) * 8_8)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error in cudaMalloc"
      stop
    endif

   istat = cuda_memset( hh_dot_dev, 0, num)
   if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error in cudaMemset"
      stop
    endif

   num =  (max_blk_size)*8_8
   istat = cuda_malloc( hh_tau_dev, max_blk_size * 8_8)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error in cudaMalloc"
      stop
    endif

   istat = cuda_memset( hh_tau_dev, 0, num)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_real: error in cudaMemset"
      stop
    endif

#endif

   current_tv_off = 0 ! Offset of next row to be broadcast


    ! ------------------- start of work loop -------------------

   a_off = 0 ! offset in A (to avoid unnecessary shifts)

   top_msg_length = 0
   bottom_msg_length = 0

   do sweep = 0, (na-1)/nbw

     current_n = na - sweep*nbw
     call determine_workload(current_n, nbw, np_rows, limits)
     current_n_start = limits(my_prow)
     current_n_end   = limits(my_prow+1)
     current_local_n = current_n_end - current_n_start

     next_n = max(current_n - nbw, 0)
     call determine_workload(next_n, nbw, np_rows, limits)
     next_n_start = limits(my_prow)
     next_n_end   = limits(my_prow+1)
     next_local_n = next_n_end - next_n_start

     if (next_n_end < next_n) then
       bottom_msg_length = current_n_end - next_n_end
     else
       bottom_msg_length = 0
     endif

     if (next_local_n > 0) then
       next_top_msg_length = current_n_start - next_n_start
     else
       next_top_msg_length = 0
     endif

     if (sweep==0 .and. current_n_end < current_n .and. l_nev > 0) then
       do i = 1, stripe_count
#ifdef WITH_OPENMP

#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif
         csw = min(stripe_width, thread_width-(i-1)*stripe_width) ! "current_stripe_width"
         b_len = csw*nbw*max_threads
         call MPI_Irecv(bottom_border_recv_buffer(1,i), b_len, MPI_REAL8, my_prow+1, bottom_recv_tag, &
                           mpi_comm_rows, bottom_recv_request(i), mpierr)
#else
         call MPI_Irecv(bottom_border_recv_buffer(1,1,i), nbw*stripe_width, MPI_REAL8, my_prow+1, bottom_recv_tag, &
                        mpi_comm_rows, bottom_recv_request(i), mpierr)
#endif
       enddo
     endif

     if (current_local_n > 1) then
       if (my_pcol == mod(sweep,np_cols)) then
         bcast_buffer(:,1:current_local_n) = hh_trans_real(:,current_tv_off+1:current_tv_off+current_local_n)
         current_tv_off = current_tv_off + current_local_n
       endif
       call mpi_bcast(bcast_buffer, nbw*current_local_n, MPI_REAL8, mod(sweep,np_cols), mpi_comm_cols, mpierr)

#ifdef WITH_GPU_VERSION
       istat =  cuda_memcpy(bcast_buffer_dev, loc(bcast_buffer(1,1)), nbw * current_local_n * 8_8 , 1)
       if (istat .ne. 0) then
         print *,"trans_ev_tridi_to_band_real: error in cudaMemcpy"
         stop
       endif

       call extract_hh_tau(nbw, current_local_n, .false.)
       call compute_hh_dot_products_real(nbw, current_local_n)
#endif
     else
       ! for current_local_n == 1 the one and only HH vector is 0 and not stored in hh_trans_real
       bcast_buffer(:,1) = 0
#ifdef WITH_GPU_VERSION
       istat = cuda_memset(bcast_buffer_dev, 0, nbw * 8_8)
       if (istat .ne. 0) then
         print *,"trans_ev_tridi_to_band_real: error in cudaMemset"
         stop
       endif

       call extract_hh_tau(nbw, 1, .true.)
#endif
     endif

     if (l_nev == 0) cycle

     if (current_local_n > 0) then

       do i = 1, stripe_count
#ifdef WITH_OPENMP

#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
#endif
         ! Get real stripe width for strip i;
         ! The last OpenMP tasks may have an even smaller stripe with,
         ! but we don't care about this, i.e. we send/recv a bit too much in this case.
         ! csw: current_stripe_width

         csw = min(stripe_width, thread_width-(i-1)*stripe_width)
#endif /* WITH_OPENMP */

         !wait_b
         if (current_n_end < current_n) then
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif

           call MPI_Wait(bottom_recv_request(i), MPI_STATUS, mpierr)
#ifdef HAVE_DETAILED_TIMINGS
           call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, n_off, b_len, b_off), schedule(static, 1)
           do my_thread = 1, max_threads
             n_off = current_local_n+a_off
             b_len = csw*nbw
             b_off = (my_thread-1)*b_len
             a(1:csw,n_off+1:n_off+nbw,i,my_thread) = &
               reshape(bottom_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, nbw /))
           enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
           call timer%stop("OpenMP parallel")
#endif

#else /* WITH_OPENMP */
           call MPI_Wait(bottom_recv_request(i), MPI_STATUS_IGNORE, mpierr)
           n_off = current_local_n+a_off
#ifdef WITH_GPU_VERSION
           dev_offset = (0 + (n_off * stripe_width) + ( (i-1) * stripe_width *a_dim2 )) *8_8
           istat =  cuda_memcpy( a_dev + dev_offset , loc(bottom_border_recv_buffer(1,1,i)) ,stripe_width*nbw*8_8 ,1)
           if (istat .ne. 0) then
             print *,"trans_ev_tridi_to_band_real: error in cudaMemcpy"
             stop
           endif

#else
           a(:,n_off+1:n_off+nbw,i) = bottom_border_recv_buffer(:,1:nbw,i)
#endif

#endif /* WITH_OPENMP */

           if (next_n_end < next_n) then
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif

             call MPI_Irecv(bottom_border_recv_buffer(1,i), csw*nbw*max_threads, &
                                   MPI_REAL8, my_prow+1, bottom_recv_tag, &
                                   mpi_comm_rows, bottom_recv_request(i), mpierr)
#else
             call MPI_Irecv(bottom_border_recv_buffer(1,1,i), nbw*stripe_width, MPI_REAL8, my_prow+1, bottom_recv_tag, &
                                   mpi_comm_rows, bottom_recv_request(i), mpierr)
#endif
           endif
         endif

         if (current_local_n <= bottom_msg_length + top_msg_length) then

           !wait_t
           if (top_msg_length>0) then
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif
             call MPI_Wait(top_recv_request(i), MPI_STATUS, mpierr)
#else
             call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)

#ifdef WITH_GPU_VERSION
             dev_offset = (0 + (a_off * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) *8_8
!             host_offset= (0 + (0 * stripe_width) + ( (i-1) * stripe_width * nbw ) ) * 8
             istat =  cuda_memcpy( a_dev+dev_offset , loc(top_border_recv_buffer(1,1,i)),stripe_width*top_msg_length*8_8 ,1)
             if (istat .ne. 0) then
               print *,"trans_ev_tridi_to_band_real: error in cudaMemcpy"
               stop
              endif

#else
             a(:,a_off+1:a_off+top_msg_length,i) = top_border_recv_buffer(:,1:top_msg_length,i)
#endif

#endif
           endif

           !compute
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
           call timer%start("OpenMP parallel")
#endif

#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif

!$omp parallel do private(my_thread, n_off, b_len, b_off), schedule(static, 1)
           do my_thread = 1, max_threads
             if (top_msg_length>0) then
               b_len = csw*top_msg_length
               b_off = (my_thread-1)*b_len
               a(1:csw,a_off+1:a_off+top_msg_length,i,my_thread) = &
                          reshape(top_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, top_msg_length /))
             endif
             call compute_hh_trafo_real(0, current_local_n, i, my_thread, &
                                      THIS_REAL_ELPA_KERNEL)
           enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
           call timer%stop("OpenMP parallel")
#endif

#else /* WITH_OPENMP */
           call compute_hh_trafo_real(0, current_local_n, i, &
                                      THIS_REAL_ELPA_KERNEL)
#endif /* WITH_OPENMP */

           !send_b
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif
           call MPI_Wait(bottom_send_request(i), mpi_status, mpierr)
           if (bottom_msg_length>0) then
             n_off = current_local_n+nbw-bottom_msg_length+a_off
             b_len = csw*bottom_msg_length*max_threads
             bottom_border_send_buffer(1:b_len,i) = &
                 reshape(a(1:csw,n_off+1:n_off+bottom_msg_length,i,:), (/ b_len /))
             call MPI_Isend(bottom_border_send_buffer(1,i), b_len, MPI_REAL8, my_prow+1, &
                            top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
           endif
#else
           call MPI_Wait(bottom_send_request(i), MPI_STATUS_IGNORE, mpierr)
           if (bottom_msg_length>0) then
             n_off = current_local_n+nbw-bottom_msg_length+a_off
#ifdef WITH_GPU_VERSION
             dev_offset = (0 + (n_off * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) *8_8
             istat =  cuda_memcpy( loc(bottom_border_send_buffer(1,1,i)), a_dev + dev_offset, &
                                   stripe_width * bottom_msg_length * 8_8 ,2)
             if (istat .ne. 0) then
               print *,"trans_ev_tridi_to_band_real: error in cudaMemcpy"
               stop
             endif

#else
             bottom_border_send_buffer(:,1:bottom_msg_length,i) = a(:,n_off+1:n_off+bottom_msg_length,i)
#endif
             call MPI_Isend(bottom_border_send_buffer(1,1,i), bottom_msg_length*stripe_width, MPI_REAL8, my_prow+1, &
                            top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
           endif
#endif
         else

         !compute
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif


#ifdef HAVE_DETAILED_TIMINGS
         call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, b_len, b_off), schedule(static, 1)
        do my_thread = 1, max_threads
          call compute_hh_trafo_real(current_local_n - bottom_msg_length, bottom_msg_length, i, my_thread, &
                              THIS_REAL_ELPA_KERNEL)
        enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("OpenMP parallel")
#endif

        !send_b
        call MPI_Wait(bottom_send_request(i), mpi_status, mpierr)
        if (bottom_msg_length > 0) then
          n_off = current_local_n+nbw-bottom_msg_length+a_off
          b_len = csw*bottom_msg_length*max_threads
          bottom_border_send_buffer(1:b_len,i) = &
              reshape(a(1:csw,n_off+1:n_off+bottom_msg_length,i,:), (/ b_len /))
          call MPI_Isend(bottom_border_send_buffer(1,i), b_len, MPI_REAL8, my_prow+1, &
                           top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
        endif
#else /* WITH_OPENMP */
        call compute_hh_trafo_real(current_local_n - bottom_msg_length, bottom_msg_length, i, &
                                      THIS_REAL_ELPA_KERNEL)

        !send_b
        call MPI_Wait(bottom_send_request(i), MPI_STATUS_IGNORE, mpierr)
        if (bottom_msg_length > 0) then
          n_off = current_local_n+nbw-bottom_msg_length+a_off
#ifdef WITH_GPU_VERSION
          dev_offset = (0 + (n_off * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) *8_8
          istat =  cuda_memcpy( loc(bottom_border_send_buffer(1,1,i)), a_dev + dev_offset,stripe_width*bottom_msg_length*8_8 ,2)
          if (istat .ne. 0) then
            print *,"trans_ev_tridi_to_band_real: error cudaMemcpy"
            stop
          endif


#else
          bottom_border_send_buffer(:,1:bottom_msg_length,i) = a(:,n_off+1:n_off+bottom_msg_length,i)
#endif
          call MPI_Isend(bottom_border_send_buffer(1,1,i), bottom_msg_length*stripe_width, MPI_REAL8, my_prow+1, &
                         top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
        endif
#endif /* WITH_OPENMP */

        !compute
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif

#ifdef HAVE_DETAILED_TIMINGS
        call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
        do my_thread = 1, max_threads
          call compute_hh_trafo_real(top_msg_length, current_local_n-top_msg_length-bottom_msg_length, i, my_thread, &
                                THIS_REAL_ELPA_KERNEL)
        enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("OpenMP parallel")
#endif

#else /* WITH_OPENMP */
        call compute_hh_trafo_real(top_msg_length, current_local_n-top_msg_length-bottom_msg_length, i, &
                              THIS_REAL_ELPA_KERNEL)

#endif /* WITH_OPENMP */

        !wait_t
        if (top_msg_length>0) then
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif

          call MPI_Wait(top_recv_request(i), mpi_status, mpierr)
#else /* WITH_OPENMP */
          call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)
#ifdef WITH_GPU_VERSION
          dev_offset = (0 + (a_off * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) *8_8
          istat =  cuda_memcpy( a_dev + dev_offset , loc( top_border_recv_buffer(:,1,i)), stripe_width * top_msg_length *8_8 ,1)
          if (istat .ne. 0) then
            print *,"trans_ev_tridi_to_band_real: error in cudaMemcpy"
            stop
          endif
#else
          a(:,a_off+1:a_off+top_msg_length,i) = top_border_recv_buffer(:,1:top_msg_length,i)
#endif

#endif /* WITH_OPENMP */

        endif

        !compute
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif

#ifdef HAVE_DETAILED_TIMINGS
        call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, b_len, b_off), schedule(static, 1)
        do my_thread = 1, max_threads
          if (top_msg_length>0) then
            b_len = csw*top_msg_length
            b_off = (my_thread-1)*b_len
            a(1:csw,a_off+1:a_off+top_msg_length,i,my_thread) = &
              reshape(top_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, top_msg_length /))
          endif
          call compute_hh_trafo_real(0, top_msg_length, i, my_thread, THIS_REAL_ELPA_KERNEL)
        enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("OpenMP parallel")
#endif

#else /* WITH_OPENMP */
        call compute_hh_trafo_real(0, top_msg_length, i, THIS_REAL_ELPA_KERNEL)
#endif /* WITH_OPENMP */

      endif

      if (next_top_msg_length > 0) then
        !request top_border data
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif

        b_len = csw*next_top_msg_length*max_threads
        call MPI_Irecv(top_border_recv_buffer(1,i), b_len, MPI_REAL8, my_prow-1, &
                       top_recv_tag, mpi_comm_rows, top_recv_request(i), mpierr)
#else /* WITH_OPENMP */
        call MPI_Irecv(top_border_recv_buffer(1,1,i), next_top_msg_length*stripe_width, MPI_REAL8, my_prow-1, &
                       top_recv_tag, mpi_comm_rows, top_recv_request(i), mpierr)
#endif /* WITH_OPENMP */

      endif

      !send_t
      if (my_prow > 0) then
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif

        call MPI_Wait(top_send_request(i), mpi_status, mpierr)
        b_len = csw*nbw*max_threads
        top_border_send_buffer(1:b_len,i) = reshape(a(1:csw,a_off+1:a_off+nbw,i,:), (/ b_len /))
        call MPI_Isend(top_border_send_buffer(1,i), b_len, MPI_REAL8, &
                       my_prow-1, bottom_recv_tag, &
                       mpi_comm_rows, top_send_request(i), mpierr)
#else /* WITH_OPENMP */

        call MPI_Wait(top_send_request(i), MPI_STATUS_IGNORE, mpierr)

#ifdef WITH_GPU_VERSION
        dev_offset = (0 + (a_off * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) * 8_8
        istat =  cuda_memcpy( loc(top_border_send_buffer(:,1,i)), a_dev + dev_offset, stripe_width*nbw*8_8 ,2)
        if (istat .ne. 0) then
          print *,"trans_ev_tridi_to_band_real: error in cudaMemcpy"
          stop
        endif

#else
        top_border_send_buffer(:,1:nbw,i) = a(:,a_off+1:a_off+nbw,i)
#endif
        call MPI_Isend(top_border_send_buffer(1,1,i), nbw*stripe_width, MPI_REAL8, my_prow-1, bottom_recv_tag, &
                       mpi_comm_rows, top_send_request(i), mpierr)

#endif /* WITH_OPENMP */

      endif

      ! Care that there are not too many outstanding top_recv_request's
      if (stripe_count > 1) then
        if (i>1) then
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif

          call MPI_Wait(top_recv_request(i-1), MPI_STATUS, mpierr)
#else
          call MPI_Wait(top_recv_request(i-1), MPI_STATUS_IGNORE, mpierr)
#endif
        else
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif

          call MPI_Wait(top_recv_request(stripe_count), MPI_STATUS, mpierr)
#else
          call MPI_Wait(top_recv_request(stripe_count), MPI_STATUS_IGNORE, mpierr)
#endif
        endif
      endif

    enddo

    top_msg_length = next_top_msg_length

  else
    ! wait for last top_send_request
    do i = 1, stripe_count
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif

      call MPI_Wait(top_send_request(i), MPI_STATUS, mpierr)
#else
      call MPI_Wait(top_send_request(i), MPI_STATUS_IGNORE, mpierr)
#endif
      enddo
    endif

    ! Care about the result

    if (my_prow == 0) then

      ! topmost process sends nbw rows to destination processes

      do j=0,nfact-1
        num_blk = sweep*nfact+j ! global number of destination block, 0 based
        if (num_blk*nblk >= na) exit

          nbuf = mod(num_blk, num_result_buffers) + 1 ! buffer number to get this block

#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif

          call MPI_Wait(result_send_request(nbuf), MPI_STATUS, mpierr)
#else
          call MPI_Wait(result_send_request(nbuf), MPI_STATUS_IGNORE, mpierr)
#endif
          dst = mod(num_blk, np_rows)

          if (dst == 0) then
#ifdef WITH_GPU_VERSION
            row_group_size = min(na - num_blk*nblk, nblk)
            call pack_row_group(row_group(:, :), j * nblk + a_off, row_group_size)

            do i = 1, row_group_size
              q((num_blk / np_rows) * nblk + i, 1 : l_nev) = row_group(:, i)
            enddo
#else
            do i = 1, min(na - num_blk*nblk, nblk)
              call pack_row(row, j*nblk+i+a_off)
              q((num_blk/np_rows)*nblk+i,1:l_nev) = row(:)
            enddo
#endif
          else
#ifdef WITH_GPU_VERSION
            call pack_row_group(result_buffer(:, :, nbuf), j * nblk + a_off, nblk)
#else
            do i = 1, nblk
              call pack_row(result_buffer(:,i,nbuf),j*nblk+i+a_off)
            enddo
#endif
            call MPI_Isend(result_buffer(1,1,nbuf), l_nev*nblk, MPI_REAL8, dst, &
                                    result_recv_tag, mpi_comm_rows, result_send_request(nbuf), mpierr)

          endif
        enddo

      else

        ! receive and store final result

        do j = num_bufs_recvd, num_result_blocks-1

          nbuf = mod(j, num_result_buffers) + 1 ! buffer number to get this block

          ! If there is still work to do, just test for the next result request
          ! and leave the loop if it is not ready, otherwise wait for all
          ! outstanding requests

          if (next_local_n > 0) then
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif

            call MPI_Test(result_recv_request(nbuf), flag, MPI_STATUS, mpierr)
#else
            call MPI_Test(result_recv_request(nbuf), flag, MPI_STATUS_IGNORE, mpierr)

#endif
            if (.not.flag) exit
          else
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif

            call MPI_Wait(result_recv_request(nbuf), MPI_STATUS, mpierr)
#else
            call MPI_Wait(result_recv_request(nbuf), MPI_STATUS_IGNORE, mpierr)
#endif
          endif

          ! Fill result buffer into q
           num_blk = j*np_rows + my_prow ! global number of current block, 0 based
           do i = 1, min(na - num_blk*nblk, nblk)
             q(j*nblk+i, 1:l_nev) = result_buffer(1:l_nev, i, nbuf)
           enddo

           ! Queue result buffer again if there are outstanding blocks left
           if (j+num_result_buffers < num_result_blocks) &
                     call MPI_Irecv(result_buffer(1,1,nbuf), l_nev*nblk, MPI_REAL8, 0, result_recv_tag, &
                                    mpi_comm_rows, result_recv_request(nbuf), mpierr)

         enddo
         num_bufs_recvd = j

       endif

       ! Shift the remaining rows to the front of A (if necessary)

       offset = nbw - top_msg_length
       if (offset<0) then
         if (wantDebug) write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_real: internal error, offset for shifting = ',offset
         success = .false.
         return
       endif

       a_off = a_off + offset
       if (a_off + next_local_n + nbw > a_dim2) then
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif

#ifdef HAVE_DETAILED_TIMINGS
         call timer%start("OpenMP parallel")
#endif

 !$omp parallel do private(my_thread, i, j), schedule(static, 1)
         do my_thread = 1, max_threads
           do i = 1, stripe_count
             do j = top_msg_length+1, top_msg_length+next_local_n
               A(:,j,i,my_thread) = A(:,j+a_off,i,my_thread)
             enddo
#else /* WITH_OPENMP */
         do i = 1, stripe_count

#ifdef WITH_GPU_VERSION
           chunk = min(next_local_n - 1, a_off)
           do j = top_msg_length + 1, top_msg_length + next_local_n, chunk
             top = min(j + chunk, top_msg_length + next_local_n)
             this_chunk = top - j + 1
             dev_offset = (0 + ( (j-1) * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) *8_8
             dev_offset_1 = (0 + ( (j + a_off-1) * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) *8_8
             tmp = cuda_d2d(1)
             istat =  cuda_memcpy( a_dev + dev_offset , a_dev +dev_offset_1, stripe_width*this_chunk*8_8, tmp)
             if (istat .ne. 0) then
               print *,"trans_ev_tridi_to_band_real: error cudaMemcpy"
               stop
             endif

#else
           do j = top_msg_length+1, top_msg_length+next_local_n
             A(:,j,i) = A(:,j+a_off,i)

#endif
#endif /* WITH_OPENMP */

           enddo
         enddo
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
         call timer%stop("OpenMP parallel")
#endif
#endif
         a_off = 0
       endif

     enddo

     ! Just for safety:
     if (ANY(top_send_request    /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR top_send_request ***',my_prow,my_pcol
     if (ANY(bottom_send_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR bottom_send_request ***',my_prow,my_pcol
     if (ANY(top_recv_request    /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR top_recv_request ***',my_prow,my_pcol
     if (ANY(bottom_recv_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR bottom_recv_request ***',my_prow,my_pcol

     if (my_prow == 0) then
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
         print *,"trans_ev_tridi_to_band_real: not yet implemented"
         stop
#endif

       allocate(mpi_statuses(MPI_STATUS_SIZE,num_result_buffers), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"trans_ev_tridi_to_band_real: error when allocating mpi_statuses"//errorMessage
         stop
       endif

       call MPI_Waitall(num_result_buffers, result_send_request, mpi_statuses, mpierr)
       deallocate(mpi_statuses, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"trans_ev_tridi_to_band_real: error when deallocating mpi_statuses"//errorMessage
         stop
       endif

#else
       call MPI_Waitall(num_result_buffers, result_send_request, MPI_STATUSES_IGNORE, mpierr)
#endif
     endif

     if (ANY(result_send_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR result_send_request ***',my_prow,my_pcol
     if (ANY(result_recv_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR result_recv_request ***',my_prow,my_pcol

     if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
         write(error_unit,'(" Kernel time:",f10.3," MFlops: ",f10.3)')  kernel_time, kernel_flops/kernel_time*1.d-6

     ! deallocate all working space

#ifndef WITH_GPU_VERSION
     deallocate(a, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error when deallocating a "//errorMessage
       stop
     endif

#endif
     deallocate(row, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error when deallocating row "//errorMessage
       stop
     endif

     deallocate(limits, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error when deallocating limits"//errorMessage
       stop
     endif

     deallocate(result_send_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error when deallocating result_send_request "//errorMessage
       stop
     endif

     deallocate(result_recv_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error when deallocating result_recv_request "//errorMessage
       stop
     endif

     deallocate(top_border_send_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error when deallocating top_border_send_buffer "//errorMessage
       stop
     endif

     deallocate(top_border_recv_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error when deallocating top_border_recv_buffer "//errorMessage
       stop
     endif

     deallocate(bottom_border_send_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error when deallocating bottom_border_send_buffer "//errorMessage
       stop
     endif

     deallocate(bottom_border_recv_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error when deallocating bottom_border_recv_buffer "//errorMessage
       stop
     endif

     deallocate(result_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error when deallocating result_buffer "//errorMessage
       stop
     endif

     deallocate(bcast_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error when deallocating bcast_buffer "//errorMessage
       stop
     endif

     deallocate(top_send_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error when deallocating top_send_request "//errorMessage
       stop
     endif

     deallocate(top_recv_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error when deallocating top_recv_request "//errorMessage
       stop
     endif

     deallocate(bottom_send_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error when deallocating bottom_send_request "//errorMessage
       stop
     endif

     deallocate(bottom_recv_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error when deallocating bottom_recv_request "//errorMessage
       stop
     endif


#ifdef WITH_GPU_VERSION
     istat = cuda_free(hh_dot_dev)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error in cudaFree "//errorMessage
       stop
     endif

     istat = cuda_free(hh_tau_dev)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error in cudaFree "//errorMessage
       stop
     endif

     istat = cuda_free(row_dev)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error in cudaFree "//errorMessage
       stop
     endif

     deallocate(row_group, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error when deallocating row_group "//errorMessage
       stop
     endif

     istat= cuda_free(row_group_dev)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error in cudaFree "//errorMessage
       stop
     endif

     istat =  cuda_free(bcast_buffer_dev)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_real: error in cudaFree "//errorMessage
       stop
     endif

#endif

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("trans_ev_tridi_to_band_real")
#endif
   return
 contains

#ifndef WITH_GPU_VERSION

   subroutine pack_row(row, n)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     implicit none
     real*8  :: row(:)
     integer :: n, i, noff, nl
#ifdef WITH_OPENMP
     integer :: nt
#endif

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("pack_row")
#endif

#ifdef WITH_OPENMP
     do nt = 1, max_threads
       do i = 1, stripe_count
         noff = (nt-1)*thread_width + (i-1)*stripe_width
         nl   = min(stripe_width, nt*thread_width-noff, l_nev-noff)
         if (nl<=0) exit
         row(noff+1:noff+nl) = a(1:nl,n,i,nt)
       enddo
     enddo
#else
     do i=1,stripe_count
       nl = merge(stripe_width, last_stripe_width, i<stripe_count)
       noff = (i-1)*stripe_width
       row(noff+1:noff+nl) = a(1:nl,n,i)
     enddo
#endif

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("pack_row")
#endif

   end subroutine pack_row

#ifdef WITH_OPENMP
   subroutine unpack_row(row, n, my_thread)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     implicit none

     ! Private variables in OMP regions (my_thread) should better be in the argument list!
     integer, intent(in) :: n, my_thread
     real*8, intent(in)  :: row(:)
     integer             :: i, noff, nl

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("unpack_row")
#endif
     do i=1,stripe_count
       noff = (my_thread-1)*thread_width + (i-1)*stripe_width
       nl   = min(stripe_width, my_thread*thread_width-noff, l_nev-noff)
       if(nl<=0) exit
       a(1:nl,n,i,my_thread) = row(noff+1:noff+nl)
     enddo

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("unpack_row")
#endif

   end subroutine unpack_row

#else /* WITH_OPENMP */

   subroutine unpack_row(row, n)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     implicit none

     real*8  :: row(:)
     integer :: n, i, noff, nl

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("unpack_row")
#endif

     do i=1,stripe_count
       nl = merge(stripe_width, last_stripe_width, i<stripe_count)
       noff = (i-1)*stripe_width
       a(1:nl,n,i) = row(noff+1:noff+nl)
     enddo

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("unpack_row")
#endif
   end subroutine unpack_row
#endif /* WITH_OPENMP */

#endif /* WITH_GPU_VERSION */

#ifdef WITH_GPU_VERSION
    ! Pack a filled row group (i.e. an array of consecutive rows)
   subroutine pack_row_group(rows, n_offset, row_count)

     implicit none
     integer, intent(in) :: n_offset, row_count
     real*8              :: rows(:,:)
     integer             :: max_idx

     ! Use many blocks for higher GPU occupancy
     max_idx = (stripe_count - 1) * stripe_width + last_stripe_width

     ! Use one kernel call to pack the entire row group

!     call my_pack_kernel<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, a_dev, row_group_dev)

     call launch_my_pack_c_kernel(row_count, n_offset, max_idx, stripe_width, a_dim2, stripe_count, l_nev, a_dev, row_group_dev)

     ! Issue one single transfer call for all rows (device to host)
!       rows(:, 1 : row_count) = row_group_dev(:, 1 : row_count)

     istat =  cuda_memcpy( loc(rows(:, 1: row_count)), row_group_dev , row_count * l_nev * 8_8 ,2)
     if (istat .ne. 0) then
       print *,"pack_row_group: error in cudaMemcpy"
       stop
     endif
     !write(*,*) cudaGetErrorString(istat)

   end subroutine


   ! Unpack a filled row group (i.e. an array of consecutive rows)
   subroutine unpack_row_group(rows, n_offset, row_count)

     implicit none
     integer, intent(in) :: n_offset, row_count
     real*8, intent(in)  :: rows(:, :)
     integer             :: max_idx
     integer             :: i

     ! Use many blocks for higher GPU occupancy
     max_idx = (stripe_count - 1) * stripe_width + last_stripe_width

     ! Issue one single transfer call for all rows (host to device)
!     row_group_dev(:, 1 : row_count) = rows(:, 1 : row_count)

      !istat =  cuda_memcpy( row_group_dev , loc(rows(:, 1: row_count)),row_count * l_nev * 8_8 ,1)

     istat =  cuda_memcpy( row_group_dev , loc(rows(1, 1)),row_count * l_nev * 8_8 ,1)
     if (istat .ne. 0) then
       print *,"unpack_row_group: error in cudaMemcpy"
       stop
     endif
     !write(*,*) cudaGetErrorString(istat)

     ! Use one kernel call to pack the entire row group
     !        call my_unpack_kernel<<<grid_size, stripe_width>>>(n_offset, max_idx, stripe_width, a_dim2, stripe_count, row_group_dev, a_dev)

     call launch_my_unpack_c_kernel( row_count, n_offset, max_idx,stripe_width,a_dim2, stripe_count, l_nev, row_group_dev,a_dev)

   end subroutine

   ! This subroutine must be called before queuing the next row for unpacking; it ensures that an unpacking of the current row group
   ! occurs when the queue is full or when the next row belongs to another group
   subroutine unpack_and_prepare_row_group(next_unpack_idx, force)

     implicit none
     integer, intent(in) :: next_unpack_idx
     logical, intent(in) :: force

     if (row_group_size == 0) then
       ! Nothing to flush, just prepare for the upcoming row
       row_group_size = 1
     else
       if (force .or. (row_group_size == nblk) .or. (unpack_idx + 1 /= next_unpack_idx)) then
         ! A flush and a reset must be performed
         call unpack_row_group(row_group(:, :), unpack_idx - row_group_size, row_group_size)
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
   subroutine compute_hh_dot_products_real(nbw, n)

     implicit none
     integer, value :: nbw, n

     if (n .le. 1) return
     call launch_compute_hh_dotp_c_kernel( bcast_buffer_dev, hh_dot_dev, nbw, n)
   end subroutine

   ! The host wrapper for extracting "tau" from the HH reflectors (see the kernel below)
   subroutine extract_hh_tau(nbw, n, is_zero)

     implicit none
     integer, value :: nbw, n
     logical, value :: is_zero
     integer val_is_zero
     if (is_zero) then
     val_is_zero = 1
     else
      val_is_zero = 0
     endif

     call launch_extract_hh_tau_c_kernel(bcast_buffer_dev,hh_tau_dev, nbw, n, val_is_zero)
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
#endif /* WITH_GPU_VERSION */

#ifdef WITH_GPU_VERSION
    ! Host wrapper for the Householder backtransformation step. Several kernels are available. Performance note:
    ! - "compute_hh_trafo_c_kernel" is the C kernel for the backtransformation (this exhibits best performance)
    ! - "compute_hh_trafo_kernel" is the Fortran equivalent of the C kernel
    ! - "compute_hh_trafo_single_kernel" is the reference Fortran kernel
#endif
#ifdef WITH_OPENMP
   subroutine compute_hh_trafo_real(off, ncols, istripe, my_thread, THIS_REAL_ELPA_KERNEL)
#else
   subroutine compute_hh_trafo_real(off, ncols, istripe, THIS_REAL_ELPA_KERNEL)
#endif

#if defined(WITH_REAL_GENERIC_SIMPLE_KERNEL)
      use real_generic_simple_kernel, only : double_hh_trafo_generic_simple
#endif

!#if defined(WITH_REAL_GENERIC_KERNEL)
!      use real_generic_kernel, only : double_hh_trafo_generic
!#endif

#if defined(WITH_REAL_BGP_KERNEL)
      use real_bgp_kernel, only : double_hh_trafo_bgp
#endif

#if defined(WITH_REAL_BGQ_KERNEL)
      use real_bgq_kernel, only : double_hh_trafo_bgq
#endif
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      implicit none

      integer, intent(in) :: THIS_REAL_ELPA_KERNEL

      ! Private variables in OMP regions (my_thread) should better be in the argument list!
      integer             :: off, ncols, istripe
#ifdef WITH_OPENMP
      integer             :: my_thread, noff
#endif
      integer             :: j, nl, jj, jjj
      real*8              :: w(nbw,6), ttt


#ifdef WITH_GPU_VERSION
      ! ncols - indicates the number of HH reflectors to apply; at least 1 must be available
      if (ncols < 1) return
#endif

#ifdef HAVE_DETAILED_TIMINGS
      call timer%start("compute_hh_trafo_real")
#endif

      ttt = mpi_wtime()

#ifndef WITH_OPENMP
      nl = merge(stripe_width, last_stripe_width, istripe<stripe_count)
#else
#ifdef WITH_GPU_VERSION
         print *,"compute_hh_trafo_real: not yet implemented"
         stop
#endif
      if (istripe<stripe_count) then
        nl = stripe_width
      else
        noff = (my_thread-1)*thread_width + (istripe-1)*stripe_width
        nl = min(my_thread*thread_width-noff, l_nev-noff)
        if (nl<=0) return
      endif
#endif

#ifdef WITH_GPU_VERSION
      dev_offset = (0 + (a_off * stripe_width) + ( (istripe - 1) * stripe_width *a_dim2 )) *8
      call launch_compute_hh_trafo_c_kernel(a_dev + dev_offset, bcast_buffer_dev, hh_dot_dev, &
                                            hh_tau_dev, nl, nbw, stripe_width, off, ncols)
#else /* WITH_GPU_VERSION */


#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
      if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_AVX_BLOCK2 .or. &
          THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_GENERIC    .or. &
          THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_GENERIC_SIMPLE .or. &
          THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_SSE .or.        &
          THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_BGP .or.        &
          THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_BGQ) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */

        !FORTRAN CODE / X86 INRINISIC CODE / BG ASSEMBLER USING 2 HOUSEHOLDER VECTORS
#if defined(WITH_REAL_GENERIC_KERNEL)
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_GENERIC) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
          do j = ncols, 2, -2
            w(:,1) = bcast_buffer(1:nbw,j+off)
            w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP
            call double_hh_trafo_generic(a(1,j+off+a_off-1,istripe,my_thread), w, &
                                      nbw, nl, stripe_width, nbw)
#else
            call double_hh_trafo_generic(a(1,j+off+a_off-1,istripe),           w, &
                                      nbw, nl, stripe_width, nbw)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
#endif /* WITH_REAL_GENERIC_KERNEL */


#if defined(WITH_REAL_GENERIC_SIMPLE_KERNEL)
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_GENERIC_SIMPLE) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
          do j = ncols, 2, -2
            w(:,1) = bcast_buffer(1:nbw,j+off)
            w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP
            call double_hh_trafo_generic_simple(a(1,j+off+a_off-1,istripe,my_thread), &
                                                     w, nbw, nl, stripe_width, nbw)
#else
            call double_hh_trafo_generic_simple(a(1,j+off+a_off-1,istripe), &
                                                     w, nbw, nl, stripe_width, nbw)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
#endif /* WITH_REAL_GENERIC_SIMPLE_KERNEL */


#if defined(WITH_REAL_SSE_KERNEL)
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_SSE) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
          do j = ncols, 2, -2
            w(:,1) = bcast_buffer(1:nbw,j+off)
            w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP
            call double_hh_trafo(a(1,j+off+a_off-1,istripe,my_thread), w, nbw, nl, &
                                      stripe_width, nbw)
#else
            call double_hh_trafo(a(1,j+off+a_off-1,istripe), w, nbw, nl, &
                                      stripe_width, nbw)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
#endif /* WITH_REAL_SSE_KERNEL */


#if defined(WITH_REAL_AVX_BLOCK2_KERNEL)
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_AVX_BLOCK2) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
          do j = ncols, 2, -2
            w(:,1) = bcast_buffer(1:nbw,j+off)
            w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP
            call double_hh_trafo_real_sse_avx_2hv(a(1,j+off+a_off-1,istripe,my_thread), &
                                                       w, nbw, nl, stripe_width, nbw)
#else
            call double_hh_trafo_real_sse_avx_2hv(a(1,j+off+a_off-1,istripe), &
                                                       w, nbw, nl, stripe_width, nbw)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
#endif /* WITH_REAL_AVX_BLOCK2_KERNEL */

#if defined(WITH_REAL_BGP_KERNEL)
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_BGP) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
          do j = ncols, 2, -2
            w(:,1) = bcast_buffer(1:nbw,j+off)
            w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP
            call double_hh_trafo_bgp(a(1,j+off+a_off-1,istripe,my_thread), w, nbw, nl, &
                                          stripe_width, nbw)
#else
            call double_hh_trafo_bgp(a(1,j+off+a_off-1,istripe), w, nbw, nl, &
                                          stripe_width, nbw)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
#endif /* WITH_REAL_BGP_KERNEL */


#if defined(WITH_REAL_BGQ_KERNEL)
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_BGQ) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
          do j = ncols, 2, -2
            w(:,1) = bcast_buffer(1:nbw,j+off)
            w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP
            call double_hh_trafo_bgq(a(1,j+off+a_off-1,istripe,my_thread), w, nbw, nl, &
                                          stripe_width, nbw)
#else
            call double_hh_trafo_bgq(a(1,j+off+a_off-1,istripe), w, nbw, nl, &
                                          stripe_width, nbw)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
#endif /* WITH_REAL_BGQ_KERNEL */


!#if defined(WITH_AVX_SANDYBRIDGE)
!              call double_hh_trafo_real_sse_avx_2hv(a(1,j+off+a_off-1,istripe), w, nbw, nl, stripe_width, nbw)
!#endif

#ifdef WITH_OPENMP
        if(j==1) call single_hh_trafo_real(a(1,1+off+a_off,istripe,my_thread), &
                                      bcast_buffer(1,off+1), nbw, nl,     &
                                      stripe_width)
#else
        if(j==1) call single_hh_trafo_real(a(1,1+off+a_off,istripe),           &
                                      bcast_buffer(1,off+1), nbw, nl,     &
                                      stripe_width)
#endif


#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
      endif !
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */



#if defined(WITH_REAL_AVX_BLOCK4_KERNEL)
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
      if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_AVX_BLOCK4) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
        ! X86 INTRINSIC CODE, USING 4 HOUSEHOLDER VECTORS
        do j = ncols, 4, -4
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
          w(:,3) = bcast_buffer(1:nbw,j+off-2)
          w(:,4) = bcast_buffer(1:nbw,j+off-3)
#ifdef WITH_OPENMP
          call quad_hh_trafo_real_sse_avx_4hv(a(1,j+off+a_off-3,istripe,my_thread), w, &
                                                  nbw, nl, stripe_width, nbw)
#else
          call quad_hh_trafo_real_sse_avx_4hv(a(1,j+off+a_off-3,istripe), w, &
                                                  nbw, nl, stripe_width, nbw)
#endif
        enddo
        do jj = j, 2, -2
          w(:,1) = bcast_buffer(1:nbw,jj+off)
          w(:,2) = bcast_buffer(1:nbw,jj+off-1)
#ifdef WITH_OPENMP
          call double_hh_trafo_real_sse_avx_2hv(a(1,jj+off+a_off-1,istripe,my_thread), &
                                                    w, nbw, nl, stripe_width, nbw)
#else
          call double_hh_trafo_real_sse_avx_2hv(a(1,jj+off+a_off-1,istripe), &
                                                    w, nbw, nl, stripe_width, nbw)
#endif
        enddo
#ifdef WITH_OPENMP
        if (jj==1) call single_hh_trafo_real(a(1,1+off+a_off,istripe,my_thread), &
                                          bcast_buffer(1,off+1), nbw, nl, stripe_width)
#else
        if (jj==1) call single_hh_trafo_real(a(1,1+off+a_off,istripe), &
                                          bcast_buffer(1,off+1), nbw, nl, stripe_width)
#endif
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
      endif
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
#endif /* WITH_REAL_AVX_BLOCK4_KERNEL */


#if defined(WITH_REAL_AVX_BLOCK6_KERNEL)
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
      if (THIS_REAL_ELPA_KERNEL .eq. REAL_ELPA_KERNEL_AVX_BLOCK6) then
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
        ! X86 INTRINSIC CODE, USING 6 HOUSEHOLDER VECTORS
        do j = ncols, 6, -6
          w(:,1) = bcast_buffer(1:nbw,j+off)
          w(:,2) = bcast_buffer(1:nbw,j+off-1)
          w(:,3) = bcast_buffer(1:nbw,j+off-2)
          w(:,4) = bcast_buffer(1:nbw,j+off-3)
          w(:,5) = bcast_buffer(1:nbw,j+off-4)
          w(:,6) = bcast_buffer(1:nbw,j+off-5)
#ifdef WITH_OPENMP
          call hexa_hh_trafo_real_sse_avx_6hv(a(1,j+off+a_off-5,istripe,my_thread), w, &
                                                  nbw, nl, stripe_width, nbw)
#else
          call hexa_hh_trafo_real_sse_avx_6hv(a(1,j+off+a_off-5,istripe), w, &
                                                  nbw, nl, stripe_width, nbw)
#endif
        enddo
        do jj = j, 4, -4
          w(:,1) = bcast_buffer(1:nbw,jj+off)
          w(:,2) = bcast_buffer(1:nbw,jj+off-1)
          w(:,3) = bcast_buffer(1:nbw,jj+off-2)
          w(:,4) = bcast_buffer(1:nbw,jj+off-3)
#ifdef WITH_OPENMP
          call quad_hh_trafo_real_sse_avx_4hv(a(1,jj+off+a_off-3,istripe,my_thread), w, &
                                                  nbw, nl, stripe_width, nbw)
#else
          call quad_hh_trafo_real_sse_avx_4hv(a(1,jj+off+a_off-3,istripe), w, &
                                                  nbw, nl, stripe_width, nbw)
#endif
        enddo
        do jjj = jj, 2, -2
          w(:,1) = bcast_buffer(1:nbw,jjj+off)
          w(:,2) = bcast_buffer(1:nbw,jjj+off-1)
#ifdef WITH_OPENMP
        call double_hh_trafo_real_sse_avx_2hv(a(1,jjj+off+a_off-1,istripe,my_thread), &
                                                    w, nbw, nl, stripe_width, nbw)
#else
        call double_hh_trafo_real_sse_avx_2hv(a(1,jjj+off+a_off-1,istripe), &
                                                    w, nbw, nl, stripe_width, nbw)
#endif
      enddo
#ifdef WITH_OPENMP
      if (jjj==1) call single_hh_trafo_real(a(1,1+off+a_off,istripe,my_thread), &
                                           bcast_buffer(1,off+1), nbw, nl, stripe_width)
#else
      if (jjj==1) call single_hh_trafo_real(a(1,1+off+a_off,istripe), &
                                           bcast_buffer(1,off+1), nbw, nl, stripe_width)
#endif
#if defined(WITH_NO_SPECIFIC_REAL_KERNEL)
    endif
#endif /* WITH_NO_SPECIFIC_REAL_KERNEL */
#endif /* WITH_REAL_AVX_BLOCK4_KERNEL */

#endif /* WITH_GPU_VERSION */


#ifdef WITH_OPENMP
    if (my_thread==1) then
#endif
      kernel_flops = kernel_flops + 4*int(nl,8)*int(ncols,8)*int(nbw,8)
      kernel_time = kernel_time + mpi_wtime()-ttt
#ifdef WITH_OPENMP
    endif
#endif
#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("compute_hh_trafo_real")
#endif

  end subroutine compute_hh_trafo_real

 end subroutine  trans_ev_tridi_to_band_real

!-------------------------------------------------------------------------------

subroutine single_hh_trafo_real(q, hh, nb, nq, ldq)
#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif

    ! Perform single real Householder transformation.
    ! This routine is not performance critical and thus it is coded here in Fortran

    implicit none
    integer  :: nb, nq, ldq
#ifdef WITH_GPU_VERSION
    real*8   :: q(:,:) ! remove this
#else
    real*8   :: q(ldq,*) ! remove this
#endif
    real*8   :: hh(*) ! carefull hh is in the calling subroutine a MPI bcast_buffer(:,:) !

    integer  :: i
    real*8   :: v(nq)

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("single_hh_trafo_real")
#endif

    ! v = q * hh
    v(:) = q(1:nq,1)
    do i=2,nb
      v(:) = v(:) + q(1:nq,i) * hh(i)
    enddo

    ! v = v * tau
    v(:) = v(:) * hh(1)

    ! q = q - v * hh**T
    q(1:nq,1) = q(1:nq,1) - v(:)
    do i=2,nb
      q(1:nq,i) = q(1:nq,i) - v(:) * hh(i)
    enddo

#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("single_hh_trafo_real")
#endif


end subroutine

!-------------------------------------------------------------------------------

subroutine determine_workload(na, nb, nprocs, limits)
#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif
    implicit none

    integer, intent(in)  :: na, nb, nprocs
    integer, intent(out) :: limits(0:nprocs)

    integer              :: i

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("determine_workload")
#endif

    if (na <= 0) then
      limits(:) = 0
      return
    endif

    if (nb*nprocs > na) then
        ! there is not enough work for all
      do i = 0, nprocs
        limits(i) = min(na, i*nb)
      enddo
    else
       do i = 0, nprocs
         limits(i) = (i*na)/nprocs
       enddo
    endif

#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("determine_workload")
#endif
end subroutine

!-------------------------------------------------------------------------------

subroutine bandred_complex(na, a, lda, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols, tmat, wantDebug, success)

!-------------------------------------------------------------------------------
!  bandred_complex: Reduces a distributed hermitian matrix to band form
!
!  Parameters
!
!  na          Order of matrix
!
!  a(lda,matrixCols)    Distributed matrix which should be reduced.
!              Distribution is like in Scalapack.
!              Opposed to Scalapack, a(:,:) must be set completely (upper and lower half)
!              a(:,:) is overwritten on exit with the band and the Householder vectors
!              in the upper half.
!
!  lda         Leading dimension of a
!  matrixCols  local columns of matrix a
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  nbw         semi bandwith of output matrix
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!
!  tmat(nbw,nbw,numBlocks)    where numBlocks = (na-1)/nbw + 1
!              Factors for the Householder vectors (returned), needed for back transformation
!
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
   use timings
 endif
#endif

#ifdef WITH_GPU_VERSION
   use cuda_routines
   use iso_c_binding
#endif
   implicit none

   integer                 :: na, lda, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols
   complex*16              :: a(lda,matrixCols), tmat(nbw,nbw,numBlocks)

   complex*16, parameter   :: CZERO = (0.d0,0.d0), CONE = (1.d0,0.d0)

   integer                 :: my_prow, my_pcol, np_rows, np_cols, mpierr
   integer                 :: l_cols, l_rows
   integer                 :: i, j, lcs, lce, lre, lc, lr, cur_pcol, n_cols, nrow
   integer                 :: istep, ncol, lch, lcx, nlc
   integer                 :: tile_size, l_rows_tile, l_cols_tile

   real*8                  :: vnorm2
   complex*16              :: xf, aux1(nbw), aux2(nbw), vrl, tau, vav(nbw,nbw)

   complex*16, allocatable :: tmp(:,:), vr(:), vmr(:,:), umc(:,:)

#ifdef WITH_GPU_VERSION
   integer(c_size_t)       :: umc_dev, tmat_dev,vav_dev,vmr_dev,a_dev
   integer                 :: cur_l_rows, cur_l_cols,vmr_size ,umc_size
   integer(c_size_t)       :: lc_start, lc_end, lr_end, lce_1, lcs_1,lre_1
   integer                 :: na_rows, na_cols
   integer, external       :: numroc
#endif

   logical, intent(in)     :: wantDebug
   logical, intent(out)    :: success
   character(200)          :: errorMessage
   integer                 :: istat

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("bandred_complex")
#endif
   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   success = .true.

   ! Semibandwith nbw must be a multiple of blocksize nblk

   if (mod(nbw,nblk)/=0) then
     if (my_prow==0 .and. my_pcol==0) then
       if (wantDebug) then
         write(error_unit,*) 'ELPA2_bandred_complex: ERROR: nbw=',nbw,', nblk=',nblk
         write(error_unit,*) 'ELPA2_bandred_complex: ELPA2 works only for nbw==n*nblk'
       endif
       success = .false.
       return
     endif
   endif

#ifdef WITH_GPU_VERSION
   na_rows = numroc(na, nblk, my_prow, 0, np_rows)
!   if (na_rows .ne. na_rows2) then
!     print *,"bandred_complex: Why is na_rows not equal? ",na_rows,na_rows2
!   endif
   na_cols = numroc(na, nblk, my_pcol, 0, np_cols)
!   if (na_cols .ne. na_cols2) then
!     print *,"bandred_complex: Why is na_cols not equal? ",na_cols,na_cols2
!   endif

   istat = cuda_malloc(tmat_dev, nbw*nbw*16_8)
   if (istat .ne. 0) then
     print *, " bandred_complex: cuda malloc failed tmat_dev ", istat
     stop
   endif

   istat = cuda_malloc(vav_dev, nbw*nbw*16_8)
   if (istat .ne. 0) then
     print *, "bandred_complex:  cuda malloc failed vav_dev ", istat
     stop
   endif

   istat = cuda_malloc(a_dev, lda*na_cols*16_8)
   if (istat .ne. 0) then
     print *, "bandred_complex:  cuda malloc failed a_dev ", istat
     stop
   endif
#endif

   ! Matrix is split into tiles; work is done only for tiles on the diagonal or above

   tile_size = nblk*least_common_multiple(np_rows,np_cols) ! minimum global tile size
   tile_size = ((128*max(np_rows,np_cols)-1)/tile_size+1)*tile_size ! make local tiles at least 128 wide

   l_rows_tile = tile_size/np_rows ! local rows of a tile
   l_cols_tile = tile_size/np_cols ! local cols of a tile

#ifdef WITH_GPU_VERSION
   if (size(a,dim=1) .ne. lda .or. size(a,dim=2) .ne. na_cols) then
     print *,"bandred_complex: sizes of a wrong ? ",lda,size(a,dim=1),na_cols,size(a,dim=2)
   endif
   istat = cuda_memcpy(a_dev, loc(a(1,1)),(lda)*(na_cols)*16_8,cudaMemcpyHostToDevice)
   if (istat .ne. 0) then
     print *, "bandred_complex:  cuda memcpy faild a_dev ", istat
     stop
   endif
#endif
   do istep = (na-1)/nbw, 1, -1

     n_cols = MIN(na,(istep+1)*nbw) - istep*nbw ! Number of columns in current step

     ! Number of local columns/rows of remaining matrix
     l_cols = local_index(istep*nbw, my_pcol, np_cols, nblk, -1)
     l_rows = local_index(istep*nbw, my_prow, np_rows, nblk, -1)

     ! Allocate vmr and umc to their exact sizes so that they can be used in bcasts and reduces
#ifdef WITH_GPU_VERSION
     cur_l_rows = max(l_rows, 1)
     cur_l_cols = max(l_cols, 1)

     vmr_size = cur_l_rows * 2 * n_cols
     umc_size = cur_l_cols * 2 * n_cols

     if ((.not. allocated(umc)) .or. (umc_size .gt. ubound(umc, dim=1))) then
       if (allocated(umc)) then
         deallocate(umc, stat=istat, errmsg=errorMessage)
         if (istat .ne. 0) then
           print *,"bandred_complex: error when allocating umc "//errorMessage
           stop
         endif
         istat = cuda_free(umc_dev)
         if (istat .ne. 0)then
           print *,"bandred_complex: error in cudaFree"
           stop
         endif
       endif

       allocate(umc(max(l_cols,1),2*n_cols), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_complex: error when allocating umc "//errorMessage
         stop
       endif

       if (max(l_cols,1) * 2*n_cols .gt. umc_size) then
         print *,"bandred_complex: umc_size ",max(l_cols,1) * 2*n_cols,umc_size
       endif

       istat = cuda_malloc(umc_dev, umc_size*16_8)
       if (istat .ne. 0) then
         print *, "bandred_complex:  cuda malloc failed umc_dev ", istat
         stop
       endif
     endif

     if ((.not. allocated(vmr)) .or. (vmr_size .gt. ubound(vmr, dim=1))) then
       if (allocated(vmr)) then
         deallocate(vmr, stat=istat, errmsg=errorMessage)
         if (istat .ne. 0) then
           print *,"bandred_complex: error when deallocating vmr "//errorMessage
           stop
         endif
         istat = cuda_free(vmr_dev)
         if (istat .ne. 0)then
           print *,"bandred_complex: error in cudaFree"
           stop
         endif
       endif

       allocate(vmr(max(l_rows,1),2*n_cols), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_complex: error when allocating vmr "//errorMessage
         stop
       endif

       if (max(l_rows,1) * 2*n_cols .gt. vmr_size) then
         print *,"bandred_complex: vmc_size ",max(l_rows,1) * 2*n_cols,vmr_size
       endif


       istat = cuda_malloc(vmr_dev, vmr_size*16_8)
       if (istat .ne. 0) then
         print *, "bandred_complex:  cuda malloc failed vmr_dev ", istat
         stop
       endif

     endif

     if ((.not. allocated(vr)) .or. (l_rows + 1 .gt. ubound(vr, dim=1))) then
       if (allocated(vr)) then
         deallocate(vr, stat=istat, errmsg=errorMessage)
         if (istat .ne. 0) then
           print *,"bandred_complex: error when deallocating vr "//errorMessage
           stop
         endif
       endif

       allocate(vr(l_rows + 1), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_complex: error when allocating vr "//errorMessage
         stop
       endif
     endif
#else
     allocate(vmr(max(l_rows,1),2*n_cols), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"bandred_complex: error when allocating vmr "//errorMessage
       stop
     endif

     allocate(umc(max(l_cols,1),2*n_cols), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"bandred_complex: error when allocating umc "//errorMessage
       stop
     endif

     allocate(vr(l_rows+1), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"bandred_complex: error when allocating vr "//errorMessage
       stop
     endif
#endif

     vmr(1:l_rows,1:n_cols) = 0.
     vr(:) = 0
     tmat(:,:,istep) = 0

#ifdef WITH_GPU_VERSION
     lc_start = local_index(istep*nbw+1, my_pcol, np_cols, nblk, -1)
     lc_end   = local_index(istep*nbw+n_cols, my_pcol, np_cols, nblk, -1)
     lr_end   = local_index((istep-1)*nbw + n_cols, my_prow, np_rows, nblk, -1)

     if (lc_start .le. 0) lc_start = 1
       cur_pcol = pcol(istep*nbw+1, nblk, np_cols)
       if (my_pcol == cur_pcol) then
         istat = cuda_memcpy2d(loc(a(1, lc_start)), lda*16_8, (a_dev + ((lc_start-1) * lda*16_8)), lda*16_8, &
                               lr_end*16_8, (lc_end - lc_start+1),cudaMemcpyDeviceToHost)
         if (istat .ne. 0) then
           print *, "bandred_complex: error in cudaMemcpy2"
           stop
         endif
     endif
#endif

     ! Reduce current block to lower triangular form

     do lc = n_cols, 1, -1

       ncol = istep*nbw + lc ! absolute column number of householder vector
       nrow = ncol - nbw ! Absolute number of pivot row

       lr  = local_index(nrow, my_prow, np_rows, nblk, -1) ! current row length
       lch = local_index(ncol, my_pcol, np_cols, nblk, -1) ! HV local column number

       tau = 0

       if(nrow == 1) exit ! Nothing to do

       cur_pcol = pcol(ncol, nblk, np_cols) ! Processor column owning current block

       if (my_pcol==cur_pcol) then

         ! Get vector to be transformed; distribute last element and norm of
         ! remaining elements to all procs in current column

         vr(1:lr) = a(1:lr,lch) ! vector to be transformed

         if (my_prow==prow(nrow, nblk, np_rows)) then
           aux1(1) = dot_product(vr(1:lr-1),vr(1:lr-1))
           aux1(2) = vr(lr)
         else
           aux1(1) = dot_product(vr(1:lr),vr(1:lr))
           aux1(2) = 0.
         endif

         call mpi_allreduce(aux1,aux2,2,MPI_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_rows,mpierr)

         vnorm2 = aux2(1)
         vrl    = aux2(2)

         ! Householder transformation

         call hh_transform_complex(vrl, vnorm2, xf, tau)

         ! Scale vr and store Householder vector for back transformation

         vr(1:lr) = vr(1:lr) * xf
         if (my_prow==prow(nrow, nblk, np_rows)) then
           a(1:lr-1,lch) = vr(1:lr-1)
           a(lr,lch) = vrl
           vr(lr) = 1.
         else
           a(1:lr,lch) = vr(1:lr)
         endif

       endif

       ! Broadcast Householder vector and tau along columns

       vr(lr+1) = tau
       call MPI_Bcast(vr,lr+1,MPI_DOUBLE_COMPLEX,cur_pcol,mpi_comm_cols,mpierr)
       vmr(1:lr,lc) = vr(1:lr)
       tau = vr(lr+1)
       tmat(lc,lc,istep) = conjg(tau) ! Store tau in diagonal of tmat

       ! Transform remaining columns in current block with Householder vector

       ! Local dot product

       aux1 = 0

       nlc = 0 ! number of local columns
       do j=1,lc-1
         lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
         if (lcx>0) then
           nlc = nlc+1
           aux1(nlc) = dot_product(vr(1:lr),a(1:lr,lcx))
         endif
       enddo

       ! Get global dot products
       if (nlc>0) call mpi_allreduce(aux1,aux2,nlc,MPI_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_rows,mpierr)

       ! Transform

       nlc = 0
       do j=1,lc-1
         lcx = local_index(istep*nbw+j, my_pcol, np_cols, nblk, 0)
         if (lcx>0) then
           nlc = nlc+1
           a(1:lr,lcx) = a(1:lr,lcx) - conjg(tau)*aux2(nlc)*vr(1:lr)
         endif
       enddo

     enddo

     ! Calculate scalar products of stored Householder vectors.
     ! This can be done in different ways, we use zherk

#ifdef WITH_GPU_VERSION
     cur_pcol = pcol(istep*nbw+1, nblk, np_cols)
     if (my_pcol == cur_pcol) then
        istat = cuda_memcpy2d((a_dev+((lc_start-1)*lda*16_8)), lda*16_8, loc(a(1,lc_start)), &
                lda*16_8,  lr_end*16_8, (lc_end - lc_start+1),cudaMemcpyHostToDevice)
        if (istat .ne. 0) then
          print *, "bandred_complex: cuda memcpy a_dev  failed ", istat
          stop
        endif
     endif
#endif
     vav = 0
     if (l_rows>0) &
        call zherk('U','C',n_cols,l_rows,CONE,vmr,ubound(vmr,dim=1),CZERO,vav,ubound(vav,dim=1))
     call herm_matrix_allreduce(n_cols,vav, nbw,nbw,mpi_comm_rows)

     ! Calculate triangular matrix T for block Householder Transformation

     do lc=n_cols,1,-1
       tau = tmat(lc,lc,istep)
       if (lc<n_cols) then
         call ztrmv('U','C','N',n_cols-lc,tmat(lc+1,lc+1,istep),ubound(tmat,dim=1),vav(lc+1,lc),1)
         tmat(lc,lc+1:n_cols,istep) = -tau * conjg(vav(lc+1:n_cols,lc))
       endif
     enddo

     ! Transpose vmr -> vmc (stored in umc, second half)

     call elpa_transpose_vectors_complex  (vmr, ubound(vmr,dim=1), mpi_comm_rows, &
                                   umc(1,n_cols+1), ubound(umc,dim=1), mpi_comm_cols, &
                                   1, istep*nbw, n_cols, nblk)

     ! Calculate umc = A**T * vmr
     ! Note that the distributed A has to be transposed
     ! Opposed to direct tridiagonalization there is no need to use the cache locality
     ! of the tiles, so we can use strips of the matrix

     umc(1:l_cols,1:n_cols) = 0.d0
     vmr(1:l_rows,n_cols+1:2*n_cols) = 0
     if (l_cols>0 .and. l_rows>0) then
#ifdef WITH_GPU_VERSION
       if (size(vmr,dim=1)*size(vmr,dim=2) .gt. vmr_size) then
         print *,"bandred_complex: vmr size 2 :",size(vmr,dim=1)*size(vmr,dim=2),vmr_size
         stop
       endif
       istat = cuda_memcpy(vmr_dev, loc(vmr(1,1)),vmr_size*16_8,cudaMemcpyHostToDevice)
       if (istat .ne. 0) then
         print *, "bandred_complex:  cuda memcpy vmr_dev failed ", istat
         stop
       endif
       if (size(umc,dim=1)*size(umc,dim=2) .gt. umc_size) then
         print *,"bandred_complex: umc size 2 :",size(umc,dim=1)*size(umc,dim=2),umc_size
         stop
       endif

       istat = cuda_memcpy(umc_dev, loc(umc(1,1)),umc_size*16_8,cudaMemcpyHostToDevice)
       if (istat .ne. 0) then
         print *, "bandred_complex:  cuda memcpy umc_dev failed  ", istat
         stop
       endif
#endif
       do i=0,(istep*nbw-1)/tile_size

         lcs = i*l_cols_tile+1
         lce = min(l_cols,(i+1)*l_cols_tile)
         if (lce<lcs) cycle

         lre = min(l_rows,(i+1)*l_rows_tile)
#ifdef WITH_GPU_VERSION
         call cublas_ZGEMM('C','N',lce-lcs+1,n_cols,lre,CONE, (a_dev + ((lcs-1)*lda*16_8)), lda, &
                      vmr_dev,cur_l_rows,CONE,(umc_dev +(lcs-1)*16_8), cur_l_cols)
#else
         call ZGEMM('C','N',lce-lcs+1,n_cols,lre,CONE,a(1,lcs),ubound(a,dim=1), &
                      vmr,ubound(vmr,dim=1),CONE,umc(lcs,1),ubound(umc,dim=1))
#endif

         if (i==0) cycle
         lre = min(l_rows,i*l_rows_tile)
#ifdef WITH_GPU_VERSION
         call cublas_ZGEMM('N','N',lre,n_cols,lce-lcs+1,CONE,  (a_dev+((lcs-1)*lda*16_8)),lda,  &
                           (umc_dev+(cur_l_cols * n_cols+lcs-1)*16_8), cur_l_cols,CONE, &
                           (vmr_dev+(cur_l_rows * n_cols)*16_8), cur_l_rows)
#else
         call ZGEMM('N','N',lre,n_cols,lce-lcs+1,CONE,a(1,lcs),lda, &
                    umc(lcs,n_cols+1),ubound(umc,dim=1),CONE,vmr(1,n_cols+1),ubound(vmr,dim=1))
#endif
       enddo

#ifdef WITH_GPU_VERSION
       if (size(vmr,dim=1)*size(vmr,dim=2) .gt. vmr_size) then
         print *,"bandred_complex: vmr size 3 :",size(vmr,dim=1)*size(vmr,dim=2),vmr_size
         stop
       endif

       istat = cuda_memcpy(loc(vmr(1,1)),vmr_dev,vmr_size*16_8,cudaMemcpyDeviceToHost)
       if (istat .ne. 0) then
         print *, "bandred_complex:  cuad memcpy failed vmr ", istat
         stop
       endif
       if (size(umc,dim=1)*size(umc,dim=2) .gt. umc_size) then
         print *,"bandred_complex: umc size 3 :",size(umc,dim=1)*size(umc,dim=2),umc_size
         stop
       endif

       istat = cuda_memcpy(loc(umc(1,1)), umc_dev,umc_size*16_8,cudaMemcpyDeviceToHost)
       if (istat .ne. 0) then
         print *, "bandred_complex:  cuad memcpy failed umc ", istat
         stop
       endif
#endif
     endif

     ! Sum up all ur(:) parts along rows and add them to the uc(:) parts
     ! on the processors containing the diagonal
     ! This is only necessary if ur has been calculated, i.e. if the
     ! global tile size is smaller than the global remaining matrix

     if (tile_size < istep*nbw) then
       call elpa_reduce_add_vectors_complex  (vmr(1,n_cols+1),ubound(vmr,dim=1),mpi_comm_rows, &
                                       umc, ubound(umc,dim=1), mpi_comm_cols, &
                                       istep*nbw, n_cols, nblk)
     endif

     if (l_cols>0) then
       allocate(tmp(l_cols,n_cols), stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_complex: error when allocating tmp "//errorMessage
         stop
       endif

       call mpi_allreduce(umc,tmp,l_cols*n_cols,MPI_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_rows,mpierr)

       umc(1:l_cols,1:n_cols) = tmp(1:l_cols,1:n_cols)
       deallocate(tmp, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_complex: error when deallocating tmp "//errorMessage
         stop
       endif
     endif

     ! U = U * Tmat**T
#ifdef WITH_GPU_VERSION
     if (size(umc,dim=1)*size(umc,dim=2) .gt. umc_size) then
       print *,"bandred_complex: umc size 4 :",size(umc,dim=1)*size(umc,dim=2),umc_size
       stop
     endif

     istat = cuda_memcpy(umc_dev, loc(umc(1,1)),umc_size*16_8,cudaMemcpyHostToDevice)
     if (istat .ne. 0) then
       print *, "bandred_complex:  cuad memcpy failed umc_dev ", istat
       stop
     endif

     istat = cuda_memcpy(tmat_dev,loc(tmat(1,1,istep)),nbw*nbw*16_8,cudaMemcpyHostToDevice)
     if (istat .ne. 0) then
       print *, "bandred_complex:  cuad memcpy failed tmat_dev ", istat
       stop
     endif

     call  cublas_ztrmm('Right','Upper','C','Nonunit',l_cols,n_cols,CONE,tmat_dev,nbw,umc_dev,cur_l_cols)
#else

     call ztrmm('Right','Upper','C','Nonunit',l_cols,n_cols,CONE,tmat(1,1,istep),ubound(tmat,dim=1),umc,ubound(umc,dim=1))
#endif

     ! VAV = Tmat * V**T * A * V * Tmat**T = (U*Tmat**T)**T * V * Tmat**T
#ifdef WITH_GPU_VERSION
     istat = cuda_memcpy(vav_dev,loc(vav(1,1)), nbw*nbw*16_8,cudaMemcpyHostToDevice)
     if (istat .ne. 0) then
       print *, "bandred_complex:  cuad memcpy failed vav_dev ", istat
       stop
     endif

     call cublas_zgemm('C','N',n_cols,n_cols,l_cols,CONE,umc_dev,cur_l_cols,(umc_dev +( cur_l_cols *n_cols) *16_8 ), &
                       cur_l_cols,CZERO,vav_dev, nbw)

     call cublas_ztrmm('Right','Upper','C','Nonunit',n_cols,n_cols,CONE,tmat_dev,nbw,vav_dev,nbw)

     istat = cuda_memcpy(loc(vav(1,1)), vav_dev,nbw*nbw*16_8,cudaMemcpyDeviceToHost)
     if (istat .ne. 0) then
       print *, "bandred_complex:  cuad memcpy failed vav ", istat
       stop
     endif

     call herm_matrix_allreduce(n_cols,vav, nbw, nbw,mpi_comm_cols)

     istat = cuda_memcpy(vav_dev,loc(vav(1,1)),nbw*nbw*16_8,cudaMemcpyHostToDevice)
     if (istat .ne. 0) then
       print *, "bandred_complex:  cuad memcpy failed vav_dev ", istat
       stop
     endif
#else
     call zgemm('C','N',n_cols,n_cols,l_cols,CONE,umc,ubound(umc,dim=1),umc(1,n_cols+1),ubound(umc,dim=1),CZERO,vav,ubound(vav,dim=1))
     call ztrmm('Right','Upper','C','Nonunit',n_cols,n_cols,CONE,tmat(1,1,istep),ubound(tmat,dim=1),vav,ubound(vav,dim=1))

     call herm_matrix_allreduce(n_cols,vav,ubound(vav,dim=1),mpi_comm_cols)
#endif
     ! U = U - 0.5 * V * VAV

#ifdef WITH_GPU_VERSION
     call cublas_zgemm('N','N',l_cols,n_cols,n_cols,(-0.5d0,0.d0),(umc_dev + (cur_l_cols * n_cols )*16_8),cur_l_cols,vav_dev, &
                       nbw,CONE,umc_dev,cur_l_cols)
     ! Transpose umc -> umr (stored in vmr, second half)

     if (size(umc,dim=1)*size(umc,dim=2) .gt. umc_size) then
       print *,"bandred_complex: umc size 5 :",size(umc,dim=1)*size(umc,dim=2),umc_size
       stop
     endif

      istat = cuda_memcpy(loc(umc(1,1)),umc_dev,umc_size*16_8,cudaMemcpyDeviceToHost)
      if (istat .ne. 0) then
        print *, "bandred_complex:  cuad memcpy failed umc ", istat
        stop
      endif

      call elpa_transpose_vectors_complex  (umc, ubound(umc,dim=1), mpi_comm_cols, &
                                    vmr(1,n_cols+1), ubound(vmr,dim=1), mpi_comm_rows, &
                                    1, istep*nbw, n_cols, nblk)
     if (size(vmr,dim=1)*size(vmr,dim=2) .gt. vmr_size) then
       print *,"bandred_complex: vmr size 4 :",size(vmr,dim=1)*size(vmr,dim=2),vmr_size
       stop
     endif

     istat = cuda_memcpy(vmr_dev,loc(vmr(1,1)),vmr_size*16_8,cudaMemcpyHostToDevice)
     if (istat .ne. 0) then
       print *, "bandred_complex:  cuda memcpy failed vav_dev", istat
       stop
     endif

     if (size(umc,dim=1)*size(umc,dim=2) .gt. umc_size) then
       print *,"bandred_complex: umc size 6 :",size(umc,dim=1)*size(umc,dim=2),umc_size
       stop
     endif

     istat = cuda_memcpy(umc_dev,loc(umc(1,1)),umc_size*16_8,cudaMemcpyHostToDevice)
     if (istat .ne. 0) then
       print *, "bandred_complex:  cuda memcpy failed umc_dev ", istat
       stop
     endif
#else
     call zgemm('N','N',l_cols,n_cols,n_cols,(-0.5d0,0.d0),umc(1,n_cols+1),ubound(umc,dim=1),vav,ubound(vav,dim=1), &
         CONE,umc,ubound(umc,dim=1))

     ! Transpose umc -> umr (stored in vmr, second half)

     call elpa_transpose_vectors_complex  (umc, ubound(umc,dim=1), mpi_comm_cols, &
                                    vmr(1,n_cols+1), ubound(vmr,dim=1), mpi_comm_rows, &
                                    1, istep*nbw, n_cols, nblk)

#endif
     ! A = A - V*U**T - U*V**T

     do i=0,(istep*nbw-1)/tile_size
       lcs = i*l_cols_tile+1
       lce = min(l_cols,(i+1)*l_cols_tile)
       lre = min(l_rows,(i+1)*l_rows_tile)
       if (lce<lcs .or. lre<1) cycle
#ifdef WITH_GPU_VERSION
       call cublas_zgemm('N','C',lre,lce-lcs+1,2*n_cols,-CONE, &
                         vmr_dev ,cur_l_rows,(umc_dev +(lcs-1)*16_8),cur_l_cols, &
                         CONE,(a_dev + (lcs-1)*lda*16_8),lda)
#else
       call zgemm('N','C',lre,lce-lcs+1,2*n_cols,-CONE, &
                   vmr,ubound(vmr,dim=1),umc(lcs,1),ubound(umc,dim=1), &
                   CONE,a(1,lcs),lda)
#endif
     enddo

#ifdef WITH_GPU_VERSION
   enddo ! istep loop
     if (size(a,dim=1)*size(a,dim=2) .ne. lda*na_cols) then
       print *,"bandred_complex: size a ",size(a,dim=1)*size(a,dim=2) , lda*na_cols
     endif

     istat = cuda_memcpy ( loc(a(1,1)), a_dev, lda*na_cols*16_8,cudaMemcpyDeviceToHost)
     if (istat .ne. 0) then
       print *, "bandred_complex:  cuad memcpy failed a ", istat
       stop
     endif

     istat = cuda_free(a_dev)
     if (istat .ne. 0) then
       print *,"bandred_complex: error in cudaFree"
       stop
     endif

     istat = cuda_free(tmat_dev)
     if (istat .ne. 0) then
       print *,"bandred_complex: error in cudaFree"
       stop
     endif

     istat = cuda_free(vav_dev)
     if (istat .ne. 0) then
       print *,"bandred_complex: error in cudaFree"
       stop
     endif

#endif
     if (allocated(vr)) then
       deallocate(vr, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_complex: error when deallocating vr "//errorMessage
         stop
       endif
     endif
     if (allocated(vmr)) then
       deallocate(vmr, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_complex: error when deallocating vmr "//errorMessage
         stop
       endif
#ifdef WITH_GPU_VERSION
       istat = cuda_free(vmr_dev)
       if (istat .ne. 0) then
         print *,"bandred_complex: error in cudaFree"
         stop
       endif

#endif
     endif
     if (allocated(umc)) then
       deallocate(umc, stat=istat, errmsg=errorMessage)
       if (istat .ne. 0) then
         print *,"bandred_complex: error when deallocating umc "//errorMessage
         stop
       endif

#ifdef WITH_GPU_VERSION
       istat = cuda_free(umc_dev)
       if (istat .ne. 0) then
         print *,"bandred_complex: error in cudaFree"
         stop
       endif

#endif
     endif

#ifndef WITH_GPU_VERSION
   enddo ! istep-loop
#endif

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("bandred_complex")
#endif

end subroutine bandred_complex
!-------------------------------------------------------------------------------

subroutine herm_matrix_allreduce(n,a,lda,ldb,comm)

!-------------------------------------------------------------------------------
!  herm_matrix_allreduce: Does an mpi_allreduce for a hermitian matrix A.
!  On entry, only the upper half of A needs to be set
!  On exit, the complete matrix is set
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   implicit none
   integer    :: n, lda, ldb, comm
   complex*16 :: a(lda,ldb)

   integer    :: i, nc, mpierr
   complex*16 :: h1(n*n), h2(n*n)

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("herm_matrix_allreduce")
#endif

   nc = 0
   do i=1,n
     h1(nc+1:nc+i) = a(1:i,i)
     nc = nc+i
   enddo

   call mpi_allreduce(h1,h2,nc,MPI_DOUBLE_COMPLEX,MPI_SUM,comm,mpierr)

   nc = 0
   do i=1,n
     a(1:i,i) = h2(nc+1:nc+i)
     a(i,1:i-1) = conjg(a(1:i-1,i))
     nc = nc+i
   enddo

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("herm_matrix_allreduce")
#endif

end subroutine herm_matrix_allreduce

!-------------------------------------------------------------------------------

subroutine trans_ev_band_to_full_complex(na, nqc, nblk, nbw, a, lda, tmat, q, ldq, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols)

!-------------------------------------------------------------------------------
!  trans_ev_band_to_full_complex:
!  Transforms the eigenvectors of a band matrix back to the eigenvectors of the original matrix
!
!  Parameters
!
!  na          Order of matrix a, number of rows of matrix q
!
!  nqc         Number of columns of matrix q
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  nbw         semi bandwith
!
!  a(lda,matrixCols)    Matrix containing the Householder vectors (i.e. matrix a after bandred_complex)
!              Distribution is like in Scalapack.
!
!  lda         Leading dimension of a
!  matrixCols  local columns of matrix a and q
!
!  tmat(nbw,nbw,numBlocks) Factors returned by bandred_complex
!
!  q           On input: Eigenvectors of band matrix
!              On output: Transformed eigenvectors
!              Distribution is like in Scalapack.
!
!  ldq         Leading dimension of q
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
#ifdef WITH_GPU_VERSION
   use cuda_routines
   use iso_c_binding
#endif
   implicit none

   complex*16, allocatable   :: q_temp(:,:)
   complex*16, allocatable   :: tmat_temp(:,:)
   integer                 :: na, nqc, lda, ldq, nblk, nbw, matrixCols, numBlocks, mpi_comm_rows, mpi_comm_cols
   complex*16              :: a(lda,matrixCols), q(ldq,matrixCols), tmat(nbw, nbw, numBlocks)

   complex*16, parameter     :: CZERO = (0.d0,0.d0), CONE = (1.d0,0.d0)

   integer                   :: my_prow, my_pcol, np_rows, np_cols, mpierr
   integer                   :: max_blocks_row, max_blocks_col, max_local_rows, max_local_cols
   integer                   :: l_cols, l_rows, l_colh, n_cols
   integer                   :: istep, lc, ncol, nrow, nb, ns

#ifdef WITH_GPU_VERSION
   integer(C_SIZE_T)         :: hvm_dev, q_dev, tmat_dev, tmp_dev
#endif
   complex*16, allocatable   :: tmp1(:), tmp2(:), hvb(:), hvm(:,:)
   integer                   :: i
   integer                   :: istat
   character(200)            :: errorMessage

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("trans_ev_band_to_full_complex")
#endif
   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   max_blocks_row = ((na -1)/nblk)/np_rows + 1  ! Rows of A
   max_blocks_col = ((nqc-1)/nblk)/np_cols + 1  ! Columns of q!

   max_local_rows = max_blocks_row*nblk
   max_local_cols = max_blocks_col*nblk

   allocate(tmp1(max_local_cols*nbw), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error when allocating tmp1 "//errorMessage
     stop
   endif

   allocate(tmp2(max_local_cols*nbw), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error when allocating tmp2 "//errorMessage
     stop
   endif

   allocate(hvb(max_local_rows*nbw), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error when allocating hvb "//errorMessage
     stop
   endif

   allocate(hvm(max_local_rows,nbw), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error when allocating hvm "//errorMessage
     stop
   endif

#ifdef WITH_GPU_VERSION
!   allocate(q_temp(ldq,max_local_cols), stat=istat, errmsg=errorMessage)
!   if (istat .ne. 0) then
!     print *,"trans_ev_band_to_full_complex: error when allocating q_temp "//errorMessage
!   endif

   allocate(tmat_temp(nbw,nbw), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error when allocating tmat_temp "//errorMessage
   endif

   istat = cuda_malloc(hvm_dev, max_local_rows*nbw*16_8)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error in cudaMalloc"
     stop
   endif

   istat = cuda_malloc(tmat_dev, nbw*nbw*16_8)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error in cudaMalloc"
     stop
   endif

   istat = cuda_malloc(q_dev, ldq*matrixCols*16_8)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error in cudaMalloc"
     stop
   endif

   istat = cuda_malloc(tmp_dev, max_local_cols*nbw*16_8)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error in cudaMalloc"
     stop
   endif

!!e   istat = cuda_memset(tmp_dev, 0, (max_local_rows)*(nbw)*16_8)
!   istat = cuda_memset(tmp_dev, 0, (max_local_cols)*(nbw)*16_8)
!   if (istat .ne. 0) then
!     print *,"trans_ev_band_to_full_complex: error in cudaMalloc"
!     stop
!   endif
#endif

   hvm = 0   ! Must be set to 0 !!!
   hvb = 0   ! Safety only

#ifdef WITH_GPU_VERSION
!   q_temp(:,:) = 0.0
!   q_temp(1:ldq,1:na_cols) = q(1:ldq,1:na_cols)

   istat = cuda_memcpy(q_dev, loc(q),ldq*matrixCols*16_8, cudaMemcpyHostToDevice)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error in cudaMemcpy"
     stop
   endif

   istat = cuda_memset(hvm_dev, 0, (max_local_rows)*(nbw)*16_8)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error in cudaMemset"
     stop
   endif

#endif

   l_cols = local_index(nqc, my_pcol, np_cols, nblk, -1) ! Local columns of q

   do istep=1,(na-1)/nbw

     n_cols = MIN(na,(istep+1)*nbw) - istep*nbw ! Number of columns in current step

     ! Broadcast all Householder vectors for current step compressed in hvb

     nb = 0
     ns = 0

     do lc = 1, n_cols
       ncol = istep*nbw + lc ! absolute column number of householder vector
       nrow = ncol - nbw ! absolute number of pivot row

       l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast
       l_colh = local_index(ncol  , my_pcol, np_cols, nblk, -1) ! HV local column number

       if (my_pcol==pcol(ncol, nblk, np_cols)) hvb(nb+1:nb+l_rows) = a(1:l_rows,l_colh)

       nb = nb+l_rows

       if (lc==n_cols .or. mod(ncol,nblk)==0) then
         call MPI_Bcast(hvb(ns+1),nb-ns,MPI_DOUBLE_COMPLEX,pcol(ncol, nblk, np_cols),mpi_comm_cols,mpierr)
         ns = nb
       endif
     enddo

     ! Expand compressed Householder vectors into matrix hvm

     nb = 0
     do lc = 1, n_cols
       nrow = (istep-1)*nbw+lc ! absolute number of pivot row
       l_rows = local_index(nrow-1, my_prow, np_rows, nblk, -1) ! row length for bcast

       hvm(1:l_rows,lc) = hvb(nb+1:nb+l_rows)
       if (my_prow==prow(nrow, nblk, np_rows)) hvm(l_rows+1,lc) = 1.

       nb = nb+l_rows
     enddo
#ifdef WITH_GPU_VERSION
     istat =  cuda_memcpy(hvm_dev,loc(hvm),(max_local_rows*nbw*16_8),cudaMemcpyHostToDevice)
     if (istat .ne. 0) then
       print *,"trans_ev_band_to_full_complex: error in cudaMemcpy"
       stop
     endif

#endif
     l_rows = local_index(MIN(na,(istep+1)*nbw), my_prow, np_rows, nblk, -1)

     ! Q = Q - V * T**T * V**T * Q

     if (l_rows>0) then
#ifdef WITH_GPU_VERSION
       call cublas_zgemm('C','N',n_cols,l_cols,l_rows,CONE,hvm_dev,max_local_rows, &
                 q_dev,ldq,CZERO,tmp_dev,n_cols)
       istat = cuda_memcpy(loc(tmp1), tmp_dev, n_cols*l_cols*16_8, &
                           cudaMemcpyDeviceToHost)
       if (istat .ne. 0) then
         print *,"trans_ev_band_to_full_complex: error in cudaMemcpy"
         stop
       endif
#else
       call zgemm('C','N',n_cols,l_cols,l_rows,CONE,hvm,ubound(hvm,dim=1), &
                   q,ldq,CZERO,tmp1,n_cols)
#endif
     else
#ifdef WITH_GPU_VERSION
       if (l_cols*n_cols .gt. (max_local_cols)*(nbw)) then
         print *,"trans_ev_band_to_full_complex: tmp_dev ",l_cols*n_cols,max_local_cols
         stop
       endif

!       istat = cuda_memset(tmp_dev, 0, l_cols*n_cols*16_8)
!       if (istat .ne. 0) then
!         print *,"trans_ev_band_to_full_complex: error in cudaMemset"
!         stop
!       endif
#endif

       tmp1(1:l_cols*n_cols) = 0

     endif

     call mpi_allreduce(tmp1,tmp2,n_cols*l_cols,MPI_DOUBLE_COMPLEX,MPI_SUM,mpi_comm_rows,mpierr)

     if (l_rows>0) then
#ifdef WITH_GPU_VERSION


       istat = cuda_memcpy(tmp_dev,loc(tmp2),l_cols*n_cols*16_8,cudaMemcpyHostToDevice)
       if (istat .ne. 0) then
         print *,"trans_ev_band_to_full_complex: error in cudaMemcpy"
         stop
       endif



       tmat_temp(1:nbw,1:nbw) = tmat(1:nbw,1:nbw,istep)

       istat = cuda_memcpy(tmat_dev, loc(tmat_temp(1,1)),nbw*nbw*16_8,cudaMemcpyHostToDevice)
       if (istat .ne. 0) then
         print *,"trans_ev_band_to_full_complex: error in cudaMemcpy"
         stop
       endif

       call cublas_ztrmm('L','U','C','N',n_cols,l_cols,CONE,tmat_dev,nbw,tmp_dev,n_cols)
       call cublas_zgemm('N','N',l_rows,l_cols,n_cols,-CONE,hvm_dev, max_local_rows, &
                    tmp_dev,n_cols,CONE,q_dev,ldq)
#else
       call ztrmm('L','U','C','N',n_cols,l_cols,CONE,tmat(1,1,istep),ubound(tmat,dim=1),tmp2,n_cols)
       call zgemm('N','N',l_rows,l_cols,n_cols,-CONE,hvm,ubound(hvm,dim=1), &
                   tmp2,n_cols,CONE,q,ldq)
#endif
     endif

!#ifdef WITH_GPU_VERSION
!     istat =cuda_memcpy(loc(hvm(1,1)),hvm_dev,((max_local_rows)*nbw*16_8),cudaMemcpyDeviceToHost)
!     if (istat .ne. 0) then
!       print *,"trans_ev_band_to_full_complex: error in cudaMemcpy"
!       stop
!     endif
!#endif

   enddo

   deallocate(tmp1, tmp2, hvb, hvm, stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error when deallocating tmp1, tmp2, hvb, hvm "//errorMessage
     stop
   endif

#ifdef WITH_GPU_VERSION

   istat = cuda_free(hvm_dev)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error in cudaFree"
     stop
   endif

   istat = cuda_free(tmp_dev)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error in cudaFree"
     stop
   endif

   istat = cuda_free(tmat_dev)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error in cudaFree"
     stop
   endif

   istat = cuda_memcpy(loc(q), q_dev,ldq*matrixCols*16_8, cudaMemcpyDeviceToHost)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error in cudaMemcpy"
     stop
   endif
!   q(1:ldq,1:na_cols) = q_temp(1:ldq,1:na_cols)

   istat = cuda_free(q_dev)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error in cudaFree"
     stop
   endif

!   deallocate(q_temp, stat=istat, errmsg=errorMessage)
!   if (istat .ne. 0) then
!     print *,"trans_ev_band_to_full_complex: error when deallocating q_temp "//errorMessage
!   endif

   deallocate(tmat_temp, stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"trans_ev_band_to_full_complex: error when deallocating tmat_temp "//errorMessage
   endif

#endif

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("trans_ev_band_to_full_complex")
#endif

 end subroutine trans_ev_band_to_full_complex
!---------------------------------------------------------------------------------------------------

subroutine tridiag_band_complex(na, nb, nblk, a, lda, d, e, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm)

!-------------------------------------------------------------------------------
! tridiag_band_complex:
! Reduces a complex hermitian symmetric band matrix to tridiagonal form
!
!  na          Order of matrix a
!
!  nb          Semi bandwith
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  a(lda,matrixCols)    Distributed system matrix reduced to banded form in the upper diagonal
!
!  lda         Leading dimension of a
!  matrixCols  local columns of matrix a
!
!  d(na)       Diagonal of tridiagonal matrix, set only on PE 0 (output)
!
!  e(na)       Subdiagonal of tridiagonal matrix, set only on PE 0 (output)
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns
!  mpi_comm
!              MPI-Communicator for the total processor set
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
   implicit none

   integer, intent(in)      ::  na, nb, nblk, lda, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm
   complex*16, intent(in)   :: a(lda,matrixCols)
   real*8, intent(out)      :: d(na), e(na) ! set only on PE 0

   integer                  :: mpierr
   real*8                   :: vnorm2
   complex*16               :: hv(nb), tau, x, h(nb), ab_s(1+nb), hv_s(nb), hv_new(nb), tau_new, hf
   complex*16               :: hd(nb), hs(nb)

!#ifdef WITH_GPU_VERSION
!   integer(C_SIZE_T)        :: h_dev, hv_new_dev ,ab_dev,x_dev,hs_dev,tau_new_dev,hv_dev,hd_dev
!   complex*16, allocatable  :: ab_temp(:,:)
!#endif

   integer                  :: i, j, n, nc, nr, ns, ne, istep, iblk, nblocks_total, nblocks, nt
   integer                  :: my_pe, n_pes, mpier
   integer                  :: my_prow, np_rows, my_pcol, np_cols
   integer                  :: ireq_ab, ireq_hv
   integer                  :: na_s, nx, num_hh_vecs, num_chunks, local_size, max_blk_size, n_off
#ifdef WITH_OPENMP
    integer, allocatable    :: mpi_statuses(:,:)
    integer, allocatable    :: omp_block_limits(:)
    integer                 :: max_threads, my_thread, my_block_s, my_block_e, iter
    integer                 :: omp_get_max_threads
    integer                 :: mpi_status(MPI_STATUS_SIZE)
    complex*16, allocatable :: hv_t(:,:), tau_t(:)
#endif
   integer, allocatable     :: ireq_hhr(:), ireq_hhs(:), global_id(:,:), hh_cnt(:), hh_dst(:)
   integer, allocatable     :: limits(:), snd_limits(:,:)
   integer, allocatable     :: block_limits(:)
   complex*16, allocatable  :: ab(:,:), hh_gath(:,:,:), hh_send(:,:,:)
   integer                  :: istat
   character(200)           :: errorMessage
!   ! dummies for calling redist_band
!   real*8                   :: r_a(1,1), r_ab(1,1)

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("tridiag_band_complex")
#endif
   call mpi_comm_rank(mpi_comm,my_pe,mpierr)
   call mpi_comm_size(mpi_comm,n_pes,mpierr)

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

!#ifdef WITH_GPU_VERSION
!   t_1 = 0
!   t_2 = 0
!#endif
   ! Get global_id mapping 2D procssor coordinates to global id

   allocate(global_id(0:np_rows-1,0:np_cols-1), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_complex: error when allocating global_id "//errorMessage
     stop
   endif
   global_id(:,:) = 0
   global_id(my_prow, my_pcol) = my_pe

   call mpi_allreduce(mpi_in_place, global_id, np_rows*np_cols, mpi_integer, mpi_sum, mpi_comm, mpierr)


   ! Total number of blocks in the band:

   nblocks_total = (na-1)/nb + 1

   ! Set work distribution

   allocate(block_limits(0:n_pes), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_complex: error when allocating block_limits "//errorMessage
     stop
   endif

   call divide_band(nblocks_total, n_pes, block_limits)

   ! nblocks: the number of blocks for my task
   nblocks = block_limits(my_pe+1) - block_limits(my_pe)

   ! allocate the part of the band matrix which is needed by this PE
   ! The size is 1 block larger than needed to avoid extensive shifts
   allocate(ab(2*nb,(nblocks+1)*nb), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_complex: error when allocating ab "//errorMessage
     stop
   endif

!#ifdef WITH_GPU_VERSION
!   allocate(ab_temp(2*nb,nblocks*nb), stat=istat, errmsg=errorMessage)
!   if (istat .ne. 0) then
!     print *,"error when allocating ab_temp "//errorMessage
!     stop
!   endif
!#endif
   ab = 0 ! needed for lower half, the extra block should also be set to 0 for safety



!#ifdef WITH_GPU_VERSION
!
!   istat = cuda_malloc(ab_dev, 2*nb*(nblocks+1)*nb*16_8)
!   if (istat .ne. 0) print *, " cuda malloc failed ab_dev", istat
!
!   istat = cuda_malloc(hv_new_dev, nb*16_8 )
!   if (istat .ne. 0) print *, " cuda malloc failed hv_new_dev", istat
!
!!   istat = cuda_malloc(temp_c_dev,  nb*nb*16_8 )
!!   if(istat .ne. 0) print *, " cuda malloc failed temp_c", istat
!
!   istat = cuda_malloc(h_dev , nb*16_8)
!   if (istat .ne. 0) print *, " cuda malloc failed h_dev", istat
!
!   istat = cuda_malloc(hs_dev , nb*16_8)
!   if (istat .ne. 0) print *, " cuda malloc failed hs_dev", istat
!
!   istat = cuda_malloc(x_dev , 1*16_8)
!   if (istat .ne. 0) print *, " cuda malloc failed x_dev", istat
!
!   istat = cuda_malloc( tau_new_dev , 1*16_8)
!   if (istat .ne. 0) print *, " cuda malloc failed tau_new_dev", istat
!
!   istat = cuda_malloc(hv_dev , nb*16_8)
!   if (istat .ne. 0) print *, " cuda malloc failed hv_dev", istat
!
!   istat = cuda_malloc(hd_dev , nb*16_8)
!   if (istat .ne. 0) print *, " cuda malloc failed hd_dev", istat
!#endif

   ! n_off: Offset of ab within band
   n_off = block_limits(my_pe)*nb

   ! Redistribute band in a to ab
   call redist_band_complex(a, lda, na, nblk, nb, matrixCols, mpi_comm_rows, mpi_comm_cols, mpi_comm, ab)

   ! Calculate the workload for each sweep in the back transformation
   ! and the space requirements to hold the HH vectors

   allocate(limits(0:np_rows), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_complex: error when allocating limits "//errorMessage
     stop
   endif

   call determine_workload(na, nb, np_rows, limits)
   max_blk_size = maxval(limits(1:np_rows) - limits(0:np_rows-1))

   num_hh_vecs = 0
   num_chunks  = 0
   nx = na
   do n = 1, nblocks_total
     call determine_workload(nx, nb, np_rows, limits)
     local_size = limits(my_prow+1) - limits(my_prow)
     ! add to number of householder vectors
     ! please note: for nx==1 the one and only HH vector is 0 and is neither calculated nor send below!
     if (mod(n-1,np_cols) == my_pcol .and. local_size>0 .and. nx>1) then
       num_hh_vecs = num_hh_vecs + local_size
       num_chunks  = num_chunks+1
     endif
     nx = nx - nb
   enddo

   ! Allocate space for HH vectors

   allocate(hh_trans_complex(nb,num_hh_vecs), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_complex: error when allocating hh_trans_comples "//errorMessage
     stop
   endif
   ! Allocate and init MPI requests

   allocate(ireq_hhr(num_chunks), stat=istat, errmsg=errorMessage) ! Recv requests
   if (istat .ne. 0) then
     print *,"tridiag_band_complex: error when allocating ireq_hhr "//errorMessage
     stop
   endif

   allocate(ireq_hhs(nblocks), stat=istat, errmsg=errorMessage)    ! Send requests
   if (istat .ne. 0) then
     print *,"tridiag_band_complex: error when allocating ireq_hhs "//errorMessage
     stop
   endif

   num_hh_vecs = 0
   num_chunks  = 0
   nx = na
   nt = 0
   do n = 1, nblocks_total
     call determine_workload(nx, nb, np_rows, limits)
     local_size = limits(my_prow+1) - limits(my_prow)
     if (mod(n-1,np_cols) == my_pcol .and. local_size>0 .and. nx>1) then
       num_chunks  = num_chunks+1
       call mpi_irecv(hh_trans_complex(1,num_hh_vecs+1), nb*local_size, MPI_COMPLEX16, nt, &
                        10+n-block_limits(nt), mpi_comm, ireq_hhr(num_chunks), mpierr)
       num_hh_vecs = num_hh_vecs + local_size
     endif
     nx = nx - nb
     if (n == block_limits(nt+1)) then
       nt = nt + 1
     endif
   enddo

   ireq_hhs(:) = MPI_REQUEST_NULL

   ! Buffers for gathering/sending the HH vectors

   allocate(hh_gath(nb,max_blk_size,nblocks), stat=istat, errmsg=errorMessage) ! gathers HH vectors
   if (istat .ne. 0) then
     print *,"tridiag_band_complex: error when allocating hh_gath "//errorMessage
     stop
   endif

   allocate(hh_send(nb,max_blk_size,nblocks), stat=istat, errmsg=errorMessage) ! send buffer for HH vectors
   if (istat .ne. 0) then
     print *,"tridiag_band_complex: error when allocating hh_sebd "//errorMessage
     stop
   endif

   hh_gath(:,:,:) = 0
   hh_send(:,:,:) = 0

   ! Some counters

   allocate(hh_cnt(nblocks), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_complex: error when allocating hh_cnt "//errorMessage
     stop
   endif
   allocate(hh_dst(nblocks), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_complex: error when allocating hh_dst "//errorMessage
     stop
   endif

   hh_cnt(:) = 1 ! The first transfomation vector is always 0 and not calculated at all
   hh_dst(:) = 0 ! PE number for receive

   ireq_ab = MPI_REQUEST_NULL
   ireq_hv = MPI_REQUEST_NULL

   ! Limits for sending

   allocate(snd_limits(0:np_rows,nblocks), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"tridiag_band_complex: error when allocating snd_limits "//errorMessage
     stop
   endif
   do iblk=1,nblocks
     call determine_workload(na-(iblk+block_limits(my_pe)-1)*nb, nb, np_rows, snd_limits(:,iblk))
   enddo

#ifdef WITH_OPENMP
    ! OpenMP work distribution:

    max_threads = 1
!$ max_threads = omp_get_max_threads()

    ! For OpenMP we need at least 2 blocks for every thread
    max_threads = MIN(max_threads, nblocks/2)
    if (max_threads==0) max_threads = 1

    allocate(omp_block_limits(0:max_threads), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"tridiag_band_complex: error when allocating omp_block_limits "//errorMessage
      stop
    endif

    ! Get the OpenMP block limits
    call divide_band(nblocks, max_threads, omp_block_limits)

    allocate(hv_t(nb,max_threads), tau_t(max_threads), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"tridiag_band_complex: error when allocating hv_t, tau_t "//errorMessage
      stop
    endif
    hv_t = 0
    tau_t = 0
#endif


   ! ---------------------------------------------------------------------------
   ! Start of calculations

   na_s = block_limits(my_pe)*nb + 1

   if (my_pe>0 .and. na_s<=na) then
     ! send first column to previous PE
     ! Only the PE owning the diagonal does that (sending 1 element of the subdiagonal block also)
     ab_s(1:nb+1) = ab(1:nb+1,na_s-n_off)
     call mpi_isend(ab_s,nb+1,MPI_COMPLEX16,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
   endif

#ifdef WITH_OPENMP
   do istep=1,na-1-block_limits(my_pe)*nb
#else
   do istep=1,na-1
#endif
     if (my_pe==0) then
       n = MIN(na-na_s,nb) ! number of rows to be reduced
       hv(:) = 0
       tau = 0
       ! Transform first column of remaining matrix
       ! Opposed to the real case, the last step (istep=na-1) is needed here for making
       ! the last subdiagonal element a real number
       vnorm2 = sum(dble(ab(3:n+1,na_s-n_off))**2+dimag(ab(3:n+1,na_s-n_off))**2)
       if (n<2) vnorm2 = 0. ! Safety only
       call hh_transform_complex(ab(2,na_s-n_off),vnorm2,hf,tau)

       hv(1) = 1
       hv(2:n) = ab(3:n+1,na_s-n_off)*hf

       d(istep) = ab(1,na_s-n_off)
       e(istep) = ab(2,na_s-n_off)
       if (istep == na-1) then
         d(na) = ab(1,na_s+1-n_off)
         e(na) = 0
       endif
     else
       if (na>na_s) then
         ! Receive Householder vector from previous task, from PE owning subdiagonal
#ifdef WITH_OPENMP
         call mpi_recv(hv,nb,MPI_COMPLEX16,my_pe-1,2,mpi_comm,mpi_status,mpierr)
#else
         call mpi_recv(hv,nb,MPI_COMPLEX16,my_pe-1,2,mpi_comm,MPI_STATUS_IGNORE,mpierr)
#endif
         tau = hv(1)
         hv(1) = 1.
       endif
     endif

     na_s = na_s+1
     if (na_s-n_off > nb) then
!#ifdef WITH_GPU_VERSION
!       ab_temp(:,1:nblocks*nb) =  ab(:,nb+1:(nblocks +1)*nb)
!       ab(:, 1:nblocks*nb) = ab_temp(:, 1:nblocks*nb)
!#else
       ab(:,1:nblocks*nb) = ab(:,nb+1:(nblocks+1)*nb)
!#endif
       ab(:,nblocks*nb+1:(nblocks+1)*nb) = 0
       n_off = n_off + nb
     endif
#ifdef WITH_OPENMP
     if (max_threads > 1) then

       ! Codepath for OpenMP

       ! Please note that in this case it is absolutely necessary to have at least 2 blocks per thread!
       ! Every thread is one reduction cycle behind its predecessor and thus starts one step later.
       ! This simulates the behaviour of the MPI tasks which also work after each other.
       ! The code would be considerably easier, if the MPI communication would be made within
       ! the parallel region - this is avoided here since this would require
       ! MPI_Init_thread(MPI_THREAD_MULTIPLE) at the start of the program.

       hv_t(:,1) = hv
       tau_t(1) = tau

       do iter = 1, 2

         ! iter=1 : work on first block
         ! iter=2 : work on remaining blocks
         ! This is done in 2 iterations so that we have a barrier in between:
         ! After the first iteration, it is guaranteed that the last row of the last block
         ! is completed by the next thread.
         ! After the first iteration it is also the place to exchange the last row
         ! with MPI calls
#ifdef HAVE_DETAILED_TIMINGS
         call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, my_block_s, my_block_e, iblk, ns, ne, hv, tau, &
!$omp&                    nc, nr, hs, hd, vnorm2, hf, x, h, i), schedule(static,1), num_threads(max_threads)
         do my_thread = 1, max_threads

           if (iter == 1) then
             my_block_s = omp_block_limits(my_thread-1) + 1
             my_block_e = my_block_s
           else
             my_block_s = omp_block_limits(my_thread-1) + 2
             my_block_e = omp_block_limits(my_thread)
           endif

           do iblk = my_block_s, my_block_e

             ns = na_s + (iblk-1)*nb - n_off - my_thread + 1 ! first column in block
             ne = ns+nb-1                    ! last column in block

             if (istep<my_thread .or. ns+n_off>na) exit

             hv = hv_t(:,my_thread)
             tau = tau_t(my_thread)

             ! Store Householder vector for back transformation

             hh_cnt(iblk) = hh_cnt(iblk) + 1

             hh_gath(1   ,hh_cnt(iblk),iblk) = tau
             hh_gath(2:nb,hh_cnt(iblk),iblk) = hv(2:nb)

             nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
             nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                            ! Note that nr>=0 implies that diagonal block is full (nc==nb)!

             ! Transform diagonal block

             call ZHEMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,(0.d0,0.d0),hd,1)

             x = dot_product(hv(1:nc),hd(1:nc))*conjg(tau)
             hd(1:nc) = hd(1:nc) - 0.5*x*hv(1:nc)

             call ZHER2('L',nc,(-1.d0,0.d0),hd,1,hv,1,ab(1,ns),2*nb-1)

             hv_t(:,my_thread) = 0
             tau_t(my_thread)  = 0

             if (nr<=0) cycle ! No subdiagonal block present any more

             ! Transform subdiagonal block

             call ZGEMV('N',nr,nb,tau,ab(nb+1,ns),2*nb-1,hv,1,(0.d0,0.d0),hs,1)

             if (nr>1) then

               ! complete (old) Householder transformation for first column

               ab(nb+1:nb+nr,ns) = ab(nb+1:nb+nr,ns) - hs(1:nr) ! Note: hv(1) == 1

               ! calculate new Householder transformation for first column
               ! (stored in hv_t(:,my_thread) and tau_t(my_thread))

               vnorm2 = sum(dble(ab(nb+2:nb+nr,ns))**2+dimag(ab(nb+2:nb+nr,ns))**2)
               call hh_transform_complex(ab(nb+1,ns),vnorm2,hf,tau_t(my_thread))
               hv_t(1   ,my_thread) = 1.
               hv_t(2:nr,my_thread) = ab(nb+2:nb+nr,ns)*hf
               ab(nb+2:,ns) = 0

               ! update subdiagonal block for old and new Householder transformation
               ! This way we can use a nonsymmetric rank 2 update which is (hopefully) faster

               call ZGEMV('C',nr,nb-1,tau_t(my_thread),ab(nb,ns+1),2*nb-1,hv_t(1,my_thread),1,(0.d0,0.d0),h(2),1)
               x = dot_product(hs(1:nr),hv_t(1:nr,my_thread))*tau_t(my_thread)
               h(2:nb) = h(2:nb) - x*hv(2:nb)
               ! Unfortunately there is no BLAS routine like DSYR2 for a nonsymmetric rank 2 update ("DGER2")
               do i=2,nb
                 ab(2+nb-i:1+nb+nr-i,i+ns-1) = ab(2+nb-i:1+nb+nr-i,i+ns-1) &
                                                - hv_t(1:nr,my_thread)*conjg(h(i)) - hs(1:nr)*conjg(hv(i))
               enddo

             else

               ! No new Householder transformation for nr=1, just complete the old one
               ab(nb+1,ns) = ab(nb+1,ns) - hs(1) ! Note: hv(1) == 1
               do i=2,nb
                 ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*conjg(hv(i))
               enddo
               ! For safety: there is one remaining dummy transformation (but tau is 0 anyways)
               hv_t(1,my_thread) = 1.

             endif

           enddo

         enddo ! my_thread
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
         call timer%stop("OpenMP parallel")
#endif



         if (iter==1) then
           ! We are at the end of the first block

           ! Send our first column to previous PE
           if (my_pe>0 .and. na_s <= na) then
             call mpi_wait(ireq_ab,mpi_status,mpierr)
             ab_s(1:nb+1) = ab(1:nb+1,na_s-n_off)
             call mpi_isend(ab_s,nb+1,MPI_COMPLEX16,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
           endif

           ! Request last column from next PE
           ne = na_s + nblocks*nb - (max_threads-1) - 1
           if (istep>=max_threads .and. ne <= na) then
             call mpi_recv(ab(1,ne-n_off),nb+1,MPI_COMPLEX16,my_pe+1,1,mpi_comm,mpi_status,mpierr)
           endif

         else
           ! We are at the end of all blocks

           ! Send last HH vector and TAU to next PE if it has been calculated above
           ne = na_s + nblocks*nb - (max_threads-1) - 1
           if (istep>=max_threads .and. ne < na) then
             call mpi_wait(ireq_hv,mpi_status,mpierr)
             hv_s(1) = tau_t(max_threads)
             hv_s(2:) = hv_t(2:,max_threads)
             call mpi_isend(hv_s,nb,MPI_COMPLEX16,my_pe+1,2,mpi_comm,ireq_hv,mpierr)
           endif

           ! "Send" HH vector and TAU to next OpenMP thread
           do my_thread = max_threads, 2, -1
             hv_t(:,my_thread) = hv_t(:,my_thread-1)
             tau_t(my_thread)  = tau_t(my_thread-1)
           enddo

         endif
       enddo ! iter

     else

       ! Codepath for 1 thread without OpenMP

       ! The following code is structured in a way to keep waiting times for
       ! other PEs at a minimum, especially if there is only one block.
       ! For this reason, it requests the last column as late as possible
       ! and sends the Householder vector and the first column as early
       ! as possible.

#endif /* WITH_OPENMP */

!#ifdef WITH_GPU_VERSION
!       call cpu_time(start)
!#endif
       do iblk=1,nblocks

         ns = na_s + (iblk-1)*nb - n_off ! first column in block
         ne = ns+nb-1                    ! last column in block

         if (ns+n_off>na) exit

         ! Store Householder vector for back transformation

         hh_cnt(iblk) = hh_cnt(iblk) + 1

         hh_gath(1   ,hh_cnt(iblk),iblk) = tau
         hh_gath(2:nb,hh_cnt(iblk),iblk) = hv(2:nb)


#ifndef WITH_OPENMP
         if (hh_cnt(iblk) == snd_limits(hh_dst(iblk)+1,iblk)-snd_limits(hh_dst(iblk),iblk)) then
           ! Wait for last transfer to finish
           call mpi_wait(ireq_hhs(iblk), MPI_STATUS_IGNORE, mpierr)
           ! Copy vectors into send buffer
           hh_send(:,1:hh_cnt(iblk),iblk) = hh_gath(:,1:hh_cnt(iblk),iblk)
           ! Send to destination
           call mpi_isend(hh_send(1,1,iblk), nb*hh_cnt(iblk), MPI_COMPLEX16, &
                           global_id(hh_dst(iblk),mod(iblk+block_limits(my_pe)-1,np_cols)), &
                           10+iblk, mpi_comm, ireq_hhs(iblk), mpierr)
           ! Reset counter and increase destination row
           hh_cnt(iblk) = 0
           hh_dst(iblk) = hh_dst(iblk)+1
         endif


         ! The following code is structured in a way to keep waiting times for
         ! other PEs at a minimum, especially if there is only one block.
         ! For this reason, it requests the last column as late as possible
         ! and sends the Householder vector and the first column as early
         ! as possible.
#endif /* OpenMP */

         nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
         nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                       ! Note that nr>=0 implies that diagonal block is full (nc==nb)!


         ! Multiply diagonal block and subdiagonal block with Householder vector

         if (iblk==nblocks .and. nc==nb) then

           ! We need the last column from the next PE.
           ! First do the matrix multiplications without last column ...

           ! Diagonal block, the contribution of the last element is added below!
           ab(1,ne) = 0

           call ZHEMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,(0.d0,0.d0),hd,1)

           ! Subdiagonal block
           if (nr>0) call ZGEMV('N',nr,nb-1,tau,ab(nb+1,ns),2*nb-1,hv,1,(0.d0,0.d0),hs,1)

           ! ... then request last column ...
#ifdef WITH_OPENMP
           call mpi_recv(ab(1,ne),nb+1,MPI_COMPLEX16,my_pe+1,1,mpi_comm,mpi_status,mpierr)

#else
           call mpi_recv(ab(1,ne),nb+1,MPI_COMPLEX16,my_pe+1,1,mpi_comm,MPI_STATUS_IGNORE,mpierr)
#endif
           ! ... and complete the result
           hs(1:nr) = hs(1:nr) + ab(2:nr+1,ne)*tau*hv(nb)
           hd(nb) = hd(nb) + ab(1,ne)*hv(nb)*tau

         else
           ! Normal matrix multiply
           call ZHEMV('L',nc,tau,ab(1,ns),2*nb-1,hv,1,(0.d0,0.d0),hd,1)
           if (nr>0) call ZGEMV('N',nr,nb,tau,ab(nb+1,ns),2*nb-1,hv,1,(0.d0,0.d0),hs,1)

         endif

           ! Calculate first column of subdiagonal block and calculate new
           ! Householder transformation for this column

           hv_new(:) = 0 ! Needed, last rows must be 0 for nr < nb
!#ifdef WITH_GPU_VERSION
!           istat = cuda_memset(hv_new_dev, 0,nb*16_8 )
!           if (istat .ne. 0) print *, " cuda memset failed hv_new_dev", istat
!#endif
           tau_new = 0

           if (nr>0) then

             ! complete (old) Householder transformation for first column

             ab(nb+1:nb+nr,ns) = ab(nb+1:nb+nr,ns) - hs(1:nr) ! Note: hv(1) == 1

             ! calculate new Householder transformation ...
             if (nr>1) then
               vnorm2 = sum(dble(ab(nb+2:nb+nr,ns))**2+dimag(ab(nb+2:nb+nr,ns))**2)
               call hh_transform_complex(ab(nb+1,ns),vnorm2,hf,tau_new)
               hv_new(1) = 1.
               hv_new(2:nr) = ab(nb+2:nb+nr,ns)*hf
               ab(nb+2:,ns) = 0
             endif

             ! ... and send it away immediatly if this is the last block

             if (iblk==nblocks) then
#ifdef WITH_OPENMP
               call mpi_wait(ireq_hv,mpi_status,mpierr)
#else
               call mpi_wait(ireq_hv,MPI_STATUS_IGNORE,mpierr)
#endif
               hv_s(1) = tau_new
               hv_s(2:) = hv_new(2:)
               call mpi_isend(hv_s,nb,MPI_COMPLEX16,my_pe+1,2,mpi_comm,ireq_hv,mpierr)
             endif

           endif


          ! Transform diagonal block
          x = dot_product(hv(1:nc),hd(1:nc))*conjg(tau)
          hd(1:nc) = hd(1:nc) - 0.5*x*hv(1:nc)

!#ifdef WITH_GPU_VERSION
!         istat = cuda_memcpy2d((ab_dev +  (ns-1)*2*nb*16_8), 2*nb*16_8,loc(a(1,ns)), 2*nb*16_8, 2*16_8 , &
!                               2*nb*16_8,cudaMemcpyHostToDevice)
!         if (istat .ne. 0) print *, "cuda memcpy a_dev H2D failed ", istat
!         istat =cuda_memcpy(hv_dev,loc(hv),nc*16_8,cudaMemcpyHostToDevice)
!         if (istat .ne. 0) print *,"cuda memcpy failed hv_dev", istat
!         istat =cuda_memcpy(hd_dev,loc(hd), nb*16_8,cudaMemcpyHostToDevice)
!         if (istat .ne. 0) print *,"cuda memcpy failed hd_dev", istat
!#endif

          if (my_pe>0 .and. iblk==1) then

            ! The first column of the diagonal block has to be send to the previous PE
            ! Calculate first column only ...

!#ifdef WITH_GPU_VERSION
!            call double_hh_transform_2( ns, nc, nb  )
!            istat=cuda_memcpy(loc(ab),ab_dev,(2*nb*(nblocks+1)*nb)*16_8,cudaMemcpyDeviceToHost)
!            if (istat .ne. 0) print *, " cuda memcpy failed ab ", istat
!#else
            ab(1:nc,ns) = ab(1:nc,ns) - hd(1:nc)*conjg(hv(1)) - hv(1:nc)*conjg(hd(1))
!#endif

            ! ... send it away ...
#ifdef WITH_OPENMP
            call mpi_wait(ireq_ab,mpi_status,mpierr)
#else
            call mpi_wait(ireq_ab,MPI_STATUS_IGNORE,mpierr)
#endif
            ab_s(1:nb+1) = ab(1:nb+1,ns)
            call mpi_isend(ab_s,nb+1,MPI_COMPLEX16,my_pe-1,1,mpi_comm,ireq_ab,mpierr)

            ! ... and calculate remaining columns with rank-2 update
            if (nc>1) then

!#ifdef WITH_GPU_VERSION
!              call cublas_ZHER2( 'L',nc -1,(-1.d0,0.d0), hd_dev + 1*16, 1, hv_dev +1*16, 1 , ab_dev + (ns*2*nb )*16, 2*nb-1)
!#else
              call ZHER2('L',nc-1,(-1.d0,0.d0),hd(2),1,hv(2),1,ab(1,ns+1),2*nb-1)
!#endif
            endif
          else

            ! No need to  send, just a rank-2 update
!#ifdef WITH_GPU_VERSION
!            call cublas_ZHER2( 'L',nc ,(-1.d0,0.d0), hd_dev, 1, hv_dev, 1 , ab_dev + ((ns-1)*2*nb )*16, 2*nb-1)
!#else
            call ZHER2('L',nc,(-1.d0,0.d0),hd,1,hv,1,ab(1,ns),2*nb-1)
!#endif
          endif

!#ifdef WITH_GPU_VERSION
!          istat=cuda_memcpy( loc(hd),hd_dev,nb*16_8,cudaMemcpyDeviceToHost)
!          if (istat .ne. 0) print *,"cuda memcpy failed hd_dev", istat
!#endif

          ! Do the remaining double Householder transformation on the subdiagonal block cols 2 ... nb

!#ifdef WITH_GPU_VERSION
!         istat =cuda_memcpy(hs_dev,loc(hs),nb*16_8,cudaMemcpyHostToDevice)
!         if (istat .ne. 0) print *,"cuda memcpy failed hs_dev", istat
!#endif

          if (nr>0) then
            if (nr>1) then
!#ifdef WITH_GPU_VERSION
!              istat = cuda_memcpy(hv_new_dev,loc(hv_new),nb*16_8,cudaMemcpyHostToDevice)
!              if (istat .ne. 0) print *,"cuda memcpy failed hv_new_dev", istat
!
!              istat = cuda_memcpy(h_dev,loc(h),nb*16_8,cudaMemcpyHostToDevice)
!              if (istat .ne. 0) print *,"cuda memcpy failed h_dev", istat
!
!              call cublas_ZGEMV('C',nr,nb-1,tau_new,ab_dev + (nb-1 + ns *2*nb)*16,2*nb-1,hv_new_dev,1,(0.d0,0.d0),h_dev + 1* 16,1)
!
!              istat = cuda_memcpy(tau_new_dev,loc(tau_new),1*16_8,cudaMemcpyHostToDevice)
!              if (istat .ne. 0) print *,"cuda memcpy failed tau_new_dev", istat
!
!              call dot_product_kernel(nr , tau_new)
!              call dot_product_kernel_1( nb, nr , ns)
!
!              istat = cuda_memcpy(loc(x),x_dev,1*16_8,cudaMemcpyDeviceToHost)
!              if (istat .ne. 0) print *, " cuda memcpy failed x_dev ", istat
!
!              istat =cuda_memcpy(loc(h),h_dev,nb*16_8,cudaMemcpyDeviceToHost)
!              if (istat .ne. 0) print *, " cuda memcpy failed h ", istat
!#else
              call ZGEMV('C',nr,nb-1,tau_new,ab(nb,ns+1),2*nb-1,hv_new,1,(0.d0,0.d0),h(2),1)
              x = dot_product(hs(1:nr),hv_new(1:nr))*tau_new
               h(2:nb) = h(2:nb) - x*hv(2:nb)
               ! Unfortunately there is no BLAS routine like DSYR2 for a nonsymmetric rank 2 update
               do i=2,nb
                 ab(2+nb-i:1+nb+nr-i,i+ns-1) = ab(2+nb-i:1+nb+nr-i,i+ns-1) - hv_new(1:nr)*conjg(h(i)) - hs(1:nr)*conjg(hv(i))
               enddo
!#endif
             else
               ! No double Householder transformation for nr=1, just complete the row
!#ifdef WITH_GPU_VERSION
!               call double_hh_transform_1(nb, ns)
!#else
               do i=2,nb
                 ab(2+nb-i,i+ns-1) = ab(2+nb-i,i+ns-1) - hs(1)*conjg(hv(i))
               enddo
!#endif
             endif
           endif

           ! Use new HH vector for the next block
           hv(:) = hv_new(:)
           tau = tau_new

         enddo
!#ifdef WITH_GPU_VERSION
!      call cpu_time(finish)
!      tstep2 = finish-start
!      t_2 = t_2 + tstep2
!#endif
#ifdef WITH_OPENMP
       endif
#endif

#ifdef WITH_OPENMP
       do iblk = 1, nblocks

         if (hh_dst(iblk) >= np_rows) exit
         if (snd_limits(hh_dst(iblk)+1,iblk) == snd_limits(hh_dst(iblk),iblk)) exit

         if (hh_cnt(iblk) == snd_limits(hh_dst(iblk)+1,iblk)-snd_limits(hh_dst(iblk),iblk)) then
           ! Wait for last transfer to finish
           call mpi_wait(ireq_hhs(iblk), mpi_status, mpierr)
           ! Copy vectors into send buffer
           hh_send(:,1:hh_cnt(iblk),iblk) = hh_gath(:,1:hh_cnt(iblk),iblk)
           ! Send to destination
           call mpi_isend(hh_send(1,1,iblk), nb*hh_cnt(iblk), mpi_complex16, &
                         global_id(hh_dst(iblk),mod(iblk+block_limits(my_pe)-1,np_cols)), &
                         10+iblk, mpi_comm, ireq_hhs(iblk), mpierr)
           ! Reset counter and increase destination row
           hh_cnt(iblk) = 0
           hh_dst(iblk) = hh_dst(iblk)+1
         endif
       enddo
#endif
     enddo
!#ifdef WITH_GPU_VERSION
!     call cpu_time(finish_1)
!     tstep1 = finish_1-start_1
!     t_1 = t_1 + tstep1
!#endif

     ! Finish the last outstanding requests
#ifdef WITH_OPENMP
     call mpi_wait(ireq_ab,mpi_status,mpierr)
     call mpi_wait(ireq_hv,mpi_status,mpierr)

     allocate(mpi_statuses(MPI_STATUS_SIZE,max(nblocks,num_chunks)), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"tridiag_band_complex: error when allocating mpi_statuses "//errorMessage
       stop
     endif
     call mpi_waitall(nblocks, ireq_hhs, mpi_statuses, mpierr)
     call mpi_waitall(num_chunks, ireq_hhr, mpi_statuses, mpierr)
     deallocate(mpi_statuses, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"tridiag_band_complex: error when deallocating mpi_statuses "//errorMessage
       stop
     endif

#else
     call mpi_wait(ireq_ab,MPI_STATUS_IGNORE,mpierr)
     call mpi_wait(ireq_hv,MPI_STATUS_IGNORE,mpierr)

     call mpi_waitall(nblocks, ireq_hhs, MPI_STATUSES_IGNORE, mpierr)
     call mpi_waitall(num_chunks, ireq_hhr, MPI_STATUSES_IGNORE, mpierr)

#endif
     call mpi_barrier(mpi_comm,mpierr)

     deallocate(ab, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"tridiag_band_complex: error when deallocating ab "//errorMessage
       stop
     endif

!#ifdef WITH_GPU_VERSION
!     deallocate(ab_temp, stat=istat, errmsg=errorMessage)
!     if (istat .ne. 0) then
!       print *,"error when deallocating ab_temp "//errorMessage
!       stop
!     endif
!
!#endif

     deallocate(ireq_hhr, ireq_hhs, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"tridiag_band_complex: error when deallocating ireq_hhr, ireq_hhs "//errorMessage
       stop
     endif

      deallocate(hh_cnt, hh_dst, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"tridiag_band_complex: error when deallocating hh_cnt, hh_dst "//errorMessage
        stop
      endif

      deallocate(hh_gath, hh_send, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"tridiag_band_complex: error when deallocating hh_gath, hh_send,  "//errorMessage
        stop
      endif

      deallocate(limits, snd_limits, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"tridiag_band_complex: error when deallocating limits, snd_limits  "//errorMessage
        stop
      endif

      deallocate(block_limits, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"tridiag_band_complex: error when deallocating block_limits,  "//errorMessage
        stop
      endif

      deallocate(global_id, stat=istat, errmsg=errorMessage)
      if (istat .ne. 0) then
        print *,"tridiag_band_complex: error when deallocating global_id,  "//errorMessage
        stop
      endif

!#ifdef WITH_GPU_VERSION
!     istat = cuda_free(ab_dev)
!     if (istat .ne. 0) then
!       print *,"error in cudaFree"
!       stop
!     endif
!
!     istat = cuda_free(hv_new_dev)
!     if (istat .ne. 0) then
!       print *,"error in cudaFree"
!       stop
!     endif
!
!     istat = cuda_free(hs_dev)
!     if (istat .ne. 0) then
!       print *,"error in cudaFree"
!       stop
!     endif
!
!     istat = cuda_free(h_dev)
!     if (istat .ne. 0) then
!       print *,"error in cudaFree"
!       stop
!     endif
!
!     istat = cuda_free(tau_new_dev)
!     if (istat .ne. 0) then
!       print *,"error in cudaFree"
!       stop
!     endif
!
!     istat = cuda_free(x_dev)
!     if (istat .ne. 0) then
!       print *,"error in cudaFree"
!       stop
!     endif
!
!#endif
#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("tridiag_band_complex")
#endif

!#ifdef WITH_GPU_VERSION
!   contains
!
!     subroutine dot_product_kernel(nr,tau_new)
!       implicit none
!       integer, intent(in)    :: nr
!       complex*16, intent(in) :: tau_new
!
!       call launch_dot_product_kernel( hs_dev,hv_new_dev,tau_new,x_dev,h_dev,hv_dev, nr )
!     end subroutine
!
!     subroutine dot_product_kernel_1( nb , nr , ns)
!       implicit none
!       integer, intent(in) ::  nb, nr, ns
!
!       call launch_dot_product_kernel_1(ab_dev,hs_dev, hv_new_dev,x_dev,h_dev,hv_dev,nb , nr, ns)
!     end subroutine
!
!     subroutine double_hh_transform_1( nb , ns)
!       implicit none
!       integer, intent(in) ::  nb, ns
!
!       call launch_double_hh_transform_1(ab_dev,hs_dev,hv_dev,nb , ns)
!     end subroutine
!
!     subroutine double_hh_transform_2( ns,nc, nb)
!       implicit none
!       integer, intent(in) ::  nc, ns, nb
!
!       call launch_double_hh_transform_2(ab_dev,hd_dev,hv_dev,nc , ns, nb)
!     end subroutine
!#endif
 end subroutine tridiag_band_complex ! has to be checked for GPU

!---------------------------------------------------------------------------------------------------

#define ATODEV istat = cuda_memcpy(loc(a), a_dev, stripe_width*a_dim2*stripe_count*16_8, cudaMemcpyDeviceToHost)
#define ATOHOST istat = cuda_memcpy(a_dev, loc(a), stripe_width*a_dim2*stripe_count*16_8, cudaMemcpyDeviceToHost)


subroutine trans_ev_tridi_to_band_complex(na, nev, nblk, nbw, q, ldq, matrixCols,  &
                                          mpi_comm_rows, mpi_comm_cols, &
                                          wantDebug, success, THIS_COMPLEX_ELPA_KERNEL)

!-------------------------------------------------------------------------------
!  trans_ev_tridi_to_band_complex:
!  Transforms the eigenvectors of a tridiagonal matrix back to the eigenvectors of the band matrix
!
!  Parameters
!
!  na          Order of matrix a, number of rows of matrix q
!
!  nev         Number eigenvectors to compute (= columns of matrix q)
!
!  nblk        blocksize of cyclic distribution, must be the same in both directions!
!
!  nb          semi bandwith
!
!  q           On input: Eigenvectors of tridiagonal matrix
!              On output: Transformed eigenvectors
!              Distribution is like in Scalapack.
!
!  ldq         Leading dimension of q
! matrixCols   local columns of matrix q
!
!  mpi_comm_rows
!  mpi_comm_cols
!              MPI-Communicators for rows/columns/both
!
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif
    implicit none

    integer, intent(in)     :: THIS_COMPLEX_ELPA_KERNEL
    integer, intent(in)     :: na, nev, nblk, nbw, ldq, matrixCols, mpi_comm_rows, mpi_comm_cols
    complex*16              :: q(ldq,matrixCols)

!    complex*16              :: q(ldq,*)

    integer                 :: np_rows, my_prow, np_cols, my_pcol
#ifdef WITH_GPU_VERSION
    integer                 :: tmp
#endif
    integer                 :: i, j, ip, sweep, nbuf, l_nev, a_dim2
    integer                 :: current_n, current_local_n, current_n_start, current_n_end
    integer                 :: next_n, next_local_n, next_n_start, next_n_end
    integer                 :: bottom_msg_length, top_msg_length, next_top_msg_length
    integer                 :: stripe_width, last_stripe_width, stripe_count
#ifdef WITH_OPENMP
    integer                 :: thread_width, csw, b_off, b_len
#endif
    integer                 :: num_result_blocks, num_result_buffers, num_bufs_recvd
    integer                 :: a_off, current_tv_off, max_blk_size
    integer                 :: mpierr, src, src_offset, dst, offset, nfact, num_blk
    logical                 :: flag
#ifdef WITH_GPU_VERSION
    integer                 :: n_times
#endif

#ifndef WITH_GPU_VERSION

#ifdef WITH_OPENMP
    complex*16, allocatable :: a(:,:,:,:), row(:)
#else
    complex*16, allocatable :: a(:,:,:), row(:)
#endif

#endif /* WITH_GPU_VERSION */

    complex*16, allocatable :: row(:)

#ifdef WITH_GPU_VERSION
    complex*16, allocatable :: row_group(:,:)
#endif

#ifdef WITH_OPENMP
    complex*16, allocatable :: top_border_send_buffer(:,:), top_border_recv_buffer(:,:)
    complex*16, allocatable :: bottom_border_send_buffer(:,:), bottom_border_recv_buffer(:,:)
#else
    complex*16, allocatable :: top_border_send_buffer(:,:,:), top_border_recv_buffer(:,:,:)
    complex*16, allocatable :: bottom_border_send_buffer(:,:,:), bottom_border_recv_buffer(:,:,:)
#endif
    complex*16, allocatable :: result_buffer(:,:,:)
    complex*16, allocatable :: bcast_buffer(:,:)
#ifdef WITH_GPU_VERSION
    integer(c_size_t)       :: a_dev
    integer(c_size_t)       :: bcast_buffer_dev
    integer(c_size_t)       :: num
    integer(c_size_t)       :: dev_offset, dev_offset_1, dev_offset_2


    integer(c_size_t)       :: row_dev
    integer(c_size_t)       :: row_group_dev
    integer(c_size_t)       :: hh_tau_dev
    integer(c_size_t)       :: hh_dot_dev
    integer                 :: row_group_size, unpack_idx

    integer                 :: top, chunk, this_chunk
#endif

    integer                 :: n_off
    integer, allocatable    :: result_send_request(:), result_recv_request(:), limits(:)
    integer, allocatable    :: top_send_request(:), bottom_send_request(:)
    integer, allocatable    :: top_recv_request(:), bottom_recv_request(:)
#ifdef WITH_OPENMP
    integer, allocatable    :: mpi_statuses(:,:)
    integer                 :: mpi_status(MPI_STATUS_SIZE)
#endif

#ifdef WITH_GPU_VERSION
!    real*8                  :: ttt0, ttt1, ttt2, t2_compute_kernel, t0_compute_kernel,t1_compute_kernel, &
!                               t0_mpi_time, t1_mpi_time,t2_mpi_time
!    real*8                  :: t0_cpu_code,t1_cpu_code,t2_cpu_code,t0_block_time,t1_block_time,t2_block_time,t0_cuda_memcpy

!    real*8                  :: t0_inner_do_time, t1_inner_do_time , t2_inner_do_time,t0_outer_do_time ,t1_outer_do_time , &
!                               t2_outer_do_time ,t0_result_time ,t1_result_time, t2_result_time,t0_mpi_recv_time,         &
!                               t1_mpi_recv_time,t2_mpi_recv_time
   integer                  :: top, chunk, this_chunk

!   real*8                   :: t1_mpi_wait_time,t0_mpi_wait_time,t2_mpi_wait_time,t1_memcpy_time,t0_memcpy_time,t2_memcpy_time, &
!                               t1_mpi_irecv_time,t0_mpi_irecv_time,t2_mpi_irecv_time,t0_mpi_outer_wait_time,t1_mpi_outer_wait_time,&
!                               t2_mpi_outer_wait_time, time0
!   real*4                   :: time1
#endif

    ! MPI send/recv tags, arbitrary

    integer, parameter      :: bottom_recv_tag = 111
    integer, parameter      :: top_recv_tag    = 222
    integer, parameter      :: result_recv_tag = 333

#ifdef WITH_OPENMP
    integer                 :: max_threads, my_thread
    integer                 :: omp_get_max_threads
#endif

    ! Just for measuring the kernel performance
    real*8                  :: kernel_time
    integer*8               :: kernel_flops

    logical, intent(in)     :: wantDebug
    logical                 :: success
    integer                 :: istat
    character(200)          :: errorMessage

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("trans_ev_tridi_to_band_complex")
#endif

#ifdef WITH_GPU_VERSION
    n_times =0
    !    n_times_1 =0
    unpack_idx = 0
    row_group_size = 0
!    time0=0
!    t0_compute_kernel=0
#endif
    kernel_time = 1.d-100
    kernel_flops = 0

#ifdef WITH_OPENMP
    max_threads = 1
    max_threads = omp_get_max_threads()
#endif

    call MPI_Comm_rank(mpi_comm_rows, my_prow, mpierr)
    call MPI_Comm_size(mpi_comm_rows, np_rows, mpierr)
    call MPI_Comm_rank(mpi_comm_cols, my_pcol, mpierr)
    call MPI_Comm_size(mpi_comm_cols, np_cols, mpierr)

    success = .true.

    if (mod(nbw,nblk)/=0) then
      if (my_prow==0 .and. my_pcol==0) then
        if (wantDebug) then
          write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_complex: ERROR: nbw=',nbw,', nblk=',nblk
          write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_complex: band backtransform works only for nbw==n*nblk'
        endif

        success = .false.
        return
      endif
    endif

    nfact = nbw / nblk


    ! local number of eigenvectors
    l_nev = local_index(nev, my_pcol, np_cols, nblk, -1)

    if (l_nev==0) then
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif
      thread_width = 0
#endif
      stripe_width = 0
      stripe_count = 0
      last_stripe_width = 0
    else
      ! Suggested stripe width is 48 - should this be reduced for the complex case ???
#ifdef WITH_OPENMP
      thread_width = (l_nev-1)/max_threads + 1 ! number of eigenvectors per OMP thread
#endif

#ifdef WITH_GPU_VERSION
      stripe_width = 256
#else
      stripe_width = 48 ! Must be a multiple of 4
#endif

#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif
      stripe_count = (thread_width-1)/stripe_width + 1
#else /* WITH_OPENMP */

      stripe_count = (l_nev-1)/stripe_width + 1
#endif /* WITH_OPENMP */

      ! Adapt stripe width so that last one doesn't get too small
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif

      stripe_width = (thread_width-1)/stripe_count + 1
#else /* WITH_OPENMP */

#ifndef WITH_GPU_VERSION
      stripe_width = (l_nev-1)/stripe_count + 1
      stripe_width = ((stripe_width+3)/4)*4 ! Must be a multiple of 4 !!!
#endif

#endif /* WITH_OPENMP */

#ifndef WITH_OPENMP
      last_stripe_width = l_nev - (stripe_count-1)*stripe_width
#endif /* WITH_OPENMP */

    endif

    ! Determine the matrix distribution at the beginning

    allocate(limits(0:np_rows), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error when allocating limits "//errorMessage
      stop
    endif

    call determine_workload(na, nbw, np_rows, limits)

    max_blk_size = maxval(limits(1:np_rows) - limits(0:np_rows-1))

    a_dim2 = max_blk_size + nbw
#ifndef WITH_GPU_VERSION
!DEC$ ATTRIBUTES ALIGN: 64:: a
#endif

#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif


#ifndef WITH_GPU_VERSION
    allocate(a(stripe_width,a_dim2,stripe_count,max_threads), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating a "//errorMessage
      stop
    endif

    ! a(:,:,:,:) should be set to 0 in a parallel region, not here!
#endif

#else /* OpenMP */


#ifndef WITH_GPU_VERSION
    allocate(a(stripe_width,a_dim2,stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating a "//errorMessage
      stop
    endif

    a(:,:,:) = 0
#endif

#endif /* WITH_OPENMP */

    allocate(row(l_nev), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating row "//errorMessage
      stop
    endif

    row(:) = 0

#ifdef WITH_GPU_VERSION
    num =  (stripe_width*a_dim2*stripe_count)*16_8
    if (na_rows * na_cols .lt. stripe_width*a_dim2*stripe_count) then
      print *,"trans_ev_tridi_to_band_complex a_dev ",na_rows * na_cols, stripe_width*a_dim2*stripe_count
!      stop
    endif

    istat = cuda_malloc(a_dev, stripe_width*a_dim2*stripe_count*16_8)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error in cudaMalloc "
      stop
    endif

    if (num .gt. na_rows * na_cols) then
      print *,"trans_ev_tridi_to_band_complex a_dev 1",num, na_rows * na_cols
!      stop
    endif
    istat = cuda_memset(a_dev , 0, num)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error in cudaMemset "
      stop
    endif

    num =  (l_nev)*16_8
    istat = cuda_malloc( row_dev,l_nev*16_8)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error in cudaMalloc "
      stop
    endif

    istat = cuda_memset(row_dev , 0, num)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error in cudaMemset "
      stop
    endif

     ! "row_group" and "row_group_dev" are needed for GPU optimizations
    allocate(row_group(l_nev, nblk), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating row_group "//errorMessage
      stop
    endif

    row_group(:, :) = 0

    num =  (l_nev*nblk)*16_8
    istat = cuda_malloc(row_group_dev, l_nev*nblk*16_8)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error in cudaMalloc "
      stop
    endif

    istat = cuda_memset(row_group_dev , 0, num)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error in cudaMemset "
      stop
    endif

#endif

    ! Copy q from a block cyclic distribution into a distribution with contiguous rows,
    ! and transpose the matrix using stripes of given stripe_width for cache blocking.

    ! The peculiar way it is done below is due to the fact that the last row should be
    ! ready first since it is the first one to start below
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif

    ! Please note about the OMP usage below:
    ! This is not for speed, but because we want the matrix a in the memory and
    ! in the cache of the correct thread (if possible)
#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
    do my_thread = 1, max_threads
      a(:,:,:,my_thread) = 0 ! if possible, do first touch allocation!
    enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("OpenMP parallel")
#endif

#endif /* WITH_OPENMP */

    do ip = np_rows-1, 0, -1
      if (my_prow == ip) then
        ! Receive my rows which have not yet been received
        src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
        do i=limits(ip)+1,limits(ip+1)
          src = mod((i-1)/nblk, np_rows)
          if (src < my_prow) then
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif

            call MPI_Recv(row, l_nev, MPI_COMPLEX16, src, 0, mpi_comm_rows, mpi_status, mpierr)

#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
            call unpack_and_prepare_row_group_complex(i - limits(ip), .false.)
            call MPI_Recv(row_group(:, row_group_size), l_nev,MPI_COMPLEX16, src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)
#else
            call MPI_Recv(row, l_nev, MPI_COMPLEX16, src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)
#endif

#endif /* WITH_OPENMP */


#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif

#ifdef HAVE_DETAILED_TIMINGS
            call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
            do my_thread = 1, max_threads
              call unpack_row_complex(row,i-limits(ip),my_thread)
            enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
            call timer%stop("OpenMP parallel")
#endif

#else /* WITH_OPENMP */

#ifndef WITH_GPU_VERSION
            call unpack_row_complex(row,i-limits(ip))
#endif

#endif /* WITH_OPENMP */

          elseif (src==my_prow) then
            src_offset = src_offset+1
#ifdef WITH_GPU_VERSION
            call unpack_and_prepare_row_group_complex(i - limits(ip),.false.)
            row_group(:, row_group_size) = q(src_offset, 1:l_nev)
#else
            row(:) = q(src_offset, 1:l_nev)
#endif

#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
            call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
            do my_thread = 1, max_threads
              call unpack_row_complex(row,i-limits(ip),my_thread)
            enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
            call timer%stop("OpenMP parallel")
#endif

#else /* WITH_OPENMP */

#ifndef WITH_GPU_VERSION
            call unpack_row_complex(row,i-limits(ip))
#endif

#endif /* WITH_OPENMP */

          endif
        enddo
        ! Send all rows which have not yet been send
        src_offset = 0
        do dst = 0, ip-1
          do i=limits(dst)+1,limits(dst+1)
            if(mod((i-1)/nblk, np_rows) == my_prow) then
                src_offset = src_offset+1
                row(:) = q(src_offset, 1:l_nev)
                call MPI_Send(row, l_nev, MPI_COMPLEX16, dst, 0, mpi_comm_rows, mpierr)
            endif
          enddo
        enddo
      else if(my_prow < ip) then
        ! Send all rows going to PE ip
        src_offset = local_index(limits(ip), my_prow, np_rows, nblk, -1)
        do i=limits(ip)+1,limits(ip+1)
          src = mod((i-1)/nblk, np_rows)
          if (src == my_prow) then
            src_offset = src_offset+1
            row(:) = q(src_offset, 1:l_nev)
            call MPI_Send(row, l_nev, MPI_COMPLEX16, ip, 0, mpi_comm_rows, mpierr)
          endif
        enddo
        ! Receive all rows from PE ip
        do i=limits(my_prow)+1,limits(my_prow+1)
          src = mod((i-1)/nblk, np_rows)
          if (src == ip) then
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif

            call MPI_Recv(row, l_nev, MPI_COMPLEX16, src, 0, mpi_comm_rows, mpi_status, mpierr)
#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
            call unpack_and_prepare_row_group_complex(i - limits(my_prow), .false.)
            call MPI_Recv(row_group(:, row_group_size), l_nev,MPI_COMPLEX16, src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)
#else
            call MPI_Recv(row, l_nev, MPI_COMPLEX16, src, 0, mpi_comm_rows, MPI_STATUS_IGNORE, mpierr)
#endif

#endif /* WITH_OPENMP */


#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif

#ifdef HAVE_DETAILED_TIMINGS
            call timer%start("OpenMP parallel")
#endif
!$omp parallel do private(my_thread), schedule(static, 1)
            do my_thread = 1, max_threads
              call unpack_row_complex(row,i-limits(my_prow),my_thread)
            enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
            call timer%stop("OpenMP parallel")
#endif

#else /* WITH_OPENMP */

#ifndef WITH_GPU_VERSION
            call unpack_row_complex(row,i-limits(my_prow))
#endif

#endif /* WITH_OPENMP */

          endif
        enddo
      endif
    enddo

#ifdef WITH_GPU_VERSION
    call unpack_and_prepare_row_group_complex(-1, .true.)
    istat = cuda_devicesynchronize()
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error in cudaDeviceSynchronize"
      stop
    endif
#endif

    ! Set up result buffer queue

    num_result_blocks = ((na-1)/nblk + np_rows - my_prow) / np_rows

    num_result_buffers = 4*nfact
    allocate(result_buffer(l_nev,nblk,num_result_buffers), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating result_buffer "//errorMessage
      stop
    endif

    allocate(result_send_request(num_result_buffers), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating result_send_request "//errorMessage
      stop
    endif

    allocate(result_recv_request(num_result_buffers), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating result_recv_request "//errorMessage
      stop
    endif

    result_send_request(:) = MPI_REQUEST_NULL
    result_recv_request(:) = MPI_REQUEST_NULL

    ! Queue up buffers

    if (my_prow > 0 .and. l_nev>0) then ! note: row 0 always sends
      do j = 1, min(num_result_buffers, num_result_blocks)
        call MPI_Irecv(result_buffer(1,1,j), l_nev*nblk, MPI_COMPLEX16, 0, result_recv_tag, &
                           mpi_comm_rows, result_recv_request(j), mpierr)
      enddo
    endif

    num_bufs_recvd = 0 ! No buffers received yet

    ! Initialize top/bottom requests

    allocate(top_send_request(stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating top_send_request "//errorMessage
      stop
    endif

    allocate(top_recv_request(stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating top_recv_request "//errorMessage
      stop
    endif

    allocate(bottom_send_request(stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating bottom_send_request "//errorMessage
      stop
    endif

    allocate(bottom_recv_request(stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating bottom_recv_request "//errorMessage
      stop
    endif

    top_send_request(:) = MPI_REQUEST_NULL
    top_recv_request(:) = MPI_REQUEST_NULL
    bottom_send_request(:) = MPI_REQUEST_NULL
    bottom_recv_request(:) = MPI_REQUEST_NULL

#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif
    allocate(top_border_send_buffer(stripe_width*nbw*max_threads, stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating top_border_send_buffer "//errorMessage
      stop
    endif

    allocate(top_border_recv_buffer(stripe_width*nbw*max_threads, stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating top_border_recv_buffer "//errorMessage
      stop
    endif

    allocate(bottom_border_send_buffer(stripe_width*nbw*max_threads, stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating bottom_border_send_buffer "//errorMessage
      stop
    endif

    allocate(bottom_border_recv_buffer(stripe_width*nbw*max_threads, stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating bottom_border_recv_buffer "//errorMessage
      stop
    endif

    top_border_send_buffer(:,:) = 0
    top_border_recv_buffer(:,:) = 0
    bottom_border_send_buffer(:,:) = 0
    bottom_border_recv_buffer(:,:) = 0
#else /* OpenMP */
    allocate(top_border_send_buffer(stripe_width, nbw, stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating top_border_send_buffer "//errorMessage
      stop
    endif

    allocate(top_border_recv_buffer(stripe_width, nbw, stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating top_border_recv_buffer "//errorMessage
      stop
    endif

    allocate(bottom_border_send_buffer(stripe_width, nbw, stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating bottom_border_send_buffer "//errorMessage
      stop
    endif

    allocate(bottom_border_recv_buffer(stripe_width, nbw, stripe_count), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating bottom_border_recv_buffer "//errorMessage
      stop
    endif

    top_border_send_buffer(:,:,:) = 0
    top_border_recv_buffer(:,:,:) = 0
    bottom_border_send_buffer(:,:,:) = 0
    bottom_border_recv_buffer(:,:,:) = 0
#endif

    ! Initialize broadcast buffer

    allocate(bcast_buffer(nbw, max_blk_size), stat=istat, errmsg=errorMessage)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error allocating bcast_buffer "//errorMessage
      stop
    endif
    bcast_buffer = 0

#ifdef WITH_GPU_VERSION
    num =  ( nbw * max_blk_size) * 16_8
    istat = cuda_malloc(bcast_buffer_dev, nbw * max_blk_size * 16_8)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error in cudaMalloc"
      stop
    endif

    istat = cuda_memset( bcast_buffer_dev, 0, num)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error in cudaMemset"
      stop
    endif

    num =  ((max_blk_size-1))*16_8
    istat = cuda_malloc( hh_dot_dev, (max_blk_size -1) * 16_8)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error in cudaMalloc"
      stop
    endif

    istat = cuda_memset( hh_dot_dev, 0, num)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error in cudaMemset"
      stop
    endif

    num =  (max_blk_size)*16_8
    istat = cuda_malloc( hh_tau_dev, max_blk_size * 16_8)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error in cudaMalloc"
      stop
    endif

    istat = cuda_memset( hh_tau_dev, 0, num)
    if (istat .ne. 0) then
      print *,"trans_ev_tridi_to_band_complex: error in cudaMemset"
      stop
    endif
#endif

    current_tv_off = 0 ! Offset of next row to be broadcast


    ! ------------------- start of work loop -------------------

    a_off = 0 ! offset in A (to avoid unnecessary shifts)

    top_msg_length = 0
    bottom_msg_length = 0

#ifdef WITH_GPU_VERSION
!!    istat = cuda_ProfilerStart()
!!    istat = cudaFuncSetCacheConfig  ( launch_compute_hh_trafo_c_kernel_complex,  cudaFuncCachePreferShared)
!!    t0_compute_kernel = 0
!    t0_mpi_time = 0
!    t0_cuda_memcpy =0
!    t0_cpu_code =0
!    t0_outer_do_time =0
!    t0_inner_do_time =0
!    t1_outer_do_time =MPI_Wtime()
!    t0_block_time =0
!    t0_mpi_wait_time = 0
!    t0_memcpy_time = 0
!    t0_mpi_outer_wait_time=0
#endif

    do sweep = 0, (na-1)/nbw

#ifdef WITH_GPU_VERSION
!      t1_cpu_code =MPI_Wtime()
#endif

      current_n = na - sweep*nbw
      call determine_workload(current_n, nbw, np_rows, limits)
      current_n_start = limits(my_prow)
      current_n_end   = limits(my_prow+1)
      current_local_n = current_n_end - current_n_start

      next_n = max(current_n - nbw, 0)
      call determine_workload(next_n, nbw, np_rows, limits)
      next_n_start = limits(my_prow)
      next_n_end   = limits(my_prow+1)
      next_local_n = next_n_end - next_n_start

      if (next_n_end < next_n) then
        bottom_msg_length = current_n_end - next_n_end
      else
        bottom_msg_length = 0
      endif

      if (next_local_n > 0) then
        next_top_msg_length = current_n_start - next_n_start
      else
        next_top_msg_length = 0
      endif

#ifdef WITH_GPU_VERSION
!        t2_cpu_code =MPI_Wtime()
!        t0_cpu_code =  t0_cpu_code + (t2_cpu_code - t1_cpu_code)
#endif

      if (sweep==0 .and. current_n_end < current_n .and. l_nev > 0) then
        do i = 1, stripe_count
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif

          csw = min(stripe_width, thread_width-(i-1)*stripe_width) ! "current_stripe_width"
          b_len = csw*nbw*max_threads
          call MPI_Irecv(bottom_border_recv_buffer(1,i), b_len, MPI_COMPLEX16, my_prow+1, bottom_recv_tag, &
                     mpi_comm_rows, bottom_recv_request(i), mpierr)
#else /* WITH_OPENMP */
          call MPI_Irecv(bottom_border_recv_buffer(1,1,i), nbw*stripe_width, MPI_COMPLEX16, my_prow+1, bottom_recv_tag, &
                         mpi_comm_rows, bottom_recv_request(i), mpierr)
#endif /* WITH_OPENMP */

        enddo
      endif

#ifdef WITH_GPU_VERSION
!        t1_block_time = MPI_Wtime()
#endif
      if (current_local_n > 1) then
        if (my_pcol == mod(sweep,np_cols)) then
          bcast_buffer(:,1:current_local_n) = hh_trans_complex(:,current_tv_off+1:current_tv_off+current_local_n)
          current_tv_off = current_tv_off + current_local_n
        endif
        call mpi_bcast(bcast_buffer, nbw*current_local_n, MPI_COMPLEX16, mod(sweep,np_cols), mpi_comm_cols, mpierr)
#ifdef WITH_GPU_VERSION
        istat =  cuda_memcpy(bcast_buffer_dev, loc(bcast_buffer(1,1)), nbw * current_local_n * 16_8 , 1)
        call extract_hh_tau_complex(nbw, current_local_n, .false.)
        call compute_hh_dot_products_complex(nbw, current_local_n)
#endif
      else
        ! for current_local_n == 1 the one and only HH vector is 0 and not stored in hh_trans_complex
        bcast_buffer(:,1) = 0
#ifdef WITH_GPU_VERSION
        istat = cuda_memset(bcast_buffer_dev, 0, nbw * 16_8)
        if (istat .ne. 0) then
          print *,"trans_ev_tridi_to_band_complex: error in cudaMemset"
          stop
        endif

        call extract_hh_tau_complex(nbw, 1, .true.)

!NOTE(ca): I commented out the following line
!        istat =  cuda_memcpy(loc(bcast_buffer(1,1)),bcast_buffer_dev,nbw*current_local_n * 16_8 , 2)
!        if (istat .ne. 0) then
!          print *,"trans_ev_tridi_to_band_complex: error in cudaMalloc"
!          stop
!        endif

#endif
      endif

#ifdef WITH_GPU_VERSION
!      t2_block_time =MPI_Wtime()
!      t0_block_time = t0_block_time + ( t2_block_time - t1_block_time)
#endif

      if (l_nev == 0) cycle

        if (current_local_n > 0) then
#ifdef WITH_GPU_VERSION
!          t1_inner_do_time =MPI_Wtime()
#endif

          do i = 1, stripe_count

#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif
            ! Get real stripe width for strip i;
            ! The last OpenMP tasks may have an even smaller stripe with,
            ! but we don't care about this, i.e. we send/recv a bit too much in this case.
            ! csw: current_stripe_width

            csw = min(stripe_width, thread_width-(i-1)*stripe_width)
#endif /* WITH_OPENMP */

            !wait_b
            if (current_n_end < current_n) then
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif

              call MPI_Wait(bottom_recv_request(i), mpi_status, mpierr)
#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
!              t1_mpi_wait_time =MPI_Wtime()
#endif
              call MPI_Wait(bottom_recv_request(i), MPI_STATUS_IGNORE, mpierr)

#ifdef WITH_GPU_VERSION
!              t2_mpi_wait_time =MPI_Wtime()
!              t0_mpi_wait_time = t0_mpi_wait_time + ( t2_mpi_wait_time - t1_mpi_wait_time)
!
!              t1_memcpy_time =MPI_Wtime()
#endif

#endif /* WITH_OPENMP */

#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif

#ifdef HAVE_DETAILED_TIMINGS
              call timer%start("OpenMP parallel")
#endif
!$omp parallel do private(my_thread, n_off, b_len, b_off), schedule(static, 1)
              do my_thread = 1, max_threads
                n_off = current_local_n+a_off
                b_len = csw*nbw
                b_off = (my_thread-1)*b_len
                a(1:csw,n_off+1:n_off+nbw,i,my_thread) = &
                   reshape(bottom_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, nbw /))
              enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
              call timer%stop("OpenMP parallel")
#endif

#else /* WITH_OPENMP */

              n_off = current_local_n+a_off
#ifdef WITH_GPU_VERSION
!              t1_memcpy_time =MPI_Wtime()
              dev_offset = (0 + (n_off * stripe_width) + ( (i-1) * stripe_width *a_dim2 )) * 16_8
              istat =  cuda_memcpy( a_dev + dev_offset ,loc(bottom_border_recv_buffer(1,1,i)) ,stripe_width*nbw*16_8 ,1)
              if (istat .ne. 0) then
                print *,"trans_ev_tridi_to_band_complex: error in cudaMalloc"
                stop
              endif

!              t2_memcpy_time =MPI_Wtime()
!              t0_memcpy_time = t0_memcpy_time + ( t2_memcpy_time - t1_memcpy_time)
#else
              a(:,n_off+1:n_off+nbw,i) = bottom_border_recv_buffer(:,1:nbw,i)
#endif

#endif /* WITH_OPENMP */

              if (next_n_end < next_n) then
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"not yet implemented"
      stop
#endif


                call MPI_Irecv(bottom_border_recv_buffer(1,i), csw*nbw*max_threads, &
                                   MPI_COMPLEX16, my_prow+1, bottom_recv_tag, &
                                   mpi_comm_rows, bottom_recv_request(i), mpierr)
#else /* WITH_OPENMP */

                call MPI_Irecv(bottom_border_recv_buffer(1,1,i), nbw*stripe_width, MPI_COMPLEX16, my_prow+1, bottom_recv_tag, &
                                   mpi_comm_rows, bottom_recv_request(i), mpierr)
#endif /* WITH_OPENMP */

              endif
            endif

            if (current_local_n <= bottom_msg_length + top_msg_length) then

              !wait_t
              if (top_msg_length>0) then
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif

                call MPI_Wait(top_recv_request(i), mpi_status, mpierr)
#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
!                t1_mpi_wait_time =MPI_Wtime()
#endif
                call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)

#ifdef WITH_GPU_VERSION
!                t2_mpi_wait_time =MPI_Wtime()
!                t0_mpi_wait_time = t0_mpi_wait_time + ( t2_mpi_wait_time -t1_mpi_wait_time)
!                t1_memcpy_time =MPI_Wtime()
!
                dev_offset = (0 + (a_off * stripe_width) + ( (i-1) * stripe_width *a_dim2 )) *16_8
!                host_offset= (0 + (0 * stripe_width) + ( (i-1) * stripe_width * nbw ))* 16
                istat =  cuda_memcpy( a_dev+dev_offset ,loc(top_border_recv_buffer(1,1,i)),stripe_width*top_msg_length*16_8 ,1)
                if (istat .ne. 0) then
                  print *,"trans_ev_tridi_to_band_complex: error in cudaMemcpy"
                  stop
                endif

!                t2_memcpy_time =MPI_Wtime()
!                t0_memcpy_time = t0_memcpy_time + ( t2_memcpy_time - t1_memcpy_time)
#else
                a(:,a_off+1:a_off+top_msg_length,i) = top_border_recv_buffer(:,1:top_msg_length,i)

#endif

#endif /* WITH_OPENMP */
              endif

              !compute
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif

#ifdef HAVE_DETAILED_TIMINGS
              call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, n_off, b_len, b_off), schedule(static, 1)
              do my_thread = 1, max_threads
                if (top_msg_length>0) then
                  b_len = csw*top_msg_length
                  b_off = (my_thread-1)*b_len
                  a(1:csw,a_off+1:a_off+top_msg_length,i,my_thread) = &
                           reshape(top_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, top_msg_length /))
                endif
                call compute_hh_trafo_complex(0, current_local_n, i, my_thread, &
                                       THIS_COMPLEX_ELPA_KERNEL)
              enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
              call timer%stop("OpenMP parallel")
#endif

#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
              call compute_hh_trafo_complex_gpu(0, current_local_n, i, a_off, dev_offset, dev_offset_1, dev_offset_2)
!              call compute_hh_trafo_complex_gpu(0, current_local_n, i)
#else
              call compute_hh_trafo_complex(0, current_local_n, i, &
                                      THIS_COMPLEX_ELPA_KERNEL)
#endif

#endif /* WITH_OPENMP */

              !send_b
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif

              call MPI_Wait(bottom_send_request(i), mpi_status, mpierr)
#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
!              t1_mpi_wait_time =MPI_Wtime()
#endif
              call MPI_Wait(bottom_send_request(i), MPI_STATUS_IGNORE, mpierr)

#ifdef WITH_GPU_VERSION
!              t2_mpi_wait_time =MPI_Wtime()
!              t0_mpi_wait_time = t0_mpi_wait_time + ( t2_mpi_wait_time-t1_mpi_wait_time)
#endif

#endif /* WITH_OPENMP */

              if (bottom_msg_length>0) then
                n_off = current_local_n+nbw-bottom_msg_length+a_off
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif

                b_len = csw*bottom_msg_length*max_threads
                bottom_border_send_buffer(1:b_len,i) = &
                        reshape(a(1:csw,n_off+1:n_off+bottom_msg_length,i,:), (/ b_len /))
                call MPI_Isend(bottom_border_send_buffer(1,i), b_len, MPI_COMPLEX16, my_prow+1, &
                                   top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
!                t1_memcpy_time =MPI_Wtime()
                dev_offset = (0 + (n_off * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) * 16_8
                istat =  cuda_memcpy( loc(bottom_border_send_buffer(1,1,i)), a_dev + dev_offset, &
                                     stripe_width * bottom_msg_length * 16_8 , 2)
                if (istat .ne. 0) then
                  print *,"trans_ev_tridi_to_band_complex: error in cudaMemcpy"
                  stop
                endif

!                t2_memcpy_time =MPI_Wtime()
!                t0_memcpy_time = t0_memcpy_time + ( t2_memcpy_time -t1_memcpy_time)
#else
               bottom_border_send_buffer(:,1:bottom_msg_length,i) = a(:,n_off+1:n_off+bottom_msg_length,i)
#endif
                call MPI_Isend(bottom_border_send_buffer(1,1,i), bottom_msg_length*stripe_width, MPI_COMPLEX16, my_prow+1, &
                              top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
#endif /* WITH_OPENMP */
              endif

            else

              !compute
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif

#ifdef HAVE_DETAILED_TIMINGS
              call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, b_len, b_off), schedule(static, 1)
              do my_thread = 1, max_threads
                call compute_hh_trafo_complex(current_local_n - bottom_msg_length, bottom_msg_length, i, my_thread, &
                                      THIS_COMPLEX_ELPA_KERNEL)
              enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
              call timer%stop("OpenMP parallel")
#endif

#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
              call compute_hh_trafo_complex_gpu(current_local_n -bottom_msg_length, bottom_msg_length, i, a_off, &
                                                dev_offset, dev_offset_1, dev_offset_2)
!              call compute_hh_trafo_complex_gpu(current_local_n -bottom_msg_length, bottom_msg_length, i)
#else
              call compute_hh_trafo_complex(current_local_n - bottom_msg_length, bottom_msg_length, i, &
                                      THIS_COMPLEX_ELPA_KERNEL)

#endif

#endif /* WITH_OPENMP */

              !send_b
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif
              call MPI_Wait(bottom_send_request(i), mpi_status, mpierr)
#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
!              t1_mpi_wait_time =MPI_Wtime()
#endif

              call MPI_Wait(bottom_send_request(i), MPI_STATUS_IGNORE, mpierr)

#ifdef WITH_GPU_VERSION
!              t2_mpi_wait_time =MPI_Wtime()
!              t0_mpi_wait_time = t0_mpi_wait_time + ( t2_mpi_wait_time-t1_mpi_wait_time)
#endif

#endif /* WITH_OPENMP */
              if (bottom_msg_length > 0) then
                n_off = current_local_n+nbw-bottom_msg_length+a_off
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif

                b_len = csw*bottom_msg_length*max_threads
                bottom_border_send_buffer(1:b_len,i) = &
                      reshape(a(1:csw,n_off+1:n_off+bottom_msg_length,i,:), (/ b_len /))
                call MPI_Isend(bottom_border_send_buffer(1,i), b_len, MPI_COMPLEX16, my_prow+1, &
                                   top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
!                t1_memcpy_time =MPI_Wtime()
                dev_offset = (0 + (n_off * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) * 16_8
                istat =  cuda_memcpy( loc(bottom_border_send_buffer(1,1,i)), a_dev + dev_offset, &
                                     stripe_width * bottom_msg_length * 16_8 , 2)
                if (istat .ne. 0) then
                  print *,"trans_ev_tridi_to_band_complex: error in cudaMemcpy"
                  stop
                endif

!                t2_memcpy_time =MPI_Wtime()
!                t0_memcpy_time = t0_memcpy_time + ( t2_memcpy_time -t1_memcpy_time)
#else
                bottom_border_send_buffer(:,1:bottom_msg_length,i) = a(:,n_off+1:n_off+bottom_msg_length,i)
#endif
                call MPI_Isend(bottom_border_send_buffer(1,1,i), bottom_msg_length*stripe_width, MPI_COMPLEX16, my_prow+1, &
                              top_recv_tag, mpi_comm_rows, bottom_send_request(i), mpierr)
#endif /* WITH_OPENMP */
              endif

              !compute
#ifdef WITH_OPENMP
#ifdef WITH_GPU_VERSION
     print *,"trans_ev_tridi_to_band_complex: not yet implemented"
      stop
#endif

#ifdef HAVE_DETAILED_TIMINGS
              call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread), schedule(static, 1)
              do my_thread = 1, max_threads
                call compute_hh_trafo_complex(top_msg_length, current_local_n-top_msg_length-bottom_msg_length, i, my_thread, &
                                      THIS_COMPLEX_ELPA_KERNEL)
              enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
              call timer%stop("OpenMP parallel")
#endif

#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
!              call compute_hh_trafo_complex_gpu(top_msg_length,current_local_n-top_msg_length-bottom_msg_length, i)

              call compute_hh_trafo_complex_gpu(top_msg_length,current_local_n-top_msg_length-bottom_msg_length, i, a_off, &
                                                dev_offset, dev_offset_1, dev_offset_2)
#else
              call compute_hh_trafo_complex(top_msg_length, current_local_n-top_msg_length-bottom_msg_length, i, &
                                      THIS_COMPLEX_ELPA_KERNEL)
#endif

#endif /* WITH_OPENMP */

              !wait_t
              if (top_msg_length>0) then
#ifdef WITH_OPENMP
                call MPI_Wait(top_recv_request(i), mpi_status, mpierr)
#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
!                t1_mpi_wait_time =MPI_Wtime()
#endif
                call MPI_Wait(top_recv_request(i), MPI_STATUS_IGNORE, mpierr)

#ifdef WITH_GPU_VERSION
!                t2_mpi_wait_time =MPI_Wtime()
!                t0_mpi_wait_time = t0_mpi_wait_time +(t2_mpi_wait_time-t1_mpi_wait_time)
!
!                t1_memcpy_time =MPI_Wtime()
                dev_offset = (0 + (a_off * stripe_width) + ( (i-1) * stripe_width *a_dim2 )) *16_8
                istat =  cuda_memcpy( a_dev + dev_offset , loc(top_border_recv_buffer(:,1,i)), &
                                     stripe_width * top_msg_length *16_8 ,1)
                if (istat .ne. 0) then
                  print *,"trans_ev_tridi_to_band_complex: error in cudaMemcpy"
                  stop
                endif

!
!                t2_memcpy_time =MPI_Wtime()
!                t0_memcpy_time = t0_memcpy_time + ( t2_memcpy_time-t1_memcpy_time)
#else
                a(:,a_off+1:a_off+top_msg_length,i) = top_border_recv_buffer(:,1:top_msg_length,i)
#endif

#endif /* WITH_OPENMP */

              endif

              !compute
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
              call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, b_len, b_off), schedule(static, 1)
              do my_thread = 1, max_threads
                if (top_msg_length>0) then
                  b_len = csw*top_msg_length
                  b_off = (my_thread-1)*b_len
                  a(1:csw,a_off+1:a_off+top_msg_length,i,my_thread) = &
                          reshape(top_border_recv_buffer(b_off+1:b_off+b_len,i), (/ csw, top_msg_length /))
                endif
                call compute_hh_trafo_complex(0, top_msg_length, i, my_thread, &
                                      THIS_COMPLEX_ELPA_KERNEL)
              enddo
!$omp end parallel do
#ifdef HAVE_DETAILED_TIMINGS
              call timer%stop("OpenMP parallel")
#endif

#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
              call compute_hh_trafo_complex_gpu(0, top_msg_length, i, a_off, dev_offset, dev_offset_1, dev_offset_2)
!              call compute_hh_trafo_complex_gpu(0, top_msg_length, i)
#else
              call compute_hh_trafo_complex(0, top_msg_length, i, &
                                      THIS_COMPLEX_ELPA_KERNEL)
#endif

#endif /* WITH_OPENMP */

            endif

            if (next_top_msg_length > 0) then
              !request top_border data
#ifdef WITH_OPENMP
              b_len = csw*next_top_msg_length*max_threads
              call MPI_Irecv(top_border_recv_buffer(1,i), b_len, MPI_COMPLEX16, my_prow-1, &
                               top_recv_tag, mpi_comm_rows, top_recv_request(i), mpierr)
#else /* WITH_OPENMP */
              call MPI_Irecv(top_border_recv_buffer(1,1,i), next_top_msg_length*stripe_width, MPI_COMPLEX16, my_prow-1, &
                               top_recv_tag, mpi_comm_rows, top_recv_request(i), mpierr)
#endif /* WITH_OPENMP */
            endif

            !send_t
            if (my_prow > 0) then
#ifdef WITH_OPENMP
              call MPI_Wait(top_send_request(i), mpi_status, mpierr)
#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
!              t1_mpi_wait_time =MPI_Wtime()
#endif
              call MPI_Wait(top_send_request(i), MPI_STATUS_IGNORE, mpierr)

#ifdef WITH_GPU_VERSION
!              t2_mpi_wait_time =MPI_Wtime()
!              t0_mpi_wait_time = t0_mpi_wait_time+(t2_mpi_wait_time-t1_mpi_wait_time)
#endif

#endif /* WITH_OPENMP */


#ifdef WITH_OPENMP
              b_len = csw*nbw*max_threads
              top_border_send_buffer(1:b_len,i) = reshape(a(1:csw,a_off+1:a_off+nbw,i,:), (/ b_len /))
              call MPI_Isend(top_border_send_buffer(1,i), b_len, MPI_COMPLEX16, &
                               my_prow-1, bottom_recv_tag, &
                               mpi_comm_rows, top_send_request(i), mpierr)
#else /* WITH_OPENMP */

#ifdef WITH_GPU_VERSION
!              t1_memcpy_time =MPI_Wtime()
              dev_offset = (0 + (a_off * stripe_width) + ( (i-1) * stripe_width *a_dim2 )) * 16_8
              istat =  cuda_memcpy( loc(top_border_send_buffer(:,1,i)), a_dev + dev_offset, stripe_width*nbw*16_8 ,2)
              if (istat .ne. 0) then
                print *,"trans_ev_tridi_to_band_complex: error in cudaMemcpy"
                stop
              endif

!              t2_memcpy_time =MPI_Wtime()
!              t0_memcpy_time = t0_memcpy_time + (t2_memcpy_time-t1_memcpy_time)
!
#else
              top_border_send_buffer(:,1:nbw,i) = a(:,a_off+1:a_off+nbw,i)
#endif
              call MPI_Isend(top_border_send_buffer(1,1,i), nbw*stripe_width, MPI_COMPLEX16, my_prow-1, bottom_recv_tag, &
                               mpi_comm_rows, top_send_request(i), mpierr)

#endif /* WITH_OPENMP */
            endif

            ! Care that there are not too many outstanding top_recv_request's
#ifdef WITH_GPU_VERSION
!            t1_mpi_wait_time =MPI_Wtime()
#endif
            if (stripe_count > 1) then
              if (i>1) then
#ifdef WITH_OPENMP
                call MPI_Wait(top_recv_request(i-1), mpi_status, mpierr)
#else
                call MPI_Wait(top_recv_request(i-1), MPI_STATUS_IGNORE, mpierr)
#endif
              else
#ifdef WITH_OPENMP
                call MPI_Wait(top_recv_request(stripe_count), mpi_status, mpierr)
#else
                call MPI_Wait(top_recv_request(stripe_count), MPI_STATUS_IGNORE, mpierr)
#endif
              endif
            endif
#ifdef WITH_GPU_VERSION
!            t2_mpi_wait_time =MPI_Wtime()
!            t0_mpi_wait_time = t0_mpi_wait_time+(t2_mpi_wait_time-t1_mpi_wait_time)
#endif
          enddo

#ifdef WITH_GPU_VERSION
!          t2_inner_do_time =MPI_Wtime()
!          t0_inner_do_time = t0_inner_do_time + ( t2_inner_do_time - t1_inner_do_time)
#endif
          top_msg_length = next_top_msg_length

        else
          ! wait for last top_send_request
#ifdef WITH_GPU_VERSION
!          t1_mpi_outer_wait_time =MPI_Wtime()
#endif
          do i = 1, stripe_count
#ifdef WITH_OPENMP
            call MPI_Wait(top_send_request(i), mpi_status, mpierr)
#else
            call MPI_Wait(top_send_request(i), MPI_STATUS_IGNORE, mpierr)
#endif
          enddo
#ifdef WITH_GPU_VERSION
!          t2_mpi_outer_wait_time =MPI_Wtime()
!          t0_mpi_outer_wait_time =t0_mpi_outer_wait_time+(t2_mpi_outer_wait_time-t1_mpi_outer_wait_time)
#endif
        endif
#ifdef WITH_GPU_VERSION
!        t0_result_time = MPI_Wtime()
#endif
        ! Care about the result

        if (my_prow == 0) then

          ! topmost process sends nbw rows to destination processes

          do j=0,nfact-1

            num_blk = sweep*nfact+j ! global number of destination block, 0 based
            if (num_blk*nblk >= na) exit

            nbuf = mod(num_blk, num_result_buffers) + 1 ! buffer number to get this block

#ifdef WITH_OPENMP
            call MPI_Wait(result_send_request(nbuf), mpi_status, mpierr)
#else
            call MPI_Wait(result_send_request(nbuf), MPI_STATUS_IGNORE, mpierr)
#endif

            dst = mod(num_blk, np_rows)

            if (dst == 0) then
#ifdef WITH_GPU_VERSION
              row_group_size = min(na - num_blk*nblk, nblk)
              call pack_row_group_complex(row_group(:, :), j * nblk + a_off,row_group_size)
              do i = 1, row_group_size
                q((num_blk / np_rows) * nblk + i, 1 : l_nev) = row_group(:, i)
              enddo
#else
              do i = 1, min(na - num_blk*nblk, nblk)
                call pack_row_complex(row, j*nblk+i+a_off)
                q((num_blk/np_rows)*nblk+i,1:l_nev) = row(:)
              enddo
#endif
            else
#ifdef WITH_GPU_VERSION
              call pack_row_group_complex(result_buffer(:, :, nbuf), j * nblk + a_off, nblk)
#else
              do i = 1, nblk
                call pack_row_complex(result_buffer(:,i,nbuf),j*nblk+i+a_off)
              enddo
#endif
              call MPI_Isend(result_buffer(1,1,nbuf), l_nev*nblk, MPI_COMPLEX16, dst, &
                                   result_recv_tag, mpi_comm_rows, result_send_request(nbuf), mpierr)
            endif
          enddo

        else

          ! receive and store final result

          do j = num_bufs_recvd, num_result_blocks-1

            nbuf = mod(j, num_result_buffers) + 1 ! buffer number to get this block

            ! If there is still work to do, just test for the next result request
            ! and leave the loop if it is not ready, otherwise wait for all
            ! outstanding requests

            if (next_local_n > 0) then
#ifdef WITH_OPENMP
              call MPI_Test(result_recv_request(nbuf), flag, mpi_status, mpierr)

#else
              call MPI_Test(result_recv_request(nbuf), flag, MPI_STATUS_IGNORE, mpierr)
#endif
              if (.not.flag) exit
            else
#ifdef WITH_OPENMP
              call MPI_Wait(result_recv_request(nbuf), mpi_status, mpierr)

#else

              call MPI_Wait(result_recv_request(nbuf), MPI_STATUS_IGNORE, mpierr)
#endif
            endif

              ! Fill result buffer into q
              num_blk = j*np_rows + my_prow ! global number of current block, 0 based
              do i = 1, min(na - num_blk*nblk, nblk)
                q(j*nblk+i, 1:l_nev) = result_buffer(1:l_nev, i, nbuf)
              enddo

              ! Queue result buffer again if there are outstanding blocks left
              if (j+num_result_buffers < num_result_blocks) &
                    call MPI_Irecv(result_buffer(1,1,nbuf), l_nev*nblk, MPI_COMPLEX16, 0, result_recv_tag, &
                                   mpi_comm_rows, result_recv_request(nbuf), mpierr)

            enddo
            num_bufs_recvd = j

          endif

#ifdef WITH_GPU_VERSION
!          t2_result_time =MPI_Wtime()
!          t0_result_time = t0_result_time + ( t2_result_time - t1_result_time)
#endif

          ! Shift the remaining rows to the front of A (if necessary)

          offset = nbw - top_msg_length

          if (offset<0) then
            if (wantDebug) then
              write(error_unit,*) 'ELPA2_trans_ev_tridi_to_band_complex: internal error, offset for shifting = ',offset
            endif
            success = .false.
            return
          endif

          a_off = a_off + offset
          if (a_off + next_local_n + nbw > a_dim2) then
#ifdef WITH_OPENMP
#ifdef HAVE_DETAILED_TIMINGS
            call timer%start("OpenMP parallel")
#endif

!$omp parallel do private(my_thread, i, j), schedule(static, 1)
            do my_thread = 1, max_threads
              do i = 1, stripe_count
                do j = top_msg_length+1, top_msg_length+next_local_n
                  A(:,j,i,my_thread) = A(:,j+a_off,i,my_thread)
                enddo
              enddo
            enddo
#ifdef HAVE_DETAILED_TIMINGS
            call timer%stop("OpenMP parallel")
#endif

#else /*WITH_OPENMP */
            do i = 1, stripe_count
#ifdef WITH_GPU_VERSION
              chunk = min(next_local_n - 1, a_off)
              do j = top_msg_length + 1, top_msg_length + next_local_n, chunk
                top = min(j + chunk, top_msg_length + next_local_n)
                this_chunk = top - j + 1
                dev_offset = (0 + ( (j-1) * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) * 16_8
                dev_offset_1 = (0 + ( (j + a_off-1) * stripe_width) + ( (i-1) * stripe_width * a_dim2 )) *16_8
                tmp = cuda_d2d(1)
                istat =  cuda_memcpy( a_dev + dev_offset , a_dev +dev_offset_1,stripe_width*this_chunk*16_8, tmp)
                if (istat .ne. 0) then
                  print *,"trans_ev_tridi_to_band_complex: error in cudaMemcpy"
                  stop
                endif

              enddo
#else
              do j = top_msg_length+1, top_msg_length+next_local_n
                A(:,j,i) = A(:,j+a_off,i)
              enddo
#endif
            enddo
#endif /*WITH_OPENMP */

            a_off = 0
          endif
        enddo

#ifdef WITH_GPU_VERSION
!        t2_outer_do_time =MPI_Wtime()
!        t0_outer_do_time = t0_outer_do_time + ( t2_outer_do_time - t1_outer_do_time)
!
!        istat = cuda_ProfilerStop()
#endif

        ! Just for safety:
        if (ANY(top_send_request    /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR top_send_request ***',my_prow,my_pcol
        if (ANY(bottom_send_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR bottom_send_request ***',my_prow,my_pcol
        if (ANY(top_recv_request    /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR top_recv_request ***',my_prow,my_pcol
        if (ANY(bottom_recv_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR bottom_recv_request ***',my_prow,my_pcol

        if (my_prow == 0) then
#ifdef WITH_OPENMP
          allocate(mpi_statuses(MPI_STATUS_SIZE,num_result_buffers), stat=istat, errmsg=errorMessage)
          if (istat .ne. 0) then
            print *,"trans_ev_tridi_to_band_complex: error allocating mpi_statuses "//errorMessage
            stop
          endif

          call MPI_Waitall(num_result_buffers, result_send_request, mpi_statuses, mpierr)
          deallocate(mpi_statuses, stat=istat, errmsg=errorMessage)
          if (istat .ne. 0) then
            print *,"trans_ev_tridi_to_band_complex: error deallocating mpi_statuses "//errorMessage
            stop
          endif

#else
          call MPI_Waitall(num_result_buffers, result_send_request, MPI_STATUSES_IGNORE, mpierr)
#endif
        endif

        if (ANY(result_send_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR result_send_request ***',my_prow,my_pcol
        if (ANY(result_recv_request /= MPI_REQUEST_NULL)) write(error_unit,*) '*** ERROR result_recv_request ***',my_prow,my_pcol

        if (my_prow==0 .and. my_pcol==0 .and. elpa_print_times) &
          write(error_unit,'(" Kernel time:",f10.3," MFlops: ",f10.3)') kernel_time, kernel_flops/kernel_time*1.d-6

        ! deallocate all working space

#ifndef WITH_GPU_VERSION
        deallocate(a, stat=istat, errmsg=errorMessage)
        if (istat .ne. 0) then
          print *,"trans_ev_tridi_to_band_complex: error deallocating a "//errorMessage
          stop
        endif

#endif
     deallocate(row, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error deallocating row "//errorMessage
       stop
     endif

     deallocate(limits, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error deallocating limits "//errorMessage
       stop
     endif

     deallocate(result_send_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error deallocating result_send_request "//errorMessage
       stop
     endif

     deallocate(result_recv_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error deallocating result_recv_request "//errorMessage
       stop
     endif

     deallocate(top_border_send_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error deallocating top_border_send_buffer "//errorMessage
       stop
     endif

     deallocate(top_border_recv_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error deallocating top_border_recv_buffer "//errorMessage
       stop
     endif

     deallocate(bottom_border_send_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error deallocating top_border_send_buffer "//errorMessage
       stop
     endif

     deallocate(bottom_border_recv_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error deallocating bottom_border_recv_buffer "//errorMessage
       stop
     endif

     deallocate(result_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error deallocating result_buffer "//errorMessage
       stop
     endif

     deallocate(bcast_buffer, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error deallocating bcast_buffer "//errorMessage
       stop
     endif

     deallocate(top_send_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error deallocating top_send_request "//errorMessage
       stop
     endif

     deallocate(top_recv_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error deallocating top_recv_request "//errorMessage
       stop
     endif

     deallocate(bottom_send_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error deallocating bottom_send_request "//errorMessage
       stop
     endif

     deallocate(bottom_recv_request, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error deallocating bottom_recv_request "//errorMessage
       stop
     endif

#ifdef WITH_GPU_VERSION
     istat = cuda_free(a_dev)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error in cudaFree"
       stop
     endif

     istat = cuda_free(hh_tau_dev)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error in cudaFree"
       stop
     endif

     istat = cuda_free(hh_dot_dev)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error in cudaFree"
       stop
     endif

     istat = cuda_free(row_dev)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error in cudaFree"
       stop
     endif

     deallocate(row_group, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error deallocating row_group "//errorMessage
       stop
     endif

     istat= cuda_free(row_group_dev)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error in cudaFree"
       stop
     endif

     istat =  cuda_free(bcast_buffer_dev)
     if (istat .ne. 0) then
       print *,"trans_ev_tridi_to_band_complex: error in cudaFree"
       stop
     endif

#endif

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("trans_ev_tridi_to_band_complex")
#endif
     return

contains
#ifndef WITH_GPU_VERSION

#ifdef WITH_OPENMP
  subroutine pack_row_complex(row, n)
#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif
    implicit none
    complex*16 :: row(:)
    integer    :: n, i, noff, nl, nt

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("pack_row_complex")
#endif
    do nt = 1, max_threads
      do i = 1, stripe_count
        noff = (nt-1)*thread_width + (i-1)*stripe_width
        nl   = min(stripe_width, nt*thread_width-noff, l_nev-noff)
        if (nl<=0) exit
        row(noff+1:noff+nl) = a(1:nl,n,i,nt)
      enddo
    enddo

#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("pack_row_complex")
#endif

  end subroutine pack_row_complex
#else /* WITH_OPENMP */

  subroutine pack_row_complex(row, n)

#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif
    implicit none
    complex*16 :: row(:)
    integer    :: n, i, noff, nl

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("unpack_row_complex")
#endif

    do i=1,stripe_count
      nl = merge(stripe_width, last_stripe_width, i<stripe_count)
      noff = (i-1)*stripe_width
      row(noff+1:noff+nl) = a(1:nl,n,i)
    enddo

#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("unpack_row_complex")
#endif


  end subroutine pack_row_complex
#endif /* WITH_OPENMP */

#ifdef WITH_OPENMP
  subroutine unpack_row_complex(row, n, my_thread)

#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif

    implicit none

    ! Private variables in OMP regions (my_thread) should better be in the argument list!
    integer, intent(in)     :: n, my_thread
    complex*16, intent(in)  :: row(:)
    integer                 :: i, noff, nl

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("unpack_row_complex")
#endif

    do i=1,stripe_count
      noff = (my_thread-1)*thread_width + (i-1)*stripe_width
      nl   = min(stripe_width, my_thread*thread_width-noff, l_nev-noff)
      if (nl<=0) exit
      a(1:nl,n,i,my_thread) = row(noff+1:noff+nl)
    enddo

#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("unpack_row_complex")
#endif
  end subroutine unpack_row_complex

#else /* WITH_OPENMP */

  subroutine unpack_row_complex(row, n)
#ifdef HAVE_DETAILED_TIMINGS
    use timings
#endif

    implicit none

    complex*16 :: row(:)
    integer    :: n, i, noff, nl

#ifdef HAVE_DETAILED_TIMINGS
    call timer%start("unpack_row_complex")
#endif


    do i=1,stripe_count
      nl = merge(stripe_width, last_stripe_width, i<stripe_count)
      noff = (i-1)*stripe_width
      a(1:nl,n,i) = row(noff+1:noff+nl)
    enddo

#ifdef HAVE_DETAILED_TIMINGS
    call timer%stop("unpack_row_complex")
#endif

  end  subroutine unpack_row_complex
#endif /* WITH_OPENMP */

#ifdef WITH_OPENMP
  subroutine compute_hh_trafo_complex(off, ncols, istripe, my_thread, THIS_COMPLEX_ELPA_KERNEL)
#else
  subroutine compute_hh_trafo_complex(off, ncols, istripe, THIS_COMPLEX_ELPA_KERNEL)
#endif

#if defined(WITH_COMPLEX_GENERIC_SIMPLE_KERNEL)
      use complex_generic_simple_kernel, only : single_hh_trafo_complex_generic_simple
#endif
#if defined(WITH_COMPLEX_GENERIC_SIMPLE_KERNEL)
      use complex_generic_kernel, only : single_hh_trafo_complex_generic
#endif
#ifdef HAVE_DETAILED_TIMINGS
      use timings
#endif
      implicit none
      integer, intent(in) :: THIS_COMPLEX_ELPA_KERNEL

        ! Private variables in OMP regions (my_thread) should better be in the argument list!

        integer           :: off, ncols, istripe, j, nl, jj
#ifdef WITH_OPENMP
        integer           :: my_thread, noff
#endif
        real*8            :: ttt

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!        Currently (on Sandy Bridge), single is faster than double
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        complex*16        :: w(nbw,2)

#ifdef HAVE_DETAILED_TIMINGS
        call timer%start("compute_hh_trafo_complex")
#endif

#ifdef WITH_OPENMP
        if (istripe<stripe_count) then
          nl = stripe_width
        else
          noff = (my_thread-1)*thread_width + (istripe-1)*stripe_width
          nl = min(my_thread*thread_width-noff, l_nev-noff)
          if(nl<=0) return
        endif
#else
        nl = merge(stripe_width, last_stripe_width, istripe<stripe_count)
#endif


#if defined(WITH_COMPLEX_AVX_BLOCK2_KERNEL)
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        if (THIS_COMPLEX_ELPA_KERNEL .eq. COMPLEX_ELPA_KERNEL_AVX_BLOCK2) then
#endif  /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
          ttt = mpi_wtime()
          do j = ncols, 2, -2
            w(:,1) = bcast_buffer(1:nbw,j+off)
            w(:,2) = bcast_buffer(1:nbw,j+off-1)
#ifdef WITH_OPENMP
            call double_hh_trafo_complex_sse_avx_2hv(a(1,j+off+a_off-1,istripe,my_thread), &
                                                       w, nbw, nl, stripe_width, nbw)
#else
            call double_hh_trafo_complex_sse_avx_2hv(a(1,j+off+a_off-1,istripe), &
                                                       w, nbw, nl, stripe_width, nbw)
#endif
          enddo
#ifdef WITH_OPENMP
          if (j==1) call single_hh_trafo_complex_sse_avx_1hv(a(1,1+off+a_off,istripe,my_thread), &
                                                             bcast_buffer(1,off+1), nbw, nl, stripe_width)
#else
          if (j==1) call single_hh_trafo_complex_sse_avx_1hv(a(1,1+off+a_off,istripe), &
                                                             bcast_buffer(1,off+1), nbw, nl, stripe_width)
#endif
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        endif
#endif  /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_AVX_BLOCK2_KERNEL */


#if defined(WITH_COMPLEX_GENERIC_SIMPLE_KERNEL)
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        if (THIS_COMPLEX_ELPA_KERNEL .eq. COMPLEX_ELPA_KERNEL_GENERIC_SIMPLE) then
#endif /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
          ttt = mpi_wtime()
          do j = ncols, 1, -1
#ifdef WITH_OPENMP
            call single_hh_trafo_complex_generic_simple(a(1,j+off+a_off,istripe,my_thread), &
                                                          bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
            call single_hh_trafo_complex_generic_simple(a(1,j+off+a_off,istripe), &
                                                          bcast_buffer(1,j+off),nbw,nl,stripe_width)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_GENERIC_SIMPLE_KERNEL */


#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        if (THIS_COMPLEX_ELPA_KERNEL .eq. COMPLEX_ELPA_KERNEL_GENERIC .or. &
            THIS_COMPLEX_ELPA_KERNEL .eq. COMPLEX_ELPA_KERNEL_BGP .or. &
            THIS_COMPLEX_ELPA_KERNEL .eq. COMPLEX_ELPA_KERNEL_BGQ ) then
#endif /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
          ttt = mpi_wtime()
          do j = ncols, 1, -1
#ifdef WITH_OPENMP
            call single_hh_trafo_complex_generic(a(1,j+off+a_off,istripe,my_thread), &
                                                   bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
            call single_hh_trafo_complex_generic(a(1,j+off+a_off,istripe), &
                                                   bcast_buffer(1,j+off),nbw,nl,stripe_width)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */


#if defined(WITH_COMPLEX_SSE_KERNEL)
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        if (THIS_COMPLEX_ELPA_KERNEL .eq. COMPLEX_ELPA_KERNEL_SSE) then
#endif /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
          ttt = mpi_wtime()
          do j = ncols, 1, -1
#ifdef WITH_OPENMP
            call single_hh_trafo_complex(a(1,j+off+a_off,istripe,my_thread), &
                                           bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
            call single_hh_trafo_complex(a(1,j+off+a_off,istripe), &
                                           bcast_buffer(1,j+off),nbw,nl,stripe_width)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_SSE_KERNEL */


!#if defined(WITH_AVX_SANDYBRIDGE)
!              call single_hh_trafo_complex_sse_avx_1hv(a(1,j+off+a_off,istripe),bcast_buffer(1,j+off),nbw,nl,stripe_width)
!#endif

!#if defined(WITH_AMD_BULLDOZER)
!              call single_hh_trafo_complex_sse_avx_1hv(a(1,j+off+a_off,istripe),bcast_buffer(1,j+off),nbw,nl,stripe_width)
!#endif

#if defined(WITH_COMPLEX_AVX_BLOCK1_KERNEL)
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        if (THIS_COMPLEX_ELPA_KERNEL .eq. COMPLEX_ELPA_KERNEL_AVX_BLOCK1) then
#endif /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
          ttt = mpi_wtime()
          do j = ncols, 1, -1
#ifdef WITH_OPENMP
            call single_hh_trafo_complex_sse_avx_1hv(a(1,j+off+a_off,istripe,my_thread), &
                                                       bcast_buffer(1,j+off),nbw,nl,stripe_width)
#else
            call single_hh_trafo_complex_sse_avx_1hv(a(1,j+off+a_off,istripe), &
                                                       bcast_buffer(1,j+off),nbw,nl,stripe_width)
#endif
          enddo
#if defined(WITH_NO_SPECIFIC_COMPLEX_KERNEL)
        endif
#endif /* WITH_NO_SPECIFIC_COMPLEX_KERNEL */
#endif /* WITH_COMPLEX_AVX_BLOCK1_KERNE */

#ifdef WITH_OPENMP
        if (my_thread==1) then
#endif
          kernel_flops = kernel_flops + 4*4*int(nl,8)*int(ncols,8)*int(nbw,8)
          kernel_time  = kernel_time + mpi_wtime()-ttt
#ifdef WITH_OPENMP
        endif
#endif
#ifdef HAVE_DETAILED_TIMINGS
        call timer%stop("compute_hh_trafo_complex")
#endif


    end subroutine compute_hh_trafo_complex

#endif /* WITH_GPU_VERSION */

#ifdef WITH_GPU_VERSION
    ! The host wrapper for extracting "tau" from the HH reflectors (see the
    ! kernel below)
    subroutine extract_hh_tau_complex(nbw, n, is_zero)

      implicit none
      integer, value :: nbw, n
      logical, value :: is_zero
      integer        :: val_is_zero

      if (is_zero) then
        val_is_zero = 1
      else
        val_is_zero = 0
      endif
      call launch_extract_hh_tau_c_kernel_complex(bcast_buffer_dev,hh_tau_dev, nbw, n,val_is_zero)
    end subroutine

    subroutine compute_hh_dot_products_complex(nbw, n)

      implicit none
      integer, value :: nbw, n

      if (n .le. 1) return
      call launch_compute_hh_dotp_c_kernel_complex( bcast_buffer_dev, hh_dot_dev, nbw,n)
     end subroutine

     subroutine pack_row_group_complex(rows, n_offset, row_count)

       implicit none
       integer, intent(in) :: n_offset, row_count
       complex*16          :: rows(:,:)
       integer             :: max_idx

       max_idx = (stripe_count - 1) * stripe_width + last_stripe_width
       call launch_my_pack_c_kernel_complex(row_count, n_offset, max_idx, stripe_width,a_dim2, stripe_count, &
                                            l_nev, a_dev, row_group_dev)
       istat =  cuda_memcpy( loc(rows(:, 1: row_count)), row_group_dev ,row_count * l_nev * 16_8 ,2)
       if (istat .ne. 0) then
         print *,"error in cudaMemcpy"
         stop
       endif

     end subroutine

     subroutine unpack_row_group_complex(rows, n_offset, row_count)

       implicit none
       integer, intent(in)    :: n_offset, row_count
       complex*16, intent(in) :: rows(:, :)
       integer                :: max_idx
       integer                :: i

       max_idx = (stripe_count - 1) * stripe_width + last_stripe_width
       istat =  cuda_memcpy( row_group_dev , loc(rows(1, 1)),row_count * l_nev* 16_8 ,1)
       if (istat .ne. 0) then
         print *,"error in cudaMemcpy"
         stop
       endif

       call launch_my_unpack_c_kernel_complex( row_count, n_offset,max_idx,stripe_width,a_dim2, stripe_count, l_nev, &
                                              row_group_dev,a_dev)
     end subroutine

     subroutine unpack_and_prepare_row_group_complex(next_unpack_idx, force)

       implicit none
       integer, intent(in) :: next_unpack_idx
       logical, intent(in) :: force

       if (row_group_size == 0) then
         ! Nothing to flush, just prepare for the upcoming row
         row_group_size = 1
       else
         if (force .or. (row_group_size == nblk) .or. (unpack_idx + 1 /=next_unpack_idx)) then
           ! A flush and a reset must  performed
           call unpack_row_group_complex(row_group(:, :), unpack_idx - row_group_size, row_group_size)
           row_group_size = 1
         else
           ! Just prepare for the upcoming row
           row_group_size = row_group_size + 1
         endif
       endif
       ! Always update the index for the upcoming row
       unpack_idx = next_unpack_idx

    end subroutine

    subroutine compute_hh_trafo_complex_gpu(off, ncols, istripe, a_off, dev_offset, dev_offset_1, dev_offset_2)

      use iso_c_binding

      implicit none
      integer, intent(in) :: off, ncols, istripe
      integer             :: nl
      real*8              :: ttt

      integer             :: a_off
      integer(c_size_t)   :: dev_offset, dev_offset_1, dev_offset_2

      if (ncols < 1) return
      ttt = mpi_wtime()
      nl = merge(stripe_width, last_stripe_width, istripe < stripe_count)

      dev_offset = (0 + ( (  a_off + off-1 )* stripe_width) + ( (istripe - 1)*stripe_width*a_dim2 )) *16_8
      dev_offset_1 = (0 +  (  off-1 )* nbw) *16_8
      dev_offset_2 =( off-1 )*16_8

!      t1_compute_kernel =MPI_Wtime()
      call launch_compute_hh_trafo_c_kernel_complex(a_dev + dev_offset,bcast_buffer_dev + dev_offset_1, &
                                                    hh_tau_dev + dev_offset_2, nl, nbw,stripe_width, off,ncols)

!      time0 = time0 + time1
!      t2_compute_kernel =MPI_Wtime()
!      t0_compute_kernel =  t0_compute_kernel + t2_compute_kernel-t1_compute_kernel

      kernel_flops = kernel_flops + 4 * int(nl, 8) * int(ncols, 8) * int(nbw,8)
      kernel_time = kernel_time + mpi_wtime() - ttt
      n_times =n_times +1
    end subroutine compute_hh_trafo_complex_gpu

#endif /* WITH_GPU_VERSION */

end subroutine
#define DATATYPE REAL
#define BYTESIZE 8
#define REALCASE 1
#include "redist_band.X90"
#undef DATATYPE
#undef BYTESIZE
#undef REALCASE

#define DATATYPE COMPLEX
#define BYTESIZE 16
#define COMPLEXCASE 1
#include "redist_band.X90"
#undef DATATYPE
#undef BYTESIZE
#undef COMPLEXCASE

!---------------------------------------------------------------------------------------------------
! divide_band: sets the work distribution in band
! Proc n works on blocks block_limits(n)+1 .. block_limits(n+1)

subroutine divide_band(nblocks_total, n_pes, block_limits)
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   implicit none
   integer, intent(in)  :: nblocks_total ! total number of blocks in band
   integer, intent(in)  :: n_pes         ! number of PEs for division
   integer, intent(out) :: block_limits(0:n_pes)

   integer              :: n, nblocks, nblocks_left

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("divide_band")
#endif

   block_limits(0) = 0
   if (nblocks_total < n_pes) then
     ! Not enough work for all: The first tasks get exactly 1 block
     do n=1,n_pes
       block_limits(n) = min(nblocks_total,n)
     enddo
   else
     ! Enough work for all. If there is no exact loadbalance,
     ! the LAST tasks get more work since they are finishing earlier!
     nblocks = nblocks_total/n_pes
     nblocks_left = nblocks_total - n_pes*nblocks
     do n=1,n_pes
       if (n<=n_pes-nblocks_left) then
         block_limits(n) = block_limits(n-1) + nblocks
       else
         block_limits(n) = block_limits(n-1) + nblocks + 1
       endif
     enddo
   endif

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("divide_band")
#endif

end subroutine

subroutine band_band_real(na, nb, nb2, ab, ab2, d, e, mpi_comm)

!-------------------------------------------------------------------------------
! band_band_real:
! Reduces a real symmetric banded matrix to a real symmetric matrix with smaller bandwidth. Householder transformations are not stored.
! Matrix size na and original bandwidth nb have to be a multiple of the target bandwidth nb2. (Hint: expand your matrix with zero entries, if this
! requirement doesn't hold)
!
!  na          Order of matrix
!
!  nb          Semi bandwidth of original matrix
!
!  nb2         Semi bandwidth of target matrix
!
!  ab          Input matrix with bandwidth nb. The leading dimension of the banded matrix has to be 2*nb. The parallel data layout
!              has to be accordant to divide_band(), i.e. the matrix columns block_limits(n)*nb+1 to min(na, block_limits(n+1)*nb)
!              are located on rank n.
!
!  ab2         Output matrix with bandwidth nb2. The leading dimension of the banded matrix is 2*nb2. The parallel data layout is
!              accordant to divide_band(), i.e. the matrix columns block_limits(n)*nb2+1 to min(na, block_limits(n+1)*nb2) are located
!              on rank n.
!
!  d(na)       Diagonal of tridiagonal matrix, set only on PE 0, set only if ab2 = 1 (output)
!
!  e(na)       Subdiagonal of tridiagonal matrix, set only on PE 0, set only if ab2 = 1 (output)
!
!  mpi_comm
!              MPI-Communicator for the total processor set
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   implicit none

   integer, intent(in)    ::  na, nb, nb2, mpi_comm
   real*8, intent(inout)  :: ab(2*nb,*)
   real*8, intent(inout)  :: ab2(2*nb2,*)
   real*8, intent(out)    :: d(na), e(na) ! set only on PE 0

!----------------

   real*8                 :: hv(nb,nb2), w(nb,nb2), w_new(nb,nb2), tau(nb2), hv_new(nb,nb2), &
                             tau_new(nb2), ab_s(1+nb,nb2), ab_r(1+nb,nb2), ab_s2(2*nb2,nb2), hv_s(nb,nb2)

   real*8                 :: work(nb*nb2), work2(nb2*nb2)
   integer                :: lwork, info

   integer                :: istep, i, n, dest
   integer                :: n_off, na_s
   integer                :: my_pe, n_pes, mpierr
   integer                :: nblocks_total, nblocks
   integer                :: nblocks_total2, nblocks2
   integer                :: ireq_ab, ireq_hv
   integer                :: mpi_status(MPI_STATUS_SIZE)
   integer, allocatable   :: mpi_statuses(:,:)
   integer, allocatable   :: block_limits(:), block_limits2(:), ireq_ab2(:)

!----------------

   integer                :: j, nc, nr, ns, ne, iblk
   integer                :: istat
   character(200)         :: errorMessage

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("band_band_real")
#endif
   call mpi_comm_rank(mpi_comm,my_pe,mpierr)
   call mpi_comm_size(mpi_comm,n_pes,mpierr)

   ! Total number of blocks in the band:
   nblocks_total = (na-1)/nb + 1
   nblocks_total2 = (na-1)/nb2 + 1

   ! Set work distribution
   allocate(block_limits(0:n_pes), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"error allocating block_limits "//errorMessage
     stop
   endif
   call divide_band(nblocks_total, n_pes, block_limits)

   allocate(block_limits2(0:n_pes), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"error allocating block_limits2 "//errorMessage
     stop
   endif

   call divide_band(nblocks_total2, n_pes, block_limits2)

   ! nblocks: the number of blocks for my task
   nblocks = block_limits(my_pe+1) - block_limits(my_pe)
   nblocks2 = block_limits2(my_pe+1) - block_limits2(my_pe)

   allocate(ireq_ab2(1:nblocks2), stat=istat, errmsg=errorMessage)
   if (istat .ne. 0) then
     print *,"error allocating ireq_ab2 "//errorMessage
     stop
   endif

   ireq_ab2 = MPI_REQUEST_NULL
   if (nb2>1) then
     do i=0,nblocks2-1
       call mpi_irecv(ab2(1,i*nb2+1),2*nb2*nb2,mpi_real8,0,3,mpi_comm,ireq_ab2(i+1),mpierr)
     enddo
   endif

   ! n_off: Offset of ab within band
   n_off = block_limits(my_pe)*nb
   lwork = nb*nb2
   dest = 0

   ireq_ab = MPI_REQUEST_NULL
   ireq_hv = MPI_REQUEST_NULL

   ! ---------------------------------------------------------------------------
   ! Start of calculations

   na_s = block_limits(my_pe)*nb + 1

   if (my_pe>0 .and. na_s<=na) then
     ! send first nb2 columns to previous PE
     ! Only the PE owning the diagonal does that (sending 1 element of the subdiagonal block also)
     do i=1,nb2
       ab_s(1:nb+1,i) = ab(1:nb+1,na_s-n_off+i-1)
     enddo
     call mpi_isend(ab_s,(nb+1)*nb2,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
   endif

   do istep=1,na/nb2

     if (my_pe==0) then

       n = MIN(na-na_s-nb2+1,nb) ! number of rows to be reduced
       hv(:,:) = 0
       tau(:) = 0

       ! The last step (istep=na-1) is only needed for sending the last HH vectors.
       ! We don't want the sign of the last element flipped (analogous to the other sweeps)
       if (istep < na/nb2) then

         ! Transform first block column of remaining matrix
         call dgeqrf(n, nb2, ab(1+nb2,na_s-n_off), 2*nb-1, tau, work, lwork, info);

         do i=1,nb2
           hv(i,i) = 1.0
           hv(i+1:n,i) = ab(1+nb2+1:1+nb2+n-i,na_s-n_off+i-1)
           ab(1+nb2+1:2*nb,na_s-n_off+i-1) = 0
         enddo

       endif

       if (nb2==1) then
         d(istep) = ab(1,na_s-n_off)
         e(istep) = ab(2,na_s-n_off)
         if (istep == na) then
           e(na) = 0
         endif
       else
         ab_s2 = 0
         ab_s2(:,:) = ab(1:nb2+1,na_s-n_off:na_s-n_off+nb2-1)
         if (block_limits2(dest+1)<istep) then
           dest = dest+1
         endif
         call mpi_send(ab_s2,2*nb2*nb2,mpi_real8,dest,3,mpi_comm,mpierr)
       endif

     else
       if (na>na_s+nb2-1) then
         ! Receive Householder vectors from previous task, from PE owning subdiagonal
         call mpi_recv(hv,nb*nb2,mpi_real8,my_pe-1,2,mpi_comm,mpi_status,mpierr)
         do i=1,nb2
           tau(i) = hv(i,i)
           hv(i,i) = 1.
         enddo
       endif
     endif

     na_s = na_s+nb2
     if (na_s-n_off > nb) then
       ab(:,1:nblocks*nb) = ab(:,nb+1:(nblocks+1)*nb)
       ab(:,nblocks*nb+1:(nblocks+1)*nb) = 0
       n_off = n_off + nb
     endif

     do iblk=1,nblocks
       ns = na_s + (iblk-1)*nb - n_off ! first column in block
       ne = ns+nb-nb2                    ! last column in block

       if (ns+n_off>na) exit

         nc = MIN(na-ns-n_off+1,nb) ! number of columns in diagonal block
         nr = MIN(na-nb-ns-n_off+1,nb) ! rows in subdiagonal block (may be < 0!!!)
                                       ! Note that nr>=0 implies that diagonal block is full (nc==nb)!

         call wy_gen(nc,nb2,w,hv,tau,work,nb)

         if (iblk==nblocks .and. nc==nb) then
           !request last nb2 columns
           call mpi_recv(ab_r,(nb+1)*nb2,mpi_real8,my_pe+1,1,mpi_comm,mpi_status,mpierr)
           do i=1,nb2
             ab(1:nb+1,ne+i-1) = ab_r(:,i)
           enddo
         endif

         hv_new(:,:) = 0 ! Needed, last rows must be 0 for nr < nb
         tau_new(:) = 0

         if (nr>0) then
           call wy_right(nr,nb,nb2,ab(nb+1,ns),2*nb-1,w,hv,work,nb)

           call dgeqrf(nr,nb2,ab(nb+1,ns),2*nb-1,tau_new,work,lwork,info);

           do i=1,nb2
             hv_new(i,i) = 1.0
             hv_new(i+1:,i) = ab(nb+2:2*nb-i+1,ns+i-1)
             ab(nb+2:,ns+i-1) = 0
           enddo

           !send hh-vector
           if (iblk==nblocks) then
             call mpi_wait(ireq_hv,mpi_status,mpierr)
             hv_s = hv_new
             do i=1,nb2
               hv_s(i,i) = tau_new(i)
             enddo
             call mpi_isend(hv_s,nb*nb2,mpi_real8,my_pe+1,2,mpi_comm,ireq_hv,mpierr)
           endif

         endif

         call wy_symm(nc,nb2,ab(1,ns),2*nb-1,w,hv,work,work2,nb)

         if (my_pe>0 .and. iblk==1) then
           !send first nb2 columns to previous PE
           call mpi_wait(ireq_ab,mpi_status,mpierr)
           do i=1,nb2
             ab_s(1:nb+1,i) = ab(1:nb+1,ns+i-1)
           enddo
           call mpi_isend(ab_s,(nb+1)*nb2,mpi_real8,my_pe-1,1,mpi_comm,ireq_ab,mpierr)
         endif

         if (nr>0) then
           call wy_gen(nr,nb2,w_new,hv_new,tau_new,work,nb)
           call wy_left(nb-nb2,nr,nb2,ab(nb+1-nb2,ns+nb2),2*nb-1,w_new,hv_new,work,nb)
         endif

         ! Use new HH vector for the next block
         hv(:,:) = hv_new(:,:)
         tau = tau_new
       enddo
     enddo

     ! Finish the last outstanding requests
     call mpi_wait(ireq_ab,mpi_status,mpierr)
     call mpi_wait(ireq_hv,mpi_status,mpierr)
     allocate(mpi_statuses(MPI_STATUS_SIZE,nblocks2), stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"error allocating mpi_statuses "//errorMessage
       stop
     endif

     call mpi_waitall(nblocks2,ireq_ab2,mpi_statuses,mpierr)
     deallocate(mpi_statuses, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"error deallocating mpi_statuses "//errorMessage
       stop
     endif

     call mpi_barrier(mpi_comm,mpierr)

     deallocate(block_limits, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"error deallocating block_limits "//errorMessage
       stop
     endif

     deallocate(block_limits2, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"error deallocating block_limits2 "//errorMessage
       stop
     endif

     deallocate(ireq_ab2, stat=istat, errmsg=errorMessage)
     if (istat .ne. 0) then
       print *,"error deallocating ireq_ab2 "//errorMessage
       stop
     endif
#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("band_band_real")
#endif

end subroutine

subroutine wy_gen(n, nb, W, Y, tau, mem, lda)
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   implicit none
   integer, intent(in) :: n		!length of householder-vectors
   integer, intent(in) :: nb		!number of householder-vectors
   integer, intent(in) :: lda		!leading dimension of Y and W
   real*8, intent(in)  :: Y(lda,nb)	!matrix containing nb householder-vectors of length b
   real*8, intent(in)  :: tau(nb)	!tau values
   real*8, intent(out) :: W(lda,nb)	!output matrix W
   real*8, intent(in)  :: mem(nb)	!memory for a temporary matrix of size nb

   integer             :: i

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("wy_gen")
#endif

   W(1:n,1) = tau(1)*Y(1:n,1)
   do i=2,nb
     W(1:n,i) = tau(i)*Y(1:n,i)
     call DGEMV('T',n,i-1,1.d0,Y,lda,W(1,i),1,0.d0,mem,1)
     call DGEMV('N',n,i-1,-1.d0,W,lda,mem,1,1.d0,W(1,i),1)
   enddo

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("wy_gen")
#endif

end subroutine

subroutine wy_left(n, m, nb, A, lda, W, Y, mem, lda2)
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   implicit none
   integer, intent(in)   :: n		!width of the matrix A
   integer, intent(in)   :: m		!length of matrix W and Y
   integer, intent(in)   :: nb		!width of matrix W and Y
   integer, intent(in)   :: lda		!leading dimension of A
   integer, intent(in)   :: lda2		!leading dimension of W and Y
   real*8, intent(inout) :: A(lda,*)	!matrix to be transformed
   real*8, intent(in)    :: W(m,nb)	!blocked transformation matrix W
   real*8, intent(in)    :: Y(m,nb)	!blocked transformation matrix Y
   real*8, intent(inout) :: mem(n,nb)	!memory for a temporary matrix of size n x nb

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("wy_left")
#endif

   call DGEMM('T', 'N', nb, n, m, 1.d0, W, lda2, A, lda, 0.d0, mem, nb)
   call DGEMM('N', 'N', m, n, nb, -1.d0, Y, lda2, mem, nb, 1.d0, A, lda)

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("wy_left")
#endif

end subroutine

! --------------------------------------------------------------------------------------------------

subroutine wy_right(n, m, nb, A, lda, W, Y, mem, lda2)
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   implicit none
   integer, intent(in)   :: n		!height of the matrix A
   integer, intent(in)   :: m		!length of matrix W and Y
   integer, intent(in)   :: nb		!width of matrix W and Y
   integer, intent(in)   :: lda		!leading dimension of A
   integer, intent(in)   :: lda2		!leading dimension of W and Y
   real*8, intent(inout) :: A(lda,*)	!matrix to be transformed
   real*8, intent(in)    :: W(m,nb)	!blocked transformation matrix W
   real*8, intent(in)    :: Y(m,nb)	!blocked transformation matrix Y
   real*8, intent(inout) :: mem(n,nb)	!memory for a temporary matrix of size n x nb

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("wy_right")
#endif

   call DGEMM('N', 'N', n, nb, m, 1.d0, A, lda, W, lda2, 0.d0, mem, n)
   call DGEMM('N', 'T', n, m, nb, -1.d0, mem, n, Y, lda2, 1.d0, A, lda)

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("wy_right")
#endif

end subroutine

! --------------------------------------------------------------------------------------------------

subroutine wy_symm(n, nb, A, lda, W, Y, mem, mem2, lda2)
#ifdef HAVE_DETAILED_TIMINGS
   use timings
#endif
   implicit none
   integer, intent(in)   :: n		!width/heigth of the matrix A; length of matrix W and Y
   integer, intent(in)   :: nb		!width of matrix W and Y
   integer, intent(in)   :: lda		!leading dimension of A
   integer, intent(in)   :: lda2		!leading dimension of W and Y
   real*8, intent(inout) :: A(lda,*)	!matrix to be transformed
   real*8, intent(in)    :: W(n,nb)	!blocked transformation matrix W
   real*8, intent(in)    :: Y(n,nb)	!blocked transformation matrix Y
   real*8                :: mem(n,nb)	!memory for a temporary matrix of size n x nb
   real*8                :: mem2(nb,nb)	!memory for a temporary matrix of size nb x nb

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("wy_symm")
#endif

   call DSYMM('L', 'L', n, nb, 1.d0, A, lda, W, lda2, 0.d0, mem, n)
   call DGEMM('T', 'N', nb, nb, n, 1.d0, mem, n, W, lda2, 0.d0, mem2, nb)
   call DGEMM('N', 'N', n, nb, nb, -0.5d0, Y, lda2, mem2, nb, 1.d0, mem, n)
   call DSYR2K('L', 'N', n, nb, -1.d0, Y, lda2, mem, n, 1.d0, A, lda)

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("wy_symm")
#endif
end subroutine

end module ELPA2
