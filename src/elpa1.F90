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
!    This particular source code file contains additions, changes and
!    enhancements authored by Intel Corporation which is not part of
!    the ELPA consortium.
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

#include "config-f90.h"

module ELPA1

  use elpa_utilities
  use elpa1_compute

#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
  implicit none

  PRIVATE ! By default, all routines contained are private

  ! The following routines are public:

  public :: get_elpa_row_col_comms     ! Sets MPI row/col communicators

  public :: solve_evp_real             ! Driver routine for real eigenvalue problem
  public :: solve_evp_complex          ! Driver routine for complex eigenvalue problem

  ! Timing results, set by every call to solve_evp_xxx

  real*8, public :: time_evp_fwd    ! forward transformations (to tridiagonal form)
  real*8, public :: time_evp_solve  ! time for solving the tridiagonal system
  real*8, public :: time_evp_back   ! time for back transformations of eigenvectors

  ! Set elpa_print_times to .true. for explicit timing outputs

  logical, public :: elpa_print_times = .false.

  include 'mpif.h'

contains

!-------------------------------------------------------------------------------

function get_elpa_row_col_comms(mpi_comm_global, my_prow, my_pcol, mpi_comm_rows, mpi_comm_cols) result(mpierr)

!-------------------------------------------------------------------------------
! get_elpa_row_col_comms:
! All ELPA routines need MPI communicators for communicating within
! rows or columns of processes, these are set here.
! mpi_comm_rows/mpi_comm_cols can be free'd with MPI_Comm_free if not used any more.
!
!  Parameters
!
!  mpi_comm_global   Global communicator for the calculations (in)
!
!  my_prow           Row coordinate of the calling process in the process grid (in)
!
!  my_pcol           Column coordinate of the calling process in the process grid (in)
!
!  mpi_comm_rows     Communicator for communicating within rows of processes (out)
!
!  mpi_comm_cols     Communicator for communicating within columns of processes (out)
!
!-------------------------------------------------------------------------------

   implicit none

   integer, intent(in)  :: mpi_comm_global, my_prow, my_pcol
   integer, intent(out) :: mpi_comm_rows, mpi_comm_cols

   integer :: mpierr

   ! mpi_comm_rows is used for communicating WITHIN rows, i.e. all processes
   ! having the same column coordinate share one mpi_comm_rows.
   ! So the "color" for splitting is my_pcol and the "key" is my row coordinate.
   ! Analogous for mpi_comm_cols

   call mpi_comm_split(mpi_comm_global,my_pcol,my_prow,mpi_comm_rows,mpierr)
   call mpi_comm_split(mpi_comm_global,my_prow,my_pcol,mpi_comm_cols,mpierr)

end function get_elpa_row_col_comms

!-------------------------------------------------------------------------------

function solve_evp_real(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols) result(success)

!-------------------------------------------------------------------------------
!  solve_evp_real: Solves the real eigenvalue problem
!
!  Parameters
!
!  na          Order of matrix a
!
!  nev         Number of eigenvalues needed.
!              The smallest nev eigenvalues/eigenvectors are calculated.
!
!  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!              Distribution is like in Scalapack.
!              The full matrix must be set (not only one half like in scalapack).
!              Destroyed on exit (upper and lower half).
!
!  lda         Leading dimension of a
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
!
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif
   implicit none

   integer, intent(in)  :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
   real*8               :: a(lda,matrixCols), ev(na), q(ldq,matrixCols)

   integer              :: my_prow, my_pcol, mpierr
   real*8, allocatable  :: e(:), tau(:)
   real*8               :: ttt0, ttt1
   logical              :: success
   logical, save        :: firstCall = .true.
   logical              :: wantDebug

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("solve_evp_real")
#endif

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)

   success = .true.

   wantDebug = .false.
   if (firstCall) then
     ! are debug messages desired?
     wantDebug = debug_messages_via_environment_variable()
     firstCall = .false.
   endif

   allocate(e(na), tau(na))

   ttt0 = MPI_Wtime()
   call tridiag_real(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, ev, e, tau)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) write(error_unit,*) 'Time tridiag_real :',ttt1-ttt0
   time_evp_fwd = ttt1-ttt0

   ttt0 = MPI_Wtime()
   call solve_tridi(na, nev, ev, e, q, ldq, nblk, matrixCols, mpi_comm_rows, &
                    mpi_comm_cols, wantDebug, success)
   if (.not.(success)) return

   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) write(error_unit,*) 'Time solve_tridi  :',ttt1-ttt0
   time_evp_solve = ttt1-ttt0

   ttt0 = MPI_Wtime()
   call trans_ev_real(na, nev, a, lda, tau, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) write(error_unit,*) 'Time trans_ev_real:',ttt1-ttt0
   time_evp_back = ttt1-ttt0

   deallocate(e, tau)

#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("solve_evp_real")
#endif

end function solve_evp_real

!-------------------------------------------------------------------------------


function solve_evp_complex(na, nev, a, lda, ev, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols) result(success)

!-------------------------------------------------------------------------------
!  solve_evp_complex: Solves the complex eigenvalue problem
!
!  Parameters
!
!  na          Order of matrix a
!
!  nev         Number of eigenvalues needed
!              The smallest nev eigenvalues/eigenvectors are calculated.
!
!  a(lda,matrixCols)    Distributed matrix for which eigenvalues are to be computed.
!              Distribution is like in Scalapack.
!              The full matrix must be set (not only one half like in scalapack).
!              Destroyed on exit (upper and lower half).
!
!  lda         Leading dimension of a
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
!
!-------------------------------------------------------------------------------
#ifdef HAVE_DETAILED_TIMINGS
 use timings
#endif

   implicit none

   integer, intent(in)     :: na, nev, lda, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols
   complex*16              :: a(lda,matrixCols), q(ldq,matrixCols)
   real*8                  :: ev(na)

   integer                 :: my_prow, my_pcol, np_rows, np_cols, mpierr
   integer                 :: l_rows, l_cols, l_cols_nev
   real*8, allocatable     :: q_real(:,:), e(:)
   complex*16, allocatable :: tau(:)
   real*8 ttt0, ttt1

   logical                 :: success
   logical, save           :: firstCall = .true.
   logical                 :: wantDebug

#ifdef HAVE_DETAILED_TIMINGS
   call timer%start("solve_evp_complex")
#endif

   call mpi_comm_rank(mpi_comm_rows,my_prow,mpierr)
   call mpi_comm_size(mpi_comm_rows,np_rows,mpierr)
   call mpi_comm_rank(mpi_comm_cols,my_pcol,mpierr)
   call mpi_comm_size(mpi_comm_cols,np_cols,mpierr)

   success = .true.

   wantDebug = .false.
   if (firstCall) then
     ! are debug messages desired?
     wantDebug = debug_messages_via_environment_variable()
     firstCall = .false.
   endif


   l_rows = local_index(na, my_prow, np_rows, nblk, -1) ! Local rows of a and q
   l_cols = local_index(na, my_pcol, np_cols, nblk, -1) ! Local columns of q

   l_cols_nev = local_index(nev, my_pcol, np_cols, nblk, -1) ! Local columns corresponding to nev

   allocate(e(na), tau(na))
   allocate(q_real(l_rows,l_cols))

   ttt0 = MPI_Wtime()
   call tridiag_complex(na, a, lda, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols, ev, e, tau)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) write(error_unit,*) 'Time tridiag_complex :',ttt1-ttt0
   time_evp_fwd = ttt1-ttt0

   ttt0 = MPI_Wtime()
   call solve_tridi(na, nev, ev, e, q_real, l_rows, nblk, matrixCols, mpi_comm_rows, &
                    mpi_comm_cols, wantDebug, success)
   if (.not.(success)) return

   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) write(error_unit,*) 'Time solve_tridi     :',ttt1-ttt0
   time_evp_solve = ttt1-ttt0

   ttt0 = MPI_Wtime()
   q(1:l_rows,1:l_cols_nev) = q_real(1:l_rows,1:l_cols_nev)

   call trans_ev_complex(na, nev, a, lda, tau, q, ldq, nblk, matrixCols, mpi_comm_rows, mpi_comm_cols)
   ttt1 = MPI_Wtime()
   if(my_prow==0 .and. my_pcol==0 .and. elpa_print_times) write(error_unit,*) 'Time trans_ev_complex:',ttt1-ttt0
   time_evp_back = ttt1-ttt0

   deallocate(q_real)
   deallocate(e, tau)
#ifdef HAVE_DETAILED_TIMINGS
   call timer%stop("solve_evp_complex")
#endif

end function solve_evp_complex



end module ELPA1
