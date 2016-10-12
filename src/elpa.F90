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
!
! writen by A. Marek (MPCDF), andreas.marek@mpcdf.mpg.de

#include "config-f90.h"
!> \brief Fortran module which provides the routines to the ELPA solver (1 and 2 stage)
module elpa
  use, intrinsic :: iso_c_binding, only : c_double, c_int
  use elpa1
  use elpa2

  implicit none

  public  :: elpa_solve_evp_real, elpa_solve_evp_complex

  contains
!-------------------------------------------------------------------------------
!>  \brief solve_evp_real: Fortran function to solve the real eigenvalue
!>         problem with either the ELPA 1stage or the ELPA 2stage solver
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
!>  \param THIS_REAL_ELPA_KERNEL_API (optional) specify used ELPA 2stage
!>                                              kernel via API (only evalulated if 2 stage solver is used_
!>
!>  \param use_qr (optional)                    use QR decomposition in the ELPA 2stage solver
!>
!>  \param method                               choose whether to use ELPA 1stage or 2stage solver
!>                                              possible values: "1stage" => use ELPA 1stage solver
!>                                                               "2stage" => use ELPA 2stage solver
!>                                                               "auto"   => (at the moment) use ELPA 2stage solver
!>
!>  \result success                             logical, false if error occured
!-------------------------------------------------------------------------------
    function elpa_solve_evp_real(na, nev, a, lda, ev, q, ldq, nblk,         &
                                 matrixCols, mpi_comm_rows, mpi_comm_cols,  &
                                 mpi_comm_all, THIS_REAL_ELPA_KERNEL_API,   &
                                 useQR, method) result(success)
      use iso_c_binding
      use elpa_utilities
      implicit none
      integer(kind=c_int), intent(in)           :: na, nev, lda, ldq, matrixCols, mpi_comm_rows, &
                                                   mpi_comm_cols, mpi_comm_all
      integer(kind=c_int), intent(in)           :: nblk
      real(kind=c_double), intent(inout)        :: ev(na)
#ifdef USE_ASSUMED_SIZE
      real(kind=c_double), intent(inout)        :: a(lda,*), q(ldq,*)
#else
      real(kind=c_double), intent(inout)        :: a(lda,matrixCols), q(ldq,matrixCols)
#endif
      logical, intent(in), optional             :: useQR
      integer(kind=c_int), intent(in), optional :: THIS_REAL_ELPA_KERNEL_API
      character(*), intent(in), optional        :: method

      logical                                   :: useELPA1
      logical                                   :: success

      useELPA1 = .false.

      if (present(method)) then
        if (trim(method) .eq. "1stage" .or. trim(method) .eq. "1STAGE") then
          useELPA1 = .true.
        else if (trim(method) .eq. "2stage" .or. trim(method) .eq. "2STAGE") then
          useELPA1 = .false.
        else if (trim(method) .eq. "auto" .or. trim(method) .eq. "AUTO") then
          useELPA1 = .false.
        else
          write(error_unit,*) "Specified method not known! Using ELPA 2-stage"
          useELPA1 = .false.
        endif
      endif

      if (useELPA1) then
        success = solve_evp_real_1stage(na, nev, a, lda, ev, q, ldq, nblk,                     &
                                        matrixCols, mpi_comm_rows, mpi_comm_cols)
      else
        success = solve_evp_real_2stage(na, nev, a, lda, ev, q, ldq, nblk,                     &
                                        matrixCols, mpi_comm_rows, mpi_comm_cols,              &
                                        mpi_comm_all,                                          &
                                        THIS_REAL_ELPA_KERNEL_API = THIS_REAL_ELPA_KERNEL_API, &
                                        useQR = useQR)
      endif

    end function elpa_solve_evp_real


!-------------------------------------------------------------------------------
!>  \brief solve_evp_complex: Fortran function to solve the complex eigenvalue
!>         problem with either the ELPA 1stage or the ELPA 2stage solver
!>
!>  Parameters
!>
!>  \param na                                      Order of matrix a
!>
!>  \param nev                                     Number of eigenvalues needed
!>
!>  \param a(lda,matrixCols)                       Distributed matrix for which eigenvalues are to be computed.
!>                                                 Distribution is like in Scalapack.
!>                                                 The full matrix must be set (not only one half like in scalapack).
!>                                                 Destroyed on exit (upper and lower half).
!>
!>  \param lda                                     Leading dimension of a
!>
!>  \param ev(na)                                  On output: eigenvalues of a, every processor gets the complete set
!>
!>  \param q(ldq,matrixCols)                       On output: Eigenvectors of a
!>                                                 Distribution is like in Scalapack.
!>                                                 Must be always dimensioned to the full size (corresponding to (na,na))
!>                                                 even if only a part of the eigenvalues is needed.
!>
!>  \param ldq                                     Leading dimension of q
!>
!>  \param nblk                                    blocksize of cyclic distribution, must be the same in both directions!
!>
!>  \param matrixCols                              local columns of matrix a and q
!>
!>  \param mpi_comm_rows                           MPI communicator for rows
!>  \param mpi_comm_cols                           MPI communicator for columns
!>  \param mpi_comm_all                            MPI communicator for the total processor set
!>
!>  \param THIS_REAL_COMPLEX_KERNEL_API (optional) specify used ELPA 2stage
!>                                                 kernel via API (only evalulated if 2 stage solver is used_
!>
!>  \param method                                  choose whether to use ELPA 1stage or 2stage solver
!>                                                 possible values: "1stage" => use ELPA 1stage solver
!>                                                                  "2stage" => use ELPA 2stage solver
!>                                                                  "auto"   => (at the moment) use ELPA 2stage solver
!>
!>  \result success                                logical, false if error occured
!-------------------------------------------------------------------------------
    function elpa_solve_evp_complex(na, nev, a, lda, ev, q, ldq, nblk,         &
                                    matrixCols, mpi_comm_rows, mpi_comm_cols,  &
                                    mpi_comm_all, THIS_COMPLEX_ELPA_KERNEL_API,&
                                    method) result(success)
      use iso_c_binding
      use elpa_utilities

      implicit none
      integer(kind=c_int), intent(in)           :: na, nev, lda, ldq, matrixCols, mpi_comm_rows, &
                                                   mpi_comm_cols, mpi_comm_all
      integer(kind=c_int), intent(in)           :: nblk
      real(kind=c_double), intent(inout)        :: ev(na)
#ifdef USE_ASSUMED_SIZE
      complex(kind=c_double), intent(inout)     :: a(lda,*), q(ldq,*)
#else
      complex(kind=c_double), intent(inout)     :: a(lda,matrixCols), q(ldq,matrixCols)
#endif
      integer(kind=c_int), intent(in), optional :: THIS_COMPLEX_ELPA_KERNEL_API
      character(*), intent(in), optional        :: method

      logical                                   :: useELPA1
      logical                                   :: success

      useELPA1 = .false.

      if (present(method)) then
        if (trim(method) .eq. "1stage" .or. trim(method) .eq. "1STAGE") then
          useELPA1 = .true.
        else if (trim(method) .eq. "2stage" .or. trim(method) .eq. "2STAGE") then
          useELPA1 = .false.
        else if (trim(method) .eq. "auto" .or. trim(method) .eq. "AUTO") then
          useELPA1 = .false.
        else
          write(error_unit,*) "Specified method not known! Using ELPA 2-stage"
          useELPA1 = .false.
        endif
      endif

      if (useELPA1) then
        success = solve_evp_complex_1stage(na, nev, a, lda, ev, q, ldq, nblk,                     &
                                        matrixCols, mpi_comm_rows, mpi_comm_cols)
      else
        success = solve_evp_complex_2stage(na, nev, a, lda, ev, q, ldq, nblk,                     &
                                        matrixCols, mpi_comm_rows, mpi_comm_cols,                 &
                                        mpi_comm_all,                                             &
                                        THIS_COMPLEX_ELPA_KERNEL_API = THIS_COMPLEX_ELPA_KERNEL_API)
      endif

    end function elpa_solve_evp_complex
end module elpa

