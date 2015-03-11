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
module from_c
  implicit none

  public

  interface
    integer(kind=c_int) function elpa1_real_c(na, nev, ncols, a, lda, ev, q, ldq,         &
                                       nblk, mpi_comm_rows, mpi_comm_cols ) &
                                       bind(C, name="call_elpa1_real_solver_from_c")

      use iso_c_binding
      implicit none

      integer(kind=c_int), value :: na, nev, ncols, lda, ldq, nblk, mpi_comm_rows, mpi_comm_cols
      real(kind=c_double)        :: a(1:lda,1:ncols), ev(1:na), q(1:ldq,1:ncols)
    end function elpa1_real_c


  end interface

  interface
    integer(kind=c_int) function elpa_get_comm_c(mpi_comm_world, my_prow, my_pcol, &
                                                 mpi_comm_rows, mpi_comm_cols)     &
                                                 bind(C, name="call_elpa_get_comm_from_c")
      use iso_c_binding
      implicit none
      integer(kind=c_int), value :: mpi_comm_world, my_prow, my_pcol
      integer(kind=c_int)        :: mpi_comm_rows, mpi_comm_cols

    end function
  end interface

  contains

  function solve_elpa1_real_call_from_c(na, nev, ncols, a, lda, ev, q, ldq,         &
                      nblk, mpi_comm_rows, mpi_comm_cols ) &
                      result(success)

    use iso_c_binding
    implicit none

    integer :: na, nev, ncols, lda, ldq, nblk, mpi_comm_rows, mpi_comm_cols
    logical :: success
    integer :: successC

    real*8  :: a(1:lda,1:ncols), ev(1:na), q(1:ldq,1:ncols)

    successC = elpa1_real_c(na, nev, ncols, a, lda, ev, q, ldq, nblk, &
                            mpi_comm_rows, mpi_comm_cols)

    if (successC .eq. 1) then
      success = .true.
    else
      success = .false.
    endif

  end function

  function call_elpa_get_comm_from_c(mpi_comm_world, my_prow, my_pcol, &
                                     mpi_comm_rows, mpi_comm_cols) result(mpierr)

      use iso_c_binding
      implicit none

      integer :: mpierr
      integer :: mpi_comm_world, my_prow, my_pcol, &
                 mpi_comm_rows, mpi_comm_cols

      mpierr = elpa_get_comm_c(mpi_comm_world, my_prow, my_pcol, &
                                    mpi_comm_rows, mpi_comm_cols)
  end function
end module from_c
