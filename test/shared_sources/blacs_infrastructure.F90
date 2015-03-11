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
module mod_blacs_infrastructure

  contains

    subroutine set_up_blacsgrid(mpi_comm_world, my_blacs_ctxt, np_rows, &
                                np_cols, nprow, npcol, my_prow, my_pcol)

      implicit none
      integer, intent(in)     :: mpi_comm_world
      integer, intent(inout)  :: my_blacs_ctxt, np_rows, &
                                 np_cols, nprow, npcol, my_prow, my_pcol

      my_blacs_ctxt = mpi_comm_world
      call BLACS_Gridinit(my_blacs_ctxt, 'C', np_rows, np_cols)
      call BLACS_Gridinfo(my_blacs_ctxt, nprow, npcol, my_prow, my_pcol)
    end subroutine

    subroutine set_up_blacsgrid_wrapper(mpi_comm_world, my_blacs_ctxt, np_rows, &
                                np_cols, nprow, npcol, my_prow, my_pcol)        &
                                bind(C, name="set_up_blacsgrid_from_fortran")
      use iso_c_binding
      implicit none
      integer(kind=c_int), value :: mpi_comm_world
      integer(kind=c_int)        :: my_blacs_ctxt, np_rows, &
                                    np_cols, nprow, npcol, my_prow, my_pcol

      call set_up_blacsgrid(mpi_comm_world, my_blacs_ctxt, np_rows, &
                                np_cols, nprow, npcol, my_prow, my_pcol)
    end subroutine

    subroutine set_up_blacs_descriptor(na, nblk, my_prow, my_pcol, &
                                       np_rows, np_cols, na_rows,  &
                                       na_cols, sc_desc, my_blacs_ctxt, info)

      implicit none

      integer, intent(inout)  :: na, nblk, my_prow, my_pcol, np_rows,   &
                                 np_cols, na_rows, na_cols, sc_desc(1:9), &
                                 my_blacs_ctxt, info

      integer, external       :: numroc

      ! determine the neccessary size of the distributed matrices,
      ! we use the scalapack tools routine NUMROC

      na_rows = numroc(na, nblk, my_prow, 0, np_rows)
      na_cols = numroc(na, nblk, my_pcol, 0, np_cols)

      ! set up the scalapack descriptor for the checks below
      ! For ELPA the following restrictions hold:
      ! - block sizes in both directions must be identical (args 4 a. 5)
      ! - first row and column of the distributed matrix must be on
      !   row/col 0/0 (arg 6 and 7)

      call descinit(sc_desc, na, na, nblk, nblk, 0, 0, my_blacs_ctxt, na_rows, info)
    end subroutine

    subroutine set_up_blacs_descriptor_wrapper(na, nblk, my_prow, my_pcol, &
                                               np_rows, np_cols, na_rows,  &
                                               na_cols, sc_desc,           &
                                               my_blacs_ctxt, info)        &
                                               bind(C, name="set_up_blacs_descriptor_from_fortran")

      use iso_c_binding
      implicit none


      integer(kind=c_int), value :: na, nblk, my_prow, my_pcol, np_rows, &
                                    np_cols, my_blacs_ctxt
      integer(kind=c_int)        :: na_rows, na_cols, info, sc_desc(1:9)

      call set_up_blacs_descriptor(na, nblk, my_prow, my_pcol, &
                                   np_rows, np_cols, na_rows,  &
                                   na_cols, sc_desc, my_blacs_ctxt, info)


    end subroutine

end module
