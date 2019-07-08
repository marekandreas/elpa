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
!
#include "config-f90.h"
#define WITH_MPI 1
module test_blacs_infrastructure
    use iso_c_binding
    use mpi
    integer, parameter :: ik  = C_INT32_T

  contains

    !c> void set_up_blacsgrid_f(int mpi_comm_parent, int np_rows, int np_cols, char layout,
    !c>                         int* my_blacs_ctxt, int *my_prow, int *my_pcol);
    subroutine set_up_blacsgrid(mpi_comm_parent, np_rows, np_cols, layout, &
                                my_blacs_ctxt, my_prow, my_pcol) bind(C, name="set_up_blacsgrid_f")

      !use test_util

      implicit none
      integer(kind=c_int), intent(in), value  :: mpi_comm_parent, np_rows, np_cols
      character(len=1), intent(in), value     :: layout
      integer(kind=c_int), intent(out)        :: my_blacs_ctxt, my_prow, my_pcol

#ifdef WITH_MPI
      integer :: np_rows_, np_cols_
#endif

      if (layout /= 'R' .and. layout /= 'C') then
        print *, "layout must be 'R' or 'C'"
        stop 1
      end if

      my_blacs_ctxt = mpi_comm_parent
#ifdef WITH_MPI
      call BLACS_Gridinit(my_blacs_ctxt, layout, np_rows, np_cols)
      call BLACS_Gridinfo(my_blacs_ctxt, np_rows_, np_cols_, my_prow, my_pcol)
      if (np_rows /= np_rows_) then
        print *, "BLACS_Gridinfo returned different values for np_rows as set by BLACS_Gridinit"
        stop 1
      endif
      if (np_cols /= np_cols_) then
        print *, "BLACS_Gridinfo returned different values for np_cols as set by BLACS_Gridinit"
        stop 1
      endif
#else
      my_prow = 0
      my_pcol = 0
#endif
    end subroutine

    subroutine set_up_blacs_descriptor(na, nblk, my_prow, my_pcol, &
                                       np_rows, np_cols, na_rows,  &
                                       na_cols, sc_desc, my_blacs_ctxt, info)

      !use elpa_utilities, only : error_unit
      !use test_util
      implicit none

      integer(kind=ik), intent(in)  :: na, nblk, my_prow, my_pcol, np_rows,   &
                                       np_cols, &
                                       my_blacs_ctxt, info
      integer(kind=ik), intent(out)  :: na_rows, na_cols, sc_desc(1:9)

      integer(kind=ik), parameter    :: error_unit=0
#ifdef WITH_MPI
      integer(kind=ik), external       :: numroc
      integer(kind=ik)                 :: mpierr

      sc_desc(:) = 0
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

      if (info .ne. 0) then
        write(error_unit,*) 'Error in BLACS descinit! info=',info
        write(error_unit,*) 'Most likely this happend since you want to use'
        write(error_unit,*) 'more MPI tasks than are possible for your'
        write(error_unit,*) 'problem size (matrix size and blocksize)!'
        write(error_unit,*) 'The blacsgrid can not be set up properly'
        write(error_unit,*) 'Try reducing the number of MPI tasks...'
        call MPI_ABORT(mpi_comm_world, 1, mpierr)
      endif
#else /* WITH_MPI */
      na_rows = na
      na_cols = na
#endif /* WITH_MPI */

    end subroutine

    !c> void set_up_blacs_descriptor_f(int na, int nblk, int my_prow, int my_pcol,
    !c>                                int np_rows, int np_cols,
    !c>                                int *na_rows, int *na_cols,
    !c>                                int sc_desc[9],
    !c>                                int my_blacs_ctxt,
    !c>                                int *info);
    subroutine set_up_blacs_descriptor_f(na, nblk, my_prow, my_pcol, &
                                         np_rows, np_cols, na_rows,  &
                                         na_cols, sc_desc,           &
                                         my_blacs_ctxt, info)        &
                                         bind(C, name="set_up_blacs_descriptor_f")

      use iso_c_binding
      implicit none


      integer(kind=c_int), value :: na, nblk, my_prow, my_pcol, np_rows, &
                                    np_cols, my_blacs_ctxt
      integer(kind=c_int)        :: na_rows, na_cols, info, sc_desc(1:9)

      call set_up_blacs_descriptor(na, nblk, my_prow, my_pcol, &
                                   np_rows, np_cols, na_rows,  &
                                   na_cols, sc_desc, my_blacs_ctxt, info)


    end subroutine

    integer function index_l2g(idx_loc, nblk, iproc, nprocs)
     index_l2g = nprocs * nblk * ((idx_loc-1) / nblk) + mod(idx_loc-1,nblk) + mod(nprocs+iproc, nprocs)*nblk + 1
     return
   end function

end module
