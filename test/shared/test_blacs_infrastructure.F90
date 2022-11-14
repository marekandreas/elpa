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

#ifdef HAVE_64BIT_INTEGER_MATH_SUPPORT
#define TEST_INT_TYPE integer(kind=c_int64_t)
#define INT_TYPE c_int64_t
#define TEST_C_INT_TYPE_PTR long int*
#define TEST_C_INT_TYPE long int
#else
#define TEST_INT_TYPE integer(kind=c_int32_t)
#define INT_TYPE c_int32_t
#define TEST_C_INT_TYPE_PTR int*
#define TEST_C_INT_TYPE int
#endif
#ifdef HAVE_64BIT_INTEGER_MPI_SUPPORT
#define TEST_INT_MPI_TYPE integer(kind=c_int64_t)
#define INT_MPI_TYPE c_int64_t
#define TEST_C_INT_MPI_TYPE_PTR long int*
#define TEST_C_INT_MPI_TYPE long int
#else
#define TEST_INT_MPI_TYPE integer(kind=c_int32_t)
#define INT_MPI_TYPE c_int32_t
#define TEST_C_INT_MPI_TYPE_PTR int*
#define TEST_C_INT_MPI_TYPE int
#endif

module test_blacs_infrastructure

  contains

    !c> void set_up_blacsgrid_f(TEST_C_INT_TYPE mpi_comm_parent, TEST_C_INT_TYPE np_rows, 
    !c>                         TEST_C_INT_TYPE np_cols, char layout,
    !c>                         TEST_C_INT_TYPE_PTR my_blacs_ctxt, TEST_C_INT_TYPE_PTR my_prow, 
    !c>                         TEST_C_INT_TYPE_PTR my_pcol);
    subroutine set_up_blacsgrid(mpi_comm_parent, np_rows, np_cols, layout, &
                                my_blacs_ctxt, my_prow, my_pcol) bind(C, name="set_up_blacsgrid_f")

      use precision_for_tests
      use test_util
      use iso_c_binding

      implicit none
      TEST_INT_TYPE, intent(in), value          :: mpi_comm_parent, np_rows, np_cols
#ifdef SXAURORA
      character(len=1), intent(in)              :: layout
#else
      character(kind=c_char), intent(in), value :: layout
#endif
      TEST_INT_TYPE, intent(out)                :: my_blacs_ctxt, my_prow, my_pcol

#ifdef WITH_MPI
      TEST_INT_TYPE :: np_rows_, np_cols_
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
                                       na_cols, sc_desc, my_blacs_ctxt, info, &
                                       blacs_ok)

      use elpa_utilities, only : error_unit
      use test_util
      use precision_for_tests
      use tests_scalapack_interfaces
      implicit none

      TEST_INT_TYPE, intent(in)    :: na, nblk, my_prow, my_pcol, np_rows,   &
                                       np_cols, &
                                       my_blacs_ctxt
      TEST_INT_TYPE, intent(inout) :: info
      TEST_INT_TYPE, intent(out)   :: na_rows, na_cols, sc_desc(1:9), blacs_ok

#ifdef WITH_MPI
      TEST_INT_MPI_TYPE            :: mpierr

      sc_desc(:) = 0
      blacs_ok = 0
      ! determine the neccessary size of the distributed matrices,
      ! we use the scalapack tools routine NUMROC

      na_rows = numroc(na, nblk, my_prow, 0_BLAS_KIND, np_rows)
      na_cols = numroc(na, nblk, my_pcol, 0_BLAS_KIND, np_cols)

      ! set up the scalapack descriptor for the checks below
      ! For ELPA the following restrictions hold:
      ! - block sizes in both directions must be identical (args 4 a. 5)
      ! - first row and column of the distributed matrix must be on
      !   row/col 0/0 (arg 6 and 7)

      call descinit(sc_desc, na, na, nblk, nblk, 0_BLAS_KIND, 0_BLAS_KIND, &
                    my_blacs_ctxt, na_rows, info)

      blacs_ok = 1
      if (info .ne. 0) then
        write(error_unit,*) 'Error in BLACS descinit! info=',info
        write(error_unit,*) 'Most likely this happend since you want to use'
        write(error_unit,*) 'more MPI tasks than are possible for your'
        write(error_unit,*) 'problem size (matrix size and blocksize)!'
        write(error_unit,*) 'The blacsgrid can not be set up properly'
        write(error_unit,*) 'Try reducing the number of MPI tasks...'
        write(error_unit,*) 'For ELPA it is mandatory that the first row/col'
        write(error_unit,*) 'of the distributed matrix must be on 0/0'
        write(error_unit,*) 'arguments 6 & 7 of descinit'
        !call MPI_ABORT(int(mpi_comm_world,kind=MPI_KIND), 1_MPI_KIND, mpierr)
        blacs_ok = 0
        return
      endif
#else /* WITH_MPI */
      na_rows = na
      na_cols = na
      blacs_ok = 1
#endif /* WITH_MPI */

    end subroutine

    !c> void set_up_blacs_descriptor_f(TEST_C_INT_TYPE na, TEST_C_INT_TYPE nblk, 
    !c>                                TEST_C_INT_TYPE my_prow, TEST_C_INT_TYPE my_pcol,
    !c>                                TEST_C_INT_TYPE np_rows, TEST_C_INT_TYPE np_cols,
    !c>                                TEST_C_INT_TYPE_PTR na_rows, TEST_C_INT_TYPE_PTR na_cols,
    !c>                                TEST_C_INT_TYPE sc_desc[9],
    !c>                                TEST_C_INT_TYPE my_blacs_ctxt,
    !c>                                TEST_C_INT_TYPE_PTR info, TEST_C_INT_TYPE_PTR blacs_ok);
    
    !c> #ifdef __cplusplus
    !c> }
    !c> #endif
    
    subroutine set_up_blacs_descriptor_f(na, nblk, my_prow, my_pcol, &
                                         np_rows, np_cols, na_rows,  &
                                         na_cols, sc_desc,           &
                                         my_blacs_ctxt, info, blacs_ok)        &
                                         bind(C, name="set_up_blacs_descriptor_f")

      use iso_c_binding
      implicit none


      TEST_INT_TYPE, value :: na, nblk, my_prow, my_pcol, np_rows, &
                                    np_cols, my_blacs_ctxt
      TEST_INT_TYPE        :: na_rows, na_cols, info, sc_desc(1:9), blacs_ok

      call set_up_blacs_descriptor(na, nblk, my_prow, my_pcol, &
                                   np_rows, np_cols, na_rows,  &
                                   na_cols, sc_desc, my_blacs_ctxt, info, blacs_ok)


    end subroutine

    
   function index_l2g(idx_loc, nblk, iproc, nprocs) result(indexl2g)
     use precision_for_tests
     implicit none
     TEST_INT_TYPE :: indexl2g
     TEST_INT_TYPE :: idx_loc, nblk, iproc, nprocs
     indexl2g = nprocs * nblk * ((idx_loc-1) / nblk) + mod(idx_loc-1,nblk) + mod(nprocs+iproc, nprocs)*nblk + 1
     return
   end function

end module
