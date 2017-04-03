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

#define stringify_(x) "x"
#define stringify(x) stringify_(x)
#define assert(x) call x_assert(x, stringify(x), __FILE__, __LINE__)

module assert
  implicit none
  contains
    subroutine x_assert(condition, condition_string, file, line)
      use elpa_utilities, only : error_unit
      logical, intent(in) :: condition
      character(len=*), intent(in) :: condition_string
      character(len=*), intent(in) :: file
      integer, intent(in) :: line

      if (.not. condition) then
        write(error_unit,'(a,i0)') "Assertion failed:" // condition_string // " at " // file // ":", line
      end if
    end subroutine
end module

program test_interface
   use precision
   use assert
   use mod_setup_mpi
   use elpa_mpi
   use elpa_type
   use mod_blacs_infrastructure

   implicit none

   ! matrix dimensions
   integer :: na, nev, nblk

   ! mpi
   integer :: myid, nprocs
   integer :: na_cols, na_rows  ! local matrix size
   integer :: np_cols, np_rows  ! number of MPI processes per column/row
   integer :: my_prow, my_pcol  ! local MPI task position (my_prow, my_pcol) in the grid (0..np_cols -1, 0..np_rows -1)
   integer :: mpierr

   ! blacs
   integer :: my_blacs_ctxt, sc_desc(9), info

   ! The Matrix
   real(kind=C_DOUBLE), allocatable :: a(:,:)
   ! eigenvectors
   real(kind=C_DOUBLE), allocatable :: z(:,:)
   ! eigenvalues
   real(kind=C_DOUBLE), allocatable :: ev(:)

   integer :: success

   integer :: solver
   integer(kind=C_INT) :: qr


   type(elpa_t) :: e

   call setup_mpi(myid, nprocs)

   na = 100
   nblk = 16
   nev = 25

   !-------------------------------------------------------------------------------
   ! Selection of number of processor rows/columns
   ! We try to set up the grid square-like, i.e. start the search for possible
   ! divisors of nprocs with a number next to the square root of nprocs
   ! and decrement it until a divisor is found.

   do np_cols = NINT(SQRT(REAL(nprocs))),2,-1
      if(mod(nprocs,np_cols) == 0 ) exit
   enddo
   ! at the end of the above loop, nprocs is always divisible by np_cols

   np_rows = nprocs/np_cols

   my_prow = mod(myid, np_cols)
   my_pcol = myid / np_cols

   call set_up_blacs_descriptor(na, nblk, my_prow, my_pcol, np_rows, np_cols, &
                                na_rows, na_cols, sc_desc, my_blacs_ctxt, info)

   allocate(a (na_rows,na_cols))
   allocate(z (na_rows,na_cols))
   allocate(ev(na))

   a(:,:) = 0.0
   z(:,:) = 0.0
   ev(:) = 0.0

   if (elpa_init(20170403) /= ELPA_OK) then
     error stop "ELPA API version not supported"
   endif

   e = elpa_create(na, nev, na_rows, na_cols, nblk, mpi_comm_world, my_prow, my_pcol, success)
   assert(success == ELPA_OK)

   qr = e%get("qr", success)
   print *, "qr =", qr
   assert(success == ELPA_OK)

   solver = e%get("solver", success)
   print *, "solver =", solver
   assert(success == ELPA_OK)

   call e%set("solver", ELPA_SOLVER_2STAGE, success)
   assert(success == ELPA_OK)

   call e%set("real_kernel", ELPA_2STAGE_REAL_GENERIC, success)
   assert(success == ELPA_OK)

   call e%set("complex_kernel", ELPA_2STAGE_COMPLEX_GENERIC, success)
   assert(success == ELPA_OK)

   call e%solve(a, ev, z, success)
   assert(success == ELPA_OK)

   call e%destroy()

   call elpa_uninit()

   deallocate(a)
   deallocate(z)
   deallocate(ev)

#ifdef WITH_MPI
   call mpi_finalize(mpierr)
#endif

end program
