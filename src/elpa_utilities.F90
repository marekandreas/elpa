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
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
! Author: Andreas Marek, MPCDF

#include "config-f90.h"

module ELPA_utilities

#ifdef HAVE_ISO_FORTRAN_ENV
  use iso_fortran_env, only : error_unit
#endif
  implicit none

  private ! By default, all routines contained are private

  public :: debug_messages_via_environment_variable, error_unit
  public :: check_alloc, check_alloc_CUDA_f, check_memcpy_CUDA_f, check_dealloc_CUDA_f
  public :: map_global_array_index_to_local_index
  public :: pcol, prow
  public :: local_index                ! Get local index of a block cyclic distributed matrix
  public :: least_common_multiple      ! Get least common multiple

#ifndef HAVE_ISO_FORTRAN_ENV
  integer, parameter :: error_unit = 0
#endif


  !******
  contains

   function debug_messages_via_environment_variable() result(isSet)
#ifdef HAVE_DETAILED_TIMINGS
     use timings
#endif
     use precision
     implicit none
     logical              :: isSet
     CHARACTER(len=255)   :: ELPA_DEBUG_MESSAGES

#ifdef HAVE_DETAILED_TIMINGS
     call timer%start("debug_messages_via_environment_variable")
#endif

     isSet = .false.

#if defined(HAVE_ENVIRONMENT_CHECKING)
     call get_environment_variable("ELPA_DEBUG_MESSAGES",ELPA_DEBUG_MESSAGES)
#endif
     if (trim(ELPA_DEBUG_MESSAGES) .eq. "yes") then
       isSet = .true.
     endif
     if (trim(ELPA_DEBUG_MESSAGES) .eq. "no") then
       isSet = .true.
     endif

#ifdef HAVE_DETAILED_TIMINGS
     call timer%stop("debug_messages_via_environment_variable")
#endif

   end function debug_messages_via_environment_variable

!-------------------------------------------------------------------------------

  !Processor col for global col number
  pure function pcol(global_col, nblk, np_cols) result(local_col)
    use precision
    implicit none
    integer(kind=ik), intent(in) :: global_col, nblk, np_cols
    integer(kind=ik)             :: local_col
    local_col = MOD((global_col-1)/nblk,np_cols)
  end function

!-------------------------------------------------------------------------------

  !Processor row for global row number
  pure function prow(global_row, nblk, np_rows) result(local_row)
    use precision
    implicit none
    integer(kind=ik), intent(in) :: global_row, nblk, np_rows
    integer(kind=ik)             :: local_row
    local_row = MOD((global_row-1)/nblk,np_rows)
  end function

!-------------------------------------------------------------------------------

 function map_global_array_index_to_local_index(iGLobal, jGlobal, iLocal, jLocal , nblk, np_rows, np_cols, my_prow, my_pcol) &
   result(possible)
   use precision

   implicit none

   integer(kind=ik)              :: pi, pj, li, lj, xi, xj
   integer(kind=ik), intent(in)  :: iGlobal, jGlobal, nblk, np_rows, np_cols, my_prow, my_pcol
   integer(kind=ik), intent(out) :: iLocal, jLocal
   logical                       :: possible

   possible = .true.
   iLocal = 0
   jLocal = 0

   pi = prow(iGlobal, nblk, np_rows)

   if (my_prow .ne. pi) then
     possible = .false.
     return
   endif

   pj = pcol(jGlobal, nblk, np_cols)

   if (my_pcol .ne. pj) then
     possible = .false.
     return
   endif
   li = (iGlobal-1)/(np_rows*nblk) ! block number for rows
   lj = (jGlobal-1)/(np_cols*nblk) ! block number for columns

   xi = mod( (iGlobal-1),nblk)+1   ! offset in block li
   xj = mod( (jGlobal-1),nblk)+1   ! offset in block lj

   iLocal = li * nblk + xi
   jLocal = lj * nblk + xj

 end function

 integer function local_index(idx, my_proc, num_procs, nblk, iflag)

!-------------------------------------------------------------------------------
!  local_index: returns the local index for a given global index
!               If the global index has no local index on the
!               processor my_proc behaviour is defined by iflag
!
!  Parameters
!
!  idx         Global index
!
!  my_proc     Processor row/column for which to calculate the local index
!
!  num_procs   Total number of processors along row/column
!
!  nblk        Blocksize
!
!  iflag       Controls the behaviour if idx is not on local processor
!              iflag< 0 : Return last local index before that row/col
!              iflag==0 : Return 0
!              iflag> 0 : Return next local index after that row/col
!-------------------------------------------------------------------------------
    use precision
    implicit none

    integer(kind=ik) :: idx, my_proc, num_procs, nblk, iflag

    integer(kind=ik) :: iblk

    iblk = (idx-1)/nblk  ! global block number, 0 based

    if (mod(iblk,num_procs) == my_proc) then

    ! block is local, always return local row/col number

    local_index = (iblk/num_procs)*nblk + mod(idx-1,nblk) + 1

    else

    ! non local block

    if (iflag == 0) then

        local_index = 0

    else

        local_index = (iblk/num_procs)*nblk

        if (mod(iblk,num_procs) > my_proc) local_index = local_index + nblk

        if (iflag>0) local_index = local_index + 1
    endif
    endif

 end function local_index

 integer function least_common_multiple(a, b)

    ! Returns the least common multiple of a and b
    ! There may be more efficient ways to do this, we use the most simple approach
    use precision
    implicit none
    integer(kind=ik), intent(in) :: a, b

    do least_common_multiple = a, a*(b-1), a
    if(mod(least_common_multiple,b)==0) exit
    enddo
    ! if the loop is left regularly, least_common_multiple = a*b

 end function least_common_multiple

 subroutine check_alloc(function_name, variable_name, istat, errorMessage)
    use precision

    implicit none

    character(len=*), intent(in)    :: function_name
    character(len=*), intent(in)    :: variable_name
    integer(kind=ik), intent(in)    :: istat
    character(len=*), intent(in)    :: errorMessage

    if (istat .ne. 0) then
      print *, function_name, ": error when allocating ", variable_name, " ", errorMessage
      stop
    endif
 end subroutine

 subroutine check_alloc_CUDA_f(file_name, line, successCUDA)
    use precision

    implicit none

    character(len=*), intent(in)    :: file_name
    integer(kind=ik), intent(in)    :: line
    logical                         :: successCUDA

    if (.not.(successCUDA)) then
      print *, file_name, ":", line,  " error in cuda_malloc when allocating "
      stop
    endif
 end subroutine

 subroutine check_dealloc_CUDA_f(file_name, line, successCUDA)
    use precision

    implicit none

    character(len=*), intent(in)    :: file_name
    integer(kind=ik), intent(in)    :: line
    logical                         :: successCUDA

    if (.not.(successCUDA)) then
      print *, file_name, ":", line,  " error in cuda_free when deallocating "
      stop
    endif
 end subroutine

 subroutine check_memcpy_CUDA_f(file_name, line, successCUDA)
    use precision

    implicit none

    character(len=*), intent(in)    :: file_name
    integer(kind=ik), intent(in)    :: line
    logical                         :: successCUDA

    if (.not.(successCUDA)) then
      print *, file_name, ":", line,  " error in cuda_memcpy when copying "
      stop
    endif
 end subroutine

end module ELPA_utilities
