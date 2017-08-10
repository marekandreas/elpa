! (c) Copyright Pavel Kus, 2017, MPCDF
!
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

#include "config-f90.h"

module test_scalapack
  use test_util

  interface solve_scalapack_all
    module procedure solve_pdsyevd
  end interface  

contains

  subroutine solve_pdsyevd(na, a, sc_desc, ev, z)
    implicit none 
    integer(kind=ik), intent(in)     :: na
    real(kind=rk8), intent(in)       :: a(:,:)
    real(kind=rk8), intent(inout)    :: z(:,:), ev(:)
    integer(kind=ik), intent(in)     :: sc_desc(:)
    integer(kind=ik)                 :: info, lwork, liwork
    real(kind=rk8), allocatable      :: work(:)
    integer, allocatable             :: iwork(:) 
  
    allocate(work(1), iwork(1))

    ! query for required workspace
    call pdsyevd('V', 'U', na, a, 1, 1, sc_desc, ev, z, 1, 1, sc_desc, work, -1, iwork, -1, info)
    !  write(*,*) "computed sizes", lwork, liwork, "required sizes ", work(1), iwork(1)
    lwork = work(1)
    liwork = iwork(1)
      
    deallocate(work, iwork)
    allocate(work(lwork), stat = info)
    allocate(iwork(liwork), stat = info)
    
    ! the actuall call to the method
    call pdsyevd('V', 'U', na, a, 1, 1, sc_desc, ev, z, 1, 1, sc_desc, work, lwork, iwork, liwork, info)

    deallocate(iwork)
    deallocate(work)
  end subroutine

end module
