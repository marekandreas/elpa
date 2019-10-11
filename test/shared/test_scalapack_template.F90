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

  ! compute all eigenvectors
  subroutine solve_p&
      &BLAS_CHAR_AND_SY_OR_HE&
      &evd(na, a, sc_desc, ev, z)
    implicit none
#include "./test_precision_kinds.F90"
    integer(kind=ik), intent(in)     :: na
    MATH_DATATYPE(kind=rck), intent(in)       :: a(:,:)
    MATH_DATATYPE(kind=rck), intent(inout)    :: z(:,:)
    real(kind=rk), intent(inout)    :: ev(:)
    integer(kind=ik), intent(in)     :: sc_desc(:)
    integer(kind=ik)                 :: info, lwork, liwork, lrwork
    MATH_DATATYPE(kind=rck), allocatable      :: work(:)
    real(kind=rk), allocatable       :: rwork(:)
    integer, allocatable             :: iwork(:)

    allocate(work(1), iwork(1), rwork(1))

    ! query for required workspace
#ifdef REALCASE
    call p&
         &BLAS_CHAR&
         &syevd('V', 'U', na, a, 1, 1, sc_desc, ev, z, 1, 1, sc_desc, work, -1, iwork, -1, info)
#endif
#ifdef COMPLEXCASE
    call p&
         &BLAS_CHAR&
         &heevd('V', 'U', na, a, 1, 1, sc_desc, ev, z, 1, 1, sc_desc, work, -1, rwork, -1, iwork, -1, info)
#endif
    !  write(*,*) "computed sizes", lwork, liwork, "required sizes ", work(1), iwork(1)
    lwork = work(1)
    liwork = iwork(1)
    deallocate(work, iwork)
    allocate(work(lwork), stat = info)
    allocate(iwork(liwork), stat = info)
#ifdef COMPLEXCASE
    lrwork = rwork(1)
    deallocate(rwork)
    allocate(rwork(lrwork), stat = info)
#endif
    ! the actuall call to the method
#ifdef REALCASE
    call p&
         &BLAS_CHAR&
         &syevd('V', 'U', na, a, 1, 1, sc_desc, ev, z, 1, 1, sc_desc, work, lwork, iwork, liwork, info)
#endif
#ifdef COMPLEXCASE
    call p&
         &BLAS_CHAR&
         &heevd('V', 'U', na, a, 1, 1, sc_desc, ev, z, 1, 1, sc_desc, work, lwork, rwork, lrwork, iwork, liwork, info)
#endif

    deallocate(iwork, work, rwork)
  end subroutine


  ! compute part of eigenvectors
  subroutine solve_p&
      &BLAS_CHAR_AND_SY_OR_HE&
      &evr(na, a, sc_desc, nev, ev, z)
    implicit none
#include "./test_precision_kinds.F90"
    integer(kind=ik), intent(in)     :: na, nev
    MATH_DATATYPE(kind=rck), intent(in)       :: a(:,:)
    MATH_DATATYPE(kind=rck), intent(inout)    :: z(:,:)
    real(kind=rk), intent(inout)    :: ev(:)
    integer(kind=ik), intent(in)     :: sc_desc(:)
    integer(kind=ik)                 :: info, lwork, liwork, lrwork
    MATH_DATATYPE(kind=rck), allocatable      :: work(:)
    real(kind=rk), allocatable       :: rwork(:)
    integer, allocatable             :: iwork(:)
    integer(kind=ik)                 :: comp_eigenval, comp_eigenvec, smallest_ev_idx, largest_ev_idx

    allocate(work(1), iwork(1), rwork(1))
    smallest_ev_idx = 1
    largest_ev_idx = nev
    ! query for required workspace
#ifdef REALCASE
    call p&
         &BLAS_CHAR&
         &syevr('V', 'I', 'U', na, a, 1, 1, sc_desc, 0.0_rk, 0.0_rk, smallest_ev_idx, largest_ev_idx, &
                comp_eigenval, comp_eigenvec,  ev, z, 1, 1, sc_desc, work, -1, iwork, -1, info)
#endif
#ifdef COMPLEXCASE
    call p&
         &BLAS_CHAR&
         &heevr('V', 'I', 'U', na, a, 1, 1, sc_desc, 0.0_rk, 0.0_rk, smallest_ev_idx, largest_ev_idx, &
                comp_eigenval, comp_eigenvec,  ev, z, 1, 1, sc_desc, work, -1, rwork, -1, iwork, -1, info)
#endif
    !  write(*,*) "computed sizes", lwork, liwork, "required sizes ", work(1), iwork(1)
    lwork = work(1)
    liwork = iwork(1)
    deallocate(work, iwork)
    allocate(work(lwork), stat = info)
    allocate(iwork(liwork), stat = info)
#ifdef COMPLEXCASE
    lrwork = rwork(1)
    deallocate(rwork)
    allocate(rwork(lrwork), stat = info)
#endif
    ! the actuall call to the method
#ifdef REALCASE
    call p&
         &BLAS_CHAR&
         &syevr('V', 'I', 'U', na, a, 1, 1, sc_desc, 0.0_rk, 0.0_rk, smallest_ev_idx, largest_ev_idx, &
                comp_eigenval, comp_eigenvec,  ev, z, 1, 1, sc_desc, work, lwork, iwork, liwork, info)
#endif
#ifdef COMPLEXCASE
    call p&
         &BLAS_CHAR&
         &heevr('V', 'I', 'U', na, a, 1, 1, sc_desc, 0.0_rk, 0.0_rk, smallest_ev_idx, largest_ev_idx, &
                comp_eigenval, comp_eigenvec,  ev, z, 1, 1, sc_desc, work, lwork, rwork, lrwork, iwork, liwork, info)
#endif
    assert(comp_eigenval == nev)
    assert(comp_eigenvec == nev)
    deallocate(iwork, work, rwork)
  end subroutine

