#if 0
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
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
#endif

subroutine solve_secular_equation_&
&PRECISION&
&(obj, n, i, d, z, delta, rho, dlam)
!-------------------------------------------------------------------------------
! This routine solves the secular equation of a symmetric rank 1 modified
! diagonal matrix:
!
!    1. + rho*SUM(z(:)**2/(d(:)-x)) = 0
!
! It does the same as the LAPACK routine DLAED4 but it uses a bisection technique
! which is more robust (it always yields a solution) but also slower
! than the algorithm used in DLAED4.
!
! The same restictions than in DLAED4 hold, namely:
!
!   rho > 0   and   d(i+1) > d(i)
!
! but this routine will not terminate with error if these are not satisfied
! (it will normally converge to a pole in this case).
!
! The output in DELTA(j) is always (D(j) - lambda_I), even for the cases
! N=1 and N=2 which is not compatible with DLAED4.
! Thus this routine shouldn't be used for these cases as a simple replacement
! of DLAED4.
!
! The arguments are the same as in DLAED4 (with the exception of the INFO argument):
!
!
!  N      (input) INTEGER
!         The length of all arrays.
!
!  I      (input) INTEGER
!         The index of the eigenvalue to be computed.  1 <= I <= N.
!
!  D      (input) DOUBLE PRECISION array, dimension (N)
!         The original eigenvalues.  It is assumed that they are in
!         order, D(I) < D(J)  for I < J.
!
!  Z      (input) DOUBLE PRECISION array, dimension (N)
!         The components of the updating Vector.
!
!  DELTA  (output) DOUBLE PRECISION array, dimension (N)
!         DELTA contains (D(j) - lambda_I) in its  j-th component.
!         See remark above about DLAED4 compatibility!
!
!  RHO    (input) DOUBLE PRECISION
!         The scalar in the symmetric updating formula.
!
!  DLAM   (output) DOUBLE PRECISION
!         The computed lambda_I, the I-th updated eigenvalue.
!-------------------------------------------------------------------------------

  use precision
  use elpa_abstract_impl
  implicit none
#include "../../src/general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik)           :: n, i
  real(kind=rk)   :: d(n), z(n), delta(n), rho, dlam

  integer(kind=ik)           :: iter
  real(kind=rk)   :: a, b, x, y, dshift

  ! In order to obtain sufficient numerical accuracy we have to shift the problem
  ! either by d(i) or d(i+1), whichever is closer to the solution

  ! Upper and lower bound of the shifted solution interval are a and b

  call obj%timer%start("solve_secular_equation" // PRECISION_SUFFIX)
  if (i==n) then

   ! Special case: Last eigenvalue
   ! We shift always by d(n), lower bound is d(n),
   ! upper bound is determined by a guess:

   dshift = d(n)
   delta(:) = d(:) - dshift

   a = 0.0_rk ! delta(n)
   b = rho*SUM(z(:)**2) + 1.0_rk ! rho*SUM(z(:)**2) is the lower bound for the guess
  else

    ! Other eigenvalues: lower bound is d(i), upper bound is d(i+1)
    ! We check the sign of the function in the midpoint of the interval
    ! in order to determine if eigenvalue is more close to d(i) or d(i+1)
    x = 0.5_rk*(d(i)+d(i+1))
    y = 1.0_rk + rho*SUM(z(:)**2/(d(:)-x))
    if (y>0) then
      ! solution is next to d(i)
      dshift = d(i)
    else
      ! solution is next to d(i+1)
      dshift = d(i+1)
    endif

    delta(:) = d(:) - dshift
    a = delta(i)
    b = delta(i+1)

  endif

  ! Bisection:

  do iter=1,200

    ! Interval subdivision
    x = 0.5_rk*(a+b)
    if (x==a .or. x==b) exit   ! No further interval subdivisions possible
#ifdef DOUBLE_PRECISION_REAL
    if (abs(x) < 1.e-200_rk8) exit ! x next to pole
#else
    if (abs(x) < 1.e-20_rk4) exit ! x next to pole
#endif
    ! evaluate value at x

    y = 1. + rho*SUM(z(:)**2/(delta(:)-x))

    if (y==0) then
      ! found exact solution
      exit
    elseif (y>0) then
      b = x
    else
      a = x
    endif

  enddo

  ! Solution:

  dlam = x + dshift
  delta(:) = delta(:) - x
  call  obj%timer%stop("solve_secular_equation" // PRECISION_SUFFIX)

end subroutine solve_secular_equation_&
    &PRECISION

