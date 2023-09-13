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
!
! ELPA1 -- Faster replacements for ScaLAPACK symmetric eigenvalue routines
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
#endif

#include "../general/sanity.F90"


#if REALCASE == 1
subroutine hh_transform_real_&
#endif
#if COMPLEXCASE == 1
subroutine hh_transform_complex_&
#endif
&PRECISION &
(obj, alpha, xnorm_sq, xf, tau, wantDebug)
#if REALCASE  == 1
  ! Similar to LAPACK routine DLARFP, but uses ||x||**2 instead of x(:)
#endif
#if COMPLEXCASE == 1
  ! Similar to LAPACK routine ZLARFP, but uses ||x||**2 instead of x(:)
#endif
  ! and returns the factor xf by which x has to be scaled.
  ! It also hasn't the special handling for numbers < 1.d-300 or > 1.d150
  ! since this would be expensive for the parallel implementation.
  use precision
  use elpa_abstract_impl
  implicit none
#include "../general/precision_kinds.F90"
  class(elpa_abstract_impl_t), intent(inout)    :: obj
  logical, intent(in)                           :: wantDebug
#if REALCASE == 1
  real(kind=rk), intent(inout)       :: alpha
#endif
#if COMPLEXCASE == 1
  complex(kind=ck), intent(inout) :: alpha
#endif
  real(kind=rk), intent(in)          :: xnorm_sq
#if REALCASE == 1
  real(kind=rk), intent(out)         :: xf, tau
#endif
#if COMPLEXCASE == 1
  complex(kind=ck), intent(out)   :: xf, tau
  real(kind=rk)                      :: ALPHR, ALPHI
#endif

  real(kind=rk)                      :: BETA

  if (wantDebug) call obj%timer%start("hh_transform_&
                   &MATH_DATATYPE&
     	      &" // &
                   &PRECISION_SUFFIX )

#if COMPLEXCASE == 1
  ALPHR = real( ALPHA, kind=rk )
  ALPHI = PRECISION_IMAG( ALPHA )
#endif

#if REALCASE == 1
  if ( XNORM_SQ==0.0_rk ) then
#endif
#if COMPLEXCASE == 1
  if ( XNORM_SQ==0.0_rk .AND. ALPHI==0.0_rk ) then
#endif

#if REALCASE == 1
    if ( ALPHA>=0.0_rk ) then
#endif
#if COMPLEXCASE == 1
    if ( ALPHR>=0.0_rk ) then
#endif
      TAU = 0.0_rk
    else
      TAU = 2.0_rk
      ALPHA = -ALPHA
    endif
    XF = 0.0_rk

  else

#if REALCASE == 1
    BETA = SIGN( SQRT( ALPHA**2 + XNORM_SQ ), ALPHA )
#endif
#if COMPLEXCASE == 1
    BETA = SIGN( SQRT( ALPHR**2 + ALPHI**2 + XNORM_SQ ), ALPHR )
#endif
    ALPHA = ALPHA + BETA
    IF ( BETA<0 ) THEN
      BETA = -BETA
      TAU  = -ALPHA / BETA
    ELSE
#if REALCASE == 1
      ALPHA = XNORM_SQ / ALPHA
#endif
#if COMPLEXCASE == 1
      ALPHR = ALPHI * (ALPHI/real( ALPHA , kind=rk))
      ALPHR = ALPHR + XNORM_SQ/real( ALPHA, kind=rk )
#endif

#if REALCASE == 1
      TAU = ALPHA / BETA
      ALPHA = -ALPHA
#endif
#if COMPLEXCASE == 1
      TAU = PRECISION_CMPLX( ALPHR/BETA, -ALPHI/BETA )
      ALPHA = PRECISION_CMPLX( -ALPHR, ALPHI )
#endif
    END IF
    XF = 1.0_rk/ALPHA
    ALPHA = BETA
  endif

  if (wantDebug) call obj%timer%stop("hh_transform_&
  &MATH_DATATYPE&
  &" // &
  &PRECISION_SUFFIX )

#if REALCASE == 1
end subroutine hh_transform_real_&
#endif
#if COMPLEXCASE == 1
    end subroutine hh_transform_complex_&
#endif
    &PRECISION
