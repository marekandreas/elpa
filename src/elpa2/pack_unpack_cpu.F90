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
! This file was written by A. Marek, MPCDF
#endif

subroutine pack_row_&
&MATH_DATATYPE&
#ifdef WITH_OPENMP_TRADITIONAL
&_cpu_openmp_&
#else
&_cpu_&
#endif
&PRECISION &
(obj, a, row, n, stripe_width,  &
#ifdef WITH_OPENMP_TRADITIONAL
stripe_count, nt, thread_width, l_nev)
#else
last_stripe_width, stripe_count)
#endif
  use elpa_abstract_impl
  use precision
  implicit none
  class(elpa_abstract_impl_t), intent(inout) :: obj

  integer(kind=ik), intent(in)               :: n, stripe_count, stripe_width
#ifdef WITH_OPENMP_TRADITIONAL
  integer(kind=ik), intent(in)               :: thread_width, l_nev
  logical                                    :: useOPENMP

#if REALCASE == 1
  real(kind=C_DATATYPE_KIND), intent(in)     :: a(:,:,:,:)
#endif
#if COMPLEXCASE == 1
  complex(kind=C_DATATYPE_KIND), intent(in)  :: a(:,:,:,:)
#endif

#else /* WITH_OPENMP_TRADITIONAL */
  integer(kind=ik), intent(in)               :: last_stripe_width
#if REALCASE == 1
  real(kind=C_DATATYPE_KIND), intent(in)     :: a(:,:,:)
#endif
#if COMPLEXCASE == 1
  complex(kind=C_DATATYPE_KIND), intent(in)  :: a(:,:,:)
#endif

#endif /* WITH_OPENMP_TRADITIONAL */

#if REALCASE == 1
  real(kind=C_DATATYPE_KIND)                 :: row(:)
#endif
#if COMPLEXCASE == 1
  complex(kind=C_DATATYPE_KIND)              :: row(:)
#endif

  integer(kind=ik)                           :: i, noff, nl
#ifdef WITH_OPENMP_TRADITIONAL
  integer(kind=ik), intent(in)                           :: nt
#endif

  call obj%timer%start("pack_row_&
  &MATH_DATATYPE&
#ifdef WITH_OPENMP_TRADITIONAL
  &_cpu_openmp" // &
#else
  &_cpu" // &
#endif
  &PRECISION_SUFFIX &
  )

  do i = 1, stripe_count
#ifdef WITH_OPENMP_TRADITIONAL
  !do nt = 1, max_threads
      noff = (nt-1)*thread_width + (i-1)*stripe_width
      nl   = min(stripe_width, nt*thread_width-noff, l_nev-noff)
      if (nl<=0) exit
      row(noff+1:noff+nl) = a(1:nl,n,i,nt)
  !enddo
#else
  !do i=1,stripe_count
    nl = merge(stripe_width, last_stripe_width, i<stripe_count)
    noff = (i-1)*stripe_width
    row(noff+1:noff+nl) = a(1:nl,n,i)
  !enddo
#endif
  enddo

  call obj%timer%stop("pack_row_&
  &MATH_DATATYPE&
#ifdef WITH_OPENMP_TRADITIONAL
  &_cpu_openmp" // &
#else
  &_cpu" // &
#endif
  &PRECISION_SUFFIX &
  )

end subroutine

subroutine unpack_row_&
&MATH_DATATYPE&
#ifdef WITH_OPENMP_TRADITIONAL
&_cpu_openmp_&
#else
&_cpu_&
#endif
&PRECISION &
(obj, a, row, n, &
#ifdef WITH_OPENMP_TRADITIONAL
my_thread, &
#endif
stripe_count, &
#ifdef WITH_OPENMP_TRADITIONAL
thread_width, &
#endif
stripe_width, &
#ifdef WITH_OPENMP_TRADITIONAL
l_nev)
#else
last_stripe_width)
#endif
  use elpa_abstract_impl
  use precision
  implicit none
  class(elpa_abstract_impl_t), intent(inout) :: obj
  integer(kind=ik), intent(in)               :: n, stripe_count, stripe_width

#ifdef WITH_OPENMP_TRADITIONAL
  ! Private variables in OMP regions (my_thread) should better be in the argument list!
  integer(kind=ik), intent(in)               :: thread_width, l_nev, my_thread
#if REALCASE == 1
  real(kind=C_DATATYPE_KIND)                 :: a(:,:,:,:)
#endif
#if COMPLEXCASE == 1
  complex(kind=C_DATATYPE_KIND)              :: a(:,:,:,:)

#endif
#else /* WITH_OPENMP_TRADITIONAL */
  integer(kind=ik), intent(in)               :: last_stripe_width
#if REALCASE == 1
  real(kind=C_DATATYPE_KIND)                 :: a(:,:,:)
#endif
#if COMPLEXCASE == 1
  complex(kind=C_DATATYPE_KIND)              :: a(:,:,:)
#endif

#endif /* WITH_OPENMP_TRADITIONAL */

#if REALCASE == 1
  real(kind=C_DATATYPE_KIND), intent(in)     :: row(:)
#endif
#if COMPLEXCASE == 1
  complex(kind=C_DATATYPE_KIND), intent(in)  :: row(:)
#endif
  integer(kind=ik)                           :: i, noff, nl

  call obj%timer%start("unpack_row_&
  &MATH_DATATYPE&
#ifdef WITH_OPENMP_TRADITIONAL
  &_cpu_opemp" // &
#else
  &_cpu" // &
#endif
  &PRECISION_SUFFIX &
  )

  do i=1,stripe_count
#ifdef WITH_OPENMP_TRADITIONAL
    noff = (my_thread-1)*thread_width + (i-1)*stripe_width
    nl   = min(stripe_width, my_thread*thread_width-noff, l_nev-noff)
    if ( nl<= 0) exit
    a(1:nl,n,i,my_thread) = row(noff+1:noff+nl)
#else
    nl = merge(stripe_width, last_stripe_width, i<stripe_count)
    noff = (i-1)*stripe_width
    a(1:nl,n,i) = row(noff+1:noff+nl)
#endif

  enddo

  call obj%timer%stop("unpack_row_&
  &MATH_DATATYPE&
#ifdef WITH_OPENMP_TRADITIONAL
  &_cpu_opemp" // &
#else
  &_cpu" // &
#endif
  &PRECISION_SUFFIX &
  )

end subroutine

