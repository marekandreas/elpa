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
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaftrn,
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

module pack_unpack_complex
#include "config-f90.h"
  implicit none

#ifdef WITH_OPENMP
  public pack_row_complex_cpu_openmp_double
#else
  public pack_row_complex_cpu_double
#endif

#ifdef WANT_SINGLE_PRECISION_COMPLEX

#ifdef WITH_OPENMP
  public pack_row_complex_cpu_openmp_single
#else
  public pack_row_complex_cpu_single
#endif

#endif /* WANT_SINGLE_PRECISION_COMPLEX */

  contains
#ifdef WITH_OPENMP
         subroutine pack_row_complex_cpu_openmp_double(a, row, n, stripe_width, stripe_count, max_threads, thread_width, l_nev)
#else
         subroutine pack_row_complex_cpu_double(a, row, n, stripe_width, last_stripe_width, stripe_count)
#endif

#ifdef HAVE_DETAILED_TIMINGS
           use timings
#endif
           use precision
           implicit none
#ifdef WITH_OPENMP
           integer(kind=ik), intent(in) :: stripe_width, stripe_count, max_threads, thread_width, l_nev
           complex(kind=ck8), intent(in) :: a(:,:,:,:)
#else
           integer(kind=ik), intent(in) :: stripe_width, last_stripe_width, stripe_count
           complex(kind=ck8), intent(in) :: a(:,:,:)
#endif
           complex(kind=ck8)             :: row(:)
           integer(kind=ik)             :: n, i, noff, nl, nt

#ifdef HAVE_DETAILED_TIMINGS
#ifdef WITH_OPENMP
           call timer%start("pack_row_complex_cpu_openmp_double")
#else
           call timer%start("pack_row_complex_cpu_double")
#endif
#endif

#ifdef WITH_OPENMP
           do nt = 1, max_threads
             do i = 1, stripe_count
               noff = (nt-1)*thread_width + (i-1)*stripe_width
               nl   = min(stripe_width, nt*thread_width-noff, l_nev-noff)
               if (nl<=0) exit
               row(noff+1:noff+nl) = a(1:nl,n,i,nt)
             enddo
           enddo
#else
           do i=1,stripe_count
             nl = merge(stripe_width, last_stripe_width, i<stripe_count)
             noff = (i-1)*stripe_width
             row(noff+1:noff+nl) = a(1:nl,n,i)
           enddo
#endif

#ifdef HAVE_DETAILED_TIMINGS
#ifdef WITH_OPENMP
           call timer%stop("pack_row_complex_cpu_openmp_double")
#else
           call timer%stop("pack_row_complex_cpu_double")
#endif
#endif

#ifdef WITH_OPENMP
         end subroutine pack_row_complex_cpu_openmp_double
#else
         end subroutine pack_row_complex_cpu_double
#endif

#ifdef WITH_OPENMP
         subroutine unpack_row_complex_cpu_openmp_double(a, row, n, my_thread, stripe_count, thread_width, stripe_width, l_nev)
#ifdef HAVE_DETAILED_TIMINGS
           use timings
#endif
           use precision
           implicit none

           ! Private variables in OMP regions (my_thread) should better be in the argument list!
           integer(kind=ik), intent(in)  :: n, my_thread
           integer(kind=ik), intent(in)  :: stripe_count, thread_width, stripe_width, l_nev
           complex(kind=ck8), intent(in)  :: row(:)
           complex(kind=ck8)              :: a(:,:,:,:)
           integer(kind=ik)              :: i, noff, nl

#ifdef HAVE_DETAILED_TIMINGS
           call timer%start("unpack_row_complex_cpu_openmp_double")
#endif

           do i=1,stripe_count
             noff = (my_thread-1)*thread_width + (i-1)*stripe_width
             nl   = min(stripe_width, my_thread*thread_width-noff, l_nev-noff)
             if (nl<=0) exit
             a(1:nl,n,i,my_thread) = row(noff+1:noff+nl)
           enddo

#ifdef HAVE_DETAILED_TIMINGS
           call timer%stop("unpack_row_complex_cpu_openmp_double")
#endif
         end subroutine unpack_row_complex_cpu_openmp_double

#else /* WITH_OPENMP */

         subroutine unpack_row_complex_cpu_double(a, row, n, stripe_count, stripe_width, last_stripe_width)
#ifdef HAVE_DETAILED_TIMINGS
           use timings
#endif
           use precision
           implicit none
           integer(kind=ik), intent(in) :: stripe_count, stripe_width, last_stripe_width, n
           complex(kind=ck8), intent(in) :: row(:)
           complex(kind=ck8)             :: a(:,:,:)
           integer(kind=ik)             :: i, noff, nl

#ifdef HAVE_DETAILED_TIMINGS
           call timer%start("unpack_row_complex_cpu_double")
#endif
           do i=1,stripe_count
             nl = merge(stripe_width, last_stripe_width, i<stripe_count)
             noff = (i-1)*stripe_width
             a(1:nl,n,i) = row(noff+1:noff+nl)
           enddo

#ifdef HAVE_DETAILED_TIMINGS
           call timer%stop("unpack_row_complex_cpu_double")
#endif

         end  subroutine unpack_row_complex_cpu_double
#endif /* WITH_OPENMP */

#ifdef WANT_SINGLE_PRECISION_COMPLEX
 ! single precision implementation, at the moment duplivated !!

#ifdef WITH_OPENMP
         subroutine pack_row_complex_cpu_openmp_single(a, row, n, stripe_width, stripe_count, max_threads, thread_width, l_nev)
#else
         subroutine pack_row_complex_cpu_single(a, row, n, stripe_width, last_stripe_width, stripe_count)
#endif

#ifdef HAVE_DETAILED_TIMINGS
           use timings
#endif
           use precision
           implicit none
#ifdef WITH_OPENMP
           integer(kind=ik), intent(in) :: stripe_width, stripe_count, max_threads, thread_width, l_nev
           complex(kind=ck4), intent(in) :: a(:,:,:,:)
#else
           integer(kind=ik), intent(in) :: stripe_width, last_stripe_width, stripe_count
           complex(kind=ck4), intent(in) :: a(:,:,:)
#endif
           complex(kind=ck4)             :: row(:)
           integer(kind=ik)             :: n, i, noff, nl, nt

#ifdef HAVE_DETAILED_TIMINGS
#ifdef WITH_OPENMP
           call timer%start("pack_row_complex_cpu_openmp_single")
#else
           call timer%start("pack_row_complex_cpu_single")
#endif
#endif

#ifdef WITH_OPENMP
           do nt = 1, max_threads
             do i = 1, stripe_count
               noff = (nt-1)*thread_width + (i-1)*stripe_width
               nl   = min(stripe_width, nt*thread_width-noff, l_nev-noff)
               if (nl<=0) exit
               row(noff+1:noff+nl) = a(1:nl,n,i,nt)
             enddo
           enddo
#else
           do i=1,stripe_count
             nl = merge(stripe_width, last_stripe_width, i<stripe_count)
             noff = (i-1)*stripe_width
             row(noff+1:noff+nl) = a(1:nl,n,i)
           enddo
#endif

#ifdef HAVE_DETAILED_TIMINGS
#ifdef WITH_OPENMP
           call timer%stop("pack_row_complex_cpu_openmp_single")
#else
           call timer%stop("pack_row_complex_cpu_single")
#endif
#endif

#ifdef WITH_OPENMP
         end subroutine pack_row_complex_cpu_openmp_single
#else
         end subroutine pack_row_complex_cpu_single
#endif

#ifdef WITH_OPENMP
         subroutine unpack_row_complex_cpu_openmp_single(a, row, n, my_thread, stripe_count, thread_width, stripe_width, l_nev)
#ifdef HAVE_DETAILED_TIMINGS
           use timings
#endif
           use precision
           implicit none

           ! Private variables in OMP regions (my_thread) should better be in the argument list!
           integer(kind=ik), intent(in)  :: n, my_thread
           integer(kind=ik), intent(in)  :: stripe_count, thread_width, stripe_width, l_nev
           complex(kind=ck4), intent(in)  :: row(:)
           complex(kind=ck4)              :: a(:,:,:,:)
           integer(kind=ik)              :: i, noff, nl

#ifdef HAVE_DETAILED_TIMINGS
           call timer%start("unpack_row_complex_cpu_openmp_single")
#endif

           do i=1,stripe_count
             noff = (my_thread-1)*thread_width + (i-1)*stripe_width
             nl   = min(stripe_width, my_thread*thread_width-noff, l_nev-noff)
             if (nl<=0) exit
             a(1:nl,n,i,my_thread) = row(noff+1:noff+nl)
           enddo

#ifdef HAVE_DETAILED_TIMINGS
           call timer%stop("unpack_row_complex_cpu_openmp_single")
#endif
         end subroutine unpack_row_complex_cpu_openmp_single

#else /* WITH_OPENMP */

         subroutine unpack_row_complex_cpu_single(a, row, n, stripe_count, stripe_width, last_stripe_width)
#ifdef HAVE_DETAILED_TIMINGS
           use timings
#endif
           use precision
           implicit none
           integer(kind=ik), intent(in) :: stripe_count, stripe_width, last_stripe_width, n
           complex(kind=ck4), intent(in) :: row(:)
           complex(kind=ck4)             :: a(:,:,:)
           integer(kind=ik)             :: i, noff, nl

#ifdef HAVE_DETAILED_TIMINGS
           call timer%start("unpack_row_complex_cpu_single")
#endif
           do i=1,stripe_count
             nl = merge(stripe_width, last_stripe_width, i<stripe_count)
             noff = (i-1)*stripe_width
             a(1:nl,n,i) = row(noff+1:noff+nl)
           enddo

#ifdef HAVE_DETAILED_TIMINGS
           call timer%stop("unpack_row_complex_cpu_single")
#endif

         end  subroutine unpack_row_complex_cpu_single
#endif /* WITH_OPENMP */

#endif /* WANT_SINGLE_PRECISION_COMPLEX */
end module
