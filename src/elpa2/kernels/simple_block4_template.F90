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
!
! --------------------------------------------------------------------------------------------------
!
! This file contains the compute intensive kernels for the Householder transformations.
!
! This is the small and simple version (no hand unrolling of loops etc.) but for some
! compilers this performs better than a sophisticated version with transformed and unrolled loops.
!
! It should be compiled with the highest possible optimization level.
!
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
! --------------------------------------------------------------------------------------------------
#endif

  subroutine quad_hh_trafo_&
  &MATH_DATATYPE&
  &_generic_simple_4hv_&
  &PRECISION&
  & (q, hh, nb, nq, ldq, ldh)

    use precision
    use elpa_abstract_impl
    implicit none

    !class(elpa_abstract_impl_t), intent(inout) :: obj
    integer(kind=ik), intent(in)    :: nb, nq, ldq, ldh
#if REALCASE==1

#ifdef USE_ASSUMED_SIZE
    real(kind=C_DATATYPE_KIND), intent(inout) :: q(ldq,*)
    real(kind=C_DATATYPE_KIND), intent(in)    :: hh(ldh,*)
#else
    real(kind=C_DATATYPE_KIND), intent(inout) :: q(1:ldq,1:nb+3)
    real(kind=C_DATATYPE_KIND), intent(in)    :: hh(1:ldh,1:6)
#endif
    real(kind=C_DATATYPE_KIND)                :: s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4
    real(kind=C_DATATYPE_KIND)                :: vs_1_2, vs_1_3, vs_2_3, vs_1_4, vs_2_4, vs_3_4
    real(kind=C_DATATYPE_KIND)                :: h_2_1, h_3_2, h_3_1, h_4_3, h_4_2, h_4_1
    real(kind=C_DATATYPE_KIND)                :: a_1_1(nq), a_2_1(nq), a_3_1(nq), a_4_1(nq)
    real(kind=C_DATATYPE_KIND)                :: h1, h2, h3, h4
    real(kind=C_DATATYPE_KIND)                :: w(nq), z(nq), x(nq), y(nq)
    real(kind=C_DATATYPE_KIND)                :: tau1, tau2, tau3, tau4
#endif /* REALCASE==1 */

#if COMPLEXCASE==1

#ifdef USE_ASSUMED_SIZE
    complex(kind=C_DATATYPE_KIND), intent(inout) :: q(ldq,*)
    complex(kind=C_DATATYPE_KIND), intent(in)    :: hh(ldh,*)
#else
    complex(kind=C_DATATYPE_KIND), intent(inout) :: q(1:ldq,1:nb+3)
    complex(kind=C_DATATYPE_KIND), intent(in)    :: hh(1:ldh,1:6)
#endif
    complex(kind=C_DATATYPE_KIND)                :: s_1_2, s_1_3, s_2_3, s_1_4, s_2_4, s_3_4
    complex(kind=C_DATATYPE_KIND)                :: vs_1_2, vs_1_3, vs_2_3, vs_1_4, vs_2_4, vs_3_4
    complex(kind=C_DATATYPE_KIND)                :: h_2_1, h_3_2, h_3_1, h_4_3, h_4_2, h_4_1
    complex(kind=C_DATATYPE_KIND)                :: a_1_1(nq), a_2_1(nq), a_3_1(nq), a_4_1(nq)
    complex(kind=C_DATATYPE_KIND)                :: w(nq), z(nq), x(nq), y(nq)
    complex(kind=C_DATATYPE_KIND)                :: h1, h2, h3, h4
    complex(kind=C_DATATYPE_KIND)                :: tau1, tau2, tau3, tau4
#endif /* COMPLEXCASE==1 */
    integer(kind=ik)                             :: i
    ! Calculate dot product of the two Householder vectors

#if REALCASE==1
    s_1_2 = hh(2,2)
    s_1_3 = hh(3,3)
    s_2_3 = hh(2,3) 
    s_1_4 = hh(4,4)
    s_2_4 = hh(3,4)
    s_3_4 = hh(2,4)

    s_1_2 = s_1_2 + hh(2,1) * hh(3,2)
    s_2_3 = s_2_3 + hh(2,2) * hh(3,3)
    s_3_4 = s_3_4 + hh(2,3) * hh(3,4)

    s_1_2 = s_1_2 + hh(3,1) * hh(4,2)
    s_2_3 = s_2_3 + hh(3,2) * hh(4,3)
    s_3_4 = s_3_4 + hh(3,3) * hh(4,4)

    s_1_3 = s_1_3 + hh(2,1) * hh(4,3)
    s_2_4 = s_2_4 + hh(2,2) * hh(4,4)

    !DIR$ IVDEP
    do i=5,nb
       s_1_2 = s_1_2 + hh(i-1,1) * hh(i,2)
       s_2_3 = s_2_3 + hh(i-1,2) * hh(i,3)
       s_3_4 = s_3_4 + hh(i-1,3) * hh(i,4)

       s_1_3 = s_1_3 + hh(i-2,1) * hh(i,3)
       s_2_4 = s_2_4 + hh(i-2,2) * hh(i,4)

       s_1_4 = s_1_4 + hh(i-3,1) * hh(i,4)
    enddo
#endif

#if COMPLEXCASE==1
    print *, "simple_block4_template.F90: Error, aborting..."
    stop 1
    !s = conjg(hh(2,2))*1.0
    !do i=3,nb
    !   s = s+(conjg(hh(i,2))*hh(i-1,1))
    !enddo
#endif

    ! Do the Householder transformations
    a_1_1(1:nq) = q(1:nq,4)
    a_2_1(1:nq) = q(1:nq,3)
    a_3_1(1:nq) = q(1:nq,2)
    a_4_1(1:nq) = q(1:nq,1)

    h_2_1 = hh(2,2)
    h_3_2 = hh(2,3)
    h_3_1 = hh(3,3)
    h_4_3 = hh(2,4)
    h_4_2 = hh(3,4)
    h_4_1 = hh(4,4)

#if REALCASE == 1
    w(1:nq) = a_3_1(1:nq) * h_4_3 + a_4_1(1:nq)
    w(1:nq) = a_2_1(1:nq) * h_4_2 +     w(1:nq)
    w(1:nq) = a_1_1(1:nq) * h_4_1 +     w(1:nq)

    z(1:nq) = a_2_1(1:nq) * h_3_2 + a_3_1(1:nq)
    z(1:nq) = a_1_1(1:nq) * h_3_1 +     z(1:nq)

    y(1:nq) = a_1_1(1:nq) * h_2_1 + a_2_1(1:nq)

    x(1:nq) = a_1_1(1:nq)
#endif

#if COMPLEXCASE==1
    print *, "simple_block4_template.F90: Error, aborting..."
    stop 1
    !y(1:nq) = q(1:nq,1) + q(1:nq,2)*conjg(hh(2,2))
#endif

    do i=5,nb
#if REALCASE == 1
      h1 = hh(i-3,1)
      h2 = hh(i-2,2)
      h3 = hh(i-1,3)
      h4 = hh(i  ,4)
#endif
#if COMPLEXCASE==1
      print *, "simple_block4_template.F90: Error, aborting..."
      stop 1
    !   h1 = conjg(hh(i-1,1))
    !   h2 = conjg(hh(i,2))
#endif

      x(1:nq) = x(1:nq) + q(1:nq,i) * h1
      y(1:nq) = y(1:nq) + q(1:nq,i) * h2
      z(1:nq) = z(1:nq) + q(1:nq,i) * h3
      w(1:nq) = w(1:nq) + q(1:nq,i) * h4
    enddo

    h1 = hh(nb-2,1)
    h2 = hh(nb-1,2)
    h3 = hh(nb  ,3)

#if REALCASE==1
    x(1:nq) = x(1:nq) + q(1:nq,nb+1) * h1 
    y(1:nq) = y(1:nq) + q(1:nq,nb+1) * h2
    z(1:nq) = z(1:nq) + q(1:nq,nb+1) * h3
#endif

#if COMPLEXCASE==1
    print *, "simple_block4_template.F90: Error, aborting..."
    stop 1
    !x(1:nq) = x(1:nq) + q(1:nq,nb+1)*conjg(hh(nb,1))
#endif

    h1 = hh(nb-1,1)
    h2 = hh(nb  ,2)

    x(1:nq) = x(1:nq) + q(1:nq,nb+2) * h1
    y(1:nq) = y(1:nq) + q(1:nq,nb+2) * h2

    h1 = hh(nb,1)

    x(1:nq) = x(1:nq) + q(1:nq,nb+3) * h1


    ! Rank-1 update
    tau1 = hh(1,1)
    tau2 = hh(1,2)
    tau3 = hh(1,3)
    tau4 = hh(1,4)

    vs_1_2 = s_1_2
    vs_1_3 = s_1_3
    vs_2_3 = s_2_3
    vs_1_4 = s_1_4
    vs_2_4 = s_2_4
    vs_3_4 = s_3_4

    h1 = tau1
    x(1:nq) = x(1:nq) * h1

    h1 = tau2
    h2 = tau2 * vs_1_2
    y(1:nq) = y(1:nq) * h1 - x(1:nq) * h2

    h1 = tau3
    h2 = tau3 * vs_1_3
    h3 = tau3 * vs_2_3
    z(1:nq) = z(1:nq) * h1  - (y(1:nq) * h3 + x(1:nq) * h2)

    h1 = tau4
    h2 = tau4 * vs_1_4
    h3 = tau4 * vs_2_4
    h4 = tau4 * vs_3_4

    w(1:nq) = w(1:nq) * h1 - ( z(1:nq) * h4 + y(1:nq) * h3 + x(1:nq) * h2)

    q(1:nq,1) = q(1:nq,1) - w(1:nq)

    h4 = hh(2,4)

    q(1:nq,2) = q(1:nq,2) - (w(1:nq) * h4 + z(1:nq))

    h3 = hh(2,3)
    h4 = hh(3,4)

    q(1:nq,3) = q(1:nq,3) - y(1:nq)
    q(1:nq,3) = -( z(1:nq) * h3) + q(1:nq,3)
    q(1:nq,3) = -( w(1:nq) * h4) + q(1:nq,3)

    h2 = hh(2,2)
    h3 = hh(3,3)
    h4 = hh(4,4)

    q(1:nq,4) =  q(1:nq,4) - x(1:nq)
    q(1:nq,4) = -(y(1:nq) * h2) + q(1:nq,4)
    q(1:nq,4) = -(z(1:nq) * h3) + q(1:nq,4)
    q(1:nq,4) = -(w(1:nq) * h4) + q(1:nq,4)

    do i=5,nb
       h1 = hh(i-3,1)
       h2 = hh(i-2,2)
       h3 = hh(i-1,3)
       h4 = hh(i  ,4)

       q(1:nq,i) = -(x(1:nq) * h1) + q(1:nq,i)
       q(1:nq,i) = -(y(1:nq) * h2) + q(1:nq,i)
       q(1:nq,i) = -(z(1:nq) * h3) + q(1:nq,i)
       q(1:nq,i) = -(w(1:nq) * h4) + q(1:nq,i)
   enddo

   h1 = hh(nb-2,1)
   h2 = hh(nb-1,2)
   h3 = hh(nb  ,3)

   q(1:nq,nb+1) = -(x(1:nq) * h1) + q(1:nq,nb+1)
   q(1:nq,nb+1) = -(y(1:nq) * h2) + q(1:nq,nb+1)
   q(1:nq,nb+1) = -(z(1:nq) * h3) + q(1:nq,nb+1)

   h1 = hh(nb-1,1)
   h2 = hh(nb  ,2)

   q(1:nq,nb+2) = - (x(1:nq) * h1) + q(1:nq,nb+2)
   q(1:nq,nb+2) = - (y(1:nq) * h2) + q(1:nq,nb+2)

   h1 = hh(nb,1)
   q(1:nq,nb+3) = - (x(1:nq) * h1) + q(1:nq,nb+3)

  end subroutine
