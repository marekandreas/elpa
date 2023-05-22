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
! Author: A. Marek, MPCDF
! --------------------------------------------------------------------------------------------------
#endif

  subroutine hexa_hh_trafo_&
  &MATH_DATATYPE&
  &_generic_simple_6hv_&
  &PRECISION&
  & (q, hh, nb, nq, ldq, ldh)

    use precision
    use elpa_abstract_impl
    implicit none

    integer(kind=ik), intent(in)              :: nb, nq, ldq, ldh
#if REALCASE==1

#ifdef USE_ASSUMED_SIZE
    real(kind=C_DATATYPE_KIND), intent(inout) :: q(ldq,*)
    real(kind=C_DATATYPE_KIND), intent(in)    :: hh(ldh,*)
#else
    real(kind=C_DATATYPE_KIND), intent(inout) :: q(1:ldq,1:nb+5)
    real(kind=C_DATATYPE_KIND), intent(in)    :: hh(1:ldh,1:6)
#endif
    real(kind=C_DATATYPE_KIND)                :: scalarproduct(15)
    real(kind=C_DATATYPE_KIND)                :: vs_1_2, vs_1_3, vs_2_3, vs_1_4, vs_2_4, vs_3_4
    real(kind=C_DATATYPE_KIND)                :: vs_1_5, vs_1_6, vs_2_5, vs_2_6, vs_3_5
    real(kind=C_DATATYPE_KIND)                :: vs_3_6, vs_4_5, vs_4_6, vs_5_6
    real(kind=C_DATATYPE_KIND)                :: a_1_1(nq), a_2_1(nq), a_3_1(nq), a_4_1(nq), a_5_1(nq), a_6_1(nq)
    real(kind=C_DATATYPE_KIND)                :: h_6_5, h_6_4, h_6_3, h_6_2, h_6_1
    real(kind=C_DATATYPE_KIND)                :: h_5_4, h_5_3, h_5_2, h_5_1
    real(kind=C_DATATYPE_KIND)                :: h_4_3, h_4_2, h_4_1
    real(kind=C_DATATYPE_KIND)                :: h_2_1, h_3_2, h_3_1
    real(kind=C_DATATYPE_KIND)                :: h1, h2, h3, h4, h5, h6
    real(kind=C_DATATYPE_KIND)                :: w(nq), z(nq), x(nq), y(nq), t(nq), v(nq)
    real(kind=C_DATATYPE_KIND)                :: tau1, tau2, tau3, tau4, tau5, tau6
#endif /* REALCASE==1 */


    integer(kind=ik)                             :: i, j
    ! Calculate dot product of the two Householder vectors

    scalarproduct(1)  = hh(2,2)
    scalarproduct(2)  = hh(3,3)
    scalarproduct(3)  = hh(2,3)
    scalarproduct(4)  = hh(4,4)
    scalarproduct(5)  = hh(3,4)
    scalarproduct(6)  = hh(2,4)
    scalarproduct(7)  = hh(5,5)
    scalarproduct(8)  = hh(4,5)
    scalarproduct(9)  = hh(3,5)
    scalarproduct(10) = hh(2,5)
    scalarproduct(11) = hh(6,6)
    scalarproduct(12) = hh(5,6)
    scalarproduct(13) = hh(4,6)
    scalarproduct(14) = hh(3,6)
    scalarproduct(15) = hh(2,6)

    scalarproduct(1)  = scalarproduct(1)  + hh(2,1) * hh(3,2)
    scalarproduct(3)  = scalarproduct(3)  + hh(2,2) * hh(3,3)
    scalarproduct(6)  = scalarproduct(6)  + hh(2,3) * hh(3,4)
    scalarproduct(10) = scalarproduct(10) + hh(2,4) * hh(3,5)
    scalarproduct(15) = scalarproduct(15) + hh(2,5) * hh(3,6)

    scalarproduct(1)  = scalarproduct(1)  + hh(3,1) * hh(4,2)
    scalarproduct(3)  = scalarproduct(3)  + hh(3,2) * hh(4,3)
    scalarproduct(6)  = scalarproduct(6)  + hh(3,3) * hh(4,4)
    scalarproduct(10) = scalarproduct(10) + hh(3,4) * hh(4,5)
    scalarproduct(15) = scalarproduct(15) + hh(3,5) * hh(4,6)

    scalarproduct(2)  = scalarproduct(2)  + hh(2,1) * hh(4,3)
    scalarproduct(5)  = scalarproduct(5)  + hh(2,2) * hh(4,4)
    scalarproduct(9)  = scalarproduct(9)  + hh(2,3) * hh(4,5)
    scalarproduct(14) = scalarproduct(14) + hh(2,4) * hh(4,6)

    scalarproduct(1)  = scalarproduct(1)  + hh(4,1) * hh(5,2)
    scalarproduct(3)  = scalarproduct(3)  + hh(4,2) * hh(5,3)
    scalarproduct(6)  = scalarproduct(6)  + hh(4,3) * hh(5,4)
    scalarproduct(10) = scalarproduct(10) + hh(4,4) * hh(5,5)
    scalarproduct(15) = scalarproduct(15) + hh(4,5) * hh(5,6)

    scalarproduct(2)  = scalarproduct(2)  + hh(3,1) * hh(5,3)
    scalarproduct(5)  = scalarproduct(5)  + hh(3,2) * hh(5,4)
    scalarproduct(9)  = scalarproduct(9)  + hh(3,3) * hh(5,5)
    scalarproduct(14) = scalarproduct(14) + hh(3,4) * hh(5,6)

    scalarproduct(4)  = scalarproduct(4)  + hh(2,1) * hh(5,4)
    scalarproduct(8)  = scalarproduct(8)  + hh(2,2) * hh(5,5)
    scalarproduct(13) = scalarproduct(13) + hh(2,3) * hh(5,6)

    scalarproduct(1)  = scalarproduct(1)  + hh(5,1) * hh(6,2)
    scalarproduct(3)  = scalarproduct(3)  + hh(5,2) * hh(6,3)
    scalarproduct(6)  = scalarproduct(6)  + hh(5,3) * hh(6,4)
    scalarproduct(10) = scalarproduct(10) + hh(5,4) * hh(6,5)
    scalarproduct(15) = scalarproduct(15) + hh(5,5) * hh(6,6)

    scalarproduct(2)  = scalarproduct(2)  + hh(4,1) * hh(6,3)
    scalarproduct(5)  = scalarproduct(5)  + hh(4,2) * hh(6,4)
    scalarproduct(9)  = scalarproduct(9)  + hh(4,3) * hh(6,5)
    scalarproduct(14) = scalarproduct(14) + hh(4,4) * hh(6,6)

    scalarproduct(4)  = scalarproduct(4)  + hh(3,1) * hh(6,4)
    scalarproduct(8)  = scalarproduct(8)  + hh(3,2) * hh(6,5)
    scalarproduct(13) = scalarproduct(13) + hh(3,3) * hh(6,6)

    scalarproduct(7)  = scalarproduct(7)  + hh(2,1) * hh(6,5)
    scalarproduct(12) = scalarproduct(12) + hh(2,2) * hh(6,6)

    !DIR$ IVDEP
    do i=7,nb
       scalarproduct(1)  = scalarproduct(1)  + hh(i-1,1) * hh(i,2)
       scalarproduct(3)  = scalarproduct(3)  + hh(i-1,2) * hh(i,3)
       scalarproduct(6)  = scalarproduct(6)  + hh(i-1,3) * hh(i,4)
       scalarproduct(10) = scalarproduct(10) + hh(i-1,4) * hh(i,5)
       scalarproduct(15) = scalarproduct(15) + hh(i-1,5) * hh(i,6)

       scalarproduct(2)  = scalarproduct(2)  + hh(i-2,1) * hh(i,3)
       scalarproduct(5)  = scalarproduct(5)  + hh(i-2,2) * hh(i,4)
       scalarproduct(9)  = scalarproduct(9)  + hh(i-2,3) * hh(i,5)
       scalarproduct(14) = scalarproduct(14) + hh(i-2,4) * hh(i,6)

       scalarproduct(4)  = scalarproduct(4)  + hh(i-3,1) * hh(i,4)
       scalarproduct(8)  = scalarproduct(8)  + hh(i-3,2) * hh(i,5)
       scalarproduct(13) = scalarproduct(13) + hh(i-3,3) * hh(i,6)

       scalarproduct(7)  = scalarproduct(7)  + hh(i-4,1) * hh(i,5)
       scalarproduct(12) = scalarproduct(12) + hh(i-4,2) * hh(i,6)

       scalarproduct(11) = scalarproduct(11) + hh(i-5,1) * hh(i,6)
    enddo

#if COMPLEXCASE==1
    print *, "simple_block6_template.F90: Error, aborting..."
    stop 1
    !s = conjg(hh(2,2))*1.0
    !do i=3,nb
    !   s = s+(conjg(hh(i,2))*hh(i-1,1))
    !enddo
#endif

    ! Do the Householder transformations
    a_1_1(1:nq) = q(1:nq,6)
    a_2_1(1:nq) = q(1:nq,5)
    a_3_1(1:nq) = q(1:nq,4)
    a_4_1(1:nq) = q(1:nq,3)
    a_5_1(1:nq) = q(1:nq,2)
    a_6_1(1:nq) = q(1:nq,1)

    h_6_5 = hh(2,6)
    h_6_4 = hh(3,6)
    h_6_3 = hh(4,6)
    h_6_2 = hh(5,6)
    h_6_1 = hh(6,6)

    t(1:nq) = a_6_1(1:nq) + a_5_1(1:nq) * h_6_5 + a_4_1(1:nq) * h_6_4 + a_3_1(1:nq) * h_6_3 + a_2_1(1:nq) * h_6_2 + &
                            a_1_1(1:nq) * h_6_1

    h_5_4 = hh(2,5)
    h_5_3 = hh(3,5)
    h_5_2 = hh(4,5)
    h_5_1 = hh(5,5)

    v(1:nq) = a_5_1(1:nq) + a_4_1(1:nq) * h_5_4 + a_3_1(1:nq) * h_5_3 + a_2_1(1:nq) * h_5_2 + a_1_1(1:nq) * h_5_1

    h_4_3 = hh(2,4)
    h_4_2 = hh(3,4)
    h_4_1 = hh(4,4)

    w(1:nq) = a_4_1(1:nq) + a_3_1(1:nq) * h_4_3 + a_2_1(1:nq) * h_4_2 + a_1_1(1:nq) * h_4_1

    h_2_1 = hh(2,2)
    h_3_2 = hh(2,3)
    h_3_1 = hh(3,3)

    z(1:nq) = a_3_1(1:nq) + a_2_1(1:nq) * h_3_2 + a_1_1(1:nq) * h_3_1

    y(1:nq) = a_2_1(1:nq) + a_1_1(1:nq) * h_2_1

    x(1:nq) = a_1_1(1:nq)

    do i=7,nb
      h1 = hh(i-5,1) !
      h2 = hh(i-4,2) !
      h3 = hh(i-3,3) !
      h4 = hh(i-2,4) !
      h5 = hh(i-1,5) !
      h6 = hh(i  ,6) !
#if COMPLEXCASE==1
       print *, "simple_block6_template.F90: Error, aborting..."
       stop 1
    !   h1 = conjg(hh(i-1,1))
    !   h2 = conjg(hh(i,2))
#endif

      x(1:nq) = x(1:nq) + q(1:nq,i) * h1
      y(1:nq) = y(1:nq) + q(1:nq,i) * h2
      z(1:nq) = z(1:nq) + q(1:nq,i) * h3
      w(1:nq) = w(1:nq) + q(1:nq,i) * h4
      v(1:nq) = v(1:nq) + q(1:nq,i) * h5
      t(1:nq) = t(1:nq) + q(1:nq,i) * h6
    enddo

    h1 = hh(nb-4,1)
    h2 = hh(nb-3,2)
    h3 = hh(nb-2,3)
    h4 = hh(nb-1,4)
    h5 = hh(nb  ,5)

    x(1:nq) = x(1:nq) + q(1:nq,nb+1) * h1
    y(1:nq) = y(1:nq) + q(1:nq,nb+1) * h2
    z(1:nq) = z(1:nq) + q(1:nq,nb+1) * h3
    w(1:nq) = w(1:nq) + q(1:nq,nb+1) * h4
    v(1:nq) = v(1:nq) + q(1:nq,nb+1) * h5

#if COMPLEXCASE==1
    stop
    !x(1:nq) = x(1:nq) + q(1:nq,nb+1)*conjg(hh(nb,1))
#endif

    h1 = hh(nb-3,1)
    h2 = hh(nb-2,2)
    h3 = hh(nb-1,3)
    h4 = hh(nb  ,4)

    x(1:nq) = x(1:nq) + q(1:nq,nb+2) * h1
    y(1:nq) = y(1:nq) + q(1:nq,nb+2) * h2
    z(1:nq) = z(1:nq) + q(1:nq,nb+2) * h3
    w(1:nq) = w(1:nq) + q(1:nq,nb+2) * h4

    h1 = hh(nb-2,1)
    h2 = hh(nb-1,2)
    h3 = hh(nb  ,3)

    x(1:nq) = x(1:nq) + q(1:nq,nb+3) * h1
    y(1:nq) = y(1:nq) + q(1:nq,nb+3) * h2
    z(1:nq) = z(1:nq) + q(1:nq,nb+3) * h3

    h1 = hh(nb-1,1)
    h2 = hh(nb  ,2)

    x(1:nq) = x(1:nq)  + q(1:nq,nb+4) * h1
    y(1:nq) = y(1:nq)  + q(1:nq,nb+4) * h2

    h1 = hh(nb,1)

    x(1:nq) = x(1:nq) + q(1:nq,nb+5) * h1
 
    ! Rank-1 update
    tau1 = hh(1,1)
    x(1:nq) = x(1:nq) * tau1

    tau2 = hh(1,2)
    vs_1_2 = scalarproduct(1)

    h2 = tau2 * vs_1_2 !
    y(1:nq) = y(1:nq) * tau2 - (x(1:nq) * h2)

    tau3 = hh(1,3)
    vs_1_3 = scalarproduct(2)
    vs_2_3 = scalarproduct(3)

    h2 = tau3 * vs_1_3
    h3 = tau3 * vs_2_3
    z(1:nq) = z(1:nq) * tau3  - (y(1:nq) * h3 + x(1:nq) * h2)
 
    tau4 = hh(1,4)
    vs_1_4 = scalarproduct(4)
    vs_2_4 = scalarproduct(5)

    h2 = tau4 * vs_1_4
    h3 = tau4 * vs_2_4

    vs_3_4 = scalarproduct(6)

    h4 = tau4 * vs_3_4

    w(1:nq) = w(1:nq) * tau4 - ( z(1:nq) * h4 + y(1:nq) * h3 + x(1:nq) * h2)

    tau5 = hh(1,5)
    vs_1_5 = scalarproduct(7)
    vs_2_5 = scalarproduct(8)

    h2 = tau5 * vs_1_5
    h3 = tau5 * vs_2_5

    vs_3_5 = scalarproduct(9)
    vs_4_5 = scalarproduct(10)

    h4 = tau5 * vs_3_5
    h5 = tau5 * vs_4_5

    v(1:nq) = v(1:nq) * tau5 - ( w(1:nq) * h5 + z(1:nq) * h4 + y(1:nq) * h3 + x(1:nq) * h2)

    tau6 = hh(1,6)
    vs_1_6 = scalarproduct(11)
    vs_2_6 = scalarproduct(12)

    h2 = tau6 * vs_1_6
    h3 = tau6 * vs_2_6

    vs_3_6 = scalarproduct(13)
    vs_4_6 = scalarproduct(14)
    vs_5_6 = scalarproduct(15)

    h4 = tau6 * vs_3_6
    h5 = tau6 * vs_4_6
    h6 = tau6 * vs_5_6

    t(1:nq) = t(1:nq) * tau6 - ( v(1:nq) * h6 + w(1:nq) * h5 + z(1:nq) * h4 + y(1:nq) * h3 + x(1:nq) * h2)

    q(1:nq,1) = q(1:nq,1) - t(1:nq)

    h6 = hh(2,6)

    q(1:nq,2) = q(1:nq,2) - (v(1:nq) + t(1:nq) * h6) 

    h5 = hh(2,5)
    h6 = hh(3,6)

    q(1:nq,3) = q(1:nq,3) - (w(1:nq) + v(1:nq) * h5 + t(1:nq) * h6) 

    h4 = hh(2,4)
    h5 = hh(3,5)
    h6 = hh(4,6)

    q(1:nq,4) = q(1:nq,4) - (z(1:nq) + w(1:nq) * h4 + v(1:nq) * h5 + t(1:nq) * h6)

    h3 = hh(2,3)
    h4 = hh(3,4)
    h5 = hh(4,5)
    h6 = hh(5,6)

    q(1:nq,5) =  q(1:nq,5) - (y(1:nq) + z(1:nq) * h3 + w(1:nq) * h4 + v(1:nq) * h5 + t(1:nq) * h6)

    h2 = hh(2,2)
    h3 = hh(3,3)
    h4 = hh(4,4)
    h5 = hh(5,5)
    h6 = hh(6,6)

    q(1:nq,6) = q(1:nq,6) - (x(1:nq) + y(1:nq) * h2 + z(1:nq) * h3 + w(1:nq) * h4 + v(1:nq) * h5 + t(1:nq) * h6)

    do i=7,nb
       h1 = hh(i-5,1)
       h2 = hh(i-4,2)
       h3 = hh(i-3,3)
       h4 = hh(i-2,4)
       h5 = hh(i-1,5)
       h6 = hh(i  ,6)

       q(1:nq,i) = q(1:nq,i) -(x(1:nq) * h1 + y(1:nq) * h2 + z(1:nq) * h3 + w(1:nq) * h4 + v(1:nq) * h5 + t(1:nq) * h6)
   enddo

   h1 = hh(nb-4,1)
   h2 = hh(nb-3,2)
   h3 = hh(nb-2,3)
   h4 = hh(nb-1,4)
   h5 = hh(nb  ,5)

   q(1:nq,nb+1) = q(1:nq,nb+1) -(x(1:nq) * h1 + y(1:nq) * h2 + z(1:nq) * h3 + w(1:nq) * h4 + v(1:nq) * h5)

   h1 = hh(nb-3,1)
   h2 = hh(nb-2,2)
   h3 = hh(nb-1,3)
   h4 = hh(nb  ,4)

   q(1:nq,nb+2) = q(1:nq,nb+2) - (x(1:nq) * h1 + y(1:nq) * h2 + z(1:nq) * h3 + w(1:nq) * h4)

   h1 = hh(nb-2,1)
   h2 = hh(nb-1,2)
   h3 = hh(nb  ,3)

   q(1:nq,nb+3) = q(1:nq,nb+3) - (x(1:nq) * h1 + y(1:nq) * h2 + z(1:nq) * h3)
   h1 = hh(nb-1,1)
   h2 = hh(nb  ,2)

   q(1:nq,nb+4) = q(1:nq,nb+4) - (x(1:nq) * h1 +y(1:nq) * h2)

   h1 = hh(nb ,1)

   q(1:nq,nb+5) = q(1:nq,nb+5) - (x(1:nq) * h1)

  end subroutine
