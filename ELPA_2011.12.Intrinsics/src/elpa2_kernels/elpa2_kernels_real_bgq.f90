! --------------------------------------------------------------------------------------------------
!
! This file contains the compute intensive kernels for the Householder transformations.
!
! *** Special IBM BlueGene/Q version with QPX intrinsics in Fortran ***
! 
! Copyright of the original code rests with the authors inside the ELPA
! consortium. The copyright of any additional modifications shall rest
! with their original authors, but shall adhere to the licensing terms
! distributed along with the original code in the file "COPYING".
!
! --------------------------------------------------------------------------------------------------

subroutine double_hh_trafo_2hv(q, hh, nb, nq, ldq, ldh)

   implicit none

   integer, intent(in) :: nb, nq, ldq, ldh
   real*8, intent(inout) :: q(ldq,*)
   real*8, intent(in) :: hh(ldh,*)

   real*8 s
   integer i

   ! Safety only:

   if(mod(ldq,4) /= 0) STOP 'double_hh_trafo: ldq not divisible by 4!'

   call alignx(32,q)

   ! Calculate dot product of the two Householder vectors

   s = hh(2,2)*1
   do i=3,nb
      s = s+hh(i,2)*hh(i-1,1)
   enddo
   
   do i=1,nq-20,24
      call hh_trafo_kernel_24_bgq(q(i   ,1), hh, nb, ldq, ldh, s)
   enddo
   
   if(nq-i+1 > 16) then
      call hh_trafo_kernel_16_bgq(q(i  ,1), hh, nb, ldq, ldh, s)
      call hh_trafo_kernel_4_bgq(q(i+16,1), hh, nb, ldq, ldh, s)
   else if(nq-i+1 > 12) then
      call hh_trafo_kernel_8_bgq(q(i  ,1), hh, nb, ldq, ldh, s)
      call hh_trafo_kernel_8_bgq(q(i+8,1), hh, nb, ldq, ldh, s)
   else if(nq-i+1 > 8) then
      call hh_trafo_kernel_8_bgq(q(i  ,1), hh, nb, ldq, ldh, s)
      call hh_trafo_kernel_4_bgq(q(i+8,1), hh, nb, ldq, ldh, s)
   else if(nq-i+1 > 4) then
      call hh_trafo_kernel_8_bgq(q(i  ,1), hh, nb, ldq, ldh, s)
   else if(nq-i+1 > 0) then
      call hh_trafo_kernel_4_bgq(q(i  ,1), hh, nb, ldq, ldh, s)
   endif
   
end


! --------------------------------------------------------------------------------------------------
! The following kernels perform the Householder transformation on Q for 24/16/8/4 rows.
! --------------------------------------------------------------------------------------------------

subroutine hh_trafo_kernel_24_bgq(q, hh, nb, ldq, ldh, s)

   implicit none

   include 'mpif.h'
   
   integer, intent(in) :: nb, ldq, ldh
   
   real*8, intent(inout)::q(ldq,*)
   real*8, intent(in)::hh(ldh,*), s
   
   VECTOR(REAL(8))::QPX_x1, QPX_x2, QPX_x3, QPX_x4, QPX_x5, QPX_x6
   VECTOR(REAL(8))::QPX_y1, QPX_y2, QPX_y3, QPX_y4, QPX_y5, QPX_y6
   VECTOR(REAL(8))::QPX_q1, QPX_q2, QPX_q3, QPX_q4, QPX_q5, QPX_q6
   VECTOR(REAL(8))::QPX_h1, QPX_h2, QPX_tau1, QPX_tau2, QPX_s
   integer i

   call alignx(32,q)

   !--- multiply Householder vectors with matrix q ---

   QPX_x1 = VEC_LD(0,q(1,2))
   QPX_x2 = VEC_LD(0,q(5,2))
   QPX_x3 = VEC_LD(0,q(9,2))
   QPX_x4 = VEC_LD(0,q(13,2))
   QPX_x5 = VEC_LD(0,q(17,2))
   QPX_x6 = VEC_LD(0,q(21,2))
   
   QPX_h2 = VEC_SPLATS(hh(2,2))
   QPX_q1 = VEC_LD(0,q(1,1))
   QPX_q2 = VEC_LD(0,q(5,1))
   QPX_q3 = VEC_LD(0,q(9,1))
   QPX_q4 = VEC_LD(0,q(13,1))
   QPX_q5 = VEC_LD(0,q(17,1))
   QPX_q6 = VEC_LD(0,q(21,1))
   QPX_y1 = VEC_MADD(QPX_x1, QPX_h2, QPX_q1)
   QPX_y2 = VEC_MADD(QPX_x2, QPX_h2, QPX_q2)
   QPX_y3 = VEC_MADD(QPX_x3, QPX_h2, QPX_q3)
   QPX_y4 = VEC_MADD(QPX_x4, QPX_h2, QPX_q4)
   QPX_y5 = VEC_MADD(QPX_x5, QPX_h2, QPX_q5)
   QPX_y6 = VEC_MADD(QPX_x6, QPX_h2, QPX_q6)

   do i=3,nb,1

      QPX_q1 = VEC_LD(0,q(1,i))
      QPX_q2 = VEC_LD(0,q(5,i))
      QPX_q3 = VEC_LD(0,q(9,i))
      QPX_q4 = VEC_LD(0,q(13,i))
      QPX_q5 = VEC_LD(0,q(17,i))
      QPX_q6 = VEC_LD(0,q(21,i))
      QPX_h1 = VEC_SPLATS(hh(i-1,1))
      QPX_x1 = VEC_MADD(QPX_q1, QPX_h1, QPX_x1)
      QPX_x2 = VEC_MADD(QPX_q2, QPX_h1, QPX_x2)
      QPX_x3 = VEC_MADD(QPX_q3, QPX_h1, QPX_x3)
      QPX_x4 = VEC_MADD(QPX_q4, QPX_h1, QPX_x4)
      QPX_x5 = VEC_MADD(QPX_q5, QPX_h1, QPX_x5)
      QPX_x6 = VEC_MADD(QPX_q6, QPX_h1, QPX_x6)
      QPX_h2 = VEC_SPLATS(hh(i,2))
      QPX_y1 = VEC_MADD(QPX_q1, QPX_h2, QPX_y1)
      QPX_y2 = VEC_MADD(QPX_q2, QPX_h2, QPX_y2)
      QPX_y3 = VEC_MADD(QPX_q3, QPX_h2, QPX_y3)
      QPX_y4 = VEC_MADD(QPX_q4, QPX_h2, QPX_y4)
      QPX_y5 = VEC_MADD(QPX_q5, QPX_h2, QPX_y5)
      QPX_y6 = VEC_MADD(QPX_q6, QPX_h2, QPX_y6)

   enddo

   QPX_h1 = VEC_SPLATS(hh(nb,1))
   QPX_q1 = VEC_LD(0,q(1,nb+1))
   QPX_q2 = VEC_LD(0,q(5,nb+1))
   QPX_q3 = VEC_LD(0,q(9,nb+1))
   QPX_q4 = VEC_LD(0,q(13,nb+1))
   QPX_q5 = VEC_LD(0,q(17,nb+1))
   QPX_q6 = VEC_LD(0,q(21,nb+1))
   QPX_x1 = VEC_MADD(QPX_q1, QPX_h1, QPX_x1)
   QPX_x2 = VEC_MADD(QPX_q2, QPX_h1, QPX_x2)
   QPX_x3 = VEC_MADD(QPX_q3, QPX_h1, QPX_x3)
   QPX_x4 = VEC_MADD(QPX_q4, QPX_h1, QPX_x4)
   QPX_x5 = VEC_MADD(QPX_q5, QPX_h1, QPX_x5)
   QPX_x6 = VEC_MADD(QPX_q6, QPX_h1, QPX_x6)
   
   !--- multiply T matrix ---
   
   QPX_tau1 = VEC_SPLATS(-hh(1,1))
   QPX_x1 = VEC_MUL(QPX_x1, QPX_tau1)
   QPX_x2 = VEC_MUL(QPX_x2, QPX_tau1)
   QPX_x3 = VEC_MUL(QPX_x3, QPX_tau1)
   QPX_x4 = VEC_MUL(QPX_x4, QPX_tau1)
   QPX_x5 = VEC_MUL(QPX_x5, QPX_tau1)
   QPX_x6 = VEC_MUL(QPX_x6, QPX_tau1)
   QPX_tau2 = VEC_SPLATS(-hh(1,2))
   QPX_s = VEC_SPLATS(-hh(1,2)*s)
   QPX_y1 = VEC_MUL(QPX_y1, QPX_tau2)
   QPX_y2 = VEC_MUL(QPX_y2, QPX_tau2)
   QPX_y3 = VEC_MUL(QPX_y3, QPX_tau2)
   QPX_y4 = VEC_MUL(QPX_y4, QPX_tau2)
   QPX_y5 = VEC_MUL(QPX_y5, QPX_tau2)
   QPX_y6 = VEC_MUL(QPX_y6, QPX_tau2)
   QPX_y1 = VEC_MADD(QPX_x1, QPX_s, QPX_y1)
   QPX_y2 = VEC_MADD(QPX_x2, QPX_s, QPX_y2)
   QPX_y3 = VEC_MADD(QPX_x3, QPX_s, QPX_y3)
   QPX_y4 = VEC_MADD(QPX_x4, QPX_s, QPX_y4)
   QPX_y5 = VEC_MADD(QPX_x5, QPX_s, QPX_y5)
   QPX_y6 = VEC_MADD(QPX_x6, QPX_s, QPX_y6)

   !--- rank-2 update of q ---

   QPX_q1 = VEC_LD(0,q(1,1))
   QPX_q2 = VEC_LD(0,q(5,1))
   QPX_q3 = VEC_LD(0,q(9,1))
   QPX_q4 = VEC_LD(0,q(13,1))
   QPX_q5 = VEC_LD(0,q(17,1))
   QPX_q6 = VEC_LD(0,q(21,1))
   QPX_q1 = VEC_ADD(QPX_q1, QPX_y1)
   QPX_q2 = VEC_ADD(QPX_q2, QPX_y2)
   QPX_q3 = VEC_ADD(QPX_q3, QPX_y3)
   QPX_q4 = VEC_ADD(QPX_q4, QPX_y4)
   QPX_q5 = VEC_ADD(QPX_q5, QPX_y5)
   QPX_q6 = VEC_ADD(QPX_q6, QPX_y6)
   call VEC_ST(QPX_q1, 0, q(1,1))
   call VEC_ST(QPX_q2, 0, q(5,1))
   call VEC_ST(QPX_q3, 0, q(9,1))
   call VEC_ST(QPX_q4, 0, q(13,1))
   call VEC_ST(QPX_q5, 0, q(17,1))
   call VEC_ST(QPX_q6, 0, q(21,1))
   
   QPX_h2 = VEC_SPLATS(hh(2,2))
   QPX_q1 = VEC_LD(0,q(1,2))
   QPX_q2 = VEC_LD(0,q(5,2))
   QPX_q3 = VEC_LD(0,q(9,2))
   QPX_q4 = VEC_LD(0,q(13,2))
   QPX_q5 = VEC_LD(0,q(17,2))
   QPX_q6 = VEC_LD(0,q(21,2))
   QPX_q1 = VEC_MADD(QPX_y1, QPX_h2, QPX_q1)
   QPX_q2 = VEC_MADD(QPX_y2, QPX_h2, QPX_q2)
   QPX_q3 = VEC_MADD(QPX_y3, QPX_h2, QPX_q3)
   QPX_q4 = VEC_MADD(QPX_y4, QPX_h2, QPX_q4)
   QPX_q5 = VEC_MADD(QPX_y5, QPX_h2, QPX_q5)
   QPX_q6 = VEC_MADD(QPX_y6, QPX_h2, QPX_q6)
   QPX_q1 = VEC_ADD(QPX_q1, QPX_x1)
   QPX_q2 = VEC_ADD(QPX_q2, QPX_x2)
   QPX_q3 = VEC_ADD(QPX_q3, QPX_x3)
   QPX_q4 = VEC_ADD(QPX_q4, QPX_x4)
   QPX_q5 = VEC_ADD(QPX_q5, QPX_x5)
   QPX_q6 = VEC_ADD(QPX_q6, QPX_x6)
   call VEC_ST(QPX_q1, 0, q(1,2))
   call VEC_ST(QPX_q2, 0, q(5,2))
   call VEC_ST(QPX_q3, 0, q(9,2))
   call VEC_ST(QPX_q4, 0, q(13,2))
   call VEC_ST(QPX_q5, 0, q(17,2))
   call VEC_ST(QPX_q6, 0, q(21,2))
   
   do i=3,nb,1

      QPX_q1 = VEC_LD(0,q(1,i))
      QPX_q2 = VEC_LD(0,q(5,i))
      QPX_q3 = VEC_LD(0,q(9,i))
      QPX_q4 = VEC_LD(0,q(13,i))
      QPX_q5 = VEC_LD(0,q(17,i))
      QPX_q6 = VEC_LD(0,q(21,i))
      QPX_h1 = VEC_SPLATS(hh(i-1,1))
      QPX_q1 = VEC_MADD(QPX_x1, QPX_h1, QPX_q1)
      QPX_q2 = VEC_MADD(QPX_x2, QPX_h1, QPX_q2)
      QPX_q3 = VEC_MADD(QPX_x3, QPX_h1, QPX_q3)
      QPX_q4 = VEC_MADD(QPX_x4, QPX_h1, QPX_q4)
      QPX_q5 = VEC_MADD(QPX_x5, QPX_h1, QPX_q5)
      QPX_q6 = VEC_MADD(QPX_x6, QPX_h1, QPX_q6)
      QPX_h2 = VEC_SPLATS(hh(i,2))
      QPX_q1 = VEC_MADD(QPX_y1, QPX_h2, QPX_q1)
      QPX_q2 = VEC_MADD(QPX_y2, QPX_h2, QPX_q2)
      QPX_q3 = VEC_MADD(QPX_y3, QPX_h2, QPX_q3)
      QPX_q4 = VEC_MADD(QPX_y4, QPX_h2, QPX_q4)
      QPX_q5 = VEC_MADD(QPX_y5, QPX_h2, QPX_q5)
      QPX_q6 = VEC_MADD(QPX_y6, QPX_h2, QPX_q6)
      
      call VEC_ST(QPX_q1, 0, q(1,i))
      call VEC_ST(QPX_q2, 0, q(5,i))
      call VEC_ST(QPX_q3, 0, q(9,i))
      call VEC_ST(QPX_q4, 0, q(13,i))
      call VEC_ST(QPX_q5, 0, q(17,i))
      call VEC_ST(QPX_q6, 0, q(21,i))

   enddo

   QPX_h1 = VEC_SPLATS(hh(nb,1))
   QPX_q1 = VEC_LD(0,q(1,nb+1))
   QPX_q2 = VEC_LD(0,q(5,nb+1))
   QPX_q3 = VEC_LD(0,q(9,nb+1))
   QPX_q4 = VEC_LD(0,q(13,nb+1))
   QPX_q5 = VEC_LD(0,q(17,nb+1))
   QPX_q6 = VEC_LD(0,q(21,nb+1))
   QPX_q1 = VEC_MADD(QPX_x1, QPX_h1, QPX_q1)
   QPX_q2 = VEC_MADD(QPX_x2, QPX_h1, QPX_q2)
   QPX_q3 = VEC_MADD(QPX_x3, QPX_h1, QPX_q3)
   QPX_q4 = VEC_MADD(QPX_x4, QPX_h1, QPX_q4)
   QPX_q5 = VEC_MADD(QPX_x5, QPX_h1, QPX_q5)
   QPX_q6 = VEC_MADD(QPX_x6, QPX_h1, QPX_q6)
   call VEC_ST(QPX_q1, 0, q(1,nb+1))
   call VEC_ST(QPX_q2, 0, q(5,nb+1))
   call VEC_ST(QPX_q3, 0, q(9,nb+1))
   call VEC_ST(QPX_q4, 0, q(13,nb+1))
   call VEC_ST(QPX_q5, 0, q(17,nb+1))
   call VEC_ST(QPX_q6, 0, q(21,nb+1))

end subroutine

! --------------------------------------------------------------------------------------------------

subroutine hh_trafo_kernel_16_bgq(q, hh, nb, ldq, ldh, s)

   implicit none

   include 'mpif.h'
   
   integer, intent(in) :: nb, ldq, ldh
   
   real*8, intent(inout)::q(ldq,*)
   real*8, intent(in)::hh(ldh,*), s
   
   VECTOR(REAL(8))::QPX_x1, QPX_x2, QPX_x3, QPX_x4
   VECTOR(REAL(8))::QPX_y1, QPX_y2, QPX_y3, QPX_y4
   VECTOR(REAL(8))::QPX_q1, QPX_q2, QPX_q3, QPX_q4
   VECTOR(REAL(8))::QPX_h1, QPX_h2, QPX_tau1, QPX_tau2, QPX_s
   integer i

   call alignx(32,q)

   !--- multiply Householder vectors with matrix q ---

   QPX_x1 = VEC_LD(0,q(1,2))
   QPX_x2 = VEC_LD(0,q(5,2))
   QPX_x3 = VEC_LD(0,q(9,2))
   QPX_x4 = VEC_LD(0,q(13,2))
   
   QPX_h2 = VEC_SPLATS(hh(2,2))
   QPX_q1 = VEC_LD(0,q(1,1))
   QPX_q2 = VEC_LD(0,q(5,1))
   QPX_q3 = VEC_LD(0,q(9,1))
   QPX_q4 = VEC_LD(0,q(13,1))
   QPX_y1 = VEC_MADD(QPX_x1, QPX_h2, QPX_q1)
   QPX_y2 = VEC_MADD(QPX_x2, QPX_h2, QPX_q2)
   QPX_y3 = VEC_MADD(QPX_x3, QPX_h2, QPX_q3)
   QPX_y4 = VEC_MADD(QPX_x4, QPX_h2, QPX_q4)

   do i=3,nb,1

      QPX_q1 = VEC_LD(0,q(1,i))
      QPX_q2 = VEC_LD(0,q(5,i))
      QPX_q3 = VEC_LD(0,q(9,i))
      QPX_q4 = VEC_LD(0,q(13,i))
      QPX_h1 = VEC_SPLATS(hh(i-1,1))
      QPX_x1 = VEC_MADD(QPX_q1, QPX_h1, QPX_x1)
      QPX_x2 = VEC_MADD(QPX_q2, QPX_h1, QPX_x2)
      QPX_x3 = VEC_MADD(QPX_q3, QPX_h1, QPX_x3)
      QPX_x4 = VEC_MADD(QPX_q4, QPX_h1, QPX_x4)
      QPX_h2 = VEC_SPLATS(hh(i,2))
      QPX_y1 = VEC_MADD(QPX_q1, QPX_h2, QPX_y1)
      QPX_y2 = VEC_MADD(QPX_q2, QPX_h2, QPX_y2)
      QPX_y3 = VEC_MADD(QPX_q3, QPX_h2, QPX_y3)
      QPX_y4 = VEC_MADD(QPX_q4, QPX_h2, QPX_y4)

   enddo

   QPX_h1 = VEC_SPLATS(hh(nb,1))
   QPX_q1 = VEC_LD(0,q(1,nb+1))
   QPX_q2 = VEC_LD(0,q(5,nb+1))
   QPX_q3 = VEC_LD(0,q(9,nb+1))
   QPX_q4 = VEC_LD(0,q(13,nb+1))
   QPX_x1 = VEC_MADD(QPX_q1, QPX_h1, QPX_x1)
   QPX_x2 = VEC_MADD(QPX_q2, QPX_h1, QPX_x2)
   QPX_x3 = VEC_MADD(QPX_q3, QPX_h1, QPX_x3)
   QPX_x4 = VEC_MADD(QPX_q4, QPX_h1, QPX_x4)
   
   !--- multiply T matrix ---
   
   QPX_tau1 = VEC_SPLATS(-hh(1,1))
   QPX_x1 = VEC_MUL(QPX_x1, QPX_tau1)
   QPX_x2 = VEC_MUL(QPX_x2, QPX_tau1)
   QPX_x3 = VEC_MUL(QPX_x3, QPX_tau1)
   QPX_x4 = VEC_MUL(QPX_x4, QPX_tau1)
   QPX_tau2 = VEC_SPLATS(-hh(1,2))
   QPX_s = VEC_SPLATS(-hh(1,2)*s)
   QPX_y1 = VEC_MUL(QPX_y1, QPX_tau2)
   QPX_y2 = VEC_MUL(QPX_y2, QPX_tau2)
   QPX_y3 = VEC_MUL(QPX_y3, QPX_tau2)
   QPX_y4 = VEC_MUL(QPX_y4, QPX_tau2)
   QPX_y1 = VEC_MADD(QPX_x1, QPX_s, QPX_y1)
   QPX_y2 = VEC_MADD(QPX_x2, QPX_s, QPX_y2)
   QPX_y3 = VEC_MADD(QPX_x3, QPX_s, QPX_y3)
   QPX_y4 = VEC_MADD(QPX_x4, QPX_s, QPX_y4)

   !--- rank-2 update of q ---

   QPX_q1 = VEC_LD(0,q(1,1))
   QPX_q2 = VEC_LD(0,q(5,1))
   QPX_q3 = VEC_LD(0,q(9,1))
   QPX_q4 = VEC_LD(0,q(13,1))
   QPX_q1 = VEC_ADD(QPX_q1, QPX_y1)
   QPX_q2 = VEC_ADD(QPX_q2, QPX_y2)
   QPX_q3 = VEC_ADD(QPX_q3, QPX_y3)
   QPX_q4 = VEC_ADD(QPX_q4, QPX_y4)
   call VEC_ST(QPX_q1, 0, q(1,1))
   call VEC_ST(QPX_q2, 0, q(5,1))
   call VEC_ST(QPX_q3, 0, q(9,1))
   call VEC_ST(QPX_q4, 0, q(13,1))
   
   QPX_h2 = VEC_SPLATS(hh(2,2))
   QPX_q1 = VEC_LD(0,q(1,2))
   QPX_q2 = VEC_LD(0,q(5,2))
   QPX_q3 = VEC_LD(0,q(9,2))
   QPX_q4 = VEC_LD(0,q(13,2))
   QPX_q1 = VEC_MADD(QPX_y1, QPX_h2, QPX_q1)
   QPX_q2 = VEC_MADD(QPX_y2, QPX_h2, QPX_q2)
   QPX_q3 = VEC_MADD(QPX_y3, QPX_h2, QPX_q3)
   QPX_q4 = VEC_MADD(QPX_y4, QPX_h2, QPX_q4)
   QPX_q1 = VEC_ADD(QPX_q1, QPX_x1)
   QPX_q2 = VEC_ADD(QPX_q2, QPX_x2)
   QPX_q3 = VEC_ADD(QPX_q3, QPX_x3)
   QPX_q4 = VEC_ADD(QPX_q4, QPX_x4)
   call VEC_ST(QPX_q1, 0, q(1,2))
   call VEC_ST(QPX_q2, 0, q(5,2))
   call VEC_ST(QPX_q3, 0, q(9,2))
   call VEC_ST(QPX_q4, 0, q(13,2))
   
   do i=3,nb,1

      QPX_q1 = VEC_LD(0,q(1,i))
      QPX_q2 = VEC_LD(0,q(5,i))
      QPX_q3 = VEC_LD(0,q(9,i))
      QPX_q4 = VEC_LD(0,q(13,i))
      QPX_h1 = VEC_SPLATS(hh(i-1,1))
      QPX_q1 = VEC_MADD(QPX_x1, QPX_h1, QPX_q1)
      QPX_q2 = VEC_MADD(QPX_x2, QPX_h1, QPX_q2)
      QPX_q3 = VEC_MADD(QPX_x3, QPX_h1, QPX_q3)
      QPX_q4 = VEC_MADD(QPX_x4, QPX_h1, QPX_q4)
      QPX_h2 = VEC_SPLATS(hh(i,2))
      QPX_q1 = VEC_MADD(QPX_y1, QPX_h2, QPX_q1)
      QPX_q2 = VEC_MADD(QPX_y2, QPX_h2, QPX_q2)
      QPX_q3 = VEC_MADD(QPX_y3, QPX_h2, QPX_q3)
      QPX_q4 = VEC_MADD(QPX_y4, QPX_h2, QPX_q4)
      
      call VEC_ST(QPX_q1, 0, q(1,i))
      call VEC_ST(QPX_q2, 0, q(5,i))
      call VEC_ST(QPX_q3, 0, q(9,i))
      call VEC_ST(QPX_q4, 0, q(13,i))

   enddo

   QPX_h1 = VEC_SPLATS(hh(nb,1))
   QPX_q1 = VEC_LD(0,q(1,nb+1))
   QPX_q2 = VEC_LD(0,q(5,nb+1))
   QPX_q3 = VEC_LD(0,q(9,nb+1))
   QPX_q4 = VEC_LD(0,q(13,nb+1))
   QPX_q1 = VEC_MADD(QPX_x1, QPX_h1, QPX_q1)
   QPX_q2 = VEC_MADD(QPX_x2, QPX_h1, QPX_q2)
   QPX_q3 = VEC_MADD(QPX_x3, QPX_h1, QPX_q3)
   QPX_q4 = VEC_MADD(QPX_x4, QPX_h1, QPX_q4)
   call VEC_ST(QPX_q1, 0, q(1,nb+1))
   call VEC_ST(QPX_q2, 0, q(5,nb+1))
   call VEC_ST(QPX_q3, 0, q(9,nb+1))
   call VEC_ST(QPX_q4, 0, q(13,nb+1))

end subroutine

! --------------------------------------------------------------------------------------------------

subroutine hh_trafo_kernel_8_bgq(q, hh, nb, ldq, ldh, s)

   implicit none

   include 'mpif.h'
   
   integer, intent(in) :: nb, ldq, ldh
   
   real*8, intent(inout)::q(ldq,*)
   real*8, intent(in)::hh(ldh,*), s
   
   VECTOR(REAL(8))::QPX_x1, QPX_x2, QPX_y1, QPX_y2
   VECTOR(REAL(8))::QPX_q1, QPX_q2
   VECTOR(REAL(8))::QPX_h1, QPX_h2, QPX_tau1, QPX_tau2, QPX_s
   integer i

   call alignx(32,q)

   !--- multiply Householder vectors with matrix q ---

   QPX_x1 = VEC_LD(0,q(1,2))
   QPX_x2 = VEC_LD(0,q(5,2))
   
   QPX_h2 = VEC_SPLATS(hh(2,2))
   QPX_q1 = VEC_LD(0,q(1,1))
   QPX_q2 = VEC_LD(0,q(5,1))
   QPX_y1 = VEC_MADD(QPX_x1, QPX_h2, QPX_q1)
   QPX_y2 = VEC_MADD(QPX_x2, QPX_h2, QPX_q2)

   do i=3,nb,1

      QPX_q1 = VEC_LD(0,q(1,i))
      QPX_q2 = VEC_LD(0,q(5,i))
      QPX_h1 = VEC_SPLATS(hh(i-1,1))
      QPX_x1 = VEC_MADD(QPX_q1, QPX_h1, QPX_x1)
      QPX_x2 = VEC_MADD(QPX_q2, QPX_h1, QPX_x2)
      QPX_h2 = VEC_SPLATS(hh(i,2))
      QPX_y1 = VEC_MADD(QPX_q1, QPX_h2, QPX_y1)
      QPX_y2 = VEC_MADD(QPX_q2, QPX_h2, QPX_y2)

   enddo

   QPX_h1 = VEC_SPLATS(hh(nb,1))
   QPX_q1 = VEC_LD(0,q(1,nb+1))
   QPX_q2 = VEC_LD(0,q(5,nb+1))
   QPX_x1 = VEC_MADD(QPX_q1, QPX_h1, QPX_x1)
   QPX_x2 = VEC_MADD(QPX_q2, QPX_h1, QPX_x2)
   
   !--- multiply T matrix ---
   
   QPX_tau1 = VEC_SPLATS(-hh(1,1))
   QPX_x1 = VEC_MUL(QPX_x1, QPX_tau1)
   QPX_x2 = VEC_MUL(QPX_x2, QPX_tau1)
   QPX_tau2 = VEC_SPLATS(-hh(1,2))
   QPX_s = VEC_SPLATS(-hh(1,2)*s)
   QPX_y1 = VEC_MUL(QPX_y1, QPX_tau2)
   QPX_y2 = VEC_MUL(QPX_y2, QPX_tau2)
   QPX_y1 = VEC_MADD(QPX_x1, QPX_s, QPX_y1)
   QPX_y2 = VEC_MADD(QPX_x2, QPX_s, QPX_y2)

   !--- rank-2 update of q ---

   QPX_q1 = VEC_LD(0,q(1,1))
   QPX_q2 = VEC_LD(0,q(5,1))
   QPX_q1 = VEC_ADD(QPX_q1, QPX_y1)
   QPX_q2 = VEC_ADD(QPX_q2, QPX_y2)
   call VEC_ST(QPX_q1, 0, q(1,1))
   call VEC_ST(QPX_q2, 0, q(5,1))
   
   QPX_h2 = VEC_SPLATS(hh(2,2))
   QPX_q1 = VEC_LD(0,q(1,2))
   QPX_q2 = VEC_LD(0,q(5,2))
   QPX_q1 = VEC_MADD(QPX_y1, QPX_h2, QPX_q1)
   QPX_q2 = VEC_MADD(QPX_y2, QPX_h2, QPX_q2)
   QPX_q1 = VEC_ADD(QPX_q1, QPX_x1)
   QPX_q2 = VEC_ADD(QPX_q2, QPX_x2)
   call VEC_ST(QPX_q1, 0, q(1,2))
   call VEC_ST(QPX_q2, 0, q(5,2))
   
   do i=3,nb,1

      QPX_q1 = VEC_LD(0,q(1,i))
      QPX_q2 = VEC_LD(0,q(5,i))
      QPX_h1 = VEC_SPLATS(hh(i-1,1))
      QPX_q1 = VEC_MADD(QPX_x1, QPX_h1, QPX_q1)
      QPX_q2 = VEC_MADD(QPX_x2, QPX_h1, QPX_q2)
      QPX_h2 = VEC_SPLATS(hh(i,2))
      QPX_q1 = VEC_MADD(QPX_y1, QPX_h2, QPX_q1)
      QPX_q2 = VEC_MADD(QPX_y2, QPX_h2, QPX_q2)
      
      call VEC_ST(QPX_q1, 0, q(1,i))
      call VEC_ST(QPX_q2, 0, q(5,i))

   enddo

   QPX_h1 = VEC_SPLATS(hh(nb,1))
   QPX_q1 = VEC_LD(0,q(1,nb+1))
   QPX_q2 = VEC_LD(0,q(5,nb+1))
   QPX_q1 = VEC_MADD(QPX_x1, QPX_h1, QPX_q1)
   QPX_q2 = VEC_MADD(QPX_x2, QPX_h1, QPX_q2)
   call VEC_ST(QPX_q1, 0, q(1,nb+1))
   call VEC_ST(QPX_q2, 0, q(5,nb+1))

end subroutine

! --------------------------------------------------------------------------------------------------

subroutine hh_trafo_kernel_4_bgq(q, hh, nb, ldq, ldh, s)

   implicit none

   include 'mpif.h'
   
   integer, intent(in) :: nb, ldq, ldh
   
   real*8, intent(inout)::q(ldq,*)
   real*8, intent(in)::hh(ldh,*), s
   
   VECTOR(REAL(8))::QPX_x1, QPX_y1
   VECTOR(REAL(8))::QPX_q1
   VECTOR(REAL(8))::QPX_h1, QPX_h2, QPX_tau1, QPX_tau2, QPX_s
   integer i

   call alignx(32,q)

   !--- multiply Householder vectors with matrix q ---

   QPX_x1 = VEC_LD(0,q(1,2))
   
   QPX_h2 = VEC_SPLATS(hh(2,2))
   QPX_q1 = VEC_LD(0,q(1,1))
   QPX_y1 = VEC_MADD(QPX_x1, QPX_h2, QPX_q1)
   
   do i=3,nb,1
   
      QPX_q1 = VEC_LD(0,q(1,i))
      QPX_h1 = VEC_SPLATS(hh(i-1,1))
      QPX_x1 = VEC_MADD(QPX_q1, QPX_h1, QPX_x1)
      QPX_h2 = VEC_SPLATS(hh(i,2))
      QPX_y1 = VEC_MADD(QPX_q1, QPX_h2, QPX_y1)
   
   enddo
   
   QPX_h1 = VEC_SPLATS(hh(nb,1))
   QPX_q1 = VEC_LD(0,q(1,nb+1))
   QPX_x1 = VEC_MADD(QPX_q1, QPX_h1, QPX_x1)
   
   !--- multiply T matrix ---
   
   QPX_tau1 = VEC_SPLATS(-hh(1,1))
   QPX_x1 = VEC_MUL(QPX_x1, QPX_tau1)
   QPX_tau2 = VEC_SPLATS(-hh(1,2))
   QPX_s = VEC_SPLATS(-hh(1,2)*s)
   QPX_y1 = VEC_MUL(QPX_y1, QPX_tau2)
   QPX_y1 = VEC_MADD(QPX_x1, QPX_s, QPX_y1)

   !--- rank-2 update of q ---

   QPX_q1 = VEC_LD(0,q(1,1))
   QPX_q1 = VEC_ADD(QPX_q1, QPX_y1)
   call VEC_ST(QPX_q1, 0, q(1,1))
   
   QPX_h2 = VEC_SPLATS(hh(2,2))
   QPX_q1 = VEC_LD(0,q(1,2))
   QPX_q1 = VEC_MADD(QPX_y1, QPX_h2, QPX_q1)
   QPX_q1 = VEC_ADD(QPX_q1, QPX_x1)
   call VEC_ST(QPX_q1, 0, q(1,2))
   
   do i=3,nb,1

      QPX_q1 = VEC_LD(0,q(1,i))
      QPX_h1 = VEC_SPLATS(hh(i-1,1))
      QPX_q1 = VEC_MADD(QPX_x1, QPX_h1, QPX_q1)
      QPX_h2 = VEC_SPLATS(hh(i,2))
      QPX_q1 = VEC_MADD(QPX_y1, QPX_h2, QPX_q1)
      
      call VEC_ST(QPX_q1, 0, q(1,i))

   enddo

   QPX_h1 = VEC_SPLATS(hh(nb,1))
   QPX_q1 = VEC_LD(0,q(1,nb+1))
   QPX_q1 = VEC_MADD(QPX_x1, QPX_h1, QPX_q1)
   call VEC_ST(QPX_q1, 0, q(1,nb+1))

end subroutine

! --------------------------------------------------------------------------------------------------
