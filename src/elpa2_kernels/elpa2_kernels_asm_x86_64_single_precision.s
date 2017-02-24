#    This file is part of ELPA.
#
#    The ELPA library was originally created by the ELPA consortium,
#    consisting of the following organizations:
#
#    - Max Planck Computing and Data Facility (MPCDF), formerly known as
#      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
#    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
#      Informatik,
#    - Technische Universität München, Lehrstuhl für Informatik mit
#      Schwerpunkt Wissenschaftliches Rechnen ,
#    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
#    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
#      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
#      and
#    - IBM Deutschland GmbH
#
#
#    More information can be found here:
#    http://elpa.mpcdf.mpg.de/
#
#    ELPA is free software: you can redistribute it and/or modify
#    it under the terms of the version 3 of the license of the
#    GNU Lesser General Public License as published by the Free
#    Software Foundation.
#
#    ELPA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
#
#    ELPA reflects a substantial effort on the part of the original
#    ELPA consortium, and we ask you to respect the spirit of the
#    license that we chose: i.e., please contribute any changes you
#    may have back to the original ELPA library distribution, and keep
#    any derivatives of ELPA under the same license that we chose for
#    the original distribution, the GNU Lesser General Public License.
#
# Author: Andreas Marek, MPCDF

        .globl double_hh_trafo_single
        .globl single_hh_trafo_complex_single

	.text
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

        .macro hh_trafo_real_single nrows

        # When this macro is called, the following registers are set and must not be changed
        # %rdi: Address of q
        # %rsi: Address of hh
        # %rdx: nb
        # %rcx: Remaining rows nq
        # %r8:  ldq in bytes
        # %r9:  ldh in bytes
        # %rax: address of hh at the end of the loops
        # The top of the stack must contain the dot product of the two Householder vectors

        movq      %rdi, %r10   # Copy address of q
        movq      %rsi, %r11   # Copy address of hh


#   x1 = q(1,2)
#   x2 = q(2,2)
#
#   y1 = q(1,1) + q(1,2)*hh(2,2)
#   y2 = q(2,1) + q(2,2)*hh(2,2)

# single precision implementation does not rely on complex packing !
        movaps      (%r10), %xmm6       # y1 = q(1,1) ; copy content (16 bytes) starting at address %r10 (q(1,1)) into xmm6 (16 bytes = first 4 single precision values)
                                        # y2 = q(2,1)
					# y3 = q(3,1)
					# y4 = q(4,1)
        .if \nrows>=8
        movaps    16(%r10), %xmm7       # y5 = q(5,1)  ; copy content od address r10+16 = q(5,1) into xmm7 (16 bytes = single precision values 5 and 6, 7, 8)
	                                # y6 = q(6,1)
					# y7 = q(7,1)
					# y8 = q(8,1)
        .if \nrows==12
        movaps    32(%r10), %xmm8       # y9  = q(9,1)  ; copy content od address r10+32 = q(9,1) into xmm8 (16 bytes = single precision values 9 ,10, 11, 12)
                                        # y10 = q(10,1)
					# y11 = q(11,1)
					# y12 = q(12,1)
        .endif
        .endif

        addq      %r8, %r10             # %r10 => q(.,2)   # add to r10 ldq -> r10 now is q(*,2) r10 = r10 + r8
	# carefull here ! we want to store in xmm9 four times the value of h(2,2) !
        movddup   4(%r11,%r9), %xmm13   #  hh(2,2)         # copy from starting address r11 ldh bytes into xmm13 (wolud be hh(1,2)) shift by 4 bytes hh(2,2) and duplicate; xmm13 contains h(2,2), h(3,2), h(2,2), h(3,2)
	movsldup   %xmm13, %xmm9        # copy the first 4 bytes (h(2,2)) and duplicate, the same for the third 4 bytes => xmm9 contains h(2,2), h(2,2), h(2,2), h(2,2)
#        movshdup   %xmm13, %xmm9

        .macro mac_pre_loop1_single qoff, X, Y
        movaps    \qoff(%r10), \X       # xn = q(n,2)          # x = r10 + qoff = q(1+qoff,2) ; x contains the values q(1+qoff,2), q(2+qoff,2) , q(3+qoff,2), q(4+qoff,2) (=4 single precision floats)
        movaps    \X, %xmm10                                   # copy x into xmm10 = q(1+qoff,2) .. q(4+qoff,2) = 4 single precision floats)
        mulps     %xmm9, %xmm10                                # multiply 4 single precision values xmm9 (four times h(2,2)) with four single precision values q(1+qoff,2)..q(4+qoff,2) stored in xmm10; store result in xmm10
        addps     %xmm10, \Y            # yn = yn + xn*h(2,2)  # add the four values in xmm10 (q(1+qoff,2)*h(2,2)..q(4+qoff,2)*h(2,2)) and \Y ; store in Y
        .endm

        mac_pre_loop1_single  0, %xmm0, %xmm6  # do the step y(1:4) = q(1:4,1) +q(1:4,2)*h(2,2) for the first 4 single precision floats
        .if \nrows>=8
        mac_pre_loop1_single 16, %xmm1, %xmm7 # for the next 4 floats
        .if \nrows==12
        mac_pre_loop1_single 32, %xmm2, %xmm8 # for the next 4 floats
        .endif
        .endif
        .purgem   mac_pre_loop1_single

#   do i=3,nb
#      h1 = hh(i-1,1)
#      h2 = hh(i,2)
#      x1 = x1 + q(1,i)*h1
#      y1 = y1 + q(1,i)*h2
#      x2 = x2 + q(2,i)*h1
#      y2 = y2 + q(2,i)*h2
#      ...
#   enddo

        addq      $4, %r11              # r11 points to hh(1,1) + 4 bytes = hh(2,1)
        .align 16
1:
        cmpq %rax, %r11                 # Jump out of the loop if %r11 >= %rax
        jge       2f

        addq      %r8, %r10             # advance i %r10 => q(.,i)
        # careful here we want xmm11 to contain four times the value of hh(i-1,1)
        movddup   (%r11), %xmm13        # copy the first 8 bytes at r11 and duplicate ; xmm13 contains hh(i-1,1), hh(i,1), hh(i-1,1), hh(i,1)
       movsldup   %xmm13, %xmm11        # copy the first 4 bytes (h(i-1,1)) and duplicate, the same for the third 4 bytes => xmm11 contains h(i-1,1), h(i-1,1), h(i-1,1), h(i-1,1)
#        movshdup   %xmm13, %xmm11

        # carefull here we want xmm9 to contain four times the value of hh(i,2)
        movddup   4(%r11,%r9), %xmm13   # add to hh(i-1,1) ldh (r9) bytes => hh(i-1,2) add 4 extra bytes => hh(i,2) and duplicate ; xmm13 contains hh(i,2), hh(i+1,2), hh(i,2), hh(i+1,2)
       movsldup   %xmm13, %xmm9        # copy the first 4 bytes (h(i,2)) and duplicate, the same for the third 4 bytes => xmm9 contains h(i,2), h(i,2), h(i,2), h(i,2)
#        movshdup   %xmm13, %xmm9

        .macro mac_loop1_single qoff, X, Y
        movaps    \qoff(%r10), %xmm13   # q(.,i)  copy q(1,i), q(2,i), q(3,i), q(4,i) into xmm13
        movaps    %xmm13, %xmm10        # copy q(1,i), q(2,i), q(3,i) and q(4,i) into xmm10
        mulps     %xmm11, %xmm13        # multiply q(1,i), q(2,i), q(3,i), q(4,i) with  hh(i-1,i), h(i-1,1), h(i-1,1), h(i-1,1) ; store in xmm13
        addps     %xmm13, \X            # xn = xn + q(.,i)*h1 ; add to h1*q(.,i) the valye of x store in x
        mulps     %xmm9, %xmm10         # multiply hh(i,2), h(i,2), h(i,2), h(i,2) with q(1,i), q(2,i), q(3,i), q(4,i) store into xmm10
        addps     %xmm10, \Y            # yn = yn + q(.,i)*h2 ; add q(.,i)*h2 to Y store in y
        .endm

        mac_loop1_single  0, %xmm0, %xmm6
        .if \nrows>=8
        mac_loop1_single 16, %xmm1, %xmm7
        .if \nrows==12
        mac_loop1_single 32, %xmm2, %xmm8
        .endif
        .endif
        .purgem   mac_loop1_single

        addq      $4, %r11
        jmp       1b
2:

#   x1 = x1 + q(1,nb+1)*hh(nb,1)
#   x2 = x2 + q(2,nb+1)*hh(nb,1)

        addq      %r8, %r10             # %r10 => q(.,nb+1) # add ldq on q +> q(.,nb+1)
	# careful here we want xm11 to contain four times the value hh(nb,1)
        movddup   (%r11), %xmm13        # copy hh(nb,1) hh(nb+1,1) into xmm13 and duplicate
       movsldup   %xmm13, %xmm11       # copy the first 4 bytes (h(nb,1)) and duplicate, the same for the third 4 bytes => xmm11 contains h(nb,1), h(nb,1), h(nb,1), h(nb,1)
#        movshdup   %xmm13, %xmm11

        .macro mac_post_loop1_single qoff, X
        movaps    \qoff(%r10), %xmm13   # q(.,nb+1) copy q(1,nb+1), q(2,nb+1) q(3,nb+1), q(4,nb+1) into xmm13
        mulps     %xmm11, %xmm13        # multiply hh(nb,1) hh(nb,1) hh(nb,1) hh(nb,1) with q(1,nb+1), q(2,nb+1) q(3,nb+1), q(4,nb+1) store in xmm13
        addps     %xmm13, \X            # add hh(nb,1)*q(.,nb+1) and x store in x
        .endm

        mac_post_loop1_single  0, %xmm0
        .if \nrows>=8
        mac_post_loop1_single 16, %xmm1
        .if \nrows==12
        mac_post_loop1_single 32, %xmm2
        .endif
        .endif
        .purgem   mac_post_loop1_single

#   tau1 = hh(1,1)
#   tau2 = hh(1,2)
#
#   h1 = -tau1
#   x1 = x1*h1
#   x2 = x2*h1

        movq      %rsi, %r11    # restore %r11 (hh(1,1))

	# carefull here we want xmm10 to contains for times the value hh(1,1)
        movddup   (%r11), %xmm13        # copy hh(1,1) hh(2,1) into xmm13 and duplicate
       movsldup   %xmm13, %xmm10       # copy the first 4 bytes (hh(n1,1)) and duplicate, the same for the third 4 bytes => xmm10 contains h(1,1), h(1,1), h(1,1), h(1,1)
#        movshdup   %xmm13, %xmm10

        xorps   %xmm11, %xmm11
        subps   %xmm10, %xmm11 # %xmm11 = -hh(1,1)

        mulps   %xmm11, %xmm0
        .if \nrows>=8
        mulps   %xmm11, %xmm1
        .if \nrows==12
        mulps   %xmm11, %xmm2
        .endif
        .endif


#   h1 = -tau2
#   h2 = -tau2*s
#   y1 = y1*h1 + x1*h2
#   y2 = y2*h1 + x2*h2

	# careful here we want xmm12 to contain four times hh(1,2)
        movddup (%r11,%r9), %xmm13  # xmm13 contains hh(1,2) hh(2,2) and duplicate
       movsldup   %xmm13, %xmm10   # copy the first 4 bytes (hh(1,2)) and duplicate, the same for the third 4 bytes => xmm10 contains h(1,2), h(1,2), h(1,2), h(1,2)
#        movshdup   %xmm13, %xmm10

        xorps   %xmm9, %xmm9
        subps   %xmm10, %xmm9       # %xmm9 = -hh(1,2) = h1
        movaps  %xmm9, %xmm11

	# careful here we want xmm10 to contain four times the value of s
        movddup (%rsp), %xmm13      # Get s from top of stack plus unknown x and duplicate |s | x| s | x
       movsldup   %xmm13, %xmm10   # copy the first 4 bytes (s) and duplicate, the same for the third 4 bytes => xmm10 contains s,s,s,s
#        movshdup   %xmm13, %xmm10

        mulps   %xmm10, %xmm11 # %xmm14 = h2

        .macro mac_xform_y_single X, Y
        mulps   %xmm9, \Y  # y1 = y1*h1
        movaps  \X, %xmm10
        mulps   %xmm11, %xmm10
        addps   %xmm10, \Y
        .endm

        mac_xform_y_single %xmm0, %xmm6
        .if \nrows>=8
        mac_xform_y_single %xmm1, %xmm7
        .if \nrows==12
        mac_xform_y_single %xmm2, %xmm8
        .endif
        .endif
        .purgem   mac_xform_y_single

#   q(1,1) = q(1,1) + y1
#   q(2,1) = q(2,1) + y2

        movq   %rdi, %r10   # restore original Q

        .macro mac_pre_loop2_1_single qoff, Y
        movaps    \qoff(%r10), %xmm13   # q(.,1)
        addps     \Y, %xmm13
        movaps    %xmm13, \qoff(%r10)
        .endm

        mac_pre_loop2_1_single  0, %xmm6
        .if \nrows>=8
        mac_pre_loop2_1_single 16, %xmm7
        .if \nrows==12
        mac_pre_loop2_1_single 32, %xmm8
        .endif
        .endif
        .purgem   mac_pre_loop2_1_single

#   q(1,2) = q(1,2) + x1 + y1*hh(2,2)
#   q(2,2) = q(2,2) + x2 + y2*hh(2,2)

        addq      %r8, %r10             # %r10 => q(.,2)

	# careful here we want xmm9 to contain 4 times the value of h(2,2)
        movddup 4(%r11,%r9), %xmm13  # xmm13 contains hh(2,2) hh(2,3) and duplicate
       movsldup   %xmm13, %xmm9     # copy the first 4 bytes (hh(2,2)) and duplicate, the same for the third 4 bytes => xmm10 contains h(2,2), h(2,2), h(2,2), h(2,2)
#        movshdup   %xmm13, %xmm9

        .macro mac_pre_loop2_2_single qoff, X, Y
        movaps    \X, %xmm13
        movaps    \Y, %xmm10
        mulps     %xmm9, %xmm10
        addps     %xmm10, %xmm13
        addps     \qoff(%r10), %xmm13
        movaps    %xmm13, \qoff(%r10)
        .endm

        mac_pre_loop2_2_single  0, %xmm0, %xmm6
        .if \nrows>=8
        mac_pre_loop2_2_single 16, %xmm1, %xmm7
        .if \nrows==12
        mac_pre_loop2_2_single 32, %xmm2, %xmm8
        .endif
        .endif
        .purgem   mac_pre_loop2_2_single


#   do i=3,nb
#      h1 = hh(i-1,1)
#      h2 = hh(i,2)
#      q(1,i) = q(1,i) + x1*h1 + y1*h2
#      q(2,i) = q(2,i) + x2*h1 + y2*h2
#   enddo

        addq      $4, %r11
        .align 16
1:
        cmpq %rax, %r11                 # Jump out of the loop if %r11 >= %rax
        jge       2f

        addq      %r8, %r10             # %r10 => q(.,i)

	# careful here we want xmm11 to contain 4 times the value of hh(i-1,1)
        movddup   (%r11), %xmm13      # hh(i-1,1) | hh(i,1) | hh(i-1,1) | hh(i,1)
        movsldup   %xmm13, %xmm11     # copy the first 4 bytes hh(i-1,1)
#         movshdup   %xmm13, %xmm11

        # careful here we want xmm9 to contain 4 times the value of hh(i,2)
        movddup   4(%r11,%r9), %xmm13   # hh(i,2) | hh(i+1,2) and duplicate
        movsldup   %xmm13, %xmm9        # copy the first 4 bytes hh(i,2)
#        movshdup   %xmm13, %xmm9

        .macro mac_loop2_single qoff, X, Y
        movaps    \X, %xmm13
        mulps     %xmm11, %xmm13
        movaps    \Y, %xmm10
        mulps     %xmm9, %xmm10
        addps     %xmm10, %xmm13
        addps     \qoff(%r10), %xmm13
        movaps    %xmm13, \qoff(%r10)
        .endm

        mac_loop2_single  0, %xmm0, %xmm6
        .if \nrows>=8
        mac_loop2_single 16, %xmm1, %xmm7
        .if \nrows==12
        mac_loop2_single 32, %xmm2, %xmm8
        .endif
        .endif
        .purgem   mac_loop2_single

        addq      $4, %r11
        jmp       1b

2:

#   q(1,nb+1) = q(1,nb+1) + x1*hh(nb,1)
#   q(2,nb+1) = q(2,nb+1) + x2*hh(nb,1)

        addq      %r8, %r10             # %r10 => q(.,nb+1)

	# carefule here we want xm11 to contain 4 times the value of hh(nb,1)
        movddup   (%r11), %xmm13  # hh(nb,1) | hh(nb+1,1) and duplicate
        movsldup   %xmm13, %xmm11 # copy the first 4 bytes hh(nb,1)
#        movshdup   %xmm13, %xmm11

        .macro mac_post_loop2_single qoff, X
        movaps    \qoff(%r10), %xmm13   # q(.,nb+1)
        mulps     %xmm11, \X
        addps     \X, %xmm13
        movaps    %xmm13, \qoff(%r10)
        .endm

        mac_post_loop2_single  0, %xmm0
        .if \nrows>=8
        mac_post_loop2_single 16, %xmm1
        .if \nrows==12
        mac_post_loop2_single 32, %xmm2
        .endif
        .endif
        .purgem   mac_post_loop2_single

        .endm

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# FORTRAN Interface:
#
# subroutine double_hh_trafo(q, hh, nb, nq, ldq, ldh)
#
#   integer, intent(in) :: nb, nq, ldq, ldh
#   real*8, intent(inout) :: q(ldq,*)
#   real*8, intent(in) :: hh(ldh,*)
#
# Parameter mapping to registers
#   parameter 1: %rdi : q
#   parameter 2: %rsi : hh
#   parameter 3: %rdx : nb
#   parameter 4: %rcx : nq
#   parameter 5: %r8  : ldq
#   parameter 6: %r9  : ldh
#
#-------------------------------------------------------------------------------
#!f>#ifdef WITH_REAL_SSE_ASSEMBLY_KERNEL
#!f>#ifdef WANT_SINGLE_PRECISION_REAL
#!f>  interface
#!f>    subroutine double_hh_trafo_single(q, hh, nb, nq, ldq, ldh) bind(C,name="double_hh_trafo_single")
#!f>      use, intrinsic :: iso_c_binding
#!f>      integer(kind=c_int) :: nb, nq, ldq, ldh
#!f>      type(c_ptr), value  :: q
#!f>      real(kind=c_float)  :: hh(nb,6)
#!f>    end subroutine
#!f>  end interface
#!f>#endif
#!f>#endif
        .align    16,0x90
double_hh_trafo_single:

        # Get integer parameters into corresponding registers

        movslq    (%rdx), %rdx # nb
        movslq    (%rcx), %rcx # nq
        movslq    (%r8),  %r8  # ldq
        movslq    (%r9),  %r9  # ldh

        # Get ldq in bytes
        addq      %r8, %r8
        addq      %r8, %r8 # 4*ldq, i.e. ldq in bytes

        # Get ldh in bytes
        addq      %r9, %r9
        addq      %r9, %r9 # 4*ldq, i.e. ldh in bytes

        # set %rax to the address of hh at the end of the loops,
        # i.e. if %rdx >= %rax we must jump out of the loop.
        # please note: %rax = 4*%rdx + %rsi - 4
        movq %rdx, %rax
        addq %rax, %rax
        addq %rax, %rax
        addq %rsi, %rax
        subq $4, %rax

#-----------------------------------------------------------
        # Calculate the dot product of the two Householder vectors

        # decrement stack pointer to make space for s
        subq $4, %rsp

#   Fortran code:
#   s = hh(2,2)*1
#   do i=3,nb
#      s = s+hh(i,2)*hh(i-1,1)
#   enddo

        movq      %rsi, %r11   # Copy address of hh

        movss     4(%r11,%r9), %xmm0 #  hh(2,2)
        addq      $4, %r11
1:
        cmpq %rax, %r11
        jge       2f
        movss   (%r11), %xmm11       # hh(i-1,1)
        movss   4(%r11,%r9), %xmm9   # hh(i,2)
        mulss   %xmm11, %xmm9
        addss   %xmm9, %xmm0
        addq      $4, %r11
        jmp       1b
2:
        movss   %xmm0, (%rsp)   # put s on top of stack
#-----------------------------------------------------------

rloop_single:
        cmpq      $8, %rcx   # if %rcx <= 8 jump out of loop
        jle       rloop_e
        hh_trafo_real_single 12 # transform 12 rows
        addq      $48, %rdi  # increment q start adress by 48 bytes (6 rows)
        subq      $12, %rcx  # decrement nq
        jmp       rloop_single

rloop_e:
        cmpq      $4, %rcx   # if %rcx <= 4 jump to test_2
        jle       test_4
        hh_trafo_real_single 8 # transform 8 rows
        jmp       return1

test_4:
        cmpq      $0, %rcx   # if %rcx <= 0 jump to return
        jle       return1
        hh_trafo_real_single 4 # transform 4 rows

return1:
        addq      $4, %rsp   # reset stack pointer
        ret

        .align    16,0x90

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

        .macro hh_trafo_complex_single nrows

        # When this macro is called, the following registers are set and must not be changed
        # %rdi: Address of q
        # %rsi: Address of hh
        # %rdx: nb
        # %rcx: Remaining rows nq
        # %r8:  ldq in bytes

        movq      %rdi, %r10   # Copy address of q
        movq      %rsi, %r11   # Copy address of hh

        # set %rax to the address of hh at the end of the loops,
        # i.e. if %rdx >= %rax we must jump out of the loop.
        # please note: %rax = 8*%rdx + %rsi
        movq %rdx, %rax
        addq %rax, %rax
        addq %rax, %rax
        addq %rax, %rax # 8 * rax
        addq %rsi, %rax

#   x1 = q(1,1); y1 = 0
#   x2 = q(2,1); y2 = 0
#   ...

        movaps      (%r10), %xmm0   # xmm0 now contains the first 16 bytes of q => TWO single precision complex q(1,1), q(2,1)
        xorps     %xmm3, %xmm3
        .if \nrows>=4
        movaps    16(%r10), %xmm1   # xmm1 now contains the second 16 bytes of q => TWO single precision complex q(3,1), q(4,1)
        xorps     %xmm4, %xmm4
        .if \nrows==6
        movaps    32(%r10), %xmm2  # xmm2 now contains the third 16 bytes of q => TWO single precision complex q(5,1), q(6,1)
        xorps     %xmm5, %xmm5
        .endif
        .endif

#   do i=2,nb
#      h1 = conjg(hh(i))
#      x1 = x1 + q(1,i)*h1
#      x2 = x2 + q(2,i)*h1
#      ...
#   enddo

        addq      $8, %r11  # %r11 => hh(2)
        .align 16
1:
        cmpq      %rax, %r11      # Jump out of the loop if %r11 >= %rax
        jge 2f

        addq      %r8, %r10       # %r10 => q(.,i)

        # movddup    (%r11), %xmm7 # real(hh(i))
        # movddup   8(%r11), %xmm8 # imag(hh(i))

	# we use xmm6 as dummy variable
	xorps     %xmm6, %xmm6

	movddup    (%r11), %xmm6  # copy the single precision complex value h(i) in xmm6 and duplicate real(h(i)) | imag(h(i)) | real(h(i)) | imag(h(i))
	movsldup   %xmm6, %xmm7   # copy the fist 4 bytes of xmm6 and duplicate in lower half, copy the third 4 bytes of xmm6 and duplicate in upper half -> real(h(i)), real(h(i)), real(h(i)), real(h(i))
	movshdup   %xmm6, %xmm8   # as before but with 2nd 4bytes and fouth 4 bytes; xmm8 contains complex(h(i)), complex(h(i)), complex(h(i)), complex(h(i))

#	movshdup   %xmm6, %xmm7   # copy the real part of h(i)  into xmm7 four times ; xmm7 contains real(h(i)), real(h(i)), real(h(i)), real(h(i))
#	movsldup   %xmm6, %xmm8   # copy the complex part of h(i) into xmm8 four times ; xmm8 contains complex(h(i)), complex(h(i)), complex(h(i)), complex(h(i))



        .macro mac_loop1_single qoff, X, Y
        movaps    \qoff(%r10), %xmm13     # q(.,i) ; copy TWO single precision complex q(1,1) and q(2,1) into xmm6
        movaps    %xmm13, %xmm9           # copy xmm6 into xmm9
        mulps     %xmm7, %xmm13           # q(.,i)*real(hh(i)) # multiply real(hh(i)), real(h(i)), real(h(i)), real(h(i)) with TWO single precision COMPLEX q(1,1), q(2,1)
        addps     %xmm13, \X              # x1 = x1 + q(.,i)*real(hh(i))  # add the four single precision parts
        mulps     %xmm8, %xmm9            # q(.,i)*imag(hh(i))  # multiply contains complex(h(i)), complex(h(i)), complex(h(i)), complex(h(i)) with TWO single precision COMPLEX q(1,1), q(2,1)
        addsubps  %xmm9, \Y               # y1 = y1 -/+ q(.,i)*imag(hh(i)) # add the four single precision parts
        .endm

        mac_loop1_single 0, %xmm0, %xmm3
        .if \nrows>=4
        mac_loop1_single 16, %xmm1, %xmm4
        .if \nrows==6
        mac_loop1_single  32, %xmm2, %xmm5
        .endif
        .endif

        .purgem   mac_loop1_single

        addq      $8, %r11                # %r11 => hh(i+1)
        jmp       1b
2:

        # Now the content of the yn has to be swapped and added to xn
        .macro mac_post_loop_1_single X, Y
        shufps $0b10110001, \Y, \Y
        addps  \Y, \X
        .endm

        mac_post_loop_1_single  %xmm0, %xmm3
        .if \nrows>=4
        mac_post_loop_1_single   %xmm1, %xmm4
        .if \nrows==6
        mac_post_loop_1_single   %xmm2, %xmm5
        .endif
        .endif
        .purgem   mac_post_loop_1_single

#   tau1 = hh(1)
#
#   h1 = -tau1
#   x1 = x1*h1; y1 = x1 with halfes exchanged
#   x2 = x2*h1; y2 = x2 with halfes exchanged
#   ...

        movq      %rsi, %r11      # restore address of hh

	# copy four times the real part of hh(1) and change sign, same for complex part
	# in the end xmm8 should be -im(hh(1)) | -im(hh(1)) | -im(hh(1)) | -im(hh(1))
	# in the end xmm7 should be -re(hh(1)) | -re(hh(1)) | -re(hh(1)) | -re(hh(1))

#        movddup    (%r11), %xmm9 # real(hh(1))
#        movddup   8(%r11), %xmm7 # imag(hh(1))

        xorps     %xmm10, %xmm10    # dummy variable
	xorps     %xmm7, %xmm7
	xorps     %xmm8, %xmm8

	movddup    (%r11), %xmm6  # copy the single precision complex value h(i) in xmm6 and duplicate! xmm6 = re | im | re | im
	subps     %xmm6, %xmm10    # change the signs of real and imaginary parts; xmm10 = - re | -im | -re | - im
        movsldup   %xmm10, %xmm7  # copy the real part of -h(i)  into xmm7 four times ; xmm7 contains -real(h(i)), -real(h(i)), -real(h(i)), -real(h(i))
        movshdup   %xmm10, %xmm8  # copy the complex part of h(i) into xmm8 four times ; xmm8 contains -complex(h(i)), -complex(h(i)), -complex(h(i)), -complex(h(i))
#       movshdup   %xmm10, %xmm7  # copy the real part of -h(i)  into xmm7 four times ; xmm7 contains -real(h(i)), -real(h(i)), -real(h(i)), -real(h(i))
#       movsldup   %xmm10, %xmm8  # copy the complex part of h(i) into xmm8 four times ; xmm8 contains -complex(h(i)), -complex(h(i)), -complex(h(i)), -complex(h(i))


# maybe not neccessrary
        xorps %xmm9, %xmm9

        .macro mac_xform_single X, Y
        movaps    \X, %xmm6
        shufps    $0b10110001, \X, %xmm6
        mulps     %xmm8, %xmm6
        mulps     %xmm7, \X
        addsubps  %xmm6, \X
        movaps    \X, \Y          # copy to y
        shufps    $0b10110001, \X, \Y
        .endm

        mac_xform_single %xmm0, %xmm3
        .if \nrows>=4
        mac_xform_single %xmm1, %xmm4
        .if \nrows==6
        mac_xform_single %xmm2, %xmm5
        .endif
        .endif
        .purgem mac_xform_single

#   q(1,1) = q(1,1) + x1
#   q(2,1) = q(2,1) + x2
#   ...

        movq      %rdi, %r10      # restore address of q
        .macro mac_pre_loop2_single qoff, X
        movaps    \qoff(%r10), %xmm6     # q(.,1)
        addps     \X, %xmm6
        movaps    %xmm6, \qoff(%r10)
        .endm

        mac_pre_loop2_single   0, %xmm0
        .if \nrows>=4
        mac_pre_loop2_single  16, %xmm1
        .if \nrows==6
        mac_pre_loop2_single  32, %xmm2
        .endif
        .endif
        .purgem mac_pre_loop2_single

#   do i=2,nb
#      h1 = hh(i)
#      q(1,i) = q(1,i) + x1*h1
#      q(2,i) = q(2,i) + x2*h1
#      ...
#   enddo

        addq      $8, %r11
        .align 16
1:
        cmpq      %rax, %r11      # Jump out of the loop if %r11 >= %rax
        jge 2f

        addq      %r8, %r10       # %r10 => q(.,i)

	# carefull here we want xmm7 to contain four times the value of real(hh(i))
	# and xmm8 to contain four times the value of imag(hh(i))
#        movddup    (%r11), %xmm7 # real(hh(i))
#        movddup   8(%r11), %xmm8 # imag(hh(i))

	movddup    (%r11), %xmm6  # copy the single precision complex value h(i) in xmm6 and duplicate ; real(h(i)) | imag(h(i)) | real(h(i)) | imag(h(i))
        movsldup   %xmm6, %xmm7  # copy the real part of h(i)  into xmm7 four times ; xmm7 contains real(h(i)), real(h(i)), real(h(i)), real(h(i))
        movshdup   %xmm6, %xmm8  # copy the complex part of h(i) into xmm8 four times ; xmm8 contains complex(h(i)), complex(h(i)), complex(h(i)), complex(h(i))

        .macro mac_loop2_single qoff, X, Y
        movaps    \X, %xmm6
        mulps     %xmm7, %xmm6
        movaps    \Y, %xmm9
        mulps     %xmm8, %xmm9
        addsubps  %xmm9, %xmm6
        addps     \qoff(%r10), %xmm6
        movaps    %xmm6, \qoff(%r10)
        .endm

        mac_loop2_single   0, %xmm0, %xmm3
        .if \nrows>=4
        mac_loop2_single  16, %xmm1, %xmm4
        .if \nrows==6
        mac_loop2_single  32, %xmm2, %xmm5
        .endif
        .endif
        .purgem   mac_loop2_single

        addq      $8, %r11
        jmp       1b
2:
        .endm


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# FORTRAN Interface:
#
# subroutine single_hh_trafo_complex_single(q, hh, nb, nq, ldq)
#
#   integer, intent(in) :: nb, nq, ldq
#   complex*8, intent(inout) :: q(ldq,*)
#   complex*8, intent(in) :: hh(*)
#
# Parameter mapping to registers
#   parameter 1: %rdi : q
#   parameter 2: %rsi : hh
#   parameter 3: %rdx : nb
#   parameter 4: %rcx : nq
#   parameter 5: %r8  : ldq
#
#-------------------------------------------------------------------------------
#!f>#ifdef WITH_COMPLEX_SSE_ASSEMBLY_KERNEL
#!f>#ifdef WANT_SINGLE_PRECISION_COMPLEX
#!f>  interface
#!f>    subroutine single_hh_trafo_complex_single(q, hh, nb, nq, ldq) bind(C,name="single_hh_trafo_complex_single")
#!f>      use, intrinsic :: iso_c_binding
#!f>      integer(kind=c_int)   :: nb, nq, ldq
#!f>      complex(kind=c_float) :: q(*)
#!f>      complex(kind=c_float) :: hh(nb,2)
#!f>    end subroutine
#!f>  end interface
#!f>#endif
#!f>#endif

        .align    16,0x90
single_hh_trafo_complex_single:

        # Get integer parameters into corresponding registers

        movslq    (%rdx), %rdx # nb
        movslq    (%rcx), %rcx # nq
        movslq    (%r8),  %r8  # ldq

        # Get ldq in bytes
        addq      %r8, %r8
        addq      %r8, %r8
        addq      %r8, %r8 # 8*ldq, i.e. ldq in bytes

cloop_s:
        cmpq      $4, %rcx   # if %rcx <= 4 jump out of loop
        jle       cloop_e
        hh_trafo_complex_single 6 # transform 6 rows
        addq      $48, %rdi  # increment q start adress by 48 bytes (6 rows)
        subq      $6,  %rcx  # decrement nq
        jmp       cloop_s
cloop_e:

        cmpq      $2, %rcx   # if %rcx <= 2 jump to test_2
        jle       test_2
        hh_trafo_complex_single 4 # transform 4 rows
        jmp       return2

test_2:
        cmpq      $0, %rcx   # if %rcx <= 0 jump to return
        jle       return2
        hh_trafo_complex_single 2 # transform 2 rows

return2:
        ret

        .align    16,0x90
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# Declare that we do not need an executable stack here
	.section	.note.GNU-stack,"",@progbits
