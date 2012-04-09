# --------------------------------------------------------------------------------------------------
#
# This file contains the compute intensive kernels for the Householder transformations,
# coded in x86_64 assembler and using SSE2/SSE3 instructions.
#
# It must be assembled with GNU assembler (just "as" on most Linux machines)
# 
# Copyright of the original code rests with the authors inside the ELPA
# consortium. The copyright of any additional modifications shall rest
# with their original authors, but shall adhere to the licensing terms
# distributed along with the original code in the file "COPYING".
#
# --------------------------------------------------------------------------------------------------

        .globl double_hh_trafo_
        .globl single_hh_trafo_complex_
        .text

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

        .macro hh_trafo_real nrows

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

        movaps      (%r10), %xmm6       # y1 = q(1,1)
        movaps    16(%r10), %xmm7       # y2 = q(2,1)
        .if \nrows>=8
        movaps    32(%r10), %xmm8
        movaps    48(%r10), %xmm9
        .if \nrows==12
        movaps    64(%r10), %xmm10
        movaps    80(%r10), %xmm11
        .endif
        .endif

        addq      %r8, %r10             # %r10 => q(.,2)
        movddup   8(%r11,%r9), %xmm15   #  hh(2,2)

        .macro mac_pre_loop1 qoff, X, Y
        movaps    \qoff(%r10), \X       # xn = q(n,2)
        movaps    \X, %xmm12
        mulpd     %xmm15, %xmm12
        addpd     %xmm12, \Y            # yn = yn + xn*h(2,2)
        .endm

        mac_pre_loop1  0, %xmm0, %xmm6
        mac_pre_loop1 16, %xmm1, %xmm7
        .if \nrows>=8
        mac_pre_loop1 32, %xmm2, %xmm8
        mac_pre_loop1 48, %xmm3, %xmm9
        .if \nrows==12
        mac_pre_loop1 64, %xmm4, %xmm10
        mac_pre_loop1 80, %xmm5, %xmm11
        .endif
        .endif
        .purgem   mac_pre_loop1

#   do i=3,nb
#      h1 = hh(i-1,1)
#      h2 = hh(i,2)
#      x1 = x1 + q(1,i)*h1
#      y1 = y1 + q(1,i)*h2
#      x2 = x2 + q(2,i)*h1
#      y2 = y2 + q(2,i)*h2
#      ...
#   enddo

        addq      $8, %r11
        .align 16
1:
        cmpq %rax, %r11                 # Jump out of the loop if %r11 >= %rax
        jge       2f

        addq      %r8, %r10             # %r10 => q(.,i)

        movddup   (%r11), %xmm14        # hh(i-1,1)
        movddup   8(%r11,%r9), %xmm15   # hh(i,2)

        .macro mac_loop1 qoff, X, Y
        movaps    \qoff(%r10), %xmm13   # q(.,i)
        movaps    %xmm13, %xmm12
        mulpd     %xmm14, %xmm13
        addpd     %xmm13, \X            # xn = xn + q(.,i)*h1
        mulpd     %xmm15, %xmm12
        addpd     %xmm12, \Y            # yn = yn + q(.,i)*h2
        .endm

        mac_loop1  0, %xmm0, %xmm6
        mac_loop1 16, %xmm1, %xmm7
        .if \nrows>=8
        mac_loop1 32, %xmm2, %xmm8
        mac_loop1 48, %xmm3, %xmm9
        .if \nrows==12
        mac_loop1 64, %xmm4, %xmm10
        mac_loop1 80, %xmm5, %xmm11
        .endif
        .endif
        .purgem   mac_loop1

        addq      $8, %r11
        jmp       1b
2:

#   x1 = x1 + q(1,nb+1)*hh(nb,1)
#   x2 = x2 + q(2,nb+1)*hh(nb,1)

        addq      %r8, %r10             # %r10 => q(.,nb+1)
        movddup   (%r11), %xmm14

        .macro mac_post_loop1 qoff, X
        movaps    \qoff(%r10), %xmm13   # q(.,nb+1)
        mulpd     %xmm14, %xmm13
        addpd     %xmm13, \X
        .endm

        mac_post_loop1  0, %xmm0
        mac_post_loop1 16, %xmm1
        .if \nrows>=8
        mac_post_loop1 32, %xmm2
        mac_post_loop1 48, %xmm3
        .if \nrows==12
        mac_post_loop1 64, %xmm4
        mac_post_loop1 80, %xmm5
        .endif
        .endif
        .purgem   mac_post_loop1

#   tau1 = hh(1,1)
#   tau2 = hh(1,2)
#
#   h1 = -tau1
#   x1 = x1*h1
#   x2 = x2*h1

        movq      %rsi, %r11    # restore %r11 (hh(1,1))

        movddup (%r11), %xmm12 # hh(1,1)
        xorps   %xmm14, %xmm14
        subpd   %xmm12, %xmm14 # %xmm14 = -hh(1,1)

        mulpd   %xmm14, %xmm0
        mulpd   %xmm14, %xmm1
        .if \nrows>=8
        mulpd   %xmm14, %xmm2
        mulpd   %xmm14, %xmm3
        .if \nrows==12
        mulpd   %xmm14, %xmm4
        mulpd   %xmm14, %xmm5
        .endif
        .endif

#   h1 = -tau2
#   h2 = -tau2*s
#   y1 = y1*h1 + x1*h2
#   y2 = y2*h1 + x2*h2

        movddup (%r11,%r9), %xmm12  # hh(1,2)
        xorps   %xmm15, %xmm15
        subpd   %xmm12, %xmm15 # %xmm15 = -hh(1,2) = h1
        movaps  %xmm15, %xmm14
        movddup (%rsp), %xmm12 # Get s from top of stack
        mulpd   %xmm12, %xmm14 # %xmm14 = h2

        .macro mac_xform_y X, Y
        mulpd   %xmm15, \Y  # y1 = y1*h1
        movaps  \X, %xmm12
        mulpd   %xmm14, %xmm12
        addpd   %xmm12, \Y
        .endm

        mac_xform_y %xmm0, %xmm6
        mac_xform_y %xmm1, %xmm7
        .if \nrows>=8
        mac_xform_y %xmm2, %xmm8
        mac_xform_y %xmm3, %xmm9
        .if \nrows==12
        mac_xform_y %xmm4, %xmm10
        mac_xform_y %xmm5, %xmm11
        .endif
        .endif
        .purgem   mac_xform_y

#   q(1,1) = q(1,1) + y1
#   q(2,1) = q(2,1) + y2

        movq   %rdi, %r10   # restore original Q

        .macro mac_pre_loop2_1 qoff, Y
        movaps    \qoff(%r10), %xmm13   # q(.,1)
        addpd     \Y, %xmm13
        movaps    %xmm13, \qoff(%r10)
        .endm

        mac_pre_loop2_1  0, %xmm6
        mac_pre_loop2_1 16, %xmm7
        .if \nrows>=8
        mac_pre_loop2_1 32, %xmm8
        mac_pre_loop2_1 48, %xmm9
        .if \nrows==12
        mac_pre_loop2_1 64, %xmm10
        mac_pre_loop2_1 80, %xmm11
        .endif
        .endif
        .purgem   mac_pre_loop2_1

#   q(1,2) = q(1,2) + x1 + y1*hh(2,2)
#   q(2,2) = q(2,2) + x2 + y2*hh(2,2)

        addq      %r8, %r10             # %r10 => q(.,2)

        movddup   8(%r11,%r9), %xmm15   # hh(2,2)

        .macro mac_pre_loop2_2 qoff, X, Y
        movaps    \X, %xmm13
        movaps    \Y, %xmm12
        mulpd     %xmm15, %xmm12
        addpd     %xmm12, %xmm13
        addpd     \qoff(%r10), %xmm13
        movaps    %xmm13, \qoff(%r10)
        .endm

        mac_pre_loop2_2  0, %xmm0, %xmm6
        mac_pre_loop2_2 16, %xmm1, %xmm7
        .if \nrows>=8
        mac_pre_loop2_2 32, %xmm2, %xmm8
        mac_pre_loop2_2 48, %xmm3, %xmm9
        .if \nrows==12
        mac_pre_loop2_2 64, %xmm4, %xmm10
        mac_pre_loop2_2 80, %xmm5, %xmm11
        .endif
        .endif
        .purgem   mac_pre_loop2_2

#   do i=3,nb
#      h1 = hh(i-1,1)
#      h2 = hh(i,2)
#      q(1,i) = q(1,i) + x1*h1 + y1*h2
#      q(2,i) = q(2,i) + x2*h1 + y2*h2
#   enddo

        addq      $8, %r11
        .align 16
1:
        cmpq %rax, %r11                 # Jump out of the loop if %r11 >= %rax
        jge       2f

        addq      %r8, %r10             # %r10 => q(.,i)

        movddup   (%r11), %xmm14        # hh(i-1,1)
        movddup   8(%r11,%r9), %xmm15   # hh(i,2)

        .macro mac_loop2 qoff, X, Y
        movaps    \X, %xmm13
        mulpd     %xmm14, %xmm13
        movaps    \Y, %xmm12
        mulpd     %xmm15, %xmm12
        addpd     %xmm12, %xmm13
        addpd     \qoff(%r10), %xmm13
        movaps    %xmm13, \qoff(%r10)
        .endm

        mac_loop2  0, %xmm0, %xmm6
        mac_loop2 16, %xmm1, %xmm7
        .if \nrows>=8
        mac_loop2 32, %xmm2, %xmm8
        mac_loop2 48, %xmm3, %xmm9
        .if \nrows==12
        mac_loop2 64, %xmm4, %xmm10
        mac_loop2 80, %xmm5, %xmm11
        .endif
        .endif
        .purgem   mac_loop2

        addq      $8, %r11
        jmp       1b
2:

#   q(1,nb+1) = q(1,nb+1) + x1*hh(nb,1)
#   q(2,nb+1) = q(2,nb+1) + x2*hh(nb,1)

        addq      %r8, %r10             # %r10 => q(.,nb+1)
        movddup   (%r11), %xmm14

        .macro mac_post_loop2 qoff, X
        movaps    \qoff(%r10), %xmm13   # q(.,nb+1)
        mulpd     %xmm14, \X
        addpd     \X, %xmm13
        movaps    %xmm13, \qoff(%r10)
        .endm

        mac_post_loop2  0, %xmm0
        mac_post_loop2 16, %xmm1
        .if \nrows>=8
        mac_post_loop2 32, %xmm2
        mac_post_loop2 48, %xmm3
        .if \nrows==12
        mac_post_loop2 64, %xmm4
        mac_post_loop2 80, %xmm5
        .endif
        .endif
        .purgem   mac_post_loop2

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
        .align    16,0x90
double_hh_trafo_:

        # Get integer parameters into corresponding registers

        movslq    (%rdx), %rdx # nb
        movslq    (%rcx), %rcx # nq
        movslq    (%r8),  %r8  # ldq
        movslq    (%r9),  %r9  # ldh

        # Get ldq in bytes
        addq      %r8, %r8
        addq      %r8, %r8
        addq      %r8, %r8 # 8*ldq, i.e. ldq in bytes

        # Get ldh in bytes
        addq      %r9, %r9
        addq      %r9, %r9
        addq      %r9, %r9 # 8*ldh, i.e. ldh in bytes

        # set %rax to the address of hh at the end of the loops,
        # i.e. if %rdx >= %rax we must jump out of the loop.
        # please note: %rax = 8*%rdx + %rsi - 8
        movq %rdx, %rax
        addq %rax, %rax
        addq %rax, %rax
        addq %rax, %rax
        addq %rsi, %rax
        subq $8, %rax

#-----------------------------------------------------------
        # Calculate the dot product of the two Householder vectors

        # decrement stack pointer to make space for s
        subq $8, %rsp

#   Fortran code:
#   s = hh(2,2)*1
#   do i=3,nb
#      s = s+hh(i,2)*hh(i-1,1)
#   enddo

        movq      %rsi, %r11   # Copy address of hh

        movsd     8(%r11,%r9), %xmm0 #  hh(2,2)
        addq      $8, %r11
1:
        cmpq %rax, %r11
        jge       2f
        movsd   (%r11), %xmm14       # hh(i-1,1)
        movsd   8(%r11,%r9), %xmm15  # hh(i,2)
        mulsd   %xmm14, %xmm15
        addsd   %xmm15, %xmm0
        addq      $8, %r11
        jmp       1b
2:
        movsd   %xmm0, (%rsp)   # put s on top of stack
#-----------------------------------------------------------

rloop_s:
        cmpq      $8, %rcx   # if %rcx <= 8 jump out of loop
        jle       rloop_e
        hh_trafo_real 12 # transform 12 rows
        addq      $96, %rdi  # increment q start adress by 96 bytes (6 rows)
        subq      $12, %rcx  # decrement nq
        jmp       rloop_s
rloop_e:

        cmpq      $4, %rcx   # if %rcx <= 4 jump to test_2
        jle       test_4
        hh_trafo_real 8 # transform 8 rows
        jmp       return1

test_4:
        cmpq      $0, %rcx   # if %rcx <= 0 jump to return
        jle       return1
        hh_trafo_real 4 # transform 4 rows

return1:
        addq      $8, %rsp   # reset stack pointer
        ret

        .align    16,0x90

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

        .macro hh_trafo_complex nrows

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
        # please note: %rax = 16*%rdx + %rsi
        movq %rdx, %rax
        addq %rax, %rax
        addq %rax, %rax
        addq %rax, %rax
        addq %rax, %rax
        addq %rsi, %rax

#   x1 = q(1,1); y1 = 0
#   x2 = q(2,1); y2 = 0
#   ...

        movaps      (%r10), %xmm0
        movaps    16(%r10), %xmm1
        xorps     %xmm6, %xmm6
        xorps     %xmm7, %xmm7
        .if \nrows>=4
        movaps    32(%r10), %xmm2
        movaps    48(%r10), %xmm3
        xorps     %xmm8, %xmm8
        xorps     %xmm9, %xmm9
        .if \nrows==6
        movaps    64(%r10), %xmm4
        movaps    80(%r10), %xmm5
        xorps     %xmm10, %xmm10
        xorps     %xmm11, %xmm11
        .endif
        .endif

#   do i=2,nb
#      h1 = conjg(hh(i))
#      x1 = x1 + q(1,i)*h1
#      x2 = x2 + q(2,i)*h1
#      ...
#   enddo

        addq      $16, %r11  # %r11 => hh(2)
        .align 16
1:
        cmpq      %rax, %r11      # Jump out of the loop if %r11 >= %rax
        jge 2f

        addq      %r8, %r10       # %r10 => q(.,i)

        movddup    (%r11), %xmm14 # real(hh(i))
        movddup   8(%r11), %xmm15 # imag(hh(i))

        .macro mac_loop1 qoff, X, Y
        movaps    \qoff(%r10), %xmm13     # q(.,i)
        movaps    %xmm13, %xmm12
        mulpd     %xmm14, %xmm13          # q(.,i)*real(hh(i))
        addpd     %xmm13, \X              # x1 = x1 + q(.,i)*real(hh(i))
        mulpd     %xmm15, %xmm12          # q(.,i)*imag(hh(i))
        addsubpd  %xmm12, \Y              # y1 = y1 -/+ q(.,i)*imag(hh(i))
        .endm

        mac_loop1   0, %xmm0, %xmm6
        mac_loop1  16, %xmm1, %xmm7
        .if \nrows>=4
        mac_loop1  32, %xmm2, %xmm8
        mac_loop1  48, %xmm3, %xmm9
        .if \nrows==6
        mac_loop1  64, %xmm4, %xmm10
        mac_loop1  80, %xmm5, %xmm11
        .endif
        .endif

        .purgem   mac_loop1

        addq      $16, %r11                # %r11 => hh(i+1)
        jmp       1b
2:

        # Now the content of the yn has to be swapped and added to xn
        .macro mac_post_loop_1 X, Y
        shufpd $1, \Y, \Y
        addpd  \Y, \X
        .endm

        mac_post_loop_1  %xmm0, %xmm6
        mac_post_loop_1  %xmm1, %xmm7
        .if \nrows>=4
        mac_post_loop_1  %xmm2, %xmm8
        mac_post_loop_1  %xmm3, %xmm9
        .if \nrows==6
        mac_post_loop_1  %xmm4, %xmm10
        mac_post_loop_1  %xmm5, %xmm11
        .endif
        .endif
        .purgem   mac_post_loop_1

#   tau1 = hh(1)
#
#   h1 = -tau1
#   x1 = x1*h1; y1 = x1 with halfes exchanged
#   x2 = x2*h1; y2 = x2 with halfes exchanged
#   ...

        movq      %rsi, %r11      # restore address of hh

        xorps     %xmm14, %xmm14
        movddup    (%r11), %xmm12 # real(hh(1))
        subpd     %xmm12, %xmm14  #-real(hh(1))
        xorps     %xmm15, %xmm15
        movddup   8(%r11), %xmm12 # imag(hh(1))
        subpd     %xmm12, %xmm15  #-imag(hh(1))

        .macro mac_xform X, Y
        movaps    \X, %xmm12
        shufpd    $1, \X, %xmm12
        mulpd     %xmm15, %xmm12
        mulpd     %xmm14, \X
        addsubpd  %xmm12, \X
        movaps    \X, \Y          # copy to y
        shufpd    $1, \X, \Y      # exchange halfes
        .endm

        mac_xform %xmm0, %xmm6
        mac_xform %xmm1, %xmm7
        .if \nrows>=4
        mac_xform %xmm2, %xmm8
        mac_xform %xmm3, %xmm9
        .if \nrows==6
        mac_xform %xmm4, %xmm10
        mac_xform %xmm5, %xmm11
        .endif
        .endif
        .purgem mac_xform

#   q(1,1) = q(1,1) + x1
#   q(2,1) = q(2,1) + x2
#   ...

        movq      %rdi, %r10      # restore address of q
        .macro mac_pre_loop2 qoff, X
        movaps    \qoff(%r10), %xmm13     # q(.,1)
        addpd     \X, %xmm13
        movaps    %xmm13, \qoff(%r10)
        .endm

        mac_pre_loop2   0, %xmm0
        mac_pre_loop2  16, %xmm1
        .if \nrows>=4
        mac_pre_loop2  32, %xmm2
        mac_pre_loop2  48, %xmm3
        .if \nrows==6
        mac_pre_loop2  64, %xmm4
        mac_pre_loop2  80, %xmm5
        .endif
        .endif
        .purgem mac_pre_loop2

#   do i=2,nb
#      h1 = hh(i)
#      q(1,i) = q(1,i) + x1*h1
#      q(2,i) = q(2,i) + x2*h1
#      ...
#   enddo

        addq      $16, %r11
        .align 16
1:
        cmpq      %rax, %r11      # Jump out of the loop if %r11 >= %rax
        jge 2f

        addq      %r8, %r10       # %r10 => q(.,i)

        movddup    (%r11), %xmm14 # real(hh(i))
        movddup   8(%r11), %xmm15 # imag(hh(i))

        .macro mac_loop2 qoff, X, Y
        movaps    \X, %xmm13
        mulpd     %xmm14, %xmm13
        movaps    \Y, %xmm12
        mulpd     %xmm15, %xmm12
        addsubpd  %xmm12, %xmm13
        addpd     \qoff(%r10), %xmm13
        movaps    %xmm13, \qoff(%r10)
        .endm

        mac_loop2   0, %xmm0, %xmm6
        mac_loop2  16, %xmm1, %xmm7
        .if \nrows>=4
        mac_loop2  32, %xmm2, %xmm8
        mac_loop2  48, %xmm3, %xmm9
        .if \nrows==6
        mac_loop2  64, %xmm4, %xmm10
        mac_loop2  80, %xmm5, %xmm11
        .endif
        .endif
        .purgem   mac_loop2

        addq      $16, %r11
        jmp       1b
2:
        .endm

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# FORTRAN Interface:
#
# subroutine single_hh_trafo_complex(q, hh, nb, nq, ldq)
#
#   integer, intent(in) :: nb, nq, ldq
#   complex*16, intent(inout) :: q(ldq,*)
#   complex*16, intent(in) :: hh(*)
#
# Parameter mapping to registers
#   parameter 1: %rdi : q
#   parameter 2: %rsi : hh
#   parameter 3: %rdx : nb
#   parameter 4: %rcx : nq
#   parameter 5: %r8  : ldq
#
#-------------------------------------------------------------------------------
        .align    16,0x90
single_hh_trafo_complex_:

        # Get integer parameters into corresponding registers

        movslq    (%rdx), %rdx # nb
        movslq    (%rcx), %rcx # nq
        movslq    (%r8),  %r8  # ldq

        # Get ldq in bytes
        addq      %r8, %r8
        addq      %r8, %r8
        addq      %r8, %r8
        addq      %r8, %r8 # 16*ldq, i.e. ldq in bytes

cloop_s:
        cmpq      $4, %rcx   # if %rcx <= 4 jump out of loop
        jle       cloop_e
        hh_trafo_complex 6 # transform 6 rows
        addq      $96, %rdi  # increment q start adress by 96 bytes (6 rows)
        subq      $6,  %rcx  # decrement nq
        jmp       cloop_s
cloop_e:

        cmpq      $2, %rcx   # if %rcx <= 2 jump to test_2
        jle       test_2
        hh_trafo_complex 4 # transform 4 rows
        jmp       return2

test_2:
        cmpq      $0, %rcx   # if %rcx <= 0 jump to return
        jle       return2
        hh_trafo_complex 2 # transform 2 rows

return2:
        ret

        .align    16,0x90
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
