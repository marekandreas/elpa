AC_PREREQ([2.69])
AC_INIT([elpa],[2014.06.000], elpa-library@rzg.mpg.de)
AC_CONFIG_SRCDIR([src/elpa1.F90])

AM_INIT_AUTOMAKE([foreign -Wall subdir-objects])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config.h])
AM_SILENT_RULES([yes])

AC_SUBST([ELPA_LIB_VERSION], [2014.06.000])
# this is the version of the API, should be changed in the major revision
# if and only if the actual API changes
# see http://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
AC_SUBST([ELPA_SO_VERSION], [1:0:0])

AX_CHECK_GNU_MAKE()
if test x$_cv_gnu_make_command = x ; then
        AC_MSG_ERROR([Need GNU Make])
fi

AC_CHECK_PROG(CPP_FOUND,cpp,yes,no)
if test x"${CPP_FOUND}" = xno; then
  AC_MSG_ERROR([no cpp found])
fi

# gnu-make fortran module dependencies
m4_include([fdep/fortran_dependencies.m4])
FDEP_F90_GNU_MAKE_DEPS

AC_PROG_INSTALL
AM_PROG_CC_C_O
AM_PROG_AR
AM_PROG_AS
AC_PROG_CXX


dnl variables needed for the tests

dnl these test will cause an abort of configure if not
dnl successful. However, if MKL is found then the blas, blacs,
dnl lapack, and scalapack test can fail and vice versa
have_blas=no
have_blacs=no
have_mkl=no
have_mpi=no
have_lapack=no
have_scalapack=no
	
dnl these tests will decide which kernels can be build
dnl the usual case is all except the BlueGene (bg) kernels

can_compile_sse=no
can_compile_avx=no
can_compile_bgp=no
can_compile_bqq=no

fortran_can_check_environment=no
use_specific_real_kernel=no
use_specific_complex_kernel=no

install_real_generic=yes
install_real_generic_simple=yes
install_real_sse=no
install_real_bgp=no
install_real_bgq=no
install_real_avx_block2=no
install_real_avx_block4=no
install_real_avx_block6=no

install_complex_generic=yes
install_complex_generic_simple=yes
install_complex_sse=no
install_complex_bgp=no
install_complex_bgq=no
install_complex_avx_block1=no
install_complex_avx_block2=no
AC_MSG_CHECKING(whether SSE assembler kernel can be compiled)

echo "  .globl double_hh_trafo_
        .globl single_hh_trafo_complex_
        .text

         .macro hh_trafo_real nrows

        movq      %rdi, %r10   # Copy address of q
        movq      %rsi, %r11   # Copy address of hh

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

        addq      \$8, %r11
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

        addq      \$8, %r11
        jmp       1b
2:

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

        addq      \$8, %r11
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

        addq      \$8, %r11
        jmp       1b
2:

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

        .align    16,0x90
double_hh_trafo_:

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

        movq %rdx, %rax
        addq %rax, %rax
        addq %rax, %rax
        addq %rax, %rax
        addq %rsi, %rax
        subq \$8, %rax

        subq \$8, %rsp


        movq      %rsi, %r11   # Copy address of hh

        movsd     8(%r11,%r9), %xmm0 #  hh(2,2)
        addq      \$8, %r11
1:
        cmpq %rax, %r11
        jge       2f
        movsd   (%r11), %xmm14       # hh(i-1,1)
        movsd   8(%r11,%r9), %xmm15  # hh(i,2)
        mulsd   %xmm14, %xmm15
        addsd   %xmm15, %xmm0
        addq      \$8, %r11
        jmp       1b
2:
        movsd   %xmm0, (%rsp)   # put s on top of stack
#-----------------------------------------------------------

rloop_s:
        cmpq      \$8, %rcx   # if %rcx <= 8 jump out of loop
        jle       rloop_e
        hh_trafo_real 12 # transform 12 rows
        addq      \$96, %rdi  # increment q start adress by 96 bytes (6 rows)
        subq      \$12, %rcx  # decrement nq
        jmp       rloop_s
rloop_e:

        cmpq      \$4, %rcx   # if %rcx <= 4 jump to test_2
        jle       test_4
        hh_trafo_real 8 # transform 8 rows
        jmp       return1

test_4:
        cmpq      \$0, %rcx   # if %rcx <= 0 jump to return
        jle       return1
        hh_trafo_real 4 # transform 4 rows

return1:
        addq      \$8, %rsp   # reset stack pointer
        ret

        .align    16,0x90

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

        .macro hh_trafo_complex nrows


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


        addq      \$16, %r11  # %r11 => hh(2)
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

        addq      \$16, %r11                # %r11 => hh(i+1)
        jmp       1b
2:

        # Now the content of the yn has to be swapped and added to xn
        .macro mac_post_loop_1 X, Y
        shufpd \$1, \Y, \Y
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
        shufpd    \$1, \X, %xmm12
        mulpd     %xmm15, %xmm12
        mulpd     %xmm14, \X
        addsubpd  %xmm12, \X
        movaps    \X, \Y          # copy to y
        shufpd    \$1, \X, \Y      # exchange halfes
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

        addq      \$16, %r11
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

        addq      \$16, %r11
        jmp       1b
2:
        .endm

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
        cmpq      \$4, %rcx   # if %rcx <= 4 jump out of loop
        jle       cloop_e
        hh_trafo_complex 6 # transform 6 rows
        addq      \$96, %rdi  # increment q start adress by 96 bytes (6 rows)
        subq      \$6,  %rcx  # decrement nq
        jmp       cloop_s
cloop_e:

        cmpq      \$2, %rcx   # if %rcx <= 2 jump to test_2
        jle       test_2
        hh_trafo_complex 4 # transform 4 rows
        jmp       return2

test_2:
        cmpq      \$0, %rcx   # if %rcx <= 0 jump to return
        jle       return2
        hh_trafo_complex 2 # transform 2 rows

return2:
        ret

        .align    16,0x90" > ./test.s

 $CC -c ./test.s
 if test "$?" == 0; then
  can_compile_sse=yes
  install_real_sse=yes
  install_complex_sse=yes

 else
  can_compile_sse=no
  install_real_sse=no
  install_complex_sse=no
 fi

 rm -f ./test.s ./test.o
AC_MSG_RESULT([${can_compile_sse}])

dnl check whether one can compile with avx - gcc intrinsics

 AC_MSG_CHECKING([whether we can compile a gcc intrinsic AVX program])

 dnl first pass: try with specified CFLAGS and CXXFLAGS
 AC_COMPILE_IFELSE([AC_LANG_SOURCE([     
  #include <x86intrin.h> 
  void main(){
  double* q;
  __m256d a1_1 = _mm256_load_pd(q);
  }
  ])],
  [can_compile_avx=yes],
  [can_compile_avx=no]
 )
dnl first test failed: try again after updating  CFLAGS and CXXFLAGS with -mavx
 if test x"${can_compile_avx}" = x"no"; then
 CFLAGS="$CFLAGS -mavx"
 CXXFLAGS="$CXXFLAGS -mavx"
 AC_COMPILE_IFELSE([AC_LANG_SOURCE([     
  #include <x86intrin.h> 
  void main(){
  double* q;
  __m256d a1_1 = _mm256_load_pd(q);
  }
  ])],
  [can_compile_avx=yes],
  [can_compile_avx=no]
 )
 fi
AC_MSG_RESULT([${can_compile_avx}])

if test x"${can_compile_avx}" = x"yes"; then
  install_real_avx_block2=yes
  install_real_avx_block4=yes
  install_real_avx_block6=yes

  install_complex_avx_block1=yes
  install_complex_avx_block2=yes

fi


dnl set the AVX optimization flags if this option is specified
AC_MSG_CHECKING(whether AVX optimization flags should be set automatically)
AC_ARG_WITH([avx-optimization],
		AS_HELP_STRING([--with-avx-optimization],
			       [use AVX optimization, default no.]),
	      [with_avx_optimization=yes],
	      [with_avx_optimization=no])
AC_MSG_RESULT([${with_avx_optimization}])
if test x"${with_avx_optimization}" = xyes; then
 CFLAGS="$CFLAGS -funsafe-loop-optimizations -funsafe-math-optimizations -ftree-vect-loop-version -ftree-vectorize"
 CXXFLAGS="$CXXFLAGS -funsafe-loop-optimizations -funsafe-math-optimizations -ftree-vect-loop-version -ftree-vectorize"
fi

AC_LANG(Fortran)
m4_include([m4/ax_prog_fc_mpi.m4])
dnl check whether an mpi compiler is available;
dnl if not abort since it is mandatory
AX_PROG_FC_MPI([],[have_mpi=yes],[have_mpi=no
		 if test x"${have_mpi}" = xno; then
  		  AC_MSG_ERROR([no mpi found])
		  fi])

AC_FC_FREEFORM
AC_FC_MODULE_FLAG
AC_FC_MODULE_OUTPUT_FLAG

AC_MSG_CHECKING(whether OpenMP usage is specified)
AC_ARG_WITH([openmp],
		AS_HELP_STRING([--with-openmp],
			       [use OpenMP threading, default no.]),
	      [with_openmp=yes],
	      [with_openmp=no])
  AC_MSG_RESULT([${with_openmp}])
  if test x"${enable_openmp}" = xyes; then
     with_openmp=yes
     AC_MSG_CHECKING(whether --enable-openmp is specified)
     AC_MSG_RESULT([${enable_openmp}])
  fi
  AM_CONDITIONAL([WITH_OPENMP],[test x"$with_openmp" = x"yes"])
  if test x"${with_openmp}" = xyes; then
	AC_DEFINE([WITH_OPENMP], [1], [use OpenMP threading])
        AX_ELPA_OPENMP
  fi

FCFLAGS="$FCFLAGS $OPENMP_FCFLAGS $OPENMP_FFFLAGS"
#LDFLAGS="$LDFLAGS $OPENMP_FCFLAGS $OPENMP_FFFLAGS"

save_FCFLAGS=$FCFLAGS
save_LDFLAGS=$LDFLAGS

FCFLAGS="$FCFLAGS $BLACS_FCFLAGS"
LDFLAGS="$LDFLAGS $BLACS_LDFLAGS"

dnl check wheter fortran error_unit is defined
AC_MSG_CHECKING([whether Fortran module iso_fortran_env is available])
AC_COMPILE_IFELSE([AC_LANG_SOURCE([     
  program test_error_unit
    use ISO_FORTRAN_ENV, only : error_unit
    implicit none

    write(error_unit,*) "error_unit is defined"
  end program
])],
  [can_use_iso_fortran_env=yes],
  [can_use_iso_fortran_env=no]
)

dnl check whether one can link with specified MKL (desired method)
AC_MSG_CHECKING([whether we can compile a Fortran program using MKL])


AC_COMPILE_IFELSE([AC_LANG_SOURCE([     
  program test_mkl
    use mkl_service	
    character*198 :: string
    call mkl_get_version_string(string)
    write(*,'(a)') string
  end program
])],
  [can_compile_with_mkl=yes],
  [can_compile_with_mkl=no]
)
AC_MSG_RESULT([${can_compile_with_mkl}])

if test x"$can_compile_with_mkl" = x"yes" ; then
  AC_MSG_CHECKING([whether we can link a Fortran program with MKL])
  AC_LINK_IFELSE([AC_LANG_SOURCE([     
    program test_mkl
      use mkl_service	
      character*198 :: string
      call mkl_get_version_string(string)
      write(*,'(a)') string
    end program
  ])],
    [have_mkl=yes],
    [have_mkl=no]
  )
  AC_MSG_RESULT([${have_mkl}])
fi

dnl if not mkl, check all the necessary individually
if test x"${have_mkl}" = x"yes" ; then
  WITH_MKL=1
else

  dnl first check blas
  AC_SEARCH_LIBS([dgemm],[blas],[have_blas=yes],[have_blas=no])
  AC_MSG_CHECKING([whether we can link a program with a blas lib])
  AC_MSG_RESULT([${have_blas}])

  if test x"${have_blas}" = x"no" ; then
    AC_MSG_ERROR([could not link with blas: specify path]) 
  fi	
  dnl now lapack
  AC_SEARCH_LIBS([dlarrv],[lapack],[have_lapack=yes],[have_lapack=no])
  AC_MSG_CHECKING([whether we can link a program with a lapack lib])
  AC_MSG_RESULT([${have_lapack}])

  if test x"${have_lapack}" = x"no" ; then
    AC_MSG_ERROR([could not link with lapack: specify path]) 
  fi
	
  dnl now blacs
  AC_SEARCH_LIBS([blacs_gridinit],[mpiblacs],[have_blacs=yes],[have_blacs=no])
  AC_MSG_CHECKING([whether we can link a program with a blacs lib])
  AC_MSG_RESULT([${have_blacs}])



  if test x"${have_blacs}" = x"no" ; then
    AC_MSG_ERROR([could not link with blacs: specify path]) 
  fi
	
  dnl now scalapack
  AC_SEARCH_LIBS([pdtran],[mpiscalapack],[have_scalapack=yes],[have_scalapack=no])
  AC_MSG_CHECKING([whether we can link a program with a scalapack lib])
  AC_MSG_RESULT([${have_scalapack}])

  if test x"${have_scalapack}" = x"no" ; then
    AC_MSG_ERROR([could not link with scalapack: specify path]) 
  fi

  dnl check whether we can link alltogehter

  AC_MSG_CHECKING([whether we can link a Fortran program with all blacs/scalapack])
  AC_LINK_IFELSE([AC_LANG_SOURCE([     
    program dgemm_test

      integer , parameter:: n_cols=3,l_cols=3
      real :: hvm(n_cols,l_cols)

      call dgemm('T','N',n_cols,n_cols,l_cols,1.,hvm,ubound(hvm,1), &
      hvm(1,n_cols+1),ubound(hvm,1),0.,hvm,ubound(hvm,1))


     end program dgemm_test
     ])],
     [can_link_with_blacs_scalapack=yes],
     [can_link_with_blacs_scalapack=no]
   )
   AC_MSG_RESULT([${can_link_with_blacs_scalapack}])

   if test x"${can_link_with_blacs_scalapack}" = x"yes" ; then
     WITH_BLACS=1
   else
   AC_MSG_ERROR([We can neither link with MKL or another Scalpack. Please specify BLACS_LDFLAGS and BLACS_FCFLAGS!])
   fi
fi

dnl check for intrinsic fortran function of 2003 standard
AC_MSG_CHECKING([whether we can use the intrinsic Fortran function "get_environment_variable"])


AC_COMPILE_IFELSE([AC_LANG_SOURCE([     
  program test_get_environment

    character(len=256) :: homedir
    call get_environment_variable("HOME",homedir)
  end program
])],
  [fortran_can_check_environment=yes],
  [fortran_can_check_environment=no]
)
AC_MSG_RESULT([${fortran_can_check_environment}])


dnl important: reset them again!
FCFLAGS=$save_FCFLAGS
LDFLAGS=$save_LDFLAGS


dnl now check which kernels can be compiled

dnl the checks for SSE were already done before
dnl the checks for AVX were already done before

dnl check BGP kernel
AC_MSG_CHECKING([whether we can compile with BGP intrinsics])


AC_LINK_IFELSE([AC_LANG_SOURCE([     
  program test_bgp
    complex*16 :: y3,q3,h2
    y3 = fxcpmadd(y3,q3,h2)
  
  end program
])],
  [can_compile_bgp=yes],
  [can_compile_bgp=no]
)
AC_MSG_RESULT([${can_compile_bgp}])

if test x"${can_compile_bgp}" = x"yes" ; then
  install_real_bgp=yes
  install_complex_bgp=yes
fi

dnl check BGQ kernel
AC_MSG_CHECKING([whether we can compile with BGQ intrinsics])

AC_LINK_IFELSE([AC_LANG_SOURCE([     
  program test_bgq
    VECTOR(REAL(8))::QPX_x1
    real*8 :: hh(10,2))
    QPX_h2 = VEC_SPLATS(hh(2,2))
  
  end program
])],
  [can_compile_bgq=yes],
  [can_compile_bgq=no]
)
AC_MSG_RESULT([${can_compile_bgq}])

if test x"${can_compile_bgq}" = x"yes" ; then
  install_real_bgq=yes
  install_complex_bgq=yes
fi


dnl environment variable setting of kernel
if test x"${fortran_can_check_environment}" = x"yes" ; then
 AC_DEFINE([HAVE_ENVIRONMENT_CHECKING],[1],[Fortran can querry environment variables])
fi

dnl last check whether user wants to compile ony a specific kernel
dnl
dnl real kernels
dnl
dnl generic kernel
AC_ARG_WITH([real-generic-kernel],
 [AS_HELP_STRING([-with-real-generic-kernel],[only compile generic-kernel for real case])],[],[with_real_generic_kernel=no])

if test x"${with_real_generic_kernel}" = x"yes" ; then
 if test x"${use_specific_real_kernel}" = x"no" ; then
#  if test x"${can_compile_sse}" = x"yes" ; then

    dnl make sure that all the other kernels are unset
    install_real_generic=yes
    install_real_generic_simple=no
    install_real_sse=no
    install_real_bgp=no	
    install_real_bgq=no
    install_real_avx_block2=no
    install_real_avx_block4=no
    install_real_avx_block6=no

    use_specific_real_kernel=yes
    AC_MSG_NOTICE([real-generic-kernel will be the only compiled kernel for real case])
#  else
#  AC_MSG_FAILURE([--with-sse-generic-kernel failed; SSE kernel cannot be compiled on the system])
#  fi 
 else
  AC_MSG_FAILURE([--with-real-generic-kernel failed; A specific kernel for real case has already been defined before!])
 fi
fi

dnl generic-simple kernel
AC_ARG_WITH([real-generic-simple-kernel],
 [AS_HELP_STRING([-with-real-generic-simple-kernel],[only compile generic-simple-kernel for real case])],[],[with_real_generic_simple_kernel=no])

if test x"${with_real_generic_simple_kernel}" = x"yes" ; then
 if test x"${use_specific_real_kernel}" = x"no" ; then

    dnl make sure that all the other kernels are unset
    install_real_generic=no
    install_real_generic_simple=yes
    install_real_sse=no
    install_real_bgp=no	
    install_real_bgq=no
    install_real_avx_block2=no
    install_real_avx_block4=no
    install_real_avx_block6=no

    use_specific_real_kernel=yes
    AC_MSG_NOTICE([real-generic-simple-kernel will be the only compiled kernel for real case])

 else
  AC_MSG_FAILURE([--with-real-generic-simple-kernel failed; A specific kernel for real case has already been defined before!])
 fi
fi

dnl sse kernel
AC_ARG_WITH([real-sse-kernel],
 [AS_HELP_STRING([-with-real-sse-kernel],[only compile sse-kernel for real case])],[],[with_real_sse_kernel=no])

if test x"${with_real_sse_kernel}" = x"yes" ; then
 if test x"${use_specific_real_kernel}" = x"no" ; then
  if test x"${can_compile_sse}" = x"yes" ; then

    dnl make sure that all the other kernels are unset
    install_real_generic=no
    install_real_generic_simple=no
    install_real_sse=yes
    install_real_bgp=no	
    install_real_bgq=no
    install_real_avx_block2=no
    install_real_avx_block4=no
    install_real_avx_block6=no

    use_specific_real_kernel=yes
    AC_MSG_NOTICE([real-sse-kernel will be the only compiled kernel for real case])
  else
  AC_MSG_FAILURE([--with-real-sse-kernel failed; SSE kernel cannot be compiled on the system])
  fi 
 else
  AC_MSG_FAILURE([--with-real-sse-kernel failed; A specific kernel for real case has already been defined before!])
 fi
fi

dnl bgp kernel
AC_ARG_WITH([real-bgp-kernel],
 [AS_HELP_STRING([-with-real-bgp-kernel],[only compile bgp-kernel for real case])],[],[with_real_bgp_kernel=no])

if test x"${with_real_bgp_kernel}" = x"yes" ; then
 if test x"${use_specific_real_kernel}" = x"no" ; then
  if test x"${can_compile_bgp}" = x"yes" ; then

    dnl make sure that all the other kernels are unset
    install_real_generic=no
    install_real_generic_simple=no
    install_real_sse=no
    install_real_bgp=yes	
    install_real_bgq=no
    install_real_avx_block2=no
    install_real_avx_block4=no
    install_real_avx_block6=no

    use_specific_real_kernel=yes
    AC_MSG_NOTICE([real-bgp-kernel will be the only compiled kernel for real case])
  else
  AC_MSG_FAILURE([--with-real-bgp-kernel failed; BGP kernel cannot be compiled on the system])
  fi 
 else
  AC_MSG_FAILURE([--with-real-bgp-kernel failed; A specific kernel for real case has already been defined before!])
 fi
fi

dnl bgq kernel
AC_ARG_WITH([real-bgq-kernel],
 [AS_HELP_STRING([-with-real-bgq-kernel],[only compile bgq-kernel for real case])],[],[with_real_bgq_kernel=no])

if test x"${with_real_bgq_kernel}" = x"yes" ; then
 if test x"${use_specific_real_kernel}" = x"no" ; then
  if test x"${can_compile_bgq}" = x"yes" ; then

    dnl make sure that all the other kernels are unset
    install_real_generic=no
    install_real_generic_simple=no
    install_real_sse=no
    install_real_bgp=no	
    install_real_bgq=yes
    install_real_avx_block2=no
    install_real_avx_block4=no
    install_real_avx_block6=no

    use_specific_real_kernel=yes
    AC_MSG_NOTICE([real-bgq-kernel will be the only compiled kernel for real case])
  else
  AC_MSG_FAILURE([--with-real-bgq-kernel failed; BGQ kernel cannot be compiled on the system])
  fi 
 else
  AC_MSG_FAILURE([--with-real-bgq-kernel failed; A specific kernel for real case has already been defined before!])
 fi
fi

dnl real-avx-block2 kernel
AC_ARG_WITH([real-avx-block2-kernel],
 [AS_HELP_STRING([-with-real-avx-block2-kernel],[only compile real-avx-block2-kernel for real case])],[],[with_real_avx_block2_kernel=no])

if test x"${with_real_avx_block2_kernel}" = x"yes" ; then
 if test x"${use_specific_real_kernel}" = x"no" ; then
  if test x"${can_compile_avx}" = x"yes" ; then

    dnl make sure that all the other kernels are unset
    install_real_generic=no
    install_real_generic_simple=no
    install_real_sse=no
    install_real_bgp=no	
    install_real_bgq=no
    install_real_avx_block2=yes
    install_real_avx_block4=no
    install_real_avx_block6=no

    use_specific_real_kernel=yes
    AC_MSG_NOTICE([real-avx-block2-kernel will be the only compiled kernel for real case])
  else
  AC_MSG_FAILURE([--with-real-avx-block2-kernel failed; AVX kernels cannot be compiled on the system])
  fi 
 else
  AC_MSG_FAILURE([--with-real-avx-block2-kernel failed; A specific kernel for real case has already been defined before!])
 fi
fi

dnl real-avx-block4 kernel
AC_ARG_WITH([real-avx-block4-kernel],
 [AS_HELP_STRING([-with-real-avx-block4-kernel],[only compile real-avx-block4-kernel for real case])],[],[with_real_avx_block4_kernel=no])

if test x"${with_real_avx_block4_kernel}" = x"yes" ; then
 if test x"${use_specific_real_kernel}" = x"no" ; then
  if test x"${can_compile_avx}" = x"yes" ; then

    dnl make sure that all the other kernels are unset
    install_real_generic=no
    install_real_generic_simple=no
    install_real_sse=no
    install_real_bgp=no	
    install_real_bgq=no
    install_real_avx_block2=no
    install_real_avx_block4=yes
    install_real_avx_block6=no

    use_specific_real_kernel=yes
    AC_MSG_NOTICE([real-avx-block4-kernel will be the only compiled kernel for real case])
  else
  AC_MSG_FAILURE([--with-real-avx-block4-kernel failed; AVX kernels cannot be compiled on the system])
  fi 
 else
  AC_MSG_FAILURE([--with-real-avx-block4-kernel failed; A specific kernel for real case has already been defined before!])
 fi
fi

dnl real-avx-block6 kernel
AC_ARG_WITH([real-avx-block6-kernel],
 [AS_HELP_STRING([-with-real-avx-block6-kernel],[only compile real-avx-block6-kernel for real case])],[],[with_real_avx_block6_kernel=no])

if test x"${with_real_avx_block6_kernel}" = x"yes" ; then
 if test x"${use_specific_real_kernel}" = x"no" ; then
  if test x"${can_compile_avx}" = x"yes" ; then

    dnl make sure that all the other kernels are unset
    install_real_generic=no
    install_real_generic_simple=no
    install_real_sse=no
    install_real_bgp=no	
    install_real_bgq=no
    install_real_avx_block2=no
    install_real_avx_block4=no
    install_real_avx_block6=yes

    use_specific_real_kernel=yes
    AC_MSG_NOTICE([real-avx-block6-kernel will be the only compiled kernel for real case])
  else
  AC_MSG_FAILURE([--with-real-avx-block6-kernel failed; AVX kernels cannot be compiled on the system])
  fi 
 else
  AC_MSG_FAILURE([--with-real-avx-block6-kernel failed; A specific kernel for real case has already been defined before!])
 fi
fi


dnl last check whether user wants to compile ony a specific kernel
dnl
dnl complex kernels
dnl
dnl generic kernel
AC_ARG_WITH([complex-generic-kernel],
 [AS_HELP_STRING([-with-complex-generic-kernel],[only compile generic-kernel for complex case])],[],[with_complex_generic_kernel=no])

if test x"${with_complex_generic_kernel}" = x"yes" ; then
 if test x"${use_specific_complex_kernel}" = x"no" ; then

    dnl make sure that all the other kernels are unset
    install_complex_generic=yes
    install_complex_generic_simple=no
    install_complex_sse=no
    install_complex_bgp=no
    install_complex_bgq=no
    install_complex_avx_block1=no
    install_complex_avx_block2=no

    use_specific_complex_kernel=yes
    AC_MSG_NOTICE([complex-generic-kernel will be the only compiled kernel for complex case])
 
 else
  AC_MSG_FAILURE([--with-complex-generic-kernel failed; A specific kernel for complex case has already been defined before!])
 fi
fi

dnl generic-simple kernel
AC_ARG_WITH([complex-generic-simple-kernel],
 [AS_HELP_STRING([-with-complex-generic-simple-kernel],[only compile generic-simple-kernel for complex case])],[],[with_complex_generic_simple_kernel=no])

if test x"${with_complex_generic_simple_kernel}" = x"yes" ; then
 if test x"${use_specific_complex_kernel}" = x"no" ; then

    dnl make sure that all the other kernels are unset
    install_complex_generic=no
    install_complex_generic_simple=yes
    install_complex_sse=no
    install_complex_bgp=no
    install_complex_bgq=no
    install_complex_avx_block1=no
    install_complex_avx_block2=no

    use_specific_complex_kernel=yes

    AC_MSG_NOTICE([complex-generic-simple-kernel will be the only compiled kernel for complex case])

 else
  AC_MSG_FAILURE([--with-complex-generic-simple-kernel failed; A specific kernel for complex case has already been defined before!])
 fi
fi

dnl sse kernel
AC_ARG_WITH([complex-sse-kernel],
 [AS_HELP_STRING([-with-complex-sse-kernel],[only compile sse-kernel for complex case])],[],[with_complex_sse_kernel=no])

if test x"${with_complex_sse_kernel}" = x"yes" ; then
 if test x"${use_specific_complex_kernel}" = x"no" ; then
  if test x"${can_compile_sse}" = x"yes" ; then

    dnl make sure that all the other kernels are unset
    install_complex_generic=no
    install_complex_generic_simple=no
    install_complex_sse=yes
    install_complex_bgp=no	
    install_complex_bgq=no
    install_complex_avx_block1=no
    install_complex_avx_block2=no

    use_specific_complex_kernel=yes

    AC_MSG_NOTICE([complex-sse-kernel will be the only compiled kernel for complex case])
  else
  AC_MSG_FAILURE([--with-complex-sse-kernel failed; SSE kernel cannot be compiled on the system])
  fi 
 else
  AC_MSG_FAILURE([--with-complex-sse-kernel failed; A specific kernel for complex case has already been defined before!])
 fi
fi

dnl complex-bqp kernel
AC_ARG_WITH([complex-bgp-kernel],
 [AS_HELP_STRING([-with-complex-bgp-kernel],[only compile BGP-kernel (identical with complex-generic kernel) for complex case])],[],[with_complex_bgp_kernel=no])

if test x"${with_complex_bgp_kernel}" = x"yes" ; then
 if test x"${use_specific_complex_kernel}" = x"no" ; then
  if test x"${can_compile_bgp}" = x"yes" ; then

    dnl make sure that all the other kernels are unset
    install_complex_generic=no
    install_complex_generic_simple=no
    install_complex_sse=no
    install_complex_bgp=yes	
    install_complex_bgq=no
    install_complex_avx_block1=no
    install_complex_avx_block2=no

    use_specific_complex_kernel=yes

    AC_MSG_NOTICE([complex-bgp-kernel will be the only compiled kernel for complex case])
  else
  AC_MSG_FAILURE([--with-complex-bgp-kernel failed; BGP kernel cannot be compiled on the system])
  fi 
 else
  AC_MSG_FAILURE([--with-complex-bgp-kernel failed; A specific kernel for complex case has already been defined before!])
 fi
fi


dnl complex-bqq kernel
AC_ARG_WITH([complex-bgq-kernel],
 [AS_HELP_STRING([-with-complex-bgq-kernel],[only compile BGQ-kernel (identical with complex-generic kernel) for complex case])],[],[with_complex_bgq_kernel=no])

if test x"${with_complex_bgq_kernel}" = x"yes" ; then
 if test x"${use_specific_complex_kernel}" = x"no" ; then
  if test x"${can_compile_bgq}" = x"yes" ; then

    dnl make sure that all the other kernels are unset
    install_complex_generic=no
    install_complex_generic_simple=no
    install_complex_sse=no
    install_complex_bgp=no	
    install_complex_bgq=yes
    install_complex_avx_block1=no
    install_complex_avx_block2=no

    use_specific_complex_kernel=yes

    AC_MSG_NOTICE([complex-bgq-kernel will be the only compiled kernel for complex case])
  else
  AC_MSG_FAILURE([--with-complex-bgq-kernel failed; BGQ kernel cannot be compiled on the system])
  fi 
 else
  AC_MSG_FAILURE([--with-complex-bgq-kernel failed; A specific kernel for complex case has already been defined before!])
 fi
fi

dnl complex-avx-block1 kernel
AC_ARG_WITH([complex-avx-block1-kernel],
 [AS_HELP_STRING([-with-complex-avx-block1-kernel],[only compile complex-avx-block1-kernel for complex case])],[],[with_complex_avx_block1_kernel=no])

if test x"${with_complex_avx_block1_kernel}" = x"yes" ; then
 if test x"${use_specific_complex_kernel}" = x"no" ; then
  if test x"${can_compile_avx}" = x"yes" ; then

    dnl make sure that all the other kernels are unset
    install_complex_generic=no
    install_complex_generic_simple=no
    install_complex_sse=no
    install_complex_avx_block1=yes
    install_complex_avx_block2=no

    use_specific_complex_kernel=yes

    AC_MSG_NOTICE([complex-avx-block1-kernel will be the only compiled kernel for complex case])
  else
  AC_MSG_FAILURE([--with-complex-avx-block1-kernel failed; AVX kernels cannot be compiled on the system])
  fi 
 else
  AC_MSG_FAILURE([--with-complex-avx-block1-kernel failed; A specific kernel for complex case has already been defined before!])
 fi
fi

dnl complex-avx-block2 kernel
AC_ARG_WITH([complex-avx-block2-kernel],
 [AS_HELP_STRING([-with-complex-avx-block2-kernel],[only compile complex-avx-block2-kernel for complex case])],[],[with_complex_avx_block2_kernel=no])

if test x"${with_complex_avx_block2_kernel}" = x"yes" ; then
 if test x"${use_specific_complex_kernel}" = x"no" ; then
  if test x"${can_compile_avx}" = x"yes" ; then

    dnl make sure that all the other kernels are unset
    install_complex_generic=no
    install_complex_generic_simple=no
    install_complex_sse=no
    install_complex_avx_block1=no
    install_complex_avx_block2=yes

    use_specific_complex_kernel=yes

    AC_MSG_NOTICE([complex-avx-block2-kernel will be the only compiled kernel for complex case])
  else
  AC_MSG_FAILURE([--with-complex-avx-block2-kernel failed; AVX kernels cannot be compiled on the system])
  fi 
 else
  AC_MSG_FAILURE([--with-complex-avx-block2-kernel failed; A specific kernel for complex case has already been defined before!])
 fi
fi


dnl set the conditionals according to the previous tests
if test x"${can_use_iso_fortran_env}" = x"yes" ; then
 AC_DEFINE([HAVE_ISO_FORTRAN_ENV],[1],[can use module iso_fortran_env])
fi


AM_CONDITIONAL([WITH_REAL_GENERIC_KERNEL],[test x"$install_real_generic" = x"yes"])
if test x"${install_real_generic}" = x"yes" ; then
 AC_DEFINE([WITH_REAL_GENERIC_KERNEL],[1],[can use real generic kernel])
fi

AM_CONDITIONAL([WITH_COMPLEX_GENERIC_KERNEL],[test x"$install_complex_generic" = x"yes"])
if test x"${install_complex_generic}" = x"yes" ; then
 AC_DEFINE([WITH_COMPLEX_GENERIC_KERNEL],[1],[can use complex generic kernel])
fi

AM_CONDITIONAL([WITH_REAL_GENERIC_SIMPLE_KERNEL],[test x"$install_real_generic_simple" = x"yes"])
if test x"${install_real_generic_simple}" = x"yes" ; then
 AC_DEFINE([WITH_REAL_GENERIC_SIMPLE_KERNEL],[1],[can use real generic-simple kernel])
fi

AM_CONDITIONAL([WITH_COMPLEX_GENERIC_SIMPLE_KERNEL],[test x"$install_complex_generic_simple" = x"yes"])
if test x"${install_complex_generic_simple}" = x"yes" ; then
 AC_DEFINE([WITH_COMPLEX_GENERIC_SIMPLE_KERNEL],[1],[can use complex generic-simple kernel])
fi

AM_CONDITIONAL([WITH_REAL_SSE_KERNEL],[test x"$install_real_sse" = x"yes"])
if test x"${install_real_sse}" = x"yes" ; then
 AC_DEFINE([WITH_REAL_SSE_KERNEL],[1],[can use real SSE kernel])
fi

AM_CONDITIONAL([WITH_COMPLEX_SSE_KERNEL],[test x"$install_complex_sse" = x"yes"])
if test x"${install_complex_sse}" = x"yes" ; then
 AC_DEFINE([WITH_COMPLEX_SSE_KERNEL],[1],[can use complex SSE kernel])
fi

AM_CONDITIONAL([WITH_REAL_AVX_BLOCK2_KERNEL],[test x"$install_real_avx_block2" = x"yes"])
if test x"${install_real_avx_block2}" = x"yes" ; then
 AC_DEFINE([WITH_REAL_AVX_BLOCK2_KERNEL],[1],[can use real_avx_block2 kernel])
fi

AM_CONDITIONAL([WITH_REAL_AVX_BLOCK4_KERNEL],[test x"$install_real_avx_block4" = x"yes"])
if test x"${install_real_avx_block4}" = x"yes" ; then
 AC_DEFINE([WITH_REAL_AVX_BLOCK4_KERNEL],[1],[can use real_avx_block4 kernel])
fi

AM_CONDITIONAL([WITH_REAL_AVX_BLOCK6_KERNEL],[test x"$install_real_avx_block6" = x"yes"])
if test x"${install_real_avx_block6}" = x"yes" ; then
 AC_DEFINE([WITH_REAL_AVX_BLOCK6_KERNEL],[1],[can use real_avx_block6 kernel])
fi

AM_CONDITIONAL([WITH_COMPLEX_AVX_BLOCK1_KERNEL],[test x"$install_complex_avx_block1" = x"yes"])
if test x"${install_complex_avx_block1}" = x"yes" ; then
 AC_DEFINE([WITH_COMPLEX_AVX_BLOCK1_KERNEL],[1],[can use complex_avx_block1 kernel])
fi

AM_CONDITIONAL([WITH_COMPLEX_AVX_BLOCK2_KERNEL],[test x"$install_complex_avx_block2" = x"yes"])
if test x"${install_complex_avx_block2}" = x"yes" ; then
 AC_DEFINE([WITH_COMPLEX_AVX_BLOCK2_KERNEL],[1],[can use complex_avx_block2 kernel])
fi

AM_CONDITIONAL([WITH_REAL_BGP_KERNEL],[test x"$install_real_bgp" = x"yes"])
if test x"${install_real_bgp}" = x"yes" ; then
 AC_DEFINE([WITH_REAL_BGP_KERNEL],[1],[can use real BGP kernel])
fi

AM_CONDITIONAL([WITH_REAL_BGQ_KERNEL],[test x"$install_real_bgq" = x"yes"])
if test x"${install_real_bgq}" = x"yes" ; then
 AC_DEFINE([WITH_REAL_BGQ_KERNEL],[1],[can use real BGQ kernel])
fi

if test x"${use_specific_complex_kernel}" = x"yes" ; then
 AC_DEFINE([WITH_SPECIFIC_COMPLEX_KERNEL],[1],[use specific complex kernel])
fi

if test x"${use_specific_real_kernel}" = x"yes" ; then
 AC_DEFINE([WITH_SPECIFIC_COMPLEX_KERNEL],[1],[use specific real kernel])
fi

LT_INIT

AC_SUBST([AM_FCFLAGS])
AC_SUBST([AM_LDFLAGS])
AC_SUBST([WITH_MKL])
AC_SUBST([WITH_BLACS])
AC_SUBST([with_amd_bulldozer_kernel])
AC_SUBST([CFLAGS])
AC_SUBST([FCFLAGS])
AC_SUBST([CPPFLAGS])
AC_SUBST([LDFLAGS])
AC_SUBST([RANLIB])
AC_SUBST([FC_MODINC])
AC_SUBST([FC_MODOUT])
AC_SUBST(BLACS_LDFLAGS)
AC_SUBST(BLACS_FCFLAGS)

#AC_SUBST(OPT_FCFLAGS)

rm -rf modules/
mkdir modules

#gl_VISIBILITY
#AH_BOTTOM([#if HAVE_VISIBILITY
#define EXPORTED __attribute__((__visibility__("default")))
#define HIDDEN   __attribute__((__visibility__("hidden")))
#else
#define EXPORTED
#define HIDDEN
#endif])

AC_CONFIG_FILES([
  Makefile
  elpa.pc:elpa.pc.in
])
AC_OUTPUT
grep "^#define" config.h > config-f90.h
