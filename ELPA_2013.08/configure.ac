AC_PREREQ([2.69])
AC_INIT([elpa],[2013.08.002], elpa-library@rzg.mpg.de)
AC_CONFIG_SRCDIR([src/elpa1.f90])

AM_INIT_AUTOMAKE([foreign -Wall subdir-objects])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config.h])
#AM_SILENT_RULES([yes])


AC_PROG_INSTALL
AM_PROG_CC_C_O
AM_PROG_AR
AM_PROG_AS
AC_PROG_CXX

AC_LANG(Fortran)
m4_include([m4/ax_prog_fc_mpi.m4])
dnl check whether an mpi compiler is available;
dnl if not abort since it is mandatory
AX_PROG_FC_MPI([],[have_mpi=yes],[have_mpi=no
		 if test "x${have_mpi}" = xno; then
  		  AC_MSG_ERROR([no mpi found])
		  fi])


AC_SUBST([ELPA_LIB_VERSION], [2013.08.002])
# this is the version of the API, should be changed in the major revision
# if and only if the actual API changes
AC_SUBST([ELPA_SO_VERSION], [0:0:0])

AC_FC_FREEFORM
AC_FC_MODULE_FLAG
AC_FC_MODULE_OUTPUT_FLAG



dnl  macro for an --with-$2 switch that sets the
dnl  preprocessor define $1, with description $3, default $4, possible values $5
AC_DEFUN([DEFINE_OPTION],[
  AC_MSG_CHECKING(whether the $2 kernel is specified)
  AC_ARG_WITH([$2],
		AS_HELP_STRING([--with-$2],
			       [$3, default $4.]),
	      [with_option=yes],
	      [with_option=$4])
  AC_MSG_RESULT([${with_option}])
  AM_CONDITIONAL([$1],[test x"$with_option" = x"yes"])
  if test "x${with_option}" = xyes; then
	AC_DEFINE([$1], [1], [$3])
#  elif test "x${with_option}" = xno; then
#	AC_DEFINE([$1], [0], [$3])
  fi
])

DEFINE_OPTION([WITH_GENERIC], [generic],
	      [use generic kernel for all architectures (with some hand-coded optimizations)],
	      [no],[])

DEFINE_OPTION([WITH_GENERIC_SIMPLE], [generic-simple],
	      [use generic simple kernel for all architectures (without any hand-coded optimizations)],
	      [no],[])

DEFINE_OPTION([WITH_BGP], [bgp],
	      [use optimized kernel for IBM BG/P],
	      [no],[])

DEFINE_OPTION([WITH_BGQ], [bgq],
	      [use optimized kernel for IBM BG/Q],
	      [no],[])

DEFINE_OPTION([WITH_SSE_AS], [sse-assembler],
	      [use kernel tuned for SSE (written in gcc assembler)],
	      [no],[])

DEFINE_OPTION([WITH_AVX_SANDYBRIDGE], [avx-sandybridge],
	      [use kernel tuned for AVX on Intel Sandybridge (written in gcc assembler)],
	      [no],[])

DEFINE_OPTION([WITH_AMD_BULLDOZER], [amd-bulldozer],
	      [use kernel tuned for AVX on AMD Bulldozer (written in gcc assembler)],
	      [no],[])

DEFINE_OPTION([WITH_AVX_COMPLEX_BLOCK1], [avx-complex-block1],
	      [use AVX optimized complex kernel with blocking 1 (written in gcc assembler)],
	      [no],[])

DEFINE_OPTION([WITH_AVX_COMPLEX_BLOCK2], [avx-complex-block2],
	      [use AVX optimized complex kernel with blocking 2 (written in gcc assembler)],
	      [no],[])

DEFINE_OPTION([WITH_AVX_REAL_BLOCK2], [avx-real-block2],
	      [use AVX optimized real kernel with blocking 2 (written in gcc assembler)],
	      [no],[])

DEFINE_OPTION([WITH_AVX_REAL_BLOCK4], [avx-real-block4],
	      [use AVX optimized real kernel with blocking 4 (written in gcc assembler)],
	      [no],[])

DEFINE_OPTION([WITH_AVX_REAL_BLOCK6], [avx-real-block6],
	      [use AVX optimized real kernel with blocking 6 (written in gcc assembler)],
	      [no],[])



save_FCFLAGS=$FCFLAGS
save_LDFLAGS=$LDFLAGS

FCFLAGS="$FCFLAGS $BLACS_FCFLAGS"
LDFLAGS="$LDFLAGS $BLACS_LDFLAGS"
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
    [can_link_with_mkl=yes],
    [can_link_with_mkl=no]
  )
  AC_MSG_RESULT([${can_link_with_mkl}])
fi

dnl if not mkl, check all the necessary individually
if test "x${can_link_with_mkl}" = "xyes" ; then
  WITH_MKL=1
else

  dnl first check blas
  AC_SEARCH_LIBS([dgemm],[blas],[can_link_with_blas=yes],[can_link_with_blas=no])
  AC_MSG_CHECKING([whether we can link a program with a blas lib])
  AC_MSG_RESULT([${can_link_with_blas}])

  if test "x${can_link_with_blas}" = "xno" ; then
    AC_MSG_ERROR([could not link with blas: specify path]) 
  fi	
  dnl now lapack
  AC_SEARCH_LIBS([dlarrv],[lapack],[can_link_with_lapack=yes],[can_link_with_lapack=no])
  AC_MSG_CHECKING([whether we can link a program with a lapack lib])
  AC_MSG_RESULT([${can_link_with_lapack}])

  if test "x${can_link_with_lapack}" = "xno" ; then
    AC_MSG_ERROR([could not link with lapack: specify path]) 
  fi
	
  dnl now blacs
  AC_SEARCH_LIBS([blacs_gridinit],[mpiblacs],[can_link_with_blacs=yes],[can_link_with_blacs=no])
  AC_MSG_CHECKING([whether we can link a program with a blacs lib])
  AC_MSG_RESULT([${can_link_with_blacs}])

  if test "x${can_link_with_blacs}" = "xno" ; then
    AC_MSG_ERROR([could not link with blacs: specify path]) 
  fi
	
  dnl now scalapack
  AC_SEARCH_LIBS([pdtran],[mpiscalapack],[can_link_with_scalapack=yes],[can_link_with_scalapack=no])
  AC_MSG_CHECKING([whether we can link a program with a scalapack lib])
  AC_MSG_RESULT([${can_link_with_scalapack}])

  if test "x${can_link_with_scalapack}" = "xno" ; then
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

   if test "x${can_link_with_blacs_scalapack}" = "xyes" ; then
     WITH_BLACS=1
   else
   AC_MSG_ERROR([We can neither link with MKL or another Scalpack. Please specify BLACS_LDFLAGS and BLACS_FCFLAGS!])
   fi
fi

dnl important: reset them again!
FCFLAGS=$save_FCFLAGS
LDFLAGS=$save_LDFLAGS



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
