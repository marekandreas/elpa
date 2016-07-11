AC_PREREQ([2.69])

# The version is set in elpa.spec, to have a single point of reference
AC_INIT([elpa],m4_esyscmd_s([awk '/^ *Version:/ {print $2;}' elpa.spec]), [elpa-library@mpcdf.mpg.de])
AC_SUBST([PACKAGE_VERSION])

AC_CONFIG_SRCDIR([src/elpa1.F90])

AM_INIT_AUTOMAKE([foreign -Wall subdir-objects])

# Without this, automake tries to be smart and rebuilt
# the autoconf generated files such as configure, aclocal.m4, etc.,
# in case the timestamps of files such as configure.ac are newer
#
# This only makes trouble for end users with out-of-date autoconf versions
# that cannot produce these files
AM_MAINTAINER_MODE([disable])

AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config.h])
AM_SILENT_RULES([yes])

# Set the libtool library version, see LIBRARY_INTERFACE
#
# See http://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
#
# [c:r:a]
#
# c: The currently implement interface
# r: The revision number of the current interface, that is the number
#    of released source-code changes for the current interface
# a: The "age" is the number of interfaces prior to c that are also supported
#    by the current interface, as they are ABI compatible (e.g. only new symbols
#    were added by the new interface)
#
AC_SUBST([ELPA_SO_VERSION], [6:0:2])
#


AX_CHECK_GNU_MAKE()
if test x$_cv_gnu_make_command = x ; then
        AC_MSG_ERROR([Need GNU Make])
fi

#AC_CHECK_PROG(CPP_FOUND,cpp,yes,no)
#if test x"${CPP_FOUND}" = xno; then
#  AC_MSG_ERROR([no cpp found])
#fi

# gnu-make fortran module dependencies
m4_include([fdep/fortran_dependencies.m4])
FDEP_F90_GNU_MAKE_DEPS

###

m4_include([m4/ax_elpa_openmp.m4])

AC_MSG_CHECKING(whether --enable-openmp is specified)
AC_ARG_ENABLE([openmp],
		AS_HELP_STRING([--enable-openmp],
			       [use OpenMP threading, default no.]),
	      [],
	      [enable_openmp=no])
AC_MSG_RESULT([${enable_openmp}])
AM_CONDITIONAL([WITH_OPENMP],[test x"$enable_openmp" = x"yes"])
if test x"${enable_openmp}" = x"yes"; then
	AC_DEFINE([WITH_OPENMP], [1], [use OpenMP threading])
fi

dnl mpi
AC_ARG_WITH(mpi, [AS_HELP_STRING([--with-mpi=[[yes|no]]], [compile with MPI. Default: yes])],,[with_mpi=yes])
AM_CONDITIONAL([WITH_MPI],[test x"$with_mpi" = x"yes"])
if test x"${with_mpi}" = x"yes"; then
       AC_DEFINE([WITH_MPI], [1], [use MPI])
fi

# C
AC_LANG([C])
AX_PROG_CC_MPI([test x"$with_mpi" = x"yes"],[found_mpi_c=yes],[found_mpi_c=no])
if test x"$with_mpi" = x"yes"; then
  if test x"$found_mpi_c" = x"no"; then
    AC_MSG_ERROR([Could not compile an MPI C program])
  fi
fi

if test x"${enable_openmp}" = x"yes"; then
  AX_ELPA_OPENMP
  if test "$ac_cv_prog_cc_openmp" = unsupported; then
    AC_MSG_ERROR([Could not compile a C program with OpenMP, adjust CFLAGS])
  fi
  CFLAGS="$OPENMP_CFLAGS $CFLAGS"
fi

AC_PROG_INSTALL
AM_PROG_AR
AM_PROG_AS
AM_PROG_CC_C_O

# Fortran

dnl check whether single precision is requested
AC_MSG_CHECKING(whether ELPA library should contain also single precision functions)
AC_ARG_ENABLE(single-precision,[AS_HELP_STRING([--enable-single-precision],
	                                   [build with single precision])],
              want_single_precision="yes", want_single_precision="no")
AC_MSG_RESULT([${want_single_precision}])

AC_LANG([Fortran])
m4_include([m4/ax_prog_fc_mpi.m4])
AX_PROG_FC_MPI([test x"$with_mpi" = x"yes"],[found_mpi_f=yes],[found_mpi_f=no])
if test x"$with_mpi" = x"yes"; then
  if test x"$found_mpi_f" = x"no"; then
    AC_MSG_ERROR([Could not compile an MPI Fortran program])
  fi
fi
if test x"${enable_openmp}" = x"yes"; then
  AX_ELPA_OPENMP
  if test "$ac_cv_prog_fc_openmp" = unsupported; then
    AC_MSG_ERROR([Could not compile a Fortran program with OpenMP, adjust FCFLAGS])
  fi
  FCFLAGS="$OPENMP_FCFLAGS $FCFLAGS"
fi

dnl check which MPI binray invokes a MPI job
if test x"$with_mpi" = x"yes"; then
  AC_CHECK_PROGS([MPI_BINARY], [mpiexec.hydra mpiexec mpirun], [no])
  if test x"$MPI_BINARY" = x"no"; then
    AC_MSG_ERROR([Could not find either of the MPI binaries: mpiexec.hydra, mpiexec, mpirun])
  fi
fi

## C++
#AC_LANG([C++])
#AC_PROG_CXX
#
#if test x"${enable_openmp}" = x"yes"; then
#  AX_ELPA_OPENMP
#  if test "$ac_cv_prog_cxx_openmp" = unsupported; then
#    AC_MSG_ERROR([Could not compile a C++ program with OpenMP, adjust CXXFLAGS])
#  fi
#  CXXFLAGS="$OPENMP_CXXFLAGS $CXXFLAGS"
#fi



dnl variables needed for the tests

dnl do NOT remove any variables here, until
dnl 1. you know 100% what you are doing
dnl 2. you tested ALL configure functionality afterwards
dnl Otherwise, you most likely break some functionality

dnl as default always define the generic kernels to be build
dnl this is only unset if gpu_support_only is defined, or
dnl other specific real/complex kernels are wanted

install_real_generic=yes
install_real_generic_simple=yes

install_complex_generic=yes
install_complex_generic_simple=yes

#want_avx=yes
#want_avx2=yes
#want_sse=yes

AC_LANG([C])

dnl build with the possibilty to redirect stdout and stderr
dnl per MPI task in a file
AC_MSG_CHECKING(whether stdout/stderr file redirect should be enabled)
AC_ARG_ENABLE([redirect],
		AS_HELP_STRING([--enable-redirect],
			       [for test programs, allow redirection of stdout/stderr per MPI taks in a file (useful for timing), default no.]),
	      [],
	      [enable_redirect=no])
AC_MSG_RESULT([${enable_redirect}])

dnl redirect
if test x"${enable_redirect}" = x"yes"; then
  AC_DEFINE([HAVE_REDIRECT], [1], [Redirect stdout and stderr of test programs per MPI tasks to a file])
fi
AM_CONDITIONAL([HAVE_REDIRECT],[test x"$enable_redirect" = x"yes"])

dnl build with ftimings support
AC_MSG_CHECKING(whether ELPA should be build with more detailed timing support)
AC_ARG_ENABLE([timing],
		AS_HELP_STRING([--enable-timing],
			       [more detailed timing, default no.]),
	      [],
	      [enable_timings=no])
AC_MSG_RESULT([${enable_timings}])

if test x"${enable_timings}" = x"yes"; then
  AC_DEFINE([HAVE_DETAILED_TIMINGS], [1], [Enable more timing])
fi
AM_CONDITIONAL([HAVE_DETAILED_TIMINGS],[test x"$enable_timings" = x"yes"])

dnl PAPI for ftimings
AC_ARG_WITH([papi],
		[AS_HELP_STRING([--with-papi],[Use PAPI to also measure flop count in the detailed timing (--enable-timing), disabled by default])],
		[],
		[with_papi="no"])
if test x"${enable_timings}" = x"yes"; then
  if test x"$with_papi" = x"yes" ; then
    AC_SEARCH_LIBS([PAPI_library_init],[papi],
                  [papi_found=yes],[papi_found=no])
    if test x"$papi_found" = x"no" ; then
      AC_MSG_ERROR(["Could not find usable PAPI installation, please install or adjust CFLAGS, LDFLAGS"])
    fi
    AC_DEFINE([HAVE_LIBPAPI], [1], [Use the PAPI library])
  fi
fi

AC_MSG_CHECKING(whether double-precision SSE assembly kernel can be compiled)

$CC -c $srcdir/src/elpa2_kernels/elpa2_kernels_asm_x86_64_double_precision.s -o test.o 2>/dev/null
if test "$?" == 0; then
  can_compile_sse_assembly=yes
  install_real_sse_assembly=yes
  install_complex_sse_assembly=yes
else
  can_compile_sse_assembly=no
  install_real_sse_assembly=no
  install_complex_sse_assembly=no
fi

rm -f ./test.o
AC_MSG_RESULT([${can_compile_sse_assembly}])

if test x"${want_single_precision}" = x"yes" ; then
  AC_MSG_CHECKING(whether single-precision SSE assembly kernel can be compiled)

  $CC -c $srcdir/src/elpa2_kernels/elpa2_kernels_asm_x86_64_single_precision.s -o test.o 2>/dev/null
  if test "$?" == 0; then
    can_compile_sse_assembly=yes
    install_real_sse_assembly=yes
    install_complex_sse_assembly=yes
  else
    can_compile_sse_assembly=no
    install_real_sse_assembly=no
    install_complex_sse_assembly=no
  fi
  rm -f ./test.o
  AC_MSG_RESULT([${can_compile_sse_assembly}])
  if test x"${can_compile_sse_assembly}" = x"no" ; then
    AC_MSG_WARN([Cannot compile single-precision SSE assembly kernel: disabling SSE assembly kernels alltogether])
  fi
fi

dnl check whether on can compile with sse-gcc intrinsics
AC_MSG_CHECKING(whether we can compile SSE with gcc intrinsics in C)
AC_COMPILE_IFELSE([AC_LANG_SOURCE([
 #include <x86intrin.h>
 int main(int argc, char **argv){
 double* q;
 __m128d h1 = _mm_loaddup_pd(q);
 return 0;
 }
 ])],
 [can_compile_sse_intrinsics=yes],
 [can_compile_sse_intrinsics=no]
)
AC_MSG_RESULT([${can_compile_sse_intrinsics}])

if test "${can_compile_sse_intrinsics}" = "yes"; then
  install_real_sse_intrinsics=yes
  install_real_sse_block2=yes
  install_real_sse_block4=yes
  install_real_sse_block6=yes

  install_complex_sse_intrinsics=yes
  install_complex_sse_block1=yes
  install_complex_sse_block2=yes
else
  install_real_sse_intrinsics=no
  install_real_sse_block2=no
  install_real_sse_block4=no
  install_real_sse_block6=no

  install_complex_sse_intrinsics=no
  install_complex_sse_block1=no
  install_complex_sse_block2=no
fi

dnl check whether one can compile with avx - gcc intrinsics

dnl first pass: try with specified CFLAGS and CXXFLAGS
AC_MSG_CHECKING([whether we can compile AVX intrinsics in C])
AC_COMPILE_IFELSE([AC_LANG_SOURCE([
 #include <x86intrin.h>
 int main(int argc, char **argv){
 double* q;
 __m256d a1_1 = _mm256_load_pd(q);
 return 0;
 }
 ])],
 [can_compile_avx=yes],
 [can_compile_avx=no]
)
AC_MSG_RESULT([${can_compile_avx}])

dnl if test "${can_compile_avx}" = "yes" ; then
dnl   AC_MSG_CHECKING([whether we can compile AVX intrinsics in C++])
dnl   AC_LANG_PUSH([C++])
dnl   AC_COMPILE_IFELSE([AC_LANG_SOURCE([
dnl    #include <x86intrin.h>
dnl    int main(int argc, char **argv){
dnl    double* q;
dnl    __m256d a1_1 = _mm256_load_pd(q);
dnl    return 0;
dnl    }
dnl    ])],
dnl    [can_compile_avx=yes],
dnl    [can_compile_avx=no]
dnl   )
dnl   AC_LANG_POP([C++])
dnl   AC_MSG_RESULT([${can_compile_avx}])
dnl   if test "${can_compile_avx}" = "no" ; then
dnl     AC_MSG_WARN([Cannot compile C++ with AVX: disabling AVX alltogether])
dnl   fi
dnl fi

AC_MSG_CHECKING([whether we can compile AVX2 intrinsics in C])
AC_COMPILE_IFELSE([AC_LANG_SOURCE([
 #include <x86intrin.h>
 int main(int argc, char **argv){
 double* q;
 __m256d q1 = _mm256_load_pd(q);
 __m256d y1 = _mm256_fmadd_pd(q1, q1, q1);
 return 0;
 }
 ])],
 [can_compile_avx2=yes],
 [can_compile_avx2=no]
)
AC_MSG_RESULT([${can_compile_avx2}])

dnl if test "${can_compile_avx2}" = "yes" ; then
dnl   AC_MSG_CHECKING([whether we can compile AVX2 intrinsics in C++])
dnl   AC_LANG_PUSH([C++])
dnl   AC_COMPILE_IFELSE([AC_LANG_SOURCE([
dnl    #include <x86intrin.h>
dnl    int main(int argc, char **argv){
dnl    double* q;
dnl    __m256d q1 = _mm256_load_pd(q);
dnl    __m256d y1 = _mm256_fmadd_pd(q1, q1, q1);
dnl    return 0;
dnl    }
dnl    ])],
dnl    [can_compile_avx2=yes],
dnl    [can_compile_avx2=no]
dnl   )
dnl   AC_LANG_POP([C++])
dnl   AC_MSG_RESULT([${can_compile_avx2}])
dnl   if test "${can_compile_avx2}" = "no" ; then
dnl     AC_MSG_WARN([Cannot compile C++ with AVX2!])
dnl   fi
dnl fi

if test "${can_compile_avx}" = "yes" ; then
  install_real_avx_block2=yes
  install_real_avx_block4=yes
  install_real_avx_block6=yes

  install_complex_avx_block1=yes
  install_complex_avx_block2=yes
else
  install_real_avx_block2=no
  install_real_avx_block4=no
  install_real_avx_block6=no

  install_complex_avx_block1=no
  install_complex_avx_block2=no
fi

if test "${can_compile_avx2}" = "yes" ; then
  install_real_avx2_block2=yes
  install_real_avx2_block4=yes
  install_real_avx2_block6=yes

  install_complex_avx2_block1=yes
  install_complex_avx2_block2=yes
else
  install_real_avx2_block2=no
  install_real_avx2_block4=no
  install_real_avx2_block6=no

  install_complex_avx2_block1=no
  install_complex_avx2_block2=no
fi

AM_CONDITIONAL([HAVE_SSE_ASSEMBLY],[test x"$can_compile_sse_assembly" = x"yes"])
if test x"${can_compile_sse_assembly}" = x"yes" ; then
  AC_DEFINE([HAVE_SSE_ASSEMBLY],[1],[assembly SSE is supported on this CPU])
fi
AM_CONDITIONAL([HAVE_SSE_INTRINSICS],[test x"$can_compile_sse_intrinsics" = x"yes"])
if test x"${can_compile_sse_intrinsics}" = x"yes" ; then
  AC_DEFINE([HAVE_SSE_INTRINSICS],[1],[gcc intrinsics SSE is supported on this CPU])
fi

AM_CONDITIONAL([HAVE_AVX],[test x"$can_compile_avx" = x"yes"])
if test x"${can_compile_avx}" = x"yes" ; then
  AC_DEFINE([HAVE_AVX],[1],[AVX is supported on this CPU])
fi
AM_CONDITIONAL([HAVE_AVX2],[test x"$can_compile_avx2" = x"yes"])
if test x"${can_compile_avx2}" = x"yes" ; then
  AC_DEFINE([HAVE_AVX2],[1],[AVX2 is supported on this CPU])
fi

dnl set the AVX optimization flags if this option is specified
AC_MSG_CHECKING(whether AVX optimization flags should be set automatically)
AC_ARG_WITH([avx-optimization],
		AS_HELP_STRING([--with-avx-optimization],
			       [use AVX optimization, default no.]),
	      [with_avx_optimization=yes],
	      [with_avx_optimization=no])
AC_MSG_RESULT([${with_avx_optimization}])
if test x"${with_avx_optimization}" = x"yes"; then
 CFLAGS="$CFLAGS -funsafe-loop-optimizations -funsafe-math-optimizations -ftree-vect-loop-version -ftree-vectorize"
 CXXFLAGS="$CXXFLAGS -funsafe-loop-optimizations -funsafe-math-optimizations -ftree-vect-loop-version -ftree-vectorize"
fi

AC_LANG([Fortran])
AC_FC_FREEFORM
AC_FC_MODULE_FLAG
AC_FC_MODULE_OUTPUT_FLAG
AC_FC_LIBRARY_LDFLAGS

save_FCFLAGS=$FCFLAGS
save_LDFLAGS=$LDFLAGS

AC_ARG_VAR([SCALAPACK_LDFLAGS],[Extra LDFLAGS necessary to link a program with Scalapack])
AC_ARG_VAR([SCALAPACK_FCFLAGS],[Extra FCFLAGS necessary to compile a Fortran program with Scalapack])

FCFLAGS="$FCFLAGS $SCALAPACK_FCFLAGS"
LDFLAGS="$LDFLAGS $SCALAPACK_LDFLAGS"

dnl check whether fortran error_unit is defined
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
AC_MSG_RESULT([${can_use_iso_fortran_env}])

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

  if test x"${with_mpi}" = x"yes"; then
    dnl test whether scalapack already contains blacs
    scalapack_libs="mpiscalapack scalapack scalapack-openmpi"
    old_LIBS="$LIBS"
    for lib in ${scalapack_libs}; do
      LIBS="-l${lib} ${old_LIBS}"
      AC_MSG_CHECKING([whether -l${lib} already contains a BLACS implementation])
      AC_LINK_IFELSE([AC_LANG_FUNC_LINK_TRY([blacs_gridinit])],[blacs_in_scalapack=yes],[blacs_in_scalapack=no])
      AC_MSG_RESULT([${blacs_in_scalapack}])
      if test x"${blacs_in_scalapack}" = x"yes"; then
        break
      fi
    done

    if test x"${blacs_in_scalapack}" = x"no"; then
      LIBS="${old_LIBS}"

      dnl Test for stand-alone blacs
      AC_SEARCH_LIBS([bi_f77_init],[mpiblacsF77init],[],[],[-lmpiblacs])

      dnl for debian
      AC_SEARCH_LIBS([blacs_gridinit],[blacs-openmpi],[have_blacs=yes],[have_blacs=no],[-lblacsCinit-openmpi -lscalapack-openmpi])
      if test x"${have_blacs}" = x"no"; then
        unset ac_cv_search_blacs_gridinit
      fi
      AC_SEARCH_LIBS([blacs_gridinit],[mpiblacs blacs],[have_blacs=yes],[have_blacs=no])

      if test x"${have_blacs}" = x"no"; then
        AC_MSG_ERROR([No usable BLACS found. If installed in a non-standard place, please specify suitable LDFLAGS and FCFLAGS as arguments to configure])
      fi
    fi

    AC_SEARCH_LIBS([pdtran],[$scalapack_libs],[have_scalapack=yes],[have_scalapack=no])

    if test x"${have_scalapack}" = x"no" ; then
      AC_MSG_ERROR([could not link with scalapack: specify path])
    fi
  fi

  dnl check whether we can link alltogehter
  AC_MSG_CHECKING([whether we can link a Fortran program with all blacs/scalapack])
  AC_LINK_IFELSE([AC_LANG_SOURCE([
    program dgemm_test

      integer , parameter:: M = 4, N = 3, K = 2
      real :: A(M,K), B(K,N), C(M,N)

      call dgemm('N','N',M,N,K,1.0,A,M,B,K,0.0,C,M)

     end program dgemm_test
     ])],
     [can_link_with_blacs_scalapack=yes],
     [can_link_with_blacs_scalapack=no]
   )
   AC_MSG_RESULT([${can_link_with_blacs_scalapack}])

   if test x"${can_link_with_blacs_scalapack}" = x"yes" ; then
     WITH_BLACS=1
   else
   AC_MSG_ERROR([We can neither link with MKL or another Scalpack. Please specify SCALAPACK_LDFLAGS and SCALAPACK_FCFLAGS!])
   fi
fi

dnl important: reset them again!
FCFLAGS=$save_FCFLAGS
LDFLAGS=$save_LDFLAGS

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
if test x"${fortran_can_check_environment}" = x"yes" ; then
 AC_DEFINE([HAVE_ENVIRONMENT_CHECKING],[1],[Fortran can query environment variables])
fi

dnl check whether GPU version is requested

#CUDA_INSTALL_PATH="/usr/local/cuda/"
#CUDA_SDK_INSTALL_PATH="/usr/local/NVIDIA_GPU_Computing_SDK"

AC_MSG_CHECKING(whether GPU support is requested)
AC_ARG_ENABLE(gpu-support,[AS_HELP_STRING([--enable-gpu-support],
	                                   [build ELPA2 with GPU-support])],
              want_gpu="yes", want_gpu="no")
AC_MSG_RESULT([${want_gpu}])


AC_ARG_WITH([cuda-path],[AS_HELP_STRING([--with-cuda-path=PATH],[prefix where CUDA is installed @<:@default=auto@:>@])],
            [CUDA_INSTALL_PATH=$withval], [with_cuda=auto])

AC_ARG_WITH([cuda-sdk-path],[AS_HELP_STRING([--with-cuda-sdk-path=PATH],[prefix where CUDA SDK is installed @<:@default=auto@:>@])],
            [CUDA_SDK_INSTALL_PATH=$withval],[with_cuda_sdk=auto])

dnl setup nvcc flags and use them in later tests
if test x"${want_gpu}" = x"yes" ; then
  AC_LANG_PUSH([C])
  CUDA_CFLAGS="$CUDA_CFLAGS -arch sm_35 -O2 -I$CUDA_INSTALL_PATH/include"
  LDFLAGS="$LDFLAGS -L$CUDA_INSTALL_PATH/lib64"
  NVCCFLAGS="$NVCCFLAGS $CUDA_CFLAGS $CUDA_LDFLAGS"
  NVCC="nvcc"
  AC_SUBST(NVCC)
  AC_SUBST(NVCCFLAGS)

  dnl check whether nvcc compiler is found
  AC_CHECK_PROG(nvcc_found,nvcc,yes,no)
  if test x"${nvcc_found}" = x"no" ; then
    AC_MSG_ERROR([nvcc not found; try to set the cuda-path or disable GPU support])
  fi

  dnl check whether we find cublas
  AC_SEARCH_LIBS([cublasDgemm],[cublas],[have_cublas=yes],[have_cublas=no])
  if test x"${have_cublas}" = x"no"; then
    AC_MSG_ERROR([Could not link cublas; try to set the cuda-path or disable GPU support])
  fi
  AC_SEARCH_LIBS([cudaMemcpy],[cudart],[have_cudart=yes],[have_cudart=no])
  if test x"${have_cudart}" = x"no"; then
    AC_MSG_ERROR([Could not link cudart; try to set the cuda-path or disable GPU support])
  fi
  AC_LANG_POP([C])
  install_gpu=yes
  can_compile_gpu=yes
fi

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
else
  install_real_bgp=no
  install_complex_bgp=no
fi

dnl check BGQ kernel
AC_MSG_CHECKING([whether we can compile with BGQ intrinsics])

AC_LINK_IFELSE([AC_LANG_SOURCE([
  program test_bgq
    VECTOR(REAL(8))::QPX_h2
    real*8         :: hh(10,2)
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
else
  install_real_bgq=no
  install_complex_bgq=no
fi

dnl Test possibility of 'use mpi', if requested
if test x"${with_mpi}" = x"yes" ; then
  AC_ARG_ENABLE([mpi-module],
                AS_HELP_STRING([--disable-mpi-module],
                               [Do not use the Fortran MPI module, get interfaces by 'include "mpif.h')]),
                [],
                [enable_mpi_module=yes])
  if test x"${enable_mpi_module}" = x"yes" ; then
    AC_MSG_CHECKING(whether Fortran mpi module can be used)
    AC_COMPILE_IFELSE([AC_LANG_SOURCE([
      program test_mpi_module
        use mpi
        real :: time
        time = MPI_WTime()
      end program
    ])],
      [can_use_fortran_mpi_module=yes],
      [can_use_fortran_mpi_module=no]
    )
    AC_MSG_RESULT([${can_use_fortran_mpi_module}])
    if test x"${can_use_fortran_mpi_module}" = x"yes" ; then
      AC_DEFINE([HAVE_MPI_MODULE],[1],[can use the Fortran mpi module])
    else
      AC_MSG_ERROR([Could not compile a Fortran program with an 'use mpi' statement. You can try again with --disable-mpi-module])
    fi
  fi
fi

dnl default value
use_specific_real_kernel=no
use_specific_complex_kernel=no

dnl GPU version only
m4_include([m4/ax_elpa_gpu_version_only.m4])
DEFINE_OPTION_GPU_SUPPORT_ONLY([gpu-version-only],[gpu-support],[install_gpu])

dnl last check whether user wants to compile only a specific kernel
dnl
m4_include([m4/ax_elpa_specific_kernels.m4])

dnl real kernels
dnl do not remove this variable it is needed in the macros

dnl generic kernel
DEFINE_OPTION_SPECIFIC_REAL_KERNEL([real-generic-kernel-only],[generic-kernel],[install_real_generic])

dnl generic-simple kernel
DEFINE_OPTION_SPECIFIC_REAL_KERNEL([real-generic-simple-kernel-only],[generic-simple-kernel],[install_real_generic_simple])

dnl sse kernel
DEFINE_OPTION_SPECIFIC_REAL_KERNEL([real-sse-assembly-kernel-only],[sse-assembly-kernel],[install_real_sse_assembly])

dnl bgp kernel
DEFINE_OPTION_SPECIFIC_REAL_KERNEL([real-bgp-kernel-only],[bgp-kernel],[install_real_bgp])

dnl bgq kernel
DEFINE_OPTION_SPECIFIC_REAL_KERNEL([real-bgq-kernel-only],[bgq-kernel],[install_real_bgq])

dnl real-sse-block2 kernel
DEFINE_OPTION_SPECIFIC_REAL_KERNEL([real-sse-block2-kernel-only],[real-sse-block2-kernel],[install_real_sse_block2])

dnl real-sse-block4 kernel
DEFINE_OPTION_SPECIFIC_REAL_KERNEL([real-sse-block4-kernel]-only,[real-sse-block4-kernel],[install_real_sse_block4])

dnl real-sse-block6 kernel
DEFINE_OPTION_SPECIFIC_REAL_KERNEL([real-sse-block6-kernel-only],[real-sse-block6-kernel],[install_real_sse_block6])

dnl real-avx-block2 kernel
DEFINE_OPTION_SPECIFIC_REAL_KERNEL([real-avx-block2-kernel-only],[real-avx-block2-kernel],[install_real_avx_block2])

dnl real-avx-block4 kernel
DEFINE_OPTION_SPECIFIC_REAL_KERNEL([real-avx-block4-kernel]-only,[real-avx-block4-kernel],[install_real_avx_block4])

dnl real-avx-block6 kernel
DEFINE_OPTION_SPECIFIC_REAL_KERNEL([real-avx-block6-kernel-only],[real-avx-block6-kernel],[install_real_avx_block6])

dnl real-avx2-block2 kernel
DEFINE_OPTION_SPECIFIC_REAL_KERNEL([real-avx2-block2-kernel-only],[real-avx2-block2-kernel],[install_real_avx2_block2])

dnl real-avx2-block4 kernel
DEFINE_OPTION_SPECIFIC_REAL_KERNEL([real-avx2-block4-kernel]-only,[real-avx2-block4-kernel],[install_real_avx2_block4])

dnl real-avx2-block6 kernel
DEFINE_OPTION_SPECIFIC_REAL_KERNEL([real-avx2-block6-kernel-only],[real-avx2-block6-kernel],[install_real_avx2_block6])

dnl complex kernels

dnl generic kernel
DEFINE_OPTION_SPECIFIC_COMPLEX_KERNEL([complex-generic-kernel-only],[generic-kernel],[install_complex_generic])

dnl generic-simple kernel
DEFINE_OPTION_SPECIFIC_COMPLEX_KERNEL([complex-generic-simple-kernel-only],[generic-simple-kernel],[install_complex_generic_simple])

dnl sse kernel
DEFINE_OPTION_SPECIFIC_COMPLEX_KERNEL([complex-sse-assembly-kernel-only],[sse-assembly-kernel],[install_complex_sse_assembly])

dnl complex-bqp kernel
DEFINE_OPTION_SPECIFIC_COMPLEX_KERNEL([complex-bgp-kernel-only],[bgp-kernel],[install_complex_bgp])

dnl complex-bqq kernel
DEFINE_OPTION_SPECIFIC_COMPLEX_KERNEL([complex-bgq-kernel-only],[bgq-kernel],[install_complex_bgq])

dnl complex-sse-block1 kernel
DEFINE_OPTION_SPECIFIC_COMPLEX_KERNEL([complex-sse-block1-kernel-only],[complex-sse-block1-kernel],[install_complex_sse_block1])

dnl complex-sse-block2 kernel
DEFINE_OPTION_SPECIFIC_COMPLEX_KERNEL([complex-sse-block2-kernel-only],[complex-sse-block2-kernel],[install_complex_sse_block2])

dnl complex-avx-block1 kernel
DEFINE_OPTION_SPECIFIC_COMPLEX_KERNEL([complex-avx-block1-kernel-only],[complex-avx-block1-kernel],[install_complex_avx_block1])

dnl complex-avx-block2 kernel
DEFINE_OPTION_SPECIFIC_COMPLEX_KERNEL([complex-avx-block2-kernel-only],[complex-avx-block2-kernel],[install_complex_avx_block2])

dnl complex-avx2-block1 kernel
DEFINE_OPTION_SPECIFIC_COMPLEX_KERNEL([complex-avx2-block1-kernel-only],[complex-avx2-block1-kernel],[install_complex_avx2_block1])

dnl complex-avx2-block2 kernel
DEFINE_OPTION_SPECIFIC_COMPLEX_KERNEL([complex-avx2-block2-kernel-only],[complex-avx2-block2-kernel],[install_complex_avx2_block2])


dnl set the conditionals according to the previous tests

if test x"${can_use_iso_fortran_env}" = x"yes" ; then
 AC_DEFINE([HAVE_ISO_FORTRAN_ENV],[1],[can use module iso_fortran_env])
fi

AM_CONDITIONAL([WITH_GPU_VERSION],[test x"$install_gpu" = x"yes"])
if test x"${install_gpu}" = x"yes" ; then
  AC_DEFINE([WITH_GPU_VERSION],[1],[enable GPU support])
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

AM_CONDITIONAL([WITH_REAL_SSE_ASSEMBLY_KERNEL],[test x"$install_real_sse_assembly" = x"yes"])
if test x"${install_real_sse_assembly}" = x"yes" ; then
 AC_DEFINE([WITH_REAL_SSE_ASSEMBLY_KERNEL],[1],[can use real SSE assembly kernel])
fi

AM_CONDITIONAL([WITH_COMPLEX_SSE_ASSEMBLY_KERNEL],[test x"$install_complex_sse_assembly" = x"yes"])
if test x"${install_complex_sse_assembly}" = x"yes" ; then
 AC_DEFINE([WITH_COMPLEX_SSE_ASSEMBLY_KERNEL],[1],[can use complex SSE assembly kernel])
fi

AM_CONDITIONAL([WITH_REAL_SSE_BLOCK2_KERNEL],[test x"$install_real_sse_block2" = x"yes"])
if test x"${install_real_sse_block2}" = x"yes" ; then
 AC_DEFINE([WITH_REAL_SSE_BLOCK2_KERNEL],[1],[can use real_sse_block2 kernel])
fi

AM_CONDITIONAL([WITH_REAL_SSE_BLOCK4_KERNEL],[test x"$install_real_sse_block4" = x"yes"])
if test x"${install_real_sse_block4}" = x"yes" ; then
 AC_DEFINE([WITH_REAL_SSE_BLOCK4_KERNEL],[1],[can use real_sse_block4 kernel])
fi

AM_CONDITIONAL([WITH_REAL_SSE_BLOCK6_KERNEL],[test x"$install_real_sse_block6" = x"yes"])
if test x"${install_real_sse_block6}" = x"yes" ; then
 AC_DEFINE([WITH_REAL_SSE_BLOCK6_KERNEL],[1],[can use real_sse_block6 kernel])
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

AM_CONDITIONAL([WITH_REAL_AVX2_BLOCK2_KERNEL],[test x"$install_real_avx2_block2" = x"yes"])
if test x"${install_real_avx2_block2}" = x"yes" ; then
 AC_DEFINE([WITH_REAL_AVX2_BLOCK2_KERNEL],[1],[can use real_avx2_block2 kernel])
fi

AM_CONDITIONAL([WITH_REAL_AVX2_BLOCK4_KERNEL],[test x"$install_real_avx2_block4" = x"yes"])
if test x"${install_real_avx2_block4}" = x"yes" ; then
 AC_DEFINE([WITH_REAL_AVX2_BLOCK4_KERNEL],[1],[can use real_avx2_block4 kernel])
fi

AM_CONDITIONAL([WITH_REAL_AVX2_BLOCK6_KERNEL],[test x"$install_real_avx2_block6" = x"yes"])
if test x"${install_real_avx2_block6}" = x"yes" ; then
 AC_DEFINE([WITH_REAL_AVX2_BLOCK6_KERNEL],[1],[can use real_avx2_block6 kernel])
fi

AM_CONDITIONAL([WITH_COMPLEX_SSE_BLOCK1_KERNEL],[test x"$install_complex_sse_block1" = x"yes"])
if test x"${install_complex_sse_block1}" = x"yes" ; then
 AC_DEFINE([WITH_COMPLEX_SSE_BLOCK1_KERNEL],[1],[can use complex_sse_block1 kernel])
fi

AM_CONDITIONAL([WITH_COMPLEX_SSE_BLOCK2_KERNEL],[test x"$install_complex_sse_block2" = x"yes"])
if test x"${install_complex_sse_block2}" = x"yes" ; then
 AC_DEFINE([WITH_COMPLEX_SSE_BLOCK2_KERNEL],[1],[can use complex_sse_block2 kernel])
fi

AM_CONDITIONAL([WITH_COMPLEX_AVX_BLOCK1_KERNEL],[test x"$install_complex_avx_block1" = x"yes"])
if test x"${install_complex_avx_block1}" = x"yes" ; then
 AC_DEFINE([WITH_COMPLEX_AVX_BLOCK1_KERNEL],[1],[can use complex_avx_block1 kernel])
fi

AM_CONDITIONAL([WITH_COMPLEX_AVX_BLOCK2_KERNEL],[test x"$install_complex_avx_block2" = x"yes"])
if test x"${install_complex_avx_block2}" = x"yes" ; then
 AC_DEFINE([WITH_COMPLEX_AVX_BLOCK2_KERNEL],[1],[can use complex_avx_block2 kernel])
fi

AM_CONDITIONAL([WITH_COMPLEX_AVX2_BLOCK1_KERNEL],[test x"$install_complex_avx2_block1" = x"yes"])
if test x"${install_complex_avx2_block1}" = x"yes" ; then
 AC_DEFINE([WITH_COMPLEX_AVX2_BLOCK1_KERNEL],[1],[can use complex_avx2_block1 kernel])
fi

AM_CONDITIONAL([WITH_COMPLEX_AVX2_BLOCK2_KERNEL],[test x"$install_complex_avx2_block2" = x"yes"])
if test x"${install_complex_avx2_block2}" = x"yes" ; then
 AC_DEFINE([WITH_COMPLEX_AVX2_BLOCK2_KERNEL],[1],[can use complex_avx2_block2 kernel])
fi

AM_CONDITIONAL([WITH_REAL_BGP_KERNEL],[test x"$install_real_bgp" = x"yes"])
if test x"${install_real_bgp}" = x"yes" ; then
 AC_DEFINE([WITH_REAL_BGP_KERNEL],[1],[can use real BGP kernel])
fi

AM_CONDITIONAL([WITH_REAL_BGQ_KERNEL],[test x"$install_real_bgq" = x"yes"])
if test x"${install_real_bgq}" = x"yes" ; then
 AC_DEFINE([WITH_REAL_BGQ_KERNEL],[1],[can use real BGQ kernel])
fi

if test x"${use_specific_complex_kernel}" = x"no" ; then
  AC_DEFINE([WITH_NO_SPECIFIC_COMPLEX_KERNEL],[1],[do not use only one specific complex kernel (set at compile time)])
else
  AC_DEFINE([WITH_ONE_SPECIFIC_COMPLEX_KERNEL],[1],[use only one specific complex kernel (set at compile time)])
fi

if test x"${use_specific_real_kernel}" = x"no" ; then
  AC_DEFINE([WITH_NO_SPECIFIC_REAL_KERNEL],[1],[do not use only one specific real kernel (set at compile time)])
else
  AC_DEFINE([WITH_ONE_SPECIFIC_REAL_KERNEL],[1],[use only one specific real kernel (set at compile time)])
fi

LT_INIT

DX_PDF_FEATURE(OFF)
DX_PS_FEATURE(OFF)
DX_MAN_FEATURE(ON)
DX_HTML_FEATURE(ON)
DX_INIT_DOXYGEN([ELPA], [Doxyfile], [docs])

DESPERATELY_WANT_ASSUMED_SIZE=no
AC_MSG_CHECKING(whether --enable-assumed-size is specified)
AC_ARG_ENABLE([assumed-size],
                AS_HELP_STRING([--enable-assumed-size],
                               [use assumed-size Fortran arrays]),
              [],
              [DESPERATELY_WANT_ASSUMED_SIZE=no])
AC_MSG_RESULT([${DESPERATELY_WANT_ASSUMED_SIZE}])
AM_CONDITIONAL([WITH_DESPERATELY_WANT_ASSUMED_SIZE],[test x"$DESPERATELY_WANT_ASSUMED_SIZE" = x"yes"])
if test x"${DESPERATELY_WANT_ASSUMED_SIZE}" = x"yes" ; then
  AC_DEFINE([DESPERATELY_WANT_ASSUMED_SIZE],[1],[use assumed size Fortran arrays, even if not debuggable])
fi

if test x"${want_single_precision}" = x"yes" ; then
  AC_DEFINE([WANT_SINGLE_PRECISION_REAL],[1],[build also single-precision for real calculation])
  AC_DEFINE([WANT_SINGLE_PRECISION_COMPLEX],[1],[build also single-precision for complex calculation])
fi
AM_CONDITIONAL([WANT_SINGLE_PRECISION_REAL],[test x"$want_single_precision" = x"yes"])
AM_CONDITIONAL([WANT_SINGLE_PRECISION_COMPLEX],[test x"$want_single_precision" = x"yes"])

AC_SUBST([MPI_BINARY])
AC_SUBST([WITH_MKL])
AC_SUBST([WITH_BLACS])
AC_SUBST([with_amd_bulldozer_kernel])
AC_SUBST([FC_MODINC])
AC_SUBST([FC_MODOUT])
AC_SUBST([OPENMP_CFLAGS])
AC_SUBST([OPENMP_FCFLAGS])
AC_SUBST([OPENMP_LDFLAGS])
AC_SUBST([DOXYGEN_OUTPUT_DIR], [docs])

rm -rf modules/ private_modules/ .fortran_dependencies/
mkdir modules private_modules

#gl_VISIBILITY
#AH_BOTTOM([#if HAVE_VISIBILITY
#define EXPORTED __attribute__((__visibility__("default")))
#define HIDDEN   __attribute__((__visibility__("hidden")))
#else
#define EXPORTED
#define HIDDEN
#endif])


# Some part of libtool is too smart and tries to parse the output of
#   gfortran -v
# and catches anything that resembles a -l library link specification.
# Unfortunately, recent versions of gfortran emit
#   -l gfortran
# with a space between -l and gfortran. The empty -l is then included
# into "postdeps_FC" and causes linking errors later on.
postdeps_FC=$(echo $postdeps_FC | sed 's/-l //g')

if test x"${with_mpi}" = x"yes"; then
  if test x"${enable_openmp}" = x"yes"; then
	SUFFIX="_openmp"
  else
	SUFFIX=""
  fi
else
  if test x"${enable_openmp}" = x"yes"; then
	SUFFIX="_onenode_openmp"
  else
	SUFFIX="_onenode"
  fi
fi

AC_SUBST([SUFFIX])
AC_SUBST([PKG_CONFIG_FILE],[elpa${SUFFIX}-${PACKAGE_VERSION}.pc])

AC_CONFIG_FILES([
  Makefile
  Doxyfile
  ${PKG_CONFIG_FILE}:elpa.pc.in
])

AC_OUTPUT

if test "${can_compile_avx}" = "no" ; then
#  if test x"${want_avx}" = x"yes" ; then
    AC_MSG_WARN([Could not compile AVX instructions])
#  fi
fi
if test "${can_compile_avx2}" = "no" ; then
#  if test x"${want_avx2}" = x"yes" ; then
    AC_MSG_WARN([Could not compile AVX2 instructions])
#  fi
fi

if test "${can_compile_sse}" = "no" ; then
  AC_MSG_WARN([Could not compile SSE instructions])
fi

make -f $srcdir/generated_headers.am generated-headers top_srcdir="$srcdir"
