AC_PREREQ([2.69])

# The version is set in elpa.spec, to have a single point of reference
AC_INIT([elpa],m4_esyscmd_s([awk '/^ *Version:/ {print $2;}' elpa.spec]), [elpa-library@mpcdf.mpg.de])
AC_SUBST([PACKAGE_VERSION])

AC_CONFIG_SRCDIR([src/elpa.F90])

AM_INIT_AUTOMAKE([foreign -Wall subdir-objects])

AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config.h])
AM_SILENT_RULES([yes])

# ABI version
#
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
AC_SUBST([ELPA_SO_VERSION], [9:0:1])

# API Version
AC_DEFINE([EARLIEST_API_VERSION], [20170403], [Earliest supported ELPA API version])
AC_DEFINE([CURRENT_API_VERSION], [20170403], [Current ELPA API version])


AX_CHECK_GNU_MAKE()
if test x$_cv_gnu_make_command = x ; then
        AC_MSG_ERROR([Need GNU Make])
fi

AC_ARG_ENABLE([legacy],
              AS_HELP_STRING([--disable-legacy],
                             [do not build legacy API, default yes]),
              [],
              [enable_legacy=yes])
AM_CONDITIONAL([ENABLE_LEGACY],[test x"$enable_legacy" = x"yes"])

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
AC_LANG_PUSH([C])
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
AC_PROG_CC_C99
AM_PROG_CC_C_O
AC_LANG_POP([C])

# Fortran
AC_LANG_PUSH([Fortran])
m4_include([m4/ax_prog_fc_mpi.m4])
AX_PROG_FC_MPI([test x"$with_mpi" = x"yes"],[found_mpi_f=yes],[found_mpi_f=no])
if test x"$with_mpi" = x"yes"; then
  if test x"$found_mpi_f" = x"no"; then
    AC_MSG_ERROR([Could not compile an MPI Fortran program])
  fi
fi

AC_FC_SRCEXT([F90])
AC_FC_FREEFORM
AC_FC_MODULE_FLAG
AC_FC_MODULE_OUTPUT_FLAG
AC_FC_LIBRARY_LDFLAGS


if test x"${enable_openmp}" = x"yes"; then
  AX_ELPA_OPENMP
  if test "$ac_cv_prog_fc_openmp" = unsupported; then
    AC_MSG_ERROR([Could not compile a Fortran program with OpenMP, adjust FCFLAGS])
  fi
  FCFLAGS="$OPENMP_FCFLAGS $FCFLAGS"
fi

dnl check which MPI binray invokes a MPI job
if test x"$with_mpi" = x"yes"; then
  AC_CHECK_PROGS([MPI_BINARY], [srun mpiexec.hydra mpiexec mpirun poe runjob], [no])
  if test x"$MPI_BINARY" = x"no"; then
    AC_MSG_ERROR([Could not find either of the MPI binaries: mpiexec.hydra, mpiexec, mpirun, poe, runjob, srun])
  fi
fi

dnl build with the possibilty to redirect stdout and stderr
dnl per MPI task in a file
AC_MSG_CHECKING(whether stdout/stderr file redirect should be enabled)
AC_ARG_ENABLE([redirect],
              [AS_HELP_STRING([--enable-redirect],
                              [for test programs, allow redirection of stdout/stderr per MPI taks in a file (useful for timing), default no.])],
              [],
              [enable_redirect=no])
AC_MSG_RESULT([${enable_redirect}])

dnl check whether single precision is requested
AC_MSG_CHECKING(whether ELPA library should contain also single precision functions)
AC_ARG_ENABLE(single-precision,
              [AS_HELP_STRING([--enable-single-precision],
                              [build with single precision])],
              [want_single_precision="$enableval"],
              [want_single_precision="no"])
AC_MSG_RESULT([${want_single_precision}])


dnl redirect
if test x"${enable_redirect}" = x"yes"; then
  AC_DEFINE([HAVE_REDIRECT], [1], [Redirect stdout and stderr of test programs per MPI tasks to a file])
fi
AM_CONDITIONAL([HAVE_REDIRECT],[test x"$enable_redirect" = x"yes"])

dnl build with ftimings support

AC_ARG_ENABLE([timings],
              [AS_HELP_STRING([--disable-timings],
                              [more detailed timing, default yes])],
              [],
              [enable_timings=yes])

if test x"${enable_timings}" = x"yes"; then
  AC_DEFINE([HAVE_DETAILED_TIMINGS], [1], [Enable more timing])
fi
AM_CONDITIONAL([HAVE_DETAILED_TIMINGS], [test x"$enable_timings" = x"yes"])

dnl PAPI for ftimings
AC_ARG_WITH([papi],
            [AS_HELP_STRING([--with-papi],
                            [Use PAPI to also measure flop count in the detailed timing (--enable-timing), disabled by default])],
            [],
            [with_papi="no"])
if test x"${enable_timings}" = x"yes"; then
  if test x"$with_papi" = x"yes" ; then
    AC_SEARCH_LIBS([PAPI_library_init], [papi], [papi_found=yes], [papi_found=no])
    if test x"$papi_found" = x"no" ; then
      AC_MSG_ERROR(["Could not find usable PAPI installation, please install or adjust CFLAGS, LDFLAGS"])
    fi
    AC_DEFINE([HAVE_LIBPAPI], [1], [Use the PAPI library])
  fi
fi

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
    use iso_fortran_env, only : error_unit
    implicit none

    write(error_unit,*) "error_unit is defined"
  end program
])],
  [can_use_iso_fortran_env=yes],
  [can_use_iso_fortran_env=no]
)
AC_MSG_RESULT([${can_use_iso_fortran_env}])
if test x"${can_use_iso_fortran_env}" = x"yes" ; then
 AC_DEFINE([HAVE_ISO_FORTRAN_ENV],[1],[can use module iso_fortran_env])
fi


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
  AC_SEARCH_LIBS([dgemm],[openblas satlas blas],[have_blas=yes],[have_blas=no])
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


dnl check whether BAND_TO_FULL_BLOCKING is set
AC_MSG_CHECKING(whether BAND_TO_FLULL_BLOCKING is requested)
AC_ARG_ENABLE(band-to-full-blocking,[AS_HELP_STRING([--disable-band-to-full-blocking],
                                                    [build ELPA2 with blocking in band_to_full (default: enabled)])],
              [],
	      [enable_band_to_full_blocking="yes"])
AC_MSG_RESULT([${enable_band_to_full_blocking}])

AM_CONDITIONAL([BAND_TO_FULL_BLOCKING],[test x"$enable_band_to_full_blocking" = x"yes"])
if test x"${enable_band_to_full_blocking}" = x"yes"; then
       AC_DEFINE([BAND_TO_FULL_BLOCKING], [1], [use blocking in trans_ev_band_to_full])
fi


AC_ARG_WITH([cuda-path],[AS_HELP_STRING([--with-cuda-path=PATH],[prefix where CUDA is installed @<:@default=auto@:>@])],
            [CUDA_INSTALL_PATH=$withval], [with_cuda=auto])

AC_ARG_WITH([cuda-sdk-path],[AS_HELP_STRING([--with-cuda-sdk-path=PATH],[prefix where CUDA SDK is installed @<:@default=auto@:>@])],
            [CUDA_SDK_INSTALL_PATH=$withval],[with_cuda_sdk=auto])


dnl setup nvcc flags and use them in later tests
user_sets_gpu_compute_capability="no"
AC_MSG_CHECKING(whether a GPU compute capability is specified)
AC_ARG_WITH([GPU-compute-capability],
            [AS_HELP_STRING([--with-GPU-compute-capability=VALUE],
                            [use compute capability VALUE for GPU version, default: "sm_35"])],
            [user_sets_gpu_compute_capability="yes"],[cuda_compute_capability="sm_35"])
AC_MSG_RESULT([${user_sets_gpu_compute_capability}])


dnl sanity check whether compute capability setting by user is reasonable
if test x"${user_sets_gpu_compute_capability}" = x"yes" ; then
  dnl the user must set a value which starts with "sm_"
  value=$(echo $withval | cut -c1-3)
  if test x"${value}" = x"sm_" ; then
    cuda_compute_capability=$withval
  else
    AC_MSG_ERROR([Unknown GPU compute capability set: ${withval}])
  fi
fi


dnl Test possibility of 'use mpi', if requested
if test x"${with_mpi}" = x"yes" ; then
  AC_ARG_ENABLE([mpi-module],
                AS_HELP_STRING([--disable-mpi-module],
                               [do not use the Fortran MPI module, get interfaces by 'include "mpif.h')]),
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
AC_LANG_POP([Fortran])

dnl Assemble the list of kernels to build
m4_pattern_forbid([elpa_m4])

m4_define(elpa_m4_generic_kernels, [
        real_generic
        real_generic_simple
        complex_generic
        complex_generic_simple
])

m4_define(elpa_m4_sse_assembly_kernels, [
        real_sse_assembly
        complex_sse_assembly
])

m4_define(elpa_m4_sse_kernels, [
        real_sse_block2
        real_sse_block4
        real_sse_block6
        complex_sse_block1
        complex_sse_block2
])

m4_define(elpa_m4_avx_kernels, [
        real_avx_block2
        real_avx_block4
        real_avx_block6
        complex_avx_block1
        complex_avx_block2
])

m4_define(elpa_m4_avx2_kernels, [
        real_avx2_block2
        real_avx2_block4
        real_avx2_block6
        complex_avx2_block1
        complex_avx2_block2
])

m4_define(elpa_m4_avx512_kernels, [
        real_avx512_block2
        real_avx512_block4
        real_avx512_block6
        complex_avx512_block1
        complex_avx512_block2
])

m4_define(elpa_m4_bgp_kernels, [
        real_bgp
        complex_bgp
])

m4_define(elpa_m4_bgq_kernels, [
        real_bgq
        complex_bgq
])

m4_define(elpa_m4_gpu_kernels, [
        real_gpu
        complex_gpu
])

m4_define(elpa_m4_kernel_types, [generic sse sse_assembly avx avx2 avx512 bgp bgq gpu])

m4_define(elpa_m4_all_kernels,
          m4_foreach_w([elpa_m4_type],
                       elpa_m4_kernel_types,
                       [m4_foreach_w([elpa_m4_kernel],[elpa_m4_]elpa_m4_type[_kernels],elpa_m4_kernel )]))

m4_define(elpa_m4_real_kernels,
          m4_foreach_w(elpa_m4_kernel,
                       elpa_m4_all_kernels,
                       [m4_bmatch(elpa_m4_kernel,real,elpa_m4_kernel) ]))

m4_define(elpa_m4_complex_kernels,
          m4_foreach_w(elpa_m4_kernel,
                       elpa_m4_all_kernels,
                       [m4_bmatch(elpa_m4_kernel,complex,elpa_m4_kernel) ]))

dnl
dnl ELPA_SELECT_KERNELS([flagname],[default])
dnl
dnl  default should be one of `enable' or `disable'
dnl
AC_DEFUN([ELPA_SELECT_KERNELS], [
        AC_ARG_ENABLE(m4_bpatsubst($1,[_],[-]),[AS_HELP_STRING([--]m4_case([$2],[enable],[disable],[disable],[enable])[-]m4_bpatsubst($1,[_],[-]),
                m4_case([$2],[enable],[do not build],[disable],[build])[ ]m4_toupper($1)[ kernels, default: $2d])],
                      [],[enable_$1=]m4_case([$2],[enable],[yes],[disable],[no]))
        m4_foreach_w(elpa_m4_kernel,elpa_m4_$1_kernels,[
                use_[]elpa_m4_kernel[=$enable_$1]
        ])
])

dnl Modify list of kernels with configure arguments
ELPA_SELECT_KERNELS([generic],[enable])
ELPA_SELECT_KERNELS([sse],[enable])
ELPA_SELECT_KERNELS([sse_assembly],[enable])
ELPA_SELECT_KERNELS([avx],[enable])
ELPA_SELECT_KERNELS([avx2],[enable])
ELPA_SELECT_KERNELS([avx512],[disable])
ELPA_SELECT_KERNELS([gpu],[disable])
ELPA_SELECT_KERNELS([bgp],[disable])
ELPA_SELECT_KERNELS([bgq],[disable])

m4_foreach_w([elpa_m4_kind],[real complex],[
        default_[]elpa_m4_kind[]_kernel=""
])

if test x"${enable_bgp}" = x"yes" -o x"$enable_bgq" = x"yes"; then
        m4_foreach_w([elpa_m4_kernel], elpa_m4_sse_kernels elpa_m4_avx_kernels elpa_m4_avx2_kernels elpa_m4_avx512_kernels, [
                if x"$use_[]elpa_m4_kernel[]" = x"yes" ; then
                        echo "Disabling elpa_m4_kernel due to BGP/BGQ option"
                fi
                use_[]elpa_m4_kernel[]=no
        ])
fi


dnl Deal with --with-fixed-[real|complex]-kernel arguments
m4_foreach_w([elpa_m4_kind],[real complex],[
        AC_ARG_WITH([fixed-]elpa_m4_kind[-kernel], m4_expand([AS_HELP_STRING([--with-fixed-]elpa_m4_kind[-kernel]=KERNEL,
                [compile with only a single specific ]elpa_m4_kind[ kernel. Available kernels are:]
                 m4_foreach_w([elpa_m4_kernel],m4_expand(elpa_m4_[]elpa_m4_kind[]_kernels),[m4_bpatsubst(elpa_m4_kernel,elpa_m4_kind[]_,[]) ]))]),
                 [fixed_]elpa_m4_kind[_kernel="]elpa_m4_kind[_$withval"],[fixed_]elpa_m4_kind[_kernel=""])
        if test -n "$fixed_[]elpa_m4_kind[]_kernel" ; then
                found="no"
                m4_foreach_w([elpa_m4_otherkernel],m4_expand(elpa_m4_[]elpa_m4_kind[]_kernels),[
                        if test "$fixed_]elpa_m4_kind[_kernel" = "]elpa_m4_otherkernel[" ; then
                                use_[]elpa_m4_otherkernel[]=yes
                                found="yes"
                        else
                                use_[]elpa_m4_otherkernel[]=no
                        fi
                ])
                if test x"$found" = x"no" ; then
                        AC_MSG_ERROR([Invalid kernel "$fixed_]elpa_m4_kind[_kernel" specified for --with-fixed-]elpa_m4_kind[-kernel])
                fi
                default_[]elpa_m4_kind[]_kernel="$fixed_[]elpa_m4_kind[]_kernel"
                AC_DEFINE([WITH_FIXED_]m4_toupper(elpa_m4_kind)[_KERNEL],[1],[use only one specific ]elpa_m4_kind[ kernel (set at compile time)])
        fi
])

AC_ARG_WITH(gpu-support-only, [AS_HELP_STRING([--with-gpu-support-only],
            [Compile and always use the GPU version])],
            [],[with_gpu_support_only=no])
if test x"$with_gpu_support_only" = x"yes" ; then
        m4_foreach_w([elpa_m4_kernel],elpa_m4_all_kernels,[
                      use_[]elpa_m4_kernel[]=no
        ])
        use_real_gpu=yes
        use_complex_gpu=yes
fi


dnl
dnl  ELPA_KERNEL_DEPENDS([kernel],[other kernels])
dnl
dnl   Switch on each of the other kernels if the given kernel is selected
dnl
AC_DEFUN([ELPA_KERNEL_DEPENDS],[
          if test x"$use_$1" = x"yes"; then
                  m4_foreach_w([elpa_m4_requiredkernel],[$2],[
                               if test x"$use_[]elpa_m4_requiredkernel[]" = x"no" ; then
                                       echo "Enabling elpa_m4_requiredkernel kernel, is a prerequisite for $1"
                               fi
                               use_[]elpa_m4_requiredkernel[]=yes
                  ])
          fi
])
m4_foreach_w([elpa_m4_arch],[sse avx avx2 avx512],[
        ELPA_KERNEL_DEPENDS([real_]elpa_m4_arch[_block6], [real_]elpa_m4_arch[_block4 real_]elpa_m4_arch[_block2])
        ELPA_KERNEL_DEPENDS([real_]elpa_m4_arch[_block4], [real_]elpa_m4_arch[_block2])
        ELPA_KERNEL_DEPENDS([complex_]elpa_m4_arch[_block2], [complex_]elpa_m4_arch[_block1])
])

m4_foreach_w([elpa_m4_type],elpa_m4_kernel_types,[
        need_[]elpa_m4_type=no
        need_[]elpa_m4_type[]_kernels=""
        m4_foreach_w([elpa_m4_kernel],m4_expand([elpa_m4_]elpa_m4_type[_kernels]),[
                if test x"$use_[]elpa_m4_kernel" = x"yes" ; then
                        need_[]elpa_m4_type=yes
                        need_[]elpa_m4_type[]_kernels="$need_[]elpa_m4_type[]_kernels elpa_m4_kernel"
                fi
        ])
])
m4_foreach_w([elpa_m4_type],elpa_m4_kernel_types,[
        if test x"$need_[]elpa_m4_type[]" = x"yes" ; then
               echo "Using m4_toupper(elpa_m4_type) for kernels$need_[]elpa_m4_type[]_kernels"
        fi
])
m4_foreach_w([elpa_m4_type],elpa_m4_kernel_types,[
        if test x"$need_[]elpa_m4_type[]" != x"yes" ; then
               echo "Not using m4_toupper(elpa_m4_type) as no selected kernel needs it"
        fi
])


dnl the list of kernels is now assembled
dnl choosing a default kernel

m4_foreach_w([elpa_m4_kind],[real complex],[
        m4_foreach_w([elpa_m4_kernel],
                     m4_foreach_w([elpa_m4_cand_kernel],
                                  elpa_m4_avx512_kernels elpa_m4_avx2_kernels elpa_m4_avx_kernels elpa_m4_sse_kernels elpa_m4_sse_assembly_kernels elpa_m4_generic_kernels,
                                  [m4_bmatch(elpa_m4_cand_kernel,elpa_m4_kind,elpa_m4_cand_kernel)] ),
                     [
                     if test -z "$default_[]elpa_m4_kind[]_kernel"; then
                             if test x"$use_[]elpa_m4_kernel" = x"yes"; then
                                     default_[]elpa_m4_kind[]_kernel="elpa_m4_kernel"
                             fi
                     fi
                     ])
        if test -z "$default_[]elpa_m4_kind[]_kernel"; then
                AC_MSG_ERROR([Internal error, could not determine a default kernel])
        fi
        # find the number of this kernel
        ELPA_2STAGE_[]m4_toupper(elpa_m4_kind)[]_DEFAULT=`grep -i '^ *X(ELPA_2STAGE_'$default_[]elpa_m4_kind[]_kernel'\>' $srcdir/elpa/elpa_constants.h.in | \
                                                                awk '// { gsub(/^[[^,]]*, */, ""); gsub(/,.*/, ""); print;}'`
        AC_SUBST([ELPA_2STAGE_]m4_toupper(elpa_m4_kind)[_DEFAULT])
])

AC_LANG_PUSH([C])

if test x"${need_sse}" = x"yes"; then
  AC_MSG_CHECKING(whether we can compile SSE3 with gcc intrinsics in C)
  AC_COMPILE_IFELSE([AC_LANG_SOURCE([
#include <x86intrin.h>
int main(int argc, char **argv) {
  double* q;
  __m128d h1 = _mm_loaddup_pd(q);
  return 0;
}
    ])],
    [can_compile_sse=yes],
    [can_compile_sse=no]
  )
  AC_MSG_RESULT([${can_compile_sse}])
  if test x"$can_compile_sse" != x"yes"; then
    AC_MSG_ERROR([Could not compile test program, try with --disable-sse, or adjust the C compiler or CFLAGS])
  fi
  AC_DEFINE([HAVE_SSE_INTRINSICS],[1],[gcc intrinsics SSE is supported on this CPU])
fi


if test x"${need_sse_assembly}" = x"yes"; then
  AC_MSG_CHECKING(whether double-precision SSE assembly kernels can be compiled)

  $CC $CFLAGS -c $srcdir/src/elpa2/kernels/asm_x86_64_double_precision.s -o conftest.o 2>&5

  if test "$?" == 0; then
    can_compile_sse_asm_double=yes
  else
    can_compile_sse_asm_double=no
  fi
  rm -f ./conftest.o
  AC_MSG_RESULT([${can_compile_sse_asm_double}])
  if test x"$can_compile_sse_asm_double" != x"yes"; then
    AC_MSG_ERROR([Could not compile test program, try with --disable-sse-assembly, or adjust the C compiler or CFLAGS])
  fi

  if test x"${want_single_precision}" = x"yes" ; then
    AC_MSG_CHECKING(whether single-precision SSE assembly kernels can be compiled)

    $CC $CFLAGS -c $srcdir/src/elpa2/kernels/asm_x86_64_single_precision.s -o conftest.o 2>&5

    if test "$?" == 0; then
      can_compile_sse_asm_single=yes
    else
      can_compile_sse_asm_single=no
    fi
    rm -f ./conftest.o
    AC_MSG_RESULT([${can_compile_sse_asm_single}])
    if test x"$can_compile_sse_asm_single" != x"yes"; then
      AC_MSG_ERROR([Could not compile test program, try with --disable-sse-assembly, or adjust the C compiler or CFLAGS])
    fi
  fi
fi



if test x"${need_avx}" = x"yes"; then
  dnl check whether one can compile AVX gcc intrinsics
  AC_MSG_CHECKING([whether we can compile AVX gcc intrinsics in C])
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
  if test x"$can_compile_avx" != x"yes"; then
    AC_MSG_ERROR([Could not compile a test program with AVX, try with --disable-avx, or adjust the C compiler or CFLAGS])
  fi
  AC_DEFINE([HAVE_AVX],[1],[AVX is supported on this CPU])
fi


if test x"${need_avx2}" = x"yes"; then
  AC_MSG_CHECKING([whether we can compile AVX2 gcc intrinsics in C])
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
  if test x"$can_compile_avx2" != x"yes"; then
    AC_MSG_ERROR([Could not compile a test program with AVX2, try with --disable-avx2, or adjust the C compiler or CFLAGS])
  fi
  AC_DEFINE([HAVE_AVX2],[1],[AVX2 is supported on this CPU])
fi


if test x"${need_avx512}" = x"yes"; then
  AC_MSG_CHECKING([whether we can compile AVX512 gcc intrinsics in C])
  AC_COMPILE_IFELSE([AC_LANG_SOURCE([
   #include <x86intrin.h>
   int main(int argc, char **argv){
   double* q;
   __m512d q1 = _mm512_load_pd(q);
   __m512d y1 = _mm512_fmadd_pd(q1, q1, q1);
   return 0;
   }
   ])],
   [can_compile_avx512=yes],
   [can_compile_avx512=no]
  )
  AC_MSG_RESULT([${can_compile_avx512}])
  if test x"$can_compile_avx512" != x"yes"; then
    AC_MSG_ERROR([Could not compile a test program with AVX512, adjust the C compiler or CFLAGS])
  fi
  AC_DEFINE([HAVE_AVX512],[1],[AVX512 is supported on this CPU])
fi
AC_LANG_POP([C])


AC_LANG_PUSH([Fortran])
if test x"${need_bgp}" = x"yes"; then
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
  if test x"$can_compile_bgp" != x"yes"; then
    AC_MSG_ERROR([Could not compile a test program with BGP intrinsics, adjust the FC compiler or FCFLAGS])
  fi
fi


if test x"${need_bgq}" = x"yes"; then
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
  if test x"$can_compile_bgq" != x"yes"; then
    AC_MSG_ERROR([Could not compile a test program with BGQ intrinsics, adjust the FC compiler or FCFLAGS])
  fi
fi
AC_LANG_POP([Fortran])

if test x"${need_gpu}" = x"yes" ; then
  AC_LANG_PUSH([C])
  CUDA_CFLAGS="$CUDA_CFLAGS -arch $cuda_compute_capability -O2 -I$CUDA_INSTALL_PATH/include"
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
fi


m4_foreach_w([elpa_m4_kernel],elpa_m4_all_kernels,[
  AM_CONDITIONAL([WITH_]m4_toupper(elpa_m4_kernel)[_KERNEL],[test x"$use_[]elpa_m4_kernel[]" = x"yes"])
  if test x"$use_[]elpa_m4_kernel[]" = x"yes" ; then
    AC_DEFINE([WITH_]m4_toupper(elpa_m4_kernel)[_KERNEL],[1],[Build elpa_m4_kernel kernel])
    kernel_defined=1
  else
    kernel_defined=0
  fi
  [ELPA_2STAGE_]m4_toupper(elpa_m4_kernel)[_COMPILED]=$kernel_defined
  AC_SUBST([ELPA_2STAGE_]m4_toupper(elpa_m4_kernel)[_COMPILED])
])


AM_CONDITIONAL([WITH_GPU_VERSION],[test x"$use_real_gpu" = x"yes" -o x"$use_complex_gpu" = x"yes"])
if test x"$use_real_gpu" = x"yes" -o x"$use_complex_gpu" = x"yes" ; then
  AC_DEFINE([WITH_GPU_VERSION],[1],[enable GPU support])
  AC_DEFINE([WITH_GPU_KERNEL],[1],[GPU kernel should be build])
  ELPA_2STAGE_COMPLEX_GPU_COMPILED=1
  ELPA_2STAGE_REAL_GPU_COMPILED=1
else
  ELPA_2STAGE_COMPLEX_GPU_COMPILED=0
  ELPA_2STAGE_REAL_GPU_COMPILED=0
fi
AC_SUBST([ELPA_2STAGE_COMPLEX_GPU_COMPILED])
AC_SUBST([ELPA_2STAGE_REAL_GPU_COMPILED])

LT_INIT

DX_PDF_FEATURE(OFF)
DX_PS_FEATURE(OFF)
DX_MAN_FEATURE(ON)
DX_HTML_FEATURE(ON)
DX_INIT_DOXYGEN([ELPA], [Doxyfile], [docs])

USE_ASSUMED_SIZE=yes
AC_MSG_CHECKING(whether assumed size Fortran arrays should be used)
AC_ARG_ENABLE([assumed-size],
                AS_HELP_STRING([--disable-assumed-size],
                               [do NOT use assumed-size Fortran arrays]),
                               [if test x"$enableval" = x"yes"; then
                                  USE_ASSUMED_SIZE=yes
                                else
                                  USE_ASSUMED_SIZE=no
                                fi],
              [USE_ASSUMED_SIZE=yes])
AC_MSG_RESULT([${USE_ASSUMED_SIZE}])
AM_CONDITIONAL([WITH_USE_ASSUMED_SIZE],[test x"$USE_ASSUMED_SIZE" = x"yes"])
if test x"${USE_ASSUMED_SIZE}" = x"yes" ; then
  AC_DEFINE([USE_ASSUMED_SIZE],[1],[for performance reasons use assumed size Fortran arrays, even if not debuggable])
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
AC_SUBST([FC_MODINC])
AC_SUBST([FC_MODOUT])
AC_SUBST([OPENMP_CFLAGS])
AC_SUBST([OPENMP_FCFLAGS])
AC_SUBST([OPENMP_LDFLAGS])
AC_SUBST([DOXYGEN_OUTPUT_DIR], [docs])

mkdir -p modules private_modules test_modules


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
  elpa/elpa_constants.h
])

m4_include([m4/ax_fc_check_define.m4])
AC_MSG_CHECKING([if workaround for broken preprocessor is needed])

need_manual_cpp=no
AX_FC_CHECK_DEFINE([__INTEL_COMPILER],[is_intel=yes],[])
AX_FC_CHECK_DEFINE([__PGI],[is_pgi=yes],[])
ACTUAL_FC="$FC"
AC_SUBST([ACTUAL_FC])

if test x"$is_intel" = x"yes" ; then
	need_manual_cpp=yes
fi
if test x"$is_pgi" = x"yes" ; then
	need_manual_cpp=yes
fi

if test x"$need_manual_cpp" = x"yes" ; then
        AC_MSG_RESULT([yes])
        FC="\$(top_srcdir)/manual_cpp $FC"
else
        AC_MSG_RESULT([no])
fi

if test x"$is_pgi" = x"yes" ; then
	AC_DEFINE([PGI_VARIABLE_STRING_BUG], [1], [Work around a PGI bug with variable-length string results])
fi


dnl PGI compiler uses -module to specify module output
dnl directory. This clashes with libtools -module link option
dnl => escape it for libtool with -Xcompiler -module
AC_MSG_CHECKING([whether we have to escape '-module' for libtool])
if test x"$FC_MODOUT" = x'-module ' ; then
        FC_MODOUT="-Xcompiler $FC_MODOUT -Xcompiler \$(ac_empty)"
        FC="\$(top_srcdir)/remove_xcompiler $FC"
        AC_MSG_RESULT([yes])
else
        AC_MSG_RESULT([no])
fi


AC_OUTPUT

echo ""
echo "The following ELPA2 kernels will be build:"
echo ""
m4_foreach_w([elpa_m4_kind],[real complex],[
        m4_foreach_w([elpa_m4_kernel],m4_expand(elpa_m4_[]elpa_m4_kind[]_kernels),[
                if test x"$use_[]elpa_m4_kernel" = x"yes" ; then
                        echo -n "  elpa_m4_kernel"
                        if test "$fixed_]elpa_m4_kind[_kernel" = "]elpa_m4_kernel[" ; then
                                echo -n " (selected as fixed kernel)"
                        fi
                        if test "$default_]elpa_m4_kind[_kernel" = "]elpa_m4_kernel[" ; then
                                echo -n " (default)"
                        fi
                        echo ""
                fi
        ])
])

#echo " "
#echo "***********************************************************************"
#echo "*  This is a the release candidate  ELPA 2017.05.001.rc2              *"
#echo "*  All functionality of the release ELPA 2017.05.001 is already there.*"
#echo "*  This release candidate reflects, except for maybe smaller bug      *"
#echo "*  corrections, the release of ELPA 2017.05.001                       *"
#echo "***********************************************************************"
#echo " "
make -f $srcdir/generated_headers.am generated-headers top_srcdir="$srcdir"
