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
AC_SUBST([ELPA_SO_VERSION], [15:2:0])

# AC_DEFINE_SUBST(NAME, VALUE, DESCRIPTION)
# -----------------------------------------
AC_DEFUN([AC_DEFINE_SUBST], [
AC_DEFINE([$1], [$2], [$3])
AC_SUBST([$1], ['$2'])
])

AX_BUILD_DATE_EPOCH(ELPA_BUILDTIME)

# API Version
AC_DEFINE([EARLIEST_API_VERSION], [20170403], [Earliest supported ELPA API version])
print_earliest_api_version=20170403
AC_DEFINE_SUBST(CURRENT_API_VERSION, 20200417, "Current ELPA API version")
print_current_api_version=20200417
# Autotune Version
AC_DEFINE([EARLIEST_AUTOTUNE_VERSION], [20171201], [Earliest ELPA API version, which supports autotuning])
print_earliest_autotune_version=20171201
AC_DEFINE([CURRENT_AUTOTUNE_VERSION], [20200417], [Current ELPA autotune version])
print_current_autotune_version=20200417
AC_DEFINE([CURRENT_AUTOTUNE_VERSION], [20200417], [Current ELPA autotune version])
AC_DEFINE_SUBST(CURRENT_AUTOTUNE_VERSION, 20200417, "Current ELPA autotune version")
AC_DEFINE_UNQUOTED([ELPA_BUILDTIME], [$ELPA_BUILDTIME], ["Time of build"])
AX_COMPARE_VERSION([$ELPA_BUILDTIME], [gt], [1623715200],[old_elpa_version=yes],[old_elpa_version=no])

AX_CHECK_GNU_MAKE()
if test x$_cv_gnu_make_command = x ; then
        AC_MSG_ERROR([Need GNU Make])
fi

enable_legacy=no
AC_MSG_CHECKING(whether in C interface the error argument should be optional)
AC_ARG_ENABLE([optional-argument-in-C-API],
              AS_HELP_STRING([--enable-optional-argument-in-C-API],
                             [do not build C API with error argument as optional, default no]),
              [
	       if test x"$enableval" = x"yes"; then
	         optional_c_error_argument=yes
	       else
	         optional_c_error_argument=no
	       fi
	       ],
              [optional_c_error_argument=no])
AC_MSG_RESULT([${optional_c_error_argument}])
AM_CONDITIONAL([OPTIONAL_C_ERROR_ARGUMENT],[test x"$optional_c_error_argument" = x"yes"])
if test x"${optional_c_error_argument}" = x"yes"; then
        AC_DEFINE([OPTIONAL_C_ERROR_ARGUMENT], [1], [enable error argument in C-API to be optional])
fi


# gnu-make fortran module dependencies
m4_include([fdep/fortran_dependencies.m4])
FDEP_F90_GNU_MAKE_DEPS


dnl OpenMP
m4_include([m4/ax_elpa_openmp.m4])
AC_MSG_CHECKING(whether --enable-openmp is specified)
AC_ARG_ENABLE([openmp],
              AS_HELP_STRING([--enable-openmp],
                             [use OpenMP threading, default no.]),
              [
	       if test x"$enableval" = x"yes"; then
	         enable_openmp=yes
	       else
	         enable_openmp=no
	       fi
	       ],
              [enable_openmp=no])
AC_MSG_RESULT([${enable_openmp}])
AM_CONDITIONAL([WITH_OPENMP_TRADITIONAL],[test x"$enable_openmp" = x"yes"])
if test x"${enable_openmp}" = x"yes"; then
        AC_DEFINE([WITH_OPENMP_TRADITIONAL], [1], [use OpenMP threading])
fi


dnl mpi
AC_ARG_WITH(mpi, [AS_HELP_STRING([--with-mpi=[[yes|no]]], [compile with MPI. Default: yes])],,[with_mpi=yes])
AM_CONDITIONAL([WITH_MPI],[test x"$with_mpi" = x"yes"])
if test x"${with_mpi}" = x"yes"; then
  AC_DEFINE([WITH_MPI], [1], [use MPI])
fi

dnl CPP
AC_LANG_PUSH([C++])
AC_PROG_CXX
AC_LANG_POP([C++])

dnl C
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

c11_standard=no
AX_CHECK_COMPILE_FLAG([-std=gnu11], [
  c11_standard=yes
], [
  echo "C compiler cannot compile -std=gnu11 code"
  echo "testing -std=c11.."
])
if test x"$c11_standard" = x"yes"; then
  CFLAGS+=" -std=gnu11"
fi

if test x"$c11_standard" = x"no"; then
  AX_CHECK_COMPILE_FLAG([-std=c11], [
    c11_standard=yes
  ], [
    echo "C compiler cannot compile -std=c11 code"
    echo "testing -c11.."
  ])
  if test x"$c11_standard" = x"yes"; then
    CFLAGS+=" -std=c11"
  fi
fi

if test x"$c11_standard" = x"no"; then
  AX_CHECK_COMPILE_FLAG([-c11], [
    c11_standard=yes
  ], [
    echo "C compiler cannot compile -c11 code"
    echo "C compiler cannot compile C11 code"
    exit -1
  ])
  if test x"$c11_standard" = x"yes"; then
    CFLAGS+=" -c11"
  fi
fi

AX_EXT

dnl heterogenous-cluster-support
AC_MSG_CHECKING(whether heterogenous-cluster-support should be enabled)
AC_ARG_ENABLE([heterogenous-cluster-support],
              AS_HELP_STRING([--heterogenous-cluster-support],
                             [allow ELPA to automatically switch to a kernel supported on all CPUs of a cluster (only works for Intel CPUs at the moment), default no. Activate only if necessary has a performance penalty! This feature is exerpimental!]),
              [
	       if test x"$enableval" = x"yes"; then
	         enable_heterogenous_cluster_support=yes
	       else
	         enable_heterogenous_cluster_support=no
	       fi
	       ],
              [enable_heterogenous_cluster_support="no"])
AC_MSG_RESULT([$enable_heterogenous_cluster_support])
if test x"${enable_heterogenous_cluster_support}" = x"yes"; then
  AC_DEFINE([HAVE_HETEROGENOUS_CLUSTER_SUPPORT], [1], [automatically support clusters with different Intel CPUs])
fi
AM_CONDITIONAL([HAVE_HETEROGENOUS_CLUSTER_SUPPORT],[test x"$enable_heterogenous_cluster_support" = x"yes"])

dnl 64bit integer support for BLACS/LAPACK/SCALAPACK support
dnl first long int
AC_CHECK_SIZEOF([long int])
size_of_long_int="${ac_cv_sizeof_long_int}"

dnl then 64bit blas
AC_MSG_CHECKING(whether 64bit integers should be used for math libraries (BLAS/LAPACK/SCALAPACK))
AC_ARG_ENABLE([64bit-integer-math-support],
              AS_HELP_STRING([--64bit-integer-math-support],
                             [allows to link against the 64bit integer versions of the math libraries BLAS, LAPACK, and SCALAPACK]),
              [
	       if test x"$enableval" = x"yes"; then
	         enable_64bit_integer_math_support=yes
	       else
	         enable_64bit_integer_math_support=no
	       fi
	       ],
              [enable_64bit_integer_math_support="no"])
AC_MSG_RESULT([$enable_64bit_integer_math_support])
if test x"${enable_64bit_integer_math_support}" = x"yes"; then
  dnl at least INTEL MPI does _NOT_ support 64BIT integer mode  for C thus disable C tests in this Case
  if test x"${enable_c_tests}" = x"yes"; then
     AC_MSG_ERROR([You cannot both define 64bit integer support and C tests. Reconfigure!])
  fi
  dnl check whether long int is the correct data-type in C
  if test x"${size_of_long_int}" = x"8"; then
    echo "Found C data-type \"long int\" with 8 bytes"
  else
    AC_MSG_ERROR([The C data-type "long int" is only ${size_of_long_int} bytes; Needed is 8 bytes])
  fi

  AC_DEFINE([HAVE_64BIT_INTEGER_MATH_SUPPORT], [1], [allow to link against the 64bit integer versions of math libraries])
fi
AM_CONDITIONAL([HAVE_64BIT_INTEGER_MATH_SUPPORT],[test x"$enable_64bit_integer_math_support" = x"yes"])

dnl then 64bit blas
AC_MSG_CHECKING(whether 64bit integers should be used for the MPI library)
AC_ARG_ENABLE([64bit-integer-mpi-support],
              AS_HELP_STRING([--64bit-integer-mpi-support],
                             [allows to link against the 64bit integer versions of the MPI library]),
              [
	       if test x"$enableval" = x"yes"; then
	         enable_64bit_integer_mpi_support=yes
	       else
	         enable_64bit_integer_mpi_support=no
	       fi
	       ],
              [enable_64bit_integer_mpi_support="no"])
AC_MSG_RESULT([$enable_64bit_integer_mpi_support])
if test x"${enable_64bit_integer_mpi_support}" = x"yes"; then
  AC_DEFINE([HAVE_64BIT_INTEGER_MPI_SUPPORT], [1], [allow to link against the 64bit integer versions of the MPI library])
fi
AM_CONDITIONAL([HAVE_64BIT_INTEGER_MPI_SUPPORT],[test x"$enable_64bit_integer_mpi_support" = x"yes"])

AC_MSG_CHECKING(whether C compiler can use _Generic )
AC_COMPILE_IFELSE([AC_LANG_SOURCE([
int main(int argc, char **argv) {
#define elpa_set(e, name, value, error) _Generic((value), \
                int: \
                  elpa_set_integer, \
                \
                double: \
                  elpa_set_double \
        )(e, name, value, error)

  return 0;
}
    ])],
    [can_compile_generic=yes],
    [can_compile_generic=no]
  )
AC_MSG_RESULT([${can_compile_generic}])
if test x"$can_compile_generic" != x"yes"; then
  AC_MSG_ERROR([C compiler cannot handle _Generic statement! Upgrade or change C compiler])
fi
AC_DEFINE([HAVE_VSX_SSE],[1],[Altivec VSX intrinsics are supported on this CPU])

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

AC_MSG_CHECKING(whether Fortran compiler infers interfaces )
AC_COMPILE_IFELSE([AC_LANG_SOURCE([
program test
implicit none
integer :: a
real*8  :: b

  call foo(a)
  call foo(b)

end program
    ])],
    [FC_does_infer_interfaces=no],
    [FC_does_infer_interfaces=yes]
  )
AC_MSG_RESULT([${FC_does_infer_interfaces}])

if test x"${FC_does_infer_interfaces}" = x"yes"; then
  # two problems
  # 1. some MPIs (at least Intel) do not provide all
  #    interfaces, which will fail
  #    in case of MPI and not all interfaces -> switch of
  #    argument checking or fail
  # 2. no MPI case: switch of PACK_REAL_TO_COMPLEX
  if test x"${with_mpi}" = x"yes"; then
    AC_MSG_CHECKING(whether MPI module defines all interfaces )
    # check whether MPI module defines all interfaces; not the case for Intel MPI!
    AC_COMPILE_IFELSE([AC_LANG_SOURCE([
    program test_mpi_interfaces
      use mpi
      implicit none
      integer :: rank
      integer :: buf(10)
      integer :: ierr
      real*8  :: a(2)
      complex*16 :: b(2)

      call MPI_Init(ierr)
      call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

      if (rank == 0) then
         buf(:) = 42;
      end if

      
      !! this is OK
      !call MPI_Bcast(buf, 10, MPI_INT, 0, MPI_COMM_WORLD, ierr)

      !! Oops, wrong order here:
      !call MPI_Bcast(10, buf, MPI_INT, 0, MPI_COMM_WORLD, ierr)

      ! if the correct interfaces exists in the MPI module
      ! this must work. If not and the compiler infers interfaces
      ! this will fail
      call MPI_Bcast(a, 2, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
      call MPI_Bcast(b, 2, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD, ierr)
      call MPI_Finalize(ierr)
    end program
      ])],
      [MPImodule_has_interfaces=yes],
      [MPImodule_has_interfaces=no]
    )
    AC_MSG_RESULT([${MPImodule_has_interfaces}])
    if test x"${MPImodule_has_interfaces}" = x"no"; then
      AC_MSG_CHECKING(whether we can cure missing interfaces by allowing argument mismatch )
      #check whether we can cure this by disabling argument checking
      FCFLAGS_SAVE2="$FCFLAGS"
      FCFLAGS="$FCFLAGS -fallow-argument-mismatch"
      # test again
      AC_COMPILE_IFELSE([AC_LANG_SOURCE([
      program test
      implicit none
      integer :: a
      real*8  :: b
      
        call foo(a)
        call foo(b)
      
      end program

      ])],
      [FC_infer_interfaces_cured=yes],
      [FC_infer_interfaces_cured=no]
      )
      AC_MSG_RESULT([${FC_infer_interfaces_cured}])
      if test x"${FC_infer_interfaces_cured}" = x"no"; then
        FCFLAGS = "$FCFLAGS_SAVE2"
        AC_MSG_ERROR([Fortran compiler infers interfaces; but MPI module does not supply all of them])
      fi
    fi
  #else
  #  # no MPI case switch of PACK_REAL_TO_COMPLEX
  #  # i.e. do _not_ define it
  fi
else
  AC_DEFINE([PACK_REAL_TO_COMPLEX], [1], [In some kernels pack real to complex])
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
  AC_CHECK_PROGS([MPI_BINARY], [mpiexec.hydra mpiexec mpirun poe runjob srun aprun], [no])
  if test x"$MPI_BINARY" = x"no"; then
    AC_MSG_ERROR([Could not find either of the MPI binaries: mpiexec.hydra, mpiexec, mpirun, poe, runjob, srun, aprun])
  fi
fi

dnl build with the possibilty to redirect stdout and stderr
dnl per MPI task in a file
AC_MSG_CHECKING(whether stdout/stderr file redirect should be enabled)
AC_ARG_ENABLE([redirect],
              [AS_HELP_STRING([--enable-redirect],
                              [for test programs, allow redirection of stdout/stderr per MPI taks in a file (useful for timing), default no.])],
              [
               if test x"$enableval" = x"yes"; then
                 enable_redirect=yes
               else
                 enable_redirect=no
               fi
	       ],
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
              [
               if test x"$enableval" = x"yes"; then
                 enable_timings=yes
               else
                 enable_timings=no
               fi
	       ],
              [enable_timings=yes])

if test x"${enable_timings}" = x"yes"; then
  AC_DEFINE([HAVE_DETAILED_TIMINGS], [1], [Enable more timing])
fi
AM_CONDITIONAL([HAVE_DETAILED_TIMINGS], [test x"$enable_timings" = x"yes"])

dnl PAPI for ftimings
AC_LANG_PUSH([C])
AC_ARG_WITH([papi],
            [AS_HELP_STRING([--with-papi],
                            [Use PAPI to also measure flop count in the detailed timing (--enable-timing), disabled by default])],
            [
               if test x"$withval" = x"yes"; then
                 with_papi=yes
               else
                 with_papi=no
               fi
	     ],
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
AC_LANG_POP([C])

dnl Likwid
AC_LANG_PUSH([Fortran])
AC_ARG_WITH([likwid],
            [AS_HELP_STRING([--with-likwid=[[yes|no(default)|PATH]]],
                            [Use Likwid to measure performance in some parts of the library])],
            [with_likwid="$withval"],
            [with_likwid="no"])

if test x"$with_likwid" != x"no" ; then
  if test -d $with_likwid/lib ; then
    LDFLAGS="-L$with_likwid/lib $LDFLAGS"
  fi
  if test -d $with_likwid/lib64 ; then
    LDFLAGS="-L$with_likwid/lib64 $LDFLAGS"
  fi
  if test -d $with_likwid/include ; then
    FCFLAGS="-I$with_likwid/include $FCFLAGS"
  fi
  AC_SEARCH_LIBS([likwid_markerInit], [likwid], [liblikwid_found="yes"], [liblikwid_found="no"])
  if test x"$liblikwid_found" = x"no" ; then
    AC_MSG_ERROR([Could not find a usable likwid library, please adjust LDFLAGS])
  fi
  AC_MSG_CHECKING([whether we can use the likwid module in a Fortran program])
  AC_COMPILE_IFELSE([
       program foo
       use likwid

       implicit none

       call likwid_markerInit()
       call likwid_markerThreadInit()

       call likwid_markerStartRegion("foobar")
       call likwid_markerStopRegion("foobar")

       call likwid_markerClose()
       end
    ],
    [AC_MSG_RESULT([yes])],
    [AC_MSG_RESULT([no])
     AC_MSG_ERROR([Could not compile a Fortran program using the likwid module, adjust FCFLAGS])])
  AC_DEFINE([HAVE_LIKWID], [1], [Use likwid])
fi
AC_LANG_POP([Fortran])

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
  AC_SEARCH_LIBS([dgemm],[flexiblas openblas satlas blas],[have_blas=yes],[have_blas=no])
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
              [
               if test x"$enableval" = x"yes"; then
                 enable_band_to_full_blocking=yes
               else
                 enable_band_to_full_blocking=no
               fi
	       ],
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
user_sets_nvidia_gpu_compute_capability="no"
AC_MSG_CHECKING(whether a Nvidia GPU compute capability is specified)
AC_ARG_WITH([NVIDIA-GPU-compute-capability],
            [AS_HELP_STRING([--with-NVIDIA-GPU-compute-capability=VALUE],
                            [use compute capability VALUE for NVIDIA GPU version, default: "sm_35"])],
            [user_sets_nvidia_gpu_compute_capability="yes"],[cuda_compute_capability="sm_35"])
AC_MSG_RESULT([${user_sets_nvidia_gpu_compute_capability}])


dnl sanity check whether compute capability setting by user is reasonable
if test x"${user_sets_nvidia_gpu_compute_capability}" = x"yes" ; then
  dnl the user must set a value which starts with "sm_"
  value=$(echo $withval | cut -c1-3)
  if test x"${value}" = x"sm_" ; then
    cuda_compute_capability=$withval
  else
    AC_MSG_ERROR([Unknown Nvidia GPU compute capability set: ${withval}])
  fi
fi


dnl Test possibility of 'use mpi', if requested
if test x"${with_mpi}" = x"yes" ; then
  AC_ARG_ENABLE([mpi-module],
                AS_HELP_STRING([--disable-mpi-module],
                               [do not use the Fortran MPI module, get interfaces by 'include "mpif.h']),
                [
                 if test x"$enableval" = x"yes"; then
                   enable_mpi_module=yes
                 else
                   enable_mpi_module=no
                 fi
		 ],
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
        real_generic_simple_block4
        real_generic_simple_block6
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

m4_define(elpa_m4_sparc64_kernels, [
        real_sparc64_block2
        real_sparc64_block4
        real_sparc64_block6
])

m4_define(elpa_m4_neon_arch64_kernels, [
        real_neon_arch64_block2
        real_neon_arch64_block4
        real_neon_arch64_block6
        complex_neon_arch64_block1
        complex_neon_arch64_block2
])

m4_define(elpa_m4_vsx_kernels, [
        real_vsx_block2
        real_vsx_block4
        real_vsx_block6
])

m4_define(elpa_m4_sve128_kernels, [
        real_sve128_block2
        real_sve128_block4
        real_sve128_block6
        complex_sve128_block1
        complex_sve128_block2
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

m4_define(elpa_m4_sve256_kernels, [
        real_sve256_block2
        real_sve256_block4
        real_sve256_block6
        complex_sve256_block1
        complex_sve256_block2
])

m4_define(elpa_m4_avx512_kernels, [
        real_avx512_block2
        real_avx512_block4
        real_avx512_block6
        complex_avx512_block1
        complex_avx512_block2
])

m4_define(elpa_m4_sve512_kernels, [
        real_sve512_block2
        real_sve512_block4
        real_sve512_block6
        complex_sve512_block1
        complex_sve512_block2
])

m4_define(elpa_m4_bgp_kernels, [
        real_bgp
        complex_bgp
])

m4_define(elpa_m4_bgq_kernels, [
        real_bgq
        complex_bgq
])

m4_define(elpa_m4_nvidia_gpu_kernels, [
        real_nvidia_gpu
        complex_nvidia_gpu
])

m4_define(elpa_m4_amd_gpu_kernels, [
        real_amd_gpu
        complex_amd_gpu
])

m4_define(elpa_m4_intel_gpu_kernels, [
        real_intel_gpu
        complex_intel_gpu
])

m4_define(elpa_m4_kernel_types, [generic sparc64 neon_arch64 vsx sse sse_assembly sve128 avx avx2 sve256 avx512 sve512 bgp bgq nvidia_gpu amd_gpu intel_gpu])

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
ELPA_SELECT_KERNELS([sparc64],[disable])
ELPA_SELECT_KERNELS([neon_arch64],[disable])
ELPA_SELECT_KERNELS([vsx],[disable])
ELPA_SELECT_KERNELS([sse],[enable])
ELPA_SELECT_KERNELS([sse_assembly],[enable])
ELPA_SELECT_KERNELS([avx],[enable])
ELPA_SELECT_KERNELS([avx2],[enable])
ELPA_SELECT_KERNELS([avx512],[enable])
ELPA_SELECT_KERNELS([sve128],[disable])
ELPA_SELECT_KERNELS([sve256],[disable])
ELPA_SELECT_KERNELS([sve512],[disable])
ELPA_SELECT_KERNELS([nvidia_gpu],[disable])
ELPA_SELECT_KERNELS([amd_gpu],[disable])
ELPA_SELECT_KERNELS([intel_gpu],[disable])
ELPA_SELECT_KERNELS([bgp],[disable])
ELPA_SELECT_KERNELS([bgq],[disable])

m4_foreach_w([elpa_m4_kind],[real complex],[
        default_[]elpa_m4_kind[]_kernel=""
])

if test x"${enable_bgp}" = x"yes" -o x"$enable_bgq" = x"yes"; then
        m4_foreach_w([elpa_m4_kernel], elpa_m4_sparc64_kernels elpa_m4_neon_arch64_kernels elpa_m4_vsx_kernels elpa_m4_sse_kernels elpa_m4_avx_kernels elpa_m4_avx2_kernels elpa_m4_avx512_kernels elpa_m4_sve128_kernels elpa_m4_sve256_kernels elpa_m4_sve512_kernels, [
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

AC_ARG_WITH(nvidia-gpu-support-only, [AS_HELP_STRING([--with-nvidia-gpu-support-only],
            [Compile and always use the NVIDIA GPU version])],
            [],[with_nvidia_gpu_support_only=no])
if test x"$with_nvidia_gpu_support_only" = x"yes" ; then
        m4_foreach_w([elpa_m4_kernel],elpa_m4_all_kernels,[
                      use_[]elpa_m4_kernel[]=no
        ])
        use_real_nvidia_gpu=yes
        use_complex_nvidia_gpu=yes
fi

AC_ARG_WITH(amd-gpu-support-only, [AS_HELP_STRING([--with-amd-gpu-support-only],
            [Compile and always use the AMD GPU version])],
            [],[with_amd_gpu_support_only=no])
if test x"$with_amd_gpu_support_only" = x"yes" ; then
        m4_foreach_w([elpa_m4_kernel],elpa_m4_all_kernels,[
                      use_[]elpa_m4_kernel[]=no
        ])
        use_real_amd_gpu=yes
        use_complex_amd_gpu=yes
fi

AC_ARG_WITH(intel-gpu-support-only, [AS_HELP_STRING([--with-intel-gpu-support-only],
            [Compile and always use the INTEL GPU version])],
            [],[with_intel_gpu_support_only=no])
if test x"$with_intel_gpu_support_only" = x"yes" ; then
        m4_foreach_w([elpa_m4_kernel],elpa_m4_all_kernels,[
                      use_[]elpa_m4_kernel[]=no
        ])
        use_real_intel_gpu=yes
        use_complex_intel_gpu=yes
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
m4_foreach_w([elpa_m4_arch],[sparc64 neon_arch64 vsx sse avx avx2 avx512 sve128 sve256 sve512],[
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
        AC_ARG_WITH([default-]elpa_m4_kind[-kernel], m4_expand([AS_HELP_STRING([--with-default-]elpa_m4_kind[-kernel]=KERNEL,
                [set a specific ]elpa_m4_kind[ kernel as default kernel. Available kernels are:]
                 m4_foreach_w([elpa_m4_kernel],m4_expand(elpa_m4_[]elpa_m4_kind[]_kernels),[m4_bpatsubst(elpa_m4_kernel,elpa_m4_kind[]_,[]) ]))]),
                 [default_]elpa_m4_kind[_kernel="]elpa_m4_kind[_$withval"],[default_]elpa_m4_kind[_kernel=""])
        #if test -n "$default_[]elpa_m4_kind[]_kernel" ; then
        #        found="no"
        #        m4_foreach_w([elpa_m4_otherkernel],m4_expand(elpa_m4_[]elpa_m4_kind[]_kernels),[
        #                if test "$default_]elpa_m4_kind[_kernel" = "]elpa_m4_otherkernel[" ; then
        #                        use_[]elpa_m4_otherkernel[]=yes
        #                        found="yes"
        #                else
        #                        use_[]elpa_m4_otherkernel[]=no
        #                fi
        #        ])
        #        if test x"$found" = x"no" ; then
        #                AC_MSG_ERROR([Invalid kernel "$default_]elpa_m4_kind[_kernel" specified for --with-default-]elpa_m4_kind[-kernel])
        #        fi
        #        AC_DEFINE([WITH_DEFAULT_]m4_toupper(elpa_m4_kind)[_KERNEL],[1],[use specific ]elpa_m4_kind[ default kernel (set at compile time)])
        #fi
])



m4_foreach_w([elpa_m4_kind],[real complex],[
        m4_foreach_w([elpa_m4_kernel],
                     m4_foreach_w([elpa_m4_cand_kernel],
                                  elpa_m4_avx512_kernels elpa_m4_avx2_kernels elpa_m4_avx_kernels elpa_m4_sse_kernels elpa_m4_sse_assembly_kernels elpa_m4_sve128_kernels elpa_m4_sve256_kernels elpa_m4_sve512_kernels elpa_m4_sparc64_kernels elpa_m4_neon_arch64_kernels elpa_m4_vsx_kernels elpa_m4_generic_kernels elpa_m4_nvidia_gpu_kernels elpa_m4_amd_gpu_kernels elpa_m4_intel_gpu_kernels,
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
                                                                perl -pe 's/^[[^,]]*, *//; s/,.*//;'`
        AC_SUBST([ELPA_2STAGE_]m4_toupper(elpa_m4_kind)[_DEFAULT])
])

dnl #include <fjmfunc.h>
dnl #include <emmintrin.h>
dnl int main(int argc, char **argv) {
dnl   __m128d q;
dnl   __m128d h1 = _fjsp_neg_v2r8(q);
dnl   return 0;
dnl }
AC_LANG_PUSH([C])
dnl check whether one can link against Fortran programs from C
AC_MSG_CHECKING(whether we need _ in C programs to link against a Fortran library)
AC_LINK_IFELSE([AC_LANG_SOURCE([
int main(int argc, char **argv) {

  int m, n, k, lda, ldb, ldc;
  double alpha, beta;
  double *a, *b, *c;
  dgemm_("N", "N", &m, &n, &k, &alpha, a, lda, b, &ldb, &beta, c, &ldc);
}
 ])],
   [can_link_with_=yes],
   [can_link_with_=no]
 )
AC_MSG_RESULT([${can_link_with_}])
if test x"$can_link_with_" = x"yes"; then
  AC_DEFINE([NEED_UNDERSCORE_TO_LINK_AGAINST_FORTRAN],[1],[need to append an underscore])
fi
  
AC_MSG_CHECKING(whether we can link C programs against Fortran without _ )
AC_LINK_IFELSE([AC_LANG_SOURCE([
int main(int argc, char **argv) {

  int m, n, k, lda, ldb, ldc;
  double alpha, beta;
  double *a, *b, *c;
  dgemm("N", "N", &m, &n, &k, &alpha, a, lda, b, &ldb, &beta, c, &ldc);
}
 ])],
   [can_link_without_=yes],
   [can_link_without_=no]
 )
AC_MSG_RESULT([${can_link_without_}])
if test x"$can_link_without_" = x"yes"; then
  AC_DEFINE([NEED_NO_UNDERSCORE_TO_LINK_AGAINST_FORTRAN],[1],[need not to append an underscore])
fi

if test x"${need_vsx}" = x"yes"; then
  AC_MSG_CHECKING(whether we can compile Altivec VSX with intrinsics in C)
  AC_COMPILE_IFELSE([AC_LANG_SOURCE([
#include <altivec.h>
int main(int argc, char **argv) {
  __vector double a, b, c;
  c = vec_add(a,b);
  return 0;
}
    ])],
    [can_compile_vsx=yes],
    [can_compile_vsx=no]
  )
  AC_MSG_RESULT([${can_compile_vsx}])
  if test x"$can_compile_vsx" != x"yes"; then
    AC_MSG_ERROR([Could not compile test program, try with --disable-vsx, or adjust the C compiler or CFLAGS])
  fi
  AC_DEFINE([HAVE_VSX_SSE],[1],[Altivec VSX intrinsics are supported on this CPU])
fi

if test x"${need_sparc64}" = x"yes"; then
  AC_MSG_CHECKING(whether we can compile SPARC64 with intrinsics in C)
  AC_COMPILE_IFELSE([AC_LANG_SOURCE([
#include <fjmfunc.h>
#include <emmintrin.h>
int main(int argc, char **argv) {
  __m128d  tau1;
  __m128d h1 = _fjsp_neg_v2r8(tau1);
  return 0;
}
    ])],
    [can_compile_sparc64=yes],
    [can_compile_sparc64=no]
  )
  AC_MSG_RESULT([${can_compile_sparc64}])
  if test x"$can_compile_sparc64" != x"yes"; then
    AC_MSG_ERROR([Could not compile test program, try with --disable-sparc64, or adjust the C compiler or CFLAGS])
  fi
  AC_DEFINE([HAVE_SPARC64_SSE],[1],[SPARC64 intrinsics are supported on this CPU])
fi

if test x"${need_neon_arch64}" = x"yes"; then
  AC_MSG_CHECKING(whether we can compile NEON ARCH64 with intrinsics in C)
  AC_COMPILE_IFELSE([AC_LANG_SOURCE([
#include <arm_neon.h>
int main(int argc, char **argv) {
  __Float64x2_t  x1, x2, x3, x4;
  x4 = vfmaq_f64(x1, x2, x3);
  return 0;
}
    ])],
    [can_compile_neon_arch64=yes],
    [can_compile_neon_arch64=no]
  )
  AC_MSG_RESULT([${can_compile_neon_arch64}])
  if test x"$can_compile_neon_arch64" != x"yes"; then
    AC_MSG_ERROR([Could not compile test program, try with --disable-neon_arch64, or adjust the C compiler or CFLAGS])
  fi
  AC_DEFINE([HAVE_NEON_ARCH64_SSE],[1],[NEON_ARCH64 intrinsics are supported on this CPU])
fi


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
    AC_MSG_ERROR([Could not compile test program, try with --disable-sse-assembly, or adjust the C compiler or CFLAGS. Possibly (some of) the flags " $SIMD_FLAGS " solve this issue])
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
      AC_MSG_ERROR([Could not compile test program, try with --disable-sse-assembly, or adjust the C compiler or CFLAGS. Possibly (some of) the flags " $SIMD_FLAGS " solve this issue])
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
    AC_MSG_ERROR([Could not compile a test program with AVX, try with --disable-avx, or adjust the C compiler or CFLAGS. Possibly (some of) the flags " $SIMD_FLAGS " solve this issue])
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
    AC_MSG_ERROR([Could not compile a test program with AVX2, try with --disable-avx2, or adjust the C compiler or CFLAGS.  Possibly (some of) the flags " $SIMD_FLAGS " solve this issue])
  fi
  AC_DEFINE([HAVE_AVX2],[1],[AVX2 is supported on this CPU])
fi

if test x"${need_sve128}" = x"yes"; then
  AC_MSG_CHECKING([whether we can compile SVE128 gcc intrinsics in C])
  AC_COMPILE_IFELSE([AC_LANG_SOURCE([
   #include <arm_sve.h>
   int main(int argc, char **argv){
   double *q;
   svfloat64_t q1 = svld1_f64(svptrue_b64(), q);
   svfloat64_t y1 = svmad_f64_z(svptrue_b64(), q1, q1, q1);
   return 0;
   }
   ])],
   [can_compile_sve128=yes],
   [can_compile_sve128=no]
  )
  AC_MSG_RESULT([${can_compile_sve128}])
  if test x"$can_compile_sve128" != x"yes"; then
    AC_MSG_ERROR([Could not compile a test program with SVE128, adjust the C compiler or CFLAGS. Possibly (some of) the flags " $SIMD_FLAGS " solve this issue])
  fi
  AC_DEFINE([HAVE_SVE128],[1],[SVE128 is supported on this CPU])
fi

if test x"${need_sve256}" = x"yes"; then
  AC_MSG_CHECKING([whether we can compile SVE256 gcc intrinsics in C])
  AC_COMPILE_IFELSE([AC_LANG_SOURCE([
   #include <arm_sve.h>
   int main(int argc, char **argv){
   double *q;
   svfloat64_t q1 = svld1_f64(svptrue_b64(), q);
   svfloat64_t y1 = svmad_f64_z(svptrue_b64(), q1, q1, q1);
   return 0;
   }
   ])],
   [can_compile_sve256=yes],
   [can_compile_sve256=no]
  )
  AC_MSG_RESULT([${can_compile_sve256}])
  if test x"$can_compile_sve256" != x"yes"; then
    AC_MSG_ERROR([Could not compile a test program with SVE256, adjust the C compiler or CFLAGS. Possibly (some of) the flags " $SIMD_FLAGS " solve this issue])
  fi
  AC_DEFINE([HAVE_SVE256],[1],[SVE256 is supported on this CPU])
fi

if test x"${need_sve512}" = x"yes"; then
  AC_MSG_CHECKING([whether we can compile SVE512 gcc intrinsics in C])
  AC_COMPILE_IFELSE([AC_LANG_SOURCE([
   #include <arm_sve.h>
   int main(int argc, char **argv){
   double *q;
   svfloat64_t q1 = svld1_f64(svptrue_b64(), q);
   svfloat64_t y1 = svmad_f64_z(svptrue_b64(), q1, q1, q1);
   return 0;
   }
   ])],
   [can_compile_sve512=yes],
   [can_compile_sve512=no]
  )
  AC_MSG_RESULT([${can_compile_sve512}])
  if test x"$can_compile_sve512" != x"yes"; then
    AC_MSG_ERROR([Could not compile a test program with SVE512, adjust the C compiler or CFLAGS. Possibly (some of) the flags " $SIMD_FLAGS " solve this issue])
  fi
  AC_DEFINE([HAVE_SVE512],[1],[SVE512 is supported on this CPU])
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
    AC_MSG_ERROR([Could not compile a test program with AVX512, adjust the C compiler or CFLAGS. Possibly (some of) the flags " $SIMD_FLAGS " solve this issue])
  fi
  AC_DEFINE([HAVE_AVX512],[1],[AVX512 is supported on this CPU])

  if test x"$can_compile_avx512" = x"yes"; then
    AC_MSG_CHECKING([whether we compile for Xeon])
    AC_COMPILE_IFELSE([AC_LANG_SOURCE([
     #include <x86intrin.h>
     int main(int argc, char **argv){
     __m512d sign;
     __m512d h1_real;

     __m512d x1 = _mm512_xor_pd(h1_real, sign);
     return 0;
     }
     ])],
     [can_compile_avx512_xeon=yes],
     [can_compile_avx512_xeon=no]
    )
    AC_MSG_RESULT([${can_compile_avx512_xeon}])

    AC_MSG_CHECKING([whether we compile for Xeon PHI])
    AC_COMPILE_IFELSE([AC_LANG_SOURCE([
     #include <x86intrin.h>
     int main(int argc, char **argv){
     __m512d sign;
     __m512d h1;
     __m512d h2_real;

     __m512d x1 = (__m512d) _mm512_xor_epi64((__m512i) h2_real, (__m512i) sign);
     return 0;
     }
     ])],
     [can_compile_avx512_xeon_phi=yes],
     [can_compile_avx512_xeon_phi=no]
    )
    AC_MSG_RESULT([${can_compile_avx512_xeon_phi}])

    # this is needed for the intel compiler
    if test x"$can_compile_avx512_xeon" = x"yes" ; then
      if test x"$can_compile_avx512_xeon_phi" = x"yes" ; then
        # we want only one to be true; this is ugly but could not come up with a better way
        grep Phi /proc/cpuinfo > /dev/null
        if test x"$?" = x"0" ; then
	  echo "Xeon PHI found ... disabling AVX512 Xeon"
          can_compile_avx512_xeon=no
        fi
      fi
    fi

    if test x"$can_compile_avx512_xeon" = x"yes"; then
      AC_DEFINE([HAVE_AVX512_XEON],[1],[AVX512 for Xeon is supported on this CPU])
    else
      if test x"$can_compile_avx512_xeon_phi" = x"yes"; then
        AC_DEFINE([HAVE_AVX512_XEON_PHI],[1],[AVX512 for Xeon-PHI is supported on this CPU])
      else
        AC_MSG_ERROR([Oho! We can neither compile AVX512 intrinsics for Xeon nor Xeon Phi. This should not happen!])
      fi
    fi
  fi
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

#compatibiility flag
AC_MSG_CHECKING(whether GPU version should be used)
AC_ARG_ENABLE([gpu],
                AS_HELP_STRING([--enable-gpu],
                               [do use Nvidia GPU version (compatibility flag, better set explicitely)]),
                               [if test x"$enableval" = x"yes"; then
                                  use_gpu=yes
                                else
                                  use_gpu=no
                                fi],
              [use_gpu=no])
AC_MSG_RESULT([${use_gpu}])
if test x"${use_gpu}" = x"yes" ; then
  need_nvidia_gpu=yes
  use_real_nvidia_gpu=yes
  use_complex_nvidia_gpu=yes
fi

AC_MSG_CHECKING(whether Nvidia-GPU version should be used)
AC_ARG_ENABLE([Nvidia-gpu],
                AS_HELP_STRING([--enable-Nvidia-gpu],
                               [do use Nvidia GPU version]),
                               [if test x"$enableval" = x"yes"; then
                                  use_nvidia_gpu=yes
                                else
                                  use_nvidia_gpu=no
                                fi],
              [use_nvidia_gpu=no])
AC_MSG_RESULT([${use_nvidia_gpu}])
if test x"${use_nvidia_gpu}" = x"yes" ; then
  need_nvidia_gpu=yes
  use_nvidia_real_gpu=yes
  use_nvidia_complex_gpu=yes
fi

#AC_MSG_CHECKING(whether INTEL GPU version should be used)
#AC_ARG_ENABLE([intel-gpu],
#                AS_HELP_STRING([--enable-intel-gpu],
#                               [do use INTEL GPU version]),
#                               [if test x"$enableval" = x"yes"; then
#                                  use_intel_gpu=yes
#                                else
#                                  use_intel_gpu=no
#                                fi],
#              [use_intel_gpu=no])
#AC_MSG_RESULT([${use_intel_gpu}])
#if test x"${use_intel_gpu}" = x"yes" ; then
#  need_intel_gpu=yes
#  use_real_intel_gpu=yes
#  use_complex_intel_gpu=yes
#fi

AC_MSG_CHECKING(whether AMD-GPU version should be used)
AC_ARG_ENABLE([AMD-gpu],
                AS_HELP_STRING([--enable-AMD-gpu],
                               [do use AMD GPU version]),
                               [if test x"$enableval" = x"yes"; then
                                  use_amd_gpu=yes
                                else
                                  use_amd_gpu=no
                                fi],
              [use_amd_gpu=no])
AC_MSG_RESULT([${use_amd_gpu}])
if test x"${use_amd_gpu}" = x"yes" ; then
  need_amd_gpu=yes
  use_real_amd_gpu=yes
  use_complex_amd_gpu=yes
fi


if test x"${need_nvidia_gpu}" = x"yes" ; then
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
    AC_MSG_ERROR([nvcc not found; try to set the cuda-path or disable Nvidia GPU support])
  fi

  dnl check whether we find cublas
  AC_SEARCH_LIBS([cublasDgemm],[cublas],[have_cublas=yes],[have_cublas=no])
  if test x"${have_cublas}" = x"no"; then
    AC_MSG_ERROR([Could not link cublas; try to set the cuda-path or disable Nvidia GPU support])
  fi
  AC_SEARCH_LIBS([cudaMemcpy],[cudart],[have_cudart=yes],[have_cudart=no])
  if test x"${have_cudart}" = x"no"; then
    AC_MSG_ERROR([Could not link cudart; try to set the cuda-path or disable Nvidia GPU support])
  fi
  AC_LANG_POP([C])
fi


if test x"${need_amd_gpu}" = x"yes" ; then
  AC_LANG_PUSH([C])
  #CUDA_CFLAGS="$CUDA_CFLAGS -arch $cuda_compute_capability -O2 -I$CUDA_INSTALL_PATH/include"
  #LDFLAGS="$LDFLAGS -L$CUDA_INSTALL_PATH/lib64"
  #NVCCFLAGS="$NVCCFLAGS $CUDA_CFLAGS $CUDA_LDFLAGS"
  #NVCC="nvcc"
  #AC_SUBST(NVCC)
  #AC_SUBST(NVCCFLAGS)

  dnl check whether hipcc compiler is found
  AC_CHECK_PROG(hipcc_found,hipcc,yes,no)
  if test x"${hipcc_found}" = x"no" ; then
    AC_MSG_ERROR([hipcc not found; try to set the hip-path or disable AMD GPU support])
  fi

  #dnl check whether we find rocblas
  AC_SEARCH_LIBS([rocblas_dgemm],[rocblas],[have_rocblas=yes],[have_rocblas=no])
  if test x"${have_rocblas}" = x"no"; then
    AC_MSG_ERROR([Could not link rocblas; try to set the hip-path or disable AMD GPU support])
  fi
  #AC_SEARCH_LIBS([hipMemcpy],[cudart],[have_cudart=yes],[have_cudart=no])
  #if test x"${have_cudart}" = x"no"; then
  #  AC_MSG_ERROR([Could not link cudart; try to set the cuda-path or disable Nvidia GPU support])
  #fi
  #AC_LANG_POP([C])
fi

if test x"${need_intel_gpu}" = x"yes" ; then
  AC_LANG_PUSH([C])
  #CUDA_CFLAGS="$CUDA_CFLAGS -arch $cuda_compute_capability -O2 -I$CUDA_INSTALL_PATH/include"
  #LDFLAGS="$LDFLAGS -L$CUDA_INSTALL_PATH/lib64"
  #NVCCFLAGS="$NVCCFLAGS $CUDA_CFLAGS $CUDA_LDFLAGS"
  #NVCC="nvcc"
  #AC_SUBST(NVCC)
  #AC_SUBST(NVCCFLAGS)

  dnl check whether hipcc compiler is found
  AC_CHECK_PROG(icx_found,icx,yes,no)
  if test x"${icx_found}" = x"no" ; then
    AC_MSG_ERROR([icx not found; try to set the oneapi-path or disable INTEL GPU support])
  fi

  ##dnl check whether we find rocblas
  #AC_SEARCH_LIBS([rocblas_dgemm],[rocblas],[have_rocblas=yes],[have_rocblas=no])
  #if test x"${have_rocblas}" = x"no"; then
  #  AC_MSG_ERROR([Could not link rocblas; try to set the hip-path or disable AMD GPU support])
  #fi
  #AC_SEARCH_LIBS([hipMemcpy],[cudart],[have_cudart=yes],[have_cudart=no])
  #if test x"${have_cudart}" = x"no"; then
  #  AC_MSG_ERROR([Could not link cudart; try to set the cuda-path or disable Nvidia GPU support])
  #fi
  #AC_LANG_POP([C])
fi

AC_MSG_CHECKING(whether Nvidia GPU memory debugging should be enabled)
AC_ARG_ENABLE([nvidia-gpu-memory-debug],
                AS_HELP_STRING([--enable-nvidia-gpu-memory-debug],
                               [Output information on Nvidia GPU memory to be processed by utils/memory/check_memory.py]),
                               [if test x"$enableval" = x"yes"; then
                                  enable_nvidia_gpu_memory_debug=yes
                                else
                                  enable_nvidia_gpu_memory_debug=no
                                fi],
              [enable_nvidia_gpu_memory_debug=no])
AC_MSG_RESULT([${enable_nvidia_gpu_memory_debug}])
if test x"${enable_nvidia_gpu_memory_debug}" = x"yes" ; then
  AC_DEFINE([DEBUG_CUDA],[1],[enable CUDA debugging])
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

AM_CONDITIONAL([WITH_NVIDIA_GPU_VERSION],[test x"$use_real_nvidia_gpu" = x"yes" -o x"$use_complex_nvidia_gpu" = x"yes"])
if test x"$use_real_nvidia_gpu" = x"yes" -o x"$use_complex_nvidia_gpu" = x"yes" ; then
  AC_DEFINE([WITH_NVIDIA_GPU_VERSION],[1],[enable Nvidia GPU support])
  AC_DEFINE([WITH_NVIDIA_GPU_KERNEL],[1],[Nvidia GPU kernel should be build])
  ELPA_2STAGE_COMPLEX_NVIDIA_GPU_COMPILED=1
  ELPA_2STAGE_REAL_NVIDIA_GPU_COMPILED=1

  AC_MSG_CHECKING(whether --enable-nvtx is specified)
  AC_ARG_ENABLE([nvtx],
                AS_HELP_STRING([--enable-nvtx],
                               [build and install nvtx wrapper for profiling the Nvidia GPU version, default no.]),
                [
                 if test x"$enableval" = x"yes"; then
                   enable_nvtx=yes
                 else
                   enable_nvtx=no
                 fi
                 ],
                [enable_nvtx=no])
  AC_MSG_RESULT([${enable_nvtx}])
  if test x"${enable_nvtx}" = x"yes"; then
    AC_DEFINE([WITH_NVTX],[1],[enable NVTX support])
    AC_LANG_PUSH([C])
    AC_SEARCH_LIBS([nvtxRangePop],[nvToolsExt],[have_nvtoolsext=yes],[have_nvtoolsext=no])
    if test x"${have_nvtoolsext}" = x"no"; then
      AC_MSG_ERROR([Could not link nvToolsExt; try to set the cuda-path or disable Nvidia GPU support ])
    fi
    AC_LANG_POP([C])
  fi
else
  ELPA_2STAGE_COMPLEX_NVIDIA_GPU_COMPILED=0
  ELPA_2STAGE_REAL_NVIDIA_GPU_COMPILED=0
fi
AC_SUBST([ELPA_2STAGE_COMPLEX_NVIDIA_GPU_COMPILED])
AC_SUBST([ELPA_2STAGE_REAL_NVIDIA_GPU_COMPILED])

AM_CONDITIONAL([WITH_AMD_GPU_VERSION],[test x"$use_real_amd_gpu" = x"yes" -o x"$use_complex_amd_gpu" = x"yes"])
if test x"$use_real_amd_gpu" = x"yes" -o x"$use_complex_amd_gpu" = x"yes" ; then
  AC_DEFINE([WITH_AMD_GPU_VERSION],[1],[enable AMD GPU support])
  AC_DEFINE([WITH_AMD_GPU_KERNEL],[1],[AMD GPU kernel should be build])
  ELPA_2STAGE_COMPLEX_AMD_GPU_COMPILED=1
  ELPA_2STAGE_REAL_AMD_GPU_COMPILED=1

  #AC_MSG_CHECKING(whether --enable-nvtx is specified)
  #AC_ARG_ENABLE([nvtx],
  #              AS_HELP_STRING([--enable-nvtx],
  #                             [build and install nvtx wrapper for profiling the Nvidia GPU version, default no.]),
  #              [
  #               if test x"$enableval" = x"yes"; then
  #                 enable_nvtx=yes
  #               else
  #                 enable_nvtx=no
  #               fi
  #               ],
  #              [enable_nvtx=no])
  #AC_MSG_RESULT([${enable_nvtx}])
  #if test x"${enable_nvtx}" = x"yes"; then
  #  AC_DEFINE([WITH_NVTX],[1],[enable NVTX support])
  #  AC_LANG_PUSH([C])
  #  AC_SEARCH_LIBS([nvtxRangePop],[nvToolsExt],[have_nvtoolsext=yes],[have_nvtoolsext=no])
  #  if test x"${have_nvtoolsext}" = x"no"; then
  #    AC_MSG_ERROR([Could not link nvToolsExt; try to set the cuda-path or disable Nvidia GPU support ])
  #  fi
  #  AC_LANG_POP([C])
  #fi
else
  ELPA_2STAGE_COMPLEX_AMD_GPU_COMPILED=0
  ELPA_2STAGE_REAL_AMD_GPU_COMPILED=0
fi
AC_SUBST([ELPA_2STAGE_COMPLEX_AMD_GPU_COMPILED])
AC_SUBST([ELPA_2STAGE_REAL_AMD_GPU_COMPILED])

AM_CONDITIONAL([WITH_INTEL_GPU_VERSION],[test x"$use_real_intel_gpu" = x"yes" -o x"$use_complex_intel_gpu" = x"yes"])
if test x"$use_real_intel_gpu" = x"yes" -o x"$use_complex_intel_gpu" = x"yes" ; then
  AC_DEFINE([WITH_INTEL_GPU_VERSION],[1],[enable INTEL GPU support])
  AC_DEFINE([WITH_INTEL_GPU_KERNEL],[1],[INTEL GPU kernel should be build])
  ELPA_2STAGE_COMPLEX_INTEL_GPU_COMPILED=1
  ELPA_2STAGE_REAL_INTEL_GPU_COMPILED=1
else
  ELPA_2STAGE_COMPLEX_INTEL_GPU_COMPILED=0
  ELPA_2STAGE_REAL_INTEL_GPU_COMPILED=0
fi
AC_SUBST([ELPA_2STAGE_COMPLEX_INTEL_GPU_COMPILED])
AC_SUBST([ELPA_2STAGE_REAL_INTEL_GPU_COMPILED])


LT_INIT

DX_PDF_FEATURE(OFF)
DX_PS_FEATURE(OFF)
DX_MAN_FEATURE(ON)
DX_HTML_FEATURE(ON)
DX_INIT_DOXYGEN([ELPA], [Doxyfile], [docs])

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

AC_MSG_CHECKING(whether Fortran2008 features should be enabled)
AC_ARG_ENABLE([Fortran2008-features],
              AS_HELP_STRING([--enable-Fortran2008-features],
                             [enables some Fortran 2008 features, default yes.]),
			     [
			      if test x"$enableval" = x"yes"; then
			        enable_fortran2008_features=yes
		              else
			        enable_fortran2008_features=no
			      fi
			      ],
              [enable_fortran2008_features=yes])
AC_MSG_RESULT([${enable_fortran2008_features}])
AM_CONDITIONAL([USE_FORTRAN2008],[test x"$enable_fortran2008_features" = x"yes"])
if test x"${enable_fortran2008_features}" = x"yes"; then
 AC_DEFINE([USE_FORTRAN2008], [1], [use some Fortran 2008 features])
fi

AC_MSG_CHECKING(whether autotuning functionality should be enabled)
AC_ARG_ENABLE([autotuning],
              AS_HELP_STRING([--enable-autotuning],
                             [enables autotuning functionality, default yes.]),
			     [
			      if test x"$enableval" = x"yes"; then
			        enable_autotuning=yes
		              else
			        enable_autotuning=no
			      fi
			      ],
              [enable_autotuning=yes])
AC_MSG_RESULT([${enable_autotuning}])
AM_CONDITIONAL([ENABLE_AUTOTUNING],[test x"$enable_autotuning" = x"yes"])
if test x"${enable_autotuning}" = x"yes"; then
 AC_DEFINE([ENABLE_AUTOTUNING], [1], [enable autotuning functionality])
fi

dnl Scalapack tests
AC_MSG_CHECKING(whether --enable-scalapack-tests is specified)
AC_ARG_ENABLE([scalapack-tests],
              AS_HELP_STRING([--enable-scalapack-tests],
                             [build SCALAPACK test cases for performance comparison, needs MPI, default no.]),
              [
	       if test x"$enableval" = x"yes"; then
	         enable_scalapack_tests=yes
	       else
	         enable_scalapack_tests=no
	       fi
	       ],
              [enable_scalapack_tests="no"])
AC_MSG_RESULT([$enable_scalapack_tests])
if test x"${enable_scalapack_tests}" = x"yes"; then
        if test x"$with_mpi" = x"no"; then
                AC_MSG_ERROR([You cannot build the SCALAPCK test cases without MPI])
        fi
        AC_DEFINE([WITH_SCALAPACK_TESTS], [1], [build SCALAPACK test cases])
fi
AM_CONDITIONAL([WITH_SCALAPACK_TESTS], [test x"$enable_scalapack_tests" = x"yes"])


AC_MSG_CHECKING(whether matrix redistribution should be considered in autotuning)
AC_ARG_ENABLE([autotune-redistribute-matrix],
                AS_HELP_STRING([--enable-autotune-redistribute-matrix],
			       [Allows ELPA during autotuning to re-distribute the matrix to find the best (ELPA internal) block size for block-cyclic distribution (Needs Scalapack functionality)]),
                               [if test x"$enableval" = x"yes"; then
                                  enable_autotune_redistribute_matrix=yes
                                else
                                  enable_autotune_redistribute_matrix=no
                                fi],
              [enable_autotune_redistribute_matrix=no])
AC_MSG_RESULT([${enable_autotune_redistribute_matrix}])
if test x"${enable_autotune_redistribute_matrix}" = x"yes" ; then
  if test x"${enable_scalapack_tests}" = x"no"; then
    AC_MSG_ERROR([Please also set --enable_scalapack_tests in this case])
  fi
  if test x"${with_mpi}" = x"no"; then
    AC_MSG_ERROR([For this option ELPA must be build with MPI enabled])
  fi
  AC_DEFINE([REDISTRIBUTE_MATRIX],[1],[enable matrix re-distribution during autotuning])
fi




dnl Scalapack tests
AC_MSG_CHECKING(whether --enable-scalapack-tests is specified)
AC_ARG_ENABLE([scalapack-tests],
              AS_HELP_STRING([--enable-scalapack-tests],
                             [build SCALAPACK test cases for performance comparison, needs MPI, default no.]),
              [
	       if test x"$enableval" = x"yes"; then
	         enable_scalapack_tests=yes
	       else
	         enable_scalapack_tests=no
	       fi
	       ],
              [enable_scalapack_tests="no"])
AC_MSG_RESULT([$enable_scalapack_tests])
if test x"${enable_scalapack_tests}" = x"yes"; then
        if test x"$with_mpi" = x"no"; then
                AC_MSG_ERROR([You cannot build the SCALAPCK test cases without MPI])
        fi
        AC_DEFINE([WITH_SCALAPACK_TESTS], [1], [build SCALAPACK test cases])
fi
AM_CONDITIONAL([WITH_SCALAPACK_TESTS], [test x"$enable_scalapack_tests" = x"yes"])


AC_MSG_CHECKING(whether matrix redistribution should be considered in autotuning)
AC_ARG_ENABLE([autotune-redistribute-matrix],
                AS_HELP_STRING([--enable-autotune-redistribute-matrix],
			       [Allows ELPA during autotuning to re-distribute the matrix to find the best (ELPA internal) block size for block-cyclic distribution (Needs Scalapack functionality)]),
                               [if test x"$enableval" = x"yes"; then
                                  enable_autotune_redistribute_matrix=yes
                                else
                                  enable_autotune_redistribute_matrix=no
                                fi],
              [enable_autotune_redistribute_matrix=no])
AC_MSG_RESULT([${enable_autotune_redistribute_matrix}])
if test x"${enable_autotune_redistribute_matrix}" = x"yes" ; then
  if test x"${enable_scalapack_tests}" = x"no"; then
    AC_MSG_ERROR([Please also set --enable_scalapack_tests in this case])
  fi
  if test x"${with_mpi}" = x"no"; then
    AC_MSG_ERROR([For this option ELPA must be build with MPI enabled])
  fi
  AC_DEFINE([REDISTRIBUTE_MATRIX],[1],[enable matrix re-distribution during autotuning])
fi




AC_MSG_CHECKING(whether C tests should be provided)
AC_ARG_ENABLE([c-tests],
              AS_HELP_STRING([--enable-c-tests],
                             [enables the C tests for elpa, default yes.]),
			     [
			      if test x"$enableval" = x"yes"; then
			        enable_c_tests=yes
		              else
			        enable_c_tests=no
			      fi
			      ],
              [enable_c_tests=yes])
AC_MSG_RESULT([${enable_c_tests}])
AM_CONDITIONAL([ENABLE_C_TESTS],[test x"$enable_c_tests" = x"yes"])
if test x"${enable_c_tests}" = x"yes"; then
 AC_DEFINE([ENABLE_C_TESTS], [1], [enable C tests])
fi

AC_MSG_CHECKING(whether we build for K-computer)
AC_ARG_ENABLE([K-computer],
              AS_HELP_STRING([--enable-K-computer],
                             [enable builds on K-Computer, default no.]),
			     [if test x"$enableval" = x"yes"; then
			        enable_kcomputer=yes
		              else
				enable_kcomputer=no
			      fi],
              [enable_kcomputer=no])
AC_MSG_RESULT([${enable_kcomputer}])
AM_CONDITIONAL([BUILD_KCOMPUTER],[test x"$enable_kcomputer" = x"yes"])
if test x"${enable_kcomputer}" = x"yes"; then
 AC_DEFINE([BUILD_KCOMPUTER], [1], [build for K-Computer])
 FC_MODINC="-I"
  if test x"${USE_ASSUMED_SIZE}" = x"yes" ; then
    AC_MSG_ERROR(on K-computer you have to switch off assumed-size arrays!)
  fi
  if test x"${enable_fortran2008_features}" = x"yes" ; then
    AC_MSG_ERROR(on K-computer you have to switch off Fortran 2008 features!)
  fi
fi

AC_MSG_CHECKING(whether we build for NEC SX-Auroa)
AC_ARG_ENABLE([SX-Aurora],
              AS_HELP_STRING([--enable-SX-Aurora],
                             [enable builds on SX-Aurora, default no.]),
			     [if test x"$enableval" = x"yes"; then
			        enable_sxaurora=yes
		              else
				enable_sxaurora=no
			      fi],
              [enable_kcomputer=no])
AC_MSG_RESULT([${enable_sxaurora}])
AM_CONDITIONAL([BUILD_KCOMPUTER],[test x"$enable_sxaurora" = x"yes"])
if test x"${enable_sxaurora}" = x"yes"; then
 AC_DEFINE([BUILD_SXAURORA], [1], [build for SX-Aurora])
 FC_MODINC="-I"
  #if test x"${USE_ASSUMED_SIZE}" = x"yes" ; then
  #  AC_MSG_ERROR(on K-computer you have to switch off assumed-size arrays!)
  #fi
  if test x"${enable_fortran2008_features}" = x"yes" ; then
    AC_MSG_ERROR(on SX-Aurora you have to switch off Fortran 2008 features!)
  fi
fi

if test x"${want_single_precision}" = x"yes" ; then
  AC_DEFINE([WANT_SINGLE_PRECISION_REAL],[1],[build also single-precision for real calculation])
  AC_DEFINE([WANT_SINGLE_PRECISION_COMPLEX],[1],[build also single-precision for complex calculation])
fi
AM_CONDITIONAL([WANT_SINGLE_PRECISION_REAL],[test x"$want_single_precision" = x"yes"])
AM_CONDITIONAL([WANT_SINGLE_PRECISION_COMPLEX],[test x"$want_single_precision" = x"yes"])

#always define SKEWSYMMETRIC for the moment

AC_MSG_CHECKING(whether we should enable skew-symmetric support)
AC_ARG_ENABLE([skew-symmetric-support],
              AS_HELP_STRING([--enable-skew-symmetric-support],
                             [enable support for real valued skew-symmetric matrices]),
			     [if test x"$enableval" = x"yes"; then
			        enable_skewsymmetric=yes
		              else
				enable_skewsymmetric=no
			      fi],
              [enable_skewsymmetric=no])
AC_MSG_RESULT([${enable_skewsymmetric}])
AM_CONDITIONAL([HAVE_SKEWSYMMETRIC],[test x"$enable_skewsymmetric" = x"yes"])
if test x"${enable_skewsymmetric}" = x"yes"; then
  AC_DEFINE([HAVE_SKEWSYMMETRIC],[1],[build for skewsyemmtric case])
fi

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

dnl store-build-config
AC_ARG_ENABLE([store-build-config],
              AS_HELP_STRING([--enable-store-build-config],
                             [compile build config into the library object, default no]),
              [
	       if test x"$enableval" = x"yes"; then
	         store_build_config=yes
	       else
	         store_build_config=no
	       fi
	       ],
              [store_build_config=no])
AM_CONDITIONAL([STORE_BUILD_CONFIG],[test x"$store_build_config" = x"yes"])
if test x"${store_build_config}" = x"yes"; then
	echo "build config should be compiled into the library: yes"
        AC_DEFINE([STORE_BUILD_CONFIG], [1], [compile build config into the library object])
        echo "checking whether build config should be compiled into the library..."
        AC_CHECK_PROG(xxd_CHECK,xxd,yes)
        AS_IF([test x"$xxd_CHECK" != x"yes"], [AC_MSG_ERROR([Please install xxd before configuring.])])
else
	echo "build config should be compiled into the library: no"
fi

if test x"$have_loop_blocking" = x"yes"; then
  AC_DEFINE([LOOP_BLOCKING],[1],[use blocking in loops])
fi

AC_SUBST([SUFFIX])
AC_SUBST([PKG_CONFIG_FILE],[elpa${SUFFIX}-${PACKAGE_VERSION}.pc])

AC_CONFIG_FILES([
  Makefile
  Doxyfile
  ${PKG_CONFIG_FILE}:elpa.pc.in
  elpa/elpa_constants.h
  elpa/elpa_version.h
  elpa/elpa_build_config.h
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

AC_MSG_CHECKING(whether --enable-python is specified)
AC_ARG_ENABLE([python],
              AS_HELP_STRING([--enable-python],
                             [build and install python wrapper, default no.]),
              [
	       if test x"$enableval" = x"yes"; then
	         enable_python=yes
	       else
	         enable_python=no
	       fi
	       ],
              [enable_python=no])
AC_MSG_RESULT([${enable_python}])
AM_CONDITIONAL([WITH_PYTHON],[test x"$enable_python" = x"yes"])
if test x"${enable_python}" = x"yes"; then
        AC_DEFINE([WITH_PYTHON], [1], [build and install python wrapper])
        # check for python and dependencies
        AM_PATH_PYTHON([3.6])
        AC_ARG_VAR([PYTHON_INCLUDE], [Include flags for python, bypassing python-config])
        AC_ARG_VAR([PYTHON_CONFIG], [Path to python-config])
        AS_IF([test -z "$PYTHON_INCLUDE"], [
          AS_IF([test -z "$PYTHON_CONFIG"], [
            AC_PATH_PROGS([PYTHON_CONFIG],
                          [python$PYTHON_VERSION-config python-config],
                          [no],
                          [`dirname $PYTHON`])
            AS_IF([test "$PYTHON_CONFIG" = no], [AC_MSG_ERROR([cannot find python-config for $PYTHON.])])
          ])
          AC_MSG_CHECKING([python include flags])
          PYTHON_INCLUDE=`$PYTHON_CONFIG --includes`
          AC_MSG_RESULT([$PYTHON_INCLUDE])
        ])
        AC_MSG_CHECKING([numpy module])
        AS_IF([$PYTHON -c "import numpy"], [AC_MSG_RESULT([found.])],
              [AC_MSG_ERROR([cannot find numpy.])])
        AC_MSG_CHECKING([mpi4py module])
        AS_IF([$PYTHON -c "import mpi4py"], [AC_MSG_RESULT([found.])],
              [AC_MSG_ERROR([cannot find mpi4py.])])
        AC_MSG_CHECKING([cython module])
        AS_IF([$PYTHON -c "import cython"], [AC_MSG_RESULT([found.])],
              [AC_MSG_ERROR([cannot find cython.])])
        AC_CHECK_PROG([cython_found], [cython], [yes], [no])
        if test x"$cython_found" != x"yes" ; then
          AC_MSG_ERROR([cython not found.])
        fi
        AC_ARG_VAR([NUMPY_INCLUDE], [Include flags for numpy])
        AC_MSG_CHECKING([numpy include flags])
        NUMPY_INCLUDE=-I`$PYTHON -c "import numpy; print(numpy.get_include())"`
        AS_IF([test "$NUMPY_INCLUDE" = "-I"], [AC_MSG_ERROR([cannot get numpy include path.])])
        AC_MSG_RESULT([$NUMPY_INCLUDE])
fi
AC_MSG_CHECKING(whether --enable-python-tests is specified)
AC_ARG_ENABLE([python-tests],
              AS_HELP_STRING([--enable-python-tests],
                             [enable python tests, default no.]),
              [
	       if test x"$enableval" = x"yes"; then
	         enable_python_tests=yes
	       else
	         enable_python_tests=no
	       fi
	       ],
              [enable_python_tests=no])
AC_MSG_RESULT([${enable_python_tests}])
AM_CONDITIONAL([WITH_PYTHON_TESTS],[test x"$enable_python_tests" = x"yes"])
if test x"${enable_python_tests}" = x"yes"; then
        if test x"${enable_python}" = x"no"; then
          AC_MSG_ERROR([Python tests can only be enabled it python is enabled.])
        fi
        AC_CHECK_PROG([pytest_found], [pytest], [yes], [no])
        if test x"$pytest_found" != x"yes" ; then
          AC_MSG_ERROR([pytest not found.])
        fi
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

echo ""
echo " This version of ELPA support the minimal API version: ${print_earliest_api_version}"
echo " The current API version is: ${print_current_api_version}"
echo " This version of ELPA support the minimal autotuning version: ${print_earliest_autotune_version}"
echo " The current autotune version is: ${print_current_autotune_version}"
echo ""

if test x"${ax_cv_have_sse3_cpu_ext}" = x"yes" -a  x"${need_sse}" = x"no"; then
  echo " "
  AC_MSG_WARN([You did not request SSE support (--enable-sse), but your local CPU supports it.])
  AC_MSG_WARN([You might want to re-configure, except you are cross-compiling])
  echo " "
fi

if test x"${ax_cv_have_sse3_cpu_ext}" = x"yes" -a  x"${need_sse_assembly}" = x"no"; then
  echo " "
  AC_MSG_WARN([You did not request SSE-ASSEMBLY support (--enable-sse-assembly), but your local CPU supports it.])
  AC_MSG_WARN([You might want to re-configure, except you are cross-compiling])
  echo " "
fi

if test x"${ax_cv_have_avx_cpu_ext}" = x"yes" -a  x"${need_avx}" = x"no"; then
  echo " "
  AC_MSG_WARN([You did not request AVX support (--enable-avx), but your local CPU supports it.])
  AC_MSG_WARN([You might want to re-configure, except you are cross-compiling])
  echo " "
fi

if test x"${ax_cv_have_avx2_cpu_ext}" = x"yes" -a  x"${need_avx2}" = x"no"; then
  echo " "
  AC_MSG_WARN([You did not request AVX2 support (--enable-avx2), but your local CPU supports it.])
  AC_MSG_WARN([You might want to re-configure, except you are cross-compiling])
  echo " "
fi

if test x"${ax_cv_have_avx512f_cpu_ext}" = x"yes" -a  x"${need_avx512}" = x"no"; then
  echo " "
  AC_MSG_WARN([You did not request AVX512 support (--enable-avx512), but your local CPU supports it.])
  AC_MSG_WARN([You might want to re-configure, except you are cross-compiling])
  echo " "
fi

#echo " "
#echo "***********************************************************************"
#echo "*  This is the first release candidate of ELPA 2020.11.001.rc1        *"
#echo "*  There might be still some changes until the final release of       *"
#echo "*  ELPA 2020.11.001                                                   *"
#echo "***********************************************************************"
#echo " "

if test x"$enable_kcomputer" = x"yes" ; then
  echo " "
  echo "Important message:"
  echo "On K-computer (at the moment) the automatic creation of the generated"
  echo "headers does not work."
  echo "call: make -f ../generated_headers.am generated-headers top_srcdir=.."
  echo "BEFORE triggering the build with make!"
else
  if test x"$optional_c_error_argument" = x"yes" ; then
    echo "#define OPTIONAL_C_ERROR_ARGUMENT" > elpa/elpa_generated_c_api.h
  else
    echo "#undef OPTIONAL_C_ERROR_ARGUMENT" > elpa/elpa_generated_c_api.h
  fi
  if test x"$store_build_config" = x"yes"; then
    cat config.log > elpa_build_object
    xxd -i elpa_build_object >> elpa/elpa_build_config.h
  fi

  make -f $srcdir/generated_headers.am generated-headers top_srcdir="$srcdir" CPP="$CPP"
fi
if test x"$old_elpa_version" = x"yes"; then
  echo " "
  echo " It is possible that your current version of ELPA is not the latest one."
  echo " You might want to have a look at https://elpa.mpcdf.mpg.de, whether a more recent"
  echo " version has been released already"
  echo " "
fi
