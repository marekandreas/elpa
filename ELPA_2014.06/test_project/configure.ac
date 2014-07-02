AC_PREREQ([2.69])
AC_INIT([elpa_test_project],[2014.06.000], elpa-library@rzg.mpg.de)
AC_CONFIG_SRCDIR([src/test_real.F90])

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

rm -rf config.h config-f90.h

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

AC_LANG([Fortran])
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
  if test x"${enable_openmp}" = x"yes"; then
     with_openmp=yes
     AC_MSG_CHECKING(whether --enable-openmp is specified)
     AC_MSG_RESULT([${enable_openmp}])
  fi
  AM_CONDITIONAL([WITH_OPENMP],[test x"$with_openmp" = x"yes"])
  if test x"${with_openmp}" = x"yes"; then
	AC_DEFINE([WITH_OPENMP], [1], [use OpenMP threading])
        AX_ELPA_OPENMP
	elpa="elpa_openmp-2014.06.001"
  else
	elpa="elpa-2014.06.001"
  fi

# Here comes the ELPA specific part
PKG_PROG_PKG_CONFIG
PKG_CHECK_MODULES([ELPA],[${elpa}],[],[AC_MSG_ERROR(["Need ${elpa}"])])
PKG_CHECK_VAR([ELPA_FCFLAGS],[${elpa}],[fcflags])

LT_INIT

AC_SUBST([FC_MODINC])
AC_SUBST([FC_MODOUT])

rm -rf modules/ .fortran_dependencies/
mkdir modules

AC_CONFIG_FILES([
  Makefile
])
AC_OUTPUT

grep "^#define" config.h > config-f90.h
