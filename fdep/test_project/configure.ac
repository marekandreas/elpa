AC_INIT([fdep_test], [0.1])
AC_PREREQ([2.59])
AM_INIT_AUTOMAKE([foreign -Wall subdir-objects])

AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config.h])
AM_SILENT_RULES([yes])

# gnu-make fortran module dependencies
m4_include([fdep/fortran_dependencies.m4])
FDEP_F90_GNU_MAKE_DEPS

# necessary tools
AC_PROG_FC
AC_PROG_INSTALL
AM_PROG_CC_C_O
AM_PROG_AR

# fortran stuff
AC_FC_MODULE_FLAG
AC_FC_MODULE_OUTPUT_FLAG
AC_FC_FREEFORM

# libtool
LT_INIT

AC_SUBST([AM_CFLAGS])
AC_SUBST([AM_LDFLAGS])
AC_SUBST([FC_MODINC])
AC_SUBST([FC_MODOUT])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT

for target in foo test_bar test_baz libdings-2.la; do
	mkdir -p fortran_mod_files/$target
done
