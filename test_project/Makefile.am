ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS} -I m4
AM_CFLAGS = @AM_CFLAGS@
AM_LDFLAGS = @AM_LDFLAGS@

# programs
bin_PROGRAMS = foo test_bar test_baz

foo_SOURCES = src/foo.F90 src2/baz.F90 src/bar.F90
foo_CPPFLAGS = -DPROGRAM_foo
foo_FCFLAGS = $(FC_MODOUT)./fortran_mod_files/foo $(FC_MODINC)./fortran_mod_files/foo

test_bar_SOURCES = src/bar.F90
test_bar_CPPFLAGS = -DPROGRAM_test_bar
test_bar_FCFLAGS = $(FC_MODOUT)./fortran_mod_files/test_bar $(FC_MODINC)./fortran_mod_files/test_bar

test_baz_SOURCES = src2/baz.F90 src/bar.F90
test_baz_CPPFLAGS = -DPROGRAM_test_baz
test_baz_FCFLAGS = $(FC_MODOUT)./fortran_mod_files/test_baz $(FC_MODINC)./fortran_mod_files/test_baz

# a library
lib_LTLIBRARIES = libdings-2.la

libdings_2_la_SOURCES = src2/baz.F90 src/bar.F90
libdings_2_la_FCFLAGS = $(FC_MODOUT)./fortran_mod_files/libdings-2.la $(FC_MODINC)./fortran_mod_files/libdings-2.la

CLEANFILES = fortran_mod_files/*/*

@FORTRAN_MODULE_DEPS@
