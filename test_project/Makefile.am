ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS} -I m4
AM_CFLAGS = @AM_CFLAGS@
AM_LDFLAGS = @AM_LDFLAGS@

# programs
bin_PROGRAMS = foo test_bar test_baz

foo_SOURCES = src/foo.F90 src2/baz.F90 src/bar.F90
foo_CPPFLAGS = -DPROGRAM_foo
foo_FCFLAGS = $(FC_MODOUT)$(_f90_moddir)/foo $(FC_MODINC)$(_f90_moddir)/foo

test_bar_SOURCES = src/bar.F90
test_bar_CPPFLAGS = -DPROGRAM_test_bar
test_bar_FCFLAGS = $(FC_MODOUT)$(_f90_moddir)/test_bar $(FC_MODINC)$(_f90_moddir)/test_bar

test_baz_SOURCES = src2/baz.F90 src/bar.F90
test_baz_CPPFLAGS = -DPROGRAM_test_baz
test_baz_FCFLAGS = $(FC_MODOUT)$(_f90_moddir)/test_baz $(FC_MODINC)$(_f90_moddir)/test_baz

@FORTRAN_MODULE_DEPS@
