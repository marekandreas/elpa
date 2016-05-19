## Process this file with automake to produce Makefile.in

ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS} -I m4

AM_FCFLAGS = @FC_MODINC@modules @FC_MODOUT@modules $(ELPA_FCFLAGS)
AM_LDFLAGS = $(ELPA_LIBS)

#bindir = $(abs_top_builddir)
bin_PROGRAMS = test_real
test_real_SOURCES = src/test_real.F90

distclean-local:
	-rm config-f90.h
