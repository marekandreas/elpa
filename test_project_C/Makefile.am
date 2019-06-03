## Process this file with automake to produce Makefile.in

ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS} -I m4

AM_FCFLAGS = @FC_MODINC@modules $(ELPA_FCFLAGS)
AM_CFLAGS = $(ELPA_CFLAGS)
AM_LDFLAGS = $(ELPA_LIBS)

#bindir = $(abs_top_builddir)
bin_PROGRAMS = test_real
test_real_SOURCES = src/test_real.c src/test_blacs_infrastructure.F90

distclean-local:
	-rm config-f90.h
