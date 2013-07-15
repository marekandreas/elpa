ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS} -I m4

AM_FCFLAGS = @AM_FCFLAGS@ @FC_MODINC@modules @FC_MODOUT@modules 
#@OPT_FCFLAGS@
AM_LDFLAGS = @AM_LDFLAGS@ @BLACS_LDFLAGS@

BLACS_LDFLAGS = @BLACS_LDFLAGS@

# libelpa
lib_LTLIBRARIES = libelpa-@ELPA_LIB_VERSION@.la

libelpa_@ELPA_LIB_VERSION@_la_SOURCES = src/elpa1.f90 src/elpa2.f90

if WITH_BGP
  libelpa_@ELPA_LIB_VERSION@_la_SOURCES += src/elpa2_kernels_bg.f90
else
  libelpa_@ELPA_LIB_VERSION@_la_SOURCES += src/elpa2_kernels.f90
endif
libelpa_@ELPA_LIB_VERSION@_la_LDFLAGS = -version-info $(ELPA_SO_VERSION)

# install any .mod files in the include/ dir
elpa_includedir = $(includedir)/elpa-$(ELPA_LIB_VERSION)
nobase_elpa_include_HEADERS = $(wildcard modules/*)

# other files to distribute
filesdir = $(datarootdir)
files_DATA = \
	test/read_real.f90 \
	test/read_real_gen.f90 \
	test/test_complex2.f90 \
	test/test_complex.f90 \
	test/test_complex_gen.f90 \
	test/test_real2.f90 \
	test/test_real.f90 \
	test/test_real_gen.f90

# pkg-config stuff
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = elpa-$(ELPA_LIB_VERSION).pc

# test programs
noinst_bindir = $(abs_top_builddir)
noinst_bin_PROGRAMS = test_real test_real2 test_complex test_complex2

test_real_SOURCES = test/test_real.f90
test_real_LDADD = libelpa-@ELPA_LIB_VERSION@.la 

test_real2_SOURCES = test/test_real2.f90
test_real2_LDADD = libelpa-@ELPA_LIB_VERSION@.la 

test_complex_SOURCES = test/test_complex.f90
test_complex_LDADD = libelpa-@ELPA_LIB_VERSION@.la

test_complex2_SOURCES = test/test_complex2.f90
test_complex2_LDADD = libelpa-@ELPA_LIB_VERSION@.la


check_SCRIPTS = test_real.sh test_real2.sh test_complex.sh test_complex2.sh   
TESTS = $(check_SCRIPTS)

test_real.sh:
	echo "mpiexec -n 2 ./test_real > /dev/null 2>&1" > test_real.sh
	chmod +x test_real.sh	

test_real2.sh:
	echo "mpiexec -n 2 ./test_real2 > /dev/null 2>&1" > test_real2.sh
	chmod +x test_real2.sh	

test_complex.sh:
	echo "mpiexec -n 2 ./test_complex > /dev/null 2>&1" > test_complex.sh
	chmod +x test_complex.sh	

test_complex2.sh:
	echo "mpiexec -n 2 ./test_complex2 > /dev/null 2>&1" > test_complex2.sh
	chmod +x test_complex2.sh	


CLEANFILES = test_real.sh test_real2.sh test_complex.sh test_complex2.sh
