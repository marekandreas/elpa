ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS} -I m4

AM_FCFLAGS = @AM_FCFLAGS@ @FC_MODINC@modules @FC_MODOUT@modules 

AM_LDFLAGS = @AM_LDFLAGS@ @BLACS_LDFLAGS@

BLACS_LDFLAGS = @BLACS_LDFLAGS@

# libelpa
lib_LTLIBRARIES = libelpa.la

##rule to produce fortran config file:
#config_f90.h: ./config.h
#	grep "^#define" ./config.h > $@


libelpa_la_SOURCES = src/elpa1.f90 src/elpa2.F90


if WITH_GENERIC_SIMPLE
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_simple.f90 \
                        src/elpa2_kernels/elpa2_kernels_real_simple.f90
endif

if WITH_GENERIC
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex.f90 \
                       src/elpa2_kernels/elpa2_kernels_real.f90
endif

if WITH_BGP
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgp.f90 \
                        src/elpa2_kernels/elpa2_kernels_complex.f90 
endif

if WITH_BGQ
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgq.f90 \
                        src/elpa2_kernels/elpa2_kernels_complex.f90 
endif

if WITH_SSE_AS
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64.s
endif

if WITH_AVX_SANDYBRIDGE
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c \
                        src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
endif

if WITH_AMD_BULLDOZER
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_4hv.c \
                        src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c \
                        src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
endif
if WITH_AVX_COMPLEX_BLOCK1
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
endif

if WITH_AVX_COMPLEX_BLOCK2
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse-avx_2hv.cpp \
                        src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
endif

if WITH_AVX_REAL_BLOCK2
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c
endif

if WITH_AVX_REAL_BLOCK4
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_4hv.c
endif

if WITH_AVX_REAL_BLOCK6
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_6hv.c
endif


libelpa_la_LDFLAGS = -version-info $(ELPA_SO_VERSION)

# install any .mod files in the include/ dir
elpa_includedir = $(includedir)/elpa
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
pkgconfig_DATA = elpa.pc

# test programs
#noinst_bindir = $(abs_top_builddir)
bin_PROGRAMS = test_real test_real2 test_complex test_complex2

test_real_SOURCES = test/test_real.f90
test_real_LDADD = libelpa.la 

test_real2_SOURCES = test/test_real2.f90
test_real2_LDADD = libelpa.la 

test_complex_SOURCES = test/test_complex.f90
test_complex_LDADD = libelpa.la

test_complex2_SOURCES = test/test_complex2.f90
test_complex2_LDADD = libelpa.la


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

@FORTRAN_MODULE_DEPS@
