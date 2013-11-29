ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS} -I m4

AM_FCFLAGS = @AM_FCFLAGS@ @FC_MODINC@modules @FC_MODOUT@modules 

AM_LDFLAGS = @AM_LDFLAGS@ @BLACS_LDFLAGS@

BLACS_LDFLAGS = @BLACS_LDFLAGS@

# libelpa
if WITH_OPENMP
lib_LTLIBRARIES = libelpa_mt.la
else
lib_LTLIBRARIES = libelpa.la
endif

if WITH_QR
if WITH_OPENMP
libelpa_mt_la_SOURCES = src/elpa_qr/qr_utils.f90 src/elpa_qr/elpa_qrkernels.f90 src/elpa_qr/elpa_pdlarfb.f90 src/elpa_qr/elpa_pdgeqrf.f90  src/elpa1.F90 src/elpa2.F90
else
libelpa_la_SOURCES = src/elpa_qr/qr_utils.f90 src/elpa_qr/elpa_qrkernels.f90 src/elpa_qr/elpa_pdlarfb.f90 src/elpa_qr/elpa_pdgeqrf.f90  src/elpa1.F90 src/elpa2.F90 
endif

else
if WITH_OPENMP
libelpa_mt_la_SOURCES = src/elpa1.F90 src/elpa2.F90
else
libelpa_la_SOURCES = src/elpa1.F90 src/elpa2.F90
endif
endif


##rule to produce fortran config file:
#config_f90.h: ./config.h
#	grep "^#define" ./config.h > $@


if WITH_GENERIC_SIMPLE
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_simple.f90 \
                                           src/elpa2_kernels/elpa2_kernels_real_simple.f90
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_simple.f90 \
                                           src/elpa2_kernels/elpa2_kernels_real_simple.f90
endif
endif

if WITH_GENERIC
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex.f90 \
                                           src/elpa2_kernels/elpa2_kernels_real.f90
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex.f90 \
                                           src/elpa2_kernels/elpa2_kernels_real.f90
endif
endif

if WITH_BGP
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgp.f90 \
                                           src/elpa2_kernels/elpa2_kernels_complex.f90
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgp.f90 \
                                           src/elpa2_kernels/elpa2_kernels_complex.f90
endif 
endif

if WITH_BGQ
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgq.f90 \
                                           src/elpa2_kernels/elpa2_kernels_complex.f90 
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgq.f90 \
                                           src/elpa2_kernels/elpa2_kernels_complex.f90 
endif
endif

if WITH_SSE_AS
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64.s
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64.s
endif
endif

if WITH_AVX_SANDYBRIDGE
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c \
                                           src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c \
                                           src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
endif
endif

if WITH_AMD_BULLDOZER
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_4hv.c \
                                           src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c \
                                           src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_4hv.c \
                                           src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c \
                                           src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
endif
endif
if WITH_AVX_COMPLEX_BLOCK1
if WITH_OPENMP
 libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
endif
endif

if WITH_AVX_COMPLEX_BLOCK2
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse-avx_2hv.cpp \
                                           src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse-avx_2hv.cpp \
                                           src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
endif
endif

if WITH_AVX_REAL_BLOCK2
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c
endif
endif

if WITH_AVX_REAL_BLOCK4
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_4hv.c
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_4hv.c
endif
endif

if WITH_AVX_REAL_BLOCK6
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_6hv.c
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_6hv.c
endif
endif

if WITH_OPENMP
libelpa_la_LDFLAGS = -version-info $(ELPA_SO_VERSION)
else
libelpa_mt_la_LDFLAGS = -version-info $(ELPA_SO_VERSION)
endif

# install any .mod files in the include/ dir
elpa_includedir = $(includedir)/elpa
nobase_elpa_include_HEADERS = $(wildcard modules/*)

# other files to distribute
filesdir = $(datarootdir)
files_DATA = \
	test/read_real.F90 \
	test/read_real_gen.F90 \
	test/test_complex2.F90 \
	test/test_complex.F90 \
	test/test_complex_gen.F90 \
	test/test_real2.F90 \
	test/test_real.F90 \
	test/test_real_gen.F90

# pkg-config stuff
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = elpa.pc

# test programs
if WITH_OPENMP
build_lib = libelpa_mt.la 
else
build_lib = libelpa.la 
endif

#bindir = $(abs_top_builddir)
bin_PROGRAMS = test_real test_real2 test_complex test_complex2

test_real_SOURCES = test/test_real.F90
test_real_LDADD = $(build_lib)

test_real2_SOURCES = test/test_real2.F90
test_real2_LDADD = $(build_lib)

test_complex_SOURCES = test/test_complex.F90
test_complex_LDADD = $(build_lib)

test_complex2_SOURCES = test/test_complex2.F90
test_complex2_LDADD = $(build_lib)


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
