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

##rule to produce fortran config file:
#config_f90.h: ./config.h
#	grep "^#define" ./config.h > $@

if WITH_OPENMP
libelpa_mt_la_SOURCES = src/elpa1.F90 src/elpa2.F90
else
libelpa_la_SOURCES = src/elpa1.F90 src/elpa2.F90
endif

if WITH_REAL_GENERIC_KERNEL
if WITH_OPENMP
  libelpa_mt_la_SOURCES +=  src/elpa2_kernels/elpa2_kernels_real.f90
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real.f90
endif
endif

if WITH_COMPLEX_GENERIC_KERNEL
if WITH_OPENMP
  libelpa_mt_la_SOURCES +=  src/elpa2_kernels/elpa2_kernels_complex.f90
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex.f90
endif
endif

if WITH_REAL_GENERIC_SIMPLE_KERNEL
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_simple.f90
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_simple.f90
endif
endif

if WITH_COMPLEX_GENERIC_SIMPLE_KERNEL
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_simple.f90
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_simple.f90
endif
endif

if WITH_REAL_BGP_KERNEL
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgp.f90 
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgp.f90
endif 
endif

if WITH_REAL_BGQ_KERNEL
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgq.f90
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgq.f90
endif
endif

if WITH_REAL_SSE_KERNEL
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64.s
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64.s
endif
else
if WITH_COMPLEX_SSE_KERNEL
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64.s
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64.s
endif
endif
endif

if WITH_REAL_AVX_BLOCK2_KERNEL
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c
endif
endif

if WITH_REAL_AVX_BLOCK4_KERNEL
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_4hv.c
else
 libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_4hv.c
endif
endif

if WITH_REAL_AVX_BLOCK6_KERNEL
if WITH_OPENMP
  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_6hv.c
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_6hv.c
endif
endif

if WITH_COMPLEX_AVX_BLOCK1_KERNEL
if WITH_OPENMP
 libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
endif
endif

if WITH_COMPLEX_AVX_BLOCK2_KERNEL
if WITH_OPENMP
 libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse-avx_2hv.cpp
else
  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse-avx_2hv.cpp
endif
endif

#if WITH_AVX_SANDYBRIDGE
#if WITH_OPENMP
#  libelpa_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c \
                                           src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
#else
#  libelpa_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c \
                                           src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
#endif
#endif

if WITH_OPENMP
libelpa_mt_la_LDFLAGS = -version-info $(ELPA_SO_VERSION)
else
libelpa_la_LDFLAGS = -version-info $(ELPA_SO_VERSION)
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
	test/test_complex2_default_kernel.F90 \
	test/test_complex2_choose_kernel_with_api.F90 \
	test/test_complex.F90 \
	test/test_complex_gen.F90 \
	test/test_real2.F90 \
	test/test_real2_default_kernel.F90 \
	test/test_real2_choose_kernel_with_api.F90 \
	test/print_available_elpa2_kernels.F90 \
	test/test_real.F90 \
	test/test_real_gen.F90

# test programs
if WITH_OPENMP
build_lib = libelpa_mt.la 
else
build_lib = libelpa.la 
endif

# pkg-config stuff
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = elpa.pc

# test programs

#bindir = $(abs_top_builddir)
bin_PROGRAMS = test_real test_real2 test_real2_default_kernel test_complex test_complex2 test_complex2_default_kernel test_real2_choose_kernel_with_api test_complex2_choose_kernel_with_api print_available_elpa2_kernels


test_real_SOURCES = test/test_real.F90
test_real_LDADD = $(build_lib)

test_real2_SOURCES = test/test_real2.F90
test_real2_LDADD = $(build_lib)

test_real2_default_kernel_SOURCES = test/test_real2_default_kernel.F90
test_real2_default_kernel_LDADD = $(build_lib)

test_real2_choose_kernel_with_api_SOURCES = test/test_real2_choose_kernel_with_api.F90
test_real2_choose_kernel_with_api_LDADD = $(build_lib)

test_complex_SOURCES = test/test_complex.F90
test_complex_LDADD = $(build_lib)

test_complex2_SOURCES = test/test_complex2.F90
test_complex2_LDADD = $(build_lib)

test_complex2_default_kernel_SOURCES = test/test_complex2_default_kernel.F90
test_complex2_default_kernel_LDADD = $(build_lib) 

test_complex2_choose_kernel_with_api_SOURCES = test/test_complex2_choose_kernel_with_api.F90
test_complex2_choose_kernel_with_api_LDADD = $(build_lib)

print_available_elpa2_kernels_SOURCES = test/print_available_elpa2_kernels.F90
print_available_elpa2_kernels_LDADD = $(build_lib)

check_SCRIPTS = test_real.sh test_real2.sh test_real2_default_kernel.sh test_complex.sh test_complex2.sh test_complex2_default_kernel.sh test_real2_choose_kernel_with_api.sh test_complex2_choose_kernel_with_api.sh print_available_elpa2_kernels.sh 
TESTS = $(check_SCRIPTS)

test_real.sh:
	echo "mpiexec -n 2 ./test_real > /dev/null 2>&1" > test_real.sh
	chmod +x test_real.sh	

test_real2.sh:
	echo "mpiexec -n 2 ./test_real2 > /dev/null 2>&1" > test_real2.sh
	chmod +x test_real2.sh	

test_real2_default_kernel.sh:
	echo "mpiexec -n 2 ./test_real2_default_kernel > /dev/null 2>&1" > test_real2_default_kernel.sh
	chmod +x test_real2_default_kernel.sh	

test_real2_choose_kernel_with_api.sh:
	echo "mpiexec -n 2 ./test_real2_choose_kernel_with_api > /dev/null 2>&1" > test_real2_choose_kernel_with_api.sh
	chmod +x test_real2_choose_kernel_with_api.sh	

test_complex.sh:
	echo "mpiexec -n 2 ./test_complex > /dev/null 2>&1" > test_complex.sh
	chmod +x test_complex.sh	

test_complex2.sh:
	echo "mpiexec -n 2 ./test_complex2 > /dev/null 2>&1" > test_complex2.sh
	chmod +x test_complex2.sh	
test_complex2_default_kernel.sh:
	echo "mpiexec -n 2 ./test_complex2_default_kernel > /dev/null 2>&1" > test_complex2_default_kernel.sh
	chmod +x test_complex2_default_kernel.sh

test_complex2_choose_kernel_with_api.sh:
	echo "mpiexec -n 2 ./test_complex2_choose_kernel_with_api > /dev/null 2>&1" > test_complex2_choose_kernel_with_api.sh
	chmod +x test_complex2_choose_kernel_with_api.sh

print_available_elpa2_kernels.sh:
	echo "./print_available_elpa2_kernels" > print_available_elpa2_kernels.sh
	chmod +x print_available_elpa2_kernels.sh


elpa2.i: $(top_srcdir)/src/elpa2.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/elpa2.F90 -o $@

elpa1.i: $(top_srcdir)/src/elpa1.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/elpa1.F90 -o $@

CLEANFILES = test_real.sh test_real2.sh test_real2_default_kernel.sh test_complex.sh test_complex2.sh  test_complex2_default_kernel.sh test_real2_choose_kernel_with_api.sh test_complex2_choose_kernel_with_api.sh print_available_elpa2_kernels

@FORTRAN_MODULE_DEPS@
