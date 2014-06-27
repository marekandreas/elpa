## Process this file with automake to produce Makefile.in

ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS} -I m4

AM_FCFLAGS = $(SCALAPACK_FCFLAGS) @FC_MODINC@modules @FC_MODOUT@modules
AM_LDFLAGS = $(SCALAPACK_LDFLAGS)

# libelpa
if WITH_OPENMP
lib_LTLIBRARIES = libelpa-@PACKAGE_VERSION@_mt.la
libelpa_@PACKAGE_VERSION@_mt_la_LINK = $(FCLINK)
libelpa_@PACKAGE_VERSION@_mt_la_LDFLAGS = $(AM_LDFLAGS) -version-info $(ELPA_SO_VERSION) -lstdc++
else
lib_LTLIBRARIES = libelpa-@PACKAGE_VERSION@.la
libelpa_@PACKAGE_VERSION@_la_LINK = $(FCLINK)
libelpa_@PACKAGE_VERSION@_la_LDFLAGS = $(AM_LDFLAGS) -version-info $(ELPA_SO_VERSION) -lstdc++
endif


##rule to produce fortran config file:
#config_f90.h: ./config.h
#	grep "^#define" ./config.h > $@

if WITH_OPENMP
libelpa_@PACKAGE_VERSION@_mt_la_SOURCES = src/elpa1.F90 src/elpa2.F90
else
libelpa_@PACKAGE_VERSION@_la_SOURCES = src/elpa1.F90 src/elpa2.F90
endif

if WITH_REAL_GENERIC_KERNEL
if WITH_OPENMP
  libelpa_@PACKAGE_VERSION@_mt_la_SOURCES +=  src/elpa2_kernels/elpa2_kernels_real.f90
else
  libelpa_@PACKAGE_VERSION@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real.f90
endif
endif

if WITH_COMPLEX_GENERIC_KERNEL
if WITH_OPENMP
  libelpa_@PACKAGE_VERSION@_mt_la_SOURCES +=  src/elpa2_kernels/elpa2_kernels_complex.f90
else
  libelpa_@PACKAGE_VERSION@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex.f90
endif
endif

if WITH_REAL_GENERIC_SIMPLE_KERNEL
if WITH_OPENMP
  libelpa_@PACKAGE_VERSION@_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_simple.f90
else
  libelpa_@PACKAGE_VERSION@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_simple.f90
endif
endif

if WITH_COMPLEX_GENERIC_SIMPLE_KERNEL
if WITH_OPENMP
  libelpa_@PACKAGE_VERSION@_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_simple.f90
else
  libelpa_@PACKAGE_VERSION@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_simple.f90
endif
endif

if WITH_REAL_BGP_KERNEL
if WITH_OPENMP
  libelpa_@PACKAGE_VERSION@_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgp.f90
else
  libelpa_@PACKAGE_VERSION@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgp.f90
endif
endif

if WITH_REAL_BGQ_KERNEL
if WITH_OPENMP
  libelpa_@PACKAGE_VERSION@_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgq.f90
else
  libelpa_@PACKAGE_VERSION@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgq.f90
endif
endif

if WITH_REAL_SSE_KERNEL
if WITH_OPENMP
  libelpa_@PACKAGE_VERSION@_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64.s
else
  libelpa_@PACKAGE_VERSION@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64.s
endif
else
if WITH_COMPLEX_SSE_KERNEL
if WITH_OPENMP
  libelpa_@PACKAGE_VERSION@_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64.s
else
  libelpa_@PACKAGE_VERSION@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64.s
endif
endif
endif

if WITH_REAL_AVX_BLOCK2_KERNEL
if WITH_OPENMP
  libelpa_@PACKAGE_VERSION@_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c
else
  libelpa_@PACKAGE_VERSION@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c
endif
endif

if WITH_REAL_AVX_BLOCK4_KERNEL
if WITH_OPENMP
  libelpa_@PACKAGE_VERSION@_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_4hv.c
else
 libelpa_@PACKAGE_VERSION@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_4hv.c
endif
endif

if WITH_REAL_AVX_BLOCK6_KERNEL
if WITH_OPENMP
  libelpa_@PACKAGE_VERSION@_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_6hv.c
else
  libelpa_@PACKAGE_VERSION@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_6hv.c
endif
endif

if WITH_COMPLEX_AVX_BLOCK1_KERNEL
if WITH_OPENMP
 libelpa_@PACKAGE_VERSION@_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
else
  libelpa_@PACKAGE_VERSION@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
endif
endif

if WITH_COMPLEX_AVX_BLOCK2_KERNEL
if WITH_OPENMP
 libelpa_@PACKAGE_VERSION@_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse-avx_2hv.cpp
else
  libelpa_@PACKAGE_VERSION@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse-avx_2hv.cpp
endif
endif

#if WITH_AVX_SANDYBRIDGE
#if WITH_OPENMP
#  libelpa_@PACKAGE_VERSION@_mt_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c \
#                                           src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
#else
#  libelpa_@PACKAGE_VERSION@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c \
#                                           src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
#endif
#endif

# install any .mod files in the include/ dir
elpa_includedir = $(includedir)/elpa-@PACKAGE_VERSION@
nobase_elpa_include_HEADERS = $(wildcard modules/*)

# other files to distribute
filesdir = $(datadir)/@PACKAGE@-@PACKAGE_VERSION@/examples
dist_files_DATA = \
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
	src/print_available_elpa2_kernels.F90 \
	test/test_real.F90 \
	test/test_real_gen.F90

dist_doc_DATA = README COPYING/COPYING COPYING/gpl.txt COPYING/lgpl.txt

# test programs
if WITH_OPENMP
build_lib = libelpa-@PACKAGE_VERSION@_mt.la
else
build_lib = libelpa-@PACKAGE_VERSION@.la
endif

# pkg-config stuff
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = @PKG_CONFIG_FILE@

# programs
bin_PROGRAMS = \
	elpa1_test_real \
	elpa1_test_complex \
	elpa2_print_kernels \
	elpa2_test_real \
	elpa2_test_real_default_kernel \
	elpa2_test_complex \
	elpa2_test_complex_default_kernel \
	elpa2_test_real_choose_kernel_with_api \
	elpa2_test_complex_choose_kernel_with_api

elpa1_test_real_SOURCES = test/test_real.F90
elpa1_test_real_LDADD = $(build_lib)

elpa2_test_real_SOURCES = test/test_real2.F90
elpa2_test_real_LDADD = $(build_lib)

elpa2_test_real_default_kernel_SOURCES = test/test_real2_default_kernel.F90
elpa2_test_real_default_kernel_LDADD = $(build_lib)

elpa2_test_real_choose_kernel_with_api_SOURCES = test/test_real2_choose_kernel_with_api.F90
elpa2_test_real_choose_kernel_with_api_LDADD = $(build_lib)

elpa1_test_complex_SOURCES = test/test_complex.F90
elpa1_test_complex_LDADD = $(build_lib)

elpa2_test_complex_SOURCES = test/test_complex2.F90
elpa2_test_complex_LDADD = $(build_lib)

elpa2_test_complex_default_kernel_SOURCES = test/test_complex2_default_kernel.F90
elpa2_test_complex_default_kernel_LDADD = $(build_lib)

elpa2_test_complex_choose_kernel_with_api_SOURCES = test/test_complex2_choose_kernel_with_api.F90
elpa2_test_complex_choose_kernel_with_api_LDADD = $(build_lib)

elpa2_print_kernels_SOURCES = src/print_available_elpa2_kernels.F90
elpa2_print_kernels_LDADD = $(build_lib)

check_SCRIPTS = \
	elpa1_test_real.sh \
	elpa2_test_real.sh \
	elpa2_test_real_default_kernel.sh \
	elpa1_test_complex.sh \
	elpa2_test_complex.sh \
	elpa2_test_complex_default_kernel.sh \
	elpa2_test_real_choose_kernel_with_api.sh \
	elpa2_test_complex_choose_kernel_with_api.sh \
	elpa2_print_kernels

TESTS = $(check_SCRIPTS)

elpa1_test_real.sh:
	echo "mpiexec -n 2 ./elpa1_test_real $$TEST_FLAGS" > elpa1_test_real.sh
	chmod +x elpa1_test_real.sh

elpa2_test_real.sh:
	echo "mpiexec -n 2 ./elpa2_test_real $$TEST_FLAGS" > elpa2_test_real.sh
	chmod +x elpa2_test_real.sh

elpa2_test_real_default_kernel.sh:
	echo "mpiexec -n 2 ./elpa2_test_real_default_kernel $$TEST_FLAGS" > elpa2_test_real_default_kernel.sh
	chmod +x elpa2_test_real_default_kernel.sh

elpa2_test_real_choose_kernel_with_api.sh:
	echo "mpiexec -n 2 ./elpa2_test_real_choose_kernel_with_api $$TEST_FLAGS" > elpa2_test_real_choose_kernel_with_api.sh
	chmod +x elpa2_test_real_choose_kernel_with_api.sh

elpa1_test_complex.sh:
	echo "mpiexec -n 2 ./elpa1_test_complex $$TEST_FLAGS" > elpa1_test_complex.sh
	chmod +x elpa1_test_complex.sh

elpa2_test_complex.sh:
	echo "mpiexec -n 2 ./elpa2_test_complex $$TEST_FLAGS" > elpa2_test_complex.sh
	chmod +x elpa2_test_complex.sh

elpa2_test_complex_default_kernel.sh:
	echo "mpiexec -n 2 ./elpa2_test_complex_default_kernel $$TEST_FLAGS" > elpa2_test_complex_default_kernel.sh
	chmod +x elpa2_test_complex_default_kernel.sh

elpa2_test_complex_choose_kernel_with_api.sh:
	echo "mpiexec -n 2 ./elpa2_test_complex_choose_kernel_with_api $$TEST_FLAGS" > elpa2_test_complex_choose_kernel_with_api.sh
	chmod +x elpa2_test_complex_choose_kernel_with_api.sh

elpa2.i: $(top_srcdir)/src/elpa2.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/elpa2.F90 -o $@

elpa1.i: $(top_srcdir)/src/elpa1.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/elpa1.F90 -o $@

CLEANFILES = \
	elpa1_test_real.sh \
	elpa1_test_complex.sh \
	elpa2_test_real.sh \
	elpa2_test_real_default_kernel.sh \
	elpa2_test_complex.sh \
	elpa2_test_complex_default_kernel.sh \
	elpa2_test_real_choose_kernel_with_api.sh \
	elpa2_test_complex_choose_kernel_with_api.sh \
	elpa2_print_kernels.sh \
	*.i

clean-local:
	-rm -rf modules/* .fortran_dependencies/*

distclean-local:
	-rm config-f90.h
	-rmdir ./src/.deps
	-rmdir ./src
	-rmdir ./test/.deps
	-rmdir ./test
	-rmdir modules/

EXTRA_DIST = fdep/fortran_dependencies.pl fdep/fortran_dependencies.mk

LIBTOOL_DEPS = @LIBTOOL_DEPS@
libtool: $(LIBTOOL_DEPS)
	$(SHELL) ./config.status libtool


@FORTRAN_MODULE_DEPS@

# Fortran module dependencies only work within each target,
# specify that the test programs need a finished library before
# one can compile them

# $1	Object name
define require_elpa_lib
$1: $(build_lib)

endef
$(foreach p,$(bin_PROGRAMS),$(foreach o,$($p_OBJECTS),$(eval $(call require_elpa_lib,$o))))
