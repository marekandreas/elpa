## Process this file with automake to produce Makefile.in

ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS} -I m4

AM_FCFLAGS = $(SCALAPACK_FCFLAGS) @FC_MODINC@modules @FC_MODOUT@modules
AM_LDFLAGS = $(SCALAPACK_LDFLAGS)

# libelpa
lib_LTLIBRARIES = libelpa@SUFFIX@.la
libelpa@SUFFIX@_la_LINK = $(FCLINK) $(AM_LDFLAGS) -version-info $(ELPA_SO_VERSION) -lstdc++

libelpa@SUFFIX@_la_SOURCES = src/elpa_utilities.F90 \
	                     src/elpa1.F90 \
			     src/elpa2_utilities.F90 \
			     src/elpa2.F90 \
			     src/elpa_c_interface.F90 \
                             src/elpa_qr/qr_utils.f90 \
                             src/elpa_qr/elpa_qrkernels.f90 \
                             src/elpa_qr/elpa_pdlarfb.f90 \
                             src/elpa_qr/elpa_pdgeqrf.f90
if HAVE_DETAILED_TIMINGS
  libelpa@SUFFIX@_la_SOURCES += src/timer.F90 \
                                src/ftimings/ftimings.F90 \
                                src/ftimings/ftimings_type.F90 \
                                src/ftimings/ftimings_value.F90 \
                                src/ftimings/highwater_mark.c \
                                src/ftimings/resident_set_size.c \
                                src/ftimings/time.c \
                                src/ftimings/virtual_memory.c \
                                src/ftimings/papi.c
endif

if WITH_REAL_GENERIC_KERNEL
  libelpa@SUFFIX@_la_SOURCES +=  src/elpa2_kernels/elpa2_kernels_real.f90
endif

if WITH_COMPLEX_GENERIC_KERNEL
  libelpa@SUFFIX@_la_SOURCES +=  src/elpa2_kernels/elpa2_kernels_complex.f90
endif

if WITH_REAL_GENERIC_SIMPLE_KERNEL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_simple.f90
endif

if WITH_COMPLEX_GENERIC_SIMPLE_KERNEL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_simple.f90
endif

if WITH_REAL_BGP_KERNEL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgp.f90
endif

if WITH_REAL_BGQ_KERNEL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgq.f90
endif

if WITH_REAL_SSE_KERNEL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64.s
else
if WITH_COMPLEX_SSE_KERNEL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64.s
endif
endif

if WITH_REAL_AVX_BLOCK2_KERNEL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c
endif

if WITH_REAL_AVX_BLOCK4_KERNEL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_4hv.c
endif

if WITH_REAL_AVX_BLOCK6_KERNEL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_6hv.c
endif

if WITH_COMPLEX_AVX_BLOCK1_KERNEL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
endif

if WITH_COMPLEX_AVX_BLOCK2_KERNEL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse-avx_2hv.cpp
endif

#if WITH_AVX_SANDYBRIDGE
#  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse-avx_2hv.c \
#                                           src/elpa2_kernels/elpa2_kernels_complex_sse-avx_1hv.cpp
#endif

# install any .mod files in the include/ dir
elpa_includedir = $(includedir)/elpa@SUFFIX@-@PACKAGE_VERSION@
nobase_elpa_include_HEADERS = $(wildcard modules/*)
nobase_elpa_include_HEADERS += elpa/elpa.h elpa/elpa_kernel_constants.h elpa/elpa_generated.h

# other files to distribute
filesdir = $(docdir)/examples
dist_files_DATA = \
  test/read_real.F90 \
  test/test_complex2.F90 \
  test/test_complex2_default_kernel.F90 \
  test/test_complex2_choose_kernel_with_api.F90 \
  test/test_complex.F90 \
  test/test_real2.F90 \
  test/test_real2_default_kernel.F90 \
  test/test_real2_default_kernel_qr_decomposition.F90 \
  test/test_real2_choose_kernel_with_api.F90 \
  test/test_real.F90 \
  test/test_real_with_c.F90 \
  src/print_available_elpa2_kernels.F90

dist_doc_DATA = README COPYING/COPYING COPYING/gpl.txt COPYING/lgpl.txt

# pkg-config stuff
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = @PKG_CONFIG_FILE@

# programs
bin_PROGRAMS = \
  elpa1_test_real@SUFFIX@ \
  elpa1_test_complex@SUFFIX@ \
  elpa2_test_real@SUFFIX@ \
  elpa2_test_complex@SUFFIX@ \
  elpa2_print_kernels@SUFFIX@

noinst_PROGRAMS = \
  elpa2_test_real_default_kernel@SUFFIX@ \
  elpa2_test_real_default_kernel_qr_decomposition@SUFFIX@ \
  elpa2_test_complex_default_kernel@SUFFIX@ \
  elpa2_test_real_choose_kernel_with_api@SUFFIX@ \
  elpa2_test_complex_choose_kernel_with_api@SUFFIX@ \
  elpa1_test_real_c_version@SUFFIX@ \
  elpa1_test_real_with_c@SUFFIX@


build_lib = libelpa@SUFFIX@.la

if HAVE_REDIRECT
  redirect_sources = test/shared_sources/redir.c test/shared_sources/redirect.F90
else
  redirect_sources =
endif

shared_sources = test/shared_sources/util.F90 test/shared_sources/read_input_parameters.F90  test/shared_sources/check_correctnes.F90 test/shared_sources/setup_mpi.F90 \
		 test/shared_sources/blacs_infrastructure.F90 test/shared_sources/prepare_matrix.F90

elpa1_test_real_c_version@SUFFIX@_SOURCES = test/test_real_c_version.c $(shared_sources) $(redirect_sources)
elpa1_test_real_c_version@SUFFIX@_LDADD = $(build_lib)
elpa1_test_real_c_version@SUFFIX@_LINK = $(LINK) $(FCLIBS)

elpa1_test_real@SUFFIX@_SOURCES = test/test_real.F90 $(shared_sources) $(redirect_sources)
elpa1_test_real@SUFFIX@_LDADD = $(build_lib)

elpa1_test_real_with_c@SUFFIX@_SOURCES = test/test_real_with_c.F90 test/shared_sources/mod_from_c.F90 test/shared_sources/call_elpa1_real.c  $(shared_sources) $(redirect_sources)
elpa1_test_real_with_c@SUFFIX@_LDADD = $(build_lib)

elpa2_test_real@SUFFIX@_SOURCES = test/test_real2.F90 $(shared_sources) $(redirect_sources)

elpa2_test_real@SUFFIX@_LDADD = $(build_lib)

elpa2_test_real_default_kernel@SUFFIX@_SOURCES = test/test_real2_default_kernel.F90 $(shared_sources) $(redirect_sources)

elpa2_test_real_default_kernel@SUFFIX@_LDADD = $(build_lib)

elpa2_test_real_default_kernel_qr_decomposition@SUFFIX@_SOURCES = test/test_real2_default_kernel_qr_decomposition.F90 $(shared_sources) $(redirect_sources)

elpa2_test_real_default_kernel_qr_decomposition@SUFFIX@_LDADD = $(build_lib)

elpa2_test_real_choose_kernel_with_api@SUFFIX@_SOURCES = test/test_real2_choose_kernel_with_api.F90 $(shared_sources) $(redirect_sources)

elpa2_test_real_choose_kernel_with_api@SUFFIX@_LDADD = $(build_lib)

elpa1_test_complex@SUFFIX@_SOURCES = test/test_complex.F90 $(shared_sources) $(redirect_sources)

elpa1_test_complex@SUFFIX@_LDADD = $(build_lib)

elpa2_test_complex@SUFFIX@_SOURCES = test/test_complex2.F90 $(shared_sources) $(redirect_sources)

elpa2_test_complex@SUFFIX@_LDADD = $(build_lib)

elpa2_test_complex_default_kernel@SUFFIX@_SOURCES = test/test_complex2_default_kernel.F90 $(shared_sources) $(redirect_sources)

elpa2_test_complex_default_kernel@SUFFIX@_LDADD = $(build_lib)

elpa2_test_complex_choose_kernel_with_api@SUFFIX@_SOURCES = test/test_complex2_choose_kernel_with_api.F90 $(shared_sources) $(redirect_sources)

elpa2_test_complex_choose_kernel_with_api@SUFFIX@_LDADD = $(build_lib)

elpa2_print_kernels@SUFFIX@_SOURCES = src/print_available_elpa2_kernels.F90 $(shared_sources) $(redirect_sources)

elpa2_print_kernels@SUFFIX@_LDADD = $(build_lib)

check_SCRIPTS = \
  elpa1_test_real.sh \
  elpa1_test_real_with_c.sh \
  elpa1_test_real_c_version.sh \
  elpa2_test_real.sh \
  elpa2_test_real_default_kernel.sh \
  elpa1_test_complex.sh \
  elpa2_test_complex.sh \
  elpa2_test_complex_default_kernel.sh \
  elpa2_test_real_default_kernel_qr_decomposition.sh \
  elpa2_test_real_choose_kernel_with_api.sh \
  elpa2_test_complex_choose_kernel_with_api.sh \
  elpa2_print_kernels@SUFFIX@

TESTS = $(check_SCRIPTS)
elpa1_test_real.sh:
	echo 'mpiexec -n 2 ./elpa1_test_real@SUFFIX@ $$TEST_FLAGS' > elpa1_test_real.sh
	chmod +x elpa1_test_real.sh

elpa1_test_real_with_c.sh:
	echo 'mpiexec -n 2 ./elpa1_test_real_with_c@SUFFIX@ $$TEST_FLAGS' > elpa1_test_real_with_c.sh
	chmod +x elpa1_test_real_with_c.sh

elpa1_test_real_c_version.sh:
	echo 'mpiexec -n 2 ./elpa1_test_real_c_version@SUFFIX@ $$TEST_FLAGS' > elpa1_test_real_c_version.sh
	chmod +x elpa1_test_real_c_version.sh

elpa2_test_real.sh:
	echo 'mpiexec -n 2 ./elpa2_test_real@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real.sh
	chmod +x elpa2_test_real.sh

elpa2_test_real_default_kernel.sh:
	echo 'mpiexec -n 2 ./elpa2_test_real_default_kernel@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real_default_kernel.sh
	chmod +x elpa2_test_real_default_kernel.sh

elpa2_test_real_default_kernel_qr_decomposition.sh:
	echo 'mpiexec -n 2 ./elpa2_test_real_default_kernel_qr_decomposition@SUFFIX@' > elpa2_test_real_default_kernel_qr_decomposition.sh
	chmod +x elpa2_test_real_default_kernel_qr_decomposition.sh

elpa2_test_real_choose_kernel_with_api.sh:
	echo 'mpiexec -n 2 ./elpa2_test_real_choose_kernel_with_api@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real_choose_kernel_with_api.sh
	chmod +x elpa2_test_real_choose_kernel_with_api.sh

elpa1_test_complex.sh:
	echo 'mpiexec -n 2 ./elpa1_test_complex@SUFFIX@ $$TEST_FLAGS' > elpa1_test_complex.sh
	chmod +x elpa1_test_complex.sh

elpa2_test_complex.sh:
	echo 'mpiexec -n 2 ./elpa2_test_complex@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex.sh
	chmod +x elpa2_test_complex.sh

elpa2_test_complex_default_kernel.sh:
	echo 'mpiexec -n 2 ./elpa2_test_complex_default_kernel@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex_default_kernel.sh
	chmod +x elpa2_test_complex_default_kernel.sh

elpa2_test_complex_choose_kernel_with_api.sh:
	echo 'mpiexec -n 2 ./elpa2_test_complex_choose_kernel_with_api@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex_choose_kernel_with_api.sh
	chmod +x elpa2_test_complex_choose_kernel_with_api.sh

elpa2_utilities.i: $(top_srcdir)/src/elpa2_utilities.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/elpa2_utilities.F90 -o $@

elpa2.i: $(top_srcdir)/src/elpa2.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/elpa2.F90 -o $@

elpa1.i: $(top_srcdir)/src/elpa1.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/elpa1.F90 -o $@

CLEANFILES = \
  elpa-generated.h \
  elpa1_test_real.sh \
  elpa1_test_complex.sh \
  elpa2_test_real.sh \
  elpa2_test_real_default_kernel.sh \
  elpa2_test_real_default_kernel_qr_decomposition.sh \
  elpa2_test_complex.sh \
  elpa2_test_complex_default_kernel.sh \
  elpa2_test_real_choose_kernel_with_api.sh \
  elpa2_test_complex_choose_kernel_with_api.sh \
  *.i

clean-local:
	-rm -rf modules/* .fortran_dependencies/*

distclean-local:
	-rm config-f90.h
	-rm -rf ./src/elpa2_kernels/.deps
	-rm -rf ./src/.deps
	-rm -rf ./test/.deps
	-rmdir ./src/elpa2_kernels/
	-rmdir ./src
	-rmdir ./test
	-rmdir ./m4
	-rmdir modules/
	-rmdir .fortran_dependencies/

EXTRA_DIST = fdep/fortran_dependencies.pl fdep/fortran_dependencies.mk

LIBTOOL_DEPS = @LIBTOOL_DEPS@
libtool: $(LIBTOOL_DEPS)
	$(SHELL) ./config.status libtool


@FORTRAN_MODULE_DEPS@

# Fortran module dependencies only work within each target,
# specify that the test programs need a finished library before
# one can compile them

# $1  Object name
define require_elpa_lib
$1: libelpa@SUFFIX@.la

endef
$(foreach p,$(bin_PROGRAMS) $(noinst_PROGRAMS),$(foreach o,$($p_OBJECTS),$(eval $(call require_elpa_lib,$o))))
