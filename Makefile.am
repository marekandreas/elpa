## Process this file with automake to produce Makefile.in

ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS} -I m4

AM_FCFLAGS = $(SCALAPACK_FCFLAGS) @FC_MODINC@modules
AM_LDFLAGS = $(SCALAPACK_LDFLAGS)

# libelpa

# parts with public interface
noinst_LTLIBRARIES = libelpa@SUFFIX@_public.la
libelpa@SUFFIX@_public_la_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@modules @FC_MODINC@private_modules
libelpa@SUFFIX@_public_la_SOURCES = \
        src/elpa1.F90 \
        src/elpa2.F90 \
        src/elpa2_utilities.F90 \
        src/elpa_utilities.F90

# internal parts
noinst_LTLIBRARIES += libelpa@SUFFIX@_private.la
libelpa@SUFFIX@_private_la_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
libelpa@SUFFIX@_private_la_SOURCES = \
        src/mod_precision.f90 \
        src/mod_mpi.F90 \
        src/mod_mpi_stubs.F90 \
        src/mod_pack_unpack_real.F90 \
        src/mod_compute_hh_trafo_real.F90 \
        src/mod_compute_hh_trafo_complex.F90 \
        src/mod_pack_unpack_complex.F90 \
        src/aligned_mem.F90 \
        src/elpa1_compute.F90 \
        src/elpa2_compute.F90 \
        src/elpa2_kernels/mod_fortran_interfaces.F90 \
        src/elpa2_kernels/mod_single_hh_trafo_real.F90 \
        src/elpa_c_interface.F90 \
        src/elpa_qr/qr_utils.F90 \
        src/elpa_qr/elpa_qrkernels.f90 \
        src/elpa_qr/elpa_pdlarfb.F90 \
        src/elpa_qr/elpa_pdgeqrf.F90

EXTRA_libelpa@SUFFIX@_private_la_DEPENDENCIES = \
        src/elpa_reduce_add_vectors.X90 \
        src/elpa_transpose_vectors.X90 \
        src/redist_band.X90

lib_LTLIBRARIES = libelpa@SUFFIX@.la
libelpa@SUFFIX@_la_LINK = $(FCLINK) $(AM_LDFLAGS) -version-info $(ELPA_SO_VERSION) -lstdc++
libelpa@SUFFIX@_la_LIBADD = libelpa@SUFFIX@_public.la libelpa@SUFFIX@_private.la
libelpa@SUFFIX@_la_SOURCES =


if HAVE_DETAILED_TIMINGS
  libelpa@SUFFIX@_private_la_SOURCES += \
        src/timer.F90 \
        src/ftimings/ftimings.F90 \
        src/ftimings/ftimings_type.F90 \
        src/ftimings/ftimings_value.F90 \
        src/ftimings/highwater_mark.c \
        src/ftimings/resident_set_size.c \
        src/ftimings/time.c \
        src/ftimings/virtual_memory.c \
        src/ftimings/papi.c
endif

if !WITH_MPI
  libelpa@SUFFIX@_private_la_SOURCES += src/mod_time_c.F90
if !HAVE_DETAILED_TIMINGS
  libelpa@SUFFIX@_private_la_SOURCES += src/ftimings/time.c
endif
endif

if WITH_REAL_GENERIC_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES +=  src/elpa2_kernels/elpa2_kernels_real.F90
endif

if WITH_COMPLEX_GENERIC_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES +=  src/elpa2_kernels/elpa2_kernels_complex.F90
endif

if WITH_REAL_GENERIC_SIMPLE_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_simple.F90
endif

if WITH_COMPLEX_GENERIC_SIMPLE_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_simple.F90
endif

if WITH_REAL_BGP_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgp.f90
endif

if WITH_REAL_BGQ_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgq.f90
endif

if WITH_REAL_SSE_ASSEMBLY_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64.s
else
if WITH_COMPLEX_SSE_ASSEMBLY_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64.s
endif
endif

if WITH_REAL_SSE_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse_2hv.c
endif

if WITH_REAL_AVX_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_2hv.c
else
if WITH_REAL_AVX2_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_2hv.c
endif
endif

if WITH_REAL_SSE_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse_4hv.c
endif

if WITH_REAL_AVX_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_4hv.c
else
if WITH_REAL_AVX2_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_4hv.c
endif
endif

if WITH_REAL_SSE_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse_6hv.c
endif

if WITH_REAL_AVX_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_6hv.c
else
if WITH_REAL_AVX2_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_6hv.c
endif
endif

if WITH_COMPLEX_SSE_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse_1hv.c
endif

if WITH_COMPLEX_AVX_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_avx-avx2_1hv.c
else
if WITH_COMPLEX_AVX2_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_avx-avx2_1hv.c
endif
endif

if WITH_COMPLEX_SSE_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse_2hv.c
endif

if WITH_COMPLEX_AVX_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_avx-avx2_2hv.c
else
if WITH_COMPLEX_AVX2_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_avx-avx2_2hv.c
endif
endif

include generated_headers.am
BUILT_SOURCES = $(generated_headers)

# install public Fortran modules files in the include/ dir
elpa_includedir = $(includedir)/elpa@SUFFIX@-@PACKAGE_VERSION@
nobase_elpa_include_HEADERS = $(wildcard modules/*)
nobase_elpa_include_HEADERS += elpa/elpa.h elpa/elpa_kernel_constants.h elpa/elpa_generated.h

dist_man_MANS = \
	   man/solve_evp_real.3 \
	   man/solve_evp_real_1stage.3 \
	   man/solve_evp_complex.3 \
	   man/solve_evp_complex_1stage.3 \
	   man/solve_evp_real_2stage.3 \
	   man/solve_evp_complex_2stage.3 \
	   man/get_elpa_row_col_comms.3 \
	   man/get_elpa_communicators.3 \
	   man/elpa2_print_kernels.1

# other files to distribute
filesdir = $(docdir)/examples
dist_files_DATA = \
  test/fortran_test_programs/read_real.F90 \
  test/fortran_test_programs/test_complex2.F90 \
  test/fortran_test_programs/test_complex2_default_kernel.F90 \
  test/fortran_test_programs/test_complex2_choose_kernel_with_api.F90 \
  test/fortran_test_programs/test_complex.F90 \
  test/fortran_test_programs/test_real2.F90 \
  test/fortran_test_programs/test_real2_default_kernel.F90 \
  test/fortran_test_programs/test_real2_default_kernel_qr_decomposition.F90 \
  test/fortran_test_programs/test_real2_choose_kernel_with_api.F90 \
  test/fortran_test_programs/test_real.F90 \
  test/fortran_test_programs/test_real_with_c.F90 \
  src/elpa2_print_kernels.F90

dist_doc_DATA = README.md USERS_GUIDE.md INSTALL.md CONTRIBUTING.md LICENSE Changelog COPYING/COPYING COPYING/gpl.txt COPYING/lgpl.txt

# pkg-config stuff
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = @PKG_CONFIG_FILE@

# programs
bin_PROGRAMS = \
  elpa2_print_kernels@SUFFIX@

noinst_PROGRAMS = \
  elpa1_test_real@SUFFIX@ \
  elpa1_test_complex@SUFFIX@ \
  elpa2_test_real@SUFFIX@ \
  elpa2_test_complex@SUFFIX@ \
  elpa2_test_real_default_kernel@SUFFIX@ \
  elpa2_test_real_default_kernel_qr_decomposition@SUFFIX@ \
  elpa2_test_complex_default_kernel@SUFFIX@ \
  elpa2_test_real_choose_kernel_with_api@SUFFIX@ \
  elpa2_test_complex_choose_kernel_with_api@SUFFIX@ \
  elpa1_test_real_with_c@SUFFIX@
if !WITH_OPENMP
noinst_PROGRAMS += \
  elpa1_test_real_c_version@SUFFIX@ \
  elpa1_test_complex_c_version@SUFFIX@ \
  elpa2_test_real_c_version@SUFFIX@ \
  elpa2_test_complex_c_version@SUFFIX@
endif

build_lib = libelpa@SUFFIX@.la libelpatest@SUFFIX@.la

# library with shared sources for the test files
noinst_LTLIBRARIES += libelpatest@SUFFIX@.la
libelpatest@SUFFIX@_la_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
libelpatest@SUFFIX@_la_SOURCES = \
	test/shared_sources/util.F90 \
	test/shared_sources/read_input_parameters.F90  \
	test/shared_sources/check_correctnes.F90 \
	test/shared_sources/setup_mpi.F90 \
	test/shared_sources/blacs_infrastructure.F90 \
	test/shared_sources/prepare_matrix.F90 \
	test/shared_sources/mod_output_types.F90 \
	test/shared_sources/mod_from_c.F90 \
	test/shared_sources/call_elpa1.c

if HAVE_REDIRECT
libelpatest@SUFFIX@_la_SOURCES += \
	test/shared_sources/redir.c \
	test/shared_sources/redirect.F90
endif

if !WITH_OPENMP
elpa1_test_real_c_version@SUFFIX@_SOURCES = test/c_test_programs/elpa1_test_real_c_version.c
elpa1_test_real_c_version@SUFFIX@_LDADD = $(build_lib)
elpa1_test_real_c_version@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
elpa1_test_real_c_version@SUFFIX@_LINK = $(LINK) $(FCLIBS)
EXTRA_elpa1_test_real_c_version@SUFFIX@_DEPENDENCIES = test/fortran_test_programs/elpa_test_programs_print_headers.X90

elpa1_test_complex_c_version@SUFFIX@_SOURCES = test/c_test_programs/elpa1_test_complex_c_version.c
elpa1_test_complex_c_version@SUFFIX@_LDADD = $(build_lib)
elpa1_test_complex_c_version@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
elpa1_test_complex_c_version@SUFFIX@_LINK = $(LINK) $(FCLIBS)
EXTRA_elpa1_test_complex_c_version@SUFFIX@_DEPENDENCIES = test/fortran_test_programs/elpa_test_programs_print_headers.X90

elpa2_test_real_c_version@SUFFIX@_SOURCES = test/c_test_programs/elpa2_test_real_c_version.c
elpa2_test_real_c_version@SUFFIX@_LDADD = $(build_lib)
elpa2_test_real_c_version@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
elpa2_test_real_c_version@SUFFIX@_LINK = $(LINK) $(FCLIBS)
EXTRA_elpa2_test_real_c_version@SUFFIX@_DEPENDENCIES = test/fortran_test_programs/elpa_test_programs_print_headers.X90

elpa2_test_complex_c_version@SUFFIX@_SOURCES = test/c_test_programs/elpa2_test_complex_c_version.c
elpa2_test_complex_c_version@SUFFIX@_LDADD = $(build_lib)
elpa2_test_complex_c_version@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
elpa2_test_complex_c_version@SUFFIX@_LINK = $(LINK) $(FCLIBS)
EXTRA_elpa2_test_complex_c_version@SUFFIX@_DEPENDENCIES = test/fortran_test_programs/elpa_test_programs_print_headers.X90
endif

elpa1_test_real@SUFFIX@_SOURCES = test/fortran_test_programs/test_real.F90
elpa1_test_real@SUFFIX@_LDADD = $(build_lib)
elpa1_test_real@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_test_real@SUFFIX@_DEPENDENCIES = test/fortran_test_programs/elpa_test_programs_print_headers.X90

elpa1_test_real_with_c@SUFFIX@_SOURCES = test/fortran_test_programs/test_real_with_c.F90
elpa1_test_real_with_c@SUFFIX@_LDADD = $(build_lib)
elpa1_test_real_with_c@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_test_real_with_c@SUFFIX@_DEPENDENCIES = test/fortran_test_programs/elpa_test_programs_print_headers.X90

elpa2_test_real@SUFFIX@_SOURCES = test/fortran_test_programs/test_real2.F90
elpa2_test_real@SUFFIX@_LDADD = $(build_lib)
elpa2_test_real@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_real@SUFFIX@_DEPENDENCIES = test/fortran_test_programs/elpa_test_programs_print_headers.X90

elpa2_test_real_default_kernel@SUFFIX@_SOURCES = test/fortran_test_programs/test_real2_default_kernel.F90
elpa2_test_real_default_kernel@SUFFIX@_LDADD = $(build_lib)
elpa2_test_real_default_kernel@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_real_default_kernel@SUFFIX@_DEPENDENCIES = test/fortran_test_programs/elpa_test_programs_print_headers.X90


elpa2_test_real_default_kernel_qr_decomposition@SUFFIX@_SOURCES = test/fortran_test_programs/test_real2_default_kernel_qr_decomposition.F90
elpa2_test_real_default_kernel_qr_decomposition@SUFFIX@_LDADD = $(build_lib)
elpa2_test_real_default_kernel_qr_decomposition@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_real_default_kernel_qr_decomposition@SUFFIX@_DEPENDENCIES = test/fortran_test_programs/elpa_test_programs_print_headers.X90

elpa2_test_real_choose_kernel_with_api@SUFFIX@_SOURCES = test/fortran_test_programs/test_real2_choose_kernel_with_api.F90
elpa2_test_real_choose_kernel_with_api@SUFFIX@_LDADD = $(build_lib)
elpa2_test_real_choose_kernel_with_api@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_real_choose_kernel_with_api@SUFFIX@_DEPENDENCIES = test/fortran_test_programs/elpa_test_programs_print_headers.X90

elpa1_test_complex@SUFFIX@_SOURCES = test/fortran_test_programs/test_complex.F90
elpa1_test_complex@SUFFIX@_LDADD = $(build_lib)
elpa1_test_complex@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_test_complex@SUFFIX@_DEPENDENCIES = test/fortran_test_programs/elpa_test_programs_print_headers.X90


elpa2_test_complex@SUFFIX@_SOURCES = test/fortran_test_programs/test_complex2.F90
elpa2_test_complex@SUFFIX@_LDADD = $(build_lib)
elpa2_test_complex@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_complex@SUFFIX@_DEPENDENCIES = test/fortran_test_programs/elpa_test_programs_print_headers.X90

elpa2_test_complex_default_kernel@SUFFIX@_SOURCES = test/fortran_test_programs/test_complex2_default_kernel.F90
elpa2_test_complex_default_kernel@SUFFIX@_LDADD = $(build_lib)
elpa2_test_complex_default_kernel@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_complex_default_kernel@SUFFIX@_DEPENDENCIES = test/fortran_test_programs/elpa_test_programs_print_headers.X90

elpa2_test_complex_choose_kernel_with_api@SUFFIX@_SOURCES = test/fortran_test_programs/test_complex2_choose_kernel_with_api.F90
elpa2_test_complex_choose_kernel_with_api@SUFFIX@_LDADD = $(build_lib)
elpa2_test_complex_choose_kernel_with_api@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_complex_choose_kernel_with_api@SUFFIX@_DEPENDENCIES = test/fortran_test_programs/elpa_test_programs_print_headers.X90

elpa2_print_kernels@SUFFIX@_SOURCES = src/elpa2_print_kernels.F90
elpa2_print_kernels@SUFFIX@_LDADD = $(build_lib)
elpa2_print_kernels@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules

check_SCRIPTS = \
  elpa1_test_real@SUFFIX@.sh \
  elpa1_test_real_with_c@SUFFIX@.sh \
  elpa2_test_real@SUFFIX@.sh \
  elpa2_test_real_default_kernel@SUFFIX@.sh \
  elpa1_test_complex@SUFFIX@.sh \
  elpa2_test_complex@SUFFIX@.sh \
  elpa2_test_complex_default_kernel@SUFFIX@.sh \
  elpa2_test_real_default_kernel_qr_decomposition@SUFFIX@.sh \
  elpa2_test_real_choose_kernel_with_api@SUFFIX@.sh \
  elpa2_test_complex_choose_kernel_with_api@SUFFIX@.sh \
  elpa2_print_kernels@SUFFIX@

if !WITH_OPENMP
check_SCRIPTS += \
  elpa1_test_real_c_version@SUFFIX@.sh \
  elpa1_test_complex_c_version@SUFFIX@.sh \
  elpa2_test_real_c_version@SUFFIX@.sh \
  elpa2_test_complex_c_version@SUFFIX@.sh
endif


# test scripts
if WITH_MPI
  wrapper=mpiexec -n 2
else
  wrapper=
endif
TESTS = $(check_SCRIPTS)
%.sh: %
	@echo $(wrapper) ./$^ '$$TEST_FLAGS' > $@
	@chmod +x $@

# Preprocessed files (just used for manual inspection)
elpa2_utilities.i: $(top_srcdir)/src/elpa2_utilities.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/elpa2_utilities.F90 -o $@

elpa2.i: $(top_srcdir)/src/elpa2.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/elpa2.F90 -o $@

elpa1.i: $(top_srcdir)/src/elpa1.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/elpa1.F90 -o $@

elpa2_kernels_real.i: $(top_srcdir)/src/elpa2_kernels/elpa2_kernels_real.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/elpa2_kernels/elpa2_kernels_real.F90 -o $@

mod_compute_hh_trafo_real.i: $(top_srcdir)/src/mod_compute_hh_trafo_real.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/mod_compute_hh_trafo_real.F90 -o $@

mod_compute_hh_trafo_complex.i: $(top_srcdir)/src/mod_compute_hh_trafo_complex.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/mod_compute_hh_trafo_complex.F90 -o $@


include doxygen.am

CLEANFILES = \
  elpa-generated.h \
  elpa1_test* \
  elpa2_test*\
  *.i

clean-local:
	-rm -rf modules/* .fortran_dependencies/*
	-rm -rf $(generated_headers)

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

EXTRA_DIST = \
  fdep/fortran_dependencies.pl \
  fdep/fortran_dependencies.mk \
  test/fortran_test_programs/elpa_test_programs_print_headers.X90 \
  src/elpa_reduce_add_vectors.X90 \
  src/elpa_transpose_vectors.X90 \
  src/redist_band.X90 \
  elpa.spec

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
