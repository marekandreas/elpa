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
	src/elpa1_auxiliary.F90 \
        src/elpa2_utilities.F90 \
        src/elpa_utilities.F90

# internal parts
noinst_LTLIBRARIES += libelpa@SUFFIX@_private.la
libelpa@SUFFIX@_private_la_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
libelpa@SUFFIX@_private_la_SOURCES = \
        src/mod_precision.F90 \
        src/mod_mpi.F90 \
        src/mod_mpi_stubs.F90 \
        src/mod_pack_unpack_real.F90 \
        src/mod_compute_hh_trafo_real.F90 \
        src/mod_compute_hh_trafo_complex.F90 \
        src/mod_pack_unpack_complex.F90 \
        src/aligned_mem.F90 \
        src/elpa1_compute_private.F90 \
        src/elpa2_compute.F90 \
        src/elpa2_kernels/mod_fortran_interfaces.F90 \
        src/elpa2_kernels/mod_single_hh_trafo_real.F90 \
        src/elpa_c_interface.F90 \
        src/check_for_gpu.F90 \
        src/mod_cuda.F90 \
        src/interface_c_kernel.F90 \
        src/mod_pack_unpack_real_gpu.F90 \
        src/elpa_qr/qr_utils.F90 \
        src/elpa_qr/elpa_qrkernels.F90 \
        src/elpa_qr/elpa_pdlarfb.F90 \
        src/elpa_qr/elpa_pdgeqrf.F90

EXTRA_libelpa@SUFFIX@_private_la_DEPENDENCIES = \
        src/elpa_reduce_add_vectors.X90 \
        src/elpa_transpose_vectors.X90 \
        src/elpa1_compute_complex_template.X90 \
        src/elpa1_compute_real_template.X90 \
        src/elpa2_compute_real_template.X90 \
        src/elpa2_compute_complex_template.X90 \
        src/redist_band.X90

lib_LTLIBRARIES = libelpa@SUFFIX@.la
libelpa@SUFFIX@_la_LINK = $(FCLINK) $(AM_LDFLAGS) -version-info $(ELPA_SO_VERSION)
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

if WITH_GPU_VERSION
  libelpa@SUFFIX@_private_la_SOURCES +=  src/cudaFunctions.cu src/cuUtils.cu src/ev_tridi_band_gpu_c_v2.cu
  EXTRA_libelpa@SUFFIX@_private_la_DEPENDENCIES +=   src/ev_tridi_band_gpu_c_v2_complex_template.Xcu src/ev_tridi_band_gpu_c_v2_real_template.Xcu
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
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64_double_precision.s
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64_single_precision.s
endif

else
if WITH_COMPLEX_SSE_ASSEMBLY_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64_double_precision.s
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64_single_precision.s
endif
endif
endif

if WITH_REAL_SSE_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse_2hv_single_precision.c
endif
endif

if WITH_REAL_AVX_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_2hv_single_precision.c
endif
else
if WITH_REAL_AVX2_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_2hv_single_precision.c
endif
endif
endif

if WITH_REAL_SSE_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse_4hv_single_precision.c
endif
endif

if WITH_REAL_AVX_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_4hv_single_precision.c
endif
else
if WITH_REAL_AVX2_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_4hv_single_precision.c
endif
endif
endif

if WITH_REAL_SSE_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_sse_6hv_single_precision.c
endif
endif

if WITH_REAL_AVX_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_6hv_single_precision.c
endif
else
if WITH_REAL_AVX2_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_avx-avx2_6hv_single_precision.c
endif
endif
endif

if WITH_COMPLEX_SSE_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse_1hv_single_precision.c
endif
endif

if WITH_COMPLEX_AVX_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_avx-avx2_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_avx-avx2_1hv_single_precision.c
endif
else
if WITH_COMPLEX_AVX2_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_avx-avx2_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_avx-avx2_1hv_single_precision.c
endif
endif
endif

if WITH_COMPLEX_SSE_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_sse_2hv_single_precision.c
endif
endif

if WITH_COMPLEX_AVX_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_avx-avx2_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_avx-avx2_2hv_single_precision.c
endif
else
if WITH_COMPLEX_AVX2_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_avx-avx2_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_avx-avx2_2hv_single_precision.c
endif
endif
endif

.cu.lo:
	NVCC="$(NVCC)" libtool --mode=compile --tag=CC $(top_srcdir)/nvcc_wrap $(NVCCFLAGS) $(LDFLAGS) -I$(top_builddir)/ -I$(top_srcdir)/  -c $< -o $@

include generated_headers.am
BUILT_SOURCES = $(generated_headers)

# install public Fortran modules files in the include/ dir
elpa_includedir = $(includedir)/elpa@SUFFIX@-@PACKAGE_VERSION@
nobase_elpa_include_HEADERS = $(wildcard modules/*)
nobase_elpa_include_HEADERS += elpa/elpa.h elpa/elpa_kernel_constants.h elpa/elpa_generated.h

dist_man_MANS = \
           man/solve_evp_real.3 \
	   man/solve_evp_real_1stage_double.3 \
	   man/solve_evp_complex.3 \
	   man/solve_evp_complex_1stage_double.3 \
	   man/solve_evp_real_2stage_double.3 \
	   man/solve_evp_complex_2stage_double.3 \
	   man/get_elpa_row_col_comms.3 \
	   man/get_elpa_communicators.3 \
	   man/elpa2_print_kernels.1

if WANT_SINGLE_PRECISION_REAL

dist_man_MANS += man/solve_evp_real_1stage_single.3 \
	    man/solve_evp_real_2stage_single.3
endif

if WANT_SINGLE_PRECISION_COMPLEX

dist_man_MANS += man/solve_evp_complex_1stage_single.3 \
	    man/solve_evp_complex_2stage_single.3
endif

#needed?

# other files to distribute
filesdir = $(docdir)/examples
dist_files_DATA = \
  test/Fortran/read_real.F90 \
  test/Fortran/test_complex2.F90 \
  test/Fortran/test_complex2_default.F90 \
  test/Fortran/test_complex2_api.F90 \
  test/Fortran/test_complex.F90 \
  test/Fortran/test_real2.F90 \
  test/Fortran/test_real2_default.F90 \
  test/Fortran/test_real2_qr.F90 \
  test/Fortran/test_real2_api.F90 \
  test/Fortran/test_real.F90 \
  test/Fortran/test_real_with_c.F90 \
  test/Fortran/test_toeplitz.F90 \
  test/Fortran/test_transpose_multiply_real.F90 \
  test/Fortran/test_transpose_multiply_complex.F90 \
  test/Fortran/test_cholesky_real.F90 \
  test/Fortran/test_invert_trm_real.F90 \
  test/Fortran/test_cholesky_complex.F90 \
  test/Fortran/test_invert_trm_complex.F90 \
  src/elpa2_print_kernels.F90

#end needed

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
  elpa2_test_real_default@SUFFIX@ \
  elpa2_test_real_qr@SUFFIX@ \
  elpa2_test_real_api@SUFFIX@ \
  elpa2_test_complex@SUFFIX@ \
  elpa2_test_complex_default@SUFFIX@ \
  elpa2_test_complex_api@SUFFIX@ \
  elpa1_real_toeplitz@SUFFIX@ \
  elpa1_real_transpose_multiply@SUFFIX@ \
  elpa1_complex_transpose_multiply@SUFFIX@ \
  elpa1_real_cholesky@SUFFIX@ \
  elpa1_real_invert_trm@SUFFIX@ \
  elpa1_complex_cholesky@SUFFIX@ \
  elpa1_complex_invert_trm@SUFFIX@ \
  elpa1_test_real_with_c@SUFFIX@ \
  elpa1_test_real_c_version@SUFFIX@ \
  elpa1_test_complex_c_version@SUFFIX@ \
  elpa2_test_real_c_version@SUFFIX@ \
  elpa2_test_complex_c_version@SUFFIX@

if WANT_SINGLE_PRECISION_COMPLEX
noinst_PROGRAMS += \
  elpa1_test_complex_single_precision@SUFFIX@ \
  elpa2_test_complex_single_precision@SUFFIX@ \
  elpa2_test_complex_default_single_precision@SUFFIX@ \
  elpa1_complex_transpose_multiply_single_precision@SUFFIX@ \
  elpa1_complex_cholesky_single_precision@SUFFIX@ \
  elpa1_complex_invert_trm_single_precision@SUFFIX@ \
  elpa2_test_complex_api_single_precision@SUFFIX@
endif

if WANT_SINGLE_PRECISION_REAL
noinst_PROGRAMS += \
  elpa1_test_real_single_precision@SUFFIX@ \
  elpa2_test_real_single_precision@SUFFIX@ \
  elpa2_test_real_default_single_precision@SUFFIX@ \
  elpa2_test_real_qr_single_precision@SUFFIX@ \
  elpa2_test_real_api_single_precision@SUFFIX@ \
  elpa1_real_transpose_multiply_single_precision@SUFFIX@ \
  elpa1_real_cholesky_single_precision@SUFFIX@ \
  elpa1_real_invert_trm_single_precision@SUFFIX@ \
  elpa1_real_toeplitz_single_precision@SUFFIX@
endif

if WITH_GPU_VERSION
noinst_PROGRAMS += \
  elpa2_test_complex_gpu@SUFFIX@ \
  elpa2_test_real_gpu@SUFFIX@

if WANT_SINGLE_PRECISION_REAL
noinst_PROGRAMS += \
  elpa2_test_real_gpu_single_precision@SUFFIX@
endif

if WANT_SINGLE_PRECISION_COMPLEX
noinst_PROGRAMS += \
  elpa2_test_complex_gpu_single_precision@SUFFIX@
endif

endif

build_lib = libelpatest@SUFFIX@.la libelpa@SUFFIX@.la

# library with shared sources for the test files
noinst_LTLIBRARIES += libelpatest@SUFFIX@.la
libelpatest@SUFFIX@_la_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
libelpatest@SUFFIX@_la_SOURCES = \
	test/shared/util.F90 \
	test/shared/read_input_parameters.F90  \
	test/shared/check_correctnes.F90 \
	test/shared/setup_mpi.F90 \
	test/shared/blacs_infrastructure.F90 \
	test/shared/prepare_matrix.F90 \
	test/shared/mod_output_types.F90 \
	test/shared/mod_from_c.F90 \
	test/shared/call_elpa1.c

if HAVE_REDIRECT
libelpatest@SUFFIX@_la_SOURCES += \
	test/shared/redir.c \
	test/shared/redirect.F90
endif

elpa1_test_real_c_version@SUFFIX@_SOURCES = test/C/elpa1_test_real_c_version.c
elpa1_test_real_c_version@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
elpa1_test_real_c_version@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_test_real_c_version@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_test_real_with_c@SUFFIX@_SOURCES = test/Fortran/test_real_with_c.F90
elpa1_test_real_with_c@SUFFIX@_LDADD = $(build_lib)
elpa1_test_real_with_c@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_test_real_with_c@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_test_complex_c_version@SUFFIX@_SOURCES = test/C/elpa1_test_complex_c_version.c
elpa1_test_complex_c_version@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
elpa1_test_complex_c_version@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_test_complex_c_version@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_test_real_c_version@SUFFIX@_SOURCES = test/C/elpa2_test_real_c_version.c
elpa2_test_real_c_version@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
elpa2_test_real_c_version@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_real_c_version@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_test_complex_c_version@SUFFIX@_SOURCES = test/C/elpa2_test_complex_c_version.c
elpa2_test_complex_c_version@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
elpa2_test_complex_c_version@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_complex_c_version@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_test_real@SUFFIX@_SOURCES = test/Fortran/test_real.F90
elpa1_test_real@SUFFIX@_LDADD = $(build_lib)
elpa1_test_real@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_test_real@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_real_toeplitz@SUFFIX@_SOURCES = test/Fortran/test_toeplitz.F90
elpa1_real_toeplitz@SUFFIX@_LDADD = $(build_lib)
elpa1_real_toeplitz@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_real_toeplitz@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_real_transpose_multiply@SUFFIX@_SOURCES = test/Fortran/test_transpose_multiply_real.F90
elpa1_real_transpose_multiply@SUFFIX@_LDADD = $(build_lib)
elpa1_real_transpose_multiply@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_real_transpose_multiply@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_complex_transpose_multiply@SUFFIX@_SOURCES = test/Fortran/test_transpose_multiply_complex.F90
elpa1_complex_transpose_multiply@SUFFIX@_LDADD = $(build_lib)
elpa1_complex_transpose_multiply@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_complex_transpose_multiply@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_real_cholesky@SUFFIX@_SOURCES = test/Fortran/test_cholesky_real.F90
elpa1_real_cholesky@SUFFIX@_LDADD = $(build_lib)
elpa1_real_cholesky@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_real_cholesky@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_real_invert_trm@SUFFIX@_SOURCES = test/Fortran/test_invert_trm_real.F90
elpa1_real_invert_trm@SUFFIX@_LDADD = $(build_lib)
elpa1_real_invert_trm@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_real_invert_trm@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_complex_cholesky@SUFFIX@_SOURCES = test/Fortran/test_cholesky_complex.F90
elpa1_complex_cholesky@SUFFIX@_LDADD = $(build_lib)
elpa1_complex_cholesky@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_complex_cholesky@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_complex_invert_trm@SUFFIX@_SOURCES = test/Fortran/test_invert_trm_complex.F90
elpa1_complex_invert_trm@SUFFIX@_LDADD = $(build_lib)
elpa1_complex_invert_trm@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_complex_invert_trm@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_test_real@SUFFIX@_SOURCES = test/Fortran/test_real2.F90
elpa2_test_real@SUFFIX@_LDADD = $(build_lib)
elpa2_test_real@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_real@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_test_real_default@SUFFIX@_SOURCES = test/Fortran/test_real2_default.F90
elpa2_test_real_default@SUFFIX@_LDADD = $(build_lib)
elpa2_test_real_default@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_real_default@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_test_real_qr@SUFFIX@_SOURCES = test/Fortran/test_real2_qr.F90
elpa2_test_real_qr@SUFFIX@_LDADD = $(build_lib)
elpa2_test_real_qr@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_real_qr@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_test_real_api@SUFFIX@_SOURCES = test/Fortran/test_real2_api.F90
elpa2_test_real_api@SUFFIX@_LDADD = $(build_lib)
elpa2_test_real_api@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_real_api@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_test_complex@SUFFIX@_SOURCES = test/Fortran/test_complex.F90
elpa1_test_complex@SUFFIX@_LDADD = $(build_lib)
elpa1_test_complex@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_test_complex@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_test_complex@SUFFIX@_SOURCES = test/Fortran/test_complex2.F90
elpa2_test_complex@SUFFIX@_LDADD = $(build_lib)
elpa2_test_complex@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_complex@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_test_complex_default@SUFFIX@_SOURCES = test/Fortran/test_complex2_default.F90
elpa2_test_complex_default@SUFFIX@_LDADD = $(build_lib)
elpa2_test_complex_default@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_complex_default@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_test_complex_api@SUFFIX@_SOURCES = test/Fortran/test_complex2_api.F90
elpa2_test_complex_api@SUFFIX@_LDADD = $(build_lib)
elpa2_test_complex_api@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_complex_api@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_print_kernels@SUFFIX@_SOURCES = src/elpa2_print_kernels.F90
elpa2_print_kernels@SUFFIX@_LDADD = $(build_lib)
elpa2_print_kernels@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules

if WANT_SINGLE_PRECISION_REAL
elpa1_test_real_single_precision@SUFFIX@_SOURCES = test/Fortran/test_real_single.F90
elpa1_test_real_single_precision@SUFFIX@_LDADD = $(build_lib)
elpa1_test_real_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_test_real_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_test_real_single_precision@SUFFIX@_SOURCES = test/Fortran/test_real2_single.F90
elpa2_test_real_single_precision@SUFFIX@_LDFLAGS = -static
elpa2_test_real_single_precision@SUFFIX@_LDADD = $(build_lib)
elpa2_test_real_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_real_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_real_toeplitz_single_precision@SUFFIX@_SOURCES = test/Fortran/test_toeplitz_single.F90
elpa1_real_toeplitz_single_precision@SUFFIX@_LDADD = $(build_lib)
elpa1_real_toeplitz_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_real_toeplitz_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_real_transpose_multiply_single_precision@SUFFIX@_SOURCES = test/Fortran/test_transpose_multiply_real_single.F90
elpa1_real_transpose_multiply_single_precision@SUFFIX@_LDADD = $(build_lib)
elpa1_real_transpose_multiply_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_real_transpose_multiply_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_real_cholesky_single_precision@SUFFIX@_SOURCES = test/Fortran/test_cholesky_real_single.F90
elpa1_real_cholesky_single_precision@SUFFIX@_LDADD = $(build_lib)
elpa1_real_cholesky_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_real_cholesky_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_real_invert_trm_single_precision@SUFFIX@_SOURCES = test/Fortran/test_invert_trm_real_single.F90
elpa1_real_invert_trm_single_precision@SUFFIX@_LDADD = $(build_lib)
elpa1_real_invert_trm_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_real_invert_trm_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_test_real_default_single_precision@SUFFIX@_SOURCES = test/Fortran/test_real2_default.F90
elpa2_test_real_default_single_precision@SUFFIX@_LDADD = $(build_lib)
#elpa2_test_real_default_single_precision@SUFFIX@_LDFLAGS = -static
elpa2_test_real_default_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_real_default_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_test_real_qr_single_precision@SUFFIX@_SOURCES = \
	        test/Fortran/test_real2_qr_single.F90
elpa2_test_real_qr_single_precision@SUFFIX@_LDADD = $(build_lib)
elpa2_test_real_qr_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_real_qr_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_test_real_api_single_precision@SUFFIX@_SOURCES = test/Fortran/test_real2_api_single.F90
elpa2_test_real_api_single_precision@SUFFIX@_LDADD = $(build_lib)
elpa2_test_real_api_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_real_api_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90
endif

if WANT_SINGLE_PRECISION_COMPLEX
elpa1_test_complex_single_precision@SUFFIX@_SOURCES = test/Fortran/test_complex_single.F90
elpa1_test_complex_single_precision@SUFFIX@_LDADD = $(build_lib)
elpa1_test_complex_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_test_complex_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_test_complex_single_precision@SUFFIX@_SOURCES = test/Fortran/test_complex2_single.F90
elpa2_test_complex_single_precision@SUFFIX@_LDADD = $(build_lib)
elpa2_test_complex_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_complex_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_test_complex_default_single_precision@SUFFIX@_SOURCES = test/Fortran/test_complex2_default_single.F90 $(shared_sources) $(redirect_sources)
elpa2_test_complex_default_single_precision@SUFFIX@_LDADD = $(build_lib)
#elpa2_test_complex_default_single_precision@SUFFIX@_LDFLAGS = -static
elpa2_test_complex_default_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_complex_default_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_complex_transpose_multiply_single_precision@SUFFIX@_SOURCES = test/Fortran/test_transpose_multiply_complex_single.F90
elpa1_complex_transpose_multiply_single_precision@SUFFIX@_LDADD = $(build_lib)
elpa1_complex_transpose_multiply_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_complex_transpose_multiply_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_complex_cholesky_single_precision@SUFFIX@_SOURCES = test/Fortran/test_cholesky_complex_single.F90
elpa1_complex_cholesky_single_precision@SUFFIX@_LDADD = $(build_lib)
elpa1_complex_cholesky_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_complex_cholesky_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa1_complex_invert_trm_single_precision@SUFFIX@_SOURCES = test/Fortran/test_invert_trm_complex_single.F90
elpa1_complex_invert_trm_single_precision@SUFFIX@_LDADD = $(build_lib)
elpa1_complex_invert_trm_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa1_complex_invert_trm_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_test_complex_api_single_precision@SUFFIX@_SOURCES = test/Fortran/test_complex2_api_single.F90
elpa2_test_complex_api_single_precision@SUFFIX@_LDADD = $(build_lib)
elpa2_test_complex_api_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_complex_api_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90
endif

if WITH_GPU_VERSION
elpa2_test_real_gpu@SUFFIX@_SOURCES = test/Fortran/test_real2_gpu.F90
elpa2_test_real_gpu@SUFFIX@_LDADD = $(build_lib)
elpa2_test_real_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_real_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa2_test_complex_gpu@SUFFIX@_SOURCES = test/Fortran/test_complex2_gpu.F90
elpa2_test_complex_gpu@SUFFIX@_LDADD = $(build_lib)
elpa2_test_complex_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_complex_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

if WANT_SINGLE_PRECISION_REAL
elpa2_test_real_gpu_single_precision@SUFFIX@_SOURCES = test/Fortran/test_real2_gpu_single.F90
elpa2_test_real_gpu_single_precision@SUFFIX@_LDADD = $(build_lib)
elpa2_test_real_gpu_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_real_gpu_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90
endif

if WANT_SINGLE_PRECISION_COMPLEX
elpa2_test_complex_gpu_single_precision@SUFFIX@_SOURCES = test/Fortran/test_complex2_gpu_single.F90
elpa2_test_complex_gpu_single_precision@SUFFIX@_LDADD = $(build_lib)
elpa2_test_complex_gpu_single_precision@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) @FC_MODOUT@private_modules @FC_MODINC@private_modules
EXTRA_elpa2_test_complex_gpu_single_precision@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90
endif

endif
#  elpa2_test_real_default_kernel_qr_decomposition@SUFFIX@.sh

check_SCRIPTS = \
  elpa1_test_real@SUFFIX@.sh \
  elpa2_test_real@SUFFIX@.sh \
  elpa2_test_real_default@SUFFIX@.sh \
  elpa1_test_complex@SUFFIX@.sh \
  elpa2_test_complex@SUFFIX@.sh \
  elpa2_test_real_qr@SUFFIX@.sh \
  elpa2_test_complex_default@SUFFIX@.sh \
  elpa2_test_real_api@SUFFIX@.sh \
  elpa2_test_complex_api@SUFFIX@.sh \
  elpa1_real_toeplitz@SUFFIX@.sh \
  elpa1_test_real_with_c@SUFFIX@.sh \
  elpa1_real_cholesky@SUFFIX@.sh \
  elpa1_real_invert_trm@SUFFIX@.sh \
  elpa1_real_transpose_multiply@SUFFIX@.sh \
  elpa1_complex_cholesky@SUFFIX@.sh \
  elpa1_complex_invert_trm@SUFFIX@.sh \
  elpa1_complex_transpose_multiply@SUFFIX@.sh \
  elpa2_print_kernels@SUFFIX@ \
  elpa1_test_real_c_version@SUFFIX@.sh \
  elpa1_test_complex_c_version@SUFFIX@.sh \
  elpa2_test_real_c_version@SUFFIX@.sh \
  elpa2_test_complex_c_version@SUFFIX@.sh

if WANT_SINGLE_PRECISION_REAL
check_SCRIPTS += \
  elpa1_test_real_single_precision@SUFFIX@.sh \
  elpa2_test_real_single_precision@SUFFIX@.sh \
  elpa2_test_real_default_single_precision@SUFFIX@.sh \
  elpa2_test_real_qr_single_precision@SUFFIX@.sh \
  elpa1_real_cholesky_single_precision@SUFFIX@.sh \
  elpa1_real_invert_trm_single_precision@SUFFIX@.sh \
  elpa1_real_transpose_multiply_single_precision@SUFFIX@.sh \
  elpa2_test_real_api_single_precision@SUFFIX@.sh
endif

if WANT_SINGLE_PRECISION_COMPLEX
check_SCRIPTS += \
  elpa1_test_complex_single_precision@SUFFIX@.sh \
  elpa2_test_complex_single_precision@SUFFIX@.sh \
  elpa2_test_complex_default_single_precision@SUFFIX@.sh \
  elpa1_complex_cholesky_single_precision@SUFFIX@.sh \
  elpa1_complex_invert_trm_single_precision@SUFFIX@.sh \
  elpa1_complex_transpose_multiply_single_precision@SUFFIX@.sh \
  elpa2_test_complex_api_single_precision@SUFFIX@.sh
endif

if WITH_GPU_VERSION
check_SCRIPTS += \
  elpa2_test_real_gpu@SUFFIX@.sh \
  elpa2_test_complex_gpu@SUFFIX@.sh
if WANT_SINGLE_PRECISION_REAL
check_SCRIPTS += \
  elpa2_test_real_gpu_single_precision@SUFFIX@.sh
endif
if WANT_SINGLE_PRECISION_COMPLEX
check_SCRIPTS += \
  elpa2_test_complex_gpu_single_precision@SUFFIX@.sh
endif

endif

# test scripts
MPIEXEC ?= $(MPI_BINARY) -n 2
if WITH_MPI
  wrapper=$(MPIEXEC)
else
  wrapper=
endif
TESTS = $(check_SCRIPTS)
%.sh: %
	@echo $(wrapper) ./$^ '$$TEST_FLAGS' > $@
	@chmod +x $@

# Preprocessed files (just used for manual inspection)
mod_precision.i: $(top_srcdir)/src/mod_precision.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -I$(top_srcdir)/ -c $(top_srcdir)/src/mod_precision.F90 -o $@

elpa2_utilities.i: $(top_srcdir)/src/elpa2_utilities.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -I$(top_srcdir)/ -c $(top_srcdir)/src/elpa2_utilities.F90 -o $@

elpa2.i: $(top_srcdir)/src/elpa2.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -I$(top_srcdir)/ -c $(top_srcdir)/src/elpa2.F90 -o $@

elpa1.i: $(top_srcdir)/src/elpa1.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -I$(top_srcdir)/ -c $(top_srcdir)/src/elpa1.F90 -o $@

elpa1_compute.i: $(top_srcdir)/src/elpa1_compute.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -I$(top_srcdir)/ -c $(top_srcdir)/src/elpa1_compute.F90 -o $@

elpa2_compute.i: $(top_srcdir)/src/elpa2_compute.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -I$(top_srcdir)/ -c $(top_srcdir)/src/elpa2_compute.F90 -o $@

elpa2_kernels_real.i: $(top_srcdir)/src/elpa2_kernels/elpa2_kernels_real.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/elpa2_kernels/elpa2_kernels_real.F90 -o $@

mod_compute_hh_trafo_real.i: $(top_srcdir)/src/mod_compute_hh_trafo_real.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/mod_compute_hh_trafo_real.F90 -o $@

mod_compute_hh_trafo_complex.i: $(top_srcdir)/src/mod_compute_hh_trafo_complex.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/mod_compute_hh_trafo_complex.F90 -o $@

test_real.i: $(top_srcdir)/test/Fortran/test_real1.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/test/Fortran/test_real1.F90 -o $@

blacs_infrastructure.i: $(top_srcdir)/test/shared_sources/blacs_infrastructure.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/test/shared_sources/blacs_infrastructure.F90 -o $@

check_correctnes.i: $(top_srcdir)/test/shared_sources/check_correctnes.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/test/shared_sources/check_correctnes.F90 -o $@

prepare_matrix.i: $(top_srcdir)/test/shared_sources/prepare_matrix.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/test/shared_sources/prepare_matrix.F90 -o $@

read_input_parameters.i: $(top_srcdir)/test/shared_sources/read_input_parameters.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/test/shared_sources/read_input_parameters.F90 -o $@

setup_mpi.i: $(top_srcdir)/test/shared_sources/setup_mpi.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/test/shared_sources/setup_mpi.F90 -o $@

cuUtils.i: $(top_srcdir)/src/cuUtils.cu
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/cuUtils.cu -o $@

include doxygen.am

CLEANFILES = \
  elpa-generated.h \
  elpa1_test* \
  elpa2_test*\
  elpa2_real* \
  elpa1_real* \
  *.sh \
  *.i

clean-local:
	-rm -rf modules/* private_modules/* .fortran_dependencies/*
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
  test/Fortran/elpa_print_headers.X90 \
  src/elpa_reduce_add_vectors.X90 \
  src/elpa_transpose_vectors.X90 \
  src/elpa1_compute_real_template.X90 \
  src/elpa1_compute_complex_template.X90 \
  src/elpa2_compute_real_template.X90 \
  src/elpa2_compute_complex_template.X90 \
  src/redist_band.X90 \
  src/elpa_qr/elpa_qrkernels.X90 \
  src/ev_tridi_band_gpu_c_v2_complex_template.Xcu \
  src/ev_tridi_band_gpu_c_v2_real_template.Xcu \
  src/cuUtils_complex_template.Xcu \
  src/cuUtils_real_template.Xcu \
  nvcc_wrap \
  test_project/Makefile.am \
  test_project/autogen.sh \
  test_project/configure.ac \
  test_project/fdep \
  test_project/src/test_real.F90 \
  test_project/m4 \
  elpa.spec

## Rules to re-generated the headers
#elpa/elpa_generated.h: $(top_srcdir)/src/elpa_c_interface.F90
#	grep -h "^ *!c>" $^ | sed 's/^ *!c>//;' > $@ || { rm $@; exit 1; }
#
#test/shared_sources/generated.h: $(wildcard $(top_srcdir)/test/shared_sources/*.F90)
#	grep -h "^ *!c>" $^ | sed 's/^ *!c>//;' > $@ || { rm $@; exit 1; }

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
