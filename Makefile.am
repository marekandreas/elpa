## Process this file with automake to produce Makefile.in

ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS} -I m4

AM_FCFLAGS = $(SCALAPACK_FCFLAGS) $(FC_MODINC)modules
AM_LDFLAGS = $(SCALAPACK_LDFLAGS)

FCLD = @ACTUAL_FC@

# libelpa
lib_LTLIBRARIES = libelpa@SUFFIX@.la
libelpa@SUFFIX@_la_LINK = $(FCLINK) $(AM_LDFLAGS) -version-info $(ELPA_SO_VERSION)
libelpa@SUFFIX@_la_LIBADD = libelpa@SUFFIX@_public.la libelpa@SUFFIX@_private.la
libelpa@SUFFIX@_la_SOURCES =

# parts with public interface
noinst_LTLIBRARIES = libelpa@SUFFIX@_public.la
libelpa@SUFFIX@_public_la_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)modules $(FC_MODINC)private_modules
libelpa@SUFFIX@_public_la_SOURCES = \
  src/elpa1/elpa1_auxiliary.F90 \
  src/elpa.F90 \
  src/elpa_api.F90 \
  src/elpa_constants.F90 \
  src/general/elpa_utilities.F90

if ENABLE_LEGACY
libelpa@SUFFIX@_public_la_SOURCES += \
  src/elpa_driver/legacy_interface/elpa.F90 \
  src/elpa1/legacy_interface/elpa1.F90 \
  src/elpa2/legacy_interface/elpa2.F90 \
  src/elpa2/legacy_interface/elpa2_utilities.F90 \
  src/elpa1/legacy_interface/elpa1_auxiliary.F90
endif

# internal parts
noinst_LTLIBRARIES += libelpa@SUFFIX@_private.la
libelpa@SUFFIX@_private_la_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
libelpa@SUFFIX@_private_la_SOURCES = \
  src/elpa_impl.F90 \
  src/elpa_abstract_impl.F90 \
  src/helpers/mod_precision.F90 \
  src/helpers/mod_mpi.F90 \
  src/helpers/mod_mpi_stubs.F90 \
  src/elpa_generated_fortran_interfaces.F90 \
  src/elpa2/mod_redist_band.F90 \
  src/elpa2/mod_pack_unpack_cpu.F90 \
  src/elpa2/mod_compute_hh_trafo.F90 \
  src/helpers/aligned_mem.F90 \
  src/elpa1/elpa1_compute_private.F90 \
  src/elpa2/elpa2_determine_workload.F90 \
  src/elpa2/elpa2_compute.F90 \
  src/elpa2/kernels/mod_single_hh_trafo_real.F90 \
  src/GPU/check_for_gpu.F90 \
  src/GPU/mod_cuda.F90 \
  src/elpa2/GPU/interface_c_kernel.F90 \
  src/elpa2/mod_pack_unpack_gpu.F90 \
  src/elpa2/qr/qr_utils.F90 \
  src/elpa2/qr/elpa_qrkernels.F90 \
  src/elpa2/qr/elpa_pdlarfb.F90 \
  src/elpa2/qr/elpa_pdgeqrf.F90 \
  src/elpa1/elpa1.F90 \
  src/elpa2/elpa2.F90 \
  src/elpa_c_interface.c \
  src/elpa_index.c

if ENABLE_LEGACY
libelpa@SUFFIX@_private_la_SOURCES += \
  src/elpa_driver/legacy_interface/elpa_driver_c_interface.F90 \
  src/elpa1/legacy_interface/elpa_1stage_c_interface.F90 \
  src/elpa2/legacy_interface/elpa_2stage_c_interface.F90
endif

EXTRA_libelpa@SUFFIX@_private_la_DEPENDENCIES = \
  src/elpa1/elpa_reduce_add_vectors.X90 \
  src/elpa1/elpa_transpose_vectors.X90 \
  src/elpa1/elpa1_compute_template.X90 \
  src/elpa2/elpa2_compute_real_template.X90 \
  src/elpa2/elpa2_compute_complex_template.X90 \
  src/elpa1/elpa1_template.X90 \
  src/elpa2/elpa2_template.X90 \
  src/elpa2/qr/qr_utils_template.X90 \
  src/elpa2/qr/elpa_pdlarfb_template.X90 \
  src/elpa2/qr/elpa_pdgeqrf_template.X90 \
  src/elpa2/elpa2_bandred_template.X90 \
  src/elpa2/elpa2_symm_matrix_allreduce_real_template.X90 \
  src/elpa2/elpa2_trans_ev_band_to_full_template.X90 \
  src/elpa2/elpa2_tridiag_band_template.X90 \
  src/elpa2/elpa2_trans_ev_tridi_to_band_template.X90 \
  src/elpa2/elpa2_herm_matrix_allreduce_complex_template.X90 \
  src/elpa2/kernels/elpa2_kernels_real_template.X90 \
  src/elpa2/kernels/elpa2_kernels_complex_template.X90 \
  src/elpa2/kernels/elpa2_kernels_simple_template.X90 \
  src/elpa2/pack_unpack_cpu.X90 \
  src/elpa2/pack_unpack_gpu.X90 \
  src/elpa2/compute_hh_trafo.X90 \
  src/elpa2/redist_band.X90 \
  src/general/sanity.X90 \
  src/elpa1/elpa_cholesky_template.X90 \
  src/elpa1/elpa_invert_trm.X90 \
  src/elpa1/elpa_multiply_a_b.X90 \
  src/elpa1/elpa_solve_tridi_impl_public.X90 \
  src/general/precision_macros.h

if ENABLE_LEGACY
EXTRA_libelpa@SUFFIX@_private_la_DEPENDENCIES += \
  src/elpa1/legacy_interface/elpa1_template.X90 \
  src/elpa2/legacy_interface/elpa2_template.X90 \
  src/elpa1/legacy_interface/elpa1_c_interface_template.X90 \
  src/elpa2/legacy_interface/elpa2_c_interface_template.X90 \
  src/elpa_driver/legacy_interface/elpa_driver_c_interface_template.X90 \
  src/elpa1/legacy_interface/elpa_solve_tridi_c_interface_template.X90 \
  src/elpa1/legacy_interface/elpa_mult_at_b_c_interface_template.X90 \
  src/elpa1/legacy_interface/elpa_mult_ah_b_c_interface_template.X90 \
  src/elpa1/legacy_interface/elpa_invert_trm_c_interface_template.X90 \
  src/elpa1/legacy_interface/elpa_cholesky_c_interface_template.X90 \
  src/elpa1/legacy_interface/elpa_cholesky_template.X90 \
  src/elpa1/legacy_interface/elpa_invert_trm.X90 \
  src/elpa1/legacy_interface/elpa_multiply_a_b.X90 \
  src/elpa1/legacy_interface/elpa_solve_tridi.X90
endif

if HAVE_DETAILED_TIMINGS
libelpa@SUFFIX@_private_la_SOURCES += \
  src/ftimings/ftimings.F90 \
  src/ftimings/ftimings_type.F90 \
  src/ftimings/ftimings_value.F90 \
  src/ftimings/highwater_mark.c \
  src/ftimings/resident_set_size.c \
  src/ftimings/time.c \
  src/ftimings/virtual_memory.c \
  src/ftimings/papi.c

if ENABLE_LEGACY
libelpa@SUFFIX@_private_la_SOURCES += \
  src/helpers/timer.F90
endif

else
libelpa@SUFFIX@_private_la_SOURCES += \
  src/helpers/timer_dummy.F90
endif

if WITH_GPU_VERSION
  libelpa@SUFFIX@_private_la_SOURCES +=  src/GPU/cudaFunctions.cu src/GPU/cuUtils.cu src/elpa2/GPU/ev_tridi_band_gpu_c_v2.cu
  EXTRA_libelpa@SUFFIX@_private_la_DEPENDENCIES +=   src/elpa2/GPU/ev_tridi_band_gpu_c_v2_complex_template.Xcu src/elpa2/GPU/ev_tridi_band_gpu_c_v2_real_template.Xcu
endif

if !WITH_MPI
  libelpa@SUFFIX@_private_la_SOURCES += src/helpers/mod_time_c.F90
if !HAVE_DETAILED_TIMINGS
  libelpa@SUFFIX@_private_la_SOURCES += src/ftimings/time.c
endif
endif

if WITH_REAL_GENERIC_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES +=  src/elpa2/kernels/elpa2_kernels_real.F90
endif

if WITH_COMPLEX_GENERIC_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES +=  src/elpa2/kernels/elpa2_kernels_complex.F90
endif

if WITH_REAL_GENERIC_SIMPLE_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_simple.F90
endif

if WITH_COMPLEX_GENERIC_SIMPLE_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_complex_simple.F90
endif

if WITH_REAL_BGP_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_bgp.f90
endif

if WITH_REAL_BGQ_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_bgq.f90
endif

if WITH_REAL_SSE_ASSEMBLY_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_asm_x86_64_double_precision.s
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_asm_x86_64_single_precision.s
endif

else
if WITH_COMPLEX_SSE_ASSEMBLY_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_asm_x86_64_double_precision.s
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_asm_x86_64_single_precision.s
endif
endif
endif

if WITH_REAL_SSE_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_sse_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_sse_2hv_single_precision.c
endif
endif

if WITH_REAL_AVX_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx-avx2_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx-avx2_2hv_single_precision.c
endif
else
if WITH_REAL_AVX2_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx-avx2_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx-avx2_2hv_single_precision.c
endif
endif
endif

if WITH_REAL_AVX512_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx512_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx512_2hv_single_precision.c
endif
endif


if WITH_REAL_SSE_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_sse_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_sse_4hv_single_precision.c
endif
endif

if WITH_REAL_AVX_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx-avx2_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx-avx2_4hv_single_precision.c
endif
else
if WITH_REAL_AVX2_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx-avx2_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx-avx2_4hv_single_precision.c
endif
endif
endif

if WITH_REAL_AVX512_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx512_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx512_4hv_single_precision.c
endif
endif



if WITH_REAL_SSE_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_sse_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_sse_6hv_single_precision.c
endif
endif

if WITH_REAL_AVX_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx-avx2_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx-avx2_6hv_single_precision.c
endif
else
if WITH_REAL_AVX2_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx-avx2_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx-avx2_6hv_single_precision.c
endif
endif
endif

if WITH_REAL_AVX512_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx512_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_real_avx512_6hv_single_precision.c
endif
endif


if WITH_COMPLEX_SSE_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_complex_sse_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_complex_sse_1hv_single_precision.c
endif
endif

if WITH_COMPLEX_AVX_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_complex_avx-avx2_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_complex_avx-avx2_1hv_single_precision.c
endif
else
if WITH_COMPLEX_AVX2_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_complex_avx-avx2_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_complex_avx-avx2_1hv_single_precision.c
endif
endif
endif


if WITH_COMPLEX_AVX512_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_complex_avx512_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_complex_avx512_1hv_single_precision.c
endif
endif

if WITH_COMPLEX_SSE_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_complex_sse_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_complex_sse_2hv_single_precision.c
endif
endif

if WITH_COMPLEX_AVX_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_complex_avx-avx2_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_complex_avx-avx2_2hv_single_precision.c
endif
else
if WITH_COMPLEX_AVX2_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_complex_avx-avx2_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_complex_avx-avx2_2hv_single_precision.c
endif
endif
endif

if WITH_COMPLEX_AVX512_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_complex_avx512_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/elpa2_kernels_complex_avx512_2hv_single_precision.c
endif
endif

# Cuda files
.cu.lo:
	NVCC="$(NVCC)" libtool --mode=compile --tag=CC $(top_srcdir)/nvcc_wrap $(NVCCFLAGS) $(LDFLAGS) -I$(top_builddir)/ -I$(top_srcdir)/  -c $< -o $@

# Assembly files
LTCCASCOMPILE = $(LIBTOOL) $(AM_V_lt) --tag=CC $(AM_LIBTOOLFLAGS) \
	$(LIBTOOLFLAGS) --mode=compile $(CCAS) $(AM_CCASFLAGS) \
	$(CCASFLAGS)

include generated_headers.am
BUILT_SOURCES = $(generated_headers)

# install public headers and Fortran modules files in the include/ dir
elpa_includedir = $(includedir)/elpa@SUFFIX@-@PACKAGE_VERSION@
nobase_elpa_include_HEADERS = $(wildcard modules/*) $(wildcard elpa/*)

dist_man_MANS = \
  man/elpa2_print_kernels.1

if ENABLE_LEGACY
dist_man_MANS += \
  man/solve_evp_real.3 \
  man/solve_evp_real_1stage_double.3 \
  man/solve_evp_complex.3 \
  man/solve_evp_complex_1stage_double.3 \
  man/solve_evp_real_2stage_double.3 \
  man/solve_evp_complex_2stage_double.3 \
  man/elpa_solve_evp_real_1stage_double.3 \
  man/elpa_solve_evp_complex_1stage_double.3 \
  man/elpa_solve_evp_real_2stage_double.3 \
  man/elpa_solve_evp_complex_2stage_double.3 \
  man/get_elpa_row_col_comms.3 \
  man/get_elpa_communicators.3 \
  man/elpa_mult_at_b_real_double.3 \
  man/elpa_mult_at_b_real_single.3 \
  man/elpa_mult_ah_b_complex_double.3 \
  man/elpa_mult_ah_b_complex_single.3 \
  man/elpa_invert_trm_real_double.3 \
  man/elpa_invert_trm_real_single.3 \
  man/elpa_invert_trm_complex_double.3 \
  man/elpa_invert_trm_complex_single.3 \
  man/elpa_solve_evp_real_double.3 \
  man/elpa_solve_evp_real_single.3 \
  man/elpa_solve_evp_complex_double.3 \
  man/elpa_solve_evp_complex_single.3 \
  man/elpa_tests.1
endif


if WANT_SINGLE_PRECISION_REAL
if ENABLE_LEGACY
dist_man_MANS += \
  man/solve_evp_real_1stage_single.3 \
  man/elpa_solve_evp_real_1stage_single.3 \
  man/elpa_solve_evp_real_2stage_single.3 \
  man/solve_evp_real_2stage_single.3
endif
endif

if WANT_SINGLE_PRECISION_COMPLEX
if ENABLE_LEGACY
dist_man_MANS += \
  man/solve_evp_complex_1stage_single.3 \
  man/elpa_solve_evp_complex_1stage_single.3 \
  man/elpa_solve_evp_complex_2stage_single.3 \
  man/solve_evp_complex_2stage_single.3
endif
endif

dist_doc_DATA = README.md USERS_GUIDE.md INSTALL.md CONTRIBUTING.md LICENSE Changelog COPYING/COPYING COPYING/gpl.txt COPYING/lgpl.txt

# pkg-config stuff
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = @PKG_CONFIG_FILE@

# programs
bin_PROGRAMS = \
  elpa2_print_kernels@SUFFIX@

if ENABLE_LEGACY
bin_PROGRAMS += \
  elpa_tests@SUFFIX@
endif

noinst_PROGRAMS = \
  test_real_double_1stage \
  test_real_double_2stage \
  test_complex_double_1stage \
  test_complex_double_2stage \
  test_c_real_double_1stage \
  test_c_real_double_2stage \
  test_c_complex_double_1stage \
  test_c_complex_double_2stage \
  double_instance@SUFFIX@ \
  real_2stage_banded@SUFFIX@ \
  complex_2stage_banded@SUFFIX@ \
  real_2stage@SUFFIX@ \
  complex_2stage@SUFFIX@ \
  real_1stage@SUFFIX@ \
  complex_1stage@SUFFIX@

if WANT_SINGLE_PRECISION_REAL
noinst_PROGRAMS += \
  test_single_real_1stage \
  test_single_real_2stage \
  test_c_single_real_1stage \
  test_c_single_real_2stage \
  single_real_2stage_banded@SUFFIX@ \
  single_real_2stage@SUFFIX@ \
  single_real_1stage@SUFFIX@
endif

if WANT_SINGLE_PRECISION_COMPLEX
noinst_PROGRAMS += \
  test_single_complex_1stage \
  test_single_complex_2stage \
  test_c_single_complex_1stage \
  test_c_single_complex_2stage \
  single_complex_2stage_banded@SUFFIX@ \
  single_complex_2stage@SUFFIX@ \
  single_complex_1stage@SUFFIX@
endif

if WITH_GPU_VERSION
noinst_PROGRAMS += \
  complex_1stage_gpu@SUFFIX@ \
  complex_2stage_gpu@SUFFIX@ \
  real_1stage_gpu@SUFFIX@ \
  real_2stage_gpu@SUFFIX@

if WANT_SINGLE_PRECISION_REAL
noinst_PROGRAMS += \
  single_real_1stage_gpu@SUFFIX@ \
  single_real_2stage_gpu@SUFFIX@
endif

if WANT_SINGLE_PRECISION_COMPLEX
noinst_PROGRAMS += \
  single_complex_1stage_gpu@SUFFIX@ \
  single_complex_2stage_gpu@SUFFIX@
endif
endif


if ENABLE_LEGACY
noinst_PROGRAMS += \
  legacy_real_1stage@SUFFIX@ \
  legacy_complex_1stage@SUFFIX@ \
  legacy_real_2stage@SUFFIX@ \
  legacy_real_2stage_default@SUFFIX@ \
  legacy_real_2stage_qr@SUFFIX@ \
  legacy_real_2stage_api@SUFFIX@ \
  legacy_complex_2stage@SUFFIX@ \
  legacy_complex_2stage_default@SUFFIX@ \
  legacy_complex_2stage_api@SUFFIX@ \
  legacy_real_driver@SUFFIX@ \
  legacy_complex_driver@SUFFIX@ \
  legacy_real_toeplitz@SUFFIX@ \
  legacy_real_transpose_multiply@SUFFIX@ \
  legacy_complex_transpose_multiply@SUFFIX@ \
  legacy_real_cholesky@SUFFIX@ \
  legacy_real_invert_trm@SUFFIX@ \
  legacy_complex_cholesky@SUFFIX@ \
  legacy_complex_invert_trm@SUFFIX@ \
  legacy_real_1stage_with_c@SUFFIX@ \
  legacy_real_1stage_c_version@SUFFIX@ \
  legacy_complex_1stage_c_version@SUFFIX@ \
  legacy_real_2stage_c_version@SUFFIX@ \
  legacy_complex_2stage_c_version@SUFFIX@ \
  legacy_real_driver_c_version@SUFFIX@ \
  legacy_complex_driver_c_version@SUFFIX@

if WANT_SINGLE_PRECISION_COMPLEX
noinst_PROGRAMS += \
  legacy_single_complex_1stage@SUFFIX@ \
  legacy_single_complex_2stage@SUFFIX@ \
  legacy_single_complex_2stage_default@SUFFIX@ \
  legacy_single_complex_transpose_multiply@SUFFIX@ \
  legacy_single_complex_driver@SUFFIX@ \
  legacy_single_complex_cholesky@SUFFIX@ \
  legacy_single_complex_invert_trm@SUFFIX@ \
  legacy_single_complex_2stage_api@SUFFIX@ \
  legacy_single_complex_driver_c_version@SUFFIX@
endif

if WANT_SINGLE_PRECISION_REAL
noinst_PROGRAMS += \
  legacy_single_real_1stage@SUFFIX@ \
  legacy_single_real_2stage@SUFFIX@ \
  legacy_single_real_2stage_default@SUFFIX@ \
  legacy_single_real_2stage_qr@SUFFIX@ \
  legacy_single_real_2stage_api@SUFFIX@ \
  legacy_single_real_driver@SUFFIX@ \
  legacy_single_real_transpose_multiply@SUFFIX@ \
  legacy_single_real_cholesky@SUFFIX@ \
  legacy_single_real_invert_trm@SUFFIX@ \
  legacy_single_real_driver_c_version@SUFFIX@ \
  legacy_single_real_toeplitz@SUFFIX@
endif

if WITH_GPU_VERSION
noinst_PROGRAMS += \
  legacy_complex_1stage_gpu@SUFFIX@ \
  legacy_real_1stage_gpu@SUFFIX@ \
  legacy_complex_2stage_gpu@SUFFIX@ \
  legacy_real_2stage_gpu@SUFFIX@

if WANT_SINGLE_PRECISION_REAL
noinst_PROGRAMS += \
  legacy_single_real_1stage_gpu@SUFFIX@ \
  legacy_single_real_2stage_gpu@SUFFIX@
endif

if WANT_SINGLE_PRECISION_COMPLEX
noinst_PROGRAMS += \
  legacy_single_complex_1stage_gpu@SUFFIX@ \
  legacy_single_complex_2stage_gpu@SUFFIX@
endif
endif
endif

build_lib = libelpatest@SUFFIX@.la libelpa@SUFFIX@.la

# library with shared sources for the test files
noinst_LTLIBRARIES += libelpatest@SUFFIX@.la
libelpatest@SUFFIX@_la_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
libelpatest@SUFFIX@_la_SOURCES = \
  test/shared/util.F90 \
  test/shared/read_input_parameters.F90 \
  test/shared/check_correctness.F90 \
  test/shared/check_correctness_template.X90 \
  test/shared/setup_mpi.F90 \
  test/shared/blacs_infrastructure.F90 \
  test/shared/prepare_matrix.F90 \
  test/shared/mod_assert.F90 \
  test/shared/prepare_matrix_template.X90 \
  test/shared/mod_output_types.F90 \
  test/shared/mod_from_c.F90 \
  test/shared/call_elpa1.c

if HAVE_REDIRECT
libelpatest@SUFFIX@_la_SOURCES += \
  test/shared/redir.c \
  test/shared/redirect.F90
endif


elpa2_print_kernels@SUFFIX@_SOURCES = src/elpa2/elpa2_print_kernels.F90
elpa2_print_kernels@SUFFIX@_LDADD = $(build_lib)
elpa2_print_kernels@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules

test_real_double_1stage_SOURCES = test/Fortran/test.F90
test_real_double_1stage_LDADD = $(build_lib) $(FCLIBS)
test_real_double_1stage_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules \
        -DTEST_REAL \
        -DTEST_DOUBLE \
        -DTEST_SOLVER_1STAGE \
        -DTEST_GPU=0

test_real_double_2stage_SOURCES = test/Fortran/test.F90
test_real_double_2stage_LDADD = $(build_lib) $(FCLIBS)
test_real_double_2stage_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules \
        -DTEST_REAL \
        -DTEST_DOUBLE \
        -DTEST_SOLVER_2STAGE \
        -DTEST_GPU=0

test_complex_double_1stage_SOURCES = test/Fortran/test.F90
test_complex_double_1stage_LDADD = $(build_lib) $(FCLIBS)
test_complex_double_1stage_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules \
        -DTEST_COMPLEX \
        -DTEST_DOUBLE \
        -DTEST_SOLVER_1STAGE \
        -DTEST_GPU=0

test_complex_double_2stage_SOURCES = test/Fortran/test.F90
test_complex_double_2stage_LDADD = $(build_lib) $(FCLIBS)
test_complex_double_2stage_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules \
        -DTEST_COMPLEX \
        -DTEST_DOUBLE \
        -DTEST_SOLVER_2STAGE \
        -DTEST_GPU=0

if WANT_SINGLE_PRECISION_REAL
test_single_real_1stage_SOURCES = test/Fortran/test.F90
test_single_real_1stage_LDADD = $(build_lib) $(FCLIBS)
test_single_real_1stage_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules \
        -DTEST_REAL \
        -DTEST_SINGLE \
        -DTEST_SOLVER_1STAGE \
        -DTEST_GPU=0

test_single_real_2stage_SOURCES = test/Fortran/test.F90
test_single_real_2stage_LDADD = $(build_lib) $(FCLIBS)
test_single_real_2stage_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules \
        -DTEST_REAL \
        -DTEST_SINGLE \
        -DTEST_SOLVER_2STAGE \
        -DTEST_GPU=0
endif

if WANT_SINGLE_PRECISION_COMPLEX
test_single_complex_1stage_SOURCES = test/Fortran/test.F90
test_single_complex_1stage_LDADD = $(build_lib) $(FCLIBS)
test_single_complex_1stage_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules \
        -DTEST_COMPLEX \
        -DTEST_SINGLE \
        -DTEST_SOLVER_1STAGE \
        -DTEST_GPU=0

test_single_complex_2stage_SOURCES = test/Fortran/test.F90
test_single_complex_2stage_LDADD = $(build_lib) $(FCLIBS)
test_single_complex_2stage_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules \
        -DTEST_COMPLEX \
        -DTEST_SINGLE \
        -DTEST_SOLVER_2STAGE \
        -DTEST_GPU=0
endif

test_c_real_double_1stage_SOURCES = test/C/test.c
test_c_real_double_1stage_LDADD = $(build_lib) $(FCLIBS)
test_c_real_double_1stage_CFLAGS = $(AM_CFLAGS) \
        -DTEST_REAL \
        -DTEST_DOUBLE \
        -DTEST_SOLVER_1STAGE \
        -DTEST_GPU=0

test_c_real_double_2stage_SOURCES = test/C/test.c
test_c_real_double_2stage_LDADD = $(build_lib) $(FCLIBS)
test_c_real_double_2stage_CFLAGS = $(AM_CFLAGS) \
        -DTEST_REAL \
        -DTEST_DOUBLE \
        -DTEST_SOLVER_2STAGE \
        -DTEST_GPU=0

test_c_complex_double_1stage_SOURCES = test/C/test.c
test_c_complex_double_1stage_LDADD = $(build_lib) $(FCLIBS)
test_c_complex_double_1stage_CFLAGS = $(AM_CFLAGS) \
        -DTEST_COMPLEX \
        -DTEST_DOUBLE \
        -DTEST_SOLVER_1STAGE \
        -DTEST_GPU=0

test_c_complex_double_2stage_SOURCES = test/C/test.c
test_c_complex_double_2stage_LDADD = $(build_lib) $(FCLIBS)
test_c_complex_double_2stage_CFLAGS = $(AM_CFLAGS) \
        -DTEST_COMPLEX \
        -DTEST_DOUBLE \
        -DTEST_SOLVER_2STAGE \
        -DTEST_GPU=0

if WANT_SINGLE_PRECISION_REAL
test_c_single_real_1stage_SOURCES = test/C/test.c
test_c_single_real_1stage_LDADD = $(build_lib) $(FCLIBS)
test_c_single_real_1stage_CFLAGS = $(AM_CFLAGS) \
        -DTEST_REAL \
        -DTEST_SINGLE \
        -DTEST_SOLVER_1STAGE \
        -DTEST_GPU=0

test_c_single_real_2stage_SOURCES = test/C/test.c
test_c_single_real_2stage_LDADD = $(build_lib) $(FCLIBS)
test_c_single_real_2stage_CFLAGS = $(AM_CFLAGS) \
        -DTEST_REAL \
        -DTEST_SINGLE \
        -DTEST_SOLVER_2STAGE \
        -DTEST_GPU=0
endif

if WANT_SINGLE_PRECISION_COMPLEX
test_c_single_complex_1stage_SOURCES = test/C/test.c
test_c_single_complex_1stage_LDADD = $(build_lib) $(FCLIBS)
test_c_single_complex_1stage_CFLAGS = $(AM_CFLAGS) \
        -DTEST_COMPLEX \
        -DTEST_SINGLE \
        -DTEST_SOLVER_1STAGE \
        -DTEST_GPU=0

test_c_single_complex_2stage_SOURCES = test/C/test.c
test_c_single_complex_2stage_LDADD = $(build_lib) $(FCLIBS)
test_c_single_complex_2stage_CFLAGS = $(AM_CFLAGS) \
        -DTEST_COMPLEX \
        -DTEST_SINGLE \
        -DTEST_SOLVER_2STAGE \
        -DTEST_GPU=0
endif

double_instance@SUFFIX@_SOURCES = test/Fortran/elpa2/double_instance.F90
double_instance@SUFFIX@_LDADD = $(build_lib)
double_instance@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_double_instance@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

real_2stage_banded@SUFFIX@_SOURCES = test/Fortran/elpa2/real_2stage_banded.F90
real_2stage_banded@SUFFIX@_LDADD = $(build_lib)
real_2stage_banded@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_real_2stage_banded@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

complex_2stage_banded@SUFFIX@_SOURCES = test/Fortran/elpa2/complex_2stage_banded.F90
complex_2stage_banded@SUFFIX@_LDADD = $(build_lib)
complex_2stage_banded@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_complex_2stage_banded@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

real_2stage@SUFFIX@_SOURCES = test/Fortran/elpa2/real_2stage.F90
real_2stage@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
real_2stage@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_real_2stage@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

complex_2stage@SUFFIX@_SOURCES = test/Fortran/elpa2/complex_2stage.F90
complex_2stage@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
complex_2stage@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_complex_2stage@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

real_1stage@SUFFIX@_SOURCES = test/Fortran/elpa1/real_1stage.F90
real_1stage@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
real_1stage@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_real_1stage@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

complex_1stage@SUFFIX@_SOURCES = test/Fortran/elpa1/complex_1stage.F90
complex_1stage@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
complex_1stage@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_complex_1stage@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

if WANT_SINGLE_PRECISION_REAL
single_real_2stage@SUFFIX@_SOURCES = test/Fortran/elpa2/single_real_2stage.F90
single_real_2stage@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
single_real_2stage@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_single_real_2stage@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

single_real_1stage@SUFFIX@_SOURCES = test/Fortran/elpa1/single_real_1stage.F90
single_real_1stage@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
single_real_1stage@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_single_real_1stage@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90
endif

if WANT_SINGLE_PRECISION_COMPLEX
single_complex_2stage@SUFFIX@_SOURCES = test/Fortran/elpa2/single_complex_2stage.F90
single_complex_2stage@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
single_complex_2stage@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_single_complex_2stage@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

single_complex_1stage@SUFFIX@_SOURCES = test/Fortran/elpa1/single_complex_1stage.F90
single_complex_1stage@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
single_complex_1stage@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_single_complex_1stage@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90
endif

if WITH_GPU_VERSION
real_1stage_gpu@SUFFIX@_SOURCES = test/Fortran/elpa1/real_1stage_gpu.F90
real_1stage_gpu@SUFFIX@_LDADD = $(build_lib)
real_1stage_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_real_1stage_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

complex_1stage_gpu@SUFFIX@_SOURCES = test/Fortran/elpa1/complex_1stage_gpu.F90
complex_1stage_gpu@SUFFIX@_LDADD = $(build_lib)
complex_1stage_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_complex_1stage_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

real_2stage_gpu@SUFFIX@_SOURCES = test/Fortran/elpa2/real_2stage_gpu.F90
real_2stage_gpu@SUFFIX@_LDADD = $(build_lib)
real_2stage_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_real_2stage_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

complex_2stage_gpu@SUFFIX@_SOURCES = test/Fortran/elpa2/complex_2stage_gpu.F90
complex_2stage_gpu@SUFFIX@_LDADD = $(build_lib)
complex_2stage_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_complex_2stage_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90
endif

if WANT_SINGLE_PRECISION_REAL
single_real_2stage_banded@SUFFIX@_SOURCES = test/Fortran/elpa2/single_real_2stage_banded.F90
single_real_2stage_banded@SUFFIX@_LDADD = $(build_lib)
single_real_2stage_banded@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_single_real_2stage_banded@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

single_real_1stage_gpu@SUFFIX@_SOURCES = test/Fortran/elpa1/single_real_1stage_gpu.F90
single_real_1stage_gpu@SUFFIX@_LDADD = $(build_lib)
single_real_1stage_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_single_real_1stage_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

single_real_2stage_gpu@SUFFIX@_SOURCES = test/Fortran/elpa2/single_real_2stage_gpu.F90
single_real_2stage_gpu@SUFFIX@_LDADD = $(build_lib)
single_real_2stage_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_single_real_2stage_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90
endif

if WANT_SINGLE_PRECISION_COMPLEX
single_complex_2stage_banded@SUFFIX@_SOURCES = test/Fortran/elpa2/single_complex_2stage_banded.F90
single_complex_2stage_banded@SUFFIX@_LDADD = $(build_lib)
single_complex_2stage_banded@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_single_complex_2stage_banded@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

single_complex_1stage_gpu@SUFFIX@_SOURCES = test/Fortran/elpa1/single_complex_1stage_gpu.F90
single_complex_1stage_gpu@SUFFIX@_LDADD = $(build_lib)
single_complex_1stage_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_single_complex_1stage_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

single_complex_2stage_gpu@SUFFIX@_SOURCES = test/Fortran/elpa2/single_complex_2stage_gpu.F90
single_complex_2stage_gpu@SUFFIX@_LDADD = $(build_lib)
single_complex_2stage_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_single_complex_2stage_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90
endif


if ENABLE_LEGACY
legacy_real_1stage_c_version@SUFFIX@_SOURCES = test/C/elpa1/legacy_interface/legacy_real_1stage_c_version.c
legacy_real_1stage_c_version@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
legacy_real_1stage_c_version@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_real_1stage_c_version@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_real_1stage_with_c@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_real_1stage_with_c.F90
legacy_real_1stage_with_c@SUFFIX@_LDADD = $(build_lib)
legacy_real_1stage_with_c@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_real_1stage_with_c@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_real_driver_c_version@SUFFIX@_SOURCES = test/C/driver/legacy_interface/legacy_real_driver_c_version.c
legacy_real_driver_c_version@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
legacy_real_driver_c_version@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_real_driver_c_version@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_complex_1stage_c_version@SUFFIX@_SOURCES = test/C/elpa1/legacy_interface/legacy_complex_1stage_c_version.c
legacy_complex_1stage_c_version@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
legacy_complex_1stage_c_version@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_complex_1stage_c_version@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_complex_driver_c_version@SUFFIX@_SOURCES = test/C/driver/legacy_interface/legacy_complex_driver_c_version.c
legacy_complex_driver_c_version@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
legacy_complex_driver_c_version@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_complex_driver_c_version@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_real_2stage_c_version@SUFFIX@_SOURCES = test/C/elpa2/legacy_interface/legacy_real_2stage_c_version.c
legacy_real_2stage_c_version@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
legacy_real_2stage_c_version@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_real_2stage_c_version@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_complex_2stage_c_version@SUFFIX@_SOURCES = test/C/elpa2/legacy_interface/legacy_complex_2stage_c_version.c
legacy_complex_2stage_c_version@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
legacy_complex_2stage_c_version@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_complex_2stage_c_version@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_real_1stage@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_real.F90
legacy_real_1stage@SUFFIX@_LDADD = $(build_lib)
legacy_real_1stage@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_real_1stage@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_real_toeplitz@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_toeplitz.F90
legacy_real_toeplitz@SUFFIX@_LDADD = $(build_lib)
legacy_real_toeplitz@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_real_toeplitz@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_real_transpose_multiply@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_real_transp_multiply.F90
legacy_real_transpose_multiply@SUFFIX@_LDADD = $(build_lib)
legacy_real_transpose_multiply@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_real_transpose_multiply@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_complex_transpose_multiply@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_complex_transp_multiply.F90
legacy_complex_transpose_multiply@SUFFIX@_LDADD = $(build_lib)
legacy_complex_transpose_multiply@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_complex_transpose_multiply@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_real_cholesky@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_real_cholesky.F90
legacy_real_cholesky@SUFFIX@_LDADD = $(build_lib)
legacy_real_cholesky@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_real_cholesky@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_real_invert_trm@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_real_invert_trm.F90
legacy_real_invert_trm@SUFFIX@_LDADD = $(build_lib)
legacy_real_invert_trm@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_real_invert_trm@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_complex_cholesky@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_complex_cholesky.F90
legacy_complex_cholesky@SUFFIX@_LDADD = $(build_lib)
legacy_complex_cholesky@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_complex_cholesky@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_complex_invert_trm@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_complex_invert_trm.F90
legacy_complex_invert_trm@SUFFIX@_LDADD = $(build_lib)
legacy_complex_invert_trm@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_complex_invert_trm@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_real_2stage@SUFFIX@_SOURCES = test/Fortran/elpa2/legacy_interface/legacy_real.F90
legacy_real_2stage@SUFFIX@_LDADD = $(build_lib)
legacy_real_2stage@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_real_2stage@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_real_2stage_default@SUFFIX@_SOURCES = test/Fortran/elpa2/legacy_interface/legacy_real_default_kernel.F90
legacy_real_2stage_default@SUFFIX@_LDADD = $(build_lib)
legacy_real_2stage_default@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_real_2stage_default@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_real_2stage_qr@SUFFIX@_SOURCES = test/Fortran/elpa2/legacy_interface/legacy_real_qr.F90
legacy_real_2stage_qr@SUFFIX@_LDADD = $(build_lib)
legacy_real_2stage_qr@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_real_2stage_qr@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_real_2stage_api@SUFFIX@_SOURCES = test/Fortran/elpa2/legacy_interface/legacy_real_api.F90
legacy_real_2stage_api@SUFFIX@_LDADD = $(build_lib)
legacy_real_2stage_api@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_real_2stage_api@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_complex_1stage@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_complex.F90
legacy_complex_1stage@SUFFIX@_LDADD = $(build_lib)
legacy_complex_1stage@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_complex_1stage@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_complex_2stage@SUFFIX@_SOURCES = test/Fortran/elpa2/legacy_interface/legacy_complex.F90
legacy_complex_2stage@SUFFIX@_LDADD = $(build_lib)
legacy_complex_2stage@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_complex_2stage@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_complex_2stage_default@SUFFIX@_SOURCES = test/Fortran/elpa2/legacy_interface/legacy_complex_default_kernel.F90
legacy_complex_2stage_default@SUFFIX@_LDADD = $(build_lib)
legacy_complex_2stage_default@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_complex_2stage_default@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_complex_2stage_api@SUFFIX@_SOURCES = test/Fortran/elpa2/legacy_interface/legacy_complex_api.F90
legacy_complex_2stage_api@SUFFIX@_LDADD = $(build_lib)
legacy_complex_2stage_api@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_complex_2stage_api@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_real_driver@SUFFIX@_SOURCES = test/Fortran/driver/legacy_interface/legacy_real_driver.F90
legacy_real_driver@SUFFIX@_LDADD = $(build_lib)
legacy_real_driver@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_real_driver@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_complex_driver@SUFFIX@_SOURCES = test/Fortran/driver/legacy_interface/legacy_complex_driver.F90
legacy_complex_driver@SUFFIX@_LDADD = $(build_lib)
legacy_complex_driver@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_complex_driver@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

elpa_tests@SUFFIX@_SOURCES = test/Fortran/elpa_tests.F90
elpa_tests@SUFFIX@_LDADD = $(build_lib)
elpa_tests@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules

if WANT_SINGLE_PRECISION_REAL

legacy_single_real_1stage@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_single_real.F90
legacy_single_real_1stage@SUFFIX@_LDADD = $(build_lib)
legacy_single_real_1stage@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_real_1stage@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_real_2stage@SUFFIX@_SOURCES = test/Fortran/elpa2/legacy_interface/legacy_single_real.F90
legacy_single_real_2stage@SUFFIX@_LDADD = $(build_lib)
legacy_single_real_2stage@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_real_2stage@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_real_driver@SUFFIX@_SOURCES = test/Fortran/driver/legacy_interface/legacy_single_real_driver.F90
legacy_single_real_driver@SUFFIX@_LDADD = $(build_lib)
legacy_single_real_driver@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_real_driver@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_real_toeplitz@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_single_toeplitz.F90
legacy_single_real_toeplitz@SUFFIX@_LDADD = $(build_lib)
legacy_single_real_toeplitz@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_real_toeplitz@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_real_transpose_multiply@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_single_real_transp_multiply.F90
legacy_single_real_transpose_multiply@SUFFIX@_LDADD = $(build_lib)
legacy_single_real_transpose_multiply@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_real_transpose_multiply@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_real_driver_c_version@SUFFIX@_SOURCES = test/C/driver/legacy_interface/legacy_single_real_driver_c_version.c
legacy_single_real_driver_c_version@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
legacy_single_real_driver_c_version@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_real_driver_c_version@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_real_cholesky@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_single_real_cholesky.F90
legacy_single_real_cholesky@SUFFIX@_LDADD = $(build_lib)
legacy_single_real_cholesky@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_real_cholesky@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_real_invert_trm@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_single_real_invert_trm.F90
legacy_single_real_invert_trm@SUFFIX@_LDADD = $(build_lib)
legacy_single_real_invert_trm@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_real_invert_trm@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_real_2stage_default@SUFFIX@_SOURCES = test/Fortran/elpa2/legacy_interface/legacy_single_real_default_kernel.F90
legacy_single_real_2stage_default@SUFFIX@_LDADD = $(build_lib)
legacy_single_real_2stage_default@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_real_2stage_default@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_real_2stage_qr@SUFFIX@_SOURCES = \
                test/Fortran/elpa2/legacy_interface/legacy_single_real_qr.F90
legacy_single_real_2stage_qr@SUFFIX@_LDADD = $(build_lib)
legacy_single_real_2stage_qr@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_real_2stage_qr@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_real_2stage_api@SUFFIX@_SOURCES = test/Fortran/elpa2/legacy_interface/legacy_single_real_api.F90
legacy_single_real_2stage_api@SUFFIX@_LDADD = $(build_lib)
legacy_single_real_2stage_api@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_real_2stage_api@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90
endif

if WANT_SINGLE_PRECISION_COMPLEX
legacy_single_complex_1stage@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_single_complex.F90
legacy_single_complex_1stage@SUFFIX@_LDADD = $(build_lib)
legacy_single_complex_1stage@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_complex_1stage@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_complex_driver@SUFFIX@_SOURCES = test/Fortran/driver/legacy_interface/legacy_single_complex_driver.F90
legacy_single_complex_driver@SUFFIX@_LDADD = $(build_lib)
legacy_single_complex_driver@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_complex_driver@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_complex_driver_c_version@SUFFIX@_SOURCES = test/C/driver/legacy_interface/legacy_single_complex_driver_c_version.c
legacy_single_complex_driver_c_version@SUFFIX@_LDADD = $(build_lib) $(FCLIBS)
legacy_single_complex_driver_c_version@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
legacy_single_complex_driver_c_version@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_complex_2stage@SUFFIX@_SOURCES = test/Fortran/elpa2/legacy_interface/legacy_single_complex.F90
legacy_single_complex_2stage@SUFFIX@_LDADD = $(build_lib)
legacy_single_complex_2stage@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_complex_2stage@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_complex_2stage_default@SUFFIX@_SOURCES = test/Fortran/elpa2/legacy_interface/legacy_single_complex_default_kernel.F90 $(shared_sources) $(redirect_sources)
legacy_single_complex_2stage_default@SUFFIX@_LDADD = $(build_lib)
legacy_single_complex_2stage_default@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_complex_2stage_default@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_complex_transpose_multiply@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_single_complex_transp_multiply.F90
legacy_single_complex_transpose_multiply@SUFFIX@_LDADD = $(build_lib)
legacy_single_complex_transpose_multiply@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_complex_transpose_multiply@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_complex_cholesky@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_single_complex_cholesky.F90
legacy_single_complex_cholesky@SUFFIX@_LDADD = $(build_lib)
legacy_single_complex_cholesky@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_complex_cholesky@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_complex_invert_trm@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_single_complex_invert_trm.F90
legacy_single_complex_invert_trm@SUFFIX@_LDADD = $(build_lib)
legacy_single_complex_invert_trm@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_complex_invert_trm@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_complex_2stage_api@SUFFIX@_SOURCES = test/Fortran/elpa2/legacy_interface/legacy_single_complex_api.F90
legacy_single_complex_2stage_api@SUFFIX@_LDADD = $(build_lib)
legacy_single_complex_2stage_api@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_complex_2stage_api@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90
endif

if WITH_GPU_VERSION
legacy_real_1stage_gpu@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_real_gpu.F90
legacy_real_1stage_gpu@SUFFIX@_LDADD = $(build_lib)
legacy_real_1stage_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_real_1stage_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_complex_1stage_gpu@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_complex_gpu.F90
legacy_complex_1stage_gpu@SUFFIX@_LDADD = $(build_lib)
legacy_complex_1stage_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_complex_1stage_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_real_2stage_gpu@SUFFIX@_SOURCES = test/Fortran/elpa2/legacy_interface/legacy_real_gpu.F90
legacy_real_2stage_gpu@SUFFIX@_LDADD = $(build_lib)
legacy_real_2stage_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_real_2stage_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_complex_2stage_gpu@SUFFIX@_SOURCES = test/Fortran/elpa2/legacy_interface/legacy_complex_gpu.F90
legacy_complex_2stage_gpu@SUFFIX@_LDADD = $(build_lib)
legacy_complex_2stage_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_complex_2stage_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

if WANT_SINGLE_PRECISION_REAL
legacy_single_real_1stage_gpu@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_single_real_gpu.F90
legacy_single_real_1stage_gpu@SUFFIX@_LDADD = $(build_lib)
legacy_single_real_1stage_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_real_1stage_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_real_2stage_gpu@SUFFIX@_SOURCES = test/Fortran/elpa2/legacy_interface/legacy_single_real_gpu.F90
legacy_single_real_2stage_gpu@SUFFIX@_LDADD = $(build_lib)
legacy_single_real_2stage_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_real_2stage_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90
endif

if WANT_SINGLE_PRECISION_COMPLEX
legacy_single_complex_1stage_gpu@SUFFIX@_SOURCES = test/Fortran/elpa1/legacy_interface/legacy_single_complex_gpu.F90
legacy_single_complex_1stage_gpu@SUFFIX@_LDADD = $(build_lib)
legacy_single_complex_1stage_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_complex_1stage_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90

legacy_single_complex_2stage_gpu@SUFFIX@_SOURCES = test/Fortran/elpa2/legacy_interface/legacy_single_complex_gpu.F90
legacy_single_complex_2stage_gpu@SUFFIX@_LDADD = $(build_lib)
legacy_single_complex_2stage_gpu@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
EXTRA_legacy_single_complex_2stage_gpu@SUFFIX@_DEPENDENCIES = test/Fortran/elpa_print_headers.X90
endif
endif
endif

check_SCRIPTS = \
  test_real_double_1stage.sh \
  test_real_double_2stage.sh \
  test_complex_double_1stage.sh \
  test_complex_double_2stage.sh \
  test_c_real_double_1stage.sh \
  test_c_real_double_2stage.sh \
  test_c_complex_double_1stage.sh \
  test_c_complex_double_2stage.sh \
  double_instance@SUFFIX@.sh \
  real_2stage_banded@SUFFIX@.sh \
  complex_2stage_banded@SUFFIX@.sh \
  real_2stage@SUFFIX@.sh \
  complex_2stage@SUFFIX@.sh \
  real_1stage@SUFFIX@.sh \
  complex_1stage@SUFFIX@.sh

if WANT_SINGLE_PRECISION_REAL
check_SCRIPTS += \
  test_single_real_1stage.sh \
  test_single_real_2stage.sh \
  test_c_single_real_1stage.sh \
  test_c_single_real_2stage.sh \
  single_real_2stage_banded@SUFFIX@.sh \
  single_real_2stage@SUFFIX@.sh \
  single_real_1stage@SUFFIX@.sh
endif

if WANT_SINGLE_PRECISION_COMPLEX
check_SCRIPTS += \
  test_single_complex_1stage.sh \
  test_single_complex_2stage.sh \
  test_c_single_complex_1stage.sh \
  test_c_single_complex_2stage.sh \
  single_complex_2stage_banded@SUFFIX@.sh \
  single_complex_2stage@SUFFIX@.sh \
  single_complex_1stage@SUFFIX@.sh
endif

if WITH_GPU_VERSION
check_SCRIPTS += \
  complex_1stage_gpu@SUFFIX@.sh \
  complex_2stage_gpu@SUFFIX@.sh \
  real_1stage_gpu@SUFFIX@.sh \
  real_2stage_gpu@SUFFIX@.sh

if WANT_SINGLE_PRECISION_REAL
check_SCRIPTS += \
  single_real_1stage_gpu@SUFFIX@.sh \
  single_real_2stage_gpu@SUFFIX@.sh
endif

if WANT_SINGLE_PRECISION_COMPLEX
check_SCRIPTS += \
  single_complex_1stage_gpu@SUFFIX@.sh \
  single_complex_2stage_gpu@SUFFIX@.sh
endif
endif

if ENABLE_LEGACY
check_SCRIPTS += \
  legacy_real_1stage@SUFFIX@.sh \
  legacy_real_2stage@SUFFIX@.sh \
  legacy_real_2stage_default@SUFFIX@.sh \
  legacy_complex_1stage@SUFFIX@.sh \
  legacy_complex_2stage@SUFFIX@.sh \
  legacy_real_2stage_qr@SUFFIX@.sh \
  legacy_complex_2stage_default@SUFFIX@.sh \
  legacy_real_2stage_api@SUFFIX@.sh \
  legacy_complex_2stage_api@SUFFIX@.sh \
  legacy_real_driver@SUFFIX@.sh \
  legacy_complex_driver@SUFFIX@.sh \
  legacy_real_toeplitz@SUFFIX@.sh \
  legacy_real_1stage_with_c@SUFFIX@.sh \
  legacy_real_cholesky@SUFFIX@.sh \
  legacy_real_invert_trm@SUFFIX@.sh \
  legacy_real_transpose_multiply@SUFFIX@.sh \
  legacy_complex_cholesky@SUFFIX@.sh \
  legacy_complex_invert_trm@SUFFIX@.sh \
  legacy_complex_transpose_multiply@SUFFIX@.sh \
  legacy_real_1stage_c_version@SUFFIX@.sh \
  legacy_complex_1stage_c_version@SUFFIX@.sh \
  legacy_real_2stage_c_version@SUFFIX@.sh \
  legacy_complex_2stage_c_version@SUFFIX@.sh \
  legacy_real_driver_c_version@SUFFIX@.sh \
  legacy_complex_driver_c_version@SUFFIX@.sh

if WANT_SINGLE_PRECISION_REAL
check_SCRIPTS += \
  legacy_single_real_1stage@SUFFIX@.sh \
  legacy_single_real_2stage@SUFFIX@.sh \
  legacy_single_real_2stage_default@SUFFIX@.sh \
  legacy_single_real_2stage_qr@SUFFIX@.sh \
  legacy_single_real_driver@SUFFIX@.sh \
  legacy_single_real_cholesky@SUFFIX@.sh \
  legacy_single_real_invert_trm@SUFFIX@.sh \
  legacy_single_real_transpose_multiply@SUFFIX@.sh \
  legacy_single_real_2stage_api@SUFFIX@.sh \
  legacy_single_real_driver_c_version@SUFFIX@.sh
endif

if WANT_SINGLE_PRECISION_COMPLEX
check_SCRIPTS += \
  legacy_single_complex_1stage@SUFFIX@.sh \
  legacy_single_complex_2stage@SUFFIX@.sh \
  legacy_single_complex_2stage_default@SUFFIX@.sh \
  legacy_single_complex_cholesky@SUFFIX@.sh \
  legacy_single_complex_driver@SUFFIX@.sh \
  legacy_single_complex_invert_trm@SUFFIX@.sh \
  legacy_single_complex_transpose_multiply@SUFFIX@.sh \
  legacy_single_complex_2stage_api@SUFFIX@.sh \
  legacy_single_complex_driver_c_version@SUFFIX@.sh
endif

if WITH_GPU_VERSION
check_SCRIPTS += \
  legacy_real_1stage_gpu@SUFFIX@.sh \
  legacy_complex_1stage_gpu@SUFFIX@.sh \
  legacy_real_2stage_gpu@SUFFIX@.sh \
  legacy_complex_2stage_gpu@SUFFIX@.sh

if WANT_SINGLE_PRECISION_REAL
check_SCRIPTS += \
  legacy_single_real_1stage_gpu@SUFFIX@.sh \
  legacy_single_real_2stage_gpu@SUFFIX@.sh
endif
if WANT_SINGLE_PRECISION_COMPLEX
check_SCRIPTS += \
  legacy_single_complex_1stage_gpu@SUFFIX@.sh \
  legacy_single_complex_2stage_gpu@SUFFIX@.sh
endif
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

include doxygen.am

CLEANFILES = \
  elpa-generated.h \
  elpa1_test* \
  elpa2_test*\
  elpa2_real* \
  elpa1_real* \
  elpa*.sh \
  test*.sh \
  legacy_real* \
  legacy_complex* \
  legacy_single_real* \
  legacy_single_complex* \
  single_real* \
  single_complex* \
  real* \
  complex* \
  double_instance* \
  *.i

clean-local:
	-rm -rf modules/* private_modules/* .fortran_dependencies/*
	-rm -rf $(generated_headers)

distclean-local:
	-rm config-f90.h
	-rm -rf ./src/elpa2/kernels/.deps
	-rm -rf ./src/.deps
	-rm -rf ./test/.deps
	-rmdir ./src/elpa2/kernels/
	-rmdir ./src
	-rmdir ./test
	-rmdir ./m4
	-rmdir modules/
	-rmdir .fortran_dependencies/

EXTRA_DIST = \
  elpa/elpa.h \
  elpa/elpa_legacy.h \
  elpa/elpa_generic.h \
  fdep/fortran_dependencies.pl \
  fdep/fortran_dependencies.mk \
  test/Fortran/elpa_print_headers.X90 \
  test/Fortran/assert.h \
  src/elpa1/elpa_reduce_add_vectors.X90 \
  src/elpa1/elpa_transpose_vectors.X90 \
  src/elpa1/elpa1_compute_template.X90 \
  src/elpa1/elpa1_merge_systems_real_template.X90 \
  src/elpa1/elpa1_solve_tridi_real_template.X90 \
  src/elpa1/elpa1_tools_template.X90 \
  src/elpa1/elpa1_trans_ev_template.X90 \
  src/elpa1/elpa1_tridiag_template.X90 \
  src/elpa2/elpa2_compute_real_template.X90 \
  src/elpa2/elpa2_compute_complex_template.X90 \
  src/elpa2/elpa2_bandred_template.X90 \
  src/elpa2/elpa2_herm_matrix_allreduce_complex_template.X90 \
  src/elpa2/elpa2_symm_matrix_allreduce_real_template.X90 \
  src/elpa1/elpa1_template.X90 \
  src/elpa2/elpa2_template.X90 \
  src/elpa2/qr/qr_utils_template.X90 \
  src/elpa2/qr/elpa_pdlarfb_template.X90 \
  src/elpa2/qr/elpa_pdgeqrf_template.X90 \
  src/elpa2/elpa2_tridiag_band_template.X90 \
  src/elpa2/elpa2_trans_ev_band_to_full_template.X90 \
  src/elpa2/elpa2_trans_ev_tridi_to_band_template.X90 \
  src/general/precision_macros.h \
  src/elpa_index.h \
  src/fortran_constants.h \
  src/elpa2/kernels/elpa2_kernels_real_template.X90 \
  src/elpa2/kernels/elpa2_kernels_complex_template.X90 \
  src/elpa2/kernels/elpa2_kernels_simple_template.X90 \
  src/elpa2/redist_band.X90 \
  src/elpa2/pack_unpack_cpu.X90 \
  src/elpa2/pack_unpack_gpu.X90 \
  src/elpa2/compute_hh_trafo.X90 \
  src/general/sanity.X90 \
  src/elpa1/elpa_cholesky_template.X90 \
  src/elpa1/elpa_invert_trm.X90 \
  src/elpa1/elpa_multiply_a_b.X90 \
  src/elpa1/elpa_solve_tridi_impl_public.X90 \
  src/elpa2/qr/elpa_qrkernels_template.X90 \
  src/elpa2/GPU/ev_tridi_band_gpu_c_v2_complex_template.Xcu \
  src/elpa2/GPU/ev_tridi_band_gpu_c_v2_real_template.Xcu \
  src/GPU/cuUtils_template.Xcu \
  test/shared/prepare_matrix_template.X90\
  test/shared/check_correctness_template.X90 \
  nvcc_wrap \
  test_project/Makefile.am \
  test_project/autogen.sh \
  test_project/configure.ac \
  test_project/fdep \
  test_project/src/test_real.F90 \
  test_project/m4 \
  elpa.spec

if ENABLE_LEGACY
EXTRA_DIST += \
  src/elpa1/elpa1_solve_tridi_real_template.X90 \
  src/elpa1/legacy_interface/elpa1_template.X90 \
  src/elpa2/legacy_interface/elpa2_template.X90 \
  src/elpa1/legacy_interface/elpa1_c_interface_template.X90 \
  src/elpa2/legacy_interface/elpa2_c_interface_template.X90 \
  src/elpa_driver/legacy_interface/elpa_driver_c_interface_template.X90 \
  src/elpa1/legacy_interface/elpa_solve_tridi_c_interface_template.X90 \
  src/elpa1/legacy_interface/elpa_mult_at_b_c_interface_template.X90 \
  src/elpa1/legacy_interface/elpa_mult_ah_b_c_interface_template.X90 \
  src/elpa1/legacy_interface/elpa_invert_trm_c_interface_template.X90 \
  src/elpa1/legacy_interface/elpa_cholesky_c_interface_template.X90 \
  src/elpa1/legacy_interface/elpa_cholesky_template.X90 \
  src/elpa1/legacy_interface/elpa_invert_trm.X90 \
  src/elpa1/legacy_interface/elpa_multiply_a_b.X90 \
  src/elpa1/legacy_interface/elpa_solve_tridi.X90
endif


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
