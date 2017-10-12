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
  src/elpa.F90 \
  src/elpa_api.F90 \
  src/elpa_constants.F90

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
  src/elpa1/elpa1_auxiliary.F90 \
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
  src/helpers/matrix_plot.F90 \
  src/elpa_c_interface.c \
  src/elpa_index.c

# elpa_utilities.F90 is private in new API, public in legacy
if ENABLE_LEGACY
libelpa@SUFFIX@_public_la_SOURCES += \
  src/general/elpa_utilities.F90
else
libelpa@SUFFIX@_private_la_SOURCES += \
  src/general/elpa_utilities.F90
endif


if ENABLE_LEGACY
libelpa@SUFFIX@_private_la_SOURCES += \
  src/elpa_driver/legacy_interface/elpa_driver_c_interface.F90 \
  src/elpa1/legacy_interface/elpa_1stage_c_interface.F90 \
  src/elpa2/legacy_interface/elpa_2stage_c_interface.F90
endif

EXTRA_libelpa@SUFFIX@_private_la_DEPENDENCIES = \
  src/elpa1/elpa_reduce_add_vectors.F90 \
  src/elpa1/elpa_transpose_vectors.F90 \
  src/elpa1/elpa1_compute_template.F90 \
  src/elpa2/elpa2_compute_real_template.F90 \
  src/elpa2/elpa2_compute_complex_template.F90 \
  src/elpa1/elpa1_template.F90 \
  src/elpa2/elpa2_template.F90 \
  src/elpa2/qr/qr_utils_template.F90 \
  src/elpa2/qr/elpa_pdlarfb_template.F90 \
  src/elpa2/qr/elpa_pdgeqrf_template.F90 \
  src/elpa2/elpa2_bandred_template.F90 \
  src/elpa2/elpa2_symm_matrix_allreduce_real_template.F90 \
  src/elpa2/elpa2_trans_ev_band_to_full_template.F90 \
  src/elpa2/elpa2_tridiag_band_template.F90 \
  src/elpa2/elpa2_trans_ev_tridi_to_band_template.F90 \
  src/elpa2/elpa2_herm_matrix_allreduce_complex_template.F90 \
  src/elpa2/kernels/real_template.F90 \
  src/elpa2/kernels/complex_template.F90 \
  src/elpa2/kernels/simple_template.F90 \
  src/elpa2/pack_unpack_cpu.F90 \
  src/elpa2/pack_unpack_gpu.F90 \
  src/elpa2/compute_hh_trafo.F90 \
  src/elpa2/redist_band.F90 \
  src/general/sanity.F90 \
  src/elpa1/elpa_cholesky_template.F90 \
  src/elpa1/elpa_invert_trm.F90 \
  src/elpa1/elpa_multiply_a_b.F90 \
  src/elpa1/elpa_solve_tridi_impl_public.F90 \
  src/general/precision_macros.h \
  src/general/precision_kinds.F90

if ENABLE_LEGACY
EXTRA_libelpa@SUFFIX@_private_la_DEPENDENCIES += \
  src/elpa1/legacy_interface/elpa1_template.F90 \
  src/elpa2/legacy_interface/elpa2_template.F90 \
  src/elpa1/legacy_interface/elpa1_c_interface_template.F90 \
  src/elpa2/legacy_interface/elpa2_c_interface_template.F90 \
  src/elpa_driver/legacy_interface/elpa_driver_c_interface_template.F90 \
  src/elpa1/legacy_interface/elpa_solve_tridi_c_interface_template.F90 \
  src/elpa1/legacy_interface/elpa_mult_at_b_c_interface_template.F90 \
  src/elpa1/legacy_interface/elpa_mult_ah_b_c_interface_template.F90 \
  src/elpa1/legacy_interface/elpa_invert_trm_c_interface_template.F90 \
  src/elpa1/legacy_interface/elpa_cholesky_c_interface_template.F90 \
  src/elpa1/legacy_interface/elpa_cholesky_template.F90 \
  src/elpa1/legacy_interface/elpa_invert_trm.F90 \
  src/elpa1/legacy_interface/elpa_multiply_a_b.F90 \
  src/elpa1/legacy_interface/elpa_solve_tridi.F90
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
  EXTRA_libelpa@SUFFIX@_private_la_DEPENDENCIES +=   src/elpa2/GPU/ev_tridi_band_gpu_c_v2_complex_template.cu src/elpa2/GPU/ev_tridi_band_gpu_c_v2_real_template.cu
endif

if !WITH_MPI
  libelpa@SUFFIX@_private_la_SOURCES += src/helpers/mod_time_c.F90
if !HAVE_DETAILED_TIMINGS
  libelpa@SUFFIX@_private_la_SOURCES += src/ftimings/time.c
endif
endif

if WITH_REAL_GENERIC_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES +=  src/elpa2/kernels/real.F90
endif

if WITH_COMPLEX_GENERIC_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES +=  src/elpa2/kernels/complex.F90
endif

if WITH_REAL_GENERIC_SIMPLE_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_simple.F90
endif

if WITH_COMPLEX_GENERIC_SIMPLE_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_simple.F90
endif

if WITH_REAL_BGP_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_bgp.f90
endif

if WITH_REAL_BGQ_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_bgq.f90
endif

if WITH_REAL_SSE_ASSEMBLY_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/asm_x86_64_double_precision.s
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/asm_x86_64_single_precision.s
endif

else
if WITH_COMPLEX_SSE_ASSEMBLY_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/asm_x86_64_double_precision.s
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/asm_x86_64_single_precision.s
endif
endif
endif

if WITH_REAL_SSE_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sse_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sse_2hv_single_precision.c
endif
endif

if WITH_REAL_AVX_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx-avx2_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx-avx2_2hv_single_precision.c
endif
else
if WITH_REAL_AVX2_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx-avx2_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx-avx2_2hv_single_precision.c
endif
endif
endif

if WITH_REAL_AVX512_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx512_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx512_2hv_single_precision.c
endif
endif


if WITH_REAL_SSE_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sse_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sse_4hv_single_precision.c
endif
endif

if WITH_REAL_AVX_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx-avx2_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx-avx2_4hv_single_precision.c
endif
else
if WITH_REAL_AVX2_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx-avx2_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx-avx2_4hv_single_precision.c
endif
endif
endif

if WITH_REAL_AVX512_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx512_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx512_4hv_single_precision.c
endif
endif



if WITH_REAL_SSE_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sse_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sse_6hv_single_precision.c
endif
endif

if WITH_REAL_AVX_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx-avx2_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx-avx2_6hv_single_precision.c
endif
else
if WITH_REAL_AVX2_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx-avx2_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx-avx2_6hv_single_precision.c
endif
endif
endif

if WITH_REAL_AVX512_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx512_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx512_6hv_single_precision.c
endif
endif


if WITH_COMPLEX_SSE_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sse_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sse_1hv_single_precision.c
endif
endif

if WITH_COMPLEX_AVX_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx-avx2_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx-avx2_1hv_single_precision.c
endif
else
if WITH_COMPLEX_AVX2_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx-avx2_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx-avx2_1hv_single_precision.c
endif
endif
endif


if WITH_COMPLEX_AVX512_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx512_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx512_1hv_single_precision.c
endif
endif

if WITH_COMPLEX_SSE_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sse_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sse_2hv_single_precision.c
endif
endif

if WITH_COMPLEX_AVX_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx-avx2_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx-avx2_2hv_single_precision.c
endif
else
if WITH_COMPLEX_AVX2_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx-avx2_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx-avx2_2hv_single_precision.c
endif
endif
endif

if WITH_COMPLEX_AVX512_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx512_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx512_2hv_single_precision.c
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
nobase_elpa_include_HEADERS = \
  $(wildcard modules/*) \
  elpa/elpa.h \
  elpa/elpa_generic.h \
  elpa/elpa_legacy.h

nobase_nodist_elpa_include_HEADERS = \
  elpa/elpa_constants.h \
  elpa/elpa_generated.h \
  elpa/elpa_generated_legacy.h

dist_man_MANS = \
  man/elpa2_print_kernels.1 \
  man/elpa_init.3 \
  man/elpa_allocate.3 \
  man/elpa_set.3 \
  man/elpa_setup.3 \
  man/elpa_eigenvalues.3 \
  man/elpa_eigenvectors.3 \
  man/elpa_cholesky.3 \
  man/elpa_invert_triangular.3 \
  man/elpa_solve_tridiagonal.3 \
  man/elpa_hermitian_multiply.3 \
  man/elpa_deallocate.3 \
  man/elpa_uninit.3

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
  man/elpa_solve_evp_complex_single.3
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

noinst_PROGRAMS =
check_SCRIPTS =

test_program_ldadd = libelpatest@SUFFIX@.la libelpa@SUFFIX@.la
test_program_fcflags = $(AM_FCFLAGS) $(FC_MODOUT)test_modules $(FC_MODINC)test_modules $(FC_MODINC)modules $(FC_MODINC)private_modules

# library with shared sources for the test files
noinst_LTLIBRARIES += libelpatest@SUFFIX@.la
libelpatest@SUFFIX@_la_FCFLAGS = $(test_program_fcflags)
libelpatest@SUFFIX@_la_SOURCES = \
  test/shared/test_util.F90 \
  test/shared/test_read_input_parameters.F90 \
  test/shared/test_check_correctness.F90 \
  test/shared/test_setup_mpi.F90 \
  test/shared/test_blacs_infrastructure.F90 \
  test/shared/test_prepare_matrix.F90 \
  test/shared/test_analytic.F90 \
  test/shared/test_output_type.F90

if WITH_SCALAPACK_TESTS
libelpatest@SUFFIX@_la_SOURCES += \
  test/shared/test_scalapack.F90
endif

if HAVE_REDIRECT
libelpatest@SUFFIX@_la_SOURCES += \
  test/shared/test_redir.c \
  test/shared/test_redirect.F90
endif


elpa2_print_kernels@SUFFIX@_SOURCES = src/elpa2/elpa2_print_kernels.F90
elpa2_print_kernels@SUFFIX@_LDADD = libelpa@SUFFIX@.la
elpa2_print_kernels@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODINC)modules

include test_programs.am

if ENABLE_LEGACY
include legacy_test_programs.am
endif

noinst_PROGRAMS += double_instance@SUFFIX@
check_SCRIPTS += double_instance@SUFFIX@.sh
double_instance@SUFFIX@_SOURCES = test/Fortran/elpa2/double_instance.F90
double_instance@SUFFIX@_LDADD = $(test_program_ldadd)
double_instance@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODINC)test_modules $(FC_MODINC)modules

noinst_PROGRAMS += real_2stage_banded@SUFFIX@
check_SCRIPTS += real_2stage_banded@SUFFIX@.sh
real_2stage_banded@SUFFIX@_SOURCES = test/Fortran/elpa2/real_2stage_banded.F90
real_2stage_banded@SUFFIX@_LDADD = $(test_program_ldadd)
real_2stage_banded@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODINC)test_modules $(FC_MODINC)modules

noinst_PROGRAMS += complex_2stage_banded@SUFFIX@
check_SCRIPTS += complex_2stage_banded@SUFFIX@.sh
complex_2stage_banded@SUFFIX@_SOURCES = test/Fortran/elpa2/complex_2stage_banded.F90
complex_2stage_banded@SUFFIX@_LDADD = $(test_program_ldadd)
complex_2stage_banded@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODINC)test_modules $(FC_MODINC)modules

if WANT_SINGLE_PRECISION_REAL
noinst_PROGRAMS += single_real_2stage_banded@SUFFIX@
check_SCRIPTS += single_real_2stage_banded@SUFFIX@.sh
single_real_2stage_banded@SUFFIX@_SOURCES = test/Fortran/elpa2/single_real_2stage_banded.F90
single_real_2stage_banded@SUFFIX@_LDADD = $(test_program_ldadd)
single_real_2stage_banded@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODINC)test_modules $(FC_MODINC)modules
endif

if WANT_SINGLE_PRECISION_COMPLEX
noinst_PROGRAMS += single_complex_2stage_banded@SUFFIX@
check_SCRIPTS += single_complex_2stage_banded@SUFFIX@.sh
single_complex_2stage_banded@SUFFIX@_SOURCES = test/Fortran/elpa2/single_complex_2stage_banded.F90
single_complex_2stage_banded@SUFFIX@_LDADD = $(test_program_ldadd)
single_complex_2stage_banded@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODINC)test_modules $(FC_MODINC)modules
endif


# test scriptsT
TASKS ?= 2
if WITH_MPI
  wrapper=$(MPI_BINARY) -n $${TASKS:-$(TASKS)}
else
  wrapper=
endif
TESTS = $(check_SCRIPTS)
%.sh: %
	@echo "#!/bin/bash" > $@
	@echo '$(wrapper)' ./$^ '$$TEST_FLAGS' >> $@
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
	-rm -rf modules/* private_modules/* test_modules/* .fortran_dependencies/*
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
  elpa.spec \
  elpa/elpa.h \
  elpa/elpa_generic.h \
  elpa/elpa_legacy.h \
  fdep/fortran_dependencies.mk \
  fdep/fortran_dependencies.pl \
  manual_cpp \
  nvcc_wrap \
  src/GPU/cuUtils_template.cu \
  src/elpa1/elpa1_compute_template.F90 \
  src/elpa1/elpa1_merge_systems_real_template.F90 \
  src/elpa1/elpa1_solve_tridi_real_template.F90 \
  src/elpa1/elpa1_template.F90 \
  src/elpa1/elpa1_tools_template.F90 \
  src/elpa1/elpa1_trans_ev_template.F90 \
  src/elpa1/elpa1_tridiag_template.F90 \
  src/elpa1/elpa_cholesky_template.F90 \
  src/elpa1/elpa_invert_trm.F90 \
  src/elpa1/elpa_multiply_a_b.F90 \
  src/elpa1/elpa_reduce_add_vectors.F90 \
  src/elpa1/elpa_solve_tridi_impl_public.F90 \
  src/elpa1/elpa_transpose_vectors.F90 \
  src/elpa2/GPU/ev_tridi_band_gpu_c_v2_complex_template.cu \
  src/elpa2/GPU/ev_tridi_band_gpu_c_v2_real_template.cu \
  src/elpa2/compute_hh_trafo.F90 \
  src/elpa2/elpa2_bandred_template.F90 \
  src/elpa2/elpa2_compute_complex_template.F90 \
  src/elpa2/elpa2_compute_real_template.F90 \
  src/elpa2/elpa2_herm_matrix_allreduce_complex_template.F90 \
  src/elpa2/elpa2_symm_matrix_allreduce_real_template.F90 \
  src/elpa2/elpa2_template.F90 \
  src/elpa2/elpa2_trans_ev_band_to_full_template.F90 \
  src/elpa2/elpa2_trans_ev_tridi_to_band_template.F90 \
  src/elpa2/elpa2_tridiag_band_template.F90 \
  src/elpa2/kernels/complex_avx-avx2_1hv_template.c \
  src/elpa2/kernels/complex_avx-avx2_2hv_template.c \
  src/elpa2/kernels/complex_avx512_1hv_template.c \
  src/elpa2/kernels/complex_avx512_2hv_template.c \
  src/elpa2/kernels/complex_sse_1hv_template.c \
  src/elpa2/kernels/complex_sse_2hv_template.c \
  src/elpa2/kernels/complex_template.F90 \
  src/elpa2/kernels/real_avx-avx2_2hv_template.c \
  src/elpa2/kernels/real_avx-avx2_4hv_template.c \
  src/elpa2/kernels/real_avx-avx2_6hv_template.c \
  src/elpa2/kernels/real_avx512_2hv_template.c \
  src/elpa2/kernels/real_avx512_4hv_template.c \
  src/elpa2/kernels/real_avx512_6hv_template.c \
  src/elpa2/kernels/real_sse_2hv_template.c \
  src/elpa2/kernels/real_sse_4hv_template.c \
  src/elpa2/kernels/real_sse_6hv_template.c \
  src/elpa2/kernels/real_template.F90 \
  src/elpa2/kernels/simple_template.F90 \
  src/elpa2/pack_unpack_cpu.F90 \
  src/elpa2/pack_unpack_gpu.F90 \
  src/elpa2/qr/elpa_pdgeqrf_template.F90 \
  src/elpa2/qr/elpa_pdlarfb_template.F90 \
  src/elpa2/qr/elpa_qrkernels_template.F90 \
  src/elpa2/qr/qr_utils_template.F90 \
  src/elpa2/redist_band.F90 \
  src/elpa_index.h \
  src/fortran_constants.h \
  src/general/map_global_to_local.F90 \
  src/general/precision_macros.h \
  src/general/precision_kinds.F90 \
  src/general/prow_pcol.F90 \
  src/general/sanity.F90 \
  test/Fortran/assert.h \
  test/Fortran/elpa_print_headers.F90 \
  test/shared/test_check_correctness_template.F90 \
  test/shared/test_prepare_matrix_template.F90 \
  test/shared/test_analytic_template.F90 \
  test_project/Makefile.am \
  test_project/autogen.sh \
  test_project/configure.ac \
  test_project/fdep \
  test_project/m4 \
  test_project/src/test_real.F90 \
  test_project_2stage/Makefile.am \
  test_project_2stage/autogen.sh \
  test_project_2stage/configure.ac \
  test_project_2stage/fdep \
  test_project_2stage/m4 \
  test_project_2stage/src/test_real2.F90 \
  test_project_C/Makefile.am \
  test_project_C/autogen.sh \
  test_project_C/configure.ac \
  test_project_C/fdep \
  test_project_C/m4 \
  test_project_C/src/test_real.c \
  test_project_C/src/test_blacs_infrastructure.F90

if ENABLE_LEGACY
EXTRA_DIST += \
  src/elpa1/elpa1_solve_tridi_real_template.F90 \
  src/elpa1/legacy_interface/elpa1_template.F90 \
  src/elpa2/legacy_interface/elpa2_template.F90 \
  src/elpa1/legacy_interface/elpa1_c_interface_template.F90 \
  src/elpa2/legacy_interface/elpa2_c_interface_template.F90 \
  src/elpa_driver/legacy_interface/elpa_driver_c_interface_template.F90 \
  src/elpa1/legacy_interface/elpa_solve_tridi_c_interface_template.F90 \
  src/elpa1/legacy_interface/elpa_mult_at_b_c_interface_template.F90 \
  src/elpa1/legacy_interface/elpa_mult_ah_b_c_interface_template.F90 \
  src/elpa1/legacy_interface/elpa_invert_trm_c_interface_template.F90 \
  src/elpa1/legacy_interface/elpa_cholesky_c_interface_template.F90 \
  src/elpa1/legacy_interface/elpa_cholesky_template.F90 \
  src/elpa1/legacy_interface/elpa_invert_trm.F90 \
  src/elpa1/legacy_interface/elpa_multiply_a_b.F90 \
  src/elpa1/legacy_interface/elpa_solve_tridi.F90
endif

if WITH_SCALAPACK_TESTS
EXTRA_DIST += \
  test/shared/test_scalapack_template.F90
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
