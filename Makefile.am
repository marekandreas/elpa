## Process this file with automake to produce Makefile.in

ACLOCAL_AMFLAGS = ${ACLOCAL_FLAGS} -I m4

AM_FCFLAGS = $(SCALAPACK_FCFLAGS) @FC_MODINC@modules @FC_MODOUT@modules
AM_LDFLAGS = $(SCALAPACK_LDFLAGS)

# libelpa
lib_LTLIBRARIES = libelpa@SUFFIX@.la
libelpa@SUFFIX@_la_LINK = $(FCLINK) $(AM_LDFLAGS) -version-info $(ELPA_SO_VERSION) -lstdc++

libelpa@SUFFIX@_la_SOURCES = src/mod_precision.F90 \
                             src/mod_mpi.F90 \
			     src/mod_mpi_stubs.F90 \
                             src/elpa_utilities.F90 \
	                     src/elpa1_compute.F90 \
	                     src/elpa1.F90 \
			     src/elpa2_utilities.F90 \
			     src/check_for_gpu.F90 \
			     src/mod_cuda.F90 \
			     src/interface_c_kernel.F90 \
			     src/mod_pack_unpack_real.F90 \
			     src/mod_pack_unpack_real_gpu.F90 \
			     src/elpa2_kernels/mod_single_hh_trafo_real.F90 \
			     src/mod_compute_hh_trafo_real.F90 \
			     src/mod_compute_hh_trafo_complex.F90 \
			     src/mod_pack_unpack_complex.F90 \
			     src/elpa2_compute.F90 \
			     src/elpa2.F90 \
			     src/elpa_c_interface.F90 \
                             src/elpa_qr/qr_utils.F90 \
                             src/elpa_qr/elpa_qrkernels.F90 \
                             src/elpa_qr/elpa_pdlarfb.F90 \
                             src/elpa_qr/elpa_pdgeqrf.F90
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

if WITH_GPU_VERSION
  libelpa@SUFFIX@_la_SOURCES +=  src/cudaFunctions.cu src/cuUtils.cu src/ev_tridi_band_gpu_c_v2.cu
  #src/interface_cuda.F90 src/interface_c_kernel.F90 src/ev_tridi_band_gpu_c_v2.cu src/cuUtils.cu
endif

if !WITH_MPI
  libelpa@SUFFIX@_la_SOURCES += src/mod_time_c.F90
if !HAVE_DETAILED_TIMINGS
  libelpa@SUFFIX@_la_SOURCES += src/ftimings/time.c
endif
endif

if WITH_REAL_GENERIC_KERNEL
  libelpa@SUFFIX@_la_SOURCES +=  src/elpa2_kernels/elpa2_kernels_real.F90
endif

if WITH_COMPLEX_GENERIC_KERNEL
  libelpa@SUFFIX@_la_SOURCES +=  src/elpa2_kernels/elpa2_kernels_complex.F90
endif

if WITH_REAL_GENERIC_SIMPLE_KERNEL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_simple.F90
endif

if WITH_COMPLEX_GENERIC_SIMPLE_KERNEL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_complex_simple.F90
endif

if WITH_REAL_BGP_KERNEL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgp.f90
endif

if WITH_REAL_BGQ_KERNEL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_real_bgq.f90
endif

if WITH_REAL_SSE_KERNEL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64_double_precision.s
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64_single_precision.s
endif
else
if WITH_COMPLEX_SSE_KERNEL
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64_double_precision.s
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_la_SOURCES += src/elpa2_kernels/elpa2_kernels_asm_x86_64_single_precision.s
endif

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

.cu.lo:
	NVCC="$(NVCC)" libtool --mode=compile --tag=CC $(top_srcdir)/nvcc_wrap $(NVCCFLAGS) $(LDFLAGS) -I$(top_builddir)/ -I$(top_srcdir)/  -c $< -o $@

# install any .mod files in the include/ dir
elpa_includedir = $(includedir)/elpa@SUFFIX@-@PACKAGE_VERSION@
nobase_elpa_include_HEADERS = $(wildcard modules/*)
nobase_elpa_include_HEADERS += elpa/elpa.h elpa/elpa_kernel_constants.h elpa/elpa_generated.h

man_MANS = man/solve_evp_real.3 \
	   man/solve_evp_real_1stage_double.3 \
	   man/solve_evp_complex.3 \
	   man/solve_evp_complex_1stage_double.3 \
	   man/solve_evp_real_2stage_double.3 \
	   man/solve_evp_complex_2stage_double.3 \
	   man/get_elpa_row_col_comms.3 \
	   man/get_elpa_communicators.3 \
	   man/print_available_elpa2_kernels.1

if WANT_SINGLE_PRECISION_REAL

man_MANS += man/solve_evp_real_1stage_single.3 \
	    man/solve_evp_real_2stage_single.3
endif

if WANT_SINGLE_PRECISION_COMPLEX

man_MANS += man/solve_evp_complex_1stage_single.3 \
	    man/solve_evp_complex_2stage_single.3
endif


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
  src/print_available_elpa2_kernels.F90

if WANT_SINGLE_PRECISION_COMPLEX
dist_files_DATA += test/fortran_test_programs/test_complex2_single_precision.F90 \
                   test/fortran_test_programs/test_complex2_default_kernel_single_precision.F90 \
                   test/fortran_test_programs/test_complex2_choose_kernel_with_api_single_precision.F90 \
                   test/fortran_test_programs/test_complex_single_precision.F90
endif

if WANT_SINGLE_PRECISION_REAL
dist_files_DATA += test/fortran_test_programs/test_real2_single_precision.F90 \
                   test/fortran_test_programs/test_real2_default_kernel_single_precision.F90 \
                   test/fortran_test_programs/test_real2_default_kernel_qr_decomposition_single_precision.F90 \
                   test/fortran_test_programs/test_real2_choose_kernel_with_api_single_precision.F90 \
                   test/fortran_test_programs/test_real_single_precision.F90
endif

if WITH_GPU_VERSION
dist_files_DATA += test/fortran_test_programs/test_real2_gpu_version.F90 \
                   test/fortran_test_programs/test_complex2_gpu_version.F90

if WANT_SINGLE_PRECISION_REAL
dist_files_DATA += test/fortran_test_programs/test_real2_gpu_version_single_precision.F90
endif

if WANT_SINGLE_PRECISION_COMPLEX
dist_files_DATA += test/fortran_test_programs/test_complex2_gpu_version_single_precision.F90
endif

endif
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

if WANT_SINGLE_PRECISION_COMPLEX
bin_PROGRAMS += \
  elpa1_test_complex_single_precision@SUFFIX@ \
  elpa2_test_complex_single_precision@SUFFIX@
endif

if WANT_SINGLE_PRECISION_REAL
bin_PROGRAMS += \
  elpa1_test_real_single_precision@SUFFIX@ \
  elpa2_test_real_single_precision@SUFFIX@
endif

noinst_PROGRAMS = \
  elpa2_test_real_default_kernel@SUFFIX@ \
  elpa2_test_real_default_kernel_qr_decomposition@SUFFIX@ \
  elpa2_test_complex_default_kernel@SUFFIX@ \
  elpa2_test_real_choose_kernel_with_api@SUFFIX@ \
  elpa2_test_complex_choose_kernel_with_api@SUFFIX@ \
  elpa1_test_real_c_version@SUFFIX@ \
  elpa1_test_complex_c_version@SUFFIX@ \
  elpa2_test_real_c_version@SUFFIX@ \
  elpa2_test_complex_c_version@SUFFIX@ \
  elpa1_test_real_with_c@SUFFIX@

if WANT_SINGLE_PRECISION_COMPLEX
noinst_PROGRAMS += \
  elpa2_test_complex_default_kernel_single_precision@SUFFIX@ \
  elpa2_test_complex_choose_kernel_with_api_single_precision@SUFFIX@
endif

if WANT_SINGLE_PRECISION_REAL
noinst_PROGRAMS += \
  elpa2_test_real_default_kernel_single_precision@SUFFIX@ \
  elpa2_test_real_default_kernel_qr_decomposition_single_precision@SUFFIX@ \
  elpa2_test_real_choose_kernel_with_api_single_precision@SUFFIX@
endif

if WITH_GPU_VERSION
noinst_PROGRAMS += \
  elpa2_test_complex_gpu_version@SUFFIX@ \
  elpa2_test_real_gpu_version@SUFFIX@

if WANT_SINGLE_PRECISION_REAL
noinst_PROGRAMS += \
  elpa2_test_real_gpu_version_single_precision@SUFFIX@
endif

if WANT_SINGLE_PRECISION_COMPLEX
noinst_PROGRAMS += \
  elpa2_test_complex_gpu_version_single_precision@SUFFIX@
endif

endif


build_lib = libelpa@SUFFIX@.la

if HAVE_REDIRECT
  redirect_sources = test/shared_sources/redir.c test/shared_sources/redirect.F90
else
  redirect_sources =
endif

shared_sources = test/shared_sources/util.F90 \
		 test/shared_sources/read_input_parameters.F90  \
		 test/shared_sources/check_correctnes.F90 \
		 test/shared_sources/setup_mpi.F90 \
		 test/shared_sources/blacs_infrastructure.F90 \
		 test/shared_sources/mod_output_types.F90 \
		 test/shared_sources/prepare_matrix.F90

elpa1_test_real_c_version@SUFFIX@_SOURCES = test/c_test_programs/elpa1_test_real_c_version.c $(shared_sources) $(redirect_sources)
elpa1_test_real_c_version@SUFFIX@_LDADD = $(build_lib)
elpa1_test_real_c_version@SUFFIX@_LINK = $(LINK) $(FCLIBS)

elpa1_test_complex_c_version@SUFFIX@_SOURCES = test/c_test_programs/elpa1_test_complex_c_version.c $(shared_sources) $(redirect_sources)
elpa1_test_complex_c_version@SUFFIX@_LDADD = $(build_lib)
elpa1_test_complex_c_version@SUFFIX@_LINK = $(LINK) $(FCLIBS)

elpa2_test_real_c_version@SUFFIX@_SOURCES = test/c_test_programs/elpa2_test_real_c_version.c $(shared_sources) $(redirect_sources)
elpa2_test_real_c_version@SUFFIX@_LDADD = $(build_lib)
elpa2_test_real_c_version@SUFFIX@_LINK = $(LINK) $(FCLIBS)

elpa2_test_complex_c_version@SUFFIX@_SOURCES = test/c_test_programs/elpa2_test_complex_c_version.c $(shared_sources) $(redirect_sources)
elpa2_test_complex_c_version@SUFFIX@_LDADD = $(build_lib)
elpa2_test_complex_c_version@SUFFIX@_LINK = $(LINK) $(FCLIBS)


elpa1_test_real@SUFFIX@_SOURCES = test/fortran_test_programs/test_real.F90 $(shared_sources) $(redirect_sources)
elpa1_test_real@SUFFIX@_LDADD = $(build_lib)

elpa1_test_real_with_c@SUFFIX@_SOURCES = test/fortran_test_programs/test_real_with_c.F90 test/shared_sources/mod_from_c.F90 test/shared_sources/call_elpa1.c $(shared_sources) $(redirect_sources)
elpa1_test_real_with_c@SUFFIX@_LDADD = $(build_lib)

#elpa1_test_complex_with_c@SUFFIX@_SOURCES = test/fortran_test_programs/test_complex_with_c.F90 test/shared_sources/mod_from_c.F90 test/shared_sources/call_elpa1.c $(shared_sources) $(redirect_sources)
#elpa1_test_complex_with_c@SUFFIX@_LDADD = $(build_lib)

elpa2_test_real@SUFFIX@_SOURCES = test/fortran_test_programs/test_real2.F90 $(shared_sources) $(redirect_sources)
elpa2_test_real@SUFFIX@_LDFLAGS = -static
elpa2_test_real@SUFFIX@_LDADD = $(build_lib)

elpa2_test_real_default_kernel@SUFFIX@_SOURCES = test/fortran_test_programs/test_real2_default_kernel.F90 $(shared_sources) $(redirect_sources)
elpa2_test_real_default_kernel@SUFFIX@_LDADD = $(build_lib)

elpa2_test_real_default_kernel_qr_decomposition@SUFFIX@_SOURCES = test/fortran_test_programs/test_real2_default_kernel_qr_decomposition.F90 $(shared_sources) $(redirect_sources)
elpa2_test_real_default_kernel_qr_decomposition@SUFFIX@_LDADD = $(build_lib)

elpa2_test_real_choose_kernel_with_api@SUFFIX@_SOURCES = test/fortran_test_programs/test_real2_choose_kernel_with_api.F90 $(shared_sources) $(redirect_sources)
elpa2_test_real_choose_kernel_with_api@SUFFIX@_LDADD = $(build_lib)

elpa1_test_complex@SUFFIX@_SOURCES = test/fortran_test_programs/test_complex.F90 $(shared_sources) $(redirect_sources)
elpa1_test_complex@SUFFIX@_LDADD = $(build_lib)

elpa2_test_complex@SUFFIX@_SOURCES = test/fortran_test_programs/test_complex2.F90 $(shared_sources) $(redirect_sources)
elpa2_test_complex@SUFFIX@_LDADD = $(build_lib)

elpa2_test_complex_default_kernel@SUFFIX@_SOURCES = test/fortran_test_programs/test_complex2_default_kernel.F90 $(shared_sources) $(redirect_sources)
elpa2_test_complex_default_kernel@SUFFIX@_LDADD = $(build_lib)

elpa2_test_complex_choose_kernel_with_api@SUFFIX@_SOURCES = test/fortran_test_programs/test_complex2_choose_kernel_with_api.F90 $(shared_sources) $(redirect_sources)
elpa2_test_complex_choose_kernel_with_api@SUFFIX@_LDADD = $(build_lib)

elpa2_print_kernels@SUFFIX@_SOURCES = src/print_available_elpa2_kernels.F90 $(shared_sources) $(redirect_sources)
elpa2_print_kernels@SUFFIX@_LDADD = $(build_lib)

if WANT_SINGLE_PRECISION_REAL
elpa1_test_real_single_precision@SUFFIX@_SOURCES = test/fortran_test_programs/test_real_single_precision.F90 $(shared_sources) $(redirect_sources)
elpa1_test_real_single_precision@SUFFIX@_LDADD = $(build_lib)

elpa2_test_real_single_precision@SUFFIX@_SOURCES = test/fortran_test_programs/test_real2_single_precision.F90 $(shared_sources) $(redirect_sources)
elpa2_test_real_single_precision@SUFFIX@_LDFLAGS = -static
elpa2_test_real_single_precision@SUFFIX@_LDADD = $(build_lib)

elpa2_test_real_default_kernel_single_precision@SUFFIX@_SOURCES = test/fortran_test_programs/test_real2_default_kernel_single_precision.F90 $(shared_sources) $(redirect_sources)
elpa2_test_real_default_kernel_single_precision@SUFFIX@_LDADD = $(build_lib)
#elpa2_test_real_default_kernel_single_precision@SUFFIX@_LDFLAGS = -static

elpa2_test_real_default_kernel_qr_decomposition_single_precision@SUFFIX@_SOURCES = test/fortran_test_programs/test_real2_default_kernel_qr_decomposition_single_precision.F90 $(shared_sources) $(redirect_sources)
elpa2_test_real_default_kernel_qr_decomposition_single_precision@SUFFIX@_LDADD = $(build_lib)

elpa2_test_real_choose_kernel_with_api_single_precision@SUFFIX@_SOURCES = test/fortran_test_programs/test_real2_choose_kernel_with_api_single_precision.F90 $(shared_sources) $(redirect_sources)
elpa2_test_real_choose_kernel_with_api_single_precision@SUFFIX@_LDADD = $(build_lib)
endif

if WANT_SINGLE_PRECISION_COMPLEX
elpa1_test_complex_single_precision@SUFFIX@_SOURCES = test/fortran_test_programs/test_complex_single_precision.F90 $(shared_sources) $(redirect_sources)
elpa1_test_complex_single_precision@SUFFIX@_LDADD = $(build_lib)

elpa2_test_complex_single_precision@SUFFIX@_SOURCES = test/fortran_test_programs/test_complex2_single_precision.F90 $(shared_sources) $(redirect_sources)
elpa2_test_complex_single_precision@SUFFIX@_LDADD = $(build_lib)

elpa2_test_complex_default_kernel_single_precision@SUFFIX@_SOURCES = test/fortran_test_programs/test_complex2_default_kernel_single_precision.F90 $(shared_sources) $(redirect_sources)
elpa2_test_complex_default_kernel_single_precision@SUFFIX@_LDADD = $(build_lib)
#elpa2_test_complex_default_kernel_single_precision@SUFFIX@_LDFLAGS = -static

elpa2_test_complex_choose_kernel_with_api_single_precision@SUFFIX@_SOURCES = test/fortran_test_programs/test_complex2_choose_kernel_with_api_single_precision.F90 $(shared_sources) $(redirect_sources)
elpa2_test_complex_choose_kernel_with_api_single_precision@SUFFIX@_LDADD = $(build_lib)
endif

if WITH_GPU_VERSION
elpa2_test_real_gpu_version@SUFFIX@_SOURCES = test/fortran_test_programs/test_real2_gpu_version.F90 $(shared_sources) $(redirect_sources)
elpa2_test_real_gpu_version@SUFFIX@_LDADD = $(build_lib)

elpa2_test_complex_gpu_version@SUFFIX@_SOURCES = test/fortran_test_programs/test_complex2_gpu_version.F90 $(shared_sources) $(redirect_sources)
elpa2_test_complex_gpu_version@SUFFIX@_LDADD = $(build_lib)

if WANT_SINGLE_PRECISION_REAL
elpa2_test_real_gpu_version_single_precision@SUFFIX@_SOURCES = test/fortran_test_programs/test_real2_gpu_version_single_precision.F90 $(shared_sources) $(redirect_sources)
elpa2_test_real_gpu_version_single_precision@SUFFIX@_LDADD = $(build_lib)
endif

if WANT_SINGLE_PRECISION_COMPLEX
elpa2_test_complex_gpu_version_single_precision@SUFFIX@_SOURCES = test/fortran_test_programs/test_complex2_gpu_version_single_precision.F90 $(shared_sources) $(redirect_sources)
elpa2_test_complex_gpu_version_single_precision@SUFFIX@_LDADD = $(build_lib)
endif

endif

check_SCRIPTS = \
  elpa1_test_real.sh \
  elpa1_test_real_with_c.sh \
  elpa1_test_real_c_version.sh \
  elpa1_test_complex_c_version.sh \
  elpa2_test_real_c_version.sh \
  elpa2_test_complex_c_version.sh \
  elpa2_test_real.sh \
  elpa2_test_real_default_kernel.sh \
  elpa1_test_complex.sh \
  elpa2_test_complex.sh \
  elpa2_test_complex_default_kernel.sh \
  elpa2_test_real_default_kernel_qr_decomposition.sh \
  elpa2_test_real_choose_kernel_with_api.sh \
  elpa2_test_complex_choose_kernel_with_api.sh \
  elpa2_print_kernels@SUFFIX@

if WANT_SINGLE_PRECISION_REAL
check_SCRIPTS += \
  elpa1_test_real_single_precision.sh \
  elpa2_test_real_single_precision.sh \
  elpa2_test_real_default_kernel_single_precision.sh \
  elpa2_test_real_default_kernel_qr_decomposition_single_precision.sh \
  elpa2_test_real_choose_kernel_with_api_single_precision.sh
endif

if WANT_SINGLE_PRECISION_COMPLEX
check_SCRIPTS += \
  elpa1_test_complex_single_precision.sh \
  elpa2_test_complex_single_precision.sh \
  elpa2_test_complex_default_kernel_single_precision.sh
endif

if WITH_GPU_VERSION
check_SCRIPTS += \
  elpa2_test_real_gpu_version.sh \
  elpa2_test_complex_gpu_version.sh
if WANT_SINGLE_PRECISION_REAL
check_SCRIPTS += \
  elpa2_test_real_gpu_version_single_precision.sh
endif
if WANT_SINGLE_PRECISION_COMPLEX
check_SCRIPTS += \
  elpa2_test_complex_gpu_version_single_precision.sh
endif

endif

TESTS = $(check_SCRIPTS)
if WITH_MPI
elpa1_test_real.sh:
	echo 'mpiexec -n 2 ./elpa1_test_real@SUFFIX@ $$TEST_FLAGS' > elpa1_test_real.sh
	chmod +x elpa1_test_real.sh

elpa1_test_real_with_c.sh:
	echo 'mpiexec -n 2 ./elpa1_test_real_with_c@SUFFIX@ $$TEST_FLAGS' > elpa1_test_real_with_c.sh
	chmod +x elpa1_test_real_with_c.sh

elpa2_test_real_c_version.sh:
	echo 'mpiexec -n 2 ./elpa2_test_real_c_version@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real_c_version.sh
	chmod +x elpa2_test_real_c_version.sh

elpa2_test_complex_c_version.sh:
	echo 'mpiexec -n 2 ./elpa2_test_complex_c_version@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex_c_version.sh
	chmod +x elpa2_test_complex_c_version.sh

elpa1_test_real_c_version.sh:
	echo 'mpiexec -n 2 ./elpa1_test_real_c_version@SUFFIX@ $$TEST_FLAGS' > elpa1_test_real_c_version.sh
	chmod +x elpa1_test_real_c_version.sh

elpa1_test_complex_c_version.sh:
	echo 'mpiexec -n 2 ./elpa1_test_complex_c_version@SUFFIX@ $$TEST_FLAGS' > elpa1_test_complex_c_version.sh
	chmod +x elpa1_test_complex_c_version.sh

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

if WANT_SINGLE_PRECISION_REAL
elpa1_test_real_single_precision.sh:
	echo 'mpiexec -n 2 ./elpa1_test_real_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa1_test_real_single_precision.sh
	chmod +x elpa1_test_real_single_precision.sh

elpa2_test_real_single_precision.sh:
	echo 'mpiexec -n 2 ./elpa2_test_real_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real_single_precision.sh
	chmod +x elpa2_test_real_single_precision.sh

elpa2_test_real_default_kernel_single_precision.sh:
	echo 'mpiexec -n 2 ./elpa2_test_real_default_kernel_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real_default_kernel_single_precision.sh
	chmod +x elpa2_test_real_default_kernel_single_precision.sh

elpa2_test_real_default_kernel_qr_decomposition_single_precision.sh:
	echo 'mpiexec -n 2 ./elpa2_test_real_default_kernel_qr_decomposition_single_precision@SUFFIX@' > elpa2_test_real_default_kernel_qr_decomposition_single_precision.sh
	chmod +x elpa2_test_real_default_kernel_qr_decomposition_single_precision.sh

elpa2_test_real_choose_kernel_with_api_single_precision.sh:
	echo 'mpiexec -n 2 ./elpa2_test_real_choose_kernel_with_api_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real_choose_kernel_with_api_single_precision.sh
	chmod +x elpa2_test_real_choose_kernel_with_api_single_precision.sh
endif

if WANT_SINGLE_PRECISION_COMPLEX
elpa1_test_complex_single_precision.sh:
	echo 'mpiexec -n 2 ./elpa1_test_complex_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa1_test_complex_single_precision.sh
	chmod +x elpa1_test_complex_single_precision.sh

elpa2_test_complex_single_precision.sh:
	echo 'mpiexec -n 2 ./elpa2_test_complex_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex_single_precision.sh
	chmod +x elpa2_test_complex_single_precision.sh

elpa2_test_complex_default_kernel_single_precision.sh:
	echo 'mpiexec -n 2 ./elpa2_test_complex_default_kernel_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex_default_kernel_single_precision.sh
	chmod +x elpa2_test_complex_default_kernel_single_precision.sh

elpa2_test_complex_choose_kernel_with_api_single_precision.sh:
	echo 'mpiexec -n 2 ./elpa2_test_complex_choose_kernel_with_api_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex_choose_kernel_with_api_single_precision.sh
	chmod +x elpa2_test_complex_choose_kernel_with_api_single_precision.sh
endif

if WITH_GPU_VERSION
elpa2_test_real_gpu_version.sh:
	echo 'mpiexec -n 2 ./elpa2_test_real_gpu_version@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real_gpu_version.sh
	chmod +x elpa2_test_real_gpu_version.sh

elpa2_test_complex_gpu_version.sh:
	echo 'mpiexec -n 2 ./elpa2_test_complex_gpu_version@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex_gpu_version.sh
	chmod +x elpa2_test_complex_gpu_version.sh

if WANT_SINGLE_PRECISION_REAL
elpa2_test_real_gpu_version_single_precision.sh:
	echo 'mpiexec -n 2 ./elpa2_test_real_gpu_version_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real_gpu_version_single_precision.sh
	chmod +x elpa2_test_real_gpu_version_single_precision.sh
endif

if WANT_SINGLE_PRECISION_COMPLEX
elpa2_test_complex_gpu_version_single_precision.sh:
	echo 'mpiexec -n 2 ./elpa2_test_complex_gpu_version_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex_gpu_version_single_precision.sh
	chmod +x elpa2_test_complex_gpu_version_single_precision.sh
endif
# GPU_VERSION
endif

else
# build tests without mpi support

elpa1_test_real.sh:
	echo './elpa1_test_real@SUFFIX@ $$TEST_FLAGS' > elpa1_test_real.sh
	chmod +x elpa1_test_real.sh

elpa1_test_real_with_c.sh:
	echo './elpa1_test_real_with_c@SUFFIX@ $$TEST_FLAGS' > elpa1_test_real_with_c.sh
	chmod +x elpa1_test_real_with_c.sh

elpa2_test_real_c_version.sh:
	echo './elpa2_test_real_c_version@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real_c_version.sh
	chmod +x elpa2_test_real_c_version.sh

elpa2_test_complex_c_version.sh:
	echo './elpa2_test_complex_c_version@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex_c_version.sh
	chmod +x elpa2_test_complex_c_version.sh

elpa1_test_real_c_version.sh:
	echo './elpa1_test_real_c_version@SUFFIX@ $$TEST_FLAGS' > elpa1_test_real_c_version.sh
	chmod +x elpa1_test_real_c_version.sh

elpa1_test_complex_c_version.sh:
	echo './elpa1_test_complex_c_version@SUFFIX@ $$TEST_FLAGS' > elpa1_test_complex_c_version.sh
	chmod +x elpa1_test_complex_c_version.sh

elpa2_test_real.sh:
	echo './elpa2_test_real@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real.sh
	chmod +x elpa2_test_real.sh

elpa2_test_real_default_kernel.sh:
	echo './elpa2_test_real_default_kernel@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real_default_kernel.sh
	chmod +x elpa2_test_real_default_kernel.sh

elpa2_test_real_default_kernel_qr_decomposition.sh:
	echo './elpa2_test_real_default_kernel_qr_decomposition@SUFFIX@' > elpa2_test_real_default_kernel_qr_decomposition.sh
	chmod +x elpa2_test_real_default_kernel_qr_decomposition.sh

elpa2_test_real_choose_kernel_with_api.sh:
	echo './elpa2_test_real_choose_kernel_with_api@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real_choose_kernel_with_api.sh
	chmod +x elpa2_test_real_choose_kernel_with_api.sh

elpa1_test_complex.sh:
	echo './elpa1_test_complex@SUFFIX@ $$TEST_FLAGS' > elpa1_test_complex.sh
	chmod +x elpa1_test_complex.sh

elpa2_test_complex.sh:
	echo './elpa2_test_complex@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex.sh
	chmod +x elpa2_test_complex.sh

elpa2_test_complex_default_kernel.sh:
	echo './elpa2_test_complex_default_kernel@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex_default_kernel.sh
	chmod +x elpa2_test_complex_default_kernel.sh

elpa2_test_complex_choose_kernel_with_api.sh:
	echo './elpa2_test_complex_choose_kernel_with_api@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex_choose_kernel_with_api.sh
	chmod +x elpa2_test_complex_choose_kernel_with_api.sh

if WANT_SINGLE_PRECISION_REAL
elpa1_test_real_single_precision.sh:
	echo './elpa1_test_real_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa1_test_real_single_precision.sh
	chmod +x elpa1_test_real_single_precision.sh

elpa2_test_real_single_precision.sh:
	echo './elpa2_test_real_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real_single_precision.sh
	chmod +x elpa2_test_real_single_precision.sh

elpa2_test_real_default_kernel_single_precision.sh:
	echo './elpa2_test_real_default_kernel_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real_default_kernel_single_precision.sh
	chmod +x elpa2_test_real_default_kernel_single_precision.sh

elpa2_test_real_default_kernel_qr_decomposition_single_precision.sh:
	echo './elpa2_test_real_default_kernel_qr_decomposition_single_precision@SUFFIX@' > elpa2_test_real_default_kernel_qr_decomposition_single_precision.sh
	chmod +x elpa2_test_real_default_kernel_qr_decomposition_single_precision.sh

elpa2_test_real_choose_kernel_with_api_single_precision.sh:
	echo './elpa2_test_real_choose_kernel_with_api_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real_choose_kernel_with_api_single_precision.sh
	chmod +x elpa2_test_real_choose_kernel_with_api_single_precision.sh
endif

if WANT_SINGLE_PRECISION_COMPLEX
elpa1_test_complex_single_precision.sh:
	echo './elpa1_test_complex_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa1_test_complex_single_precision.sh
	chmod +x elpa1_test_complex_single_precision.sh

elpa2_test_complex_single_precision.sh:
	echo './elpa2_test_complex_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex_single_precision.sh
	chmod +x elpa2_test_complex_single_precision.sh

elpa2_test_complex_default_kernel_single_precision.sh:
	echo './elpa2_test_complex_default_kernel_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex_default_kernel_single_precision.sh
	chmod +x elpa2_test_complex_default_kernel_single_precision.sh

elpa2_test_complex_choose_kernel_with_api_single_precision.sh:
	echo './elpa2_test_complex_choose_kernel_with_api_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex_choose_kernel_with_api_single_precision.sh
	chmod +x elpa2_test_complex_choose_kernel_with_api_single_precision.sh
endif

if WITH_GPU_VERSION
elpa2_test_real_gpu_version.sh:
	echo './elpa2_test_real_gpu_version@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real_gpu_version.sh
	chmod +x elpa2_test_real_gpu_version.sh

elpa2_test_complex_gpu_version.sh:
	echo './elpa2_test_complex_gpu_version@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex_gpu_version.sh
	chmod +x elpa2_test_complex_gpu_version.sh

if WANT_SINGLE_PRECISION_REAL
elpa2_test_real_gpu_version_single_precision.sh:
	echo './elpa2_test_real_gpu_version_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa2_test_real_gpu_version_single_precision.sh
	chmod +x elpa2_test_real_gpu_version_single_precision.sh
endif

if WANT_SINGLE_PRECISION_COMPLEX
elpa2_test_complex_gpu_version_single_precision.sh:
	echo './elpa2_test_complex_gpu_version_single_precision@SUFFIX@ $$TEST_FLAGS' > elpa2_test_complex_gpu_version_single_precision.sh
	chmod +x elpa2_test_complex_gpu_version_single_precision.sh
endif
# GPU_VERSION
endif



# use mpi
endif

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

elpa2_kernels_real.i: $(top_srcdir)/src/elpa2_kernels/elpa2_kernels_real.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/elpa2_kernels/elpa2_kernels_real.F90 -o $@

mod_compute_hh_trafo_real.i: $(top_srcdir)/src/mod_compute_hh_trafo_real.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/src/mod_compute_hh_trafo_real.F90 -o $@

test_real.i: $(top_srcdir)/test/fortran_test_programs/test_real.F90
	$(CPP) $(CPPFLAGS) -I$(top_builddir)/ -c $(top_srcdir)/test/fortran_test_programs/test_real.F90 -o $@

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
  elpa/elpa-generated.h \
  elpa1_test_real.sh \
  elpa1_test_complex.sh \
  elpa2_test_real.sh \
  elpa2_test_real_default_kernel.sh \
  elpa2_test_real_default_kernel_qr_decomposition.sh \
  elpa2_test_complex.sh \
  elpa2_test_complex_default_kernel.sh \
  elpa2_test_real_choose_kernel_with_api.sh \
  elpa2_test_complex_choose_kernel_with_api.sh \
  elpa1_test_real_with_c.sh \
  elpa1_test_real_c_version.sh \
  elpa1_test_complex_c_version.sh \
  elpa2_test_real_c_version.sh \
  elpa2_test_complex_c_version.sh \
  *.i

clean-local:
	-rm -rf modules/* .fortran_dependencies/*

distclean-local:
	-rm config-f90.h
	-rm -rf ./src/elpa2_kernels/.deps
	-rm -rf ./src/.deps
	-rm -rf ./test/.deps
	-rm -rf ./elpa/elpa_generated.h
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
  src/elpa1_compute_template_real.X90 \
  src/elpa1_compute_template_complex.X90 \
  src/elpa2_compute_template_real.X90 \
  src/elpa2_compute_template_complex.X90 \
  src/redist_band.X90 \
  src/elpa_qr/elpa_qrkernels.X90 \
  src/ev_tridi_band_gpu_c_v2_complex_template.Xcu \
  src/ev_tridi_band_gpu_c_v2_real_template.Xcu \
  src/cuUtils_complex_template.Xcu \
  src/cuUtils_real_template.Xcu \
  nvcc_wrap

# Rules to re-generated the headers
elpa/elpa_generated.h: $(top_srcdir)/src/elpa_c_interface.F90
	grep -h "^ *!c>" $^ | sed 's/^ *!c>//;' > $@ || { rm $@; exit 1; }

test/shared_sources/generated.h: $(wildcard $(top_srcdir)/test/shared_sources/*.F90)
	grep -h "^ *!c>" $^ | sed 's/^ *!c>//;' > $@ || { rm $@; exit 1; }

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
