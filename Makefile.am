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

# internal parts
noinst_LTLIBRARIES += libelpa@SUFFIX@_private.la
libelpa@SUFFIX@_private_la_FCFLAGS = $(AM_FCFLAGS) $(FC_MODOUT)private_modules $(FC_MODINC)private_modules
libelpa@SUFFIX@_private_la_SOURCES = \
  src/elpa_impl.F90 \
  src/elpa_autotune_impl.F90 \
  src/elpa_abstract_impl.F90 \
  src/helpers/mod_precision.F90 \
  src/helpers/mod_blas_interfaces.F90 \
  src/helpers/mod_scalapack_interfaces.F90 \
  src/helpers/mod_mpi.F90 \
  src/helpers/mod_mpi_stubs.F90 \
  src/helpers/mod_omp.F90 \
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
  src/GPU/mod_vendor_agnostic_layer.F90 \
  src/GPU/CUDA/mod_cuda.F90 \
  src/GPU/INTEL/mod_mkl_offload.F90 \
  src/GPU/ROCm/mod_hip.F90 \
  src/elpa2/GPU/interface_c_gpu_kernel.F90 \
  src/elpa2/GPU/CUDA/interface_c_cuda_kernel.F90 \
  src/elpa2/GPU/ROCm/interface_c_hip_kernel.F90 \
  src/elpa2/mod_pack_unpack_gpu.F90 \
  src/elpa2/qr/qr_utils.F90 \
  src/elpa2/qr/elpa_qrkernels.F90 \
  src/elpa2/qr/elpa_pdlarfb.F90 \
  src/elpa2/qr/elpa_pdgeqrf.F90 \
  src/elpa1/elpa1.F90 \
  src/elpa2/elpa2.F90 \
  src/elpa_generalized/cannon.c \
  src/helpers/matrix_plot.F90 \
  src/general/mod_elpa_skewsymmetric_blas.F90 \
  src/solve_tridi/mod_global_product.F90 \
  src/solve_tridi/mod_global_gather.F90 \
  src/solve_tridi/mod_resort_ev.F90 \
  src/solve_tridi/mod_transform_columns.F90 \
  src/solve_tridi/mod_check_monotony.F90 \
  src/solve_tridi/mod_add_tmp.F90 \
  src/solve_tridi/mod_merge_systems.F90 \
  src/solve_tridi/mod_merge_recursive.F90 \
  src/solve_tridi/mod_solve_tridi.F90 \
  src/elpa1/mod_distribute_global_column.F90 \
  src/elpa1/mod_v_add_s.F90 \
  src/elpa1/mod_solve_secular_equation.F90 \
  src/helpers/mod_thread_affinity.F90 \
  src/helpers/check_thread_affinity.c \
  src/elpa_index.c

libelpa@SUFFIX@_private_la_SOURCES += src/elpa_c_interface.c


libelpa@SUFFIX@_private_la_SOURCES += \
  src/general/elpa_utilities.F90

EXTRA_libelpa@SUFFIX@_private_la_DEPENDENCIES = \
  src/elpa1/elpa_reduce_add_vectors.F90 \
  src/elpa1/elpa_transpose_vectors.F90 \
  src/elpa_api_math_template.F90 \
  src/elpa_impl_math_template.F90 \
  src/elpa_impl_generalized_transform_template.F90 \
  src/elpa1/elpa1_compute_template.F90 \
  src/elpa2/elpa2_compute_real_template.F90 \
  src/elpa2/elpa2_compute_complex_template.F90 \
  src/helpers/elpa_redistribute_template.F90 \
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
  src/elpa2/kernels/simple_block4_template.F90 \
  src/elpa2/kernels/simple_block6_template.F90 \
  src/elpa2/pack_unpack_cpu.F90 \
  src/elpa2/pack_unpack_gpu.F90 \
  src/elpa2/compute_hh_trafo.F90 \
  src/elpa2/redist_band.F90 \
  src/general/sanity.F90 \
  src/general/error_checking.inc \
  src/elpa1/elpa_cholesky_template.F90 \
  src/elpa1/elpa_invert_trm.F90 \
  src/elpa1/elpa_multiply_a_b.F90 \
  src/elpa1/elpa_solve_tridi_impl_public.F90 \
  src/general/elpa_ssr2_template.F90 \
  src/general/elpa_ssmv_template.F90 \
  src/general/precision_macros.h \
  src/general/precision_typedefs.h \
  src/general/precision_kinds.F90

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

else
libelpa@SUFFIX@_private_la_SOURCES += \
  src/helpers/timer_dummy.F90
endif

if WITH_NVIDIA_GPU_VERSION
  libelpa@SUFFIX@_private_la_SOURCES += src/GPU/CUDA/elpa_index_nvidia_gpu.cu src/GPU/CUDA/cudaFunctions.cu src/GPU/CUDA/cuUtils.cu src/elpa2/GPU/CUDA/ev_tridi_band_nvidia_gpu_real.cu src/elpa2/GPU/CUDA/ev_tridi_band_nvidia_gpu_complex.cu
endif

if WITH_INTEL_GPU_VERSION
  libelpa@SUFFIX@_private_la_SOURCES += src/GPU/INTEL/mkl_offload.cpp
endif

if WITH_AMD_GPU_VERSION
  libelpa@SUFFIX@_private_la_SOURCES += src/GPU/ROCm/elpa_index_amd_gpu.cpp src/GPU/ROCm/rocmFunctions.cpp src/GPU/ROCm/hipUtils.cpp src/elpa2/GPU/ROCm/ev_tridi_band_amd_gpu_real.cpp src/elpa2/GPU/ROCm/ev_tridi_band_amd_gpu_complex.cpp
endif

if !WITH_MPI
  libelpa@SUFFIX@_private_la_SOURCES += src/helpers/mod_time_c.F90
if !HAVE_DETAILED_TIMINGS
  libelpa@SUFFIX@_private_la_SOURCES += src/ftimings/time.c
endif
endif


if HAVE_HETEROGENOUS_CLUSTER_SUPPORT
  libelpa@SUFFIX@_private_la_SOURCES +=  src/helpers/get_cpuid_set.c src/helpers/mod_simd_kernel.F90
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

if WITH_REAL_GENERIC_SIMPLE_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_simple_block4.F90
endif

if WITH_REAL_GENERIC_SIMPLE_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_simple_block6.F90
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

if WITH_REAL_SPARC64_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sparc64_2hv_double_precision.c
#if WANT_SINGLE_PRECISION_REAL
#  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sparc64_2hv_single_precision.c
#endif
endif

if WITH_REAL_NEON_ARCH64_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_neon_arch64_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_neon_arch64_2hv_single_precision.c
endif
endif

if WITH_REAL_VSX_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_vsx_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_vsx_2hv_single_precision.c
endif
endif

if WITH_REAL_SSE_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sse_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sse_2hv_single_precision.c
endif
endif

if WITH_REAL_AVX_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx_2hv_single_precision.c
endif
endif

if WITH_REAL_AVX2_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx2_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx2_2hv_single_precision.c
endif
endif

if WITH_REAL_AVX512_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx512_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx512_2hv_single_precision.c
endif
endif

if WITH_REAL_SVE128_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve128_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve128_2hv_single_precision.c
endif
endif

if WITH_REAL_SVE256_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve256_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve256_2hv_single_precision.c
endif
endif

if WITH_REAL_SVE512_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve512_2hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve512_2hv_single_precision.c
endif
endif

if WITH_REAL_SPARC64_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sparc64_4hv_double_precision.c
#if WANT_SINGLE_PRECISION_REAL
#  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sparc64_4hv_single_precision.c
#endif
endif

if WITH_REAL_NEON_ARCH64_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_neon_arch64_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_neon_arch64_4hv_single_precision.c
endif
endif

if WITH_REAL_VSX_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_vsx_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_vsx_4hv_single_precision.c
endif
endif

if WITH_REAL_SSE_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sse_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sse_4hv_single_precision.c
endif
endif

if WITH_REAL_AVX_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx_4hv_single_precision.c
endif
endif

if WITH_REAL_AVX2_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx2_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx2_4hv_single_precision.c
endif
endif

if WITH_REAL_AVX512_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx512_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx512_4hv_single_precision.c
endif
endif

if WITH_REAL_SVE128_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve128_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve128_4hv_single_precision.c
endif
endif

if WITH_REAL_SVE256_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve256_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve256_4hv_single_precision.c
endif
endif

if WITH_REAL_SVE512_BLOCK4_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve512_4hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve512_4hv_single_precision.c
endif
endif

if WITH_REAL_SPARC64_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sparc64_6hv_double_precision.c
#if WANT_SINGLE_PRECISION_REAL
#  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sparc64_6hv_single_precision.c
#endif
endif

if WITH_REAL_NEON_ARCH64_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_neon_arch64_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_neon_arch64_6hv_single_precision.c
endif
endif

if WITH_REAL_VSX_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_vsx_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_vsx_6hv_single_precision.c
endif
endif

if WITH_REAL_SSE_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sse_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sse_6hv_single_precision.c
endif
endif

if WITH_REAL_AVX_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx_6hv_single_precision.c
endif
endif

if WITH_REAL_AVX2_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx2_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx2_6hv_single_precision.c
endif
endif

if WITH_REAL_AVX512_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx512_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_avx512_6hv_single_precision.c
endif
endif

if WITH_REAL_SVE128_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve128_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve128_6hv_single_precision.c
endif
endif

if WITH_REAL_SVE256_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve256_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve256_6hv_single_precision.c
endif
endif

if WITH_REAL_SVE512_BLOCK6_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve512_6hv_double_precision.c
if WANT_SINGLE_PRECISION_REAL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/real_sve512_6hv_single_precision.c
endif
endif

#if WITH_COMPLEX_SPARC64_BLOCK1_KERNEL
#  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sparc64_1hv_double_precision.c
#if WANT_SINGLE_PRECISION_COMPLEX
#  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sparc64_1hv_single_precision.c
#endif
#endif
#
#if WITH_COMPLEX_VSX_BLOCK1_KERNEL
#  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_vsx_1hv_double_precision.c
#if WANT_SINGLE_PRECISION_COMPLEX
#  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_vsx_1hv_single_precision.c
#endif
#endif

if WITH_COMPLEX_SSE_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sse_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sse_1hv_single_precision.c
endif
endif

if WITH_COMPLEX_NEON_ARCH64_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_neon_arch64_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_neon_arch64_1hv_single_precision.c
endif
endif

if WITH_COMPLEX_AVX_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx_1hv_single_precision.c
endif
endif

if WITH_COMPLEX_AVX2_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx2_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx2_1hv_single_precision.c
endif
endif

if WITH_COMPLEX_AVX512_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx512_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx512_1hv_single_precision.c
endif
endif

if WITH_COMPLEX_SVE128_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sve128_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sve128_1hv_single_precision.c
endif
endif

if WITH_COMPLEX_SVE256_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sve256_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sve256_1hv_single_precision.c
endif
endif

if WITH_COMPLEX_SVE512_BLOCK1_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sve512_1hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sve512_1hv_single_precision.c
endif
endif

#if WITH_COMPLEX_SPARC64_BLOCK2_KERNEL
#  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sparc64_2hv_double_precision.c
#if WANT_SINGLE_PRECISION_COMPLEX
#  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sparc64_2hv_single_precision.c
#endif
#endif
#
#if WITH_COMPLEX_VSX_BLOCK2_KERNEL
#  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_vsx_2hv_double_precision.c
#if WANT_SINGLE_PRECISION_COMPLEX
#  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_vsx_2hv_single_precision.c
#endif
#endif

if WITH_COMPLEX_SSE_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sse_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sse_2hv_single_precision.c
endif
endif

if WITH_COMPLEX_NEON_ARCH64_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_neon_arch64_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_neon_arch64_2hv_single_precision.c
endif
endif

if WITH_COMPLEX_AVX_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx_2hv_single_precision.c
endif
endif

if WITH_COMPLEX_AVX2_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx2_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx2_2hv_single_precision.c
endif
endif

if WITH_COMPLEX_AVX512_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx512_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_avx512_2hv_single_precision.c
endif
endif

if WITH_COMPLEX_SVE128_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sve128_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sve128_2hv_single_precision.c
endif
endif

if WITH_COMPLEX_SVE256_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sve256_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sve256_2hv_single_precision.c
endif
endif

if WITH_COMPLEX_SVE512_BLOCK2_KERNEL
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sve512_2hv_double_precision.c
if WANT_SINGLE_PRECISION_COMPLEX
  libelpa@SUFFIX@_private_la_SOURCES += src/elpa2/kernels/complex_sve512_2hv_single_precision.c
endif
endif


if STORE_BUILD_CONFIG
  libelpa@SUFFIX@_private_la_SOURCES += src/helpers/print_build_config.c
endif

# rule for ROCm files

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
  src/helpers/lapack_interfaces.h \
  src/helpers/scalapack_interfaces.h \
  elpa/elpa_simd_constants.h \
  elpa/elpa.h \
  elpa/elpa_generic.h

nobase_nodist_elpa_include_HEADERS = \
  elpa/elpa_version.h \
  elpa/elpa_constants.h \
  elpa/elpa_generated.h \
  elpa/elpa_generated_c_api.h

dist_man_MANS = \
  man/elpa2_print_kernels.1 \
  man/elpa_init.3 \
  man/elpa_allocate.3 \
  man/elpa_set.3 \
  man/elpa_setup.3 \
  man/elpa_eigenvalues.3 \
  man/elpa_eigenvectors.3 \
  man/elpa_skew_eigenvalues.3 \
  man/elpa_skew_eigenvectors.3 \
  man/elpa_generalized_eigenvectors.3 \
  man/elpa_generalized_eigenvalues.3 \
  man/elpa_cholesky.3 \
  man/elpa_invert_triangular.3 \
  man/elpa_solve_tridiagonal.3 \
  man/elpa_hermitian_multiply.3 \
  man/elpa_deallocate.3 \
  man/elpa_load_settings.3 \
  man/elpa_store_settings.3 \
  man/elpa_print_settings.3 \
  man/elpa_autotune_save_state.3 \
  man/elpa_autotune_load_state.3 \
  man/elpa_autotune_print_state.3 \
  man/elpa_autotune_setup.3 \
  man/elpa_autotune_step.3 \
  man/elpa_autotune_set_best.3 \
  man/elpa_autotune_deallocate.3 \
  man/elpa_uninit.3

dist_doc_DATA = README.md USERS_GUIDE.md USERS_GUIDE_DEPRECATED_LEGACY_API.md INSTALL.md CONTRIBUTING.md LICENSE Changelog COPYING/COPYING COPYING/gpl.txt COPYING/lgpl.txt

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
  test/shared/tests_variable_definitions.F90 \
  test/shared/mod_tests_scalapack_interfaces.F90 \
  test/shared/mod_tests_blas_interfaces.F90 \
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

#include test_programs_manual.am

noinst_PROGRAMS += validate_double_instance@SUFFIX@
check_SCRIPTS += validate_double_instance@SUFFIX@_default.sh
validate_double_instance@SUFFIX@_SOURCES = test/Fortran/elpa2/double_instance.F90
validate_double_instance@SUFFIX@_LDADD = $(test_program_ldadd)
validate_double_instance@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODINC)test_modules $(FC_MODINC)modules $(FC_MODINC)private_modules

noinst_PROGRAMS += validate_real_2stage_banded@SUFFIX@
check_SCRIPTS += validate_real_2stage_banded@SUFFIX@_default.sh
validate_real_2stage_banded@SUFFIX@_SOURCES = test/Fortran/elpa2/real_2stage_banded.F90 
validate_real_2stage_banded@SUFFIX@_LDADD = $(test_program_ldadd)
validate_real_2stage_banded@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODINC)test_modules $(FC_MODINC)modules $(FC_MODINC)private_modules

noinst_PROGRAMS += validate_complex_2stage_banded@SUFFIX@
check_SCRIPTS += validate_complex_2stage_banded@SUFFIX@_default.sh
validate_complex_2stage_banded@SUFFIX@_SOURCES = test/Fortran/elpa2/complex_2stage_banded.F90
validate_complex_2stage_banded@SUFFIX@_LDADD = $(test_program_ldadd)
validate_complex_2stage_banded@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODINC)test_modules $(FC_MODINC)modules $(FC_MODINC)private_modules

if WANT_SINGLE_PRECISION_REAL
noinst_PROGRAMS += validate_single_real_2stage_banded@SUFFIX@
check_SCRIPTS += validate_single_real_2stage_banded@SUFFIX@_default.sh
validate_single_real_2stage_banded@SUFFIX@_SOURCES = test/Fortran/elpa2/single_real_2stage_banded.F90
validate_single_real_2stage_banded@SUFFIX@_LDADD = $(test_program_ldadd)
validate_single_real_2stage_banded@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODINC)test_modules $(FC_MODINC)modules $(FC_MODINC)private_modules
endif

if WANT_SINGLE_PRECISION_COMPLEX
noinst_PROGRAMS += validate_single_complex_2stage_banded@SUFFIX@
check_SCRIPTS += validate_single_complex_2stage_banded@SUFFIX@_default.sh
validate_single_complex_2stage_banded@SUFFIX@_SOURCES = test/Fortran/elpa2/single_complex_2stage_banded.F90
validate_single_complex_2stage_banded@SUFFIX@_LDADD = $(test_program_ldadd)
validate_single_complex_2stage_banded@SUFFIX@_FCFLAGS = $(AM_FCFLAGS) $(FC_MODINC)test_modules $(FC_MODINC)modules $(FC_MODINC)private_modules
endif

# python wrapper
pyelpadir = $(pythondir)/pyelpa
if WITH_PYTHON
pyelpa_PYTHON = python/pyelpa/__init__.py python/pyelpa/distributedmatrix.py
pyelpa_LTLIBRARIES = wrapper.la
else
pyelpa_PYTHON =
pyelpa_LTLIBRARIES =
endif
nodist_wrapper_la_SOURCES = python/pyelpa/wrapper.c
wrapper_la_LDFLAGS = -module -avoid-version -shared $(AM_LDFLAGS)
wrapper_la_LIBADD = libelpa@SUFFIX@.la
wrapper_la_CFLAGS = $(PYTHON_INCLUDE) $(NUMPY_INCLUDE) $(AM_CFLAGS)

python/pyelpa/wrapper.c: python/pyelpa/wrapper.pyx
	cython $< -o $@
# test scripts
TASKS ?= 2
if WITH_MPI
  wrapper=$(MPI_BINARY) -n $${TASKS:-$(TASKS)}
else
  wrapper=
endif
TESTS = $(check_SCRIPTS)
%_extended.sh: %
	@echo "#!/bin/bash" > $@
	@echo 'if [ "$$CHECK_LEVEL" = "extended" ] ; then $(wrapper) ./$^ $$TEST_FLAGS ; else exit 77; fi' >> $@
	@chmod +x $@

%_default.sh: %
	@echo "#!/bin/bash" > $@
	@echo '$(wrapper)' ./$^ '$$TEST_FLAGS' >> $@
	@chmod +x $@

if WITH_PYTHON_TESTS
check_SCRIPTS += test_python.sh
endif

test_python.sh:
	@echo '#!/bin/bash' > $@
# this is kind of hacky... is there a better way to get wrapper.so?
	@echo 'export PYTHONPATH=./python-copy:$$PYTHONPATH' >> $@
	@echo 'cp -r $(abs_top_srcdir)/python python-copy || exit 1' >> $@
	@echo 'chmod u+rwX -R python-copy || exit 1' >> $@
	@echo 'cp .libs/wrapper.so python-copy/pyelpa/ || exit 1' >> $@
# the dlopen flags are needed for MKL to work properly...
# only in os from python 3.3 on
	@echo "$(wrapper) $(PYTHON) -c 'import sys, os; sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL); import pytest; sys.exit(pytest.main([\"./python-copy\", \"-p\", \"no:cacheprovider\"]))'" >> $@
	@echo 'exit_code=$$?' >> $@
	@echo 'rm -rf python-copy || exit 1' >> $@
	@echo 'exit $$exit_code' >> $@
	@chmod +x $@

include doxygen.am

CLEANFILES = \
  elpa_generated.h \
  elpa_generated_c_api.h \
  elpa1_test* \
  elpa2_test*\
  elpa2_real* \
  elpa1_real* \
  elpa*.sh \
  test*.sh \
  single_real* \
  single_complex* \
  real* \
  complex* \
  double_instance* \
  *.i \
  python/pyelpa/wrapper.c \
  check_python.sh

clean-local:
	-rm -rf modules/* private_modules/* test_modules/* .fortran_dependencies/*
	-rm -rf validate_*.sh
	-rm -rf test_*.sh
	-rm -rf real_2stage*.sh
	-rm -rf complex_2stage*.sh
	-rm -rf single_complex_2stage*.sh
	-rm -rf single_real_2stage*.sh
	-rm -rf double_instance_onenode*.sh
	-rm -rf test_*.sh
	-rm -rf $(generated_headers)

distclean-local:
	-rm -rf ./m4
	-rm -rf ./src
	-rm -rf ./test
	-rm -rf ./modules
	-rm -rf ./private_modules
	-rm -rf ./test_modules
	-rm -rf ./.fortran-dependencies
	-rm -rf .fortran_dependencies
	-rm config-f90.h
	-rm -rf ./test/.deps
	-rm -rf elpa/elpa_generated_c_api.h

EXTRA_DIST = \
  elpa.spec \
  elpa/elpa.h \
  elpa/elpa_generic.h \
  fdep/fortran_dependencies.mk \
  fdep/fortran_dependencies.pl \
  manual_cpp \
  nvcc_wrap \
  remove_xcompiler \
  src/helpers/fortran_blas_interfaces.F90 \
  src/helpers/fortran_scalapack_interfaces.F90 \
  src/GPU/CUDA/cuUtils_template.cu \
  src/GPU/ROCm/hipUtils_template.cpp \
  src/elpa_api_math_template.F90 \
  src/elpa_impl_math_template.F90 \
  src/helpers/elpa_redistribute_template.F90 \
  src/elpa_impl_generalized_transform_template.F90 \
  src/elpa1/elpa1_compute_template.F90 \
  src/solve_tridi/global_product_template.F90 \
  src/solve_tridi/global_gather_template.F90 \
  src/solve_tridi/resort_ev_template.F90 \
  src/solve_tridi/transform_columns_template.F90 \
  src/solve_tridi/check_monotony_template.F90 \
  src/solve_tridi/add_tmp_template.F90 \
  src/elpa1/v_add_s_template.F90 \
  src/elpa1/solve_secular_equation_template.F90 \
  src/elpa1/distribute_global_column_template.F90 \
  src/solve_tridi/merge_systems_template.F90 \
  src/solve_tridi/merge_recursive_template.F90 \
  src/solve_tridi/solve_tridi_template.F90 \
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
  src/elpa2/kernels/complex_128bit_256bit_512bit_BLOCK_template.c \
  src/elpa2/kernels/complex_template.F90 \
  src/elpa2/kernels/real_128bit_256bit_512bit_BLOCK_template.c \
  src/elpa2/kernels/real_template.F90 \
  src/elpa2/kernels/simple_template.F90 \
  src/elpa2/kernels/simple_block4_template.F90 \
  src/elpa2/kernels/simple_block6_template.F90 \
  src/elpa2/pack_unpack_cpu.F90 \
  src/elpa2/pack_unpack_gpu.F90 \
  src/elpa2/qr/elpa_pdgeqrf_template.F90 \
  src/elpa2/qr/elpa_pdlarfb_template.F90 \
  src/elpa2/qr/elpa_qrkernels_template.F90 \
  src/elpa2/qr/qr_utils_template.F90 \
  src/elpa2/redist_band.F90 \
  src/elpa_generalized/cannon_forw_template.c \
  src/elpa_generalized/cannon_back_template.c \
  src/elpa_index.h \
  src/fortran_constants.h \
  src/general/map_global_to_local.F90 \
  src/general/precision_macros.h \
  src/general/precision_typedefs.h \
  src/general/precision_kinds.F90 \
  test/shared/test_precision_kinds.F90 \
  src/general/prow_pcol.F90 \
  src/general/sanity.F90 \
  src/general/error_checking.inc \
  src/general/elpa_ssr2_template.F90 \
  src/general/elpa_ssmv_template.F90 \
  test/Fortran/assert.h \
  test/Fortran/elpa_print_headers.F90 \
  test/shared/test_check_correctness_template.F90 \
  test/shared/test_prepare_matrix_template.F90 \
  test/shared/test_analytic_template.F90 \
  test/shared/test_scalapack_template.F90 \
  test_project_1stage/Makefile.am \
  test_project_1stage/autogen.sh \
  test_project_1stage/configure.ac \
  test_project_1stage/fdep \
  test_project_1stage/m4 \
  test_project_1stage/src/test_real.F90 \
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

if WITH_SCALAPACK_TESTS
EXTRA_DIST += \
  test/shared/test_scalapack_template.F90
endif

# python wrapper files
EXTRA_DIST += python/pyelpa/__init__.py \
	      python/pyelpa/distributedmatrix.py \
	      python/pyelpa/wrapper.pyx \
	      python/tests/test_elpa_import.py \
	      python/tests/test_mpi4py.py \
	      python/tests/test_numroc.py \
	      python/tests/test_with_mpi.py

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
