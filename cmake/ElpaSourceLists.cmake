# ElpaSourceLists.cmake — Source file lists for the ELPA library
#
# Defines CMake variables with source files grouped by feature condition.
# Consumed by src/CMakeLists.txt to build the ELPA library.

# ===========================================================================
# Public F90 sources (the public API modules)
# ===========================================================================
set(ELPA_PUBLIC_SOURCES src/elpa.F90 src/elpa_api.F90 src/elpa_constants.F90)

# ===========================================================================
# Private unconditional sources (always compiled)
# ===========================================================================
set(ELPA_PRIVATE_SOURCES
    elpa/elpa_explicit_name.c
    src/elpa_impl.F90
    src/elpa_autotune_impl.F90
    src/elpa_abstract_impl.F90
    src/helpers/mod_precision.F90
    src/helpers/mod_blas_interfaces.F90
    src/helpers/mod_scalapack_interfaces.F90
    src/helpers/mod_mpi.F90
    src/helpers/mod_mpi_stubs.F90
    src/helpers/mod_omp.F90
    src/helpers/mod_query_gpu_settings.F90
    src/elpa_generated_fortran_interfaces.F90
    src/elpa2/mod_redist_band.F90
    src/elpa2/mod_pack_unpack_cpu.F90
    src/elpa2/mod_compute_hh_trafo.F90
    src/helpers/aligned_mem.F90
    src/helpers/posix_memalign_compat.c
    src/elpa1/elpa1_compute_private.F90
    src/elpa1/elpa1_auxiliary.F90
    src/elpa1/GPU/mod_tridiag_gpu.F90
    src/elpa1/GPU/mod_trans_ev_gpu.F90
    src/elpa1/GPU/mod_elpa1_gpu.F90
    src/elpa1/GPU/CUDA/mod_elpa1_cuda.F90
    src/elpa1/GPU/ROCm/mod_elpa1_hip.F90
    src/elpa1/GPU/SYCL/mod_elpa1_sycl.F90
    src/solve_tridi/GPU/mod_merge_systems_gpu.F90
    src/solve_tridi/GPU/mod_solve_single_problem_gpu.F90
    src/solve_tridi/GPU/mod_solve_tridi_col_gpu.F90
    src/solve_tridi/GPU/mod_distribute_global_column_gpu.F90
    src/solve_tridi/GPU/mod_transform_columns_gpu.F90
    src/elpa2/elpa2_determine_workload.F90
    src/elpa2/elpa2_compute.F90
    src/elpa2/kernels/mod_single_hh_trafo_real.F90
    src/GPU/mod_gpu_setup.F90
    src/general/mod_mpi_setup.F90
    src/GPU/check_for_gpu.F90
    src/GPU/mod_vendor_agnostic_layer.F90
    src/GPU/mod_vendor_agnostic_general_layer.F90
    src/GPU/mod_vendor_agnostic_blas_layer.F90
    src/GPU/mod_vendor_agnostic_solver_layer.F90
    src/GPU/mod_vendor_agnostic_utilities_layer.F90
    src/GPU/mod_vendor_agnostic_ccl_layer.F90
    src/GPU/CUDA/mod_cuda.F90
    src/GPU/CUDA/mod_cusolver.F90
    src/GPU/CUDA/mod_nccl.F90
    src/GPU/ROCm/mod_hip.F90
    src/GPU/ROCm/mod_rocsolver.F90
    src/GPU/ROCm/mod_rccl.F90
    src/GPU/OpenMP/mod_openmp_offload.F90
    src/GPU/OpenMP/mod_openmp_offload_solver.F90
    src/GPU/SYCL/mod_sycl.F90
    src/GPU/SYCL/mod_syclsolver.F90
    src/GPU/SYCL/mod_oneccl.F90
    src/elpa2/mod_elpa2_utils.F90
    src/elpa2/GPU/interface_c_gpu_kernel.F90
    src/elpa2/GPU/CUDA/interface_c_cuda_kernel.F90
    src/elpa2/GPU/ROCm/interface_c_hip_kernel.F90
    src/elpa2/GPU/SYCL/interface_c_sycl_kernel.F90
    src/elpa2/mod_pack_unpack_gpu.F90
    src/elpa2/qr/qr_utils.F90
    src/elpa2/qr/elpa_qrkernels.F90
    src/elpa2/qr/elpa_pdlarfb.F90
    src/elpa2/qr/elpa_pdgeqrf.F90
    src/elpa1/elpa1.F90
    src/elpa2/elpa2.F90
    src/elpa_generalized/cannon.c
    src/elpa_generalized/gpu_vendor_agnostic_layer.c
    src/helpers/matrix_plot.F90
    src/general/mod_elpa_skewsymmetric_blas.F90
    src/solve_tridi/mod_local_to_global.F90
    src/solve_tridi/mod_global_product.F90
    src/solve_tridi/mod_global_gather.F90
    src/solve_tridi/mod_resort_ev.F90
    src/solve_tridi/mod_transform_columns.F90
    src/solve_tridi/mod_check_monotony.F90
    src/solve_tridi/mod_add_tmp.F90
    src/solve_tridi/mod_merge_systems.F90
    src/solve_tridi/mod_merge_recursive.F90
    src/solve_tridi/mod_solve_tridi.F90
    src/invert_trm/GPU/mod_invert_trm_gpu.F90
    src/invert_trm/GPU/CUDA/mod_invert_trm_cuda.F90
    src/invert_trm/GPU/ROCm/mod_invert_trm_hip.F90
    src/invert_trm/GPU/SYCL/mod_invert_trm_sycl.F90
    src/cholesky/mod_elpa_cholesky.F90
    src/cholesky/GPU/mod_cholesky_gpu.F90
    src/cholesky/GPU/CUDA/mod_cholesky_cuda.F90
    src/cholesky/GPU/ROCm/mod_cholesky_hip.F90
    src/cholesky/GPU/SYCL/mod_cholesky_sycl.F90
    src/invert_trm/mod_elpa_invert_trm.F90
    src/multiply_a_b/mod_elpa_hermitian_multiply.F90
    src/multiply_a_b/mod_elpa_pxgemm_multiply.F90
    src/multiply_a_b/mod_elpa_pxgemm_transpose.F90
    src/multiply_a_b/mod_elpa_pxgemm_helpers.F90
    src/multiply_a_b/GPU/mod_multiply_a_b_gpu.F90
    src/multiply_a_b/GPU/mod_pxgemm_multiply_gpu.F90
    src/multiply_a_b/GPU/OpenMP/mod_multiply_a_b_openmp_offload.F90
    src/solve_tridi/mod_distribute_global_column.F90
    src/solve_tridi/mod_v_add_s.F90
    src/solve_tridi/mod_solve_secular_equation.F90
    src/elpa_index.c
    src/elpa_c_interface.c
    src/general/elpa_utilities.F90
)

# ===========================================================================
# Timing sources (conditional)
# ===========================================================================
set(ELPA_TIMING_SOURCES
    src/ftimings/ftimings.F90
    src/ftimings/ftimings_type.F90
    src/ftimings/ftimings_value.F90
    src/ftimings/highwater_mark.c
    src/ftimings/resident_set_size.c
    src/ftimings/time.c
    src/ftimings/virtual_memory.c
    src/ftimings/papi.c
)

set(ELPA_TIMER_DUMMY_SOURCES src/helpers/timer_dummy.F90)

# Non-MPI builds need the Fortran interface to the wall-clock helper.
set(ELPA_NON_MPI_SOURCES src/helpers/mod_time_c.F90)

# When !WITH_MPI && !HAVE_DETAILED_TIMINGS, time.c is also needed.
set(ELPA_TIME_ONLY_SOURCES src/ftimings/time.c)

# ===========================================================================
# Affinity checking sources (conditional on HAVE_AFFINITY_CHECKING)
# ===========================================================================
set(ELPA_AFFINITY_SOURCES
    src/helpers/mod_thread_affinity.F90
    src/helpers/check_thread_affinity.c
)

# ===========================================================================
# CUDA sources (conditional on WITH_NVIDIA_GPU_VERSION)
# ===========================================================================
set(ELPA_CUDA_SOURCES
    src/GPU/CUDA/elpa_index_nvidia_gpu.cu
    src/GPU/CUDA/elpa_explicit_name_nvidia_gpu.cu
    src/GPU/CUDA/cudaFunctions.cu
    src/GPU/CUDA/cuUtils.cu
    src/cholesky/GPU/CUDA/elpa_cholesky_cuda.cu
    src/invert_trm/GPU/CUDA/elpa_invert_trm_cuda.cu
    src/elpa1/GPU/CUDA/tridiag_cuda.cu
    src/elpa1/GPU/CUDA/trans_ev_cuda.cu
    src/elpa1/GPU/CUDA/elpa1_cuda.cu
    src/elpa2/GPU/CUDA/ev_tridi_band_nvidia_gpu_real.cu
    src/elpa2/GPU/CUDA/ev_tridi_band_nvidia_gpu_complex.cu
    src/solve_tridi/GPU/CUDA/cuda_distribute_global_column.cu
    src/solve_tridi/GPU/CUDA/cuda_merge_systems.cu
    src/solve_tridi/GPU/CUDA/cuda_solve_tridi_col.cu
    src/solve_tridi/GPU/CUDA/cuda_solve_tridi_single_problem.cu
    src/solve_tridi/GPU/CUDA/cuda_transform_columns.cu
    src/multiply_a_b/GPU/CUDA/elpa_hermitian_multiply_cuda.cu
    src/multiply_a_b/GPU/CUDA/elpa_pxgemm_multiply_cuda.cu
)

# SM80 (A100+) CUDA sources
set(ELPA_CUDA_SM80_SOURCES
    src/elpa2/GPU/CUDA/ev_tridi_band_nvidia_gpu_real_sm80.cu
)
# mma_m8n8k4_fp64_sm80.cuh is included by the SM80 source above.

# NCCL sources
set(ELPA_NCCL_SOURCES src/GPU/CUDA/ncclFunctions.cpp)

# ===========================================================================
# Generic kernels (Fortran)
# ===========================================================================
set(ELPA_KERNEL_REAL_GENERIC_SOURCES src/elpa2/kernels/real.F90)
set(ELPA_KERNEL_COMPLEX_GENERIC_SOURCES src/elpa2/kernels/complex.F90)
set(ELPA_KERNEL_REAL_GENERIC_SIMPLE_SOURCES src/elpa2/kernels/real_simple.F90)
set(ELPA_KERNEL_COMPLEX_GENERIC_SIMPLE_SOURCES
    src/elpa2/kernels/complex_simple.F90
)
set(ELPA_KERNEL_REAL_GENERIC_SIMPLE_BLOCK4_SOURCES
    src/elpa2/kernels/real_simple_block4.F90
)
set(ELPA_KERNEL_REAL_GENERIC_SIMPLE_BLOCK6_SOURCES
    src/elpa2/kernels/real_simple_block6.F90
)

# ===========================================================================
# SSE Assembly kernels
# ===========================================================================
set(ELPA_KERNEL_REAL_SSE_ASSEMBLY_SOURCES
    src/elpa2/kernels/asm_x86_64_double_precision.s
)
set(ELPA_KERNEL_REAL_SSE_ASSEMBLY_SP_SOURCES
    src/elpa2/kernels/asm_x86_64_single_precision.s
)
set(ELPA_KERNEL_COMPLEX_SSE_ASSEMBLY_SOURCES
    src/elpa2/kernels/asm_x86_64_double_precision.s
)

# ===========================================================================
# Macro to define kernel source variables for a SIMD architecture
# ===========================================================================
# For real kernels: 2hv, 4hv, 6hv (block2, block4, block6)
# For complex kernels: 1hv, 2hv (block1, block2)
# Each has double precision always, single precision conditionally
macro(elpa_define_simd_kernel_sources _arch)
    # Real block2 (2hv)
    set(ELPA_KERNEL_REAL_${_arch}_BLOCK2_SOURCES
        src/elpa2/kernels/real_${_arch_lower}_2hv_double_precision.c
    )
    set(ELPA_KERNEL_REAL_${_arch}_BLOCK2_SP_SOURCES
        src/elpa2/kernels/real_${_arch_lower}_2hv_single_precision.c
    )
    # Real block4 (4hv)
    set(ELPA_KERNEL_REAL_${_arch}_BLOCK4_SOURCES
        src/elpa2/kernels/real_${_arch_lower}_4hv_double_precision.c
    )
    set(ELPA_KERNEL_REAL_${_arch}_BLOCK4_SP_SOURCES
        src/elpa2/kernels/real_${_arch_lower}_4hv_single_precision.c
    )
    # Real block6 (6hv)
    set(ELPA_KERNEL_REAL_${_arch}_BLOCK6_SOURCES
        src/elpa2/kernels/real_${_arch_lower}_6hv_double_precision.c
    )
    set(ELPA_KERNEL_REAL_${_arch}_BLOCK6_SP_SOURCES
        src/elpa2/kernels/real_${_arch_lower}_6hv_single_precision.c
    )
    # Complex block1 (1hv)
    set(ELPA_KERNEL_COMPLEX_${_arch}_BLOCK1_SOURCES
        src/elpa2/kernels/complex_${_arch_lower}_1hv_double_precision.c
    )
    set(ELPA_KERNEL_COMPLEX_${_arch}_BLOCK1_SP_SOURCES
        src/elpa2/kernels/complex_${_arch_lower}_1hv_single_precision.c
    )
    # Complex block2 (2hv)
    set(ELPA_KERNEL_COMPLEX_${_arch}_BLOCK2_SOURCES
        src/elpa2/kernels/complex_${_arch_lower}_2hv_double_precision.c
    )
    set(ELPA_KERNEL_COMPLEX_${_arch}_BLOCK2_SP_SOURCES
        src/elpa2/kernels/complex_${_arch_lower}_2hv_single_precision.c
    )
endmacro()

# Define sources for each SIMD architecture
foreach(
    _arch
    IN
    ITEMS SSE AVX AVX2 AVX512 SVE128 SVE256 SVE512 NEON_ARCH64 VSX SPARC64
)
    string(TOLOWER "${_arch}" _arch_lower)
    elpa_define_simd_kernel_sources(${_arch})
endforeach()

# ===========================================================================
# Test library sources (libelpatest)
# ===========================================================================
set(ELPA_TEST_SOURCES
    test/shared/tests_variable_definitions.F90
    test/shared/mod_tests_scalapack_interfaces.F90
    test/shared/mod_tests_blas_interfaces.F90
    test/shared/test_util.F90
    test/shared/test_read_input_parameters.F90
    test/shared/test_check_correctness.F90
    test/shared/test_setup_mpi.F90
    test/shared/test_blacs_infrastructure.F90
    test/shared/test_prepare_matrix.F90
    test/shared/test_analytic.F90
    test/shared/GPU/test_gpu_vendor_agnostic_layer.F90
    test/shared/GPU/test_gpu_vendor_agnostic_layerFunctions.c
    test/shared/test_output_type.F90
)

set(ELPA_TEST_CUDA_SOURCES
    test/shared/GPU/CUDA/test_cuda.F90
    test/shared/GPU/CUDA/test_cudaFunctions.cu
)

set(ELPA_TEST_SCALAPACK_SOURCES test/shared/test_scalapack.F90)

set(ELPA_TEST_REDIRECT_SOURCES
    test/shared/test_redir.c
    test/shared/test_redirect.F90
)

# ===========================================================================
# Assemble final source list
# ===========================================================================
# This function collects all enabled kernel sources into ELPA_KERNEL_SOURCES
function(elpa_collect_kernel_sources _outvar)
    set(_sources "")

    # Generic Fortran kernels
    foreach(
        _k
        IN
        ITEMS
            REAL_GENERIC
            COMPLEX_GENERIC
            REAL_GENERIC_SIMPLE
            COMPLEX_GENERIC_SIMPLE
            REAL_GENERIC_SIMPLE_BLOCK4
            REAL_GENERIC_SIMPLE_BLOCK6
    )
        if(WITH_${_k}_KERNEL)
            list(APPEND _sources ${ELPA_KERNEL_${_k}_SOURCES})
        endif()
    endforeach()

    # SSE Assembly
    if(WITH_REAL_SSE_ASSEMBLY_KERNEL)
        list(APPEND _sources ${ELPA_KERNEL_REAL_SSE_ASSEMBLY_SOURCES})
        if(WANT_SINGLE_PRECISION_REAL)
            list(APPEND _sources ${ELPA_KERNEL_REAL_SSE_ASSEMBLY_SP_SOURCES})
        endif()
    endif()
    if(WITH_COMPLEX_SSE_ASSEMBLY_KERNEL AND NOT WITH_REAL_SSE_ASSEMBLY_KERNEL)
        # If real assembly wasn't already added, add the .s file for complex too
        list(APPEND _sources ${ELPA_KERNEL_COMPLEX_SSE_ASSEMBLY_SOURCES})
    endif()

    # SIMD C kernels (block patterns)
    foreach(
        _arch
        IN
        ITEMS SSE AVX AVX2 AVX512 SVE128 SVE256 SVE512 NEON_ARCH64 VSX SPARC64
    )
        # Real block2
        if(WITH_REAL_${_arch}_BLOCK2_KERNEL)
            list(APPEND _sources ${ELPA_KERNEL_REAL_${_arch}_BLOCK2_SOURCES})
            if(WANT_SINGLE_PRECISION_REAL)
                list(
                    APPEND _sources
                    ${ELPA_KERNEL_REAL_${_arch}_BLOCK2_SP_SOURCES}
                )
            endif()
        endif()
        # Real block4
        if(WITH_REAL_${_arch}_BLOCK4_KERNEL)
            list(APPEND _sources ${ELPA_KERNEL_REAL_${_arch}_BLOCK4_SOURCES})
            if(WANT_SINGLE_PRECISION_REAL)
                list(
                    APPEND _sources
                    ${ELPA_KERNEL_REAL_${_arch}_BLOCK4_SP_SOURCES}
                )
            endif()
        endif()
        # Real block6
        if(WITH_REAL_${_arch}_BLOCK6_KERNEL)
            list(APPEND _sources ${ELPA_KERNEL_REAL_${_arch}_BLOCK6_SOURCES})
            if(WANT_SINGLE_PRECISION_REAL)
                list(
                    APPEND _sources
                    ${ELPA_KERNEL_REAL_${_arch}_BLOCK6_SP_SOURCES}
                )
            endif()
        endif()
        # Complex block1
        if(WITH_COMPLEX_${_arch}_BLOCK1_KERNEL)
            list(APPEND _sources ${ELPA_KERNEL_COMPLEX_${_arch}_BLOCK1_SOURCES})
            if(WANT_SINGLE_PRECISION_COMPLEX)
                list(
                    APPEND _sources
                    ${ELPA_KERNEL_COMPLEX_${_arch}_BLOCK1_SP_SOURCES}
                )
            endif()
        endif()
        # Complex block2
        if(WITH_COMPLEX_${_arch}_BLOCK2_KERNEL)
            list(APPEND _sources ${ELPA_KERNEL_COMPLEX_${_arch}_BLOCK2_SOURCES})
            if(WANT_SINGLE_PRECISION_COMPLEX)
                list(
                    APPEND _sources
                    ${ELPA_KERNEL_COMPLEX_${_arch}_BLOCK2_SP_SOURCES}
                )
            endif()
        endif()
    endforeach()

    set(${_outvar} "${_sources}" PARENT_SCOPE)
endfunction()
