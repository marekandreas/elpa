function(elpa_apply_fortran_compile_options target)
    if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
        target_compile_options(
            ${target}
            PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-free>
        )
        if(WIN32)
            target_compile_options(
                ${target}
                PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:/check:noarg_temp_created>
            )
        else()
            target_compile_options(
                ${target}
                PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-check noarg_temp_created>
            )
        endif()
    elseif(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
        target_compile_options(
            ${target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:Fortran>:-ffree-form
                -ffree-line-length-none>
        )
    endif()

    if(
        CMAKE_Fortran_COMPILER_ID MATCHES "LLVMFlang"
        AND ELPA_MPI
        AND NOT HAVE_MPI_MODULE
    )
        # Intel MPI's mpi.mod / mpi_f08.mod files are not readable by flang-new
        # on this host, so ELPA must fall back to mpif.h. That fallback uses
        # implicit MPI interfaces and flang emits a large number of false
        # positive -Wincompatible-implicit-interfaces diagnostics. flang-new
        # has no granular suppression for this warning class, so silence
        # warnings only for this specific unsupported module-ABI combination.
        target_compile_options(
            ${target}
            PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-w>
        )
    endif()
endfunction()

# gfortran 10+ treats type-mismatch at call sites as a hard error.
# ELPA's generic kernels intentionally pass real arrays as complex arguments
# (type-punning with halved leading dimension) in real_template.F90.
# -fallow-argument-mismatch downgrades the error to a warning; -w silences
# the remaining diagnostic.  Scoped to this one file only.
function(elpa_apply_gfortran_argument_mismatch_workaround)
    if(NOT CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
        return()
    endif()

    set(_pp_dir "${CMAKE_BINARY_DIR}/_pp")
    set_source_files_properties(
        "${_pp_dir}/src/src/elpa2/kernels/real.F90"
        PROPERTIES COMPILE_OPTIONS "-fallow-argument-mismatch;-w"
    )
endfunction()

function(elpa_apply_cuda_target_options target)
    if(WITH_NVIDIA_GPU_VERSION)
        target_compile_options(
            ${target}
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${ELPA_CUDA_FLAGS}>
        )
        if(WIN32)
            target_compile_definitions(
                ${target}
                PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:_SILENCE_CXX17_C_HEADER_DEPRECATION_WARNING>
            )
        endif()
    endif()
endfunction()

function(elpa_apply_ifort_diagnostic_suppressions)
    if(NOT CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
        return()
    endif()

    set(_pp_dir "${CMAKE_BINARY_DIR}/_pp")
    if(WIN32)
        set(_diag_5462_flag "/Qdiag-disable:5462")
        set(_diag_6536_flag "/Qdiag-disable:6536")
    else()
        set(_diag_5462_flag "-diag-disable=5462")
        set(_diag_6536_flag "-diag-disable=6536")
    endif()

    # These preprocessed sources exceed ifort's internal symbol-name limit.
    foreach(
        _source
        IN
        ITEMS
            "${_pp_dir}/src/src/multiply_a_b/mod_elpa_hermitian_multiply.F90"
            "${_pp_dir}/src/src/multiply_a_b/mod_elpa_pxgemm_multiply.F90"
            "${_pp_dir}/src/src/multiply_a_b/mod_elpa_pxgemm_transpose.F90"
            "${_pp_dir}/src/src/cholesky/mod_elpa_cholesky.F90"
            "${_pp_dir}/src/src/invert_trm/mod_elpa_invert_trm.F90"
            "${_pp_dir}/src/src/elpa1/elpa1.F90"
            "${_pp_dir}/src/src/elpa2/elpa2.F90"
            "${_pp_dir}/src/src/elpa2/qr/elpa_pdgeqrf.F90"
    )
        set_source_files_properties(
            "${_source}"
            PROPERTIES COMPILE_OPTIONS "${_diag_5462_flag}"
        )
    endforeach()

    # mod_redist_band repeats a harmless ONLY import under Intel Fortran.
    set_source_files_properties(
        "${_pp_dir}/src/src/elpa2/mod_redist_band.F90"
        PROPERTIES COMPILE_OPTIONS "${_diag_6536_flag}"
    )
endfunction()

# icx emits -Wdefault-const-init-var-unsafe for the VLA-based compile-time
# enum ranking trick in elpa_index.c.  icx / clang emit
# -Wunused-command-line-argument for C flags passed to .s assembly sources.
function(elpa_apply_clang_diagnostic_suppressions)
    if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang|IntelLLVM")
        return()
    endif()

    # icx (IntelLLVM) warns about VLA trick in elpa_index.c; older clang does
    # not have this diagnostic, so restrict the suppression to IntelLLVM.
    if(CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM")
        set_source_files_properties(
            "${CMAKE_CURRENT_SOURCE_DIR}/elpa_index.c"
            PROPERTIES COMPILE_OPTIONS "-Wno-default-const-init-var-unsafe"
        )
    endif()

    # Suppress -Wunused-command-line-argument on .s assembly sources
    # (CMake passes C flags like -D, -MD, -isystem which the assembler ignores)
    foreach(
        _source
        IN
        ITEMS
            "${CMAKE_CURRENT_SOURCE_DIR}/elpa2/kernels/asm_x86_64_double_precision.s"
            "${CMAKE_CURRENT_SOURCE_DIR}/elpa2/kernels/asm_x86_64_single_precision.s"
    )
        if(EXISTS "${_source}")
            set_source_files_properties(
                "${_source}"
                PROPERTIES COMPILE_OPTIONS "-Wno-unused-command-line-argument"
            )
        endif()
    endforeach()
endfunction()

# LLVMFlang (flang-new) emits warnings that other Fortran compilers do not:
#  - c_ptr vs integer(c_intptr_t) interface mismatches in GPU binding overloads
#    (both types are ABI-identical; the dual-overload design is intentional)
#  - implicit-interface shape mismatches for BLAS/ScaLAPACK calls
#    (standard Fortran 77 calling convention, correct at runtime)
#  - unrecognised Intel !DIR$ directives (harmless, silently ignored)
#  - implicit kernel call interfaces in real.F90
# LLVMFlang has no granular -Wno- flags, so we suppress all warnings (-w)
# on the specific source files that trigger these diagnostics.
function(elpa_apply_flang_diagnostic_suppressions)
    if(NOT CMAKE_Fortran_COMPILER_ID MATCHES "LLVMFlang")
        return()
    endif()

    set(_pp_dir "${CMAKE_BINARY_DIR}/_pp")
    foreach(
        _source
        IN
        ITEMS
            # GPU c_ptr vs integer(c_intptr_t) dual-binding overloads
            "${_pp_dir}/src/src/GPU/CUDA/mod_cuda.F90"
            "${_pp_dir}/src/src/GPU/ROCm/mod_hip.F90"
            "${_pp_dir}/src/src/GPU/SYCL/mod_sycl.F90"
            "${_pp_dir}/src/src/GPU/OpenMP/mod_openmp_offload.F90"
            "${_pp_dir}/src/src/multiply_a_b/GPU/OpenMP/mod_multiply_a_b_openmp_offload.F90"
            # Implicit BLAS/ScaLAPACK interface shape mismatches
            "${_pp_dir}/src/src/elpa_impl.F90"
            "${_pp_dir}/src/src/elpa2/qr/elpa_pdgeqrf.F90"
            # Implicit kernel call interfaces
            "${_pp_dir}/src/src/elpa2/kernels/real.F90"
            # Intel !DIR$ OPTIMIZE / !DIR$ IVDEP directives
            "${_pp_dir}/src/src/elpa2/kernels/real_simple_block4.F90"
            "${_pp_dir}/src/src/elpa2/kernels/real_simple_block6.F90"
            "${_pp_dir}/src/src/solve_tridi/mod_add_tmp.F90"
            "${_pp_dir}/src/src/solve_tridi/mod_merge_systems.F90"
            # GPU c_ptr vs integer(c_intptr_t) in test support module
            "${_pp_dir}/src/test/shared/GPU/CUDA/test_cuda.F90"
    )
        set_source_files_properties(
            "${_source}"
            PROPERTIES COMPILE_OPTIONS "-w"
        )
    endforeach()
endfunction()
