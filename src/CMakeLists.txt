# ===========================================================================
# Collect kernel sources
# ===========================================================================
elpa_collect_kernel_sources(ELPA_KERNEL_SOURCES)

# ===========================================================================
# Assemble complete source list (all paths relative to project root)
# ===========================================================================
set(_all_sources "")

# Prefix all source paths with the project source dir
foreach(
    _src
    IN
    LISTS ELPA_PUBLIC_SOURCES ELPA_PRIVATE_SOURCES ELPA_KERNEL_SOURCES
)
    list(APPEND _all_sources "${CMAKE_SOURCE_DIR}/${_src}")
endforeach()

# fortran_constants.F90 is included by elpa_constants.F90 and remains only as a
# dependency of elpa_fortran_constants.

# Non-MPI support
if(NOT WITH_MPI)
    foreach(_src IN LISTS ELPA_NON_MPI_SOURCES)
        list(APPEND _all_sources "${CMAKE_SOURCE_DIR}/${_src}")
    endforeach()
endif()

# Timing
if(HAVE_DETAILED_TIMINGS)
    foreach(_src IN LISTS ELPA_TIMING_SOURCES)
        list(APPEND _all_sources "${CMAKE_SOURCE_DIR}/${_src}")
    endforeach()
else()
    foreach(_src IN LISTS ELPA_TIMER_DUMMY_SOURCES)
        list(APPEND _all_sources "${CMAKE_SOURCE_DIR}/${_src}")
    endforeach()
    if(NOT WITH_MPI)
        foreach(_src IN LISTS ELPA_TIME_ONLY_SOURCES)
            list(APPEND _all_sources "${CMAKE_SOURCE_DIR}/${_src}")
        endforeach()
    endif()
endif()

# Affinity
if(HAVE_AFFINITY_CHECKING)
    foreach(_src IN LISTS ELPA_AFFINITY_SOURCES)
        list(APPEND _all_sources "${CMAKE_SOURCE_DIR}/${_src}")
    endforeach()
endif()

# CUDA
if(WITH_NVIDIA_GPU_VERSION)
    foreach(_src IN LISTS ELPA_CUDA_SOURCES)
        list(APPEND _all_sources "${CMAKE_SOURCE_DIR}/${_src}")
    endforeach()
    if(WITH_NVIDIA_GPU_SM80_COMPUTE_CAPABILITY)
        foreach(_src IN LISTS ELPA_CUDA_SM80_SOURCES)
            list(APPEND _all_sources "${CMAKE_SOURCE_DIR}/${_src}")
        endforeach()
    endif()
endif()

# NCCL
if(WITH_NVIDIA_NCCL)
    foreach(_src IN LISTS ELPA_NCCL_SOURCES)
        list(APPEND _all_sources "${CMAKE_SOURCE_DIR}/${_src}")
    endforeach()
endif()

# ===========================================================================
# Create the library target
# ===========================================================================
set(_libname "elpa${ELPA_SUFFIX}")

# ---------------------------------------------------------------------------
# Two-step Fortran preprocessing with "cpp -P -traditional"
# ---------------------------------------------------------------------------
# ELPA uses cpp macros that must expand inside Fortran tokens (e.g. 1.0_rk).
# Standard fpp treats 1.0_rk as a single token.  Traditional cpp does not.
include(ElpaFortranPreprocess)
include(ElpaTargetOptions)

# Gather include dirs and defines for the external cpp pass
set(_pp_inc_dirs
    "${CMAKE_BINARY_DIR}"
    "${CMAKE_BINARY_DIR}/src"
    "${CMAKE_BINARY_DIR}/elpa"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/elpa"
)
# MKL include paths for Fortran preprocessor (set by find_package(MKL CONFIG))
if(MKL_INCLUDE)
    list(APPEND _pp_inc_dirs ${MKL_INCLUDE})
endif()
set(_pp_defines "HAVE_CONFIG_H")
if(WIN32)
    list(APPEND _pp_defines "_WIN32")
endif()

elpa_preprocess_fortran(_pp_sources
    SOURCES     ${_all_sources}
    INCLUDE_DIRS ${_pp_inc_dirs}
    DEFINES      ${_pp_defines}
)

add_library(${_libname} ${_pp_sources})
set_target_properties(${_libname} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(${_libname} PUBLIC c_std_11 cxx_std_17)

elpa_apply_ifort_diagnostic_suppressions()
elpa_apply_gfortran_argument_mismatch_workaround()
elpa_apply_clang_diagnostic_suppressions()
elpa_apply_flang_diagnostic_suppressions()

if(BUILD_SHARED_LIBS)
    include(ElpaSelectiveExport)
    elpa_selective_export(${_libname})
endif()

# Depend on generated headers and config
add_dependencies(${_libname} elpa_generated_headers elpa_fortran_constants)

# ===========================================================================
# Fortran preprocessing — DISABLED (done externally via cpp -P -traditional)
# ===========================================================================
set_target_properties(${_libname} PROPERTIES Fortran_PREPROCESS OFF)

# ===========================================================================
# Fortran module output directory
# ===========================================================================
set(_mod_dir "${CMAKE_BINARY_DIR}/modules")
set_target_properties(
    ${_libname}
    PROPERTIES Fortran_MODULE_DIRECTORY "${_mod_dir}"
)

# ===========================================================================
# Include directories
# ===========================================================================
target_include_directories(
    ${_libname}
    PRIVATE
        "${CMAKE_BINARY_DIR}" # config.h, config-f90.h
        "${CMAKE_BINARY_DIR}/src" # generated Fortran interfaces
        "${CMAKE_BINARY_DIR}/elpa" # elpa_generated.h, elpa_constants.h
        "${CMAKE_CURRENT_SOURCE_DIR}" # src/ relative includes
        "${CMAKE_CURRENT_SOURCE_DIR}/.." # root-relative includes (elpa/)
        "${CMAKE_CURRENT_SOURCE_DIR}/../elpa" # elpa/ headers
        "${_mod_dir}" # Fortran modules
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../elpa>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/elpa>
        $<BUILD_INTERFACE:${_mod_dir}>
        $<INSTALL_INTERFACE:include/elpa${ELPA_SUFFIX}-${PROJECT_VERSION}>
        $<INSTALL_INTERFACE:include/elpa${ELPA_SUFFIX}-${PROJECT_VERSION}/elpa>
        $<INSTALL_INTERFACE:include/elpa${ELPA_SUFFIX}-${PROJECT_VERSION}/modules>
)

# ===========================================================================
# Compiler flags
# ===========================================================================
elpa_apply_fortran_compile_options(${_libname})

# Fortran ISA targeting and optimisation/debug flags (set by ElpaConfig.cmake)
if(ELPA_FORTRAN_ISA_FLAGS)
    target_compile_options(
        ${_libname}
        PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:${ELPA_FORTRAN_ISA_FLAGS}>
    )
endif()
if(ELPA_FORTRAN_OPT_FLAGS)
    target_compile_options(
        ${_libname}
        PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<NOT:$<CONFIG:Debug>>>:${ELPA_FORTRAN_OPT_FLAGS}>
    )
endif()
if(ELPA_FORTRAN_DEBUG_FLAGS)
    target_compile_options(
        ${_libname}
        PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Debug>>:${ELPA_FORTRAN_DEBUG_FLAGS}>
    )
endif()

# GCC aggressive C optimisation flags (set by ElpaConfig.cmake)
if(ELPA_GCC_C_OPT_FLAGS)
    target_compile_options(
        ${_libname}
        PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:C>,$<NOT:$<CONFIG:Debug>>>:${ELPA_GCC_C_OPT_FLAGS}>
    )
endif()

# C flags — ISA-specific flags for kernel source files
# GCC warns '-Wattributes' on ELPA's __attribute__((always_inline)) functions
# in the BLOCK_template kernels when inlining cannot be guaranteed (e.g. at
# -O0 or when the function is too large).  The attribute is intentional for
# performance; suppress per kernel file.
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(_kernel_warn_flags "-Wno-attributes")
else()
    set(_kernel_warn_flags "")
endif()
if(HAVE_SSE_INTRINSICS)
    file(
        GLOB _sse_c_sources
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/elpa2/kernels/*sse*precision.c"
    )
    if(_sse_c_sources)
        set(_sse_opts -msse3)
        if(_kernel_warn_flags)
            list(APPEND _sse_opts ${_kernel_warn_flags})
        endif()
        set_source_files_properties(
            ${_sse_c_sources}
            PROPERTIES COMPILE_OPTIONS "${_sse_opts}"
        )
    endif()
endif()
if(HAVE_AVX)
    file(
        GLOB _avx_c_sources
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/elpa2/kernels/*avx_*precision.c"
    )
    if(_avx_c_sources)
        set(_avx_opts -mavx)
        if(_kernel_warn_flags)
            list(APPEND _avx_opts ${_kernel_warn_flags})
        endif()
        set_source_files_properties(
            ${_avx_c_sources}
            PROPERTIES COMPILE_OPTIONS "${_avx_opts}"
        )
    endif()
endif()
if(HAVE_AVX2)
    file(
        GLOB _avx2_c_sources
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/elpa2/kernels/*avx2*precision.c"
    )
    if(_avx2_c_sources)
        set(_avx2_opts -mavx2 -mfma)
        if(_kernel_warn_flags)
            list(APPEND _avx2_opts ${_kernel_warn_flags})
        endif()
        set_source_files_properties(
            ${_avx2_c_sources}
            PROPERTIES COMPILE_OPTIONS "${_avx2_opts}"
        )
    endif()
endif()
if(HAVE_AVX512)
    file(
        GLOB _avx512_c_sources
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/elpa2/kernels/*avx512*precision.c"
    )
    if(_avx512_c_sources)
        set(_avx512_opts -mavx512f)
        if(HAVE_AVX512_XEON)
            list(APPEND _avx512_opts -mavx512dq)
        endif()
        if(_kernel_warn_flags)
            list(APPEND _avx512_opts ${_kernel_warn_flags})
        endif()
        set_source_files_properties(
            ${_avx512_c_sources}
            PROPERTIES COMPILE_OPTIONS "${_avx512_opts}"
        )
    endif()
endif()

# CUDA flags
if(WITH_NVIDIA_GPU_VERSION)
    set_source_files_properties(${ELPA_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
    if(ELPA_CUDA_SM80_SOURCES AND WITH_NVIDIA_GPU_SM80_COMPUTE_CAPABILITY)
        set_source_files_properties(
            ${ELPA_CUDA_SM80_SOURCES}
            PROPERTIES LANGUAGE CUDA
        )
    endif()
    elpa_apply_cuda_target_options(${_libname})
endif()

# ===========================================================================
# Portability / warning suppression
# ===========================================================================
# Suppress MSVC deprecation warnings for portable C APIs used by elpa_index.c.
if(WIN32)
    target_compile_definitions(${_libname} PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# BLOCK_template intentionally redefines ROW_LENGTH across mutually exclusive
# branches; clang still warns on the active branch.
if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set_source_files_properties(
        "${CMAKE_SOURCE_DIR}/src/elpa2/kernels/real_128bit_256bit_512bit_BLOCK_template.c"
        PROPERTIES COMPILE_OPTIONS "-Wno-macro-redefined"
    )
endif()

# ===========================================================================
# Link dependencies
# ===========================================================================
# MPI
if(WITH_MPI)
    target_link_libraries(${_libname} PUBLIC MPI::MPI_Fortran MPI::MPI_C)
endif()

# OpenMP
if(WITH_OPENMP_TRADITIONAL)
    target_link_libraries(
        ${_libname}
        PUBLIC OpenMP::OpenMP_Fortran OpenMP::OpenMP_C
    )
endif()

# SCALAPACK / MKL — consume the imported target created in ElpaConfig.cmake
target_link_libraries(${_libname} PUBLIC ELPA::scalapack)

# CUDA libraries
if(WITH_NVIDIA_GPU_VERSION)
    target_link_libraries(${_libname} PRIVATE CUDA::cublas CUDA::cudart)
    if(HAVE_CUBLASLT AND TARGET CUDA::cublasLt)
        target_link_libraries(${_libname} PRIVATE CUDA::cublasLt)
    endif()
    if(WITH_NVIDIA_CUSOLVER AND TARGET CUDA::cusolver)
        target_link_libraries(${_libname} PRIVATE CUDA::cusolver)
    endif()
    if(WITH_NVIDIA_NCCL AND NCCL_LIBRARY)
        target_link_libraries(${_libname} PRIVATE ${NCCL_LIBRARY})
        target_include_directories(${_libname} PRIVATE ${NCCL_INCLUDE_DIR})
    endif()
    if(WITH_NVTX)
        if(ELPA_NVTX3_HEADER_ONLY)
            # NVTX3 header-only: compile the C shim that emits externally-
            # visible nvtxRangePushA / nvtxRangePop symbols for Fortran.
            target_sources(${_libname} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/general/nvtx_impl.c)
            target_include_directories(${_libname} PRIVATE
                ${NVTX3_INCLUDE_DIR} ${NVTX3_INCLUDE_DIR}/nvtx3)
        elseif(NVTOOLSEXT_LIBRARY)
            # Legacy NVTX v1/v2 shared library
            target_link_libraries(${_libname} PRIVATE ${NVTOOLSEXT_LIBRARY})
        endif()
    endif()
endif()

# Math library (libm on UNIX, not needed on MSVC)
if(UNIX)
    target_link_libraries(${_libname} PRIVATE m)
endif()

# ===========================================================================
# Library versioning
# ===========================================================================
set_target_properties(
    ${_libname}
    PROPERTIES
        SOVERSION "${ELPA_SOVERSION}"
        VERSION "${ELPA_LIBVERSION}"
        OUTPUT_NAME "elpa${ELPA_SUFFIX}"
)

# ===========================================================================
# Aliases & exports
# ===========================================================================
add_library(ELPA::elpa ALIAS ${_libname})
