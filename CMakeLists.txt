cmake_minimum_required(VERSION 3.24)

set(_elpa_languages C CXX Fortran ASM)
if(WIN32)
    list(REMOVE_ITEM _elpa_languages ASM)
endif()

project(
    elpa
    VERSION 2026.02.001
    LANGUAGES ${_elpa_languages}
    DESCRIPTION "Eigenvalue SoLvers for Petaflop-Applications"
)

# ===========================================================================
# Toolchain constraint: MSVC cl.exe cannot compile ELPA's C code
# ===========================================================================
# ELPA's C interface layer uses C99 _Complex / "double complex" pervasively
# (72+ occurrences across 15 header/source files, plus _Generic selections
# and Fortran-C interop signatures).  MSVC cl.exe does not implement C99
# _Complex and Microsoft has stated this is "not planned":
#   https://learn.microsoft.com/en-us/cpp/c-language/c11-c17-conformance
#
# On Windows, use clang-cl (Clang with MSVC ABI) which supports _Complex
# natively.  When CUDA is enabled, nvcc still requires real MSVC cl.exe as
# its host compiler — this is handled separately in ElpaCUDA.cmake.
if(WIN32 AND CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    message(
        FATAL_ERROR
        "ELPA cannot be compiled with MSVC cl.exe — it lacks C99 _Complex support.\n"
        "Use clang-cl instead: -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl\n"
        "For CUDA builds, nvcc will auto-detect cl.exe as its host compiler via ElpaCUDA.cmake."
    )
endif()

# ===========================================================================
# Build options (matching autotools --enable/--with flags)
# ===========================================================================
option(ELPA_MPI "Build with MPI support" ON)
option(ELPA_OPENMP "Build with OpenMP support" OFF)
option(ELPA_CUDA "Build with NVIDIA CUDA GPU support" OFF)
option(ELPA_SINGLE_PRECISION "Build single-precision variants" ON)
option(ELPA_SKEWSYMMETRIC "Build skew-symmetric matrix support" ON)
option(ELPA_TIMINGS "Build with detailed timing support" ON)
option(ELPA_AUTOTUNE "Enable autotuning support" ON)
option(ELPA_64BIT_INTEGER_MATH "Use 64-bit integers for BLAS/LAPACK" OFF)
option(ELPA_64BIT_INTEGER_MPI "Use 64-bit integers for MPI" OFF)
option(ELPA_STORE_BUILD_CONFIG "Embed build config in library" OFF)
option(
    ELPA_OPTIONAL_C_ERROR_ARGUMENT
    "Add optional error argument to C API"
    OFF
)
option(
    ELPA_INSTALL_CMAKE_PACKAGE
    "Install the ELPA CMake package config alongside the library"
    ON
)
option(BUILD_TESTING "Build tests" ON)
option(BUILD_SHARED_LIBS "Build ELPA as a shared library" ON)
option(
    ELPA_TEST_EXTENDED
    "Enable the extended test set (larger matrices, all MPI layouts; disabled by default)"
    OFF
)
option(
    ELPA_TEST_AUTOTUNE
    "Enable autotuning tests (very long runtime; disabled by default)"
    OFF
)

# SO version: libtool convention current:revision:age → SOVERSION=current-age
# ELPA_SO_VERSION = 24:1:5  →  SOVERSION=19, VERSION=19.5.1
set(ELPA_SO_CURRENT 24)
set(ELPA_SO_REVISION 1)
set(ELPA_SO_AGE 5)
math(EXPR ELPA_SOVERSION "${ELPA_SO_CURRENT} - ${ELPA_SO_AGE}")
set(ELPA_LIBVERSION "${ELPA_SOVERSION}.${ELPA_SO_AGE}.${ELPA_SO_REVISION}")

# ===========================================================================
# Module path
# ===========================================================================
set(_elpa_overlay_cmake_dir "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# Keep the public helper modules on CMAKE_MODULE_PATH. Dependency discovery
# stays under cmake/dependencies and is included explicitly by the entry
# points that own it.
list(APPEND CMAKE_MODULE_PATH "${_elpa_overlay_cmake_dir}")

# ===========================================================================
# Feature detection and configuration
# ===========================================================================
# CUDA must be processed first so it can set kernel options for ElpaKernels
include("${_elpa_overlay_cmake_dir}/dependencies/ElpaCUDA.cmake")
include(ElpaConfig)
include(ElpaKernels)
include(ElpaSourceLists)
include(ElpaGeneratedFiles)

# ===========================================================================
# Generated headers and extracted interfaces
# ===========================================================================
elpa_generate_build_artifacts()

# ===========================================================================
# Library build
# ===========================================================================
add_subdirectory(src)

# ===========================================================================
# Tests
# ===========================================================================
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()

# ===========================================================================
# Installation
# ===========================================================================
include(ElpaInstall)
