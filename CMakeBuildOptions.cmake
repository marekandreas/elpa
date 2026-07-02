# Central declaration of ELPA's user-facing CMake build options.
#
# Keep dependency discovery, feature checks, and option validation in their
# respective modules. This file only defines build configuration knobs and
# their defaults.

# ---------------------------------------------------------------------------
# Core library features
# ---------------------------------------------------------------------------
option(BUILD_SHARED_LIBS "Build ELPA as a shared library" ON)
option(BUILD_TESTING "Build tests" ON)

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

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
option(
    ELPA_INSTALL_CMAKE_PACKAGE
    "Install the ELPA CMake package config alongside the library"
    ON
)

# ---------------------------------------------------------------------------
# Math libraries and framework ISA
# ---------------------------------------------------------------------------
set(
    ELPA_BLAS_VENDOR
    "AUTO"
    CACHE STRING
    "BLAS/LAPACK vendor (AUTO, MKL, BLAS, or a CMake BLA_VENDOR value)"
)
set_property(
    CACHE ELPA_BLAS_VENDOR
    PROPERTY STRINGS AUTO MKL BLAS OpenBLAS FlexiBLAS Generic
)

set(
    ELPA_FRAMEWORK_ISA
    "native"
    CACHE STRING
    "Baseline SIMD ISA for framework (non-kernel) Fortran and C code (native, avx2, avx512)"
)
set_property(CACHE ELPA_FRAMEWORK_ISA PROPERTY STRINGS native avx2 avx512)

# ---------------------------------------------------------------------------
# CPU and GPU kernel families
# ---------------------------------------------------------------------------
option(ELPA_ENABLE_GENERIC_KERNELS "Build generic kernels" ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
    set(_elpa_x86_kernel_default ON)
else()
    set(_elpa_x86_kernel_default OFF)
endif()
option(ELPA_ENABLE_SSE_KERNELS "Build SSE intrinsics kernels" ${_elpa_x86_kernel_default})
option(ELPA_ENABLE_SSE_ASSEMBLY_KERNELS "Build SSE assembly kernels" ${_elpa_x86_kernel_default})
option(ELPA_ENABLE_AVX_KERNELS "Build AVX kernels" ${_elpa_x86_kernel_default})
option(ELPA_ENABLE_AVX2_KERNELS "Build AVX2 kernels" ${_elpa_x86_kernel_default})
option(ELPA_ENABLE_AVX512_KERNELS "Build AVX512 kernels" ${_elpa_x86_kernel_default})
unset(_elpa_x86_kernel_default)

option(ELPA_ENABLE_SVE128_KERNELS "Build SVE128 kernels" OFF)
option(ELPA_ENABLE_SVE256_KERNELS "Build SVE256 kernels" OFF)
option(ELPA_ENABLE_SVE512_KERNELS "Build SVE512 kernels" OFF)
option(ELPA_ENABLE_SPARC64_KERNELS "Build SPARC64 kernels" OFF)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    set(_elpa_neon_kernel_default ON)
else()
    set(_elpa_neon_kernel_default OFF)
endif()
option(ELPA_ENABLE_NEON_ARCH64_KERNELS "Build NEON AARCH64 kernels" ${_elpa_neon_kernel_default})
unset(_elpa_neon_kernel_default)

option(ELPA_ENABLE_VSX_KERNELS "Build VSX (PPC) kernels" OFF)
option(ELPA_ENABLE_NVIDIA_GPU_KERNELS "Build NVIDIA GPU kernels" OFF)
option(
    ELPA_ENABLE_NVIDIA_SM80_GPU_KERNELS
    "Build NVIDIA SM80 (A100+) kernels"
    OFF
)
option(ELPA_ENABLE_AMD_GPU_KERNELS "Build AMD GPU (ROCm) kernels" OFF)
option(ELPA_ENABLE_INTEL_GPU_SYCL_KERNELS "Build Intel GPU (SYCL) kernels" OFF)

# ---------------------------------------------------------------------------
# NVIDIA CUDA support
# ---------------------------------------------------------------------------
# Preserve the existing cache surface: CUDA-only options are declared only
# when CUDA support itself is enabled.
if(ELPA_CUDA)
    set(
        ELPA_CUDA_ARCHITECTURES
        "native"
        CACHE STRING
        "CUDA compute capabilities: semicolon-separated SM list or \"native\" to auto-detect the host GPU"
    )

    option(ELPA_CUSOLVER "Use NVIDIA cuSOLVER library" ON)
    option(ELPA_NCCL "Use NVIDIA NCCL library" OFF)
    option(ELPA_GPU_STREAMS "Use CUDA streams" ON)
    option(ELPA_CUDA_AWARE_MPI "Use CUDA-aware MPI" OFF)
    option(ELPA_CUDA_DEBUG "Enable CUDA memory debugging" OFF)
    option(ELPA_NVIDIA_CUB "Use CUB reductions in real NVIDIA GPU kernel" OFF)
    option(ELPA_NVTX "Enable NVTX profiler annotations" OFF)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
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

if(BUILD_TESTING)
    set(
        ELPA_TEST_PROGRAMS
        "auto"
        CACHE STRING
        "Which test programs to build: all, gpu, cpu, auto (default: auto)"
    )
    set_property(CACHE ELPA_TEST_PROGRAMS PROPERTY STRINGS all gpu cpu auto)

    if(ELPA_MPI)
        set(ELPA_TEST_NPROCS "2" CACHE STRING "Number of MPI processes for tests")
    else()
        set(ELPA_TEST_NPROCS "1")
    endif()
    set(ELPA_TEST_MATRIX_SIZE "200" CACHE STRING "Matrix size for ELPA tests")
    set(ELPA_TEST_NEV "30" CACHE STRING "Number of eigenvectors for ELPA tests")
    set(ELPA_TEST_NBLK "16" CACHE STRING "Block size for ELPA tests")
endif()
