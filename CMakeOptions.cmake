# SPDX-License-Identifier: LGPL-3.0-or-later

include_guard(GLOBAL)

# This file contains ELPA's user-facing CMake cache variables. Compiler flags,
# feature-test results, dependency paths, and other implementation details must
# remain in the corresponding CMake modules instead of becoming public options.

# =================================================================================================
# Build configuration
# =================================================================================================

option(BUILD_SHARED_LIBS "Build ELPA as a shared library" ON)
option(BUILD_TESTING "Build ELPA tests" OFF)
option(ELPA_BUILD_TOOLS "Build ELPA command-line utilities" ON)

# =================================================================================================
# Parallel programming models
# =================================================================================================

option(ELPA_USE_MPI "Enable MPI support" ON)
option(ELPA_USE_64BIT_MPI_INTEGERS "Use 64-bit MPI integer arguments" OFF)

# Enable host-side OpenMP threading. Accelerator offloading is selected
# independently through ELPA_ACCELERATOR.
option(ELPA_USE_OPENMP "Enable OpenMP threading" ON)

# =================================================================================================
# Numerical interfaces and optional functionality
# =================================================================================================

# Build both single- and double-precision interfaces. Disabling this option
# reduces build time and library size but removes the single-precision API.
option(ELPA_ENABLE_SINGLE_PRECISION "Build single-precision interfaces" ON)

# Enable ELPA's runtime autotuning infrastructure.
option(ELPA_ENABLE_AUTOTUNING "Enable autotuning support" ON)

# Enable skew-symmetric eigensolver support.
option(ELPA_ENABLE_SKEWSYMMETRIC "Enable skew-symmetric solvers" ON)

# Enable loop blocking while transforming an ELPA2 band matrix back to full
# form. This matches the default Autotools configuration.
option(
  ELPA_ENABLE_BAND_TO_FULL_BLOCKING
  "Enable blocking in the ELPA2 band-to-full transformation"
  ON
)

# Enable detailed internal timings. PAPI counters remain disabled unless
# ELPA_USE_PAPI is enabled separately.
option(ELPA_ENABLE_TIMINGS "Enable detailed timing support" ON)

# Enable process and thread affinity diagnostics. Configuration checks the
# required platform interfaces and reports an error when they are unavailable.
option(ELPA_ENABLE_AFFINITY "Enable affinity diagnostics" ON)

# Use assumed-size arrays in performance-critical Fortran interfaces, matching
# the default Autotools configuration.
option(ELPA_USE_ASSUMED_SIZE "Use assumed-size Fortran arrays" ON)

# Enable optional Fortran 2008 code paths.
option(ELPA_USE_FORTRAN_2008 "Enable Fortran 2008 code paths" ON)

# Make the error argument optional in the generated C API. This changes the
# generated declarations and is disabled by default for API compatibility.
option(ELPA_OPTIONAL_C_ERROR_ARGUMENT "Make the C API error argument optional" OFF)

# Embed a human-readable description of the build configuration in the ELPA
# library. Keep this disabled for reproducible builds.
option(ELPA_STORE_BUILD_CONFIG "Embed the build configuration in ELPA" OFF)

# Collect hardware-counter data through PAPI as part of detailed timings.
option(ELPA_USE_PAPI "Enable PAPI performance counters" OFF)

# =================================================================================================
# CPU kernels
# =================================================================================================

# Select the CPU kernels compiled into ELPA.
#
#   AUTO     Build the portable kernels and every architecture-specific kernel
#            supported by the target architecture and active compiler.
#   GENERIC  Build only the portable Fortran kernels.
#
# Architecture-specific compiler options are detected internally and applied
# only to the corresponding source files. They are not exposed as cache FLAGS.
set(ELPA_CPU_KERNELS "AUTO" CACHE STRING "CPU kernel selection")
set_property(
  CACHE ELPA_CPU_KERNELS
  PROPERTY STRINGS AUTO GENERIC
)

# Enable runtime selection of an instruction set supported by every process in
# a heterogeneous x86 cluster. This feature has a runtime cost and is intended
# only for clusters where nodes may provide different SIMD instruction sets.
option(
  ELPA_ENABLE_HETEROGENEOUS_CLUSTER
  "Enable heterogeneous-cluster CPU dispatch"
  OFF
)

# =================================================================================================
# Accelerator backend
# =================================================================================================

# Select exactly one accelerator backend for this build tree.
#
#   NONE    Build CPU implementations only.
#   CUDA    Build the NVIDIA CUDA backend.
#   HIP     Build the AMD ROCm/HIP backend.
#   SYCL    Build the Intel SYCL backend with a SYCL-capable C++ compiler.
#   OPENMP  Build the Intel OpenMP target-offload backend.
set(ELPA_ACCELERATOR "NONE" CACHE STRING "Accelerator backend")
set_property(
  CACHE ELPA_ACCELERATOR
  PROPERTY STRINGS NONE CUDA HIP SYCL OPENMP
)

# Enable asynchronous streams or queues for accelerator operations. This
# option is ignored when ELPA_ACCELERATOR is NONE.
option(ELPA_ENABLE_GPU_STREAMS "Enable accelerator streams or queues" ON)

# Enable direct transfers between CUDA device buffers and MPI. Configuration
# verifies that the selected MPI implementation advertises CUDA-aware support.
option(ELPA_ENABLE_CUDA_AWARE_MPI "Enable CUDA-aware MPI" OFF)

# CUDA builds use CMake's standard CMAKE_CUDA_ARCHITECTURES cache variable.
# Packagers should set it explicitly when targeting a reproducible set of GPU
# architectures. No ELPA-specific architecture-FLAGS option is provided.

# =================================================================================================
# BLAS, LAPACK, ScaLAPACK, and oneMKL
# =================================================================================================

# Select the BLAS/LAPACK provider.
#
#   AUTO      Use CMake's normal BLAS/LAPACK discovery.
#   MKL       Require Intel oneMKL.
#   OPENBLAS  Require OpenBLAS.
set(ELPA_BLAS_VENDOR "AUTO" CACHE STRING "BLAS and LAPACK provider")
set_property(
  CACHE ELPA_BLAS_VENDOR
  PROPERTY STRINGS AUTO MKL OPENBLAS
)

# Select the BLAS/LAPACK integer interface. ILP64 also enables ELPA's 64-bit
# integer math paths.
set(ELPA_BLAS_INTERFACE "LP64" CACHE STRING "BLAS and LAPACK integer interface")
set_property(
  CACHE ELPA_BLAS_INTERFACE
  PROPERTY STRINGS LP64 ILP64
)

# Select the oneMKL threading layer. This setting is used when oneMKL is
# selected explicitly or resolved by ELPA_BLAS_VENDOR=AUTO.
set(ELPA_MKL_THREADING "SEQUENTIAL" CACHE STRING "oneMKL threading layer")
set_property(
  CACHE ELPA_MKL_THREADING
  PROPERTY STRINGS SEQUENTIAL THREADED
)

# Select the ScaLAPACK provider used by MPI builds.
#
#   AUTO     Use oneMKL ScaLAPACK with oneMKL BLAS; otherwise search for a
#            standalone ScaLAPACK implementation.
#   MKL      Require oneMKL ScaLAPACK and BLACS.
#   GENERIC  Require a standalone ScaLAPACK implementation.
set(ELPA_SCALAPACK_VENDOR "AUTO" CACHE STRING "ScaLAPACK provider")
set_property(
  CACHE ELPA_SCALAPACK_VENDOR
  PROPERTY STRINGS AUTO MKL GENERIC
)

# Select the oneMKL BLACS ABI. AUTO derives the ABI from the detected MPI
# implementation. This setting is used only with oneMKL ScaLAPACK.
set(ELPA_MKL_BLACS "AUTO" CACHE STRING "oneMKL BLACS ABI")
set_property(
  CACHE ELPA_MKL_BLACS
  PROPERTY STRINGS AUTO OPENMPI INTELMPI MPICH
)

# =================================================================================================
# Validation
# =================================================================================================

# Normalize and validate a STRING cache variable with a fixed set of values.
function(_elpa_validate_choice variable)
  string(TOUPPER "${${variable}}" _elpa_value)
  get_property(_elpa_help CACHE "${variable}" PROPERTY HELPSTRING)
  set("${variable}" "${_elpa_value}" CACHE STRING "${_elpa_help}" FORCE)

  if(NOT _elpa_value IN_LIST ARGN)
    list(JOIN ARGN ", " _elpa_choices)
    message(
      FATAL_ERROR
        "Invalid ${variable} value '${_elpa_value}'. Supported values: ${_elpa_choices}."
    )
  endif()
endfunction()

_elpa_validate_choice(ELPA_CPU_KERNELS AUTO GENERIC)
_elpa_validate_choice(ELPA_ACCELERATOR NONE CUDA HIP SYCL OPENMP)
_elpa_validate_choice(ELPA_BLAS_VENDOR AUTO MKL OPENBLAS)
_elpa_validate_choice(ELPA_BLAS_INTERFACE LP64 ILP64)
_elpa_validate_choice(ELPA_MKL_THREADING SEQUENTIAL THREADED)
_elpa_validate_choice(ELPA_SCALAPACK_VENDOR AUTO MKL GENERIC)
_elpa_validate_choice(ELPA_MKL_BLACS AUTO OPENMPI INTELMPI MPICH)

if(ELPA_USE_64BIT_MPI_INTEGERS AND NOT ELPA_USE_MPI)
  message(FATAL_ERROR "ELPA_USE_64BIT_MPI_INTEGERS requires ELPA_USE_MPI=ON")
endif()

if(ELPA_USE_PAPI AND NOT ELPA_ENABLE_TIMINGS)
  message(FATAL_ERROR "ELPA_USE_PAPI requires ELPA_ENABLE_TIMINGS=ON")
endif()

if(ELPA_SCALAPACK_VENDOR STREQUAL "MKL" AND ELPA_BLAS_VENDOR STREQUAL "OPENBLAS")
  message(FATAL_ERROR
    "ELPA_SCALAPACK_VENDOR=MKL is incompatible with ELPA_BLAS_VENDOR=OPENBLAS"
  )
endif()

if(ELPA_ACCELERATOR MATCHES "^(SYCL|OPENMP)$" AND ELPA_BLAS_VENDOR STREQUAL "OPENBLAS")
  message(FATAL_ERROR
    "ELPA_ACCELERATOR=${ELPA_ACCELERATOR} is incompatible with ELPA_BLAS_VENDOR=OPENBLAS"
  )
endif()

if(ELPA_ENABLE_CUDA_AWARE_MPI AND NOT ELPA_USE_MPI)
  message(FATAL_ERROR "ELPA_ENABLE_CUDA_AWARE_MPI requires ELPA_USE_MPI=ON")
endif()

if(ELPA_ENABLE_CUDA_AWARE_MPI AND NOT ELPA_ACCELERATOR STREQUAL "CUDA")
  message(FATAL_ERROR "ELPA_ENABLE_CUDA_AWARE_MPI requires ELPA_ACCELERATOR=CUDA")
endif()
