# SPDX-License-Identifier: LGPL-3.0-only

cmake_minimum_required(VERSION 3.22)
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

# CMake build metadata.
#
# Keep these values independent from every other build-system input. A release
# that changes the CMake ABI or API must update this block together with the
# corresponding CMake package metadata.
set(ELPA_VERSION "2026.02.002")
set(ELPA_ABI_CURRENT 24)
set(ELPA_ABI_REVISION 1)
set(ELPA_ABI_AGE 5)
set(ELPA_API_VERSION 20260202)
set(ELPA_AUTOTUNE_VERSION 202600202)
set(ELPA_EARLIEST_API_VERSION 20170403)
set(ELPA_EARLIEST_AUTOTUNE_VERSION 20171201)

math(EXPR ELPA_SOVERSION "${ELPA_ABI_CURRENT} - ${ELPA_ABI_AGE}")
set(ELPA_LIBRARY_VERSION
    "${ELPA_SOVERSION}.${ELPA_ABI_AGE}.${ELPA_ABI_REVISION}")

project(
  elpa
  VERSION "${ELPA_VERSION}"
  DESCRIPTION "Eigenvalue SoLvers for Petaflop-Applications"
  HOMEPAGE_URL "https://elpa.mpcdf.mpg.de"
  LANGUAGES C CXX Fortran)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# ==============================================================================
# Build options
# ==============================================================================

option(BUILD_SHARED_LIBS "Build ELPA as a shared library" ON)
option(BUILD_TESTING "Build ELPA tests" OFF)
option(ELPA_BUILD_TOOLS "Build ELPA command-line utilities" ON)

# Testing
set(ELPA_TEST_MPI_PROCS
    2
    CACHE STRING "MPI processes per test")
set(ELPA_TEST_OMP_THREADS
    1
    CACHE STRING "OpenMP threads per test process")

# Parallelism
option(ELPA_USE_OPENMP "Enable OpenMP threading" ON)
option(ELPA_USE_MPI "Enable MPI support" ON)
option(ELPA_USE_64BIT_MPI_INTEGERS "Use 64-bit MPI integer arguments" OFF)

# Optional features
option(ELPA_ENABLE_SINGLE_PRECISION "Build single-precision interfaces" ON)
option(ELPA_ENABLE_AUTOTUNING "Enable autotuning support" ON)
option(ELPA_ENABLE_SKEWSYMMETRIC "Enable skew-symmetric solvers" ON)
option(ELPA_ENABLE_BAND_TO_FULL_BLOCKING
       "Enable blocking in the ELPA2 band-to-full transformation" ON)
option(ELPA_ENABLE_TIMINGS "Enable detailed timing support" ON)
option(ELPA_ENABLE_AFFINITY "Enable affinity diagnostics" ON)
option(ELPA_USE_ASSUMED_SIZE "Use assumed-size Fortran arrays" ON)
option(ELPA_USE_FORTRAN_2008 "Enable Fortran 2008 code paths" ON)
option(ELPA_OPTIONAL_C_ERROR_ARGUMENT "Make the C API error argument optional"
       OFF)
option(ELPA_STORE_BUILD_CONFIG "Embed the build configuration in ELPA" OFF)
option(ELPA_USE_PAPI "Enable PAPI performance counters" OFF)

# CPU kernels
set(ELPA_CPU_KERNELS
    "AUTO"
    CACHE STRING "CPU kernel selection")
set_property(CACHE ELPA_CPU_KERNELS PROPERTY STRINGS AUTO GENERIC)
option(ELPA_ENABLE_HETEROGENEOUS_CLUSTER
       "Enable heterogeneous-cluster CPU dispatch" OFF)

# Accelerator backend
set(ELPA_ACCELERATOR
    "NONE"
    CACHE STRING "Accelerator backend")
set_property(CACHE ELPA_ACCELERATOR PROPERTY STRINGS NONE CUDA HIP SYCL
                                             IOMP_OFFLOAD)
option(ELPA_ENABLE_GPU_STREAMS "Enable accelerator streams or queues" ON)
option(ELPA_ENABLE_CUDA_AWARE_MPI "Enable CUDA-aware MPI" OFF)

# Numerical libraries
set(ELPA_BLAS_VENDOR
    "AUTO"
    CACHE STRING "BLAS and LAPACK provider")
set_property(CACHE ELPA_BLAS_VENDOR PROPERTY STRINGS AUTO MKL OPENBLAS GENERIC
                                             CUSTOM)

set(ELPA_BLAS_LIBRARIES
    ""
    CACHE STRING "Custom BLAS link libraries")
set(ELPA_LAPACK_LIBRARIES
    ""
    CACHE STRING "Custom LAPACK link libraries")

set(ELPA_BLAS_INTERFACE
    "LP64"
    CACHE STRING "BLAS and LAPACK integer interface")
set_property(CACHE ELPA_BLAS_INTERFACE PROPERTY STRINGS LP64 ILP64)

set(ELPA_MKL_THREADING
    "SEQUENTIAL"
    CACHE STRING "oneMKL threading layer")
set_property(
  CACHE ELPA_MKL_THREADING
  PROPERTY STRINGS
           SEQUENTIAL
           THREADED
           OPENMP
           GNU
           INTEL
           TBB)

set(ELPA_MKL_LINK
    "DYNAMIC"
    CACHE STRING "oneMKL link mode")
set_property(CACHE ELPA_MKL_LINK PROPERTY STRINGS DYNAMIC STATIC)

set(ELPA_SCALAPACK_VENDOR
    "AUTO"
    CACHE STRING "ScaLAPACK provider")
set_property(CACHE ELPA_SCALAPACK_VENDOR PROPERTY STRINGS AUTO MKL GENERIC
                                                  CUSTOM)

set(ELPA_SCALAPACK_LIBRARIES
    ""
    CACHE STRING "Custom ScaLAPACK link libraries")

set(ELPA_MKL_BLACS
    "AUTO"
    CACHE STRING "oneMKL BLACS ABI")
set_property(CACHE ELPA_MKL_BLACS PROPERTY STRINGS AUTO OPENMPI INTELMPI MPICH)

# Option validation
function(_elpa_validate_choice variable)
  string(TOUPPER "${${variable}}" _elpa_value)
  get_property(
    _elpa_help
    CACHE "${variable}"
    PROPERTY HELPSTRING)
  set("${variable}"
      "${_elpa_value}"
      CACHE STRING "${_elpa_help}" FORCE)

  if(NOT _elpa_value IN_LIST ARGN)
    list(JOIN ARGN ", " _elpa_choices)
    message(
      FATAL_ERROR
        "Invalid ${variable} value '${_elpa_value}'. Supported values: ${_elpa_choices}."
    )
  endif()
endfunction()

_elpa_validate_choice(ELPA_CPU_KERNELS AUTO GENERIC)
_elpa_validate_choice(ELPA_ACCELERATOR NONE CUDA HIP SYCL IOMP_OFFLOAD)
_elpa_validate_choice(ELPA_BLAS_VENDOR AUTO MKL OPENBLAS GENERIC CUSTOM)
_elpa_validate_choice(ELPA_BLAS_INTERFACE LP64 ILP64)
_elpa_validate_choice(
  ELPA_MKL_THREADING
  SEQUENTIAL
  THREADED
  OPENMP
  GNU
  INTEL
  TBB)
_elpa_validate_choice(ELPA_MKL_LINK DYNAMIC STATIC)
_elpa_validate_choice(ELPA_SCALAPACK_VENDOR AUTO MKL GENERIC CUSTOM)
_elpa_validate_choice(ELPA_MKL_BLACS AUTO OPENMPI INTELMPI MPICH)

if(ELPA_USE_64BIT_MPI_INTEGERS AND NOT ELPA_USE_MPI)
  message(FATAL_ERROR "ELPA_USE_64BIT_MPI_INTEGERS requires ELPA_USE_MPI=ON")
endif()

if(ELPA_USE_PAPI AND NOT ELPA_ENABLE_TIMINGS)
  message(FATAL_ERROR "ELPA_USE_PAPI requires ELPA_ENABLE_TIMINGS=ON")
endif()

if(ELPA_BLAS_VENDOR STREQUAL "CUSTOM" AND NOT ELPA_BLAS_LIBRARIES)
  message(FATAL_ERROR "ELPA_BLAS_VENDOR=CUSTOM requires ELPA_BLAS_LIBRARIES")
endif()

if(ELPA_BLAS_VENDOR STREQUAL "CUSTOM" AND NOT ELPA_LAPACK_LIBRARIES)
  message(FATAL_ERROR "ELPA_BLAS_VENDOR=CUSTOM requires ELPA_LAPACK_LIBRARIES")
endif()

if(ELPA_SCALAPACK_VENDOR STREQUAL "CUSTOM" AND NOT ELPA_SCALAPACK_LIBRARIES)
  message(
    FATAL_ERROR "ELPA_SCALAPACK_VENDOR=CUSTOM requires ELPA_SCALAPACK_LIBRARIES"
  )
endif()

if(ELPA_SCALAPACK_VENDOR STREQUAL "MKL" AND NOT ELPA_BLAS_VENDOR MATCHES
                                            "^(AUTO|MKL)$")
  message(
    FATAL_ERROR
      "ELPA_SCALAPACK_VENDOR=MKL requires ELPA_BLAS_VENDOR=AUTO or MKL")
endif()

if(ELPA_ACCELERATOR MATCHES "^(SYCL|IOMP_OFFLOAD)$" AND NOT ELPA_BLAS_VENDOR
                                                        MATCHES "^(AUTO|MKL)$")
  message(FATAL_ERROR "ELPA_ACCELERATOR=${ELPA_ACCELERATOR} requires "
                      "ELPA_BLAS_VENDOR=AUTO or MKL")
endif()

if(ELPA_ENABLE_CUDA_AWARE_MPI AND NOT ELPA_USE_MPI)
  message(FATAL_ERROR "ELPA_ENABLE_CUDA_AWARE_MPI requires ELPA_USE_MPI=ON")
endif()

if(ELPA_ENABLE_CUDA_AWARE_MPI AND NOT ELPA_ACCELERATOR STREQUAL "CUDA")
  message(
    FATAL_ERROR "ELPA_ENABLE_CUDA_AWARE_MPI requires ELPA_ACCELERATOR=CUDA")
endif()

include(CheckCompilerSupport)
include(CompilerConfiguration)
include(CTest)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "RelWithDebInfo")
endif()

# ==============================================================================
# Dependencies
# ==============================================================================

if(ELPA_BLAS_INTERFACE STREQUAL "ILP64")
  set(BLA_SIZEOF_INTEGER 8)
else()
  set(BLA_SIZEOF_INTEGER 4)
endif()

if(ELPA_USE_MPI)
  find_package(MPI REQUIRED COMPONENTS C Fortran)
endif()

if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  set(ELPA_MKL_FORTRAN_INTERFACE GNU)
else()
  set(ELPA_MKL_FORTRAN_INTERFACE INTEL)
endif()
set(ELPA_MKL_COMPILER_ID "${CMAKE_Fortran_COMPILER_ID}")

find_package(Lapack REQUIRED)
set(ELPA_BLAS_TARGET BLAS::BLAS LAPACK::LAPACK)

if(ELPA_ACCELERATOR MATCHES "^(SYCL|IOMP_OFFLOAD)$"
   AND NOT ELPA_BLAS_VENDOR_RESOLVED STREQUAL "MKL")
  message(
    FATAL_ERROR
      "ELPA_ACCELERATOR=${ELPA_ACCELERATOR} requires oneMKL BLAS/LAPACK. "
      "Set ELPA_BLAS_VENDOR=MKL or make oneMKL discoverable for AUTO.")
endif()

if(ELPA_ACCELERATOR STREQUAL "SYCL")
  find_package(MKL REQUIRED COMPONENTS SYCL)
endif()

if(ELPA_USE_MPI)
  if(ELPA_SCALAPACK_VENDOR STREQUAL "MKL" AND NOT ELPA_BLAS_VENDOR_RESOLVED
                                              STREQUAL "MKL")
    message(
      FATAL_ERROR
        "ELPA_SCALAPACK_VENDOR=MKL requires oneMKL BLAS/LAPACK. "
        "Set ELPA_BLAS_VENDOR=MKL or make oneMKL discoverable for AUTO.")
  endif()

  set(ELPA_SCALAPACK_VENDOR_RESOLVED "${ELPA_SCALAPACK_VENDOR}")
  if(ELPA_SCALAPACK_VENDOR STREQUAL "AUTO")
    if(ELPA_BLAS_VENDOR_RESOLVED STREQUAL "MKL")
      find_package(MKL QUIET COMPONENTS SCALAPACK)
      if(MKL_SCALAPACK_FOUND)
        set(ELPA_SCALAPACK_VENDOR_RESOLVED MKL)
      else()
        set(ELPA_SCALAPACK_VENDOR_RESOLVED GENERIC)
      endif()
    else()
      set(ELPA_SCALAPACK_VENDOR_RESOLVED GENERIC)
    endif()
  endif()

  if(ELPA_SCALAPACK_VENDOR_RESOLVED STREQUAL "MKL")
    # Keep the full oneMKL cluster closure in FindMKL.  FindScalapack is only
    # responsible for standalone ScaLAPACK implementations.
    if(NOT TARGET MKL::ScaLAPACK)
      find_package(MKL REQUIRED COMPONENTS SCALAPACK)
    endif()
    set(ELPA_SCALAPACK_TARGET MKL::ScaLAPACK)
  else()
    set(SCALAPACK_BLAS_TARGETS ${ELPA_BLAS_TARGET})
    find_package(Scalapack REQUIRED)
    set(ELPA_SCALAPACK_TARGET ScaLAPACK::scalapack)
  endif()
endif()

if(ELPA_USE_OPENMP OR ELPA_ACCELERATOR STREQUAL "IOMP_OFFLOAD")
  find_package(OpenMP REQUIRED COMPONENTS C CXX Fortran)
endif()

if(ELPA_USE_PAPI)
  find_path(PAPI_INCLUDE_DIR NAMES papi.h REQUIRED)
  find_library(PAPI_LIBRARY NAMES papi REQUIRED)

  if(NOT TARGET PAPI::PAPI)
    add_library(PAPI::PAPI UNKNOWN IMPORTED)
    set_target_properties(
      PAPI::PAPI PROPERTIES IMPORTED_LOCATION "${PAPI_LIBRARY}"
                            INTERFACE_INCLUDE_DIRECTORIES "${PAPI_INCLUDE_DIR}")
  endif()
endif()

if(ELPA_ACCELERATOR STREQUAL "CUDA")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
elseif(ELPA_ACCELERATOR STREQUAL "HIP")
  enable_language(HIP)
  find_package(hip REQUIRED CONFIG)
  find_package(rocblas REQUIRED CONFIG)
endif()

elpa_check_compiler_support()

# ==============================================================================
# Targets
# ==============================================================================

set(ELPA_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/elpa")
set(ELPA_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/elpa")
set(ELPA_INSTALL_FMODDIR
    "${ELPA_INSTALL_INCLUDEDIR}"
    CACHE PATH "ELPA Fortran modules installation directory")

add_subdirectory(src)
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

# ==============================================================================
# Installation and package export
# ==============================================================================

install(
  TARGETS elpa
  EXPORT elpa-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  INCLUDES
  DESTINATION "${ELPA_INSTALL_INCLUDEDIR}")

if(TARGET elpa2_print_kernels)
  install(TARGETS elpa2_print_kernels
          RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(FILES man/elpa2_print_kernels.1
          DESTINATION "${CMAKE_INSTALL_MANDIR}/man1")
endif()

install(
  FILES elpa/elpa.h
        elpa/elpa_explicit_name.h
        elpa/elpa_generic.h
        elpa/elpa_simd_constants.h
        "${CMAKE_CURRENT_BINARY_DIR}/elpa/elpa_configured_options.h"
        "${CMAKE_CURRENT_BINARY_DIR}/elpa/elpa_constants.h"
        "${CMAKE_CURRENT_BINARY_DIR}/elpa/elpa_generated.h"
        "${CMAKE_CURRENT_BINARY_DIR}/elpa/elpa_generated_c_api.h"
        "${CMAKE_CURRENT_BINARY_DIR}/elpa/elpa_version.h"
        src/helpers/lapack_interfaces.h
        src/helpers/scalapack_interfaces.h
  DESTINATION "${ELPA_INSTALL_INCLUDEDIR}")

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/modules/elpa.mod"
        "${CMAKE_CURRENT_BINARY_DIR}/modules/elpa_api.mod"
        "${CMAKE_CURRENT_BINARY_DIR}/modules/elpa_constants.mod"
  DESTINATION "${ELPA_INSTALL_FMODDIR}")

if(BUILD_SHARED_LIBS)
  set(ELPA_BUILD_STATIC OFF)
else()
  set(ELPA_BUILD_STATIC ON)
endif()

# A static export must recreate imported dependency targets in a consuming
# project. Store the resolved provider rather than the original AUTO request.
set(ELPA_BUILD_BLAS_VENDOR "${ELPA_BLAS_VENDOR_RESOLVED}")

set(ELPA_BUILD_MKL_BLACS "${ELPA_MKL_BLACS}")
if(ELPA_USE_MPI AND ELPA_SCALAPACK_VENDOR_RESOLVED STREQUAL "MKL")
  set(ELPA_BUILD_MKL_BLACS "${MKL_BLACS_ABI}")
endif()

configure_package_config_file(
  cmake/elpa-config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/elpa-config.cmake"
  INSTALL_DESTINATION "${ELPA_INSTALL_CMAKEDIR}")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/elpa-config-version.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/elpa-config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/elpa-config-version.cmake"
        DESTINATION "${ELPA_INSTALL_CMAKEDIR}")
install(
  DIRECTORY cmake/modules/
  DESTINATION "${ELPA_INSTALL_CMAKEDIR}/modules"
  FILES_MATCHING
  PATTERN "Find*.cmake")
install(
  EXPORT elpa-targets
  FILE elpa-targets.cmake
  NAMESPACE ELPA::
  DESTINATION "${ELPA_INSTALL_CMAKEDIR}")

if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
  set(ELPA_PC_LIBDIR "${CMAKE_INSTALL_LIBDIR}")
else()
  set(ELPA_PC_LIBDIR "\${prefix}/${CMAKE_INSTALL_LIBDIR}")
endif()
if(IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
  set(ELPA_PC_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}")
else()
  set(ELPA_PC_INCLUDEDIR "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
endif()
if(IS_ABSOLUTE "${ELPA_INSTALL_FMODDIR}")
  set(ELPA_PC_FMODDIR "${ELPA_INSTALL_FMODDIR}")
else()
  set(ELPA_PC_FMODDIR "\${prefix}/${ELPA_INSTALL_FMODDIR}")
endif()
configure_file(cmake/elpa.pc.in "elpa.pc" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/elpa.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

# Install documentation
install(DIRECTORY "${PROJECT_SOURCE_DIR}/documentation/"
        DESTINATION "${CMAKE_INSTALL_DOCDIR}")
