# SPDX-License-Identifier: LGPL-3.0-or-later

cmake_minimum_required(VERSION 3.22)

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
  ELPA
  VERSION "${ELPA_VERSION}"
  DESCRIPTION "Eigenvalue SoLvers for Petaflop-Applications"
  HOMEPAGE_URL "https://elpa.mpcdf.mpg.de"
  LANGUAGES C CXX Fortran)

include(CMakePackageConfigHelpers)
include(CTest)
include(GNUInstallDirs)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include("${CMAKE_CURRENT_SOURCE_DIR}/CMakeOptions.cmake")
include(CheckCompilerSupport)
include(CompilerConfiguration)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "RelWithDebInfo")
endif()

# =================================================================================================
# Dependencies
# =================================================================================================

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

if(ELPA_ACCELERATOR MATCHES "^(SYCL|OPENMP)$" AND NOT ELPA_BLAS_VENDOR_RESOLVED
                                                  STREQUAL "MKL")
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
    set(ELPA_SCALAPACK_TARGET SCALAPACK::SCALAPACK)
  endif()
endif()

if(ELPA_USE_OPENMP OR ELPA_ACCELERATOR STREQUAL "OPENMP")
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

# =================================================================================================
# Targets
# =================================================================================================

set(ELPA_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/elpa")
set(ELPA_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/elpa")

add_subdirectory(src)

# =================================================================================================
# Installation and package export
# =================================================================================================

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
  DESTINATION "${ELPA_INSTALL_INCLUDEDIR}")

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

# pkg-config cannot represent CMake imported targets. For static CPU builds,
# record the concrete dependency libraries so `pkg-config --static` remains
# useful to non-CMake consumers.
set(_elpa_pc_private_libraries)
if(NOT BUILD_SHARED_LIBS)
  if(ELPA_BLAS_VENDOR_RESOLVED STREQUAL "MKL")
    list(APPEND _elpa_pc_private_libraries ${MKL_LINK_LIBRARIES})
  else()
    list(APPEND _elpa_pc_private_libraries ${BLAS_LIBRARIES}
         ${LAPACK_LIBRARIES})
  endif()

  if(ELPA_USE_MPI)
    if(ELPA_SCALAPACK_VENDOR_RESOLVED STREQUAL "MKL")
      list(APPEND _elpa_pc_private_libraries ${MKL_SCALAPACK_LINK_LIBRARIES})
    else()
      list(APPEND _elpa_pc_private_libraries ${SCALAPACK_LIBRARIES}
           ${MPI_C_LIBRARIES} ${MPI_Fortran_LIBRARIES})
    endif()
  endif()
  if(ELPA_USE_OPENMP)
    list(APPEND _elpa_pc_private_libraries ${OpenMP_C_LIBRARIES}
         ${OpenMP_CXX_LIBRARIES} ${OpenMP_Fortran_LIBRARIES})
  endif()
  if(ELPA_USE_PAPI)
    list(APPEND _elpa_pc_private_libraries "${PAPI_LIBRARY}")
  endif()

  list(APPEND _elpa_pc_private_libraries
       ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
  if(ELPA_ACCELERATOR MATCHES "^(SYCL|OPENMP)$")
    list(APPEND _elpa_pc_private_libraries ${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES})
  endif()
  list(REMOVE_ITEM _elpa_pc_private_libraries "")
endif()

set(ELPA_PC_PRIVATE_LIBS)
foreach(_elpa_library IN LISTS _elpa_pc_private_libraries)
  if(_elpa_library MATCHES "^-" OR IS_ABSOLUTE "${_elpa_library}")
    string(APPEND ELPA_PC_PRIVATE_LIBS " ${_elpa_library}")
  elseif(NOT _elpa_library MATCHES "::")
    string(APPEND ELPA_PC_PRIVATE_LIBS " -l${_elpa_library}")
  endif()
endforeach()
string(STRIP "${ELPA_PC_PRIVATE_LIBS}" ELPA_PC_PRIVATE_LIBS)

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

configure_file(cmake/elpa.pc.in "elpa.pc" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/elpa.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

# =================================================================================================
# Configuration summary
# =================================================================================================

if(ELPA_USE_MPI)
  if(ELPA_HAVE_MPI_MODULE)
    set(_elpa_mpi_interface "mpi module")
  else()
    set(_elpa_mpi_interface mpif.h)
  endif()
  if(ELPA_MPI_ARGUMENT_MISMATCH_FLAG)
    string(APPEND _elpa_mpi_interface " (${ELPA_MPI_ARGUMENT_MISMATCH_FLAG})")
  endif()
else()
  set(_elpa_mpi_interface disabled)
endif()

message(STATUS "ELPA ${PROJECT_VERSION}")
message(STATUS "  MPI interface: ${_elpa_mpi_interface}")
message(STATUS "  OpenMP:        ${ELPA_USE_OPENMP}")
message(
  STATUS
    "  BLAS:          ${ELPA_BLAS_VENDOR} -> ${ELPA_BLAS_VENDOR_RESOLVED} ${ELPA_BLAS_INTERFACE}"
)
if(ELPA_USE_MPI)
  message(
    STATUS
      "  ScaLAPACK:     ${ELPA_SCALAPACK_VENDOR} -> ${ELPA_SCALAPACK_VENDOR_RESOLVED}"
  )
else()
  message(STATUS "  ScaLAPACK:     disabled")
endif()
message(STATUS "  CPU kernels:   ${ELPA_CPU_KERNELS}")
message(STATUS "  Accelerator:   ${ELPA_ACCELERATOR}")
