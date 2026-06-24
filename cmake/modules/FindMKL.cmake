# SPDX-License-Identifier: LGPL-3.0-or-later

include(FindPackageHandleStandardArgs)

set(
  MKL_ROOT
  "$ENV{MKLROOT}"
  CACHE PATH "Intel oneMKL installation prefix"
)

set(_mkl_root_hints "${MKL_ROOT}" "$ENV{MKLROOT}")
find_path(
  MKL_INCLUDE_DIR
  NAMES mkl.h
  HINTS ${_mkl_root_hints}
  PATH_SUFFIXES include mkl/include
)

if(ELPA_BLAS_INTERFACE STREQUAL "ILP64")
  set(_mkl_interface_suffix ilp64)
else()
  set(_mkl_interface_suffix lp64)
endif()

if(ELPA_MKL_FORTRAN_INTERFACE STREQUAL "GNU")
  set(_mkl_interface_name "mkl_gf_${_mkl_interface_suffix}")
else()
  set(_mkl_interface_name "mkl_intel_${_mkl_interface_suffix}")
endif()

if(ELPA_MKL_THREADING STREQUAL "THREADED")
  find_package(OpenMP REQUIRED COMPONENTS C)
  if(ELPA_MKL_FORTRAN_INTERFACE STREQUAL "GNU")
    set(_mkl_thread_name mkl_gnu_thread)
  else()
    set(_mkl_thread_name mkl_intel_thread)
  endif()
else()
  set(_mkl_thread_name mkl_sequential)
endif()

find_package(Threads REQUIRED)

function(_mkl_find_library variable name)
  find_library(
    ${variable}
    NAMES ${name}
    HINTS ${_mkl_root_hints}
    PATH_SUFFIXES lib lib/intel64 mkl/lib mkl/lib/intel64
  )
  mark_as_advanced(${variable})
endfunction()

_mkl_find_library(MKL_INTERFACE_LIBRARY "${_mkl_interface_name}")
_mkl_find_library(MKL_THREAD_LIBRARY "${_mkl_thread_name}")
_mkl_find_library(MKL_CORE_LIBRARY mkl_core)

set(MKL_BLAS_FOUND FALSE)
set(MKL_LAPACK_FOUND FALSE)
if(MKL_INCLUDE_DIR AND MKL_INTERFACE_LIBRARY AND MKL_THREAD_LIBRARY AND MKL_CORE_LIBRARY)
  set(MKL_BLAS_FOUND TRUE)
  set(MKL_LAPACK_FOUND TRUE)
endif()

set(
  _mkl_required_variables
  MKL_INCLUDE_DIR
  MKL_INTERFACE_LIBRARY
  MKL_THREAD_LIBRARY
  MKL_CORE_LIBRARY
)

if("SYCL" IN_LIST MKL_FIND_COMPONENTS)
  _mkl_find_library(MKL_SYCL_LIBRARY mkl_sycl)
  find_package(OpenCL REQUIRED)

  set(MKL_SYCL_FOUND FALSE)
  if(MKL_SYCL_LIBRARY AND OpenCL_FOUND)
    set(MKL_SYCL_FOUND TRUE)
  endif()
  list(APPEND _mkl_required_variables MKL_SYCL_LIBRARY)
endif()

find_package_handle_standard_args(
  MKL
  REQUIRED_VARS ${_mkl_required_variables}
  HANDLE_COMPONENTS
)

if(MKL_FOUND AND NOT TARGET MKL::MKL)
  add_library(MKL::MKL INTERFACE IMPORTED)
  target_include_directories(MKL::MKL INTERFACE "${MKL_INCLUDE_DIR}")
  target_link_libraries(
    MKL::MKL
    INTERFACE
      "${MKL_INTERFACE_LIBRARY}"
      "${MKL_THREAD_LIBRARY}"
      "${MKL_CORE_LIBRARY}"
      Threads::Threads
      "${CMAKE_DL_LIBS}"
      m
  )

  if(ELPA_BLAS_INTERFACE STREQUAL "ILP64")
    target_compile_definitions(MKL::MKL INTERFACE MKL_ILP64)
  endif()
  if(ELPA_MKL_THREADING STREQUAL "THREADED")
    target_link_libraries(MKL::MKL INTERFACE OpenMP::OpenMP_C)
  endif()
endif()

if(MKL_FOUND AND MKL_SYCL_FOUND AND NOT TARGET MKL::SYCL)
  add_library(MKL::SYCL INTERFACE IMPORTED)
  target_link_libraries(
    MKL::SYCL
    INTERFACE
      "${MKL_SYCL_LIBRARY}"
      MKL::MKL
      OpenCL::OpenCL
  )
endif()

mark_as_advanced(MKL_ROOT MKL_INCLUDE_DIR)
