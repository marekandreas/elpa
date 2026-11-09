# SPDX-License-Identifier: LGPL-3.0-only

#[=======================================================================[.rst:
FindLapack
----------

Find LAPACK for the BLAS provider selected by :module:`FindBlas` and provide
the conventional ``LAPACK::LAPACK`` imported target.

oneMKL and OpenBLAS expose BLAS and LAPACK through the same link closure.  For
generic implementations this module delegates the symbol checks to CMake's
standard, upper-case ``FindLAPACK`` module.  ``custom`` consumes
``ELPA_LAPACK_LIBRARIES``.

Result variables are ``Lapack_FOUND``, ``LAPACK_FOUND``, and
``LAPACK_LIBRARIES``.

#]=======================================================================]

include(FindPackageHandleStandardArgs)

if(NOT Blas_FOUND)
  find_package(Blas QUIET)
endif()

set(_elpa_lapack_link_item)
set(_elpa_lapack_libraries)

if(ELPA_BLAS_VENDOR_RESOLVED MATCHES "^(mkl|openblas)$")
  set(_elpa_lapack_libraries ${BLAS_LIBRARIES})
  set(_elpa_lapack_link_item BLAS::BLAS)
elseif(ELPA_BLAS_VENDOR_RESOLVED STREQUAL "custom")
  set(_elpa_lapack_libraries ${ELPA_LAPACK_LIBRARIES})
  set(_elpa_lapack_link_item ${ELPA_LAPACK_LIBRARIES})
elseif(ELPA_BLAS_VENDOR_RESOLVED STREQUAL "generic")
  find_package(LAPACK QUIET)
  if(LAPACK_FOUND)
    set(_elpa_lapack_libraries ${LAPACK_LIBRARIES})
    set(_elpa_lapack_link_item LAPACK::LAPACK)
  endif()
endif()

set(LAPACK_LIBRARIES ${_elpa_lapack_libraries})
find_package_handle_standard_args(Lapack REQUIRED_VARS LAPACK_LIBRARIES
                                                       BLAS_FOUND)
set(LAPACK_FOUND ${Lapack_FOUND})

if(Lapack_FOUND AND NOT TARGET LAPACK::LAPACK)
  add_library(LAPACK::LAPACK INTERFACE IMPORTED)
  if(_elpa_lapack_link_item STREQUAL "BLAS::BLAS")
    set(_elpa_lapack_link_closure BLAS::BLAS)
  else()
    set(_elpa_lapack_link_closure ${_elpa_lapack_link_item} BLAS::BLAS)
  endif()
  set_property(TARGET LAPACK::LAPACK PROPERTY INTERFACE_LINK_LIBRARIES
                                              "${_elpa_lapack_link_closure}")
  if(BLAS_INCLUDE_DIRS)
    set_property(TARGET LAPACK::LAPACK PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                                "${BLAS_INCLUDE_DIRS}")
  endif()
endif()
