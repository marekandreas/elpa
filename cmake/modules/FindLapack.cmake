# SPDX-License-Identifier: LGPL-3.0-only

#[=======================================================================[.rst:
FindLapack
----------

Find LAPACK for the BLAS provider selected by :module:`FindBlas` and provide
the conventional ``LAPACK::LAPACK`` imported target.

oneMKL and OpenBLAS expose BLAS and LAPACK through the same link closure.  For
generic implementations this module delegates the symbol checks to CMake's
standard, upper-case ``FindLAPACK`` module.  ``CUSTOM`` consumes
``ELPA_LAPACK_LIBRARIES``.

Result variables are ``Lapack_FOUND``, ``LAPACK_FOUND``, and
``LAPACK_LIBRARIES``.

#]=======================================================================]

include(FindPackageHandleStandardArgs)

find_package(Blas REQUIRED)

set(_elpa_lapack_link_item)
set(_elpa_lapack_libraries)

if(ELPA_BLAS_VENDOR_RESOLVED MATCHES "^(MKL|OPENBLAS)$")
  set(_elpa_lapack_libraries ${BLAS_LIBRARIES})
  set(_elpa_lapack_link_item BLAS::BLAS)
elseif(ELPA_BLAS_VENDOR_RESOLVED STREQUAL "CUSTOM")
  set(_elpa_lapack_libraries ${ELPA_LAPACK_LIBRARIES})
  set(_elpa_lapack_link_item ${ELPA_LAPACK_LIBRARIES})
elseif(ELPA_BLAS_VENDOR_RESOLVED STREQUAL "GENERIC")
  find_package(LAPACK QUIET)
  if(LAPACK_FOUND)
    set(_elpa_lapack_libraries ${LAPACK_LIBRARIES})
    if(TARGET LAPACK::LAPACK)
      set(_elpa_lapack_link_item LAPACK::LAPACK)
    else()
      set(_elpa_lapack_link_item ${LAPACK_LIBRARIES})
    endif()
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

mark_as_advanced(LAPACK_LIBRARIES)
