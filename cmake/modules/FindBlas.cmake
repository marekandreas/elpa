# SPDX-License-Identifier: LGPL-3.0-or-later

#[=======================================================================[.rst:
FindBlas
--------

Select the BLAS implementation requested by ``ELPA_BLAS_VENDOR`` and provide
the conventional ``BLAS::BLAS`` imported target.

Supported providers are ``AUTO``, ``MKL``, ``OPENBLAS``, ``GENERIC``, and
``CUSTOM``.  ``AUTO`` tries oneMKL and OpenBLAS before falling back to CMake's
standard ``FindBLAS`` module.  ``CUSTOM`` consumes ``ELPA_BLAS_LIBRARIES``.

Result variables
^^^^^^^^^^^^^^^^

``Blas_FOUND`` and ``BLAS_FOUND``
  Whether a usable BLAS implementation was found.

``BLAS_LIBRARIES``
  The concrete libraries reported by the selected implementation.

``ELPA_BLAS_VENDOR_RESOLVED``
  The provider selected after resolving ``AUTO``.

#]=======================================================================]

include(FindPackageHandleStandardArgs)

if(NOT DEFINED ELPA_BLAS_VENDOR)
  set(ELPA_BLAS_VENDOR AUTO)
endif()
string(TOUPPER "${ELPA_BLAS_VENDOR}" _elpa_blas_vendor)

set(_elpa_blas_link_item)
set(_elpa_blas_include_dirs)
set(_elpa_blas_libraries)
set(_elpa_blas_resolved)

if(_elpa_blas_vendor STREQUAL "CUSTOM")
  set(_elpa_blas_libraries ${ELPA_BLAS_LIBRARIES})
  set(_elpa_blas_link_item ${ELPA_BLAS_LIBRARIES})
  set(_elpa_blas_resolved CUSTOM)
endif()

# oneMKL is intentionally tried first for AUTO.  This keeps BLAS, LAPACK,
# ScaLAPACK, BLACS, and accelerator support on one coherent implementation.
if(NOT _elpa_blas_libraries AND _elpa_blas_vendor MATCHES "^(AUTO|MKL)$")
  if(_elpa_blas_vendor STREQUAL "MKL")
    find_package(MKL REQUIRED COMPONENTS BLAS)
  else()
    find_package(MKL QUIET COMPONENTS BLAS)
  endif()

  if(MKL_FOUND AND TARGET MKL::MKL)
    set(_elpa_blas_libraries ${MKL_LINK_LIBRARIES})
    set(_elpa_blas_link_item MKL::MKL)
    set(_elpa_blas_include_dirs ${MKL_INCLUDE_DIR})
    set(_elpa_blas_resolved MKL)
  endif()
endif()

# Prefer pkg-config for OpenBLAS because it preserves a distribution's chosen
# threading variant and any extra link options.  Fall back to common library
# names and finally an upstream OpenBLAS config package.
if(NOT _elpa_blas_libraries AND _elpa_blas_vendor MATCHES "^(AUTO|OPENBLAS)$")
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    if(ELPA_BLAS_INTERFACE STREQUAL "ILP64")
      pkg_check_modules(PC_ELPA_OPENBLAS QUIET IMPORTED_TARGET GLOBAL
                        openblas64)
    else()
      pkg_check_modules(PC_ELPA_OPENBLAS QUIET IMPORTED_TARGET GLOBAL openblas)
    endif()
  endif()

  if(TARGET PkgConfig::PC_ELPA_OPENBLAS)
    set(_elpa_blas_libraries ${PC_ELPA_OPENBLAS_LINK_LIBRARIES})
    set(_elpa_blas_link_item PkgConfig::PC_ELPA_OPENBLAS)
    set(_elpa_blas_include_dirs ${PC_ELPA_OPENBLAS_INCLUDE_DIRS})
  else()
    if(ELPA_BLAS_INTERFACE STREQUAL "ILP64")
      set(_elpa_openblas_names openblas64 openblas64_ openblas_ilp64
                               openblas64_p)
    else()
      set(_elpa_openblas_names openblas openblas_omp openblas_threads
                               openblas_pthread)
    endif()

    find_library(
      _elpa_openblas_library
      NAMES ${_elpa_openblas_names}
      HINTS "${OpenBLAS_ROOT}" "$ENV{OpenBLAS_ROOT}"
      PATH_SUFFIXES lib lib64 openblas openblas64 NO_CACHE)
    find_path(
      _elpa_openblas_include_dir
      NAMES cblas.h openblas_config.h
      HINTS "${OpenBLAS_ROOT}" "$ENV{OpenBLAS_ROOT}"
      PATH_SUFFIXES include include/openblas openblas NO_CACHE)

    if(_elpa_openblas_library)
      set(_elpa_blas_libraries "${_elpa_openblas_library}")
      set(_elpa_blas_link_item "${_elpa_openblas_library}")
      set(_elpa_blas_include_dirs "${_elpa_openblas_include_dir}")
    else()
      find_package(OpenBLAS QUIET CONFIG)
      if(ELPA_BLAS_INTERFACE STREQUAL "ILP64" AND TARGET OpenBLAS::OpenBLAS64)
        set(_elpa_blas_libraries OpenBLAS::OpenBLAS64)
        set(_elpa_blas_link_item OpenBLAS::OpenBLAS64)
      elseif(TARGET OpenBLAS::OpenBLAS)
        if(NOT ELPA_BLAS_INTERFACE STREQUAL "ILP64")
          set(_elpa_blas_libraries OpenBLAS::OpenBLAS)
          set(_elpa_blas_link_item OpenBLAS::OpenBLAS)
        endif()
      elseif(
        OpenBLAS_FOUND
        AND OpenBLAS_LIBRARIES
        AND NOT ELPA_BLAS_INTERFACE STREQUAL "ILP64")
        set(_elpa_blas_libraries ${OpenBLAS_LIBRARIES})
        set(_elpa_blas_link_item ${OpenBLAS_LIBRARIES})
      endif()
    endif()
  endif()

  if(_elpa_blas_libraries)
    set(_elpa_blas_resolved OPENBLAS)
  endif()
endif()

# The generic path delegates the platform- and compiler-specific symbol checks
# to CMake's standard, upper-case FindBLAS module.
if(NOT _elpa_blas_libraries AND _elpa_blas_vendor MATCHES "^(AUTO|GENERIC)$")
  find_package(BLAS QUIET)
  if(BLAS_FOUND)
    set(_elpa_blas_libraries ${BLAS_LIBRARIES})
    if(TARGET BLAS::BLAS)
      set(_elpa_blas_link_item BLAS::BLAS)
    else()
      set(_elpa_blas_link_item ${BLAS_LIBRARIES})
    endif()
    set(_elpa_blas_resolved GENERIC)
  endif()
endif()

set(BLAS_LIBRARIES ${_elpa_blas_libraries})
set(BLAS_INCLUDE_DIRS ${_elpa_blas_include_dirs})

find_package_handle_standard_args(Blas REQUIRED_VARS BLAS_LIBRARIES
                                                     _elpa_blas_resolved)
set(BLAS_FOUND ${Blas_FOUND})

if(Blas_FOUND)
  set(ELPA_BLAS_VENDOR_RESOLVED "${_elpa_blas_resolved}")
  if(NOT TARGET BLAS::BLAS)
    add_library(BLAS::BLAS INTERFACE IMPORTED)
    set_property(TARGET BLAS::BLAS PROPERTY INTERFACE_LINK_LIBRARIES
                                            "${_elpa_blas_link_item}")
    if(BLAS_INCLUDE_DIRS)
      set_property(TARGET BLAS::BLAS PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                              "${BLAS_INCLUDE_DIRS}")
    endif()
  endif()
endif()

mark_as_advanced(BLAS_LIBRARIES BLAS_INCLUDE_DIRS)
