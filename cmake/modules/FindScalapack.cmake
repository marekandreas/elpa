# SPDX-License-Identifier: LGPL-3.0-only

#[=======================================================================[.rst:
FindScalapack
-------------

Find a standalone ScaLAPACK installation and provide
``SCALAPACK::SCALAPACK``.  oneMKL ScaLAPACK is deliberately outside this
module and is provided by the ``SCALAPACK`` component of :module:`FindMKL`.

The module first honors ``ELPA_SCALAPACK_LIBRARIES``, then tries pkg-config,
and finally searches common MPI-specific library names.  The imported target
also carries the selected BLAS/LAPACK and MPI targets so static link closures
remain complete.

Result variables are ``Scalapack_FOUND``, ``SCALAPACK_FOUND``,
``SCALAPACK_LIBRARIES``, and the compatibility variable
``SCALAPACK_LIBRARY``.

#]=======================================================================]

include(FindPackageHandleStandardArgs)

if(NOT TARGET MPI::MPI_Fortran)
  find_package(MPI QUIET COMPONENTS Fortran)
endif()

if(NOT DEFINED SCALAPACK_BLAS_TARGETS)
  find_package(Lapack QUIET)
  if(TARGET BLAS::BLAS AND TARGET LAPACK::LAPACK)
    set(SCALAPACK_BLAS_TARGETS BLAS::BLAS LAPACK::LAPACK)
  endif()
endif()

set(_elpa_scalapack_link_item)
set(_elpa_scalapack_libraries)
set(_elpa_scalapack_include_dirs)

if(ELPA_SCALAPACK_LIBRARIES)
  set(_elpa_scalapack_libraries ${ELPA_SCALAPACK_LIBRARIES})
  set(_elpa_scalapack_link_item ${ELPA_SCALAPACK_LIBRARIES})
else()
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(PC_ELPA_SCALAPACK QUIET IMPORTED_TARGET GLOBAL scalapack)
  endif()

  if(TARGET PkgConfig::PC_ELPA_SCALAPACK)
    set(_elpa_scalapack_libraries ${PC_ELPA_SCALAPACK_LINK_LIBRARIES})
    set(_elpa_scalapack_link_item PkgConfig::PC_ELPA_SCALAPACK)
    set(_elpa_scalapack_include_dirs ${PC_ELPA_SCALAPACK_INCLUDE_DIRS})
  else()
    set(_elpa_scalapack_names)
    string(
      JOIN
      ";"
      _elpa_scalapack_mpi_identity
      "${MPI_C_LIBRARY_VERSION_STRING}"
      "${MPI_Fortran_LIBRARY_VERSION_STRING}"
      "${MPI_C_LIBRARIES}"
      "${MPI_Fortran_LIBRARIES}"
      "${MPI_C_COMPILER}"
      "${MPI_Fortran_COMPILER}")
    string(TOLOWER "${_elpa_scalapack_mpi_identity}"
                   _elpa_scalapack_mpi_identity)

    if(_elpa_scalapack_mpi_identity MATCHES "open[ _-]?mpi|(^|[/;])ompi")
      list(APPEND _elpa_scalapack_names scalapack-openmpi)
    elseif(_elpa_scalapack_mpi_identity MATCHES "mpich|hydra")
      list(APPEND _elpa_scalapack_names scalapack-mpich)
    endif()
    list(APPEND _elpa_scalapack_names scalapack)

    find_library(
      _elpa_scalapack_library
      NAMES ${_elpa_scalapack_names}
      HINTS "${Scalapack_ROOT}" "${SCALAPACK_ROOT}" "$ENV{Scalapack_ROOT}"
            "$ENV{SCALAPACK_ROOT}"
      PATH_SUFFIXES lib lib64 NO_CACHE)
    if(_elpa_scalapack_library)
      set(_elpa_scalapack_libraries "${_elpa_scalapack_library}")
      set(_elpa_scalapack_link_item "${_elpa_scalapack_library}")
    endif()
  endif()
endif()

set(SCALAPACK_LIBRARIES ${_elpa_scalapack_libraries})
if(SCALAPACK_LIBRARIES)
  list(GET SCALAPACK_LIBRARIES 0 SCALAPACK_LIBRARY)
else()
  set(SCALAPACK_LIBRARY "")
endif()
set(SCALAPACK_INCLUDE_DIRS ${_elpa_scalapack_include_dirs})
if(TARGET MPI::MPI_Fortran)
  set(_elpa_scalapack_mpi_found TRUE)
else()
  set(_elpa_scalapack_mpi_found FALSE)
endif()

find_package_handle_standard_args(
  Scalapack REQUIRED_VARS SCALAPACK_LIBRARIES SCALAPACK_BLAS_TARGETS
                          _elpa_scalapack_mpi_found)
set(SCALAPACK_FOUND ${Scalapack_FOUND})

if(Scalapack_FOUND AND NOT TARGET SCALAPACK::SCALAPACK)
  add_library(SCALAPACK::SCALAPACK INTERFACE IMPORTED)
  set_property(
    TARGET SCALAPACK::SCALAPACK
    PROPERTY
      INTERFACE_LINK_LIBRARIES
      "${_elpa_scalapack_link_item};${SCALAPACK_BLAS_TARGETS};MPI::MPI_Fortran")
  if(SCALAPACK_INCLUDE_DIRS)
    set_property(
      TARGET SCALAPACK::SCALAPACK PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                           "${SCALAPACK_INCLUDE_DIRS}")
  endif()
endif()

mark_as_advanced(SCALAPACK_LIBRARY SCALAPACK_LIBRARIES SCALAPACK_INCLUDE_DIRS)
