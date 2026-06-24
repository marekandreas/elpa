# SPDX-License-Identifier: LGPL-3.0-or-later

include(FindPackageHandleStandardArgs)

if(NOT DEFINED SCALAPACK_PROVIDER)
  set(SCALAPACK_PROVIDER GENERIC)
endif()
string(TOUPPER "${SCALAPACK_PROVIDER}" SCALAPACK_PROVIDER)

if(SCALAPACK_PROVIDER STREQUAL "MKL")
  if(SCALAPACK_BLAS_INTERFACE STREQUAL "ILP64")
    set(_scalapack_mkl_integer_suffix ilp64)
  else()
    set(_scalapack_mkl_integer_suffix lp64)
  endif()

  set(_scalapack_mkl_root_hints "${MKL_ROOT}" "$ENV{MKLROOT}")
  foreach(_scalapack_blas_library IN LISTS BLAS_LIBRARIES LAPACK_LIBRARIES)
    if(IS_ABSOLUTE "${_scalapack_blas_library}")
      get_filename_component(_scalapack_mkl_library_dir "${_scalapack_blas_library}" DIRECTORY)
      list(APPEND _scalapack_mkl_root_hints "${_scalapack_mkl_library_dir}")
    endif()
  endforeach()
  list(REMOVE_ITEM _scalapack_mkl_root_hints "")
  list(REMOVE_DUPLICATES _scalapack_mkl_root_hints)

  find_library(
    SCALAPACK_LIBRARY
    NAMES "mkl_scalapack_${_scalapack_mkl_integer_suffix}"
    HINTS ${_scalapack_mkl_root_hints}
    PATH_SUFFIXES lib lib/intel64 mkl/lib mkl/lib/intel64
  )

  if(NOT DEFINED SCALAPACK_MKL_BLACS)
    set(SCALAPACK_MKL_BLACS AUTO)
  endif()
  string(TOUPPER "${SCALAPACK_MKL_BLACS}" _scalapack_mkl_blacs)

  if(_scalapack_mkl_blacs STREQUAL "AUTO")
    string(
      JOIN ";"
      _scalapack_mpi_identity
      "${MPI_C_LIBRARY_VERSION_STRING}"
      "${MPI_Fortran_LIBRARY_VERSION_STRING}"
      "${MPI_C_LIBRARIES}"
      "${MPI_Fortran_LIBRARIES}"
      "${MPI_C_COMPILER}"
      "${MPI_Fortran_COMPILER}"
      "${MPIEXEC_EXECUTABLE}"
    )
    string(TOLOWER "${_scalapack_mpi_identity}" _scalapack_mpi_identity)

    if(_scalapack_mpi_identity MATCHES "open[ _-]?mpi|(^|[/;])ompi")
      set(SCALAPACK_MKL_BLACS_ABI OPENMPI)
    elseif(_scalapack_mpi_identity MATCHES "intel[ _-]?mpi|(^|[/;])impi")
      set(SCALAPACK_MKL_BLACS_ABI INTELMPI)
    elseif(_scalapack_mpi_identity MATCHES "mpich|hydra")
      set(SCALAPACK_MKL_BLACS_ABI MPICH)
    else()
      message(
        FATAL_ERROR
          "Could not determine the oneMKL BLACS ABI from the detected MPI. "
          "Set ELPA_MKL_BLACS to OPENMPI, INTELMPI, or MPICH."
      )
    endif()
  else()
    set(SCALAPACK_MKL_BLACS_ABI "${_scalapack_mkl_blacs}")
  endif()

  if(SCALAPACK_MKL_BLACS_ABI STREQUAL "OPENMPI")
    set(_scalapack_mkl_blacs_abi openmpi)
  else()
    # Intel MPI and MPICH share oneMKL's intelmpi BLACS ABI on Linux.
    set(_scalapack_mkl_blacs_abi intelmpi)
  endif()

  find_library(
    SCALAPACK_BLACS_LIBRARY
    NAMES
      "mkl_blacs_${_scalapack_mkl_blacs_abi}_${_scalapack_mkl_integer_suffix}"
    HINTS ${_scalapack_mkl_root_hints}
    PATH_SUFFIXES lib lib/intel64 mkl/lib mkl/lib/intel64
  )

  find_package_handle_standard_args(
    SCALAPACK
    REQUIRED_VARS SCALAPACK_LIBRARY SCALAPACK_BLACS_LIBRARY
  )
else()
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(PC_SCALAPACK QUIET IMPORTED_TARGET GLOBAL scalapack)
  endif()

  if(TARGET PkgConfig::PC_SCALAPACK)
    set(SCALAPACK_LIBRARY PkgConfig::PC_SCALAPACK)
  else()
    set(_scalapack_names scalapack)
    string(
      JOIN ";"
      _scalapack_mpi_identity
      "${MPI_C_LIBRARY_VERSION_STRING}"
      "${MPI_Fortran_LIBRARY_VERSION_STRING}"
      "${MPI_C_LIBRARIES}"
      "${MPI_Fortran_LIBRARIES}"
      "${MPI_C_COMPILER}"
      "${MPI_Fortran_COMPILER}"
    )
    string(TOLOWER "${_scalapack_mpi_identity}" _scalapack_mpi_identity)

    if(_scalapack_mpi_identity MATCHES "open[ _-]?mpi|(^|[/;])ompi")
      list(PREPEND _scalapack_names scalapack-openmpi)
    elseif(_scalapack_mpi_identity MATCHES "mpich|hydra")
      list(PREPEND _scalapack_names scalapack-mpich)
    endif()

    find_library(SCALAPACK_LIBRARY NAMES ${_scalapack_names})
  endif()

  find_package_handle_standard_args(
    SCALAPACK
    REQUIRED_VARS SCALAPACK_LIBRARY
  )
endif()

if(SCALAPACK_FOUND AND NOT TARGET SCALAPACK::SCALAPACK)
  add_library(SCALAPACK::SCALAPACK INTERFACE IMPORTED)

  if(SCALAPACK_PROVIDER STREQUAL "MKL")
    target_link_libraries(
      SCALAPACK::SCALAPACK
      INTERFACE
        "${SCALAPACK_LIBRARY}"
        "${SCALAPACK_BLACS_LIBRARY}"
        ${SCALAPACK_BLAS_TARGETS}
        MPI::MPI_C
        MPI::MPI_Fortran
    )
  else()
    target_link_libraries(SCALAPACK::SCALAPACK INTERFACE "${SCALAPACK_LIBRARY}")
  endif()
endif()

mark_as_advanced(SCALAPACK_LIBRARY SCALAPACK_BLACS_LIBRARY)
