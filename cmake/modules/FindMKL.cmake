# SPDX-License-Identifier: LGPL-3.0-only

#[=======================================================================[.rst:
FindMKL
-------

Find the oneMKL link closure requested by ELPA.  The following components are
supported:

``BLAS`` and ``LAPACK``
  The host BLAS/LAPACK interface, available as ``MKL::MKL`` (and the
  ``MKL::BLAS`` and ``MKL::LAPACK`` aliases).

``SCALAPACK``
  oneMKL ScaLAPACK and the BLACS library matching the active MPI, available as
  ``MKL::ScaLAPACK`` (and the ``MKL::SCALAPACK`` alias).

``SYCL``
  The oneMKL SYCL interface, available as ``MKL::SYCL``.

Search hints are ``MKL_ROOT``, ``MKLROOT``, and the ``MKLROOT`` environment
variable.  The selected libraries follow these ELPA cache variables:

``ELPA_BLAS_INTERFACE``
  ``lp64`` or ``ilp64``.

``ELPA_MKL_THREADING``
  ``auto`` (default), ``sequential``, ``gnu``, ``intel``, or ``tbb``.
  ``auto`` follows ``ELPA_USE_OPENMP`` and selects the matching GNU or Intel
  OpenMP layer; other compilers use sequential MKL.

``ELPA_MKL_LINK``
  ``dynamic`` (default) or ``static``.

``ELPA_MKL_BLACS``
  ``auto``, ``openmpi``, ``intelmpi``, or ``mpich``.  Set this explicitly when
  cross compiling and requesting ``SCALAPACK``.

``MKL_THREADING`` reports the resolved layer (``sequential``, ``gnu``, ``intel``,
or ``tbb``). Option values use the lowercase spellings shown above.

#]=======================================================================]

include(FindPackageHandleStandardArgs)

if(NOT MKL_FIND_COMPONENTS)
  set(MKL_FIND_COMPONENTS BLAS LAPACK)
endif()

if(NOT DEFINED ELPA_BLAS_INTERFACE)
  set(ELPA_BLAS_INTERFACE lp64)
endif()
if(NOT DEFINED ELPA_MKL_THREADING)
  set(ELPA_MKL_THREADING auto)
endif()
set(ELPA_MKL_THREADING
    "${ELPA_MKL_THREADING}"
    CACHE STRING "oneMKL threading layer")
set_property(CACHE ELPA_MKL_THREADING PROPERTY STRINGS auto sequential gnu
                                               intel tbb)

if(NOT DEFINED ELPA_MKL_LINK)
  set(ELPA_MKL_LINK dynamic)
endif()
if(NOT DEFINED ELPA_MKL_BLACS)
  set(ELPA_MKL_BLACS auto)
endif()

if(NOT ELPA_BLAS_INTERFACE MATCHES "^(lp64|ilp64)$")
  message(FATAL_ERROR "ELPA_BLAS_INTERFACE must be lp64 or ilp64")
endif()
if(NOT ELPA_MKL_THREADING MATCHES "^(auto|sequential|gnu|intel|tbb)$")
  message(
    FATAL_ERROR
      "ELPA_MKL_THREADING must be auto, sequential, gnu, intel, or tbb")
endif()
if(NOT ELPA_MKL_LINK MATCHES "^(dynamic|static)$")
  message(FATAL_ERROR "ELPA_MKL_LINK must be dynamic or static")
endif()
if(NOT ELPA_MKL_BLACS MATCHES "^(auto|openmpi|intelmpi|mpich)$")
  message(
    FATAL_ERROR "ELPA_MKL_BLACS must be auto, openmpi, intelmpi, or mpich")
endif()

# GNU Fortran uses the GNU interface except on macOS, where oneMKL does not ship
# it.  Persisting this choice also lets a static ELPA package reconstruct the
# same oneMKL closure in a consumer that did not enable Fortran.
if(NOT DEFINED ELPA_MKL_FORTRAN_INTERFACE)
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    set(ELPA_MKL_FORTRAN_INTERFACE gnu)
  else()
    set(ELPA_MKL_FORTRAN_INTERFACE intel)
  endif()
endif()
if(ELPA_MKL_FORTRAN_INTERFACE STREQUAL "gnu" AND NOT APPLE)
  set(_mkl_interface_name "mkl_gf_${ELPA_BLAS_INTERFACE}")
else()
  set(_mkl_interface_name "mkl_intel_${ELPA_BLAS_INTERFACE}")
endif()

set(MKL_ROOT
    "$ENV{MKLROOT}"
    CACHE PATH "Intel oneMKL installation prefix")
set(_mkl_root_hints "${MKL_ROOT}" "${MKLROOT}" "$ENV{MKLROOT}")
list(REMOVE_ITEM _mkl_root_hints "")
list(REMOVE_DUPLICATES _mkl_root_hints)

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(_mkl_arch ia32)
else()
  set(_mkl_arch intel64)
endif()
set(_mkl_arch_suffixes "${_mkl_arch}")

if(WIN32)
  list(APPEND _mkl_arch_suffixes "${_mkl_arch}_win")
  set(_mkl_library_prefix "")
  if(ELPA_MKL_LINK STREQUAL "static")
    set(_mkl_library_suffix ".lib")
  else()
    set(_mkl_library_suffix "_dll.lib")
  endif()
elseif(APPLE)
  list(APPEND _mkl_arch_suffixes "${_mkl_arch}_mac")
  set(_mkl_library_prefix "lib")
  if(ELPA_MKL_LINK STREQUAL "static")
    set(_mkl_library_suffix ".a")
  else()
    set(_mkl_library_suffix ".dylib")
  endif()
else()
  list(APPEND _mkl_arch_suffixes "${_mkl_arch}_lin")
  set(_mkl_library_prefix "lib")
  if(ELPA_MKL_LINK STREQUAL "static")
    set(_mkl_library_suffix ".a")
  else()
    set(_mkl_library_suffix ".so")
  endif()
endif()

set(_mkl_search_paths)
foreach(_mkl_root IN LISTS _mkl_root_hints)
  list(APPEND _mkl_search_paths "${_mkl_root}" "${_mkl_root}/lib"
       "${_mkl_root}/mkl/lib" "${_mkl_root}/compiler/lib")
endforeach()
list(REMOVE_DUPLICATES _mkl_search_paths)

find_path(
  MKL_INCLUDE_DIR
  NAMES mkl.h
  HINTS ${_mkl_root_hints}
  PATH_SUFFIXES include mkl/include NO_CACHE)

function(_elpa_mkl_find_library variable name)
  find_library(
    _mkl_library
    NAMES "${_mkl_library_prefix}${name}${_mkl_library_suffix}"
    HINTS ${_mkl_search_paths}
    PATH_SUFFIXES ${_mkl_arch_suffixes} NO_CACHE)
  set(${variable}
      "${_mkl_library}"
      PARENT_SCOPE)
endfunction()

set(_mkl_threading_supported TRUE)
set(_mkl_thread_dependency_found TRUE)
set(_mkl_thread_targets)

# The compiler's OpenMP layer is also used to check explicit threaded requests.
set(_mkl_openmp_layer)
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  set(_mkl_openmp_layer gnu)
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "^(Intel|IntelLLVM)$")
  set(_mkl_openmp_layer intel)
endif()

set(MKL_THREADING "${ELPA_MKL_THREADING}")
if(MKL_THREADING STREQUAL "auto")
  if((DEFINED ELPA_USE_OPENMP AND NOT ELPA_USE_OPENMP) OR NOT _mkl_openmp_layer)
    set(MKL_THREADING sequential)
  else()
    set(MKL_THREADING "${_mkl_openmp_layer}")
  endif()
endif()

if(MKL_THREADING STREQUAL "sequential")
  set(_mkl_thread_name mkl_sequential)
elseif(MKL_THREADING STREQUAL "tbb")
  set(_mkl_thread_name mkl_tbb_thread)
  find_package(TBB QUIET CONFIG HINTS "${TBB_ROOT}" "$ENV{TBBROOT}")
  if(TARGET TBB::tbb)
    list(APPEND _mkl_thread_targets TBB::tbb)
  else()
    set(_mkl_thread_dependency_found FALSE)
  endif()
else()
  set(_mkl_thread_name "mkl_${MKL_THREADING}_thread")
  if(CMAKE_Fortran_COMPILER_LOADED)
    if(NOT MKL_THREADING STREQUAL _mkl_openmp_layer)
      set(_mkl_threading_supported FALSE)
    else()
      find_package(OpenMP QUIET COMPONENTS Fortran)
      if(TARGET OpenMP::OpenMP_Fortran)
        list(APPEND _mkl_thread_targets OpenMP::OpenMP_Fortran)
      else()
        set(_mkl_thread_dependency_found FALSE)
      endif()
    endif()
  else()
    # A C-only static consumer uses the layer recorded when ELPA was built.
    if(MKL_THREADING STREQUAL "gnu")
      set(_mkl_openmp_runtime_name gomp)
    else()
      set(_mkl_openmp_runtime_name iomp5)
    endif()
    find_library(
      _mkl_openmp_runtime_library
      NAMES ${_mkl_openmp_runtime_name}
      HINTS ${_mkl_search_paths} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
      PATH_SUFFIXES ${_mkl_arch_suffixes} NO_CACHE)
    if(_mkl_openmp_runtime_library)
      list(APPEND _mkl_thread_targets "${_mkl_openmp_runtime_library}")
    else()
      set(_mkl_thread_dependency_found FALSE)
    endif()
  endif()
endif()

find_package(Threads QUIET)

_elpa_mkl_find_library(MKL_INTERFACE_LIBRARY "${_mkl_interface_name}")
if(_mkl_threading_supported)
  _elpa_mkl_find_library(MKL_THREAD_LIBRARY "${_mkl_thread_name}")
endif()
_elpa_mkl_find_library(MKL_CORE_LIBRARY mkl_core)

set(_mkl_base_found FALSE)
if(MKL_INCLUDE_DIR
   AND MKL_INTERFACE_LIBRARY
   AND MKL_THREAD_LIBRARY
   AND MKL_CORE_LIBRARY
   AND Threads_FOUND
   AND _mkl_threading_supported
   AND _mkl_thread_dependency_found)
  set(_mkl_base_found TRUE)
endif()

set(MKL_BLAS_FOUND ${_mkl_base_found})
set(MKL_LAPACK_FOUND ${_mkl_base_found})
set(MKL_SCALAPACK_FOUND FALSE)
set(MKL_SYCL_FOUND FALSE)

function(_elpa_mkl_link_group output)
  set(_mkl_link ${ARGN})
  if(ELPA_MKL_LINK STREQUAL "static")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24
       AND (CMAKE_LINK_GROUP_USING_RESCAN_SUPPORTED
            OR CMAKE_Fortran_LINK_GROUP_USING_RESCAN_SUPPORTED))
      list(JOIN ARGN "," _mkl_group_items)
      set(_mkl_link "$<LINK_GROUP:RESCAN,${_mkl_group_items}>")
    elseif(UNIX AND NOT APPLE)
      set(_mkl_link -Wl,--start-group ${ARGN} -Wl,--end-group)
    else()
      set(_mkl_link ${ARGN} ${ARGN} ${ARGN})
    endif()
  endif()
  set(${output}
      "${_mkl_link}"
      PARENT_SCOPE)
endfunction()

set(_mkl_base_libraries "${MKL_INTERFACE_LIBRARY}" "${MKL_THREAD_LIBRARY}"
                        "${MKL_CORE_LIBRARY}")
_elpa_mkl_link_group(_mkl_base_link ${_mkl_base_libraries})

set(_mkl_system_targets Threads::Threads)
if(UNIX)
  list(APPEND _mkl_system_targets m)
  if(ELPA_MKL_LINK STREQUAL "static" AND CMAKE_DL_LIBS)
    list(APPEND _mkl_system_targets ${CMAKE_DL_LIBS})
  endif()
endif()

# Keep one CMake link closure for both result variables and imported targets.
set(MKL_LINK_LIBRARIES ${_mkl_base_link} ${_mkl_thread_targets}
                       ${_mkl_system_targets})

if("SCALAPACK" IN_LIST MKL_FIND_COMPONENTS)
  set(_mkl_mpi_interface_found TRUE)
  set(MKL_BLACS_ABI "${ELPA_MKL_BLACS}")

  if(ELPA_MKL_BLACS STREQUAL "auto")
    if(CMAKE_CROSSCOMPILING)
      set(_mkl_mpi_interface_found FALSE)
    elseif(MPI_Fortran_LIBRARY_VERSION_STRING MATCHES "Open MPI")
      set(MKL_BLACS_ABI openmpi)
    elseif(MPI_Fortran_LIBRARY_VERSION_STRING MATCHES "Intel\\(R\\) MPI|Intel MPI|MPICH|HYDRA")
      set(MKL_BLACS_ABI intelmpi)
    else()
      set(_mkl_mpi_interface_found FALSE)
    endif()
  elseif(ELPA_MKL_BLACS STREQUAL "mpich")
    set(MKL_BLACS_ABI intelmpi)
  endif()

  if(_mkl_mpi_interface_found)
    if(MKL_BLACS_ABI STREQUAL "openmpi")
      set(_mkl_blacs_name "mkl_blacs_openmpi_${ELPA_BLAS_INTERFACE}")
    elseif(APPLE AND MKL_BLACS_ABI STREQUAL "mpich")
      set(_mkl_blacs_name "mkl_blacs_mpich_${ELPA_BLAS_INTERFACE}")
    else()
      # Intel MPI and MPICH share the intelmpi BLACS ABI on Linux.
      set(_mkl_blacs_name "mkl_blacs_intelmpi_${ELPA_BLAS_INTERFACE}")
    endif()

    _elpa_mkl_find_library(MKL_SCALAPACK_LIBRARY
                           "mkl_scalapack_${ELPA_BLAS_INTERFACE}")
    _elpa_mkl_find_library(MKL_BLACS_LIBRARY "${_mkl_blacs_name}")
  endif()

  if(_mkl_base_found
     AND _mkl_mpi_interface_found
     AND MKL_SCALAPACK_LIBRARY
     AND MKL_BLACS_LIBRARY
     AND TARGET MPI::MPI_Fortran)
    set(MKL_SCALAPACK_FOUND TRUE)
  endif()

  set(_mkl_cluster_libraries
      "${MKL_SCALAPACK_LIBRARY}" "${MKL_INTERFACE_LIBRARY}"
      "${MKL_THREAD_LIBRARY}" "${MKL_CORE_LIBRARY}" "${MKL_BLACS_LIBRARY}")
  _elpa_mkl_link_group(_mkl_cluster_link ${_mkl_cluster_libraries})
  set(MKL_SCALAPACK_LINK_LIBRARIES
      ${_mkl_cluster_link} MPI::MPI_Fortran ${_mkl_thread_targets}
      ${_mkl_system_targets})
endif()

if("SYCL" IN_LIST MKL_FIND_COMPONENTS)
  _elpa_mkl_find_library(MKL_SYCL_LIBRARY mkl_sycl)
  find_package(OpenCL QUIET)
  if(_mkl_base_found
     AND MKL_SYCL_LIBRARY
     AND TARGET OpenCL::OpenCL)
    set(MKL_SYCL_FOUND TRUE)
  endif()
endif()

foreach(_mkl_component IN LISTS MKL_FIND_COMPONENTS)
  if(NOT _mkl_component MATCHES "^(BLAS|LAPACK|SCALAPACK|SYCL)$")
    set(MKL_${_mkl_component}_FOUND FALSE)
  endif()
endforeach()

set(_mkl_failure_reason)
if(NOT _mkl_threading_supported)
  string(CONCAT _mkl_failure_reason
                "ELPA_MKL_THREADING=${ELPA_MKL_THREADING} is incompatible with "
                "${CMAKE_Fortran_COMPILER_ID} Fortran compiler.")
elseif(NOT _mkl_thread_dependency_found)
  string(CONCAT _mkl_failure_reason "The runtime dependency for "
                "ELPA_MKL_THREADING=${ELPA_MKL_THREADING} was not found.")
elseif("SCALAPACK" IN_LIST MKL_FIND_COMPONENTS AND NOT _mkl_mpi_interface_found)
  string(CONCAT _mkl_failure_reason
                "Cannot determine the oneMKL BLACS ABI. Set ELPA_MKL_BLACS "
                "explicitly, especially when cross compiling.")
endif()

find_package_handle_standard_args(
  MKL
  REQUIRED_VARS MKL_INCLUDE_DIR _mkl_base_found
  HANDLE_COMPONENTS REASON_FAILURE_MESSAGE "${_mkl_failure_reason}")

if(MKL_FOUND AND NOT TARGET MKL::MKL)
  add_library(MKL::MKL INTERFACE IMPORTED)
  set_target_properties(
    MKL::MKL PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
                        INTERFACE_LINK_LIBRARIES "${MKL_LINK_LIBRARIES}")
  if(ELPA_BLAS_INTERFACE STREQUAL "ilp64")
    set_property(
      TARGET MKL::MKL
      APPEND
      PROPERTY INTERFACE_COMPILE_DEFINITIONS MKL_ILP64)
  endif()
endif()

if(TARGET MKL::MKL AND NOT TARGET MKL::BLAS)
  add_library(MKL::BLAS ALIAS MKL::MKL)
  add_library(MKL::LAPACK ALIAS MKL::MKL)
endif()

if(MKL_SCALAPACK_FOUND AND NOT TARGET MKL::ScaLAPACK)
  add_library(MKL::ScaLAPACK INTERFACE IMPORTED)
  set_target_properties(
    MKL::ScaLAPACK
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
               INTERFACE_LINK_LIBRARIES "${MKL_SCALAPACK_LINK_LIBRARIES}")
  add_library(MKL::SCALAPACK ALIAS MKL::ScaLAPACK)
endif()

if(MKL_SYCL_FOUND AND NOT TARGET MKL::SYCL)
  add_library(MKL::SYCL INTERFACE IMPORTED)
  set_target_properties(
    MKL::SYCL
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
               INTERFACE_LINK_LIBRARIES
               "${MKL_SYCL_LIBRARY};MKL::MKL;OpenCL::OpenCL")
endif()

mark_as_advanced(MKL_ROOT)
