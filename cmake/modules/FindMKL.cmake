# SPDX-License-Identifier: LGPL-3.0-or-later

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
  ``LP64`` or ``ILP64``.

``ELPA_MKL_THREADING``
  ``SEQUENTIAL``, ``THREADED``/``OPENMP``, ``GNU``, ``INTEL``, or ``TBB``.

``ELPA_MKL_LINK``
  ``DYNAMIC`` or ``STATIC``.

``ELPA_MKL_BLACS``
  ``AUTO``, ``OPENMPI``, ``INTELMPI``, or ``MPICH``.  Set this explicitly when
  cross compiling and requesting ``SCALAPACK``.

#]=======================================================================]

include(FindPackageHandleStandardArgs)

if(NOT MKL_FIND_COMPONENTS)
  set(MKL_FIND_COMPONENTS BLAS LAPACK)
endif()

if(NOT DEFINED ELPA_BLAS_INTERFACE)
  set(ELPA_BLAS_INTERFACE LP64)
endif()
if(NOT DEFINED ELPA_MKL_THREADING)
  set(ELPA_MKL_THREADING SEQUENTIAL)
endif()
if(NOT DEFINED ELPA_MKL_LINK)
  set(ELPA_MKL_LINK DYNAMIC)
endif()
if(NOT DEFINED ELPA_MKL_BLACS)
  set(ELPA_MKL_BLACS AUTO)
endif()

string(TOUPPER "${ELPA_BLAS_INTERFACE}" _mkl_integer_interface)
string(TOUPPER "${ELPA_MKL_THREADING}" _mkl_threading)
string(TOUPPER "${ELPA_MKL_LINK}" _mkl_link_mode)
string(TOUPPER "${ELPA_MKL_BLACS}" _mkl_blacs_request)

if(NOT _mkl_integer_interface MATCHES "^(LP64|ILP64)$")
  message(FATAL_ERROR "ELPA_BLAS_INTERFACE must be LP64 or ILP64")
endif()
if(NOT _mkl_threading MATCHES "^(SEQUENTIAL|THREADED|OPENMP|GNU|INTEL|TBB)$")
  message(
    FATAL_ERROR "ELPA_MKL_THREADING must be SEQUENTIAL, THREADED, OPENMP, GNU, "
                "INTEL, or TBB")
endif()
if(NOT _mkl_link_mode MATCHES "^(DYNAMIC|STATIC)$")
  message(FATAL_ERROR "ELPA_MKL_LINK must be DYNAMIC or STATIC")
endif()
if(NOT _mkl_blacs_request MATCHES "^(AUTO|OPENMPI|INTELMPI|MPICH)$")
  message(
    FATAL_ERROR "ELPA_MKL_BLACS must be AUTO, OPENMPI, INTELMPI, or MPICH")
endif()

string(TOLOWER "${_mkl_integer_interface}" _mkl_integer_suffix)

# GNU Fortran uses the GNU interface except on macOS, where oneMKL does not ship
# it.  Persisting this choice also lets a static ELPA package reconstruct the
# same oneMKL closure in a consumer that did not enable Fortran.
if(NOT DEFINED ELPA_MKL_FORTRAN_INTERFACE)
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    set(ELPA_MKL_FORTRAN_INTERFACE GNU)
  else()
    set(ELPA_MKL_FORTRAN_INTERFACE INTEL)
  endif()
endif()
string(TOUPPER "${ELPA_MKL_FORTRAN_INTERFACE}" _mkl_fortran_interface)
if(NOT DEFINED ELPA_MKL_COMPILER_ID)
  set(ELPA_MKL_COMPILER_ID "${CMAKE_Fortran_COMPILER_ID}")
endif()
string(TOUPPER "${ELPA_MKL_COMPILER_ID}" _mkl_compiler_id)
if(_mkl_fortran_interface STREQUAL "GNU" AND NOT APPLE)
  set(_mkl_interface_name "mkl_gf_${_mkl_integer_suffix}")
else()
  set(_mkl_interface_name "mkl_intel_${_mkl_integer_suffix}")
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
  if(_mkl_link_mode STREQUAL "STATIC")
    set(_mkl_library_suffix ".lib")
  else()
    set(_mkl_library_suffix "_dll.lib")
  endif()
elseif(APPLE)
  list(APPEND _mkl_arch_suffixes "${_mkl_arch}_mac")
  set(_mkl_library_prefix "lib")
  if(_mkl_link_mode STREQUAL "STATIC")
    set(_mkl_library_suffix ".a")
  else()
    set(_mkl_library_suffix ".dylib")
  endif()
else()
  list(APPEND _mkl_arch_suffixes "${_mkl_arch}_lin")
  set(_mkl_library_prefix "lib")
  if(_mkl_link_mode STREQUAL "STATIC")
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
set(_mkl_thread_runtime_libraries)

if(_mkl_threading STREQUAL "SEQUENTIAL")
  set(_mkl_thread_name mkl_sequential)
elseif(_mkl_threading STREQUAL "TBB")
  set(_mkl_thread_name mkl_tbb_thread)
  find_package(TBB QUIET CONFIG HINTS "${TBB_ROOT}" "$ENV{TBBROOT}")
  if(TARGET TBB::tbb)
    list(APPEND _mkl_thread_targets TBB::tbb)
    list(APPEND _mkl_thread_runtime_libraries tbb)
  else()
    set(_mkl_thread_dependency_found FALSE)
  endif()
else()
  if(_mkl_threading MATCHES "^(THREADED|OPENMP)$")
    if(_mkl_compiler_id STREQUAL "GNU")
      set(_mkl_openmp_flavour GNU)
    elseif(_mkl_compiler_id MATCHES "^(INTEL|INTELLLVM)$")
      set(_mkl_openmp_flavour INTEL)
    else()
      set(_mkl_threading_supported FALSE)
    endif()
  elseif(_mkl_threading STREQUAL "GNU")
    set(_mkl_openmp_flavour GNU)
    if(NOT _mkl_compiler_id STREQUAL "GNU")
      set(_mkl_threading_supported FALSE)
    endif()
  else()
    set(_mkl_openmp_flavour INTEL)
    if(NOT _mkl_compiler_id MATCHES "^(INTEL|INTELLLVM)$")
      set(_mkl_threading_supported FALSE)
    endif()
  endif()

  if(_mkl_threading_supported AND _mkl_openmp_flavour STREQUAL "GNU")
    set(_mkl_thread_name mkl_gnu_thread)
    if(NOT _mkl_fortran_interface STREQUAL "GNU")
      set(_mkl_threading_supported FALSE)
    endif()
  elseif(_mkl_threading_supported)
    set(_mkl_thread_name mkl_intel_thread)
    if(_mkl_fortran_interface STREQUAL "GNU")
      set(_mkl_threading_supported FALSE)
    endif()
  endif()

  if(_mkl_threading_supported AND CMAKE_Fortran_COMPILER_LOADED)
    find_package(OpenMP QUIET COMPONENTS Fortran)
    if(TARGET OpenMP::OpenMP_Fortran)
      list(APPEND _mkl_thread_targets OpenMP::OpenMP_Fortran)
      list(APPEND _mkl_thread_runtime_libraries ${OpenMP_Fortran_LIBRARIES})
    else()
      set(_mkl_thread_dependency_found FALSE)
    endif()
  elseif(_mkl_threading_supported)
    # Static ELPA packages can be consumed by C-only projects.  In that case the
    # recorded oneMKL Fortran interface still identifies the required OpenMP
    # runtime even though an OpenMP Fortran target cannot be created.
    if(_mkl_openmp_flavour STREQUAL "GNU")
      set(_mkl_openmp_runtime_names gomp)
    else()
      set(_mkl_openmp_runtime_names iomp5)
    endif()
    find_library(
      _mkl_openmp_runtime_library
      NAMES ${_mkl_openmp_runtime_names}
      HINTS ${_mkl_search_paths} ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
      PATH_SUFFIXES ${_mkl_arch_suffixes} NO_CACHE)
    if(_mkl_openmp_runtime_library)
      list(APPEND _mkl_thread_targets "${_mkl_openmp_runtime_library}")
      list(APPEND _mkl_thread_runtime_libraries
           "${_mkl_openmp_runtime_library}")
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
  if(_mkl_link_mode STREQUAL "STATIC"
     AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24
     AND (CMAKE_LINK_GROUP_USING_RESCAN_SUPPORTED
          OR CMAKE_Fortran_LINK_GROUP_USING_RESCAN_SUPPORTED))
    list(JOIN ARGN "," _mkl_group_items)
    set(${output}
        "$<LINK_GROUP:RESCAN,${_mkl_group_items}>"
        PARENT_SCOPE)
  elseif(
    _mkl_link_mode STREQUAL "STATIC"
    AND UNIX
    AND NOT APPLE)
    set(${output}
        "-Wl,--start-group;${ARGN};-Wl,--end-group"
        PARENT_SCOPE)
  elseif(_mkl_link_mode STREQUAL "STATIC")
    # CMake 3.22 and 3.23 cannot express a rescan group portably.  Repeating the
    # archives retains compatibility on platforms without GNU link groups.
    set(${output}
        "${ARGN};${ARGN};${ARGN}"
        PARENT_SCOPE)
  else()
    set(${output}
        "${ARGN}"
        PARENT_SCOPE)
  endif()
endfunction()

set(_mkl_base_libraries "${MKL_INTERFACE_LIBRARY}" "${MKL_THREAD_LIBRARY}"
                        "${MKL_CORE_LIBRARY}")
_elpa_mkl_link_group(_mkl_base_link ${_mkl_base_libraries})

set(_mkl_system_targets Threads::Threads)
set(_mkl_system_libraries ${CMAKE_THREAD_LIBS_INIT})
if(UNIX)
  list(APPEND _mkl_system_targets m)
  list(APPEND _mkl_system_libraries m)
  if(_mkl_link_mode STREQUAL "STATIC" AND CMAKE_DL_LIBS)
    list(APPEND _mkl_system_targets ${CMAKE_DL_LIBS})
    list(APPEND _mkl_system_libraries ${CMAKE_DL_LIBS})
  endif()
endif()

if(_mkl_link_mode STREQUAL "STATIC"
   AND UNIX
   AND NOT APPLE)
  set(_mkl_base_raw_link -Wl,--start-group ${_mkl_base_libraries}
                         -Wl,--end-group)
elseif(_mkl_link_mode STREQUAL "STATIC")
  set(_mkl_base_raw_link ${_mkl_base_libraries} ${_mkl_base_libraries}
                         ${_mkl_base_libraries})
else()
  set(_mkl_base_raw_link ${_mkl_base_libraries})
endif()
set(MKL_LINK_LIBRARIES ${_mkl_base_raw_link} ${_mkl_thread_runtime_libraries}
                       ${_mkl_system_libraries})

if("SCALAPACK" IN_LIST MKL_FIND_COMPONENTS)
  set(_mkl_mpi_interface_found TRUE)
  set(MKL_BLACS_ABI "${_mkl_blacs_request}")

  if(_mkl_blacs_request STREQUAL "AUTO")
    if(CMAKE_CROSSCOMPILING)
      set(_mkl_mpi_interface_found FALSE)
    else()
      string(
        JOIN
        ";"
        _mkl_mpi_identity
        "${MPI_C_LIBRARY_VERSION_STRING}"
        "${MPI_Fortran_LIBRARY_VERSION_STRING}"
        "${MPI_C_LIBRARIES}"
        "${MPI_Fortran_LIBRARIES}"
        "${MPI_C_COMPILER}"
        "${MPI_Fortran_COMPILER}"
        "${MPIEXEC_EXECUTABLE}")
      string(TOLOWER "${_mkl_mpi_identity}" _mkl_mpi_identity)
      if(_mkl_mpi_identity MATCHES "open[ _-]?mpi|(^|[/;])ompi")
        set(MKL_BLACS_ABI OPENMPI)
      elseif(_mkl_mpi_identity MATCHES "intel(\\(r\\))?[ _-]?mpi|(^|[/;])impi")
        set(MKL_BLACS_ABI INTELMPI)
      elseif(_mkl_mpi_identity MATCHES "mpich|hydra")
        set(MKL_BLACS_ABI MPICH)
      else()
        set(_mkl_mpi_interface_found FALSE)
      endif()
    endif()
  endif()

  if(_mkl_mpi_interface_found)
    if(MKL_BLACS_ABI STREQUAL "OPENMPI")
      set(_mkl_blacs_name "mkl_blacs_openmpi_${_mkl_integer_suffix}")
    elseif(APPLE AND MKL_BLACS_ABI STREQUAL "MPICH")
      set(_mkl_blacs_name "mkl_blacs_mpich_${_mkl_integer_suffix}")
    else()
      # Intel MPI and MPICH share the intelmpi BLACS ABI on Linux.
      set(_mkl_blacs_name "mkl_blacs_intelmpi_${_mkl_integer_suffix}")
    endif()

    _elpa_mkl_find_library(MKL_SCALAPACK_LIBRARY
                           "mkl_scalapack_${_mkl_integer_suffix}")
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
  if(_mkl_link_mode STREQUAL "STATIC"
     AND UNIX
     AND NOT APPLE)
    set(_mkl_cluster_raw_link -Wl,--start-group ${_mkl_cluster_libraries}
                              -Wl,--end-group)
  elseif(_mkl_link_mode STREQUAL "STATIC")
    set(_mkl_cluster_raw_link
        ${_mkl_cluster_libraries} ${_mkl_cluster_libraries}
        ${_mkl_cluster_libraries})
  else()
    set(_mkl_cluster_raw_link ${_mkl_cluster_libraries})
  endif()
  set(MKL_SCALAPACK_LINK_LIBRARIES
      ${_mkl_cluster_raw_link} ${MPI_Fortran_LIBRARIES}
      ${_mkl_thread_runtime_libraries} ${_mkl_system_libraries})
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
  string(
    CONCAT _mkl_failure_reason
           "ELPA_MKL_THREADING=${ELPA_MKL_THREADING} is incompatible with "
           "${ELPA_MKL_COMPILER_ID} Fortran and the "
           "${ELPA_MKL_FORTRAN_INTERFACE} oneMKL interface.")
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
    MKL::MKL
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
               INTERFACE_LINK_LIBRARIES
               "${_mkl_base_link};${_mkl_thread_targets};${_mkl_system_targets}"
  )
  if(_mkl_integer_interface STREQUAL "ILP64")
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
  set(_mkl_cluster_target_link ${_mkl_cluster_link} MPI::MPI_Fortran
                               ${_mkl_thread_targets} ${_mkl_system_targets})
  set_target_properties(
    MKL::ScaLAPACK
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
               INTERFACE_LINK_LIBRARIES "${_mkl_cluster_target_link}")
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

mark_as_advanced(
  MKL_ROOT
  MKL_INCLUDE_DIR
  MKL_INTERFACE_LIBRARY
  MKL_THREAD_LIBRARY
  MKL_CORE_LIBRARY
  MKL_SCALAPACK_LIBRARY
  MKL_BLACS_LIBRARY
  MKL_SYCL_LIBRARY)
