# SPDX-License-Identifier: LGPL-3.0-or-later

include_guard(GLOBAL)

# Apply compiler options to an ELPA target. Feature macros are deliberately
# not passed as command-line definitions: ELPA's existing sources include the
# private, generated config-f90.h from the build tree.
function(elpa_configure_compiler target)
  target_compile_features(${target} PRIVATE c_std_11 cxx_std_11)

  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    target_compile_options(
      ${target}
      PRIVATE
        $<$<COMPILE_LANGUAGE:Fortran>:-ffree-line-length-none>
    )
  endif()

  if(ELPA_MPI_ARGUMENT_MISMATCH_FLAG)
    target_compile_options(
      ${target}
      PRIVATE
        $<$<COMPILE_LANGUAGE:Fortran>:${ELPA_MPI_ARGUMENT_MISMATCH_FLAG}>
    )
  endif()

  if(ELPA_ACCELERATOR_COMPILE_OPTIONS)
    target_compile_options(
      ${target}
      PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${ELPA_ACCELERATOR_COMPILE_OPTIONS}>
    )
  endif()

  if(ELPA_ACCELERATOR_LINK_OPTIONS)
    target_link_options(${target} PRIVATE ${ELPA_ACCELERATOR_LINK_OPTIONS})
  endif()
endfunction()

# Produce the common configuration macros required by the existing ELPA
# sources. Architecture-specific kernel macros are appended by
# src/CMakeLists.txt after the selected source set is known.
function(elpa_get_config_f90_definitions output_variable)
  set(_elpa_definitions
    "CURRENT_API_VERSION=${ELPA_API_VERSION}"
    "CURRENT_AUTOTUNE_VERSION=${ELPA_AUTOTUNE_VERSION}"
    "EARLIEST_API_VERSION=${ELPA_EARLIEST_API_VERSION}"
    "EARLIEST_AUTOTUNE_VERSION=${ELPA_EARLIEST_AUTOTUNE_VERSION}"
    "ELPA_BUILDTIME=0"
  )

  if(ELPA_USE_MPI)
    list(APPEND _elpa_definitions WITH_MPI)
    if(ELPA_HAVE_MPI_MODULE)
      list(APPEND _elpa_definitions HAVE_MPI_MODULE)
    endif()
  endif()

  if(ELPA_USE_OPENMP)
    list(APPEND _elpa_definitions WITH_OPENMP_TRADITIONAL)
  endif()
  if(ELPA_ENABLE_SINGLE_PRECISION)
    list(APPEND _elpa_definitions WANT_SINGLE_PRECISION_REAL WANT_SINGLE_PRECISION_COMPLEX)
  endif()
  if(ELPA_ENABLE_AUTOTUNING)
    list(APPEND _elpa_definitions ENABLE_AUTOTUNING)
  endif()
  if(ELPA_ENABLE_SKEWSYMMETRIC)
    list(APPEND _elpa_definitions HAVE_SKEWSYMMETRIC)
  endif()
  if(ELPA_ENABLE_BAND_TO_FULL_BLOCKING)
    list(APPEND _elpa_definitions BAND_TO_FULL_BLOCKING)
  endif()
  if(ELPA_ENABLE_TIMINGS)
    list(APPEND _elpa_definitions HAVE_DETAILED_TIMINGS)
  endif()
  if(ELPA_ENABLE_AFFINITY)
    list(APPEND _elpa_definitions HAVE_AFFINITY_CHECKING)
  endif()
  if(ELPA_USE_ASSUMED_SIZE)
    list(APPEND _elpa_definitions USE_ASSUMED_SIZE)
  endif()
  if(ELPA_USE_FORTRAN_2008)
    list(APPEND _elpa_definitions USE_FORTRAN2008)
  endif()
  if(ELPA_OPTIONAL_C_ERROR_ARGUMENT)
    list(APPEND _elpa_definitions OPTIONAL_C_ERROR_ARGUMENT=1)
  endif()
  if(ELPA_STORE_BUILD_CONFIG)
    list(APPEND _elpa_definitions STORE_BUILD_CONFIG)
  endif()
  if(ELPA_USE_PAPI)
    list(APPEND _elpa_definitions HAVE_LIBPAPI)
  endif()
  if(ELPA_USE_64BIT_MPI_INTEGERS)
    list(APPEND _elpa_definitions HAVE_64BIT_INTEGER_MPI_SUPPORT)
  endif()
  if(ELPA_BLAS_INTERFACE STREQUAL "ILP64")
    list(APPEND _elpa_definitions HAVE_64BIT_INTEGER_MATH_SUPPORT)
  endif()
  if(ELPA_ENABLE_HETEROGENEOUS_CLUSTER)
    list(APPEND _elpa_definitions HAVE_HETEROGENEOUS_CLUSTER_SUPPORT)
  endif()
  if(ELPA_HAVE_ISO_FORTRAN_ENV)
    list(APPEND _elpa_definitions HAVE_ISO_FORTRAN_ENV)
  endif()
  if(ELPA_HAVE_ENVIRONMENT_CHECKING)
    list(APPEND _elpa_definitions HAVE_ENVIRONMENT_CHECKING)
  endif()
  if(ELPA_PACK_REAL_TO_COMPLEX)
    list(APPEND _elpa_definitions PACK_REAL_TO_COMPLEX)
  endif()
  if(ELPA_USE_MPI AND ELPA_USE_OPENMP)
    list(APPEND _elpa_definitions THREADING_SUPPORT_CHECK ALLOW_THREAD_LIMITING)
  endif()
  if(ELPA_ACCELERATOR STREQUAL "CUDA")
    list(APPEND _elpa_definitions WITH_NVIDIA_GPU_VERSION WITH_NVIDIA_GPU_KERNEL)
  elseif(ELPA_ACCELERATOR STREQUAL "HIP")
    list(APPEND _elpa_definitions WITH_AMD_GPU_VERSION WITH_AMD_GPU_KERNEL)
  elseif(ELPA_ACCELERATOR STREQUAL "SYCL")
    list(APPEND _elpa_definitions WITH_SYCL_GPU_VERSION)
  elseif(ELPA_ACCELERATOR STREQUAL "IOMP_OFFLOAD")
    list(APPEND _elpa_definitions WITH_OPENMP_OFFLOAD_GPU_VERSION)
  endif()
  if(ELPA_ENABLE_CUDA_AWARE_MPI)
    list(APPEND _elpa_definitions WITH_CUDA_AWARE_MPI)
  endif()
  if(ELPA_ENABLE_GPU_STREAMS AND NOT ELPA_ACCELERATOR STREQUAL "NONE")
    list(APPEND _elpa_definitions WITH_GPU_STREAMS)
  endif()

  if(ELPA_FORTRAN_GLOBAL_SUFFIX STREQUAL "_")
    list(APPEND _elpa_definitions NEED_UNDERSCORE_TO_LINK_AGAINST_FORTRAN)
  elseif(ELPA_FORTRAN_GLOBAL_SUFFIX STREQUAL "")
    list(APPEND _elpa_definitions NEED_NO_UNDERSCORE_TO_LINK_AGAINST_FORTRAN)
  endif()

  set(${output_variable} "${_elpa_definitions}" PARENT_SCOPE)
endfunction()

# Generate the existing Autotools-style private configuration header in the
# build tree. It is neither installed nor added to the source tree.
function(elpa_write_config_f90_header output_file)
  file(WRITE "${output_file}"
    "/* Generated by CMake. Do not edit. */\n"
    "#ifndef ELPA_CMAKE_CONFIG_F90_H\n"
    "#define ELPA_CMAKE_CONFIG_F90_H\n"
  )

  foreach(_elpa_definition IN LISTS ARGN)
    string(FIND "${_elpa_definition}" "=" _elpa_equals)
    if(_elpa_equals EQUAL -1)
      file(APPEND "${output_file}" "#define ${_elpa_definition} 1\n")
    else()
      string(SUBSTRING "${_elpa_definition}" 0 ${_elpa_equals} _elpa_name)
      math(EXPR _elpa_value_start "${_elpa_equals} + 1")
      string(SUBSTRING "${_elpa_definition}" ${_elpa_value_start} -1 _elpa_value)
      file(APPEND "${output_file}" "#define ${_elpa_name} ${_elpa_value}\n")
    endif()
  endforeach()

  file(APPEND "${output_file}" "#endif\n")
endfunction()
