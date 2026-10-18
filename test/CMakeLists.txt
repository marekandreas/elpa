# SPDX-License-Identifier: LGPL-3.0-only

set(ELPA_TEST_MPI_PROCS
    2
    CACHE STRING "MPI processes per test")
set(ELPA_TEST_OMP_THREADS
    1
    CACHE STRING "OpenMP threads per test process")

foreach(_elpa_setting IN ITEMS ELPA_TEST_MPI_PROCS ELPA_TEST_OMP_THREADS)
  if(NOT "${${_elpa_setting}}" MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "${_elpa_setting} must be a positive integer")
  endif()
endforeach()

if(ELPA_ENABLE_CUDA_AWARE_MPI)
  message(FATAL_ERROR "The current CMake test suite requires CPU solver paths; "
                      "disable ELPA_ENABLE_CUDA_AWARE_MPI to run it.")
endif()

if(ELPA_USE_MPI)
  set(MPI_CXX_SKIP_MPICXX TRUE)
  find_package(MPI REQUIRED COMPONENTS CXX)
  if(NOT MPIEXEC_EXECUTABLE)
    message(FATAL_ERROR "Set MPIEXEC_EXECUTABLE to run the MPI tests")
  endif()
endif()

set(_elpa_test_module_dir "${CMAKE_CURRENT_BINARY_DIR}/modules")
file(MAKE_DIRECTORY "${_elpa_test_module_dir}"
     "${CMAKE_CURRENT_BINARY_DIR}/shared")

set(_elpa_test_interface_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/shared/test_prepare_matrix_template.F90"
    "${CMAKE_CURRENT_SOURCE_DIR}/shared/test_check_correctness_template.F90"
    "${CMAKE_CURRENT_SOURCE_DIR}/shared/test_analytic_template.F90"
    "${CMAKE_CURRENT_SOURCE_DIR}/shared/test_blacs_infrastructure.F90")
_elpa_extract_marked_lines("${CMAKE_CURRENT_BINARY_DIR}/shared/generated.h"
                           "!c>" "${_elpa_test_interface_sources}")

add_library(
  elpa_test_support STATIC
  shared/tests_variable_definitions.F90
  shared/mod_tests_scalapack_interfaces.F90
  shared/mod_tests_blas_interfaces.F90
  shared/test_util.F90
  shared/test_read_input_parameters.F90
  shared/test_check_correctness.F90
  shared/test_setup_mpi.F90
  shared/test_blacs_infrastructure.F90
  shared/test_prepare_matrix.F90
  shared/test_analytic.F90
  shared/test_output_type.F90)
set_target_properties(
  elpa_test_support PROPERTIES Fortran_MODULE_DIRECTORY
                               "${_elpa_test_module_dir}" Fortran_PREPROCESS ON)
target_include_directories(elpa_test_support PUBLIC "${_elpa_test_module_dir}")
target_link_libraries(elpa_test_support PUBLIC ELPA::elpa ${ELPA_BLAS_TARGET})
elpa_configure_compiler(elpa_test_support)

if(ELPA_USE_MPI)
  target_link_libraries(elpa_test_support PUBLIC MPI::MPI_C MPI::MPI_Fortran
                                                 ${ELPA_SCALAPACK_TARGET})
endif()
if(ELPA_USE_OPENMP)
  target_link_libraries(
    elpa_test_support PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX
                             OpenMP::OpenMP_Fortran)
endif()

function(elpa_add_test_program target source)
  add_executable(${target} "${source}")
  target_link_libraries(${target} PRIVATE elpa_test_support)
  target_compile_definitions(${target} PRIVATE ${ARGN})
  set_target_properties(${target} PROPERTIES Fortran_PREPROCESS ON)
  elpa_configure_compiler(${target})
  # Keep API error checks active in optimized test builds.
  if(MSVC)
    target_compile_options(${target}
                           PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:/UNDEBUG>)
  else()
    target_compile_options(${target}
                           PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-UNDEBUG>)
  endif()
endfunction()

function(elpa_register_test target)
  set(_elpa_command "$<TARGET_FILE:${target}>")
  set(_elpa_processes 1)
  set(_elpa_threads 1)
  set(_elpa_parallel_label serial)
  if(ELPA_USE_MPI)
    set(_elpa_processes "${ELPA_TEST_MPI_PROCS}")
    set(_elpa_parallel_label mpi)
    set(_elpa_command
        "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} ${_elpa_processes}
        ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:${target}>" ${MPIEXEC_POSTFLAGS})
  endif()
  if(ELPA_USE_OPENMP)
    set(_elpa_threads "${ELPA_TEST_OMP_THREADS}")
  endif()
  math(EXPR _elpa_slots "${_elpa_processes} * ${_elpa_threads}")
  set(_elpa_work_dir "${CMAKE_CURRENT_BINARY_DIR}/runs/${target}")
  file(MAKE_DIRECTORY "${_elpa_work_dir}")
  add_test(NAME ${target} COMMAND ${_elpa_command} 256 128 16)
  set_tests_properties(
    ${target}
    PROPERTIES
      WORKING_DIRECTORY
      "${_elpa_work_dir}"
      PROCESSORS
      "${_elpa_slots}"
      TIMEOUT
      180
      SKIP_RETURN_CODE
      77
      LABELS
      "cpu;default;eigenvectors;${_elpa_parallel_label};${ARGN}"
      ENVIRONMENT
      "OMP_NUM_THREADS=${_elpa_threads};OMP_DYNAMIC=FALSE;MKL_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
  )
endfunction()

set(_elpa_test_precisions double)
if(ELPA_ENABLE_SINGLE_PRECISION)
  list(APPEND _elpa_test_precisions single)
endif()

foreach(_elpa_language IN ITEMS Fortran C CXX)
  if(_elpa_language STREQUAL "Fortran")
    set(_elpa_source Fortran/test.F90)
  elseif(_elpa_language STREQUAL "C")
    set(_elpa_source C/test.c)
  else()
    set(_elpa_source C++/test.cpp)
  endif()
  string(TOLOWER "${_elpa_language}" _elpa_language_label)
  foreach(_elpa_domain IN ITEMS real complex)
    string(TOUPPER "${_elpa_domain}" _elpa_domain_macro)
    foreach(_elpa_precision IN LISTS _elpa_test_precisions)
      string(TOUPPER "${_elpa_precision}" _elpa_precision_macro)
      foreach(_elpa_solver IN ITEMS 1stage 2stage)
        string(TOUPPER "${_elpa_solver}" _elpa_solver_macro)
        set(_elpa_target
            "elpa_${_elpa_language_label}_${_elpa_domain}_${_elpa_precision}_eigenvectors_${_elpa_solver}"
        )
        elpa_add_test_program(
          ${_elpa_target}
          "${_elpa_source}"
          TEST_${_elpa_domain_macro}
          TEST_${_elpa_precision_macro}
          TEST_SOLVER_${_elpa_solver_macro}
          TEST_EIGENVECTORS
          TEST_MATRIX_RANDOM
          TEST_CASE="${_elpa_target}"
          TEST_NVIDIA_GPU=0
          TEST_INTEL_GPU=0
          TEST_AMD_GPU=0
          TEST_OPENMP_OFFLOAD_GPU=0
          TEST_INTEL_GPU_OPENMP=0
          TEST_INTEL_GPU_SYCL=0
          TEST_GPU_SET_ID=0
          TEST_GPU_DEVICE_POINTER_API=0
          TEST_QR_DECOMPOSITION=0)
        if(_elpa_solver STREQUAL "2stage")
          target_compile_definitions(
            ${_elpa_target}
            PRIVATE TEST_KERNEL=ELPA_2STAGE_${_elpa_domain_macro}_DEFAULT)
        endif()
        if(ELPA_USE_MPI AND _elpa_language STREQUAL "CXX")
          target_link_libraries(${_elpa_target} PRIVATE MPI::MPI_CXX)
        endif()
        elpa_register_test(
          ${_elpa_target} "${_elpa_language_label}" "${_elpa_domain}"
          "${_elpa_precision}" "${_elpa_solver}")
      endforeach()
    endforeach()
  endforeach()
endforeach()
