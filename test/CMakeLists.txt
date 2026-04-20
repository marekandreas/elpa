set(ELPA_LIB_TARGET "elpa${ELPA_SUFFIX}")

if(WITH_MPI)
    set(ELPA_TEST_NPROCS "2" CACHE STRING "Number of MPI processes for tests")
    if(NOT MPIEXEC_EXECUTABLE)
        find_program(MPIEXEC_EXECUTABLE NAMES mpiexec mpirun srun)
    endif()
    if(NOT MPIEXEC_NUMPROC_FLAG)
        set(MPIEXEC_NUMPROC_FLAG "-np")
    endif()
else()
    set(ELPA_TEST_NPROCS "1")
endif()

set(_test_lib_sources "")
foreach(_src IN LISTS ELPA_TEST_SOURCES)
    list(APPEND _test_lib_sources "${CMAKE_SOURCE_DIR}/${_src}")
endforeach()
if(WITH_NVIDIA_GPU_VERSION)
    foreach(_src IN LISTS ELPA_TEST_CUDA_SOURCES)
        list(APPEND _test_lib_sources "${CMAKE_SOURCE_DIR}/${_src}")
    endforeach()
endif()
if(WITH_MPI)
    foreach(_src IN LISTS ELPA_TEST_SCALAPACK_SOURCES)
        list(APPEND _test_lib_sources "${CMAKE_SOURCE_DIR}/${_src}")
    endforeach()
endif()

include(ElpaFortranPreprocess)
include(ElpaTargetOptions)
include(ElpaTestSupport)

set(_test_pp_inc_dirs
    "${CMAKE_BINARY_DIR}"
    "${CMAKE_BINARY_DIR}/src"
    "${CMAKE_BINARY_DIR}/elpa"
    "${CMAKE_BINARY_DIR}/test/shared"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/elpa"
    "${CMAKE_SOURCE_DIR}/test/shared"
)
if(MKL_INCLUDE)
    list(APPEND _test_pp_inc_dirs ${MKL_INCLUDE})
endif()

set(_test_pp_defines "HAVE_CONFIG_H")
if(WIN32)
    list(APPEND _test_pp_defines "_WIN32")
endif()

elpa_preprocess_fortran(_pp_test_lib_sources
    SOURCES ${_test_lib_sources}
    INCLUDE_DIRS ${_test_pp_inc_dirs}
    DEFINES ${_test_pp_defines}
)

add_library(elpatest STATIC ${_pp_test_lib_sources})
add_dependencies(elpatest elpa_generated_headers)
set_target_properties(
    elpatest
    PROPERTIES
        Fortran_PREPROCESS OFF
        Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/test_modules"
)

set(_test_common_includes
    "${CMAKE_BINARY_DIR}"
    "${CMAKE_BINARY_DIR}/src"
    "${CMAKE_BINARY_DIR}/elpa"
    "${CMAKE_BINARY_DIR}/test/shared"
    "${CMAKE_BINARY_DIR}/modules"
    "${CMAKE_BINARY_DIR}/test_modules"
    "${CMAKE_CURRENT_SOURCE_DIR}/shared"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../elpa"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

target_include_directories(elpatest PRIVATE ${_test_common_includes})
target_link_libraries(elpatest PUBLIC ${ELPA_LIB_TARGET})
if(WITH_NVIDIA_GPU_VERSION)
    target_link_libraries(elpatest PRIVATE CUDA::cublas CUDA::cudart)
    if(HAVE_CUBLASLT AND TARGET CUDA::cublasLt)
        target_link_libraries(elpatest PRIVATE CUDA::cublasLt)
    endif()
    if(WITH_NVIDIA_CUSOLVER AND TARGET CUDA::cusolver)
        target_link_libraries(elpatest PRIVATE CUDA::cusolver)
    endif()
endif()
elpa_apply_fortran_compile_options(elpatest)
elpa_apply_cuda_target_options(elpatest)
elpa_apply_flang_diagnostic_suppressions()

set(_config_vars_file "${CMAKE_BINARY_DIR}/test_config_vars.txt")
elpa_write_test_config_vars("${_config_vars_file}")

set(ELPA_TEST_MATRIX_SIZE "200" CACHE STRING "Matrix size for ELPA tests")
set(ELPA_TEST_NEV "30" CACHE STRING "Number of eigenvectors for ELPA tests")
set(ELPA_TEST_NBLK "16" CACHE STRING "Block size for ELPA tests")

elpa_prepare_fortran_test_source(_f_src "${CMAKE_CURRENT_SOURCE_DIR}/Fortran/test.F90")
set(_c_src "${CMAKE_CURRENT_SOURCE_DIR}/C/test.c")
set(_cpp_src "${CMAKE_CURRENT_SOURCE_DIR}/C++/test.cpp")

foreach(
    _variant
    autotune
    invert_triangular
    multiple_objs
    skewsymmetric
    split_comm
)
    set(_orig "${CMAKE_CURRENT_SOURCE_DIR}/Fortran/test_${_variant}.F90")
    if(EXISTS "${_orig}")
        elpa_prepare_fortran_test_source(_f_${_variant}_src "${_orig}")
    endif()
endforeach()
foreach(_variant autotune multiple_objs skewsymmetric)
    set(_orig "${CMAKE_CURRENT_SOURCE_DIR}/C/test_${_variant}.c")
    if(EXISTS "${_orig}")
        set(_c_${_variant}_src "${_orig}")
    endif()
endforeach()
foreach(_variant autotune multiple_objs skewsymmetric)
    set(_orig "${CMAKE_CURRENT_SOURCE_DIR}/C++/test_${_variant}.cpp")
    if(EXISTS "${_orig}")
        set(_cpp_${_variant}_src "${_orig}")
    endif()
endforeach()

set(_elpa_tests_cmake "${CMAKE_BINARY_DIR}/ElpaTests.cmake")
elpa_generate_tests_include("${_config_vars_file}" "${_elpa_tests_cmake}")
include("${_elpa_tests_cmake}")

# Hand-written tests from Makefile.am that are not part of test_programs.am.
set(_elpa2_test_dir "${CMAKE_CURRENT_SOURCE_DIR}/Fortran/elpa2")
set(_handwritten_tests
    "validate_double_instance:double_instance.F90"
    "validate_real_2stage_banded:real_2stage_banded.F90"
    "validate_complex_2stage_banded:complex_2stage_banded.F90"
    "validate_single_real_2stage_banded:single_real_2stage_banded.F90"
    "validate_single_complex_2stage_banded:single_complex_2stage_banded.F90"
)
foreach(_entry IN LISTS _handwritten_tests)
    string(REPLACE ":" ";" _parts "${_entry}")
    list(GET _parts 0 _name)
    list(GET _parts 1 _src_file)

    set(_full_name "${_name}${ELPA_SUFFIX}")
    set(_src "${_elpa2_test_dir}/${_src_file}")
    if(EXISTS "${_src}")
        elpa_prepare_fortran_test_source(_prepared_src "${_src}")
        elpa_add_test(${_full_name}_default ${_prepared_src} Fortran)
    endif()
endforeach()

message(STATUS "ELPA: Test infrastructure configured")
