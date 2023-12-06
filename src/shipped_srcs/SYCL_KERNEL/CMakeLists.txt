cmake_minimum_required(VERSION 3.9)

project(hh_test VERSION 0.0.1 LANGUAGES Fortran CXX)

add_executable(hh_test
  src/compute_hh.cpp
  src/compute_hh_wrapper.f90
  src/sycl_c_interface.cpp
  src/sycl_f_interface.f90
  src/syclCommon.hpp
  src/syclCommon.cpp
  src/hh_functions.f90
  src/hh_test.f90)

find_package(OpenMP REQUIRED Fortran)


# Options
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release Debug RelWithDebInfo MinSizeRel)

set(SYCL_BACKEND SPIRV CACHE STRING "Select which GPUs to target - Intel (SPIRV) or NVIDIA (NVPTX)")
set_property(CACHE SYCL_BACKEND PROPERTY STRINGS SPIRV NVPTX)


# Explicitly add the fortran main function
get_filename_component(compiler_folder ${CMAKE_Fortran_COMPILER} DIRECTORY)
message(STATUS "Fortran Compiler folder at ${compiler_folder}")
target_sources(hh_test PRIVATE ${compiler_folder}/../lib/for_main.o)

if (SYCL_BACKEND STREQUAL "SPIRV")
  message(STATUS "Compiler options for SPIR-V")
  target_compile_options(hh_test PRIVATE
    $<$<COMPILE_LANGUAGE:Fortran>: >
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -fsycl>
  )
elseif(SYCL_BACKEND STREQUAL "NVPTX")
  message(STATUS "Compiler options for NVIDIA PTX")
  target_compile_options(hh_test PRIVATE
    $<$<COMPILE_LANGUAGE:Fortran>: >
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -fsycl -fsycl-targets=nvptx64-nvidia-cuda>
    $<$<COMPILE_LANGUAGE:CXX>:--cuda-path=/opt/hpc_software/compilers/nvidia/cuda-12.0 -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80>
  )
else()
  message(ERROR "Unsupported Backend: ${SYCL_BACKEND}")
endif()

target_compile_definitions(hh_test PUBLIC ELPA_SYCL_IN_PROXY_APP)

set_target_properties(hh_test PROPERTIES CXX_STANDARD 20)
set_target_properties(hh_test PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(hh_test PROPERTIES Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include)

target_link_libraries(hh_test PRIVATE ${OpenMP_Fortran_LIBRARIES})

if (SYCL_BACKEND STREQUAL "SPIRV")
  message(STATUS "Linker options for SPIR-V")
  target_link_libraries(hh_test PUBLIC -fsycl -lpthread -lm -ldl -lifcore)
elseif(SYCL_BACKEND STREQUAL "NVPTX")
  message(STATUS "Linker options for NVIDIA PTX")
  target_link_libraries(hh_test PUBLIC --cuda-path=/opt/hpc_software/compilers/nvidia/cuda-12.0 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -lpthread -lm -ldl -lifcore)
else()
  message(ERROR "Unsupported Backend: ${SYCL_BACKEND}")
endif()
