cmake_minimum_required(VERSION 3.23)

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

option(ENABLE_AMD_SYCL_BACKEND "Should the AMD SYCL backend be targeted? If so, make sure ROCM is in the environment" OFF)

include(CheckLanguage)
check_language(CUDA)

if (NOT CMAKE_CUDA_COMPILER STREQUAL "NOTFOUND")
  message(STATUS "NVIDIA CUDA found: ${CMAKE_CUDA_COMPILER}")
  enable_language(CUDA)
  set_target_properties(hh_test PROPERTIES CUDA_ARCHITECTURES "all-major")
  target_sources(hh_test PRIVATE
    src/cuda/compute_hh_cuda.cu
    src/cuda/compute_hh_cuda_complex_wrapper.cu
    src/cuda/compute_hh_wrapper_cuda.f90
    src/cuda/cuda_c_interface.cu
    src/cuda/cuda_f_interface.f90
    src/cuda/config-f90.h
  )

  get_filename_component(CUDA_COMPILER_TMP ${CMAKE_CUDA_COMPILER} DIRECTORY)
  get_filename_component(CUDA_COMPILER_FOLDER ${CUDA_COMPILER_TMP}/.. ABSOLUTE)
endif()

# Explicitly add the fortran main function
get_filename_component(compiler_folder ${CMAKE_Fortran_COMPILER} DIRECTORY)
message(STATUS "Fortran Compiler folder at ${compiler_folder}")
target_sources(hh_test PRIVATE ${compiler_folder}/../lib/for_main.o)

target_compile_options(hh_test PRIVATE
  $<$<COMPILE_LANGUAGE:Fortran>:-fpp -warn >
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)


if (NOT CMAKE_CUDA_COMPILER STREQUAL "NOTFOUND")
  target_compile_definitions(hh_test PRIVATE ENABLE_REFERENCE_CUDA_IMPLEMENTATION)
  if (ENABLE_AMD_SYCL_BACKEND)
    message(STATUS "Configuring a fat binary with SYCL (NVIDIA, AMD and Intel) and CUDA")
    message(STATUS "Make sure ROCm is actually installed! The Configure script doesn't check this")
    target_compile_options(hh_test PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fsycl -fsycl-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda,spir64>
      $<$<COMPILE_LANGUAGE:CXX>: -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a>
    )
  else()
    message(STATUS "Configuring a fat binary with SYCL (NVIDIA and Intel) and CUDA")
    target_compile_options(hh_test PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64>
    )
  endif()
  target_compile_options(hh_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>: --cuda-path=${CUDA_COMPILER_FOLDER} -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80>
  )
else()
  message(STATUS "No CUDA found.")
  message(STATUS "Configuring a binary with SYCL (Intel) only")
  target_compile_options(hh_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fsycl>
  )
endif()

target_compile_definitions(hh_test PUBLIC ELPA_SYCL_IN_PROXY_APP)

set_target_properties(hh_test PROPERTIES CXX_STANDARD 20)
set_target_properties(hh_test PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(hh_test PROPERTIES Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include)

target_link_libraries(hh_test PRIVATE ${OpenMP_Fortran_LIBRARIES})

if (NOT CMAKE_CUDA_COMPILER STREQUAL "NOTFOUND")
  if (ENABLE_AMD_SYCL_BACKEND)
    target_link_libraries(hh_test PUBLIC -fsycl -fsycl-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda,spir64)
    target_link_libraries(hh_test PUBLIC -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a)
  else()
    target_link_libraries(hh_test PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64)
  endif()
  target_link_libraries(hh_test PUBLIC --cuda-path=${CUDA_COMPILER_FOLDER} -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80)
  target_link_libraries(hh_test PUBLIC -lpthread -lm -ldl -lifcore)
else()
  target_link_libraries(hh_test PUBLIC -fsycl -lpthread -lm -ldl -lifcore)
endif()
