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

set(GPU_BACKEND INTEL CACHE STRING "Select which GPU to target - Intel (INTEL), or NVIDIA (SYCL & CUDA)")
set_property(CACHE GPU_BACKEND PROPERTY STRINGS INTEL NVIDIA)

if (GPU_BACKEND STREQUAL "NVIDIA")
  enable_language(CUDA)
  set_target_properties(hh_test PROPERTIES CUDA_ARCHITECTURES "all-major")
  target_sources(hh_test PRIVATE
    src/cuda/compute_hh_cuda.cu
    src/cuda/compute_hh_wrapper_cuda.f90
    src/cuda/cuda_c_interface.cu
    src/cuda/cuda_f_interface.f90
  )
endif()

# Explicitly add the fortran main function
get_filename_component(compiler_folder ${CMAKE_Fortran_COMPILER} DIRECTORY)
message(STATUS "Fortran Compiler folder at ${compiler_folder}")
target_sources(hh_test PRIVATE ${compiler_folder}/../lib/for_main.o)

target_compile_options(hh_test PRIVATE
  $<$<COMPILE_LANGUAGE:Fortran>:-fpp -warn >
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)


if (GPU_BACKEND STREQUAL "INTEL")
  message(STATUS "Compiler options for SPIR-V")
  target_compile_options(hh_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fsycl>
  )
elseif(GPU_BACKEND STREQUAL "NVIDIA")
  message(STATUS "Compiler options for NVIDIA PTX")
  target_compile_definitions(hh_test PRIVATE ENABLE_REFERENCE_CUDA_IMPLEMENTATION)
  target_compile_options(hh_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fsycl -fsycl-targets=nvptx64-nvidia-cuda>
    $<$<COMPILE_LANGUAGE:CXX>:--cuda-path=/opt/hpc_software/compilers/nvidia/cuda-12.0 -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80>
  )
else()
  message(SEND_ERROR "Unsupported Backend: ${GPU_BACKEND}")
endif()

target_compile_definitions(hh_test PUBLIC ELPA_SYCL_IN_PROXY_APP)

set_target_properties(hh_test PROPERTIES CXX_STANDARD 20)
set_target_properties(hh_test PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(hh_test PROPERTIES Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include)

target_link_libraries(hh_test PRIVATE ${OpenMP_Fortran_LIBRARIES})

if (GPU_BACKEND STREQUAL "INTEL")
  message(STATUS "Linker options for SPIR-V")
  target_link_libraries(hh_test PUBLIC -fsycl -lpthread -lm -ldl -lifcore)
elseif(GPU_BACKEND STREQUAL "NVIDIA")
  message(STATUS "Linker options for NVIDIA PTX")
  target_link_libraries(hh_test PUBLIC --cuda-path=/opt/hpc_software/compilers/nvidia/cuda-12.0 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -lpthread -lm -ldl -lifcore)
else()
  message(SEND_ERROR "Unsupported Backend: ${GPU_BACKEND}")
endif()
