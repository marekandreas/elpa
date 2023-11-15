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


# Explicitly add the fortran main function
get_filename_component(compiler_folder ${CMAKE_Fortran_COMPILER} DIRECTORY)
message(STATUS "Fortran Compiler folder at ${compiler_folder}")
target_sources(hh_test PRIVATE ${compiler_folder}/../compiler/lib/intel64_lin/for_main.o)

target_compile_options(hh_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:Fortran,IntelLLVM>: >
  $<$<COMPILE_LANG_AND_ID:CXX,IntelLLVM>:-Wall -fsycl>
  $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fsycl -fsycl-targets=nvptx64-nvidia-cuda>
  $<$<COMPILE_LANG_AND_ID:CXX,Clang>:--cuda-path=/opt/hpc_software/compilers/nvidia/cuda-12.0 -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80>
)
target_compile_definitions(hh_test PUBLIC ELPA_SYCL_IN_PROXY_APP)

set_target_properties(hh_test PROPERTIES CXX_STANDARD 20)
set_target_properties(hh_test PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(hh_test PROPERTIES Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include)

target_link_libraries(hh_test PRIVATE ${OpenMP_Fortran_LIBRARIES})
target_link_libraries(hh_test PUBLIC --cuda-path=/opt/hpc_software/compilers/nvidia/cuda-12.0 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -lpthread -lm -ldl -lifcore)

