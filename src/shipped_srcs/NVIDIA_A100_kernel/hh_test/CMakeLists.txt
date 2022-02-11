cmake_minimum_required(VERSION 3.8)

project(hh_test VERSION 0.0.1 LANGUAGES Fortran CUDA)

add_executable(hh_test
  src/compute_hh.cu
  src/compute_hh_wrapper.f90
  src/cuda_c_interface.cu
  src/cuda_f_interface.f90
  src/hh_test.f90)

set_target_properties(hh_test PROPERTIES LINKER_LANGUAGE Fortran)
set_target_properties(hh_test PROPERTIES Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include)
