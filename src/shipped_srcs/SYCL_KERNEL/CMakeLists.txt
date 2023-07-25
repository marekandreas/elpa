cmake_minimum_required(VERSION 3.8)

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

target_compile_options(hh_test PUBLIC -fsycl -fiopenmp)
target_compile_definitions(hh_test PUBLIC ELPA_SYCL_IN_PROXY_APP)

set_target_properties(hh_test PROPERTIES CXX_STANDARD 20)
set_target_properties(hh_test PROPERTIES LINKER_LANGUAGE Fortran)
set_target_properties(hh_test PROPERTIES Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include)

target_link_libraries(hh_test PUBLIC -fiopenmp -fsycl -lsycl -lOpenCL -lpthread -lm -ldl)
target_link_libraries(hh_test ${OpenMP_Fortran_LIBRARIES})

