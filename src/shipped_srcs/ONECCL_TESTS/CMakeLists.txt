cmake_minimum_required(VERSION 3.10)

project(ccltest LANGUAGES CXX Fortran CUDA)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED Fortran)
find_package(oneCCL REQUIRED)
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})

# Add the executable
add_executable(ccltest
  oneccl_test.F90
  oneccl_test_main.F90

  # Stuff with Symlinks into test driver.
  GPU/mod_vendor_agnostic_utilities_layer.F90
  GPU/mod_vendor_agnostic_blas_layer.F90
  GPU/mod_vendor_agnostic_ccl_layer.F90
  GPU/mod_vendor_agnostic_general_layer.F90
  GPU/mod_vendor_agnostic_solver_layer.F90
  GPU/CUDA/cudaFunctions.cu
  GPU/CUDA/cuUtils.cu
  GPU/CUDA/mod_cuda.F90
  GPU/CUDA/mod_cusolver.F90
  GPU/SYCL/mod_sycl.F90
  GPU/SYCL/mod_syclsolver.F90
  GPU/SYCL/mod_oneccl.F90
  GPU/SYCL/onecclFunctions.cpp
  GPU/SYCL/syclFunctions.cpp
  GPU/SYCL/syclCommon.cpp
  GPU/SYCL/syclCommon.hpp
  GPU/mod_vendor_agnostic_layer.F90
  GPU/mod_gpu_setup.F90
  helpers/mod_precision.F90
  general/elpa_utilities.F90
)

# Explicitly add the fortran main function
get_filename_component(compiler_folder ${CMAKE_Fortran_COMPILER} DIRECTORY)
message(STATUS "Fortran Compiler folder at ${compiler_folder}")
target_sources(ccltest PRIVATE ${compiler_folder}/../lib/for_main.o)

# Compatibility with Autotools
target_include_directories(ccltest PUBLIC .)

message(STATUS "${MKL_IMPORTED_TARGETS}")

# Link MPI and oneCCL
target_include_directories(ccltest PUBLIC ${oneCCL_DIR}/include)
target_include_directories(ccltest PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${MPI_CXX_INCLUDE_DIRS}>)
target_include_directories(ccltest PUBLIC $<$<COMPILE_LANGUAGE:Fortran>:${MPI_Fortran_INCLUDE_DIRS}>)
target_include_directories(ccltest PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_options(ccltest     PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Wall -Wextra>
                                          $<$<COMPILE_LANGUAGE:CXX>:-g -fsycl -Wall -Wextra>
                                          $<$<COMPILE_LANGUAGE:CXX>:$<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>>
                                          $<$<COMPILE_LANGUAGE:Fortran>:-g -fpp>)

target_link_libraries(ccltest      PUBLIC -fsycl)
target_link_libraries(ccltest      PUBLIC MPI::MPI_CXX)
target_link_libraries(ccltest      PUBLIC MPI::MPI_Fortran)
target_link_libraries(ccltest      PUBLIC OpenMP::OpenMP_Fortran)
target_link_libraries(ccltest      PUBLIC -lpthread -lm -ldl -lifcore)
target_link_libraries(ccltest      PUBLIC oneCCL)
target_link_libraries(ccltest      PUBLIC $<LINK_ONLY:MKL::MKL>)
target_link_libraries(ccltest      PUBLIC $<LINK_ONLY:MKL::MKL_SYCL>)

# Make sure the oneCCL wrapper is used
target_compile_definitions(ccltest PRIVATE WITH_ONEAPI_ONECCL)
target_compile_definitions(ccltest PRIVATE WITH_SYCL_GPU_VERSION)
target_compile_definitions(ccltest PRIVATE WITH_GPU_STREAMS)

# fix broken mpi link in oneccl
set_target_properties(oneCCL PROPERTIES INTERFACE_LINK_LIBRARIES "")

# Set the C++ standard
set_property(TARGET ccltest PROPERTY CXX_STANDARD 20)
set_property(TARGET ccltest PROPERTY CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
