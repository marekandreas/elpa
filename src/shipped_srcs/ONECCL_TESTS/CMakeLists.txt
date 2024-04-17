#CMakeLists file that includes MPI and oneCCL
cmake_minimum_required(VERSION 3.10)

project(ccltest LANGUAGES CXX)

find_package(MPI REQUIRED)
find_package(oneCCL REQUIRED)

# Add the executable
add_executable(ccltest 
  ccl-test2.cpp 
  syclCommon.cpp
  syclCommon.hpp
  elpa-oneccl-wrapper.h
)

# Link MPI and oneCCL
target_include_directories(ccltest PUBLIC ${oneCCL_DIR}/include)
target_include_directories(ccltest PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_compile_options(ccltest     PUBLIC ${MPI_CXX_COMPILE_OPTIONS})
target_compile_options(ccltest     PUBLIC -fsycl -Wall -Wextra)
target_link_libraries(ccltest      PUBLIC -fsycl)
target_link_libraries(ccltest      PUBLIC MPI::MPI_CXX)

target_link_libraries(ccltest      PRIVATE oneCCL)

# Make sure the oneCCL wrapper is used
target_compile_definitions(ccltest PRIVATE WITH_INTEL_ONECCL)

# fix broken mpi link in oneccl
set_target_properties(oneCCL PROPERTIES INTERFACE_LINK_LIBRARIES "")

# Set the C++ standard
set_property(TARGET ccltest PROPERTY CXX_STANDARD 20)
set_property(TARGET ccltest PROPERTY CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
