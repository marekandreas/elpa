# SPDX-License-Identifier: LGPL-3.0-only

cmake_minimum_required(VERSION 3.20)

project(elpa_test_project_1stage LANGUAGES C Fortran)

list(PREPEND CMAKE_MODULE_PATH
     "${CMAKE_CURRENT_LIST_DIR}/../cmake/modules")

find_package(MPI REQUIRED COMPONENTS Fortran)
find_package(elpa CONFIG REQUIRED)

find_package(Lapack REQUIRED)
if(MKL_ROOT)
  find_package(MKL REQUIRED COMPONENTS SCALAPACK)
  set(SCALAPACK_TARGET MKL::ScaLAPACK)
else()
  find_package(Scalapack REQUIRED)
  set(SCALAPACK_TARGET SCALAPACK::SCALAPACK)
endif()

add_executable(test_real src/test_real.F90)
target_link_libraries(
  test_real
  PRIVATE
    ELPA::elpa
    ${SCALAPACK_TARGET}
)
