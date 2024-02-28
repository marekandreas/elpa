#    This file is part of ELPA (HH Test SYCL Proxy App).
#
#    The ELPA library was originally created by the ELPA consortium,
#    consisting of the following organizations:
#
#    - Max Planck Computing and Data Facility (MPCDF), formerly known as
#      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
#    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
#      Informatik,
#    - Technische Universität München, Lehrstuhl für Informatik mit
#      Schwerpunkt Wissenschaftliches Rechnen ,
#    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
#    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
#      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
#      and
#    - IBM Deutschland GmbH
#
#    This particular source code file contains additions, changes and
#    enhancements authored by Intel Corporation which is not part of
#    the ELPA consortium.
#
#    More information can be found here:
#    http://elpa.mpcdf.mpg.de/
#
#    ELPA is free software: you can redistribute it and/or modify
#    it under the terms of the version 3 of the license of the
#    GNU Lesser General Public License as published by the Free
#    Software Foundation.
#
#    ELPA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
#
#    ELPA reflects a substantial effort on the part of the original
#    ELPA consortium, and we ask you to respect the spirit of the
#    license that we chose: i.e., please contribute any changes you
#    may have back to the original ELPA library distribution, and keep
#    any derivatives of ELPA under the same license that we chose for
#    the original distribution, the GNU Lesser General Public License.
#
# This file was written by A. Poeppl, Intel Corporation (2024) for MPCDF

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

set(TARGET_GPU_AMD "gfx90a" CACHE STRING "Set the type of AMD GPU that will be targeted. (Both ROCm and SYCL")
set_property(CACHE TARGET_GPU_AMD PROPERTY STRINGS gfx900 gfx906 gfx908 gfx90a gfx942)

set(TARGET_GPU_NVIDIA "sm_80" CACHE STRING "Set the type of NVIDIA GPU that will be targeted. (SYCL only, CUDA uses all)")
set_property(CACHE TARGET_GPU_NVIDIA PROPERTY STRINGS sm_60 sm_70 sm_75 sm_80 sm_89 sm_90 sm_95)



# See if either (or both) CUDA or ROCm are avaiblable, and then add the native backend code components if applicable.
include(CheckLanguage)
check_language(CUDA)
check_language(HIP)

if (NOT CMAKE_CUDA_COMPILER STREQUAL "NOTFOUND")
  message(STATUS "NVIDIA CUDA found: ${CMAKE_CUDA_COMPILER}")
  enable_language(CUDA)
  set_target_properties(hh_test PROPERTIES CUDA_ARCHITECTURES all)
  target_sources(hh_test PUBLIC
    src/cuda/compute_hh_cuda.cu
    src/cuda/compute_hh_cuda_complex_wrapper.cu
    src/cuda/compute_hh_wrapper_cuda.f90
    src/cuda/cuda_c_interface.cu
    src/cuda/cuda_f_interface.f90
    src/cuda/config-f90.h
  )

  target_compile_definitions(hh_test PUBLIC ENABLE_REFERENCE_CUDA_IMPLEMENTATION)
  get_filename_component(CUDA_COMPILER_TMP ${CMAKE_CUDA_COMPILER} DIRECTORY)
  get_filename_component(CUDA_COMPILER_FOLDER ${CUDA_COMPILER_TMP}/.. ABSOLUTE)
endif()

if (CMAKE_HIP_COMPILER)
  message(STATUS "AMD ROCm found: ${CMAKE_HIP_COMPILER}")
  enable_language(HIP)
  set_target_properties(hh_test PROPERTIES HIP_ARCHITECTURES ${TARGET_GPU_AMD})
  target_sources(hh_test PUBLIC
    src/rocm/compute_hh_kernel_wrapper.hip
    src/rocm/rocm_c_interface.hip
    src/rocm/rocm_f_interface.f90
    src/rocm/compute_hh_wrapper_rocm.f90
  )
  target_compile_definitions(hh_test PUBLIC ENABLE_REFERENCE_ROCM_IMPLEMENTATION)
endif()

# Explicitly add the fortran main function
get_filename_component(compiler_folder ${CMAKE_Fortran_COMPILER} DIRECTORY)
message(STATUS "Fortran Compiler folder at ${compiler_folder}")
target_sources(hh_test PRIVATE ${compiler_folder}/../lib/for_main.o)

target_compile_options(hh_test PRIVATE
  $<$<COMPILE_LANGUAGE:Fortran>:-fpp -warn >
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)

set(TARGET_BACKENDS "")

if (CMAKE_CUDA_COMPILER)
  message(STATUS "Adding SYCL NVIDIA Backend.")
  list(APPEND TARGET_BACKENDS "nvptx64-nvidia-cuda")
  target_compile_options(hh_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>: --cuda-path=${CUDA_COMPILER_FOLDER} -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${TARGET_GPU_NVIDIA}>
  )
  target_link_libraries(hh_test PUBLIC --cuda-path=${CUDA_COMPILER_FOLDER} -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${TARGET_GPU_NVIDIA})
endif()

if (CMAKE_HIP_COMPILER)
  message(STATUS "Adding SYCL AMD Backend.")
  list(APPEND TARGET_BACKENDS "amdgcn-amd-amdhsa")
  target_compile_options(hh_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>: -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${TARGET_GPU_AMD}>
  )
  target_link_libraries(hh_test PUBLIC -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${TARGET_GPU_AMD})
endif()

message(STATUS "Adding SYCL Intel Backend")
list(APPEND TARGET_BACKENDS "spir64")

string(REPLACE ";" "," TARGET_BACKENDS "${TARGET_BACKENDS}")

target_compile_options(hh_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsycl -fsycl-targets=${TARGET_BACKENDS}>)
target_link_libraries (hh_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsycl -fsycl-targets=${TARGET_BACKENDS}>)

target_compile_definitions(hh_test PUBLIC ELPA_SYCL_IN_PROXY_APP)

set_target_properties(hh_test PROPERTIES CXX_STANDARD 20)
set_target_properties(hh_test PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(hh_test PROPERTIES Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include)

target_link_libraries(hh_test PRIVATE ${OpenMP_Fortran_LIBRARIES})
target_link_libraries(hh_test PUBLIC -lpthread -lm -ldl -lifcore)
