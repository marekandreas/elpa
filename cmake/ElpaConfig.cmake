# ElpaConfig.cmake — Feature detection for ELPA
#
# Keeps the public entry point stable while delegating dependency discovery,
# compiler setup, and feature checks to focused helper modules.

set(_elpa_dependency_cmake_dir "${CMAKE_CURRENT_LIST_DIR}/dependencies")

include("${_elpa_dependency_cmake_dir}/ElpaMPI.cmake")
include("${_elpa_dependency_cmake_dir}/ElpaMathLibraries.cmake")
include("${_elpa_dependency_cmake_dir}/ElpaOpenMP.cmake")

if(ELPA_MPI)
    if(ELPA_OPENMP)
        set(ELPA_SUFFIX "_openmp")
    else()
        set(ELPA_SUFFIX "")
    endif()
else()
    if(ELPA_OPENMP)
        set(ELPA_SUFFIX "_onenode_openmp")
    else()
        set(ELPA_SUFFIX "_onenode")
    endif()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/ElpaCompilerOptions.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ElpaFeatureChecks.cmake")

set(CURRENT_API_VERSION 20260202)
set(EARLIEST_API_VERSION 20170403)
set(CURRENT_AUTOTUNE_VERSION 202600202)
set(EARLIEST_AUTOTUNE_VERSION 20171201)
string(TIMESTAMP ELPA_BUILDTIME "%s" UTC)

if(NOT DEFINED CURRENT_WITH_NVIDIA_GPU_VERSION)
    set(CURRENT_WITH_NVIDIA_GPU_VERSION 0)
endif()
if(NOT DEFINED CURRENT_WITH_AMD_GPU_VERSION)
    set(CURRENT_WITH_AMD_GPU_VERSION 0)
endif()
if(NOT DEFINED CURRENT_WITH_SYCL_GPU_VERSION)
    set(CURRENT_WITH_SYCL_GPU_VERSION 0)
endif()

set(PACKAGE "elpa")
set(PACKAGE_NAME "elpa")
set(PACKAGE_VERSION "${PROJECT_VERSION}")
set(PACKAGE_STRING "elpa ${PROJECT_VERSION}")
set(PACKAGE_TARNAME "elpa")
set(PACKAGE_BUGREPORT "elpa-library@mpcdf.mpg.de")
set(PACKAGE_URL "")

message(
    STATUS
    "ELPA: suffix='${ELPA_SUFFIX}' MPI=${ELPA_MPI} OpenMP=${ELPA_OPENMP} BLAS=${ELPA_BLAS_VENDOR} MKL=${WITH_MKL}"
)
