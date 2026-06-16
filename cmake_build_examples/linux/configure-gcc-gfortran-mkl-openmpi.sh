#!/usr/bin/env bash
# configure-gcc-gfortran-mkl-openmpi.sh
#
# Configure ELPA: gcc + gfortran + OpenMPI + Intel MKL + CUDA
#
# Uses MKL for BLAS/LAPACK/ScaLAPACK with the OpenMPI BLACS variant.
# GCC and gfortran still compile OpenMP code with -fopenmp, but ELPA's
# CMake OpenMP overrides strip that flag from link commands and link against
# Intel's libiomp5 directly. This keeps ELPA and MKL on the same runtime.
#
# Prerequisites
#   gcc >= 10, cmake >= 3.24, ninja >= 1.10
#   Intel oneAPI Base Toolkit (MKL)
#   Intel oneAPI compiler runtime (libiomp5)
#   OpenMPI — system packages or custom build
#   NVIDIA CUDA Toolkit >= 12.x from developer.nvidia.com
#
# Usage
#   # Override any PATHS variable via the environment if needed, then:
#   ./configure-gcc-gfortran-mkl-openmpi.sh
#   cmake --build <BLD> -j8
#   cd <BLD> && /path/to/cmake_build_examples/linux/test.sh --all -j8

set -euo pipefail

# ===========================================================================
# PATHS — override any of these via the environment
# ===========================================================================
SRC="${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
SCRATCH_DIR="${SCRATCH_DIR:-${SRC}/.scratch}"
BLD="${BLD:-${SCRATCH_DIR}/build-gcc-mkl-openmpi}"

# Custom GCC installation (leave empty to use system gcc)
GCC_HOME="${GCC_HOME:-}"

# Custom OpenMPI installation (leave empty to use system mpicc)
OMPI_HOME="${OMPI_HOME:-}"

# Intel oneAPI root (used to locate MKLConfig.cmake)
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
COMPILER_ROOT="${ONEAPI_ROOT}/compiler/latest"
COMPILER_VARS_SH="${COMPILER_ROOT}/env/vars.sh"
IOMP5_LIBRARY="${COMPILER_ROOT}/lib/libiomp5.so"

# CUDA Toolkit root
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"

# ===========================================================================
# ENVIRONMENT
# ===========================================================================
if [[ ! -f "${COMPILER_VARS_SH}" ]]; then
    echo "Missing oneAPI compiler environment script: ${COMPILER_VARS_SH}" >&2
    exit 1
fi

if [[ ! -f "${IOMP5_LIBRARY}" ]]; then
    echo "Missing Intel OpenMP runtime: ${IOMP5_LIBRARY}" >&2
    exit 1
fi

set +u
source "${COMPILER_VARS_SH}" >/dev/null 2>&1
set -u

unset CPATH
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH
unset INCLUDE
unset OBJC_INCLUDE_PATH

if [[ -n "${GCC_HOME}" ]]; then
    export PATH="${GCC_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${GCC_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ -n "${OMPI_HOME}" ]]; then
    export PATH="${OMPI_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${OMPI_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export PATH="${CUDA_ROOT}/bin:${PATH}"

MKL_ROOT="${ONEAPI_ROOT}/mkl/latest"

# ===========================================================================
# CMAKE CONFIGURE
# ===========================================================================
cmake_args=(
    -S "${SRC}" -B "${BLD}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release

    # Compilers — gcc found via PATH
    -DCMAKE_C_COMPILER=gcc
    -DCMAKE_CXX_COMPILER=g++
    -DCMAKE_Fortran_COMPILER=gfortran

    # MKL with OpenMPI BLACS variant and Intel OpenMP. ELPA's OpenMP helper
    # keeps -fopenmp for GNU compilation but strips it from link commands so
    # the final link uses explicit libiomp5 instead of libgomp.
    -DMKL_ROOT="${MKL_ROOT}"
    -DCMAKE_PREFIX_PATH="${MKL_ROOT};${COMPILER_ROOT}"
    -DMKL_MPI=openmpi
    -DMKL_THREADING=intel_thread
    -DOMP_LIBRARY="${IOMP5_LIBRARY}"

    # Features
    -DELPA_OPENMP=ON
    -DELPA_CUDA=ON
    # -DELPA_CUDA_ARCHITECTURES="75;80;90"  # "native" is default
    -DELPA_TEST_EXTENDED=ON

    # Example for restricting the framework code and default kernels to AVX2,
    # but allowing AVX-512 kernels to be selected at runtime.
    # The default is to use the highest available ISA of the build host.
    # -DELPA_ENABLE_AVX512_KERNELS=ON
    # -DELPA_DEFAULT_REAL_KERNEL=real_avx2_block2
    # -DELPA_DEFAULT_COMPLEX_KERNEL=complex_avx2_block1
)

echo "=== ELPA configure: gcc + gfortran + OpenMPI + MKL + CUDA ==="
cmake "${cmake_args[@]}"
echo "=== Configure exit code: $? ==="
