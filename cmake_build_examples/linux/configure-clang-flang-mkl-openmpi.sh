#!/usr/bin/env bash
# configure-clang-flang-mkl-openmpi.sh
#
# Configure ELPA: clang + flang-new (LLVM) + OpenMPI + Intel MKL + CUDA
#
# Full LLVM toolchain with Intel MKL for math libraries.  Uses the
# OpenMPI BLACS variant from MKL and Intel's libiomp5 OpenMP runtime.
# Requires LLVM >= 19 for flang-new polymorphic type support; LLVM >= 21
# recommended (eliminates the experimental OpenMP warning).
#
# Prerequisites (Ubuntu/Debian — example for LLVM 21)
#   From apt.llvm.org:
#     sudo apt install clang-21 flang-21 libflang-21-dev libomp-21-dev
#   System packages:
#     sudo apt install cmake ninja-build
#   Intel oneAPI Base Toolkit (MKL)
#   OpenMPI built with clang/flang-new (system packages use gfortran
#   modules which are ABI-incompatible with flang-new)
#   NVIDIA CUDA Toolkit >= 12.x from developer.nvidia.com
#   GCC (any version — only needed as nvcc host compiler)
#
# Usage
#   # Override any PATHS variable via the environment if needed, then:
#   ./configure-clang-flang-mkl-openmpi.sh
#   cmake --build <BLD> -j8
#   cd <BLD> && /path/to/cmake_build_examples/linux/test.sh --all -j8

set -euo pipefail

# ===========================================================================
# PATHS — override any of these via the environment
# ===========================================================================
SRC="${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
SCRATCH_DIR="${SCRATCH_DIR:-${SRC}/.scratch}"
BLD="${BLD:-${SCRATCH_DIR}/build-clang-mkl}"

# LLVM version suffix (e.g. "-21" for clang-21, flang-new-21)
LLVM_VER="${LLVM_VER:--21}"

# LLVM installation directory
LLVM_DIR="${LLVM_DIR:-/usr/lib/llvm${LLVM_VER#-}}"

# Custom OpenMPI installation built with clang/flang-new.
OMPI_CLANG="${OMPI_CLANG:-}"

# Intel oneAPI root
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
COMPILER_ROOT="${ONEAPI_ROOT}/compiler/latest"
COMPILER_VARS_SH="${COMPILER_ROOT}/env/vars.sh"
IOMP5_LIBRARY="${COMPILER_ROOT}/lib/libiomp5.so"

# GCC installation (for its C++ stdlib and as nvcc host compiler)
GCC_HOME="${GCC_HOME:-}"

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

OMPI_HOME="${OMPI_CLANG}"
if [[ -n "${OMPI_HOME}" ]]; then
    export PATH="${OMPI_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${OMPI_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export PATH="${LLVM_DIR}/bin:${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${LLVM_DIR}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Tell OpenMPI wrappers which compilers to use
export OMPI_CC="clang${LLVM_VER}"
export OMPI_CXX="clang++${LLVM_VER}"
export OMPI_FC="flang-new${LLVM_VER}"

MKL_ROOT="${ONEAPI_ROOT}/mkl/latest"

# ===========================================================================
# CMAKE CONFIGURE
# ===========================================================================
cmake_args=(
    -S "${SRC}" -B "${BLD}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release

    # LLVM compilers
    -DCMAKE_C_COMPILER="clang${LLVM_VER}"
    -DCMAKE_CXX_COMPILER="clang++${LLVM_VER}"
    -DCMAKE_Fortran_COMPILER="flang-new${LLVM_VER}"

    # MKL with OpenMPI BLACS variant
    -DMKL_ROOT="${MKL_ROOT}"
    -DCMAKE_PREFIX_PATH="${MKL_ROOT};${COMPILER_ROOT}"
    -DMKL_MPI=openmpi
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

echo "=== ELPA configure: clang${LLVM_VER} + flang-new${LLVM_VER} + OpenMPI + MKL + CUDA ==="
cmake "${cmake_args[@]}"
echo "=== Configure exit code: $? ==="
