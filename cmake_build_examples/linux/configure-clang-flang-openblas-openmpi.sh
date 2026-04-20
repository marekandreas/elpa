#!/usr/bin/env bash
# configure-clang-flang-openblas-openmpi.sh
#
# Configure ELPA: clang + flang-new (LLVM) + OpenMPI + OpenBLAS + CUDA
#
# Full LLVM toolchain.  Requires LLVM >= 19 for flang-new polymorphic
# type support; LLVM >= 21 recommended (eliminates the experimental
# OpenMP warning).  nvcc uses gcc as its CUDA host compiler
# (auto-detected by cmake when CC is Clang).
#
# Prerequisites (Ubuntu/Debian — example for LLVM 21)
#   From apt.llvm.org:
#     sudo apt install clang-21 flang-21 libflang-21-dev libomp-21-dev
#   System packages:
#     sudo apt install cmake ninja-build libopenblas-dev
#   OpenMPI built with clang/flang-new (system packages use gfortran
#   modules which are ABI-incompatible with flang-new)
#   NVIDIA CUDA Toolkit >= 12.x from developer.nvidia.com
#   GCC (any version — only needed as nvcc host compiler)
#
# Usage
#   # Override any PATHS variable via the environment if needed, then:
#   ./configure-clang-flang-openblas-openmpi.sh
#   cmake --build <BLD> -j8
#   cd <BLD> && /path/to/cmake_build_examples/linux/test.sh --all -j8

set -euo pipefail

# ===========================================================================
# PATHS — override any of these via the environment
# ===========================================================================
SRC="${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
SCRATCH_DIR="${SCRATCH_DIR:-${SRC}/.scratch}"
BLD="${BLD:-${SCRATCH_DIR}/build-clang-openblas}"

# LLVM version suffix (e.g. "-21" for clang-21, flang-new-21)
LLVM_VER="${LLVM_VER:--21}"

# LLVM installation directory
LLVM_DIR="${LLVM_DIR:-/usr/lib/llvm${LLVM_VER#-}}"

# Custom OpenMPI installation built with clang/flang-new.
# System OpenMPI Fortran modules are built with gfortran and will cause
# module-format errors when used with flang-new.
OMPI_CLANG="${OMPI_CLANG:-}"

# GCC installation (for its C++ stdlib and as nvcc host compiler)
GCC_HOME="${GCC_HOME:-}"

# CUDA Toolkit root
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"

# ===========================================================================
# ENVIRONMENT
# ===========================================================================
if [[ -n "${GCC_HOME}" ]]; then
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

    # Community BLAS/LAPACK/ScaLAPACK
    -DELPA_USE_MKL=OFF

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

echo "=== ELPA configure: clang${LLVM_VER} + flang-new${LLVM_VER} + OpenMPI + OpenBLAS + CUDA ==="
cmake "${cmake_args[@]}"
echo "=== Configure exit code: $? ==="
