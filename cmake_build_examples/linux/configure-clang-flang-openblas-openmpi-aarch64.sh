#!/usr/bin/env bash
# configure-clang-flang-openblas-openmpi-aarch64.sh
#
# Configure ELPA: clang + flang (LLVM) + OpenMPI + OpenBLAS  (AArch64 / ARM64)
#
# Validated on: Oracle Cloud Infrastructure A1 instance,
#               Ubuntu 24.04 LTS (aarch64), no GPU.
#
# Prerequisites (Ubuntu/Debian — example for LLVM 21)
#   From apt.llvm.org:
#     sudo apt install clang-21 flang-21 libflang-21-dev libomp-21-dev
#   System packages:
#     sudo apt install cmake ninja-build libopenblas-dev
#   OpenMPI must be built with the same flang version; the system
#   OpenMPI Fortran modules are built with gfortran and are ABI-incompatible.
#
# On AArch64, -march=native enables the NEON ARCH64 kernel families
# (BLOCK2, BLOCK4, BLOCK6). x86 kernel families are excluded automatically.
# CUDA is not enabled because A1 instances have no GPU.
#
# Usage
#   # Override any PATHS variable via the environment if needed, then:
#   ./configure-clang-flang-openblas-openmpi-aarch64.sh
#   cmake --build <BLD> -j8
#   cd <BLD> && /path/to/cmake_build_examples/linux/test.sh --all -j8

set -euo pipefail

# ===========================================================================
# PATHS — override any of these via the environment
# ===========================================================================
SRC="${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
SCRATCH_DIR="${SCRATCH_DIR:-${SRC}/.scratch}"
BLD="${BLD:-${SCRATCH_DIR}/build-clang-aarch64}"

# LLVM version suffix (e.g. "-21" for clang-21, flang-21)
LLVM_VER="${LLVM_VER:--21}"

# LLVM installation directory
LLVM_DIR="${LLVM_DIR:-/usr/lib/llvm${LLVM_VER#-}}"

# Custom OpenMPI installation built with clang/flang of the same version.
OMPI_CLANG="${OMPI_CLANG:-}"

# ===========================================================================
# ENVIRONMENT
# ===========================================================================
OMPI_HOME="${OMPI_CLANG}"
if [[ -n "${OMPI_HOME}" ]]; then
    export PATH="${OMPI_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${OMPI_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export PATH="${LLVM_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${LLVM_DIR}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Tell OpenMPI wrappers which compilers to use
export OMPI_CC="clang${LLVM_VER}"
export OMPI_CXX="clang++${LLVM_VER}"
export OMPI_FC="flang${LLVM_VER}"

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
    -DCMAKE_Fortran_COMPILER="flang${LLVM_VER}"

    # Community BLAS/LAPACK/ScaLAPACK (not MKL)
    -DELPA_BLAS_VENDOR=OpenBLAS

    # Features — no CUDA on A1
    -DELPA_OPENMP=ON
    -DELPA_CUDA=OFF
    -DELPA_TEST_EXTENDED=ON

    # -march=native enables NEON ARCH64 kernels on AArch64 hosts.
    # x86 kernel families are disabled automatically on non-x86 platforms.
    -DELPA_FRAMEWORK_ISA=native
)

echo "=== ELPA configure: clang${LLVM_VER} + flang${LLVM_VER} + OpenMPI + OpenBLAS (AArch64) ==="
cmake "${cmake_args[@]}"
echo "=== Configure exit code: $? ==="
