#!/usr/bin/env bash
# configure-gcc-gfortran-openblas-openmpi-aarch64.sh
#
# Configure ELPA: gcc + gfortran + OpenMPI + OpenBLAS  (AArch64 / ARM64)
#
# Validated on: Oracle Cloud Infrastructure A1 instance,
#               Ubuntu 24.04 LTS (aarch64), no GPU.
#
# Prerequisites (Ubuntu/Debian)
#   sudo apt install gcc g++ gfortran cmake ninja-build
#   sudo apt install libopenblas-dev libscalapack-openmpi-dev
#   OpenMPI — system packages or custom build
#
# On AArch64 there are no x86 SIMD kernels.  ELPA automatically builds the
# NEON AArch64 kernel families (BLOCK2, BLOCK4, BLOCK6) when compiling with
# -march=native on an AArch64 host.  CUDA is not enabled here because A1
# instances do not have a GPU.
#
# Usage
#   # Override any PATHS variable via the environment if needed, then:
#   ./configure-gcc-gfortran-openblas-openmpi-aarch64.sh
#   cmake --build <BLD> -j8
#   cd <BLD> && /path/to/cmake_build_examples/linux/test.sh --all -j8

set -euo pipefail

# ===========================================================================
# PATHS — override any of these via the environment
# ===========================================================================
SRC="${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
SCRATCH_DIR="${SCRATCH_DIR:-${SRC}/.scratch}"
BLD="${BLD:-${SCRATCH_DIR}/build-gcc-aarch64}"

# Custom GCC installation (leave empty to use system gcc)
GCC_HOME="${GCC_HOME:-}"

# Custom OpenMPI installation (leave empty to use system mpicc)
OMPI_HOME="${OMPI_HOME:-}"

# ===========================================================================
# ENVIRONMENT
# ===========================================================================
if [[ -n "${GCC_HOME}" ]]; then
    export PATH="${GCC_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${GCC_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ -n "${OMPI_HOME}" ]]; then
    export PATH="${OMPI_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${OMPI_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

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

    # Community BLAS/LAPACK/ScaLAPACK (not MKL)
    -DELPA_USE_MKL=OFF

    # Features — no CUDA on A1
    -DELPA_OPENMP=ON
    -DELPA_CUDA=OFF
    -DELPA_TEST_EXTENDED=ON

    # -march=native enables NEON ARCH64 kernels on AArch64 hosts.
    # x86 kernel families are disabled automatically on non-x86 platforms.
    -DELPA_FRAMEWORK_ISA=native
)

echo "=== ELPA configure: gcc + gfortran + OpenMPI + OpenBLAS (AArch64) ==="
cmake "${cmake_args[@]}"
echo "=== Configure exit code: $? ==="
