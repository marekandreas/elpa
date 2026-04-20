#!/usr/bin/env bash
# configure-gcc-gfortran-openblas-openmpi.sh
#
# Configure ELPA: gcc + gfortran + OpenMPI + OpenBLAS + CUDA
#
# Community-stack build requiring no vendor compilers or math libraries.
# All dependencies are available from your distribution's package manager.
# OpenBLAS provides good general-purpose BLAS performance across x86_64
# platforms.
#
# Prerequisites (Ubuntu/Debian)
#   sudo apt install gcc g++ gfortran cmake ninja-build
#   sudo apt install libopenblas-dev libscalapack-openmpi-dev
#   NVIDIA CUDA Toolkit >= 12.x from developer.nvidia.com
#   OpenMPI — system packages or custom build
#
# Usage
#   # Override any PATHS variable via the environment if needed, then:
#   ./configure-gcc-gfortran-openblas-openmpi.sh
#   cmake --build <BLD> -j8
#   cd <BLD> && /path/to/cmake_build_examples/linux/test.sh --all -j8

set -euo pipefail

# ===========================================================================
# PATHS — override any of these via the environment
# ===========================================================================
SRC="${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
SCRATCH_DIR="${SCRATCH_DIR:-${SRC}/.scratch}"
BLD="${BLD:-${SCRATCH_DIR}/build-gcc-openblas}"

# Custom GCC installation (leave empty to use system gcc)
GCC_HOME="${GCC_HOME:-}"

# Custom OpenMPI installation (leave empty to use system mpicc)
OMPI_HOME="${OMPI_HOME:-}"

# CUDA Toolkit root
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"

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

export PATH="${CUDA_ROOT}/bin:${PATH}"

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

echo "=== ELPA configure: gcc + gfortran + OpenMPI + OpenBLAS + CUDA ==="
cmake "${cmake_args[@]}"
echo "=== Configure exit code: $? ==="
