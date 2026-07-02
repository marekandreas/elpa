#!/usr/bin/env bash
# configure-gcc-gfortran-blis-openmpi.sh
#
# Configure ELPA: gcc + gfortran + OpenMPI + AOCL-BLIS + CUDA
#
# Uses AMD AOCL libraries: BLIS-mt (BLAS), FLAME (LAPACK), and
# AOCL-ScaLAPACK.  This combination is well-suited for AMD Zen
# processors where BLIS can outperform OpenBLAS significantly.
#
# Note: AOCL-ScaLAPACK is typically built with AOCC and links against
# AOCC runtime libraries (libflang.so, libflangrti.so, libpgmath.so).
# When used with GCC, the AOCC runtime directory must be on the
# linker search path — this script handles it via CMAKE linker flags.
# If your AOCL-ScaLAPACK was built with GCC, set AOCC_ROOT="" to
# skip the runtime path injection.
#
# Prerequisites
#   gcc >= 10, cmake >= 3.24, ninja >= 1.10
#   AOCL >= 5.x (BLIS-mt, FLAME, ScaLAPACK)
#   OpenMPI — system packages or custom build
#   NVIDIA CUDA Toolkit >= 12.x from developer.nvidia.com
#
# Usage
#   # Override any PATHS variable via the environment if needed, then:
#   ./configure-gcc-gfortran-blis-openmpi.sh
#   cmake --build <BLD> -j8
#   cd <BLD> && /path/to/cmake_build_examples/linux/test.sh --all -j8

set -euo pipefail

# ===========================================================================
# PATHS — override any of these via the environment
# ===========================================================================
SRC="${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
SCRATCH_DIR="${SCRATCH_DIR:-${SRC}/.scratch}"
BLD="${BLD:-${SCRATCH_DIR}/build-gcc-blis}"

# Custom GCC installation (leave empty to use system gcc)
GCC_HOME="${GCC_HOME:-}"

# Custom OpenMPI installation (leave empty to use system mpicc)
OMPI_HOME="${OMPI_HOME:-}"

# AOCL root — expects lib/libblis-mt.so, lib/libflame.so, lib/libscalapack.so
AOCL_ROOT="${AOCL_ROOT:-}"

# AOCC runtime (only needed if AOCL-ScaLAPACK was built with AOCC).
# Leave empty if your AOCL was built with GCC or if AOCC libs are already
# in your system library path.
AOCC_ROOT="${AOCC_ROOT:-}"

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

# AOCC runtime linker flags for AOCL-ScaLAPACK dependencies
EXTRA_LINKER_FLAGS=""
if [[ -n "${AOCC_ROOT}" ]]; then
    EXTRA_LINKER_FLAGS="-L${AOCC_ROOT}/lib -Wl,-rpath,${AOCC_ROOT}/lib"
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

    # AOCL BLAS/LAPACK/ScaLAPACK
    -DBLAS_LIBRARIES="${AOCL_ROOT}/lib/libblis-mt.so"
    -DLAPACK_LIBRARIES="${AOCL_ROOT}/lib/libflame.so"
    -DSCALAPACK_LIBRARY="${AOCL_ROOT}/lib/libscalapack.so"

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

# Inject AOCC runtime path if needed
if [[ -n "${EXTRA_LINKER_FLAGS}" ]]; then
    cmake_args+=(
        -DCMAKE_EXE_LINKER_FLAGS="${EXTRA_LINKER_FLAGS}"
        -DCMAKE_SHARED_LINKER_FLAGS="${EXTRA_LINKER_FLAGS}"
    )
fi

echo "=== ELPA configure: gcc + gfortran + OpenMPI + AOCL-BLIS + CUDA ==="
cmake "${cmake_args[@]}"
echo "=== Configure exit code: $? ==="
