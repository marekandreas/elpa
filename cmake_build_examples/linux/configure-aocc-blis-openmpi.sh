#!/usr/bin/env bash
# configure-aocc-blis-openmpi.sh
#
# Configure ELPA: AOCC (clang/flang) + OpenMPI + AOCL-BLIS — CPU only
#
# AMD toolchain optimised for Zen processors.  AOCC provides tuned
# classic Flang (F18-based) and Clang (LLVM 17-based) compilers.
# Combined with AOCL's BLIS, FLAME, and ScaLAPACK this delivers
# excellent performance on AMD hardware.
#
# CUDA is disabled because AOCC's classic flang does not support the
# Fortran-CUDA interop paths used by ELPA.
#
# Important: AOCC's classic flang does NOT support the OpenMP 5.1
# !$omp masked directive.  ELPA's CMake detects this at configure
# time via a runtime test and automatically falls back to
# !$omp master.
#
# Prerequisites
#   AOCC >= 5.x  (from developer.amd.com)
#   AOCL >= 5.x  (BLIS-mt, FLAME, ScaLAPACK — from developer.amd.com)
#   cmake >= 3.24, ninja >= 1.10
#   OpenMPI built with AOCC compilers
#
# Usage
#   # Override any PATHS variable via the environment if needed, then:
#   ./configure-aocc-blis-openmpi.sh
#   cmake --build <BLD> -j8
#   cd <BLD> && /path/to/cmake_build_examples/linux/test.sh --all -j8

set -euo pipefail

# ===========================================================================
# PATHS — override any of these via the environment
# ===========================================================================
SRC="${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
SCRATCH_DIR="${SCRATCH_DIR:-${SRC}/.scratch}"
BLD="${BLD:-${SCRATCH_DIR}/build-aocc}"

# AOCC installation root — expects bin/clang, bin/flang
AOCC_ROOT="${AOCC_ROOT:-}"

# AOCL root — expects lib/libblis-mt.so, lib/libflame.so, lib/libscalapack.so
AOCL_ROOT="${AOCL_ROOT:-}"

# Custom OpenMPI installation built with AOCC compilers
OMPI_AOCC="${OMPI_AOCC:-}"

# ===========================================================================
# ENVIRONMENT
# ===========================================================================
OMPI_HOME="${OMPI_AOCC}"

if [[ -z "${AOCC_ROOT}" ]]; then
    echo "ERROR: AOCC_ROOT not set — point it to the AOCC installation directory." >&2
    exit 1
fi
if [[ -z "${AOCL_ROOT}" ]]; then
    echo "ERROR: AOCL_ROOT not set — point it to the AOCL installation directory." >&2
    exit 1
fi

export PATH="${AOCC_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${AOCC_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -n "${OMPI_HOME}" ]]; then
    export PATH="${OMPI_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${OMPI_HOME}/lib:${AOCL_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Tell OpenMPI wrappers which compilers to use
export OMPI_CC=clang
export OMPI_CXX=clang++
export OMPI_FC=flang

# Point CMake's OpenMP detection to the AOCC libomp to avoid conflicts
# with any system-installed libomp.
AOCC_LIBOMP="${AOCC_ROOT}/lib/libomp.so"

# ===========================================================================
# CMAKE CONFIGURE
# ===========================================================================
cmake_args=(
    -S "${SRC}" -B "${BLD}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release

    # AOCC compilers
    -DCMAKE_C_COMPILER=clang
    -DCMAKE_CXX_COMPILER=clang++
    -DCMAKE_Fortran_COMPILER=flang

    # AOCL BLAS/LAPACK/ScaLAPACK
    -DBLAS_LIBRARIES="${AOCL_ROOT}/lib/libblis-mt.so"
    -DLAPACK_LIBRARIES="${AOCL_ROOT}/lib/libflame.so"
    -DSCALAPACK_LIBRARY="${AOCL_ROOT}/lib/libscalapack.so"

    # OpenMP — use AOCC's libomp
    -DOpenMP_omp_LIBRARY="${AOCC_LIBOMP}"

    # CPU-only (no CUDA with AOCC)
    -DELPA_OPENMP=ON
    -DELPA_CUDA=OFF
    -DELPA_TEST_EXTENDED=ON

    # Example for restricting the framework code and default kernels to AVX2,
    # but allowing AVX-512 kernels to be selected at runtime.
    # The default is to use the highest available ISA of the build host.
    # -DELPA_ENABLE_AVX512_KERNELS=ON
    # -DELPA_DEFAULT_REAL_KERNEL=real_avx2_block2
    # -DELPA_DEFAULT_COMPLEX_KERNEL=complex_avx2_block1
)

echo "=== ELPA configure: AOCC + OpenMPI + AOCL-BLIS — CPU only ==="
cmake "${cmake_args[@]}"
echo "=== Configure exit code: $? ==="
