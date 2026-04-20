#!/usr/bin/env bash
# configure-gcc-ifx-mkl-impi.sh
#
# Configure ELPA: gcc + ifx + Intel MKL (intel_thread) + Intel MPI + CUDA
#
# Mixed toolchain: GCC for C/C++, Intel Fortran (ifx) for Fortran.
# OpenMP uses Intel's libiomp5 throughout so MKL intel_thread is consistent:
#   - C/C++ code compiled with gcc -fopenmp; the -fopenmp link flag is then
#     stripped from the OpenMP imported targets and replaced with explicit
#     libiomp5 linkage (see cmake/dependencies/ElpaOpenMP.cmake).
#   - Fortran code compiled with ifx -qopenmp, which links libiomp5 natively.
# Intel MPI uses the dedicated gcc wrappers for C/C++ and mpiifx for Fortran.
# nvcc uses gcc as its CUDA host compiler (auto-detected by cmake).
#
# Note: MKL SDL (MKL_LINK=sdl / libmkl_rt.so) is not supported because
# ELPA requires ScaLAPACK and MKLConfig.cmake rejects SDL when cluster
# libraries (BLACS/ScaLAPACK) are enabled.
#
# Prerequisites
#   gcc >= 10, g++              (system packages)
#   Intel oneAPI HPC Toolkit    (ifx, Intel MPI)
#   Intel oneAPI Base Toolkit   (MKL, icx — icx not used here but provides
#                                libiomp5 and MKLConfig.cmake)
#   cmake >= 3.24, ninja >= 1.10
#   NVIDIA CUDA Toolkit >= 12.x from developer.nvidia.com
#
# Usage
#   # Override any PATHS variable via the environment if needed, then:
#   ./configure-gcc-ifx-mkl-impi.sh
#   cmake --build <BLD> -j8
#   cd <BLD> && /path/to/cmake_build_examples/linux/test.sh --all -j8

set -euo pipefail

# ===========================================================================
# PATHS — override any of these via the environment
# ===========================================================================
SRC="${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
SCRATCH_DIR="${SCRATCH_DIR:-${SRC}/.scratch}"
BLD="${BLD:-${SCRATCH_DIR}/build-gcc-ifx}"

# Custom GCC installation (leave empty to use system gcc)
GCC_HOME="${GCC_HOME:-}"

# Intel oneAPI root
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

# Prevent contamination of GCC compiler discovery by Intel CPATH injections.
unset CPATH
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH
unset INCLUDE
unset OBJC_INCLUDE_PATH

MKL_ROOT="${ONEAPI_ROOT}/mkl/latest"
MPI_ROOT="${ONEAPI_ROOT}/mpi/latest"

export PATH="${MPI_ROOT}/bin:${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${MPI_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export I_MPI_ROOT="${MPI_ROOT}"

if [[ -n "${GCC_HOME}" ]]; then
    export PATH="${GCC_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${GCC_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ===========================================================================
# CMAKE CONFIGURE
# ===========================================================================
cmake_args=(
    -S "${SRC}" -B "${BLD}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release

    # Mixed compiler: GCC for C/C++, Intel ifx for Fortran
    -DCMAKE_C_COMPILER=gcc
    -DCMAKE_CXX_COMPILER=g++
    -DCMAKE_Fortran_COMPILER="${COMPILER_ROOT}/bin/ifx"

    # Intel MPI wrappers: gcc for C/C++, mpiifx for Fortran
    -DMPI_C_COMPILER="${MPI_ROOT}/bin/mpigcc"
    -DMPI_CXX_COMPILER="${MPI_ROOT}/bin/mpigxx"
    -DMPI_Fortran_COMPILER="${MPI_ROOT}/bin/mpiifx"
    -DELPA_MPI_ROOT="${MPI_ROOT}"
    -DCMAKE_PREFIX_PATH="${MPI_ROOT};${MKL_ROOT};${COMPILER_ROOT}"
    -DMPI_C_HEADER_DIR="${MPI_ROOT}/include"
    -DMPI_CXX_HEADER_DIR="${MPI_ROOT}/include"

    # Intel compiler runtime rpath (libifport, libintlc, libsvml, libifcoremt)
    -DCMAKE_BUILD_RPATH="${COMPILER_ROOT}/lib"

    # MKL — intel_thread uses libmkl_intel_thread + libiomp5
    -DELPA_MKL_ROOT="${MKL_ROOT}"
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

echo "=== ELPA configure: gcc + ifx + MKL (intel_thread) + Intel MPI + CUDA ==="
cmake "${cmake_args[@]}"
echo "=== Configure exit code: $? ==="
