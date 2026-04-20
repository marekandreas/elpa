#!/usr/bin/env bash
# configure-clang-ifx-mkl-impi.sh
#
# Configure ELPA: clang + ifx + Intel MKL (intel_thread) + Intel MPI + CUDA
#
# Mixed LLVM/Intel toolchain: clang/clang++ for C/C++, ifx for Fortran.
# OpenMP uses Intel's libiomp5 throughout, consistent with MKL intel_thread:
#   - C/C++ compiled with clang -fopenmp=libiomp5 (selects iomp5 directly).
#   - Fortran compiled with ifx -qopenmp (links libiomp5 natively).
# Intel MPI is configured via I_MPI_CC/CXX env vars so the generic mpicc/
# mpicxx wrappers use clang; mpiifx is the dedicated Fortran wrapper.
# nvcc uses gcc as its CUDA host compiler (auto-detected by cmake when
# CMAKE_C_COMPILER is not gcc).
#
# Note: MKL SDL (MKL_LINK=sdl / libmkl_rt.so) is not supported because
# ELPA requires ScaLAPACK and MKLConfig.cmake rejects SDL when cluster
# libraries (BLACS/ScaLAPACK) are enabled.
#
# Prerequisites
#   LLVM 21 (clang-21, clang++-21, libomp-21-dev)  from apt.llvm.org
#   Intel oneAPI HPC Toolkit    (ifx, Intel MPI)
#   Intel oneAPI Base Toolkit   (MKL, libiomp5.so, MKLConfig.cmake)
#   cmake >= 3.24, ninja >= 1.10
#   GCC (any version — only needed as nvcc host compiler)
#   NVIDIA CUDA Toolkit >= 12.x from developer.nvidia.com
#
# Usage
#   # Override any PATHS variable via the environment if needed, then:
#   ./configure-clang-ifx-mkl-impi.sh
#   cmake --build <BLD> -j8
#   cd <BLD> && /path/to/cmake_build_examples/linux/test.sh --all -j8

set -euo pipefail

# ===========================================================================
# PATHS — override any of these via the environment
# ===========================================================================
SRC="${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
SCRATCH_DIR="${SCRATCH_DIR:-${SRC}/.scratch}"
BLD="${BLD:-${SCRATCH_DIR}/build-clang-ifx}"

# LLVM version suffix (e.g. "-21" for clang-21)
LLVM_VER="${LLVM_VER:--21}"

# LLVM installation directory
LLVM_DIR="${LLVM_DIR:-/usr/lib/llvm${LLVM_VER#-}}"

# Custom GCC installation used for nvcc host compiler (leave empty to use
# system gcc found on PATH)
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

# Prevent CPATH contamination of clang include discovery.
unset CPATH
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH
unset INCLUDE
unset OBJC_INCLUDE_PATH

MKL_ROOT="${ONEAPI_ROOT}/mkl/latest"
MPI_ROOT="${ONEAPI_ROOT}/mpi/latest"

export PATH="${LLVM_DIR}/bin:${MPI_ROOT}/bin:${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${MPI_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export I_MPI_ROOT="${MPI_ROOT}"

if [[ -n "${GCC_HOME}" ]]; then
    export PATH="${GCC_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${GCC_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Tell Intel MPI generic wrappers to use clang for C/C++.
# mpiifx has its own dedicated wrapper so no env var needed for Fortran.
export I_MPI_CC="clang${LLVM_VER}"
export I_MPI_CXX="clang++${LLVM_VER}"

# ===========================================================================
# CMAKE CONFIGURE
# ===========================================================================
cmake_args=(
    -S "${SRC}" -B "${BLD}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release

    # Mixed compiler: LLVM clang for C/C++, Intel ifx for Fortran
    -DCMAKE_C_COMPILER="clang${LLVM_VER}"
    -DCMAKE_CXX_COMPILER="clang++${LLVM_VER}"
    -DCMAKE_Fortran_COMPILER="${COMPILER_ROOT}/bin/ifx"

    # Intel MPI: generic wrappers use I_MPI_CC/CXX (set above); mpiifx for Fortran
    -DMPI_C_COMPILER="${MPI_ROOT}/bin/mpicc"
    -DMPI_CXX_COMPILER="${MPI_ROOT}/bin/mpicxx"
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

echo "=== ELPA configure: clang${LLVM_VER} + ifx + MKL (intel_thread) + Intel MPI + CUDA ==="
cmake "${cmake_args[@]}"
echo "=== Configure exit code: $? ==="
