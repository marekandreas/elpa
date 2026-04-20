#!/usr/bin/env bash
# configure-gcc-gfortran-openblas-openmpi-macos-arm64.sh
#
# Configure ELPA: gcc + gfortran + OpenMPI + OpenBLAS  (macOS / Apple Silicon)
#
# Validated on: Apple MacBook Air (Mac16,12)
#               Apple M4 (4 performance + 6 efficiency cores, 24 GB)
#               macOS Sequoia 15.5
#
# Prerequisites — install via Homebrew (https://brew.sh):
#   brew install cmake gcc openmpi openblas scalapack libomp pkgconf
#
# Key macOS / Apple Silicon notes:
#   - NEON AArch64 kernel families (BLOCK2, BLOCK4, BLOCK6) are auto-enabled
#     on Apple Silicon and selected as the default kernel with -march=native.
#   - openblas and scalapack are keg-only in Homebrew (not symlinked into
#     /opt/homebrew) because macOS ships Accelerate.framework.  The paths
#     below point directly into the keg.
#   - There is no CUDA support on Apple Silicon; ELPA_CUDA is OFF.
#   - DYLD_LIBRARY_PATH is restricted on macOS (SIP).  The Homebrew library
#     paths are embedded at link time via the keg rpath, so no runtime
#     environment variable is needed.
#
# Usage
#   # Override any PATHS variable via the environment if needed, then:
#   ./cmake_build_examples/macos/configure-gcc-gfortran-openblas-openmpi-macos-arm64.sh
#   cmake --build <BLD> -j$(sysctl -n hw.ncpu)
#   cd <BLD> && ctest -j$(sysctl -n hw.ncpu) --timeout 300 --output-on-failure -E autotune

set -euo pipefail

# ===========================================================================
# PATHS — override any of these via the environment
# ===========================================================================
SRC="${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
BREW_PREFIX="${BREW_PREFIX:-$(brew --prefix 2>/dev/null || echo /opt/homebrew)}"
GCC_VER="${GCC_VER:-15}"
BLD="${BLD:-${SRC}/build-macos-arm64-gcc-openmpi-openblas}"

# ===========================================================================
# DERIVED PATHS  (no edits needed below this line for a standard Homebrew setup)
# ===========================================================================
OPENBLAS_PREFIX="${BREW_PREFIX}/opt/openblas"
SCALAPACK_PREFIX="${BREW_PREFIX}/opt/scalapack"
OMPI_PREFIX="${BREW_PREFIX}/opt/open-mpi"

# ===========================================================================
# CMAKE CONFIGURE
# ===========================================================================
cmake_args=(
    -S "${SRC}" -B "${BLD}"
    -DCMAKE_BUILD_TYPE=Release

    # Homebrew GCC / GFortran (versioned binaries to avoid picking up Apple clang)
    -DCMAKE_C_COMPILER="${BREW_PREFIX}/bin/gcc-${GCC_VER}"
    -DCMAKE_CXX_COMPILER="${BREW_PREFIX}/bin/g++-${GCC_VER}"
    -DCMAKE_Fortran_COMPILER="${BREW_PREFIX}/bin/gfortran-${GCC_VER}"

    # Point ELPA's MPI detection at the Homebrew OpenMPI wrappers explicitly.
    # This ensures the wrappers use the same GCC version selected above.
    -DELPA_MPI_C_COMPILER="${OMPI_PREFIX}/bin/mpicc"
    -DELPA_MPI_CXX_COMPILER="${OMPI_PREFIX}/bin/mpicxx"
    -DELPA_MPI_Fortran_COMPILER="${OMPI_PREFIX}/bin/mpifort"

    # Community BLAS/LAPACK/ScaLAPACK — not MKL, not Accelerate
    -DELPA_USE_MKL=OFF
    -DBLAS_LIBRARIES="${OPENBLAS_PREFIX}/lib/libopenblas.dylib"
    -DLAPACK_LIBRARIES="${OPENBLAS_PREFIX}/lib/libopenblas.dylib"
    -DSCALAPACK_LIBRARY="${SCALAPACK_PREFIX}/lib/libscalapack.dylib"

    # Help CMake find the keg-only packages
    -DCMAKE_PREFIX_PATH="${OPENBLAS_PREFIX};${SCALAPACK_PREFIX};${OMPI_PREFIX};${BREW_PREFIX}"

    # Features — no CUDA on Apple Silicon
    -DELPA_OPENMP=ON
    -DELPA_CUDA=OFF
    -DELPA_TEST_EXTENDED=ON

    # -march=native enables NEON AArch64 kernel families on Apple Silicon.
    # All x86 kernel families are excluded automatically on non-x86 hosts.
    -DELPA_FRAMEWORK_ISA=native
)

echo "=== ELPA configure: gcc-${GCC_VER} + gfortran-${GCC_VER} + OpenMPI + OpenBLAS (macOS Apple Silicon) ==="
cmake "${cmake_args[@]}"
echo "=== Configure exit code: $? ==="