#!/usr/bin/env bash
# configure-gcc-gfortran-accelerate-scalapack-openmpi-macos-arm64.sh
#
# Configure ELPA: gcc + gfortran + OpenMPI + Accelerate + source-built ScaLAPACK
# (macOS / Apple Silicon)
#
# Validated on: Apple MacBook Air (Mac16,12)
#               Apple M4 (4 performance + 6 efficiency cores, 24 GB)
#               macOS Sequoia 15.5
#
# Prerequisites — install via Homebrew (https://brew.sh):
#   brew install cmake gcc openmpi libomp pkgconf
#
# Before running this script, build ScaLAPACK from source against Accelerate
# and install it to ${SCALAPACK_PREFIX}.  See the companion helper script:
#   ./cmake_build_examples/macos/build-scalapack-accelerate-macos-arm64.sh
#
# Key macOS / Apple Silicon notes:
#   - macOS reports CMAKE_SYSTEM_PROCESSOR as "arm64".
#   - Accelerate supplies BLAS/LAPACK; ScaLAPACK must be built separately.
#   - NEON AArch64 kernel families (BLOCK2, BLOCK4, BLOCK6) are auto-enabled
#     on Apple Silicon and selected as the default kernel with -march=native.
#   - There is no CUDA support on Apple Silicon; ELPA_CUDA is OFF.
#   - DYLD_LIBRARY_PATH is restricted on macOS (SIP).  The ScaLAPACK install
#     prefix is linked explicitly and no runtime environment variable is needed.
#
# Usage
#   # Override any PATHS variable via the environment if needed, then:
#   ./cmake_build_examples/macos/configure-gcc-gfortran-accelerate-scalapack-openmpi-macos-arm64.sh
#   cmake --build <BLD> -j$(sysctl -n hw.ncpu)
#   cd <BLD> && ctest -j$(sysctl -n hw.ncpu) --timeout 300 --output-on-failure -E autotune

set -euo pipefail

# ===========================================================================
# PATHS — override any of these via the environment
# ===========================================================================
SRC="${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
BREW_PREFIX="${BREW_PREFIX:-$(brew --prefix 2>/dev/null || echo /opt/homebrew)}"
GCC_VER="${GCC_VER:-15}"
SCALAPACK_PREFIX="${SCALAPACK_PREFIX:-${HOME}/opt/scalapack-accelerate}"
BLD="${BLD:-${SRC}/build-macos-arm64-gcc-openmpi-accelerate}"

# ===========================================================================
# DERIVED PATHS  (no edits needed below this line for a standard Homebrew setup)
# ===========================================================================
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
    -DELPA_MPI_C_COMPILER="${OMPI_PREFIX}/bin/mpicc"
    -DELPA_MPI_CXX_COMPILER="${OMPI_PREFIX}/bin/mpicxx"
    -DELPA_MPI_Fortran_COMPILER="${OMPI_PREFIX}/bin/mpifort"

    # Clean macOS math stack: Accelerate for BLAS/LAPACK, source-built ScaLAPACK.
    -DELPA_USE_MKL=OFF
    -DBLA_VENDOR=Apple
    -DSCALAPACK_LIBRARY="${SCALAPACK_PREFIX}/lib/libscalapack.dylib"

    # Help CMake find the ScaLAPACK install and OpenMPI wrappers.
    -DCMAKE_PREFIX_PATH="${SCALAPACK_PREFIX};${OMPI_PREFIX};${BREW_PREFIX}"

    # Features — no CUDA on Apple Silicon
    -DELPA_OPENMP=ON
    -DELPA_CUDA=OFF
    -DELPA_TEST_EXTENDED=ON

    # -march=native enables NEON AArch64 kernel families on Apple Silicon.
    # All x86 kernel families are excluded automatically on non-x86 hosts.
    -DELPA_FRAMEWORK_ISA=native

    # GCC ≥14 on Apple Silicon (macOS Sequoia) has a vectorization miscompilation
    # bug (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=120817) at -O2/-O3.
    # The flag must live in CMAKE_Fortran_FLAGS_RELEASE so it appends after -O3.
    # NOTE: Apple Accelerate vecLib has two broken single-precision LAPACK
    # auxiliary routines (as of macOS Sequoia 15.5):
    #   - SLAMCH('E') returns 0.0 (should be ~1.19e-7, machine epsilon)
    #   - SLAPY2(x,y) returns ~0.0 or negative (should be sqrt(x²+y²))
    # Both are used by ELPA's distributed D&C tridiagonal eigensolver and QR
    # factorisation.  The fixes in src/solve_tridi/merge_systems_template.F90
    # and src/elpa2/qr/elpa_pdgeqrf_template.F90 replace these calls with the
    # equivalent portable Fortran intrinsics (epsilon(), hypot()).
    "-DCMAKE_Fortran_FLAGS_RELEASE=-O3 -fno-tree-loop-vectorize"
)

echo "=== ELPA configure: gcc-${GCC_VER} + gfortran-${GCC_VER} + OpenMPI + Accelerate + source-built ScaLAPACK (macOS Apple Silicon) ==="
cmake "${cmake_args[@]}"
echo "=== Configure exit code: $? ==="
