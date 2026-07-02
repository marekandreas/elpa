#!/usr/bin/env bash
# custom-openmpi-build.sh
#
# Download, build, and install OpenMPI from source using a non-gfortran
# compiler toolchain.  This is required because system-packaged OpenMPI
# ships Fortran .mod files compiled with gfortran, which are
# ABI-incompatible with flang-new (LLVM) and flang (AOCC).
#
# The resulting installation is self-contained and can be pointed at via
# OMPI_CLANG or OMPI_AOCC in the ELPA configure scripts.
#
# Usage
#   # LLVM clang/flang-new (uses LLVM_VER for version suffix)
#   ./cmake_build_examples/linux/custom-openmpi-build.sh clang
#
#   # AOCC clang/flang
#   AOCC_ROOT=/opt/AMD/aocc-compiler-5.1.0 \
#       ./cmake_build_examples/linux/custom-openmpi-build.sh aocc
#
#   # Override any variable
#   OMPI_VERSION=5.0.6 PREFIX=$HOME/opt/openmpi-clang21 \
#       ./cmake_build_examples/linux/custom-openmpi-build.sh clang
#
#   # Just download (skip build)
#   ./cmake_build_examples/linux/custom-openmpi-build.sh --download-only

set -euo pipefail

# ---------------------------------------------------------------------------
# Variables — override any of these via the environment
# ---------------------------------------------------------------------------
OMPI_VERSION="${OMPI_VERSION:-5.0.6}"
LLVM_VER="${LLVM_VER:--21}"
AOCC_ROOT="${AOCC_ROOT:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${SRC:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SCRATCH_DIR="${SCRATCH_DIR:-${SRC}/.scratch}"

JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
TOOLCHAIN=""
DOWNLOAD_ONLY=0
for _arg in "$@"; do
    case "$_arg" in
        clang|aocc)       TOOLCHAIN="$_arg" ;;
        --download-only)  DOWNLOAD_ONLY=1 ;;
        -h|--help)
            sed -n '2,/^[^#]/{ /^#/s/^# \?//p }' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $_arg" >&2
            echo "Usage: $0 [clang|aocc] [--download-only]" >&2
            exit 1
            ;;
    esac
done

if [[ ${DOWNLOAD_ONLY} -eq 0 && -z "${TOOLCHAIN}" ]]; then
    echo "Usage: $0 <clang|aocc> [--download-only]" >&2
    echo "       $0 --download-only" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Toolchain setup
# ---------------------------------------------------------------------------
case "${TOOLCHAIN}" in
    clang)
        LLVM_VER_NUM="${LLVM_VER#-}"
        LLVM_DIR="${LLVM_DIR:-/usr/lib/llvm${LLVM_VER_NUM}}"
        _CC="${LLVM_DIR}/bin/clang"
        _CXX="${LLVM_DIR}/bin/clang++"
        _FC="${LLVM_DIR}/bin/flang-new"
        # Fall back to unversioned names if the versioned binary doesn't exist
        [[ -x "${_CC}" ]]  || _CC="clang${LLVM_VER}"
        [[ -x "${_CXX}" ]] || _CXX="clang++${LLVM_VER}"
        [[ -x "${_FC}" ]]  || _FC="flang-new${LLVM_VER}"
        PREFIX="${PREFIX:-${SCRATCH_DIR}/openmpi-clang${LLVM_VER_NUM}}"
        _LABEL="LLVM clang${LLVM_VER}/flang-new${LLVM_VER}"
        ;;
    aocc)
        if [[ -z "${AOCC_ROOT}" ]]; then
            echo "ERROR: AOCC_ROOT must be set for the aocc toolchain." >&2
            exit 1
        fi
        _CC="${AOCC_ROOT}/bin/clang"
        _CXX="${AOCC_ROOT}/bin/clang++"
        _FC="${AOCC_ROOT}/bin/flang"
        PREFIX="${PREFIX:-${SCRATCH_DIR}/openmpi-aocc}"
        _LABEL="AOCC (${AOCC_ROOT})"
        ;;
    "")
        # --download-only, no toolchain needed
        PREFIX=""
        _LABEL=""
        ;;
esac

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
TARBALL="openmpi-${OMPI_VERSION}.tar.gz"
SRC_DIR="${SCRATCH_DIR}/openmpi-${OMPI_VERSION}"

mkdir -p "${SCRATCH_DIR}"

if [[ ! -f "${SCRATCH_DIR}/${TARBALL}" ]]; then
    echo "=== Downloading OpenMPI ${OMPI_VERSION} ==="
    # Try the official download site first, then GitHub
    _url="https://download.open-mpi.org/release/open-mpi/v${OMPI_VERSION%.*}/openmpi-${OMPI_VERSION}.tar.bz2"
    _gh_url="https://github.com/open-mpi/ompi/archive/refs/tags/v${OMPI_VERSION}.tar.gz"

    if command -v wget &>/dev/null; then
        _dl() { wget -q --show-progress -O "$2" "$1"; }
    elif command -v curl &>/dev/null; then
        _dl() { curl -fSL -o "$2" "$1"; }
    else
        echo "ERROR: neither wget nor curl found." >&2
        exit 1
    fi

    if _dl "${_url}" "${SCRATCH_DIR}/openmpi-${OMPI_VERSION}.tar.bz2" 2>/dev/null; then
        # bz2 tarball from official site — unpack to tar.gz-compatible name
        echo "Downloaded from official site (bz2)."
        TARBALL="openmpi-${OMPI_VERSION}.tar.bz2"
    elif _dl "${_gh_url}" "${SCRATCH_DIR}/${TARBALL}" 2>/dev/null; then
        echo "Downloaded from GitHub."
    else
        echo "ERROR: could not download OpenMPI ${OMPI_VERSION} from either source." >&2
        echo "  Tried: ${_url}" >&2
        echo "  Tried: ${_gh_url}" >&2
        exit 1
    fi
else
    echo "=== Using cached ${TARBALL} ==="
fi

# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------
if [[ ! -d "${SRC_DIR}" ]]; then
    echo "=== Extracting ==="
    tar -xf "${SCRATCH_DIR}/${TARBALL}" -C "${SCRATCH_DIR}"
    # GitHub tarballs extract to ompi-X.Y.Z, rename if needed
    if [[ ! -d "${SRC_DIR}" && -d "${SCRATCH_DIR}/ompi-${OMPI_VERSION}" ]]; then
        mv "${SCRATCH_DIR}/ompi-${OMPI_VERSION}" "${SRC_DIR}"
    fi
    if [[ ! -d "${SRC_DIR}" ]]; then
        echo "ERROR: expected source directory ${SRC_DIR} not found after extraction." >&2
        ls "${SCRATCH_DIR}"/openmpi-* "${SCRATCH_DIR}"/ompi-* 2>/dev/null
        exit 1
    fi
fi

if [[ ${DOWNLOAD_ONLY} -eq 1 ]]; then
    echo "=== Download complete: ${SRC_DIR} ==="
    exit 0
fi

# ---------------------------------------------------------------------------
# Verify compilers exist
# ---------------------------------------------------------------------------
echo "=== Building OpenMPI ${OMPI_VERSION} with ${_LABEL} ==="
echo "  CC  = ${_CC}"
echo "  CXX = ${_CXX}"
echo "  FC  = ${_FC}"
echo "  PREFIX = ${PREFIX}"

for _bin in "${_CC}" "${_CXX}" "${_FC}"; do
    if ! command -v "${_bin}" &>/dev/null; then
        echo "ERROR: compiler not found: ${_bin}" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------
BUILD_DIR="${SRC_DIR}/build-${TOOLCHAIN}"
mkdir -p "${BUILD_DIR}"
cd "${SRC_DIR}"

# Run autogen.sh if configure doesn't exist (GitHub source tarballs)
if [[ ! -x configure ]]; then
    if [[ -x autogen.pl ]]; then
        echo "=== Running autogen.pl ==="
        ./autogen.pl
    elif [[ -x autogen.sh ]]; then
        echo "=== Running autogen.sh ==="
        ./autogen.sh
    else
        echo "ERROR: no configure script and no autogen.pl/autogen.sh found." >&2
        exit 1
    fi
fi

echo "=== Configuring ==="
cd "${BUILD_DIR}"
"${SRC_DIR}/configure" \
    CC="${_CC}" CXX="${_CXX}" FC="${_FC}" \
    --prefix="${PREFIX}" \
    --enable-mpi-fortran=usempif08 \
    --disable-oshmem \
    --without-verbs

# ---------------------------------------------------------------------------
# Build & install
# ---------------------------------------------------------------------------
echo "=== Building (${JOBS} jobs) ==="
make -j"${JOBS}"

echo "=== Installing to ${PREFIX} ==="
make install

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo
echo "=== Verifying installation ==="
"${PREFIX}/bin/mpicc" --version 2>&1 | head -1
"${PREFIX}/bin/mpif90" --version 2>&1 | head -1

# Check that the Fortran wrapper uses the right compiler
_fc_wrapper=$("${PREFIX}/bin/mpif90" --showme:command 2>/dev/null || echo "unknown")
echo "mpif90 underlying compiler: ${_fc_wrapper}"

case "${TOOLCHAIN}" in
    clang)
        if echo "${_fc_wrapper}" | grep -qi "flang"; then
            echo "OK: mpif90 wraps flang-new"
        else
            echo "WARNING: mpif90 does not appear to wrap flang (got: ${_fc_wrapper})"
        fi
        ;;
    aocc)
        if echo "${_fc_wrapper}" | grep -qi "flang"; then
            echo "OK: mpif90 wraps AOCC flang"
        else
            echo "WARNING: mpif90 does not appear to wrap AOCC flang (got: ${_fc_wrapper})"
        fi
        ;;
esac

# Check that .mod files exist
_mod_count=$(find "${PREFIX}" -name '*.mod' 2>/dev/null | wc -l)
echo "Fortran .mod files installed: ${_mod_count}"

echo
echo "=== Done ==="
echo "Set this in your environment before running ELPA configure scripts:"
case "${TOOLCHAIN}" in
    clang) echo "  export OMPI_CLANG=\"${PREFIX}\"" ;;
    aocc)  echo "  export OMPI_AOCC=\"${PREFIX}\"" ;;
esac
