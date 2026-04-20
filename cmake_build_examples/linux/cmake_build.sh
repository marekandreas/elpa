#!/usr/bin/env bash
# cmake_build.sh — Build ELPA after a configure step
#
# Usage
#   ./cmake_build.sh <build-dir>              # build
#   ./cmake_build.sh <build-dir> --install    # build + install

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <build-dir> [--install]" >&2
    exit 1
fi

BLD="$1"
shift

INSTALL=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --install) INSTALL=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "${BLD}/build.ninja" && ! -f "${BLD}/Makefile" ]]; then
    echo "ERROR: ${BLD} does not look like a configured build directory" >&2
    echo "Run a configure script first." >&2
    exit 1
fi

JOBS=$(nproc 2>/dev/null || echo 4)

echo "=== Building ELPA in ${BLD} (-j${JOBS}) ==="
cmake --build "${BLD}" -j"${JOBS}"
echo "=== Build exit code: $? ==="

if [[ ${INSTALL} -eq 1 ]]; then
    echo "=== Installing ELPA from ${BLD} ==="
    cmake --install "${BLD}"
    echo "=== Install exit code: $? ==="
fi
