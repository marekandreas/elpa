#!/usr/bin/env bash
# test.sh — Run ELPA tests via CTest (Linux)
#
# Usage
#   cd <build-dir>
#   /path/to/test.sh                    # default tests only
#   /path/to/test.sh --extended-only    # extended tests only
#   /path/to/test.sh --all              # all non-autotune tests
#   /path/to/test.sh -j 8 --timeout 900

set -euo pipefail

JOBS=4
TIMEOUT=600
EXTENDED_ONLY=0
ALL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --extended-only) EXTENDED_ONLY=1; shift ;;
        --all)           ALL=1; shift ;;
        -j|--jobs)       JOBS="$2"; shift 2 ;;
        -j*)             JOBS="${1#-j}"; shift ;;
        --timeout)       TIMEOUT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Intel MPI: force shared-memory fabric for single-node validation.
# Without this, mpiexec hangs in OFI provider initialisation when
# no high-speed fabric is present.
if [[ -n "${I_MPI_ROOT:-}" ]]; then
    export I_MPI_FABRICS="shm"
fi

# Parallel ctest jobs already consume host-level concurrency. If the caller has
# not pinned OMP_NUM_THREADS explicitly, keep each test to one OpenMP thread to
# avoid severe oversubscription and timeout-heavy runs under -jN.
if [[ -z "${OMP_NUM_THREADS:-}" && "${JOBS}" -gt 1 ]]; then
    export OMP_NUM_THREADS=1
fi

# Label filter
if [[ ${EXTENDED_ONLY} -eq 1 ]]; then
    label_args=(--label-regex extended)
elif [[ ${ALL} -eq 1 ]]; then
    label_args=()
else
    label_args=(--label-exclude extended)
fi

echo "=== Running ELPA tests (${JOBS} parallel, timeout=${TIMEOUT}s) ==="
ctest "${label_args[@]}" --exclude-regex "autotune" \
    -j "${JOBS}" --timeout "${TIMEOUT}" --output-on-failure
echo "=== ctest exit code: $? ==="
