#!/usr/bin/env bash
# validate-all.sh — Full end-to-end ELPA validation across all x86_64 Linux
# configurations.  Configures, builds, and tests every compiler/math/MPI combo.
#
# Usage
#   # Edit HOST SETTINGS below, then:
#   cd /path/to/elpa-source
#   ./cmake_build_examples/linux/validate-all.sh
#   ./cmake_build_examples/linux/validate-all.sh gcc-openblas gcc-mkl-impi
#   ./cmake_build_examples/linux/validate-all.sh --exclude aocc --exclude gcc-ifort
#   ./cmake_build_examples/linux/validate-all.sh --build-only
#   ./cmake_build_examples/linux/validate-all.sh --test-only -j 16
#
# Prerequisites
#   - All toolchains installed (run check-prerequisites.sh first)
#   - Custom OpenMPI instances built for clang/flang and AOCC
#   - CUDA Toolkit available
#
# Each configuration runs in a subshell for full environment isolation.
# Logs: <SCRATCH_DIR>/logs/validate-<config>-{configure,build,test}.log

set -uo pipefail

# ===========================================================================
# HOST SETTINGS — override any of these via the environment
# ===========================================================================
# ELPA source root (auto-detected from script location)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${SRC:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

# Scratch directory for builds and logs (default: .scratch under source root)
SCRATCH_DIR="${SCRATCH_DIR:-${SRC}/.scratch}"

# LLVM version suffix
LLVM_VER="${LLVM_VER:--21}"

# Intel oneAPI
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
COMPILER_ROOT="${ONEAPI_ROOT}/compiler/latest"
IFORT_COMPILER_ROOT="${IFORT_COMPILER_ROOT:-${ONEAPI_ROOT}/compiler/2024.0}"
MKL_ROOT="${ONEAPI_ROOT}/mkl/latest"
MPI_ROOT="${ONEAPI_ROOT}/mpi/latest"
IOMP5_LIBRARY="${COMPILER_ROOT}/lib/libiomp5.so"

# CUDA Toolkit
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"

# GCC installation (leave empty to use system gcc)
GCC_HOME="${GCC_HOME:-}"

# AMD compilers and math libraries
AOCC_ROOT="${AOCC_ROOT:-}"
AOCL_ROOT="${AOCL_ROOT:-}"

# Custom OpenMPI installations
OMPI_HOME="${OMPI_HOME:-}"
OMPI_CLANG="${OMPI_CLANG:-}"
OMPI_AOCC="${OMPI_AOCC:-}"

# AOCL library path (LP64 variant; change to lib_ILP64 for ILP64)
if [[ -d "${AOCL_ROOT}/lib_LP64" ]]; then
    AOCL_LIB="${AOCL_ROOT}/lib_LP64"
elif [[ -d "${AOCL_ROOT}/lib" ]]; then
    AOCL_LIB="${AOCL_ROOT}/lib"
else
    AOCL_LIB=""
fi

# ===========================================================================
# OPTIONS
# ===========================================================================
JOBS=$(( $(nproc 2>/dev/null || echo 8) / 2 ))
[[ $JOBS -lt 1 ]] && JOBS=1
BUILD_JOBS=$(nproc 2>/dev/null || echo 8)
TIMEOUT=600
DO_CONFIGURE=1
DO_BUILD=1
DO_TEST=1
SELECTED_CONFIGS=()
EXCLUDED_CONFIGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)           SELECTED_CONFIGS+=("$2"); shift 2 ;;
        --config=*)         SELECTED_CONFIGS+=("${1#--config=}"); shift ;;
        --exclude)          EXCLUDED_CONFIGS+=("$2"); shift 2 ;;
        --exclude=*)        EXCLUDED_CONFIGS+=("${1#--exclude=}"); shift ;;
        --build-only)       DO_TEST=0; shift ;;
        --test-only)        DO_CONFIGURE=0; DO_BUILD=0; shift ;;
        --no-configure)     DO_CONFIGURE=0; shift ;;
        -j|--jobs)          JOBS="$2"; shift 2 ;;
        -j*)                JOBS="${1#-j}"; shift ;;
        --build-jobs)       BUILD_JOBS="$2"; shift 2 ;;
        --timeout)          TIMEOUT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/{ s/^# //; s/^#//; p }' "$0"
            exit 0
            ;;
        --*)  echo "Unknown option: $1" >&2; exit 1 ;;
        *)    SELECTED_CONFIGS+=("$1"); shift ;;
    esac
done

LOGDIR="${SCRATCH_DIR}/logs"
mkdir -p "$LOGDIR"

TEST_SH="${SRC}/cmake_build_examples/linux/test.sh"

# ===========================================================================
# CONFIGURATION TABLE
# ===========================================================================
# Each entry: NAME:CONFIGURE_SCRIPT
# The configure script is in cmake_build_examples/linux/.
declare -a CONFIG_TABLE=(
    "gcc-openblas:configure-gcc-gfortran-openblas-openmpi.sh"
    "gcc-mkl-impi:configure-gcc-gfortran-mkl-impi.sh"
    "gcc-mkl-openmpi:configure-gcc-gfortran-mkl-openmpi.sh"
    "gcc-blis:configure-gcc-gfortran-blis-openmpi.sh"
    "clang-openblas:configure-clang-flang-openblas-openmpi.sh"
    "clang-blis:configure-clang-flang-blis-openmpi.sh"
    "clang-mkl:configure-clang-flang-mkl-openmpi.sh"
    "clang-ifx:configure-clang-ifx-mkl-impi.sh"
    "gcc-ifx:configure-gcc-ifx-mkl-impi.sh"
    "gcc-ifort:configure-gcc-ifort-mkl-impi.sh"
    "icx-ifx:configure-icx-ifx-mkl-impi.sh"
    "aocc:configure-aocc-blis-openmpi.sh"
)

# Build the CONFIGS list
CONFIGS=()
for entry in "${CONFIG_TABLE[@]}"; do
    CONFIGS+=("${entry%%:*}")
done

# If configs were selected (--config or positional args), use only those
if [[ ${#SELECTED_CONFIGS[@]} -gt 0 ]]; then
    CONFIGS=("${SELECTED_CONFIGS[@]}")
fi

# Apply --exclude filter
if [[ ${#EXCLUDED_CONFIGS[@]} -gt 0 ]]; then
    _filtered=()
    for cfg in "${CONFIGS[@]}"; do
        _skip=0
        for excl in "${EXCLUDED_CONFIGS[@]}"; do
            [[ "$cfg" == "$excl" ]] && _skip=1 && break
        done
        [[ $_skip -eq 0 ]] && _filtered+=("$cfg")
    done
    CONFIGS=("${_filtered[@]}")
fi

# Lookup configure script for a config name
get_configure_script() {
    local name="$1"
    for entry in "${CONFIG_TABLE[@]}"; do
        if [[ "${entry%%:*}" == "$name" ]]; then
            echo "${entry#*:}"
            return 0
        fi
    done
    return 1
}

declare -A RESULTS

# ===========================================================================
# ENVIRONMENT SETUP FUNCTIONS
# ===========================================================================
# Each function sets up the environment for a specific configuration.
# Called inside a subshell — modifications are isolated.
# All functions also export the shared variables so that the configure
# scripts (which use ${VAR:-default} patterns) pick them up.

_export_common() {
    export SRC SCRATCH_DIR CUDA_ROOT GCC_HOME LLVM_VER
    export ONEAPI_ROOT IFORT_COMPILER_ROOT
    export OMPI_HOME OMPI_CLANG OMPI_AOCC
    export AOCC_ROOT AOCL_ROOT
}

setup_env() {
    local name="$1"
    _export_common
    case "$name" in
        gcc-openblas)
            export PATH="${CUDA_ROOT}/bin:${PATH}"
            ;;
        gcc-mkl-impi|gcc-ifx|clang-ifx)
            set +u; source "${COMPILER_ROOT}/env/vars.sh" > /dev/null 2>&1; set -u
            unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH INCLUDE OBJC_INCLUDE_PATH
            export PATH="${MPI_ROOT}/bin:${CUDA_ROOT}/bin:${PATH}"
            export LD_LIBRARY_PATH="${MPI_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export I_MPI_ROOT="${MPI_ROOT}"
            export I_MPI_FABRICS="shm"
            ;;
        gcc-mkl-openmpi)
            set +u; source "${COMPILER_ROOT}/env/vars.sh" > /dev/null 2>&1; set -u
            unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH INCLUDE OBJC_INCLUDE_PATH
            if [[ -n "${OMPI_HOME}" ]]; then
                export PATH="${OMPI_HOME}/bin:${PATH}"
                export LD_LIBRARY_PATH="${OMPI_HOME}/lib:${LD_LIBRARY_PATH:-}"
            fi
            export PATH="${CUDA_ROOT}/bin:${PATH}"
            export LD_LIBRARY_PATH="${MKL_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            ;;
        gcc-blis)
            if [[ -n "${OMPI_HOME}" ]]; then
                export PATH="${OMPI_HOME}/bin:${PATH}"
                export LD_LIBRARY_PATH="${OMPI_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            fi
            export PATH="${CUDA_ROOT}/bin:${PATH}"
            # AOCL ScaLAPACK depends on AOCC Flang runtime (libflang, libflangrti, libpgmath)
            [[ -n "${AOCL_LIB}" ]] && export LD_LIBRARY_PATH="${AOCL_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            [[ -n "${AOCC_ROOT}" ]] && export LD_LIBRARY_PATH="${AOCC_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            ;;
        clang-openblas)
            if [[ -n "${OMPI_CLANG}" ]]; then
                export PATH="${OMPI_CLANG}/bin:${PATH}"
                export LD_LIBRARY_PATH="${OMPI_CLANG}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            fi
            export PATH="${CUDA_ROOT}/bin:${PATH}"
            ;;
        clang-blis)
            if [[ -n "${OMPI_CLANG}" ]]; then
                export PATH="${OMPI_CLANG}/bin:${PATH}"
                export LD_LIBRARY_PATH="${OMPI_CLANG}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            fi
            export PATH="${CUDA_ROOT}/bin:${PATH}"
            # AOCL ScaLAPACK depends on AOCC Flang runtime
            [[ -n "${AOCL_LIB}" ]] && export LD_LIBRARY_PATH="${AOCL_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            [[ -n "${AOCC_ROOT}" ]] && export LD_LIBRARY_PATH="${AOCC_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            ;;
        clang-mkl)
            set +u; source "${COMPILER_ROOT}/env/vars.sh" > /dev/null 2>&1; set -u
            unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH INCLUDE OBJC_INCLUDE_PATH
            if [[ -n "${OMPI_CLANG}" ]]; then
                export PATH="${OMPI_CLANG}/bin:${PATH}"
                export LD_LIBRARY_PATH="${OMPI_CLANG}/lib:${LD_LIBRARY_PATH:-}"
            fi
            export PATH="${CUDA_ROOT}/bin:${PATH}"
            export LD_LIBRARY_PATH="${MKL_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            ;;
        gcc-ifort)
            set +u; source "${IFORT_COMPILER_ROOT}/env/vars.sh" > /dev/null 2>&1; set -u
            unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH INCLUDE OBJC_INCLUDE_PATH
            export PATH="${MPI_ROOT}/bin:${CUDA_ROOT}/bin:${PATH}"
            export LD_LIBRARY_PATH="${MPI_ROOT}/lib:${IFORT_COMPILER_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export I_MPI_ROOT="${MPI_ROOT}"
            export I_MPI_FABRICS="shm"
            ;;
        icx-ifx)
            set +u; source "${COMPILER_ROOT}/env/vars.sh" > /dev/null 2>&1; set -u
            unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH INCLUDE OBJC_INCLUDE_PATH
            export PATH="${MPI_ROOT}/bin:${CUDA_ROOT}/bin:${PATH}"
            export LD_LIBRARY_PATH="${MPI_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export I_MPI_ROOT="${MPI_ROOT}"
            export I_MPI_FABRICS="shm"
            ;;
        aocc)
            [[ -n "${AOCC_ROOT}" ]] && export PATH="${AOCC_ROOT}/bin:${PATH}"
            [[ -n "${OMPI_AOCC}" ]] && export PATH="${OMPI_AOCC}/bin:${PATH}"
            export LD_LIBRARY_PATH="${AOCC_ROOT:+${AOCC_ROOT}/lib:}${OMPI_AOCC:+${OMPI_AOCC}/lib:}${AOCL_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            ;;
        *)
            echo "Unknown config: $name" >&2
            return 1
            ;;
    esac
}

# ===========================================================================
# VALIDATION LOOP
# ===========================================================================

omp_check() {
    local bld="$1"
    local elpa_so
    elpa_so=$(find "$bld" -name 'libelpa*.so' ! -name '*.so.*' 2>/dev/null | head -1)
    if [[ -z "$elpa_so" ]]; then
        echo "  OMP: libelpa.so not found (build may have failed)"
        return
    fi
    local omp_libs
    omp_libs=$(ldd "$elpa_so" 2>/dev/null | grep -E 'libomp|libgomp|libiomp' | awk '{print $1}' | sort -u | tr '\n' ' ')
    local cnt
    cnt=$(echo "$omp_libs" | wc -w)
    if [[ "$cnt" -eq 1 ]]; then
        echo "  OMP OK: $omp_libs"
    elif [[ "$cnt" -eq 0 ]]; then
        echo "  OMP WARNING: no OpenMP runtime linked"
    else
        echo "  OMP WARNING: $cnt runtimes linked: $omp_libs"
    fi
}

validate_config() {
    local name="$1"
    local script
    script=$(get_configure_script "$name") || {
        echo "*** [$name] Unknown configuration ***"
        RESULTS[$name]="UNKNOWN_CONFIG"
        return 1
    }
    local cfg_script="${SCRIPT_DIR}/${script}"
    local bld="${SCRATCH_DIR}/build-${name}"

    echo
    echo "================================================================"
    echo "=== [$name] Validate ==="
    echo "================================================================"

    # --- Configure ---
    if [[ $DO_CONFIGURE -eq 1 ]]; then
        echo "--- [$name] Configuring ---"
        (
            setup_env "$name"
            rm -rf "$bld"
            mkdir -p "$bld"

            # Override SRC and BLD in the configure script's environment
            export SRC BLD="$bld"

            # The configure scripts use their own cmake_args array; we source
            # them in a subshell with the correct SRC/BLD.
            # Most scripts just run cmake with their cmake_args, so we
            # call the script directly.
            bash "$cfg_script"
        ) 2>&1 | tee "$LOGDIR/validate-${name}-configure.log" | tail -5
        local rc=${PIPESTATUS[0]}
        if [[ $rc -ne 0 ]]; then
            echo "*** [$name] CONFIGURE FAILED (exit $rc) ***"
            RESULTS[$name]="CONFIGURE_FAIL"
            return 1
        fi
    fi

    # --- Build ---
    if [[ $DO_BUILD -eq 1 ]]; then
        echo "--- [$name] Building (-j${BUILD_JOBS}) ---"
        (
            setup_env "$name"
            cmake --build "$bld" -j"${BUILD_JOBS}"
        ) 2>&1 | tee "$LOGDIR/validate-${name}-build.log" | tail -5
        local rc=${PIPESTATUS[0]}
        if [[ $rc -ne 0 ]]; then
            echo "*** [$name] BUILD FAILED (exit $rc) ***"
            RESULTS[$name]="BUILD_FAIL"
            return 1
        fi
        omp_check "$bld"
    fi

    # --- Test ---
    if [[ $DO_TEST -eq 1 ]]; then
        echo "--- [$name] Testing (--all -j${JOBS} --timeout ${TIMEOUT}) ---"
        (
            setup_env "$name"
            cd "$bld"
            export OMP_NUM_THREADS=1
            bash "$TEST_SH" --all -j "$JOBS" --timeout "$TIMEOUT"
        ) 2>&1 | tee "$LOGDIR/validate-${name}-test.log" | tail -20
        local rc=${PIPESTATUS[0]}
        local summary
        summary=$(grep -E 'tests passed|tests failed' "$LOGDIR/validate-${name}-test.log" | tail -1)
        if [[ $rc -ne 0 ]] || ! echo "$summary" | grep -q '0 tests failed'; then
            echo "*** [$name] TEST FAILED: $summary ***"
            RESULTS[$name]="TEST_FAIL: $summary"
            return 1
        fi
        RESULTS[$name]="PASS: $summary"
        echo "=== [$name] PASS ==="
        return 0
    fi

    RESULTS[$name]="OK"
    echo "=== [$name] OK ==="
}

# ===========================================================================
# MAIN
# ===========================================================================
echo "================================================================"
echo "=== ELPA validate-all.sh ==="
echo "=== Source: $SRC"
echo "=== Scratch: $SCRATCH_DIR"
echo "=== Configs: ${#CONFIGS[@]}"
echo "=== Build jobs: $BUILD_JOBS, Test jobs: $JOBS, Timeout: ${TIMEOUT}s"
echo "================================================================"

for cfg in "${CONFIGS[@]}"; do
    validate_config "$cfg"
done

echo
echo "================================================================"
echo "=== VALIDATION SUMMARY ==="
echo "================================================================"
total=0
pass=0
fail=0
for cfg in "${CONFIGS[@]}"; do
    total=$((total + 1))
    status="${RESULTS[$cfg]:-UNKNOWN}"
    if [[ "$status" == PASS* || "$status" == "OK" ]]; then
        echo "  [PASS] $cfg — $status"
        pass=$((pass + 1))
    else
        echo "  [FAIL] $cfg — $status"
        fail=$((fail + 1))
    fi
done
echo "================================================================"
echo "Total: $total  Pass: $pass  Fail: $fail"
echo "================================================================"

[[ $fail -eq 0 ]]
