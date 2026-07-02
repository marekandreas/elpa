#!/usr/bin/env bash
# check-prerequisites.sh
#
# Audit the prerequisites for all ELPA CMake build configurations.
# Prints a FOUND/MISSING status for each required component and suggests
# install commands for missing items.
#
# All path variables honour the environment: set them before running this
# script to override the defaults.  With --export-env the script prints
# shell export statements for every resolved variable so they can be
# sourced by configure and validate scripts.
#
# Usage
#   ./cmake_build_examples/linux/check-prerequisites.sh
#   ./cmake_build_examples/linux/check-prerequisites.sh --json
#   ./cmake_build_examples/linux/check-prerequisites.sh --export-env
#   eval "$(./cmake_build_examples/linux/check-prerequisites.sh --export-env)"

set -uo pipefail

# ---------------------------------------------------------------------------
# Path variables — override any of these via the environment
# ---------------------------------------------------------------------------

# LLVM version suffix (e.g. "-21" for clang-21, flang-new-21, libomp-21-dev)
LLVM_VER="${LLVM_VER:--21}"

# Intel oneAPI root
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"

# Classic ifort lives in a separate compiler version directory
IFORT_COMPILER_ROOT="${IFORT_COMPILER_ROOT:-${ONEAPI_ROOT}/compiler/2024.0}"

# Custom OpenMPI installation built with clang/flang-new
OMPI_CLANG="${OMPI_CLANG:-}"

# Custom OpenMPI installation built with AOCC compilers
OMPI_AOCC="${OMPI_AOCC:-}"

# AMD compilers and math libraries
AOCC_ROOT="${AOCC_ROOT:-}"
AOCL_ROOT="${AOCL_ROOT:-}"

# CUDA toolkit root
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"

# GCC installation (leave empty to use system gcc)
GCC_HOME="${GCC_HOME:-}"

# Custom system OpenMPI (leave empty to use system mpicc on PATH)
OMPI_HOME="${OMPI_HOME:-}"

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

JSON=0
EXPORT_ENV=0
for _arg in "$@"; do
    case "$_arg" in
        --json)       JSON=1 ;;
        --export-env) EXPORT_ENV=1 ;;
    esac
done

# When --export-env, send diagnostics to stderr so only exports reach stdout
if [[ ${EXPORT_ENV} -eq 1 ]]; then
    exec 3>&1 1>&2
fi

PASS=0
FAIL=0
SKIP=0
declare -a MISSING_ITEMS=()
declare -a SKIPPED_ITEMS=()

_ok()    { echo "  [FOUND]   $*"; ((PASS++)) || true; }
_fail()  { echo "  [MISSING] $*"; ((FAIL++)) || true; MISSING_ITEMS+=("$*"); }
_skip()  { echo "  [SKIPPED] $*"; ((SKIP++)) || true; SKIPPED_ITEMS+=("$*"); }
_info()  { echo "            $*"; }
_head()  { echo; echo "=== $* ==="; }

check_cmd() {
    local cmd="$1"
    local desc="${2:-$cmd}"
    if command -v "${cmd}" &>/dev/null; then
        local ver
        ver=$("${cmd}" --version 2>&1 | head -1) || true
        _ok "${desc}: ${ver}"
    else
        _fail "${cmd} — ${desc} not found in PATH"
    fi
}

check_file() {
    local path="$1"
    local desc="$2"
    if [[ -f "${path}" ]]; then
        _ok "${desc}: ${path}"
    else
        _fail "${desc}: ${path} — not found"
    fi
}

check_lib() {
    local name="$1"
    local desc="$2"
    if ldconfig -p 2>/dev/null | grep -q "${name}"; then
        local path
        path=$(ldconfig -p 2>/dev/null | grep "${name}" | awk '{print $NF}' | head -1)
        _ok "${desc}: ${path}"
    else
        _fail "${desc}: lib${name} not found via ldconfig"
    fi
}

# ---------------------------------------------------------------------------
echo "ELPA Build Prerequisites Check"
echo "================================"
echo "Date: $(date)"
echo

# ---------------------------------------------------------------------------
_head "Build tools"
check_cmd cmake "CMake (>=3.24 required)"
# version check
if command -v cmake &>/dev/null; then
    CMAKE_VER=$(cmake --version 2>&1 | head -1 | grep -oP '\d+\.\d+\.\d+')
    IFS='.' read -r maj min _ <<< "${CMAKE_VER}"
    if (( maj > 3 || ( maj == 3 && min >= 24 ) )); then
        _info "CMake ${CMAKE_VER} — OK (>=3.24)"
    else
        _fail "CMake ${CMAKE_VER} is too old — need >=3.24"
    fi
fi
check_cmd ninja "Ninja build system"
check_cmd make "GNU Make (fallback)"

# ---------------------------------------------------------------------------
_head "GCC toolchain (configs 1, 2, 3, 7)"
check_cmd gcc  "gcc C compiler"
check_cmd g++  "g++ C++ compiler"
check_cmd gfortran "gfortran Fortran compiler"
check_lib "gomp" "libgomp (GCC OpenMP runtime)"

# ---------------------------------------------------------------------------
_head "LLVM/Clang toolchain (configs 4, 5, 8)"
check_cmd "clang${LLVM_VER}"      "clang (LLVM${LLVM_VER}) C compiler"
check_cmd "clang++${LLVM_VER}"    "clang++ (LLVM${LLVM_VER}) C++ compiler"
check_cmd "flang-new${LLVM_VER}"  "flang-new (LLVM${LLVM_VER}) Fortran compiler"
# libomp header — use find to handle version-subdirectory variations
LLVM_VER_NUM="${LLVM_VER#-}"
if OMP_H=$(find "/usr/lib/llvm${LLVM_VER}" -name "omp.h" 2>/dev/null | head -1) && [[ -n "${OMP_H}" ]]; then
    _ok "omp.h (libomp-dev${LLVM_VER}): ${OMP_H}"
elif [[ -f "/usr/include/omp.h" ]]; then
    _ok "omp.h (system libomp-dev): /usr/include/omp.h"
else
    _fail "omp.h not found — install libomp${LLVM_VER}-dev"
    _info "  sudo apt install libomp${LLVM_VER}-dev"
fi
# Use grep -w to avoid matching libseccomp etc.
if ldconfig -p 2>/dev/null | grep -qw 'libomp.so'; then
    OMP_PATH=$(ldconfig -p 2>/dev/null | grep -w 'libomp.so' | awk '{print $NF}' | head -1)
    _ok "libomp (LLVM OpenMP runtime): ${OMP_PATH}"
else
    _fail "libomp.so not found via ldconfig — install libomp${LLVM_VER}-dev"
fi

# ---------------------------------------------------------------------------
_head "Intel oneAPI toolchain"
check_file "${ONEAPI_ROOT}/compiler/latest/bin/ifx" "ifx (Intel Fortran, oneAPI)"
check_file "${ONEAPI_ROOT}/compiler/latest/bin/icx" "icx (Intel C compiler, oneAPI)"
check_file "${ONEAPI_ROOT}/compiler/latest/bin/icpx" "icpx (Intel C++ compiler, oneAPI)"
check_file "${ONEAPI_ROOT}/compiler/latest/lib/libiomp5.so" "libiomp5.so (Intel OpenMP runtime)"
check_file "${ONEAPI_ROOT}/compiler/latest/env/vars.sh" "oneAPI compiler vars.sh"

# ---------------------------------------------------------------------------
_head "Intel MKL"
check_file "${ONEAPI_ROOT}/mkl/latest/lib/libmkl_core.so" "libmkl_core.so"
check_file "${ONEAPI_ROOT}/mkl/latest/lib/libmkl_intel_lp64.so" "libmkl_intel_lp64.so"
check_file "${ONEAPI_ROOT}/mkl/latest/lib/libmkl_intel_thread.so" "libmkl_intel_thread.so (intel_thread)"
check_file "${ONEAPI_ROOT}/mkl/latest/lib/libmkl_gnu_thread.so" "libmkl_gnu_thread.so (gnu_thread)"
check_file "${ONEAPI_ROOT}/mkl/latest/lib/cmake/mkl/MKLConfig.cmake" "MKLConfig.cmake"

# ---------------------------------------------------------------------------
_head "Intel MPI"
check_file "${ONEAPI_ROOT}/mpi/latest/bin/mpicc"   "mpicc (Intel MPI)"
check_file "${ONEAPI_ROOT}/mpi/latest/bin/mpiicx"  "mpiicx (Intel MPI + icx)"
check_file "${ONEAPI_ROOT}/mpi/latest/bin/mpiifx"  "mpiifx (Intel MPI + ifx)"
check_file "${ONEAPI_ROOT}/mpi/latest/bin/mpigcc"  "mpigcc (Intel MPI + gcc)"
check_file "${ONEAPI_ROOT}/mpi/latest/include/mpi.h" "mpi.h (Intel MPI)"

# ---------------------------------------------------------------------------
_head "Classic ifort (oneAPI <= 2024)"
if [[ -d "${IFORT_COMPILER_ROOT}" ]]; then
    check_file "${IFORT_COMPILER_ROOT}/bin/ifort" "ifort (classic Intel Fortran)"
else
    _skip "classic ifort — IFORT_COMPILER_ROOT=${IFORT_COMPILER_ROOT} not found"
fi

# ---------------------------------------------------------------------------
_head "OpenMPI — system build"
check_cmd mpicc "mpicc (system OpenMPI, for gcc configs)"
if command -v mpicc &>/dev/null; then
    MPICC_VER=$(mpicc --version 2>&1 | head -1) || true
    _info "mpicc version: ${MPICC_VER}"
fi

# ---------------------------------------------------------------------------
_head "OpenMPI — clang/flang-new build"
if [[ -n "${OMPI_CLANG}" ]]; then
    check_file "${OMPI_CLANG}/bin/mpicc" "mpicc (clang-built OpenMPI)"
    check_file "${OMPI_CLANG}/bin/mpif90" "mpif90 (clang-built OpenMPI)"
    if [[ -f "${OMPI_CLANG}/bin/mpif90" ]]; then
        _wdata="${OMPI_CLANG}/share/openmpi/mpif90-wrapper-data.txt"
        if [[ -f "${_wdata}" ]]; then
            FC_USED=$(grep -m1 '^compiler=' "${_wdata}" | cut -d= -f2-)
        else
            FC_USED=$("${OMPI_CLANG}/bin/mpif90" --showme:command 2>/dev/null || echo "unknown")
        fi
        _info "mpif90 underlying compiler: ${FC_USED}"
        if echo "${FC_USED}" | grep -qi "flang"; then
            _ok "Confirmed: mpif90 uses flang"
        else
            _fail "mpif90 does NOT use flang (got: ${FC_USED}) — modules will be ABI-incompatible"
        fi
    fi
else
    _skip "clang-built OpenMPI — set OMPI_CLANG to enable"
fi

# ---------------------------------------------------------------------------
_head "OpenMPI — AOCC build"
if [[ -n "${OMPI_AOCC}" ]]; then
    check_file "${OMPI_AOCC}/bin/mpicc" "mpicc (AOCC-built OpenMPI)"
    check_file "${OMPI_AOCC}/bin/mpif90" "mpif90 (AOCC-built OpenMPI)"
else
    _skip "AOCC-built OpenMPI — set OMPI_AOCC to enable"
fi

# ---------------------------------------------------------------------------
_head "OpenBLAS + ScaLAPACK"
check_lib "openblas"  "libopenblas (OpenBLAS)"
if ldconfig -p 2>/dev/null | grep -qE 'scalapack.*openmpi|scalapack_openmpi'; then
    _ok "libscalapack-openmpi found"
elif ldconfig -p 2>/dev/null | grep -q 'scalapack'; then
    SCAL=$(ldconfig -p 2>/dev/null | grep scalapack | head -1)
    _ok "libscalapack found: ${SCAL}"
else
    _fail "libscalapack not found — install libscalapack-openmpi-dev or libscalapack-mpi-dev"
    _info "  sudo apt install libscalapack-openmpi-dev"
fi

# ---------------------------------------------------------------------------
_head "AOCL"
if [[ -n "${AOCL_ROOT}" ]]; then
    _aocl_lib="${AOCL_ROOT}/lib_LP64"
    [[ -d "${_aocl_lib}" ]] || _aocl_lib="${AOCL_ROOT}/lib"
    check_file "${_aocl_lib}/libblis-mt.so" "libblis-mt.so (AOCL BLAS)"
    check_file "${_aocl_lib}/libflame.so"   "libflame.so (AOCL LAPACK)"
    check_file "${_aocl_lib}/libscalapack.so" "libscalapack.so (AOCL ScaLAPACK)"
else
    _skip "AOCL (AMD Optimizing CPU Libraries) — set AOCL_ROOT to enable"
fi

# ---------------------------------------------------------------------------
_head "AOCC"
if [[ -n "${AOCC_ROOT}" ]]; then
    check_file "${AOCC_ROOT}/bin/clang" "clang (AOCC)"
    check_file "${AOCC_ROOT}/bin/flang" "flang (AOCC classic)"
else
    _skip "AOCC (AMD Optimizing C/C++ Compiler) — set AOCC_ROOT to enable"
fi

# ---------------------------------------------------------------------------
_head "CUDA"
check_file "${CUDA_ROOT}/bin/nvcc" "nvcc (NVIDIA CUDA compiler)"
if [[ -f "${CUDA_ROOT}/bin/nvcc" ]]; then
    NVCC_VER=$("${CUDA_ROOT}/bin/nvcc" --version 2>&1 | grep release | grep -oP 'V\S+') || true
    _info "nvcc ${NVCC_VER}"
fi
check_lib "cudart" "libcudart (CUDA runtime)"

# ---------------------------------------------------------------------------
echo
echo "================================"
echo "Summary: ${PASS} FOUND, ${FAIL} MISSING, ${SKIP} SKIPPED"
if [[ ${FAIL} -gt 0 ]]; then
    echo
    echo "Missing items:"
    for item in "${MISSING_ITEMS[@]}"; do
        echo "  - ${item}"
    done
    echo
    echo "Suggested installs (Ubuntu/Debian):"
    echo "  sudo apt install gcc g++ gfortran cmake ninja-build"
    echo "  sudo apt install clang${LLVM_VER} flang${LLVM_VER} libomp${LLVM_VER}-dev"
    echo "  sudo apt install libopenblas-dev libscalapack-openmpi-dev"
    echo "  Intel oneAPI: https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html"
    echo "  AOCL:         https://www.amd.com/en/developer/aocl/dense.html"
    echo "  CUDA:         https://developer.nvidia.com/cuda-downloads"
fi

if [[ ${SKIP} -gt 0 ]]; then
    echo
    echo "Skipped items (set the variable to enable the check):"
    for item in "${SKIPPED_ITEMS[@]}"; do
        echo "  - ${item}"
    done
fi

if [[ ${JSON} -eq 1 ]]; then
    echo
    echo "=== JSON ==="
    echo "{ \"pass\": ${PASS}, \"fail\": ${FAIL}, \"skip\": ${SKIP} }"
fi

# ---------------------------------------------------------------------------
# --export-env: emit shell export statements for all resolved variables
# ---------------------------------------------------------------------------
if [[ ${EXPORT_ENV} -eq 1 ]]; then
    exec 1>&3 3>&-
    echo "# === ELPA build environment (source this output) ==="
    echo "export LLVM_VER=\"${LLVM_VER}\""
    echo "export ONEAPI_ROOT=\"${ONEAPI_ROOT}\""
    echo "export IFORT_COMPILER_ROOT=\"${IFORT_COMPILER_ROOT}\""
    echo "export CUDA_ROOT=\"${CUDA_ROOT}\""
    echo "export GCC_HOME=\"${GCC_HOME}\""
    echo "export OMPI_HOME=\"${OMPI_HOME}\""
    echo "export OMPI_CLANG=\"${OMPI_CLANG}\""
    echo "export OMPI_AOCC=\"${OMPI_AOCC}\""
    echo "export AOCC_ROOT=\"${AOCC_ROOT}\""
    echo "export AOCL_ROOT=\"${AOCL_ROOT}\""
fi

[[ ${FAIL} -eq 0 ]]
