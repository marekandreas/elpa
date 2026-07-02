#Requires -Version 7
<#
.SYNOPSIS
    Run ELPA tests via CTest after a successful build (Windows).
.DESCRIPTION
    Sets up the runtime DLL search PATH and invokes CTest with Intel MPI.
    I_MPI_FABRICS=shm prevents an OFI initialization hang when running on
    a single Windows node without a proper IB/OFI fabric.

    Test labels:       "extended" is the only CTest label in the test suite.
    Default tests:     all tests NOT labelled "extended" (~612 for CPU, ~420 for GPU).
    Extended tests:    tests labelled "extended" (~188 for CPU, ~380 for GPU).
    Autotune tests:    generated but disabled by default (very long runtime).

    All test names end in _default or _extended.  Autotune tests are
    permanently excluded via --exclude-regex.
#>
param(
    [int]$Jobs    = 4,
    [int]$Timeout = 600,
    [switch]$ExtendedOnly,
    [switch]$AllIncludingExtended,
    # Override the build directory (used by validate.ps1 for multi-compiler runs)
    [string]$BuildDir = ""
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ===========================================================================
# USER-EDITABLE PATHS
# ===========================================================================
$BLD = "C:\path\to\build-ifx"   # or build-ifort
if ($BuildDir) { $BLD = $BuildDir }

$ONEAPI_ROOT = "C:\Program Files (x86)\Intel\oneAPI"
$MSVS_ROOT   = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
# MSVC toolset version — run: dir "$MSVS_ROOT\VC\Tools\MSVC" to find it
$MSVC_VER    = "14.42.34433"
$CUDA_ROOT   = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

# ===========================================================================
# RUNTIME DLL PATH
# ===========================================================================
$MPI_BIN    = "$ONEAPI_ROOT\mpi\latest\bin"
$MPI_FABRIC = "$ONEAPI_ROOT\mpi\latest\opt\mpi\libfabric\bin"
$MKL_BIN    = "$ONEAPI_ROOT\mkl\latest\bin"
$IFX_BIN    = "$ONEAPI_ROOT\compiler\latest\bin"
$MSVC_BIN   = "$MSVS_ROOT\VC\Tools\MSVC\$MSVC_VER\bin\HostX64\x64"
$CUDA_BIN   = "$CUDA_ROOT\bin"
# elpa_openmp.dll is in $BLD\src; test executables in $BLD\test need it on PATH
$ELPA_DLL   = "$BLD\src"

$env:PATH = "$MPI_BIN;$MPI_FABRIC;$MKL_BIN;$IFX_BIN;$CUDA_BIN;$ELPA_DLL;$MSVC_BIN;$env:PATH"

# ctest lives in the same directory as cmake.  Add it to PATH if not found;
# this handles VS-bundled cmake which is not on PATH in a plain PowerShell session.
if (-not (Get-Command ctest -ErrorAction SilentlyContinue)) {
    $vscmake = "$MSVS_ROOT\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
    if (Test-Path "$vscmake\ctest.exe") {
        $env:PATH = "$vscmake;$env:PATH"
    } else {
        Write-Error "ctest not found on PATH and not found at $vscmake.`nAdd cmake to PATH or run from a VS Developer PowerShell."
    }
}

# Intel MPI environment
$env:I_MPI_ROOT    = "$ONEAPI_ROOT\mpi\latest"
# Force shared-memory fabric on a single Windows node.
# Without this, mpiexec hangs in OFI provider initialisation when no
# high-speed fabric (InfiniBand, OmniPath) is present.
$env:I_MPI_FABRICS = "shm"

# ===========================================================================
# LABEL FILTER
# ===========================================================================
# "extended" is the only CTest label; default tests are unlabelled.
if ($ExtendedOnly) {
    $label_args = @("--label-regex", "extended")
} elseif ($AllIncludingExtended) {
    $label_args = @()   # run everything (excluding autotune via --exclude-regex)
} else {
    $label_args = @("--label-exclude", "extended")   # default: CPU validation suite
}

# ===========================================================================
# RUN
# ===========================================================================
Write-Host "=== Running ELPA tests ($Jobs parallel, timeout=${Timeout}s) ==="
$junitFile = Join-Path $BLD "test-results.xml"
Write-Host "=== Results will be written to: $junitFile ==="
Push-Location $BLD
ctest @label_args --exclude-regex "autotune" -j $Jobs --timeout $Timeout --output-on-failure --output-junit $junitFile
$ec = $LASTEXITCODE
Pop-Location
Write-Host "=== ctest exit code: $ec ==="
Write-Host "=== JUnit results: $junitFile ==="
