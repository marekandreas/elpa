#Requires -Version 7
<#
.SYNOPSIS
    Configure ELPA: clang-cl + ifort (classic Intel Fortran) + Intel MPI + MKL.
.DESCRIPTION
    Compatibility validation configuration using the classic ifort compiler
    (available in oneAPI 2023.x and earlier).  ifort was deprecated in 2024
    and removed in 2025; prefer configure-clang-ifx.ps1 for new builds.

    Prerequisites — same as configure-clang-ifx.ps1.

    Linker note (ifort / Intel)
    ---------------------------
    cmake -E vs_link_dll calls lld-link.exe DIRECTLY, bypassing ifort entirely,
    so /DEF: must be passed straight to the linker (not via -Qoption,link,).
    ElpaSelectiveExport.cmake branches on CMAKE_Fortran_COMPILER_ID to handle
    this automatically; no user action is required.
#>
param(
    [switch]$EnableCUDA,
    # "native" auto-detects the host GPU at configure time (project minimum).
    # For explicit architectures use a semicolon-separated SM list, e.g. "80;86;90".
    [string]$CudaArch = "native"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ===========================================================================
# PATHS — override any of these via environment variables
# ===========================================================================
$SRC            = if ($env:SRC)            { $env:SRC }            else { (Resolve-Path "$PSScriptRoot\..\..").Path }
$ONEAPI_ROOT    = if ($env:ONEAPI_ROOT)    { $env:ONEAPI_ROOT }    else { "C:\Program Files (x86)\Intel\oneAPI" }
$MSVS_ROOT      = if ($env:MSVS_ROOT)     { $env:MSVS_ROOT }     else { "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools" }
$MSVC_VER       = if ($env:MSVC_VER)       { $env:MSVC_VER }       else { "14.42.34433" }
$CUDA_ROOT      = if ($env:CUDA_ROOT)      { $env:CUDA_ROOT }      else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" }
$IFORT_VERSION  = if ($env:IFORT_VERSION)  { $env:IFORT_VERSION }  else { "2023.0.0" }
$WINSDK_VER     = if ($env:WINSDK_VER)     { $env:WINSDK_VER }     else { "10.0.22621.0" }
$BLD            = if ($env:BLD)            { $env:BLD }            else { "$SRC\build-ifort" }

# ===========================================================================
# DERIVED PATHS
# ===========================================================================
$LLVM_BIN = "$MSVS_ROOT\VC\Tools\Llvm\x64\bin"
$CC       = "$LLVM_BIN\clang-cl.exe".Replace('\', '/')
$CXX      = "$LLVM_BIN\clang-cl.exe".Replace('\', '/')
$FC       = "$ONEAPI_ROOT\compiler\$IFORT_VERSION\windows\bin\intel64\ifort.exe".Replace('\', '/')
$MKL_ROOT = "$ONEAPI_ROOT\mkl\latest"
$MPI_ROOT = "$ONEAPI_ROOT\mpi\latest"

if (-not (Test-Path $FC)) {
    Write-Error "ifort not found at $FC.  Check IFORT_VERSION=$IFORT_VERSION."
}

# ===========================================================================
# CONFIGURE-TIME ENVIRONMENT
# ===========================================================================
# Windows SDK bin provides rc.exe / mt.exe used by cmake -E vs_link_exe.
$WINSDK_BIN  = "C:\Program Files (x86)\Windows Kits\10\bin\$WINSDK_VER\x64"
$WINSDK_BASE = "C:\Program Files (x86)\Windows Kits\10\Lib\$WINSDK_VER"
$IFORT_LIB   = "$ONEAPI_ROOT\compiler\$IFORT_VERSION\windows\compiler\lib\intel64_win"
$env:PATH = "$MSVS_ROOT\VC\Tools\MSVC\$MSVC_VER\bin\HostX64\x64;$WINSDK_BIN;$env:PATH"
# LIB: link.exe (called by ifort) needs MSVC + Windows SDK import libraries
# during CMake's Fortran ABI try_compile checks.
$env:LIB = "$IFORT_LIB;$MSVS_ROOT\VC\Tools\MSVC\$MSVC_VER\lib\x64;$WINSDK_BASE\um\x64;$WINSDK_BASE\ucrt\x64"

# cmake and ninja may be bundled with Visual Studio rather than installed
# system-wide.  With VS 2017+ they live under Common7\IDE\CommonExtensions\Microsoft\CMake\
# and are NOT added to PATH by a plain PowerShell session.
# Run from a VS Developer PowerShell, or let the block below find them.
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    $vscmake = "$MSVS_ROOT\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
    $vsninja = "$MSVS_ROOT\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
    if (Test-Path "$vscmake\cmake.exe") {
        $env:PATH = "$vscmake;$vsninja;$env:PATH"
        Write-Host "cmake not on PATH — using VS-bundled cmake from $vscmake"
    } else {
        Write-Error "cmake not found on PATH and not found at $vscmake.`nAdd cmake to PATH or run from a VS Developer PowerShell."
    }
}

$USE_CCACHE = [bool](Get-Command ccache -ErrorAction SilentlyContinue)
if ($USE_CCACHE) {
    $env:CCACHE_SLOPPINESS = "include_file_ctime,include_file_mtime,pch_defines,system_headers,time_macros,modules"
}

# ===========================================================================
# CMAKE CONFIGURE
# ===========================================================================
$cmake_args = @(
    "-B", $BLD, "-S", $SRC,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_C_COMPILER=$CC",
    "-DCMAKE_CXX_COMPILER=$CXX",
    "-DCMAKE_Fortran_COMPILER=$FC",
    "-DELPA_MKL_ROOT=$MKL_ROOT",
    "-DELPA_MPI_ROOT=$MPI_ROOT",
    "-DELPA_OPENMP=ON",
    "-DBUILD_TESTING=ON",
    "-DELPA_TEST_EXTENDED=ON",
    "-DELPA_FRAMEWORK_ISA=avx2",
    "-DELPA_ENABLE_AVX512_KERNELS=ON",
    "-DELPA_DEFAULT_REAL_KERNEL=real_avx2_block2",
    "-DELPA_DEFAULT_COMPLEX_KERNEL=complex_avx2_block1"
)

if ($USE_CCACHE) {
    $cmake_args += @(
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_Fortran_COMPILER_LAUNCHER=ccache"
    )
}

if ($EnableCUDA) {
    if (-not (Test-Path "$CUDA_ROOT\bin\nvcc.exe")) {
        Write-Error "nvcc not found at $CUDA_ROOT"
    }
    $cmake_args += @(
        "-DCMAKE_CUDA_COMPILER=$($CUDA_ROOT.Replace('\','/'))/bin/nvcc.exe",
        "-DELPA_CUDA=ON",
        # "native" detects the host GPU automatically (CMake >= 3.24).
        # Pass a semicolon-separated SM list for explicit targets, e.g. "80;86;90".
        "-DELPA_CUDA_ARCHITECTURES=$CudaArch"
    )
    if ($USE_CCACHE) { $cmake_args += "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache" }
}

$cuda_label = if ($EnableCUDA) { " + CUDA ($CudaArch)" } else { "" }
Write-Host "=== ELPA configure: clang-cl + ifort $IFORT_VERSION + Intel MPI + MKL$cuda_label ==="
cmake @cmake_args
Write-Host "=== Configure exit code: $LASTEXITCODE ==="
