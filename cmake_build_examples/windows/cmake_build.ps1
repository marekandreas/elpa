#Requires -Version 7
<#
.SYNOPSIS
    Build ELPA after a configure step (Windows).
.DESCRIPTION
    Runs cmake --build on the configured build directory.
    Optionally installs the built library.

.PARAMETER BuildDir
    Path to the configured build directory.

.PARAMETER Jobs
    Number of parallel build jobs.  Defaults to the logical CPU count.

.PARAMETER Install
    If set, also runs cmake --install after building.

.PARAMETER Verbose
    If set, passes --verbose to cmake --build.

.EXAMPLE
    .\build.ps1 -BuildDir C:\path\to\build-ifx
    .\build.ps1 -BuildDir C:\path\to\build-ifx -Install
    .\build.ps1 -BuildDir C:\path\to\build-ifx -Jobs 16 -Verbose
#>
param(
    [Parameter(Mandatory)]
    [string]$BuildDir,
    [int]   $Jobs    = [Environment]::ProcessorCount,
    [switch]$Install,
    [switch]$Verbose
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path "$BuildDir\build.ninja") -and -not (Test-Path "$BuildDir\Makefile")) {
    Write-Error "${BuildDir} does not look like a configured build directory.`nRun a configure script first."
}

# cmake may be VS-bundled and not on PATH in a standalone PowerShell session.
# If cmake_build.ps1 is run in the same session as a configure script,
# PATH is already set up correctly.  Otherwise cmake must be on PATH;
# run from a VS Developer PowerShell or add cmake's bin directory to PATH first.
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Error "cmake not found on PATH.`nRun from a VS Developer PowerShell or add cmake to PATH first."
}

$build_args = @("--build", $BuildDir, "-j", $Jobs)
if ($Verbose) { $build_args += "--verbose" }

Write-Host "=== Building ELPA in ${BuildDir} (-j${Jobs}) ==="
cmake @build_args
$ec = $LASTEXITCODE
Write-Host "=== Build exit code: $ec ==="
if ($ec -ne 0) { exit $ec }

if ($Install) {
    Write-Host "=== Installing ELPA from ${BuildDir} ==="
    cmake --install $BuildDir
    $ec = $LASTEXITCODE
    Write-Host "=== Install exit code: $ec ==="
    if ($ec -ne 0) { exit $ec }
}
