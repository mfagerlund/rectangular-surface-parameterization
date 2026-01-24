# build_libqex.ps1 - Build libQEx for Windows x64
#
# Prerequisites:
#   - CMake (https://cmake.org/download/)
#   - Visual Studio 2019+ with C++ workload
#   - Internet connection (to download dependencies)
#
# Usage:
#   .\scripts\build_libqex.ps1
#
# Environment variables:
#   LIBQEX_BUILD_DIR - Override build directory (default: %TEMP%\libqex_build)
#
# Output:
#   bin/qex_extract.exe, bin/OpenMesh*.dll

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
# Build directory - defaults to temp, can be overridden with environment variable
$BuildDir = if ($env:LIBQEX_BUILD_DIR) { $env:LIBQEX_BUILD_DIR } else { Join-Path $env:TEMP "libqex_build" }
$BinDir = Join-Path $RepoRoot "bin"
$DepsDir = $BuildDir

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "libQEx Build Script for Windows x64" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# -----------------------------------------------------------------------------
# Check prerequisites
# -----------------------------------------------------------------------------

Write-Host "Checking prerequisites..." -ForegroundColor Yellow

# Check CMake
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    Write-Host "ERROR: CMake not found. Install from https://cmake.org/download/" -ForegroundColor Red
    Write-Host "Make sure to add CMake to PATH during installation." -ForegroundColor Red
    exit 1
}
Write-Host "  CMake: $($cmake.Source)" -ForegroundColor Green

# Check Visual Studio / MSBuild
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -property installationPath
    if ($vsPath) {
        Write-Host "  Visual Studio: $vsPath" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Visual Studio not found. Install VS 2019+ with C++ workload." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "WARNING: vswhere not found, assuming VS is installed..." -ForegroundColor Yellow
}

# -----------------------------------------------------------------------------
# Create directories
# -----------------------------------------------------------------------------

Write-Host ""
Write-Host "Creating build directories..." -ForegroundColor Yellow

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $DepsDir | Out-Null
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

# -----------------------------------------------------------------------------
# Clone OpenMesh (source)
# -----------------------------------------------------------------------------

$OpenMeshDir = Join-Path $DepsDir "OpenMesh-src"

if (-not (Test-Path $OpenMeshDir)) {
    Write-Host ""
    Write-Host "Cloning OpenMesh from GitLab..." -ForegroundColor Yellow

    git clone --depth 1 https://gitlab.vci.rwth-aachen.de:9000/OpenMesh/OpenMesh.git $OpenMeshDir

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to clone OpenMesh" -ForegroundColor Red
        exit 1
    }
    Write-Host "  OpenMesh cloned: $OpenMeshDir" -ForegroundColor Green
} else {
    Write-Host "OpenMesh already present: $OpenMeshDir" -ForegroundColor Green
}

# Build OpenMesh
$OpenMeshBuild = Join-Path $OpenMeshDir "build"
$OpenMeshInstall = Join-Path $DepsDir "OpenMesh-install"

if (-not (Test-Path "$OpenMeshInstall\lib\OpenMeshCore.lib")) {
    Write-Host ""
    Write-Host "Building OpenMesh..." -ForegroundColor Yellow

    New-Item -ItemType Directory -Force -Path $OpenMeshBuild | Out-Null
    Push-Location $OpenMeshBuild

    cmake .. -G "Visual Studio 17 2022" -A x64 `
        -DCMAKE_BUILD_TYPE=Release `
        -DCMAKE_INSTALL_PREFIX="$OpenMeshInstall" `
        -DBUILD_APPS=OFF

    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: VS2022 failed, trying VS2019..." -ForegroundColor Yellow
        cmake .. -G "Visual Studio 16 2019" -A x64 `
            -DCMAKE_BUILD_TYPE=Release `
            -DCMAKE_INSTALL_PREFIX="$OpenMeshInstall" `
            -DBUILD_APPS=OFF
    }

    cmake --build . --config Release
    cmake --install . --config Release

    Pop-Location
    Write-Host "  OpenMesh built and installed: $OpenMeshInstall" -ForegroundColor Green
} else {
    Write-Host "OpenMesh already built: $OpenMeshInstall" -ForegroundColor Green
}

# -----------------------------------------------------------------------------
# Clone libQEx
# -----------------------------------------------------------------------------

$LibQExDir = Join-Path $DepsDir "libQEx"

if (-not (Test-Path $LibQExDir)) {
    Write-Host ""
    Write-Host "Cloning libQEx..." -ForegroundColor Yellow

    git clone https://github.com/hcebke/libQEx.git $LibQExDir

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to clone libQEx" -ForegroundColor Red
        exit 1
    }
    Write-Host "  libQEx cloned: $LibQExDir" -ForegroundColor Green
} else {
    Write-Host "libQEx already present: $LibQExDir" -ForegroundColor Green
}

# -----------------------------------------------------------------------------
# Configure with CMake
# -----------------------------------------------------------------------------

Write-Host ""
Write-Host "Configuring with CMake..." -ForegroundColor Yellow

$LibQExBuild = Join-Path $LibQExDir "build"
New-Item -ItemType Directory -Force -Path $LibQExBuild | Out-Null

Push-Location $LibQExBuild

# Set OpenMesh paths (from built install)
$OpenMeshInclude = Join-Path $OpenMeshInstall "include"
$OpenMeshLib = Join-Path $OpenMeshInstall "lib"

cmake .. -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DOpenMesh_INCLUDE_DIR="$OpenMeshInclude" `
    -DOpenMesh_LIBRARY="$OpenMeshLib\OpenMeshCore.lib" `
    -DBUILD_UNIT_TESTS=OFF

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: VS2022 failed, trying VS2019..." -ForegroundColor Yellow
    cmake .. -G "Visual Studio 16 2019" -A x64 `
        -DCMAKE_BUILD_TYPE=Release `
        -DOpenMesh_INCLUDE_DIR="$OpenMeshInclude" `
        -DOpenMesh_LIBRARY="$OpenMeshLib\OpenMeshCore.lib" `
        -DBUILD_UNIT_TESTS=OFF

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: CMake configuration failed" -ForegroundColor Red
        Pop-Location
        exit 1
    }
}

Write-Host "  CMake configuration complete" -ForegroundColor Green

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------

Write-Host ""
Write-Host "Building libQEx (Release)..." -ForegroundColor Yellow

cmake --build . --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed" -ForegroundColor Red
    Pop-Location
    exit 1
}

Write-Host "  Build complete" -ForegroundColor Green

Pop-Location

# -----------------------------------------------------------------------------
# Copy binaries to bin/
# -----------------------------------------------------------------------------

Write-Host ""
Write-Host "Copying binaries to bin/..." -ForegroundColor Yellow

# Find built files
$ReleaseDir = Join-Path $LibQExBuild "Release"
$DemoDir = Join-Path $LibQExBuild "demo\Release"

# Copy libQEx library
$qexDll = Get-ChildItem -Path $LibQExBuild -Recurse -Filter "*.dll" | Select-Object -First 1
if ($qexDll) {
    Copy-Item $qexDll.FullName -Destination $BinDir -Force
    Write-Host "  Copied: $($qexDll.Name)" -ForegroundColor Green
}

# Copy demo executable if exists
$qexExe = Get-ChildItem -Path $LibQExBuild -Recurse -Filter "*.exe" | Select-Object -First 1
if ($qexExe) {
    Copy-Item $qexExe.FullName -Destination $BinDir -Force
    Write-Host "  Copied: $($qexExe.Name)" -ForegroundColor Green
}

# Copy OpenMesh DLLs from install location
$OpenMeshBin = Join-Path $OpenMeshInstall "bin"
$OpenMeshLibDir = Join-Path $OpenMeshInstall "lib"

# Try bin first, then lib (depends on build type)
foreach ($searchDir in @($OpenMeshBin, $OpenMeshLibDir)) {
    if (Test-Path $searchDir) {
        Get-ChildItem -Path $searchDir -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item $_.FullName -Destination $BinDir -Force
            Write-Host "  Copied: $($_.Name)" -ForegroundColor Green
        }
    }
}

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Build complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Binaries are in: $BinDir" -ForegroundColor White
Write-Host ""
Get-ChildItem -Path $BinDir -Filter "*.dll" | ForEach-Object { Write-Host "  $_" }
Get-ChildItem -Path $BinDir -Filter "*.exe" | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Test: bin\qex_extract.exe <input.obj> <output.obj>" -ForegroundColor White
Write-Host "  2. Note: Binaries are gitignored. For releases, use GitHub Actions workflow." -ForegroundColor White
