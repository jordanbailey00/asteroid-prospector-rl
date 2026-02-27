$ErrorActionPreference = "Stop"

$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")

$gccCmd = Get-Command gcc -ErrorAction SilentlyContinue
if (-not $gccCmd) {
    throw "gcc not found in PATH. Install a MinGW toolchain (e.g. WinLibs) and retry."
}

$root = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $root "engine_core\build"
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null

$include = Join-Path $root "engine_core\include"
$coreSrc = Join-Path $root "engine_core\src\abp_core.c"
$rngSrc = Join-Path $root "engine_core\src\abp_rng.c"
$runnerSrc = Join-Path $root "engine_core\core_test_runner.c"

$dllOut = Join-Path $buildDir "abp_core.dll"
$runnerOut = Join-Path $buildDir "core_test_runner.exe"

Write-Host "==> Building abp_core.dll"
& gcc -std=c11 -O2 -Wall -Wextra -shared -I $include $coreSrc $rngSrc -o $dllOut
if ($LASTEXITCODE -ne 0) {
    throw "Failed to build abp_core.dll"
}

Write-Host "==> Building core_test_runner.exe"
& gcc -std=c11 -O2 -Wall -Wextra -I $include $runnerSrc $coreSrc $rngSrc -o $runnerOut
if ($LASTEXITCODE -ne 0) {
    throw "Failed to build core_test_runner.exe"
}

Write-Host "Built: $dllOut"
Write-Host "Built: $runnerOut"
