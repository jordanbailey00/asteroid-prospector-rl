$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host "==> $Label"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

Invoke-Step -Label "Black (check)" -Command { python -m black --check python tests tools }
Invoke-Step -Label "Ruff (lint)" -Command { python -m ruff check python tests tools }

$clangTargets = Get-ChildItem -Path engine_core -Recurse -File | Where-Object {
    $_.Extension -in ".c", ".h"
}
if ($clangTargets.Count -gt 0) {
    if (-not (Get-Command clang-format -ErrorAction SilentlyContinue)) {
        throw "clang-format was not found in PATH. Install clang-format to run C formatting checks."
    }

    Invoke-Step -Label "clang-format (check)" -Command {
        foreach ($target in $clangTargets) {
            clang-format --dry-run --Werror $target.FullName
        }
    }
}

Invoke-Step -Label "Pytest" -Command { python -m pytest -q }
