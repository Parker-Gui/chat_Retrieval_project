param(
    [int]$Limit = 0,
    [int[]]$Hop = @(),
    [switch]$RetryErrors
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "backend\venv311\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Python 3.11 environment not found at $python"
}

& (Join-Path $projectRoot "scripts\neo4j\start_neo4j.ps1")
& $python (Join-Path $projectRoot "scripts\evaluation\smoke_test_kg_retrieval.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$evaluationArgs = @(Join-Path $projectRoot "scripts\evaluation\evaluate_kg_retrieval.py")
foreach ($hopValue in $Hop) {
    $evaluationArgs += @("--hop", $hopValue)
}
if ($Limit -gt 0) {
    $evaluationArgs += @("--limit", $Limit)
}
if ($RetryErrors) {
    $evaluationArgs += "--retry-errors"
}
& $python @evaluationArgs
exit $LASTEXITCODE
