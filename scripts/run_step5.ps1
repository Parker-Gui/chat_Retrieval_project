param(
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "backend\venv311\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Python 3.11 environment not found at $python"
}

& (Join-Path $projectRoot "scripts\neo4j\start_neo4j.ps1")

$buildArgs = @(Join-Path $projectRoot "scripts\indexing\build_neo4j_vector_index.py")
if ($Rebuild) {
    $buildArgs += "--rebuild"
}
& $python @buildArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python (Join-Path $projectRoot "scripts\evaluation\evaluate_vector_retrieval.py")
exit $LASTEXITCODE
