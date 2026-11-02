param(
    [ValidateSet("pure_llm", "vector_rag", "kg_retrieval")]
    [string[]]$System = @("pure_llm", "vector_rag", "kg_retrieval"),
    [switch]$RetryErrors,
    [switch]$RetryWrong,
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "backend\venv311\Scripts\python.exe"
$evaluator = Join-Path $projectRoot "scripts\evaluation\evaluate_answer_systems.py"

if (-not (Test-Path $python)) {
    throw "Python 3.11 environment not found at $python"
}

if (-not $SkipSmoke) {
    & $python (Join-Path $projectRoot "scripts\evaluation\smoke_test_answer_pipelines.py")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if ($System -contains "vector_rag" -or $System -contains "kg_retrieval") {
        & (Join-Path $projectRoot "scripts\neo4j\start_neo4j.ps1")
        & $python (Join-Path $projectRoot "scripts\evaluation\smoke_test_kg_retrieval.py")
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

foreach ($name in $System) {
    $evaluationArgs = @($evaluator, "--system", $name)
    if ($RetryErrors) { $evaluationArgs += "--retry-errors" }
    if ($RetryWrong) { $evaluationArgs += "--retry-wrong" }
    & $python @evaluationArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

exit 0
