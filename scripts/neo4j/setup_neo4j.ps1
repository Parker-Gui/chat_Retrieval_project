$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $projectRoot "backend\venv311\Scripts\python.exe"

& (Join-Path $PSScriptRoot "install_neo4j_native.ps1")
& (Join-Path $PSScriptRoot "start_neo4j_native.ps1")

& $python (Join-Path $projectRoot "scripts\neo4j\import_metaqa.py")
if ($LASTEXITCODE -ne 0) {
    throw "MetaQA graph import failed"
}

& $python (Join-Path $projectRoot "scripts\neo4j\verify_graph.py")
if ($LASTEXITCODE -ne 0) {
    throw "MetaQA graph verification failed"
}

Write-Host "Native Neo4j setup completed. Browser: http://localhost:7474"
