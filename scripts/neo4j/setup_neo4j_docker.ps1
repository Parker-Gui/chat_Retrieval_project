$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$composeFile = Join-Path $projectRoot "neo4j\docker-compose.yml"
$envFile = Join-Path $projectRoot "neo4j\.env"
$envExample = Join-Path $projectRoot "neo4j\.env.example"
$python = Join-Path $projectRoot "backend\venv311\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $envFile)) {
    Copy-Item -LiteralPath $envExample -Destination $envFile
}

docker compose --env-file $envFile -f $composeFile up -d --wait --wait-timeout 180
if ($LASTEXITCODE -ne 0) {
    throw "Neo4j Docker container failed to start"
}
& $python (Join-Path $projectRoot "scripts\neo4j\import_metaqa.py")
& $python (Join-Path $projectRoot "scripts\neo4j\verify_graph.py")
