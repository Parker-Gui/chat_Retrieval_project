$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$composeFile = Join-Path $projectRoot "neo4j\docker-compose.yml"
$envFile = Join-Path $projectRoot "neo4j\.env"

docker compose --env-file $envFile -f $composeFile up -d --wait --wait-timeout 180
if ($LASTEXITCODE -ne 0) {
    throw "Neo4j Docker container failed to start"
}
