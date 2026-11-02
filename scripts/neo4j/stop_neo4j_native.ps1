$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$pidFile = Join-Path $projectRoot "neo4j\native.pid"
$python = Join-Path $projectRoot "backend\venv311\Scripts\python.exe"
$envFile = Join-Path $projectRoot "neo4j\.env"

function Get-BoltListenerProcessId {
    $listeners = netstat -ano | Select-String "^\s*TCP\s+127\.0\.0\.1:7687\s+\S+\s+LISTENING\s+(\d+)"
    foreach ($listener in $listeners) {
        if ($listener.Matches.Count -gt 0) {
            return $listener.Matches[0].Groups[1].Value
        }
    }
    return $null
}

function Test-BoltReady {
    if (-not (Test-Path -LiteralPath $python)) {
        return $null
    }
    if (-not (Test-Path -LiteralPath $envFile)) {
        return $null
    }
    $passwordLine = Get-Content -LiteralPath $envFile |
        Where-Object { $_ -match "^NEO4J_PASSWORD=" } |
        Select-Object -First 1
    if (-not $passwordLine) {
        return $null
    }
    $neo4jPassword = $passwordLine.Substring("NEO4J_PASSWORD=".Length)
    & $python -c "from neo4j import GraphDatabase; import sys; d=GraphDatabase.driver('bolt://localhost:7687',auth=('neo4j',sys.argv[1])); d.verify_connectivity(); d.close()" $neo4jPassword 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

$neo4jPid = $null
if (Test-Path -LiteralPath $pidFile) {
    $recordedPid = (Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    # The recorded PID can go stale when the database was started outside this
    # script, so fall back to the process actually listening on the Bolt port.
    if ($recordedPid -and (Get-Process -Id $recordedPid -ErrorAction SilentlyContinue)) {
        $neo4jPid = $recordedPid
    }
}
if (-not $neo4jPid) {
    $neo4jPid = Get-BoltListenerProcessId
}

if (-not $neo4jPid) {
    Write-Host "Native Neo4j is not running."
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    exit 0
}

taskkill.exe /PID $neo4jPid /T /F | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Unable to stop Neo4j process tree $neo4jPid"
}

for ($attempt = 1; $attempt -le 20; $attempt++) {
    if (-not (Get-BoltListenerProcessId)) {
        break
    }
    Start-Sleep -Milliseconds 500
}

$stillListening = Get-BoltListenerProcessId
if ($stillListening) {
    throw "Neo4j is still listening on bolt://localhost:7687 (process $stillListening)"
}

Write-Host "Stopped native Neo4j process tree $neo4jPid"
Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue

$ready = Test-BoltReady
if ($ready -eq $true) {
    throw "Neo4j still answers Bolt connections after stopping."
}
