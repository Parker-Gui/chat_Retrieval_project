param(
    [string]$JavaHome = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$neo4jHome = Join-Path $projectRoot "tools\neo4j\neo4j-community-5.26.32"
$python = Join-Path $projectRoot "backend\venv311\Scripts\python.exe"
$pidFile = Join-Path $projectRoot "neo4j\native.pid"
$envFile = Join-Path $projectRoot "neo4j\.env"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Neo4j environment file not found: $envFile"
}
$passwordLine = Get-Content -LiteralPath $envFile |
    Where-Object { $_ -match "^NEO4J_PASSWORD=" } |
    Select-Object -First 1
if (-not $passwordLine) {
    throw "NEO4J_PASSWORD is missing from $envFile"
}
$neo4jPassword = $passwordLine.Substring("NEO4J_PASSWORD=".Length)

# A successful Bolt connection is the authoritative readiness signal. Process
# lookups are unreliable here because the database may be owned by another
# session, so the script never depends on Get-Process to decide whether to start.
function Test-BoltReady {
    & $python -c "from neo4j import GraphDatabase; import sys; d=GraphDatabase.driver('bolt://localhost:7687',auth=('neo4j',sys.argv[1])); d.verify_connectivity(); d.close()" $neo4jPassword 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Get-BoltListenerProcessId {
    $listeners = netstat -ano | Select-String "^\s*TCP\s+127\.0\.0\.1:7687\s+\S+\s+LISTENING\s+(\d+)"
    foreach ($listener in $listeners) {
        if ($listener.Matches.Count -gt 0) {
            return $listener.Matches[0].Groups[1].Value
        }
    }
    return $null
}

if (Test-BoltReady) {
    $listenerPid = Get-BoltListenerProcessId
    if ($listenerPid) {
        try {
            Set-Content -LiteralPath $pidFile -Value $listenerPid -ErrorAction Stop
        } catch {
            Write-Warning "Could not update $pidFile; Neo4j is already running with process $listenerPid."
        }
    }
    Write-Host "Neo4j is already running and accepting Bolt connections on bolt://localhost:7687"
    Write-Host "Browser: http://localhost:7474"
    exit 0
}

if ($JavaHome) {
    $env:JAVA_HOME = $JavaHome
}
if (-not $env:JAVA_HOME) {
    $candidate = Get-ChildItem -LiteralPath (Join-Path $env:USERPROFILE ".jdks") -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "^(jbr|jdk)-17" } |
        Select-Object -First 1
    if ($candidate) {
        $env:JAVA_HOME = $candidate.FullName
    }
}
if (-not $env:JAVA_HOME -or -not (Test-Path -LiteralPath (Join-Path $env:JAVA_HOME "bin\java.exe"))) {
    throw "JDK 17 was not found. Pass -JavaHome or set JAVA_HOME."
}
if (-not (Test-Path -LiteralPath $neo4jHome)) {
    throw "Neo4j is not installed. Run scripts/neo4j/install_neo4j_native.ps1 first."
}

# Start-Process inherits the current environment, so JAVA_HOME above is enough.
# The -Environment parameter is PowerShell 7.4+ only and must not be used here.
$process = Start-Process `
    -FilePath (Join-Path $neo4jHome "bin\neo4j.bat") `
    -ArgumentList "console" `
    -WorkingDirectory $neo4jHome `
    -WindowStyle Hidden `
    -PassThru
Set-Content -LiteralPath $pidFile -Value $process.Id -ErrorAction SilentlyContinue

$ready = $false
for ($attempt = 1; $attempt -le 45; $attempt++) {
    if (Test-BoltReady) {
        $ready = $true
        break
    }
    Start-Sleep -Seconds 2
}
if (-not $ready) {
    throw "Neo4j did not become ready. Check tools/neo4j/neo4j-community-5.26.32/logs/neo4j.log"
}

Write-Host "Neo4j is ready. Browser: http://localhost:7474"
