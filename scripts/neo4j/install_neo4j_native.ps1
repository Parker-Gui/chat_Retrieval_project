param(
    [string]$JavaHome = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$neo4jVersion = "5.26.32"
$toolsDir = Join-Path $projectRoot "tools"
$archive = Join-Path $toolsDir "neo4j-community-$neo4jVersion-windows.zip"
$extractRoot = Join-Path $toolsDir "neo4j"
$neo4jHome = Join-Path $extractRoot "neo4j-community-$neo4jVersion"
$downloadUrl = "https://dist.neo4j.org/neo4j-community-$neo4jVersion-windows.zip"
$envFile = Join-Path $projectRoot "neo4j\.env"
$envExample = Join-Path $projectRoot "neo4j\.env.example"

function Resolve-Java17Home {
    param([string]$RequestedHome)

    $candidates = @()
    if ($RequestedHome) {
        $candidates += $RequestedHome
    }
    if ($env:JAVA_HOME) {
        $candidates += $env:JAVA_HOME
    }
    if (Test-Path -LiteralPath (Join-Path $env:USERPROFILE ".jdks")) {
        $candidates += Get-ChildItem -LiteralPath (Join-Path $env:USERPROFILE ".jdks") -Directory |
            Where-Object { $_.Name -match "^(jbr|jdk)-17" } |
            ForEach-Object { $_.FullName }
    }

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath (Join-Path $candidate "bin\java.exe"))) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "JDK 17 was not found. Pass -JavaHome or set JAVA_HOME to a JDK 17 directory."
}

$resolvedJavaHome = Resolve-Java17Home -RequestedHome $JavaHome
$env:JAVA_HOME = $resolvedJavaHome

if (-not (Test-Path -LiteralPath $toolsDir)) {
    New-Item -ItemType Directory -Path $toolsDir | Out-Null
}
if (-not (Test-Path -LiteralPath $envFile)) {
    Copy-Item -LiteralPath $envExample -Destination $envFile
}
if (-not (Test-Path -LiteralPath $archive)) {
    Write-Host "Downloading Neo4j Community $neo4jVersion..."
    curl.exe -L --fail --output $archive $downloadUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Neo4j download failed"
    }
}
if (-not (Test-Path -LiteralPath $neo4jHome)) {
    Write-Host "Extracting Neo4j..."
    Expand-Archive -LiteralPath $archive -DestinationPath $extractRoot
}

$passwordLine = Get-Content -LiteralPath $envFile |
    Where-Object { $_ -match "^NEO4J_PASSWORD=" } |
    Select-Object -First 1
if (-not $passwordLine) {
    throw "NEO4J_PASSWORD is missing from $envFile"
}
$password = $passwordLine.Substring("NEO4J_PASSWORD=".Length)
$authFile = Join-Path $neo4jHome "data\dbms\auth.ini"
if (-not (Test-Path -LiteralPath $authFile)) {
    & (Join-Path $neo4jHome "bin\neo4j-admin.bat") dbms set-initial-password $password --require-password-change=false
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to set the initial Neo4j password"
    }
}

Write-Host "Neo4j native installation is ready: $neo4jHome"
Write-Host "JAVA_HOME: $resolvedJavaHome"
