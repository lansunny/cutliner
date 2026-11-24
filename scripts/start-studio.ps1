param(
    [ValidateSet('postgresql', 'sqlite')][string]$Database = 'postgresql',
    [string]$Python = 'python',
    [ValidateRange(1, 65535)][int]$Port = 7861,
    [switch]$Build
)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$originalUrl = $env:CUTLINER_DATABASE_URL
$originalPassword = $env:CUTLINER_DB_PASSWORD
Push-Location -LiteralPath $project
try {
    if ($Build -or -not (Test-Path -LiteralPath 'frontend/dist/index.html')) {
        Push-Location -LiteralPath 'frontend'
        try {
            if (-not (Test-Path -LiteralPath 'node_modules')) {
                npm ci
                if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
            }
            npm run build
            if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
        } finally { Pop-Location }
    }
    if ($Database -eq 'sqlite') {
        # An explicit empty environment variable prevents dotenv restoring a DSN.
        $env:CUTLINER_DATABASE_URL = ' '
    } elseif (-not $env:CUTLINER_DATABASE_URL) {
        $null = New-Item -ItemType Directory -Path 'studio_data' -Force
        $secretPath = Join-Path $project 'studio_data/postgres.password'
        if (-not (Test-Path -LiteralPath $secretPath)) {
            $bytes = New-Object byte[] 24
            $random = [Security.Cryptography.RandomNumberGenerator]::Create()
            try { $random.GetBytes($bytes) } finally { $random.Dispose() }
            [IO.File]::WriteAllText($secretPath, ([BitConverter]::ToString($bytes) -replace '-', '').ToLowerInvariant())
        }
        $env:CUTLINER_DB_PASSWORD = [IO.File]::ReadAllText($secretPath).Trim()
        docker compose up -d --wait db
        if ($LASTEXITCODE -ne 0) { throw 'PostgreSQL did not start. Check Docker Desktop or select -Database sqlite.' }
        $env:CUTLINER_DATABASE_URL = "postgresql://cutliner:$($env:CUTLINER_DB_PASSWORD)@127.0.0.1:54329/cutliner?sslmode=disable"
    }
    & $Python studio_server.py --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Cutliner server stopped with an error.' }
} finally {
    $env:CUTLINER_DATABASE_URL = $originalUrl
    $env:CUTLINER_DB_PASSWORD = $originalPassword
    Pop-Location
}
