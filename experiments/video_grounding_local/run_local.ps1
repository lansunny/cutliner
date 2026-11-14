param(
    [string]$Video = 'E:\迅雷云盘\Euphoria.S02E08_10m00s-20m00s.mkv',
    [string]$Query = 'A woman is riding a wooden horse.'
)

$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$taskScript = Join-Path $PSScriptRoot 'run_local.py'
& $taskPython -u $taskScript --video $Video --query $Query
exit $LASTEXITCODE
