param(
    [string]$Video = '',
    [string]$Action = 'push_left_to_right',
    [double]$Start = 0,
    [Nullable[double]]$End = $null
)
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '..\video_grounding_local\.venv\Scripts\python.exe'
$taskScript = Join-Path $PSScriptRoot 'run_action.py'
$taskArguments = @('--action', $Action, '--start', "$Start")
if ($Video) { $taskArguments += @('--video', $Video) }
if ($null -ne $End) { $taskArguments += @('--end', "$End") }
& $taskPython -u $taskScript @taskArguments
exit $LASTEXITCODE
