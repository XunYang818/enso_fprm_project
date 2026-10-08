param(
    [ValidateSet('run', 'test', 'prepare')] [string]$Mode = 'run',
    [ValidateSet('smoke', 'full')] [string]$Suite = 'smoke',
    [string]$Config = '',
    [string]$Resume = '',
    [string]$CondaExe = 'D:\miniforge3\Scripts\conda.exe',
    [switch]$NoPlots,
    [int]$LimitCases = 0
)
$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$workspacePath = Split-Path -Parent $projectPath
$artifactRoot = Join-Path $workspacePath 'codex_proc\enso_fprm'
New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:MPLCONFIGDIR = Join-Path $artifactRoot 'cache\matplotlib'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
if (-not (Test-Path -LiteralPath $CondaExe)) { throw "Miniforge conda.exe missing: $CondaExe" }
Push-Location -LiteralPath $projectPath
try {
    if ($Mode -eq 'test') {
        New-Item -ItemType Directory -Path (Join-Path $artifactRoot 'tests') -Force | Out-Null
        $testRoot = Join-Path $artifactRoot ('tests\' + [guid]::NewGuid().ToString('N'))
        $condaArgs = @('run', '--no-capture-output', '-n', 'enso-fprm', 'python', '-B', '-m', 'pytest', '--basetemp', $testRoot)
    } else {
        if (-not $Config) { $Config = Join-Path $projectPath 'configs\default.yaml' }
        $condaArgs = @('run', '--no-capture-output', '-n', 'enso-fprm', 'python', '-B', '-m', 'enso_fprm', '--suite', $Suite, '--config', $Config)
        if ($Mode -eq 'prepare') { $condaArgs += '--prepare' }
        if ($Resume) { $condaArgs += @('--resume', $Resume) }
        if ($NoPlots) { $condaArgs += '--no-plots' }
        if ($LimitCases -gt 0) { $condaArgs += @('--limit-cases', "$LimitCases") }
    }
    & $CondaExe @condaArgs
    if ($LASTEXITCODE -ne 0) { throw "Execution failed (exit code $LASTEXITCODE). Inspect logged failures." }
} finally { Pop-Location }
