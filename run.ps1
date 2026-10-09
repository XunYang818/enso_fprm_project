param(
    [ValidateSet('run', 'test', 'prepare')] [string]$Mode = 'run',
    [ValidateSet('smoke', 'full')] [string]$Suite = 'smoke',
    [string]$Config = '',
    [string]$Resume = '',
    [string]$CondaExe = 'conda',
    [switch]$NoPlots,
    [int]$LimitCases = 0
)
$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$artifactRoot = Join-Path $projectPath 'outputs'
New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:MPLCONFIGDIR = Join-Path $artifactRoot 'cache\matplotlib'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
if ($CondaExe -eq 'conda' -and $env:CONDA_EXE) { $CondaExe = $env:CONDA_EXE }
$condaCommand = Get-Command $CondaExe -ErrorAction SilentlyContinue
if (-not $condaCommand) { throw 'Conda is unavailable. Run from Miniforge Prompt or an initialized Conda terminal.' }
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
