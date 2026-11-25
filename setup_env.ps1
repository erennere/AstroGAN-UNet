# Create .venv in this folder (or update it) and install astrounets into it (Windows, CPU TensorFlow).
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1
# An existing .venv with Python 3.11-3.13 is reused. To create one, Python is taken from $env:PYTHON
# (a python.exe path), else the Python an earlier .venv was made from, else the py launcher, else `python`.
#   -Check   only report which Python would be used (exit code 1 if none fits)
param([switch]$Check)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Test-Python([string[]]$Command) {
    $ErrorActionPreference = "Continue"  # a missing interpreter or launcher version only writes to stderr
    $exe = $Command[0]; $pre = @($Command | Select-Object -Skip 1)
    try {
        $ok = & $exe @pre -c "import sys; print(int((3, 11) <= sys.version_info[:2] <= (3, 13)))" 2>$null
    } catch { return $false }
    return ($LASTEXITCODE -eq 0 -and "$ok".Trim() -eq "1")
}

function Find-Python {
    $candidates = @()
    if ($env:PYTHON) { $candidates += , @($env:PYTHON) }
    if (Test-Path .venv\pyvenv.cfg) {
        $line = Get-Content .venv\pyvenv.cfg | Where-Object { $_ -match '^\s*executable\s*=' } | Select-Object -First 1
        if ($line) { $candidates += , @(($line -split '=', 2)[1].Trim()) }
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in "-3.11", "-3.12", "-3.13") { $candidates += , @("py", $v) }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) { $candidates += , @("python") }
    foreach ($candidate in $candidates) {
        if (Test-Python $candidate) { return , $candidate }
    }
}

$venv = ".venv\Scripts\python.exe"
$reuse = (Test-Path $venv) -and (Test-Python @($venv))
if ($reuse) {
    $python = @($venv)
} else {
    $python = Find-Python
    if (-not $python) {
        Write-Host "No Python 3.11-3.13 found. Point to one and run again, e.g.:" -ForegroundColor Red
        Write-Host '  $env:PYTHON = "D:\path\to\python.exe"' -ForegroundColor Red
        exit 1
    }
}
$exe = $python[0]; $pre = @($python | Select-Object -Skip 1)
$version = & $exe @pre -c "import sys; print(sys.version.split()[0], sys.executable)"
Write-Host "Python: $version$(if ($reuse) { ' (existing .venv)' })"
if ($Check) { exit 0 }

if (-not $reuse) {
    & $exe @pre -m venv --clear .venv
    if ($LASTEXITCODE -ne 0) { throw "creating .venv failed" }
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& .\.venv\Scripts\python.exe -m pip install -e ".[cpu,dev]"
if ($LASTEXITCODE -ne 0) { throw "installing astrounets failed" }
# The SLURM plugin is Linux-only and stops Snakemake from starting on Windows; remove it if an
# earlier install put it here.
& .\.venv\Scripts\python.exe -m pip uninstall -y -q snakemake-executor-plugin-slurm snakemake-executor-plugin-slurm-jobstep
& .\.venv\Scripts\astrounets.exe --help | Out-Null
if ($LASTEXITCODE -ne 0) { throw "the astrounets command does not start" }
Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"
