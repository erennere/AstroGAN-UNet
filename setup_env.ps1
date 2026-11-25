# Create .venv in this folder and install astrounets into it (Windows, CPU TensorFlow).
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1
# Python 3.11-3.13 is taken from $env:PYTHON (a python.exe path), else the py launcher, else `python`.
#   -Check   only report which Python would be used (exit code 1 if none fits)
param([switch]$Check)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-Python {
    $ErrorActionPreference = "Continue"  # a missing launcher version only writes to stderr
    $candidates = @()
    if ($env:PYTHON) { $candidates += "$env:PYTHON|" }
    if (Get-Command py -ErrorAction SilentlyContinue) { $candidates += @("py|-3.11", "py|-3.12", "py|-3.13") }
    if (Get-Command python -ErrorAction SilentlyContinue) { $candidates += "python|" }
    foreach ($candidate in $candidates) {
        $exe, $flag = $candidate.Split("|")
        $pre = @($flag | Where-Object { $_ })
        try {
            $ok = & $exe @pre -c "import sys; print(int((3, 11) <= sys.version_info[:2] <= (3, 13)))" 2>$null
        } catch { continue }
        if ($LASTEXITCODE -eq 0 -and "$ok".Trim() -eq "1") { return @($exe) + $pre }
    }
}

$python = @(Find-Python)
if ($python.Count -eq 0) {
    Write-Host "No Python 3.11-3.13 found. Point to one and run again, e.g.:" -ForegroundColor Red
    Write-Host '  $env:PYTHON = "D:\path\to\python.exe"' -ForegroundColor Red
    exit 1
}
$exe = $python[0]; $pre = @($python | Select-Object -Skip 1)
$version = & $exe @pre -c "import sys; print(sys.version.split()[0], sys.executable)"
Write-Host "Python: $version"
if ($Check) { exit 0 }

if (-not (Test-Path .venv\Scripts\python.exe)) {
    & $exe @pre -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "creating .venv failed" }
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& .\.venv\Scripts\python.exe -m pip install -e ".[cpu,dev]"
if ($LASTEXITCODE -ne 0) { throw "installing astrounets failed" }
& .\.venv\Scripts\astrounets.exe --help | Out-Null
if ($LASTEXITCODE -ne 0) { throw "the astrounets command does not start" }
Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"
