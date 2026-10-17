param(
    [string]$Python = "py -3.12",
    [switch]$FetchModels,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Backend = Join-Path $ProjectRoot "backend"
$Venv = Join-Path $Backend ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"

function Invoke-PythonSelector {
    param([string[]]$Arguments)
    $parts = $Python -split " ", 2
    if ($parts.Count -eq 1) { & $parts[0] @Arguments }
    else { & $parts[0] $parts[1] @Arguments }
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $Python $Arguments" }
}

Set-Location $ProjectRoot
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Creating backend virtual environment at $Venv"
    Invoke-PythonSelector @("-m", "venv", $Venv)
}
& $VenvPython -m pip install --timeout 120 --retries 10 --upgrade pip
& $VenvPython -m pip install --timeout 120 --retries 10 -e "$Backend[service]"
if ($LASTEXITCODE -ne 0) { throw "Backend dependency installation failed." }
& $VenvPython -m pip install --timeout 120 --retries 10 -r (Join-Path $ProjectRoot "deploy\requirements-cpu.txt")
if ($LASTEXITCODE -ne 0) { throw "Local model runtime dependency installation failed." }

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or -not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    throw "ffmpeg and ffprobe are required on PATH. Install FFmpeg, then rerun setup."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "Node.js/npm is required to build the Lado interface."
}
npm ci
if ($LASTEXITCODE -ne 0) { throw "npm ci failed." }
npm run build
if ($LASTEXITCODE -ne 0) { throw "Web build failed." }

$FetchScript = Join-Path $ProjectRoot "scripts\fetch_models.py"
if ($FetchModels) {
    if (-not (Test-Path -LiteralPath $FetchScript)) { throw "scripts/fetch_models.py is not available yet." }
    & $VenvPython $FetchScript
    if ($LASTEXITCODE -ne 0) { throw "Model download or integrity verification failed." }
}
if (-not $SkipModels) {
    & $VenvPython (Join-Path $ProjectRoot "scripts\preflight.py") --require-models
    if ($LASTEXITCODE -ne 0) {
        throw "Runtime/model preflight failed. Use -FetchModels after model pins are available, or -SkipModels only for API development."
    }
}
Write-Host "Setup complete. Start with: $VenvPython scripts/run_local.py"
