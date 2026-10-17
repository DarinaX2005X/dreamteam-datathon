param(
    [string]$Python = "py -3.12",
    [string]$VenvPath = "backend/.venv",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [switch]$FetchModels,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Backend = Join-Path $ProjectRoot "backend"
$Venv = if ([System.IO.Path]::IsPathRooted($VenvPath)) { $VenvPath } else { Join-Path $ProjectRoot $VenvPath }
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
$SelectedDevice = $Device
if ($SelectedDevice -eq "auto") {
    $GpuMemory = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        [double](nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | Select-Object -First 1)
    } else { 0 }
    $SelectedDevice = if ($GpuMemory -ge 8192) { "cuda" } else { "cpu" }
}
$TorchIndex = if ($SelectedDevice -eq "cuda") { "https://download.pytorch.org/whl/cu124" } else { "https://download.pytorch.org/whl/cpu" }
Write-Host "Installing the $SelectedDevice runtime in $Venv"
& $VenvPython -m pip install --timeout 120 --retries 10 --index-url $TorchIndex "torch==2.6.0" "torchvision==0.21.0"
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
& $VenvPython -m pip install --timeout 120 --retries 10 -e "$Backend[service,runtime,$SelectedDevice]"
if ($LASTEXITCODE -ne 0) { throw "Backend dependency installation failed." }

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
    & $VenvPython $FetchScript --profile $SelectedDevice
    if ($LASTEXITCODE -ne 0) { throw "Model download or integrity verification failed." }
}
if (-not $SkipModels) {
    & $VenvPython (Join-Path $ProjectRoot "scripts\preflight.py") --device $SelectedDevice --require-models
    if ($LASTEXITCODE -ne 0) {
        throw "Runtime/model preflight failed. Use -FetchModels after model pins are available, or -SkipModels only for API development."
    }
}
Write-Host "Setup complete. Start with: $VenvPython scripts/run_local.py"
