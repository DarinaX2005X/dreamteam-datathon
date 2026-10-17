param(
    [ValidateSet('cpu', 'cuda')]
    [string]$Device = 'cuda',
    [string]$VenvPath = '.venv-spike',
    [switch]$SkipModels
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$venvDirectory = if ([System.IO.Path]::IsPathRooted($VenvPath)) {
    $VenvPath
} else {
    Join-Path $PSScriptRoot $VenvPath
}
$pythonPath = Join-Path $venvDirectory 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    & py -3.10 -m venv $venvDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.10 virtual environment creation failed.' }
}
$torchIndex = if ($Device -eq 'cuda') {
    'https://download.pytorch.org/whl/cu118'
} else {
    'https://download.pytorch.org/whl/cpu'
}
& $pythonPath -m pip install --disable-pip-version-check --timeout 120 --retries 5 `
    torch==2.6.0 torchvision==0.21.0 --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $pythonPath -m pip install --disable-pip-version-check --timeout 120 --retries 5 `
    -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Probe dependency installation failed.' }
if (-not $SkipModels) {
    # This measured laptop profile keeps UINT8 RF-DETR on CPU. The temporal
    # model uses the selected PyTorch device; CUDA ONNX is a separate profile.
    & $pythonPath (Join-Path $repoRoot 'scripts/fetch_models.py') --profile cpu
    if ($LASTEXITCODE -ne 0) { throw 'Pinned model download or verification failed.' }
}
Write-Host "Research environment ready: $pythonPath"
Write-Host 'Inference scripts enforce offline operation. Model installation alone does not validate analysis quality.'
