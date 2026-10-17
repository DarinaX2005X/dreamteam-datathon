$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$venvPath = Join-Path $PSScriptRoot '.venv'
if (-not (Test-Path -LiteralPath $venvPath)) { py -3.12 -m venv $venvPath }
$pythonPath = Join-Path $venvPath 'Scripts/python.exe'
& $pythonPath -m pip install --disable-pip-version-check torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $pythonPath -m pip install --disable-pip-version-check -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Probe dependencies installation failed' }
& $pythonPath (Join-Path $PSScriptRoot 'download_models.py') rfdetr-nano rfdetr-small grounding-dino-tiny rtmpose-body
if ($LASTEXITCODE -ne 0) { throw 'Model download failed' }
