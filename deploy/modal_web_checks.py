"""Compile and unit-test web code on a small CPU container, without a browser."""
from __future__ import annotations

import io
import json
import subprocess
import zipfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("lado-web-build")
image = modal.Image.debian_slim(python_version="3.11").apt_install("nodejs", "npm").workdir("/workspace")
for name in ("package.json", "package-lock.json", "web/package.json", "landing/package.json"):
    image = image.add_local_file(ROOT / name, "/workspace/" + name, copy=True)
image = image.run_commands("npm ci --ignore-scripts --no-audit --no-fund")
# The Windows-created lock lacks Linux optional binaries. Match the versions
# already pinned in that lock; do not update Rollup or esbuild themselves.
image = image.run_commands("npm install --no-save --package-lock=false --ignore-scripts --no-audit --no-fund "
                           "@rollup/rollup-linux-x64-gnu@4.64.7 @esbuild/linux-x64@0.21.5")
for name in ("web/src", "web/tests", "landing/src", "landing/dev", "contracts/ts"):
    image = image.add_local_dir(ROOT / name, "/workspace/" + name, copy=True)
for name in ("web/tsconfig.json", "web/vite.config.ts", "web/index.html", "web/public/favicon.svg",
             "landing/tsconfig.json", "landing/tsconfig.build.json", "landing/vite.config.ts",
             "contracts/product.json", "identity/logo/lado-logo.svg",
             "identity/logo/lado-mark-on-stage.svg", "identity/art/desk-scene.svg",
             "identity/tokens/tokens.css", "identity/logo/lado-logo-mono.svg"):
    image = image.add_local_file(ROOT / name, "/workspace/" + name, copy=True)


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=600,
              retries=0, max_containers=1, scaledown_window=2)
def build() -> bytes:
    for command in (["npm", "run", "build"], ["npm", "run", "typecheck"], ["npm", "test"],
                    ["node", "node_modules/typescript/bin/tsc", "--project", "contracts/ts/tsconfig.json", "--noEmit"]):
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        print(result.stdout[-2500:], flush=True)
        if result.returncode:
            print(result.stderr[-2500:], flush=True)
            raise RuntimeError("Web build or unit check failed")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path("web/dist").rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to("web/dist"))
    return buffer.getvalue()


@app.local_entrypoint()
def main():
    payload = build.remote()
    destination = (ROOT / "web/dist").resolve()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination) or member.is_dir():
                raise RuntimeError("Unexpected build archive member")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
    print(json.dumps({"gpu_allocated": False, "web_build_and_checks": "passed",
                      "archive_bytes": len(payload), "destination": str(destination)}))
