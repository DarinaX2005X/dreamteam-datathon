"""CPU-only PowerPoint-to-PDF export for the Lado pitch deck.

The script moves only the authored presentation bytes into the project's private
Modal workspace. It does not access video, frames, annotations, model weights or
other project data.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import modal

app = modal.App("lado-pitch-build")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libreoffice-impress", "fonts-liberation", "fontconfig")
)


@app.function(image=image, cpu=2.0, timeout=600)
def export_pdf(pptx_bytes: bytes) -> bytes:
    with tempfile.TemporaryDirectory(prefix="lado-pitch-") as temp_dir:
        workspace = Path(temp_dir)
        source = workspace / "lado-pitch.pptx"
        source.write_bytes(pptx_bytes)
        completed = subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "pdf:impress_pdf_Export",
                "--outdir",
                str(workspace),
                str(source),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=540,
        )
        output = workspace / "lado-pitch.pdf"
        if completed.returncode != 0 or not output.exists():
            raise RuntimeError(
                "LibreOffice export failed: "
                f"returncode={completed.returncode}; "
                f"stdout={completed.stdout[-2000:]}; stderr={completed.stderr[-2000:]}"
            )
        return output.read_bytes()


@app.local_entrypoint()
def main(source: str, output: str) -> None:
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_bytes = export_pdf.remote(source_path.read_bytes())
    output_path.write_bytes(pdf_bytes)
    print(f"Exported {len(pdf_bytes)} bytes to {output_path}")
