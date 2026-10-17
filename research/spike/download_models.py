"""Explicit download step only; inference scripts never download models.

Run from the repository root. Sources are pinned official checkpoints/model repos.
Only GET requests are made. No video, image, prompt or other user data is sent.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
SOURCES = {
    "rfdetr-nano-onnx": {
        "name": "RF-DETR Nano COCO ONNX UINT8 conversion", "runtime": "onnxruntime", "license": "Apache-2.0",
        "license_url": "https://github.com/roboflow/rf-detr/blob/develop/LICENSE",
        "weights_license_url": "https://huggingface.co/onnx-community/rfdetr_nano-ONNX/blob/eae21cee0687a91bcf9fa071605c48d7705d2d91/README.md",
        "source": "https://huggingface.co/onnx-community/rfdetr_nano-ONNX",
        "repo": "onnx-community/rfdetr_nano-ONNX", "revision": "eae21cee0687a91bcf9fa071605c48d7705d2d91",
        "local_dir": "rfdetr-nano-onnx",
        "hf_files": ["README.md", "config.json", "preprocessor_config.json", "onnx/model_uint8.onnx"],
    },
    "rfdetr-nano-onnx-int8": {
        "name": "RF-DETR Nano COCO ONNX INT8 conversion", "runtime": "onnxruntime", "license": "Apache-2.0",
        "license_url": "https://github.com/roboflow/rf-detr/blob/develop/LICENSE",
        "weights_license_url": "https://huggingface.co/onnx-community/rfdetr_nano-ONNX/blob/eae21cee0687a91bcf9fa071605c48d7705d2d91/README.md",
        "source": "https://huggingface.co/onnx-community/rfdetr_nano-ONNX",
        "repo": "onnx-community/rfdetr_nano-ONNX", "revision": "eae21cee0687a91bcf9fa071605c48d7705d2d91",
        "local_dir": "rfdetr-nano-onnx-int8",
        "hf_files": ["README.md", "config.json", "preprocessor_config.json", "onnx/model_int8.onnx"],
    },
    "rfdetr-nano": {
        "name": "RF-DETR Nano COCO", "runtime": "pytorch", "license": "Apache-2.0",
        "license_url": "https://github.com/roboflow/rf-detr/blob/develop/LICENSE",
        "weights_license_url": "https://rfdetr.roboflow.com/latest/",
        "source": "https://github.com/roboflow/rf-detr",
        "files": [{"path": "rfdetr/rf-detr-nano.pth", "url": "https://repo.roboflow.com/rfdetr/nano_coco/checkpoint_best_regular.pth", "md5": "fb6504cce7fbdc783f7a46991f07639f"}],
    },
    "rfdetr-small": {
        "name": "RF-DETR Small COCO", "runtime": "pytorch", "license": "Apache-2.0",
        "license_url": "https://github.com/roboflow/rf-detr/blob/develop/LICENSE",
        "weights_license_url": "https://rfdetr.roboflow.com/latest/",
        "source": "https://github.com/roboflow/rf-detr",
        "files": [{"path": "rfdetr/rf-detr-small.pth", "url": "https://repo.roboflow.com/rfdetr/small_coco/checkpoint_best_regular.pth", "md5": "fb37061c1af7bace359c91b723a8d5c1"}],
    },
    "grounding-dino-tiny": {
        "name": "Grounding DINO Tiny", "runtime": "transformers-pytorch", "license": "Apache-2.0",
        "license_url": "https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE",
        "weights_license_url": "https://huggingface.co/IDEA-Research/grounding-dino-tiny/blob/a2bb814dd30d776dcf7e30523b00659f4f141c71/README.md",
        "source": "https://huggingface.co/IDEA-Research/grounding-dino-tiny",
        "repo": "IDEA-Research/grounding-dino-tiny", "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
        "local_dir": "grounding-dino-tiny",
        "hf_files": ["README.md", "added_tokens.json", "config.json", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"],
    },
    "florence2-base": {
        "name": "Florence-2 Base (Transformers conversion)", "runtime": "transformers-pytorch", "license": "MIT",
        "license_url": "https://huggingface.co/microsoft/Florence-2-base/blob/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac/LICENSE",
        "weights_license_url": "https://huggingface.co/florence-community/Florence-2-base/blob/00921df66db728a9ceb750f5eca43e5c203a2051/README.md",
        "source": "https://huggingface.co/florence-community/Florence-2-base",
        "repo": "florence-community/Florence-2-base", "revision": "00921df66db728a9ceb750f5eca43e5c203a2051",
        "local_dir": "florence2-base",
        "hf_files": ["README.md", "added_tokens.json", "config.json", "generation_config.json", "merges.txt", "model.safetensors", "preprocessor_config.json", "processor_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
    },
    "rtmpose-body": {
        "name": "RTMPose-M Body7 256x192", "runtime": "onnxruntime", "license": "Apache-2.0",
        "license_url": "https://github.com/open-mmlab/mmpose/blob/main/LICENSE",
        "weights_license_url": "https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose",
        "source": "https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose",
        "files": [{"path": "rtmpose/body.zip", "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"}],
    },
}


def checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    headers = {"User-Agent": "local-video-model-setup/1.0"}
    request = urllib.request.Request(url, headers=headers, method="HEAD")
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length", "0"))
        supports_ranges = response.headers.get("Accept-Ranges", "").lower() == "bytes"
    if supports_ranges and total > 8 * 1024 * 1024:
        # Public checkpoints only. Bounded range downloads avoid very slow single
        # CDN streams; preserve an interrupted contiguous prefix for resume.
        offset = temp.stat().st_size if temp.exists() else 0
        if offset > total:
            raise ValueError(f"Oversized partial checkpoint: {temp}")
        chunks = [(start, min(start + 4 * 1024 * 1024, total) - 1) for start in range(offset, total, 4 * 1024 * 1024)]
        def get_range(bounds):
            start, end = bounds
            piece = temp.with_name(f"{temp.name}.{start}-{end}")
            if piece.exists() and piece.stat().st_size == end - start + 1:
                return piece
            req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as response, piece.open("wb") as target:
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                    raise ValueError("CDN did not honor the requested range")
                for chunk in iter(lambda: response.read(64 * 1024), b""):
                    target.write(chunk)
            if piece.stat().st_size != end - start + 1:
                raise ValueError("Truncated checkpoint range")
            return piece
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            pieces = list(pool.map(get_range, chunks))
        with temp.open("ab") as target:
            for piece in pieces:
                with piece.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
                piece.unlink()
        if temp.stat().st_size != total:
            raise ValueError("Incorrect complete checkpoint size")
    else:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as target:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                target.write(chunk)
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", choices=list(SOURCES))
    parser.add_argument("--model-dir", type=Path, default=MODELS)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = MODELS / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {"schema_version": 1, "models": []}
    existing = {model["id"]: model for model in manifest["models"]}
    for model_id in args.models:
        source = dict(SOURCES[model_id])
        files = source.pop("files", [])
        for filename in source.pop("hf_files", []):
            files.append({"path": f"{source['local_dir']}/{filename}", "url": f"https://huggingface.co/{source['repo']}/resolve/{source['revision']}/{filename}"})
        records = []
        previous = {item["path"]: item for item in existing.get(model_id, {}).get("files", [])}
        for spec in files:
            path = model_dir / spec["path"]
            if not path.exists():
                print(f"Downloading {model_id}: {spec['path']}", flush=True)
                download(spec["url"], path)
            sha = checksum(path)
            if spec.get("md5") and checksum(path, "md5") != spec["md5"]:
                raise ValueError(f"Upstream MD5 mismatch: {path}")
            if spec["path"] in previous and sha != previous[spec["path"]]["sha256"]:
                raise ValueError(f"Manifest SHA256 mismatch: {path}")
            records.append({**spec, "sha256": sha, "size_bytes": path.stat().st_size})
            if path.suffix == ".zip":
                extraction = path.parent / path.stem
                with zipfile.ZipFile(path) as archive:
                    for member in archive.infolist():
                        target = (extraction / member.filename).resolve()
                        if not target.is_relative_to(extraction.resolve()):
                            raise ValueError("Unsafe archive member")
                    archive.extractall(extraction)
                for child in extraction.rglob("*.onnx"):
                    records.append({"path": child.relative_to(model_dir).as_posix(), "derived_from": spec["path"], "sha256": checksum(child), "size_bytes": child.stat().st_size})
        existing[model_id] = {"id": model_id, **source, "files": records}
        manifest["models"] = list(existing.values())
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Verified {model_id}: {sum(r['size_bytes'] for r in records):,} bytes", flush=True)


if __name__ == "__main__":
    main()
