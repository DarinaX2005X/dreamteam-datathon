"""RF-DETR ONNX adapter, preserving its published input/output semantics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


class RfDetrOnnx:
    def __init__(self, directory: Path, entry: dict, device: str):
        import onnxruntime as ort

        weights = next((item for item in entry["files"] if item["path"].endswith(".onnx")), None)
        config = next((item for item in entry["files"] if Path(item["path"]).name == "config.json"), None)
        preprocessing = next((item for item in entry["files"] if Path(item["path"]).name == "preprocessor_config.json"), None)
        if weights is None or config is None or preprocessing is None:
            raise RuntimeError("MODEL_MANIFEST_INVALID: для RF-DETR нужны ONNX, config и preprocessor_config.")
        self.model_id = entry["id"]
        self.labels = json.loads((directory / config["path"]).read_text("utf-8"))["id2label"]
        self.preprocessing = json.loads((directory / preprocessing["path"]).read_text("utf-8"))
        self.threshold = float(os.getenv("NAVIDU_DETECTION_THRESHOLD", "0.45"))
        if not 0 < self.threshold < 1:
            raise RuntimeError("INVALID_CONFIG: NAVIDU_DETECTION_THRESHOLD должен быть между 0 и 1.")
        is_quantized = "uint8" in weights["path"] or "int8" in weights["path"]
        available = ort.get_available_providers()
        use_cuda = device in {"auto", "cuda"} and not is_quantized and "CUDAExecutionProvider" in available
        if device == "cuda" and not use_cuda:
            raise RuntimeError("CUDA_UNAVAILABLE: нужны RF-DETR FP32/FP16 и работающий ONNX Runtime CUDA provider.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("NAVIDU_CPU_THREADS", "6"))
        options.inter_op_num_threads = 1
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(directory / weights["path"]), sess_options=options, providers=providers)
        actual = self.session.get_providers()
        if use_cuda and "CUDAExecutionProvider" not in actual:
            raise RuntimeError("CUDA_UNAVAILABLE: ONNX Runtime не смог загрузить CUDA provider; проверьте CUDA/cuDNN.")
        self.device = "cuda" if "CUDAExecutionProvider" in actual else "cpu"
        self.output_names = [output.name for output in self.session.get_outputs()]
        if not {"pred_boxes", "logits"}.issubset(self.output_names):
            raise RuntimeError("MODEL_OUTPUT_INVALID: ожидаются именованные pred_boxes и logits.")
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("RF-DETR expects an RGB uint8 frame")
        size = self.preprocessing["size"]
        image = Image.fromarray(rgb).resize((size["width"], size["height"]), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32)
        if self.preprocessing.get("do_rescale", True):
            pixels *= self.preprocessing.get("rescale_factor", 1 / 255.0)
        if self.preprocessing.get("do_normalize", False):
            pixels = (pixels - np.asarray(self.preprocessing["image_mean"])) / np.asarray(self.preprocessing["image_std"])
        pixels = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32)
        outputs = dict(zip(self.output_names, self.session.run(None, {self.input_name: pixels})))
        raw_boxes = outputs["pred_boxes"][0]
        logits = outputs["logits"][0]
        if raw_boxes.ndim != 2 or raw_boxes.shape[-1] != 4 or logits.shape[0] != raw_boxes.shape[0]:
            raise RuntimeError("MODEL_OUTPUT_INVALID: несовместимые размеры RF-DETR output.")
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))
        class_ids = probabilities.argmax(axis=-1)
        scores = probabilities.max(axis=-1)
        keep = (scores >= self.threshold) & np.isfinite(scores)
        centers, wh = raw_boxes[keep, :2], raw_boxes[keep, 2:]
        boxes = np.concatenate((centers - wh / 2, centers + wh / 2), axis=1).clip(0, 1)
        valid = np.isfinite(boxes).all(axis=1) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[valid], class_ids[keep][valid], scores[keep][valid]

