"""Lazy adapter: the API process never warms or runs perception models."""

import importlib
from pathlib import Path

from navidu.contracts.pipeline import PipelineConfig

from .config import Settings


class LocalPipelineAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._module = None
        self._prepared = False

    def _load(self):
        if self._module is None:
            importlib.invalidate_caches()
            self._module = importlib.import_module("navidu.pipeline.run")
        return self._module

    def config(self, check_id="000000000000", version=None):
        return PipelineConfig(
            check_id=check_id, pipeline_version=version or self.settings.pipeline_version,
            models_dir=self.settings.models_dir, rules_dir=self.settings.rules_dir,
            device=self.settings.device, analysis_fps=self.settings.analysis_fps,
            evidence_overlay=self.settings.evidence_overlay,
            max_duration_sec=self.settings.max_duration_sec,
        )

    def prepare(self):
        if self._prepared:
            return
        module = self._load()
        if hasattr(module, "prepare"):
            module.prepare(self.config())
        if hasattr(module, "runtime_status"):
            value = module.runtime_status(self.config())
            value = value.model_dump() if hasattr(value, "model_dump") else value
            self._prepared = value.get("model_status") == "ready"
        else:
            self._prepared = True

    def runtime_status(self):
        module = self._load()
        if hasattr(module, "runtime_status"):
            value = module.runtime_status(self.config())
            return value.model_dump() if hasattr(value, "model_dump") else value
        return {
            "model_status": "ready" if self._prepared else "loading",
            "device": self.settings.device,
            "limitations": [],
            "pipeline_version": self.settings.pipeline_version,
        }

    def analyze(self, video_path: Path, **kwargs):
        self.prepare()
        return self._load().analyze(video_path, **kwargs)

    def reevaluate(self, artifacts, **kwargs):
        # Deterministic reevaluation uses saved facts, not a model forward pass.
        return self._load().reevaluate(artifacts, **kwargs)
