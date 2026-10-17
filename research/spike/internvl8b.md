# InternVL3.5-8B candidate pin

Status: **candidate only; not activated and not quality-approved**. This pin makes the official 8B checkpoint reproducible if the 4B `max_patches=4` comparison justifies a measured follow-up. It does not assert better event detection.

## Primary sources and immutable pin

- Official repository: <https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF>
- Pinned revision: [`741a7d03020411e666c6109218ab71e08151ef86`](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF/tree/741a7d03020411e666c6109218ab71e08151ef86)
- Pinned model card: <https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF/blob/741a7d03020411e666c6109218ab71e08151ef86/README.md>
- Pinned configuration: <https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF/blob/741a7d03020411e666c6109218ab71e08151ef86/config.json>
- Official blob metadata, used for weight sizes and SHA-256 values without downloading weights: <https://huggingface.co/api/models/OpenGVLab/InternVL3_5-8B-HF/revision/741a7d03020411e666c6109218ab71e08151ef86?blobs=true>

The four BF16 safetensors shards total 17,056,744,056 bytes (15.885 GiB); the index is 79,937 bytes. Their SHA-256 values in `models/manifest.json` come from the official LFS blob metadata. Hashes for README, configuration, tokenizer, and index files were computed from the small public files at the pinned revision. No weight shard was downloaded.

## License and compatibility

The repository/model card declares Apache-2.0. The InternVL project license is [Apache-2.0](https://github.com/OpenGVLab/InternVL/blob/main/LICENSE), and the Qwen3-8B language component has the official [Apache-2.0 license](https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE). This is compatible with commercial use subject to the Apache-2.0 notice and license obligations.

The pinned config declares `InternVLForConditionalGeneration`, `model_type: internvl`, a Qwen3 text model, BF16, and `transformers_version: 4.55.0`. The official model card requires `transformers>=4.52.1`; the project's 4.57.6 satisfies that metadata requirement. Runtime compatibility remains to be established by a bounded local/self-hosted smoke run after candidate activation.

## Resource gate

Weights alone require about 15.9 GiB. For 16 frames, four actual patches per frame, and roughly 256 visual tokens per patch, the Qwen3 KV cache estimate is about 2.25 GiB in BF16, before vision activations, prefill workspaces, and allocator overhead. Actual patch count depends on aspect ratio and the thumbnail policy and must be measured. An L40S 46,068 MiB run is plausible. The training launchers use a 70% allocator cap; the application inference profile currently has no such configured cap.

Keep this candidate inactive until the 4B patch comparison and event-format diagnostics are decided. If activated, first run model load plus one representative label-blind validation source while recording peak VRAM and elapsed time; proceed to the remaining validation set only if it fits available memory and the 300-second limit. Training and quality claims require separate evidence.
