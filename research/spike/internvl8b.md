# InternVL3.5-8B candidate pin

Status: **evaluated experimentally; replacement rejected and not quality-approved**. The exact 18-file pin was downloaded and verified on the authorized L40S. The unchanged v5/w16/t1024/p4 pipeline completed all eight inner-validation sources, but produced TP0/TN3/FP1/FN4 versus the original 4B's TP0/TN4/FP0/FN4. The working app was restored to 4B and its ready/HTTP state verified; 8B files are retained. Numeric evidence: [l40s-pipeline-8b-local-frames.json](../../eval/reports/l40s-pipeline-8b-local-frames.json). The following pin and preflight estimates document the evaluated candidate, not a deployment recommendation.

## Primary sources and immutable pin

- Official repository: <https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF>
- Pinned revision: [`741a7d03020411e666c6109218ab71e08151ef86`](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF/tree/741a7d03020411e666c6109218ab71e08151ef86)
- Pinned model card: <https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF/blob/741a7d03020411e666c6109218ab71e08151ef86/README.md>
- Pinned configuration: <https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF/blob/741a7d03020411e666c6109218ab71e08151ef86/config.json>
- Official blob metadata, used for weight sizes and SHA-256 values without downloading weights: <https://huggingface.co/api/models/OpenGVLab/InternVL3_5-8B-HF/revision/741a7d03020411e666c6109218ab71e08151ef86?blobs=true>

The four BF16 safetensors shards total 17,056,744,056 bytes (15.885 GiB); the index is 79,937 bytes. Their SHA-256 values in `models/manifest.json` come from the official LFS blob metadata. Hashes for README, configuration, tokenizer, and index files were computed from the small public files at the pinned revision. All 18 files subsequently passed full local size/SHA verification on the server.

## License and compatibility

The repository/model card declares Apache-2.0. The InternVL project license is [Apache-2.0](https://github.com/OpenGVLab/InternVL/blob/main/LICENSE), and the Qwen3-8B language component has the official [Apache-2.0 license](https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE). This is compatible with commercial use subject to the Apache-2.0 notice and license obligations.

The pinned config declares `InternVLForConditionalGeneration`, `model_type: internvl`, a Qwen3 text model, BF16, and `transformers_version: 4.55.0`. The official model card requires `transformers>=4.52.1`; the project's 4.57.6 satisfies that metadata requirement. Native runtime loading and full HTTP execution were confirmed without OOM or technical failures. This establishes compatibility, not semantic quality.

## Resource gate

Weights alone require about 15.9 GiB. For 16 frames, four actual patches per frame, and roughly 256 visual tokens per patch, the Qwen3 KV cache estimate is about 2.25 GiB in BF16, before vision activations, prefill workspaces, and allocator overhead. Actual patch count depends on aspect ratio and the thumbnail policy and must be measured. An L40S 46,068 MiB run is plausible. The training launchers use a 70% allocator cap; the application inference profile currently has no such configured cap.

The measured fixed-source smoke took 55.199 seconds with 19,771.1 MiB peak CUDA allocation. Full-eight HTTP polling observations ranged from 15.1 to 85.3 seconds; they are not exact worker execution times. Despite fitting the hardware and time limit, the candidate was rejected because it added a false positive without detecting any positive source. No 8B training was attempted, and no reserved holdout was used.
