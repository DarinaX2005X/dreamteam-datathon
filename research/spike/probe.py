"""Local temporal perception probe; all media and outputs remain on this machine.

Examples:
  python research/spike/probe.py --model rfdetr-nano --video data/dataset/open/videos/X-1.mov --fps 3 --offline
  python research/spike/probe.py --model grounding-dino-tiny --video data/dataset/open/videos/X-1.mov --fps 1 --offline
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import socket
import statistics
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
ROOT = Path(__file__).resolve().parents[2]


def deny_network():
    """Deny socket connection attempts in this process, in addition to HF flags."""
    def denied(*_args, **_kwargs):
        raise RuntimeError("Offline probe forbids network access")
    socket.create_connection = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.getaddrinfo = denied


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights(model_id, model_dir):
    manifest = json.loads((ROOT / "models/manifest.json").read_text("utf-8"))
    model = next(entry for entry in manifest["models"] if entry["id"] == model_id)
    for entry in model["files"]:
        path = model_dir / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Weights missing or modified: {path}")
    return model


def samples(video, fps, start=0.0, end=None):
    import av
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
        next_t = start
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("A video frame has no PTS; temporal alignment cannot be guaranteed")
            timestamp = float(frame.pts * frame.time_base) - origin
            if timestamp + 1e-7 < next_t:
                continue
            if end is not None and timestamp > end:
                return
            # Rotation is absent in the bank dataset. Production applies metadata.
            yield round(timestamp, 6), frame.to_image().convert("RGB")
            next_t += 1.0 / fps


def normalized(box, width, height):
    return [round(max(0.0, min(1.0, float(v) / (width if i % 2 == 0 else height))), 6) for i, v in enumerate(box)]


def rfdetr_runner(args):
    import numpy as np
    import supervision as sv
    from rfdetr import RFDETRNano, RFDETRSmall
    from rfdetr.assets.coco_classes import COCO_CLASSES
    cls = RFDETRNano if args.model == "rfdetr-nano" else RFDETRSmall
    model = cls(pretrain_weights=str(args.model_dir / "rfdetr" / ("rf-detr-nano.pth" if args.model == "rfdetr-nano" else "rf-detr-small.pth")), device=args.device)
    tracker = sv.ByteTrack(frame_rate=max(1, round(args.fps)), minimum_consecutive_frames=2, lost_track_buffer=max(2, round(args.fps * 2)))
    def run(image):
        detections = model.predict(image, threshold=args.threshold)
        people = detections[detections.class_id == 1]
        tracked = tracker.update_with_detections(people)
        objects = [{"class_id": int(c), "label": COCO_CLASSES[int(c)], "bbox": normalized(b, image.width, image.height), "score": round(float(s), 6)} for b, c, s in zip(detections.xyxy, detections.class_id, detections.confidence)]
        tracks = [{"track_id": f"p{int(k)}", "kind": "person", "bbox": normalized(b, image.width, image.height), "score": round(float(s), 6)} for b, s, k in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id)]
        return {"objects": objects, "tracks": tracks}
    return run


def rfdetr_onnx_runner(args):
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    folder = args.model_dir / "rfdetr-nano-onnx"
    labels = json.loads((folder / "config.json").read_text("utf-8"))["id2label"]
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 6
    settings.inter_op_num_threads = 1
    session = ort.InferenceSession(str(folder / "onnx/model_uint8.onnx"), sess_options=settings, providers=["CPUExecutionProvider"])
    print(json.dumps({"event": "onnx_contract", "inputs": [(v.name, v.shape) for v in session.get_inputs()], "outputs": [(v.name, v.shape) for v in session.get_outputs()]}), flush=True)
    import supervision as sv
    tracker = sv.ByteTrack(frame_rate=max(1, round(args.fps)), minimum_consecutive_frames=2, lost_track_buffer=max(2, round(args.fps * 2)))
    def run(image):
        # Match this conversion's published RTDetrImageProcessor config: resize
        # square + rescale; normalization is already within the exported graph.
        resized = image.resize((384, 384), resample=Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        raw = session.run(None, {session.get_inputs()[0].name: pixels})
        outputs = dict(zip([o.name for o in session.get_outputs()], raw))
        boxes = outputs["pred_boxes"][0]
        logits = outputs["logits"][0]
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -88, 88)))
        class_ids = probabilities.argmax(axis=-1)
        scores = probabilities.max(axis=-1)
        keep = scores >= args.threshold
        centers, wh = boxes[keep, :2], boxes[keep, 2:]
        xyxy = np.concatenate((centers - wh / 2, centers + wh / 2), axis=-1).clip(0, 1)
        class_ids, scores = class_ids[keep], scores[keep]
        objects = [{"class_id": int(c), "label": labels[str(int(c))], "bbox": b.round(6).tolist(), "score": round(float(s), 6)} for b, c, s in zip(xyxy, class_ids, scores)]
        people = class_ids == 1
        detections = sv.Detections(xyxy=xyxy[people] * np.array([image.width, image.height, image.width, image.height]), confidence=scores[people], class_id=class_ids[people])
        tracked = tracker.update_with_detections(detections)
        tracks = [{"track_id": f"p{int(k)}", "kind": "person", "bbox": normalized(b, image.width, image.height), "score": round(float(s), 6)} for b, s, k in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id)]
        return {"objects": objects, "tracks": tracks}
    return run


def grounding_runner(args):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    source = args.model_dir / "grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(source, local_files_only=True).to(args.device).eval()
    prompt = "a person. a mobile phone. banknotes. a bundle of cash. a bag. a money counting machine. a cash tray. a safe. a desk."
    def run(image):
        inputs = processor(images=image, text=prompt, return_tensors="pt", size={"shortest_edge": 640, "longest_edge": 960}).to(args.device)
        with torch.inference_mode():
            output = model(**inputs)
        detected = processor.post_process_grounded_object_detection(output, inputs.input_ids, threshold=args.threshold, text_threshold=0.25, target_sizes=[(image.height, image.width)])[0]
        labels = detected.get("text_labels", detected.get("labels"))
        objects = [{"label": str(label), "bbox": normalized(box.tolist(), image.width, image.height), "score": round(float(score), 6)} for box, score, label in zip(detected["boxes"], detected["scores"], labels)]
        return {"objects": objects, "tracks": [], "prompt_version": "cash-domain-v1"}
    return run


def florence_runner(args):
    import torch
    from transformers import AutoProcessor, Florence2ForConditionalGeneration
    source = args.model_dir / "florence2-base"
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    model = Florence2ForConditionalGeneration.from_pretrained(source, local_files_only=True, dtype=dtype).to(args.device).eval()
    def run(image):
        task = "<MORE_DETAILED_CAPTION>"
        inputs = processor(text=task, images=image, return_tensors="pt").to(args.device, dtype=dtype)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=128, do_sample=False, num_beams=1)
        generated = processor.batch_decode(output, skip_special_tokens=False)[0]
        # This unverified caption stays in local artifacts; never print it to logs.
        return {"objects": [], "tracks": [], "unverified_local_caption": processor.post_process_generation(generated, task=task, image_size=image.size), "for_rule_inference": False}
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rfdetr-nano", "rfdetr-small", "rfdetr-nano-onnx", "grounding-dino-tiny", "florence2-base"], required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("fps must be positive")
    if args.offline:
        deny_network()
    import torch
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    args.model_dir = args.model_dir.resolve()
    model_meta = verify_weights(args.model, args.model_dir)
    output = args.output or ROOT / "research/spike/artifacts" / args.video.stem / f"{args.model}-{args.device}-{args.fps:g}fps"
    output.mkdir(parents=True, exist_ok=True)
    runners = {"rfdetr-nano": rfdetr_runner, "rfdetr-small": rfdetr_runner, "rfdetr-nano-onnx": rfdetr_onnx_runner, "grounding-dino-tiny": grounding_runner, "florence2-base": florence_runner}
    began = time.perf_counter()
    runner = runners[args.model](args)
    load_sec = time.perf_counter() - began
    print(json.dumps({"event": "model_loaded", "model": args.model, "device": args.device, "load_sec": round(load_sec, 3)}), flush=True)
    timings, counts, person_counts, timestamps = [], [], [], []
    track_aggregate = {}
    processing_start = time.perf_counter()
    with (output / "objects.jsonl").open("w", encoding="utf-8") as stream:
        for index, (timestamp, image) in enumerate(samples(args.video, args.fps, args.start, args.end)):
            if args.device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = runner(image)
            if args.device == "cuda":
                torch.cuda.synchronize()
            duration = time.perf_counter() - started
            row = {"timestamp_sec": round(timestamp, 3), "width": image.width, "height": image.height, "inference_sec": round(duration, 6), "model_id": args.model, **result}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            for track in result["tracks"]:
                aggregate = track_aggregate.setdefault(track["track_id"], {"track_id": track["track_id"], "kind": "person", "object_class": None, "first_sec": row["timestamp_sec"], "last_sec": row["timestamp_sec"], "role": {"value": "unknown", "source": "auto"}, "samples": []})
                aggregate["last_sec"] = row["timestamp_sec"]
                aggregate["samples"].append({"t": row["timestamp_sec"], "bbox": track["bbox"], "score": track["score"]})
            timings.append(duration)
            timestamps.append(timestamp)
            counts.append(len(result["objects"]))
            person_counts.append(sum(1 for ob in result["objects"] if ob["label"] in {"person", "a person"}))
            if index % 20 == 0:
                print(json.dumps({"event": "frame", "index": index, "timestamp_sec": timestamp, "inference_sec": round(duration, 3), "detections": len(result["objects"])}), flush=True)
    processing_sec = time.perf_counter() - processing_start
    for name, data in [("tracks.jsonl", track_aggregate.values())]:
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data), encoding="utf-8")
    summary = {
        "model_id": args.model, "runtime": model_meta["runtime"], "device": args.device,
        "video_sha256": sha256(args.video), "video_file": args.video.name,
        "analysis_fps": args.fps, "sample_count": len(timings),
        "first_timestamp_sec": min(timestamps) if timestamps else None, "last_timestamp_sec": max(timestamps) if timestamps else None,
        "load_sec": round(load_sec, 3), "processing_sec": round(processing_sec, 3),
        "mean_frame_sec": round(statistics.mean(timings), 4) if timings else None,
        "warm_mean_frame_sec": round(statistics.mean(timings[1:]), 4) if len(timings) > 1 else None,
        "min_person_detections": min(person_counts) if person_counts else 0, "max_person_detections": max(person_counts) if person_counts else 0,
        "person_tracks": len(track_aggregate), "offline_socket_guard": args.offline,
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated() / 1048576, 1) if args.device == "cuda" else None,
        "limitations": ["Object predictions are unverified; no violation conclusion is generated by this probe", "Sampled frames do not prove unobserved actions between samples", "Dataset-specific object/action accuracy is not established by this timing run"],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
