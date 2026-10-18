import copy

import pytest

from training.object_data import (
    CLASSES,
    crop_objects,
    detection_metrics,
    split_by_source,
    suppress_duplicates,
    validate_annotations,
)


def record(source="a", timestamp=1):
    return {"source_sha256": source, "timestamp_sec": timestamp, "width": 100, "height": 80,
            "remote_frame": "/private/object-labels/run/frame.png", "label_source": "assistant_manual_review",
            "reviewed_classes": list(CLASSES), "objects": [{"kind": "cash", "bbox_xyxy": [20, 20, 40, 40]}]}


def test_source_split_keeps_nearby_frames_together():
    records = [record(str(source), timestamp) for source in range(10) for timestamp in (1, 1.2, 2)]
    train, valid = split_by_source(records)
    assert {r["source_sha256"] for r in train}.isdisjoint(r["source_sha256"] for r in valid)
    assert len(train) + len(valid) == len(records)


def test_class_cutoffs_retain_small_objects_without_lowering_machine_cutoff():
    from training.object_data import DETECTION_THRESHOLDS

    cash = {"kind": "cash", "bbox_xyxy": [20, 20, 40, 40]}
    counter = {"kind": "money_counter", "bbox_xyxy": [50, 50, 70, 70]}
    predictions = [[{**cash, "score": .12}, {**counter, "score": .12}]]
    metrics = detection_metrics(predictions, [[cash]], threshold=DETECTION_THRESHOLDS)
    assert metrics["classes"]["cash"]["tp"] == 1
    assert metrics["classes"]["money_counter"]["fp"] == 0


@pytest.mark.parametrize("change", [{"source_sha256": "holdout"}, {"label_source": "unreviewed_teacher"},
                                    {"objects": [{"kind": "cash", "bbox_xyxy": [20, 20, 101, 40]}]},
                                    {"timestamp_sec": float("nan")}])
def test_invalid_labels_rejected_before_opening_any_image(change):
    item = record()
    item.update(change)
    with pytest.raises(ValueError):
        validate_annotations({"schema_version": 1, "classes": list(CLASSES), "annotations": [item]}, {"a"})


def test_nested_banknote_detections_count_once_but_separate_stacks_survive():
    objects = [{"kind": "cash", "bbox_xyxy": [20, 20, 40, 40], "score": .8},
               {"kind": "cash", "bbox_xyxy": [21, 21, 30, 30], "score": .7},
               {"kind": "cash", "bbox_xyxy": [60, 20, 80, 40], "score": .6}]
    assert len(suppress_duplicates(objects)) == 2


def test_crop_coordinates_and_visibility():
    original = record()["objects"]
    before = copy.deepcopy(original)
    assert crop_objects(original, (10, 10, 50, 50))[0]["bbox_xyxy"] == [10, 10, 30, 30]
    assert crop_objects(original, (39, 39, 50, 50)) == []
    assert original == before


def test_each_ground_truth_matches_at_most_one_prediction():
    truth = record()["objects"]
    prediction = [{**truth[0], "score": .9}, {**truth[0], "score": .8}]
    cash = detection_metrics([prediction], [truth])["classes"]["cash"]
    assert (cash["tp"], cash["fp"], cash["fn"]) == (1, 1, 0)


def test_nested_processor_label_containers_move_to_the_target_device():
    from collections import UserDict

    import torch

    from training.train_objects import move_to

    nested = {"labels": [UserDict({"class_labels": torch.tensor([1]), "boxes": torch.ones((1, 4))})]}
    moved = move_to(nested, "meta")
    assert moved["labels"][0]["class_labels"].device.type == "meta"
    assert moved["labels"][0]["boxes"].device.type == "meta"


def test_backbone_adapts_without_changing_batch_statistics():
    import torch

    from training.train_objects import freeze_batch_statistics, parameter_groups

    model = torch.nn.Module()
    model.backbone = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
    model.head = torch.nn.Linear(3, 1)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    groups = parameter_groups(model, 1e-4, 1e-5)
    assert [group["lr"] for group in groups] == [1e-5, 1e-4]
    model.train()
    freeze_batch_statistics(model)
    before = model.backbone[1].running_mean.clone()
    model.head(model.backbone(torch.ones(2, 3))).sum().backward()
    assert torch.equal(before, model.backbone[1].running_mean)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert sum(len(group["params"]) for group in groups) == len(list(model.parameters()))


def test_inner_calibration_can_keep_a_low_scoring_true_object_without_background():
    from training.object_data import calibrated_metrics

    cash = {"kind": "cash", "bbox_xyxy": [20, 20, 40, 40]}
    predictions = [[{**cash, "score": .06}, {"kind": "cash", "score": .04, "bbox_xyxy": [60, 60, 80, 80]}]]
    metrics = calibrated_metrics(predictions, [[cash]])
    assert metrics["threshold"]["cash"] == .05
    assert (metrics["classes"]["cash"]["tp"], metrics["classes"]["cash"]["fp"]) == (1, 0)
    assert predictions[0][0]["score"] == .06
