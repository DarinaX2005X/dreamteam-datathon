"""Explicit model precision must be supported and reproducible."""

import pytest
from navidu.perception.temporal import resolve_dtype


@pytest.mark.parametrize(("device", "requested", "supported", "expected"), [
    ("cpu", "auto", False, "fp32"),
    ("cpu", "fp32", False, "fp32"),
    ("cuda", "auto", False, "fp16"),
    ("cuda", "fp16", False, "fp16"),
    ("cuda", "fp32", False, "fp32"),
    ("cuda", "bf16", True, "bf16"),
])
def test_supported_precision(device, requested, supported, expected):
    assert resolve_dtype(device, requested, bf16_supported=supported) == expected


@pytest.mark.parametrize(("device", "requested", "supported"), [
    ("cpu", "fp16", False),
    ("cpu", "bf16", True),
    ("cuda", "bf16", False),
    ("cuda", "float16", True),
])
def test_unsupported_precision_is_not_silently_substituted(device, requested, supported):
    with pytest.raises(RuntimeError, match="INVALID_CONFIG"):
        resolve_dtype(device, requested, bf16_supported=supported)
