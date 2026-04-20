"""ONNX Runtime execution-provider selection.

Preference order:
  1. CUDA — NVIDIA GPU (server deployment). Used when onnxruntime-gpu is
     installed and a CUDA device is visible.
  2. CPU  — fallback (local dev).

CoreML is deliberately NOT in the stack: LUNetV2 uses Swin-v2 transformer
ops with unbounded reshape dims that CoreML's MLProgram builder rejects
at session construction. Local CPU is slow but correct; the fast path is
the GPU server.

Override the choice with the `UWF_FORCE_PROVIDER` env var if you want to
experiment (e.g. `UWF_FORCE_PROVIDER=CoreMLExecutionProvider`).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import onnxruntime as ort


def preferred_provider_stack() -> list:
    """Return the provider stack we'd like ORT to try, best-first."""
    available = set(ort.get_available_providers())
    stack: list = []
    if "CUDAExecutionProvider" in available:
        stack.append("CUDAExecutionProvider")
    stack.append("CPUExecutionProvider")
    return stack


def build_session(model_path: str | Path) -> ort.InferenceSession:
    """Construct an InferenceSession that uses the best available provider."""
    providers = preferred_provider_stack()
    allow_override = os.environ.get("UWF_FORCE_PROVIDER")
    if allow_override:
        providers = [allow_override, "CPUExecutionProvider"]

    try:
        sess = ort.InferenceSession(str(model_path), providers=providers)
    except Exception as exc:
        warnings.warn(
            f"ONNX session with providers {providers} failed "
            f"({type(exc).__name__}: {exc}); falling back to CPU only.",
            UserWarning,
            stacklevel=2,
        )
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return sess
