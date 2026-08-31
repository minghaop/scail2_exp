"""Versioned, validated T5-only conditioning artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Mapping


SCHEMA_VERSION = "scail2-t5-cache-v1"
TEXT_CONTEXT = "text_context"
NEGATIVE_CONTEXT = "negative_context"
TENSOR_NAMES = frozenset((TEXT_CONTEXT, NEGATIVE_CONTEXT))


def _sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _checkpoint_fields(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "t5_checkpoint_name": resolved.name,
        "t5_checkpoint_bytes": str(stat.st_size),
        "t5_checkpoint_mtime_ns": str(stat.st_mtime_ns),
    }


def expected_t5_metadata(
    *,
    profile: str,
    text_len: int,
    t5_checkpoint: Path,
) -> dict[str, str]:
    """Build runtime-verifiable identity fields for a T5 cache."""
    metadata = {
        "schema": SCHEMA_VERSION,
        "profile": profile,
        "text_len": str(text_len),
    }
    metadata.update(_checkpoint_fields(t5_checkpoint))
    return metadata


def build_t5_metadata(
    *,
    prompt: str,
    negative_prompt: str,
    profile: str,
    text_len: int,
    t5_checkpoint: Path,
) -> dict[str, str]:
    """Build cache metadata, including non-runtime prompt provenance."""
    metadata = expected_t5_metadata(
        profile=profile,
        text_len=text_len,
        t5_checkpoint=t5_checkpoint,
    )
    metadata.update(
        {
            "prompt": prompt,
            "prompt_sha256": _sha256_bytes(prompt),
            "negative_prompt_sha256": _sha256_bytes(negative_prompt),
        }
    )
    return metadata


def _validate_metadata(metadata: Mapping[str, str]) -> None:
    if metadata.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"Invalid T5 cache schema: {metadata.get('schema')!r}")
    prompt = metadata.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("T5 cache metadata is missing prompt")
    for field_name in ("prompt_sha256", "negative_prompt_sha256"):
        if not metadata.get(field_name):
            raise ValueError(f"T5 cache metadata is missing {field_name}")
    if metadata["prompt_sha256"] != _sha256_bytes(prompt):
        raise ValueError("T5 cache prompt does not match prompt_sha256")


def _validate_tensors(tensors: Mapping[str, object]) -> None:
    import torch

    if set(tensors) != TENSOR_NAMES:
        raise ValueError(
            "Conditioning tensor names mismatch: "
            f"expected {sorted(TENSOR_NAMES)}, got {sorted(tensors)}"
        )
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
        raise TypeError("Conditioning values must all be torch tensors")
    text = tensors[TEXT_CONTEXT]
    negative = tensors[NEGATIVE_CONTEXT]
    if text.ndim != 2 or text.shape[0] <= 0 or text.shape[1] != 4096:
        raise ValueError(f"Invalid text context shape: {tuple(text.shape)}")
    if negative.ndim != 2 or negative.shape[0] <= 0 or negative.shape[1] != 4096:
        raise ValueError(f"Invalid negative context shape: {tuple(negative.shape)}")
    if text.dtype != torch.bfloat16 or negative.dtype != torch.bfloat16:
        raise ValueError(
            "T5 contexts must be bfloat16, got "
            f"{text.dtype} and {negative.dtype}"
        )
    if any(tensor.is_cuda or tensor.is_meta for tensor in tensors.values()):
        raise ValueError("Conditioning tensors must be materialized on CPU")


def save_t5_cache(
    path: Path,
    tensors: Mapping[str, object],
    metadata: Mapping[str, str],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically save a validated T5 cache."""
    from safetensors.torch import save_file

    target = path.resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"T5 cache already exists: {target}")
    _validate_tensors(tensors)
    _validate_metadata(metadata)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.inprogress-{os.getpid()}")
    try:
        save_file(
            {name: tensor.contiguous() for name, tensor in tensors.items()},
            str(temp),
            metadata=dict(metadata),
        )
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def load_t5_cache(
    path: Path,
    expected: Mapping[str, str],
) -> dict[str, object]:
    """Load a T5 cache only when identity and tensor contracts match."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    source = path.resolve(strict=True)
    with safe_open(source, framework="pt", device="cpu") as handle:
        actual = handle.metadata() or {}
        keys = set(handle.keys())
    _validate_metadata(actual)
    if keys != TENSOR_NAMES:
        raise ValueError(
            "T5 cache tensor names mismatch: "
            f"expected {sorted(TENSOR_NAMES)}, got {sorted(keys)}"
        )
    mismatches = [
        f"{key}: expected {value!r}, got {actual.get(key)!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]
    if mismatches:
        raise ValueError("T5 cache metadata mismatch: " + "; ".join(mismatches))
    tensors = load_file(str(source), device="cpu")
    _validate_tensors(tensors)
    return dict(tensors)
