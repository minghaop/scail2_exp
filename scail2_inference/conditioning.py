"""Versioned, validated T5/CLIP conditioning artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Mapping


SCHEMA_VERSION = "scail2-conditioning-v1"
TEXT_CONTEXT = "text_context"
NEGATIVE_CONTEXT = "negative_context"
CLIP_CONTEXT = "clip_context"
TENSOR_NAMES = frozenset((TEXT_CONTEXT, NEGATIVE_CONTEXT, CLIP_CONTEXT))


def _sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_fields(prefix: str, path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        f"{prefix}_path": str(resolved),
        f"{prefix}_bytes": str(stat.st_size),
        f"{prefix}_mtime_ns": str(stat.st_mtime_ns),
    }


def expected_metadata(
    *,
    prompt: str,
    negative_prompt: str,
    reference_image: Path,
    target_width: int,
    target_height: int,
    t5_checkpoint: Path,
    clip_checkpoint: Path,
) -> dict[str, str]:
    """Build the identity fields that bind a cache to its source inputs."""
    image = reference_image.resolve(strict=True)
    metadata = {
        "schema": SCHEMA_VERSION,
        "prompt_sha256": _sha256_bytes(prompt),
        "negative_prompt_sha256": _sha256_bytes(negative_prompt),
        "reference_image_path": str(image),
        "reference_image_sha256": _sha256_file(image),
        "target_width": str(target_width),
        "target_height": str(target_height),
    }
    metadata.update(_checkpoint_fields("t5_checkpoint", t5_checkpoint))
    metadata.update(_checkpoint_fields("clip_checkpoint", clip_checkpoint))
    return metadata


def _validate_tensors(tensors: Mapping[str, object]) -> None:
    import torch

    if set(tensors) != TENSOR_NAMES:
        raise ValueError(
            "Conditioning tensor names mismatch: "
            f"expected {sorted(TENSOR_NAMES)}, got {sorted(tensors)}"
        )
    text = tensors[TEXT_CONTEXT]
    negative = tensors[NEGATIVE_CONTEXT]
    clip = tensors[CLIP_CONTEXT]
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
        raise TypeError("Conditioning values must all be torch tensors")
    if text.ndim != 2 or text.shape[0] <= 0 or text.shape[1] != 4096:
        raise ValueError(f"Invalid text context shape: {tuple(text.shape)}")
    if negative.ndim != 2 or negative.shape[0] <= 0 or negative.shape[1] != 4096:
        raise ValueError(f"Invalid negative context shape: {tuple(negative.shape)}")
    if clip.ndim != 3 or clip.shape[0] != 1 or clip.shape[-1] != 1280:
        raise ValueError(f"Invalid CLIP context shape: {tuple(clip.shape)}")
    if text.dtype != torch.bfloat16 or negative.dtype != torch.bfloat16:
        raise ValueError(
            "T5 contexts must be bfloat16, got "
            f"{text.dtype} and {negative.dtype}"
        )
    if clip.dtype != torch.float16:
        raise ValueError(f"CLIP context must be float16, got {clip.dtype}")
    if any(tensor.is_cuda or tensor.is_meta for tensor in tensors.values()):
        raise ValueError("Conditioning tensors must be materialized on CPU")


def save_conditioning(
    path: Path,
    tensors: Mapping[str, object],
    metadata: Mapping[str, str],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically save a validated conditioning artifact."""
    from safetensors.torch import save_file

    target = path.resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"Conditioning artifact already exists: {target}")
    _validate_tensors(tensors)
    if metadata.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"Invalid conditioning schema: {metadata.get('schema')!r}")
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


def load_conditioning(
    path: Path,
    expected: Mapping[str, str],
) -> dict[str, object]:
    """Load a cache only when all identity fields and tensor contracts match."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    source = path.resolve(strict=True)
    with safe_open(source, framework="pt", device="cpu") as handle:
        actual = handle.metadata() or {}
        keys = set(handle.keys())
    if keys != TENSOR_NAMES:
        raise ValueError(
            "Conditioning tensor names mismatch: "
            f"expected {sorted(TENSOR_NAMES)}, got {sorted(keys)}"
        )
    mismatches = [
        f"{key}: expected {value!r}, got {actual.get(key)!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]
    if mismatches:
        raise ValueError("Conditioning metadata mismatch: " + "; ".join(mismatches))
    tensors = load_file(str(source), device="cpu")
    _validate_tensors(tensors)
    return dict(tensors)
