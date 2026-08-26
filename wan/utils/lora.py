from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Callable

import torch


LORA_DOWN_SUFFIX = ".lora_down.weight"
LORA_UP_SUFFIX = ".lora_up.weight"
WEIGHT_DIFF_SUFFIX = ".diff"
BIAS_DIFF_SUFFIX = ".diff_b"
DIFFUSION_MODEL_PREFIX = "diffusion_model."


@dataclass(frozen=True)
class LoraFusionReport:
    lora_pairs: int
    weight_diffs: int
    bias_diffs: int
    consumed_tensors: int
    source_tensors: int
    target_tensors: int


@dataclass(frozen=True)
class _FusionPlan:
    lora_pairs: tuple[tuple[str, str, str], ...]
    weight_diffs: tuple[tuple[str, str], ...]
    bias_diffs: tuple[tuple[str, str], ...]
    report: LoraFusionReport


FusionProgress = Callable[[str, int, int, str], None]


def _target_prefix(source_prefix: str) -> str:
    if source_prefix.startswith(DIFFUSION_MODEL_PREFIX):
        return source_prefix[len(DIFFUSION_MODEL_PREFIX) :]
    return source_prefix


def _require_target(
    model_state: Mapping[str, torch.Tensor],
    target_key: str,
    source_key: str,
) -> torch.Tensor:
    if target_key not in model_state:
        raise KeyError(
            f"LoRA tensor {source_key!r} maps to missing model tensor "
            f"{target_key!r}"
        )
    target = model_state[target_key]
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Model state {target_key!r} is not a tensor")
    if not target.is_floating_point():
        raise TypeError(f"Model state {target_key!r} is not floating point")
    return target


def _require_source(
    lora_state: Mapping[str, torch.Tensor], source_key: str
) -> torch.Tensor:
    source = lora_state[source_key]
    if not isinstance(source, torch.Tensor):
        raise TypeError(f"LoRA state {source_key!r} is not a tensor")
    if not source.is_floating_point():
        raise TypeError(f"LoRA state {source_key!r} is not floating point")
    if not bool(torch.isfinite(source).all()):
        raise ValueError(f"LoRA state {source_key!r} contains NaN or Inf")
    return source


def plan_lora_fusion(
    model_state: Mapping[str, torch.Tensor],
    lora_state: Mapping[str, torch.Tensor],
) -> _FusionPlan:
    """Validate every LoRA tensor and build a deterministic fusion plan.

    Lightx2v checkpoints contain ordinary low-rank weight updates as well as
    standalone ``.diff`` and ``.diff_b`` tensors. Validation is deliberately
    strict so a new or partially incompatible checkpoint cannot silently lose
    learned deltas.
    """

    source_keys = set(lora_state)
    if not source_keys:
        raise ValueError("LoRA state dict is empty")

    consumed: set[str] = set()
    pair_targets: dict[str, str] = {}
    weight_diff_targets: dict[str, str] = {}
    bias_diff_targets: dict[str, str] = {}
    lora_pairs: list[tuple[str, str, str]] = []
    weight_diffs: list[tuple[str, str]] = []
    bias_diffs: list[tuple[str, str]] = []

    down_keys = sorted(key for key in source_keys if key.endswith(LORA_DOWN_SUFFIX))
    for down_key in down_keys:
        source_prefix = down_key[: -len(LORA_DOWN_SUFFIX)]
        up_key = source_prefix + LORA_UP_SUFFIX
        if up_key not in lora_state:
            raise KeyError(f"Missing matching LoRA up tensor for {down_key!r}: {up_key!r}")

        target_key = _target_prefix(source_prefix) + ".weight"
        if target_key in pair_targets:
            raise ValueError(
                f"LoRA pairs {pair_targets[target_key]!r} and {down_key!r} both "
                f"map to {target_key!r}"
            )
        target = _require_target(model_state, target_key, down_key)
        down = _require_source(lora_state, down_key)
        up = _require_source(lora_state, up_key)
        if down.ndim != 2 or up.ndim != 2:
            raise ValueError(
                f"LoRA matrices must be 2D: {down_key}={tuple(down.shape)}, "
                f"{up_key}={tuple(up.shape)}"
            )
        if down.shape[0] == 0:
            raise ValueError(f"LoRA rank must be positive for {source_prefix!r}")
        if up.shape[1] != down.shape[0]:
            raise ValueError(
                f"LoRA rank mismatch for {source_prefix!r}: "
                f"up={tuple(up.shape)}, down={tuple(down.shape)}"
            )
        delta_shape = (up.shape[0], down.shape[1])
        if tuple(target.shape) != delta_shape:
            raise ValueError(
                f"LoRA delta for {source_prefix!r} has shape {delta_shape}, but "
                f"model tensor {target_key!r} has shape {tuple(target.shape)}"
            )

        pair_targets[target_key] = down_key
        consumed.update((down_key, up_key))
        lora_pairs.append((down_key, up_key, target_key))

    orphan_up_keys = sorted(
        key
        for key in source_keys
        if key.endswith(LORA_UP_SUFFIX) and key not in consumed
    )
    if orphan_up_keys:
        raise ValueError(
            "LoRA up tensors have no matching down tensor: "
            + ", ".join(orphan_up_keys[:8])
        )

    for diff_key in sorted(
        key for key in source_keys if key.endswith(WEIGHT_DIFF_SUFFIX)
    ):
        source_prefix = diff_key[: -len(WEIGHT_DIFF_SUFFIX)]
        target_key = _target_prefix(source_prefix) + ".weight"
        if target_key in weight_diff_targets:
            raise ValueError(
                f"Weight diffs {weight_diff_targets[target_key]!r} and "
                f"{diff_key!r} both map to {target_key!r}"
            )
        target = _require_target(model_state, target_key, diff_key)
        diff = _require_source(lora_state, diff_key)
        if tuple(diff.shape) != tuple(target.shape):
            raise ValueError(
                f"Weight diff {diff_key!r} has shape {tuple(diff.shape)}, but "
                f"model tensor {target_key!r} has shape {tuple(target.shape)}"
            )
        weight_diff_targets[target_key] = diff_key
        consumed.add(diff_key)
        weight_diffs.append((diff_key, target_key))

    for diff_key in sorted(
        key for key in source_keys if key.endswith(BIAS_DIFF_SUFFIX)
    ):
        source_prefix = diff_key[: -len(BIAS_DIFF_SUFFIX)]
        target_key = _target_prefix(source_prefix) + ".bias"
        if target_key in bias_diff_targets:
            raise ValueError(
                f"Bias diffs {bias_diff_targets[target_key]!r} and {diff_key!r} "
                f"both map to {target_key!r}"
            )
        target = _require_target(model_state, target_key, diff_key)
        diff = _require_source(lora_state, diff_key)
        if tuple(diff.shape) != tuple(target.shape):
            raise ValueError(
                f"Bias diff {diff_key!r} has shape {tuple(diff.shape)}, but model "
                f"tensor {target_key!r} has shape {tuple(target.shape)}"
            )
        bias_diff_targets[target_key] = diff_key
        consumed.add(diff_key)
        bias_diffs.append((diff_key, target_key))

    unconsumed = sorted(source_keys - consumed)
    if unconsumed:
        raise ValueError(
            f"Unsupported or unconsumed LoRA tensors ({len(unconsumed)}): "
            + ", ".join(unconsumed[:8])
        )

    target_tensors = set(pair_targets) | set(weight_diff_targets) | set(bias_diff_targets)
    report = LoraFusionReport(
        lora_pairs=len(lora_pairs),
        weight_diffs=len(weight_diffs),
        bias_diffs=len(bias_diffs),
        consumed_tensors=len(consumed),
        source_tensors=len(source_keys),
        target_tensors=len(target_tensors),
    )
    return _FusionPlan(
        lora_pairs=tuple(lora_pairs),
        weight_diffs=tuple(weight_diffs),
        bias_diffs=tuple(bias_diffs),
        report=report,
    )


@torch.no_grad()
def apply_lora_fusion_plan(
    model_state: MutableMapping[str, torch.Tensor],
    lora_state: Mapping[str, torch.Tensor],
    plan: _FusionPlan,
    alpha: float = 1.0,
    compute_dtype: torch.dtype = torch.float32,
    progress: FusionProgress | None = None,
) -> LoraFusionReport:
    """Apply a previously validated fusion plan in place."""

    alpha = float(alpha)
    if not math.isfinite(alpha):
        raise ValueError(f"LoRA alpha must be finite, got {alpha}")
    if not torch.empty((), dtype=compute_dtype).is_floating_point():
        raise TypeError(f"LoRA compute dtype must be floating point, got {compute_dtype}")

    pair_total = len(plan.lora_pairs)
    for index, (down_key, up_key, target_key) in enumerate(
        plan.lora_pairs, start=1
    ):
        target = model_state[target_key]
        down = lora_state[down_key].to(
            device=target.device, dtype=compute_dtype
        )
        up = lora_state[up_key].to(device=target.device, dtype=compute_dtype)
        delta = torch.matmul(up, down)
        target.add_(delta.to(dtype=target.dtype), alpha=alpha)
        del down, up, delta
        if progress is not None:
            progress("low-rank", index, pair_total, target_key)

    weight_diff_total = len(plan.weight_diffs)
    for index, (diff_key, target_key) in enumerate(plan.weight_diffs, start=1):
        target = model_state[target_key]
        target.add_(
            lora_state[diff_key].to(device=target.device, dtype=target.dtype),
            alpha=alpha,
        )
        if progress is not None:
            progress("weight-diff", index, weight_diff_total, target_key)

    bias_diff_total = len(plan.bias_diffs)
    for index, (diff_key, target_key) in enumerate(plan.bias_diffs, start=1):
        target = model_state[target_key]
        target.add_(
            lora_state[diff_key].to(device=target.device, dtype=target.dtype),
            alpha=alpha,
        )
        if progress is not None:
            progress("bias-diff", index, bias_diff_total, target_key)

    return plan.report


def fuse_lora_state_dict(
    model_state: MutableMapping[str, torch.Tensor],
    lora_state: Mapping[str, torch.Tensor],
    alpha: float = 1.0,
    compute_dtype: torch.dtype = torch.float32,
    progress: FusionProgress | None = None,
) -> LoraFusionReport:
    """Validate and fuse all supported Lightx2v deltas in place."""

    plan = plan_lora_fusion(model_state, lora_state)
    return apply_lora_fusion_plan(
        model_state,
        lora_state,
        plan,
        alpha=alpha,
        compute_dtype=compute_dtype,
        progress=progress,
    )


def fuse_lora_with_diff_b(
    model: torch.nn.Module,
    lora_state_dict: Mapping[str, torch.Tensor],
    alpha: float = 1.0,
) -> LoraFusionReport:
    """Backward-compatible model wrapper for complete Lightx2v fusion."""

    return fuse_lora_state_dict(
        model.state_dict(),
        lora_state_dict,
        alpha=alpha,
        compute_dtype=torch.float32,
    )
