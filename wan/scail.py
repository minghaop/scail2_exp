# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math
import os
import random
import sys
import time
import types
import gc
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import load_file
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model_scail import SCAILModel
from .modules.model_scail2 import SCAIL2Model
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.lora import fuse_lora_with_diff_b
from .utils.scail_utils import (
    extract_and_compress_mask_to_latent,
    normalize_condition_segment,
)
from scail2_segments import plan_frame_segments


DIT_RESIDENT_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}

SAFETENSORS_RESIDENT_DTYPES = {
    torch.float32: "F32",
    torch.bfloat16: "BF16",
}

FP8_SCALE_SUFFIXES = (
    "scale_inv",
    ".scale_inv",
    "_scale_inv",
    "weight_scale",
    ".weight_scale",
    "_weight_scale",
    "scale_weight",
    ".scale_weight",
    "_scale_weight",
    "input_scale",
    ".input_scale",
    "_input_scale",
    "output_scale",
    ".output_scale",
    "_output_scale",
)


def _emit_pipeline_init_event(rank, stage, status, started_at=None, **details):
    fields = [
        "SCAIL2_INIT",
        f"rank={rank}",
        f"stage={stage}",
        f"status={status}",
    ]
    if started_at is not None:
        fields.append(f"elapsed_seconds={time.monotonic() - started_at:.3f}")
    fields.extend(f"{key}={value}" for key, value in details.items())
    print(" ".join(fields), file=sys.stderr, flush=True)


def resolve_dit_resident_dtype(value):
    """Return the supported torch dtype for a CLI name or torch dtype."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            return DIT_RESIDENT_DTYPES[normalized]
        except KeyError as exc:
            raise ValueError(
                "dit_resident_dtype must be one of "
                f"{sorted(DIT_RESIDENT_DTYPES)}, got {value!r}"
            ) from exc
    if value in DIT_RESIDENT_DTYPES.values():
        return value
    raise ValueError(
        "dit_resident_dtype must be torch.float32, torch.bfloat16, "
        f"'fp32', or 'bf16'; got {value!r}"
    )


def dit_resident_dtype_name(dtype):
    dtype = resolve_dit_resident_dtype(dtype)
    return next(name for name, candidate in DIT_RESIDENT_DTYPES.items()
                if candidate == dtype)


def validate_scail_checkpoint_header(path, expected_dtype):
    """Validate safetensors schema/dtypes without loading tensor payloads."""
    expected_dtype = resolve_dit_resident_dtype(expected_dtype)
    expected_storage_dtype = SAFETENSORS_RESIDENT_DTYPES[expected_dtype]
    dtype_counts = {}
    floating_tensor_count = 0
    parameter_count = 0
    mismatches = []
    quantization_markers = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        for name in handle.keys():
            tensor_slice = handle.get_slice(name)
            storage_dtype = tensor_slice.get_dtype()
            dtype_counts[storage_dtype] = dtype_counts.get(storage_dtype, 0) + 1
            lower_name = name.lower()
            if (
                "fp8" in lower_name
                or "quantized" in lower_name
                or lower_name.endswith(FP8_SCALE_SUFFIXES)
            ):
                quantization_markers.append(name)
            is_floating = storage_dtype == "BF16" or storage_dtype.startswith("F")
            if not is_floating:
                continue
            floating_tensor_count += 1
            shape = tensor_slice.get_shape()
            parameter_count += math.prod(shape)
            if storage_dtype != expected_storage_dtype:
                mismatches.append((name, storage_dtype))

    for key, value in metadata.items():
        lowered = f"{key}={value}".lower()
        if "fp8" in lowered:
            quantization_markers.append(f"metadata:{key}={value}")
        if "quantization" in key.lower() and str(value).strip().lower() not in {
            "",
            "false",
            "none",
        }:
            quantization_markers.append(f"metadata:{key}={value}")
    if quantization_markers:
        examples = ", ".join(quantization_markers[:8])
        raise TypeError(
            "SCAIL resident checkpoints must not contain FP8/quantization "
            f"scale markers: {examples}"
        )
    if not floating_tensor_count:
        raise ValueError("SCAIL checkpoint header contains no floating tensors")
    if mismatches:
        examples = ", ".join(
            f"{name}={dtype}" for name, dtype in mismatches[:8]
        )
        if len(mismatches) > 8:
            examples += f", ... ({len(mismatches)} mismatches total)"
        raise TypeError(
            "SCAIL checkpoint header dtype does not match the requested "
            f"resident dtype {expected_storage_dtype}: {examples}"
        )
    return {
        "tensor_count": sum(dtype_counts.values()),
        "floating_tensor_count": floating_tensor_count,
        "parameter_count": parameter_count,
        "dtype_counts": dtype_counts,
        "metadata": metadata,
    }


def validate_checkpoint_floating_dtypes(state_dict, expected_dtype):
    """Reject checkpoints whose floating tensors do not match the profile."""
    expected_dtype = resolve_dit_resident_dtype(expected_dtype)
    floating = []
    mismatches = []
    parameter_count = 0
    tensor_bytes = 0
    for name, tensor in state_dict.items():
        if not tensor.is_floating_point():
            continue
        floating.append(name)
        parameter_count += tensor.numel()
        tensor_bytes += tensor.numel() * tensor.element_size()
        if tensor.dtype != expected_dtype:
            mismatches.append((name, tensor.dtype))
    if not floating:
        raise ValueError("SCAIL checkpoint contains no floating tensors")
    if mismatches:
        examples = ", ".join(
            f"{name}={dtype}" for name, dtype in mismatches[:8]
        )
        if len(mismatches) > 8:
            examples += f", ... ({len(mismatches)} mismatches total)"
        raise TypeError(
            "SCAIL checkpoint floating dtype does not match the requested "
            f"resident dtype {expected_dtype}: {examples}"
        )
    return {
        "tensor_count": len(floating),
        "parameter_count": parameter_count,
        "tensor_bytes": tensor_bytes,
    }


def assert_module_floating_parameter_dtype(module, expected_dtype, stage):
    """Validate parameters without materializing an FSDP full state dict."""
    expected_dtype = resolve_dit_resident_dtype(expected_dtype)
    floating_count = 0
    mismatches = []
    for name, parameter in module.named_parameters():
        if not parameter.is_floating_point():
            continue
        floating_count += 1
        if parameter.dtype != expected_dtype:
            mismatches.append((name, parameter.dtype))
    if floating_count == 0:
        raise RuntimeError(f"SCAIL model has no floating parameters at {stage}")
    if mismatches:
        examples = ", ".join(
            f"{name}={dtype}" for name, dtype in mismatches[:8]
        )
        if len(mismatches) > 8:
            examples += f", ... ({len(mismatches)} mismatches total)"
        raise TypeError(
            f"SCAIL model parameter dtype mismatch at {stage}; expected "
            f"{expected_dtype}: {examples}"
        )
    return floating_count


class SCAIL2Pipeline:

    def __init__(
        self,
        config,
        checkpoint_dir,
        scail_safetensors_path, 
        scail_config_path="./config.json",
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
        lora_path=None,
        lora_alpha=None,
        dit_resident_dtype="fp32",
        dit_meta_load=False,
        t5_meta_load=False,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            dit_resident_dtype (`str` or `torch.dtype`, *optional*, defaults to
                "fp32"):
                Storage dtype expected in the SCAIL checkpoint and retained by
                DiT parameters. BF16 requires an offline fused checkpoint and
                does not allow runtime LoRA fusion.
            dit_meta_load (`bool`, *optional*, defaults to False):
                Build the DiT structure on the meta device and assign checkpoint
                tensors directly instead of initializing disposable parameters.
            t5_meta_load (`bool`, *optional*, defaults to False):
                Build T5 on the meta device and directly assign an mmap-backed
                weights-only checkpoint before FSDP wrapping.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        if dist.is_initialized():
            if self.rank != dist.get_rank():
                raise ValueError(
                    f"Pipeline rank {self.rank} does not match distributed rank "
                    f"{dist.get_rank()}"
                )
        elif self.rank != 0:
            raise ValueError("A non-distributed SCAIL2Pipeline must use rank 0")
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu
        self.lora_path = lora_path
        self.lora_alpha = lora_alpha
        self.dit_resident_dtype = resolve_dit_resident_dtype(
            dit_resident_dtype
        )
        self.dit_resident_dtype_name = dit_resident_dtype_name(
            self.dit_resident_dtype
        )
        self.dit_meta_load = bool(dit_meta_load)
        self.t5_meta_load = bool(t5_meta_load)
        if self.dit_resident_dtype == torch.bfloat16 and self.lora_path is not None:
            raise ValueError(
                "Runtime LoRA fusion is disabled for BF16-resident SCAIL. "
                "Use an offline fused BF16 checkpoint and omit lora_path."
            )
        # Fail on a wrong, mixed, or quantized checkpoint before loading T5,
        # VAE, CLIP, or any multi-GiB tensor payload.
        checkpoint_header_started = time.monotonic()
        _emit_pipeline_init_event(
            self.rank,
            "dit_checkpoint_header",
            "start",
            checkpoint_bytes=os.path.getsize(scail_safetensors_path),
        )
        self.scail_checkpoint_header = validate_scail_checkpoint_header(
            scail_safetensors_path, self.dit_resident_dtype
        )
        _emit_pipeline_init_event(
            self.rank,
            "dit_checkpoint_header",
            "complete",
            started_at=checkpoint_header_started,
            tensor_count=self.scail_checkpoint_header["tensor_count"],
        )
        logging.info(
            "Validated SCAIL header: %d tensors, %d floating tensors, %d "
            "parameters, dtypes=%s",
            self.scail_checkpoint_header["tensor_count"],
            self.scail_checkpoint_header["floating_tensor_count"],
            self.scail_checkpoint_header["parameter_count"],
            self.scail_checkpoint_header["dtype_counts"],
        )

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        t5_checkpoint_path = os.path.join(
            checkpoint_dir, config.t5_checkpoint
        )
        t5_started = time.monotonic()
        _emit_pipeline_init_event(
            self.rank,
            "t5_load",
            "start",
            checkpoint_bytes=os.path.getsize(t5_checkpoint_path),
            fsdp=t5_fsdp,
        )
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=t5_checkpoint_path,
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
            meta_load=self.t5_meta_load,
            init_event=partial(_emit_pipeline_init_event, self.rank),
        )
        _emit_pipeline_init_event(
            self.rank,
            "t5_load",
            "complete",
            started_at=t5_started,
            fsdp=t5_fsdp,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        vae_checkpoint_path = os.path.join(
            checkpoint_dir, config.vae_checkpoint
        )
        vae_started = time.monotonic()
        _emit_pipeline_init_event(
            self.rank,
            "vae_load",
            "start",
            checkpoint_bytes=os.path.getsize(vae_checkpoint_path),
        )
        self.vae = WanVAE(
            vae_pth=vae_checkpoint_path,
            device=self.device)
        _emit_pipeline_init_event(
            self.rank,
            "vae_load",
            "complete",
            started_at=vae_started,
        )

        clip_checkpoint_path = os.path.join(
            checkpoint_dir, config.clip_checkpoint
        )
        clip_started = time.monotonic()
        _emit_pipeline_init_event(
            self.rank,
            "clip_load",
            "start",
            checkpoint_bytes=os.path.getsize(clip_checkpoint_path),
        )
        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=clip_checkpoint_path,
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))
        _emit_pipeline_init_event(
            self.rank,
            "clip_load",
            "complete",
            started_at=clip_started,
        )

        logging.info(
            "Creating WanSCAILModel from %s with %s resident parameters",
            scail_safetensors_path,
            self.dit_resident_dtype_name,
        )
        dit_construct_started = time.monotonic()
        _emit_pipeline_init_event(
            self.rank,
            "dit_model_construct",
            "start",
            resident_dtype=self.dit_resident_dtype_name,
            load_mode="meta_assign" if self.dit_meta_load else "standard",
        )
        model_config = SCAIL2Model.load_config(scail_config_path)
        if self.dit_meta_load:
            with torch.device("meta"):
                self.model = SCAIL2Model.from_config(model_config)
            if not all(parameter.is_meta for parameter in self.model.parameters()):
                raise RuntimeError("DiT meta construction left materialized parameters")
        else:
            self.model = SCAIL2Model.from_config(model_config)
            # Cast the CPU model before loading so a BF16 checkpoint is never
            # expanded into an additional FP32 parameter copy by load_state_dict.
            if self.dit_resident_dtype != torch.float32:
                self.model.to(dtype=self.dit_resident_dtype)
        _emit_pipeline_init_event(
            self.rank,
            "dit_model_construct",
            "complete",
            started_at=dit_construct_started,
            resident_dtype=self.dit_resident_dtype_name,
            load_mode="meta_assign" if self.dit_meta_load else "standard",
        )
        state_dict = None
        checkpoint_copy_started = time.monotonic()
        _emit_pipeline_init_event(
            self.rank, "dit_checkpoint_copy", "start"
        )
        try:
            checkpoint_read_started = time.monotonic()
            _emit_pipeline_init_event(
                self.rank, "dit_checkpoint_read", "start"
            )
            state_dict = load_file(scail_safetensors_path)
            _emit_pipeline_init_event(
                self.rank,
                "dit_checkpoint_read",
                "complete",
                started_at=checkpoint_read_started,
            )
            checkpoint_validate_started = time.monotonic()
            _emit_pipeline_init_event(
                self.rank, "dit_checkpoint_validate", "start"
            )
            checkpoint_stats = validate_checkpoint_floating_dtypes(
                state_dict, self.dit_resident_dtype
            )
            _emit_pipeline_init_event(
                self.rank,
                "dit_checkpoint_validate",
                "complete",
                started_at=checkpoint_validate_started,
            )
            logging.info(
                "Validated SCAIL checkpoint: %d floating tensors, %d "
                "parameters, %.3f GiB tensor storage, dtype=%s",
                checkpoint_stats["tensor_count"],
                checkpoint_stats["parameter_count"],
                checkpoint_stats["tensor_bytes"] / 2**30,
                self.dit_resident_dtype_name,
            )
            checkpoint_assign_started = time.monotonic()
            _emit_pipeline_init_event(
                self.rank,
                "dit_checkpoint_assign",
                "start",
                assign=self.dit_meta_load,
            )
            self.model.load_state_dict(
                state_dict, strict=True, assign=self.dit_meta_load
            )
            _emit_pipeline_init_event(
                self.rank,
                "dit_checkpoint_assign",
                "complete",
                started_at=checkpoint_assign_started,
                assign=self.dit_meta_load,
            )
            if self.dit_meta_load:
                with torch.device("cpu"):
                    self.model.freqs = self.model._make_freqs()
                remaining_meta = [
                    name
                    for name, parameter in self.model.named_parameters()
                    if parameter.is_meta
                ]
                remaining_meta.extend(
                    name
                    for name, buffer in self.model.named_buffers()
                    if buffer.is_meta
                )
                if remaining_meta:
                    examples = ", ".join(remaining_meta[:8])
                    raise RuntimeError(
                        "DiT checkpoint assignment left meta tensors: " + examples
                    )
                if self.model.freqs.is_meta:
                    raise RuntimeError("DiT RoPE frequencies remain on the meta device")
        finally:
            del state_dict
            gc.collect()
        _emit_pipeline_init_event(
            self.rank,
            "dit_checkpoint_copy",
            "complete",
            started_at=checkpoint_copy_started,
        )
        assert_module_floating_parameter_dtype(
            self.model,
            self.dit_resident_dtype,
            "after checkpoint load",
        )
        if self.lora_path is not None:
            if self.lora_alpha is None:
                self.lora_alpha = 1.0
            self.fuse_lora(self.lora_path, self.lora_alpha)
            assert_module_floating_parameter_dtype(
                self.model,
                self.dit_resident_dtype,
                "after runtime LoRA fusion",
            )
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            barrier_started = time.monotonic()
            _emit_pipeline_init_event(
                self.rank, "pre_fsdp_barrier", "start"
            )
            dist.barrier()
            _emit_pipeline_init_event(
                self.rank,
                "pre_fsdp_barrier",
                "complete",
                started_at=barrier_started,
            )
        if dit_fsdp:
            fsdp_started = time.monotonic()
            _emit_pipeline_init_event(
                self.rank,
                "dit_fsdp_wrap",
                "start",
                sync_module_states=True,
            )
            self.model = shard_fn(self.model, sync_module_states=True)
            _emit_pipeline_init_event(
                self.rank,
                "dit_fsdp_wrap",
                "complete",
                started_at=fsdp_started,
                sync_module_states=True,
            )
            if os.getenv("SCAIL2_FSDP_DIAGNOSTICS") == "1":
                root_module = getattr(self.model, "module", self.model)

                def emit_block_event(stage, block_index):
                    allocated = torch.cuda.memory_allocated(self.device) / 2**20
                    reserved = torch.cuda.memory_reserved(self.device) / 2**20
                    print(
                        " ".join(
                            [
                                "SCAIL2_FSDP_DIAG",
                                f"rank={self.rank}",
                                f"stage={stage}",
                                f"block={block_index}",
                                f"allocated_mib={allocated:.1f}",
                                f"reserved_mib={reserved:.1f}",
                            ]
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

                for block_index, block in enumerate(root_module.blocks):
                    block.register_forward_pre_hook(
                        lambda _module, _args, index=block_index: emit_block_event(
                            "block_pre", index
                        )
                    )
                    block.register_forward_hook(
                        lambda _module, _args, _output, index=block_index: emit_block_event(
                            "block_post", index
                        )
                    )
        else:
            if not init_on_cpu:
                placement_started = time.monotonic()
                _emit_pipeline_init_event(
                    self.rank, "dit_device_placement", "start"
                )
                self.model.to(self.device)
                _emit_pipeline_init_event(
                    self.rank,
                    "dit_device_placement",
                    "complete",
                    started_at=placement_started,
                )
        assert_module_floating_parameter_dtype(
            self.model,
            self.dit_resident_dtype,
            "after FSDP/device placement",
        )

        self.sample_neg_prompt = config.sample_neg_prompt

    def fuse_lora(self, lora_path, alpha=1.0):
        logging.info(f"Fusing LoRA from {lora_path}, strength = {alpha}.")
        lora_state_dict = load_file(lora_path)
        report = fuse_lora_with_diff_b(
            self.model, lora_state_dict, alpha=alpha
        )
        logging.info(
            "Fused LoRA completely: %d low-rank pairs, %d weight diffs, "
            "%d bias diffs, %d/%d source tensors consumed.",
            report.lora_pairs,
            report.weight_diffs,
            report.bias_diffs,
            report.consumed_tensors,
            report.source_tensors,
        )

    def generate(self,
                 input_prompt,
                 img,
                 ref_mask_img: torch.Tensor,
                 pose_video: torch.Tensor,
                 driving_mask_video: torch.Tensor,
                 replace_flag: bool,
                 segment_len=81,
                 segment_overlap=5,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt=None,
                 seed=-1,
                 offload_model=True,
                 additional_ref_imgs: list[torch.Tensor] = None,
                 additional_ref_mask_imgs: list[torch.Tensor] = None,
                 **kwargs):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (torch.Tensor):
                Input image tensor. Shape: [3, H, W], Range: (-1, 1)
            ref_mask_img (torch.Tensor):
                Input image mask tensor. Shape: [3, H, W], Range: (-1, 1)
            pose_video (torch.Tensor):
                Input pose video. Shape: [T, C, H, W]
            driving_mask_video (torch.Tensor):
                Input driving mask tensor. Shape: [3, T, H, W], Range: (-1, 1)
            replace_flag (bool):
                True for replacement mode, False for animation mode
            segment_len (`int`, *optional*, defaults to 81):
                Number of pixel frames sampled in each segment.
            segment_overlap (`int`, *optional*, defaults to 5):
                Number of pixel frames shared with the previous segment as clean history.
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to None):
                Negative prompt for content exclusion. If not given, use ""
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, T, H, W).
        """
        if segment_len <= 0:
            raise ValueError("segment_len must be positive")
        if sampling_steps <= 0:
            raise ValueError("sampling_steps must be positive")
        if segment_overlap <= 0 or segment_overlap >= segment_len:
            raise ValueError("segment_overlap must be in (0, segment_len)")
        if (segment_overlap - 1) % self.vae_stride[0]:
            raise ValueError(
                f"segment_overlap must equal {self.vae_stride[0]}*n+1, "
                f"got {segment_overlap}"
            )

        # Long inputs can occupy several GiB per condition tensor at 704p. Keep
        # the full sequences on CPU and move only the active segment to the GPU.
        pose_video = pose_video.cpu()
        driving_mask_video = driving_mask_video.cpu()
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device) # 3 H W
        else:
            img = img.to(self.device) # 3 H W, -1 ~ 1
        ori_img = img.unsqueeze(0).to(self.device) # 1, 3, H, W

        if not isinstance(ref_mask_img, torch.Tensor):
            ref_mask_img = TF.to_tensor(ref_mask_img).sub_(0.5).div_(0.5).to(self.device) # 3 H W
        else:
            ref_mask_img = ref_mask_img.to(self.device) # 3 H W, -1 ~ 1

        if additional_ref_imgs is not None:
            if additional_ref_mask_imgs is None:
                raise ValueError('additional_ref_mask_imgs is required when additional_ref_imgs is provided.')
            if isinstance(additional_ref_imgs, torch.Tensor):
                additional_ref_imgs = [additional_ref_imgs]
            if isinstance(additional_ref_mask_imgs, torch.Tensor):
                additional_ref_mask_imgs = [additional_ref_mask_imgs]
            if len(additional_ref_imgs) != len(additional_ref_mask_imgs):
                raise ValueError(
                    'additional_ref_imgs and additional_ref_mask_imgs must have the same length, '
                    'got %d and %d.' % (len(additional_ref_imgs), len(additional_ref_mask_imgs)))
            additional_ref_imgs = [
                TF.to_tensor(u).sub_(0.5).div_(0.5).to(self.device)
                if not isinstance(u, torch.Tensor) else u.to(self.device)
                for u in additional_ref_imgs
            ]
            additional_ref_mask_imgs = [
                TF.to_tensor(u).sub_(0.5).div_(0.5).to(self.device)
                if not isinstance(u, torch.Tensor) else u.to(self.device)
                for u in additional_ref_mask_imgs
            ]
        elif additional_ref_mask_imgs is not None:
            raise ValueError('additional_ref_mask_imgs requires additional_ref_imgs.')
        num_frames = pose_video.shape[0]
        if driving_mask_video.shape[1] != num_frames:
            raise ValueError(
                f"pose_video and driving_mask_video must have the same frame count, "
                f"got {num_frames} and {driving_mask_video.shape[1]}")

        segments = plan_frame_segments(
            num_frames,
            segment_len=segment_len,
            segment_overlap=segment_overlap,
            temporal_stride=self.vae_stride[0],
        )
        if len(segments) > 1:
            logging.info(
                f"Sampling {len(segments)} segments with segment_len={segment_len}, "
                f"segment_overlap={segment_overlap}.")

        ref_latent = self.vae.encode([rearrange(ori_img, 't c h w -> c t h w')])[0]
        
        additional_ref_latent = None
        additional_ref_mask_latent_28ch = None
        if additional_ref_imgs is not None:
            additional_ref_latents = []
            additional_ref_mask_latents = []
            for additional_ref_img, additional_ref_mask_img in zip(additional_ref_imgs, additional_ref_mask_imgs):
                ori_additional_ref_img = additional_ref_img.unsqueeze(0).to(self.device)
                additional_ref_latents.append(
                    self.vae.encode([rearrange(ori_additional_ref_img, 't c h w -> c t h w')])[0]
                )
                additional_ref_mask_latents.append(
                    extract_and_compress_mask_to_latent(
                        additional_ref_mask_img.unsqueeze(1), additional_spatial_downsample=1
                    )
                )
            additional_ref_latent = torch.cat(additional_ref_latents, dim=1)
            additional_ref_mask_latent_28ch = torch.cat(additional_ref_mask_latents, dim=1)
        ref_mask_latent_28ch = extract_and_compress_mask_to_latent(
            ref_mask_img.unsqueeze(1), additional_spatial_downsample=1
        )  # (28, 1, H_lat, W_lat)
        lat_c = ref_latent.shape[0]

        # TODO: support sequence_parallel
        max_seq_len = 1e10
        # max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
        #     self.patch_size[1] * self.patch_size[2])
        # max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if n_prompt is None:
            n_prompt = ""

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        def apply_clean_history(latent, history_latent):
            if history_latent is None:
                return latent
            history_t = history_latent.shape[1]
            latent[:, :history_t] = history_latent.to(
                device=latent.device, dtype=latent.dtype
            )
            return latent

        output_segments = []
        prev_history_latent = None

        with torch.amp.autocast(
            "cuda", dtype=self.param_dtype
        ), torch.no_grad(), no_sync():

            def build_sample_scheduler():
                if sample_solver == 'unipc':
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                elif sample_solver == 'dpm++':
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                elif sample_solver == 'euler':
                    sample_scheduler = FlowMatchEulerDiscreteScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(
                        sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                else:
                    raise NotImplementedError(
                        f"Unsupported solver: {sample_solver}")
                return sample_scheduler, timesteps

            def sample_func(latent, arg_c, arg_null, history_latent):
                if offload_model:
                    self.model.to(self.device)
                latent = apply_clean_history(latent, history_latent)
                for step_index, t in enumerate(tqdm(timesteps)):
                    if os.getenv("SCAIL2_FSDP_DIAGNOSTICS") == "1":
                        allocated = torch.cuda.memory_allocated(self.device) / 2**20
                        reserved = torch.cuda.memory_reserved(self.device) / 2**20
                        print(
                            " ".join(
                                [
                                    "SCAIL2_FSDP_DIAG",
                                    f"rank={self.rank}",
                                    "stage=step_pre",
                                    f"step={step_index + 1}",
                                    f"allocated_mib={allocated:.1f}",
                                    f"reserved_mib={reserved:.1f}",
                                ]
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                    latent_model_input = [apply_clean_history(latent.to(self.device), history_latent)]
                    timestep = [t]

                    timestep = torch.stack(timestep).to(self.device)

                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0].to(
                            torch.device('cpu') if offload_model else self.device)
                    if offload_model:
                        torch.cuda.empty_cache()
                    if guide_scale <= 1.0:
                        noise_pred = noise_pred_cond
                    else:
                        noise_pred_uncond = self.model(
                            latent_model_input, t=timestep, **arg_null)[0].to(
                                torch.device('cpu') if offload_model else self.device)
                        if offload_model:
                            torch.cuda.empty_cache()
                        noise_pred = noise_pred_uncond + guide_scale * (
                            noise_pred_cond - noise_pred_uncond)

                    latent = latent.to(
                        torch.device('cpu') if offload_model else self.device)

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latent = apply_clean_history(temp_x0.squeeze(0), history_latent)

                    x0 = [latent.to(self.device)]
                    del latent_model_input, timestep

                if offload_model:
                    self.model.cpu()
                    torch.cuda.empty_cache()

                # Return the compact final latent first. Per-segment diffusion
                # tensors are released before rank 0 enters the replicated VAE
                # decoder, so their allocations do not overlap its memory peak.
                return x0[0]

            for seg_idx, segment in enumerate(segments):
                seg_start = segment.start
                seg_valid_end = segment.valid_end
                logging.info(
                    f"Processing segment {seg_idx + 1}/{len(segments)}: "
                    f"frames [{seg_start}, {seg_valid_end}), "
                    f"padded_length={segment.padded_frames}")
                sample_scheduler, timesteps = build_sample_scheduler()

                pose_segment = pose_video[seg_start:seg_valid_end]
                pad_frames = segment.padded_frames - segment.valid_frames
                if pad_frames:
                    pose_segment = torch.cat(
                        [pose_segment, pose_segment[-1:].expand(pad_frames, -1, -1, -1)],
                        dim=0,
                    )
                pose_segment = normalize_condition_segment(
                    pose_segment, self.device
                )
                smpl_render_video = F.interpolate(
                    pose_segment, scale_factor=0.5, mode='bilinear', align_corners=False)
                pose_latent = self.vae.encode([rearrange(smpl_render_video, 't c h w -> c t h w')])[0]

                lat_t = pose_latent.shape[1]
                _, lat_h, lat_w = ref_latent.shape[1:]

                null_noisy_mask = torch.zeros(
                    ref_mask_latent_28ch.shape[0], lat_t, lat_h, lat_w,
                    device=self.device, dtype=ref_mask_latent_28ch.dtype)
                ref_masks = torch.cat([ref_mask_latent_28ch, null_noisy_mask], dim=1)

                driving_mask_segment = driving_mask_video[:, seg_start:seg_valid_end]
                if pad_frames:
                    driving_mask_segment = torch.cat(
                        [
                            driving_mask_segment,
                            driving_mask_segment[:, -1:].expand(-1, pad_frames, -1, -1),
                        ],
                        dim=1,
                    )
                driving_mask_segment = normalize_condition_segment(
                    driving_mask_segment, self.device
                )
                driving_mask_segment = F.interpolate(
                    driving_mask_segment, scale_factor=0.5, mode='bilinear', align_corners=False)
                driving_masks = extract_and_compress_mask_to_latent(
                    driving_mask_segment, additional_spatial_downsample=1
                )

                history_latent = prev_history_latent
                prev_history_latent = None
                history_mask = None
                if seg_idx > 0:
                    if history_latent is None:
                        raise RuntimeError("Missing previous segment history latent.")
                    history_t = history_latent.shape[1]
                    if history_t >= lat_t:
                        raise RuntimeError(
                            f"History latent has {history_t} frames, but the "
                            f"current segment has only {lat_t}"
                        )
                    history_mask = torch.zeros(
                        4, lat_t, lat_h, lat_w, device=self.device, dtype=torch.float32)
                    history_mask[:, :history_t] = 1
                    logging.info(
                        f"Using {segment_overlap} clean history frames "
                        f"({history_t} latent frames).")

                noise = torch.randn(
                    lat_c,
                    lat_t,
                    lat_h,
                    lat_w,
                    dtype=torch.float32,
                    generator=seed_g,
                    device=self.device)

                arg_c = {
                    'context': [context[0]],
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'ref_latents': [ref_latent],
                    'ref_masks': [ref_masks],
                    'pose_latents': [pose_latent],
                    'driving_masks': [driving_masks],
                    'history_mask': [history_mask] if history_mask is not None else None,
                    'replace_flag': replace_flag,
                    'additional_ref_latents': None if additional_ref_latent is None else [additional_ref_latent],
                    'additional_ref_masks': None if additional_ref_mask_latent_28ch is None else [additional_ref_mask_latent_28ch],
                }

                arg_null = {
                    'context': context_null,
                    'clip_fea': clip_context,
                    'seq_len': max_seq_len,
                    'ref_latents': [ref_latent],
                    'ref_masks': [ref_masks],
                    'pose_latents': [pose_latent],
                    'driving_masks': [driving_masks],
                    'history_mask': [history_mask] if history_mask is not None else None,
                    'replace_flag': replace_flag,
                    'additional_ref_latents': None if additional_ref_latent is None else [additional_ref_latent],
                    'additional_ref_masks': None if additional_ref_mask_latent_28ch is None else [additional_ref_mask_latent_28ch],
                }

                if offload_model:
                    torch.cuda.empty_cache()

                final_latent = sample_func(
                    noise, arg_c, arg_null, history_latent
                )

                del (
                    noise,
                    pose_segment,
                    smpl_render_video,
                    pose_latent,
                    null_noisy_mask,
                    ref_masks,
                    driving_mask_segment,
                    driving_masks,
                    arg_c,
                    arg_null,
                    sample_scheduler,
                    timesteps,
                )
                if history_latent is not None:
                    del history_latent, history_mask
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                # The VAE is replicated rather than sharded. Decode only on
                # rank 0 after clearing the diffusion inputs. Other ranks can
                # release their copy of the final latent immediately and wait
                # for the compact history-latent broadcast below.
                next_history_pixel = None
                if self.rank == 0:
                    videos = self.vae.decode([final_latent])
                    segment_video = videos[0]
                    output_segments.append(
                        segment_video[
                            :,
                            segment.overlap:segment.valid_frames,
                        ].to(device='cpu', dtype=torch.float32)
                    )
                    if seg_idx < len(segments) - 1:
                        # Keep the overlap on CPU in the decoder's FP32 output
                        # dtype. It is quantized to the DiT BF16 dtype only when
                        # uploaded for VAE encoding, matching the test_1 path.
                        next_history_pixel = (
                            segment_video[:, -segment_overlap:]
                            .detach()
                            .to(device='cpu', dtype=torch.float32)
                            .contiguous()
                        )
                    del videos, segment_video
                del final_latent
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                if seg_idx < len(segments) - 1:
                    if self.rank == 0:
                        if next_history_pixel is None:
                            raise RuntimeError("Missing next-segment history pixels.")
                        next_history_pixel_gpu = next_history_pixel.to(
                            self.device, dtype=self.param_dtype
                        )
                        next_history_latent = self.vae.encode(
                            [next_history_pixel_gpu]
                        )[0].contiguous()
                        del next_history_pixel_gpu, next_history_pixel
                    if dist.is_initialized():
                        if self.rank == 0:
                            history_shape = torch.tensor(
                                next_history_latent.shape,
                                device=self.device,
                                dtype=torch.int64,
                            )
                        else:
                            history_shape = torch.empty(
                                4, device=self.device, dtype=torch.int64
                            )
                        dist.broadcast(history_shape, src=0)
                        history_shape_tuple = tuple(
                            int(value) for value in history_shape.cpu().tolist()
                        )
                        del history_shape
                        if self.rank != 0:
                            next_history_latent = torch.empty(
                                history_shape_tuple,
                                device=self.device,
                                dtype=torch.float32,
                            )
                    else:
                        history_shape_tuple = tuple(next_history_latent.shape)

                    expected_history_shape = (
                        lat_c,
                        (segment_overlap - 1) // self.vae_stride[0] + 1,
                        lat_h,
                        lat_w,
                    )
                    if history_shape_tuple != expected_history_shape:
                        raise RuntimeError(
                            f"History latent shape is {history_shape_tuple}, "
                            f"expected {expected_history_shape}"
                        )
                    if dist.is_initialized():
                        dist.broadcast(next_history_latent, src=0)
                    if next_history_latent.dtype != torch.float32:
                        raise RuntimeError(
                            "History latent must be FP32, got "
                            f"{next_history_latent.dtype}"
                        )
                    if not bool(torch.isfinite(next_history_latent).all()):
                        raise RuntimeError("History latent contains NaN or Inf values.")
                    prev_history_latent = next_history_latent
                    del next_history_latent
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()

        if self.rank == 0:
            video = torch.cat(output_segments, dim=1)
            if video.shape[1] != num_frames:
                raise RuntimeError(
                    f"Generated {video.shape[1]} frames for {num_frames} input frames"
                )
            return video
        return None
