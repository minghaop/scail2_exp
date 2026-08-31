#!/home/panminghao/miniconda3/envs/scail2-single-gpu/bin/python
"""Precompute a validated T5 cache for one inference prompt."""

from __future__ import annotations

import argparse
import codecs
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from run_fsdp_experiment import (
    CHECKPOINT_DIR,
    CONDA_ENV,
    DEFAULT_PROMPT,
    PROFILE_NAME,
    PYTHON_BIN,
    ROOT,
    TEST_CASE,
    TimestampedLogWriter,
)


DEFAULT_OUTPUT = ROOT / "experiment_cache/t5" / f"{TEST_CASE}.safetensors"
ALLOWED_PHYSICAL_GPUS = frozenset(("0", "1", "2", "3", "6", "7"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--physical-gpu", default="2")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.physical_gpu not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError(
            "Conditioning preprocessing may use only physical GPUs "
            f"0,1,2,3,6,7; got {args.physical_gpu!r}"
        )
    if not args.prompt.strip():
        raise ValueError("Prompt cannot be empty")
    if not CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(f"Invalid checkpoint directory: {CHECKPOINT_DIR}")


def launch_worker(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = ROOT / "experiment_logs/conditioning" / f"{TEST_CASE}-{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON_BIN),
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--physical-gpu",
        args.physical_gpu,
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--output",
        str(args.output.resolve()),
    ]
    if args.overwrite:
        command.append("--overwrite")
    env = {
        "CUDA_VISIBLE_DEVICES": args.physical_gpu,
        "HOME": "/home/panminghao",
        "LANG": "C.UTF-8",
        "PATH": f"{CONDA_ENV / 'bin'}:/usr/bin:/bin",
        "TMPDIR": "/tmp",
    }
    header = (
        f"Log file: {log_path}\n"
        f"Preparing T5 cache on physical GPU {args.physical_gpu}\n"
        + " ".join(command)
        + "\n"
    )
    sys.stdout.write(header)
    sys.stdout.flush()
    with log_path.open("wb") as log_file:
        timestamped_log = TimestampedLogWriter(log_file)
        timestamped_log.write(header)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        if process.stdout is None:
            raise RuntimeError("Failed to capture preprocessing output")
        for chunk in iter(lambda: process.stdout.read(65536), b""):
            text = decoder.decode(chunk)
            timestamped_log.write(text)
            sys.stdout.write(text)
            sys.stdout.flush()
        tail = decoder.decode(b"", final=True)
        if tail:
            timestamped_log.write(tail)
            sys.stdout.write(tail)
        timestamped_log.close()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def run_worker(args: argparse.Namespace) -> None:
    started = time.monotonic()
    import torch

    from scail2_inference.conditioning import (
        NEGATIVE_CONTEXT,
        TEXT_CONTEXT,
        build_t5_metadata,
        save_t5_cache,
    )
    from scail2_inference.contracts import ProductionProfile
    from wan.configs import SCAIL_CONFIGS
    from wan.modules.t5 import T5EncoderModel

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != args.physical_gpu:
        raise RuntimeError(
            f"Expected CUDA_VISIBLE_DEVICES={args.physical_gpu}, got {visible!r}"
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    profile = ProductionProfile.from_name(PROFILE_NAME)
    cfg = SCAIL_CONFIGS[profile.model.upper()]
    t5_checkpoint = CHECKPOINT_DIR / cfg.t5_checkpoint

    torch.cuda.reset_peak_memory_stats(device)
    stage_started = time.monotonic()
    print("SCAIL2_PREPROCESS stage=t5 status=start", flush=True)
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=device,
        checkpoint_path=str(t5_checkpoint),
        tokenizer_path=str(CHECKPOINT_DIR / cfg.t5_tokenizer),
        shard_fn=None,
        meta_load=True,
    )
    with torch.inference_mode():
        text_context = text_encoder([args.prompt], device)[0].detach().cpu().contiguous()
        negative_context = text_encoder([args.negative_prompt], device)[0].detach().cpu().contiguous()
    torch.cuda.synchronize(device)
    print(
        "SCAIL2_PREPROCESS stage=t5 status=complete "
        f"elapsed_seconds={time.monotonic() - stage_started:.3f} "
        f"peak_allocated_mib={torch.cuda.max_memory_allocated(device) / 2**20:.1f} "
        f"text_shape={tuple(text_context.shape)} "
        f"negative_shape={tuple(negative_context.shape)}",
        flush=True,
    )
    metadata = build_t5_metadata(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        profile=profile.name,
        text_len=cfg.text_len,
        t5_checkpoint=t5_checkpoint,
    )
    output = args.output.resolve()
    save_t5_cache(
        output,
        {
            TEXT_CONTEXT: text_context,
            NEGATIVE_CONTEXT: negative_context,
        },
        metadata,
        overwrite=args.overwrite,
    )
    summary = {
        "status": "success",
        "output": str(output),
        "bytes": output.stat().st_size,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "text_shape": list(text_context.shape),
        "negative_shape": list(negative_context.shape),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    args.prompt = args.prompt.strip()
    validate_args(args)
    if args.worker:
        run_worker(args)
    else:
        launch_worker(args)


if __name__ == "__main__":
    main()
