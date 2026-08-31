#!/home/panminghao/miniconda3/envs/scail2-single-gpu/bin/python
"""Run the production single-GPU pipeline without the external worker service."""

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

from experiment_support import (
    ALLOWED_PHYSICAL_GPUS,
    CHECKPOINT_DIR,
    CONDA_ENV,
    DIT_CHECKPOINT,
    PROFILE_NAME,
    PYTHON_BIN,
    ROOT,
    TEST_CASE,
    TimestampedLogWriter,
    resolve_job_paths,
    validate_media_contract,
)

DEFAULT_T5_CACHE = ROOT / "experiment_cache/t5" / f"{TEST_CASE}.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physical-gpu", default="2")
    parser.add_argument("--t5-cache", type=Path, default=DEFAULT_T5_CACHE)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--init-only", action="store_true")
    mode.add_argument("--full-inference", action="store_true")
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_payload(args: argparse.Namespace) -> dict[str, object]:
    gpu = args.physical_gpu.strip()
    if gpu not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError(f"Physical GPU {gpu!r} is not allowed")
    if args.repeat_count <= 0 or (args.repeat_count != 1 and not args.full_inference):
        raise ValueError("--repeat-count requires full inference and must be positive")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (args.output or ROOT / "experiment_outputs/single_gpu" / f"{TEST_CASE}-{timestamp}.mp4").resolve()
    return {
        **resolve_job_paths(),
        "job_id": args.job_id or f"single-{TEST_CASE}-{output.stem}",
        "seed": args.seed,
        "physical_gpu": gpu,
        "t5_cache": args.t5_cache.resolve(),
        "output": output,
        "output_audio": "driving" if args.full_inference else "none",
        "log": ROOT / "experiment_logs/single_gpu" / f"{output.stem}.log",
    }


def validate_paths(payload: dict[str, object]) -> None:
    for key in ("reference_image", "reference_mask", "driving_video", "driving_mask", "t5_cache"):
        path = Path(payload[key])
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Invalid {key}: {path}")
    if not CHECKPOINT_DIR.is_dir() or not DIT_CHECKPOINT.is_file():
        raise FileNotFoundError("Model checkpoint installation is incomplete")
    if int(payload["seed"]) < 0 or Path(payload["output"]).suffix.lower() != ".mp4":
        raise ValueError("Seed must be nonnegative and output must be MP4")


def launch_worker(args: argparse.Namespace, payload: dict[str, object]) -> None:
    Path(payload["output"]).parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(payload["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(PYTHON_BIN), "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    env = {
        "CUDA_VISIBLE_DEVICES": str(payload["physical_gpu"]),
        "HOME": "/home/panminghao",
        "LANG": "C.UTF-8",
        "PATH": f"{CONDA_ENV / 'bin'}:/usr/bin:/bin",
        "TMPDIR": "/tmp",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    header = f"Log file: {log_path}\nSingle-GPU production pipeline on physical GPU {payload['physical_gpu']}\n{' '.join(command)}\n"
    sys.stdout.write(header)
    with log_path.open("wb") as log_file:
        writer = TimestampedLogWriter(log_file)
        writer.write(header)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        assert process.stdout is not None
        for chunk in iter(lambda: process.stdout.read(65536), b""):
            text = decoder.decode(chunk)
            writer.write(text)
            sys.stdout.write(text)
            sys.stdout.flush()
        tail = decoder.decode(b"", final=True)
        if tail:
            writer.write(tail)
        writer.close()
        return_code = process.wait()
    if return_code:
        raise SystemExit(return_code)


def run_worker(args: argparse.Namespace, payload: dict[str, object]) -> None:
    from scail2_inference import EngineConfig, InferenceJob, ProductionProfile, Scail2InferenceEngine

    profile = ProductionProfile.from_name(PROFILE_NAME)
    engine = Scail2InferenceEngine(EngineConfig(
        checkpoint_dir=CHECKPOINT_DIR,
        scail_checkpoint=DIT_CHECKPOINT,
        profile=profile,
        output_audio_mode=str(payload["output_audio"]),
    ))
    try:
        engine.load()
        engine.warmup()
        if args.init_only:
            print(json.dumps({"status": "initialized", "physical_gpu": int(payload["physical_gpu"])}, indent=2))
            return
        results = []
        base_output = Path(payload["output"])
        for iteration in range(1, args.repeat_count + 1):
            output = base_output if iteration == 1 else base_output.with_name(f"{base_output.stem}.repeat-{iteration}.mp4")
            job = InferenceJob(
                job_id=str(payload["job_id"]) if iteration == 1 else f"{payload['job_id']}-repeat-{iteration}",
                reference_image=Path(payload["reference_image"]),
                reference_mask=Path(payload["reference_mask"]),
                driving_video=Path(payload["driving_video"]),
                driving_mask=Path(payload["driving_mask"]),
                t5_cache_path=Path(payload["t5_cache"]),
                output_path=output,
                seed=int(payload["seed"]),
                metadata={"test_case": TEST_CASE, "iteration": iteration},
            )
            started = time.monotonic()
            result = engine.infer(job)
            results.append({**result.to_dict(), "elapsed_seconds": round(time.monotonic() - started, 3)})
        print(json.dumps(results, ensure_ascii=False, indent=2))
    finally:
        engine.close()


def main() -> None:
    args = parse_args()
    payload = resolve_payload(args)
    validate_paths(payload)
    if args.full_inference or args.dry_run:
        validate_media_contract(payload)
    if args.dry_run:
        print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}, ensure_ascii=False, indent=2))
    elif args.worker:
        run_worker(args, payload)
    else:
        launch_worker(args, payload)


if __name__ == "__main__":
    main()
