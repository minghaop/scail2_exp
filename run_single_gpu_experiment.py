#!/home/panminghao/miniconda3/envs/scail2-single-gpu/bin/python
"""Run isolated single-GPU SCAIL-2 residency experiments."""

from __future__ import annotations

import argparse
import codecs
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from run_fsdp_experiment import (
    ALLOWED_PHYSICAL_GPUS,
    CHECKPOINT_DIR,
    CONDA_ENV,
    DEFAULT_PROMPT,
    DIT_CHECKPOINT,
    NvmlMemorySampler,
    PROFILE_NAME,
    PYTHON_BIN,
    ROOT,
    TEST_CASE,
    TimestampedLogWriter,
    resolve_job_paths,
    validate_media_contract,
    validate_static_paths,
)


sys.dont_write_bytecode = True

DEFAULT_PHYSICAL_GPU = "2"
DEFAULT_CONDITIONING_CACHE = (
    ROOT / "experiment_cache/conditioning" / f"{TEST_CASE}.safetensors"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physical-gpu", default=DEFAULT_PHYSICAL_GPU)
    parser.add_argument(
        "--conditioning-cache",
        type=Path,
        default=DEFAULT_CONDITIONING_CACHE,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--init-only",
        action="store_true",
        help="Load a full BF16 DiT on CUDA, retain its CPU master, then exit.",
    )
    mode.add_argument(
        "--memory-probe",
        action="store_true",
        help="Run segment 1 / diffusion step 1 without VAE decode or output.",
    )
    mode.add_argument(
        "--dit-segment-probe",
        action="store_true",
        help="Run all 6 DiT steps for segment 1 without VAE decode or output.",
    )
    mode.add_argument(
        "--vae-offload-probe",
        action="store_true",
        help="Run one full segment with 7 DiT blocks offloaded during VAE.",
    )
    mode.add_argument(
        "--full-inference",
        action="store_true",
        help="Run and validate all segments with 7-block VAE-phase offload.",
    )
    parser.add_argument("--ffn-chunk-size", type=int, default=8192)
    parser.add_argument("--rope-chunk-size", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ROOT / "experiment_outputs/single_gpu" / f"{TEST_CASE}-{timestamp}.mp4"


def resolve_payload(args: argparse.Namespace) -> dict[str, object]:
    physical_gpu = args.physical_gpu.strip()
    if physical_gpu not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError(
            "Single-GPU experiments may use only physical GPUs 0,1,2,3,6,7; "
            f"got {physical_gpu!r}"
        )
    if args.ffn_chunk_size < 0 or args.rope_chunk_size < 0:
        raise ValueError("Chunk sizes must be nonnegative")
    output = (args.output or default_output()).resolve()
    job_id = args.job_id or f"single-{TEST_CASE}-{output.stem}"
    return {
        "case": TEST_CASE,
        "job_id": job_id,
        "prompt": args.prompt,
        "seed": args.seed,
        "overwrite": False,
        "init_only": args.init_only,
        "memory_probe": args.memory_probe,
        "dit_segment_probe": args.dit_segment_probe,
        "vae_offload_probe": args.vae_offload_probe,
        "full_inference": args.full_inference,
        "ffn_chunk_size": args.ffn_chunk_size,
        "rope_chunk_size": args.rope_chunk_size,
        "profile": PROFILE_NAME,
        "checkpoint_dir": CHECKPOINT_DIR,
        "dit_checkpoint": DIT_CHECKPOINT,
        "conditioning_cache": args.conditioning_cache.resolve(),
        "physical_gpu": physical_gpu,
        "physical_gpus": (physical_gpu,),
        "output_audio": "driving" if args.full_inference else "none",
        "output": output,
        "log": ROOT / "experiment_logs/single_gpu" / f"{output.stem}.log",
        **resolve_job_paths(),
    }


def launch_worker(payload: dict[str, object]) -> None:
    physical_gpu = str(payload["physical_gpu"])
    Path(payload["output"]).parent.mkdir(parents=True, exist_ok=True)
    env = {
        "CUDA_VISIBLE_DEVICES": physical_gpu,
        "HOME": "/home/panminghao",
        "LANG": "C.UTF-8",
        "PATH": f"{CONDA_ENV / 'bin'}:/usr/bin:/bin",
        "TMPDIR": "/tmp",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    if payload["memory_probe"]:
        env["SCAIL2_DIT_MEMORY_DIAGNOSTICS"] = "1"
    if payload["vae_offload_probe"]:
        env["SCAIL2_FULL_MEMORY_PROFILE"] = "1"
    if payload["ffn_chunk_size"]:
        env["SCAIL2_FFN_CHUNK_SIZE"] = str(payload["ffn_chunk_size"])
    if payload["rope_chunk_size"]:
        env["SCAIL2_ROPE_CHUNK_SIZE"] = str(payload["rope_chunk_size"])

    child_args = list(sys.argv[1:])
    child_args.extend(
        [
            "--worker",
            "--output",
            str(payload["output"]),
            "--job-id",
            str(payload["job_id"]),
        ]
    )
    command = [str(PYTHON_BIN), "-u", str(Path(__file__).resolve()), *child_args]
    log_path = Path(payload["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"Log file: {log_path}\n"
        f"Launching single-GPU experiment on physical GPU {physical_gpu}\n"
        "World size: 1; process group: disabled; FSDP: disabled\n"
        "T5: precomputed; CLIP: online CUDA encode then CPU offload; "
        "DiT: full BF16 CUDA resident + CPU master\n"
        "Mode: "
        + (
            "init-only"
            if payload["init_only"]
            else "memory-probe"
            if payload["memory_probe"]
            else "dit-segment-probe"
            if payload["dit_segment_probe"]
            else "vae-offload-probe"
            if payload["vae_offload_probe"]
            else "full-inference"
        )
        + "\n"
        f"FFN chunk size: {payload['ffn_chunk_size']}\n"
        f"RoPE chunk size: {payload['rope_chunk_size']}\n"
        "Expandable segments: enabled\n"
        + " ".join(command)
        + "\n"
    )
    sys.stdout.write(header)
    sys.stdout.flush()

    with log_path.open("wb") as log_file:
        writer = TimestampedLogWriter(log_file)
        writer.write(header)
        sampler = NvmlMemorySampler((physical_gpu,), writer)
        sampler.start()
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
            raise RuntimeError("Failed to capture single-GPU worker output")
        for chunk in iter(lambda: process.stdout.read(65536), b""):
            text = decoder.decode(chunk)
            writer.write(text)
            sys.stdout.write(text)
            sys.stdout.flush()
        trailing = decoder.decode(b"", final=True)
        if trailing:
            writer.write(trailing)
            sys.stdout.write(trailing)
            sys.stdout.flush()
        for record in sampler.stop():
            writer.write_record(record)
            sys.stdout.write(record + "\n")
        writer.close()
        return_code = process.wait()
    if return_code != 0:
        raise SystemExit(return_code)


def current_rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("VmRSS is unavailable")


def log_residency(engine: object, event: str) -> None:
    import torch

    pipeline = engine.pipeline
    if pipeline is None:
        raise RuntimeError("Pipeline is unavailable")
    cpu_master = pipeline.dit_cpu_state_dict
    if cpu_master is None:
        raise RuntimeError("DiT CPU master was not retained")
    cpu_master_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in cpu_master.values()
        if isinstance(tensor, torch.Tensor)
    )
    model_parameters = list(pipeline.model.parameters())
    non_cuda = [parameter for parameter in model_parameters if not parameter.is_cuda]
    if non_cuda:
        raise RuntimeError(f"DiT has {len(non_cuda)} non-CUDA parameters")
    model_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model_parameters
    )
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(
        " ".join(
            [
                "SCAIL2_SINGLE_RESIDENCY",
                f"event={event}",
                f"cpu_master_tensors={len(cpu_master)}",
                f"cpu_master_mib={cpu_master_bytes / 2**20:.1f}",
                f"cuda_model_mib={model_bytes / 2**20:.1f}",
                f"process_rss_mib={current_rss_mib():.1f}",
                f"allocated_mib={torch.cuda.memory_allocated() / 2**20:.1f}",
                f"reserved_mib={torch.cuda.memory_reserved() / 2**20:.1f}",
                f"device_used_mib={(total - free) / 2**20:.1f}",
            ]
        ),
        file=sys.stderr,
        flush=True,
    )


def run_worker(args: argparse.Namespace, payload: dict[str, object]) -> None:
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if visible != str(payload["physical_gpu"]):
        raise RuntimeError(f"Unexpected CUDA_VISIBLE_DEVICES={visible!r}")
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Single-GPU worker requires WORLD_SIZE=1")

    from scail2_inference import (
        EngineConfig,
        InferenceJob,
        ProductionProfile,
        Scail2InferenceEngine,
    )

    profile = ProductionProfile.from_name(str(payload["profile"]))
    config = EngineConfig(
        checkpoint_dir=Path(payload["checkpoint_dir"]),
        scail_checkpoint=Path(payload["dit_checkpoint"]),
        profile=profile,
        expected_world_size=1,
        initialize_process_group=False,
        t5_fsdp=False,
        dit_fsdp=False,
        dit_meta_load=True,
        dit_init_on_cpu=False,
        keep_dit_cpu_state_dict=True,
        vae_dit_offload_blocks=(
            7 if args.vae_offload_probe or args.full_inference else 0
        ),
        precomputed_conditioning=True,
        online_clip_conditioning=True,
        offload_model=False,
        output_audio_mode=str(payload["output_audio"]),
    )
    job = InferenceJob(
        job_id=str(payload["job_id"]),
        reference_image=Path(payload["reference_image"]),
        reference_mask=Path(payload["reference_mask"]),
        driving_video=Path(payload["driving_video"]),
        driving_mask=Path(payload["driving_mask"]),
        prompt=str(payload["prompt"]),
        output_path=Path(payload["output"]),
        conditioning_path=Path(payload["conditioning_cache"]),
        seed=int(payload["seed"]),
        metadata={"test_case": str(payload["case"]), "launcher": "single-gpu"},
    )

    engine = Scail2InferenceEngine(config)
    try:
        engine.load()
        engine.warmup()
        log_residency(engine, "ready")
        if args.init_only:
            print(
                json.dumps(
                    {
                        "job_id": payload["job_id"],
                        "status": "single_gpu_initialized",
                        "world_size": 1,
                        "physical_gpu": int(str(payload["physical_gpu"])),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
            return
        if args.full_inference:
            result = engine.infer(job)
            log_residency(engine, "full_inference_complete")
            print(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                flush=True,
            )
            return
        normalized_job, _ = engine._normalize_job(job)
        engine._run_generation(
            normalized_job,
            temp_output=Path(payload["output"]),
            seed=int(payload["seed"]),
            diagnostic_memory_probe=not args.vae_offload_probe,
            diagnostic_memory_probe_steps=(
                1 if args.memory_probe else profile.sample_steps
            ),
            diagnostic_segment_limit=1 if args.vae_offload_probe else None,
        )
        completion_event = (
            "vae_offload_probe_complete"
            if args.vae_offload_probe
            else "memory_probe_complete"
            if args.memory_probe
            else "dit_segment_probe_complete"
        )
        log_residency(engine, completion_event)
        print(
            json.dumps(
                {
                    "job_id": payload["job_id"],
                    "status": f"single_gpu_{completion_event}",
                    "world_size": 1,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    finally:
        engine.close()


def main() -> None:
    args = parse_args()
    payload = resolve_payload(args)
    validate_static_paths(payload)
    if args.dry_run or (
        not args.worker
        and (
            args.memory_probe
            or args.dit_segment_probe
            or args.vae_offload_probe
            or args.full_inference
        )
    ):
        validate_media_contract(payload)
    if args.dry_run:
        printable = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in payload.items()
        }
        printable["physical_gpus"] = [int(str(payload["physical_gpu"]))]
        printable["world_size"] = 1
        printable["dit_fsdp"] = False
        printable["keep_dit_cpu_state_dict"] = True
        printable["online_clip_conditioning"] = True
        print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.worker:
        run_worker(args, payload)
        return
    launch_worker(payload)


if __name__ == "__main__":
    main()
