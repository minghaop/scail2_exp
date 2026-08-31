#!/home/panminghao/miniconda3/envs/scail2-single-gpu/bin/python
"""Run one service-free SCAIL-2 inference job with two-GPU FSDP."""

from __future__ import annotations

import argparse
import codecs
import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
CONDA_ENV = Path("/home/panminghao/miniconda3/envs/scail2-single-gpu")
PYTHON_BIN = CONDA_ENV / "bin/python"
CHECKPOINT_DIR = Path("/raid/scail-2-20260819")
DIT_CHECKPOINT = Path(
    "/raid/scail-2-20260819/derived/"
    "SCAIL-2-lightx2v-r128-dpo-alpha1-full-bf16.safetensors"
)
PROFILE_NAME = "scail2-512p-bf16-v1"
TEST_CASE = "101"
TEST_CASE_DIR = ROOT / "testdata" / TEST_CASE
OUTPUT_AUDIO_MODE = "driving"
DEFAULT_PHYSICAL_GPUS = ("2", "3")
ALLOWED_PHYSICAL_GPUS = frozenset(("0", "1", "2", "3", "6", "7"))
DEFAULT_PROMPT = (
    "A photorealistic video of the adult person shown in the reference image "
    "performing natural expressive body and hand movements. Preserve the exact "
    "facial identity, facial proportions, eyes, mouth, hairstyle, clothing, "
    "accessories, body proportions, lighting, background, and camera framing "
    "throughout. Stable detailed face, realistic skin, natural hands and limbs, "
    "coherent motion, consistent appearance, consistent lighting, steady camera, "
    "high detail."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output MP4 path; defaults to a timestamped experiment output.",
    )
    parser.add_argument("--job-id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--t5-cache",
        type=Path,
        default=ROOT / "experiment_cache/t5" / f"{TEST_CASE}.safetensors",
        help="Validated T5-only conditioning cache.",
    )
    parser.add_argument(
        "--physical-gpus",
        default=",".join(DEFAULT_PHYSICAL_GPUS),
        help="Two comma-separated physical GPU indices; GPUs 4 and 5 are forbidden.",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Load and warm up both FSDP ranks, then exit without inference.",
    )
    parser.add_argument(
        "--diagnose-fsdp",
        action="store_true",
        help="Enable block-level FSDP/NCCL diagnostics and a 120-second timeout.",
    )
    parser.add_argument(
        "--memory-probe",
        action="store_true",
        help=(
            "Run only segment 1 / diffusion step 1 without VAE decode or output, "
            "and log detailed block-0 CUDA memory peaks."
        ),
    )
    parser.add_argument(
        "--compare-trace",
        action="store_true",
        help="Hash first-step intermediate tensors for single/FSDP comparison.",
    )
    parser.add_argument(
        "--full-memory-profile",
        action="store_true",
        help=(
            "Run the complete inference while logging synchronized CUDA memory "
            "stages, per-block DiT peaks, and device-level NVML samples."
        ),
    )
    parser.add_argument(
        "--ffn-chunk-size",
        type=int,
        default=0,
        help="Experimentally process the DiT FFN in token chunks; 0 disables it.",
    )
    parser.add_argument(
        "--rope-chunk-size",
        type=int,
        default=0,
        help=(
            "Experimentally apply FP64/complex128 RoPE in token chunks; "
            "0 disables it."
        ),
    )
    parser.add_argument(
        "--bf16-residual",
        action="store_true",
        help="Experimentally retain inter-block DiT residuals in BF16.",
    )
    parser.add_argument(
        "--expandable-segments",
        action="store_true",
        help="Enable PyTorch CUDA allocator expandable segments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate paths without starting torchrun or touching a GPU.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_job_paths() -> dict[str, Path]:
    return {
        "reference_image": (TEST_CASE_DIR / "reference_image.png").resolve(),
        "reference_mask": (TEST_CASE_DIR / "reference_mask.png").resolve(),
        "driving_video": (TEST_CASE_DIR / "driving_video.mp4").resolve(),
        "driving_mask": (TEST_CASE_DIR / "driving_mask.mp4").resolve(),
    }


def default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        ROOT
        / "experiment_outputs/fsdp_baseline"
        / f"{TEST_CASE}-{timestamp}.mp4"
    )


def log_path_for_output(output: Path) -> Path:
    return ROOT / "experiment_logs/fsdp_baseline" / f"{output.stem}.log"


class TimestampedLogWriter:
    """Write newline/carriage-return delimited child output with timestamps."""

    def __init__(self, output_file: object) -> None:
        self.output_file = output_file
        self.pending = ""
        self.lock = threading.Lock()

    def write(self, text: str) -> None:
        with self.lock:
            normalized = (self.pending + text).replace("\r\n", "\n").replace(
                "\r", "\n"
            )
            records = normalized.split("\n")
            self.pending = records.pop()
            for record in records:
                self._write_record(record)

    def write_record(self, record: str) -> None:
        with self.lock:
            self._write_record(record)

    def close(self) -> None:
        with self.lock:
            if self.pending:
                self._write_record(self.pending)
                self.pending = ""
            self.output_file.flush()

    def _write_record(self, record: str) -> None:
        if not record:
            return
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self.output_file.write(f"{timestamp} {record}\n".encode("utf-8"))
        self.output_file.flush()


class NvmlMemorySampler:
    """Continuously sample physical device memory through nvidia-smi/NVML."""

    def __init__(
        self,
        physical_gpus: tuple[str, ...],
        log_writer: TimestampedLogWriter,
    ) -> None:
        self.physical_gpus = physical_gpus
        self.log_writer = log_writer
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.peaks: dict[str, tuple[int, int]] = {}

    def start(self) -> None:
        command = [
            "nvidia-smi",
            "-i",
            ",".join(self.physical_gpus),
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
            "--loop-ms=200",
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._read_samples, daemon=True)
        self.thread.start()

    def _read_samples(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for raw_line in self.process.stdout:
            line = raw_line.strip()
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4 or not fields[0].isdigit():
                self.log_writer.write_record(f"SCAIL2_NVML_MESSAGE {line}")
                continue
            physical_gpu, used, total, utilization = fields
            try:
                used_mib = int(used)
                total_mib = int(total)
            except ValueError:
                self.log_writer.write_record(f"SCAIL2_NVML_MESSAGE {line}")
                continue
            previous = self.peaks.get(physical_gpu)
            if previous is None or used_mib > previous[0]:
                self.peaks[physical_gpu] = (used_mib, total_mib)
            self.log_writer.write_record(
                " ".join(
                    [
                        "SCAIL2_NVML_SAMPLE",
                        f"physical_gpu={physical_gpu}",
                        f"used_mib={used_mib}",
                        f"total_mib={total_mib}",
                        f"utilization_percent={utilization}",
                    ]
                )
            )

    def stop(self) -> list[str]:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.thread is not None:
            self.thread.join(timeout=5)
        records = []
        for physical_gpu in self.physical_gpus:
            if physical_gpu not in self.peaks:
                continue
            used_mib, total_mib = self.peaks[physical_gpu]
            records.append(
                " ".join(
                    [
                        "SCAIL2_NVML_PEAK",
                        f"physical_gpu={physical_gpu}",
                        f"used_mib={used_mib}",
                        f"total_mib={total_mib}",
                    ]
                )
            )
        return records


def resolved_payload(args: argparse.Namespace) -> dict[str, object]:
    paths = resolve_job_paths()
    output = (args.output or default_output()).resolve()
    job_id = args.job_id or f"fsdp-{TEST_CASE}-{output.stem}"
    physical_gpus = tuple(
        item.strip() for item in args.physical_gpus.split(",") if item.strip()
    )
    if len(physical_gpus) != 2 or len(set(physical_gpus)) != 2:
        raise ValueError("--physical-gpus requires two distinct GPU indices")
    if not set(physical_gpus).issubset(ALLOWED_PHYSICAL_GPUS):
        raise ValueError(
            "Experiments may use only physical GPUs 0,1,2,3,6,7; "
            f"got {physical_gpus}"
        )
    if args.ffn_chunk_size < 0:
        raise ValueError("--ffn-chunk-size must be nonnegative")
    if args.rope_chunk_size < 0:
        raise ValueError("--rope-chunk-size must be nonnegative")
    if args.memory_probe and args.full_memory_profile:
        raise ValueError(
            "--memory-probe and --full-memory-profile are mutually exclusive"
        )
    if args.compare_trace and not args.memory_probe:
        raise ValueError("--compare-trace requires --memory-probe")
    return {
        "case": TEST_CASE,
        "job_id": job_id,
        "seed": args.seed,
        "overwrite": args.overwrite,
        "init_only": args.init_only,
        "diagnose_fsdp": args.diagnose_fsdp,
        "memory_probe": args.memory_probe,
        "compare_trace": args.compare_trace,
        "full_memory_profile": args.full_memory_profile,
        "ffn_chunk_size": args.ffn_chunk_size,
        "rope_chunk_size": args.rope_chunk_size,
        "bf16_residual": args.bf16_residual,
        "expandable_segments": args.expandable_segments,
        "profile": PROFILE_NAME,
        "checkpoint_dir": CHECKPOINT_DIR,
        "dit_checkpoint": DIT_CHECKPOINT,
        "dit_meta_load": True,
        "t5_cache": args.t5_cache.resolve(),
        "physical_gpus": physical_gpus,
        "output_audio": OUTPUT_AUDIO_MODE,
        "output": output,
        "log": log_path_for_output(output),
        **paths,
    }


def validate_static_paths(payload: dict[str, object]) -> None:
    directory = payload["checkpoint_dir"]
    if not isinstance(directory, Path) or not directory.is_dir():
        raise FileNotFoundError(f"Invalid checkpoint directory: {directory}")
    for key in (
        "dit_checkpoint",
        "reference_image",
        "reference_mask",
        "driving_video",
        "driving_mask",
    ):
        path = payload[key]
        if not isinstance(path, Path) or not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Invalid {key}: {path}")
    t5_cache = payload["t5_cache"]
    if (
        not isinstance(t5_cache, Path)
        or not t5_cache.is_file()
        or t5_cache.stat().st_size == 0
    ):
        raise FileNotFoundError(f"Invalid t5_cache: {t5_cache}")
    if int(payload["seed"]) < 0:
        raise ValueError("Seed must be nonnegative")
    output = payload["output"]
    if not isinstance(output, Path) or output.suffix.lower() != ".mp4":
        raise ValueError(f"Output must use the .mp4 suffix: {output}")


def validate_media_contract(payload: dict[str, object]) -> None:
    bin_dir = str(CONDA_ENV / "bin")
    os.environ["PATH"] = f"{bin_dir}:/usr/bin:/bin"

    from scail2_inference.media import probe_audio, probe_video

    video_info = probe_video(Path(payload["driving_video"]))
    mask_info = probe_video(Path(payload["driving_mask"]))
    for field in ("width", "height", "frames", "fps_fraction"):
        if video_info[field] != mask_info[field]:
            raise ValueError(
                f"Driving video/mask {field} mismatch: "
                f"{video_info[field]} vs {mask_info[field]}"
            )
    if payload["output_audio"] == "driving":
        probe_audio(Path(payload["driving_video"]))

    payload["input_video"] = video_info


def launch_workers(payload: dict[str, object]) -> None:
    if not PYTHON_BIN.is_file():
        raise FileNotFoundError(f"Experiment Python is unavailable: {PYTHON_BIN}")

    physical_gpus = tuple(str(item) for item in payload["physical_gpus"])
    env = {
        "CUDA_VISIBLE_DEVICES": ",".join(physical_gpus),
        "HOME": "/home/panminghao",
        "LANG": "C.UTF-8",
        "PATH": f"{CONDA_ENV / 'bin'}:/usr/bin:/bin",
        "TMPDIR": "/tmp",
    }
    if payload["diagnose_fsdp"]:
        env.update(
            {
                "SCAIL2_FSDP_DIAGNOSTICS": "1",
                "SCAIL2_PROCESS_GROUP_TIMEOUT_SECONDS": "120",
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "COLL",
                "TORCH_NCCL_TRACE_BUFFER_SIZE": "100000",
                "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
                "TORCH_NCCL_DESYNC_DEBUG": "1",
                "TORCH_NCCL_ENABLE_TIMING": "1",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            }
        )
    if payload["memory_probe"]:
        env["SCAIL2_DIT_MEMORY_DIAGNOSTICS"] = "1"
    if payload["compare_trace"]:
        env["SCAIL2_COMPARE_TRACE"] = "1"
    if payload["full_memory_profile"]:
        env["SCAIL2_FULL_MEMORY_PROFILE"] = "1"
    if payload["ffn_chunk_size"]:
        env["SCAIL2_FFN_CHUNK_SIZE"] = str(payload["ffn_chunk_size"])
    if payload["rope_chunk_size"]:
        env["SCAIL2_ROPE_CHUNK_SIZE"] = str(payload["rope_chunk_size"])
    if payload["bf16_residual"]:
        env["SCAIL2_BF16_RESIDUAL"] = "1"
    if payload["expandable_segments"]:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    command = [
        str(PYTHON_BIN),
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--max-restarts=0",
        str(Path(__file__).resolve()),
        *child_args,
    ]
    log_path = payload["log"]
    if not isinstance(log_path, Path):
        raise TypeError(f"Invalid log path: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"Log file: {log_path}\n"
        f"Launching two-GPU FSDP on physical GPUs {','.join(physical_gpus)}:\n"
        + "Load modes: T5=file cache, CLIP=online, DiT=meta-assign\n"
        + f"FSDP diagnostics: {'enabled' if payload['diagnose_fsdp'] else 'disabled'}\n"
        + f"DiT memory probe: {'enabled' if payload['memory_probe'] else 'disabled'}\n"
        + f"Comparison trace: {'enabled' if payload['compare_trace'] else 'disabled'}\n"
        + (
            "Full memory profile: enabled (CUDA stages + 200 ms NVML samples)\n"
            if payload["full_memory_profile"]
            else "Full memory profile: disabled\n"
        )
        + f"FFN chunk size: {payload['ffn_chunk_size']}\n"
        + f"RoPE chunk size: {payload['rope_chunk_size']}\n"
        + f"BF16 residual: {'enabled' if payload['bf16_residual'] else 'disabled'}\n"
        + f"Expandable segments: {'enabled' if payload['expandable_segments'] else 'disabled'}\n"
        + " ".join(command)
        + "\n"
    )
    sys.stdout.write(header)
    sys.stdout.flush()

    with log_path.open("wb") as log_file:
        timestamped_log = TimestampedLogWriter(log_file)
        timestamped_log.write(header)
        nvml_sampler = None
        if payload["full_memory_profile"]:
            nvml_sampler = NvmlMemorySampler(physical_gpus, timestamped_log)
            nvml_sampler.start()
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
            raise RuntimeError("Failed to capture torchrun output")
        for chunk in iter(lambda: process.stdout.read(65536), b""):
            text = decoder.decode(chunk)
            timestamped_log.write(text)
            sys.stdout.write(text)
            sys.stdout.flush()
        trailing_text = decoder.decode(b"", final=True)
        if trailing_text:
            timestamped_log.write(trailing_text)
            sys.stdout.write(trailing_text)
            sys.stdout.flush()
        if nvml_sampler is not None:
            for record in nvml_sampler.stop():
                timestamped_log.write_record(record)
                sys.stdout.write(record + "\n")
                sys.stdout.flush()
        timestamped_log.close()
        return_code = process.wait()

    if return_code != 0:
        raise SystemExit(return_code)


def assert_worker_environment(expected_physical_gpus: tuple[str, ...]) -> None:
    visible = [
        item.strip()
        for item in os.getenv("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    if visible != list(expected_physical_gpus):
        raise RuntimeError(
            "Worker CUDA binding does not match the requested physical GPUs; "
            f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')!r}"
        )
    if int(os.getenv("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("The FSDP experiment requires WORLD_SIZE=2")


def run_worker(args: argparse.Namespace, payload: dict[str, object]) -> None:
    expected_physical_gpus = tuple(
        str(item) for item in payload["physical_gpus"]
    )
    assert_worker_environment(expected_physical_gpus)

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
        expected_world_size=2,
        initialize_process_group=True,
        t5_fsdp=False,
        dit_fsdp=True,
        dit_meta_load=bool(payload["dit_meta_load"]),
        dit_init_on_cpu=True,
        keep_dit_cpu_state_dict=False,
        vae_dit_offload_blocks=0,
        offload_vae_during_dit=False,
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
        t5_cache_path=Path(payload["t5_cache"]),
        output_path=Path(payload["output"]),
        seed=int(payload["seed"]),
        overwrite=bool(payload["overwrite"]),
        metadata={"test_case": str(payload["case"]), "launcher": "cli-fsdp"},
    )

    engine = Scail2InferenceEngine(config)
    try:
        engine.load()
        engine.warmup()
        if payload["full_memory_profile"]:
            import torch

            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            print(
                " ".join(
                    [
                        "SCAIL2_MEMORY_STAGE",
                        f"rank={os.getenv('RANK', '0')}",
                        "stage=engine_ready",
                        "event=snapshot",
                        f"allocated_mib={torch.cuda.memory_allocated() / 2**20:.1f}",
                        f"reserved_mib={torch.cuda.memory_reserved() / 2**20:.1f}",
                        f"device_used_mib={(total - free) / 2**20:.1f}",
                    ]
                ),
                file=sys.stderr,
                flush=True,
            )
        if args.init_only:
            if engine.is_primary:
                print(
                    json.dumps(
                        {
                            "job_id": str(payload["job_id"]),
                            "status": "initialized",
                            "profile": str(payload["profile"]),
                            "world_size": 2,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    flush=True,
                )
            return
        if args.memory_probe:
            normalized_job, _ = engine._normalize_job(job)
            engine._run_generation(
                normalized_job,
                temp_output=Path(payload["output"]),
                seed=int(payload["seed"]),
                diagnostic_memory_probe=True,
            )
            if engine.is_primary:
                print(
                    json.dumps(
                        {
                            "job_id": str(payload["job_id"]),
                            "status": "memory_probe_complete",
                            "profile": str(payload["profile"]),
                            "world_size": 2,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    flush=True,
                )
            return
        result = engine.infer(job)
        if payload["full_memory_profile"]:
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            print(
                " ".join(
                    [
                        "SCAIL2_MEMORY_STAGE",
                        f"rank={os.getenv('RANK', '0')}",
                        "stage=inference_complete",
                        "event=snapshot",
                        f"allocated_mib={torch.cuda.memory_allocated() / 2**20:.1f}",
                        f"reserved_mib={torch.cuda.memory_reserved() / 2**20:.1f}",
                        f"device_used_mib={(total - free) / 2**20:.1f}",
                    ]
                ),
                file=sys.stderr,
                flush=True,
            )
        if engine.is_primary:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), flush=True)
    finally:
        engine.close()


def main() -> None:
    args = parse_args()
    payload = resolved_payload(args)
    validate_static_paths(payload)
    # The parent validates media before launching. Worker ranks skip this
    # duplicate scan because Scail2InferenceEngine performs the same strict
    # probe immediately before inference.
    if args.dry_run or (not args.worker and not args.init_only):
        validate_media_contract(payload)

    if args.dry_run:
        printable = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in payload.items()
        }
        printable["physical_gpus"] = [
            int(item) for item in payload["physical_gpus"]
        ]
        printable["world_size"] = 2
        printable["t5_fsdp"] = False
        printable["dit_fsdp"] = True
        printable["t5_meta_load"] = False
        printable["dit_meta_load"] = True
        printable["precomputed_conditioning"] = True
        printable["online_clip_conditioning"] = True
        print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.worker:
        run_worker(args, payload)
        return
    launch_workers(payload)


if __name__ == "__main__":
    main()
