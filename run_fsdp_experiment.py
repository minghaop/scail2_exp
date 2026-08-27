#!/home/panminghao/miniconda3/envs/scail2-single-gpu/bin/python
"""Run one service-free SCAIL-2 inference job with two-GPU FSDP."""

from __future__ import annotations

import argparse
import codecs
import json
import os
import subprocess
import sys
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
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output MP4 path; defaults to a timestamped experiment output.",
    )
    parser.add_argument("--job-id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
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

    def write(self, text: str) -> None:
        normalized = (self.pending + text).replace("\r\n", "\n").replace(
            "\r", "\n"
        )
        records = normalized.split("\n")
        self.pending = records.pop()
        for record in records:
            self._write_record(record)

    def close(self) -> None:
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
    return {
        "case": TEST_CASE,
        "job_id": job_id,
        "prompt": args.prompt,
        "seed": args.seed,
        "overwrite": args.overwrite,
        "init_only": args.init_only,
        "diagnose_fsdp": args.diagnose_fsdp,
        "expandable_segments": args.expandable_segments,
        "profile": PROFILE_NAME,
        "checkpoint_dir": CHECKPOINT_DIR,
        "dit_checkpoint": DIT_CHECKPOINT,
        "dit_meta_load": True,
        "t5_meta_load": True,
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
    if not str(payload["prompt"]).strip():
        raise ValueError("Prompt cannot be empty")
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
        "Load modes: T5=meta-assign, DiT=meta-assign\n"
        f"FSDP diagnostics: {'enabled' if payload['diagnose_fsdp'] else 'disabled'}\n"
        f"Expandable segments: {'enabled' if payload['expandable_segments'] else 'disabled'}\n"
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
        t5_fsdp=True,
        t5_meta_load=bool(payload["t5_meta_load"]),
        dit_fsdp=True,
        dit_meta_load=bool(payload["dit_meta_load"]),
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
        seed=int(payload["seed"]),
        overwrite=bool(payload["overwrite"]),
        metadata={"test_case": str(payload["case"]), "launcher": "cli-fsdp"},
    )

    engine = Scail2InferenceEngine(config)
    try:
        engine.load()
        engine.warmup()
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
        result = engine.infer(job)
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
        printable["t5_fsdp"] = True
        printable["dit_fsdp"] = True
        printable["t5_meta_load"] = True
        printable["dit_meta_load"] = True
        print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.worker:
        run_worker(args, payload)
        return
    launch_workers(payload)


if __name__ == "__main__":
    main()
