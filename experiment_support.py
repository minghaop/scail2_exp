"""Shared paths and logging helpers for local single-GPU experiments."""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONDA_ENV = Path("/home/panminghao/miniconda3/envs/scail2-single-gpu")
PYTHON_BIN = CONDA_ENV / "bin/python"
CHECKPOINT_DIR = Path("/raid/scail-2-20260819")
DIT_CHECKPOINT = CHECKPOINT_DIR / "derived/SCAIL-2-lightx2v-r128-dpo-alpha1-full-bf16.safetensors"
PROFILE_NAME = "scail2-512p-bf16-v1"
TEST_CASE = "101"
TEST_CASE_DIR = ROOT / "testdata" / TEST_CASE
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


def resolve_job_paths() -> dict[str, Path]:
    return {
        "reference_image": (TEST_CASE_DIR / "reference_image.png").resolve(),
        "reference_mask": (TEST_CASE_DIR / "reference_mask.png").resolve(),
        "driving_video": (TEST_CASE_DIR / "driving_video.mp4").resolve(),
        "driving_mask": (TEST_CASE_DIR / "driving_mask.mp4").resolve(),
    }


def validate_media_contract(payload: dict[str, object]) -> None:
    os.environ["PATH"] = f"{CONDA_ENV / 'bin'}:/usr/bin:/bin"
    from scail2_single_gpu_runtime.media import probe_audio, probe_video

    video_info = probe_video(Path(payload["driving_video"]))
    mask_info = probe_video(Path(payload["driving_mask"]))
    for field in ("width", "height", "frames", "fps_fraction"):
        if video_info[field] != mask_info[field]:
            raise ValueError(
                f"Driving video/mask {field} mismatch: "
                f"{video_info[field]} vs {mask_info[field]}"
            )
    if payload.get("output_audio") == "driving":
        probe_audio(Path(payload["driving_video"]))


class TimestampedLogWriter:
    def __init__(self, output_file: object) -> None:
        self.output_file = output_file
        self.pending = ""
        self.lock = threading.Lock()

    def write(self, text: str) -> None:
        with self.lock:
            records = (self.pending + text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
            self.pending = records.pop()
            for record in records:
                self._write_record(record)

    def close(self) -> None:
        with self.lock:
            if self.pending:
                self._write_record(self.pending)
                self.pending = ""
            self.output_file.flush()

    def _write_record(self, record: str) -> None:
        if record:
            timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
            self.output_file.write(f"{timestamp} {record}\n".encode("utf-8"))
            self.output_file.flush()
