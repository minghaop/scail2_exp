"""Runtime diagnostics used by Podman health and installation checks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from typing import Any


VALIDATED_VERSIONS = {
    "python": "3.10",
    "torch": "2.7.0+cu126",
    "torchvision": "0.22.0+cu126",
    "flash-attn": "2.8.3.post1",
    "diffusers": "0.39.0",
    "transformers": "5.13.0",
    "safetensors": "0.8.0",
    "decord": "0.6.0",
    "imageio-ffmpeg": "0.6.0",
    "fastapi": "0.139.0",
    "uvicorn": "0.51.0",
}

VALIDATED_CUDNN_VERSION = 90501
MINIMUM_GPU_MEMORY_BYTES = 40_000_000_000


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_version(command: str) -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "-version"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.splitlines()[0] if result.stdout else executable


def collect_environment() -> dict[str, Any]:
    packages = {
        name: _distribution_version(name)
        for name in (
            "torch",
            "torchvision",
            "flash-attn",
            "diffusers",
            "transformers",
            "safetensors",
            "decord",
            "imageio-ffmpeg",
            "fastapi",
            "uvicorn",
        )
    }
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "commands": {
            "ffmpeg": _command_version("ffmpeg"),
            "ffprobe": _command_version("ffprobe"),
        },
        "validated_versions": VALIDATED_VERSIONS,
    }
    try:
        import imageio_ffmpeg

        report["bundled_ffmpeg"] = {
            "path": imageio_ffmpeg.get_ffmpeg_exe(),
            "version": imageio_ffmpeg.get_ffmpeg_version(),
        }
    except Exception as error:
        report["bundled_ffmpeg"] = {"error": str(error)}
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        report["cuda"] = {
            "available": cuda_available,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if cuda_available else None,
            "visible_devices": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ]
            if cuda_available
            else [],
        }
    except Exception as error:
        report["cuda"] = {"available": False, "error": str(error)}
    return report


def validate_environment(
    report: dict[str, Any], *, expected_gpu_count: int = 2
) -> list[str]:
    errors: list[str] = []
    if sys.version_info[:2] != (3, 10):
        errors.append(f"Python must be 3.10, got {report['python']}")
    for package, expected in VALIDATED_VERSIONS.items():
        if package == "python":
            continue
        actual = report["packages"].get(package)
        if actual is None:
            errors.append(f"Required package is missing: {package}")
        elif actual != expected:
            errors.append(
                f"{package} version is {actual}, expected validated {expected}"
            )
    if report["commands"].get("ffprobe") is None:
        errors.append("ffprobe is unavailable")
    bundled = report.get("bundled_ffmpeg", {})
    if bundled.get("version") != "7.0.2-static":
        errors.append(
            "imageio-ffmpeg must provide the validated 7.0.2-static encoder"
        )
    cuda = report.get("cuda", {})
    if not cuda.get("available"):
        errors.append("CUDA is unavailable")
    elif cuda.get("visible_devices") != expected_gpu_count:
        errors.append(
            f"Expected exactly {expected_gpu_count} visible GPUs, got "
            f"{cuda.get('visible_devices')}"
        )
    else:
        if cuda.get("torch_cuda") != "12.6":
            errors.append(
                f"PyTorch CUDA runtime is {cuda.get('torch_cuda')}, expected 12.6"
            )
        if cuda.get("cudnn") != VALIDATED_CUDNN_VERSION:
            errors.append(
                f"cuDNN runtime is {cuda.get('cudnn')}, expected "
                f"{VALIDATED_CUDNN_VERSION}"
            )
        for device in cuda.get("devices", []):
            capability = tuple(device.get("capability", (0, 0)))
            if capability < (8, 0):
                errors.append(
                    f"GPU {device.get('index')} capability is {capability}, expected >= (8, 0)"
                )
            if int(device.get("memory_bytes", 0)) < MINIMUM_GPU_MEMORY_BYTES:
                errors.append(
                    f"GPU {device.get('index')} has less than 40 GB memory"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the SCAIL-2 runtime")
    parser.add_argument("--expected-gpu-count", type=int, default=1)
    parser.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="Report the environment without failing when CUDA is unavailable.",
    )
    args = parser.parse_args()
    report = collect_environment()
    errors = validate_environment(
        report, expected_gpu_count=args.expected_gpu_count
    )
    if args.allow_no_cuda:
        errors = [error for error in errors if error != "CUDA is unavailable"]
    report["validation_errors"] = errors
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
