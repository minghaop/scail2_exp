"""Strict media inspection and atomic publication helpers."""

from __future__ import annotations

import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def probe_video(path: Path) -> dict[str, int | float | str]:
    """Return exact timeline metadata for the first video stream."""
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found: {path}")
    stream = streams[0]
    frame_text = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frame_text in (None, "N/A"):
        raise ValueError(f"Could not determine frame count: {path}")
    rate = Fraction(stream.get("avg_frame_rate", "0/1"))
    if rate <= 0:
        raise ValueError(f"Could not determine a positive frame rate: {path}")
    frames = int(frame_text)
    duration_text = stream.get("duration")
    duration = (
        frames / float(rate)
        if duration_text in (None, "N/A")
        else float(duration_text)
    )
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(rate),
        "fps_fraction": f"{rate.numerator}/{rate.denominator}",
        "frames": frames,
        "duration": duration,
    }


def probe_audio(path: Path) -> dict[str, int | float | str]:
    """Return metadata for the first audio stream or raise when it is absent."""
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No audio stream found: {path}")
    stream = streams[0]
    return {
        "codec": str(stream.get("codec_name", "unknown")),
        "sample_rate": int(stream.get("sample_rate", 0)),
        "channels": int(stream.get("channels", 0)),
        "duration": 0.0
        if stream.get("duration") in (None, "N/A")
        else float(stream["duration"]),
    }


def mux_driving_audio(
    video_path: Path,
    audio_source: Path,
    output_path: Path,
    *,
    frames: int,
    fps_fraction: str,
    audio_bitrate: str = "192k",
) -> None:
    """Mux driving-video audio into a generated MP4 without re-encoding video."""
    duration = Fraction(frames, 1) / Fraction(fps_fraction)
    duration_text = f"{float(duration):.9f}".rstrip("0").rstrip(".")
    if output_path.exists():
        output_path.unlink()
    try:
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(audio_source),
                "-filter_complex",
                (
                    "[1:a:0]asetpts=PTS-STARTPTS,apad,"
                    f"atrim=duration={duration_text}[audio]"
                ),
                "-map",
                "0:v:0",
                "-map",
                "[audio]",
                "-map_metadata",
                "0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                audio_bitrate,
                "-t",
                duration_text,
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or str(error)
        raise ValueError(f"FFmpeg audio mux failed: {detail}") from error


def output_validation_error(
    path: Path,
    *,
    expected_width: int,
    expected_height: int,
    expected_fps_fraction: str,
    expected_frames: int,
    expected_duration: float,
    require_audio: bool = False,
) -> str | None:
    if not path.is_file() or path.stat().st_size == 0:
        return "file is missing or empty"
    try:
        metadata = probe_video(path)
    except Exception as error:
        return f"ffprobe failed: {error}"
    if metadata["width"] != expected_width or metadata["height"] != expected_height:
        return (
            f"resolution is {metadata['width']}x{metadata['height']}, expected "
            f"{expected_width}x{expected_height}"
        )
    if Fraction(str(metadata["fps_fraction"])) != Fraction(expected_fps_fraction):
        return (
            f"FPS is {metadata['fps_fraction']}, expected {expected_fps_fraction}"
        )
    if metadata["frames"] != expected_frames:
        return f"frame count is {metadata['frames']}, expected {expected_frames}"
    expected_rate = float(Fraction(expected_fps_fraction))
    if abs(float(metadata["duration"]) - expected_duration) > 0.5 / expected_rate:
        return (
            f"duration is {metadata['duration']:.6f}s, "
            f"expected {expected_duration:.6f}s"
        )
    if require_audio:
        try:
            audio = probe_audio(path)
        except Exception as error:
            return f"audio validation failed: {error}"
        if int(audio["sample_rate"]) <= 0 or int(audio["channels"]) <= 0:
            return "audio stream has invalid sample rate or channel count"
    return None


def atomic_publish_output(temp: Path, target: Path, *, overwrite: bool) -> None:
    """Publish a validated file without exposing partial output."""
    if overwrite:
        temp.replace(target)
        return
    try:
        os.link(temp, target)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite output created concurrently: {target}"
        ) from exc
    temp.unlink()


def checkpoint_provenance(path: Path) -> dict[str, Any]:
    """Read stable checkpoint identity fields without loading tensor payloads."""
    from safetensors import safe_open

    stat = path.stat()
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    fields = (
        "conversion_format",
        "source_sha256",
        "target_schema_sha256",
    )
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "header_metadata": {
            field: metadata[field] for field in fields if field in metadata
        },
    }
