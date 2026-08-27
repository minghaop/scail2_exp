# SCAIL-2 Podman handoff contract

The tested Chinese step-by-step runbook is available at
[`MANUAL_DUAL_GPU_VALIDATION.zh-CN.md`](MANUAL_DUAL_GPU_VALIDATION.zh-CN.md).
Instructions for producing both the OCI image delivery and the SDK/build-kit
delivery are in [`DELIVERY_GUIDE.zh-CN.md`](DELIVERY_GUIDE.zh-CN.md).

One container is one persistent HTTP worker: two visible GPUs, two torchrun
ranks, one resident `Scail2InferenceEngine`, and one active job at a time.
Only rank 0 starts FastAPI on port 8000. Model files are not included in the
wheel or image and must be mounted read-only.

The 0.1.3 container requires an audio stream in `driving_video` and publishes a
final MP4 with that audio padded or trimmed to the generated timeline. FFmpeg
copies the generated video stream without re-encoding it and encodes the audio
as AAC. Set `SCAIL2_OUTPUT_AUDIO_MODE=none` only for an explicitly silent
derived deployment.

Uvicorn listens on `0.0.0.0` inside the container. Remote clients can connect
only when Podman also publishes the host port on `0.0.0.0`, for example
`--publish 0.0.0.0:8000:8000`; publishing `127.0.0.1:8000:8000` intentionally
restricts access to the deployment host.

The default `SCAIL2_WORKER_MODULE=scail2_fastapi_service` accepts the five input
fields documented in the Chinese runbook. A service-owned replacement module
implements `JobBackend` on rank 0 and passes it to
`Scail2DistributedRuntime`. Rank 1 must not start HTTP or connect to an external
queue or object store.

Before building the image:

1. Build `scail2_inference-0.1.3-py3-none-any.whl` into `dist/`.
2. Stage the validated FFmpeg 7.0.2 static `ffmpeg`, `ffprobe`, and GPLv3
   license in `dist/` with the filenames and SHA256 values required by the
   `Containerfile`.
3. Mirror and pin the CUDA/PyTorch base image by digest.
4. Redirect the PyTorch index and Python package index to approved internal
   mirrors. Prefer a prebuilt matching FlashAttention wheel over compiling it
   in every production build.

At container startup, `scail2-runtime-info` fails before model loading if CUDA,
the two-GPU device assignment, FlashAttention, ffprobe, or the bundled encoder
does not satisfy the validated contract. The service should advertise READY
only after `Scail2DistributedRuntime` finishes loading and synchronizing both
ranks and emits `SCAIL2_WORKER_READY`, both ranks initialize a separate CPU/Gloo
control channel, and rank 0 emits `SCAIL2_FASTAPI_READY`. The Gloo channel sends
idle heartbeats without occupying a GPU with an unmatched NCCL collective.
Before the first model load on a new host, run `/opt/scail2/nccl_smoke.py` with
two `torchrun` ranks as documented in the Chinese runbook.
