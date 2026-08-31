# SCAIL-2 Podman handoff contract

One container owns one visible GPU, one persistent `Scail2InferenceEngine`, and
one active job at a time. Model files are mounted read-only and are not included
in the wheel or image.

The inference request supplies reference image/mask files, driving video/mask
files, and a validated T5 cache file. CLIP is evaluated online. The final MP4
can preserve the driving video's audio by copying the generated video stream
and encoding the audio as AAC.

Before building the image:

1. Build the `scail2_inference` wheel into `dist/`.
2. Stage the validated static FFmpeg and FFprobe artifacts required by the
   `Containerfile`.
3. Mirror and pin the CUDA/PyTorch base image and Python dependencies.
4. Prefer a prebuilt matching FlashAttention wheel.

At startup, `scail2-runtime-info --expected-gpu-count 1` verifies the visible
GPU and runtime dependencies before model loading. The service is ready only
after the engine emits `SCAIL2_WORKER_READY`.
