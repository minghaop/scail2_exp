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

## Source overlay worker

`Containerfile.worker` layers the Dispatcher WebSocket worker on an existing
validated runtime image. During the build it clones the latest `main` branch
and runs the renamed `scail2_single_gpu_runtime` source package, so it does not
import the `scail2_inference` package installed in the base image.

Build from this directory's repository root. Use `--no-cache` when the result
must include the latest remote `main`, because an image builder may otherwise
reuse the cached clone layer:

```bash
podman build --no-cache \
  --file deployment/podman/Containerfile.worker \
  --build-arg BASE_IMAGE=localhost/scail2-inference:0.1.3 \
  --tag localhost/scail2-worker:main \
  .
```

The shallow clone retains its `.git` directory, so the exact source revision is
available from `git -C /opt/scail2-src log -1`. The worker listens for
Dispatcher WebSocket connections on port 3000 and expects the full checkpoint
directory to be mounted read-only at `/models`.

The derived image replaces the base image's two-rank `torchrun` entrypoint with
one ordinary Python process. Start one container with exactly one visible GPU:

```bash
podman run --detach \
  --name scail2-worker \
  --restart unless-stopped \
  --ipc host \
  --publish 3000:3000 \
  --device nvidia.com/gpu=2 \
  --volume /raid/scail-2-20260819:/models:ro \
  localhost/scail2-worker:main
```

Readiness is reported by
`SCAIL2_DISPATCHER_WORKER_READY host=0.0.0.0 port=3000` after the single-GPU
engine has loaded successfully.

Dispatcher submissions contain `params.prompt` plus four downloaded media
inputs: reference image/mask and driving video/mask. The worker does not accept
a downloaded T5 cache. Before creating the SDK job it posts the prompt to the
fixed service URL `http://192.168.190.2:8001/v1/t5-cache` and stores the returned
artifact in its PID-specific `/dev/shm` work directory. The container network
must therefore be able to reach `192.168.190.2:8001`.
