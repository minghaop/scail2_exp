# SCAIL-2 单卡推理 SDK 使用说明

当前 SDK 只提供单进程、单 GPU、串行调用的生产路径。调用方保证同一 worker
不并发提交推理任务。

## 接口

推理请求通过文件传递以下五项输入：

- reference image
- reference mask
- driving video
- driving mask
- 预先生成并校验过的 T5 cache

prompt 不再进入在线推理接口。使用 `prepare_conditioning.py` 在推理前生成 T5
cache；在线推理仍会用 reference image 计算 CLIP context。

## 生命周期

```python
from pathlib import Path

from scail2_inference import (
    EngineConfig,
    InferenceJob,
    ProductionProfile,
    Scail2InferenceEngine,
)

profile = ProductionProfile.from_name("scail2-512p-bf16-v1")
engine = Scail2InferenceEngine(
    EngineConfig(
        checkpoint_dir=Path("/models"),
        scail_checkpoint=Path(
            "/models/derived/"
            "SCAIL-2-lightx2v-r128-dpo-alpha1-full-bf16.safetensors"
        ),
        profile=profile,
        output_audio_mode="driving",
    )
)

engine.load()
engine.warmup()
try:
    result = engine.infer(
        InferenceJob(
            job_id="example-001",
            reference_image=Path("reference.png"),
            reference_mask=Path("reference_mask.png"),
            driving_video=Path("driving.mp4"),
            driving_mask=Path("driving_mask.mp4"),
            t5_cache_path=Path("conditioning.safetensors"),
            output_path=Path("output.mp4"),
        )
    )
finally:
    engine.close()
```

服务端队列可实现 `JobBackend`，并交给 `Scail2Runtime(engine).run(backend)`。
运行时会常驻模型并串行执行任务，直到 `backend.acquire()` 返回 `None`。

## 显存驻留顺序

READY 状态下 CLIP、DiT 和 VAE 位于 GPU，DiT 同时保留完整 CPU master：

1. 请求开始时计算 CLIP context，然后把 CLIP 移到 CPU。
2. 每个 segment 的 DiT 阶段把 VAE 移到 CPU，DiT 完整驻留 GPU。
3. VAE 阶段把 DiT 的最后 7 个 block 切换到 CPU master，并把 VAE 移回 GPU。
4. VAE 完成后恢复 7 个 DiT block，继续下一个 segment。
5. 请求结束后恢复 CLIP、DiT、VAE 的 READY 驻留状态。

## 容器约束

容器必须只看到一张满足要求的 GPU。启动前可运行：

```bash
scail2-runtime-info --expected-gpu-count 1
```

worker 直接用普通 Python 进程启动，不需要额外的进程组或多进程 launcher。
必要的启动、模型切换、任务结果和错误日志会保留；实验用显存 profile、trace 和
debug 开关不属于生产接口。
