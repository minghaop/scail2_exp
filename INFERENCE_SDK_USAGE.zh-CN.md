# SCAIL-2 推理 SDK（wheel）接入与使用手册

## 0. 文档用途

本文面向首次接收 SCAIL-2 推理交付、未参与 SDK 开发的同事。目标不是要求接入方
理解 `wan/` 或 FSDP 内部实现，而是说明：

1. 收到的 wheel 负责什么、不负责什么；
2. 如何在已经验证的环境中安装并确认 wheel；
3. 如何把本地输入组织成 `InferenceJob`；
4. 如何通过 `JobBackend` 接入现有调度系统；
5. 如何正确启动一个双卡、模型常驻显存的 worker；
6. 出现任务错误或 CUDA/NCCL 错误时应该如何处理。

当前交付版本为 `scail2-inference 0.1.3`。标准生产单元是：

```text
一个容器 + 两张 GPU + 两个 torchrun rank + 一个常驻模型 worker
```

最重要的结论：接入方通常不需要修改 `Scail2InferenceEngine`。如果只调整任务来源、
排队方式、状态存储或结果上传，只需要修改服务适配层，重点是实现 `JobBackend`。

## 1. 先判断需要哪种接入方式

本次交付支持三种使用方式。请先确认自己的工作属于哪一种：

| 目标 | 推荐做法 | 是否需要编写 SDK 接入代码 |
| --- | --- | --- |
| 直接部署现成 HTTP 服务 | 使用已经构建的镜像，按照 Podman 部署手册启动 | 不需要 |
| 在自己的机器重新构建镜像 | 使用 build kit 中的 `Containerfile`；它会自动安装 wheel | 通常不需要 |
| 接入公司内部队列、数据库、对象存储或 API | 保留 wheel，参考示例实现自己的 `JobBackend` 和服务入口 | 需要 |

如果只是启动现有 HTTP 服务，请直接阅读：

- [`MANUAL_DUAL_GPU_VALIDATION.zh-CN.md`](../deployment/podman/MANUAL_DUAL_GPU_VALIDATION.zh-CN.md)

如果要自行制作镜像或离线交付包，请阅读：

- [`DELIVERY_GUIDE.zh-CN.md`](../deployment/podman/DELIVERY_GUIDE.zh-CN.md)

本文余下内容主要服务于第三种情况：在不修改模型算法代码的前提下，将 wheel 接入
自己的工程化服务。

## 2. 收到的交付物及边界

### 2.1 wheel 包含什么

wheel 文件名：

```text
dist/scail2_inference-0.1.3-py3-none-any.whl
```

wheel 内包含：

- `scail2_inference` 公共 SDK；
- `wan` 模型推理实现；
- `generate.py` 和分段推理辅助代码；
- `scail2-512p-bf16-v1` 生产 profile；
- SCAIL-14B/1.3B 架构 JSON。

wheel 负责：模型加载、双 rank 协调、任务串行执行、模型常驻显存、输入输出合同
校验、可选音频封装以及最终 MP4 的原子发布。

### 2.2 wheel 不包含什么

wheel 不包含：

- 模型权重；
- CUDA、PyTorch、Torchvision 和 FlashAttention；
- 系统 FFmpeg/ffprobe；
- FastAPI/Uvicorn 服务入口；
- Podman GPU、端口和 volume 配置；
- 业务队列、数据库、对象存储及重试策略。

模型文件单独存放在共享目录：

```text
/mist/emochat/models/scail-2-20260819
```

容器启动时将该目录只读映射为 `/models`，不需要把约 44 GiB 模型复制进 wheel、
镜像或 build kit。

### 2.3 示例代码与 wheel 的关系

以下文件是随 build kit 交付的参考代码，不属于 wheel 本体：

| 文件 | 用途 | 接入方如何使用 |
| --- | --- | --- |
| [`scail2_fastapi_service.py`](../examples/scail2_fastapi_service.py) | 可运行的 HTTP 服务、内存队列和结果下载 | 可直接使用，或替换其中的 Backend |
| [`scail2_worker_service.py`](../examples/scail2_worker_service.py) | 最小 JSONL Backend | 用于理解最小接入边界 |
| [`scail2_sdk_debug.py`](../examples/scail2_sdk_debug.py) | 内置 014/025/089 的 SDK 调试入口 | 先 dry-run，再做双卡调试 |

参考 `Containerfile` 会把这些入口文件单独复制到镜像 `/opt/scail2`。因此，升级
wheel 不会自动替换业务服务代码；重新构建镜像时应同时确认 wheel 和服务入口版本。

### 2.4 可以修改与不应修改的内容

| 可以按业务需要修改 | 不应由部署接入方修改 |
| --- | --- |
| `JobBackend` 实现 | `wan/` 模型内部代码 |
| HTTP/消息队列/数据库适配 | wheel 内生产 profile 参数 |
| 输入文件下载和本地缓存 | FSDP 双 rank 调用顺序 |
| 任务状态、限流、重试和监控 | `Scail2InferenceEngine` 生命周期 |
| 结果上传和对外返回方式 | 输入输出媒体校验及原子发布逻辑 |

如确需改变分辨率、steps、shift、seed、dtype、FSDP 或模型加载方式，应由 SDK 维护方
发布新的 profile 或 wheel 版本，不要在部署服务中临时覆盖。

## 3. 安装并确认 wheel

如果使用交付的 Podman 镜像，wheel 已经安装完成，不需要再执行本节的
`pip install`。直接进入部署手册的容器启动步骤即可。

只有在自建镜像或自有 Python 环境中接入 SDK 时，才需要手工安装 wheel。生产环境
应严格复用 `deployment/podman/Containerfile` 和 `requirements-runtime.lock` 中
已经验证的版本，不建议在一套普通 Python 环境里临时补依赖。

### 3.1 校验交付文件

在 build kit 根目录执行：

```bash
sha256sum --check SHA256SUMS
```

校验通过后再安装。若 wheel 或 FFmpeg 校验失败，应停止部署并重新取得交付文件，
不要继续使用损坏或来源不明的文件。

### 3.2 安装 wheel

在已经准备好完整运行环境的 Python 3.10 中执行：

```bash
python3.10 -m pip install --no-deps \
  dist/scail2_inference-0.1.3-py3-none-any.whl

python3.10 -m pip show scail2-inference
```

这里使用 `--no-deps`，是因为 CUDA/PyTorch 以及其他 Python 依赖已经由镜像构建文件
按验证版本安装。不要让 pip 在生产环境中自行解析出另一套依赖版本。

### 3.3 确认版本和环境

确认实际导入的是本次交付版本：

```bash
python3.10 -c 'import scail2_inference as sdk; print(sdk.__version__)'
```

期望输出：

```text
0.1.3
```

再执行生产环境门禁：

```bash
scail2-runtime-info --expected-gpu-count 2
```

该命令返回码为 `0` 且 JSON 中 `validation_errors` 为空，才表示 Python、PyTorch、
CUDA、cuDNN、FlashAttention、FFmpeg 和两张可见 GPU 满足当前验证合同。

注意：

- 仅执行 `pip install` 并不能准备 CUDA/PyTorch/FlashAttention/FFmpeg；
- wheel 文件名中的 `py3-none-any` 不表示它能脱离 CUDA 环境运行；
- wheel 的 Python 要求是 `>=3.10,<3.11`，本交付固定使用 Python 3.10；
- `scail2-runtime-info` 必须看到恰好两张分配给当前 worker 的 GPU，而不是宿主机
  上的全部 GPU。

## 4. 公共 API 是什么

正确导入方式如下：

```python
from scail2_inference import (
    EngineConfig,
    InferenceJob,
    InferenceResult,
    JobBackend,
    ProductionProfile,
    Scail2DistributedRuntime,
    Scail2InferenceEngine,
    __version__,
)
```

这里是 `__version__`（前后各两个下划线），不是 `**version**`。

| 名称 | 用途 | 通常由谁创建/实现 |
| --- | --- | --- |
| `ProductionProfile` | 版本化的算法参数与输出效果合同 | SDK 从 wheel 内读取 |
| `EngineConfig` | 模型路径、双卡拓扑、FSDP、常驻和音频策略 | 服务入口创建 |
| `InferenceJob` | 一次已经落到本地文件系统的推理任务 | 服务适配层创建 |
| `InferenceResult` | 成功/跳过后的结构化结果元数据 | Engine 返回 |
| `JobBackend` | 队列和任务状态的接口协议 | 部署服务实现 |
| `Scail2InferenceEngine` | 模型加载、常驻、推理、校验和发布的核心 | 每个 rank 创建一个 |
| `Scail2DistributedRuntime` | rank 0/1 的任务广播和常驻任务循环 | 每个 rank 创建一个 |
| `__version__` | 当前 wheel 版本字符串 | SDK 提供 |

接入方实际编写代码时，通常按下面的顺序使用：

```text
1. ProductionProfile.from_name(...)       读取固定算法参数
2. EngineConfig(...)                      填模型路径和双卡配置
3. Scail2InferenceEngine(config)           创建 Engine（此时尚未加载模型）
4. Scail2DistributedRuntime(engine)        创建双 rank 任务循环
5. runtime.run(backend 或 None)            自动 load、warmup、推理和 close
```

`InferenceJob` 由 rank 0 的 Backend 提供；`InferenceResult` 由 SDK 产生，接入方一般
不需要手工构造。`JobBackend` 是接入方与 SDK 之间最主要的扩展接口。

不要从部署服务直接调用 `wan.scail.SCAIL2Pipeline`。这样会绕过 SDK 的双 rank
协调、生命周期和输入输出校验。

## 5. 双卡执行模型

下面的双 rank 规则是 SDK 运行合同，不是性能优化建议。生产配置必须由
`torchrun` 启动两个进程：

```text
                         ┌─ rank 0 / GPU 0
外部调度 → JobBackend ──┤  Engine.infer(job)
                         │        ⇅ NCCL/FSDP
                         └─ rank 1 / GPU 1
                            Engine.infer(job)
```

关键规则：

1. rank 0 和 rank 1 都要创建相同的 `ProductionProfile`、`EngineConfig`、Engine
   和 Runtime。
2. `JobBackend` 只允许存在于 rank 0；rank 1 必须调用 `runtime.run(None)`。
3. 两个 rank 必须按相同顺序参与每次推理，不能只在 rank 0 调 `engine.infer()`。
4. 一名 worker 串行执行任务。服务可以排队，但不能同时在同一 Engine 上执行两个任务。
5. `runtime.run()` 会自动执行 `load()`、`warmup()`、任务循环和 `close()`；使用
   Runtime 时不要提前手工重复调用这些方法。
6. 两张 GPU 共同执行同一个任务，不是 GPU 0 和 GPU 1 分别执行两个任务。
7. 模型只在 worker 启动时加载一次；任务完成后继续驻留显存，直到 worker 停止。

不能用下面的单进程方式启动生产 worker：

```bash
python3.10 your_service.py
```

应使用：

```bash
torchrun --standalone --nproc_per_node=2 -m your_service
```

## 6. 准备 profile 和 EngineConfig

推荐从 wheel 读取经过验证的 profile：

```python
from pathlib import Path

from scail2_inference import EngineConfig, ProductionProfile

profile = ProductionProfile.from_name("scail2-512p-bf16-v1")

config = EngineConfig(
    checkpoint_dir=Path("/models"),
    scail_checkpoint=Path(
        "/models/derived/"
        "SCAIL-2-lightx2v-r128-dpo-alpha1-full-bf16.safetensors"
    ),
    profile=profile,
    expected_world_size=2,
    t5_fsdp=True,
    dit_fsdp=True,
    offload_model=False,
    output_audio_mode="driving",
)
```

上面的 `/models` 是容器内路径。Podman 启动时应建立如下只读映射：

```text
宿主机 /mist/emochat/models/scail-2-20260819  ->  容器 /models
```

服务代码运行在容器内，因此 `EngineConfig` 应填写 `/models`，不要把宿主机
`/mist/...` 绝对路径写进容器内的 Python 代码。

常用字段含义：

- `checkpoint_dir`：公共模型目录，包含 UMT5、VAE、CLIP 等文件。
- `scail_checkpoint`：SCAIL-2 BF16 DiT checkpoint。
- `scail_config_path`：通常不填；SDK 会根据 profile 从 wheel 读取架构 JSON。
- `expected_world_size=2`：要求两个 torchrun rank。
- `t5_fsdp=True`、`dit_fsdp=True`：让两张 GPU 分担模型。
- `offload_model=False`：禁止任务后把模型卸载到 CPU，是常驻显存合同。
- `output_audio_mode="driving"`：把 `driving_video` 的音轨封装进结果；输入视频
  没有音轨时任务会失败。若需要无声结果，设置为 `"none"`。

不要由 HTTP 请求随意构造 `ProductionProfile`。当前
`scail2-512p-bf16-v1` 固定了 BF16、512×896、Euler、6 steps、shift 5、seed 42
和分段参数，这些字段属于算法效果和兼容性合同。

模型目录应满足 `deployment/podman/model-manifest.yaml`，主要结构为：

```text
/models/
├── derived/
│   └── SCAIL-2-lightx2v-r128-dpo-alpha1-full-bf16.safetensors
├── umt5-xxl/
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   ├── special_tokens_map.json
│   ├── spiece.model
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── Wan2.1_VAE.pth
└── models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth
```

## 7. 创建 InferenceJob

`InferenceJob` 表示“输入文件已经在本地可见”的任务。它不负责下载 URL 或对象存储：

在创建任务之前，接入服务应先完成：

1. 将对象存储或远程 URL 下载到 worker 可见的输入挂载目录；
2. 确认四个输入文件都已写完且非空，不能把仍在下载的临时文件提交给 SDK；
3. 为业务请求分配唯一 `job_id`；如需幂等，由业务服务维护请求键与 job_id 的关系；
4. 在受控输出目录中生成 `.mp4` 路径，不允许外部请求任意指定宿主机路径。

```python
from pathlib import Path

from scail2_inference import InferenceJob

job = InferenceJob(
    job_id="request-0001",
    reference_image=Path("/inputs/014/ref.png"),
    reference_mask=Path("/inputs/014/ref_mask.png"),
    driving_video=Path("/inputs/014/rendered_v2.mp4"),
    driving_mask=Path("/inputs/014/rendered_mask_v2.mp4"),
    prompt="A person moving naturally.",
    output_path=Path("/outputs/request-0001.mp4"),
    overwrite=False,
    metadata={"request_id": "request-0001"},
)

job.validate(check_paths=True)
```

必填字段：

- `job_id`：非空、在当前服务内唯一。
- `reference_image`：参考人物图。
- `reference_mask`：参考 mask，Engine 要求使用 `.png`。
- `driving_video`：驱动视频。
- `driving_mask`：与驱动视频严格对齐的 mask 视频。
- `prompt`：非空提示词。
- `output_path`：必须以 `.mp4` 结尾。

可选字段：

- `output_fps_fraction`：如 `"30/1"` 或 `"30000/1001"`；不填则从驱动视频读取。
- `expected_output_frames`：不填则使用驱动视频帧数。
- `expected_output_duration`：不填则使用驱动视频时长。
- `seed`：不填则使用 profile 的 seed。
- `overwrite`：默认 `False`，禁止覆盖已有结果。
- `metadata`：调用方附加信息，会原样进入 `InferenceResult`。

Engine 还会检查：四个输入文件非空；驱动视频和 mask 的宽、高、帧数和 FPS
完全一致；帧数、FPS、时长符合恒定帧率关系；输出媒体属性符合 profile。

如果 `EngineConfig.output_audio_mode="driving"`，`driving_video` 还必须包含音轨。
SDK 会把该音轨封装进最终视频；如果驱动视频没有音轨，本任务会以
`InputValidationError` 失败，不会静默生成无声结果。

`InferenceJob.validate()` 本身只检查本地文件合同，不会限制输入必须位于 `/inputs`。
如果任务来自外部请求，服务适配层必须像 FastAPI 示例一样消解 `..` 和软链接、
限制输入根目录，并由服务端生成受控的输出路径，不能直接信任客户端路径。

已有输出在 `overwrite=False` 时的行为：

- 已有 MP4 完全符合本任务合同：不重复计算，返回 `status="skipped"`。
- 已有 MP4 存在但不符合合同：抛出 `OutputValidationError`。
- `overwrite=True`：重新生成，并通过临时文件原子发布。

## 8. 实现 JobBackend

`JobBackend` 是一个 `Protocol`，不能直接作为生产队列实例使用。部署服务需要实现：

```python
from scail2_inference import InferenceJob, InferenceResult, JobBackend


class MyJobBackend(JobBackend):
    def acquire(self) -> InferenceJob | None:
        """可以阻塞等待；返回 None 表示要求 worker 正常退出。"""
        ...

    def mark_running(self, job: InferenceJob) -> None:
        ...

    def mark_success(
        self,
        job: InferenceJob,
        result: InferenceResult,
    ) -> None:
        ...

    def mark_failed(
        self,
        job: InferenceJob,
        error: BaseException,
        traceback_text: str,
    ) -> None:
        ...
```

Runtime 对一个任务的调用顺序固定为：

```text
backend.acquire()
  -> backend.mark_running(job)
  -> 两个 rank 同时 engine.infer(job)
  -> 成功：backend.mark_success(job, result)
     失败：backend.mark_failed(job, error, traceback)
  -> 再次 backend.acquire()
```

可以在这些方法里接 Redis、数据库、消息队列或公司内部调度系统。注意：

- Backend 只在 rank 0 创建和连接外部系统。
- `acquire()` 在没有任务时应阻塞等待，而不是立即返回 `None`。返回 `None` 的含义是
  “永久停止当前 worker”，Runtime 随后会执行 `engine.close()` 并释放显存中的模型。
- Runtime 会在独立线程调用阻塞的 `acquire()`，空闲期间仍会维持 rank 间控制心跳。
- Runtime 回调和 HTTP/API 线程可能并发访问任务记录，自定义 Backend 要自行加锁。
- 四个方法自身抛出的异常会被视为服务基础设施故障，并停止整个 worker。
- `InferenceResult` 是结果元数据，视频文件位于 `result.output_path`。
- 不要在 Backend 中调用 `engine.infer()`；Backend 只提供任务和记录状态，推理由
  `Scail2DistributedRuntime` 统一触发。

现成交付中的 `FastApiJobBackend` 已实现上述接口，并演示了有界队列、锁、任务状态
和优雅停机。若接入内部调度系统，建议复制其状态处理思路，替换任务来源与结果存储，
不要修改 Engine。

## 9. 推荐的服务入口代码

一个服务入口的核心结构如下。`MyJobBackend` 应替换为上一节的实际实现：

```python
import os
from pathlib import Path

from scail2_inference import (
    EngineConfig,
    ProductionProfile,
    Scail2DistributedRuntime,
    Scail2InferenceEngine,
)


def main() -> None:
    profile = ProductionProfile.from_name("scail2-512p-bf16-v1")
    config = EngineConfig(
        checkpoint_dir=Path(os.environ["SCAIL2_CHECKPOINT_DIR"]),
        scail_checkpoint=Path(os.environ["SCAIL2_DIT_CHECKPOINT"]),
        profile=profile,
        expected_world_size=2,
        t5_fsdp=True,
        dit_fsdp=True,
        offload_model=False,
        output_audio_mode="driving",
    )

    engine = Scail2InferenceEngine(config)
    runtime = Scail2DistributedRuntime(engine)

    # 外部队列只能由 rank 0 持有；rank 1 从 Runtime 接收广播任务。
    backend = MyJobBackend() if engine.is_primary else None
    runtime.run(backend)


if __name__ == "__main__":
    main()
```

启动前设置模型路径：

```bash
export SCAIL2_CHECKPOINT_DIR=/models
export SCAIL2_DIT_CHECKPOINT=/models/derived/SCAIL-2-lightx2v-r128-dpo-alpha1-full-bf16.safetensors

torchrun --standalone --nproc_per_node=2 -m your_service
```

可直接参考：

- `examples/scail2_worker_service.py`：最小 JSONL Backend。
- `examples/scail2_fastapi_service.py`：HTTP、内存队列、状态查询和结果下载。
- `examples/scail2_sdk_debug.py`：内置 014/025/089 的 SDK dry-run 和双卡调试入口。

### 9.1 使用 SDK 调试入口

先用普通 Python 做 dry-run。它会创建并打印 `ProductionProfile`、`EngineConfig`、
处于 CREATED 状态的 `Scail2InferenceEngine` 和 `InferenceJob`，但不会初始化 CUDA
或加载模型：

```bash
python3.10 -m examples.scail2_sdk_debug \
  --dry-run \
  --cases 014 \
  --input-dir ./testdata \
  --output-dir ./debug-outputs
```

真正推理必须使用两个 torchrun rank：

```bash
export SCAIL2_CHECKPOINT_DIR=/models
export SCAIL2_DIT_CHECKPOINT=/models/derived/SCAIL-2-lightx2v-r128-dpo-alpha1-full-bf16.safetensors

torchrun --standalone --nproc_per_node=2 \
  -m examples.scail2_sdk_debug \
  --cases 014,025,089 \
  --input-dir ./testdata \
  --output-dir ./debug-outputs
```

建议依次在 `build_engine_config()`、`Scail2InferenceEngine(config)`、
`DebugJobBackend.acquire()`、`runtime.run()` 和 `engine.infer()` 设置断点。调试事件以
`SCAIL2_DEBUG` 开头，并包含 rank、job、结果或完整失败 traceback。

## 10. Engine 和 Runtime 的生命周期

Engine 状态流转：

```text
CREATED
  -> load()
WARMING_UP
  -> warmup()
READY
  -> infer(job)
BUSY
  -> 成功或可恢复的任务合同错误
READY
  -> close()
CLOSED
```

使用 `Scail2DistributedRuntime` 时，`runtime.run()` 会管理整套生命周期：

```python
runtime.run(backend_if_rank_0_else_none)
```

- 启动时只加载一次模型。
- `warmup()` 当前执行 CUDA 同步和双 rank barrier，不生成额外热身视频。
- `SCAIL2_WORKER_READY` 出现后，模型才可以接任务。
- 一次任务完成后的 `torch.cuda.empty_cache()` 只清理不再引用的临时缓存；
  `engine.pipeline` 仍持有模型参数，因此模型不会从显存释放。
- 只有 Runtime 退出并执行 `engine.close()`，或整个进程/容器退出时，才释放模型。

Runtime 状态接口：

- `runtime.ready`：Engine 为 READY 或 BUSY 时都为 `True`，表示服务可用。
- `runtime.stopped`：任务循环是否已经结束。
- `runtime.failure`：导致 Runtime 退出的异常；正常时为 `None`。
- `wait_until_ready(timeout)`：等待模型完成加载和双 rank 同步。
- `wait_until_stopped(timeout)`：等待 Runtime 退出。

注意 `engine.health()["ready"]` 在 BUSY 时是 `False`，而 `runtime.ready` 在 BUSY 时
仍是 `True`。FastAPI 健康检查会同时返回两者，排障时应结合 `engine.state` 判断。

## 11. 什么时候直接使用 Scail2InferenceEngine

Engine 也支持手工生命周期：

```python
engine = Scail2InferenceEngine(config)
engine.load()
engine.warmup()
try:
    result = engine.infer(job)
finally:
    engine.close()
```

但双卡下两个 rank 必须自行保证以相同顺序调用同一个任务，否则可能卡在 NCCL
集合通信中。因此，手工调用只适合底层调试或已有分布式调度器的高级集成；普通
服务部署应优先使用 `Scail2DistributedRuntime + JobBackend`。

## 12. InferenceResult 怎么用

成功或跳过时，Engine 返回：

```python
def mark_success(self, job: InferenceJob, result: InferenceResult) -> None:
    print(result.status)          # success 或 skipped
    print(result.output_path)     # 最终 MP4 路径
    print(result.frames)
    print(result.fps_fraction)
    print(result.duration)
    print(result.width, result.height)
    print(result.profile)
    print(result.seed)
    print(result.checkpoint)
    payload = result.to_dict()    # 可写入 JSON/数据库
```

失败不会返回 `InferenceResult`，而是由 Runtime 调用 `mark_failed()`。

## 13. FastAPI 示例如何映射 SDK

`examples/scail2_fastapi_service.py` 的主要调用链：

```text
POST /v1/jobs
  -> InferenceRequest（HTTP 五字段）
  -> build_job（生成 job_id/output_path，校验路径）
  -> InferenceJob
  -> FastApiJobBackend.submit
  -> Scail2DistributedRuntime
  -> 两个 rank 的 Scail2InferenceEngine.infer
  -> InferenceResult
  -> FastApiJobBackend.mark_success/mark_failed
```

参考服务提供以下接口：

| 接口 | 用途 | 主要返回码 |
| --- | --- | --- |
| `GET /` | 查看服务名和 wheel 版本 | 200 |
| `GET /v1/health` | 查看 Runtime、Engine 和队列状态 | READY/BUSY 为 200，否则 503 |
| `POST /v1/jobs` | 校验并提交任务 | 202、422、429 或 503 |
| `GET /v1/jobs/{job_id}` | 查询 queued/running/success/skipped/failed | 200 或 404 |
| `GET /v1/jobs/{job_id}/result` | 成功后下载 MP4 | 200、404 或 409 |

`POST /v1/jobs` 中的 `job_id` 由参考服务通过 `uuid.uuid4().hex` 分配，调用方不需要
提供。输出文件固定为 `/outputs/<job_id>.mp4`。同一个业务请求重复提交会生成两个
不同 job_id；如果业务要求幂等，应由接入方在服务层增加幂等键。

参考服务支持多个客户端提交请求，但含义是“并发接收、排队执行”：

- 默认最多保留 8 个等待任务，由 `SCAIL2_MAX_QUEUE_SIZE` 控制；
- 当前正在运行的 1 个任务不计入等待队列容量；
- 两张 GPU 始终共同执行同一个任务，不会同时跑两个 `engine.infer()`；
- 队列已满时新请求返回 HTTP 429，由上游决定重试或选择其他 worker。

HTTP 的五个字段不等于 `InferenceJob` 的全部字段。`job_id`、`output_path`、seed、
overwrite 和 metadata 由服务端控制，这是有意的安全和算法稳定性设计。

该参考服务的任务记录只保存在 rank 0 内存中，服务重启后不会恢复。如果业务要求
可靠任务、重试和历史查询，应替换 `FastApiJobBackend`，而不是修改 Engine。

## 14. 错误处理原则

SDK 还公开以下错误类型：

```python
from scail2_inference import (
    EngineStateError,
    EnvironmentValidationError,
    InputValidationError,
    OutputValidationError,
    Scail2InferenceError,
)
```

- `EnvironmentValidationError`：环境、GPU 拓扑、配置或模型路径不满足合同。
- `EngineStateError`：生命周期顺序错误、重复并发推理、BUSY 时关闭等。
- `InputValidationError`：输入文件、prompt、媒体对齐或音频不合规。
- `OutputValidationError`：生成结果、已有结果、音频封装或原子发布不合规。
- `Scail2InferenceError`：上述 SDK 错误的基类。

接入服务应按错误类型采取不同动作：

| 错误范围 | 当前任务 | worker 后续动作 |
| --- | --- | --- |
| `InputValidationError` | 标记失败，返回明确输入错误 | Engine 恢复 READY，继续下一任务 |
| `OutputValidationError` / `FileExistsError` | 标记失败，保留错误信息 | Engine 恢复 READY，继续下一任务 |
| `EnvironmentValidationError` | 启动失败，不接任务 | 修复环境/模型挂载后重启 |
| CUDA/NCCL/FSDP/模型执行异常 | 标记当前任务失败 | 整个双 rank worker 退出并由 Podman 重启 |
| Backend 回调或外部状态系统异常 | 状态一致性不再可靠 | Runtime 退出并由 Podman 重启 |

输入/输出合同错误发生后，如果所有 rank 仍同步，Engine 会恢复为 READY，Runtime
记录当前任务失败后继续取下一个任务。是否重试该业务任务由外部调度系统决定。

CUDA、NCCL、FSDP 或模型执行异常可能让集合通信处于未知状态。此时 Engine 进入
ERROR，Runtime 停止，正确做法是让 Podman/进程管理器重启整个双 rank worker，
不要在同一进程内强行继续下一任务。

## 15. 接入方通常只需要修改什么

推荐按以下顺序完成接入：

1. 不改代码，先按 Podman 手册运行现成交付镜像和 014 用例，确认机器、GPU、模型
   挂载和输入数据均可用。
2. 使用 `examples/scail2_sdk_debug.py --dry-run` 查看本次 wheel 的 profile、配置和
   `InferenceJob` 序列化内容。
3. 以 `examples/scail2_fastapi_service.py` 或 `scail2_worker_service.py` 为入口，
   替换为自己的 `JobBackend`。
4. 在 Backend 前完成远程输入下载和路径约束，创建本地 `InferenceJob`。
5. 在 `mark_success()` 中记录结果或上传对象存储，在 `mark_failed()` 中保存异常类型
   和错误文本。
6. 保留 `EngineConfig -> Scail2InferenceEngine -> Scail2DistributedRuntime` 的创建
   顺序以及 rank 0/1 的 Backend 边界。
7. 仍使用 `torchrun --nproc_per_node=2` 启动，并由 Podman 管理 worker 重启。

除非 SDK 维护方发布新的 profile 或 wheel，不建议接入方修改 `wan/`、生产 profile、
FSDP 配置和模型加载逻辑。

## 16. 上线前验收清单

交付接入完成后，至少确认以下项目：

- [ ] `python3.10 -c 'import scail2_inference as sdk; print(sdk.__version__)'`
      输出 `0.1.3`；
- [ ] `scail2-runtime-info --expected-gpu-count 2` 返回码为 0；
- [ ] 模型目录按 `model-manifest.yaml` 挂载到容器 `/models`；
- [ ] 日志依次出现 `SCAIL2_WORKER_READY` 和服务层 READY 标记；
- [ ] 空闲时两张 GPU 都保留模型显存，任务之间没有重新加载模型；
- [ ] 014 用例成功，结果帧数、FPS、时长、尺寸和音轨校验通过；
- [ ] 连续提交两个任务时，第二个任务排队而不是与第一个并发推理；
- [ ] 输入路径越界、文件缺失和无音轨等错误能够被记录，worker 仍可继续；
- [ ] 模拟 Runtime fatal 后，HTTP 服务退出且 Podman 能重启整个容器；
- [ ] SIGTERM 停机时不再接新任务，并等待当前任务有序结束；
- [ ] 业务状态系统保存 job_id、status、result 或 error，容器重启后不会丢失关键记录。

完成以上检查后，接入方可以把调度、存储和 API 作为自己的服务代码独立迭代；wheel
升级时只需重新执行版本确认、环境门禁和回归用例，不需要复制或修改 SDK 内部源码。
