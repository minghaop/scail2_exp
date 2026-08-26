# SCAIL2 推理吞吐量实验交接文档

更新时间：2026-08-26

本文档用于在服务器上的 VS Code Remote-SSH/Codex 新会话中恢复本次讨论。开始实验前应复核本文中的推导值、服务器运行环境和线上代码版本。

## 1. 当前目标与硬性条件

- 硬件：单机 8 张 NVIDIA A100 40 GB GPU。
- GPU 拓扑：`nvidia-smi topo -m` 显示任意 GPU 对之间均为 `NV12`，属于全互联 NVLink/NVSwitch 拓扑。
- 服务设计可以调整，worker 结构和并行约束均可重新设计。
- 优化指标：稳定状态下的视频吞吐量，单位建议统一为 `videos/hour` 和 `frames/second/server`。
- 单个视频基线耗时约 8 分钟；一份已保存日志中的 339 帧请求耗时约 8 分钟。
- 初始服务一次仅接收一个活动任务；新方案可以通过多个独立 worker 并发处理多个视频。
- 当前优先研究：单 GPU worker，尽量常驻必要权重，其余权重放在 CPU 内存并按阶段加载。
- 暂缓研究：7 张计算卡共享 1 张权重存储卡的“7+1”设计。

## 2. 需要先阅读的文件

1. `AGENTS.md`（如果服务器仓库中存在）
2. `INFERENCE_SDK_USAGE.zh-CN.md`
3. `scail2_worker_service.py`
4. `scail2_inference/engine.py`
5. `scail2_inference/runtime.py`
6. `scail2_inference/contracts.py`
7. `scail2_inference/profiles/scail2-512p-bf16-v1.json`
8. `scail2_inference/model_configs/config-14b.json`
9. `wan/scail.py`
10. `wan/modules/model_scail2.py`
11. `wan/modules/t5.py`
12. `wan/modules/vae.py`
13. `wan/modules/clip.py`
14. `wan/distributed/fsdp.py`
15. `wan/distributed/sequence_parallel.py`
16. `experiment_logs/startup-2026-08-24.log`
17. `experiment_logs/inference-2026-08-25.log`

服务器安装包中的 `/usr/local/lib/python3.10/dist-packages/wan/...` 可能与本地文件不同。实验前必须用哈希或 `diff` 比较关键文件。

## 3. 当前服务与 rank 的确定方式

当前启动方式：

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 --max-restarts=0 \
  -m scail2_worker_service
```

`torchrun` 创建两个 Python 进程，并为每个进程设置 `RANK`、`LOCAL_RANK` 和 `WORLD_SIZE`。`scail2_inference/engine.py` 直接读取这些环境变量：

```python
self.rank = int(os.getenv("RANK", "0"))
self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
self.world_size = int(os.getenv("WORLD_SIZE", "1"))
```

`LOCAL_RANK=0/1` 对应当前进程使用的本机 GPU。`scail2_worker_service.py` 接受 `--local-rank` 参数是为了兼容部分 torchrun 版本，实际 rank 判断来自上述环境变量。

当前 `EngineConfig` 的关键参数：

```python
expected_world_size=2
t5_fsdp=True
dit_fsdp=True
offload_model=False
```

rank 0 提供 WebSocket 服务并管理任务；rank 1 进入分布式 runtime。两个 rank 都参与 T5 和 DiT 的 FSDP 前向过程。只有 rank 0 执行最终输出管理以及代码中明确受 `self.rank == 0` 约束的 VAE 解码/保存路径。

## 4. 当前 FSDP 行为

`wan/distributed/fsdp.py` 使用：

```python
ShardingStrategy.FULL_SHARD
```

自动包装粒度为 `model.blocks`。对每个被包装的 Transformer block，常规推理过程包含：

1. 各 rank 常驻该 block 的参数分片。
2. block 前向前执行参数 all-gather，使每个 rank 临时获得该 block 的完整参数。
3. 两个 rank 都对各自持有的同一逻辑输入执行该 block 的计算。
4. 前向结束后按 FSDP 的 reshard 策略释放完整参数，仅保留本 rank 分片。
5. 下一个 block 重复上述过程。

因此当前双卡 FSDP 的主要收益是降低单卡常驻权重显存。当前代码没有将一个 block 的矩阵乘法按两张 GPU 做 Tensor Parallel 拆分，也没有将视频 token 按两张 GPU 做 Sequence/Context Parallel 拆分。对单个请求，两个 rank 的大部分 T5/DiT FLOPs 是重复的。

FSDP 参数 gather 的频率取决于 FSDP 包装单元的每次 forward 调用。DiT 有 40 个 block，每个扩散 step 都会依次触发这些 block。生产 profile 设置 6 个 step、`guide_scale=1.0`，代码在该条件下跳过 unconditional DiT 前向，所以每个完整 segment 约触发 `6 × 40 = 240` 次 block 前向/all-gather。5 个 segment 的示例任务约触发 1200 次，末段 token 数较少。

T5 每个视频请求针对 prompt 编码一次，不按帧执行。固定模型、tokenizer、prompt、截断/填充参数以及 eval 模式时，T5 embedding 可缓存并复用。缓存时仍应记录 dtype、shape、tokenizer/model 文件哈希和代码版本。

## 5. 已确认的模型与 profile 参数

来源：`config-14b.json`、`scail2-512p-bf16-v1.json`、模型实现和启动日志。

### DiT/SCAIL2

- `dim = 5120`
- `ffn_dim = 13824`
- `freq_dim = 256`
- `in_dim = 20`
- `mask_dim = 28`
- `out_dim = 16`
- `num_heads = 40`
- `head_dim = 128`
- `num_layers = 40`
- `text_dim = 4096`
- `text_len = 512`
- `patch_size = (1, 2, 2)`
- checkpoint：1307 个 BF16 tensor
- 参数量：16,395,544,384
- checkpoint tensor storage：30.539 GiB

### 生产 profile

- 输出宽度：512
- 输出高度：896
- segment 长度：81 帧
- segment overlap：1 帧
- Euler steps：6
- `guide_scale = 1.0`

## 6. 模型文件大小与权重显存下界

服务器模型文件：

| 组件 | 文件字节数 | 约 GiB |
|---|---:|---:|
| DiT BF16 | 32,791,228,224 | 30.539 |
| UMT5-XXL BF16 | 11,361,920,418 | 10.582 |
| CLIP visual checkpoint | 2,528,485,611 | 2.355 |
| Wan VAE checkpoint | 507,609,880 | 0.473 |
| 合计 | 47,189,244,133 | 43.949 |

文件大小包含少量序列化元数据，实际参数显存应通过运行时逐组件测量。该合计已经超过单张 A100 40 GB 的容量，还没有计入激活、CUDA context、allocator 保留空间、临时 workspace 和输入输出张量。

单 GPU 方案需要阶段化驻留：

- 固定 prompt 时优先离线计算并缓存 T5 embedding，热路径无需保留 T5 权重。
- CLIP visual 仅在参考图变化时执行；得到 `clip_context` 后可以释放或 offload CLIP。
- VAE 负责参考图/pose 编码和分段解码，可以与 DiT 分阶段换入换出，或进一步测量是否可以与 DiT 同时驻留。
- DiT BF16 权重约 30.539 GiB。40 GB 卡只剩约 9 GiB 的理论空间，实际可用于激活的空间更少，单卡能否运行 81 帧 segment 必须用真实模型测量。

## 7. 81 帧 segment 的主要张量规模

对 896×512 输入，VAE 空间下采样后使用 `H=112`、`W=64`；81 个像素帧对应 `T=21` 个 latent 时间位置。DiT patch 为 `(1,2,2)`。

在当前讨论所采用的“1 个 reference latent、无 additional reference”的条件下：

- video token：`21 × 56 × 32 = 37,632`
- reference token：`1 × 56 × 32 = 1,792`
- 主序列 token：`37,632 + 1,792 = 39,424`
- pose token：`21 × 28 × 16 = 9,408`
- attention 总 token：`39,424 + 9,408 = 48,832`

主要中间形状：

- hidden state：`[1, 48,832, 5,120]`
- Q/K/V 单个张量：`[1, 48,832, 40, 128]`
- FFN 中间张量：`[1, 48,832, 13,824]`

BF16 单张量理论大小：

- hidden：`48,832 × 5,120 × 2 bytes ≈ 0.466 GiB`
- 单个 Q/K/V：同样约 `0.466 GiB`
- FFN 中间：`48,832 × 13,824 × 2 bytes ≈ 1.257 GiB`

这些数值只表示单个逻辑张量，不等于峰值激活显存。峰值还取决于 attention kernel、临时 buffer、张量生命周期、是否原地操作、PyTorch allocator、VAE/CLIP 是否同时驻留。服务端实验需要用 `torch.cuda.max_memory_allocated()` 和 `max_memory_reserved()` 测量。

## 8. 已保存推理日志的结论

日志：`experiment_logs/inference-2026-08-25.log`

- 输入视频共 339 帧。
- 分为 5 个 segment：`[0,81)`、`[80,161)`、`[160,241)`、`[240,321)`、`[320,339)`。
- 前 4 个完整 segment 每个约 103–104 秒完成 6 个采样 step，即约 17.2–17.4 秒/step。
- 最后一个 padded length 为 21 的 segment，6 steps 约 21 秒，即约 3.51 秒/step。
- 完整 segment 的 DiT/通信计算占据绝大部分运行时间。
- rank 0 与 rank 1 的进度基本同步，符合两个 rank 共同参与同一个 FSDP 请求的行为。
- 日志中同一消息分别来自 `MainThread` 和 `scail2-runtime`，对应两个 rank 的独立进程输出合并到同一容器日志。

## 9. 已保存启动日志的结论

日志：`experiment_logs/startup-2026-08-24.log`

- 两个 rank 的 engine load 均约 362.35 秒。
- process group 初始化约 4.4 秒。
- 从 pipeline load 开始到 T5 checkpoint load 日志出现，约 61 秒。
- T5 checkpoint load/FSDP 和随后组件加载约 10 秒量级。
- VAE/CLIP 的可见加载阶段约 2 秒。
- 从打印 `Creating WanSCAILModel` 到 `dit_checkpoint_copy start` 约 250 秒，是启动时间的最大区段；该区段包含构造完整 DiT、参数初始化以及 BF16 cast。
- 两个 rank 都构造完整 CPU 模型并读取完整 DiT checkpoint。
- DiT checkpoint copy：rank 0 为 90.937 秒，rank 1 为 63.280 秒。
- rank 1 在 pre-FSDP barrier 等待 12.371 秒。
- DiT FSDP wrap 约 10.4 秒。

`SCAIL2Model.__init__` 创建全部模块，`init_weights()` 对线性层做随机初始化；随后 `wan/scail.py` 将模型转为目标 dtype，再加载 checkpoint。checkpoint 会覆盖初始随机权重，因此这部分初始化是潜在的启动优化点。

候选启动优化：meta-device 初始化、`load_state_dict(assign=True)` 或等价的空权重加载机制；需要处理 `self.freqs` 这类当前未注册为 parameter/buffer 的张量，并验证 FSDP `sync_module_states`、dtype、输出一致性和峰值 CPU 内存。

## 10. 当前方案假设与待验证问题

### 单 GPU worker 假设

理想目标是在 8 张卡上运行 8 个彼此独立的单 GPU worker，每个 worker 同时处理一个视频。若单卡 81 帧 segment 无法容纳，应按以下顺序测试：

1. T5 embedding 缓存，移除热路径 T5 权重。
2. CLIP 与 VAE 分阶段 offload，DiT 尽量常驻。
3. 降低 segment_len，并测量吞吐量变化；更短 segment 会增加 overlap/编解码/调度开销。
4. 精确控制 CPU pinned memory 和异步 H2D copy，评估传输与计算重叠。
5. 若 DiT 权重加激活仍超出 40 GB，再评估每任务 2 GPU 的 Sequence/Context Parallel 或 Tensor Parallel。

### 关键未知量

- 单卡仅驻留 DiT 时，81 帧 segment 的真实峰值 allocated/reserved 显存。
- attention 实际使用的 kernel，以及是否物化 `L×L` attention matrix。
- 当前 FSDP 每 block 的 all-gather 字节数、时间占比和 NCCL 带宽。
- VAE、CLIP、T5 各阶段的峰值显存、执行时间和 H2D/D2H 传输量。
- CPU 内存容量、NUMA 放置、GPU Direct P2P 和 CPU→GPU 实测带宽。
- 8 个 worker 同时读取 checkpoint 时的存储和 CPU 内存压力。
- 固定 prompt 是否覆盖全部请求；参考图通常变化，因此 CLIP 输出需按参考图缓存键管理。

## 11. 服务器实验顺序

所有性能实验先使用独立目录和非生产端口。先保存基线，随后每次只改变一个因素。

### 阶段 A：环境与版本复核

```bash
hostname
nvidia-smi -L
nvidia-smi topo -m
python -V
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())"
sha256sum scail2_worker_service.py wan/scail.py wan/modules/model_scail2.py wan/distributed/fsdp.py
```

记录容器镜像 ID、启动命令、环境变量、模型文件哈希和 CPU/NUMA 信息。

### 阶段 B：双卡 FSDP 基线

- 单任务冷启动时间和稳定启动时间。
- 每个 segment、每个 step、T5、CLIP、VAE encode/decode、保存/复用阶段耗时。
- 每张卡的 GPU utilization、显存 allocated/reserved、功耗和 NVLink 传输。
- 4 组双卡 worker 并发时的服务器吞吐量。

### 阶段 C：单卡显存可行性

先禁用 FSDP，仅使用一张空闲 GPU，并按组件逐个构造/加载。每阶段调用：

```python
torch.cuda.reset_peak_memory_stats()
# run stage
torch.cuda.synchronize()
print(torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
print(torch.cuda.memory_reserved(), torch.cuda.max_memory_reserved())
```

先测只加载 DiT，再测最小 81 帧 forward。发生 OOM 时保存完整错误、峰值统计和输入 shape，避免直接开始长视频生成。

### 阶段 D：吞吐量比较

至少比较：

- 4×双卡 FSDP worker。
- 8×单卡 worker（若可运行）。
- 8×单卡、较短 segment（若 81 帧 OOM）。
- 经验证可运行的其他并行配置。

每个配置包含预热任务和多个正式任务；报告平均值、P50、P95、失败率、每小时视频数和每小时帧数。

## 12. 对远端 Codex 新会话的首条指令

```text
请完整阅读 AGENTS.md（若存在）、SCAIL2_EXPERIMENT_CONTEXT.md、
INFERENCE_SDK_USAGE.zh-CN.md、scail2_worker_service.py、
scail2_inference/engine.py、wan/scail.py、wan/modules/model_scail2.py、
wan/distributed/fsdp.py，以及 experiment_logs 下的两份日志。

目标是在 8×A100 40GB 单机上最大化视频推理 throughput。
先执行只读环境检查，比较服务器安装代码与当前目录代码，复核文档中的
事实、推导和假设。不要立即运行完整 8 分钟任务，也不要修改生产服务。
随后给出第一轮低风险显存测量实验及命令，等待我确认后执行。
```

## 13. 实验安全约束

- 不修改或重启生产容器。
- 不占用已有任务使用的 GPU；先用 `nvidia-smi` 确认空闲卡。
- 使用独立端口、独立 `/dev/shm` 子目录和独立输出目录。
- 首轮使用短输入或单 segment，避免直接运行完整视频。
- 修改前备份或建立新的实验副本；当前本地目录没有 `.git`，服务器侧建议先初始化实验仓库或复制到版本化目录。
- 每次记录精确命令、代码 diff、环境变量、GPU 绑定、开始/结束时间和结果。
